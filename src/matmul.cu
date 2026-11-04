// matmul.cu
// CUDA SGEMM optimization study: C = A * B  (N x N, float32, row-major)
//
//   v0  CPU baseline          (i-k-j loop order, single thread)
//   v1  naive GPU kernel      (one thread per C element, UNcoalesced mapping)
//   v2  coalesced GPU kernel  (same math, threadIdx.x -> column => coalesced)
//   v3  shared-memory tiled   (TILE x TILE tiles staged in shared memory)
//   ref cuBLAS SGEMM          (vendor-tuned upper bound)
//
// Build (RTX 3050 = Ampere, sm_86):
//   nvcc -O3 -arch=sm_86 -o matmul src/matmul.cu -lcublas
//
// Usage:
//   ./matmul [--n N] [--tile 8|16|32] [--reps R] [--cpu-max N] [--tiled-only] [--dump DIR]
//
// The last line printed starts with "CSV," so bench.py can parse it.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <random>
#include <chrono>
#include <algorithm>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err_ = (call);                                              \
        if (err_ != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error '%s' at %s:%d\n",                       \
                    cudaGetErrorString(err_), __FILE__, __LINE__);              \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

#define CUBLAS_CHECK(call)                                                      \
    do {                                                                        \
        cublasStatus_t st_ = (call);                                            \
        if (st_ != CUBLAS_STATUS_SUCCESS) {                                     \
            fprintf(stderr, "cuBLAS error %d at %s:%d\n", (int)st_,             \
                    __FILE__, __LINE__);                                        \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

// ---------------------------------------------------------------------------
// v1: naive kernel -- UNCOALESCED on purpose.
// threadIdx.x selects the ROW. A warp is 32 threads with consecutive
// threadIdx.x, so the warp touches 32 different rows:
//   A[row*N + k]  -> 32 addresses N floats apart  (32 separate memory sectors)
//   C[row*N + col]-> 32 writes N floats apart     (uncoalesced store)
//   B[k*N + col]  -> same address for all 32      (broadcast)
// ---------------------------------------------------------------------------
__global__ void matmul_naive(const float* __restrict__ A,
                             const float* __restrict__ B,
                             float* __restrict__ C, int N)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < N && col < N) {
        float acc = 0.0f;
        for (int k = 0; k < N; ++k)
            acc += A[row * N + k] * B[k * N + col];
        C[row * N + col] = acc;
    }
}

// ---------------------------------------------------------------------------
// v2: coalesced kernel -- identical math, swapped thread-to-data mapping.
// threadIdx.x now selects the COLUMN, so within a warp:
//   B[k*N + col]  -> 32 consecutive floats = 128 B (fully coalesced)
//   C[row*N + col]-> 32 consecutive floats       (coalesced store)
//   A[row*N + k]  -> same address for all 32     (broadcast)
// ---------------------------------------------------------------------------
__global__ void matmul_coalesced(const float* __restrict__ A,
                                 const float* __restrict__ B,
                                 float* __restrict__ C, int N)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < N && col < N) {
        float acc = 0.0f;
        for (int k = 0; k < N; ++k)
            acc += A[row * N + k] * B[k * N + col];
        C[row * N + col] = acc;
    }
}

// ---------------------------------------------------------------------------
// v3: shared-memory tiled kernel.
// Each block computes a TILE x TILE tile of C. It walks along K in steps of
// TILE: every thread loads ONE element of A's tile and ONE of B's tile into
// shared memory (coalesced loads), then the block reuses those values TILE
// times from on-chip shared memory. Global loads drop by a factor of TILE.
//
// Shared-memory bank behaviour in the inner loop:
//   As[ty][k] -> same address across a warp row      (broadcast, no conflict)
//   Bs[k][tx] -> consecutive addresses across a warp (conflict-free)
// Out-of-range elements are padded with 0 so any N works, not just multiples.
// ---------------------------------------------------------------------------
template <int TILE>
__global__ void matmul_tiled(const float* __restrict__ A,
                             const float* __restrict__ B,
                             float* __restrict__ C, int N)
{
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    const int tx  = threadIdx.x;
    const int ty  = threadIdx.y;
    const int row = blockIdx.y * TILE + ty;
    const int col = blockIdx.x * TILE + tx;

    float acc = 0.0f;
    const int numTiles = (N + TILE - 1) / TILE;

    for (int t = 0; t < numTiles; ++t) {
        const int aCol = t * TILE + tx;
        const int bRow = t * TILE + ty;
        As[ty][tx] = (row < N && aCol < N) ? A[row * N + aCol] : 0.0f;
        Bs[ty][tx] = (bRow < N && col < N) ? B[bRow * N + col] : 0.0f;
        __syncthreads();            // tile fully loaded before anyone reads it

#pragma unroll
        for (int k = 0; k < TILE; ++k)
            acc += As[ty][k] * Bs[k][tx];
        __syncthreads();            // everyone done reading before next overwrite
    }

    if (row < N && col < N)
        C[row * N + col] = acc;
}

// v0: CPU baseline. i-k-j order keeps the innermost loop streaming through
// contiguous rows of B and C (cache friendly), unlike textbook i-j-k.

static void matmul_cpu(const float* A, const float* B, float* C, int N)
{
    std::fill(C, C + (size_t)N * N, 0.0f);
    for (int i = 0; i < N; ++i) {
        float* c = C + (size_t)i * N;
        for (int k = 0; k < N; ++k) {
            const float a = A[(size_t)i * N + k];
            const float* b = B + (size_t)k * N;
            for (int j = 0; j < N; ++j)
                c[j] += a * b[j];
        }
    }
}


// Timing helpers

template <typename Launch>
static float time_gpu_ms(Launch launch, int reps)
{
    launch();                                   // warm-up (JIT, clocks, caches)
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for (int r = 0; r < reps; ++r) launch();
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaGetLastError());

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return ms / reps;
}

static double gflops(int N, double ms)
{
    if (ms <= 0) return 0.0;
    return 2.0 * (double)N * N * N / (ms * 1e6);
}

static double max_rel_err(const std::vector<float>& x, const std::vector<float>& ref)
{
    double maxDiff = 0.0, maxRef = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
        maxDiff = std::max(maxDiff, (double)std::fabs(x[i] - ref[i]));
        maxRef  = std::max(maxRef,  (double)std::fabs(ref[i]));
    }
    return maxRef > 0 ? maxDiff / maxRef : maxDiff;
}

static void dump(const std::string& dir, const char* name, const std::vector<float>& v)
{
    std::string path = dir + "/" + name;
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) { fprintf(stderr, "cannot write %s\n", path.c_str()); exit(EXIT_FAILURE); }
    fwrite(v.data(), sizeof(float), v.size(), f);
    fclose(f);
}

static void usage(const char* prog)
{
    printf("usage: %s [--n N] [--tile 8|16|32] [--reps R] [--cpu-max N] "
           "[--tiled-only] [--dump DIR]\n", prog);
}

int main(int argc, char** argv)
{
    int N = 1024, tile = 32, reps = 10, cpuMax = 2048;
    bool tiledOnly = false;
    std::string dumpDir;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> const char* {
            if (i + 1 >= argc) { usage(argv[0]); exit(EXIT_FAILURE); }
            return argv[++i];
        };
        if      (a == "--n")          N = atoi(next());
        else if (a == "--tile")       tile = atoi(next());
        else if (a == "--reps")       reps = atoi(next());
        else if (a == "--cpu-max")    cpuMax = atoi(next());
        else if (a == "--tiled-only") tiledOnly = true;
        else if (a == "--dump")       dumpDir = next();
        else { usage(argv[0]); return a == "--help" || a == "-h" ? 0 : 1; }
    }
    if (tile != 8 && tile != 16 && tile != 32) {
        fprintf(stderr, "--tile must be 8, 16 or 32\n");
        return 1;
    }
    if (N <= 0 || reps <= 0) { usage(argv[0]); return 1; }

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s | SMs: %d | CC %d.%d | N=%d tile=%d reps=%d\n",
           prop.name, prop.multiProcessorCount, prop.major, prop.minor, N, tile, reps);

    const size_t elems = (size_t)N * N;
    const size_t bytes = elems * sizeof(float);

    // Deterministic inputs in [-1, 1]
    std::vector<float> hA(elems), hB(elems);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& x : hA) x = dist(rng);
    for (auto& x : hB) x = dist(rng);

    float *dA, *dB, *dC;
    CUDA_CHECK(cudaMalloc(&dA, bytes));
    CUDA_CHECK(cudaMalloc(&dB, bytes));
    CUDA_CHECK(cudaMalloc(&dC, bytes));
    CUDA_CHECK(cudaMemcpy(dA, hA.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB.data(), bytes, cudaMemcpyHostToDevice));

    auto fetch = [&](std::vector<float>& out) {
        out.resize(elems);
        CUDA_CHECK(cudaMemcpy(out.data(), dC, bytes, cudaMemcpyDeviceToHost));
    };

    double cpuMs = -1, naiveMs = -1, coalMs = -1, tiledMs = -1, cublasMs = -1;
    std::vector<float> cCpu, cNaive, cCoal, cTiled, cCublas;

    // v0 CPU 
    if (!tiledOnly && N <= cpuMax) {
        cCpu.resize(elems);
        auto t0 = std::chrono::high_resolution_clock::now();
        matmul_cpu(hA.data(), hB.data(), cCpu.data(), N);
        auto t1 = std::chrono::high_resolution_clock::now();
        cpuMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
    }

    const dim3 block32(32, 32);
    const dim3 grid32((N + 31) / 32, (N + 31) / 32);

    // v1 naive 
    if (!tiledOnly) {
        naiveMs = time_gpu_ms([&] {
            matmul_naive<<<grid32, block32>>>(dA, dB, dC, N);
        }, reps);
        fetch(cNaive);
    }

    // v2 coalesced 
    if (!tiledOnly) {
        coalMs = time_gpu_ms([&] {
            matmul_coalesced<<<grid32, block32>>>(dA, dB, dC, N);
        }, reps);
        fetch(cCoal);
    }

    // v3 tiled 
    {
        const dim3 block(tile, tile);
        const dim3 grid((N + tile - 1) / tile, (N + tile - 1) / tile);
        tiledMs = time_gpu_ms([&] {
            switch (tile) {
                case 8:  matmul_tiled<8> <<<grid, block>>>(dA, dB, dC, N); break;
                case 16: matmul_tiled<16><<<grid, block>>>(dA, dB, dC, N); break;
                default: matmul_tiled<32><<<grid, block>>>(dA, dB, dC, N); break;
            }
        }, reps);
        fetch(cTiled);
    }

    // cuBLAS 
    // cuBLAS is column-major. A row-major matrix X in memory IS X^T in
    // column-major. We want row-major C = A*B, i.e. column-major C^T = B^T A^T,
    // so we pass B first, then A.
    if (!tiledOnly) {
        cublasHandle_t handle;
        CUBLAS_CHECK(cublasCreate(&handle));
        CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH)); // pure FP32, no TF32
        const float alpha = 1.0f, beta = 0.0f;
        cublasMs = time_gpu_ms([&] {
            CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                     N, N, N, &alpha, dB, N, dA, N, &beta, dC, N));
        }, reps);
        fetch(cCublas);
        CUBLAS_CHECK(cublasDestroy(handle));
    }

    // quick self-check (bench.py does the full verification) 
    const std::vector<float>* ref = !cCpu.empty() ? &cCpu
                                  : !cCublas.empty() ? &cCublas : nullptr;
    const char* refName = !cCpu.empty() ? "CPU" : "cuBLAS";

    printf("\n%-12s %12s %12s %12s %14s\n", "kernel", "time (ms)", "GFLOP/s",
           "vs CPU", ref ? "max rel err" : "");
    auto row = [&](const char* name, double ms, const std::vector<float>& out) {
        if (ms < 0) return;
        char speed[32] = "-";
        if (cpuMs > 0) snprintf(speed, sizeof(speed), "%.1fx", cpuMs / ms);
        char err[32] = "";
        if (ref && !out.empty() && &out != ref)
            snprintf(err, sizeof(err), "%.2e", max_rel_err(out, *ref));
        printf("%-12s %12.3f %12.1f %12s %14s\n", name, ms, gflops(N, ms), speed, err);
    };
    row("cpu",        cpuMs,    cCpu);
    row("naive",      naiveMs,  cNaive);
    row("coalesced",  coalMs,   cCoal);
    row("tiled",      tiledMs,  cTiled);
    row("cublas",     cublasMs, cCublas);
    if (ref) printf("(errors relative to %s result)\n", refName);

    if (!dumpDir.empty()) {
        dump(dumpDir, "A.bin", hA);
        dump(dumpDir, "B.bin", hB);
        if (!cCpu.empty())    dump(dumpDir, "C_cpu.bin", cCpu);
        if (!cNaive.empty())  dump(dumpDir, "C_naive.bin", cNaive);
        if (!cCoal.empty())   dump(dumpDir, "C_coalesced.bin", cCoal);
        if (!cTiled.empty())  dump(dumpDir, "C_tiled.bin", cTiled);
        if (!cCublas.empty()) dump(dumpDir, "C_cublas.bin", cCublas);
    }

    // machine-readable line for bench.py
    printf("CSV,%d,%d,%.4f,%.4f,%.4f,%.4f,%.4f\n",
           N, tile, cpuMs, naiveMs, coalMs, tiledMs, cublasMs);

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
    return 0;
}
