# CUDA Matrix Multiplication Optimization

Step-by-step optimization of single-precision matrix multiplication (SGEMM, `C = A × B`) in CUDA C++, benchmarked against a CPU baseline and cuBLAS on an NVIDIA RTX 3050 Laptop GPU (Ampere, compute capability 8.6).

| Version | File / kernel | Idea |
|---|---|---|
| v0 | `matmul_cpu` | Single-thread CPU, cache-friendly i-k-j loop order |
| v1 | `matmul_naive` | One thread per output element, uncoalesced thread-to-data mapping |
| v2 | `matmul_coalesced` | Same math, `threadIdx.x` mapped to the column so warps read contiguous memory |
| v3 | `matmul_tiled<TILE>` | Shared-memory tiling (8/16/32) for data reuse |
| ref | `cublasSgemm` | Vendor-tuned upper bound (pure FP32, TF32 disabled) |

## Build

**Linux / WSL**
```bash
make                      # nvcc -O3 -arch=sm_86 -lineinfo -o matmul src/matmul.cu -lcublas
```

**Windows (Developer Command Prompt for VS, CUDA Toolkit installed)**
```bat
nvcc -O3 -arch=sm_86 -lineinfo -o matmul.exe src\matmul.cu -lcublas
```

Check your GPU's compute capability with `nvidia-smi --query-gpu=name,compute_cap --format=csv` and change `sm_86` if needed.

## Run

```bash
./matmul --n 1024 --tile 32 --reps 10     # one size, prints a table
pip install numpy matplotlib
python bench.py                            # verify + full sweep + plots
```

`bench.py`:
1. **Verifies** every kernel at N = 127, 256, 1000 (non-multiples of the tile size test the boundary code) against the CPU output and an independent NumPy float64 reference. Tolerance is 1e-3 relative error, since float addition order differs between kernels.
2. **Benchmarks** N = 256…4096 for all kernels plus a tile-size sweep (8/16/32) for the tiled kernel, taking the median of 3 runs of 10 launches each.
3. Writes `results/results.csv`, `results/gflops_vs_n.png` and `results/tile_sweep.png`.

Timing uses CUDA events around the kernel only (after one warm-up launch), so host-device copies are excluded. GFLOP/s = 2N³ / time.

## Results

> Fill this table in from **your own** `bench.py` output. Don't put numbers here or on the resume that you did not measure.

| Kernel | N=1024 ms | N=1024 GFLOP/s | N=4096 GFLOP/s | % of cuBLAS (N=4096) |
|---|---|---|---|---|
| CPU | | | — | — |
| naive | | | | |
| coalesced | | | | |
| tiled (best tile) | | | | |
| cuBLAS | | | | 100% |

## Why each step is faster

### v1 → v2: coalescing
A warp is 32 threads with consecutive `threadIdx.x`. In v1, `threadIdx.x` picks the **row**, so in each loop iteration the 32 threads read `A[row*N + k]` from 32 different rows, which are N×4 bytes apart. That is 32 separate 32-byte memory sectors for 128 useful bytes, and the store to `C` is also strided.

In v2, `threadIdx.x` picks the **column**: `B[k*N + col]` becomes 32 consecutive floats (one 128-byte request, 4 sectors), `A[row*N + k]` is the same address for the whole warp (a broadcast), and stores to `C` are contiguous. The arithmetic is identical; only the mapping changed.

### v2 → v3: shared-memory tiling
v2 still does 2 global loads per multiply-add, so arithmetic intensity is about 0.25 FLOP/byte and the kernel is memory-bound. In v3 each block loads a TILE×TILE tile of A and of B into shared memory once (each thread loads one element of each, coalesced), synchronizes, and then every element is reused TILE times from on-chip memory. Global traffic drops by roughly a factor of TILE.

- Two `__syncthreads()` per tile: the first so nobody reads a tile before it is fully loaded, the second so nobody overwrites it while others are still reading.
- No bank conflicts: in the inner loop `As[ty][k]` is a broadcast within a warp and `Bs[k][tx]` is consecutive words.
- Out-of-range elements are loaded as 0, so any N works.

### Tile size trade-off
Bigger tiles mean more reuse but more threads and shared memory per block, which can reduce how many blocks fit on an SM (occupancy). TILE=8 gives 64-thread blocks (only 2 warps each), which is poor for latency hiding. Compare 16 vs 32 in your own sweep and explain the result with occupancy and reuse.

### Why cuBLAS is still much faster
cuBLAS uses **register tiling** (each thread computes a 4×4 or 8×8 patch of C, so values are reused from registers, not just shared memory), vectorized `float4` loads, double-buffered shared memory so loads for the next tile overlap with compute on the current one, and hand-tuned kernels picked per GPU and per matrix shape. Tensor cores are not used here because TF32 is disabled.

## Profiling (good interview material)

```bash
ncu --set full -k regex:matmul -o profile ./matmul --n 2048 --reps 1
```
Compare across kernels in Nsight Compute:
- `l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio` (sectors per load request) — much higher for the naive kernel (its A loads need 32 sectors per warp request) than for coalesced (4 sectors for a contiguous 128-byte read): direct proof of coalescing. Nsight Compute also flags uncoalesced accesses in its "Source Counters" section.
- DRAM throughput and the **Memory Workload Analysis** section: tiled should show far less DRAM traffic.
- **Occupancy** section for the tile-size explanation.
- `-Xptxas -v` at compile time shows registers and shared memory per kernel (all kernels use about 37–40 registers, no spills).

## Possible next steps
1. 2D register tiling (each thread computes an 8×8 block of C)
2. `float4` vectorized loads
3. Double buffering with `cp.async` (Ampere)
4. Transposing the A tile in shared memory to enable vectorized reads
5. TF32 tensor cores with WMMA, compared against cuBLAS with TF32 enabled
