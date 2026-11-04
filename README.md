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

Measured on an NVIDIA GeForce RTX 3050 6GB Laptop GPU (20 SMs, CC 8.6), plugged in. Median of 3 runs × 10 launches, from `python bench.py`. All kernels passed verification at N = 127, 256 and 1000 against the CPU result and a NumPy float64 reference (max relative error ≤ 1.3e-6).

| Kernel | N=1024 ms | N=1024 GFLOP/s | vs naive (1024) | N=4096 GFLOP/s | % of cuBLAS (4096) |
|---|---|---|---|---|---|
| CPU (1 thread) | 124.76 | 17.2 | — | skipped | — |
| naive | 33.03 | 65.0 | 1.0× | 65.1 | 1.8% |
| coalesced | 4.48 | 479.4 | 7.4× | 397.8 | 10.8% |
| tiled, TILE=16 | 3.18 | **675.7** | **10.4×** | 488.4 | 13.3% |
| tiled, TILE=32 | 3.80 | 564.8 | 8.7× | **530.3** | **14.4%** |
| cuBLAS | 0.44 | 4929.9 | 75.8× | 3678.2 | 100% |

Highest speedup over the single-threaded CPU: **65×** (tiled, TILE=32, N=2048). Full data in `results/results.csv`; plots in `results/gflops_vs_n.png` and `results/tile_sweep.png`.

N=4096 numbers (including cuBLAS) are lower than at 1024–2048 because the laptop GPU throttles under sustained load; compare kernels within the same N, not across sizes.

**Tile-size sweep (GFLOP/s):**

| N | TILE=8 | TILE=16 | TILE=32 |
|---|---|---|---|
| 256 | 407 | **524** | 440 |
| 1024 | 479 | **676** | 565 |
| 2048 | 350 | 600 | **610** |
| 4096 | 270 | 488 | **530** |

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
The best tile size changes with N, because two effects pull in opposite directions:

- **Occupancy favours TILE=16.** An SM on this GPU holds at most 1536 threads. A 32×32 block is 1024 threads, so only 1 block fits (32 of 48 warps, 67% occupancy). 16×16 blocks are 256 threads, so up to 6 fit (48 warps, 100%), giving the scheduler more warps to hide latency with. This wins at small and medium N (TILE=16 is 20% faster at N=1024).
- **Data reuse favours TILE=32.** Each element loaded from global memory is reused TILE times, so TILE=32 moves half the global/L2 traffic of TILE=16. At N=2048 the two are tied, and at N=4096, where the working set is far larger than the 2 MB L2, the lower traffic wins (TILE=32 is 9% faster).
- **TILE=8 loses at every size from N=1024 up, and is even slower than the untiled coalesced kernel.** 64-thread blocks run into the per-SM block limit, reuse is only 8×, and there are 4× more `__syncthreads()` per unit of work.

### Why tiling gives only about 1.3× over coalesced
Shared-memory tiling cuts DRAM traffic by about TILE×, but the speedup over the coalesced kernel is only 1.2–1.4× at every size. Two reasons:
1. **Hardware caches already capture much of the reuse.** In the coalesced kernel all threads of a warp read the same `A` element (a broadcast), and neighbouring blocks read the same rows of `B`, so L1/L2 serve much of that traffic.
2. **The tiled kernel is now bound by shared-memory loads.** Its inner loop issues 2 shared-memory loads per fused multiply-add, so load instructions, not DRAM bandwidth, limit throughput. The next step is register tiling (below).

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
