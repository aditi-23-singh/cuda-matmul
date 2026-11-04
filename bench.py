#!/usr/bin/env python3
"""
bench.py -- verify and benchmark the CUDA matmul kernels.

  1. Verification: for small sizes, runs the binary with --dump, loads every
     output matrix, and checks it against the CPU result (and an independent
     NumPy float64 reference).
  2. Benchmark: sweeps matrix sizes (all kernels) and tile sizes (tiled
     kernel), averages several runs, writes results.csv and plots.

Usage:
  python bench.py                       # default sweep
  python bench.py --sizes 512 1024 2048 --tiles 8 16 32 --reps 20
"""
import argparse
import csv
import os
import statistics
import subprocess
import sys
import tempfile

import numpy as np

KERNELS = ["cpu", "naive", "coalesced", "tiled", "cublas"]
REL_TOL = 1e-3  # FP32 accumulation order differs between kernels


def default_bin():
    name = "matmul.exe" if os.name == "nt" else "matmul"
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), name)


def run(binary, n, tile, reps, tiled_only=False, dump_dir=None, cpu_max=2048):
    cmd = [binary, "--n", str(n), "--tile", str(tile), "--reps", str(reps),
           "--cpu-max", str(cpu_max)]
    if tiled_only:
        cmd.append("--tiled-only")
    if dump_dir:
        cmd += ["--dump", dump_dir]
    out = subprocess.run(cmd, capture_output=True, text=True)
    if out.returncode != 0:
        sys.exit(f"binary failed: {' '.join(cmd)}\n{out.stdout}\n{out.stderr}")
    line = next(l for l in out.stdout.splitlines() if l.startswith("CSV,"))
    f = line.split(",")[1:]
    vals = [float(x) for x in f[2:]]
    return {k: (v if v >= 0 else None) for k, v in zip(KERNELS, vals)}


def gflops(n, ms):
    return 2.0 * n ** 3 / (ms * 1e6) if ms else None


def rel_err(x, ref):
    return float(np.max(np.abs(x.astype(np.float64) - ref)) / np.max(np.abs(ref)))


def verify(binary, sizes, tiles):
    print("=== Verification ===")
    all_ok = True
    for n in sizes:
        for tile in tiles:
            with tempfile.TemporaryDirectory() as d:
                run(binary, n, tile, reps=1, dump_dir=d)
                load = lambda name: np.fromfile(os.path.join(d, name),
                                                dtype=np.float32).reshape(n, n)
                A, B = load("A.bin"), load("B.bin")
                ref64 = A.astype(np.float64) @ B.astype(np.float64)
                c_cpu = load("C_cpu.bin") if os.path.exists(os.path.join(d, "C_cpu.bin")) else None

                for k in ["cpu", "naive", "coalesced", "tiled", "cublas"]:
                    path = os.path.join(d, f"C_{k}.bin")
                    if not os.path.exists(path):
                        continue
                    C = load(f"C_{k}.bin")
                    e64 = rel_err(C, ref64)
                    ecpu = rel_err(C, c_cpu.astype(np.float64)) if (c_cpu is not None and k != "cpu") else 0.0
                    ok = e64 < REL_TOL and ecpu < REL_TOL
                    all_ok &= ok
                    print(f"N={n:5d} tile={tile:2d} {k:10s} "
                          f"err_vs_cpu={ecpu:.2e} err_vs_numpy64={e64:.2e} "
                          f"{'PASS' if ok else 'FAIL'}")
    print("ALL PASS\n" if all_ok else "SOME CHECKS FAILED\n")
    return all_ok


def bench(binary, sizes, tiles, reps, trials, cpu_max):
    print("=== Benchmark ===")
    rows = []
    for n in sizes:
        # Full run (all kernels) at the default tile 32, repeated `trials` times
        samples = [run(binary, n, 32, reps, cpu_max=cpu_max) for _ in range(trials)]
        med = {k: (statistics.median([s[k] for s in samples]) if samples[0][k] else None)
               for k in KERNELS}
        for k in KERNELS:
            if med[k]:
                rows.append({"N": n, "kernel": k, "tile": 32 if k == "tiled" else "",
                             "ms": med[k], "gflops": gflops(n, med[k])})
        line = "  ".join(f"{k}={med[k]:.3f}ms" for k in KERNELS if med[k])
        print(f"N={n:5d}  {line}")

        # Tile sweep for the tiled kernel only
        for tile in tiles:
            if tile == 32:
                continue
            ms = statistics.median(run(binary, n, tile, reps, tiled_only=True)["tiled"]
                                   for _ in range(trials))
            rows.append({"N": n, "kernel": "tiled", "tile": tile,
                         "ms": ms, "gflops": gflops(n, ms)})
            print(f"         tiled tile={tile:2d} {ms:.3f} ms")
    return rows


def summarize(rows):
    print("\n=== Summary (GFLOP/s, % of cuBLAS) ===")
    by_n = {}
    for r in rows:
        by_n.setdefault(r["N"], []).append(r)
    for n, rs in by_n.items():
        cub = next((r["gflops"] for r in rs if r["kernel"] == "cublas"), None)
        cpu = next((r["ms"] for r in rs if r["kernel"] == "cpu"), None)
        print(f"N={n}")
        for r in rs:
            label = r["kernel"] + (f"(tile={r['tile']})" if r["kernel"] == "tiled" else "")
            pct = f"{100 * r['gflops'] / cub:5.1f}% of cuBLAS" if cub else ""
            spd = f"{cpu / r['ms']:8.1f}x vs CPU" if cpu else ""
            print(f"   {label:16s} {r['gflops']:9.1f} GFLOP/s  {pct}  {spd}")


def plot(rows, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed -- skipping plots")
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for k in KERNELS:
        pts = sorted((r["N"], r["gflops"]) for r in rows
                     if r["kernel"] == k and (k != "tiled" or r["tile"] == 32))
        if pts:
            ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o",
                    label=k if k != "tiled" else "tiled (32)")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Matrix size N")
    ax.set_ylabel("GFLOP/s (log)")
    ax.set_title("SGEMM throughput by kernel")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "gflops_vs_n.png"), dpi=150)

    tiled = [r for r in rows if r["kernel"] == "tiled"]
    if tiled:
        n_max = max(r["N"] for r in tiled)
        pts = sorted((int(r["tile"]), r["gflops"]) for r in tiled if r["N"] == n_max)
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.bar([str(p[0]) for p in pts], [p[1] for p in pts])
        ax.set_xlabel("Tile size")
        ax.set_ylabel("GFLOP/s")
        ax.set_title(f"Tiled kernel vs tile size (N={n_max})")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "tile_sweep.png"), dpi=150)
    print(f"plots written to {out_dir}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bin", default=default_bin())
    p.add_argument("--sizes", type=int, nargs="+", default=[256, 512, 1024, 2048, 4096])
    p.add_argument("--tiles", type=int, nargs="+", default=[8, 16, 32])
    p.add_argument("--verify-sizes", type=int, nargs="+", default=[127, 256, 1000])
    p.add_argument("--reps", type=int, default=10, help="kernel launches averaged per run")
    p.add_argument("--trials", type=int, default=3, help="runs per config (median taken)")
    p.add_argument("--cpu-max", type=int, default=2048, help="skip CPU baseline above this N")
    p.add_argument("--out", default="results")
    a = p.parse_args()

    if not os.path.exists(a.bin):
        sys.exit(f"binary not found: {a.bin} (build it first, see README)")
    os.makedirs(a.out, exist_ok=True)

    if not verify(a.bin, a.verify_sizes, a.tiles):
        sys.exit("verification failed -- not benchmarking")

    rows = bench(a.bin, a.sizes, a.tiles, a.reps, a.trials, a.cpu_max)
    csv_path = os.path.join(a.out, "results.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["N", "kernel", "tile", "ms", "gflops"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nresults written to {csv_path}")
    summarize(rows)
    plot(rows, a.out)


if __name__ == "__main__":
    main()
