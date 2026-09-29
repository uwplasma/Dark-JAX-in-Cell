"""Isolated-process native/SOLVAX recurrence timings for one PIC objective.

Run ``python docs/scripts/benchmark_recurrence.py --all`` after installing
SOLVAX 0.27.0. This benchmark never changes the package's default backend.
"""

import argparse
import json
import os
import platform
import resource
import subprocess
import sys
from pathlib import Path
from statistics import median
from time import perf_counter

import jax
import numpy as np

HERE = Path(__file__).resolve()
OUT = HERE.parents[1] / "_static" / "figures" / "recurrence_benchmark.json"
METHODS = ("scan", "remat", "segmented", "solvax")


def synchronized(call):
    start = perf_counter()
    result = call()
    for leaf in jax.tree.leaves(result):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()
    return result, perf_counter() - start


def one(method, case):
    sys.path.insert(0, str(HERE.parents[2]))
    from examples.optimize_dark_photon import build_objective

    dt_bar = 0.2 * 2 * np.pi / 16
    stop = 256 * dt_bar if case == "exact" else 75.0
    start = 0.6 * stop
    objective, value_grad, steps = build_objective(16, 64, start, stop, method, 32)
    primal, primal_first = synchronized(lambda: objective(0.97))
    pair, gradient_first = synchronized(lambda: value_grad(0.97))
    primal_times = [synchronized(lambda: objective(0.97))[1] for _ in range(5)]
    gradient_times = [synchronized(lambda: value_grad(0.97))[1] for _ in range(5)]
    memory = value_grad.lower(0.97).compile().memory_analysis()
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {"case": case, "method": method, "steps": steps, "checkpoint_size": 32,
            "objective": float(primal), "gradient": float(pair[1]),
            "precision_x64": bool(jax.config.read("jax_enable_x64")),
            "backend": jax.default_backend(),
            "first_primal_s": primal_first, "first_gradient_s": gradient_first,
            "warm_primal_median_s": median(primal_times),
            "warm_primal_range_s": [min(primal_times), max(primal_times)],
            "warm_gradient_median_s": median(gradient_times),
            "warm_gradient_range_s": [min(gradient_times), max(gradient_times)],
            "compiled_temp_bytes": memory.temp_size_in_bytes,
            "process_peak_bytes": peak if platform.system() == "Darwin" else peak * 1024,
            "host_load_1m": os.getloadavg()[0]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--case", choices=("exact", "tail"))
    args = parser.parse_args()
    if args.all:
        import importlib.metadata as metadata

        rows = []
        for case in ("exact", "tail"):
            for method in METHODS:
                command = [sys.executable, str(HERE), "--method", method, "--case", case]
                rows.append(json.loads(subprocess.check_output(command, text=True)))
        for case in ("exact", "tail"):
            baseline = next(row for row in rows if row["case"] == case and row["method"] == "scan")
            for row in rows:
                if row["case"] == case:
                    if not row["precision_x64"] or row["backend"] != baseline["backend"]:
                        raise RuntimeError("benchmark methods must share a float64 backend")
                    np.testing.assert_allclose(row["objective"], baseline["objective"], rtol=1e-12)
                    np.testing.assert_allclose(row["gradient"], baseline["gradient"], rtol=1e-10)
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        result = {"git": sha, "jax": jax.__version__, "solvax": metadata.version("solvax"),
                  "equinox": metadata.version("equinox"), "platform": platform.platform(),
                  "backend": rows[0]["backend"], "precision_x64": rows[0]["precision_x64"],
                  "rows": rows}
        OUT.write_text(json.dumps(result, indent=2) + "\n")
        print(f"🦇 wrote {OUT}: {len(rows)} isolated-process comparisons")
    elif args.method and args.case:
        print(json.dumps(one(args.method, args.case)))
    else:
        parser.error("choose --all or both --method and --case")


if __name__ == "__main__":
    main()
