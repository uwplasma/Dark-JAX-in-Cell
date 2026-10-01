"""Isolated-process native/SOLVAX recurrence timings for one PIC objective.

Set all_cases=True after installing SOLVAX 0.27.0 to compare all methods.
This benchmark never changes the package's default backend.
"""

from importlib import metadata
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
OUT = Path("artifacts/recurrence/recurrence_benchmark.json")
METHODS = ("scan", "remat", "segmented", "solvax")

# Editable inputs also accept runpy.run_path(..., init_globals={...}) batch overrides.
method = globals().get("method", "scan")
case = globals().get("case", "exact")
all_cases = globals().get("all_cases", False)
output = Path(globals().get("output", OUT))


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
    print(f"Compiling {method}/{case}: {steps} steps", file=sys.stderr, flush=True)
    primal, primal_first = synchronized(lambda: objective(0.97))
    pair, gradient_first = synchronized(lambda: value_grad(0.97))
    primal_times, gradient_times = [], []
    for _ in range(5):
        primal_times.append(synchronized(lambda: objective(0.97))[1])
        print(f"{method}/{case}: warm primal {len(primal_times)}/5", file=sys.stderr, flush=True)
    for _ in range(5):
        gradient_times.append(synchronized(lambda: value_grad(0.97))[1])
        print(f"{method}/{case}: warm gradient {len(gradient_times)}/5", file=sys.stderr, flush=True)
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


def isolated(method, case):
    """One method/loading case per fresh process."""
    command = (f"import runpy; runpy.run_path({str(HERE)!r}, run_name='__main__', "
               f"init_globals={dict(method=method, case=case)!r})")
    return json.loads(subprocess.check_output([sys.executable, "-c", command], text=True))


if __name__ == "__main__":
    if method not in METHODS or case not in ("exact", "tail"):
        raise ValueError("choose scan/remat/segmented/solvax and exact/tail")
    print("Starting recurrence cost benchmark", file=sys.stderr, flush=True)
    if all_cases:
        rows = [isolated(method, case) for case in ("exact", "tail") for method in METHODS]
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
                  "backend": rows[0]["backend"], "precision_x64": rows[0]["precision_x64"], "rows": rows}
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n")
        print(f"wrote {output}: {len(rows)} isolated-process comparisons", file=sys.stderr, flush=True)
    else:
        print(json.dumps(one(method, case)))
    print("Finished recurrence cost benchmark", file=sys.stderr, flush=True)
