"""Measure parent, configured zero-coupling, and active Proca on matched PIC work."""

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
import jax.numpy as jnp
from jaxincell import (Domain, Simulation, Species, elementary_charge as e,
                       epsilon_0, mass_electron, speed_of_light as c)
from darkjaxincell import DarkField, DarkSimulation


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path("artifacts/field_cost/field_cost.json")
STORAGE_OUTPUT = Path("artifacts/field_cost/storage_cost.json")

# Edit inputs here; runpy.run_path(..., run_name="__main__", init_globals={...}) overrides them for batches.
case = globals().get("case", "parent")
all_cases = globals().get("all_cases", False)
storage = globals().get("storage", None)
storage_all = globals().get("storage_all", False)
particles = globals().get("particles", 128)
steps = globals().get("steps", 16)
stride = globals().get("stride", 4)
output = Path(globals().get("output", OUTPUT))


def measure(case, particles, steps):
    """One fresh process, including JAX's first compilation and five warm runs."""
    jax.config.update("jax_enable_x64", True)
    omega, cells = 1e9, 128
    density = 0.8 * epsilon_0 * mass_electron * omega**2 / e**2
    domain = Domain(length=2 * 3.141592653589793 * c / omega,
                    cells=cells, dt_over_dx_c=0.2)
    plasma = Simulation(domain, (Species.electrons(particles, density=density),))
    if case == "parent":
        simulation = plasma
    else:
        initial = jnp.tile(jnp.array([0.0, 1e-5, 0.0]), (cells, 1))
        eta = 0.0 if case == "eta_zero" else 0.05
        simulation = DarkSimulation(plasma, DarkField(omega, eta, initial_E=initial))

    def run():
        output = simulation.run(steps, seed=0, store_every=steps, store_particles=False)
        final_E = output.E if case == "parent" else output.ordinary.E
        final_E.block_until_ready()
        return float(jnp.sum(final_E[-1] ** 2))

    print(f"Compiling {case}: {particles} particles, {steps} steps", file=sys.stderr, flush=True)
    start = perf_counter()
    checksum = run()
    first = perf_counter() - start
    samples = []
    for _ in range(5):
        start = perf_counter()
        checksum = run()
        samples.append(perf_counter() - start)
        print(f"{case}: warm execution {len(samples)} finished in {samples[-1]:.3f} s",
              file=sys.stderr, flush=True)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {"case": case, "first_call_s": first, "warm_s": samples,
            "warm_median_s": median(samples), "peak_rss_bytes": int(
                rss if platform.system() == "Darwin" else rss * 1024),
            "host_load": os.getloadavg(),
            "ordinary_E_square_checksum": checksum}


def measure_storage(case, particles, steps, stride):
    """Benchmark the complete energy objective with and without particle histories."""
    jax.config.update("jax_enable_x64", True)
    omega, cells = 1e9, 128
    density = 0.8 * epsilon_0 * mass_electron * omega**2 / e**2
    domain = Domain(length=2 * 3.141592653589793 * c / omega,
                    cells=cells, dt_over_dx_c=0.2)
    plasma = Simulation(domain, (Species.electrons(particles, density=density),))
    initial = jnp.tile(jnp.array([0.0, 1e-5, 0.0]), (cells, 1))
    sim = DarkSimulation(plasma, DarkField(omega, 0.05, initial_E=initial))

    def run():
        out = sim.run(steps, seed=0, store_every=stride,
                      store_particles=case == "full")
        final = out.energy()["total_with_dark"][-1]
        maximum = out.state.max_balance_error
        final.block_until_ready()
        maximum.block_until_ready()
        if case == "full":
            for values in (out.ordinary.x, out.ordinary.v, out.ordinary.weight):
                values.block_until_ready()
        return float(final), float(maximum)

    print(f"Compiling {case} storage: {particles} particles, {steps} steps", file=sys.stderr, flush=True)
    start = perf_counter()
    final, maximum = run()
    first = perf_counter() - start
    samples = []
    for _ in range(3):
        start = perf_counter()
        final, maximum = run()
        samples.append(perf_counter() - start)
        print(f"{case}: warm execution {len(samples)} finished in {samples[-1]:.3f} s",
              file=sys.stderr, flush=True)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {"case": case, "first_call_s": first, "warm_s": samples,
            "warm_median_s": median(samples), "peak_rss_bytes": int(
                rss if platform.system() == "Darwin" else rss * 1024),
            "host_load": os.getloadavg(), "final_energy_J_m2": final,
            "all_step_max_balance_J_m2": maximum}


def isolated(inputs):
    """Keep each timing row in a fresh process, with editable-input overrides."""
    command = (f"import runpy; runpy.run_path({str(Path(__file__).resolve())!r}, "
               f"run_name='__main__', init_globals={inputs!r})")
    return json.loads(subprocess.check_output([sys.executable, "-c", command], cwd=ROOT, text=True))


if __name__ == "__main__":
    if min(particles, steps, stride) < 1 or ((storage or storage_all) and steps % stride):
        raise ValueError("positive particles/steps/stride and steps divisible by stride are required")
    if case not in (None, "parent", "eta_zero", "active") or storage not in (None, "sparse", "full"):
        raise ValueError("choose parent/eta_zero/active or sparse/full storage")
    print("Starting matched field/storage cost benchmark", file=sys.stderr, flush=True)
    if storage:
        print(json.dumps(measure_storage(storage, particles, steps, stride)))
    elif storage_all or all_cases:
        names = ("sparse", "full") if storage_all else ("parent", "eta_zero", "active")
        rows = [isolated(dict(storage=name, particles=particles, steps=steps, stride=stride)
                         if storage_all else dict(case=name, particles=particles, steps=steps)) for name in names]
        record = {"settings": {"cells": 128, "particles": particles, "steps": steps,
                               "precision": "float64", "backend": jax.default_backend(),
                               "jax": jax.__version__, "platform": platform.platform(),
                               "python": platform.python_version(),
                               "git": subprocess.check_output(["git", "rev-parse", "HEAD"],
                                                              cwd=ROOT, text=True).strip(),
                               "samples_per_case": 3 if storage_all else 5,
                               "note": "fresh process per case; first call includes compile"}, "rows": rows}
        if storage_all:
            record["settings"].update(store_every=stride,
                                      note="fresh process per case; final total and all-step balance consumed")
        path = STORAGE_OUTPUT if storage_all and output == OUTPUT else output
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, indent=2) + "\n")
        print(f"wrote {path}", file=sys.stderr, flush=True)
    elif case:
        print(json.dumps(measure(case, particles, steps)))
    else:
        raise ValueError("choose a case or set all_cases/storage_all")
    print("Finished matched field/storage cost benchmark", file=sys.stderr, flush=True)
