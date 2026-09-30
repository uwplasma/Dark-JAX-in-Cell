"""Measure parent, configured zero-coupling, and active Proca on matched PIC work."""

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
import jax.numpy as jnp
from jaxincell import (Domain, Simulation, Species, elementary_charge as e,
                       epsilon_0, mass_electron, speed_of_light as c)
from darkjaxincell import DarkField, DarkSimulation


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "docs" / "_static" / "figures" / "field_cost.json"
STORAGE_OUTPUT = ROOT / "docs" / "_static" / "figures" / "storage_cost.json"


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

    start = perf_counter()
    checksum = run()
    first = perf_counter() - start
    samples = []
    for _ in range(5):
        start = perf_counter()
        checksum = run()
        samples.append(perf_counter() - start)
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

    start = perf_counter()
    final, maximum = run()
    first = perf_counter() - start
    samples = []
    for _ in range(3):
        start = perf_counter()
        final, maximum = run()
        samples.append(perf_counter() - start)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {"case": case, "first_call_s": first, "warm_s": samples,
            "warm_median_s": median(samples), "peak_rss_bytes": int(
                rss if platform.system() == "Darwin" else rss * 1024),
            "host_load": os.getloadavg(), "final_energy_J_m2": final,
            "all_step_max_balance_J_m2": maximum}


def main():
    parser = argparse.ArgumentParser(description="Count the ghost's matched field-step bill")
    parser.add_argument("--case", choices=("parent", "eta_zero", "active"))
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--storage", choices=("sparse", "full"))
    parser.add_argument("--storage-all", action="store_true")
    parser.add_argument("--particles", type=int, default=8192)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.storage:
        print(json.dumps(measure_storage(args.storage, args.particles, args.steps, args.stride)))
        return
    if args.storage_all:
        rows = [json.loads(subprocess.check_output(
            [sys.executable, str(Path(__file__).resolve()), "--storage", case,
             "--particles", str(args.particles), "--steps", str(args.steps),
             "--stride", str(args.stride)], cwd=ROOT, text=True))
            for case in ("sparse", "full")]
        record = {"settings": {"cells": 128, "particles": args.particles,
                               "steps": args.steps, "store_every": args.stride,
                               "precision": "float64", "backend": jax.default_backend(),
                               "jax": jax.__version__, "platform": platform.platform(),
                               "python": platform.python_version(),
                               "git": subprocess.check_output(["git", "rev-parse", "HEAD"],
                                                              cwd=ROOT, text=True).strip(),
                               "samples_per_case": 3,
                               "note": "fresh process per case; final total and all-step balance consumed"},
                  "rows": rows}
        path = args.output if args.output != OUTPUT else STORAGE_OUTPUT
        path.write_text(json.dumps(record, indent=2) + "\n")
        print(f"wrote {path}")
        return
    if args.case:
        print(json.dumps(measure(args.case, args.particles, args.steps)))
        return
    if not args.all:
        parser.error("choose --all or --case")
    rows = [json.loads(subprocess.check_output(
        [sys.executable, str(Path(__file__).resolve()), "--case", case,
         "--particles", str(args.particles), "--steps", str(args.steps)],
        cwd=ROOT, text=True)) for case in ("parent", "eta_zero", "active")]
    record = {"settings": {"cells": 128, "particles": args.particles, "steps": args.steps,
                           "precision": "float64", "backend": jax.default_backend(),
                           "jax": jax.__version__, "platform": platform.platform(),
                           "python": platform.python_version(),
                           "git": subprocess.check_output(["git", "rev-parse", "HEAD"],
                                                          cwd=ROOT, text=True).strip(),
                           "samples_per_case": 5,
                           "note": "fresh process per case; first call includes compile"},
              "rows": rows}
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
