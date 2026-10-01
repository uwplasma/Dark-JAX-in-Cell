"""Measure matched PIC costs or repeated calls to the same compiled executables."""

import json
from importlib import metadata
import os
import platform
import resource
import shlex
import subprocess
import sys
from pathlib import Path
from functools import partial
from statistics import median
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
from jaxincell import (Domain, Simulation, Species, elementary_charge as e,
                       epsilon_0, mass_electron, save_run, speed_of_light as c)
from darkjaxincell import DarkField, DarkSimulation, PrescribedDrive


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path("artifacts/field_cost/field_cost.json")
STORAGE_OUTPUT = Path("artifacts/field_cost/storage_cost.json")

# Edit inputs here; runpy.run_path(..., run_name="__main__", init_globals={...}) overrides them for batches.
case = globals().get("case", "parent")
all_cases = globals().get("all_cases", False)
storage = globals().get("storage", None)
storage_all = globals().get("storage_all", False)
replay = globals().get("replay", False)  # One executable per kernel, three calls on the same initial arrays.
cells = globals().get("cells", 2000)
dt = globals().get("dt", .005)
seed = globals().get("seed", 0)
initial_state = globals().get("initial_state", None)
shape_order = globals().get("shape_order", 2)
particles = globals().get("particles", 206000 if replay else 128)
steps = globals().get("steps", 100 if replay else 16)
stride = globals().get("stride", steps if replay else 4)
output = Path(globals().get("output", "artifacts/reproducibility" if replay else OUTPUT))


def numpy_deposit(x, amounts, first, dx, cells, shape_order=2):
    """Independent periodic spline deposition with serial NumPy accumulation."""
    coordinate = (x - first) / dx
    if shape_order == 2:
        nearest = np.floor(coordinate + .5).astype(np.int64)
        offset = coordinate - nearest
        indices = nearest[:, None] + np.arange(-1, 2)
        weights = np.stack((.5 * (.5 - offset)**2, .75 - offset**2, .5 * (.5 + offset)**2), axis=1)
    elif shape_order == 5:
        from scipy.interpolate import BSpline
        indices = np.floor(coordinate).astype(np.int64)[:, None] + np.arange(-2, 4)
        basis = BSpline.basis_element(np.arange(7) - 3, extrapolate=False)
        weights = np.nan_to_num(basis(coordinate[:, None] - indices))
    else:
        raise ValueError('shape_order must be 2 or 5')
    result = np.zeros(cells, dtype=amounts.dtype)
    for column in range(weights.shape[1]):
        np.add.at(result, indices[:, column] % cells, amounts / dx * weights[:, column])
    return result


def measure_replay(cells, particles, dt, steps, stride, seed, initial_state, shape_order=2):
    """Test within-process repeatability; XLA_FLAGS must be set before importing JAX."""
    sys.path.insert(0, str(ROOT))
    from jaxincell._core import deposit
    from examples.dark_reservoir import array_fingerprint, paper_initial, paper_plasma
    from docs.scripts.conservation import measured_run, parent_revision, snapshot
    jax.config.update("jax_enable_x64", True)
    plasma, wp = paper_plasma(cells, particles, dt, seed, shape_order=shape_order)
    field = mass_electron * c * wp / e
    sim = DarkSimulation(plasma, PrescribedDrive(1., jnp.array([.03 * np.sqrt(.001) * field, 0., 0.]), wp))
    start, fingerprints = paper_initial(sim, seed, initial_state)
    reference = snapshot(sim, start)
    jax.block_until_ready((start, reference))

    def arrays(value):
        return {jax.tree_util.keystr(path): np.array(leaf)
                for path, leaf in jax.tree_util.tree_flatten_with_path(value)[0]}

    def repeat(lower, inputs):
        before = perf_counter()
        executable = lower().compile()
        compilation, memory = perf_counter() - before, executable.memory_analysis()
        baseline, rows = None, []
        for index in range(3):
            before = perf_counter()
            value = executable(*inputs)
            jax.block_until_ready(value)
            warm = perf_counter() - before
            values = arrays(value)
            baseline = values if baseline is None else baseline
            hashes = {key: array_fingerprint(a) for key, a in values.items()}
            errors = {key: float(np.max(abs(a.astype(np.complex128) - baseline[key]), initial=0))
                      for key, a in values.items()}
            rows.append(dict(warm_s=warm, array_sha256=hashes, max_absolute_error=errors,
                             bitwise_equal_to_first=hashes == rows[0]['array_sha256'] if rows else True))
            print(f'reproducibility call {index + 1}/3: {warm:.3f} s', file=sys.stderr, flush=True)
        return dict(compile_s=compilation, compiled_objects=1, calls=rows,
                    warm_median_s=median(row['warm_s'] for row in rows),
                    compiler_temporary_MiB=memory.temp_size_in_bytes / 2**20 if memory else None,
                    leaf_schema={key: dict(dtype=str(a.dtype), shape=list(a.shape))
                                 for key, a in baseline.items()}), baseline

    d, charge_scale = plasma.domain, e * float(plasma.species[0].density)
    x, amounts = start.ordinary.x[:, 0], plasma.per_particle[1] * start.ordinary.w
    weighting = partial(deposit, shape_order=5) if shape_order == 5 else deposit
    charge = jax.jit(lambda positions, charges: weighting(positions, charges, d.grid[0], d.dx, cells, (0, 0)))
    deposition, values = repeat(lambda: charge.lower(x, amounts), (x, amounts))
    rho = next(iter(values.values()))
    expected = numpy_deposit(np.asarray(x), np.asarray(amounts), float(d.grid[0]), float(d.dx), cells, shape_order)
    error = float(np.max(abs(rho - expected)) / charge_scale)
    deposition.update(numpy_max_error_over_en=error, numpy_tolerance_over_en=1e-11, numpy_check_passed=error <= 1e-11,
                      integrated_charge_error_over_enL=float(
                          abs(d.dx * np.sum(rho) - np.sum(np.asarray(amounts))) / (charge_scale * d.length)))
    run, _ = repeat(lambda: measured_run.lower(sim, start, steps, stride, reference, None, 1),
                    (sim, start, reference, None, 1))
    packages = {}
    for name in ('jax', 'jaxlib', 'jax-cuda12-plugin', 'jax-cuda12-pjrt', 'jax-cuda13-plugin', 'jax-cuda13-pjrt'):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    settings = dict(cells=cells, particles_per_species=particles, dt_omega_p=dt, steps=steps, stride=stride, seed=seed,
                    drive_quiver_over_sigma=.03, coupling=None, shape_order=shape_order,
                    parent_revision=parent_revision(), initial_fingerprints=fingerprints,
                    initial_leaf_sha256={key: array_fingerprint(a) for key, a in arrays(start).items()},
                    initial_state_source='complete zero-time archive' if initial_state else 'native initialization',
                    XLA_FLAGS=' '.join(token if '/' not in token and '\\' not in token
                                       else token.split('=')[0] + '=<path omitted>'
                                       for token in shlex.split(os.environ.get('XLA_FLAGS', ''))),
                    runtime_packages=packages, device_kinds=[device.device_kind for device in jax.devices()],
                    normalization=dict(charge_density_C_m3=float(charge_scale)),
                    note='One compiled object per kernel; three original-input calls. Device warm times exclude '
                    'host copies and hashes. Deposition uses native half-step x; errors are per leaf in native units. '
                    'Within-process repeatability does not establish late convergence or adopt compiler flags.')
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    save_run(output, 'pic_reproducibility', settings, dict(deposition=deposition, short_run=run,
             peak_rss_bytes=int(rss if platform.system() == 'Darwin' else rss * 1024)))


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
    if min(particles, steps, stride) < 1 or ((storage or storage_all or replay) and steps % stride):
        raise ValueError("positive particles/steps/stride and steps divisible by stride are required")
    if case not in (None, "parent", "eta_zero", "active") or storage not in (None, "sparse", "full"):
        raise ValueError("choose parent/eta_zero/active or sparse/full storage")
    print("Starting matched field/storage cost benchmark", file=sys.stderr, flush=True)
    if replay:
        if storage or storage_all or all_cases:
            raise ValueError('replay must run alone in its process')
        measure_replay(cells, particles, dt, steps, stride, seed, initial_state, shape_order)
    elif storage:
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
