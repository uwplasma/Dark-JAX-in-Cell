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
waveform_sampling = globals().get("waveform_sampling", False)
cells = globals().get("cells", 4096 if waveform_sampling else 2000)
particles_per_cell = globals().get("particles_per_cell", 128)
dt = globals().get("dt", .005)
seed = globals().get("seed", 0)
initial_state = globals().get("initial_state", None)
shape_order = globals().get("shape_order", 2)
particles = globals().get("particles", 206000 if replay else 128)
steps = globals().get("steps", 100 if replay else 16)
stride = globals().get("stride", steps if replay else 4)
output = Path(globals().get("output", "artifacts/waveform_sampling" if waveform_sampling
                            else "artifacts/reproducibility" if replay else OUTPUT))


def execution_settings():
    """Pin accelerator dependencies and compiler flags without exposing local paths."""
    packages = {}
    for name in ('jax', 'jaxlib', 'jax-cuda12-plugin', 'jax-cuda12-pjrt', 'jax-cuda13-plugin', 'jax-cuda13-pjrt'):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    flags = ' '.join(token if '/' not in token and '\\' not in token
                     else token.split('=')[0] + '=<path omitted>'
                     for token in shlex.split(os.environ.get('XLA_FLAGS', '')))
    return dict(runtime_packages=packages, device_kinds=[device.device_kind for device in jax.devices()],
                XLA_FLAGS=flags)


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
    settings = dict(cells=cells, particles_per_species=particles, dt_omega_p=dt, steps=steps, stride=stride, seed=seed,
                    drive_quiver_over_sigma=.03, coupling=None, shape_order=shape_order,
                    parent_revision=parent_revision(), initial_fingerprints=fingerprints,
                    initial_leaf_sha256={key: array_fingerprint(a) for key, a in arrays(start).items()},
                    initial_state_source='complete zero-time archive' if initial_state else 'native initialization',
                    **execution_settings(),
                    normalization=dict(charge_density_C_m3=float(charge_scale)),
                    note='One compiled object per kernel; three original-input calls. Device warm times exclude '
                    'host copies and hashes. Deposition uses native half-step x; errors are per leaf in native units. '
                    'Within-process repeatability does not establish late convergence or adopt compiler flags.')
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    save_run(output, 'pic_reproducibility', settings, dict(deposition=deposition, short_run=run,
             peak_rss_bytes=int(rss if platform.system() == 'Darwin' else rss * 1024)))


def measure_waveform(cells, particles_per_cell, dt, steps, stride):
    """Measure sparse moments with dense forcing, against the full native record."""
    sys.path.insert(0, str(ROOT))
    from examples.dark_reservoir import array_fingerprint, pair_plasma, pair_push_table
    from docs.scripts.conservation import elapsed_progress, measured_run, parent_revision

    jax.config.update('jax_enable_x64', True)
    plasma, _, _, wp = pair_plasma(cells, particles_per_cell, dt, 2e-4, True, pair_loading='global')
    field = mass_electron * c * wp / e
    sim = DarkSimulation(plasma, DarkField(wp, .5, initial_E=jnp.tile(
        jnp.array([.1 * field, 0., 0.]), (cells, 1))))
    start, _ = sim.initial_state(jax.random.PRNGKey(0))
    jax.block_until_ready(start)
    initial_hashes = {jax.tree_util.keystr(path): array_fingerprint(value)
                      for path, value in jax.tree_util.tree_flatten_with_path(start)[0]}
    scales, runs, timing = jnp.array([.1, .2]) * c / wp, {}, {}
    for label, every, pump in (('dense', 1, False), ('sparse_moments', stride, True)):
        with elapsed_progress(label):
            before = perf_counter()
            executable = measured_run.lower(sim, start, steps, every, scales=scales, pump=pump).compile()
            compilation, memory = perf_counter() - before, executable.memory_analysis()
            rows, calls, baseline = [], [], None
            for index in range(3):
                before = perf_counter()
                result = executable(sim, start, scales=scales)
                jax.block_until_ready(result)
                rows.append(perf_counter() - before)
                values = {jax.tree_util.keystr(path): np.asarray(value)
                          for path, value in jax.tree_util.tree_flatten_with_path(result)[0]}
                baseline = values if baseline is None else baseline
                hashes = {key: array_fingerprint(value) for key, value in values.items()}
                calls.append(dict(array_sha256=hashes,
                                  bitwise_equal_to_first=hashes == calls[0]['array_sha256'] if calls else True,
                                  max_absolute_difference_SI={key: float(np.max(abs(value - baseline[key]), initial=0))
                                                              for key, value in values.items()}))
                print(f'{label}: call {index + 1}/3, {rows[-1]:.3f} s', flush=True)
        runs[label] = jax.tree.map(np.asarray, result)
        timing[label] = dict(compile_s=compilation, warm_s=rows, warm_median_s=median(rows), calls=calls,
                             compiler_temporary_MiB=memory.temp_size_in_bytes / 2**20 if memory else None)
    dense, sampled = runs['dense'], runs['sparse_moments']
    fields = dict(E=1 / field, B=c / field, A=wp / field, phi=wp / (c * field))
    state_errors = {key: float(np.max(abs(getattr(dense[0], key) - getattr(sampled[0], key))) * factor)
                    for key, factor in fields.items()}
    for key, factor in dict(E=1 / field, B=c / field, x=1 / plasma.domain.length, u=1 / c,
                            rho=1 / (e * sum(float(s.density) for s in plasma.species))).items():
        difference = getattr(dense[0].ordinary, key) - getattr(sampled[0].ordinary, key)
        state_errors['ordinary_' + key] = float(np.max(abs(difference)) * factor)
    fixed = ('w', 'qm', 'time', 'steps', 'key', 'sigma', 'wall', 'moments')
    fixed_leaves = [jax.tree.leaves(tuple(getattr(result[0].ordinary, key) for key in fixed))
                    for result in (dense, sampled)]
    state_equal = (np.max(list(state_errors.values())) < 1e-10
                   and all(np.array_equal(a, b) for a, b in zip(*fixed_leaves)))
    force = {key: sampled[1]['pump_' + key] for key in ('t', 'mean', 'mean_D', 'mean_A')}
    original = pair_push_table(plasma, start.ordinary, dense[1], wp, .5)
    sparse = pair_push_table(plasma, start.ordinary, force, wp, .5)
    force_error = float(np.max(abs(original[1] - sparse[1])) / (.1 * field))
    scalar_errors = {key: float(np.max(abs(dense[1][key][::stride] - sampled[1][key])))
                     for key in dense[1]}
    scale = epsilon_0 * plasma.domain.length * field**2
    energy_max_error = float(abs(dense[2][0] - sampled[2][0]) / scale)
    charge_scale = e * sum(float(s.density) for s in plasma.species)
    maximum_scales = np.array([scale, scale / c, charge_scale * plasma.domain.length, charge_scale * wp,
                              charge_scale / epsilon_0, charge_scale / epsilon_0,
                              charge_scale * plasma.domain.length, scale, scale])
    maximum_errors = abs(dense[2] - sampled[2]) / maximum_scales
    ledger_scales = dict(background=charge_scale, work=scale, initial_ordinary=scale, initial_dark=scale,
                         initial_projection_norm=field * np.sqrt(plasma.domain.length), max_balance_error=scale,
                         max_ordinary_gauss=charge_scale / epsilon_0, max_dark_gauss=charge_scale / epsilon_0)
    ledger_errors = {key: float(abs(getattr(dense[0], key) - getattr(sampled[0], key)) / divisor)
                     for key, divisor in ledger_scales.items()}
    scalar_scales = dict.fromkeys(('electric', 'magnetic', 'dark', 'kinetic', 'spread', 'balance', 'work',
                                  'dark_coherent', 'local_spread'), scale)
    scalar_scales.update(dict.fromkeys(('mean', 'rms', 'max_speed'), c))
    scalar_scales.update(dict.fromkeys(('mean_E', 'mean_D', 'mode_E', 'dark_mode_E'), field))
    scalar_scales.update(dict.fromkeys(('ordinary_gauss', 'dark_gauss'), charge_scale / epsilon_0))
    scalar_scales.update(t=steps * dt / wp, momentum=scale / c, charge=charge_scale * plasma.domain.length,
                         grid_charge=charge_scale * plasma.domain.length, mean_A=field / wp,
                         density_rms=1., local_density_rms=1.)
    scalar_normalized = {key: error / scalar_scales[key] for key, error in scalar_errors.items()}
    errors = np.r_[list(state_errors.values()), list(ledger_errors.values()), list(scalar_normalized.values()),
                   maximum_errors, force_error]
    checks = dict(state_allclose=state_equal, forcing_clock_equal=bool(np.array_equal(original[0], sparse[0])),
                  normalized_state_max_absolute=state_errors, normalized_state_tolerance=1e-10,
                  max_force_error_over_initial=force_error, all_step_energy_max_difference_over_Uref=energy_max_error,
                  all_step_max_difference_normalized=maximum_errors.tolist(),
                  final_ledger_difference_normalized=ledger_errors,
                  sampled_scalar_difference_normalized=scalar_normalized,
                  sampled_scalar_max_absolute_SI=scalar_errors,
                  passed=bool(state_equal and np.array_equal(original[0], sparse[0])
                              and np.all(np.isfinite(errors)) and np.max(errors) < 1e-10))
    settings = dict(cells=cells, particles_per_cell_per_species=particles_per_cell, dt_omega0=dt,
                    steps=steps, scalar_stride=stride, dense_force_stride=1, local_scales_c_over_omega0=[.1, .2],
                    pair_loading='global', velocity_seed_over_c=2e-4, eta=.5, dark_mass_over_omega0=1.,
                    force_quiver_over_c=.05, shape_order=2,
                    parent_revision=parent_revision(), samples=3, initial_leaf_sha256=initial_hashes,
                    **execution_settings(),
                    note='One compiled object per sampling mode; three synchronized calls. '
                    'All-step maxima and dense push forcing are retained. Short-run sampling check only.')
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    save_run(output, 'waveform_sampling_cost', settings, dict(timing=timing, checks=checks,
             peak_rss_bytes=int(rss if platform.system() == 'Darwin' else rss * 1024),
             speedup=timing['dense']['warm_median_s'] / timing['sparse_moments']['warm_median_s']))
    if not checks['passed']:
        raise ValueError('sparse waveform sampling disagrees with the native dense control')


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


if __name__ == "__main__":  # noqa: C901
    if min(particles, steps, stride) < 1 or ((storage or storage_all or replay) and steps % stride):
        raise ValueError("positive particles/steps/stride and steps divisible by stride are required")
    if case not in (None, "parent", "eta_zero", "active") or storage not in (None, "sparse", "full"):
        raise ValueError("choose parent/eta_zero/active or sparse/full storage")
    print("Starting matched field/storage cost benchmark", file=sys.stderr, flush=True)
    if waveform_sampling:
        if replay or storage or storage_all or all_cases or steps % stride:
            raise ValueError('waveform_sampling must run alone with steps divisible by stride')
        measure_waveform(cells, particles_per_cell, dt, steps, stride)
    elif replay:
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
