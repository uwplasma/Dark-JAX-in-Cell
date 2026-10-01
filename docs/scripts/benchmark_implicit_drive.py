"""Validation prototype: a uniform cosine force on the parent's implicit PIC.

For F held at the step midpoint, U(E+F)-U(E)=epsilon0*dx*sum(E.F+F.F/2).
Periodic Ampere gives sum(delta E)=-dt*sum(J)/epsilon0. Hence
delta U_physical-dt*dx*sum(J.F) is exactly the shifted parent energy defect.
Its accepted continuity current keeps Gauss at finite Picard count; energy
additionally needs the orbit/field iteration to converge. This is a prescribed
external force, with work supplied externally, not implicit Proca evolution.
"""

from functools import partial
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import sys
from time import perf_counter
from types import SimpleNamespace

import jax
import jax.numpy as jnp
from jax import lax
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from numpy.polynomial.hermite import hermgauss  # noqa: E402
from scipy.integrate import solve_ivp  # noqa: E402
from jaxincell import (Domain, Simulation, Solver, Species, elementary_charge as e, epsilon_0,  # noqa: E402
                       mass_electron as m, save_run, speed_of_light as c)  # noqa: E402
from jaxincell._core import deposit, E_x_from_rho  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from darkjaxincell import PrescribedDrive, load_state, midnight  # noqa: E402
from darkjaxincell._simulation import DarkState  # noqa: E402
from darkjaxincell._proca import divergence  # noqa: E402
from docs.scripts.conservation import elapsed_progress, snapshot  # noqa: E402
from docs.scripts.drive_reference import homogeneous  # noqa: E402


WP, MASS_RATIO = 1e9, 1836.
N = epsilon_0 * m * WP**2 / e**2
FIELD = m * c * WP / e

# Editable inputs also accept runpy.run_path(..., init_globals={...}) batch overrides.
# Full Gaussian studies explicitly set cells=1000, particles=103000 and their required horizon.
cells = globals().get("cells", 8)
nodes = globals().get("nodes", 4)
rings = globals().get("rings", 1)
dt = globals().get("dt", .04)
iterations = globals().get("iterations", 8)
substeps = globals().get("substeps", 2)
horizon = globals().get("horizon", 8.)
amplitude = globals().get("amplitude", .03 * np.sqrt(.001))
temperature = globals().get("temperature", .001)
samples = globals().get("samples", 1)
gradient_horizon = globals().get("gradient_horizon", 0.)
newtonian = globals().get("newtonian", False)
paper_loading = globals().get("paper_loading", False)
particles = globals().get("particles", 64)
initial_state = globals().get("initial_state", None)
audit_state = globals().get("audit_state", None)
output = Path(globals().get("output", "artifacts/implicit_drive"))


def drive_state(plasma, ordinary):
    """Start the existing complete work ledger at native integer-time positions."""
    values = snapshot(plasma, ordinary)
    zero = jnp.zeros(())
    return DarkState(ordinary, None, None, None, None, -jnp.mean(ordinary.rho), zero,
                     values["balance"], zero, zero, zero, values["ordinary_gauss"], zero)


def implicit_drive_step(plasma, state, drive):
    """Reuse one parent orbit solve and return its accepted midpoint current."""
    d, s = plasma.domain, plasma.solver
    if (s.algorithm != "implicit" or s.electrostatic or s.field_solver != "ampere" or s.filter_passes
            or d.field_bc != (0, 0) or d.particle_bc != (0, 0) or plasma.collisions is not None
            or plasma.sources or plasma.external_E is not None or plasma.external_B is not None):
        raise ValueError("prototype requires periodic implicit electromagnetic Ampere PIC without other sources")
    if not isinstance(drive, PrescribedDrive) or jnp.shape(drive.amplitude) != (3,):
        raise ValueError("prototype requires a uniform three-component PrescribedDrive")
    old = state.ordinary
    force = drive.eta * jnp.asarray(drive.amplitude) * jnp.cos(
        drive.omega * (old.time + d.dt / 2) + drive.phase)
    ordinary, output = plasma._implicit_step(old.replace(E=old.E + force), plasma.per_particle)
    current = output[5]
    ordinary = ordinary.replace(E=ordinary.E - force)
    work = state.work + d.dt * d.dx * jnp.sum(current * force)
    values = snapshot(plasma, ordinary, state.background)
    return state.replace(ordinary=ordinary, work=work,
                         max_balance_error=jnp.maximum(state.max_balance_error,
                                                       abs(values["balance"] - work - state.initial_ordinary)),
                         max_ordinary_gauss=jnp.maximum(state.max_ordinary_gauss, values["ordinary_gauss"])), current


def sample(plasma, state):
    values = snapshot(plasma, state.ordinary, state.background)
    current = jnp.sum(plasma.per_particle[1] * state.ordinary.w
                      * plasma._velocity(state.ordinary.u)[:, 0]) / plasma.domain.length
    fluctuation = state.ordinary.E - jnp.mean(state.ordinary.E, axis=0)
    nonzero = epsilon_0 * plasma.domain.dx * jnp.sum(fluctuation**2) / 2
    return {**values, "balance": values["balance"] - state.work, "work": state.work,
            "current": current, "nonzero_electric": nonzero}


@partial(jax.jit, static_argnames=("steps", "stride"))
def run_drive(plasma, initial, drive, steps, stride=1):
    """Sparse scalar histories, actual initial sample and all-step SI maxima.

    Maxima: work balance, particle charge, grid charge, continuity, Gauss,
    mean Ampere residual, continuum momentum. The native state/work ledger
    can be resumed with the same model; absolute time fixes the drive phase.
    """
    if steps < 1 or stride < 1 or steps % stride:
        raise ValueError("steps must be positive and divisible by stride")
    reference = sample(plasma, initial)

    def one(carry, _):
        state, maxima = carry
        before = state.ordinary
        state, current = implicit_drive_step(plasma, state, drive)
        after, values = state.ordinary, sample(plasma, state)
        continuity = (after.rho - before.rho) / plasma.domain.dt + divergence(current, plasma.domain.dx)
        ampere = jnp.mean((after.E - before.E) / plasma.domain.dt + current / epsilon_0, axis=0)
        errors = jnp.array([state.max_balance_error, abs(values["charge"] - reference["charge"]),
                            abs(values["grid_charge"] - reference["grid_charge"]), jnp.max(abs(continuity)),
                            state.max_ordinary_gauss, jnp.max(abs(ampere)),
                            jnp.max(abs(values["momentum"] - reference["momentum"]))])
        return (state, jnp.maximum(maxima, errors)), None

    def chunk(carry, _):
        carry, _ = lax.scan(one, carry, None, length=stride)
        return carry, sample(plasma, carry[0])

    (state, maxima), history = lax.scan(chunk, (initial, jnp.zeros(7)), None, length=steps // stride)
    history = jax.tree.map(lambda a, b: jnp.concatenate((a[None], b)), reference, history)
    return state, history, maxima


def homogeneous_box(cells=8, nodes=8, rings=1, dtau=.02, iterations=8, *,
                    temperature=1e-3, density=1., relativistic=True):
    """Replicate each weighted velocity quadrature uniformly over the periodic grid.

    Gaussian velocity weights use Gauss-Hermite quadrature, as in the independent
    homogeneous oracle. Density and thermal initialization remain differentiable.
    This supplies a k=0 control while spatial modes remain negligible. Its discrete
    velocity beams can support late spatial growth; they are not a continuum plasma.
    """
    if min(cells, nodes, rings, iterations) < 1 or dtau <= 0:
        raise ValueError("positive resolution, iteration count and time step are required")
    points, weights = hermgauss(nodes)
    for value, label in ((temperature, "thermal width"), (density, "density")):
        if not isinstance(value, jax.core.Tracer):
            number = np.asarray(value)
            if number.shape != () or not np.isfinite(number) or number < 0 or (label == "density" and number == 0):
                raise ValueError("finite nonnegative thermal width and positive density are required")
    if not isinstance(temperature, jax.core.Tracer) and relativistic:
        if 2 * float(temperature) * max(abs(points))**2 >= 1 - 1e-5:
            raise ValueError("quadrature velocities exceed the parent's relativistic input margin")
    weights /= np.sqrt(np.pi)
    length, markers = 2 * np.pi * c / WP, cells * rings
    x = -length / 2 + (np.arange(markers) + .5) * length / markers
    positions = jnp.zeros((nodes * markers, 3)).at[:, 0].set(jnp.tile(jnp.asarray(x), nodes))
    populations = []
    for name, charge, ratio in (("electrons", -1, 1.), ("ions", 1, MASS_RATIO)):
        velocity = jnp.repeat(jnp.sqrt(2 * temperature / ratio) * jnp.asarray(points) * c, markers)
        v = jnp.zeros_like(positions).at[:, 0].set(velocity)
        populations.append(Species(name, nodes * markers, charge, ratio * m, density * N, x=positions, v=v))
    plasma = Simulation(Domain(length, cells, time_step=dtau / WP), tuple(populations),
                        Solver(algorithm="implicit", relativistic=relativistic,
                               picard_iterations=iterations, substeps=2))
    ordinary, (_, q) = plasma.initial_state(jax.random.PRNGKey(0))
    w = density * N * length / markers * jnp.tile(jnp.repeat(jnp.asarray(weights), markers), 2)
    rho = deposit(ordinary.x[:, 0], q * w, plasma.domain.grid[0], plasma.domain.dx, cells, (0, 0))
    ordinary = ordinary.replace(w=w, rho=rho, E=ordinary.E.at[:, 0].set(E_x_from_rho(rho, plasma.domain.dx, (0, 0))))
    return plasma, drive_state(plasma, ordinary)


def tangent_reference(times, controls, temperature=1e-3, nodes=64, relativistic=True):
    """Independent ODE/Frechet response to amplitude, density, frequency and phase.

    Momentum-distribution response: I'=E+A*cos(omega*t+phase), E'=-n*(vi-ve).
    The analytic derivative dv/du=gamma^-3 propagates the four sensitivities.
    All units use the fixed initial electron reference wp, c and n me c squared.
    """
    amplitude, density, omega, phase = np.asarray(controls)
    points, weights = hermgauss(nodes)
    weights /= np.sqrt(np.pi)
    velocity = np.sqrt(2 * temperature / np.array([1., MASS_RATIO])[:, None]) * points
    if relativistic and np.max(abs(velocity)) >= 1:
        raise ValueError("quadrature velocities exceed c")
    initial = velocity / np.sqrt(1 - velocity**2) if relativistic else velocity

    def rhs(time, state):
        electric, impulse = state[:2]
        u = initial + np.array([-1., 1 / MASS_RATIO])[:, None] * impulse
        gamma = np.sqrt(1 + u**2) if relativistic else np.ones_like(u)
        mean = (u / gamma) @ weights
        response = mean[1] - mean[0]
        slope = density * ((gamma[0]**-3 + gamma[1]**-3 / MASS_RATIO) @ weights)
        cosine, sine = np.cos(omega * time + phase), np.sin(omega * time + phase)
        sensitivity = state[2:].reshape(2, 4)
        current_derivative = slope * sensitivity[1] + np.array([0., response, 0., 0.])
        force_derivative = np.array([cosine, 0., -amplitude * time * sine, -amplitude * sine])
        return np.r_[-density * response, electric + amplitude * cosine,
                     -current_derivative, sensitivity[0] + force_derivative]

    solution = solve_ivp(rhs, (0., float(times[-1])), np.zeros(10), t_eval=times,
                         method="DOP853", rtol=2e-11, atol=2e-13)
    if not solution.success:
        raise RuntimeError(solution.message)
    return dict(mean_E=solution.y[0], impulse=solution.y[1],
                energy_gradient=solution.y[0, -1] * solution.y[2:6, -1], nfev=solution.nfev)


def objective(controls, steps, *, cells=8, nodes=8, rings=1, dtau=.02, iterations=8, temperature=1e-3):
    """Physical final mean-field energy, including density weights and initialization."""
    plasma, state = homogeneous_box(cells, nodes, rings, dtau, iterations,
                                    temperature=temperature, density=controls[1])
    drive = PrescribedDrive(1., jnp.array([controls[0] * FIELD, 0., 0.]), controls[2] * WP, controls[3])
    step = jax.checkpoint(lambda carry: implicit_drive_step(plasma, carry, drive)[0])
    state = lax.fori_loop(0, steps, lambda _, carry: step(carry), state)
    return .5 * (jnp.mean(state.ordinary.E[:, 0]) / FIELD)**2


def windows(t):
    """Cumulative sampled windows, including the actual final time if shorter."""
    for end in sorted({min(float(t[-1]), limit) for limit in (40., 100., 250., 500., 1000.)}):
        yield f"{end:g}", t <= end + 1e-9 * max(1., end)


def norm_errors(t, observed, reference):
    """Raw dimensionless L2 norms; a zero reference has no relative error."""
    results = {}
    for end, mask in windows(t):
        delta, baseline = observed[mask] - reference[mask], reference[mask]
        error, norm = np.linalg.norm(delta, axis=0), np.linalg.norm(baseline, axis=0)
        relative = np.divide(error, norm, out=np.full_like(error, np.nan), where=norm > 0)
        results[end] = dict(samples=int(mask.sum()), final_time=float(t[mask][-1]),
                            difference_l2=error.tolist(), reference_l2=norm.tolist(),
                            observed_l2=np.linalg.norm(observed[mask], axis=0).tolist(),
                            max_abs_difference=np.max(abs(delta), axis=0).tolist(),
                            relative_l2=np.where(np.isfinite(relative), relative, None).tolist())
    return results


def crossings(t, nonzero):
    """First recorded threshold exceedance; no interpolation between samples."""
    return {f"{threshold:g}": float(t[np.flatnonzero(nonzero > threshold)[0]])
            if np.any(nonzero > threshold) else None for threshold in (1e-16, 1e-12, 1e-8, 1e-4)}


def load_plasma(args):
    """Keep the supplied physical loading, then initialize the native implicit clock."""
    substeps = getattr(args, 'substeps', 2)
    if (not isinstance(substeps, (int, np.integer)) or isinstance(substeps, (bool, np.bool_)) or substeps < 1
            or (not args.paper_loading and substeps != 2)):
        raise ValueError('positive integer substeps are required; changing them requires paper loading')
    if not args.paper_loading:
        return homogeneous_box(args.cells, args.nodes, args.rings, args.dt, args.iterations,
                               temperature=args.temperature, relativistic=not args.newtonian)
    from examples.dark_reservoir import paper_plasma
    plasma, _ = paper_plasma(args.cells, args.particles, args.dt, 0)
    plasma = plasma.replace(solver=Solver(algorithm="implicit", relativistic=True,
                                          picard_iterations=args.iterations, substeps=substeps))
    ordinary, _ = plasma.initial_state(jax.random.PRNGKey(0))
    return plasma, drive_state(plasma, ordinary)


def archived_initial(path, initial, model):
    """Reuse every native array while checking the supplied particles and zero clock."""
    if path is None:
        return initial
    restored = load_state(path, model)
    for key in ('x', 'u', 'w'):
        np.testing.assert_array_equal(getattr(initial.ordinary, key), getattr(restored.ordinary, key))
    if float(restored.ordinary.time) != 0 or float(restored.work) != 0:
        raise ValueError('implicit benchmark requires a zero-time, zero-work initial archive')
    return restored


def audit_orbits(args):
    """Probe one archived pure-electric step against independent NumPy orbits."""
    from docs.scripts.drive_reference import midpoint_orbits
    plasma, _ = load_plasma(args)
    drive = PrescribedDrive(1., jnp.array([args.amplitude * FIELD, 0., 0.]), WP)
    state = load_state(args.audit_state, SimpleNamespace(plasma=plasma, dark=drive))
    before, d = state.ordinary, plasma.domain
    if np.any(before.B) or np.any(before.E[:, 1:]) or np.any(before.u[:, 1:]):
        raise ValueError('orbit audit requires pure-electric 1V motion')
    accepted, current = jax.jit(implicit_drive_step)(plasma, state, drive)
    after, current = accepted.ordinary, np.asarray(current[:, 0]) / (e * N * c)
    length, dt = d.length * WP / c, d.dt * WP
    mass, charge = np.asarray(plasma.per_particle[0]) / m, np.asarray(plasma.per_particle[1]) / e
    weights = np.asarray(before.w) / (N * d.length)
    force = args.amplitude * np.cos(float(before.time) * WP + dt / 2)
    reference = midpoint_orbits((np.asarray(before.E[:, 0]) + np.asarray(after.E[:, 0])) / (2 * FIELD),
                                np.asarray(before.x[:, 0]) * WP / c, np.asarray(before.u[:, 0]) / c,
                                charge, mass, length, dt, drive=force, weights=weights,
                                substeps=plasma.solver.substeps)
    reference_current = -length / d.cells * np.cumsum((reference['rho'] - reference['rho_initial']) / dt)
    reference_current += reference['mean_current'] - np.mean(reference_current)
    position_error = (np.asarray(after.x[:, 0]) * WP / c - reference['x'] + length / 2) % length - length / 2
    momentum = np.sum(mass * weights * np.asarray(after.u[:, 0] - before.u[:, 0]) / c)
    reference_momentum = np.sum(mass * weights * (reference['u'] - np.asarray(before.u[:, 0]) / c))
    result = dict(checkpoint_time_omega_p=float(before.time) * WP,
                  checkpoint_sha256=hashlib.sha256(args.audit_state.read_bytes()).hexdigest(),
                  reference_converged=reference['converged'], reference_iterations=reference['iterations'],
                  reference_residual_u=reference['residual_u'],
                  reference_residual_x_over_dx=reference['residual_x_over_dx'],
                  max_position_difference_over_c_wp=float(np.max(abs(position_error))),
                  max_momentum_per_mass_difference_over_c=float(np.max(abs(
                      np.asarray(after.u[:, 0]) / c - reference['u']))),
                  max_charge_difference_over_en=float(np.max(abs(
                      np.asarray(after.rho) / (e * N) - reference['rho']))),
                  max_current_difference_over_enc=float(np.max(abs(current - reference_current))),
                  reference_ampere_residual_over_fieldwp=float(np.max(abs(
                      (np.asarray(after.E[:, 0] - before.E[:, 0]) / FIELD) / dt + reference_current))),
                  momentum_increment_over_nmecL=float(momentum),
                  uniform_drive_impulse_over_nmecL=float(dt * np.sum(charge * weights) * force),
                  reference_momentum_increment_over_nmecL=float(reference_momentum),
                  momentum_disagreement_over_nmecL=float(momentum - reference_momentum),
                  scope='Frozen accepted-midpoint 1V orbit/deposit reference; differences include finite iteration '
                        'and native force evaluation roundoff. Not an internal parent Picard residual.')
    settings = dict(dt=args.dt, iterations=args.iterations, particle_substeps=plasma.solver.substeps,
                    particles_per_species=plasma.species[0].n,
                    cells=args.cells, parent_revision='83d327118163833f93e2588edcb5029241f6ba2a')
    save_run(args.output, 'implicit_orbit_audit', settings, result)
    print(json.dumps(result, indent=2))


def benchmark(args):
    from examples.dark_reservoir import array_fingerprint, save_compressed_state
    plasma, initial = load_plasma(args)
    drive = PrescribedDrive(1., jnp.array([args.amplitude * FIELD, 0., 0.]), WP)
    archive_model = SimpleNamespace(plasma=plasma, dark=drive)
    initial = archived_initial(args.initial_state, initial, archive_model)
    stride = max(1, round(.1 / args.dt))
    steps = stride * round(args.horizon / (stride * args.dt))
    jax.block_until_ready(initial)
    fingerprints = {key: array_fingerprint(np.asarray(getattr(initial.ordinary, key)))
                    for key in ("x", "u", "w", "E", "B", "rho", "time")}
    fingerprints.update(mass=array_fingerprint(plasma.per_particle[0]),
                        charge=array_fingerprint(plasma.per_particle[1]))
    loading_fingerprints = {species.name: {key: array_fingerprint(getattr(species, key))
                                           for key in ("x", "v")} for species in plasma.species}
    scale = N * m * c**2 * plasma.domain.length
    save_compressed_state(args.output / "initial_state.npz", initial, archive_model)
    start = perf_counter()
    executable = run_drive.lower(plasma, initial, drive, steps, stride).compile()
    compile_seconds = perf_counter() - start
    print(f"Compiled in {compile_seconds:.2f} s; {args.samples + 1} executions of {steps} steps", flush=True)
    timings, executions = [], []
    trace_keys = ("mean_E", "current", "nonzero_electric", "momentum", "mean", "rms")
    for _ in range(args.samples + 1):
        start = perf_counter()
        final, history, maxima = executable(plasma, initial, drive)
        jax.block_until_ready((final, history, maxima))
        timings.append(perf_counter() - start)
        print(f"Execution {len(timings)}/{args.samples + 1}: {timings[-1]:.3f} s", flush=True)
        # Transfers and scalar postprocessing are outside the synchronized timer.
        executions.append({key: np.asarray(history[key]).copy()
                           for key in ("t", *trace_keys)})
    save_compressed_state(args.output / "final_state.npz", final, archive_model)
    history = jax.tree.map(np.asarray, history)
    t, electric = history["t"] * WP, history["mean_E"] / FIELD
    oracle_nodes = 64 if args.paper_loading else max(64, args.nodes)
    reference = homogeneous(t, args.amplitude, temperature=args.temperature,
                            relativistic=not args.newtonian, nodes=oracle_nodes, rtol=2e-11)
    current = history["current"] / (epsilon_0 * FIELD * WP)
    oracle_current = reference["mean"][:, 1] - reference["mean"][:, 0]
    z, exact = electric + 1j * current, reference["mean_E"] + 1j * oracle_current
    phase_error = np.unwrap(np.angle(z)) - np.unwrap(np.angle(exact))
    phase_threshold = .01 * max(abs(exact))
    phase_resolved = (abs(exact) > phase_threshold) & (abs(z) > phase_threshold) & (t >= 2 * np.pi)
    if args.amplitude == 0:
        phase_resolved[:] = False
    for execution in executions:
        execution["t"] *= WP
        execution["mean_E"] /= FIELD
        execution["current"] /= epsilon_0 * FIELD * WP
        execution["nonzero_electric"] /= scale
        execution["momentum"] *= c / scale
        execution["mean"] /= c
        execution["rms"] /= c
        np.testing.assert_array_equal(execution["t"], t)
    variability = [{key: norm_errors(t, execution[key], executions[0][key])
                    for key in trace_keys}
                   for execution in executions[1:]]
    oracle_windows = dict(mean_E=norm_errors(t, electric, reference["mean_E"]),
                          current=norm_errors(t, current, oracle_current), phase={})
    for end, mask in windows(t):
        resolved = mask & phase_resolved
        oracle_windows["phase"][end] = dict(samples=int(resolved.sum()),
                                            max_error_rad=float(np.max(abs(phase_error[resolved])))
                                            if resolved.any() else None,
                                            max_nonzero_electric_over_nmc2L=float(np.max(
                                                history["nonzero_electric"][mask]) / scale))
    maxima = np.asarray(maxima)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result = dict(compile_s=compile_seconds, first_primal_s=timings[0], warm_primal_s=timings[1:],
                  compiler_temporary_bytes=executable.memory_analysis().temp_size_in_bytes,
                  peak_rss_bytes=int(rss if platform.system() == "Darwin" else rss * 1024),
                  host_load=os.getloadavg(),
                  device_kinds=[device.device_kind for device in jax.devices()],
                  max_balance_over_nmc2L=float(maxima[0] / scale),
                  max_particle_charge_change_over_enL=float(maxima[1] / (e * N * plasma.domain.length)),
                  max_grid_charge_change_over_enL=float(maxima[2] / (e * N * plasma.domain.length)),
                  max_continuity_over_enwp=float(maxima[3] / (e * N * WP)),
                  max_gauss_over_en_eps0=float(maxima[4] * epsilon_0 / (e * N)),
                  max_mean_ampere_over_fieldwp=float(maxima[5] / (FIELD * WP)),
                  max_momentum_over_nmecL=float(maxima[6] * c / scale),
                  max_mean_E_error=float(max(abs(electric - reference["mean_E"]))),
                  max_current_error=float(max(abs(current - oracle_current))),
                  max_phase_error_rad=(float(max(abs(phase_error[phase_resolved])))
                                       if phase_resolved.any() else None),
                  phase_definition="unwrapped arg(E+iJ) against oracle; exclude near-zero radius and startup",
                  phase_mask_samples=int(phase_resolved.sum()), phase_total_samples=len(t),
                  phase_radius_threshold_normalized=float(phase_threshold),
                  phase_mask="t>=2π and both |E+iJ| exceed 1% of oracle peak radius",
                  phase_unavailable_reason=("zero drive: equilibrium oracle has no resolved oscillatory phase"
                                            if args.amplitude == 0 else None),
                  max_sampled_nonzero_electric_over_nmc2L=float(max(history["nonzero_electric"]) / scale),
                  final_nonzero_electric_over_nmc2L=float(history["nonzero_electric"][-1] / scale),
                  nonzero_electric_convention="epsilon0 dx sum|E-mean(E)|²/2, all components; no clamp",
                  nonzero_threshold_crossings=[crossings(t, execution["nonzero_electric"])
                                               for execution in executions],
                  oracle_windows=oracle_windows, execution_variability=variability,
                  execution_comparison=("execution 0 against each later call to the same compiled callable and input; "
                                        "norms use fixed physical scales; recorded arrays/results use the final call"),
                  oracle_scope=("homogeneous momentum-distribution reference; late differences mix time integration, "
                                "spatial PIC dynamics and velocity loading once nonzero modes grow"),
                  max_work_error_over_nmc2L=float(max(abs(history["work"] / scale - reference["work"]))),
                  oracle_nodes=oracle_nodes, particle_substeps=plasma.solver.substeps,
                  claim="uniform prescribed-drive prototype; phase and work checks, no implicit Proca")
    if args.gradient_horizon:
        controls = jnp.array([args.amplitude, 1., 1., 0.])
        gradient_steps = round(args.gradient_horizon / args.dt)
        signal = partial(objective, steps=gradient_steps, cells=args.cells, nodes=args.nodes, rings=args.rings,
                         dtau=args.dt, iterations=args.iterations, temperature=args.temperature)
        start = perf_counter()
        gradient_executable = jax.jit(jax.value_and_grad(signal)).lower(controls).compile()
        gradient_compile = perf_counter() - start
        gradient_timings = []
        for _ in range(args.samples + 1):
            start = perf_counter()
            value, gradient = gradient_executable(controls)
            jax.block_until_ready((value, gradient))
            gradient_timings.append(perf_counter() - start)
        oracle = tangent_reference(np.array([0., gradient_steps * args.dt]), np.asarray(controls),
                                   temperature=args.temperature, nodes=args.nodes)
        primal = jax.jit(signal)
        finite = []
        for i in range(4):
            h = 1e-5 * max(abs(float(controls[i])), .1)
            perturbation = jnp.zeros(4).at[i].set(h)
            finite.append((float(primal(controls + perturbation)) - float(primal(controls - perturbation))) / (2 * h))
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        result.update(gradient_compile_s=gradient_compile, gradient_first_primal_s=gradient_timings[0],
                      gradient_warm_s=gradient_timings[1:], gradient_horizon=gradient_steps * args.dt,
                      objective=float(value), automatic_gradient=np.asarray(gradient).tolist(),
                      frechet_gradient=oracle["energy_gradient"].tolist(),
                      finite_difference_gradient=finite,
                      gradient_peak_rss_bytes=int(rss if platform.system() == "Darwin" else rss * 1024),
                      gradient_temporary_bytes=gradient_executable.memory_analysis().temp_size_in_bytes)
    with midnight():
        figure, axes = plt.subplots(2, 2, figsize=(10, 6), layout="constrained")
        early = t <= 20
        axes[0, 0].plot(t[early], electric[early], label="implicit PIC")
        axes[0, 0].plot(t[early], reference["mean_E"][early], "--", label="homogeneous oracle")
        axes[0, 0].set_ylabel("mean E / (mₑcωₚ/e)")
        axes[0, 0].legend()
        axes[0, 1].plot(t, electric - reference["mean_E"])
        axes[0, 1].set_ylabel("PIC − oracle mean E")
        axes[1, 0].plot(t, (history["balance"] - initial.initial_ordinary) / scale)
        axes[1, 0].set_ylabel("(ΔU − W) / (nmₑc²L)")
        axes[1, 1].plot(t[phase_resolved], phase_error[phase_resolved])
        axes[1, 1].set_ylabel("state-plane phase error [rad]")
        for axis in axes.ravel():
            axis.set_xlabel("ωₚt")
        settings = {key: value for key, value in vars(args).items() if key not in ("output", "initial_state")}
        for key in (("nodes", "rings") if args.paper_loading else ("particles",)):
            settings.pop(key)
        settings.update(parent_revision="83d327118163833f93e2588edcb5029241f6ba2a", precision="float64",
                        initial_state_source=("complete zero-time archive" if args.initial_state
                                              else "native initialization"),
                        loading=("paper_plasma: co-located spatial lattice, Gaussian velocities; "
                                 "exact mean and variance, seed 0" if args.paper_loading else
                                 "weighted Gauss-Hermite velocity rings; two mobile species"),
                        wp_rad_s=WP, mass_ratio=MASS_RATIO, particles_per_species=plasma.species[0].n,
                        length_over_c_wp=float(plasma.domain.length * WP / c),
                        steps=steps, store_every=stride, actual_horizon=float(t[-1]),
                        native_archives=["initial_state.npz", "final_state.npz"],
                        archive_final_execution=args.samples,
                        initial_fingerprints=fingerprints, position_time="native implicit integer time",
                        physical_loading_fingerprints=loading_fingerprints,
                        species_order=[species.name for species in plasma.species],
                        trace_scales=("E: me c wp/e; J: epsilon0 Eref wp; "
                                      "nonzero/kinetic/spread energy: n me c² L; momentum: n me c L; mean/rms: c"),
                        velocity_spread_energy_definition="0.5 (ms/me) (sigma_s/c)²; lab frame, each species' "
                                                          "global mean removed; not a relativistic temperature",
                        thermal_parameter_definition="Gaussian velocity sigma_e squared / c squared")
        arrays = dict(t=t, electric=electric, oracle_E=reference["mean_E"], phase_error=phase_error,
                      phase_resolved=phase_resolved, balance=(history["balance"] - initial.initial_ordinary) / scale,
                      work=history["work"] / scale, oracle_work=reference["work"], current=current,
                      nonzero_electric=history["nonzero_electric"] / scale,
                      momentum=history["momentum"] * c / scale,
                      mean=history["mean"] / c, rms=history["rms"] / c, kinetic=history["kinetic"] / scale,
                      velocity_spread_energy=.5 * np.array([1., MASS_RATIO]) * (history["rms"] / c)**2,
                      phase_radius=abs(z), oracle_phase_radius=abs(exact))
        arrays.update({f"execution_{index}_{key}": execution[key]
                       for index, execution in enumerate(executions)
                       for key in trace_keys})
        save_run(args.output, "implicit_drive", settings, result, figure, **arrays)
        np.savez_compressed(args.output / "data.npz", **arrays)
        plt.close(figure)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    args = SimpleNamespace(**{key: globals()[key] for key in (
        "cells", "nodes", "rings", "dt", "iterations", "substeps", "horizon", "amplitude", "temperature",
        "samples", "gradient_horizon", "newtonian", "paper_loading", "particles", "output")},
        initial_state=Path(initial_state) if initial_state is not None else None,
        audit_state=Path(audit_state) if audit_state is not None else None)
    finite = np.all(np.isfinite([dt, horizon, amplitude, temperature, gradient_horizon]))
    counts = (cells, nodes, rings, iterations, samples, substeps, particles)
    if (not finite or not all(isinstance(value, (int, np.integer)) for value in counts) or min(counts) < 1
            or dt <= 0 or dt > horizon or horizon < 2 * np.pi or temperature < 0 or gradient_horizon < 0
            or (gradient_horizon and gradient_horizon < dt)):
        raise ValueError("positive integer counts/time controls, nonnegative temperature and horizon >= 2π required")
    if paper_loading and (cells < 4 or particles < 2 or newtonian or temperature != .001 or gradient_horizon):
        raise ValueError("paper loading requires cells >= 4, particles >= 2, relativistic T=0.001 "
                         "and no gradient study")
    if not paper_loading and substeps != 2:
        raise ValueError("changing particle substeps requires paper_loading=True")
    if not paper_loading and not newtonian and 2 * temperature * max(abs(hermgauss(nodes)[0]))**2 >= 1 - 1e-5:
        raise ValueError("quadrature velocities exceed the parent's relativistic input margin")
    if newtonian and gradient_horizon:
        raise ValueError("gradient prototype uses the relativistic solver")
    print(f"Starting implicit drive benchmark: {cells} cells, dtωₚ={dt:g}, horizon={horizon:g}", flush=True)
    if args.audit_state is not None:
        if newtonian or gradient_horizon:
            raise ValueError("orbit audit requires relativistic dynamics without a gradient study")
        audit_orbits(args)
    else:
        with elapsed_progress("Implicit drive"):
            benchmark(args)
    print(f"Finished implicit drive benchmark; wrote {output}", flush=True)
