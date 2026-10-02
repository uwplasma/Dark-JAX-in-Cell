"""Coupled PIC conservation, phase/growth and cost on identical neutral loading.

The ordinary implicit rows call the pinned parent's discrete-gradient method.
The Proca row is explicit; a vacuum midpoint clock is not an implicit dark PIC.
"""

import hashlib
import json
import os
import platform
import resource
import subprocess
import sys
from math import comb, factorial
from pathlib import Path
from statistics import median
from time import perf_counter

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import linregress  # noqa: E402
from jaxincell import (Domain, Simulation, Solver, Species, elementary_charge as e,  # noqa: E402
                       epsilon_0, mass_electron as m, quiet_start, save_run, save_state,
                       speed_of_light as c)  # noqa: E402
from jaxincell._core import deposit, E_x_from_rho  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from darkjaxincell import DarkField, DarkSimulation, midnight  # noqa: E402
from docs.scripts.conservation import elapsed_progress, measured_run, snapshot  # noqa: E402
from docs.scripts.drive_reference import _quadratic_charge, midpoint_orbits  # noqa: E402
from examples.dark_kinetic import longitudinal_root  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
WP = 1e9
METHODS = ("explicit", "implicit1", "implicit2", "implicit4", "implicit8", "implicit12", "proca")

# Editable inputs also accept runpy.run_path(..., init_globals={...}) batch overrides.
full = globals().get("full", False)
case = globals().get("case", None if full else "explicit")
overhead_case = globals().get("overhead_case", None)
compare_overhead = globals().get("compare_overhead", False)
render_only = globals().get("render_only", False)
translations = globals().get("translations", False)  # short fractional-cell force audit
translation_meshes = globals().get("translation_meshes", (16, 32, 64))
symbols = globals().get("symbols", False)  # independent smooth-mode operator quadrature
symbol_cells = globals().get("symbol_cells", 32)
symbol_modes = globals().get("symbol_modes", (0, 1, 3, 6))
cells = globals().get("cells", 128 if full else 16)
particles = globals().get("particles", 8192 if full else 64)
dt = globals().get("dt", .002 if full else .01)
horizon = globals().get("horizon", 40. if full else 16.)
samples = globals().get("samples", 1)
output = Path(globals().get("output", "artifacts/smooth_force_symbols" if symbols else "artifacts/pic_conservation"))


def cardinal_spline(coordinate, degree, derivative=False):
    """Independent centred cardinal spline, without a production shape kernel."""
    inside = abs(coordinate) < (degree + 1) / 2
    shifted = np.where(inside, coordinate + (degree + 1) / 2, 0.)
    power = degree - int(derivative)
    value = sum((-1)**j * comb(degree + 1, j) * np.maximum(shifted - j, 0.)**power
                for j in range(degree + 2)) / factorial(power)
    return np.where(inside, value, 0.)


def smooth_force_symbols(cells=32, modes=(0, 1, 3, 6)):
    """Fundamental MC/EC force coefficients from continuous particle quadrature.

    MC averages face E to centres and gathers its charge spline. EC differentiates
    the transpose of the zero-mean continuity current and explicitly restores
    mean E. Their separated nonzero fundamental ratio is sin(k dx)/(k dx).
    Aliases remain in the full force arrays; this is not a pointwise identity.
    """
    if not isinstance(cells, (int, np.integer)) or not 8 <= cells <= 64:
        raise ValueError("symbol quadrature requires 8 to 64 cells")
    modes = tuple(modes)
    if (not 1 <= len(modes) <= 8
            or any(not isinstance(k, (int, np.integer)) or not 0 <= k < cells / 2 for k in modes)
            or len(set(modes)) != len(modes) or 0 not in modes):
        raise ValueError("symbol modes must include zero and be distinct integers below Nyquist")
    centre, face = np.arange(cells), np.arange(cells) + .5  # dx=1, L=cells
    div = np.eye(cells) - np.roll(np.eye(cells), 1, axis=0)
    current = -np.linalg.pinv(div)
    arrays, rows = dict(centres=centre, faces=face, divergence=div, continuity=current), []
    for order in (4, 8):
        nodes, weights = np.polynomial.legendre.leggauss(order)
        # Half-cell pieces contain every S2/S5 breakpoint; chunks bound cloud matrices.
        x = (np.arange(2 * cells)[:, None] / 2 + (nodes[None, :] + 1) / 4).ravel()
        quadrature = np.tile(weights / (4 * cells), 2 * cells)
        arrays.update({f"q{order}_x": x, f"q{order}_weights": quadrature})
        for degree in (2, 5):
            cloud, slope = [], []
            for start in range(0, len(x), 256):
                offset = (x[start:start + 256, None] - centre + cells / 2) % cells - cells / 2
                cloud.append(cardinal_spline(offset, degree))
                slope.append(cardinal_spline(offset, degree, True))
            cloud, slope = np.concatenate(cloud), np.concatenate(slope)
            prefix = f"q{order}_s{degree}"
            arrays.update({prefix + "_shape": cloud, prefix + "_shape_derivative": slope})
            print(f"Smooth force: S{degree}, {cells} cells, Gauss{order}, {len(x)} points", flush=True)
            for mode in modes:
                theta = 2 * np.pi * mode / cells
                electric = np.exp(1j * theta * face)
                potential = current.T @ electric
                force_mc = cloud @ ((electric + np.roll(electric, 1)) / 2)
                force_ec = slope @ potential + np.mean(electric)
                projection = quadrature * np.exp(-1j * theta * x)
                mc, ec = np.sum(projection * force_mc), np.sum(projection * force_ec)
                ratio, shape = mc / ec, np.sinc(theta / (2 * np.pi))**(degree + 1)
                predicted_mc, predicted_ec = np.cos(theta / 2) * shape, shape / np.sinc(theta / (2 * np.pi))
                label = prefix + f"_k{mode}"
                arrays.update({label + "_electric": electric, label + "_potential": potential,
                               label + "_MC": force_mc, label + "_EC": force_ec,
                               label + "_coefficients": np.array([mc, ec, ratio])})
                rows.append(dict(label=label, cells=cells, degree=degree, quadrature_order=order,
                                 quadrature_points=len(x), mode=mode, theta=theta,
                                 MC=[float(mc.real), float(mc.imag)], EC=[float(ec.real), float(ec.imag)],
                                 ratio=[float(ratio.real), float(ratio.imag)],
                                 predicted_MC=float(predicted_mc), predicted_EC=float(predicted_ec),
                                 predicted_ratio=float(np.sinc(theta / np.pi)),
                                 MC_error=float(abs(mc - predicted_mc)), EC_error=float(abs(ec - predicted_ec)),
                                 ratio_error=float(abs(ratio - np.sinc(theta / np.pi)))))
    return rows, arrays


def symbol_audit(folder, cells=32, modes=(0, 1, 3, 6)):
    """Save independent measured force arrays, white figure and native provenance."""
    rows, arrays = smooth_force_symbols(cells, modes)
    projector = np.eye(cells) - np.ones((cells, cells)) / cells
    continuity_error = float(np.max(abs(arrays["divergence"] @ arrays["continuity"] + projector)))
    finest = [row for row in rows if row["quadrature_order"] == 8]
    ratio_error = max(row["ratio_error"] for row in finest)
    mean_error = max(float(np.max(abs(arrays[row["label"] + "_" + method] - 1)))
                     for row in rows if row["mode"] == 0 for method in ("MC", "EC"))
    if (any(not np.all(np.isfinite(value)) for value in arrays.values())
            or continuity_error > 1e-13 or ratio_error > 1e-12 or mean_error > 1e-12):
        raise ValueError("smooth-symbol continuity, fundamental or restored-mean check failed")
    with midnight():
        figure, axes = plt.subplots(1, 2, figsize=(10, 3.8), layout="constrained")
        theta = np.linspace(0, max(row["theta"] for row in rows), 256)
        arrays["curve_theta"], arrays["curve_ratio"] = theta, np.sinc(theta / np.pi)
        for degree, colour in ((2, "#6A3D9A"), (5, "#0072B2")):
            shape = np.sinc(theta / (2 * np.pi))**(degree + 1)
            for method, marker, line in (("MC", "o", "-"), ("EC", "^", "--")):
                expected = np.cos(theta / 2) * shape if method == "MC" else shape / np.sinc(theta / (2 * np.pi))
                arrays[f"curve_s{degree}_{method}"] = expected
                axes[0].plot(theta, expected, line, color=colour, label=f"S{degree} {method} prediction")
                selected = [row for row in finest if row["degree"] == degree]
                axes[0].scatter([row["theta"] for row in selected], [row[method][0] for row in selected],
                                marker=marker, color=colour, facecolors="white", zorder=3)
            axes[1].scatter([row["theta"] for row in selected], [row["ratio"][0] for row in selected],
                            marker="o" if degree == 2 else "s", color=colour, facecolors="white",
                            s=80 if degree == 2 else 30, label=f"S{degree} Gauss8", zorder=3)
        axes[1].plot(theta, arrays["curve_ratio"], color="#30343B", label=r"$\sin\theta/\theta$")
        axes[0].set(title="Fundamental force (points: Gauss8)", ylabel=r"$F_0/E_{\mathrm{face},0}$")
        axes[1].set(title="MC / EC fundamental ratio", ylabel=r"$F_{\mathrm{MC},0}/F_{\mathrm{EC},0}$")
        for axis in axes:
            axis.set_xlabel(r"$\theta=k\Delta x$")
            axis.grid(alpha=.3)
            axis.legend(fontsize=8)
        runtime = dict(python=platform.python_version(), numpy=np.__version__, matplotlib=matplotlib.__version__,
                       platform=f"{platform.system()} {platform.machine()}", dtype="float64",
                       numpy_linalg_sha256=hashlib.sha256(Path(
                           sys.modules["numpy.linalg._umath_linalg"].__file__).read_bytes()).hexdigest())
        settings = dict(cells=cells, dx=1., modes=list(modes), degrees=[2, 5], quadrature_orders=[4, 8],
                        scope="Continuous-particle fundamental operators; no PIC dynamics or late-error attribution",
                        computation="NumPy; JAX version/backend in save_run describe the imported parent runtime",
                        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), runtime=runtime,
                        runtime_sha256=hashlib.sha256(json.dumps(runtime, sort_keys=True).encode()).hexdigest())
        save_run(folder, "smooth_force_symbols", settings, dict(rows=rows, continuity_error=continuity_error,
                 gauss8_ratio_error=ratio_error, restored_mean_error=mean_error,
                 oracle="Independent periodic divergence transpose and continuous cardinal-spline quadrature"),
                 figure, **arrays)
        np.savez_compressed(Path(folder) / "data.npz", **arrays)
        path = Path(folder) / "run.json"
        record = json.loads(path.read_text())
        record["results"]["arrays_sha256"] = hashlib.sha256((Path(folder) / "data.npz").read_bytes()).hexdigest()
        path.write_text(json.dumps(record, indent=1, allow_nan=False) + "\n")
        plt.close(figure)


def fit_mode(t, mode, window=(8., 16.)):
    """Fit a shared inclusive window despite floating accumulation of the clock."""
    t, mode = np.asarray(t), np.asarray(mode)
    selected = (t >= window[0] - 1e-9) & (t <= window[1] + 1e-9)
    if selected.sum() < 3:
        raise ValueError("mode fit needs at least three samples in its physical window")
    growth = linregress(t[selected], np.log(np.maximum(abs(mode[selected]), np.finfo(float).tiny)))
    angles = np.unwrap(np.angle(mode[selected]))
    phase = linregress(t[selected], angles)
    return dict(fitted_growth_over_wp=float(growth.slope), regression_standard_error=float(growth.stderr),
                fitted_frequency_over_wp=float(-phase.slope), fit_samples=int(selected.sum()),
                phase_regression_standard_error=float(phase.stderr),
                phase_excursion_radians=float(np.max(abs(angles - angles[0]))),
                fit_window_omega_p=list(window))


def plasma(cells, particles, dtau, iterations=0, displacement=1e-4):
    """Warm opposed electrons and mobile ions; all rows share physical particles."""
    density = epsilon_0 * m * WP**2 / e**2
    drift, sigma = .05 * c, .003 * c
    length = 4 * np.pi * drift / WP
    k = 2 * np.pi / length
    species, populations = [], []
    loadings = (("right", -1, m, .5, drift), ("left", -1, m, .5, -drift), ("ions", 1, 1836 * m, 1., 0.))
    for name, charge, mass, fraction, speed in loadings:
        vth = np.sqrt(2 * m / mass) * sigma
        x, v = quiet_start(particles, length, (vth, 0., 0.), (speed, 0., 0.))
        v = v.at[:, 0].add(speed - jnp.mean(v[:, 0]))
        if name == "right":
            x = x.at[:, 0].add(displacement * jnp.sin(k * x[:, 0]) / k)
        species.append(Species(name, particles, charge, mass, fraction * density, x=x, v=v))
        populations.append(dict(wp=WP * np.sqrt(fraction * m / mass), u=speed, vth=vth))
    solver = Solver(algorithm="implicit" if iterations else "explicit", picard_iterations=iterations or 8)
    return Simulation(Domain(length, cells, time_step=dtau / WP), tuple(species), solver), populations


def translated_step(cells, particles, dtau, iterations, fraction):
    """One neutral 1V step, with independent force/orbit reconstruction in wp,c,me units."""
    p, _ = plasma(cells, particles, dtau, iterations, displacement=.03)
    p = p.replace(solver=p.solver.replace(relativistic=True))
    initial, (mass, charge) = p.initial_state(jax.random.PRNGKey(0))
    d, n = p.domain, 2 * p.species[0].density
    physical = initial.x if iterations else initial.x - d.dt * p._velocity(initial.u) / 2
    physical = physical.at[:, 0].set((physical[:, 0] + fraction * d.dx + d.length / 2) % d.length - d.length / 2)
    rho = deposit(physical[:, 0], charge * initial.w, d.grid[0], d.dx, cells, (0, 0))
    carried = physical if iterations else physical + d.dt * p._velocity(initial.u) / 2
    carried = carried.at[:, 0].set((carried[:, 0] + d.length / 2) % d.length - d.length / 2)
    initial = initial.replace(x=carried, rho=rho, E=initial.E.at[:, 0].set(E_x_from_rho(rho, d.dx, (0, 0))))
    final, _, maxima = measured_run(p, initial, 1, 1)
    jax.block_until_ready(final)
    weights, q, masses = (np.asarray(value) for value in (initial.w / (n * d.length), charge / e, mass / m))
    x, u = np.asarray(physical[:, 0]) * WP / c, np.asarray(initial.u[:, 0]) / c
    accepted = np.asarray(final.u[:, 0]) / c - u
    impulse = float(np.sum(masses * weights * accepted))
    if iterations:
        field = np.asarray((initial.E[:, 0] + final.E[:, 0]) / 2) * e / (m * c * WP)
        reference = midpoint_orbits(field, x, u, q, masses, d.length * WP / c, dtau, weights=weights,
                                    substeps=p.solver.substeps, tolerance=2e-14)
        if not reference['converged']:
            raise ValueError('independent midpoint orbits did not converge')
        reference_impulse = float(np.sum(masses * weights * (reference['u'] - u)))
        orbit_error = float(np.max(abs(reference['u'] - np.asarray(final.u[:, 0]) / c)))
        closure = [max(reference['residual_u']), max(reference['residual_x_over_dx'])]
        grid_impulse = None
    else:
        half_rho = _quadratic_charge(x + dtau * u / np.hypot(1., u) / 2, q * weights, d.length * WP / c, cells)
        face = np.cumsum(half_rho - half_rho.mean()) * d.dx * WP / c
        face -= face.mean() + dtau / 2 * np.sum(q * weights * u / np.hypot(1., u))
        # Charge-transpose gather and periodic Gauss give a telescoping summed force.
        centred = (face + np.roll(face, 1)) / 2
        grid_impulse = float(dtau * d.dx / d.length * np.sum(half_rho * centred))
        coordinate = (x + dtau * u / np.hypot(1., u) / 2 + d.length * WP / (2 * c)) * cells / (d.length * WP / c) - .5
        indices = np.floor(coordinate + .5).astype(int)[:, None] + np.array([-1, 0, 1])
        distance = abs(coordinate[:, None] - indices)
        shape = np.where(distance <= .5, .75 - distance**2, .5 * np.maximum(1.5 - distance, 0)**2)
        predicted = dtau * q / masses * np.sum(shape * centred[indices % cells], axis=1)
        reference_impulse = float(np.sum(masses * weights * predicted))
        orbit_error = float(np.max(abs(predicted - accepted)))
        closure = None
    row = dict(cells=cells, particles_per_species=particles, dt_omega_p=dtau,
               method='implicit8' if iterations else 'explicit', shift_over_dx=fraction,
               impulse_over_nmecL=impulse, force_over_nmecLwp=impulse / dtau,
               reference_impulse_over_nmecL=reference_impulse,
               impulse_reference_error=abs(impulse - reference_impulse), max_orbit_error_over_c=orbit_error,
               reference_orbit_closure=closure, explicit_grid_impulse_over_nmecL=grid_impulse,
               energy_defect_over_initial=float(maxima[0] / snapshot(p, initial)['balance']),
               continuity_over_enwp=float(maxima[3] / (e * n * WP)),
               gauss_over_en_eps0=float(maxima[4] * epsilon_0 / (e * n)),
               charge_change_over_enL=float(maxima[2] / (e * n * d.length)))
    return row, p, initial, final


def translation_audit(folder, meshes, particles, dtau):
    """Keep complete native endpoints privately and publish small numerical ledgers."""
    folder.mkdir(parents=True, exist_ok=True)
    rows = []
    for grid in meshes:
        for step in (dtau, dtau / 2):
            for iterations in (0, 8):
                for fraction in (0., .25, .5):
                    print(f"Force audit: {grid} cells, dtωp={step:g}, K={iterations}, shift={fraction:g}dx", flush=True)
                    row, p, initial, final = translated_step(grid, particles, step, iterations, fraction)
                    label = f'{grid}_{step:g}_{iterations}_{fraction:g}'
                    save_state(folder / f'{label}_initial.npz', initial, p)
                    save_state(folder / f'{label}_final.npz', final, p)
                    rows.append(row)
    save_run(folder, 'fractional_cell_force', dict(meshes=list(meshes), particles_per_species=particles,
             displacement_over_1_k=.03, shifts_over_dx=[0., .25, .5], dt_omega_p=[dtau, dtau / 2],
             parent_revision='83d327118163833f93e2588edcb5029241f6ba2a',
             normalization='n is total electron density; force=delta P/(dt*n*me*c*L*wp)',
             relativistic=True,
             scope='Neutral ordinary 1V, one accepted step; not a late Gaussian or Proca convergence test'),
             dict(rows=rows))
    print(f"Saved force audit and native endpoints to {folder}", flush=True)


def measure(case, cells, particles, dtau, horizon, samples):
    """Compile separately; synchronize primal execution and report executable memory."""
    if min(cells, particles, samples) < 1 or dtau <= 0 or horizon < 16:
        raise ValueError("positive resolution/time step/samples and horizon >= 16 are required")
    iterations = int(case.removeprefix("implicit")) if case.startswith("implicit") else 0
    p, populations = plasma(cells, particles, dtau, iterations)
    sim = DarkSimulation(p, DarkField(.7 * WP, .3)) if case == "proca" else p
    initial, _ = sim.initial_state(jax.random.PRNGKey(0))
    jax.block_until_ready(initial)
    stride = max(1, round(.1 / dtau))
    steps = stride * round(horizon / (stride * dtau))
    if steps < stride:
        raise ValueError("time step leaves no output interval")
    start = perf_counter()
    executable = measured_run.lower(sim, initial, steps, stride).compile()
    compile_seconds = perf_counter() - start
    print(f"Compiled {case} in {compile_seconds:.3f} s", file=sys.stderr, flush=True)
    start = perf_counter()
    final, history, maxima = executable(sim, initial)
    jax.block_until_ready((final, history, maxima))
    first_seconds = perf_counter() - start
    warm = []
    for _ in range(samples):
        start = perf_counter()
        final, history, maxima = executable(sim, initial)
        jax.block_until_ready((final, history, maxima))
        warm.append(perf_counter() - start)
        print(f"{case}: warm execution {len(warm)}/{samples} in {warm[-1]:.3f} s", file=sys.stderr, flush=True)
    history = jax.tree.map(np.asarray, history)
    scale = p.species[0].density * 2 * m * c**2 * p.domain.length
    charge_scale = 2 * e * p.species[0].density * p.domain.length
    t, mode = history["t"] * WP, history["mode_E"]
    root, residual = longitudinal_root(2 * np.pi / p.domain.length, populations, .7 * WP,
                                       .3 if case == "proca" else 0, .34j,
                                       model="full" if case == "proca" else "ordinary")
    root_scale = np.sqrt(sum(s["wp"]**2 for s in populations)) / WP
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    memory = executable.memory_analysis()
    row = dict(case=case, cells=cells, particles_per_species=particles, dt_omega_p=dtau,
               horizon_omega_p=float(t[-1]), picard_iterations=iterations,
               particle_substeps=p.solver.substeps if iterations else 1,
               compile_s=compile_seconds, first_primal_s=first_seconds, warm_primal_s=warm,
               warm_median_s=median(warm), peak_rss_bytes=int(rss if platform.system() == "Darwin" else rss * 1024),
               compiler_temporary_bytes=memory.temp_size_in_bytes, host_load=os.getloadavg(),
               max_energy_error_over_initial=float(maxima[0] / history["balance"][0]),
               max_momentum_error_over_nmecL=float(maxima[1] * c / scale),
               max_particle_charge_change_over_enL=float(maxima[2] / charge_scale),
               max_continuity_over_enwp=float(maxima[3] * p.domain.length / (charge_scale * WP)),
               max_ordinary_gauss_over_en_eps0=float(maxima[4] * epsilon_0 * p.domain.length / charge_scale),
               max_dark_gauss_over_en_eps0=float(maxima[5] * epsilon_0 * p.domain.length / charge_scale),
               max_grid_charge_change_over_enL=float(maxima[6] / charge_scale),
               max_dark_work_error_over_initial=float(maxima[7] / history["balance"][0]),
               max_ordinary_work_error_over_initial=float(maxima[8] / history["balance"][0]),
               max_grid_particle_charge_disagreement_over_enL=float(np.max(abs(
                   history["grid_charge"] - history["charge"])) / charge_scale),
               **fit_mode(t, mode),
               theory_growth_over_wp=float(root.imag * root_scale),
               theory_frequency_over_wp=float(root.real * root_scale), theory_residual=residual,
               reference_frequency_over_electron_wp=float(root_scale),
               backend=jax.default_backend(),
               devices=[str(d) for d in jax.devices()], device_kinds=[d.device_kind for d in jax.devices()],
               jax=jax.__version__)
    energy_error = (history["balance"] - history["balance"][0]) / history["balance"][0]
    arrays = dict(t=t.tolist(), energy_error=energy_error.tolist(),
                  momentum_error=((history["momentum"][:, 0] - history["momentum"][0, 0]) * c / scale).tolist(),
                  mode_real=(mode.real * e / (m * c * WP)).tolist(),
                  mode_imag=(mode.imag * e / (m * c * WP)).tolist())
    return row, arrays


def figure_rows(rows, histories):
    """Show selected method/refinement rows; retain every row in the evidence."""
    with midnight():
        figure, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
        axes = axes.ravel()
        handles = []
        base_dt = rows[0]["dt_omega_p"]
        for row, data in zip(rows, histories):
            if not ((row["case"] in ("explicit", "implicit4", "proca") and row["dt_omega_p"] == base_dt)
                    or (row["case"] == "implicit8" and row["dt_omega_p"] == .02)):
                continue
            method = "Proca explicit" if row["case"] == "proca" else row["case"].replace("implicit", "implicit ×")
            label = f'{method}, Δtωₚ={row["dt_omega_p"]:g}, {row["cells"]} cells'
            t = np.asarray(data["t"])
            mode = np.asarray(data["mode_real"]) + 1j * np.asarray(data["mode_imag"])
            handles += axes[0].semilogy(t, np.maximum(abs(np.asarray(data["energy_error"])), 1e-17), label=label)
            colour = handles[-1].get_color()
            axes[1].semilogy(t, np.maximum(abs(np.asarray(data["momentum_error"])), 1e-19), color=colour)
            axes[2].semilogy(t, abs(mode), color=colour)
            selected = (t >= 8 - 1e-9) & (t <= 16 + 1e-9)
            phase = np.unwrap(np.angle(mode[selected]))
            axes[3].plot(t[selected], phase - phase[0], color=colour)
            if row["case"] in ("explicit", "proca") and row["cells"] == 128:
                theory = abs(mode[selected][0]) * np.exp(row["theory_growth_over_wp"] * (t[selected] - t[selected][0]))
                axes[2].plot(t[selected], theory, "--", color=colour, linewidth=1)
        for axis in axes:
            axis.set_xlabel("ωₚt")
            axis.grid(alpha=.4)
        titles = ("Complete energy", "Continuum momentum", "First electric mode", "Mode phase in fit window")
        for axis, title in zip(axes, titles):
            axis.set_title(title)
        axes[0].set_ylabel("|ΔU| / U(0)")
        axes[1].set_ylabel("|ΔPₓ| / (nmₑcL)")
        axes[2].set_ylabel("|E₁| / (mₑcωₚ/e)")
        axes[3].set_ylabel("arg E₁(t) − arg E₁(8/ωₚ) [rad]")
        axes[3].axhline(0, color="#30343B", linestyle="--", linewidth=1)
        figure.legend(handles=handles, loc="outside upper center", ncol=2, fontsize=9)
        return figure


def render(folder, settings, rows, histories):
    """Save the computation provenance and compressed scalar histories."""
    figure = figure_rows(rows, histories)
    arrays = {f'{i}_{key}': np.asarray(value) for i, data in enumerate(histories) for key, value in data.items()}
    save_run(folder, "pic_conservation", settings, dict(rows=rows,
             claim="method convergence comparison; implicit Proca not implemented"), figure, **arrays)
    np.savez_compressed(Path(folder) / "data.npz", **arrays)
    plt.close(figure)


def reanalyse(folder):
    """Refit stored modes without altering their computation or timing provenance."""
    record = json.loads((folder / "run.json").read_text())
    rows = record["results"]["rows"]
    with np.load(folder / "data.npz") as arrays:
        histories = [{key.split("_", 1)[1]: arrays[key] for key in arrays.files if key.startswith(f"{i}_")}
                     for i in range(len(rows))]
    for row, data in zip(rows, histories):
        mode = data["mode_real"] + 1j * data["mode_imag"]
        row.setdefault("original_fit", {key: row[key] for key in (
            "fitted_growth_over_wp", "regression_standard_error", "fitted_frequency_over_wp", "fit_window_omega_p")})
        row.update(fit_mode(data["t"], mode))
    record["analysis"] = dict(method="inclusive first-mode log-amplitude and unwrapped-phase regression",
                              endpoint_tolerance_omega_p=1e-9,
                              script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                              arrays_sha256=hashlib.sha256((folder / "data.npz").read_bytes()).hexdigest(),
                              note="stored arrays reused; original fits and computation provenance retained")
    figure = figure_rows(rows, histories)
    figure.savefig(folder / "figure.png", dpi=160, bbox_inches="tight")
    plt.close(figure)
    (folder / "run.json").write_text(json.dumps(record, indent=1) + "\n")


def overhead(case, cells, particles, dtau, samples):
    """Real reduced scalar output against production ``store_particles=False``."""
    if min(cells, particles, samples) < 1 or dtau <= 0:
        raise ValueError("positive resolution/time step/samples are required")
    p, _ = plasma(cells, particles, dtau)
    sim = DarkSimulation(p, DarkField(.7 * WP, .3))
    initial, _ = sim.initial_state(jax.random.PRNGKey(0))
    jax.block_until_ready(initial)
    production = jax.jit(lambda s, i: s.run(256, state=i, store_every=8, store_particles=False))
    start = perf_counter()
    executable = (production.lower(sim, initial).compile() if case == "production"
                  else measured_run.lower(sim, initial, 256, 8).compile())
    compile_seconds = perf_counter() - start
    print(f"Compiled {case} in {compile_seconds:.3f} s", file=sys.stderr, flush=True)
    timings = []
    for _ in range(samples + 1):
        start = perf_counter()
        output = executable(sim, initial)
        jax.block_until_ready(output)
        timings.append(perf_counter() - start)
        print(f"{case}: execution {len(timings)}/{samples + 1} in {timings[-1]:.3f} s",
              file=sys.stderr, flush=True)
    if case == "production":
        final, balance_error = output.energy()["total_with_dark"][-1], output.state.max_balance_error
    else:
        final, balance_error = output[1]["balance"][-1], output[2][0]
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return dict(case=case, cells=cells, particles_per_species=particles, steps=256, stride=8,
                compile_s=compile_seconds, first_primal_s=timings[0], warm_primal_s=timings[1:],
                warm_median_s=median(timings[1:]),
                compiler_temporary_bytes=executable.memory_analysis().temp_size_in_bytes,
                peak_rss_bytes=int(rss if platform.system() == "Darwin" else rss * 1024),
                final_energy_J_m2=float(final), all_step_balance_J_m2=float(balance_error),
                backend=jax.default_backend(), jax=jax.__version__, host_load=os.getloadavg())


def isolated(inputs):
    """Keep compiler/runtime memory and timings in a fresh process per row."""
    command = (f"import runpy; runpy.run_path({str(Path(__file__).resolve())!r}, "
               f"run_name='__main__', init_globals={inputs!r})")
    return json.loads(subprocess.check_output([sys.executable, "-c", command], text=True, cwd=ROOT))


def validate_inputs():
    """Reject invalid edited resolutions and method names before compilation."""
    if (min(cells, particles, samples) < 1 or not np.isfinite(dt) or dt <= 0
            or not np.isfinite(horizon) or horizon < 16):
        raise ValueError("positive resolution/time step/samples and horizon >= 16 are required")
    if case not in (None, *METHODS) or overhead_case not in (None, "production", "reduced"):
        raise ValueError("unknown method or overhead case")


if __name__ == "__main__":  # noqa: C901 — independent numerical studies
    print("Starting smooth-mode operator benchmark" if symbols else "Starting coupled PIC method benchmark",
          file=sys.stderr, flush=True)
    if symbols:
        symbol_audit(output, symbol_cells, symbol_modes)
    elif translations:
        translation_audit(output, translation_meshes, particles, dt)
    elif render_only:
        reanalyse(output)
    else:
        validate_inputs()
        if overhead_case:
            print(json.dumps(overhead(overhead_case, cells, particles, dt, samples)))
        elif compare_overhead:
            rows = [isolated(dict(overhead_case=name, cells=cells, particles=particles, dt=dt, samples=samples))
                    for name in ("production", "reduced")]
            scale = abs(rows[0]["final_energy_J_m2"])
            for key in ("final_energy_J_m2", "all_step_balance_J_m2"):
                np.testing.assert_allclose(rows[0][key], rows[1][key], rtol=1e-13, atol=1e-14 * scale)
            output.mkdir(parents=True, exist_ok=True)
            revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, cwd=ROOT).strip()
            record = dict(source_revision=revision,
                          source_status=subprocess.check_output(["git", "describe", "--always", "--dirty"],
                                                                text=True, cwd=ROOT).strip(),
                          parent_revision="83d327118163833f93e2588edcb5029241f6ba2a", rows=rows)
            (output / "overhead.json").write_text(json.dumps(record, indent=2) + "\n")
            print(json.dumps(rows, indent=2))
        elif case:
            with elapsed_progress(f"PIC {case}"):
                row, data = measure(case, cells, particles, dt, horizon, samples)
            print(json.dumps(dict(row=row, data=data)))
        else:
            rows, histories = [], []
            cases = [(name, dt, cells) for name in METHODS]
            if full:
                cases += [(name, dt / 2, cells) for name in ("explicit", "implicit8", "proca")]
                cases += [("proca", dt, cells * 2), ("implicit8", .02, cells)]
            for name, dtau, grid in cases:
                print(f"Measuring {name}, {grid} cells, Δtωₚ={dtau:g}", file=sys.stderr, flush=True)
                result = isolated(dict(case=name, cells=grid, particles=particles, dt=dtau,
                                       horizon=horizon, samples=samples))
                rows.append(result["row"])
                histories.append(result["data"])
            render(output, dict(preset="full" if full else "quick", particles_per_species=particles,
                                thermal_sigma_over_c=.003, electron_drift_over_c=.05, mass_ratio=1836,
                                ku_over_omega_p=.5, perturbation_density_fraction=1e-4,
                                parent_revision="83d327118163833f93e2588edcb5029241f6ba2a",
                                samples_per_case=samples, precision="float64", note="fresh process per row"),
                   rows, histories)
    print("Finished smooth-mode operator benchmark" if symbols else "Finished coupled PIC method benchmark",
          file=sys.stderr, flush=True)
