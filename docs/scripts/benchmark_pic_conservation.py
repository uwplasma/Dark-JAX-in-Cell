"""Coupled PIC conservation, phase/growth and cost on identical neutral loading.

The ordinary implicit rows call the pinned parent's discrete-gradient method.
The Proca row is explicit; a vacuum midpoint clock is not an implicit dark PIC.
"""

import argparse
import hashlib
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import linregress  # noqa: E402
from jaxincell import (Domain, Simulation, Solver, Species, elementary_charge as e,  # noqa: E402
                       epsilon_0, mass_electron as m, quiet_start, save_run, speed_of_light as c)  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from darkjaxincell import DarkField, DarkSimulation, midnight  # noqa: E402
from docs.scripts.conservation import measured_run  # noqa: E402
from examples.dark_kinetic import longitudinal_root  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
WP = 1e9


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


def plasma(cells, particles, dtau, iterations=0):
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
            x = x.at[:, 0].add(1e-4 * jnp.sin(k * x[:, 0]) / k)
        species.append(Species(name, particles, charge, mass, fraction * density, x=x, v=v))
        populations.append(dict(wp=WP * np.sqrt(fraction * m / mass), u=speed, vth=vth))
    solver = Solver(algorithm="implicit" if iterations else "explicit", picard_iterations=iterations or 8)
    return Simulation(Domain(length, cells, time_step=dtau / WP), tuple(species), solver), populations


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
    timings = []
    for _ in range(samples + 1):
        start = perf_counter()
        output = executable(sim, initial)
        jax.block_until_ready(output)
        timings.append(perf_counter() - start)
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    methods = ("explicit", "implicit1", "implicit2", "implicit4", "implicit8", "implicit12", "proca")
    parser.add_argument("--case", choices=methods)
    parser.add_argument("--overhead-case", choices=("production", "reduced"))
    parser.add_argument("--overhead", action="store_true")
    parser.add_argument("--render", action="store_true", help="refit and plot saved scalar arrays; no simulation")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--cells", type=int)
    parser.add_argument("--particles", type=int)
    parser.add_argument("--dt", type=float, default=.002)
    parser.add_argument("--horizon", type=float)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path("artifacts/pic_conservation"))
    args = parser.parse_args()
    if args.render:
        reanalyse(args.output)
        return
    cells = args.cells if args.cells is not None else (128 if args.full else 64)
    particles = args.particles if args.particles is not None else (8192 if args.full else 1024)
    horizon = args.horizon if args.horizon is not None else (40 if args.full else 20)
    if min(cells, particles, args.samples) < 1 or args.dt <= 0 or horizon < 16:
        parser.error("positive resolution/time step/samples and horizon >= 16 are required")
    if args.overhead_case:
        print(json.dumps(overhead(args.overhead_case, cells, particles, args.dt, args.samples)))
        return
    if args.overhead:
        rows = [json.loads(subprocess.check_output(
            [sys.executable, str(Path(__file__).resolve()), "--overhead-case", case,
             "--cells", str(cells), "--particles", str(particles), "--dt", str(args.dt),
             "--samples", str(args.samples)], text=True, cwd=ROOT)) for case in ("production", "reduced")]
        scale = abs(rows[0]["final_energy_J_m2"])
        for key in ("final_energy_J_m2", "all_step_balance_J_m2"):
            np.testing.assert_allclose(rows[0][key], rows[1][key], rtol=1e-13, atol=1e-14 * scale)
        args.output.mkdir(parents=True, exist_ok=True)
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, cwd=ROOT).strip()
        record = dict(source_revision=revision,
                      source_status=subprocess.check_output(["git", "describe", "--always", "--dirty"],
                                                            text=True, cwd=ROOT).strip(),
                      parent_revision="83d327118163833f93e2588edcb5029241f6ba2a", rows=rows)
        (args.output / "overhead.json").write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps(rows, indent=2))
        return
    if args.case:
        row, data = measure(args.case, cells, particles, args.dt, horizon, args.samples)
        print(json.dumps(dict(row=row, data=data)))
        return
    rows, histories = [], []
    cases = [(case, args.dt, cells) for case in methods]
    if args.full:
        cases += [(case, args.dt / 2, cells) for case in ("explicit", "implicit8", "proca")]
        cases += [("proca", args.dt, cells * 2)]
        cases += [("implicit8", .02, cells)]
    for case, dtau, grid in cases:
        print(f"measuring {case}, {grid} cells, Δtωₚ={dtau:g}", flush=True)
        output = subprocess.check_output([sys.executable, str(Path(__file__).resolve()), "--case", case,
                                          "--cells", str(grid), "--particles", str(particles), "--dt", str(dtau),
                                          "--horizon", str(horizon), "--samples", str(args.samples)],
                                         text=True, cwd=ROOT)
        result = json.loads(output)
        rows.append(result["row"])
        histories.append(result["data"])
    render(args.output, dict(preset="full" if args.full else "quick", particles_per_species=particles,
                             thermal_sigma_over_c=.003, electron_drift_over_c=.05, mass_ratio=1836,
                             ku_over_omega_p=.5, perturbation_density_fraction=1e-4,
                             parent_revision="83d327118163833f93e2588edcb5029241f6ba2a",
                             samples_per_case=args.samples, precision="float64", note="fresh process per row"),
           rows, histories)


if __name__ == "__main__":
    main()
