"""A matched bump-on-tail experiment with a self-consistent Proca field.

The Maxwellian beam follows the parent bump-on-tail example. A small bulk
counterdrift makes the loaded plasma current-neutral; both runs use exactly
the same particle arrays, ordinary field, grid and step.
"""

from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress

from jaxincell import (Domain, Simulation, Species, elementary_charge as e,
                       energies, epsilon_0, mass_electron, quiet_start, save_run,
                       speed_of_light as c)
from darkjaxincell import DarkField, DarkSimulation, midnight
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from examples.dark_kinetic import longitudinal_root  # noqa: E402


# Each row is (cells, bulk markers, beam markers, dt * omega_p, horizon * omega_p).
full = globals().get("full", False)
output = Path(globals().get("output", "artifacts/dark_bump"))
eta = globals().get("eta", 0.3)
cases = globals().get("cases", ((128, 80000, 40000, 0.025, 45),
                                (256, 160000, 80000, 0.0125, 45)) if full else
                      ((64, 4000, 2000, 0.025, 20),))


def build_plasma(cells, bulk_count, beam_count, dt_wp=0.025):
    """Load two resolved drifting Maxwellians and a fixed neutralizer."""
    length, mode, fraction = 1.0, 5, 0.03
    wp = 0.05 * c * 128  # fixed physical plasma while refining the grid
    k = 2 * np.pi * mode / length
    vth = wp / (5 * k)
    drift = 5 * vth
    bulk_drift = -fraction * drift / (1 - fraction)
    density = wp**2 * epsilon_0 * mass_electron / e**2
    bx, bv = quiet_start(bulk_count, length, vth=(vth, 0, 0), drift=(bulk_drift, 0, 0))
    tx, tv = quiet_start(beam_count, length, vth=(0.7 * vth, 0, 0), drift=(drift, 0, 0))
    bv = bv.at[:, 0].add(bulk_drift - jnp.mean(bv[:, 0]))
    tv = tv.at[:, 0].add(drift - jnp.mean(tv[:, 0]))
    bx = bx.at[:, 0].add(0.002 * jnp.sin(k * bx[:, 0]) / k)
    bulk = Species.electrons(bulk_count, (1 - fraction) * density, name="graveyard bulk").replace(x=bx, v=bv)
    beam = Species.electrons(beam_count, fraction * density, name="wandering tail").replace(x=tx, v=tv)
    plasma = Simulation(Domain(length, cells, time_step=dt_wp / wp), (bulk, beam))
    return plasma, wp, vth


def kinetic_root(wp, vth, eta, mu_over_wp=0.7):
    """Independent two-Maxwellian Vlasov–Proca growing root, in plasma units."""
    answer, residual = selected_bump_pole(wp, vth, 0.03, 5.0, mu_over_wp,
                                          eta, "full")
    if answer.imag <= 0:
        raise RuntimeError("the bump's growing branch was not found")
    return answer, residual


def selected_bump_pole(wp, vth, fraction, drift_over_vth, mu_over_wp,
                       eta, model):
    """Track the beam-like pole; a damped pole alone is not a stability census."""
    k = 2 * np.pi * 5
    populations = ({"wp": wp * np.sqrt(1 - fraction),
                    "u": -fraction * drift_over_vth * vth / (1 - fraction),
                    "vth": vth},
                   {"wp": wp * np.sqrt(fraction), "u": drift_over_vth * vth,
                    "vth": 0.7 * vth})
    return longitudinal_root(k, populations, mu_over_wp * wp, eta,
                             0.85 + 0.15j, model)


def bump_reference_scan(wp, vth, eta=0.3):
    """Small beam-fraction, drift and mass scan of the selected kinetic branch."""
    fractions, drifts, masses = (0.001, 0.003, 0.01, 0.03, 0.05), (4.5, 5.0, 5.5), (0.1, 0.7, 2.0)
    models = ("ordinary", "full", "quasistatic", "effective_charge")
    roots = {model: np.asarray([[[selected_bump_pole(
        wp, vth, fraction, drift, mass, eta, model)[0]
        for mass in masses] for drift in drifts] for fraction in fractions])
        for model in models}
    return {"fractions": fractions, "drifts_over_vth": drifts,
            "masses_over_wp": masses, "roots": roots}


def number_histogram(velocity, weight, counts, edges, speed):
    """Physical number distribution and population contributions per ``v/speed``."""
    velocity, weight = np.asarray(velocity), np.asarray(weight)
    total = np.sum(weight)
    start, pieces = 0, []
    for count in counts:
        stop = start + count
        bins = np.histogram(velocity[start:stop], edges, weights=weight[start:stop])[0]
        pieces.append(speed * bins / (total * np.diff(edges)))
        start = stop
    outside = float(np.sum(weight[(velocity < edges[0]) | (velocity > edges[-1])]) / total)
    return np.sum(pieces, axis=0), np.asarray(pieces), outside


def experiment(cells, bulk_count, beam_count, dt_wp, horizon, eta=0.3):
    """Measure growth, conservation and the late velocity distribution."""
    plasma, wp, vth = build_plasma(cells, bulk_count, beam_count, dt_wp)
    stride = max(1, round(0.5 / dt_wp))
    steps = stride * round(horizon / (stride * dt_wp))
    parent = plasma.run(steps, store_every=stride, store_particles=True, verbose=True)
    dark = DarkSimulation(plasma, DarkField(0.7 * wp, eta)).run(
        steps, store_every=stride, store_particles=False, verbose=True)
    t = np.asarray(parent.t) * wp

    def mode(history):
        return np.abs(np.fft.rfft(np.asarray(history[:, :, 0]), axis=1)[:, 5]) / cells

    parent_mode, mixed_mode, dark_mode = mode(parent.E), mode(dark.ordinary.E), mode(dark.E)
    parent_root, parent_residual = kinetic_root(wp, vth, 0)
    mixed_root, mixed_residual = kinetic_root(wp, vth, eta)
    window = (t >= 18) & (t <= 30)
    fits = ({name: {"growth_over_wp": float(reg.slope), "stderr": float(reg.stderr)}
             for name, reg in (("parent", linregress(t[window], np.log(parent_mode[window]))),
                               ("mixed", linregress(t[window], np.log(mixed_mode[window]))))}
            if horizon >= 35 else None)
    energy_p, energy_d = np.asarray(energies(parent)["total"]), np.asarray(dark.energy()["total_with_dark"])
    initial_p = float(dark.state.initial_ordinary)
    initial_d = float(dark.state.initial_ordinary + dark.state.initial_dark)
    edges = np.linspace(-4 * vth, 9 * vth, 180)
    histograms, components, tails = {}, {}, {}
    initial, _ = plasma.initial_state(jax.random.PRNGKey(0))
    counts = (bulk_count, beam_count)
    for name, final in (("parent", parent.state), ("mixed", dark.state.ordinary)):
        pairs = ((plasma._velocity(initial.u)[:, 0], initial.w),
                 (plasma._velocity(final.u)[:, 0], final.w))
        measured = [number_histogram(v, w, counts, edges, vth) for v, w in pairs]
        histograms[name] = np.asarray([item[0] for item in measured])
        components[name] = np.asarray([item[1] for item in measured])
        tails[name] = [item[2] for item in measured]
    rho_scale = float(np.max(np.abs(np.asarray(dark.ordinary.rho))) / epsilon_0)
    return {"t": t, "parent_mode": parent_mode, "mixed_mode": mixed_mode,
            "dark_mode": dark_mode, "parent_total": energy_p, "mixed_total": energy_d,
            "initial_parent_total": initial_p, "initial_mixed_total": initial_d,
            "velocity_edges_over_vth": edges / vth,
            "parent_histogram": histograms["parent"], "mixed_histogram": histograms["mixed"],
            "parent_population_histogram": components["parent"],
            "mixed_population_histogram": components["mixed"],
            "settings": {"cells": cells, "bulk_particles": bulk_count, "beam_particles": beam_count,
                         "dt_omega_p": dt_wp, "steps": steps, "store_every": stride,
                         "horizon_omega_p": steps * dt_wp, "mode": 5, "beam_density_fraction": 0.03,
                         "beam_drift_over_vth": 5.0, "beam_vth_over_bulk_vth": 0.7,
                         "bulk_current": "compensated by drift", "eta": eta, "mu_over_wp": 0.7},
            "results": {"parent_root_over_wp": [parent_root.real, parent_root.imag],
                        "mixed_root_over_wp": [mixed_root.real, mixed_root.imag],
                        "root_residuals": [parent_residual, mixed_residual], "fits": fits,
                        "fit_window_omega_p": [18, 30] if fits else None,
                        "parent_max_energy_drift": float(np.max(np.abs(energy_p / initial_p - 1))),
                        "mixed_max_energy_drift": float(np.max(np.abs(energy_d / initial_d - 1))),
                        "dark_gauss_relative": float(np.max(np.abs(np.asarray(dark.dark_gauss())))
                                                     / max(rho_scale, 1.0)),
                        "histogram_support_over_vth": [float(edges[0] / vth), float(edges[-1] / vth)],
                        "histogram_outside_number_fraction": tails,
                        "initial_beam_number_fraction_in_support": float(np.sum(
                            components["parent"][0, 1] * np.diff(edges / vth))),
                        "matched_loading": "identical x, v, weights, ordinary E, grid and step"}}


if __name__ == "__main__":
    runs = [experiment(*case, eta) for case in cases]
    scan = bump_reference_scan(0.05 * c * 128, (0.05 * c * 128) / (5 * 2 * np.pi * 5), eta)
    with midnight():
        fig, panels = plt.subplots(2, 2, figsize=(10, 7), layout="constrained")
        axes = (panels[0, 0], panels[0, 1], panels[1, 0])
        for run in runs:
            suffix = f"{run['settings']['cells']} cells"
            axes[0].semilogy(run["t"], run["parent_mode"], color="#0072B2", alpha=0.7,
                             label=f"parent, {suffix}")
            axes[0].semilogy(run["t"], run["mixed_mode"], color="#6A3D9A", alpha=0.7,
                             label=f"dark, {suffix}")
            axes[2].plot(run["t"], run["mixed_total"] / run["initial_mixed_total"] - 1,
                         label=f"dark, {suffix}")
        base = runs[0]
        middle = 0.5 * (base["velocity_edges_over_vth"][:-1] + base["velocity_edges_over_vth"][1:])
        for name, color in (("parent", "#0072B2"), ("mixed", "#6A3D9A")):
            axes[1].plot(middle, base[f"{name}_histogram"][1], color=color, label=name)
        axes[1].plot(middle, base["parent_histogram"][0], "--", color="#30343B", label="initial")
        axes[1].plot(middle, base["mixed_population_histogram"][1, 0], ":",
                     color="#D55E00", label="late core")
        axes[1].plot(middle, base["mixed_population_histogram"][1, 1], ":",
                     color="#009E73", label="late beam")
        axes[0].set(xlabel=r"$\omega_p t$", ylabel=r"$|E_{x,k_5}|$ (V/m)", title="Resonant beam mode")
        axes[1].set(xlabel=r"$v_x/v_{th}$", ylabel=r"$v_{th}f(v_x)$", title="Velocity distribution")
        axes[2].set(xlabel=r"$\omega_p t$", ylabel="fractional closed-energy change",
                    title="Particles + both fields")
        for model, label in (("ordinary", "ordinary"), ("full", "full Proca"),
                             ("quasistatic", "Yukawa"),
                             ("effective_charge", "constant charge")):
            panels[1, 1].semilogx(scan["fractions"], scan["roots"][model][:, 1, 1].imag,
                                  marker="o", label=label)
        panels[1, 1].axhline(0, color="#30343B", ls=":")
        panels[1, 1].set(
            xlabel="beam number fraction", ylabel=r"selected pole $\Im\omega/\omega_p$",
            title="Near-threshold kinetic reference")
        for ax in panels.flat:
            ax.grid(alpha=0.25)
            ax.legend(facecolor="#FFFFFF", edgecolor="#6B7280", fontsize=7)
        reference_scan = {"fractions": scan["fractions"],
                          "drifts_over_vth": scan["drifts_over_vth"],
                          "masses_over_wp": scan["masses_over_wp"],
                          "selected_roots_over_wp": {
                              model: np.stack((roots.real, roots.imag), axis=-1).tolist()
                              for model, roots in scan["roots"].items()}}
        save_run(output, "dark_bump", {"preset": "full" if full else "quick",
                 "cases": [run["settings"] for run in runs]},
                 {"cases": [run["results"] for run in runs], "reference_scan": reference_scan}, fig,
                 **{f"case_{i}_{key}": run[key] for i, run in enumerate(runs)
                    for key in ("t", "parent_mode", "mixed_mode", "dark_mode", "parent_total",
                                "mixed_total", "initial_parent_total", "initial_mixed_total",
                                "velocity_edges_over_vth", "parent_histogram",
                                "mixed_histogram", "parent_population_histogram",
                                "mixed_population_histogram")})
        plt.close(fig)
    print("🦇 BUMP-ON-TAIL:", [run["results"] for run in runs])
