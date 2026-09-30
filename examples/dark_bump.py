"""A matched bump-on-tail experiment with a finite Proca reservoir.

The Maxwellian beam follows the parent bump-on-tail example. A small bulk
counterdrift makes the loaded plasma current-neutral; both runs use exactly
the same particle arrays, ordinary field, grid and step.
"""

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import root
from scipy.stats import linregress

from jaxincell import (Domain, Simulation, Species, elementary_charge as e,
                       energies, epsilon_0, mass_electron, quiet_start, save_run,
                       speed_of_light as c)
from jaxincell.theory import electrostatic_epsilon
from darkjaxincell import DarkField, DarkSimulation, midnight


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
    k, fraction = 2 * np.pi * 5, 0.03
    populations = ({"wp": wp * np.sqrt(1 - fraction),
                    "u": -fraction * 5 * vth / (1 - fraction), "vth": vth},
                   {"wp": wp * np.sqrt(fraction), "u": 5 * vth, "vth": 0.7 * vth})

    def determinant(z):
        epsilon, _ = electrostatic_epsilon(z * wp, k, populations)
        s = z**2 - (k * c / wp)**2
        return (s - mu_over_wp**2) * epsilon + eta**2 * s * (epsilon - 1)

    solved = root(lambda pair: (determinant(complex(*pair)).real,
                                determinant(complex(*pair)).imag), (0.85, 0.15))
    answer = complex(*solved.x)
    if not solved.success or abs(determinant(answer)) > 1e-7 or answer.imag <= 0:
        raise RuntimeError("the bump's independent growing root did not converge")
    return answer, abs(determinant(answer))


def experiment(cells, bulk_count, beam_count, dt_wp, horizon, eta=0.3):
    """Measure growth, conservation and the late velocity distribution."""
    plasma, wp, vth = build_plasma(cells, bulk_count, beam_count, dt_wp)
    stride = max(1, round(0.5 / dt_wp))
    steps = stride * round(horizon / (stride * dt_wp))
    parent = plasma.run(steps, store_every=stride, store_particles=True)
    dark = DarkSimulation(plasma, DarkField(0.7 * wp, eta)).run(
        steps, store_every=stride, store_particles=True)
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
    edges = np.linspace(-4 * vth, 9 * vth, 180)
    histograms = {}
    initial, _ = plasma.initial_state(jax.random.PRNGKey(0))
    for name, initial_v, final_u in (("parent", initial.u[:, 0], parent.state.u[:, 0]),
                                     ("mixed", initial.u[:, 0], dark.state.ordinary.u[:, 0])):
        histograms[name] = [vth * np.histogram(np.asarray(v), edges, density=True)[0]
                            for v in (initial_v, final_u)]
    rho_scale = float(np.max(np.abs(np.asarray(dark.ordinary.rho))) / epsilon_0)
    return {"t": t, "parent_mode": parent_mode, "mixed_mode": mixed_mode,
            "dark_mode": dark_mode, "parent_total": energy_p, "mixed_total": energy_d,
            "velocity_edges_over_vth": edges / vth,
            "parent_histogram": histograms["parent"], "mixed_histogram": histograms["mixed"],
            "settings": {"cells": cells, "bulk_particles": bulk_count, "beam_particles": beam_count,
                         "dt_omega_p": dt_wp, "steps": steps, "store_every": stride,
                         "horizon_omega_p": steps * dt_wp, "mode": 5, "beam_density_fraction": 0.03,
                         "beam_drift_over_vth": 5.0, "beam_vth_over_bulk_vth": 0.7,
                         "bulk_current": "compensated by drift", "eta": eta, "mu_over_wp": 0.7},
            "results": {"parent_root_over_wp": [parent_root.real, parent_root.imag],
                        "mixed_root_over_wp": [mixed_root.real, mixed_root.imag],
                        "root_residuals": [parent_residual, mixed_residual], "fits": fits,
                        "fit_window_omega_p": [18, 30] if fits else None,
                        "parent_max_energy_drift": float(np.max(np.abs(energy_p / energy_p[0] - 1))),
                        "mixed_max_energy_drift": float(np.max(np.abs(energy_d / energy_d[0] - 1))),
                        "dark_gauss_relative": float(np.max(np.abs(np.asarray(dark.dark_gauss())))
                                                     / max(rho_scale, 1.0)),
                        "matched_loading": "identical x, v, weights, ordinary E, grid and step"}}


def main():
    parser = argparse.ArgumentParser(description="Watch a beam haunt its own tail")
    parser.add_argument("--full", action="store_true", help="resolved two-level kinetic replay")
    parser.add_argument("--output", type=Path, default=Path("artifacts/dark_bump"))
    args = parser.parse_args()
    cases = ((128, 80000, 40000, 0.025, 45), (256, 160000, 80000, 0.0125, 45)) if args.full else (
        (64, 4000, 2000, 0.025, 20),)
    runs = [experiment(*case) for case in cases]
    with midnight():
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), layout="constrained")
        for run in runs:
            suffix = f"{run['settings']['cells']} cells"
            axes[0].semilogy(run["t"], run["parent_mode"], color="#0072B2", alpha=0.7,
                             label=f"parent, {suffix}")
            axes[0].semilogy(run["t"], run["mixed_mode"], color="#6A3D9A", alpha=0.7,
                             label=f"dark, {suffix}")
            axes[2].plot(run["t"], run["mixed_total"] / run["mixed_total"][0] - 1,
                         label=f"dark, {suffix}")
        base = runs[0]
        middle = 0.5 * (base["velocity_edges_over_vth"][:-1] + base["velocity_edges_over_vth"][1:])
        for name, color in (("parent", "#0072B2"), ("mixed", "#6A3D9A")):
            axes[1].plot(middle, base[f"{name}_histogram"][1], color=color, label=name)
        axes[1].plot(middle, base["parent_histogram"][0], "--", color="#30343B", label="initial")
        axes[0].set(xlabel=r"$\omega_p t$", ylabel=r"$|E_{x,k_5}|$ (V/m)", title="Resonant beam mode")
        axes[1].set(xlabel=r"$v_x/v_{th}$", ylabel=r"$v_{th}f(v_x)$", title="Velocity distribution")
        axes[2].set(xlabel=r"$\omega_p t$", ylabel="fractional closed-energy change",
                    title="Particles + both fields")
        for ax in axes:
            ax.grid(alpha=0.25)
            ax.legend(facecolor="#FFFFFF", edgecolor="#6B7280", fontsize=7)
        save_run(args.output, "dark_bump", {"preset": "full" if args.full else "quick",
                 "cases": [run["settings"] for run in runs]},
                 {"cases": [run["results"] for run in runs]}, fig,
                 **{f"case_{i}_{key}": run[key] for i, run in enumerate(runs)
                    for key in ("t", "parent_mode", "mixed_mode", "dark_mode", "parent_total",
                                "mixed_total", "velocity_edges_over_vth", "parent_histogram",
                                "mixed_histogram")})
        plt.close(fig)
    print("🦇 BUMP-ON-TAIL:", [run["results"] for run in runs])


if __name__ == "__main__":
    main()
