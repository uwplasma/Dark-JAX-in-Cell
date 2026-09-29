"""Matched ordinary and dark two-stream growth through nonlinear saturation.

Both runs start from the same cold beams and ordinary field. The full preset
separates timestep, grid and particle-count refinement.
"""

import argparse
from pathlib import Path

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
from dark_instabilities import two_stream_growth


def experiment(cells, particles, dtau, horizon, eta):
    """Run the exact same loading through both explicit field systems."""
    length, k, wp = 1.0, 2 * np.pi, 0.05 * c * 64
    density = wp**2 * epsilon_0 * mass_electron / e**2
    drift = 0.25 * c
    plus_x, plus_v = quiet_start(particles, length, drift=(drift, 0, 0))
    minus_x, minus_v = quiet_start(particles, length, drift=(-drift, 0, 0))
    plus_x = plus_x.at[:, 0].add(0.001 * jnp.sin(k * plus_x[:, 0]) / k)
    species = (Species.electrons(particles, density / 2, name="moonward").replace(x=plus_x, v=plus_v),
               Species.electrons(particles, density / 2, name="cryptward").replace(x=minus_x, v=minus_v))
    plasma = Simulation(Domain(length, cells, time_step=dtau / wp), species)
    steps = round(horizon / dtau)
    # Keep the diagnostic interval fixed when extending the horizon: sparse
    # output can otherwise hide peaks in the energy-error history.
    stride = max(1, round(0.5 / dtau)) if horizon > 30 else max(1, steps // 80)
    steps = stride * round(steps / stride)
    parent = plasma.run(steps, store_every=stride)
    dark = DarkSimulation(plasma, DarkField(0.7 * wp, eta)).run(steps, store_every=stride)
    time = np.asarray(parent.t) * wp

    def mode(field):
        return np.abs(np.fft.rfft(np.asarray(field[:, :, 0]), axis=1)[:, 1]) / cells

    def fluctuating_energy(field):
        electric = np.asarray(field[:, :, 0])
        centered = electric - electric.mean(axis=1, keepdims=True)
        return 0.5 * epsilon_0 * length / cells * np.sum(centered**2, axis=1)

    ordinary, mixed = mode(parent.E), mode(dark.ordinary.E)
    parent_fluctuation = fluctuating_energy(parent.E)
    mixed_fluctuation = fluctuating_energy(dark.ordinary.E)
    parent_energy, dark_energy = energies(parent), dark.energy()
    parent_total = np.asarray(parent_energy["total"])
    dark_total = np.asarray(dark_energy["total_with_dark"])
    source_work = np.asarray(dark_energy["dark_source_work"])
    dark_work_residual = np.asarray(dark_energy["dark_work_residual"])
    transfer = source_work - source_work[0]
    window = (time > 10) & (time < 20)
    fits = {"parent": float(linregress(time[window], np.log(ordinary[window])).slope),
            "mixed": float(linregress(time[window], np.log(mixed[window])).slope)}
    late_start = 120 if horizon > 100 else 40
    late = time >= late_start
    late_rms = {"parent": float(np.sqrt(np.mean(ordinary[late]**2))),
                "mixed": float(np.sqrt(np.mean(mixed[late]**2)))} if np.any(late) else None
    late_fluctuation = {"parent": float(np.mean(parent_fluctuation[late]) / parent_total[0]),
                        "mixed": float(np.mean(mixed_fluctuation[late]) / parent_total[0])} if np.any(late) else None
    return {"t": time, "parent_mode": ordinary, "mixed_mode": mixed,
            "parent_fluctuating_energy": parent_fluctuation,
            "mixed_fluctuating_energy": mixed_fluctuation,
            "parent_total": parent_total, "mixed_total": dark_total,
            "kinetic": np.asarray(dark_energy["kinetic"]),
            "ordinary_field": np.asarray(dark_energy["electric"] + dark_energy["magnetic"]),
            "dark_field": np.asarray(dark_energy["dark"]),
            "dark_source_work": source_work, "dark_work_residual": dark_work_residual,
            "parent_x": np.asarray(parent.state.x[:, 0]),
            "parent_v": np.asarray(plasma._velocity(parent.state.u)[:, 0]),
            "mixed_x": np.asarray(dark.state.ordinary.x[:, 0]),
            "mixed_v": np.asarray(plasma._velocity(dark.state.ordinary.u)[:, 0]),
            "parent_E": np.asarray(parent.E[-1, :, 0]),
            "mixed_E": np.asarray(dark.ordinary.E[-1, :, 0]),
            "dark_E": np.asarray(dark.E[-1, :, 0]),
            "fits": fits, "late_mode_rms_V_m": late_rms,
            "late_fluctuating_energy_over_initial": late_fluctuation,
            "parent_max_energy_drift": float(np.max(np.abs(parent_total / parent_total[0] - 1))),
            "mixed_max_energy_drift": float(np.max(np.abs(dark_total / dark_total[0] - 1))),
            "max_dark_work_residual_over_transfer": float(np.max(np.abs(dark_work_residual))
                                                          / np.max(np.abs(transfer))),
            "max_dark_gauss": float(jnp.max(jnp.abs(dark.dark_gauss()))),
            "settings": {"cells": cells, "particles_per_beam": particles, "steps": steps,
                         "store_every": stride, "dt_omega_p": dtau,
                         "sample_interval_omega_p": stride * dtau,
                         "late_window_omega_p": [late_start, steps * dtau],
                         "horizon_omega_p": steps * dtau}}


def main():
    parser = argparse.ArgumentParser(description="Watch two matched streams meet the dark")
    preset = parser.add_mutually_exclusive_group()
    preset.add_argument("--full", action="store_true", help="factorial check through ωp t=80")
    preset.add_argument("--extended", action="store_true", help="matched grid/time replay through ωp t=200")
    parser.add_argument("--output", type=Path, default=Path("artifacts/dark_saturation"))
    args = parser.parse_args()
    eta = 0.3
    if args.extended:
        cases = ((64, 4000, 0.025, 200), (128, 8000, 0.0125, 200),
                 (128, 8000, 0.00625, 200), (256, 16000, 0.00625, 200))
    elif args.full:
        cases = ((64, 4000, 0.025, 80), (64, 4000, 0.0125, 80),
                 (64, 8000, 0.0125, 80), (128, 4000, 0.0125, 80),
                 (128, 8000, 0.0125, 80))
    else:
        cases = ((32, 500, 0.025, 30),)
    runs = [experiment(*case, eta) for case in cases]
    reference = two_stream_growth(2 * np.pi / (0.05 * 64), 0.25 * 2 * np.pi / (0.05 * 64), 0.7, eta)[0]
    parent_reference = two_stream_growth(2 * np.pi / (0.05 * 64), 0.25 * 2 * np.pi / (0.05 * 64), 0.7, 0)[0]
    wp = 0.05 * c * 64
    for run in runs:
        nonlinear = (run["t"] >= 20) & (run["t"] <= 30)
        run["bounce_over_growth"] = {
            name: float(np.sqrt(2 * e * 2 * np.pi * np.max(run[f"{name}_mode"][nonlinear])
                                / mass_electron) / (rate * wp))
            for name, rate in (("parent", parent_reference), ("mixed", reference))}
    with midnight():
        fig, axes = plt.subplots(2, 3, figsize=(12, 7), layout="constrained")
        t = runs[0]["t"]
        axes[0, 0].semilogy(t, runs[0]["parent_mode"], color="#0072B2", label="parent")
        axes[0, 0].semilogy(t, runs[0]["mixed_mode"], color="#6A3D9A", label="dark")
        if args.full or args.extended:
            window = (t >= 10) & (t <= 20)
            for rate, mode, color in ((parent_reference, "parent_mode", "#0072B2"),
                                      (reference, "mixed_mode", "#6A3D9A")):
                curve = runs[0][mode][np.flatnonzero(window)[0]] * np.exp(rate * (t - t[window][0]))
                axes[0, 0].semilogy(t[window], curve[window], "--", color=color)
        axes[0, 0].set(xlabel=r"$\omega_p t$", ylabel=r"$|E_{x,k_1}|$ (V/m)",
                       title="Linear growth and saturation")
        axes[0, 0].legend(facecolor="#FFFFFF", edgecolor="#6B7280")
        for key, color, label in (("kinetic", "#D55E00", "particles"),
                                  ("ordinary_field", "#0072B2", "ordinary field"),
                                  ("dark_field", "#6A3D9A", "dark field")):
            axes[0, 1].plot(t, runs[0][key] / runs[0]["mixed_total"][0], color=color, label=label)
        axes[0, 1].set(xlabel=r"$\omega_p t$", ylabel="energy / initial closed total",
                       yscale="log", ylim=(1e-5, 2), title="Where the energy goes")
        axes[0, 1].legend(facecolor="#FFFFFF", edgecolor="#6B7280", fontsize=8)
        shown = (0, 1, 4) if args.full else range(len(runs))
        for i in shown:
            run = runs[i]
            label = (f"{run['settings']['cells']} cells, {run['settings']['particles_per_beam']} / beam, "
                     f"Δtωp={run['settings']['dt_omega_p']}")
            axes[0, 2].plot(run["t"], run["mixed_total"] / run["mixed_total"][0] - 1, label=label)
        axes[0, 2].plot(t, runs[0]["parent_total"] / runs[0]["parent_total"][0] - 1,
                        "--", color="#0072B2", label="parent, base")
        axes[0, 2].set(xlabel=r"$\omega_p t$", ylabel="relative total-energy change",
                       title="Closed energy; time and grid checks")
        axes[0, 2].legend(facecolor="#FFFFFF", edgecolor="#6B7280", fontsize=7)
        for ax, prefix, title, color in ((axes[1, 0], "parent", "Parent at final time", "#0072B2"),
                                         (axes[1, 1], "mixed", "Dark at final time", "#6A3D9A")):
            ax.scatter(runs[0][f"{prefix}_x"][::5], runs[0][f"{prefix}_v"][::5] / c,
                       s=0.3, color=color, alpha=0.6, rasterized=True)
            ax.set(xlim=(-0.5, 0.5), ylim=(-1, 1), xlabel="$x/L$", ylabel="$v_x/c$", title=title)
        grid = np.arange(cases[0][0]) / cases[0][0] - 0.5
        axes[1, 2].plot(grid, runs[0]["parent_E"], color="#0072B2", label="parent $E_x$")
        axes[1, 2].plot(grid, runs[0]["mixed_E"], color="#6A3D9A", label="dark $E_x$")
        axes[1, 2].plot(grid, runs[0]["dark_E"], color="#D55E00", label="$E_{D,x}$")
        axes[1, 2].set(xlabel="$x/L$", ylabel="V/m", title="Final longitudinal fields")
        axes[1, 2].legend(facecolor="#FFFFFF", edgecolor="#6B7280", fontsize=8)
        for ax in axes.flat:
            ax.grid(alpha=0.25)
        settings = {"preset": "extended" if args.extended else "full" if args.full else "quick",
                    "eta": eta,
                    "mu_over_wp": 0.7, "cases": [run["settings"] for run in runs],
                    "matched_loading": "same species x, v, weights and ordinary initial field"}
        result_keys = ("fits", "late_mode_rms_V_m", "late_fluctuating_energy_over_initial",
                       "bounce_over_growth",
                       "parent_max_energy_drift", "mixed_max_energy_drift",
                       "max_dark_work_residual_over_transfer", "max_dark_gauss")
        results = {"ordinary_cold_growth_over_wp": parent_reference,
                   "mixed_cold_growth_over_wp": reference,
                   "cases": [{key: run[key] for key in result_keys} for run in runs],
                   "scope": "cold two-stream long-time numerical comparison; no new-physics claim"}
        curves = {f"case_{i}_{key}": run[key] for i, run in enumerate(runs)
                  for key in ("t", "parent_mode", "mixed_mode", "parent_total", "mixed_total",
                              "parent_fluctuating_energy", "mixed_fluctuating_energy",
                              "kinetic", "ordinary_field", "dark_field")}
        curves.update({f"case_{i}_{key}": run[key] for i, run in enumerate(runs)
                       for key in ("dark_source_work", "dark_work_residual")})
        save_run(args.output, "dark_saturation", settings, results, fig, **curves)
        plt.close(fig)
    print("🦇 TWO-STREAM SATURATION:", results["cases"])


if __name__ == "__main__":
    main()
