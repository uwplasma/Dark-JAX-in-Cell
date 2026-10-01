"""Matched ordinary and dark two-stream growth through nonlinear saturation.

Both runs start from the same cold beams and ordinary field. The full preset
separates timestep, grid and particle-count refinement.
"""

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
from jaxincell._core import wrap_positions
from darkjaxincell import DarkField, DarkSimulation, midnight
if __package__:
    from .dark_instabilities import two_stream_growth
else:
    from dark_instabilities import two_stream_growth


# Each row is (cells, particles per beam, dt * omega_p, horizon * omega_p).
full = globals().get("full", False)
extended = globals().get("extended", False)
output = Path(globals().get("output", "artifacts/dark_saturation"))
eta = globals().get("eta", 0.3)
cases = globals().get(
    "cases",
    ((64, 4000, 0.025, 200), (128, 8000, 0.0125, 200),
     (128, 8000, 0.00625, 200), (256, 16000, 0.00625, 200)) if extended else
    ((64, 4000, 0.025, 80), (64, 4000, 0.0125, 80), (64, 8000, 0.0125, 80),
     (128, 4000, 0.0125, 80), (128, 8000, 0.0125, 80)) if full else
    ((32, 500, 0.025, 30),))


def trapping_frequency(ordinary_mode, dark_mode, eta, k):
    """Single-wave bounce estimate from the complex particle-force mode."""
    effective = np.asarray(ordinary_mode) + eta * np.asarray(dark_mode)
    return np.sqrt(2 * e * k * np.max(np.abs(effective)) / mass_electron)


def integer_positions(plasma, state):
    """Recover integer-time positions from the parent's half-step restart state."""
    domain = plasma.domain
    velocity = plasma._velocity(state.u)
    return wrap_positions(state.x - domain.dt * velocity / 2, state.w,
                          (domain.length, domain.length_y, domain.length_z),
                          domain.particle_bc, domain.dx)


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
    parent = plasma.run(steps, store_every=stride, verbose=True)
    dark = DarkSimulation(plasma, DarkField(0.7 * wp, eta)).run(
        steps, store_every=stride, store_particles=False, verbose=True)
    time = np.asarray(parent.t) * wp

    def mode(field):
        return np.fft.rfft(np.asarray(field[:, :, 0]), axis=1)[:, 1] / cells

    def fluctuating_energy(field):
        electric = np.asarray(field[:, :, 0])
        centered = electric - electric.mean(axis=1, keepdims=True)
        return 0.5 * epsilon_0 * length / cells * np.sum(centered**2, axis=1)

    parent_complex, mixed_complex, dark_complex = map(mode, (parent.E, dark.ordinary.E, dark.E))
    ordinary, mixed = np.abs(parent_complex), np.abs(mixed_complex)
    parent_fluctuation = fluctuating_energy(parent.E)
    mixed_fluctuation = fluctuating_energy(dark.ordinary.E)
    parent_energy, dark_energy = energies(parent), dark.energy()
    parent_total = np.asarray(parent_energy["total"])
    dark_total = np.asarray(dark_energy["total_with_dark"])
    initial_parent = float(dark.state.initial_ordinary)
    initial_mixed = float(dark.state.initial_ordinary + dark.state.initial_dark)
    source_work = np.asarray(dark_energy["dark_source_work"])
    dark_work_residual = np.asarray(dark_energy["dark_work_residual"])
    transfer = source_work
    window = (time > 10) & (time < 20)
    fits = {"parent": float(linregress(time[window], np.log(ordinary[window])).slope),
            "mixed": float(linregress(time[window], np.log(mixed[window])).slope)}
    late_start = 120 if horizon > 100 else 40
    late = time >= late_start
    late_rms = {"parent": float(np.sqrt(np.mean(ordinary[late]**2))),
                "mixed": float(np.sqrt(np.mean(mixed[late]**2)))} if np.any(late) else None
    late_fluctuation = {"parent": float(np.mean(parent_fluctuation[late]) / initial_parent),
                        "mixed": float(np.mean(mixed_fluctuation[late]) / initial_parent)} if np.any(late) else None
    return {"t": time, "parent_mode": ordinary, "mixed_mode": mixed,
            "parent_complex_mode": parent_complex, "mixed_complex_mode": mixed_complex,
            "dark_complex_mode": dark_complex,
            "effective_mode": np.abs(mixed_complex + eta * dark_complex),
            "parent_fluctuating_energy": parent_fluctuation,
            "mixed_fluctuating_energy": mixed_fluctuation,
            "parent_total": parent_total, "mixed_total": dark_total,
            "initial_parent_total": initial_parent, "initial_mixed_total": initial_mixed,
            "kinetic": np.asarray(dark_energy["kinetic"]),
            "ordinary_field": np.asarray(dark_energy["electric"] + dark_energy["magnetic"]),
            "dark_field": np.asarray(dark_energy["dark"]),
            "dark_source_work": source_work, "dark_work_residual": dark_work_residual,
            "parent_x": np.asarray(parent.x[-1, :, 0]),
            "parent_v": np.asarray(plasma._velocity(parent.state.u)[:, 0]),
            "mixed_x": np.asarray(integer_positions(plasma, dark.state.ordinary)[:, 0]),
            "mixed_v": np.asarray(plasma._velocity(dark.state.ordinary.u)[:, 0]),
            "parent_E": np.asarray(parent.E[-1, :, 0]),
            "mixed_E": np.asarray(dark.ordinary.E[-1, :, 0]),
            "dark_E": np.asarray(dark.E[-1, :, 0]),
            "faces": np.asarray(plasma.domain.faces),
            "fits": fits, "late_mode_rms_V_m": late_rms,
            "late_fluctuating_energy_over_initial": late_fluctuation,
            "parent_max_energy_drift": float(np.max(np.abs(parent_total / initial_parent - 1))),
            "mixed_max_energy_drift": float(np.max(np.abs(dark_total / initial_mixed - 1))),
            "max_dark_work_residual_over_transfer": float(np.max(np.abs(dark_work_residual))
                                                          / np.max(np.abs(transfer))),
            "max_dark_gauss": float(jnp.max(jnp.abs(dark.dark_gauss()))),
            "settings": {"cells": cells, "particles_per_beam": particles, "steps": steps,
                         "store_every": stride, "dt_omega_p": dtau,
                         "sample_interval_omega_p": stride * dtau,
                         "late_window_omega_p": [late_start, steps * dtau],
                         "horizon_omega_p": steps * dtau}}


if __name__ == "__main__":
    runs = [experiment(*case, eta) for case in cases]
    reference = two_stream_growth(2 * np.pi / (0.05 * 64), 0.25 * 2 * np.pi / (0.05 * 64), 0.7, eta)[0]
    parent_reference = two_stream_growth(2 * np.pi / (0.05 * 64), 0.25 * 2 * np.pi / (0.05 * 64), 0.7, 0)[0]
    wp = 0.05 * c * 64
    for run in runs:
        nonlinear = (run["t"] >= 20) & (run["t"] <= 30)
        run["bounce_over_growth"] = {
            "parent": float(trapping_frequency(run["parent_complex_mode"][nonlinear], 0,
                                               0, 2 * np.pi) / (parent_reference * wp)),
            "mixed": float(trapping_frequency(run["mixed_complex_mode"][nonlinear],
                                              run["dark_complex_mode"][nonlinear], eta,
                                              2 * np.pi) / (reference * wp))}
    with midnight():
        fig, axes = plt.subplots(2, 3, figsize=(12, 7), layout="constrained")
        t = runs[0]["t"]
        axes[0, 0].semilogy(t, runs[0]["parent_mode"], color="#0072B2", label="parent")
        axes[0, 0].semilogy(t, runs[0]["mixed_mode"], color="#6A3D9A", label="dark")
        if full or extended:
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
            axes[0, 1].plot(t, runs[0][key] / runs[0]["initial_mixed_total"], color=color, label=label)
        axes[0, 1].set(xlabel=r"$\omega_p t$", ylabel="energy / initial closed total",
                       yscale="log", ylim=(1e-5, 2), title="Where the energy goes")
        axes[0, 1].legend(facecolor="#FFFFFF", edgecolor="#6B7280", fontsize=8)
        shown = (0, 1, 4) if full and len(runs) >= 5 else range(len(runs))
        for i in shown:
            run = runs[i]
            label = (f"{run['settings']['cells']} cells, {run['settings']['particles_per_beam']} / beam, "
                     f"Δtωp={run['settings']['dt_omega_p']}")
            axes[0, 2].plot(run["t"], run["mixed_total"] / run["initial_mixed_total"] - 1, label=label)
        axes[0, 2].plot(t, runs[0]["parent_total"] / runs[0]["initial_parent_total"] - 1,
                        "--", color="#0072B2", label="parent, base")
        axes[0, 2].set(xlabel=r"$\omega_p t$", ylabel="relative total-energy change",
                       title="Closed energy; time and grid checks")
        axes[0, 2].legend(facecolor="#FFFFFF", edgecolor="#6B7280", fontsize=7)
        for ax, prefix, title, color in ((axes[1, 0], "parent", "Parent at final time", "#0072B2"),
                                         (axes[1, 1], "mixed", "Dark at final time", "#6A3D9A")):
            ax.scatter(runs[0][f"{prefix}_x"][::5], runs[0][f"{prefix}_v"][::5] / c,
                       s=0.3, color=color, alpha=0.6, rasterized=True)
            ax.set(xlim=(-0.5, 0.5), ylim=(-1, 1), xlabel="$x/L$", ylabel="$v_x/c$", title=title)
        faces = runs[0]["faces"]
        axes[1, 2].plot(faces, runs[0]["parent_E"], color="#0072B2", label="parent $E_x$")
        axes[1, 2].plot(faces, runs[0]["mixed_E"], color="#6A3D9A", label="dark $E_x$")
        axes[1, 2].plot(faces, runs[0]["dark_E"], color="#D55E00", label="$E_{D,x}$")
        axes[1, 2].set(xlabel="$x/L$", ylabel="V/m", title="Final longitudinal fields")
        axes[1, 2].legend(facecolor="#FFFFFF", edgecolor="#6B7280", fontsize=8)
        for ax in axes.flat:
            ax.grid(alpha=0.25)
        settings = {"preset": "extended" if extended else "full" if full else "quick",
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
                  for key in ("t", "parent_mode", "mixed_mode", "effective_mode",
                              "parent_complex_mode", "mixed_complex_mode", "dark_complex_mode",
                              "parent_total", "mixed_total",
                              "initial_parent_total", "initial_mixed_total",
                              "parent_fluctuating_energy", "mixed_fluctuating_energy",
                              "kinetic", "ordinary_field", "dark_field")}
        curves.update({f"case_{i}_{key}": run[key] for i, run in enumerate(runs)
                       for key in ("dark_source_work", "dark_work_residual")})
        save_run(output, "dark_saturation", settings, results, fig, **curves)
        plt.close(fig)
    print("🦇 TWO-STREAM SATURATION:", results["cases"])
