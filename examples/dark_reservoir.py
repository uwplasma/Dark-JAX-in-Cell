"""A mobile-ion pump and the finite reservoir behind its ghostly imitation.

This bounded-time, seeded control is not the long SHARP reproduction.  The
prescribed case has an external reservoir; the Proca cases start with the
same force but different finite field energies.  All share one loading.
"""

import argparse
from pathlib import Path

import jax.numpy as jnp
from jax import random
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm
from scipy.stats import linregress
import sys

from jaxincell import (Domain, Simulation, Solver, Species, elementary_charge as e,
                       epsilon_0, mass_electron, mass_proton, quiet_start,
                       save_run, speed_of_light as c)
from darkjaxincell import DarkField, DarkSimulation, PrescribedDrive, midnight
from darkjaxincell._proca import energy as dark_energy

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "docs" / "scripts"))
from pair_reference import (coupled_response, growth, relativistic_response,
                            seeded_response)  # noqa: E402


def pair_plasma(cells, particles_per_cell, dtau, seed, relativistic=False):
    """Load matched neutral waterbags for ordinary and dark pair runs."""
    omega_0 = 1e9
    omega_p = omega_0 / np.sqrt(2)
    density = omega_p**2 * epsilon_0 * mass_electron / e**2
    length = 70 * c / omega_0
    half_width = 0.05
    mode = round(0.6 * 70 / (2 * np.pi * half_width))
    k = 2 * np.pi * mode / length
    count = cells * particles_per_cell
    cell, slot = np.divmod(np.arange(count), particles_per_cell)
    x = -length / 2 + (cell + (slot + 0.5) / particles_per_cell) * length / cells
    # A coprime permutation spreads the waterbag quantiles within every cell.
    order = (slot * (particles_per_cell - 1)) % particles_per_cell
    waterbag = half_width * c * (2 * (order + 0.5) / particles_per_cell - 1)
    positions = jnp.stack((jnp.asarray(x), jnp.zeros(count), jnp.zeros(count)), axis=1)
    species = []
    for name, charge in (("electrons", -1), ("positrons", 1)):
        velocity = jnp.stack((jnp.asarray(waterbag + charge * seed * c * np.cos(k * x)),
                              jnp.zeros(count), jnp.zeros(count)), axis=1)
        species.append(Species(name, count, charge, mass_electron, density,
                               x=positions, v=velocity))
    domain = Domain(length=length, cells=cells, time_step=dtau / omega_0)
    plasma = Simulation(domain, tuple(species), Solver(relativistic=relativistic))
    return plasma, species, k, omega_0


def pair_bulk_ratio(current, field_amplitude, relativistic):
    """Cold counterstream energy from the mean current, over field energy."""
    if relativistic:
        root = np.sqrt(1 - current**2)
        bulk = 2 * current**2 / (root * (1 + root))
    else:
        bulk = current**2
    return bulk / field_amplitude**2


def pair_experiment(cells, particles_per_cell, dtau, horizon, seed=2e-4,
                    pump_enabled=True, force_quiver=0.2 / np.sqrt(2), relativistic=False):
    """Ordinary oscillating pair plasma with an independent Vlasov reference."""
    plasma, species, k, omega_0 = pair_plasma(
        cells, particles_per_cell, dtau, seed, relativistic)
    domain = plasma.domain
    length = domain.length
    half_width, pump_scale = 0.05, force_quiver
    quiver = pump_scale if pump_enabled else 0.0
    sim = DarkSimulation(plasma, PrescribedDrive(0.0, jnp.zeros(3), omega_0))
    start, _ = sim.initial_state(random.PRNGKey(0))
    field_scale = mass_electron * c * omega_0 / e
    start = sim.continue_with_parameters(start.replace(
        ordinary=start.ordinary.replace(E=start.ordinary.E.at[:, 0].add(quiver * field_scale))))
    steps = round(horizon / dtau)
    stride = max(1, round(0.2 / dtau))
    steps = stride * round(steps / stride)
    out = sim.run(steps, state=start, store_every=stride, store_particles=False)
    t = np.asarray(out.ordinary.t) * omega_0
    field = np.asarray(out.ordinary.E[:, :, 0]) / field_scale
    faces = np.asarray(domain.faces)
    coefficient = np.mean(field * np.exp(-1j * k * faces)[None, :], axis=1)
    if relativistic:
        reference, _, background = relativistic_response(
            t, k * c / omega_0, seed, 0, 1, quiver, 0, half_width, linear_end=60)
        cold_mean = background[0]
    else:
        reference = seeded_response(t, k * c / omega_0, seed, half_width, quiver)
        cold_mean = quiver * np.cos(t)
    pump = field.mean(axis=1)
    energy = out.energy()
    total = np.asarray(energy["total"])
    current = np.asarray(out.ordinary.J[:, :, 0]).mean(axis=1)
    current /= epsilon_0 * field_scale * omega_0
    bulk = pair_bulk_ratio(current, pump_scale, relativistic)
    coherent = pump**2 / pump_scale**2 + bulk
    initial_pump_energy = 0.5 * epsilon_0 * length * (pump_scale * field_scale)**2
    random_gain = ((np.asarray(energy["kinetic"]) - float(energy["kinetic"][0]))
                   / initial_pump_energy - (bulk - bulk[0]))
    nonzero = np.mean((field - pump[:, None])**2, axis=1)
    linear = t <= 35
    residual = float(np.linalg.norm(coefficient[linear] - reference[linear]) /
                     max(np.linalg.norm(reference[linear]), 1e-30))
    rate, multiplier = growth(k * c / omega_0, half_width, quiver)

    def cycle_rate(signal):
        times, peaks = [], []
        for cycle in range(2, 7):
            indices = np.flatnonzero((t >= 2 * np.pi * cycle)
                                     & (t < 2 * np.pi * (cycle + 1)))
            if not len(indices):
                continue
            peak = indices[np.argmax(abs(signal[indices]))]
            times.append(t[peak])
            peaks.append(abs(signal[peak]))
        if len(peaks) < 3:
            return None
        fit = linregress(times, np.log(peaks))
        return {"growth_over_omega0": float(fit.slope),
                "regression_stderr": float(fit.stderr), "cycles": len(peaks)}

    crossing = np.flatnonzero(coherent < 0.5) if pump_enabled else []
    result = {"floquet_growth_over_omega0": float(rate),
              "floquet_multiplier": [float(multiplier.real), float(multiplier.imag)],
              "pic_cycle_fit": cycle_rate(coefficient),
              "vlasov_cycle_fit": cycle_rate(reference),
              "linear_mode_relative_l2_error": residual,
              "early_mean_error_over_initial_field": float(np.max(
                  abs(pump[linear] - cold_mean[linear])) / pump_scale),
              "late_mean_departure_over_initial_field": float(np.max(
                  abs(pump - cold_mean)) / pump_scale),
              "max_total_energy_drift": float(np.max(abs(total / total[0] - 1))),
              "final_nonzero_field_over_initial_pump": float(nonzero[-1] / pump_scale**2),
              "late_coherent_fraction": float(np.mean(coherent[t >= t[-1] - 20])),
              "late_random_gain_over_initial_pump": float(np.mean(random_gain[t >= t[-1] - 20])),
              "half_coherence_time_omega0": float(t[crossing[0]]) if len(crossing) else None,
              "max_final_speed_over_c": float(np.max(abs(np.asarray(
                  plasma._velocity(out.state.ordinary.u)[:, 0]))) / c),
              "max_initial_speed_over_c": float(np.max(np.abs(np.concatenate(
                  [np.asarray(species[0].v[:, 0]), np.asarray(species[1].v[:, 0])])) / c))}
    settings = {"model": "ordinary neutral electron-positron waterbag",
                "reference": "Cruz, Grismayer and Silva, arXiv:2104.04490, Fig. 2",
                "parent_revision": "83d327118163833f93e2588edcb5029241f6ba2a",
                "omega0_rad_s": omega_0, "omega_p_each_over_omega0": 1 / np.sqrt(2),
                "length_c_over_omega0": 70, "cells": cells,
                "particles_per_cell_per_species": particles_per_cell,
                "dt_omega0": dtau, "horizon_omega0": float(t[-1]),
                "waterbag_full_width_over_c": 2 * half_width,
                "initial_field_over_mec_omega_p_e": force_quiver * np.sqrt(2) if pump_enabled else 0.0,
                "quiver_over_c": quiver, "velocity_seed_over_c": seed,
                "seed_mode": round(k * length / (2 * np.pi)),
                "seed_k_vT_over_omega0": k * c / omega_0 * half_width,
                "relativistic_pusher": relativistic,
                "kinetic_reference": ("relativistic 64-node orbit quadrature" if relativistic
                                      else "nonrelativistic four-edge waterbag"),
                "diagnostic": "all nonzero electric modes; complete ordinary particle-field energy"}
    return dict(t=t, pump=pump, cold_mean=cold_mean,
                mode=coefficient, reference=reference,
                nonzero=nonzero, total=total, coherent=coherent,
                random_gain=random_gain), settings, result


def pair_dark_experiment(cells, particles_per_cell, dtau, horizon,
                         seed=2e-4, eta=0.5, mass=1.0, force=0.05,
                         relativistic=True):
    """Coupled pair plasma and its independent finite-time Vlasov–Proca trace."""
    plasma, _, k, omega_0 = pair_plasma(
        cells, particles_per_cell, dtau, seed, relativistic)
    domain = plasma.domain
    field_scale = mass_electron * c * omega_0 / e
    initial_dark = force * field_scale / eta
    dark_field = jnp.tile(jnp.array([initial_dark, 0., 0.]), (cells, 1))
    sim = DarkSimulation(plasma, DarkField(mass * omega_0, eta, initial_E=dark_field))
    start, _ = sim.initial_state(random.PRNGKey(0))
    stride = max(1, round(0.2 / dtau))
    steps = stride * round(horizon / dtau / stride)
    out = sim.run(steps, state=start, store_every=stride, store_particles=False)
    t = np.asarray(out.ordinary.t) * omega_0
    ordinary = np.asarray(out.ordinary.E[:, :, 0]) / field_scale
    dark = np.asarray(out.E[:, :, 0]) / field_scale
    ordinary_mean, dark_mean = ordinary.mean(axis=1), dark.mean(axis=1)
    phase = np.exp(-1j * k * np.asarray(domain.faces))
    ordinary_mode = np.mean(ordinary * phase[None, :], axis=1)
    dark_mode = np.mean(dark * phase[None, :], axis=1)
    oracle = relativistic_response if relativistic else coupled_response
    if relativistic:
        reference, dark_reference, background = oracle(
            t, k * c / omega_0, seed, eta, mass, 0, force / eta,
            linear_end=60)
    else:
        reference, dark_reference, background = oracle(
            t, k * c / omega_0, seed, eta, mass, 0, force / eta)
    energy = out.energy()
    total = np.asarray(energy["total_with_dark"])
    current = np.asarray(out.ordinary.J[:, :, 0]).mean(axis=1)
    current /= epsilon_0 * field_scale * omega_0
    vector = np.asarray(out.A[:, :, 0]).mean(axis=1) * omega_0 / field_scale
    reservoir_scale = (force / eta)**2
    bulk = pair_bulk_ratio(current, force / eta, relativistic)
    coherent = ((ordinary_mean**2 + dark_mean**2 + mass**2 * vector**2)
                / reservoir_scale + bulk)
    initial_energy = 0.5 * epsilon_0 * domain.length * field_scale**2 * reservoir_scale
    random_gain = ((np.asarray(energy["kinetic"]) - float(energy["kinetic"][0]))
                   / initial_energy - (bulk - bulk[0]))
    nonzero_ordinary = np.mean((ordinary - ordinary_mean[:, None])**2, axis=1)
    mean_dark_energy = (0.5 * epsilon_0 * domain.length * field_scale**2
                        * (dark_mean**2 + mass**2 * vector**2))
    nonzero_dark = (np.asarray(energy["dark"]) - mean_dark_energy) / initial_energy
    early = t <= 35

    def relative_error(measured, oracle):
        return float(np.linalg.norm(measured[early] - oracle[early]) /
                     max(np.linalg.norm(oracle[early]), 1e-30))

    crossing = np.flatnonzero(coherent < 0.5)
    final_velocity = np.asarray(plasma._velocity(out.state.ordinary.u)[:, 0])
    constraint_scale = field_scale * omega_0 / c
    result = {"ordinary_mode_early_relative_l2_error": relative_error(ordinary_mode, reference),
              "dark_mode_early_relative_l2_error": relative_error(dark_mode, dark_reference),
              "homogeneous_early_error_over_initial_dark": float(max(
                  np.max(abs(ordinary_mean[early] - background[0, early])),
                  np.max(abs(dark_mean[early] - background[1, early]))) / (force / eta)),
              "max_closed_energy_drift": float(np.max(abs(total / total[0] - 1))),
              "max_all_step_balance_error_over_initial": float(
                  out.state.max_balance_error / total[0]),
              "max_ordinary_gauss_over_scale": float(
                  out.state.max_ordinary_gauss / constraint_scale),
              "max_dark_gauss_over_scale": float(out.state.max_dark_gauss / constraint_scale),
              "max_ordinary_work_residual_over_initial": float(
                  np.max(abs(np.asarray(energy["ordinary_work_residual"]))) / initial_energy),
              "max_dark_work_residual_over_initial": float(
                  np.max(abs(np.asarray(energy["dark_work_residual"]))) / initial_energy),
              "late_coherent_fraction": float(np.mean(coherent[t >= t[-1] - 20])),
              "late_random_gain_over_initial_reservoir": float(
                  np.mean(random_gain[t >= t[-1] - 20])),
              "half_coherence_time_omega0": float(t[crossing[0]]) if len(crossing) else None,
              "max_final_speed_over_c": float(np.max(abs(final_velocity)) / c),
              "initial_dark_over_matched_ordinary_field_energy": 1 / eta**2}
    settings = {"model": "neutral waterbag pair plasma with a bare homogeneous Proca reservoir",
                "parent_revision": "83d327118163833f93e2588edcb5029241f6ba2a",
                "omega0_rad_s": omega_0, "length_c_over_omega0": 70,
                "cells": cells, "particles_per_cell_per_species": particles_per_cell,
                "dt_omega0": dtau, "horizon_omega0": float(t[-1]),
                "waterbag_full_width_over_c": 0.1, "seed_mode": round(k * domain.length / (2 * np.pi)),
                "velocity_seed_over_c": seed, "eta": eta, "dark_mass_over_omega0": mass,
                "initial_effective_quiver_over_c": force,
                "relativistic_pusher": relativistic,
                "kinetic_reference": ("relativistic 64-node orbit quadrature" if relativistic
                                      else "nonrelativistic four-edge waterbag"),
                "initial_ordinary_field": 0., "initial_dark_field_over_mec_omega0_e": force / eta,
                "reference": "relativistic orbit quadrature on a coupled two-mode background"}
    return dict(t=t, ordinary_mean=ordinary_mean, dark_mean=dark_mean,
                ordinary_background=background[0], dark_background=background[1],
                ordinary_mode=ordinary_mode, dark_mode=dark_mode,
                ordinary_reference=reference, dark_reference=dark_reference,
                coherent=coherent, random_gain=random_gain,
                nonzero_ordinary=nonzero_ordinary / reservoir_scale,
                nonzero_dark=nonzero_dark, total=total), settings, result


def pair_refinement(run):
    """Keep the grid, marker and clock settings beside each measured result."""
    _, settings, result = run
    keys = ("cells", "particles_per_cell_per_species", "dt_omega0", "horizon_omega0")
    return {key: settings[key] for key in keys} | result


def pair_figure(folder, full=False):
    """Save the ordinary bridge and its independent finite-time comparison."""
    presets = ((1024, 16, 0.0125, 50), (2048, 32, 0.0125, 50),
               (2048, 64, 0.0125, 50), (4096, 16, 0.0125, 50),
               (4096, 32, 0.0125, 50), (4096, 32, 0.00625, 170)) if full else (
        (1024, 16, 0.025, 50),)
    runs = [pair_experiment(*preset, relativistic=True) for preset in presets]
    curves, settings, result = runs[-1]
    controls = {}
    if full:
        controls["no_pump"] = pair_experiment(
            4096, 32, 0.00625, 170, pump_enabled=False, relativistic=True)
        controls["fivefold_seed"] = pair_experiment(
            4096, 32, 0.00625, 170, seed=1e-3, relativistic=True)
    t = curves["t"]
    with midnight():
        fig, axes = plt.subplots(2, 2, figsize=(10, 7), layout="constrained")
        axes[0, 0].plot(t, curves["pump"], label="ordinary PIC")
        early = t <= 60
        axes[0, 0].plot(t[early], curves["cold_mean"][early],
                        "--", label="homogeneous pair oscillator")
        axes[0, 1].plot(t, np.abs(curves["mode"]), label="PIC seeded mode")
        axes[0, 1].plot(t[early], np.abs(curves["reference"][early]),
                        "--", label="linear waterbag")
        axes[0, 1].set_yscale("log")
        axes[1, 0].plot(t, curves["coherent"], label="coherent pump")
        axes[1, 0].plot(t, curves["nonzero"] / (0.2 / np.sqrt(2))**2,
                        label="nonzero electric field")
        axes[1, 0].plot(t, curves["random_gain"], label="kinetic excess above cold flow")
        if full:
            axes[1, 0].plot(t, controls["no_pump"][0]["random_gain"], "--",
                            label="no-pump kinetic excess")
        axes[1, 1].plot(t, curves["total"] / curves["total"][0] - 1)
        axes[0, 0].set(xlabel=r"$\omega_0 t$", ylabel=r"$\langle E_x\rangle/(m_ec\omega_0/e)$")
        axes[0, 1].set(xlabel=r"$\omega_0 t$", ylabel="seeded electric mode")
        axes[1, 0].set(xlabel=r"$\omega_0 t$", ylabel="energy / initial pump energy")
        axes[1, 1].set(xlabel=r"$\omega_0 t$", ylabel="relative total-energy change")
        for ax in axes.flat:
            ax.grid(alpha=0.3)
        for ax in axes[0]:
            ax.legend(facecolor="white")
        axes[1, 0].legend(facecolor="white", fontsize="small")
        save_run(folder, "oscillating_pair", {**settings, "preset": "full" if full else "quick"},
                 {**result, "refinements": [pair_refinement(run) for run in runs],
                  "controls": {name: run[2] for name, run in controls.items()},
                  "claim": "ordinary relativistic pair benchmark"},
                 fig, **curves,
                 **{f"{name}_{key}": value for name, run in controls.items()
                    for key, value in run[0].items()
                    if key in ("coherent", "random_gain", "nonzero", "total")})
        plt.close(fig)
    print("🦇 PAIR PLASMA:", result)


def pair_dark_figure(folder, full=False):
    """Compare a bare finite reservoir with its nonperiodic kinetic reference."""
    presets = ((1024, 16, 0.025, 50), (2048, 32, 0.0125, 50),
               (2048, 32, 0.00625, 170), (4096, 16, 0.00625, 170),
               (4096, 32, 0.0125, 170),
               (4096, 32, 0.00625, 170)) if full else ((2048, 32, 0.0125, 50),)
    runs = [pair_dark_experiment(*preset) for preset in presets]
    curves, settings, result = runs[-1]
    ordinary = (pair_experiment(4096, 32, 0.00625, 170,
                                force_quiver=0.05, relativistic=True)
                if full else None)
    equal_energy = (pair_experiment(4096, 32, 0.00625, 170,
                                    force_quiver=0.1, relativistic=True)
                    if full else None)
    t = curves["t"]
    early = t <= 50
    with midnight():
        fig, axes = plt.subplots(2, 2, figsize=(10, 7), layout="constrained")
        for key, oracle, color, label in (
                ("ordinary_mean", "ordinary_background", "#0072B2", "ordinary"),
                ("dark_mean", "dark_background", "#6A3D9A", "dark")):
            axes[0, 0].plot(t, curves[key], color=color, label=label)
            axes[0, 0].plot(t[early], curves[oracle][early], "--", color=color)
        for key, oracle, color, label in (
                ("ordinary_mode", "ordinary_reference", "#0072B2", "ordinary"),
                ("dark_mode", "dark_reference", "#6A3D9A", "dark")):
            axes[0, 1].plot(t, abs(curves[key]), color=color, label=label)
            axes[0, 1].plot(t[early], abs(curves[oracle][early]), "--", color=color)
        axes[0, 1].set_yscale("log")
        if ordinary is not None:
            base = ordinary[0]
            axes[0, 0].plot(base["t"], base["pump"], color="#D55E00",
                            alpha=0.65, label="ordinary, same initial force")
            axes[0, 1].plot(base["t"], abs(base["mode"]), color="#D55E00",
                            alpha=0.65, label="ordinary, same initial force")
        axes[1, 0].plot(t, curves["coherent"], label="coherent reservoir")
        axes[1, 0].plot(t, curves["random_gain"], label="kinetic excess above cold flow")
        axes[1, 0].plot(t, curves["nonzero_ordinary"] + curves["nonzero_dark"],
                        label="finite-k fields and Proca potential")
        if equal_energy is not None:
            axes[1, 0].plot(equal_energy[0]["t"], equal_energy[0]["coherent"],
                            "--", color="#D55E00", label="ordinary, same initial energy")
        axes[1, 1].plot(t, curves["total"] / curves["total"][0] - 1,
                        label="full Maxwell–Proca")
        if ordinary is not None:
            base = ordinary[0]
            axes[1, 1].plot(base["t"], base["total"] / base["total"][0] - 1,
                            "--", label="matched-force ordinary")
        axes[0, 0].set(xlabel=r"$\omega_0 t$", ylabel=r"mean $E_x/(m_ec\omega_0/e)$")
        axes[0, 1].set(xlabel=r"$\omega_0 t$", ylabel="seeded electric modes")
        axes[1, 0].set(xlabel=r"$\omega_0 t$", ylabel="energy / initial dark reservoir")
        axes[1, 1].set(xlabel=r"$\omega_0 t$", ylabel="relative total-energy change")
        for ax in axes.flat:
            ax.grid(alpha=0.3)
            ax.legend(facecolor="white", fontsize="small")
        save_run(folder, "oscillating_dark_pair", {**settings, "preset": "full" if full else "quick"},
                 {**result, "refinements": [pair_refinement(run) for run in runs],
                  "ordinary_matched_force": ordinary[2] if ordinary is not None else None,
                  "ordinary_matched_energy": equal_energy[2] if equal_energy is not None else None,
                  "claim": "finite-time kinetic comparison; no new nonlinear mechanism claimed"},
                 fig, **curves,
                 **({f"matched_ordinary_{key}": ordinary[0][key] for key in (
                     "coherent", "random_gain", "mode", "total")}
                    if ordinary is not None else {}),
                 **({f"matched_energy_{key}": equal_energy[0][key] for key in (
                     "coherent", "random_gain", "mode", "total")}
                    if equal_energy is not None else {}))
        plt.close(fig)
    print("🦇 DARK PAIR:", result)


def cold_reference(t, initial, eta, mass_ratio, drive_amplitude=None):
    """Independent homogeneous two-fluid solution in plasma-frequency units."""
    if drive_amplitude is not None:
        def rhs(time, y):
            field, electron, ion = y
            force = field + drive_amplitude * np.cos(time)
            return [electron - ion, -force, force / mass_ratio]

        return solve_ivp(rhs, (0, float(t[-1])), initial, t_eval=t,
                         rtol=1e-11, atol=1e-13).y.T
    matrix = np.array([[0, 0, 0, 1, -1], [0, 0, 1, eta, -eta],
                       [0, -1, 0, 0, 0], [-1, -eta, 0, 0, 0],
                       [1 / mass_ratio, eta / mass_ratio, 0, 0, 0]])
    # The caller sets the dark rest frequency to omega_p: A'=-D, D'=A-eta J.
    return np.array([expm(matrix * time) @ initial for time in t])


def energy_parts(state, plasma, model):
    """Longitudinal kinetic, ordinary field, and optional full Proca energy."""
    ordinary = state.ordinary
    mass, _ = plasma.per_particle
    mass_weight = np.asarray(mass * ordinary.w)
    velocity = np.asarray(plasma._velocity(ordinary.u)[:, 0])
    kinetic = 0.5 * np.sum(mass_weight * velocity**2)
    dx = plasma.domain.dx
    fields = 0.5 * epsilon_0 * dx * np.sum(np.asarray(ordinary.E) ** 2)
    fields += 0.5 * epsilon_0 * c**2 * dx * np.sum(np.asarray(ordinary.B) ** 2)
    ghost = 0.0 if not isinstance(model, DarkField) else float(
        dark_energy(state.E, state.B, state.A, state.phi, dx, model.omega))
    return np.array([kinetic, fields, ghost])


def local_random(state, plasma):
    """Cell-local longitudinal random kinetic energy, excluding cell bulk flow."""
    ordinary = state.ordinary
    mass, _ = plasma.per_particle
    weighted_mass = np.asarray(mass * ordinary.w)
    velocity = np.asarray(plasma._velocity(ordinary.u)[:, 0])
    cell = np.floor(np.asarray(ordinary.x[:, 0]) / plasma.domain.dx).astype(int)
    cell %= plasma.domain.cells
    random_energy = 0.0
    offset = 0
    for species in plasma.species:
        section = slice(offset, offset + species.n)
        count = np.bincount(cell[section], weights=weighted_mass[section],
                            minlength=plasma.domain.cells)
        momentum = np.bincount(cell[section], weights=weighted_mass[section]
                               * velocity[section], minlength=plasma.domain.cells)
        bulk = np.divide(momentum, count, out=np.zeros_like(momentum), where=count > 0)
        random_energy += 0.5 * np.sum(weighted_mass[section]
                                      * (velocity[section] - bulk[cell[section]]) ** 2)
        offset += species.n
    return random_energy


def run_case(plasma, model, steps, store_every, force, wp):
    """Run matched particles and compare their mean with a cold-fluid oracle."""
    sim = DarkSimulation(plasma, model)
    start, _ = sim.initial_state(random.PRNGKey(0))
    out = sim.run(steps, state=start, store_every=store_every, store_particles=False)
    end = out.state
    t = np.asarray(out.ordinary.t) * wp
    mass_ratio = mass_proton / mass_electron
    velocity_scale = e * force / (mass_electron * wp)
    initial_u = np.asarray(start.ordinary.u[:, 0]) / velocity_scale
    electrons = plasma.species[0].n
    beginning = [float(np.mean(np.asarray(start.ordinary.E[:, 0])) / force),
                 float(np.mean(initial_u[:electrons])), float(np.mean(initial_u[electrons:]))]
    driven = isinstance(model, PrescribedDrive)
    if driven:
        pump_strength = float(model.eta * model.amplitude[0] / force)
        oracle = cold_reference(t, beginning, 0, mass_ratio, pump_strength)
        pump = pump_strength * np.cos(t)
        mean_error = np.max(np.abs(np.asarray(out.ordinary.E[:, :, 0]).mean(axis=1) / force
                                   - oracle[:, 0]))
    else:
        beginning = [beginning[0], float(np.mean(np.asarray(start.E[:, 0])) / force),
                     float(np.mean(np.asarray(start.A[:, 0])) * wp / force),
                     beginning[1], beginning[2]]
        oracle = cold_reference(t, beginning, float(model.eta), mass_ratio)
        pump = float(model.eta) * np.asarray(out.E[:, :, 0]).mean(axis=1) / force
        effective = np.stack((np.asarray(out.ordinary.E[:, :, 0]).mean(axis=1),
                              float(model.eta) * np.asarray(out.E[:, :, 0]).mean(axis=1)), axis=1)
        mean_error = np.max(np.abs(effective / force
                                   - oracle[:, :2] * np.array([1, float(model.eta)])))
    field = np.asarray(out.ordinary.E[:, :, 0])
    mode = np.abs(np.fft.rfft(field, axis=1)[:, 1]) / plasma.domain.cells
    fluctuation = field - field.mean(axis=1, keepdims=True)
    nonzero_energy = 0.5 * epsilon_0 * plasma.domain.dx * np.sum(fluctuation**2, axis=1)
    initial_energy, final_energy = energy_parts(start, plasma, model), energy_parts(end, plasma, model)
    work = float(np.asarray(out.work[-1]))
    balance = (sum(final_energy) - sum(initial_energy) - (work if driven else 0))
    energy_scale = max(initial_energy[0], initial_energy[2], abs(work))
    return dict(t=t, pump=pump, mean_E=field.mean(axis=1) / force,
                mode=mode, nonzero_energy=nonzero_energy, cold_mean=oracle[:, 0],
                mean_error=float(mean_error),
                initial_energy=initial_energy.tolist(), final_energy=final_energy.tolist(),
                local_random_initial=local_random(start, plasma),
                local_random_final=local_random(end, plasma),
                work=work, balance_over_scale=float(balance / energy_scale))


def experiment(cells, particles, dtau, horizon, length_units=2 * np.pi, model_names=None):
    """Use co-located neutral species and one deterministic kinetic seed."""
    wp, mass_ratio = 1e9, mass_proton / mass_electron
    density = wp**2 * epsilon_0 * mass_electron / e**2
    vth_e = np.sqrt(2e-3) * c
    vth_i = vth_e / np.sqrt(mass_ratio)
    length = length_units * c / wp
    x, v = quiet_start(particles, length, vth=(vth_e, 0, 0))
    ions_v = v.at[:, 0].set(v[:, 0] / np.sqrt(mass_ratio))
    v = v.at[:, 0].add(0.01 * vth_e * jnp.sin(2 * np.pi * x[:, 0] / length))
    electron = Species.electrons(particles, density=density).replace(x=x, v=v)
    ion = Species.ions(particles, density=density, vth=(vth_i, 0, 0)).replace(x=x, v=ions_v)
    plasma = Simulation(Domain(length=length, cells=cells, time_step=dtau / wp),
                        (electron, ion))
    force = 0.03 * vth_e * mass_electron * wp / e
    steps = round(horizon / dtau)
    steps = 20 * round(steps / 20)
    cases = {"zero": PrescribedDrive(0.0, jnp.zeros(3), wp),
             "external": PrescribedDrive(1.0, jnp.array([force, 0.0, 0.0]), wp)}
    for eta, label in ((0.2, "small_reservoir"), (0.02, "large_reservoir")):
        initial = jnp.tile(jnp.array([force / eta, 0.0, 0.0]), (cells, 1))
        cases[label] = DarkField(wp, eta, initial_E=initial)
    if model_names is not None:
        cases = {name: cases[name] for name in model_names}
    histories = {label: run_case(plasma, model, steps, 20, force, wp)
                 for label, model in cases.items()}
    return histories, dict(cells=cells, particles_per_species=particles, steps=steps,
                           dt_omega_p=dtau, horizon_omega_p=steps * dtau,
                           omega_p_rad_s=wp, length_c_over_omega_p=length_units,
                           T_e_over_mec2=1e-3, T_i_over_mec2=1e-3,
                           ion_to_electron_mass=mass_ratio, force_V_m=force,
                           drive_quiver_over_electron_vth=0.03,
                           velocity_mode_seed_over_vth=0.01,
                           model="periodic 1D3V with 1V thermal loading; co-located neutral species")


def paper_geometry_pilot(folder):
    """Early strong-drive check at SHARP's box/grid, with two loading counts."""
    runs = [experiment(1000, particles, 0.02, 80, length_units=40,
                       model_names=("zero", "external")) for particles in (20000, 40000)]
    histories, settings = runs[-1]
    t = histories["zero"]["t"]
    results = {str(config["particles_per_species"]): {
        name: {"mean_oracle_error_over_force": case["mean_error"],
               "energy_work_balance_over_scale": case["balance_over_scale"],
               "nonzero_field_energy_peak_over_initial_particle": float(
                   np.max(case["nonzero_energy"]) / case["initial_energy"][0]),
               "local_random_final_over_initial": case["local_random_final"] / case["local_random_initial"]}
        for name, case in records.items()} for records, config in runs}
    with midnight():
        fig, axes = plt.subplots(2, 2, figsize=(10, 7), layout="constrained")
        for records, config in runs:
            count = config["particles_per_species"]
            for name, color in (("zero", "#6B7280"), ("external", "#6A3D9A")):
                case = records[name]
                label = f"{name}, {count // 1000}k/species"
                linestyle = "-" if count == 40000 else "--"
                axes[0, 1].semilogy(t, np.maximum(case["nonzero_energy"] / case["initial_energy"][0], 1e-15),
                                    color=color, ls=linestyle, label=label)
                axes[1, 0].semilogy(t, case["mode"] / case["mode"][0],
                                    color=color, ls=linestyle)
        axes[0, 0].plot(t, histories["external"]["mean_E"], color="#0072B2", label="PIC")
        axes[0, 0].plot(t, histories["external"]["cold_mean"], "--", color="#D55E00",
                        label="independent cold two-fluid")
        bars = [results[str(count)][name]["local_random_final_over_initial"]
                for count in (20000, 40000) for name in ("zero", "external")]
        axes[1, 1].bar(range(4), bars, color=["#6B7280", "#6A3D9A"] * 2)
        axes[1, 1].set_xticks(range(4), ("20k zero", "20k drive", "40k zero", "40k drive"), rotation=15)
        axes[0, 0].set(xlabel=r"$\omega_p t$", ylabel="mean $E_x/F$", title="The early mean response")
        axes[0, 1].set(xlabel=r"$\omega_p t$", ylabel="nonzero-$k$ E energy / initial particle",
                       title="Higher modes, including noise")
        axes[1, 0].set(xlabel=r"$\omega_p t$", ylabel="seeded $k_1$ amplitude / initial",
                       title="One seeded mode")
        axes[1, 1].set(ylabel="cell-local random kinetic / initial", title=r"At $\omega_p t=80$")
        for ax in axes.flat:
            ax.grid(alpha=0.25)
        axes[0, 0].legend(facecolor="#FFFFFF", edgecolor="#6B7280")
        axes[0, 1].legend(facecolor="#FFFFFF", edgecolor="#6B7280", fontsize=8)
        curves = {f"{config['particles_per_species']}_{name}_{key}": case[key]
                  for records, config in runs for name, case in records.items()
                  for key in ("mode", "nonzero_energy", "mean_E", "cold_mean")}
        save_run(folder, "dark_reservoir_paper_geometry_pilot",
                 {**settings, "preset": "early paper-geometry pilot", "particles_per_species": [20000, 40000],
                  "source": "Hook, Huang and Shalaby arXiv:2510.13956v1, Appendix B",
                  "limits": "short horizon; quadratic shape; loading below inferred published count"},
                 {"cases": results, "claim": "early control only; nonlinear reproduction unverified"}, fig,
                 t=t, **curves)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Follow a mobile-ion ghost and its finite reservoir")
    preset = parser.add_mutually_exclusive_group()
    preset.add_argument("--full", action="store_true", help="two physical resolutions")
    preset.add_argument("--paper-pilot", action="store_true", help="short paper-geometry loading check")
    pair_mode = parser.add_mutually_exclusive_group()
    pair_mode.add_argument("--pair", action="store_true", help="ordinary oscillating pair-plasma bridge")
    pair_mode.add_argument("--pair-dark", action="store_true", help="finite Proca pair-plasma bridge")
    parser.add_argument("--output", type=Path, default=Path("artifacts/dark_reservoir"))
    args = parser.parse_args()
    if args.pair:
        pair_figure(args.output, args.full)
        return
    if args.pair_dark:
        pair_dark_figure(args.output, args.full)
        return
    if args.paper_pilot:
        paper_geometry_pilot(args.output)
        return
    presets = ((64, 4000, 0.04, 40), (128, 8000, 0.02, 40)) if args.full else (
        (32, 1000, 0.08, 20),)
    runs = [experiment(*preset) for preset in presets]
    histories, settings = runs[-1]
    t = histories["external"]["t"]
    zero = histories["zero"]["mode"]
    curves = ("t", "pump", "mean_E", "mode", "nonzero_energy", "cold_mean")
    results = {"cases": {name: {key: value for key, value in record.items() if key not in curves}
                         for name, record in histories.items()},
               "refinements": [{name: record["mean_error"] for name, record in cases.items()}
                               for cases, _ in runs],
               "claim": "bounded-time mobile-ion control; long SHARP reproduction unverified"}
    results["finite_vs_external"] = {
        name: {"pump_error_early_over_initial_force": float(np.max(np.abs(
                    record["pump"][t <= 5] - histories["external"]["pump"][t <= 5]))),
               "pump_error_full_over_initial_force": float(np.max(np.abs(
                    record["pump"] - histories["external"]["pump"]))),
               "initial_dark_over_initial_particle_energy": (
                   record["initial_energy"][2] / record["initial_energy"][0]),
               "dark_depletion_fraction": (
                   1 - record["final_energy"][2] / record["initial_energy"][2])}
        for name, record in histories.items() if name.endswith("reservoir")}
    with midnight():
        fig, axes = plt.subplots(2, 2, figsize=(10, 7), layout="constrained")
        colors = {"zero": "#6B7280", "external": "#B03568",
                  "small_reservoir": "#6A3D9A", "large_reservoir": "#0072B2"}
        for name, record in histories.items():
            color = colors[name]
            if name != "zero":
                axes[0, 0].plot(t, record["pump"], color=color, label=name.replace("_", " "))
            axes[0, 1].plot(t, record["mean_E"], color=color, label=name.replace("_", " "))
            axes[1, 0].semilogy(t, record["mode"] / max(zero[0], 1e-30), color=color)
        axes[1, 1].bar(range(len(histories)),
                       [record["local_random_final"] / record["local_random_initial"]
                        for record in histories.values()], color=list(colors.values()))
        axes[0, 1].plot(t, histories["small_reservoir"]["cold_mean"], "--", color="#202124",
                        lw=1, label="small reservoir cold oracle")
        axes[0, 0].set(xlabel=r"$\omega_p t$", ylabel="effective pump / F",
                       title="An imposed haunting versus finite reservoirs")
        axes[0, 1].set(xlabel=r"$\omega_p t$", ylabel="mean ordinary E / initial force",
                       title="Mobile-ion mean response")
        axes[1, 0].set(xlabel=r"$\omega_p t$", ylabel="seeded $|E_{x,k}|$ / initial",
                       title="Nonzero-k mode; no growth claim")
        axes[1, 1].set(ylabel="local random kinetic / initial",
                       title="Final one-cell thermal measure")
        axes[1, 1].set_xticks(range(len(histories)),
                              [name.replace("_", "\n") for name in histories], fontsize="small")
        axes[1, 1].set_ylim(0.98, 1.01)
        for ax in axes.flat:
            ax.grid(alpha=0.35)
        axes[0, 0].legend(facecolor="#FFFFFF", edgecolor="#6B7280", fontsize="small")
        axes[0, 1].legend(facecolor="#FFFFFF", edgecolor="#6B7280", fontsize="small")
        settings = {**settings, "preset": "full" if args.full else "quick",
                    "refinements": [setting.copy() for _, setting in runs]}
        save_run(args.output, "dark_reservoir", settings, results, fig, t=t,
                 **{f"{name}_{key}": record[key] for name, record in histories.items()
                    for key in ("pump", "mean_E", "mode", "cold_mean")})
        plt.close(fig)
    print("🦇 MOBILE IONS: mean-oracle errors", results["refinements"],
          "; finite-reservoir energy balances",
          [results["cases"][name]["balance_over_scale"]
           for name in ("small_reservoir", "large_reservoir")])


if __name__ == "__main__":
    main()
