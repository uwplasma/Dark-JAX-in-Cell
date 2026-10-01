"""A mobile-ion pump and the finite reservoir behind its ghostly imitation.

This bounded-time, seeded control is not the long SHARP reproduction.  The
prescribed case has an external reservoir; the Proca cases start with the
same force but different finite field energies.  All share one loading.

Edit the inputs below or supply them with ``runpy.run_path(...,
run_name='__main__', init_globals={...})``. ``full=True`` selects the full
study preset; the paper preset uses 103,000 markers per species through 5,000.
"""

import hashlib
from pathlib import Path
import resource
import time

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
from darkjaxincell import DarkField, DarkSimulation, PrescribedDrive, load_state, midnight, save_state
from darkjaxincell._proca import energy as dark_energy

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "docs" / "scripts"))
from pair_reference import (coupled_response, growth, relativistic_response,
                            seeded_response)  # noqa: E402
from conservation import coarse_spread, elapsed_progress, measured_run, snapshot  # noqa: E402
from drive_reference import forced_cold, homogeneous  # noqa: E402


# Inputs: the default is a small mobile-ion comparison.
study = globals().get('study', 'mobile_ions')  # paper, pair, pair_dark or paper_pilot
full = globals().get('full', False)
output = Path(globals().get('output', 'artifacts/dark_reservoir'))
# Paper controls use omega_p time units and markers per species.
cells = globals().get('cells', 1000)
particles = globals().get('particles', 103000 if full else 20000)
dt = globals().get('dt', 0.02)
horizon = globals().get('horizon', 5000 if full else 40)
seed = globals().get('seed', 0)
drive_ratio = globals().get('drive_ratio', 0.03)
coupling = globals().get('coupling', None)  # None selects the prescribed drive.
block_horizon = globals().get('block_horizon', min(100, horizon))  # None preserves a single full scan.
local_moments = globals().get('local_moments', False)
initial_state = globals().get('initial_state', None)
initial_state = None if initial_state is None else Path(initial_state)


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
    out = sim.run(steps, state=start, store_every=stride, store_particles=False, verbose=True)
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
    out = sim.run(steps, state=start, store_every=stride, store_particles=False, verbose=True)
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
               (4096, 32, 0.0125, 170), (4096, 64, 0.00625, 170),
               (8192, 32, 0.00625, 170),
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
    out = sim.run(steps, state=start, store_every=store_every, store_particles=False, verbose=True)
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


def paper_plasma(cells, particles, dtau, seed):
    """Appendix B plasma: RMS sigma, exact mass ratio, unseeded Gaussian velocities."""
    if not isinstance(cells, (int, np.integer)) or cells < 4:
        raise ValueError("paper cells must be an integer >= 4")
    if not isinstance(particles, (int, np.integer)) or particles < 2:
        raise ValueError("paper particles per species must be an integer >= 2")
    if not np.isfinite(dtau) or dtau <= 0:
        raise ValueError("paper dt must be finite and positive")
    wp, ratio = 1e9, 1836.0
    density = wp**2 * epsilon_0 * mass_electron / e**2
    length = 40 * c / wp
    x, _ = quiet_start(particles, length)
    rng = np.random.default_rng(seed)
    species = []
    for name, charge, mass in (("electrons", -1, mass_electron),
                               ("ions", 1, ratio * mass_electron)):
        velocity = rng.normal(size=particles)
        # Remove finite-sample bulk flow and match the specified initial variance.
        velocity = (velocity - velocity.mean()) / velocity.std()
        velocity *= c * np.sqrt(1e-3 * mass_electron / mass)
        v = jnp.zeros((particles, 3)).at[:, 0].set(jnp.asarray(velocity))
        species.append(Species(name, particles, charge, mass, density, x=x, v=v))
    return Simulation(Domain(length=length, cells=cells, time_step=dtau / wp),
                      tuple(species), Solver(relativistic=True)), wp


def array_fingerprint(array):
    """Exact host-array identity: dtype, shape and contiguous bytes, without storing it."""
    array = np.ascontiguousarray(array)
    digest = hashlib.sha256(f'{array.dtype.str}:{array.shape}'.encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def save_compressed_state(path, state, sim):
    """Keep complete replay states privately without uncompressed particle archives."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    archive = save_state(path, state, sim)
    with np.load(archive, allow_pickle=False) as stored:
        arrays = {key: stored[key] for key in stored.files}
    np.savez_compressed(archive, **arrays)


def paper_initial(sim, seed, initial_state):
    """Check a reused initial archive and record exact loading/state identities."""
    start, _ = sim.initial_state(random.PRNGKey(seed))
    if initial_state is not None:
        restored = load_state(initial_state, sim)
        for key in ('x', 'u', 'w'):
            np.testing.assert_array_equal(getattr(restored.ordinary, key), getattr(start.ordinary, key))
        if float(restored.ordinary.time) != 0 or float(restored.work) != 0:
            raise ValueError('paper initial archive must be a zero-time, zero-work state')
        start = restored
    fingerprints = dict(
        loading={key: array_fingerprint(np.concatenate([np.asarray(getattr(s, key)) for s in sim.plasma.species]))
                 for key in ('x', 'v')},
        state={key: array_fingerprint(getattr(start.ordinary, key)) for key in ('x', 'u', 'w', 'E', 'B', 'rho')})
    if isinstance(sim.dark, DarkField):
        fingerprints['state'].update({f'dark_{key}': array_fingerprint(getattr(start, key))
                                      for key in ('E', 'B', 'A', 'phi')})
    return start, fingerprints


def paper_run(sim, start, steps, stride, block_horizon, wp, scales, folder):
    """Compile one fixed interval, preserving the global reference across every block."""
    if block_horizon is not None and (not np.isfinite(block_horizon) or block_horizon <= 0):
        raise ValueError('paper block horizon must be finite and positive')
    block_steps = steps if block_horizon is None else round(block_horizon / sim.plasma.domain.dt / wp)
    if block_steps < 1 or block_steps % stride or steps % block_steps:
        raise ValueError('paper blocks must divide the run and contain complete output intervals')
    reference = snapshot(sim, start)
    save_compressed_state(Path(folder) / 'initial_state.npz', start, sim)
    print(f'🦇 Compiling {block_steps} paper steps per block', flush=True)
    before = time.perf_counter()
    executable = measured_run.lower(sim, start, block_steps, stride, reference, scales).compile()
    compile_seconds = time.perf_counter() - before
    memory = executable.memory_analysis()
    print(f'🦇 Compiled in {compile_seconds:.2f} s; advancing {steps // block_steps} blocks', flush=True)
    before = time.perf_counter()
    final, maxima, chunks = start, jnp.zeros(9), []
    for index in range(steps // block_steps):
        final, history, block_max = executable(sim, final, reference, scales)
        maxima = jnp.maximum(maxima, block_max)
        chunks.append({key: np.array(value)[int(index > 0):] for key, value in history.items()})
        if block_horizon is not None:
            print(f'🌒 block {index + 1}/{steps // block_steps}', flush=True)
    maxima.block_until_ready()
    warm_seconds = time.perf_counter() - before
    history = {key: np.concatenate([chunk[key] for chunk in chunks]) for key in chunks[0]}
    return final, history, maxima, block_steps, compile_seconds, warm_seconds, memory


def paper_case(folder, cells, particles, dtau, horizon, seed, ratio, eta=None,
               block_horizon=None, local_moments=False, initial_state=None):
    """Hook Fig. 2 drive with reduced moments, work and both constraint ledgers."""
    if not np.isfinite(horizon) or horizon <= 0 or not np.isfinite(ratio) or ratio < 0:
        raise ValueError("paper horizon must be positive and drive ratio nonnegative, both finite")
    if eta is not None and (not np.isfinite(eta) or eta <= 0):
        raise ValueError("paper finite-reservoir coupling must be finite and positive")
    plasma, wp = paper_plasma(cells, particles, dtau, seed)
    amplitude = ratio * np.sqrt(1e-3)
    field_scale = mass_electron * c * wp / e
    force = amplitude * field_scale
    model = (PrescribedDrive(1.0, jnp.array([force, 0., 0.]), wp) if eta is None
             else DarkField(wp, eta, initial_E=jnp.tile(
                 jnp.array([force / eta, 0., 0.]), (cells, 1))))
    sim = DarkSimulation(plasma, model)
    start, fingerprints = paper_initial(sim, seed, initial_state)
    stride = max(1, round(0.5 / dtau))
    steps = stride * max(1, round(horizon / (stride * dtau)))
    scales = np.array([2., 4.]) * np.sqrt(1e-3) * c / wp
    sampled_scales = jnp.asarray(scales) if local_moments else None
    final, history, maxima, block_steps, compile_seconds, warm_seconds, memory = paper_run(
        sim, start, steps, stride, block_horizon, wp, sampled_scales, folder)
    density = plasma.species[0].density
    energy_scale = density * mass_electron * c**2 * plasma.domain.length
    momentum_scale = energy_scale / c
    coarse_initial = np.asarray(coarse_spread(sim, start, scales)) / energy_scale
    coarse_final = np.asarray(coarse_spread(sim, final, scales)) / energy_scale
    t = history['t'] * wp
    history['t'] = t
    for key in ('electric', 'magnetic', 'dark', 'kinetic', 'spread', 'balance', 'work'):
        history[key] /= energy_scale
    if local_moments:
        history['local_spread'] /= energy_scale
    for key in ('mean', 'rms', 'max_speed'):
        history[key] /= c
    history['momentum'] /= momentum_scale
    history['mean_E'] /= field_scale
    history['mode_E'] /= field_scale
    history['nonzero_electric'] = history['electric'] - history['mean_E']**2 / 2
    maxima = np.asarray(maxima)
    # Spatially homogeneous kinetic orbits separate relativistic detuning from density waves.
    cold = forced_cold(t, amplitude)
    oracle = homogeneous(t, amplitude, eta=eta)
    early = t <= min(40, t[-1])
    scale = max(np.max(abs(oracle['mean_E'][early])), amplitude, 1e-30)
    late = t >= max(0, t[-1] - min(200, t[-1] / 5))
    results = dict(early_mean_error_over_homogeneous_peak=float(np.max(
        abs(history['mean_E'][early] - oracle['mean_E'][early])) / scale) if ratio else None,
        early_mean_error_over_cold_peak=float(np.max(
            abs(history['mean_E'][early] - cold[early])) / scale) if eta is None and ratio else None,
        max_energy_work_defect_over_initial_thermal=float(maxima[0] / (1e-3 * energy_scale)),
        max_momentum_defect_over_nmecL=float(maxima[1] / momentum_scale),
        max_particle_charge_change_over_enL=float(maxima[2] / (e * density * plasma.domain.length)),
        max_grid_charge_change_over_enL=float(maxima[6] / (e * density * plasma.domain.length)),
        max_continuity_over_enwp=float(maxima[3] / (e * density * wp)),
        max_ordinary_gauss_over_en_eps0=float(maxima[4] * epsilon_0 / (e * density)),
        max_dark_gauss_over_en_eps0=float(maxima[5] * epsilon_0 / (e * density)),
        late_electron_spread_over_initial=float(np.mean(history['spread'][late, 0]) / 5e-4),
        late_ion_spread_over_initial=float(np.mean(history['spread'][late, 1]) / 5e-4),
        late_electric_energy_over_initial_electron_thermal=float(np.mean(history['electric'][late]) / 5e-4),
        late_nonzero_electric_energy_over_initial_electron_thermal=float(
            np.mean(history['nonzero_electric'][late]) / 5e-4),
        local_spread_final_over_initial=(coarse_final / coarse_initial).tolist(),
        final_particle_kinetic_increment=float(np.sum(history['kinetic'][-1] - history['kinetic'][0])),
        final_homogeneous_particle_kinetic_increment=float(np.sum(oracle['kinetic'][-1] - oracle['kinetic'][0])),
        max_energy_work_defect_over_peak_injected_work=float(maxima[0] / max(
            np.max(abs(history['work'])) * energy_scale, 1e-30)) if eta is None and ratio else None,
        max_speed_over_c=float(np.max(history['max_speed'])), compile_s=compile_seconds,
        warm_primal_s=warm_seconds,
        compiler_temporary_MiB=memory.temp_size_in_bytes / 2**20 if memory is not None else None,
        compiler_arguments_MiB=memory.argument_size_in_bytes / 2**20 if memory is not None else None,
        compiler_outputs_MiB=memory.output_size_in_bytes / 2**20 if memory is not None else None,
        process_peak_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (
            2**20 if sys.platform == 'darwin' else 1024),
        claim='Fig. 2 parameter replay; late reproduction requires loading/grid/time/seed convergence')
    settings = dict(source='Hook, Huang, Shalaby arXiv:2510.13956v1 Fig. 2 and Appendix B',
                    cells=cells, particles_per_species=particles, dt_omega_p=dtau,
                    horizon_omega_p=float(t[-1]), seed=seed, length_c_over_omega_p=40,
                    mass_ratio=1836, T_each_over_mec2=1e-3, drive_quiver_over_sigma=ratio,
                    force_quiver_over_c=amplitude, coupling=eta, output_dt_omega_p=stride * dtau,
                    block_steps=block_steps, block_horizon_omega_p=block_steps * dtau,
                    local_moments_output_dt_omega_p=stride * dtau if local_moments else None,
                    initial_fingerprints=fingerprints,
                    initial_state_source=('complete zero-time archive' if initial_state is not None
                                          else 'native initialization'),
                    timing='one synchronized run including host scalar transfers; compilation reported separately',
                    normalization=dict(omega_p_rad_s=wp, field_scale_V_m=field_scale,
                                       energy_scale_J_m2=energy_scale, charge_density_C_m3=e * density,
                                       epsilon0_F_m=float(epsilon_0), c_m_s=float(c)),
                    local_spread_lengths_c_over_wp=(scales * wp / c).tolist(),
                    loading='co-located lattice; independent Gaussian velocities, zero mean and exact variance',
                    pusher='relativistic Boris; electric 1V uses the same momentum kick as Vay',
                    shape='quadratic parent spline; paper uses fifth-order',
                    inferred_t_noise=(40 / (amplitude * np.sqrt(3 * particles * cells))
                                      if amplitude else None),
                    parent_revision='83d327118163833f93e2588edcb5029241f6ba2a')
    with midnight():
        fig, axes = plt.subplots(2, 2, figsize=(10, 7), layout='constrained')
        axes[0, 0].plot(t, history['electric'] - history['electric'][0], label='electric change')
        for i, label in enumerate(('electron spread', 'ion spread')):
            axes[0, 0].plot(t, history['spread'][:, i] - history['spread'][0, i], label=label)
        if eta is None:
            axes[0, 0].plot(t, amplitude**2 * t**2 / 8, '--', color='#6A3D9A', label='linear t² envelope')
        axes[0, 0].plot(t, oracle['electric'], ':', color='#0072B2', label='homogeneous kinetic E')
        axes[0, 0].axhline(5e-4, color='#6B7280', ls=':', label='initial electron thermal')
        axes[0, 0].set_yscale('symlog', linthresh=1e-6)
        axes[0, 0].set(ylabel='energy / n mₑ c² L')
        for i, label in enumerate(('electrons', 'ions')):
            axes[0, 1].plot(t, history['rms'][:, i], label=label + ' RMS spread')
            axes[0, 1].plot(t, abs(history['mean'][:, i]), '--', label=label + ' |mean|')
            axes[1, 1].plot(t, history['density_rms'][:, i], label=label)
        sound_speed = np.sqrt((history['rms'][:, 0]**2 + 1836 * history['rms'][:, 1]**2) / 1836)
        axes[0, 1].plot(t, sound_speed / np.sqrt(2), ':', color='#009E73', label='sound speed / √2')
        axes[0, 1].set(yscale='log', ylabel='velocity / c', ylim=(np.sqrt(1e-3 / 1836) / 10, None))
        axes[1, 0].plot(t, (history['balance'] - history['balance'][0]) / 1e-3, label='energy − work')
        axes[1, 0].plot(t, history['momentum'][:, 0] - history['momentum'][0, 0], label='momentum / n mₑ c L')
        axes[1, 0].set(ylabel='conservation defects')
        axes[1, 1].set(ylabel='grid-scale density RMS / mean')
        for ax in axes.flat:
            ax.set(xlabel=r'$\omega_p t$')
            ax.grid(alpha=0.25)
            ax.legend(fontsize=8)
        reference = {f'homogeneous_{key}': oracle[key] for key in (
            'mean_E', 'mean_D', 'mean_A', 'mean', 'rms', 'kinetic', 'spread', 'electric', 'dark', 'work', 'balance')}
        arrays = dict(linear_mean_E=cold, local_spread_initial=coarse_initial,
                      local_spread_final=coarse_final, **reference, **history)
        save_run(folder, 'paper_resonant_conversion', settings, results, fig, **arrays)
        # Scalar histories compress well; retain the native example/provenance path.
        np.savez_compressed(Path(folder) / 'data.npz', **arrays)
        plt.close(fig)
    save_compressed_state(Path(folder) / 'final_state.npz', final, sim)
    print('🌘 PAPER REPLAY:', results, flush=True)
    return history, settings, results


if __name__ == "__main__":
    if study == 'paper':
        with elapsed_progress("Resonant replay"):
            paper_case(output, cells, particles, dt, horizon, seed, drive_ratio,
                       coupling, block_horizon, local_moments, initial_state)
    elif study == 'pair':
        pair_figure(output, full)
    elif study == 'pair_dark':
        pair_dark_figure(output, full)
    elif study == 'paper_pilot':
        paper_geometry_pilot(output)
    elif study != 'mobile_ions':
        raise ValueError('study must be mobile_ions, paper, pair, pair_dark or paper_pilot')
    else:
        presets = ((64, 4000, 0.04, 40), (128, 8000, 0.02, 40)) if full else (
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
            settings = {**settings, "preset": "full" if full else "quick",
                        "refinements": [setting.copy() for _, setting in runs]}
            save_run(output, "dark_reservoir", settings, results, fig, t=t,
                     **{f"{name}_{key}": record[key] for name, record in histories.items()
                        for key in ("pump", "mean_E", "mode", "cold_mean")})
            plt.close(fig)
        print("🦇 MOBILE IONS: mean-oracle errors", results["refinements"],
              "; finite-reservoir energy balances",
              [results["cases"][name]["balance_over_scale"]
               for name in ("small_reservoir", "large_reservoir")])
