"""A mobile-ion pump and the finite reservoir behind its ghostly imitation.

This bounded-time, seeded control is not the long SHARP reproduction.  The
prescribed case has an external reservoir; the Proca cases start with the
same force but different finite field energies.  All share one loading.

Edit the inputs below or supply them with ``runpy.run_path(...,
run_name='__main__', init_globals={...})``. ``full=True`` selects the full
study preset; the paper preset uses 103,000 markers per species through 5,000.
"""

import hashlib
import json
import os
from pathlib import Path
import resource
import shlex
import time

import jax.numpy as jnp
from jax import core, random
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import BSpline
from scipy.linalg import expm
from scipy.stats import linregress
import sys

from jaxincell import (Domain, Simulation, Solver, Species, elementary_charge as e,
                       epsilon_0, mass_electron, mass_proton, quiet_start,
                       provenance, save_run, speed_of_light as c)
from jaxincell._simulation import _van_der_corput
from darkjaxincell import DarkField, DarkSimulation, PrescribedDrive, load_state, midnight, save_state
from darkjaxincell._proca import energy as dark_energy

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "docs" / "scripts"))
from pair_reference import (coupled_response, growth, relativistic_background, relativistic_response,
                            seeded_response)  # noqa: E402
from conservation import coarse_spread, elapsed_progress, measured_run, parent_revision, snapshot  # noqa: E402
from drive_reference import forced_cold, gaussian_tangent, homogeneous  # noqa: E402


# Inputs: the default is a small mobile-ion comparison.
study = globals().get('study', 'mobile_ions')  # pair_table varies one archived prescribed branch.
full = globals().get('full', False)
output = Path(globals().get('output', 'artifacts/dark_reservoir'))
# Paper controls use omega_p time units and markers per species.
cells = globals().get('cells', (4096 if full else 512) if study == 'pair_waveform' else 1000)
particles = globals().get('particles', 103000 if full else 20000)
dt = globals().get('dt', (0.00625 if full else 0.025) if study == 'pair_waveform' else 0.02)
horizon = globals().get('horizon', (170 if full else 10) if study == 'pair_waveform' else (5000 if full else 40))
seed = globals().get('seed', 0)
drive_ratio = globals().get('drive_ratio', 0.03)
coupling = globals().get('coupling', None)  # None selects the prescribed drive.
block_horizon = globals().get('block_horizon', min(10 if study == 'pair_waveform' else 100, horizon))
local_moments = globals().get('local_moments', full and study == 'pair_waveform')
initial_state = globals().get('initial_state', None)
initial_state = None if initial_state is None else Path(initial_state)
samples = globals().get('samples', 1)  # pair_repeat executions reuse one exact archived state/table.
table_every = globals().get('table_every', 1)  # pair_table retains every nth archived midpoint.
observe_executable = globals().get('observe_executable', False)  # Host-only repeat compilation evidence.
momentum_seed = globals().get('momentum_seed', 0.)  # Electron δu / initial σe; zero keeps the paper loading.
seed_mode = globals().get('seed_mode', 16)
seed_phase = globals().get('seed_phase', 0.)
shape_order = globals().get('shape_order', 2)  # 5 selects the optional parent quintic weighting.
# Pair-waveform controls: one loading, a coupled run and two refined external-force interventions.
particles_per_cell = globals().get('particles_per_cell', 128 if full else 8)
pair_loading = globals().get('pair_loading', 'global' if study == 'pair_waveform' else 'cell')
velocity_seed = globals().get('velocity_seed', 2e-4)
eta = globals().get('eta', 0.5)
dark_mass = globals().get('dark_mass', 1.)
force = globals().get('force', 0.05)
quadrature = globals().get('quadrature', 64)
table_dt = globals().get('table_dt', 2 * dt)
output_dt = globals().get('output_dt', 0.2)
linear_end = globals().get('linear_end', min(20., horizon))
scalar_dt = globals().get('scalar_dt', None)  # Cheaper moments; push-mean forcing stays at every step.


def pair_plasma(cells, particles_per_cell, dtau, seed, relativistic=False, shape_order=2, pair_loading='cell'):
    """Load matched neutral waterbags for ordinary and dark pair runs."""
    omega_0 = 1e9
    omega_p = omega_0 / np.sqrt(2)
    density = omega_p**2 * epsilon_0 * mass_electron / e**2
    length = 70 * c / omega_0
    half_width = 0.05
    mode = round(0.6 * 70 / (2 * np.pi * half_width))
    if 2 * mode >= cells:
        raise ValueError('pair seed must lie below the mesh Nyquist mode')
    k = 2 * np.pi * mode / length
    count = cells * particles_per_cell
    cell, slot = np.divmod(np.arange(count), particles_per_cell)
    x = -length / 2 + (cell + (slot + 0.5) / particles_per_cell) * length / cells
    # A coprime permutation spreads the waterbag quantiles within every cell.
    order = (slot * (particles_per_cell - 1)) % particles_per_cell
    waterbag = half_width * c * (2 * (order + 0.5) / particles_per_cell - 1)
    if pair_loading == 'global':
        half_count = count // 2
        if particles_per_cell % 2 or half_count < 1 or half_count & (half_count - 1):
            raise ValueError('global pair loading requires even markers per cell and a power-of-two half-count')
        rank = np.floor(_van_der_corput(half_count, 2) * half_count)
        speed = half_width * c * (rank + .5) / half_count
        waterbag = np.column_stack((-speed, speed)).ravel()
    elif pair_loading != 'cell':
        raise ValueError('pair_loading must be cell or global')
    positions = jnp.stack((jnp.asarray(x), jnp.zeros(count), jnp.zeros(count)), axis=1)
    species = []
    for name, charge in (("electrons", -1), ("positrons", 1)):
        velocity = jnp.stack((jnp.asarray(waterbag + charge * seed * c * np.cos(k * x)),
                              jnp.zeros(count), jnp.zeros(count)), axis=1)
        species.append(Species(name, count, charge, mass_electron, density,
                               x=positions, v=velocity))
    domain = Domain(length=length, cells=cells, time_step=dtau / omega_0)
    if shape_order not in (2, 5):
        raise ValueError('pair shape_order must be 2 or 5')
    solver = (Solver(relativistic=relativistic) if shape_order == 2
              else Solver(relativistic=relativistic, shape_order=5))
    plasma = Simulation(domain, tuple(species), solver)
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
                "parent_revision": parent_revision(),
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
    homogeneous_dark_fraction = (background[1]**2 + mass**2 * background[2]**2) / reservoir_scale
    total_dark_fraction = np.asarray(energy['dark']) / initial_energy
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
              "late_mean_dark_fraction": float(np.mean(mean_dark_energy[t >= t[-1] - 20]) / initial_energy),
              "late_total_dark_fraction": float(np.mean(total_dark_fraction[t >= t[-1] - 20])),
              "late_additional_dark_depletion": float(np.mean(
                  (homogeneous_dark_fraction - total_dark_fraction)[t >= t[-1] - 20])),
              "late_random_gain_over_initial_reservoir": float(
                  np.mean(random_gain[t >= t[-1] - 20])),
              "half_coherence_time_omega0": float(t[crossing[0]]) if len(crossing) else None,
              "max_final_speed_over_c": float(np.max(abs(final_velocity)) / c),
              "initial_dark_over_matched_ordinary_field_energy": 1 / eta**2}
    settings = {"model": "neutral waterbag pair plasma with a bare homogeneous Proca reservoir",
                "parent_revision": parent_revision(),
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
                mean_dark_fraction=mean_dark_energy / initial_energy, total_dark_fraction=total_dark_fraction,
                homogeneous_dark_fraction=homogeneous_dark_fraction,
                dark_source_work=np.asarray(energy['dark_source_work']) / initial_energy,
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


def pair_push_table(plasma, start, history, omega, eta):
    """Bare mean dark E actually used by each Boris push, in native SI units.

    Periodic continuity closes the first half-step current to the old particle
    current. Integer-time D and A alone would miss that source/mass half kick.
    The end knots cover the run; only the midpoint values enter the pusher.
    """
    number_density, offset = [], 0
    for species in plasma.species:
        number_density.append(float(np.sum(np.asarray(start.w[offset:offset + species.n])))
                              / plasma.domain.length)
        offset += species.n
    charge_density = np.array(number_density) * np.array([float(s.charge_si) for s in plasma.species])
    current = np.asarray(history['mean']) @ charge_density
    dt = plasma.domain.dt
    if not np.allclose(np.diff(history['t']), dt, rtol=1e-8, atol=0):
        raise ValueError('push-mean reconstruction requires every native integer-time sample')
    push = history['mean_D'][:-1] + dt / 2 * (
        omega**2 * history['mean_A'][:-1] - eta * current[:-1] / epsilon_0)
    times = np.r_[history['t'][0], history['t'][:-1] + dt / 2, history['t'][-1]]
    return times, np.r_[history['mean_D'][0], push, history['mean_D'][-1]]


def pair_waveform_normalize(values, label, energy_scale, field_scale, wp, phase):
    """Normalize scalar histories after native evidence has been saved."""
    values['t'] *= wp
    for key in ('electric', 'magnetic', 'dark', 'dark_coherent', 'kinetic', 'spread', 'balance', 'work'):
        values[key] /= energy_scale
    if 'local_spread' in values:
        values['local_spread'] /= energy_scale
    for key in ('mean', 'rms', 'max_speed'):
        values[key] /= c
    values['momentum'] /= energy_scale / c
    for key in ('mean_E', 'mean_D', 'mode_E', 'dark_mode_E'):
        values[key] /= field_scale
    values['mean_A'] *= wp / field_scale
    values['mode_E'] = values['mode_E'] * phase
    values['dark_mode_E'] = values['dark_mode_E'] * phase
    values['nonzero_electric'] = values['electric'] - .5 * values['mean_E']**2
    values['source_work'] = (-1 if label == 'coupled' else 1) * values['work']


def pair_waveform_windows(curves, maxima, energy_scale, horizon, local_moments, dark_depletion):
    """Fixed raw-clock comparisons, retaining absolute norms for vanishing signals."""
    from compare_replays import metrics, window_summary

    windows = sorted(set([(0., min(20., horizon)), (max(0., horizon - 20.), horizon)]
                         + ([(20., min(40., horizon))] if horizon > 20 else [])))
    comparisons = {}
    keys = ('mean_E', 'mode_E', 'electric', 'nonzero_electric', 'source_work', 'kinetic', 'spread', 'density_rms')
    if local_moments:
        keys += ('local_spread', 'local_density_rms')
    for bounds in windows:
        selected = (curves['coupled']['t'] >= bounds[0] - 1e-8) & (curves['coupled']['t'] <= bounds[1] + 1e-8)
        reductions = {label: window_summary(values, selected, label == 'coupled')
                      for label, values in curves.items()}
        reductions['coupled'].update(
            additional_dark_depletion_fraction_mean=float(np.mean(dark_depletion[selected])),
            additional_dark_depletion_gain_mean=float(np.mean(dark_depletion[selected] - dark_depletion[0])))
        paired = {}
        for first, second in (('coupled', 'realized_fine'), ('coupled', 'homogeneous_fine'),
                              ('realized_fine', 'homogeneous_fine'),
                              ('realized_coarse', 'realized_fine'), ('homogeneous_coarse', 'homogeneous_fine')):
            if not np.allclose(curves[first]['t'], curves[second]['t'], rtol=0, atol=1e-8):
                raise ValueError('waveform runs do not share native scalar clocks')
            comparison = {key: metrics(curves[first][key][selected], curves[second][key][selected]) for key in keys}
            for key in ('reference_mean', 'comparison_mean'):
                mean = comparison['mode_E'][key]
                comparison['mode_E'][key] = dict(real=float(np.real(mean)), imag=float(np.imag(mean)))
            transfer = max(abs(reductions[first]['work_increment']), abs(reductions[second]['work_increment']))
            difference = abs(reductions[first]['plasma_mean_energy'] - reductions[second]['plasma_mean_energy'])
            defect = max(maxima[first][0], maxima[second][0], maxima[first][8], maxima[second][8]) / energy_scale
            comparison['balance_defect_over_window_transfer'] = float(defect / transfer) if transfer else None
            comparison['balance_defect_over_target_energy_difference'] = (
                float(defect / difference) if difference else None)
            paired[first + '_vs_' + second] = comparison
        comparisons[str(bounds)] = dict(bounds_omega0=list(bounds), samples=int(np.sum(selected)),
                                        reductions=reductions, comparisons=paired)
    return comparisons


def pair_waveform_plot(curves, indices, reservoir, linear):
    """White paired histories at the same saved native clocks."""
    with midnight():
        fig, axes = plt.subplots(2, 2, figsize=(10, 7), layout='constrained')
        for label, display in (('coupled', 'coupled'), ('realized_fine', 'realized dark mean force'),
                               ('homogeneous_fine', 'independent warm envelope')):
            values = curves[label]
            time = values['t'][indices]
            axes[0, 0].plot(time, values['mean_E'][indices], label=display)
            positive = time > 0  # The exact zero-time mode has no logarithm; retain it in the data.
            axes[0, 1].plot(time[positive], abs(values['mode_E'][indices][positive]), label=display)
            sign = -1 if label == 'coupled' else 1
            axes[1, 0].plot(time, sign * values['work'][indices] / reservoir, label=display)
            axes[1, 1].plot(time, (values['balance'][indices] - values['balance'][0]) / reservoir, label=display)
        selected = linear['linear_t'] > 0
        for label, color in (('coupled', '#6A3D9A'), ('homogeneous_fine', '#D55E00')):
            axes[0, 1].plot(linear['linear_t'][selected], abs(linear[f'linear_{label}_mode_E'][selected]),
                            '--', color=color, lw=1, label=f'{label.split("_")[0]} linear reference')
        axes[0, 0].set(ylabel=r'$\overline{E}/E_\star$', title='Mean electric field')
        axes[0, 1].set(ylabel=r'$|\widehat{E}_{134}|/E_\star$', title='Seeded spatial mode', yscale='log')
        axes[1, 0].set(ylabel=r'$W_{\rm ordinary}/U_D(0)$', title='Transfer into ordinary plasma')
        axes[1, 1].set(ylabel=r'$\Delta(U-W_{\rm ext})/U_D(0)$', title='Energy and external work')
        for ax in axes.flat:
            ax.set(xlabel=r'$\omega_0 t$')
            ax.grid(alpha=.25)
            ax.legend(fontsize=8)
        return fig


def pair_waveform_linear(curves, indices, k, seed, eta, mass, force, quadrature, linear_end=20.):
    """Early complex modes on one warm envelope; their difference can already be linear."""
    if not np.isfinite(linear_end) or linear_end <= 0:
        raise ValueError('linear reference horizon must be finite and positive')
    selected = curves['coupled']['t'] <= min(20., linear_end) + 1e-8
    times = curves['coupled']['t'][selected]
    if len(times) < 2:
        raise ValueError('linear comparison needs at least two native clocks')
    saved = indices[selected[indices]]
    args = (times, k, seed, eta, mass, 0., force / eta)
    arrays, results, references = dict(linear_t=curves['coupled']['t'][saved]), {}, {}
    for label, mixing in (('coupled', eta), ('homogeneous_fine', 0.)):
        coarse = relativistic_response(*args, nodes=quadrature, spatial_eta=mixing, rtol=2e-10, atol=2e-12)[0]
        fine = relativistic_response(*args, nodes=2 * quadrature, spatial_eta=mixing, rtol=2e-10, atol=2e-12)[0]
        loose = relativistic_response(*args, nodes=2 * quadrature, spatial_eta=mixing)[0]
        measured = curves[label]['mode_E'][selected]
        norm, difference = np.linalg.norm(fine), np.linalg.norm(measured - fine)
        results[label] = dict(absolute_l2_error=float(difference), reference_l2_norm=float(norm),
                              relative_l2_error=float(difference / norm) if norm else None,
                              max_absolute_error=float(np.max(abs(measured - fine))),
                              quadrature_absolute_l2_difference=float(np.linalg.norm(fine - coarse)),
                              tolerance_absolute_l2_difference=float(np.linalg.norm(fine - loose)))
        arrays[f'linear_{label}_mode_E'] = fine[saved]
        references[label] = fine
    results.update(horizon_omega0=float(times[-1]), nodes=[quadrature, 2 * quadrature],
                   perturbation_tolerances=[[2e-8, 2e-10], [2e-10, 2e-12]],
                   interbranch_absolute_l2_difference=float(np.linalg.norm(
                       references['coupled'] - references['homogeneous_fine'])),
                   qualification='Removing finite-k dark force changes linear dynamics on the same warm envelope; '
                   'a replay difference alone is not evidence of a nonlinear mechanism')
    return arrays, results


def waveform_scalar_dt(dtau, scalar_dt, output_dt, early_end):
    """Preserve dense defaults; explicit sparse moments must share output clocks."""
    if scalar_dt is None:
        return dtau
    if (not np.isfinite(scalar_dt) or scalar_dt < dtau
            or not np.isclose(scalar_dt / dtau, round(scalar_dt / dtau), rtol=0, atol=1e-10)
            or output_dt < scalar_dt
            or not np.isclose(output_dt / scalar_dt, round(output_dt / scalar_dt), rtol=0, atol=1e-10)):
        raise ValueError('scalar_dt must divide output_dt and be an integer multiple of the native timestep')
    if scalar_dt > early_end:
        raise ValueError('scalar_dt must retain at least two clocks in the early reference interval')
    return round(scalar_dt / dtau) * dtau


def pair_waveform_initial(sim, initial_state, field_scale):
    """Restore the exact zero-clock preparation, including its diagnostic ledger."""
    if initial_state is None:
        return sim.initial_state(random.PRNGKey(0))[0], dict(initial_state_source='native initialization')
    start, _ = paper_initial(sim, 0, initial_state)
    if not np.allclose(np.mean(np.asarray(start.E), axis=0), np.mean(np.asarray(sim.dark.initial_E), axis=0),
                       rtol=2e-12, atol=2e-14 * field_scale):
        raise ValueError('waveform initial archive has a different homogeneous dark force')
    return start, dict(initial_state_source='complete zero-time archive',
                       initial_state_archive_sha256=hashlib.sha256(Path(initial_state).read_bytes()).hexdigest())


def pair_all_step_maxima(maximum, plasma, wp, energy_scale):
    """The same nine physical conservation scales for each waveform branch."""
    density = sum(float(s.density) for s in plasma.species)
    return dict(
        energy_work_over_energy_scale=float(maximum[0] / energy_scale),
        momentum_over_energy_scale_over_c=float(maximum[1] / (energy_scale / c)),
        charge_over_enL=float(maximum[2] / (e * density * plasma.domain.length)),
        continuity_over_enomega0=float(maximum[3] / (e * density * wp)),
        ordinary_gauss_over_en_eps0=float(maximum[4] * epsilon_0 / (e * density)),
        dark_gauss_over_en_eps0=float(maximum[5] * epsilon_0 / (e * density)),
        grid_charge_over_enL=float(maximum[6] / (e * density * plasma.domain.length)),
        dark_sector_work_over_energy_scale=float(maximum[7] / energy_scale),
        ordinary_sector_work_over_energy_scale=float(maximum[8] / energy_scale))


def pair_xla_flags():
    """Record compiler flags without exporting configuration paths."""
    return ' '.join(token if '/' not in token and '\\' not in token
                    else token.split('=')[0] + '=<path omitted>'
                    for token in shlex.split(os.environ.get('XLA_FLAGS', '')))


def pair_repeat_initial(initial_state):
    """Validate a donor branch and restore its complete state and unscaled force."""
    from compare_replays import PAIR_RUNTIME, _pair_branch_contract

    if initial_state is None:
        raise ValueError('pair_repeat requires an initial_state archive from a prescribed waveform branch')
    path = Path(initial_state)
    record = json.loads((path.parent / 'run.json').read_text())
    donor = json.loads((path.parent.parent / 'run.json').read_text())
    setting = dict(record['settings'])
    label = setting['label']
    if (record['example'] != 'pair_waveform_branch' or donor['example'] != 'pair_waveform_control'
            or label not in ('realized_coarse', 'realized_fine', 'homogeneous_coarse', 'homogeneous_fine')):
        raise ValueError('pair_repeat requires an archived prescribed waveform branch')
    _pair_branch_contract(record, donor, label)
    runtime = provenance()
    if (any(runtime[key] != record[key] for key in PAIR_RUNTIME if key != 'git')
            or pair_xla_flags() != setting['XLA_FLAGS']
            or parent_revision() != setting['parent_revision']):
        raise ValueError('pair_repeat requires the donor runtime and imported parent revision')
    setting.update(local_moments=donor['settings']['local_moments'],
                   local_spread_lengths_c_over_omega0=donor['settings']['local_spread_lengths_c_over_omega0'])
    plasma, _, k, wp = pair_plasma(
        setting['cells'], setting['particles_per_cell_per_species'], setting['dt_omega0'],
        setting['velocity_seed_over_c'], True, setting['shape_order'], setting['pair_loading'])
    scale = setting['normalization']
    expected = [wp, mass_electron * c * wp / e, epsilon_0 * plasma.domain.length * (mass_electron * c * wp / e)**2]
    if (not np.allclose([scale[key] for key in ('omega0_rad_s', 'field_scale_V_m', 'energy_scale_J_m2')],
                        expected, rtol=2e-12, atol=0)
            or setting['seed_mode'] != round(k * plasma.domain.length / (2 * np.pi))):
        raise ValueError('pair_repeat donor normalization or selected mode differs from the physical loading')
    with np.load(path, allow_pickle=False) as archive:
        if str(archive['dark.mode']) != 'drive' or 'dark.times' not in archive or archive['dark.eta'] != setting['eta']:
            raise ValueError('pair_repeat needs the declared tabulated prescribed drive')
        drive = PrescribedDrive(float(archive['dark.eta']), jnp.asarray(archive['dark.amplitude']),
                                float(archive['dark.omega']), float(archive['dark.phase']),
                                times=jnp.asarray(archive['dark.times']))
    sim = DarkSimulation(plasma, drive)
    start = load_state(path, sim)
    if float(start.ordinary.time) != 0 or float(start.ordinary.steps) != 0 or float(start.work) != 0:
        raise ValueError('pair_repeat requires a zero-time, zero-step, zero-work archive')
    hashes = {key: array_fingerprint(getattr(start.ordinary, key)) for key in ('x', 'u', 'w', 'E', 'B', 'rho')}
    if hashes != record['results']['initial_ordinary_fingerprints']:
        raise ValueError('pair_repeat initial ordinary arrays differ from the donor fingerprints')
    setting.update(initial_state_source='complete archived prescribed branch', donor_git=record['git'],
                   **{key: hashlib.sha256(file.read_bytes()).hexdigest() for key, file in (
                       ('initial_state_archive_sha256', path), ('donor_record_sha256', path.parent / 'run.json'),
                       ('donor_control_record_sha256', path.parent.parent / 'run.json'),
                       ('producer_script_sha256', Path(__file__)))})
    return sim, start, setting


def pair_decimated_drive(sim, setting, every, steps):
    """Vary only the archived piecewise-linear force, retaining its first impulse."""
    if every == 1:
        return sim
    if not setting['label'].endswith('_fine'):
        raise ValueError('table decimation requires a fine prescribed donor')
    times, amplitude = np.asarray(sim.dark.times), np.asarray(sim.dark.amplitude)
    expected, clock, dt = np.empty(steps + 2), 0., float(sim.plasma.domain.dt)
    expected[0] = 0.
    for index in range(steps):
        expected[index + 1] = clock + dt / 2
        clock += dt
    expected[-1] = clock
    if not np.array_equal(times, expected):
        raise ValueError('fine donor must retain every accepted native force midpoint and endpoint')
    selected = np.unique(np.r_[0, np.arange(1, len(times) - 1, every), len(times) - 1])
    kept_t, kept_a = times[selected], amplitude[selected]
    model = PrescribedDrive(sim.dark.eta, jnp.asarray(kept_a), sim.dark.omega,
                            sim.dark.phase, times=jnp.asarray(kept_t))
    setting['table_replay'] = dict(
        table_every=every, original_knots=len(times), retained_knots=len(kept_t),
        maximum_gap_omega0=float(np.max(np.diff(kept_t))) * setting['normalization']['omega0_rad_s'],
        original_times_sha256=array_fingerprint(times), original_amplitude_sha256=array_fingerprint(amplitude),
        times_sha256=array_fingerprint(kept_t), amplitude_sha256=array_fingerprint(kept_a),
        selection='initial endpoint, first midpoint, every nth midpoint, terminal endpoint',
        intervention='Same PIC state/timestep; piecewise-linear prescribed force changed.')
    return DarkSimulation(sim.plasma, model)


def pair_repeat(folder, initial_state, samples=1, observe_executable=False, table_every=1):
    """Replay the archived force, optionally varying its knots at unchanged PIC clocks."""
    if (type(samples) is not int or samples < 1 or type(table_every) is not int or table_every < 1
            or table_every > 1 and samples != 1):
        raise ValueError('samples/table_every must be positive integers; varied-table replays use one execution')
    if type(observe_executable) is not bool:
        raise ValueError('observe_executable must be a boolean')
    sim, start, setting = pair_repeat_initial(initial_state)
    wp, energy_scale = [setting['normalization'][key] for key in ('omega0_rad_s', 'energy_scale_J_m2')]
    dtau, horizon, cadence = [setting[key] for key in ('dt_omega0', 'horizon_omega0', 'scalar_dt_omega0')]
    scales = np.asarray(setting['local_spread_lengths_c_over_omega0']) * c / wp
    if not np.isfinite([dtau, horizon, cadence]).all() or min(dtau, horizon, cadence) <= 0:
        raise ValueError('pair_repeat clocks must be finite and positive')
    steps, stride, blocks = round(horizon / dtau), round(cadence / dtau), setting['block_steps']
    if (steps < 2 or stride < 1 or type(blocks) is not int or blocks < 1 or steps % blocks or blocks % stride
            or not np.allclose([steps * dtau, stride * dtau], [horizon, cadence], rtol=0, atol=1e-10)
            or scales.ndim != 1 or not np.isfinite(scales).all() or np.any(scales <= 0)):
        raise ValueError('pair_repeat requires complete native blocks, scalar clocks and forcing coverage')
    accepted_time, dt = 0., float(sim.plasma.domain.dt)
    for _ in range(steps):  # Match the native floating-point clock without allocating a time history.
        accepted_time += dt
    if sim.dark.times[0] != 0 or np.nextafter(float(sim.dark.times[-1]), np.inf) < accepted_time:
        raise ValueError('pair_repeat forcing table must cover the accepted native endpoint')
    sim = pair_decimated_drive(sim, setting, table_every, steps)
    histories, loaded = [], []
    observer = dict(source=provenance()['git'],
                    producer_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    method='Observe lower/compile; return the compiled object unchanged. No traced operations added.',
                    timing_note='Observer text hashing is outside public compilation and execution timings.',
                    interpretation='Runtime object identity is local to this process. '
                                   'HLO text is not binary serialization.',
                    compilations=[])

    def observe(lowered, executable):
        before = time.perf_counter()
        runtime, text = executable.runtime_executable(), executable.as_text()
        observer['compilations'].append(dict(
            index=len(loaded) + 1, runtime_available=runtime is not None,
            same_loaded_executable_as_first=bool(loaded and runtime is not None and runtime is loaded[0]),
            stablehlo_sha256=hashlib.sha256(lowered.as_text().encode()).hexdigest(),
            optimized_hlo_text_sha256=hashlib.sha256(text.encode()).hexdigest() if text is not None else None,
            observer_seconds=time.perf_counter() - before))
        loaded.append(runtime)  # Retain references so Python identities cannot be recycled.

    for index in range(samples):
        destination = Path(folder) / f'execution_{index + 1}' if samples > 1 else Path(folder)
        action = 'table replay' if table_every > 1 else 'archived repeat'
        with elapsed_progress(f"{setting['label']} {action} {index + 1}/{samples}"):
            final, values, maximum, _, compile_s, warm_s, memory = paper_run(
                sim, start, steps, stride, blocks * dtau, wp, jnp.asarray(scales) if setting['local_moments'] else None,
                destination, setting['seed_mode'], compilation_observer=observe if observe_executable else None)
        save_compressed_state(destination / 'final_state.npz', final, sim)
        results = dict(compile_s=compile_s, warm_primal_s=warm_s,
                       compiler_temporary_MiB=memory.temp_size_in_bytes / 2**20 if memory else None,
                       all_step_maxima=pair_all_step_maxima(maximum, sim.plasma, wp, energy_scale),
                       all_step_maxima_SI=np.asarray(maximum).tolist(),
                       initial_ordinary_fingerprints={key: array_fingerprint(getattr(start.ordinary, key))
                                                      for key in ('x', 'u', 'w', 'E', 'B', 'rho')},
                       local_spread_initial=(np.asarray(coarse_spread(sim, start, scales)) / energy_scale).tolist(),
                       local_spread_final=(np.asarray(coarse_spread(sim, final, scales)) / energy_scale).tolist(),
                       execution_index=index + 1, samples=samples,
                       claim=('Archived state/timestep; prescribed table changed, not an exact-force repeat.'
                              if table_every > 1 else
                              'Exact archived force/state; donor and producer sources are recorded separately.'),
                       timing_note=('One postcompile execution per call; local executable observations are separate.'
                                    if observe_executable else
                                    'One postcompile execution per call; executable identity is not asserted.'))
        save_run(destination, 'pair_waveform_table_replay' if table_every > 1 else 'pair_waveform_repeat',
                 setting, results, **values)
        np.savez_compressed(destination / 'data.npz', **values)
        histories.append(dict(history=values, settings=setting, results=results))
    if observe_executable:
        observer['completed'] = len(observer['compilations']) == samples
        (Path(folder) / 'executable_observer.json').write_text(json.dumps(observer, indent=2) + '\n')
    return histories


def pair_waveform_control(folder, cells, particles_per_cell, dtau, horizon,
                          seed=2e-4, eta=.5, mass=1., force=.05, quadrature=64,
                          table_dt=None, output_dt=.2, block_horizon=None,
                          local_moments=False, shape_order=2, pair_loading='global', linear_end=20., scalar_dt=None,
                          initial_state=None):
    """Separate finite-k dark feedback from the realized homogeneous envelope.

    The coupled-mean replay removes spatial dark forces as an intervention.
    The independent continuum-envelope replay also changes the background;
    its difference alone cannot identify spatial feedback. Tables refine at
    fixed native PIC clocks. Velocity-ring refinement remains a separate gate.
    """
    table_dt = 2 * dtau if table_dt is None else table_dt
    if (min(dtau, horizon, table_dt, output_dt, eta, mass, force, linear_end) <= 0
            or not np.all(np.isfinite([dtau, horizon, table_dt, output_dt, eta, mass, force, seed, linear_end]))
            or cells < 1 or particles_per_cell < 2 or quadrature < 2 or min(20., horizon, linear_end) < dtau):
        raise ValueError('positive finite waveform inputs and resolved particle/quadrature counts are required')
    scalar_dt = waveform_scalar_dt(dtau, scalar_dt, output_dt, min(20., horizon, linear_end))
    stride = round(scalar_dt / dtau)
    steps = round(horizon / dtau)
    if steps < 2 or not np.isclose(steps * dtau, horizon, rtol=0, atol=1e-10):
        raise ValueError('waveform horizon must contain an integer number of at least two native steps')
    plasma, _, k, wp = pair_plasma(cells, particles_per_cell, dtau, seed, True, shape_order, pair_loading)
    mode = round(k * plasma.domain.length / (2 * np.pi))
    if 2 * mode >= cells:
        raise ValueError('pair seed must lie below the mesh Nyquist mode')
    field_scale = mass_electron * c * wp / e
    energy_scale = epsilon_0 * plasma.domain.length * field_scale**2
    nominal_reservoir = .5 * (force / eta)**2
    scales = np.array([.1, .2]) * c / wp
    sampled_scales = jnp.asarray(scales) if local_moments else None
    model = DarkField(mass * wp, eta, initial_E=jnp.tile(
        jnp.array([force * field_scale / eta, 0., 0.]), (cells, 1)))
    coupled = DarkSimulation(plasma, model)
    start, initial_source = pair_waveform_initial(coupled, initial_state, field_scale)
    reservoir = float(snapshot(coupled, start)['dark']) / energy_scale
    phase = np.exp(-1j * k * float(plasma.domain.faces[0]))
    flags = pair_xla_flags()
    initial_hashes = {key: array_fingerprint(getattr(start.ordinary, key))
                      for key in ('x', 'u', 'w', 'E', 'B', 'rho')}
    curves, records, maxima = {}, {}, {}

    def advance(label, sim, initial):
        with elapsed_progress(label):
            final, values, maximum, blocks, compile_s, warm_s, memory = paper_run(
                sim, initial, steps, stride, block_horizon, wp, sampled_scales, Path(folder) / label, mode,
                pump=label == 'coupled' and stride > 1)
        save_compressed_state(Path(folder) / label / 'final_state.npz', final, sim)
        endpoint = np.asarray(coarse_spread(sim, final, scales)) / energy_scale
        initial_local = np.asarray(coarse_spread(sim, initial, scales)) / energy_scale
        records[label] = dict(compile_s=compile_s, warm_primal_s=warm_s,
                              compiler_temporary_MiB=memory.temp_size_in_bytes / 2**20 if memory else None,
                              block_steps=blocks, local_spread_initial=initial_local.tolist(),
                              local_spread_final=endpoint.tolist(),
                              initial_ordinary_fingerprints={key: array_fingerprint(
                                  getattr(initial.ordinary, key)) for key in initial_hashes})
        maxima[label] = np.asarray(maximum)
        records[label]['all_step_maxima'] = pair_all_step_maxima(maximum, plasma, wp, energy_scale)
        case_settings = dict(label=label, model=type(sim.dark).__name__, parent_revision=parent_revision(),
                             cells=cells, particles_per_cell_per_species=particles_per_cell, pair_loading=pair_loading,
                             dt_omega0=dtau, horizon_omega0=horizon, shape_order=shape_order, seed_mode=mode,
                             scalar_dt_omega0=scalar_dt, forcing_native_dt_omega0=dtau,
                             velocity_seed_over_c=seed, eta=eta, dark_mass_over_omega0=mass, force_quiver_over_c=force,
                             block_steps=blocks, scalar_units='native SI', XLA_FLAGS=flags,
                             initial_dark_energy_over_energy_scale=reservoir,
                             normalization=dict(omega0_rad_s=wp, field_scale_V_m=field_scale,
                                                energy_scale_J_m2=energy_scale), **initial_source)
        save_run(Path(folder) / label, 'pair_waveform_branch', case_settings, records[label], **values)
        np.savez_compressed(Path(folder) / label / 'data.npz', **values)
        return final, values

    _, native = advance('coupled', coupled, start)
    dense = ({key: native.pop('pump_' + key) for key in ('t', 'mean', 'mean_D', 'mean_A')}
             if stride > 1 else native)
    knots, realized = pair_push_table(plasma, start.ordinary, dense, mass * wp, eta)
    tau = knots * wp
    oracle_args = (tau, eta, mass, 0., force / eta)
    loose = relativistic_background(*oracle_args, nodes=quadrature, method='DOP853', dense_output=True)
    precise = relativistic_background(*oracle_args, nodes=quadrature, method='DOP853',
                                      rtol=2e-13, atol=2e-15, dense_output=True)
    oracle = relativistic_background(*oracle_args, nodes=2 * quadrature, method='DOP853',
                                     rtol=2e-13, atol=2e-15, dense_output=True)
    reference_checks = dict(
        tolerance_force_error_over_initial=float(np.max(abs(loose['mean_D'] - precise['mean_D'])) / (force / eta)),
        quadrature_force_error_over_initial=float(np.max(abs(precise['mean_D'] - oracle['mean_D'])) / (force / eta)),
        max_energy_defect_over_homogeneous_initial_reservoir=float(
            np.max(abs(oracle['energy'] - oracle['energy'][0])) / nominal_reservoir))
    curves['coupled'] = native
    cadence = max(1, round(table_dt / dtau))
    forcing_checks = {}
    for source, values in (('realized', realized), ('homogeneous', oracle['mean_D'] * field_scale)):
        for resolution, every in (('coarse', cadence), ('fine', 1)):
            selected = np.unique(np.r_[0, np.arange(1, len(knots) - 1, every), len(knots) - 1])
            table_times, table = knots[selected], values[selected]
            if table_times[0] > 0 or table_times[-1] < native['t'][-1]:
                raise ValueError('prescribed table does not cover the physical run horizon')
            label = source + '_' + resolution
            amplitude = jnp.stack((jnp.asarray(table), jnp.zeros(len(table)), jnp.zeros(len(table))), axis=1)
            drive = PrescribedDrive(eta, amplitude, 0., times=jnp.asarray(table_times))
            sim = DarkSimulation(plasma, drive)
            initial = sim.continue_with_parameters(start.replace(
                E=None, B=None, A=None, phi=None, initial_projection_norm=jnp.zeros(())))
            _, curves[label] = advance(label, sim, initial)
            if records[label]['initial_ordinary_fingerprints'] != initial_hashes:
                raise ValueError('waveform intervention changed the ordinary initial state')
            forcing_checks[label] = dict(
                max_midpoint_force_error_over_initial=float(np.max(abs(np.interp(
                    knots[1:-1], table_times, table) - values[1:-1])) / (force * field_scale / eta)),
                knots=len(table), maximum_knot_dt_omega0=float(np.max(np.diff(table_times)) * wp),
                table_refinement='distinct coarse/fine cadence' if cadence > 1 else 'identical cadence; no refinement')
    for label, values in curves.items():
        pair_waveform_normalize(values, label, energy_scale, field_scale, wp, phase)
        records[label].update(
            max_balance_defect_over_initial_reservoir=float(maxima[label][0] / energy_scale / reservoir),
            max_dark_sector_defect_over_initial_reservoir=float(maxima[label][7] / energy_scale / reservoir),
            max_ordinary_sector_defect_over_initial_reservoir=float(maxima[label][8] / energy_scale / reservoir))
    # Evaluate the independent warm background at integer clocks, not forcing-table midpoints.
    integer_background = oracle['dense'](curves['coupled']['t'])
    orbit = oracle['initial_momentum'][:, None] + integer_background[3]
    homogeneous_kinetic = oracle['weights'] @ (orbit**2 / (np.sqrt(1 + orbit**2) + 1))
    homogeneous_dark = .5 * (integer_background[1]**2 + mass**2 * integer_background[2]**2)
    warm_excess = (curves['coupled']['kinetic'].sum(axis=1) - curves['coupled']['kinetic'][0].sum()
                   - homogeneous_kinetic + homogeneous_kinetic[0]) / reservoir
    dark_depletion = (homogeneous_dark - curves['coupled']['dark']) / reservoir
    comparisons = pair_waveform_windows(curves, maxima, energy_scale, horizon, local_moments, dark_depletion)
    distinct_velocities = cells * particles_per_cell if pair_loading == 'global' else particles_per_cell
    delta_v = .1 / distinct_velocities
    late = curves['coupled']['t'] >= max(0, horizon - 20) - 1e-8
    records['coupled'].update(
        late_total_dark_fraction=float(np.mean(curves['coupled']['dark'][late]) / reservoir),
        late_coherent_dark_fraction=float(np.mean(curves['coupled']['dark_coherent'][late]) / reservoir),
        late_nonzero_dark_fraction=float(np.mean(
            curves['coupled']['dark'][late] - curves['coupled']['dark_coherent'][late]) / reservoir),
        late_additional_dark_depletion_fraction=float(np.mean(dark_depletion[late])),
        initial_dark_preparation_offset_fraction=float(dark_depletion[0]),
        late_additional_dark_depletion_gain=float(np.mean(dark_depletion[late] - dark_depletion[0])),
        late_kinetic_excess_over_warm_background=float(np.mean(warm_excess[late])))
    settings = dict(model='seeded relativistic pair waterbag waveform interventions',
                    parent_revision=parent_revision(), cells=cells, particles_per_cell_per_species=particles_per_cell,
                    dt_omega0=dtau, horizon_omega0=horizon, shape_order=shape_order, seed_mode=mode,
                    length_c_over_omega0=70., waterbag_full_width_over_c=.1,
                    velocity_seed_over_c=seed, eta=eta, dark_mass_over_omega0=mass, force_quiver_over_c=force,
                    quadrature_nodes=[quadrature, 2 * quadrature], reference_method='DOP853',
                    linear_reference_horizon_omega0=min(20., horizon, linear_end),
                    reference_tolerances=[2e-11, 2e-13], table_dt_omega0=cadence * dtau,
                    native_diagnostic_dt_omega0=dtau, saved_output_dt_omega0=max(1, round(output_dt / dtau)) * dtau,
                    scalar_dt_omega0=scalar_dt, forcing_native_dt_omega0=dtau,
                    local_spread_lengths_c_over_omega0=(scales * wp / c).tolist(), local_moments=local_moments,
                    initial_ordinary_fingerprints=initial_hashes, delta_v_over_c=delta_v,
                    initial_dark_energy_over_energy_scale=reservoir,
                    homogeneous_initial_dark_energy_over_energy_scale=nominal_reservoir,
                    pair_loading=pair_loading, distinct_unperturbed_velocities=distinct_velocities,
                    estimated_ballistic_recurrence_omega0=2 * np.pi / (k * c / wp * delta_v),
                    velocity_refinement_particles_per_cell=[128, 256, 512],
                    recurrence_qualification='free-streaming estimate; driven relativistic orbits change recurrence',
                    forcing='native first-half-kick coupled mean; separately refined homogeneous continuum envelope',
                    intervention='finite-k dark force removed; external work replaces the closed reservoir',
                    work_sign='work: into dark if coupled, into ordinary if prescribed; source_work: into ordinary',
                    depletion_definition='QD = (homogeneous dark energy − coupled total dark energy) / actual UD0',
                    depletion_gain_definition='QD(t) − QD(0); raw QD retains the initial preparation offset',
                    kinetic_excess_definition='(PIC kinetic gain − warm homogeneous kinetic gain) / actual UD0',
                    figure_note='Logarithmic mode panel omits t=0; raw zero-time samples remain in data.',
                    XLA_FLAGS=flags,
                    unscaled_native_units=dict(charge='C/m²', grid_charge='C/m²',
                                               ordinary_gauss='V/m²', dark_gauss='V/m²'),
                    accuracy_targets=dict(reference_energy_over_reservoir=1e-9, quadrature_force_over_initial=1e-6,
                                          table_force_over_initial=1e-4, primary_window_fraction=.01,
                                          nonzero_mode_density_window_fraction=.05,
                                          conservation_over_transfer_and_target_difference=1e-3),
                    normalization=dict(omega0_rad_s=wp, field_scale_V_m=field_scale, energy_scale_J_m2=energy_scale),
                    **initial_source)
    results = dict(reference=reference_checks, forcing=forcing_checks, cases=records, windows=comparisons,
                   claim='finite-time intervention; late claims require loading/grid/time and table convergence')
    output_stride = max(1, round(output_dt / scalar_dt))
    indices = np.unique(np.r_[np.arange(0, len(native['t']), output_stride), len(native['t']) - 1])
    linear_arrays, results['linear_reference'] = pair_waveform_linear(
        curves, indices, k * c / wp, seed, eta, mass, force, quadrature, linear_end)
    saved = {f'{label}_{key}': value[indices] for label, values in curves.items() for key, value in values.items()}
    fig = pair_waveform_plot(curves, indices, reservoir, linear_arrays)
    save_run(folder, 'pair_waveform_control', settings, results, fig, **saved, **linear_arrays,
             coupled_total_dark_fraction=curves['coupled']['dark'][indices] / reservoir,
             coupled_coherent_dark_fraction=curves['coupled']['dark_coherent'][indices] / reservoir,
             coupled_nonzero_dark_fraction=(curves['coupled']['dark'][indices]
                                            - curves['coupled']['dark_coherent'][indices]) / reservoir,
             homogeneous_dark_fraction=homogeneous_dark[indices] / reservoir,
             additional_dark_depletion_fraction=dark_depletion[indices],
             additional_dark_depletion_gain=dark_depletion[indices] - dark_depletion[0],
             kinetic_excess_over_warm_background=warm_excess[indices],
             homogeneous_table_t=tau, homogeneous_table_D=oracle['mean_D'],
             realized_table_t=tau, realized_table_D=realized / field_scale)
    archive = np.load(Path(folder) / 'data.npz')
    compressed = {key: archive[key] for key in archive.files}
    archive.close()
    np.savez_compressed(Path(folder) / 'data.npz', **compressed)
    plt.close(fig)
    print('🌒 WAVEFORM CONTROLS:', results['reference'], flush=True)
    return curves, settings, results


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


def paper_plasma(cells, particles, dtau, seed, momentum_seed=0., seed_mode=16, seed_phase=0., shape_order=2):
    """Appendix B plasma: RMS sigma, exact mass ratio, unseeded Gaussian velocities."""
    if not isinstance(cells, (int, np.integer)) or cells < 4:
        raise ValueError("paper cells must be an integer >= 4")
    if not isinstance(particles, (int, np.integer)) or particles < 2:
        raise ValueError("paper particles per species must be an integer >= 2")
    if not np.isfinite(dtau) or dtau <= 0:
        raise ValueError("paper dt must be finite and positive")
    if shape_order not in (2, 5):
        raise ValueError('paper shape_order must be 2 or 5')
    for value in (momentum_seed, seed_phase):
        if not isinstance(value, core.Tracer) and not np.isfinite(value):
            raise ValueError('physical momentum seed and phase must be finite')
    seeded = isinstance(momentum_seed, core.Tracer) or momentum_seed != 0
    if seeded and (not isinstance(seed_mode, (int, np.integer)) or not 0 < 2 * seed_mode < min(cells, particles)):
        raise ValueError('physical seed_mode must be a positive resolved Fourier mode')
    wp, ratio = 1e9, 1836.0
    density = wp**2 * epsilon_0 * mass_electron / e**2
    length = 40 * c / wp
    x, _ = quiet_start(particles, length)
    wave = jnp.cos(2 * jnp.pi * seed_mode * x[:, 0] / length + seed_phase) if seeded else None
    rng = np.random.default_rng(seed)
    species = []
    for name, charge, mass in (("electrons", -1, mass_electron),
                               ("ions", 1, ratio * mass_electron)):
        velocity = rng.normal(size=particles)
        # Remove finite-sample bulk flow and match the specified initial variance.
        velocity = (velocity - velocity.mean()) / velocity.std()
        velocity *= c * np.sqrt(1e-3 * mass_electron / mass)
        if seeded:
            u = jnp.asarray(velocity) / jnp.sqrt(1 - (velocity / c)**2)
            u -= charge * mass_electron / mass * momentum_seed * np.sqrt(.001) * c * wave
            velocity = u / jnp.sqrt(1 + (u / c)**2)
        v = jnp.zeros((particles, 3)).at[:, 0].set(jnp.asarray(velocity))
        species.append(Species(name, particles, charge, mass, density, x=x, v=v))
    solver = Solver(relativistic=True) if shape_order == 2 else Solver(relativistic=True, shape_order=5)
    return Simulation(Domain(length=length, cells=cells, time_step=dtau / wp), tuple(species), solver), wp


def array_fingerprint(array):
    """Exact host-array identity: dtype, shape and contiguous bytes, without storing it."""
    array = np.ascontiguousarray(array)
    digest = hashlib.sha256(f'{array.dtype.str}:{array.shape}'.encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def seed_noise(plasma, ordinary, fraction, mode, phase):
    """Measure the initial physical seed against thermal current, without evolving particles."""
    d = plasma.domain
    s = plasma.solver
    order = getattr(s, 'shape_order', 2)
    if (s.algorithm != 'explicit' or order not in (2, 5) or s.filter_passes
            or d.field_bc != (0, 0) or d.particle_bc != (0, 0)):
        raise ValueError('seed current audit requires explicit, unfiltered periodic quadratic/quintic deposition')
    u, w = np.asarray(ordinary.u), np.asarray(ordinary.w)
    mass, charge = map(np.asarray, plasma.per_particle)
    velocity = u / np.sqrt(1 + np.sum((u / c)**2, axis=1))[:, None]
    x = np.asarray(ordinary.x[:, 0]) - d.dt * velocity[:, 0] / 2
    x = (x + d.length / 2) % d.length - d.length / 2
    k = 2 * np.pi * mode / d.length
    thermal_u = u.copy()
    thermal_u[:, 0] += charge / e * mass_electron / mass * fraction * np.sqrt(.001) * c * np.cos(k * x + phase)
    thermal = thermal_u[:, 0] / np.sqrt(1 + np.sum((thermal_u / c)**2, axis=1))
    amount = charge * w / (e * float(plasma.species[0].density) * d.length * c)
    centres = np.asarray(d.grid)
    coordinate = (x - centres[0]) / d.dx
    if order == 2:
        nearest = np.floor(coordinate + .5).astype(int)
        offset = coordinate - nearest
        indices = nearest[:, None] + np.array([-1, 0, 1])
        derivative = np.stack((offset - .5, -2 * offset, offset + .5), axis=1) / d.dx
    else:
        indices = np.floor(coordinate).astype(int)[:, None] + np.arange(-2, 4)
        basis = BSpline.basis_element(np.arange(7) - 3, extrapolate=False)
        derivative = np.nan_to_num(basis.derivative()(coordinate[:, None] - indices)) / d.dx
    indices %= d.cells
    kernel = np.sum(derivative * np.exp(-1j * k * centres[indices]), axis=1)
    bases = dict(particle=np.exp(-1j * k * x),
                 continuity_face=-kernel / (1j * 2 * np.sin(k * d.dx / 2) / d.dx))
    sections, begin = dict(total=slice(None)), 0
    for species in plasma.species:
        sections[species.name] = slice(begin, begin + species.n)
        begin += species.n
    result = {}
    for basis, values in bases.items():
        result[basis] = {}
        for label, section in sections.items():
            noise = np.sum(amount[section] * values[section] * thermal[section])
            signal = np.sum(amount[section] * values[section] * (velocity[section, 0] - thermal[section]))
            result[basis][label] = dict(thermal_current=[float(noise.real), float(noise.imag)],
                                        seed_current=[float(signal.real), float(signal.imag)],
                                        seed_over_thermal=float(abs(signal) / abs(noise)) if abs(noise) else None)
    result['thermal_temperature_over_mec2'] = [float(np.average(
        (thermal[s] - np.average(thermal[s], weights=w[s]))**2, weights=w[s])
        * mass[s][0] / (mass_electron * c**2)) for s in list(sections.values())[1:]]
    rho = np.mean(np.asarray(ordinary.rho) * np.exp(-1j * k * centres)) / (e * float(plasma.species[0].density))
    result['charge_mode'] = [float(rho.real), float(rho.imag)]
    result['seed_charge_mode'] = [0., 0.]  # The counterfactual retains the same integer-time x and w.
    result['scope'] = 'Initial selected-mode current in enc units; other noise modes and late dominance untested'
    return result


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


def paper_run(sim, start, steps, stride, block_horizon, wp, scales, folder, mode=1, pump=False,
              reference=None, maxima=None, compilation_observer=None):
    """Compile one fixed interval, preserving the global reference across every block."""
    if block_horizon is not None and (not np.isfinite(block_horizon) or block_horizon <= 0):
        raise ValueError('paper block horizon must be finite and positive')
    block_steps = steps if block_horizon is None else round(block_horizon / sim.plasma.domain.dt / wp)
    if block_steps < 1 or block_steps % stride or steps % block_steps:
        raise ValueError('paper blocks must divide the run and contain complete output intervals')
    reference = snapshot(sim, start) if reference is None else reference
    save_compressed_state(Path(folder) / 'initial_state.npz', start, sim)
    print(f'🦇 Compiling {block_steps} paper steps per block', flush=True)
    before = time.perf_counter()
    lowered = measured_run.lower(sim, start, block_steps, stride, reference, scales, mode, pump=pump)
    executable = lowered.compile()
    compile_seconds = time.perf_counter() - before
    if compilation_observer is not None:
        compilation_observer(lowered, executable)
    memory = executable.memory_analysis()
    print(f'🦇 Compiled in {compile_seconds:.2f} s; advancing {steps // block_steps} blocks', flush=True)
    before = time.perf_counter()
    final, maxima, chunks = start, jnp.zeros(9) if maxima is None else jnp.asarray(maxima), []
    for index in range(steps // block_steps):
        final, history, block_max = executable(sim, final, reference, scales, mode)
        maxima = jnp.maximum(maxima, block_max)
        chunks.append({key: np.array(value)[int(index > 0):] for key, value in history.items()})
        if block_horizon is not None:
            print(f'🌒 block {index + 1}/{steps // block_steps}', flush=True)
    maxima.block_until_ready()
    warm_seconds = time.perf_counter() - before
    history = {key: np.concatenate([chunk[key] for chunk in chunks]) for key in chunks[0]}
    return final, history, maxima, block_steps, compile_seconds, warm_seconds, memory


def paper_normalize(history, plasma, wp, energy_scale, field_scale, momentum_seed=0., seed_mode=1):
    """Apply the paper's original scalar units and physical mode phase in place."""
    history['t'] = history['t'] * wp
    for key in ('electric', 'magnetic', 'dark', 'dark_coherent', 'kinetic', 'spread', 'balance', 'work'):
        history[key] /= energy_scale
    if 'local_spread' in history:
        history['local_spread'] /= energy_scale
    for key in ('mean', 'rms', 'max_speed'):
        history[key] /= c
    history['momentum'] /= energy_scale / c
    history['mean_E'] /= field_scale
    history['mode_E'] /= field_scale
    history['dark_mode_E'] /= field_scale
    phase_origin = np.exp(-2j * np.pi * seed_mode * float(plasma.domain.faces[0]) / plasma.domain.length)
    history['mode_E'] = history['mode_E'] * (phase_origin if momentum_seed else 1)
    history['dark_mode_E'] = history['dark_mode_E'] * (phase_origin if momentum_seed else 1)
    history['nonzero_electric'] = history['electric'] - history['mean_E']**2 / 2
    return history


PAPER_MAXIMUM_KEYS = (
    'max_energy_work_defect_over_initial_thermal', 'max_momentum_defect_over_nmecL',
    'max_particle_charge_change_over_enL', 'max_continuity_over_enwp',
    'max_ordinary_gauss_over_en_eps0', 'max_dark_gauss_over_en_eps0',
    'max_grid_charge_change_over_enL', 'max_dark_work_defect_over_nmc2L',
    'max_ordinary_work_defect_over_nmc2L')


def paper_maximum_scales(plasma, wp, energy_scale):
    """SI scales of the nine paper maxima in the runner's fixed order."""
    charge = e * plasma.species[0].density
    return np.asarray([.001 * energy_scale, energy_scale / c, charge * plasma.domain.length,
                       charge * wp, charge / epsilon_0, charge / epsilon_0,
                       charge * plasma.domain.length, energy_scale, energy_scale])


def paper_accepted_clock(dt, steps, time=0.):
    """Match repeated native clock addition without allocating a time history."""
    for _ in range(steps):
        time += dt
    return time


def paper_archived_model(path, setting, wp, field):
    """Read unchanged model parameters; the Gaussian paper producer uses a cosine."""
    with np.load(path, allow_pickle=False) as stored:
        omega, eta = float(stored['dark.omega']), float(stored['dark.eta'])
        if setting['coupling'] is None:
            if (str(stored['dark.mode']) != 'drive' or omega != wp or eta != 1.
                    or 'dark.times' in stored or float(stored['dark.phase']) != 0.
                    or not np.array_equal(stored['dark.amplitude'], [setting['force_quiver_over_c'] * field, 0., 0.])):
                raise ValueError('paper continuation drive differs from the donor experiment')
            return PrescribedDrive(eta, jnp.asarray(stored['dark.amplitude']), omega, float(stored['dark.phase']))
        if str(stored['dark.mode']) != 'field' or omega != wp or eta != setting['coupling']:
            raise ValueError('paper continuation dark field differs from the donor experiment')
        return DarkField(omega, eta)


def paper_continuation_state(path):
    """Restore the donor's physical model, exact t0 state and final checkpoint."""
    from compare_replays import PAIR_RUNTIME, _load

    path = Path(path)
    record, prefix, data_hash, record_hash = _load(path.parent, 1e-8)
    setting, runtime = dict(record['settings']), provenance()
    if (record['example'] not in ('paper_resonant_conversion', 'paper_resonant_continuation')
            or any(runtime[key] != record[key] for key in PAIR_RUNTIME if key != 'git')
            or parent_revision() != setting['parent_revision'] or pair_xla_flags() != setting['XLA_FLAGS']):
        raise ValueError('paper continuation requires the donor runtime, parent and compiler flags')
    plasma, wp = paper_plasma(setting['cells'], setting['particles_per_species'], setting['dt_omega_p'],
                              setting['seed'], setting.get('momentum_seed_over_sigma_e', 0.),
                              setting.get('seed_mode') or 1, setting.get('seed_phase') or 0., setting['shape_order'])
    field, energy = mass_electron * c * wp / e, plasma.species[0].density * mass_electron * c**2 * plasma.domain.length
    expected = dict(omega_p_rad_s=wp, field_scale_V_m=field, energy_scale_J_m2=energy,
                    charge_density_C_m3=e * plasma.species[0].density, epsilon0_F_m=float(epsilon_0), c_m_s=float(c))
    if (not np.allclose([setting['normalization'][key] for key in expected], list(expected.values()),
                        rtol=2e-12, atol=0)
            or [setting[key] for key in ('length_c_over_omega_p', 'mass_ratio', 'T_each_over_mec2')] != [40, 1836, .001]
            or setting['force_quiver_over_c'] != setting['drive_quiver_over_sigma'] * np.sqrt(.001)):
        raise ValueError('paper continuation donor units or physical loading differ')
    model = paper_archived_model(path, setting, wp, field)
    sim = DarkSimulation(plasma, model)
    origin, start = load_state(path.parent / 'initial_state.npz', sim), load_state(path, sim)
    hashes = {key: array_fingerprint(getattr(origin.ordinary, key)) for key in ('x', 'u', 'w', 'E', 'B', 'rho')}
    if isinstance(model, DarkField):
        hashes.update({f'dark_{key}': array_fingerprint(getattr(origin, key)) for key in ('E', 'B', 'A', 'phi')})
    loading = {key: array_fingerprint(np.concatenate([np.asarray(getattr(s, key)) for s in plasma.species]))
               for key in ('x', 'v')}
    if dict(state=hashes, loading=loading) != setting['initial_fingerprints']:
        raise ValueError('paper continuation original arrays differ from the donor fingerprints')
    if (float(origin.ordinary.time) != 0 or float(origin.ordinary.steps) != 0 or float(origin.work) != 0
            or not np.issubdtype(np.asarray(start.ordinary.steps).dtype, np.integer)
            or int(start.ordinary.steps) <= 0
            or float(start.ordinary.time) != paper_accepted_clock(
                float(plasma.domain.dt), int(start.ordinary.steps))):
        raise ValueError('paper continuation requires exact zero-origin and native checkpoint clocks')
    for key in ('background', 'initial_ordinary', 'initial_dark'):
        if not np.array_equal(getattr(origin, key), getattr(start, key)):
            raise ValueError(f'paper continuation changed the global {key} ledger')
    if not np.array_equal(origin.ordinary.w, start.ordinary.w):
        raise ValueError('paper continuation changed particle weights')
    lineage = dict(prefix_native_git=record['git'], prefix_record_sha256=record_hash, prefix_data_sha256=data_hash,
                   prefix_lineage=setting.get('continuation'),
                   **{key: hashlib.sha256(file.read_bytes()).hexdigest() for key, file in (
                       ('origin_archive_sha256', path.parent / 'initial_state.npz'),
                       ('checkpoint_archive_sha256', path),
                       ('producer_script_sha256', Path(__file__)))})
    return sim, origin, start, prefix, setting, record['results'], lineage


def paper_continuation_checks(sim, origin, start, prefix, setting, results, horizon):
    """Validate prefix endpoints/cadence and recover conservative global SI bounds."""
    plasma, normalization = sim.plasma, setting['normalization']
    wp, energy, field = [normalization[key] for key in ('omega_p_rad_s', 'energy_scale_J_m2', 'field_scale_V_m')]
    dtau, cadence, blocks = setting['dt_omega_p'], setting['output_dt_omega_p'], setting['block_steps']
    if (not np.isfinite([dtau, cadence, horizon]).all() or min(dtau, cadence, horizon) <= 0
            or type(blocks) is not int or blocks < 1):
        raise ValueError('paper continuation needs positive finite clocks and integer blocks')
    total, stride = round(horizon / dtau), round(cadence / dtau)
    steps = total - int(start.ordinary.steps)
    local = setting['local_moments_output_dt_omega_p'] is not None
    scales = np.asarray(setting['local_spread_lengths_c_over_wp']) * c / wp
    if (stride < 1 or steps < 1 or steps % blocks or blocks % stride or int(start.ordinary.steps) % blocks
            or not np.allclose([total * dtau, stride * dtau, blocks * dtau],
                               [horizon, cadence, setting['block_horizon_omega_p']], rtol=0, atol=1e-10)
            or local != ('local_spread' in prefix) or (local and setting['local_moments_output_dt_omega_p'] != cadence)
            or scales.shape != (2,) or not np.isfinite(scales).all() or np.any(scales <= 0)
            or prefix['t'].shape != (int(start.ordinary.steps) // stride + 1,)
            or prefix['t'][-1] != float(start.ordinary.time) * wp):
        raise ValueError('paper continuation must preserve native blocks, clocks and physical moment sampling')
    units = dict(charge=normalization['charge_density_C_m3'] * plasma.domain.length,
                 grid_charge=normalization['charge_density_C_m3'] * plasma.domain.length,
                 ordinary_gauss=normalization['charge_density_C_m3'] / epsilon_0,
                 dark_gauss=normalization['charge_density_C_m3'] / epsilon_0, mean_D=field, mean_A=field / wp)
    for state, index in ((origin, 0), (start, -1)):
        sample = {key: np.array(value)[None]
                  for key, value in snapshot(sim, state, mode=setting['recorded_mode']).items()}
        paper_normalize(sample, plasma, wp, energy, field, setting.get('momentum_seed_over_sigma_e', 0.),
                        setting.get('seed_mode') or 1)
        mismatch = any(key not in prefix or not np.allclose(
            value[0], prefix[key][index], rtol=2e-12, atol=2e-12 * units.get(key, 1.)) for key, value in sample.items())
        if mismatch:
            raise ValueError('paper continuation normalized prefix endpoints differ from complete native states')
    maximum_scales = paper_maximum_scales(plasma, wp, energy)
    prior = np.asarray([results[key] for key in PAPER_MAXIMUM_KEYS]) * maximum_scales
    native = np.asarray([start.max_balance_error, start.max_ordinary_gauss, start.max_dark_gauss])
    if (not np.isfinite(prior).all() or np.any(prior < 0)
            or np.any(abs(prior[[0, 4, 5]] - native) > 32 * np.finfo(float).eps * np.asarray(
                [energy, units['ordinary_gauss'], units['dark_gauss']]))):
        raise ValueError('paper continuation measured maxima disagree with the complete native ledger')
    prior[[0, 4, 5]] = np.maximum(prior[[0, 4, 5]], native)
    prior = np.where(prior > 0, np.nextafter(prior, np.inf), 0.)
    return steps, stride, scales if local else None, prior


def paper_continue(folder, initial_state, horizon):
    """Extend a native Gaussian replay with its original global reference and prefix."""
    if initial_state is None or Path(folder).resolve() == Path(initial_state).parent.resolve():
        raise ValueError('paper continuation needs a donor final archive and a distinct output folder')
    sim, origin, start, prefix, setting, old_results, lineage = paper_continuation_state(initial_state)
    steps, stride, scales, prior = paper_continuation_checks(
        sim, origin, start, prefix, setting, old_results, horizon)
    normalization = setting['normalization']
    wp, energy, field = [normalization[key] for key in ('omega_p_rad_s', 'energy_scale_J_m2', 'field_scale_V_m')]
    with elapsed_progress('Global paper continuation'):
        final, history, maxima, _, compile_s, warm_s, memory = paper_run(
            sim, start, steps, stride, setting['block_horizon_omega_p'], wp,
            None if scales is None else jnp.asarray(scales), folder, setting['recorded_mode'],
            reference=snapshot(sim, origin), maxima=prior)
    folder = Path(folder)
    (folder / 'initial_state.npz').replace(folder / 'segment_initial_state.npz')
    save_compressed_state(folder / 'initial_state.npz', origin, sim)
    save_compressed_state(folder / 'final_state.npz', final, sim)
    (folder / 'prefix_data.npz').write_bytes((Path(initial_state).parent / 'data.npz').read_bytes())
    (folder / 'prefix_run.json').write_bytes((Path(initial_state).parent / 'run.json').read_bytes())
    np.savez_compressed(folder / 'segment_data.npz', **history)
    paper_normalize(history, sim.plasma, wp, energy, field, setting.get('momentum_seed_over_sigma_e', 0.),
                    setting.get('seed_mode') or 1)
    if any(key not in prefix or prefix[key].shape[1:] != value.shape[1:] for key, value in history.items()):
        raise ValueError('paper continuation scalar schema differs from the original normalized prefix')
    joined = {key: np.concatenate((prefix[key], value[1:])) for key, value in history.items()}
    results = dict(zip(PAPER_MAXIMUM_KEYS,
                       (np.asarray(maxima) / paper_maximum_scales(sim.plasma, wp, energy)).tolist()))
    results.update(all_step_maxima_SI=np.asarray(maxima).tolist(), reconstructed_prior_maxima_SI=prior.tolist(),
                   prior_maxima_note='SI bounds reconstructed from normalized donor maxima and rounded upward; '
                                     'exact native energy/Gauss maxima are cross-checked and retained.',
                   compile_s=compile_s, warm_primal_s=warm_s,
                   compiler_temporary_MiB=memory.temp_size_in_bytes / 2**20 if memory else None,
                   claim='Mixed-producer continuation at fixed physics; late convergence remains unverified.')
    lineage.update(start_step=int(start.ordinary.steps), end_step=int(final.ordinary.steps),
                   start_time_omega_p=float(history['t'][0]), end_time_omega_p=float(history['t'][-1]),
                   target_time_omega_p=horizon, segment_steps=steps,
                   lineage_note='Prefix retains its producer lineage; the segment has the current producer. '
                                'Original reference curves remain prefix-only.')
    setting.update(horizon_omega_p=float(joined['t'][-1]), initial_state_source='complete original zero-time archive',
                   timing='Continuation segment only; prefix timings belong to its separate native producer.',
                   continuation=lineage)
    save_run(folder, 'paper_resonant_continuation', setting, results, **joined)
    np.savez_compressed(folder / 'data.npz', **joined)
    print('🌘 GLOBAL PAPER CONTINUATION:', results, flush=True)
    return joined, setting, results


def seed_reference(history, amplitude, coupling, seed, mode, phase):
    """Keep the fixed-seed early complex response and its independent quadrature refinement."""
    if seed == 0:
        return {}, {}
    early = history['t'] <= min(40., history['t'][-1]) + 1e-8
    time, k = history['t'][early], 2 * np.pi * mode / 40
    controls = dict(amplitude=amplitude, eta=coupling, delta_u=seed * np.sqrt(.001) * np.exp(1j * phase))
    coarse = gaussian_tangent(time, k, nodes=64, **controls)
    fine = gaussian_tangent(time, k, nodes=128, rtol=2e-11, **controls)
    curves, results = dict(linear_t=time), {}
    for observed, key in (('mode_E', 'mode_E'), ('dark_mode_E', 'mode_D')):
        norm = np.linalg.norm(fine[key])
        curves[f'linear_{key}'] = fine[key]
        results[f'linear_{key}_absolute_l2'] = float(np.linalg.norm(history[observed][early] - fine[key]))
        results[f'linear_{key}_relative_l2'] = results[f'linear_{key}_absolute_l2'] / norm if norm else None
        results[f'linear_{key}_quadrature_absolute_l2'] = float(np.linalg.norm(coarse[key] - fine[key]))
    results['linear_reference_horizon'] = float(time[-1])
    return curves, results


def paper_case(folder, cells, particles, dtau, horizon, seed, ratio, eta=None,
               block_horizon=None, local_moments=False, initial_state=None,
               momentum_seed=0., seed_mode=16, seed_phase=0., shape_order=2):
    """Hook Fig. 2 drive with reduced moments, work and both constraint ledgers."""
    if not np.isfinite(horizon) or horizon <= 0 or not np.isfinite(ratio) or ratio < 0:
        raise ValueError("paper horizon must be positive and drive ratio nonnegative, both finite")
    if eta is not None and (not np.isfinite(eta) or eta <= 0):
        raise ValueError("paper finite-reservoir coupling must be finite and positive")
    plasma, wp = paper_plasma(cells, particles, dtau, seed, momentum_seed, seed_mode, seed_phase, shape_order)
    amplitude = ratio * np.sqrt(1e-3)
    field_scale = mass_electron * c * wp / e
    force = amplitude * field_scale
    model = (PrescribedDrive(1.0, jnp.array([force, 0., 0.]), wp) if eta is None
             else DarkField(wp, eta, initial_E=jnp.tile(
                 jnp.array([force / eta, 0., 0.]), (cells, 1))))
    sim = DarkSimulation(plasma, model)
    start, fingerprints = paper_initial(sim, seed, initial_state)
    seed_diagnostics = seed_noise(plasma, start.ordinary, momentum_seed, seed_mode, seed_phase) if momentum_seed else {}
    stride = max(1, round(0.5 / dtau))
    steps = stride * max(1, round(horizon / (stride * dtau)))
    scales = np.array([2., 4.]) * np.sqrt(1e-3) * c / wp
    sampled_scales = jnp.asarray(scales) if local_moments else None
    final, history, maxima, block_steps, compile_seconds, warm_seconds, memory = paper_run(
        sim, start, steps, stride, block_horizon, wp, sampled_scales, folder, seed_mode if momentum_seed else 1)
    density = plasma.species[0].density
    energy_scale = density * mass_electron * c**2 * plasma.domain.length
    momentum_scale = energy_scale / c
    coarse_initial = np.asarray(coarse_spread(sim, start, scales)) / energy_scale
    coarse_final = np.asarray(coarse_spread(sim, final, scales)) / energy_scale
    paper_normalize(history, plasma, wp, energy_scale, field_scale, momentum_seed, seed_mode)
    t = history['t']
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
        max_dark_work_defect_over_nmc2L=float(maxima[7] / energy_scale),
        max_ordinary_work_defect_over_nmc2L=float(maxima[8] / energy_scale),
        late_electron_spread_over_initial=float(np.mean(history['spread'][late, 0]) / history['spread'][0, 0]),
        late_ion_spread_over_initial=float(np.mean(history['spread'][late, 1]) / history['spread'][0, 1]),
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
        claim=('Fixed physical momentum-seed extension; not the unseeded Fig. 2 replay' if momentum_seed else
               'Fig. 2 parameter replay; late reproduction requires loading/grid/time/seed convergence'))
    linear_curves, linear_results = seed_reference(history, amplitude, eta, momentum_seed, seed_mode, seed_phase)
    results.update(linear_results, initial_seed_noise=seed_diagnostics or None)
    settings = dict(source=('Controlled physical-seed extension of Hook Fig. 2' if momentum_seed else
                            'Hook, Huang, Shalaby arXiv:2510.13956v1 Fig. 2 and Appendix B'),
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
                    loading=('co-located lattice; conditioned Gaussian thermal velocities plus momentum wave'
                             if momentum_seed else
                             'co-located lattice; independent Gaussian velocities, zero mean and exact variance'),
                    momentum_seed_over_sigma_e=momentum_seed,
                    initial_rms_over_c=history['rms'][0].tolist(),
                    thermal_temperature_over_mec2=seed_diagnostics.get('thermal_temperature_over_mec2', [1e-3, 1e-3]),
                    seed_mode=seed_mode if momentum_seed else None, seed_phase=seed_phase if momentum_seed else None,
                    recorded_mode=seed_mode if momentum_seed else 1,
                    mode_basis='physical exp(-ikx) at faces' if momentum_seed else 'native fft index basis',
                    pusher='relativistic Boris; electric 1V uses the same momentum kick as Vay',
                    shape='quadratic parent spline; paper uses fifth-order' if shape_order == 2
                    else 'quintic cell weights with face-centred gather; not the complete SHARP algorithm',
                    shape_order=shape_order,
                    XLA_FLAGS=' '.join(token if '/' not in token and '\\' not in token
                                       else token.split('=')[0] + '=<path omitted>'
                                       for token in shlex.split(os.environ.get('XLA_FLAGS', ''))),
                    inferred_t_noise=(40 / (amplitude * np.sqrt(3 * particles * cells))
                                      if amplitude else None),
                    parent_revision=parent_revision())
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
        axes[1, 0].plot(t, (history['balance'] - history['balance'][0]) / 1e-3,
                        label='energy − work' if eta is None else 'total energy change')
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
                      local_spread_final=coarse_final, **reference, **history, **linear_curves)
        save_run(folder, 'paper_resonant_conversion', settings, results, fig, **arrays)
        # Scalar histories compress well; retain the native example/provenance path.
        np.savez_compressed(Path(folder) / 'data.npz', **arrays)
        plt.close(fig)
    save_compressed_state(Path(folder) / 'final_state.npz', final, sim)
    print('🌘 PAPER REPLAY:', results, flush=True)
    return history, settings, results


if __name__ == "__main__":  # noqa: C901
    if study == 'paper':
        with elapsed_progress("Resonant replay"):
            paper_case(output, cells, particles, dt, horizon, seed, drive_ratio,
                       coupling, block_horizon, local_moments, initial_state,
                       momentum_seed, seed_mode, seed_phase, shape_order)
    elif study == 'paper_continue':
        paper_continue(output, initial_state, horizon)
    elif study == 'pair':
        pair_figure(output, full)
    elif study == 'pair_dark':
        pair_dark_figure(output, full)
    elif study == 'pair_waveform':
        pair_waveform_control(output, cells, particles_per_cell, dt, horizon, velocity_seed,
                              eta, dark_mass, force, quadrature, table_dt, output_dt,
                              block_horizon, local_moments, shape_order, pair_loading, linear_end, scalar_dt,
                              initial_state)
    elif study == 'pair_repeat':
        pair_repeat(output, initial_state, samples, observe_executable)
    elif study == 'pair_table':
        pair_repeat(output, initial_state, samples, observe_executable, table_every=table_every)
    elif study == 'paper_pilot':
        paper_geometry_pilot(output)
    elif study != 'mobile_ions':
        raise ValueError('study must be mobile_ions, paper, paper_continue, pair, pair_dark, pair_waveform, '
                         'pair_repeat, pair_table or paper_pilot')
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
