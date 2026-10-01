"""A traveling transverse ghost meets a neutral, fixed-column plasma slab.

The cold boundary-value solve is independent of the PIC time step. The default
only exercises the setup; ``full=True`` adds flux, refinement and finite-amplitude
checks on 256 cells through 480 steps. Edit inputs or use ``runpy.run_path``.
"""

from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
from jax import lax
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

from jaxincell import (Domain, Simulation, Solver, Species, elementary_charge as e,
                       epsilon_0, mass_electron, mass_proton, save_run,
                       speed_of_light as c)
from darkjaxincell import DarkField, DarkSimulation, midnight
from darkjaxincell._proca import energy as dark_energy


# Inputs: full=True includes the held-out and refined kinetic controls.
full = globals().get('full', False)
output = Path(globals().get('output', 'artifacts/dark_profile'))
cells = globals().get('cells', 256 if full else 128)
steps = globals().get('steps', 480 if full else 160)
particles_per_basis = globals().get('particles_per_basis', 128)
eta = globals().get('eta', 0.05)
mu_over_omega0 = globals().get('mu_over_omega0', 0.6)
target_energy = globals().get('target_energy', 2e-5)  # J/m², incident Proca packet.
training_carriers = globals().get('training_carriers', (0.75, 0.85))  # k_D c / omega0
held_carriers = globals().get('held_carriers', (0.70, 0.90))


def _bump(x, centre, width=0.9):
    """A nonnegative smooth basis with shared fixed support [-2,2]."""
    edge = np.maximum(1 - (x / 2) ** 2, 0)
    return np.exp(-0.5 * ((x - centre) / width) ** 2) * edge**4


def slab_basis(particles_per_basis=128, phase=0.5):
    """Continuum-normalized bases and deterministic fixed particle positions."""
    grid = np.linspace(-2, 2, 4001)
    curves, positions = [], []
    for centre in (-1.5, -0.5, 0.5, 1.5):
        density = _bump(grid, centre)
        density /= np.trapezoid(density, grid)
        cumulative = np.r_[0, np.cumsum((density[1:] + density[:-1]) * np.diff(grid) / 2)]
        cumulative /= cumulative[-1]
        positions.append(np.interp((np.arange(particles_per_basis) + phase) / particles_per_basis,
                                   cumulative, grid))
        curves.append(density)
    return grid, np.array(curves), np.array(positions)


def weights(theta):
    """Four positive weights with one redundant offset removed."""
    values = jnp.concatenate((jnp.asarray(theta), jnp.zeros((1,))))
    return jax.nn.softmax(values)


def profile(x, theta, basis_grid, basis_curves, column):
    """Number density in m^-3; the physical column is independent of theta."""
    ell = c / 1e9
    values = np.array([np.interp(x / ell, basis_grid, curve, left=0, right=0) / ell
                       for curve in basis_curves])
    return column * np.asarray(weights(theta)) @ values


def packet(domain, mu, carrier, centre, width, target_energy):
    """Fourier-consistent right-going transverse A/E, normalized by full Proca energy."""
    x = np.asarray(domain.faces)
    envelope = np.exp(-0.5 * ((x - centre) / width) ** 2)
    a = envelope * np.cos(carrier * (x - centre))
    k = 2 * np.pi * np.fft.fftfreq(domain.cells, d=domain.dx)
    omega = np.sqrt(mu**2 + c**2 * (2 * np.sin(k * domain.dx / 2) / domain.dx) ** 2)
    electric = np.fft.ifft(1j * np.sign(k) * omega * np.fft.fft(a)).real
    A = jnp.stack((jnp.zeros_like(jnp.asarray(a)), jnp.asarray(a), jnp.zeros_like(jnp.asarray(a))), axis=1)
    E = jnp.stack((jnp.zeros_like(jnp.asarray(a)), jnp.asarray(electric), jnp.zeros_like(jnp.asarray(a))), axis=1)
    B = jnp.zeros_like(A).at[:, 2].set((A[:, 1] - jnp.roll(A[:, 1], 1)) / domain.dx)
    initial = float(dark_energy(E, B, A, jnp.zeros(domain.cells), domain.dx, mu))
    factor = np.sqrt(target_energy / initial)
    E, A = E * factor, A * factor
    return E, A, initial * factor**2, k, np.fft.fft(a)


def cold_scattering(frequency, mu, eta, column, theta, basis_grid, basis_curves, n=401):
    """Independent second-order coupled-wave boundary-value solve in SI units.

    The left Robin boundary injects one unit dark potential. Both asymptotic
    media are vacuum. Return transmitted photon fraction and all four fluxes.
    """
    if frequency <= mu:
        raise ValueError("incident dark frequency must exceed its rest frequency")
    x = np.linspace(-4, 4, n) * c / 1e9  # fixed vacuum padding for every packet frequency
    dx = x[1] - x[0]
    omega_p2 = profile(x, theta, basis_grid, basis_curves, column) * e**2 / epsilon_0 * (
        1 / mass_electron + 1 / mass_proton)
    kg, kd = frequency / c, np.sqrt(frequency**2 - mu**2) / c
    matrix = lil_matrix((2 * n, 2 * n), dtype=complex)
    rhs = np.zeros(2 * n, dtype=complex)
    for channel, k in enumerate((kg, kd)):
        offset = channel * n
        matrix[offset, offset:offset + 3] = [-3 / (2 * dx) + 1j * k,
                                             4 / (2 * dx), -1 / (2 * dx)]
        matrix[offset + n - 1, offset + n - 3:offset + n] = [1 / (2 * dx),
                                                             -4 / (2 * dx), 3 / (2 * dx) - 1j * k]
        if channel:
            rhs[offset] = 2j * kd
        for i in range(1, n - 1):
            diagonal = -2 / dx**2 + k**2 - (eta**2 if channel else 1) * omega_p2[i] / c**2
            matrix[offset + i, offset + i - 1:offset + i + 2] = [1 / dx**2, diagonal, 1 / dx**2]
            matrix[offset + i, (1 - channel) * n + i] = -eta * omega_p2[i] / c**2
    field = spsolve(matrix.tocsr(), rhs)
    reflected_photon = kg / kd * abs(field[0])**2
    transmitted_photon = kg / kd * abs(field[n - 1])**2
    reflected_dark = abs(field[n] - 1)**2
    transmitted_dark = abs(field[-1])**2
    return np.array([reflected_photon, transmitted_photon, reflected_dark, transmitted_dark])


def make_simulation(theta, cells, particles_per_basis, eta, mu, column, positions, E, A,
                    length_units=40):
    """Co-located electron/finite-mass ion species carry each basis's weight."""
    domain = Domain(length=length_units * c / 1e9, cells=cells, dt_over_dx_c=0.4)
    species = []
    for j, position in enumerate(positions):
        x = jnp.stack((jnp.asarray(position) * c / 1e9, jnp.zeros_like(position),
                       jnp.zeros_like(position)), axis=1)
        density = column * weights(theta)[j] / domain.length
        species.extend((Species(f"e{j}", particles_per_basis, -1, mass_electron, density, x=x),
                        Species(f"i{j}", particles_per_basis, 1, mass_proton, density, x=x)))
    plasma = Simulation(domain, tuple(species), solver=Solver(relativistic=True))
    return DarkSimulation(plasma, DarkField(mu, eta, initial_E=E, initial_A=A))


def transmitted_fraction(sim, incident_energy, steps, detector_index, start_step=0):
    """Integrate ordinary Poynting flux using only the production state and a scalar."""
    dt = sim.plasma.domain.dt

    def flux(state):
        field = state.ordinary
        i = detector_index
        ey = (field.E[i, 1] + field.E[i - 1, 1]) / 2
        ez = (field.E[i, 2] + field.E[i - 1, 2]) / 2
        return epsilon_0 * c**2 * (ey * field.B[i, 2] - ez * field.B[i, 1])

    def objective():
        state, extra = sim.initial_state(jax.random.PRNGKey(0))

        def advance(carry, index):
            state, integral = carry
            before = flux(state)
            state, _ = sim._step(state, extra)
            after = flux(state)
            integral += jnp.where(index >= start_step, (before + after) * dt / 2, 0.0)
            return (state, integral), None

        (state, integral), _ = lax.scan(advance, (state, jnp.zeros(())), jnp.arange(steps))
        return integral / incident_energy

    return objective()


def build_design(cells, positions, eta, mu, column, packets, perturbations,
                 steps, detector_index, particles_per_basis=128, length_units=80):
    """A fixed-band, fixed-column PIC objective with explicit profile uncertainty."""
    fields = jnp.stack([item[0] for item in packets])
    potentials = jnp.stack([item[1] for item in packets])
    incident = jnp.asarray([item[2] for item in packets])
    shifts = jnp.asarray(perturbations)

    def one(theta, shift, E, A, energy):
        sim = make_simulation(theta + shift, cells, particles_per_basis, eta, mu,
                              column, positions, E, A, length_units)
        return transmitted_fraction(sim, energy, steps, detector_index)

    def yields(theta):
        return jax.vmap(one, in_axes=(None, 0, 0, 0, 0))(
            theta, shifts, fields, potentials, incident)

    def objective(theta):
        signal = yields(theta)
        roughness = jnp.sum(jnp.diff(weights(theta))**2)
        return jnp.mean(signal) - 10 * jnp.var(signal) - 1e-4 * roughness

    return jax.jit(jax.value_and_grad(objective)), jax.jit(objective), jax.jit(yields)


def spectral_reference(packet_data, mu, eta, column, theta, basis_grid, basis_curves):
    """Energy-weighted outgoing photon flux of the same incident packet spectrum."""
    _, _, _, k, amplitudes = packet_data
    positive = k > 0
    frequencies = np.sqrt(mu**2 + c**2 * k[positive] ** 2)
    energy = frequencies**2 * abs(amplitudes[positive])**2
    included = energy > max(energy) * 1e-7
    fluxes = np.array([cold_scattering(f, mu, eta, column, theta, basis_grid, basis_curves)
                       for f in frequencies[included]])
    fractions = np.average(fluxes, weights=energy[included], axis=0)
    return float(fractions[1]), float(abs(sum(fractions) - 1)), int(sum(included))


def replay_ledger(sim, steps, incident_energy):
    """Final field and cell-local bulk/random energy from a sparse PIC replay."""
    output = sim.run(steps, store_every=steps, store_particles=True, verbose=True)
    energy = output.energy()
    ordinary = output.ordinary
    velocity = np.asarray(ordinary.v[-1])
    position = np.asarray(ordinary.x[-1, :, 0])
    mass_weight = np.asarray(ordinary.mass * ordinary.weight[-1])
    cells = sim.plasma.domain.cells
    cell = np.floor((position + sim.plasma.domain.length / 2) / sim.plasma.domain.dx).astype(int) % cells
    group = np.asarray(ordinary.species) * cells + cell
    bins = len(sim.plasma.species) * cells
    rest = np.bincount(group, weights=mass_weight, minlength=bins)
    momentum = np.stack([np.bincount(group, weights=mass_weight * velocity[:, j], minlength=bins)
                         for j in range(3)], axis=1)
    mean_velocity = momentum / np.maximum(rest[:, None], 1e-300)
    mean2 = np.sum(mean_velocity**2, axis=1)
    root = np.sqrt(np.maximum(1 - mean2 / c**2, 1e-15))
    bulk = np.sum(rest * mean2 / (root * (1 + root)))
    kinetic = float(energy["kinetic"][-1])
    return {"ordinary_electric_over_incident": float(energy["electric"][-1] / incident_energy),
            "ordinary_magnetic_over_incident": float(energy["magnetic"][-1] / incident_energy),
            "dark_over_incident": float(energy["dark"][-1] / incident_energy),
            "bulk_kinetic_over_incident": bulk / incident_energy,
            "local_random_kinetic_over_incident": (kinetic - bulk) / incident_energy,
            "closed_total_over_incident": float(energy["total_with_dark"][-1] / incident_energy),
            "max_particle_speed_over_c": float(np.max(np.linalg.norm(velocity, axis=1)) / c),
            "coarse_graining_dx_m": sim.plasma.domain.dx}


def plot_reference(ax, reference, spectrum, order):
    """Overlay only available full-preset cold scattering curves."""
    if not reference:
        return
    for name, label in (("uniform", "uniform cold oracle"),
                        ("optimized", "optimized cold oracle")):
        cold = np.array([item[0] for item in reference[name]]).reshape(4, 2).mean(axis=1)
        ax.plot(spectrum[order], cold[order], "--", label=label)


if __name__ == "__main__":  # noqa: C901
    if len(training_carriers) != 2 or len(held_carriers) != 2:
        raise ValueError('the design uses two training and two held-out carriers')
    omega0, mu = 1e9, mu_over_omega0 * 1e9
    ell = c / omega0
    n_ref = epsilon_0 * mass_electron * omega0**2 / e**2
    column = 1.5 * n_ref * ell
    domain = Domain(length=80 * ell, cells=cells, dt_over_dx_c=0.4)
    basis_grid, basis_curves, positions = slab_basis(particles_per_basis)
    detector = int(np.argmin(abs(np.asarray(domain.grid) - 8 * ell)))

    def make_packets(carriers, energy=target_energy, grid_domain=domain):
        return [packet(grid_domain, mu, carrier / ell, -16 * ell, 3.5 * ell, energy)
                for carrier in carriers]

    training_packets = make_packets(training_carriers)
    packets = [training_packets[j] for j in (0, 0, 1, 1)]
    shifts = np.array([[0.03, -0.03, 0.03], [-0.03, 0.03, -0.03]] * 2)
    value_grad, objective, yields = build_design(cells, positions, eta, mu, column,
                                                 packets, shifts, steps, detector,
                                                 particles_per_basis=particles_per_basis)
    print(f"🦇 Compiling the {cells}-cell, {steps}-step profile gradient", flush=True)
    t0 = perf_counter()
    value_grad(np.zeros(3))[0].block_until_ready()
    compile_seconds = perf_counter() - t0

    trials = []
    for start in (np.zeros(3), np.array([0.4, -0.4, 0.4]), np.array([-0.4, 0.4, -0.4])):
        print(f"🦇 Profile optimization from {start.tolist()}", flush=True)
        trace = []

        def loss(theta):
            value, gradient = value_grad(theta)
            trace.append(float(value))
            return -1000 * float(value), -1000 * np.asarray(gradient)

        t0 = perf_counter()
        result = minimize(loss, start, jac=True, bounds=[(-0.6, 0.6)] * 3,
                          method="L-BFGS-B", options={"maxiter": 60, "gtol": 1e-8, "ftol": 1e-13})
        trials.append({"start": start.tolist(), "theta": result.x.tolist(),
                       "objective": -result.fun / 1000, "calls": int(result.nfev),
                       "success": bool(result.success), "elapsed_s": perf_counter() - t0,
                       "trace": trace})
    best = max(trials, key=lambda row: row["objective"])
    theta = np.asarray(best["theta"])
    uniform = np.zeros(3)
    baselines = {"uniform": uniform, "single_ramp": np.array([-0.6, 0.0, 0.6]),
                 "reverse_ramp": np.array([0.6, 0.0, -0.6]),
                 "two_ramp": np.array([0.6, -0.6, 0.6])}
    baseline_scores = {name: float(objective(control)) for name, control in baselines.items()}
    print("🦇 Checking equal-budget random profiles and held-out packets", flush=True)
    random = np.random.default_rng(721).uniform(-0.6, 0.6, (sum(row["calls"] for row in trials), 3))
    random_scores = np.array([float(objective(row)) for row in random])
    training = {"uniform": np.asarray(yields(uniform)), "optimized": np.asarray(yields(theta)),
                "random_best": float(max(random_scores))}

    held_packets = make_packets(held_carriers)
    held_shifts = np.array([[0.05, -0.05, 0.05], [-0.05, 0.05, -0.05]] * 2)
    held = [held_packets[j] for j in (0, 0, 1, 1)]
    _, _, held_yields = build_design(cells, positions, eta, mu, column,
                                     held, held_shifts, steps, detector,
                                     particles_per_basis=particles_per_basis)
    held_uniform = np.asarray(held_yields(uniform))
    held_optimized = np.asarray(held_yields(theta))
    h = 1e-3
    finite = (float(objective(np.array([h, 0, 0])))
              - float(objective(np.array([-h, 0, 0])))) / (2 * h)
    ad = float(value_grad(uniform)[1][0])

    def full_checks():
        print("🦇 Checking cold scattering, held-out refinement and finite amplitude", flush=True)
        reference = {}
        for name, control in (("uniform", uniform), ("optimized", theta)):
            reference[name] = [spectral_reference(p, mu, eta, column, control + shift,
                                                  basis_grid, basis_curves)
                               for p, shift in zip(packets + held, np.vstack((shifts, held_shifts)))]

        refined, finite_amplitude = {}, {}
        refined_domain = Domain(length=80 * ell, cells=512, dt_over_dx_c=0.4)
        refined_packets = make_packets(held_carriers, grid_domain=refined_domain)
        refined_held = [refined_packets[j] for j in (0, 0, 1, 1)]
        refined_detector = int(np.argmin(abs(np.asarray(refined_domain.grid) - 8 * ell)))
        for phase in (0.23, 0.5, 0.77):
            _, _, refined_positions = slab_basis(256, phase)
            _, _, replay = build_design(512, refined_positions, eta, mu, column,
                                        refined_held, held_shifts, 960, refined_detector,
                                        particles_per_basis=256)
            baseline = np.asarray(replay(uniform))
            candidate = np.asarray(replay(theta))
            refined[str(phase)] = {"uniform_yields": baseline.tolist(),
                                   "optimized_yields": candidate.tolist(),
                                   "relative_gain": float(candidate.mean() / baseline.mean() - 1)}

        high_energy = 2000.0
        high_packets = make_packets(training_carriers, energy=high_energy)
        high_train = [high_packets[j] for j in (0, 0, 1, 1)]
        high_grad, _, _ = build_design(cells, positions, eta, mu, column,
                                       high_train, shifts, steps, detector,
                                       particles_per_basis=particles_per_basis)

        def high_loss(control):
            value, gradient = high_grad(control)
            return -1000 * float(value), -1000 * np.asarray(gradient)

        high_fit = minimize(high_loss, theta, jac=True, bounds=[(-0.6, 0.6)] * 3,
                            method="L-BFGS-B", options={"maxiter": 60, "gtol": 1e-8, "ftol": 1e-13})
        high_theta = high_fit.x
        high_held_packets = make_packets(held_carriers, energy=high_energy)
        high_held = [high_held_packets[j] for j in (0, 0, 1, 1)]
        _, _, high_replay = build_design(cells, positions, eta, mu, column,
                                         high_held, held_shifts, steps, detector,
                                         particles_per_basis=particles_per_basis)
        high_uniform = np.asarray(high_replay(uniform))
        high_cold = np.asarray(high_replay(theta))
        high_optimized = np.asarray(high_replay(high_theta))
        _, _, shorter = build_design(cells, positions, eta, mu, column,
                                     held, held_shifts, 400, detector,
                                     particles_per_basis=particles_per_basis)
        early_yields = np.asarray(shorter(theta))
        window_replay = {"50": {"uniform": np.asarray(shorter(uniform)).tolist(),
                                "optimized": early_yields.tolist()},
                         "60": {"uniform": held_uniform.tolist(),
                                "optimized": held_optimized.tolist()}}

        def later_window(count):
            _, _, longer = build_design(cells, positions, eta, mu, column,
                                        held, held_shifts, count, detector,
                                        particles_per_basis=particles_per_basis)
            return {"uniform": np.asarray(longer(uniform)).tolist(),
                    "optimized": np.asarray(longer(theta)).tolist()}

        window_replay["70"] = later_window(560)
        window_replay["80"] = later_window(640)
        high_refined_packets = make_packets(held_carriers, energy=high_energy,
                                            grid_domain=refined_domain)
        high_refined_held = [high_refined_packets[j] for j in (0, 0, 1, 1)]
        high_refined = {}
        for phase in (0.23, 0.5, 0.77):
            _, _, refined_positions = slab_basis(256, phase)
            _, _, replay = build_design(512, refined_positions, eta, mu, column,
                                        high_refined_held, held_shifts, 960, refined_detector,
                                        particles_per_basis=256)
            high_refined[str(phase)] = {
                "uniform_yields": np.asarray(replay(uniform)).tolist(),
                "cold_design_yields": np.asarray(replay(theta)).tolist(),
                "high_design_yields": np.asarray(replay(high_theta)).tolist()}
        middle = packet(domain, mu, 0.8 / ell, -16 * ell, 3.5 * ell, high_energy)
        ledger_sim = make_simulation(theta, cells, particles_per_basis, eta, mu, column,
                                     positions, middle[0], middle[1], 80)
        no_wave = make_simulation(theta, cells, particles_per_basis, eta, mu, column, positions,
                                  jnp.zeros_like(middle[0]), jnp.zeros_like(middle[1]), 80)
        quiet = no_wave.run(steps, store_every=steps, store_particles=False, verbose=True)
        finite_amplitude = {"incident_energy_J_m2": high_energy,
                            "optimized_theta": high_theta.tolist(),
                            "optimizer_calls": int(high_fit.nfev),
                            "uniform_held_yields": high_uniform.tolist(),
                            "cold_design_held_yields": high_cold.tolist(),
                            "high_design_held_yields": high_optimized.tolist(),
                            "small_amplitude_window_replay": window_replay,
                            "no_wave_max_ordinary_E_V_m": float(jnp.max(jnp.abs(quiet.ordinary.E))),
                            "refined_held_replay": high_refined,
                            "cold_design_ledger": replay_ledger(ledger_sim, steps, middle[2])}

        return reference, refined, finite_amplitude

    reference, refined, finite_amplitude = full_checks() if full else ({}, {}, {})

    grid = np.linspace(-2, 2, 801) * ell
    density_uniform = profile(grid, uniform, basis_grid, basis_curves, column) / n_ref
    density_optimized = profile(grid, theta, basis_grid, basis_curves, column) / n_ref
    with midnight():
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
        axes[0].plot(grid / ell, density_uniform, label="uniform basis weights")
        axes[0].plot(grid / ell, density_optimized, label="PIC optimized")
        axes[0].axhline(1, ls=":", label="admissible peak")
        axes[0].set(xlabel=r"$x/(c/\omega_0)$", ylabel=r"$n/n_{\rm ref}$",
                    title="A fixed-column haunted slab")
        spectrum = np.r_[training_carriers, held_carriers]
        measured_uniform = np.r_[training["uniform"].reshape(2, 2).mean(axis=1),
                                 held_uniform.reshape(2, 2).mean(axis=1)]
        measured_optimized = np.r_[training["optimized"].reshape(2, 2).mean(axis=1),
                                   held_optimized.reshape(2, 2).mean(axis=1)]
        order = np.argsort(spectrum)
        axes[1].plot(spectrum[order], measured_uniform[order], "o-", label="uniform PIC")
        axes[1].plot(spectrum[order], measured_optimized[order], "o-", label="optimized PIC")
        plot_reference(axes[1], reference, spectrum, order)
        axes[1].set(xlabel=r"incident $k_D c/\omega_0$", ylabel="transmitted photon energy / incident dark energy",
                    title="Outgoing photons, with held-out points")
        axes[0].grid(alpha=0.4)
        axes[1].grid(alpha=0.4)
        axes[0].legend(facecolor="#FFFFFF", edgecolor="#6B7280", fontsize=8)
        axes[1].legend(facecolor="#FFFFFF", edgecolor="#6B7280", fontsize=8)
        settings = {"preset": "full" if full else "quick", "cells": cells, "steps": steps,
                    "particles_per_basis_per_species": particles_per_basis, "length_over_c_omega0": 80,
                    "detector_over_c_omega0": float(domain.grid[detector] / ell),
                    "mu_over_omega0": mu_over_omega0, "eta": eta, "column_over_nref_c_omega0": 1.5,
                    "packet_centre_over_c_omega0": -16, "packet_width_over_c_omega0": 3.5,
                    "target_dark_energy_J_m2": target_energy, "training_carriers": training_carriers,
                    "training_profile_shifts": shifts.tolist(), "held_carriers": held_carriers,
                    "held_profile_shifts": held_shifts.tolist(), "bounds": [-0.6, 0.6]}
        results = {"best": best, "trials": trials, "baseline_scores": baseline_scores,
                   "random_budget": len(random),
                   "random_best_objective": training["random_best"],
                   "uniform_training_yields": training["uniform"].tolist(),
                   "optimized_training_yields": training["optimized"].tolist(),
                   "uniform_held_yields": held_uniform.tolist(),
                   "optimized_held_yields": held_optimized.tolist(),
                   "cold_spectral_reference": reference, "refined_held_replay": refined,
                   "finite_amplitude": finite_amplitude,
                   "max_peak_over_nref": float(max(density_optimized)),
                   "column_ratio": float(np.trapezoid(density_optimized * n_ref, grid) / column),
                   "profile_gradient_ad": ad, "profile_gradient_fd": finite,
                   "gradient_compile_s": compile_seconds}
        save_run(output, "dark_profile", settings, results, fig,
                 x_over_ell=grid / ell, density_uniform=density_uniform,
                 density_optimized=density_optimized, carrier=spectrum,
                 pic_uniform=measured_uniform, pic_optimized=measured_optimized)
        plt.close(fig)
    print(f"🦇 PROFILE: trained {best['objective']:.6f}, held-out "
          f"{held_optimized.mean():.6f} vs uniform {held_uniform.mean():.6f}; "
          f"peak {max(density_optimized):.3f} n_ref")
