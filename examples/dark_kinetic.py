"""A current-neutral Maxwellian hears a mixed longitudinal Landau mode.

The full preset is a seeded kinetic measurement. The quick preset only checks
that the configuration runs; its noise floor is too high for a damping fit.
"""

import argparse
from pathlib import Path

import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import root
from scipy.signal import find_peaks
from scipy.stats import linregress

from jaxincell import (Domain, Simulation, Species, elementary_charge as e,
                       epsilon_0, mass_electron, quiet_start, save_run,
                       speed_of_light as c)
from jaxincell.theory import damped_mode, electrostatic_epsilon
from darkjaxincell import DarkField, DarkSimulation, midnight


def longitudinal_root(k, populations, omega_dark, eta, guess, model="full"):
    """Multispecies Landau root in units of total ``omega_p``.

    The parent supplies each drifting Maxwellian susceptibility. This reference
    supplies the independent Maxwell–Proca or screened-field algebra.
    """
    wp = np.sqrt(sum(species["wp"] ** 2 for species in populations))
    K, mu = k * c / wp, omega_dark / wp
    if model not in ("full", "ordinary", "quasistatic", "effective_charge"):
        raise ValueError(f"unknown longitudinal reference {model}")

    def determinant(z):
        eps, _ = electrostatic_epsilon(z * wp, k, populations)
        chi = eps - 1
        s = z**2 - K**2
        if model == "full":
            return (s - mu**2) * eps + eta**2 * s * chi
        if model == "quasistatic":
            return 1 + (1 + eta**2 * K**2 / (K**2 + mu**2)) * chi
        if model == "effective_charge":
            return 1 + (1 + eta**2) * chi
        return eps

    solved = root(lambda pair: (determinant(complex(*pair)).real,
                                determinant(complex(*pair)).imag),
                  (guess.real, guess.imag))
    answer = complex(*solved.x)
    residual = float(abs(determinant(answer)))
    if not solved.success or residual > 1e-7:
        raise RuntimeError(f"{model} longitudinal root did not converge: {residual:.2e}")
    return answer, residual


def mixed_root(kc_over_wp, sigma_over_c, eta, mu_over_wp):
    """Single-Maxwellian wrapper retaining the original normalized interface."""
    k = kc_over_wp / c
    population = ({"wp": 1.0, "u": 0.0, "vth": np.sqrt(2) * sigma_over_c * c},)
    return longitudinal_root(k, population, mu_over_wp, eta, 1.4 - 0.15j)


def fixed_window_mode(time, amplitude, window=(2.0, 12.0)):
    """Fit declared early absolute-field maxima without requiring a late floor."""
    peaks = find_peaks(amplitude)[0]
    peaks = peaks[(time[peaks] > window[0]) & (time[peaks] < window[1])]
    if peaks.size < 4:
        raise ValueError("fewer than four maxima in the declared Landau fit window")
    intervals = np.diff(time[peaks])
    frequency = np.pi / np.mean(intervals)
    regression = linregress(time[peaks], np.log(amplitude[peaks]))
    frequency_se = (frequency * np.std(intervals, ddof=1)
                    / (np.sqrt(intervals.size) * np.mean(intervals)))
    return regression.slope, frequency, peaks, regression.stderr, frequency_se


def main():
    parser = argparse.ArgumentParser(description="A mixed kinetic mode in the crypt")
    parser.add_argument("--full", action="store_true", help="150k-particle kinetic fit")
    parser.add_argument("--physical", action="store_true",
                        help="small thermal speed with fixed physical scales under refinement")
    parser.add_argument("--cells", type=int, help="grid cells (physical preset only)")
    parser.add_argument("--particles", type=int, help="electron markers (physical preset only)")
    parser.add_argument("--steps", type=int, help="field steps (physical preset only)")
    parser.add_argument("--output", type=Path, default=Path("artifacts/dark_kinetic"))
    args = parser.parse_args()
    if not args.physical and any(value is not None for value in
                                 (args.cells, args.particles, args.steps)):
        parser.error("resolution overrides require --physical")
    length, eta = 1.0, 0.3
    cells = (args.cells or 32) if args.physical else 64
    particles = ((args.particles or 80000) if args.full else (args.particles or 20000)) \
        if args.physical else (150000 if args.full else 20000)
    steps = ((args.steps or 3000) if args.full else (args.steps or 500)) \
        if args.physical else (1200 if args.full else 300)
    k = 2 * np.pi / length
    wp = (k * c / 10) if args.physical else (0.05 * c * cells / length)
    density = wp**2 * epsilon_0 * mass_electron / e**2
    vth = 0.5 / k * np.sqrt(2) * wp  # k lambda_D = 0.5; parent vth = sqrt(2) sigma
    x, v = quiet_start(particles, length, vth=(vth, 0.0, 0.0))
    v = v.at[:, 0].add(-jnp.mean(v[:, 0]))  # exactly current-neutral loading
    max_speed_over_c = float(jnp.max(jnp.linalg.norm(v, axis=1)) / c)
    x = x.at[:, 0].add(0.01 / k * jnp.sin(k * x[:, 0]))
    electrons = Species.electrons(particles, density=density, vth=(vth, 0.0, 0.0)).replace(x=x, v=v)
    plasma = Simulation(Domain(length=length, cells=cells, dt_over_dx_c=0.5), (electrons,))
    parent = plasma.run(steps, store_particles=False)
    output = DarkSimulation(plasma, DarkField(wp, eta)).run(steps, store_particles=False)
    time = np.asarray(output.ordinary.t) * wp
    parent_amplitude = np.abs(np.fft.rfft(np.asarray(parent.E[:, :, 0]), axis=1)[:, 1]) / cells
    amplitude = np.abs(np.fft.rfft(np.asarray(output.ordinary.E[:, :, 0]), axis=1)[:, 1]) / cells
    dark_amplitude = np.abs(np.fft.rfft(np.asarray(output.E[:, :, 0]), axis=1)[:, 1]) / cells
    mu = k * c if args.physical else wp
    analytic, residual = mixed_root(k * c / wp, vth / (np.sqrt(2) * c), eta, mu / wp)
    ordinary_root, ordinary_residual = mixed_root(k * c / wp, vth / (np.sqrt(2) * c), 0.0, mu / wp)
    populations = ({"wp": wp, "u": 0.0, "vth": float(vth)},)
    screened, screened_residual = longitudinal_root(
        k, populations, mu, eta, analytic, "quasistatic")
    effective, effective_residual = longitudinal_root(
        k, populations, mu, eta, analytic, "effective_charge")
    results = {"root_real_over_wp": analytic.real, "root_imag_over_wp": analytic.imag,
               "root_residual": residual, "parent_root_real_over_wp": ordinary_root.real,
               "parent_root_imag_over_wp": ordinary_root.imag,
               "parent_root_residual": ordinary_residual,
               "screened_root_over_wp": [screened.real, screened.imag],
               "screened_root_residual": screened_residual,
               "effective_charge_root_over_wp": [effective.real, effective.imag],
               "effective_charge_root_residual": effective_residual,
               "maximum_initial_speed_over_c": max_speed_over_c,
               "fit_status": "smoke only"}
    peaks = np.array([], dtype=int)
    total_energy = np.asarray(output.energy()["total_with_dark"])
    results["maximum_sampled_closed_energy_drift"] = float(
        np.max(np.abs(total_energy - total_energy[0])) / total_energy[0])
    results["maximum_all_step_balance_over_initial"] = float(
        output.state.max_balance_error / (output.state.initial_ordinary
                                          + output.state.initial_dark))
    if args.full:
        if args.physical:
            gamma, frequency, peaks, slope_se, frequency_se = fixed_window_mode(time, amplitude)
            parent_gamma, parent_frequency, parent_peaks, _, _ = fixed_window_mode(
                time, parent_amplitude)
            floor = parent_floor = None
            results["fit_window_sensitivity"] = {
                f"{lower:g}_{upper:g}": {
                    "mixed_gamma_over_wp": fixed_window_mode(time, amplitude, (lower, upper))[0],
                    "parent_gamma_over_wp": fixed_window_mode(
                        time, parent_amplitude, (lower, upper))[0]}
                for lower, upper in ((2.0, 14.0), (3.0, 16.0))}
        else:
            gamma, frequency, peaks, floor = damped_mode(time, amplitude)
            parent_gamma, parent_frequency, parent_peaks, parent_floor = damped_mode(
                time, parent_amplitude)
            slope_se = linregress(time[peaks], np.log(amplitude[peaks])).stderr
            intervals = np.diff(time[peaks])
            frequency_se = (frequency * np.std(intervals, ddof=1)
                            / (np.sqrt(intervals.size) * np.mean(intervals)))
        results.update({"fit_status": "resolved", "measured_real_over_wp": frequency,
                        "measured_imag_over_wp": gamma,
                        "frequency_relative_error": abs(frequency / analytic.real - 1),
                        "damping_relative_error": abs(gamma / analytic.imag - 1),
                        "damping_slope_stderr": slope_se,
                        "frequency_estimated_stderr": frequency_se,
                        "linear_window_wp_t": [2.0, 12.0] if args.physical
                        else [time[peaks[0]], time[peaks[-1]]],
                        "fitted_maxima_interval_wp_t": [time[peaks[0]], time[peaks[-1]]],
                        "maxima_used": int(peaks.size), "late_mode_floor_V_m": floor})
        results.update({"parent_measured_real_over_wp": parent_frequency,
                        "parent_measured_imag_over_wp": parent_gamma,
                        "parent_maxima_used": int(parent_peaks.size),
                        "parent_late_mode_floor_V_m": parent_floor,
                        "parent_frequency_relative_error": abs(parent_frequency / ordinary_root.real - 1),
                        "parent_damping_relative_error": abs(parent_gamma / ordinary_root.imag - 1)})

    with midnight():
        if args.physical:
            fig, (ax, energy_ax) = plt.subplots(1, 2, figsize=(11, 4.5),
                                                layout="constrained")
        else:
            fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
        ax.semilogy(time, parent_amplitude, label="JAX-in-Cell ordinary mode")
        ax.semilogy(time, amplitude, label="Dark-JAX-in-Cell ordinary mode")
        ax.semilogy(time, dark_amplitude, label="dark field mode")
        if peaks.size:
            ax.semilogy(time[peaks], amplitude[peaks], "o", label="fitted maxima")
            envelope = amplitude[peaks[0]] * np.exp(analytic.imag * (time - time[peaks[0]]))
            ax.semilogy(time, envelope, "--", label="independent mixed kinetic root")
            parent_envelope = parent_amplitude[parent_peaks[0]] * np.exp(
                ordinary_root.imag * (time - time[parent_peaks[0]]))
            ax.semilogy(time, parent_envelope, "--", label="ordinary Landau root")
            if floor is not None:
                ax.axhline(floor, ls=":", label="late mode floor")
        ax.set(xlabel=r"$\omega_p t$", ylabel=r"$|E_{x,k}|$ (V/m)",
               title="Matched loading: ordinary and mixed Landau modes")
        ax.grid(alpha=0.4)
        ax.legend(facecolor="#FFFFFF", edgecolor="#6B7280")
        if args.physical:
            energy_ax.plot(time, (total_energy - total_energy[0]) / total_energy[0])
            energy_ax.set(xlabel=r"$\omega_p t$", ylabel=r"$(U(t)-U(0))/U(0)$",
                          title="Complete particle + Maxwell + Proca energy")
            energy_ax.grid(alpha=0.4)
        settings = {"preset": "physical-full" if args.physical and args.full else
                    "physical-quick" if args.physical else "full" if args.full else "quick",
                    "cells": cells,
                    "particles": particles, "steps": steps, "length_m": length,
                    "omega_p_rad_s": wp, "eta": eta, "mu_over_wp": mu / wp,
                    "mu_over_kc": mu / (k * c), "sigma_over_c": vth / (np.sqrt(2) * c),
                    "k_lambda_D": 0.5, "seed_displacement_over_1_k": 0.01,
                    "background": "uniform fixed neutralizer",
                    "comparison": "same species arrays, grid, timestep, seed and ordinary initial field"}
        save_run(args.output, "dark_kinetic", settings, results, fig,
                 t=time, parent_amplitude=parent_amplitude,
                 ordinary_amplitude=amplitude, dark_amplitude=dark_amplitude,
                 parent_Ek=np.fft.rfft(np.asarray(parent.E[:, :, 0]), axis=1)[:, 1] / cells,
                 ordinary_Ek=np.fft.rfft(np.asarray(output.ordinary.E[:, :, 0]), axis=1)[:, 1] / cells,
                 dark_Ek=np.fft.rfft(np.asarray(output.E[:, :, 0]), axis=1)[:, 1] / cells,
                 effective_Ek=(np.fft.rfft(np.asarray(output.ordinary.E[:, :, 0]), axis=1)[:, 1]
                               + eta * np.fft.rfft(np.asarray(output.E[:, :, 0]), axis=1)[:, 1]) / cells,
                 total_energy=total_energy,
                 fitted_maxima=peaks)
        plt.close(fig)
    label = "MIXED KINETIC CHECK" if args.full else "SMOKE"
    print(f"🦇 {label}: root {analytic.real:.5f}{analytic.imag:+.5f}i", end="")
    if args.full:
        print(f"; PIC {frequency:.5f}{gamma:+.5f}i, {len(peaks)} maxima")
    else:
        print("; fit deferred to full preset")


if __name__ == "__main__":
    main()
