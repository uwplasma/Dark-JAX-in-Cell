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
from scipy.special import wofz
from scipy.stats import linregress

from jaxincell import (Domain, Simulation, Species, elementary_charge as e,
                       epsilon_0, mass_electron, quiet_start, save_run,
                       speed_of_light as c)
from jaxincell.theory import damped_mode
from darkjaxincell import DarkField, DarkSimulation, midnight


def mixed_root(kc_over_wp, sigma_over_c, eta, mu_over_wp):
    """Independent Vlasov–Proca determinant with Landau continuation."""
    def dispersion(omega):
        zeta = omega / (np.sqrt(2) * kc_over_wp * sigma_over_c)
        plasma_z = 1j * np.sqrt(np.pi) * wofz(zeta)
        chi = (1 + zeta * plasma_z) / (kc_over_wp * sigma_over_c) ** 2
        s = omega**2 - kc_over_wp**2
        return (s - mu_over_wp**2) * (1 + chi) + eta**2 * s * chi

    solution = root(lambda pair: [dispersion(pair[0] + 1j * pair[1]).real,
                                  dispersion(pair[0] + 1j * pair[1]).imag],
                    [1.4, -0.15])
    frequency = complex(*solution.x)
    if not solution.success or abs(dispersion(frequency)) > 1e-8:
        raise RuntimeError("mixed kinetic reference root did not converge")
    return frequency, float(abs(dispersion(frequency)))


def main():
    parser = argparse.ArgumentParser(description="A mixed kinetic mode in the crypt")
    parser.add_argument("--full", action="store_true", help="150k-particle kinetic fit")
    parser.add_argument("--output", type=Path, default=Path("artifacts/dark_kinetic"))
    args = parser.parse_args()
    length, cells, eta = 1.0, 64, 0.3
    particles, steps = (150000, 1200) if args.full else (20000, 300)
    k = 2 * np.pi / length
    wp = 0.05 * c * cells / length
    density = wp**2 * epsilon_0 * mass_electron / e**2
    vth = 0.5 / k * np.sqrt(2) * wp  # k lambda_D = 0.5; parent vth = sqrt(2) sigma
    x, v = quiet_start(particles, length, vth=(vth, 0.0, 0.0))
    v = v.at[:, 0].add(-jnp.mean(v[:, 0]))  # exactly current-neutral loading
    x = x.at[:, 0].add(0.01 / k * jnp.sin(k * x[:, 0]))
    electrons = Species.electrons(particles, density=density, vth=(vth, 0.0, 0.0)).replace(x=x, v=v)
    plasma = Simulation(Domain(length=length, cells=cells, dt_over_dx_c=0.5), (electrons,))
    output = DarkSimulation(plasma, DarkField(wp, eta)).run(steps, store_particles=False)
    time = np.asarray(output.ordinary.t) * wp
    amplitude = np.abs(np.fft.rfft(np.asarray(output.ordinary.E[:, :, 0]), axis=1)[:, 1]) / cells
    dark_amplitude = np.abs(np.fft.rfft(np.asarray(output.E[:, :, 0]), axis=1)[:, 1]) / cells
    analytic, residual = mixed_root(k * c / wp, vth / (np.sqrt(2) * c), eta, 1.0)
    results = {"root_real_over_wp": analytic.real, "root_imag_over_wp": analytic.imag,
               "root_residual": residual, "fit_status": "smoke only"}
    peaks = np.array([], dtype=int)
    if args.full:
        gamma, frequency, peaks, floor = damped_mode(time, amplitude)
        regression = linregress(time[peaks], np.log(amplitude[peaks]))
        intervals = np.diff(time[peaks])
        frequency_se = (frequency * np.std(intervals, ddof=1)
                        / (np.sqrt(intervals.size) * np.mean(intervals)))
        results.update({"fit_status": "resolved", "measured_real_over_wp": frequency,
                        "measured_imag_over_wp": gamma,
                        "frequency_relative_error": abs(frequency / analytic.real - 1),
                        "damping_relative_error": abs(gamma / analytic.imag - 1),
                        "damping_slope_stderr": regression.stderr,
                        "frequency_estimated_stderr": frequency_se,
                        "linear_window_wp_t": [time[peaks[0]], time[peaks[-1]]],
                        "maxima_used": int(peaks.size), "late_mode_floor_V_m": floor})

    with midnight():
        fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
        ax.semilogy(time, amplitude, label="ordinary PIC mode")
        ax.semilogy(time, dark_amplitude, label="dark PIC mode")
        if peaks.size:
            ax.semilogy(time[peaks], amplitude[peaks], "o", label="fitted maxima")
            envelope = amplitude[peaks[0]] * np.exp(analytic.imag * (time - time[peaks[0]]))
            ax.semilogy(time, envelope, "--", label="independent mixed kinetic root")
            ax.axhline(results["late_mode_floor_V_m"], ls=":", label="late mode floor")
        ax.set(xlabel=r"$\omega_p t$", ylabel=r"$|E_{x,k}|$ (V/m)",
               title="A mixed Landau whisper: seeded current-neutral plasma")
        ax.grid(alpha=0.4)
        ax.legend(facecolor="#232334", edgecolor="#8f899e")
        settings = {"preset": "full" if args.full else "quick", "cells": cells,
                    "particles": particles, "steps": steps, "length_m": length,
                    "omega_p_rad_s": wp, "eta": eta, "mu_over_wp": 1.0,
                    "k_lambda_D": 0.5, "seed_displacement_over_1_k": 0.01,
                    "background": "uniform fixed neutralizer"}
        save_run(args.output, "dark_kinetic", settings, results, fig,
                 t=time, ordinary_amplitude=amplitude, dark_amplitude=dark_amplitude,
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
