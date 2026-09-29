"""An oblique magnetic field makes all three ghost polarizations stir.

This cold 3V check compares PIC with an independent 12-state fluid matrix.
It is not a warm kinetic-root or nonlinear-instability benchmark.
"""

import argparse
from pathlib import Path

import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import expm

from jaxincell import (Domain, Simulation, Species, elementary_charge as e,
                       epsilon_0, mass_electron, speed_of_light as c, save_run)
from darkjaxincell import DarkField, DarkSimulation, midnight


def main():
    parser = argparse.ArgumentParser(description="Oblique midnight 3V response")
    parser.add_argument("--full", action="store_true", help="resolved cold-fluid comparison")
    parser.add_argument("--output", type=Path, default=Path("artifacts/dark_plasma"))
    args = parser.parse_args()
    cells, particles, steps = (32, 128, 200) if args.full else (16, 64, 100)
    omega0, p, eta, amplitude = 1e9, 0.8, 0.1, 1e-5
    n_ref = epsilon_0 * mass_electron * omega0**2 / e**2
    domain = Domain(length=2 * np.pi * c / omega0, cells=cells, dt_over_dx_c=0.2)
    magnetic = np.array([0.2, -0.15, 0.1]) * mass_electron * omega0 / e
    plasma = Simulation(domain, (Species.electrons(particles, density=p * n_ref),),
                        external_B=jnp.broadcast_to(magnetic, (cells, 3)))
    initial = jnp.broadcast_to(jnp.array([1.0, 0.4, -0.2]) * amplitude, (cells, 3))
    output = DarkSimulation(plasma, DarkField(omega0, eta, initial_E=initial)).run(steps)
    generator = np.zeros((12, 12))
    eye = np.eye(3)
    generator[0:3, 9:12] = -eye
    generator[3:6, 6:9] = eye
    generator[3:6, 9:12] = -eta * eye
    generator[6:9, 3:6] = -eye
    generator[9:12, 0:3] = p * eye
    generator[9:12, 3:6] = eta * p * eye
    bx, by, bz = magnetic
    cross_right = np.array([[0, bz, -by], [-bz, 0, bx], [by, -bx, 0]])
    generator[9:12, 9:12] = -e / (mass_electron * omega0) * cross_right
    state0 = np.r_[np.zeros(3), [1.0, 0.4, -0.2], np.zeros(6)]
    time = np.asarray(output.ordinary.t) * omega0
    reference = np.array([expm(generator * t) @ state0 for t in time])
    ordinary = np.asarray(output.ordinary.E).mean(axis=1) / amplitude
    dark = np.asarray(output.E).mean(axis=1) / amplitude
    error = float(np.max(np.abs(np.c_[ordinary, dark] - reference[:, :6])))

    with midnight():
        fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True, layout="constrained")
        for component, label in enumerate("xyz"):
            axes[0].plot(time, reference[:, component], label=f"E{label} cold oracle")
            axes[0].plot(time, ordinary[:, component], "--", lw=1.2, label=f"E{label} PIC")
            axes[1].plot(time, reference[:, 3 + component], label=f"dark E{label} cold oracle")
            axes[1].plot(time, dark[:, component], "--", lw=1.2, label=f"dark E{label} PIC")
        axes[0].set(ylabel=r"ordinary mean $E/D_0$", title="Oblique haunt: all three velocity components")
        axes[1].set(xlabel=r"$\omega_0 t$", ylabel=r"dark mean $E_D/D_0$")
        for ax in axes:
            ax.grid(alpha=0.4)
            ax.legend(facecolor="#FFFFFF", edgecolor="#6B7280", fontsize=7, ncol=2)
        settings = {"preset": "full" if args.full else "quick", "cells": cells,
                    "particles": particles, "steps": steps, "omega0_rad_s": omega0,
                    "density_ratio": p, "eta": eta, "B_T": magnetic.tolist()}
        save_run(args.output, "dark_plasma", settings,
                 {"max_all_six_field_error_over_D0": error}, fig,
                 t=time, ordinary_pic=ordinary, dark_pic=dark, cold_reference=reference)
        plt.close(fig)
    print(f"🦇 {'COLD 3V CHECK' if args.full else 'SMOKE'}: six-field error {error:.3e}")


if __name__ == "__main__":
    main()
