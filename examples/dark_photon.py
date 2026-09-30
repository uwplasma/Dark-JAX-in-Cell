"""A cold plasma meets a coherent ghost. Quick is smoke; full is a resolved check.

Run ``python examples/dark_photon.py --full`` to compare the self-consistent
Maxwell–Proca PIC mean fields with an independent cold matrix exponential.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
from scipy.linalg import expm

from jaxincell import (Domain, Simulation, Species, epsilon_0, mass_electron,
                       elementary_charge, speed_of_light as c, save_run)
from darkjaxincell import DarkField, DarkSimulation, midnight


def main():
    parser = argparse.ArgumentParser(description="Midnight Maxwell–Proca cold-plasma check")
    preset = parser.add_mutually_exclusive_group()
    preset.add_argument("--quick", action="store_true", help="smoke run only")
    preset.add_argument("--full", action="store_true", help="resolved cold-reference run")
    parser.add_argument("--output", type=Path, default=Path("artifacts/dark_photon"))
    args = parser.parse_args()
    quick = not args.full
    cells = 16 if quick else 64
    particles = 4 * cells
    omega0, eta, p, amplitude = 1e9, 0.05, 0.8, 1e-5
    n_ref = epsilon_0 * mass_electron * omega0**2 / elementary_charge**2
    domain = Domain(length=2 * np.pi * c / omega0, cells=cells, dt_over_dx_c=0.2)
    plasma = Simulation(domain, (Species.electrons(particles, density=p * n_ref),))
    initial = jnp.broadcast_to(jnp.array([0.0, amplitude, 0.0]), (cells, 3))
    simulation = DarkSimulation(plasma, DarkField(omega0, eta, initial_E=initial))
    steps = round(20 / (omega0 * domain.dt))
    output = simulation.run(steps)
    t = np.asarray(output.ordinary.t) * omega0
    generator = np.array([[0, 0, 0, -1], [0, 0, 1, -eta],
                          [0, -1, 0, 0], [p, eta * p, 0, 0]])
    reference = np.array([expm(generator * time) @ [0, 1, 0, 0] for time in t])
    ordinary = np.asarray(output.ordinary.E[:, :, 1].mean(axis=1)) / amplitude
    dark = np.asarray(output.E[:, :, 1].mean(axis=1)) / amplitude
    error = float(np.max(np.abs(np.stack([ordinary, dark], axis=1) - reference[:, :2])))
    gauss_scale = elementary_charge * n_ref / epsilon_0
    gauss_error = float(np.max(np.abs(np.asarray(output.dark_gauss()))) / gauss_scale)
    ledger = output.energy()
    closed = np.asarray(ledger["total_with_dark"])
    ordinary_energy = np.asarray(ledger["total"])
    transfer = -float(output.work[-1])
    initial_ordinary = float(output.state.initial_ordinary)
    initial_closed = float(output.state.initial_ordinary + output.state.initial_dark)
    closed_error = float(abs(closed[-1] / initial_closed - 1))
    work_error = float(abs(ordinary_energy[-1] - initial_ordinary - transfer) / abs(transfer))

    with midnight():
        fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
        ax.plot(t, reference[:, 0], label="ordinary cold oracle", lw=2)
        ax.plot(t, ordinary, "--", label="ordinary PIC", lw=1.5)
        ax.plot(t, reference[:, 1], label="dark cold oracle", lw=2)
        ax.plot(t, dark, "--", label="dark PIC", lw=1.5)
        ax.set(xlabel=r"$\omega_0 t$", ylabel=r"mean $E_y/D_0$",
               title="The ghost shares its energy: homogeneous cold plasma")
        ax.grid(alpha=0.4)
        ax.legend(facecolor="#FFFFFF", edgecolor="#6B7280")
        settings = {"preset": "quick" if quick else "full", "cells": cells,
                    "particles": particles, "steps": steps, "omega0_rad_s": omega0,
                    "density_ratio": p, "eta": eta, "dark_E0_V_m": amplitude}
        results = {"max_mean_field_error_over_D0": error,
                   "max_dark_gauss_over_enref_eps0": gauss_error,
                   "closed_energy_relative_drift": closed_error,
                   "ordinary_energy_vs_dark_work_relative_error": work_error}
        save_run(args.output, "dark_photon", settings, results, fig, t=t,
                 ordinary_pic=ordinary, dark_pic=dark,
                 ordinary_oracle=reference[:, 0], dark_oracle=reference[:, 1])
        plt.close(fig)
    print(f"🦇 {'SMOKE' if quick else 'COLD CHECK'}: mean-field {error:.3e}; "
          f"dark Gauss {gauss_error:.3e}; closed energy {closed_error:.3e}; "
          f"work balance {work_error:.3e}")


if __name__ == "__main__":
    main()
