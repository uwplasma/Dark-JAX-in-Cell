"""A prescribed ghostly electric drive, with no backreacting dark reservoir."""

import argparse
from pathlib import Path

import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from jaxincell import (Domain, Simulation, Species, elementary_charge as e,
                       epsilon_0, mass_electron, speed_of_light as c, save_run)
from darkjaxincell import DarkSimulation, PrescribedDrive, midnight


def main():
    parser = argparse.ArgumentParser(description="A prescribed ghost with an external work ledger")
    parser.add_argument("--full", action="store_true", help="resolved resonant cold check")
    parser.add_argument("--output", type=Path, default=Path("artifacts/dark_drive"))
    args = parser.parse_args()
    cells, particles = (32, 128) if args.full else (16, 64)
    omega, eta, amplitude = 1e9, 0.05, 1e-5
    density_ratio = 1.0 if args.full else 0.8
    density = density_ratio * epsilon_0 * mass_electron * omega**2 / e**2
    domain = Domain(length=2 * np.pi * c / omega, cells=cells, dt_over_dx_c=0.2)
    plasma = Simulation(domain, (Species.electrons(particles, density=density),))
    drive = PrescribedDrive(eta, jnp.array([0.0, amplitude, 0.0]), omega)
    steps = round(10 / (omega * domain.dt))
    output = DarkSimulation(plasma, drive).run(steps)
    time = np.asarray(output.ordinary.t)
    wp = np.sqrt(density_ratio) * omega
    # This is the stable resonant limit of the forced cold oscillator.
    reference = (-eta * wp**2 * amplitude * time / (wp + omega)
                 * np.sin((wp + omega) * time / 2)
                 * np.sinc((wp - omega) * time / (2 * np.pi)))
    electric = np.asarray(output.ordinary.E[:, :, 1].mean(axis=1))
    waveform_error = float(np.max(np.abs(electric - reference)) / amplitude)
    ordinary = np.asarray(output.energy()["total"])
    work = np.asarray(output.work)
    transferred = work[-1] - work[0]
    work_error = float(abs(ordinary[-1] - ordinary[0] - transferred) / abs(transferred))
    with midnight():
        fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
        ax.plot(time * omega, reference / amplitude, label="cold forced oracle")
        ax.plot(time * omega, electric / amplitude, "--", label="PIC ordinary response")
        ax.set(xlabel=r"$\omega_0 t$", ylabel=r"mean $E_y/D_0$",
               title="The borrowed ghost: resonance without a dark reservoir")
        ax.grid(alpha=0.4)
        ax.legend(facecolor="#232334", edgecolor="#8f899e")
        settings = {"preset": "full" if args.full else "quick", "cells": cells,
                    "particles": particles, "steps": steps, "density_ratio": density_ratio,
                    "omega_rad_s": omega, "eta": eta, "drive_E0_V_m": amplitude}
        save_run(args.output, "dark_drive", settings,
                 {"max_waveform_error_over_D0": waveform_error,
                  "ordinary_energy_vs_external_work_relative_error": work_error},
                 fig, t=time * omega, ordinary_pic=electric / amplitude,
                 cold_reference=reference / amplitude, external_work=work)
        plt.close(fig)
    print(f"🦇 {'FORCED CHECK' if args.full else 'SMOKE'}: waveform {waveform_error:.3e}; "
          f"external-work balance {work_error:.3e}")


if __name__ == "__main__":
    main()
