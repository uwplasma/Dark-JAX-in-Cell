"""Regenerate the README loops with ``python docs/scripts/make_movies.py [name]``."""

import json
from pathlib import Path
import sys

import jax.numpy as jnp
import matplotlib
import numpy as np
from PIL import Image
from jaxincell import (Domain, Simulation, Species, elementary_charge as e,
                       energies, epsilon_0, mass_electron, quiet_start, save_run,
                       speed_of_light as c)
from darkjaxincell import DarkField, DarkSimulation, midnight

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from examples.dark_profile import make_simulation, packet, slab_basis  # noqa: E402

MOVIES = ROOT / "docs" / "_static" / "movies"
VIOLET, BLUE, ORANGE = "#6A3D9A", "#0072B2", "#D55E00"


def _encode(fig, update, count, path):
    """Encode a short, browser-looping WebP without an ffmpeg/runtime dependency."""
    frames = []
    for index in range(count):
        update(index)
        fig.canvas.draw()
        frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:, :, :3]))
    path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(path, format="WEBP", save_all=True, append_images=frames[1:],
                   duration=110, loop=0, quality=30, method=6)
    plt.close(fig)
    return path.stat().st_size


def _energy_lines(ax, time, ledger, parent=None):
    """Keep a fixed energy scale while revealing each history frame by frame."""
    initial = float(np.asarray(ledger["total_with_dark"])[0])
    components = ((ledger["kinetic"], ORANGE, "particles"),
                  (ledger["electric"] + ledger["magnetic"], BLUE, "ordinary EM"),
                  (ledger["dark"], VIOLET, "dark field"))
    traces = []
    for energy, color, label in components:
        values = np.maximum(np.asarray(energy) / initial, 1e-12)
        ax.plot(time, values, color=color, alpha=0.18, lw=1)
        line, = ax.plot([], [], color=color, label=label, lw=1.8)
        traces.append((line, values))
    if parent is not None:
        visible = np.asarray(parent["electric"] + parent["magnetic"]) / initial
        visible = np.maximum(visible, 1e-12)
        ax.plot(time, visible, "--", color="#202124", alpha=0.18, lw=1)
        line, = ax.plot([], [], "--", color="#202124", label="parent EM", lw=1.4)
        traces.append((line, visible))
    marker = ax.axvline(time[0], color="#30343B", alpha=0.7, lw=1)
    ax.set(xlim=(time[0], time[-1]), ylim=(1e-9, 2), yscale="log",
           xlabel=r"$\omega t$", ylabel="energy / initial total")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower left", ncol=2, fontsize=7, facecolor="#FFFFFF", edgecolor="#6B7280")

    def show(index):
        for line, values in traces:
            line.set_data(time[:index + 1], values[:index + 1])
        marker.set_xdata([time[index], time[index]])

    return show


def _seeded_plasma(particles, cells, length, omega, thermal):
    """Load the same quiet Maxwellian and mode at either particle count."""
    density = omega**2 * epsilon_0 * mass_electron / e**2
    k = 2 * np.pi / length
    x, v = quiet_start(particles, length, vth=(thermal, 0.0, 0.0))
    v = v.at[:, 0].add(-jnp.mean(v[:, 0]))
    x = x.at[:, 0].add(0.06 * jnp.sin(k * x[:, 0]) / k)
    species = Species.electrons(particles, density=density, vth=(thermal, 0.0, 0.0)).replace(x=x, v=v)
    return Simulation(Domain(length=length, cells=cells, dt_over_dx_c=0.4), (species,))


def longitudinal():
    """One Maxwellian loading evolves in matched parent and mixed systems."""
    cells, particles, steps, stride, length, eta = 128, 32768, 1536, 32, 1.0, 0.3
    omega = 0.06 * c * 64 / length
    thermal = 0.26 * c
    plasma = _seeded_plasma(particles, cells, length, omega, thermal)
    parent = plasma.run(steps, store_every=stride)
    out = DarkSimulation(plasma, DarkField(omega, eta)).run(steps, store_every=stride)
    coarse_plasma = _seeded_plasma(particles // 4, cells, length, omega, thermal)
    coarse_parent = coarse_plasma.run(steps, store_every=stride, store_particles=False)
    coarse_mixed = DarkSimulation(coarse_plasma, DarkField(omega, eta)).run(
        steps, store_every=stride, store_particles=False)
    time = np.asarray(out.ordinary.t) * omega
    grid = np.asarray(out.ordinary.grid) / length
    parent_field = np.asarray(parent.E[:, :, 0])
    ordinary = np.asarray(out.ordinary.E[:, :, 0])
    dark = np.asarray(out.E[:, :, 0])

    def mode(fields):
        return np.abs(np.fft.rfft(np.asarray(fields), axis=1)[:, 1]) / cells
    parent_mode, mixed_mode = mode(parent_field), mode(ordinary)
    coarse_parent_mode = mode(coarse_parent.E[:, :, 0])
    coarse_mixed_mode = mode(coarse_mixed.ordinary.E[:, :, 0])
    # Quiet-start velocity pairs have opposite signs: an even stride hides one half.
    positions = (np.asarray(parent.x[:, ::9, 0]) / length,
                 np.asarray(out.ordinary.x[:, ::9, 0]) / length)
    speeds = (np.asarray(parent.v[:, ::9, 0]) / thermal,
              np.asarray(out.ordinary.v[:, ::9, 0]) / thermal)
    with midnight():
        fig = plt.figure(figsize=(11.2, 6.6), dpi=105, layout="constrained")
        axes = fig.subplot_mosaic([["parent", "mixed"], ["fields", "energy"]])
        scatters = []
        for name, color, title in (("parent", BLUE, "JAX-in-Cell"),
                                   ("mixed", VIOLET, "Dark-JAX-in-Cell")):
            scatters.append(axes[name].scatter([], [], s=0.9, color=color,
                                               alpha=0.48, rasterized=True))
            axes[name].set(xlim=(-0.5, 0.5), ylim=(-2.5, 2.5), xlabel="$x/L$",
                           ylabel=r"$v_x/v_{\rm th}$", title=title)
        line_p, = axes["fields"].plot(grid, parent_field[0], color=BLUE, label="parent $E_x$")
        line_e, = axes["fields"].plot(grid, ordinary[0], color=VIOLET, label="mixed $E_x$")
        line_d, = axes["fields"].plot(grid, dark[0], color=ORANGE, label="$E_{D,x}$")
        scale = max(np.max(np.abs(parent_field)), np.max(np.abs(ordinary)),
                    np.max(np.abs(dark))) * 1.1
        axes["fields"].set(xlim=(grid[0], grid[-1]), ylim=(-scale, scale), xlabel="$x/L$",
                           ylabel="V/m", title="Longitudinal electric fields")
        axes["fields"].legend(facecolor="#FFFFFF", edgecolor="#6B7280", fontsize=7)
        show_energy = _energy_lines(axes["energy"], time, out.energy(), energies(parent))

        def update(index):
            for scatter, x, v in zip(scatters, positions, speeds):
                scatter.set_offsets(np.column_stack((x[index], v[index])))
            line_p.set_ydata(parent_field[index])
            line_e.set_ydata(ordinary[index])
            line_d.set_ydata(dark[index])
            show_energy(index)
            fig.suptitle(f"One loading, two plasma futures  ·  $\\omega_p t={time[index]:.1f}$", fontsize=12)
        folder = MOVIES / "phase_space"
        size = _encode(fig, update, len(time), folder / "figure.webp")
    ledger = np.asarray(out.energy()["total_with_dark"])
    parent_ledger = np.asarray(energies(parent)["total"])
    save_run(folder, "dark_movie_phase_space",
             {"preset": "illustration", "cells": cells, "particles": particles,
              "particle_check_count": particles // 4,
              "steps": steps, "store_every": stride, "eta": eta,
              "omega_rad_s": omega, "thermal_m_s": thermal, "seed_displacement_over_1_k": 0.06,
              "cells_per_Debye_length": thermal / (np.sqrt(2) * omega * plasma.domain.dx),
              "stored_samples_per_plasma_period": 2 * np.pi / (omega * stride * plasma.domain.dt),
              "particles_plotted_per_panel": positions[0].shape[1],
              "matched_loading": "same species x, v, weights and ordinary initial field"},
             {"frames": len(time), "webp_bytes": size,
              "mode_rms_relative_particle_change": {
                  "parent": float(np.linalg.norm(parent_mode - coarse_parent_mode)
                                  / np.linalg.norm(parent_mode)),
                  "mixed": float(np.linalg.norm(mixed_mode - coarse_mixed_mode)
                                 / np.linalg.norm(mixed_mode))},
              "closed_energy_relative_drift": float(ledger[-1] / ledger[0] - 1),
              "parent_energy_relative_drift": float(parent_ledger[-1] / parent_ledger[0] - 1),
              "max_dark_gauss_V_m2": float(jnp.max(jnp.abs(out.dark_gauss())))},
             t=time, parent_mode=parent_mode, mixed_mode=mixed_mode,
             coarse_parent_mode=coarse_parent_mode, coarse_mixed_mode=coarse_mixed_mode,
             parent_total=parent_ledger, mixed_total=ledger)


def transverse():
    """A Proca packet crosses the fixed-column slab from the full design example."""
    cells, particles_per_basis, steps, stride, omega0 = 512, 256, 960, 12, 1e9
    ell, mu, eta = c / omega0, 0.6 * omega0, 0.05
    domain = Domain(length=80 * ell, cells=cells, dt_over_dx_c=0.4)
    initial_E, initial_A, incident, _, _ = packet(domain, mu, 0.8 / ell, -16 * ell,
                                                  3.5 * ell, 2e-5)
    theta = json.loads((ROOT / "docs/_static/figures/profile_design/run.json").read_text())["results"]["best"]["theta"]
    positions = slab_basis(particles_per_basis)[2]
    nref = epsilon_0 * mass_electron * omega0**2 / e**2
    sim = make_simulation(theta, cells, particles_per_basis, eta, mu, 1.5 * nref * ell,
                          positions, initial_E, initial_A, length_units=80)
    out = sim.run(steps, store_every=stride)
    time = np.asarray(out.ordinary.t) * omega0
    grid = np.asarray(out.ordinary.grid) / ell
    ordinary_e = np.asarray(out.ordinary.E[:, :, 1])
    ordinary_b = c * np.asarray(out.ordinary.B[:, :, 2])
    dark_e = np.asarray(out.E[:, :, 1])
    dark_b = c * np.asarray(out.B[:, :, 2])
    with midnight():
        fig = plt.figure(figsize=(11.2, 6.6), dpi=105, layout="constrained")
        axes = fig.subplot_mosaic([["ordinary", "dark"], ["energy", "energy"]])
        lines = []
        for name, electric, magnetic, title in (("ordinary", ordinary_e, ordinary_b, "Photon field"),
                                                ("dark", dark_e, dark_b, "Dark field")):
            ax = axes[name]
            lines.append((ax.plot(grid, electric[0], color=BLUE, label="$E_y$")[0],
                          ax.plot(grid, magnetic[0], color=VIOLET, label="$cB_z$")[0]))
            scale = max(np.max(np.abs(electric)), np.max(np.abs(magnetic))) * 1.1
            ax.set(xlim=(-25, 15), ylim=(-scale, scale), xlabel=r"$x/(c/\omega_0)$",
                   ylabel="V/m", title=title)
            ax.axvspan(-2, 2, color=ORANGE, alpha=0.13)
            ax.legend(facecolor="#FFFFFF", edgecolor="#6B7280", fontsize=8)
        show_energy = _energy_lines(axes["energy"], time, out.energy())

        def update(index):
            for pair, electric, magnetic in zip(lines, (ordinary_e, dark_e), (ordinary_b, dark_b)):
                pair[0].set_ydata(electric[index])
                pair[1].set_ydata(magnetic[index])
            show_energy(index)
            fig.suptitle(f"A ghost crosses the slab  ·  $\\omega_0 t={time[index]:.1f}$", fontsize=12)
        folder = MOVIES / "slab_packet"
        size = _encode(fig, update, len(time), folder / "figure.webp")
    ledger = np.asarray(out.energy()["total_with_dark"])
    save_run(folder, "dark_movie_slab_packet",
             {"preset": "illustration", "cells": cells, "particles_per_basis_per_species": particles_per_basis,
              "steps": steps, "store_every": stride, "eta": eta, "mu_over_omega0": 0.6,
              "cells_across_slab": 4 * ell / domain.dx,
              "cells_per_carrier_wavelength": 2 * np.pi / (0.8 / ell * domain.dx),
              "stored_samples_per_carrier_period": 2 * np.pi / (stride * domain.dt * omega0),
              "theta_from": "docs/_static/figures/profile_design/run.json", "theta": theta,
              "incident_dark_energy_J_m2": incident},
             {"frames": len(time), "webp_bytes": size,
              "closed_energy_relative_drift": float(ledger[-1] / ledger[0] - 1),
              "max_dark_gauss_V_m2": float(jnp.max(jnp.abs(out.dark_gauss())))})


if __name__ == "__main__":
    options = {"phase_space": longitudinal, "slab_packet": transverse}
    for name in sys.argv[1:] or options:
        if name not in options:
            raise SystemExit(f"choose from: {', '.join(options)}")
        options[name]()
