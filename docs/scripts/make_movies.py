"""Regenerate the two small, looping README views from production PIC histories."""

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
VIOLET, MINT, ORANGE = "#bd93f9", "#50fae4", "#ffb86c"


def _encode(fig, update, count, path):
    """Encode a short, browser-looping WebP without an ffmpeg/runtime dependency."""
    frames = []
    for index in range(count):
        update(index)
        fig.canvas.draw()
        frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:, :, :3]))
    path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(path, format="WEBP", save_all=True, append_images=frames[1:],
                   duration=140, loop=0, quality=48, method=6)
    plt.close(fig)
    return path.stat().st_size


def _energy_lines(ax, time, ledger, parent=None):
    """Keep the same energy scale across the two clips."""
    initial = float(np.asarray(ledger["total_with_dark"])[0])
    components = ((ledger["kinetic"], ORANGE, "particles"),
                  (ledger["electric"] + ledger["magnetic"], MINT, "ordinary EM"),
                  (ledger["dark"], VIOLET, "dark field"))
    for energy, color, label in components:
        values = np.maximum(np.asarray(energy) / initial, 1e-12)
        ax.plot(time, values, color=color, label=label, lw=1.5)
    if parent is not None:
        visible = np.asarray(parent["electric"] + parent["magnetic"]) / initial
        ax.plot(time, np.maximum(visible, 1e-12), "--", color="#f8f8f2",
                label="parent EM", lw=1.2)
    marker = ax.axvline(time[0], color="#eeeaf4", alpha=0.7, lw=1)
    ax.set(xlim=(time[0], time[-1]), ylim=(1e-9, 2), yscale="log",
           xlabel=r"$\omega t$", ylabel="energy / initial total")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower left", ncol=2, fontsize=7, facecolor="#232334", edgecolor="#8f899e")
    return marker


def longitudinal():
    """One Maxwellian loading evolves in matched parent and mixed systems."""
    cells, particles, steps, stride, length, eta = 64, 4096, 768, 24, 1.0, 0.3
    omega = 0.06 * c * cells / length
    density = omega**2 * epsilon_0 * mass_electron / e**2
    thermal = 0.26 * c
    k = 2 * np.pi / length
    x, v = quiet_start(particles, length, vth=(thermal, 0.0, 0.0))
    v = v.at[:, 0].add(-jnp.mean(v[:, 0]))
    x = x.at[:, 0].add(0.06 * jnp.sin(k * x[:, 0]) / k)
    species = Species.electrons(particles, density=density, vth=(thermal, 0.0, 0.0)).replace(x=x, v=v)
    plasma = Simulation(Domain(length=length, cells=cells, dt_over_dx_c=0.4), (species,))
    parent = plasma.run(steps, store_every=stride)
    out = DarkSimulation(plasma, DarkField(omega, eta)).run(steps, store_every=stride)
    time = np.asarray(out.ordinary.t) * omega
    grid = np.asarray(out.ordinary.grid) / length
    parent_field = np.asarray(parent.E[:, :, 0])
    ordinary = np.asarray(out.ordinary.E[:, :, 0])
    dark = np.asarray(out.E[:, :, 0])
    positions = (np.asarray(parent.x[:, ::5, 0]) / length,
                 np.asarray(out.ordinary.x[:, ::5, 0]) / length)
    speeds = (np.asarray(parent.v[:, ::5, 0]) / thermal,
              np.asarray(out.ordinary.v[:, ::5, 0]) / thermal)
    with midnight():
        fig = plt.figure(figsize=(8, 5), dpi=90, layout="constrained")
        axes = fig.subplot_mosaic([["parent", "mixed"], ["fields", "energy"]])
        scatters = []
        for name, color, title in (("parent", MINT, "JAX-in-Cell"),
                                   ("mixed", VIOLET, "Dark-JAX-in-Cell")):
            scatters.append(axes[name].scatter([], [], s=1.1, color=color,
                                               alpha=0.55, rasterized=True))
            axes[name].set(xlim=(-0.5, 0.5), ylim=(-2.5, 2.5), xlabel="$x/L$",
                           ylabel=r"$v_x/v_{\rm th}$", title=title)
        line_p, = axes["fields"].plot(grid, parent_field[0], color=MINT, label="parent $E_x$")
        line_e, = axes["fields"].plot(grid, ordinary[0], color=VIOLET, label="mixed $E_x$")
        line_d, = axes["fields"].plot(grid, dark[0], color=ORANGE, label="$E_{D,x}$")
        scale = max(np.max(np.abs(parent_field)), np.max(np.abs(ordinary)),
                    np.max(np.abs(dark))) * 1.1
        axes["fields"].set(xlim=(grid[0], grid[-1]), ylim=(-scale, scale), xlabel="$x/L$",
                           ylabel="V/m", title="Same initial plasma, different dynamics")
        axes["fields"].legend(facecolor="#232334", edgecolor="#8f899e", fontsize=7)
        marker = _energy_lines(axes["energy"], time, out.energy(), energies(parent))

        def update(index):
            for scatter, x, v in zip(scatters, positions, speeds):
                scatter.set_offsets(np.column_stack((x[index], v[index])))
            line_p.set_ydata(parent_field[index])
            line_e.set_ydata(ordinary[index])
            line_d.set_ydata(dark[index])
            marker.set_xdata([time[index], time[index]])
            fig.suptitle(f"One loading, two plasma futures  ·  $\\omega_p t={time[index]:.1f}$", fontsize=12)
        folder = MOVIES / "phase_space"
        size = _encode(fig, update, len(time), folder / "figure.webp")
    ledger = np.asarray(out.energy()["total_with_dark"])
    parent_ledger = np.asarray(energies(parent)["total"])
    save_run(folder, "dark_movie_phase_space",
             {"preset": "illustration", "cells": cells, "particles": particles,
              "steps": steps, "store_every": stride, "eta": eta,
              "omega_rad_s": omega, "thermal_m_s": thermal, "seed_displacement_over_1_k": 0.06,
              "matched_loading": "same species x, v, weights and ordinary initial field"},
             {"frames": len(time), "webp_bytes": size,
              "closed_energy_relative_drift": float(ledger[-1] / ledger[0] - 1),
              "parent_energy_relative_drift": float(parent_ledger[-1] / parent_ledger[0] - 1),
              "max_dark_gauss_V_m2": float(jnp.max(jnp.abs(out.dark_gauss())))},
             t=time, parent_mode=np.abs(np.fft.rfft(parent_field, axis=1)[:, 1]) / cells,
             mixed_mode=np.abs(np.fft.rfft(ordinary, axis=1)[:, 1]) / cells,
             parent_total=parent_ledger, mixed_total=ledger)


def transverse():
    """A Proca packet crosses the fixed-column slab from the full design example."""
    cells, particles_per_basis, steps, stride, omega0 = 128, 32, 240, 8, 1e9
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
        fig = plt.figure(figsize=(8, 5), dpi=90, layout="constrained")
        axes = fig.subplot_mosaic([["ordinary", "dark"], ["energy", "energy"]])
        lines = []
        for name, electric, magnetic, title in (("ordinary", ordinary_e, ordinary_b, "Photon field"),
                                                ("dark", dark_e, dark_b, "Dark field")):
            ax = axes[name]
            lines.append((ax.plot(grid, electric[0], color=MINT, label="$E_y$")[0],
                          ax.plot(grid, magnetic[0], color=VIOLET, label="$cB_z$")[0]))
            scale = max(np.max(np.abs(electric)), np.max(np.abs(magnetic))) * 1.1
            ax.set(xlim=(-25, 15), ylim=(-scale, scale), xlabel=r"$x/(c/\omega_0)$",
                   ylabel="V/m", title=title)
            ax.axvspan(-2, 2, color=ORANGE, alpha=0.13)
            ax.legend(facecolor="#232334", edgecolor="#8f899e", fontsize=8)
        marker = _energy_lines(axes["energy"], time, out.energy())

        def update(index):
            for pair, electric, magnetic in zip(lines, (ordinary_e, dark_e), (ordinary_b, dark_b)):
                pair[0].set_ydata(electric[index])
                pair[1].set_ydata(magnetic[index])
            marker.set_xdata([time[index], time[index]])
            fig.suptitle(f"A ghost crosses the slab  ·  $\\omega_0 t={time[index]:.1f}$", fontsize=12)
        folder = MOVIES / "slab_packet"
        size = _encode(fig, update, len(time), folder / "figure.webp")
    ledger = np.asarray(out.energy()["total_with_dark"])
    save_run(folder, "dark_movie_slab_packet",
             {"preset": "illustration", "cells": cells, "particles_per_basis_per_species": particles_per_basis,
              "steps": steps, "store_every": stride, "eta": eta, "mu_over_omega0": 0.6,
              "theta_from": "docs/_static/figures/profile_design/run.json", "theta": theta,
              "incident_dark_energy_J_m2": incident},
             {"frames": len(time), "webp_bytes": size,
              "closed_energy_relative_drift": float(ledger[-1] / ledger[0] - 1),
              "max_dark_gauss_V_m2": float(jnp.max(jnp.abs(out.dark_gauss())))})


if __name__ == "__main__":
    longitudinal()
    transverse()
