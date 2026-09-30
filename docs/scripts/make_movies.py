"""Regenerate the README loops with ``python docs/scripts/make_movies.py [name]``."""

import json
from pathlib import Path
import subprocess
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
from scipy.stats import linregress  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SOURCE_SHA = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
sys.path.insert(0, str(ROOT))
from examples.dark_profile import make_simulation, packet, slab_basis  # noqa: E402
from examples.dark_bump import build_plasma  # noqa: E402

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


def _energy_lines(ax, time, ledger, parent=None, initial=None):
    """Keep a fixed energy scale while revealing each history frame by frame."""
    initial = float(np.asarray(ledger["total_with_dark"])[0]) if initial is None else initial
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


def _marker_indices(plasma, count=8192):
    """Sample markers in proportion to their represented physical number."""
    total = sum(float(species.density) for species in plasma.species)
    count = min(count, sum(species.n for species in plasma.species))
    start, selected = 0, []
    for i, species in enumerate(plasma.species):
        number = (count - len(selected) if i == len(plasma.species) - 1
                  else round(count * float(species.density) / total))
        selected.extend(np.linspace(start, start + species.n - 1, number, dtype=int))
        start += species.n
    return np.asarray(selected)


def _pair_history(plasma, dark_sim, frames, stride, chunk_frames, plot_indices, wp, scale_v):
    """Keep small plotted histories while restarting the full PIC state in chunks."""
    kept = {key: [] for key in ("time", "parent_E", "mixed_E", "dark_E",
                                "parent_x", "mixed_x", "parent_v", "mixed_v")}
    parent_ledger = {key: [] for key in ("electric", "magnetic", "total")}
    dark_ledger = {key: [] for key in ("kinetic", "electric", "magnetic", "dark",
                                       "total_with_dark")}
    parent_state = dark_state = None
    max_gauss = 0.0
    for first in range(0, frames, chunk_frames):
        count = min(chunk_frames, frames - first)
        parent = plasma.run(count * stride, store_every=stride, state=parent_state)
        out = dark_sim.run(count * stride, store_every=stride, state=dark_state)
        parent_state, dark_state = parent.state, out.state
        for key, values in (("time", out.ordinary.t * wp),
                            ("parent_E", parent.E[:, :, 0]),
                            ("mixed_E", out.ordinary.E[:, :, 0]),
                            ("dark_E", out.E[:, :, 0]),
                            ("parent_x", parent.x[:, plot_indices, 0] / plasma.domain.length),
                            ("mixed_x", out.ordinary.x[:, plot_indices, 0] / plasma.domain.length),
                            ("parent_v", parent.v[:, plot_indices, 0] / scale_v),
                            ("mixed_v", out.ordinary.v[:, plot_indices, 0] / scale_v)):
            kept[key].append(np.asarray(values))
        for ledger, energy in ((parent_ledger, energies(parent)),
                               (dark_ledger, out.energy())):
            for key in ledger:
                ledger[key].append(np.asarray(energy[key]))
        max_gauss = max(max_gauss, float(jnp.max(jnp.abs(out.dark_gauss()))))
    kept = {key: np.concatenate(values) for key, values in kept.items()}
    parent_ledger = {key: np.concatenate(values) for key, values in parent_ledger.items()}
    dark_ledger = {key: np.concatenate(values) for key, values in dark_ledger.items()}
    initial_parent = float(dark_state.initial_ordinary)
    initial_mixed = float(dark_state.initial_ordinary + dark_state.initial_dark)
    return kept, parent_ledger, dark_ledger, max_gauss, initial_parent, initial_mixed, dark_state


def _paired_movie(plasma, wp, mode, scale_v, name, horizon, frame_dt, subtitle,
                  chunk_frames=None):
    """Animate one loading in the two solvers with fixed shared axes."""
    eta, cells = 0.3, plasma.domain.cells
    stride = max(1, round(frame_dt / (wp * plasma.domain.dt)))
    steps = stride * round(horizon / (stride * wp * plasma.domain.dt))
    frames = steps // stride
    chunk_frames = frames if chunk_frames is None else chunk_frames
    dark_sim = DarkSimulation(plasma, DarkField(0.7 * wp, eta))
    plot_indices = _marker_indices(plasma)
    kept, parent_ledger, dark_ledger, max_gauss, initial_parent, initial_mixed, final_state = _pair_history(
        plasma, dark_sim, frames, stride, chunk_frames, plot_indices, wp, scale_v)
    time = kept["time"]
    grid = np.asarray(plasma.domain.faces) / plasma.domain.length
    fields = [kept[key] for key in ("parent_E", "mixed_E", "dark_E")]
    positions = [kept[key] for key in ("parent_x", "mixed_x")]
    speeds = [kept[key] for key in ("parent_v", "mixed_v")]
    limits = max(np.max(np.abs(speed)) for speed in speeds) * 1.04
    field_limit = max(np.max(np.abs(field)) for field in fields) * 1.06
    with midnight():
        fig = plt.figure(figsize=(11.2, 6.6), dpi=105, layout="constrained")
        axes = fig.subplot_mosaic([["parent", "mixed"], ["fields", "energy"]])
        scatters = []
        for key, color, title in (("parent", BLUE, "JAX-in-Cell"),
                                  ("mixed", VIOLET, "Dark-JAX-in-Cell")):
            scatters.append(axes[key].scatter([], [], s=0.7, color=color,
                                              alpha=0.45, rasterized=True))
            axes[key].set(xlim=(-0.5, 0.5), ylim=(-limits, limits), xlabel="$x/L$",
                          ylabel=r"$v_x/v_{\rm scale}$", title=title)
        field_lines = [axes["fields"].plot(grid, field[0], color=color, label=label)[0]
                       for field, color, label in zip(fields, (BLUE, VIOLET, ORANGE),
                                                      ("parent $E_x$", "mixed $E_x$", "$E_{D,x}$"))]
        axes["fields"].set(xlim=(grid[0], grid[-1]), ylim=(-field_limit, field_limit),
                           xlabel="$x/L$", ylabel="V/m", title="Electric fields")
        axes["fields"].legend(facecolor="#FFFFFF", edgecolor="#6B7280", fontsize=7)
        show_energy = _energy_lines(axes["energy"], time, dark_ledger, parent_ledger,
                                    initial=initial_mixed)

        def update(index):
            for scatter, x, v in zip(scatters, positions, speeds):
                scatter.set_offsets(np.column_stack((x[index], v[index])))
            for line, values in zip(field_lines, fields):
                line.set_ydata(values[index])
            show_energy(index)
            fig.suptitle(f"{subtitle}  ·  $\\omega_p t={time[index]:.1f}$", fontsize=12)
        folder = MOVIES / name
        size = _encode(fig, update, len(time), folder / "figure.webp")
    ledger = dark_ledger["total_with_dark"]
    parent_total = parent_ledger["total"]
    mode_amplitudes = [np.abs(np.fft.rfft(field, axis=1)[:, mode]) / cells for field in fields]
    results = {"frames": len(time), "webp_bytes": size,
               "closed_energy_relative_drift": float(ledger[-1] / initial_mixed - 1),
               "parent_energy_relative_drift": float(parent_total[-1] / initial_parent - 1),
               "max_closed_energy_relative_change": float(np.max(np.abs(ledger / initial_mixed - 1))),
               "max_parent_energy_relative_change": float(np.max(np.abs(parent_total / initial_parent - 1))),
               "max_dark_gauss_V_m2": max_gauss,
               "all_step_max_dark_gauss_V_m2": float(final_state.max_dark_gauss),
               "all_step_max_balance_J_m2": float(final_state.max_balance_error)}
    if name == "two_stream":
        reference = np.load(ROOT / "docs/_static/figures/two_stream_extended/data.npz")
        check = {"reference": "two_stream_extended case_1: 128 cells, 8000 particles/beam, same step"}
        for i, branch in enumerate(("parent", "mixed")):
            amplitude = mode_amplitudes[i]
            linear = (time >= 10) & (time <= 20)
            windows = {}
            for start, stop in ((10, 20), (20, 40), (40, 55), (55, 80), (80, 120)):
                window = (time >= start) & (time <= stop)
                baseline = np.interp(time[window], reference["case_1_t"],
                                     reference[f"case_1_{branch}_mode"])
                difference = np.linalg.norm(amplitude[window] - baseline)
                windows[f"{start}_{stop}"] = float(difference / np.linalg.norm(amplitude[window]))
            check[branch] = {"high_particle_growth_over_wp": float(linregress(
                             time[linear], np.log(amplitude[linear])).slope),
                             "mode_rms_difference_over_high": windows}
        results["particle_refinement"] = check
    save_run(folder, f"dark_movie_{name}",
             {"preset": "illustration", "source_git": SOURCE_SHA, "cells": cells,
              "particles_per_species": [s.n for s in plasma.species],
              "steps": steps, "store_every": stride, "eta": eta, "mu_over_wp": 0.7,
              "mode": mode, "horizon_omega_p": float(time[-1]),
              "initial_parent_energy_J_m2": initial_parent,
              "initial_mixed_energy_J_m2": initial_mixed,
              "chunk_frames": chunk_frames,
              "stored_samples_per_plasma_period": 2 * np.pi / (stride * wp * plasma.domain.dt),
              "particles_plotted_per_panel": positions[0].shape[1],
              "plotted_marker_counts": [int(np.count_nonzero(
                  (plot_indices >= sum(s.n for s in plasma.species[:i]))
                  & (plot_indices < sum(s.n for s in plasma.species[:i + 1]))))
                  for i in range(len(plasma.species))],
              "marker_display": "sampled by represented physical number; points are not density values",
              "matched_loading": "same particle arrays, weights, ordinary field, grid and timestep"},
             results,
             t=time, parent_mode=mode_amplitudes[0], mixed_mode=mode_amplitudes[1],
             dark_mode=mode_amplitudes[2], parent_total=parent_total, mixed_total=ledger)


def two_stream():
    """Follow matched cold counterstreams through trapping and late roll-up."""
    cells, particles = 128, 131072
    wp, length, drift = 0.05 * c * 64, 1.0, 0.25 * c
    density = wp**2 * epsilon_0 * mass_electron / e**2
    x1, v1 = quiet_start(particles, length, drift=(drift, 0, 0))
    x2, v2 = quiet_start(particles, length, drift=(-drift, 0, 0))
    x1 = x1.at[:, 0].add(0.001 * jnp.sin(2 * jnp.pi * x1[:, 0]) / (2 * jnp.pi))
    species = (Species.electrons(particles, density / 2, name="moonward").replace(x=x1, v=v1),
               Species.electrons(particles, density / 2, name="cryptward").replace(x=x2, v=v2))
    plasma = Simulation(Domain(length, cells, time_step=0.0125 / wp), species)
    _paired_movie(plasma, wp, 1, drift, "two_stream", 120, 1.0,
                  "Two streams meet the dark", chunk_frames=10)


def bump_on_tail():
    """A warm tail perturbs the same loaded bulk in both field systems."""
    plasma, wp, vth = build_plasma(128, 80000, 40000)
    _paired_movie(plasma, wp, 5, vth, "bump_on_tail", 45, 0.5, "A bump haunts the tail")


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
    centres = np.asarray(out.ordinary.grid) / ell
    faces = np.asarray(out.ordinary.faces) / ell
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
            lines.append((ax.plot(faces, electric[0], color=BLUE, label="$E_y$")[0],
                          ax.plot(centres, magnetic[0], color=VIOLET, label="$cB_z$")[0]))
            scale = max(np.max(np.abs(electric)), np.max(np.abs(magnetic))) * 1.1
            ax.set(xlim=(-25, 15), ylim=(-scale, scale), xlabel=r"$x/(c/\omega_0)$",
                   ylabel="V/m", title=title)
            ax.axvspan(-2, 2, color=ORANGE, alpha=0.13)
            ax.legend(facecolor="#FFFFFF", edgecolor="#6B7280", fontsize=8)
        initial = float(out.state.initial_ordinary + out.state.initial_dark)
        show_energy = _energy_lines(axes["energy"], time, out.energy(), initial=initial)

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
             {"preset": "illustration", "source_git": SOURCE_SHA, "cells": cells,
              "particles_per_basis_per_species": particles_per_basis,
              "steps": steps, "store_every": stride, "eta": eta, "mu_over_omega0": 0.6,
              "cells_across_slab": 4 * ell / domain.dx,
              "cells_per_carrier_wavelength": 2 * np.pi / (0.8 / ell * domain.dx),
              "stored_samples_per_carrier_period": 2 * np.pi / (stride * domain.dt * omega0),
              "theta_from": "docs/_static/figures/profile_design/run.json", "theta": theta,
              "incident_dark_energy_J_m2": incident},
             {"frames": len(time), "webp_bytes": size,
              "closed_energy_relative_drift": float(ledger[-1] / initial - 1),
              "max_dark_gauss_V_m2": float(jnp.max(jnp.abs(out.dark_gauss()))),
              "all_step_max_dark_gauss_V_m2": float(out.state.max_dark_gauss),
              "all_step_max_balance_J_m2": float(out.state.max_balance_error)})


if __name__ == "__main__":
    options = {"two_stream": two_stream, "bump_on_tail": bump_on_tail,
               "slab_packet": transverse}
    for name in sys.argv[1:] or options:
        if name not in options:
            raise SystemExit(f"choose from: {', '.join(options)}")
        options[name]()
