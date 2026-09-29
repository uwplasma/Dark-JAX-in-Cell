"""A mobile-ion pump and the finite reservoir behind its ghostly imitation.

This bounded-time, seeded control is not the long SHARP reproduction.  The
prescribed case has an external reservoir; the Proca cases start with the
same force but different finite field energies.  All share one loading.
"""

import argparse
from pathlib import Path

import jax.numpy as jnp
from jax import random
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm

from jaxincell import (Domain, Simulation, Species, elementary_charge as e,
                       epsilon_0, mass_electron, mass_proton, quiet_start,
                       save_run, speed_of_light as c)
from darkjaxincell import DarkField, DarkSimulation, PrescribedDrive, midnight
from darkjaxincell._proca import energy as dark_energy


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
    out = sim.run(steps, state=start, store_every=store_every, store_particles=False)
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
            for name, color in (("zero", "#8f899e"), ("external", "#bd93f9")):
                case = records[name]
                label = f"{name}, {count // 1000}k/species"
                linestyle = "-" if count == 40000 else "--"
                axes[0, 1].semilogy(t, np.maximum(case["nonzero_energy"] / case["initial_energy"][0], 1e-15),
                                    color=color, ls=linestyle, label=label)
                axes[1, 0].semilogy(t, case["mode"] / case["mode"][0],
                                    color=color, ls=linestyle)
        axes[0, 0].plot(t, histories["external"]["mean_E"], color="#50fae4", label="PIC")
        axes[0, 0].plot(t, histories["external"]["cold_mean"], "--", color="#ffb86c",
                        label="independent cold two-fluid")
        bars = [results[str(count)][name]["local_random_final_over_initial"]
                for count in (20000, 40000) for name in ("zero", "external")]
        axes[1, 1].bar(range(4), bars, color=["#8f899e", "#bd93f9"] * 2)
        axes[1, 1].set_xticks(range(4), ("20k zero", "20k drive", "40k zero", "40k drive"), rotation=15)
        axes[0, 0].set(xlabel=r"$\omega_p t$", ylabel="mean $E_x/F$", title="The early mean response")
        axes[0, 1].set(xlabel=r"$\omega_p t$", ylabel="nonzero-$k$ E energy / initial particle",
                       title="Higher modes, including noise")
        axes[1, 0].set(xlabel=r"$\omega_p t$", ylabel="seeded $k_1$ amplitude / initial",
                       title="One seeded mode")
        axes[1, 1].set(ylabel="cell-local random kinetic / initial", title=r"At $\omega_p t=80$")
        for ax in axes.flat:
            ax.grid(alpha=0.25)
        axes[0, 0].legend(facecolor="#232334", edgecolor="#8f899e")
        axes[0, 1].legend(facecolor="#232334", edgecolor="#8f899e", fontsize=8)
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


def main():
    parser = argparse.ArgumentParser(description="Follow a mobile-ion ghost and its finite reservoir")
    preset = parser.add_mutually_exclusive_group()
    preset.add_argument("--full", action="store_true", help="two physical resolutions")
    preset.add_argument("--paper-pilot", action="store_true", help="short paper-geometry loading check")
    parser.add_argument("--output", type=Path, default=Path("artifacts/dark_reservoir"))
    args = parser.parse_args()
    if args.paper_pilot:
        paper_geometry_pilot(args.output)
        return
    presets = ((64, 4000, 0.04, 40), (128, 8000, 0.02, 40)) if args.full else (
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
        colors = {"zero": "#8f899e", "external": "#ff79c6",
                  "small_reservoir": "#bd93f9", "large_reservoir": "#50fae4"}
        for name, record in histories.items():
            color = colors[name]
            if name != "zero":
                axes[0, 0].plot(t, record["pump"], color=color, label=name.replace("_", " "))
            axes[0, 1].plot(t, record["mean_E"], color=color, label=name.replace("_", " "))
            axes[1, 0].semilogy(t, record["mode"] / max(zero[0], 1e-30), color=color)
        axes[1, 1].bar(range(len(histories)),
                       [record["local_random_final"] / record["local_random_initial"]
                        for record in histories.values()], color=list(colors.values()))
        axes[0, 1].plot(t, histories["small_reservoir"]["cold_mean"], "--", color="#f8f8f2",
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
        axes[0, 0].legend(facecolor="#232334", edgecolor="#8f899e", fontsize="small")
        axes[0, 1].legend(facecolor="#232334", edgecolor="#8f899e", fontsize="small")
        settings = {**settings, "preset": "full" if args.full else "quick",
                    "refinements": [setting.copy() for _, setting in runs]}
        save_run(args.output, "dark_reservoir", settings, results, fig, t=t,
                 **{f"{name}_{key}": record[key] for name, record in histories.items()
                    for key in ("pump", "mean_E", "mode", "cold_mean")})
        plt.close(fig)
    print("🦇 MOBILE IONS: mean-oracle errors", results["refinements"],
          "; finite-reservoir energy balances",
          [results["cases"][name]["balance_over_scale"]
           for name in ("small_reservoir", "large_reservoir")])


if __name__ == "__main__":
    main()
