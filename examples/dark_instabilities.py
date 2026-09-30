"""Current-neutral mixed two-stream and transverse anisotropy checks.

The references describe ordinary charged matter coupled to both fields. They
are not dark-charged-particle instabilities. Full presets fit declared early
linear windows; quick presets only check that each configuration runs.
"""

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq
from scipy.special import wofz
from scipy.stats import linregress

from jaxincell import (Domain, Simulation, Species, elementary_charge as e,
                       epsilon_0, mass_electron, quiet_start, save_run,
                       speed_of_light as c)
from darkjaxincell import DarkField, DarkSimulation, midnight
if __package__:
    from .dark_kinetic import longitudinal_root
else:
    from dark_kinetic import longitudinal_root


def two_stream_growth(kc_over_wp, ku_over_wp, mu_over_wp, eta):
    """Cold symmetric ordinary-beam determinant, excluding beam poles."""
    z = np.poly1d([1, 0])  # omega^2 / omega_p^2
    pole = ku_over_wp**2
    polynomial = ((z - kc_over_wp**2 - mu_over_wp**2)
                  * ((z - pole)**2 - (z + pole))
                  - eta**2 * (z - kc_over_wp**2) * (z + pole))
    roots = np.roots(polynomial)
    negative = [root.real for root in roots if abs(root.imag) < 1e-10 and root.real < 0]
    if len(negative) != 1:
        raise RuntimeError("cold mixed two-stream root is not simple and unstable")
    gamma = np.sqrt(-negative[0])
    omega = 1j * gamma
    chi = -(omega**2 + pole) / (omega**2 - pole)**2
    s = omega**2 - kc_over_wp**2
    residual = (s - mu_over_wp**2) * (1 + chi) + eta**2 * s * chi
    return float(gamma), float(abs(residual))


def cold_screened_growth(kc_over_wp, ku_over_wp, mu_over_wp, eta, model):
    """Cold-beam ordinary, quasistatic Yukawa or constant-charge surrogate."""
    K, b = kc_over_wp, ku_over_wp
    if model == "ordinary":
        alpha = 1.0
    elif model == "quasistatic":
        alpha = 1 + eta**2 * K**2 / (K**2 + mu_over_wp**2)
    elif model == "effective_charge":
        alpha = 1 + eta**2
    else:
        raise ValueError(f"unknown cold surrogate {model}")
    if b**2 >= alpha:
        return 0.0
    positive = b**2 + alpha / 2 + np.sqrt(alpha**2 + 8 * alpha * b**2) / 2
    return float(np.sqrt(b**2 * (alpha - b**2) / positive))


def weibel_cutoff_squared(Q, mu_over_wp, eta):
    """Stable quadratic evaluation of the transverse marginal ``(kc/wp)^2``."""
    if Q <= 0:
        raise ValueError("positive anisotropy drive Q is required")
    mass2 = mu_over_wp**2
    b = (1 + eta**2) * Q - mass2
    discriminant = np.hypot(b, 2 * mu_over_wp * np.sqrt(Q))
    return float(2 * mass2 * Q / (discriminant - b) if b < 0
                 else (b + discriminant) / 2)


def screening_references(K, b, mu, eta):
    """Cold controls and the fraction of the full shift explained by screening."""
    values = {model: cold_screened_growth(K, b, mu, eta, model)
              for model in ("ordinary", "quasistatic", "effective_charge")}
    values["full"] = two_stream_growth(K, b, mu, eta)[0]
    values["quasistatic_fraction_of_full_shift"] = (
        (values["quasistatic"] - values["ordinary"])
        / (values["full"] - values["ordinary"]))
    return values


def plot_screened_growth(ax, screening, K, b):
    """Show how much of a cold mass scan is Yukawa screening."""
    masses = np.geomspace(0.1, 10, 61)
    curves = {"mass_over_kc": masses / K,
              "full_growth_over_wp": np.array([
                  two_stream_growth(K, b, mass, 0.3)[0] for mass in masses]),
              "quasistatic_growth_over_wp": np.array([
                  cold_screened_growth(K, b, mass, 0.3, "quasistatic")
                  for mass in masses])}
    for key, style, label in (("full_growth_over_wp", "-", "full Proca"),
                              ("quasistatic_growth_over_wp", "--", "Yukawa limit")):
        ax.semilogx(curves["mass_over_kc"], curves[key] - screening["ordinary"],
                    style, label=label)
    ax.axvline(0.7 / K, color="#30343B", ls=":", label="PIC setting")
    ax.set(xlabel=r"$\Omega_D/(kc)$", ylabel=r"$\Delta\gamma/\omega_p$",
           title="Cold screening explains the shift")
    ax.text(0.04, 0.05,
            f"{100 * screening['quasistatic_fraction_of_full_shift']:.3f}% screened\n"
            "at dotted setting", transform=ax.transAxes, fontsize=9,
            bbox={"facecolor": "white", "edgecolor": "#6B7280", "alpha": 0.9})
    ax.grid(alpha=0.4)
    ax.legend(facecolor="#FFFFFF", edgecolor="#6B7280", fontsize=8)
    return curves


def weibel_growth(kc_over_wp, sigma_over_c, anisotropy, mu_over_wp, eta):
    """Bi-Maxwellian transverse determinant continued to omega=i gamma."""
    def determinant(gamma):
        zeta = 1j * gamma / (np.sqrt(2) * kc_over_wp * sigma_over_c)
        plasma_z = 1j * np.sqrt(np.pi) * wofz(zeta)
        response = 1 - anisotropy * (1 + zeta * plasma_z)
        s = -gamma**2 - kc_over_wp**2
        return ((s - mu_over_wp**2) * (s - response) - eta**2 * s * response).real

    gamma = brentq(determinant, 1e-10, 1.0)
    return float(gamma), float(abs(determinant(gamma)))


def two_stream_run(cells, particles, steps, eta):
    """Two equal cold beams with exactly cancelling finite-loading mean current."""
    length, k = 1.0, 2 * np.pi
    wp = 0.05 * c * 64
    density = wp**2 * epsilon_0 * mass_electron / e**2
    drift = 0.25 * c
    plus_x, plus_v = quiet_start(particles, length, drift=(drift, 0, 0))
    minus_x, minus_v = quiet_start(particles, length, drift=(-drift, 0, 0))
    plus_x = plus_x.at[:, 0].add(0.001 / k * jnp.sin(k * plus_x[:, 0]))
    species = (Species.electrons(particles, density / 2, name="moonward").replace(x=plus_x, v=plus_v),
               Species.electrons(particles, density / 2, name="cryptward").replace(x=minus_x, v=minus_v))
    plasma = Simulation(Domain(length, cells, dt_over_dx_c=0.5), species)
    sim = DarkSimulation(plasma, DarkField(0.7 * wp, eta))
    state, _ = sim.initial_state(jax.random.PRNGKey(0))
    charge = plasma.per_particle[1]
    net_current = float(jnp.sum(charge * state.ordinary.w * plasma._velocity(state.ordinary.u)[:, 0])
                        / length)
    output = sim.run(steps, store_every=steps // 200, store_particles=False)
    t = np.asarray(output.ordinary.t) * wp
    amplitude = np.abs(np.fft.rfft(np.asarray(output.ordinary.E[:, :, 0]), axis=1)[:, 1]) / cells
    return t, amplitude, net_current, float(jnp.max(jnp.abs(output.dark_gauss())))


def weibel_run(cells, particles, steps, eta, mode=1):
    """Anisotropic quiet Maxwellian with a seeded transverse magnetic mode."""
    length, k = 1.0, 2 * np.pi * mode
    wp = 2 * np.pi * c  # fixed plasma while the spatial mode changes
    density = wp**2 * epsilon_0 * mass_electron / e**2
    vth = np.sqrt(2) * 0.08 * c
    x, v = quiet_start(particles, length, vth=(vth, 0, 2 * vth))
    v = v.at[:, 0].add(-jnp.mean(v[:, 0]))
    v = v.at[:, 2].add(-jnp.mean(v[:, 2]))
    electrons = Species.electrons(particles, density, vth=(vth, 0, 2 * vth)).replace(x=x, v=v)
    plasma = Simulation(Domain(length, cells, dt_over_dx_c=0.5), (electrons,))
    sim = DarkSimulation(plasma, DarkField(0.7 * wp, eta))
    state, _ = sim.initial_state(jax.random.PRNGKey(0))
    seed = 100.0
    ordinary = state.ordinary.replace(
        E=state.ordinary.E.at[:, 2].set(seed * jnp.cos(k * plasma.domain.faces)),
        B=state.ordinary.B.at[:, 1].set(-20 * seed / c * jnp.sin(k * plasma.domain.grid)))
    state = sim.continue_with_parameters(state.replace(ordinary=ordinary))
    output = sim.run(steps, store_every=steps // 200, store_particles=False, state=state)
    t = np.asarray(output.ordinary.t) * wp
    amplitude = np.abs(np.fft.rfft(np.asarray(output.ordinary.B[:, :, 1]), axis=1)[:, mode]) / cells
    net_current = float(jnp.sum(plasma.per_particle[1] * state.ordinary.w
                                * plasma._velocity(state.ordinary.u)[:, 0]) / length)
    return t, amplitude, net_current, float(jnp.max(jnp.abs(output.dark_gauss())))


def weibel_cutoff_scan():
    """Transverse marginal wavenumber at several dark masses and mixings."""
    mixings = np.linspace(0, 0.6, 61)
    masses = (0.1, 0.7, 2.0, 10.0)
    curves = {f"cutoff_mu_{mass:g}": np.array([
        np.sqrt(weibel_cutoff_squared(3, mass, eta)) for eta in mixings])
        for mass in masses}
    return {"mixings": mixings, **curves}


def plot_weibel_cutoff(ax, scan):
    """Mark the seeded unstable/stable spatial modes against the marginal curve."""
    for key, values in scan.items():
        if key.startswith("cutoff_"):
            mass = key.removeprefix("cutoff_mu_")
            ax.plot(scan["mixings"], values, label=mass)
    ax.axhline(1, color="#30343B", ls=":", label="seeded $k_1$")
    ax.axhline(2, color="#6B7280", ls="--", label="control $k_2$")
    ax.axvline(0.3, color="#D55E00", ls=":")
    ax.set(xlabel=r"mixing $\eta$", ylabel=r"marginal $kc/\omega_p$",
           title="Transverse cutoff and dark mass")
    ax.grid(alpha=0.4)
    ax.legend(title=r"$\Omega_D/\omega_p$", facecolor="#FFFFFF",
              edgecolor="#6B7280", fontsize=8)


def warm_two_stream_reference(sigma_over_c, eta=0.6):
    """Selected symmetric-beam root with fixed physical density and drift."""
    k, wp = 2 * np.pi, 0.1 * 2 * np.pi * c
    populations = tuple({"wp": wp / np.sqrt(2), "u": sign * 0.05 * c,
                         "vth": np.sqrt(2) * sigma_over_c * c}
                        for sign in (1, -1))
    guess = 0.3j if sigma_over_c < 0.03 else -0.7j
    return {model: longitudinal_root(k, populations, k * c, eta, guess, model)[0]
            for model in ("ordinary", "full", "quasistatic", "effective_charge")}


def warm_two_stream_run(cells, particles, steps, sigma_over_c, eta=0.6):
    """Matched current-neutral warm beams, with a fixed physical seed and scales."""
    length, k, drift = 1.0, 2 * np.pi, 0.05 * c
    wp = 0.1 * k * c
    density = wp**2 * epsilon_0 * mass_electron / e**2
    vth = np.sqrt(2) * sigma_over_c * c
    seed_fraction = 0.01 if sigma_over_c > 0.03 else 0.001
    species = []
    for sign, name in ((1, "moonward"), (-1, "cryptward")):
        x, v = quiet_start(particles // 2, length, drift=(sign * drift, 0, 0),
                           vth=(vth, 0, 0))
        v = v.at[:, 0].add(sign * drift - jnp.mean(v[:, 0]))
        if sign == 1:
            x = x.at[:, 0].add(seed_fraction / k * jnp.sin(k * x[:, 0]))
        species.append(Species.electrons(particles // 2, density / 2, name=name).replace(x=x, v=v))
    plasma = Simulation(Domain(length, cells, dt_over_dx_c=0.5), tuple(species))
    parent = plasma.run(steps, store_every=max(1, steps // 200), store_particles=False)
    output = DarkSimulation(plasma, DarkField(k * c, eta)).run(
        steps, store_every=max(1, steps // 200), store_particles=False)
    time = np.asarray(output.ordinary.t) * wp
    parent_mode = np.fft.rfft(np.asarray(parent.E[:, :, 0]), axis=1)[:, 1] / cells
    mixed_mode = np.fft.rfft(np.asarray(output.ordinary.E[:, :, 0]), axis=1)[:, 1] / cells
    initial_energy = float(output.state.initial_ordinary + output.state.initial_dark)
    closed = np.asarray(output.energy()["total_with_dark"])
    return {"time": time, "parent": np.abs(parent_mode), "mixed": np.abs(mixed_mode),
            "parent_Ek": parent_mode, "mixed_Ek": mixed_mode,
            "energy_error": (closed - initial_energy) / initial_energy,
            "maximum_initial_speed_over_c": max(
                float(jnp.max(jnp.linalg.norm(item.v, axis=1)) / c) for item in species),
            "maximum_all_step_balance_over_initial": float(
                output.state.max_balance_error / initial_energy)}


def warm_growth_fit(history, branch, window):
    """Return a declared-window log-amplitude slope and regression diagnostics."""
    selected = (history["time"] > window[0]) & (history["time"] < window[1])
    fit = linregress(history["time"][selected], np.log(history[branch][selected]))
    return {"growth_over_wp": fit.slope, "stderr_over_wp": fit.stderr,
            "r_squared": fit.rvalue**2}


def warm_two_stream_main(args):
    """Compare a growing warm branch with a single-humped stable loading."""
    eta = 0.6  # manufactured coupling to separate PIC fits from loading noise
    cases = ((64, 60000, 4000), (128, 120000, 8000)) if args.full else ((32, 10000, 1000),)
    references = {name: warm_two_stream_reference(sigma, eta)
                  for name, sigma in (("unstable", 0.01), ("stable", 0.06))}
    histories, records = {}, {}
    for name, sigma in (("unstable", 0.01), ("stable", 0.06)):
        for cells, particles, steps in (cases if name == "unstable" else cases[:1]):
            history = warm_two_stream_run(cells, particles, steps, sigma, eta)
            key = f"{name}_{cells}"
            histories[key] = history
            record = {"cells": cells, "particles": particles, "steps": steps,
                      "sigma_over_c": sigma, "maximum_initial_speed_over_c":
                      history["maximum_initial_speed_over_c"],
                      "seed_displacement_over_1_k": 0.01 if name == "stable" else 0.001,
                      "maximum_all_step_balance_over_initial":
                      history["maximum_all_step_balance_over_initial"],
                      "maximum_sampled_closed_energy_drift": float(
                          np.max(np.abs(history["energy_error"])))}
            if name == "unstable" and args.full:
                record["fits"] = {branch: warm_growth_fit(history, branch, (6, 16))
                                  for branch in ("parent", "mixed")}
                record["fit_window_sensitivity"] = {
                    f"{lower}_{upper}": {
                        branch: warm_growth_fit(history, branch, (lower, upper))
                        for branch in ("parent", "mixed")}
                    for lower, upper in ((8, 16), (10, 18), (12, 19))}
            if name == "stable":
                early = (history["time"] >= 0) & (history["time"] < 2)
                late = (history["time"] > 12) & (history["time"] < 18)
                if np.any(late):
                    record["late_over_early_mode_rms"] = {
                        branch: float(np.sqrt(np.mean(history[branch][late]**2)
                                              / np.mean(history[branch][early]**2)))
                        for branch in ("parent", "mixed")}
            records[key] = record
    with midnight():
        fig, (ax, control) = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
        for key, history in histories.items():
            panel = control if key.startswith("stable") else ax
            for branch, style in (("parent", "-"), ("mixed", "--")):
                panel.semilogy(history["time"], history[branch], style,
                               label=f"{branch}, {records[key]['cells']} cells")
        for panel, title in ((ax, "Warm two-stream growth"),
                             (control, "Single-humped stable control")):
            panel.set(xlabel=r"$\omega_p t$", ylabel=r"$|E_{x,k}|$ (V/m)", title=title)
            panel.grid(alpha=0.4)
            panel.legend(facecolor="#FFFFFF", edgecolor="#6B7280", fontsize=8)
        settings = {"preset": "full" if args.full else "quick", "mode": "warm-two-stream",
                    "cases_cells_particles_steps": cases, "eta": eta, "drift_over_c": 0.05,
                    "mu_over_kc": 1.0, "linear_window_wp_t": [6, 16],
                    "stable_sigma_over_c": 0.06, "unstable_sigma_over_c": 0.01,
                    "background": "uniform fixed neutralizer"}
        results = {"references": {
            name: {model: [z.real, z.imag] for model, z in values.items()}
            for name, values in references.items()}, "cases": records}
        save_run(args.output, "warm_two_stream", settings, results, fig,
                 **{f"{key}_{field}": value for key, history in histories.items()
                    for field, value in history.items() if isinstance(value, np.ndarray)})
        plt.close(fig)
    print(f"🦇 WARM STREAMS: {len(records)} paired cases; "
          f"root {references['unstable']['full'].imag:.5f} ωp")


def main():
    parser = argparse.ArgumentParser(description="Check two honest plasma hauntings")
    parser.add_argument("mode", choices=("two-stream", "warm-two-stream", "weibel"))
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("artifacts/dark_instabilities"))
    args = parser.parse_args()
    if args.mode == "warm-two-stream":
        warm_two_stream_main(args)
    else:
        legacy_main(args)


def legacy_main(args):
    """Retain the cold-stream and transverse regressions."""
    if args.mode == "two-stream":
        reference = two_stream_growth(2 * np.pi / (0.05 * 64), 0.25 * 2 * np.pi / (0.05 * 64), 0.7, 0.3)
        uncoupled = two_stream_growth(2 * np.pi / (0.05 * 64), 0.25 * 2 * np.pi / (0.05 * 64), 0.7, 0)
        cases = ((64, 4000, 1000), (128, 8000, 2000)) if args.full else ((32, 500, 250),)
        runner, window = two_stream_run, (10, 20)
    else:
        reference = weibel_growth(1, 0.08, 4, 0.7, 0.3)
        uncoupled = weibel_growth(1, 0.08, 4, 0.7, 0)
        cases = ((64, 30000, 2400), (128, 60000, 4800)) if args.full else ((32, 2000, 400),)
        runner, window = weibel_run, (20, 70)

    histories, fits = [], []
    for cells, particles, steps in cases:
        t, amplitude, current, residual = runner(cells, particles, steps, 0.3)
        fit = None
        if args.full:
            selected = (t > window[0]) & (t < window[1])
            regression = linregress(t[selected], np.log(amplitude[selected]))
            fit = {"growth_over_wp": regression.slope, "stderr_over_wp": regression.stderr,
                   "r_squared": regression.rvalue**2,
                   "relative_error": abs(regression.slope / reference[0] - 1)}
        histories.append((t, amplitude, current, residual))
        fits.append(fit)
    zero_control = None
    if args.full:
        t0, amp0, _, _ = runner(*cases[0], 0.0)
        selected = (t0 > window[0]) & (t0 < window[1])
        zero_control = float(linregress(t0[selected], np.log(amp0[selected])).slope)

    screening, K, b = None, None, None
    if args.mode == "two-stream":
        K = 2 * np.pi / (0.05 * 64)
        b = 0.25 * K
        screening = screening_references(K, b, 0.7, 0.3)
    cutoff = weibel_cutoff_scan() if args.mode == "weibel" else None
    stable = None
    if cutoff is not None and args.full:
        stable = weibel_run(*cases[0], 0.3, mode=2)
    save_legacy_result(args, cases, histories, fits, reference, uncoupled,
                       zero_control, screening, cutoff, stable, window, K, b)


def save_legacy_result(args, cases, histories, fits, reference, uncoupled,
                       zero_control, screening, cutoff, stable, window, K, b):
    """Render the measured legacy branch with its applicable analytic control."""
    with midnight():
        if screening is None and cutoff is None:
            fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
        else:
            fig, (ax, screened_ax) = plt.subplots(1, 2, figsize=(11, 4.5),
                                                  layout="constrained")
        for index, ((cells, particles, _), (t, amplitude, _, _), fit) in enumerate(
                zip(cases, histories, fits)):
            ax.semilogy(t, amplitude, label=f"{cells} cells, {particles} particles")
            if fit is not None:
                anchor = np.argmin(abs(t - window[0]))
                curve = amplitude[anchor] * np.exp(reference[0] * (t - t[anchor]))
                linear = (t >= window[0]) & (t <= window[1])
                label = f"cold/kinetic oracle: γ={reference[0]:.4f}" if index == 0 else "_nolegend_"
                ax.semilogy(t[linear], curve[linear], "--", alpha=0.7, label=label)
        if stable is not None:
            ax.semilogy(stable[0], stable[1], ":", color="#30343B",
                        label="stable $k_2$, 64 cells")
        ylabel = r"$|E_{x,k}|$ (V/m)" if args.mode == "two-stream" else r"$|B_{y,k}|$ (T)"
        ax.set(xlabel=r"$\omega_p t$", ylabel=ylabel,
               title=f"{args.mode}: a current-neutral haunting")
        ax.grid(alpha=0.4)
        ax.legend(facecolor="#FFFFFF", edgecolor="#6B7280", fontsize=8)
        curves = {}
        if screening is not None:
            curves = plot_screened_growth(screened_ax, screening, K, b)
        if cutoff is not None:
            plot_weibel_cutoff(screened_ax, cutoff)
        settings = {"preset": "full" if args.full else "quick", "mode": args.mode,
                    "cases_cells_particles_steps": cases, "eta": 0.3, "mu_over_wp": 0.7,
                    "linear_window_wp_t": window, "background": "uniform fixed neutralizer",
                    "stable_control_mode": 2 if cutoff is not None else None}
        results = {"reference_growth_over_wp": reference[0], "reference_residual": reference[1],
                   "uncoupled_reference_growth_over_wp": uncoupled[0],
                   "pic_fits": fits, "zero_coupling_pic_growth_over_wp": zero_control,
                   "initial_net_current_A_m2": [item[2] for item in histories],
                   "max_dark_gauss_V_m2": [item[3] for item in histories],
                   "screened_cold_references": screening}
        if cutoff is not None:
            results["weibel_marginal"] = {
                "mixings": cutoff["mixings"].tolist(),
                "cutoff_kc_over_wp": {key: values.tolist() for key, values in cutoff.items()
                                      if key.startswith("cutoff_")}}
        if stable is not None:
            early = (stable[0] >= 0) & (stable[0] < 10)
            late = (stable[0] > 40) & (stable[0] < 70)
            results["stable_mode_2"] = {
                "cells": cases[0][0], "particles": cases[0][1],
                "late_over_early_magnetic_rms": float(np.sqrt(
                    np.mean(stable[1][late]**2) / np.mean(stable[1][early]**2))),
                "initial_net_current_A_m2": stable[2],
                "max_dark_gauss_V_m2": stable[3]}
        save_run(args.output, f"dark_{args.mode.replace('-', '_')}", settings, results, fig,
                 t=histories[0][0], amplitude=histories[0][1],
                 refined_t=histories[-1][0], refined_amplitude=histories[-1][1],
                 **curves, **(cutoff or {}),
                 **({"stable_t": stable[0], "stable_amplitude": stable[1]}
                    if stable is not None else {}))
        plt.close(fig)
    print(f"🦇 {args.mode.upper()}: root {reference[0]:.5f}; "
          f"PIC {fits[0]['growth_over_wp']:.5f}" if args.full
          else f"🦇 {args.mode.upper()} SMOKE: fit deferred to full preset")


if __name__ == "__main__":
    main()
