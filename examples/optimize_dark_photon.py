"""Find the cold ghost's best density with a known independent answer.

The reduced objective calls the production PIC transition; it stores a scalar,
not a particle history. The default is a short smoke preset; ``full=True``
uses the fixed [45,75] reference window with 16 cells and 64 particles.
Edit the inputs below or pass them through ``runpy.run_path``.
"""

import math
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
from jax import lax
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import expm, expm_frechet
from scipy.optimize import minimize, minimize_scalar

from jaxincell import (Domain, Simulation, Species, elementary_charge as e,
                       epsilon_0, mass_electron, speed_of_light as c, save_run)
from darkjaxincell import DarkField, DarkSimulation, midnight


# Inputs: full=True selects the fixed physical calibration window.
full = globals().get('full', False)
output = Path(globals().get('output', 'artifacts/optimize_dark_photon'))
cells = globals().get('cells', 16 if full else 8)
particles = globals().get('particles', 64 if full else 32)
start = globals().get('start', 45.0 if full else 5.0)
stop = globals().get('stop', 75.0 if full else 12.0)
recurrence = globals().get('recurrence', 'scan')
checkpoint_size = globals().get('checkpoint_size', 32)
courant = globals().get('courant', 0.2)
refined_courant = globals().get('refined_courant', courant / 2)


def cold_reference(p, start, stop, eta=0.05, nodes=128):
    """Independent continuous cold objective and Fréchet derivative."""
    points, weights = np.polynomial.legendre.leggauss(nodes)
    times = start + (points + 1) * (stop - start) / 2
    G = np.array([[0, 0, 0, -1], [0, 0, 1, -eta],
                  [0, -1, 0, 0], [p, eta * p, 0, 0]], dtype=float)
    dG = np.zeros((4, 4))
    dG[3, :2] = [1, eta]
    y0 = np.array([0, 1, 0, 0], dtype=float)
    integrand, derivative = [], []
    for t in times:
        y = expm(G * t) @ y0
        dy = expm_frechet(G * t, dG * t, compute_expm=False) @ y0
        integrand.append(y[0] ** 2 + y[3] ** 2 / p)
        derivative.append(2 * y[0] * dy[0] + 2 * y[3] * dy[3] / p - y[3] ** 2 / p**2)
    return np.dot(weights, integrand) / 2, np.dot(weights, derivative) / 2


def build_objective(cells, particles, start, stop, recurrence="scan", checkpoint_size=32, courant=0.2):
    """A fixed-horizon scalar sensor with gradients through particle loading."""
    omega0, eta, amplitude = 1e9, 0.05, 1e-5
    n_ref = epsilon_0 * mass_electron * omega0**2 / e**2
    domain = Domain(length=2 * np.pi * c / omega0, cells=cells, dt_over_dx_c=courant)
    wave = jnp.broadcast_to(jnp.array([0.0, amplitude, 0.0]), (cells, 3))
    steps = math.ceil(stop / (omega0 * domain.dt) - 1e-12)
    dt_bar = domain.dt * omega0

    def objective(p):
        plasma = Simulation(domain, (Species.electrons(particles, density=p * n_ref),))
        sim = DarkSimulation(plasma, DarkField(omega0, eta, initial_E=wave))
        state, extra = sim.initial_state(jax.random.PRNGKey(0))

        def observable(ordinary):
            electric = jnp.mean(ordinary.E[:, 1]) / amplitude
            velocity = plasma._velocity(ordinary.u)
            current = (jnp.sum(extra[1] * ordinary.w * velocity[:, 1])
                       / (domain.length * epsilon_0 * omega0 * amplitude))
            return electric**2 + current**2 / p

        def advance(carry, _):
            state, integral, before = carry
            time_before = state.ordinary.time * omega0
            state, _ = sim._step(state, extra)
            after = observable(state.ordinary)
            left = jnp.maximum(time_before, start)
            right = jnp.minimum(time_before + dt_bar, stop)
            width = jnp.maximum(right - left, 0.0)
            first = before + (after - before) * (left - time_before) / dt_bar
            last = before + (after - before) * (right - time_before) / dt_bar
            integral += width * (first + last) / (2 * (stop - start))
            return (state, integral, after), None

        initial = (state, jnp.zeros(()), observable(state.ordinary))
        if recurrence == "scan":
            (_, integral, _), _ = lax.scan(advance, initial, None, length=steps)
        elif recurrence == "remat":
            (_, integral, _), _ = lax.scan(jax.checkpoint(advance), initial, None, length=steps)
        elif recurrence == "solvax":
            from solvax.autodiff import checkpointed_fori_loop

            _, integral, _ = checkpointed_fori_loop(
                0, steps, lambda index, carry: advance(carry, None)[0],
                initial, checkpoint_size=checkpoint_size)
        elif recurrence == "segmented":
            def body(index, carry):
                return advance(carry, None)[0]

            @jax.checkpoint
            def segment(index, carry):
                return lax.fori_loop(0, checkpoint_size,
                                     lambda local, value: body(index * checkpoint_size + local, value), carry)

            carry = lax.fori_loop(0, steps // checkpoint_size, segment, initial)
            _, integral, _ = jax.checkpoint(lambda value: lax.fori_loop(
                steps // checkpoint_size * checkpoint_size, steps, body, value))(carry)
        else:
            raise ValueError(f"unknown recurrence {recurrence!r}")
        return integral

    return jax.jit(objective), jax.jit(jax.value_and_grad(objective)), steps


if __name__ == "__main__":
    if not np.isfinite([start, stop]).all() or start < 0 or stop <= start:
        raise ValueError('the objective window must be finite with 0 <= start < stop')
    quick = not full
    objective, value_grad, steps = build_objective(
        cells, particles, start, stop, recurrence, checkpoint_size, courant)
    print(f"🦇 Compiling {steps} steps and their density gradient", flush=True)
    compile_start = perf_counter()
    objective(1.0).block_until_ready()
    primal_compile = perf_counter() - compile_start
    compile_start = perf_counter()
    value_grad(1.0)[0].block_until_ready()
    gradient_compile = perf_counter() - compile_start
    samples = np.linspace(0.5, 1.5, 101)
    print("🦇 Scanning the density objective and independent cold reference", flush=True)
    start_time = perf_counter()
    values = np.array([float(objective(p)) for p in samples])
    scan_time = perf_counter() - start_time
    reference = np.array([cold_reference(p, start, stop)[0] for p in samples])
    centre = samples[int(np.argmax(reference))]
    reference_opt = minimize_scalar(lambda p: -cold_reference(p, start, stop)[0],
                                    bounds=(max(0.5, centre - 0.05), min(1.5, centre + 0.05)), method="bounded",
                                    options={"xatol": 1e-11})

    trials = []
    for initial in (0.75, 1.0, 1.25):
        print(f"🦇 Gradient optimization from density {initial:g}", flush=True)
        trace = []

        def loss(x):
            value, gradient = value_grad(x[0])
            trace.append(float(x[0]))
            return -float(value), np.array([-float(gradient)])

        result = minimize(loss, np.array([initial]), jac=True, bounds=[(0.5, 1.5)],
                          method="L-BFGS-B", options={"ftol": 1e-12, "gtol": 1e-7})
        trials.append({"start": initial, "p": float(result.x[0]), "objective": -float(result.fun),
                       "calls": int(result.nfev), "success": bool(result.success), "trace": trace})
    best = max(trials, key=lambda trial: trial["objective"])
    p_test = 0.97
    ad_gradient = float(value_grad(p_test)[1])
    h = np.logspace(-2, -6, 9)
    finite = np.array([(float(objective(p_test + step)) - float(objective(p_test - step))) / (2 * step)
                       for step in h])
    cold_value, cold_gradient = cold_reference(p_test, start, stop)
    refined = None
    if not quick:
        print("🦇 Checking the finer timestep", flush=True)
        refined_objective, refined_value_grad, _ = build_objective(
            cells, particles, start, stop, recurrence, checkpoint_size, refined_courant)
        refined_opt = minimize_scalar(lambda p: -float(refined_objective(p)),
                                      bounds=(0.98, 1.02), method="bounded",
                                      options={"xatol": 1e-10})
        refined = {"p": float(refined_opt.x), "objective": -float(refined_opt.fun),
                   "gradient_at_0p97": float(refined_value_grad(p_test)[1])}

    with midnight():
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
        axes[0].plot(samples, reference, label="continuous cold oracle")
        axes[0].plot(samples, values, "--", label="PIC scalar objective")
        axes[0].scatter([trial["p"] for trial in trials], [trial["objective"] for trial in trials],
                        marker="x", s=65, label="gradient starts")
        axes[0].set(xlabel=r"$p=n/n_{\rm ref}$", ylabel="coherent transfer objective",
                    title="The density that wakes the ghost")
        axes[0].legend(facecolor="#FFFFFF", edgecolor="#6B7280", fontsize=8)
        axes[1].loglog(h, np.abs(finite - ad_gradient), "o-", label="PIC FD − AD")
        axes[1].axhline(abs(ad_gradient - cold_gradient), ls="--", label="PIC AD − cold Fréchet")
        axes[1].set(xlabel="finite-difference step in p", ylabel="absolute gradient error",
                    title="Gradient under the moonlight")
        axes[1].legend(facecolor="#FFFFFF", edgecolor="#6B7280", fontsize=8)
        for ax in axes:
            ax.grid(alpha=0.4)
        settings = {"preset": "quick" if quick else "full", "window": [start, stop],
                    "steps": steps, "cells": cells, "particles": particles,
                    "recurrence": recurrence, "checkpoint_size": checkpoint_size, "courant": courant,
                    "refined_courant": refined_courant if full else None,
                    "eta": 0.05, "omega0_rad_s": 1e9, "dark_E0_V_m": 1e-5}
        results = {"pic_best_p": best["p"], "pic_best_objective": best["objective"],
                   "pic_scan_best_p": float(samples[int(np.argmax(values))]),
                   "refined_pic": refined,
                   "cold_best_p": float(reference_opt.x), "cold_best_objective": -float(reference_opt.fun),
                   "gradient_pic_at_0p97": ad_gradient, "gradient_cold_at_0p97": cold_gradient,
                   "cold_objective_at_0p97": cold_value, "gradient_fd_min_error": float(min(abs(finite - ad_gradient))),
                   "primal_compile_s": primal_compile, "gradient_compile_s": gradient_compile,
                   "scan_101_s": scan_time, "trials": trials}
        save_run(output, "optimize_dark_photon", settings, results, fig,
                 p=samples, pic_objective=values, cold_objective=reference,
                 fd_h=h, fd_gradient=finite)
        plt.close(fig)
    print(f"🦇 {'SMOKE' if quick else 'CALIBRATION'}: PIC p={best['p']:.7f}, "
          f"cold p={reference_opt.x:.7f}; PIC C={best['objective']:.6f}, "
          f"cold C={-reference_opt.fun:.6f}")
