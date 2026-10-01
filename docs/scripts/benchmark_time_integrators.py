"""Independent 1D Proca time-step comparison on the parent's staggered grid.

The normalized vacuum equations use c = Omega_D = epsilon_0 = 1. This checks
field-only conservation and phase, not conservation of a coupled PIC scheme.
"""

from pathlib import Path
from time import perf_counter

import matplotlib
import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import csc_matrix, eye
from scipy.sparse.linalg import expm_multiply, splu

from darkjaxincell import midnight
from jaxincell import save_run

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


# Editable inputs also accept runpy.run_path(..., init_globals={...}) batch overrides.
quick = globals().get("quick", True)
cells = globals().get("cells", 16)
end = globals().get("end", 20. if quick else 200.)
output = Path(globals().get("output", "artifacts/time_integrators"))


def operators(cells):
    """Dimensionless Yee curl, divergence and negative-adjoint gradient."""
    dx = 2 * np.pi / cells

    def forward(a):
        return (np.roll(a, -1) - a) / dx

    def backward(a):
        return (a - np.roll(a, 1)) / dx

    def curl_b(b):
        return np.stack((np.zeros(cells), -forward(b[:, 2]), forward(b[:, 1])), axis=1)

    def curl_e(e):
        return np.stack((np.zeros(cells), -backward(e[:, 2]), backward(e[:, 1])), axis=1)

    def divergence(a):
        return backward(a[:, 0])

    def gradient(phi):
        return np.stack((forward(phi), np.zeros(cells), np.zeros(cells)), axis=1)

    return dx, curl_b, curl_e, divergence, gradient


def system(cells):
    """Return the linear generator and production-equivalent vacuum split."""
    dx, curl_b, curl_e, div, grad = operators(cells)
    width = 3 * cells
    size = 3 * width + cells

    def pack(e, b, a, phi):
        return np.concatenate((e.ravel(), b.ravel(), a.ravel(), phi))

    def unpack(y):
        return (y[:width].reshape(cells, 3), y[width:2 * width].reshape(cells, 3),
                y[2 * width:3 * width].reshape(cells, 3), y[3 * width:])

    def rhs(y):
        e, b, a, phi = unpack(y)
        return pack(curl_b(b) + a, -curl_e(e), -e - grad(phi), -div(a))

    def split(y, h):
        e, b, a, phi = (part.copy() for part in unpack(y))
        e += h / 2 * (curl_b(b) + a)
        phi -= h / 2 * div(a)
        b -= h * curl_e(e)
        a -= h * (e + grad(phi))
        e += h / 2 * (curl_b(b) + a)
        phi -= h / 2 * div(a)
        return pack(e, b, a, phi)

    x = np.arange(cells) * dx
    e = np.zeros((cells, 3))
    e[:, 0] = 0.2 * np.cos(x)
    e[:, 1] = np.cos(x) + 0.1 * np.cos(4 * x)
    e[:, 2] = 0.3 * np.sin(2 * x)
    y0 = pack(e, np.zeros_like(e), np.zeros_like(e), -div(e))
    columns = np.stack([rhs(basis) for basis in np.eye(size)], axis=1)
    generator = csc_matrix(columns)
    assert np.max(np.abs(columns + columns.T)) < 1e-13
    return y0, generator, split, unpack, div


def trial(method, h, end, y0, generator, split, unpack, div, exact):
    """Measure error against a matrix-exponential oracle and both invariants."""
    steps = round(end / h)
    times = np.linspace(0, end, steps + 1)
    start = perf_counter()
    if method == "split":
        history = [y0]
        for _ in range(steps):
            history.append(split(history[-1], h))
        nfev = 0
    elif method == "midpoint":
        identity = eye(generator.shape[0], format="csc")
        left = splu(identity - h / 2 * generator)
        right = identity + h / 2 * generator
        history = [y0]
        for _ in range(steps):
            history.append(left.solve(right @ history[-1]))
        nfev = 0
    else:
        solved = solve_ivp(lambda _, y: generator @ y, (0, end), y0, method="DOP853",
                           rtol=1e-9, atol=1e-11, t_eval=times)
        assert solved.success
        history, nfev = solved.y.T, solved.nfev
    seconds = perf_counter() - start
    history = np.asarray(history)
    energies = np.sum(history**2, axis=1)
    gauss = np.array([np.max(np.abs(div(unpack(y)[0]) + unpack(y)[3])) for y in history])
    return {"method": method, "h_omega": h, "steps": steps, "seconds": seconds,
            "rhs_evaluations": nfev,
            "max_relative_energy_change": float(np.max(np.abs(energies / energies[0] - 1))),
            "max_gauss": float(np.max(gauss)),
            "final_relative_state_error": float(np.linalg.norm(history[-1] - exact)
                                                / np.linalg.norm(exact)),
            "time": times, "energy_change": energies / energies[0] - 1}


if __name__ == "__main__":
    if cells < 2 or not np.isfinite(end) or end <= 0 or not np.isclose(end / .2, round(end / .2)):
        raise ValueError("at least two cells and a positive horizon divisible by 0.2 are required")
    print(f"Starting vacuum time-integrator benchmark: {cells} cells, Ωt={end:g}", flush=True)
    y0, generator, split, unpack, div = system(cells)
    exact = expm_multiply(end * generator, y0, traceA=0)
    runs = []
    for method, h in (("split", .2), ("split", .1), ("midpoint", .2), ("midpoint", .1), ("DOP853", .2)):
        print(f"Measuring {method}, ΔtΩ={h:g}", flush=True)
        runs.append(trial(method, h, end, y0, generator, split, unpack, div, exact))
    with midnight():
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), layout="constrained")
        for run, color in zip(runs, ("#6A3D9A", "#B03568", "#0072B2", "#009E73", "#D55E00")):
            label = f"{run['method']}, ΔtΩ={run['h_omega']}"
            drift = np.maximum.accumulate(np.abs(run["energy_change"]))
            axes[0].semilogy(run["time"], np.maximum(drift, 1e-15), label=label, color=color)
        axes[0].set(xlabel=r"$\Omega_D t$", ylabel=r"largest $|\Delta U/U_0|$ so far",
                    ylim=(1e-15, 1e-1), title="Vacuum Proca energy")
        axes[0].legend(facecolor="#FFFFFF", edgecolor="#6B7280", fontsize=7, loc="center right")
        for run, color in zip(runs, ("#6A3D9A", "#B03568", "#0072B2", "#009E73", "#D55E00")):
            axes[1].scatter(run["seconds"], run["final_relative_state_error"], color=color,
                            label=f"{run['method']}, ΔtΩ={run['h_omega']}")
        axes[1].set(xlabel="wall time including setup (s)", ylabel="final state error vs exp(tL)",
                    xlim=(0, max(run["seconds"] for run in runs) * 1.1),
                    yscale="log", title="Phase and amplitude accuracy")
        axes[1].legend(loc="center", facecolor="#FFFFFF", edgecolor="#6B7280", fontsize=7)
        for ax in axes:
            ax.grid(alpha=0.25)
        settings = {"preset": "quick" if quick else "full", "cells": cells,
                    "horizon_omega_D": end, "normalized_units": "c=Omega_D=epsilon_0=1",
                    "scope": "source-free staggered Proca field only; no PIC particles"}
        results = {"methods": [{k: v for k, v in run.items() if k not in ("time", "energy_change")}
                               for run in runs], "oracle": "scipy.sparse.linalg.expm_multiply"}
        arrays = {f"case_{i}_{key}": run[key] for i, run in enumerate(runs)
                  for key in ("time", "energy_change")}
        save_run(output, "time_integrators", settings, results, fig, **arrays)
        plt.close(fig)
    for run in results["methods"]:
        print("🕰️", run)

    print(f"Finished vacuum time-integrator benchmark; wrote {output}", flush=True)
