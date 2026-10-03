"""Vacuum Proca and cold resonant integrator checks in normalized units.

The vacuum uses the parent's staggered grid; the collective model adds its
physical homogeneous current. Neither is a nonlinear PIC qualification.
"""

from pathlib import Path
from decimal import Decimal, localcontext
import hashlib
import json
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
study = globals().get("study", "vacuum")
eta = globals().get("eta", .005)
h_values = globals().get("h_values", (.1, .05, .025, .0125, .00625, .003125))
archive = globals().get("archive", None)
decimal_reference = globals().get("decimal_reference", None)
cells = globals().get("cells", 16)
end = globals().get("end", (100. if quick else 1000.) if study == "resonant" else (20. if quick else 200.))
output = Path(globals().get(
    "output", "artifacts/resonant_integrators" if study == "resonant" else "artifacts/time_integrators"))


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


def resonant_modes(state, eta):
    """Closed electric mode vectors, with frequencies sqrt(1+eta²/4) ± eta/2."""
    carrier = np.hypot(1., eta / 2)
    plus = carrier + eta / 2
    vectors = np.array([[plus, 1.], [-1., plus]]) / np.hypot(1., plus)
    frequency = np.array([carrier - eta / 2, plus])
    E, D, A, current = state
    return vectors.T @ np.array([E, D]) + 1j * (vectors.T @ np.array([-current, A - eta * current])) / frequency


def resonant_exact(eta, tau):
    """Stable carrier/beat solution from (E,D,A,j)=(0,1,0,0), wp=Omega_D=1."""
    carrier, beat = np.hypot(1., eta / 2), eta / 2
    s, c = np.sin(carrier * tau), np.cos(carrier * tau)
    sb, cb = np.sin(beat * tau), np.cos(beat * tau)
    return np.array([-s * sb / carrier, c * cb - beat * s * sb / carrier,
                     -s * cb / carrier, c * sb + beat * s * cb / carrier])


def resonant_map(eta, h, method):
    """Cold coupled CN or solved affine exact-dark/ordinary-CN endpoint-current closure."""
    identity = np.eye(4)
    if method == "cn":
        M = np.array([[0., 0., 0., -1.], [0., 0., 1., -eta],
                      [0., -1., 0., 0.], [1., eta, 0., 0.]])
        return np.linalg.solve(identity - h * M / 2, identity + h * M / 2)
    S, C2 = np.sinc(h / np.pi), .5 * np.sinc(h / (2 * np.pi))**2
    free = identity[0] + eta * S * identity[1] + eta * h * C2 * identity[2]
    current = (identity[3] + h * free / 2) / (1 + h * h / 4 + eta * eta * h * h * C2 / 2)
    return np.array([identity[0] - h * current,
                     np.cos(h) * identity[1] + h * S * identity[2] - eta * h * S * current,
                     np.cos(h) * identity[2] - h * S * identity[1] + eta * h * h * C2 * current,
                     2 * current - identity[3]])


def resonant_archive(folder):
    """Verify saved endpoint leaves before reusing them without new powers."""
    if folder is None:
        return {}, {}
    folder = Path(folder)
    source = json.loads((folder / "run.json").read_text())
    with np.load(folder / "raw.npz", allow_pickle=False) as saved:
        for row in source["rows"]:
            if "tau" in row:
                key = f'endpoint_{row["eta"]}_{row["h"]}_{row["method"]}_{row["tau"]}'
                if not np.array_equal(saved[key], row["state"]):
                    raise ValueError("archived raw endpoint differs from its native record")
    provenance = {name: hashlib.sha256((folder / name).read_bytes()).hexdigest() for name in ("run.json", "raw.npz")}
    return source, provenance


def resonant_controls(path, source):
    """Check the closed formula at the four frozen high-precision targets."""
    if path is None:
        return [], {}
    path = Path(path)
    decimal = json.loads(path.read_text())
    if source and decimal["historical_report_unchanged"] != source:
        raise ValueError("Decimal reference is not bound to the supplied native archive")
    controls = []
    for target in decimal["references"]:
        if target["precision"] == 100:
            actual = resonant_exact(target["eta"], target["tau"])
            with localcontext() as context:
                context.prec = 50
                error = sum((Decimal.from_float(float(a)) - Decimal(b))**2
                            for a, b in zip(actual, target["state"])).sqrt()
            controls.append(dict(eta=target["eta"], tau=target["tau"], error=float(error),
                                 passed=bool(error <= Decimal.from_float(2e-13))))
    targets = {(.01, 100), (.01, 1000), (.005, 100), (.005, 1000)}
    if {(r["eta"], r["tau"]) for r in controls} != targets or not all(r["passed"] for r in controls):
        raise ValueError("stable formula failed the four frozen Decimal targets")
    return controls, {"decimal_reference_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def resonant_endpoint(eta, end, h, method, initial, supplied):
    """Select one verified archived endpoint or compute its literal matrix power."""
    if supplied is None:
        start = perf_counter()
        state = np.linalg.matrix_power(resonant_map(eta, h, method), round(end / h)) @ initial
        row = dict(eta=eta, tau=end, h=h, method=method, state=state.tolist(),
                   energy_drift=float((state @ state - 1) / 2), endpoint_power_seconds=perf_counter() - start)
    else:
        matches = [r for r in supplied if (r["eta"], r.get("tau"), r["h"], r["method"]) == (eta, end, h, method)]
        if len(matches) != 1:
            raise ValueError("exactly one native endpoint is required for each method/step")
        row, state = matches[0], np.asarray(matches[0]["state"])
    if state.shape != (4,) or not np.all(np.isfinite(state)):
        raise ValueError("four finite physical endpoint coordinates are required")
    return row, state


def resonant_plot(results, eta, end):
    """Plot endpoint energy, unaligned phase and transfer errors on white axes."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.7), layout="constrained")
    for method, color, label in (("cn", "#0072B2", "coupled CN"), ("hybrid", "#6A3D9A", "exact-dark / ordinary-CN")):
        selected = [r for r in results if r["method"] == method]
        x = [r["h"] for r in selected]
        for ax, key in ((axes[0], "energy_error"), (axes[2], "transfer_error")):
            ax.loglog(x, np.maximum([r[key] for r in selected], np.finfo(float).tiny), "o-", color=color, label=label)
        for mode, style in enumerate(("o-", "s--")):
            axes[1].loglog(x, [r["unaligned_phase_error"][mode] for r in selected], style,
                           color=color, label=label + (" lower mode" if mode == 0 else " upper mode"))
    axes[0].axhline(2e-13, color="#6B7280", linestyle=":", label="original energy bound")
    for ax, title, ylabel in zip(axes, ("Endpoint energy", "Unaligned physical-mode phase", "Ordinary energy transfer"),
                                 (r"$|H-H_0|$, $H_0=1/2$", "absolute phase error (rad)",
                                  r"$|f-f_{exact}|$, $f=E^2+j^2$")):
        ax.set(xlabel=r"$h=\omega_p\Delta t$", ylabel=ylabel, title=title)
        ax.grid(alpha=.25)
        ax.legend(fontsize=7, facecolor="white", edgecolor="#6B7280")
    fig.suptitle(f"Cold collective resonance: eta={eta:g}, tau={end:g}; endpoint errors")
    return fig


if __name__ == "__main__" and study == "resonant":
    print(f"🕯️ Cold resonant integrators: eta={eta:g}, tau={end:g}", flush=True)
    if not np.isfinite(eta) or eta < 0 or not np.isfinite(end) or end <= 0:
        raise ValueError("finite nonnegative eta and positive end are required")
    if not h_values or any(not np.isfinite(h) or h <= 0 or round(end / h) * h != end for h in h_values):
        raise ValueError("each positive step must divide the horizon exactly")
    source, provenance = resonant_archive(archive)
    supplied = source.get("rows")
    controls, decimal_provenance = resonant_controls(decimal_reference, source)
    provenance.update(decimal_provenance)
    initial = np.array([0., 1., 0., 0.])
    exact = resonant_exact(eta, end)
    reference_modes = resonant_modes(exact, eta)
    results, arrays = [], {}
    for method in ("cn", "hybrid"):
        for h in h_values:
            print(f"Cold collective {method}: eta={eta:g}, tau={end:g}, h={h:g}, "
                  + ("archived endpoint" if supplied is not None else "literal endpoint power"), flush=True)
            row, state = resonant_endpoint(eta, end, h, method, initial, supplied)
            phase = np.abs(np.angle(resonant_modes(state, eta) * np.conj(reference_modes)))
            transfer = state[0]**2 + state[3]**2
            energy_error = abs(float((state @ state - 1) / 2))
            if abs(energy_error - abs(row["energy_drift"])) > 2e-15:
                raise ValueError("native energy defect disagrees with the archived endpoint")
            result = dict(method=method, h=h, energy_error=energy_error,
                          unaligned_phase_error=phase.tolist(), transfer_fraction=float(transfer),
                          transfer_error=float(abs(transfer - exact[0]**2 - exact[3]**2)), historical_row=row)
            results.append(result)
            arrays[f'{method}_{h}_endpoint'] = state
    with midnight():
        fig = resonant_plot(results, eta, end)
        settings = dict(preset="quick" if quick else "full", study="resonant", eta=eta, horizon_omega_p=end,
                        steps=list(h_values), state_order=["E", "D", "A", "j"], initial=initial.tolist(),
                        normalized_units="wp=Omega_D=epsilon_0=1",
                        scope="cold linear collective4x4; no PIC or nonlinear validation")
        record = dict(methods=results, oracle="closed carrier/beat formula and analytic electric mode vectors",
                      producer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      decimal_controls=controls, archive_provenance=provenance,
                      archived_endpoints_used=supplied is not None,
                      new_discrete_endpoints=0 if supplied is not None else len(results),
                      historical_gates=source.get("gates"),
                      historical_eta0_accuracy=source.get("physical_eta0_accuracy_gates"),
                      historical_scientific_acceptance=source.get("scientific_acceptance"), scientific_acceptance=False,
                      no_phase_alignment=True, no_energy_renormalization=True, no_PIC_adoption=True)
        save_run(output, "time_integrators_resonant", settings, record, fig, exact_endpoint=exact, **arrays)
        plt.close(fig)


if __name__ == "__main__" and study not in ("vacuum", "resonant"):
    raise ValueError("study must be vacuum or resonant")

if __name__ == "__main__" and study == "vacuum":
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
