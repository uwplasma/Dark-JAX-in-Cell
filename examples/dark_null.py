"""A homogeneous ghost cannot excite a stable electron-only electrostatic mode.

This restricted Vlasov–Poisson null test compares the seeded nonzero-k mode
with and without a prescribed spatially uniform drive. The continuum drive
can be removed by an accelerating-frame change of coordinates. A finite-grid
difference should shrink on refinement; no instability is claimed here.
"""

from pathlib import Path

import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from jaxincell import (Domain, Simulation, Species, elementary_charge as e,
                       epsilon_0, mass_electron, quiet_start, save_run,
                       speed_of_light as c)
from darkjaxincell import DarkSimulation, PrescribedDrive, midnight


# Each row is (cells, particles, steps) at the same physical horizon.
full = globals().get("full", False)
output = Path(globals().get("output", "artifacts/dark_null"))
pairs = globals().get(
    "pairs", ((64, 4000, 300), (128, 16000, 600)) if full else ((32, 1000, 150),))


def mode_history(cells, particles, steps, drive):
    """Return the seeded electric-mode magnitude with a common quiet loading."""
    length, k = 1.0, 2 * np.pi
    wp = 0.05 * c * 64 / length
    density = wp**2 * epsilon_0 * mass_electron / e**2
    vth = np.sqrt(2) * 0.5 * wp / k
    x, v = quiet_start(particles, length, vth=(vth, 0.0, 0.0))
    v = v.at[:, 0].add(-jnp.mean(v[:, 0]))
    x = x.at[:, 0].add(0.01 / k * jnp.sin(k * x[:, 0]))
    electrons = Species.electrons(particles, density=density, vth=(vth, 0.0, 0.0)).replace(x=x, v=v)
    plasma = Simulation(Domain(length=length, cells=cells, dt_over_dx_c=0.5), (electrons,))
    effective_amplitude = 0.03 * vth * mass_electron * wp / e
    model = PrescribedDrive(eta=0.1 if drive else 0.0,
                            amplitude=jnp.array([effective_amplitude / 0.1, 0.0, 0.0]),
                            omega=wp)
    result = DarkSimulation(plasma, model).run(steps, store_every=max(1, steps // 100),
                                               store_particles=False, verbose=True)
    t = np.asarray(result.ordinary.t) * wp
    mode = np.fft.rfft(np.asarray(result.ordinary.E[:, :, 0]), axis=1)[:, 1] / cells
    return t, np.abs(mode), float(effective_amplitude)


if __name__ == "__main__":
    histories, errors = [], []
    for cells, particles, steps in pairs:
        t, zero, amplitude = mode_history(cells, particles, steps, False)
        _, driven, _ = mode_history(cells, particles, steps, True)
        error = float(np.max(np.abs(driven - zero)) / zero[0])
        histories.append((t, zero, driven))
        errors.append(error)

    with midnight():
        fig, axes = plt.subplots(1, len(pairs), figsize=(5 * len(pairs), 4),
                                 squeeze=False, layout="constrained")
        for ax, (cells, particles, _), (t, zero, driven), error in zip(
                axes[0], pairs, histories, errors):
            ax.plot(t, zero / zero[0], label="zero drive")
            ax.plot(t, driven / zero[0], "--", label="homogeneous drive")
            ax.set(xlabel=r"$\omega_p t$", ylabel=r"$|E_{x,k}|/|E_{x,k}(0)|$",
                   title=f"{cells} cells, {particles} particles: Δ={error:.2e}")
            ax.grid(alpha=0.4)
            ax.legend(facecolor="#FFFFFF", edgecolor="#6B7280")
        settings = {"preset": "full" if full else "quick", "refinements": pairs,
                    "omega_p_rad_s": 0.05 * c * 64, "k_lambda_D": 0.5,
                    "seed_displacement_over_1_k": 0.01,
                    "effective_drive_E_V_m": amplitude,
                    "drive_quiver_over_thermal": 0.03,
                    "background": "uniform fixed neutralizer"}
        results = {"max_mode_amplitude_difference_over_initial": errors,
                   "refinement_ratio": errors[0] / errors[-1] if len(errors) > 1 else None,
                   "claim": "restricted accelerating-frame null, not nonlinear reproduction"}
        save_run(output, "dark_null", settings, results, fig,
                 t=histories[0][0], zero=histories[0][1], driven=histories[0][2],
                 fine_zero=histories[-1][1], fine_driven=histories[-1][2])
        plt.close(fig)
    print(f"🦇 FALSE HAUNTING: mode differences {[f'{value:.3e}' for value in errors]}")
