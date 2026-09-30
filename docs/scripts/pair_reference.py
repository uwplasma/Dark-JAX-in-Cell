"""Independent linear waterbag reference for an oscillating ordinary pair plasma.

Time is in 1/omega_0, velocity in c, and each species has omega_p^2=1/2.
The four unknowns are perturbations of the two waterbag edges per species;
Gauss's law determines the nonzero-k electric field at every stage.
"""

import numpy as np
from scipy.integrate import solve_ivp


def edge_matrix(t, k, half_width, quiver):
    """Return the co-accelerating Vlasov operator for one Fourier mode."""
    charge = np.array([-1.0, -1.0, 1.0, 1.0])
    edge = np.array([-half_width, half_width] * 2)
    drift = charge * quiver * np.sin(t)
    # f0'=n0[delta(w+vT)-delta(w-vT)]/(2vT), with n0=1/2.
    source = -charge * np.array([1.0, -1.0] * 2) / (4 * half_width)
    return np.diag(-1j * k * (edge + drift)) + np.outer(source, charge) / (1j * k)


def monodromy(k, half_width=0.05, quiver=0.2 / np.sqrt(2), rtol=2e-10):
    """Propagate one ordinary plasma period; return physical edge amplitudes."""
    def rhs(t, columns):
        return (edge_matrix(t, k, half_width, quiver)
                @ columns.reshape(4, 4)).ravel()

    solution = solve_ivp(rhs, (0, 2 * np.pi), np.eye(4, dtype=complex).ravel(),
                         rtol=rtol, atol=rtol * 0.01)
    if not solution.success:
        raise RuntimeError(solution.message)
    return solution.y[:, -1].reshape(4, 4)


def growth(k, half_width=0.05, quiver=0.2 / np.sqrt(2)):
    """Largest physical Floquet growth in omega_0 units, with its multiplier."""
    values = np.linalg.eigvals(monodromy(k, half_width, quiver))
    largest = values[np.argmax(abs(values))]
    return np.log(abs(largest)) / (2 * np.pi), largest


def seeded_response(times, k, seed, half_width=0.05,
                    quiver=0.2 / np.sqrt(2)):
    """Electric Fourier coefficient from opposite cosine velocity nudges."""
    charge = np.array([-1.0, 1.0])
    edges = np.stack((-charge, charge), axis=1).ravel() * seed / (8 * half_width)
    times = np.asarray(times)
    solution = solve_ivp(lambda t, h: edge_matrix(t, k, half_width, quiver) @ h,
                         (0, float(times[-1])), edges.astype(complex), t_eval=times,
                         rtol=2e-10, atol=2e-12)
    if not solution.success:
        raise RuntimeError(solution.message)
    return charge.repeat(2) @ solution.y / (1j * k)
