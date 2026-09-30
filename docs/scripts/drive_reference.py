"""Independent homogeneous electron-ion response in fixed wp, c, me units.

Gaussian initial velocities are integrated with Gauss-Hermite quadrature.
The relativistic response keeps the whole shifted momentum distribution,
so coherent detuning can be tested without any grid, shot noise or density wave.
"""

import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.integrate import solve_ivp


def forced_cold(times, amplitude, mass_ratio=1836.):
    """Stable exact response including ion inertia, with the fixed-ion limit."""
    t = np.asarray(times)
    frequency = np.sqrt(1 + 1 / mass_ratio)
    difference = 1 / (mass_ratio * (frequency + 1))
    return -(1 + 1 / mass_ratio) * amplitude * t * np.sin((frequency + 1) * t / 2) * np.sinc(
        difference * t / (2 * np.pi)) / (frequency + 1)


def homogeneous(times, amplitude, *, mass_ratio=1836., temperature=1e-3,
                relativistic=True, nodes=64, eta=None, omega=1., rtol=2e-10):
    """Unseeded prescribed drive or finite Proca reservoir without spatial dynamics.

    ``amplitude`` is initial effective-force quiver / c. For a finite reservoir,
    D(0)=amplitude/eta; for a prescribed drive, effective external E=A cos(t).
    Energies use n me c², momentum uses n me c, time uses electron wp inverse.
    Velocity variance is a k=0-subtracted spread, not a thermodynamic temperature.
    """
    t = np.asarray(times, dtype=float)
    if t.ndim != 1 or len(t) < 2 or t[0] < 0 or np.any(np.diff(t) <= 0):
        raise ValueError('times must be increasing and nonnegative')
    if mass_ratio <= 0 or temperature < 0 or omega <= 0 or (eta is not None and eta == 0):
        raise ValueError('invalid plasma or finite-reservoir parameters')
    points, weights = hermgauss(nodes)
    weights /= np.sqrt(np.pi)
    masses = np.array([1., mass_ratio])
    velocity = np.sqrt(2 * temperature / masses[:, None]) * points
    if relativistic and np.max(abs(velocity)) >= 1:
        raise ValueError('quadrature velocities exceed c; use a physical relativistic distribution')
    initial = velocity / np.sqrt(1 - velocity**2) if relativistic else velocity

    def moments(impulse):
        u = initial + np.array([-1., 1 / mass_ratio])[:, None] * impulse
        gamma = np.sqrt(1 + u**2) if relativistic else np.ones_like(u)
        v = u / gamma
        mean = v @ weights
        variance = ((v - mean[:, None])**2) @ weights
        kinetic = masses * ((u**2 / (gamma + 1)) @ weights)
        return mean, np.sqrt(variance), kinetic

    def rhs(time, state):
        E, D, A, impulse, _ = state
        mean, _, _ = moments(impulse)
        current = mean[1] - mean[0]
        if eta is None:
            drive = amplitude * np.cos(time)
            return [-current, 0., 0., E + drive, current * drive]
        return [-current, omega**2 * A - eta * current, -D, E + eta * D, 0.]

    start = [0., 0. if eta is None else amplitude / eta, 0., 0., 0.]
    solution = solve_ivp(rhs, (0., float(t[-1])), start, t_eval=t,
                         method='DOP853', rtol=rtol, atol=rtol * 0.01)
    if not solution.success:
        raise RuntimeError(solution.message)
    E, D, A, impulse, work = solution.y
    mean, rms, kinetic = (np.stack(values) for values in zip(*(moments(value) for value in impulse)))
    electric = E**2 / 2
    dark = (D**2 + omega**2 * A**2) / 2
    return dict(t=t, mean_E=E, mean_D=D, mean_A=A, mean=mean, rms=rms,
                kinetic=kinetic, spread=0.5 * masses[None, :] * rms**2,
                electric=electric, dark=dark, work=work,
                balance=electric + dark + kinetic.sum(axis=1) - work,
                nfev=solution.nfev, nodes=nodes)
