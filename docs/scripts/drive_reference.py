"""Independent homogeneous electron-ion response in fixed wp, c, me units.

Gaussian initial velocities are integrated with Gauss-Hermite quadrature.
The relativistic response keeps the whole shifted momentum distribution,
so coherent detuning can be tested without any grid, shot noise or density wave.
"""

import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.integrate import solve_ivp
from scipy.interpolate import BSpline


_QUARTIC = BSpline.basis_element(np.arange(6) - 2.5, extrapolate=False)
_QUINTIC = BSpline.basis_element(np.arange(7) - 3., extrapolate=False)
_GAUSS_NODES, _GAUSS_WEIGHTS = np.polynomial.legendre.leggauss(3)


def forced_cold(times, amplitude, mass_ratio=1836.):
    """Stable exact response including ion inertia, with the fixed-ion limit."""
    t = np.asarray(times)
    frequency = np.sqrt(1 + 1 / mass_ratio)
    difference = 1 / (mass_ratio * (frequency + 1))
    return -(1 + 1 / mass_ratio) * amplitude * t * np.sin((frequency + 1) * t / 2) * np.sinc(
        difference * t / (2 * np.pi)) / (frequency + 1)


def _quartic_field(electric, position, length):
    """Independent Cox-de Boor face spline, with five periodic weights per position."""
    coordinate = (position + length / 2) * len(electric) / length - 1
    indices = np.floor(coordinate + .5).astype(int)[..., None] + np.arange(-2, 3)
    weights = np.nan_to_num(_QUARTIC(coordinate[..., None] - indices))
    return np.sum(electric[indices % len(electric)] * weights, axis=-1)


def orbit_average(electric, x, displacement, length, *, shape_order=2):
    """Exact periodic face-spline average over an unwrapped 1V orbit.

    Face i is at -L/2+(i+1)dx. At most one face may be crossed; zero-length
    orbits return the local field. Quadratic charge gives a linear field;
    quintic charge gives a quartic field. Knot-split quadrature avoids large
    potential subtraction and integrates each quartic piece exactly.
    """
    electric, x = np.asarray(electric, dtype=float), np.asarray(x, dtype=float)
    displacement = np.broadcast_to(np.asarray(displacement, dtype=float), x.shape)
    if (electric.ndim != 1 or len(electric) < 2 or not np.isfinite(length) or length <= 0
            or not all(np.all(np.isfinite(a)) for a in (electric, x, displacement))):
        raise ValueError('finite periodic face fields and positive length are required')
    dx = length / len(electric)
    faces = -length / 2 + dx * np.arange(1, len(electric) + 1)
    first = np.floor((x + length / 2) / dx)
    last = np.floor((x + displacement + length / 2) / dx)
    if np.any(abs(last - first) > 1):
        raise ValueError('reference orbit crosses more than one face in a substep')
    if (not isinstance(shape_order, (int, np.integer)) or isinstance(shape_order, (bool, np.bool_))
            or shape_order not in (2, 5)):
        raise ValueError('reference shape_order must be 2 or 5')
    if shape_order == 5:
        # Quartic knots lie at cell centres. The guard above permits at most
        # two of them; integrate in orbit fractions, including zero/tiny shifts.
        knots = -length / 2 + (np.minimum(first, last)[..., None] + np.array([.5, 1.5])) * dx
        fractions = np.divide(knots - x[..., None], displacement[..., None],
                              out=np.zeros_like(knots), where=displacement[..., None] != 0)
        cuts = np.sort(np.concatenate((np.zeros(x.shape + (1,)), np.clip(fractions, 0, 1),
                                       np.ones(x.shape + (1,))), axis=-1), axis=-1)
        width = np.diff(cuts)
        points = x[..., None, None] + displacement[..., None, None] * (
            (cuts[..., :-1, None] + cuts[..., 1:, None]) / 2 + width[..., None] * _GAUSS_NODES / 2)
        return np.sum(width * (_quartic_field(electric, points, length) @ _GAUSS_WEIGHTS) / 2, axis=-1)
    edge = -length / 2 + (first + (displacement > 0)) * dx
    crossed = first != last
    fraction = np.divide(edge - x, displacement, out=np.zeros_like(displacement), where=crossed)
    start, stop, middle = (np.interp(a, faces, electric, period=length) for a in (x, x + displacement, edge))
    return np.where(crossed, .5 * ((start + middle) * fraction + (middle + stop) * (1 - fraction)),
                    .5 * (start + stop))


def _quadratic_charge(x, amounts, length, cells):
    """Independent piecewise quadratic spline; amounts are qw/(enL)."""
    coordinate = (x + length / 2) * cells / length - .5
    nearest = np.floor(coordinate + .5).astype(int)
    indices = nearest[:, None] + np.array([-1, 0, 1])
    distance = abs(coordinate[:, None] - indices)
    shape = np.where(distance <= .5, .75 - distance**2,
                     np.where(distance <= 1.5, .5 * (1.5 - distance)**2, 0.))
    density = np.zeros(cells)
    np.add.at(density, indices % cells, cells * amounts[:, None] * shape)
    return density


def _quintic_charge(x, amounts, length, cells):
    """Independent six-point Cox-de Boor deposit, including overlapping periodic images."""
    coordinate = (x + length / 2) * cells / length - .5
    indices = np.floor(coordinate).astype(int)[:, None] + np.arange(-2, 4)
    shape = np.nan_to_num(_QUINTIC(coordinate[:, None] - indices))
    density = np.zeros(cells)
    np.add.at(density, indices % cells, cells * amounts[:, None] * shape)
    return density


def midpoint_orbits(electric, x, u, charge, mass, length, dt, *, drive=0., weights=None,
                    tolerance=2e-13, iterations=64, substeps=2, shape_order=2):
    """Independently solve relativistic 1V substeps in a frozen midpoint field.

    Units are x:c/wp, u:c, E:me*c*wp/e, q:e and mass:me. ``drive`` is the
    uniform force already evaluated at the whole field-step midpoint. Optional
    weights w/(nL) give endpoint charge/(en) and mean current/(en*c).
    ``substeps`` is a positive integer; the default is two, as in the parent.
    ``shape_order`` selects quadratic (2) or quintic (5) endpoint charge and
    its conjugate linear or quartic orbit field, respectively.
    No magnetic/transverse motion is supported. Residuals describe this frozen
    field reference, not the parent's inaccessible internal Picard residual.
    """
    electric, x, u = (np.asarray(a, dtype=float) for a in (electric, x, u))
    charge, mass = (np.broadcast_to(np.asarray(a, dtype=float), x.shape) for a in (charge, mass))
    if (x.ndim != 1 or not x.size or u.shape != x.shape or electric.ndim != 1 or electric.size < 2
            or not isinstance(iterations, (int, np.integer)) or iterations < 1
            or not isinstance(substeps, (int, np.integer)) or isinstance(substeps, (bool, np.bool_)) or substeps < 1
            or not np.all(np.isfinite([length, dt, drive, tolerance])) or min(length, dt, tolerance) <= 0
            or not all(np.all(np.isfinite(a)) for a in (electric, x, u, charge, mass)) or np.any(mass <= 0)):
        raise ValueError('finite 1V arrays, positive mass/length/dt/tolerance and integer iteration/substep counts '
                         'are required')
    if np.any((x < -length / 2) | (x >= length / 2)):
        raise ValueError('initial integer-time positions must lie in the periodic box')
    initial_x, total = x.copy(), np.zeros_like(x)
    counts, residual_u, residual_x, fields, equation_u = [], [], [], [], []
    h, dx, qm = dt / substeps, length / electric.size, charge / mass
    for _ in range(substeps):
        old_u, old_x = u.copy(), x.copy()
        shift = h * old_u / np.hypot(1., old_u)
        for count in range(1, iterations + 1):
            average = orbit_average(electric, old_x, shift, length, shape_order=shape_order) + drive
            new_u = old_u + qm * h * average
            new_shift = h * (old_u + new_u) / (np.hypot(1., old_u) + np.hypot(1., new_u))
            if not np.all(np.isfinite(new_u)) or not np.all(np.isfinite(new_shift)):
                raise ValueError('reference orbit overflow')
            error = max(float(np.max(abs(new_u - u))), float(np.max(abs(new_shift - shift))) / dx)
            u, shift = new_u, new_shift
            if error <= tolerance:
                break
        average = orbit_average(electric, old_x, shift, length, shape_order=shape_order) + drive
        equation_u.append(u - old_u - qm * h * average)
        residual_u.append(float(np.max(abs(equation_u[-1]))))
        closed_shift = h * (old_u + u) / (np.hypot(1., old_u) + np.hypot(1., u))
        residual_x.append(float(np.max(abs(shift - closed_shift))) / dx)
        counts.append(count)
        fields.append(average)
        total += shift
        x = (old_x + shift + length / 2) % length - length / 2
    result = dict(x=x, u=u, displacement=total, orbit_E=np.stack(fields), iterations=counts,
                  equation_u_residual=np.stack(equation_u),
                  residual_u=residual_u, residual_x_over_dx=residual_x,
                  converged=max(*residual_u, *residual_x) <= tolerance)
    if weights is not None:
        weights = np.broadcast_to(np.asarray(weights, dtype=float), x.shape)
        if not np.all(np.isfinite(weights)) or np.any(weights < 0):
            raise ValueError('reference marker weights must be finite and nonnegative')
        amounts = charge * weights
        deposit = _quadratic_charge if shape_order == 2 else _quintic_charge
        result.update(rho_initial=deposit(initial_x, amounts, length, electric.size),
                      rho=deposit(x, amounts, length, electric.size),
                      mean_current=float(np.sum(amounts * total) / dt))
    return result


def homogeneous(times, amplitude, *, mass_ratio=1836., temperature=1e-3,
                relativistic=True, nodes=64, eta=None, omega=1., rtol=2e-10, dense_output=False):
    """Unseeded prescribed drive or finite Proca reservoir without spatial dynamics.

    ``amplitude`` is initial effective-force quiver / c. For a finite reservoir,
    D(0)=amplitude/eta; for a prescribed drive, effective external E=A cos(t).
    Energies use n me c², momentum uses n me c, time uses electron wp inverse.
    Velocity variance is a k=0-subtracted spread, not a thermodynamic temperature.
    ``dense_output`` also exposes the continuous state and initial quadrature
    for independent tangent integration; the fourth state component is impulse.
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
                         method='DOP853', rtol=rtol, atol=rtol * 0.01, dense_output=dense_output)
    if not solution.success:
        raise RuntimeError(solution.message)
    E, D, A, impulse, work = solution.y
    mean, rms, kinetic = (np.stack(values) for values in zip(*(moments(value) for value in impulse)))
    electric = E**2 / 2
    dark = (D**2 + omega**2 * A**2) / 2
    result = dict(t=t, mean_E=E, mean_D=D, mean_A=A, mean=mean, rms=rms,
                  kinetic=kinetic, spread=0.5 * masses[None, :] * rms**2,
                  electric=electric, dark=dark, work=work,
                  balance=electric + dark + kinetic.sum(axis=1) - work,
                  nfev=solution.nfev, nodes=nodes)
    if dense_output:
        result.update(dense=solution.sol, initial_u=initial, weights=weights, impulse=impulse)
    return result


def gaussian_tangent(times, k, amplitude=0., *, seed=1., delta_u=None, mass_ratio=1836.,
                     temperature=1e-3, nodes=64, eta=None, omega=1., rtol=2e-10):
    """Finite-time continuum 1V tangent about the same Gaussian homogeneous orbit.

    Units are wp_e=c=me=epsilon0=1, q=(-1,+1), and both densities are one.
    The +k coefficient of the initial cosine momentum kick is
    pi_s=-q_s/m_s*delta_u/2; default delta_u=seed*sqrt(T). Positions, A and
    phi initially have zero perturbation, so both Gauss laws start neutral.
    eta=None uses the uniform prescribed-force background without a dynamic
    dark response. Finite eta uses its bare Proca reservoir; eta=0 requires
    amplitude=0 since that amplitude denotes effective quiver, not bare D.
    This is an infinitesimal seed extension, not the unseeded Hook experiment
    or a late nonlinear trajectory. Finite quadrature needs refinement before
    interpreting long-time phase mixing; no Floquet rate is fitted.
    """
    values = [k, amplitude, seed, mass_ratio, temperature, omega, rtol,
              eta if eta is not None else 0., delta_u if delta_u is not None else 0.]
    if not np.all(np.isfinite(values)) or k == 0 or rtol <= 0:
        raise ValueError('finite tangent controls, nonzero k and positive tolerance are required')
    if eta == 0 and amplitude != 0:
        raise ValueError('zero mixing requires zero effective reservoir amplitude')
    background = homogeneous(times, amplitude, mass_ratio=mass_ratio, temperature=temperature,
                             nodes=nodes, eta=None if eta == 0 else eta, omega=omega,
                             rtol=rtol, dense_output=True)
    dense, initial, weights = background.pop('dense'), background['initial_u'], background['weights']
    charge, qm = np.array([-1., 1.]), np.array([-1., 1 / mass_ratio])
    mixing, kick = 0. if eta is None else eta, seed * np.sqrt(temperature) if delta_u is None else delta_u
    start = np.zeros(4 * nodes + 2, dtype=complex)
    start[2 * nodes:4 * nodes] = np.broadcast_to(-qm[:, None] * kick / 2, (2, nodes)).ravel()

    def rhs(time, state):
        xi, pi = state[:-2].reshape(2, 2, nodes)
        A, phi = state[-2:]
        u = initial + qm[:, None] * dense(time)[3]
        gamma = np.hypot(1., u)
        E = -np.sum(charge * (xi @ weights))
        D = (mixing * 1j * k * E - omega**2 * phi) / (1j * k)
        advect = 1j * k * u / gamma
        return np.r_[(pi / gamma**3 - advect * xi).ravel(),
                     (qm[:, None] * (E + mixing * D) - advect * pi).ravel(),
                     -D - 1j * k * phi, -1j * k * A]

    solution = solve_ivp(rhs, (0., float(background['t'][-1])), start, t_eval=background['t'],
                         method='DOP853', rtol=rtol, atol=rtol * .01 * max(abs(kick), 1e-12))
    if not solution.success:
        raise RuntimeError(solution.message)
    xi = solution.y[:2 * nodes].reshape(2, nodes, -1)
    E = -np.einsum('s,snt,n->t', charge, xi, weights)
    A, phi = solution.y[-2:]
    rho = 1j * k * E
    return dict(t=background['t'], mode_E=E, mode_D=(mixing * rho - omega**2 * phi) / (1j * k),
                mode_A=A, mode_phi=phi, rho=rho, delta_u=kick, background=background, nfev=solution.nfev)
