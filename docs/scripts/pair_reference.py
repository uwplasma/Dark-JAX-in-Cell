"""Independent linear waterbag references for ordinary and Proca pair plasmas.

Time is in 1/omega_0, velocity in c, and each species has omega_p^2=1/2.
The nonrelativistic limit propagates four edges; the relativistic reference
propagates orbit quadrature. Gauss's law closes each nonzero-k response.
"""

import numpy as np
from scipy.integrate import solve_ivp
from numpy.polynomial.legendre import leggauss


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


def coupled_response(times, k, seed, eta, mass, ordinary_field, dark_field,
                     half_width=0.05):
    """Finite-time pair response to a homogeneous Maxwell–Proca reservoir.

    The bare reservoir generally contains two incommensurate normal modes, so
    this returns an initial-value history rather than a Floquet exponent.
    Potentials and both Gauss laws are retained at finite wavenumber.
    """
    times = np.asarray(times)
    matrix = np.array([[0, 0, 0, -1], [0, 0, mass**2, -eta],
                       [0, -1, 0, 0], [1, eta, 0, 0]], dtype=float)
    background = solve_ivp(lambda t, y: matrix @ y, (0, float(times[-1])),
                           [ordinary_field, dark_field, 0, 0], dense_output=True,
                           rtol=2e-11, atol=2e-13)
    if not background.success:
        raise RuntimeError(background.message)
    charge = np.array([-1.0, -1.0, 1.0, 1.0])
    edge = np.array([-half_width, half_width] * 2)
    source = -charge * np.array([1.0, -1.0] * 2) / (4 * half_width)
    initial_edges = -seed * charge * np.array([1.0, -1.0] * 2) / (8 * half_width)

    def rhs(t, values):
        edges, potential, scalar = values[:4], values[4], values[5]
        drift = charge * background.sol(t)[3]
        rho = charge @ edges
        ordinary = rho / (1j * k)
        dark = (eta * rho - mass**2 * scalar) / (1j * k)
        next_edges = (-1j * k * (edge + drift) * edges
                      + source * (ordinary + eta * dark))
        return np.r_[next_edges, -dark - 1j * k * scalar, -1j * k * potential]

    solution = solve_ivp(rhs, (0, float(times[-1])),
                         np.r_[initial_edges, 0j, 0j], t_eval=times,
                         rtol=2e-10, atol=2e-12)
    if not solution.success:
        raise RuntimeError(solution.message)
    rho = charge @ solution.y[:4]
    ordinary = rho / (1j * k)
    dark = (eta * rho - mass**2 * solution.y[5]) / (1j * k)
    return ordinary, dark, background.sol(times)


def relativistic_background(times, eta, mass, ordinary_field, dark_field,
                            half_width=.05, nodes=64, rtol=2e-11, atol=2e-13,
                            dense_output=False, method='RK45'):
    """Warm homogeneous pair orbits; energies use total density times mec².

    This continuum background has no spatial modes. Its bare D(t) can supply
    an independent prescribed envelope; it is not the mean of a nonlinear PIC run.
    """
    times = np.asarray(times, dtype=float)
    if (times.ndim != 1 or times.size < 2 or times[0] < 0 or np.any(np.diff(times) <= 0)
            or not np.all(np.isfinite([*times, eta, mass, ordinary_field, dark_field, half_width, rtol, atol]))
            or not 0 <= half_width < 1 or mass <= 0 or min(rtol, atol) <= 0
            or not isinstance(nodes, (int, np.integer)) or nodes < 1):
        raise ValueError('finite increasing times, positive mass/tolerance/nodes and subluminal waterbag are required')
    points, weights = leggauss(nodes)
    velocity = half_width * points
    momentum = velocity / np.sqrt(1 - velocity**2)
    weights = weights / 2

    def mean_velocity(pump_momentum):
        orbit = momentum + pump_momentum
        return weights @ (orbit / np.sqrt(1 + orbit**2))

    def pump_rhs(t, state):
        ordinary, dark, potential, pump_momentum = state
        current = mean_velocity(pump_momentum)
        return [-current, mass**2 * potential - eta * current,
                -dark, ordinary + eta * dark]

    pump = solve_ivp(pump_rhs, (0, float(times[-1])),
                     [ordinary_field, dark_field, 0, 0], dense_output=True,
                     rtol=rtol, atol=atol, method=method)
    if not pump.success:
        raise RuntimeError(pump.message)
    ordinary, dark, potential, impulse = pump.sol(times)
    orbit = momentum[:, None] + impulse[None, :]
    kinetic = weights @ (orbit**2 / (np.sqrt(1 + orbit**2) + 1))
    result = dict(t=times, mean_E=ordinary, mean_D=dark, mean_A=potential, impulse=impulse,
                  current=np.array([mean_velocity(value) for value in impulse]), kinetic=kinetic,
                  energy=kinetic + .5 * (ordinary**2 + dark**2 + mass**2 * potential**2),
                  initial_momentum=momentum, weights=weights, nfev=pump.nfev)
    if dense_output:
        result['dense'] = pump.sol
    return result


def relativistic_response(times, k, seed, eta, mass, ordinary_field,
                          dark_field, half_width=0.05, nodes=64, linear_end=None, spatial_eta=None,
                          rtol=2e-8, atol=2e-10):
    """Quadrature Vlasov–Proca initial value problem with a relativistic pusher.

    Initial velocities are uniform on [-vT,vT]. Linearized displacement and
    momentum along each unperturbed orbit avoid differentiating a waterbag's
    discontinuous edge; Gauss closes the ordinary and massive fields.
    ``spatial_eta=0`` removes finite-k dark forces while preserving the coupled
    homogeneous pump. Its default retains the original self-consistent tangent.
    ``rtol`` and ``atol`` govern the perturbation ODE; the background keeps its
    original tighter tolerances.
    """
    times = np.asarray(times)
    background = relativistic_background(times, eta, mass, ordinary_field, dark_field,
                                         half_width, nodes, dense_output=True)
    momentum, weights, pump = background['initial_momentum'], background['weights'], background['dense']
    charge = np.array([-1.0, 1.0])[:, None]
    mixing = eta if spatial_eta is None else spatial_eta
    if not np.isfinite(mixing):
        raise ValueError('spatial coupling must be finite')
    start_displacement = np.zeros((2, nodes), dtype=complex)
    start_momentum = charge * seed / 2 * (1 + momentum**2)**1.5
    initial = np.r_[start_displacement.ravel(), start_momentum.ravel(), 0j, 0j]

    def fields(state):
        displacement = state[:2 * nodes].reshape(2, nodes)
        rho = -1j * k * np.sum(charge[:, 0] * (displacement @ weights)) / 2
        ordinary = rho / (1j * k)
        dark = (mixing * rho - mass**2 * state[-1]) / (1j * k)
        return ordinary, dark

    def rhs(t, state):
        displacement = state[:2 * nodes].reshape(2, nodes)
        perturbation = state[2 * nodes:4 * nodes].reshape(2, nodes)
        p = momentum[None, :] + charge * pump(t)[3]
        speed = p / np.sqrt(1 + p**2)
        acceleration = (1 + p**2)**-1.5
        ordinary, dark = fields(state)
        next_displacement = -1j * k * speed * displacement + acceleration * perturbation
        next_momentum = -1j * k * speed * perturbation + charge * (ordinary + mixing * dark)
        return np.r_[next_displacement.ravel(), next_momentum.ravel(),
                     -dark - 1j * k * state[-1], -1j * k * state[-2]]

    selected = times if linear_end is None else times[times <= linear_end]
    solution = solve_ivp(rhs, (0, float(selected[-1])), initial, t_eval=selected,
                         rtol=rtol, atol=atol)
    if not solution.success:
        raise RuntimeError(solution.message)
    coefficients = np.full((2, len(times)), np.nan + 0j)
    coefficients[:, :len(selected)] = np.array([
        fields(solution.y[:, index]) for index in range(len(selected))]).T
    return coefficients[0], coefficients[1], np.array([
        background[key] for key in ('mean_E', 'mean_D', 'mean_A', 'current')])
