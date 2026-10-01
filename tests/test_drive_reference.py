"""Independent homogeneous oracles, including detuning and source-work signs."""

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from docs.scripts.drive_reference import forced_cold, homogeneous, midpoint_orbits


def test_mobile_ion_oracle_and_infinite_mass_limit():
    t = np.linspace(0, 80, 401)
    amplitude = .03 * np.sqrt(.001)
    exact = forced_cold(t, amplitude)
    response = homogeneous(t, amplitude, relativistic=False)
    np.testing.assert_allclose(response['mean_E'], exact, atol=2e-10)
    np.testing.assert_allclose(forced_cold(t, amplitude, 1e18),
                               -amplitude * t * np.sin(t) / 2, atol=1e-15)
    np.testing.assert_allclose(response['balance'], response['balance'][0], atol=3e-11)
    np.testing.assert_allclose(response['rms'],
                               np.broadcast_to(np.sqrt(.001 / np.array([1., 1836.])), (len(t), 2)),
                               atol=1e-16)


def test_relativistic_quadrature_and_closed_reservoir():
    t = np.linspace(0, 80, 161)
    response = homogeneous(t, .02, eta=.2)
    refined = homogeneous(t, .02, eta=.2, nodes=96, rtol=1e-11)
    np.testing.assert_allclose(response['mean_E'], refined['mean_E'], atol=2e-9)
    np.testing.assert_allclose(response['balance'], response['balance'][0], atol=2e-10)
    assert np.max(abs(response['mean_E'])) > .02
    assert np.max(abs(response['mean'])) < 1
    np.testing.assert_allclose(response['work'], 0., atol=0)


def test_zero_drive_and_oracle_controls():
    t = np.linspace(0, 4, 9)
    result = homogeneous(t, 0.)
    np.testing.assert_allclose(result['mean_E'], 0., atol=1e-16)
    np.testing.assert_allclose(result['spread'], .0005, atol=1e-15)
    with pytest.raises(ValueError, match='increasing'):
        homogeneous([0, 1, .5], .01)
    with pytest.raises(ValueError, match='parameters'):
        homogeneous(t, .01, eta=0.)
    with pytest.raises(ValueError, match='exceed c'):
        homogeneous(t, .01, temperature=.1)


@pytest.mark.parametrize('substeps', [1, 2, 3, 4, 8])
def test_constant_force_orbit_is_exact_for_any_positive_substep_count(substeps):
    x, u = np.array([-.9, -.1, .99]), np.array([.05, -.1, .7])
    charge, mass = np.array([-1., 1., -1.]), np.array([1., 1836., 2.])
    result = midpoint_orbits(np.full(32, .2), x, u, charge, mass, 2., .1, drive=.03, substeps=substeps)
    exact_u = u + charge / mass * .1 * .23
    # For du/dt=constant, integral(v dt)=(gamma_end-gamma_start)/(du/dt).
    exact_shift = .1 * (u + exact_u) / (np.hypot(1., u) + np.hypot(1., exact_u))
    assert result['converged'] and len(result['iterations']) == substeps
    np.testing.assert_allclose(result['u'], exact_u, rtol=0, atol=2e-15)
    np.testing.assert_allclose(result['displacement'], exact_shift, rtol=0, atol=2e-15)


def test_linear_field_closes_relativistic_work_and_converges_to_continuum_orbit():
    # All orbits stay in -1<x<1, where this periodic face field is E=.3x.
    field, x, u, dt = np.array([-.3, 0., .3, 0.]), .2, .1, .4
    exact = solve_ivp(lambda t, y: [y[1] / np.hypot(1., y[1]), .3 * y[0]],
                      (0., dt), [x, u], method='DOP853', rtol=2e-12, atol=1e-14).y[:, -1]
    errors = []
    for substeps in (1, 2, 4):
        result = midpoint_orbits(field, [x], [u], 1., 1., 4., dt, substeps=substeps)
        assert result['converged']
        end_x, end_u = result['x'][0], result['u'][0]
        kinetic = end_u**2 / (np.hypot(1., end_u) + 1) - u**2 / (np.hypot(1., u) + 1)
        np.testing.assert_allclose(kinetic, .15 * (end_x**2 - x**2), rtol=0, atol=2e-14)
        errors.append(np.linalg.norm([end_x - exact[0], end_u - exact[1]]))
    assert errors[0] > 3 * errors[1] > 9 * errors[2]


def test_substep_refinement_resolves_face_crossings_and_rejects_unsupported_counts():
    args = (np.zeros(4), [-.1], [2.], 1., 1., 2., 1.2)
    with pytest.raises(ValueError, match='more than one face'):
        midpoint_orbits(*args, substeps=1)
    resolved = midpoint_orbits(*args, substeps=4)
    np.testing.assert_allclose(resolved['displacement'], 1.2 * 2 / np.sqrt(5), rtol=0, atol=2e-15)
    for substeps in (0, -1, 2.5, True):
        with pytest.raises(ValueError, match='integer iteration/substep'):
            midpoint_orbits(*args, substeps=substeps)
    default = midpoint_orbits(np.zeros(4), [0.], [0.], 1., 1., 2., .1)
    explicit = midpoint_orbits(np.zeros(4), [0.], [0.], 1., 1., 2., .1, substeps=2)
    np.testing.assert_array_equal(default['orbit_E'], explicit['orbit_E'])
