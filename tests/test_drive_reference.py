"""Independent homogeneous oracles, including detuning and source-work signs."""

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from docs.scripts.drive_reference import forced_cold, homogeneous, midpoint_orbits, gaussian_tangent


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


def test_cold_gaussian_tangent_has_exact_plasma_frequency_phase_and_charge_sign():
    t, mass, kick, k = np.linspace(0, 12, 121), 7., .002, .6
    result = gaussian_tangent(t, k, temperature=0., delta_u=kick, mass_ratio=mass, nodes=1, rtol=2e-11)
    frequency = np.sqrt(1 + 1 / mass)
    exact = kick * frequency * np.sin(frequency * t) / 2
    np.testing.assert_allclose(result['mode_E'], exact, rtol=1e-9, atol=2e-12)
    np.testing.assert_allclose(result['rho'], 1j * k * exact, rtol=1e-9, atol=2e-12)
    np.testing.assert_array_equal(result['mode_D'], 0.)
    assert result['mode_E'][1].real > 0 and result['rho'][1].imag > 0


def test_cold_proca_tangent_matches_independent_two_frequency_solution():
    t, mass, kick, k, eta, omega = np.linspace(0, 12, 121), 7., .002, .6, .25, 1.3
    result = gaussian_tangent(t, k, temperature=0., delta_u=kick, mass_ratio=mass,
                              nodes=1, eta=eta, omega=omega, rtol=2e-11)
    # Cold E,D satisfy y''=-K y, y(0)=0 and y'(0)=kick*(1+1/m)/2*(1,eta).
    plasma = 1 + 1 / mass
    matrix = np.array([[plasma, eta * plasma], [eta * (plasma - k**2), omega**2 + k**2 + eta**2 * plasma]])
    low, high = np.sort(np.linalg.eigvals(matrix))
    upper = (matrix - low * np.eye(2)) / (high - low)
    derivative = kick * plasma / 2 * np.array([1., eta])
    exact = np.outer((np.eye(2) - upper) @ derivative, np.sin(np.sqrt(low) * t) / np.sqrt(low))
    exact += np.outer(upper @ derivative, np.sin(np.sqrt(high) * t) / np.sqrt(high))
    np.testing.assert_allclose(result['mode_E'], exact[0], rtol=1e-9, atol=2e-12)
    np.testing.assert_allclose(result['mode_D'], exact[1], rtol=1e-9, atol=2e-12)
    np.testing.assert_allclose(result['mode_phi'], 1j * k * (eta * exact[0] - exact[1]) / omega**2,
                               rtol=1e-9, atol=2e-12)


def test_zero_mixing_tangent_reduces_to_ordinary_gaussian_response():
    t = np.linspace(0, 8, 41)
    ordinary = gaussian_tangent(t, .8, seed=.02, nodes=16)
    unmixed = gaussian_tangent(t, .8, seed=.02, nodes=16, eta=0.)
    np.testing.assert_allclose(unmixed['mode_E'], ordinary['mode_E'], rtol=0, atol=1e-15)
    np.testing.assert_array_equal(unmixed['mode_D'], 0.)
    with pytest.raises(ValueError, match='zero effective'):
        gaussian_tangent(t, .8, .01, eta=0.)
    with pytest.raises(ValueError, match='nonzero k'):
        gaussian_tangent(t, 0.)


def test_gaussian_tangent_refines_quadrature_and_background_tolerance():
    t = np.linspace(0, 8, 41)
    controls = dict(amplitude=.008, seed=.02, mass_ratio=32., eta=.3, omega=.9)
    coarse = gaussian_tangent(t, .8, nodes=16, rtol=1e-8, **controls)
    resolved = gaussian_tangent(t, .8, nodes=32, rtol=1e-8, **controls)
    refined = gaussian_tangent(t, .8, nodes=32, rtol=2e-11, **controls)
    for key in ('mode_E', 'mode_D'):
        np.testing.assert_allclose(coarse[key], resolved[key], rtol=2e-6, atol=2e-10)
        np.testing.assert_allclose(resolved[key], refined[key], rtol=2e-6, atol=2e-10)
    background = refined['background']
    for key in ('mean_E', 'mean_D', 'impulse', 'mean', 'rms', 'kinetic'):
        np.testing.assert_allclose(resolved['background'][key], background[key], rtol=2e-6, atol=2e-9)
    np.testing.assert_allclose(background['balance'], background['balance'][0], rtol=0, atol=3e-10)
    assert 'dense' not in background and background['initial_u'].shape == (2, 32)
