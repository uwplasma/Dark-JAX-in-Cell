"""Independent homogeneous oracles, including detuning and source-work signs."""

import numpy as np
import pytest

from docs.scripts.drive_reference import forced_cold, homogeneous


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
