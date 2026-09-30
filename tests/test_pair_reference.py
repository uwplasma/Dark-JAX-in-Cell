"""Analytic checks for the independent oscillating-pair waterbag calculation."""

import numpy as np

from docs.scripts.pair_reference import (coupled_response, edge_matrix, growth,
                                         relativistic_response, seeded_response)


def test_unpumped_pair_dispersion_and_ballistic_edges():
    """The stationary pair has one Langmuir pair and two free waterbag edges."""
    for b in (0.2, 0.6, 1.0):
        frequency = np.sort(np.abs(np.linalg.eigvals(edge_matrix(0, b / .05, .05, 0))))
        np.testing.assert_allclose(frequency, [b, b, np.hypot(1, b), np.hypot(1, b)],
                                   rtol=1e-12)


def test_oscillating_pair_has_thermal_band_and_stable_control():
    """The time-dependent propagator resolves growth absent from the static limit."""
    growing, multiplier = growth(0.6 / .05)
    stable, _ = growth(0.4 / .05)
    assert 0.09 < growing < 0.11
    assert multiplier.real > 1
    assert abs(stable) < 1e-8
    assert abs(growth(0.6 / .05, quiver=0)[0]) < 1e-8


def test_seeded_response_starts_from_ampere_current():
    """Opposite velocity nudges have zero initial charge and J_k=seed/2."""
    seed = 2e-4
    times = np.array([0., 1e-4, 2e-4])
    field = seeded_response(times, 12, seed)
    assert abs(field[0]) < 1e-14
    np.testing.assert_allclose((field[2] - field[0]).real / times[2], -seed / 2,
                               rtol=1e-6)


def test_coupled_initial_value_reference_has_ordinary_limit_and_energy():
    """No mixing recovers the ordinary kinetic mode; a closed pump preserves energy."""
    times = np.linspace(0, 35, 176)
    quiver, k, seed = 0.2 / np.sqrt(2), 12, 2e-4
    ordinary, dark, _ = coupled_response(times, k, seed, 0, 1, quiver, 0)
    np.testing.assert_allclose(ordinary, seeded_response(times, k, seed), atol=1e-10)
    np.testing.assert_allclose(dark, 0, atol=1e-12)
    _, _, background = coupled_response(times, k, seed, 0.5, 1, 0, 0.2)
    np.testing.assert_allclose(np.sum(background**2, axis=0), 0.2**2, atol=1e-10)


def test_relativistic_quadrature_preserves_initial_ampere_current():
    """A velocity nudge gives the exact current even though momentum varies with speed."""
    times = np.array([0., 1e-4, 2e-4])
    ordinary, dark, _ = relativistic_response(times, 12, 2e-4, 0.5, 1, 0, 0.1, nodes=16)
    np.testing.assert_allclose(ordinary[0], 0, atol=1e-14)
    np.testing.assert_allclose(ordinary[2].real / times[2], -1e-4, rtol=1e-6)
    np.testing.assert_allclose(dark[2].real / times[2], -5e-5, rtol=1e-6)


def test_relativistic_velocity_quadrature_refines():
    """The independent kinetic trace is stable when velocity nodes double."""
    times = np.linspace(0, 10, 51)
    coarse = relativistic_response(times, 12, 2e-4, 0.5, 1, 0, 0.1, nodes=16)
    fine = relativistic_response(times, 12, 2e-4, 0.5, 1, 0, 0.1, nodes=32)
    assert np.linalg.norm(coarse[0] - fine[0]) / np.linalg.norm(fine[0]) < 1e-4
