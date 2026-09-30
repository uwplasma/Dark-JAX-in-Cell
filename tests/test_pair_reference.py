"""Analytic checks for the independent oscillating-pair waterbag calculation."""

import numpy as np

from docs.scripts.pair_reference import edge_matrix, growth, seeded_response


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
