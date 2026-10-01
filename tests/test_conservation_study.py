"""Independent momentum identities and reduced-ledger integration checks."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jaxincell import (Domain, Simulation, Solver, Species, elementary_charge as e,
                       epsilon_0, mass_electron as m, quiet_start, speed_of_light as c)

from darkjaxincell import DarkField, DarkSimulation, PrescribedDrive
from docs.scripts.conservation import measured_run, snapshot


WP = 1e9
N = epsilon_0 * m * WP**2 / e**2


def neutral_box(algorithm="explicit", cells=16, relativistic=False):
    """A homogeneous neutral pair has no background momentum sink."""
    length = 2 * np.pi * c / WP
    x, v = quiet_start(64, length)
    species = (Species.electrons(64, N / 2).replace(x=x, v=v),
               Species("positrons", 64, 1, m, N / 2, x=x, v=v))
    return Simulation(Domain(length, cells, time_step=.01 / WP), species,
                      Solver(algorithm=algorithm, relativistic=relativistic))


@pytest.mark.parametrize("algorithm", ["explicit", "implicit"])
def test_neutral_homogeneous_charge_momentum_and_continuity(algorithm):
    sim = neutral_box(algorithm)
    state, _ = sim.initial_state(jax.random.PRNGKey(0))
    state = state.replace(E=state.E.at[:, 0].set(.01 * m * c * WP / e))
    final, history, maxima = measured_run(sim, state, 200, 10)
    history = jax.tree.map(np.asarray, history)
    scale = N * m * c**2 * sim.domain.length
    np.testing.assert_allclose(history["t"][0], 0)
    np.testing.assert_allclose(history["grid_charge"], history["charge"],
                               atol=1e-13 * e * N * sim.domain.length)
    assert float(maxima[1]) / (scale / c) < 1e-13
    assert float(maxima[3]) / (e * N * WP) < 1e-13
    assert float(maxima[4]) * epsilon_0 / (e * N) < 1e-13
    assert float(maxima[6]) / (e * N * sim.domain.length) < 1e-13
    expected = .01 * m * c * WP / e * np.cos(history["t"] * WP)
    np.testing.assert_allclose(history["mean_E"], expected, atol=5e-5 * abs(expected[0]))
    tolerance = 3e-5 if algorithm == "explicit" else 1e-12
    assert float(maxima[0]) / float(history["balance"][0]) < tolerance
    assert float(final.time) == pytest.approx(2 / WP)


@pytest.mark.parametrize("model", ["ordinary", "proca", "drive"])
def test_every_step_ledgers_and_true_initial_sample_ignore_output_stride(model):
    p = neutral_box(relativistic=True)
    field = jnp.tile(jnp.array([.02 * m * c * WP / e, 0., 0.]), (p.domain.cells, 1))
    sim = (p if model == "ordinary" else DarkSimulation(p, DarkField(WP, .3, initial_E=field))
           if model == "proca" else DarkSimulation(p, PrescribedDrive(.3, field[0], WP)))
    state, _ = sim.initial_state(jax.random.PRNGKey(0))
    if model == "ordinary":
        state = state.replace(E=field)
    dense, history, dense_max = measured_run(sim, state, 24, 1)
    sparse, reduced, sparse_max = measured_run(sim, state, 24, 6)
    jax.tree.map(lambda a, b: np.testing.assert_allclose(a, b, rtol=1e-14, atol=1e-30), dense, sparse)
    for key in history:
        np.testing.assert_allclose(np.asarray(history[key])[::6], reduced[key], rtol=1e-14, atol=1e-30)
    np.testing.assert_array_equal(dense_max, sparse_max)
    np.testing.assert_allclose(history["balance"][0], snapshot(sim, state)["balance"], rtol=1e-14)
    assert float(dense_max[1]) < 1e-25
    if model == "proca":
        assert float(dense_max[5]) * epsilon_0 / (e * N) < 1e-13
        np.testing.assert_allclose(dense_max[0], dense.max_balance_error,
                                   atol=1e-14 * float(history["balance"][0]))
        residual = history["dark"] - history["dark"][0] - history["work"]
        np.testing.assert_allclose(dense_max[7], np.max(abs(residual)),
                                   atol=1e-14 * float(history["balance"][0]))
    if model == "drive":
        np.testing.assert_allclose(history["balance"] + history["work"],
                                   history["electric"] + history["kinetic"].sum(axis=1), rtol=1e-14)


def test_longitudinal_proca_wave_carries_potential_momentum_without_magnetic_field():
    """A travelling massive longitudinal wave has P/U=k/omega in the continuum."""
    errors = []
    for cells in (32, 64, 128):
        p = neutral_box(cells=cells)
        k, omega, amplitude = WP / c, np.sqrt(2) * WP, 1e-3
        A = jnp.zeros((cells, 3)).at[:, 0].set(amplitude * jnp.cos(k * p.domain.faces))
        phi = c**2 * k / omega * amplitude * jnp.cos(k * p.domain.grid)
        sim = DarkSimulation(p, DarkField(WP, 0, initial_A=A, initial_phi=phi))
        state, _ = sim.initial_state(jax.random.PRNGKey(0))
        # Use continuum samples independently of the constructor's discrete Gauss projection.
        E = jnp.zeros_like(A).at[:, 0].set(-WP**2 / omega * amplitude * jnp.sin(k * p.domain.faces))
        state = state.replace(E=E)
        values = snapshot(sim, state)
        np.testing.assert_array_equal(state.B, jnp.zeros_like(A))
        exact_energy = epsilon_0 * p.domain.length * WP**2 * amplitude**2 / 2
        np.testing.assert_allclose(values["dark"], exact_energy, rtol=1e-14)
        exact_momentum = exact_energy * k / omega
        errors.append(abs(float(values["momentum"][0]) / exact_momentum - 1))
        np.testing.assert_allclose(values["momentum"][1:], 0, atol=1e-30)
    np.testing.assert_allclose(np.array(errors[:-1]) / errors[1:], 4, rtol=.01)
    assert errors[-1] < 4e-4


def test_reduced_runner_rejects_an_incomplete_tail():
    sim = neutral_box()
    state, _ = sim.initial_state(jax.random.PRNGKey(0))
    with pytest.raises(ValueError, match="divisible"):
        measured_run(sim, state, 10, 3)
