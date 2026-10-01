"""Fixed-scale moment checks against continuum flow and population invariance."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jaxincell import Domain, Simulation, Species, mass_electron as m, quiet_start

from docs.scripts.conservation import coarse_spread


def loading(cells, cohorts=1):
    count, length = cells * 16, 2 * np.pi
    x, v = quiet_start(count, length)
    v = v.at[:, 0].set(.1 * jnp.sin(x[:, 0]))
    species = tuple(Species.electrons(count, 1 / cohorts).replace(name=f"cohort{i}", x=x, v=v)
                    for i in range(cohorts))
    sim = Simulation(Domain(length, cells, time_step=1e-12), species)
    state, _ = sim.initial_state(jax.random.PRNGKey(0))
    return sim, state


def test_gaussian_scale_has_continuum_mean_flow_limit():
    """Smoothing sinusoidal cold flow gives nmLV²(1-exp(-k²ell²))/4."""
    scales = (.2, .4)
    exact = m * 2 * np.pi * .1**2 / 4 * (1 - np.exp(-np.array(scales)**2))
    errors = []
    for cells in (32, 64, 128):
        sim, state = loading(cells)
        measured = np.asarray(coarse_spread(sim, state, scales))[0]
        errors.append(np.max(abs(measured / exact - 1)))
    np.testing.assert_allclose(np.array(errors[:-1]) / errors[1:], 4, rtol=.02)
    assert errors[-1] < .02


def test_population_splitting_and_uniform_flow_leave_random_energy_unchanged():
    sim, state = loading(64)
    split, split_state = loading(64, cohorts=2)
    reference = coarse_spread(sim, state, (.2, .4))
    np.testing.assert_allclose(coarse_spread(split, split_state, (.2, .4), ((0, 1),)),
                               reference, rtol=1e-13)
    shifted = state.replace(u=state.u.at[:, 0].add(.3))
    np.testing.assert_allclose(coarse_spread(sim, shifted, (.2, .4)), reference, rtol=1e-10)
    opposed = split_state.replace(u=split_state.u.at[:, 0].set(
        jnp.concatenate((jnp.ones(sim.species[0].n), -jnp.ones(sim.species[0].n)))))
    combined = coarse_spread(split, opposed, (.2,), ((0, 1),))
    np.testing.assert_allclose(combined, m * np.pi, rtol=1e-13)
    assert float(coarse_spread(split, opposed, (.2,)).sum()) < 1e-12 * float(combined[0, 0])


@pytest.mark.parametrize("change", [{"mass": 2 * m}, {"charge": 1}])
def test_different_mass_or_charge_cannot_share_a_physical_species(change):
    sim, state = loading(32, cohorts=2)
    sim = sim.replace(species=(sim.species[0], sim.species[1].replace(**change)))
    with pytest.raises(ValueError, match="one mass"):
        coarse_spread(sim, state, (.2,), ((0, 1),))
