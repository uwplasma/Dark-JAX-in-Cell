"""The attached Hook–Huang–Shalaby setup, before long nonlinear comparisons."""

import numpy as np
import pytest
from jax import random
import jax.numpy as jnp
from jaxincell import elementary_charge as e, epsilon_0, mass_electron, speed_of_light as c

from examples.dark_reservoir import (array_fingerprint, paper_case, paper_initial,
                                     paper_plasma, save_compressed_state)
from darkjaxincell import DarkSimulation, PrescribedDrive, load_state


def test_paper_thermal_normalization_and_shared_neutral_loading():
    """Paper sigma is RMS, each species has T=.001 mec², and no spatial seed."""
    plasma, wp = paper_plasma(64, 512, .02, 17)
    electrons, ions = plasma.species
    np.testing.assert_array_equal(electrons.x, ions.x)
    for species in plasma.species:
        velocity = np.asarray(species.v[:, 0])
        np.testing.assert_allclose(np.mean(velocity) / c, 0, atol=2e-18)
        np.testing.assert_allclose(species.mass * np.mean(velocity**2) / (mass_electron * c**2),
                                   .001, rtol=1e-14)
        np.testing.assert_array_equal(species.v[:, 1:], 0)
    assert float(ions.mass / electrons.mass) == 1836
    assert plasma.solver.relativistic
    assert plasma.domain.cells == 64
    np.testing.assert_allclose(plasma.domain.length * wp / c, 40)
    first, _ = plasma.initial_state(random.PRNGKey(0))
    scale = e * float(electrons.density)
    assert np.max(abs(np.asarray(first.rho))) / scale < 1e-12
    np.testing.assert_allclose(electrons.density, epsilon_0 * mass_electron * wp**2 / e**2)


def test_paper_seed_is_reproducible_and_species_draws_are_independent():
    """One host seed recreates loading; electrons and ions use distinct Gaussian draws."""
    first, _ = paper_plasma(16, 256, .02, 3)
    repeat, _ = paper_plasma(32, 256, .01, 3)
    changed, _ = paper_plasma(16, 256, .02, 4)
    for a, b in zip(first.species, repeat.species):
        np.testing.assert_array_equal(a.x, b.x)
        np.testing.assert_array_equal(a.v, b.v)
    assert not np.array_equal(first.species[0].v, changed.species[0].v)
    electron = np.asarray(first.species[0].v[:, 0]) / c
    ion = np.asarray(first.species[1].v[:, 0]) * np.sqrt(1836) / c
    assert abs(np.corrcoef(electron, ion)[0, 1]) < .2


@pytest.mark.parametrize('args', [(3, 100, .02, 0), (16, 1, .02, 0),
                                  (16, 100, 0, 0), (16, 100, np.nan, 0)])
def test_paper_rejects_invalid_loading(args):
    with pytest.raises(ValueError):
        paper_plasma(*args)


@pytest.mark.parametrize('horizon,ratio,eta', [
    (0, .03, None), (1, -1, None), (1, np.inf, None), (1, .03, 0), (1, .03, np.nan)])
def test_paper_rejects_invalid_experiment(tmp_path, horizon, ratio, eta):
    with pytest.raises(ValueError):
        paper_case(tmp_path, 16, 64, .02, horizon, 0, ratio, eta)


def test_paper_scalar_archive_retains_initial_values_and_independent_control(tmp_path):
    """Host normalization and figure rendering complete without particle histories."""
    history, settings, results = paper_case(tmp_path, 16, 64, .02, 1, 0, .03,
                                            block_horizon=.5, local_moments=True)
    assert history['t'][0] == 0 and history['t'][-1] == pytest.approx(1)
    assert settings['force_quiver_over_c'] == pytest.approx(.03 * np.sqrt(.001))
    assert settings['inferred_t_noise'] == pytest.approx(40 / (.03 * np.sqrt(.001 * 3 * 64 * 16)))
    np.testing.assert_allclose(history['spread'][0], [.0005, .0005], rtol=1e-13)
    assert (tmp_path / 'figure.png').is_file()
    with np.load(tmp_path / 'data.npz') as stored:
        np.testing.assert_array_equal(stored['kinetic'], history['kinetic'])
        assert stored['homogeneous_balance'].shape == stored['t'].shape
        assert stored['momentum'].shape == (len(stored['t']), 3)
        assert stored['local_spread_initial'].shape == (2, 2)
        assert stored['local_spread_final'].shape == (2, 2)
        assert np.all(stored['local_spread_initial'] > 0)
        assert np.all(stored['local_spread_final'] > 0)
        np.testing.assert_allclose(stored['local_spread'][0], stored['local_spread_initial'])
        np.testing.assert_allclose(stored['local_spread'][-1], stored['local_spread_final'])
        assert stored['local_density_rms'].shape == (3, 2, 2)
    assert settings['block_steps'] == 25
    assert settings['initial_fingerprints']['state']['w']
    assert (tmp_path / 'initial_state.npz').is_file()
    assert np.isfinite(results['max_energy_work_defect_over_initial_thermal'])
    plasma, wp = paper_plasma(16, 64, .02, 0)
    force = .03 * np.sqrt(.001) * mass_electron * c * wp / e
    sim = DarkSimulation(plasma, PrescribedDrive(1., jnp.array([force, 0., 0.]), wp))
    restored = load_state(tmp_path / 'final_state.npz', sim)
    assert restored.ordinary.time * wp == pytest.approx(1)
    assert len(restored.ordinary.w) == 128
    energy_scale = float(plasma.species[0].density) * mass_electron * c**2 * plasma.domain.length
    assert restored.work / energy_scale == pytest.approx(history['work'][-1])


def test_paper_archive_reuses_exact_initial_arrays_and_rejects_wrong_loading(tmp_path):
    plasma, wp = paper_plasma(8, 16, .02, 0)
    sim = DarkSimulation(plasma, PrescribedDrive(1., jnp.array([1., 0., 0.]), wp))
    start, original = paper_initial(sim, 0, None)
    path = tmp_path / 'initial.npz'
    save_compressed_state(path, start, sim)
    repeated, fingerprint = paper_initial(sim, 0, path)
    assert original == fingerprint
    assert array_fingerprint(np.arange(4, dtype=np.float64)) != array_fingerprint(np.arange(4, dtype=np.float32))
    assert repeated.ordinary.time == 0
    other, _ = paper_plasma(8, 16, .02, 1)
    with pytest.raises(AssertionError):
        paper_initial(DarkSimulation(other, sim.dark), 1, path)
