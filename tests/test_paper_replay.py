"""The attached Hook–Huang–Shalaby setup, before long nonlinear comparisons."""

import numpy as np
import jax
import pytest
from jax import random
import jax.numpy as jnp
from jaxincell import elementary_charge as e, epsilon_0, mass_electron, speed_of_light as c

from examples.dark_reservoir import (array_fingerprint, paper_case, paper_initial,
                                     paper_plasma, save_compressed_state, seed_noise)
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


def test_fixed_physical_momentum_seed_is_neutral_resolved_and_differentiable():
    plain, wp = paper_plasma(32, 128, .01, 3)
    seeded, _ = paper_plasma(32, 128, .01, 3, .05, 2, .3)
    before, _ = plain.initial_state(random.PRNGKey(0))
    after, _ = seeded.initial_state(random.PRNGKey(0))
    wave = np.tile(np.cos(2 * np.pi * 2 * np.asarray(plain.species[0].x[:, 0]) / plain.domain.length + .3), 2)
    mass, charge = map(np.asarray, plain.per_particle)
    delta = -charge / e * mass_electron / mass * .05 * np.sqrt(.001) * c * wave
    np.testing.assert_allclose(np.asarray(after.u[:, 0] - before.u[:, 0]), delta, atol=3e-16 * c)
    assert abs(np.sum(mass * np.asarray(after.w) * delta)) < 1e-17 * mass_electron * c * np.sum(after.w)
    np.testing.assert_array_equal(after.w, before.w)
    assert np.max(abs(np.asarray(after.rho))) / (e * float(plain.species[0].density)) < 2e-12

    def objective(value):
        plasma, _ = paper_plasma(32, 128, .01, 3, value, 2, .3)
        state, _ = plasma.initial_state(random.PRNGKey(0))
        return jnp.sum(state.w * plasma._kinetic(plasma.per_particle[0], state.u))

    derivative = float(jax.grad(objective)(.05))
    velocity = np.asarray(seeded._velocity(after.u)[:, 0])
    exact = np.sum(mass * np.asarray(after.w) * velocity * delta / .05)
    np.testing.assert_allclose(derivative, exact, rtol=2e-10)
    finite = (float(objective(.05001)) - float(objective(.04999))) / .00002
    np.testing.assert_allclose(derivative, finite, rtol=1e-7)
    for value, mode in ((np.nan, 2), (.05, 0), (.05, 16)):
        with pytest.raises(ValueError):
            paper_plasma(32, 128, .01, 3, value, mode)
    with pytest.raises(ValueError):
        paper_plasma(64, 16, .01, 3, .05, 16)


@pytest.mark.parametrize('order', [2, 5])
def test_seed_current_removes_the_momentum_kick_and_matches_charge_frechet(order):
    from jaxincell._core import deposit
    plain, _ = paper_plasma(64, 256, .01, 3, shape_order=order)
    seeded, _ = paper_plasma(64, 256, .01, 3, .05, 2, .37, order)
    state, _ = seeded.initial_state(random.PRNGKey(0))
    result = seed_noise(seeded, state, .05, 2, .37)
    mass, charge = map(np.asarray, seeded.per_particle)
    x = np.concatenate([np.asarray(s.x[:, 0]) for s in plain.species])
    thermal = np.concatenate([np.asarray(s.v[:, 0]) for s in plain.species])
    d, density = seeded.domain, float(seeded.species[0].density)
    k = 4 * np.pi / d.length
    coefficients = charge * np.asarray(state.w) / (e * density * d.length * c)
    expected = np.sum(coefficients * thermal * np.exp(-1j * k * x))
    observed = complex(*result['particle']['total']['thermal_current'])
    np.testing.assert_allclose(observed, expected, rtol=2e-13, atol=2e-17)
    np.testing.assert_allclose(result['thermal_temperature_over_mec2'], .001, rtol=2e-13)
    np.testing.assert_allclose(result['charge_mode'], 0, atol=1e-13)
    np.testing.assert_array_equal(result['seed_charge_mode'], [0., 0.])
    # Differentiate actual charge deposition along thermal velocities; continuity fixes face current.
    delta = 1e-6 / 1e9
    rho = [(np.asarray(deposit(x + sign * delta * thermal, charge * np.asarray(state.w),
                               d.grid[0], d.dx, d.cells, (0, 0), order))) for sign in (-1, 1)]
    rho_dot = np.mean((rho[1] - rho[0]) * np.exp(-1j * k * np.asarray(d.grid))) / (2 * delta)
    exact_face = -rho_dot / (1j * 2 * np.sin(k * d.dx / 2) / d.dx * e * density * c)
    observed_face = complex(*result['continuity_face']['total']['thermal_current'])
    np.testing.assert_allclose(observed_face, exact_face, rtol=2e-6)
    with pytest.raises(ValueError, match='quadratic/quintic deposition'):
        seed_noise(seeded.replace(solver=seeded.solver.replace(filter_passes=1)), state, .05, 2, .37)


@pytest.mark.parametrize('coupling', [None, .1])
@pytest.mark.parametrize('order', [2, 5])
def test_seeded_scalar_archive_separates_thermal_loading_and_internal_work(tmp_path, coupling, order):
    from docs.scripts.compare_replays import _load
    history, settings, result = paper_case(tmp_path, 32, 256, .02, 1, 3, .03, coupling,
                                           block_horizon=.5, momentum_seed=.05, seed_mode=2, seed_phase=.37,
                                           shape_order=order)
    _load(tmp_path, 1e-8)
    np.testing.assert_allclose(result['initial_seed_noise']['thermal_temperature_over_mec2'], .001, rtol=2e-13)
    np.testing.assert_allclose(settings['initial_rms_over_c'], history['rms'][0], rtol=2e-13)
    with np.load(tmp_path / 'data.npz') as a:
        assert a['linear_mode_E'].shape == a['linear_t'].shape == history['t'].shape
        assert np.isfinite(a['linear_mode_D']).all()
    if coupling is None:
        np.testing.assert_array_equal(history['dark_coherent'], 0)
    else:
        np.testing.assert_allclose(history['dark_coherent'][0], history['dark'][0], rtol=2e-13)
    assert result['max_dark_work_defect_over_nmc2L'] >= 0
    assert result['max_ordinary_work_defect_over_nmc2L'] >= 0
    assert settings['mode_basis'] == 'physical exp(-ikx) at faces'
    assert settings['shape_order'] == order
    from docs.scripts.compare_replays import _normalization
    changed = {**settings, 'thermal_temperature_over_mec2': [.001, .0010000001]}
    with pytest.raises(ValueError, match='reconstructed thermal'):
        _normalization(history, changed)
    with pytest.raises(ValueError, match='reconstructed thermal'):
        _normalization(history, {**settings, 'initial_rms_over_c': settings['initial_rms_over_c'][:1]})


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


@pytest.fixture(scope='module', params=[None, .1])
def paper_checkpoint(tmp_path_factory, request):
    """Four native steps, with a deliberately inherited global diagnostic bound."""
    from examples import dark_reservoir as example
    from darkjaxincell import DarkField

    folder = tmp_path_factory.mktemp('paper_checkpoint')
    plasma, wp = paper_plasma(8, 16, .125, 0, shape_order=5)
    field, energy = mass_electron * c * wp / e, plasma.species[0].density * mass_electron * c**2 * plasma.domain.length
    amplitude = .03 * np.sqrt(.001)
    coupling = request.param
    model = (PrescribedDrive(1., jnp.array([amplitude * field, 0., 0.]), wp) if coupling is None
             else DarkField(wp, coupling, initial_E=jnp.tile(
                 jnp.array([amplitude * field / coupling, 0., 0.]), (8, 1))))
    sim = DarkSimulation(plasma, model)
    origin, fingerprints = paper_initial(sim, 0, None)
    origin = origin.replace(max_balance_error=jnp.asarray(1e-4 * energy))
    inherited = jnp.array([1e-4 * energy, 0., 0., 0., 0., 0., 0., 0., 0.])
    scales = jnp.array([2., 4.]) * np.sqrt(.001) * c / wp
    final, history, maximum, *_ = example.paper_run(sim, origin, 4, 2, .5, wp, scales, folder, maxima=inherited)
    save_compressed_state(folder / 'final_state.npz', final, sim)
    example.paper_normalize(history, plasma, wp, energy, field)
    settings = dict(cells=8, particles_per_species=16, dt_omega_p=.125, horizon_omega_p=float(history['t'][-1]),
                    seed=0, length_c_over_omega_p=40, mass_ratio=1836, T_each_over_mec2=.001,
                    drive_quiver_over_sigma=.03, force_quiver_over_c=amplitude, coupling=coupling,
                    output_dt_omega_p=.25, block_steps=4, block_horizon_omega_p=.5,
                    local_moments_output_dt_omega_p=.25,
                    local_spread_lengths_c_over_wp=(np.array([2., 4.]) * np.sqrt(.001)).tolist(),
                    normalization=dict(omega_p_rad_s=wp, field_scale_V_m=field, energy_scale_J_m2=energy,
                                       charge_density_C_m3=e * plasma.species[0].density,
                                       epsilon0_F_m=float(epsilon_0), c_m_s=float(c)),
                    initial_fingerprints=fingerprints, shape_order=5, recorded_mode=1,
                    momentum_seed_over_sigma_e=0., seed_mode=None, seed_phase=None,
                    XLA_FLAGS=example.pair_xla_flags(), parent_revision=example.parent_revision())
    results = dict(zip(example.PAPER_MAXIMUM_KEYS,
                       (np.asarray(maximum) / example.paper_maximum_scales(plasma, wp, energy)).tolist()))
    # Reference arrays are prefix evidence; they must not acquire a fictitious longer time axis.
    example.save_run(folder, 'paper_resonant_conversion', settings, results, **history,
                     homogeneous_mean_E=np.zeros(len(history['t'])))
    return folder, sim, origin, inherited, scales


def test_paper_continuation_matches_uninterrupted_native_ledgers_and_prefix(tmp_path, paper_checkpoint):
    import json
    from examples import dark_reservoir as example

    donor, sim, origin, inherited, scales = paper_checkpoint
    setting = json.loads((donor / 'run.json').read_text())['settings']
    normalization = setting['normalization']
    wp, energy, field = [normalization[key] for key in ('omega_p_rad_s', 'energy_scale_J_m2', 'field_scale_V_m')]
    full, full_history, full_maximum, *_ = example.paper_run(
        sim, origin, 8, 2, .5, wp, scales, tmp_path / 'full', maxima=inherited)
    save_compressed_state(tmp_path / 'full' / 'final_state.npz', full, sim)
    example.paper_normalize(full_history, sim.plasma, wp, energy, field)
    joined, metadata, result = example.paper_continue(tmp_path / 'continued', donor / 'final_state.npz', 1.)
    with np.load(tmp_path / 'full' / 'final_state.npz') as whole, \
            np.load(tmp_path / 'continued' / 'final_state.npz') as restarted:
        assert whole.files == restarted.files
        for key in whole.files:
            np.testing.assert_array_equal(whole[key], restarted[key])
    for key in joined:
        np.testing.assert_array_equal(joined[key], full_history[key])
    with np.load(donor / 'data.npz') as prefix, np.load(tmp_path / 'continued' / 'data.npz') as merged:
        for key in joined:
            np.testing.assert_array_equal(merged[key][:len(prefix['t'])], prefix[key])
        assert 'homogeneous_mean_E' not in merged
        # Resetting the reference would conceal the nonzero defect already accumulated at the boundary.
        global_defect = joined['balance'][-1] - joined['balance'][0]
        segment_defect = joined['balance'][-1] - prefix['balance'][-1]
        assert abs(global_defect - segment_defect) > 1e-14
    assert (tmp_path / 'continued' / 'prefix_data.npz').read_bytes() == (donor / 'data.npz').read_bytes()
    assert (tmp_path / 'continued' / 'prefix_run.json').read_bytes() == (donor / 'run.json').read_bytes()
    with np.load(tmp_path / 'continued' / 'segment_data.npz') as segment:
        np.testing.assert_equal(segment['t'][0] * wp, joined['t'][2])
        np.testing.assert_equal(segment['work'][-1] / energy, joined['work'][-1])
    for old, new in (('initial_state.npz', 'initial_state.npz'), ('final_state.npz', 'segment_initial_state.npz')):
        with np.load(donor / old) as a, np.load(tmp_path / 'continued' / new) as b:
            for key in a.files:
                np.testing.assert_array_equal(a[key], b[key])
    np.testing.assert_array_equal(result['all_step_maxima_SI'], np.maximum(
        np.asarray(full_maximum), np.asarray(result['reconstructed_prior_maxima_SI'])))
    assert result['reconstructed_prior_maxima_SI'][0] >= float(inherited[0])
    assert metadata['continuation']['start_step'] == 4 and metadata['continuation']['end_step'] == 8
    assert metadata['continuation']['prefix_native_git']
    assert str(donor) not in json.dumps(metadata)


@pytest.mark.parametrize('change', ['clock', 'steps', 'work', 'background', 'model', 'dt', 'runtime',
                                    'blocks', 'cadence', 'maximum', 'horizon'])
def test_paper_continuation_rejects_inconsistent_checkpoint_metadata(tmp_path, paper_checkpoint, change):
    import json
    import shutil
    from examples import dark_reservoir as example

    original = paper_checkpoint[0]
    donor = tmp_path / 'donor'
    shutil.copytree(original, donor)
    path = donor / 'final_state.npz'
    if change in ('clock', 'steps', 'work', 'background', 'model', 'dt'):
        with np.load(path) as stored:
            arrays = dict(stored)
        key = dict(clock='time', steps='steps', work='dark.work', background='dark.background',
                   model='dark.omega', dt='dark.dt')[change]
        arrays[key] = np.asarray(.5) if change == 'steps' else arrays[key] + (1. if change != 'clock' else 1e-12)
        np.savez_compressed(path, **arrays)
    else:
        record = json.loads((donor / 'run.json').read_text())
        if change == 'runtime':
            record['jax'] = 'unsupported'
        elif change == 'blocks':
            record['settings']['block_steps'] = 4.
        elif change == 'cadence':
            record['settings']['local_moments_output_dt_omega_p'] = .125
        elif change == 'maximum':
            record['results']['max_energy_work_defect_over_initial_thermal'] = 0.
        (donor / 'run.json').write_text(json.dumps(record))
    with pytest.raises(ValueError):
        example.paper_continue(tmp_path / 'continued', path, .5 if change == 'horizon' else 1.)


def test_paper_continuation_named_input_dispatch():
    """Exercise the named mode without launching a second physics fixture."""
    import ast
    from pathlib import Path
    from examples import dark_reservoir as example

    tree = ast.parse(Path(example.__file__).read_text())
    dispatcher = next(node for node in tree.body if isinstance(node, ast.If))
    calls = []
    namespace = dict(__name__='__main__', study='paper_continue', output='out', initial_state='donor/final_state.npz',
                     horizon=5000, paper_continue=lambda *args: calls.append(args))
    exec(compile(ast.Module(body=[dispatcher], type_ignores=[]), example.__file__, 'exec'), namespace)
    assert calls == [('out', 'donor/final_state.npz', 5000)]
