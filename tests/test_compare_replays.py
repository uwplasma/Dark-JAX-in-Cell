"""Native-record guards reject mismatched clocks, units and controlled loading."""
import json
from shutil import copytree

import numpy as np
import pytest

from docs.scripts.compare_replays import (_projection, compare_replays, constraint_audit,
                                          fingerprint, implicit_comparison, metrics)


@pytest.fixture
def records(tmp_path):
    settings = dict(cells=1000, particles_per_species=103000, seed=0, dt_omega_p=.01,
                    output_dt_omega_p=.5, horizon_omega_p=1., length_c_over_omega_p=40,
                    mass_ratio=1836, T_each_over_mec2=.001, coupling=None,
                    drive_quiver_over_sigma=.03, force_quiver_over_c=.03 * np.sqrt(.001),
                    loading='conditioned Gaussian', pusher='Boris', shape='quadratic', parent_revision='1' * 40,
                    block_steps=50, block_horizon_omega_p=.5, local_moments_output_dt_omega_p=.5,
                    local_spread_lengths_c_over_wp=[2 * np.sqrt(.001), 4 * np.sqrt(.001)],
                    initial_fingerprints=dict(loading={key: '2' * 64 for key in ('x', 'v')},
                                              state={key: '3' * 64 for key in ('x', 'u', 'w', 'E', 'B', 'rho')}))
    time = np.arange(3) / 2
    rms = np.tile(np.sqrt(.001 / np.array([1, 1836])), (3, 1))
    mean_E = time / 50
    electric = mean_E**2 / 2 + time * 1e-6
    spread = .5 * np.array([1, 1836]) * rms**2
    data = dict(t=time, rms=rms, spread=spread, mean=np.zeros((3, 2)), mean_E=mean_E,
                electric=electric, nonzero_electric=electric - mean_E**2 / 2,
                kinetic=np.full((3, 2), .00051), work=electric, balance=np.full(3, .00102),
                magnetic=np.zeros(3), dark=np.zeros(3), density_rms=np.tile(time[:, None] / 20, (1, 2)),
                local_spread=np.tile(spread[:, :, None], (1, 1, 2)),
                local_density_rms=np.tile(time[:, None, None] / 30, (1, 2, 2)))
    paths = [tmp_path / name for name in ('first', 'second')]
    record = dict(settings=settings, results={}, git='4' * 40, jax='0.6.2', jaxincell='pinned',
                  numpy='2.2.4', jax_enable_x64=True, backend='gpu')
    for path in paths:
        path.mkdir()
        (path / 'run.json').write_text(json.dumps(record))
        np.savez_compressed(path / 'data.npz', **data)
    return paths


def change_settings(path, **changes):
    record = json.loads((path / 'run.json').read_text())
    record['settings'].update(changes)
    (path / 'run.json').write_text(json.dumps(record))


def test_exact_native_repeat_and_portable_fingerprints(records):
    result = compare_replays(*records, windows=((0, 1),))
    assert result['initial_fingerprints']['verified']
    assert result['windows'][0]['observables']['electric']['relative_l2_difference'] == 0
    assert 'local_density_rms' in result['windows'][0]['observables']
    summary = result['windows'][0]['realization_summaries'][0]
    assert summary['work_increment'] == pytest.approx(.000201)
    assert summary['injection_rate_over_wp'] == pytest.approx(.000201 / (.00102 + (.0000505 + .000201) / 3))
    np.testing.assert_array_equal(summary['local_spread_increment_mean'], np.zeros((2, 2)))
    assert str(records[0]) not in json.dumps(result, allow_nan=False)
    assert fingerprint(np.zeros(2)) != fingerprint(np.zeros((1, 2)))
    assert metrics(np.zeros(3), np.ones(3))['relative_l2_difference'] is None


def test_window_reduction_uses_dark_work_sign_and_rejects_a_single_sample(records):
    from docs.scripts.compare_replays import window_summary
    with np.load(records[0] / 'data.npz') as stored:
        data = dict(stored)
    selected = data['t'] >= 0
    first = window_summary(data, selected)
    second = window_summary(data, selected, coupled=True)
    assert first['injection_rate_over_wp'] == -second['injection_rate_over_wp']
    with pytest.raises(ValueError, match='two native samples'):
        window_summary(data, data['t'] == 0)


def test_physical_seed_changes_cannot_be_hidden_in_a_resolution_comparison(records):
    change_settings(records[1], momentum_seed_over_sigma_e=.05, seed_mode=16, seed_phase=0.)
    with pytest.raises(ValueError, match='physical seed'):
        compare_replays(*records, windows=((0, 1),))


def test_particle_loading_repeat_keeps_raw_arrays_and_incomplete_endpoint_scope(records, monkeypatch):
    from docs.scripts import compare_replays as renderer
    loading = records[0].parent / 'loading'
    copytree(records[1], loading)
    change_settings(loading, particles_per_species=206000)
    repeated = records[0].parent / 'loading_repeat'
    copytree(loading, repeated)
    original = renderer.compare_replays
    monkeypatch.setattr(renderer, 'compare_replays',
                        lambda *args, **kwargs: original(*args, **kwargs, windows=((0, 1),)))
    folder = records[0].parent / 'publication'
    renderer.publish(*records, folder, original(*records, windows=((0, 1),)),
                     loading_refined=loading, loading_repeat=repeated)
    with np.load(folder / 'data.npz') as data:
        assert 'loading_t' in data and 'loading_repeat_t' in data and 'refined_t' not in data
        np.testing.assert_array_equal(data['loading_t'], data['loading_repeat_t'])
    result = json.loads((folder / 'run.json').read_text())['results']
    assert result['loading_execution_comparison']['initial_fingerprints']['verified']
    assert not result['loading_repeat_endpoint']['available']


@pytest.mark.parametrize('quantity', ['clock', 'units', 'fingerprints', 'scales', 'blocks'])
def test_corrupted_native_comparison_is_rejected(records, quantity):
    path = records[1]
    if quantity in ('clock', 'units'):
        with np.load(path / 'data.npz') as stored:
            data = dict(stored)
        if quantity == 'clock':
            data['t'][1] += .0001
        else:
            data['spread'] *= 2
        np.savez_compressed(path / 'data.npz', **data)
    else:
        record = json.loads((path / 'run.json').read_text())
        if quantity == 'fingerprints':
            record['settings']['initial_fingerprints']['state']['rho'] = '9' * 64
        elif quantity == 'scales':
            record['settings']['local_spread_lengths_c_over_wp'] = [.1, .2]
        else:
            record['settings']['block_horizon_omega_p'] = .7
        (path / 'run.json').write_text(json.dumps(record))
    with pytest.raises(ValueError):
        compare_replays(*records, windows=((0, 1),))


def test_timestep_loading_and_prepared_state_are_distinct(records):
    change_settings(records[1], dt_omega_p=.005, block_steps=100)
    record = json.loads((records[1] / 'run.json').read_text())
    record['settings']['initial_fingerprints']['state']['x'] = '9' * 64
    (records[1] / 'run.json').write_text(json.dumps(record))
    result = compare_replays(*records, variant='dt', windows=((0, 1),))
    assert result['initial_fingerprints']['matches']['loading']['x']
    assert not result['initial_fingerprints']['matches']['state']['x']


@pytest.mark.parametrize('change', ['potential', 'missing_key'])
def test_repeat_checks_dark_potential_and_state_key_sets(records, change):
    for index, path in enumerate(records):
        record = json.loads((path / 'run.json').read_text())
        state = record['settings']['initial_fingerprints']['state']
        state.update(dark_A='5' * 64, dark_phi='6' * 64)
        if index == 1 and change == 'potential':
            state['dark_phi'] = '7' * 64
        if index == 1 and change == 'missing_key':
            del state['dark_A']
        (path / 'run.json').write_text(json.dumps(record))
    with pytest.raises(ValueError, match='initial.*disagree'):
        compare_replays(*records, windows=((0, 1),))


def test_legacy_fingerprints_remain_explicitly_unverified(records):
    change_settings(records[1], initial_fingerprints=None)
    with pytest.raises(ValueError, match='fingerprints'):
        compare_replays(*records, windows=((0, 1),))
    assert not compare_replays(*records, windows=((0, 1),), legacy=True)['initial_fingerprints']['verified']


def test_manufactured_periodic_projection_has_correct_sign_and_energy():
    """Independent discrete sinusoid derivative includes an incompatible constant."""
    cells, amplitude, mean = 64, .2, .07
    dx = 1 / cells
    x = np.arange(cells) * dx
    E = np.zeros((cells, 3))
    E[:, 0] = 2 + amplitude * np.sin(2 * np.pi * x)
    residual = 2 * amplitude * np.sin(np.pi * dx) / dx * np.cos(2 * np.pi * (x - dx / 2)) + mean
    normalization = dict(epsilon0_F_m=2, charge_density_C_m3=1, field_scale_V_m=1, energy_scale_J_m2=3)
    audit = _projection(E, residual, dx, normalization)
    assert audit['max_correction_over_field_scale'] == pytest.approx(amplitude, abs=2e-14)
    assert audit['mean_residual_over_charge_scale'] == pytest.approx(2 * mean, abs=2e-14)
    assert audit['max_remaining_residual_over_charge_scale'] == pytest.approx(2 * mean, abs=2e-14)
    assert audit['energy_change_over_scale'] == pytest.approx(-2 * amplitude**2 / (4 * 3), abs=2e-14)


def test_manufactured_proca_projection_retains_scalar_potential(tmp_path):
    """The phi constraint contributes to the incompatible mean, but phi is unchanged."""
    cells = 64
    E = np.zeros((cells, 3))
    E[:, 0] = np.sin(2 * np.pi * np.arange(cells) / cells)
    path = tmp_path / 'final_state.npz'
    np.savez_compressed(path, cells=cells, time=1., E=E * 0, rho=np.zeros(cells), **{
        'dark.length': 1., 'dark.background': 0., 'dark.mode': 'field',
        'dark.E': E, 'dark.phi': np.full(cells, .3), 'dark.omega': 2., 'dark.eta': .2})
    before = path.read_bytes()
    record = dict(settings=dict(cells=cells, horizon_omega_p=1., normalization=dict(
        epsilon0_F_m=2, charge_density_C_m3=1, field_scale_V_m=1,
        energy_scale_J_m2=3, c_m_s=4, omega_p_rad_s=1.)))
    audit = constraint_audit(tmp_path, record)
    assert audit['dark']['mean_residual_over_charge_scale'] == pytest.approx(.15)
    assert audit['dark']['unchanged_phi_energy_over_scale'] == pytest.approx(.0075)
    assert audit['dark']['phi_energy_change'] == 0
    assert path.read_bytes() == before


@pytest.fixture
def method_records(records):
    """Independent scalar fixture: electric energy and mean field are distinct."""
    explicit, implicit = records
    record = json.loads((explicit / 'run.json').read_text())
    a = record['settings']
    b = dict(paper_loading=True, newtonian=False, cells=a['cells'],
             particles_per_species=a['particles_per_species'], length_over_c_wp=40,
             mass_ratio=1836, temperature=.001, amplitude=a['force_quiver_over_c'],
             parent_revision=a['parent_revision'], dt=.01, store_every=25, steps=100, actual_horizon=1.,
             initial_fingerprints={key: ('2' if key == 'x' else '3') * 64
                                   for key in ('x', 'u', 'w', 'E', 'B', 'rho', 'time', 'mass', 'charge')})
    with np.load(explicit / 'data.npz') as stored:
        data = dict(stored)
    data['momentum'] = np.tile(np.array([.003, 0., 0.]), (3, 1))
    np.savez_compressed(explicit / 'data.npz', **data)
    time = np.arange(5) / 4
    np.savez_compressed(implicit / 'data.npz', t=time, electric=time / 50, current=time / 100,
                        nonzero_electric=time * 1e-6, work=(time / 50)**2 / 2 + time * 1e-6,
                        balance=np.zeros(5), momentum=np.tile([.003, 0., 0.], (5, 1)))
    record.update(settings=b, results=dict(max_balance_over_nmc2L=1e-15), git='5' * 40)
    (implicit / 'run.json').write_text(json.dumps(record))
    return explicit, implicit


def test_cross_method_native_units_and_momentum_offset(method_records):
    result, _, _, first, second, _ = implicit_comparison(*method_records)
    assert all(result['initial_hash_matches'].values())
    assert result['common_samples'] == 3 and result['implicit_native_samples'] == 5
    np.testing.assert_array_equal(first['mean_E'], second['mean_E'])
    np.testing.assert_array_equal(first['momentum'], np.zeros((3, 3)))
    np.testing.assert_array_equal(second['momentum'], np.zeros((3, 3)))
    np.testing.assert_allclose(second['current'], [0, .005, .01])
    np.testing.assert_allclose(second['work'], [0, .0000505, .000201])
    assert result['implicit_balance_over_peak_recorded_work'] == pytest.approx(1e-15 / .000201)
    assert str(method_records[0]) not in json.dumps(result, allow_nan=False)


def test_iteration_comparison_requires_exact_native_fields_and_keeps_species_arrays(method_records):
    from docs.scripts.compare_replays import publish_iterations
    first = method_records[1]
    with np.load(first / 'data.npz') as stored:
        data = dict(stored)
    for key in ('mean', 'rms', 'kinetic'):
        data[key] = np.ones((len(data['t']), 2))
    np.savez_compressed(first / 'data.npz', **data)
    change_settings(first, iterations=4)
    second, folder = first.parent / 'iterations8', first.parent / 'iteration_evidence'
    copytree(first, second)
    change_settings(second, iterations=8)
    publish_iterations(first, second, folder)
    with np.load(folder / 'data.npz') as result:
        np.testing.assert_array_equal(result['0_rms'], data['rms'])
    record = json.loads((second / 'run.json').read_text())
    record['settings']['initial_fingerprints']['E'] = '9' * 64
    (second / 'run.json').write_text(json.dumps(record))
    with pytest.raises(ValueError, match='initial_fingerprints'):
        publish_iterations(first, second, folder)
    for path in (first, second):
        record = json.loads((path / 'run.json').read_text())
        record['settings']['initial_fingerprints'].pop('E')
        (path / 'run.json').write_text(json.dumps(record))
    with pytest.raises(ValueError, match='complete initial fingerprints'):
        publish_iterations(first, second, folder)


@pytest.mark.parametrize('corrupt', [None, 'native_field', 'particle_weight', 'runtime'])
def test_implicit_mesh_substep_controls_keep_loading_and_reject_hidden_changes(method_records, corrupt):
    from docs.scripts.compare_replays import publish_method_controls
    first = method_records[1]
    with np.load(first / 'data.npz') as stored:
        data = dict(stored)
    for key in ('mean', 'rms', 'kinetic'):
        data[key] = np.ones((len(data['t']), 2))
    data['balance'] = data['t'] * 1e-15
    np.savez_compressed(first / 'data.npz', **data)
    change_settings(first, substeps=2, iterations=4,
                    initial_fingerprints={key: '3' * 64
                                          for key in ('x', 'u', 'w', 'time', 'mass', 'charge', 'E', 'B', 'rho')})
    second, mesh = first.parent / 'substeps4', first.parent / 'mesh2000'
    copytree(first, second)
    copytree(first, mesh)
    change_settings(second, substeps=4)
    record = json.loads((mesh / 'run.json').read_text())
    record['settings']['cells'] = 2000
    record['settings']['initial_fingerprints']['E'] = '7' * 64
    (mesh / 'run.json').write_text(json.dumps(record))
    if corrupt:
        target = second if corrupt == 'native_field' else mesh
        record = json.loads((target / 'run.json').read_text())
        if corrupt == 'runtime':
            record['jax'] = 'different'
        else:
            record['settings']['initial_fingerprints']['E' if corrupt == 'native_field' else 'w'] = '8' * 64
        (target / 'run.json').write_text(json.dumps(record))
        with pytest.raises(ValueError):
            publish_method_controls(first, second, mesh, first.parent / 'evidence')
    else:
        folder = first.parent / 'evidence'
        finer = first.parent / 'mesh4000'
        copytree(mesh, finer)
        change_settings(finer, cells=4000)
        publish_method_controls(first, second, mesh, folder, finer)
        result = json.loads((folder / 'run.json').read_text())['results']
        np.testing.assert_array_equal(result['mesh_observables']['kinetic']['relative_l2_difference'], [0, 0])
        assert len(result['native_runs']) == 4 and len(result['native_run_sha256']) == 4
        with np.load(folder / 'data.npz') as stored:
            np.testing.assert_array_equal(stored['3_rms'], data['rms'])


@pytest.mark.parametrize('corrupt', [None, 'missing_phase', 'weights', 'magnetic', 'clock'])
def test_grid_phase_controls_retain_native_arrays_and_reject_unmatched_controls(method_records, corrupt):
    from docs.scripts.compare_replays import publish_phases
    template = method_records[1]
    with np.load(template / 'data.npz') as stored:
        data = dict(stored)
    for key in ('mean', 'rms', 'kinetic'):
        data[key] = np.ones((len(data['t']), 2))
    data['execution_1_nonzero_electric'] = data['nonzero_electric'].copy()
    np.savez_compressed(template / 'data.npz', **data)
    change_settings(template, grid_phase=0.,
                    initial_fingerprints={key: '3' * 64 for key in
                                          ('x', 'u', 'w', 'time', 'mass', 'charge', 'E', 'B', 'rho')})
    folders = []
    for mesh in (1000, 2000):
        for phase in (0., .25):
            path = template.parent / f'phase_{mesh}_{phase}'
            copytree(template, path)
            record = json.loads((path / 'run.json').read_text())
            record['settings'].update(cells=mesh, grid_phase=phase)
            hashes = record['settings']['initial_fingerprints']
            if phase:
                hashes['x'] = '4' * 64
            if mesh == 2000 or phase:
                hashes['E'] = hashes['rho'] = '5' * 64
            if corrupt in ('weights', 'magnetic') and mesh == 2000 and phase:
                hashes['w' if corrupt == 'weights' else 'B'] = '6' * 64
            (path / 'run.json').write_text(json.dumps(record))
            folders.append(path)
    if corrupt == 'missing_phase':
        folders.pop()
    if corrupt == 'clock':
        change_settings(folders[-1], dt=.2)
    folder = template.parent / 'phase_evidence'
    if corrupt:
        with pytest.raises(ValueError):
            publish_phases(folders, folder)
    else:
        publish_phases(folders, folder)
        record = json.loads((folder / 'run.json').read_text())['results']
        assert len(record['phase_observables']) == 2 and len(record['native_runs']) == 4
        with np.load(folder / 'data.npz') as stored:
            assert '3_momentum' in stored and '0_execution_1_nonzero_electric' in stored


@pytest.mark.parametrize('change', ['clock', 'truncated', 'horizon', 'initial', 'amplitude', 'runtime',
                                    'finite', 'shape', 'execution'])
def test_cross_method_mismatches_are_rejected(method_records, change):
    path = method_records[1]
    if change in ('clock', 'truncated', 'finite', 'shape', 'execution'):
        with np.load(path / 'data.npz') as stored:
            data = dict(stored)
        data['t'][1] += .001 if change == 'clock' else 0
        if change == 'finite':
            data['electric'][2] = np.nan
        if change == 'shape':
            data['electric'] = data['electric'][:, None]
        if change == 'execution':
            data['execution_0_mean_E'] = np.zeros((5, 1))
        if change == 'truncated':
            data = {key: value[:-1] for key, value in data.items()}
        np.savez_compressed(path / 'data.npz', **data)
    else:
        record = json.loads((path / 'run.json').read_text())
        if change == 'initial':
            record['settings']['initial_fingerprints']['u'] = '8' * 64
        elif change == 'runtime':
            record['jax'] = 'different'
        elif change == 'horizon':
            record['settings']['actual_horizon'] = .75
        else:
            record['settings']['amplitude'] *= 2
        (path / 'run.json').write_text(json.dumps(record))
    with pytest.raises(ValueError):
        implicit_comparison(*method_records)


@pytest.mark.parametrize('corrupt', [None, 'x', 'w'])
def test_seed_control_retains_physical_positions_and_weights(records, corrupt):
    path = records[1]
    record = json.loads((path / 'run.json').read_text())
    settings = record['settings']
    settings['seed'] = 1
    settings['initial_fingerprints']['loading']['v'] = '7' * 64
    settings['initial_fingerprints']['state']['u'] = '8' * 64
    if corrupt:
        part = 'loading' if corrupt == 'x' else 'state'
        settings['initial_fingerprints'][part][corrupt] = '9' * 64
    (path / 'run.json').write_text(json.dumps(record))
    if corrupt:
        with pytest.raises(ValueError, match='initial fingerprints disagree'):
            compare_replays(*records, variant='seed', windows=((0, 1),))
    else:
        result = compare_replays(*records, variant='seed', windows=((0, 1),))
        assert result['initial_fingerprints']['matches']['loading']['x']
        assert not result['initial_fingerprints']['matches']['loading']['v']
