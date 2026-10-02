"""Native-record guards reject mismatched clocks, units and controlled loading."""
import hashlib
import json
from pathlib import Path
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
                  numpy='2.2.4', python='3.10.12', platform='Linux x86_64', jax_enable_x64=True, backend='gpu')
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
    change_settings(records[1], momentum_seed_over_sigma_e=.05, seed_mode=16, seed_phase=0.,
                    thermal_temperature_over_mec2=[.001, .001],
                    initial_rms_over_c=np.sqrt(np.array([.001, .001 / 1836])).tolist())
    with pytest.raises(ValueError, match='physical seed'):
        compare_replays(*records, windows=((0, 1),))


@pytest.mark.parametrize('variant', ['repeat', 'dt', 'mesh', 'loading', 'seed', 'shape', 'resolution'])
def test_gather_changes_cannot_be_hidden_in_a_resolution_comparison(records, variant):
    change_settings(records[1], longitudinal_gather='six_face')
    with pytest.raises(ValueError, match='longitudinal_gather'):
        compare_replays(*records, variant=variant, windows=((0, 1),))


def test_joint_resolution_records_new_count_and_unmatched_microstate(records):
    fine = json.loads((records[1] / 'run.json').read_text())['settings']['initial_fingerprints']
    fine['loading'] = {key: '5' * 64 for key in ('x', 'v')}
    fine['state'] = {key: '6' * 64 for key in fine['state']}
    change_settings(records[1], cells=2000, particles_per_species=412000, initial_fingerprints=fine)
    result = compare_replays(*records, variant='resolution', windows=((0, 1),))
    assert result['initial_fingerprints']['verified']
    assert not any(result['initial_fingerprints']['matches']['loading'].values())
    assert not any(result['initial_fingerprints']['matches']['state'].values())
    assert result['sources'][1]['varied_parameter'] == dict(cells=2000, particles_per_species=412000)
    assert 'changes the particle microstate' in result['notes']
    assert 'not the full SHARP' in result['notes']


@pytest.mark.parametrize('changed', [dict(cells=2000), dict(particles_per_species=412000),
                                     dict(cells=2000, particles_per_species=206000),
                                     dict(cells=3000, particles_per_species=412000),
                                     dict(cells=2000., particles_per_species=412000),
                                     dict(cells=2000, particles_per_species=412000.)])
def test_joint_resolution_rejects_wrong_noise_scaling(records, changed):
    change_settings(records[1], **changed)
    with pytest.raises(ValueError, match='twice the cells and four times the particles'):
        compare_replays(*records, variant='resolution', windows=((0, 1),))


@pytest.mark.parametrize('key', ['python', 'platform'])
def test_joint_resolution_requires_matching_recorded_runtime(records, key):
    change_settings(records[1], cells=2000, particles_per_species=412000)
    record = json.loads((records[1] / 'run.json').read_text())
    record[key] = 'changed'
    (records[1] / 'run.json').write_text(json.dumps(record))
    with pytest.raises(ValueError, match=key):
        compare_replays(*records, variant='resolution', windows=((0, 1),))


@pytest.mark.parametrize('side', [0, 1])
def test_matching_top_source_cannot_hide_a_continued_prefix(records, side):
    change_settings(records[side], continuation={'prefix_native_git': '9' * 40})
    with pytest.raises(ValueError, match='mixed-producer continuation'):
        compare_replays(*records, windows=((0, 1),))


@pytest.fixture
def continued_records(tmp_path):
    """NumPy-only native bookkeeping fixture; no simulated physics or frame replicates."""
    import hashlib
    folders, donors = [], []
    for arm, dtau in enumerate((.5, .25)):
        prefix, folder = [tmp_path / f'{arm}_{name}' for name in ('prefix', 'continued')]
        prefix.mkdir()
        folder.mkdir()
        wp, c, mass, density, length = 2., 10., 2., 2., 200.
        energy, field = density * mass * c**2 * length, mass * c * wp
        sigma = np.sqrt(.001 / np.array([1., 1836])) * c
        velocity = np.repeat(sigma, 2) * np.tile([1., -1.], 2)
        u = np.zeros((4, 3))
        u[:, 0] = velocity / np.sqrt(1 - (velocity / c)**2)
        w = np.full(4, density * length / 2)
        x = np.zeros((4, 3))
        x[:, 0] = np.tile([50., 150.], 2) + dtau / wp * velocity / 2
        kinetic = .5 * (1 / np.sqrt(1 - (sigma / c)**2) - 1) * np.array([1., 1836])
        # Two markers each represent nL/2; K/(n me c²L) has no factor 1/2.
        kinetic *= 2
        state = dict(format=np.array(2), E=np.zeros((2, 3)), B=np.zeros((2, 3)), x=x, u=u, w=w,
                     qm=np.repeat([-1 / mass, 1 / (1836 * mass)], 2), rho=np.zeros(2), sigma=np.zeros(2),
                     key=np.array([0, 0], dtype=np.uint32), time=np.array(0.), steps=np.array(0, dtype=np.int32),
                     names=np.array(['electrons', 'ions']), counts=np.array([2, 2]), cells=np.array(2),
                     algorithm=np.array('explicit'), shape_order=np.array(5))
        for key in ('arrived', 'collected', 'injected', 'energy_in', 'energy_out', 'energy_injected',
                    'momentum', 'momentum_injected', 'truncated', 'overflow'):
            state['wall.' + key] = np.zeros((2, 2, 3) if 'momentum' in key else () if key == 'overflow' else (2, 2))
        state.update({'dark.' + key: np.asarray(value) for key, value in dict(
            format=2, mode='drive', omega=wp, eta=1., cells=2, length=length, length_y=1., length_z=1.,
            dt=dtau / wp, relativistic=True, mass=[mass, 1836 * mass], charge=[-1., 1.], density=[density, density],
            has_external_B=False, amplitude=[.03 * np.sqrt(.001) * field, 0., 0.], phase=0., background=0., work=0.,
            initial_ordinary=energy * kinetic.sum(), initial_dark=0., initial_projection_norm=0.,
            max_balance_error=.0001 * energy, max_ordinary_gauss=0., max_dark_gauss=0.).items()})
        hashes = {key: hashlib.sha256(f'{state[key].dtype.str}:{state[key].shape}'.encode()
                                      + state[key].tobytes()).hexdigest() for key in ('x', 'u', 'w', 'E', 'B', 'rho')}
        setting = dict(cells=2, particles_per_species=2, seed=0, dt_omega_p=dtau, output_dt_omega_p=.5,
                       horizon_omega_p=1000., length_c_over_omega_p=40., mass_ratio=1836, T_each_over_mec2=.001,
                       coupling=None, drive_quiver_over_sigma=.03, force_quiver_over_c=.03 * np.sqrt(.001),
                       loading='conditioned Gaussian', pusher='relativistic Boris', shape='quintic', shape_order=5,
                       parent_revision='1' * 40, block_steps=round(100 / dtau), block_horizon_omega_p=100.,
                       local_moments_output_dt_omega_p=.5, local_spread_lengths_c_over_wp=(
                           np.array([2., 4.]) * np.sqrt(.001)).tolist(), recorded_mode=1, XLA_FLAGS='',
                       normalization=dict(omega_p_rad_s=wp, c_m_s=c, epsilon0_F_m=.25,
                                          energy_scale_J_m2=energy, field_scale_V_m=field, charge_density_C_m3=density),
                       initial_fingerprints=dict(loading={key: '2' * 64 for key in ('x', 'v')}, state=hashes))
        from docs.scripts.compare_replays import PAPER_MAXIMA
        result = dict.fromkeys(PAPER_MAXIMA, 0.)
        result[PAPER_MAXIMA[0]] = .1
        runtime = dict(jax='0.6.2', jaxincell='pinned', numpy='2.2.4', python='3.10.12',
                       platform='Linux x86_64', backend='gpu', jax_enable_x64=True)
        record = dict(runtime, git='4' * 40, example='paper_resonant_conversion', settings=setting, results=result)
        data = dict(t=np.arange(10001) / 2, mean=np.zeros((10001, 2)), rms=np.tile(sigma / c, (10001, 1)),
                    spread=np.full((10001, 2), .0005), kinetic=np.tile(kinetic, (10001, 1)),
                    balance=np.full(10001, kinetic.sum()), momentum=np.zeros((10001, 3)),
                    local_spread=np.full((10001, 2, 2), .0005), local_density_rms=np.zeros((10001, 2, 2)),
                    density_rms=np.zeros((10001, 2)))
        data.update({key: np.zeros(10001) for key in ('electric', 'nonzero_electric', 'magnetic', 'dark',
                                                      'work', 'charge', 'grid_charge', 'mean_E', 'mean_D', 'mean_A',
                                                      'ordinary_gauss', 'dark_gauss', 'dark_coherent')})
        data.update(max_speed=np.full(10001, sigma.max() / c), mode_E=np.zeros(10001, dtype=complex),
                    dark_mode_E=np.zeros(10001, dtype=complex))
        np.savez_compressed(prefix / 'initial_state.npz', **state)
        endpoint = {**state, 'steps': np.array(round(1000 / dtau), dtype=np.int32), 'time': np.array(1000 / wp)}
        np.savez_compressed(prefix / 'final_state.npz', **endpoint)
        np.savez_compressed(prefix / 'data.npz', **{key: value[:2001] for key, value in data.items()})
        (prefix / 'run.json').write_text(json.dumps(record))
        for old, new in (('initial_state.npz', 'initial_state.npz'), ('final_state.npz', 'segment_initial_state.npz'),
                         ('run.json', 'prefix_run.json'), ('data.npz', 'prefix_data.npz')):
            (folder / new).write_bytes((prefix / old).read_bytes())
        np.savez_compressed(folder / 'final_state.npz', **{
            **state, 'steps': np.array(round(5000 / dtau), dtype=np.int32), 'time': np.array(5000 / wp)})
        si = {key: value[2000:].copy() for key, value in data.items() if key != 'nonzero_electric'}
        si['t'] /= wp
        for key in ('electric', 'magnetic', 'dark', 'dark_coherent', 'work', 'spread', 'kinetic', 'balance',
                    'local_spread'):
            si[key] *= energy
        for key in ('mean', 'rms', 'max_speed'):
            si[key] *= c
        si['momentum'] *= energy / c
        si['mean_E'] *= field
        si['mode_E'] *= field
        si['dark_mode_E'] *= field
        np.savez_compressed(folder / 'segment_data.npz', **si)
        np.savez_compressed(folder / 'data.npz', **data)

        def sha(name):
            return hashlib.sha256((prefix / name).read_bytes()).hexdigest()
        lineage = dict(prefix_native_git='4' * 40, prefix_record_sha256=sha('run.json'),
                       prefix_data_sha256=sha('data.npz'), prefix_lineage=None,
                       origin_archive_sha256=sha('initial_state.npz'), checkpoint_archive_sha256=sha('final_state.npz'),
                       producer_script_sha256='6' * 64, start_step=round(1000 / dtau), end_step=round(5000 / dtau),
                       start_time_omega_p=1000., end_time_omega_p=5000., target_time_omega_p=5000.,
                       segment_steps=round(4000 / dtau))
        maximum = np.zeros(9)
        maximum[0] = np.nextafter(.0001 * energy, np.inf)
        record = {**record, 'git': '5' * 40, 'example': 'paper_resonant_continuation',
                  'settings': {**setting, 'horizon_omega_p': 5000., 'continuation': lineage},
                  'results': {**result, 'all_step_maxima_SI': maximum.tolist(),
                              'reconstructed_prior_maxima_SI': maximum.tolist()}}
        (folder / 'run.json').write_text(json.dumps(record))
        folders.append(folder)
        donors.append([prefix])
    return folders, donors, [('4' * 40, '5' * 40)]


def test_continuation_preserves_producers_raw_windows_and_global_bounds(continued_records):
    from docs.scripts.compare_replays import compare_continuations
    folders, donors, transitions = continued_records
    report = compare_continuations(*folders, donors, transitions)
    assert report['original_comparison']['sources'][0]['git'] == '4' * 40
    assert report['lineages'][0][0]['record']['git'] == '5' * 40
    assert [row['samples'] for row in report['windows']] == [401, 401]
    assert report['windows'][1]['realization_summaries'][0]['work_increment'] == 0
    assert report['retained_gates']['work'] == .02
    assert report['windows'][1]['accounting']['conservation_transfer_gate'] == [False, False]
    assert report['windows'][1]['accounting']['global_defect_over_abs_window_work'] == [None, None]
    assert report['source_vectors'][0] == ['4' * 40, '5' * 40]
    assert report['lineages'][0][0]['duplicated_scalar_boundary']['bitwise_equal']
    assert not report['lineages'][0][0]['all_step_bounds']['segment_only_maxima_available']
    assert str(folders[0]) not in json.dumps(report, allow_nan=False)
    with pytest.raises(ValueError, match='mixed-producer'):
        compare_replays(*folders, variant='dt')


@pytest.mark.parametrize('corrupt', ['prefix', 'origin', 'boundary', 'endpoint', 'units', 'join', 'lineage',
                                     'source', 'runtime', 'phase', 'maximum', 'clock', 'background',
                                     'initial_energy', 'qm', 'x64', 'prior', 'script_hash', 'gather'])
def test_continuation_rejects_corrupt_native_lineage(continued_records, corrupt):
    from docs.scripts.compare_replays import compare_continuations
    folders, donors, transitions = continued_records
    folder = folders[0]
    files = dict(origin='initial_state.npz', boundary='segment_initial_state.npz', endpoint='final_state.npz',
                 units='segment_data.npz', join='data.npz', phase='final_state.npz', clock='final_state.npz',
                 background='final_state.npz', initial_energy='final_state.npz', qm='final_state.npz')
    keys = dict(origin='key', boundary='key', endpoint='u', units='kinetic', join='work',
                phase='dark.phase', clock='time', background='dark.background',
                initial_energy='dark.initial_ordinary', qm='qm')
    if corrupt in files:
        path = folder / files[corrupt]
        with np.load(path) as stored:
            arrays = dict(stored)
        arrays[keys[corrupt]] = arrays[keys[corrupt]] + 1
        np.savez_compressed(path, **arrays)
    elif corrupt == 'prefix':
        with (folder / 'prefix_run.json').open('a') as stream:
            stream.write('\n')
    else:
        path = folder / 'run.json'
        record = json.loads(path.read_text())
        if corrupt == 'lineage':
            record['settings']['continuation']['prefix_lineage'] = {}
        elif corrupt == 'maximum':
            record['results']['all_step_maxima_SI'][0] = 0
        elif corrupt == 'prior':
            record['results']['reconstructed_prior_maxima_SI'][0] = 0
        elif corrupt == 'x64':
            record['jax_enable_x64'] = False
        elif corrupt == 'script_hash':
            record['settings']['continuation']['producer_script_sha256'] = 'unverified'
        elif corrupt == 'gather':
            record['settings']['longitudinal_gather'] = 'six_face'
        else:
            record['git' if corrupt == 'source' else 'jax'] = '9' * 40
        path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match='longitudinal_gather' if corrupt == 'gather' else None):
        compare_continuations(*folders, donors, transitions)


def test_continuation_optional_gather_format_matches_record_and_archive(continued_records):
    from docs.scripts.compare_replays import _archive, _continuation_state_checks, _load
    folder = continued_records[0][0]
    record, data, _, _ = _load(folder, 1e-5)
    record['settings']['longitudinal_gather'] = 'six_face'
    states = [_archive(folder / name) for name in ('initial_state.npz', 'final_state.npz')]
    for state in states:
        state.update({'dark.format': np.asarray(4), 'dark.longitudinal_gather': np.asarray('six_face')})
    _continuation_state_checks(states, record, data)
    for key, value in (('dark.format', np.asarray(2)), ('dark.longitudinal_gather', np.asarray('average')),
                       ('dark.longitudinal_gather', None), ('shape_order', np.asarray(2))):
        broken = [{name: array for name, array in state.items() if name != key} for state in states]
        if value is not None:
            for state in broken:
                state[key] = value
        with pytest.raises(ValueError, match='native model'):
            _continuation_state_checks(broken, record, data)
    record['settings']['longitudinal_gather'] = 'unknown'
    with pytest.raises(ValueError, match='longitudinal_gather'):
        _continuation_state_checks(states, record, data)


def test_continuation_keeps_exact_join_with_reported_recompiled_boundary_roundoff(continued_records):
    from docs.scripts.compare_replays import compare_continuations
    folders, donors, transitions = continued_records
    filename = folders[0] / 'segment_data.npz'
    with np.load(filename) as stored:
        data = dict(stored)
    data['kinetic'][0, 0] = np.nextafter(data['kinetic'][0, 0], np.inf)
    np.savez_compressed(filename, **data)
    result = compare_continuations(*folders, donors, transitions)
    boundary = result['lineages'][0][0]['duplicated_scalar_boundary']
    assert not boundary['bitwise_equal'] and boundary['maximum_difference_over_unit']['kinetic'] > 0
    assert boundary['rtol'] == boundary['atol_over_unit'] == 2e-12


def test_continuation_validates_nested_original_and_segment_sources(continued_records):
    import hashlib
    from docs.scripts.compare_replays import compare_continuations
    folders, donors, transitions = continued_records
    final_folders = []
    for folder in folders:
        child = folder.parent / (folder.name + '_next')
        child.mkdir()
        old = json.loads((folder / 'run.json').read_text())
        setting, lineage = old['settings'], old['settings']['continuation']
        with np.load(folder / 'final_state.npz') as stored:
            state = dict(stored)
        with np.load(folder / 'data.npz') as stored:
            data = dict(stored)
        with np.load(folder / 'segment_data.npz') as stored:
            raw = {key: np.repeat(stored[key][-1:], 401, axis=0) for key in stored.files}
        raw['t'] = (5000 + np.arange(401) / 2) / setting['normalization']['omega_p_rad_s']
        joined = {key: np.concatenate((value, np.repeat(value[-1:], 400, axis=0))) for key, value in data.items()}
        joined['t'][-400:] = raw['t'][1:] * setting['normalization']['omega_p_rad_s']
        inputs = dict(prefix_record='run.json', prefix_data='data.npz', origin_archive='initial_state.npz',
                      checkpoint_archive='final_state.npz')
        hashes = {key + '_sha256': hashlib.sha256((folder / name).read_bytes()).hexdigest()
                  for key, name in inputs.items()}
        next_lineage = dict(lineage, **hashes, prefix_native_git=old['git'], prefix_lineage=lineage,
                            start_step=int(state['steps']), end_step=round(5200 / setting['dt_omega_p']),
                            segment_steps=round(200 / setting['dt_omega_p']), start_time_omega_p=5000.,
                            end_time_omega_p=5200., target_time_omega_p=5200.)
        copies = (('initial_state.npz', 'initial_state.npz'), ('final_state.npz', 'segment_initial_state.npz'),
                  ('run.json', 'prefix_run.json'), ('data.npz', 'prefix_data.npz'))
        for source, target in copies:
            (child / target).write_bytes((folder / source).read_bytes())
        state.update(steps=np.asarray(next_lineage['end_step'], dtype=np.int32), time=np.asarray(raw['t'][-1]))
        np.savez_compressed(child / 'final_state.npz', **state)
        np.savez_compressed(child / 'segment_data.npz', **raw)
        np.savez_compressed(child / 'data.npz', **joined)
        record = {**old, 'git': '7' * 40,
                  'settings': {**setting, 'horizon_omega_p': 5200., 'continuation': next_lineage}}
        (child / 'run.json').write_text(json.dumps(record))
        final_folders.append(child)
    result = compare_continuations(*final_folders, [paths + [folder] for paths, folder in zip(donors, folders)],
                                   transitions + [('5' * 40, '7' * 40)])
    assert result['source_vectors'] == [['4' * 40, '5' * 40, '7' * 40]] * 2
    assert len(result['lineages'][0]) == 2 and result['windows'][1]['samples'] == 401
    with pytest.raises(ValueError, match='source vector'):
        compare_continuations(*final_folders, [paths + [folder] for paths, folder in zip(donors, folders)], transitions)


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


def test_refined_shape_run_requires_its_matching_anchor(records, monkeypatch):
    import hashlib
    from docs.scripts import compare_replays as renderer
    change_settings(records[1], shape='quintic', shape_order=5)
    refined = records[0].parent / 'refined'
    copytree(records[1], refined)
    change_settings(refined, dt_omega_p=.005, block_steps=100)
    original = renderer.compare_replays
    monkeypatch.setattr(renderer, 'compare_replays',
                        lambda *args, **kwargs: original(*args, **kwargs, windows=((0, 1),)))
    comparison = original(*records, variant='shape', windows=((0, 1),))
    folder = records[0].parent / 'publication'
    with pytest.raises(ValueError, match='share shape'):
        renderer.publish(*records, folder, comparison, refined=refined)
    with pytest.raises(ValueError, match='first or second'):
        renderer.publish(*records, folder, comparison, refined=refined, refined_against='third')
    assert not folder.exists()
    renderer.publish(*records, folder, comparison, refined=refined, refined_against='second')
    result = json.loads((folder / 'run.json').read_text())['results']
    assert result['refined_against'] == 'second'
    refinement = result['refinement_comparison']
    assert refinement['variant'] == 'dt' and refinement['initial_fingerprints']['verified']
    for source, path in zip(refinement['sources'], (records[1], refined)):
        assert source['run_sha256'] == hashlib.sha256((path / 'run.json').read_bytes()).hexdigest()
        assert source['data_sha256'] == hashlib.sha256((path / 'data.npz').read_bytes()).hexdigest()
    with np.load(folder / 'data.npz') as data, np.load(refined / 'data.npz') as native:
        np.testing.assert_array_equal(data['refined_electric'], native['electric'])
    labels = [renderer._replay_label(json.loads((path / 'run.json').read_text())['settings'], 'shape', 0)
              for path in (records[1], refined)]
    assert labels[0] != labels[1] and all('degree 5' in label for label in labels)


def test_third_timestep_preserves_shape_comparison_and_raw_contraction(records, monkeypatch):
    import hashlib
    from docs.scripts import compare_replays as renderer
    change_settings(records[1], shape='quintic', shape_order=5)
    refined, finer = [records[0].parent / name for name in ('refined', 'finer')]
    for path, dt, blocks in ((refined, .005, 100), (finer, .0025, 200)):
        copytree(records[1], path)
        change_settings(path, dt_omega_p=dt, block_steps=blocks)
    for path in (records[1], refined, finer):
        dt = json.loads((path / 'run.json').read_text())['settings']['dt_omega_p']
        with np.load(path / 'data.npz') as archive:
            data = dict(archive)
        data['mean_E'] *= 1 + dt**2
        data['electric'] = .5 * data['mean_E']**2 + data['nonzero_electric']
        data['work'] = data['electric'].copy()
        np.savez_compressed(path / 'data.npz', **data)
    original = renderer.compare_replays
    monkeypatch.setattr(renderer, 'compare_replays',
                        lambda *args, **kwargs: original(*args, **kwargs, windows=((0, 1),)))
    comparison = original(*records, variant='shape', windows=((0, 1),))
    folder = records[0].parent / 'publication'
    renderer.publish(*records, folder, comparison, refined=refined, refined_against='second', finer_step=finer)
    result = json.loads((folder / 'run.json').read_text())['results']
    assert result['comparison'] == comparison and result['refined_against'] == 'second'
    assert result['refinement_comparison'] == original(records[1], refined, 'dt', windows=((0, 1),))
    assert result['third_step_comparison'] == original(refined, finer, 'dt', windows=((0, 1),))
    contraction = result['norm_contraction'][0]
    assert contraction['mean_E']['coarse_to_fine_contraction'] == pytest.approx(4., rel=2e-11)
    assert contraction['nonzero_electric']['coarse_to_fine_contraction'] is None
    assert 'do not certify temporal order' in result['norm_contraction_note']
    assert len(result['native_runs']) == 4
    assert result['third_step_comparison']['sources'][1]['data_sha256'] == hashlib.sha256(
        (finer / 'data.npz').read_bytes()).hexdigest()
    with np.load(folder / 'data.npz') as data, np.load(finer / 'data.npz') as native:
        np.testing.assert_array_equal(data['finer_electric'], native['electric'])
        assert {'first_t', 'second_t', 'refined_t', 'finer_t'} <= set(data.files)


@pytest.mark.parametrize('corrupt', ['physics', 'clock', 'source', 'missing_refined', 'variant', 'loading', 'repeat'])
def test_third_timestep_rejects_confounded_publication(records, corrupt):
    from docs.scripts import compare_replays as renderer
    refined, finer = [records[0].parent / name for name in ('refined', 'finer')]
    for path, dt, blocks in ((refined, .005, 100), (finer, .0025, 200)):
        copytree(records[1], path)
        change_settings(path, dt_omega_p=dt, block_steps=blocks)
    kwargs = dict(refined=refined, finer_step=finer)
    if corrupt == 'physics':
        change_settings(finer, force_quiver_over_c=.03)
    elif corrupt == 'source':
        record = json.loads((finer / 'run.json').read_text())
        record['git'] = '9' * 40
        (finer / 'run.json').write_text(json.dumps(record))
    elif corrupt == 'clock':
        with np.load(finer / 'data.npz') as archive:
            data = dict(archive)
        data['t'][-1] += .1
        np.savez_compressed(finer / 'data.npz', **data)
    else:
        kwargs.update({'missing_refined': dict(refined=None), 'variant': dict(refined_variant='shape'),
                       'loading': dict(loading_refined=records[1]), 'repeat': dict(loading_repeat=records[1])}[corrupt])
    folder = records[0].parent / 'publication'
    with pytest.raises(ValueError):
        renderer.publish(*records, folder, {}, **kwargs)
    assert not folder.exists()


def test_contraction_rejects_unequal_timestep_ratios():
    from docs.scripts.compare_replays import _step_contraction
    first = dict(variant='dt', sources=[dict(varied_parameter=.01), dict(varied_parameter=.005)])
    second = dict(variant='dt', sources=[dict(varied_parameter=.005), dict(varied_parameter=.002)])
    with pytest.raises(ValueError, match='consecutive timestep halvings'):
        _step_contraction(first, second)


@pytest.fixture
def seed_pairs(records):
    pairs = []
    for seed in range(5):
        pair = [records[0].parent / f'seed{seed}_{level}' for level in ('coarse', 'fine')]
        for index, path in enumerate(pair):
            copytree(records[0], path)
            change_settings(path, seed=seed, dt_omega_p=.005 if index else .01, block_steps=100 if index else 50)
            record = json.loads((path / 'run.json').read_text())
            record['settings']['initial_fingerprints']['loading']['v'] = f'{seed + 5:x}' * 64
            (path / 'run.json').write_text(json.dumps(record))
        pairs.append(pair)
    return pairs


def test_ensemble_reduces_runs_before_seeds_and_retains_native_sources(seed_pairs):
    import runpy
    from docs.scripts.compare_replays import ensemble_comparison
    pilot = ensemble_comparison(seed_pairs[:3], window=(0, 1))
    assert pilot['status'] == 'exploratory pilot' and pilot['independent_units'] == 'seed pairs'
    assert pilot['seeds'] == [0, 1, 2] and not pilot['frames_are_replicates']
    assert len(pilot['raw_per_seed']) == 3
    row = pilot['raw_per_seed'][0]
    assert row['native_comparison']['windows'][0]['samples'] == 3
    assert row['reductions'][0]['work_increment'] == pytest.approx(.000201)
    assert row['native_comparison']['sources'][0]['git'] == '4' * 40
    assert pilot['marginal']['critical_t'] == pytest.approx(4.302652729696142)
    assert pilot['marginal']['observables']['electric']['coarse_seed_sd'] == 0
    assert not pilot['marginal']['observables']['electron_local_2D']['equivalent']
    assert pilot['marginal']['observables']['injection_rate']['equivalent']
    assert str(seed_pairs[0][0]) not in json.dumps(pilot, allow_nan=False)
    final = ensemble_comparison(seed_pairs, window=(0, 1))
    assert final['status'] == 'five-seed reduction' and 'not independently held out' in final['notes']
    assert final['simultaneous']['critical_t'] > final['marginal']['critical_t']
    output = seed_pairs[0][0].parent / 'ensemble.json'
    runpy.run_path('docs/scripts/compare_replays.py', run_name='__main__',
                   init_globals=dict(ensemble=seed_pairs[:3], ensemble_window=(0, 1), output=output))
    assert json.loads(output.read_text()) == pilot


def test_fieller_matches_independent_quadratic_and_zero_uncertainty():
    from scipy.stats import t
    from docs.scripts.compare_replays import _ensemble_intervals, _fieller
    x = np.array([1., 1.2, .9, 1.1, .8])
    y = np.array([1.02, 1.18, .94, 1.09, .81])
    critical = t.ppf(.975, len(x) - 1)
    variance = np.cov(x, y, ddof=1) / len(x)
    A = x.mean()**2 - critical**2 * variance[0, 0]
    B = x.mean() * y.mean() - critical**2 * variance[0, 1]
    C = y.mean()**2 - critical**2 * variance[1, 1]
    expected = (B + np.array([-1, 1]) * np.sqrt(B**2 - A * C)) / A - 1
    np.testing.assert_allclose(_fieller(x, y, critical), expected, rtol=2e-12, atol=1e-14)
    varying = _ensemble_intervals(np.tile(x[:, None], (1, 8)), np.tile(y[:, None], (1, 8)), False)
    delta = y - x
    half = critical * delta.std(ddof=1) / np.sqrt(len(x))
    np.testing.assert_allclose(varying['observables']['injection_rate']['additive_interval'],
                               [delta.mean() - half, delta.mean() + half])
    a = np.ones((3, 8))
    b = a.copy()
    b[:, :-1] *= 1.005
    b[:, -1] += 2e-6
    report = _ensemble_intervals(a, b, True)
    assert report['all_equivalent']
    np.testing.assert_allclose(report['observables']['work']['fieller_relative_interval'], [.005, .005], atol=1e-14)
    np.testing.assert_allclose(report['observables']['injection_rate']['additive_interval'], [2e-6, 2e-6])
    for reference in (np.zeros(3), np.array([.1, .2, -.1])):
        assert _fieller(reference, np.ones(3), t.ppf(.975, 2)) is None


@pytest.mark.parametrize('scale', [1e-100, 1., 1e100])
def test_fieller_is_scale_invariant_and_does_not_hide_uncertainty(scale):
    from scipy.stats import t
    from docs.scripts.compare_replays import _ensemble_intervals, _fieller
    a = np.ones(5)
    b = a + np.array([-.1, -.05, 0, .05, .1])
    critical = t.ppf(.975, 4)
    expected = critical * np.std(b - a, ddof=1) / np.sqrt(5)
    np.testing.assert_allclose(_fieller(a * scale, b * scale, critical), [-expected, expected], rtol=1e-14)
    rows = _ensemble_intervals(np.tile(a[:, None], (1, 8)) * scale,
                               np.tile(b[:, None], (1, 8)) * scale, False)
    assert not rows['observables']['work']['equivalent']


@pytest.mark.parametrize('energy', [0., -1.])
def test_window_rate_requires_positive_plasma_energy(energy):
    from docs.scripts.compare_replays import window_summary
    data = dict(t=np.array([0., 1.]), work=np.array([0., 1.]), electric=np.zeros(2),
                magnetic=np.zeros(2), kinetic=np.full((2, 2), energy))
    with pytest.raises(ValueError, match='positive finite plasma energy'):
        window_summary(data, np.array([True, True]))


@pytest.mark.parametrize('change', [
    'missing_seed', 'wrong_seed', 'duplicate_loading', 'source', 'runtime', 'cadence', 'physics', 'window',
    'moment_lengths', 'reversed_lengths'])
def test_ensemble_rejects_confounded_or_incomplete_inputs(seed_pairs, change):  # noqa: C901 — distinct record guards
    from docs.scripts.compare_replays import ensemble_comparison
    pairs = seed_pairs[:3]
    window = (0, 1)
    if change == 'missing_seed':
        pairs = pairs[:2]
    elif change == 'wrong_seed':
        change_settings(pairs[2][1], seed=4)
    elif change == 'duplicate_loading':
        for path in pairs[1]:
            record = json.loads((path / 'run.json').read_text())
            record['settings']['initial_fingerprints']['loading']['v'] = '5' * 64
            (path / 'run.json').write_text(json.dumps(record))
    elif change in ('source', 'runtime'):
        record = json.loads((pairs[2][1] / 'run.json').read_text())
        record['git' if change == 'source' else 'python'] = '9' * 40 if change == 'source' else '3.11.1'
        (pairs[2][1] / 'run.json').write_text(json.dumps(record))
    elif change == 'cadence':
        change_settings(pairs[2][1], output_dt_omega_p=.25)
    elif change == 'physics':
        change_settings(pairs[2][1], force_quiver_over_c=.002)
    elif change in ('moment_lengths', 'reversed_lengths'):
        lengths = [.1, .2] if change == 'moment_lengths' else [4 * np.sqrt(.001), 2 * np.sqrt(.001)]
        for pair in pairs:
            for path in pair:
                change_settings(path, local_spread_lengths_c_over_wp=lengths)
    else:
        window = (0, 2)
    with pytest.raises(ValueError):
        ensemble_comparison(pairs, window=window)


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


def test_shape_and_execution_protocol_cannot_be_hidden_in_timestep_comparison(records):
    change_settings(records[1], shape='quintic', shape_order=5)
    with pytest.raises(ValueError, match='share shape'):
        compare_replays(*records, variant='dt', windows=((0, 1),))
    result = compare_replays(*records, variant='shape', windows=((0, 1),))
    assert result['initial_fingerprints']['verified']
    change_settings(records[1], XLA_FLAGS='--xla_gpu_exclude_nondeterministic_ops')
    with pytest.raises(ValueError, match='share XLA_FLAGS'):
        compare_replays(*records, variant='shape', windows=((0, 1),))


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


@pytest.fixture
def pair_records(tmp_path):
    """Small manufactured scalar ledgers and physical initial arrays, without PIC."""
    import hashlib
    from docs.scripts.compare_replays import PAIR_CASES, window_summary

    c, mass, wp, cells, ppc = 299792458., 9.1093837e-31, 1e9, 512, 2
    length, count = 70 * c / wp, cells * ppc
    scale = dict(omega0_rad_s=wp, field_scale_V_m=mass * c * wp / 1.602176634e-19,
                 energy_scale_J_m2=2 * mass * c**2 * length)
    targets = dict(reference_energy_over_reservoir=1e-9, quadrature_force_over_initial=1e-6,
                   table_force_over_initial=1e-4, primary_window_fraction=.01,
                   nonzero_mode_density_window_fraction=.05, conservation_over_transfer_and_target_difference=.001)
    maximum = dict.fromkeys(('energy_work_over_energy_scale', 'momentum_over_energy_scale_over_c',
                             'charge_over_enL', 'continuity_over_enomega0', 'ordinary_gauss_over_en_eps0',
                             'dark_gauss_over_en_eps0', 'grid_charge_over_enL', 'dark_sector_work_over_energy_scale',
                             'ordinary_sector_work_over_energy_scale'), 0.)
    runtime = dict(git='4' * 40, jax='test', jaxincell='test', numpy='test', python='test',
                   platform='test', jax_enable_x64=True, backend='cpu')
    paths = []
    for seed, dt in ((2e-4, .2), (1e-4, .2), (0., .2), (2e-4, .1)):
        path = tmp_path / f'pair_{seed}_{dt}'
        path.mkdir()
        paths.append(path)
        time = np.arange(round(.4 / dt) + 1) * dt
        x0 = np.tile(np.linspace(-length / 2, length / 2, count, endpoint=False), 2)
        thermal = np.tile(np.array([-.025, .025]), count)
        speed = thermal + np.repeat([1., -1.], count) * seed * np.cos(2 * np.pi * 134 * x0 / length)
        u = np.zeros((2 * count, 3))
        u[:, 0] = c * speed / np.sqrt(1 - speed**2)
        x = np.zeros_like(u)
        x[:, 0] = (x0 + dt / (2 * wp) * c * speed + length / 2) % length - length / 2
        state = dict(x=x, u=u, w=np.full(2 * count, length / count), E=np.zeros((cells, 3)),
                     B=np.zeros((cells, 3)), rho=np.zeros(cells))
        hashes = {}
        for key, value in state.items():
            header = f'{value.dtype.str}:{value.shape}'.encode()
            hashes[key] = hashlib.sha256(header + value.tobytes()).hexdigest()
        speed = speed.reshape(2, count)
        mean, rms = speed.mean(axis=1), speed.std(axis=1)
        initial_k = .5 * np.mean(1 / np.sqrt(1 - speed**2) - 1, axis=1)
        setting = dict(parent_revision='1' * 40, cells=cells, particles_per_cell_per_species=ppc,
                       dt_omega0=dt, horizon_omega0=.4, shape_order=2, seed_mode=134,
                       length_c_over_omega0=70., waterbag_full_width_over_c=.1,
                       velocity_seed_over_c=seed, eta=.5, dark_mass_over_omega0=1., force_quiver_over_c=.05,
                       pair_loading='global', quadrature_nodes=[64, 128], reference_method='DOP853',
                       reference_tolerances=[2e-11, 2e-13], linear_reference_horizon_omega0=.4,
                       table_dt_omega0=2 * dt, native_diagnostic_dt_omega0=dt, saved_output_dt_omega0=.2,
                       local_spread_lengths_c_over_omega0=[.1, .2], local_moments=True,
                       initial_ordinary_fingerprints=hashes, initial_dark_energy_over_energy_scale=.005,
                       normalization=scale, accuracy_targets=targets, XLA_FLAGS='')
        top, reductions, cases = {}, {}, {}
        for label in PAIR_CASES:
            coupled = label == 'coupled'
            field = .01 * time
            kinetic = initial_k + .0001 * time[:, None]
            dark = .005 - field**2 / 2 - .0002 * time if coupled else np.zeros(len(time))
            work = dark - .005 if coupled else field**2 / 2 + .0002 * time
            data = dict(t=time, mean=np.tile(mean, (len(time), 1)), rms=np.tile(rms, (len(time), 1)),
                        max_speed=np.full(len(time), abs(speed).max()), momentum=np.zeros((len(time), 3)),
                        mean_E=field, mean_D=np.sqrt(2 * dark), mean_A=np.zeros(len(time)),
                        mode_E=seed * time * (1 + .2j), dark_mode_E=np.zeros(len(time), dtype=complex),
                        kinetic=kinetic, electric=field**2 / 2, magnetic=np.zeros(len(time)), dark=dark,
                        dark_coherent=dark, work=work, spread=np.tile(.25 * rms**2, (len(time), 1)),
                        local_spread=(np.tile(.25 * rms[None, :, None]**2, (len(time), 1, 2))
                                      + 1e-5 * time[:, None, None]),
                        density_rms=np.tile(time[:, None] * seed, (1, 2)),
                        local_density_rms=np.tile(time[:, None, None] * seed, (1, 2, 2)))
            for key in ('charge', 'grid_charge', 'ordinary_gauss', 'dark_gauss'):
                data[key] = np.zeros(len(time))
            data['balance'] = field**2 / 2 + dark + kinetic.sum(axis=1) - (0 if coupled else work)
            data['source_work'], data['nonzero_electric'] = (-1 if coupled else 1) * work, np.zeros(len(time))
            reductions[label] = window_summary(data, np.ones(len(time), dtype=bool), coupled)
            indices = np.unique(np.r_[np.arange(0, len(time), round(.2 / dt)), len(time) - 1])
            top.update({f'{label}_{key}': value[indices] for key, value in data.items()})
            native = {key: value.copy() for key, value in data.items()
                      if key not in ('source_work', 'nonzero_electric')}
            for key in ('electric', 'magnetic', 'dark', 'dark_coherent', 'kinetic', 'spread', 'balance', 'work',
                        'local_spread'):
                native[key] *= scale['energy_scale_J_m2']
            for key in ('mean', 'rms', 'max_speed'):
                native[key] *= c
            native['momentum'] *= scale['energy_scale_J_m2'] / c
            native['t'] /= wp
            native['mean_A'] *= scale['field_scale_V_m'] / wp
            phase = np.exp(-2j * np.pi * 134 * (-.5 + 1 / cells))
            for key in ('mean_E', 'mean_D', 'mode_E', 'dark_mode_E'):
                native[key] = native[key] * scale['field_scale_V_m'] / (phase if 'mode' in key else 1)
            branch = path / label
            branch.mkdir()
            cases[label] = dict(initial_ordinary_fingerprints=hashes, all_step_maxima=maximum)
            branch_setting = dict(setting, label=label, model='DarkField' if coupled else 'PrescribedDrive',
                                  scalar_units='native SI', block_steps=round(.2 / dt))
            branch_record = dict(runtime, example='pair_waveform_branch', settings=branch_setting, results=cases[label])
            (branch / 'run.json').write_text(json.dumps(branch_record))
            np.savez_compressed(branch / 'data.npz', **native)
            np.savez_compressed(branch / 'initial_state.npz', **state,
                                **{'dark.mass': np.full(2, mass), 'dark.density': np.ones(2),
                                   'dark.charge': np.array([-1., 1.]), 'dark.length': length,
                                   'dark.relativistic': True})
        depletion = .001 * time[indices]
        fraction = top['coupled_dark'] / .005
        knots = np.r_[time[0], time[:-1] + dt / 2, time[-1]]
        coupled_mean = top['coupled_mean'][0]
        with np.load(path / 'coupled' / 'data.npz') as stored:
            dark_mean = stored['mean_D'] / scale['field_scale_V_m']
        push = dark_mean[:-1] - dt / 2 * setting['eta'] * .5 * (coupled_mean[1] - coupled_mean[0])
        force = np.r_[dark_mean[0], push, dark_mean[-1]]
        top.update(realized_table_t=knots, realized_table_D=force,
                   homogeneous_table_t=knots, homogeneous_table_D=force.copy())
        for label in PAIR_CASES[1:]:
            every = 2 if label.endswith('coarse') else 1
            selected = np.unique(np.r_[0, np.arange(1, len(knots) - 1, every), len(knots) - 1])
            amplitude = np.zeros((len(selected), 3))
            amplitude[:, 0] = force[selected] * scale['field_scale_V_m']
            archive_path = path / label / 'initial_state.npz'
            with np.load(archive_path) as archive:
                initial = dict(archive)
            np.savez_compressed(archive_path, **initial,
                                **{'dark.times': knots[selected] / wp, 'dark.amplitude': amplitude,
                                   'dark.phase': 0., 'dark.omega': 0., 'dark.eta': setting['eta']})
        top.update(coupled_total_dark_fraction=fraction, coupled_coherent_dark_fraction=fraction,
                   coupled_nonzero_dark_fraction=np.zeros(len(indices)), homogeneous_dark_fraction=fraction + depletion,
                   additional_dark_depletion_fraction=depletion, additional_dark_depletion_gain=depletion,
                   linear_t=time[indices], linear_coupled_mode_E=top['coupled_mode_E'].copy(),
                   linear_homogeneous_fine_mode_E=top['homogeneous_fine_mode_E'].copy())
        reductions['coupled'].update(additional_dark_depletion_fraction_mean=.001 * time.mean(),
                                     additional_dark_depletion_gain_mean=.001 * time.mean())
        results = dict(cases=cases,
                       windows={'late': dict(bounds_omega0=[0., .4], samples=len(time), reductions=reductions)},
                       reference=dict(max_energy_defect_over_homogeneous_initial_reservoir=0.,
                                      quadrature_force_error_over_initial=0., tolerance_force_error_over_initial=0.),
                       forcing={label: dict(max_midpoint_force_error_over_initial=0.) for label in PAIR_CASES[1:]})
        record = dict(runtime, example='pair_waveform_control', settings=setting, results=results)
        (path / 'run.json').write_text(json.dumps(record))
        np.savez_compressed(path / 'data.npz', **top)
    return paths


@pytest.fixture
def pair_repeat_records(pair_records):
    """Complete manufactured prescribed restarts and ledgers; no dynamics or JAX."""
    import hashlib
    from docs.scripts.compare_replays import PAIR_MAXIMA, _accepted_ticks
    donor = pair_records[0] / 'realized_fine'
    record = json.loads((donor / 'run.json').read_text())
    s = record['settings']
    s.update(horizon_omega0=170., scalar_dt_omega0=.2, forcing_native_dt_omega0=.2, block_steps=5)
    wp, field, energy = s['normalization'].values()
    c, count, cells = 299792458., s['cells'] * s['particles_per_cell_per_species'], s['cells']
    ticks = _accepted_ticks(.2 / wp, 850, 1)
    tau = ticks * wp
    with np.load(donor / 'initial_state.npz') as archive:
        state = dict(archive)
    state.update(format=np.array(1), counts=np.array([count, count]), cells=np.array(cells),
                 algorithm=np.array('explicit'), time=np.array(0.), steps=np.array(0, dtype=np.int32),
                 qm=np.repeat(state['dark.charge'] / state['dark.mass'], count) * 1.602176634e-19,
                 sigma=np.zeros(2), key=np.array([3, 7], dtype=np.uint32), names=np.array(['electrons', 'positrons']))
    state.update({'wall.' + key: np.zeros((2, 2, 3) if 'momentum' in key else (2, 2)) for key in
                  ('arrived', 'collected', 'injected', 'energy_in', 'energy_out', 'energy_injected',
                   'momentum', 'momentum_injected', 'truncated', 'overflow')})
    state.update({'dark.' + key: np.asarray(value) for key, value in dict(
        format=3, mode='drive', cells=cells, length_y=.01, length_z=.01, dt=.2 / wp, has_external_B=False,
        background=0., work=0., initial_dark=0., initial_projection_norm=0., max_balance_error=0.,
        max_ordinary_gauss=0., max_dark_gauss=0.).items()})
    state['dark.times'] = np.array([0., ticks[-1]])
    state['dark.amplitude'] = np.array([[.05 * field, 0., 0.], [.04 * field, 0., 0.]])
    with np.load(donor / 'data.npz') as archive:
        initial = {key: archive[key][0] for key in archive.files}
    state['dark.initial_ordinary'] = initial['balance']
    np.savez_compressed(donor / 'initial_state.npz', **state)
    maximum = dict.fromkeys(PAIR_MAXIMA, 0.)
    maximum.update(energy_work_over_energy_scale=1e-10, ordinary_sector_work_over_energy_scale=1e-10,
                   ordinary_gauss_over_en_eps0=3e-13)
    en = state['dark.density'].sum() * 1.602176634e-19
    length = float(state['dark.length'])
    eps = energy / (length * field**2)
    units = np.array([energy, energy / c, en * length, en * wp, en / eps, en / eps, en * length, energy, energy])
    record['results'].update(all_step_maxima=maximum)
    paths = [donor.parent / f'execution_{index}' for index in (1, 2)]
    for index, path in enumerate([donor, *paths]):
        path.mkdir(exist_ok=True)
        data = {key: np.broadcast_to(value, (len(ticks), *np.shape(value))).copy() for key, value in initial.items()}
        data['t'] = ticks.copy()
        shape = np.sin(np.pi * tau / 170)
        shape[[0, -1]] = 0.
        data['mean_E'] = field * .002 * tau / 170 * (1 + index * 1e-5 * shape)
        mode = .0001 * tau / 170 * np.exp(index * .01j * shape)
        data['mode_E'] = field * mode / 2
        data['electric'] = energy * (.5 * (data['mean_E'] / field)**2 + .25 * abs(mode)**2)
        data['work'] = data['electric'].copy()
        data['balance'] = np.full(len(ticks), initial['balance'])
        data['local_spread'] += energy * 1e-7 * tau[:, None, None] / 170
        result = dict(record['results'], local_spread_initial=(data['local_spread'][0] / energy).tolist(),
                      local_spread_final=(data['local_spread'][-1] / energy).tolist())
        final = {key: value.copy() for key, value in state.items()}
        angle = 2 * np.pi * s['seed_mode'] * np.arange(cells) / cells
        final['E'][:, 0] = data['mean_E'][-1] + field * mode[-1].real * np.cos(angle)
        final.update(time=np.array(ticks[-1]), steps=np.array(850, dtype=np.int32))
        final.update({'dark.work': data['work'][-1], 'dark.max_balance_error': energy * 1e-10,
                      'dark.max_ordinary_gauss': en / eps * 3e-13})
        np.savez_compressed(path / 'final_state.npz', **final)
        np.savez_compressed(path / 'data.npz', **data)
        if index:
            (path / 'initial_state.npz').write_bytes((donor / 'initial_state.npz').read_bytes())
            result.update(execution_index=index, samples=2, all_step_maxima_SI=(np.array(
                [maximum[key] for key in PAIR_MAXIMA]) * units).tolist())
        native = dict(record, git='5' * 40 if index else record['git'], results=result,
                      example='pair_waveform_repeat' if index else 'pair_waveform_branch')
        (path / 'run.json').write_text(json.dumps(native))
    control = json.loads((donor.parent / 'run.json').read_text())
    control['settings'].update({key: s[key] for key in
                                ('horizon_omega0', 'scalar_dt_omega0', 'forcing_native_dt_omega0')})
    control['results']['cases']['realized_fine']['all_step_maxima'] = maximum
    (donor.parent / 'run.json').write_text(json.dumps(control))
    for path in paths:
        change_settings(path, donor_git=record['git'], initial_state_archive_sha256=hashlib.sha256(
            (donor / 'initial_state.npz').read_bytes()).hexdigest(),
            donor_record_sha256=hashlib.sha256((donor / 'run.json').read_bytes()).hexdigest(),
            donor_control_record_sha256=hashlib.sha256((donor.parent / 'run.json').read_bytes()).hexdigest(),
            producer_script_sha256='6' * 64)
    observer = donor.parent / 'observer.json'
    entries = [dict(index=index, runtime_available=True, same_loaded_executable_as_first=index == 2,
                    stablehlo_sha256='7' * 64, optimized_hlo_text_sha256='8' * 64, observer_seconds=.1)
               for index in (1, 2)]
    observed = dict(source='5' * 40, completed=True, producer_script_sha256='6' * 64, compilations=entries,
                    method='Host-only identity observation.', timing_note='Text hashing is outside native timings.',
                    interpretation='Process-local identity; text hashes do not serialize binaries.')
    observer.write_text(json.dumps(observed))
    return paths, donor, observer


def test_pair_repeat_raw_phase_budget_and_optional_source_observer(pair_repeat_records):
    import runpy
    from docs.scripts.compare_replays import pair_repeat_comparison
    paths, donor, observer = pair_repeat_records
    result = pair_repeat_comparison(paths, observer, donor, ('4' * 40, '5' * 40))
    late = result['windows'][1]
    assert late['bounds_omega0'] == [150, 170] and late['samples'] == 101
    time = np.array(late['actual_clocks_omega0'])
    tau = np.linspace(*time, 101)
    weight = (tau / 170)**2
    expected = np.sqrt(np.sum(weight * (.01 * np.sin(np.pi * tau / 170))**2) / weight.sum())
    assert late['mode_phase']['reference_amplitude_squared_weighted_rms_rad'] == pytest.approx(expected, abs=1e-14)
    assert not late['mode_phase']['alignment'] and not result['all_scalar_histories_bitwise_equal']
    assert late['accounting']['transfer_gates'] == [True, True]
    assert not late['accounting']['difference_gate']
    np.testing.assert_allclose(late['accounting']['two_endpoint_bound_over_abs_window_work'],
                               2 * np.array(late['accounting']['global_defect_over_abs_window_work']))
    assert not result['original_gauss_gates'][0]['ordinary_gauss_over_en_eps0']
    assert 'nonserializable' in result['observer']['claim']
    assert result['observer']['wrapper_script_sha256'] is None
    assert result['observer']['timing_note'] == 'Text hashing is outside native timings.'
    assert result['donor']['reviewed_source_transition'] == ['4' * 40, '5' * 40]
    archive = result['archive_diagnostics'][0]
    assert len(archive['final_leaves']) == 51 and 'shape_order' not in archive['final_leaves']
    assert archive['legacy_shape_order_default'] == 2  # The validated in-memory restart has 52 fields.
    assert str(donor) not in json.dumps(result, allow_nan=False)
    output = donor.parent / 'repeat.json'
    runpy.run_path('docs/scripts/compare_replays.py', run_name='__main__',
                   init_globals=dict(pair_repeats=paths, pair_repeat_observer=observer,
                                     pair_repeat_donor=donor, pair_repeat_transition=('4' * 40, '5' * 40),
                                     output=output))
    published = json.loads(output.read_text())
    validation = published.pop('validation_source')
    assert len(validation['git']) == 40 and validation['numpy'] == np.__version__
    assert validation['script_sha256'] == hashlib.sha256(Path(validation['script']).read_bytes()).hexdigest()
    assert published == result


@pytest.mark.parametrize('corrupt', [
    'source', 'runtime', 'forcing', 'clock', 'energy', 'rms', 'maxima',
    'SI_maxima', 'endpoint', 'missing_leaf', 'shape', 'initial_sha', 'observer'])
def test_pair_repeat_rejects_changed_exact_inputs_and_ledgers(pair_repeat_records, corrupt):
    from docs.scripts.compare_replays import pair_repeat_comparison
    paths, _, observer = pair_repeat_records
    path = paths[1]
    record = json.loads((path / 'run.json').read_text())
    if corrupt in ('source', 'runtime'):
        record['git' if corrupt == 'source' else 'jax'] = '9' * 40
    elif corrupt in ('initial_sha', 'shape'):
        record['settings']['initial_state_archive_sha256' if corrupt == 'initial_sha' else 'shape_order'] = (
            '9' * 64 if corrupt == 'initial_sha' else 5)
    elif corrupt in ('maxima', 'SI_maxima'):
        if corrupt == 'maxima':
            record['results']['all_step_maxima'].pop('continuity_over_enomega0')
        else:
            record['results']['all_step_maxima_SI'][0] *= 2
    elif corrupt == 'observer':
        observed = json.loads(observer.read_text())
        observed['compilations'][1]['same_loaded_executable_as_first'] = False
        observer.write_text(json.dumps(observed))
    else:
        filename = 'initial_state.npz' if corrupt in ('forcing', 'missing_leaf') else (
            'final_state.npz' if corrupt == 'endpoint' else 'data.npz')
        with np.load(path / filename) as archive:
            values = dict(archive)
        if corrupt == 'missing_leaf':
            values.pop('wall.collected')
        else:
            key = dict(forcing='dark.phase', clock='t', energy='electric', rms='rms', endpoint='u')[corrupt]
            values[key] = values[key] + (1e-14 if corrupt == 'clock' else 1.)
        np.savez_compressed(path / filename, **values)
    (path / 'run.json').write_text(json.dumps(record))
    with pytest.raises(ValueError):
        pair_repeat_comparison(paths, observer)


def test_pair_repeat_donor_requires_explicit_review_and_observer_text_is_not_binary(pair_repeat_records):
    from docs.scripts.compare_replays import pair_repeat_comparison
    paths, donor, observer = pair_repeat_records
    with pytest.raises(ValueError, match='reviewed'):
        pair_repeat_comparison(paths, donor=donor)
    assert 'observer' not in pair_repeat_comparison(paths)
    observed = json.loads(observer.read_text())
    observed.update(script_sha256='9' * 64, timing_note='Private wrapper hashing was inside compile timing.')
    observer.write_text(json.dumps(observed))
    result = pair_repeat_comparison(paths, observer)
    assert result['observer']['wrapper_script_sha256'] == '9' * 64
    assert 'not binary identity' in result['observer']['claim']
    assert result['observer']['timing_note'] == 'Private wrapper hashing was inside compile timing.'


def test_pair_repeat_same_source_donor_has_no_transition_or_executable_claim(pair_repeat_records):
    import hashlib
    from docs.scripts.compare_replays import pair_repeat_comparison
    paths, donor, _ = pair_repeat_records
    for file in (donor / 'run.json', donor.parent / 'run.json'):
        record = json.loads(file.read_text())
        record['git'] = '5' * 40
        file.write_text(json.dumps(record))
    for path in paths:
        change_settings(path, donor_git='5' * 40,
                        donor_record_sha256=hashlib.sha256((donor / 'run.json').read_bytes()).hexdigest(),
                        donor_control_record_sha256=hashlib.sha256(
                            (donor.parent / 'run.json').read_bytes()).hexdigest())
    result = pair_repeat_comparison(paths, donor=donor)
    assert result['donor']['reviewed_source_transition'] is None
    assert result['donor']['claim'] == 'Same producer source; donor executable identity is unobserved.'
    with pytest.raises(ValueError, match='only for differing sources'):
        pair_repeat_comparison(paths, donor=donor, transition=('5' * 40, '5' * 40))


def test_pair_repeat_zero_signal_has_undefined_phase_not_false_agreement():
    from docs.scripts.compare_replays import _pair_repeat_windows, PAIR_MAXIMA
    time = np.array([0., 150., 160., 170.])
    data = dict(t=time, work=time * 1e-6, electric=np.ones(4) * .01, magnetic=np.zeros(4),
                kinetic=np.ones((4, 2)) * .01, nonzero_electric=np.ones(4) * .001,
                mean_E=np.ones(4), rms=np.ones((4, 2)), momentum=np.zeros((4, 3)),
                local_spread=np.ones((4, 2, 2)), local_density_rms=np.zeros((4, 2, 2)), mode_E=np.ones(4, complex))
    source = dict(results=dict(all_step_maxima=dict.fromkeys(PAIR_MAXIMA, 0.)))
    other = dict(data, mode_E=np.zeros(4, complex))
    phase = _pair_repeat_windows((source, data), (source, other))[1]['mode_phase']
    assert phase['reference_amplitude_squared_weighted_rms_rad'] is None
    assert phase['reference_weight_fraction_with_defined_phase'] == 0 and phase['endpoint_rad'] is None


def test_pair_native_cadence_loading_and_raw_window_contract(pair_records):
    from docs.scripts.compare_replays import _pair_load, pair_control_comparison
    result = pair_control_comparison([_pair_load(path) for path in pair_records])
    fine = result['controlled_pairs'][-1]
    assert fine['variant'] == 'dt' and fine['initial_hash_matches']['u']
    assert not fine['initial_hash_matches']['x']
    assert fine['reconstructed_position_max_difference_over_length'] < 2e-13
    assert fine['branches']['coupled']['late_shared_native_samples'] == 3
    assert result['early_seed_controls'][0]['amplitude_multiplier'] == 2
    assert result['early_seed_controls'][1]['amplitude_multiplier'] is None
    assert result['late'][-1]['native_reductions']['coupled']['samples'] == 5
    reduction = result['late'][0]['native_reductions']['coupled']
    assert reduction['additional_dark_depletion_gain_mean'] == pytest.approx(.0002)
    assert result['saved_early_linear_errors'][0]['coupled']['l2_difference'] == pytest.approx(0)


def test_one_complete_pair_control_retains_budget_gates_without_a_refinement_claim(pair_records):
    from docs.scripts.compare_replays import _pair_load, pair_control_comparison
    result = pair_control_comparison([_pair_load(pair_records[0])])
    assert result['controlled_pairs'] == [] and result['early_seed_controls'] == []
    assert len(result['late'][0]['conservation']) == 5
    assert result['late'][0]['conservation_gates']['coupled'][
        'maximum_energy_sector_defect_over_depletion_gain']
    assert 'no late-convergence' in result['claim']
    with pytest.raises(ValueError, match='at least one complete'):
        pair_control_comparison([])


@pytest.mark.parametrize(
    'corrupt', ['source', 'x64', 'parent', 'branch_runtime', 'clock', 'top_clock',
                'field_scale', 'archive', 'raw_mean', 'blocks'])
def test_pair_rejects_unmatched_source_clocks_units_and_initial_arrays(pair_records, corrupt):
    from docs.scripts.compare_replays import _pair_load
    path = pair_records[0]
    record_path = path / 'run.json'
    if corrupt in ('clock', 'archive', 'branch_runtime', 'blocks'):
        path /= 'coupled'
        record_path = path / 'run.json'
    if corrupt in ('clock', 'top_clock', 'archive'):
        filename = 'initial_state.npz' if corrupt == 'archive' else 'data.npz'
        with np.load(path / filename) as stored:
            data = dict(stored)
        if corrupt == 'archive':
            data['u'][0, 0] += 1
        else:
            data['t' if corrupt == 'clock' else 'coupled_t'][1] += .00001
        np.savez_compressed(path / filename, **data)
    else:
        record = json.loads(record_path.read_text())
        if corrupt == 'source':
            record['git'] += '-dirty'
        elif corrupt == 'x64':
            record['jax_enable_x64'] = False
        elif corrupt == 'parent':
            record['settings']['parent_revision'] = 'unpinned'
        elif corrupt == 'branch_runtime':
            record['jax'] = 'other'
        elif corrupt == 'field_scale':
            record['settings']['normalization']['field_scale_V_m'] *= 2
        elif corrupt == 'blocks':
            record['settings']['block_steps'] = 3
        else:
            record['results']['windows']['late']['reductions']['coupled']['electric_mean'] *= 2
        record_path.write_text(json.dumps(record))
    with pytest.raises(ValueError):
        source = _pair_load(pair_records[0])
        from docs.scripts.compare_replays import _pair_late
        _pair_late(source)


def test_pair_publisher_preserves_native_records_and_effect_relative_failures(pair_records):
    from docs.scripts.compare_replays import publish_pair_controls
    for path in pair_records[:3]:
        for filename in (path / 'run.json', path / 'coupled' / 'run.json'):
            record = json.loads(filename.read_text())
            maxima = (record['results']['all_step_maxima'] if filename.parent.name == 'coupled'
                      else record['results']['cases']['coupled']['all_step_maxima'])
            maxima['energy_work_over_energy_scale'] = 1e-7
            filename.write_text(json.dumps(record))
    folder = pair_records[0].parent / 'pair_evidence'
    publish_pair_controls(pair_records[:3], folder)
    record = json.loads((folder / 'run.json').read_text())
    result = record['results']
    coupled = result['late'][0]['conservation_gates']['coupled']
    assert not coupled['maximum_energy_sector_defect_over_depletion_gain']
    assert len(result['native_branches'][0]) == 5
    assert len(result['native_source_sha256'][0]['branches']['coupled']['data_sha256']) == 64
    assert str(pair_records[0]) not in json.dumps(record, allow_nan=False)
    with np.load(folder / 'data.npz') as stored, np.load(pair_records[0] / 'data.npz') as native:
        np.testing.assert_array_equal(stored['0_linear_coupled_mode_E'], native['linear_coupled_mode_E'])
        assert not any(key.endswith('_x') or '_table_' in key for key in stored.files)


def sparse_pair(path):
    """Thin only stored scalars; retain native pump inputs and conservation maxima."""
    from docs.scripts.compare_replays import PAIR_CASES, window_summary

    record = json.loads((path / 'run.json').read_text())
    record['settings']['scalar_dt_omega0'] = .2
    for label in PAIR_CASES:
        folder = path / label
        with np.load(folder / 'data.npz') as stored:
            data = dict(stored)
        reduced = {key: value[::2] for key, value in data.items()}
        if label == 'coupled':
            reduced.update({'pump_' + key: data[key] for key in ('t', 'mean', 'mean_D', 'mean_A')})
        np.savez_compressed(folder / 'data.npz', **reduced)
        branch = json.loads((folder / 'run.json').read_text())
        branch['settings']['scalar_dt_omega0'] = .2
        branch['settings']['forcing_native_dt_omega0'] = .1
        (folder / 'run.json').write_text(json.dumps(branch))
        from docs.scripts.compare_replays import _pair_normalize
        values = _pair_normalize(reduced, record['settings'], label)
        summary = window_summary(values, np.ones(len(values['t']), dtype=bool), label == 'coupled')
        record['results']['windows']['late']['reductions'][label].update(summary)
    record['results']['windows']['late']['samples'] = 3
    (path / 'run.json').write_text(json.dumps(record))


def test_pair_sparse_pump_preserves_actual_force_and_labels_sampling(pair_records):
    from docs.scripts.compare_replays import _pair_load, pair_control_comparison, publish_pair_controls
    dense = pair_records[-1]
    sparse = dense.parent / 'sparse'
    copytree(dense, sparse)
    sparse_pair(sparse)
    sources = [_pair_load(path) for path in (dense, sparse)]
    result = pair_control_comparison(sources)
    row = result['controlled_pairs'][0]
    assert row['variant'] == 'sampling' and row['branches']['coupled']['refinement_gates'] is None
    assert all(value is None for value in row['raw_window_mean_gates'].values())
    assert row['raw_window_scalar_dt_omega0'] == [.1, .2]
    assert row['branches']['coupled']['late_shared_native_samples'] == 3
    assert result['late'][1]['native_reductions']['coupled']['samples'] == 3
    assert result['late'][0]['native_reductions']['coupled']['samples'] == 5
    assert sources[0][3]['coupled'][2]['realized_force_sha256'] == sources[1][3]['coupled'][2]['realized_force_sha256']
    assert sources[0][3]['coupled'][2]['dense_pump_sha256'] == sources[1][3]['coupled'][2]['dense_pump_sha256']
    hashes = [source[3]['realized_fine'][2]['actual_force_sha256'] for source in sources]
    assert hashes[0] == hashes[1]
    assert all(row['relative_l2_difference'] == 0 for row in result['saved_early_linear_errors'][1].values())
    folder = dense.parent / 'sampling_evidence'
    publish_pair_controls((dense, sparse), folder)
    published = json.loads((folder / 'run.json').read_text())['results']
    assert published['controlled_pairs'][0]['variant'] == 'sampling'
    assert published['native_source_sha256'][1]['branches']['coupled']['dense_pump_sha256']
    with np.load(folder / 'data.npz') as stored:
        assert stored['0_coupled_t'].shape == stored['1_coupled_t'].shape == (3,)
    sources[1][0]['settings']['initial_ordinary_fingerprints']['rho'] = '0' * 64
    with pytest.raises(ValueError, match='repeat/sampling'):
        pair_control_comparison(sources)


@pytest.mark.parametrize('corrupt', ['missing_pump', 'pump_clock', 'pump_units', 'midpoint', 'force', 'phase'])
def test_pair_sparse_force_or_pump_mismatch_is_rejected(pair_records, corrupt):
    from docs.scripts.compare_replays import _pair_load
    path = pair_records[-1]
    sparse_pair(path)
    if corrupt in ('missing_pump', 'pump_clock', 'pump_units'):
        filename = path / 'coupled' / 'data.npz'
        with np.load(filename) as stored:
            data = dict(stored)
        if corrupt == 'missing_pump':
            del data['pump_mean_A']
        elif corrupt == 'pump_clock':
            data['pump_t'][1] += .01e-9
        else:
            data['pump_mean_D'] *= 2
    elif corrupt == 'midpoint':
        filename = path / 'data.npz'
        with np.load(filename) as stored:
            data = dict(stored)
        data['realized_table_t'][1] += .01
    else:
        filename = path / 'realized_fine' / 'initial_state.npz'
        with np.load(filename) as stored:
            data = dict(stored)
        if corrupt == 'phase':
            data['dark.phase'] = np.asarray(.01)
        else:
            data['dark.amplitude'][1, 0] *= 1.01
    np.savez_compressed(filename, **data)
    with pytest.raises(ValueError):
        _pair_load(path)


@pytest.fixture
def pair_table_records(pair_repeat_records):
    """Nested force tables and complete prescribed archives, with no simulated dynamics."""
    from docs.scripts.compare_replays import _accepted_ticks
    paths, fine, _ = pair_repeat_records
    coarse, middle = fine.parent / 'realized_coarse', paths[0]
    copytree(paths[1], coarse, dirs_exist_ok=True)
    control = json.loads((fine.parent / 'run.json').read_text())
    record = json.loads((fine / 'run.json').read_text())
    s = record['settings']
    s['table_dt_omega0'] = control['settings']['table_dt_omega0'] = .8
    wp, field, _ = s['normalization'].values()
    ticks = _accepted_ticks(s['dt_omega0'] / wp, 850, 1)
    knots = np.r_[0., ticks[:-1] + s['dt_omega0'] / (2 * wp), ticks[-1]]
    with np.load(fine / 'data.npz') as stored:
        pump_data = dict(stored)
    pump_data['mean_D'] = field * (.05 + .01 * np.cos(np.pi * ticks * wp / 170))
    pump_data['mean_A'] = np.zeros(len(ticks))
    current = .5 * (pump_data['mean'][:, 1] - pump_data['mean'][:, 0]) / 299792458.
    push = pump_data['mean_D'][:-1] / field - s['dt_omega0'] / 2 * s['eta'] * current[:-1]
    force = np.zeros((len(knots), 3))
    force[:, 0] = field * np.r_[pump_data['mean_D'][0] / field, push, pump_data['mean_D'][-1] / field]
    pump_path = fine.parent / 'coupled'
    pump_record = json.loads((pump_path / 'run.json').read_text())
    pump_record['settings'] = dict(s, label='coupled', model='DarkField')
    (pump_path / 'run.json').write_text(json.dumps(pump_record))
    np.savez_compressed(pump_path / 'data.npz', **pump_data)
    with np.load(fine.parent / 'data.npz') as stored:
        top = dict(stored)
    top.update(realized_table_t=knots * wp, realized_table_D=force[:, 0] / field,
               homogeneous_table_t=knots * wp)
    np.savez_compressed(fine.parent / 'data.npz', **top)
    for folder, every in ((coarse, 4), (middle, 2), (fine, 1)):
        selected = np.unique(np.r_[0, np.arange(1, len(knots) - 1, every), len(knots) - 1])
        for filename in ('initial_state.npz', 'final_state.npz'):
            with np.load(folder / filename) as stored:
                state = dict(stored)
            state.update({'dark.times': knots[selected], 'dark.amplitude': force[selected]})
            np.savez_compressed(folder / filename, **state)
        native = json.loads((folder / 'run.json').read_text())
        native['settings']['table_dt_omega0'] = .8
        if folder == coarse:
            native.update(git=record['git'], example='pair_waveform_branch', settings=dict(s, label='realized_coarse'))
            control['results']['cases']['realized_coarse'] = native['results']
        if folder == middle:
            def native_hash(value):
                return hashlib.sha256(f'{value.dtype.str}:{value.shape}'.encode() + value.tobytes()).hexdigest()
            native['example'] = 'pair_waveform_table_replay'
            native['settings']['table_replay'] = dict(
                table_every=2, original_knots=len(knots), retained_knots=len(selected),
                maximum_gap_omega0=float(np.max(np.diff(knots[selected]))) * wp,
                original_times_sha256=native_hash(knots), original_amplitude_sha256=native_hash(force),
                times_sha256=native_hash(knots[selected]), amplitude_sha256=native_hash(force[selected]),
                selection='initial endpoint, first midpoint, every nth midpoint, terminal endpoint')
            native['results'].update(samples=1, execution_index=1)
        (folder / 'run.json').write_text(json.dumps(native))
        if folder != middle:
            actual = np.interp(knots[1:-1], knots[selected], force[selected, 0])
            control['results']['forcing'][native['settings']['label']]['max_midpoint_force_error_over_initial'] = (
                float(np.max(abs(actual - force[1:-1, 0])) / abs(force[0, 0])))
    (fine.parent / 'run.json').write_text(json.dumps(control))
    change_settings(middle, donor_record_sha256=hashlib.sha256((fine / 'run.json').read_bytes()).hexdigest(),
                    donor_control_record_sha256=hashlib.sha256((fine.parent / 'run.json').read_bytes()).hexdigest(),
                    initial_state_archive_sha256=hashlib.sha256((fine / 'initial_state.npz').read_bytes()).hexdigest())
    return (coarse, middle, fine)


def test_pair_table_resolution_preserves_original_gates_and_reproducible_raw_norms(pair_table_records):
    import runpy
    from docs.scripts.compare_replays import pair_table_comparison
    result = pair_table_comparison(pair_table_records, ('4' * 40, '5' * 40))
    tables = result['forcing']['tables']
    assert result['forcing']['coarse_exact_decimation_of_fine']
    np.testing.assert_allclose([row['nominal_spacing_omega0'] for row in tables], [.8, .4, .2])
    assert tables[0]['max_midpoint_force_error_over_initial'] > tables[1]['max_midpoint_force_error_over_initial'] > 0
    assert tables[2]['max_midpoint_force_error_over_initial'] == 0
    late = result['comparisons'][0]['windows'][1]
    assert late['samples'] == 101 and late['bounds_omega0'] == [150, 170]
    raw = result['scalar_histories']
    assert all(len(row) == 14 and 'dark' not in row for row in raw)
    a, b = [np.asarray(row['mode_E']['real']) + 1j * np.asarray(row['mode_E']['imag']) for row in raw[:2]]
    selected = np.asarray(raw[0]['t']) >= 150 - 1e-8
    assert late['observables']['mode_E']['relative_l2_difference'] == pytest.approx(
        np.linalg.norm(b[selected] - a[selected]) / np.linalg.norm(a[selected]))
    assert 'kinetic_increment' in late['refinement_gates'] and 'local_spread_increment' in late['refinement_gates']
    work = [row['work_increment'] for row in late['reductions']]
    assert late['accounting']['two_endpoint_bound_over_abs_work_difference'] == pytest.approx(
        4e-10 / abs(work[1] - work[0]))
    assert not late['accounting']['two_endpoint_difference_gate']
    assert not result['original_gauss_gates'][0]['ordinary_gauss_over_en_eps0']
    assert str(pair_table_records[0]) not in json.dumps(result, allow_nan=False)
    output = pair_table_records[0].parent / 'tables.json'
    runpy.run_path('docs/scripts/compare_replays.py', run_name='__main__', init_globals=dict(
        pair_tables=pair_table_records, pair_table_transition=('4' * 40, '5' * 40), output=output))
    published = json.loads(output.read_text())
    assert 'validation_source' in published and published['forcing'] == result['forcing']


@pytest.mark.parametrize('corrupt', [
    'source', 'runtime', 'pic_dt', 'initial_reference', 'hash', 'force',
    'first_knot', 'last_knot', 'si_bound', 'type', 'donor', 'ordinary'])
def test_pair_table_resolution_rejects_changed_protocol_or_native_force(pair_table_records, corrupt):  # noqa: C901
    from docs.scripts.compare_replays import pair_table_comparison, pair_repeat_comparison
    coarse, middle, fine = pair_table_records
    native = json.loads((middle / 'run.json').read_text())
    if corrupt in ('initial_reference', 'force', 'first_knot', 'last_knot', 'ordinary'):
        with np.load(middle / 'initial_state.npz') as stored:
            state = dict(stored)
        key = {'initial_reference': 'dark.initial_ordinary', 'force': 'dark.amplitude',
               'first_knot': 'dark.times', 'last_knot': 'dark.times', 'ordinary': 'key'}[corrupt]
        if corrupt == 'force':
            state[key][1, 0] *= 1.01
        elif corrupt == 'first_knot':
            state[key][1] += 1e-15
        elif corrupt == 'last_knot':
            state[key][-1] += 1e-15
        else:
            state[key] *= 2
        np.savez_compressed(middle / 'initial_state.npz', **state)
    else:
        if corrupt == 'source':
            native['git'] = '9' * 40
        elif corrupt == 'runtime':
            native['numpy'] = 'different'
        elif corrupt == 'pic_dt':
            native['settings']['dt_omega0'] /= 2
        elif corrupt == 'hash':
            native['settings']['table_replay']['times_sha256'] = '0' * 64
        elif corrupt == 'si_bound':
            native['results']['all_step_maxima_SI'][0] *= 2
        elif corrupt == 'type':
            native['example'] = 'pair_waveform_repeat'
        elif corrupt == 'donor':
            native['settings']['donor_record_sha256'] = '0' * 64
        (middle / 'run.json').write_text(json.dumps(native))
    with pytest.raises(ValueError):
        pair_table_comparison((coarse, middle, fine), ('4' * 40, '5' * 40))
    with pytest.raises(ValueError):
        pair_repeat_comparison((middle, middle))


def test_pair_table_resolution_requires_explicit_source_review(pair_table_records):
    from docs.scripts.compare_replays import pair_table_comparison
    with pytest.raises(ValueError, match='explicit reviewed'):
        pair_table_comparison(pair_table_records)


def test_pair_table_resolution_reports_roundoff_non_nested_coarse_knots(pair_table_records):
    from docs.scripts.compare_replays import pair_table_comparison
    coarse = pair_table_records[0]
    for name in ('initial_state.npz', 'final_state.npz'):
        with np.load(coarse / name) as stored:
            state = dict(stored)
        state['dark.times'][2] = np.nextafter(state['dark.times'][2], np.inf)
        np.savez_compressed(coarse / name, **state)
    result = pair_table_comparison(pair_table_records, ('4' * 40, '5' * 40))
    assert not result['forcing']['coarse_exact_decimation_of_fine']
    assert result['forcing']['intermediate_exact_decimation_of_fine']


def test_pair_table_resolution_inherits_original_moment_protocol_without_rewriting_records(pair_table_records):
    from docs.scripts.compare_replays import pair_table_comparison
    inherited = ('accuracy_targets', 'table_dt_omega0', 'length_c_over_omega0', 'waterbag_full_width_over_c',
                 'quadrature_nodes', 'saved_output_dt_omega0')
    for index, folder in enumerate(pair_table_records):
        record = json.loads((folder / 'run.json').read_text())
        for key in (*inherited, *(('local_moments', 'local_spread_lengths_c_over_omega0') if index != 1 else ())):
            del record['settings'][key]
        (folder / 'run.json').write_text(json.dumps(record))
    change_settings(pair_table_records[1], donor_record_sha256=hashlib.sha256(
        (pair_table_records[2] / 'run.json').read_bytes()).hexdigest())
    result = pair_table_comparison(pair_table_records, ('4' * 40, '5' * 40))
    assert 'local_moments' not in result['native_runs'][2]['settings']
    assert result['comparisons'][0]['windows'][1]['samples'] == 101


@pytest.fixture
def pair_pic_step_records(pair_table_records):
    """Two distinct accepted clocks and exactly one force; NumPy manufactured ledgers."""
    from docs.scripts.compare_replays import _accepted_ticks, _pair_native_hash
    _, replay, donor = pair_table_records
    control = json.loads((donor.parent / 'run.json').read_text())
    fine = json.loads((donor / 'run.json').read_text())
    s = fine['settings']
    s.update(dt_omega0=.1, forcing_native_dt_omega0=.1, block_steps=10)
    control['settings'].update(scalar_dt_omega0=.2, dt_omega0=.1, forcing_native_dt_omega0=.1)
    wp, field, _ = s['normalization'].values()
    dense = _accepted_ticks(.1 / wp, 1700, 1)
    knots = np.r_[0., dense[:-1] + .1 / (2 * wp), dense[-1]]
    with np.load(donor / 'data.npz') as stored:
        raw = dict(stored)
    pump = {key: np.broadcast_to(raw[key][0], (len(dense), *raw[key].shape[1:])).copy() for key in raw}
    pump['t'] = dense
    pump['mean_D'] = field * (.05 + .01 * np.cos(np.pi * dense * wp / 170))
    current = .5 * (pump['mean'][:, 1] - pump['mean'][:, 0]) / 299792458.
    force = np.zeros((len(knots), 3))
    force[:, 0] = field * np.r_[
        pump['mean_D'][0] / field, pump['mean_D'][:-1] / field - .05 * s['eta'] * current[:-1],
        pump['mean_D'][-1] / field]
    sparse = {key: value[::2] for key, value in pump.items()}
    sparse.update({'pump_' + key: pump[key] for key in ('t', 'mean', 'mean_D', 'mean_A')})
    coupled = donor.parent / 'coupled'
    coupled_record = json.loads((coupled / 'run.json').read_text())
    coupled_record['settings'] = dict(s, label='coupled', model='DarkField')
    np.savez_compressed(coupled / 'data.npz', **sparse)
    with np.load(donor.parent / 'data.npz') as stored:
        top = dict(stored)
    top.update(realized_table_t=knots * wp, homogeneous_table_t=knots * wp, realized_table_D=force[:, 0] / field)
    np.savez_compressed(donor.parent / 'data.npz', **top)
    c = 299792458.
    originals = []
    for name, index in (('initial_state.npz', 0), ('final_state.npz', -1)):
        with np.load(donor / name) as stored:
            state = dict(stored)
        velocity = state['u'][:, 0] / np.sqrt(1 + np.sum((state['u'] / c)**2, axis=1))
        length = float(state['dark.length'])
        canonical = (state['x'][:, 0] - float(state['dark.dt']) * velocity / 2 + length / 2) % length - length / 2
        state['x'][:, 0] = (canonical + .1 / wp * velocity / 2 + length / 2) % length - length / 2
        state.update({'dark.dt': np.array(.1 / wp), 'dark.times': knots, 'dark.amplitude': force})
        if index:
            state.update(time=np.array(dense[-1]), steps=np.array(1700, dtype=np.int32))
        originals.append(state)
        np.savez_compressed(donor / name, **state)
    fine['results']['initial_ordinary_fingerprints']['x'] = _pair_native_hash(originals[0]['x'])
    s['initial_ordinary_fingerprints'] = fine['results']['initial_ordinary_fingerprints']
    control['settings']['initial_ordinary_fingerprints'] = s['initial_ordinary_fingerprints']
    coupled_record['results']['initial_ordinary_fingerprints'] = s['initial_ordinary_fingerprints']
    (coupled / 'run.json').write_text(json.dumps(coupled_record))
    raw['t'] = dense[::2]
    np.savez_compressed(donor / 'data.npz', **raw)
    (donor / 'run.json').write_text(json.dumps(fine))
    (donor.parent / 'run.json').write_text(json.dumps(control))
    changed = json.loads((replay / 'run.json').read_text())
    changed['example'] = 'pair_waveform_pic_dt_replay'
    changed['settings'] = dict(s, dt_omega0=.2, block_steps=5, local_moments=True)
    changed['settings'].update(
        donor_git=fine['git'],
        initial_state_archive_sha256=hashlib.sha256((donor / 'initial_state.npz').read_bytes()).hexdigest(),
        donor_record_sha256=hashlib.sha256((donor / 'run.json').read_bytes()).hexdigest(),
        donor_control_record_sha256=hashlib.sha256((donor.parent / 'run.json').read_bytes()).hexdigest(),
        producer_script_sha256='6' * 64,
        pic_dt_replay=dict(
            donor_dt_omega0=.1, pic_dt_omega0=.2, block_horizon_omega0=1.,
            times_sha256=_pair_native_hash(knots), amplitude_sha256=_pair_native_hash(force),
            canonical_integer_positions_sha256='7' * 64, canonical_position_error_over_L=0.))
    ticks = _accepted_ticks(.2 / wp, 850, 1)
    for name, state, index in zip(('initial_state.npz', 'final_state.npz'), originals, (0, -1)):
        state = {key: value.copy() for key, value in state.items()}
        velocity = state['u'][:, 0] / np.sqrt(1 + np.sum((state['u'] / c)**2, axis=1))
        canonical = (state['x'][:, 0] - .1 / wp * velocity / 2 + length / 2) % length - length / 2
        state['x'][:, 0] = (canonical + .2 / wp * velocity / 2 + length / 2) % length - length / 2
        state['dark.dt'] = np.array(.2 / wp)
        if index:
            state.update(time=np.array(ticks[-1]), steps=np.array(850, dtype=np.int32))
        else:
            changed['results']['initial_ordinary_fingerprints']['x'] = _pair_native_hash(state['x'])
        np.savez_compressed(replay / name, **state)
    raw['t'] = ticks
    phase = .002 * np.sin(np.pi * ticks * wp / 170)
    phase[[0, -1]] = 0.
    raw['mode_E'] *= np.exp(1j * phase)
    np.savez_compressed(replay / 'data.npz', **raw)
    changed['results'].update(samples=1, execution_index=1)
    (replay / 'run.json').write_text(json.dumps(changed))
    return donor, replay


def test_pair_pic_timestep_preserves_force_restagger_clocks_failed_gates_and_raw_phase(pair_pic_step_records):
    import runpy
    from docs.scripts.compare_replays import pair_pic_step_comparison
    result = pair_pic_step_comparison(pair_pic_step_records, ('4' * 40, '5' * 40))
    assert result['initialization']['all_other_native_initial_leaves_exact']
    assert result['initialization']['fixed_force_byte_identical']
    assert max(result['initialization']['independent_numpy_position_errors'].values()) < 2e-13
    assert all(result['nominal_clock_gates'])
    assert not any(row['ordinary_gauss_over_en_eps0'] for row in result['original_gauss_gates'])
    late = result['windows'][1]
    assert late['samples'] == 101 and late['actual_clocks_per_run_omega0'][0] != late['actual_clocks_per_run_omega0'][1]
    expected_mode = [np.array(row['mode_E']['real']) + 1j * np.array(row['mode_E']['imag'])
                     for row in result['scalar_histories']]
    np.testing.assert_allclose(late['observables']['mode_E']['relative_l2_difference'],
                               np.linalg.norm(expected_mode[1][-101:] - expected_mode[0][-101:])
                               / np.linalg.norm(expected_mode[0][-101:]), rtol=1e-14)
    assert late['mode_phase']['reference_amplitude_squared_weighted_rms_rad'] > 0
    assert late['accounting']['difference_gate'] is False
    assert result['native_runs'][1]['example'] == 'pair_waveform_pic_dt_replay'
    assert 'isolated truncation cause' in result['notes'] and str(pair_pic_step_records[0]) not in json.dumps(result)
    output = pair_pic_step_records[0].parent / 'steps.json'
    runpy.run_path('docs/scripts/compare_replays.py', run_name='__main__', init_globals=dict(
        pair_pic_steps=pair_pic_step_records, pair_pic_transition=('4' * 40, '5' * 40), output=output))
    published = json.loads(output.read_text())
    assert published['windows'] == result['windows'] and published['validation_source']['script_sha256']


@pytest.mark.parametrize('corrupt', [
    'transition', 'runtime', 'block', 'cadence', 'clock', 'lineage', 'combined',
    'donor', 'force', 'weights', 'field', 'u', 'position', 'phase', 'table_metadata'])
def test_pair_pic_timestep_rejects_unmatched_native_inputs(pair_pic_step_records, corrupt):
    from docs.scripts.compare_replays import _pair_native_hash, pair_pic_step_comparison, pair_repeat_comparison
    donor, replay = pair_pic_step_records
    record = json.loads((replay / 'run.json').read_text())
    if corrupt == 'runtime':
        record['jax'] = 'different'
    elif corrupt in ('block', 'cadence', 'lineage', 'combined', 'donor', 'table_metadata'):
        changes = dict(block=('block_steps', 10), cadence=('scalar_dt_omega0', .4),
                       lineage=('continuation', dict(prefix='unreviewed')),
                       combined=('table_replay', dict(table_every=2)),
                       donor=('initial_state_archive_sha256', '0' * 64),
                       table_metadata=('pic_dt_replay', dict(
                           record['settings']['pic_dt_replay'], times_sha256='0' * 64)))
        key, value = changes[corrupt]
        record['settings'][key] = value
    elif corrupt == 'clock':
        with np.load(replay / 'data.npz') as stored:
            raw = dict(stored)
        raw['t'][1] = np.nextafter(raw['t'][1], np.inf)
        np.savez_compressed(replay / 'data.npz', **raw)
    elif corrupt in ('force', 'weights', 'field', 'u', 'position', 'phase'):
        for name in ('initial_state.npz', 'final_state.npz'):
            with np.load(replay / name) as stored:
                state = dict(stored)
            key = dict(force='dark.amplitude', weights='w', field='E', u='u', position='x', phase='dark.phase')[corrupt]
            if corrupt == 'phase':
                state[key] = np.array(.01)
            elif corrupt == 'u':
                state[key] = np.roll(state[key], 2, axis=0)  # Global velocity moments survive this hidden change.
            elif corrupt in ('weights', 'field'):
                state[key].flat[0] = np.nextafter(state[key].flat[0], np.inf)  # Sub-tolerance, but not exact inputs.
            else:
                state[key].flat[0] += .01 if corrupt == 'position' else 1e-5
            np.savez_compressed(replay / name, **state)
            if name == 'initial_state.npz' and key in record['results']['initial_ordinary_fingerprints']:
                record['results']['initial_ordinary_fingerprints'][key] = _pair_native_hash(state[key])
    (replay / 'run.json').write_text(json.dumps(record))
    with pytest.raises(ValueError):
        pair_pic_step_comparison((donor, replay), () if corrupt == 'transition' else ('4' * 40, '5' * 40))
    with pytest.raises(ValueError, match='complete quadratic realized-force branches'):
        pair_repeat_comparison((replay, replay))


@pytest.fixture
def pair_extension_records(pair_pic_step_records):
    """Manufactured complete field lineage; the producer tests use a genuinely advanced state."""
    from docs.scripts.compare_replays import PAIR_MAXIMA, _pair_extension, _pair_native_hash
    fine, replay = pair_pic_step_records
    control = json.loads((fine.parent / 'run.json').read_text())
    coupled = fine.parent / 'coupled'
    donor = json.loads((coupled / 'run.json').read_text())
    with np.load(coupled / 'data.npz') as stored:
        origin = dict(stored)
    with np.load(fine / 'initial_state.npz') as stored:
        state = {key: stored[key] for key in stored.files if key not in ('dark.times', 'dark.amplitude', 'dark.phase')}
        times, force = stored['dark.times'], stored['dark.amplitude']
    s = dict(control['settings'], **donor['settings'])
    wp, field, energy = [s['normalization'][key] for key in ('omega0_rad_s', 'field_scale_V_m', 'energy_scale_J_m2')]
    d = float(origin['pump_mean_D'][-1])
    state.update(time=np.asarray(times[-1]), steps=np.asarray(len(times)-2, dtype=np.int32))
    state.update({'dark.' + key: np.asarray(value) for key, value in dict(
        mode='field', format=2, omega=s['dark_mass_over_omega0']*wp,
        E=np.tile([d, 0., 0.], (s['cells'], 1)), B=np.zeros_like(state['B']),
        A=np.zeros_like(state['B']), phi=np.zeros(s['cells']), work=0., initial_dark=.5*energy*(d/field)**2).items()})
    state['dark.initial_ordinary'] = origin['electric'][0] + origin['kinetic'][0].sum()
    origin['dark'][:] = state['dark.initial_dark']
    origin['balance'][:] = state['dark.initial_ordinary'] + state['dark.initial_dark']
    np.savez_compressed(coupled / 'data.npz', **origin)
    np.savez_compressed(coupled / 'final_state.npz', **state)
    extension = fine.parent / 'extension'
    extension.mkdir()
    np.savez_compressed(extension / 'initial_state.npz', **state)
    final = dict(state, time=np.asarray(float(times[-1]) + float(state['dark.dt'])),
                 steps=state['steps'] + np.asarray(1, dtype=np.int32))
    np.savez_compressed(extension / 'final_state.npz', **final)
    raw = {key: np.repeat(value[0:1], 2, axis=0) for key, value in origin.items() if not key.startswith('pump_')}
    raw.update(t=np.array([state['time'], final['time']]), mean_D=np.array([d, d]),
               dark=np.full(2, state['dark.initial_dark']), work=np.zeros(2),
               force_times=np.r_[times, float(times[-1])+float(state['dark.dt'])/2, final['time']],
               force_amplitude=np.concatenate((force, np.tile([d, 0., 0.], (2, 1)))))
    np.savez_compressed(extension / 'data.npz', **raw)
    maxima = donor['results']['all_step_maxima']
    en = state['dark.density'].sum()*1.602176634e-19
    length, c = float(state['dark.length']), 299792458.
    eps = energy/(length*field**2)
    units = [energy, energy/c, en*length, en*wp, en/eps, en/eps, en*length, energy, energy]
    results = dict(donor['results'], all_step_maxima_SI=(np.array([maxima[key] for key in PAIR_MAXIMA])*units).tolist())
    files = dict(run=coupled/'run.json', data=coupled/'data.npz', force_run=fine/'run.json',
                 force_initial=fine/'initial_state.npz', force_data=fine/'data.npz', control_run=fine.parent/'run.json',
                 control_data=fine.parent/'data.npz', coupled_final=coupled/'final_state.npz',
                 coupled_initial=coupled/'initial_state.npz')
    s['force_extension'] = dict(
        reviewed_source_transition=['4'*40, '5'*40], accepted_steps=1, native_scalar_dt_omega0=s['dt_omega0'],
        prefix_knots=len(times), prefix_times_sha256=_pair_native_hash(times),
        prefix_amplitude_sha256=_pair_native_hash(force), donor_sha256={
            key+'_sha256': hashlib.sha256(path.read_bytes()).hexdigest() for key, path in files.items()})
    record = dict(donor, git='5'*40, example='pair_waveform_force_extension', settings=s, results=results)
    (extension/'run.json').write_text(json.dumps(record))
    _, _, proof = _pair_extension(fine, extension, ('4'*40, '5'*40))
    replay_record = json.loads((replay/'run.json').read_text())
    replay_record['settings']['force_extension'] = proof['native_sha256']
    replay_record['settings']['pic_dt_replay'].update(times_sha256=_pair_native_hash(raw['force_times']),
                                                      amplitude_sha256=_pair_native_hash(raw['force_amplitude']))
    for name in ('initial_state.npz', 'final_state.npz'):
        with np.load(replay/name) as stored:
            leaves = dict(stored)
        leaves.update({'dark.times': raw['force_times'], 'dark.amplitude': raw['force_amplitude']})
        np.savez_compressed(replay/name, **leaves)
    (replay/'run.json').write_text(json.dumps(replay_record))
    return fine, replay, extension


@pytest.mark.parametrize('envelope, bound, accepted', [
    (False, 1., True), (True, 2., True), (False, 2., False), (True, .5, False)])
def test_pair_checkpoint_maxima_and_continuation_envelopes(envelope, bound, accepted):
    from docs.scripts.compare_replays import PAIR_MAXIMA, _pair_repeat_bounds
    units = np.arange(1., 10.)
    maxima = dict.fromkeys(PAIR_MAXIMA, bound)
    final = dict(zip(('dark.max_balance_error', 'dark.max_ordinary_gauss', 'dark.max_dark_gauss'), units[[0, 4, 5]]))
    results = dict(all_step_maxima=maxima, all_step_maxima_SI=(bound * units).tolist())
    if accepted:
        _pair_repeat_bounds(results, final, units, dict(balance=np.zeros(2)), True, envelope=envelope)
    else:
        with pytest.raises(ValueError, match='differ from native maxima'):
            _pair_repeat_bounds(results, final, units, dict(balance=np.zeros(2)), True, envelope=envelope)


def test_pair_extension_exact_checkpoint_metadata_is_separate_and_verified(pair_extension_records):
    from docs.scripts.compare_replays import _pair_extension
    fine, _, extension = pair_extension_records
    record = json.loads((extension / 'run.json').read_text())
    with np.load(extension / 'final_state.npz') as stored:
        record['results']['checkpoint_maxima_SI'] = [float(stored['dark.' + key]) for key in (
            'max_balance_error', 'max_ordinary_gauss', 'max_dark_gauss')]
    (extension / 'run.json').write_text(json.dumps(record))
    _pair_extension(fine, extension, ('4'*40, '5'*40))
    record['results']['checkpoint_maxima_SI'][0] += 1.
    (extension / 'run.json').write_text(json.dumps(record))
    with pytest.raises(ValueError, match='exact checkpoint maxima'):
        _pair_extension(fine, extension, ('4'*40, '5'*40))


def test_pair_force_extension_requires_opt_in_complete_prefix_and_qualified_execution(pair_extension_records):
    from docs.scripts.compare_replays import pair_pic_step_comparison
    fine, replay, extension = pair_extension_records
    with pytest.raises(ValueError):
        pair_pic_step_comparison((fine, replay), ('4'*40, '5'*40))
    result = pair_pic_step_comparison((fine, replay), ('4'*40, '5'*40), extension)
    assert not result['initialization']['fixed_force_byte_identical']
    assert result['initialization']['original_force_prefix_byte_identical']
    assert result['force_extension']['appended_knots'] == 2
    assert result['force_extension']['candidate_last_midpoint_s'] <= result['force_extension']['old_terminal_s']
    assert 'compile shapes changed' in result['notes'] and 'separately compiled' in result['force_extension']['claim']
    assert not result['original_gauss_gates'][0]['ordinary_gauss_over_en_eps0']


@pytest.mark.parametrize('corrupt', [
    'source', 'runtime', 'prefix', 'tail', 'clock', 'potential', 'work', 'reference', 'maxima', 'donor', 'checkpoint'])
def test_pair_force_extension_rejects_forged_lineage_or_native_tail(pair_extension_records, corrupt):
    from docs.scripts.compare_replays import _pair_extension
    fine, _, extension = pair_extension_records
    record = json.loads((extension/'run.json').read_text())
    if corrupt in ('source', 'runtime', 'maxima', 'donor'):
        if corrupt == 'source':
            record['git'] = '9'*40
        elif corrupt == 'runtime':
            record['numpy'] = 'different'
        elif corrupt == 'maxima':
            record['results']['all_step_maxima_SI'][0] = 1.
        else:
            record['settings']['force_extension']['donor_sha256']['coupled_final_sha256'] = '0'*64
        (extension/'run.json').write_text(json.dumps(record))
    else:
        name = ('data.npz' if corrupt in ('prefix', 'tail', 'clock') else
                'final_state.npz' if corrupt == 'checkpoint' else 'initial_state.npz')
        with np.load(extension/name) as stored:
            raw = dict(stored)
        key = dict(prefix='force_amplitude', tail='force_amplitude', clock='t', potential='dark.phi',
                   work='dark.work', reference='dark.initial_dark', checkpoint='dark.max_balance_error')[corrupt]
        index = -2 if corrupt == 'tail' else 0
        raw[key].flat[index] += -1. if corrupt == 'checkpoint' else 1.
        np.savez_compressed(extension/name, **raw)
    with pytest.raises(ValueError):
        _pair_extension(fine, extension, ('4'*40, '5'*40))
