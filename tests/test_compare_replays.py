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
    change_settings(records[1], momentum_seed_over_sigma_e=.05, seed_mode=16, seed_phase=0.,
                    thermal_temperature_over_mec2=[.001, .001],
                    initial_rms_over_c=np.sqrt(np.array([.001, .001 / 1836])).tolist())
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
