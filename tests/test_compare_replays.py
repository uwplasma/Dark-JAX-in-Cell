"""Native-record guards reject mismatched clocks, units and controlled loading."""
import json

import numpy as np
import pytest

from docs.scripts.compare_replays import _projection, compare_replays, constraint_audit, fingerprint, metrics


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
    assert str(records[0]) not in json.dumps(result, allow_nan=False)
    assert fingerprint(np.zeros(2)) != fingerprint(np.zeros((1, 2)))
    assert metrics(np.zeros(3), np.ones(3))['relative_l2_difference'] is None


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
