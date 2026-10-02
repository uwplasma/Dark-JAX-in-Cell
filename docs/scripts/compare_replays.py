"""Compare native paper replays at equal horizons without shifting or interpolating."""
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

# Edit these inputs; batch studies may pass the same names through runpy.init_globals.
first = Path(globals().get('first', 'artifacts/paper_fixed_first'))
second = Path(globals().get('second', 'artifacts/paper_fixed_repeat'))
variant = globals().get('variant', 'repeat')
implicit = globals().get('implicit', False)
picard = globals().get('picard', False)
method_controls = globals().get('method_controls', False)
phase_controls = globals().get('phase_controls', ())  # Native implicit folders, grouped by mesh and phase.
pair_controls = globals().get('pair_controls', ())  # Complete five-branch pair-waveform producer folders.
pair_repeats = globals().get('pair_repeats', ())  # Two exact-state prescribed branch folders; JSON only.
pair_repeat_observer = globals().get('pair_repeat_observer', None)
pair_repeat_donor = globals().get('pair_repeat_donor', None)
pair_repeat_transition = globals().get('pair_repeat_transition', ())  # Explicit reviewed (donor SHA, repeat SHA).
pair_tables = globals().get('pair_tables', ())  # Original coarse, decimated replay, original fine; JSON only.
pair_table_transition = globals().get('pair_table_transition', ())  # Reviewed (donor SHA, replay SHA).
pair_pic_steps = globals().get('pair_pic_steps', ())  # Fine donor, changed PIC dt at exactly the same SI force.
pair_pic_transition = globals().get('pair_pic_transition', ())  # Reviewed (donor SHA, replay SHA).
pair_pic_extension = globals().get('pair_pic_extension', None)  # Explicit one accepted coupled-step force tail.
ensemble = globals().get('ensemble', ())  # Ordered (coarse, half-step) native folders for seeds 0/1/2 or 0–4.
ensemble_window = globals().get('ensemble_window', (800, 1000))
continuation = globals().get('continuation', False)  # Explicit timestep-only lineage audit.
continuation_donors = globals().get('continuation_donors', ((), ()))  # Oldest-first folders for each arm.
continuation_transitions = globals().get('continuation_transitions', ())  # Reviewed (old SHA, new SHA) pairs.
constraints = globals().get('constraints', False)
legacy = globals().get('legacy', False)
refined = globals().get('refined', None)
refined_against = globals().get('refined_against', 'first')
finer_step = globals().get('finer_step', None)  # Third timestep at fixed physics; keeps four plotted controls.
finer_mesh = globals().get('finer_mesh', None)
refined_variant = globals().get('refined_variant', 'dt')
loading_refined = globals().get('loading_refined', None)
loading_repeat = globals().get('loading_repeat', None)
orbit_audits = globals().get('orbit_audits', ())
destination = globals().get('destination', None)  # folder for figure, arrays and record
output = Path(globals().get(
    'output', 'docs/_static/figures/pair_waveform_long/table_resolution.json'
    if pair_tables else 'artifacts/replay_comparison.json'))

WINDOWS = ((0, 100), (0, 250), (0, 500), (0, 1000), (800, 1000))
SERIES = ('mean_E', 'rms', 'electric', 'nonzero_electric', 'density_rms')
LOCAL_SERIES = ('local_spread', 'local_density_rms')
PARAMETERS = ('cells', 'particles_per_species', 'seed', 'dt_omega_p', 'output_dt_omega_p',
              'length_c_over_omega_p', 'mass_ratio', 'T_each_over_mec2', 'coupling',
              'drive_quiver_over_sigma', 'force_quiver_over_c', 'loading', 'pusher', 'shape', 'parent_revision')
VARIANTS = dict(repeat=None, dt='dt_omega_p', seed='seed', mesh='cells', loading='particles_per_species', shape='shape')
ENSEMBLE_OBSERVABLES = ('work', 'electron_local_2D', 'electron_local_4D', 'ion_local_2D', 'ion_local_4D',
                        'electric', 'nonzero_electric', 'injection_rate')
ENSEMBLE_BOUNDS = (.02, .02, .02, .02, .02, .05, .05, 1e-5)
PAIR_CASES = ('coupled', 'realized_coarse', 'realized_fine', 'homogeneous_coarse', 'homogeneous_fine')
PAIR_PHYSICS = ('parent_revision', 'shape_order', 'seed_mode', 'eta', 'dark_mass_over_omega0',
                'force_quiver_over_c', 'pair_loading', 'length_c_over_omega0', 'waterbag_full_width_over_c',
                'normalization', 'local_spread_lengths_c_over_omega0', 'quadrature_nodes',
                'accuracy_targets', 'XLA_FLAGS')
PAIR_RUNTIME = ('git', 'jax', 'jaxincell', 'numpy', 'python', 'platform', 'backend', 'jax_enable_x64')


def fingerprint(array):
    """SHA256 of dtype, shape and native contiguous bytes; never a rounded curve."""
    array = np.asarray(array)
    header = json.dumps(dict(dtype=array.dtype.str, shape=array.shape), sort_keys=True).encode()
    return hashlib.sha256(header + np.ascontiguousarray(array).tobytes()).hexdigest()


def _load(folder, tolerance):
    folder = Path(folder)
    record = json.loads((folder / 'run.json').read_text())
    with np.load(folder / 'data.npz', allow_pickle=False) as stored:
        data = {key: stored[key] for key in stored.files}
    time, settings = data['t'], record['settings']
    if time.ndim != 1 or len(time) < 2 or time[0] != 0 or np.any(np.diff(time) <= 0):
        raise ValueError('native clocks must begin at zero and increase')
    if not all(np.all(np.isfinite(data[key])) for key in ('t', *SERIES, 'spread', 'mean', 'kinetic')):
        raise ValueError('native scalar histories must be finite')
    if any(data[key].shape != time.shape or not np.all(np.isfinite(data[key])) for key in ('work', 'balance')):
        raise ValueError('work and balance must be finite native scalar histories')
    if not np.allclose(np.diff(time), settings['output_dt_omega_p'], rtol=0, atol=tolerance):
        raise ValueError('native cadence differs from recorded physical cadence')
    if abs(time[-1] - settings['horizon_omega_p']) > tolerance:
        raise ValueError('native horizon differs from its record')
    _normalization(data, settings)
    scales = settings.get('local_spread_lengths_c_over_wp', [])
    for key in LOCAL_SERIES:
        if key in data and (data[key].shape != (len(time), 2, len(scales))
                            or not np.all(np.isfinite(data[key]))):
            raise ValueError('local moments require finite species/physical-scale histories')
    return (record, data, hashlib.sha256((folder / 'data.npz').read_bytes()).hexdigest(),
            hashlib.sha256((folder / 'run.json').read_bytes()).hexdigest())


def _normalization(data, settings):
    """Anchor energy/field units to the physical RMS loading and native work ledger."""
    mass = np.array([1., settings['mass_ratio']])
    if not np.allclose(data['spread'], .5 * mass * data['rms']**2, rtol=2e-12, atol=1e-30):
        raise ValueError('spread and RMS do not share the stated energy normalization')
    if settings.get('momentum_seed_over_sigma_e', 0):
        temperature = np.asarray(settings.get('thermal_temperature_over_mec2', []))
        initial = np.asarray(settings.get('initial_rms_over_c', []))
        if (temperature.shape != (2,) or initial.shape != (2,)
                or not np.allclose(temperature, settings['T_each_over_mec2'], rtol=2e-12, atol=0)
                or not np.allclose(data['rms'][0], initial, rtol=2e-12, atol=0)):
            raise ValueError('seeded initial RMS and reconstructed thermal loading must be recorded')
    elif not np.allclose(mass * data['rms'][0]**2, settings['T_each_over_mec2'], rtol=2e-12, atol=0):
        raise ValueError('initial RMS differs from the physical temperature loading')
    if not np.allclose(data['nonzero_electric'], data['electric'] - data['mean_E']**2 / 2,
                       rtol=2e-12, atol=1e-30):
        raise ValueError('electric components do not share the stated field normalization')
    if all(key in data for key in ('balance', 'magnetic', 'dark', 'work')):
        source = data['work'] if settings['coupling'] is None else 0
        expected = data['electric'] + data['magnetic'] + data['dark'] + data['kinetic'].sum(axis=1) - source
        if not np.allclose(data['balance'], expected, rtol=2e-12, atol=1e-30):
            raise ValueError('energy/work ledger does not share one normalization')


def _hash_matches(rows, keys):
    matches = {}
    for key in keys:
        values = [row.get(key) for row in rows]
        if any(value is not None and (len(value) != 64 or not set(value) <= set('0123456789abcdef'))
               for value in values):
            raise ValueError('initial fingerprints must be SHA256 strings')
        matches[key] = None not in values and values[0] == values[1]
    return matches


def _initial(first, second, variant, legacy):
    fingerprints = [settings.get('initial_fingerprints') for settings in (first, second)]
    if not all(fingerprints):
        if not legacy:
            raise ValueError('initial fingerprints are required; legacy records cannot verify exact loading')
        return dict(verified=False, reason='legacy records lack initial particle fingerprints')
    states = [row['state'] for row in fingerprints]
    ordinary = {'x', 'u', 'w', 'E', 'B', 'rho'}
    if any(not ordinary <= row.keys() for row in states):
        raise ValueError('initial fingerprints lack required ordinary state keys')
    if variant == 'repeat' and states[0].keys() != states[1].keys():
        raise ValueError('repeat initial state fingerprint key sets disagree')
    state_keys = sorted(states[0].keys() | states[1].keys())
    matches = dict(loading=_hash_matches([row['loading'] for row in fingerprints], ('x', 'v')),
                   state=_hash_matches(states, state_keys))
    required = [matches['loading']['x'], matches['state']['w']]
    if variant != 'seed':
        required.append(matches['loading']['v'])
    if variant == 'repeat':
        required.extend(matches['state'].values())
    if variant != 'loading' and not all(required):
        raise ValueError('initial fingerprints disagree for this controlled comparison')
    return dict(verified=True, matches=matches)


def metrics(first, second):
    """Absolute differences remain available when a reference norm vanishes."""
    norm = np.linalg.norm(first, axis=0)
    difference = np.linalg.norm(second - first, axis=0)
    relative = np.divide(difference, norm, out=np.full_like(difference, np.nan), where=norm > 0)
    return dict(l2_difference=difference.tolist(), reference_l2_norm=norm.tolist(),
                relative_l2_difference=np.where(np.isfinite(relative), relative, None).tolist(),
                max_absolute_difference=np.max(abs(second - first), axis=0).tolist(),
                reference_mean=np.mean(first, axis=0).tolist(), comparison_mean=np.mean(second, axis=0).tolist())


def window_summary(data, selected, coupled=False):
    """Reduce within one realization; correlated output samples are not replicates."""
    time = data['t'][selected]
    if len(time) < 2:
        raise ValueError('window injection rates require two native samples')
    work = (-1 if coupled else 1) * (data['work'][selected][-1] - data['work'][selected][0])
    plasma_energy = float(np.mean((data['electric'] + data['magnetic'] + data['kinetic'].sum(axis=1))[selected]))
    if not np.isfinite(plasma_energy) or plasma_energy <= 0:
        raise ValueError('window injection rates require positive finite plasma energy')
    result = dict(work_increment=float(work),
                  injection_rate_over_wp=float(work / ((time[-1] - time[0]) * plasma_energy)),
                  plasma_mean_energy=plasma_energy,
                  electric_mean=float(np.mean(data['electric'][selected])),
                  nonzero_electric_mean=float(np.mean(data['nonzero_electric'][selected])))
    for key in ('kinetic', 'spread', 'local_spread'):
        if key in data:
            result[f'{key}_increment_mean'] = np.mean(data[key][selected] - data[key][0], axis=0).tolist()
    return result


def _controls(records, data, variant, tolerance):
    """Equal horizons/cadences preserve sampling and compile-allocation controls."""
    settings = [record['settings'] for record in records]
    if any('continuation' in setting for setting in settings):
        raise ValueError('mixed-producer continuation requires a separate lineage audit')
    controls = [{'shape_order': 2, 'XLA_FLAGS': '', **row} for row in settings]
    allowed = (VARIANTS[variant], 'shape_order' if variant == 'shape' else None)
    for key in (*PARAMETERS, 'shape_order', 'XLA_FLAGS'):
        if key not in allowed and controls[0][key] != controls[1][key]:
            raise ValueError(f'controlled comparisons must share {key}')
    for key in ('momentum_seed_over_sigma_e', 'seed_mode', 'seed_phase'):
        default = 0. if key == 'momentum_seed_over_sigma_e' else None
        if settings[0].get(key, default) != settings[1].get(key, default):
            raise ValueError(f'controlled comparisons must share physical seed {key}')
    for key in ('git', 'jax', 'jaxincell', 'numpy', 'jax_enable_x64', 'backend'):
        if records[0][key] != records[1][key]:
            raise ValueError(f'controlled comparisons must share runtime/source {key}')
    if (data[0]['t'].shape != data[1]['t'].shape
            or not np.allclose(data[0]['t'], data[1]['t'], rtol=0, atol=tolerance)):
        raise ValueError('equal native horizons and clocks are required')
    _protocol(settings, variant, tolerance)
    return settings


def _protocol(settings, variant, tolerance):
    for key in ('block_horizon_omega_p', 'local_moments_output_dt_omega_p', 'local_spread_lengths_c_over_wp',
                'normalization'):
        if settings[0].get(key) != settings[1].get(key):
            raise ValueError(f'comparisons must share physical {key}')
    if variant != 'dt' and settings[0].get('block_steps') != settings[1].get('block_steps'):
        raise ValueError('repeat/control compilations must share block_steps')
    for setting in settings:
        if setting.get('block_steps') is not None:
            if abs(setting['block_steps'] * setting['dt_omega_p'] - setting['block_horizon_omega_p']) > tolerance:
                raise ValueError('block steps and physical block horizon disagree')


def _projection(E, residual, dx, normalization):
    """Mean-preserving longitudinal closure; the incompatible mean remains visible."""
    delta = -dx * np.cumsum(residual - np.mean(residual))
    delta -= np.mean(delta)
    eps = normalization['epsilon0_F_m']
    charge_scale = normalization['charge_density_C_m3'] / eps
    after = residual + (delta - np.roll(delta, 1)) / dx
    return dict(mean_residual_over_charge_scale=float(np.mean(residual) / charge_scale),
                max_residual_over_charge_scale=float(np.max(abs(residual)) / charge_scale),
                max_remaining_residual_over_charge_scale=float(np.max(abs(after)) / charge_scale),
                max_correction_over_field_scale=float(np.max(abs(delta)) / normalization['field_scale_V_m']),
                energy_change_over_scale=float(eps * dx * np.sum(E[:, 0] * delta + .5 * delta**2)
                                               / normalization['energy_scale_J_m2']))


def constraint_audit(folder, record):
    """Quantify an endpoint closure correction without applying it to any state."""
    normalization = record['settings'].get('normalization')
    if normalization is None:
        return dict(available=False, reason='actual imported normalization constants were not recorded')
    path = Path(folder) / 'final_state.npz'
    if not path.is_file():
        return dict(available=False, reason='no native final archive')
    with np.load(path, allow_pickle=False) as state:
        settings = record['settings']
        if (int(state['cells']) != settings['cells']
                or abs(float(state['time']) * normalization['omega_p_rad_s'] - settings['horizon_omega_p']) > 1e-5):
            raise ValueError('endpoint archive does not match the native record horizon/grid')
        dx = float(state['dark.length']) / int(state['cells'])
        rho = state['rho'] + state['dark.background']
        eps, c = normalization['epsilon0_F_m'], normalization['c_m_s']
        E = state['E']
        residual = (E[:, 0] - np.roll(E[:, 0], 1)) / dx - rho / eps
        result = dict(available=True, ordinary=_projection(E, residual, dx, normalization))
        if str(state['dark.mode']) == 'field':
            E, phi, omega, eta = [state['dark.' + key] for key in ('E', 'phi', 'omega', 'eta')]
            residual = (E[:, 0] - np.roll(E[:, 0], 1)) / dx + omega**2 / c**2 * phi - eta * rho / eps
            result['dark'] = _projection(E, residual, dx, normalization)
            result['dark']['unchanged_phi_energy_over_scale'] = float(
                eps * dx * omega**2 / (2 * c**2) * np.sum(phi**2) / normalization['energy_scale_J_m2'])
            result['dark']['phi_energy_change'] = 0.
    result.update(archive_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  note='Only a mean-zero longitudinal diagnostic. Particles, potentials and fields remain unchanged; '
                       'the residual mean cannot be corrected by periodic divergence. Endpoint size does not '
                       'bound dynamical amplification of earlier roundoff.')
    return result


def compare_replays(first, second, variant='repeat', windows=WINDOWS, legacy=False, tolerance=1e-5,
                    constraints=False):
    """Change one declared parameter; validate native clocks, loading and units first."""
    if variant not in VARIANTS:
        raise ValueError('unknown comparison variant')
    a, b = [_load(folder, tolerance) for folder in (first, second)]
    records, data = [a[0], b[0]], [a[1], b[1]]
    settings = _controls(records, data, variant, tolerance)
    initial = _initial(*settings, variant, legacy)
    time = data[0]['t']
    selected_windows = []
    for start, end in windows:
        if start < -tolerance or start >= end or end > time[-1] + tolerance:
            raise ValueError('comparison windows must fit the native horizon')
        selected = (time >= start - tolerance) & (time <= end + tolerance)
        if not np.any(selected):
            raise ValueError('comparison windows need native samples')
        series = [*SERIES, *[key for key in LOCAL_SERIES if key in data[0] and key in data[1]]]
        selected_windows.append(dict(
            window_omega_p=[start, end], samples=int(selected.sum()),
            realization_summaries=[window_summary(row, selected, setting['coupling'] is not None)
                                   for row, setting in zip(data, settings)],
            observables={key: metrics(data[0][key][selected], data[1][key][selected]) for key in series}))
    endpoint = {key: dict(first=data[0][key].tolist(), second=data[1][key].tolist(),
                          difference=(data[1][key] - data[0][key]).tolist())
                for key in ('local_spread_initial', 'local_spread_final') if key in data[0] and key in data[1]}
    result = dict(
        variant=variant, initial_fingerprints=initial, windows=selected_windows,
        maximum_clock_difference=float(np.max(abs(data[0]['t'] - data[1]['t']))),
        fixed_scale_endpoints=endpoint, physical_scales_c_over_wp=settings[0].get('local_spread_lengths_c_over_wp'),
        scalar_fingerprints={key: dict(initial=[fingerprint(row[key][0]) for row in data],
                                       history=[fingerprint(row[key]) for row in data])
                             for key in SERIES},
        sources=[dict(git=record['git'], parent=setting['parent_revision'], data_sha256=source[2], run_sha256=source[3],
                      varied_parameter=setting.get(VARIANTS[variant]),
                      block_steps=setting.get('block_steps'), block_horizon=setting.get('block_horizon_omega_p'))
                 for record, setting, source in zip(records, settings, (a, b))],
        native_all_step_maxima=[
            {key: value for key, value in record['results'].items()
             if key.startswith('max_') and key != 'max_speed_over_c'} for record in records],
        notes=('Native aligned samples, no interpolation, time shifts or curve fitting. Density RMS is at the '
               'numerical grid scale; local moments use the recorded physical Gaussian lengths. Fingerprint equality '
               'requires the same dtype/runtime. One repeat is not an uncertainty estimate; field saturation and '
               'small conservation defects do not establish convergence. Null relative errors have zero '
               'reference norm.'))
    if constraints:
        result['endpoint_constraints'] = [constraint_audit(folder, record)
                                          for folder, record in zip((first, second), records)]
    return result


CONTINUATION_WINDOWS = ((800, 1000), (4800, 5000))
PAPER_MAXIMA = ('max_energy_work_defect_over_initial_thermal', 'max_momentum_defect_over_nmecL',
                'max_particle_charge_change_over_enL', 'max_continuity_over_enwp',
                'max_ordinary_gauss_over_en_eps0', 'max_dark_gauss_over_en_eps0',
                'max_grid_charge_change_over_enL', 'max_dark_work_defect_over_nmc2L',
                'max_ordinary_work_defect_over_nmc2L')


def _archive(path, legacy_shape=None, field=False):
    """Read complete prescribed restarts without JAX or trusting ZIP recompression."""
    import zipfile
    with zipfile.ZipFile(path) as archive:
        if archive.testzip() or len(archive.namelist()) != len(set(archive.namelist())):
            raise ValueError('continuation archive CRC or duplicate leaf failure')
    with np.load(path, allow_pickle=False) as stored:
        state = dict(stored)
    if 'shape_order' not in state and legacy_shape == 2:
        state['shape_order'] = np.asarray(2)  # Complete older native restarts used quadratic shapes.
    required = {'format', 'E', 'B', 'x', 'u', 'w', 'qm', 'rho', 'sigma', 'key', 'time', 'steps',
                'names', 'counts', 'cells', 'algorithm', 'shape_order'}
    required.update('wall.' + key for key in ('arrived', 'collected', 'injected', 'energy_in', 'energy_out',
                                              'energy_injected', 'momentum', 'momentum_injected', 'truncated',
                                              'overflow'))
    required.update('dark.' + key for key in ('format', 'mode', 'omega', 'eta', 'cells', 'length', 'length_y',
                                              'length_z', 'dt', 'relativistic', 'mass', 'charge', 'density',
                                              'has_external_B', 'background', 'work',
                                              'initial_ordinary', 'initial_dark', 'initial_projection_norm',
                                              'max_balance_error', 'max_ordinary_gauss', 'max_dark_gauss'))
    required.update('dark.' + key for key in (('E', 'B', 'A', 'phi') if field else ('amplitude', 'phase')))
    if (not required <= state.keys() or any(not np.isfinite(value).all() for value in state.values()
                                            if value.dtype.kind in 'biufc')
            or any(value.dtype.itemsize != 8 for value in state.values() if value.dtype.kind == 'f')):
        raise ValueError('continuation needs complete finite native restart leaves')
    return state


def _same_leaves(first, second):
    if first.keys() != second.keys() or any(fingerprint(first[key]) != fingerprint(second[key]) for key in first):
        raise ValueError('continuation complete restart leaves disagree')


def _accepted_ticks(dt, steps, stride):
    time, ticks = 0., [0.]
    for step in range(1, steps + 1):
        time += dt
        if step % stride == 0:
            ticks.append(time)
    return np.asarray(ticks)


def _continuation_units(data, setting):
    """Independently convert the producer's SI segment without altering native input."""
    n = setting['normalization']
    energy, field, speed = [n[key] for key in ('energy_scale_J_m2', 'field_scale_V_m', 'c_m_s')]
    divisors = dict.fromkeys(('electric', 'magnetic', 'dark', 'dark_coherent', 'kinetic', 'spread',
                             'balance', 'work', 'local_spread'), energy)
    divisors.update(dict.fromkeys(('mean', 'rms', 'max_speed'), speed))
    divisors.update(dict.fromkeys(('mean_E', 'mode_E', 'dark_mode_E'), field))
    divisors.update(momentum=energy / speed)
    converted = {key: value / divisors[key] if key in divisors else value.copy() for key, value in data.items()}
    converted['t'] = data['t'] * n['omega_p_rad_s']
    converted['nonzero_electric'] = converted['electric'] - converted['mean_E']**2 / 2
    return converted


def _continuation_snapshot(state, setting):
    """Independent longitudinal relativistic energy/work and weighted moments."""
    n = setting['normalization']
    c, eps = [n[key] for key in ('c_m_s', 'epsilon0_F_m')]
    counts, mass = state['counts'], state['dark.mass']
    charge = state['dark.charge'] * n['charge_density_C_m3'] / state['dark.density'][0]
    u, weights = state['u'], state['w']
    gamma = np.sqrt(1 + np.sum((u / c)**2, axis=1))
    velocity = u[:, 0] / gamma
    groups = np.split(np.arange(len(weights)), np.cumsum(counts)[:-1])
    mean = np.array([np.average(velocity[g], weights=weights[g]) for g in groups])
    variance = np.array([np.average((velocity[g] - mean[i])**2, weights=weights[g])
                         for i, g in enumerate(groups)])
    kinetic = np.array([np.sum(weights[g] * mass[i] * np.sum(u[g]**2, axis=1) / (gamma[g] + 1))
                        for i, g in enumerate(groups)])
    dx = float(state['dark.length']) / int(state['cells'])
    electric = eps * dx * np.sum(state['E']**2) / 2
    result = dict(t=state['time'], electric=electric, magnetic=0., dark=0., kinetic=kinetic,
                  mean=mean, rms=np.sqrt(variance), work=state['dark.work'],
                  spread=.5 * mass * np.array([weights[g].sum() for g in groups]) * variance,
                  momentum=np.sum(np.repeat(mass, counts)[:, None] * weights[:, None] * u, axis=0),
                  charge=np.sum(np.repeat(charge, counts) * weights), grid_charge=dx * np.sum(state['rho']),
                  mean_E=np.mean(state['E'][:, 0]), balance=electric + kinetic.sum() - state['dark.work'])
    result.update(ordinary_gauss=np.max(abs((state['E'][:, 0] - np.roll(state['E'][:, 0], 1)) / dx
                                            - (state['rho'] + state['dark.background']) / eps)),
                  dark_gauss=0., mean_D=0., mean_A=0., dark_coherent=0., dark_mode_E=0j,
                  mode_E=np.fft.fft(state['E'][:, 0])[setting['recorded_mode']] / len(state['E']),
                  max_speed=np.max(abs(velocity)))
    return {key: value[0] for key, value in _continuation_units(
        {key: np.asarray(value)[None] for key, value in result.items()}, setting).items()}


def _continuation_model(state, setting):
    n = setting['normalization']
    wp, c, energy, field, charge, eps = [n[key] for key in (
        'omega_p_rad_s', 'c_m_s', 'energy_scale_J_m2', 'field_scale_V_m', 'charge_density_C_m3', 'epsilon0_F_m')]
    mass, density, length = state['dark.mass'], state['dark.density'], float(state['dark.length'])
    if (not np.isfinite([wp, c, energy, field, charge, eps]).all() or min(wp, c, energy, field, charge, eps) <= 0
            or mass.shape != (2,) or density.shape != (2,) or mass[0] <= 0 or density[0] <= 0
            or not np.array_equal(state['dark.charge'], [-1, 1]) or density[0] != density[1]
            or not np.allclose([mass[1] / mass[0], length * wp / c, density[0] * mass[0] * c**2 * length,
                                mass[0] * c * wp * density[0] / charge, charge**2 / (density[0] * mass[0] * eps)],
                               [setting['mass_ratio'], setting['length_c_over_omega_p'], energy, field, wp**2],
                               rtol=2e-12, atol=0)
            or state['qm'].shape != state['w'].shape
            or not np.allclose(state['qm'], np.repeat(state['dark.charge'] / mass, state['counts'])
                               * charge / density[0], rtol=2e-12, atol=0)
            or not np.allclose(np.add.reduceat(state['w'], [0, int(state['counts'][0])]), density * length,
                               rtol=2e-12, atol=0)):
        raise ValueError('continuation native masses, charge, weights and units disagree')


def _continuation_state_checks(states, record, data):
    setting, n = record['settings'], record['settings']['normalization']
    cells, count = setting['cells'], setting['particles_per_species']
    wp = n['omega_p_rad_s']
    for state, index in zip(states, (0, -1)):
        if (int(state['format']) != 2 or int(state['dark.format']) != 2
                or str(state['algorithm']) != 'explicit' or str(state['dark.mode']) != 'drive'
                or float(state['dark.omega']) != wp or float(state['dark.eta']) != 1.
                or float(state['dark.phase']) != 0 or bool(state['dark.has_external_B'])
                or not bool(state['dark.relativistic']) or int(state['cells']) != cells
                or int(state['dark.cells']) != cells or int(state['shape_order']) != setting['shape_order']
                or float(state['dark.dt']) != setting['dt_omega_p'] / wp
                or not np.array_equal(state['counts'], [count, count])
                or state['x'].shape != (2 * count, 3) or state['u'].shape != state['x'].shape
                or state['w'].shape != (2 * count,) or state['E'].shape != (cells, 3)
                or state['rho'].shape != (cells,) or state['B'].shape != state['E'].shape
                or np.any(state['B']) or np.any(state['E'][:, 1:])
                or np.any(state['u'][:, 1:]) or 'dark.times' in state):
            raise ValueError('continuation native model/loading/longitudinal shapes disagree')
        _continuation_model(state, setting)
        if not np.array_equal(state['dark.amplitude'], [setting['force_quiver_over_c'] * n['field_scale_V_m'], 0, 0]):
            raise ValueError('continuation native forcing phase/amplitude disagree')
        sample = _continuation_snapshot(state, setting)
        scales = {'charge': n['charge_density_C_m3'] * float(state['dark.length']),
                  'grid_charge': n['charge_density_C_m3'] * float(state['dark.length']),
                  'ordinary_gauss': n['charge_density_C_m3'] / n['epsilon0_F_m']}
        if any(key not in data or not np.allclose(value, data[key][index], rtol=2e-12,
                                                  atol=2e-12 * scales.get(key, 1.)) for key, value in sample.items()):
            raise ValueError('continuation restart endpoints and normalized scalars disagree')
    origin, final = states
    if (float(origin['time']) != 0 or int(origin['steps']) != 0 or float(origin['dark.work']) != 0
            or np.any(origin['w'] <= 0) or not np.array_equal(origin['w'], final['w'])):
        raise ValueError('continuation global origin or particle weights changed')
    fixed = set(origin) - {'E', 'x', 'u', 'rho', 'key', 'time', 'steps', 'dark.work',
                           'dark.max_balance_error', 'dark.max_ordinary_gauss', 'dark.max_dark_gauss'}
    for key in fixed:
        if not np.array_equal(origin[key], final[key]):
            raise ValueError('continuation global reference changed')
    if (not np.isclose(float(origin['dark.initial_ordinary']) / n['energy_scale_J_m2'], data['balance'][0],
                       rtol=2e-12, atol=0) or float(origin['dark.initial_dark']) != 0):
        raise ValueError('continuation original energy reference disagrees')
    hashes = {key: hashlib.sha256(f'{origin[key].dtype.str}:{origin[key].shape}'.encode()
                                  + np.ascontiguousarray(origin[key]).tobytes()).hexdigest()
              for key in ('x', 'u', 'w', 'E', 'B', 'rho')}
    if hashes != setting['initial_fingerprints']['state']:
        raise ValueError('continuation origin fingerprints disagree')
    _continuation_clock(final, setting, data)


def _continuation_clock(final, setting, data):
    stride = round(setting['output_dt_omega_p'] / setting['dt_omega_p'])
    if (stride < 1 or not np.issubdtype(final['steps'].dtype, np.integer) or int(final['steps']) <= 0
            or type(setting['block_steps']) is not int or setting['block_steps'] < 1
            or setting['block_steps'] % stride or int(final['steps']) % setting['block_steps']):
        raise ValueError('continuation needs complete integer blocks and native sampling')
    ticks = _accepted_ticks(float(final['dark.dt']), int(final['steps']), stride)
    if (not np.array_equal(data['t'], ticks * setting['normalization']['omega_p_rad_s'])
            or float(final['time']) != ticks[-1]):
        raise ValueError('continuation accepted clocks disagree')


def _continuation_bounds(prefix, record, start, final):
    n = record['settings']['normalization']
    charge = n['charge_density_C_m3']
    length = float(final['dark.length'])
    energy, speed, wp = [n[key] for key in ('energy_scale_J_m2', 'c_m_s', 'omega_p_rad_s')]
    scales = np.array([.001 * energy, energy / speed, charge * length, charge * wp,
                       charge / n['epsilon0_F_m'], charge / n['epsilon0_F_m'], charge * length, energy, energy])
    prior = np.array([prefix['results'][key] for key in PAPER_MAXIMA]) * scales
    native = np.array([final['dark.max_balance_error'], final['dark.max_ordinary_gauss'],
                       final['dark.max_dark_gauss']])
    maximum = np.asarray(record['results']['all_step_maxima_SI'])
    if (prior.shape != (9,) or maximum.shape != (9,) or not np.isfinite([prior, maximum]).all()
            or np.any(prior < 0) or np.any(maximum < prior)
            or not np.allclose(maximum / scales, [record['results'][key] for key in PAPER_MAXIMA],
                               rtol=2e-12, atol=0)
            or not np.allclose(maximum[[0, 4, 5]], native, rtol=2e-12, atol=0)):
        raise ValueError('continuation global all-step maxima disagree')
    restored = np.asarray(record['results']['reconstructed_prior_maxima_SI'])
    if (restored.shape != (9,) or not np.isfinite(restored).all() or np.any(restored < prior)
            or np.any(maximum < restored)
            or np.any(restored[[0, 4, 5]] < [start['dark.max_balance_error'], start['dark.max_ordinary_gauss'],
                                             start['dark.max_dark_gauss']])):
        raise ValueError('continuation inherited maxima were reset')
    return dict(prefix_normalized=prefix['results'], carried_global_SI=maximum.tolist(),
                reconstructed_prior_SI=restored.tolist(), segment_only_maxima_available=False)


def _continuation_source(old, record, transitions, tolerance):
    """Require an explicitly reviewed source transition and unchanged physical inputs."""
    setting, lineage = record['settings'], record['settings'].get('continuation', {})
    for value, size in ((old['git'], 40), (record['git'], 40), (setting['parent_revision'], 40),
                        (lineage.get('producer_script_sha256', ''), 64)):
        if not isinstance(value, str) or len(value) != size or not set(value) <= set('0123456789abcdef'):
            raise ValueError('continuation producers require complete clean source hashes')
    if (record.get('jax_enable_x64') is not True or (old['git'], record['git']) not in transitions
            or record.get('example') != 'paper_resonant_continuation'
            or lineage.get('prefix_native_git') != old['git']
            or lineage.get('prefix_lineage') != old['settings'].get('continuation')):
        raise ValueError('continuation reviewed source vector or nested prefix lineage disagrees')
    required = ('shape_order', 'XLA_FLAGS', 'recorded_mode', 'normalization', 'block_steps',
                'block_horizon_omega_p', 'local_moments_output_dt_omega_p', 'local_spread_lengths_c_over_wp')
    if any(key not in row for row in (old['settings'], setting) for key in required):
        raise ValueError('continuation requires the complete native protocol and units')
    for key in (*PARAMETERS, 'shape_order', 'XLA_FLAGS', 'recorded_mode', 'momentum_seed_over_sigma_e',
                'seed_mode', 'seed_phase'):
        if setting.get(key) != old['settings'].get(key):
            raise ValueError(f'continuation changed physical {key}')
    for key in PAIR_RUNTIME[1:]:
        if record.get(key) is None or record[key] != old.get(key):
            raise ValueError(f'continuation changed runtime {key}')
    _protocol([old['settings'], setting], 'repeat', tolerance)
    _initial(old['settings'], setting, 'repeat', False)


def _continuation_boundary(prefix, segment, setting):
    n = setting['normalization']
    units = dict(charge=n['charge_density_C_m3'] * setting['length_c_over_omega_p'] * n['c_m_s'] / n['omega_p_rad_s'])
    units.update(grid_charge=units['charge'], ordinary_gauss=n['charge_density_C_m3'] / n['epsilon0_F_m'],
                 dark_gauss=n['charge_density_C_m3'] / n['epsilon0_F_m'], mean_D=n['field_scale_V_m'],
                 mean_A=n['field_scale_V_m'] / n['omega_p_rad_s'])
    if any(not np.allclose(value[0], prefix[key][-1], rtol=2e-12, atol=2e-12 * units.get(key, 1.))
           for key, value in segment.items()):
        raise ValueError('continuation duplicated boundary scalars disagree with the native endpoint tolerance')
    return dict(bitwise_equal=all(np.array_equal(value[0], prefix[key][-1]) for key, value in segment.items()),
                rtol=2e-12, atol_over_unit=2e-12,
                comparison_units={key: units.get(key, 1.) for key in segment},
                maximum_absolute_difference={key: float(np.max(abs(value[0] - prefix[key][-1])))
                                             for key, value in segment.items()},
                maximum_difference_over_unit={key: float(np.max(abs(value[0] - prefix[key][-1]))) / units.get(key, 1.)
                                              for key, value in segment.items()})


def _continuation_node(parent, folder, transitions, tolerance):
    """Verify one explicitly supplied donor; every source remains its own producer."""
    previous = _load(parent, tolerance)
    current = _load(folder, tolerance)
    record, data, data_hash, record_hash = current
    old, setting = previous[0], record['settings']
    lineage = setting.get('continuation', {})
    _continuation_source(old, record, transitions, tolerance)
    if lineage.get('prefix_record_sha256') != previous[3] or lineage.get('prefix_data_sha256') != previous[2]:
        raise ValueError('continuation native prefix record or data hash disagrees')
    files = dict(prefix_run='run.json', prefix_data='data.npz', origin_archive='initial_state.npz',
                 checkpoint_archive='final_state.npz')
    hashes = {key: hashlib.sha256((Path(parent) / name).read_bytes()).hexdigest() for key, name in files.items()}
    if (hashlib.sha256((Path(folder) / 'prefix_run.json').read_bytes()).hexdigest() != hashes['prefix_run']
            or hashlib.sha256((Path(folder) / 'prefix_data.npz').read_bytes()).hexdigest() != hashes['prefix_data']
            or any(lineage.get(key + '_sha256') != hashes[key] for key in ('origin_archive', 'checkpoint_archive'))):
        raise ValueError('continuation prefix or donor archive hashes disagree')
    origin, start, final = [_archive(Path(folder) / name) for name in
                            ('initial_state.npz', 'segment_initial_state.npz', 'final_state.npz')]
    _same_leaves(_archive(Path(parent) / 'initial_state.npz'), origin)
    _same_leaves(_archive(Path(parent) / 'final_state.npz'), start)
    _continuation_state_checks((origin, start), old, previous[1])
    _continuation_state_checks((origin, final), record, data)
    with np.load(Path(folder) / 'segment_data.npz', allow_pickle=False) as stored:
        raw = dict(stored)
    invalid = any(not value.ndim or value.shape[0] != len(raw['t']) or not np.isfinite(value).all()
                  or value.dtype.kind not in 'fc' or value.dtype.itemsize != (16 if value.dtype.kind == 'c' else 8)
                  for value in raw.values())
    if len(raw['t']) < 2 or invalid or any(not np.isfinite(value).all() for value in data.values()):
        raise ValueError('continuation SI segment requires finite x64 scalar histories')
    segment = _continuation_units(raw, setting)
    if (segment.keys() != data.keys() or any(not np.array_equal(data[key], np.concatenate(
            (previous[1][key], value[1:]))) for key, value in segment.items())):
        raise ValueError('continuation SI segment or exact scalar join disagrees')
    boundary = _continuation_boundary(previous[1], segment, setting)
    if (lineage.get('start_step') != int(start['steps']) or lineage.get('end_step') != int(final['steps'])
            or lineage.get('segment_steps') != int(final['steps']) - int(start['steps'])
            or int(final['steps']) <= int(start['steps'])
            or lineage.get('start_time_omega_p') != float(start['time']) * setting['normalization']['omega_p_rad_s']
            or lineage.get('end_time_omega_p') != data['t'][-1]):
        raise ValueError('continuation absolute segment clocks or steps disagree')
    if (not np.issubdtype(final['steps'].dtype, np.integer) or int(final['steps']) % setting['block_steps']
            or abs(lineage.get('target_time_omega_p', np.inf) - data['t'][-1]) > tolerance):
        raise ValueError('continuation target or complete block count disagrees')
    bounds = _continuation_bounds(old, record, start, final)
    return current, dict(record=record, run_sha256=record_hash, data_sha256=data_hash,
                         input_sha256=hashes, all_step_bounds=bounds, duplicated_scalar_boundary=boundary,
                         nominal_clock_error_omega_p=abs(data['t'][-1] - int(final['steps']) * setting['dt_omega_p']),
                         nominal_clock_bound_omega_p=1e-9,
                         nominal_clock_gate=bool(abs(data['t'][-1] - int(final['steps'])
                                                     * setting['dt_omega_p']) <= 1e-9),
                         gauss_bound_over_en_eps0=2e-13,
                         ordinary_gauss_gate=bool(record['results'][PAPER_MAXIMA[4]] <= 2e-13),
                         dark_gauss_gate=bool(record['results'][PAPER_MAXIMA[5]] <= 2e-13),
                         output_sha256={name: hashlib.sha256((Path(folder) / name).read_bytes()).hexdigest()
                                        for name in ('initial_state.npz', 'segment_initial_state.npz',
                                                     'final_state.npz', 'segment_data.npz')},
                         archive_fingerprints={key: fingerprint(value) for key, value in final.items()})


def _continuation_budget(records, summaries, duration):
    """Global accounting bounds retain near-zero-transfer failures, not statistical error bars."""
    defects = np.array([record['results'][PAPER_MAXIMA[0]] * .001 for record in records])
    rates = [2 * defect / (duration * row['plasma_mean_energy']) for defect, row in zip(defects, summaries)]
    transfer = [float(defect / abs(row['work_increment'])) if row['work_increment'] else None
                for defect, row in zip(defects, summaries)]
    work_difference = abs(summaries[1]['work_increment'] - summaries[0]['work_increment'])
    rate_difference = abs(summaries[1]['injection_rate_over_wp'] - summaries[0]['injection_rate_over_wp'])
    ratio = float(defects.sum() / work_difference) if work_difference else None
    rate_ratio = float(sum(rates) / rate_difference) if rate_difference else None
    return dict(global_energy_work_defect_over_nmc2L=defects.tolist(),
                global_defect_over_abs_window_work=transfer,
                conservation_transfer_gate=[value is not None and value < .01 for value in transfer],
                global_defect_over_abs_work_difference=ratio,
                conservation_difference_gate=ratio is not None and ratio < .1,
                absolute_rate_accounting_envelope_over_wp=rates,
                accounting_over_abs_rate_difference=rate_ratio,
                rate_accounting_gate=rate_ratio is not None and rate_ratio < 1 / 3)


def compare_continuations(first, second, donors, transitions, windows=CONTINUATION_WINDOWS, tolerance=1e-5):
    """Explicit prescribed timestep lineage audit; recompilation is a distinct execution."""
    if (len(donors) != 2 or not donors[0] or len(donors[0]) != len(donors[1]) or not transitions
            or tuple(windows) != CONTINUATION_WINDOWS):
        raise ValueError('continuation requires paired explicit oldest-first donors and reviewed source transitions')
    chains = [list(paths) + [folder] for paths, folder in zip(donors, (first, second))]
    original = compare_replays(chains[0][0], chains[1][0], 'dt', windows=((800, 1000),), tolerance=tolerance)
    audits = [[], []]
    for depth in range(1, len(chains[0])):
        rows = []
        for arm, chain in enumerate(chains):
            source, audit = _continuation_node(chain[depth - 1], chain[depth], transitions, tolerance)
            audits[arm].append(audit)
            rows.append(source)
        records, data = [row[0] for row in rows], [row[1] for row in rows]
        settings = [record['settings'] for record in records]
        for key in (*PARAMETERS, 'shape_order', 'XLA_FLAGS', 'recorded_mode', 'momentum_seed_over_sigma_e',
                    'seed_mode', 'seed_phase', *PAIR_RUNTIME):
            values = [record[key] if key in PAIR_RUNTIME else setting.get(key)
                      for record, setting in zip(records, settings)]
            if key != 'dt_omega_p' and values[0] != values[1]:
                raise ValueError(f'continuation timestep arms must share {key}')
        _protocol(settings, 'dt', tolerance)
        _initial(*settings, 'dt', False)
        if (settings[0]['coupling'] is not None or settings[0].get('momentum_seed_over_sigma_e', 0)
                or not np.isclose(settings[1]['dt_omega_p'] * 2, settings[0]['dt_omega_p'], rtol=2e-12, atol=0)
                or data[0]['t'].shape != data[1]['t'].shape
                or not np.allclose(data[0]['t'], data[1]['t'], rtol=0, atol=tolerance)):
            raise ValueError('continuation requires unseeded prescribed half-timestep arms with equal native clocks')
    selected_windows = []
    for start, end in windows:
        selected = (data[0]['t'] >= start - tolerance) & (data[0]['t'] <= end + tolerance)
        if start >= end or end > data[0]['t'][-1] + tolerance or selected.sum() < 2:
            raise ValueError('continuation fixed windows must fit complete native data')
        summaries = [window_summary(row, selected) for row in data]
        selected_windows.append(dict(window_omega_p=[start, end], samples=int(selected.sum()),
                                     realization_summaries=summaries,
                                     accounting=_continuation_budget(records, summaries,
                                                                     float(np.ptp(data[0]['t'][selected]))),
                                     observables={key: metrics(data[0][key][selected], data[1][key][selected])
                                                  for key in (*SERIES, *LOCAL_SERIES)}))
    return dict(variant='dt', original_comparison=original, windows=selected_windows, lineages=audits,
                source_vectors=[[original['sources'][arm]['git'], *[node['record']['git'] for node in audit]]
                                for arm, audit in enumerate(audits)],
                prescribed_force=dict(omega_over_wp=1., eta=1., phase=0.,
                                      amplitude_over_field=settings[0]['force_quiver_over_c']),
                maximum_clock_difference=float(np.max(abs(data[0]['t'] - data[1]['t']))),
                retained_gates=dict(zip(ENSEMBLE_OBSERVABLES, ENSEMBLE_BOUNDS)),
                additional_gates=dict(conservation_over_window_transfer=.01,
                                      conservation_over_interpreted_energy_difference=.1,
                                      rate_error_over_interpreted_change=1 / 3),
                notes='Separate native producers and preserved global reference; no same-executable claim. '
                      'Raw native samples; no interpolation or phase alignment. Frozen prefix failures remain. '
                      'The additive rate envelope bounds discrete global accounting only, not temporal, spatial or '
                      'execution uncertainty. One pair combines timestep and execution sensitivity; no ensemble '
                      'or convergence claim.')


def _fieller(first, second, critical):
    """Paired ratio-of-means interval; an unresolved denominator stays unavailable."""
    scale = max(np.max(abs(first)), np.max(abs(second)))
    if not np.isfinite(scale) or scale == 0:
        return None
    first, second = first / scale, second / scale
    mean = np.mean(first)
    A = mean**2 - critical**2 * np.var(first, ddof=1) / len(first)
    if mean <= 0 or A <= 0:
        return None
    ratio = np.mean(second) / mean
    residual = second - ratio * first
    B = -critical**2 * np.cov(first, residual, ddof=1)[0, 1] / len(first)
    C = critical**2 * np.var(residual, ddof=1) / len(first)
    return (ratio - 1 + (B + np.array([-1., 1.]) * np.sqrt(B**2 + A * C)) / A).tolist()


def _ensemble_intervals(coarse, fine, simultaneous):
    """Seeds are the independent units; confidence bounds retain all eight gates."""
    from scipy.stats import t
    count = len(coarse)
    critical = float(t.ppf(1 - .05 / (2 * len(ENSEMBLE_BOUNDS) if simultaneous else 2), count - 1))
    rows = {}
    for index, (name, bound) in enumerate(zip(ENSEMBLE_OBSERVABLES, ENSEMBLE_BOUNDS)):
        a, b = coarse[:, index], fine[:, index]
        delta = b - a
        half = critical * delta.std(ddof=1) / np.sqrt(count)
        additive = [float(delta.mean() - half), float(delta.mean() + half)]
        ratio = _fieller(a, b, critical) if name != 'injection_rate' else None
        interval = additive if name == 'injection_rate' else ratio
        rows[name] = dict(coarse_mean=float(a.mean()), fine_mean=float(b.mean()),
                          coarse_seed_sd=float(a.std(ddof=1)), fine_seed_sd=float(b.std(ddof=1)),
                          paired_difference=float(delta.mean()), paired_sd=float(delta.std(ddof=1)),
                          paired_covariance=float(np.cov(a, b, ddof=1)[0, 1]), additive_interval=additive,
                          relative_mean_change=float(b.mean() / a.mean() - 1) if a.mean() > 0 else None,
                          fieller_relative_interval=ratio, bound=bound,
                          equivalent=bool(interval is not None and interval[0] > -bound and interval[1] < bound))
    return dict(confidence=.95, simultaneous=simultaneous, critical_t=critical, observables=rows,
                all_equivalent=all(row['equivalent'] for row in rows.values()))


def _ensemble_sources(pairs, tolerance):
    if len(pairs) not in (3, 5) or any(len(pair) != 2 for pair in pairs):
        raise ValueError('ensemble requires ordered pairs for all seeds 0/1/2 or 0–4')
    sources = [[_load(path, tolerance) for path in pair] for pair in pairs]
    for seed, pair in enumerate(sources):
        for level, (record, data, _, _) in enumerate(pair):
            setting = record['settings']
            scales = np.asarray(setting.get('local_spread_lengths_c_over_wp', []))
            if type(setting.get('seed')) is not int or setting['seed'] != seed:
                raise ValueError('ensemble pairs must contain the ordered preselected seeds')
            revisions = (record['git'], setting['parent_revision'])
            if (record['jax_enable_x64'] is not True or setting['coupling'] is not None or 'local_spread' not in data
                    or scales.shape != (2,) or not np.allclose(
                        scales, np.array([2., 4.]) * np.sqrt(setting['T_each_over_mec2']), rtol=2e-12, atol=0)
                    or any(not isinstance(v, str) or len(v) != 40 or set(v) - set('0123456789abcdef')
                           for v in revisions)):
                raise ValueError('ensemble needs clean sources, x64, prescribed drive and ordered 2/4 Debye lengths')
            anchor = sources[0][level]
            if any(record.get(key) is None or record[key] != sources[0][0][0].get(key)
                   for key in ('python', 'platform')):
                raise ValueError('ensemble runtime differs across seeds')
            settings = _controls([anchor[0], record], [anchor[1], data], 'seed', tolerance)
            _initial(*settings, 'seed', False)
        if not np.isclose(pair[1][0]['settings']['dt_omega_p'] * 2, pair[0][0]['settings']['dt_omega_p'],
                          rtol=2e-12, atol=0):
            raise ValueError('ensemble pairs require a halved timestep')
    if len({pair[0][0]['settings']['initial_fingerprints']['loading']['v'] for pair in sources}) != len(pairs):
        raise ValueError('different seeds must retain different physical velocity fingerprints')
    return sources


def ensemble_comparison(pairs, window=(800, 1000), tolerance=1e-5):
    """Reduce each fixed native window before seed statistics; no trajectory alignment."""
    pairs = list(pairs)
    sources = _ensemble_sources(pairs, tolerance)
    comparisons = [compare_replays(*pair, variant='dt', windows=(window,), tolerance=tolerance) for pair in pairs]
    raw = [row['windows'][0]['realization_summaries'] for row in comparisons]
    vectors = np.asarray([[np.r_[row['work_increment'], np.ravel(row['local_spread_increment_mean']),
                                 row['electric_mean'], row['nonzero_electric_mean'], row['injection_rate_over_wp']]
                           for row in pair] for pair in raw])
    return dict(seeds=list(range(len(pairs))), window_omega_p=list(window), independent_units='seed pairs',
                frames_are_replicates=False, status='exploratory pilot' if len(pairs) == 3 else 'five-seed reduction',
                raw_per_seed=[dict(seed=i, reductions=rows, native_comparison=comparisons[i])
                              for i, rows in enumerate(raw)],
                marginal=_ensemble_intervals(vectors[:, 0], vectors[:, 1], False),
                simultaneous=_ensemble_intervals(vectors[:, 0], vectors[:, 1], True),
                native_runs=[[source[0] for source in pair] for pair in sources],
                additional_gates=dict(rate_error_over_interpreted_change=1 / 3,
                                      conservation_over_window_transfer=.01,
                                      conservation_over_interpreted_energy_difference=.1),
                notes='Eight fixed gates; simultaneous intervals use Bonferroni95% bounds. Student/Fieller bounds '
                      'assume independent approximately normal seed reductions; three/five seeds cannot verify tails. '
                      'Seed scatter includes loading+execution; paired sensitivity includes timestep+execution. '
                      'Exact-state repeats are needed to isolate execution variance. No pure timestep bias, order, '
                      'continuum or late-reproduction claim. Seeds3/4 are prospective held-outs only while the pilot '
                      'recipe remains frozen; the combined five-seed analysis is not independently held out. '
                      'Unbounded/undefined relative intervals fail equivalence; additive rate bounds remain available. '
                      'Reduce within runs before averaging seeds; no profile averaging, phase shifts or interpolation. '
                      'Native conservation budgets remain separate necessary gates.')


def _replay_label(settings, variant, index):
    """Name the measured resolution and the controlled seed or execution."""
    if variant in ('repeat', 'dt'):
        label = f"Δtωₚ={settings['dt_omega_p']:g}"
        return label + f', execution {index + 1}' if variant == 'repeat' and index < 2 else label
    if variant == 'shape':
        return f"degree {settings.get('shape_order', 2)} weighting, Δtωₚ={settings['dt_omega_p']:g}"
    return f"{settings['cells']} cells, {settings['particles_per_species'] // 1000}k/species, seed {settings['seed']}"


def _loading_repeat(folder, reference):
    """Keep an extra scalar execution without requiring its final native archive."""
    if folder is None:
        return {}, {}
    if reference is None:
        raise ValueError('a loading repeat requires the particle-loading control')
    repeated = _load(folder, 1e-5)
    return (dict(loading_execution_comparison=compare_replays(folder, reference),
                 loading_repeat_run=repeated[0], loading_repeat_endpoint=constraint_audit(folder, repeated[0])),
            {f'loading_repeat_{key}': value for key, value in repeated[1].items()})


def _refined_source(first, second, against):
    if against not in ('first', 'second'):
        raise ValueError('refined_against must be first or second')
    return first if against == 'first' else second


def _step_contraction(first, second):
    """Raw adjacent differences; a ratio does not establish temporal order or convergence."""
    for row in (first, second):
        coarse, fine = [source['varied_parameter'] for source in row['sources']]
        if row['variant'] != 'dt' or not np.isclose(coarse, 2 * fine, rtol=2e-12, atol=0):
            raise ValueError('adjacent contractions require two consecutive timestep halvings')
    if [row['window_omega_p'] for row in first['windows']] != [row['window_omega_p'] for row in second['windows']]:
        raise ValueError('adjacent timestep contractions require the same native comparison windows')
    result = []
    for a, b in zip(first['windows'], second['windows']):
        row = dict(window=a['window_omega_p'])
        for key in ('mean_E', 'electric', 'nonzero_electric', 'local_spread'):
            if key in a['observables'] and key in b['observables']:
                norms = [float(np.linalg.norm(x['observables'][key]['l2_difference'])) for x in (a, b)]
                row[key] = dict(adjacent_l2=norms, coarse_to_fine_contraction=norms[0] / norms[1] if norms[1] else None)
        result.append(row)
    return result


def _finer_comparison(refined, finer, variant, loading, repeat, constraints):
    if finer is None:
        return None
    if refined is None or variant != 'dt' or loading is not None or repeat is not None:
        raise ValueError('finer_step requires a dt refinement and excludes loading controls')
    return compare_replays(refined, finer, 'dt', constraints=constraints)


def publish(first, second, folder, comparison, refined=None, refined_variant='dt', loading_refined=None,
            loading_repeat=None, refined_against='first', finer_step=None):
    """Render saved scalars through the parent figure/provenance path; no dynamics."""
    anchor = _refined_source(first, second, refined_against)
    third_step = _finer_comparison(refined, finer_step, refined_variant, loading_refined, loading_repeat,
                                   'endpoint_constraints' in comparison)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from darkjaxincell import midnight
    from jaxincell import save_run

    sources = [_load(path, 1e-5) for path in (first, second)]
    refinement = None
    if refined is not None:
        refinement = compare_replays(anchor, refined, variant=refined_variant,
                                     constraints='endpoint_constraints' in comparison)
        sources.append(_load(refined, 1e-5))
    sources += [_load(finer_step, 1e-5)] if finer_step is not None else []
    loading = None
    if loading_refined is not None:
        loading = compare_replays(second, loading_refined, variant='loading',
                                  constraints='endpoint_constraints' in comparison)
        sources.append(_load(loading_refined, 1e-5))
    time = sources[0][1]['t']
    width = max(1, round(2 * np.pi / sources[0][0]['settings']['output_dt_omega_p']))
    trim = slice(width, -width)

    def average(values):
        return np.convolve(values, np.ones(width) / width, mode='same')[trim]

    with midnight():
        figure, axes = plt.subplots(2, 2, figsize=(10, 7), layout='constrained')
        for index, (record, data, _, _) in enumerate(sources):
            label = _replay_label(record['settings'], comparison['variant'], index)
            style, color = ('-', '--', ':', '-.')[index], ('#0072B2', '#D55E00', '#009E73', '#6A3D9A')[index]
            raw = data['nonzero_electric'] / .0005
            axes[0, 0].plot(time, raw, color=color, alpha=.15, lw=.5)
            axes[0, 0].plot(time[trim], average(raw), style, color=color, label=label)
            if 'local_spread' in data:
                for species, name, species_color in ((0, 'electrons', '#6A3D9A'), (1, 'ions', '#009E73')):
                    spread = data['local_spread'][:, species, 0] / data['local_spread'][0, species, 0]
                    axes[0, 1].plot(time, spread, style, color=species_color,
                                    label=name if index == 0 else '_nolegend_')
                    density = data['local_density_rms'][:, species, 0]
                    axes[1, 0].plot(time[trim], average(density), style, color=species_color,
                                    label=name if index == 0 else '_nolegend_')
            defect = (data['balance'] - data['balance'][0]) / max(abs(data['work']).max(), 1e-30)
            axes[1, 1].plot(time, defect, style, color=color, label=label)
        upper = max(1e-3, 1.3 * max(np.max(source[1]['nonzero_electric'] / .0005) for source in sources))
        axes[0, 0].set(ylabel=r'$U_{E,k\ne0}/(nT_{e0}L/2)$', title='Nonzero-mode electric energy',
                       yscale='log', ylim=(1e-4, upper))
        axes[0, 1].set(ylabel=r'$U_{\rm local}(t)/U_{\rm local}(0)$', title=r'Local spread at $2\lambda_{D0}$',
                       yscale='log')
        axes[1, 0].set(ylabel='smoothed density RMS / mean', title=r'Density at $2\lambda_{D0}$')
        coupled = sources[0][0]['settings']['coupling'] is not None
        axes[1, 1].set(ylabel=r'$\Delta U/\max|W_D|$' if coupled else r'$(\Delta U-W)/\max|W|$',
                       title='Closed energy' if coupled else 'Energy and external work')
        for axis in axes.flat:
            axis.set(xlabel=r'$\omega_pt$', xlim=(0, time[-1]))
            axis.grid(alpha=.25)
            axis.legend(fontsize=8)
        names = ['first', 'second'] + (['refined'] if refined is not None else [])
        names += ['finer'] if finer_step is not None else []
        names += ['loading'] if loading_refined is not None else []
        arrays = {f'{name}_{key}': value for name, source in zip(names, sources)
                  for key, value in source[1].items()}
        settings = dict(variant=comparison['variant'], parent_revision=sources[0][0]['settings']['parent_revision'],
                        figure_average_samples=width, figure_average_span_omega_p=width * np.median(np.diff(time)),
                        figure_note='Faint energy traces are raw; energy and density averages trim endpoints. '
                                    'Local spread is raw; line styles identify controls, purple/green identify '
                                    'electrons/ions. All comparison metrics use raw native samples.')
        result = dict(comparison=comparison, native_runs=[source[0] for source in sources],
                      claim='Controlled numerical sensitivity study; no convergence or statistical uncertainty claim')
        if refinement is not None:
            result['refined_against'] = refined_against
            result['refinement_comparison'] = refinement
        result.update(dict(third_step_comparison=third_step,
                           norm_contraction=_step_contraction(refinement, third_step),
                           norm_contraction_note='Raw adjacent differences from one loading and separate executions; '
                                                 'ratios do not certify temporal order, convergence or uncertainty.')
                      if third_step is not None else {})
        if loading is not None:
            result['loading_comparison'] = loading
        repeated, repeated_arrays = _loading_repeat(loading_repeat, loading_refined)
        result.update(repeated)
        arrays.update(repeated_arrays)
        save_run(folder, 'controlled_paper_replay', settings, result, figure, **arrays)
        np.savez_compressed(Path(folder) / 'data.npz', **arrays)
        plt.close(figure)


def _implicit_controls(ordinary, record):
    """Match the physical plasma and exact initial arrays across native clocks."""
    a, b = ordinary['settings'], record['settings']
    if not b['paper_loading'] or b['newtonian'] or a['coupling'] is not None or a['seed'] != 0:
        raise ValueError('comparison requires the same relativistic seed-zero prescribed Gaussian plasma')
    for left, right in (('cells', 'cells'), ('particles_per_species', 'particles_per_species'),
                        ('length_c_over_omega_p', 'length_over_c_wp'), ('mass_ratio', 'mass_ratio'),
                        ('T_each_over_mec2', 'temperature'), ('force_quiver_over_c', 'amplitude'),
                        ('parent_revision', 'parent_revision')):
        if a[left] != b[right]:
            raise ValueError(f'method comparison must share {left}')
    for key in ('jax', 'jaxincell', 'numpy', 'jax_enable_x64', 'backend'):
        if ordinary[key] != record[key]:
            raise ValueError(f'method comparison must share runtime {key}')
    loading, state = a['initial_fingerprints']['loading'], a['initial_fingerprints']['state']
    hashes = b['initial_fingerprints']
    matches = dict(x=loading['x'] == hashes['x'],
                   **{key: state[key] == hashes[key] for key in ('u', 'w', 'B')})
    if not all(matches.values()):
        raise ValueError('physical initial positions, momenta, weights or magnetic field disagree')
    return matches


def _implicit_load(folder):
    """Read finite native implicit arrays without broadcasting malformed histories."""
    path = Path(folder)
    record = json.loads((path / 'run.json').read_text())
    with np.load(path / 'data.npz', allow_pickle=False) as stored:
        data = dict(stored)
    count = len(data['t'])
    settings = record['settings']
    if (data['t'].ndim != 1 or count < 2 or data['t'][0] != 0 or np.any(np.diff(data['t']) <= 0)
            or settings['steps'] % settings['store_every']
            or count != settings['steps'] // settings['store_every'] + 1
            or not np.allclose(np.diff(data['t']), settings['dt'] * settings['store_every'], rtol=0, atol=1e-5)
            or not np.allclose(data['t'][-1], [settings['actual_horizon'], settings['steps'] * settings['dt']],
                               rtol=0, atol=1e-5)):
        raise ValueError('implicit native clocks must match the recorded steps, horizon and cadence')
    for key, value in data.items():
        shape = ((count, 3) if key.endswith('momentum') else (count, 2)
                 if key in ('mean', 'rms', 'kinetic', 'velocity_spread_energy')
                 or key.endswith(('_mean', '_rms')) else (count,))
        if value.shape != shape or not np.all(np.isfinite(value)):
            raise ValueError(f'implicit {key} requires finite native shape {shape}')
    return record, data


def implicit_comparison(explicit, implicit, tolerance=1e-5):
    """Align prescribed Gaussian controls; explicit electric is energy, implicit is E.

    Particle u/w and supplied x hashes anchor loading despite position staggering.
    Instantaneous current comparisons leave accepted midpoint work in its native ledger.
    """
    ordinary = _load(explicit, tolerance)
    path = Path(implicit)
    record, data = _implicit_load(path)
    matches = _implicit_controls(ordinary[0], record)
    a, b = ordinary[0]['settings'], record['settings']
    time = ordinary[1]['t']
    cadence = b['dt'] * b['store_every']
    ratio = a['output_dt_omega_p'] / cadence
    if ratio < 1 or abs(ratio - round(ratio)) > tolerance:
        raise ValueError('native cadences must have an integer ratio')
    indices = np.arange(len(time)) * round(ratio)
    if (data['t'].ndim != 1 or len(data['t']) != indices[-1] + 1 or data['t'][0] != 0
            or not np.allclose(np.diff(data['t']), cadence, rtol=0, atol=tolerance)
            or not np.allclose(time, data['t'][indices], rtol=0, atol=tolerance)):
        raise ValueError('equal native horizons and clocks are required; interpolation is unsupported')
    ordinary_data = ordinary[1]
    first = dict(t=time, mean_E=ordinary_data['mean_E'],
                 current=ordinary_data['mean'][:, 1] - ordinary_data['mean'][:, 0],
                 nonzero_electric=ordinary_data['nonzero_electric'], work=ordinary_data['work'],
                 balance=ordinary_data['balance'] - ordinary_data['balance'][0],
                 momentum=ordinary_data['momentum'] - ordinary_data['momentum'][0])
    second = dict(t=data['t'][indices], mean_E=data['electric'][indices],
                  **{key: data[key][indices] for key in ('current', 'nonzero_electric', 'work', 'balance')},
                  momentum=data['momentum'][indices] - data['momentum'][0])
    if not all(np.all(np.isfinite(value)) for row in (first, second) for value in row.values()):
        raise ValueError('method comparison needs finite raw scalar histories')
    windows = []
    for end in (40, 100, 250, 500, 1000):
        if end > time[-1] + tolerance:
            continue
        mask = time <= end + tolerance
        windows.append(dict(end_omega_p=end, samples=int(mask.sum()),
                            observables={key: metrics(first[key][mask], second[key][mask])
                                         for key in first if key != 't'}))
    peak_work = float(np.max(abs(second['work'])))
    result = dict(initial_hash_matches=matches, windows=windows,
                  max_clock_difference=float(np.max(abs(time - second['t']))),
                  implicit_native_samples=len(data['t']), common_samples=len(time),
                  implicit_balance_over_peak_recorded_work=float(
                      record['results']['max_balance_over_nmc2L'] / peak_work) if peak_work else None,
                  native_run_sha256=[ordinary[3], hashlib.sha256((path / 'run.json').read_bytes()).hexdigest()],
                  native_data_sha256=[ordinary[2], hashlib.sha256((path / 'data.npz').read_bytes()).hexdigest()],
                  notes='Raw native samples, no interpolation or phase alignment. Momentum subtracts each '
                        'actual initial value. Energy is normalized by n me c² L and momentum by n me c L. '
                        'The peak-work denominator uses the common output cadence, not all-step work. '
                        'Implicit all-step maxima describe its final timed call; its repeat traces use one '
                        'compiled callable. No species-heating estimate is inferred from field/work histories. '
                        'Cross-method differences do not establish convergence or identify an error cause.')
    return result, ordinary[0], record, first, second, data


def _implicit_pair(first, second, variant='iterations'):
    """Require matched physical loading, native clocks and one varied control."""
    records, data = zip(*[_implicit_load(path) for path in (first, second)])
    settings = [record['settings'] for record in records]
    keys = ('x', 'u', 'w', 'E', 'B', 'rho', 'time', 'mass', 'charge')
    if not all(all(_hash_matches((row['initial_fingerprints'],) * 2, keys).values()) for row in settings):
        raise ValueError('implicit controls require complete initial fingerprints')
    changes = {'iterations': {'iterations'}, 'substeps': {'substeps'},
               'grid_phase': {'grid_phase', 'initial_fingerprints'},
               'cells': {'cells', 'initial_fingerprints', 'initial_state_source'}}[variant]
    for key in settings[0].keys() | settings[1].keys():
        if key not in changes and settings[0].get(key) != settings[1].get(key):
            raise ValueError(f'implicit controls must share {key}, including required initial hashes')
    if settings[0][variant] == settings[1][variant]:
        raise ValueError(f'implicit controls must vary {variant}')
    if variant in ('cells', 'grid_phase'):
        hashes = [row['initial_fingerprints'] for row in settings]
        shared = ('u', 'w', 'time', 'mass', 'charge') + (('x',) if variant == 'cells' else ('B',))
        if not all(_hash_matches(hashes, shared).values()):
            raise ValueError('mesh controls require identical physical particle arrays')
    for key in ('git', 'jax', 'jaxincell', 'numpy', 'backend', 'jax_enable_x64'):
        if records[0][key] != records[1][key]:
            raise ValueError(f'implicit controls must share runtime/source {key}')
    if not np.array_equal(data[0]['t'], data[1]['t']) or data[0]['t'][0] != 0:
        raise ValueError('implicit controls require identical native clocks')
    traces = [dict(row, momentum=row['momentum'] - row['momentum'][0]) for row in data]
    return records, data, {key: metrics(traces[0][key], traces[1][key]) for key in
                           ('electric', 'current', 'nonzero_electric', 'momentum', 'mean', 'rms', 'kinetic')}


def publish_phases(folders, folder, audits=()):
    """Compare fixed Gaussian loadings translated within each mesh, retaining repeat variability."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from darkjaxincell import midnight
    from jaxincell import save_run
    records, data = zip(*[_implicit_load(path) for path in folders])
    controls = {(row['settings']['cells'], row['settings']['grid_phase']): index
                for index, row in enumerate(records)}
    meshes = sorted({key[0] for key in controls})
    phases = sorted({key[1] for key in controls})
    if (len(controls) != len(records) or len(meshes) < 2 or len(phases) < 2 or phases[0] != 0
            or set(controls) != {(mesh, phase) for mesh in meshes for phase in phases}):
        raise ValueError('phase controls require a complete distinct mesh/phase grid with phase zero')
    comparisons = []
    for mesh in meshes:
        base = controls[mesh, 0]
        for phase in phases[1:]:
            index = controls[mesh, phase]
            _, _, norms = _implicit_pair(folders[base], folders[index], 'grid_phase')
            comparisons.append(dict(cells=mesh, grid_phase=phase, observables=norms))
    for mesh in meshes[1:]:
        _implicit_pair(folders[controls[meshes[0], 0]], folders[controls[mesh, 0]], 'cells')
    arrays = {f'{index}_{key}': value for index, row in enumerate(data) for key, value in row.items()}
    with midnight():
        fig, axes = plt.subplots(2, len(meshes), figsize=(4 * len(meshes), 6),
                                 squeeze=False, layout='constrained')
        for column, mesh in enumerate(meshes):
            for phase in phases:
                row = data[controls[mesh, phase]]
                axes[0, column].plot(row['t'], row['momentum'][:, 0] - row['momentum'][0, 0],
                                     label=f'{phase:g} Δx')
                axes[1, column].plot(row['t'], row['nonzero_electric'])
            axes[0, column].set(title=f'{mesh:,} cells', ylabel=r'$\Delta P_x/(nm_ecL)$')
            axes[1, column].set(ylabel=r'$U_{E,k\ne0}/(nm_ec^2L)$', xlabel=r'$\omega_pt$')
            axes[0, column].legend(fontsize=9)
        for axis in axes.flat:
            axis.grid(alpha=.25)
        save_run(folder, 'implicit_grid_phase', dict(parent_revision=records[0]['settings']['parent_revision']),
                 dict(native_runs=records, phase_observables=comparisons,
                      orbit_audits=[json.loads((Path(path) / 'run.json').read_text()) for path in audits],
                      native_data_sha256=[hashlib.sha256((Path(path) / 'data.npz').read_bytes()).hexdigest()
                                          for path in folders],
                      native_run_sha256=[hashlib.sha256((Path(path) / 'run.json').read_bytes()).hexdigest()
                                         for path in folders],
                      claim='Short Gaussian grid-phase controls with exact loading and separate repeat traces; '
                            'no late-conversion or continuum momentum-convergence claim'), fig, **arrays)
        np.savez_compressed(Path(folder) / 'data.npz', **arrays)
        plt.close(fig)


def publish_iterations(first, second, folder, audits=()):
    """Retain exact-input Picard controls, raw arrays and independent orbit records."""
    from jaxincell import save_run
    records, data, observables = _implicit_pair(first, second)
    result = dict(native_runs=records,
                  observables=observables,
                  orbit_audits=[json.loads((Path(path) / 'run.json').read_text()) for path in audits],
                  native_data_sha256=[hashlib.sha256((Path(path) / 'data.npz').read_bytes()).hexdigest()
                                      for path in (first, second)],
                  native_run_sha256=[hashlib.sha256((Path(path) / 'run.json').read_bytes()).hexdigest()
                                     for path in (first, second)],
                  claim='Iteration-count sensitivity with exact native initialization; conservation alone '
                        'does not validate late trajectories. Orbit references use frozen accepted fields.')
    arrays = {f'{index}_{key}': value for index, row in enumerate(data) for key, value in row.items()}
    save_run(folder, 'implicit_iteration_control', dict(parent_revision=records[0]['settings']['parent_revision']),
             result, **arrays)
    np.savez_compressed(Path(folder) / 'data.npz', **arrays)


def publish_method_controls(first, substeps, mesh, folder, finer=None):
    """Render independent mesh/substep controls without launching any dynamics."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from darkjaxincell import midnight
    from jaxincell import save_run
    records, data, substep_metrics = _implicit_pair(first, substeps, 'substeps')
    mesh_records, mesh_data, mesh_metrics = _implicit_pair(first, mesh, 'cells')
    records, data = (*records, mesh_records[1]), (*data, mesh_data[1])
    paths = [first, substeps, mesh]
    refinement = {}
    if finer is not None:
        fine_records, fine_data, fine_metrics = _implicit_pair(mesh, finer, 'cells')
        records, data = (*records, fine_records[1]), (*data, fine_data[1])
        paths.append(finer)
        refinement['refined_mesh_observables'] = fine_metrics
    arrays = {f'{index}_{key}': value for index, row in enumerate(data) for key, value in row.items()}
    with midnight():
        fig, axes = plt.subplots(2, 2, figsize=(10, 6), layout='constrained')
        for record, row, style in zip(records, data, ('-', '--', ':', '-.')):
            s = record['settings']
            label = f"{s['cells']} cells, {s['substeps']} substeps"
            for axis, key in zip(axes.flat, ('electric', 'nonzero_electric', 'momentum', 'balance')):
                value = row[key] - row[key][0] if key == 'momentum' else row[key]
                if key == 'momentum':
                    value = value[:, 0]
                if key == 'balance':
                    value = abs(value)
                axis.plot(row['t'], value, style, label=label)
        axes[0, 0].set(ylabel=r'$\langle E_x\rangle/E_\star$', title='Mean electric response')
        axes[0, 1].set(ylabel=r'$U_{E,k\ne0}/(nm_ec^2L)$', title='Nonzero-mode energy')
        axes[1, 0].set(ylabel=r'$\Delta P_x/(nm_ecL)$', title='Momentum balance')
        axes[1, 1].set(ylabel=r'$|\Delta U-W|/(nm_ec^2L)$', title='Energy and source work', yscale='log')
        for axis in axes.flat:
            axis.set_xlabel(r'$\omega_pt$')
            axis.grid(alpha=.25)
        axes[0, 0].legend(fontsize=8)
        save_run(folder, 'implicit_method_controls', dict(parent_revision=records[0]['settings']['parent_revision']),
                 dict(native_runs=records, substep_observables=substep_metrics, mesh_observables=mesh_metrics,
                      native_data_sha256=[hashlib.sha256((Path(path) / 'data.npz').read_bytes()).hexdigest()
                                          for path in paths],
                      native_run_sha256=[hashlib.sha256((Path(path) / 'run.json').read_bytes()).hexdigest()
                                         for path in paths],
                      norm_note='Momentum norms subtract each initial value; RMS/kinetic norms include the '
                                'initial thermal baseline. Histories are final timed calls; native records '
                                'retain first/warm execution variability. No interpolation or phase alignment.',
                      claim='Separate mesh/substep sensitivities at fixed physical particles; '
                            'no late-convergence or error-cause claim', **refinement), fig, **arrays)
        np.savez_compressed(Path(folder) / 'data.npz', **arrays)
        plt.close(fig)


def publish_implicit(explicit, implicit, folder, refined=None):
    """Compare saved conservation/mode evidence; the renderer runs no dynamics."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from darkjaxincell import midnight
    from jaxincell import save_run

    result, native, record, first, second, raw = implicit_comparison(explicit, implicit)
    explicit_label = f"explicit Δtωₚ={native['settings']['dt_omega_p']:g}"
    implicit_label = f"implicit Δtωₚ={record['settings']['dt']:g}"
    records, series = [native, record], [(explicit_label, first), (implicit_label + ', warm', second)]
    if refined is not None:
        compare_replays(explicit, refined, variant='dt')
        refinement, fine_record, _, fine, _, _ = implicit_comparison(refined, implicit)
        result['refined_method_comparison'] = refinement
        records.append(fine_record)
        series.append((f"explicit Δtωₚ={fine_record['settings']['dt_omega_p']:g}", fine))
    initial = dict(t=raw['t'], mean_E=raw['execution_0_mean_E'],
                   nonzero_electric=raw['execution_0_nonzero_electric'],
                   momentum=raw['execution_0_momentum'] - raw['execution_0_momentum'][0])
    series.append((implicit_label + ', first', initial))
    colors = ('#0072B2', '#6A3D9A', '#009E73', '#D55E00')
    with midnight():
        fig, axes = plt.subplots(2, 2, figsize=(10, 7), layout='constrained')
        for (label, row), color in zip(series, colors):
            t = row['t']
            early = t <= 40 + 1e-5
            axes[0, 0].plot(t[early], row['mean_E'][early], color=color, label=label)
            width = max(1, round(2 * np.pi / np.median(np.diff(t))))
            averaged = np.convolve(row['nonzero_electric'], np.ones(width) / width, mode='valid')
            trim = width // 2
            axes[0, 1].plot(t, row['nonzero_electric'], color=color, alpha=.12, lw=.4)
            axes[0, 1].plot(t[trim:trim + len(averaged)], averaged, color=color, label=label)
            axes[1, 0].plot(t, row['momentum'][:, 0], color=color, label=label)
            if 'balance' in row:
                axes[1, 1].plot(t, abs(row['balance']), color=color, label=label)
        axes[0, 0].plot(raw['t'][raw['t'] <= 40], raw['oracle_E'][raw['t'] <= 40], 'k--',
                        lw=.8, label='homogeneous reference')
        axes[0, 0].set(ylabel=r'$\langle E_x\rangle/E_\star$', title='Early mean response', xlim=(0, 40))
        axes[0, 1].set(ylabel=r'$U_{E,k\ne0}/(nm_ec^2L)$', title='Nonzero-mode energy', yscale='log',
                       ylim=(1e-8, None))
        axes[1, 0].set(ylabel=r'$[P_x(t)-P_x(0)]/(nm_ecL)$', title='Momentum balance')
        axes[1, 1].set(ylabel=r'$|\Delta U-W|/(nm_ec^2L)$', title='Energy and source work', yscale='log',
                       ylim=(1e-18, None))
        for axis in axes.flat:
            axis.set_xlabel(r'$\omega_pt$')
            axis.grid(alpha=.25)
        axes[0, 0].legend(fontsize=8)
        axes[0, 1].legend(fontsize=8)
        arrays = {f'{index}_{key}': value for index, (_, row) in enumerate(series[:3]) for key, value in row.items()}
        arrays.update({f'implicit_{key}': value for key, value in raw.items()})
        settings = dict(parent_revision=native['settings']['parent_revision'],
                        figure_note='Faint energy traces are raw; thick traces average approximately one plasma '
                                    'period with trimmed endpoints. All metrics use raw native samples.')
        save_run(folder, 'paper_implicit_comparison', settings,
                 dict(comparison=result, native_runs=records,
                      claim='Prescribed-force numerical comparison; late field/particle convergence remains open'),
                 fig, **arrays)
        np.savez_compressed(Path(folder) / 'data.npz', **arrays)
        plt.close(fig)


def _pair_source(folder):
    folder = Path(folder)
    record = json.loads((folder / 'run.json').read_text())
    with np.load(folder / 'data.npz', allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    for value in (record.get('git', ''), record['settings'].get('parent_revision', '')):
        if len(value) != 40 or not set(value) <= set('0123456789abcdef'):
            raise ValueError('pair source and parent must be complete clean SHAs')
    if (any(key not in record for key in PAIR_RUNTIME) or record.get('jax_enable_x64') is not True
            or not all(np.isfinite(value).all() for value in arrays.values())
            or any(value.dtype.kind == 'f' and value.dtype.itemsize != 8
                   or value.dtype.kind == 'c' and value.dtype.itemsize != 16 for value in arrays.values())):
        raise ValueError('pair records require x64 and finite native arrays')
    hashes = {name + '_sha256': hashlib.sha256((folder / filename).read_bytes()).hexdigest()
              for name, filename in (('run', 'run.json'), ('data', 'data.npz'))}
    return record, arrays, hashes


def _pair_initial(folder, setting, hashes, data, archive='initial_state.npz'):
    """Read initial arrays only for validation; publish hashes and small diagnostics."""
    count = setting['cells'] * setting['particles_per_cell_per_species']
    scale = setting['normalization']
    c = 299792458.  # Exact SI speed, independent of a CODATA release.
    if set(hashes) != {'x', 'u', 'w', 'E', 'B', 'rho'}:
        raise ValueError('pair initial fingerprints require all six ordinary arrays')
    with np.load(Path(folder) / archive, allow_pickle=False) as state:
        arrays = {key: state[key] for key in ('x', 'u', 'w', 'E', 'B', 'rho')}
        for key in ('x', 'u', 'w', 'E', 'B', 'rho'):
            array = np.ascontiguousarray(arrays[key])
            digest = hashlib.sha256(f'{array.dtype.str}:{array.shape}'.encode())
            digest.update(array.tobytes())
            if digest.hexdigest() != hashes[key] or not np.isfinite(array).all() or array.dtype != np.float64:
                raise ValueError('pair initial archive differs from its ordinary fingerprints')
        if (arrays['x'].shape != (2 * count, 3) or arrays['u'].shape != (2 * count, 3)
                or arrays['w'].shape != (2 * count,) or np.any(arrays['w'] <= 0)
                or arrays['E'].shape != (setting['cells'], 3) or arrays['B'].shape != arrays['E'].shape
                or arrays['rho'].shape != (setting['cells'],)):
            raise ValueError('pair initial particle count/shape differs from the record')
        mass, density, length = state['dark.mass'], state['dark.density'], float(state['dark.length'])
        if (mass.shape != (2,) or density.shape != (2,) or mass[0] <= 0 or mass[0] != mass[1]
                or density[0] <= 0 or density[0] != density[1] or not np.array_equal(state['dark.charge'], [-1, 1])
                or not bool(state['dark.relativistic'])
                or not np.isclose(length * scale['omega0_rad_s'] / c, setting['length_c_over_omega0'], rtol=2e-12)
                or not np.isclose(scale['field_scale_V_m'], mass[0] * c * scale['omega0_rad_s'] / 1.602176634e-19,
                                  rtol=2e-12)
                or not np.isclose(scale['energy_scale_J_m2'], sum(density * mass) * c**2 * length, rtol=2e-12)
                or not np.allclose(arrays['w'].reshape(2, count).sum(axis=1), density * length, rtol=2e-12)):
            raise ValueError('native pair masses, weights and energy units must share one physical loading')
        velocity = arrays['u'][:, 0] / np.sqrt(1 + np.sum((arrays['u'] / c)**2, axis=1))
        weights, speed = arrays['w'].reshape(2, count), velocity.reshape(2, count) / c
        mean = np.sum(weights * speed, axis=1) / weights.sum(axis=1)
        rms = np.sqrt(np.sum(weights * (speed - mean[:, None])**2, axis=1) / weights.sum(axis=1))
        square = np.sum((arrays['u'] / c)**2, axis=1).reshape(2, count)
        kinetic = np.sum(weights * square / (np.sqrt(1 + square) + 1), axis=1) * mass * c**2
        if (not np.allclose(mean, data['mean'][0], rtol=2e-12, atol=2e-14)
                or not np.allclose(rms, data['rms'][0], rtol=2e-12, atol=0)
                or not np.allclose(kinetic / scale['energy_scale_J_m2'], data['kinetic'][0], rtol=2e-12, atol=0)):
            raise ValueError('pair initial mean, RMS and relativistic energy must match the physical archive')
        physical_dt = setting['dt_omega0'] / scale['omega0_rad_s']
        positions = (arrays['x'][:, 0] - physical_dt / 2 * velocity + length / 2) % length - length / 2
    return positions / length


def _pair_normalize(data, setting, label):
    """Use the producer's declared SI scales, preserving the Fourier face phase."""
    scale, c = setting['normalization'], 299792458.
    energy, field, wp = (scale[key] for key in ('energy_scale_J_m2', 'field_scale_V_m', 'omega0_rad_s'))
    result = {key: value.copy() for key, value in data.items() if not key.startswith('pump_')}
    for key in ('electric', 'magnetic', 'dark', 'dark_coherent', 'kinetic', 'spread', 'balance', 'work',
                'local_spread'):
        result[key] /= energy
    for key in ('mean', 'rms', 'max_speed'):
        result[key] /= c
    result['momentum'] /= energy / c
    result['t'] *= wp
    result['mean_A'] *= wp / field
    phase = np.exp(-2j * np.pi * setting['seed_mode'] * (-.5 + 1 / setting['cells']))
    for key in ('mean_E', 'mean_D', 'mode_E', 'dark_mode_E'):
        result[key] = result[key] / field * (phase if 'mode' in key else 1)
    result['source_work'] = (-1 if label == 'coupled' else 1) * result['work']
    result['nonzero_electric'] = result['electric'] - .5 * result['mean_E']**2
    return result


def _pair_cadence(setting):
    value = setting.get('scalar_dt_omega0', setting['dt_omega0'])
    if not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0:
        raise ValueError('pair scalar cadence must be finite and positive')
    return value


def _pair_pump(data, setting, label, scalar, top, tolerance):
    """Validate every native push clock separately from expensive scalar samples."""
    keys = ('t', 'mean', 'mean_D', 'mean_A')
    names = {key for key in data if key.startswith('pump_')}
    dt, wp = setting['dt_omega0'], setting['normalization']['omega0_rad_s']
    stride = round(_pair_cadence(setting) / dt)
    if names != ({'pump_' + key for key in keys} if label == 'coupled' and stride > 1 else set()):
        raise ValueError('sparse coupled scalars require exactly four dense pump histories')
    if label != 'coupled':
        return {}
    pump = {key: data[('pump_' if stride > 1 else '') + key].copy() for key in keys}
    if any(value.dtype != np.float64 for value in pump.values()):
        raise ValueError('dense pump histories require real float64 SI arrays')
    pump['t'] *= wp
    pump['mean'] /= 299792458.
    pump['mean_D'] /= setting['normalization']['field_scale_V_m']
    pump['mean_A'] *= wp / setting['normalization']['field_scale_V_m']
    steps = round(setting['horizon_omega0'] / dt)
    if (pump['t'].shape != (steps + 1,) or pump['mean'].shape != (steps + 1, 2)
            or any(pump[key].shape != (steps + 1,) for key in ('mean_D', 'mean_A'))
            or pump['t'][0] != 0 or abs(pump['t'][-1] - setting['horizon_omega0']) > tolerance
            or not np.allclose(np.diff(pump['t']), dt, rtol=0, atol=tolerance)
            or any(not np.allclose(pump[key][::stride], scalar[key], rtol=2e-12, atol=1e-30) for key in keys)):
        raise ValueError('dense pump clocks, units and sampled scalar means must agree')
    # Equal-mass pair normalization gives J/(epsilon0*E_star*omega0)=(v_pos-v_ele)/(2c).
    current = .5 * (pump['mean'][:, 1] - pump['mean'][:, 0])
    push = pump['mean_D'][:-1] + dt / 2 * (
        setting['dark_mass_over_omega0']**2 * pump['mean_A'][:-1] - setting['eta'] * current[:-1])
    knots = np.r_[pump['t'][0], pump['t'][:-1] + dt / 2, pump['t'][-1]]
    force = np.r_[pump['mean_D'][0], push, pump['mean_D'][-1]]
    absolute = 64 * np.finfo(float).eps * np.max(abs(force))
    if (not np.allclose(top['realized_table_t'], knots, rtol=0, atol=tolerance)
            or not np.allclose(top['homogeneous_table_t'], knots, rtol=0, atol=tolerance)
            or not np.allclose(top['realized_table_D'], force, rtol=2e-12, atol=absolute)):
        raise ValueError('realized forcing must retain every native midpoint and its initial mean impulse')
    return dict(dense_pump_sha256={key: fingerprint(data[('pump_' if stride > 1 else '') + key]) for key in keys},
                realized_force_sha256=fingerprint(top['realized_table_D']))


def _pair_force(folder, setting, label, top, tolerance):
    """Check the actual archived drive, including phase origin and both endpoints."""
    source, resolution = label.split('_')
    knots, values = top[source + '_table_t'], top[source + '_table_D']
    every = max(1, round(setting['table_dt_omega0'] / setting['dt_omega0'])) if resolution == 'coarse' else 1
    indices = np.unique(np.r_[0, np.arange(1, len(knots) - 1, every), len(knots) - 1])
    with np.load(Path(folder) / 'initial_state.npz', allow_pickle=False) as state:
        times, amplitude = state['dark.times'], state['dark.amplitude']
        scale = setting['normalization']
        if (times.dtype != np.float64 or amplitude.dtype != np.float64
                or times.shape != indices.shape or amplitude.shape != (len(times), 3)
                or not np.allclose(times * scale['omega0_rad_s'], knots[indices], rtol=0, atol=tolerance)
                or not np.allclose(amplitude[:, 0] / scale['field_scale_V_m'], values[indices], rtol=2e-12, atol=0)
                or np.any(amplitude[:, 1:]) or times[0] != 0
                or times[-1] * scale['omega0_rad_s'] < setting['horizon_omega0'] - tolerance
                or float(state['dark.omega']) != 0 or float(state['dark.phase']) != 0
                or float(state['dark.eta']) != setting['eta']):
            raise ValueError('archived prescribed force must match its native table, phase and physical horizon')
    return dict(actual_force_sha256=fingerprint(amplitude), actual_force_times_sha256=fingerprint(times))


def _pair_branch_contract(native, record, label):
    setting, branch = record['settings'], native['settings']
    for key in PAIR_RUNTIME:
        if native[key] != record[key]:
            raise ValueError('pair branch runtime/source must match its complete producer')
    for key in ('parent_revision', 'cells', 'particles_per_cell_per_species', 'pair_loading', 'dt_omega0',
                'horizon_omega0', 'shape_order', 'seed_mode', 'velocity_seed_over_c', 'eta', 'dark_mass_over_omega0',
                'force_quiver_over_c', 'normalization', 'XLA_FLAGS'):
        if branch[key] != setting[key]:
            raise ValueError(f'pair branch settings differ in {key}')
    if (_pair_cadence(branch) != _pair_cadence(setting)
            or branch.get('forcing_native_dt_omega0', branch['dt_omega0']) != setting['dt_omega0']):
        raise ValueError('pair branch scalar and forcing cadences must match the producer')
    expected_model = 'DarkField' if label == 'coupled' else 'PrescribedDrive'
    if branch['label'] != label or branch['model'] != expected_model or branch['scalar_units'] != 'native SI':
        raise ValueError('pair branch model, label and native units must be explicit')
    ordinary = setting['initial_ordinary_fingerprints']
    if (not all(_hash_matches([ordinary, native['results']['initial_ordinary_fingerprints']], ordinary).values())
            or native['results']['all_step_maxima'] != record['results']['cases'][label]['all_step_maxima']):
        raise ValueError('pair branch initial fingerprints and all-step maxima must match the producer')


def _pair_branch(folder, label, record, top, tolerance):
    native, data, hashes = _pair_source(Path(folder) / label)
    _pair_branch_contract(native, record, label)
    setting, branch = record['settings'], native['settings']
    ordinary = setting['initial_ordinary_fingerprints']
    normalized = _pair_normalize(data, setting, label)
    positions = _pair_initial(Path(folder) / label, setting, ordinary, normalized)
    time, dt, cadence = normalized['t'], setting['dt_omega0'], _pair_cadence(setting)
    steps, sampling = round(setting['horizon_omega0'] / dt), round(cadence / dt)
    if (sampling < 1 or abs(sampling * dt - cadence) > tolerance or steps % sampling
            or time.shape != (steps // sampling + 1,) or time[0] != 0 or np.any(np.diff(time) <= 0)
            or not np.allclose(np.diff(time), cadence, rtol=0, atol=tolerance)
            or abs(time[-1] - setting['horizon_omega0']) > tolerance
            or branch['block_steps'] * dt <= 0 or steps % branch['block_steps'] or branch['block_steps'] % sampling):
        raise ValueError('pair native clocks/horizon/compiled blocks are incomplete')
    hashes.update(_pair_pump(data, setting, label, normalized, top, tolerance))
    if label != 'coupled':
        hashes.update(_pair_force(Path(folder) / label, setting, label, top, tolerance))
    stride = round(setting['saved_output_dt_omega0'] / cadence)
    if stride < 1 or abs(stride * cadence - setting['saved_output_dt_omega0']) > tolerance:
        raise ValueError('pair saved cadence must select integer native clocks')
    indices = np.unique(np.r_[np.arange(0, len(time), stride), len(time) - 1])
    for key, value in normalized.items():
        if value.shape[0] != len(time) or not np.allclose(
                top[f'{label}_{key}'], value[indices], rtol=2e-12, atol=1e-30):
            raise ValueError(f'pair normalized/native scalar mismatch: {label} {key}')
    source = normalized['work'] if label != 'coupled' else 0
    balance = (sum(normalized[key] for key in ('electric', 'magnetic', 'dark'))
               + normalized['kinetic'].sum(axis=1) - source)
    if (not np.allclose(normalized['balance'], balance, rtol=2e-12, atol=1e-30)
            or not np.allclose(normalized['spread'], .25 * normalized['rms']**2, rtol=2e-12, atol=1e-30)
            or normalized['local_spread'].shape != (len(time), 2, len(setting['local_spread_lengths_c_over_omega0']))):
        raise ValueError('pair energy/work and fixed-scale moments must share the physical normalization')
    return native, normalized, hashes, positions if label == 'coupled' else None


def _pair_load(folder, tolerance=1e-8):
    record, top, hashes = _pair_source(folder)
    setting = record['settings']
    if (record['example'] != 'pair_waveform_control' or not setting['local_moments']
            or setting['native_diagnostic_dt_omega0'] != setting['dt_omega0']
            or setting.get('forcing_native_dt_omega0', setting['dt_omega0']) != setting['dt_omega0']
            or setting['seed_mode'] * 2 >= setting['cells']
            or min(setting['normalization'].values()) <= 0):
        raise ValueError('complete pair controls require resolved modes and native physical-scale moments')
    branches = {label: _pair_branch(folder, label, record, top, tolerance) for label in PAIR_CASES}
    if len({branch[0]['settings']['block_steps'] for branch in branches.values()}) != 1:
        raise ValueError('all pair branches must share the execution partition')
    time = top['coupled_t']
    reservoir = setting['initial_dark_energy_over_energy_scale']
    if (not np.isclose(branches['coupled'][1]['dark'][0], reservoir, rtol=2e-12, atol=0)
            or not np.allclose(top['coupled_total_dark_fraction'], top['coupled_dark'] / reservoir, rtol=2e-12)
            or not np.allclose(top['coupled_coherent_dark_fraction'],
                               top['coupled_dark_coherent'] / reservoir, rtol=2e-12)
            or not np.allclose(top['coupled_nonzero_dark_fraction'], top['coupled_total_dark_fraction']
                               - top['coupled_coherent_dark_fraction'], rtol=2e-12, atol=1e-15)
            or not np.allclose(top['additional_dark_depletion_fraction'], top['homogeneous_dark_fraction']
                               - top['coupled_total_dark_fraction'], rtol=2e-12, atol=1e-15)
            or not np.allclose(top['additional_dark_depletion_gain'], top['additional_dark_depletion_fraction']
                               - top['additional_dark_depletion_fraction'][0], rtol=2e-12, atol=1e-15)):
        raise ValueError('pair reservoir, preparation offset and depletion curves must remain aligned')
    linear_time = top['linear_t']
    selected = time <= setting['linear_reference_horizon_omega0'] + tolerance
    if (linear_time.shape != time[selected].shape
            or not np.allclose(linear_time, time[selected], rtol=0, atol=tolerance)
            or any(top[f'linear_{label}_mode_E'].shape != linear_time.shape
                   for label in ('coupled', 'homogeneous_fine'))):
        raise ValueError('independent linear traces must use the saved native early clocks')
    return record, top, hashes, branches


def _pair_metric(first, second):
    result = metrics(first, second)
    if np.iscomplexobj(first):
        for key in ('reference_mean', 'comparison_mean'):
            value = np.asarray(result[key])
            result[key] = dict(real=value.real.tolist(), imag=value.imag.tolist())
    return result


PAIR_MAXIMA = ('energy_work_over_energy_scale', 'momentum_over_energy_scale_over_c', 'charge_over_enL',
               'continuity_over_enomega0', 'ordinary_gauss_over_en_eps0', 'dark_gauss_over_en_eps0',
               'grid_charge_over_enL', 'dark_sector_work_over_energy_scale', 'ordinary_sector_work_over_energy_scale')


def _pair_prescribed_load(folder, example='pair_waveform_repeat', labels=('realized_fine',)):
    """Validate complete prescribed endpoints and native SI ledgers without evolving particles."""
    record, raw, hashes = _pair_source(folder)
    s, results = dict(record['settings']), record['results']
    repeat = example != 'pair_waveform_branch'
    if not repeat:
        control = json.loads((Path(folder).parent / 'run.json').read_text())
        _pair_branch_contract(record, control, s['label'])
        s.update({key: control['settings'][key] for key in ('local_moments', 'local_spread_lengths_c_over_omega0')})
    interventions = {'pair_waveform_table_replay': 'table_replay', 'pair_waveform_pic_dt_replay': 'pic_dt_replay'}
    declared = {interventions[example]} if example in interventions else set()
    if (example not in ('pair_waveform_repeat', 'pair_waveform_branch', *interventions)
            or record['example'] != example
            or set(s) & {'table_replay', 'pic_dt_replay'} != declared
            or 'force_extension' in s and example != 'pair_waveform_pic_dt_replay'
            or any(not isinstance(s[key], dict) for key in set(s) & {'table_replay', 'pic_dt_replay'})
            or s['model'] != 'PrescribedDrive' or s['label'] not in labels or s['scalar_units'] != 'native SI'
            or not s.get('local_moments', repeat) or s['shape_order'] != 2
            or 2 * s['seed_mode'] >= s['cells'] or min(s['normalization'].values()) <= 0):
        raise ValueError('prescribed pair comparisons require complete quadratic realized-force branches '
                         'and physical moments')
    states = [_archive(Path(folder) / name, legacy_shape=s['shape_order'])
              for name in ('initial_state.npz', 'final_state.npz')]
    with np.load(Path(folder) / 'final_state.npz', allow_pickle=False) as archive:
        native_final_keys = archive.files
    initial, final = states
    evolving = {'E', 'x', 'u', 'rho', 'time', 'steps', 'dark.work', 'dark.max_balance_error',
                'dark.max_ordinary_gauss', 'dark.max_dark_gauss'}
    _same_leaves({key: value for key, value in initial.items() if key not in evolving},
                 {key: value for key, value in final.items() if key not in evolving})
    hashes.update({name + '_sha256': hashlib.sha256((Path(folder) / (name + '.npz')).read_bytes()).hexdigest()
                   for name in ('initial_state', 'final_state')})
    wp, field, energy = [s['normalization'][key] for key in ('omega0_rad_s', 'field_scale_V_m', 'energy_scale_J_m2')]
    dt, cadence, horizon = s['dt_omega0'], _pair_cadence(s), s['horizon_omega0']
    forcing_dt = s['pic_dt_replay'].get('donor_dt_omega0') if example == 'pair_waveform_pic_dt_replay' else dt
    steps, stride = round(horizon / dt), round(cadence / dt)
    if (not np.isfinite([dt, cadence, horizon]).all() or min(dt, cadence) <= 0 or horizon < 170
            or type(s['block_steps']) is not int or s['block_steps'] < 1 or stride < 1
            or steps % s['block_steps'] or s['block_steps'] % stride
            or abs(steps * dt - horizon) > 1e-10 or abs(stride * dt - cadence) > 1e-10
            or s['forcing_native_dt_omega0'] != forcing_dt):
        raise ValueError('prescribed pair comparisons require complete native blocks, fixed windows '
                         'and recorded cadences')
    ticks = _accepted_ticks(float(initial['dark.dt']), steps, stride)
    times, force = initial['dark.times'], initial['dark.amplitude']
    if (len(initial) != 52 or len(final) != 52 or float(initial['dark.dt']) != dt / wp
            or not np.array_equal(raw['t'], ticks)
            or initial['time'] != 0 or initial['steps'] != 0 or initial['dark.work'] != 0
            or initial['dark.initial_dark'] != 0 or initial['dark.initial_projection_norm'] != 0
            or not np.isclose(initial['dark.initial_ordinary'], raw['balance'][0], rtol=2e-12, atol=0)
            or final['steps'] != steps or final['time'] != ticks[-1]
            or times.ndim != 1 or force.shape != (len(times), 3) or times[0] != 0 or np.any(np.diff(times) <= 0)
            or np.nextafter(float(times[-1]), np.inf) < ticks[-1] or np.any(force[:, 1:])
            or any(state['format'] != 1 or state['dark.format'] != 3 or state['dark.mode'] != 'drive'
                   or state['algorithm'] != 'explicit' or state['shape_order'] != s['shape_order']
                   or state['cells'] != s['cells'] or state['dark.cells'] != s['cells']
                   or state['dark.eta'] != s['eta'] or state['dark.omega'] != 0 or state['dark.phase'] != 0
                   or state['dark.has_external_B'] or np.any(state['B']) or np.any(state['E'][:, 1:])
                   for state in states)):
        raise ValueError('pair restart model/forcing or exact accepted scalar clocks disagree')
    s['length_c_over_omega0'] = float(initial['dark.length']) * wp / 299792458.
    required = {'t', 'mean', 'rms', 'max_speed', 'momentum', 'mean_E', 'mean_D', 'mean_A', 'mode_E', 'dark_mode_E',
                'kinetic', 'electric', 'magnetic', 'dark', 'dark_coherent', 'work', 'spread', 'local_spread',
                'density_rms', 'local_density_rms', 'charge', 'grid_charge', 'ordinary_gauss', 'dark_gauss', 'balance'}
    if raw.keys() != required:
        raise ValueError('prescribed pair comparisons must retain every native physical scalar observable')
    data = _pair_normalize(raw, s, s['label'])
    shape = dict(mean=(2,), rms=(2,), kinetic=(2,), spread=(2,), momentum=(3,), density_rms=(2,),
                 local_spread=(2, len(s['local_spread_lengths_c_over_omega0'])),
                 local_density_rms=(2, len(s['local_spread_lengths_c_over_omega0'])))
    if (any(value.shape != (len(ticks), *shape.get(key, ())) for key, value in data.items())
            or any(np.any(data[key]) for key in
                   ('dark', 'dark_coherent', 'dark_gauss', 'dark_mode_E', 'mean_D', 'mean_A'))
            or not np.allclose(data['spread'], .25 * data['rms']**2, rtol=2e-12, atol=1e-30)
            or not np.allclose(data['balance'], data['electric'] + data['magnetic']
                               + data['kinetic'].sum(axis=1) - data['work'], rtol=2e-12, atol=1e-30)):
        raise ValueError('pair scalar shapes, energy/work or physical moments disagree')
    length = float(initial['dark.length'])
    en = float(initial['dark.density'].sum()) * 1.602176634e-19  # Total pair number density.
    eps = energy / (length * field**2)  # Declared pair U*=epsilon0 L E*²; no fit to fields or charge.
    units = np.asarray([energy, energy / 299792458., en * length, en * wp, en / eps, en / eps,
                        en * length, energy, energy])
    _pair_repeat_bounds(results, final, units, data, repeat)
    _pair_repeat_endpoints(folder, states, s, data, raw)
    if any(not np.allclose(results[key], data['local_spread'][index], rtol=2e-12, atol=1e-30)
           for key, index in (('local_spread_initial', 0), ('local_spread_final', -1))):
        raise ValueError('pair recorded local spread endpoints differ from normalized SI moments')
    for key in ('charge', 'grid_charge', 'ordinary_gauss', 'dark_gauss'):
        data[key] /= en * length if 'charge' in key else en / eps
    diagnostics = dict(
        actual_force_sha256=fingerprint(force), actual_force_times_sha256=fingerprint(times),
        native_scalar_fingerprints={key: fingerprint(value) for key, value in raw.items()},
        final_leaves={key: fingerprint(final[key]) for key in native_final_keys})
    if 'shape_order' not in native_final_keys:
        diagnostics['legacy_shape_order_default'] = 2
    diagnostics.update({'initial_leaves_without_table': {
        key: fingerprint(value) for key, value in initial.items() if key not in ('dark.times', 'dark.amplitude')}}
        if example != 'pair_waveform_repeat' else {})
    return record, data, hashes, diagnostics


def _pair_repeat_bounds(results, final, units, data, repeat, envelope=False):
    maxima = results['all_step_maxima']
    if (set(maxima) != set(PAIR_MAXIMA) or any(not np.isfinite(value) or value < 0 for value in maxima.values())
            or repeat and not np.allclose(results['all_step_maxima_SI'],
                                          np.array([maxima[key] for key in PAIR_MAXIMA]) * units, rtol=2e-12, atol=0)):
        raise ValueError('prescribed pair comparisons require all nine native SI conservation maxima')
    for key, index in (('dark.max_balance_error', 0), ('dark.max_ordinary_gauss', 4), ('dark.max_dark_gauss', 5)):
        valid = (final[key] <= results['all_step_maxima_SI'][index] if envelope else
                 np.isclose(final[key], maxima[PAIR_MAXIMA[index]] * units[index], rtol=2e-12, atol=0))
        if not valid:
            raise ValueError('pair archived all-step bounds differ from native maxima')
    defect = max(maxima[key] for key in (PAIR_MAXIMA[0], *PAIR_MAXIMA[7:]))
    if np.max(abs(data['balance'] - data['balance'][0])) > defect + 2e-14:
        raise ValueError('pair native sampled ledger exceeds its all-step bound')


def _pair_repeat_endpoints(folder, states, setting, data, raw):
    """Check both endpoints using physical species weights and relativistic moments."""
    count = setting['cells'] * setting['particles_per_cell_per_species']
    field, energy = [setting['normalization'][key] for key in ('field_scale_V_m', 'energy_scale_J_m2')]
    for state, index, filename in zip(states, (0, -1), ('initial_state.npz', 'final_state.npz')):
        hashes = {key: hashlib.sha256(
            f'{state[key].dtype.str}:{state[key].shape}'.encode() +
            np.ascontiguousarray(state[key]).tobytes()).hexdigest()
            for key in ('x', 'u', 'w', 'E', 'B', 'rho')}
        if index == 0:
            expected = json.loads((Path(folder) / 'run.json').read_text())['results']['initial_ordinary_fingerprints']
            if hashes != expected:
                raise ValueError('pair initial native ordinary fingerprints disagree')
        _pair_initial(folder, setting, hashes, {key: value[index][None] for key, value in data.items()}, filename)
        if (not np.array_equal(state['counts'], [count, count])
                or not np.allclose(state['qm'], np.repeat(state['dark.charge'] / state['dark.mass'], count)
                                   * 1.602176634e-19, rtol=2e-12, atol=0)):
            raise ValueError('pair restart species counts and physical charge/mass disagree')
        electric = .5 * np.mean(np.sum((state['E'] / field)**2, axis=1))
        mode = np.fft.fft(state['E'][:, 0])[setting['seed_mode']] / setting['cells'] / field
        mode *= np.exp(-2j * np.pi * setting['seed_mode'] * (-.5 + 1 / setting['cells']))
        momentum = np.einsum('s,sp,spc->c', state['dark.mass'], state['w'].reshape(2, count),
                             state['u'].reshape(2, count, 3)) / (energy / 299792458.)
        if (not np.allclose(electric, data['electric'][index], rtol=2e-12, atol=2e-14)
                or not np.allclose(np.mean(state['E'][:, 0]) / field, data['mean_E'][index], rtol=2e-12, atol=2e-14)
                or not np.allclose(mode, data['mode_E'][index], rtol=2e-12, atol=2e-14)
                or not np.allclose(momentum, data['momentum'][index], rtol=2e-12, atol=2e-13)
                or state['dark.work'] != raw['work'][index]):
            raise ValueError('pair endpoint field, momentum or work differs from native scalars')


def _pair_repeat_windows(a, b, conservation_budget=.001):
    rows, windows = [a[1], b[1]], []
    defects = [max(source[0]['results']['all_step_maxima'][key] for key in (PAIR_MAXIMA[0], *PAIR_MAXIMA[7:]))
               for source in (a, b)]
    for start, end in ((0., float(rows[0]['t'][-1])), (150., 170.)):
        selected = (rows[0]['t'] >= start - 1e-8) & (rows[0]['t'] <= end + 1e-8)
        summaries = [window_summary(row, selected) for row in rows]
        phases = np.angle(rows[1]['mode_E'][selected] * rows[0]['mode_E'][selected].conj())
        weights = abs(rows[0]['mode_E'][selected])**2
        valid = weights > 0
        valid &= abs(rows[1]['mode_E'][selected]) > 0
        transfer = [defect / abs(row['work_increment']) if row['work_increment'] else None
                    for defect, row in zip(defects, summaries)]
        rates = [2 * defect / (float(np.ptp(data['t'][selected])) * row['plasma_mean_energy'])
                 for defect, data, row in zip(defects, rows, summaries)]
        difference = abs(summaries[1]['work_increment'] - summaries[0]['work_increment'])
        rate_difference = abs(summaries[1]['injection_rate_over_wp'] - summaries[0]['injection_rate_over_wp'])
        effect = sum(defects) / difference if difference else None
        accounting = sum(rates) / rate_difference if rate_difference else None
        observables = {key: _pair_metric(rows[0][key][selected], rows[1][key][selected]) for key in
                       ('mean_E', 'mode_E', 'electric', 'nonzero_electric', 'rms', 'work', 'momentum',
                        'local_spread', 'local_density_rms')}
        observables.update({key + '_increment': _pair_metric(rows[0][key][selected] - rows[0][key][0],
                                                             rows[1][key][selected] - rows[1][key][0])
                            for key in ('momentum', 'local_spread')})
        phase = dict(alignment=False, reference_amplitude_squared_weighted_rms_rad=float(np.sqrt(
            np.sum(weights[valid] * phases[valid]**2) / weights[valid].sum())) if weights[valid].sum() else None,
            reference_weight_fraction_with_defined_phase=float(weights[valid].sum() / weights.sum())
            if weights.sum() else None, endpoint_rad=float(phases[-1]) if valid[-1] else None)
        budget = dict(global_energy_sector_defects=defects,
                      global_defect_over_abs_window_work=transfer,
                      two_endpoint_bound_over_abs_window_work=[2 * x if x is not None else None for x in transfer],
                      transfer_gates=[x is not None and x <= conservation_budget for x in transfer],
                      two_endpoint_transfer_gates=[x is not None and 2 * x <= conservation_budget for x in transfer],
                      global_defect_sum_over_abs_work_difference=effect,
                      two_endpoint_bound_over_abs_work_difference=2 * effect if effect is not None else None,
                      difference_gate=effect is not None and effect <= conservation_budget,
                      two_endpoint_difference_gate=effect is not None and 2 * effect <= conservation_budget,
                      absolute_rate_accounting_envelope_over_omega0=rates,
                      accounting_over_abs_rate_difference=accounting,
                      rate_difference_gate=accounting is not None and accounting <= 1 / 3)
        windows.append(dict(bounds_omega0=[start, end], samples=int(selected.sum()),
                            actual_clocks_omega0=rows[0]['t'][selected][[0, -1]].tolist(), reductions=summaries,
                            observables=observables, mode_phase=phase, accounting=budget))
    return windows


def pair_repeat_comparison(folders, observer=None, donor=None, transition=()):
    """Exact-state execution evidence; different donor sources require explicit review."""
    if len(folders) != 2:
        raise ValueError('pair repeats require exactly two complete execution folders')
    sources = [_pair_prescribed_load(path) for path in folders]
    a, b = sources
    if (any(a[0][key] != b[0][key] for key in PAIR_RUNTIME) or a[0]['settings'] != b[0]['settings']
            or a[2]['initial_state_sha256'] != b[2]['initial_state_sha256']
            or not np.array_equal(a[1]['t'], b[1]['t'])
            or any(source[0]['results'].get('execution_index') != index
                   or source[0]['results'].get('samples') != 2 for index, source in enumerate(sources, 1))):
        raise ValueError('pair repeats require identical source/runtime/protocol, complete initial files and clocks')
    s = a[0]['settings']
    if any(not isinstance(s.get(key), str) or len(s[key]) != length or not set(s[key]) <= set('0123456789abcdef')
           for key, length in (('donor_git', 40), ('initial_state_archive_sha256', 64),
                               ('donor_record_sha256', 64), ('donor_control_record_sha256', 64),
                               ('producer_script_sha256', 64))) or (
            s['initial_state_archive_sha256'] != a[2]['initial_state_sha256']):
        raise ValueError('pair repeat restored state and producer/donor SHA declarations disagree')
    result = dict(comparison='exact-state pair execution repeat', windows=_pair_repeat_windows(a, b),
                  native_runs=[source[0] for source in sources], native_sha256=[source[2] for source in sources],
                  archive_diagnostics=[source[3] for source in sources],
                  all_scalar_histories_bitwise_equal=(a[3]['native_scalar_fingerprints']
                                                      == b[3]['native_scalar_fingerprints']),
                  nominal_clock_bound_omega0=1e-9,
                  nominal_clock_errors_omega0=[abs(source[1]['t'][-1] - source[0]['settings']['horizon_omega0'])
                                               for source in sources],
                  nominal_clock_gates=[bool(abs(source[1]['t'][-1] - source[0]['settings']['horizon_omega0']) <= 1e-9)
                                       for source in sources],
                  original_gauss_bound_over_en_eps0=2e-13,
                  original_gauss_gates=[{key: source[0]['results']['all_step_maxima'][key] <= 2e-13
                                         for key in PAIR_MAXIMA[4:6]} for source in sources],
                  conservation_budget=.001, rate_accounting_budget=1 / 3,
                  notes='Raw native samples and Fourier face phase; no interpolation or alignment. Rates use omega0. '
                        'Global bounds remain global; accounting envelopes do not bound discretization or execution '
                        'uncertainty. Tiny repeat differences are not a physical effect or convergence certification.')
    if donor is not None:
        old = _pair_prescribed_load(donor, 'pair_waveform_branch')
        same_source = old[0]['git'] == a[0]['git']
        if ((bool(transition) if same_source else tuple(transition) != (old[0]['git'], a[0]['git']))
                or s['donor_git'] != old[0]['git'] or s['donor_record_sha256'] != old[2]['run_sha256']
                or s['donor_control_record_sha256'] != hashlib.sha256(
                    (Path(donor).parent / 'run.json').read_bytes()).hexdigest()
                or old[2]['initial_state_sha256'] != a[2]['initial_state_sha256']
                or any(old[0][key] != a[0][key] for key in PAIR_RUNTIME[1:])
                or any(s.get(key) != value for key, value in old[0]['settings'].items())
                or not np.array_equal(old[1]['t'], a[1]['t'])):
            raise ValueError('donor needs exact inputs and an explicit reviewed transition only for differing sources')
        result['donor'] = dict(reviewed_source_transition=list(transition) if not same_source else None,
                               native_run=old[0], native_sha256=old[2],
                               original_gauss_gates={key: old[0]['results']['all_step_maxima'][key] <= 2e-13
                                                     for key in PAIR_MAXIMA[4:6]},
                               comparisons=[_pair_repeat_windows(old, source) for source in sources],
                               claim=('Same producer source; donor executable identity is unobserved.' if same_source
                                      else 'Different producer sources; reviewed algorithm-matched only. '
                                           'No shared executable claim.'))
    elif transition:
        raise ValueError('a reviewed donor transition requires its native donor folder')
    if observer is not None:
        observed = json.loads(Path(observer).read_text())
        entries = observed.get('compilations', [])
        if (observed.get('source') != a[0]['git'] or observed.get('completed') is not True or len(entries) != 2
                or observed.get('producer_script_sha256') != s['producer_script_sha256']
                or any(not isinstance(observed.get(key), str) or not observed[key]
                       for key in ('method', 'timing_note', 'interpretation'))
                or 'script_sha256' in observed and (
                    not isinstance(observed['script_sha256'], str) or len(observed['script_sha256']) != 64
                    or not set(observed['script_sha256']) <= set('0123456789abcdef'))
                or any(entry.get('index') != index or entry.get('runtime_available') is not True
                       or entry.get('same_loaded_executable_as_first') is not (index == 2)
                       or not np.isfinite(entry.get('observer_seconds', np.nan)) or entry['observer_seconds'] < 0
                       for index, entry in enumerate(entries, 1))
                or not all(_hash_matches(entries, ('stablehlo_sha256', 'optimized_hlo_text_sha256')).values())):
            raise ValueError('observer source, identity assertions or text hashes disagree')
        fields = ('index', 'runtime_available', 'same_loaded_executable_as_first', 'stablehlo_sha256',
                  'optimized_hlo_text_sha256', 'observer_seconds')
        result['observer'] = dict(source=observed['source'], wrapper_script_sha256=observed.get('script_sha256'),
                                  producer_script_sha256=observed['producer_script_sha256'],
                                  compilations=[{key: entry[key] for key in fields} for entry in entries],
                                  observer_sha256=hashlib.sha256(Path(observer).read_bytes()).hexdigest(),
                                  claim='Observed runtime-object identity in one process is nonserializable. '
                                        'Equal HLO text hashes are not binary identity or bitwise trajectories.',
                                  method=observed['method'], timing_note=observed['timing_note'],
                                  native_interpretation=observed['interpretation'])
    return result


def _pair_fine_forcing(folder):
    """Bind the fine archived force to its original coupled current and accepted clocks."""
    control, top, control_hashes = _pair_source(Path(folder).parent)
    pump, raw, pump_hashes = _pair_source(Path(folder).parent / 'coupled')
    _pair_branch_contract(pump, control, 'coupled')
    pump_hashes.update(_pair_pump(
        raw, control['settings'], 'coupled', _pair_normalize(raw, control['settings'], 'coupled'), top, 1e-8))
    with np.load(Path(folder) / 'initial_state.npz', allow_pickle=False) as stored:
        times, force = stored['dark.times'], stored['dark.amplitude']
    s = control['settings']
    wp = s['normalization']['omega0_rad_s']
    ticks = _accepted_ticks(s['dt_omega0'] / wp, round(s['horizon_omega0'] / s['dt_omega0']), 1)
    knots = np.r_[0., ticks[:-1] + s['dt_omega0'] / (2 * wp), ticks[-1]]
    if not np.array_equal(times, knots):
        raise ValueError('fine table must contain every exact accepted midpoint and both endpoints')
    _pair_force(folder, s, 'realized_fine', top, 1e-8)
    return control, top, control_hashes, pump_hashes, times, force


def _pair_native_hash(value):
    return hashlib.sha256(f'{value.dtype.str}:{value.shape}'.encode()
                          + np.ascontiguousarray(value).tobytes()).hexdigest()


def _pair_extension(fine, folder, transition=()):  # noqa: C901 — scoped complete-checkpoint lineage gates
    """Validate one accepted coupled step; ordinary table readers never accept tails."""
    fine, folder = Path(fine), Path(folder)
    control, _, control_hashes, _, times, force = _pair_fine_forcing(fine)
    donor, origin, donor_hashes = _pair_source(fine.parent / 'coupled')
    record, raw, hashes = _pair_source(folder)
    s, detail = record['settings'], record['settings'].get('force_extension', {})
    if (record['example'] != 'pair_waveform_force_extension'
            or tuple(transition) != (() if record['git'] == donor['git'] else (donor['git'], record['git']))
            or detail.get('reviewed_source_transition') != list(transition)
            or any(record[key] != donor[key] for key in PAIR_RUNTIME[1:])
            or s.get('label') != 'coupled' or s.get('model') != 'DarkField'
            or any(s.get(key) != control['settings'].get(key) for key in (
                *PAIR_PHYSICS, 'cells', 'particles_per_cell_per_species', 'dt_omega0', 'horizon_omega0',
                'velocity_seed_over_c', 'scalar_dt_omega0', 'forcing_native_dt_omega0', 'local_moments'))
            or detail.get('accepted_steps') != 1 or detail.get('native_scalar_dt_omega0') != s['dt_omega0']):
        raise ValueError('force extension requires one accepted step and explicitly reviewed source/runtime/physics')
    files = (('force_run', fine / 'run.json'), ('force_initial', fine / 'initial_state.npz'),
             ('force_data', fine / 'data.npz'), ('control_run', fine.parent / 'run.json'),
             ('control_data', fine.parent / 'data.npz'), ('coupled_final', fine.parent / 'coupled' / 'final_state.npz'),
             ('coupled_initial', fine.parent / 'coupled' / 'initial_state.npz'))
    expected_hashes = dict(donor_hashes)
    expected_hashes.update({name + '_sha256': hashlib.sha256(path.read_bytes()).hexdigest() for name, path in files})
    if (detail.get('donor_sha256') != expected_hashes or detail.get('prefix_knots') != len(times)
            or detail.get('prefix_times_sha256') != _pair_native_hash(times)
            or detail.get('prefix_amplitude_sha256') != _pair_native_hash(force)):
        raise ValueError('force extension donor files and exact original prefix must be bound by hashes')
    states = [_archive(folder / name, legacy_shape=2, field=True)
              for name in ('initial_state.npz', 'final_state.npz')]
    with np.load(fine.parent / 'coupled' / 'final_state.npz', allow_pickle=False) as old, np.load(
            folder / 'initial_state.npz', allow_pickle=False) as restored:
        _same_leaves(dict(old), dict(restored))
        native_leaves = {key: fingerprint(restored[key]) for key in restored.files}
    start, final = states
    evolving = {'E', 'x', 'u', 'rho', 'time', 'steps', 'dark.E', 'dark.B', 'dark.A', 'dark.phi', 'dark.work',
                'dark.max_balance_error', 'dark.max_ordinary_gauss', 'dark.max_dark_gauss'}
    _same_leaves({key: value for key, value in start.items() if key not in evolving},
                 {key: value for key, value in final.items() if key not in evolving})
    wp, field_scale, energy = [s['normalization'][key] for key in (
        'omega0_rad_s', 'field_scale_V_m', 'energy_scale_J_m2')]
    dt, length = float(start['dark.dt']), float(start['dark.length'])
    scalar_keys = {key for key in origin if not key.startswith('pump_')}
    if (len(start) != 53 or len(final) != 53 or start['time'] != times[-1] or start['steps'] != len(times) - 2
            or final['steps'] != start['steps'] + 1 or final['time'] != float(start['time']) + dt
            or raw.keys() != scalar_keys | {'force_times', 'force_amplitude'}
            or any(raw[key].shape != (2, *origin[key].shape[1:]) for key in scalar_keys)
            or not np.array_equal(raw['t'], [float(start['time']), float(final['time'])])
            or any(state['dark.mode'] != 'field' or state['dark.format'] != 2 or state['shape_order'] != 2
                   or state['algorithm'] != 'explicit' or state['dark.dt'] != s['dt_omega0'] / wp
                   or state['cells'] != s['cells'] or state['dark.cells'] != s['cells']
                   or state['dark.omega'] != s['dark_mass_over_omega0'] * wp or state['dark.eta'] != s['eta']
                   or state['dark.has_external_B'] or np.any(state['B']) or np.any(state['dark.B'])
                   or any(state[key].shape != (s['cells'], 3) for key in ('dark.E', 'dark.B', 'dark.A'))
                   or state['dark.phi'].shape != (s['cells'],)
                   or any(np.any(state[key][:, 1:]) for key in ('u', 'E', 'dark.E', 'dark.A')) for state in states)):
        raise ValueError('force extension complete native model, accepted clock or field restart disagrees')
    data = _pair_normalize({key: value for key, value in raw.items() if not key.startswith('force_')}, s, 'coupled')
    count, dx = s['cells'] * s['particles_per_cell_per_species'], length / s['cells']
    eps, c = energy / (length * field_scale**2), 299792458.
    s_physical = dict(s, length_c_over_omega0=length * wp / c)
    for state, index, filename in zip(states, (0, -1), ('initial_state.npz', 'final_state.npz')):
        ordinary = {key: _pair_native_hash(state[key]) for key in ('x', 'u', 'w', 'E', 'B', 'rho')}
        _pair_initial(folder, s_physical, ordinary, {key: value[index][None] for key, value in data.items()}, filename)
        dark = .5 * eps * dx * (np.sum(state['dark.E']**2 + state['dark.omega']**2 * state['dark.A']**2)
                                + state['dark.omega']**2 / c**2 * np.sum(state['dark.phi']**2))
        momentum = np.einsum('s,sp,spc->c', state['dark.mass'], state['w'].reshape(2, count),
                             state['u'].reshape(2, count, 3))
        centred_A = (state['dark.A'] + np.roll(state['dark.A'], 1, axis=0)) / 2
        momentum += eps * state['dark.omega']**2 / c**2 * dx * np.sum(state['dark.phi'][:, None] * centred_A, axis=0)
        electric = .5 * eps * dx * np.sum(state['E']**2)
        if (not np.allclose([electric, dark], [raw['electric'][index], raw['dark'][index]],
                            rtol=2e-12, atol=2e-14 * energy)
                or not np.allclose([np.mean(state['dark.E'][:, 0]) / field_scale,
                                    np.mean(state['dark.A'][:, 0]) * wp / field_scale],
                                   [raw['mean_D'][index] / field_scale, raw['mean_A'][index] * wp / field_scale],
                                   rtol=2e-12, atol=2e-14)
                or not np.allclose(momentum, raw['momentum'][index], rtol=2e-12, atol=2e-13 * energy / c)
                or state['dark.work'] != raw['work'][index]):
            raise ValueError('force extension endpoint energy, momentum or unchanged work ledger disagrees')
    en = float(start['dark.density'].sum()) * 1.602176634e-19
    units = np.asarray([energy, energy / c, en * length, en * wp, en / eps, en / eps, en * length, energy, energy])
    keys = ('max_balance_error', 'max_ordinary_gauss', 'max_dark_gauss')
    native_maxima = [float(final['dark.' + key]) for key in keys]
    if any(start['dark.' + key] < 0 or end < start['dark.' + key] for key, end in zip(keys, native_maxima)):
        raise ValueError('force extension checkpoint maxima must remain nonnegative and nondecreasing')
    _pair_repeat_bounds(record['results'], final, units, data, True, envelope=True)
    if record['results'].get('checkpoint_maxima_SI', native_maxima) != native_maxima:
        raise ValueError('force extension exact checkpoint maxima disagree with the native archive')
    total = raw['electric'] + raw['dark'] + raw['kinetic'].sum(axis=1)
    ordinary = raw['electric'] + raw['kinetic'].sum(axis=1)
    if (any(record['results']['all_step_maxima'][key] < donor['results']['all_step_maxima'][key] for key in PAIR_MAXIMA)
            or np.max(abs(raw['balance'] - origin['balance'][0])) > record['results']['all_step_maxima_SI'][0]
            + 2e-14 * energy or not np.allclose(raw['balance'], total, rtol=2e-12, atol=2e-14 * energy)
            or not np.allclose([start['dark.initial_ordinary'], start['dark.initial_dark']],
                               [origin['electric'][0] + origin['kinetic'][0].sum(), origin['dark'][0]], rtol=2e-12)
            or np.max(abs(raw['dark'] - start['dark.initial_dark'] - raw['work']))
            > record['results']['all_step_maxima_SI'][7] + 2e-14 * energy
            or np.max(abs(ordinary - start['dark.initial_ordinary'] + raw['work']))
            > record['results']['all_step_maxima_SI'][8] + 2e-14 * energy):
        raise ValueError('force extension must retain all original nine maxima and the global energy reference')
    charge_density = start['w'].reshape(2, count).sum(axis=1) / length * start['dark.charge'] * 1.602176634e-19
    source = float(start['dark.omega'])**2 * raw['mean_A'][0] - s['eta'] * (raw['mean'][0] @ charge_density) / eps
    push = raw['mean_D'][0] + dt / 2 * source
    tail_t = np.array([float(start['time']) + dt / 2, float(final['time'])])
    tail_a = np.column_stack(([push, raw['mean_D'][-1]], np.zeros((2, 2))))
    expected_t, expected_a = np.r_[times, tail_t], np.concatenate((force, tail_a))
    if (raw['force_amplitude'][-1, 0] != raw['mean_D'][-1]
            or not np.array_equal(raw['force_times'], expected_t)) or not np.allclose(
            raw['force_amplitude'], expected_a, rtol=2e-12, atol=2e-14 * field_scale):
        raise ValueError('force extension must append exactly two genuine accepted force samples')
    _same_leaves({'times': raw['force_times'][:len(times)], 'force': raw['force_amplitude'][:len(times)]},
                 {'times': times, 'force': force})
    hashes.update({name + '_sha256': hashlib.sha256((folder / (name + '.npz')).read_bytes()).hexdigest()
                   for name in ('initial_state', 'final_state')})
    return raw['force_times'], raw['force_amplitude'], dict(
        native_run=record, native_sha256=hashes, donor_sha256=expected_hashes, restored_native_leaves=native_leaves,
        prefix_knots=len(times), appended_knots=2, old_terminal_s=float(times[-1]),
        new_terminal_s=float(final['time']), original_gauss_gates={
            key: record['results']['all_step_maxima'][key] <= 2e-13 for key in PAIR_MAXIMA[4:6]},
        claim='Exact native restoration and unchanged force prefix; '
        'one genuine accepted continuation step from reviewed sources, separately compiled.')


def _pair_table_forces(folders, sources):  # noqa: C901 — separate native-table integrity gates
    """Check the actual donor force, accepted clocks and replay decimation separately."""
    control, top, control_hashes, pump_hashes, times, force = _pair_fine_forcing(folders[2])
    tables = []
    for folder in folders:
        with np.load(Path(folder) / 'initial_state.npz', allow_pickle=False) as stored:
            tables.append((stored['dark.times'], stored['dark.amplitude']))
    s, replay = control['settings'], sources[1][0]['settings']['table_replay']
    wp = s['normalization']['omega0_rad_s']
    knots = times
    for index in (0, 2):
        _pair_force(folders[index], control['settings'], sources[index][0]['settings']['label'], top, 1e-8)
    every = replay.get('table_every')
    if type(every) is not int or every < 2:
        raise ValueError('table replay needs a positive integer decimation greater than one')
    selected = np.unique(np.r_[0, np.arange(1, len(times) - 1, every), len(times) - 1])

    expected = dict(original_knots=len(times), retained_knots=len(selected),
                    original_times_sha256=_pair_native_hash(times), original_amplitude_sha256=_pair_native_hash(force),
                    times_sha256=_pair_native_hash(times[selected]),
                    amplitude_sha256=_pair_native_hash(force[selected]),
                    selection='initial endpoint, first midpoint, every nth midpoint, terminal endpoint')
    if (any(replay.get(key) != value for key, value in expected.items())
            or not np.array_equal(tables[1][0], times[selected])
            or not np.array_equal(tables[1][1], force[selected])
            or not np.isclose(replay.get('maximum_gap_omega0', np.nan), np.max(np.diff(times[selected])) * wp,
                              rtol=2e-12, atol=0)):
        raise ValueError('table replay decimation, first/last knots or native force fingerprints disagree')
    coarse_every = round(s['table_dt_omega0'] / s['dt_omega0'])
    if not coarse_every > every > 1:
        raise ValueError('table comparison requires coarse, intermediate and fine spacings in that order')
    coarse_indices = np.unique(np.r_[0, np.arange(1, len(times) - 1, coarse_every), len(times) - 1])
    nested = (np.array_equal(tables[0][0], times[coarse_indices])
              and np.array_equal(tables[0][1], force[coarse_indices]))
    reference, scale = force[1:-1, 0], abs(force[0, 0])
    if scale == 0:
        raise ValueError('table force requires a nonzero initial amplitude for its original accuracy gate')
    rows = []
    for index, (table_t, amplitude) in enumerate(tables):
        accepted = np.interp(knots[1:-1], table_t, amplitude[:, 0])
        error = float(np.max(abs(accepted - reference)) / scale)
        if index != 1 and not np.isclose(
                error, control['results']['forcing'][sources[index][0]['settings']['label']][
                    'max_midpoint_force_error_over_initial'], rtol=2e-10, atol=2e-12):
            raise ValueError('original recorded table error differs from its actual accepted force')
        rows.append(dict(knots=len(table_t), maximum_gap_omega0=float(np.max(np.diff(table_t)) * wp),
                         nominal_spacing_omega0=s['dt_omega0'] * (coarse_every if index == 0 else every
                                                                  if index == 1 else 1),
                         actual_times_sha256=_pair_native_hash(table_t),
                         actual_amplitude_sha256=_pair_native_hash(amplitude),
                         accepted_force_sha256=_pair_native_hash(accepted),
                         max_midpoint_force_error_over_initial=error,
                         force_gate=error <= s['accuracy_targets']['table_force_over_initial']))
    return dict(tables=rows, coarse_exact_decimation_of_fine=nested,
                intermediate_exact_decimation_of_fine=True,
                source_control_settings=s, source_control_sha256=control_hashes, source_coupled_sha256=pump_hashes,
                note='Nesting is checked on actual archived arrays. No table-order or nonlinear convergence claim.')


def _pair_refinement_windows(a, b, targets):
    windows = _pair_repeat_windows(a, b, targets['conservation_over_transfer_and_target_difference'])
    for window in windows:
        selected = (a[1]['t'] >= window['bounds_omega0'][0] - 1e-8) & (a[1]['t'] <= window['bounds_omega0'][1] + 1e-8)
        for key in ('kinetic', 'density_rms'):
            values = [source[1][key] - (source[1][key][0] if key == 'kinetic' else 0) for source in (a, b)]
            window['observables'][key + ('_increment' if key == 'kinetic' else '')] = _pair_metric(
                values[0][selected], values[1][selected])
        quantities = ('mean_E', 'electric', 'work', 'kinetic_increment', 'local_spread_increment',
                      'mode_E', 'nonzero_electric', 'density_rms', 'local_density_rms')
        window['refinement_gates'] = {key: [value is not None and value <= targets[
            'nonzero_mode_density_window_fraction' if key in (
                'mode_E', 'nonzero_electric', 'density_rms', 'local_density_rms') else 'primary_window_fraction']
            for value in np.asarray(window['observables'][key]['relative_l2_difference'], dtype=object).ravel()]
            for key in quantities}
    return windows


def _pair_scalar_histories(sources):
    return [{key: dict(real=value.real.tolist(), imag=value.imag.tolist()) if np.iscomplexobj(value) else value.tolist()
             for key, value in source[1].items() if key in (
                 't', 'mean_E', 'mode_E', 'electric', 'nonzero_electric', 'kinetic', 'spread', 'local_spread',
                 'rms', 'momentum', 'local_density_rms', 'density_rms', 'magnetic', 'work')} for source in sources]


def pair_table_comparison(folders, transition=()):
    """Three archived forces at fixed PIC clocks, with original gates and source lineage."""
    if len(folders) != 3:
        raise ValueError('pair tables require original coarse, intermediate replay and original fine folders')
    sources = [_pair_prescribed_load(folder, example, (label,)) for folder, example, label in zip(
        folders, ('pair_waveform_branch', 'pair_waveform_table_replay', 'pair_waveform_branch'),
        ('realized_coarse', 'realized_fine', 'realized_fine'))]
    records = [source[0] for source in sources]
    control = json.loads((Path(folders[2]).parent / 'run.json').read_text())
    settings = [{**control['settings'], **record['settings']} for record in records]
    old, new = records[2]['git'], records[1]['git']
    if records[0]['git'] != old or (bool(transition) if old == new else tuple(transition) != (old, new)):
        raise ValueError('table sources require the same original producer and an explicit reviewed replay transition')
    for source, s in zip(sources, settings):
        if (any(source[0][key] != records[2][key] for key in PAIR_RUNTIME[1:])
                or any(s.get(key) != settings[2].get(key) for key in (
                    *PAIR_PHYSICS, 'cells', 'particles_per_cell_per_species', 'dt_omega0', 'horizon_omega0',
                    'velocity_seed_over_c', 'block_steps', 'table_dt_omega0', 'local_moments',
                    'saved_output_dt_omega0', 'forcing_native_dt_omega0'))
                or _pair_cadence(s) != _pair_cadence(settings[2])
                or not np.array_equal(source[1]['t'], sources[2][1]['t'])
                or source[3]['initial_leaves_without_table'] != sources[2][3]['initial_leaves_without_table']):
            raise ValueError('table controls require exact ordinary initialization, runtime '
                             'and fixed physical/PIC protocol')
    s, result = settings[1], records[1]['results']
    control_hash = hashlib.sha256((Path(folders[2]).parent / 'run.json').read_bytes()).hexdigest()
    if (hashlib.sha256((Path(folders[0]).parent / 'run.json').read_bytes()).hexdigest() != control_hash
            or s.get('donor_git') != old or s.get('donor_record_sha256') != sources[2][2]['run_sha256']
            or s.get('donor_control_record_sha256') != control_hash
            or s.get('initial_state_archive_sha256') != sources[2][2]['initial_state_sha256']
            or not isinstance(s.get('producer_script_sha256'), str) or len(s['producer_script_sha256']) != 64
            or not set(s['producer_script_sha256']) <= set('0123456789abcdef')
            or result.get('samples') != 1 or result.get('execution_index') != 1 or 'pic_dt_replay' in s):
        raise ValueError('table replay needs its exact fine donor, producer fingerprints '
                         'and single table-only intervention')
    forces = _pair_table_forces(folders, sources)
    comparisons = []
    targets = settings[2]['accuracy_targets']
    for first, second in ((0, 1), (1, 2), (0, 2)):
        windows = _pair_refinement_windows(sources[first], sources[second], targets)
        comparisons.append(dict(first=first, second=second, windows=windows))
    return dict(comparison='fixed-PIC archived prescribed-table resolution',
                reviewed_source_transition=list(transition) if old != new else None, forcing=forces,
                comparisons=comparisons, accuracy_targets=targets,
                native_runs=records, native_sha256=[source[2] for source in sources],
                archive_diagnostics=[source[3] for source in sources],
                original_gauss_bound_over_en_eps0=2e-13,
                original_gauss_gates=[{key: source[0]['results']['all_step_maxima'][key] <= 2e-13
                                       for key in PAIR_MAXIMA[4:6]} for source in sources],
                nominal_clock_bound_omega0=1e-9,
                nominal_clock_gates=[bool(abs(source[1]['t'][-1] - source[0]['settings']['horizon_omega0']) <= 1e-9)
                                     for source in sources],
                scalar_histories=_pair_scalar_histories(sources),
                notes='One execution per table; norms include execution variation. Raw shared native clocks, '
                      'unaligned complex modes and wrapped reference-amplitude-weighted phase. '
                      'Frames are correlated samples, not realizations. Global and two-endpoint ledger bounds '
                      'remain distinct; passing conservation or knot gates does not certify late convergence.')


def _pair_pic_initial(folders, sources, detail, times, force, extension=None):
    """Exact native leaves except half-step x/dt; independent physical restagger check."""
    leaves = [source[3]['initial_leaves_without_table'] for source in sources]
    if (any({key: value for key, value in row.items() if key not in ('x', 'dark.dt')}
            != {key: value for key, value in leaves[0].items() if key not in ('x', 'dark.dt')} for row in leaves[1:])
            or extension is None and (sources[0][3]['actual_force_sha256'] != sources[1][3]['actual_force_sha256']
                                      or sources[0][3]['actual_force_times_sha256'] !=
                                      sources[1][3]['actual_force_times_sha256'])
            or sources[1][3]['actual_force_sha256'] != fingerprint(force)
            or sources[1][3]['actual_force_times_sha256'] != fingerprint(times)):
        raise ValueError('PIC timestep replay requires identical force and all other complete initial leaves')
    with np.load(Path(folders[0]) / 'initial_state.npz', allow_pickle=False) as old, np.load(
            Path(folders[1]) / 'initial_state.npz', allow_pickle=False) as new:
        if set(old.files) != set(new.files):
            raise ValueError('PIC timestep replay native initial leaf sets must agree')
        length, x, u = float(old['dark.length']), old['x'], old['u']
        velocity = u[:, 0] / np.sqrt(1 + np.sum((u / 299792458.)**2, axis=1))
        canonical = (x[:, 0] - float(old['dark.dt']) * velocity / 2 + length / 2) % length - length / 2
        positions = new['x'][:, 0]
        recovered = (positions - float(new['dark.dt']) * velocity / 2 + length / 2) % length - length / 2
        expected = (canonical + float(new['dark.dt']) * velocity / 2 + length / 2) % length - length / 2
        errors = {key: float(np.max(abs((value + length / 2) % length - length / 2))) / length
                  for key, value in (('canonical_positions_over_L', recovered - canonical),
                                     ('restaggered_positions_over_L', positions - expected))}
        errors['transverse_positions_over_L'] = float(np.max(abs(new['x'][:, 1:] - x[:, 1:]))) / length
        count = len(old.files)
    if (max(errors.values()) > 2e-13 or detail.get('times_sha256') != _pair_native_hash(times)
            or detail.get('amplitude_sha256') != _pair_native_hash(force)
            or not isinstance(detail.get('canonical_integer_positions_sha256'), str)
            or len(detail['canonical_integer_positions_sha256']) != 64
            or not set(detail['canonical_integer_positions_sha256']) <= set('0123456789abcdef')
            or not 0 <= detail.get('canonical_position_error_over_L', np.inf) <= 2e-13):
        raise ValueError('PIC timestep canonical positions, declared restagger or exact force fingerprints disagree')
    result = dict(native_initial_leaf_count=count, unchanged_leaves_except=['x', 'dark.dt'],
                  all_other_native_initial_leaves_exact=True, fixed_force_byte_identical=True,
                  independent_numpy_position_errors=errors, position_bound_over_L=2e-13,
                  native_canonical_sha256=detail['canonical_integer_positions_sha256'],
                  note='Canonical hash is native provenance; NumPy restagger is a separate floating-point check.')
    if extension is not None:
        result.update(fixed_force_byte_identical=False, original_force_prefix_byte_identical=True,
                      unchanged_leaves_except=['x', 'dark.dt', 'dark.times', 'dark.amplitude'])
    return result


def _pair_pic_extension(folders, replay, extension, transition):
    times, force, proof = _pair_extension(folders[0], extension, transition)
    if (proof['native_run']['git'] != replay['git']
            or replay['settings'].get('force_extension') != proof['native_sha256']):
        raise ValueError('PIC force extension needs explicit native lineage and the reviewed replay source')
    s = replay['settings']
    dt = s['dt_omega0'] / s['normalization']['omega0_rad_s']
    ticks = _accepted_ticks(dt, round(s['horizon_omega0'] / s['dt_omega0']), 1)
    if ticks[-2] + dt / 2 > proof['old_terminal_s']:
        raise ValueError('scoped PIC force extension requires every accepted midpoint in the original prefix')
    proof.update(candidate_last_midpoint_s=float(ticks[-2] + dt / 2), candidate_endpoint_s=float(ticks[-1]),
                 original_endpoint_coverage_failure=bool(np.nextafter(proof['old_terminal_s'], np.inf) < ticks[-1]))
    return times, force, proof


def pair_pic_step_comparison(folders, transition=(), extension=None):
    """Fixed archived SI force, distinct rephased PIC timesteps and executions."""
    if len(folders) != 2:
        raise ValueError('pair PIC steps require one fine donor and one changed-timestep folder')
    sources = [_pair_prescribed_load(path, example) for path, example in zip(
        folders, ('pair_waveform_branch', 'pair_waveform_pic_dt_replay'))]
    a, b = sources
    old, new = a[0]['git'], b[0]['git']
    control, _, control_hashes, pump_hashes, times, force = _pair_fine_forcing(folders[0])
    extension_proof = None
    if extension is not None:
        times, force, extension_proof = _pair_pic_extension(folders, b[0], extension, transition)
    settings = [{**control['settings'], **source[0]['settings']} for source in sources]
    s, detail = settings[1], settings[1]['pic_dt_replay']
    if ((bool(transition) if old == new else tuple(transition) != (old, new))
            or any(a[0][key] != b[0][key] for key in PAIR_RUNTIME[1:])
            or any(value != b[0]['settings'].get(key) for key, value in a[0]['settings'].items()
                   if key not in ('dt_omega0', 'block_steps'))
            or any(settings[0].get(key) != s.get(key) for key in (
                *PAIR_PHYSICS, 'cells', 'particles_per_cell_per_species', 'horizon_omega0', 'velocity_seed_over_c',
                'table_dt_omega0', 'local_moments', 'saved_output_dt_omega0', 'forcing_native_dt_omega0'))
            or _pair_cadence(settings[0]) != _pair_cadence(s)
            or settings[0]['dt_omega0'] == s['dt_omega0']
            or settings[0]['block_steps'] * settings[0]['dt_omega0'] != s['block_steps'] * s['dt_omega0']
            or a[1]['t'].shape != b[1]['t'].shape or np.max(abs(a[1]['t'] - b[1]['t'])) > 1e-8):
        raise ValueError('PIC timestep controls require reviewed sources, equal runtime/physical protocol '
                         'and native cadence')
    extras = {'initial_state_source', 'donor_git', 'initial_state_archive_sha256', 'donor_record_sha256',
              'donor_control_record_sha256', 'producer_script_sha256', 'local_moments',
              'local_spread_lengths_c_over_omega0', 'pic_dt_replay'}
    if extension is not None:
        extras.add('force_extension')
    if (set(b[0]['settings']) - set(a[0]['settings']) - extras
            or s.get('donor_git') != old or s.get('donor_record_sha256') != a[2]['run_sha256']
            or s.get('donor_control_record_sha256') != control_hashes['run_sha256']
            or s.get('initial_state_archive_sha256') != a[2]['initial_state_sha256']
            or any(detail.get(key) != value for key, value in dict(
                donor_dt_omega0=settings[0]['dt_omega0'], pic_dt_omega0=s['dt_omega0'],
                block_horizon_omega0=s['block_steps'] * s['dt_omega0']).items())
            or b[0]['results'].get('samples') != 1 or b[0]['results'].get('execution_index') != 1
            or not isinstance(s.get('producer_script_sha256'), str) or len(s['producer_script_sha256']) != 64
            or not set(s['producer_script_sha256']) <= set('0123456789abcdef')):
        raise ValueError('PIC timestep replay donor hashes, lineage, metadata or single-intervention protocol disagree')
    initialization = _pair_pic_initial(folders, sources, detail, times, force, extension_proof)
    windows = _pair_refinement_windows(a, b, s['accuracy_targets'])
    for window in windows:
        low, high = window['bounds_omega0']
        masks = [(source[1]['t'] >= low - 1e-8) & (source[1]['t'] <= high + 1e-8) for source in sources]
        if not np.array_equal(*masks):
            raise ValueError('PIC timestep windows need corresponding native samples, without interpolation')
        window['actual_clocks_per_run_omega0'] = [
            source[1]['t'][mask][[0, -1]].tolist() for source, mask in zip(sources, masks)]
    result = dict(comparison='fixed archived-force PIC timestep and execution sensitivity',
                  reviewed_source_transition=list(transition) if old != new else None,
                  initialization=initialization, windows=windows, accuracy_targets=s['accuracy_targets'],
                  native_runs=[source[0] for source in sources], native_sha256=[source[2] for source in sources],
                  archive_diagnostics=[source[3] for source in sources],
                  scalar_histories=_pair_scalar_histories(sources),
                  source_control_sha256=control_hashes, source_coupled_sha256=pump_hashes,
                  conservation_budget=s['accuracy_targets']['conservation_over_transfer_and_target_difference'],
                  rate_accounting_budget=1 / 3,
                  original_gauss_bound_over_en_eps0=2e-13,
                  original_gauss_gates=[{key: source[0]['results']['all_step_maxima'][key] <= 2e-13
                                         for key in PAIR_MAXIMA[4:6]} for source in sources],
                  nominal_clock_bound_omega0=1e-9,
                  nominal_clock_gates=[bool(abs(source[1]['t'][-1] - source[0]['settings']['horizon_omega0']) <= 1e-9)
                                       for source in sources],
                  notes='Identical SI piecewise-linear force, different PIC midpoint sampling and position staggering. '
                        'Separate accepted native clocks at equal scalar cadence; no interpolation/alignment. '
                        'Sources are reviewed algorithm-matched only; executable identity and isolated truncation '
                        'cause are unobserved. Raw reductions and trajectory norms remain distinct. '
                        'Frames are not realizations; '
                        'global/two-endpoint conservation gates remain unchanged and do not certify convergence.')
    if extension_proof is not None:
        result['force_extension'] = extension_proof
        result['notes'] = ('Whole table hashes and compile shapes changed by an explicit accepted continuation. '
                           'Every requested PIC midpoint lies in its byte-identical original force prefix. '
                           + result['notes'])
    return result


def _pair_late(source, tolerance=1e-8):
    """Retain raw native window means, including the producer's dense-background QD."""
    record, _, _, branches = source
    setting, reservoir = record['settings'], record['settings']['initial_dark_energy_over_energy_scale']
    bounds = [max(0, setting['horizon_omega0'] - 20), setting['horizon_omega0']]
    supplied = next((window for window in record['results']['windows'].values()
                     if np.allclose(window['bounds_omega0'], bounds, rtol=0, atol=tolerance)), None)
    if supplied is None:
        raise ValueError('pair controls require the original raw final-20 window reduction')
    reductions, conservation = {}, {}
    for label, (native, data, _, _) in branches.items():
        selected = (data['t'] >= bounds[0] - tolerance) & (data['t'] <= bounds[1] + tolerance)
        reductions[label] = window_summary(data, selected, label == 'coupled')
        reductions[label]['samples'] = int(selected.sum())
        for key, value in reductions[label].items():
            if key != 'samples' and not np.allclose(value, supplied['reductions'][label][key], rtol=2e-12, atol=1e-15):
                raise ValueError('pair supplied window means differ from native SI scalar reductions')
        maxima = native['results']['all_step_maxima']
        if len(maxima) != 9 or any(not np.isfinite(value) or value < 0 for value in maxima.values()):
            raise ValueError('pair conservation requires all nine finite native all-step maxima')
        peak_work = float(np.max(abs(data['source_work'])))
        defect = max(maxima[key] for key in ('energy_work_over_energy_scale', 'dark_sector_work_over_energy_scale',
                                             'ordinary_sector_work_over_energy_scale'))
        ratio = defect / peak_work if peak_work else None
        conservation[label] = dict(native_all_step_maxima=maxima, maximum_energy_sector_defect_over_peak_transfer=ratio,
                                   peak_transfer_note='Stored scalar peak work is a lower bound on the all-step peak.')
    coupled = branches['coupled'][1]
    selected = (coupled['t'] >= bounds[0] - tolerance) & (coupled['t'] <= bounds[1] + tolerance)
    if supplied['samples'] != int(selected.sum()):
        raise ValueError('pair raw QD means must retain their actual native sample count')
    reductions['coupled'].update({key: supplied['reductions']['coupled'][key] for key in (
        'additional_dark_depletion_fraction_mean', 'additional_dark_depletion_gain_mean')})
    reductions['coupled'].update(
        total_dark_fraction_mean=float(np.mean(coupled['dark'][selected]) / reservoir),
        coherent_dark_fraction_mean=float(np.mean(coupled['dark_coherent'][selected]) / reservoir),
        nonzero_dark_fraction_mean=float(np.mean(
            coupled['dark'][selected] - coupled['dark_coherent'][selected]) / reservoir))
    effect = abs(reductions['coupled']['additional_dark_depletion_gain_mean']) * reservoir
    maximum = branches['coupled'][0]['results']['all_step_maxima']
    defect = max(maximum[key] for key in ('energy_work_over_energy_scale', 'dark_sector_work_over_energy_scale',
                                          'ordinary_sector_work_over_energy_scale'))
    conservation['coupled']['maximum_energy_sector_defect_over_depletion_gain'] = defect / effect if effect else None
    budget = setting['accuracy_targets']['conservation_over_transfer_and_target_difference']
    gates = {label: {key: value <= budget if value is not None else None
                     for key, value in row.items() if key.startswith('maximum_')}
             for label, row in conservation.items()}
    targets, reference = setting['accuracy_targets'], record['results']['reference']
    reference_gates = dict(
        energy=(reference['max_energy_defect_over_homogeneous_initial_reservoir']
                <= targets['reference_energy_over_reservoir']),
        quadrature=reference['quadrature_force_error_over_initial'] <= targets['quadrature_force_over_initial'],
        tolerance=reference['tolerance_force_error_over_initial'] <= targets['quadrature_force_over_initial'])
    table_gates = {label: row['max_midpoint_force_error_over_initial'] <= targets['table_force_over_initial']
                   for label, row in record['results']['forcing'].items()}
    return dict(bounds_omega0=bounds, scalar_dt_omega0=_pair_cadence(setting),
                QD_mean_note='Original producer means on stored scalar clocks; no reconstructed or interpolated QD.',
                native_reductions=reductions, conservation=conservation,
                conservation_gates=gates, reference_gates=reference_gates, table_gates=table_gates)


def _pair_comparison(first, second, variant, tolerance=1e-8):
    a, b = [source[0]['settings'] for source in (first, second)]
    count = [s['cells'] * s['particles_per_cell_per_species'] for s in (a, b)]
    hashes = [s['initial_ordinary_fingerprints'] for s in (a, b)]
    matches = _hash_matches(hashes, hashes[0])
    position_difference = None
    if count[0] == count[1]:
        x, y = [source[3]['coupled'][3] for source in (first, second)]
        difference = (y - x + .5) % 1 - .5
        position_difference = float(np.max(abs(difference)))
        if not matches['w'] or position_difference > 2e-13 or variant != 'seed' and not matches['u']:
            raise ValueError('fixed-count pair controls must preserve physical positions, weights and thermal momentum')
    if variant in ('repeat', 'sampling') and not all(matches.values()):
        raise ValueError('pair repeat/sampling controls require identical complete initial ordinary fingerprints')
    comparisons = {}
    for label in PAIR_CASES:
        rows = [source[3][label][1] for source in (first, second)]
        cadence = max(_pair_cadence(a), _pair_cadence(b))
        strides = [round(cadence / _pair_cadence(setting)) for setting in (a, b)]
        indices = [np.arange(0, len(row['t']), stride) for row, stride in zip(rows, strides)]
        times = [row['t'][index] for row, index in zip(rows, indices)]
        if (any(abs(stride * _pair_cadence(setting) - cadence) > tolerance
                for stride, setting in zip(strides, (a, b))) or times[0].shape != times[1].shape
                or not np.allclose(*times, rtol=0, atol=tolerance)):
            raise ValueError('pair comparisons require exact shared native clocks at integer cadence ratios')
        selected = times[0] >= max(0, a['horizon_omega0'] - 20) - tolerance
        quantities = ('mean_E', 'electric', 'source_work', 'kinetic', 'local_spread', 'mode_E',
                      'nonzero_electric', 'density_rms', 'local_density_rms')
        norms = {}
        for key in quantities:
            values = [row[key][index] for row, index in zip(rows, indices)]
            if key in ('kinetic', 'local_spread'):
                values = [value - row[key][0] for value, row in zip(values, rows)]
            norms[key] = _pair_metric(values[0][selected], values[1][selected])
        target = a['accuracy_targets']
        gates = {key: np.asarray(row['relative_l2_difference'], dtype=object).tolist() for key, row in norms.items()}
        modes = ('mode_E', 'nonzero_electric', 'density_rms', 'local_density_rms')
        gates = {key: [None if value is None else value <= target[
                    'nonzero_mode_density_window_fraction' if key in modes else 'primary_window_fraction']
                    for value in np.asarray(row, dtype=object).ravel()]
                 for key, row in gates.items()}
        comparisons[label] = dict(late_shared_native_samples=int(selected.sum()), shared_scalar_dt_omega0=cadence,
                                  observables=norms,
                                  refinement_gates=gates if variant not in ('seed', 'sampling') else None)
    return dict(variant=variant, initial_hash_matches=matches,
                reconstructed_position_max_difference_over_length=position_difference,
                position_note='Native half-step positions are reversed at t=0; this is a roundoff-bounded comparison.',
                branches=comparisons)


def _pair_controls(sources):
    if not sources:
        raise ValueError('pair evidence requires at least one complete five-branch control')
    baseline = sources[0][0]
    if baseline['settings']['velocity_seed_over_c'] != 2e-4:
        raise ValueError('pair comparisons require the declared 2e-4 seed baseline first')
    settings = [source[0]['settings'] for source in sources]
    for source, setting in zip(sources, settings):
        for key in PAIR_RUNTIME:
            if source[0][key] != baseline[key]:
                raise ValueError('pair controls must share native runtime and clean source')
        for key in (*PAIR_PHYSICS, 'horizon_omega0', 'saved_output_dt_omega0', 'reference_method',
                    'reference_tolerances', 'linear_reference_horizon_omega0'):
            if setting[key] != settings[0][key]:
                raise ValueError(f'pair controls must share fixed physical/protocol setting {key}')
        block = source[3]['coupled'][0]['settings']['block_steps'] * setting['dt_omega0']
        first_block = sources[0][3]['coupled'][0]['settings']['block_steps'] * settings[0]['dt_omega0']
        if not np.isclose(block, first_block, rtol=0, atol=1e-8):
            raise ValueError('pair controls must share physical execution block duration')
    return settings


def pair_control_comparison(sources):
    """Separate seed, clock, fixed-count mesh and fixed-mesh loading comparisons."""
    _pair_controls(sources)
    comparisons, seed_scaling = [], []
    late = [_pair_late(source) for source in sources]
    for index, source in enumerate(sources[1:], 1):
        for previous, reference in enumerate(sources[:index]):
            a, b = [row[0]['settings'] for row in (reference, source)]
            parameters = dict(seed=(a['velocity_seed_over_c'], b['velocity_seed_over_c']),
                              dt=(a['dt_omega0'], b['dt_omega0']), mesh=(a['cells'], b['cells']),
                              loading=(a['cells'] * a['particles_per_cell_per_species'],
                                       b['cells'] * b['particles_per_cell_per_species']))
            changed = [key for key, values in parameters.items() if values[0] != values[1]]
            cadence_changed = _pair_cadence(a) != _pair_cadence(b)
            sampling_only = not changed and cadence_changed
            if len(changed) <= 1:
                variant = changed[0] if changed else 'sampling' if sampling_only else 'repeat'
                comparison = _pair_comparison(reference, source, variant)
                quantities = ('additional_dark_depletion_gain_mean', 'total_dark_fraction_mean',
                              'coherent_dark_fraction_mean', 'nonzero_dark_fraction_mean')
                means = [late[i]['native_reductions']['coupled'] for i in (previous, index)]
                comparison['raw_window_mean_differences'] = {
                    key: _pair_metric(np.atleast_1d(means[0][key]), np.atleast_1d(means[1][key])) for key in quantities}
                comparison['raw_window_mean_gates'] = {
                    key: (row['relative_l2_difference'] <= a['accuracy_targets'][
                              'nonzero_mode_density_window_fraction' if key == 'nonzero_dark_fraction_mean'
                              else 'primary_window_fraction']
                          if row['relative_l2_difference'] is not None and variant != 'seed' and not cadence_changed
                          else None)
                    for key, row in comparison['raw_window_mean_differences'].items()}
                comparison['raw_window_scalar_dt_omega0'] = [_pair_cadence(a), _pair_cadence(b)]
                comparison['raw_window_note'] = ('Different scalar cadences: sampling reduction differences; '
                                                 'no raw-mean resolution gate.' if cadence_changed else
                                                 'Original means retain their matched native scalar cadence.')
                comparisons.append(dict(first=previous, second=index, **comparison))
            if changed == ['seed'] and previous == 0:
                arrays = [row[1] for row in (reference, source)]
                selected = arrays[0]['coupled_t'] <= min(20, a['horizon_omega0']) + 1e-8
                multiplier = (a['velocity_seed_over_c'] / b['velocity_seed_over_c']
                              if b['velocity_seed_over_c'] else None)
                quantities = {label: _pair_metric(arrays[0][f'{label}_mode_E'][selected],
                                                  arrays[1][f'{label}_mode_E'][selected] * (multiplier or 1))
                              for label in ('coupled', 'homogeneous_fine')}
                seed_scaling.append(dict(first=0, second=index, amplitude_multiplier=multiplier,
                                         interpretation='seed scaling' if multiplier else 'unseeded numerical floor',
                                         early_complex_modes=quantities))
    early = [{label: _pair_metric(source[1][f'linear_{label}_mode_E'],
                                  source[1][f'{label}_mode_E'][:len(source[1]['linear_t'])])
              for label in ('coupled', 'homogeneous_fine')} for source in sources]
    return dict(controlled_pairs=comparisons, early_seed_controls=seed_scaling, late=late,
                saved_early_linear_errors=early,
                claim='Native waveform evidence; no late-convergence, uncertainty or nonlinear-cause claim')


def publish_pair_controls(folders, folder):
    """Render the original pair scalar evidence; no particle arrays are published."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from darkjaxincell import midnight
    from jaxincell import save_run

    sources = [_pair_load(path) for path in folders]
    result = pair_control_comparison(sources)
    arrays = {f'{index}_{key}': value for index, source in enumerate(sources) for key, value in source[1].items()
              if '_table_' not in key}
    hashes = [dict(top=source[2], branches={label: branch[2] for label, branch in source[3].items()})
              for source in sources]
    result.update(native_runs=[source[0] for source in sources],
                  native_branches=[{label: branch[0] for label, branch in source[3].items()} for source in sources],
                  native_source_sha256=hashes)
    with midnight():
        plt.rcParams['axes.prop_cycle'] = plt.cycler(
            color=plt.rcParams['axes.prop_cycle'].by_key()['color'] + ['#626262'])
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout='constrained')
        baseline = sources[0][0]['settings']
        count0 = baseline['cells'] * baseline['particles_per_cell_per_species']
        fig.suptitle(f"Baseline: {baseline['cells']} cells, {count0 / 1000:g}k/species, "
                     f"Δτ={baseline['dt_omega0']:g}, δv/c={baseline['velocity_seed_over_c']:g}", fontsize=11)
        for source in sources:
            setting, data = source[0]['settings'], source[1]
            time, reservoir = data['coupled_t'], setting['initial_dark_energy_over_energy_scale']
            count = setting['cells'] * setting['particles_per_cell_per_species']
            inputs = dict(zip(('δv/c', 'Δτ', 'cells', 'N/N₀'),
                              (setting['velocity_seed_over_c'], setting['dt_omega0'], setting['cells'],
                               count / count0)))
            defaults = (baseline['velocity_seed_over_c'], baseline['dt_omega0'], baseline['cells'],
                        1.)
            label = ', '.join(f'{key}={value:g}' for (key, value), default in zip(inputs.items(), defaults)
                              if value != default)
            if (_pair_cadence(setting) != _pair_cadence(baseline)
                    and (_pair_cadence(setting) != setting['dt_omega0']
                         or _pair_cadence(baseline) != baseline['dt_omega0'])):
                label += (', ' if label else '') + f'sample Δτ={_pair_cadence(setting):g}'
            label = label or 'baseline'
            axes[0, 0].plot(time[1:], abs(data['coupled_mode_E'][1:]), label=label)
            axes[0, 1].plot(time, data['additional_dark_depletion_gain'])
            gain = (data['coupled_local_spread'][:, :, 0] - data['coupled_local_spread'][0, :, 0]).sum(axis=1)
            axes[1, 0].plot(time, gain / reservoir)
            axes[1, 1].plot(time, (data['coupled_balance'] - data['coupled_balance'][0]) / reservoir)
        reference = sources[0][1]
        axes[0, 0].plot(reference['linear_t'][1:], abs(reference['linear_coupled_mode_E'][1:]),
                        'k--', lw=1, label='independent linear reference, δv/c=2e−4')
        axes[0, 0].set(ylabel='|ordinary seeded E mode| / field scale', yscale='log',
                       title='Raw complex-mode amplitude')
        axes[0, 1].set(ylabel='QD(t) − QD(0)', title='Additional dark depletion gain')
        length = sources[0][0]['settings']['local_spread_lengths_c_over_omega0'][0]
        axes[1, 0].set(ylabel='local variance-energy gain / initial dark energy',
                       title=f'Gaussian length ℓω₀/c={length:g}')
        axes[1, 1].set(ylabel='closed energy change / initial dark energy', title='Full energy balance')
        axes[0, 0].legend(fontsize=7)
        for axis in axes.flat:
            axis.set(xlabel=r'$\omega_0t$')
            axis.grid(alpha=.25)
        save_run(folder, 'pair_control_comparison',
                 dict(parent_revision=sources[0][0]['settings']['parent_revision'],
                      producer='examples/dark_reservoir.py: study=pair_waveform',
                      comparison='docs/scripts/compare_replays.py: pair_controls',
                      accuracy_targets=sources[0][0]['settings']['accuracy_targets'],
                      figure_note='Raw saved samples, no smoothing/interpolation/phase alignment. Log plot omits t=0. '
                                  'Windows use stored scalar samples; QD means retain producer cadence and reductions. '
                                  'Local lab-frame variance energy is not thermodynamic temperature.'),
                 result, fig, **arrays)
        np.savez_compressed(Path(folder) / 'data.npz', **arrays)
        plt.close(fig)
    return result


if __name__ == '__main__':  # noqa: C901 — sequential evidence rendering
    print('Comparing native records', flush=True)
    if sum((implicit, picard, method_controls, bool(phase_controls), bool(pair_controls), bool(pair_repeats),
            bool(pair_tables), bool(pair_pic_steps),
            bool(ensemble), continuation)) > 1:
        raise ValueError('select one comparison mode')
    if (pair_repeat_observer or pair_repeat_donor or pair_repeat_transition) and not pair_repeats:
        raise ValueError('pair repeat observer/donor inputs require pair_repeats')
    if pair_table_transition and not pair_tables:
        raise ValueError('pair table source transition requires pair_tables')
    if pair_pic_transition and not pair_pic_steps:
        raise ValueError('pair PIC source transition requires pair_pic_steps')
    if pair_pic_extension is not None and not pair_pic_steps:
        raise ValueError('pair PIC extension requires explicit pair_pic_steps')
    if (refined or finer_step or loading_refined or loading_repeat) and destination is None:
        raise ValueError('refinements require a publish folder')
    if finer_step and (implicit or picard or method_controls or phase_controls or pair_controls or pair_repeats
                       or pair_tables or pair_pic_steps
                       or ensemble or continuation):
        raise ValueError('finer_step extends only the native explicit replay publisher')
    if loading_repeat and loading_refined is None:
        raise ValueError('a loading repeat requires its first record')
    if finer_mesh and not method_controls or orbit_audits and not (picard or phase_controls):
        raise ValueError('finer_mesh requires method_controls; orbit_audits requires picard or phase_controls')
    if pair_repeats or pair_tables or pair_pic_steps:
        if destination is not None or variant != 'repeat' or constraints or legacy:
            raise ValueError('prescribed pair comparisons write scalar JSON only')
        result = (pair_repeat_comparison(pair_repeats, pair_repeat_observer, pair_repeat_donor, pair_repeat_transition)
                  if pair_repeats else pair_table_comparison(pair_tables, pair_table_transition) if pair_tables else
                  pair_pic_step_comparison(pair_pic_steps, pair_pic_transition, extension=pair_pic_extension))
        source, root = Path(__file__).resolve(), Path(__file__).resolve().parents[2]
        result['validation_source'] = dict(
            git=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
            script=source.relative_to(root).as_posix(), script_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            numpy=np.__version__, tracked_source_dirty=bool(subprocess.check_output(
                ['git', 'status', '--porcelain', '--untracked-files=no'], cwd=root, text=True).strip()))
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, allow_nan=False))
    elif continuation:
        if variant != 'dt' or legacy or constraints or refined or loading_refined or loading_repeat:
            raise ValueError('continuation opts into prescribed timestep lineage comparison only')
        result = compare_continuations(first, second, continuation_donors, continuation_transitions)
        if destination is not None:
            publish(first, second, destination, result)
        else:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(result, indent=2, allow_nan=False))
    elif ensemble:
        if destination is not None:
            raise ValueError('ensemble reduction writes scalar JSON only')
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(ensemble_comparison(ensemble, ensemble_window), indent=2, allow_nan=False))
    elif pair_controls:
        if destination is None:
            raise ValueError('pair controls require a publish folder')
        publish_pair_controls(pair_controls, destination)
    elif phase_controls:
        if destination is None:
            raise ValueError('phase controls require a publish folder')
        publish_phases(phase_controls, destination, orbit_audits)
    elif method_controls:
        if destination is None or refined is None:
            raise ValueError('method_controls requires substep, mesh and publish folders')
        publish_method_controls(first, second, refined, destination, finer_mesh)
    elif picard:
        if destination is None:
            raise ValueError('iteration figures require a publish folder')
        publish_iterations(first, second, destination, orbit_audits)
    elif implicit:
        if constraints or legacy or variant != 'repeat' or loading_refined or loading_repeat:
            raise ValueError('implicit uses the matched physical-loading and native-clock contract')
        if destination is not None:
            publish_implicit(first, second, destination, refined)
        else:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(implicit_comparison(first, second)[0], indent=2))
    else:
        result = compare_replays(first, second, variant, legacy=legacy, constraints=constraints)
        if destination is not None:
            publish(first, second, destination, result, refined, refined_variant, loading_refined, loading_repeat,
                    refined_against=refined_against, finer_step=finer_step)
        else:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(result, indent=2))
    print(f"Saved comparison to {destination or output}", flush=True)
