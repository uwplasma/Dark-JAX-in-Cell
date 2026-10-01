"""Compare native paper replays at equal horizons without shifting or interpolating."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

WINDOWS = ((0, 100), (0, 250), (0, 500), (0, 1000), (800, 1000))
SERIES = ('mean_E', 'rms', 'electric', 'nonzero_electric', 'density_rms')
LOCAL_SERIES = ('local_spread', 'local_density_rms')
PARAMETERS = ('cells', 'particles_per_species', 'seed', 'dt_omega_p', 'output_dt_omega_p',
              'length_c_over_omega_p', 'mass_ratio', 'T_each_over_mec2', 'coupling',
              'drive_quiver_over_sigma', 'force_quiver_over_c', 'loading', 'pusher', 'shape', 'parent_revision')
VARIANTS = dict(repeat=None, dt='dt_omega_p', seed='seed', mesh='cells', loading='particles_per_species')


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
    if not np.allclose(mass * data['rms'][0]**2, settings['T_each_over_mec2'], rtol=2e-12):
        raise ValueError('initial RMS differs from the physical temperature loading')
    if not np.allclose(data['nonzero_electric'], data['electric'] - data['mean_E']**2 / 2,
                       rtol=2e-12, atol=1e-30):
        raise ValueError('electric components do not share the stated field normalization')
    if all(key in data for key in ('balance', 'magnetic', 'dark', 'work')):
        expected = data['electric'] + data['magnetic'] + data['dark'] + data['kinetic'].sum(axis=1) - data['work']
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


def _controls(records, data, variant, tolerance):
    """Equal horizons/cadences preserve sampling and compile-allocation controls."""
    settings = [record['settings'] for record in records]
    for key in PARAMETERS:
        if key != VARIANTS[variant] and settings[0][key] != settings[1][key]:
            raise ValueError(f'controlled comparisons must share {key}')
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


def _replay_label(settings, variant, index):
    """Name the measured resolution and the controlled seed or execution."""
    if variant in ('repeat', 'dt'):
        label = f"Δtωₚ={settings['dt_omega_p']:g}"
        return label + f', execution {index + 1}' if variant == 'repeat' and index < 2 else label
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


def publish(first, second, folder, comparison, refined=None, refined_variant='dt', loading_refined=None,
            loading_repeat=None):
    """Render saved scalars through the parent figure/provenance path; no dynamics."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from darkjaxincell import midnight
    from jaxincell import save_run

    sources = [_load(path, 1e-5) for path in (first, second)]
    refinement = None
    if refined is not None:
        refinement = compare_replays(first, refined, variant=refined_variant,
                                     constraints='endpoint_constraints' in comparison)
        sources.append(_load(refined, 1e-5))
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
        axes[0, 0].set(ylabel=r'$U_{E,k\ne0}/(nT_{e0}L/2)$', title='Nonzero-mode electric energy',
                       yscale='log', ylim=(1e-4, None))
        axes[0, 1].set(ylabel=r'$U_{\rm local}(t)/U_{\rm local}(0)$', title=r'Local spread at $2\lambda_{D0}$',
                       yscale='log')
        axes[1, 0].set(ylabel='smoothed density RMS / mean', title=r'Density at $2\lambda_{D0}$')
        axes[1, 1].set(ylabel=r'$(\Delta U-W)/\max|W|$', title='Energy and external work')
        for axis in axes.flat:
            axis.set(xlabel=r'$\omega_pt$', xlim=(0, time[-1]))
            axis.grid(alpha=.25)
            axis.legend(fontsize=8)
        names = ['first', 'second'] + (['refined'] if refined is not None else [])
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
            result['refinement_comparison'] = refinement
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
               'cells': {'cells', 'initial_fingerprints', 'initial_state_source'}}[variant]
    for key in settings[0].keys() | settings[1].keys():
        if key not in changes and settings[0].get(key) != settings[1].get(key):
            raise ValueError(f'implicit controls must share {key}, including required initial hashes')
    if settings[0][variant] == settings[1][variant]:
        raise ValueError(f'implicit controls must vary {variant}')
    if variant == 'cells':
        hashes = [row['initial_fingerprints'] for row in settings]
        if not all(_hash_matches(hashes, ('x', 'u', 'w', 'time', 'mass', 'charge')).values()):
            raise ValueError('mesh controls require identical physical particle arrays')
    for key in ('git', 'jax', 'jaxincell', 'numpy', 'backend', 'jax_enable_x64'):
        if records[0][key] != records[1][key]:
            raise ValueError(f'implicit controls must share runtime/source {key}')
    if not np.array_equal(data[0]['t'], data[1]['t']) or data[0]['t'][0] != 0:
        raise ValueError('implicit controls require identical native clocks')
    traces = [dict(row, momentum=row['momentum'] - row['momentum'][0]) for row in data]
    return records, data, {key: metrics(traces[0][key], traces[1][key]) for key in
                           ('electric', 'current', 'nonzero_electric', 'momentum', 'mean', 'rms', 'kinetic')}


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


def _publish_controls(args, parser):
    if args.method_controls:
        if (not args.publish or not args.refined or args.picard or args.implicit or args.orbit_audits
                or args.loading_refined or args.loading_repeat or args.constraints or args.legacy
                or args.variant != 'repeat'):
            parser.error('--method-controls requires a substep record, --refined mesh record and --publish')
        publish_method_controls(args.first, args.second, args.refined, args.publish, args.finer_mesh)
        return True
    if args.picard:
        if (not args.publish or args.implicit or args.refined or args.loading_refined or args.loading_repeat
                or args.constraints or args.legacy or args.finer_mesh or args.variant != 'repeat'):
            parser.error('--picard requires two iteration records and --publish')
        publish_iterations(args.first, args.second, args.publish, args.orbit_audits)
        return True
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('first', type=Path)
    parser.add_argument('second', type=Path)
    parser.add_argument('--implicit', action='store_true', help='second record is a Gaussian implicit drive')
    parser.add_argument('--picard', action='store_true', help='compare two exact-input implicit iteration counts')
    parser.add_argument('--method-controls', action='store_true', help='second varies substeps, --refined varies mesh')
    parser.add_argument('--orbit-audits', type=Path, nargs='+', default=(), help='retain independent orbit records')
    parser.add_argument('--variant', choices=VARIANTS, default='repeat')
    parser.add_argument('--legacy', action='store_true', help='retain explicit missing-fingerprint status')
    parser.add_argument('--constraints', action='store_true', help='audit endpoint closure without applying it')
    parser.add_argument('--refined', type=Path, help='add a controlled refinement to a published figure')
    parser.add_argument('--finer-mesh', type=Path, help='second mesh refinement for --method-controls')
    parser.add_argument('--refined-variant', choices=VARIANTS, default='dt')
    parser.add_argument('--loading-refined', type=Path, help='change particle count relative to the second record')
    parser.add_argument('--loading-repeat', type=Path, help='retain a scalar repeat of --loading-refined')
    destination = parser.add_mutually_exclusive_group(required=True)
    destination.add_argument('--output', type=Path, help='write comparison JSON only')
    destination.add_argument('--publish', type=Path, help='render a compact figure, arrays and native run records')
    args = parser.parse_args()
    if _publish_controls(args, parser):
        return
    if args.finer_mesh:
        parser.error('--finer-mesh requires --method-controls')
    if args.orbit_audits:
        parser.error('--orbit-audits requires --picard')
    if (args.refined or args.loading_refined or args.loading_repeat) and not args.publish:
        parser.error('refinements require --publish')
    if args.loading_repeat and not args.loading_refined:
        parser.error('--loading-repeat requires --loading-refined')
    if args.implicit:
        if args.constraints or args.legacy or args.variant != 'repeat' or args.loading_refined or args.loading_repeat:
            parser.error('--implicit uses its own physical-loading and native-clock contract')
        if args.publish:
            publish_implicit(args.first, args.second, args.publish, args.refined)
        else:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(implicit_comparison(args.first, args.second)[0], indent=2))
        return
    result = compare_replays(args.first, args.second, args.variant, legacy=args.legacy, constraints=args.constraints)
    if args.publish:
        publish(args.first, args.second, args.publish, result, args.refined, args.refined_variant,
                args.loading_refined, args.loading_repeat)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
