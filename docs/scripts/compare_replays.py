"""Compare native paper replays at equal horizons without shifting or interpolating."""
import hashlib
import json
from pathlib import Path

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
constraints = globals().get('constraints', False)
legacy = globals().get('legacy', False)
refined = globals().get('refined', None)
finer_mesh = globals().get('finer_mesh', None)
refined_variant = globals().get('refined_variant', 'dt')
loading_refined = globals().get('loading_refined', None)
loading_repeat = globals().get('loading_repeat', None)
orbit_audits = globals().get('orbit_audits', ())
destination = globals().get('destination', None)  # folder for figure, arrays and record
output = Path(globals().get('output', 'artifacts/replay_comparison.json'))

WINDOWS = ((0, 100), (0, 250), (0, 500), (0, 1000), (800, 1000))
SERIES = ('mean_E', 'rms', 'electric', 'nonzero_electric', 'density_rms')
LOCAL_SERIES = ('local_spread', 'local_density_rms')
PARAMETERS = ('cells', 'particles_per_species', 'seed', 'dt_omega_p', 'output_dt_omega_p',
              'length_c_over_omega_p', 'mass_ratio', 'T_each_over_mec2', 'coupling',
              'drive_quiver_over_sigma', 'force_quiver_over_c', 'loading', 'pusher', 'shape', 'parent_revision')
VARIANTS = dict(repeat=None, dt='dt_omega_p', seed='seed', mesh='cells', loading='particles_per_species', shape='shape')
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


def _replay_label(settings, variant, index):
    """Name the measured resolution and the controlled seed or execution."""
    if variant in ('repeat', 'dt'):
        label = f"Δtωₚ={settings['dt_omega_p']:g}"
        return label + f', execution {index + 1}' if variant == 'repeat' and index < 2 else label
    if variant == 'shape':
        return f"degree {settings.get('shape_order', 2)} weighting"
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


def _pair_initial(folder, setting, hashes, data):
    """Read initial arrays only for validation; publish hashes and small diagnostics."""
    count = setting['cells'] * setting['particles_per_cell_per_species']
    scale = setting['normalization']
    c = 299792458.  # Exact SI speed, independent of a CODATA release.
    if set(hashes) != {'x', 'u', 'w', 'E', 'B', 'rho'}:
        raise ValueError('pair initial fingerprints require all six ordinary arrays')
    with np.load(Path(folder) / 'initial_state.npz', allow_pickle=False) as state:
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
    result = {key: value.copy() for key, value in data.items()}
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
    time, dt = normalized['t'], setting['dt_omega0']
    steps = round(setting['horizon_omega0'] / dt)
    if (time.shape != (steps + 1,) or time[0] != 0 or np.any(np.diff(time) <= 0)
            or not np.allclose(np.diff(time), dt, rtol=0, atol=tolerance)
            or abs(time[-1] - setting['horizon_omega0']) > tolerance
            or branch['block_steps'] * dt <= 0 or steps % branch['block_steps']):
        raise ValueError('pair native clocks/horizon/compiled blocks are incomplete')
    stride = round(setting['saved_output_dt_omega0'] / dt)
    if stride < 1 or abs(stride * dt - setting['saved_output_dt_omega0']) > tolerance:
        raise ValueError('pair saved cadence must select integer native clocks')
    indices = np.unique(np.r_[np.arange(0, steps + 1, stride), steps])
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
        conservation[label] = dict(native_all_step_maxima=maxima, maximum_energy_sector_defect_over_peak_transfer=ratio)
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
    return dict(bounds_omega0=bounds, native_reductions=reductions, conservation=conservation,
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
    if variant == 'repeat' and not all(matches.values()):
        raise ValueError('pair repeats require identical complete initial ordinary fingerprints')
    comparisons = {}
    for label in PAIR_CASES:
        rows = [source[3][label][1] for source in (first, second)]
        cadence = max(a['dt_omega0'], b['dt_omega0'])
        strides = [round(cadence / setting['dt_omega0']) for setting in (a, b)]
        indices = [np.arange(0, len(row['t']), stride) for row, stride in zip(rows, strides)]
        times = [row['t'][index] for row, index in zip(rows, indices)]
        if (any(abs(stride * setting['dt_omega0'] - cadence) > tolerance
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
        comparisons[label] = dict(late_shared_native_samples=int(selected.sum()), observables=norms,
                                  refinement_gates=gates if variant != 'seed' else None)
    return dict(variant=variant, initial_hash_matches=matches,
                reconstructed_position_max_difference_over_length=position_difference,
                position_note='Native half-step positions are reversed at t=0; this is a roundoff-bounded comparison.',
                branches=comparisons)


def _pair_controls(sources):
    if len(sources) < 2:
        raise ValueError('pair comparisons require at least two complete controls')
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
            if len(changed) <= 1:
                comparison = _pair_comparison(reference, source, changed[0] if changed else 'repeat')
                quantities = ('additional_dark_depletion_gain_mean', 'total_dark_fraction_mean',
                              'coherent_dark_fraction_mean', 'nonzero_dark_fraction_mean')
                means = [late[i]['native_reductions']['coupled'] for i in (previous, index)]
                comparison['raw_window_mean_differences'] = {
                    key: _pair_metric(np.atleast_1d(means[0][key]), np.atleast_1d(means[1][key])) for key in quantities}
                comparison['raw_window_mean_gates'] = {
                    key: (row['relative_l2_difference'] <= a['accuracy_targets'][
                              'nonzero_mode_density_window_fraction' if key == 'nonzero_dark_fraction_mean'
                              else 'primary_window_fraction']
                          if row['relative_l2_difference'] is not None and changed != ['seed'] else None)
                    for key, row in comparison['raw_window_mean_differences'].items()}
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
                claim='Bounded seed and numerical controls; no late-convergence, uncertainty or nonlinear-cause claim')


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
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout='constrained')
        for source in sources:
            setting, data = source[0]['settings'], source[1]
            time, reservoir = data['coupled_t'], setting['initial_dark_energy_over_energy_scale']
            count = setting['cells'] * setting['particles_per_cell_per_species']
            label = f"δv/c={setting['velocity_seed_over_c']:g}, Δτ={setting['dt_omega0']:g}, "
            label += f"{setting['cells']} cells, {count / 1000:g}k/species"
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
                                  'Windows use all native scalar samples; QD means retain producer reductions. '
                                  'Local lab-frame variance energy is not thermodynamic temperature.'),
                 result, fig, **arrays)
        np.savez_compressed(Path(folder) / 'data.npz', **arrays)
        plt.close(fig)
    return result


if __name__ == '__main__':  # noqa: C901 — sequential evidence rendering
    print('Comparing native records', flush=True)
    if sum((implicit, picard, method_controls, bool(phase_controls), bool(pair_controls))) > 1:
        raise ValueError('select one comparison mode')
    if (refined or loading_refined or loading_repeat) and destination is None:
        raise ValueError('refinements require a publish folder')
    if loading_repeat and loading_refined is None:
        raise ValueError('a loading repeat requires its first record')
    if finer_mesh and not method_controls or orbit_audits and not (picard or phase_controls):
        raise ValueError('finer_mesh requires method_controls; orbit_audits requires picard or phase_controls')
    if pair_controls:
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
            publish(first, second, destination, result, refined, refined_variant, loading_refined, loading_repeat)
        else:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(result, indent=2))
    print(f"Saved comparison to {destination or output}", flush=True)
