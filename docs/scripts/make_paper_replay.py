"""Render existing Fig. 2 parameter-replay records; no simulation is launched."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from darkjaxincell import midnight  # noqa: E402
from jaxincell import save_run  # noqa: E402

# Edit these inputs; batch studies may pass the same names through runpy.init_globals.
records = Path(globals().get('records', 'artifacts'))
output = Path(globals().get('output', 'artifacts/paper_summary'))
reference = Path(globals().get('reference', Path(__file__).resolve().parents[1]
                               / '_static/figures/paper_replay'))
refined = tuple(map(Path, globals().get('refined', ())))  # consecutive half-step driven folders
repeat = globals().get('repeat', None)  # same-step execution control

if __name__ == '__main__':  # noqa: C901 — sequential evidence rendering
    print('Rendering saved resonant-drive records', flush=True)
    paths = {name: records / folder / 'data.npz' for name, folder in (
        ('drive', 'paper_full_drive'), ('zero', 'paper_full_zero'))}
    paths['paper'] = reference / 'reference.npz'
    a, zero, paper = [np.load(paths[key]) for key in ('drive', 'zero', 'paper')]
    drive_record = json.loads((paths['drive'].parent / 'run.json').read_text())
    zero_record = json.loads((paths['zero'].parent / 'run.json').read_text())
    reference_record = json.loads((reference / 'reference.json').read_text())
    if drive_record['settings']['drive_quiver_over_sigma'] != .03 or drive_record['settings']['coupling'] is not None:
        raise ValueError('this summary compares the strong prescribed-drive Figure 2 case')
    for key, value in (('length_c_over_omega_p', 40), ('mass_ratio', 1836), ('T_each_over_mec2', .001)):
        if drive_record['settings'][key] != value:
            raise ValueError(f'the published comparison requires {key}={value}')
    t = a['t']
    np.testing.assert_allclose(t[-1], 5000, rtol=0, atol=1e-5)
    np.testing.assert_array_equal(t, zero['t'])
    refinements = [(np.load(folder / 'data.npz'), json.loads((folder / 'run.json').read_text()))
                   for folder in refined]
    for record in [zero_record, *[row[1] for row in refinements]]:
        for key in ('length_c_over_omega_p', 'mass_ratio', 'T_each_over_mec2', 'coupling', 'parent_revision'):
            if record['settings'][key] != drive_record['settings'][key]:
                raise ValueError(f'replays must share physical {key}')
    for data, record in refinements:
        if not 1000 - 1e-5 <= data['t'][-1] <= 5000 + 1e-5:
            raise ValueError('refinements must cover the transition through tau=1000, within the comparison horizon')
        np.testing.assert_allclose(record['settings']['output_dt_omega_p'], .5, rtol=0, atol=1e-9)
        if record['settings']['drive_quiver_over_sigma'] != .03:
            raise ValueError('refinements must retain the same prescribed drive')
        for key in ('cells', 'particles_per_species', 'seed', 'loading', 'pusher', 'shape'):
            if record['settings'][key] != drive_record['settings'][key]:
                raise ValueError(f'timestep refinements must share {key}')
    for previous, (_, record) in zip([drive_record, *[row[1] for row in refinements]], refinements):
        np.testing.assert_allclose(record['settings']['dt_omega_p'], previous['settings']['dt_omega_p'] / 2,
                                   rtol=0, atol=1e-12)
    width = round(2 * np.pi / np.median(np.diff(t)))
    late = (t >= 4800 - 1e-5) & (t <= 5000 + 1e-5)
    refined_colors = ('#30343B', '#009E73', '#CC79A7')

    def averaged(values):
        """An approximately one-period moving average; endpoints are trimmed below."""
        return np.convolve(values, np.ones(width) / width, mode='same')

    def compare_windows(first, second):
        """Compare native samples in fixed inclusive windows, without interpolation."""
        count = min(len(first['t']), len(second['t']))
        np.testing.assert_allclose(first['t'][:count], second['t'][:count], rtol=0, atol=1e-5)
        clock = (first['t'][:count] + second['t'][:count]) / 2
        windows = []
        for start, end in ((0, 100), (0, 250), (0, 500), (0, 1000), (4800, 5000)):
            if end > clock[-1] + 1e-5:
                continue
            selected = (clock >= start - 1e-5) & (clock <= end + 1e-5)
            observables = {}
            for key in ('mean_E', 'rms', 'electric', 'nonzero_electric', 'density_rms'):
                x, y = first[key][:count][selected], second[key][:count][selected]
                difference, norm = np.linalg.norm(y - x, axis=0), np.linalg.norm(x, axis=0)
                observables[key] = dict(relative_l2_difference=(difference / norm).tolist())
                observables[key].update(l2_difference=difference.tolist(), reference_l2_norm=norm.tolist())
                if key != 'mean_E':
                    observables[key]['mean_relative_change'] = (y.mean(axis=0) / x.mean(axis=0) - 1).tolist()
            windows.append(dict(window_omega_p=[start, end], samples=int(selected.sum()), observables=observables))
        return windows

    with midnight():
        fig, axes = plt.subplots(2, 2, figsize=(10, 7), layout='constrained')
        colors = {'PIC': '#6A3D9A', 'no drive': '#6B7280', 'homogeneous': '#0072B2', 'paper': '#D55E00'}
        for i, species in enumerate(('electron', 'ion')):
            ax = axes[0, i]
            ax.plot(t, a['rms'][:, i], color=colors['PIC'], label='spatial PIC')
            ax.plot(t, zero['rms'][:, i], color=colors['no drive'], label='no drive')
            ax.plot(t, a['homogeneous_rms'][:, i], color=colors['homogeneous'], label='homogeneous kinetic')
            ax.plot(paper[f'strong_{species}_rms_t'], paper[f'strong_{species}_rms'], '--',
                    color=colors['paper'], label='Fig. 2, PDF curve')
            for j, (data, record) in enumerate(refinements):
                controls = record['settings']
                label = f"Δtωₚ={controls['dt_omega_p']:g}"
                if data['t'][-1] < 5000 - 1e-5:
                    label += f", τ≤{round(data['t'][-1])}"
                ax.plot(data['t'], data['rms'][:, i], ':', color=refined_colors[j % 3], label=label)
            ax.set(title=species.capitalize() + ' velocity spread', ylabel=rf'$\sigma_{{{species[0]}}}/c$')
        ax = axes[1, 0]
        trim = slice(width, -width)
        for key, label, color in (
                ('electric', 'spatial PIC, total', colors['PIC']),
                ('nonzero_electric', 'spatial PIC, nonzero k', '#B03568'),
                ('homogeneous_electric', 'homogeneous kinetic', colors['homogeneous'])):
            raw = a[key] / .0005
            ax.plot(t, raw, color=color, alpha=.12, lw=.5)
            ax.plot(t[trim], averaged(raw)[trim], color=color, label=label)
        ax.plot(paper['strong_electric_t'], paper['strong_electric'] / .0005, '--',
                color=colors['paper'], label='Fig. 2, PDF curve')
        for j, (data, record) in enumerate(refinements):
            color = refined_colors[j % 3]
            label = f"Δtωₚ={record['settings']['dt_omega_p']:g}"
            for key, style, kind in (('electric', ':', 'total'), ('nonzero_electric', '-.', 'nonzero k')):
                ax.plot(data['t'][trim], averaged(data[key] / .0005)[trim], style,
                        color=color, label=f'{label}, total/nonzero k' if kind == 'total' else '_nolegend_')
        ax.set(title='Electric-field energy', yscale='log', ylim=(.003, 70),
               ylabel=r'$U_E/(nT_{e0}L/2)$')
        ax = axes[1, 1]
        for i, color, label in ((0, colors['PIC'], 'electrons'), (1, '#009E73', 'ions')):
            ax.plot(t, a['density_rms'][:, i], color=color, alpha=.15, lw=.5)
            ax.plot(t[trim], averaged(a['density_rms'][:, i])[trim], color=color, label=label)
        ax.plot(t[trim], averaged(zero['density_rms'][:, 1])[trim], '--',
                color=colors['no drive'], label='no drive, ions')
        ax.set(title='Spatial density contrast', ylabel='density RMS / mean (grid scale)')
        for ax in axes.flat:
            ax.set(xlabel=r'$\omega_p t$', xlim=(0, 5000))
            ax.grid(alpha=.25)
            location = 'lower right' if ax is axes[0, 1] else 'upper right' if ax is axes[1, 0] else 'best'
            ax.legend(fontsize=9, loc=location, frameon=True, facecolor='white', framealpha=.9, edgecolor='none')
        target = [np.interp(t[-1], paper[f'strong_{s}_rms_t'], paper[f'strong_{s}_rms'])
                  for s in ('electron', 'ion')]
        masses = np.array([1, drive_record['settings']['mass_ratio']])
        b3_increment = .5 * masses * a['rms'][-1]**2 - .0005
        target_increment = .5 * masses * np.asarray(target)**2 - .0005
        results = {'final_rms_over_c': a['rms'][-1].tolist(),
                   'digitized_final_rms_over_c': target,
                   'final_rms_relative_difference': (a['rms'][-1] / target - 1).tolist(),
                   'final_B3_spread_increment_over_nmec2L': b3_increment.tolist(),
                   'digitized_B3_spread_increment_from_rms_over_nmec2L': target_increment.tolist(),
                   'B3_increment_relative_difference': (b3_increment / target_increment - 1).tolist(),
                   'all_step_conservation': {key: value for key, value in drive_record['results'].items()
                                             if key.startswith('max_') and key != 'max_speed_over_c'},
                   'no_drive_late_global_spread_over_initial': zero_record['results'][
                       'late_electron_spread_over_initial'],
                   'late_global_spread_over_initial': (a['spread'][late].mean(axis=0) / .0005).tolist(),
                   'native_runs': {'drive': drive_record, 'no_drive': zero_record},
                   'refinements': [record for _, record in refinements],
                   'claim': 'Figure 2 parameter replay; grid, timestep, loading and seed convergence pending'}
        results['refinement_comparisons'] = []
        for i, (data, record) in enumerate(refinements):
            mask = (data['t'] >= 4800 - 1e-5) & (data['t'] <= 5000 + 1e-5)
            comparison = dict(
                cells=record['settings']['cells'], dt_omega_p=record['settings']['dt_omega_p'],
                particles_per_species=record['settings']['particles_per_species'],
                compared_to_dt_omega_p=(drive_record if i == 0 else refinements[i - 1][1])['settings']['dt_omega_p'],
                windows=compare_windows(a if i == 0 else refinements[i - 1][0], data),
                late_samples=int(mask.sum()))
            if mask.any():
                comparison.update(
                    late_global_spread_relative_change=(data['spread'][mask].mean(axis=0)
                                                        / a['spread'][late].mean(axis=0) - 1).tolist(),
                    **{f'late_{key}_relative_change': float(data[key][mask].mean() / a[key][late].mean() - 1)
                       for key in ('electric', 'nonzero_electric')})
            results['refinement_comparisons'].append(comparison)
        if len(refinements) == 2:
            contraction = []
            first, second = results['refinement_comparisons']
            for window in second['windows']:
                preceding = next(w for w in first['windows'] if w['window_omega_p'] == window['window_omega_p'])
                ratios = {key: (np.asarray(preceding['observables'][key]['l2_difference'])
                                / np.asarray(values['l2_difference'])).tolist()
                          for key, values in window['observables'].items()}
                contraction.append(dict(window_omega_p=window['window_omega_p'], error_ratio=ratios))
            results['adjacent_error_contraction'] = contraction
        results['first_electron_spread_crossings'] = []
        for data in (a, *[row[0] for row in refinements]):
            crossings = {}
            for threshold in (2, 5, 10, 20):
                indices = np.flatnonzero(data['spread'][:, 0] / data['spread'][0, 0] >= threshold)
                index = int(indices[0]) if len(indices) else None
                crossings[threshold] = None if index is None else data['t'][max(index - 1, 0):index + 1].tolist()
            results['first_electron_spread_crossings'].append(crossings)
        repeat_data = None
        if repeat:
            repeat_data = np.load(repeat / 'data.npz')
            repeat_record = json.loads((repeat / 'run.json').read_text())
            controls = ('cells', 'particles_per_species', 'dt_omega_p', 'seed', 'length_c_over_omega_p',
                        'mass_ratio', 'T_each_over_mec2', 'drive_quiver_over_sigma', 'coupling', 'parent_revision',
                        'loading', 'pusher', 'shape', 'output_dt_omega_p')
            original, original_record = next((data, record) for data, record in ((a, drive_record), *refinements)
                                             if all(record['settings'][key] == repeat_record['settings'][key]
                                                    for key in controls))
            revision = repeat_record['git']
            if revision != original_record['git'] or revision == 'unknown' or revision.endswith('-dirty'):
                raise ValueError('repeat control must share the original clean computational revision')
            results['repeat_comparison'] = dict(record=repeat_record, compared_to_git=original_record['git'],
                                                windows=compare_windows(original, repeat_data),
                                                note='same timestep and loading; different horizon/compiler allocation')
        settings = {'raw_computation_git': drive_record['git'],
                    'computation': {name: {key: record[key] for key in (
                        'git', 'jaxincell', 'jax', 'numpy', 'python', 'platform', 'jax_enable_x64', 'backend')}
                        for name, record in (('drive', drive_record), ('zero', zero_record))},
                    'cells': drive_record['settings']['cells'],
                    'total_markers': 2 * drive_record['settings']['particles_per_species'],
                    'dt_omega_p': drive_record['settings']['dt_omega_p'],
                    'reference': 'arXiv:2510.13956v1 Figure 2, original 2025 PDF',
                    'reference_note': 'PDF vector curves, not raw author simulation data; RMS comparison uses B2',
                    'reference_pdf_sha256': reference_record['source_sha256'],
                    'reference_visibility_filter': reference_record['visibility_filter'],
                    'reference_axis_calibration': {key: reference_record[key] for key in (
                        'x_pdf_zero', 'x_pdf_5000', 'strong_electric', 'strong_electron_rms', 'strong_ion_rms')},
                    'energy_density_plot_average_samples': width,
                    'average_width_omega_p': float(width * np.median(np.diff(t))),
                    'late_window_omega_p': [4800, 5000], 'window_endpoint_tolerance_omega_p': 1e-5,
                    'input_sha256': {key: hashlib.sha256(path.read_bytes()).hexdigest() for key, path in paths.items()},
                    'analysis_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        settings['input_sha256'].update({f'refined{i}': hashlib.sha256((folder / 'data.npz').read_bytes()).hexdigest()
                                         for i, folder in enumerate(refined)})
        if repeat:
            settings['input_sha256']['repeat'] = hashlib.sha256((repeat / 'data.npz').read_bytes()).hexdigest()
        keys = ('t', 'rms', 'mean_E', 'spread', 'electric', 'nonzero_electric', 'density_rms')
        arrays = {key: a[key] for key in (*keys, 'homogeneous_rms', 'homogeneous_electric')}
        arrays.update(no_drive_rms=zero['rms'], no_drive_density_rms=zero['density_rms'])
        arrays.update({f'refined{i}_{key}': data[key] for i, (data, _) in enumerate(refinements)
                       for key in (*keys, 'balance')})
        if repeat_data is not None:
            arrays.update({f'repeat_{key}': repeat_data[key] for key in keys})
        arrays.update({key: paper[key] for key in paper.files if key.startswith('strong_')})
        save_run(output, 'hook_parameter_replay_summary', settings, results, fig, **arrays)
        np.savez_compressed(output / 'data.npz', **arrays)
        plt.close(fig)
    print(f'Saved figure and data to {output}', flush=True)
