"""Render existing Fig. 2 parameter-replay records; no simulation is launched."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from darkjaxincell import midnight  # noqa: E402
from jaxincell import save_run  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--records', type=Path, default=Path('artifacts'))
parser.add_argument('--output', type=Path, default=Path('artifacts/paper_summary'))
parser.add_argument('--reference', type=Path, default=Path(__file__).resolve().parents[1]
                    / '_static' / 'figures' / 'paper_replay', help='digitized reference folder')
parser.add_argument('--refined', type=Path, action='append', default=[], help='additional driven replay folders')
args = parser.parse_args()
paths = {name: args.records / folder / 'data.npz' for name, folder in (
    ('drive', 'paper_full_drive'), ('zero', 'paper_full_zero'))}
paths['paper'] = args.reference / 'reference.npz'
a, zero, paper = [np.load(paths[key]) for key in ('drive', 'zero', 'paper')]
drive_record = json.loads((paths['drive'].parent / 'run.json').read_text())
zero_record = json.loads((paths['zero'].parent / 'run.json').read_text())
reference_record = json.loads((args.reference / 'reference.json').read_text())
if drive_record['settings']['drive_quiver_over_sigma'] != .03 or drive_record['settings']['coupling'] is not None:
    raise ValueError('this summary compares the strong prescribed-drive Figure 2 case')
t = a['t']
np.testing.assert_array_equal(t, zero['t'])
refinements = [(np.load(folder / 'data.npz'), json.loads((folder / 'run.json').read_text()))
               for folder in args.refined]
for record in [zero_record, *[row[1] for row in refinements]]:
    for key in ('length_c_over_omega_p', 'mass_ratio', 'T_each_over_mec2', 'coupling', 'parent_revision'):
        if record['settings'][key] != drive_record['settings'][key]:
            raise ValueError(f'replays must share physical {key}')
for _, record in refinements:
    if record['settings']['drive_quiver_over_sigma'] != .03:
        raise ValueError('refinements must retain the same prescribed drive')
width = round(2 * np.pi / np.median(np.diff(t)))


def averaged(values):
    """An approximately one-period moving average; endpoints are trimmed below."""
    return np.convolve(values, np.ones(width) / width, mode='same')


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
        for data, record in refinements:
            controls = record['settings']
            label = f"{controls['cells']} cells, Δtωₚ={controls['dt_omega_p']:g}"
            ax.plot(data['t'], data['rms'][:, i], ':', label=label)
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
        ax.legend(fontsize=8)
    target = [np.interp(t[-1], paper[f'strong_{s}_rms_t'], paper[f'strong_{s}_rms'])
              for s in ('electron', 'ion')]
    results = {'final_rms_over_c': a['rms'][-1].tolist(),
               'digitized_final_rms_over_c': target,
               'final_rms_relative_difference': (a['rms'][-1] / target - 1).tolist(),
               'all_step_conservation': {key: value for key, value in drive_record['results'].items()
                                         if key.startswith('max_') and key != 'max_speed_over_c'},
               'no_drive_late_global_spread_over_initial': zero_record['results'][
                   'late_electron_spread_over_initial'],
               'late_global_spread_over_initial': (a['spread'][t >= 4800].mean(axis=0) / .0005).tolist(),
               'refinements': [record for _, record in refinements],
               'claim': 'Figure 2 parameter replay; grid, timestep, loading and seed convergence pending'}
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
                'input_sha256': {key: hashlib.sha256(path.read_bytes()).hexdigest() for key, path in paths.items()},
                'analysis_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    keys = ('t', 'rms', 'electric', 'nonzero_electric', 'density_rms',
            'homogeneous_rms', 'homogeneous_electric')
    arrays = {key: a[key] for key in keys}
    arrays.update(no_drive_rms=zero['rms'], no_drive_density_rms=zero['density_rms'])
    arrays.update({f'refined{i}_{key}': data[key] for i, (data, _) in enumerate(refinements)
                   for key in ('t', 'rms', 'balance', 'electric')})
    arrays.update({key: paper[key] for key in paper.files if key.startswith('strong_')})
    save_run(args.output, 'hook_parameter_replay_summary', settings, results, fig, **arrays)
    np.savez_compressed(args.output / 'data.npz', **arrays)
    plt.close(fig)
