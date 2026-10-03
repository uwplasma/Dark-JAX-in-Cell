"""Recheck archived experimental PIC impulses without advancing particles.

This reproduces the force-accounting figure, not the unreleased integrator.
The complete selected map and independent interpolation controls are archived.
"""
from pathlib import Path
import hashlib
import json

import matplotlib
import numpy as np

from darkjaxincell import midnight
from jaxincell import save_run

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402


archive = Path(globals().get(
    'archive', Path(__file__).resolve().parents[1] / '_static/figures/paired_midpoint'))
output = Path(globals().get('output', 'artifacts/paired_midpoint'))

evidence = json.loads((archive / 'evidence.json').read_text())
assert hashlib.sha256((archive / 'data.npz').read_bytes()).hexdigest() == evidence['data_sha256']
with np.load(archive / 'data.npz', allow_pickle=False) as saved:
    data = {key: saved[key] for key in saved.files}
old = {key.removeprefix('initial.'): value for key, value in data.items() if key.startswith('initial.')}
end = {key.removeprefix('selected.after.'): value for key, value in data.items()
       if key.startswith('selected.after.')}
trace = {key.removeprefix('selected.native.trace.'): value for key, value in data.items()
         if key.startswith('selected.native.trace.')}
c, eps, mass = (evidence['constants_SI'][key] for key in ('c', 'epsilon_0', 'mass_electron'))
wp, pref = (evidence['normalization'][key] for key in ('WP', 'PREF'))
length = 2 * np.pi * c / wp
dx, dt, eta, omega = length / 8, float(data['selected.dt']), .17, .83 * wp
weights, charge = old['ordinary.w'], mass * old['ordinary.qm']


def centre(value):
    """The recorded grid's face-to-centre average."""
    return (value + np.roll(value, 1, axis=0)) / 2


def momentum(state):
    """Physical particle and Maxwell–Proca field momentum, in SI units."""
    particle = np.sum((mass * state['ordinary.w'])[:, None] * state['ordinary.u'], axis=0)
    density = np.cross(centre(state['ordinary.E']), state['ordinary.B'])
    density += np.cross(centre(state['E']), state['B'])
    density += (omega / c)**2 * state['phi'][:, None] * centre(state['A'])
    return particle, eps * dx * np.sum(density, axis=0)


print('Checking the saved forward map and signed force transfer', flush=True)
p0, f0 = momentum(old)
p1, f1 = momentum(end)
F = (old['ordinary.E'] + end['ordinary.E'] + eta * (old['E'] + end['E'])) / 2
H = (old['ordinary.B'] + end['ordinary.B'] + eta * (old['B'] + end['B'])) / 2
rho = (old['ordinary.rho'] + end['ordinary.rho']) / 2
J = data['selected.native.J']
grid_e = dt * dx * np.sum((rho + old['background'])[:, None] * centre(F), axis=0)
grid_b = dt * dx * np.sum(np.cross(centre(J), H), axis=0)
electric, magnetic, rotation = np.zeros((3, 3))
for j in range(2):
    E, B = trace['fields'][j, :, :3], trace['fields'][j, :, 3:]
    u0, u1 = trace['u0'][j], trace['u1'][j]
    minus = u0 + dt / 4 * old['ordinary.qm'][:, None] * E
    vB = (u0 + u1) / (2 * np.sqrt(1 + np.sum((minus / c)**2, axis=1, keepdims=True)))
    amount = dt / 2 * (charge * weights)[:, None]
    electric += np.sum(amount * E, axis=0)
    magnetic += np.sum(amount * np.cross(trace['velocity'][j], B), axis=0)
    rotation += np.sum(amount * np.cross(vB - trace['velocity'][j], B), axis=0)
pieces = dict(electric_mismatch=(electric - grid_e) / pref,
              magnetic_mismatch=(magnetic - grid_b) / pref, Boris_correction=rotation / pref,
              pusher_residual=(p1 - p0 - electric - magnetic - rotation) / pref,
              field_grid_residual=(f1 - f0 + grid_e + grid_b) / pref)
change = (p1 - p0 + f1 - f0) / pref
bound = evidence['bound']
assert np.array_equal(old['background'], end['background'])
assert np.array_equal(old['ordinary.w'], end['ordinary.w'])
assert np.max(abs(change - sum(pieces.values()))) <= bound
assert np.max(abs(pieces['field_grid_residual'])) <= bound
for key in ('electric_mismatch', 'magnetic_mismatch'):
    assert np.max(abs(pieces[key] - evidence['force_split'][key])) <= bound
assert np.max(abs(J - data['selected.info.current'])) / evidence['normalization']['JREF'] <= bound
assert np.max(abs(change)) > bound  # Preserve the measured physical momentum failure.
results = dict(physical_momentum_change=change.tolist(),
               signed_terms={key: value.tolist() for key, value in pieces.items()},
               momentum_gate=False, bound=bound, production_adoption=False,
               source_evidence_sha256=hashlib.sha256((archive / 'evidence.json').read_bytes()).hexdigest())
for method, rows in evidence['temporal']['self_convergence'].items():
    for row in rows:
        order = np.log2(row['coarse_difference'][-1] / row['fine_difference'][-1])
        assert abs(order - row['order'][-1]) < 1e-12
        print(f'{method}: aggregate order {order:.6f}; controls valid={row["controls_valid"]}', flush=True)

with midnight():
    fig, ax = plt.subplots(figsize=(9, 4.5), layout='constrained')
    keys = ('electric_mismatch', 'magnetic_mismatch', 'Boris_correction', 'field_grid_residual')
    values = [change[1], *[pieces[key][1] for key in keys]]
    ax.bar(range(5), np.abs(values), color=['#202124', '#0072B2', '#D55E00', '#6A3D9A', '#009E73'])
    ax.set_xticks(range(5), [
        'Total physical\nchange', 'Electric\ntransfer', 'Magnetic\ntransfer',
        'Boris\ncorrection', 'Field + grid\nresidual'])
    ax.set_yscale('log')
    ax.set_ylim(1e-21, 1e-5)
    for j, value in enumerate(values):
        ax.annotate(f'{value:+.2e}', (j, abs(value)), xytext=(0, 5),
                    textcoords='offset points', ha='center', fontsize=10)
    ax.axhline(bound, color='#30343B', lw=1, ls='--', label='Original momentum bound')
    ax.legend(loc='upper right', frameon=True, facecolor='white', framealpha=1)
    ax.set(ylabel=r'Magnitude of $y$ impulse / $P_\star$',
           title='One warm Maxwell–Proca step: magnetic transfer dominates the momentum defect')
    save_run(output, 'audit_paired_midpoint',
             dict(archived_map_only=True, native_steps=0, source_sha256=evidence['source_sha256'],
                  analysis_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()), results, fig)
    plt.close(fig)
print(f'Physical momentum defect {np.max(abs(change)):.6e}; original bound {bound:.1e}', flush=True)
