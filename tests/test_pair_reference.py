"""Analytic checks for the independent oscillating-pair waterbag calculation."""

import numpy as np
import pytest

from docs.scripts.pair_reference import (coupled_response, edge_matrix, growth,
                                         relativistic_background, relativistic_response, seeded_response)


def test_unpumped_pair_dispersion_and_ballistic_edges():
    """The stationary pair has one Langmuir pair and two free waterbag edges."""
    for b in (0.2, 0.6, 1.0):
        frequency = np.sort(np.abs(np.linalg.eigvals(edge_matrix(0, b / .05, .05, 0))))
        np.testing.assert_allclose(frequency, [b, b, np.hypot(1, b), np.hypot(1, b)],
                                   rtol=1e-12)


def test_oscillating_pair_has_thermal_band_and_stable_control():
    """The time-dependent propagator resolves growth absent from the static limit."""
    growing, multiplier = growth(0.6 / .05)
    stable, _ = growth(0.4 / .05)
    assert 0.09 < growing < 0.11
    assert multiplier.real > 1
    assert abs(stable) < 1e-8
    assert abs(growth(0.6 / .05, quiver=0)[0]) < 1e-8


def test_seeded_response_starts_from_ampere_current():
    """Opposite velocity nudges have zero initial charge and J_k=seed/2."""
    seed = 2e-4
    times = np.array([0., 1e-4, 2e-4])
    field = seeded_response(times, 12, seed)
    assert abs(field[0]) < 1e-14
    np.testing.assert_allclose((field[2] - field[0]).real / times[2], -seed / 2,
                               rtol=1e-6)


def test_coupled_initial_value_reference_has_ordinary_limit_and_energy():
    """No mixing recovers the ordinary kinetic mode; a closed pump preserves energy."""
    times = np.linspace(0, 35, 176)
    quiver, k, seed = 0.2 / np.sqrt(2), 12, 2e-4
    ordinary, dark, _ = coupled_response(times, k, seed, 0, 1, quiver, 0)
    np.testing.assert_allclose(ordinary, seeded_response(times, k, seed), atol=1e-10)
    np.testing.assert_allclose(dark, 0, atol=1e-12)
    _, _, background = coupled_response(times, k, seed, 0.5, 1, 0, 0.2)
    np.testing.assert_allclose(np.sum(background**2, axis=0), 0.2**2, atol=1e-10)


def test_relativistic_quadrature_preserves_initial_ampere_current():
    """A velocity nudge gives the exact current even though momentum varies with speed."""
    times = np.array([0., 1e-4, 2e-4])
    ordinary, dark, _ = relativistic_response(times, 12, 2e-4, 0.5, 1, 0, 0.1, nodes=16)
    np.testing.assert_allclose(ordinary[0], 0, atol=1e-14)
    np.testing.assert_allclose(ordinary[2].real / times[2], -1e-4, rtol=1e-6)
    np.testing.assert_allclose(dark[2].real / times[2], -5e-5, rtol=1e-6)


def test_relativistic_velocity_quadrature_refines():
    """The independent kinetic trace is stable when velocity nodes double."""
    times = np.linspace(0, 10, 51)
    coarse = relativistic_response(times, 12, 2e-4, 0.5, 1, 0, 0.1, nodes=16)
    fine = relativistic_response(times, 12, 2e-4, 0.5, 1, 0, 0.1, nodes=32)
    assert np.linalg.norm(coarse[0] - fine[0]) / np.linalg.norm(fine[0]) < 1e-4


def test_homogeneous_warm_pair_energy_quadrature_and_table_refinement():
    """Continuous energy and forcing interpolation have independent error budgets."""
    times = np.linspace(0, 4, 401)
    coarse = relativistic_background(times, .5, 1., 0., .1, nodes=32)
    fine = relativistic_background(times, .5, 1., 0., .1, nodes=64)
    np.testing.assert_allclose(coarse['mean_D'], fine['mean_D'], rtol=0, atol=2e-13)
    np.testing.assert_allclose(fine['energy'], fine['energy'][0], rtol=0, atol=2e-12)
    errors = [np.max(abs(np.interp(times, times[::stride], fine['mean_D'][::stride]) - fine['mean_D']))
              for stride in (4, 2)]
    assert errors[1] < .3 * errors[0]
    assert fine['impulse'][0] == 0 and fine['mean_A'][0] == 0


def test_cold_pair_background_has_two_frequencies_and_zero_mixing_limit():
    """Low-amplitude cold fields obey the independently diagonalized two-fluid oscillator."""
    times = np.linspace(0, 3, 31)
    eta, mass, amplitude = .4, 1.3, 1e-6
    frequency_squared, vectors = np.linalg.eigh([[1., eta], [eta, mass**2 + eta**2]])
    exact = vectors @ (np.cos(np.sqrt(frequency_squared)[:, None] * times)
                       * (vectors.T @ [0., amplitude])[:, None])
    result = relativistic_background(times, eta, mass, 0., amplitude, half_width=0, nodes=1,
                                     rtol=2e-13, atol=2e-15)
    np.testing.assert_allclose([result['mean_E'], result['mean_D']], exact, rtol=0, atol=2e-13)
    uncoupled = relativistic_background(times, 0., mass, 0., .1, nodes=16)
    np.testing.assert_allclose(uncoupled['mean_D'], .1 * np.cos(mass * times), rtol=0, atol=3e-12)
    np.testing.assert_allclose(uncoupled['mean_A'], -.1 * np.sin(mass * times) / mass, rtol=0, atol=3e-12)
    np.testing.assert_allclose(uncoupled['mean_E'], 0, rtol=0, atol=1e-15)


def test_prescribed_pair_tangent_removes_spatial_force_but_keeps_envelope():
    """The intervention has the same warm orbits and initial current, with no dark mode."""
    times = np.array([0., 1e-4, 2e-4])
    args = (times, 12., 2e-4, .5, 1., 0., .1)
    coupled = relativistic_response(*args, nodes=16)
    prescribed = relativistic_response(*args, nodes=16, spatial_eta=0.)
    np.testing.assert_array_equal(coupled[2], prescribed[2])
    np.testing.assert_allclose(prescribed[0][-1].real / times[-1], -1e-4, rtol=1e-6)
    np.testing.assert_array_equal(prescribed[1], 0)
    plain = relativistic_response(times, 12., 2e-4, 0., 1., 0., 0., nodes=16)
    zero_mixing = relativistic_response(times, 12., 2e-4, 0., 1., 0., .1, nodes=16, spatial_eta=0.)
    np.testing.assert_allclose(plain[0], zero_mixing[0], rtol=0, atol=1e-13)


def test_pair_push_table_matches_source_mass_half_kick():
    """A nonzero old weighted current and potential produce the actual midpoint force."""
    import jax.numpy as jnp
    from jax import random
    from jaxincell import elementary_charge as e, mass_electron, speed_of_light as c
    from darkjaxincell import DarkField, DarkSimulation, PrescribedDrive
    from darkjaxincell._proca import kick
    from examples.dark_reservoir import pair_plasma, pair_push_table, snapshot

    plasma, _, _, wp = pair_plasma(512, 2, .02, 0., True)
    scale, eta, omega = mass_electron * c * wp / e, .4, 1.3 * wp
    sim = DarkSimulation(plasma, DarkField(omega, eta, initial_E=jnp.tile(
        jnp.array([.1 * scale, 0., 0.]), (512, 1))))
    start, _ = sim.initial_state(random.PRNGKey(0))
    _, charge = plasma.per_particle
    ordinary = start.ordinary.replace(u=start.ordinary.u.at[:, 0].add(charge / e * .003 * c))
    start = sim.continue_with_parameters(start.replace(
        ordinary=ordinary, A=start.A.at[:, 0].set(.02 * scale / wp)))
    first = {key: np.asarray(value) for key, value in snapshot(sim, start).items()}
    history = {key: np.stack((first[key], first[key])) for key in ('t', 'mean', 'mean_D', 'mean_A')}
    history['t'][1] += plasma.domain.dt
    times, table = pair_push_table(plasma, ordinary, history, omega, eta)
    velocity = plasma._velocity(ordinary.u)
    particle_charge = charge * ordinary.w
    closure = plasma._current_closure(particle_charge * velocity[:, 0])
    _, current = plasma._sources(ordinary.x, velocity, particle_charge, plasma.domain.dt / 2,
                                 closure, ordinary.rho)
    actual, _ = kick(start.E, start.B, start.A, start.phi, current,
                     plasma.domain.dt / 2, plasma.domain.dx, omega, eta)
    np.testing.assert_allclose(table[1], np.mean(np.asarray(actual[:, 0])), rtol=2e-15)
    amplitude = jnp.stack((jnp.asarray(table), jnp.zeros(3), jnp.zeros(3)), axis=1)
    drive = PrescribedDrive(eta, amplitude, 0., times=jnp.asarray(times))
    np.testing.assert_allclose(drive.at(times[1])[0], table[1], rtol=0, atol=1e-14 * scale)


def test_global_pair_loading_has_balanced_uniform_quantiles_and_longer_recurrence():
    """Spatial pairing preserves neutrality; midpoint quadrature has an analytic kernel."""
    from jaxincell import speed_of_light as c
    from examples.dark_reservoir import pair_plasma

    times = np.array([0., 10., 50., 167., 170.])
    errors = []
    for loading, count in (('cell', 32), ('global', 512 * 32)):
        plasma, species, k, wp = pair_plasma(512, 32, .01, 0., True, pair_loading=loading)
        velocity = np.asarray(species[0].v[:, 0]) / c
        np.testing.assert_array_equal(species[0].v, species[1].v)
        np.testing.assert_array_equal(species[0].x, species[1].x)
        np.testing.assert_allclose(velocity.reshape(512, 32).mean(axis=1), 0, atol=3e-18)
        assert species[0].charge == -species[1].charge and species[0].density == species[1].density
        assert len(np.unique(velocity)) == count
        np.testing.assert_allclose(np.mean(velocity**2), .05**2 / 3 * (1 - count**-2), rtol=2e-15)
        z = k * c / wp * .05 * times
        kernel = np.mean(np.exp(-1j * velocity[:, None] * k * c / wp * times), axis=0)
        np.testing.assert_allclose(kernel, np.sinc(z / np.pi) / np.sinc(z / (count * np.pi)), atol=2e-13)
        errors.append(np.max(abs(kernel - np.sinc(z / np.pi))))
    assert errors[0] > .9 and errors[1] < 1e-7


def test_global_loading_refines_seeded_ballistic_charge_not_just_velocity_kernel():
    """The 2k counterterm from x/v correlations is checked in the actual seed functional."""
    from jaxincell import speed_of_light as c
    from examples.dark_reservoir import pair_plasma

    times = np.array([0., 20., 50., 100., 150., 167.5, 170.])
    errors = []
    for ppc in (32, 64, 128):
        _, species, k, wp = pair_plasma(512, ppc, .01, 0., True, pair_loading='global')
        k *= c / wp
        x, velocity = np.asarray(species[0].x[:, 0]) * wp / c, np.asarray(species[0].v[:, 0]) / c
        ballistic_phase = np.exp(-1j * k * (x[:, None] + velocity[:, None] * times))
        charge_tangent = -1j * k * times * np.mean(np.cos(k * x)[:, None] * ballistic_phase, axis=0)
        continuum = -.5j * k * times * np.sinc(k * .05 * times / np.pi)
        assert charge_tangent[0] == 0
        errors.append(np.max(abs(charge_tangent - continuum)))
    assert errors[1] < .4 * errors[0] and errors[2] < .15 * errors[1]
    assert errors[2] < .1  # Per unit velocity seed; this does not certify nonlinear PIC accuracy.


@pytest.fixture
def waveform_initial_archive(tmp_path):
    import jax.numpy as jnp
    from jax import random
    from darkjaxincell import DarkField, DarkSimulation
    from examples.dark_reservoir import pair_plasma, save_compressed_state
    from jaxincell import elementary_charge as e, mass_electron, speed_of_light as c

    plasma, _, _, wp = pair_plasma(512, 2, .025, 2e-4, True, pair_loading='global')
    model = DarkField(wp, .5, initial_E=jnp.tile(
        jnp.array([.1 * mass_electron * c * wp / e, 0., 0.]), (512, 1)))
    sim = DarkSimulation(plasma, model)
    state, _ = sim.initial_state(random.PRNGKey(0))
    # One field ULP and a ledger value distinguish restoration from regeneration.
    ordinary = state.ordinary.replace(E=state.ordinary.E.at[0, 0].set(
        np.nextafter(float(state.ordinary.E[0, 0]), np.inf)))
    state = state.replace(ordinary=ordinary, max_balance_error=jnp.asarray(1e-20))
    path = tmp_path / 'initial.npz'
    save_compressed_state(path, state, sim)
    return path, state, sim


def test_waveform_restores_complete_initial_archive_before_advancing(tmp_path, monkeypatch, waveform_initial_archive):
    from contextlib import nullcontext
    from examples import dark_reservoir as example

    path, _, _ = waveform_initial_archive

    class AcceptedArchive(Exception):
        pass

    def inspect(sim, initial, *args, **kwargs):
        saved = tmp_path / 'restored.npz'
        example.save_compressed_state(saved, initial, sim)
        with np.load(path) as original, np.load(saved) as restored:
            assert original.files == restored.files
            for key in original.files:
                np.testing.assert_array_equal(restored[key], original[key])
        raise AcceptedArchive

    monkeypatch.setattr(example, 'paper_run', inspect)
    monkeypatch.setattr(example, 'elapsed_progress', lambda *args: nullcontext())
    with pytest.raises(AcceptedArchive):
        example.pair_waveform_control(tmp_path, 512, 2, .025, .05, quadrature=4, initial_state=path)


@pytest.mark.parametrize('change', ['time', 'work', 'dt', 'eta', 'force', 'loading', 'shape'])
def test_waveform_initial_archive_rejects_changed_experiment(tmp_path, waveform_initial_archive, change):
    from examples.dark_reservoir import pair_waveform_control, save_compressed_state

    path, state, sim = waveform_initial_archive
    settings = dict(quadrature=4, initial_state=path)
    dtau = .025
    if change in ('time', 'work'):
        state = (state.replace(ordinary=state.ordinary.replace(time=sim.plasma.domain.dt)) if change == 'time'
                 else state.replace(work=1.))
        save_compressed_state(path, state, sim)
    elif change == 'dt':
        dtau = .0125
    else:
        settings.update({{'eta': 'eta', 'force': 'force', 'loading': 'seed', 'shape': 'shape_order'}[change]:
                         {'eta': .4, 'force': .04, 'loading': 0., 'shape': 5}[change]})
    with pytest.raises((ValueError, AssertionError)):
        pair_waveform_control(tmp_path, 512, 2, dtau, .05, **settings)


@pytest.mark.parametrize('study', ['pair_waveform', 'pair_repeat'])
def test_waveform_named_input_dispatcher_passes_initial_archive(tmp_path, study):
    import ast
    from pathlib import Path
    from examples import dark_reservoir as example

    calls = []

    def stub(*args, **kwargs):
        calls.append((args, kwargs))

    namespace = {**vars(example), '__name__': '__main__', 'study': study, 'samples': 2,
                 'initial_state': tmp_path / 'initial.npz', 'pair_waveform_control': stub, 'pair_repeat': stub}
    # Execute the actual dispatch block with a producer stub, avoiding five runs.
    dispatch = ast.parse(Path(example.__file__).read_text()).body[-1]
    exec(compile(ast.Module(body=[dispatch], type_ignores=[]), example.__file__, 'exec'), namespace)
    assert len(calls) == 1
    assert calls[0][0][-1 if study == 'pair_waveform' else 1] == namespace['initial_state']
    if study == 'pair_repeat':
        assert calls[0][0][2:] == (2, False)


@pytest.fixture
def prescribed_pair_archive(tmp_path, waveform_initial_archive):
    import json
    import jax.numpy as jnp
    from darkjaxincell import DarkSimulation, PrescribedDrive
    from examples import dark_reservoir as example

    _, state, coupled = waveform_initial_archive
    plasma = coupled.plasma
    wp, field = 1e9, example.mass_electron * example.c * 1e9 / example.e
    energy = example.epsilon_0 * plasma.domain.length * field**2
    times = jnp.array([0., .5, 1.5, 2.]) * plasma.domain.dt
    amplitude = jnp.stack((jnp.array([.1, .12, .08, .09]) * field, jnp.zeros(4), jnp.zeros(4)), axis=1)
    sim = DarkSimulation(plasma, PrescribedDrive(.5, amplitude, 0., times=times))
    start = sim.continue_with_parameters(state.replace(
        E=None, B=None, A=None, phi=None, initial_projection_norm=jnp.zeros(())))
    start = start.replace(max_balance_error=jnp.asarray(1e-20))
    path = tmp_path / 'donor' / 'realized_fine' / 'initial_state.npz'
    example.save_compressed_state(path, start, sim)
    settings = dict(label='realized_fine', model='PrescribedDrive', parent_revision=example.parent_revision(),
                    cells=512, particles_per_cell_per_species=2, pair_loading='global', dt_omega0=.025,
                    horizon_omega0=.05, shape_order=2, seed_mode=134, scalar_dt_omega0=.025,
                    forcing_native_dt_omega0=.025, velocity_seed_over_c=2e-4, eta=.5,
                    dark_mass_over_omega0=1., force_quiver_over_c=.05, block_steps=2,
                    scalar_units='native SI', XLA_FLAGS=example.pair_xla_flags(),
                    normalization=dict(omega0_rad_s=wp, field_scale_V_m=field, energy_scale_J_m2=energy))
    hashes = {key: example.array_fingerprint(getattr(start.ordinary, key)) for key in ('x', 'u', 'w', 'E', 'B', 'rho')}
    results = dict(initial_ordinary_fingerprints=hashes,
                   all_step_maxima=example.pair_all_step_maxima(jnp.zeros(9), plasma, wp, energy))
    native = example.provenance(example='pair_waveform_branch', settings=settings, results=results)
    top = example.provenance(
        example='pair_waveform_control',
        settings={**settings, 'local_moments': True, 'local_spread_lengths_c_over_omega0': [.1, .2],
                  'initial_ordinary_fingerprints': hashes}, results=dict(cases={'realized_fine': results}))
    (path.parent / 'run.json').write_text(json.dumps(native))
    (path.parent.parent / 'run.json').write_text(json.dumps(top))
    return path, sim, start


def test_archived_pair_repeat_preserves_table_midpoint_force_complete_restart_and_work(
        tmp_path, prescribed_pair_archive):
    import hashlib
    import json
    from darkjaxincell import DarkSimulation, PrescribedDrive, load_state
    from examples import dark_reservoir as example

    path, donor, initial = prescribed_pair_archive
    sim, start, _ = example.pair_repeat_initial(path)
    np.testing.assert_array_equal(sim.dark.amplitude, donor.dark.amplitude)
    np.testing.assert_array_equal(sim.dark.times, donor.dark.times)
    expected = donor.dark.amplitude[1]  # The first exact knot is the first accepted midpoint.
    np.testing.assert_array_equal(sim.dark.at(sim.plasma.domain.dt / 2), expected)
    accepted = sim._step(start, sim.plasma.per_particle)[0]
    constant = DarkSimulation(sim.plasma, PrescribedDrive(sim.dark.eta, expected, 0.))
    known = constant._step(start, sim.plasma.per_particle)[0]
    np.testing.assert_allclose(accepted.ordinary.u / example.c, known.ordinary.u / example.c, atol=2e-14)
    np.testing.assert_allclose(accepted.work, known.work, rtol=2e-13)
    midpoint = tmp_path / 'midpoint.npz'
    example.save_compressed_state(midpoint, accepted, sim)
    resumed = sim._step(load_state(midpoint, sim), sim.plasma.per_particle)[0]
    runs = example.pair_repeat(tmp_path / 'repeat', path, samples=2, observe_executable=True)
    observer = json.loads((tmp_path / 'repeat' / 'executable_observer.json').read_text())
    assert observer['completed'] and len(observer['compilations']) == 2
    assert observer['compilations'][0]['same_loaded_executable_as_first'] is False
    assert all(row['runtime_available'] and len(row['stablehlo_sha256']) == 64
               for row in observer['compilations'])
    assert 'not binary serialization' in observer['interpretation']
    for index, run in enumerate(runs, 1):
        folder = tmp_path / 'repeat' / f'execution_{index}'
        with np.load(path) as original, np.load(folder / 'initial_state.npz') as restored:
            assert original.files == restored.files
            for key in original.files:
                np.testing.assert_array_equal(original[key], restored[key])
        final = load_state(folder / 'final_state.npz', sim)
        np.testing.assert_allclose(final.ordinary.u / example.c, resumed.ordinary.u / example.c, atol=2e-14)
        np.testing.assert_allclose(final.work, resumed.work, rtol=2e-13)
        values, settings, results = [run[key] for key in ('history', 'settings', 'results')]
        assert values['t'].shape == (3,) and values['local_spread'].shape == (3, 2, 2)
        np.testing.assert_equal(values['work'][0], float(initial.work))
        np.testing.assert_equal(values['work'][-1], float(final.work))
        assert len(results['all_step_maxima_SI']) == len(results['all_step_maxima']) == 9
        assert settings['initial_state_archive_sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
        record = json.loads((folder / 'run.json').read_text())
        assert record['example'] == 'pair_waveform_repeat' and str(path) not in json.dumps(record)


@pytest.mark.parametrize('change', ['time', 'steps', 'work', 'fingerprint', 'runtime', 'coverage',
                                    'clock', 'blocks', 'boolean_blocks', 'samples'])
def test_archived_pair_repeat_rejects_incompatible_inputs(tmp_path, prescribed_pair_archive, change):
    import json
    from examples import dark_reservoir as example

    path, sim, start = prescribed_pair_archive
    if change in ('time', 'steps', 'work', 'coverage', 'fingerprint'):
        with np.load(path) as data:
            arrays = dict(data)
        if change == 'time':
            arrays['time'] = np.asarray(sim.plasma.domain.dt)
        elif change == 'steps':
            arrays['steps'] = np.asarray(.5)
        elif change == 'work':
            arrays['dark.work'] = np.asarray(1.)
        elif change == 'coverage':
            arrays['dark.times'][-1] -= sim.plasma.domain.dt / 4
        else:
            arrays['E'][0, 0] = np.nextafter(arrays['E'][0, 0], np.inf)
        np.savez_compressed(path, **arrays)
    else:
        record = json.loads((path.parent / 'run.json').read_text())
        if change == 'runtime':
            record['jax'] = 'unsupported'
        elif change == 'clock':
            record['settings']['scalar_dt_omega0'] = .03
        elif change in ('blocks', 'boolean_blocks'):
            record['settings']['block_steps'] = 2. if change == 'blocks' else True
        (path.parent / 'run.json').write_text(json.dumps(record))
    with pytest.raises(ValueError):
        example.pair_repeat(tmp_path / 'repeat', path, samples=0 if change == 'samples' else 1)


@pytest.mark.parametrize('scalar_dt', [None, .05])
def test_waveform_producer_keeps_shared_loading_and_native_restart_ledgers(
        tmp_path, scalar_dt, waveform_initial_archive):
    """Two native steps exercise tables, phase serialization and signed sector work."""
    import json
    from examples.dark_reservoir import pair_waveform_control

    initial = waveform_initial_archive[0] if scalar_dt else None
    curves, settings, results = pair_waveform_control(
        tmp_path, 512, 2, .025, .05, quadrature=4, table_dt=.025, block_horizon=.05, scalar_dt=scalar_dt,
        initial_state=initial)
    assert len(curves) == 5
    record = json.loads((tmp_path / 'run.json').read_text())
    assert record['settings']['pair_loading'] == 'global'
    assert settings['initial_state_source'] == ('complete zero-time archive' if initial else 'native initialization')
    if initial:
        import hashlib
        assert settings['initial_state_archive_sha256'] == hashlib.sha256(initial.read_bytes()).hexdigest()
        assert str(initial) not in json.dumps(record)
    assert all(r['table_refinement'] == 'identical cadence; no refinement' for r in results['forcing'].values())
    assert results['linear_reference']['interbranch_absolute_l2_difference'] > 0
    assert 'linear dynamics' in results['linear_reference']['qualification']
    assert all('realized_fine_vs_homogeneous_fine' in window['comparisons']
               for window in results['windows'].values())
    for label, values in curves.items():
        assert results['cases'][label]['initial_ordinary_fingerprints'] == settings['initial_ordinary_fingerprints']
        assert all(np.isfinite(value).all() for value in values.values())
        np.testing.assert_array_equal(values['source_work'], (-1 if label == 'coupled' else 1) * values['work'])
        branch = json.loads((tmp_path / label / 'run.json').read_text())
        assert branch['settings']['scalar_units'] == 'native SI'
        with np.load(tmp_path / label / 'data.npz') as native:
            np.testing.assert_allclose(native['t'] * settings['normalization']['omega0_rad_s'], values['t'],
                                       rtol=0, atol=1e-12)
            if label == 'coupled':
                if scalar_dt:
                    assert native['pump_t'].shape == (3,) and native['t'].shape == (2,)
                    assert native['pump_mean'].shape == (3, 2)
                np.testing.assert_equal(native['dark'][0] / settings['normalization']['energy_scale_J_m2'],
                                        settings['initial_dark_energy_over_energy_scale'])
        with np.load(tmp_path / label / 'final_state.npz') as restart:
            assert int(restart['steps']) == 2 and restart['u'].shape == (2048, 3)
            assert 'dark.work' in restart and 'dark.background' in restart
            assert ('dark.phi' in restart) == (label == 'coupled')
            if label != 'coupled':
                assert float(restart['dark.times'][0]) == 0
                assert float(restart['dark.times'][-1]) >= float(restart['time'])
    with np.load(tmp_path / 'data.npz') as saved:
        np.testing.assert_array_equal(saved['coupled_t'], curves['coupled']['t'][[0, -1]])
        assert saved['homogeneous_dark_fraction'].shape == saved['coupled_t'].shape == (2,)
        np.testing.assert_allclose(saved['additional_dark_depletion_fraction'],
                                   saved['homogeneous_dark_fraction'] - saved['coupled_total_dark_fraction'],
                                   atol=1e-16)
        np.testing.assert_array_equal(saved['additional_dark_depletion_gain'],
                                      saved['additional_dark_depletion_fraction'] -
                                      results['cases']['coupled']['initial_dark_preparation_offset_fraction'])
        assert saved['additional_dark_depletion_gain'][0] == 0
        assert saved['kinetic_excess_over_warm_background'][0] == 0
        assert saved['coupled_total_dark_fraction'][0] == 1
        np.testing.assert_array_equal(saved['linear_t'], saved['coupled_t'])
        for label in ('coupled', 'homogeneous_fine'):
            assert np.iscomplexobj(saved[f'linear_{label}_mode_E'])
            assert saved[f'linear_{label}_mode_E'].shape == saved['linear_t'].shape


def test_spatial_dark_force_changes_cold_linear_response_at_cubic_order():
    """Ampere gives E'''(0)=(1+η²)seed/2; removing dark force leaves seed/2."""
    time, eta, seed = np.array([0., .001, .002]), .5, 2e-4
    args = (time, 12., seed, eta, 1., 0., 0.)
    controls = dict(half_width=0., nodes=1, rtol=2e-13, atol=2e-15)
    coupled = relativistic_response(*args, **controls)[0]
    prescribed = relativistic_response(*args, spatial_eta=0., **controls)[0]
    np.testing.assert_allclose((coupled - prescribed)[-1].real,
                               eta**2 * seed * time[-1]**3 / 12, rtol=1e-3)
