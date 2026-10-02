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


@pytest.mark.parametrize('study', ['pair_waveform', 'pair_repeat', 'pair_table', 'pair_dt', 'pair_force_extension'])
def test_waveform_named_input_dispatcher_passes_initial_archive(tmp_path, study):
    import ast
    from pathlib import Path
    from examples import dark_reservoir as example

    calls = []

    def stub(*args, **kwargs):
        calls.append((args, kwargs))

    namespace = {**vars(example), '__name__': '__main__', 'study': study,
                 'samples': 2, 'table_every': 2, 'pic_dt': .0125,
                 'initial_state': tmp_path / 'initial.npz', 'pair_waveform_control': stub, 'pair_repeat': stub,
                 'pair_force_extension': stub}
    # Execute the actual dispatch block with a producer stub, avoiding five runs.
    dispatch = ast.parse(Path(example.__file__).read_text()).body[-1]
    exec(compile(ast.Module(body=[dispatch], type_ignores=[]), example.__file__, 'exec'), namespace)
    assert len(calls) == 1
    assert calls[0][0][-1 if study == 'pair_waveform' else 1] == namespace['initial_state']
    if study in ('pair_repeat', 'pair_table', 'pair_dt'):
        assert calls[0][0][2:] == (2, False)
        assert calls[0][1] == ({'pair_table': dict(table_every=2), 'pair_dt': dict(
            pic_dt=.0125, force_extension=None, force_extension_transition=())}.get(study, {}))
    if study == 'pair_force_extension':
        assert calls[0][0][-1] == () and calls[0][1] == {}


@pytest.mark.parametrize('method', ['average', 'six_face'])
@pytest.mark.parametrize('clock', ['accumulated', 'anchored'])
def test_paper_optional_gather_dispatch_settings_and_restart(tmp_path, method, clock):
    """The named paper control reaches the native method and complete restart."""
    import ast
    from contextlib import nullcontext
    from pathlib import Path
    from darkjaxincell import load_state
    from examples import dark_reservoir as example

    calls = []
    namespace = {**vars(example), '__name__': '__main__', 'study': 'paper',
                 'longitudinal_gather': method, 'clock': clock, 'elapsed_progress': lambda *args: nullcontext(),
                 'paper_case': lambda *args, **kwargs: calls.append(kwargs)}
    dispatch = ast.parse(Path(example.__file__).read_text()).body[-1]
    exec(compile(ast.Module(body=[dispatch], type_ignores=[]), example.__file__, 'exec'), namespace)
    assert calls == [{'longitudinal_gather': method, 'clock': clock}]
    history, settings, _ = example.paper_case(tmp_path, 8, 32, .25, .5, 0, .03,
                                              shape_order=5, longitudinal_gather=method, clock=clock)
    assert settings.get('longitudinal_gather', 'average') == method
    assert ('longitudinal_gather' in settings) == (method != 'average')
    assert settings.get('clock', 'accumulated') == clock
    assert ('clock' in settings) == (clock != 'accumulated')
    plasma, wp = example.paper_plasma(8, 32, .25, 0, shape_order=5)
    field = .03 * np.sqrt(.001) * example.mass_electron * example.c * wp / example.e
    sim = example.DarkSimulation(plasma, example.PrescribedDrive(1., example.jnp.array([field, 0., 0.]), wp),
                                 longitudinal_gather=method, clock=clock)
    restored = load_state(tmp_path / 'final_state.npz', sim)
    assert int(restored.ordinary.steps) == 2 and history['t'][-1] == pytest.approx(.5)
    origin = load_state(tmp_path / 'initial_state.npz', sim)
    shifted = origin.replace(ordinary=origin.ordinary.replace(steps=example.jnp.asarray(1, example.jnp.int32)))
    if clock == 'anchored':
        shifted = shifted.replace(clock_step=shifted.ordinary.steps)
    path = tmp_path / 'nonzero_step.npz'
    example.save_compressed_state(path, shifted, sim)
    with pytest.raises(ValueError, match='zero clock origin'):
        example.paper_initial(sim, 0, path)
    if clock == 'anchored':
        shifted = shifted.replace(clock_time=-sim.plasma.domain.dt, clock_step=example.jnp.asarray(0))
        example.save_compressed_state(path, shifted, sim)
        with pytest.raises(ValueError, match='zero clock origin'):
            example.paper_initial(sim, 0, path)
    full, _, maximum, *_ = example.paper_run(sim, origin, 4, 2, .5, wp, None, tmp_path / 'whole')
    example.save_compressed_state(tmp_path / 'whole' / 'final_state.npz', full, sim)
    joined, continued_settings, result = example.paper_continue(
        tmp_path / 'continued', tmp_path / 'final_state.npz', 1.)
    assert continued_settings.get('longitudinal_gather', 'average') == method
    assert continued_settings.get('clock', 'accumulated') == clock
    np.testing.assert_array_equal(joined['t'][:len(history['t'])], history['t'])
    np.testing.assert_array_equal(result['all_step_maxima_SI'], np.maximum(
        np.asarray(maximum), result['reconstructed_prior_maxima_SI']))
    with np.load(tmp_path / 'whole' / 'final_state.npz') as whole, \
            np.load(tmp_path / 'continued' / 'final_state.npz') as continued:
        assert whole.files == continued.files
        for key in whole.files:
            np.testing.assert_array_equal(whole[key], continued[key])
    with pytest.raises(ValueError, match='cannot change longitudinal_gather'):
        example.paper_continue(tmp_path / 'wrong', tmp_path / 'final_state.npz', 1.,
                               longitudinal_gather='six_face' if method == 'average' else 'average')
    with pytest.raises(ValueError, match='cannot change clock'):
        example.paper_continue(tmp_path / 'wrong_clock', tmp_path / 'final_state.npz', 1.,
                               clock='anchored' if clock == 'accumulated' else 'accumulated')
    if method != 'average':
        with pytest.raises(ValueError, match='longitudinal_gather'):
            load_state(tmp_path / 'final_state.npz', sim.replace(longitudinal_gather='average'))
    calls.clear()
    namespace.update(study='paper_continue', paper_continue=lambda *args, **kwargs: calls.append(kwargs))
    exec(compile(ast.Module(body=[dispatch], type_ignores=[]), example.__file__, 'exec'), namespace)
    assert calls == [{'longitudinal_gather': method, 'clock': clock}]
    calls.clear()
    namespace.update(study='paper', longitudinal_gather=None, clock=None)
    exec(compile(ast.Module(body=[dispatch], type_ignores=[]), example.__file__, 'exec'), namespace)
    assert calls == [{'longitudinal_gather': 'average'}]
    namespace.update(study='pair', longitudinal_gather='six_face')
    with pytest.raises(ValueError, match='only by study=paper'):
        exec(compile(ast.Module(body=[dispatch], type_ignores=[]), example.__file__, 'exec'), namespace)
    namespace.update(longitudinal_gather=None, clock='anchored')
    with pytest.raises(ValueError, match='clock is supported only by study=paper'):
        exec(compile(ast.Module(body=[dispatch], type_ignores=[]), example.__file__, 'exec'), namespace)


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
        assert 'table_replay' not in settings
        assert values['t'].shape == (3,) and values['local_spread'].shape == (3, 2, 2)
        np.testing.assert_equal(values['work'][0], float(initial.work))
        np.testing.assert_equal(values['work'][-1], float(final.work))
        assert len(results['all_step_maxima_SI']) == len(results['all_step_maxima']) == 9
        assert settings['initial_state_archive_sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
        record = json.loads((folder / 'run.json').read_text())
        assert record['example'] == 'pair_waveform_repeat' and str(path) not in json.dumps(record)


def test_archived_pair_table_changes_only_force_knots_and_retains_complete_restart(
        tmp_path, prescribed_pair_archive, monkeypatch):
    import json
    from darkjaxincell import DarkSimulation, PrescribedDrive, load_state
    from docs.scripts import compare_replays as comparator
    from examples import dark_reservoir as example

    path, donor, initial = prescribed_pair_archive
    folder = tmp_path / 'table'
    run = example.pair_repeat(folder, path, table_every=2)[0]
    indices = [0, 1, 3]  # Initial endpoint, first midpoint, terminal endpoint.
    times, amplitude = np.asarray(donor.dark.times)[indices], np.asarray(donor.dark.amplitude)[indices]
    replay = DarkSimulation(donor.plasma, PrescribedDrive(.5, amplitude, 0., times=times))
    start = load_state(folder / 'initial_state.npz', replay)
    final = load_state(folder / 'final_state.npz', replay)
    with np.load(path) as original, np.load(folder / 'initial_state.npz') as restored:
        assert original.files == restored.files
        for key in original.files:
            np.testing.assert_array_equal(restored[key], {'dark.times': times, 'dark.amplitude': amplitude}.get(
                key, original[key]))
    np.testing.assert_array_equal(replay.dark.at(donor.plasma.domain.dt / 2), donor.dark.amplitude[1])
    expected_second = donor.dark.amplitude[1] + 2 / 3 * (donor.dark.amplitude[-1] - donor.dark.amplitude[1])
    np.testing.assert_allclose(replay.dark.at(1.5 * donor.plasma.domain.dt), expected_second, rtol=2e-15)
    known = replay._step(replay._step(start, replay.plasma.per_particle)[0], replay.plasma.per_particle)[0]
    np.testing.assert_allclose(final.ordinary.u / example.c, known.ordinary.u / example.c, atol=2e-14)
    np.testing.assert_allclose(final.work, known.work, rtol=2e-13)
    assert run['history']['t'].shape == (3,) and float(start.work) == float(initial.work)
    info = run['settings']['table_replay']
    assert info['table_every'] == 2 and (info['original_knots'], info['retained_knots']) == (4, 3)
    np.testing.assert_allclose(info['maximum_gap_omega0'], .0375, rtol=2e-15)
    for key, value in dict(original_times=donor.dark.times, original_amplitude=donor.dark.amplitude,
                           times=times, amplitude=amplitude).items():
        assert info[key + '_sha256'] == example.array_fingerprint(value)
    record = json.loads((folder / 'run.json').read_text())
    assert record['example'] == 'pair_waveform_table_replay' and str(path) not in json.dumps(record)
    assert 'not an exact-force repeat' in run['results']['claim']
    # Isolate the example-type guard from source validation of the dirty test checkout.
    monkeypatch.setattr(comparator, '_pair_source', lambda _: (record, run['history'], {}))
    with pytest.raises(ValueError, match='complete quadratic realized-force branches'):
        comparator.pair_repeat_comparison((folder, folder))


def test_archived_pair_pic_dt_preserves_force_integer_state_charge_energy_and_complete_restart(
        tmp_path, prescribed_pair_archive, monkeypatch):
    import json
    from darkjaxincell import load_state
    from docs.scripts import compare_replays as comparator
    from examples import dark_reservoir as example

    path, donor, initial = prescribed_pair_archive
    sim, start, setting = example.pair_repeat_initial(path)
    unchanged = example.pair_pic_timestep(sim, start, setting, None)
    assert unchanged[0] is sim and unchanged[1] is start
    changed, shifted = example.pair_pic_timestep(sim, start, setting, .0125)
    length, dx = float(sim.plasma.domain.length), float(sim.plasma.domain.dx)
    momentum = np.asarray(start.ordinary.u)
    gamma = np.sqrt(1 + np.sum((momentum / example.c)**2, axis=1))
    velocity = momentum / gamma[:, None]
    integer = (np.asarray(start.ordinary.x)[:, 0] - float(sim.plasma.domain.dt) * velocity[:, 0] / 2
               + length / 2) % length - length / 2
    recovered = (np.asarray(shifted.ordinary.x)[:, 0] - float(changed.plasma.domain.dt) * velocity[:, 0] / 2
                 + length / 2) % length - length / 2
    np.testing.assert_allclose(recovered / length, integer / length, rtol=0, atol=2e-13)
    # Independent cardinal S2 deposit at integer positions; stored rho is never replaced.
    r = (recovered - float(sim.plasma.domain.grid[0])) / dx
    nearest = np.floor(r + .5).astype(int)
    delta = r - nearest
    weights = np.stack((.5 * (.5 - delta)**2, .75 - delta**2, .5 * (.5 + delta)**2), axis=1)
    charge = np.concatenate([np.full(s.n, s.charge_si) for s in sim.plasma.species])
    rho = np.zeros(sim.plasma.domain.cells)
    np.add.at(rho, (nearest[:, None] + [-1, 0, 1]) % len(rho),
              charge[:, None] * np.asarray(start.ordinary.w)[:, None] * weights / dx)
    density = sim.plasma.species[0].density
    np.testing.assert_allclose(rho / (example.e * density), start.ordinary.rho / (example.e * density),
                               rtol=0, atol=1e-10)
    mass = np.concatenate([np.full(s.n, s.mass) for s in sim.plasma.species])
    kinetic = np.sum(mass * np.asarray(start.ordinary.w) * np.sum(momentum**2, axis=1) / (gamma + 1))
    total = kinetic + .5 * example.epsilon_0 * dx * np.sum(np.asarray(start.ordinary.E)**2)
    np.testing.assert_allclose(total, float(start.initial_ordinary), rtol=2e-13)
    folder = tmp_path / 'pic_dt'
    run = example.pair_repeat(folder, path, pic_dt=.0125)[0]
    with np.load(path) as original, np.load(folder / 'initial_state.npz') as restored:
        assert original.files == restored.files
        for key in original.files:
            expected = {'x': shifted.ordinary.x, 'dark.dt': changed.plasma.domain.dt}.get(key, original[key])
            np.testing.assert_array_equal(restored[key], expected)
    final = load_state(folder / 'final_state.npz', changed)
    assert int(final.ordinary.steps) == 4 and float(final.ordinary.time) == float(run['history']['t'][-1])
    assert float(final.work) == float(run['history']['work'][-1])
    with pytest.raises(ValueError, match='dark.dt'):
        load_state(folder / 'initial_state.npz', donor)
    info = run['settings']['pic_dt_replay']
    assert (info['donor_dt_omega0'], info['pic_dt_omega0'], info['block_horizon_omega0']) == (.025, .0125, .05)
    assert run['settings']['forcing_native_dt_omega0'] == .025 and run['settings']['block_steps'] == 4
    for key in ('times', 'amplitude'):
        assert info[key + '_sha256'] == example.array_fingerprint(getattr(donor.dark, key))
    record = json.loads((folder / 'run.json').read_text())
    assert record['example'] == 'pair_waveform_pic_dt_replay' and str(path) not in json.dumps(record)
    assert 'not an exact-state repeat' in run['results']['claim']
    monkeypatch.setattr(comparator, '_pair_source', lambda _: (record, run['history'], {}))
    with pytest.raises(ValueError, match='complete quadratic realized-force branches'):
        comparator.pair_repeat_comparison((folder, folder))


def test_pic_dt_restagger_crosses_periodic_wall_and_matches_independent_relativistic_work(prescribed_pair_archive):
    from examples import dark_reservoir as example

    path, _, _ = prescribed_pair_archive
    sim, state, setting = example.pair_repeat_initial(path)
    p, o, d = sim.plasma, state.ordinary, sim.plasma.domain
    mass, charge = (np.asarray(value) for value in p.per_particle)
    u0 = np.zeros_like(o.u)
    u0[:, 0] = np.where(charge < 0, .25, -.11) * example.c
    gamma0 = np.sqrt(1 + np.sum((u0 / example.c)**2, axis=1))
    v0 = u0 / gamma0[:, None]
    count = p.species[0].n
    # A uniform lattice shifted near the right wall includes genuine wrap crossings.
    integer = np.tile(np.asarray(p.species[0].x), (2, 1))
    integer[:, 0] = (integer[:, 0] + d.length / (2 * count) - .001 * example.c / 1e9
                     + d.length / 2) % d.length - d.length / 2
    old_x = integer.copy()
    old_x[:, 0] = (integer[:, 0] + d.dt * v0[:, 0] / 2 + d.length / 2) % d.length - d.length / 2
    field = setting['normalization']['field_scale_V_m']
    E0 = np.zeros_like(o.E)
    E0[:, 0] = .02 * field
    state = state.replace(ordinary=o.replace(x=example.jnp.asarray(old_x), u=example.jnp.asarray(u0),
                                             E=example.jnp.asarray(E0), rho=example.jnp.zeros_like(o.rho)))
    replay, start = example.pair_pic_timestep(sim, state, setting, .0125)
    dt = float(replay.plasma.domain.dt)
    expected_x = integer.copy()
    expected_x[:, 0] = (integer[:, 0] + dt * v0[:, 0] / 2 + d.length / 2) % d.length - d.length / 2
    np.testing.assert_allclose(start.ordinary.x / d.length, expected_x / d.length, rtol=0, atol=2e-13)
    assert np.any(abs(old_x[:, 0] - integer[:, 0]) > d.length / 2)
    # Independent uniform Ampere half-kick and the fixed table's midpoint value.
    current = np.sum(charge * np.asarray(o.w) * v0[:, 0]) / d.length
    Ehalf = .02 * field - dt / 2 * current / example.epsilon_0
    drive = np.interp(dt / 2, np.asarray(sim.dark.times), np.asarray(sim.dark.amplitude)[:, 0])
    u1 = u0.copy()
    u1[:, 0] += np.asarray(o.qm) * dt * (Ehalf + sim.dark.eta * drive)
    gamma1 = np.sqrt(1 + np.sum((u1 / example.c)**2, axis=1))
    mean_v = (u0[:, 0] + u1[:, 0]) / (gamma0 + gamma1)
    work = dt * np.sum(charge * np.asarray(o.w) * sim.dark.eta * drive * mean_v)
    accepted = replay._step(start, replay.plasma.per_particle)[0]
    np.testing.assert_allclose(accepted.ordinary.u / example.c, u1 / example.c, rtol=0, atol=2e-13)
    np.testing.assert_allclose(float(accepted.work), work, rtol=2e-13)
    energy = [np.sum(mass * np.asarray(o.w) * np.sum(u**2, axis=1) / (gamma + 1))
              for u, gamma in ((u0, gamma0), (u1, gamma1))]
    np.testing.assert_allclose(energy[1] - energy[0], work + dt * np.sum(charge * np.asarray(o.w) * mean_v * Ehalf),
                               rtol=2e-13)


@pytest.mark.parametrize('change', ['zero', 'boolean', 'equal', 'nan', 'table', 'samples', 'coverage',
                                    'B', 'E', 'u', 'drive', 'sigma', 'wall', 'moments', 'external_B'])
def test_archived_pair_pic_dt_rejects_incompatible_variations(tmp_path, prescribed_pair_archive, change):
    from darkjaxincell import DarkSimulation, PrescribedDrive
    from examples import dark_reservoir as example

    path, _, _ = prescribed_pair_archive
    dtau = {'zero': 0., 'boolean': True, 'equal': .025, 'nan': np.nan, 'coverage': .005}.get(change, .0125)
    if change in ('table', 'samples', 'coverage', 'zero', 'boolean', 'equal', 'nan'):
        with pytest.raises(ValueError):
            example.pair_repeat(tmp_path / 'bad', path, pic_dt=dtau,
                                table_every=2 if change == 'table' else 1, samples=2 if change == 'samples' else 1)
        return
    sim, state, setting = example.pair_repeat_initial(path)
    o = state.ordinary
    if change in ('B', 'E', 'u'):
        state = state.replace(ordinary=o.replace(**{change: getattr(o, change).at[0, 1].set(1.)}))
    elif change == 'drive':
        drive = PrescribedDrive(sim.dark.eta, sim.dark.amplitude.at[0, 1].set(1.), 0., times=sim.dark.times)
        sim = DarkSimulation(sim.plasma, drive)
    elif change == 'external_B':
        sim = DarkSimulation(sim.plasma.replace(external_B=example.jnp.ones_like(o.B)), sim.dark)
    else:
        value = {
            'sigma': example.jnp.ones_like(o.sigma),
            'wall': o.wall.replace(momentum=example.jnp.ones_like(o.wall.momentum)),
            'moments': example.jnp.zeros((2, 4, sim.plasma.domain.cells))}[change]
        state = state.replace(ordinary=o.replace(**{change: value}))
    with pytest.raises(ValueError, match='quadratic longitudinal'):
        example.pair_pic_timestep(sim, state, setting, dtau)


@pytest.mark.parametrize('change', ['zero', 'boolean', 'fraction', 'samples', 'coarse', 'midpoint'])
def test_archived_pair_table_rejects_invalid_stride_or_nonfine_native_force(
        tmp_path, prescribed_pair_archive, change):
    import json
    from examples import dark_reservoir as example

    path, _, _ = prescribed_pair_archive
    every = {'zero': 0, 'boolean': True, 'fraction': 1.5}.get(change, 2)
    if change == 'coarse':
        for record_path in (path.parent / 'run.json', path.parent.parent / 'run.json'):
            record = json.loads(record_path.read_text())
            record['settings']['label'] = 'realized_coarse'
            if 'cases' in record['results']:
                record['results']['cases']['realized_coarse'] = record['results']['cases'].pop('realized_fine')
            record_path.write_text(json.dumps(record))
    elif change == 'midpoint':
        with np.load(path) as archive:
            arrays = dict(archive)
        arrays['dark.times'][2] *= 1.05  # Still increasing/covered, but not a native fine force clock.
        np.savez_compressed(path, **arrays)
    with pytest.raises(ValueError):
        example.pair_repeat(tmp_path / 'table', path, samples=2 if change == 'samples' else 1, table_every=every)


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


def test_genuine_force_extension_keeps_complete_restart_prefix_current_work_and_global_bounds(tmp_path, monkeypatch):
    """A real small coupled step supplies the tail; NumPy reconstructs both continuity currents."""
    import json
    from examples import dark_reservoir as example
    from docs.scripts.compare_replays import PAIR_MAXIMA, _pair_extension, _same_leaves

    source, native_provenance, native_save = ['4' * 40], example.provenance, example.save_run
    monkeypatch.setattr(example, 'provenance', lambda **kwargs: dict(native_provenance(**kwargs), git=source[0]))

    def save(*args, **kwargs):
        native_save(*args, **kwargs)
        path = args[0] / 'run.json'
        record = json.loads(path.read_text())
        record['git'] = source[0]  # Synthetic clean provenance for this local fixture, never publication evidence.
        path.write_text(json.dumps(record))

    monkeypatch.setattr(example, 'save_run', save)
    original = tmp_path / 'original'
    example.pair_waveform_control(original, 512, 2, .025, .1, quadrature=4, table_dt=.05,
                                  output_dt=.025, block_horizon=.05, local_moments=True)
    fine, tail = original / 'realized_fine', tmp_path / 'tail'
    source[0] = '5' * 40
    transition = ('4' * 40, '5' * 40)
    values = example.pair_force_extension(tail, fine / 'initial_state.npz', transition)
    times, force, proof = _pair_extension(fine, tail, transition)
    with np.load(original / 'coupled' / 'final_state.npz') as old, np.load(tail / 'initial_state.npz') as stored:
        _same_leaves(dict(old), dict(stored))
        start = dict(stored)
    with np.load(tail / 'final_state.npz') as stored:
        end = dict(stored)
    with np.load(fine / 'initial_state.npz') as stored:
        np.testing.assert_array_equal(times[:-2], stored['dark.times'])
        np.testing.assert_array_equal(force[:-2], stored['dark.amplitude'])
    dt, length = float(start['dark.dt']), float(start['dark.length'])
    cells, count = int(start['cells']), int(start['counts'][0])
    dx, h, c, eps = length / cells, dt / 2, example.c, example.epsilon_0
    q = np.repeat(start['dark.charge'] * example.e, count) * start['w']
    position = (start['x'][:, 0] + length / 2) / dx - .5
    base = np.floor(position + .5).astype(int)
    rho_half = np.zeros(cells)
    for offset in (-1, 0, 1):
        r = abs(position - base - offset)
        weight = np.where(r < .5, .75 - r*r, np.where(r < 1.5, .5 * (1.5 - r)**2, 0.))
        np.add.at(rho_half, (base + offset) % cells, q * weight / dx)

    def current(before, after, state):
        residual = -(after - before) / h
        result = dx * np.cumsum(residual - residual.mean())
        velocity = state['u'][:, 0] / np.sqrt(1 + np.sum((state['u'] / c)**2, axis=1))
        return result - result.mean() + np.sum(q * velocity) / length

    j1, j2 = current(start['rho'], rho_half, start), current(rho_half, end['rho'], end)
    d0, a0, d1 = start['dark.E'][:, 0], start['dark.A'][:, 0], end['dark.E'][:, 0]
    omega, eta = float(start['dark.omega']), float(start['dark.eta'])
    middle = d0 + h * (omega**2 * a0 - eta * j1 / eps)
    field_scale = json.loads((fine / 'run.json').read_text())['settings']['normalization']['field_scale_V_m']
    np.testing.assert_allclose(force[-2, 0], middle.mean(), rtol=2e-12, atol=2e-14 * field_scale)
    work = -eta * h * dx * (np.sum(j1 * (d0 + middle) / 2) + np.sum(j2 * (middle + d1) / 2))
    np.testing.assert_allclose(float(end['dark.work'] - start['dark.work']), work, rtol=2e-10,
                               atol=2e-14 * float(start['dark.initial_dark']))
    assert end['time'] == start['time'] + dt and end['steps'] == start['steps'] + 1
    for key in ('background', 'initial_ordinary', 'initial_dark', 'initial_projection_norm'):
        np.testing.assert_array_equal(end['dark.' + key], start['dark.' + key])
    original_maxima = json.loads((original / 'coupled' / 'run.json').read_text())['results']['all_step_maxima']
    assert all(proof['native_run']['results']['all_step_maxima'][key] >= original_maxima[key] for key in PAIR_MAXIMA)
    results = proof['native_run']['results']
    checkpoint = [float(end['dark.' + key]) for key in ('max_balance_error', 'max_ordinary_gauss', 'max_dark_gauss')]
    np.testing.assert_array_equal(results['checkpoint_maxima_SI'], checkpoint)
    assert np.all(np.asarray(checkpoint) <= np.asarray(results['all_step_maxima_SI'])[[0, 4, 5]])
    assert values['balance'].shape == (2,) and len(proof['restored_native_leaves']) == len(start)
    replay = tmp_path / 'replay'
    example.pair_repeat(replay, fine / 'initial_state.npz', pic_dt=.0125, force_extension=tail,
                        force_extension_transition=transition)
    with np.load(replay / 'initial_state.npz') as restored:
        np.testing.assert_array_equal(restored['dark.times'], times)
        np.testing.assert_array_equal(restored['dark.amplitude'], force)
    with pytest.raises(ValueError, match='single PIC timestep'):
        example.pair_repeat(tmp_path / 'invalid', fine / 'initial_state.npz', force_extension=tail)
    with pytest.raises(ValueError, match='reviewed source'):
        example.pair_force_extension(tmp_path / 'invalid_source', fine / 'initial_state.npz')
    with pytest.raises(ValueError, match='reviewed source'):
        _pair_extension(fine, tail)


def test_pair_pic_no_extension_retains_original_accepted_clock_coverage_failure(
        tmp_path, monkeypatch, prescribed_pair_archive):
    """The historical repeated-addition deficit must fail before compilation, without a tail."""
    import jax.numpy as jnp
    from darkjaxincell import DarkSimulation, PrescribedDrive
    from examples import dark_reservoir as example
    from docs.scripts.compare_replays import _accepted_ticks

    path, donor, start = prescribed_pair_archive
    wp, dtau, steps = 1e9, .0015625, 108800
    ticks = _accepted_ticks(dtau / wp, steps, 1)
    times = np.r_[0., ticks[:-1] + dtau / (2 * wp), ticks[-1]]
    model = PrescribedDrive(.5, jnp.tile(donor.dark.amplitude[0], (len(times), 1)), 0., times=jnp.asarray(times))
    sim = DarkSimulation(donor.plasma.replace(domain=donor.plasma.domain.replace(time_step=dtau / wp)), model)
    _, _, setting = example.pair_repeat_initial(path)
    setting.update(dt_omega0=dtau, horizon_omega0=170., scalar_dt_omega0=.2, block_steps=6400)
    monkeypatch.setattr(example, 'pair_repeat_initial', lambda _: (sim, start, dict(setting)))

    def forbidden(*args, **kwargs):
        pytest.fail('uncovered PIC run reached compilation')

    monkeypatch.setattr(example, 'paper_run', forbidden)
    finer = _accepted_ticks(.00078125 / wp, 217600, 1)
    assert finer[-2] + .00078125 / (2 * wp) <= times[-1] < finer[-1]
    with pytest.raises(ValueError, match='cover the accepted native endpoint'):
        example.pair_repeat(tmp_path / 'uncovered', path, pic_dt=.00078125)
