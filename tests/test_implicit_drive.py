"""Work, continuity, homogeneous phase and AD checks for the validation prototype."""

import json
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jaxincell import Solver, epsilon_0
from jaxincell._core import deposit, E_x_from_rho

from darkjaxincell import PrescribedDrive, load_state
from darkjaxincell._proca import divergence
from docs.scripts.conservation import snapshot
from docs.scripts.drive_reference import forced_cold, homogeneous, midpoint_orbits, orbit_average
from examples.dark_reservoir import save_compressed_state
from docs.scripts.benchmark_implicit_drive import (
    FIELD, N, WP, c, m, e, drive_state, homogeneous_box, implicit_drive_step,
    objective, run_drive, sample, tangent_reference, load_plasma, norm_errors, crossings, archived_initial,
    translated_initial,
)


def cosine(amplitude=.02, phase=.3):
    return PrescribedDrive(1., jnp.array([amplitude * FIELD, 0., 0.]), WP, phase)


def assert_state_close(first, second, plasma):
    """Compare separate executions in physical units; GPU sums need not be bitwise equal."""
    length, energy = plasma.domain.length, N * m * c**2 * plasma.domain.length
    scales = dict(E=FIELD, B=FIELD / c, x=length, u=c, w=N * length, rho=e * N, time=1 / WP)

    def normalize(state):
        ordinary = state.ordinary.replace(**{key: getattr(state.ordinary, key) / scale
                                             for key, scale in scales.items()})
        return state.replace(ordinary=ordinary, work=state.work / energy,
                             initial_ordinary=state.initial_ordinary / energy,
                             max_balance_error=state.max_balance_error / energy,
                             max_ordinary_gauss=state.max_ordinary_gauss * epsilon_0 / (e * N))

    jax.tree.map(lambda a, b: np.testing.assert_allclose(a, b, rtol=2e-12, atol=2e-12),
                 normalize(first), normalize(second))


def test_nonzero_electric_resolves_a_small_wave_on_a_large_mean_field():
    p, state = homogeneous_box(cells=16, nodes=1, temperature=0.)
    wave = 1e-6 * FIELD * jnp.sin(2 * jnp.pi * p.domain.faces / p.domain.length)
    state = state.replace(ordinary=state.ordinary.replace(E=state.ordinary.E.at[:, 0].set(10 * FIELD + wave)))
    expected = epsilon_0 * p.domain.length * (1e-6 * FIELD)**2 / 4
    np.testing.assert_allclose(sample(p, state)["nonzero_electric"], expected, rtol=1e-8)


@pytest.mark.parametrize("iterations", [1, 8])
def test_shifted_field_work_identity_and_accepted_continuity_current(iterations):
    p, state = homogeneous_box(dtau=.1, iterations=iterations, nodes=4)
    o, d = state.ordinary, p.domain
    q = p.per_particle[1]
    # Break local neutrality to test the actual Gauss/current update, not just k=0.
    x = o.x.at[:, 0].add(jnp.where(q < 0, 1e-3 * d.dx * jnp.sin(2 * jnp.pi * o.x[:, 0] / d.length), 0.))
    rho = deposit(x[:, 0], q * o.w, d.grid[0], d.dx, d.cells, (0, 0))
    electric = jnp.zeros_like(o.E).at[:, 0].set(E_x_from_rho(rho, d.dx, (0, 0)))
    electric += FIELD * jnp.array([.08, .05, -.03])
    o = o.replace(x=x, rho=rho, E=electric,
                  B=jnp.ones_like(o.B) * jnp.array([.02, .03, .01]) * FIELD / c)
    state = drive_state(p, o)
    model = PrescribedDrive(.4, FIELD * jnp.array([.01, -.02, .015]), 1.1 * WP, .3)
    force = model.eta * model.amplitude * jnp.cos(model.omega * (o.time + d.dt / 2) + model.phase)
    shifted = o.replace(E=o.E + force)
    expected, output = p._implicit_step(shifted, p.per_particle)
    actual, current = implicit_drive_step(p, state, model)
    np.testing.assert_allclose(current, output[5], rtol=2e-12, atol=2e-14 * e * N * c)
    np.testing.assert_allclose(actual.ordinary.E, expected.E - force, rtol=2e-12, atol=2e-14 * FIELD)
    scale = N * m * c**2 * d.length
    parent_defect = snapshot(p, expected)["balance"] - snapshot(p, shifted)["balance"]
    physical_defect = snapshot(p, actual.ordinary)["balance"] - state.initial_ordinary - actual.work
    np.testing.assert_allclose(physical_defect, parent_defect, atol=1e-13 * scale, rtol=1e-12)
    residual = (actual.ordinary.rho - o.rho) / d.dt + divergence(current, d.dx)
    assert float(jnp.max(abs(residual))) / (e * N * WP) < 2e-12
    assert float(actual.max_ordinary_gauss) * epsilon_0 / (e * N) < 2e-12
    if iterations == 8:
        assert abs(float(physical_defect)) / scale < 1e-11
    else:
        assert abs(float(physical_defect)) / scale > 1e-10


def test_accepted_longitudinal_impulse_and_integer_cell_translation():
    """Check accepted particle impulse independently of mesh work conservation."""
    p, state = homogeneous_box(cells=16, nodes=4, iterations=8)
    p = p.replace(solver=p.solver.replace(substeps=1))
    o, d = state.ordinary, p.domain
    mass, charge = p.per_particle
    theta = 2 * jnp.pi * o.x[:, 0] / d.length
    # Break reflection symmetry, retaining two mobile species with zero total charge.
    x = o.x.at[:, 0].add(jnp.where(charge < 0, .04 * d.dx *
                                   (jnp.sin(theta + .37) + .3 * jnp.cos(2 * theta - .19)), 0.))
    rho = deposit(x[:, 0], charge * o.w, d.grid[0], d.dx, d.cells, (0, 0))
    o = o.replace(x=x, rho=rho, E=o.E.at[:, 0].set(E_x_from_rho(rho, d.dx, (0, 0))))
    state, model = drive_state(p, o), cosine()
    accepted, current = implicit_drive_step(p, state, model)
    after = accepted.ordinary
    # Within one linear-interpolation interval, the quadratic potential secant
    # equals the independent linear face-field interpolant at the orbit midpoint.
    endpoints = np.asarray(o.x[:, 0]), np.asarray(after.x[:, 0])
    np.testing.assert_array_equal(*(np.floor((x + d.length / 2) / d.dx) for x in endpoints))
    force = float(model.eta * model.amplitude[0]) * np.cos(
        float(model.omega) * (float(o.time) + d.dt / 2) + float(model.phase))
    midpoint_field = np.asarray((o.E[:, 0] + after.E[:, 0]) / 2) + force
    electric = np.interp((endpoints[0] + endpoints[1]) / 2, d.faces, midpoint_field, period=d.length)
    expected = np.asarray(charge * o.w) * d.dt * electric
    actual = np.asarray(mass * o.w * (after.u[:, 0] - o.u[:, 0]))
    scale = N * m * c * d.length
    np.testing.assert_allclose(actual / scale, expected / scale, rtol=0, atol=2e-12)
    momentum = snapshot(p, after)['momentum'] - snapshot(p, o)['momentum']
    np.testing.assert_allclose(momentum / scale, [expected.sum() / scale, 0., 0.], rtol=0, atol=2e-12)
    # An integer-cell translation is an exact grid permutation, not a claim of
    # continuous translation symmetry or exact continuum momentum conservation.
    shifted_x = o.x.at[:, 0].set((o.x[:, 0] + d.length / 2 + d.dx) % d.length - d.length / 2)
    shifted = o.replace(x=shifted_x, E=jnp.roll(o.E, 1, axis=0), rho=jnp.roll(o.rho, 1))
    translated, translated_current = implicit_drive_step(p, state.replace(ordinary=shifted), model)
    np.testing.assert_allclose(translated.ordinary.u / c, after.u / c, rtol=0, atol=2e-12)
    for key, units in (('E', FIELD), ('rho', e * N)):
        np.testing.assert_allclose(getattr(translated.ordinary, key) / units,
                                   jnp.roll(getattr(after, key), 1, axis=0) / units, rtol=0, atol=2e-12)
    np.testing.assert_allclose(translated_current / (e * N * c),
                               jnp.roll(current, 1, axis=0) / (e * N * c), rtol=0, atol=2e-12)


def test_zero_drive_is_parent_step_and_unsupported_solver_is_rejected():
    p, state = homogeneous_box(nodes=4)
    expected, _ = p._implicit_step(state.ordinary, p.per_particle)
    actual, _ = implicit_drive_step(p, state, cosine(0))
    assert_state_close(actual.replace(ordinary=expected), actual, p)
    assert float(actual.work) == 0.
    with pytest.raises(ValueError, match="implicit electromagnetic"):
        implicit_drive_step(p.replace(solver=Solver(algorithm="explicit")), state, cosine())
    with pytest.raises(ValueError, match="PrescribedDrive"):
        implicit_drive_step(p, state, None)


def test_cold_mean_response_refines_in_time_even_with_roundoff_energy_balance():
    errors = []
    for dt in (.04, .02):
        p, state = homogeneous_box(nodes=1, temperature=0., dtau=dt, relativistic=False)
        _, history, maxima = run_drive(p, state, cosine(.02, 0.), round(4 / dt))
        exact = forced_cold(np.asarray(history["t"]) * WP, .02)
        errors.append(np.max(abs(np.asarray(history["mean_E"]) / FIELD - exact)))
        assert float(maxima[0]) / (N * m * c**2 * p.domain.length) < 1e-12
    assert errors[0] < 1e-4
    assert 3.5 < errors[0] / errors[1] < 4.5


def test_relativistic_warm_response_and_all_step_constraints():
    p, state = homogeneous_box(nodes=8)
    _, history, maxima = run_drive(p, state, cosine(.04, 0.), 400, 10)
    t = np.asarray(history["t"]) * WP
    reference = homogeneous(t, .04, nodes=8, rtol=2e-11)
    np.testing.assert_allclose(np.asarray(history["mean_E"]) / FIELD, reference["mean_E"], atol=1e-4)
    np.testing.assert_allclose(np.asarray(history["mean"]) / c, reference["mean"], atol=1e-4)
    np.testing.assert_allclose(np.asarray(history["rms"]) / c, reference["rms"], atol=5e-6)
    scale = N * m * c**2 * p.domain.length
    assert float(maxima[0]) / scale < 1e-11
    assert float(maxima[1]) / (e * N * p.domain.length) < 1e-13
    assert float(maxima[2]) / (e * N * p.domain.length) < 1e-13
    assert float(maxima[3]) / (e * N * WP) < 2e-12
    assert float(maxima[4]) * epsilon_0 / (e * N) < 2e-12
    assert float(maxima[5]) / (FIELD * WP) < 2e-12
    assert float(maxima[6]) * c / scale < 1e-12


def test_stride_and_restart_preserve_absolute_phase_and_work_ledger(tmp_path):
    p, state = homogeneous_box(nodes=4)
    model = cosine()
    dense, history, maxima = run_drive(p, state, model, 64, 1)
    sparse, reduced, sparse_max = run_drive(p, state, model, 64, 8)
    first, _, _ = run_drive(p, state, model, 32, 8)
    metadata = SimpleNamespace(plasma=p, dark=model)
    save_compressed_state(tmp_path / 'midpoint.npz', first, metadata)
    restored = load_state(tmp_path / 'midpoint.npz', metadata)
    jax.tree.map(np.testing.assert_array_equal, first, restored)
    restarted, resumed, _ = run_drive(p, restored, model, 32, 8)
    for other in (sparse, restarted):
        assert_state_close(dense, other, p)
    energy = N * m * c**2 * p.domain.length
    scales = np.array([energy, e * N * p.domain.length, e * N * p.domain.length,
                       e * N * WP, e * N / epsilon_0, FIELD * WP, energy / c, energy / c])
    np.testing.assert_allclose(np.asarray(maxima) / scales, np.asarray(sparse_max) / scales, atol=2e-12)
    for key in history:
        scale = (1 / WP if key == 't' else c if key in ('mean', 'rms', 'max_speed') else
                 energy / c if key == 'momentum' else e * N * p.domain.length if key in ('charge', 'grid_charge') else
                 e * N / epsilon_0 if key in ('ordinary_gauss', 'dark_gauss') else
                 FIELD if key in ('mean_E', 'mode_E') else e * N * c if key == 'current' else energy)
        np.testing.assert_allclose(np.asarray(history[key])[::8] / scale,
                                   np.asarray(reduced[key]) / scale, rtol=2e-12, atol=2e-12)
        np.testing.assert_allclose(np.asarray(history[key])[32::8] / scale,
                                   np.asarray(resumed[key]) / scale, rtol=2e-12, atol=2e-12)
    assert float(history["t"][0]) == 0.
    assert float(resumed["t"][0]) == pytest.approx(32 * .02 / WP)


def test_drive_gradient_includes_density_weights_frequency_and_phase():
    controls = jnp.array([.025, 1., .94, .3])
    signal = jax.jit(lambda parameters: objective(parameters, 80, cells=4, nodes=4))
    value, automatic = jax.jit(jax.value_and_grad(signal))(controls)
    reference = tangent_reference(np.array([0., 1.6]), np.asarray(controls), nodes=4)
    np.testing.assert_allclose(float(value), .5 * reference["mean_E"][-1]**2, rtol=1e-3)
    np.testing.assert_allclose(automatic, reference["energy_gradient"], rtol=1e-3, atol=2e-9)
    finite = []
    for i in range(4):
        step = 1e-5 * max(abs(float(controls[i])), .1)
        perturbation = jnp.zeros(4).at[i].set(step)
        finite.append((float(signal(controls + perturbation)) - float(signal(controls - perturbation))) / (2 * step))
    np.testing.assert_allclose(automatic, finite, rtol=2e-5, atol=1e-10)


def test_thermal_initialization_remains_differentiable():
    controls = jnp.array([.025, 1., .94, .3])
    signal = jax.jit(lambda temperature: objective(controls, 40, cells=4, nodes=4, temperature=temperature))
    derivative = jax.jit(jax.grad(signal))(.001)
    finite = (float(signal(.001001)) - float(signal(.000999))) / .000002
    assert np.isfinite(derivative) and abs(float(derivative)) > 1e-8
    np.testing.assert_allclose(derivative, finite, rtol=3e-5, atol=1e-9)


def test_paper_fixture_reuses_physical_loading_at_the_implicit_position_clock():
    args = SimpleNamespace(paper_loading=True, cells=8, particles=32, dt=.02, iterations=4)
    plasma, initial = load_plasma(args)
    assert plasma.solver.algorithm == 'implicit' and plasma.solver.relativistic
    assert plasma.solver.picard_iterations == 4
    supplied = np.concatenate([np.asarray(species.x) for species in plasma.species])
    np.testing.assert_allclose(initial.ordinary.x / plasma.domain.length,
                               supplied / plasma.domain.length, atol=1e-15)
    assert plasma.species[0].n == plasma.species[1].n == 32
    np.testing.assert_allclose(np.asarray(snapshot(plasma, initial.ordinary)['rms']) / c,
                               np.sqrt(.001 / np.array([1, 1836])), rtol=2e-14)
    assert float(initial.ordinary.time) == float(initial.work) == 0.


def test_execution_comparisons_keep_zero_norms_and_native_sample_endpoints():
    time = np.arange(51.)
    metrics = norm_errors(time, np.ones(51), np.zeros(51))
    assert metrics['40']['samples'] == 41 and metrics['40']['final_time'] == 40.
    assert metrics['50']['relative_l2'] is None
    assert metrics['50']['difference_l2'] == pytest.approx(np.sqrt(51))
    json.dumps(metrics, allow_nan=False)
    values = np.r_[np.zeros(20), np.full(31, 1e-9)]
    assert crossings(time, values)['1e-12'] == 20.
    assert crossings(time, values)['0.0001'] is None
    vector = np.column_stack((np.ones(51), np.full(51, 3.)))
    reference = np.column_stack((np.zeros(51), np.ones(51)))
    metrics = norm_errors(time, vector, reference)['50']
    np.testing.assert_allclose(metrics['difference_l2'], np.sqrt(51) * np.array([1., 2.]))
    assert metrics['relative_l2'] == [None, 2.]
    assert metrics['max_abs_difference'] == [1., 2.]
    json.dumps(metrics, allow_nan=False)


def test_orbit_average_splits_linear_pieces_in_both_directions_and_at_the_wrap():
    field = np.array([2., 4., -2., 0.])  # faces -1, 0, 1, 2
    np.testing.assert_allclose(orbit_average(field, [-.75, -.25, .25, 1.75, -.25],
                                             [.5, .5, -.5, .5, 0.], 4.),
                               [3., 3.5, 3.5, 0., 3.5], rtol=0, atol=2e-15)
    with pytest.raises(ValueError, match='more than one face'):
        orbit_average(field, [-.25], [1.5], 4.)


def test_reference_constant_force_has_exact_relativistic_momentum_and_displacement():
    x, u = np.array([-.9, -.1, .99]), np.array([.05, -.1, .7])
    charge, mass = np.array([-1., 1., -1.]), np.array([1., 1836., 2.])
    result = midpoint_orbits(np.full(32, .2), x, u, charge, mass, 2., .1, drive=.03)
    exact_u = u + charge / mass * .1 * .23
    exact_shift = .1 * (u + exact_u) / (np.hypot(1., u) + np.hypot(1., exact_u))
    assert result['converged'] and max(result['residual_u']) < 2e-15
    np.testing.assert_allclose(result['u'], exact_u, rtol=0, atol=2e-15)
    np.testing.assert_allclose(result['displacement'], exact_shift, rtol=0, atol=2e-15)
    np.testing.assert_allclose(result['x'], (x + exact_shift + 1) % 2 - 1, rtol=0, atol=2e-15)


def test_reference_quadratic_endpoint_charge_and_unconverged_status():
    result = midpoint_orbits(np.zeros(4), [-.75], [0.], 1., 1., 2., .1, weights=[1.])
    np.testing.assert_array_equal(result['rho'], [3., .5, 0., .5])
    np.testing.assert_array_equal(result['rho_initial'], result['rho'])
    assert result['mean_current'] == 0. and result['converged']
    unresolved = midpoint_orbits([2., 4., -2., 0.], [.25], [.1], 1., 1., 4., .2,
                                 iterations=1, tolerance=1e-14)
    assert not unresolved['converged'] and max(unresolved['residual_u']) > 1e-6
    with pytest.raises(ValueError, match='1V arrays'):
        midpoint_orbits(np.zeros(4), [0.], [[0., 0., 0.]], 1., 1., 2., .1)
    with pytest.raises(ValueError, match='1V arrays'):
        midpoint_orbits(np.zeros(4), [0.], [0.], 1., 0., 2., .1)


def test_two_substep_parent_endpoint_matches_independent_frozen_midpoint_orbits():
    p, state = load_plasma(SimpleNamespace(paper_loading=True, cells=8, particles=32, dt=.02, iterations=8))
    model = cosine()
    accepted, current = implicit_drive_step(p, state, model)
    before, after, d = state.ordinary, accepted.ordinary, p.domain
    force = float(model.eta * model.amplitude[0] / FIELD) * np.cos(
        float(model.omega) * (float(before.time) + d.dt / 2) + float(model.phase))
    reference = midpoint_orbits((np.asarray(before.E[:, 0]) + np.asarray(after.E[:, 0])) / (2 * FIELD),
                                before.x[:, 0] * WP / c, before.u[:, 0] / c,
                                p.per_particle[1] / e, p.per_particle[0] / m, d.length * WP / c, d.dt * WP,
                                drive=force, weights=before.w / (N * d.length))
    assert reference['converged']
    np.testing.assert_allclose(after.u[:, 0] / c, reference['u'], rtol=0, atol=2e-12)
    np.testing.assert_allclose(after.x[:, 0] * WP / c, reference['x'], rtol=0, atol=2e-12)
    np.testing.assert_allclose(after.rho / (e * N), reference['rho'], rtol=0, atol=2e-12)
    np.testing.assert_allclose(np.mean(current[:, 0]) / (e * N * c), reference['mean_current'],
                               rtol=0, atol=2e-12)


def test_archived_initial_keeps_native_arrays_and_rejects_nonzero_or_mismatched_start(tmp_path):
    p, initial = homogeneous_box(cells=4, nodes=1, temperature=0.)
    model = SimpleNamespace(plasma=p, dark=cosine())
    stored = drive_state(p, initial.ordinary.replace(E=initial.ordinary.E.at[:, 0].set(1e-15 * FIELD)))
    path = tmp_path / 'initial.npz'
    save_compressed_state(path, stored, model)
    restored = archived_initial(path, initial, model)
    jax.tree.map(np.testing.assert_array_equal, stored, restored)
    assert archived_initial(None, initial, model) is initial
    for invalid in (stored.replace(ordinary=stored.ordinary.replace(time=jnp.asarray(p.domain.dt))),
                    stored.replace(work=jnp.asarray(1.))):
        save_compressed_state(path, invalid, model)
        with pytest.raises(ValueError, match='zero-time, zero-work'):
            archived_initial(path, initial, model)
    save_compressed_state(path, stored, model)
    mismatched = initial.replace(ordinary=initial.ordinary.replace(u=initial.ordinary.u.at[0, 0].add(.01 * c)))
    with pytest.raises(AssertionError):
        archived_initial(path, mismatched, model)


@pytest.mark.parametrize('phase', [0., .25, .5])
def test_grid_phase_recloses_charge_without_changing_velocity_weights_or_mean_current(phase):
    p, state = load_plasma(SimpleNamespace(paper_loading=True, cells=8, particles=32, dt=.02, iterations=8))
    d, old = p.domain, state.ordinary
    old = old.replace(x=old.x.at[:32, 0].add(.03 * d.dx), E=old.E.at[:, 0].add(.1 * FIELD))
    state = drive_state(p, old)
    shifted = translated_initial(p, state, phase)
    np.testing.assert_array_equal(shifted.background, state.background)
    if phase == 0:
        assert shifted is state
        return
    np.testing.assert_allclose((np.asarray(shifted.ordinary.x[:, 0] - old.x[:, 0]) + d.length / 2) % d.length
                               - d.length / 2, phase * d.dx, atol=2e-15 * d.length)
    for key in ('u', 'w', 'B', 'key', 'time'):
        np.testing.assert_array_equal(getattr(shifted.ordinary, key), getattr(old, key))
    rho = deposit(shifted.ordinary.x[:, 0], p.per_particle[1] * old.w, d.grid[0], d.dx, d.cells, (0, 0))
    np.testing.assert_array_equal(shifted.ordinary.rho, rho)
    np.testing.assert_allclose(jnp.mean(shifted.ordinary.E[:, 0]) / FIELD, .1, atol=2e-15)
    values = sample(p, shifted)
    assert float(values['ordinary_gauss']) * epsilon_0 / (e * N) < 2e-13
    assert float(shifted.initial_ordinary) == float(values['balance'])
    assert float(values['current']) == float(sample(p, state)['current'])
    accepted, _, maxima = run_drive(p, shifted, cosine(), 1)
    impulse = np.sum(np.asarray(p.per_particle[0] * old.w)[:, None]
                     * np.asarray(accepted.ordinary.u - old.u), axis=0)
    np.testing.assert_allclose(maxima[7], np.max(abs(impulse)), atol=2e-14 * N * m * c * d.length)


def test_grid_phase_rejects_invalid_phase_and_restart_clock():
    p, state = homogeneous_box(cells=4, nodes=1, temperature=0.)
    for phase in (-.1, 1., np.nan):
        with pytest.raises(ValueError, match='grid_phase'):
            translated_initial(p, state, phase)
    for invalid in (state.replace(work=jnp.array(1.)),
                    state.replace(ordinary=state.ordinary.replace(time=jnp.array(p.domain.dt)))):
        with pytest.raises(ValueError, match='zero-time, zero-work'):
            translated_initial(p, invalid, .25)
    with pytest.raises(ValueError, match='neutralizing charge'):
        translated_initial(p, state.replace(background=jnp.asarray(e * N)), .25)
