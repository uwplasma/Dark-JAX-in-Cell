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


@pytest.mark.parametrize('scalar_dt', [None, .05])
def test_waveform_producer_keeps_shared_loading_and_native_restart_ledgers(tmp_path, scalar_dt):
    """Two native steps exercise tables, phase serialization and signed sector work."""
    import json
    from examples.dark_reservoir import pair_waveform_control

    curves, settings, results = pair_waveform_control(
        tmp_path, 512, 2, .025, .05, quadrature=4, table_dt=.025, block_horizon=.05, scalar_dt=scalar_dt)
    assert len(curves) == 5
    record = json.loads((tmp_path / 'run.json').read_text())
    assert record['settings']['pair_loading'] == 'global'
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
