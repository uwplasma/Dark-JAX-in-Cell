"""Independent momentum identities and reduced-ledger integration checks."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jaxincell import (Domain, Simulation, Solver, Species, elementary_charge as e,
                       epsilon_0, mass_electron as m, quiet_start, speed_of_light as c)

from darkjaxincell import DarkField, DarkSimulation, PrescribedDrive
from docs.scripts.conservation import measured_run, snapshot
from docs.scripts.benchmark_pic_conservation import translated_step


WP = 1e9
N = epsilon_0 * m * WP**2 / e**2


def neutral_box(algorithm="explicit", cells=16, relativistic=False):
    """A homogeneous neutral pair has no background momentum sink."""
    length = 2 * np.pi * c / WP
    x, v = quiet_start(64, length)
    species = (Species.electrons(64, N / 2).replace(x=x, v=v),
               Species("positrons", 64, 1, m, N / 2, x=x, v=v))
    return Simulation(Domain(length, cells, time_step=.01 / WP), species,
                      Solver(algorithm=algorithm, relativistic=relativistic))


def test_snapshot_selects_complex_ordinary_mode_without_a_dark_sector():
    plasma = neutral_box()
    state, _ = plasma.initial_state(jax.random.PRNGKey(0))
    theta = 2 * np.pi * np.arange(plasma.domain.cells) / plasma.domain.cells
    scale = m * c * WP / e
    field = scale * (.7 + .1 * np.cos(theta) - .2 * np.sin(theta)
                     + .4 * np.cos(3 * theta) + .3 * np.sin(3 * theta))
    state = state.replace(E=state.E.at[:, 0].set(field))
    values = snapshot(plasma, state, mode=3)
    # FFT/N of a*cos(kx)+b*sin(kx) is (a-i*b)/2 at positive k.
    np.testing.assert_allclose(values['mode_E'] / scale, .2 - .15j, rtol=2e-14)
    np.testing.assert_allclose(snapshot(plasma, state)['mode_E'] / scale,
                               .05 + .1j, rtol=2e-14)
    np.testing.assert_allclose(snapshot(plasma, state, mode=0)['mode_E'] / scale,
                               .7, rtol=2e-14)
    assert values['dark_coherent'] == values['dark_mode_E'] == 0


def test_snapshot_separates_homogeneous_proca_energy_and_complex_wave():
    plasma = neutral_box()
    omega, phase, scale = 1.7 * WP, .37, m * c * WP / e
    sim = DarkSimulation(plasma, DarkField(omega, .2))
    state, _ = sim.initial_state(jax.random.PRNGKey(0))
    theta = 2 * np.pi * np.arange(plasma.domain.cells) / plasma.domain.cells
    e0, b0, a0 = np.array([[.2, -.1, .3], [.04, .05, -.02], [-.07, .11, .13]])
    ew, bw, aw = np.array([[.4, .1, -.2], [.03, -.08, .06], [.09, .12, -.05]])
    phi0, phiw = .17, -.14
    wave = np.cos(3 * theta + phase)[:, None]
    # This manufactured diagnostic state need not satisfy the dynamical Gauss laws.
    state = state.replace(E=jnp.asarray(scale * (e0 + ew * wave)),
                          B=jnp.asarray(scale / c * (b0 + bw * wave)),
                          A=jnp.asarray(scale / WP * (a0 + aw * wave)),
                          phi=jnp.asarray(c * scale / WP * (phi0 + phiw * wave[:, 0])))
    ordinary = scale * (.11 + .3 * np.cos(3 * theta) - .4 * np.sin(3 * theta))
    state = state.replace(ordinary=state.ordinary.replace(
        E=state.ordinary.E.at[:, 0].set(ordinary)))
    values = snapshot(sim, state, mode=3)
    prefactor = epsilon_0 * plasma.domain.length * scale**2
    # Independent integrals: <cos>=0 and <cos^2>=1/2, including all potential terms.
    coherent = prefactor / 2 * (np.sum(e0**2 + b0**2)
                                + (omega / WP)**2 * (np.sum(a0**2) + phi0**2))
    spatial = prefactor / 4 * (np.sum(ew**2 + bw**2)
                               + (omega / WP)**2 * (np.sum(aw**2) + phiw**2))
    np.testing.assert_allclose(values['dark_coherent'], coherent, rtol=2e-14)
    np.testing.assert_allclose(values['dark'], coherent + spatial, rtol=2e-14)
    np.testing.assert_allclose(values['dark'] - values['dark_coherent'], spatial, rtol=2e-14)
    np.testing.assert_allclose(values['mode_E'] / scale, .15 + .2j, rtol=2e-14)
    expected = ew[0] * np.exp(1j * phase) / 2
    np.testing.assert_allclose(values['dark_mode_E'] / scale, expected, rtol=2e-14)
    negative = snapshot(sim, state, mode=plasma.domain.cells - 3)
    np.testing.assert_allclose(negative['dark_mode_E'] / scale, expected.conjugate(), rtol=2e-14)
    zero = snapshot(sim, state, mode=0)
    np.testing.assert_allclose(zero['dark_mode_E'] / scale, e0[0], rtol=2e-14)


@pytest.mark.parametrize("iterations", [0, 8])
def test_fractional_cell_force_matches_independent_accepted_orbits(iterations):
    for fraction in (0., .25, .5):
        row, p, initial, final = translated_step(16, 64, .004, iterations, fraction)
        assert row['charge_change_over_enL'] == 0
        assert row['continuity_over_enwp'] < 1e-11
        assert row['gauss_over_en_eps0'] < 1e-12
        assert row['impulse_reference_error'] < 1e-13
        assert row['max_orbit_error_over_c'] < 1e-13
        np.testing.assert_array_equal(initial.B, final.B)
        np.testing.assert_array_equal(final.u[:, 1:], 0)
        assert float(jnp.mean(initial.E[:, 0])) * e / (m * c * WP) < 1e-14
        physical = initial.x if iterations else initial.x - p.domain.dt * p._velocity(initial.u) / 2
        assert np.max(abs(np.asarray(physical[:, 0]))) <= p.domain.length / 2 + 1e-15
        if iterations:
            assert row['energy_defect_over_initial'] < 1e-13
            assert max(row['reference_orbit_closure']) < 2e-14
        else:
            assert abs(row['force_over_nmecLwp']) < 1e-13


@pytest.mark.parametrize("algorithm", ["explicit", "implicit"])
def test_neutral_homogeneous_charge_momentum_and_continuity(algorithm):
    sim = neutral_box(algorithm)
    state, _ = sim.initial_state(jax.random.PRNGKey(0))
    state = state.replace(E=state.E.at[:, 0].set(.01 * m * c * WP / e))
    final, history, maxima = measured_run(sim, state, 200, 10)
    history = jax.tree.map(np.asarray, history)
    scale = N * m * c**2 * sim.domain.length
    np.testing.assert_allclose(history["t"][0], 0)
    np.testing.assert_allclose(history["grid_charge"], history["charge"],
                               atol=1e-13 * e * N * sim.domain.length)
    assert float(maxima[1]) / (scale / c) < 1e-13
    assert float(maxima[3]) / (e * N * WP) < 1e-13
    assert float(maxima[4]) * epsilon_0 / (e * N) < 1e-13
    assert float(maxima[6]) / (e * N * sim.domain.length) < 1e-13
    expected = .01 * m * c * WP / e * np.cos(history["t"] * WP)
    np.testing.assert_allclose(history["mean_E"], expected, atol=5e-5 * abs(expected[0]))
    tolerance = 3e-5 if algorithm == "explicit" else 1e-12
    assert float(maxima[0]) / float(history["balance"][0]) < tolerance
    assert float(final.time) == pytest.approx(2 / WP)


@pytest.mark.parametrize("model", ["ordinary", "proca", "drive"])
def test_every_step_ledgers_and_true_initial_sample_ignore_output_stride(model):
    p = neutral_box(relativistic=True)
    field = jnp.tile(jnp.array([.02 * m * c * WP / e, 0., 0.]), (p.domain.cells, 1))
    sim = (p if model == "ordinary" else DarkSimulation(p, DarkField(WP, .3, initial_E=field))
           if model == "proca" else DarkSimulation(p, PrescribedDrive(.3, field[0], WP)))
    state, _ = sim.initial_state(jax.random.PRNGKey(0))
    if model == "ordinary":
        state = state.replace(E=field)
    dense, history, dense_max = measured_run(sim, state, 24, 1)
    sparse, reduced, sparse_max = measured_run(sim, state, 24, 6)
    jax.tree.map(lambda a, b: np.testing.assert_allclose(a, b, rtol=1e-14, atol=1e-30), dense, sparse)
    for key in history:
        np.testing.assert_allclose(np.asarray(history[key])[::6], reduced[key], rtol=1e-14, atol=1e-30)
    np.testing.assert_array_equal(dense_max, sparse_max)
    np.testing.assert_allclose(history["balance"][0], snapshot(sim, state)["balance"], rtol=1e-14)
    assert float(dense_max[1]) < 1e-25
    if model == "proca":
        assert float(dense_max[5]) * epsilon_0 / (e * N) < 1e-13
        np.testing.assert_allclose(dense_max[0], dense.max_balance_error,
                                   atol=1e-14 * float(history["balance"][0]))
        residual = history["dark"] - history["dark"][0] - history["work"]
        np.testing.assert_allclose(dense_max[7], np.max(abs(residual)),
                                   atol=1e-14 * float(history["balance"][0]))
    if model == "drive":
        np.testing.assert_allclose(history["balance"] + history["work"],
                                   history["electric"] + history["kinetic"].sum(axis=1), rtol=1e-14)


def test_longitudinal_proca_wave_carries_potential_momentum_without_magnetic_field():
    """A travelling massive longitudinal wave has P/U=k/omega in the continuum."""
    errors = []
    for cells in (32, 64, 128):
        p = neutral_box(cells=cells)
        k, omega, amplitude = WP / c, np.sqrt(2) * WP, 1e-3
        A = jnp.zeros((cells, 3)).at[:, 0].set(amplitude * jnp.cos(k * p.domain.faces))
        phi = c**2 * k / omega * amplitude * jnp.cos(k * p.domain.grid)
        sim = DarkSimulation(p, DarkField(WP, 0, initial_A=A, initial_phi=phi))
        state, _ = sim.initial_state(jax.random.PRNGKey(0))
        # Use continuum samples independently of the constructor's discrete Gauss projection.
        E = jnp.zeros_like(A).at[:, 0].set(-WP**2 / omega * amplitude * jnp.sin(k * p.domain.faces))
        state = state.replace(E=E)
        values = snapshot(sim, state)
        np.testing.assert_array_equal(state.B, jnp.zeros_like(A))
        exact_energy = epsilon_0 * p.domain.length * WP**2 * amplitude**2 / 2
        np.testing.assert_allclose(values["dark"], exact_energy, rtol=1e-14)
        exact_momentum = exact_energy * k / omega
        errors.append(abs(float(values["momentum"][0]) / exact_momentum - 1))
        np.testing.assert_allclose(values["momentum"][1:], 0, atol=1e-30)
    np.testing.assert_allclose(np.array(errors[:-1]) / errors[1:], 4, rtol=.01)
    assert errors[-1] < 4e-4


def test_reduced_runner_rejects_an_incomplete_tail():
    sim = neutral_box()
    state, _ = sim.initial_state(jax.random.PRNGKey(0))
    with pytest.raises(ValueError, match="divisible"):
        measured_run(sim, state, 10, 3)


def test_fixed_blocks_keep_global_work_reference_and_physical_moments():
    """Segmented diagnostics must not reset a conservation defect at each block."""
    plasma = neutral_box(relativistic=True)
    sim = DarkSimulation(plasma, PrescribedDrive(.3, jnp.array([.02 * m * c * WP / e, 0., 0.]), WP))
    state, _ = sim.initial_state(jax.random.PRNGKey(0))
    reference = snapshot(sim, state)
    scales = jnp.array([.1, .2]) * c / WP
    final, full, maximum = measured_run(sim, state, 24, 6, reference, scales)
    middle, first, peak1 = measured_run(sim, state, 12, 6, reference, scales)
    repeated, last, peak2 = measured_run(sim, middle, 12, 6, reference, scales)
    for key in full:
        joined = np.concatenate((first[key], last[key][1:]))
        np.testing.assert_allclose(joined, full[key], rtol=1e-13, atol=1e-30)
    np.testing.assert_allclose(np.maximum(peak1, peak2), maximum, rtol=1e-13, atol=1e-30)
    np.testing.assert_allclose(repeated.work, final.work, rtol=1e-13)
    np.testing.assert_allclose(last['t'][0], middle.ordinary.time)
    assert full['local_spread'].shape == full['local_density_rms'].shape == (5, 2, 2)


def test_mode_fit_keeps_physical_window_endpoints_with_accumulated_clock_error():
    from docs.scripts.benchmark_pic_conservation import fit_mode

    t = np.linspace(8 - 1e-12, 16 + 1e-12, 81)
    mode = np.exp((.34 - .02j) * t)
    assert np.count_nonzero((t >= 8) & (t <= 16)) == 79
    fit = fit_mode(t, mode)
    assert fit["fit_samples"] == 81
    np.testing.assert_allclose(fit["fitted_growth_over_wp"], .34, rtol=1e-13)
    np.testing.assert_allclose(fit["fitted_frequency_over_wp"], .02, rtol=1e-13)
