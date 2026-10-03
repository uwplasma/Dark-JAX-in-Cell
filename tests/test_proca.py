"""Independent field identities and parent-integration regressions."""

import json
import runpy
import numpy as np
import itertools
import pytest
import jax
import jax.numpy as jnp
from scipy.linalg import expm
from scipy.integrate import trapezoid
from scipy.special import wofz

from jaxincell import (Domain, Simulation, Species, elementary_charge as e,
                       epsilon_0, mass_electron, quiet_start, speed_of_light as c, Solver)
from jaxincell._core import curl_E
from jaxincell.theory import landau_root
from darkjaxincell import (DarkField, DarkSimulation, PrescribedDrive, gauss,
                           load_for_continuation, load_state, load_toml, main,
                           midnight, project_initial_electric, save_state)
from darkjaxincell._proca import divergence, drift, gradient, kick
from examples.optimize_dark_photon import build_objective, cold_reference
from examples.dark_kinetic import fixed_window_mode, longitudinal_root, mixed_root
from examples.dark_bump import (build_plasma, bump_reference_scan, kinetic_root,
                                number_histogram)
from examples.dark_saturation import integer_positions, trapping_frequency
from examples.dark_instabilities import (cold_screened_growth, two_stream_growth,
                                         warm_two_stream_reference, weibel_cutoff_scan,
                                         weibel_cutoff_squared, weibel_growth)
from examples.dark_profile import (build_design, cold_scattering, packet, profile,
                                   slab_basis, transmitted_fraction, make_simulation)
from docs.scripts.benchmark_time_integrators import (
    resonant_exact, resonant_map, resonant_modes, system as vacuum_system)
from docs.scripts.make_movies import _marker_indices


OMEGA = 1e9
N_REF = epsilon_0 * mass_electron * OMEGA**2 / e**2


def test_bump_loading_and_independent_kinetic_limit():
    coarse, wp, vth = build_plasma(64, 512, 256)
    fine, wp_fine, vth_fine = build_plasma(128, 1024, 512, 0.0125)
    assert wp == wp_fine and vth == vth_fine
    assert fine.domain.dx == coarse.domain.dx / 2
    state, (_, charge) = coarse.initial_state(jax.random.PRNGKey(0))
    velocity = coarse._velocity(state.u)[:, 0]
    current = np.sum(np.asarray(charge * state.w * velocity))
    scale = np.sum(np.abs(np.asarray(charge * state.w * velocity)))
    assert abs(current) / scale < 1e-12
    parent, residual0 = kinetic_root(wp, vth, 0)
    mixed, residual = kinetic_root(wp, vth, 0.3)
    np.testing.assert_allclose([parent.real, parent.imag], [0.848257, 0.151347], atol=2e-6)
    assert mixed.imag > parent.imag and mixed.real > parent.real
    assert max(residual0, residual) < 1e-7


def test_bump_selected_pole_crosses_threshold_with_screening():
    wp = 0.05 * c * 128
    scan = bump_reference_scan(wp, wp / (5 * 2 * np.pi * 5))
    ordinary = scan["roots"]["ordinary"][:, 1, 1]
    full = scan["roots"]["full"][:, 1, 1]
    screened = scan["roots"]["quasistatic"][:, 1, 1]
    assert ordinary[0].imag > 0 > full[0].imag
    assert ordinary[3].imag > 0 and full[3].imag > ordinary[3].imag
    assert np.max(np.abs(full - screened)) < 1e-3
    np.testing.assert_allclose(full[3].imag, 0.15392197436, rtol=1e-8)


def test_shared_multispecies_longitudinal_reference_and_screening():
    K = 2 * np.pi / (0.05 * 64)
    k, wp = 2 * np.pi, 0.05 * c * 64
    populations = ({"wp": wp / np.sqrt(2), "u": 0.25 * c, "vth": 0.03 * c},
                   {"wp": wp / np.sqrt(2), "u": -0.25 * c, "vth": 0.03 * c})
    root, residual = longitudinal_root(k, populations, 0.7 * wp, 0.3,
                                       0.02 + 0.34j)
    assert root.imag > 0 and residual < 1e-8
    sigma = 0.03 * c / np.sqrt(2)
    chi = 0j
    for species in populations:
        zeta = (root * wp - k * species["u"]) / (np.sqrt(2) * k * sigma)
        chi += (species["wp"] / (k * sigma)) ** 2 * (
            1 + zeta * 1j * np.sqrt(np.pi) * wofz(zeta))
    s = root**2 - K**2
    assert abs((s - 0.7**2) * (1 + chi) + 0.3**2 * s * chi) < 1e-8
    ordinary, _ = longitudinal_root(k, populations, 0.7 * wp, 0,
                                    0.02 + 0.34j, model="ordinary")
    quasi, _ = longitudinal_root(k, populations, 0.7 * wp, 0.3,
                                 root, model="quasistatic")
    effective, _ = longitudinal_root(k, populations, 0.7 * wp, 0.3,
                                     root, model="effective_charge")
    assert ordinary.imag < quasi.imag < effective.imag
    assert abs(root.imag - quasi.imag) < abs(root.imag - ordinary.imag)


def test_physical_landau_scale_and_fixed_window_fit():
    """The nonrelativistic pole and a damped standing wave use compatible units."""
    ordinary, _ = mixed_root(10, 0.05, 0, 10)
    mixed, residual = mixed_root(10, 0.05, 0.3, 10)
    np.testing.assert_allclose([ordinary.real, ordinary.imag],
                               [1.4156618886, -0.1533594669], rtol=1e-8)
    np.testing.assert_allclose([mixed.real, mixed.imag],
                               [1.4324272553, -0.1457983642], rtol=1e-8)
    assert residual < 1e-8
    time = np.linspace(0, 16, 3201)
    signal = np.abs(np.exp(mixed.imag * time) * np.cos(mixed.real * time))
    damping, frequency, peaks, _, _ = fixed_window_mode(time, signal)
    np.testing.assert_allclose([frequency, damping], [mixed.real, mixed.imag], rtol=2e-3)
    assert len(peaks) >= 4


def test_physical_landau_driver_matches_its_recorded_reference(tmp_path):
    experiment = runpy.run_path("examples/dark_kinetic.py", run_name="__main__",
                                init_globals=dict(physical=True, cells=16, particles=128,
                                                  steps=4, output=tmp_path))
    model, wp, k = (experiment[key] for key in ("model", "wp", "k"))
    assert model.omega == pytest.approx(k * c)
    record = json.loads((tmp_path / "run.json").read_text())
    assert record["settings"]["mu_over_wp"] == pytest.approx(model.omega / wp)
    expected, residual = mixed_root(k * c / wp, .05, model.eta, model.omega / wp)
    assert residual < 1e-8
    np.testing.assert_allclose(experiment["analytic"], expected, rtol=1e-12)


def test_warm_two_stream_growing_and_single_humped_control():
    growing = warm_two_stream_reference(0.01)
    assert (growing["ordinary"].imag < growing["quasistatic"].imag
            < growing["full"].imag < growing["effective_charge"].imag)
    np.testing.assert_allclose(growing["full"].imag, 0.33643063655, rtol=1e-8)
    stable = warm_two_stream_reference(0.06)
    assert all(root.imag < 0 for root in stable.values())
    velocity = np.linspace(0.001, 0.3, 500)  # v/c > 0
    sigma, drift = 0.06, 0.05
    derivative = sum(-(velocity - sign * drift) / sigma**2
                     * np.exp(-0.5 * ((velocity - sign * drift) / sigma)**2)
                     for sign in (1, -1))
    assert np.all(derivative < 0)  # one hump, not a beam free-energy pocket


def test_cold_screening_explains_the_legacy_two_stream_shift():
    K = 2 * np.pi / (0.05 * 64)
    b, mu, eta = 0.25 * K, 0.7, 0.3
    ordinary = cold_screened_growth(K, b, mu, eta, "ordinary")
    quasistatic = cold_screened_growth(K, b, mu, eta, "quasistatic")
    full = two_stream_growth(K, b, mu, eta)[0]
    np.testing.assert_allclose([ordinary, quasistatic, full],
                               [0.3384711930, 0.3466709103, 0.3466973106], rtol=2e-9)
    np.testing.assert_allclose((quasistatic - ordinary) / (full - ordinary),
                               0.9967906678, rtol=2e-9)
    np.testing.assert_allclose(cold_screened_growth(K, b, 1e8, eta, "quasistatic"),
                               ordinary, rtol=1e-13)
    np.testing.assert_allclose(cold_screened_growth(K, b, 0, eta, "quasistatic"),
                               cold_screened_growth(K, b, 0, eta, "effective_charge"),
                               rtol=1e-13)
    assert cold_screened_growth(K, 1.01 * np.sqrt(1 + eta**2),
                                0, eta, "effective_charge") == 0


def test_retarded_longitudinal_identity_refines_against_pic():
    """The same nonlinear current history obeys the independent memory kernel."""
    omega, eta = OMEGA, 0.3
    wp = np.sqrt(0.2) * omega
    length = 2 * np.pi * c / omega
    k = 2 * np.pi / length
    positions, velocities = quiet_start(256, length)
    positions = positions.at[:, 0].add(0.002 * jnp.sin(k * positions[:, 0]) / k)
    electrons = Species.electrons(256, 0.2 * N_REF).replace(x=positions, v=velocities)
    errors = []
    for dt_wp in (0.04, 0.02):
        domain = Domain(length, 32, time_step=dt_wp / wp)
        base = Simulation(domain, (electrons,))
        sim = DarkSimulation(base, DarkField(omega, eta))
        initial, _ = sim.initial_state(jax.random.PRNGKey(0))
        output = sim.run(round(4 / dt_wp), store_particles=False)
        electric = np.r_[np.fft.rfft(np.asarray(initial.ordinary.E[:, 0]))[1] / 32,
                         np.fft.rfft(np.asarray(output.ordinary.E[:, :, 0]), axis=1)[:, 1] / 32]
        dark = np.r_[np.fft.rfft(np.asarray(initial.E[:, 0]))[1] / 32,
                     np.fft.rfft(np.asarray(output.E[:, :, 0]), axis=1)[:, 1] / 32]
        time = np.r_[0, np.asarray(output.ordinary.t)]
        khat = 2 * np.sin(k * domain.dx / 2) / domain.dx
        frequency = np.sqrt(c**2 * khat**2 + omega**2)
        memory = np.array([
            -eta * omega**2 * trapezoid(
                np.sin(frequency * (instant - time[:i + 1])) / frequency
                * electric[:i + 1], time[:i + 1])
            for i, instant in enumerate(time)])
        response = dark - eta * electric
        errors.append(float(np.max(np.abs(response - memory))
                            / np.max(np.abs(response))))
    assert errors[1] < 0.002
    assert 3.5 < errors[0] / errors[1] < 4.5


def test_weibel_marginal_screening_limits():
    Q, eta = 0.2, 0.3
    np.testing.assert_allclose(weibel_cutoff_squared(Q, 0.7, 0), Q, rtol=1e-14)
    np.testing.assert_allclose(weibel_cutoff_squared(Q, 1e8, eta), Q, rtol=1e-14)
    np.testing.assert_allclose(weibel_cutoff_squared(Q, 0, eta),
                               (1 + eta**2) * Q, rtol=1e-14)
    scan = weibel_cutoff_scan()
    cutoff = np.interp(0.3, scan["mixings"], scan["cutoff_mu_0.7"])
    np.testing.assert_allclose(cutoff, 1.79846552539, rtol=1e-10)
    assert 1 < cutoff < 2
    with pytest.raises(ValueError):
        weibel_growth(2, 0.08, 4, 0.7, 0.3)


def test_bump_histogram_represents_number_not_marker_count():
    velocity = np.array([-0.2, -0.1, 0.1, 0.2, 1.0, 1.1])
    weight = np.array([0.97 / 4] * 4 + [0.03 / 2] * 2)
    edges = np.linspace(-0.5, 1.5, 41)
    total, pieces, outside = number_histogram(velocity, weight, (4, 2), edges, 1.0)
    np.testing.assert_allclose(np.sum(pieces, axis=0), total)
    np.testing.assert_allclose(np.sum(pieces * np.diff(edges), axis=1), [0.97, 0.03])
    assert outside == 0
    split_v = np.concatenate((velocity[:4], np.repeat(velocity[4:], 2)))
    split_w = np.concatenate((weight[:4], np.repeat(weight[4:] / 2, 2)))
    split, _, _ = number_histogram(split_v, split_w, (4, 4), edges, 1.0)
    np.testing.assert_allclose(split, total)
    centres = (edges[:-1] + edges[1:]) / 2
    assert abs(np.sum(centres * total * np.diff(edges)) - np.sum(velocity * weight)) < 0.025
    bump, _, _ = build_plasma(32, 8000, 4000)
    selected = _marker_indices(bump)
    assert len(selected) == 8192
    assert np.count_nonzero(selected >= 8000) == 246


def test_effective_trapping_uses_complex_field_sum():
    k = 2 * np.pi
    ordinary = np.array([1 + 0j, 2 + 0j])
    np.testing.assert_allclose(trapping_frequency(ordinary, -ordinary / 0.3, 0.3, k), 0)
    shifted = trapping_frequency(np.array([1 + 0j]), np.array([1j / 0.3]), 0.3, k)
    reference = trapping_frequency(np.array([2**0.5 + 0j]), 0, 0, k)
    np.testing.assert_allclose(shifted, reference)


def test_stored_phase_positions_match_integer_time_not_half_step():
    base = plasma(N_REF * 0.2)
    parent = base.run(3)
    dark = DarkSimulation(base, DarkField(OMEGA, 0.1)).run(3)
    for output, state in ((parent, parent.state), (dark.ordinary, dark.state.ordinary)):
        last = np.asarray(output.x[-1, :, 0])
        half = np.asarray(state.x[:, 0])
        velocity = np.asarray(base._velocity(state.u)[:, 0])
        delta = (half - last + base.domain.length / 2) % base.domain.length - base.domain.length / 2
        np.testing.assert_allclose(delta, base.domain.dt * velocity / 2,
                                   rtol=2e-13, atol=1e-12)
        np.testing.assert_allclose(integer_positions(base, state), output.x[-1],
                                   rtol=2e-13, atol=1e-12)


def plasma(density=0.0, n=16, cells=16, external_E=None, external_B=None):
    domain = Domain(length=2 * np.pi * c / OMEGA, cells=cells, dt_over_dx_c=0.2)
    return Simulation(domain, (Species.electrons(n, density=density),),
                      external_E=external_E, external_B=external_B)


@pytest.mark.parametrize("drive", [False, True])
@pytest.mark.parametrize("particles", [False, True])
def test_host_progress_preserves_complete_history_and_traced_gradient(drive, particles):
    base = plasma(N_REF * .1)
    model = PrescribedDrive(.1, jnp.array([1e-5, 0., 0.]), OMEGA) if drive else DarkField(OMEGA, .1)
    sim = DarkSimulation(base, model)
    updates = []
    plain = sim.run(24, store_every=2, store_particles=particles)
    grouped = sim.run(24, store_every=2, store_particles=particles,
                      verbose=lambda done, total: updates.append((done, total)))
    assert updates[-1] == (24, 24) and len(updates) > 1
    jax.tree.map(lambda a, b: np.testing.assert_array_equal(a, b), plain, grouped)

    def objective(eta, verbose):
        output = DarkSimulation(base, DarkField(OMEGA, eta)).run(4, store_particles=False, verbose=verbose)
        return output.energy()["total_with_dark"][-1]

    derivative = jax.grad(objective)(.1, False)
    np.testing.assert_array_equal(jax.grad(objective)(.1, updates.append), derivative)
    assert updates[-1] == (24, 24)  # traced execution never calls the host meter
    with pytest.raises(ValueError, match="verbose"):
        sim.run(2, verbose="sometimes")


def test_dark_toml_and_cli_save_complete_restart(tmp_path):
    source = tmp_path / "input.toml"
    source.write_text("""[domain]
length = 1.0
cells = 16
dt_over_dx_c = 0.2
[[species]]
name = "electrons"
n = 16
mass = "electron"
charge = -1
density = 1e14
[run]
steps = 4
store_every = 2
[dark]
omega = 1e9
eta = 0.05
initial_E = [0.0, 1e-5, 0.0]
initial_A = [0.0, 0.0, 0.0]
initial_phi = 0.0
""")
    sim, run = load_toml(source)
    assert run["steps"] == 4 and isinstance(sim.dark, DarkField)
    assert np.asarray(sim.dark.initial_E).shape == (16, 3)
    folder = tmp_path / "saved"
    assert main([str(source), "--steps", "4", "--seed", "2", "--eta", "0.04",
                 "--omega", "1e9", "--save", str(folder)]) == 0
    assert (folder / "run.json").exists() and (folder / "restart.npz").exists()
    assert (folder / "input.toml").read_text() == source.read_text()
    result = json.loads((folder / "run.json").read_text())["results"]
    assert result["ordinary_gauss_max_V_m2"] < 1e-6
    assert result["dark_gauss_max_V_m2"] < 1e-6
    sim, _ = load_toml(source, eta=0.04, omega=1e9)
    loaded = load_state(folder / "restart.npz", sim)
    assert int(loaded.ordinary.steps) == 4
    assert main([str(source)]) == 0


def test_dark_toml_drive_and_errors(tmp_path):
    source = tmp_path / "drive.toml"
    prefix = """[domain]
length = 1.0
cells = 16
dt_over_dx_c = 0.2
[[species]]
name = "electrons"
n = 16
mass = "electron"
charge = -1
density = 1e14
[run]
steps = 2
store_particles = false
"""

    def check(section, error=None):
        source.write_text(prefix + section)
        if error:
            with pytest.raises(ValueError, match=error):
                load_toml(source)
        else:
            return load_toml(source)

    check("", "final \\[dark\\] table")
    check("[dark]\neta = 0.1\nomega = 1e9\n[run]\nsteps = 2\n", "final table")
    check("[dark]\neta = 0.1\nomega = 1e9\nspook = 1\n", "unknown")
    source.write_text(prefix.replace("store_particles = false", "verbose = true")
                      + "[dark]\neta = 0.1\nomega = 1e9\n")
    with pytest.raises(ValueError, match="unsupported dark \\[run\\] keys"):
        load_toml(source)
    check("[dark]\nomega = 1e9\n", "eta and omega")
    check("[dark]\neta = 0.1\nomega = 1e9\namplitude = [1.0, 0.0, 0.0]\n", "belong")
    check("[dark]\nmodel = 'drive'\neta = 0.1\nomega = 1e9\ninitial_E = [0.0, 0.0, 0.0]\n", "belong")
    check("[dark]\nmodel = 'drive'\neta = 0.1\nomega = 1e9\n", "needs amplitude")
    check("[dark]\nmodel = 'other'\neta = 0.1\nomega = 1e9\n", "field.*drive")
    sim, run = check("[dark]\nmodel = 'drive'\neta = 0.1\nomega = 1e9\n"
                     "amplitude = [1e-5, 0.0, 0.0]\nphase = 0.2\n")
    assert isinstance(sim.dark, PrescribedDrive) and run["store_particles"] is False
    assert main([str(source), "--save", str(tmp_path / "driven")]) == 0
    result = json.loads((tmp_path / "driven/run.json").read_text())["results"]
    assert result["dark_gauss_max_V_m2"] is None
    assert np.isfinite(result["energy_drift"])
    assert result["max_balance_error_J_m2"] >= 0
    check("[dark]\neta = 0.1\nomega = 1e9\ntimes = [0.0, 1e-9]\n", "belong")
    sim, run = check("[dark]\nmodel = 'drive'\neta = 0.1\nomega = 0.0\n"
                     "times = [0.0, 1e-9]\namplitude = [[1e-5, 0.0, 0.0], [0.0, 2e-5, 0.0]]\n")
    np.testing.assert_allclose(sim.dark.at(.25e-9), [7.5e-6, 5e-6, 0.], atol=1e-20, rtol=0)
    folder = tmp_path / "waveform"
    assert main([str(source), "--save", str(folder)]) == 0
    state = load_state(folder / "restart.npz", sim)
    reference = sim.run(**run).state
    np.testing.assert_allclose(state.ordinary.E, reference.ordinary.E, rtol=2e-13, atol=1e-18)
    np.testing.assert_allclose(state.work, reference.work, rtol=2e-13, atol=1e-25)
    with np.load(folder / "restart.npz", allow_pickle=False) as data:
        assert data["dark.format"] == 3
        np.testing.assert_array_equal(data["dark.times"], sim.dark.times)


def test_compatible_kick_drift_and_continuity():
    rng = np.random.default_rng(12)
    cells, dx, h = 18, 0.07, 1e-11
    A = jnp.asarray(rng.normal(size=(cells, 3)) * 1e-9)
    E = jnp.asarray(rng.normal(size=(cells, 3)))
    phi = jnp.asarray(rng.normal(size=cells))
    J = jnp.asarray(rng.normal(size=(cells, 3)) * 1e-2)
    B = curl_E(A, jnp.zeros_like(A), dx, (0, 0))
    rho = epsilon_0 / 0.13 * (divergence(E, dx) + OMEGA**2 / c**2 * phi)
    E1, phi1 = kick(E, B, A, phi, J, h, dx, OMEGA, 0.13)
    rho1 = rho - h * divergence(J, dx)
    np.testing.assert_allclose(gauss(E1, phi1, rho1, dx, OMEGA, 0.13), 0, atol=2e-13)
    B1, A1 = drift(E1, B, A, phi1, h, dx)
    np.testing.assert_allclose(B1, curl_E(A1, B1, dx, (0, 0)), atol=2e-14)
    np.testing.assert_allclose(curl_E(gradient(phi, dx), B, dx, (0, 0)), 0, atol=2e-13)


@pytest.mark.parametrize('eta', [0., .005, .3])
def test_collective_resonance_reference_and_energy(eta):
    generator = np.array([[0., 0., 0., -1.], [0., 0., 1., -eta],
                          [0., -1., 0., 0.], [1., eta, 0., 0.]])
    initial = np.array([0., 1., 0., 0.])
    exact = resonant_exact(eta, 3.7)
    np.testing.assert_allclose(exact, expm(3.7 * generator) @ initial, rtol=0, atol=2e-14)
    assert abs(np.sum(abs(resonant_modes(exact, eta))**2) - exact @ exact) < 2e-14
    for method in ('cn', 'hybrid'):
        step = resonant_map(eta, .1, method)
        np.testing.assert_allclose(step.T @ step, np.eye(4), rtol=0, atol=2e-14)
    if eta == 0:
        # The hybrid's exact dark clock and CN plasma clock split at zero coupling.
        frequencies = np.sort(np.angle(np.linalg.eigvals(resonant_map(0., .1, 'hybrid'))))[-2:] / .1
        np.testing.assert_allclose(frequencies, [2 * np.arctan(.05) / .1, 1.], rtol=0, atol=2e-14)


def test_independent_vacuum_clock_matches_production_split():
    y0, _, split, unpack, _ = vacuum_system(16)
    electric, magnetic, potential, scalar = unpack(y0)
    dx, h = 2 * np.pi * c / (16 * OMEGA), 0.2 / OMEGA
    electric = jnp.asarray(electric)
    magnetic = jnp.asarray(magnetic / c)
    potential = jnp.asarray(potential / OMEGA)
    scalar = jnp.asarray(scalar * c / OMEGA)
    electric, scalar = kick(electric, magnetic, potential, scalar, jnp.zeros_like(electric),
                            h / 2, dx, OMEGA, 0.0)
    magnetic, potential = drift(electric, magnetic, potential, scalar, h, dx)
    electric, scalar = kick(electric, magnetic, potential, scalar, jnp.zeros_like(electric),
                            h / 2, dx, OMEGA, 0.0)
    measured = np.concatenate((np.asarray(electric).ravel(), np.asarray(magnetic * c).ravel(),
                               np.asarray(potential * OMEGA).ravel(), np.asarray(scalar * OMEGA / c)))
    np.testing.assert_allclose(measured, split(y0, 0.2), rtol=2e-14, atol=2e-14)


def test_vacuum_homogeneous_mode_and_dark_energy():
    base = plasma()
    initial = jnp.broadcast_to(jnp.array([0.0, 1.0, 0.0]), (base.domain.cells, 3))
    sim = DarkSimulation(base, DarkField(omega=OMEGA, eta=0.0, initial_E=initial))
    out = sim.run(120, store_every=1)
    dt = base.domain.dt
    omega_num = 2 * np.arcsin(OMEGA * dt / 2) / dt
    expected = np.cos(omega_num * np.arange(1, 121) * dt)
    np.testing.assert_allclose(np.asarray(out.E[:, 0, 1]), expected, atol=0.002)
    np.testing.assert_allclose(out.dark_gauss(), 0, atol=1e-14)
    ledger = out.energy()
    assert np.max(np.abs(np.asarray(ledger["total_with_dark"] / ledger["total_with_dark"][0] - 1))) < 0.003


def test_zero_coupling_matches_parent_and_restart(tmp_path):
    base = plasma(density=N_REF * 0.3)
    wave = jnp.broadcast_to(jnp.array([0.0, 1e-5, 0.0]), (base.domain.cells, 3))
    potential = jnp.broadcast_to(jnp.array([0.0, 1e-14, 0.0]), (base.domain.cells, 3))
    sim = DarkSimulation(base, DarkField(omega=OMEGA, eta=0.0,
                                         initial_E=wave, initial_A=potential))
    parent = base.run(steps=8, seed=3)
    out = sim.run(steps=8, seed=3)
    for name in ("E", "B", "x", "v", "rho", "J"):
        np.testing.assert_array_equal(getattr(out.ordinary, name), getattr(parent, name))
    first = sim.run(steps=3, seed=3)
    path = save_state(tmp_path / "crypt", first.state, sim)
    restored = load_state(path, sim)
    second = sim.run(steps=5, state=restored)
    for name in ("E", "B", "A", "phi", "work"):
        np.testing.assert_array_equal(getattr(second.state, name), getattr(out.state, name))
    np.testing.assert_array_equal(second.ordinary.E, out.ordinary.E[3:])


def test_reduced_energy_and_all_step_maxima_survive_stride_and_restart(tmp_path):
    sim = DarkSimulation(plasma(N_REF * 0.3, n=64), DarkField(OMEGA, 0.1))
    dense = sim.run(8, store_every=1)
    reduced = sim.run(8, store_every=2, store_particles=False)
    for key in ("kinetic", "electric", "magnetic", "dark", "total_with_dark",
                "dark_source_work", "closed_energy_error"):
        np.testing.assert_allclose(reduced.energy()[key], dense.energy()[key][1::2], rtol=2e-14)
    for key in ("initial_ordinary", "initial_dark", "max_balance_error",
                "max_ordinary_gauss", "max_dark_gauss"):
        np.testing.assert_allclose(getattr(reduced.state, key), getattr(dense.state, key), rtol=2e-14)
    first = sim.run(4, store_every=2, store_particles=False)
    path = save_state(tmp_path / "reduced", first.state, sim)
    second = sim.run(4, store_every=2, store_particles=False, state=load_state(path, sim))
    np.testing.assert_allclose(second.energy()["total_with_dark"], reduced.energy()["total_with_dark"][2:])
    np.testing.assert_allclose(second.state.max_balance_error, reduced.state.max_balance_error)
    np.testing.assert_allclose(second.state.initial_ordinary, reduced.state.initial_ordinary)


def test_restart_parameters_and_explicit_continuation(tmp_path):
    base = plasma(N_REF * 0.2)
    original = DarkSimulation(base, DarkField(OMEGA, 0.1))
    path = save_state(tmp_path / "field", original.run(3).state, original)
    for model in (DarkField(1.1 * OMEGA, 0.1), DarkField(OMEGA, 0.2)):
        changed = DarkSimulation(base, model)
        with pytest.raises(ValueError, match="dark.(omega|eta)"):
            load_state(path, changed)
        continued = load_for_continuation(path, original, changed)
        assert int(continued.ordinary.steps) == 3
        assert float(continued.work) == 0
        assert float(continued.max_balance_error) == 0
        residual = gauss(continued.E, continued.phi,
                         continued.ordinary.rho + continued.background,
                         base.domain.dx, model.omega, model.eta)
        assert float(jnp.max(jnp.abs(residual))) < 1e-5
    other_domain = Domain(length=base.domain.length * 2, cells=16, dt_over_dx_c=0.2)
    other = DarkSimulation(Simulation(other_domain, base.species), DarkField(OMEGA, 0.1))
    with pytest.raises(ValueError, match="dark.(length|dt)"):
        load_for_continuation(path, original, other)


def test_drive_archive_rejects_changed_force(tmp_path):
    base = plasma(N_REF * 0.2)
    original = DarkSimulation(base, PrescribedDrive(0.1, jnp.array([1e-5, 0, 0]), OMEGA, 0.2))
    path = save_state(tmp_path / "drive", original.run(2).state, original)
    for model in (PrescribedDrive(0.1, jnp.array([2e-5, 0, 0]), OMEGA, 0.2),
                  PrescribedDrive(0.1, jnp.array([1e-5, 0, 0]), OMEGA, 0.4),
                  PrescribedDrive(0.1, jnp.array([1e-5, 0, 0]), 1.1 * OMEGA, 0.2)):
        with pytest.raises(ValueError, match="dark.(amplitude|phase|omega)"):
            load_state(path, DarkSimulation(base, model))


def test_initial_longitudinal_projection_reports_correction():
    base = plasma(N_REF * 0.2)
    input_E = jnp.zeros((base.domain.cells, 3)).at[:, 0].set(
        jnp.sin(2 * jnp.pi * jnp.arange(base.domain.cells) / base.domain.cells))
    sim = DarkSimulation(base, DarkField(OMEGA, 0.1, initial_E=input_E))
    state, _ = sim.initial_state(jax.random.PRNGKey(0))
    projected, norm = project_initial_electric(input_E, state.ordinary.rho, state.background,
                                               state.phi, base.domain.dx, OMEGA, 0.1)
    np.testing.assert_allclose(state.E, projected)
    np.testing.assert_allclose(state.initial_projection_norm, norm)
    assert float(norm) > 0


@pytest.mark.parametrize("value", [np.float32(np.nan), jnp.asarray(np.inf), jnp.asarray(-1.0)])
def test_invalid_dark_frequency_scalar(value):
    with pytest.raises(ValueError, match="rest frequency"):
        DarkSimulation(plasma(), DarkField(value, 0.1))


def test_external_electric_field_requires_work_accounting():
    base = plasma(external_E=jnp.zeros((16, 3)))
    with pytest.raises(ValueError, match="external_E"):
        DarkSimulation(base, DarkField(OMEGA, 0.1))


def test_prescribed_static_drive_equals_parent_external_field_and_work():
    amplitude = 2e-5
    eta = 0.1
    base = plasma(density=N_REF * 0.1)
    drive = DarkSimulation(base, PrescribedDrive(
        eta=eta, amplitude=jnp.array([amplitude, 0.0, 0.0]), omega=0.0))
    ordinary = plasma(density=N_REF * 0.1, external_E=jnp.tile(jnp.array([eta * amplitude, 0.0, 0.0]),
                                                               (base.domain.cells, 1)))
    out = drive.run(12, seed=2)
    reference = ordinary.run(12, seed=2)
    np.testing.assert_allclose(out.ordinary.E, reference.E, rtol=0, atol=1e-18)
    np.testing.assert_allclose(out.ordinary.v, reference.v, rtol=1e-13, atol=1e-18)
    assert float(out.work[-1]) != 0.0


def test_prescribed_drive_requires_vector_amplitude():
    with pytest.raises(ValueError, match="three-component E vector"):
        DarkSimulation(plasma(), PrescribedDrive(0.1, 1e-5, OMEGA))
    with pytest.raises(ValueError, match="finite"):
        DarkSimulation(plasma(), PrescribedDrive(
            0.1, jnp.array([np.nan, 0.0, 0.0]), OMEGA))
    base = plasma()
    traced = jax.jit(lambda amplitude: DarkSimulation(
        base, PrescribedDrive(0.1, amplitude, OMEGA)).dark.amplitude)
    np.testing.assert_allclose(traced(jnp.array([1e-5, 0.0, 0.0])), [1e-5, 0, 0])


def waveform_plasma():
    """A cold neutral pair has an exact homogeneous, three-component response."""
    length = 2 * np.pi * c / OMEGA
    x, v = quiet_start(32, length)
    populations = (Species.electrons(32, N_REF / 2).replace(x=x, v=v),
                   Species("positive", 32, 1, mass_electron, N_REF / 2, x=x, v=v))
    return Simulation(Domain(length, 8, time_step=.04 / OMEGA), populations)


def waveform_reference(force, dt=.04):
    """Independent cold leapfrog recurrence, in electron-plasma units."""
    field, current, work, fields, currents = np.zeros(3), np.zeros(3), 0., [], []
    for value in force:
        midpoint = field - dt * current / 2
        following = current + dt * (midpoint + value)
        average = (current + following) / 2
        field = midpoint - dt * following / 2
        work += dt * np.dot(average, value)
        fields.append(field.copy())
        currents.append(average)
        current = following
    return np.asarray(fields), np.asarray(currents), current, work


def test_tabulated_drive_interpolation_and_cosine_default():
    knots = jnp.array([-.03, .13, .27])
    values = jnp.array([[.02, -.01, .015], [-.01, .03, .005], [.025, .02, -.01]])
    model = PrescribedDrive(.4, values, 0., times=knots)
    queries = jnp.array([-.1, -.03, .05, .2, .27, .5])
    expected = np.array([values[0], values[0], (values[0] + values[1]) / 2,
                         (values[1] + values[2]) / 2, values[2], values[2]])
    np.testing.assert_allclose(jax.vmap(model.at)(queries), expected, atol=2e-17, rtol=0)
    base = plasma()
    traced = jax.jit(lambda drive: DarkSimulation(base, drive).dark.at(.05))
    np.testing.assert_allclose(traced(model), expected[2], atol=2e-17, rtol=0)
    cosine = PrescribedDrive(.4, values[0], .7, .2)
    for time in queries:
        np.testing.assert_array_equal(cosine.at(time), values[0] * jnp.cos(.7 * time + .2))


@pytest.mark.parametrize("times, amplitude, omega, phase, message", [
    (0., np.zeros((2, 3)), 0., 0., "1D"),
    ([0.], np.zeros((1, 3)), 0., 0., "two knots"),
    ([[0., 1.]], np.zeros((2, 3)), 0., 0., "1D"),
    ([0., np.nan], np.zeros((2, 3)), 0., 0., "finite"),
    ([0., 0.], np.zeros((2, 3)), 0., 0., "increasing"),
    ([1., 0.], np.zeros((2, 3)), 0., 0., "increasing"),
    ([0j, 1j], np.zeros((2, 3)), 0., 0., "real"),
    ([0., 1.], np.zeros(3), 0., 0., "knots, 3"),
    ([0., 1.], np.full((2, 3), np.inf), 0., 0., "finite"),
    ([0., 1.], np.zeros((2, 3)), 1., 0., "omega=phase=0"),
    ([0., 1.], np.zeros((2, 3)), 0., .1, "omega=phase=0"),
])
def test_invalid_tabulated_drive(times, amplitude, omega, phase, message):
    with pytest.raises(ValueError, match=message):
        DarkSimulation(plasma(), PrescribedDrive(.4, amplitude, omega, phase, times))


def test_tabulated_drive_neutral_impulse_current_constraints_and_work():
    base, eta = waveform_plasma(), .4
    scale = mass_electron * c * OMEGA / e
    knots = np.array([-.03, .13, .27])
    values = np.array([[.02, -.01, .015], [-.01, .03, .005], [.025, .02, -.01]])
    force = eta * np.stack([np.interp((np.arange(8) + .5) * .04, knots, values[:, i])
                            for i in range(3)], axis=1)
    fields, currents, final_current, work = waveform_reference(force)
    model = PrescribedDrive(eta, scale * values, 0., times=knots / OMEGA)
    out = DarkSimulation(base, model).run(8)
    energy_scale = N_REF * mass_electron * c**2 * base.domain.length
    np.testing.assert_allclose(out.ordinary.E.mean(axis=1) / scale, fields, atol=2e-13, rtol=0)
    np.testing.assert_allclose(out.ordinary.J.mean(axis=1) / (epsilon_0 * scale * OMEGA),
                               currents, atol=2e-13, rtol=0)
    np.testing.assert_allclose(out.state.ordinary.u[:32] / c, np.tile(-final_current, (32, 1)),
                               atol=2e-13, rtol=0)
    np.testing.assert_allclose(out.state.ordinary.u[32:] / c, np.tile(final_current, (32, 1)),
                               atol=2e-13, rtol=0)
    np.testing.assert_allclose(out.work[-1] / energy_scale, work, atol=2e-14, rtol=0)
    # The explicit clock has a known modified-energy defect; do not hide it as exact physical energy.
    np.testing.assert_allclose(out.energy()["closed_balance_error"][-1] / energy_scale,
                               .04**2 * np.dot(final_current, final_current) / 8, atol=2e-14, rtol=0)
    charge_scale = e * N_REF * base.domain.length
    np.testing.assert_allclose(np.sum(np.asarray(out.ordinary.charge * out.state.ordinary.w)) / charge_scale,
                               0., atol=2e-13, rtol=0)
    np.testing.assert_allclose(out.ordinary.rho.sum(axis=1) * base.domain.dx / charge_scale,
                               0., atol=2e-13, rtol=0)
    momentum = np.sum(np.asarray(out.state.ordinary.w)[:, None]
                      * np.asarray(out.state.ordinary.u), axis=0)
    np.testing.assert_allclose(momentum / (N_REF * c * base.domain.length), 0., atol=2e-13, rtol=0)
    assert float(out.state.max_ordinary_gauss) / (e * N_REF / epsilon_0) < 2e-13
    assert out.state.E is None and float(out.state.max_dark_gauss) == 0
    with pytest.raises(ValueError, match="no dark Gauss law"):
        out.dark_gauss()


@pytest.mark.parametrize('clock', ['accumulated', 'anchored'])
def test_tabulated_drive_complete_restart_and_parameter_jump(tmp_path, clock):
    base = waveform_plasma()
    model = PrescribedDrive(.4, jnp.array([[1e-5, 2e-5, -1e-5], [2e-5, -1e-5, 3e-5]]),
                            0., times=jnp.array([0., .4 / OMEGA]))
    sim = DarkSimulation(base, model, clock=clock)
    whole, first = sim.run(8), sim.run(3)
    path = save_state(tmp_path / "waveform", first.state, sim)
    restored = load_state(path, sim)
    jax.tree.map(lambda a, b: np.testing.assert_array_equal(a, b), restored, first.state)
    tail = sim.run(5, state=restored)
    np.testing.assert_allclose(tail.ordinary.E, whole.ordinary.E[3:], rtol=2e-13, atol=1e-18)
    np.testing.assert_allclose(tail.work, whole.work[3:], rtol=2e-13, atol=1e-25)
    for key in ("initial_ordinary", "initial_dark", "initial_projection_norm", "background"):
        np.testing.assert_array_equal(getattr(tail.state, key), getattr(whole.state, key))
    np.testing.assert_allclose(tail.state.ordinary.u, whole.state.ordinary.u, rtol=2e-13, atol=1e-12)
    np.testing.assert_array_equal(tail.state.ordinary.time, whole.state.ordinary.time)
    assert int(tail.state.ordinary.steps) == 8
    with np.load(path, allow_pickle=False) as data:
        saved = {key: data[key] for key in data.files}
    assert saved["dark.format"] == (5 if clock == 'anchored' else 3)
    np.testing.assert_array_equal(saved["dark.times"], model.times)
    np.testing.assert_array_equal(saved["dark.amplitude"], model.amplitude)
    for changed, key in ((model.replace(times=model.times * 1.1), "times"),
                         (model.replace(amplitude=2 * model.amplitude), "amplitude")):
        following = DarkSimulation(base, changed, clock=clock)
        with pytest.raises(ValueError, match=f"dark.{key}"):
            load_state(path, following)
        continuation = load_for_continuation(path, sim, following)
        np.testing.assert_array_equal(continuation.ordinary.E, first.state.ordinary.E)
        assert float(continuation.ordinary.time) == float(first.state.ordinary.time)
        assert float(continuation.work) == float(continuation.max_balance_error) == 0
    del saved["dark.times"]
    np.savez(path, **saved)
    with pytest.raises(ValueError, match="dark.times"):
        load_state(path, sim)
    cosine = DarkSimulation(base, PrescribedDrive(.4, jnp.array([1e-5, 0., 0.]), 0.), clock=clock)
    path = save_state(tmp_path / "cosine", cosine.initial_state(jax.random.PRNGKey(0))[0], cosine)
    with np.load(path, allow_pickle=False) as data:
        assert data["dark.format"] == (5 if clock == 'anchored' else 2) and "dark.times" not in data.files
    assert load_for_continuation(path, cosine, sim).E is None
    different = base.replace(species=tuple(s.replace(n=64, x=None, v=None) for s in base.species))
    with pytest.raises(ValueError, match="populations"):
        load_for_continuation(path, cosine, DarkSimulation(different, model))


@pytest.mark.parametrize("order", [2, 5])
def test_tabulated_drive_continuation_preserves_particle_shape(tmp_path, order):
    base = waveform_plasma()
    if not hasattr(base.solver, "shape_order"):
        pytest.skip("requires the optional quintic parent feature")
    model = PrescribedDrive(.4, jnp.zeros((2, 3)), 0., times=jnp.array([0., .4 / OMEGA]))
    previous = DarkSimulation(base.replace(solver=base.solver.replace(shape_order=order)), model)
    following = DarkSimulation(base.replace(solver=base.solver.replace(shape_order=7 - order)), model)
    state, _ = previous.initial_state(jax.random.PRNGKey(0))
    path = save_state(tmp_path / "shape", state, previous)
    # A drive parameter jump retains the spatial discretization of the saved charge/current.
    with pytest.raises(ValueError, match="shape_order"):
        load_state(path, following)
    with pytest.raises(ValueError, match="shape_order"):
        load_for_continuation(path, previous, following)


def test_tabulated_drive_physical_objective_has_independent_frechet_derivative():
    base = waveform_plasma()
    field_scale = mass_electron * c * OMEGA / e
    energy_scale = N_REF * mass_electron * c**2 * base.domain.length
    knots = np.array([-.03, .13, .37])
    values = np.array([[.02, -.01, .015], [-.01, .03, .005], [.025, .02, -.01]])
    direction = np.array([[.01, .03, -.02], [-.01, .01, .02], [.02, -.02, .01]])
    knot_direction, eta, eta_direction = np.array([.02, -.01, .01]), .4, .03

    def objective(samples, coupling, times):
        drive = PrescribedDrive(coupling, field_scale * samples, 0., times=times / OMEGA)
        return DarkSimulation(base, drive).run(8, store_particles=False).energy()["total"][-1] / energy_scale

    force, tangent = [], []
    for time in (np.arange(8) + .5) * .04:
        left = np.searchsorted(knots, time) - 1
        alpha = (time - knots[left]) / (knots[left + 1] - knots[left])
        value = (1 - alpha) * values[left] + alpha * values[left + 1]
        d_alpha = -((1 - alpha) * knot_direction[left] + alpha * knot_direction[left + 1]) / (
            knots[left + 1] - knots[left])
        change = ((1 - alpha) * direction[left] + alpha * direction[left + 1]
                  + (values[left + 1] - values[left]) * d_alpha)
        force.append(eta * value)
        tangent.append(eta * change + eta_direction * value)
    fields, _, current, _ = waveform_reference(force)
    d_fields, _, d_current, _ = waveform_reference(tangent)
    expected = np.dot(fields[-1], d_fields[-1]) + np.dot(current, d_current)
    value, derivatives = jax.value_and_grad(objective, argnums=(0, 1, 2))(values, eta, knots)
    automatic = np.sum(derivatives[0] * direction) + derivatives[1] * eta_direction + np.dot(
        derivatives[2], knot_direction)
    np.testing.assert_allclose(value, .5 * (np.dot(fields[-1], fields[-1]) + np.dot(current, current)),
                               atol=2e-14, rtol=0)
    np.testing.assert_allclose(automatic, expected, atol=2e-13, rtol=1e-10)
    h = 2e-5
    finite = (objective(values + h * direction, eta + h * eta_direction, knots + h * knot_direction)
              - objective(values - h * direction, eta - h * eta_direction, knots - h * knot_direction)) / (2 * h)
    np.testing.assert_allclose(automatic, finite, atol=2e-12, rtol=2e-7)


def test_continuation_requires_matching_complete_field_state():
    base = plasma()
    field = DarkSimulation(base, DarkField(OMEGA, 0.1))
    drive = DarkSimulation(base, PrescribedDrive(0.1, jnp.zeros(3), OMEGA))
    state, _ = field.initial_state(jax.random.PRNGKey(0))
    with pytest.raises(ValueError, match="complete Proca state"):
        field.continue_with_parameters(state.replace(A=None))
    with pytest.raises(ValueError, match="drive mode does not match"):
        drive.continue_with_parameters(state)
    driven, _ = drive.initial_state(jax.random.PRNGKey(0))
    continued = drive.continue_with_parameters(driven)
    assert continued.E is None and float(continued.work) == 0.0


def test_output_without_particles_keeps_complete_energy_and_drive_has_no_dark_gauss():
    field = DarkSimulation(plasma(N_REF * 0.1), DarkField(OMEGA, 0.1)).run(
        1, store_particles=False)
    drive = DarkSimulation(plasma(N_REF * 0.1), PrescribedDrive(
        0.1, jnp.array([1e-5, 0.0, 0.0]), OMEGA)).run(1, store_particles=False)
    assert "dark" in field.energy() and "total_with_dark" in field.energy()
    assert "external_work" in drive.energy() and "closed_balance" in drive.energy()
    with pytest.raises(ValueError, match="no dark Gauss law"):
        drive.dark_gauss()


@pytest.mark.parametrize("model, message", [
    (object(), "DarkField or PrescribedDrive"),
    (DarkField(0.0, 0.1), "rest frequency"),
    (DarkField(100 * OMEGA, 0.1), "stability margin"),
    (DarkField(OMEGA, 0.1, initial_E=jnp.zeros((15, 3))), "initial_E"),
    (DarkField(OMEGA, 0.1, initial_A=jnp.zeros((15, 3))), "initial_A"),
    (DarkField(OMEGA, 0.1, initial_phi=jnp.zeros(15)), "initial_phi"),
    (DarkField(OMEGA, 0.1, initial_E=jnp.full((16, 3), jnp.nan)), "finite"),
    (DarkField(OMEGA, 0.1, initial_phi=jnp.full(16, jnp.nan)), "finite"),
])
def test_invalid_dark_configuration_fails_early(model, message):
    with pytest.raises((TypeError, ValueError), match=message):
        DarkSimulation(plasma(), model)


@pytest.mark.parametrize("steps, store_every", [(0, 1), (2, 0), (3, 2)])
def test_invalid_run_length_fails_early(steps, store_every):
    sim = DarkSimulation(plasma(), DarkField(OMEGA, 0.1))
    with pytest.raises(ValueError, match="divisible"):
        sim.run(steps, store_every=store_every)


@pytest.mark.parametrize("missing, message", [
    ("dark.mode", "dark.mode"), ("dark.background", "dark.background"),
    ("dark.work", "dark.work"), ("dark.E", "dark.E"),
    ("dark.phi", "dark.phi"), ("dark.initial_ordinary", "dark.initial_ordinary"),
])
def test_incomplete_dark_archive_is_rejected(tmp_path, missing, message):
    sim = DarkSimulation(plasma(), DarkField(OMEGA, 0.1))
    path = save_state(tmp_path / "incomplete", sim.run(1).state, sim)
    with np.load(path, allow_pickle=False) as saved:
        arrays = {name: saved[name] for name in saved.files if name != missing}
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match=message):
        load_state(path, sim)


def test_archive_roundtrips_external_magnetic_field_and_checks_dark_shapes(tmp_path):
    base = plasma(external_B=jnp.zeros((16, 3)))
    sim = DarkSimulation(base, DarkField(OMEGA, 0.1))
    state = sim.run(1).state
    path = save_state(tmp_path / "magnetic", state, sim)
    restored = load_state(path, sim)
    np.testing.assert_allclose(restored.E, state.E)
    with np.load(path, allow_pickle=False) as saved:
        arrays = {name: saved[name] for name in saved.files}
    assert "dark.external_B" in arrays
    for key, invalid, message in (("dark.E", np.zeros((16, 2)), "dark.E"),
                                  ("dark.work", np.zeros(1), "dark.work")):
        broken = tmp_path / f"broken_{key}.npz"
        np.savez(broken, **{**arrays, key: invalid})
        with pytest.raises(ValueError, match=message):
            load_state(broken, sim)


def test_midnight_style_restores_matplotlib_settings():
    import matplotlib as mpl

    old = (mpl.rcParams["figure.facecolor"], mpl.rcParams["axes.facecolor"])
    with midnight():
        assert mpl.rcParams["figure.facecolor"] == "white"
        assert mpl.rcParams["axes.facecolor"] == "white"
    assert (mpl.rcParams["figure.facecolor"], mpl.rcParams["axes.facecolor"]) == old


@pytest.mark.parametrize("density_ratio", [0.8, 1.0])
def test_time_dependent_drive_matches_cold_oracle_and_work(density_ratio):
    base = plasma(density=density_ratio * N_REF, n=128, cells=32)
    amplitude = 1e-5
    sim = DarkSimulation(base, PrescribedDrive(0.05, jnp.array([0.0, amplitude, 0.0]), OMEGA))
    out = sim.run(250)
    time = np.asarray(out.ordinary.t)
    wp = OMEGA * np.sqrt(density_ratio)
    if density_ratio == 1.0:
        expected = -0.05 * wp * amplitude / 2 * time * np.sin(wp * time)
    else:
        expected = (0.05 * wp**2 * amplitude / (wp**2 - OMEGA**2)
                    * (np.cos(wp * time) - np.cos(OMEGA * time)))
    measured = np.asarray(out.ordinary.E[:, :, 1].mean(axis=1))
    assert np.max(np.abs(measured - expected)) / amplitude < 1e-4
    total = np.asarray(out.energy()["total"])
    work = np.asarray(out.work)
    transfer = work[-1] - work[0]
    assert abs((total[-1] - total[0]) - transfer) / abs(transfer) < 1e-3


def test_dark_coupling_gradient_matches_finite_difference():
    base = plasma(density=N_REF * 0.2)
    wave = jnp.broadcast_to(jnp.array([0.0, 1e-5, 0.0]), (base.domain.cells, 3))

    def signal(eta):
        model = DarkField(omega=OMEGA, eta=eta, initial_E=wave)
        out = DarkSimulation(base, model).run(8, store_particles=False)
        return jnp.mean(out.ordinary.E[-1, :, 1])

    automatic = jax.grad(signal)(0.0)
    finite = (signal(1e-4) - signal(-1e-4)) / 2e-4
    np.testing.assert_allclose(automatic, finite, rtol=2e-4, atol=1e-12)
    assert abs(float(automatic)) > 0


def test_reduced_energy_gradient_matches_full_history():
    base = plasma(density=N_REF * 0.2)
    wave = jnp.broadcast_to(jnp.array([0.0, 1e-5, 0.0]), (base.domain.cells, 3))

    def objective(eta, keep):
        out = DarkSimulation(base, DarkField(OMEGA, eta, initial_E=wave)).run(
            6, store_every=2, store_particles=keep)
        return out.energy()["total_with_dark"][-1]

    full = jax.grad(lambda eta: objective(eta, True))(0.1)
    reduced = jax.grad(lambda eta: objective(eta, False))(0.1)
    np.testing.assert_allclose(reduced, full, rtol=1e-12, atol=1e-21)


def test_cold_homogeneous_exchange_against_matrix_exponential():
    density_ratio, eta, amplitude = 0.8, 0.05, 1e-5
    base = plasma(density=density_ratio * N_REF, n=128, cells=32)
    wave = jnp.broadcast_to(jnp.array([0.0, amplitude, 0.0]), (base.domain.cells, 3))
    out = DarkSimulation(base, DarkField(OMEGA, eta, initial_E=wave)).run(100)
    generator = np.array([[0, 0, 0, -1], [0, 0, 1, -eta],
                          [0, -1, 0, 0], [density_ratio, eta * density_ratio, 0, 0]])
    expected = np.array([expm(generator * t * OMEGA) @ [0, 1, 0, 0]
                         for t in np.asarray(out.ordinary.t)])
    ordinary = np.asarray(out.ordinary.E[:, :, 1].mean(axis=1)) / amplitude
    dark = np.asarray(out.E[:, :, 1].mean(axis=1)) / amplitude
    np.testing.assert_allclose(ordinary, expected[:, 0], atol=3e-4)
    np.testing.assert_allclose(dark, expected[:, 1], atol=3e-4)
    scale = e * N_REF / epsilon_0  # fixed physical charge scale, not the tiny perturbation
    assert np.max(np.abs(np.asarray(out.dark_gauss()))) / scale < 1e-7


def test_mean_current_drives_both_fields():
    drift_velocity = 1e3
    base = Simulation(plasma().domain,
                      (Species.electrons(32, density=0.3 * N_REF, drift=(drift_velocity, 0, 0)),))
    sim = DarkSimulation(base, DarkField(OMEGA, 0.1))
    out = sim.run(1)
    assert abs(float(jnp.mean(out.ordinary.J[0, :, 0]))) > 0
    assert abs(float(jnp.mean(out.ordinary.E[0, :, 0]))) > 0
    assert abs(float(jnp.mean(out.E[0, :, 0]))) > 0
    np.testing.assert_allclose(jnp.mean(out.E[0, :, 0]), 0.1 * jnp.mean(out.ordinary.E[0, :, 0]),
                               rtol=0.02)


@pytest.mark.parametrize('clock', ['accumulated', 'anchored'])
def test_drive_phase_survives_native_restart(tmp_path, clock):
    base = plasma(density=0.2 * N_REF)
    sim = DarkSimulation(base, PrescribedDrive(0.1, jnp.array([1e-5, 0, 0]), OMEGA, 0.3), clock=clock)
    whole = sim.run(9)
    first = sim.run(4)
    restored = load_state(save_state(tmp_path / "drive", first.state, sim), sim)
    tail = sim.run(5, state=restored)
    np.testing.assert_array_equal(tail.ordinary.E, whole.ordinary.E[4:])
    np.testing.assert_array_equal(tail.work, whole.work[4:])


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_all_three_vacuum_polarizations(axis):
    base = plasma(cells=64, n=4)
    d = base.domain
    k = 2 * np.pi / d.length
    k_hat = 2 * np.sin(k * d.dx / 2) / d.dx
    frequency = np.sqrt(c**2 * k_hat**2 + OMEGA**2)
    amplitude = 1e-14  # A in V s/m
    A = np.zeros((d.cells, 3))
    E = np.zeros_like(A)
    A[:, axis] = amplitude * np.cos(k * np.asarray(d.faces))
    phi = np.zeros(d.cells)
    if axis == 0:
        phi = c**2 * k_hat / frequency * amplitude * np.cos(k * np.asarray(d.grid))
        E[:, 0] = -OMEGA**2 / frequency * amplitude * np.sin(k * np.asarray(d.faces))
    sim = DarkSimulation(base, DarkField(OMEGA, 0.0, E, A, phi))
    out = sim.run(100)
    time = float(out.ordinary.t[-1])
    expected = amplitude * np.cos(k * np.asarray(d.faces) - frequency * time)
    if axis == 0:
        np.testing.assert_allclose(out.A[-1, :, axis], expected, atol=7e-5 * amplitude)
        assert np.max(np.abs(np.asarray(out.dark_gauss()))) < 1e-16
    else:
        # A transverse cosine has zero initial velocity: standing, not travelling.
        expected = amplitude * np.cos(k * np.asarray(d.faces)) * np.cos(frequency * time)
        np.testing.assert_allclose(out.A[-1, :, axis], expected, atol=1e-4 * amplitude)


def test_closed_energy_error_refines():
    errors, dark_work_errors = [], []
    for courant in (0.2, 0.1, 0.05):
        d = Domain(length=2 * np.pi * c / OMEGA, cells=16, dt_over_dx_c=courant)
        base = Simulation(d, (Species.electrons(64, density=0.8 * N_REF),))
        wave = jnp.broadcast_to(jnp.array([0.0, 1e-5, 0.0]), (16, 3))
        out = DarkSimulation(base, DarkField(OMEGA, 0.05, initial_E=wave)).run(round(10 / (OMEGA * d.dt)))
        ledger = out.energy()
        total = np.asarray(ledger["total_with_dark"])
        initial = out.state.initial_ordinary + out.state.initial_dark
        errors.append(abs(total[-1] / initial - 1))
        dark_work_errors.append(np.max(np.abs(np.asarray(ledger["dark_work_residual"]))) / initial)
        np.testing.assert_allclose(
            ledger["dark_work_residual"] + ledger["ordinary_work_residual"],
            total - initial, rtol=1e-9, atol=1e-30)
    assert errors[0] > 2.5 * errors[1] > 6 * errors[2]
    assert dark_work_errors[0] > 2.5 * dark_work_errors[1] > 2.5 * dark_work_errors[2]


def test_unsupported_model_fails():
    from jaxincell import Solver

    base = Simulation(plasma().domain, (Species.electrons(4, density=0),),
                      solver=Solver(model="electrostatic"))
    with pytest.raises(ValueError, match="periodic explicit electromagnetic"):
        DarkSimulation(base, DarkField(OMEGA, 0.1))


def test_initial_potential_solves_dark_gauss_and_output_is_jittable():
    base = plasma(density=0.3 * N_REF)
    phi = 1e-6 * jnp.cos(2 * jnp.pi * base.domain.grid / base.domain.length)
    sim = DarkSimulation(base, DarkField(OMEGA, 0.1, initial_phi=phi))
    state, _ = sim.initial_state(jax.random.PRNGKey(0))
    residual = gauss(state.E, state.phi, state.ordinary.rho + state.background,
                     base.domain.dx, OMEGA, 0.1)
    assert np.max(np.abs(np.asarray(residual))) / (e * N_REF / epsilon_0) < 1e-12
    result = jax.jit(lambda model: model.run(2, store_particles=False))(sim)
    assert result.E.shape == (2, base.domain.cells, 3)
    with pytest.raises(ValueError, match="zero mean"):
        DarkSimulation(base, DarkField(OMEGA, 0.1, initial_phi=jnp.ones(base.domain.cells)))


def test_oblique_magnetized_cold_3v_matrix_response():
    p, eta, amplitude = 0.8, 0.1, 1e-5
    d = plasma(cells=32).domain
    magnetic = np.array([0.2, -0.15, 0.1]) * mass_electron * OMEGA / e
    base = Simulation(d, (Species.electrons(128, density=p * N_REF),),
                      external_B=jnp.broadcast_to(magnetic, (d.cells, 3)))
    wave = jnp.broadcast_to(jnp.array([1.0, 0.4, -0.2]) * amplitude, (d.cells, 3))
    out = DarkSimulation(base, DarkField(OMEGA, eta, initial_E=wave)).run(200)
    generator = np.zeros((12, 12))
    eye = np.eye(3)
    generator[0:3, 9:12] = -eye
    generator[3:6, 6:9] = eye
    generator[3:6, 9:12] = -eta * eye
    generator[6:9, 3:6] = -eye
    generator[9:12, 0:3] = p * eye
    generator[9:12, 3:6] = eta * p * eye
    bx, by, bz = magnetic
    cross_right = np.array([[0, bz, -by], [-bz, 0, bx], [by, -bx, 0]])
    generator[9:12, 9:12] = -e / (mass_electron * OMEGA) * cross_right
    initial = np.r_[np.zeros(3), [1.0, 0.4, -0.2], np.zeros(6)]
    reference = np.array([expm(generator * t * OMEGA) @ initial
                          for t in np.asarray(out.ordinary.t)])
    fields = np.c_[np.asarray(out.ordinary.E).mean(axis=1) / amplitude,
                   np.asarray(out.E).mean(axis=1) / amplitude]
    np.testing.assert_allclose(fields, reference[:, :6], atol=6e-4)
    current = (-e * p * N_REF * np.asarray(out.ordinary.v).mean(axis=1)
               / (epsilon_0 * OMEGA * amplitude))
    np.testing.assert_allclose(current, reference[:, 9:12], atol=2e-4)


def test_density_derivative_includes_loading_and_energy_metric():
    objective, value_grad, _ = build_objective(16, 64, 2.0, 5.0)
    p = 0.9
    value, derivative = value_grad(p)
    cold_value, cold_derivative = cold_reference(p, 2.0, 5.0)
    np.testing.assert_allclose(value, cold_value, rtol=0.006)
    np.testing.assert_allclose(derivative, cold_derivative, rtol=0.006)
    h = 1e-4
    finite = (objective(p + h) - objective(p - h)) / (2 * h)
    np.testing.assert_allclose(derivative, finite, rtol=2e-5)


def test_jvp_vjp_and_mass_gradient_of_coupled_response():
    base = plasma(density=0.4 * N_REF)
    wave = jnp.broadcast_to(jnp.array([0.0, 1e-5, 0.0]), (base.domain.cells, 3))

    def signal(theta):
        eta, mass_ratio = theta
        model = DarkField(omega=OMEGA * mass_ratio, eta=eta, initial_E=wave)
        out = DarkSimulation(base, model).run(8, store_particles=False)
        return jnp.mean(out.ordinary.E[-1, :, 1]) / 1e-5

    point = jnp.array([0.1, 1.1])
    direction = jnp.array([0.3, -0.2])
    _, tangent = jax.jvp(signal, (point,), (direction,))
    _, pullback = jax.vjp(signal, point)
    adjoint = jnp.dot(pullback(jnp.array(1.0))[0], direction)
    finite = (signal(point + 1e-4 * direction) - signal(point - 1e-4 * direction)) / 2e-4
    np.testing.assert_allclose(tangent, adjoint, rtol=1e-11, atol=1e-12)
    np.testing.assert_allclose(tangent, finite, rtol=2e-5, atol=1e-9)


@pytest.mark.parametrize("stop", [5.0, 5.4])  # exact and partial checkpoint segments
def test_reduced_recurrences_have_same_discrete_gradient(stop):
    baseline = build_objective(8, 16, 2.0, stop, "scan", 16)[1](0.9)
    methods = ["remat", "segmented"]
    try:
        import solvax  # noqa: F401
    except ImportError:
        pass
    else:
        methods.append("solvax")
    for method in methods:
        candidate = build_objective(8, 16, 2.0, stop, method, 16)[1](0.9)
        np.testing.assert_allclose(candidate[0], baseline[0], rtol=1e-12)
        np.testing.assert_allclose(candidate[1], baseline[1], rtol=1e-10)


def test_mixed_kinetic_reference_root_and_zero_coupling_limit():
    k_bar = 2 * np.pi / (0.05 * 64)
    sigma_bar = 0.5 / k_bar
    mixed, residual = mixed_root(k_bar, sigma_bar, 0.3, 1.0)
    plain, plain_residual = mixed_root(k_bar, sigma_bar, 0.0, 1.0)
    np.testing.assert_allclose([mixed.real, mixed.imag], [1.4369479716, -0.1418558526], rtol=1e-8)
    np.testing.assert_allclose(plain, landau_root(0.5), rtol=1e-8)
    assert residual < 1e-10 and plain_residual < 1e-10


def test_mixed_instability_reference_limits():
    kc = 2 * np.pi / (0.05 * 64)
    mixed, residual = two_stream_growth(kc, 0.25 * kc, 0.7, 0.3)
    plain, _ = two_stream_growth(kc, 0.25 * kc, 0.7, 0)
    pole = (0.25 * kc)**2
    cold_plain = np.sqrt(-(2 * pole + 1 - np.sqrt(1 + 8 * pole)) / 2)
    np.testing.assert_allclose(plain, cold_plain, rtol=1e-12)
    np.testing.assert_allclose(mixed, 0.3466973106, rtol=1e-9)
    np.testing.assert_allclose(two_stream_growth(kc, 0.25 * kc, 0.7, -0.3)[0], mixed)
    assert residual < 1e-12 and mixed > plain

    growing, residual = weibel_growth(1.0, 0.08, 4.0, 0.7, 0.3)
    uncoupled, _ = weibel_growth(1.0, 0.08, 4.0, 0.7, 0)
    np.testing.assert_allclose([growing, uncoupled], [0.0511341941, 0.0488668629], rtol=1e-8)
    np.testing.assert_allclose(weibel_growth(1.0, 0.08, 4.0, 0.7, -0.3)[0], growing)
    assert residual < 1e-10
    with pytest.raises(ValueError):
        weibel_growth(1.0, 0.08, 1.0, 0.7, 0.3)


def test_profile_column_peak_and_independent_scattering():
    grid, basis, _ = slab_basis(16)
    ell = c / OMEGA
    column = 1.5 * N_REF * ell
    for curve in basis:
        np.testing.assert_allclose(np.trapezoid(curve, grid), 1, atol=1e-12)
    for theta in (np.zeros(3), np.array([0.6, -0.6, 0.6])):
        density = profile(grid * ell, theta, grid, basis, column)
        np.testing.assert_allclose(np.trapezoid(density, grid * ell), column, rtol=1e-12)
        assert max(density) < N_REF
    for theta in itertools.product((-0.65, 0.65), repeat=3):
        assert max(profile(grid * ell, theta, grid, basis, column)) < N_REF
    args = (OMEGA, 0.6 * OMEGA, 0.05, column, np.zeros(3), grid, basis)
    uncoupled = cold_scattering(OMEGA, 0.6 * OMEGA, 0, column, np.zeros(3), grid, basis)
    np.testing.assert_allclose(uncoupled, [0, 0, 0, 1], atol=1e-8)
    coarse, medium, fine = (cold_scattering(*args, n=n) for n in (101, 201, 401))
    assert abs(sum(fine) - 1) < 3e-7
    assert abs(coarse[1] - fine[1]) > 3 * abs(medium[1] - fine[1])
    weak = cold_scattering(OMEGA, 0.6 * OMEGA, 0.025, column,
                           np.zeros(3), grid, basis)
    np.testing.assert_allclose(fine[1] / weak[1], 4, rtol=0.002)


def test_uniform_slab_scattering_against_analytic_transfer():
    """A constant slab has an exact finite-width coupled-wave transfer map."""
    from jaxincell import mass_proton

    omega, mu, eta, ell = OMEGA, 0.6 * OMEGA, 0.05, c / OMEGA
    n0 = 0.5 * N_REF
    grid = np.linspace(-2, 2, 4001)
    basis = np.ones((4, len(grid))) / 4
    numerical = cold_scattering(omega, mu, eta, 4 * ell * n0,
                                np.zeros(3), grid, basis, n=801)[1]
    plasma2 = n0 * e**2 / epsilon_0 * (1 / mass_electron + 1 / mass_proton)
    kg, kd = omega / c, np.sqrt(omega**2 - mu**2) / c
    generator = np.array([[0, 0, 1, 0], [0, 0, 0, 1],
                          [(plasma2 - omega**2) / c**2, eta * plasma2 / c**2, 0, 0],
                          [eta * plasma2 / c**2, (mu**2 + eta**2 * plasma2 - omega**2) / c**2, 0, 0]],
                         dtype=complex)
    transfer = expm(generator * 4 * ell)
    incoming = np.array([0, 1, 0, 1j * kd])
    reflected_photon = np.array([1, 0, -1j * kg, 0])
    reflected_dark = np.array([0, 1, 0, -1j * kd])
    transmitted_photon = np.array([1, 0, 1j * kg, 0])
    transmitted_dark = np.array([0, 1, 0, 1j * kd])
    columns = np.stack((transfer @ reflected_photon, transfer @ reflected_dark,
                        -transmitted_photon, -transmitted_dark), axis=1)
    amplitudes = np.linalg.solve(columns, -transfer @ incoming)
    analytic = kg / kd * abs(amplitudes[2])**2
    np.testing.assert_allclose(numerical, analytic, rtol=0.005)


def test_packet_energy_zero_mixing_and_profile_derivative():
    cells, ell = 128, c / OMEGA
    grid, basis, positions = slab_basis(16)
    domain = Domain(length=80 * ell, cells=cells, dt_over_dx_c=0.4)
    packet_data = packet(domain, 0.6 * OMEGA, 0.8 / ell, -16 * ell, 3.5 * ell, 1e-5)
    E, A, incident = packet_data[:3]
    np.testing.assert_allclose(incident, 1e-5, rtol=1e-12)
    packet_Bz = (np.asarray(A[:, 1]) - np.roll(np.asarray(A[:, 1]), 1)) / domain.dx
    assert np.sum(np.asarray(E[:, 1]) * packet_Bz) > 0
    column = 1.5 * N_REF * ell
    detector = int(np.argmin(abs(np.asarray(domain.grid) - 8 * ell)))
    uncoupled = make_simulation(np.zeros(3), cells, 16, 0, 0.6 * OMEGA,
                                column, positions, E, A, 80)
    np.testing.assert_allclose(transmitted_fraction(uncoupled, incident, 160, detector), 0, atol=1e-15)
    no_wave = make_simulation(np.zeros(3), cells, 16, 0.05, 0.6 * OMEGA,
                              column, positions, jnp.zeros_like(E), jnp.zeros_like(A), 80)
    quiet = no_wave.run(8, store_every=8, store_particles=False)
    np.testing.assert_allclose(quiet.ordinary.E, 0, atol=2e-10)
    value_grad, objective, _ = build_design(cells, positions, 0.05, 0.6 * OMEGA,
                                            column, [packet_data], [[0, 0, 0]], 160,
                                            detector, particles_per_basis=16)
    point = np.array([0.1, -0.1, 0.05])
    direction = np.array([0.3, -0.4, 0.1])
    gradient = np.dot(np.asarray(value_grad(point)[1]), direction)
    h = 1e-3
    finite = (objective(point + h * direction) - objective(point - h * direction)) / (2 * h)
    np.testing.assert_allclose(gradient, finite, rtol=2e-5, atol=1e-9)


def six_face_fixture(parameters=(.09, .14, .07, 1., .2), field=False):
    """Small neutral relativistic loading with position, weight, drive and density inputs."""
    seed, ripple, amplitude, density, eta = parameters
    length, count = 2 * np.pi * c / OMEGA, 16
    base = (jnp.arange(count) + .371) * 2 * np.pi / count - np.pi
    populations = []
    for index, (charge, ratio) in enumerate(((-1., 1.), (1., 1836.))):
        x = jnp.zeros((count, 3)).at[:, 0].set(
            (base + (seed / 8 * jnp.sin(base) if index == 0 else 0)) * c / OMEGA)
        velocity = .035 + .06 * jnp.sin(2 * base + .3) if index == 0 else -.0002 + .004 * jnp.sin(base)
        v = jnp.zeros_like(x).at[:, 0].set(velocity * c)
        populations.append(Species('electron' if index == 0 else 'ion', count, charge,
                                   ratio * mass_electron, density * N_REF, x=x, v=v))
    plasma = Simulation(Domain(length, 8, time_step=.02 / OMEGA), tuple(populations),
                        Solver(relativistic=True, shape_order=5))
    scale = mass_electron * c * OMEGA / e
    angle = 2 * np.pi * jnp.arange(8) / 8
    model = (DarkField(OMEGA, eta, initial_A=jnp.zeros((8, 3)).at[:, 0].set(
        scale / OMEGA * ((amplitude + .04) * jnp.sin(angle + np.pi / 8 + .3) + .04)),
        initial_phi=.08 * scale * c / OMEGA * jnp.sin(angle + .2)) if field else
        PrescribedDrive(eta, jnp.array([amplitude * scale, 0., 0.]), OMEGA, .17))
    sim = DarkSimulation(plasma, model, 'six_face')
    state, extra = sim.initial_state(jax.random.PRNGKey(0))
    weights = 1 + ripple * jnp.cos(base)
    weights *= density * N_REF * length / jnp.sum(weights)
    weights = jnp.tile(weights, 2)
    from jaxincell._core import deposit, E_x_from_rho
    from darkjaxincell._simulation import _reset_diagnostics
    integer = state.ordinary.x[:, 0] - plasma.domain.dt / 2 * plasma._velocity(state.ordinary.u)[:, 0]
    rho = deposit(integer, extra[1] * weights, plasma.domain.grid[0], plasma.domain.dx, 8, (0, 0), 5)
    electric = state.ordinary.E.at[:, 0].set(E_x_from_rho(rho, plasma.domain.dx, (0, 0)) + .021 * scale)
    state = state.replace(ordinary=state.ordinary.replace(w=weights, rho=rho, E=electric), background=-jnp.mean(rho))
    if field:
        dark_e, _ = project_initial_electric(state.E, rho, state.background, state.phi,
                                             plasma.domain.dx, OMEGA, eta)
        dark_e = dark_e.at[:, 0].add(.053 * scale - jnp.mean(dark_e[:, 0]))
        state = state.replace(E=dark_e)
    return sim, _reset_diagnostics(state, plasma, model, extra[0]), extra


def six_face_basis(x, grid, length):
    """Independent SciPy cardinal spline and its position derivative."""
    from scipy.interpolate import BSpline
    dx = length / len(grid)
    coordinate = ((x[:, None] - grid + length / 2) % length - length / 2) / dx
    basis = BSpline.basis_element(np.arange(7) - 3., extrapolate=False)
    return np.nan_to_num(basis(coordinate)), np.nan_to_num(basis.derivative()(coordinate)) / dx


def six_face_reconstruction(E):
    """Independent six-face polynomial, rather than the production Q filter."""
    return (37 * (E + np.roll(E, 1)) - 8 * (np.roll(E, -1) + np.roll(E, 2))
            + np.roll(E, -2) + np.roll(E, 3)) / 60


def test_six_face_symbol_uniform_field_and_static_guards():
    from darkjaxincell._simulation import _six_face_filter
    from jaxincell._core import to_centres
    for mode in (0, 1, 3, 4):
        theta = 2 * np.pi * mode / 8
        E = jnp.exp(1j * theta * (jnp.arange(8) + .5))
        filtered = _six_face_filter(E)
        result = to_centres(filtered[:, None], filtered[-1:], filtered[-1:])[:, 0]
        symbol = (37 * np.cos(theta / 2) - 8 * np.cos(3 * theta / 2) + np.cos(5 * theta / 2)) / 30
        np.testing.assert_allclose(result, symbol * np.exp(1j * theta * np.arange(8)), atol=2e-14, rtol=0)
        np.testing.assert_allclose(result, six_face_reconstruction(np.asarray(E)), atol=2e-14, rtol=0)
        np.testing.assert_allclose(np.mean(filtered), np.mean(E), atol=2e-15, rtol=0)
    for value in (0., -.031, 2e6):
        E = jnp.full(8, value)
        np.testing.assert_array_equal(_six_face_filter(E), E)
    with pytest.raises(ValueError, match='longitudinal_gather'):
        DarkSimulation(plasma(), DarkField(OMEGA, .1), 'unknown')
    with pytest.raises(ValueError, match='shape_order=5'):
        DarkSimulation(plasma(), DarkField(OMEGA, .1), 'six_face')
    assert DarkSimulation(plasma(), DarkField(OMEGA, .1)).longitudinal_gather == 'average'


def test_six_face_neutral_force_proca_potential_momentum_and_mean_work():
    """Instantaneous 1V cancellation uses dark C0 and the original phi*C0A momentum."""
    sim, state, extra = six_face_fixture(field=True)
    p, d = sim.plasma, sim.plasma.domain
    scale = mass_electron * c * OMEGA / e
    dx, length = d.dx * OMEGA / c, d.length * OMEGA / c
    integer = np.asarray(state.ordinary.x[:, 0] - d.dt / 2 * p._velocity(state.ordinary.u)[:, 0])
    W, _ = six_face_basis(integer, np.asarray(d.grid), d.length)
    amount = np.asarray(extra[1] * state.ordinary.w) / (e * N_REF * c / OMEGA)
    E, D = np.asarray(state.ordinary.E[:, 0]) / scale, np.asarray(state.E[:, 0]) / scale
    phi, A = np.asarray(state.phi) * OMEGA / (scale * c), np.asarray(state.A[:, 0]) * OMEGA / scale

    def centre(F):
        return (F + np.roll(F, 1)) / 2

    def div(F):
        return (F - np.roll(F, 1)) / dx

    def grad(F):
        return (np.roll(F, -1) - F) / dx
    background = float(state.background) / (e * N_REF)
    rho = np.asarray(state.ordinary.rho) / (e * N_REF)
    np.testing.assert_allclose(div(E) - rho - background, 0., atol=2e-13, rtol=0)
    np.testing.assert_allclose(div(D) + phi - sim.dark.eta * (rho + background), 0., atol=2e-13, rtol=0)
    ordinary = amount @ W @ six_face_reconstruction(E)
    dark = sim.dark.eta * amount @ W @ centre(D)
    mass_rate = dx * (np.dot(-div(A), centre(A)) + np.dot(phi, centre(-D - grad(phi))))
    reaction = -background * length * (E.mean() + sim.dark.eta * D.mean())
    np.testing.assert_allclose((ordinary + dark + mass_rate - reaction) / length, 0., atol=2e-13, rtol=0)
    invalid_global = ordinary + sim.dark.eta * amount @ W @ six_face_reconstruction(D) + mass_rate
    expected = dx * np.dot(phi, six_face_reconstruction(D) - centre(D))
    np.testing.assert_allclose(invalid_global - reaction, expected, atol=2e-13, rtol=0)
    assert abs(expected / length) > 1e-8  # Changing dark gather is detectably incompatible.
    mean_v = np.asarray(p._velocity(state.ordinary.u)[:, 0]) / c
    current = amount @ mean_v / length
    assert abs(current) > 1e-3
    np.testing.assert_allclose(amount @ W @ np.full(8, E.mean()) / length, 0., atol=2e-13, rtol=0)
    np.testing.assert_allclose((amount * mean_v) @ W @ np.full(8, E.mean()) / length,
                               E.mean() * current, atol=2e-13, rtol=0)
    out = sim.run(4, state=state, store_particles=False)
    np.testing.assert_allclose(out.dark_gauss() * epsilon_0 / (e * N_REF), 0., atol=2e-13, rtol=0)
    np.testing.assert_allclose(divergence(out.ordinary.E, d.dx) * epsilon_0 / (e * N_REF)
                               - (out.ordinary.rho + out.state.background) / (e * N_REF), 0., atol=2e-13, rtol=0)


def test_six_face_accepted_work_identity_and_transverse_gather(monkeypatch):
    """Signed S/T use actual accepted currents and physical kinetic/field energy."""
    sim, state, extra = six_face_fixture()
    p, d = sim.plasma, sim.plasma.domain
    captured = {name: [] for name in ('_sources', '_advance_fields', '_fields_at')}
    for name in captured:
        original = getattr(Simulation, name)

        def observe(*args, _name=name, _original=original, **kwargs):
            result = _original(*args, **kwargs)
            captured[_name].append(result)
            return result
        monkeypatch.setattr(Simulation, name, observe)
    final, _ = sim._step(state, extra)
    Ehalf, _ = captured['_advance_fields'][0]
    J1, J2 = (np.asarray(value[1][:, 0]) for value in captured['_sources'])
    used = np.asarray(captured['_fields_at'][0][:, 0])
    u0, u1 = np.asarray(state.ordinary.u[:, 0]) / c, np.asarray(final.ordinary.u[:, 0]) / c
    sec = (u0 + u1) / (np.hypot(1., u0) + np.hypot(1., u1))
    scale, en = mass_electron * c * OMEGA / e, e * N_REF
    dt, dx, length = d.dt * OMEGA, d.dx * OMEGA / c, d.length * OMEGA / c
    amount = np.asarray(extra[1] * state.ordinary.w) / (en * c / OMEGA)
    W, slope = six_face_basis(np.asarray(state.ordinary.x[:, 0]), np.asarray(d.grid), d.length)
    electric, j1, j2 = np.asarray(Ehalf[:, 0]) / scale, J1 / (en * c), J2 / (en * c)
    rhodot = (amount * sec) @ (slope * c / OMEGA) / dx
    current = -dx * np.cumsum(rhodot)
    current += amount @ sec / length - current.mean()
    force = float(sim.dark.eta * sim.dark.at(d.dt / 2)[0]) / scale
    np.testing.assert_allclose(used / scale, W @ six_face_reconstruction(electric) + force, atol=2e-13, rtol=0)
    S = dt * ((amount * sec) @ (used / scale - force) - dx * current @ electric) / length
    T = (dt * dx * (current - (j1 + j2) / 2) @ electric
         + dt**2 * dx / 8 * np.sum(j2**2 - j1**2)) / length
    unit = N_REF * mass_electron * c**2 * d.length

    def kinetic(o):
        return np.sum(np.asarray(o.w) * np.asarray(extra[0]) * c**2
                      * (np.asarray(o.u[:, 0]) / c)**2 / (np.hypot(1., np.asarray(o.u[:, 0]) / c) + 1))

    def total(o):
        return kinetic(o) + epsilon_0 * d.dx / 2 * np.sum(np.asarray(o.E)**2)
    ledger = (total(final.ordinary) - total(state.ordinary) - float(final.work - state.work)) / unit
    np.testing.assert_allclose(ledger, S + T, atol=2e-13, rtol=0)
    np.testing.assert_allclose((kinetic(final.ordinary) - kinetic(state.ordinary)) / unit,
                               dt * (amount * sec) @ (used / scale) / length, atol=2e-13, rtol=0)
    for velocity, current_half in ((p._velocity(state.ordinary.u), j1), (p._velocity(final.ordinary.u), j2)):
        np.testing.assert_allclose(np.mean(current_half), amount @ np.asarray(velocity[:, 0]) / (c * length),
                                   atol=2e-13, rtol=0)
    # The optional filter affects Ex only; the parent still gathers Ey/Ez and all B.
    from darkjaxincell._simulation import _six_face_filter
    ordinary = Ehalf.at[:, 1].set(jnp.sin(jnp.arange(8)) * scale).at[:, 2].set(.04 * scale)
    dark = jnp.cos(jnp.arange(24).reshape(8, 3)) * .03 * scale
    magnetic = jnp.sin(jnp.arange(24).reshape(8, 3)) * .03 * scale / c
    baseline = p._fields_at(state.ordinary.x, ordinary + sim.dark.eta * dark, magnetic, state.ordinary.rho)
    modified = ordinary.at[:, 0].set(_six_face_filter(ordinary[:, 0]))
    result = p._fields_at(state.ordinary.x, modified + sim.dark.eta * dark, magnetic, state.ordinary.rho)
    np.testing.assert_array_equal(result[:, 1:], baseline[:, 1:])


@pytest.mark.parametrize('mode', ['cosine', 'waveform', 'field'])
@pytest.mark.parametrize('clock', ['accumulated', 'anchored'])
def test_six_face_restart_and_method_guards(tmp_path, mode, clock):
    sim, state, _ = six_face_fixture(field=mode == 'field')
    if clock == 'anchored':
        sim = sim.replace(clock=clock)
        state = state.replace(clock_time=state.ordinary.time, clock_step=state.ordinary.steps)
    if mode == 'waveform':
        sim = sim.replace(dark=PrescribedDrive(sim.dark.eta, jnp.stack((sim.dark.amplitude, -sim.dark.amplitude)),
                                               0., times=jnp.array([0., .2 / OMEGA])))
    whole, first = sim.run(4, state=state), sim.run(2, state=state)
    path = save_state(tmp_path / 'optional', first.state, sim)
    loaded = load_state(path, sim)
    jax.tree.map(lambda a, b: np.testing.assert_array_equal(a, b), loaded, first.state)
    resumed = sim.run(2, state=loaded)
    jax.tree.map(lambda a, b: np.testing.assert_array_equal(a, b), whole.state, resumed.state)
    with np.load(path, allow_pickle=False) as data:
        saved = dict(data)
    assert saved['dark.format'] == (5 if clock == 'anchored' else 4)
    assert str(saved['dark.longitudinal_gather']) == 'six_face'
    for previous, following in ((sim, sim.replace(longitudinal_gather='average')),
                                (sim.replace(longitudinal_gather='average'), sim)):
        native = state if previous.longitudinal_gather == 'average' else first.state
        archive = save_state(tmp_path / previous.longitudinal_gather, native, previous)
        with pytest.raises(ValueError, match='longitudinal_gather'):
            load_for_continuation(archive, previous, following)
    with pytest.raises(ValueError, match='longitudinal_gather|dark.format'):
        load_state(path, sim.replace(longitudinal_gather='average'))
    continued = load_for_continuation(path, sim, sim.replace(dark=sim.dark.replace(eta=.25)))
    assert float(continued.work) == float(continued.max_balance_error) == 0
    for key, value in (('dark.longitudinal_gather', None), ('dark.longitudinal_gather', np.asarray('average')),
                       ('dark.format', np.asarray(2))):
        broken = {name: array for name, array in saved.items() if name != key}
        if value is not None:
            broken[key] = value
        np.savez(tmp_path / 'broken.npz', **broken)
        with pytest.raises(ValueError, match='dark.(format|longitudinal_gather)'):
            load_state(tmp_path / 'broken.npz', sim)


@pytest.mark.parametrize('field', [False, True])
def test_six_face_initialized_objective_derivative(field):
    def objective(parameters):
        sim, state, _ = six_face_fixture(parameters, field=field)
        out = sim.run(4, state=state, store_particles=False)
        unit = N_REF * mass_electron * c**2 * sim.plasma.domain.length
        return out.energy()['kinetic'][-1] / unit + .03 * jnp.mean(
            (out.ordinary.E[-1] * e / (mass_electron * c * OMEGA))**2)
    point, direction = np.array([.09, .14, .07, 1., .2]), np.array([.3, -.2, .1, .4, -.3])
    value = jax.jit(objective)
    automatic = np.dot(np.asarray(jax.grad(objective)(jnp.asarray(point))), direction)
    for epsilon in (2e-5, 1e-5):
        finite = (float(value(point + epsilon * direction)) - float(value(point - epsilon * direction))) / (2 * epsilon)
        np.testing.assert_allclose(automatic, finite, atol=2e-10, rtol=2e-6)


def test_six_face_homogeneous_density_force_frechet_derivative():
    """Independent Frechet products of the cold explicit three-variable affine map."""
    point = np.array([.8, .4, .03])  # total density, coupling, bare force
    direction = np.array([.2, -.3, .1])
    base = waveform_plasma().replace(solver=Solver(shape_order=5))
    scale = mass_electron * c * OMEGA / e

    def objective(control):
        density, eta, amplitude = control
        p = base.replace(species=tuple(s.replace(density=s.density * density) for s in base.species))
        out = DarkSimulation(p, PrescribedDrive(eta, jnp.array([amplitude * scale, 0., 0.]), OMEGA), 'six_face').run(4)
        return out.energy()['total'][-1] / (N_REF * mass_electron * c**2 * p.domain.length)
    rho, eta, amplitude = point
    drho, deta, damp = direction
    y, dy, dt = np.array([0., 0., 1.]), np.zeros(3), .04
    for step in range(4):
        force = eta * amplitude * np.cos((step + .5) * dt)
        dforce = (deta * amplitude + eta * damp) * np.cos((step + .5) * dt)
        M = np.array([[1 - rho * dt**2 / 2, -dt + rho * dt**3 / 4, -rho * dt**2 * force / 2],
                      [rho * dt, 1 - rho * dt**2 / 2, rho * dt * force], [0., 0., 1.]])
        dM = np.array([[-drho * dt**2 / 2, drho * dt**3 / 4, -dt**2 * (drho * force + rho * dforce) / 2],
                       [drho * dt, -drho * dt**2 / 2, dt * (drho * force + rho * dforce)], [0., 0., 0.]])
        dy, y = dM @ y + M @ dy, M @ y
    expected = y[0] * dy[0] + y[1] * dy[1] / rho - y[1]**2 * drho / (2 * rho**2)
    _, derivative = jax.jvp(objective, (jnp.asarray(point),), (jnp.asarray(direction),))
    np.testing.assert_allclose(derivative, expected, atol=2e-13, rtol=2e-10)
    np.testing.assert_allclose(objective(point), .5 * (y[0]**2 + y[1]**2 / rho), atol=2e-13, rtol=2e-10)


def anchored_cold(parameters):
    """Fresh cold preparation; the nonzero origin is an input, not a legacy reset."""
    electric, velocity, origin, dt, force, density = parameters
    length = 2 * np.pi * c / OMEGA
    x, _ = quiet_start(16, length)
    populations = tuple(Species(name, 16, charge, ratio * mass_electron, density * N_REF, x=x,
                                v=jnp.zeros_like(x).at[:, 0].set(velocity * c if charge < 0 else 0.))
                        for name, charge, ratio in (('electron', -1, 1.), ('ion', 1, 1836.)))
    p = Simulation(Domain(length, 8, time_step=dt / OMEGA), populations)
    scale = mass_electron * c * OMEGA / e
    sim = DarkSimulation(p, PrescribedDrive(1., jnp.array([force * scale, 0., 0.]), OMEGA), clock='anchored')
    state, _ = sim.initial_state(jax.random.PRNGKey(0))
    step = jnp.asarray(37, jnp.int32)
    state = state.replace(ordinary=state.ordinary.replace(
        E=state.ordinary.E.at[:, 0].add(electric * scale), time=origin / OMEGA, steps=step),
        clock_time=origin / OMEGA, clock_step=step)
    return sim, sim.continue_with_parameters(state)


def anchored_cold_reference(parameters, steps):
    """Independent cold affine map and all six Fréchet derivatives in fixed units."""
    E0, v0, origin, dt, amplitude, density = map(float, parameters)
    H, K = np.eye(3), np.eye(3)
    H[0, 1:], K[1:, 0] = [dt * density / 2, -dt * density / 2], [-dt, dt / 1836]
    kick = np.array([0., -dt, dt / 1836])
    Hd, Hn, Kd = np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 3))
    Hd[0, 1:], Hn[0, 1:], Kd[1:, 0] = [.5 * density, -.5 * density], [dt / 2, -dt / 2], [-1., 1 / 1836]
    M, b = H @ K @ H, H @ kick
    Md, Mn = Hd @ K @ H + H @ Kd @ H + H @ K @ Hd, Hn @ K @ H + H @ K @ Hn
    bd, bn = Hd @ kick + H @ np.array([0., -1., 1 / 1836]), Hn @ kick
    y, tangent, trace = np.array([E0, v0, 0.]), np.zeros((3, 6)), []
    tangent[0, 0] = tangent[1, 1] = 1.
    for index in range(steps):
        phase = origin + (index + .5) * dt
        force = amplitude * np.cos(phase)
        derivative = np.array([0., 0., -amplitude * np.sin(phase),
                               -(index + .5) * amplitude * np.sin(phase), np.cos(phase), 0.])
        following = M @ tangent + b[:, None] * derivative
        following[:, 3] += Md @ y + bd * force
        following[:, 5] += Mn @ y + bn * force
        y, tangent = M @ y + b * force, following
        trace.append(y.copy())
    value = .5 * (y[0]**2 + density * (y[1]**2 + 1836 * y[2]**2))
    gradient = y[0] * tangent[0] + density * (y[1] * tangent[1] + 1836 * y[2] * tangent[2])
    gradient[5] += .5 * (y[1]**2 + 1836 * y[2]**2)
    return np.asarray(trace), value, gradient


def test_anchored_clock_phase_energy_frechet_and_finite_difference():
    point = np.array([.01, .02, .37, .04, .03, 1.1])

    def objective(parameters):
        sim, state = anchored_cold(parameters)
        out = sim.run(16, state=state, store_particles=False)
        return out.energy()['total'][-1] / (N_REF * mass_electron * c**2 * sim.plasma.domain.length)

    sim, state = anchored_cold(jnp.asarray(point))
    out = sim.run(16, state=state)
    scale = mass_electron * c * OMEGA / e
    velocity = np.asarray(out.ordinary.v[:, :, 0]) / c
    actual = np.column_stack((np.mean(out.ordinary.E[:, :, 0], axis=1) / scale,
                              velocity[:, :16].mean(axis=1), velocity[:, 16:].mean(axis=1)))
    trace, value, gradient = anchored_cold_reference(point, 16)
    np.testing.assert_allclose(actual, trace, atol=2e-13, rtol=0)
    np.testing.assert_allclose(objective(point), value, atol=2e-13, rtol=0)
    np.testing.assert_allclose(jax.grad(objective)(jnp.asarray(point)), gradient, atol=2e-12, rtol=2e-10)
    compiled = jax.jit(objective)
    errors = []
    for delta in (1e-4, 2.5e-5, 6.25e-6):
        finite = np.array([(float(compiled(point + delta * row)) - float(compiled(point - delta * row))) / (2 * delta)
                           for row in np.eye(6)])
        errors.append(np.max(abs(finite - gradient)))
    assert errors[-1] < errors[0] / 100
    np.testing.assert_allclose(finite, gradient, atol=2e-10, rtol=2e-7)
    zero = point.copy()
    zero[2] = 0.
    np.testing.assert_allclose(objective(zero), anchored_cold_reference(zero, 16)[1], atol=2e-13, rtol=0)


def test_anchored_clock_archive_guards_and_arbitrary_origin(tmp_path):
    from darkjaxincell._simulation import _check_clock
    sim, state = anchored_cold(jnp.array([.01, .02, .37, .04, .03, 1.1]))
    first, whole = sim.run(3, state=state), sim.run(8, state=state)
    path = save_state(tmp_path / 'clock', first.state, sim)
    loaded = load_state(path, sim)
    jax.tree.map(np.testing.assert_array_equal, loaded, first.state)
    resumed = sim.run(5, state=loaded)
    jax.tree.map(np.testing.assert_array_equal, resumed.state, whole.state)
    for name in ('clock_time', 'clock_step', 'initial_ordinary', 'initial_dark', 'background'):
        np.testing.assert_array_equal(getattr(resumed.state, name), getattr(state, name))
    following = sim.replace(dark=sim.dark.replace(eta=.8))
    jumped = load_for_continuation(path, sim, following)
    assert float(jumped.work) == 0
    np.testing.assert_array_equal(jumped.clock_time, state.clock_time)
    np.testing.assert_array_equal(jumped.clock_step, state.clock_step)
    legacy = sim.replace(clock='accumulated')
    with pytest.raises(ValueError, match='clock'):
        load_for_continuation(path, sim, legacy)
    with pytest.raises(ValueError, match='clock'):
        legacy.run(1, state=loaded)
    bare = loaded.replace(clock_time=None, clock_step=None)
    for operation in (lambda: sim.run(1, state=bare), lambda: sim._step(bare, sim.plasma.per_particle),
                      lambda: sim.continue_with_parameters(bare), lambda: save_state(path, bare, sim)):
        with pytest.raises(ValueError, match='clock_time'):
            operation()
    with pytest.raises(ValueError, match='clock must'):
        sim.replace(clock='reset')
    with np.load(path, allow_pickle=False) as data:
        saved = dict(data)
    assert saved['dark.format'] == 5 and str(saved['dark.clock']) == 'anchored'
    changes = [('dark.clock', None), ('dark.clock', np.asarray('accumulated')), ('dark.format', np.asarray(2)),
               ('dark.clock_time', None), ('dark.clock_step', None), ('dark.clock_time', np.asarray([0.])),
               ('dark.clock_time', np.asarray(np.inf)), ('dark.clock_time', np.asarray(1j)),
               ('dark.clock_step', np.asarray(37.)), ('dark.clock_step', np.asarray(True)),
               ('dark.clock_step', np.asarray(-1)), ('dark.clock_step', np.asarray(41)),
               ('dark.clock_time', np.asarray(.5 / OMEGA))]
    for key, value in changes:
        broken = {name: array for name, array in saved.items() if name != key}
        if value is not None:
            broken[key] = value
        np.savez(tmp_path / 'broken.npz', **broken)
        with pytest.raises(ValueError, match='clock|anchored|dark.format'):
            load_state(tmp_path / 'broken.npz', sim)
    # Spoofing a legacy format cannot smuggle anchors into an accumulated run.
    np.savez(tmp_path / 'broken.npz', **{**saved, 'dark.format': np.asarray(2)})
    with pytest.raises(ValueError, match='clock'):
        load_state(tmp_path / 'broken.npz', legacy)
    # A timestamp within representation roundoff is readable; a physical shift is not.
    near = loaded.replace(ordinary=loaded.ordinary.replace(time=jnp.nextafter(loaded.ordinary.time, jnp.inf)))
    _check_clock(near, sim)


def test_anchored_clock_scalar_long_count_and_closed_nonclock_equivalence(tmp_path):
    from decimal import Decimal, localcontext
    from darkjaxincell._simulation import _check_clock
    sim, state, _ = six_face_fixture(field=True)
    angle = jnp.arange(8) * 2 * jnp.pi / 8
    scale = mass_electron * c * OMEGA / e
    model = sim.dark.replace(initial_E=jnp.tile(jnp.array([.03, .02, -.01]) * scale, (8, 1)),
                             initial_A=jnp.stack((.004 * jnp.cos(angle), .003 * jnp.sin(angle),
                                                  .002 * jnp.cos(angle)), axis=1) * scale / OMEGA)
    sim = sim.replace(dark=model)
    state, _ = sim.initial_state(jax.random.PRNGKey(0))
    state = state.replace(ordinary=state.ordinary.replace(time=.37 / OMEGA, steps=jnp.asarray(9, jnp.int32)),
                          max_ordinary_gauss=jnp.asarray(1e-12 * e * N_REF / epsilon_0),
                          max_dark_gauss=jnp.asarray(2e-12 * e * N_REF / epsilon_0))
    anchored = sim.replace(clock='anchored')
    initial = state.replace(clock_time=state.ordinary.time, clock_step=state.ordinary.steps)
    whole, native = anchored.run(8, state=initial).state, sim.run(8, state=state).state
    nonclock = whole.replace(ordinary=whole.ordinary.replace(time=native.ordinary.time),
                             clock_time=None, clock_step=None)
    jax.tree.map(np.testing.assert_array_equal, nonclock, native)
    half = anchored.run(4, state=initial).state
    resumed = anchored.run(4, state=load_state(save_state(tmp_path / 'closed', half, anchored), anchored)).state
    jax.tree.map(np.testing.assert_array_equal, resumed, whole)
    for name in ('max_balance_error', 'max_ordinary_gauss', 'max_dark_gauss'):
        assert float(getattr(whole, name)) >= float(getattr(initial, name))
    assert float(whole.max_dark_gauss) * epsilon_0 / (e * N_REF) > 2e-13
    dt, count = .0025 / OMEGA, 400000
    naive = 0.
    for _ in range(count):
        naive += dt
    expected = count * dt
    with localcontext() as context:
        context.prec = 70
        exact = Decimal.from_float(dt) * count
        assert abs(Decimal.from_float(expected) - exact) * Decimal(OMEGA) < Decimal('1e-9')
        assert abs(Decimal.from_float(naive) - exact) * Decimal(OMEGA) > Decimal('1e-9')
    long = initial.replace(clock_time=jnp.asarray(0.), clock_step=jnp.asarray(0, jnp.int32),
                           ordinary=initial.ordinary.replace(time=jnp.asarray(expected), steps=jnp.asarray(count)))
    _check_clock(long, anchored.replace(plasma=sim.plasma.replace(domain=sim.plasma.domain.replace(time_step=dt))))


@pytest.mark.parametrize('clock', ['accumulated', 'anchored'])
def test_six_face_toml_cli_and_record(tmp_path, clock):
    source = tmp_path / 'optional.toml'
    source.write_text('''[domain]
length = 1.0
cells = 8
dt_over_dx_c = 0.1
[[species]]
name = "electron"
n = 16
mass = "electron"
charge = -1
density = 1e14
[solver]
shape_order = 5
[run]
steps = 2
[dark]
model = "drive"
omega = 1e9
eta = 0.1
amplitude = [1e-5, 0.0, 0.0]
longitudinal_gather = "six_face"
''')
    source.write_text(source.read_text() + f'clock = "{clock}"\n')
    sim, _ = load_toml(source)
    assert sim.longitudinal_gather == 'six_face'
    assert sim.clock == clock
    assert load_toml(source, longitudinal_gather='average')[0].longitudinal_gather == 'average'
    opposite = 'anchored' if clock == 'accumulated' else 'accumulated'
    assert load_toml(source, clock=opposite)[0].clock == opposite
    target = tmp_path / 'cli'
    assert main([str(source), '--longitudinal-gather', 'six_face', '--clock', clock, '--save', str(target)]) == 0
    record = json.loads((target / 'run.json').read_text())
    assert record['settings']['longitudinal_gather'] == 'six_face'
    assert record['settings'].get('clock', 'accumulated') == clock
    assert ('clock' in record['settings']) == (clock != 'accumulated')
    loaded = load_state(target / 'restart.npz', sim)
    assert int(loaded.ordinary.steps) == 2


def test_six_face_coupled_gather_keeps_dark_average(monkeypatch):
    sim, state, extra = six_face_fixture(field=True)
    p, d = sim.plasma, sim.plasma.domain
    scale = mass_electron * c * OMEGA / e
    angle = jnp.arange(8) * 2 * jnp.pi / 8
    state = state.replace(E=state.E.at[:, 1].set(.03 * scale * jnp.cos(angle)),
                          ordinary=state.ordinary.replace(E=state.ordinary.E.at[:, 2].set(.04 * scale)))
    from darkjaxincell._simulation import _reset_diagnostics
    state = _reset_diagnostics(state, p, sim.dark, extra[0])
    calls = {name: [] for name in ('_sources', '_advance_fields', '_fields_at')}
    for name in calls:
        original = getattr(Simulation, name)

        def observe(*args, _name=name, _original=original, **kwargs):
            value = _original(*args, **kwargs)
            calls[_name].append(value)
            return value
        monkeypatch.setattr(Simulation, name, observe)
    final, _ = sim._step(state, extra)
    J1 = calls['_sources'][0][1]
    Ehalf = np.asarray(calls['_advance_fields'][0][0]) / scale
    E_D, phi = kick(state.E, state.B, state.A, state.phi, J1, d.dt / 2, d.dx, OMEGA, sim.dark.eta)
    B_D, _ = drift(E_D, state.B, state.A, phi, d.dt / 2, d.dx)
    W, _ = six_face_basis(np.asarray(state.ordinary.x[:, 0]), np.asarray(d.grid), d.length)
    ordinary_centre = (Ehalf + np.roll(Ehalf, 1, axis=0)) / 2
    ordinary_centre[:, 0] = six_face_reconstruction(Ehalf[:, 0])
    dark = np.asarray(E_D) / scale
    expected = W @ (ordinary_centre + sim.dark.eta * (dark + np.roll(dark, 1, axis=0)) / 2)
    np.testing.assert_allclose(np.asarray(calls['_fields_at'][0])[:, :3] / scale, expected, atol=2e-13, rtol=0)
    Bhalf = np.asarray(calls['_advance_fields'][0][1])
    np.testing.assert_allclose(np.asarray(calls['_fields_at'][0])[:, 3:] * c / scale,
                               W @ ((Bhalf + sim.dark.eta * np.asarray(B_D)) * c / scale), atol=2e-13, rtol=0)
    np.testing.assert_allclose(gauss(final.E, final.phi, final.ordinary.rho + final.background,
                               d.dx, OMEGA, sim.dark.eta) * epsilon_0 / (e * N_REF), 0., atol=2e-13, rtol=0)
