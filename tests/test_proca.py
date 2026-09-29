"""Independent field identities and parent-integration regressions."""

import numpy as np
import itertools
import pytest
import jax
import jax.numpy as jnp
from scipy.linalg import expm

from jaxincell import (Domain, Simulation, Species, elementary_charge as e,
                       epsilon_0, mass_electron, speed_of_light as c)
from jaxincell._core import curl_E
from jaxincell.theory import landau_root
from darkjaxincell import (DarkField, DarkSimulation, PrescribedDrive, gauss,
                           load_state, midnight, save_state)
from darkjaxincell._proca import divergence, drift, gradient, kick
from examples.optimize_dark_photon import build_objective, cold_reference
from examples.dark_kinetic import mixed_root
from examples.dark_instabilities import two_stream_growth, weibel_growth
from examples.dark_profile import (build_design, cold_scattering, packet, profile,
                                   slab_basis, transmitted_fraction, make_simulation)


OMEGA = 1e9
N_REF = epsilon_0 * mass_electron * OMEGA**2 / e**2


def plasma(density=0.0, n=16, cells=16, external_E=None):
    domain = Domain(length=2 * np.pi * c / OMEGA, cells=cells, dt_over_dx_c=0.2)
    return Simulation(domain, (Species.electrons(n, density=density),), external_E=external_E)


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


def test_output_without_particles_and_drive_has_no_dark_gauss():
    field = DarkSimulation(plasma(N_REF * 0.1), DarkField(OMEGA, 0.1)).run(
        1, store_particles=False)
    drive = DarkSimulation(plasma(N_REF * 0.1), PrescribedDrive(
        0.1, jnp.array([1e-5, 0.0, 0.0]), OMEGA)).run(1, store_particles=False)
    assert "dark" in field.energy() and "total_with_dark" not in field.energy()
    assert "external_work" in drive.energy() and "closed_balance" not in drive.energy()
    with pytest.raises(ValueError, match="no dark Gauss law"):
        drive.dark_gauss()


@pytest.mark.parametrize("model, message", [
    (object(), "DarkField or PrescribedDrive"),
    (DarkField(0.0, 0.1), "rest frequency"),
    (DarkField(100 * OMEGA, 0.1), "stability margin"),
    (DarkField(OMEGA, 0.1, initial_E=jnp.zeros((15, 3))), "initial_E"),
    (DarkField(OMEGA, 0.1, initial_A=jnp.zeros((15, 3))), "initial_A"),
    (DarkField(OMEGA, 0.1, initial_phi=jnp.zeros(15)), "initial_phi"),
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
    ("dark.mode", "dark mode"), ("dark.background", "missing background"),
    ("dark.work", "missing background or work"), ("dark.E", "missing Proca"),
    ("dark.phi", "missing Proca"),
])
def test_incomplete_dark_archive_is_rejected(tmp_path, missing, message):
    sim = DarkSimulation(plasma(), DarkField(OMEGA, 0.1))
    path = save_state(tmp_path / "incomplete", sim.run(1).state, sim)
    with np.load(path, allow_pickle=False) as saved:
        arrays = {name: saved[name] for name in saved.files if name != missing}
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match=message):
        load_state(path, sim)


def test_midnight_style_restores_matplotlib_settings():
    import matplotlib as mpl

    old = mpl.rcParams["figure.facecolor"]
    with midnight():
        assert mpl.rcParams["figure.facecolor"] == "#101018"
    assert mpl.rcParams["figure.facecolor"] == old


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


def test_drive_phase_survives_native_restart(tmp_path):
    base = plasma(density=0.2 * N_REF)
    sim = DarkSimulation(base, PrescribedDrive(0.1, jnp.array([1e-5, 0, 0]), OMEGA, 0.3))
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
    errors = []
    for courant in (0.2, 0.1, 0.05):
        d = Domain(length=2 * np.pi * c / OMEGA, cells=16, dt_over_dx_c=courant)
        base = Simulation(d, (Species.electrons(64, density=0.8 * N_REF),))
        wave = jnp.broadcast_to(jnp.array([0.0, 1e-5, 0.0]), (16, 3))
        out = DarkSimulation(base, DarkField(OMEGA, 0.05, initial_E=wave)).run(round(10 / (OMEGA * d.dt)))
        total = np.asarray(out.energy()["total_with_dark"])
        errors.append(abs(total[-1] / total[0] - 1))
    assert errors[0] > 2.5 * errors[1] > 6 * errors[2]


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
