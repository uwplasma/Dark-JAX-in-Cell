"""Independent field identities and parent-integration regressions."""

import numpy as np
import pytest
import jax
import jax.numpy as jnp
from scipy.linalg import expm

from jaxincell import (Domain, Simulation, Species, elementary_charge as e,
                       epsilon_0, mass_electron, speed_of_light as c)
from jaxincell._core import curl_E
from darkjaxincell import (DarkField, DarkSimulation, PrescribedDrive, gauss,
                           load_state, save_state)
from darkjaxincell._proca import divergence, drift, gradient, kick


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
