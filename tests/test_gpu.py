"""Small field-kernel checks that also run on a GPU backend (``pytest -m gpu``)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxincell import epsilon_0, speed_of_light as c
from darkjaxincell._proca import divergence, drift, energy, gradient, kick

pytestmark = pytest.mark.gpu
OMEGA, ETA, N, DX, H = 1e9, 0.13, 64, 0.05, 1e-12


def _fields(seed=0):
    k = jax.random.split(jax.random.PRNGKey(seed), 5)
    return (jax.random.normal(k[0], (N, 3)), jax.random.normal(k[1], (N, 3)),
            jax.random.normal(k[2], (N, 3)), jax.random.normal(k[3], (N,)),
            jax.random.normal(k[4], (N, 3)))


def test_divergence_gradient_adjoint():
    face, centre, *_ = _fields()
    lhs = jnp.sum(divergence(face, DX) * centre[:, 0])
    rhs = -jnp.sum(face * gradient(centre[:, 0], DX))
    np.testing.assert_allclose(lhs, rhs, rtol=1e-10)


def test_kick_drift_preserve_finite_state():
    E, B, A, phi, J = _fields(1)
    E1, phi1 = jax.jit(lambda *a: kick(*a, H, DX, OMEGA, ETA))(E, B, A, phi, J)
    B1, A1 = jax.jit(lambda *a: drift(*a, H, DX))(E1, B, A, phi1)
    assert all(bool(jnp.all(jnp.isfinite(x))) for x in (E1, phi1, B1, A1))
    assert B1.shape == B.shape and A1.shape == A.shape


def test_energy_gradient_matches_analytic():
    E, B, A, phi, _ = _fields(2)
    grad = jax.jit(jax.grad(energy, argnums=2))(E, B, A, phi, DX, OMEGA)
    np.testing.assert_allclose(grad, epsilon_0 * DX * OMEGA**2 * A, rtol=1e-8)


def test_magnetic_energy_scales_quadratically():
    E, B, A, phi, _ = _fields(3)
    e1 = energy(jnp.zeros_like(E), B, jnp.zeros_like(A), jnp.zeros_like(phi), DX, OMEGA)
    e2 = energy(jnp.zeros_like(E), 2 * B, jnp.zeros_like(A), jnp.zeros_like(phi), DX, OMEGA)
    np.testing.assert_allclose(e2, 4 * e1, rtol=1e-10)
    assert c > 0
