"""Periodic 1D3V Maxwell–Proca fields on JAX-in-Cell's Yee grid.

The ghost field has teeth: its scalar potential is part of the state, not a
post-processed decoration.  All four operators use the parent's staggering.
"""

import jax.numpy as jnp

from jaxincell import epsilon_0, speed_of_light as c
from jaxincell._core import curl_B, curl_E


def divergence(face, dx):
    """Face-to-centre divergence, including the physical periodic mean."""
    longitudinal = face[..., 0]
    return (longitudinal - jnp.roll(longitudinal, 1, axis=-1)) / dx


def gradient(centre, dx):
    """Centre-to-face gradient, negative adjoint of :func:`divergence`."""
    gx = (jnp.roll(centre, -1, axis=-1) - centre) / dx
    return jnp.stack((gx, jnp.zeros_like(gx), jnp.zeros_like(gx)), axis=-1)


def kick(E, B, A, phi, J, h, dx, omega, eta):
    """Source and mass kick; preserves dark Gauss with a continuity current."""
    E = E + h * (c**2 * curl_B(B, E, dx, (0, 0)) + omega**2 * A - eta * J / epsilon_0)
    phi = phi - h * c**2 * divergence(A, dx)
    return E, phi


def drift(E, B, A, phi, h, dx):
    """Potential and magnetic drift; preserves ``B = curl A``."""
    A = A - h * (E + gradient(phi, dx))
    B = B - h * curl_E(E, B, dx, (0, 0))
    return B, A


def energy(E, B, A, phi, dx, omega):
    """Full canonical Proca energy per transverse area, in J/m²."""
    density = jnp.sum(E**2 + c**2 * B**2 + omega**2 * A**2, axis=(-2, -1))
    density = density + omega**2 / c**2 * jnp.sum(phi**2, axis=-1)
    return 0.5 * epsilon_0 * dx * density


def gauss(E, phi, rho_total, dx, omega, eta):
    """Unscaled dark Gauss residual at centres, in V/m²."""
    return divergence(E, dx) + omega**2 / c**2 * phi - eta * rho_total / epsilon_0
