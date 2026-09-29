"""A small, separate haunted wing of JAX-in-Cell's periodic PIC engine."""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax, random

from jaxincell import Simulation, energies, epsilon_0, speed_of_light as c
from jaxincell._config import pytree_dataclass
from jaxincell._core import E_x_from_rho, curl_E, wrap_positions
from jaxincell._simulation import Output

from ._proca import drift, energy, gauss, kick


def _check_phi_mean(phi):
    """A neutral periodic Proca scalar has no homogeneous charge source."""
    if phi is None or isinstance(phi, jax.core.Tracer):
        return
    values = np.asarray(phi)
    scale = np.max(np.abs(values))
    if scale and abs(np.mean(values)) > 1e-12 * scale:
        raise ValueError("periodic neutral Proca initial_phi must have zero mean")


@pytree_dataclass(static=())
class DarkField:
    """Dynamical canonical field: ``omega`` is the rest frequency in rad/s.

    ``eta`` is the exact current-coupling ratio. Initial arrays use the parent's
    face E/A and centre B convention; E and A are (cells, 3), phi is (cells,).
    When E is omitted, ``eta * ordinary E`` satisfies the shared background Gauss
    law. An explicit E requires a constraint-consistent phi.
    """

    omega: object
    eta: object
    initial_E: object = None
    initial_A: object = None
    initial_phi: object = None


@pytree_dataclass(static=())
class PrescribedDrive:
    """Homogeneous external E vector, V/m: ``eta * amplitude * cos(omega*t+phase)``.

    ``amplitude`` has shape ``(3,)``. This is a prescribed force with external
    work, not a dark reservoir.
    """

    eta: object
    amplitude: object
    omega: object
    phase: object = 0.0


@pytree_dataclass(static=())
class DarkState:
    """Complete restart state; ``background`` is the fixed neutralizing charge density."""

    ordinary: object
    E: object
    B: object
    A: object
    phi: object
    background: object
    work: object


@pytree_dataclass(static=())
class DarkOutput:
    """Parent output and complete ghost histories at stored times."""

    ordinary: Output
    E: object
    B: object
    A: object
    phi: object
    work: object
    state: DarkState
    model: object

    def energy(self):
        """Energy and source-work ledgers in J/m²; differences start at the first sample."""
        ordinary = energies(self.ordinary)
        if isinstance(self.model, DarkField):
            dark = energy(self.E, self.B, self.A, self.phi, self.ordinary.dx, self.model.omega)
            transfer = self.work - self.work[0]
            result = {**ordinary, "dark": dark, "dark_source_work": self.work,
                      "dark_work_residual": dark - dark[0] - transfer}
            if "total" in ordinary:
                result["total_with_dark"] = ordinary["total"] + dark
                result["ordinary_work_residual"] = ordinary["total"] - ordinary["total"][0] + transfer
            return result
        result = {**ordinary, "external_work": self.work}
        if "total" in ordinary:
            result["closed_balance"] = ordinary["total"] - self.work
        return result

    def dark_gauss(self):
        """Dark Gauss residual at every stored cell, in V/m²."""
        if not isinstance(self.model, DarkField):
            raise ValueError("a prescribed drive has no dark Gauss law")
        return gauss(self.E, self.phi, self.ordinary.rho + self.state.background,
                     self.ordinary.dx, self.model.omega, self.model.eta)


@pytree_dataclass(static=())
class DarkSimulation:
    """Periodic explicit Maxwell PIC with a dynamical ghost or prescribed drive.

    All particle loading, deposits, gathers, pushers and ordinary field updates
    are imported from ``jaxincell``. Unsupported solver combinations fail early.
    """

    plasma: Simulation
    dark: object

    def __post_init__(self):
        d, s = self.plasma.domain, self.plasma.solver
        if not isinstance(self.dark, (DarkField, PrescribedDrive)):
            raise TypeError("dark must be DarkField or PrescribedDrive")
        if (d.particle_bc != (0, 0) or d.field_bc != (0, 0) or s.algorithm != "explicit"
                or s.model != "electromagnetic" or s.field_solver != "ampere"
                or s.filter_passes or self.plasma.collisions is not None or self.plasma.sources):
            raise ValueError("dark runs require periodic explicit electromagnetic Ampere PIC, "
                             "with no filtering, sources or collisions")
        if isinstance(self.dark, DarkField):
            if isinstance(self.dark.omega, (int, float)) and self.dark.omega <= 0:
                raise ValueError("dark rest frequency must be positive")
            if isinstance(d.dt, (int, float)) and isinstance(self.dark.omega, (int, float)):
                limit = d.dt * (4 * c**2 / d.dx**2 + self.dark.omega**2) ** 0.5
                if limit >= 1.9:
                    raise ValueError("ghost field exceeds the explicit Proca stability margin")
            for name, shape in (("initial_E", (d.cells, 3)), ("initial_A", (d.cells, 3)),
                                ("initial_phi", (d.cells,))):
                value = getattr(self.dark, name)
                if value is not None and jnp.shape(value) != shape:
                    raise ValueError(f"{name} must have shape {shape}")
            _check_phi_mean(self.dark.initial_phi)
        elif jnp.shape(self.dark.amplitude) != (3,):
            raise ValueError("prescribed amplitude must be a three-component E vector")

    def initial_state(self, key):
        """Use parent loading and freeze its initial neutralizing background."""
        ordinary, extra = self.plasma.initial_state(key)
        background = -jnp.mean(ordinary.rho)
        if isinstance(self.dark, DarkField):
            shape = ordinary.E.shape
            A = jnp.zeros(shape) if self.dark.initial_A is None else jnp.asarray(self.dark.initial_A)
            E = self.dark.eta * ordinary.E if self.dark.initial_E is None else jnp.asarray(self.dark.initial_E)
            phi = jnp.zeros((shape[0],)) if self.dark.initial_phi is None else jnp.asarray(self.dark.initial_phi)
            source = (self.dark.eta * (ordinary.rho + background)
                      - epsilon_0 * self.dark.omega**2 / c**2 * phi)
            E = E.at[:, 0].set(E_x_from_rho(source, self.plasma.domain.dx, (0, 0)) + jnp.mean(E[:, 0]))
            B = curl_E(A, jnp.zeros(shape), self.plasma.domain.dx, (0, 0))
        else:
            E = B = A = phi = None
        state = DarkState(ordinary, E, B, A, phi, background, jnp.zeros(()))
        return state, extra

    def _step(self, state, extra):
        """Mirrored field halves around the parent's Boris push and two deposits."""
        p, d, model = self.plasma, self.plasma.domain, self.dark
        o, h, dx = state.ordinary, d.dt / 2, d.dx
        m, q = extra
        v = p._velocity(o.u)
        charge = q * o.w
        closure = p._current_closure(charge * v[:, 0])
        rho_half, J1 = p._sources(o.x, v, charge, h, closure, o.rho)
        E, B = p._advance_fields(o.E, o.B, J1, h, rho_half, o.wall, True)
        if isinstance(model, DarkField):
            E_D, phi = kick(state.E, state.B, state.A, state.phi, J1, h, dx, model.omega, model.eta)
            B_D, A = drift(E_D, state.B, state.A, phi, h, dx)
            work1 = -model.eta * h * dx * jnp.sum(J1 * (state.E + E_D) / 2)
            effective_E, effective_B = E + model.eta * E_D, B + model.eta * B_D
        else:
            drive = jnp.asarray(model.amplitude) * jnp.cos(model.omega * (o.time + h) + model.phase)
            effective_E, effective_B = E + model.eta * drive, B
        fields = p._fields_at(o.x, effective_E, effective_B, rho_half)
        u = p._accelerate(o.u, fields, o.qm, d.dt)
        v_new = p._velocity(u)
        x_half = wrap_positions(o.x + d.dt * v_new, o.w,
                                (d.length, d.length_y, d.length_z), d.particle_bc, dx)
        x_next = wrap_positions(x_half - h * v_new, o.w,
                                (d.length, d.length_y, d.length_z), d.particle_bc, dx)
        charge = q * o.w
        closure = p._current_closure(charge * v_new[:, 0])
        rho_next, J2 = p._sources(x_next, v_new, charge, h, closure, rho_half)
        E, B = p._advance_fields(E, B, J2, h, rho_next, o.wall, False)
        if isinstance(model, DarkField):
            B_D, A = drift(E_D, B_D, A, phi, h, dx)
            E_before = E_D
            E_D, phi = kick(E_D, B_D, A, phi, J2, h, dx, model.omega, model.eta)
            work = state.work + work1 - model.eta * h * dx * jnp.sum(J2 * (E_before + E_D) / 2)
        else:
            E_D = B_D = A = phi = None
            mean_v = p._mean_velocity(o.u, u)
            work = state.work + d.dt * jnp.sum(charge[:, None] * mean_v * model.eta * drive)
        totals = p._accumulate(o.moments, x_next, v_new, o.w)
        ordinary = o.replace(E=E, B=B, x=x_half, u=u, rho=rho_next,
                             time=o.time + d.dt, steps=o.steps + 1, moments=totals)
        next_state = DarkState(ordinary, E_D, B_D, A, phi, state.background, work)
        return next_state, (x_next, v_new, o.w, E, B, (J1 + J2) / 2, rho_next)

    def run(self, steps, seed=0, store_every=1, store_particles=True, state=None):
        """Run the same discrete transition with sparse parent and dark histories."""
        if steps < 1 or store_every < 1 or steps % store_every:
            raise ValueError("steps must be positive and divisible by store_every")
        carry, extra = self.initial_state(random.PRNGKey(seed)) if state is None else (state, self.plasma.per_particle)
        carry, history = _advance(self, carry, extra, steps // store_every, store_every, store_particles)
        x, v, w, E, B, J, rho, wall, t, n, sigma, E_D, B_D, A, phi, work = history
        d, (m, q) = self.plasma.domain, extra
        ordinary = Output(t=t, steps=n, sigma=sigma, x=x, v=v, E=E, B=B, J=J, rho=rho,
                          grid=d.grid, dx=d.dx, dt=d.dt, length=d.length, charge=q, mass=m,
                          weight=w, wall=wall, moments=None,
                          species=jnp.concatenate([jnp.full((s.n,), i) for i, s in enumerate(self.plasma.species)]),
                          state=carry.ordinary, names=tuple(s.name for s in self.plasma.species),
                          counts=tuple(s.n for s in self.plasma.species),
                          relativistic=self.plasma.solver.relativistic, field_bc=d.field_bc)
        return DarkOutput(ordinary, E_D, B_D, A, phi, work, carry, self.dark)


@partial(jax.jit, static_argnames=("chunks", "store_every", "store_particles"))
def _advance(sim, carry, extra, chunks, store_every, store_particles):
    def chunk(state, _):
        state, history = lax.scan(lambda st, _: sim._step(st, extra), state, None, length=store_every)
        x, v, w, E, B, J, rho = jax.tree.map(lambda a: a[-1], history)
        if not store_particles:
            x = v = w = None
        o = state.ordinary
        return state, (x, v, w, E, B, J, rho, o.wall, o.time, o.steps, o.sigma,
                       state.E, state.B, state.A, state.phi, state.work)

    return lax.scan(chunk, carry, None, length=chunks)
