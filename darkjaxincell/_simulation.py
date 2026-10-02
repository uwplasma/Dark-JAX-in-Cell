"""A small, separate haunted wing of JAX-in-Cell's periodic PIC engine."""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax, random

from jaxincell import Simulation, energies, epsilon_0, speed_of_light as c
from jaxincell._config import pytree_dataclass
from jaxincell._core import E_x_from_rho, curl_E, wrap_positions
from jaxincell._progress import reporter
from jaxincell._simulation import Output, _groups, _join

from ._proca import divergence, drift, energy, gauss, kick


def _plain_scalar(value, name, *, positive=False):
    """Check concrete controls on the host; traced trials need an external feasibility check."""
    if isinstance(value, jax.core.Tracer):
        return None
    number = np.asarray(value)
    if number.shape != () or not np.isfinite(number):
        raise ValueError(f"{name} must be a finite scalar")
    if positive and number <= 0:
        raise ValueError(f"{name} must be positive")
    return float(number)


def _check_phi_mean(phi):
    """A neutral periodic Proca scalar has no homogeneous charge source."""
    if phi is None or isinstance(phi, jax.core.Tracer):
        return
    values = np.asarray(phi)
    scale = np.max(np.abs(values))
    if scale and abs(np.mean(values)) > 1e-12 * scale:
        raise ValueError("periodic neutral Proca initial_phi must have zero mean")


def project_initial_electric(E, rho, background, phi, dx, omega, eta):
    """Return Gauss-compatible face E and the L2 norm of its longitudinal correction."""
    source = eta * (rho + background) - epsilon_0 * omega**2 / c**2 * phi
    longitudinal = E_x_from_rho(source, dx, (0, 0)) + jnp.mean(E[:, 0])
    correction = jnp.sqrt(dx * jnp.sum((longitudinal - E[:, 0])**2))
    return E.at[:, 0].set(longitudinal), correction


def _six_face_filter(E):
    """Periodic face filter Q: the parent's adjacent average gives six-face R6.

    Its mean is unchanged in exact arithmetic; uniform fields are unchanged
    bitwise. Only the ordinary longitudinal gather uses this experimental map.
    """
    return E + (-9 * ((jnp.roll(E, 1) - E) + (jnp.roll(E, -1) - E))
                + (jnp.roll(E, 2) - E) + (jnp.roll(E, -2) - E)) / 30


def _kinetic_species(plasma, ordinary, mass):
    """Sum each physical population using the parent's pusher-energy convention."""
    particle_energy = ordinary.w * plasma._kinetic(mass, ordinary.u)
    start, values = 0, []
    for species in plasma.species:
        values.append(jnp.sum(particle_energy[start:start + species.n]))
        start += species.n
    return jnp.stack(values)


def _ordinary_total(ordinary, kinetic, dx):
    fields = 0.5 * epsilon_0 * dx * jnp.sum(ordinary.E**2 + c**2 * ordinary.B**2)
    return fields + jnp.sum(kinetic)


def _reset_diagnostics(state, plasma, model, mass):
    """Start an energy/constraint ledger at this physical state."""
    ordinary, dx = state.ordinary, plasma.domain.dx
    kinetic = _kinetic_species(plasma, ordinary, mass)
    initial_ordinary = _ordinary_total(ordinary, kinetic, dx)
    initial_dark = (energy(state.E, state.B, state.A, state.phi, dx, model.omega)
                    if isinstance(model, DarkField) else jnp.zeros(()))
    ordinary_gauss = jnp.max(jnp.abs(divergence(ordinary.E, dx)
                                     - (ordinary.rho + state.background) / epsilon_0))
    dark_gauss = (jnp.max(jnp.abs(gauss(state.E, state.phi, ordinary.rho + state.background,
                                        dx, model.omega, model.eta)))
                  if isinstance(model, DarkField) else jnp.zeros(()))
    return state.replace(work=jnp.zeros(()), initial_ordinary=initial_ordinary,
                         initial_dark=initial_dark, max_balance_error=jnp.zeros(()),
                         max_ordinary_gauss=ordinary_gauss, max_dark_gauss=dark_gauss)


def _check_field_model(model, domain):
    omega = _plain_scalar(model.omega, "dark rest frequency", positive=True)
    _plain_scalar(model.eta, "dark coupling")
    if omega is not None:
        limit = float(domain.dt) * (4 * c**2 / float(domain.dx)**2 + omega**2) ** 0.5
        if limit >= 1.9:
            raise ValueError("dark field exceeds the explicit Proca stability margin")
    for name, shape in (("initial_E", (domain.cells, 3)), ("initial_A", (domain.cells, 3)),
                        ("initial_phi", (domain.cells,))):
        value = getattr(model, name)
        if value is not None and jnp.shape(value) != shape:
            raise ValueError(f"{name} must have shape {shape}")
        if value is not None and not isinstance(value, jax.core.Tracer):
            if not np.all(np.isfinite(np.asarray(value))):
                raise ValueError(f"{name} must be finite")
    _check_phi_mean(model.initial_phi)


def _waveform_shape(model):
    """Validate concrete knots; traced time trials need an external monotonicity check."""
    shape = np.shape(model.times)
    if len(shape) != 1 or shape[0] < 2:
        raise ValueError("tabulated drive times must be a 1D array with at least two knots")
    if jnp.iscomplexobj(model.times):
        raise ValueError("tabulated drive times must be real")
    for name in ("omega", "phase"):
        value = _plain_scalar(getattr(model, name), f"drive {name}")
        if value is not None and value != 0:
            raise ValueError("tabulated drive requires omega=phase=0")
    if not isinstance(model.times, jax.core.Tracer):
        times = np.asarray(model.times)
        if not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0):
            raise ValueError("tabulated drive times must be finite and strictly increasing")
    return shape[0], 3


def _check_drive_model(model):
    for name in ("eta", "omega", "phase"):
        _plain_scalar(getattr(model, name), f"drive {name}")
    shape = (3,) if model.times is None else _waveform_shape(model)
    if np.shape(model.amplitude) != shape:
        raise ValueError("prescribed amplitude must be a three-component E vector" if model.times is None
                         else "tabulated amplitude must have shape (knots, 3)")
    if not isinstance(model.amplitude, jax.core.Tracer):
        if not np.all(np.isfinite(np.asarray(model.amplitude))):
            raise ValueError("prescribed amplitude must be finite")


@pytree_dataclass(static=())
class DarkField:
    """Dynamical canonical field: ``omega`` is the rest frequency in rad/s.

    ``eta`` is the exact current-coupling ratio. Initial arrays use the parent's
    face E/A and centre B convention; E and A are (cells, 3), phi is (cells,).
    When E is omitted, ``eta * ordinary E`` satisfies the shared background Gauss
    law. A supplied longitudinal E is projected onto dark Gauss; the correction
    norm is returned in ``DarkState.initial_projection_norm``.
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

    With ``times`` in seconds, ``amplitude`` is ``(knots, 3)`` and gives bare field
    samples in V/m. Samples interpolate linearly, with constant endpoint continuation;
    set ``omega=phase=0``. Cover the physical run horizon when constructing a control.
    """

    eta: object
    amplitude: object
    omega: object
    phase: object = 0.0
    times: object = None

    def at(self, time):
        """Bare prescribed E at a scalar time; the force multiplies this by ``eta``."""
        amplitude = jnp.asarray(self.amplitude)
        if self.times is None:
            return amplitude * jnp.cos(self.omega * time + self.phase)
        return jnp.stack([jnp.interp(time, jnp.asarray(self.times), amplitude[:, i]) for i in range(3)], axis=-1)


@pytree_dataclass(static=())
class DarkState:
    """Complete restart state, including energy references and all-step maxima.

    ``clock_time`` (seconds) and ``clock_step`` anchor an optional counter-based
    clock. They remain ``None`` for the accumulated clock and survive restarts.
    """

    ordinary: object
    E: object
    B: object
    A: object
    phi: object
    background: object
    work: object
    initial_ordinary: object
    initial_dark: object
    initial_projection_norm: object
    max_balance_error: object
    max_ordinary_gauss: object
    max_dark_gauss: object
    clock_time: object = None
    clock_step: object = None


def _check_clock(state, simulation):
    """Reject scheme changes and inconsistent anchors without rephasing a state."""
    anchors = state.clock_time, state.clock_step
    if getattr(simulation, "clock", "accumulated") == "accumulated":
        if any(value is not None for value in anchors):
            raise ValueError("accumulated clock cannot resume anchored state")
        return
    values = (*anchors, state.ordinary.time, state.ordinary.steps)
    for value, name, kind in zip(values, ("clock_time", "clock_step", "time", "steps"),
                                 (np.floating, np.integer, np.floating, np.integer)):
        if value is None or np.shape(value) != ():
            raise ValueError(f"anchored {name} must be scalar")
        dtype = getattr(value, "dtype", None)
        if not np.issubdtype(dtype if dtype is not None else np.asarray(value).dtype, kind):
            raise ValueError(f"anchored {name} has unsupported dtype")
    if any(isinstance(value, jax.core.Tracer) for value in (*values, simulation.plasma.domain.dt)):
        return
    origin, anchor_step, actual, step = map(np.asarray, values)
    if not np.isfinite(origin) or not np.isfinite(actual) or anchor_step < 0 or step < anchor_step:
        raise ValueError("anchored clock needs finite times and ordered nonnegative steps")
    elapsed = (int(step) - int(anchor_step)) * float(simulation.plasma.domain.dt)
    expected = np.asarray(float(origin) + elapsed, dtype=actual.dtype)
    # Only representation roundoff in the multiplication and addition is allowed.
    tolerance = 2 * (abs(np.spacing(np.asarray(elapsed, dtype=actual.dtype))) + abs(np.spacing(expected)))
    if not np.isfinite(expected) or abs(float(actual) - float(expected)) > tolerance:
        raise ValueError("anchored timestamp does not match clock_time/clock_step")


@pytree_dataclass(static=())
class DarkOutput:
    """Parent output and dark histories at stored times, with reduced kinetic ledgers."""

    ordinary: Output
    E: object
    B: object
    A: object
    phi: object
    work: object
    state: DarkState
    model: object
    kinetic_species: object

    def energy(self):
        """Energy and source-work ledgers in J/m², referenced to the physical t=0 state."""
        ordinary = energies(self.ordinary)
        kinetic = jnp.sum(self.kinetic_species, axis=1)
        ordinary["kinetic"] = kinetic
        for index, name in enumerate(self.ordinary.names):
            ordinary[f"kinetic_{name}"] = self.kinetic_species[:, index]
        ordinary["total"] = ordinary["electric"] + ordinary["magnetic"] + kinetic
        ordinary["energy_error"] = (jnp.abs(ordinary["total"] - self.state.initial_ordinary)
                                    / jnp.maximum(jnp.abs(self.state.initial_ordinary),
                                                  jnp.finfo(kinetic.dtype).tiny))
        if isinstance(self.model, DarkField):
            dark = energy(self.E, self.B, self.A, self.phi, self.ordinary.dx, self.model.omega)
            transfer = self.work
            result = {**ordinary, "dark": dark, "dark_source_work": self.work,
                      "dark_work_residual": dark - self.state.initial_dark - transfer}
            result["total_with_dark"] = ordinary["total"] + dark
            result["ordinary_work_residual"] = ordinary["total"] - self.state.initial_ordinary + transfer
            result["closed_energy_error"] = result["total_with_dark"] - (
                self.state.initial_ordinary + self.state.initial_dark)
            result["max_dark_gauss_V_m2"] = self.state.max_dark_gauss
            result["initial_projection_norm"] = self.state.initial_projection_norm
            result["max_balance_error"] = self.state.max_balance_error
            result["max_ordinary_gauss_V_m2"] = self.state.max_ordinary_gauss
            return result
        result = {**ordinary, "external_work": self.work}
        result["closed_balance"] = ordinary["total"] - self.work
        result["closed_balance_error"] = result["closed_balance"] - self.state.initial_ordinary
        result["max_balance_error"] = self.state.max_balance_error
        result["max_ordinary_gauss_V_m2"] = self.state.max_ordinary_gauss
        return result

    def dark_gauss(self):
        """Dark Gauss residual at every stored cell, in V/m²."""
        if not isinstance(self.model, DarkField):
            raise ValueError("a prescribed drive has no dark Gauss law")
        return gauss(self.E, self.phi, self.ordinary.rho + self.state.background,
                     self.ordinary.dx, self.model.omega, self.model.eta)


@pytree_dataclass(static=("longitudinal_gather", "clock"))
class DarkSimulation:
    """Periodic explicit Maxwell PIC with a dynamical Proca field or prescribed drive.

    All particle loading, deposits, gathers, pushers and ordinary field updates
    are imported from ``jaxincell``. Unsupported solver combinations fail early.
    ``longitudinal_gather='six_face'`` changes only ordinary E_x interpolation
    with quintic particles. This optional stencil is not an exact energy method
    or a general three-velocity momentum-conservation theorem.
    Experimental ``clock='anchored'`` evaluates absolute time from a retained
    origin and step counter, including prescribed forcing at the midpoint.
    """

    plasma: Simulation
    dark: object
    longitudinal_gather: str = "average"
    clock: str = "accumulated"

    def __post_init__(self):
        d, s = self.plasma.domain, self.plasma.solver
        if self.clock not in ("accumulated", "anchored"):
            raise ValueError("clock must be 'accumulated' or 'anchored'")
        if self.longitudinal_gather not in ("average", "six_face"):
            raise ValueError("longitudinal_gather must be 'average' or 'six_face'")
        if self.longitudinal_gather == "six_face" and getattr(s, "shape_order", 2) != 5:
            raise ValueError("six_face longitudinal_gather requires shape_order=5")
        if not isinstance(self.dark, (DarkField, PrescribedDrive)):
            raise TypeError("dark must be DarkField or PrescribedDrive")
        if (d.particle_bc != (0, 0) or d.field_bc != (0, 0) or s.algorithm != "explicit"
                or s.model != "electromagnetic" or s.field_solver != "ampere"
                or s.filter_passes or self.plasma.collisions is not None or self.plasma.sources):
            raise ValueError("dark runs require periodic explicit electromagnetic Ampere PIC, "
                             "with no filtering, sources or collisions")
        if self.plasma.external_E is not None:
            raise ValueError("prescribed ordinary external_E needs its own work ledger")
        if isinstance(self.dark, DarkField):
            _check_field_model(self.dark, d)
        else:
            _check_drive_model(self.dark)

    def initial_state(self, key):
        """Use parent loading and freeze its initial neutralizing background."""
        ordinary, extra = self.plasma.initial_state(key)
        background = -jnp.mean(ordinary.rho)
        if isinstance(self.dark, DarkField):
            shape = ordinary.E.shape
            A = jnp.zeros(shape) if self.dark.initial_A is None else jnp.asarray(self.dark.initial_A)
            E = self.dark.eta * ordinary.E if self.dark.initial_E is None else jnp.asarray(self.dark.initial_E)
            phi = jnp.zeros((shape[0],)) if self.dark.initial_phi is None else jnp.asarray(self.dark.initial_phi)
            E, correction = project_initial_electric(
                E, ordinary.rho, background, phi, self.plasma.domain.dx,
                self.dark.omega, self.dark.eta)
            B = curl_E(A, jnp.zeros(shape), self.plasma.domain.dx, (0, 0))
        else:
            E = B = A = phi = None
            correction = jnp.zeros(())
        state = DarkState(ordinary, E, B, A, phi, background, jnp.zeros(()),
                          jnp.zeros(()), jnp.zeros(()), correction,
                          jnp.zeros(()), jnp.zeros(()), jnp.zeros(()))
        if self.clock == "anchored":
            state = state.replace(clock_time=ordinary.time, clock_step=ordinary.steps)
        return _reset_diagnostics(state, self.plasma, self.dark, extra[0]), extra

    def continue_with_parameters(self, state):
        """Begin a new experiment at this state; reclose Gauss and reset all ledgers."""
        _check_clock(state, self)
        if isinstance(self.dark, DarkField):
            if state.E is None or state.A is None or state.phi is None:
                raise ValueError("continuation needs a complete Proca state")
            E, correction = project_initial_electric(
                state.E, state.ordinary.rho, state.background, state.phi,
                self.plasma.domain.dx, self.dark.omega, self.dark.eta)
            state = state.replace(E=E, initial_projection_norm=correction)
        elif state.E is not None:
            raise ValueError("continuation drive mode does not match the saved field state")
        state = _reset_diagnostics(state, self.plasma, self.dark, self.plasma.per_particle[0])
        return state

    def _step(self, state, extra):
        """Mirrored field halves around the parent's Boris push and two deposits."""
        p, d, model = self.plasma, self.plasma.domain, self.dark
        o, h, dx = state.ordinary, d.dt / 2, d.dx
        if self.clock == "anchored":
            _check_clock(state, self)
        m, q = extra
        v = p._velocity(o.u)
        charge = q * o.w
        closure = p._current_closure(charge * v[:, 0])
        rho_half, J1 = p._sources(o.x, v, charge, h, closure, o.rho)
        E, B = p._advance_fields(o.E, o.B, J1, h, rho_half, o.wall, True)
        gather_E = E
        if self.longitudinal_gather == "six_face":
            gather_E = E.at[:, 0].set(_six_face_filter(E[:, 0]))
        if isinstance(model, DarkField):
            E_D, phi = kick(state.E, state.B, state.A, state.phi, J1, h, dx, model.omega, model.eta)
            B_D, A = drift(E_D, state.B, state.A, phi, h, dx)
            work1 = -model.eta * h * dx * jnp.sum(J1 * (state.E + E_D) / 2)
            effective_E, effective_B = gather_E + model.eta * E_D, B + model.eta * B_D
        else:
            midpoint = (state.clock_time + lax.optimization_barrier((o.steps - state.clock_step + .5) * d.dt)
                        if self.clock == "anchored" else o.time + h)
            drive = model.at(midpoint)
            effective_E, effective_B = gather_E + model.eta * drive, B
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
        time = (state.clock_time + lax.optimization_barrier((o.steps - state.clock_step + 1) * d.dt)
                if self.clock == "anchored" else o.time + d.dt)
        ordinary = o.replace(E=E, B=B, x=x_half, u=u, rho=rho_next,
                             time=time, steps=o.steps + 1, moments=totals)
        kinetic = _kinetic_species(p, ordinary, m)
        ordinary_total = _ordinary_total(ordinary, kinetic, dx)
        if isinstance(model, DarkField):
            dark_total = energy(E_D, B_D, A, phi, dx, model.omega)
            balance = ordinary_total + dark_total - state.initial_ordinary - state.initial_dark
            dark_gauss = jnp.max(jnp.abs(gauss(E_D, phi, rho_next + state.background,
                                               dx, model.omega, model.eta)))
        else:
            balance = ordinary_total - work - state.initial_ordinary
            dark_gauss = state.max_dark_gauss
        ordinary_gauss = jnp.max(jnp.abs(divergence(E, dx)
                                         - (rho_next + state.background) / epsilon_0))
        next_state = state.replace(
            ordinary=ordinary, E=E_D, B=B_D, A=A, phi=phi, work=work,
            max_balance_error=jnp.maximum(state.max_balance_error, jnp.abs(balance)),
            max_ordinary_gauss=jnp.maximum(state.max_ordinary_gauss, ordinary_gauss),
            max_dark_gauss=jnp.maximum(state.max_dark_gauss, dark_gauss))
        return next_state, (x_next, v_new, o.w, E, B, (J1 + J2) / 2, rho_next, kinetic)

    def run(self, steps, seed=0, store_every=1, store_particles=True, state=None, verbose=False):
        """Run sparse histories; ``verbose`` uses the parent's host progress meter.

        Progress groups share the complete state and turn off under JAX tracing,
        leaving differentiated runs silent and the default path a single scan.
        """
        if steps < 1 or store_every < 1 or steps % store_every:
            raise ValueError("steps must be positive and divisible by store_every")
        carry, extra = self.initial_state(random.PRNGKey(seed)) if state is None else (state, self.plasma.per_particle)
        _check_clock(carry, self)
        traced = any(isinstance(leaf, jax.core.Tracer) for leaf in jax.tree.leaves((self, state, seed)))
        meter = None if traced else reporter(verbose, steps)
        histories, done = [], 0
        try:
            for count in _groups(steps // store_every, meter):
                carry, history = _advance(self, carry, extra, count, store_every, store_particles)
                histories.append(history)
                if meter is not None:
                    jax.block_until_ready(carry.ordinary.E)
                    done += count * store_every
                    meter(done, steps)
        finally:
            getattr(meter, "close", lambda: None)()
        history = _join(histories)
        x, v, w, E, B, J, rho, kinetic, wall, t, n, sigma, E_D, B_D, A, phi, work = history
        d, (m, q) = self.plasma.domain, extra
        ordinary = Output(t=t, steps=n, sigma=sigma, x=x, v=v, E=E, B=B, J=J, rho=rho,
                          grid=d.grid, dx=d.dx, dt=d.dt, length=d.length, charge=q, mass=m,
                          weight=w, wall=wall, moments=None,
                          species=jnp.concatenate([jnp.full((s.n,), i) for i, s in enumerate(self.plasma.species)]),
                          state=carry.ordinary, names=tuple(s.name for s in self.plasma.species),
                          counts=tuple(s.n for s in self.plasma.species),
                          relativistic=self.plasma.solver.relativistic, field_bc=d.field_bc)
        return DarkOutput(ordinary, E_D, B_D, A, phi, work, carry, self.dark, kinetic)


@partial(jax.jit, static_argnames=("chunks", "store_every", "store_particles"))
def _advance(sim, carry, extra, chunks, store_every, store_particles):
    placeholder = (carry.ordinary.x, sim.plasma._velocity(carry.ordinary.u), carry.ordinary.w,
                   carry.ordinary.E, carry.ordinary.B, jnp.zeros_like(carry.ordinary.E),
                   carry.ordinary.rho, _kinetic_species(sim.plasma, carry.ordinary, extra[0]))

    def advance(pair, _):
        return sim._step(pair[0], extra), None

    def chunk(state, _):
        (state, (x, v, w, E, B, J, rho, kinetic)), _ = lax.scan(
            advance, (state, placeholder), None, length=store_every)
        if not store_particles:
            x = v = w = None
        o = state.ordinary
        return state, (x, v, w, E, B, J, rho, kinetic, o.wall, o.time, o.steps, o.sigma,
                       state.E, state.B, state.A, state.phi, state.work)

    return lax.scan(chunk, carry, None, length=chunks)
