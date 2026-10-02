"""Reduced periodic PIC ledgers for validation, using the production transitions.

Momentum includes the Proca scalar-potential flux divided by c squared.
These centre/face quadratures approximate continuum momentum; they are not an
assertion of exact discrete momentum conservation. No particle history is kept.
"""

from functools import partial
from contextlib import contextmanager
from importlib import metadata
import json
from pathlib import Path
import subprocess
import sys
from threading import Event, Thread
from time import perf_counter

import jax
import jax.numpy as jnp
import jaxincell
from jax import lax
from jaxincell import epsilon_0, speed_of_light as c
from jaxincell._core import deposit

from darkjaxincell import DarkField, DarkSimulation
from darkjaxincell._proca import divergence, energy, gauss


def parent_revision():
    """Actual imported checkout SHA, or the installed wheel's source commit."""
    root = Path(jaxincell.__file__).resolve().parents[1]
    if (root / '.git').exists():
        revision = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
        changed = subprocess.check_output(['git', '-C', str(root), 'status', '--porcelain',
                                           '--untracked-files=no'], text=True).strip()
        return revision + ('-dirty' if changed else '')
    try:
        direct = metadata.distribution('jaxincell').read_text('direct_url.json')
    except metadata.PackageNotFoundError:
        direct = None
    return json.loads(direct or '{}').get('vcs_info', {}).get('commit_id', 'unknown')


@contextmanager
def elapsed_progress(label):
    """Report long compilation/execution waits on the host, outside JAX traces."""
    stop, start = Event(), perf_counter()

    def report():
        while not stop.wait(30):
            print(f"{label}: {perf_counter() - start:.0f} s elapsed", file=sys.stderr, flush=True)

    worker = Thread(target=report, daemon=True)
    worker.start()
    try:
        yield
    finally:
        stop.set()
        worker.join()


def centred(face):
    """Periodic face-to-centre average, including the wrapped left face."""
    return (face + jnp.roll(face, 1, axis=0)) / 2


def snapshot(sim, state, background=None, mode=1):
    """Integer-time moments and continuum momentum, using the pusher's energy.

    ``spread`` is longitudinal lab-frame velocity-variance energy. It is not a
    relativistic thermodynamic temperature. Species may be numerical cohorts.
    """
    dark = isinstance(sim, DarkSimulation)
    p = sim.plasma if dark else sim
    o = state.ordinary if dark else state
    m, q = p.per_particle
    v = p._velocity(o.u)[:, 0]
    kinetic, spread, mean, rms = [], [], [], []
    offset = 0
    for species in p.species:
        s = slice(offset, offset + species.n)
        w = o.w[s]
        average = jnp.sum(w * v[s]) / jnp.sum(w)
        variance = jnp.sum(w * (v[s] - average)**2) / jnp.sum(w)
        kinetic.append(jnp.sum(w * p._kinetic(m[s], o.u[s])))
        spread.append(0.5 * jnp.sum(w * m[s]) * variance)
        mean.append(average)
        rms.append(jnp.sqrt(variance))
        offset += species.n
    dx = p.domain.dx
    electric = epsilon_0 * dx * jnp.sum(o.E**2) / 2
    magnetic = epsilon_0 * c**2 * dx * jnp.sum(o.B**2) / 2
    momentum = jnp.sum((m * o.w)[:, None] * o.u, axis=0)
    momentum += epsilon_0 * dx * jnp.sum(jnp.cross(centred(o.E), o.B), axis=0)
    massive, work, coherent, dark_mode, mean_D, mean_A = (jnp.zeros(()) for _ in range(6))
    background = (state.background if dark else -jnp.mean(o.rho)) if background is None else background
    ordinary_gauss = jnp.max(jnp.abs(divergence(o.E, dx) - (o.rho + background) / epsilon_0))
    dark_gauss = jnp.zeros(())
    if dark:
        work = state.work
        if isinstance(sim.dark, DarkField):
            model = sim.dark
            massive = energy(state.E, state.B, state.A, state.phi, dx, model.omega)
            coherent = .5 * epsilon_0 * p.domain.length * (
                jnp.sum(jnp.mean(state.E, axis=0)**2 + c**2 * jnp.mean(state.B, axis=0)**2)
                + model.omega**2 * (jnp.sum(jnp.mean(state.A, axis=0)**2) + jnp.mean(state.phi)**2 / c**2))
            dark_mode = jnp.fft.fft(state.E[:, 0])[mode] / p.domain.cells
            mean_D, mean_A = jnp.mean(state.E[:, 0]), jnp.mean(state.A[:, 0])
            momentum += epsilon_0 * dx * jnp.sum(jnp.cross(centred(state.E), state.B), axis=0)
            momentum += epsilon_0 * model.omega**2 / c**2 * dx * jnp.sum(
                state.phi[:, None] * centred(state.A), axis=0)
            dark_gauss = jnp.max(jnp.abs(gauss(state.E, state.phi, o.rho + background,
                                               dx, model.omega, model.eta)))
            work = jnp.zeros(())  # internal transfer, not an external source
    return dict(t=o.time, electric=electric, magnetic=magnetic, dark=massive,
                kinetic=jnp.stack(kinetic), spread=jnp.stack(spread),
                mean=jnp.stack(mean), rms=jnp.stack(rms), momentum=momentum,
                charge=jnp.sum(q * o.w), grid_charge=dx * jnp.sum(o.rho),
                balance=electric + magnetic + massive + sum(kinetic) - work,
                work=state.work if dark else jnp.zeros(()),
                ordinary_gauss=ordinary_gauss, dark_gauss=dark_gauss,
                mean_E=jnp.mean(o.E[:, 0]), mode_E=jnp.fft.fft(o.E[:, 0])[mode] / p.domain.cells,
                dark_coherent=coherent, dark_mode_E=dark_mode,
                mean_D=mean_D, mean_A=mean_A,
                max_speed=jnp.max(jnp.linalg.norm(p._velocity(o.u), axis=1)))


def density_rms(sim, state):
    """Species density contrast at the numerical grid scale (not temperature)."""
    p = sim.plasma if isinstance(sim, DarkSimulation) else sim
    o = state.ordinary if isinstance(sim, DarkSimulation) else state
    x = (o.x - p.domain.dt / 2 * p._velocity(o.u)
         if p.solver.algorithm == "explicit" else o.x)
    offset, values = 0, []
    weighting = partial(deposit, shape_order=5) if getattr(p.solver, 'shape_order', 2) == 5 else deposit
    for species in p.species:
        s = slice(offset, offset + species.n)
        density = weighting(x[s, 0], o.w[s], p.domain.grid[0], p.domain.dx, p.domain.cells, (0, 0))
        values.append(jnp.std(density) / jnp.mean(density))
        offset += species.n
    return jnp.stack(values)


def coarse_spread(sim, state, scales, groups=None, *, with_density=False):
    """Longitudinal random energy after Gaussian smoothing at physical lengths.

    ``groups`` joins numerical populations of the same physical species before
    subtracting their local flow. This lab-frame variance is not relativistic
    temperature; coherent motion below the smoothing scale remains included.
    Group consistency is checked on the host with fixed species definitions.
    ``with_density`` also returns std/mean of density at each physical scale.
    """
    p = sim.plasma if isinstance(sim, DarkSimulation) else sim
    o = state.ordinary if isinstance(sim, DarkSimulation) else state
    groups = tuple((i,) for i in range(len(p.species))) if groups is None else groups
    x = (o.x - p.domain.dt / 2 * p._velocity(o.u)
         if p.solver.algorithm == "explicit" else o.x)
    velocity = p._velocity(o.u)[:, 0]
    weighting = partial(deposit, shape_order=5) if getattr(p.solver, 'shape_order', 2) == 5 else deposit
    moments, offset = [], 0
    for species in p.species:
        s = slice(offset, offset + species.n)
        moments.append(jnp.stack([
            weighting(x[s, 0], o.w[s] * velocity[s]**power, p.domain.grid[0], p.domain.dx, p.domain.cells, (0, 0))
            for power in range(3)]))
        offset += species.n
    k = 2 * jnp.pi * jnp.fft.rfftfreq(p.domain.cells, p.domain.dx)
    energies, contrasts = [], []
    for group in groups:
        if not group or (len(group) > 1 and len({(p.species[i].mass, p.species[i].charge)
                                                for i in group}) != 1):
            raise ValueError("a physical-species group needs one mass and charge")
        spectrum = jnp.fft.rfft(sum(moments[i] for i in group), axis=-1)
        row, density_row = [], []
        for scale in scales:
            density, flow, second = jnp.fft.irfft(
                spectrum * jnp.exp(-.5 * (k * scale)**2), n=p.domain.cells, axis=-1)
            variance = second - flow**2 / jnp.where(density > 0, density, 1.)
            row.append(.5 * p.species[group[0]].mass * p.domain.dx * jnp.sum(variance))
            density_row.append(jnp.std(density) / jnp.mean(density))
        energies.append(jnp.stack(row))
        contrasts.append(jnp.stack(density_row))
    result = jnp.stack(energies)
    return (result, jnp.stack(contrasts)) if with_density else result


@partial(jax.jit, static_argnames=("steps", "stride", "pump"))
def measured_run(sim, initial, steps, stride, reference=None, scales=None, mode=1, pump=False):
    """Sparse scalar histories and all-step maxima; valid for neutral closed boxes.

    A neutral homogeneous prescribed drive has zero total external impulse.
    For other external forces the momentum ledger needs their impulse too.
    A fixed neutralizing background can exchange momentum with the fields; the
    particle-plus-field momentum alone is then not a closed-system invariant.
    Maxima are absolute SI values ordered as energy/work, momentum, particle
    charge, continuity, ordinary Gauss, dark Gauss, grid charge, dark-sector
    work balance, and ordinary-sector work balance.
    Pass the original ``snapshot`` as ``reference`` across fixed compiled blocks
    to keep global defects; optional ``scales`` adds Gaussian local moments.
    ``pump=True`` additionally retains every integer-time mean D/A and species
    velocity, independently of the more expensive scalar/moment sampling stride.
    """
    if steps < 1 or stride < 1 or steps % stride:
        raise ValueError("steps must be positive and divisible by stride")
    dark = isinstance(sim, DarkSimulation)
    reservoir = dark and isinstance(sim.dark, DarkField)
    if pump and not reservoir:
        raise ValueError("dense pump histories require a dynamical dark field")
    p = sim.plasma if dark else sim
    if (p.domain.field_bc != (0, 0) or p.domain.particle_bc != (0, 0)
            or p.external_E is not None or p.external_B is not None or p.collisions is not None or p.sources):
        raise ValueError("reduced ledgers require periodic collisionless PIC without external parent fields")
    reference = snapshot(sim, initial) if reference is None else reference
    background = initial.background if dark else -reference["grid_charge"] / p.domain.length
    extra = p.per_particle
    step = (sim._step if dark else p._implicit_step if p.solver.algorithm == "implicit"
            else p._explicit_step)

    def one(carry, _):
        state, maxima = carry
        before = state.ordinary if dark else state
        state, output = step(state, extra)
        after = state.ordinary if dark else state
        values = snapshot(sim, state, background)
        continuity = (after.rho - before.rho) / p.domain.dt + divergence(output[5], p.domain.dx)
        dark_work = (values["dark"] - reference["dark"] - values["work"] + reference["work"]
                     if reservoir else jnp.zeros(()))
        ordinary_work = (values["balance"] - reference["balance"] - values["dark"] + reference["dark"]
                         + values["work"] - reference["work"] if reservoir
                         else values["balance"] - reference["balance"])
        defect = jnp.array([jnp.abs(values["balance"] - reference["balance"]),
                            jnp.max(jnp.abs(values["momentum"] - reference["momentum"])),
                            jnp.abs(values["charge"] - reference["charge"]),
                            jnp.max(jnp.abs(continuity)), values["ordinary_gauss"], values["dark_gauss"],
                            jnp.abs(values["grid_charge"] - reference["grid_charge"]),
                            jnp.abs(dark_work), jnp.abs(ordinary_work)])
        mean_pump = {key: values[key] for key in ("t", "mean", "mean_D", "mean_A")} if pump else None
        return (state, jnp.maximum(maxima, defect)), mean_pump

    def sample(state):
        values = {**snapshot(sim, state, background, mode), "density_rms": density_rms(sim, state)}
        if scales is not None:
            values["local_spread"], values["local_density_rms"] = coarse_spread(
                sim, state, scales, with_density=True)
        return values

    def chunk(carry, _):
        carry, dense = lax.scan(one, carry, None, length=stride)
        state = carry[0]
        return carry, (sample(state), dense)

    maxima = jnp.array([0., 0., 0., 0., reference["ordinary_gauss"], reference["dark_gauss"], 0., 0., 0.])
    (final, maxima), (history, dense) = lax.scan(chunk, (initial, maxima), None, length=steps // stride)
    first = sample(initial)
    history = jax.tree.map(lambda a, b: jnp.concatenate((a[None], b)), first, history)
    if pump:
        history.update({"pump_" + key: jnp.concatenate((first[key][None], value.reshape((-1, *value.shape[2:]))))
                        for key, value in dense.items()})
    return final, history, maxima
