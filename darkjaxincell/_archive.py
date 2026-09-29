"""Native complete restarts, built on JAX-in-Cell's versioned archive."""

import numpy as np
import jax.numpy as jnp

from jaxincell import load_state as load_ordinary_state
from jaxincell import save_state as save_ordinary_state

from ._simulation import DarkField, DarkState


def save_state(path, state, simulation):
    """Save particles, fields, potentials, background, clock and work in one NPZ."""
    path = save_ordinary_state(path, state.ordinary, simulation.plasma)
    with np.load(path, allow_pickle=False) as ordinary:
        arrays = {name: ordinary[name] for name in ordinary.files}
    arrays["dark.mode"] = np.asarray("field" if isinstance(simulation.dark, DarkField) else "drive")
    for name in ("E", "B", "A", "phi", "background", "work"):
        value = getattr(state, name)
        if value is not None:
            arrays[f"dark.{name}"] = np.asarray(value)
    np.savez(path, **arrays)
    return path


def load_state(path, simulation):
    """Restore the exact discrete state and reject a mismatched physical mode."""
    ordinary = load_ordinary_state(path, simulation.plasma)
    with np.load(path, allow_pickle=False) as data:
        mode = "field" if isinstance(simulation.dark, DarkField) else "drive"
        if "dark.mode" not in data or str(data["dark.mode"]) != mode:
            raise ValueError("archive dark mode does not match this simulation")
        values = [jnp.asarray(data[f"dark.{name}"]) if f"dark.{name}" in data else None
                  for name in ("E", "B", "A", "phi", "background", "work")]
    if any(value is None for value in values[-2:]):
        raise ValueError("incomplete dark archive: missing background or work")
    if mode == "field" and any(value is None for value in values[:4]):
        raise ValueError("incomplete dark archive: missing Proca field or potential")
    return DarkState(ordinary, *values)
