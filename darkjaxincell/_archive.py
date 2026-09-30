"""Native complete restarts, built on JAX-in-Cell's versioned archive."""

import numpy as np
import jax.numpy as jnp

from jaxincell import load_state as load_ordinary_state
from jaxincell import save_state as save_ordinary_state

from ._simulation import DarkField, DarkState

FORMAT = 2


def _model_metadata(simulation):
    """Physical parameters that would change a continued discrete experiment."""
    plasma, model = simulation.plasma, simulation.dark
    domain = plasma.domain
    data = {"dark.format": np.asarray(FORMAT),
            "dark.mode": np.asarray("field" if isinstance(model, DarkField) else "drive"),
            "dark.omega": np.asarray(model.omega), "dark.eta": np.asarray(model.eta),
            "dark.cells": np.asarray(domain.cells),
            "dark.length": np.asarray(domain.length), "dark.length_y": np.asarray(domain.length_y),
            "dark.length_z": np.asarray(domain.length_z), "dark.dt": np.asarray(domain.dt),
            "dark.relativistic": np.asarray(plasma.solver.relativistic),
            "dark.mass": np.asarray([species.mass for species in plasma.species]),
            "dark.charge": np.asarray([species.charge for species in plasma.species]),
            "dark.density": np.asarray([species.density for species in plasma.species]),
            "dark.has_external_B": np.asarray(plasma.external_B is not None)}
    if plasma.external_B is not None:
        data["dark.external_B"] = np.asarray(plasma.external_B)
    if not isinstance(model, DarkField):
        data.update({"dark.amplitude": np.asarray(model.amplitude),
                     "dark.phase": np.asarray(model.phase)})
    return data


def save_state(path, state, simulation):
    """Save particles, fields, potentials, background, clock and work in one NPZ."""
    path = save_ordinary_state(path, state.ordinary, simulation.plasma)
    with np.load(path, allow_pickle=False) as ordinary:
        arrays = {name: ordinary[name] for name in ordinary.files}
    arrays.update(_model_metadata(simulation))
    for name in ("E", "B", "A", "phi", "background", "work", "initial_ordinary",
                 "initial_dark", "initial_projection_norm", "max_balance_error",
                 "max_ordinary_gauss", "max_dark_gauss"):
        value = getattr(state, name)
        if value is not None:
            arrays[f"dark.{name}"] = np.asarray(value)
    np.savez(path, **arrays)
    return path


def load_state(path, simulation):
    """Restore only an identical physical experiment and complete diagnostic state."""
    ordinary = load_ordinary_state(path, simulation.plasma)
    with np.load(path, allow_pickle=False) as data:
        expected = _model_metadata(simulation)
        for key, wanted in expected.items():
            if key not in data or not np.array_equal(data[key], wanted):
                raise ValueError(f"archive {key} does not match this simulation")
        names = ("E", "B", "A", "phi", "background", "work", "initial_ordinary",
                 "initial_dark", "initial_projection_norm", "max_balance_error",
                 "max_ordinary_gauss", "max_dark_gauss")
        values = [jnp.asarray(data[f"dark.{name}"]) if f"dark.{name}" in data else None
                  for name in names]
        mode = str(expected["dark.mode"])
    for name, value in zip(names, values):
        if value is None and (mode == "field" or name not in names[:4]):
            raise ValueError(f"incomplete dark archive: missing dark.{name}")
    cells = simulation.plasma.domain.cells
    for name, value, shape in (("E", values[0], (cells, 3)), ("B", values[1], (cells, 3)),
                               ("A", values[2], (cells, 3)), ("phi", values[3], (cells,))):
        if value is not None and value.shape != shape:
            raise ValueError(f"archive dark.{name} must have shape {shape}")
    for name, value in zip(names[4:], values[4:]):
        if value.shape != ():
            raise ValueError(f"archive dark.{name} must be scalar")
    return DarkState(ordinary, *values)


def load_for_continuation(path, previous, following):
    """Explicit parameter jump: verify both models, reclose Gauss and start a new ledger."""
    state = load_state(path, previous)
    old, new = _model_metadata(previous), _model_metadata(following)
    allowed = {"dark.omega", "dark.eta", "dark.amplitude", "dark.phase"}
    for key in old.keys() | new.keys():
        if key not in allowed and (key not in old or key not in new
                                   or not np.array_equal(old[key], new[key])):
            raise ValueError(f"continuation changes {key}; only dark parameters may change")
    return following.continue_with_parameters(state)
