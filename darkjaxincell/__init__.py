"""Dark-JAX-in-Cell: a slim Maxwell–Proca companion to JAX-in-Cell."""

from ._simulation import (DarkField, DarkOutput, DarkSimulation, DarkState, PrescribedDrive,
                          project_initial_electric)
from ._proca import energy, gauss
from ._archive import load_for_continuation, load_state, save_state
from ._style import midnight
from ._cli import load_toml, main

__all__ = ["DarkField", "DarkOutput", "DarkSimulation", "DarkState", "PrescribedDrive", "energy", "gauss",
           "load_state", "load_for_continuation", "save_state", "project_initial_electric",
           "midnight", "load_toml", "main"]
