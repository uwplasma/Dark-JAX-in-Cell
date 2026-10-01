"""TOML and command-line entrance to the crypt, using JAX-in-Cell's loader."""

import argparse
from pathlib import Path
import re
import tempfile

import jax.numpy as jnp
import numpy as np
from jaxincell import epsilon_0, load_toml as load_plasma_toml, save_run

from ._archive import save_state
from ._simulation import DarkField, DarkSimulation, PrescribedDrive


def load_toml(path, *, eta=None, omega=None):
    """Read a parent-compatible plasma file with a final ``[dark]`` table.

    The parent parser still validates every plasma and run key. Keeping the dark
    table last lets us hand its unchanged prefix to that parser without a TOML
    writer or a second set of plasma constructors.
    """
    try:
        import tomllib
    except ModuleNotFoundError:  # pragma: no cover - Python 3.10
        import tomli as tomllib

    source = Path(path).read_text()
    match = re.search(r"(?m)^\[dark\][ \t]*(?:\#.*)?$", source)
    if match is None:
        raise ValueError("input needs a final [dark] table")
    prefix, section = source[:match.start()], source[match.start():]
    if re.search(r"(?m)^\s*\[", source[match.end():]):
        raise ValueError("[dark] must be the final table")
    dark = tomllib.loads(section)["dark"]
    unknown = set(dark) - {"model", "omega", "eta", "initial_E", "initial_A", "initial_phi", "amplitude", "phase"}
    if unknown:
        raise ValueError(f"unknown [dark] keys: {', '.join(sorted(unknown))}")
    with tempfile.NamedTemporaryFile(mode="w", suffix=".toml") as plasma_file:
        plasma_file.write(prefix)
        plasma_file.flush()
        plasma, run = load_plasma_toml(plasma_file.name)
    unsupported = set(run) - {"steps", "seed", "store_every", "store_particles"}
    if unsupported:
        raise ValueError(f"unsupported dark [run] keys: {', '.join(sorted(unsupported))}")
    coupling = dark.get("eta") if eta is None else eta
    frequency = dark.get("omega") if omega is None else omega
    if coupling is None or frequency is None:
        raise ValueError("[dark] needs eta and omega")
    return DarkSimulation(plasma, _dark_model(dark, plasma.domain.cells, coupling, frequency)), run


def _dark_model(dark, cells, coupling, frequency):
    """Make the selected dark field while rejecting mode-specific stray keys."""
    if dark.get("model", "field") == "field":
        if "amplitude" in dark or "phase" in dark:
            raise ValueError("amplitude and phase belong to model = 'drive'")
        shape = (cells, 3)
        fields = {name: jnp.broadcast_to(jnp.asarray(dark[name], float), shape)
                  for name in ("initial_E", "initial_A") if name in dark}
        phi = (jnp.broadcast_to(jnp.asarray(dark["initial_phi"], float), (shape[0],))
               if "initial_phi" in dark else None)
        return DarkField(frequency, coupling, initial_phi=phi, **fields)
    elif dark["model"] == "drive":
        if any(name in dark for name in ("initial_E", "initial_A", "initial_phi")):
            raise ValueError("initial dark fields belong to model = 'field'")
        if "amplitude" not in dark:
            raise ValueError("a prescribed drive needs amplitude")
        return PrescribedDrive(coupling, jnp.asarray(dark["amplitude"], float),
                               frequency, dark.get("phase", 0.0))
    raise ValueError("[dark] model must be 'field' or 'drive'")


def main(argv=None):
    """Run a TOML experiment, with flags overriding its common controls."""
    parser = argparse.ArgumentParser(prog="darkjaxincell")
    parser.add_argument("input", type=Path)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--eta", type=float)
    parser.add_argument("--omega", type=float)
    parser.add_argument("--save", type=Path, metavar="DIR")
    args = parser.parse_args(argv)
    sim, run = load_toml(args.input, eta=args.eta, omega=args.omega)
    settings = dict(run)
    settings["steps"] = args.steps if args.steps is not None else settings.get("steps", 100)
    settings["seed"] = args.seed if args.seed is not None else settings.get("seed", 0)
    out = sim.run(**settings, verbose=True)
    ledger = out.energy()
    balance = (ledger["closed_energy_error"] if isinstance(sim.dark, DarkField)
               else ledger["closed_balance_error"])
    initial = out.state.initial_ordinary + out.state.initial_dark
    scale = jnp.maximum(jnp.abs(initial), jnp.abs(out.work[-1]))
    ex = out.ordinary.E[:, :, 0]
    ordinary_gauss = ((ex - jnp.roll(ex, 1, axis=1)) / out.ordinary.dx
                      - (out.ordinary.rho + out.state.background) / epsilon_0)
    result = {"final_time_s": float(out.ordinary.t[-1]),
              "ordinary_gauss_max_V_m2": float(jnp.max(jnp.abs(ordinary_gauss))),
              "dark_gauss_max_V_m2": (float(jnp.max(jnp.abs(out.dark_gauss())))
                                      if isinstance(sim.dark, DarkField) else None),
              "energy_drift": float(balance[-1] / jnp.maximum(scale, jnp.finfo(scale.dtype).tiny)),
              "max_balance_error_J_m2": float(out.state.max_balance_error),
              "initial_energy_J_m2": float(initial)}
    print(f"🦇 {settings['steps']} steps; relative energy balance {result['energy_drift']}; "
          f"dark Gauss {result['dark_gauss_max_V_m2']}")
    if args.save is not None:
        args.save.mkdir(parents=True, exist_ok=True)
        record = {"input": str(args.input), **settings,
                  "dark_model": "field" if isinstance(sim.dark, DarkField) else "drive",
                  "eta": float(sim.dark.eta), "omega_rad_s": float(sim.dark.omega)}
        save_run(args.save, "dark_cli", record, result,
                 t=out.ordinary.t, E=out.ordinary.E, B=out.ordinary.B,
                 E_dark=out.E if out.E is not None else np.empty(0),
                 B_dark=out.B if out.B is not None else np.empty(0))
        save_state(args.save / "restart.npz", out.state, sim)
        (args.save / args.input.name).write_text(args.input.read_text())
    return 0
