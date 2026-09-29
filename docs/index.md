# 🌑 Dark-JAX-in-Cell

The ghostly wing of JAX-in-Cell: a separate, small package for periodic classical Maxwell–Proca PIC. The [README](https://github.com/uwplasma/Dark-JAX-in-Cell#readme) has installation and a first run.

For a file-driven run, install the package and use `darkjaxincell examples/input.toml --save artifacts/cold_run`. The input is a normal JAX-in-Cell TOML file with a final `[dark]` table. `--steps`, `--seed`, `--eta` and `--omega` override its common controls; the saved folder contains fields, the complete restart, input and provenance. Python callers can use `darkjaxincell.load_toml` and `DarkSimulation.run` directly.

```{toctree}
:maxdepth: 2

physics
validation
optimization
profile
kinetic
performance
```
