# 🌑 Dark-JAX-in-Cell

The ghostly wing of JAX-in-Cell: a separate, small package for periodic classical Maxwell–Proca PIC. The [README](https://github.com/uwplasma/Dark-JAX-in-Cell#readme) has installation and a first run.

For a file-driven run, install the package and use `darkjaxincell examples/input.toml --save artifacts/cold_run`. The input is a normal JAX-in-Cell TOML file with a final `[dark]` table. `--steps`, `--seed`, `--eta` and `--omega` override its common controls; the saved folder contains fields, the complete restart, input and provenance. Python callers can use `darkjaxincell.load_toml` and `DarkSimulation.run` directly.

Research examples place editable inputs after their imports and run with `python examples/dark_photon.py`. Defaults are short smoke cases; `full=True` selects the documented physical study. The README links the producer below every figure and movie. Batch callers can use `runpy.run_path("examples/dark_photon.py", run_name="__main__", init_globals={"full": True})`; the full evidence driver uses fresh processes sequentially. `python docs/scripts/make_all.py` refreshes documentation from archived records by default; set `records_only=False` there to rerun physics. Progress uses the parent's host meter and turns off under JAX tracing.

```{toctree}
:maxdepth: 2

physics
validation
optimization
profile
kinetic
performance
```
