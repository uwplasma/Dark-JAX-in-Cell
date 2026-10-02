# 🌑 Dark-JAX-in-Cell

The ghostly wing of JAX-in-Cell: a separate, small package for periodic classical Maxwell–Proca PIC. The [README](https://github.com/uwplasma/Dark-JAX-in-Cell#readme) has installation and a first run.

For a file-driven run, install the package and use `darkjaxincell examples/input.toml --save artifacts/cold_run`. The input is a normal JAX-in-Cell TOML file with a final `[dark]` table. `--steps`, `--seed`, `--eta` and `--omega` override its common controls; the saved folder contains fields, the complete restart, input and provenance. Python callers can use `darkjaxincell.load_toml` and `DarkSimulation.run` directly. Quintic runs also accept `--longitudinal-gather six_face`; [the experimental reconstruction](performance.md#experimental-six-face-longitudinal-gather) gives its TOML/Python controls and conservation limits.

For an external waveform control, use `PrescribedDrive(eta, amplitude, 0., times=times)`: `times` is an increasing vector in seconds and `amplitude` contains bare electric-field samples in V/m with shape `(len(times), 3)`. Fields interpolate linearly at the particle push's midpoint, with constant endpoint continuation. Cover the entire run and refine the table cadence. In TOML, `[dark]` uses `model="drive"`, `omega=0`, `phase=0`, `times=[...]` and `amplitude=[[Ex,Ey,Ez], ...]`. Complete waveform restarts retain both knots and values. With the default average gather they use dark format 3, and cosine drives use format 2; the optional six-face gather uses format 4 for either drive. Gradients pass through samples, coupling and knot times, subject to externally enforced knot order. This is an external-work experiment, with no evolved dark reservoir. [Independent impulse, work, derivative and restart checks](../tests/test_proca.py) exercise this path.

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
