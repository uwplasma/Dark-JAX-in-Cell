# 🌑 Dark-JAX-in-Cell

[![License: MIT](https://img.shields.io/github/license/uwplasma/Dark-JAX-in-Cell?color=6b46a8&labelColor=171724)](LICENSE)
[![Last commit](https://img.shields.io/github/last-commit/uwplasma/Dark-JAX-in-Cell?color=bd93f9&labelColor=171724)](https://github.com/uwplasma/Dark-JAX-in-Cell/commits/main)
[![Python](https://img.shields.io/badge/Python-3.10%2B-bd93f9?labelColor=171724)](pyproject.toml)
[![Parent](https://img.shields.io/badge/JAX--in--Cell-research--release-50fae4?labelColor=171724)](https://github.com/uwplasma/JAX-in-Cell/pull/42)

**A small, dark companion to [JAX-in-Cell](https://github.com/uwplasma/JAX-in-Cell).** It evolves a classical massive vector field beside ordinary Maxwell fields and charged particles in periodic 1D3V plasma. The ghost carries three polarizations, its scalar and vector potentials, and its own energy. The particles feel the combined Lorentz force; both fields respond to the same physical current.

![Cold ordinary and dark field exchange against an independent matrix exponential](docs/_static/figures/cold_exchange/figure.png)

*Full preset: homogeneous cold plasma, canonical coupling 0.05, and a fixed initial dark electric amplitude. Solid lines are the independent cold matrix-exponential reference; dashed lines are PIC. This is coherent field exchange, not outgoing radiation or nonlinear conversion.*

![Fixed-column slab and held-out outgoing photon fractions](docs/_static/figures/profile_design/figure.png)

*Full preset: a traveling dark packet crosses a neutral electron–proton slab. The profile is constrained to fixed column, support and peak; outgoing photon flux is compared with an independent cold scattering solve and held-out/refined PIC replay.*

## Features

- **Dynamical Maxwell–Proca:** longitudinal and both transverse polarizations, compatible Yee operators, both Gauss laws, and a shared fixed neutralizing background.
- **Prescribed drive:** a traceable homogeneous sinusoid with accumulated external work. This mode has no simulated dark reservoir.
- **Differentiable controls:** coupling, mass frequency, initial arrays, particle loading, and fixed-time output remain JAX leaves.
- **Complete restart:** the native archive carries particles, fields, potentials, background, absolute clock, and accumulated work.
- **Known-answer optimization:** a fixed-window density calibration differentiates through particle loading and the complete PIC recurrence.
- **Cold oblique 3V check:** all velocity and field components compared with an independent magnetized fluid matrix.
- **One mixed kinetic mode:** a seeded, current-neutral longitudinal Maxwellian check against a Vlasov–Proca root, with its fit uncertainty recorded.
- **Current-neutral instabilities:** cold mixed two-stream and bi-Maxwellian transverse anisotropy growth checks against independent determinants, with declared fit windows.
- **Constrained conversion design:** outgoing photon flux from a traveling dark packet, a fixed-column neutral slab, independent cold scattering, held-out/refined replay, and a finite-amplitude kinetic comparison.
- **Slim dependency:** imports JAX-in-Cell's loading, deposition, gathering, Boris push, ordinary field step, diagnostics, and archive machinery from its reviewed `research-release` SHA.

The library currently supports **periodic explicit electromagnetic Ampere runs without filters, collisions, or particle sources**. Unsupported combinations fail at construction. Broader kinetic instability surveys, nonlinear external-drive reproduction, other profile families, and full performance comparisons remain open validation work; see [validation and limits](docs/validation.md).

## Install

```sh
git clone https://github.com/uwplasma/Dark-JAX-in-Cell.git
cd Dark-JAX-in-Cell
python -m pip install -e '.[test]'
```

The install pins JAX-in-Cell to the reviewed `research-release` commit `83d327118163833f93e2588edcb5029241f6ba2a`. Its parent [draft PR #42](https://github.com/uwplasma/JAX-in-Cell/pull/42) remains open. Python 3.10 or newer is required.

## Use

```python
import jax.numpy as jnp
from jaxincell import Domain, Simulation, Species
from darkjaxincell import DarkField, DarkSimulation

domain = Domain(length=1.0, cells=32, dt_over_dx_c=0.2)
plasma = Simulation(domain, (Species.electrons(128, density=1e14),))
ghost = DarkField(omega=1e9, eta=0.05,
                  initial_E=jnp.tile(jnp.array([0.0, 1e-5, 0.0]), (domain.cells, 1)))
output = DarkSimulation(plasma, ghost).run(100)
print("🦇 dark Gauss residual:", jnp.max(jnp.abs(output.dark_gauss())))
print("🌘 total energy (J/m²):", output.energy()["total_with_dark"][-1])
```

`omega` is the **canonical rest frequency in rad/s**, `eta` is the exact current-coupling ratio, electric fields are V/m, `A` is V s/m, and `phi` is V. Explicit initial dark arrays must satisfy the Proca Gauss constraint. [The physics guide](docs/physics.md) gives the basis, units, equations, staggering, and energy ledger.

## Reproduce the current evidence

```sh
python -m pytest -q
python examples/dark_photon.py --quick
python examples/dark_drive.py
python examples/dark_plasma.py
python examples/dark_kinetic.py
python examples/dark_instabilities.py two-stream
python examples/dark_instabilities.py weibel
python examples/dark_null.py
python examples/optimize_dark_photon.py --quick
python examples/dark_profile.py
python docs/scripts/make_all.py
python -m sphinx -b html -W docs docs/_build/html
```

Quick presets are smoke runs. `make_all.py` runs the **full** cold, prescribed-drive, oblique 3V, mixed-kinetic, two-stream, transverse-anisotropy, homogeneous-null, density-calibration and slab-design presets and writes their figures, data, measured numbers, and provenance together. The [validation page](docs/validation.md) separates measured checks from pending research claims.

## Credit and license

MIT licensed, like its parent. JAX-in-Cell and its inherited human contributors retain their upstream authorship and licenses. See its [research-release source](https://github.com/uwplasma/JAX-in-Cell/tree/83d327118163833f93e2588edcb5029241f6ba2a) and our [source](https://github.com/uwplasma/Dark-JAX-in-Cell).
