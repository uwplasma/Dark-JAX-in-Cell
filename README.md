# 🌑 Dark-JAX-in-Cell

[![License: MIT](https://img.shields.io/github/license/uwplasma/Dark-JAX-in-Cell?color=6b46a8&labelColor=171724)](LICENSE)
[![Midnight checks](https://github.com/uwplasma/Dark-JAX-in-Cell/actions/workflows/test.yml/badge.svg?branch=main)](https://github.com/uwplasma/Dark-JAX-in-Cell/actions/workflows/test.yml)
[![Last commit](https://img.shields.io/github/last-commit/uwplasma/Dark-JAX-in-Cell?color=bd93f9&labelColor=171724)](https://github.com/uwplasma/Dark-JAX-in-Cell/commits/main)
[![Python](https://img.shields.io/badge/Python-3.10%2B-bd93f9?labelColor=171724)](pyproject.toml)
[![Parent](https://img.shields.io/badge/JAX--in--Cell-research--release-50fae4?labelColor=171724)](https://github.com/uwplasma/JAX-in-Cell/pull/42)

**Differentiable Maxwell–Proca particle-in-cell simulation on CPUs and GPUs.** Dark-JAX-in-Cell adds a massive vector field to [JAX-in-Cell](https://github.com/uwplasma/JAX-in-Cell)'s 1D3V electromagnetic PIC. Charged particles create both fields and feel their combined Lorentz force. The parent supplies loading, charge-conserving deposition, gathering, Boris pushing, ordinary Maxwell evolution, diagnostics and restart machinery; the extra physics stays in this small companion package.

<img src="docs/_static/movies/two_stream/figure.webp" width="900" alt="Matched two-stream phase-space roll-up, ordinary and dark electric fields, and energy">

*Two cold electron streams start from identical particle arrays in both solvers. This 128-cell [movie](docs/_static/movies/two_stream/run.json) advances 262,144 particles (2,048 per cell) through $\omega_pt=120$, plots 8,192 per panel, and stores 6.3 frames per plasma period. Its 120-frame WebP is 2.9 MB. It illustrates trapping; the growth fit and long-time convergence limits use the separate runs below.*

*The two-stream and bump movies are longitudinal, so their magnetic fields vanish by symmetry. The [slab movie](#a-finite-dark-packet-crosses-a-designed-slab) shows the evolving ordinary and dark electric **and** magnetic waves.*

## Features

- Self-consistent longitudinal and transverse Maxwell–Proca fields, with potentials, both Gauss laws and a closed energy ledger.
- End-to-end JAX gradients through particle loading, fields and trajectories; CPU and GPU execution.
- TOML, CLI and Python runs with complete particle and field restart; matched parent controls and independent analytic checks.

## Install and run

```sh
git clone https://github.com/uwplasma/Dark-JAX-in-Cell.git
cd Dark-JAX-in-Cell
python -m pip install -e .
darkjaxincell examples/input.toml --steps 160 --save artifacts/cold_run
```

The default JAX installation runs on a CPU. For a GPU, install the appropriate accelerator-enabled [JAX wheel](https://docs.jax.dev/en/latest/installation.html) first; `jax.devices()` shows the selected backend. The forward solver and a density gradient have been exercised on an NVIDIA RTX A4000 ([device smoke record](docs/_static/figures/gpu_smoke/run.json)). The [TOML input](examples/input.toml) uses JAX-in-Cell's tables plus `[dark]`; CLI flags override steps, seed, mixing and mass frequency. `--save` writes a complete restart and provenance. From Python:

```python
from darkjaxincell import load_toml

simulation, run = load_toml("examples/input.toml")
output = simulation.run(**run)
print("🌘 dark Gauss residual:", abs(output.dark_gauss()).max())
print("🦇 closed energy:", output.energy()["total_with_dark"][-1])
```

## Equations and scope

With mixing strength $\eta$ and dark rest frequency $\Omega_D$, the massive field obeys

$$
\begin{aligned}
\partial_t\mathbf E_D &= c^2\nabla\times\mathbf B_D+\Omega_D^2\mathbf A_D-\eta\mathbf J/\epsilon_0,\\
\partial_t\mathbf B_D &= -\nabla\times\mathbf E_D,\\
\partial_t\mathbf A_D &= -\mathbf E_D-\nabla\phi_D,\\
\partial_t\phi_D &= -c^2\nabla\cdot\mathbf A_D,\\
\nabla\cdot\mathbf E_D+\Omega_D^2\phi_D/c^2 &= \eta\rho_{\rm total}/\epsilon_0.
\end{aligned}
$$

The ordinary fields obey $\partial_t\mathbf E=c^2\nabla\times\mathbf B-\mathbf J/\epsilon_0$, $\partial_t\mathbf B=-\nabla\times\mathbf E$ and $\nabla\cdot\mathbf E=\rho_{\rm total}/\epsilon_0$. Particles feel $q[\mathbf E+\eta\mathbf E_D+\mathbf v\times(\mathbf B+\eta\mathbf B_D)]$. Both Gauss laws use the same neutralizing background, and the physical mean current remains in both Ampère updates. For a closed run, the diagnostic includes particle kinetic energy and

$$
U_{\rm fields}=\frac{\epsilon_0}{2}\int\left(|\mathbf E|^2+c^2|\mathbf B|^2+|\mathbf E_D|^2+c^2|\mathbf B_D|^2+\Omega_D^2\left[|\mathbf A_D|^2+\phi_D^2/c^2\right]\right)dx.
$$

A prescribed dark drive is a separate external-force control with a work ledger; it has no finite dark reservoir. [Physics and staggering](docs/physics.md) give the full equations and units.

## Numerical method and constraints

The solver is periodic, explicit **1D3V PIC**: fields vary along one spatial coordinate, while particles carry three velocity components. At each step, JAX-in-Cell's quadratic, three-cell weighting transfers charge to the grid and gathers fields back to particles. Current deposition balances changing cell charge with current flowing across cell faces and preserves the box's physical mean current. A Boris push advances particles in the combined ordinary and dark Lorentz force.

The ordinary electric and magnetic fields follow staggered Ampère and Faraday updates. Dark $\mathbf E_D,\mathbf A_D$ live on grid faces and $\mathbf B_D,\phi_D$ at cell centres; symmetric Proca kick and drift updates surround the particle push. Initialization projects the longitudinal dark electric field onto Gauss' law and reports the correction size. Charge-conserving current and compatible grid derivatives then carry both Gauss constraints through the run, while the potential update maintains $\mathbf B_D=\nabla\times\mathbf A_D$. The energy ledger includes particle, Maxwell and full Proca field and potential energies, referenced to the actual initial state; all-step conservation and Gauss maxima survive sparse output and restart. The examples use collisionless particles and unsmoothed sources. [Discrete equations, stability and measured energy drift](docs/physics.md) describe the scheme in detail.

## Cold exchange: a known answer

A homogeneous transverse dark field drives a cold electron plasma. The full [example](examples/dark_photon.py) follows the ordinary and dark mean fields through $\omega_0t=20$ and compares both with an independent four-state matrix exponential. The largest field error is **$2.922\times10^{-4}$** of the initial dark field; closed-energy drift from the physical time-zero state is **$8.709\times10^{-5}$**. This checks the coupling, mean current and potential-energy ledger before kinetic effects enter. [Settings and arrays](docs/_static/figures/cold_exchange/run.json) are saved with the figure.

<img src="docs/_static/figures/cold_exchange/figure.png" width="800" alt="Cold Maxwell–Proca PIC fields against a matrix-exponential solution">

## Kinetic Landau response

A seeded Maxwellian tests a damped mode, using the same initial particles in the parent and mixed runs. For $s=\omega^2-c^2k^2$, the independent Vlasov–Proca root solves $(s-\Omega_D^2)(1+\chi_L)+\eta^2s\chi_L=0$. The full PIC fit gives $\omega/\omega_p=1.43207-0.14576i$ against the mixed root $1.43695-0.14186i$; the parent gives $1.41195-0.15338i$ against $1.41566-0.15336i$. The fitted envelope ends at a measured late floor, so it is not extended into the nonlinear/noisy tail. See the [full fit and limits](docs/kinetic.md) and [run record](docs/_static/figures/mixed_kinetic/run.json).

<img src="docs/_static/figures/mixed_kinetic/figure.png" width="800" alt="Matched Landau mode histories and independent kinetic root">

At the smaller thermal speed $\sigma/c=0.05$, a [three-resolution replay](docs/kinetic.md#a-physical-speed-landau-replay) holds the physical plasma and dark mass fixed. The independent parent/mixed roots are $1.41566-0.15336i$ and $1.43243-0.14580i$; the 128-cell, 160,000-particle early-window fits give $1.41008-0.15175i$ and $1.44225-0.13944i$. Complete-energy drift stays below $2.64\times10^{-7}$ there. Fit-window and resolution changes still exceed the precision needed to measure the predicted damping *difference*, so that correction remains unresolved. The [record](docs/_static/figures/physical_kinetic_128/run.json) includes the complex ordinary, dark and effective mode histories.

<img src="docs/_static/figures/physical_kinetic_128/figure.png" width="900" alt="Physical-speed matched Landau histories and complete particle plus field energy">

## Two streams: growth and a long-time limit

Two current-neutral cold electron beams grow at $0.33228\omega_p$ in JAX-in-Cell and $0.34069\omega_p$ with the dark field; independent cold roots give $0.33847$ and $0.34670$. A Coulomb-plus-Yukawa reference explains **99.679%** of the analytic rate shift, identifying screening as the dominant linear effect in this case. After trapping, a [five-case grid/particle/timestep check](docs/_static/figures/two_stream_saturation/run.json) finds dark closed-energy drift of **0.680%** at 64 cells and **0.190%** at 128 cells at fixed timestep. At $120\leq\omega_pt\leq200$, a [four-case replay](docs/_static/figures/two_stream_extended/run.json) finds that the dark-minus-parent first-mode RMS changes from **−5.2%** on 128 cells to **+14.9%** on 256 cells at the *same* timestep. Thus the late difference has not converged, even in sign; a small energy drift alone does not settle it. The [full comparison](docs/kinetic.md) states the loading and windows.

At 128 cells and the same timestep, the 262,144-particle movie loading can be compared with the full run's 16,000 particles. Parent/dark growth fits over $10\leq\omega_pt\leq20$ are **0.33541/0.34388** $\omega_p$, near **0.33435/0.34302** in the smaller run. The first-mode RMS differences are **0.09%/0.14%** over $40\leq\omega_pt\leq55$, then **10.0%/11.3%** over $80\leq\omega_pt\leq120$. The longer particle replay exposes late loading sensitivity; it does not settle the spatial-convergence warning above. The largest stored-frame change in complete dark-run energy is **0.188%**; its final change is **0.0046%**. [Movie measurements and arrays](docs/_static/movies/two_stream/run.json).

<img src="docs/_static/figures/two_stream_extended/figure.png" width="900" alt="Matched two-stream growth, energy and phase space through normalized time 200">

## Bump on tail: a warm kinetic comparison

The [bump example](examples/dark_bump.py) starts from the parent's two-Maxwellian beam setup, with a compensating bulk drift so the mean current vanishes. A 3% beam travels at $5v_{th}$; the seeded $k_5$ wave resonates with its tail. Both solvers advance the same loaded electrons. The reference evaluates the drifting-Maxwellian dielectric $\epsilon_L(\omega,k)$ independently of PIC and solves

$$
(s-\Omega_D^2)\epsilon_L+\eta^2s(\epsilon_L-1)=0,\qquad s=\omega^2-c^2k^2.
$$

The full replay uses 120,000 electrons on 128 cells and 240,000 on 256 cells, with a halved timestep. Its parent/dark growth fits are **0.15384/0.15280** and **0.15378/0.15256** $\omega_p$; independent roots give **0.15135/0.15392**. The predicted dark shift is below the fit uncertainty, so this run resolves the growing branch but not a mixing-induced rate change. Maximum sampled complete-energy drift falls from **0.093%** to **0.023%** on joint refinement. The final tail is broadened. The velocity distribution uses physical number weights: the 3% beam has one-third of the numerical markers, but 3% of the plotted number distribution. The [full record](docs/_static/figures/bump_on_tail/run.json) gives fit errors, Gauss residuals and settings. This extends [JAX-in-Cell's bump example](https://github.com/uwplasma/JAX-in-Cell/blob/83d327118163833f93e2588edcb5029241f6ba2a/examples/2_intermediate/bump_on_tail.py) with a finite field; the [movie](docs/_static/movies/bump_on_tail/run.json) shows both phase spaces, all longitudinal electric fields and energy on common axes. Its displayed dots sample represented number and are trajectories rather than density values.

<img src="docs/_static/movies/bump_on_tail/figure.webp" width="900" alt="Matched 120,000-particle bump-on-tail phase spaces and electric fields">

<img src="docs/_static/figures/bump_on_tail/figure.png" width="800" alt="Bump-on-tail growth, velocity distributions and complete-energy drift at two resolutions">

## Transverse anisotropy

The [Weibel example](examples/dark_instabilities.py) seeds a transverse magnetic mode in a current-neutral bi-Maxwellian with $T_z/T_x=4$. Its independent Vlasov–Proca root gives $\gamma/\omega_p=0.05113$; the 30,000- and 60,000-particle PIC fits give $0.05123$ and $0.05117$. The smaller predicted dark-minus-parent shift is comparable to fit uncertainty, so this run validates the mixed rate but does not resolve that shift. [Equation, fits and uncertainty](docs/kinetic.md#transverse-anisotropy); [arrays](docs/_static/figures/mixed_weibel/run.json).

<img src="docs/_static/figures/mixed_weibel/figure.png" width="800" alt="Transverse anisotropy growth against the independent kinetic root">

## Prescribed drive: a control with external work

The [drive example](examples/dark_drive.py) applies a homogeneous sinusoidal force at the cold plasma resonance. It follows the independently solved forced oscillator to **$7.49\times10^{-5}$** of the force scale; ordinary energy gained and accumulated external work differ by **$2.77\times10^{-4}$** of the transfer. This control has no evolving Proca reservoir. [Settings and data](docs/_static/figures/prescribed_drive/run.json).

<img src="docs/_static/figures/prescribed_drive/figure.png" width="800" alt="Prescribed resonant drive and independent forced-oscillator response">

## Homogeneous-drive null check

For a nonrelativistic electron plasma with a fixed neutralizer, a uniform force can be removed from the nonzero spatial modes by moving to an accelerating frame. The [matched null example](examples/dark_null.py) measures the seeded mode with and without that force: its largest relative difference falls from **$2.78\times10^{-4}$** to **$3.94\times10^{-5}$** when the grid, loading and timestep are refined together. This restricted null is distinct from a mobile-ion heating test. [Record](docs/_static/figures/homogeneous_null/run.json).

<img src="docs/_static/figures/homogeneous_null/figure.png" width="800" alt="Matched zero-drive and homogeneous-drive nonzero-mode histories">

## Oblique magnetized 3V response

The [cold oblique example](examples/dark_plasma.py) sets a magnetic field with three nonzero components, exciting all particle velocity and field polarizations. A 12-state cold-fluid matrix exponential gives an independent answer for the six mean electric fields; the largest full-run error is **$4.92\times10^{-4}$** of the initial dark amplitude. [Record](docs/_static/figures/oblique_3v/run.json).

<img src="docs/_static/figures/oblique_3v/figure.png" width="800" alt="Six mean electric fields against an oblique cold-fluid matrix reference">

## Mobile ions: finite versus imposed reservoirs

The [mobile-ion example](examples/dark_reservoir.py) compares a zero-drive control, an imposed resonant force, and two finite Proca reservoirs with the same initial force. A two-fluid electron–ion solution checks the early mean response. By $\omega_pt=40$, the small dark reservoir loses about **52.8%** of its initial field energy; the large one loses about **14.8%**. A separate 1,000-cell [paper-geometry pilot](docs/_static/figures/paper_geometry_pilot/run.json) reaches only $\omega_pt=80$ and finds no loading-stable pump-induced higher-mode growth. It does not reproduce the late heating of [Hook, Huang and Shalaby](https://doi.org/10.1103/98cx-7t43). [Controls and limits](docs/kinetic.md#mobile-ions-and-a-finite-reservoir).

<img src="docs/_static/figures/mobile_ions/figure.png" width="800" alt="Matched ion-electron prescribed drive and finite dark reservoirs">

<img src="docs/_static/figures/paper_geometry_pilot/figure.png" width="800" alt="Early paper-geometry control and loading-sensitive higher-mode energy">

## Conservation and long-time clocks

The [source-free Proca comparison](docs/scripts/benchmark_time_integrators.py) evolves longitudinal and transverse fields to $\Omega_Dt=200$ against a matrix exponential. The explicit split bounds field-energy error at **2.37%** for $\Delta t\Omega_D=0.2$ and **0.577%** at half that step. Implicit midpoint preserves this vacuum field energy to roundoff but has **1.53** final relative state error at the larger step; DOP853 reaches **$3.0\times10^{-8}$** state error with tight tolerances. Phase accuracy and total PIC energy require separate checks: exact field-only conservation does not close the particle–field work ledger. [Methods and timings](docs/performance.md#which-clock-to-trust) and the [full record](docs/_static/figures/time_integrators/run.json) give the comparison.

<img src="docs/_static/figures/time_integrators/figure.png" width="800" alt="Vacuum Proca energy conservation and long-time state error for explicit, midpoint and DOP853 integrators">

## Differentiate and optimize a physical objective

JAX derivatives pass through particle loading and weights, fields, the PIC run and the measured objective. The [density calibration](examples/optimize_dark_photon.py) maximizes coherent electric plus cold bulk energy on one fixed physical interval, with $p=n/n_{\rm ref}$:

$$
\mathcal C(p)=\frac1{30}\int_{45}^{75}\left[\left(\frac{E_y}{D_0}\right)^2+\frac1p\left(\frac{J_y}{\epsilon_0\omega_0D_0}\right)^2\right]d(\omega_0t).
$$

The full PIC optimum is **$p=1.0016225$**; an independent cold matrix exponential and Fréchet derivative give **$1.0016317$**. Finite differences and time refinement check the PIC gradient. This is a known-answer optimization, not a fitted target. [Methods and limits](docs/optimization.md) and the [full run](docs/_static/figures/density_calibration/run.json) give the error measures. To evaluate the same differentiable objective:

```python
from examples.optimize_dark_photon import build_objective

objective, value_and_gradient, steps = build_objective(16, 64, 45, 75)
value, density_gradient = value_and_gradient(0.97)
```

<img src="docs/_static/figures/density_calibration/figure.png" width="800" alt="PIC density objective, independent cold reference and gradient checks">

## A finite dark packet crosses a designed slab

The [profile example](examples/dark_profile.py) keeps the slab's total electron column fixed while changing four smooth density weights. It measures **outgoing ordinary-photon flux at a detector**, rather than a homogeneous field amplitude. A JAX gradient design raises the held-out photon-energy fraction from **0.0020826** to **0.0021179**; a finer particle/grid replay gives a **1.81–1.82%** relative gain. An independent cold scattering solve predicts the same ordering. Finite-amplitude kinetic replay changes the yield but finds **no reliable additional gain** from re-optimization. This extends the tested setting from cold wave conversion and prescribed drives to a finite, self-consistent reservoir and a kinetic objective; it makes no detector or priority claim. See [constraints, reference and replay](docs/profile.md).

<img src="docs/_static/figures/profile_design/figure.png" width="800" alt="Fixed-column slab profiles and outgoing photon fraction">

<img src="docs/_static/movies/slab_packet/figure.webp" width="900" alt="Dark packet crossing a resolved slab, showing ordinary and dark electromagnetic fields and energy">

*The [packet movie](docs/_static/movies/slab_packet/run.json) uses 512 cells, about 26 cells across the slab, 256 particles per basis per species and 80 stored frames. It shows propagation; the full optimization record carries the flux and convergence claims.*

## Vacuum polarizations and complete restart

The [field tests](tests/test_proca.py) compare longitudinal and both transverse vacuum polarizations with discrete-symbol waves, check the ordinary and dark Gauss laws, and resume the complete particle/field state across a run boundary. The source-free [energy/phase experiment](docs/_static/figures/time_integrators/run.json) above includes all three polarizations. Native NPZ restart saves $\mathbf E_D,\mathbf B_D,\mathbf A_D,\phi_D$, the neutralizing background and accumulated work alongside the parent particle and Maxwell state. [Validation details](docs/validation.md).

## Runtime and differentiation cost

The [isolated field-step benchmark](docs/_static/figures/field_cost.json) uses 8,192 particles on 128 cells for 256 steps: median warm CPU times are **176 ms** for the parent and **219 ms** for active Proca. A [131,072-particle replay](docs/_static/figures/field_cost_large.json) gives **5.30 s** and **5.71 s**, with host load varying too much to infer a reliable overhead ratio. These are *timing workloads*; the two-stream movie advances 262,144 particles, and the refined kinetic bump run advances 240,000. A separate [955-step gradient recurrence](docs/_static/figures/recurrence_benchmark.json) compares native JAX checkpointing with SOLVAX. SOLVAX agrees on value and derivative but offers no clear benefit over native segmented JAX, so it remains optional. [Compile, memory and device details](docs/performance.md) are reported with the measurements.

## Related codes and scope

[JAX-in-Cell](https://github.com/uwplasma/JAX-in-Cell) supplies the ordinary PIC engine and the original bump example. [Hook, Huang and Shalaby](https://doi.org/10.1103/98cx-7t43) use high-order PIC with mobile ions and an imposed homogeneous dark drive; our short finite-reservoir control has not reproduced their long heating curve. [Corelli *et al.*](https://arxiv.org/abs/2410.16357) study cold-fluid conversion at multiple crossings, while [Caputo *et al.*](https://github.com/smsharma/dark-photons-perturbations) provide inhomogeneous-universe conversion notebooks. [Methods and literature](docs/validation.md#other-numerical-approaches) explain the distinct models.

| Documented feature | This code | [JAX-in-Cell](https://github.com/uwplasma/JAX-in-Cell/tree/83d327118163833f93e2588edcb5029241f6ba2a) | [PIConGPU](https://github.com/ComputationalRadiationPhysics/picongpu) | [Caputo *et al.* notebooks](https://github.com/smsharma/dark-photons-perturbations) |
|---|:---:|:---:|:---:|:---:|
| Self-consistent charged-particle PIC | ✅ | ✅ | ✅ | ❌ |
| Finite massive-vector field sourced by PIC current | ✅ | ❌ | ❌ | ❌ |
| Inhomogeneous dark-photon conversion model | ✅ | ❌ | ❌ | ✅ |
| Gradients through the PIC trajectory | ✅ | ✅ | ❌ | ❌ |
| Constrained PIC-gradient profile design | ✅ | ❌ | ❌ | ❌ |
| CPU and GPU execution | ✅ | ✅ | ✅ | ❌ |
| Complete particle/field restart | ✅ | ✅ | ✅ | ❌ |

✅ means the linked implementation documents the feature; ❌ means it does not provide that feature in its reviewed scope. The codes solve different problems: PIConGPU is a large-scale multidimensional Maxwell PIC, while the Caputo notebooks calculate cosmological conversion without kinetic particles. The [literature and method comparison](docs/validation.md#other-numerical-approaches) gives more context.

Regenerate full figures, records and measured documentation with `python docs/scripts/make_all.py`. Regenerate the three compressed README loops with `python docs/scripts/make_movies.py` after `python -m pip install -e '.[media]'`. Quick presets are smoke tests, not the full evidence quoted above.

MIT licensed. JAX-in-Cell and its human contributors retain upstream authorship and licenses. The parent [draft PR #42](https://github.com/uwplasma/JAX-in-Cell/pull/42) remains separate.
