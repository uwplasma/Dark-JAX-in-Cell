# The ghost's memory bill

The explicit Maxwell–Proca step has no linear system, so SOLVAX is not used for fields. We measured its `checkpointed_fori_loop` only as an optional way to differentiate a long fixed recurrence, against native `lax.scan`, rematerialized scan, and an equivalent native segmented recurrence. All four call the same production PIC transition and scalar density objective. The test parameters are 16 cells, 64 particles, float64 CPU, $p=0.97$, checkpoint width 32. The exact case has 256 steps; the tail case has 955 steps and a nondivisible final segment. Each row ran in a fresh process, with five synchronized warm primal and gradient samples. [Raw measurements](_static/figures/recurrence_benchmark.json) include the range, first-call time, host load and process memory for every row.

| 955-step method | Warm gradient median (s) | Compiler temporary estimate (MiB) | Process peak (MiB) | First gradient call incl. compile (s) |
|---|---:|---:|---:|---:|
| Native scan | {{ tail_scan_grad_s }} | {{ tail_scan_temp_mib }} | {{ tail_scan_rss_mib }} | {{ tail_scan_first_grad_s }} |
| Native rematerialized scan | {{ tail_remat_grad_s }} | {{ tail_remat_temp_mib }} | {{ tail_remat_rss_mib }} | {{ tail_remat_first_grad_s }} |
| Native segmented | {{ tail_segmented_grad_s }} | {{ tail_segmented_temp_mib }} | {{ tail_segmented_rss_mib }} | {{ tail_segmented_first_grad_s }} |
| SOLVAX checkpointed | {{ tail_solvax_grad_s }} | {{ tail_solvax_temp_mib }} | {{ tail_solvax_rss_mib }} | {{ tail_solvax_first_grad_s }} |

The 256-step exact-segment case gives native scan **{{ exact_scan_grad_s }}** s warm gradient and **{{ exact_scan_temp_mib }}** MiB compiler temporaries; SOLVAX gives **{{ exact_solvax_grad_s }}** s and **{{ exact_solvax_temp_mib }}** MiB. Objective and gradient agree across methods to the benchmark's strict numerical checks, including the tail.

The compiler estimate is the executable's temporary-buffer estimate, **not** the full process footprint. Process peak includes Python, JAX, compilation, imports and the runtime allocator; the methods were isolated but shared-host load varied. SOLVAX reduced estimated temporaries relative to plain scan but had similar warm speed and memory to the native segmented method. Its full installation also requires `equinox>=0.13.3`; the tested compatible versions were SOLVAX 0.27.0, Equinox 0.13.8 and JAX 0.10.2. We therefore **defer adopting SOLVAX as a package dependency**. The native recurrence remains the default. The optional benchmark can be reproduced with:

```sh
python -m pip install solvax==0.27.0
python docs/scripts/benchmark_recurrence.py --all
python docs/scripts/make_all.py
```

The calibration example's own [run record](_static/figures/density_calibration/run.json) includes total scan time and compiler-first-call measurements.

## Cost of the actual field step

With 128 cells, 8,192 electrons, 256 steps, float64 CPU, one final diagnostic and identical loading, we measured the parent `Simulation` against a configured zero-coupling Proca field and an active field. Each case ran in a fresh process; the [record](_static/figures/field_cost.json) contains five synchronized warm samples, first-call time including compile, process peak memory and host load.

| Case | Warm run median (ms) | First call incl. compile (s) | Process peak (MiB) |
|---|---:|---:|---:|
| JAX-in-Cell, no dark model | {{ field_parent_warm_ms }} | {{ field_parent_first_s }} | {{ field_parent_rss_mib }} |
| Proca configured, $\eta=0$ | {{ field_eta_zero_warm_ms }} | {{ field_eta_zero_first_s }} | {{ field_eta_zero_rss_mib }} |
| Proca active, $\eta=0.05$ | {{ field_active_warm_ms }} | {{ field_active_first_s }} | {{ field_active_rss_mib }} |

The **dark-disabled route is the parent `Simulation` itself**, so its source and execution path are unchanged. A configured $\eta=0$ model still evolves a free massive field, costs time and memory, and must not be called disabled. Its ordinary-field checksum agrees with the parent; the active case differs physically. Process peaks include imports and compilation. These are one-device measurements, not a portable overhead factor or a GPU claim. Reproduce them on an otherwise quiet host with `python docs/scripts/benchmark_field_cost.py --all`, then `python docs/scripts/make_all.py` to refresh the MyST numbers.

The same isolated script also has a [131,072-particle CPU run](_static/figures/field_cost_large.json) on 128 cells for 256 steps. Five warm runs in each fresh process give median times of **{{ large_parent_warm_s }} s** for the parent, **{{ large_eta_zero_warm_s }} s** for configured $\eta=0$, and **{{ large_active_warm_s }} s** for active Proca. First-call times are {{ large_parent_first_s }}, {{ large_eta_zero_first_s }} and {{ large_active_first_s }} s; process peaks are {{ large_parent_rss_mib }}, {{ large_eta_zero_rss_mib }} and {{ large_active_rss_mib }} MiB. System load rose from 11.9 to 32.3 across the cases, so these measurements establish that the larger loading executes within the recorded memory but **do not give a reliable fractional overhead**. The first call includes compilation or executable-cache loading. Reproduce with `python docs/scripts/benchmark_field_cost.py --all --particles 131072 --steps 256 --output docs/_static/figures/field_cost_large.json` on a quiet host. This is a timing workload; the kinetic runs above provide physics-resolution checks.

## Sparse particle histories

The reduced run keeps both field histories, per-species kinetic energy and all-step conservation maxima while omitting particle trajectories from the sampled output. A [fresh-process CPU comparison](_static/figures/storage_cost.json) used 120,000 electrons, 128 cells, 128 steps and 16 stored samples. Both cases consumed the final closed energy and all-step maximum; the full case also consumed every stored position, velocity and weight array.

| History | Warm median (s) | First call incl. compile (s) | Process peak (MiB) |
|---|---:|---:|---:|
| Sparse particle history | {{ storage_sparse_warm_s }} | {{ storage_sparse_first_s }} | {{ storage_sparse_rss_mib }} |
| Full particle history | {{ storage_full_warm_s }} | {{ storage_full_first_s }} | {{ storage_full_rss_mib }} |

The final complete energy and all-step maximum balance agree exactly between these runs. The memory and timing figures include compilation and shared-host load, so they characterize this workload and machine, not a universal speedup. Reproduce with `python docs/scripts/benchmark_field_cost.py --storage-all --particles 120000 --steps 128 --stride 8`, then refresh substitutions with `python docs/scripts/make_all.py --records-only`.

## Which clock to trust

The source-free [time-step experiment](scripts/benchmark_time_integrators.py) uses the same staggered 1D Proca difference operators as the package, in normalized units $c=\Omega_D=\epsilon_0=1$. Its initial field contains longitudinal and both transverse components, with $\phi_D=-D E_{D,x}$, $B_D=\operatorname{curl} A_D=0$. The matrix exponential of the **same spatially discrete generator** is the temporal oracle. The 16-cell run lasts $\Omega_Dt=200$; these are vacuum field tests, without particles or current deposition.

![Proca vacuum energy and long-time state error](_static/figures/time_integrators/figure.png)

| Method | $\Delta t\Omega_D$ | Max $\lvert\Delta U_D/U_D(0)\rvert$ | Final state error against $e^{tL}$ | Max dark Gauss residual |
|---|---:|---:|---:|---:|
| Current kick–drift–kick | 0.2 | 2.37% | 1.03 | $3.3\times10^{-15}$ |
| Current kick–drift–kick | 0.1 | 0.577% | 0.356 | $5.4\times10^{-15}$ |
| Implicit midpoint, sparse LU | 0.2 | $5.4\times10^{-14}$ | 1.53 | $2.5\times10^{-15}$ |
| Implicit midpoint, sparse LU | 0.1 | $6.6\times10^{-14}$ | 0.633 | $3.9\times10^{-15}$ |
| Adaptive DOP853, $10^{-9}$ relative tolerance | output every 0.2 | $1.3\times10^{-9}$ | $3.0\times10^{-8}$ | $2.7\times10^{-15}$ |

The split method's energy error is bounded and falls about fourfold when the step halves. Midpoint keeps this **quadratic vacuum-field energy** to roundoff because the discrete generator is skew symmetric, but has *larger phase error* than the split at either tested step. DOP853 accurately follows this smooth source-free system; it needed 20,129 right-hand-side evaluations. The exploratory SciPy wall times include sparse factorization and NumPy-loop overhead, so they are **incomparable with the compiled JAX PIC path**. The [record](_static/figures/time_integrators/run.json) and [arrays](_static/figures/time_integrators/data.npz) include the measured times; `--quick` shortens the horizon to 20.

For the **coupled** two-stream problem, halving $\Delta t\omega_p$ from 0.025 to 0.0125 at the same 64 cells and loading changes the largest sampled total-energy drift from 0.677% to 0.680%. At fixed $\Delta t\omega_p=0.0125$, independently doubling quiet-start particles and grid cells shows where the error lies: particle doubling changes dark total drift by less than $10^{-9}$ in fractional units, while grid doubling reduces it from **{{ saturation_short_coarse_drift_percent }}%** to **{{ saturation_short_refined_drift_percent }}%**. The dark-field source-work residual falls from $1.91\times10^{-5}$ to $4.65\times10^{-6}$ of the maximum dark work transfer on time refinement. The 200/$\omega_p$ replay keeps the same $0.5/\omega_p$ sampling interval; its largest sampled dark total-energy changes are **{{ saturation_long_coarse_drift_percent }}%** and **{{ saturation_long_refined_drift_percent }}%**. A different vacuum field integrator would address the already small dark-field work error, not the dominant coupled spatial error in this case. Exact field-only conservation does **not** imply exact PIC conservation: the current deposit and particle work must use conjugate space-time weights. The [factorial record](_static/figures/two_stream_saturation/run.json) and [extended record](_static/figures/two_stream_extended/run.json) support this narrower conclusion.

[Markidis and Lapenta](https://arxiv.org/abs/1108.1959) use a jointly implicit particle/Maxwell midpoint solve for exact coupled energy. [Kormann and Sonnendrücker](https://arxiv.org/abs/1910.04000) derive discrete-gradient variants and identify the extra condition needed to keep Gauss. [Chen *et al.*](https://arxiv.org/abs/1903.01565) use a time-centred particle solve with leapfrog Maxwell fields to retain light-wave dispersion and conserve charge and energy. [Ricketson and Hu](https://arxiv.org/abs/2411.09605) give an explicit particle correction with very small energy error; their [2026 relativistic extension](https://arxiv.org/abs/2605.18542) covers Yee and PSATD field solvers. Either correction requires changing the particle-field coupling, not just this Proca clock. [SHARP](https://arxiv.org/abs/1702.04732) shows why higher-order particle shapes and separate grid/particle convergence matter for long kinetic runs. The parent currently uses a quadratic spline. Its Boris pusher already sees $\mathbf E+\eta\mathbf E_D$ and $\mathbf B+\eta\mathbf B_D$; a separate dark-force rotation would solve the same Lorentz equation. [Ripperda *et al.*](https://arxiv.org/abs/1710.09164) compare Boris, Vay, Higuera–Cary and implicit particle pushers for regimes where relativistic trajectory error can justify changing the pusher.

The relativistic [pair-reservoir replay](_static/figures/oscillating_dark_pair/run.json) provides another coupled check through $\omega_0t=170$. At fixed $\Delta t\omega_0=0.00625$ and 32 markers per cell, the largest complete-energy change falls from **{{ dark_pair_grid_drift }}** at 2,048 cells to **{{ dark_pair_energy_drift }}** at 4,096 and **{{ dark_pair_large_grid_drift }}** at 8,192. Doubling the timestep at 4,096 cells changes that drift to **{{ dark_pair_clock_drift }}**. The late coherent fraction still moves from **{{ dark_pair_grid_control }}** to **{{ dark_pair_coherent }}** to **{{ dark_pair_large_grid }}** on spatial refinement. Energy conservation is improving, but this nonlinear observable is not converged.

The present tests favour keeping the explicit coupled step while improving spatial convergence. The $\omega_pt=200$ first-mode difference reverses sign between 128 and 256 cells at fixed timestep even as the finest closed-energy drift falls to **{{ saturation_long_fine_drift_percent }}%**. Further grid refinement and a warm, seeded ensemble should precede a production change to interpolation order. An implicit or IMEX mass solve becomes worth revisiting for $\Omega_D\Delta t$ near the explicit stability limit, with a charge-conserving current and full PIC work balance. An adaptive DOP853 field clock would require synchronized particle trajectories and current integration at its internal stages; applying it to frozen deposits would test another model. None of these source-free timings establishes a faster or more accurate nonlinear PIC solver.

## Coupled kinetic methods

The pinned parent provides an ordinary Crank–Nicolson PIC method with fixed Picard iterations and particle substeps. Its longitudinal force is the discrete gradient conjugate to the trajectory continuity current; transverse current is the transpose of the field gather. Once the particle orbit and fields converge together, particle work cancels mesh work. This is the mechanism of [Chen, Chacón and Barnes](https://arxiv.org/abs/1101.3701) and the discrete-gradient construction of [Kormann and Sonnendrücker](https://arxiv.org/abs/1910.04000). Energy and Gauss can be conserved together while momentum has a finite error. The coupled Proca method uses the explicit compatible split described in [the equations](physics.md).

### Conservation with the same particles

The [coupled benchmark](scripts/benchmark_pic_conservation.py) loads two warm electron beams and mobile ions identically in each row: 8,192 markers per population, $u_e=\pm0.05c$, $\sigma_e=0.003c$, $m_i/m_e=1836$, $T_i=T_e$, $ku_e/\omega_p=0.5$ and a $10^{-4}$ density seed on one beam. Here $\omega_p$ is the total **electron** plasma frequency. The inherited three-velocity solver evolves a longitudinal, Newtonian experiment through $\omega_pt=40$. The massive-field counterpart has $\Omega_D/\omega_p=0.7$, $\eta=0.3$ and the specified bare initialization $E_D=\eta E$, $\phi_D=0$. These Proca rows change the physical interaction; the ordinary rows compare algorithms for the same equations.

![Energy, momentum, first electric mode and transient phase for identical particle loading](_static/figures/pic_conservation/figure.png)

The upper panels show sampled defects in complete energy and continuum momentum; the table uses maxima from **every step**. The lower panels show the first complex electric mode and its phase relative to $\omega_pt=8$. Dashed amplitude segments use the kinetic pole growth rate over the fit window. A flat phase would describe a pure symmetric growing eigenmode; the curved traces expose the loading transient.

| Method | Cells | $\Delta t\omega_p$ | Max $\lvert\Delta U\rvert/U_0$ | Max $\lvert\Delta P\rvert/(nm_ecL)$ | Fitted $\gamma/\omega_p$ | Fitted $\omega_r/\omega_p$ |
|---|---:|---:|---:|---:|---:|---:|
| Ordinary explicit | 128 | 0.002 | {{ pic_explicit_energy }} | {{ pic_explicit_momentum }} | {{ pic_explicit_growth }} | {{ pic_explicit_frequency }} |
| Ordinary implicit, 4 iterations | 128 | 0.002 | {{ pic_implicit4_energy }} | {{ pic_implicit4_momentum }} | {{ pic_implicit4_growth }} | {{ pic_implicit4_frequency }} |
| Ordinary implicit, 8 iterations | 128 | 0.002 | {{ pic_implicit8_energy }} | {{ pic_implicit8_momentum }} | {{ pic_implicit8_growth }} | {{ pic_implicit8_frequency }} |
| Ordinary implicit, 8 iterations | 128 | 0.02 | {{ pic_implicit_large_energy }} | {{ pic_implicit_large_momentum }} | {{ pic_implicit_large_growth }} | {{ pic_implicit_large_frequency }} |
| Proca explicit | 128 | 0.002 | {{ pic_proca_energy }} | {{ pic_proca_momentum }} | {{ pic_proca_growth }} | {{ pic_proca_frequency }} |
| Proca explicit | 128 | 0.001 | {{ pic_proca_halfstep_energy }} | {{ pic_proca_halfstep_momentum }} | {{ pic_proca_halfstep_growth }} | {{ pic_proca_halfstep_frequency }} |
| Proca explicit | 256 | 0.002 | {{ pic_proca_fine_energy }} | {{ pic_proca_fine_momentum }} | {{ pic_proca_fine_growth }} | {{ pic_proca_fine_frequency }} |

The ordinary implicit solver uses two particle substeps. At $\Delta t\omega_p=0.002$, one Picard iteration gives $8.79\times10^{-4}$ energy error, two give $9.50\times10^{-10}$, and four, eight and twelve reach roundoff. Increasing the count beyond four changes the fitted growth by less than $2\times10^{-11}$; it leaves the finite momentum defect. This reproduces the energy/charge mechanism of the cited implicit methods without demonstrating exact discrete momentum conservation. The explicit momentum agreement is specific to this neutral longitudinal loading; the separate travelling-wave check exercises the longitudinal Proca potential term.

Across all twelve rows, particle charge is unchanged, grid charge changes by at most $3.67\times10^{-16}enL$, and grid/particle charge differs by at most $3.65\times10^{-16}enL$. The continuity residual is below $4.57\times10^{-13}en\omega_p$; ordinary and dark Gauss residuals are below $2.65\times10^{-13}en/\epsilon_0$ and $6.59\times10^{-14}en/\epsilon_0$. These normalizations use the fixed total electron density $n$. For Proca, the dark-sector defect $|\Delta U_D-W_D|/U_0$ falls from $2.08\times10^{-11}$ to $5.20\times10^{-12}$ when the step halves. The larger complete-energy defect falls about fourfold on grid doubling and is almost unchanged by time refinement. The spatial particle/mesh work error dominates this experiment.

### Growth, phase and cost

The multispecies kinetic determinant gives $\gamma/\omega_p=0.3383986$ for ordinary Maxwell and $0.3478317$ for the coupled field, with $\omega_r\simeq0$. Its root routine uses the combined electron-plus-ion plasma frequency; the record converts to the electron convention by $\sqrt{1+1/1836}$. Fits use all 81 samples in the inclusive window $8\le\omega_pt\le16$. Their regression standard errors are approximately 0.0030; the ordinary and Proca slopes differ from their poles by about 2.4% and 1.3%. The phase excursions reach 0.42 and 0.35 radians despite small average frequencies. These are transient initial-value traces, with unresolved preparation and fit-window error. The small mixed-minus-ordinary growth shift is consequently **not confirmed** by this benchmark.

Time refinement from $\Delta t\omega_p=0.002$ to 0.001 changes the ordinary explicit growth by $1.6\times10^{-8}$ and the Proca growth by $1.2\times10^{-8}$. Earlier strict floating-point endpoint selection dropped two samples in some rows and produced a spurious time-fit difference; [the record](_static/figures/pic_conservation/run.json) preserves those original fits and the corrected analysis hash. Increasing the implicit step tenfold changes growth by $6.6\times10^{-6}$ in this short growing-mode test. Resolving driven oscillation phase and late nonlinear observables needs separate long-time checks.

| Method | $\Delta t\omega_p$ | Compile (s) | Synchronized warm run (s) | Compiler temporaries (MiB) | Process peak (MiB) |
|---|---:|---:|---:|---:|---:|
| Ordinary explicit | 0.002 | {{ pic_explicit_compile }} | {{ pic_explicit_warm }} | {{ pic_explicit_temp_mib }} | {{ pic_explicit_rss_mib }} |
| Ordinary implicit, 4 iterations | 0.002 | {{ pic_implicit4_compile }} | {{ pic_implicit4_warm }} | {{ pic_implicit4_temp_mib }} | {{ pic_implicit4_rss_mib }} |
| Ordinary implicit, 8 iterations | 0.002 | {{ pic_implicit8_compile }} | {{ pic_implicit8_warm }} | {{ pic_implicit8_temp_mib }} | {{ pic_implicit8_rss_mib }} |
| Ordinary implicit, 8 iterations | 0.02 | {{ pic_implicit_large_compile }} | {{ pic_implicit_large_warm }} | {{ pic_implicit_large_temp_mib }} | {{ pic_implicit_large_rss_mib }} |

Each row ran in a fresh process with JAX 0.6.2, float64 and one synchronized warm sample on an RTX A4000 GPU. Compilation is separated from execution, and the entire reduced diagnostic run is consumed. Other device activity and one timing sample limit performance conclusions: the large-step implicit and small-step explicit times are similar here, without an established speed advantage. Compiler temporaries exclude the runtime, imports and allocator; process peak includes them. Complete timings, first executions, memory, constraints and all twelve rows are in the [computation record](_static/figures/pic_conservation/run.json) and [compressed scalar histories](_static/figures/pic_conservation/data.npz).

Reproduce the campaign serially, or refit and render its saved arrays without running dynamics. The default full benchmark takes three warm samples; this evidence used one. A separate overhead mode compares the reduced validation runner against production sparse particle output and checks the final energy and all-step energy maximum agree:

```sh
python docs/scripts/benchmark_pic_conservation.py --full --samples 1
python docs/scripts/benchmark_pic_conservation.py --render --output docs/_static/figures/pic_conservation
python docs/scripts/benchmark_pic_conservation.py --overhead --particles 40000
```

The [long resonant-drive replay](kinetic.md#time-refinement-through-the-nonlinear-transition) recovers early second-order behavior but fails the nonlinear field-convergence check. A same-step prefix replay also develops finite trajectory differences. Those gates remain necessary before transferring this short benchmark's timestep or cost conclusions to a late driven result.

### Choosing a coupled method

The [uniform-drive prototype](scripts/benchmark_implicit_drive.py) reuses the parent's relativistic implicit orbit solve. During one step it shifts the electric field by the uniform midpoint force $\mathbf F=\eta\mathbf D_0\cos[\Omega_D(t+h/2)+\varphi]$, then restores the physical field. The accepted trajectory current supplies the external work,

$$
\Delta W=h\Delta x\sum_i\overline{\mathbf J}_i\cdot\mathbf F.
$$

Periodic mean Ampère evolution makes the physical energy-minus-work defect equal to the shifted parent's energy defect. Nonuniform-charge tests independently check this identity and continuity at one and eight Picard iterations. Cold analytic response, relativistic homogeneous orbits, finite differences and an analytic tangent ODE test accuracy and derivatives; complete native restart checks retain absolute drive phase and work.

#### Energy balance and resolved phase

The first long control has 256 cells, $L=2\pi c/\omega_p$, 4,096 particles, eight weighted velocity nodes per species, eight Picard iterations and two particle substeps. Its drive and temperatures match the strong paper case, but its velocity distribution comprises eight cold beams. Its homogeneous accuracy check applies while nonzero modes remain negligible; late Maxwellian behavior requires a resolved velocity distribution.

| $h\omega_p$ | Mean-field relative $L^2$ error through $\tau=40$ (%) | Max $|\Delta U-W|/(nm_ec^2L)$ through 1000 | Max $|\Delta P|/(nm_ecL)$ | Compile / warm median (s) |
|---|---:|---:|---:|---:|
| 0.04 | {{ implicit_drive_dt04_early_wave_percent }} | {{ implicit_drive_dt04_balance }} | {{ implicit_drive_dt04_momentum }} | {{ implicit_drive_dt04_compile }} / {{ implicit_drive_dt04_warm }} |
| 0.02 | {{ implicit_drive_dt02_early_wave_percent }} | {{ implicit_drive_dt02_balance }} | {{ implicit_drive_dt02_momentum }} | {{ implicit_drive_dt02_compile }} / {{ implicit_drive_dt02_warm }} |

![Implicit prescribed-drive energy balance, mean-field reference and phase](_static/figures/implicit_drive_dt02/figure.png)

Halving the step reduces early field and phase error approximately fourfold even though both energy defects are already at roundoff. Nonzero spatial fields subsequently grow at both steps. The late PIC-minus-homogeneous curve and state-plane phase then include spatial dynamics and velocity-loading effects, and cannot be interpreted as timestep error alone. Finite-beam equilibria have a different spectrum from a smooth Maxwellian; increasing velocity resolution is essential ([Dawson's multibeam limit](https://doi.org/10.1103/PhysRev.118.381)). The finite momentum defects remain an accuracy gate: a neutral periodic plasma under this uniform electric force has zero net applied impulse.

At $\tau=8$, gradients of final mean-field energy with respect to drive amplitude, density, frequency and phase agree with centered finite differences to **{{ implicit_drive_dt02_fd_relative }}** in maximum relative error. Density includes particle weights; thermal initialization has a separate derivative test. Against the independent Fréchet/tangent ODE, the error contracts from **{{ implicit_drive_dt04_frechet_relative }}** to **{{ implicit_drive_dt02_frechet_relative }}** on step halving. Warm gradient medians are **{{ implicit_drive_dt04_gradient_warm }} / {{ implicit_drive_dt02_gradient_warm }} s**, with **{{ implicit_drive_dt04_gradient_temp_mib }} / {{ implicit_drive_dt02_gradient_temp_mib }} MiB** compiler temporaries; compilation takes {{ implicit_drive_dt04_gradient_compile }} / {{ implicit_drive_dt02_gradient_compile }} s. These gradients validate the short physical objective, not the late kinetic state.

Both rows used float64 JAX 0.6.2 on one RTX A4000, with compilation separated and two synchronized warm samples. The primal temporary estimates are {{ implicit_drive_dt04_temp_mib }} / {{ implicit_drive_dt02_temp_mib }} MiB; process peaks are {{ implicit_drive_dt04_rss_mib }} / {{ implicit_drive_dt02_rss_mib }} MiB. [Coarser](_static/figures/implicit_drive_dt04/run.json) and [finer](_static/figures/implicit_drive_dt02/run.json) native records retain the scalar arrays, gradient checks and Gauss maxima. This prescribed-force prototype does not evolve a Proca reservoir.

```sh
python docs/scripts/benchmark_implicit_drive.py --cells 256 --nodes 8 --dt .04 --iterations 8 --horizon 1000 --samples 2 --gradient-horizon 8 --output artifacts/implicit_dt04
python docs/scripts/benchmark_implicit_drive.py --cells 256 --nodes 8 --dt .02 --iterations 8 --horizon 1000 --samples 2 --gradient-horizon 8 --output artifacts/implicit_dt02
```

#### Coupling an implicit dark field

A conservative dynamical Proca extension requires one accepted orbit current and conjugate particle work:

$$
\Delta K=h\Delta x\sum_i\overline{\mathbf J}_i\cdot
(\overline{\mathbf E}_i+\eta\overline{\mathbf E}_{D,i}).
$$

With midpoint fields, compatible curls and the divergence/gradient adjoint cancel the field terms against this work. The particles and both fields must converge together while preserving both [Gauss laws](physics.md#constraint-propagation-and-divergence-control). Vacuum midpoint energy conservation and phase tests establish only the field part of this construction; the coupled method must resolve physical oscillation phase and growth.

[ECSIM](https://arxiv.org/abs/1602.06326) uses a linear mass-matrix field solve for energy conservation, without guaranteeing local charge. [ChECSIM](https://doi.org/10.1016/j.jcp.2021.110912) adds compatible deposition/coupling to conserve both, with a different finite-element discretization. [iVPIC, Sections 3.1–3.3](https://arxiv.org/html/1903.01565v2) combines implicit particles with leapfrog fields to retain light-wave dispersion under the CFL restriction; its conserved magnetic energy uses staggered-time products. These ordinary Maxwell methods need a separate Proca work derivation.

[Christlieb, Chacón and Gong, Sections 4–5.3](https://arxiv.org/html/2606.15035v1) isolate orbit-chain-rule and mesh/particle work defects in nonrelativistic potential-based PIC: their unsplit-orbit control retains Gauss while accumulating energy error. Their [relativistic Part II, Sections 4–5](https://arxiv.org/html/2609.36383v1) matches Higuera–Cary mechanics, orbit current and kinetic secant velocity in a coupled potential solve. Energy accuracy depends on nonlinear tolerance and orbit quadrature; its late Weibel traces differ under timestep refinement. Its self-adjoint Nyquist filter acts throughout source/gather/field coupling, and Gauss uses continuity-evolved charge. Independently redeposited endpoint charge and physical momentum need separate checks.

[Ricketson and Hu's relativistic explicit correction, Section 3.1](https://arxiv.org/html/2605.18542v1) enforces local particle work when its analytic correction is real. Adoption requires tracking correction failures, charge-conserving trajectories, momentum, phase and derivatives. [Campos Pinto, Kormann and Sonnendrücker, Figure 3](https://link.springer.com/article/10.1007/s10915-022-01781-3#Fig3) demonstrate finite momentum defects alongside conserved energy and Gauss. A discrete-gradient work identity alone does not establish translation symmetry of the mesh coupling.

[SHARP](https://arxiv.org/abs/1702.04732v2) motivates matched higher-order particle shapes and joint grid/loading refinement. [Adams, Werner and Cary](https://arxiv.org/html/2503.13697v2) show that higher-order field differences alone do not remove grid instability in their explicit electrostatic schemes. [Schmitz, Sections 3–4](https://arxiv.org/html/2603.06509v1) compares relativistic pushers and higher-order compositions in prescribed fields; improving the pusher alone does not establish the order or conservation of a coupled PIC update.

### Repeated execution and compiled horizons

Archive the complete initial state and fix dtype, package versions, chunk length, output shapes and sampling when comparing repeated execution. Reuse one compiled callable to separate execution variability from changes of executable. JAX specializes compilation to argument shapes and static values ([compilation guide](https://docs.jax.dev/en/latest/201/jit.html)); changing a static scan horizon can produce another executable. A fixed-shape chunk can be reused across longer runs, with output transferred between chunks.

The parent deposit uses repeated-index `.at.add` updates, whose accumulation order can be implementation-dependent ([JAX indexing semantics](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.ndarray.at.html)). That API permits variability but does not identify the cause of the observed prefix differences. [JAX's compatibility policy](https://docs.jax.dev/en/latest/api_compatibility.html#numerics-and-randomness) also leaves exact numerics and PRNG samples unguaranteed across versions. Record the archived inputs and compiled configuration before attributing a nonlinear trajectory difference to a particular operation.
