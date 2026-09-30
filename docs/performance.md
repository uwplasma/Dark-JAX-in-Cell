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

## Which clock to trust

The source-free [time-step experiment](scripts/benchmark_time_integrators.py) uses the same staggered 1D Proca difference operators as the package, in normalized units $c=\Omega_D=\epsilon_0=1$. Its initial field contains longitudinal and both transverse components, with $\phi_D=-D E_{D,x}$, $B_D=\operatorname{curl} A_D=0$. The matrix exponential of the **same spatially discrete generator** is the temporal oracle. The 16-cell run lasts $\Omega_Dt=200$; these are vacuum field tests, without particles or current deposition.

![Proca vacuum energy and long-time state error](_static/figures/time_integrators/figure.png)

| Method | $\Delta t\Omega_D$ | Max $|\Delta U_D/U_D(0)|$ | Final state error against $e^{tL}$ | Max dark Gauss residual |
|---|---:|---:|---:|---:|
| Current kick–drift–kick | 0.2 | 2.37% | 1.03 | $3.3\times10^{-15}$ |
| Current kick–drift–kick | 0.1 | 0.577% | 0.356 | $5.4\times10^{-15}$ |
| Implicit midpoint, sparse LU | 0.2 | $5.4\times10^{-14}$ | 1.53 | $2.5\times10^{-15}$ |
| Implicit midpoint, sparse LU | 0.1 | $6.6\times10^{-14}$ | 0.633 | $3.9\times10^{-15}$ |
| Adaptive DOP853, $10^{-9}$ relative tolerance | output every 0.2 | $1.3\times10^{-9}$ | $3.0\times10^{-8}$ | $2.7\times10^{-15}$ |

The split method's energy error is bounded and falls about fourfold when the step halves. Midpoint keeps this **quadratic vacuum-field energy** to roundoff because the discrete generator is skew symmetric, but has *larger phase error* than the split at either tested step. DOP853 accurately follows this smooth source-free system; it needed 20,129 right-hand-side evaluations. The exploratory SciPy wall times include sparse factorization and NumPy-loop overhead, so they are **incomparable with the compiled JAX PIC path**. The [record](_static/figures/time_integrators/run.json) and [arrays](_static/figures/time_integrators/data.npz) include the measured times; `--quick` shortens the horizon to 20.

For the **coupled** two-stream problem, halving $\Delta t\omega_p$ from 0.025 to 0.0125 at the same 64 cells and loading changes the largest sampled total-energy drift from 0.677% to 0.680%. At fixed $\Delta t\omega_p=0.0125$, independently doubling quiet-start particles and grid cells shows where the error lies: particle doubling changes dark total drift by less than $10^{-9}$ in fractional units, while grid doubling reduces it from **{{ saturation_short_coarse_drift_percent }}%** to **{{ saturation_short_refined_drift_percent }}%**. The dark-field source-work residual falls from $1.91\times10^{-5}$ to $4.65\times10^{-6}$ of the maximum dark work transfer on time refinement. The 200/$\omega_p$ replay keeps the same $0.5/\omega_p$ sampling interval; its largest sampled dark total-energy changes are **{{ saturation_long_coarse_drift_percent }}%** and **{{ saturation_long_refined_drift_percent }}%**. A different vacuum field integrator would address the already small dark-field work error, not the dominant coupled spatial error in this case. Exact field-only conservation does **not** imply exact PIC conservation: the current deposit and particle work must use conjugate space-time weights. The [factorial record](_static/figures/two_stream_saturation/run.json) and [extended record](_static/figures/two_stream_extended/run.json) support this narrower conclusion.

[Markidis and Lapenta](https://arxiv.org/abs/1108.1959) use a jointly implicit particle/Maxwell midpoint solve for exact coupled energy. [Kormann and Sonnendrücker](https://arxiv.org/abs/1910.04000) derive discrete-gradient variants and identify the extra condition needed to keep Gauss. [Chen *et al.*](https://arxiv.org/abs/1903.01565) use a time-centred particle solve with leapfrog Maxwell fields to retain light-wave dispersion and conserve charge and energy. [Ricketson and Hu](https://arxiv.org/abs/2411.09605) give an explicit particle correction with very small energy error; their [2026 relativistic extension](https://arxiv.org/abs/2605.18542) covers Yee and PSATD field solvers. Either correction requires changing the particle-field coupling, not just this Proca clock. [SHARP](https://arxiv.org/abs/1702.04732) shows why higher-order particle shapes and separate grid/particle convergence matter for long kinetic runs. The parent currently uses a quadratic spline. Its Boris pusher already sees $\mathbf E+\eta\mathbf E_D$ and $\mathbf B+\eta\mathbf B_D$; a separate dark-force rotation would solve the same Lorentz equation. [Ripperda *et al.*](https://arxiv.org/abs/1710.09164) compare Boris, Vay, Higuera–Cary and implicit particle pushers for regimes where relativistic trajectory error can justify changing the pusher.

The present tests favour keeping the explicit coupled step while improving spatial convergence. The $\omega_pt=200$ first-mode difference reverses sign between 128 and 256 cells at fixed timestep even as the finest closed-energy drift falls to **{{ saturation_long_fine_drift_percent }}%**. Further grid refinement and a warm, seeded ensemble should precede a production change to interpolation order. An implicit or IMEX mass solve becomes worth revisiting for $\Omega_D\Delta t$ near the explicit stability limit, with a charge-conserving current and full PIC work balance. An adaptive DOP853 field clock would require synchronized particle trajectories and current integration at its internal stages; applying it to frozen deposits would test another model. None of these source-free timings establishes a faster or more accurate nonlinear PIC solver.
