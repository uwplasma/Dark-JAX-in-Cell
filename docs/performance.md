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

The calibration example's own [run record](_static/figures/density_calibration/run.json) includes total scan time and compiler-first-call measurements. No disabled-feature overhead claim or GPU memory claim is made here: a separate JAX-in-Cell versus Dark-JAX-in-Cell matched benchmark is still required.
