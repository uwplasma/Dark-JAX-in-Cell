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
