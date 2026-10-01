"""Regenerate full-preset midnight evidence and measured MyST substitutions."""

import json
import math
import subprocess
import sys
from pathlib import Path
from shutil import copyfile

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
# True refreshes documentation from archived full records without launching physics.
records_only = globals().get("records_only", True)
EVIDENCE = ROOT / "docs" / "_static" / "figures" / "cold_exchange"
print("Refreshing archived measurements" if records_only else "Running full studies sequentially", flush=True)


def run_example(script, **inputs):
    """Run one parameter-first script per fresh process; keep physics sequential."""
    if not records_only:
        code = ("import runpy; from pathlib import PosixPath; "
                f"runpy.run_path({str(ROOT / script)!r}, run_name='__main__', init_globals={inputs!r})")
        print(f"Running {script}", flush=True)
        subprocess.run([sys.executable, "-u", "-c", code], cwd=ROOT, check=True)


def provenance(record, folder):
    """Keep substitutions slim; the complete run record lives beside its figure."""
    keys = ("jaxincell", "jax", "numpy", "python", "platform", "jax_enable_x64", "backend", "git")
    return {**{key: record[key] for key in keys}, "run": f"{folder}/run.json"}


def replay_measurements(record, labels):
    """Use raw controlled norms and fixed physical smoothing scales in the docs."""
    for label, key, fmt in labels:
        comparison = record["results"][key]
        for window in comparison["windows"]:
            start, end = window["window_omega_p"]
            for observable in ("mean_E", "electric", "nonzero_electric"):
                error = window["observables"][observable]["relative_l2_difference"]
                measured[f"{label}_{start}_{end}_{observable}_l2_percent"] = format(100 * error, fmt)
        late = comparison["windows"][-1]["observables"]
        for observable, suffix in (("nonzero_electric", "nonzero"), ("local_spread", "local_spread")):
            values = late[observable]
            change = 100 * (np.asarray(values["comparison_mean"]) / np.asarray(values["reference_mean"]) - 1)
            if observable == "nonzero_electric":
                measured[f"{label}_{suffix}_mean_change_percent"] = f"{float(change):.2f}"
            else:
                for species in range(2):
                    measured[f"{label}_{suffix}_{species}_mean_change_percent"] = f"{change[species, 0]:.3f}"


run_example('examples/dark_photon.py', full=True, output=EVIDENCE)
record = json.loads((EVIDENCE / "run.json").read_text())
cold_record = record
results = record["results"]
measured = {"cold_mean_error": f"{results['max_mean_field_error_over_D0']:.3e}",
            "cold_gauss_error": f"{results['max_dark_gauss_over_enref_eps0']:.3e}",
            "cold_energy_error": f"{results['closed_energy_relative_drift']:.3e}",
            "cold_work_error": f"{results['ordinary_energy_vs_dark_work_relative_error']:.3e}",
            "_provenance": record}
drive = EVIDENCE.parent / "prescribed_drive"
run_example('examples/dark_drive.py', full=True, output=drive)
drive_record = json.loads((drive / "run.json").read_text())
drive_results = drive_record["results"]
measured.update({"drive_wave_error": f"{drive_results['max_waveform_error_over_D0']:.3e}",
                 "drive_work_error": f"{drive_results['ordinary_energy_vs_external_work_relative_error']:.3e}"})
calibration = EVIDENCE.parent / "density_calibration"
run_example('examples/optimize_dark_photon.py', full=True, output=calibration)
calibration_record = json.loads((calibration / "run.json").read_text())
cal = calibration_record["results"]
measured.update({"calibration_pic_p": f"{cal['pic_best_p']:.7f}",
                 "calibration_cold_p": f"{cal['cold_best_p']:.7f}",
                 "calibration_pic_objective": f"{cal['pic_best_objective']:.6f}",
                 "calibration_cold_objective": f"{cal['cold_best_objective']:.6f}",
                 "calibration_fd_error": f"{cal['gradient_fd_min_error']:.3e}",
                 "calibration_refined_p": f"{cal['refined_pic']['p']:.7f}",
                 "calibration_refined_objective": f"{cal['refined_pic']['objective']:.6f}",
                 "calibration_gradient_error": f"{abs(cal['gradient_pic_at_0p97'] - cal['gradient_cold_at_0p97']):.3e}",
                 "calibration_refined_gradient_error": (
                     f"{abs(cal['refined_pic']['gradient_at_0p97'] - cal['gradient_cold_at_0p97']):.3e}")})
oblique = EVIDENCE.parent / "oblique_3v"
run_example('examples/dark_plasma.py', full=True, output=oblique)
oblique_record = json.loads((oblique / "run.json").read_text())
measured["oblique_six_field_error"] = f"{oblique_record['results']['max_all_six_field_error_over_D0']:.3e}"
kinetic = EVIDENCE.parent / "mixed_kinetic"
run_example('examples/dark_kinetic.py', full=True, output=kinetic)
kinetic_record = json.loads((kinetic / "run.json").read_text())
kin = kinetic_record["results"]
measured.update({"kinetic_pic_real": f"{kin['measured_real_over_wp']:.5f}",
                 "kinetic_pic_imag": f"{kin['measured_imag_over_wp']:.5f}",
                 "kinetic_root_real": f"{kin['root_real_over_wp']:.5f}",
                 "kinetic_root_imag": f"{kin['root_imag_over_wp']:.5f}",
                 "kinetic_frequency_error": f"{100 * kin['frequency_relative_error']:.2f}",
                 "kinetic_damping_error": f"{100 * kin['damping_relative_error']:.2f}",
                 "kinetic_damping_stderr": f"{kin['damping_slope_stderr']:.4f}"})
physical_records = {}
for cells, particles, steps in ((32, 40000, 3000), (64, 80000, 6000),
                                (128, 160000, 12000)):
    folder = f"physical_kinetic_{cells}"
    path = EVIDENCE.parent / folder
    run_example('examples/dark_kinetic.py', physical=True, full=True, cells=int(cells), particles=int(particles),
                steps=int(steps), output=path)
    record = json.loads((path / "run.json").read_text())
    physical_records[folder] = provenance(record, folder)
    result = record["results"]
    for name in ("measured_real_over_wp", "measured_imag_over_wp",
                 "parent_measured_real_over_wp", "parent_measured_imag_over_wp",
                 "maximum_sampled_closed_energy_drift", "maximum_initial_speed_over_c"):
        measured[f"physical_{cells}_{name}"] = f"{result[name]:.6g}"
measured["physical_reference_parent"] = (
    f"{result['parent_root_real_over_wp']:.5f}{result['parent_root_imag_over_wp']:+.5f}i")
measured["physical_reference_mixed"] = (
    f"{result['root_real_over_wp']:.5f}{result['root_imag_over_wp']:+.5f}i")
measured["physical_reference_screened"] = (
    f"{result['screened_root_over_wp'][0]:.5f}"
    f"{result['screened_root_over_wp'][1]:+.5f}i")
bump = EVIDENCE.parent / "bump_on_tail"
run_example('examples/dark_bump.py', full=True, output=bump)
bump_record = json.loads((bump / "run.json").read_text())
selected = bump_record["results"]["reference_scan"]["selected_roots_over_wp"]
for model in ("ordinary", "full", "quasistatic", "effective_charge"):
    measured[f"bump_threshold_{model}_imag"] = f"{selected[model][0][1][1][1]:+.5f}"
for i, name in enumerate(("base", "refined")):
    result = bump_record["results"]["cases"][i]
    for branch in ("parent", "mixed"):
        measured[f"bump_{name}_{branch}_fit"] = f"{result['fits'][branch]['growth_over_wp']:.5f}"
        measured[f"bump_{name}_{branch}_root"] = f"{result[f'{branch}_root_over_wp'][1]:.5f}"
    measured[f"bump_{name}_energy_percent"] = f"{100 * result['mixed_max_energy_drift']:.4f}"
instability_records = {}
for mode, folder in (("two-stream", "mixed_two_stream"), ("weibel", "mixed_weibel")):
    path = EVIDENCE.parent / folder
    run_example('examples/dark_instabilities.py', mode=mode, full=True, output=path)
    instability_records[folder] = json.loads((path / "run.json").read_text())
warm_path = EVIDENCE.parent / "warm_two_stream"
run_example('examples/dark_instabilities.py', mode='warm-two-stream', full=True, output=warm_path)
warm_record = json.loads((warm_path / "run.json").read_text())
warm = warm_record["results"]
for model, root in warm["references"]["unstable"].items():
    measured[f"warm_{model}_root"] = f"{root[1]:.5f}"
for cells in (64, 128):
    case = warm["cases"][f"unstable_{cells}"]
    for branch in ("parent", "mixed"):
        measured[f"warm_{cells}_{branch}_fit"] = (
            f"{case['fits'][branch]['growth_over_wp']:.5f}")
    measured[f"warm_{cells}_energy_drift"] = (
        f"{case['maximum_sampled_closed_energy_drift']:.3e}")
measured["warm_stable_late_over_early"] = (
    f"{warm['cases']['stable_64']['late_over_early_mode_rms']['mixed']:.3f}")
two = instability_records["mixed_two_stream"]["results"]
weibel = instability_records["mixed_weibel"]["results"]
two_record = instability_records["mixed_two_stream"]
for name, result in (("two_stream", two), ("weibel", weibel)):
    measured[f"{name}_root"] = f"{result['reference_growth_over_wp']:.5f}"
    measured[f"{name}_pic"] = f"{result['pic_fits'][0]['growth_over_wp']:.5f}"
    measured[f"{name}_refined"] = f"{result['pic_fits'][1]['growth_over_wp']:.5f}"
    measured[f"{name}_zero_pic"] = f"{result['zero_coupling_pic_growth_over_wp']:.5f}"
measured["weibel_cutoff_kc_over_wp"] = (
    f"{weibel['weibel_marginal']['cutoff_kc_over_wp']['cutoff_mu_0.7'][30]:.5f}")
measured["weibel_stable_late_over_early"] = (
    f"{weibel['stable_mode_2']['late_over_early_magnetic_rms']:.3f}")
saturation = {}
for extended, folder in ((False, "two_stream_saturation"), (True, "two_stream_extended")):
    path = EVIDENCE.parent / folder
    run_example('examples/dark_saturation.py', full=not extended, extended=extended, output=path)
    saturation[folder] = json.loads((path / "run.json").read_text())
short = saturation["two_stream_saturation"]["results"]["cases"]
long = saturation["two_stream_extended"]["results"]["cases"]
for label, cases in (("short", short), ("long", long)):
    coarse = cases[1] if label == "short" else cases[0]
    refined = cases[-1] if label == "short" else cases[1]
    measured[f"saturation_{label}_coarse_drift_percent"] = f"{100 * coarse['mixed_max_energy_drift']:.3f}"
    measured[f"saturation_{label}_refined_drift_percent"] = f"{100 * refined['mixed_max_energy_drift']:.3f}"
for i, name in enumerate(("coarse", "refined")):
    case = long[i]
    rms = case["late_mode_rms_V_m"]
    fields = case["late_fluctuating_energy_over_initial"]
    measured[f"saturation_long_{name}_mode_drop_percent"] = f"{100 * (1 - rms['mixed'] / rms['parent']):.1f}"
    measured[f"saturation_long_{name}_field_drop_percent"] = f"{100 * (1 - fields['mixed'] / fields['parent']):.1f}"
for i, name in ((2, "halfstep"), (3, "fine")):
    case = long[i]
    rms = case["late_mode_rms_V_m"]
    fields = case["late_fluctuating_energy_over_initial"]
    measured[f"saturation_long_{name}_mode_change_percent"] = f"{100 * (rms['mixed'] / rms['parent'] - 1):+.1f}"
    measured[f"saturation_long_{name}_field_change_percent"] = (
        f"{100 * (fields['mixed'] / fields['parent'] - 1):+.1f}")
measured["saturation_long_fine_drift_percent"] = f"{100 * long[3]['mixed_max_energy_drift']:.3f}"
null = EVIDENCE.parent / "homogeneous_null"
run_example('examples/dark_null.py', full=True, output=null)
null_record = json.loads((null / "run.json").read_text())
measured["null_coarse_error"] = f"{null_record['results']['max_mode_amplitude_difference_over_initial'][0]:.3e}"
measured["null_fine_error"] = f"{null_record['results']['max_mode_amplitude_difference_over_initial'][1]:.3e}"
mobile = EVIDENCE.parent / "mobile_ions"
run_example('examples/dark_reservoir.py', full=True, output=mobile)
mobile_record = json.loads((mobile / "run.json").read_text())
m = mobile_record["results"]
measured.update({"mobile_external_oracle_error": f"{m['cases']['external']['mean_error']:.3e}",
                 "mobile_large_oracle_error": f"{m['cases']['large_reservoir']['mean_error']:.3e}",
                 "mobile_small_energy_ratio": (
                     f"{m['finite_vs_external']['small_reservoir']['initial_dark_over_initial_particle_energy']:.3f}"),
                 "mobile_large_energy_ratio": (
                     f"{m['finite_vs_external']['large_reservoir']['initial_dark_over_initial_particle_energy']:.3f}"),
                 "mobile_large_early_pump_error": (
                     f"{m['finite_vs_external']['large_reservoir']['pump_error_early_over_initial_force']:.3e}"),
                 "mobile_small_depletion": (
                     f"{100 * m['finite_vs_external']['small_reservoir']['dark_depletion_fraction']:.1f}"),
                 "mobile_large_depletion": (
                     f"{100 * m['finite_vs_external']['large_reservoir']['dark_depletion_fraction']:.1f}"),
                 "mobile_max_balance": f"{max(abs(v['balance_over_scale']) for v in m['cases'].values()):.3e}"})
pair = EVIDENCE.parent / "oscillating_pair"
run_example('examples/dark_reservoir.py', study='pair', full=True, output=pair)
pair_record = json.loads((pair / "run.json").read_text())
pair_result = pair_record["results"]
pair_refined = pair_result["refinements"]
seed_shift = (pair_result["half_coherence_time_omega0"]
              - pair_result["controls"]["fivefold_seed"]["half_coherence_time_omega0"])
measured.update({"pair_pic_growth": f"{pair_result['pic_cycle_fit']['growth_over_omega0']:.6f}",
                 "pair_vlasov_growth": f"{pair_result['vlasov_cycle_fit']['growth_over_omega0']:.6f}",
                 "pair_floquet_growth": f"{pair_result['floquet_growth_over_omega0']:.6f}",
                 "pair_coarse_error": f"{pair_refined[0]['linear_mode_relative_l2_error']:.3f}",
                 "pair_fine_error": f"{pair_result['linear_mode_relative_l2_error']:.3f}",
                 "pair_late_coherent": f"{pair_result['late_coherent_fraction']:.3f}",
                 "pair_late_random": f"{pair_result['late_random_gain_over_initial_pump']:.3f}",
                 "pair_no_pump_random": (
                     f"{pair_result['controls']['no_pump']['late_random_gain_over_initial_pump']:.2e}"),
                 "pair_seed_shift": f"{seed_shift:.1f}",
                 "pair_seed_expected_shift": (
                     f"{math.log(5) / pair_result['floquet_growth_over_omega0']:.1f}"),
                 "pair_energy_drift": f"{pair_result['max_total_energy_drift']:.3e}"})
dark_pair = EVIDENCE.parent / "oscillating_dark_pair"
run_example('examples/dark_reservoir.py', study='pair_dark', full=True, output=dark_pair)
dark_pair_record = json.loads((dark_pair / "run.json").read_text())
dark_pair_result = dark_pair_record["results"]
dark_pair_gauss = max(dark_pair_result["max_ordinary_gauss_over_scale"],
                      dark_pair_result["max_dark_gauss_over_scale"])
dark_pair_refinements = {
    (item["cells"], item["particles_per_cell_per_species"], item["dt_omega0"]): item
    for item in dark_pair_result["refinements"] if item["horizon_omega0"] > 100}
measured.update({"dark_pair_ordinary_error": (
                     f"{dark_pair_result['ordinary_mode_early_relative_l2_error']:.3f}"),
                 "dark_pair_dark_error": (
                     f"{dark_pair_result['dark_mode_early_relative_l2_error']:.3f}"),
                 "dark_pair_mean_error": (
                     f"{dark_pair_result['homogeneous_early_error_over_initial_dark']:.3e}"),
                 "dark_pair_gauss": f"{dark_pair_gauss:.3e}",
                 "dark_pair_coherent": f"{dark_pair_result['late_coherent_fraction']:.3f}",
                 "dark_pair_kinetic": f"{dark_pair_result['late_random_gain_over_initial_reservoir']:.3f}",
                 "dark_pair_force_control": (
                     f"{dark_pair_result['ordinary_matched_force']['late_coherent_fraction']:.3f}"),
                 "dark_pair_energy_control": (
                     f"{dark_pair_result['ordinary_matched_energy']['late_coherent_fraction']:.3f}"),
                 "dark_pair_grid_control": (
                     f"{dark_pair_refinements[(2048, 32, 0.00625)]['late_coherent_fraction']:.3f}"),
                 "dark_pair_grid_drift": (
                     f"{dark_pair_refinements[(2048, 32, 0.00625)]['max_closed_energy_drift']:.3e}"),
                 "dark_pair_large_grid": (
                     f"{dark_pair_refinements[(8192, 32, 0.00625)]['late_coherent_fraction']:.3f}"),
                 "dark_pair_large_grid_drift": (
                     f"{dark_pair_refinements[(8192, 32, 0.00625)]['max_closed_energy_drift']:.3e}"),
                 "dark_pair_marker_control": (
                     f"{dark_pair_refinements[(4096, 16, 0.00625)]['late_coherent_fraction']:.3f}"),
                 "dark_pair_dense_markers": (
                     f"{dark_pair_refinements[(4096, 64, 0.00625)]['late_coherent_fraction']:.3f}"),
                 "dark_pair_clock_control": (
                     f"{dark_pair_refinements[(4096, 32, 0.0125)]['late_coherent_fraction']:.3f}"),
                 "dark_pair_clock_drift": (
                     f"{dark_pair_refinements[(4096, 32, 0.0125)]['max_closed_energy_drift']:.3e}"),
                 "dark_pair_final_speed": f"{dark_pair_result['max_final_speed_over_c']:.3f}",
                 "dark_pair_energy_drift": f"{dark_pair_result['max_closed_energy_drift']:.3e}"})
design = EVIDENCE.parent / "profile_design"
run_example('examples/dark_profile.py', full=True, output=design)
design_record = json.loads((design / "run.json").read_text())
p = design_record["results"]
high = p["finite_amplitude"]
small_uniform = sum(p["uniform_held_yields"]) / 4
small_optimized = sum(p["optimized_held_yields"]) / 4
short = sum(high["small_amplitude_window_replay"]["50"]["optimized"]) / 4
long_window = high["small_amplitude_window_replay"]["80"]
gain_80 = 100 * (sum(long_window["optimized"]) / sum(long_window["uniform"]) - 1)
gain = [100 * row["relative_gain"] for row in p["refined_held_replay"].values()]
reference_residual = max(row[1] for rows in p["cold_spectral_reference"].values() for row in rows)
measured.update({"profile_no_wave_E": f"{high['no_wave_max_ordinary_E_V_m']:.3e}",
                 "profile_window_tail_percent": f"{100 * (small_optimized - short) / small_optimized:.2f}",
                 "profile_window_gain_80_percent": f"{gain_80:.2f}",
                 "profile_reference_flux_error": f"{reference_residual:.3e}",
                 "profile_uniform_objective": f"{p['baseline_scores']['uniform']:.7f}",
                 "profile_ramp_objective": f"{p['baseline_scores']['single_ramp']:.7f}",
                 "profile_two_ramp_objective": f"{p['baseline_scores']['two_ramp']:.7f}",
                 "profile_random_objective": f"{p['random_best_objective']:.7f}",
                 "profile_best_objective": f"{p['best']['objective']:.7f}",
                 "profile_held_uniform": f"{small_uniform:.7f}",
                 "profile_held_optimized": f"{small_optimized:.7f}",
                 "profile_refined_gain_min": f"{min(gain):.2f}",
                 "profile_refined_gain_max": f"{max(gain):.2f}",
                 "profile_gradient_error": f"{abs(p['profile_gradient_ad'] - p['profile_gradient_fd']):.3e}",
                 "profile_high_max_speed": f"{high['cold_design_ledger']['max_particle_speed_over_c']:.3f}",
                 "profile_high_cold_yield": f"{sum(high['cold_design_held_yields']) / 4:.7f}",
                 "profile_high_uniform_yield": f"{sum(high['uniform_held_yields']) / 4:.7f}",
                 "profile_high_design_yield": f"{sum(high['high_design_held_yields']) / 4:.7f}",
                 "profile_high_energy_drift": (
                     f"{abs(high['cold_design_ledger']['closed_total_over_incident'] - 1):.3e}"),
                 "profile_high_random_energy": (
                     f"{high['cold_design_ledger']['local_random_kinetic_over_incident']:.3e}")})
measured["_provenance"] = {"cold_exchange": provenance(cold_record, "cold_exchange"),
                           "prescribed_drive": provenance(drive_record, "prescribed_drive"),
                           "density_calibration": provenance(calibration_record, "density_calibration"),
                           "oblique_3v": provenance(oblique_record, "oblique_3v"),
                           "mixed_kinetic": provenance(kinetic_record, "mixed_kinetic"),
                           **physical_records,
                           "bump_on_tail": provenance(bump_record, "bump_on_tail"),
                           "mixed_two_stream": provenance(two_record, "mixed_two_stream"),
                           "two_stream_saturation": provenance(saturation["two_stream_saturation"],
                                                               "two_stream_saturation"),
                           "two_stream_extended": provenance(saturation["two_stream_extended"],
                                                             "two_stream_extended"),
                           "mixed_weibel": provenance(instability_records["mixed_weibel"], "mixed_weibel"),
                           "warm_two_stream": provenance(warm_record, "warm_two_stream"),
                           "homogeneous_null": provenance(null_record, "homogeneous_null"),
                           "mobile_ions": provenance(mobile_record, "mobile_ions"),
                           "oscillating_pair": provenance(pair_record, "oscillating_pair"),
                           "oscillating_dark_pair": provenance(dark_pair_record, "oscillating_dark_pair"),
                           "profile_design": provenance(design_record, "profile_design")}
benchmark = EVIDENCE.parent / "recurrence_benchmark.json"
if benchmark.exists():
    data = json.loads(benchmark.read_text())
    for row in data["rows"]:
        key = f"{row['case']}_{row['method']}"
        measured[f"{key}_grad_s"] = f"{row['warm_gradient_median_s']:.4f}"
        measured[f"{key}_temp_mib"] = f"{row['compiled_temp_bytes'] / 2**20:.2f}"
        measured[f"{key}_rss_mib"] = f"{row['process_peak_bytes'] / 2**20:.1f}"
        measured[f"{key}_first_grad_s"] = f"{row['first_gradient_s']:.3f}"
    measured["_provenance"]["recurrence_benchmark"] = {
        key: data[key] for key in ("git", "jax", "solvax", "equinox", "platform", "backend", "precision_x64")}
field_cost = EVIDENCE.parent / "field_cost.json"
if field_cost.exists():
    data = json.loads(field_cost.read_text())
    for row in data["rows"]:
        name = row["case"]
        measured[f"field_{name}_warm_ms"] = f"{1000 * row['warm_median_s']:.1f}"
        measured[f"field_{name}_first_s"] = f"{row['first_call_s']:.2f}"
        measured[f"field_{name}_rss_mib"] = f"{row['peak_rss_bytes'] / 2**20:.1f}"
    measured["_provenance"]["field_cost"] = data["settings"]
replay_records = {}
for setting in ('default', 'deterministic'):
    folder = EVIDENCE.parent / 'pic_reproducibility' / setting
    native = json.loads((folder / 'run.json').read_text())
    values = native['results']
    replay_records[setting] = values
    measured['_provenance'][f'pic_replay_{setting}'] = provenance(native, f'pic_reproducibility/{setting}')
    for kernel in ('deposition', 'short_run'):
        row, prefix = values[kernel], f'replay_{setting}_{kernel}'
        for key in ('compile_s', 'compiler_temporary_MiB'):
            measured[f'{prefix}_{key}'] = f'{row[key]:.4f}'
        factor = 1000 if kernel == 'deposition' else 1
        samples = [factor * call['warm_s'] for call in row['calls']]
        measured[f'{prefix}_warm'] = f'{factor * row["warm_median_s"]:.4f} [{min(samples):.4f}, {max(samples):.4f}]'
    measured[f'replay_{setting}_rss'] = f'{values["peak_rss_bytes"] / 2**20:.2f}'
    for key in ('numpy_max_error_over_en', 'integrated_charge_error_over_enL'):
        measured[f'replay_{setting}_{key}'] = f'{values["deposition"][key]:.3e}'
for kernel in ('deposition', 'short_run'):
    cost_ratio = (replay_records['deterministic'][kernel]['warm_median_s']
                  / replay_records['default'][kernel]['warm_median_s'])
    measured[f'replay_{kernel}_cost_ratio'] = f'{cost_ratio:.2f}'

large_cost = EVIDENCE.parent / "field_cost_large.json"
if large_cost.exists():
    data = json.loads(large_cost.read_text())
    for row in data["rows"]:
        name = row["case"]
        measured[f"large_{name}_warm_s"] = f"{row['warm_median_s']:.2f}"
        measured[f"large_{name}_first_s"] = f"{row['first_call_s']:.2f}"
        measured[f"large_{name}_rss_mib"] = f"{row['peak_rss_bytes'] / 2**20:.0f}"
    measured["_provenance"]["field_cost_large"] = data["settings"]
storage_cost = EVIDENCE.parent / "storage_cost.json"
if storage_cost.exists():
    data = json.loads(storage_cost.read_text())
    for row in data["rows"]:
        name = row["case"]
        measured[f"storage_{name}_warm_s"] = f"{row['warm_median_s']:.3f}"
        measured[f"storage_{name}_first_s"] = f"{row['first_call_s']:.2f}"
        measured[f"storage_{name}_rss_mib"] = f"{row['peak_rss_bytes'] / 2**20:.1f}"
    measured["_provenance"]["storage_cost"] = data["settings"]
paper = EVIDENCE.parent / "paper_replay"
for name, ratio in (("drive", ".03"), ("zero", "0")):
    run_example('examples/dark_reservoir.py', study='paper', full=True, drive_ratio=float(ratio),
                output=ROOT / 'artifacts' / f'paper_full_{name}')
halfstep = ROOT / "artifacts" / "paper_dt_half"
run_example('examples/dark_reservoir.py', study='paper', full=True, dt=0.01, output=halfstep)
transition = ROOT / "artifacts" / "paper_dt_quarter_transition"
run_example('examples/dark_reservoir.py', study='paper', full=True, dt=0.005, horizon=1000.0, output=transition)
repeat = ROOT / "artifacts" / "paper_dt_half_prefix_replay"
run_example('examples/dark_reservoir.py', study='paper', full=True, dt=0.01, horizon=1000.0, output=repeat)
run_example('docs/scripts/make_paper_replay.py', records=ROOT / 'artifacts', refined=(halfstep, transition),
            repeat=repeat, output=paper)
paper_record = json.loads((paper / "run.json").read_text())
hook = paper_record["results"]
budget = hook["all_step_conservation"]
measured.update({"hook_electron_rms": f"{hook['final_rms_over_c'][0]:.4f}",
                 "hook_ion_rms": f"{hook['final_rms_over_c'][1]:.6f}",
                 "hook_target_electron_rms": f"{hook['digitized_final_rms_over_c'][0]:.4f}",
                 "hook_target_ion_rms": f"{hook['digitized_final_rms_over_c'][1]:.6f}",
                 "hook_electron_difference_percent": f"{100 * hook['final_rms_relative_difference'][0]:.1f}",
                 "hook_ion_difference_percent": f"{100 * hook['final_rms_relative_difference'][1]:.1f}",
                 "hook_electron_B3_increment": f"{hook['final_B3_spread_increment_over_nmec2L'][0]:.5f}",
                 "hook_ion_B3_increment": f"{hook['final_B3_spread_increment_over_nmec2L'][1]:.7f}",
                 "hook_target_electron_B3_increment": (
                     f"{hook['digitized_B3_spread_increment_from_rms_over_nmec2L'][0]:.5f}"),
                 "hook_target_ion_B3_increment": (
                     f"{hook['digitized_B3_spread_increment_from_rms_over_nmec2L'][1]:.7f}"),
                 "hook_electron_B3_difference_percent": f"{100 * hook['B3_increment_relative_difference'][0]:.1f}",
                 "hook_ion_B3_difference_percent": f"{100 * hook['B3_increment_relative_difference'][1]:.1f}",
                 "hook_spread_gain": f"{hook['late_global_spread_over_initial'][0]:.2f}",
                 "hook_no_drive_spread": f"{hook['no_drive_late_global_spread_over_initial']:.4f}",
                 "hook_work_error_percent": f"{100 * budget['max_energy_work_defect_over_peak_injected_work']:.4f}",
                 "hook_momentum": f"{budget['max_momentum_defect_over_nmecL']:.2e}",
                 "hook_gauss": f"{budget['max_ordinary_gauss_over_en_eps0']:.2e}",
                 "hook_continuity": f"{budget['max_continuity_over_enwp']:.2e}"})
measured["_provenance"]["paper_replay"] = provenance(paper_record, "paper_replay")
for label, comparison in zip(("halfstep", "quarter"), hook["refinement_comparisons"]):
    for window in comparison["windows"]:
        end = window["window_omega_p"][1]
        for key in ("mean_E", "electric", "nonzero_electric"):
            error = window["observables"][key]["relative_l2_difference"]
            measured[f"hook_{label}_{end}_{key}_l2_percent"] = f"{100 * error:.4f}"
for window in hook["adjacent_error_contraction"]:
    end = window["window_omega_p"][1]
    for key in ("mean_E", "electric", "nonzero_electric"):
        measured[f"hook_contraction_{end}_{key}"] = f"{window['error_ratio'][key]:.2f}"
for window in hook["repeat_comparison"]["windows"]:
    end = window["window_omega_p"][1]
    for key in ("mean_E", "electric", "nonzero_electric"):
        error = window["observables"][key]["relative_l2_difference"]
        measured[f"hook_repeat_{end}_{key}_l2_percent"] = f"{100 * error:.6g}"
controls = EVIDENCE.parent / "replay_controls"
fixed_first, fixed_repeat = [ROOT / "artifacts" / name for name in ("paper_fixed_first", "paper_fixed_repeat")]
for folder in (fixed_first, fixed_repeat):
    run_example('examples/dark_reservoir.py', study='paper', full=True, dt=.01, horizon=1000.,
                block_horizon=100., local_moments=True, output=folder,
                initial_state=fixed_first / 'initial_state.npz' if folder == fixed_repeat else None)

fixed_fine = ROOT / "artifacts" / "paper_fixed_fine"
run_example('examples/dark_reservoir.py', study='paper', full=True, dt=0.005, horizon=1000.0, block_horizon=100.0,
            local_moments=True, output=fixed_fine)
run_example('docs/scripts/compare_replays.py', first=fixed_first, second=fixed_repeat, constraints=True,
            refined=fixed_fine, destination=controls)
control_record = json.loads((controls / "run.json").read_text())
replay_measurements(control_record, (("hook_fixed", "comparison", ".6g"),
                                     ("hook_fixed_dt", "refinement_comparison", ".4f")))
audits = [row["ordinary"] for row in control_record["results"]["comparison"]["endpoint_constraints"]]
measured["hook_fixed_projection_field"] = f"{max(row['max_correction_over_field_scale'] for row in audits):.2e}"
measured["hook_fixed_projection_energy"] = f"{max(abs(row['energy_change_over_scale']) for row in audits):.2e}"
measured["_provenance"]["replay_controls"] = provenance(control_record, "replay_controls")
late_folder = EVIDENCE.parent / 'late_step_controls'
late_runs = [ROOT / 'artifacts' / 'conversion_controls' / name
             for name in ('paper_dt005', 'paper_dt005_repeat', 'paper_dt0025')]
for index, folder in enumerate(late_runs):
    run_example('examples/dark_reservoir.py', study='paper', cells=2000, particles=206000,
                dt=.0025 if index == 2 else .005, horizon=1000., block_horizon=100., local_moments=True,
                initial_state=late_runs[0] / 'initial_state.npz' if index == 1 else None, output=folder)
run_example('docs/scripts/compare_replays.py', first=late_runs[0], second=late_runs[1], constraints=True,
            refined=late_runs[2], destination=late_folder)
late_record = json.loads((late_folder / 'run.json').read_text())
replay_measurements(late_record, (
    ("late_repeat", "comparison", ".4f"), ("late_dt", "refinement_comparison", ".4f")))
for label, key in (("late_repeat", "comparison"), ("late_dt", "refinement_comparison")):
    a, b = late_record['results'][key]['windows'][-1]['realization_summaries']
    for observable in ('electric_mean', 'work'):
        field = 'work_increment' if observable == 'work' else observable
        measured[f'{label}_{observable}_change_percent'] = f'{100 * (b[field] / a[field] - 1):.2f}'
    for species in range(2):
        increments = [r['local_spread_increment_mean'][species][0] for r in (a, b)]
        change = 100 * (increments[1] / increments[0] - 1)
        measured[f'{label}_local_spread_increment_{species}_change_percent'] = f'{change:.3f}'
    measured[f'{label}_injection_rate_difference'] = f"{b['injection_rate_over_wp'] - a['injection_rate_over_wp']:.3e}"
measured['_provenance']['late_step_controls'] = provenance(late_record, late_folder.name)
resolution = EVIDENCE.parent / "replay_resolution"
mesh, seed, loading = [ROOT / "artifacts" / name for name in
                       ("paper_fixed_mesh", "paper_fixed_seed", "paper_fixed_particles")]
for folder, cells, loading_seed, particles in ((mesh, "2000", "0", "103000"),
                                               (seed, "1000", "1", "103000"),
                                               (loading, "2000", "0", "206000")):
    run_example('examples/dark_reservoir.py', study='paper', full=True, cells=int(cells), seed=int(loading_seed),
                particles=int(particles), dt=0.005, horizon=1000.0, block_horizon=100.0, local_moments=True,
                output=folder)
loading_repeat = ROOT / "artifacts" / "paper_fixed_particles_repeat"
run_example('examples/dark_reservoir.py', study='paper', cells=2000, particles=206000, dt=0.005, horizon=1000.0,
            block_horizon=100.0, local_moments=True, initial_state=loading / 'initial_state.npz',
            output=loading_repeat)
run_example('docs/scripts/compare_replays.py', first=fixed_fine, second=mesh, variant='mesh', constraints=True,
            refined=seed, refined_variant='seed', loading_refined=loading, loading_repeat=loading_repeat,
            destination=resolution)
resolution_record = json.loads((resolution / "run.json").read_text())
replay_measurements(resolution_record, (("hook_mesh", "comparison", ".4f"),
                                        ("hook_seed", "refinement_comparison", ".4f"),
                                        ("hook_loading", "loading_comparison", ".4f"),
                                        ("hook_loading_repeat", "loading_execution_comparison", ".4f")))
for index, label in enumerate(("fine", "mesh", "seed", "loading")):
    native = resolution_record["results"]["native_runs"][index]["results"]
    measured[f"hook_resolution_{label}_balance"] = f"{native['max_energy_work_defect_over_initial_thermal'] * .001:.2e}"
    measured[f"hook_resolution_{label}_momentum"] = f"{native['max_momentum_defect_over_nmecL']:.2e}"
measured["_provenance"]["replay_resolution"] = provenance(resolution_record, resolution.name)
pic = EVIDENCE.parent / "pic_conservation"
run_example('docs/scripts/benchmark_pic_conservation.py', full=True, samples=1, output=pic)
pic_record = json.loads((pic / "run.json").read_text())
rows = pic_record["results"]["rows"]
for label, case, cells, dt in (("explicit", "explicit", 128, .002),
                               ("implicit4", "implicit4", 128, .002),
                               ("implicit8", "implicit8", 128, .002),
                               ("implicit_large", "implicit8", 128, .02),
                               ("proca", "proca", 128, .002),
                               ("proca_halfstep", "proca", 128, .001),
                               ("proca_fine", "proca", 256, .002)):
    row = next(r for r in rows if (r["case"], r["cells"], r["dt_omega_p"]) == (case, cells, dt))
    for suffix, key, fmt in (("energy", "max_energy_error_over_initial", ".2e"),
                             ("momentum", "max_momentum_error_over_nmecL", ".2e"),
                             ("growth", "fitted_growth_over_wp", ".7f"),
                             ("frequency", "fitted_frequency_over_wp", ".2e"),
                             ("dark_work", "max_dark_work_error_over_initial", ".2e"),
                             ("compile", "compile_s", ".3f"), ("warm", "warm_median_s", ".3f")):
        measured[f"pic_{label}_{suffix}"] = format(row[key], fmt)
    measured[f"pic_{label}_temp_mib"] = f"{row['compiler_temporary_bytes'] / 2**20:.2f}"
    measured[f"pic_{label}_rss_mib"] = f"{row['peak_rss_bytes'] / 2**20:.0f}"
measured["_provenance"]["pic_conservation"] = provenance(pic_record, "pic_conservation")
for label in ("dt04", "dt02"):
    folder = EVIDENCE.parent / f"implicit_drive_{label}"
    computational = ROOT / "artifacts" / f"implicit_{label}"
    run_example('docs/scripts/benchmark_implicit_drive.py', cells=256, nodes=8,
                dt=0.04 if label == 'dt04' else 0.02, iterations=8, horizon=1000.0, samples=2,
                gradient_horizon=8.0, output=computational)
    if not records_only:
        folder.mkdir(exist_ok=True)
        for name in ("run.json", "data.npz", "figure.png"):
            copyfile(computational / name, folder / name)
    record = json.loads((folder / "run.json").read_text())
    result = record["results"]
    prefix = f"implicit_drive_{label}"
    for suffix, key, fmt in (("balance", "max_balance_over_nmc2L", ".2e"),
                             ("momentum", "max_momentum_over_nmecL", ".2e"),
                             ("gauss", "max_gauss_over_en_eps0", ".2e"),
                             ("compile", "compile_s", ".2f"),
                             ("gradient_compile", "gradient_compile_s", ".2f")):
        measured[f"{prefix}_{suffix}"] = format(result[key], fmt)
    for suffix, key in (("warm", "warm_primal_s"), ("gradient_warm", "gradient_warm_s")):
        measured[f"{prefix}_{suffix}"] = f"{np.median(result[key]):.2f}"
    for suffix, key in (("temp", "compiler_temporary_bytes"), ("rss", "peak_rss_bytes"),
                        ("gradient_temp", "gradient_temporary_bytes")):
        measured[f"{prefix}_{suffix}_mib"] = f"{result[key] / 2**20:.2f}"
    gradient = np.asarray(result["automatic_gradient"])
    for label_ref, key in (("fd", "finite_difference_gradient"), ("frechet", "frechet_gradient")):
        measured[f"{prefix}_{label_ref}_relative"] = f"{np.max(abs((gradient - result[key]) / result[key])):.2e}"
    with np.load(folder / "data.npz") as arrays:
        mask = arrays["t"] <= 40 + 1e-9
        error = np.linalg.norm((arrays["electric"] - arrays["oracle_E"])[mask])
        measured[f"{prefix}_early_wave_percent"] = f"{100 * error / np.linalg.norm(arrays['oracle_E'][mask]):.4f}"
    measured["_provenance"][prefix] = provenance(record, folder.name)
implicit_paper = ROOT / "artifacts" / "implicit_paper"
method = EVIDENCE.parent / "paper_implicit_comparison"
run_example('docs/scripts/benchmark_implicit_drive.py', paper_loading=True, dt=0.01, iterations=4, horizon=1000.0,
            samples=1, output=implicit_paper, cells=1000, particles=103000)
run_example('docs/scripts/compare_replays.py', first=fixed_first, second=implicit_paper, implicit=True,
            refined=fixed_fine, destination=method)
method_record = json.loads((method / "run.json").read_text())
implicit_result = method_record["results"]["native_runs"][1]["results"]
for suffix, key, fmt in (("balance", "max_balance_over_nmc2L", ".2e"),
                         ("momentum", "max_momentum_over_nmecL", ".2e"),
                         ("gauss", "max_gauss_over_en_eps0", ".2e"), ("compile", "compile_s", ".2f")):
    measured[f"implicit_paper_{suffix}"] = format(implicit_result[key], fmt)
measured["implicit_paper_warm"] = f"{np.median(implicit_result['warm_primal_s']):.2f}"
for suffix, key in (("mean_E", "mean_E"), ("nonzero", "nonzero_electric")):
    measured[f"implicit_paper_repeat_{suffix}_percent"] = (
        f"{100 * implicit_result['execution_variability'][0][key]['1000']['relative_l2']:.2f}")
    for window in method_record["results"]["comparison"]["windows"]:
        value = window["observables"][key]["relative_l2_difference"]
        measured[f"implicit_paper_vs_explicit_{suffix}_{window['end_omega_p']}"] = f"{100 * value:.3f}"
iterations_folder = EVIDENCE.parent / "implicit_iteration_control"
iteration_runs = [ROOT / "artifacts" / f"implicit_picard{count}_exact_40" for count in (4, 8)]
for count, folder in zip((4, 8), iteration_runs):
    run_example('docs/scripts/benchmark_implicit_drive.py', paper_loading=True, dt=0.01, iterations=int(count),
                horizon=40.0, samples=1, initial_state=implicit_paper / 'initial_state.npz', output=folder,
                cells=1000, particles=103000)
orbit_folders = []
for stage in ("initial", "final"):
    for count in (4, 8):
        folder = ROOT / "artifacts" / f"orbit_{stage}_{count}"
        orbit_folders.append(folder)
        run_example('docs/scripts/benchmark_implicit_drive.py', paper_loading=True, dt=0.01, iterations=int(count),
                    audit_state=iteration_runs[0] / f'{stage}_state.npz', output=folder, cells=1000,
                    particles=103000)
run_example('docs/scripts/compare_replays.py', first=iteration_runs[0], second=iteration_runs[1],
            picard=True, orbit_audits=orbit_folders, destination=iterations_folder)
iteration_record = json.loads((iterations_folder / "run.json").read_text())
for count, native in zip((4, 8), iteration_record["results"]["native_runs"]):
    values = native["results"]
    for suffix, key in (("balance", "max_balance_over_nmc2L"), ("momentum", "max_momentum_over_nmecL")):
        measured[f"implicit_iterations_{count}_{suffix}"] = f"{values[key]:.3e}"
    measured[f"implicit_iterations_{count}_warm"] = f"{values['warm_primal_s'][0]:.2f}"
for suffix, key in (("mean_E", "electric"), ("nonzero", "nonzero_electric")):
    measured[f"implicit_iterations_{suffix}_l2"] = (
        f"{iteration_record['results']['observables'][key]['relative_l2_difference']:.3e}")
orbit_error = max(abs(r["results"]["momentum_disagreement_over_nmecL"])
                  for r in iteration_record["results"]["orbit_audits"])
measured["implicit_orbit_momentum_disagreement"] = f"{orbit_error:.3e}"
measured["_provenance"]["implicit_iteration_control"] = provenance(iteration_record, iterations_folder.name)
measured["_provenance"]["paper_implicit_comparison"] = provenance(method_record, method.name)
controls_folder = EVIDENCE.parent / "implicit_method_controls"
control_runs = []
for cells, substeps in ((1000, 2), (1000, 4), (2000, 2), (4000, 2)):
    folder = ROOT / "artifacts" / f"implicit_mesh{cells}_sub{substeps}"
    control_runs.append(folder)
    run_example('docs/scripts/benchmark_implicit_drive.py', paper_loading=True, cells=cells, substeps=substeps,
                dt=0.01, iterations=4, horizon=40.0, samples=1, output=folder,
                initial_state=iteration_runs[0] / 'initial_state.npz' if cells == 1000 else None, particles=103000)
run_example('docs/scripts/compare_replays.py', first=control_runs[0], second=control_runs[1],
            method_controls=True, refined=control_runs[2], finer_mesh=control_runs[3], destination=controls_folder)
controls_record = json.loads((controls_folder / "run.json").read_text())
for native in controls_record["results"]["native_runs"]:
    s, values = native["settings"], native["results"]
    prefix = f"implicit_method_{s['cells']}_{s['substeps']}"
    for suffix, key in (("balance", "max_balance_over_nmc2L"), ("momentum", "max_momentum_over_nmecL"),
                        ("gauss", "max_gauss_over_en_eps0"), ("continuity", "max_continuity_over_enwp")):
        measured[f"{prefix}_{suffix}"] = f"{values[key]:.3e}"
    measured[f"{prefix}_compile"] = f"{values['compile_s']:.2f}"
    measured[f"{prefix}_warm"] = f"{values['warm_primal_s'][0]:.2f}"
for control in ("substep", "mesh", "refined_mesh"):
    for suffix, key in (("mean_E", "electric"), ("nonzero", "nonzero_electric")):
        value = controls_record["results"][f"{control}_observables"][key]["relative_l2_difference"]
        measured[f"implicit_method_{control}_{suffix}_percent"] = f"{100 * value:.4g}"
measured["_provenance"]["implicit_method_controls"] = provenance(controls_record, controls_folder.name)
phase_folder = EVIDENCE.parent / "implicit_grid_phase"
phase_runs = []
for cells in (1000, 2000, 4000):
    anchor = ROOT / "artifacts" / f"phase_{cells}_0"
    for phase in (0., .25, .5):
        folder = ROOT / "artifacts" / f"phase_{cells}_{phase:g}"
        phase_runs.append(folder)
        run_example('docs/scripts/benchmark_implicit_drive.py', paper_loading=True, cells=cells,
                    particles=103000, dt=.01, horizon=40., iterations=4, substeps=2, samples=1,
                    grid_phase=phase, initial_state=anchor / 'initial_state.npz' if phase else None,
                    output=folder)
phase_orbits = []
for cells in (1000, 4000):
    for stage in ('initial', 'final'):
        folder = ROOT / 'artifacts' / f'orbit_phase_{cells}_{stage}'
        phase_orbits.append(folder)
        run_example('docs/scripts/benchmark_implicit_drive.py', paper_loading=True, cells=cells,
                    particles=103000, dt=.01, iterations=4, substeps=2,
                    audit_state=ROOT / 'artifacts' / f'phase_{cells}_0.25' / f'{stage}_state.npz', output=folder)
run_example('docs/scripts/compare_replays.py', phase_controls=phase_runs, orbit_audits=phase_orbits,
            destination=phase_folder)
phase_record = json.loads((phase_folder / 'run.json').read_text())
phase_results = [r['results'] for r in phase_record['results']['native_runs']]
for suffix, key in (('balance', 'max_balance_over_nmc2L'), ('gauss', 'max_gauss_over_en_eps0'),
                    ('continuity', 'max_continuity_over_enwp'), ('force_max', 'max_step_particle_force_over_nmecLwp')):
    measured[f'phase_{suffix}'] = f'{max(r[key] for r in phase_results):.3e}'
for cells in (1000, 2000, 4000):
    rows = [r['results'] for r in phase_record['results']['native_runs'] if r['settings']['cells'] == cells]
    measured[f'phase_momentum_{cells}'] = f"{max(r['max_momentum_over_nmecL'] for r in rows):.3e}"
errors = [r['observables']['nonzero_electric']['relative_l2_difference']
          for r in phase_record['results']['phase_observables']]
measured['phase_nonzero_min_percent'] = f'{100 * min(errors):.3f}'
measured['phase_nonzero_max_percent'] = f'{100 * max(errors):.3f}'
repeat_error = max(e['nonzero_electric']['40']['relative_l2'] for r in phase_results
                   for e in r['execution_variability'])
orbit_error = max(abs(r['results']['momentum_disagreement_over_nmecL'])
                  for r in phase_record['results']['orbit_audits'])
measured['phase_repeat_nonzero_l2'], measured['phase_orbit_impulse'] = f'{repeat_error:.3e}', f'{orbit_error:.3e}'
warm = [t for r in phase_results for t in r['warm_primal_s']]
measured['phase_warm_min'], measured['phase_warm_max'] = f'{min(warm):.2f}', f'{max(warm):.2f}'
measured['phase_temp_mib'] = f"{max(r['compiler_temporary_bytes'] for r in phase_results) / 2**20:.2f}"
measured['phase_rss_mib'] = f"{max(r['peak_rss_bytes'] for r in phase_results) / 2**20:.2f}"
measured['_provenance']['implicit_grid_phase'] = provenance(phase_record, phase_folder.name)
(EVIDENCE.parent / "measurements.json").write_text(json.dumps(measured, indent=2) + "\n")

print("Saved measured documentation substitutions", flush=True)
