"""Regenerate full-preset midnight evidence and measured MyST substitutions."""

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "docs" / "_static" / "figures" / "cold_exchange"


def provenance(record, folder):
    """Keep substitutions slim; the complete run record lives beside its figure."""
    keys = ("jaxincell", "jax", "numpy", "python", "platform", "jax_enable_x64", "backend", "git")
    return {**{key: record[key] for key in keys}, "run": f"{folder}/run.json"}


subprocess.run([sys.executable, str(ROOT / "examples" / "dark_photon.py"), "--full",
                "--output", str(EVIDENCE)], cwd=ROOT, check=True)
record = json.loads((EVIDENCE / "run.json").read_text())
results = record["results"]
measured = {"cold_mean_error": f"{results['max_mean_field_error_over_D0']:.3e}",
            "cold_gauss_error": f"{results['max_dark_gauss_over_enref_eps0']:.3e}",
            "cold_energy_error": f"{results['closed_energy_relative_drift']:.3e}",
            "cold_work_error": f"{results['ordinary_energy_vs_dark_work_relative_error']:.3e}",
            "_provenance": record}
drive = EVIDENCE.parent / "prescribed_drive"
subprocess.run([sys.executable, str(ROOT / "examples" / "dark_drive.py"), "--full",
                "--output", str(drive)], cwd=ROOT, check=True)
drive_record = json.loads((drive / "run.json").read_text())
drive_results = drive_record["results"]
measured.update({"drive_wave_error": f"{drive_results['max_waveform_error_over_D0']:.3e}",
                 "drive_work_error": f"{drive_results['ordinary_energy_vs_external_work_relative_error']:.3e}"})
calibration = EVIDENCE.parent / "density_calibration"
subprocess.run([sys.executable, str(ROOT / "examples" / "optimize_dark_photon.py"), "--full",
                "--output", str(calibration)], cwd=ROOT, check=True)
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
subprocess.run([sys.executable, str(ROOT / "examples" / "dark_plasma.py"), "--full",
                "--output", str(oblique)], cwd=ROOT, check=True)
oblique_record = json.loads((oblique / "run.json").read_text())
measured["oblique_six_field_error"] = f"{oblique_record['results']['max_all_six_field_error_over_D0']:.3e}"
kinetic = EVIDENCE.parent / "mixed_kinetic"
subprocess.run([sys.executable, str(ROOT / "examples" / "dark_kinetic.py"), "--full",
                "--output", str(kinetic)], cwd=ROOT, check=True)
kinetic_record = json.loads((kinetic / "run.json").read_text())
kin = kinetic_record["results"]
measured.update({"kinetic_pic_real": f"{kin['measured_real_over_wp']:.5f}",
                 "kinetic_pic_imag": f"{kin['measured_imag_over_wp']:.5f}",
                 "kinetic_root_real": f"{kin['root_real_over_wp']:.5f}",
                 "kinetic_root_imag": f"{kin['root_imag_over_wp']:.5f}",
                 "kinetic_frequency_error": f"{100 * kin['frequency_relative_error']:.2f}",
                 "kinetic_damping_error": f"{100 * kin['damping_relative_error']:.2f}",
                 "kinetic_damping_stderr": f"{kin['damping_slope_stderr']:.4f}"})
null = EVIDENCE.parent / "homogeneous_null"
subprocess.run([sys.executable, str(ROOT / "examples" / "dark_null.py"), "--full",
                "--output", str(null)], cwd=ROOT, check=True)
null_record = json.loads((null / "run.json").read_text())
measured["null_coarse_error"] = f"{null_record['results']['max_mode_amplitude_difference_over_initial'][0]:.3e}"
measured["null_fine_error"] = f"{null_record['results']['max_mode_amplitude_difference_over_initial'][1]:.3e}"
measured["_provenance"] = {"cold_exchange": provenance(record, "cold_exchange"),
                           "prescribed_drive": provenance(drive_record, "prescribed_drive"),
                           "density_calibration": provenance(calibration_record, "density_calibration"),
                           "oblique_3v": provenance(oblique_record, "oblique_3v"),
                           "mixed_kinetic": provenance(kinetic_record, "mixed_kinetic"),
                           "homogeneous_null": provenance(null_record, "homogeneous_null")}
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
(EVIDENCE.parent / "measurements.json").write_text(json.dumps(measured, indent=2) + "\n")
