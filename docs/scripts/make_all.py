"""Regenerate full-preset midnight evidence and measured MyST substitutions."""

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "docs" / "_static" / "figures" / "cold_exchange"
subprocess.run([sys.executable, str(ROOT / "examples" / "dark_photon.py"), "--full",
                "--output", str(EVIDENCE)], cwd=ROOT, check=True)
record = json.loads((EVIDENCE / "run.json").read_text())
results = record["results"]
measured = {"cold_mean_error": f"{results['max_mean_field_error_over_D0']:.3e}",
            "cold_gauss_error": f"{results['max_dark_gauss_over_enref_eps0']:.3e}",
            "_provenance": record}
(EVIDENCE.parent / "measurements.json").write_text(json.dumps(measured, indent=2) + "\n")
