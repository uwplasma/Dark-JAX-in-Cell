"""MyST documentation for the midnight Maxwell–Proca package."""

import json
from pathlib import Path

project = "Dark-JAX-in-Cell"
extensions = ["myst_parser"]
source_suffix = {".md": "markdown"}
master_doc = "index"
html_theme = "alabaster"
html_static_path = ["_static"]
html_css_files = ["dark.css"]
exclude_patterns = ["_build"]
myst_enable_extensions = ["dollarmath", "amsmath", "substitution"]
_measured = Path(__file__).parent / "_static" / "figures" / "measurements.json"
myst_substitutions = json.loads(_measured.read_text()) if _measured.exists() else {}
