"""Register Demo 18, then execute one of Isaac Lab's official CLI scripts."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

import franka_tray_mimic  # noqa: F401 - importing performs Gym registration


if len(sys.argv) < 2:
    raise SystemExit(
        "Usage: ./isaaclab.sh -p 18_franka_isaac_lab_mimic/run_official.py "
        "<IsaacLab script path> [script arguments...]"
    )

script_path = Path(sys.argv[1]).resolve()
if not script_path.is_file():
    raise SystemExit(f"Isaac Lab script does not exist: {script_path}")

sys.argv = [str(script_path), *sys.argv[2:]]
runpy.run_path(str(script_path), run_name="__main__")
