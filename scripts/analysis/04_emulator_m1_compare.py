#!/usr/bin/env python3
"""
Step 04 — Compare CFM vs diffusion emulator m₁ distributions at a fixed SSPC Λ.

Requires trained ``checkpoints/cfm_final.pt`` and ``checkpoints/diffusion_final.pt``.

Usage::

    python scripts/analysis/04_emulator_m1_compare.py --device cuda

SLURM: ``slurm/09_emulator_m1_distribution.sh``
"""
from __future__ import annotations

import sys
from pathlib import Path

_ANALYSIS_DIR = Path(__file__).resolve().parent
if str(_ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(_ANALYSIS_DIR))

from _bootstrap import setup  # noqa: E402

setup()

from lib.distribution import run_emulator_m1_compare_cli  # noqa: E402

if __name__ == "__main__":
    run_emulator_m1_compare_cli(script_path=Path(__file__))
