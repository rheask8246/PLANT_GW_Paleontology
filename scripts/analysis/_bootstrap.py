"""Shared import bootstrap for scripts under ``scripts/analysis/``."""
from __future__ import annotations

import sys
from pathlib import Path

ANALYSIS_DIR = Path(__file__).resolve().parent
# scripts/analysis/ → parents[1] is PLANT_GW_Paleontology/
PROJECT_ROOT = ANALYSIS_DIR.parents[1]


def setup() -> Path:
    """Put project root and ``analysis/`` on ``sys.path``; enable ``models/`` imports."""
    for p in (PROJECT_ROOT, ANALYSIS_DIR):
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)
    from plant_paths import ensure_paths  # noqa: WPS433

    ensure_paths()
    return PROJECT_ROOT
