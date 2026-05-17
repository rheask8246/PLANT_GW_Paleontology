#!/usr/bin/env python3
"""
Step 00 — Compare TNG Figure 5/4 mass and redshift-rate distributions with SSPC HDF5.

Reads ``data/sspc/models_sspc.hdf5`` (from ``00_sspc_data_generation.py``) and optional
TNG reference data under ``Fit_SFRD_TNG/data/``.

Usage (from project root)::

    python scripts/analysis/00_distribution_compare.py
    python scripts/analysis/00_distribution_compare.py --sspc-hdf5 data/sspc/models_sspc.hdf5

SLURM: ``slurm/06a_distribution_analysis.sh``
"""
from __future__ import annotations

import sys
from pathlib import Path

_ANALYSIS_DIR = Path(__file__).resolve().parent
if str(_ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(_ANALYSIS_DIR))

from _bootstrap import setup  # noqa: E402

setup()

from lib.distribution import main  # noqa: E402

if __name__ == "__main__":
    main(script_path=Path(__file__))
