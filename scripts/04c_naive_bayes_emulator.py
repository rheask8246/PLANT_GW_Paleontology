#!/usr/bin/env python3
"""
Naive Bayes emulator baseline (Step 04c): fit per-grid statistics from Step 02 data.

No gradient training — aggregates all_events.parquet into checkpoints/naive_bayes_final.pt.
Plots: scripts/analysis/04c_naive_bayes_emulator_plots.py (or slurm/04c_naive_bayes_emulator_plots.sh).
"""

from __future__ import annotations

import sys
from pathlib import Path as _Path

_PROJECT_ROOT = _Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from plant_paths import (  # noqa: E402
    ALL_EVENTS_PARQUET,
    CHECKPOINT_DIR,
    HYPERPARAM_TABLE_ENCODED_CSV,
    OBS_NORMALIZER_JSON,
    ensure_paths,
    ml_data_dir,
)

ensure_paths()

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from models.cfm_emulator import normalize_obs
from models.naive_bayes_emulator import NaiveBayesEmulator, _lambda_cols, save_checkpoint


def configure_worker_threads(workers: int) -> int:
    """Set CPU thread workers for BLAS/OpenMP (NB is CPU-bound)."""
    w = max(1, int(workers))
    os.environ["OMP_NUM_THREADS"] = str(w)
    os.environ["MKL_NUM_THREADS"] = str(w)
    os.environ["OPENBLAS_NUM_THREADS"] = str(w)
    os.environ["NUMEXPR_NUM_THREADS"] = str(w)
    return w


def load_normalizer(path: Path) -> Dict:
    with open(path) as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit Naive Bayes emulator baseline (04c).")
    parser.add_argument(
        "--mode",
        choices=("gaussian", "nearest"),
        default="gaussian",
        help="Sampling mode at inference (default: gaussian mixture).",
    )
    parser.add_argument(
        "--bandwidth",
        type=float,
        default=None,
        help="Λ-kernel bandwidth τ (default: median pairwise grid distance).",
    )
    parser.add_argument(
        "--output-checkpoint",
        type=Path,
        default=None,
        help="Output path (default: checkpoints/naive_bayes_final.pt).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")),
        help="CPU worker threads for parquet aggregation / BLAS/OpenMP (default: SLURM_CPUS_PER_TASK or 1).",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    workers = configure_worker_threads(args.workers)

    data_dir = ml_data_dir()
    hp_csv = data_dir / HYPERPARAM_TABLE_ENCODED_CSV.name
    events_pq = data_dir / ALL_EVENTS_PARQUET.name
    nrm_path = OBS_NORMALIZER_JSON

    for p in (hp_csv, events_pq, nrm_path):
        if not p.exists():
            raise FileNotFoundError(f"Missing {p}. Run 02_build_dataset.py first.")

    print("=" * 60)
    print("Naive Bayes emulator fit (04c)")
    print("=" * 60)
    print(f"  Worker threads: {workers}")

    hp_df = pd.read_csv(hp_csv)
    events_df = pd.read_parquet(
        events_pq,
        columns=["mchirp", "q", "z", "grid_idx"],
    )
    normalizer = load_normalizer(nrm_path)
    lambda_cols = _lambda_cols(hp_df)

    print(f"  Grid points: {len(hp_df)}  |  lambda dim: {len(lambda_cols)}  |  mode: {args.mode}")
    print(f"  Events: {len(events_df):,}")

    model = NaiveBayesEmulator.fit_from_data(
        hp_df,
        events_df,
        normalizer,
        mode=args.mode,
        bandwidth=args.bandwidth,
    )
    tau = float(model.kernel_bandwidth.item())
    print(f"  Kernel bandwidth τ = {tau:.4f}")

    ckpt_path = args.output_checkpoint or (CHECKPOINT_DIR / "naive_bayes_final.pt")
    save_checkpoint(ckpt_path, model, normalizer, lambda_cols, args.seed)
    print(f"  Saved checkpoint → {ckpt_path}")
    print("  Plots: python scripts/analysis/04c_naive_bayes_emulator_plots.py")
    print("Done.")


if __name__ == "__main__":
    main()
