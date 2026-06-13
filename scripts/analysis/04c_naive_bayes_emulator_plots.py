#!/usr/bin/env python3
"""
Step 04c — Naive Bayes emulator validation plots (checkpoint only, no refit).

SLURM: ``slurm/04c_naive_bayes_emulator_plots.sh``
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ANALYSIS_DIR = Path(__file__).resolve().parent
if str(_ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(_ANALYSIS_DIR))

from _bootstrap import setup  # noqa: E402

setup()

import pandas as pd
import torch

from plant_paths import (  # noqa: E402
    ALL_EVENTS_PARQUET,
    CHECKPOINT_DIR,
    HYPERPARAM_TABLE_ENCODED_CSV,
    OBS_NORMALIZER_JSON,
    ml_data_dir,
)
from models.naive_bayes_emulator import load_from_checkpoint
from lib.naive_bayes_emulator_plots import run_naive_bayes_emulator_plots


def main() -> None:
    parser = argparse.ArgumentParser(description="Naive Bayes emulator plots (no refit).")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=CHECKPOINT_DIR / "naive_bayes_final.pt",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ckpt_path = args.checkpoint.resolve()
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Missing {ckpt_path}. Run 04c_naive_bayes_emulator.py first.")

    data_dir = ml_data_dir()
    hp_csv = data_dir / HYPERPARAM_TABLE_ENCODED_CSV.name
    events_pq = data_dir / ALL_EVENTS_PARQUET.name
    if not hp_csv.exists() or not events_pq.exists():
        raise FileNotFoundError("Run 02_build_dataset.py first.")

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model, lambda_cols, normalizer = load_from_checkpoint(ck, device=torch.device("cpu"))

    hp_df = pd.read_csv(hp_csv)
    events_df = pd.read_parquet(events_pq, columns=["mchirp", "q", "z", "grid_idx"])

    plots_dir = run_naive_bayes_emulator_plots(
        model=model,
        hp_df=hp_df,
        events_df=events_df,
        normalizer=normalizer,
        lambda_cols=lambda_cols,
        plot_script_path=Path(__file__),
        seed=args.seed,
    )
    print(f"Done. Plots in {plots_dir}")


if __name__ == "__main__":
    main()
