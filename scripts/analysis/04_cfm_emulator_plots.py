#!/usr/bin/env python3
"""
Step 04 — CFM emulator validation plots (checkpoint only, no retraining).

Reads ``checkpoints/cfm_final.pt`` and Step 02 parquet/CSV. Optional training curves
from ``checkpoints/cfm_training_metrics.json`` (written by ``04_cfm_emulator.py``).

SLURM: ``slurm/04_cfm_emulator_plots.sh``
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

import numpy as np
import pandas as pd
import torch

from plant_paths import (  # noqa: E402
    ALL_EVENTS_PARQUET,
    CHECKPOINT_DIR,
    HYPERPARAM_TABLE_ENCODED_CSV,
    PROJECT_ROOT,
    SPLITS_JSON,
    ml_data_dir,
)
from lib.generative_emulator_plots import TrainingMetrics, run_generative_emulator_plots


def main() -> None:
    parser = argparse.ArgumentParser(description="CFM emulator validation plots (no training).")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=CHECKPOINT_DIR / "cfm_final.pt",
        help="Trained CFM checkpoint (default: checkpoints/cfm_final.pt).",
    )
    parser.add_argument(
        "--training-metrics",
        type=Path,
        default=None,
        help="JSON from training (default: <checkpoint_stem>_training_metrics.json).",
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ckpt_path = args.checkpoint.resolve()
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint {ckpt_path}. Run 04_cfm_emulator.py first.")

    metrics_path = args.training_metrics
    if metrics_path is None:
        metrics_path = ckpt_path.with_name(ckpt_path.stem + "_training_metrics.json")
    training_metrics = None
    if metrics_path.is_file():
        training_metrics = TrainingMetrics.from_json(metrics_path)
        print(f"  Loaded training metrics from {metrics_path}")
    else:
        print(f"  No training metrics at {metrics_path} (generation plots only).")

    data_dir = ml_data_dir()
    hp_csv = data_dir / HYPERPARAM_TABLE_ENCODED_CSV.name
    events_pq = data_dir / ALL_EVENTS_PARQUET.name
    splits_path = data_dir / SPLITS_JSON.name
    for p in (hp_csv, events_pq, splits_path):
        if not p.exists():
            raise FileNotFoundError(f"Missing {p}. Run 02_build_dataset.py first.")

    device = torch.device(args.device)
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    lambda_cols = ck["lambda_cols"]
    normalizer = ck["normalizer"]
    hdim = int(ck.get("hidden_dim", 256))
    ctxd = int(ck.get("context_dim", 128))

    from models.cfm_emulator import CFMEmulator

    model = CFMEmulator(lambda_dim=len(lambda_cols), context_dim=ctxd, hidden_dim=hdim)
    model.load_state_dict(ck["model_state"], strict=True)
    model.to(device)
    model.eval()

    hp_df = pd.read_csv(hp_csv)
    events_df = pd.read_parquet(events_pq, columns=["mchirp", "q", "z", "grid_idx"])
    rng = np.random.default_rng(args.seed)

    tm = training_metrics
    plots_dir = run_generative_emulator_plots(
        emulator_kind="cfm",
        model=model,
        normalizer=normalizer,
        hp_df=hp_df,
        events_df=events_df,
        train_losses=tm.train_losses if tm else [],
        val_losses=tm.val_losses if tm else [],
        grad_norms=tm.grad_norms if tm else [],
        loss_0=tm.loss_0 if tm else 0.0,
        loss_500=tm.loss_final if tm else 0.0,
        work_dir=PROJECT_ROOT,
        splits_path=splits_path,
        device=device,
        rng=rng,
        steps=tm.steps if tm else int(ck.get("steps", 0)),
        lambda_cols=lambda_cols,
        plot_script_path=Path(__file__),
        training_metrics=tm,
    )
    print(f"Done. Plots in {plots_dir}")


if __name__ == "__main__":
    main()
