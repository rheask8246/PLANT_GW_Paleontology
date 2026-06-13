"""Post-training validation plots for Naive Bayes emulator (04c)."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

from plant_paths import plot_run_dir
from models.naive_bayes_emulator import NaiveBayesEmulator, generate_catalog

from lib.emulator_plot_utils import histogram_kl, is_sspc_hyperparam_df


def _lambda_cols_from_df(df: pd.DataFrame) -> List[str]:
    return sorted(
        [c for c in df.columns if c.startswith("lambda_")],
        key=lambda x: int(x.split("_")[1]),
    )


def reference_lambda_row(hp_df: pd.DataFrame, lambda_cols: List[str]) -> np.ndarray:
    if is_sspc_hyperparam_df(hp_df):
        for ch in ["SMT", "CE", "CHE"]:
            ch_rows = hp_df[hp_df["channel"] == ch]
            if len(ch_rows) > 0:
                break
        mid_p1 = float(np.median(ch_rows["sfra"]))
        mid_p2 = float(np.median(ch_rows["mu0"]))
        dists = (ch_rows["sfra"] - mid_p1).abs() + (ch_rows["mu0"] - mid_p2).abs()
        grid_idx = int(dists.idxmin())
    else:
        ce_match = hp_df[
            (hp_df["channel"] == "CE")
            & (hp_df["chi_b"] == 0.2)
            & (hp_df["alpha_CE"] == 1.0)
        ]
        if len(ce_match) == 0:
            ce_match = hp_df[(hp_df["channel"] == "CE") & (hp_df["chi_b"] == 0.2)]
        grid_idx = int(ce_match.index[0]) if len(ce_match) > 0 else 0
    return hp_df.iloc[grid_idx][lambda_cols].values.astype(np.float32)


def run_naive_bayes_emulator_plots(
    model: NaiveBayesEmulator,
    hp_df: pd.DataFrame,
    events_df: pd.DataFrame,
    normalizer: Dict,
    lambda_cols: List[str] | None,
    plot_script_path: Path,
    seed: int = 42,
) -> Path:
    if lambda_cols is None:
        lambda_cols = _lambda_cols_from_df(hp_df)

    plots_dir = plot_run_dir(
        plot_script_path,
        timestamp=datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
    )
    rng = np.random.default_rng(seed)
    lam_ref = reference_lambda_row(hp_df, lambda_cols)
    grid_idx = int(
        np.argmin(
            np.sum((hp_df[lambda_cols].values - lam_ref.reshape(1, -1)) ** 2, axis=1)
        )
    )

    mask = events_df["grid_idx"] == grid_idx
    sub = events_df.loc[mask, ["mchirp", "q", "z"]]
    n_ref = min(2000, len(sub))
    if n_ref > 0:
        idx = rng.choice(len(sub), size=n_ref, replace=len(sub) < n_ref)
        true_raw = sub.iloc[idx].values.astype(np.float32)
    else:
        true_raw = np.zeros((0, 3), dtype=np.float32)

    syn = generate_catalog(lam_ref, max(n_ref, 500), model, normalizer)
    syn_raw = syn[["mchirp", "q", "z"]].values.astype(np.float32)

    obs_cols = ["mchirp", "q", "z"]
    metrics: Dict[str, float] = {}
    if len(true_raw) > 10:
        for col, j in zip(obs_cols, range(3)):
            metrics[f"kl_{col}"] = histogram_kl(true_raw[:, j], syn_raw[:, j])

    # 02d-style KDE marginals
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, col, j in zip(axes, obs_cols, range(3)):
        if len(true_raw) > 10:
            try:
                kde_true = gaussian_kde(true_raw[:, j])
                kde_syn = gaussian_kde(syn_raw[:, j])
                x_plot = np.linspace(
                    min(true_raw[:, j].min(), syn_raw[:, j].min()),
                    max(true_raw[:, j].max(), syn_raw[:, j].max()),
                    200,
                )
                ax.plot(x_plot, kde_true(x_plot), "b-", lw=2, label="Parquet")
                ax.plot(x_plot, kde_syn(x_plot), "r-", lw=2, label="NB emulator")
            except Exception:
                ax.hist(true_raw[:, j], bins=40, alpha=0.5, density=True, label="Parquet")
                ax.hist(syn_raw[:, j], bins=40, alpha=0.5, density=True, label="NB emulator")
            kl = metrics.get(f"kl_{col}", float("nan"))
            ax.set_title(f"{col} — KL = {kl:.3f}")
        else:
            ax.hist(syn_raw[:, j], bins=40, alpha=0.5, density=True, label="NB emulator")
        ax.set_xlabel(col)
        ax.legend(fontsize=8)
    mode = model.mode
    fig.suptitle(f"Naive Bayes ({mode}) vs grid {grid_idx}")
    fig.tight_layout()
    fig.savefig(plots_dir / "02d_1d_marginals.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Legacy single-panel histogram (kept for backward compatibility)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for ax, col, j in zip(axes, obs_cols, range(3)):
        if len(true_raw) > 0:
            ax.hist(true_raw[:, j], bins=40, alpha=0.5, density=True, label="parquet")
        ax.hist(syn_raw[:, j], bins=40, alpha=0.5, density=True, label="NB emulator")
        ax.set_xlabel(col)
        ax.legend(fontsize=8)
    fig.suptitle(f"Naive Bayes ({mode}) vs grid {grid_idx} subsample")
    fig.tight_layout()
    fig.savefig(plots_dir / "marginals_ref_grid.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    meta = {
        "mode": mode,
        "kernel_bandwidth": float(model.kernel_bandwidth.item()),
        "lambda_cols": lambda_cols,
        "grid_idx": grid_idx,
        "validation": metrics,
    }
    (plots_dir / "plot_summary.json").write_text(json.dumps(meta, indent=2))
    print(f"  Saved plots → {plots_dir}")
    return plots_dir
