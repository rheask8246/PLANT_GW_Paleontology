#!/usr/bin/env python3
"""
Step 04 — KL-divergence heatmaps on the (sfr_a, mu0) grid.

Creates 6 figures:
  - naive_bayes vs train
  - cfm vs train
  - diffusion vs train
  - naive_bayes vs test
  - cfm vs test
  - diffusion vs test

Each figure has 9 subplots (3 observables x 3 channels):
  rows    = mchirp, q, z
  columns = SMT, CE, CHE

Every panel is a heatmap over (sfr_a, mu0) where color is:
  KL( truth_split(observable | grid cell) || model_prediction(observable | grid cell) )

SLURM: ``slurm/04_kl_divergence_heatmaps.sh``
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

_ANALYSIS_DIR = Path(__file__).resolve().parent
if str(_ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(_ANALYSIS_DIR))

from _bootstrap import setup  # noqa: E402

setup()

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from lib.grid_heatmap_plot import pcolormesh_sfra_mu0  # noqa: E402
from plant_paths import (  # noqa: E402
    ALL_EVENTS_PARQUET,
    CHECKPOINT_DIR,
    HYPERPARAM_TABLE_ENCODED_CSV,
    SPLITS_JSON,
    ml_data_dir,
    resolve_plot_output,
)
from sspc_param_ranges import MU0_RANGE, SFRA_RANGE  # noqa: E402
from models.cfm_emulator import CFMEmulator
from models.diffusion_emulator import DiffusionEmulator
from models.naive_bayes_emulator import load_from_checkpoint


CHANNELS = ("SMT", "CE", "CHE")
OBS_COLS = ("mchirp", "q", "z")
SPLITS = ("train", "test")
MODELS = ("naive_bayes", "cfm", "diffusion")


def configure_worker_threads(workers: int) -> int:
    """Set threaded-kernel worker counts for CPU-heavy parquet/math operations."""
    w = max(1, int(workers))
    os.environ["OMP_NUM_THREADS"] = str(w)
    os.environ["MKL_NUM_THREADS"] = str(w)
    os.environ["OPENBLAS_NUM_THREADS"] = str(w)
    os.environ["NUMEXPR_NUM_THREADS"] = str(w)
    torch.set_num_threads(w)
    return w


def _lambda_cols_from_df(df: pd.DataFrame) -> List[str]:
    return sorted(
        [c for c in df.columns if c.startswith("lambda_")],
        key=lambda x: int(x.split("_")[1]),
    )


def _histogram_kl(x_true: np.ndarray, x_syn: np.ndarray, bins: int = 50) -> float:
    lo = min(float(np.min(x_true)), float(np.min(x_syn)))
    hi = max(float(np.max(x_true)), float(np.max(x_syn)))
    if hi <= lo:
        return 0.0
    edges = np.linspace(lo, hi, bins + 1)
    p, _ = np.histogram(x_true, bins=edges, density=True)
    q, _ = np.histogram(x_syn, bins=edges, density=True)
    eps = 1e-10
    p = p + eps
    q = q + eps
    p = p / np.sum(p)
    q = q / np.sum(q)
    return float(np.sum(p * np.log(p / q)))


def _ensure_sspc_grid_axes(hp_df: pd.DataFrame) -> pd.DataFrame:
    out = hp_df.copy()
    if "sfra" in out.columns and "mu0" in out.columns:
        out["sfra"] = out["sfra"].astype(np.float64).round(8)
        out["mu0"] = out["mu0"].astype(np.float64).round(8)
        return out

    if "key" in out.columns:
        # Preferred for SSPC: exact lattice coordinates like /CH/sfra0300/mu00100
        sfra_vals = []
        mu0_vals = []
        for key in out["key"].astype(str).tolist():
            parts = key.strip("/").split("/")
            if len(parts) >= 3:
                m1 = re.fullmatch(r"sfra(-?\d+)", parts[1])
                m2 = re.fullmatch(r"mu0(-?\d+)", parts[2])
                if m1 and m2:
                    sfra_vals.append(int(m1.group(1)) / 10000.0)
                    mu0_vals.append(int(m2.group(1)) / 10000.0)
                    continue
            sfra_vals.append(np.nan)
            mu0_vals.append(np.nan)
        sfra_arr = np.asarray(sfra_vals, dtype=np.float64)
        mu0_arr = np.asarray(mu0_vals, dtype=np.float64)
        if np.isfinite(sfra_arr).all() and np.isfinite(mu0_arr).all():
            out["sfra"] = np.round(sfra_arr, 8)
            out["mu0"] = np.round(mu0_arr, 8)
            return out

    if {"sspc_sfr_a_mean", "sspc_mu0_mean"}.issubset(out.columns):
        out["sfra"] = out["sspc_sfr_a_mean"].astype(np.float64).round(8)
        out["mu0"] = out["sspc_mu0_mean"].astype(np.float64).round(8)
        return out

    if {"chi_b", "alpha_CE"}.issubset(out.columns):
        # Some SSPC exports store grid axes under legacy chi_b/alpha_CE names.
        out["sfra"] = out["chi_b"].astype(np.float64).round(8)
        out["mu0"] = out["alpha_CE"].astype(np.float64).round(8)
        return out

    raise ValueError(
        "Could not infer SSPC grid axes. Expected sfra/mu0, "
        "sspc_sfr_a_mean/sspc_mu0_mean, or chi_b/alpha_CE columns."
    )


def _load_models(device: torch.device) -> Dict[str, Dict]:
    out: Dict[str, Dict] = {}

    # Naive Bayes
    nb_ck = torch.load(CHECKPOINT_DIR / "naive_bayes_final.pt", map_location="cpu", weights_only=False)
    nb_model, nb_lambda_cols, nb_normalizer = load_from_checkpoint(nb_ck, device=torch.device("cpu"))
    out["naive_bayes"] = {
        "model": nb_model,
        "normalizer": nb_normalizer,
        "lambda_cols": nb_lambda_cols,
    }

    # CFM
    cfm_ck = torch.load(CHECKPOINT_DIR / "cfm_final.pt", map_location=device, weights_only=False)
    cfm_lambda_cols = cfm_ck["lambda_cols"]
    cfm_model = CFMEmulator(
        lambda_dim=len(cfm_lambda_cols),
        context_dim=int(cfm_ck.get("context_dim", 128)),
        hidden_dim=int(cfm_ck.get("hidden_dim", 256)),
    )
    cfm_model.load_state_dict(cfm_ck["model_state"], strict=True)
    cfm_model.to(device)
    cfm_model.eval()
    out["cfm"] = {
        "model": cfm_model,
        "normalizer": cfm_ck["normalizer"],
        "lambda_cols": cfm_lambda_cols,
    }

    # Diffusion
    diff_ck = torch.load(CHECKPOINT_DIR / "diffusion_final.pt", map_location=device, weights_only=False)
    diff_lambda_cols = diff_ck["lambda_cols"]
    diff_model = DiffusionEmulator(
        lambda_dim=len(diff_lambda_cols),
        context_dim=int(diff_ck.get("context_dim", 128)),
        hidden_dim=int(diff_ck.get("hidden_dim", 256)),
        n_timesteps=int(diff_ck.get("n_timesteps", 100)),
    )
    diff_model.load_state_dict(diff_ck["model_state"], strict=True)
    diff_model.to(device)
    diff_model.eval()
    out["diffusion"] = {
        "model": diff_model,
        "normalizer": diff_ck["normalizer"],
        "lambda_cols": diff_lambda_cols,
    }

    return out


def _generate_catalog(
    model_name: str,
    model,
    normalizer: Dict,
    lambda_vec: np.ndarray,
    n_events: int,
) -> pd.DataFrame:
    if model_name == "naive_bayes":
        from models.naive_bayes_emulator import generate_catalog as generate_nb

        return generate_nb(lambda_vec, n_events, model, normalizer)
    if model_name == "cfm":
        from models.cfm_emulator import generate_catalog as generate_cfm

        return generate_cfm(lambda_vec, n_events, model, normalizer)
    if model_name == "diffusion":
        from models.diffusion_emulator import generate_catalog as generate_diff

        return generate_diff(lambda_vec, n_events, model, normalizer)
    raise ValueError(f"Unknown model_name {model_name}")


def _sample_truth_from_parquet(
    events_parquet: Path,
    grid_idx: int,
    n_events: int,
    rng: np.random.Generator,
) -> np.ndarray:
    sub = pd.read_parquet(
        events_parquet,
        columns=["mchirp", "q", "z", "grid_idx"],
        filters=[("grid_idx", "==", int(grid_idx))],
    )
    if len(sub) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    n_take = min(n_events, len(sub))
    idx = rng.choice(len(sub), size=n_take, replace=False)
    return sub.iloc[idx][["mchirp", "q", "z"]].values.astype(np.float32)


def _collect_truth_samples_for_split(
    *,
    events_parquet: Path,
    split_indices: List[int],
    events_per_grid: int,
    rng: np.random.Generator,
) -> Dict[int, np.ndarray]:
    truth_cache: Dict[int, np.ndarray] = {}
    for grid_idx in split_indices:
        truth_cache[int(grid_idx)] = _sample_truth_from_parquet(
            events_parquet=events_parquet,
            grid_idx=int(grid_idx),
            n_events=events_per_grid,
            rng=rng,
        )
    return truth_cache


def _compute_kl_table(
    *,
    hp_df: pd.DataFrame,
    split_indices: List[int],
    truth_cache: Dict[int, np.ndarray],
    model_name: str,
    model_bundle: Dict,
) -> pd.DataFrame:
    hp_ref_lambda_cols = _lambda_cols_from_df(hp_df)
    model_lambda_cols = model_bundle["lambda_cols"]
    model = model_bundle["model"]
    normalizer = model_bundle["normalizer"]

    rows = []
    for grid_idx in split_indices:
        row = hp_df.iloc[int(grid_idx)]
        channel = str(row["channel"])
        if channel not in CHANNELS:
            continue

        true_xyz = truth_cache.get(int(grid_idx), np.zeros((0, 3), dtype=np.float32))
        true_df = pd.DataFrame(true_xyz, columns=OBS_COLS)
        if len(true_df) < 8:
            continue

        if model_lambda_cols == hp_ref_lambda_cols:
            lambda_vec = row[hp_ref_lambda_cols].values.astype(np.float32)
        else:
            lambda_vec = row[model_lambda_cols].values.astype(np.float32)

        syn_df = _generate_catalog(
            model_name=model_name,
            model=model,
            normalizer=normalizer,
            lambda_vec=lambda_vec,
            n_events=len(true_df),
        )

        rows.append(
            {
                "grid_idx": int(grid_idx),
                "channel": channel,
                "sfra": float(row["sfra"]),
                "mu0": float(row["mu0"]),
                "kl_mchirp": float(_histogram_kl(true_df["mchirp"].values, syn_df["mchirp"].values)),
                "kl_q": float(_histogram_kl(true_df["q"].values, syn_df["q"].values)),
                "kl_z": float(_histogram_kl(true_df["z"].values, syn_df["z"].values)),
            }
        )

    return pd.DataFrame(rows)


def _compute_pooled_kl(
    *,
    hp_df: pd.DataFrame,
    split_indices: List[int],
    truth_cache: Dict[int, np.ndarray],
    model_name: str,
    model_bundle: Dict,
    channel: str,
) -> Dict[str, float]:
    """
    Pooled (mixture) KL over a split: concatenate truth/model samples across all cells,
    then compute KL on the pooled 1D marginals. This matches how 04d density overlays
    are visually interpreted (split-wise mixture distributions).
    """
    hp_ref_lambda_cols = _lambda_cols_from_df(hp_df)
    model_lambda_cols = model_bundle["lambda_cols"]
    model = model_bundle["model"]
    normalizer = model_bundle["normalizer"]

    truth_parts: List[np.ndarray] = []
    syn_parts: List[np.ndarray] = []

    for grid_idx in split_indices:
        row = hp_df.iloc[int(grid_idx)]
        row_channel = str(row["channel"])
        if row_channel != channel:
            continue

        true_xyz = truth_cache.get(int(grid_idx), np.zeros((0, 3), dtype=np.float32))
        if true_xyz.shape[0] < 8:
            continue

        if model_lambda_cols == hp_ref_lambda_cols:
            lambda_vec = row[hp_ref_lambda_cols].values.astype(np.float32)
        else:
            lambda_vec = row[model_lambda_cols].values.astype(np.float32)

        syn_df = _generate_catalog(
            model_name=model_name,
            model=model,
            normalizer=normalizer,
            lambda_vec=lambda_vec,
            n_events=int(true_xyz.shape[0]),
        )
        syn_xyz = syn_df[list(OBS_COLS)].values.astype(np.float32, copy=False)

        truth_parts.append(true_xyz)
        syn_parts.append(syn_xyz)

    if not truth_parts or not syn_parts:
        return {"kl_mchirp": float("nan"), "kl_q": float("nan"), "kl_z": float("nan")}

    truth_all = np.concatenate(truth_parts, axis=0)
    syn_all = np.concatenate(syn_parts, axis=0)

    return {
        "kl_mchirp": float(_histogram_kl(truth_all[:, 0], syn_all[:, 0])),
        "kl_q": float(_histogram_kl(truth_all[:, 1], syn_all[:, 1])),
        "kl_z": float(_histogram_kl(truth_all[:, 2], syn_all[:, 2])),
    }


def _summarize_kl_tables(
    metrics: pd.DataFrame,
    pooled_rows: List[Dict[str, object]],
) -> pd.DataFrame:
    """
    Combine per-cell KL summaries (mean/median) with pooled KL (mixture-level).
    """
    per_cell = (
        metrics.groupby(["split", "model", "channel"], as_index=False)
        .agg(
            kl_mchirp_mean=("kl_mchirp", "mean"),
            kl_mchirp_median=("kl_mchirp", "median"),
            kl_q_mean=("kl_q", "mean"),
            kl_q_median=("kl_q", "median"),
            kl_z_mean=("kl_z", "mean"),
            kl_z_median=("kl_z", "median"),
        )
    )

    pooled = pd.DataFrame(pooled_rows)
    # pooled has split/model/channel + kl_*_pooled
    out = per_cell.merge(pooled, on=["split", "model", "channel"], how="left")
    return out


def _plot_summary_bars(summary_df: pd.DataFrame, out_path: Path) -> None:
    """
    Plot pooled vs per-cell mean/median KL for quick comparison.
    """
    fig, axes = plt.subplots(3, 3, figsize=(13.5, 8.0), constrained_layout=True)
    obs_to_row = {"mchirp": 0, "q": 1, "z": 2}
    split_to_col = {"train": 0, "test": 1}
    # Third column: pooled across channels
    for split_name in SPLITS:
        for obs in OBS_COLS:
            r = obs_to_row[obs]
            c = split_to_col[split_name]
            ax = axes[r, c]
            block = summary_df[summary_df["split"] == split_name]
            # average across channels for plotting
            agg = block.groupby("model", as_index=False).mean(numeric_only=True)
            x = np.arange(len(MODELS))
            width = 0.25
            ax.bar(
                x - width,
                [agg.loc[agg["model"] == m, f"kl_{obs}_mean"].values[0] for m in MODELS],
                width=width,
                label="mean per-cell",
            )
            ax.bar(
                x,
                [agg.loc[agg["model"] == m, f"kl_{obs}_median"].values[0] for m in MODELS],
                width=width,
                label="median per-cell",
            )
            ax.bar(
                x + width,
                [agg.loc[agg["model"] == m, f"kl_{obs}_pooled"].values[0] for m in MODELS],
                width=width,
                label="pooled (mixture)",
            )
            ax.set_xticks(x)
            ax.set_xticklabels(list(MODELS), rotation=20, ha="right", fontsize=9)
            ax.set_ylabel("KL(True || Model)")
            ax.set_title(f"{obs} — {split_name}")
            ax.grid(True, alpha=0.25, linewidth=0.8)
            if r == 0 and c == 0:
                ax.legend(fontsize=8, frameon=False)

    # Rightmost column: per-channel pooled KL (as a small table-like heatmap)
    for r, obs in enumerate(OBS_COLS):
        ax = axes[r, 2]
        # rows = channels, cols = models (avg over splits for compactness)
        arr = np.full((len(CHANNELS), len(MODELS)), np.nan, dtype=float)
        for i, ch in enumerate(CHANNELS):
            for j, m in enumerate(MODELS):
                sub = summary_df[(summary_df["channel"] == ch) & (summary_df["model"] == m)]
                arr[i, j] = float(np.nanmean(sub[f"kl_{obs}_pooled"].values))
        im = ax.imshow(arr, aspect="auto", cmap="viridis")
        ax.set_yticks(np.arange(len(CHANNELS)))
        ax.set_yticklabels(CHANNELS)
        ax.set_xticks(np.arange(len(MODELS)))
        ax.set_xticklabels(MODELS, rotation=20, ha="right", fontsize=9)
        ax.set_title(f"{obs} pooled KL (avg splits)")
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                if np.isfinite(arr[i, j]):
                    ax.text(j, i, f"{arr[i, j]:.3f}", ha="center", va="center", color="white", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _pivot_heatmap(
    df: pd.DataFrame,
    channel: str,
    kl_col: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    sub = df[df["channel"] == channel].copy()
    if sub.empty:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64), np.zeros((0, 0), dtype=np.float64)
    sfra_vals = np.sort(sub["sfra"].unique())
    mu0_vals = np.sort(sub["mu0"].unique())
    z = np.full((len(sfra_vals), len(mu0_vals)), np.nan, dtype=np.float64)
    sfra_to_i = {float(v): i for i, v in enumerate(sfra_vals)}
    mu0_to_j = {float(v): j for j, v in enumerate(mu0_vals)}
    for _, row in sub.iterrows():
        i = sfra_to_i[float(row["sfra"])]
        j = mu0_to_j[float(row["mu0"])]
        z[i, j] = float(row[kl_col])
    return sfra_vals, mu0_vals, z


def _plot_figure_9panel(
    *,
    kl_df: pd.DataFrame,
    model_name: str,
    split_name: str,
    out_path: Path,
    cmap: str,
) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(10, 7.5), constrained_layout=False)
    fig.set_dpi(80)
    kl_cols = ["kl_mchirp", "kl_q", "kl_z"]

    finite_vals = []
    for kl_col in kl_cols:
        v = kl_df[kl_col].values.astype(float)
        v = v[np.isfinite(v)]
        if len(v):
            finite_vals.append(v)
    if finite_vals:
        pooled = np.concatenate(finite_vals)
        vmin, vmax = float(np.min(pooled)), float(np.max(pooled))
    else:
        vmin, vmax = 0.0, 1.0
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

    im_ref = None
    for r, (obs, kl_col) in enumerate(zip(OBS_COLS, kl_cols)):
        for c, channel in enumerate(CHANNELS):
            ax = axes[r, c]
            sfra, mu0, z = _pivot_heatmap(kl_df, channel, kl_col)
            if len(sfra) == 0 or len(mu0) == 0:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            else:
                z_plot = np.ma.masked_invalid(z)
                im = pcolormesh_sfra_mu0(
                    ax,
                    mu0,
                    sfra,
                    z_plot,
                    mu0_range=MU0_RANGE,
                    sfra_range=SFRA_RANGE,
                    cmap=cmap,
                    norm=norm,
                )
                im_ref = im
            if r == 0:
                ax.set_title(channel)
            if c == 0:
                ax.set_ylabel(f"{obs}\n" + r"$a_{\mathrm{SF}}$")
            else:
                ax.set_ylabel(r"$a_{\mathrm{SF}}$")
            ax.set_xlabel(r"$\mu_0$")

    fig.suptitle(
        f"KL heatmaps — {model_name} vs {split_name} truth "
        "(rows: mchirp/q/z, cols: SMT/CE/CHE)"
    )
    if im_ref is not None:
        cbar = fig.colorbar(im_ref, ax=axes.ravel().tolist(), shrink=0.88, pad=0.02)
        cbar.set_label("KL(True || Model)")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=80)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="KL heatmaps over (sfr_a,mu0) for NB/CFM/Diffusion vs train/test."
    )
    p.add_argument("--events-per-grid", type=int, default=512, help="True/model samples per grid point.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--max-grids-per-split", type=int, default=0, help="0 means use all split grid points.")
    p.add_argument("--colormap", type=str, default="viridis")
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--no-timestamp-subdir", action="store_true")
    p.add_argument(
        "--workers",
        type=int,
        default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")),
        help="CPU worker threads for BLAS/OpenMP (default: SLURM_CPUS_PER_TASK or 1).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    workers = configure_worker_threads(args.workers)
    rng = np.random.default_rng(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    data_dir = ml_data_dir()
    hp_csv = data_dir / HYPERPARAM_TABLE_ENCODED_CSV.name
    events_pq = data_dir / ALL_EVENTS_PARQUET.name
    splits_path = data_dir / SPLITS_JSON.name
    for p in [hp_csv, events_pq, splits_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing {p}. Run 02_build_dataset.py first.")

    hp_df = _ensure_sspc_grid_axes(pd.read_csv(hp_csv))
    with open(splits_path) as f:
        splits = json.load(f)

    device = torch.device(args.device)
    models = _load_models(device)
    print(f"Using worker threads: {workers}")

    if args.out_dir is not None:
        out_dir = args.out_dir.resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = resolve_plot_output(
            Path(__file__),
            no_timestamp_subdir=args.no_timestamp_subdir,
        )

    all_tables = []
    pooled_rows: List[Dict[str, object]] = []
    for split_name in SPLITS:
        split_idx = list(map(int, splits[split_name]))
        if args.max_grids_per_split > 0 and len(split_idx) > args.max_grids_per_split:
            split_idx = rng.choice(split_idx, size=args.max_grids_per_split, replace=False).tolist()

        print(f"Loading truth samples for split={split_name}, grids={len(split_idx)}")
        truth_cache = _collect_truth_samples_for_split(
            events_parquet=events_pq,
            split_indices=split_idx,
            events_per_grid=int(args.events_per_grid),
            rng=rng,
        )

        for model_name in MODELS:
            print(f"Computing KL table: model={model_name}, split={split_name}, grids={len(split_idx)}")
            kl_df = _compute_kl_table(
                hp_df=hp_df,
                split_indices=split_idx,
                truth_cache=truth_cache,
                model_name=model_name,
                model_bundle=models[model_name],
            )
            if kl_df.empty:
                raise RuntimeError(f"No KL rows generated for model={model_name}, split={split_name}")

            kl_df["model"] = model_name
            kl_df["split"] = split_name
            all_tables.append(kl_df)

            png_name = f"kl_heatmap_{model_name}_vs_{split_name}.png"
            _plot_figure_9panel(
                kl_df=kl_df,
                model_name=model_name,
                split_name=split_name,
                out_path=out_dir / png_name,
                cmap=args.colormap,
            )
            print(f"  Saved {png_name}")

            # Pooled KL per channel (mixture over grid points in the split)
            for ch in CHANNELS:
                pooled = _compute_pooled_kl(
                    hp_df=hp_df,
                    split_indices=split_idx,
                    truth_cache=truth_cache,
                    model_name=model_name,
                    model_bundle=models[model_name],
                    channel=ch,
                )
                pooled_rows.append(
                    {
                        "split": split_name,
                        "model": model_name,
                        "channel": ch,
                        "kl_mchirp_pooled": float(pooled["kl_mchirp"]),
                        "kl_q_pooled": float(pooled["kl_q"]),
                        "kl_z_pooled": float(pooled["kl_z"]),
                    }
                )

    metrics = pd.concat(all_tables, ignore_index=True)
    metrics_csv = out_dir / "kl_grid_metrics.csv"
    metrics.to_csv(metrics_csv, index=False)

    summary_df = _summarize_kl_tables(metrics, pooled_rows)
    summary_csv = out_dir / "kl_summary_pooled_vs_cell.csv"
    summary_df.to_csv(summary_csv, index=False)
    _plot_summary_bars(summary_df, out_dir / "kl_summary_pooled_vs_cell.png")
    meta = {
        "splits": list(SPLITS),
        "models": list(MODELS),
        "channels": list(CHANNELS),
        "observables": list(OBS_COLS),
        "events_per_grid": int(args.events_per_grid),
        "max_grids_per_split": int(args.max_grids_per_split),
        "seed": int(args.seed),
        "device": args.device,
        "outputs": [f"kl_heatmap_{m}_vs_{s}.png" for s in SPLITS for m in MODELS],
        "metrics_csv": str(metrics_csv.name),
        "summary_csv": str(summary_csv.name),
        "summary_png": "kl_summary_pooled_vs_cell.png",
    }
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Done. Outputs in {out_dir}")


if __name__ == "__main__":
    main()
