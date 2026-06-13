#!/usr/bin/env python3
"""
Step 04d — Compare NB / CFM / diffusion on mchirp, q, z across train/val/test.

Plot sets (--plot-set):
  full (default) — density grid, collated density, ECDF, metric heatmaps, test Q-Q
  nb-kernel-bandwidth — collated density lines only: truth + NB at several Λ-kernel τ
  nb-ablation — one run, multiple panels: τ, σ-scale, σ-floor (refit), mode (refit)
  collated-density — collated density panel only (subset of full, faster iteration)

It writes figures and a metrics CSV under:
  plots/04d_split_observable_model_compare/<timestamp>/[<run-tag>/]
"""
import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

_ANALYSIS_DIR = Path(__file__).resolve().parent
if str(_ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(_ANALYSIS_DIR))

from _bootstrap import setup  # noqa: E402

setup()

import numpy as np
import pandas as pd

from plant_paths import (  # noqa: E402
    ALL_EVENTS_PARQUET,
    CHECKPOINT_DIR,
    HYPERPARAM_TABLE_ENCODED_CSV,
    SPLITS_JSON,
    plot_run_dir,
    ml_data_dir,
)

# Deferred until needed (matplotlib/scipy/torch models are slow on login nodes).
_plt = None
_gaussian_kde = None
_torch = None


OBS_COLS = ["mchirp", "q", "z"]
SPLIT_ORDER = ["train", "val", "test"]
MODEL_ORDER = ["naive_bayes", "cfm", "diffusion"]
MODEL_LABELS = {
    "naive_bayes": "Naive Bayes",
    "cfm": "CFM",
    "diffusion": "Diffusion",
}
MODEL_COLORS = {
    "truth": "black",
    "naive_bayes": "C0",
    "cfm": "C1",
    "diffusion": "C3",
}

SPLIT_LINESTYLE = {
    "train": "-",
    "val": "--",
    "test": ":",
}


def _linestyle_for_split(
    split: str,
    splits: Sequence[str],
    *,
    encode_split_linestyles: bool = False,
) -> str:
    """Solid lines by default; optional train/val/test linestyle encoding."""
    if not encode_split_linestyles or len(splits) <= 1:
        return "-"
    return SPLIT_LINESTYLE[split]

# mchirp density panels use log-y by default (wide dynamic range in KDE height).
MCHIRP_LOG_Y_FLOOR = 1e-12

PLOT_SETS = ("full", "nb-kernel-bandwidth", "nb-ablation", "nb-mode-collated", "collated-density")
NB_ABLATION_AXES = ("tau", "sigma-scale", "sigma-floor", "mode")

# Default sweeps for --plot-set nb-ablation (override per axis via CLI).
DEFAULT_NB_ABLATION_TAUS = "0.05,0.1,0.15,0.2,0.3,0.5,0.75,1.0"
DEFAULT_NB_SIGMA_SCALES = "0.5,0.75,1.0,1.25,1.5,2.0"
DEFAULT_NB_SIGMA_FLOORS = "0.02,0.05,0.1,0.2"
DENSITY_STYLES = ("filled", "lines")

# Figures produced per plot set (basename without extension)
PLOT_SET_OUTPUTS = {
    "full": [
        "01_density_overlays.png",
        "01b_density_overlays_collated.png",
        "02_ecdf_overlays.png",
        "03_ks_heatmaps.png",
        "03_kl_heatmaps.png",
        "03_wasserstein_heatmaps.png",
        "04_test_qq.png",
        "metrics_by_split_model_observable.csv",
        "plot_guide.md",
    ],
    "nb-kernel-bandwidth": [
        "01b_density_overlays_collated_nb_kernel_bandwidth.png",
        "nb_kernel_bandwidth_ablation.json",
    ],
    "nb-ablation": [
        "nb_ablation_tau.png",
        "nb_ablation_sigma_scale.png",
        "nb_ablation_sigma_floor.png",
        "nb_ablation_mode.png",
        "nb_ablation.json",
    ],
    "nb-mode-collated": [
        "01b_density_overlays_collated_nb_mode.png",
    ],
    "collated-density": [
        "01b_density_overlays_collated.png",
    ],
}


def _density_eval_grid(obs: str, values: np.ndarray) -> np.ndarray:
    """Linear KDE evaluation grid in observable units."""
    del obs  # same grid rule for all observables
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        vals = np.array([1.0])
    lo, hi = np.percentile(vals, [0.5, 99.5])
    return np.linspace(float(lo), float(hi), 320)


def _density_fill_base(obs: str, y_kde: np.ndarray) -> float:
    """Baseline for fill_between (log-y on mchirp needs a positive floor)."""
    if obs != "mchirp":
        return 0.0
    pos = y_kde[np.asarray(y_kde) > 0]
    if pos.size == 0:
        return MCHIRP_LOG_Y_FLOOR
    return max(float(np.min(pos)) * 0.1, MCHIRP_LOG_Y_FLOOR)


def _finalize_density_axis(ax, obs: str) -> None:
    ax.set_title(obs, fontsize=12)
    ax.set_xlabel(obs)
    if obs == "mchirp":
        ax.set_yscale("log")
        ax.set_ylabel("density (log scale)")
    else:
        ax.set_ylabel("density")
    ax.grid(True, alpha=0.25, linewidth=0.8, which="both")


def _parse_csv_floats(text: str) -> List[float]:
    values: List[float] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        values.append(float(part))
    if not values:
        raise ValueError("Expected at least one numeric value.")
    return values


def _parse_csv_tokens(text: str) -> List[str]:
    return [p.strip() for p in text.split(",") if p.strip()]


def _nb_series_key_tau(tau: float) -> str:
    return f"nb_tau_{tau:g}"


def _nb_series_key_sigma_scale(scale: float) -> str:
    return f"nb_sigscale_{scale:g}"


def _nb_series_key_sigma_floor(floor: float) -> str:
    return f"nb_sigfloor_{floor:g}"


def _nb_series_key_mode(mode: str) -> str:
    return f"nb_mode_{mode}"


def _resolve_nb_kernel_bandwidths(
    requested: Optional[str],
    checkpoint_tau: float,
) -> List[float]:
    """Merge CLI τ list with checkpoint default; dedupe and sort."""
    if requested is None:
        return [float(checkpoint_tau)]
    values = _parse_csv_floats(requested)
    merged = sorted({float(checkpoint_tau), *values})
    return merged


# High-contrast palette for NB τ curves (avoid black/near-black; truth uses black).
NB_TAU_COLORS = [
    "#d62728",  # red
    "#1f77b4",  # blue
    "#2ca02c",  # green
    "#ff7f0e",  # orange
    "#9467bd",  # purple
    "#17becf",  # cyan
    "#e377c2",  # pink
    "#bcbd22",  # olive
    "#8c564b",  # brown
    "#7f7f7f",  # gray (last resort)
]


def _variant_color_map(keys: Sequence[str]) -> Dict[str, str]:
    return {key: NB_TAU_COLORS[i % len(NB_TAU_COLORS)] for i, key in enumerate(keys)}


def _log(msg: str) -> None:
    print(msg, flush=True)


def _ensure_torch():
    global _torch
    if _torch is None:
        _log("Importing torch...")
        import torch as _torch_mod

        _torch = _torch_mod
    return _torch


def _ensure_plotting():
    global _plt, _gaussian_kde
    if _plt is None:
        _log("Importing matplotlib/scipy (plotting)...")
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt_mod
        from scipy.stats import gaussian_kde as gkde_mod

        _plt = plt_mod
        _gaussian_kde = gkde_mod
    return _plt, _gaussian_kde


def configure_worker_threads(workers: int) -> int:
    """Set threaded-kernel worker counts for CPU-heavy sampling and plotting."""
    w = max(1, int(workers))
    os.environ["OMP_NUM_THREADS"] = str(w)
    os.environ["MKL_NUM_THREADS"] = str(w)
    os.environ["OPENBLAS_NUM_THREADS"] = str(w)
    os.environ["NUMEXPR_NUM_THREADS"] = str(w)
    _ensure_torch().set_num_threads(w)
    return w


@dataclass
class NBGenVariant:
    """One NB curve in an ablation panel (key used in split_catalogs dict)."""

    key: str
    label: str
    bundle: "ModelBundle"
    kernel_bandwidth: Optional[float] = None
    sigma_scale: Optional[float] = None


@dataclass
class ModelBundle:
    name: str
    model: object
    normalizer: Dict
    lambda_cols: List[str]

    def generate(
        self,
        lambda_vec: np.ndarray,
        n_events: int,
        *,
        nb_kernel_bandwidth: Optional[float] = None,
        nb_sigma_scale: Optional[float] = None,
    ) -> pd.DataFrame:
        if self.name == "naive_bayes":
            from models.naive_bayes_emulator import generate_catalog as generate_nb

            orig_tau = None
            orig_sigma = None
            try:
                if nb_kernel_bandwidth is not None:
                    orig_tau = float(self.model.kernel_bandwidth.item())
                    self.model.kernel_bandwidth.fill_(float(nb_kernel_bandwidth))
                if nb_sigma_scale is not None:
                    orig_sigma = self.model.grid_sigma.clone()
                    self.model.grid_sigma.mul_(float(nb_sigma_scale))
                return generate_nb(lambda_vec, n_events, self.model, self.normalizer)
            finally:
                if orig_tau is not None:
                    self.model.kernel_bandwidth.fill_(orig_tau)
                if orig_sigma is not None:
                    self.model.grid_sigma.copy_(orig_sigma)
        if self.name == "cfm":
            from models.cfm_emulator import generate_catalog as generate_cfm

            return generate_cfm(lambda_vec, n_events, self.model, self.normalizer)
        if self.name == "diffusion":
            from models.diffusion_emulator import generate_catalog as generate_diff

            return generate_diff(lambda_vec, n_events, self.model, self.normalizer)
        raise ValueError(f"Unknown model name: {self.name}")


def _lambda_cols_from_df(df: pd.DataFrame) -> List[str]:
    return sorted(
        [c for c in df.columns if c.startswith("lambda_")],
        key=lambda x: int(x.split("_")[1]),
    )


def _load_nb_bundle(nb_checkpoint: Path) -> Tuple[ModelBundle, float]:
    from models.naive_bayes_emulator import load_from_checkpoint

    torch = _ensure_torch()
    nb_ckpt = torch.load(nb_checkpoint, map_location="cpu", weights_only=False)
    nb_model, nb_lambda_cols, nb_normalizer = load_from_checkpoint(
        nb_ckpt, device=torch.device("cpu")
    )
    checkpoint_tau = float(nb_ckpt.get("kernel_bandwidth", nb_model.kernel_bandwidth.item()))
    bundle = ModelBundle(
        name="naive_bayes",
        model=nb_model,
        normalizer=nb_normalizer,
        lambda_cols=nb_lambda_cols,
    )
    return bundle, checkpoint_tau


def _load_models_for_run(
    device: object,
    model_names: Sequence[str],
    *,
    nb_checkpoint: Path,
    cfm_checkpoint: Path,
    diffusion_checkpoint: Path,
) -> Tuple[Dict[str, ModelBundle], float]:
    bundles: Dict[str, ModelBundle] = {}
    checkpoint_tau = 1.0

    if "naive_bayes" in model_names:
        bundles["naive_bayes"], checkpoint_tau = _load_nb_bundle(nb_checkpoint)

    if "cfm" in model_names:
        from models.cfm_emulator import CFMEmulator

        torch = _ensure_torch()
        cfm_ckpt = torch.load(cfm_checkpoint, map_location=device, weights_only=False)
        cfm_lambda_cols = cfm_ckpt["lambda_cols"]
        cfm_model = CFMEmulator(
            lambda_dim=len(cfm_lambda_cols),
            context_dim=int(cfm_ckpt.get("context_dim", 128)),
            hidden_dim=int(cfm_ckpt.get("hidden_dim", 256)),
        )
        cfm_model.load_state_dict(cfm_ckpt["model_state"], strict=True)
        cfm_model.to(device)
        cfm_model.eval()
        bundles["cfm"] = ModelBundle(
            name="cfm",
            model=cfm_model,
            normalizer=cfm_ckpt["normalizer"],
            lambda_cols=cfm_lambda_cols,
        )

    if "diffusion" in model_names:
        from models.diffusion_emulator import DiffusionEmulator

        torch = _ensure_torch()
        diff_ckpt = torch.load(diffusion_checkpoint, map_location=device, weights_only=False)
        diff_lambda_cols = diff_ckpt["lambda_cols"]
        diff_model = DiffusionEmulator(
            lambda_dim=len(diff_lambda_cols),
            context_dim=int(diff_ckpt.get("context_dim", 128)),
            hidden_dim=int(diff_ckpt.get("hidden_dim", 256)),
            n_timesteps=int(diff_ckpt.get("n_timesteps", 100)),
        )
        diff_model.load_state_dict(diff_ckpt["model_state"], strict=True)
        diff_model.to(device)
        diff_model.eval()
        bundles["diffusion"] = ModelBundle(
            name="diffusion",
            model=diff_model,
            normalizer=diff_ckpt["normalizer"],
            lambda_cols=diff_lambda_cols,
        )

    return bundles, checkpoint_tau


def _select_grid_indices(
    split_indices: List[int],
    max_grids: int,
    rng: np.random.Generator,
) -> np.ndarray:
    idx = np.asarray(split_indices, dtype=int)
    if len(idx) <= max_grids:
        return idx
    return rng.choice(idx, size=max_grids, replace=False)


def _load_events_parquet(path: Path, grid_indices: Sequence[int]) -> pd.DataFrame:
    """Load only rows for the selected grid points (one parquet scan)."""
    import pyarrow.parquet as pq

    cols = ["mchirp", "q", "z", "grid_idx"]
    grids = sorted({int(g) for g in grid_indices})
    if not grids:
        return pd.DataFrame(columns=cols)
    table = pq.read_table(path, columns=cols, filters=[("grid_idx", "in", grids)])
    return table.to_pandas()


def _build_events_by_grid(events_df: pd.DataFrame) -> Dict[int, np.ndarray]:
    """Group events once per grid (avoids O(n_events) scan per grid in the sampling loop)."""
    out: Dict[int, np.ndarray] = {}
    for grid_idx, grp in events_df.groupby("grid_idx", sort=False):
        out[int(grid_idx)] = grp[list(OBS_COLS)].values.astype(np.float32, copy=False)
    return out


def _sample_events_from_pool(
    pool: np.ndarray,
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if pool.size == 0:
        return np.zeros((n, 3), dtype=np.float32)
    if pool.ndim == 1:
        pool = pool.reshape(-1, 3)
    n_pool = pool.shape[0]
    idx = rng.integers(0, n_pool, size=min(n, n_pool))
    if idx.size < n:
        idx = rng.choice(n_pool, size=n, replace=True)
    return pool[idx]


def _collect_split_catalogs(
    split_name: str,
    split_indices: np.ndarray,
    hp_df: pd.DataFrame,
    events_by_grid: Dict[int, np.ndarray],
    models: Dict[str, ModelBundle],
    n_events_per_grid: int,
    rng: np.random.Generator,
    *,
    model_names: Sequence[str],
    nb_kernel_bandwidths: Optional[Sequence[float]] = None,
) -> Dict[str, pd.DataFrame]:
    truth_chunks: List[pd.DataFrame] = []
    model_chunks: Dict[str, List[pd.DataFrame]] = {}

    nb_taus = list(nb_kernel_bandwidths) if nb_kernel_bandwidths else [None]
    for model_name in model_names:
        if model_name == "naive_bayes" and nb_kernel_bandwidths:
            for tau in nb_taus:
                model_chunks[_nb_series_key_tau(float(tau))] = []
        else:
            model_chunks[model_name] = []

    lambda_cols_ref = _lambda_cols_from_df(hp_df)
    for grid_idx in split_indices:
        pool = events_by_grid.get(int(grid_idx), np.zeros((0, 3), dtype=np.float32))
        true_samples = _sample_events_from_pool(pool, n_events_per_grid, rng)
        truth_chunks.append(pd.DataFrame(true_samples, columns=OBS_COLS))

        lambda_vec_ref = hp_df.iloc[int(grid_idx)][lambda_cols_ref].values.astype(np.float32)
        for model_name in model_names:
            bundle = models[model_name]
            if bundle.lambda_cols == lambda_cols_ref:
                lambda_vec = lambda_vec_ref
            else:
                lambda_vec = hp_df.iloc[int(grid_idx)][bundle.lambda_cols].values.astype(np.float32)

            if model_name == "naive_bayes" and nb_kernel_bandwidths:
                for tau in nb_taus:
                    gen_df = bundle.generate(
                        lambda_vec,
                        n_events_per_grid,
                        nb_kernel_bandwidth=float(tau),
                    )[OBS_COLS]
                    model_chunks[_nb_series_key_tau(float(tau))].append(gen_df.reset_index(drop=True))
            else:
                gen_df = bundle.generate(lambda_vec, n_events_per_grid)[OBS_COLS]
                model_chunks[model_name].append(gen_df.reset_index(drop=True))

    out = {"truth": pd.concat(truth_chunks, ignore_index=True)}
    for key, chunks in model_chunks.items():
        out[key] = pd.concat(chunks, ignore_index=True)

    n_total = len(out["truth"])
    extra = ""
    if nb_kernel_bandwidths:
        extra = f", nb_τ={len(nb_kernel_bandwidths)}"
    _log(f"  {split_name:>5}: grids={len(split_indices):4d}, pooled samples={n_total:7d}{extra}")
    return out


def _collect_split_catalogs_nb_variants(
    split_name: str,
    split_indices: np.ndarray,
    hp_df: pd.DataFrame,
    events_by_grid: Dict[int, np.ndarray],
    variants: Sequence[NBGenVariant],
    n_events_per_grid: int,
    rng: np.random.Generator,
) -> Dict[str, pd.DataFrame]:
    truth_chunks: List[pd.DataFrame] = []
    model_chunks: Dict[str, List[pd.DataFrame]] = {v.key: [] for v in variants}

    lambda_cols_ref = _lambda_cols_from_df(hp_df)
    for grid_idx in split_indices:
        pool = events_by_grid.get(int(grid_idx), np.zeros((0, 3), dtype=np.float32))
        true_samples = _sample_events_from_pool(pool, n_events_per_grid, rng)
        truth_chunks.append(pd.DataFrame(true_samples, columns=OBS_COLS))

        lambda_vec_ref = hp_df.iloc[int(grid_idx)][lambda_cols_ref].values.astype(np.float32)
        for variant in variants:
            bundle = variant.bundle
            if bundle.lambda_cols == lambda_cols_ref:
                lambda_vec = lambda_vec_ref
            else:
                lambda_vec = hp_df.iloc[int(grid_idx)][bundle.lambda_cols].values.astype(np.float32)
            gen_df = bundle.generate(
                lambda_vec,
                n_events_per_grid,
                nb_kernel_bandwidth=variant.kernel_bandwidth,
                nb_sigma_scale=variant.sigma_scale,
            )[OBS_COLS]
            model_chunks[variant.key].append(gen_df.reset_index(drop=True))

    out = {"truth": pd.concat(truth_chunks, ignore_index=True)}
    for key, chunks in model_chunks.items():
        out[key] = pd.concat(chunks, ignore_index=True)

    n_total = len(out["truth"])
    _log(
        f"  {split_name:>5}: grids={len(split_indices):4d}, pooled samples={n_total:7d}, "
        f"variants={len(variants)}"
    )
    return out


def _plot_density_grid(split_catalogs: Dict[str, Dict[str, pd.DataFrame]], out_path: Path) -> None:
    plt, gaussian_kde = _ensure_plotting()
    fig, axes = plt.subplots(len(OBS_COLS), len(SPLIT_ORDER), figsize=(15, 10))
    for i, obs in enumerate(OBS_COLS):
        for j, split in enumerate(SPLIT_ORDER):
            ax = axes[i, j]
            split_data = split_catalogs[split]

            x_true = split_data["truth"][obs].values
            x = _density_eval_grid(obs, x_true)

            try:
                kde_true = gaussian_kde(x_true)
                y_true = kde_true(x)
                y0 = _density_fill_base(obs, y_true)
                ax.fill_between(x, y0, y_true, color=MODEL_COLORS["truth"], alpha=0.08, linewidth=0)
                ax.plot(x, y_true, color=MODEL_COLORS["truth"], lw=2.2, label="Truth")
            except Exception:
                ax.hist(
                    x_true,
                    bins=40,
                    density=True,
                    alpha=0.25,
                    color=MODEL_COLORS["truth"],
                    label="Truth",
                )

            for model_name in MODEL_ORDER:
                x_model = split_data[model_name][obs].values
                try:
                    kde_model = gaussian_kde(x_model)
                    y_model = kde_model(x)
                    y0 = _density_fill_base(obs, y_model)
                    ax.fill_between(
                        x,
                        y0,
                        y_model,
                        color=MODEL_COLORS[model_name],
                        alpha=0.08,
                        linewidth=0,
                    )
                    ax.plot(
                        x,
                        y_model,
                        color=MODEL_COLORS[model_name],
                        lw=2.0,
                        alpha=0.9,
                        label=MODEL_LABELS[model_name],
                    )
                except Exception:
                    ax.hist(
                        x_model,
                        bins=40,
                        density=True,
                        alpha=0.2,
                        color=MODEL_COLORS[model_name],
                        label=MODEL_LABELS[model_name],
                    )

            if i == 0:
                ax.set_title(split.upper())
            if j == 0 and obs != "mchirp":
                ax.set_ylabel(f"{obs} density")
            _finalize_density_axis(ax, obs)
            if i == 0 and j == 0:
                ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_density_collated(
    split_catalogs: Dict[str, Dict[str, pd.DataFrame]],
    out_path: Path,
    *,
    splits: Sequence[str],
    series: Sequence[Tuple[str, str, str]],
    density_style: str = "lines",
    title: Optional[str] = None,
    encode_split_linestyles: bool = False,
) -> None:
    """
    Collated density overlays (one row × len(OBS_COLS)).

    *series* entries are (catalog_key, legend_label, color).
    Series by color; splits use solid lines unless encode_split_linestyles.
    """
    plt, gaussian_kde = _ensure_plotting()
    fill = density_style == "filled"
    fig, axes = plt.subplots(1, len(OBS_COLS), figsize=(16, 4.6), constrained_layout=True)
    if len(OBS_COLS) == 1:
        axes = [axes]

    series_handles = [
        plt.Line2D([0], [0], color=color, lw=2.4, label=label) for _, label, color in series
    ]
    split_handles = [
        plt.Line2D(
            [0],
            [0],
            color="gray",
            lw=2.4,
            linestyle=_linestyle_for_split(s, splits, encode_split_linestyles=encode_split_linestyles),
            label=s.upper(),
        )
        for s in splits
    ] if len(splits) > 1 and encode_split_linestyles else []

    for ax, obs in zip(axes, OBS_COLS):
        x_all_true = np.concatenate(
            [split_catalogs[s]["truth"][obs].values for s in splits], axis=0
        )
        x = _density_eval_grid(obs, x_all_true)

        for split in splits:
            ls = _linestyle_for_split(
                split, splits, encode_split_linestyles=encode_split_linestyles
            )
            split_data = split_catalogs[split]
            for key, _label, color in series:
                y = split_data[key][obs].values
                lw = 2.2 if key == "truth" else 2.0
                if density_style == "lines":
                    lw = 2.4 if key == "truth" else 2.2
                try:
                    kde = gaussian_kde(y)
                    y_kde = kde(x)
                    if fill:
                        y0 = _density_fill_base(obs, y_kde)
                        ax.fill_between(x, y0, y_kde, color=color, alpha=0.06, linewidth=0)
                    ax.plot(x, y_kde, color=color, lw=lw, linestyle=ls, alpha=0.95)
                except Exception:
                    ax.hist(
                        y,
                        bins=40,
                        density=True,
                        alpha=0.12 if fill else 0.0,
                        histtype="step" if not fill else "bar",
                        color=color,
                    )

        _finalize_density_axis(ax, obs)

    if title:
        fig.suptitle(title, fontsize=11, y=1.02)
    axes[0].legend(handles=series_handles, fontsize=8, frameon=False, loc="upper right")
    if split_handles:
        axes[-1].legend(handles=split_handles, fontsize=8, frameon=False, loc="upper right")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_density_collated_by_observable(
    split_catalogs: Dict[str, Dict[str, pd.DataFrame]],
    out_path: Path,
    *,
    splits: Sequence[str] = SPLIT_ORDER,
    density_style: str = "lines",
    encode_split_linestyles: bool = False,
) -> None:
    series = [("truth", "Truth", MODEL_COLORS["truth"])]
    series.extend((m, MODEL_LABELS[m], MODEL_COLORS[m]) for m in MODEL_ORDER)
    _plot_density_collated(
        split_catalogs,
        out_path,
        splits=splits,
        series=series,
        density_style=density_style,
        encode_split_linestyles=encode_split_linestyles,
    )


def _plot_density_collated_nb_comparison(
    split_catalogs: Dict[str, Dict[str, pd.DataFrame]],
    out_path: Path,
    *,
    splits: Sequence[str],
    variant_keys: Sequence[str],
    variant_labels: Sequence[str],
    suptitle: str,
    legend_title: str = "NB variants",
    encode_split_linestyles: bool = False,
) -> None:
    """Truth = thick black; each variant = solid colored KDE line."""
    plt, gaussian_kde = _ensure_plotting()
    colors = _variant_color_map(variant_keys)

    n_var = len(variant_keys)
    fig_w = max(16.0, 12.0 + 0.35 * n_var)
    fig, axes = plt.subplots(1, len(OBS_COLS), figsize=(fig_w, 5.2), constrained_layout=True)
    if len(OBS_COLS) == 1:
        axes = [axes]

    if len(splits) > 1 and encode_split_linestyles:
        truth_handles = [
            plt.Line2D(
                [0],
                [0],
                color="black",
                lw=3.2,
                linestyle=_linestyle_for_split(
                    s, splits, encode_split_linestyles=encode_split_linestyles
                ),
                label=f"Truth ({s})",
            )
            for s in splits
        ]
    else:
        truth_handles = [
            plt.Line2D([0], [0], color="black", lw=3.2, linestyle="-", label="Truth")
        ]
    variant_handles = [
        plt.Line2D([0], [0], color=colors[key], lw=2.4, linestyle="-", label=label)
        for key, label in zip(variant_keys, variant_labels)
    ]

    for ax, obs in zip(axes, OBS_COLS):
        x_all_true = np.concatenate(
            [split_catalogs[s]["truth"][obs].values for s in splits], axis=0
        )
        x = _density_eval_grid(obs, x_all_true)

        for split in splits:
            ls = _linestyle_for_split(
                split, splits, encode_split_linestyles=encode_split_linestyles
            )
            split_data = split_catalogs[split]
            y_true = split_data["truth"][obs].values
            try:
                kde_true = gaussian_kde(y_true)
                y_kde = kde_true(x)
                ax.plot(
                    x,
                    y_kde,
                    color="black",
                    lw=3.2,
                    linestyle=ls,
                    alpha=1.0,
                    zorder=20,
                    solid_capstyle="round",
                )
            except Exception:
                ax.hist(
                    y_true,
                    bins=40,
                    density=True,
                    histtype="step",
                    color="black",
                    lw=2.5,
                    linestyle=ls,
                )

            for key in variant_keys:
                y_nb = split_data[key][obs].values
                try:
                    kde_nb = gaussian_kde(y_nb)
                    y_kde = kde_nb(x)
                    ax.plot(
                        x,
                        y_kde,
                        color=colors[key],
                        lw=2.4,
                        linestyle="-",
                        alpha=0.92,
                        zorder=10,
                        solid_capstyle="round",
                    )
                except Exception:
                    ax.hist(
                        y_nb,
                        bins=40,
                        density=True,
                        histtype="step",
                        color=colors[key],
                        lw=2.0,
                        linestyle="-",
                    )

        _finalize_density_axis(ax, obs)

    fig.suptitle(suptitle, fontsize=11, y=1.03)
    axes[0].legend(
        handles=truth_handles,
        fontsize=7,
        frameon=False,
        loc="upper left",
        title="Truth",
        title_fontsize=8,
    )
    axes[-1].legend(
        handles=variant_handles,
        fontsize=7,
        frameon=False,
        loc="upper right",
        ncol=1 if n_var <= 6 else 2,
        title=legend_title,
        title_fontsize=8,
    )
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_density_collated_nb_kernel_bandwidth(
    split_catalogs: Dict[str, Dict[str, pd.DataFrame]],
    out_path: Path,
    *,
    splits: Sequence[str],
    taus: Sequence[float],
    checkpoint_tau: float,
    encode_split_linestyles: bool = False,
) -> None:
    keys = [_nb_series_key_tau(t) for t in taus]
    labels = [f"τ={t:g}" for t in taus]
    _plot_density_collated_nb_comparison(
        split_catalogs,
        out_path,
        splits=splits,
        variant_keys=keys,
        variant_labels=labels,
        suptitle=(
            f"NB Λ-kernel τ (checkpoint τ={checkpoint_tau:g}; black = truth, colors = NB)"
        ),
        legend_title="Λ-kernel τ",
        encode_split_linestyles=encode_split_linestyles,
    )


def _plot_ecdf_grid(split_catalogs: Dict[str, Dict[str, pd.DataFrame]], out_path: Path) -> None:
    plt, _ = _ensure_plotting()
    fig, axes = plt.subplots(len(OBS_COLS), len(SPLIT_ORDER), figsize=(15, 10))
    for i, obs in enumerate(OBS_COLS):
        for j, split in enumerate(SPLIT_ORDER):
            ax = axes[i, j]
            split_data = split_catalogs[split]
            for key in ["truth"] + MODEL_ORDER:
                x = np.sort(split_data[key][obs].values)
                y = np.arange(1, len(x) + 1) / len(x)
                label = "Truth" if key == "truth" else MODEL_LABELS[key]
                lw = 2.0 if key == "truth" else 1.4
                ax.plot(x, y, color=MODEL_COLORS[key], lw=lw, alpha=0.9, label=label)
            if i == 0:
                ax.set_title(split.upper())
            if j == 0:
                ax.set_ylabel(f"{obs} ECDF")
            ax.set_xlabel(obs)
            if i == 0 and j == 0:
                ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _compute_metrics(split_catalogs: Dict[str, Dict[str, pd.DataFrame]]) -> pd.DataFrame:
    from scipy.stats import ks_2samp, wasserstein_distance

    from lib.emulator_plot_utils import histogram_kl

    rows = []
    for split in SPLIT_ORDER:
        true_df = split_catalogs[split]["truth"]
        for model_name in MODEL_ORDER:
            model_df = split_catalogs[split][model_name]
            for obs in OBS_COLS:
                x_true = true_df[obs].values
                x_model = model_df[obs].values
                ks_stat = ks_2samp(x_true, x_model).statistic
                kl_val = histogram_kl(x_true, x_model)
                wd = wasserstein_distance(x_true, x_model)
                rows.append(
                    {
                        "split": split,
                        "model": model_name,
                        "observable": obs,
                        "ks": float(ks_stat),
                        "kl": float(kl_val),
                        "wasserstein": float(wd),
                    }
                )
    return pd.DataFrame(rows)


def _plot_metric_heatmaps(metrics_df: pd.DataFrame, out_dir: Path) -> None:
    plt, _ = _ensure_plotting()
    metric_specs = [
        ("ks", "KS statistic"),
        ("kl", "KL divergence"),
        ("wasserstein", "Wasserstein distance"),
    ]

    for metric_col, metric_title in metric_specs:
        fig, axes = plt.subplots(1, len(OBS_COLS), figsize=(14, 4.2))
        for i, obs in enumerate(OBS_COLS):
            ax = axes[i]
            block = metrics_df[metrics_df["observable"] == obs]
            arr = np.zeros((len(SPLIT_ORDER), len(MODEL_ORDER)), dtype=float)
            for r, split in enumerate(SPLIT_ORDER):
                for c, model in enumerate(MODEL_ORDER):
                    val = block[(block["split"] == split) & (block["model"] == model)][metric_col].iloc[0]
                    arr[r, c] = val

            im = ax.imshow(arr, aspect="auto", cmap="viridis")
            ax.set_xticks(np.arange(len(MODEL_ORDER)))
            ax.set_xticklabels([MODEL_LABELS[m] for m in MODEL_ORDER], rotation=30, ha="right")
            ax.set_yticks(np.arange(len(SPLIT_ORDER)))
            ax.set_yticklabels([s.upper() for s in SPLIT_ORDER])
            ax.set_title(obs)
            for r in range(arr.shape[0]):
                for c in range(arr.shape[1]):
                    ax.text(c, r, f"{arr[r, c]:.3f}", ha="center", va="center", color="white", fontsize=8)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        fig.suptitle(f"{metric_title} by split and model")
        fig.tight_layout()
        fig.savefig(out_dir / f"03_{metric_col}_heatmaps.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


def _plot_test_qq(split_catalogs: Dict[str, Dict[str, pd.DataFrame]], out_path: Path) -> None:
    plt, _ = _ensure_plotting()
    test_data = split_catalogs["test"]
    fig, axes = plt.subplots(len(OBS_COLS), len(MODEL_ORDER), figsize=(14, 10))
    for i, obs in enumerate(OBS_COLS):
        x_true = np.sort(test_data["truth"][obs].values)
        n_true = len(x_true)
        for j, model_name in enumerate(MODEL_ORDER):
            ax = axes[i, j]
            x_model = np.sort(test_data[model_name][obs].values)
            n = min(n_true, len(x_model))
            q = np.linspace(0.0, 1.0, n, endpoint=False)
            qt = np.quantile(x_true, q)
            qm = np.quantile(x_model, q)
            lo = min(qt.min(), qm.min())
            hi = max(qt.max(), qm.max())
            ax.plot(qt, qm, "o", ms=2.2, alpha=0.45, color=MODEL_COLORS[model_name])
            ax.plot([lo, hi], [lo, hi], "k--", lw=1.2)
            if i == 0:
                ax.set_title(MODEL_LABELS[model_name])
            if j == 0:
                ax.set_ylabel(f"Model {obs}")
            ax.set_xlabel(f"Truth {obs}")
            ax.set_aspect("equal", adjustable="box")
    fig.suptitle("Test split quantile alignment (ideal = diagonal)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _write_plot_guide(out_path: Path, plot_set: str) -> None:
    lines = [
        "# Split-aware model comparison outputs",
        "",
        f"Plot set: `{plot_set}`",
        "",
    ]
    for name in PLOT_SET_OUTPUTS.get(plot_set, []):
        lines.append(f"- `{name}`")
    if plot_set == "full":
        lines.extend(
            [
                "",
                "Collated density (`01b_density_overlays_collated.png`): solid KDE lines by default; use `--encode-split-linestyles` for train/val/test dashes.",
                "Use `--plot-set nb-kernel-bandwidth` to sweep NB Λ-kernel τ with lines-only collated panels.",
            ]
        )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def _nb_bundle_from_fitted(model: object, normalizer: Dict, lambda_cols: List[str]) -> ModelBundle:
    return ModelBundle(
        name="naive_bayes",
        model=model,
        normalizer=normalizer,
        lambda_cols=lambda_cols,
    )


def _fit_nb_bundle(
    hp_df: pd.DataFrame,
    events_df: pd.DataFrame,
    normalizer: Dict,
    *,
    mode: str,
    bandwidth: float,
    sigma_floor: float,
) -> ModelBundle:
    from models.naive_bayes_emulator import NaiveBayesEmulator

    lambda_cols = _lambda_cols_from_df(hp_df)
    model = NaiveBayesEmulator.fit_from_data(
        hp_df,
        events_df,
        normalizer,
        mode=mode,
        bandwidth=bandwidth,
        sigma_floor=sigma_floor,
    )
    return _nb_bundle_from_fitted(model, normalizer, lambda_cols)


def _collect_catalogs_for_variants(
    *,
    split_names: Sequence[str],
    selected_by_split: Dict[str, np.ndarray],
    hp_df: pd.DataFrame,
    events_by_grid: Dict[int, np.ndarray],
    variants: Sequence[NBGenVariant],
    n_events_per_grid: int,
    rng: np.random.Generator,
    axis_label: str,
) -> Dict[str, Dict[str, pd.DataFrame]]:
    _log(f"Sampling axis '{axis_label}' ({len(variants)} variants × {len(split_names)} split(s))")
    catalogs: Dict[str, Dict[str, pd.DataFrame]] = {}
    for split_name in split_names:
        catalogs[split_name] = _collect_split_catalogs_nb_variants(
            split_name=split_name,
            split_indices=selected_by_split[split_name],
            hp_df=hp_df,
            events_by_grid=events_by_grid,
            variants=variants,
            n_events_per_grid=n_events_per_grid,
            rng=rng,
        )
    return catalogs


def _parse_ablation_axes(text: Optional[str]) -> List[str]:
    if text is None:
        return list(NB_ABLATION_AXES)
    axes = _parse_csv_tokens(text)
    bad = [a for a in axes if a not in NB_ABLATION_AXES]
    if bad:
        raise ValueError(f"Unknown --nb-ablation-axes: {bad}. Choose from {NB_ABLATION_AXES}")
    return axes


def _run_nb_mode_collated(
    plots_dir: Path,
    *,
    gaussian_bundle: ModelBundle,
    checkpoint_tau: float,
    hp_df: pd.DataFrame,
    events_df_full: pd.DataFrame,
    events_by_grid: Dict[int, np.ndarray],
    selected_by_split: Dict[str, np.ndarray],
    split_names: Sequence[str],
    rng: np.random.Generator,
    n_events_per_grid: int,
    encode_split_linestyles: bool = False,
) -> None:
    """Collated density: truth + checkpoint Gaussian NB + refit nearest NB."""
    _log("Refitting NB nearest mode for mode comparison...")
    nearest_bundle = _fit_nb_bundle(
        hp_df,
        events_df_full,
        gaussian_bundle.normalizer,
        mode="nearest",
        bandwidth=checkpoint_tau,
        sigma_floor=0.05,
    )
    variants = [
        NBGenVariant(
            key="nb_gaussian",
            label="NB Gaussian",
            bundle=gaussian_bundle,
        ),
        NBGenVariant(
            key="nb_nearest",
            label="NB Nearest",
            bundle=nearest_bundle,
        ),
    ]
    catalogs = _collect_catalogs_for_variants(
        split_names=split_names,
        selected_by_split=selected_by_split,
        hp_df=hp_df,
        events_by_grid=events_by_grid,
        variants=variants,
        n_events_per_grid=n_events_per_grid,
        rng=rng,
        axis_label="nb-mode-collated",
    )
    _log("Rendering 01b_density_overlays_collated_nb_mode.png ...")
    _plot_density_collated(
        catalogs,
        plots_dir / "01b_density_overlays_collated_nb_mode.png",
        splits=split_names,
        series=[
            ("truth", "Truth", MODEL_COLORS["truth"]),
            ("nb_gaussian", "NB Gaussian", NB_TAU_COLORS[0]),
            ("nb_nearest", "NB Nearest", NB_TAU_COLORS[1]),
        ],
        density_style="lines",
        title="Truth vs NB Gaussian (checkpoint) vs NB Nearest (refit)",
        encode_split_linestyles=encode_split_linestyles,
    )


def _run_nb_ablation_suite(
    plots_dir: Path,
    *,
    axes: Sequence[str],
    base_bundle: ModelBundle,
    checkpoint_tau: float,
    hp_df: pd.DataFrame,
    events_df_full: Optional[pd.DataFrame],
    events_by_grid: Dict[int, np.ndarray],
    selected_by_split: Dict[str, np.ndarray],
    split_names: Sequence[str],
    rng: np.random.Generator,
    tau_values: List[float],
    sigma_scales: List[float],
    sigma_floors: List[float],
    args: argparse.Namespace,
) -> None:
    meta: Dict = {
        "plot_set": "nb-ablation",
        "checkpoint_tau": float(checkpoint_tau),
        "splits": list(split_names),
        "max_grids_per_split": int(args.max_grids_per_split),
        "events_per_grid": int(args.events_per_grid),
        "axes": {},
    }

    if "tau" in axes:
        variants = [
            NBGenVariant(
                key=_nb_series_key_tau(tau),
                label=f"τ={tau:g}",
                bundle=base_bundle,
                kernel_bandwidth=tau,
            )
            for tau in tau_values
        ]
        catalogs = _collect_catalogs_for_variants(
            split_names=split_names,
            selected_by_split=selected_by_split,
            hp_df=hp_df,
            events_by_grid=events_by_grid,
            variants=variants,
            n_events_per_grid=args.events_per_grid,
            rng=rng,
            axis_label="tau",
        )
        _log("Rendering nb_ablation_tau.png ...")
        _plot_density_collated_nb_comparison(
            catalogs,
            plots_dir / "nb_ablation_tau.png",
            splits=split_names,
            variant_keys=[v.key for v in variants],
            variant_labels=[v.label for v in variants],
            suptitle=f"NB Λ-kernel τ (checkpoint τ={checkpoint_tau:g})",
            legend_title="Λ-kernel τ",
            encode_split_linestyles=args.encode_split_linestyles,
        )
        meta["axes"]["tau"] = [float(t) for t in tau_values]

    if "sigma-scale" in axes:
        variants = [
            NBGenVariant(
                key=_nb_series_key_sigma_scale(s),
                label=f"σ×{s:g}",
                bundle=base_bundle,
                sigma_scale=s,
            )
            for s in sigma_scales
        ]
        catalogs = _collect_catalogs_for_variants(
            split_names=split_names,
            selected_by_split=selected_by_split,
            hp_df=hp_df,
            events_by_grid=events_by_grid,
            variants=variants,
            n_events_per_grid=args.events_per_grid,
            rng=rng,
            axis_label="sigma-scale",
        )
        _log("Rendering nb_ablation_sigma_scale.png ...")
        _plot_density_collated_nb_comparison(
            catalogs,
            plots_dir / "nb_ablation_sigma_scale.png",
            splits=split_names,
            variant_keys=[v.key for v in variants],
            variant_labels=[v.label for v in variants],
            suptitle=f"NB per-grid σ scale (fixed τ={checkpoint_tau:g}; multiply grid_sigma)",
            legend_title="σ scale",
            encode_split_linestyles=args.encode_split_linestyles,
        )
        meta["axes"]["sigma_scale"] = [float(s) for s in sigma_scales]

    normalizer = base_bundle.normalizer
    if "sigma-floor" in axes:
        if events_df_full is None:
            raise RuntimeError("sigma-floor axis requires full events parquet load.")
        variants = []
        for floor in sigma_floors:
            _log(f"Refitting NB (gaussian) with sigma_floor={floor:g} ...")
            bundle = _fit_nb_bundle(
                hp_df,
                events_df_full,
                normalizer,
                mode="gaussian",
                bandwidth=checkpoint_tau,
                sigma_floor=float(floor),
            )
            variants.append(
                NBGenVariant(
                    key=_nb_series_key_sigma_floor(floor),
                    label=f"σ_floor={floor:g}",
                    bundle=bundle,
                )
            )
        catalogs = _collect_catalogs_for_variants(
            split_names=split_names,
            selected_by_split=selected_by_split,
            hp_df=hp_df,
            events_by_grid=events_by_grid,
            variants=variants,
            n_events_per_grid=args.events_per_grid,
            rng=rng,
            axis_label="sigma-floor",
        )
        _log("Rendering nb_ablation_sigma_floor.png ...")
        _plot_density_collated_nb_comparison(
            catalogs,
            plots_dir / "nb_ablation_sigma_floor.png",
            splits=split_names,
            variant_keys=[v.key for v in variants],
            variant_labels=[v.label for v in variants],
            suptitle=f"NB σ_floor refit (fixed τ={checkpoint_tau:g})",
            legend_title="σ_floor",
            encode_split_linestyles=args.encode_split_linestyles,
        )
        meta["axes"]["sigma_floor"] = [float(f) for f in sigma_floors]

    if "mode" in axes:
        if events_df_full is None:
            raise RuntimeError("mode axis requires full events parquet load.")
        variants = []
        for mode in ("gaussian", "nearest"):
            _log(f"Refitting NB with mode={mode} ...")
            bundle = _fit_nb_bundle(
                hp_df,
                events_df_full,
                normalizer,
                mode=mode,
                bandwidth=checkpoint_tau,
                sigma_floor=0.05,
            )
            variants.append(
                NBGenVariant(
                    key=_nb_series_key_mode(mode),
                    label=f"mode={mode}",
                    bundle=bundle,
                )
            )
        catalogs = _collect_catalogs_for_variants(
            split_names=split_names,
            selected_by_split=selected_by_split,
            hp_df=hp_df,
            events_by_grid=events_by_grid,
            variants=variants,
            n_events_per_grid=args.events_per_grid,
            rng=rng,
            axis_label="mode",
        )
        _log("Rendering nb_ablation_mode.png ...")
        _plot_density_collated_nb_comparison(
            catalogs,
            plots_dir / "nb_ablation_mode.png",
            splits=split_names,
            variant_keys=[v.key for v in variants],
            variant_labels=[v.label for v in variants],
            suptitle=f"NB mode compare (fixed τ={checkpoint_tau:g})",
            legend_title="mode",
            encode_split_linestyles=args.encode_split_linestyles,
        )
        meta["axes"]["mode"] = ["gaussian", "nearest"]

    (plots_dir / "nb_ablation.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def _write_nb_bandwidth_meta(
    out_path: Path,
    *,
    checkpoint_tau: float,
    taus: Sequence[float],
    splits: Sequence[str],
    args: argparse.Namespace,
) -> None:
    meta = {
        "plot_set": "nb-kernel-bandwidth",
        "checkpoint_kernel_bandwidth": float(checkpoint_tau),
        "taus_evaluated": [float(t) for t in taus],
        "splits": list(splits),
        "max_grids_per_split": int(args.max_grids_per_split),
        "events_per_grid": int(args.events_per_grid),
        "seed": int(args.seed),
        "nb_checkpoint": str(args.nb_checkpoint),
    }
    out_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare NB/CFM/diffusion on mchirp,q,z over train/val/test splits."
    )
    parser.add_argument(
        "--plot-set",
        choices=PLOT_SETS,
        default="full",
        help="Which figure bundle to produce (default: full).",
    )
    parser.add_argument(
        "--run-tag",
        type=str,
        default="",
        help="Optional subdirectory under the timestamped run folder (e.g. nb_tau_sweep).",
    )
    parser.add_argument(
        "--nb-kernel-bandwidths",
        type=str,
        default=None,
        help=(
            "Comma-separated NB Λ-kernel τ values for nb-kernel-bandwidth / nb-ablation (tau axis). "
            "Checkpoint τ is always included."
        ),
    )
    parser.add_argument(
        "--nb-ablation-axes",
        type=str,
        default=None,
        help=(
            "For --plot-set nb-ablation: comma-separated axes among "
            "tau,sigma-scale,sigma-floor,mode (default: all four)."
        ),
    )
    parser.add_argument(
        "--nb-sigma-scales",
        type=str,
        default=None,
        help=f"σ scale factors (multiply grid_sigma) for nb-ablation (default: {DEFAULT_NB_SIGMA_SCALES}).",
    )
    parser.add_argument(
        "--nb-sigma-floors",
        type=str,
        default=None,
        help=f"σ_floor refit values for nb-ablation (default: {DEFAULT_NB_SIGMA_FLOORS}).",
    )
    parser.add_argument(
        "--density-style",
        choices=DENSITY_STYLES,
        default=None,
        help="Collated density rendering (default: lines / solid KDE curves).",
    )
    parser.add_argument(
        "--encode-split-linestyles",
        action="store_true",
        help="Use train (-), val (--), test (:) linestyles when multiple splits are plotted (default: all solid).",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated subset of naive_bayes,cfm,diffusion (default: all for full; naive_bayes only for nb-kernel-bandwidth).",
    )
    parser.add_argument(
        "--splits",
        type=str,
        default=None,
        help="Comma-separated subset of train,val,test (default: all).",
    )
    parser.add_argument("--nb-checkpoint", type=Path, default=CHECKPOINT_DIR / "naive_bayes_final.pt")
    parser.add_argument("--cfm-checkpoint", type=Path, default=CHECKPOINT_DIR / "cfm_final.pt")
    parser.add_argument("--diffusion-checkpoint", type=Path, default=CHECKPOINT_DIR / "diffusion_final.pt")
    parser.add_argument(
        "--max-grids-per-split",
        type=int,
        default=400,
        help="Subsample at most this many grid points per split for runtime control.",
    )
    parser.add_argument(
        "--events-per-grid",
        type=int,
        default=128,
        help="Number of truth/model events sampled per selected grid point.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--workers",
        type=int,
        default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")),
        help="CPU worker threads for BLAS/OpenMP (default: SLURM_CPUS_PER_TASK or 1).",
    )
    args = parser.parse_args()
    _log("04d: arguments parsed")
    workers = configure_worker_threads(args.workers)

    plot_set = args.plot_set
    if plot_set == "nb-kernel-bandwidth" and args.nb_kernel_bandwidths is None:
        parser.error("--nb-kernel-bandwidths is required when --plot-set nb-kernel-bandwidth")
    if args.nb_kernel_bandwidths is not None and plot_set not in (
        "nb-kernel-bandwidth",
        "nb-ablation",
    ):
        parser.error("--nb-kernel-bandwidths only applies to nb-kernel-bandwidth or nb-ablation")
    if args.nb_ablation_axes is not None and plot_set != "nb-ablation":
        parser.error("--nb-ablation-axes only applies to --plot-set nb-ablation")
    if args.nb_sigma_scales is not None and plot_set != "nb-ablation":
        parser.error("--nb-sigma-scales only applies to --plot-set nb-ablation")
    if args.nb_sigma_floors is not None and plot_set != "nb-ablation":
        parser.error("--nb-sigma-floors only applies to --plot-set nb-ablation")

    if args.splits is None:
        split_names = list(SPLIT_ORDER)
    else:
        split_names = _parse_csv_tokens(args.splits)
        bad = [s for s in split_names if s not in SPLIT_ORDER]
        if bad:
            raise ValueError(f"Unknown splits: {bad}. Expected subset of {SPLIT_ORDER}")

    if plot_set in ("nb-kernel-bandwidth", "nb-ablation", "nb-mode-collated"):
        model_names = ["naive_bayes"]
    elif args.models is None:
        model_names = list(MODEL_ORDER)
    else:
        model_names = _parse_csv_tokens(args.models)
        bad = [m for m in model_names if m not in MODEL_ORDER]
        if bad:
            raise ValueError(f"Unknown models: {bad}. Expected subset of {MODEL_ORDER}")

    density_style = args.density_style or "lines"
    encode_split_linestyles = bool(args.encode_split_linestyles)

    ablation_axes: List[str] = []
    if plot_set == "nb-ablation":
        ablation_axes = _parse_ablation_axes(args.nb_ablation_axes)

    need_cfm = "cfm" in model_names
    need_diff = "diffusion" in model_names
    need_nb = "naive_bayes" in model_names or plot_set in (
        "nb-kernel-bandwidth",
        "nb-ablation",
        "nb-mode-collated",
    )

    ckpts_to_check = []
    if need_nb:
        ckpts_to_check.append(args.nb_checkpoint)
    if need_cfm:
        ckpts_to_check.append(args.cfm_checkpoint)
    if need_diff:
        ckpts_to_check.append(args.diffusion_checkpoint)
    for ckpt in ckpts_to_check:
        if not ckpt.exists():
            raise FileNotFoundError(f"Missing checkpoint: {ckpt}")

    data_dir = ml_data_dir()
    hp_csv = data_dir / HYPERPARAM_TABLE_ENCODED_CSV.name
    events_pq = data_dir / ALL_EVENTS_PARQUET.name
    splits_json = data_dir / SPLITS_JSON.name
    for p in (hp_csv, events_pq, splits_json):
        if not p.exists():
            raise FileNotFoundError(f"Missing {p}. Run 02_build_dataset.py first.")

    _log(f"Plot set: {plot_set}")
    _log("Loading hyperparam table and splits...")
    hp_df = pd.read_csv(hp_csv)
    with open(splits_json) as f:
        splits = json.load(f)

    torch = _ensure_torch()
    device = torch.device(args.device)
    rng = np.random.default_rng(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    selected_by_split: Dict[str, np.ndarray] = {}
    all_grids: List[int] = []
    for split_name in split_names:
        sel = _select_grid_indices(
            split_indices=splits[split_name],
            max_grids=args.max_grids_per_split,
            rng=rng,
        )
        selected_by_split[split_name] = sel
        all_grids.extend(int(g) for g in sel)
    unique_grids = sorted(set(all_grids))
    _log(
        f"Selected {len(unique_grids)} unique grids "
        f"({args.max_grids_per_split} max per split × {len(split_names)} split(s))"
    )

    _log("Loading checkpoints...")
    models, checkpoint_tau = _load_models_for_run(
        device,
        model_names,
        nb_checkpoint=args.nb_checkpoint,
        cfm_checkpoint=args.cfm_checkpoint,
        diffusion_checkpoint=args.diffusion_checkpoint,
    )

    nb_kernel_bandwidths: Optional[List[float]] = None
    if plot_set == "nb-kernel-bandwidth":
        nb_kernel_bandwidths = _resolve_nb_kernel_bandwidths(
            args.nb_kernel_bandwidths,
            checkpoint_tau,
        )
        _log(
            f"NB Λ-kernel τ sweep: {nb_kernel_bandwidths} "
            f"(checkpoint τ={checkpoint_tau:g}; {len(nb_kernel_bandwidths)} generations per grid)"
        )

    tau_values: List[float] = []
    sigma_scales: List[float] = []
    sigma_floors: List[float] = []
    if plot_set == "nb-ablation":
        tau_values = _resolve_nb_kernel_bandwidths(
            args.nb_kernel_bandwidths or DEFAULT_NB_ABLATION_TAUS,
            checkpoint_tau,
        )
        sigma_scales = _parse_csv_floats(args.nb_sigma_scales or DEFAULT_NB_SIGMA_SCALES)
        sigma_floors = _parse_csv_floats(args.nb_sigma_floors or DEFAULT_NB_SIGMA_FLOORS)
        _log(f"NB ablation axes: {ablation_axes}")
        _log(f"  τ values: {tau_values}")
        _log(f"  σ scales: {sigma_scales}")
        _log(f"  σ_floor refits: {sigma_floors}")

    _log(f"Loading events parquet for {len(unique_grids)} grids (truth sampling)...")
    events_df = _load_events_parquet(events_pq, unique_grids)
    _log(f"  Loaded {len(events_df):,} event rows")
    events_by_grid = _build_events_by_grid(events_df)
    del events_df

    events_df_full: Optional[pd.DataFrame] = None
    if plot_set in ("nb-ablation", "nb-mode-collated") and (
        plot_set == "nb-mode-collated"
        or any(a in ablation_axes for a in ("sigma-floor", "mode"))
    ):
        _log("Loading full events parquet for NB refit axes (sigma-floor, mode)...")
        events_df_full = pd.read_parquet(
            events_pq,
            columns=["mchirp", "q", "z", "grid_idx"],
        )
        _log(f"  Loaded {len(events_df_full):,} event rows for refit")

    plots_dir = plot_run_dir(Path(__file__), timestamp=datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    if args.run_tag.strip():
        plots_dir = plots_dir / args.run_tag.strip()
        plots_dir.mkdir(parents=True, exist_ok=True)
    _log(f"Writing outputs to {plots_dir}")
    _log(f"Using worker threads: {workers}")

    split_catalogs: Dict[str, Dict[str, pd.DataFrame]] = {}
    if plot_set not in ("nb-ablation", "nb-mode-collated"):
        for split_name in split_names:
            split_catalogs[split_name] = _collect_split_catalogs(
                split_name=split_name,
                split_indices=selected_by_split[split_name],
                hp_df=hp_df,
                events_by_grid=events_by_grid,
                models=models,
                n_events_per_grid=args.events_per_grid,
                rng=rng,
                model_names=model_names,
                nb_kernel_bandwidths=nb_kernel_bandwidths,
            )

    if plot_set == "nb-mode-collated":
        assert events_df_full is not None
        _run_nb_mode_collated(
            plots_dir,
            gaussian_bundle=models["naive_bayes"],
            checkpoint_tau=checkpoint_tau,
            hp_df=hp_df,
            events_df_full=events_df_full,
            events_by_grid=events_by_grid,
            selected_by_split=selected_by_split,
            split_names=split_names,
            rng=rng,
            n_events_per_grid=args.events_per_grid,
            encode_split_linestyles=encode_split_linestyles,
        )

    if plot_set == "nb-ablation":
        base_bundle = models["naive_bayes"]
        _run_nb_ablation_suite(
            plots_dir,
            axes=ablation_axes,
            base_bundle=base_bundle,
            checkpoint_tau=checkpoint_tau,
            hp_df=hp_df,
            events_df_full=events_df_full,
            events_by_grid=events_by_grid,
            selected_by_split=selected_by_split,
            split_names=split_names,
            rng=rng,
            tau_values=tau_values,
            sigma_scales=sigma_scales,
            sigma_floors=sigma_floors,
            args=args,
        )

    if plot_set in ("full", "collated-density"):
        _plot_density_collated_by_observable(
            split_catalogs,
            plots_dir / "01b_density_overlays_collated.png",
            splits=split_names,
            density_style=density_style,
            encode_split_linestyles=encode_split_linestyles,
        )

    if plot_set == "nb-kernel-bandwidth":
        assert nb_kernel_bandwidths is not None
        _log("Rendering NB kernel-bandwidth collated density...")
        _plot_density_collated_nb_kernel_bandwidth(
            split_catalogs,
            plots_dir / "01b_density_overlays_collated_nb_kernel_bandwidth.png",
            splits=split_names,
            taus=nb_kernel_bandwidths,
            checkpoint_tau=checkpoint_tau,
            encode_split_linestyles=encode_split_linestyles,
        )
        _write_nb_bandwidth_meta(
            plots_dir / "nb_kernel_bandwidth_ablation.json",
            checkpoint_tau=checkpoint_tau,
            taus=nb_kernel_bandwidths,
            splits=split_names,
            args=args,
        )

    if plot_set == "full":
        _plot_density_grid(split_catalogs, plots_dir / "01_density_overlays.png")
        _plot_ecdf_grid(split_catalogs, plots_dir / "02_ecdf_overlays.png")
        metrics_df = _compute_metrics(split_catalogs)
        metrics_df.to_csv(plots_dir / "metrics_by_split_model_observable.csv", index=False)
        _plot_metric_heatmaps(metrics_df, plots_dir)
        _plot_test_qq(split_catalogs, plots_dir / "04_test_qq.png")
        _write_plot_guide(plots_dir / "plot_guide.md", plot_set)

    _log("Done.")
    _log(f"  Plots and metrics: {plots_dir}")


if __name__ == "__main__":
    _log("04d: script entry")
    main()
