#!/usr/bin/env python3
"""
Step 00 — Heatmaps of intrinsic merger rate or event count on the (sfr_a, mu0) grid.

Uses per-grid summaries from ``data/hyperparam_table.csv`` (written by ``02_build_dataset.py``
from ``00`` HDF5 output). Each SSPC grid cell stores:

- ``sum_weight`` — total intrinsic merger-rate weight in that cell (recommended for "rate")
- ``n_systems`` — number of sampled merger rows stored in the HDF5 for that cell

Produces one figure with three panels (SMT, CE, CHE).

Usage::

    python scripts/analysis/00_grid_rate_heatmaps.py
    python scripts/analysis/00_grid_rate_heatmaps.py --metric count --log-scale
    python scripts/analysis/00_grid_rate_heatmaps.py --sspc-hdf5 data/sspc/models_sspc.hdf5

SLURM: ``slurm/00_grid_rate_heatmaps.sh``
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Literal, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_ANALYSIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _ANALYSIS_DIR.parents[2]
for _p in (_PROJECT_ROOT, _ANALYSIS_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from plant_paths import (  # noqa: E402
    HYPERPARAM_TABLE_CSV,
    PROJECT_ROOT,
    ensure_paths,
    find_data_dir,
    resolve_plot_output,
)

ensure_paths()

CHANNELS = ("SMT", "CE", "CHE")
MetricName = Literal["rate", "count", "log_rate"]


def _load_build_dataset_module():
    path = _PROJECT_ROOT / "scripts" / "02_build_dataset.py"
    spec = importlib.util.spec_from_file_location("_plant_build_dataset", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_grid_table(hyperparam_csv: Path | None, sspc_hdf5: Path | None) -> pd.DataFrame:
    if hyperparam_csv is not None and hyperparam_csv.is_file():
        hp = pd.read_csv(hyperparam_csv)
    elif sspc_hdf5 is not None and sspc_hdf5.is_file():
        print(f"Building grid table from HDF5: {sspc_hdf5}", flush=True)
        mod = _load_build_dataset_module()
        hp = mod.build_hyperparam_table(sspc_hdf5, data_source="sspc")
    else:
        raise FileNotFoundError(
            "Need data/hyperparam_table.csv (run 02_build_dataset.py) or --sspc-hdf5."
        )
    for col in ("channel", "sfra", "mu0", "sum_weight", "n_systems"):
        if col not in hp.columns:
            raise ValueError(f"hyperparam table missing column {col!r}")
    return hp


def metric_values(hp: pd.DataFrame, metric: MetricName) -> pd.Series:
    if metric == "rate":
        return hp["sum_weight"].astype(np.float64)
    if metric == "count":
        return hp["n_systems"].astype(np.float64)
    if metric == "log_rate":
        w = hp["sum_weight"].astype(np.float64)
        return np.log10(np.maximum(w, 1e-30))
    raise ValueError(metric)


def pivot_channel(
    hp: pd.DataFrame,
    channel: str,
    metric: MetricName,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (sfra_axis, mu0_axis, Z) with shape (n_sfra, n_mu0)."""
    sub = hp.loc[hp["channel"].astype(str) == channel].copy()
    if sub.empty:
        raise ValueError(f"No rows for channel {channel!r}")

    sub["_metric"] = metric_values(sub, metric)
    sfra_vals = np.sort(sub["sfra"].unique())
    mu0_vals = np.sort(sub["mu0"].unique())

    z = np.full((len(sfra_vals), len(mu0_vals)), np.nan, dtype=np.float64)
    sfra_to_i = {float(v): i for i, v in enumerate(sfra_vals)}
    mu0_to_j = {float(v): j for j, v in enumerate(mu0_vals)}

    for _, row in sub.iterrows():
        i = sfra_to_i[float(row["sfra"])]
        j = mu0_to_j[float(row["mu0"])]
        z[i, j] = float(row["_metric"])

    if np.isnan(z).any():
        missing = int(np.isnan(z).sum())
        print(f"  [{channel}] warning: {missing} grid cells missing in table", flush=True)

    return sfra_vals, mu0_vals, z


def plot_heatmaps(
    hp: pd.DataFrame,
    metric: MetricName,
    *,
    log_colorbar: bool,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)

    label_map = {
        "rate": r"$\sum w$ (intrinsic merger-rate weight)",
        "count": "Stored merger count ($N$)",
        "log_rate": r"$\log_{10}(\sum w)$",
    }
    title_map = {
        "rate": "Intrinsic merger-rate weight",
        "count": "Merger count per grid cell",
        "log_rate": r"$\log_{10}$ intrinsic merger-rate weight",
    }

    ims = []
    for ax, ch in zip(axes, CHANNELS):
        sfra, mu0, z = pivot_channel(hp, ch, metric)
        z_plot = np.ma.masked_invalid(z)
        if log_colorbar:
            positive = z_plot[np.isfinite(z_plot)] > 0
            if not np.any(positive):
                raise ValueError(f"Channel {ch}: no positive values for log color scale")
            im = ax.pcolormesh(
                mu0,
                sfra,
                z_plot,
                shading="nearest",
                norm=matplotlib.colors.LogNorm(
                    vmin=float(np.nanmin(z_plot[z_plot > 0])),
                    vmax=float(np.nanmax(z_plot)),
                ),
                cmap="rocket_r",
            )
        else:
            im = ax.pcolormesh(mu0, sfra, z_plot, shading="nearest", cmap="rocket_r")
        ax.set_title(ch)
        ax.set_xlabel(r"$\mu_0$")
        ax.set_ylabel(r"$a_{\rm SF}$")
        ims.append(im)

    fig.suptitle(title_map[metric], fontsize=12)
    cbar = fig.colorbar(ims[-1], ax=axes.ravel().tolist(), shrink=0.85, pad=0.02)
    cbar.set_label(label_map[metric])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Heatmaps of merger rate or count on the SSPC (sfr_a, mu0) grid."
    )
    p.add_argument(
        "--hyperparam-csv",
        type=Path,
        default=HYPERPARAM_TABLE_CSV,
        help="Per-grid table from 02_build_dataset.py (default: data/hyperparam_table.csv).",
    )
    p.add_argument(
        "--sspc-hdf5",
        type=Path,
        default=None,
        help="If CSV missing, build table from this HDF5 (slow; same as 02).",
    )
    p.add_argument(
        "--metric",
        choices=("rate", "count", "log_rate"),
        default="rate",
        help="rate=sum_weight; count=n_systems; log_rate=log10(sum_weight).",
    )
    p.add_argument(
        "--log-scale",
        action="store_true",
        help="Log color scale (only for metric=rate; ignored for log_rate).",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output PNG (default: plots/00_grid_rate_heatmaps/<timestamp>/grid_<metric>.png).",
    )
    p.add_argument(
        "--no-timestamp-subdir",
        action="store_true",
        help="Write directly under plots/00_grid_rate_heatmaps/ (no timestamp subdir).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    hdf5 = args.sspc_hdf5
    if hdf5 is None:
        default_h5 = find_data_dir() / "sspc" / "models_sspc.hdf5"
        if not args.hyperparam_csv.is_file() and default_h5.is_file():
            hdf5 = default_h5

    hp = load_grid_table(
        args.hyperparam_csv if args.hyperparam_csv.is_file() else None,
        hdf5,
    )

    metric: MetricName = args.metric
    log_cbar = args.log_scale and metric == "rate"

    if args.out is not None:
        out = args.out.resolve()
    else:
        out = resolve_plot_output(
            Path(__file__),
            no_timestamp_subdir=args.no_timestamp_subdir,
            filename=f"grid_{metric}.png",
        )

    plot_heatmaps(hp, metric, log_colorbar=log_cbar, out_path=out)

    meta = {
        "metric": metric,
        "log_colorbar": log_cbar,
        "n_rows": int(len(hp)),
        "channels": list(CHANNELS),
        "sfra_range": [float(hp["sfra"].min()), float(hp["sfra"].max())],
        "mu0_range": [float(hp["mu0"].min()), float(hp["mu0"].max())],
    }
    meta_path = out.with_suffix(".json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Saved → {meta_path}", flush=True)


if __name__ == "__main__":
    main()
