#!/usr/bin/env python3
"""
Step 02 — Intrinsic data validation (``data/`` parquets and splits).

Validates Step-02 outputs using ``data/all_events.parquet`` (no detection weighting).

Usage::

    python scripts/analysis/02_validation.py

SLURM: ``slurm/02b_data_validation.sh``

Outputs: ``test/reports/validation/<timestamp>/`` and ``plots/02_validation/<timestamp>/``.
"""
from __future__ import annotations

import sys
from pathlib import Path as _Path

_ANALYSIS_DIR = _Path(__file__).resolve().parent
_PROJECT_ROOT = _ANALYSIS_DIR.parents[2]
for _p in (_PROJECT_ROOT, _ANALYSIS_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from plant_paths import (  # noqa: E402
    ALL_EVENTS_PARQUET,
    HYPERPARAM_TABLE_CSV,
    HYPERPARAM_TABLE_ENCODED_CSV,
    PLOTS_ROOT,
    SPLITS_JSON,
    ensure_paths,
    plot_run_dir,
    plot_script_stem,
)

ensure_paths()

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


@dataclass
class ValidationConfig:
    project_root: Path
    hyperparam_csv: Path
    hyperparam_encoded_csv: Path
    splits_json: Path
    events_parquet: Path
    report_dir: Path
    plot_dir: Path
    rare_quantile: float = 0.05
    max_scatter_points: int = 30_000
    seed: int = 42


def _default_config(project_root: Path, timestamp: str | None = None) -> ValidationConfig:
    ts = timestamp or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base_reports = project_root / "test" / "reports" / "validation"
    return ValidationConfig(
        project_root=project_root,
        hyperparam_csv=HYPERPARAM_TABLE_CSV,
        hyperparam_encoded_csv=HYPERPARAM_TABLE_ENCODED_CSV,
        splits_json=SPLITS_JSON,
        events_parquet=ALL_EVENTS_PARQUET,
        report_dir=base_reports / ts,
        plot_dir=plot_run_dir(Path(__file__), timestamp=ts),
    )


def _to_builtin(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): _to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_builtin(v) for v in obj]
    return obj


def _load_inputs(cfg: ValidationConfig) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, list[int]], pd.DataFrame]:
    missing = [str(p) for p in [cfg.hyperparam_csv, cfg.hyperparam_encoded_csv, cfg.splits_json, cfg.events_parquet] if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required files:\n- " + "\n- ".join(missing) + "\nRun 02_build_dataset.py first."
        )
    hp = pd.read_csv(cfg.hyperparam_csv)
    hp_enc = pd.read_csv(cfg.hyperparam_encoded_csv)
    with open(cfg.splits_json, "r", encoding="utf-8") as f:
        splits = json.load(f)
    events = pd.read_parquet(cfg.events_parquet)
    return hp, hp_enc, splits, events


def _save_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_to_builtin(obj), f, indent=2)


def _safe_hist_kl(x: np.ndarray, y: np.ndarray, bins: int = 80) -> float:
    lo = float(min(np.nanmin(x), np.nanmin(y)))
    hi = float(max(np.nanmax(x), np.nanmax(y)))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        return float("nan")
    h1, edges = np.histogram(x, bins=bins, range=(lo, hi), density=True)
    h2, _ = np.histogram(y, bins=edges, density=True)
    eps = 1e-12
    p = h1 + eps
    q = h2 + eps
    p = p / np.sum(p)
    q = q / np.sum(q)
    return float(np.sum(p * np.log(p / q)))


def _rbf_mmd2(x: np.ndarray, y: np.ndarray, gamma: float | None = None) -> float:
    # Lightweight 1D MMD^2 estimate for fast sanity checks.
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    if gamma is None:
        pooled = np.vstack([x, y]).ravel()
        med = np.median(np.abs(pooled[:, None] - pooled[None, :]))
        gamma = 1.0 / (2.0 * (med**2 + 1e-12))

    def _k(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        d2 = (a - b.T) ** 2
        return np.exp(-gamma * d2)

    kxx = _k(x, x)
    kyy = _k(y, y)
    kxy = _k(x, y)
    m = len(x)
    n = len(y)
    if m < 2 or n < 2:
        return float("nan")
    term_xx = (np.sum(kxx) - np.trace(kxx)) / (m * (m - 1))
    term_yy = (np.sum(kyy) - np.trace(kyy)) / (n * (n - 1))
    term_xy = 2.0 * np.mean(kxy)
    return float(term_xx + term_yy - term_xy)


def check_grid_coverage(cfg: ValidationConfig, hp_enc: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {"name": "grid_coverage", "status": "pass", "notes": []}

    lambda_cols = sorted([c for c in hp_enc.columns if c.startswith("lambda_")], key=lambda c: int(c.split("_")[1]))
    if not lambda_cols:
        out["status"] = "fail"
        out["notes"].append("No lambda_* columns found in hyperparam_table_encoded.csv.")
        return out

    stats = {}
    for col in lambda_cols:
        vals = hp_enc[col].to_numpy(dtype=float)
        stats[col] = {
            "min": float(np.nanmin(vals)),
            "max": float(np.nanmax(vals)),
            "n_unique": int(np.unique(np.round(vals, 10)).size),
        }
    out["lambda_stats"] = stats

    # CE occupancy heatmap in intrinsic hyperparameter space.
    ce = hp_enc[hp_enc["channel"] == "CE"].copy()
    if len(ce) > 0:
        x = ce["chi_b"].to_numpy(dtype=float)
        y = ce["alpha_CE"].to_numpy(dtype=float)
        bins_x = np.linspace(np.nanmin(x), np.nanmax(x), 20)
        bins_y = np.linspace(np.nanmin(y), np.nanmax(y), 20)
        h, _, _ = np.histogram2d(x, y, bins=[bins_x, bins_y])
        occ = float(np.mean(h > 0))
        out["ce_occupancy_fraction"] = occ
        if occ < 0.5:
            out["status"] = "warn"
            out["notes"].append(f"Low CE occupancy fraction: {occ:.3f}")

        plt.figure(figsize=(7, 5))
        plt.imshow(h.T, origin="lower", aspect="auto", interpolation="nearest")
        plt.colorbar(label="Grid points per bin")
        plt.title("CE Occupancy in (chi_b, alpha_CE)")
        plt.xlabel("chi_b bin")
        plt.ylabel("alpha_CE bin")
        plt.tight_layout()
        plt.savefig(cfg.plot_dir / "ce_occupancy_heatmap.png", dpi=180)
        plt.close()
    else:
        out["status"] = "warn"
        out["notes"].append("No CE rows found in hyperparameter table.")

    # Simple pair plot (sampled).
    pairs_cols = [c for c in ["chi_b", "alpha_CE"] if c in hp_enc.columns]
    if len(pairs_cols) == 2:
        sample = hp_enc.sample(min(len(hp_enc), 4000), random_state=cfg.seed)
        channels = sorted(sample["channel"].unique())
        cmap = plt.get_cmap("tab10")
        plt.figure(figsize=(7, 6))
        for i, ch in enumerate(channels):
            sub = sample[sample["channel"] == ch]
            plt.scatter(sub["chi_b"], sub["alpha_CE"], s=10, alpha=0.6, label=ch, color=cmap(i))
        plt.xlabel("chi_b")
        plt.ylabel("alpha_CE")
        plt.title("Grid Coverage by Channel")
        plt.legend(frameon=False, fontsize=8)
        plt.tight_layout()
        plt.savefig(cfg.plot_dir / "grid_pairplot.png", dpi=180)
        plt.close()

    return out


def check_channel_health(cfg: ValidationConfig, hp: pd.DataFrame, events: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {"name": "channel_health", "status": "pass", "notes": []}

    required = {"channel", "sum_weight", "log_efficiency"}
    if not required.issubset(hp.columns):
        out["status"] = "fail"
        out["notes"].append(f"hyperparam_table.csv missing required columns: {sorted(required - set(hp.columns))}")
        return out

    summary = (
        hp.groupby("channel", dropna=False)
        .agg(
            n_grid=("channel", "size"),
            sum_weight_total=("sum_weight", "sum"),
            sum_weight_median=("sum_weight", "median"),
            log_eff_mean=("log_efficiency", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(cfg.report_dir / "channel_health_summary.csv", index=False)
    out["channels"] = summary.to_dict(orient="records")

    channel_counts = hp["channel"].value_counts().sort_index()
    if (channel_counts == 0).any():
        out["status"] = "warn"
        out["notes"].append("At least one channel has zero grid points.")

    plt.figure(figsize=(7, 4))
    channel_counts.plot(kind="bar", color="#5B8FF9")
    plt.ylabel("Grid points")
    plt.title("Grid Points per Channel")
    plt.tight_layout()
    plt.savefig(cfg.plot_dir / "channel_counts.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7, 4))
    hp.boxplot(column="log_efficiency", by="channel", grid=False)
    plt.suptitle("")
    plt.title("Intrinsic log_efficiency by Channel")
    plt.xlabel("channel")
    plt.ylabel("log_efficiency")
    plt.tight_layout()
    plt.savefig(cfg.plot_dir / "channel_rate_distributions.png", dpi=180)
    plt.close()

    return out


def check_event_validity(cfg: ValidationConfig, events: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {"name": "event_validity", "status": "pass", "notes": []}
    required = ["mchirp", "q", "z", "grid_idx"]
    missing = [c for c in required if c not in events.columns]
    if missing:
        out["status"] = "fail"
        out["notes"].append(f"all_events.parquet missing columns: {missing}")
        return out

    violations: dict[str, int] = {}
    violations["nan_or_inf_mchirp"] = int((~np.isfinite(events["mchirp"])).sum())
    violations["nan_or_inf_q"] = int((~np.isfinite(events["q"])).sum())
    violations["nan_or_inf_z"] = int((~np.isfinite(events["z"])).sum())
    violations["mchirp_nonpositive"] = int((events["mchirp"] <= 0).sum())
    violations["q_out_of_bounds"] = int(((events["q"] <= 0) | (events["q"] > 1)).sum())
    violations["z_negative"] = int((events["z"] < 0).sum())
    if "weight" in events.columns:
        violations["weight_negative"] = int((events["weight"] < 0).sum())
    else:
        out["notes"].append("Optional column 'weight' missing; skipped weight-bound checks.")

    total_bad = int(sum(violations.values()))
    out["violations"] = violations
    if total_bad > 0:
        out["status"] = "warn"
        out["notes"].append(f"Found {total_bad} hard-bound/NaN violations.")

    qtiles = {}
    for col in ["mchirp", "q", "z"]:
        vals = events[col].to_numpy(dtype=float)
        qtiles[col] = {
            "p001": float(np.nanpercentile(vals, 0.1)),
            "p50": float(np.nanpercentile(vals, 50)),
            "p999": float(np.nanpercentile(vals, 99.9)),
        }
    if "weight" in events.columns:
        vals = events["weight"].to_numpy(dtype=float)
        qtiles["weight"] = {
            "p001": float(np.nanpercentile(vals, 0.1)),
            "p50": float(np.nanpercentile(vals, 50)),
            "p999": float(np.nanpercentile(vals, 99.9)),
        }
    out["quantiles"] = qtiles

    # Save detailed violations table.
    bad_mask = (
        (~np.isfinite(events["mchirp"]))
        | (~np.isfinite(events["q"]))
        | (~np.isfinite(events["z"]))
        | (events["mchirp"] <= 0)
        | (events["q"] <= 0)
        | (events["q"] > 1)
        | (events["z"] < 0)
    )
    if "weight" in events.columns:
        bad_mask = bad_mask | (events["weight"] < 0)
    events.loc[bad_mask].head(10000).to_csv(cfg.report_dir / "event_bounds_violations.csv", index=False)

    # Hist panel.
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.5))
    cols = ["mchirp", "q", "z"]
    for ax, col in zip(axs.ravel(), cols):
        vals = events[col].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        ax.hist(vals, bins=80, color="#5AD8A6", alpha=0.85)
        ax.set_title(col)
    fig.tight_layout()
    fig.savefig(cfg.plot_dir / "event_histograms.png", dpi=180)
    plt.close(fig)

    return out


def check_split_hygiene(cfg: ValidationConfig, hp: pd.DataFrame, splits: dict[str, list[int]], events: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {"name": "split_hygiene", "status": "pass", "notes": []}
    split_names = ["train", "val", "test"]
    if not all(s in splits for s in split_names):
        out["status"] = "fail"
        out["notes"].append("splits.json must contain train/val/test keys.")
        return out

    split_sets = {s: set(map(int, splits[s])) for s in split_names}
    overlaps = {
        "train_val": sorted(split_sets["train"] & split_sets["val"]),
        "train_test": sorted(split_sets["train"] & split_sets["test"]),
        "val_test": sorted(split_sets["val"] & split_sets["test"]),
    }
    out["overlaps"] = {k: len(v) for k, v in overlaps.items()}
    if any(len(v) > 0 for v in overlaps.values()):
        out["status"] = "fail"
        out["notes"].append("Grid index overlap found across splits.")

    all_idx = split_sets["train"] | split_sets["val"] | split_sets["test"]
    expected = set(range(len(hp)))
    missing = sorted(expected - all_idx)
    extra = sorted(all_idx - expected)
    out["n_missing_idx"] = len(missing)
    out["n_extra_idx"] = len(extra)
    if missing or extra:
        out["status"] = "warn" if out["status"] != "fail" else out["status"]
        out["notes"].append("Split index set does not exactly match hyperparameter rows.")

    # Split/channel balance table.
    rows = []
    for split_name in split_names:
        idx = list(split_sets[split_name])
        sub = hp.iloc[idx]
        dist = sub["channel"].value_counts(normalize=True).to_dict()
        rows.append({"split": split_name, **dist, "n_grid": len(idx)})
    split_balance = pd.DataFrame(rows).fillna(0.0)
    split_balance.to_csv(cfg.report_dir / "split_channel_balance.csv", index=False)

    # Distance of test lambda to nearest train lambda.
    lambda_cols = sorted([c for c in hp.columns if c.startswith("lambda_")], key=lambda c: int(c.split("_")[1]))
    if lambda_cols and len(split_sets["train"]) > 0 and len(split_sets["test"]) > 0:
        train_x = hp.iloc[list(split_sets["train"])][lambda_cols].to_numpy(dtype=float)
        test_x = hp.iloc[list(split_sets["test"])][lambda_cols].to_numpy(dtype=float)
        # brute-force nearest distance (small table, acceptable)
        dmin = []
        for row in test_x:
            d = np.sqrt(np.sum((train_x - row[None, :]) ** 2, axis=1))
            dmin.append(float(np.min(d)))
        out["test_to_train_lambda_distance"] = {
            "min": float(np.min(dmin)),
            "median": float(np.median(dmin)),
            "max": float(np.max(dmin)),
        }
        plt.figure(figsize=(7, 4))
        plt.hist(dmin, bins=30, color="#F6BD16", alpha=0.9)
        plt.title("Nearest Train Distance for Test Lambda Points")
        plt.xlabel("L2 distance in lambda-space")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(cfg.plot_dir / "lambda_distance_test_to_train.png", dpi=180)
        plt.close()

    # Event index leakage check by grid_idx
    ev_idx = set(events["grid_idx"].astype(int).unique())
    if not ev_idx.issubset(expected):
        out["status"] = "warn" if out["status"] != "fail" else out["status"]
        out["notes"].append("Events contain grid_idx outside hyperparameter table range.")

    return out


def check_rare_events(cfg: ValidationConfig, hp: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {"name": "rare_event_diagnostics", "status": "pass", "notes": []}
    if "sum_weight" not in hp.columns:
        out["status"] = "fail"
        out["notes"].append("hyperparam_table.csv missing sum_weight for intrinsic rarity check.")
        return out

    threshold = float(np.quantile(hp["sum_weight"], cfg.rare_quantile))
    rare = hp[hp["sum_weight"] <= threshold].copy()
    out["rare_threshold_sum_weight"] = threshold
    out["n_rare_grid"] = int(len(rare))
    out["rare_fraction"] = float(len(rare) / max(len(hp), 1))
    out["rare_by_channel"] = rare["channel"].value_counts().to_dict()
    if len(rare) == 0:
        out["status"] = "warn"
        out["notes"].append("No rare grids found at configured quantile.")

    # CDF plot.
    vals = np.sort(hp["sum_weight"].to_numpy(dtype=float))
    y = np.linspace(0, 1, len(vals), endpoint=True)
    plt.figure(figsize=(7, 4))
    plt.plot(vals, y, color="#722ED1", lw=2)
    plt.axvline(threshold, color="red", ls="--", lw=1.5, label=f"{int(cfg.rare_quantile*100)}th percentile")
    plt.xscale("log")
    plt.xlabel("sum_weight (intrinsic)")
    plt.ylabel("CDF")
    plt.title("Intrinsic Rare-Region CDF")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(cfg.plot_dir / "rare_event_cdf.png", dpi=180)
    plt.close()

    # CE rare coverage map.
    ce = hp[hp["channel"] == "CE"].copy()
    if len(ce) > 0:
        ce["is_rare"] = ce["sum_weight"] <= threshold
        pivot = ce.pivot_table(index="alpha_CE", columns="chi_b", values="is_rare", aggfunc="mean")
        plt.figure(figsize=(7, 5))
        plt.imshow(pivot.values, origin="lower", aspect="auto", interpolation="nearest", vmin=0, vmax=1)
        plt.colorbar(label="Rare flag fraction")
        plt.title("CE Rare Coverage in (chi_b, alpha_CE)")
        plt.xlabel("chi_b index")
        plt.ylabel("alpha_CE index")
        plt.tight_layout()
        plt.savefig(cfg.plot_dir / "rare_ce_coverage.png", dpi=180)
        plt.close()

    pd.DataFrame(
        {
            "key": hp["key"],
            "channel": hp["channel"],
            "sum_weight": hp["sum_weight"],
            "is_rare": hp["sum_weight"] <= threshold,
        }
    ).to_csv(cfg.report_dir / "rare_event_summary.csv", index=False)
    return out


def check_distribution_sanity(cfg: ValidationConfig, hp: pd.DataFrame, events: pd.DataFrame) -> dict[str, Any]:
    """
    Intrinsic distribution sanity check.
    Uses train-vs-test split comparisons as an internal consistency check
    when external TNG comparison files are not guaranteed.
    """
    out: dict[str, Any] = {"name": "distribution_sanity", "status": "pass", "notes": []}
    splits_path = cfg.splits_json
    with open(splits_path, "r", encoding="utf-8") as f:
        splits = json.load(f)
    train_idx = set(map(int, splits["train"]))
    test_idx = set(map(int, splits["test"]))
    ev_train = events[events["grid_idx"].astype(int).isin(train_idx)]
    ev_test = events[events["grid_idx"].astype(int).isin(test_idx)]
    if len(ev_train) == 0 or len(ev_test) == 0:
        out["status"] = "warn"
        out["notes"].append("Train/test events unavailable for intrinsic distribution sanity check.")
        return out

    metrics = []
    cols = ["mchirp", "q", "z"]
    for col in cols:
        x = ev_train[col].to_numpy(dtype=float)
        y = ev_test[col].to_numpy(dtype=float)
        x = x[np.isfinite(x)]
        y = y[np.isfinite(y)]
        # subsample for MMD runtime
        if len(x) > 8000:
            x = np.random.default_rng(cfg.seed).choice(x, size=8000, replace=False)
        if len(y) > 8000:
            y = np.random.default_rng(cfg.seed + 1).choice(y, size=8000, replace=False)
        kl = _safe_hist_kl(x, y)
        ks = float(ks_2samp(x, y).statistic) if len(x) > 10 and len(y) > 10 else float("nan")
        mmd2 = _rbf_mmd2(x, y)
        metrics.append({"observable": col, "kl_train_test": kl, "ks_train_test": ks, "mmd2_train_test": mmd2})

    mdf = pd.DataFrame(metrics)
    mdf.to_csv(cfg.report_dir / "distribution_metrics.csv", index=False)
    out["metrics"] = mdf.to_dict(orient="records")

    # Simple warning thresholds (internal consistency).
    if np.nanmean(mdf["ks_train_test"]) > 0.25:
        out["status"] = "warn"
        out["notes"].append("Large train-vs-test KS divergence; possible split/domain mismatch.")

    # Plot overlays.
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.5))
    for ax, col in zip(axs.ravel(), cols):
        tr = ev_train[col].to_numpy(dtype=float)
        te = ev_test[col].to_numpy(dtype=float)
        tr = tr[np.isfinite(tr)]
        te = te[np.isfinite(te)]
        ax.hist(tr, bins=80, density=True, histtype="step", lw=1.8, label="train")
        ax.hist(te, bins=80, density=True, histtype="step", lw=1.8, label="test")
        ax.set_title(col)
    axs[0, 0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(cfg.plot_dir / "marginal_comparison_panel.png", dpi=180)
    plt.close(fig)

    # Intrinsic z-shape by channel.
    if "channel_id" in events.columns:
        fig, ax = plt.subplots(figsize=(8, 4))
        for ch_id in sorted(events["channel_id"].dropna().unique()):
            sub = events[events["channel_id"] == ch_id]
            z = sub["z"].to_numpy(dtype=float)
            z = z[np.isfinite(z)]
            if len(z) == 0:
                continue
            h, bins = np.histogram(z, bins=80, density=True)
            centers = 0.5 * (bins[:-1] + bins[1:])
            ax.plot(centers, h, lw=1.6, label=f"channel_id={int(ch_id)}")
        ax.set_yscale("log")
        ax.set_xlabel("z")
        ax.set_ylabel("density")
        ax.set_title("Intrinsic Redshift Shape by Channel")
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(cfg.plot_dir / "redshift_rate_shape_comparison.png", dpi=180)
        plt.close(fig)

    return out


def _build_markdown_summary(results: list[dict[str, Any]], out_path: Path) -> None:
    lines = [
        "# Intrinsic Data Validation Summary",
        "",
        "Validation source: `all_events.parquet` (intrinsic full-range events; no detectability filter).",
        "",
        "| Check | Status | Notes |",
        "|---|---|---|",
    ]
    for r in results:
        notes = "; ".join(r.get("notes", [])) if r.get("notes") else ""
        lines.append(f"| `{r.get('name','unknown')}` | **{r.get('status','unknown')}** | {notes} |")
    lines += [
        "",
        "## Generated Artifacts",
        "",
        "- Reports: `test/reports/validation/<timestamp>/`; plots: `plots/02_validation/<timestamp>/` when using default paths",
        "",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def run_validation(cfg: ValidationConfig) -> dict[str, Any]:
    cfg.report_dir.mkdir(parents=True, exist_ok=True)
    cfg.plot_dir.mkdir(parents=True, exist_ok=True)

    hp, hp_enc, splits, events = _load_inputs(cfg)
    checks = [
        check_grid_coverage(cfg, hp_enc),
        check_channel_health(cfg, hp, events),
        check_event_validity(cfg, events),
        check_split_hygiene(cfg, hp_enc, splits, events),
        check_rare_events(cfg, hp),
        check_distribution_sanity(cfg, hp, events),
    ]
    overall = "pass"
    if any(c["status"] == "fail" for c in checks):
        overall = "fail"
    elif any(c["status"] == "warn" for c in checks):
        overall = "warn"

    summary = {
        "overall_status": overall,
        "n_checks": len(checks),
        "checks": checks,
        "inputs": {
            "hyperparam_csv": str(cfg.hyperparam_csv),
            "hyperparam_encoded_csv": str(cfg.hyperparam_encoded_csv),
            "splits_json": str(cfg.splits_json),
            "events_parquet": str(cfg.events_parquet),
        },
        "output_dirs": {"report_dir": str(cfg.report_dir), "plot_dir": str(cfg.plot_dir)},
    }
    _save_json(cfg.report_dir / "validation_summary.json", summary)
    _build_markdown_summary(checks, cfg.report_dir / "validation_summary.md")
    return summary


def parse_args() -> argparse.Namespace:
    default_root = Path(__file__).resolve().parents[2]
    p = argparse.ArgumentParser(description="Run intrinsic data validation suite.")
    p.add_argument("--project-root", type=Path, default=default_root, help="Path to PLANT_GW_Paleontology root")
    p.add_argument("--events-parquet", type=Path, default=None, help="Override intrinsic events parquet path")
    p.add_argument("--rare-quantile", type=float, default=0.05, help="Quantile threshold for rare intrinsic regions")
    p.add_argument("--strict", action="store_true", help="Return non-zero exit code on warn/fail")
    p.add_argument(
        "--no-timestamp-subdir",
        action="store_true",
        help="Write reports to test/reports/validation and plots to plots/02_validation/ (no timestamp subdir).",
    )
    p.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Optional subfolder name (default: current timestamp). Ignored if --no-timestamp-subdir.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ts: str | None = None
    if args.no_timestamp_subdir:
        root = args.project_root.resolve()
        cfg = ValidationConfig(
            project_root=root,
            hyperparam_csv=HYPERPARAM_TABLE_CSV,
            hyperparam_encoded_csv=HYPERPARAM_TABLE_ENCODED_CSV,
            splits_json=SPLITS_JSON,
            events_parquet=ALL_EVENTS_PARQUET,
            report_dir=root / "test" / "reports" / "validation",
            plot_dir=PLOTS_ROOT / plot_script_stem(Path(__file__)),
            rare_quantile=float(args.rare_quantile),
            max_scatter_points=30_000,
            seed=42,
        )
    else:
        ts = args.run_id or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        cfg = _default_config(args.project_root.resolve(), timestamp=ts)
    if args.events_parquet is not None:
        cfg.events_parquet = args.events_parquet.resolve()
    cfg.rare_quantile = float(args.rare_quantile)

    summary = run_validation(cfg)
    print("=== Intrinsic Data Validation ===")
    print(f"Overall status: {summary['overall_status']}")
    print(f"Summary JSON: {cfg.report_dir / 'validation_summary.json'}")
    print(f"Summary MD  : {cfg.report_dir / 'validation_summary.md'}")
    print(f"Plots dir   : {cfg.plot_dir}")

    if args.strict and summary["overall_status"] in {"warn", "fail"}:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

