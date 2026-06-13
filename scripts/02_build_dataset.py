#!/usr/bin/env python3
"""
Build ML-ready training dataset from a GW population HDF5 file.

Supports two data sources (--data-source flag):
  zenodo  (default) : Zevin et al. Zenodo data  (chi_b × alpha_CE grid)
  sspc              : SSPC-based data from 00_sspc_data_generation.py
                      (sfr_a × mu0 grid, keys /CH/sfra{NNNN}/mu0{MMMM})

Outputs (under ``data/`` by default):
- data/hyperparam_table.csv          : Λ → ε table (channel, n_systems, …). Zenodo: chi_b, alpha_CE. SSPC: sfra, mu0.
- data/hyperparam_table_encoded.csv  : Zenodo: 5-ch one-hot + chi_b/alpha_CE norms + lambda (≥7 dims).
                                       SSPC: 3-ch one-hot + sfra/mu0 norms + 7 nuisance norms → lambda_0..11 (12 dims).
- data/all_events.parquet           : Intrinsic merger-rate samples (N_sample per grid point); main input for 04/04b
- data/all_detected_events.parquet  : Optional detection-subsampled table (N_det per grid) for other analyses
- data/splits.json                   : Train/val/test grid point indices (stratified by channel)
- {out_dir}/checkpoints/obs_normalizer.json : Observable normalizer (self-contained under custom ``--out-dir``)
- checkpoints/obs_normalizer.json           : Same file when ``--out-dir`` is the default ``data/``
"""

from __future__ import annotations

import sys
from pathlib import Path as _Path

_PROJECT_ROOT = _Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from plant_paths import (  # noqa: E402
    ALL_DETECTED_EVENTS_PARQUET,
    ALL_EVENTS_PARQUET,
    HYPERPARAM_TABLE_CSV,
    HYPERPARAM_TABLE_ENCODED_CSV,
    OBS_NORMALIZER_JSON,
    PROJECT_ROOT,
    REPO_ROOT,
    SPLITS_JSON,
    ensure_paths,
    find_data_dir,
    ML_DATA_DIR,
    ml_data_dir,
)

ensure_paths()

import argparse
import json
import multiprocessing as mp
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Populated in the parent before fork-based worker pools (Linux/HPC).
_BUILD_POOL: Dict[str, Any] = {}

# =============================================================================
# CONFIGURABLE PATHS (all paths defined here)
# =============================================================================


HDF5_PATH = find_data_dir() / "models_reduced.hdf5"
OUTPUT_DIR = ML_DATA_DIR

# Sampling parameters
N_SAMPLE = 5000   # events per grid point (weighted)
N_DET = 2000      # events per grid point (detection-weighted)
PDET_THRESHOLD = 0.01  # for n_detectable count


def default_normalizer_path(out_dir: Path) -> Path:
    """Project ``checkpoints/`` for default ``data/``; else under the dataset directory."""
    if out_dir.resolve() == ML_DATA_DIR.resolve():
        return OBS_NORMALIZER_JSON
    return out_dir / "checkpoints" / "obs_normalizer.json"

# ── Zenodo (Zevin) normalization ranges ──────────────────────────────────────
CHI_B_RANGE    = (0.0,  0.5)
ALPHA_CE_RANGE = (0.2,  5.0)

# Comoving volume for SSPC intrinsic rate density (matches 00_sspc_data_generation.py / Planck18)
SSPC_MAX_REDSHIFT = 10.0


def sspc_comoving_volume_gpc3(z_max: float = SSPC_MAX_REDSHIFT) -> float:
    """Comoving volume [Gpc³] out to *z_max* (same cosmology as Step 00)."""
    import astropy.units as u
    from astropy.cosmology import Planck18

    return float(Planck18.comoving_volume(z_max).to(u.Gpc**3).value)


# ── SSPC normalization ranges (sfr_a × mu0 grid) ─────────────────────────────
# Must match scripts/sspc_param_ranges.py / 00_sspc_data_generation.py.
from sspc_param_ranges import MU0_RANGE as SSPC_MU0_RANGE  # noqa: E402
from sspc_param_ranges import SFRA_RANGE as SSPC_SFRA_RANGE  # noqa: E402
# Minimum z for storage / log10(z); must match 00_sspc_data_generation.Z_LOG_FLOOR
Z_LOG_FLOOR = 1e-6


def configure_worker_threads(workers: int) -> int:
    """Set threaded-kernel worker counts for CPU-heavy parquet/numpy operations."""
    w = max(1, int(workers))
    os.environ["OMP_NUM_THREADS"] = str(w)
    os.environ["MKL_NUM_THREADS"] = str(w)
    os.environ["OPENBLAS_NUM_THREADS"] = str(w)
    os.environ["NUMEXPR_NUM_THREADS"] = str(w)
    return w


def _worker_init() -> None:
    """One BLAS thread per process when using a process pool."""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

# Zenodo (Zevin): five formation channels
CHANNEL_TO_ID = {"CE": 0, "CHE": 1, "GC": 2, "NSC": 3, "SMT": 4}
CHANNEL_NAMES = {0: "CE", 1: "CHE", 2: "GC", 3: "NSC", 4: "SMT"}

# SSPC (00_sspc_data_generation): three channels only, same order as 00 CHANNEL_NAMES
SSPC_CHANNEL_ORDER = ("CE", "CHE", "SMT")
SSPC_CHANNEL_TO_ID = {"CE": 0, "CHE": 1, "SMT": 2}

# Zenodo legacy key maps
CHI_MAP   = {"chi00": 0.0, "chi01": 0.1, "chi02": 0.2, "chi05": 0.5}
ALPHA_MAP = {"alpha02": 0.2, "alpha05": 0.5, "alpha10": 1.0, "alpha20": 2.0, "alpha50": 5.0}

# SSPC column names written by 00_sspc_data_generation.py
SSPC_PARAM_COLS = [
    "sspc_sfr_a",
    "sspc_sfr_b",
    "sspc_sfr_c",
    "sspc_sfr_d",
    "sspc_mu0",
    "sspc_muz",
    "sspc_sigma0",
    "sspc_sigmaz",
    "sspc_alpha_skew",
]

# Per-grid means for sfr_a / mu0 duplicate lambda_3 / lambda_4 on SSPC (same physics as sfra, mu0 columns).
SSPC_GRID_MEAN_COLS = frozenset({"sspc_sfr_a_mean", "sspc_mu0_mean"})


def _parse_param_key(token: str, prefix: str, legacy_map: Dict[str, float],
                     scale: int = 1000) -> float | None:
    """
    Parse a parameter token, supporting legacy short keys and dense integer keys.

    Examples
    --------
    chi00     → 0.0  (Zenodo legacy)
    chi0400   → 0.4  (Zenodo dense, scale=1000)
    sfra0200  → 0.02 (SSPC, scale=10000)
    mu00350   → 0.035 (SSPC, scale=10000)
    """
    if token in legacy_map:
        return legacy_map[token]
    m = re.fullmatch(rf"{prefix}(-?\d+)", token)
    if m:
        return int(m.group(1)) / float(scale)
    return None


def _detect_data_source(hdf5_path: Path) -> str:
    """Auto-detect data source by inspecting key names in the HDF5 file."""
    with pd.HDFStore(str(hdf5_path), "r") as store:
        for key in store.keys():
            parts = key.strip("/").split("/")
            if len(parts) >= 2 and parts[1].startswith("sfra"):
                return "sspc"
    return "zenodo"


def iter_grid_keys(hdf5_path: Path, data_source: str = "auto"):
    """
    Yield (key, channel, channel_id, param1, param2) for each grid point.

    For Zenodo data: param1 = chi_b, param2 = alpha_CE (NaN for non-CE).
    For SSPC data:   param1 = sfr_a,  param2 = mu0 (present for all channels).
    """
    if data_source == "auto":
        data_source = _detect_data_source(hdf5_path)

    store = pd.HDFStore(str(hdf5_path), "r")
    try:
        for key in store.keys():
            parts = key.strip("/").split("/")
            if len(parts) < 2:
                continue
            channel = parts[0]
            if data_source == "sspc":
                if channel not in SSPC_CHANNEL_TO_ID:
                    continue
                ch_id = SSPC_CHANNEL_TO_ID[channel]
                if len(parts) < 3:
                    continue
                p1 = _parse_param_key(parts[1], "sfra", {}, scale=10_000)
                p2 = _parse_param_key(parts[2], "mu0", {}, scale=10_000)
                if p1 is None or p2 is None:
                    continue
                yield key, channel, ch_id, p1, p2

            else:  # zenodo
                if channel not in CHANNEL_TO_ID:
                    continue
                ch_id = CHANNEL_TO_ID[channel]
                p1 = _parse_param_key(parts[1], "chi", CHI_MAP, scale=1000)
                if p1 is None:
                    continue
                if channel == "CE" and len(parts) >= 3:
                    p2 = _parse_param_key(parts[2], "alpha", ALPHA_MAP, scale=1000)
                    if p2 is None:
                        p2 = 1.0
                else:
                    p2 = np.nan
                yield key, channel, ch_id, p1, p2
    finally:
        store.close()


def _hyperparam_row_from_key(
    hdf5_path: Path,
    data_source: str,
    key: str,
    channel: str,
    ch_id: int,
    p1: float,
    p2: float,
    *,
    v_gpc3_sspc: float | None = None,
) -> Dict[str, object]:
    """Read one HDF5 grid group and return a hyperparam_table row."""
    df = pd.read_hdf(hdf5_path, key=key)
    n_systems = len(df)
    weight = df["weight"].values if "weight" in df.columns else np.ones(n_systems)
    pdet_col = "pdet_midhighlatelow_network"
    pdet = df[pdet_col].values if pdet_col in df.columns else np.ones(n_systems)

    w_arr = np.asarray(weight, dtype=np.float64)
    sum_weight = float(np.sum(w_arr))
    sum_weight_sq = float(np.sum(w_arr * w_arr))
    sum_pdet_raw = float(np.sum(pdet))
    n_detectable = int(np.sum(pdet > PDET_THRESHOLD))
    w_eff = np.asarray(weight * pdet, dtype=np.float64)
    w_eff_sum = float(np.sum(w_eff))

    if data_source == "sspc":
        sum_pdet = sum_weight
    else:
        sum_pdet = sum_pdet_raw

    ratio = sum_pdet / n_systems if n_systems > 0 else 0.0
    log_efficiency = np.log10(ratio) if ratio > 0 else -999.0

    row_data: Dict[str, object] = {
        "key": key,
        "channel": channel,
        "n_systems": n_systems,
        "sum_weight": sum_weight,
        "sum_weight_sq": sum_weight_sq,
        "sum_pdet": sum_pdet,
        "log_efficiency": log_efficiency,
        "n_detectable": n_detectable,
    }
    if data_source == "sspc":
        row_data["sfra"] = p1
        row_data["mu0"] = p2
        if n_systems > 0 and sum_weight > 0.0:
            intrinsic_rate_yr = n_systems * sum_weight_sq / sum_weight
        else:
            intrinsic_rate_yr = 0.0
        v_gpc3 = float(v_gpc3_sspc if v_gpc3_sspc is not None else sspc_comoving_volume_gpc3())
        row_data["intrinsic_rate_yr"] = intrinsic_rate_yr
        row_data["rate_per_gpc3_yr"] = (
            intrinsic_rate_yr / v_gpc3 if v_gpc3 > 0.0 else 0.0
        )
    else:
        row_data["chi_b"] = p1
        row_data["alpha_CE"] = p2

    w_sspc = np.asarray(weight, dtype=np.float64)
    w_sspc_sum = float(np.sum(w_sspc))
    for col in SSPC_PARAM_COLS:
        if col in df.columns:
            x = df[col].values.astype(np.float64)
            if data_source == "sspc":
                w_use, wsum = w_sspc, w_sspc_sum
            else:
                w_use, wsum = w_eff, w_eff_sum
            if wsum > 0:
                mean_x = float(np.sum(w_use * x) / wsum)
                var_x = float(np.sum(w_use * (x - mean_x) ** 2) / wsum)
                std_x = float(np.sqrt(max(var_x, 0.0)))
            else:
                mean_x = float(np.mean(x))
                std_x = float(np.std(x))
            row_data[f"{col}_mean"] = mean_x
            row_data[f"{col}_std"] = std_x

    return row_data


def _hyperparam_task(task_idx: int) -> Tuple[int, Dict[str, object]]:
    meta = _BUILD_POOL["grid_meta"][task_idx]
    key, channel, ch_id, p1, p2 = meta
    row = _hyperparam_row_from_key(
        _BUILD_POOL["hdf5_path"],
        _BUILD_POOL["data_source"],
        key,
        channel,
        int(ch_id),
        float(p1),
        float(p2),
        v_gpc3_sspc=_BUILD_POOL.get("v_gpc3_sspc"),
    )
    return task_idx, row


def build_hyperparam_table(
    hdf5_path: Path,
    data_source: str = "auto",
    *,
    workers: int = 1,
) -> pd.DataFrame:
    """Build hyperparam_table.csv with per-grid-point stats."""
    global _BUILD_POOL
    if data_source == "auto":
        data_source = _detect_data_source(hdf5_path)

    grid_meta = list(iter_grid_keys(hdf5_path, data_source))
    n = len(grid_meta)
    if n == 0:
        return pd.DataFrame()

    rows: List[Dict[str, object] | None] = [None] * n
    workers = max(1, int(workers))

    if workers <= 1:
        v_gpc3 = sspc_comoving_volume_gpc3() if data_source == "sspc" else None
        for i, (key, channel, ch_id, p1, p2) in enumerate(grid_meta):
            rows[i] = _hyperparam_row_from_key(
                hdf5_path, data_source, key, channel, ch_id, p1, p2, v_gpc3_sspc=v_gpc3,
            )
            if (i + 1) % max(1, n // 20) == 0 or i + 1 == n:
                print(f"   [hyperparam] {100.0 * (i + 1) / n:5.1f}%", flush=True)
    else:
        _BUILD_POOL = {
            "hdf5_path": hdf5_path,
            "data_source": data_source,
            "grid_meta": grid_meta,
            "v_gpc3_sspc": sspc_comoving_volume_gpc3() if data_source == "sspc" else None,
        }
        print(f"   [hyperparam] {workers} worker processes, {n} grid keys", flush=True)

        done = 0
        ctx = mp.get_context("fork")
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=ctx,
            initializer=_worker_init,
        ) as ex:
            futures = [ex.submit(_hyperparam_task, i) for i in range(n)]
            for fut in as_completed(futures):
                idx, row = fut.result()
                rows[idx] = row
                done += 1
                if done % max(1, n // 20) == 0 or done == n:
                    print(
                        f"   [hyperparam] {100.0 * done / n:5.1f}% ({workers} workers)",
                        flush=True,
                    )

    return pd.DataFrame([r for r in rows if r is not None])


def encode_hyperparams(df: pd.DataFrame, data_source: str = "zenodo") -> pd.DataFrame:
    """
    Add channel_id, channel_onehot_*, primary norms, and lambda_*.

    Zenodo: 5-dim channel one-hot + chi_b_norm + alpha_CE_norm → lambda_0..lambda_6 (7),
            then optional SSPC nuisance blocks from columns if present.

    SSPC: 3-dim channel one-hot (CE, CHE, SMT) + sfra_norm + mu0_norm → lambda_0..lambda_4 (5),
          then 7 min–max nuisance means → lambda_5..lambda_11 (12 total).
    """
    out = df.copy()

    if data_source == "sspc":
        out["channel_id"] = out["channel"].map(SSPC_CHANNEL_TO_ID)
        n_ch = len(SSPC_CHANNEL_ORDER)
        onehot = np.zeros((len(df), n_ch), dtype=np.float64)
        for i, ch in enumerate(out["channel"].astype(str)):
            onehot[i, SSPC_CHANNEL_TO_ID[ch]] = 1.0
        for i in range(n_ch):
            out[f"channel_onehot_{i}"] = onehot[:, i]

        p1 = out["sfra"].values.astype(float)
        p2 = out["mu0"].values.astype(float)
        p1_min, p1_max = SSPC_SFRA_RANGE
        p2_min, p2_max = SSPC_MU0_RANGE
        out["sfra_norm"] = (p1 - p1_min) / (p1_max - p1_min) if p1_max > p1_min else 0.0
        out["mu0_norm"] = (p2 - p2_min) / (p2_max - p2_min) if p2_max > p2_min else 0.0

        for j in range(n_ch):
            out[f"lambda_{j}"] = onehot[:, j]
        out["lambda_3"] = out["sfra_norm"]
        out["lambda_4"] = out["mu0_norm"]

        sspc_mean_cols = [f"{c}_mean" for c in SSPC_PARAM_COLS if f"{c}_mean" in out.columns]
        sspc_lambda_mean_cols = [c for c in sspc_mean_cols if c not in SSPC_GRID_MEAN_COLS]

        for col in sspc_lambda_mean_cols:
            cmin = float(out[col].min())
            cmax = float(out[col].max())
            out[f"{col}_norm"] = (out[col] - cmin) / (cmax - cmin) if cmax > cmin else 0.0

        next_lambda_idx = 5
        for col in sspc_lambda_mean_cols:
            norm_col = f"{col}_norm"
            out[f"lambda_{next_lambda_idx}"] = out[norm_col]
            next_lambda_idx += 1
        return out

    # ── Zenodo (5 channels, chi_b × alpha_CE on CE) ─────────────────────────
    out["channel_id"] = out["channel"].map(CHANNEL_TO_ID)
    onehot = np.zeros((len(df), 5), dtype=np.float64)
    for i, ch_id in enumerate(out["channel_id"]):
        onehot[i, int(ch_id)] = 1.0
    for i in range(5):
        out[f"channel_onehot_{i}"] = onehot[:, i]

    p1 = out["chi_b"].values.astype(float)
    p1_min = float(np.nanmin(p1)) if not np.all(np.isnan(p1)) else CHI_B_RANGE[0]
    p1_max = float(np.nanmax(p1)) if not np.all(np.isnan(p1)) else CHI_B_RANGE[1]
    out["chi_b_norm"] = (p1 - p1_min) / (p1_max - p1_min) if p1_max > p1_min else 0.0

    p2 = out["alpha_CE"].values.astype(float)
    ce_alpha = out.loc[out["channel"] == "CE", "alpha_CE"].dropna()
    alpha_min = float(ce_alpha.min()) if len(ce_alpha) > 0 else ALPHA_CE_RANGE[0]
    alpha_max = float(ce_alpha.max()) if len(ce_alpha) > 0 else ALPHA_CE_RANGE[1]
    alpha_ce_filled = pd.Series(p2).fillna(0.0).values
    out["alpha_CE_norm"] = np.where(
        out["channel"] == "CE",
        (alpha_ce_filled - alpha_min) / (alpha_max - alpha_min),
        0.0,
    )

    for j in range(7):
        if j < 5:
            out[f"lambda_{j}"] = onehot[:, j]
        elif j == 5:
            out["lambda_5"] = out["chi_b_norm"]
        else:
            out["lambda_6"] = out["alpha_CE_norm"]

    sspc_mean_cols = [f"{c}_mean" for c in SSPC_PARAM_COLS if f"{c}_mean" in out.columns]
    sspc_lambda_mean_cols = list(sspc_mean_cols)

    for col in sspc_lambda_mean_cols:
        cmin = float(out[col].min())
        cmax = float(out[col].max())
        out[f"{col}_norm"] = (out[col] - cmin) / (cmax - cmin) if cmax > cmin else 0.0

    next_lambda_idx = 7
    for col in sspc_lambda_mean_cols:
        norm_col = f"{col}_norm"
        out[f"lambda_{next_lambda_idx}"] = out[norm_col]
        next_lambda_idx += 1
    return out


def sample_events_for_grid(
    df: pd.DataFrame,
    grid_idx: int,
    channel_id: int,
    p1_norm: float,
    p2_norm: float,
    p1_norm_col: str,
    p2_norm_col: str,
    lambda_vec: List[float],
    n: int,
    use_detection_weight: bool,
    *,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Sample n events from df with optional detection weighting."""
    n_rows = len(df)
    if n_rows == 0:
        return pd.DataFrame()

    weight = df["weight"].values if "weight" in df.columns else np.ones(n_rows)
    pdet_col = "pdet_midhighlatelow_network"
    pdet = df[pdet_col].values if pdet_col in df.columns else np.ones(n_rows)

    if use_detection_weight:
        probs = weight * pdet
    else:
        probs = weight

    probs_sum = np.sum(probs)
    if probs_sum <= 0:
        probs = np.ones(n_rows) / n_rows
    else:
        probs = probs / probs_sum

    if rng is None:
        idx = np.random.choice(n_rows, size=n, replace=True, p=probs)
    else:
        idx = rng.choice(n_rows, size=n, replace=True, p=probs)
    sampled = df.iloc[idx]

    # Build output row
    result = pd.DataFrame({
        "mchirp": sampled["mchirp"].values,
        "q": sampled["q"].values,
        "z": np.maximum(sampled["z"].values, Z_LOG_FLOOR),
        "pdet": sampled[pdet_col].values if pdet_col in sampled.columns else np.ones(n),
        "grid_idx": grid_idx,
        "channel_id": channel_id,
        p1_norm_col: p1_norm,
        p2_norm_col: p2_norm,
    })
    for j, v in enumerate(lambda_vec):
        result[f"lambda_{j}"] = v
    for col in SSPC_PARAM_COLS:
        if col in sampled.columns:
            result[col] = sampled[col].values
    return result


def _grid_sample_seed(base_seed: int, grid_idx: int, *, det: bool) -> int:
    """Stable per-grid RNG seed (independent of worker scheduling order)."""
    off = 1 if det else 0
    return int(base_seed) + int(grid_idx) * 10007 + off


def _process_one_grid_events(grid_idx: int) -> Tuple[int, pd.DataFrame, pd.DataFrame]:
    """Read one HDF5 group and return intrinsic + detection-weighted samples."""
    pool = _BUILD_POOL
    row = pool["encoded_rows"][grid_idx]
    hdf5_path: Path = pool["hdf5_path"]
    n_sample = int(pool["n_sample"])
    n_det = int(pool["n_det"])
    sample_seed = int(pool["sample_seed"])
    lambda_cols: List[str] = pool["lambda_cols"]
    p1c: str = pool["p1c"]
    p2c: str = pool["p2c"]

    key = row["key"]
    lambda_vec = [float(row[c]) for c in lambda_cols]
    df = pd.read_hdf(hdf5_path, key=key)

    ev = sample_events_for_grid(
        df,
        grid_idx,
        int(row["channel_id"]),
        float(row[p1c]),
        float(row[p2c]),
        p1c,
        p2c,
        lambda_vec,
        n_sample,
        use_detection_weight=False,
        rng=np.random.default_rng(_grid_sample_seed(sample_seed, grid_idx, det=False)),
    )
    ev_det = sample_events_for_grid(
        df,
        grid_idx,
        int(row["channel_id"]),
        float(row[p1c]),
        float(row[p2c]),
        p1c,
        p2c,
        lambda_vec,
        n_det,
        use_detection_weight=True,
        rng=np.random.default_rng(_grid_sample_seed(sample_seed, grid_idx, det=True)),
    )
    return grid_idx, ev, ev_det


def build_event_datasets(
    encoded_df: pd.DataFrame,
    hdf5_path: Path,
    n_sample: int,
    n_det: int,
    data_source: str,
    *,
    workers: int = 1,
    sample_seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build all_events.parquet and all_detected_events.parquet."""
    global _BUILD_POOL

    encoded_df = encoded_df.reset_index(drop=True)
    n = len(encoded_df)
    if n == 0:
        return pd.DataFrame(), pd.DataFrame()

    lambda_cols = sorted(
        [c for c in encoded_df.columns if c.startswith("lambda_")],
        key=lambda x: int(x.split("_")[1]),
    )
    if data_source == "sspc":
        p1c, p2c = "sfra_norm", "mu0_norm"
    else:
        p1c, p2c = "chi_b_norm", "alpha_CE_norm"

    workers = max(1, int(workers))
    all_events_parts: List[pd.DataFrame | None] = [None] * n
    all_detected_parts: List[pd.DataFrame | None] = [None] * n

    if workers <= 1:
        for grid_idx in range(n):
            row = encoded_df.iloc[grid_idx]
            key = row["key"]
            lambda_vec = [float(row[c]) for c in lambda_cols]
            df = pd.read_hdf(hdf5_path, key=key)
            ev = sample_events_for_grid(
                df,
                grid_idx,
                int(row["channel_id"]),
                float(row[p1c]),
                float(row[p2c]),
                p1c,
                p2c,
                lambda_vec,
                n_sample,
                use_detection_weight=False,
                rng=np.random.default_rng(
                    _grid_sample_seed(sample_seed, grid_idx, det=False)
                ),
            )
            ev_det = sample_events_for_grid(
                df,
                grid_idx,
                int(row["channel_id"]),
                float(row[p1c]),
                float(row[p2c]),
                p1c,
                p2c,
                lambda_vec,
                n_det,
                use_detection_weight=True,
                rng=np.random.default_rng(
                    _grid_sample_seed(sample_seed, grid_idx, det=True)
                ),
            )
            if len(ev) > 0:
                all_events_parts[grid_idx] = ev
            if len(ev_det) > 0:
                all_detected_parts[grid_idx] = ev_det
            if (grid_idx + 1) % max(1, n // 20) == 0 or grid_idx + 1 == n:
                print(f"   [events] {100.0 * (grid_idx + 1) / n:5.1f}%", flush=True)
    else:
        _BUILD_POOL = {
            "hdf5_path": hdf5_path,
            "encoded_rows": encoded_df.to_dict("records"),
            "n_sample": n_sample,
            "n_det": n_det,
            "sample_seed": sample_seed,
            "lambda_cols": lambda_cols,
            "p1c": p1c,
            "p2c": p2c,
        }
        print(f"   [events] {workers} worker processes, {n} grid keys", flush=True)
        done = 0
        ctx = mp.get_context("fork")
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=ctx,
            initializer=_worker_init,
        ) as ex:
            futures = [ex.submit(_process_one_grid_events, i) for i in range(n)]
            for fut in as_completed(futures):
                grid_idx, ev, ev_det = fut.result()
                if len(ev) > 0:
                    all_events_parts[grid_idx] = ev
                if len(ev_det) > 0:
                    all_detected_parts[grid_idx] = ev_det
                done += 1
                if done % max(1, n // 20) == 0 or done == n:
                    print(
                        f"   [events] {100.0 * done / n:5.1f}% ({workers} workers)",
                        flush=True,
                    )

    all_events = [df for df in all_events_parts if df is not None]
    all_detected = [df for df in all_detected_parts if df is not None]
    all_events_df = pd.concat(all_events, ignore_index=True) if all_events else pd.DataFrame()
    all_detected_df = (
        pd.concat(all_detected, ignore_index=True) if all_detected else pd.DataFrame()
    )
    return all_events_df, all_detected_df


def stratified_split(encoded_df: pd.DataFrame) -> Dict[str, List[int]]:
    """Split grid points 70/15/15 stratified by channel. Returns indices."""
    indices = np.arange(len(encoded_df))
    channels = np.asarray(encoded_df["channel"].values)

    # First split: 70% train, 30% temp
    idx_train, idx_temp = train_test_split(
        indices, test_size=0.3, stratify=channels, random_state=42
    )
    ch_temp = channels[idx_temp]
    # Second split: 50% of 30% = 15% val, 15% test
    # Stratify if possible; else fall back to random (small datasets)
    try:
        idx_val_rel, idx_test_rel = train_test_split(
            np.arange(len(idx_temp)),
            test_size=0.5,
            stratify=ch_temp,
            random_state=42,
        )
    except ValueError:
        idx_val_rel, idx_test_rel = train_test_split(
            np.arange(len(idx_temp)),
            test_size=0.5,
            random_state=42,
        )
    idx_val = idx_temp[idx_val_rel]
    idx_test = idx_temp[idx_test_rel]

    return {
        "train": [int(i) for i in idx_train],
        "val": [int(i) for i in idx_val],
        "test": [int(i) for i in idx_test],
    }


def compute_and_save_obs_normalizer(events_df: pd.DataFrame, out_path: Path) -> Dict:
    """
    Compute normalization from the intrinsic (merger-rate–weighted) event table,
    so the same stats apply to 04/04b training on `all_events.parquet`.

    mchirp, z: log10 transform FIRST, then mean/std of log values
    q: mean/std on raw values
    """
    cols = ["mchirp", "q", "z"]
    normalizer = {}
    eps = 1e-8

    for col in cols:
        x = events_df[col].values.astype(np.float64)
        if col == "mchirp":
            x_log = np.log10(np.maximum(x, 1e-3))
            mean_val = float(np.mean(x_log))
            std_val = float(np.std(x_log) + eps)
            normalizer[col] = {"mean": mean_val, "std": std_val}
        elif col == "z":
            # Floor z for log10 (z=0 → log10(0)=-inf); keep floor consistent with 00 / CFM.
            x_log = np.log10(np.maximum(x, Z_LOG_FLOOR))
            mean_val = float(np.mean(x_log))
            std_val = float(np.std(x_log) + eps)
            normalizer[col] = {"mean": mean_val, "std": std_val}
        else:
            mean_val = float(np.mean(x))
            std_val = float(np.std(x) + eps)
            normalizer[col] = {"mean": mean_val, "std": std_val}

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(normalizer, f, indent=2)

    print("   Observable normalization (from all_events.parquet):")
    print(f"   mchirp: mean_log_mchirp = {normalizer['mchirp']['mean']:.6f}, std_log_mchirp = {normalizer['mchirp']['std']:.6f}")
    print(f"   q: mean_q = {normalizer['q']['mean']:.6f}, std_q = {normalizer['q']['std']:.6f}")
    print(f"   z: mean_log_z = {normalizer['z']['mean']:.6f}, std_log_z = {normalizer['z']['std']:.6f}")
    print(f"   Saved: {out_path}")

    return normalizer


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build ML dataset from a GW population HDF5 file."
    )
    parser.add_argument("--hdf5", type=Path, default=HDF5_PATH)
    parser.add_argument("--out-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--n-sample", type=int, default=N_SAMPLE)
    parser.add_argument("--n-det", type=int, default=N_DET)
    parser.add_argument(
        "--workers",
        type=int,
        default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")),
        help="Parallel worker processes over grid keys (default: SLURM_CPUS_PER_TASK or 1).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--data-source", type=str, default="auto",
        choices=["auto", "zenodo", "sspc"],
        help="Key format in the HDF5: 'zenodo' (chi_b/alpha_CE) or 'sspc' (sfr_a/mu0). "
             "'auto' detects from the file (default).",
    )
    parser.add_argument(
        "--normalizer-out",
        type=Path,
        default=None,
        help=(
            "Path for obs_normalizer.json (default: checkpoints/ under --out-dir, "
            "or project checkpoints/ when --out-dir is data/)."
        ),
    )
    args = parser.parse_args()
    workers = max(1, int(args.workers))
    if workers > 1:
        _worker_init()
    else:
        configure_worker_threads(workers)
    print(f"   Parallel worker processes: {workers}", flush=True)

    np.random.seed(args.seed)
    data_source = args.data_source
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if data_source == "auto":
        data_source = _detect_data_source(args.hdf5)
        print(f"   Auto-detected data source: {data_source}")

    hyperparam_csv = out_dir / "hyperparam_table.csv"
    hyperparam_encoded_csv = out_dir / "hyperparam_table_encoded.csv"
    all_events_pq = out_dir / "all_events.parquet"
    all_detected_pq = out_dir / "all_detected_events.parquet"
    splits_json = out_dir / "splits.json"
    normalizer_path = (
        args.normalizer_out.resolve()
        if args.normalizer_out is not None
        else default_normalizer_path(out_dir)
    )

    print(f"   HDF5 input     : {args.hdf5.resolve()}")
    print(f"   Output dir     : {out_dir.resolve()}")
    print(f"   Normalizer out : {normalizer_path}")

    print("1. Building hyperparameter table...")
    hp_df = build_hyperparam_table(args.hdf5, data_source=data_source, workers=workers)
    hp_df.to_csv(hyperparam_csv, index=False)
    print(f"   Saved: {hyperparam_csv} ({len(hp_df)} rows)")

    print("2. Encoding hyperparameters...")
    encoded_df = encode_hyperparams(hp_df, data_source=data_source)
    encoded_df.to_csv(hyperparam_encoded_csv, index=False)
    print(f"   Saved: {hyperparam_encoded_csv}")
    if data_source == "sspc":
        preview_cols = ["channel", "sfra", "mu0", "channel_id", "sfra_norm", "mu0_norm"]
    else:
        preview_cols = ["channel", "chi_b", "alpha_CE", "channel_id", "chi_b_norm", "alpha_CE_norm"]
    print(encoded_df[preview_cols].head(10).to_string())

    print("3. Building event datasets...")
    all_events_df, all_detected_df = build_event_datasets(
        encoded_df,
        args.hdf5,
        args.n_sample,
        args.n_det,
        data_source=data_source,
        workers=workers,
        sample_seed=int(args.seed),
    )
    all_events_df.to_parquet(all_events_pq, index=False)
    all_detected_df.to_parquet(all_detected_pq, index=False)
    print(f"   Saved: {all_events_pq} ({len(all_events_df)} rows)")
    print(f"   Saved: {all_detected_pq} ({len(all_detected_df)} rows)")

    print("4. Building stratified splits...")
    splits = stratified_split(encoded_df)
    with open(splits_json, "w") as f:
        json.dump(splits, f, indent=2)
    print(f"   Saved: {splits_json}")

    print("5. Computing observable normalization (from intrinsic all_events rows)...")
    normalizer_path.parent.mkdir(parents=True, exist_ok=True)
    compute_and_save_obs_normalizer(all_events_df, normalizer_path)
    print(f"   Saved: {normalizer_path}")

    print("Done.")


if __name__ == "__main__":
    main()
