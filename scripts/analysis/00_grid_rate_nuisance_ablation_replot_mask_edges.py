#!/usr/bin/env python3
"""
Re-plot nuisance ablation PNGs from saved *_table.csv.gz outputs, while masking:
  - the leftmost mu0 column (mu0 == min(mu0))
  - the bottommost sfr_a row (sfra == min(sfra_a))

This is useful when edge cells contain extreme values that dominate the color scale.

Example:
  python scripts/analysis/00_grid_rate_nuisance_ablation_replot_mask_edges.py \
    --array-dir plots/00_grid_rate_nuisance_ablation/array_49820077
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

_ANALYSIS_DIR = Path(__file__).resolve().parent
if str(_ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(_ANALYSIS_DIR))

from _bootstrap import setup  # noqa: E402

setup()

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def _load_ablation_module() -> Any:
    path = _ANALYSIS_DIR / "00_grid_rate_nuisance_ablation.py"
    spec = importlib.util.spec_from_file_location("_plant_grid_rate_ablation", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _mask_edges_inplace(hp: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with min-mu0 column and min-sfra row set to NaN."""
    out = hp.copy()
    mu0_min = float(out["mu0"].min())
    sfra_min = float(out["sfra"].min())
    edge = (out["mu0"].astype(float) == mu0_min) | (out["sfra"].astype(float) == sfra_min)
    if "rate_per_gpc3_yr" in out.columns:
        out.loc[edge, "rate_per_gpc3_yr"] = np.nan
    if "intrinsic_rate_yr" in out.columns:
        out.loc[edge, "intrinsic_rate_yr"] = np.nan
    return out


def _nuisance_range_from_json(meta_path: Path) -> Tuple[float, float]:
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if "nuisance_range" in meta and isinstance(meta["nuisance_range"], list):
        lo, hi = meta["nuisance_range"]
        return float(lo), float(hi)
    return 0.0, 0.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Replot ablation PNGs masking edge cells.")
    p.add_argument(
        "--array-dir",
        type=Path,
        required=True,
        help="Directory containing grid_rate_ablation_*_table.csv.gz files.",
    )
    p.add_argument(
        "--out-subdir",
        type=str,
        default="masked_edges",
        help="Subdirectory under array-dir for outputs (default: masked_edges).",
    )
    p.add_argument(
        "--no-tex",
        action="store_true",
        help="Disable LaTeX rendering (recommended on compute nodes).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    array_dir = args.array_dir.resolve()
    if not array_dir.exists():
        raise FileNotFoundError(array_dir)

    ab = _load_ablation_module()
    hm_mod = ab._load_heatmaps_module()

    out_dir = array_dir / args.out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    tables = sorted(array_dir.glob("grid_rate_ablation_*_table.csv.gz"))
    if not tables:
        raise FileNotFoundError(f"No *_table.csv.gz found under {array_dir}")

    for table_path in tables:
        stem = table_path.name.replace("_table.csv.gz", "")
        # e.g. grid_rate_ablation_sfr_b
        target = stem.replace("grid_rate_ablation_", "")
        meta_path = array_dir / f"{stem}.json"
        nuisance_range = _nuisance_range_from_json(meta_path) if meta_path.exists() else (0.0, 0.0)

        hp = pd.read_csv(table_path)
        hp_masked = _mask_edges_inplace(hp)

        sfra_centers = np.sort(hp_masked["sfra"].astype(float).unique())
        mu0_centers = np.sort(hp_masked["mu0"].astype(float).unique())

        out_path = out_dir / f"{stem}_masked_edges.png"
        ab.plot_ablation_heatmaps(
            hp_masked,
            ablated=target,
            nuisance_range=(float(nuisance_range[0]), float(nuisance_range[1])),
            sfra_centers=sfra_centers,
            mu0_centers=mu0_centers,
            aggregate="mean",
            metric="rate",
            color_scale="log",
            cmap_style="sequential",
            out_path=out_path,
            hm_mod=hm_mod,
            use_tex=not bool(args.no_tex),
        )


if __name__ == "__main__":
    main()

