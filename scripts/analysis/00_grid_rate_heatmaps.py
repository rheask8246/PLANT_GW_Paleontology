#!/usr/bin/env python3
"""
Step 00 — Heatmaps of intrinsic merger rate density on the (sfr_a, mu0) grid.

Reads **Step 00 output directly**: ``data/sspc/models_sspc.hdf5`` from
``00_sspc_data_generation.py``. Each grid cell stores ``intrinsic_rate_yr`` (total
intrinsic merger rate summed over **all** BPS binaries before event sampling) and
``rate_per_gpc3_yr`` = ``intrinsic_rate_yr / V_comov(z ≤ 10)``.

This is **not** the Step 02 ``hyperparam_table.csv`` (that table sums weights over the
*sampled* merger rows used for ML). Use ``--hyperparam-csv`` only for legacy comparisons.

Default color scale: ``rate_per_gpc3_yr`` [Gpc⁻³ yr⁻¹]. Also ``count`` (sampled rows per cell).

Color mapping (global across SMT/CE/CHE): ``--color-scale log`` (default) or
``linear`` / ``--linear-scale``. Colormap: ``--colormap sequential`` (default) or
``diverging`` (two hues with a light neutral midpoint, e.g. ``RdBu_r``).

Reduce nuisance speckle: ``--average-over mu0`` or ``--average-over sfra`` (mean over
that axis, broadcast for display).

By default the minimum ``sfr_a`` row and minimum ``mu0`` column are masked (legacy
edge bins). Use ``--no-mask-edges`` to show the full grid.

``--mark-fiducial-study`` overlays five comparison points (SMT/CE only): fiducial,
μ₀ low/high at a_SF=0.02, and a_SF low/high at μ₀=0.025.

SLURM: ``slurm/00_grid_rate_heatmaps.sh``
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
import warnings
from pathlib import Path
from typing import Literal, Tuple

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

from lib.grid_heatmap_plot import (  # noqa: E402
    FIDUCIAL_STUDY_CHANNELS,
    FIDUCIAL_STUDY_MARKS,
    mask_grid_z_edges,
    overlay_fiducial_study_marks,
    pcolormesh_sfra_mu0,
)
from plant_paths import (  # noqa: E402
    PROJECT_ROOT,
    find_data_dir,
    resolve_plot_output,
)
from sspc_param_ranges import MU0_RANGE, SFRA_RANGE  # noqa: E402

CHANNELS = ("SMT", "CE", "CHE")
MetricName = Literal["rate", "count", "log_rate", "rate_weight"]
ColorScaleMode = Literal["log", "linear"]
ColormapStyle = Literal["sequential", "diverging"]
AverageOver = Literal["none", "mu0", "sfra"]
_SSPC_CHANNELS = frozenset(CHANNELS)
_GRID_ROUND_DECIMALS = 8
_SSPC_Z_MAX = 10.0


def _heatmap_cmap(cmap_style: ColormapStyle) -> str:
    """Sequential (dark→bright) or two-color diverging (hue–white/light–hue)."""
    if cmap_style == "diverging":
        # Red–white–blue: distinct hues meeting at a pale neutral midpoint.
        return "RdBu_r"
    try:
        import seaborn  # noqa: F401

        return "rocket"
    except ImportError:
        return "inferno"


def _comoving_volume_gpc3(z_max: float = _SSPC_Z_MAX) -> float:
    import astropy.units as u
    from astropy.cosmology import Planck18

    return float(Planck18.comoving_volume(z_max).to(u.Gpc**3).value)


def _matplotlib_usetex_works() -> bool:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    prev = bool(plt.rcParams["text.usetex"])
    try:
        plt.rcParams["text.usetex"] = True
        fig = Figure(figsize=(0.6, 0.6))
        canvas = FigureCanvasAgg(fig)
        ax = fig.subplots()
        ax.axis("off")
        ax.text(0.5, 0.5, r"$\mathrm{Gpc}^{-3}$", usetex=True, ha="center", va="center")
        canvas.draw()
        return True
    except Exception:
        return False
    finally:
        plt.rcParams["text.usetex"] = prev


def _configure_matplotlib(*, use_tex: bool) -> bool:
    """Serif + LaTeX labels when available."""
    requested = bool(use_tex)
    tex_ok = requested and _matplotlib_usetex_works()
    plt.rc("font", family="serif")
    plt.rc("text", usetex=tex_ok)
    if requested and not tex_ok:
        matplotlib.rcParams["mathtext.fontset"] = "dejavuserif"
    return tex_ok


def _parse_param_token(token: str, prefix: str, *, scale: float = 10_000) -> float | None:
    m = re.fullmatch(rf"{prefix}(-?\d+)", token)
    if m:
        return int(m.group(1)) / scale
    return None


def _grid_coords_from_key(key: str) -> Tuple[float, float] | None:
    parts = str(key).strip("/").split("/")
    if len(parts) < 3:
        return None
    p1 = _parse_param_token(parts[1], "sfra")
    p2 = _parse_param_token(parts[2], "mu0")
    if p1 is None or p2 is None:
        return None
    return p1, p2


def _table_uses_sspc_axes(df: pd.DataFrame) -> bool:
    if "key" in df.columns and df["key"].astype(str).str.contains("sfra", regex=False).any():
        return True
    if "channel" in df.columns:
        return bool(_SSPC_CHANNELS & set(df["channel"].astype(str).unique()))
    return False


def _round_grid_series(s: pd.Series) -> pd.Series:
    return s.astype(np.float64).round(_GRID_ROUND_DECIMALS)


def default_00_hdf5_path() -> Path:
    """``models_sspc.hdf5`` from ``00_sspc_data_generation.py``."""
    return find_data_dir() / "sspc" / "models_sspc.hdf5"


def hdf5_output_tag(hdf5_path: Path) -> str:
    """Short suffix for plot filenames when not using the default Step 00 HDF5."""
    resolved = hdf5_path.resolve()
    if resolved == default_00_hdf5_path().resolve():
        return ""
    stem = hdf5_path.stem
    if stem.startswith("models_sspc_"):
        return stem[len("models_sspc_") :]
    if stem != "models_sspc":
        return stem
    return "custom"


def _intrinsic_rate_yr_from_catalog(hp: pd.DataFrame) -> pd.Series:
    """Total intrinsic rate [yr⁻¹] per grid cell."""
    if "intrinsic_rate_yr" in hp.columns:
        return hp["intrinsic_rate_yr"].astype(np.float64)
    raise ValueError(
        "Table missing intrinsic_rate_yr. Read Step 00 HDF5 or re-run 00_sspc_data_generation.py."
    )


def add_rate_density_columns(hp: pd.DataFrame, *, v_gpc3: float | None = None) -> pd.DataFrame:
    """Add ``intrinsic_rate_yr`` and ``rate_per_gpc3_yr`` if not already present."""
    out = hp.copy()
    if v_gpc3 is None:
        v_gpc3 = _comoving_volume_gpc3()
    out["intrinsic_rate_yr"] = _intrinsic_rate_yr_from_catalog(out)
    if "rate_per_gpc3_yr" not in out.columns:
        out["rate_per_gpc3_yr"] = out["intrinsic_rate_yr"] / max(float(v_gpc3), 1e-30)
    return out


def normalize_hyperparam_grid(hp: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure ``sfra`` and ``mu0`` columns exist for heatmap axes.

    Accepts tables from ``02_build_dataset.py`` whether written with SSPC names
    (``sfra``, ``mu0``), grid columns mislabeled as Zenodo (``chi_b``, ``alpha_CE``),
    or weighted means (``sspc_sfr_a_mean``, …; rounded to avoid float duplicates).
    """
    out = hp.copy()

    if "sfra" not in out.columns or "mu0" not in out.columns:
        if {"chi_b", "alpha_CE"}.issubset(out.columns) and _table_uses_sspc_axes(out):
            out["sfra"] = out["chi_b"].astype(np.float64)
            out["mu0"] = out["alpha_CE"].astype(np.float64)
        elif {"sspc_sfr_a_mean", "sspc_mu0_mean"}.issubset(out.columns):
            out["sfra"] = out["sspc_sfr_a_mean"].astype(np.float64)
            out["mu0"] = out["sspc_mu0_mean"].astype(np.float64)
        elif "key" in out.columns:
            coords = out["key"].map(_grid_coords_from_key)
            if coords.isna().all():
                raise ValueError(
                    "Could not infer (sfr_a, mu0) grid coordinates. Expected columns "
                    "'sfra'/'mu0', 'sspc_sfr_a_mean'/'sspc_mu0_mean', SSPC-style "
                    "'chi_b'/'alpha_CE', or parseable 'key' paths (/CH/sfra…/mu0…)."
                )
            out["sfra"] = coords.map(lambda c: c[0] if c is not None else np.nan)
            out["mu0"] = coords.map(lambda c: c[1] if c is not None else np.nan)
        else:
            raise ValueError(
                "hyperparam table missing grid axes. Need 'sfra'/'mu0', "
                "'sspc_sfr_a_mean'/'sspc_mu0_mean', or SSPC 'key' column."
            )

    out["sfra"] = _round_grid_series(out["sfra"])
    out["mu0"] = _round_grid_series(out["mu0"])

    if out["sfra"].isna().any() or out["mu0"].isna().any():
        raise ValueError("Some rows have missing sfr_a or mu0 after grid-axis normalization.")

    if "sum_weight" not in out.columns and "sum_pdet" in out.columns:
        out["sum_weight"] = out["sum_pdet"]

    return add_rate_density_columns(out)


def _load_build_dataset_module():
    path = PROJECT_ROOT / "scripts" / "02_build_dataset.py"
    spec = importlib.util.spec_from_file_location("_plant_build_dataset", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_grid_table_from_00_hdf5(
    hdf5_path: Path,
    *,
    v_gpc3: float | None = None,
) -> pd.DataFrame:
    """One row per SSPC grid key using Step 00 ``intrinsic_rate_yr`` (pre-sampling)."""
    mod = _load_build_dataset_module()
    if v_gpc3 is None:
        v_gpc3 = _comoving_volume_gpc3()

    rows = []
    legacy_keys = 0
    for key, channel, _ch_id, p1, p2 in mod.iter_grid_keys(hdf5_path, "sspc"):
        df = pd.read_hdf(hdf5_path, key=key)
        n_systems = int(len(df))
        w = (
            df["weight"].values.astype(np.float64)
            if "weight" in df.columns
            else np.ones(n_systems, dtype=np.float64)
        )
        sum_weight = float(np.sum(w))

        if "intrinsic_rate_yr" in df.columns:
            intrinsic_rate_yr = float(df["intrinsic_rate_yr"].iloc[0])
        else:
            legacy_keys += 1
            if sum_weight > 0.0:
                intrinsic_rate_yr = n_systems * float(np.sum(w * w)) / sum_weight
            else:
                intrinsic_rate_yr = 0.0

        if "rate_per_gpc3_yr" in df.columns:
            rate_per_gpc3_yr = float(df["rate_per_gpc3_yr"].iloc[0])
        else:
            rate_per_gpc3_yr = intrinsic_rate_yr / max(float(v_gpc3), 1e-30)

        rows.append(
            {
                "key": key,
                "channel": channel,
                "sfra": float(p1),
                "mu0": float(p2),
                "n_systems": n_systems,
                "sum_weight": sum_weight,
                "intrinsic_rate_yr": intrinsic_rate_yr,
                "rate_per_gpc3_yr": rate_per_gpc3_yr,
            }
        )

    if legacy_keys:
        warnings.warn(
            f"{legacy_keys} HDF5 keys lack intrinsic_rate_yr (old Step 00 output). "
            "Catalog-weight estimate used; re-run 00_sspc_data_generation.py for exact rates.",
            stacklevel=2,
        )

    hp = pd.DataFrame(rows)
    if hp.empty:
        raise ValueError(f"No SSPC grid keys found in {hdf5_path}")
    return normalize_hyperparam_grid(hp)


def load_grid_table(
    *,
    sspc_hdf5: Path,
    hyperparam_csv: Path | None = None,
    v_gpc3: float | None = None,
) -> pd.DataFrame:
    if hyperparam_csv is not None:
        warnings.warn(
            "--hyperparam-csv uses Step 02 aggregates over *sampled* catalog rows, "
            "not the full Step 00 intrinsic rate. Omit this flag to read models_sspc.hdf5.",
            stacklevel=2,
        )
        hp = pd.read_csv(hyperparam_csv)
        if not _table_uses_sspc_axes(hp):
            raise ValueError("hyperparam CSV is not an SSPC (sfr_a, mu0) grid table.")
        return normalize_hyperparam_grid(hp)

    hdf5_path = sspc_hdf5.resolve()
    if not hdf5_path.is_file():
        raise FileNotFoundError(
            f"Step 00 HDF5 not found: {hdf5_path}\n"
            "Run: python scripts/00_sspc_data_generation.py"
        )
    print(f"Reading Step 00 HDF5: {hdf5_path}", flush=True)
    return build_grid_table_from_00_hdf5(hdf5_path, v_gpc3=v_gpc3)


def metric_values(hp: pd.DataFrame, metric: MetricName) -> pd.Series:
    if metric == "rate":
        return hp["rate_per_gpc3_yr"].astype(np.float64)
    if metric == "rate_weight":
        return hp["sum_weight"].astype(np.float64)
    if metric == "count":
        return hp["n_systems"].astype(np.float64)
    if metric == "log_rate":
        r = hp["rate_per_gpc3_yr"].astype(np.float64)
        return np.log10(np.maximum(r, 1e-30))
    raise ValueError(metric)


def pivot_channel(
    hp: pd.DataFrame,
    channel: str,
    metric: MetricName,
    *,
    sfra_axis: np.ndarray | None = None,
    mu0_axis: np.ndarray | None = None,
    mask_edges: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (sfra_axis, mu0_axis, Z) with shape (n_sfra, n_mu0)."""
    sub = hp.loc[hp["channel"].astype(str) == channel].copy()
    if sub.empty:
        raise ValueError(f"No rows for channel {channel!r}")

    sub["_metric"] = metric_values(sub, metric)
    if sfra_axis is None:
        sfra_vals = np.sort(sub["sfra"].unique())
    else:
        sfra_vals = np.asarray(sfra_axis, dtype=np.float64)
    if mu0_axis is None:
        mu0_vals = np.sort(sub["mu0"].unique())
    else:
        mu0_vals = np.asarray(mu0_axis, dtype=np.float64)

    z = np.full((len(sfra_vals), len(mu0_vals)), np.nan, dtype=np.float64)
    sfra_to_i = {float(v): i for i, v in enumerate(sfra_vals)}
    mu0_to_j = {float(v): j for j, v in enumerate(mu0_vals)}

    for _, row in sub.iterrows():
        sfra_key = float(row["sfra"])
        mu0_key = float(row["mu0"])
        if sfra_key not in sfra_to_i or mu0_key not in mu0_to_j:
            continue
        i = sfra_to_i[sfra_key]
        j = mu0_to_j[mu0_key]
        z[i, j] = float(row["_metric"])

    if np.isnan(z).any():
        missing = int(np.isnan(z).sum())
        print(f"  [{channel}] warning: {missing} grid cells missing in table", flush=True)

    if mask_edges:
        z = mask_grid_z_edges(z)

    return sfra_vals, mu0_vals, z


def apply_grid_average(z: np.ndarray, average_over: AverageOver) -> np.ndarray:
    """Mean over ``mu0`` (rows constant in μ₀) or ``sfra`` (columns constant in a_SF)."""
    if average_over == "none":
        return z
    with np.errstate(invalid="ignore"):
        if average_over == "mu0":
            reduced = np.nanmean(z, axis=1, keepdims=True)
        elif average_over == "sfra":
            reduced = np.nanmean(z, axis=0, keepdims=True)
        else:
            raise ValueError(average_over)
    return np.broadcast_to(reduced, z.shape).copy()


def _pooled_metric_values(
    hp: pd.DataFrame,
    metric: MetricName,
    *,
    average_over: AverageOver = "none",
    mask_edges: bool = True,
) -> np.ndarray:
    pooled: list[np.ndarray] = []
    for ch in CHANNELS:
        _, _, z = pivot_channel(hp, ch, metric, mask_edges=mask_edges)
        z = apply_grid_average(z, average_over)
        fin = z[np.isfinite(z)]
        if fin.size:
            pooled.append(fin)
    if not pooled:
        return np.array([], dtype=np.float64)
    return np.concatenate(pooled)


def _global_color_norm(
    hp: pd.DataFrame,
    metric: MetricName,
    *,
    color_scale: ColorScaleMode,
    average_over: AverageOver = "none",
    mask_edges: bool = True,
) -> matplotlib.colors.Normalize:
    """One color scale shared across all channel panels."""
    all_vals = _pooled_metric_values(
        hp, metric, average_over=average_over, mask_edges=mask_edges
    )
    if all_vals.size == 0:
        return matplotlib.colors.Normalize(vmin=0.0, vmax=1.0)

    vmin = float(np.nanmin(all_vals))
    vmax = float(np.nanmax(all_vals))

    if color_scale == "log" and metric != "log_rate":
        nonpos = int(np.sum(all_vals <= 0))
        if nonpos > 0:
            print(
                f"[plot] warning: {nonpos} grid values are <= 0; "
                "log color scale would mask them. Falling back to linear scale.",
                flush=True,
            )
            return matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        pos = all_vals[all_vals > 0]
        if pos.size == 0:
            print(
                "[plot] warning: no positive values; using linear global scale.",
                flush=True,
            )
            return matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        return matplotlib.colors.LogNorm(
            vmin=float(np.min(pos)),
            vmax=float(np.max(pos)),
        )

    return matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)


def plot_heatmaps(
    hp: pd.DataFrame,
    metric: MetricName,
    *,
    color_scale: ColorScaleMode,
    cmap_style: ColormapStyle,
    average_over: AverageOver,
    mask_edges: bool,
    mark_fiducial_study: bool,
    out_path: Path,
    use_tex: bool,
    v_gpc3: float,
) -> None:
    tex_ok = _configure_matplotlib(use_tex=use_tex)
    if use_tex and not tex_ok:
        print(
            "[plot] LaTeX unavailable; using matplotlib mathtext for labels.",
            flush=True,
        )

    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.4), constrained_layout=True)

    label_map = {
        "rate": r"$\mathcal{R}$ [Gpc$^{-3}\,\mathrm{yr}^{-1}$]",
        "rate_weight": r"$\sum w$",
        "count": r"$N_{\mathrm{events}}$",
        "log_rate": r"$\log_{10}\mathcal{R}$ [Gpc$^{-3}\,\mathrm{yr}^{-1}$]",
    }
    title_map = {
        "rate": r"Intrinsic merger rate density",
        "rate_weight": r"Sum of catalog weights $\sum w$",
        "count": r"Stored merger count per grid cell",
        "log_rate": r"$\log_{10}$ intrinsic merger rate density",
    }
    scale_note = ""
    if color_scale == "log" and metric != "log_rate":
        scale_note = " (log intensity)"
    elif color_scale == "linear":
        scale_note = " (linear intensity)"
    avg_notes = {
        "mu0": r", mean over $\mu_0$",
        "sfra": r", mean over $a_{\mathrm{SF}}$",
    }
    scale_note += avg_notes.get(average_over, "")

    cmap = _heatmap_cmap(cmap_style)
    color_norm = _global_color_norm(
        hp,
        metric,
        color_scale=color_scale,
        average_over=average_over,
        mask_edges=mask_edges,
    )

    ims = []
    for ax, ch in zip(axes, CHANNELS):
        sfra, mu0, z = pivot_channel(hp, ch, metric, mask_edges=mask_edges)
        z = apply_grid_average(z, average_over)
        z_plot = np.ma.masked_invalid(z)
        im = pcolormesh_sfra_mu0(
            ax,
            mu0,
            sfra,
            z_plot,
            mu0_range=MU0_RANGE,
            sfra_range=SFRA_RANGE,
            norm=color_norm,
            cmap=cmap,
        )
        ax.set_title(ch, fontsize=11)
        ax.set_xlabel(r"$\mu_0$")
        ax.set_ylabel(r"$a_{\mathrm{SF}}$")
        if mark_fiducial_study and ch in FIDUCIAL_STUDY_CHANNELS:
            overlay_fiducial_study_marks(
                ax,
                legend=(ch == "SMT"),
                mu0_range=MU0_RANGE,
                sfra_range=SFRA_RANGE,
            )
        ims.append(im)

    if mark_fiducial_study:
        scale_note += r"; fiducial-study marks on SMT/CE ($z=0.2$, fixed nuisances)"
    fig.suptitle(title_map[metric] + scale_note, fontsize=13)
    cbar = fig.colorbar(ims[-1], ax=axes.ravel().tolist(), shrink=0.85, pad=0.02)
    cbar.set_label(label_map[metric], fontsize=11)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Heatmaps of intrinsic merger rate density on the SSPC (sfr_a, mu0) grid."
    )
    p.add_argument(
        "--sspc-hdf5",
        type=Path,
        default=None,
        help="Step 00 output HDF5 (default: data/sspc/models_sspc.hdf5).",
    )
    p.add_argument(
        "--hyperparam-csv",
        type=Path,
        default=None,
        help="Optional Step 02 CSV (sampled-catalog aggregates; not recommended).",
    )
    p.add_argument(
        "--metric",
        choices=("rate", "count", "log_rate", "rate_weight"),
        default="rate",
        help="rate=Gpc^-3 yr^-1 (default); rate_weight=sum w; count=n_systems; log_rate=log10(rate).",
    )
    p.add_argument(
        "--color-scale",
        choices=("log", "linear"),
        default="log",
        help="Global intensity mapping: log (default) or linear.",
    )
    p.add_argument(
        "--colormap",
        choices=("sequential", "diverging"),
        default="sequential",
        help=(
            "Colormap style: sequential (rocket/inferno, default) or diverging "
            "(RdBu_r: two hues with a white midpoint; same log/linear norm)."
        ),
    )
    p.add_argument(
        "--linear-scale",
        action="store_true",
        help="Shorthand for --color-scale linear.",
    )
    p.add_argument(
        "--average-over",
        choices=("none", "mu0", "sfra"),
        default="none",
        help=(
            "Average the metric over one grid axis before plotting (reduces nuisance "
            "speckle). mu0: constant along μ₀ (shows trend vs a_SF). "
            "sfra: constant along a_SF (shows trend vs μ₀)."
        ),
    )
    p.add_argument(
        "--z-max",
        type=float,
        default=_SSPC_Z_MAX,
        help="Comoving-volume upper limit z_max for rate density (default: 10, matches Step 00).",
    )
    p.add_argument(
        "--no-mask-edges",
        action="store_true",
        help=(
            "Plot all grid cells. Default masks the minimum sfr_a row and "
            "minimum mu0 column (unphysical SSPC edge bins)."
        ),
    )
    p.add_argument(
        "--mark-fiducial-study",
        action="store_true",
        help=(
            "Overlay five fiducial-comparison grid points (colored x) on SMT and CE "
            "only: fiducial (a_SF=0.02, mu0=0.025), mu0 low/high at a_SF=0.02, "
            "a_SF low/high at mu0=0.025."
        ),
    )
    p.add_argument(
        "--no-tex",
        action="store_true",
        help="Disable LaTeX text rendering.",
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
    v_gpc3 = _comoving_volume_gpc3(float(args.z_max))
    hdf5 = (args.sspc_hdf5 or default_00_hdf5_path()).resolve()

    hp = load_grid_table(
        sspc_hdf5=hdf5,
        hyperparam_csv=args.hyperparam_csv.resolve() if args.hyperparam_csv else None,
        v_gpc3=v_gpc3,
    )
    # Re-normalize density if user changes the comoving-volume limit.
    if abs(float(args.z_max) - _SSPC_Z_MAX) > 1e-9:
        hp = add_rate_density_columns(hp, v_gpc3=v_gpc3)

    metric: MetricName = args.metric
    color_scale: ColorScaleMode = args.color_scale
    cmap_style: ColormapStyle = args.colormap
    average_over: AverageOver = args.average_over
    if args.linear_scale:
        if color_scale != "log":
            raise SystemExit(
                "Use only one of --linear-scale and --color-scale (not both)."
            )
        color_scale = "linear"

    if args.out is not None:
        out = args.out.resolve()
    else:
        stem = f"grid_{metric}"
        tag = hdf5_output_tag(hdf5)
        if tag:
            stem = f"{stem}_{tag}"
        if average_over != "none":
            stem = f"{stem}_avg_{average_over}"
        if args.mark_fiducial_study:
            stem = f"{stem}_fiducial_marks"
        out = resolve_plot_output(
            Path(__file__),
            no_timestamp_subdir=args.no_timestamp_subdir,
            filename=f"{stem}.png",
        )

    mask_edges = not bool(args.no_mask_edges)
    mark_fiducial_study = bool(args.mark_fiducial_study)

    plot_heatmaps(
        hp,
        metric,
        color_scale=color_scale,
        cmap_style=cmap_style,
        average_over=average_over,
        mask_edges=mask_edges,
        mark_fiducial_study=mark_fiducial_study,
        out_path=out,
        use_tex=not bool(args.no_tex),
        v_gpc3=v_gpc3,
    )

    meta = {
        "metric": metric,
        "mask_edges": mask_edges,
        "mark_fiducial_study": mark_fiducial_study,
        "fiducial_study_marks": (
            [
                {
                    "label": m["label"],
                    "mu0": float(m["mu0"]),
                    "sfra": float(m["sfra"]),
                    "color": m["color"],
                }
                for m in FIDUCIAL_STUDY_MARKS
            ]
            if mark_fiducial_study
            else None
        ),
        "fiducial_study_channels": sorted(FIDUCIAL_STUDY_CHANNELS)
        if mark_fiducial_study
        else None,
        "average_over": average_over,
        "color_scale": color_scale,
        "colormap_style": cmap_style,
        "colormap": _heatmap_cmap(cmap_style),
        "usetex": not bool(args.no_tex),
        "comoving_volume_gpc3": v_gpc3,
        "z_max": float(args.z_max),
        "n_rows": int(len(hp)),
        "channels": list(CHANNELS),
        "sfra_range": [float(hp["sfra"].min()), float(hp["sfra"].max())],
        "mu0_range": [float(hp["mu0"].min()), float(hp["mu0"].max())],
        "data_source": "step_00_hdf5" if args.hyperparam_csv is None else "step_02_csv",
        "sspc_hdf5": str(hdf5),
        "rate_density_note": (
            "From Step 00: intrinsic_rate_yr = sum of BPS binary weights before sampling; "
            "rate_per_gpc3_yr = intrinsic_rate_yr / V_comoving(z_max)."
        ),
    }
    meta_path = out.with_suffix(".json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Saved → {meta_path}", flush=True)


if __name__ == "__main__":
    main()
