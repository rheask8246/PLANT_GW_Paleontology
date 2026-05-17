#!/usr/bin/env python3
"""
Shared TNG vs SSPC distribution plotting (Figure 5 / Figure 4 style).

Used by ``00_distribution_compare.py`` and ``04_emulator_m1_compare.py``.
Reproduces Figure 5 of Briel et al. (Fit_SFRD_TNG paper) and overplots
the SSPC-generated data so both can be directly compared.

Figure layout (3 rows × 2 columns):
  Left column  : TNG100-1 simulation (Rate_info.h5 + COMPAS_Output_wWeights.h5)
  Right column : SSPC data (data/sspc/models_sspc.hdf5)
  Row 0 : all channels (stable + CE; CHE excluded following original paper)
  Row 1 : stable mass-transfer (SMT) only
  Row 2 : common-envelope (CE) only

X-axis : BBH primary mass m₁ [M☉]
Y-axis : dR/dm₁  (left) intrinsic rate density [Gpc⁻³ yr⁻¹ M☉⁻¹],
                 (right) intrinsic merger-rate weighted dN/dm₁ (area-normalised for shape comparison)

Colors : rocket_r colormap, darkest = lowest z, lightest = highest z
Gray   : GWTC-4 B-Spline mass distribution (BBHMassSpinRedshift_BSplineIID.h5)

Library module — run ``scripts/analysis/00_distribution_compare.py`` or
``04_emulator_m1_compare.py`` from the project root.
"""
from __future__ import annotations

import sys
from pathlib import Path as _Path

_PROJECT_ROOT = _Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from plant_paths import (  # noqa: E402
    CHECKPOINT_DIR,
    HYPERPARAM_TABLE_ENCODED_CSV,
    PLOTS_ROOT,
    PROJECT_ROOT,
    REPO_ROOT,
    ensure_paths,
    find_data_dir,
    load_posterior_network_module,
    plot_run_dir,
    plot_script_stem,
    resolve_plot_output,
)

ensure_paths()

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns
from astropy.table import Table
from astropy.cosmology import Planck15 as cosmo
from scipy import stats

# Optional GWTC-4 overlay (requires popsummary)
try:
    from popsummary.popresult import PopulationResult as _PopResult
    _HAS_POPSUMMARY = True
except ImportError:
    _HAS_POPSUMMARY = False
    print("[WARNING] popsummary not found – GWTC-4 overlay will be skipped.")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_TNG_DATA_DEFAULT  = REPO_ROOT / "Fit_SFRD_TNG" / "data"
_SSPC_HDF5_DEFAULT = PROJECT_ROOT / "data" / "sspc" / "models_sspc.hdf5"
_COMPAS_DEFAULT    = _TNG_DATA_DEFAULT / "COMPAS_Output_wWeights.h5"
_GWTC4_DEFAULT     = _TNG_DATA_DEFAULT / "BBHMassSpinRedshift_BSplineIID.h5"
_ANALYSIS_DIR = Path(__file__).resolve().parent.parent
_DIST_COMPARE_SCRIPT = _ANALYSIS_DIR / "00_distribution_compare.py"
_EMULATOR_M1_SCRIPT = _ANALYSIS_DIR / "04_emulator_m1_compare.py"


def _legacy_plot_dir(script_path: Path | str) -> Path:
    return PLOTS_ROOT / plot_script_stem(script_path)

# Redshift slices for SSPC column (intrinsic rate spans z=0.1–10)
SSPC_Z_PLOT = [0.1, 0.5, 1.0, 2.0, 5.0]

# Redshift slices for TNG column: auto-detected from Rate_info.h5.
# All available TNG z values are used, matching the original Figure 5.
TNG_Z_MAX = 10.0   # include all TNG redshifts up to this value

# Style
_COLORMAP   = sns.color_palette("rocket_r", as_cmap=True)
_X_BINS     = np.arange(0.0, 85.0, 2.5)
_X_KDE      = np.linspace(0.1, 80.0, 500)
_X_LIM      = (0.0, 80.0)
_Y_LIM      = (1e-3, 40.0)

# CFM vs diffusion `--emulator-m1-compare` only (Figure 5 uses _X_LIM / _Y_LIM above).
_EMULATOR_M1_XLIM = (0.0, 50.0)
# Upper y: geometric mean of 10⁻¹ and 10⁰ (half a dex in log₁₀).
_EMULATOR_M1_YMAX = float(10.0 ** ((-1.0 + 0.0) / 2.0))
_EMULATOR_M1_YMIN = 1e-4

CHANNEL_ROW = {"all": 0, "SMT": 1, "CE": 2}


# ---------------------------------------------------------------------------
# m₁ from chirp mass + mass ratio
# ---------------------------------------------------------------------------

def m1_from_mchirp_q(mchirp: np.ndarray, q: np.ndarray) -> np.ndarray:
    """
    Primary mass from chirp mass Mc and mass ratio q = m2/m1 ≤ 1.
        m1 = Mc * ((1 + q) / q³)^(1/5)
    """
    return mchirp * ((1.0 + q) / np.power(q, 3.0)) ** 0.2


# ---------------------------------------------------------------------------
# TNG data loading (adapted from MassDistHelperFunctions.read_data and
# TNG_BBHpop_properties.plot_BBH_mass_dist_formation_channels)
# ---------------------------------------------------------------------------

def load_tng_dco(compas_path: Path) -> Table:
    """Load BBH / DCO table from COMPAS_Output_wWeights.h5."""
    with h5py.File(compas_path, "r") as f:
        dco_key = ("BSE_Double_Compact_Objects" if "BSE_Double_Compact_Objects" in f
                   else "DoubleCompactObjects")
        sys_key = ("BSE_System_Parameters" if "BSE_System_Parameters" in f
                   else "SystemParameters")
        ce_key  = ("CE_Event_Counter" if "CE_Event_Counter" in f[dco_key]
                   else "CE_Event_Count")
        dco = Table()
        dco["SEED"]              = f[dco_key]["SEED"][()]
        dco[ce_key]              = f[dco_key][ce_key][()]
        dco["Mass(1)"]           = f[dco_key]["Mass(1)"][()]
        dco["Mass(2)"]           = f[dco_key]["Mass(2)"][()]
        dco["M_moreMassive"]     = np.maximum(f[dco_key]["Mass(1)"][()],
                                              f[dco_key]["Mass(2)"][()])
        dco["Stellar_Type(1)"]   = f[dco_key]["Stellar_Type(1)"][()]
        dco["Stellar_Type(2)"]   = f[dco_key]["Stellar_Type(2)"][()]
        seeds_sys = f[sys_key]["SEED"][()]
        bool_sys  = np.isin(seeds_sys, dco["SEED"])
        dco["Stellar_Type@ZAMS(1)"] = f[sys_key]["Stellar_Type@ZAMS(1)"][()][bool_sys]
    return dco, ce_key


def load_tng_rates(rate_h5: Path):
    """
    Load the first available rate key from Rate_info.h5.
    Returns (redshifts, DCO_mask, merger_rate_density).
    """
    with h5py.File(rate_h5, "r") as f:
        rate_key = list(f.keys())[0]
        redshifts = f[rate_key]["redshifts"][()]
        dco_mask  = f[rate_key]["DCOmask"][()]
        merger_rate = f[rate_key]["merger_rate"][()]
    return redshifts, dco_mask, merger_rate, rate_key


def tng_channel_filter(dco: Table, ce_key: str,
                       channel: str) -> np.ndarray:
    """
    Return boolean mask selecting the requested channel from a BBH-only DCO table.
    CHE is excluded following the original Figure 5 convention.
    """
    bbh_bool  = ((dco["Stellar_Type(1)"] == 14) & (dco["Stellar_Type(2)"] == 14))
    not_che   = dco["Stellar_Type@ZAMS(1)"] != 16   # exclude chemically homogeneous
    if channel == "all":
        ch_bool = np.ones(len(dco), dtype=bool)
    elif channel == "SMT":
        ch_bool = dco[ce_key] == 0
    elif channel == "CE":
        ch_bool = dco[ce_key] > 0
    else:
        raise ValueError(f"Unknown channel: {channel}")
    return bbh_bool & not_che & ch_bool


def tng_weighted_kde(masses: np.ndarray, weights: np.ndarray,
                     x_kde: np.ndarray = _X_KDE) -> np.ndarray | None:
    """Return KDE of masses weighted by weights, renormalised to sum(hist)."""
    weights = np.clip(weights, 0.0, None)
    if weights.sum() == 0 or len(masses) < 2:
        return None
    hist, _ = np.histogram(masses, weights=weights, bins=_X_BINS)
    try:
        kernel = stats.gaussian_kde(masses, bw_method="scott", weights=weights)
    except (np.linalg.LinAlgError, ValueError):
        return None
    return kernel(x_kde) * hist.sum()


# ---------------------------------------------------------------------------
# SSPC data loading
# ---------------------------------------------------------------------------

def load_sspc_channel(sspc_path: Path, channel: str,
                      z_values: list[float] | None = None
                      ) -> dict[float, tuple[np.ndarray, np.ndarray]]:
    """
    Load all grid points for one channel from models_sspc.hdf5.
    Returns {z: (m1_array, weight_array)} aggregated across all (sfra, mu0) grid points.
    Only z values found in the data are returned; optionally filtered by z_values.
    """
    agg: dict[float, list] = {}
    with h5py.File(sspc_path, "r") as f:
        if channel not in f:
            return {}
        ch_grp = f[channel]
        for sfra_key in ch_grp.keys():
            for mu0_key in ch_grp[sfra_key].keys():
                raw = ch_grp[sfra_key][mu0_key]["table"][()]
                df  = pd.DataFrame(raw)
                for z_val in df["z"].unique():
                    if z_values and not any(abs(z_val - zv) < 0.01 for zv in z_values):
                        continue
                    mask = np.abs(df["z"] - z_val) < 0.01
                    z_key = round(float(z_val), 2)
                    if z_key not in agg:
                        agg[z_key] = ([], [])
                    m1 = m1_from_mchirp_q(df.loc[mask, "mchirp"].values,
                                          df.loc[mask, "q"].values)
                    w  = df.loc[mask, "weight"].values
                    agg[z_key][0].append(m1)
                    agg[z_key][1].append(w)
    return {z: (np.concatenate(v[0]), np.concatenate(v[1]))
            for z, v in agg.items() if len(v[0])}


def sspc_weighted_kde(m1: np.ndarray, weights: np.ndarray,
                      x_kde: np.ndarray = _X_KDE) -> np.ndarray | None:
    """KDE of SSPC m1 values, area-normalised for shape comparison."""
    weights = np.clip(weights, 0.0, None)
    if weights.sum() == 0 or len(m1) < 2:
        return None
    hist, _ = np.histogram(m1, weights=weights, bins=_X_BINS)
    if hist.sum() == 0:
        return None
    try:
        kernel = stats.gaussian_kde(m1, bw_method="scott", weights=weights)
    except (np.linalg.LinAlgError, ValueError):
        return None
    kde = kernel(x_kde) * hist.sum()
    # Area-normalise for shape comparison
    area = np.trapezoid(kde, x_kde) if hasattr(np, "trapezoid") else np.trapz(kde, x_kde)
    return kde / area if area > 0 else kde


# ---------------------------------------------------------------------------
# GWTC-4 B-Spline overlay
# ---------------------------------------------------------------------------

def load_gwtc4(gwtc4_path: Path):
    """
    Returns (m1_grid, median_pdf, low_pdf, high_pdf) from the GWTC-4 B-Spline
    mass distribution (rate_vs_mass_1_at_z0-2 key).  Returns None if unavailable.
    """
    if not _HAS_POPSUMMARY or not gwtc4_path.exists():
        return None
    try:
        result = _PopResult(str(gwtc4_path))
        dat = result.get_rates_on_grids("rate_vs_mass_1_at_z0-2")
        m1  = dat[0][0]
        pdfs = dat[1]
        return (m1,
                np.median(pdfs, axis=0),
                np.percentile(pdfs, 5, axis=0),
                np.percentile(pdfs, 95, axis=0))
    except Exception as e:
        print(f"[WARNING] Could not load GWTC-4 data: {e}")
        return None


# ---------------------------------------------------------------------------
# GWTC-4 redshift-rate overlay (Figure 6 style)
# ---------------------------------------------------------------------------

def load_gwtc4_rate_vs_redshift(gwtc4_path: Path):
    """
    Returns (z_grid, median_rate, low_rate, high_rate) from GWTC-4
    rate_vs_redshift. Returns None if unavailable.
    """
    if not _HAS_POPSUMMARY or not gwtc4_path.exists():
        return None
    try:
        result = _PopResult(str(gwtc4_path))
        dat = result.get_rates_on_grids("rate_vs_redshift")
        z_grid = dat[0][0]
        rate_post = dat[1]
        return (
            z_grid,
            np.median(rate_post, axis=0),
            np.percentile(rate_post, 5, axis=0),
            np.percentile(rate_post, 95, axis=0),
        )
    except Exception as e:
        print(f"[WARNING] Could not load GWTC-4 rate-vs-redshift data: {e}")
        return None


def load_sspc_rate_vs_redshift(
    sspc_path: Path,
    include_channels: tuple[str, ...] = ("CE", "SMT"),
) -> tuple[np.ndarray, np.ndarray]:
    """
    Aggregate SSPC intrinsic merger-rate weights as a function of z.
    By default CHE is excluded to mirror the paper convention used for Figure 5.
    """
    if not sspc_path.exists():
        return np.array([]), np.array([])

    z_weight_sum: dict[float, float] = {}
    with h5py.File(sspc_path, "r") as f:
        for ch in include_channels:
            if ch not in f:
                continue
            for sfra_key in f[ch].keys():
                for mu0_key in f[ch][sfra_key].keys():
                    raw = f[ch][sfra_key][mu0_key]["table"][()]
                    df = pd.DataFrame(raw)
                    if "z" not in df or "weight" not in df:
                        continue
                    grouped = df.groupby("z")["weight"].sum()
                    for z_val, w_sum in grouped.items():
                        z_key = round(float(z_val), 3)
                        z_weight_sum[z_key] = z_weight_sum.get(z_key, 0.0) + float(w_sum)

    if not z_weight_sum:
        return np.array([]), np.array([])

    z_sorted = np.array(sorted(z_weight_sum.keys()), dtype=float)
    rate_sorted = np.array([z_weight_sum[z] for z in z_sorted], dtype=float)
    return z_sorted, rate_sorted


# ---------------------------------------------------------------------------
# Main plot
# ---------------------------------------------------------------------------

def make_figure(
    tng_data_dir: Path,
    sspc_path: Path,
    compas_path: Path,
    output_path: Path,
):
    channels  = ["all", "SMT", "CE"]
    ch_titles = ["All channels (stable + CE)", "Stable mass transfer (SMT)", "CE channel"]

    # ── Load TNG data ────────────────────────────────────────────────────────
    tng_available = tng_data_dir.exists() and compas_path.exists()
    rate_h5 = tng_data_dir / "Rate_info.h5"
    tng_rate_available = tng_available and rate_h5.exists()

    tng_z_plot = []
    if tng_rate_available:
        print("Loading TNG data …")
        dco, ce_key = load_tng_dco(compas_path)
        redshifts, dco_mask, merger_rate, rate_key = load_tng_rates(rate_h5)
        print(f"  Rate key: {rate_key}")
        print(f"  TNG redshifts: {redshifts}")
        # Use ALL available TNG z values (as in original Figure 5)
        tng_z_plot = [float(round(z, 3)) for z in redshifts if z <= TNG_Z_MAX and z > 0]
        print(f"  TNG z values for plot: {tng_z_plot}")
    else:
        print("[INFO] TNG data not found – left column will be blank.")

    # Independent colormap for TNG (dark→light over TNG's z range)
    n_tng = max(len(tng_z_plot), 1)
    tng_colors = _COLORMAP(np.linspace(0.1, 0.9, n_tng))

    # ── Detect available SSPC z values ───────────────────────────────────────
    all_sspc_z: set[float] = set()
    if sspc_path.exists():
        with h5py.File(sspc_path, "r") as f:
            for ch in ["CE", "CHE", "SMT"]:
                if ch not in f:
                    continue
                sfra0 = list(f[ch].keys())[0]
                mu0_0 = list(f[ch][sfra0].keys())[0]
                raw   = f[ch][sfra0][mu0_0]["table"][()]
                df    = pd.DataFrame(raw)
                all_sspc_z.update(round(float(z), 2) for z in df["z"].unique())

    sspc_z_plot = sorted(z for z in SSPC_Z_PLOT if z in all_sspc_z)
    if not sspc_z_plot:
        sspc_z_plot = sorted(all_sspc_z)[:5]
    print(f"SSPC z values for plot: {sspc_z_plot}")

    # Independent colormap for SSPC (dark→light over SSPC's z range)
    n_sspc = max(len(sspc_z_plot), 1)
    sspc_colors = _COLORMAP(np.linspace(0.1, 0.9, n_sspc))

    # ── Load GWTC-4 overlay ──────────────────────────────────────────────────
    gwtc4 = load_gwtc4(_TNG_DATA_DEFAULT / "BBHMassSpinRedshift_BSplineIID.h5")

    # ── Load SSPC data for each channel ──────────────────────────────────────
    sspc_data: dict[str, dict] = {}
    sspc_plot_channel_map = {
        "all": ["CE", "SMT"],   # CHE excluded, matching TNG Figure 5
        "SMT": ["SMT"],
        "CE":  ["CE"],
    }
    if sspc_path.exists():
        print("Loading SSPC data …")
        for ch_label, src_channels in sspc_plot_channel_map.items():
            combined: dict[float, list] = {}
            for src_ch in src_channels:
                ch_d = load_sspc_channel(sspc_path, src_ch, sspc_z_plot)
                for z, (m1, w) in ch_d.items():
                    if z not in combined:
                        combined[z] = ([], [])
                    combined[z][0].append(m1)
                    combined[z][1].append(w)
            sspc_data[ch_label] = {z: (np.concatenate(v[0]), np.concatenate(v[1]))
                                   for z, v in combined.items() if v[0]}
            print(f"  SSPC '{ch_label}': {sorted(sspc_data[ch_label].keys())} z values")
    else:
        print(f"[WARNING] SSPC HDF5 not found at {sspc_path}")
        sspc_data = {ch: {} for ch in channels}

    # ── Figure setup ─────────────────────────────────────────────────────────
    xlabel = r"$M_{\mathrm{BH,1}} \ [\mathrm{M}_\odot]$"
    ylabel_tng  = (r"$\frac{d\mathcal{R}}{dM_{\mathrm{BH,1}}} "
                   r"\ [\mathrm{Gpc}^{-3}\,\mathrm{yr}^{-1}\,\mathrm{M}_\odot^{-1}]$")
    ylabel_sspc = (r"$\frac{d\mathcal{R}_\mathrm{intr}}{dM_{\mathrm{BH,1}}}$ "
                   r"(intrinsic rate, area-normalised)")

    fig, axes = plt.subplots(3, 2, sharex=True, sharey=False,
                             figsize=(14, 16))
    fig.subplots_adjust(wspace=0.32, hspace=0.08)

    # ── Plot each row (channel) ───────────────────────────────────────────────
    for row, (ch_label, ch_title) in enumerate(zip(channels, ch_titles)):
        ax_tng  = axes[row, 0]
        ax_sspc = axes[row, 1]

        # GWTC-4 gray overlay on both columns
        for ax in (ax_tng, ax_sspc):
            if gwtc4 is not None:
                m1_g, med_g, lo_g, hi_g = gwtc4
                lbl = "GWTC-4" if row == 0 else None
                ax.plot(m1_g, med_g, lw=1.8, color="grey", zorder=1, label=lbl)
                ax.fill_between(m1_g, lo_g, hi_g, color="grey", alpha=0.14, zorder=0)

        # ── Left: TNG (unchanged from original Figure 5) ─────────────────────
        if tng_rate_available and tng_z_plot:
            merging_dco = dco[dco_mask]
            ch_mask     = tng_channel_filter(merging_dco, ce_key, ch_label)
            dco_ch      = merging_dco[ch_mask]
            rate_ch     = merger_rate[ch_mask, :]
            masses      = dco_ch["M_moreMassive"]

            for iz, z_val in enumerate(tng_z_plot):
                iz_tng = np.argmin(np.abs(redshifts - z_val))
                w   = rate_ch[:, iz_tng]
                kde = tng_weighted_kde(masses, w)
                if kde is not None:
                    lbl = f"$z = {z_val:.2f}$" if row == 0 else None
                    ax_tng.plot(_X_KDE, kde, color=tng_colors[iz], lw=2.5, label=lbl)

        ax_tng.set_yscale("log")
        ax_tng.set_xlim(*_X_LIM)
        ax_tng.set_ylim(*_Y_LIM)
        ax_tng.xaxis.set_major_locator(ticker.MultipleLocator(10))
        ax_tng.xaxis.set_minor_locator(ticker.MultipleLocator(5))
        ax_tng.tick_params(length=10, width=2, which="major")
        ax_tng.tick_params(length=6,  width=1.5, which="minor")
        ax_tng.text(0.05, 0.05, ch_title, transform=ax_tng.transAxes,
                    fontsize=12, va="bottom")
        if not tng_rate_available:
            ax_tng.text(0.5, 0.5, "TNG data\nnot found",
                        transform=ax_tng.transAxes, ha="center", va="center",
                        fontsize=12, color="gray")

        # ── Right: SSPC (independent z range and colormap) ───────────────────
        ch_sspc    = sspc_data.get(ch_label, {})
        sspc_y_max = 0.0
        for iz, z_val in enumerate(sspc_z_plot):
            if z_val not in ch_sspc:
                continue
            m1, w = ch_sspc[z_val]
            kde   = sspc_weighted_kde(m1, w)
            if kde is not None:
                lbl = f"$z = {z_val:.1f}$" if row == 0 else None
                ax_sspc.plot(_X_KDE, kde, color=sspc_colors[iz], lw=2.5, label=lbl)
                sspc_y_max = max(sspc_y_max, kde.max())

        ax_sspc.set_yscale("log")
        ax_sspc.set_xlim(*_X_LIM)
        y_hi = max(sspc_y_max * 3, 10.0) if sspc_y_max > 0 else 10.0
        ax_sspc.set_ylim(1e-6, y_hi)
        ax_sspc.xaxis.set_major_locator(ticker.MultipleLocator(10))
        ax_sspc.xaxis.set_minor_locator(ticker.MultipleLocator(5))
        ax_sspc.tick_params(length=10, width=2, which="major")
        ax_sspc.tick_params(length=6,  width=1.5, which="minor")
        ax_sspc.text(0.05, 0.05, ch_title, transform=ax_sspc.transAxes,
                     fontsize=12, va="bottom")

        if row == 0:
            ax_tng.set_ylabel(ylabel_tng,  fontsize=11)
            ax_sspc.set_ylabel(ylabel_sspc, fontsize=10)

    # ── Column titles ────────────────────────────────────────────────────────
    axes[0, 0].set_title("TNG100-1 simulation  (intrinsic rate)", fontsize=14, pad=8)
    axes[0, 1].set_title("SSPC — intrinsic rate  (all grid points)", fontsize=14, pad=8)

    # ── Bottom row x-labels ──────────────────────────────────────────────────
    for col in range(2):
        axes[2, col].set_xlabel(xlabel, fontsize=13)

    # ── Separate colorbars: TNG (left) and SSPC (right) ──────────────────────
    # TNG colorbar
    if tng_z_plot:
        norm_tng = matplotlib.colors.BoundaryNorm(np.arange(n_tng + 1) - 0.5, _COLORMAP.N)
        cbar_tng_ax = fig.add_axes([0.46, 0.12, 0.012, 0.75])
        cbar_tng = fig.colorbar(
            matplotlib.cm.ScalarMappable(norm=norm_tng, cmap=_COLORMAP),
            cax=cbar_tng_ax, ticks=np.arange(n_tng),
        )
        cbar_tng.set_ticklabels([f"{z:.2f}" for z in tng_z_plot], fontsize=8)
        cbar_tng.set_label(r"$z_\mathrm{merger}$ (TNG)", fontsize=11, labelpad=8)

    # SSPC colorbar
    norm_sspc = matplotlib.colors.BoundaryNorm(np.arange(n_sspc + 1) - 0.5, _COLORMAP.N)
    cbar_sspc_ax = fig.add_axes([0.92, 0.12, 0.012, 0.75])
    cbar_sspc = fig.colorbar(
        matplotlib.cm.ScalarMappable(norm=norm_sspc, cmap=_COLORMAP),
        cax=cbar_sspc_ax, ticks=np.arange(n_sspc),
    )
    cbar_sspc.set_ticklabels([str(z) for z in sspc_z_plot], fontsize=10)
    cbar_sspc.set_label(r"$z_\mathrm{merger}$ (SSPC)", fontsize=11, labelpad=8)

    # ── Legends (top row only) ───────────────────────────────────────────────
    for col, ax in enumerate([axes[0, 0], axes[0, 1]]):
        handles, labels = ax.get_legend_handles_labels()
        # Only keep GWTC-4 entry in legend (z labels covered by colorbars)
        gwtc_h = [(h, l) for h, l in zip(handles, labels) if "GWTC" in l]
        if gwtc_h:
            ax.legend(*zip(*gwtc_h), fontsize=9, frameon=False,
                      bbox_to_anchor=(0.98, 0.98), loc="upper right")

    fig.suptitle(
        "BBH primary mass distribution vs redshift\n"
        r"Left: TNG100-1 intrinsic rate  $\cdot$  Right: SSPC intrinsic rate (area-normalised)",
        fontsize=13, y=0.995,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"\nSaved → {output_path}")


def make_figure4(
    tng_data_dir: Path,
    sspc_path: Path,
    output_path: Path,
):
    """
    Reproduce Fit_SFRD_TNG Figure 4 style (BBH merger-rate density vs redshift) for TNG and
    overlay SSPC-generated data for direct comparison.
    """
    def _load_total_rate(rate_path: Path) -> tuple[np.ndarray, np.ndarray] | None:
        if not rate_path.exists():
            return None
        z_vals, _dco_mask, merger_rate, _rate_key = load_tng_rates(rate_path)
        total = np.sum(merger_rate, axis=0)
        return z_vals, np.clip(total, 1e-12, None)

    tng_labels = ["50-1", "100-1", "300-1"]
    tng_colors = {"50-1": "#0072B2", "100-1": "#C51B8A", "300-1": "#8BC34A"}
    curves: dict[str, dict[str, tuple[np.ndarray, np.ndarray] | None]] = {}
    for lbl in tng_labels:
        sim_file = tng_data_dir / f"data_Rate_info_TNG{lbl}.h5"
        fit_file = tng_data_dir / f"Rate_info_TNG{lbl}.h5"
        curves[lbl] = {
            "sim": _load_total_rate(sim_file),
            "fit": _load_total_rate(fit_file),
        }

    # Backward-compatible fallback: if only one Rate_info.h5 exists, map it to TNG100 simulation.
    if all(v["sim"] is None for v in curves.values()):
        fallback = _load_total_rate(tng_data_dir / "Rate_info.h5")
        if fallback is not None:
            curves["100-1"]["sim"] = fallback

    if all((v["sim"] is None and v["fit"] is None) for v in curves.values()):
        print("[WARNING] Cannot make Figure 4 comparison; no Rate_info files found.")
        return

    # Reference simulation for SSPC scaling and ratio panel.
    sim_ref = curves["100-1"]["sim"]
    if sim_ref is None:
        for lbl in tng_labels:
            if curves[lbl]["sim"] is not None:
                sim_ref = curves[lbl]["sim"]
                break

    sspc_z, sspc_rate = load_sspc_rate_vs_redshift(sspc_path, include_channels=("CE", "SMT"))
    sspc_rate = np.clip(sspc_rate, 1e-12, None) if len(sspc_rate) else sspc_rate
    sspc_scaled = None
    if len(sspc_z) and sim_ref is not None:
        z_ref, rate_ref = sim_ref
        scale_ref = float(np.interp(0.2, z_ref, rate_ref)) / max(float(np.interp(0.2, sspc_z, sspc_rate)), 1e-12)
        sspc_scaled = sspc_rate * scale_ref

    gwtc4_rate = load_gwtc4_rate_vs_redshift(_GWTC4_DEFAULT)

    fig = plt.figure(figsize=(12, 9))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 3.6], hspace=0.0)
    ax_ratio = fig.add_subplot(gs[0, 0])
    ax = fig.add_subplot(gs[1, 0], sharex=ax_ratio)

    max_z = 0.0
    for lbl in tng_labels:
        clr = tng_colors[lbl]
        sim = curves[lbl]["sim"]
        fit = curves[lbl]["fit"]

        if sim is not None:
            z_sim, r_sim = sim
            ax.plot(z_sim, r_sim, color=clr, lw=4.0, label=f"TNG{lbl}")
            max_z = max(max_z, float(np.max(z_sim)))
        if fit is not None:
            z_fit, r_fit = fit
            ax.plot(z_fit, r_fit, color=clr, lw=2.0, ls="--")
            max_z = max(max_z, float(np.max(z_fit)))
        if sim is not None and fit is not None:
            z_sim, r_sim = sim
            z_fit, r_fit = fit
            fit_on_sim = np.interp(z_sim, z_fit, r_fit, left=np.nan, right=np.nan)
            valid = np.isfinite(fit_on_sim) & (r_sim > 0) & (fit_on_sim > 0)
            if np.any(valid):
                ax_ratio.plot(z_sim[valid], fit_on_sim[valid] / r_sim[valid], color=clr, lw=2.2)

    if sspc_scaled is not None:
        ax.plot(sspc_z, sspc_scaled, color="#009688", lw=2.4, ls="-.", label="SSPC (scaled)")
        if sim_ref is not None:
            z_ref, r_ref = sim_ref
            sspc_on_ref = np.interp(z_ref, sspc_z, sspc_scaled, left=np.nan, right=np.nan)
            valid = np.isfinite(sspc_on_ref) & (r_ref > 0) & (sspc_on_ref > 0)
            if np.any(valid):
                ax_ratio.plot(z_ref[valid], sspc_on_ref[valid] / r_ref[valid], color="#009688", lw=2.0, ls="-.")
        max_z = max(max_z, float(np.max(sspc_z)))

    if gwtc4_rate is not None:
        z_g, med_g, lo_g, hi_g = gwtc4_rate
        ax.plot(z_g, med_g, lw=1.8, color="grey", zorder=1, label="GWTC-4")
        ax.fill_between(z_g, lo_g, hi_g, alpha=0.2, color="grey", zorder=0)
        max_z = max(max_z, float(np.max(z_g)))

    # Styling to mirror paper layout.
    x_max = max(14.0, max_z)
    ax.set_xlim(0.0, x_max)
    ax.set_yscale("log")
    ax.set_ylim(1e-3, 6e2)
    ax.set_ylabel(r"$\frac{d\mathcal{R}}{dz}\ [\mathrm{Gpc}^{-3}\,\mathrm{yr}^{-1}]$", fontsize=22)
    ax.set_xlabel(r"Redshift $z$", fontsize=22)
    ax.tick_params(axis="both", which="major", labelsize=15, length=9, width=1.8)
    ax.tick_params(axis="both", which="minor", length=5, width=1.0)
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))

    ax_ratio.set_yscale("log")
    ax_ratio.set_ylim(1e-2, 1e2)
    ax_ratio.axhline(1.0, color="gray", lw=1.2)
    ax_ratio.set_ylabel(r"$\frac{d\mathcal{R}}{dz}_{\mathrm{fit}} \,/\, \frac{d\mathcal{R}}{dz}_{\mathrm{sim}}$", fontsize=16)
    ax_ratio.tick_params(axis="y", which="major", labelsize=12, length=7, width=1.5)
    ax_ratio.tick_params(axis="y", which="minor", length=4, width=1.0)
    ax_ratio.tick_params(axis="x", which="both", labelbottom=False, length=6, width=1.2)

    # Top axis: lookback time, as in the paper figure.
    ax_top = ax_ratio.twiny()
    ax_top.set_xlim(ax_ratio.get_xlim())
    z_ticks = [0, 1, 2, 6, 10, 14]
    z_ticks = [z for z in z_ticks if z <= x_max]
    ax_top.set_xticks(z_ticks)
    ax_top.set_xticklabels([f"{cosmo.lookback_time(z).value:.1f}" for z in z_ticks], fontsize=15)
    ax_top.set_xlabel("Lookback time [Gyr]", fontsize=20, labelpad=8)
    ax_top.tick_params(axis="x", which="major", length=7, width=1.2)

    from matplotlib.lines import Line2D
    style_handles = [
        Line2D([0], [0], color="black", lw=3, ls="-", label="TNG simulation"),
        Line2D([0], [0], color="black", lw=2, ls="--", label="Analytical fit"),
    ]
    if sspc_scaled is not None:
        style_handles.append(Line2D([0], [0], color="#009688", lw=2.4, ls="-.", label="SSPC comparison"))
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles + style_handles, labels + [h.get_label() for h in style_handles],
              frameon=False, fontsize=10, loc="lower left", ncol=1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {output_path}")


# ---------------------------------------------------------------------------
# Emulator m₁ comparison (fixed SSPC Λ; CFM vs diffusion)
# ---------------------------------------------------------------------------

# Training SSPC grid encoding (must match 02_build_dataset.py).
_SSPC_SFRA_RANGE = (0.010, 0.030)
_SSPC_MU0_RANGE = (0.010, 0.060)
_SSPC_PARAM_COLS = [
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
_CHANNEL_TO_ID = {"CE": 0, "CHE": 1, "GC": 2, "NSC": 3, "SMT": 4}

# Default fixed Λ for --emulator-m1-compare (TNG100-1 best-fit style; aligned with
# ``00_sspc_data_generation.py`` / Briel+ Table 1 comment block there).
_DEFAULT_EMULATOR_SSPC_MEANS = {
    "sspc_sfr_a_mean": 0.0172,
    "sspc_sfr_b_mean": 1.4425,
    "sspc_sfr_c_mean": 4.5299,
    "sspc_sfr_d_mean": 6.2261,
    "sspc_mu0_mean": 0.0247,
    "sspc_muz_mean": -0.0521,
    "sspc_sigma0_mean": 1.1509,
    "sspc_sigmaz_mean": 0.0477,
    "sspc_alpha_skew_mean": -1.8801,
}


def _lambda_values_by_index(
    *,
    channel: str,
    sfr_a: float,
    mu0: float,
    sspc_means: Dict[str, float],
    hp_encoded: pd.DataFrame,
) -> Dict[str, float]:
    """
    Build ``lambda_*`` values matching ``encode_hyperparams`` (SSPC source) for the
    hyperparameter table's min/max normalization of nuisance SSPC means.
    """
    if channel not in _CHANNEL_TO_ID:
        raise ValueError(f"channel must be one of {list(_CHANNEL_TO_ID)}, got {channel!r}")
    ch_id = int(_CHANNEL_TO_ID[channel])
    onehot = np.zeros(5, dtype=np.float64)
    onehot[ch_id] = 1.0
    lo_a, hi_a = _SSPC_SFRA_RANGE
    lo_m, hi_m = _SSPC_MU0_RANGE
    sa = (float(sfr_a) - lo_a) / (hi_a - lo_a) if hi_a > lo_a else 0.0
    mz = (float(mu0) - lo_m) / (hi_m - lo_m) if hi_m > lo_m else 0.0
    sa = float(np.clip(sa, 0.0, 1.0))
    mz = float(np.clip(mz, 0.0, 1.0))

    out: Dict[str, float] = {f"lambda_{i}": float(onehot[i]) for i in range(5)}
    out["lambda_5"] = sa
    out["lambda_6"] = mz

    mean_cols = [f"{c}_mean" for c in _SSPC_PARAM_COLS]
    for j, mc in enumerate(mean_cols):
        if mc not in hp_encoded.columns:
            raise ValueError(f"hyperparam table missing column {mc!r}")
        lo = float(hp_encoded[mc].min())
        hi = float(hp_encoded[mc].max())
        v = float(sspc_means[mc])
        if hi > lo:
            nv = (v - lo) / (hi - lo)
        else:
            nv = 0.0
        out[f"lambda_{7 + j}"] = float(np.clip(nv, 0.0, 1.0))
    return out


def _vector_for_checkpoint(lambda_cols: List[str], by_name: Dict[str, float]) -> np.ndarray:
    return np.array([float(by_name[c]) for c in lambda_cols], dtype=np.float32)


def make_emulator_m1_compare_figure(
    *,
    output_path: Path,
    hyperparam_encoded_csv: Path,
    cfm_checkpoint: Path,
    diffusion_checkpoint: Path,
    sfr_a: float,
    mu0: float,
    sspc_means: Dict[str, float],
    n_events: int,
    seed: int,
    device: str,
) -> None:
    """
    Three stacked panels (Figure 5–style channel rows): all = CE+SMT stacked samples,
    then SMT-only, then CE-only. Each panel overlays CFM vs diffusion emulators
    (area-normalised m₁ KDE). No TNG / paper overlay.
    """
    import torch

    _05 = load_posterior_network_module()
    load_frozen_emulator = _05.load_frozen_emulator

    hp = pd.read_csv(hyperparam_encoded_csv)
    by_ce = _lambda_values_by_index(
        channel="CE",
        sfr_a=sfr_a,
        mu0=mu0,
        sspc_means=sspc_means,
        hp_encoded=hp,
    )
    by_smt = _lambda_values_by_index(
        channel="SMT",
        sfr_a=sfr_a,
        mu0=mu0,
        sspc_means=sspc_means,
        hp_encoded=hp,
    )

    dev = torch.device("cuda" if device == "cuda" and torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA requested but unavailable; using CPU.", flush=True)

    n_ce_all = int(n_events) // 2
    n_smt_all = int(n_events) - n_ce_all
    meta: Dict[str, Any] = {
        "panels": ["all_ce_smt", "SMT", "CE"],
        "all_row": {"n_ce": n_ce_all, "n_smt": n_smt_all, "n_total": int(n_events)},
        "sfr_a": sfr_a,
        "mu0": mu0,
        "sspc_means": sspc_means,
        "lambda_by_name_ce": by_ce,
        "lambda_by_name_smt": by_smt,
        "n_events_per_single_channel_row": int(n_events),
        "seed": int(seed),
        "device": str(dev),
    }

    emulators: list[tuple[str, str, Any, Any, List[str], Any]] = []
    seed_ctr = int(seed)

    def _next_seed() -> int:
        nonlocal seed_ctr
        s = seed_ctr
        seed_ctr += 1
        return s

    for kind, ckpt in (("cfm", cfm_checkpoint), ("diffusion", diffusion_checkpoint)):
        ckpt = Path(ckpt).resolve()
        if not ckpt.is_file():
            raise FileNotFoundError(f"Missing {kind} checkpoint: {ckpt}")
        model, lambda_cols, nrm = load_frozen_emulator(ckpt, dev, kind)
        if kind == "cfm":
            from models.cfm_emulator import generate_catalog as _gc
        else:
            from models.diffusion_emulator import generate_catalog as _gc
        lam_cols = list(lambda_cols)
        emulators.append((kind, str(ckpt.name), model, nrm, lam_cols, _gc))
        meta[kind] = {"checkpoint": str(ckpt), "lambda_cols": lam_cols}

    def _m1_from_cat(cat) -> np.ndarray:
        return m1_from_mchirp_q(
            cat["mchirp"].values.astype(np.float64),
            cat["q"].values.astype(np.float64),
        )

    def _kde_for_m1(m1: np.ndarray) -> np.ndarray:
        m1 = np.clip(m1.astype(np.float64), 1e-3, None)
        w = np.ones_like(m1)
        kde = sspc_weighted_kde(m1, w, x_kde=_X_KDE)
        if kde is None:
            raise RuntimeError("KDE failed (too few samples?)")
        return kde

    # panel_key -> kind -> kde
    kdes: Dict[str, Dict[str, np.ndarray]] = {k: {} for k in ("all_ce_smt", "SMT", "CE")}

    for kind, _ckname, model, nrm, lam_cols, _gc in emulators:
        lam_ce = _vector_for_checkpoint(lam_cols, by_ce)
        lam_smt = _vector_for_checkpoint(lam_cols, by_smt)
        if len(lam_ce) != len(lam_cols) or len(lam_smt) != len(lam_cols):
            raise RuntimeError(f"{kind}: lambda dim mismatch")

        # --- all: half CE + half SMT (same convention as SSPC “all” row in Figure 5) ---
        torch.manual_seed(_next_seed())
        with torch.inference_mode():
            cat_ce = _gc(lam_ce, n_ce_all, model, nrm)
        torch.manual_seed(_next_seed())
        with torch.inference_mode():
            cat_smt = _gc(lam_smt, n_smt_all, model, nrm)
        m1_all = np.concatenate([_m1_from_cat(cat_ce), _m1_from_cat(cat_smt)])
        kdes["all_ce_smt"][kind] = _kde_for_m1(m1_all)

        # --- SMT ---
        torch.manual_seed(_next_seed())
        with torch.inference_mode():
            cat = _gc(lam_smt, int(n_events), model, nrm)
        kdes["SMT"][kind] = _kde_for_m1(_m1_from_cat(cat))

        # --- CE ---
        torch.manual_seed(_next_seed())
        with torch.inference_mode():
            cat = _gc(lam_ce, int(n_events), model, nrm)
        kdes["CE"][kind] = _kde_for_m1(_m1_from_cat(cat))

    row_titles = [
        "All channels (stable + CE)",
        "Stable mass transfer (SMT)",
        "CE channel",
    ]
    row_keys = ["all_ce_smt", "SMT", "CE"]

    fig, axes = plt.subplots(3, 1, sharex=True, sharey=True, figsize=(9, 10.5))
    colors = {"cfm": "#785EF0", "diffusion": "#DC267F"}
    for ax, rtitle, pkey in zip(axes, row_titles, row_keys):
        for kind, ckname, *_ in emulators:
            kde = kdes[pkey][kind]
            ax.plot(
                _X_KDE,
                kde,
                lw=2.0,
                color=colors.get(kind, "C0"),
                label=f"{kind.upper()} ({ckname})",
            )
        ax.set_yscale("log")
        ax.set_xlim(*_EMULATOR_M1_XLIM)
        ax.set_ylim(_EMULATOR_M1_YMIN, _EMULATOR_M1_YMAX)
        ax.grid(ls=":", alpha=0.35)
        ax.legend(frameon=True, fontsize=9, loc="upper right")
        ax.set_title(rtitle, loc="left", fontsize=10)

    meta["plot_xlim"] = list(_EMULATOR_M1_XLIM)
    meta["plot_ylim"] = [_EMULATOR_M1_YMIN, _EMULATOR_M1_YMAX]
    meta["plot_ymax_note"] = "geometric_mean(1e-1, 1e0) = 10**(-0.5)"
    axes[1].set_ylabel(
        r"$\frac{d\mathcal{R}_\mathrm{emul}}{dM_{\mathrm{BH,1}}}$ "
        r"(area-normalised emulator samples)",
        fontsize=12,
    )
    axes[-1].set_xlabel(r"$M_{\mathrm{BH,1}} \ [\mathrm{M}_\odot]$", fontsize=13)
    fig.suptitle(
        "CFM vs diffusion — primary mass at fixed SSPC Λ\n"
        r"(TNG100-1-style reference SSPC; $\Lambda$ encoded like training grid)",
        fontsize=11,
        y=0.995,
    )
    txt = (
        f"rows: all = CE({n_ce_all}) + SMT({n_smt_all}); SMT/CE rows n={n_events}  |  seed={seed}\n"
        f"sfr_a={sfr_a}, mu0={mu0}\n"
        + ", ".join(f"{k.split('_mean')[0].replace('sspc_','')}={v:.4g}" for k, v in sorted(sspc_means.items()))
    )
    # Reserve bottom margin so the hyperparameter strip sits under the x-axis label, not on the curves.
    plt.tight_layout(rect=(0.02, 0.18, 0.98, 0.93))
    hp_caption = fig.text(
        0.5,
        0.02,
        txt,
        ha="center",
        va="bottom",
        fontsize=7,
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="wheat", alpha=0.35),
    )
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_path,
        dpi=160,
        bbox_inches="tight",
        bbox_extra_artists=(hp_caption,),
        pad_inches=0.12,
    )
    plt.close(fig)
    meta_path = output_path.with_suffix(".json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Saved → {output_path}", flush=True)
    print(f"Saved → {meta_path}", flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(script_path: Path | str | None = None) -> None:
    script_path = Path(script_path or _DIST_COMPARE_SCRIPT)
    parser = argparse.ArgumentParser(
        description="Compare TNG Figure 5/4 distributions with SSPC data."
    )
    parser.add_argument("--tng-data-dir", type=Path, default=_TNG_DATA_DEFAULT,
                        help="Directory containing Rate_info.h5 and COMPAS_Output_wWeights.h5")
    parser.add_argument("--sspc-hdf5", type=Path, default=_SSPC_HDF5_DEFAULT,
                        help="Path to models_sspc.hdf5")
    parser.add_argument("--compas-hdf5", type=Path, default=_COMPAS_DEFAULT,
                        help="Path to COMPAS_Output_wWeights.h5")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="If set, write all figures into this directory. If unset, use "
        "plots/<script_stem>/<timestamp>/ unless --no-timestamped-dir is given.",
    )
    parser.add_argument(
        "--no-timestamped-dir",
        action="store_true",
        help="Write under plots/<script_stem>/ without a timestamp subfolder.",
    )
    parser.add_argument("--output", type=Path, default=None,
                        help="Output path for Figure 5 (overrides --output-dir).")
    parser.add_argument("--fig4-output", "--fig6-output", dest="fig4_output", type=Path, default=None,
                        help="Output path for Figure 4/6 (overrides --output-dir).")
    parser.add_argument("--skip-fig4", "--skip-fig6", dest="skip_fig4", action="store_true",
                        help="Skip generating Figure 4 comparison plot")

    args = parser.parse_args()

    if args.output_dir is not None:
        args.output_dir = args.output_dir.resolve()
        args.output_dir.mkdir(parents=True, exist_ok=True)
        if args.output is None:
            args.output = args.output_dir / "data_distribution_analysis.png"
        if args.fig4_output is None:
            args.fig4_output = args.output_dir / "merger_rate_density_redshift.png"
    elif args.output is None and args.fig4_output is None and not args.no_timestamped_dir:
        base = plot_run_dir(script_path)
        args.output = base / "data_distribution_analysis.png"
        args.fig4_output = base / "merger_rate_density_redshift.png"
    else:
        legacy = _legacy_plot_dir(script_path)
        if args.output is None:
            args.output = legacy / "data_distribution_analysis.png"
        if args.fig4_output is None:
            args.fig4_output = args.output.parent / "merger_rate_density_redshift.png"

    print("=== BBH Primary Mass Distribution Analysis ===")
    print(f"TNG data dir : {args.tng_data_dir}")
    print(f"SSPC HDF5    : {args.sspc_hdf5}")
    print(f"COMPAS HDF5  : {args.compas_hdf5}")
    print(f"Figure 5 out : {args.output}")
    print(f"Figure 4 out : {args.fig4_output}")
    print()

    make_figure(
        tng_data_dir  = args.tng_data_dir,
        sspc_path     = args.sspc_hdf5,
        compas_path   = args.compas_hdf5,
        output_path   = args.output,
    )

    if not args.skip_fig4:
        make_figure4(
            tng_data_dir=args.tng_data_dir,
            sspc_path=args.sspc_hdf5,
            output_path=args.fig4_output,
        )


def run_emulator_m1_compare_cli(
    argv: list[str] | None = None,
    script_path: Path | str | None = None,
) -> None:
    """CLI for CFM vs diffusion m₁ comparison (``04_emulator_m1_compare.py``)."""
    import torch

    script_path = Path(script_path or _EMULATOR_M1_SCRIPT)
    parser = argparse.ArgumentParser(
        description="Area-normalised m₁ KDE: CFM vs diffusion at fixed SSPC Λ."
    )
    parser.add_argument(
        "--emulator-output",
        type=Path,
        default=None,
        help="Output PNG (default: plots/04_emulator_m1_compare/<timestamp>/emulator_m1_cfm_vs_diffusion.png).",
    )
    parser.add_argument(
        "--hyperparam-encoded-csv",
        type=Path,
        default=HYPERPARAM_TABLE_ENCODED_CSV,
    )
    parser.add_argument("--cfm-checkpoint", type=Path, default=CHECKPOINT_DIR / "cfm_final.pt")
    parser.add_argument("--diffusion-checkpoint", type=Path, default=CHECKPOINT_DIR / "diffusion_final.pt")
    parser.add_argument("--sfr-a", type=float, default=None)
    parser.add_argument("--mu0", type=float, default=None)
    parser.add_argument("--sspc-params-json", type=Path, default=None)
    parser.add_argument("--n-events", type=int, default=12000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu", "auto"])
    args = parser.parse_args(argv)

    sspc_means = dict(_DEFAULT_EMULATOR_SSPC_MEANS)
    if args.sspc_params_json is not None:
        extra = json.loads(Path(args.sspc_params_json).read_text(encoding="utf-8"))
        for k, v in extra.items():
            sspc_means[str(k)] = float(v)
    sfr_a = float(args.sfr_a) if args.sfr_a is not None else float(sspc_means["sspc_sfr_a_mean"])
    mu0 = float(args.mu0) if args.mu0 is not None else float(sspc_means["sspc_mu0_mean"])
    dev = args.device
    if dev == "auto":
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    if args.emulator_output is not None:
        out_png = Path(args.emulator_output).resolve()
    else:
        out_png = resolve_plot_output(
            script_path,
            filename="emulator_m1_cfm_vs_diffusion.png",
        )
    make_emulator_m1_compare_figure(
        output_path=out_png,
        hyperparam_encoded_csv=Path(args.hyperparam_encoded_csv).resolve(),
        cfm_checkpoint=Path(args.cfm_checkpoint).resolve(),
        diffusion_checkpoint=Path(args.diffusion_checkpoint).resolve(),
        sfr_a=sfr_a,
        mu0=mu0,
        sspc_means=sspc_means,
        n_events=int(args.n_events),
        seed=int(args.seed),
        device=str(dev),
    )
