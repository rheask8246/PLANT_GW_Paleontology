#!/usr/bin/env python3
"""
GWTC-4 Figure 1 style validation plot for PLANT emulators.

Recreates the **formatting** of `gwtc4/o4a-astro/figure_scripts/figure_1.py`:
- two main axes (center panels) with triangle-clipped heatmaps
- left skinny colorbar axis for "Fractional uncertainty" (upper triangle)
- right skinny colorbar axis for "rate density" (lower triangle)
- log-scaled tick labels shown as 1, 3, 10, 30, 100 on both axes

Instead of popsummary population posteriors, we estimate an emulator-induced
population rate density on (m1, m2) by:
- sampling hyperparameter rows (Λ) with probability ∝ per-row intrinsic rate
- drawing synthetic events from the emulator at that Λ
- converting (mchirp, q) → (m1, m2)
- accumulating a weighted histogram in (ln m1, ln m2)

Uncertainty is an empirical fractional uncertainty estimated from repeated
stochastic draws (bootstrap over RNG seeds), reported per pixel as:
    frac_uncert = std(rate_pixel) / mean(rate_pixel)

This is **not** the GWTC-4 population posterior uncertainty; it is a
model-output uncertainty proxy that still matches the paper figure’s visual
grammar exactly.

**Matching the paper’s smooth, full-triangle look (usually no retraining):**
GWTC ``figure_1.py`` reads dense HDF5 grids; emulators only provide samples, so
you need either **much more Monte Carlo** (same checkpoints, longer reruns of
this script) or **post-bin smoothing** plus **auto color limits**. Use
``--paper-quality`` for a sensible bundle, or tune ``--n-rows``,
``--n-events-per-row``, ``--smooth-sigma``, ``--auto-rate-limits``. Retrain
04/04b only if the checkpoint is an undertrained smoke model and the mass
distribution is still wrong, not for smoothness alone.

LaTeX text rendering is **on by default** (``text.usetex``) when a quick probe
succeeds; pass ``--no-tex`` to force matplotlib mathtext only (e.g. broken TeX on a node).
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
    CHECKPOINT_DIR,
    HYPERPARAM_TABLE_ENCODED_CSV,
    PROJECT_ROOT,
    REPO_ROOT,
    ensure_paths,
    load_posterior_network_module,
    ml_data_dir,
    plot_run_dir,
)

ensure_paths()

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _resolve_device(s: str) -> str:
    """``auto`` → CUDA if available, else CPU (matches typical Slurm GPU nodes)."""
    key = str(s).lower().strip()
    if key in ("auto", ""):
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    return str(s).strip()


def _grid_rate_column(hp_cols: List[str]) -> str:
    # Prefer intrinsic total, else pdet proxy.
    return "sum_weight" if "sum_weight" in hp_cols else "sum_pdet"


def _m1_from_mchirp_q(mchirp: np.ndarray, q: np.ndarray) -> np.ndarray:
    # Same map as PLANT_GW_Paleontology/data_distribution_analysis.py
    q = np.clip(q.astype(np.float64), 1e-6, 1.0)
    mc = np.clip(mchirp.astype(np.float64), 1e-6, None)
    return mc * ((1.0 + q) / np.power(q, 3.0)) ** 0.2


def _m1m2_from_catalog(cat: "np.ndarray | Dict[str, np.ndarray] | Any") -> Tuple[np.ndarray, np.ndarray]:
    # cat is expected to be a pandas DataFrame returned by generate_catalog()
    mchirp = np.asarray(cat["mchirp"].values if hasattr(cat["mchirp"], "values") else cat["mchirp"])
    q = np.asarray(cat["q"].values if hasattr(cat["q"], "values") else cat["q"])
    m1 = _m1_from_mchirp_q(mchirp, q)
    m2 = np.clip(q, 0.0, 1.0) * m1
    # enforce ordering m1 >= m2
    swap = m2 > m1
    if np.any(swap):
        m1s = m1.copy()
        m1[swap] = m2[swap]
        m2[swap] = m1s[swap]
    return m1.astype(np.float64), m2.astype(np.float64)


@dataclass(frozen=True)
class RateGrid:
    ln_edges: np.ndarray  # (nbins+1,)
    rate: np.ndarray  # (nbins, nbins)
    frac_unc: np.ndarray  # (nbins, nbins)


def _load_emulator(ckpt_path: Path, device: str, kind: str):
    """
    Load frozen emulator using the canonical loader from 05_posterior_network.py
    so we match training-time checkpoint conventions.
    """
    import torch

    _05 = load_posterior_network_module()
    emulator, lambda_cols, nrm = _05.load_frozen_emulator(ckpt_path, torch.device(device), kind)
    return emulator, lambda_cols, nrm


def _generate_catalog(emulator, kind: str, lambda_vec: np.ndarray, n_events: int, normalizer: Dict[str, Any]):
    if kind == "cfm":
        from models.cfm_emulator import generate_catalog
    elif kind == "diffusion":
        from models.diffusion_emulator import generate_catalog
    else:
        raise ValueError(f"Unknown emulator kind {kind!r}")
    return generate_catalog(np.asarray(lambda_vec, dtype=np.float32), int(n_events), emulator, normalizer)


def _estimate_rate_grid(
    *,
    hp_df,
    lambda_cols: List[str],
    emulator,
    emulator_kind: str,
    normalizer: Dict[str, Any],
    nbins: int,
    mmax: float,
    n_rows_per_boot: int,
    n_events_per_row: int,
    n_boot: int,
    seed: int,
) -> RateGrid:
    # Build log-space bin edges matching the plot extent.
    ln_edges = np.linspace(np.log(1.0), np.log(float(mmax)), int(nbins) + 1)
    ln_centers = 0.5 * (ln_edges[:-1] + ln_edges[1:])
    _ = ln_centers  # reserved for potential future bin-area corrections

    # Sampling distribution over Λ rows.
    rate_col = _grid_rate_column(list(hp_df.columns))
    if rate_col not in hp_df.columns:
        raise ValueError(f"Could not find per-row rate column ({rate_col!r}) in hyperparam table.")
    row_rate = np.asarray(hp_df[rate_col].values, dtype=np.float64)
    row_rate = np.clip(row_rate, 0.0, None)
    if not np.isfinite(row_rate).all() or row_rate.sum() <= 0:
        raise ValueError(f"Invalid or zero total rate in column {rate_col!r}.")
    p_row = row_rate / row_rate.sum()

    import torch

    rng = np.random.default_rng(int(seed))
    rate_draws = np.zeros((int(n_boot), int(nbins), int(nbins)), dtype=np.float64)

    t0 = time.time()
    # Inference-only: slightly less overhead than no_grad(); same numerics for this sampling loop.
    with torch.inference_mode():
        for b in range(int(n_boot)):
            b0 = time.time()
            if int(n_boot) > 1:
                print(f"[rate_grid] bootstrap {b + 1}/{int(n_boot)} ...", flush=True)

            # Independent stochastic draw per bootstrap (new RNG stream).
            rng_b = np.random.default_rng(int(seed) + 100_000 * (b + 1))
            # Sample Λ rows proportional to intrinsic rate; allow repeats.
            rows = rng_b.choice(len(hp_df), size=int(n_rows_per_boot), replace=True, p=p_row)

            H = np.zeros((int(nbins), int(nbins)), dtype=np.float64)
            for i_row, ri in enumerate(rows):
                lam = hp_df.iloc[int(ri)][lambda_cols].values.astype(np.float32)
                # For reproducibility across emulators, seed torch per row per boot.
                try:
                    torch.manual_seed(int(seed) + 13_337 * (b + 1) + 97 * int(ri))
                except Exception:
                    pass

                cat = _generate_catalog(emulator, emulator_kind, lam, n_events_per_row, normalizer)
                m1, m2 = _m1m2_from_catalog(cat)

                # Convert to ln and filter range.
                ln1 = np.log(np.clip(m1, 1.0, mmax))
                ln2 = np.log(np.clip(m2, 1.0, mmax))

                # Weight events so total weight contributed by this Λ row is its intrinsic rate.
                w_evt = float(row_rate[int(ri)]) / max(int(n_events_per_row), 1)
                w = np.full_like(ln1, w_evt, dtype=np.float64)

                # Histogram into ln bins.
                h, _, _ = np.histogram2d(ln1, ln2, bins=(ln_edges, ln_edges), weights=w)
                H += h

                # Periodic progress to keep Slurm logs alive and provide some feedback.
                if (i_row + 1) % 64 == 0 or (i_row + 1) == len(rows):
                    print(
                        f"[rate_grid]  row {i_row + 1}/{len(rows)} (boot {b + 1}/{int(n_boot)})",
                        flush=True,
                    )

            # Convert sum of weights per bin to a density per dlnm1 dlnm2.
            # Since bins are uniform in ln, divide by (Δln)^2 so values are comparable across nbins.
            dln = float(ln_edges[1] - ln_edges[0])
            H = H / max(dln * dln, 1e-12)
            rate_draws[b] = H

            if int(n_boot) > 1:
                b_elapsed = time.time() - b0
                total_elapsed = time.time() - t0
                boots_left = int(n_boot) - (b + 1)
                eta = boots_left * (total_elapsed / max(b + 1, 1))
                print(
                    f"[rate_grid] done bootstrap {b + 1}/{int(n_boot)} in {b_elapsed:.1f}s "
                    f"(elapsed {total_elapsed/60:.1f} min, ETA {eta/60:.1f} min)",
                    flush=True,
                )

    rate_mean = np.mean(rate_draws, axis=0)
    rate_std = np.std(rate_draws, axis=0, ddof=1) if int(n_boot) > 1 else np.zeros_like(rate_mean)
    frac_unc = rate_std / np.clip(rate_mean, 1e-30, None)

    # Symmetrize like the original script (U + U^T - diag), but here we do it
    # for both mean and uncertainty so the triangles look clean.
    rate_mean = 0.5 * (rate_mean + rate_mean.T)
    frac_unc = 0.5 * (frac_unc + frac_unc.T)
    # Ensure no NaNs.
    rate_mean = np.nan_to_num(rate_mean, nan=0.0, posinf=0.0, neginf=0.0)
    frac_unc = np.nan_to_num(frac_unc, nan=0.0, posinf=0.0, neginf=0.0)

    return RateGrid(ln_edges=ln_edges, rate=rate_mean, frac_unc=frac_unc)


def _ln_bin_centers(ln_edges: np.ndarray) -> np.ndarray:
    return 0.5 * (ln_edges[:-1] + ln_edges[1:])


def _m1m2_scale_matrix(ln_edges: np.ndarray) -> np.ndarray:
    """Match ``figure_1.py`` post-process ``R *= np.outer(m1, m2)`` on bin centers."""
    ln_c = _ln_bin_centers(ln_edges)
    m = np.exp(ln_c)
    return np.outer(m, m)


def _lower_triangle_mask(n: int) -> np.ndarray:
    """True where m2-bin index <= m1-bin index (lower triangle incl. diagonal)."""
    i = np.arange(n)[:, None]
    j = np.arange(n)[None, :]
    return i >= j


def _upper_triangle_mask(n: int) -> np.ndarray:
    """True where m2-bin index >= m1-bin index (upper triangle incl. diagonal)."""
    i = np.arange(n)[:, None]
    j = np.arange(n)[None, :]
    return i <= j


def _smooth_positive_log_field(z: np.ndarray, sigma: float) -> np.ndarray:
    """Multiplicative smoothing: Gaussian filter on log10(z), then clip."""
    if sigma <= 0:
        return z
    try:
        from scipy.ndimage import gaussian_filter
    except ImportError as e:
        raise ImportError("Smoothing requires scipy (e.g. pip install scipy).") from e
    eps = np.finfo(np.float64).tiny * 1e6
    logz = np.log10(np.maximum(z.astype(np.float64), eps))
    logz_s = gaussian_filter(logz, sigma=float(sigma), mode="reflect")
    out = np.power(10.0, logz_s)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def _smooth_linear_field(z: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return z
    try:
        from scipy.ndimage import gaussian_filter
    except ImportError as e:
        raise ImportError("Smoothing requires scipy (e.g. pip install scipy).") from e
    return np.nan_to_num(
        gaussian_filter(z.astype(np.float64), sigma=float(sigma), mode="reflect"),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )


def _auto_log_limits_positive(
    z: np.ndarray,
    mask: np.ndarray,
    q_lo: float,
    q_hi: float,
    floor: float,
) -> tuple[float, float]:
    vals = z[mask]
    vals = vals[np.isfinite(vals) & (vals > float(floor))]
    if vals.size < 10:
        return float(floor), float(max(np.max(vals), float(floor) * 10))
    lo = float(np.quantile(vals, q_lo))
    hi = float(np.quantile(vals, q_hi))
    lo = max(lo, float(floor))
    hi = max(hi, lo * 10)
    return lo, hi


def _postprocess_grid_for_display(
    grid: RateGrid,
    *,
    smooth_sigma: float,
    smooth_unc_sigma: float,
    fullpop_m1m2_scale: bool,
) -> RateGrid:
    """
    Optional Jacobian-style scaling (``figure_1.py`` ``R *= outer(m1,m2)`` on
    bin centers) and Gaussian smoothing for paper-like fields.
    """
    ln = grid.ln_edges
    r = np.array(grid.rate, dtype=np.float64, copy=True)
    u = np.array(grid.frac_unc, dtype=np.float64, copy=True)
    if fullpop_m1m2_scale:
        r *= _m1m2_scale_matrix(ln)
    if smooth_sigma > 0:
        r = _smooth_positive_log_field(r, smooth_sigma)
    if smooth_unc_sigma > 0:
        u = _smooth_linear_field(u, smooth_unc_sigma)
    u = np.clip(u, 1e-6, None)
    r = np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)
    u = np.nan_to_num(u, nan=1e-6, posinf=1e6, neginf=1e-6)
    return RateGrid(ln_edges=ln, rate=r, frac_unc=u)


def _matplotlib_usetex_works() -> bool:
    """
    True if matplotlib can render at least one usetex string via the Agg backend.
    Incomplete TeX installs (common on clusters / TinyTeX) fail here instead of at tight_layout.
    """
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    prev = bool(plt.rcParams["text.usetex"])
    try:
        plt.rcParams["text.usetex"] = True
        fig = Figure(figsize=(0.6, 0.6))
        canvas = FigureCanvasAgg(fig)
        ax = fig.subplots()
        ax.axis("off")
        ax.text(0.5, 0.5, r"$\mathrm{d}x$", usetex=True, ha="center", va="center")
        canvas.draw()
        return True
    except Exception:
        return False
    finally:
        plt.rcParams["text.usetex"] = prev


def _ensure_agg_backend() -> None:
    """Force a non-interactive matplotlib backend (Slurm-safe)."""
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
    except Exception:
        pass


def _import_popsummary() -> Any:
    """Lazy import so PLANT-only runs don't require popsummary installed."""
    try:
        import popsummary  # type: ignore

        return popsummary
    except Exception as e:
        raise ImportError(
            "popsummary is required for GWTC-4 paper overlays.\n"
            "Install it in your environment and re-run (e.g. pip install popsummary)."
        ) from e


def _load_gwtc4_figure1_grids(
    data_release_dir: Path,
    *,
    mmax: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load paper Figure-1 grids from popsummary.

    Returns (m1, m2, full_R_scaled, full_U, bgp_R, bgp_U) truncated to m<=mmax.
    """
    popsummary = _import_popsummary()
    dr = Path(data_release_dir).expanduser().resolve()
    pdb_file = dr / "AllCBC_FullPop.h5"
    bgp_file = dr / "AllCBC_FullPopBGP.h5"
    if not pdb_file.exists() or not bgp_file.exists():
        raise FileNotFoundError(
            "Missing expected GWTC-4 data_release files:\n"
            f"- {pdb_file}\n- {bgp_file}\n"
            "Point --gwtc4-data-release to the folder that contains these files."
        )

    pdb_result = popsummary.popresult.PopulationResult(fname=str(pdb_file))
    bgp_result = popsummary.popresult.PopulationResult(fname=str(bgp_file))

    (m1, m2), full_R = pdb_result.get_rates_on_grids("primary_mass_secondary_mass_joint_median")
    (_, _), full_U = pdb_result.get_rates_on_grids("primary_mass_secondary_mass_joint_uncertainty")
    full_U = np.nan_to_num(full_U)
    full_U = full_U + full_U.T - np.diag(np.diag(full_U))

    (_, _), bgp_U = bgp_result.get_rates_on_grids("uncert_ppd_primary_and_secondary_mass")
    (bgp_m1, bgp_m2), bgp_R = bgp_result.get_rates_on_grids("ppd_primary_and_secondary_mass")

    max_ind = int(np.digitize(float(mmax), m1)) + 1
    m1 = m1[:max_ind]
    m2 = m2[:max_ind]
    full_R = full_R[:max_ind, :max_ind]
    full_U = full_U[:max_ind, :max_ind]

    # Match paper display: R *= outer(m1,m2)
    full_R = full_R * np.outer(m1, m2)

    if bgp_R.shape[0] >= max_ind and bgp_R.shape[1] >= max_ind:
        bgp_R = bgp_R[:max_ind, :max_ind]
        bgp_U = bgp_U[:max_ind, :max_ind]
        bgp_m1 = bgp_m1[:max_ind]
        bgp_m2 = bgp_m2[:max_ind]
        _ = (bgp_m1, bgp_m2)

    return m1, m2, full_R, full_U, bgp_R, bgp_U


def _plot_figure1_compare(
    *,
    out_path: Path,
    mmax: float,
    vmin: float,
    vmax: float,
    uncmin: float,
    uncmax: float,
    usetex: bool,
    compare_mode: str,
    paper_fullpop: Optional[tuple[np.ndarray, np.ndarray]] = None,
    paper_bgp: Optional[tuple[np.ndarray, np.ndarray]] = None,
    plant_cfm: Optional[RateGrid] = None,
    plant_diff: Optional[RateGrid] = None,
) -> None:
    """
    Figure-1 style plot with paper panels (FullPop/BGP) and PLANT panels (CFM/Diffusion).

    compare_mode:
      - panels: 4 side-by-side panels when both are available.
      - overlay: paper panels with PLANT rate contours (quick visual check).
    """
    _ensure_agg_backend()
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from matplotlib.patches import Polygon

    if compare_mode not in ("panels", "overlay"):
        raise ValueError("compare_mode must be 'panels' or 'overlay'")

    use_tex = bool(usetex) and _matplotlib_usetex_works()
    plt.rc("text", usetex=use_tex)
    plt.rc("font", family="serif", size=13)

    have_paper = paper_fullpop is not None and paper_bgp is not None
    have_plant = plant_cfm is not None and plant_diff is not None

    if compare_mode == "overlay" and not (have_paper and have_plant):
        raise ValueError("overlay mode requires both paper grids and PLANT grids.")

    if have_paper and have_plant:
        titles = [r"\textsc{FullPop}-4.0", "BGP", r"\textsc{CFM}", r"\textsc{Diffusion}"]
        panels: list[tuple[np.ndarray, np.ndarray]] = [
            (paper_fullpop[0], paper_fullpop[1]),
            (paper_bgp[0], paper_bgp[1]),
            (plant_cfm.rate, plant_cfm.frac_unc.T),
            (plant_diff.rate, plant_diff.frac_unc.T),
        ]
    elif have_paper:
        titles = [r"\textsc{FullPop}-4.0", "BGP"]
        panels = [(paper_fullpop[0], paper_fullpop[1]), (paper_bgp[0], paper_bgp[1])]
    elif have_plant:
        titles = [r"\textsc{CFM}", r"\textsc{Diffusion}"]
        panels = [(plant_cfm.rate, plant_cfm.frac_unc.T), (plant_diff.rate, plant_diff.frac_unc.T)]
    else:
        raise ValueError("Nothing to plot (need paper and/or PLANT inputs).")

    if compare_mode == "overlay":
        titles = [r"\textsc{FullPop}-4.0 + PLANT contours", "BGP + PLANT contours"]
        panels = [(paper_fullpop[0], paper_fullpop[1]), (paper_bgp[0], paper_bgp[1])]

    n_center = len(panels)
    fig_w = 10 if n_center <= 2 else 14
    fig, axes = plt.subplots(
        ncols=n_center + 2,
        figsize=(fig_w, 4),
        width_ratios=[0.075] + [1] * n_center + [0.075],
    )
    cax_unc = axes[0]
    cax_rate = axes[-1]
    axs = list(axes[1:-1])

    majors = [1, 3, 10, 30, 100]
    major_names = ["1", "3", "10", "30", "100"]
    xticks = np.log(np.concatenate((np.arange(1, 10), np.arange(10, 110, 10))))
    extent = (0.0, float(np.log(mmax)), 0.0, float(np.log(mmax)))

    last_rate = None
    last_unc = None
    for ax, (R, U), title in zip(axs, panels, titles):
        im_rate = ax.imshow(
            R,
            cmap="Blues",
            norm=LogNorm(vmin=float(vmin), vmax=float(vmax)),
            origin="lower",
            extent=extent,
        )
        im_unc = ax.imshow(
            U,
            cmap="Reds",
            norm=LogNorm(vmin=float(uncmin), vmax=float(uncmax)),
            origin="lower",
            extent=extent,
        )
        last_rate = im_rate
        last_unc = im_unc

        im_rate.set_clip_path(
            Polygon(
                np.array([[np.log(1), np.log(1)], [np.log(mmax), np.log(1)], [np.log(mmax), np.log(mmax)]]),
                closed=True,
                transform=ax.transData,
            )
        )
        im_unc.set_clip_path(
            Polygon(
                np.array([[np.log(1), np.log(1)], [np.log(1), np.log(mmax)], [np.log(mmax), np.log(mmax)]]),
                closed=True,
                transform=ax.transData,
            )
        )

        ax.set_xticks(xticks, minor=True)
        ax.set_yticks(xticks, minor=True)
        ax.set_xticks(np.log(np.array(majors)), major_names, minor=False)
        ax.set_yticks(np.log(np.array(majors)), major_names, minor=False)
        ax.set_xlabel(r"$m_1$ [$M_\odot$]")
        ax.grid(ls=":", alpha=0.5, lw=1, color="k")
        ax.set_title(title)

    axs[0].set_ylabel(r"$m_2$ [$M_\odot$]")

    if compare_mode == "overlay" and have_plant:
        # Contours on top of the paper maps (rate only).
        levels = np.logspace(np.log10(max(float(vmin), 1e-30)), np.log10(float(vmax)), 6)
        axs[0].contour(plant_cfm.rate, levels=levels, colors="gold", linewidths=0.8, alpha=0.8, origin="lower", extent=extent)
        axs[1].contour(plant_diff.rate, levels=levels, colors="gold", linewidths=0.8, alpha=0.8, origin="lower", extent=extent)

    if last_rate is None or last_unc is None:
        raise RuntimeError("No images drawn.")

    n_order = int(np.log10(float(vmax) / float(vmin)) + 0.01)
    cbar_rate = fig.colorbar(last_rate, cax=cax_rate, ticks=np.logspace(np.log10(vmin), np.log10(vmax), n_order + 1))
    cbar_unc = fig.colorbar(last_unc, cax=cax_unc)

    tick_spacing = 2
    tick_labels = []
    for i, x in enumerate(np.linspace(np.log10(vmin), np.log10(vmax), n_order + 1, dtype=int)):
        tick_labels.append(f"$10^{{{x}}}$" if i % tick_spacing == 0 else "")
    cbar_rate.ax.set_yticklabels(tick_labels)
    cbar_rate.set_label(
        r"$\frac{\mathrm{d}\mathcal{R}}{\mathrm{d}(\ln m_1)\mathrm{d}(\ln m_2)}$ [Gpc${}^{-3}$ yr${}^{-1}$]",
        rotation=90,
    )
    cbar_unc.set_label("Fractional uncertainty", rotation=90)
    cbar_rate.ax.yaxis.set_label_position("left")
    cbar_unc.ax.yaxis.set_label_position("left")

    plt.tight_layout(pad=0.0, w_pad=0, h_pad=1.0)

    for ax in (cax_unc, cax_rate):
        pos = ax.get_position()
        yscale = pos.y1 - pos.y0
        w = 0.051
        ax.set_position([pos.x0 * 0.95, pos.y0 + yscale * w, pos.x1 - pos.x0, yscale * (1 - 2 * w)])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close(fig)


def _estimate_1d_marginals_from_emulator(
    *,
    hp_df,
    lambda_cols: List[str],
    emulator,
    emulator_kind: str,
    normalizer: Dict[str, Any],
    m_edges: np.ndarray,
    n_rows_per_boot: int,
    n_events_per_row: int,
    n_boot: int,
    seed: int,
    m_clip_hi: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (m_centers, dRdm1_draws, dRdm2_draws), each (n_boot, nbins) except centers."""
    rate_col = _grid_rate_column(list(hp_df.columns))
    if rate_col not in hp_df.columns:
        raise ValueError(f"Missing per-row rate column {rate_col!r} in hyperparameter table.")
    row_rate = np.asarray(hp_df[rate_col].values, dtype=np.float64)
    row_rate = np.clip(row_rate, 0.0, None)
    if row_rate.sum() <= 0:
        raise ValueError(f"Invalid total rate in {rate_col!r}.")
    p_row = row_rate / row_rate.sum()

    nbins = len(m_edges) - 1
    m_centers = 0.5 * (m_edges[:-1] + m_edges[1:])

    import torch

    d1 = np.zeros((int(n_boot), nbins), dtype=np.float64)
    d2 = np.zeros((int(n_boot), nbins), dtype=np.float64)
    for b in range(int(n_boot)):
        rng_b = np.random.default_rng(int(seed) + 100_000 * (b + 1))
        rows = rng_b.choice(len(hp_df), size=int(n_rows_per_boot), replace=True, p=p_row)
        h1 = np.zeros(nbins, dtype=np.float64)
        h2 = np.zeros(nbins, dtype=np.float64)
        with torch.inference_mode():
            for ri in rows:
                lam = hp_df.iloc[int(ri)][lambda_cols].values.astype(np.float32)
                try:
                    torch.manual_seed(int(seed) + 13_337 * (b + 1) + 97 * int(ri))
                except Exception:
                    pass
                cat = _generate_catalog(emulator, emulator_kind, lam, int(n_events_per_row), normalizer)
                m1, m2 = _m1m2_from_catalog(cat)
                m1 = np.clip(m1, 1.0, float(m_clip_hi))
                m2 = np.clip(m2, 1.0, float(m_clip_hi))
                w_evt = float(row_rate[int(ri)]) / max(int(n_events_per_row), 1)
                w = np.full_like(m1, w_evt, dtype=np.float64)
                h1 += np.histogram(m1, bins=m_edges, weights=w)[0]
                h2 += np.histogram(m2, bins=m_edges, weights=w)[0]
        widths = np.diff(m_edges).astype(np.float64)
        d1[b] = h1 / np.clip(widths, 1e-12, None)
        d2[b] = h2 / np.clip(widths, 1e-12, None)

    return m_centers, d1, d2


def _plot_figure2_compare(
    *,
    out_path: Path,
    data_release_dir: Optional[Path],
    plant_cfm_marg: Optional[tuple[np.ndarray, np.ndarray, np.ndarray]],
    plant_diff_marg: Optional[tuple[np.ndarray, np.ndarray, np.ndarray]],
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    usetex: bool,
) -> None:
    _ensure_agg_backend()
    import matplotlib.pyplot as plt

    use_tex = bool(usetex) and _matplotlib_usetex_works()
    plt.rc("text", usetex=use_tex)
    plt.rc("font", family="serif", size=13)

    fig, axes = plt.subplots(nrows=2, figsize=(10, 8))
    color_full = "#648FFF"
    color_bgp = "#FE6100"
    color_cfm = "#785EF0"
    color_dif = "#DC267F"

    if data_release_dir is not None:
        popsummary = _import_popsummary()
        dr = Path(data_release_dir).expanduser().resolve()
        pdb_file = dr / "AllCBC_FullPop.h5"
        bgp_file = dr / "AllCBC_FullPopBGP.h5"
        pdb_result = popsummary.popresult.PopulationResult(fname=str(pdb_file))
        bgp_result = popsummary.popresult.PopulationResult(fname=str(bgp_file))
        mass_key = ["primary_mass", "secondary_mass"]
        for ii in range(2):
            ax = axes[ii]
            pdb_m, pdb_Rm = pdb_result.get_rates_on_grids(mass_key[ii])
            bgp_m, bgp_Rm = bgp_result.get_rates_on_grids(mass_key[ii])
            ax.fill_between(pdb_m[0], np.percentile(pdb_Rm, 5, axis=0), np.percentile(pdb_Rm, 95, axis=0), color=color_full, alpha=0.25, rasterized=True)
            ax.fill_between(bgp_m[0], np.percentile(bgp_Rm, 5, axis=0), np.percentile(bgp_Rm, 95, axis=0), color=color_bgp, alpha=0.25, rasterized=True)
            ax.plot(pdb_m[0], np.mean(pdb_Rm, axis=0), color=color_full, label=r"\textsc{FullPop}-4.0")
            ax.plot(bgp_m[0], np.mean(bgp_Rm, axis=0), color=color_bgp, label="BGP")

    def _plot_plant(ax, marg, color, label, which: str) -> None:
        if marg is None:
            return
        m, d1, d2 = marg
        y = d1 if which == "m1" else d2
        ax.fill_between(m, np.percentile(y, 5, axis=0), np.percentile(y, 95, axis=0), color=color, alpha=0.18, rasterized=True)
        ax.plot(m, np.mean(y, axis=0), color=color, lw=1.4, label=label)

    _plot_plant(axes[0], plant_cfm_marg, color_cfm, "PLANT CFM", "m1")
    _plot_plant(axes[0], plant_diff_marg, color_dif, "PLANT Diffusion", "m1")
    _plot_plant(axes[1], plant_cfm_marg, color_cfm, "PLANT CFM", "m2")
    _plot_plant(axes[1], plant_diff_marg, color_dif, "PLANT Diffusion", "m2")

    for ii in range(2):
        ax = axes[ii]
        ax.set_yscale("log")
        ax.set_ylim(float(ylim[0]), float(ylim[1]))
        ax.set_xlim(float(xlim[0]), float(xlim[1]))
        ax.grid(ls=":", alpha=0.2, lw=1, color="k")
        if ii == 0:
            ax.set_ylabel(r"$\mathrm{d}\mathcal{R}/\mathrm{d}m_1$ [Gpc${}^{-3}$ yr${}^{-1} M_\odot^{-1}$]")
            ax.set_xlabel(r"$m_1$ [$M_\odot$]")
        else:
            ax.set_ylabel(r"$\mathrm{d}\mathcal{R}/\mathrm{d}m_2$ [Gpc${}^{-3}$ yr${}^{-1} M_\odot^{-1}$]")
            ax.set_xlabel(r"$m_2$ [$M_\odot$]")

    axes[0].legend(frameon=True, loc="upper right")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_figure3_compare(
    *,
    out_path: Path,
    data_release_dir: Optional[Path],
    gwtc3_dir: Optional[Path],
    plant_marg: Optional[tuple[np.ndarray, np.ndarray, np.ndarray]],
    usetex: bool,
) -> None:
    _ensure_agg_backend()
    import matplotlib.pyplot as plt

    use_tex = bool(usetex) and _matplotlib_usetex_works()
    plt.rc("text", usetex=use_tex)
    plt.rc("font", family="serif", size=13)

    fig = plt.figure(figsize=(11, 4.5), tight_layout=True)
    ax = plt.subplot(111)
    plt.subplots_adjust(bottom=0.2)

    pf = None
    try:
        # Import the paper's helper if present in this repo.
        fig_script_dir = REPO_ROOT / "gwtc4" / "o4a-astro" / "figure_scripts"
        if fig_script_dir.exists() and str(fig_script_dir) not in sys.path:
            sys.path.insert(0, str(fig_script_dir))
        import plot_funcs_bbh_mass as pf  # type: ignore

        pf.setup()
        pf.setup_mass_plot(ax, grid_kwargs=dict(ls="dotted", color="k", alpha=0), xrange=(2, 100), yrange=(1e-3, 40))
    except Exception:
        pf = None
        ax.grid(ls=":", alpha=0.2)
        ax.set_yscale("log")
        ax.set_xlim(2, 100)
        ax.set_ylim(1e-3, 40)

    if data_release_dir is not None and pf is not None:
        popsummary = _import_popsummary()
        dr = Path(data_release_dir).expanduser().resolve()
        bptp = dr / "BBHMassSpinRedshift_BrokenPowerLawTwoPeaks_GaussianComponentSpins_PowerLawRedshift.h5"
        bs = dr / "BBHMassSpinRedshift_BSplineIID.h5"
        if bptp.exists() and bs.exists():
            bptp_o4 = popsummary.popresult.PopulationResult(str(bptp))
            bs_o4 = popsummary.popresult.PopulationResult(str(bs))
            bptp_m1, bptp_pdfs = pf.get_params(bptp_o4, "mass_1")
            bs_m1, bs_pdfs = pf.get_params(bs_o4, "rate_vs_mass_1_at_z0-2", rate=False)
            pf.plot_90CI(ax, bs_m1, bs_pdfs, color="#648FFF", label=r"\textsc{B-Spline}, \textsc{GWTC-4.0}", fill_alpha=0.35)
            pf.plot_90CI(ax, bptp_m1, bptp_pdfs, color="#FE6100", label=r"\textsc{Broken Power Law + 2 Peaks}, \textsc{GWTC-4.0}", fill_alpha=0.35)

    if gwtc3_dir is not None and pf is not None:
        try:
            g3 = Path(gwtc3_dir).expanduser().resolve()
            plp_m1, plplow, plppd, plphi = pf.get_03b_plp_ppds(str(g3))
            ax.plot(plp_m1, plppd, color="k", lw=1.5, alpha=0.5, ls="-")
            ax.plot(plp_m1, plplow, color="k", lw=0.75, alpha=0.7, ls="--", label=r"\textsc{Power Law + Peak}, \textsc{GWTC-3.0}")
            ax.plot(plp_m1, plphi, color="k", lw=0.75, alpha=0.7, ls="--")
        except Exception:
            pass

    if plant_marg is not None:
        m, d1, _d2 = plant_marg
        lo = np.percentile(d1, 5, axis=0)
        hi = np.percentile(d1, 95, axis=0)
        mu = np.mean(d1, axis=0)
        ax.fill_between(m, lo, hi, color="#DC267F", alpha=0.18, label="PLANT (bootstrap band)")
        ax.plot(m, mu, color="#DC267F", lw=1.5)

    ax.set_xlabel(r"$m_1$ [$M_\odot$]")
    ax.grid(ls=":", alpha=0.2)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, frameon=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

def _apply_figure1_style_and_save(
    *,
    out_path: Path,
    grid_left: RateGrid,
    grid_right: RateGrid,
    left_title: str,
    right_title: str,
    mmax: float,
    vmin: float,
    vmax: float,
    uncmin: float,
    uncmax: float,
    usetex: bool,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from matplotlib.patches import Polygon

    # rc(text, usetex=True) does not validate LaTeX; tight_layout/save can still crash (e.g. missing type1cm.sty).
    use_tex = bool(usetex) and _matplotlib_usetex_works()
    if use_tex:
        plt.rc("text", usetex=True)
    else:
        plt.rc("text", usetex=False)
        if usetex and ("\\textsc" in str(left_title) or "\\textsc" in str(right_title)):
            print(
                "[plot] LaTeX/usetex unavailable or incomplete; using matplotlib mathtext "
                "(titles CFM / Diffusion instead of \\textsc).",
                flush=True,
            )
            if "\\textsc" in str(left_title):
                left_title = "CFM"
            if "\\textsc" in str(right_title):
                right_title = "Diffusion"
    plt.rc("font", family="serif", size=13)

    fig, axes = plt.subplots(
        ncols=4,
        figsize=(10, 4),
        width_ratios=[0.075, 1, 1, 0.075],
    )

    majors = [1, 3, 10, 30, 100]
    major_names = ["1", "3", "10", "30", "100"]
    xticks = np.log(np.concatenate((np.arange(1, 10), np.arange(10, 110, 10))))

    left_ax = axes[1]
    right_ax = axes[2]

    # Note: imshow extent is (xmin, xmax, ymin, ymax) in data coords.
    extent = (0.0, float(np.log(mmax)), 0.0, float(np.log(mmax)))

    im_left_rate = left_ax.imshow(
        grid_left.rate,
        cmap="Blues",
        norm=LogNorm(vmin=float(vmin), vmax=float(vmax)),
        origin="lower",
        extent=extent,
    )
    im_right_rate = right_ax.imshow(
        grid_right.rate,
        cmap="Blues",
        norm=LogNorm(vmin=float(vmin), vmax=float(vmax)),
        origin="lower",
        extent=extent,
    )

    im_left_unc = left_ax.imshow(
        grid_left.frac_unc.T,
        cmap="Reds",
        norm=LogNorm(vmin=float(uncmin), vmax=float(uncmax)),
        origin="lower",
        extent=extent,
    )
    im_right_unc = right_ax.imshow(
        grid_right.frac_unc.T,
        cmap="Reds",
        norm=LogNorm(vmin=float(uncmin), vmax=float(uncmax)),
        origin="lower",
        extent=extent,
    )

    # Clip lower-triangle for rate, upper-triangle for uncertainty, exactly like figure_1.py.
    for il, iu, a in [
        (im_left_rate, im_left_unc, left_ax),
        (im_right_rate, im_right_unc, right_ax),
    ]:
        a.set_xticks(xticks, minor=True)
        a.set_yticks(xticks, minor=True)
        a.set_xticks(np.log(np.array(majors)), major_names, minor=False)
        a.set_yticks(np.log(np.array(majors)), major_names, minor=False)

        il.set_clip_path(
            Polygon(
                np.array(
                    [
                        [np.log(1), np.log(1)],
                        [np.log(mmax), np.log(1)],
                        [np.log(mmax), np.log(mmax)],
                    ]
                ),
                closed=True,
                transform=a.transData,
            )
        )
        iu.set_clip_path(
            Polygon(
                np.array(
                    [
                        [np.log(1), np.log(1)],
                        [np.log(1), np.log(mmax)],
                        [np.log(mmax), np.log(mmax)],
                    ]
                ),
                closed=True,
                transform=a.transData,
            )
        )
        a.set_xlabel(r"$m_1$ [$M_\odot$]")
        a.grid(ls=":", alpha=0.5, lw=1, color="k")

    # Colorbars on skinny side axes.
    # Use the last-created images for consistent norms.
    n_order = int(np.log10(float(vmax) / float(vmin)) + 0.01)
    cbar_rate = fig.colorbar(
        im_right_rate,
        cax=axes[3],
        ticks=np.logspace(np.log10(vmin), np.log10(vmax), n_order + 1),
    )
    cbar_unc = fig.colorbar(im_right_unc, cax=axes[0])

    tick_spacing = 2
    tick_labels = []
    for i, x in enumerate(np.linspace(np.log10(vmin), np.log10(vmax), n_order + 1, dtype=int)):
        tick_labels.append(f"$10^{{{x}}}$" if i % tick_spacing == 0 else "")
    cbar_rate.ax.set_yticklabels(tick_labels)
    cbar_rate.set_label(
        r"$\frac{\mathrm{d}\mathcal{R}}{\mathrm{d}(\ln m_1)\mathrm{d}(\ln m_2)}$ "
        r"[Gpc${}^{-3}$ yr${}^{-1}$]",
        rotation=90,
    )
    cbar_unc.set_label("Fractional uncertainty", rotation=90)

    cbar_rate.ax.yaxis.set_label_position("left")
    cbar_unc.ax.yaxis.set_label_position("left")

    left_ax.set_title(left_title)
    left_ax.set_ylabel(r"$m_2$ [$M_\odot$]")
    right_ax.set_title(right_title)

    plt.tight_layout(pad=0.0, w_pad=0, h_pad=1.0)

    # Match figure_1.py colorbar axis nudges.
    for ax in (axes[0], axes[3]):
        pos = ax.get_position()
        yscale = pos.y1 - pos.y0
        w = 0.051
        ax.set_position([pos.x0 * 0.95, pos.y0 + yscale * w, pos.x1 - pos.x0, yscale * (1 - 2 * w)])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(
        description="GWTC-4 Figure 1/2/3 comparison plots (paper popsummary vs PLANT emulators)."
    )
    p.add_argument("--figs", type=str, default="1,2,3", help="Comma-separated subset: 1,2,3 (default all).")
    p.add_argument(
        "--compare-mode",
        type=str,
        default="panels",
        choices=("panels", "overlay"),
        help="Figure-1 comparison: panels (side-by-side) or overlay (PLANT contours on paper).",
    )

    p.add_argument(
        "--gwtc4-data-release",
        type=Path,
        default=None,
        help="Path to GWTC-4 Zenodo 'data_release' directory (contains AllCBC_FullPop*.h5, etc.).",
    )
    p.add_argument(
        "--gwtc3-powerlawpeak-dir",
        type=Path,
        default=None,
        help="Optional GWTC-3 PowerLawPeak directory for Figure-3 black comparison curves.",
    )

    p.add_argument("--hyperparam-csv", type=Path, default=None)
    p.add_argument("--checkpoint-dir", type=Path, default=None, help="Default: ./checkpoints")
    p.add_argument("--cfm-checkpoint", type=Path, default=None, help="Default: checkpoints/cfm_final.pt")
    p.add_argument("--diffusion-checkpoint", type=Path, default=None, help="Default: checkpoints/diffusion_final.pt")
    p.add_argument("--device", type=str, default="auto", help="auto | cuda | cpu")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: plots/04_gwtc4_validation/<timestamp>/)",
    )

    p.add_argument("--nbins", type=int, default=60, help="Histogram bins per axis in ln m (Figure 1)")
    p.add_argument("--mmax", type=float, default=180.0, help="Max mass for Figure-1 axis in Msun")
    p.add_argument("--n-rows", type=int, default=256, help="Hyperparameter rows sampled per bootstrap")
    p.add_argument("--n-events-per-row", type=int, default=256, help="Synthetic events per sampled row")
    p.add_argument("--n-boot", type=int, default=12, help="Bootstrap repeats for uncertainty bands")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--vmin", type=float, default=1e-3)
    p.add_argument("--vmax", type=float, default=1e3)
    p.add_argument("--uncmin", type=float, default=0.5)
    p.add_argument("--uncmax", type=float, default=50.0)

    p.add_argument("--m12-max", type=float, default=15.0, help="Max mass for Figure-2 x-axis (paper uses 15)")
    p.add_argument("--m12-bins", type=int, default=70, help="Number of linear bins for Figure-2 curves")

    p.add_argument(
        "--no-tex",
        action="store_true",
        help="Disable LaTeX (matplotlib mathtext only). By default LaTeX is used when available.",
    )

    p.add_argument("--paper-quality", action="store_true", help="Preset: higher MC + smoothing + auto limits + m1*m2 scaling.")
    p.add_argument("--smooth-sigma", type=float, default=0.0, help="Gaussian smoothing width (bins) on log10(rate). Needs scipy.")
    p.add_argument("--smooth-unc-sigma", type=float, default=0.0, help="Gaussian smoothing width (bins) on uncertainty. Needs scipy.")
    p.add_argument("--fullpop-m1m2-scale", action="store_true", help="Multiply PLANT rate by m1*m2 at bin centers (paper FullPop display).")
    p.add_argument("--auto-rate-limits", action="store_true", help="Auto rate limits from quantiles (PLANT only).")
    p.add_argument("--auto-unc-limits", action="store_true", help="Auto uncertainty limits from quantiles (PLANT only).")
    p.add_argument("--rate-q-lo", type=float, default=0.05)
    p.add_argument("--rate-q-hi", type=float, default=0.98)
    p.add_argument("--unc-q-lo", type=float, default=0.05)
    p.add_argument("--unc-q-hi", type=float, default=0.95)
    p.add_argument("--rate-floor", type=float, default=1e-30)

    args = p.parse_args()

    figs = {s.strip() for s in str(args.figs).split(",") if s.strip()}
    if not figs.issubset({"1", "2", "3"}):
        raise ValueError("--figs must be a subset of 1,2,3 (e.g. '1,2').")

    if args.paper_quality:
        args.nbins = max(int(args.nbins), 90)
        args.n_rows = max(int(args.n_rows), 1000)
        args.n_events_per_row = max(int(args.n_events_per_row), 512)
        args.n_boot = max(int(args.n_boot), 20)
        if float(args.smooth_sigma) <= 0:
            args.smooth_sigma = 1.25
        if float(args.smooth_unc_sigma) <= 0:
            args.smooth_unc_sigma = 0.9
        args.auto_rate_limits = True
        args.auto_unc_limits = True
        args.fullpop_m1m2_scale = True
        print("[main] --paper-quality enabled", flush=True)

    import torch

    _cpt = os.environ.get("SLURM_CPUS_PER_TASK", "").strip()
    if _cpt.isdigit():
        _n = int(_cpt)
        torch.set_num_threads(_n)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass

    device = _resolve_device(args.device)
    if str(device).lower().startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")
        dev = torch.device(device)
        idx = 0 if dev.index is None else int(dev.index)
        print(f"[main] CUDA device: {torch.cuda.get_device_name(idx)}", flush=True)
    else:
        print(f"[main] device: {device}", flush=True)

    data_dir = ml_data_dir()
    hp_csv = args.hyperparam_csv.resolve() if args.hyperparam_csv else (data_dir / HYPERPARAM_TABLE_ENCODED_CSV.name)
    ckpt_dir = args.checkpoint_dir.resolve() if args.checkpoint_dir else CHECKPOINT_DIR
    cfm_ckpt = args.cfm_checkpoint.resolve() if args.cfm_checkpoint else (ckpt_dir / "cfm_final.pt")
    dif_ckpt = args.diffusion_checkpoint.resolve() if args.diffusion_checkpoint else (ckpt_dir / "diffusion_final.pt")

    out_dir = args.out_dir.resolve() if args.out_dir else plot_run_dir(Path(__file__))

    import pandas as pd

    if not hp_csv.exists():
        raise FileNotFoundError(f"Missing hyperparameter table: {hp_csv}")
    hp = pd.read_csv(hp_csv)

    need_plant = True
    need_paper = args.gwtc4_data_release is not None

    print(f"[main] loading CFM checkpoint: {cfm_ckpt}", flush=True)
    cfm_model, cfm_lambda_cols, cfm_nrm = _load_emulator(cfm_ckpt, device, "cfm")
    print(f"[main] loading Diffusion checkpoint: {dif_ckpt}", flush=True)
    dif_model, dif_lambda_cols, dif_nrm = _load_emulator(dif_ckpt, device, "diffusion")

    for c in cfm_lambda_cols:
        if c not in hp.columns:
            raise ValueError(f"Column {c!r} required by CFM checkpoint is missing from {hp_csv}.")
    for c in dif_lambda_cols:
        if c not in hp.columns:
            raise ValueError(f"Column {c!r} required by diffusion checkpoint is missing from {hp_csv}.")

    plot_cfm: Optional[RateGrid] = None
    plot_dif: Optional[RateGrid] = None
    cfm_marg: Optional[tuple[np.ndarray, np.ndarray, np.ndarray]] = None
    dif_marg: Optional[tuple[np.ndarray, np.ndarray, np.ndarray]] = None

    vmin, vmax = float(args.vmin), float(args.vmax)
    uncmin, uncmax = float(args.uncmin), float(args.uncmax)

    if "1" in figs:
        print("[main] estimating PLANT Figure-1 grids ...", flush=True)
        grid_cfm = _estimate_rate_grid(
            hp_df=hp,
            lambda_cols=cfm_lambda_cols,
            emulator=cfm_model,
            emulator_kind="cfm",
            normalizer=cfm_nrm,
            nbins=args.nbins,
            mmax=args.mmax,
            n_rows_per_boot=args.n_rows,
            n_events_per_row=args.n_events_per_row,
            n_boot=args.n_boot,
            seed=args.seed,
        )
        grid_dif = _estimate_rate_grid(
            hp_df=hp,
            lambda_cols=dif_lambda_cols,
            emulator=dif_model,
            emulator_kind="diffusion",
            normalizer=dif_nrm,
            nbins=args.nbins,
            mmax=args.mmax,
            n_rows_per_boot=args.n_rows,
            n_events_per_row=args.n_events_per_row,
            n_boot=args.n_boot,
            seed=args.seed + 1,
        )
        plot_cfm = _postprocess_grid_for_display(
            grid_cfm,
            smooth_sigma=float(args.smooth_sigma),
            smooth_unc_sigma=float(args.smooth_unc_sigma),
            fullpop_m1m2_scale=bool(args.fullpop_m1m2_scale),
        )
        plot_dif = _postprocess_grid_for_display(
            grid_dif,
            smooth_sigma=float(args.smooth_sigma),
            smooth_unc_sigma=float(args.smooth_unc_sigma),
            fullpop_m1m2_scale=bool(args.fullpop_m1m2_scale),
        )

        nbin = int(plot_cfm.rate.shape[0])
        tri_lo = _lower_triangle_mask(nbin)
        tri_hi = _upper_triangle_mask(nbin)
        if args.auto_rate_limits:
            rl_lo, rl_hi = _auto_log_limits_positive(plot_cfm.rate, tri_lo, float(args.rate_q_lo), float(args.rate_q_hi), float(args.rate_floor))
            rr_lo, rr_hi = _auto_log_limits_positive(plot_dif.rate, tri_lo, float(args.rate_q_lo), float(args.rate_q_hi), float(args.rate_floor))
            vmin = min(rl_lo, rr_lo)
            vmax = max(rl_hi, rr_hi)
            print(f"[main] auto rate limits: vmin={vmin:.3e} vmax={vmax:.3e}", flush=True)
        if args.auto_unc_limits:
            ul_lo, ul_hi = _auto_log_limits_positive(plot_cfm.frac_unc, tri_hi, float(args.unc_q_lo), float(args.unc_q_hi), 0.5)
            ur_lo, ur_hi = _auto_log_limits_positive(plot_dif.frac_unc, tri_hi, float(args.unc_q_lo), float(args.unc_q_hi), 0.5)
            uncmin = min(ul_lo, ur_lo)
            uncmax = max(ul_hi, ur_hi)
            print(f"[main] auto unc limits: uncmin={uncmin:.3e} uncmax={uncmax:.3e}", flush=True)

    if ("2" in figs) or ("3" in figs):
        print("[main] estimating PLANT Figure-2/3 marginals ...", flush=True)
        m_edges = np.linspace(1.0, float(args.m12_max), int(args.m12_bins) + 1)
        cfm_marg = _estimate_1d_marginals_from_emulator(
            hp_df=hp,
            lambda_cols=cfm_lambda_cols,
            emulator=cfm_model,
            emulator_kind="cfm",
            normalizer=cfm_nrm,
            m_edges=m_edges,
            n_rows_per_boot=args.n_rows,
            n_events_per_row=args.n_events_per_row,
            n_boot=args.n_boot,
            seed=args.seed + 10,
            m_clip_hi=float(args.m12_max),
        )
        dif_marg = _estimate_1d_marginals_from_emulator(
            hp_df=hp,
            lambda_cols=dif_lambda_cols,
            emulator=dif_model,
            emulator_kind="diffusion",
            normalizer=dif_nrm,
            m_edges=m_edges,
            n_rows_per_boot=args.n_rows,
            n_events_per_row=args.n_events_per_row,
            n_boot=args.n_boot,
            seed=args.seed + 11,
            m_clip_hi=float(args.m12_max),
        )

    paper_fullpop = paper_bgp = None
    if need_paper and ("1" in figs):
        print("[main] loading paper Figure-1 grids ...", flush=True)
        _m1, _m2, full_R, full_U, bgp_R, bgp_U = _load_gwtc4_figure1_grids(args.gwtc4_data_release, mmax=float(args.mmax))
        paper_fullpop = (full_R, full_U.T)
        paper_bgp = (bgp_R, bgp_U.T)

    outputs: list[Path] = []
    if "1" in figs:
        out1 = out_dir / f"figure_1_compare_{args.compare_mode}.pdf"
        _plot_figure1_compare(
            out_path=out1,
            mmax=float(args.mmax),
            vmin=float(vmin),
            vmax=float(vmax),
            uncmin=float(uncmin),
            uncmax=float(uncmax),
            usetex=not bool(args.no_tex),
            compare_mode=str(args.compare_mode),
            paper_fullpop=paper_fullpop,
            paper_bgp=paper_bgp,
            plant_cfm=plot_cfm,
            plant_diff=plot_dif,
        )
        outputs.append(out1)

    if "2" in figs:
        out2 = out_dir / "figure_2_compare.pdf"
        _plot_figure2_compare(
            out_path=out2,
            data_release_dir=args.gwtc4_data_release,
            plant_cfm_marg=cfm_marg,
            plant_diff_marg=dif_marg,
            xlim=(1.0, float(args.m12_max)),
            ylim=(1e-2, 2e3),
            usetex=not bool(args.no_tex),
        )
        outputs.append(out2)

    if "3" in figs:
        out3 = out_dir / "figure_3_compare.pdf"
        _plot_figure3_compare(
            out_path=out3,
            data_release_dir=args.gwtc4_data_release,
            gwtc3_dir=args.gwtc3_powerlawpeak_dir,
            plant_marg=cfm_marg,
            usetex=not bool(args.no_tex),
        )
        outputs.append(out3)

    meta = {
        "figs": sorted(list(figs)),
        "compare_mode": str(args.compare_mode),
        "hyperparam_csv": str(hp_csv),
        "cfm_checkpoint": str(cfm_ckpt),
        "diffusion_checkpoint": str(dif_ckpt),
        "gwtc4_data_release": str(args.gwtc4_data_release) if args.gwtc4_data_release else None,
        "gwtc3_powerlawpeak_dir": str(args.gwtc3_powerlawpeak_dir) if args.gwtc3_powerlawpeak_dir else None,
        "usetex_requested": not bool(args.no_tex),
        "outputs": [str(p) for p in outputs],
    }
    meta_path = out_dir / "compare_run_metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    for o in outputs:
        print(f"Wrote: {o}", flush=True)
    print(f"Wrote: {meta_path}", flush=True)


if __name__ == "__main__":
    main()

