#!/usr/bin/env python3
"""
Step 00 — Figure-2-style SSPC marginal mass-rate curves (training-grid subset).

Mimics `o4a-astro/figure_scripts/figure_2.py` layout (two stacked panels, log y,
x in [1, 15] Msun, serif, similar axis labels) but replaces FullPop/BGP curves
with **intrinsic** SSPC merger-rate-weighted marginals at a fixed redshift slice.

For each selected (sfr_a, mu0) grid point (sparse subset of the Λ grid used in
`hyperparam_table.csv`), we aggregate **all train-split rows** matching that pair
(across formation channels), load `models_sspc.hdf5` keys, keep events with
|z - z_target| ≤ z_tol, map (mchirp, q) -> (m1, m2), and form::

    d R / d m_i  ~  (sum of merger-rate weights in bin j) / Δm

Raw histograms are **noisy** (finite SSPC rows per bin). By default we apply a
**multiplicative Gaussian smoother on log₁₀(rate)** (same idea as
`gwtc4_validation.py` / display choices for emulator grids) so curves resemble
the smooth GWTC-4 figure-2 *style* while staying on a low-*z* SSPC slice (default **z = 0.1**, first bin).

This is **not** the GWTC-4.0 detected population rate (no p_det, different
cosmology/units); it is the same *visual grammar* as figure 2 for SSPC diagnostics.

Usage: ``python scripts/analysis/00_fig2_spread.py``
SLURM: ``slurm/09_fig2_spread.sh``
"""

from __future__ import annotations

import sys
from pathlib import Path as _Path

_ANALYSIS_DIR = _Path(__file__).resolve().parent
_PROJECT_ROOT = _ANALYSIS_DIR.parents[1]
for _p in (_PROJECT_ROOT, _ANALYSIS_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from plant_paths import (  # noqa: E402
    HYPERPARAM_TABLE_CSV,
    PROJECT_ROOT,
    SPLITS_JSON,
    ensure_paths,
    find_data_dir,
    ml_data_dir,
    resolve_plot_output,
)

ensure_paths()

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lib.distribution import m1_from_mchirp_q



def _m1m2(mchirp: np.ndarray, q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    q = np.clip(q.astype(np.float64), 1e-8, 1.0)
    mc = np.clip(mchirp.astype(np.float64), 1e-8, None)
    m1 = m1_from_mchirp_q(mc, q)
    m2 = q * m1
    swap = m2 > m1
    if np.any(swap):
        t = m1.copy()
        m1[swap] = m2[swap]
        m2[swap] = t[swap]
    return m1, m2


def _load_splits(path: Path) -> Dict[str, List[int]]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _unique_pairs(df: pd.DataFrame) -> List[Tuple[float, float]]:
    sub = df[["chi_b", "alpha_CE"]].drop_duplicates()
    pairs = sorted({(float(r.chi_b), float(r.alpha_CE)) for r in sub.itertuples(index=False)})
    return pairs


def _subsample_pairs(pairs: Sequence[Tuple[float, float]], n_target: int, seed: int) -> List[Tuple[float, float]]:
    _ = seed  # reserved for future stratified sampling
    if len(pairs) <= n_target:
        return list(pairs)
    idx = np.linspace(0, len(pairs) - 1, num=n_target, dtype=float)
    idx = np.unique(np.round(idx).astype(int))
    return [pairs[i] for i in idx]


def _z_slice_mask(z: np.ndarray, z_target: float, z_tol: float) -> np.ndarray:
    """
    Select mergers near z_target. Use **inclusive** band |Δz| ≤ z_tol (+ tiny eps).

    SSPC catalogs often start at z ≈ 0.1 (first cosmology bin). With z_target=0,
    z_tol=0.1, the boundary case z=0.1 must satisfy |0.1−0| ≤ 0.1 — strict `<`
    incorrectly excludes **all** events.
    """
    dz = np.abs(z.astype(np.float64) - float(z_target))
    return dz <= float(z_tol) + 1e-12


def _probe_z_range(hdf5_path: Path, keys: List[str]) -> Tuple[float, float]:
    """Min/max z across first few keys (for diagnostics)."""
    zmin, zmax = np.inf, -np.inf
    for key in keys[:12]:
        df = pd.read_hdf(hdf5_path, key=key)
        if "z" not in df.columns:
            continue
        zz = df["z"].values.astype(np.float64)
        zmin = min(zmin, float(np.nanmin(zz)))
        zmax = max(zmax, float(np.nanmax(zz)))
    return zmin, zmax


def _collect_weighted_marginals(
    hdf5_path: Path,
    row_keys: List[str],
    z_target: float,
    z_tol: float,
    m_edges: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (m_centers, dRdm1, dRdm2) with same binning for all keys (summed).
    """
    h1 = np.zeros(len(m_edges) - 1, dtype=np.float64)
    h2 = np.zeros(len(m_edges) - 1, dtype=np.float64)
    for key in row_keys:
        df = pd.read_hdf(hdf5_path, key=key)
        if "z" not in df.columns or "weight" not in df.columns:
            continue
        z = df["z"].values.astype(np.float64)
        m = _z_slice_mask(z, z_target, z_tol)
        if not np.any(m):
            continue
        sub = df.loc[m]
        mch = sub["mchirp"].values.astype(np.float64)
        qv = sub["q"].values.astype(np.float64)
        wv = sub["weight"].values.astype(np.float64)
        m1, m2 = _m1m2(mch, qv)
        m1 = np.clip(m1, float(m_edges[0]), float(m_edges[-1]))
        m2 = np.clip(m2, float(m_edges[0]), float(m_edges[-1]))
        wv = np.clip(wv, 0.0, None)
        h1 += np.histogram(m1, bins=m_edges, weights=wv)[0]
        h2 += np.histogram(m2, bins=m_edges, weights=wv)[0]
    dm = np.diff(m_edges).astype(np.float64)
    d1 = h1 / np.clip(dm, 1e-30, None)
    d2 = h2 / np.clip(dm, 1e-30, None)
    centers = 0.5 * (m_edges[:-1] + m_edges[1:])
    return centers, d1, d2


def _smooth_positive_log_field(z: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian filter on log10(z); preserves positivity (cf. gwtc4_validation.py)."""
    if sigma <= 0:
        return z
    try:
        from scipy.ndimage import gaussian_filter
    except ImportError as e:
        raise ImportError("Smoothing requires scipy (conda env `plant` includes it).") from e
    eps = np.finfo(np.float64).tiny * 1e6
    logz = np.log10(np.maximum(z.astype(np.float64), eps))
    logz_s = gaussian_filter(logz, sigma=float(sigma), mode="reflect")
    out = np.power(10.0, logz_s)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def main() -> None:
    p = argparse.ArgumentParser(description="SSPC figure-2-style mass marginals at z ~ z_target.")
    p.add_argument("--work-dir", type=Path, default=None, help="Dir with hyperparam_table.csv (default: data/)")
    p.add_argument("--hyperparam-csv", type=Path, default=None)
    p.add_argument("--splits-json", type=Path, default=None)
    p.add_argument("--split", type=str, default="train", choices=("train", "val", "test", "all"))
    p.add_argument("--sspc-hdf5", type=Path, default=None, help="Default: <work>/data/sspc/models_sspc.hdf5")
    p.add_argument("--n-pairs", type=int, default=9, help="How many (sfr_a, mu0) combinations to plot (evenly spaced).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--z-target",
        type=float,
        default=0.1,
        help="Redshift centre for slice (default 0.1 matches typical SSPC lowest bin).",
    )
    p.add_argument(
        "--z-tol",
        type=float,
        default=0.10,
        help="Half-width: keep mergers with |z - z_target| ≤ z_tol (inclusive).",
    )
    p.add_argument(
        "--smooth-sigma",
        type=float,
        default=1.4,
        help="Gaussian width (in mass bins) on log10(rate); 0 disables.",
    )
    p.add_argument("--no-smooth", action="store_true", help="Plot raw histogram marginals.")
    p.add_argument("--m12-max", type=float, default=15.0)
    p.add_argument("--m12-bins", type=int, default=56, help="Linear bins in m (fewer → naturally smoother).")
    p.add_argument("--ylim", type=float, nargs=2, default=(1e-2, 2e3))
    p.add_argument("--auto-y", action="store_true", help="Set y-limits from non-zero data quantiles.")
    p.add_argument("--out", type=Path, default=None, help="Output PDF path.")
    p.add_argument("--no-tex", action="store_true", help="Disable matplotlib usetex.")
    args = p.parse_args()

    work = args.work_dir.resolve() if args.work_dir else ml_data_dir()
    hp_path = args.hyperparam_csv.resolve() if args.hyperparam_csv else (work / HYPERPARAM_TABLE_CSV.name)
    splits_path = args.splits_json.resolve() if args.splits_json else (work / SPLITS_JSON.name)
    h5 = args.sspc_hdf5.resolve() if args.sspc_hdf5 else (find_data_dir() / "sspc" / "models_sspc.hdf5")

    if not hp_path.exists():
        raise FileNotFoundError(f"Missing {hp_path}")
    if not h5.exists():
        raise FileNotFoundError(f"Missing SSPC HDF5: {h5}")

    hp = pd.read_csv(hp_path)
    splits = _load_splits(splits_path)
    if args.split != "all" and splits:
        idx = splits.get(args.split)
        if not idx:
            raise ValueError(f"Split {args.split!r} empty or missing in {splits_path}")
        hp_use = hp.iloc[list(idx)].copy()
    else:
        hp_use = hp.copy()

    pairs_all = _unique_pairs(hp_use)
    if not pairs_all:
        raise RuntimeError("No (chi_b, alpha_CE) pairs in filtered hyperparam table.")

    pairs_sel = _subsample_pairs(pairs_all, int(args.n_pairs), int(args.seed))
    m_edges = np.linspace(1.0, float(args.m12_max), int(args.m12_bins) + 1)

    # Keys for z-range probe (first pair that has rows)
    probe_keys: List[str] = []
    for sa0, mu0 in pairs_sel[:3]:
        r0 = hp_use[np.isclose(hp_use["chi_b"].values, sa0, rtol=1e-9, atol=1e-12) & np.isclose(hp_use["alpha_CE"].values, mu0, rtol=1e-9, atol=1e-12)]
        if len(r0):
            probe_keys = r0["key"].astype(str).tolist()
            break
    if not probe_keys and len(hp_use):
        probe_keys = [str(hp_use.iloc[0]["key"])]
    if probe_keys:
        zm, zM = _probe_z_range(h5, probe_keys)
        print(f"[fig2_spread] SSPC z range (sample keys): z_min={zm:.4g}, z_max={zM:.4g}", flush=True)

    curves_m1: List[Tuple[str, np.ndarray, np.ndarray]] = []
    curves_m2: List[Tuple[str, np.ndarray, np.ndarray]] = []
    for sa, mu0 in pairs_sel:
        rows = hp_use[np.isclose(hp_use["chi_b"].values, sa, rtol=1e-9, atol=1e-12) & np.isclose(hp_use["alpha_CE"].values, mu0, rtol=1e-9, atol=1e-12)]
        keys = rows["key"].astype(str).tolist()
        if not keys:
            continue
        centers, d1, d2 = _collect_weighted_marginals(h5, keys, args.z_target, args.z_tol, m_edges)
        sig = 0.0 if args.no_smooth else float(args.smooth_sigma)
        if sig > 0:
            d1 = _smooth_positive_log_field(d1, sig)
            d2 = _smooth_positive_log_field(d2, sig)
        label = rf"[{len(curves_m1) + 1}] $a_{{\rm SF}}={sa:.3f}$, $\mu_0={mu0:.3f}$"
        curves_m1.append((label, centers, d1))
        curves_m2.append((label, centers, d2))

    if not curves_m1:
        raise RuntimeError("No curves after aggregation (check z_tol / HDF5 keys).")

    total_y = sum(float(np.sum(np.maximum(y, 0.0))) for _l, _c, y in curves_m1)
    if total_y <= 0.0:
        zm, zM = _probe_z_range(h5, probe_keys) if probe_keys else (float("nan"), float("nan"))
        hint = ""
        if np.isfinite(zm) and np.isfinite(zM):
            hint = f" Sample z in data: [{zm:.4g}, {zM:.4g}]. Try --z-tol 0.15 or --z-target {zm:.3f}."
        raise RuntimeError(
            "All marginal rates are zero for this z-slice (often strict z band misses SSPC bins)."
            + hint
        )

    plt.rc("text", usetex=not bool(args.no_tex))
    plt.rc("font", family="serif", size=11)
    fig, axes = plt.subplots(nrows=2, figsize=(10.5, 8.5))
    cmap = plt.get_cmap("viridis")

    ymax_track = 1e-30
    ymin_track = 1e30
    for i, (lab, c, y) in enumerate(curves_m1):
        color = cmap(i / max(len(curves_m1) - 1, 1))
        axes[0].plot(c, np.maximum(y, 1e-30), color=color, lw=1.6, label=lab)
        ymax_track = max(ymax_track, float(np.nanmax(y)))
        pos = y[y > 0]
        if pos.size:
            ymin_track = min(ymin_track, float(np.nanmin(pos)))
    for i, (lab, c, y) in enumerate(curves_m2):
        color = cmap(i / max(len(curves_m2) - 1, 1))
        axes[1].plot(c, np.maximum(y, 1e-30), color=color, lw=1.6)

    y0, y1 = float(args.ylim[0]), float(args.ylim[1])
    if args.auto_y and np.isfinite(ymin_track) and np.isfinite(ymax_track) and ymax_track > 0:
        y0 = max(1e-4, ymin_track * 0.5)
        y1 = ymax_track * 5.0

    for ii in range(2):
        ax = axes[ii]
        ax.set_yscale("log")
        ax.set_ylim(y0, y1)
        ax.set_xlim(1.0, float(args.m12_max))
        ax.grid(ls=":", alpha=0.2, lw=1, color="k")
        if ii == 0:
            ax.set_ylabel(
                r"$\mathrm{d}\mathcal{R}/\mathrm{d}m_1$ [yr$^{-1}$ M$_\odot^{-1}$]"
                "\n(SSPC weights $\div \Delta m$)"
            )
            ax.set_xlabel(r"$m_1$ [$M_\odot$]")
        else:
            ax.set_ylabel(
                r"$\mathrm{d}\mathcal{R}/\mathrm{d}m_2$ [yr$^{-1}$ M$_\odot^{-1}$]"
                "\n(SSPC weights $\div \Delta m$)"
            )
            ax.set_xlabel(r"$m_2$ [$M_\odot$]")

    sm_txt = "raw bins" if args.no_smooth or float(args.smooth_sigma) <= 0 else f"smooth σ={float(args.smooth_sigma):.2f} bins (log domain)"
    fig.suptitle(
        rf"SSPC train split: $N={len(pairs_sel)}$ $(a_{{\rm SF}},\mu_0)$ pairs — "
        rf"$|z-{args.z_target:.2f}| \leq {args.z_tol:.2f}$, {sm_txt}",
        fontsize=11,
        y=0.98,
    )

    handles, labels = axes[0].get_legend_handles_labels()
    ncol = 3 if len(labels) > 5 else 2
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=ncol,
        fontsize=7,
        frameon=True,
        columnspacing=0.9,
        handlelength=1.8,
    )

    plt.tight_layout(rect=(0, 0.14, 1, 0.93))
    out = (
        args.out.resolve()
        if args.out
        else resolve_plot_output(Path(__file__), filename="fig2_sspc_spread.pdf")
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
