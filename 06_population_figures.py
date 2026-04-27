#!/usr/bin/env python3
"""
Post-training population / intrinsic forward figures from SSPC HDF5.

Produces:
  - d(aggregated rate proxy)/dz from summed merger-rate weights in z (0–10)
  - M1, M2, and q = m2/m1 in redshift slices (default z≈0.2 and z≈1.0) per channel

Reuses the same m1(Mc,q) map as `data_distribution_analysis.m1_from_mchirp_q`.
Outputs under `plots/population_results/<timestamp>/` by default.
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Safe import from same package
from data_distribution_analysis import m1_from_mchirp_q, load_sspc_rate_vs_redshift

_HERE = Path(__file__).resolve().parent
_DEFAULT_SSPC = _HERE / "data" / "sspc" / "models_sspc.hdf5"
_DEFAULT_PLOTS = _HERE / "plots" / "population_results"


def _collect_m1m2q_at_z(
    sspc_path: Path,
    channels: Tuple[str, ...],
    z_target: float,
    z_tol: float = 0.06,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (m1, m2, q, w) for all grid keys where |z - z_target| < z_tol.
    m2 = q * m1.
    """
    m1a, m2a, qa, wa = [], [], [], []
    if not sspc_path.exists():
        return (
            np.array([]), np.array([]), np.array([]), np.array([]),
        )
    with h5py.File(sspc_path, "r") as f:
        for ch in channels:
            if ch not in f:
                continue
            gch = f[ch]
            for sfra_key in gch.keys():
                for mu0_key in gch[sfra_key].keys():
                    raw = gch[sfra_key][mu0_key]["table"][()]
                    df = pd.DataFrame(raw)
                    if "z" not in df or "mchirp" not in df or "q" not in df or "weight" not in df:
                        continue
                    m = np.abs(df["z"].values.astype(float) - z_target) < z_tol
                    if not np.any(m):
                        continue
                    sub = df.loc[m]
                    mch = sub["mchirp"].values
                    qv = sub["q"].values
                    wv = sub["weight"].values
                    m1b = m1_from_mchirp_q(mch, qv)
                    m2b = qv * m1b
                    m1a.append(m1b)
                    m2a.append(m2b)
                    qa.append(qv)
                    wa.append(wv)
    if not m1a:
        return (
            np.array([]), np.array([]), np.array([]), np.array([]),
        )
    return (
        np.concatenate(m1a),
        np.concatenate(m2a),
        np.concatenate(qa),
        np.concatenate(wa),
    )


def _plot_weighted_hist(
    x: np.ndarray,
    w: np.ndarray,
    ax: plt.Axes,
    xlabel: str,
    title: str,
    bins: int = 50,
) -> None:
    w = np.clip(w.astype(float), 0.0, None)
    if len(x) == 0 or w.sum() <= 0:
        ax.set_title(f"{title} (no data)")
        return
    h, e = np.histogram(x, bins=bins, weights=w, density=True)
    c = 0.5 * (e[:-1] + e[1:])
    ax.plot(c, h, "b-", lw=1.2)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)


def run_all(
    sspc_hdf5: Path,
    out_dir: Path,
    z_slices: List[float],
    z_tol: float,
    include_che: bool,
) -> None:
    chans: Tuple[str, ...]
    if include_che:
        chans = ("SMT", "CE", "CHE")
    else:
        chans = ("SMT", "CE")

    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Rate vs z (from shared helper)
    z_all, r_all = load_sspc_rate_vs_redshift(sspc_hdf5, include_channels=chans)
    if len(z_all) > 0:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(z_all, np.maximum(r_all, 1e-30), "b-", lw=1.4)
        ax.set_yscale("log")
        for zm in z_slices:
            j = int(np.argmin(np.abs(z_all - zm)))
            ax.axvline(zm, color="gray", ls="--", alpha=0.4)
            ax.plot(z_all[j], max(r_all[j], 1e-30), "ro", ms=5, zorder=5)
        ax.set_xlim(0, 10.0)
        ax.set_xlabel("Redshift z")
        ax.set_ylabel("Sum of weights (SSPC, intrinsic) — proxy to dR/dz shape")
        ax.set_title("Aggregated SSPC rate-weight vs z (per-bin merger-rate weights)")
        fig.savefig(out_dir / "rate_vs_redshift_sspc.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        with open(out_dir / "rate_at_z_markers.txt", "w", encoding="utf-8") as f:
            for zm in z_slices:
                j = int(np.argmin(np.abs(z_all - zm)))
                f.write(f"z_target={zm:.3f}  z_bin={z_all[j]:.4f}  sum_weight={r_all[j]:.6e}\n")

    # 2) M1, M2, q in z-slices
    for zt in z_slices:
        for ch in chans:
            m1, m2, q, w = _collect_m1m2q_at_z(
                sspc_hdf5, (ch,), z_target=zt, z_tol=z_tol
            )
            fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
            _plot_weighted_hist(m1, w, axes[0], r"$m_1$ ($M_\odot$)", f"{ch}  z≈{zt:.1f}  m1")
            _plot_weighted_hist(m2, w, axes[1], r"$m_2$ ($M_\odot$)", f"{ch}  m2")
            _plot_weighted_hist(q, w, axes[2], r"$q=m_2/m_1$", f"{ch}  q")
            fig.suptitle(f"Intrinsic mass ratio / masses — channel {ch}, |z−{zt:.1f}|<{z_tol}", fontsize=10)
            fig.tight_layout()
            safe_z = f"{zt:.2f}".replace(".", "p")
            fig.savefig(
                out_dir / f"channel_{ch}_zslice_{safe_z}.png", dpi=150, bbox_inches="tight"
            )
            plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="SSPC population figures (forward / intrinsic).")
    p.add_argument("--sspc-hdf5", type=Path, default=_DEFAULT_SSPC, help="models_sspc.hdf5")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: plots/population_results/<timestamp>/",
    )
    p.add_argument(
        "--z-slices",
        type=float,
        nargs="*",
        default=[0.2, 1.0],
        help="Redshift reference values for M1/M2/q (nearest events |z-zt|<tol).",
    )
    p.add_argument("--z-tol", type=float, default=0.06, help="Half-width in z for slices.")
    p.add_argument(
        "--include-che",
        action="store_true",
        help="Include CHE channel in aggregation (off by default, matches some paper figs).",
    )
    args = p.parse_args()
    out = args.output_dir
    if out is None:
        out = _DEFAULT_PLOTS / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out = out.resolve()
    print(f"Output directory: {out}")
    run_all(
        args.sspc_hdf5.resolve(),
        out,
        z_slices=list(args.z_slices),
        z_tol=args.z_tol,
        include_che=bool(args.include_che),
    )
    print("Done.")


if __name__ == "__main__":
    main()
