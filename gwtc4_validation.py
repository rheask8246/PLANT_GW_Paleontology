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
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


def _find_work_dir() -> Path:
    for d in [Path("."), Path(__file__).resolve().parent]:
        if (d / "hyperparam_table_encoded.csv").exists():
            return d.resolve()
    return Path(".").resolve()


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

    here = Path(__file__).resolve().parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))
    from importlib import import_module

    _05 = import_module("05_posterior_network")
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

    rng = np.random.default_rng(int(seed))
    rate_draws = np.zeros((int(n_boot), int(nbins), int(nbins)), dtype=np.float64)

    for b in range(int(n_boot)):
        # Independent stochastic draw per bootstrap (new RNG stream).
        rng_b = np.random.default_rng(int(seed) + 100_000 * (b + 1))
        # Sample Λ rows proportional to intrinsic rate; allow repeats.
        rows = rng_b.choice(len(hp_df), size=int(n_rows_per_boot), replace=True, p=p_row)

        H = np.zeros((int(nbins), int(nbins)), dtype=np.float64)
        for ri in rows:
            lam = hp_df.iloc[int(ri)][lambda_cols].values.astype(np.float32)
            # For reproducibility across emulators, seed torch per row per boot.
            try:
                import torch

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

        # Convert sum of weights per bin to a density per dlnm1 dlnm2.
        # Since bins are uniform in ln, divide by (Δln)^2 so values are comparable across nbins.
        dln = float(ln_edges[1] - ln_edges[0])
        H = H / max(dln * dln, 1e-12)
        rate_draws[b] = H

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

    if usetex:
        try:
            plt.rc("text", usetex=True)
        except Exception:
            # If TeX isn't available, fall back silently (keeps script usable on fresh machines).
            plt.rc("text", usetex=False)
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
    p = argparse.ArgumentParser(description="Recreate GWTC-4 Figure 1 style plot from PLANT emulator outputs.")
    p.add_argument("--hyperparam-csv", type=Path, default=None)
    p.add_argument("--checkpoint-dir", type=Path, default=None, help="Default: ./checkpoints")
    p.add_argument("--cfm-checkpoint", type=Path, default=None, help="Default: checkpoints/cfm_final.pt")
    p.add_argument("--diffusion-checkpoint", type=Path, default=None, help="Default: checkpoints/diffusion_final.pt")
    p.add_argument("--device", type=str, default="cpu", help="cpu | cuda")
    p.add_argument("--out", type=Path, default=None, help="Output PDF path")

    # Grid + Monte Carlo controls
    p.add_argument("--nbins", type=int, default=60, help="Histogram bins per axis in ln m")
    p.add_argument("--mmax", type=float, default=180.0, help="Max mass for plot in Msun")
    p.add_argument("--n-rows", type=int, default=256, help="Number of hyperparameter rows sampled per bootstrap")
    p.add_argument("--n-events-per-row", type=int, default=256, help="Synthetic events per sampled row")
    p.add_argument("--n-boot", type=int, default=12, help="Number of stochastic repeats for uncertainty")
    p.add_argument("--seed", type=int, default=42)

    # Match paper-like display ranges (formatting), adjustable if needed.
    p.add_argument("--vmin", type=float, default=1e-3)
    p.add_argument("--vmax", type=float, default=1e3)
    p.add_argument("--uncmin", type=float, default=0.5)
    p.add_argument("--uncmax", type=float, default=50.0)
    p.add_argument("--no-tex", action="store_true", help="Disable LaTeX text rendering")
    args = p.parse_args()

    work = _find_work_dir()
    hp_csv = args.hyperparam_csv.resolve() if args.hyperparam_csv else (work / "hyperparam_table_encoded.csv")
    ckpt_dir = args.checkpoint_dir.resolve() if args.checkpoint_dir else (work / "checkpoints")
    cfm_ckpt = args.cfm_checkpoint.resolve() if args.cfm_checkpoint else (ckpt_dir / "cfm_final.pt")
    dif_ckpt = args.diffusion_checkpoint.resolve() if args.diffusion_checkpoint else (ckpt_dir / "diffusion_final.pt")

    if args.out is None:
        out = work / "plots" / "gwtc4_validation" / "figure_1_emulators.pdf"
    else:
        out = args.out.resolve()

    import pandas as pd

    if not hp_csv.exists():
        raise FileNotFoundError(f"Missing hyperparameter table: {hp_csv}")
    hp = pd.read_csv(hp_csv)

    # Load emulators.
    cfm_model, cfm_lambda_cols, cfm_nrm = _load_emulator(cfm_ckpt, args.device, "cfm")
    dif_model, dif_lambda_cols, dif_nrm = _load_emulator(dif_ckpt, args.device, "diffusion")

    # Ensure lambda cols exist in hp.
    for c in cfm_lambda_cols:
        if c not in hp.columns:
            raise ValueError(f"Column {c!r} required by CFM checkpoint is missing from {hp_csv}.")
    for c in dif_lambda_cols:
        if c not in hp.columns:
            raise ValueError(f"Column {c!r} required by diffusion checkpoint is missing from {hp_csv}.")

    # Estimate grids.
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

    # Save plot with exact figure_1.py formatting.
    _apply_figure1_style_and_save(
        out_path=out,
        grid_left=grid_cfm,
        grid_right=grid_dif,
        left_title=r"\textsc{CFM}" if not args.no_tex else "CFM",
        right_title=r"\textsc{Diffusion}" if not args.no_tex else "Diffusion",
        mmax=float(args.mmax),
        vmin=float(args.vmin),
        vmax=float(args.vmax),
        uncmin=float(args.uncmin),
        uncmax=float(args.uncmax),
        usetex=not bool(args.no_tex),
    )

    meta = {
        "hyperparam_csv": str(hp_csv),
        "cfm_checkpoint": str(cfm_ckpt),
        "diffusion_checkpoint": str(dif_ckpt),
        "nbins": int(args.nbins),
        "mmax": float(args.mmax),
        "n_rows": int(args.n_rows),
        "n_events_per_row": int(args.n_events_per_row),
        "n_boot": int(args.n_boot),
        "seed": int(args.seed),
        "vmin": float(args.vmin),
        "vmax": float(args.vmax),
        "uncmin": float(args.uncmin),
        "uncmax": float(args.uncmax),
    }
    meta_path = out.with_suffix(".json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Wrote: {out}")
    print(f"Wrote: {meta_path}")


if __name__ == "__main__":
    main()

