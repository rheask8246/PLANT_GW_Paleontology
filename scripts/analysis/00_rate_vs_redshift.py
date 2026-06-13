#!/usr/bin/env python3
"""
Step 00 — Intrinsic merger rate R(z) for a few grid models.

Recomputes cosmic integration (same numerics as ``00_sspc_data_generation.py``) with
**nuisance parameters fixed at the Step-00 best-fit values**
(DOI:10.3847/1538-4357/acbf51), and
either:

  --vary sfra   : several ``a_SF`` values, ``mu0`` fixed at grid midpoint
  --vary mu0    : several ``mu0`` values, ``a_SF`` fixed at grid midpoint

Curves are summed over formation channels (SMT + CE + CHE) by default.

SLURM: ``slurm/00_rate_vs_redshift.sh``
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Literal, Sequence, Tuple

_ANALYSIS_DIR = Path(__file__).resolve().parent
if str(_ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(_ANALYSIS_DIR))

from _bootstrap import setup  # noqa: E402

PROJECT_ROOT = setup()

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plant_paths import find_data_dir, resolve_plot_output  # noqa: E402

VaryAxis = Literal["sfra", "mu0"]


def _load_sspc00_module():
    path = PROJECT_ROOT / "scripts" / "00_sspc_data_generation.py"
    spec = importlib.util.spec_from_file_location("_plant_sspc00", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _mid(range_pair: Tuple[float, float]) -> float:
    lo, hi = range_pair
    return 0.5 * (float(lo) + float(hi))


def fixed_nuisance_params(mod: Any) -> Dict[str, float]:
    """Step-00 fixed nuisance parameters (matches --fixed-nuisance-tng100)."""
    return mod.nuisance_tng100_params()


def default_mid_sfra(mod: Any) -> float:
    return _mid(mod.SFRA_RANGE)


def default_mid_mu0(mod: Any) -> float:
    return _mid(mod.MU0_RANGE)


def curve_values(
    mod: Any,
    vary: VaryAxis,
    *,
    n_curves: int,
    values: Sequence[float] | None,
    hdf5_path: Path | None,
) -> List[float]:
    if values is not None:
        return [float(v) for v in values]
    if hdf5_path is not None and hdf5_path.is_file():
        from_hdf5 = _axis_values_from_hdf5(hdf5_path, vary)
        if from_hdf5:
            if len(from_hdf5) <= n_curves:
                return from_hdf5
            idx = np.linspace(0, len(from_hdf5) - 1, n_curves, dtype=int)
            return [from_hdf5[i] for i in idx]
    rng = mod.SFRA_RANGE if vary == "sfra" else mod.MU0_RANGE
    return [float(v) for v in np.linspace(rng[0], rng[1], n_curves)]


def _parse_key_token(token: str, prefix: str) -> float | None:
    if token.startswith(prefix):
        try:
            return int(token[len(prefix) :]) / 10_000.0
        except ValueError:
            return None
    return None


def _axis_values_from_hdf5(hdf5_path: Path, vary: VaryAxis) -> List[float]:
    """Unique ``sfra`` or ``mu0`` values present in Step 00 HDF5 keys."""
    import pandas as pd

    vals: set[float] = set()
    with pd.HDFStore(str(hdf5_path), mode="r") as store:
        for key in store.keys():
            parts = key.strip("/").split("/")
            if len(parts) < 3:
                continue
            sfra = _parse_key_token(parts[1], "sfra")
            mu0 = _parse_key_token(parts[2], "mu0")
            if sfra is None or mu0 is None:
                continue
            vals.add(sfra if vary == "sfra" else mu0)
    return sorted(vals)


def rate_vs_redshift(
    mod: Any,
    bps_by_channel: Dict[str, Any],
    *,
    sfr_a: float,
    mu0: float,
    nuisances: Dict[str, float],
    channels: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sum intrinsic merger rate per redshift shell over selected BPS channels.

    Uses ``pz * weight`` from ``compute_merger_weights`` (same as Step 00).
    """
    redshifts, times_Myr, shell_volumes, time_first_SF = mod.build_redshift_grid()
    n_z = len(redshifts)
    rate_z = np.zeros(n_z, dtype=np.float64)

    sfr_z = mod.find_sfr(
        redshifts,
        sfr_a,
        nuisances["sfr_b"],
        nuisances["sfr_c"],
        nuisances["sfr_d"],
    )

    # Metallicity grid from global BPS Z range
    all_met = np.concatenate(
        [bps_by_channel[ch]["metallicity"].values for ch in channels if ch in bps_by_channel]
    )
    logZ_min = float(np.log(all_met.min()))
    logZ_max = float(np.log(all_met.max()))

    dPdlogZ, met_grid, p_draw = mod.find_metallicity_distribution(
        redshifts,
        logZ_min,
        logZ_max,
        mu0=mu0,
        muz=nuisances["muz"],
        sigma0=nuisances["sigma0"],
        sigmaz=nuisances["sigmaz"],
        alpha=nuisances["alpha_skew"],
    )

    for ch in channels:
        bps_ch = bps_by_channel.get(ch)
        if bps_ch is None or len(bps_ch) == 0:
            continue
        weight, pz = mod.compute_merger_weights(
            bps_ch["delay_time"].values,
            bps_ch["metallicity"].values,
            bps_ch["formation_efficiency_per_solar_mass"].values,
            sfr_z,
            dPdlogZ,
            met_grid,
            p_draw,
            times_Myr,
            redshifts,
            shell_volumes,
            time_first_SF,
        )
        rate_z += (pz.astype(np.float64) * weight[:, np.newaxis]).sum(axis=0)

    return redshifts, rate_z


def plot_rate_curves(
    curves: List[Tuple[float, np.ndarray, np.ndarray]],
    *,
    vary: VaryAxis,
    fixed_sfra: float,
    fixed_mu0: float,
    nuisances: Dict[str, float],
    channels: Sequence[str],
    log_y: bool,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.2), constrained_layout=True)

    cmap = plt.get_cmap("viridis")
    n = max(len(curves), 1)
    for i, (param_val, z, rate_z) in enumerate(curves):
        color = cmap(i / max(n - 1, 1))
        if vary == "sfra":
            label = rf"$a_{{\mathrm{{SF}}}}={param_val:.4f}$"
        else:
            label = rf"$\mu_0={param_val:.4f}$"
        ax.plot(z, rate_z, color=color, lw=1.8, label=label)

    if vary == "sfra":
        title_fix = rf"$\mu_0={fixed_mu0:.4f}$ fixed"
        xvar = r"Vary $a_{\mathrm{SF}}$"
    else:
        title_fix = rf"$a_{{\mathrm{{SF}}}}={fixed_sfra:.4f}$ fixed"
        xvar = r"Vary $\mu_0$"

    ch_label = "+".join(channels)
    ax.set_xlabel(r"Redshift $z$")
    ax.set_ylabel(r"$R(z)$ [yr$^{-1}$ per $\Delta z$ shell]")
    ax.set_title(
        rf"Intrinsic merger rate vs $z$ ({xvar}; {title_fix}; {ch_label})",
        fontsize=11,
    )
    z_max = max(float(c[1].max()) for c in curves) if curves else 10.0
    ax.set_xlim(0.0, z_max)
    if log_y:
        all_pos = np.concatenate([c[2][c[2] > 0] for c in curves if np.any(c[2] > 0)])
        if all_pos.size:
            ax.set_yscale("log")
    ax.legend(fontsize=8, loc="best", framealpha=0.9)
    ax.grid(True, alpha=0.25)

    note = (
        "Nuisance params fixed at Step-00 best-fit values: "
        + ", ".join(f"{k}={v:.3g}" for k, v in sorted(nuisances.items()))
    )
    fig.text(0.01, 0.01, note, fontsize=6.5, ha="left", va="bottom", wrap=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot intrinsic R(z) for several Step 00 grid models.")
    p.add_argument(
        "--vary",
        choices=("sfra", "mu0"),
        required=True,
        help="Vary a_SF with fixed mu0, or vary mu0 with fixed a_SF.",
    )
    p.add_argument(
        "--n-curves",
        type=int,
        default=5,
        help="Number of curves (evenly spaced in range, or subsampled from HDF5 grid).",
    )
    p.add_argument(
        "--values",
        type=float,
        nargs="+",
        default=None,
        help="Explicit a_SF or mu0 values (overrides --n-curves spacing).",
    )
    p.add_argument(
        "--sspc-hdf5",
        type=Path,
        default=None,
        help="Optional Step 00 HDF5: if set, default curve values are taken from its keys.",
    )
    p.add_argument(
        "--bps-hdf5",
        type=Path,
        default=None,
        help="BPS catalog (default: data/bps_output.h5).",
    )
    p.add_argument(
        "--channels",
        nargs="+",
        default=["SMT", "CE", "CHE"],
        help="Formation channels to sum (default: all three).",
    )
    p.add_argument(
        "--log-y",
        action="store_true",
        help="Logarithmic y-axis.",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output PNG (default: plots/00_rate_vs_redshift/<timestamp>/rate_z_<vary>.png).",
    )
    p.add_argument(
        "--no-timestamp-subdir",
        action="store_true",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    vary: VaryAxis = args.vary
    mod = _load_sspc00_module()

    bps_path = (args.bps_hdf5 or mod.BPS_PATH_DEFAULT).resolve()
    hdf5_path = (args.sspc_hdf5 or find_data_dir() / "sspc" / "models_sspc.hdf5").resolve()

    nuisances = fixed_nuisance_params(mod)
    mid_sfra = default_mid_sfra(mod)
    mid_mu0 = default_mid_mu0(mod)

    param_vals = curve_values(
        mod,
        vary,
        n_curves=max(2, int(args.n_curves)),
        values=args.values,
        hdf5_path=hdf5_path if hdf5_path.is_file() else None,
    )

    print(f"Loading BPS: {bps_path}", flush=True)
    bps_full = mod.load_bps(bps_path)
    channels = [str(c) for c in args.channels]
    bps_by_channel = {
        ch: bps_full[bps_full["channel"] == ch].reset_index(drop=True)
        for ch in channels
    }

    curves: List[Tuple[float, np.ndarray, np.ndarray]] = []
    for pv in param_vals:
        sfr_a = pv if vary == "sfra" else mid_sfra
        mu0 = mid_mu0 if vary == "sfra" else pv
        print(f"  R(z): {vary}={pv:.4f}  (a_SF={sfr_a:.4f}, mu0={mu0:.4f})", flush=True)
        z, rate_z = rate_vs_redshift(
            mod,
            bps_by_channel,
            sfr_a=sfr_a,
            mu0=mu0,
            nuisances=nuisances,
            channels=channels,
        )
        curves.append((pv, z, rate_z))

    if args.out is not None:
        out = args.out.resolve()
    else:
        out = resolve_plot_output(
            Path(__file__),
            no_timestamp_subdir=args.no_timestamp_subdir,
            filename=f"rate_z_{vary}.png",
        )

    plot_rate_curves(
        curves,
        vary=vary,
        fixed_sfra=mid_sfra,
        fixed_mu0=mid_mu0,
        nuisances=nuisances,
        channels=channels,
        log_y=bool(args.log_y),
        out_path=out,
    )

    meta = {
        "vary": vary,
        "param_values": param_vals,
        "fixed_sfra": mid_sfra,
        "fixed_mu0": mid_mu0,
        "nuisances_tng100": nuisances,
        "channels": channels,
        "bps_hdf5": str(bps_path),
        "sspc_hdf5": str(hdf5_path) if hdf5_path.is_file() else None,
        "log_y": bool(args.log_y),
    }
    meta_path = out.with_suffix(".json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Saved → {meta_path}", flush=True)


if __name__ == "__main__":
    main()
