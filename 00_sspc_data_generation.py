#!/usr/bin/env python3
"""
SSPC-based GW event data generation  (intrinsic / pdet-free).

Performs cosmic integration of COMPAS/GROWL BPS output over the Madau-Dickinson
star formation rate and a log-skew-normal metallicity distribution (Neijssel+19),
sampling events from the INTRINSIC merger-rate distribution across all redshifts.

No detection-probability weighting is applied: the output represents the full
physical population of merging compact binaries, regardless of observability.

════════════════════════════════════════════════════════════════════════════════
INPUTS
────────────────────────────────────────────────────────────────────────────────
data/bps_output.h5
    COMPAS/GROWL binary population.  Columns per row:
        dco_mass_1                      [Msun]  primary remnant mass
        dco_mass_2                      [Msun]  secondary remnant mass
        delay_time                      [Myr]   inspiral delay time
        metallicity                     [Z_sun] progenitor metallicity
        formation_efficiency_per_solar_mass  [Msun^-1]  BPS formation weight
        formation_channel               (int)   1=SMT, 2=CE, 3=CHE, 4→SMT, -1=skip

════════════════════════════════════════════════════════════════════════════════
HYPERPARAMETER GRID (the "λ" training axes)
────────────────────────────────────────────────────────────────────────────────
Primary grid axes (varied on a dense regular grid):
    sfr_a  (aSF)  SFR Madau-Dickinson amplitude  [0.010, 0.030]   N_SFRA values
    mu0           mean metallicity at z = 0       [0.010, 0.060]   N_MU0 values

Nuisance parameters (one random draw per grid point, stored as sspc_* columns).
Ranges are centred on TNG100-1 best-fit values (Briel+):
    sfr_b      [1.0,  3.0]   MD14 rising slope      (TNG100 best-fit ≈ 1.46)
    sfr_c      [2.0,  6.0]   MD14 turnover redshift (TNG100 best-fit ≈ 4.51)
    sfr_d      [4.0,  8.0]   MD14 falling slope     (TNG100 best-fit ≈ 6.21)
    muz        [-0.5, 0.1]   metallicity evo. slope (TNG100 best-fit ≈ -0.052)
    sigma0     [0.5,  1.5]   metallicity log-spread (TNG100 best-fit ≈ 1.15)
    sigmaz     [-0.1, 0.1]   redshift evo. of spread(TNG100 best-fit ≈ 0.047)
    alpha_skew [-2.0, 2.0]   log-skew-normal skewness(TNG100 best-fit ≈ -1.85)

════════════════════════════════════════════════════════════════════════════════
OUTPUTS
────────────────────────────────────────────────────────────────────────────────
data/sspc/models_sspc.hdf5  (pandas HDFStore, format="table")

HDF5 key structure:
    /CE/sfra{NNNN}/mu0{MMMM}   — CE channel, aSF=NNNN/10000, mu0=MMMM/10000
    /CHE/sfra{NNNN}/mu0{MMMM}  — CHE channel
    /SMT/sfra{NNNN}/mu0{MMMM}  — SMT channel

Columns per key:
    mchirp              [Msun]   source-frame chirp mass
    q                   [—]      mass ratio m2/m1 ∈ (0,1]
    chieff              [—]      effective aligned spin ∈ [-1,1]
    z                   [—]      merger redshift (intrinsic-rate weighted)
    weight              [—]      intrinsic merger rate weight (merger/yr/binary)
    sspc_sfr_a/b/c/d    [—]      Madau-Dickinson SFR parameters used
    sspc_mu0/muz/sigma0/sigmaz  [—] metallicity distribution parameters used
    sspc_alpha_skew     [—]      metallicity skewness parameter used

════════════════════════════════════════════════════════════════════════════════
PIPELINE COMPATIBILITY
────────────────────────────────────────────────────────────────────────────────
The output is consumed by 02_build_dataset.py:
    python 02_build_dataset.py --hdf5 data/sspc/models_sspc.hdf5 --data-source sspc
════════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from typing import Dict, Tuple

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.cosmology import Planck18 as cosmology
from scipy.interpolate import interp1d
from scipy.stats import norm as NormDist

# ═════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═════════════════════════════════════════════════════════════════════════════

_HERE = Path(__file__).resolve().parent

BPS_PATH_DEFAULT = _HERE / "data" / "bps_output.h5"
OUT_PATH_DEFAULT = _HERE / "data" / "sspc" / "models_sspc.hdf5"

# Formation channel integer → channel name
CHANNEL_MAP: Dict[int, str] = {1: "SMT", 2: "CE", 3: "CHE", 4: "SMT"}
CHANNEL_NAMES = ["CE", "CHE", "SMT"]

# Primary hyperparameter grid ranges
# Centred on TNG100-1 best-fit (sfr_a≈0.017, mu0≈0.025)
SFRA_RANGE = (0.010, 0.030)   # Madau-Dickinson amplitude aSF [Msun/yr/Mpc³]
MU0_RANGE  = (0.010, 0.060)   # mean metallicity at z=0

# Nuisance parameter sampling ranges
# Centred on / covering TNG100-1 best-fit values (Briel+ Table 1):
#   sfr_a=0.0170, sfr_b=1.456, sfr_c=4.514, sfr_d=6.210
#   mu0=0.0249, muz=-0.0519, sigma0(omega0)=1.151, sigmaz(omegaz)=0.0474
#   alpha_skew(alpha0)=-1.854
NUISANCE_RANGES = {
    "sfr_b":      (1.0,   3.0),   # MD14 rising slope           (TNG ≈ 1.46)
    "sfr_c":      (2.0,   6.0),   # MD14 turnover redshift       (TNG ≈ 4.51)
    "sfr_d":      (4.0,   8.0),   # MD14 falling slope           (TNG ≈ 6.21)
    "muz":        (-0.5,  0.1),   # metallicity evolution slope  (TNG ≈ -0.052)
    "sigma0":     (0.5,   1.5),   # metallicity log-spread σ₀    (TNG ≈ 1.15)
    "sigmaz":     (-0.1,  0.1),   # redshift evo. of spread      (TNG ≈ 0.047)
    "alpha_skew": (-2.0,  2.0),   # log-skew-normal skewness     (TNG ≈ -1.85)
}

# Chieff generation per channel (mean, std) — no spin info in BPS
CHIEFF_PARAMS: Dict[str, Tuple[float, float]] = {
    "CE":  (0.00, 0.10),   # random orientation from CE → near-zero chieff
    "CHE": (0.25, 0.15),   # tidal synchronisation → higher, aligned spins
    "SMT": (0.05, 0.12),   # partial tidal alignment in stable mass transfer
}

# Cosmological integration grid
MAX_REDSHIFT  = 10.0   # integrate to z=10 (covers all significant SF history)
REDSHIFT_STEP = 0.1    # coarse grid for speed
Z_FIRST_SF    = 10.0   # first star-formation redshift


# ═════════════════════════════════════════════════════════════════════════════
# COSMOLOGICAL UTILITIES
# ═════════════════════════════════════════════════════════════════════════════

def build_redshift_grid() -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Build redshift, time, and comoving shell-volume arrays."""
    redshifts    = np.arange(0.0, MAX_REDSHIFT + REDSHIFT_STEP, REDSHIFT_STEP)
    times_Myr    = cosmology.age(redshifts).to(u.Myr).value
    volumes_Gpc3 = cosmology.comoving_volume(redshifts).to(u.Gpc**3).value
    shell_volumes = np.diff(volumes_Gpc3)
    shell_volumes = np.append(shell_volumes, shell_volumes[-1])
    time_first_SF = cosmology.age(Z_FIRST_SF).to(u.Myr).value
    return redshifts, times_Myr, shell_volumes, time_first_SF


def find_sfr(redshifts: np.ndarray, a: float, b: float,
             c: float, d: float) -> np.ndarray:
    """
    Madau & Dickinson 2014 SFRD in Msun/yr/Gpc³.
    ψ(z) = a (1+z)^b / [1 + ((1+z)/c)^d]
    """
    sfr = a * (1.0 + redshifts)**b / (1.0 + ((1.0 + redshifts) / c)**d)
    return sfr * (u.Msun / u.yr / u.Mpc**3).to(u.Msun / u.yr / u.Gpc**3)


def find_metallicity_distribution(
    redshifts:    np.ndarray,
    min_logZ_bps: float,
    max_logZ_bps: float,
    mu0:    float = 0.035,
    muz:    float = -0.23,
    sigma0: float = 0.39,
    sigmaz: float = 0.0,
    alpha:  float = 0.0,
    min_logZ:  float = -12.0,
    max_logZ:  float =   0.0,
    step_logZ: float =   0.01,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Log-skew-normal metallicity distribution dP/dlogZ(z).
    Returns dPdlogZ [n_z × n_Z], metallicities [n_Z], p_draw.
    Follows Neijssel+19 / Langer & Norman 2007.
    """
    sigma   = sigma0 * 10.0 ** (sigmaz * redshifts)
    mean_Z  = mu0   * 10.0 ** (muz    * redshifts)

    beta   = alpha / np.sqrt(1.0 + alpha**2)
    PHI    = NormDist.cdf(beta * sigma)
    mu_logZ = np.log(mean_Z / (2.0 * PHI)) - sigma**2 / 2.0

    log_metallicities = np.arange(min_logZ, max_logZ + step_logZ, step_logZ)
    metallicities     = np.exp(log_metallicities)

    dPdlogZ = (
        2.0 / sigma[:, np.newaxis]
        * NormDist.pdf(
            (log_metallicities - mu_logZ[:, np.newaxis]) / sigma[:, np.newaxis]
        )
        * NormDist.cdf(
            alpha * (log_metallicities - mu_logZ[:, np.newaxis]) / sigma[:, np.newaxis]
        )
    )
    norm    = dPdlogZ.sum(axis=-1) * step_logZ
    dPdlogZ = dPdlogZ / np.where(norm > 0, norm, 1.0)[:, np.newaxis]

    p_draw = 1.0 / (max_logZ_bps - min_logZ_bps)
    return dPdlogZ, metallicities, p_draw


# ═════════════════════════════════════════════════════════════════════════════
# VECTORISED COSMIC INTEGRATION (intrinsic rate only — no pdet)
# ═════════════════════════════════════════════════════════════════════════════

def compute_merger_weights(
    delay_times_Myr: np.ndarray,
    metallicities:   np.ndarray,
    formation_eff:   np.ndarray,
    sfr_z:           np.ndarray,
    dPdlogZ:         np.ndarray,
    met_grid:        np.ndarray,
    p_draw:          float,
    times_Myr:       np.ndarray,
    redshifts:       np.ndarray,
    shell_volumes:   np.ndarray,
    time_first_SF:   float,
    chunk_size:      int = 50_000,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Vectorised computation of per-binary intrinsic merger rate weights.

    For each binary i:
      weight[i]  = Σ_z  feff * sfr(z_form) * dP/p_draw * shell_vol   [merger/yr]
      pz[i, z]   = rate[i, z] / weight[i]  (probability distribution over z for sampling)

    No detection-probability weighting is applied; all redshifts contribute.
    """
    n_binary = len(delay_times_Myr)
    n_z      = len(redshifts)

    t_to_z = interp1d(
        times_Myr[::-1], redshifts[::-1],
        bounds_error=False, fill_value=(redshifts[-1], 0.0),
    )

    Z_idx_all = np.clip(np.searchsorted(met_grid, metallicities) - 1,
                        0, len(met_grid) - 1)

    weight = np.zeros(n_binary, dtype=np.float64)
    pz     = np.zeros((n_binary, n_z), dtype=np.float32)

    for i0 in range(0, n_binary, chunk_size):
        i1 = min(i0 + chunk_size, n_binary)
        nc = i1 - i0

        td    = delay_times_Myr[i0:i1]
        feff  = formation_eff[i0:i1]
        Z_idx = Z_idx_all[i0:i1]

        t_form = times_Myr[np.newaxis, :] - td[:, np.newaxis]   # (nc, n_z)
        valid  = (t_form > time_first_SF) & (t_form < times_Myr[0])

        t_form_safe = np.where(valid, t_form, times_Myr[0])
        z_form      = t_to_z(t_form_safe)

        sfr_at_form = np.interp(z_form.ravel(), redshifts, sfr_z).reshape(nc, n_z)

        # Use ceil to map z_form → metallicity grid index (matches FastCosmicIntegration)
        z_form_idx = np.clip(
            np.ceil(z_form / REDSHIFT_STEP).astype(np.int32), 0, n_z - 1
        )
        dP = dPdlogZ[z_form_idx, Z_idx[:, np.newaxis]]

        # Intrinsic merger rate per z bin: feff × sfr × dP/p_draw × shell_vol
        rate = np.where(
            valid,
            feff[:, np.newaxis] * sfr_at_form * dP / p_draw * shell_volumes[np.newaxis, :],
            0.0,
        )                                                         # (nc, n_z)

        w = rate.sum(axis=1)
        weight[i0:i1] = w

        w_safe = np.where(w > 0, w, 1.0)
        pz[i0:i1] = (rate / w_safe[:, np.newaxis]).astype(np.float32)

    return weight, pz


# ═════════════════════════════════════════════════════════════════════════════
# CHIEFF GENERATION
# ═════════════════════════════════════════════════════════════════════════════

def sample_chieff(channel: str, n: int, rng: np.random.Generator) -> np.ndarray:
    """
    Sample chieff from a channel-specific Gaussian (no spin info in BPS).
    CE  : N(0.00, 0.10) — random orientation → near-zero chieff
    CHE : N(0.25, 0.15) — tidal synchronisation → aligned, higher spins
    SMT : N(0.05, 0.12) — partial tidal alignment
    """
    mu, sig = CHIEFF_PARAMS[channel]
    return np.clip(rng.normal(mu, sig, size=n), -1.0, 1.0).astype(np.float32)


# ═════════════════════════════════════════════════════════════════════════════
# BPS DATA LOADING
# ═════════════════════════════════════════════════════════════════════════════

def _is_git_lfs_pointer(path: Path) -> bool:
    """True if `path` is a checked-in Git LFS stub instead of real file contents."""
    try:
        with path.open("rb") as f:
            head = f.read(128)
    except OSError:
        return False
    return head.startswith(b"version https://git-lfs.github.com/spec/v1")


def load_bps(path: Path) -> pd.DataFrame:
    """Load COMPAS/GROWL BPS output from HDF5 (pandas format)."""
    if not path.is_file():
        raise FileNotFoundError(f"BPS HDF5 not found: {path}")
    if _is_git_lfs_pointer(path):
        raise RuntimeError(
            f"{path} is a Git LFS pointer (small text stub), not the real HDF5. "
            "After clone/pull, run:  git lfs install && git lfs pull\n"
            "Or copy the actual bps_output.h5 onto this machine / set --bps-hdf5."
        ) from None
    df = pd.read_hdf(str(path), key="input_data")
    if "formation_channel" not in df.columns:
        raise KeyError("BPS file missing 'formation_channel' column.")
    df["formation_channel"] = df["formation_channel"].astype(int)

    m1 = df["dco_mass_1"].values
    m2 = df["dco_mass_2"].values
    df["mchirp"] = (m1 * m2)**0.6 / (m1 + m2)**0.2
    df["q"]      = np.minimum(m1, m2) / np.maximum(m1, m2)  # q ∈ (0,1]

    df["channel"] = df["formation_channel"].map(CHANNEL_MAP)
    df = df[df["channel"].notna()].copy()

    print(f"Loaded {len(df):,} DCO systems from {path.name}")
    for ch in CHANNEL_NAMES:
        n = (df["channel"] == ch).sum()
        print(f"  {ch}: {n:,}  ({100*n/len(df):.1f}%)")
    return df


# ═════════════════════════════════════════════════════════════════════════════
# KEY FORMATTING
# ═════════════════════════════════════════════════════════════════════════════

def _sfra_key(sfr_a: float) -> str:
    return f"sfra{int(round(sfr_a * 10_000)):04d}"

def _mu0_key(mu0: float) -> str:
    return f"mu0{int(round(mu0 * 10_000)):04d}"


# ═════════════════════════════════════════════════════════════════════════════
# MAIN: GENERATE DATA
# ═════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="SSPC-based GW event data generation (intrinsic, pdet-free)"
    )
    parser.add_argument("--bps-hdf5", type=Path, default=BPS_PATH_DEFAULT)
    parser.add_argument("--output-hdf5", type=Path, default=OUT_PATH_DEFAULT)
    parser.add_argument("--n-sfra",   type=int, default=8)
    parser.add_argument("--n-mu0",    type=int, default=8)
    parser.add_argument("--n-events", type=int, default=50_000)
    parser.add_argument("--seed",     type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    out = args.output_hdf5
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and not args.overwrite:
        raise FileExistsError(f"{out} exists. Use --overwrite to replace.")

    rng = np.random.default_rng(args.seed)

    # ── 1. Load BPS data ────────────────────────────────────────────────────
    bps_full = load_bps(args.bps_hdf5)

    # ── 2. Build cosmological grid ──────────────────────────────────────────
    print("\nBuilding cosmological grid …")
    redshifts, times_Myr, shell_volumes, time_first_SF = build_redshift_grid()
    n_z = len(redshifts)

    logZ_min = np.log(bps_full["metallicity"].min())
    logZ_max = np.log(bps_full["metallicity"].max())

    # ── 3. Define parameter grid ────────────────────────────────────────────
    sfra_vals = np.linspace(*SFRA_RANGE, args.n_sfra)
    mu0_vals  = np.linspace(*MU0_RANGE,  args.n_mu0)

    print(f"\nSSPC grid: {args.n_sfra} × {args.n_mu0} = {args.n_sfra * args.n_mu0} "
          f"param sets × {len(CHANNEL_NAMES)} channels = "
          f"{args.n_sfra * args.n_mu0 * len(CHANNEL_NAMES)} grid points")
    print(f"  sfr_a  : {sfra_vals[0]:.4f} – {sfra_vals[-1]:.4f}  (TNG100 best-fit ≈ 0.017)")
    print(f"  mu0    : {mu0_vals[0]:.4f}  – {mu0_vals[-1]:.4f}  (TNG100 best-fit ≈ 0.025)")
    print(f"  events per grid point: {args.n_events:,}")
    print(f"  Sampling: INTRINSIC merger rate (no pdet weighting)")
    print(f"  Redshift range: 0 – {MAX_REDSHIFT}")
    print(f"  Nuisance parameter ranges (centred on TNG100-1 best-fit):")
    for k, (lo, hi) in NUISANCE_RANGES.items():
        print(f"    {k:15s}: [{lo}, {hi}]")

    # ── 4. Write output ──────────────────────────────────────────────────────
    n_written = 0
    with pd.HDFStore(str(out), mode="w") as store:
        for channel in CHANNEL_NAMES:
            bps_ch = bps_full[bps_full["channel"] == channel].reset_index(drop=True)
            if len(bps_ch) == 0:
                warnings.warn(f"No BPS systems for channel {channel}, skipping.")
                continue

            n_ch = len(bps_ch)
            delay_times   = bps_ch["delay_time"].values
            metallicities = bps_ch["metallicity"].values
            formation_eff = bps_ch["formation_efficiency_per_solar_mass"].values
            mchirp_bps    = bps_ch["mchirp"].values.astype(np.float32)
            q_bps         = bps_ch["q"].values.astype(np.float32)

            print(f"\n{'='*70}")
            print(f"Channel: {channel}  ({n_ch:,} binaries)")

            for sfr_a in sfra_vals:
                for mu0 in mu0_vals:
                    # Sample nuisance parameters for this grid point
                    sfr_b      = float(rng.uniform(*NUISANCE_RANGES["sfr_b"]))
                    sfr_c      = float(rng.uniform(*NUISANCE_RANGES["sfr_c"]))
                    sfr_d      = float(rng.uniform(*NUISANCE_RANGES["sfr_d"]))
                    muz        = float(rng.uniform(*NUISANCE_RANGES["muz"]))
                    sigma0     = float(rng.uniform(*NUISANCE_RANGES["sigma0"]))
                    sigmaz     = float(rng.uniform(*NUISANCE_RANGES["sigmaz"]))
                    alpha_skew = float(rng.uniform(*NUISANCE_RANGES["alpha_skew"]))

                    # ── SFR(z) ───────────────────────────────────────────────
                    sfr_z_vals = find_sfr(redshifts, sfr_a, sfr_b, sfr_c, sfr_d)

                    # ── Metallicity distribution ─────────────────────────────
                    dPdlogZ, met_grid, p_draw = find_metallicity_distribution(
                        redshifts, logZ_min, logZ_max,
                        mu0=mu0, muz=muz, sigma0=sigma0, sigmaz=sigmaz,
                        alpha=alpha_skew,
                    )

                    # ── Per-binary intrinsic merger rate weights ──────────────
                    weight, pz = compute_merger_weights(
                        delay_times, metallicities, formation_eff,
                        sfr_z_vals, dPdlogZ, met_grid, p_draw,
                        times_Myr, redshifts, shell_volumes, time_first_SF,
                    )

                    weight = np.where(np.isfinite(weight), weight, 0.0)

                    total_rate = weight.sum()
                    if total_rate < 1e-30:
                        warnings.warn(
                            f"  {channel}/sfra={sfr_a:.4f}/mu0={mu0:.4f}: "
                            "zero intrinsic weight, skipping."
                        )
                        continue

                    # ── Sample events from the intrinsic population ──────────
                    prob   = weight / weight.sum()
                    ev_idx = rng.choice(n_ch, size=args.n_events,
                                        replace=True, p=prob)

                    # Draw merger redshift from intrinsic-rate distribution
                    pz_ev  = pz[ev_idx].astype(np.float64)
                    pz_ev /= pz_ev.sum(axis=1, keepdims=True) + 1e-30
                    cum    = np.cumsum(pz_ev, axis=1)
                    u_draw = rng.random(args.n_events)[:, np.newaxis]
                    z_idx  = (cum < u_draw).sum(axis=1)
                    # Clip to z ≥ 0.1 (avoid z=0 → log10(0) = -inf)
                    z_idx  = np.clip(z_idx, 1, n_z - 1)
                    z_ev   = redshifts[z_idx].astype(np.float32)

                    # ── Observables ──────────────────────────────────────────
                    mchirp_ev = mchirp_bps[ev_idx]
                    q_ev      = q_bps[ev_idx]
                    chieff_ev = sample_chieff(channel, args.n_events, rng)
                    weight_ev = weight[ev_idx].astype(np.float32)

                    # ── Assemble DataFrame ───────────────────────────────────
                    df_out = pd.DataFrame({
                        "mchirp":          mchirp_ev,
                        "q":               q_ev,
                        "chieff":          chieff_ev,
                        "z":               z_ev,
                        "weight":          weight_ev,
                        "sspc_sfr_a":      np.float32(sfr_a),
                        "sspc_sfr_b":      np.float32(sfr_b),
                        "sspc_sfr_c":      np.float32(sfr_c),
                        "sspc_sfr_d":      np.float32(sfr_d),
                        "sspc_mu0":        np.float32(mu0),
                        "sspc_muz":        np.float32(muz),
                        "sspc_sigma0":     np.float32(sigma0),
                        "sspc_sigmaz":     np.float32(sigmaz),
                        "sspc_alpha_skew": np.float32(alpha_skew),
                    })

                    key = f"/{channel}/{_sfra_key(sfr_a)}/{_mu0_key(mu0)}"
                    store.put(key, df_out, format="table", data_columns=True)
                    n_written += 1

                    print(f"  {key}  | total_w={total_rate:.3e} "
                          f"| median_z={np.median(z_ev):.2f} "
                          f"| m1=[{mchirp_ev.min():.1f},{mchirp_ev.max():.1f}]")

    print(f"\n{'='*70}")
    print("Done.")
    print(f"Output HDF5 : {out}")
    print(f"Grid points : {n_written}")
    print(f"Events/grid : {args.n_events:,}")
    print(f"\nNext step:")
    print(f"  python 02_build_dataset.py --hdf5 {out} --data-source sspc")


if __name__ == "__main__":
    main()
