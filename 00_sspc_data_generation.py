#!/usr/bin/env python3
"""
SSPC-based GW event data generation.

Uses real COMPAS/GROWL binary population synthesis (BPS) output convolved with
the Madau-Dickinson star formation rate and a log-skew-normal metallicity
distribution, following the cosmic integration framework of Neijssel+19.

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

data/SNR_Grid_IMRPhenomPv2_FD_all_noise.hdf5
    Pre-computed SNR grid for detection probability interpolation.

════════════════════════════════════════════════════════════════════════════════
HYPERPARAMETER GRID (the "λ" training axes)
────────────────────────────────────────────────────────────────────────────────
Primary grid axes (varied on a dense regular grid):
    sfr_a  (aSF)  SFR Madau-Dickinson amplitude  [0.012, 0.030]   N_SFRA values
    mu0           mean metallicity at z = 0       [0.010, 0.060]   N_MU0 values

Nuisance parameters (one random draw per grid point, stored as sspc_* columns):
    sfr_b, sfr_c, sfr_d   Madau-Dickinson shape (Eq. 6 of Neijssel+19)
    muz                   redshift slope of mean metallicity
    sigma0                metallicity spread at z = 0
    sigmaz                redshift evolution of spread
    alpha_skew            log-skew-normal skewness (0 = log-normal)

════════════════════════════════════════════════════════════════════════════════
OUTPUTS
────────────────────────────────────────────────────────────────────────────────
data/sspc/models_sspc.hdf5  (pandas HDFStore, format="table")

HDF5 key structure:
    /CE/sfra{NNNN}/mu0{MMMM}   — CE channel, aSF=NNNN/10000, mu0=MMMM/10000
    /CHE/sfra{NNNN}/mu0{MMMM}  — CHE channel
    /SMT/sfra{NNNN}/mu0{MMMM}  — SMT channel

Columns per key:
    mchirp                      [Msun]   detector-frame chirp mass
    q                           [—]      mass ratio m2/m1 ∈ (0,1]
    chieff                      [—]      effective aligned spin ∈ [-1,1]
    z                           [—]      merger redshift
    weight                      [—]      cosmic merger rate weight (merger/yr/Gpc³)
    pdet_midhighlatelow_network [—]      detection probability (O3 sensitivity)
    sspc_sfr_a/b/c/d            [—]      Madau-Dickinson SFR parameters used
    sspc_mu0/muz/sigma0/sigmaz  [—]      metallicity distribution parameters used
    sspc_alpha_skew             [—]      metallicity skewness parameter used

════════════════════════════════════════════════════════════════════════════════
PIPELINE COMPATIBILITY
────────────────────────────────────────────────────────────────────────────────
The output is consumed by 02_build_dataset.py:
    python 02_build_dataset.py --hdf5 data/sspc/models_sspc.hdf5 --data-source sspc

The sfra/mu0 key slots play the same role as chi_b/alpha_CE in the Zenodo data:
    sfr_a  → conditioning axis 1 (normalised to [0,1] by 02_build_dataset.py)
    mu0    → conditioning axis 2 (normalised to [0,1] by 02_build_dataset.py)
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

# ── local copy of selection_effects (in this directory) ──────────────────────
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import selection_effects  # noqa: E402  (local copy with data/ path fix)

# ═════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═════════════════════════════════════════════════════════════════════════════

BPS_PATH_DEFAULT = _HERE / "data" / "bps_output.h5"
OUT_PATH_DEFAULT = _HERE / "data" / "sspc" / "models_sspc.hdf5"

# Formation channel integer → channel name
# van Son / GROWL convention: 1=SMT, 2=CE, 3=CHE, 4=SMT-like, -1=unclassified
CHANNEL_MAP: Dict[int, str] = {1: "SMT", 2: "CE", 3: "CHE", 4: "SMT"}
CHANNEL_NAMES = ["CE", "CHE", "SMT"]

# Primary hyperparameter grid ranges
SFRA_RANGE = (0.008, 0.035)   # Madau-Dickinson amplitude aSF
MU0_RANGE  = (0.005, 0.065)   # mean metallicity at z=0

# Nuisance parameter sampling ranges
NUISANCE_RANGES = {
    "sfr_b":      (1.0,   3.5),   # MD14 power-law rising slope
    "sfr_c":      (2.0,   5.5),   # MD14 turnover redshift
    "sfr_d":      (3.5,   6.5),   # MD14 falling slope
    "muz":        (-0.5,   0.1),  # metallicity evolution slope
    "sigma0":     (0.2,   0.6),   # metallicity spread at z=0
    "sigmaz":     (-0.1,  0.1),   # redshift evolution of spread
    "alpha_skew": (-2.0,  2.0),   # log-skew-normal skewness
}

# Chieff generation per channel (mean, std) — no spin info in BPS
CHIEFF_PARAMS: Dict[str, Tuple[float, float]] = {
    "CE":  (0.00, 0.10),   # random orientation from CE → near-zero chieff
    "CHE": (0.25, 0.15),   # tidal synchronisation → higher, aligned spins
    "SMT": (0.05, 0.12),   # partial tidal alignment from SMT
}

# Cosmological integration grid
MAX_REDSHIFT        = 10.0
MAX_REDSHIFT_DET    = 1.5    # detection probability computed to this z
REDSHIFT_STEP       = 0.1    # coarse grid (step=0.1, N_z=100) for speed
Z_FIRST_SF          = 10.0   # first star-formation redshift

# IMF / population normalization (matches COMPAS defaults)
M1_MIN  = 5.0    # Msun
M1_MAX  = 150.0  # Msun
M2_MIN  = 0.1    # Msun
FBIN    = 0.7

SNR_THRESHOLD = 8.0
SENSITIVITY   = "O3"


# ═════════════════════════════════════════════════════════════════════════════
# COSMOLOGICAL UTILITIES
# ═════════════════════════════════════════════════════════════════════════════

def build_redshift_grid() -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Build redshift, time, and comoving shell-volume arrays.

    Returns
    -------
    redshifts, times_Myr, shell_volumes_Gpc3, time_first_SF_Myr
    """
    redshifts = np.arange(0.0, MAX_REDSHIFT + REDSHIFT_STEP, REDSHIFT_STEP)
    times_Myr = cosmology.age(redshifts).to(u.Myr).value
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
    redshifts: np.ndarray,
    min_logZ_bps: float,
    max_logZ_bps: float,
    mu0: float    = 0.035,
    muz: float    = -0.23,
    sigma0: float = 0.39,
    sigmaz: float = 0.0,
    alpha: float  = 0.0,
    min_logZ: float = -12.0,
    max_logZ: float =   0.0,
    step_logZ: float =   0.01,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Log-skew-normal metallicity distribution dP/dlogZ(z).

    Returns dPdlogZ [n_z × n_Z], metallicities [n_Z], p_draw_metallicity.
    Follows Neijssel+19 / Langer & Norman 2007.
    """
    sigma = sigma0 * 10.0 ** (sigmaz * redshifts)
    mean_Z = mu0 * 10.0 ** (muz * redshifts)

    beta = alpha / np.sqrt(1.0 + alpha**2)
    PHI  = NormDist.cdf(beta * sigma)
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


def star_forming_mass_per_binary(m1_min: float, m1_max: float,
                                  m2_min: float, fbin: float) -> float:
    """
    Representative star-forming mass per sampled binary (Kroupa IMF).
    """
    from scipy.integrate import quad

    def kroupa(m):
        if m < 0.5:
            return m**-1.3
        elif m < 1.0:
            return 0.5**(-0.3) * m**-2.3
        else:
            return 0.5**(-0.3) * m**-2.3

    total_mass, _ = quad(lambda m: m * kroupa(m), 0.1, 300.0)
    n_singles, _  = quad(kroupa, m1_min, m1_max)
    n_primary     = fbin * n_singles
    # rough: average secondary mass ~ (m1_min + m1_max)/2 * 0.5
    mass_per_binary = total_mass / (n_primary + 1e-12)
    return mass_per_binary


# ═════════════════════════════════════════════════════════════════════════════
# VECTORISED COSMIC INTEGRATION
# ═════════════════════════════════════════════════════════════════════════════

def compute_merger_weights(
    delay_times_Myr: np.ndarray,
    metallicities:   np.ndarray,
    formation_eff:   np.ndarray,
    mchirp_bps:      np.ndarray,
    q_bps:           np.ndarray,
    n_formed_z:      np.ndarray,
    dPdlogZ:         np.ndarray,
    met_grid:        np.ndarray,
    p_draw:          float,
    times_Myr:       np.ndarray,
    redshifts:       np.ndarray,
    shell_volumes:   np.ndarray,
    distances_Mpc:   np.ndarray,
    time_first_SF:   float,
    snr_grid:        np.ndarray,
    pdet_snr:        np.ndarray,
    Mc_step:         float,
    eta_step:        float,
    snr_step:        float,
    n_det:           int,
    chunk_size:      int = 50_000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Vectorised computation of per-binary merger rate weights.

    For each binary i, computes:
      - weight[i]     : total merger rate (sum over all z)
      - det_weight[i] : detection-rate weight (sum over z < MAX_REDSHIFT_DET,
                        weighted by p_det(binary, z))  ← used for event sampling
      - pz_det[i]     : normalised detection-rate probability per z bin (for z draw)

    Sampling from det_weight ensures all generated events have pdet > 0.
    """
    n_binary = len(delay_times_Myr)
    n_z      = len(redshifts)

    t_to_z = interp1d(
        times_Myr[::-1], redshifts[::-1],
        bounds_error=False, fill_value=(redshifts[-1], 0.0),
    )

    Z_idx_all = np.clip(np.searchsorted(met_grid, metallicities) - 1,
                        0, len(met_grid) - 1)

    weight     = np.zeros(n_binary, dtype=np.float64)
    det_weight = np.zeros(n_binary, dtype=np.float64)
    pz_det     = np.zeros((n_binary, n_det), dtype=np.float32)

    dist_det = distances_Mpc[:n_det]   # luminosity distances for z < MAX_REDSHIFT_DET

    for i0 in range(0, n_binary, chunk_size):
        i1 = min(i0 + chunk_size, n_binary)
        nc = i1 - i0

        td    = delay_times_Myr[i0:i1]
        feff  = formation_eff[i0:i1]
        Z_idx = Z_idx_all[i0:i1]
        mc    = mchirp_bps[i0:i1]
        q_c   = q_bps[i0:i1]

        t_form = times_Myr[np.newaxis, :] - td[:, np.newaxis]   # (nc, n_z)
        valid  = (t_form > time_first_SF) & (t_form < times_Myr[0])

        t_form_safe = np.where(valid, t_form, times_Myr[0])
        z_form      = t_to_z(t_form_safe)

        nf = np.interp(z_form.ravel(), redshifts, n_formed_z).reshape(nc, n_z)

        z_form_idx = np.clip(
            np.round(z_form / REDSHIFT_STEP).astype(np.int32), 0, n_z - 1
        )
        dP = dPdlogZ[z_form_idx, Z_idx[:, np.newaxis]]

        rate = np.where(
            valid,
            feff[:, np.newaxis] * nf * dP / p_draw * shell_volumes[np.newaxis, :],
            0.0,
        )                                                         # (nc, n_z)

        weight[i0:i1] = rate.sum(axis=1)

        # ── Detection-rate weight (only z < MAX_REDSHIFT_DET) ────────────────
        # SNR at 1 Mpc for each binary
        eta     = q_c / (1.0 + q_c) ** 2
        eta_idx = np.clip(np.round(eta / eta_step).astype(int) - 1,
                          0, snr_grid.shape[0] - 1)
        Mc_det  = mc[:, np.newaxis] * (1.0 + redshifts[:n_det][np.newaxis, :])  # (nc, n_det)
        Mc_idx  = np.clip(np.round(Mc_det / Mc_step).astype(int) - 1,
                          0, snr_grid.shape[1] - 1)
        snr_1Mpc = snr_grid[eta_idx[:, np.newaxis], Mc_idx]       # (nc, n_det)

        snr_z    = snr_1Mpc / np.maximum(dist_det[np.newaxis, :], 1e-3)
        det_idx  = np.clip(np.round(snr_z / snr_step).astype(int) - 1,
                           0, len(pdet_snr) - 1)
        pdet_z   = np.where(det_idx >= 0, pdet_snr[det_idx], 0.0)  # (nc, n_det)

        det_rate_chunk = rate[:, :n_det] * pdet_z               # (nc, n_det)
        dw = det_rate_chunk.sum(axis=1)
        det_weight[i0:i1] = dw

        dw_safe = np.where(dw > 0, dw, 1.0)
        pz_det[i0:i1] = (det_rate_chunk / dw_safe[:, np.newaxis]).astype(np.float32)

    return weight, det_weight, pz_det


# ═════════════════════════════════════════════════════════════════════════════
# DETECTION PROBABILITY
# ═════════════════════════════════════════════════════════════════════════════

def build_snr_grid(sensitivity: str = SENSITIVITY):
    """Return (snr_grid_at_1Mpc, detection_prob_from_snr, Mc_step, eta_step, snr_step)."""
    interp = selection_effects.SNRinterpolator(sensitivity)

    Mc_step, eta_step, snr_step = 0.1, 0.01, 0.1
    Mc_arr  = np.arange(Mc_step, 300.0 + Mc_step, Mc_step)
    eta_arr = np.arange(eta_step, 0.25 + eta_step, eta_step)

    Mt  = Mc_arr / eta_arr[:, np.newaxis]**0.6
    M1  = Mt * 0.5 * (1.0 + np.sqrt(np.maximum(1.0 - 4 * eta_arr[:, np.newaxis], 0)))
    M2  = Mt - M1

    snr_grid = interp(M1, M2)

    snr_arr  = np.arange(snr_step, 1000.0 + snr_step, snr_step)
    pdet_snr = selection_effects.detection_probability_from_snr(snr_arr, SNR_THRESHOLD)

    return snr_grid, pdet_snr, Mc_step, eta_step, snr_step


def compute_pdet_vectorised(
    mchirp: np.ndarray,
    q:      np.ndarray,
    z:      np.ndarray,
    distances_at_z: np.ndarray,
    redshifts:      np.ndarray,
    snr_grid:       np.ndarray,
    pdet_snr:       np.ndarray,
    Mc_step: float, eta_step: float, snr_step: float,
) -> np.ndarray:
    """
    Vectorised detection probability at each event's merger redshift.
    """
    eta = q / (1.0 + q)**2
    dist = np.interp(z, redshifts, distances_at_z)

    Mc_shifted = mchirp * (1.0 + z)                            # detector-frame chirp mass

    eta_idx = np.clip(np.round(eta / eta_step).astype(int) - 1,
                      0, snr_grid.shape[0] - 1)
    Mc_idx  = np.clip(np.round(Mc_shifted / Mc_step).astype(int) - 1,
                      0, snr_grid.shape[1] - 1)

    snr_at_1Mpc = snr_grid[eta_idx, Mc_idx]
    snr         = snr_at_1Mpc / np.maximum(dist, 1e-3)

    det_idx = np.clip(np.round(snr / snr_step).astype(int) - 1,
                      0, len(pdet_snr) - 1)
    pdet    = np.where(det_idx >= 0, pdet_snr[det_idx], 0.0)
    return pdet.astype(np.float32)


# ═════════════════════════════════════════════════════════════════════════════
# CHIEFF GENERATION
# ═════════════════════════════════════════════════════════════════════════════

def sample_chieff(channel: str, n: int, rng: np.random.Generator) -> np.ndarray:
    """
    Sample chieff from a channel-specific distribution.

    CE  : N(0.00, 0.10) — random orientation after common-envelope
    CHE : N(0.25, 0.15) — tidal synchronisation → aligned, higher spins
    SMT : N(0.05, 0.12) — partial tidal alignment in stable mass transfer
    """
    mu, sig = CHIEFF_PARAMS[channel]
    chieff  = rng.normal(mu, sig, size=n)
    return np.clip(chieff, -1.0, 1.0).astype(np.float32)


# ═════════════════════════════════════════════════════════════════════════════
# BPS DATA LOADING
# ═════════════════════════════════════════════════════════════════════════════

def load_bps(path: Path) -> pd.DataFrame:
    """
    Load COMPAS/GROWL BPS output from HDF5 (pandas format).

    Returned columns: dco_mass_1, dco_mass_2, delay_time, metallicity,
                      formation_efficiency_per_solar_mass, formation_channel,
                      mchirp, q
    """
    df = pd.read_hdf(str(path), key="input_data")

    # Ensure formation_channel is integer
    if "formation_channel" not in df.columns:
        raise KeyError("BPS file missing 'formation_channel' column.")
    df["formation_channel"] = df["formation_channel"].astype(int)

    # Derived observables
    m1 = df["dco_mass_1"].values
    m2 = df["dco_mass_2"].values
    df["mchirp"] = (m1 * m2)**0.6 / (m1 + m2)**0.2
    df["q"]      = np.minimum(m1, m2) / np.maximum(m1, m2)  # q ∈ (0,1]

    # Map channel codes → names and filter unknowns
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
    parser = argparse.ArgumentParser(description="SSPC-based GW event data generation")
    parser.add_argument("--bps-hdf5", type=Path, default=BPS_PATH_DEFAULT,
                        help="Path to COMPAS/GROWL BPS output HDF5.")
    parser.add_argument("--output-hdf5", type=Path, default=OUT_PATH_DEFAULT,
                        help="Output HDF5 path.")
    parser.add_argument("--n-sfra", type=int, default=8,
                        help="Grid points along sfr_a axis.")
    parser.add_argument("--n-mu0", type=int, default=8,
                        help="Grid points along mu0 axis.")
    parser.add_argument("--n-events", type=int, default=50_000,
                        help="Events sampled per grid point.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--sensitivity", type=str, default=SENSITIVITY,
                        choices=["design", "O1", "O3"],
                        help="GW detector sensitivity for pdet.")
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
    distances_Mpc = cosmology.luminosity_distance(redshifts).to(u.Mpc).value
    n_z  = len(redshifts)
    n_det = int(MAX_REDSHIFT_DET / REDSHIFT_STEP) + 1  # bins within detection horizon

    # log metallicity bounds from BPS data
    logZ_min = np.log(bps_full["metallicity"].min())
    logZ_max = np.log(bps_full["metallicity"].max())

    # ── 3. Build SNR / pdet grid ────────────────────────────────────────────
    print("Building SNR detection grid …")
    snr_grid, pdet_snr, Mc_step, eta_step, snr_step = build_snr_grid(args.sensitivity)

    # ── 4. Define parameter grid ────────────────────────────────────────────
    sfra_vals = np.linspace(*SFRA_RANGE, args.n_sfra)
    mu0_vals  = np.linspace(*MU0_RANGE,  args.n_mu0)

    print(f"\nSSPC grid: {args.n_sfra} × {args.n_mu0} = {args.n_sfra * args.n_mu0} "
          f"param sets × {len(CHANNEL_NAMES)} channels = "
          f"{args.n_sfra * args.n_mu0 * len(CHANNEL_NAMES)} grid points")
    print(f"  sfr_a  : {sfra_vals[0]:.4f} – {sfra_vals[-1]:.4f}")
    print(f"  mu0    : {mu0_vals[0]:.4f}  – {mu0_vals[-1]:.4f}")
    print(f"  events per grid point: {args.n_events:,}")
    print(f"  nuisance parameter ranges:")
    for k, (lo, hi) in NUISANCE_RANGES.items():
        print(f"    {k:15s}: [{lo}, {hi}]")

    # ── 5. Compute mass_evolved_per_binary for n_formed normalisation ────────
    mass_per_binary = star_forming_mass_per_binary(M1_MIN, M1_MAX, M2_MIN, FBIN)

    # ── 6. Write output ──────────────────────────────────────────────────────
    n_written = 0
    with pd.HDFStore(str(out), mode="w") as store:
        for channel in CHANNEL_NAMES:
            bps_ch = bps_full[bps_full["channel"] == channel].reset_index(drop=True)
            if len(bps_ch) == 0:
                warnings.warn(f"No BPS systems for channel {channel}, skipping.")
                continue

            n_ch = len(bps_ch)
            delay_times = bps_ch["delay_time"].values          # Myr
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

                    # ── SFR(z) and n_formed(z) ──────────────────────────────
                    sfr_z    = find_sfr(redshifts, sfr_a, sfr_b, sfr_c, sfr_d)
                    n_formed = sfr_z / mass_per_binary            # [1/yr/Gpc³]

                    # ── Metallicity distribution dP/dlogZ(z) ────────────────
                    dPdlogZ, met_grid, p_draw = find_metallicity_distribution(
                        redshifts, logZ_min, logZ_max,
                        mu0=mu0, muz=muz, sigma0=sigma0, sigmaz=sigmaz,
                        alpha=alpha_skew,
                    )

                    # ── Per-binary merger rate weights ───────────────────────
                    weight, det_weight, pz_det = compute_merger_weights(
                        delay_times, metallicities, formation_eff,
                        mchirp_bps, q_bps,
                        n_formed, dPdlogZ, met_grid, p_draw,
                        times_Myr, redshifts, shell_volumes, distances_Mpc,
                        time_first_SF,
                        snr_grid, pdet_snr, Mc_step, eta_step, snr_step,
                        n_det,
                    )

                    weight     = np.where(np.isfinite(weight),     weight,     0.0)
                    det_weight = np.where(np.isfinite(det_weight), det_weight, 0.0)

                    if det_weight.sum() < 1e-30:
                        warnings.warn(
                            f"  {channel}/sfra={sfr_a:.4f}/mu0={mu0:.4f}: "
                            "zero detectable weight, skipping."
                        )
                        continue

                    # ── Sample events from the DETECTABLE population ─────────
                    # Use det_weight so all drawn events have pdet > 0
                    prob = det_weight / det_weight.sum()
                    ev_idx = rng.choice(n_ch, size=args.n_events,
                                        replace=True, p=prob)

                    # Draw merger redshift from detection-rate distribution
                    pz_ev  = pz_det[ev_idx].astype(np.float64)   # (N_ev, n_det)
                    pz_ev /= pz_ev.sum(axis=1, keepdims=True) + 1e-30
                    cum    = np.cumsum(pz_ev, axis=1)
                    u_draw = rng.random(args.n_events)[:, np.newaxis]
                    z_idx  = (cum < u_draw).sum(axis=1)
                    # Start from bin 1 (z=0.1) — z=0 is unphysical and causes
                    # log10(0)=-inf which corrupts the obs_normalizer.
                    z_idx  = np.clip(z_idx, 1, n_det - 1)
                    z_ev   = redshifts[z_idx].astype(np.float32)  # z in [0.1, MAX_REDSHIFT_DET]

                    # ── Observables ──────────────────────────────────────────
                    mchirp_ev = mchirp_bps[ev_idx]
                    q_ev      = q_bps[ev_idx]
                    chieff_ev = sample_chieff(channel, args.n_events, rng)

                    # ── Detection probability ────────────────────────────────
                    pdet_ev = compute_pdet_vectorised(
                        mchirp_ev, q_ev, z_ev,
                        distances_Mpc, redshifts,
                        snr_grid, pdet_snr, Mc_step, eta_step, snr_step,
                    )

                    # ── Detection-rate weight per sampled event ──────────────
                    # Events are drawn from the detectable population; store
                    # det_weight so that downstream weight*pdet gives a sensible
                    # importance weight and sum_weight ∝ total detection rate.
                    weight_ev = det_weight[ev_idx].astype(np.float32)

                    # ── Assemble DataFrame ───────────────────────────────────
                    df_out = pd.DataFrame({
                        "mchirp":                      mchirp_ev,
                        "q":                           q_ev,
                        "chieff":                      chieff_ev,
                        "z":                           z_ev,
                        "weight":                      weight_ev,
                        "pdet_midhighlatelow_network": pdet_ev,
                        # SSPC hyperparameters for this grid point
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

                    print(f"  {key}  | det_w={det_weight.sum():.3e} "
                          f"| median_z={np.median(z_ev):.2f} "
                          f"| mean_pdet={pdet_ev.mean():.3f}")

    print(f"\n{'='*70}")
    print("Done.")
    print(f"Output HDF5  : {out}")
    print(f"Grid points  : {n_written}")
    print(f"Events/grid  : {args.n_events:,}")
    print(f"\nNext step:")
    print(f"  python 02_build_dataset.py --hdf5 {out} --data-source sspc")


if __name__ == "__main__":
    main()
