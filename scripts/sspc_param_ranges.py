"""
Van Son et al. (2022) TNG100 SSPC parameter ranges (Table 1, DOI:10.3847/1538-4357/acbf51).

All grid / nuisance sampling ranges are the published best-fit ± 1σ, clipped to
physically valid intervals (strictly positive where required for SFR/metallicity models).
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

# DOI: https://doi.org/10.3847/1538-4357/acbf51
VAN_SON_DOI = "10.3847/1538-4357/acbf51"

# Lower bound for parameters that must be > 0 (log-safe, positive SFR / metallicity scale).
SSPC_PARAM_EPS: float = 1e-6

# Table 1 best-fit values (TNG100).
VAN_SON_BEST_FIT: Dict[str, float] = {
    "sfr_a": 0.02,
    "mu0": 0.025,
    "sfr_b": 1.48,
    "sfr_c": 4.44,
    "sfr_d": 5.90,
    "muz": -0.049,
    "sigma0": 1.122,
    "sigmaz": 0.049,
    "alpha_skew": -1.778,
}

# Table 1 fit uncertainties (1σ).
VAN_SON_SIGMA: Dict[str, float] = {
    "sfr_a": 0.072,
    "mu0": 0.036,
    "sfr_b": 0.002,
    "sfr_c": 0.001,
    "sfr_d": 0.002,
    "muz": 0.006,
    "sigma0": 0.001,
    "sigmaz": 0.009,
    "alpha_skew": 0.002,
}

# Primary grid axes only: van Son ±1σ includes unphysical ≤0 values for these two.
STRICTLY_POSITIVE: frozenset[str] = frozenset({"sfr_a", "mu0"})

NUISANCE_KEYS: Tuple[str, ...] = (
    "sfr_b",
    "sfr_c",
    "sfr_d",
    "muz",
    "sigma0",
    "sigmaz",
    "alpha_skew",
)

DEFAULT_N_SFRA: int = 50
DEFAULT_N_MU0: int = 50


def van_son_1sigma_range(name: str) -> Tuple[float, float]:
    """Unclipped best-fit ± 1σ interval from van Son Table 1."""
    center = VAN_SON_BEST_FIT[name]
    sigma = VAN_SON_SIGMA[name]
    return center - sigma, center + sigma


def physical_range(name: str, lo: float, hi: float) -> Tuple[float, float]:
    """Clip to physically valid bounds (strict positivity where required)."""
    if name in STRICTLY_POSITIVE:
        lo = max(float(lo), SSPC_PARAM_EPS)
    return float(lo), float(hi)


def grid_range(name: str) -> Tuple[float, float]:
    """Published ± 1σ range, excluding unphysical values."""
    return physical_range(name, *van_son_1sigma_range(name))


def linspace_grid(name: str, n: int) -> np.ndarray:
    """``n`` evenly spaced points over the physical grid range for ``name``."""
    lo, hi = grid_range(name)
    return np.linspace(lo, hi, int(n), dtype=np.float64)


# Primary (sfr_a, mu0) training axes — re-exported by 00_sspc_data_generation.py.
SFRA_RANGE: Tuple[float, float] = grid_range("sfr_a")
MU0_RANGE: Tuple[float, float] = grid_range("mu0")

# Nuisance sampling ranges (one draw per grid cell unless fixed).
NUISANCE_RANGES: Dict[str, Tuple[float, float]] = {
    key: grid_range(key) for key in NUISANCE_KEYS
}

# Fixed nuisance values at van Son best-fit (``--fixed-nuisance-tng100``).
TNG100_BEST_FIT_NUISANCE: Dict[str, float] = {
    key: VAN_SON_BEST_FIT[key] for key in NUISANCE_KEYS
}
