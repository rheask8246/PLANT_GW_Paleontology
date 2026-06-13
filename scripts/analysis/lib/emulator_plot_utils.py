"""Shared helpers for emulator validation plots (04 / 04b / 04c analysis)."""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd

def sample_events_from_grid(
    events_df: pd.DataFrame,
    grid_idx: int,
    n: int,
    rng: np.random.Generator,
    z_jitter: bool = True,
) -> np.ndarray:
    """Uniform row subsample for grid_idx (expect intrinsic `all_events` from 02). Returns (n,3) [mchirp,q,z]."""
    mask = events_df["grid_idx"] == grid_idx
    sub = events_df.loc[mask, ["mchirp", "q", "z"]]
    if len(sub) == 0:
        return np.zeros((n, 3), dtype=np.float32)
    idx = rng.integers(0, len(sub), size=min(n, len(sub)))
    if len(idx) < n:
        idx = rng.choice(len(sub), size=n, replace=True)
    x = sub.iloc[idx].values.astype(np.float32)
    if z_jitter:
        # Smooth the discrete z grid (bin width 0.1) so the model learns a
        # continuous distribution rather than a delta function at each bin edge.
        # IMPORTANT: do not hard-code the z upper bound (older pipelines used z<=1.55).
        # Use the dataset range so training/analysis matches the current SSPC output.
        z_hi = float(events_df["z"].max()) if "z" in events_df.columns and len(events_df) else 10.0
        x[:, 2] = np.clip(
            x[:, 2] + rng.uniform(-0.05, 0.05, size=n).astype(np.float32),
            1e-6,
            z_hi,
        )
    return x



CHI_B_RANGE = (0.0, 0.5)
ALPHA_CE_RANGE = (0.2, 5.0)

# Must match scripts/sspc_param_ranges.py (Step 00 / 02 normalization).
from sspc_param_ranges import MU0_RANGE as SSPC_MU0_RANGE  # noqa: E402
from sspc_param_ranges import SFRA_RANGE as SSPC_SFRA_RANGE  # noqa: E402


def is_sspc_hyperparam_df(hp_df: pd.DataFrame) -> bool:
    """SSPC tables from 02 use sfra/mu0 instead of chi_b/alpha_CE."""
    return "sfra" in hp_df.columns


def sspc_interp_lambda(p1: float, p2: float, lam_template: np.ndarray) -> np.ndarray:
    """
    Update lambda_3 (sfra_norm) and lambda_4 (mu0_norm) with fixed ranges; keep
    channel one-hot (lambda_0..2) and nuisances (lambda_5..) from lam_template.
    """
    lam = np.array(lam_template, dtype=np.float32).copy()
    s0, s1 = SSPC_SFRA_RANGE
    m0, m1 = SSPC_MU0_RANGE
    lam[3] = (p1 - s0) / (s1 - s0) if s1 > s0 else 0.0
    lam[4] = (p2 - m0) / (m1 - m0) if m1 > m0 else 0.0
    return lam


def ce_lambda_vec(
    chi_b: float,
    alpha_ce: float,
    lambda_template: np.ndarray,
    chi_range: Tuple[float, float],
    alpha_range: Tuple[float, float],
) -> np.ndarray:
    """Build lambda_vec for CE channel from (chi_b, alpha_CE)."""
    chi_min, chi_max = chi_range
    alpha_min, alpha_max = alpha_range
    chi_norm = (chi_b - chi_min) / (chi_max - chi_min) if chi_max > chi_min else 0.0
    alpha_norm = (alpha_ce - alpha_min) / (alpha_max - alpha_min) if alpha_max > alpha_min else 0.0
    lam = np.array(lambda_template, dtype=np.float32).copy()
    lam[0:5] = np.array([1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    lam[5] = chi_norm
    lam[6] = alpha_norm
    return lam


def histogram_kl(x_true: np.ndarray, x_syn: np.ndarray, bins: int = 50) -> float:
    """KL divergence (true || syn) using histograms. Returns np.inf if bins have zeros."""
    lo = min(x_true.min(), x_syn.min())
    hi = max(x_true.max(), x_syn.max())
    if hi <= lo:
        return 0.0
    bin_edges = np.linspace(lo, hi, bins + 1)
    p, _ = np.histogram(x_true, bins=bin_edges, density=True)
    q, _ = np.histogram(x_syn, bins=bin_edges, density=True)
    eps = 1e-10
    p = p + eps
    q = q + eps
    p = p / p.sum()
    q = q / q.sum()
    from scipy.stats import entropy
    return float(entropy(p, q))


def mmd_rbf(x: np.ndarray, y: np.ndarray, gamma: float = None) -> float:
    """MMD^2 with RBF kernel. x, y: (n, d)."""
    if gamma is None:
        # Median heuristic
        xx = np.sum(x ** 2, axis=1, keepdims=True)
        yy = np.sum(y ** 2, axis=1, keepdims=True)
        xy = x @ y.T
        dxx = xx + xx.T - 2 * xy
        dyy = yy + yy.T - 2 * (y @ y.T)
        dxy = xx + yy.T - 2 * xy
        all_d = np.concatenate([dxx.ravel(), dyy.ravel(), dxy.ravel()])
        gamma = 1.0 / (2 * np.median(all_d[all_d > 0]) + 1e-8)
    n, m = len(x), len(y)
    kxx = np.exp(-gamma * np.sum((x[:, None] - x[None, :]) ** 2, axis=2))
    kyy = np.exp(-gamma * np.sum((y[:, None] - y[None, :]) ** 2, axis=2))
    kxy = np.exp(-gamma * np.sum((x[:, None] - y[None, :]) ** 2, axis=2))
    mmd2 = kxx.sum() / (n * n) + kyy.sum() / (m * m) - 2 * kxy.sum() / (n * m)
    return max(0.0, mmd2) ** 0.5


