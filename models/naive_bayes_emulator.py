"""
Naive Bayes emulator baseline: conditional p(mchirp, q, z | Λ) without neural training.

Pipeline diagram: ``docs/step04c_naive_bayes_emulator_flow.png``
(same figure at ``docs/figures/04c_naive_bayes_pipeline.png``).

Step 04c (``scripts/04c_naive_bayes_emulator.py``)
-------------------------------------------------
1. Load Step 02: ``hyperparam_table_encoded.csv``, ``all_events.parquet``, ``obs_normalizer.json``.
2. ``NaiveBayesEmulator.fit_from_data(hp_df, events_df, normalizer)`` — one CPU pass, no SGD.
3. Save ``checkpoints/naive_bayes_final.pt`` (buffers + normalizer + ``lambda_cols`` + mode + τ).
4. Plots (optional, no refit): ``scripts/analysis/04c_naive_bayes_emulator_plots.py``.

Naive Bayes architecture (this module)
--------------------------------------
Stored per training grid point *g* (≈7500 rows):

* ``grid_lambdas[g]`` — encoded Λ_g (``lambda_0..lambda_{d-1}``)
* ``grid_mu[g]``, ``grid_sigma[g]`` — mean / std of normalized (mchirp, q, z) from 02 parquet
* ``kernel_bandwidth`` τ — median pairwise ‖Λ_i − Λ_j‖ on the grid

**gaussian** — kernel-weighted mixture::

    π_g(Λ) ∝ exp(−‖Λ − Λ_g‖² / 2τ²)   →   sample grid g ~ π, then x ~ N(μ_g, diag σ_g²)

**nearest** — empirical resample from argmin_g ‖Λ − Λ_g‖ (packed ``events_norm_packed`` + ``grid_ptr``).

``generate_catalog(Λ, n, model, normalizer)`` denormalizes to physical (mchirp, q, z).

Diffusion emulator architecture (04b baseline, ``models/diffusion_emulator.py``)
-------------------------------------------------------------------------------
For comparison with CFM/04b (gradient-trained generative model):

* **EventEncoder**: Λ → MLP (Linear–GELU–…) → context vector (dim 128).
* **DenoisingNet**: concat [x_t, t, context] → MLP → predicted noise ε̂ ∈ R³.
* **DiffusionEmulator**: DDPM with T=100, linear β schedule; training predicts ε;
  sampling runs reverse chain x_T ~ N(0,I) → x_0, then ``denormalize_obs``.

Same ``generate_catalog(Λ, n, model, normalizer)`` interface as CFM and this NB baseline.

Modes:
  gaussian — mixture of per-grid diagonal Gaussians with Λ-kernel weights
  nearest  — subsample empirical events from the nearest grid row
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from models.cfm_emulator import denormalize_obs, normalize_obs


def _lambda_cols(df: pd.DataFrame) -> List[str]:
    return sorted(
        [c for c in df.columns if c.startswith("lambda_")],
        key=lambda x: int(x.split("_")[1]),
    )


def _median_pairwise_bandwidth(lambdas: np.ndarray) -> float:
    """Heuristic τ: median pairwise ‖Λ_i − Λ_j‖ on the training grid."""
    n = lambdas.shape[0]
    if n < 2:
        return 1.0
    # Subsample for large grids
    if n > 400:
        rng = np.random.default_rng(0)
        idx = rng.choice(n, size=400, replace=False)
        lam = lambdas[idx]
    else:
        lam = lambdas
    m = lam.shape[0]
    d2 = np.sum((lam[:, None, :] - lam[None, :, :]) ** 2, axis=-1)
    tri = d2[np.triu_indices(m, k=1)]
    if tri.size == 0:
        return 1.0
    med = float(np.median(np.sqrt(tri)))
    return max(med, 1e-4)


def _pack_events_by_grid(
    events_df: pd.DataFrame,
    *,
    n_grid: int,
    obs_cols: List[str],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Pack events into one contiguous array, grouped by grid_idx.

    Returns:
      obs_sorted: (N, len(obs_cols)) float32
      ptr: (n_grid+1,) int64 offsets into obs_sorted for each grid
    """
    grid = events_df["grid_idx"].values.astype(np.int64, copy=False)
    order = np.argsort(grid, kind="mergesort")
    grid_sorted = grid[order]
    obs_sorted = events_df[obs_cols].values.astype(np.float32, copy=False)[order]
    counts = np.bincount(grid_sorted.clip(min=0, max=n_grid - 1), minlength=n_grid).astype(
        np.int64, copy=False
    )
    ptr = np.zeros(n_grid + 1, dtype=np.int64)
    ptr[1:] = np.cumsum(counts)
    return obs_sorted, ptr


class NaiveBayesEmulator(nn.Module):
    """
    Non-neural generative emulator with the same sample / generate_catalog API as CFM.
    """

    def __init__(self, lambda_dim: int = 7, mode: str = "gaussian", n_grid: int = 0):
        super().__init__()
        self.lambda_dim = lambda_dim
        self.mode = mode
        self.register_buffer("grid_lambdas", torch.zeros(n_grid, lambda_dim))
        self.register_buffer("grid_mu", torch.zeros(n_grid, 3))
        self.register_buffer("grid_sigma", torch.ones(n_grid, 3))
        self.register_buffer("kernel_bandwidth", torch.tensor(1.0))
        # Packed normalized events for nearest mode: (total_events, 3), ptr (n_grid+1,)
        self.register_buffer("events_norm_packed", torch.zeros(0, 3))
        self.register_buffer("grid_ptr", torch.zeros(n_grid + 1, dtype=torch.long))

    @classmethod
    def fit_from_data(
        cls,
        hp_df: pd.DataFrame,
        events_df: pd.DataFrame,
        normalizer: Dict,
        *,
        mode: str = "gaussian",
        bandwidth: Optional[float] = None,
        sigma_floor: float = 0.05,
    ) -> "NaiveBayesEmulator":
        lambda_cols = _lambda_cols(hp_df)
        n_grid = len(hp_df)
        lambdas = hp_df[lambda_cols].values.astype(np.float32)

        obs_cols = ["mchirp", "q", "z"]
        # Pack + normalize once (vectorized) to avoid per-grid pandas groupby overhead.
        raw_sorted, ptr = _pack_events_by_grid(events_df, n_grid=n_grid, obs_cols=obs_cols)
        norm_sorted = normalize_obs(raw_sorted, normalizer).astype(np.float32, copy=False)

        counts = (ptr[1:] - ptr[:-1]).astype(np.int64, copy=False)
        mu = np.zeros((n_grid, 3), dtype=np.float32)
        sigma = np.ones((n_grid, 3), dtype=np.float32) * sigma_floor

        nz = counts > 0
        if np.any(nz):
            starts = ptr[:-1][nz]
            # Sum + sumsq per grid (reduceat expects sorted contiguous blocks)
            sum_x = np.add.reduceat(norm_sorted, starts, axis=0)
            sum_x2 = np.add.reduceat(norm_sorted * norm_sorted, starts, axis=0)
            c = counts[nz].astype(np.float32)[:, None]
            mu[nz] = sum_x / np.maximum(c, 1.0)
            var = sum_x2 / np.maximum(c, 1.0) - mu[nz] * mu[nz]
            sigma[nz] = np.maximum(np.sqrt(np.maximum(var, 0.0)), sigma_floor)

        if bandwidth is None:
            bandwidth = _median_pairwise_bandwidth(lambdas)

        model = cls(lambda_dim=len(lambda_cols), mode=mode)
        model.grid_lambdas = torch.from_numpy(lambdas)
        model.grid_mu = torch.from_numpy(mu)
        model.grid_sigma = torch.from_numpy(sigma)
        model.kernel_bandwidth = torch.tensor(float(bandwidth))

        if mode == "nearest":
            model.events_norm_packed = torch.from_numpy(norm_sorted)
            model.grid_ptr = torch.from_numpy(ptr)
        else:
            model.events_norm_packed = torch.zeros(0, 3)
            model.grid_ptr = torch.zeros(n_grid + 1, dtype=torch.long)
        return model

    def _grid_weights(self, lambda_vec: torch.Tensor) -> torch.Tensor:
        """π_g(Λ) ∝ exp(-‖Λ−Λ_g‖² / 2τ²). Returns (n_grid,) on same device."""
        lam = lambda_vec.reshape(-1)
        diff = self.grid_lambdas - lam.unsqueeze(0)
        d2 = (diff ** 2).sum(dim=-1)
        tau2 = (self.kernel_bandwidth ** 2).clamp(min=1e-12)
        logits = -0.5 * d2 / tau2
        logits = logits - logits.max()
        w = torch.softmax(logits, dim=0)
        return w

    def _nearest_grid_index(self, lambda_vec: torch.Tensor) -> int:
        lam = lambda_vec.reshape(-1)
        diff = self.grid_lambdas - lam.unsqueeze(0)
        return int(torch.argmin((diff ** 2).sum(dim=-1)).item())

    def _sample_nearest(
        self,
        lambda_vec: torch.Tensor,
        n_samples: int,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        g = self._nearest_grid_index(lambda_vec)
        start = int(self.grid_ptr[g].item())
        end = int(self.grid_ptr[g + 1].item())
        device = lambda_vec.device
        if end <= start:
            return self.grid_mu[g].unsqueeze(0).expand(n_samples, -1).clone()

        pool = self.events_norm_packed[start:end]
        n_pool = end - start
        idx = torch.randint(0, n_pool, (n_samples,), generator=generator)
        return pool[idx]

    def _sample_gaussian(
        self,
        lambda_vec: torch.Tensor,
        n_samples: int,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        device = lambda_vec.device
        w = self._grid_weights(lambda_vec)
        g_idx = torch.multinomial(w, n_samples, replacement=True, generator=generator)
        mu = self.grid_mu[g_idx]
        sig = self.grid_sigma[g_idx]
        eps = torch.randn(n_samples, 3, generator=generator, device=device, dtype=mu.dtype)
        return mu + sig * eps

    def sample(
        self,
        lambda_vec: torch.Tensor,
        n_samples: int,
        device: Optional[torch.device] = None,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Generate n_samples events in normalized (mchirp, q, z) space."""
        if device is not None:
            lambda_vec = lambda_vec.to(device)
        else:
            device = self.grid_lambdas.device
        if lambda_vec.dim() == 1:
            lambda_vec = lambda_vec.unsqueeze(0)
        lam = lambda_vec[0]

        if self.mode == "nearest":
            out = self._sample_nearest(lam, n_samples, generator=generator)
        else:
            out = self._sample_gaussian(lam, n_samples, generator=generator)
        return out.to(device)


def generate_catalog(
    lambda_vec: np.ndarray,
    n_events: int,
    model: NaiveBayesEmulator,
    normalizer: Dict,
) -> pd.DataFrame:
    """Same interface as CFM / diffusion generate_catalog."""
    model.eval()
    lam = torch.from_numpy(np.asarray(lambda_vec, dtype=np.float32))
    with torch.inference_mode():
        x = model.sample(lam, n_events)
    x_np = x.cpu().numpy()
    if x_np.ndim == 1:
        x_np = x_np.reshape(1, -1)
    x_denorm = denormalize_obs(x_np, normalizer)
    x_denorm[:, 1] = np.clip(x_denorm[:, 1], 0.0, 1.0)
    x_denorm[:, 2] = np.maximum(x_denorm[:, 2], 1e-6)
    x_denorm[:, 0] = np.maximum(x_denorm[:, 0], 1e-2)
    return pd.DataFrame(x_denorm, columns=["mchirp", "q", "z"])


def save_checkpoint(
    path: Path,
    model: NaiveBayesEmulator,
    normalizer: Dict,
    lambda_cols: List[str],
    seed: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "normalizer": normalizer,
            "lambda_cols": lambda_cols,
            "mode": model.mode,
            "kernel_bandwidth": float(model.kernel_bandwidth.item()),
            "seed": seed,
        },
        path,
    )


def load_from_checkpoint(
    ck: Dict,
    device: Optional[torch.device] = None,
) -> Tuple[NaiveBayesEmulator, List[str], Dict]:
    lambda_cols: List[str] = ck["lambda_cols"]
    nrm: Dict = ck["normalizer"]
    mode = ck.get("mode", "gaussian")
    state = ck["model_state"]
    n_grid = int(state["grid_lambdas"].shape[0])
    lambda_dim = int(state["grid_lambdas"].shape[1])
    m = NaiveBayesEmulator(lambda_dim=lambda_dim, mode=mode, n_grid=n_grid)
    if mode == "nearest" and "events_norm_packed" in state:
        n_events = int(state["events_norm_packed"].shape[0])
        m.events_norm_packed = torch.zeros(n_events, 3)
        m.grid_ptr = torch.zeros(n_grid + 1, dtype=torch.long)
    m.load_state_dict(state, strict=True)
    if device is not None:
        m.to(device)
    m.eval()
    return m, lambda_cols, nrm
