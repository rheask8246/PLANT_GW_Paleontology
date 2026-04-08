"""
Diffusion Emulator: DDPM for conditional merger event generation.

Takes hyperparameter vector Λ and generates (mchirp, q, chieff, z) events.
Uses same normalization as CFM for direct comparison.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn


def normalize_obs(obs: np.ndarray, normalizer: Dict, columns: list = None) -> np.ndarray:
    """Same as cfm_emulator.normalize_obs."""
    if columns is None:
        columns = ["mchirp", "q", "chieff", "z"]
    out = np.zeros_like(obs, dtype=np.float32)
    for i, col in enumerate(columns):
        x = obs[:, i].astype(np.float64)
        if col in ("mchirp", "z"):
            x = np.log10(np.maximum(x, 1e-10))
        m, s = normalizer[col]["mean"], normalizer[col]["std"]
        out[:, i] = ((x - m) / (s + 1e-8)).astype(np.float32)
    return out


def denormalize_obs(obs: np.ndarray, normalizer: Dict, columns: list = None) -> np.ndarray:
    """Same as cfm_emulator.denormalize_obs."""
    if columns is None:
        columns = ["mchirp", "q", "chieff", "z"]
    out = np.zeros_like(obs, dtype=np.float64)
    for i, col in enumerate(columns):
        x = obs[:, i].astype(np.float64)
        m, s = normalizer[col]["mean"], normalizer[col]["std"]
        x = x * (s + 1e-8) + m
        if col in ("mchirp", "z"):
            x = 10.0 ** x
        out[:, i] = x
    return out


class EventEncoder(nn.Module):
    """Encodes lambda_vec into context (same as CFM)."""

    def __init__(self, input_dim: int = 7, hidden_dim: int = 128, output_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, lambda_vec: torch.Tensor) -> torch.Tensor:
        return self.net(lambda_vec)


class DenoisingNet(nn.Module):
    """Predicts noise epsilon given x_t, t, and context. Same input dims as CFM VectorFieldNet."""

    def __init__(self, context_dim: int = 128, hidden_dim: int = 256):
        super().__init__()
        self.input_dim = 4 + 1 + context_dim  # x_t(4) + t(1) + context
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 4),
        )

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        """Predict epsilon. x: (..., 4), t: (..., 1) or (...,), context: (..., 128)."""
        if t.dim() == x.dim() - 1:
            t = t.unsqueeze(-1)
        inp = torch.cat([x, t, context], dim=-1)
        return self.net(inp)


class DiffusionEmulator(nn.Module):
    """Conditional DDPM for event generation."""

    def __init__(
        self,
        lambda_dim: int = 7,
        context_dim: int = 128,
        hidden_dim: int = 256,
        n_timesteps: int = 100,
    ):
        super().__init__()
        self.n_timesteps = n_timesteps
        self.lambda_dim = lambda_dim
        self.encoder = EventEncoder(input_dim=lambda_dim, hidden_dim=context_dim, output_dim=context_dim)
        self.denoise = DenoisingNet(context_dim=context_dim, hidden_dim=hidden_dim)

        # Linear variance schedule (DDPM)
        self.register_buffer("betas", torch.linspace(1e-4, 0.02, n_timesteps))
        self.register_buffer("alphas", 1.0 - self.betas)
        self.register_buffer("alphas_cumprod", torch.cumprod(self.alphas, dim=0))
        self.register_buffer("alphas_cumprod_prev", torch.cat([torch.ones(1), self.alphas_cumprod[:-1]]))
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(self.alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - self.alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / self.alphas))
        self.register_buffer("posterior_variance", self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod))

    def _encode_context(self, lambda_vec: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Expand context to match x batch shape."""
        context = self.encoder(lambda_vec)
        if context.dim() == 2 and x.dim() == 3:
            context = context.unsqueeze(1).expand(-1, x.shape[1], -1)
        return context

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        lambda_vec: torch.Tensor,
    ) -> torch.Tensor:
        """Predict noise epsilon. x: (..., 4), t: (...,) in [0, n_timesteps-1]."""
        t_norm = t.float() / (self.n_timesteps - 1)  # [0, 1] for network
        context = self._encode_context(lambda_vec, x)
        if t.dim() == 1 and x.dim() == 3:
            t_norm = t_norm.unsqueeze(1).expand(-1, x.shape[1])
        return self.denoise(x, t_norm, context)

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward diffusion: add noise to x_0."""
        if noise is None:
            noise = torch.randn_like(x_start)
        t_int = t.long().clamp(0, self.n_timesteps - 1)
        sqrt_alpha = self.sqrt_alphas_cumprod[t_int]
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t_int]
        while sqrt_alpha.dim() < x_start.dim():
            sqrt_alpha = sqrt_alpha.unsqueeze(-1)
            sqrt_one_minus = sqrt_one_minus.unsqueeze(-1)
        return sqrt_alpha * x_start + sqrt_one_minus * noise

    def p_sample_step(self, x_t: torch.Tensor, t: int, lambda_vec: torch.Tensor) -> torch.Tensor:
        """Single reverse step from t to t-1."""
        t_norm = torch.full((x_t.shape[0],), t / max(1, self.n_timesteps - 1), device=x_t.device, dtype=x_t.dtype)
        if x_t.dim() == 3:
            t_norm = t_norm.unsqueeze(1).expand(-1, x_t.shape[1])
        context = self._encode_context(lambda_vec, x_t)
        eps_pred = self.denoise(x_t, t_norm, context)

        alpha_t = self.alphas[t].item()
        alpha_cumprod_t = self.alphas_cumprod[t].item()
        beta_t = self.betas[t].item()
        sqrt_alpha_cumprod_t = self.sqrt_alphas_cumprod[t].item()
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].item()

        # Predict x_0 from x_t and eps
        x_0_pred = (x_t - sqrt_one_minus_alpha_cumprod_t * eps_pred) / sqrt_alpha_cumprod_t
        if t == 0:
            return x_0_pred

        # Posterior mean (DDPM formula)
        mean = (1.0 / (alpha_t ** 0.5)) * (x_t - beta_t / sqrt_one_minus_alpha_cumprod_t * eps_pred)
        posterior_var = self.posterior_variance[t].item()
        noise = torch.randn_like(x_t)
        return mean + (posterior_var ** 0.5) * noise

    def sample(
        self,
        lambda_vec: torch.Tensor,
        n_samples: int,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Generate samples via reverse diffusion."""
        if device is None:
            device = next(self.parameters()).device
        lambda_vec = lambda_vec.to(device)
        if lambda_vec.dim() == 1:
            lambda_vec = lambda_vec.unsqueeze(0)

        x = torch.randn(1, n_samples, 4, device=device, dtype=lambda_vec.dtype)
        for t in reversed(range(self.n_timesteps)):
            x = self.p_sample_step(x, t, lambda_vec)
        return x.squeeze(0)


def generate_catalog(
    lambda_vec: np.ndarray,
    n_events: int,
    model: "DiffusionEmulator",
    normalizer: Dict,
) -> "pd.DataFrame":
    """Generate catalog (same interface as CFM)."""
    import pandas as pd

    model.eval()
    lam = torch.from_numpy(np.asarray(lambda_vec, dtype=np.float32))
    if lam.dim() == 1:
        lam = lam.unsqueeze(0)
    with torch.no_grad():
        x = model.sample(lam, n_events)
    x_np = x.cpu().numpy()
    x_denorm = denormalize_obs(x_np, normalizer)
    x_denorm[:, 1] = np.clip(x_denorm[:, 1], 0.0, 1.0)
    x_denorm[:, 2] = np.clip(x_denorm[:, 2], -1.0, 1.0)
    x_denorm[:, 3] = np.maximum(x_denorm[:, 3], 1e-6)
    x_denorm[:, 0] = np.maximum(x_denorm[:, 0], 1e-2)
    return pd.DataFrame(x_denorm, columns=["mchirp", "q", "chieff", "z"])
