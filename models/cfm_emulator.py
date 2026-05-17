"""
CFM Emulator: Conditional Flow Matching for merger event generation.

Takes hyperparameter vector Λ and generates (mchirp, q, z) events.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


def normalize_obs(obs: np.ndarray, normalizer: Dict, columns: list = None) -> np.ndarray:
    """
    Normalize observables for CFM input. Same transform as 02_build_dataset stats.

    mchirp, z: log10(x) FIRST, then x_norm = (log10(x) - mean) / std
              (mean, std match 02’s `all_events` / obs_normalizer.json)
    q: x_norm = (x - mean) / std directly

    obs: (N, 3) with columns [mchirp, q, z]
    """
    if columns is None:
        columns = ["mchirp", "q", "z"]
    out = np.zeros_like(obs, dtype=np.float32)
    for i, col in enumerate(columns):
        x = obs[:, i].astype(np.float64)
        if col in ("mchirp", "z"):
            x = np.log10(np.maximum(x, 1e-10))
        m, s = normalizer[col]["mean"], normalizer[col]["std"]
        out[:, i] = ((x - m) / (s + 1e-8)).astype(np.float32)
    return out


def denormalize_obs(obs: np.ndarray, normalizer: Dict, columns: list = None) -> np.ndarray:
    """
    Denormalize observables back to physical units.
    Inverts normalize_obs: x_denorm = 10^(x_norm * std + mean) for mchirp, z;
    x_denorm = x_norm * std + mean for q.
    """
    if columns is None:
        columns = ["mchirp", "q", "z"]
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
    """Encodes the conditioning variable Λ into a context vector."""

    def __init__(self, input_dim: int = 7, hidden_dim: int = 128, output_dim: int = 128,
                 dropout: float = 0.05):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, lambda_vec: torch.Tensor) -> torch.Tensor:
        return self.net(lambda_vec)


class VectorFieldNet(nn.Module):
    """The learned vector field for the flow: predicts dx/dt."""

    def __init__(self, context_dim: int = 128, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.input_dim = 3 + 1 + context_dim  # x(3) + t(1) + context
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3),
        )

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        """x: (..., 3), t: (..., 1) or (...,), context: (..., 128)."""
        if t.dim() == x.dim() - 1:
            t = t.unsqueeze(-1)
        inp = torch.cat([x, t, context], dim=-1)
        return self.net(inp)


class CFMEmulator(nn.Module):
    """Wraps EventEncoder + VectorFieldNet for conditional event generation."""

    def __init__(self, lambda_dim: int = 7, context_dim: int = 128, hidden_dim: int = 256):
        super().__init__()
        self.lambda_dim = lambda_dim
        self.encoder = EventEncoder(input_dim=lambda_dim, hidden_dim=context_dim, output_dim=context_dim)
        self.vector_field = VectorFieldNet(context_dim=context_dim, hidden_dim=hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        lambda_vec: torch.Tensor,
    ) -> torch.Tensor:
        """Predict velocity dx/dt. x: (..., 3), t: (...,), lambda_vec: (..., D_lambda)."""
        context = self.encoder(lambda_vec)
        if context.dim() == 2 and x.dim() == 3:
            # x: (B, N, 3), context: (B, 128) -> expand context to (B, N, 128)
            context = context.unsqueeze(1).expand(-1, x.shape[1], -1)
            t = t.unsqueeze(1).expand(-1, x.shape[1]) if t.dim() == 1 else t
        return self.vector_field(x, t, context)

    def sample(
        self,
        lambda_vec: torch.Tensor,
        n_samples: int,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Generate n_samples events. Uses ODE integration from t=0 to t=1."""
        if device is None:
            device = next(self.parameters()).device
        lambda_vec = lambda_vec.to(device)
        if lambda_vec.dim() == 1:
            lambda_vec = lambda_vec.unsqueeze(0)

        x0 = torch.randn(1, n_samples, 3, device=device, dtype=lambda_vec.dtype)

        def ode_fn(t, x):
            # t is scalar, x is (1, n_samples, 3)
            t_flat = torch.full((x.shape[0], x.shape[1]), t.item(), device=x.device, dtype=x.dtype)
            return self.forward(x, t_flat, lambda_vec)

        try:
            from torchdiffeq import odeint
            t_span = torch.tensor([0.0, 1.0], device=device)
            xt = odeint(ode_fn, x0, t_span, method="dopri5")
            x1 = xt[-1]
        except ImportError:
            # Fallback: Euler integration
            x = x0
            n_steps = 50
            for i in range(n_steps):
                t = torch.full((1, n_samples), (i + 0.5) / n_steps, device=device)
                v = self.forward(x, t, lambda_vec)
                x = x + v * (1.0 / n_steps)
            x1 = x

        return x1.squeeze(0)


def generate_catalog(
    lambda_vec: np.ndarray,
    n_events: int,
    model: CFMEmulator,
    normalizer: Dict,
) -> pd.DataFrame:
    """
    Given a hyperparameter vector, generate a synthetic merger catalog.

    Args:
        lambda_vec : np.array of shape (D_lambda,) — the Λ vector
        n_events   : int — how many events to generate
        model      : trained CFMEmulator
        normalizer : loaded from obs_normalizer.json
    Returns:
        catalog : pd.DataFrame with columns [mchirp, q, z]
                  values are in ORIGINAL (denormalized) units
    """
    model.eval()
    lam = torch.from_numpy(np.asarray(lambda_vec, dtype=np.float32))
    if lam.dim() == 1:
        lam = lam.unsqueeze(0)
    with torch.inference_mode():
        x = model.sample(lam, n_events)
    x_np = x.cpu().numpy()
    x_denorm = denormalize_obs(x_np, normalizer)
    # Clamp to physical bounds (model may extrapolate slightly)
    x_denorm[:, 1] = np.clip(x_denorm[:, 1], 0.0, 1.0)   # q in [0, 1]
    x_denorm[:, 2] = np.maximum(x_denorm[:, 2], 1e-6)    # z > 0
    x_denorm[:, 0] = np.maximum(x_denorm[:, 0], 1e-2)    # mchirp > 0
    return pd.DataFrame(x_denorm, columns=["mchirp", "q", "z"])
