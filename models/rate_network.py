"""
Rate network: predicts log10(sum_pdet) from hyperparameters (lambda_vec).

Architecture: 7 → 64 → 64 → 32 → 1 with LayerNorm and GELU.
Also provides GaussianProcessRate baseline for sparse grid sanity check.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn


def _xavier_init(module: nn.Module) -> None:
    """Xavier uniform for weights, zero for biases."""
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class RateNetwork(nn.Module):
    """
    Predicts log10(sum_pdet) from lambda_vec (7 dims).

    Input: lambda_vec = [channel_onehot (5), chi_b_norm (1), alpha_CE_norm (1)]
    Output: scalar, predicted log detectable rate
    """

    def __init__(self, input_dim: int = 7, hidden_dims: tuple = (64, 64, 32)):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims

        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.LayerNorm(h))
            layers.append(nn.GELU())
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

        self.apply(_xavier_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (..., 7) -> (..., 1)"""
        return self.net(x).squeeze(-1)

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class GaussianProcessRate:
    """
    sklearn GP baseline with Matern kernel.
    Use when neural net can't beat GP on sparse grid (~20 pts/channel).
    """

    def __init__(self):
        self._gp = None
        self._fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GaussianProcessRate":
        """X: (N, 7), y: (N,) log10(sum_pdet)"""
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import Matern

        self._gp = GaussianProcessRegressor(
            kernel=Matern(nu=2.5),
            alpha=1e-6,
            normalize_y=True,
            random_state=42,
        )
        self._gp.fit(X, y)
        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return mean prediction."""
        if not self._fitted:
            raise RuntimeError("GP not fitted")
        return self._gp.predict(X)
