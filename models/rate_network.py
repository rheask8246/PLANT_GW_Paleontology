"""
Rate network model for SSPC data.

Input:  5-dim feature vector [CE_ind, CHE_ind, SMT_ind, sfra_norm, mu0_norm]
Output: log10(Σ det_weight) — total observer-frame detection rate

The NormalizedNet wrapper inverts z-score normalisation so forward() always
returns values in log10 scale.  Checkpoint format matches 03_rate_network.py.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class RateNet(nn.Module):
    """MLP: input_dim → 64 → 32 → 1  with LayerNorm + GELU."""

    def __init__(self, input_dim: int = 5, hidden: tuple = (64, 32)):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.GELU()]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class NormalizedNet(nn.Module):
    """Wraps RateNet; forward() returns predictions in original log10 scale."""

    def __init__(self, base: RateNet, y_mean: float, y_std: float):
        super().__init__()
        self.base = base
        self.register_buffer("y_mean", torch.tensor(float(y_mean)))
        self.register_buffer("y_std",  torch.tensor(float(y_std)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) * self.y_std + self.y_mean

    @property
    def n_params(self) -> int:
        return self.base.n_params


# Keep old names as aliases for backwards compatibility with any imports
RateNetwork     = RateNet
NormalizedRateNet = NormalizedNet
