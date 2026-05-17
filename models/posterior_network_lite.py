"""
Amortized posterior p(Λ | catalog) for SSPC — **architecture** (set encoder + flow).

**Training data (05_posterior_network.py):** Catalogs are **synthetic draws** from
a **frozen** `CFMEmulator` or `DiffusionEmulator` at each row’s `lambda_*`, not
subsamples of 02’s parquet files (see PopFlow proposal Stage 2 → 4).

**Theta (9-d):** `SSPC_THETA_PARAM_COLS` / `03_rate_network` — channel fixed by row.

**Events (6-d per row):** z-scored observables + three σ-broadcasts from the same
`obs_normalizer` dict bundled in the 04/04b checkpoint.
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Same semantic labels as 03_rate_network.SSPC_PARAM_COLS (9 physical parameters).
SSPC_THETA_PARAM_COLS: Tuple[str, ...] = (
    "sspc_sfr_a_mean",
    "sspc_sfr_b_mean",
    "sspc_sfr_c_mean",
    "sspc_sfr_d_mean",
    "sspc_mu0_mean",
    "sspc_muz_mean",
    "sspc_sigma0_mean",
    "sspc_sigmaz_mean",
    "sspc_alpha_skew_mean",
)

THETA_DIM: int = len(SSPC_THETA_PARAM_COLS)


def _mha_key_padding_mask(event_mask: torch.Tensor) -> torch.Tensor:
    """event_mask: (B, L) 1=real, 0=pad -> (B, L) bool True=ignore for PyTorch MHA."""
    return event_mask == 0


class SetTransformerEncoder(nn.Module):
    """Pre-norm Transformer over events; no positional encoding (set structure)."""

    def __init__(
        self,
        input_dim: int = 6,
        hidden_dim: int = 128,
        ffn_dim: int = 512,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.in_proj = nn.Linear(input_dim, hidden_dim)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            enc_layer, num_layers=num_layers, enable_nested_tensor=False
        )

    def forward(
        self, events: torch.Tensor, event_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        events: (B, L, input_dim)
        event_mask: (B, L) 1=valid
        returns: (B, L, hidden_dim)
        """
        x = self.in_proj(events)
        key_padding_mask = _mha_key_padding_mask(event_mask)
        return self.encoder(x, src_key_padding_mask=key_padding_mask)


def _masked_mean_pool(
    h: torch.Tensor, event_mask: torch.Tensor
) -> torch.Tensor:
    """h: (B, L, D); event_mask: (B, L) 1=valid -> (B, D)"""
    w = event_mask.unsqueeze(-1).to(h.dtype)
    s = (h * w).sum(dim=1)
    c = event_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
    return s / c


class MLP2(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int,
        n_layers: int = 2,
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        d = in_dim
        for i in range(n_layers - 1):
            layers += [nn.Linear(d, hidden_dim), nn.GELU()]
            d = hidden_dim
        tail = nn.Linear(d, out_dim)
        nn.init.normal_(tail.weight, std=1e-3)
        nn.init.zeros_(tail.bias)
        layers.append(tail)
        self.net = nn.Sequential(*layers)
        for m in self.net.modules():
            if isinstance(m, nn.Linear) and m is not tail:
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ConditionalRealNVP(nn.Module):
    """
    Alternating affine coupling: maps data x (theta, z-scored) to base z ~ N(0,I)
    and back. `context` is the catalog embedding (B, context_dim).
    """

    def __init__(
        self,
        dim: int,
        context_dim: int,
        n_coupling: int = 4,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.n_coupling = n_coupling
        self.s_nets = nn.ModuleList()
        self.t_nets = nn.ModuleList()
        # Split: alternate which half is transformed
        for i in range(n_coupling):
            # half fixed, half to transform; alternate 4/5 for dim=9
            d_fix = (dim + i) // 2
            d_trans = dim - d_fix
            in_fix = d_fix
            in_st = in_fix + context_dim
            self.s_nets.append(MLP2(in_st, d_trans, hidden_dim, n_layers=3))
            self.t_nets.append(MLP2(in_st, d_trans, hidden_dim, n_layers=3))
        # Random permutation between layers
        perms: List[torch.Tensor] = []
        gen = torch.Generator()
        gen.manual_seed(12345)
        p = torch.arange(dim, dtype=torch.long)
        for _ in range(n_coupling - 1):
            p = p[torch.randperm(dim, generator=gen)]
            perms.append(p)
        self.register_buffer("perm_0", torch.arange(dim, dtype=torch.long))
        for i, p_ in enumerate(perms):
            self.register_buffer(f"perm_{i+1}", p_)

    def _permute(self, x: torch.Tensor, layer_idx: int) -> torch.Tensor:
        p = self.get_buffer(f"perm_{layer_idx}")
        return x[:, p]

    def _inv_permute(self, y: torch.Tensor, layer_idx: int) -> torch.Tensor:
        p = self.get_buffer(f"perm_{layer_idx}")
        out = torch.empty_like(y)
        out[:, p] = y
        return out

    def _coupling(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        layer: int,
        forward_to_base: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """forward_to_base True: x (data) -> y towards base; return log|dy/dx|."""
        dim = self.dim
        d_fix = (dim + layer) // 2
        x1, x2 = x[:, :d_fix], x[:, d_fix:]
        in_st = torch.cat([x1, context], dim=-1)
        s = self.s_nets[layer](in_st)
        t = self.t_nets[layer](in_st)
        s = torch.tanh(s) * 2.0  # bound scale
        if forward_to_base:
            # y1 = x1, y2 = (x2 - t) * exp(-s)  (inverse of generative y2 = x2*exp(s)+t)
            y1 = x1
            y2 = (x2 - t) * torch.exp(-s)
            log_det = -s.sum(dim=-1)
        else:
            y1 = x1
            y2 = x2 * torch.exp(s) + t
            log_det = s.sum(dim=-1)
        y = torch.cat([y1, y2], dim=-1)
        return y, log_det

    def forward_to_base(
        self, x: torch.Tensor, context: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: (B, dim) data (z-scored theta)
        -> z: (B, dim) base, log_det: (B,) log|dz/dx|
        """
        z = x
        total = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        for l in range(self.n_coupling):
            z = self._permute(z, l)
            z, logdet = self._coupling(z, context, l, forward_to_base=True)
            total = total + logdet
        return z, total

    def inverse_from_base(
        self, z: torch.Tensor, context: torch.Tensor
    ) -> torch.Tensor:
        """z ~ N(0,I) -> x in data space."""
        x = z
        for l in range(self.n_coupling - 1, -1, -1):
            x, _ = self._coupling(x, context, l, forward_to_base=False)
            x = self._inv_permute(x, l)
        return x


class PosteriorNet(nn.Module):
    """
    Set encoder + conditional RealNVP for p(theta | catalog).

    `theta` in `log_prob` / training must be z-scored with buffers `theta_mean`,
    `theta_std` (register_buffer) matching training data.
    """

    def __init__(
        self,
        event_input_dim: int = 6,
        theta_dim: int = THETA_DIM,
        hidden_dim: int = 128,
        ffn_dim: int = 512,
        num_encoder_layers: int = 2,
        num_heads: int = 4,
        n_coupling: int = 4,
        flow_hidden_dim: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.event_input_dim = event_input_dim
        self.theta_dim = theta_dim
        self.hidden_dim = hidden_dim
        self.encoder = SetTransformerEncoder(
            input_dim=event_input_dim,
            hidden_dim=hidden_dim,
            ffn_dim=ffn_dim,
            num_layers=num_encoder_layers,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.flow = ConditionalRealNVP(
            dim=theta_dim,
            context_dim=hidden_dim,
            n_coupling=n_coupling,
            hidden_dim=flow_hidden_dim,
        )
        self.register_buffer("theta_mean", torch.zeros(theta_dim))
        self.register_buffer("theta_std", torch.ones(theta_dim))
        for m in self.modules():
            if isinstance(m, nn.Linear) and not isinstance(m, MLP2):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def set_theta_stats(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.theta_mean.copy_(mean.to(self.theta_mean.device))
        self.theta_std.copy_(std.to(self.theta_std.device).clamp_min(1e-8))

    def _zscore(self, theta: torch.Tensor) -> torch.Tensor:
        return (theta - self.theta_mean) / self.theta_std

    def _zscore_inv(self, z: torch.Tensor) -> torch.Tensor:
        return z * self.theta_std + self.theta_mean

    def encode_catalog(
        self, events: torch.Tensor, event_mask: torch.Tensor
    ) -> torch.Tensor:
        h = self.encoder(events, event_mask)
        return _masked_mean_pool(h, event_mask)

    def log_prob(
        self,
        theta: torch.Tensor,
        events: torch.Tensor,
        event_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        theta: (B, K) in **physical** units (same as CSV)
        events: (B, L, 6)
        event_mask: (B, L)
        -> (B,)
        """
        c = self.encode_catalog(events, event_mask)
        x = self._zscore(theta)
        z, logdet = self.flow.forward_to_base(x, c)
        log_pz = -0.5 * (z**2 + math.log(2 * math.pi)).sum(dim=-1)
        theta_scale_log = -self.theta_std.log().sum()  # Jacobian of z-scoring
        return log_pz + logdet + theta_scale_log

    @torch.no_grad()
    def sample(
        self,
        events: torch.Tensor,
        event_mask: torch.Tensor,
        num_samples: int = 1000,
    ) -> torch.Tensor:
        """
        events: (B, L, 6), event_mask: (B, L)
        -> (B, num_samples, K) physical units
        """
        self.eval()
        c = self.encode_catalog(events, event_mask)  # (B, H)
        B = c.shape[0]
        K = self.theta_dim
        H = c.shape[1]
        c_rep = c.unsqueeze(1).expand(-1, num_samples, -1).reshape(
            B * num_samples, H
        )
        z = torch.randn(
            B * num_samples, K, device=events.device, dtype=events.dtype
        )
        x = self.flow.inverse_from_base(z, c_rep)
        x = x.view(B, num_samples, K)
        return self._zscore_inv(x)


class LitePosteriorNet(PosteriorNet):
    """Small defaults for CPU smoke tests and debugging."""

    def __init__(self) -> None:
        super().__init__(
            event_input_dim=6,
            theta_dim=THETA_DIM,
            hidden_dim=128,
            ffn_dim=512,
            num_encoder_layers=2,
            num_heads=4,
            n_coupling=4,
            flow_hidden_dim=64,
            dropout=0.1,
        )


# Backwards-friendly aliases
LitePosteriorNetwork = LitePosteriorNet
PosteriorNetworkLite = LitePosteriorNet
