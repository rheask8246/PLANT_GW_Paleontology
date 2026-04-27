"""
K trained posteriors p_k(Λ | C): combine at inference (epistemic spread over 04+05 runs).
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import torch

from .posterior_network_lite import PosteriorNet, SSPC_THETA_PARAM_COLS

__all__ = [
    "SSPC_THETA_PARAM_COLS",
    "load_posterior_member",
    "log_mean_of_log_probs",
    "mixture_sample",
]


def load_posterior_member(
    checkpoint_dir: Path,
    model_name: str,
    device: torch.device,
) -> PosteriorNet:
    """
    Load one Step-5 checkpoint from a directory that contains
    `posterior_network_best.pt` and `posterior_network_config.json`.
    """
    import json
    import sys

    from .posterior_network_lite import LitePosteriorNet
    from .posterior_network_full import FullPosteriorNet

    d = Path(checkpoint_dir)
    cfg_path = d / "posterior_network_config.json"
    pt_path = d / "posterior_network_best.pt"
    if not cfg_path.is_file() or not pt_path.is_file():
        sys.exit(f"ensemble: missing {cfg_path} or {pt_path}")
    with open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)
    meta = cfg.get("norm_meta") or {}
    tm = torch.tensor(meta["theta_mean"], dtype=torch.float32, device=device)
    ts = torch.tensor(meta["theta_std"], dtype=torch.float32, device=device)
    if model_name == "lite":
        m: PosteriorNet = LitePosteriorNet()
    else:
        m = FullPosteriorNet()
    ck = torch.load(pt_path, map_location=device, weights_only=False)
    m.load_state_dict(ck["state_dict"], strict=True)
    m.set_theta_stats(tm, ts)
    m.to(device)
    m.eval()
    return m


@torch.no_grad()
def log_mean_of_log_probs(
    members: List[PosteriorNet],
    theta: torch.Tensor,
    events: torch.Tensor,
    event_mask: torch.Tensor,
) -> torch.Tensor:
    """
    (1/K) * sum_k log p_k(Λ|C) — log of the geometric mean of the member densities
    in log-space. `theta` and `events` same shapes as a single `PosteriorNet.log_prob`.
    -> (B,)
    """
    if not members:
        raise ValueError("ensemble: empty members list")
    acc = None
    for m in members:
        lp = m.log_prob(theta, events, event_mask)
        acc = lp if acc is None else acc + lp
    return acc / float(len(members))


@torch.no_grad()
def mixture_sample(
    members: List[PosteriorNet],
    events: torch.Tensor,
    event_mask: torch.Tensor,
    num_samples: int,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """
    For each member k, draw ~floor(n/K) i.i.d. samples; concatenate on the sample
    axis -> discrete mixture (equal *mixture* weight per member, not per sample).
    Shape: (B, num_samples, K_theta).
    """
    if not members:
        raise ValueError("ensemble: empty members list")
    k = len(members)
    n = int(num_samples)
    base = n // k
    rem = n - base * k
    chunks: List[torch.Tensor] = []
    for i, m in enumerate(members):
        n_i = base + (1 if i < rem else 0)
        if n_i <= 0:
            continue
        s = m.sample(
            events,
            event_mask,
            num_samples=n_i,
        )
        chunks.append(s)
    out = torch.cat(chunks, dim=1)
    if out.shape[1] != n:
        # Rounding: trim or pad (should not happen)
        if out.shape[1] > n:
            out = out[:, :n, :]
    return out
