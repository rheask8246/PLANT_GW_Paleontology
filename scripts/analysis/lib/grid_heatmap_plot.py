"""Shared (sfr_a, mu0) heatmap plotting helpers."""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import matplotlib.axes
import numpy as np

# Fiducial-rate comparison points (fixed nuisances, z = 0.2) for overlay on SMT/CE panels.
FIDUCIAL_STUDY_CHANNELS: frozenset[str] = frozenset({"SMT", "CE"})
FIDUCIAL_STUDY_MARKS: Tuple[Dict[str, Any], ...] = (
    {
        "label": "Fiducial",
        "mu0": 0.025,
        "sfra": 0.02,
        "color": "#ffffff",
    },
    {
        "label": r"Low $\mu_0$",
        "mu0": 0.007,
        "sfra": 0.02,
        "color": "#29b6f6",
    },
    {
        "label": r"High $\mu_0$",
        "mu0": 0.035,
        "sfra": 0.02,
        "color": "#e040fb",
    },
    {
        "label": r"Low $a_{\mathrm{SF}}$",
        "mu0": 0.025,
        "sfra": 0.01,
        "color": "#66bb6a",
    },
    {
        "label": r"High $a_{\mathrm{SF}}$",
        "mu0": 0.025,
        "sfra": 0.03,
        "color": "#00e5ff",
    },
)


def mask_grid_z_edges(z: np.ndarray) -> np.ndarray:
    """
    Mask the minimum-``sfr_a`` row and minimum-``mu0`` column (grid indices 0, 0).

    SSPC linspace grids place unphysical/near-zero parameter values on these edges;
    masking keeps heatmaps comparable to legacy ``masked_edges`` ablation plots.
    """
    out = np.asarray(z, dtype=np.float64).copy()
    if out.size == 0:
        return out
    if out.shape[0] > 0:
        out[0, :] = np.nan
    if out.shape[1] > 0:
        out[:, 0] = np.nan
    return out


def cell_edges(centers: np.ndarray) -> np.ndarray:
    """
    Bin edges for ``pcolormesh(..., shading='flat')`` from cell centers.
    """
    centers = np.asarray(centers, dtype=np.float64)
    n = len(centers)
    if n == 0:
        raise ValueError("empty axis")
    if n == 1:
        half = max(abs(float(centers[0])) * 0.5, 1e-6)
        return np.array([centers[0] - half, centers[0] + half], dtype=np.float64)
    edges = np.empty(n + 1, dtype=np.float64)
    edges[1:-1] = 0.5 * (centers[:-1] + centers[1:])
    edges[0] = centers[0] - 0.5 * (centers[1] - centers[0])
    edges[-1] = centers[-1] + 0.5 * (centers[-1] - centers[-2])
    return edges


def pcolormesh_sfra_mu0(
    ax: matplotlib.axes.Axes,
    mu0_centers: np.ndarray,
    sfra_centers: np.ndarray,
    z: np.ndarray,
    *,
    mu0_range: Optional[Tuple[float, float]] = None,
    sfra_range: Optional[Tuple[float, float]] = None,
    **kwargs: Any,
):
    """
    ``pcolormesh`` on (mu0, sfr_a) with the colored region spanning the full axis range.

    When ``mu0_range`` / ``sfra_range`` are given, bin edges are placed on a uniform
    partition of those endpoints so the heatmap fills the square exactly.
    """
    mu0_centers = np.asarray(mu0_centers, dtype=np.float64)
    sfra_centers = np.asarray(sfra_centers, dtype=np.float64)
    z_arr = np.asarray(z, dtype=np.float64)
    if z_arr.shape != (len(sfra_centers), len(mu0_centers)):
        raise ValueError(
            f"z shape {z_arr.shape} != ({len(sfra_centers)}, {len(mu0_centers)})"
        )

    if mu0_range is not None:
        mu0_e = np.linspace(float(mu0_range[0]), float(mu0_range[1]), len(mu0_centers) + 1)
    else:
        mu0_e = cell_edges(mu0_centers)
    if sfra_range is not None:
        sfra_e = np.linspace(float(sfra_range[0]), float(sfra_range[1]), len(sfra_centers) + 1)
    else:
        sfra_e = cell_edges(sfra_centers)

    mesh = ax.pcolormesh(mu0_e, sfra_e, z_arr, shading="flat", **kwargs)
    ax.set_xlim(float(mu0_e[0]), float(mu0_e[-1]))
    ax.set_ylim(float(sfra_e[0]), float(sfra_e[-1]))
    return mesh


def overlay_fiducial_study_marks(
    ax: matplotlib.axes.Axes,
    *,
    legend: bool = False,
    mu0_range: Optional[Tuple[float, float]] = None,
    sfra_range: Optional[Tuple[float, float]] = None,
) -> None:
    """
    Overlay five comparison-grid locations (fiducial, μ₀ low/high at a_SF=0.02,
    a_SF low/high at μ₀=0.025) as colored ``x`` markers. Axes: x = μ₀, y = a_SF.
    """
    handles = []
    for spec in FIDUCIAL_STUDY_MARKS:
        mu0 = float(spec["mu0"])
        sfra = float(spec["sfra"])
        if mu0_range is not None:
            lo, hi = float(mu0_range[0]), float(mu0_range[1])
            if mu0 < lo or mu0 > hi:
                continue
        if sfra_range is not None:
            lo, hi = float(sfra_range[0]), float(sfra_range[1])
            if sfra < lo or sfra > hi:
                continue
        (line,) = ax.plot(
            mu0,
            sfra,
            marker="x",
            markersize=9,
            markeredgewidth=2.2,
            color=str(spec["color"]),
            linestyle="none",
            zorder=10,
            label=str(spec["label"]),
        )
        handles.append(line)
    if legend and handles:
        ax.legend(
            handles=handles,
            loc="upper right",
            fontsize=8,
            framealpha=0.85,
            edgecolor="0.5",
        )
