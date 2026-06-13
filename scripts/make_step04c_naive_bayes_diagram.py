#!/usr/bin/env python3
"""
Generate the Step 04c Naive Bayes emulator pipeline diagram.

This writes:
  docs/step04c_naive_bayes_emulator_flow.png

The diagram is meant to match the repo implementation in:
  models/naive_bayes_emulator.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def _box(ax, xy, w, h, title, lines, *, fc="#ffffff", ec="#1f2937", tc="#111827"):
    x, y = xy
    rect = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=1.2,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(rect)
    ax.text(
        x + 0.02 * w,
        y + h - 0.18 * h,
        title,
        fontsize=12,
        fontweight="bold",
        color=tc,
        va="top",
    )
    ax.text(
        x + 0.02 * w,
        y + h - 0.34 * h,
        "\n".join(lines),
        fontsize=10,
        color=tc,
        va="top",
        family="DejaVu Sans",
        linespacing=1.2,
    )
    return rect


def _arrow(ax, p0, p1, *, color="#374151"):
    ax.add_patch(
        FancyArrowPatch(
            p0,
            p1,
            arrowstyle="-|>",
            mutation_scale=14,
            linewidth=1.2,
            color=color,
        )
    )


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    out_png = project_root / "docs" / "step04c_naive_bayes_emulator_flow.png"
    out_png.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(16, 9), dpi=200)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.5,
        0.97,
        "Step 04c: Naive Bayes Emulator (repo implementation)",
        ha="center",
        va="top",
        fontsize=18,
        fontweight="bold",
        color="#111827",
    )

    # Left: training
    ax.text(0.08, 0.90, "Fit / Training (no SGD)", fontsize=13, fontweight="bold", color="#111827")
    b_inputs = _box(
        ax,
        (0.05, 0.70),
        0.40,
        0.17,
        "Inputs from Step 02",
        [
            "hyperparam_table_encoded.csv → grid lambdas Λ_g (lambda_0…)",
            "all_events.parquet → events (mchirp, q, z) with grid_idx",
            "obs_normalizer.json → normalize / denormalize",
        ],
        fc="#eff6ff",
        ec="#2563eb",
    )
    b_norm = _box(
        ax,
        (0.05, 0.52),
        0.40,
        0.13,
        "Normalize observations",
        [
            "x = normalize(mchirp, q, z)   where x ∈ R³",
            "Pack events by grid_idx (vectorized)",
        ],
        fc="#eff6ff",
        ec="#2563eb",
    )
    b_stats = _box(
        ax,
        (0.05, 0.30),
        0.40,
        0.20,
        "Per-grid summary statistics",
        [
            "For each grid point g:",
            "μ_g = mean(x | grid=g)",
            "σ_g = std(x | grid=g), with floor σ_min",
            "Store: grid_lambdas, grid_mu, grid_sigma",
        ],
        fc="#eff6ff",
        ec="#2563eb",
    )
    b_tau = _box(
        ax,
        (0.05, 0.16),
        0.40,
        0.10,
        "Kernel bandwidth τ",
        ["τ = median pairwise ||Λ_i − Λ_j|| on the grid"],
        fc="#eff6ff",
        ec="#2563eb",
    )

    _arrow(ax, (0.25, 0.70), (0.25, 0.65))
    _arrow(ax, (0.25, 0.52), (0.25, 0.50))
    _arrow(ax, (0.25, 0.30), (0.25, 0.26))

    # Right: inference
    ax.text(0.56, 0.90, "Inference / Generation", fontsize=13, fontweight="bold", color="#111827")
    b_given = _box(
        ax,
        (0.55, 0.78),
        0.40,
        0.10,
        "Given new Λ",
        ["Goal: generate synthetic events (mchirp, q, z)"],
        fc="#fff7ed",
        ec="#f97316",
    )
    b_weights = _box(
        ax,
        (0.55, 0.62),
        0.40,
        0.14,
        "Kernel weights over grid",
        [
            "π_g(Λ) ∝ exp( −||Λ − Λ_g||² / (2τ²) )",
            "Normalize with softmax so Σ_g π_g = 1",
        ],
        fc="#fff7ed",
        ec="#f97316",
    )

    # Branches
    b_gauss = _box(
        ax,
        (0.55, 0.36),
        0.19,
        0.22,
        "mode = gaussian (default)",
        [
            "1) Sample grid index g ~ Categorical(π)",
            "2) Sample x ~ N(μ_g, diag(σ_g²))",
            "   x = μ_g + σ_g ⊙ ε,  ε ~ N(0, I)",
        ],
        fc="#fff7ed",
        ec="#f97316",
    )
    b_near = _box(
        ax,
        (0.76, 0.36),
        0.19,
        0.22,
        "mode = nearest",
        [
            "1) g* = argmin_g ||Λ − Λ_g||²",
            "2) Resample x from stored pool",
            "   (empirical bootstrap at grid g*)",
        ],
        fc="#fff7ed",
        ec="#f97316",
    )

    b_den = _box(
        ax,
        (0.55, 0.20),
        0.40,
        0.13,
        "Denormalize + constraints",
        [
            "(mchirp, q, z) = denormalize(x)",
            "Clip: q ∈ [0, 1],  z ≥ 1e-6,  mchirp ≥ 1e-2",
        ],
        fc="#f0fdf4",
        ec="#16a34a",
    )
    b_out = _box(
        ax,
        (0.55, 0.07),
        0.40,
        0.10,
        "Output catalog",
        ["Pandas DataFrame with columns: mchirp, q, z"],
        fc="#f0fdf4",
        ec="#16a34a",
    )

    _arrow(ax, (0.75, 0.78), (0.75, 0.76))
    _arrow(ax, (0.75, 0.62), (0.75, 0.58))
    # Split
    _arrow(ax, (0.75, 0.62), (0.64, 0.58))
    _arrow(ax, (0.75, 0.62), (0.86, 0.58))
    # Merge
    _arrow(ax, (0.64, 0.36), (0.70, 0.33))
    _arrow(ax, (0.86, 0.36), (0.80, 0.33))
    _arrow(ax, (0.75, 0.20), (0.75, 0.17))

    # Legend
    ax.text(
        0.5,
        0.01,
        "Legend:  Λ = encoded hyperparameters (lambda_0..);  g = grid row (~7.5k);  x = normalized (mchirp,q,z)",
        ha="center",
        va="bottom",
        fontsize=10,
        color="#374151",
    )

    fig.savefig(out_png, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()

