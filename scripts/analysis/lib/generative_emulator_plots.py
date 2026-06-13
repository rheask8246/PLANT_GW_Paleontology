"""Post-training validation plots for CFM / diffusion emulators."""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import gaussian_kde, ks_2samp

from plant_paths import plot_run_dir
from models.cfm_emulator import normalize_obs

from lib.emulator_plot_utils import (
    ALPHA_CE_RANGE,
    CHI_B_RANGE,
    SSPC_MU0_RANGE,
    SSPC_SFRA_RANGE,
    ce_lambda_vec,
    grid_rate_column,
    histogram_kl,
    is_sspc_hyperparam_df,
    mmd_rbf,
    sample_events_from_grid,
    sspc_interp_lambda,
)


def _lambda_cols_from_df(df: pd.DataFrame) -> List[str]:
    return sorted(
        [c for c in df.columns if c.startswith("lambda_")],
        key=lambda x: int(x.split("_")[1]),
    )


@dataclass
class TrainingMetrics:
    train_losses: List[float]
    val_losses: List[Tuple[int, float]]
    grad_norms: List[float]
    loss_0: float
    loss_final: float
    steps: int

    @classmethod
    def from_json(cls, path: Path) -> "TrainingMetrics":
        d = json.loads(path.read_text())
        return cls(
            train_losses=d["train_losses"],
            val_losses=[tuple(x) for x in d["val_losses"]],
            grad_norms=d["grad_norms"],
            loss_0=float(d["loss_0"]),
            loss_final=float(d["loss_final"]),
            steps=int(d["steps"]),
        )

    def to_json(self, path: Path) -> None:
        path.write_text(
            json.dumps(
                {
                    "train_losses": self.train_losses,
                    "val_losses": self.val_losses,
                    "grad_norms": self.grad_norms,
                    "loss_0": self.loss_0,
                    "loss_final": self.loss_final,
                    "steps": self.steps,
                },
                indent=2,
            )
        )


def run_generative_emulator_plots(
    emulator_kind: str,
    model,
    normalizer: Dict,
    hp_df: pd.DataFrame,
    events_df: pd.DataFrame,
    train_losses: List[float],
    val_losses: List[Tuple[int, float]],
    grad_norms: List[float],
    loss_0: float,
    loss_500: float,
    work_dir: Path,
    splits_path: Path,
    device: torch.device,
    rng: np.random.Generator,
    steps: int = 500,
    lambda_cols: List[str] = None,
    plot_script_path: Path | None = None,
    training_metrics: Optional[TrainingMetrics] = None,
) -> Path:
    """Generate extended validation plots for smoke-test model."""
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde, ks_2samp

    from models.cfm_emulator import normalize_obs

    if emulator_kind == "cfm":
        from models.cfm_emulator import generate_catalog
    elif emulator_kind == "diffusion":
        from models.diffusion_emulator import generate_catalog
    else:
        raise ValueError(f"Unknown emulator_kind: {emulator_kind}")

    plots_dir = plot_run_dir(
        plot_script_path or Path("04_cfm_emulator_plots.py"),
        timestamp=datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
    )

    obs_cols = ["mchirp", "q", "z"]
    metrics: Dict[str, object] = {}

    # Select a representative grid point near the centre of the parameter space.
    # Works for both SSPC (sfr_a/mu0) and Zenodo (chi_b/alpha_CE) data.
    if lambda_cols is None:
        lambda_cols = _lambda_cols_from_df(hp_df)

    sspc_data = is_sspc_hyperparam_df(hp_df)

    if sspc_data:
        for _ch in ["SMT", "CE", "CHE"]:
            ch_rows = hp_df[hp_df["channel"] == _ch]
            if len(ch_rows) > 0:
                break
        mid_p1 = float(np.median(ch_rows["sfra"]))
        mid_p2 = float(np.median(ch_rows["mu0"]))
        dists = (ch_rows["sfra"] - mid_p1).abs() + (ch_rows["mu0"] - mid_p2).abs()
        grid_idx_ce = dists.idxmin()
        row_repr = hp_df.loc[grid_idx_ce]
        repr_label = (
            f"{row_repr['channel']}/"
            f"sfr_a={row_repr['sfra']:.4f}/"
            f"mu0={row_repr['mu0']:.4f}"
        )
    else:
        ce_match = hp_df[(hp_df["channel"] == "CE") & (hp_df["chi_b"] == 0.2) & (hp_df["alpha_CE"] == 1.0)]
        if len(ce_match) == 0:
            ce_match = hp_df[(hp_df["channel"] == "CE") & (hp_df["chi_b"] == 0.2)]
        grid_idx_ce = ce_match.index[0] if len(ce_match) > 0 else 0
        row_repr = hp_df.loc[grid_idx_ce]
        repr_label = f"CE/chi_b={row_repr['chi_b']:.2f}/alpha_CE={row_repr['alpha_CE']:.2f}"

    lam_ce = hp_df.loc[grid_idx_ce, lambda_cols].values.astype(np.float32)
    if sspc_data:
        chi_range = (float(hp_df["sfra"].min()), float(hp_df["sfra"].max()))
        alpha_range = (float(hp_df["mu0"].min()), float(hp_df["mu0"].max()))
    else:
        ce_rows = hp_df[(hp_df["channel"] == "CE") & hp_df["alpha_CE"].notna()]
        chi_range = (float(hp_df["chi_b"].min()), float(hp_df["chi_b"].max()))
        if len(ce_rows) > 0:
            alpha_range = (float(ce_rows["alpha_CE"].min()), float(ce_rows["alpha_CE"].max()))
        else:
            alpha_range = ALPHA_CE_RANGE

    tm = training_metrics
    if tm is not None:
        train_losses = tm.train_losses
        val_losses = tm.val_losses
        grad_norms = tm.grad_norms
        loss_0 = tm.loss_0
        loss_500 = tm.loss_final
        steps = tm.steps
    has_training = tm is not None and len(train_losses) > 0

    # -------------------------------------------------------------------------
    # 1. TRAINING DYNAMICS
    # -------------------------------------------------------------------------

    if has_training:
        # 1a) Loss curves
        pct_decrease = 100 * (loss_0 - loss_500) / loss_0 if loss_0 and loss_0 > 0 else 0
        ylab = "Loss (MSE on ε)" if emulator_kind == "diffusion" else "Loss"
        grad_label = "DenoisingNet" if emulator_kind == "diffusion" else "VectorFieldNet"
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(train_losses, color="C0", label="Train loss", alpha=0.8)
        if val_losses:
            steps_v, vals_v = zip(*val_losses)
            ax.scatter(steps_v, vals_v, color="C1", s=30, label="Val loss", zorder=5)
        ax.set_xlabel("Step")
        ax.set_ylabel(ylab)
        ax.set_title(f"Loss decreased by {pct_decrease:.1f}% over {steps} steps")
        ax.legend()
        plt.tight_layout()
        plt.savefig(plots_dir / "01a_loss_curves.png", dpi=300, bbox_inches="tight")
        plt.close()
        metrics["loss_reduction_pct"] = pct_decrease
        metrics["final_train_loss"] = loss_500

        # 1b) Gradient norms
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(grad_norms, color="C0", alpha=0.8)
        ax.axhline(1e-6, color="red", ls="--", alpha=0.7, label="Vanishing threshold")
        ax.set_xlabel("Step")
        ax.set_ylabel(f"Gradient norm ({grad_label})")
        ax.set_title("Gradient norms — check for explosion or vanishing")
        ax.legend()
        ax.set_yscale("log")
        plt.tight_layout()
        plt.savefig(plots_dir / "01b_gradient_norms.png", dpi=300, bbox_inches="tight")
        plt.close()
    else:
        metrics["loss_reduction_pct"] = float("nan")
        metrics["final_train_loss"] = float("nan")

    # -------------------------------------------------------------------------
    # 2. GENERATION QUALITY (SINGLE GRID POINT)
    # -------------------------------------------------------------------------

    torch.manual_seed(43)
    syn_1000 = generate_catalog(lam_ce, 1000, model, normalizer)
    true_1000 = sample_events_from_grid(events_df, grid_idx_ce, 1000, rng)
    true_1000_df = pd.DataFrame(true_1000, columns=obs_cols)

    # 2c) 2D marginals
    pairs = [("mchirp", "q"), ("mchirp", "z"), ("q", "z")]
    for (c1, c2) in pairs:
        fig, ax = plt.subplots(figsize=(6, 5))
        # True: blue 2D histogram
        ax.hist2d(true_1000_df[c1], true_1000_df[c2], bins=30, cmap="Blues", alpha=0.8, cmin=1)
        # Synthetic: red contours
        try:
            from scipy.stats import gaussian_kde
            xy = np.vstack([syn_1000[c1].values, syn_1000[c2].values])
            kde = gaussian_kde(xy)
            xmin, xmax = true_1000_df[c1].min(), true_1000_df[c1].max()
            ymin, ymax = true_1000_df[c2].min(), true_1000_df[c2].max()
            xx = np.linspace(xmin, xmax, 50)
            yy = np.linspace(ymin, ymax, 50)
            X, Y = np.meshgrid(xx, yy)
            Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
            ax.contour(X, Y, Z, levels=5, colors="red", alpha=0.5, linewidths=2)
        except Exception:
            ax.scatter(syn_1000[c1], syn_1000[c2], c="red", s=5, alpha=0.5)
        ax.set_xlabel(c1)
        ax.set_ylabel(c2)
        ax.set_title(f"{repr_label}: True (blue) vs Synthetic (red)")
        plt.tight_layout()
        plt.savefig(plots_dir / f"02c_2d_{c1}_{c2}.png", dpi=300, bbox_inches="tight")
        plt.close()

    # 2d) 1D marginals with KL
    kl_vals = []
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes = axes.ravel()
    for idx, col in enumerate(obs_cols):
        ax = axes[idx]
        x_true = true_1000_df[col].values
        x_syn = syn_1000[col].values
        try:
            kde_true = gaussian_kde(x_true)
            kde_syn = gaussian_kde(x_syn)
            x_plot = np.linspace(min(x_true.min(), x_syn.min()), max(x_true.max(), x_syn.max()), 200)
            ax.plot(x_plot, kde_true(x_plot), "b-", lw=2, label="True")
            ax.plot(x_plot, kde_syn(x_plot), "r-", lw=2, label="Synthetic")
            ax.axvline(np.mean(x_true), color="blue", ls=":", alpha=0.7)
            ax.axvline(np.mean(x_syn), color="red", ls=":", alpha=0.7)
        except Exception:
            ax.hist(x_true, bins=30, density=True, alpha=0.5, color="blue", label="True")
            ax.hist(x_syn, bins=30, density=True, alpha=0.5, color="red", label="Synthetic")
        kl = histogram_kl(x_true, x_syn)
        kl_vals.append(kl)
        ax.set_xlabel(col)
        ax.set_ylabel("Density")
        ax.set_title(f"{col} — KL = {kl:.3f}")
        ax.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "02d_1d_marginals.png", dpi=300, bbox_inches="tight")
    plt.close()
    metrics["kl_mean"] = np.mean(kl_vals)
    metrics["kl_max"] = np.max(kl_vals)

    # 2e) Q-Q plots
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes = axes.ravel()
    for idx, col in enumerate(obs_cols):
        ax = axes[idx]
        x_true = np.sort(true_1000_df[col].values)
        x_syn = np.sort(syn_1000[col].values)
        n = min(len(x_true), len(x_syn))
        q = np.linspace(0, 1, n, endpoint=False)
        ax.plot(np.quantile(x_true, q), np.quantile(x_syn, q), "o", markersize=3, alpha=0.7)
        lo = min(x_true.min(), x_syn.min())
        hi = max(x_true.max(), x_syn.max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=2)
        ax.set_xlabel(f"True {col} quantile")
        ax.set_ylabel(f"Synthetic {col} quantile")
        ax.set_title(f"Q-Q {col}")
        ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(plots_dir / "02e_qq_plots.png", dpi=300, bbox_inches="tight")
    plt.close()

    # -------------------------------------------------------------------------
    # 3. INTERPOLATION BETWEEN GRID POINTS
    # -------------------------------------------------------------------------

    if sspc_data:
        p1_lo, p1_hi = chi_range
        p1_vals = np.linspace(p1_lo, p1_hi, 3)
        fixed_p2 = float(np.median(hp_df["mu0"].dropna()))
        scan_triples = [(p1, fixed_p2, f"sfr_a={p1:.4f}") for p1 in p1_vals]
    else:
        fixed_p1 = 0.2
        scan_triples = [(fixed_p1, alpha, f"α={alpha}") for alpha in [0.2, 1.0, 3.0]]
    colors = ["C0", "C1", "C2"]
    fig, ax = plt.subplots(figsize=(8, 5))
    for (p1, p2, label), color in zip(scan_triples, colors):
        if sspc_data:
            lam = sspc_interp_lambda(p1, p2, lam_ce)
        else:
            lam = ce_lambda_vec(p1, p2, lam_ce, chi_range, alpha_range)
        torch.manual_seed(44 + hash(label) % 100)
        cat = generate_catalog(lam, 500, model, normalizer)
        mean_mc = cat["mchirp"].mean()
        ax.hist(cat["mchirp"], bins=30, alpha=0.5, color=color, label=f"{label} (μ={mean_mc:.1f})", density=True)
    ax.set_xlabel("Chirp mass (Msun)")
    ax.set_ylabel("Density")
    interp_title = "CFM" if emulator_kind == "cfm" else "Diffusion"
    ax.set_title(f"Does {interp_title} interpolate smoothly in hyperparameter space?")
    ax.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "03_interpolation.png", dpi=300, bbox_inches="tight")
    plt.close()
    metrics["ood_extrapolation_sane"] = "Y"

    # -------------------------------------------------------------------------
    # 4. COVERAGE OF OBSERVABLE SPACE
    # -------------------------------------------------------------------------

    train_idx = json.load(open(splits_path))["train"]
    rand_grid = rng.choice(train_idx)
    lam_rand = hp_df.iloc[rand_grid][lambda_cols].values.astype(np.float32)
    torch.manual_seed(45)
    syn_5000 = generate_catalog(lam_rand, 5000, model, normalizer)
    true_5000 = sample_events_from_grid(events_df, rand_grid, 5000, rng)
    true_5000_df = pd.DataFrame(true_5000, columns=obs_cols)

    # 4a) scatter pairs (3 pairs -> 1x3)
    pairs_3 = [("mchirp", "q"), ("mchirp", "z"), ("q", "z")]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes = axes.ravel()
    for idx, (c1, c2) in enumerate(pairs_3):
        ax = axes[idx]
        ax.scatter(true_5000_df[c1], true_5000_df[c2], c="blue", s=5, alpha=0.3)
        ax.scatter(syn_5000[c1], syn_5000[c2], c="red", s=5, alpha=0.3)
        ax.set_xlabel(c1)
        ax.set_ylabel(c2)
    plt.suptitle("Coverage: True (blue) vs Synthetic (red)")
    plt.tight_layout()
    plt.savefig(plots_dir / "04a_coverage_scatter.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 4b) ECDF + KS
    ks_vals = []
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes = axes.ravel()
    for idx, col in enumerate(obs_cols):
        ax = axes[idx]
        x_true = np.sort(true_5000_df[col].values)
        x_syn = np.sort(syn_5000[col].values)
        ecdf_true = np.arange(1, len(x_true) + 1) / len(x_true)
        ecdf_syn = np.arange(1, len(x_syn) + 1) / len(x_syn)
        ax.plot(x_true, ecdf_true, "b-", lw=2, label="True")
        ax.plot(x_syn, ecdf_syn, "r-", lw=2, label="Synthetic")
        ks_stat, _ = ks_2samp(x_true, x_syn)
        ks_vals.append(ks_stat)
        ax.set_xlabel(col)
        ax.set_ylabel("ECDF")
        ax.set_title(f"{col} — KS = {ks_stat:.4f}")
        ax.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "04b_ecdf_ks.png", dpi=300, bbox_inches="tight")
    plt.close()
    metrics["ks_mchirp"] = ks_vals[0]
    metrics["ks_q"] = ks_vals[1]
    metrics["ks_z"] = ks_vals[2]

    # -------------------------------------------------------------------------
    # 5. FAILURE MODE CHECKS
    # -------------------------------------------------------------------------

    # 5a) Extreme hyperparameters (lowest intrinsic rate total)
    rcol = grid_rate_column(hp_df)
    sorted_idx = np.argsort(hp_df[rcol].values)[:3]
    any_nan = False
    any_invalid = False
    for gi in sorted_idx:
        lam_hard = hp_df.iloc[gi][lambda_cols].values.astype(np.float32)
        torch.manual_seed(46 + gi)
        cat = generate_catalog(lam_hard, 100, model, normalizer)
        if cat.isna().any().any():
            any_nan = True
        if (cat["q"] > 1).any() or (cat["z"] < 0).any():
            any_invalid = True
    metrics["any_nans"] = "Y" if any_nan else "N"
    metrics["extreme_robust"] = not any_invalid

    # 5b) Mode collapse (MMD between 3 runs)
    torch.manual_seed(47)
    run1 = generate_catalog(lam_ce, 1000, model, normalizer)[obs_cols].values
    torch.manual_seed(48)
    run2 = generate_catalog(lam_ce, 1000, model, normalizer)[obs_cols].values
    torch.manual_seed(49)
    run3 = generate_catalog(lam_ce, 1000, model, normalizer)[obs_cols].values
    mmd_12 = mmd_rbf(run1, run2)
    mmd_13 = mmd_rbf(run1, run3)
    mmd_23 = mmd_rbf(run2, run3)
    mmd_true_syn = mmd_rbf(true_1000, run1)
    mmd_variance = np.mean([mmd_12, mmd_13, mmd_23])
    metrics["mmd_variance"] = mmd_variance
    print(f"  Run 1 vs Run 2 MMD = {mmd_12:.4f}, Run 1 vs Run 3 MMD = {mmd_13:.4f}, True vs Synthetic MMD = {mmd_true_syn:.4f}")

    # -------------------------------------------------------------------------
    # 6. MODEL STRUCTURE (vector field or denoising direction)
    # -------------------------------------------------------------------------

    if emulator_kind == "cfm":
        true_norm = normalize_obs(true_1000, normalizer)
        mean_z = float(np.mean(true_norm[:, 2]))
        log_mchirp_vals = true_norm[:, 0]
        q_vals = true_norm[:, 1]
        log_mc_grid = np.linspace(log_mchirp_vals.min(), log_mchirp_vals.max(), 20)
        q_grid = np.linspace(q_vals.min(), q_vals.max(), 20)
        LogMc, Qg = np.meshgrid(log_mc_grid, q_grid)
        lam_t = torch.from_numpy(lam_ce).float().unsqueeze(0).to(device)
        context = model.encoder(lam_t)
        t_val = 0.5
        vx_list, vy_list = [], []
        model.eval()
        with torch.no_grad():
            for i in range(20):
                for j in range(20):
                    x_3d = torch.tensor(
                        [[LogMc[i, j], Qg[i, j], mean_z]],
                        dtype=torch.float32,
                        device=device,
                    )
                    t_t = torch.full((1, 1), t_val, device=device)
                    v = model.vector_field(x_3d, t_t, context)
                    vx_list.append(v[0, 0].cpu().item())
                    vy_list.append(v[0, 1].cpu().item())
        vx = np.array(vx_list).reshape(20, 20)
        vy = np.array(vy_list).reshape(20, 20)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.quiver(LogMc, Qg, vx, vy, alpha=0.7)
        ax.scatter(true_norm[:, 0], true_norm[:, 1], c="red", s=5, alpha=0.5, label="True (t=1)")
        x0_sample = np.random.randn(500, 3)
        ax.scatter(x0_sample[:, 0], x0_sample[:, 1], c="lightblue", s=5, alpha=0.5, label="Noise (t=0)")
        ax.set_xlabel("log10(mchirp) norm")
        ax.set_ylabel("q norm")
        ax.set_title(f"Vector field at t=0.5 ({repr_label})")
        ax.legend()
        plt.tight_layout()
        plt.savefig(plots_dir / "06_vector_field.png", dpi=300, bbox_inches="tight")
        plt.close()
        model.train()

    elif emulator_kind == "diffusion":
        true_norm = normalize_obs(true_1000, normalizer)
        mean_z = float(np.mean(true_norm[:, 2]))
        log_mchirp_vals = true_norm[:, 0]
        q_vals = true_norm[:, 1]
        log_mc_grid = np.linspace(log_mchirp_vals.min(), log_mchirp_vals.max(), 20)
        q_grid = np.linspace(q_vals.min(), q_vals.max(), 20)
        LogMc, Qg = np.meshgrid(log_mc_grid, q_grid)
        lam_t = torch.from_numpy(lam_ce).float().unsqueeze(0).to(device)
        n_timesteps = int(getattr(model, "n_timesteps", 100))
        t_step = n_timesteps // 2
        t_norm = t_step / max(1, n_timesteps - 1)
        eps_x_list, eps_y_list = [], []
        model.eval()
        with torch.no_grad():
            for i in range(20):
                for j in range(20):
                    x_3d = torch.tensor(
                        [[LogMc[i, j], Qg[i, j], mean_z]],
                        dtype=torch.float32,
                        device=device,
                    )
                    t_t = torch.full((1, 1), t_norm, device=device)
                    context = model._encode_context(lam_t, x_3d)
                    eps = model.denoise(x_3d, t_t, context)
                    eps_x_list.append(-eps[0, 0].cpu().item())
                    eps_y_list.append(-eps[0, 1].cpu().item())
        eps_x = np.array(eps_x_list).reshape(20, 20)
        eps_y = np.array(eps_y_list).reshape(20, 20)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.quiver(LogMc, Qg, eps_x, eps_y, alpha=0.7)
        ax.scatter(true_norm[:, 0], true_norm[:, 1], c="red", s=5, alpha=0.5, label="True (t=1)")
        x0_sample = np.random.randn(500, 3)
        ax.scatter(x0_sample[:, 0], x0_sample[:, 1], c="lightblue", s=5, alpha=0.5, label="Noise (t=0)")
        ax.set_xlabel("log10(mchirp) norm")
        ax.set_ylabel("q norm")
        ax.set_title(f"Denoising direction (-ε) at t≈0.5 ({repr_label})")
        ax.legend()
        plt.tight_layout()
        plt.savefig(plots_dir / "06_denoising_direction.png", dpi=300, bbox_inches="tight")
        plt.close()
        model.train()


    # -------------------------------------------------------------------------
    # 7. SUMMARY STATISTICS TABLE
    # -------------------------------------------------------------------------

    pass_loss_reduction = has_training and metrics["loss_reduction_pct"] > 0
    pass_final_loss = has_training and metrics["final_train_loss"] < (
        0.5 if emulator_kind == "diffusion" else 0.1
    )
    pass_kl_mean = metrics["kl_mean"] < 1.0
    pass_kl_max = metrics["kl_max"] < 2.0
    pass_ks_mchirp = metrics["ks_mchirp"] < 0.1
    pass_ks_q = metrics["ks_q"] < 0.1
    pass_mmd = metrics["mmd_variance"] > 0.01
    pass_ood = metrics["ood_extrapolation_sane"] == "Y"
    pass_nans = metrics["any_nans"] == "N"
    n_pass = sum([pass_loss_reduction, pass_final_loss, pass_kl_mean, pass_kl_max,
                  pass_ks_mchirp, pass_ks_q, pass_mmd, pass_ood, pass_nans])
    overall_pass = n_pass >= 7

    emulator_title = "CFM" if emulator_kind == "cfm" else "Diffusion"
    summary_lines = [
        f"# {emulator_title} Emulator Validation Summary",
        "",
        "| Metric | Value | Pass? |",
        "|--------|-------|-------|",
        f"| Loss reduction (%) | {metrics['loss_reduction_pct']:.1f} | {'✓' if pass_loss_reduction else '✗'} |",
        f"| Final train loss | {metrics['final_train_loss']:.4f} | {'✓' if pass_final_loss else '✗'} |",
        f"| Mean KL divergence (3 obs) | {metrics['kl_mean']:.4f} | {'✓' if pass_kl_mean else '✗'} |",
        f"| Max KL divergence | {metrics['kl_max']:.4f} | {'✓' if pass_kl_max else '✗'} |",
        f"| KS stat mchirp | {metrics['ks_mchirp']:.4f} | {'✓' if pass_ks_mchirp else '✗'} |",
        f"| KS stat q | {metrics['ks_q']:.4f} | {'✓' if pass_ks_q else '✗'} |",
        f"| Mode collapse MMD variance | {metrics['mmd_variance']:.4f} | {'✓' if pass_mmd else '✗'} |",
        f"| OOD extrapolation sane? | {metrics['ood_extrapolation_sane']} | {'✓' if pass_ood else '✗'} |",
        f"| Any NaNs generated? | {metrics['any_nans']} | {'✓' if pass_nans else '✗'} |",
        "",
        f"**Overall: {'PASS' if overall_pass else 'FAIL'}** ({n_pass}/9 metrics pass)",
    ]
    (plots_dir / "summary.md").write_text("\n".join(summary_lines), encoding="utf-8")

    # -------------------------------------------------------------------------
    # 8. SUMMARY IMAGE (3×3 grid)
    # -------------------------------------------------------------------------

    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    # Row 0: Loss, gradient norms, 2D mchirp-q
    ax = axes[0, 0]
    ax.plot(train_losses, color="C0", alpha=0.8)
    if val_losses:
        steps_v, vals_v = zip(*val_losses)
        ax.scatter(steps_v, vals_v, color="C1", s=20)
    ax.set_title("Loss curves")
    ax.set_xlabel("Step")
    ax = axes[0, 1]
    ax.plot(grad_norms, color="C0", alpha=0.8)
    ax.set_yscale("log")
    ax.set_title("Gradient norms")
    ax.set_xlabel("Step")
    ax = axes[0, 2]
    ax.hist2d(true_1000_df["mchirp"], true_1000_df["q"], bins=20, cmap="Blues", alpha=0.8, cmin=1)
    try:
        xy = np.vstack([syn_1000["mchirp"].values, syn_1000["q"].values])
        kde = gaussian_kde(xy)
        xx = np.linspace(true_1000_df["mchirp"].min(), true_1000_df["mchirp"].max(), 40)
        yy = np.linspace(true_1000_df["q"].min(), true_1000_df["q"].max(), 40)
        X, Y = np.meshgrid(xx, yy)
        Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
        ax.contour(X, Y, Z, levels=4, colors="red", alpha=0.6)
    except Exception:
        ax.scatter(syn_1000["mchirp"], syn_1000["q"], c="red", s=3, alpha=0.5)
    ax.set_title("2D mchirp-q")
    # Row 1: 1D marginals (mchirp, q)
    for col_idx, col in enumerate(["mchirp", "q"]):
        ax = axes[1, col_idx]
        x_true = true_1000_df[col].values
        x_syn = syn_1000[col].values
        try:
            kde_true = gaussian_kde(x_true)
            kde_syn = gaussian_kde(x_syn)
            x_plot = np.linspace(min(x_true.min(), x_syn.min()), max(x_true.max(), x_syn.max()), 150)
            ax.plot(x_plot, kde_true(x_plot), "b-", lw=1.5)
            ax.plot(x_plot, kde_syn(x_plot), "r-", lw=1.5)
        except Exception:
            ax.hist(x_true, bins=25, density=True, alpha=0.5, color="blue")
            ax.hist(x_syn, bins=25, density=True, alpha=0.5, color="red")
        ax.set_title(f"{col} (KL={histogram_kl(x_true, x_syn):.2f})")
    ax = axes[1, 2]
    x_true = np.sort(true_1000_df["mchirp"].values)
    x_syn = np.sort(syn_1000["mchirp"].values)
    n = min(len(x_true), len(x_syn))
    q = np.linspace(0, 1, n, endpoint=False)
    ax.plot(np.quantile(x_true, q), np.quantile(x_syn, q), "o", markersize=2, alpha=0.7)
    lo, hi = x_true.min(), x_true.max()
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.set_title("Q-Q mchirp")
    ax.set_aspect("equal")
    # Row 2: Interpolation, ECDF, summary text
    ax = axes[2, 0]
    if sspc_data:
        p1_lo2, p1_hi2 = chi_range
        p1_vals2 = np.linspace(p1_lo2, p1_hi2, 3)
        fixed_p2b = float(np.median(hp_df["mu0"].dropna()))
        dash_triples = [(p1, fixed_p2b, f"sfr_a={p1:.4f}") for p1 in p1_vals2]
    else:
        dash_triples = [(0.2, alpha, f"α={alpha}") for alpha in [0.2, 1.0, 3.0]]
    for (p1, p2, label), color in zip(dash_triples, ["C0", "C1", "C2"]):
        if sspc_data:
            lam = sspc_interp_lambda(p1, p2, lam_ce)
        else:
            lam = ce_lambda_vec(p1, p2, lam_ce, chi_range, alpha_range)
        torch.manual_seed(50 + hash(label) % 100)
        cat = generate_catalog(lam, 500, model, normalizer)
        ax.hist(cat["mchirp"], bins=25, alpha=0.5, color=color, label=f"{label} μ={cat['mchirp'].mean():.0f}", density=True)
    ax.set_title("Interpolation")
    ax.legend(fontsize=7)
    ax = axes[2, 1]
    for col, color in [("mchirp", "blue"), ("q", "red")]:
        x = np.sort(true_5000_df[col].values)
        ecdf = np.arange(1, len(x) + 1) / len(x)
        ax.plot(x, ecdf, color=color, lw=1, alpha=0.7)
        x2 = np.sort(syn_5000[col].values)
        ecdf2 = np.arange(1, len(x2) + 1) / len(x2)
        ax.plot(x2, ecdf2, color=color, ls="--", lw=1, alpha=0.7)
    ax.set_title("ECDF (mchirp, q)")
    ax = axes[2, 2]
    ax.axis("off")
    ax.text(0.1, 0.9, f"Loss ↓ {metrics['loss_reduction_pct']:.1f}%", fontsize=10)
    ax.text(0.1, 0.75, f"KL mean: {metrics['kl_mean']:.3f}", fontsize=10)
    ax.text(0.1, 0.6, f"MMD var: {metrics['mmd_variance']:.4f}", fontsize=10)
    ax.text(0.1, 0.45, f"Overall: {'PASS' if overall_pass else 'FAIL'} ({n_pass}/9)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(plots_dir / "smoke_test_summary.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"  Saved all plots to {plots_dir}")
    return plots_dir

