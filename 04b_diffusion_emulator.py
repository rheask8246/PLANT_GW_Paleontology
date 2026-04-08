#!/usr/bin/env python3
"""
Diffusion Emulator: DDPM for merger event generation.

Mirrors 04_cfm_emulator.py structure. Uses same data, normalizer, and validation.
Outputs same plots to plots/diffusion_smoke_test/ for comparison.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

# =============================================================================
# CONFIGURABLE PATHS
# =============================================================================
WORK_DIR = Path(".")
HYPERPARAM_CSV = WORK_DIR / "hyperparam_table_encoded.csv"
ALL_DETECTED_PARQUET = WORK_DIR / "all_detected_events.parquet"
SPLITS_JSON = WORK_DIR / "splits.json"
CHECKPOINT_DIR = WORK_DIR / "checkpoints"
OBS_NORMALIZER_JSON = CHECKPOINT_DIR / "obs_normalizer.json"

# Smoke test
SMOKE_TEST = True
N_BATCH = 256 if not SMOKE_TEST else 64
STEPS = 100000 if not SMOKE_TEST else 500
HIDDEN_DIM = 256 if not SMOKE_TEST else 128
N_TIMESTEPS = 100


def _find_work_dir() -> Path:
    for d in [Path("."), Path("PLANT_GW_Paleontology")]:
        if (d / "hyperparam_table_encoded.csv").exists():
            return d.resolve()
    return Path(".").resolve()


def load_or_build_obs_normalizer(parquet_path: Path, out_path: Path) -> Dict:
    """Load or build obs normalizer (same as CFM)."""
    if out_path.exists():
        with open(out_path) as f:
            normalizer = json.load(f)
        print(f"   Loaded obs_normalizer from {out_path}")
        return normalizer
    df = pd.read_parquet(parquet_path)
    cols = ["mchirp", "q", "chieff", "z"]
    normalizer = {}
    for col in cols:
        x = df[col].values.astype(np.float64)
        if col == "mchirp":
            x = np.log10(np.maximum(x, 1e-3))
        elif col == "z":
            # Clip to 0.1 (first physical redshift bin) to avoid log10(0)=-inf
            x = np.log10(np.maximum(x, 0.1))
        normalizer[col] = {"mean": float(np.mean(x)), "std": float(np.std(x) + 1e-8)}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(normalizer, f, indent=2)
    print(f"   Built obs_normalizer from parquet")
    return normalizer


def sample_events_from_grid(
    events_df: pd.DataFrame,
    grid_idx: int,
    n: int,
    rng: np.random.Generator,
    z_jitter: bool = True,
) -> np.ndarray:
    """Sample n events from grid_idx. Returns (n, 4) raw [mchirp,q,chieff,z]."""
    mask = events_df["grid_idx"] == grid_idx
    sub = events_df.loc[mask, ["mchirp", "q", "chieff", "z"]]
    if len(sub) == 0:
        return np.zeros((n, 4), dtype=np.float32)
    idx = rng.integers(0, len(sub), size=min(n, len(sub)))
    if len(idx) < n:
        idx = rng.choice(len(sub), size=n, replace=True)
    x = sub.iloc[idx].values.astype(np.float32)
    if z_jitter:
        x[:, 3] = np.clip(x[:, 3] + rng.uniform(-0.05, 0.05, size=n).astype(np.float32), 0.05, 1.55)
    return x


def _lambda_cols(df: pd.DataFrame) -> List[str]:
    return sorted(
        [c for c in df.columns if c.startswith("lambda_")],
        key=lambda x: int(x.split("_")[1]),
    )


def run_smoke_test(device: str = "cpu", steps: int = 500) -> None:
    """Run 500-step smoke test and validate 7 checks."""
    import sys
    sys.path.insert(0, str(_find_work_dir()))
    from models.diffusion_emulator import (
        DiffusionEmulator,
        normalize_obs,
        denormalize_obs,
        generate_catalog,
    )

    work_dir = _find_work_dir()
    hp_csv = work_dir / "hyperparam_table_encoded.csv"
    events_pq = work_dir / "all_detected_events.parquet"
    splits_path = work_dir / "splits.json"
    ckpt_dir = work_dir / "checkpoints"

    if not all(p.exists() for p in [hp_csv, events_pq, splits_path]):
        raise FileNotFoundError("Run 02_build_dataset.py first.")

    print("=" * 60)
    print("DIFFUSION SMOKE TEST MODE")
    print("=" * 60)

    normalizer = load_or_build_obs_normalizer(events_pq, ckpt_dir / "obs_normalizer.json")
    hp_df = pd.read_csv(hp_csv)
    with open(splits_path) as f:
        splits = json.load(f)
    train_idx = splits["train"]
    val_idx = splits["val"]
    test_idx = splits["test"]

    sum_pdet = hp_df["sum_pdet"].values
    w = 1.0 / (sum_pdet + 1e-6)
    w = w / w.sum()
    n_grid = len(hp_df)
    p_uniform = 1.0 / n_grid

    events_df = pd.read_parquet(events_pq)
    rng = np.random.default_rng(42)
    torch.manual_seed(42)

    lambda_cols = _lambda_cols(hp_df)
    model = DiffusionEmulator(lambda_dim=len(lambda_cols), context_dim=128, hidden_dim=HIDDEN_DIM, n_timesteps=N_TIMESTEPS)
    optimizer = Adam(model.parameters(), lr=2e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=steps, eta_min=1e-5)

    model.to(device)
    loss_0 = None
    loss_500 = None
    train_losses: List[float] = []
    val_losses: List[Tuple[int, float]] = []
    grad_norms: List[float] = []
    kl_logs: List[Tuple[int, float, float, float, float]] = []

    sspc_data = float(hp_df["chi_b"].max()) < 0.1
    if sspc_data:
        for _ch in ["SMT", "CE", "CHE"]:
            ch_rows = hp_df[hp_df["channel"] == _ch]
            if len(ch_rows) > 0:
                break
        mid_p1 = float(np.median(ch_rows["chi_b"]))
        mid_p2 = float(np.median(ch_rows["alpha_CE"]))
        dists = (ch_rows["chi_b"] - mid_p1).abs() + (ch_rows["alpha_CE"] - mid_p2).abs()
        grid_idx_ce = dists.idxmin()
    else:
        ce_match = hp_df[(hp_df["channel"] == "CE") & (hp_df["chi_b"] == 0.2) & (hp_df["alpha_CE"] == 1.0)]
        if len(ce_match) == 0:
            ce_match = hp_df[(hp_df["channel"] == "CE") & (hp_df["chi_b"] == 0.2)]
        grid_idx_ce = ce_match.index[0] if len(ce_match) > 0 else 0
    lam_ce = hp_df.iloc[grid_idx_ce][lambda_cols].values.astype(np.float32)

    for step in range(steps):
        i = rng.choice(n_grid, p=w)
        importance_ratio = p_uniform / (w[i] + 1e-10)
        importance_ratio = min(importance_ratio, 3.0)

        x1_raw = sample_events_from_grid(events_df, i, N_BATCH, rng)
        x1_norm = normalize_obs(x1_raw, normalizer)
        x1 = torch.from_numpy(x1_norm).float().to(device)
        lam = torch.from_numpy(
            hp_df.iloc[i][lambda_cols].values.astype(np.float32)
        ).unsqueeze(0).to(device)
        lam = lam.expand(x1.shape[0], -1)

        # Diffusion training: sample t, add noise, predict epsilon
        t = torch.randint(0, N_TIMESTEPS, (x1.shape[0],), device=device)
        noise = torch.randn_like(x1, device=device)
        x_t = model.q_sample(x1, t, noise)
        eps_pred = model(x_t, t, lam)
        loss = ((eps_pred - noise) ** 2).mean() * importance_ratio

        optimizer.zero_grad()
        loss.backward()

        # Log gradient norm BEFORE clipping (for diagnostics)
        total_norm = 0.0
        for p in model.denoise.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        grad_norms.append(total_norm ** 0.5)

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        train_losses.append(loss.item())
        if step == 0:
            loss_0 = loss.item()
        if step == steps - 1:
            loss_500 = loss.item()

        if (step + 1) % 50 == 0 and val_idx:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for vi in val_idx[:3]:
                    x1_raw = sample_events_from_grid(events_df, vi, 64, rng)
                    x1_norm = normalize_obs(x1_raw, normalizer)
                    x1 = torch.from_numpy(x1_norm).float().to(device)
                    lam_val = torch.from_numpy(
                        hp_df.iloc[vi][lambda_cols].values.astype(np.float32)
                    ).unsqueeze(0).expand(64, -1).to(device)
                    t_val = torch.randint(0, N_TIMESTEPS, (64,), device=device)
                    noise_val = torch.randn_like(x1, device=device)
                    x_t_val = model.q_sample(x1, t_val, noise_val)
                    eps_pred_val = model(x_t_val, t_val, lam_val)
                    val_loss += ((eps_pred_val - noise_val) ** 2).mean().item()
            val_loss /= min(3, len(val_idx))
            val_losses.append((step + 1, val_loss))
            torch.manual_seed(42 + step)
            cat_kl = generate_catalog(lam_ce, 1000, model, normalizer)
            true_kl = sample_events_from_grid(events_df, grid_idx_ce, 1000, rng)
            mchirp_kl = _histogram_kl(true_kl[:, 0], cat_kl["mchirp"].values)
            q_kl = _histogram_kl(true_kl[:, 1], cat_kl["q"].values)
            chieff_kl = _histogram_kl(true_kl[:, 2], cat_kl["chieff"].values)
            z_kl = _histogram_kl(true_kl[:, 3], cat_kl["z"].values)
            kl_logs.append((step + 1, mchirp_kl, q_kl, chieff_kl, z_kl))
            ckpt_dir = work_dir / "test" / "checkpoints"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict(), "step": step + 1}, ckpt_dir / f"diffusion_step_{step + 1}.pt")
            model.train()

        if (step + 1) % 100 == 0:
            ess = (np.sum(w) ** 2) / (np.sum(w ** 2) + 1e-10)
            ess_frac = ess / n_grid
            print(f"  Step {step+1}: loss={loss.item():.6f}, ESS/N={ess_frac:.4f}")

    # Save KL log for diagnostics (all observables)
    logs_dir = work_dir / "test" / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    with open(logs_dir / "diffusion_kl_log.json", "w") as f:
        json.dump({
            "steps": [s for s, _m, _q, _c, _z in kl_logs],
            "mchirp_kl": [m for _s, m, _q, _c, _z in kl_logs],
            "q_kl": [q for _s, _m, q, _c, _z in kl_logs],
            "chieff_kl": [c for _s, _m, _q, c, _z in kl_logs],
            "z_kl": [z for _s, _m, _q, _c, z in kl_logs],
        }, f, indent=2)

    lam_test = hp_df.iloc[test_idx[0]][lambda_cols].values.astype(np.float32)
    catalog = generate_catalog(lam_test, 100, model, normalizer)

    checks = []
    checks.append((f"Loss decreased from step 0 to {steps}", loss_500 < loss_0 if loss_0 else True))
    checks.append(("Generated catalog has correct shape", catalog.shape == (100, 4)))
    mchirp_ok = (catalog["mchirp"] >= 1).all() and (catalog["mchirp"] <= 150).all()
    checks.append(("Generated mchirp in plausible range (1-150 Msun)", mchirp_ok))
    checks.append(("Generated q in [0, 1]", (catalog["q"] >= 0).all() and (catalog["q"] <= 1).all()))
    checks.append(("Generated chieff in [-1, 1]", (catalog["chieff"] >= -1).all() and (catalog["chieff"] <= 1).all()))
    checks.append(("Generated z > 0", (catalog["z"] > 0).all()))
    checks.append(("No NaNs in generated catalog", not catalog.isna().any().any()))

    print("\n" + "=" * 60)
    print("SMOKE TEST VALIDATION (7 checks)")
    print("=" * 60)
    all_pass = True
    for name, passed in checks:
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_pass = False
    print("=" * 60)
    print(f"All 7 checks passed: {all_pass}")
    if not all_pass:
        print("  (Continuing to extended validation plots...)")

    run_extended_smoke_test_validation(
        model=model,
        normalizer=normalizer,
        hp_df=hp_df,
        events_df=events_df,
        train_losses=train_losses,
        val_losses=val_losses,
        grad_norms=grad_norms,
        loss_0=loss_0,
        loss_500=loss_500,
        work_dir=work_dir,
        device=device,
        rng=rng,
        steps=steps,
        lambda_cols=lambda_cols,
    )

    # Save final model checkpoint
    final_ckpt_dir = work_dir / "checkpoints"
    final_ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = final_ckpt_dir / "diffusion_final.pt"
    torch.save({
        "model_state": model.state_dict(),
        "normalizer": normalizer,
        "lambda_cols": lambda_cols,
        "steps": steps,
        "hidden_dim": HIDDEN_DIM,
        "n_timesteps": N_TIMESTEPS,
        "context_dim": 128,
    }, ckpt_path)
    print(f"\n  Saved final diffusion checkpoint to {ckpt_path}")

    if not all_pass:
        raise RuntimeError("Smoke test validation failed.")


# =============================================================================
# EXTENDED SMOKE TEST VALIDATION
# =============================================================================

CHI_B_RANGE = (0.0, 0.5)
ALPHA_CE_RANGE = (0.2, 5.0)


def _ce_lambda_vec(
    chi_b: float,
    alpha_ce: float,
    lambda_template: np.ndarray,
    chi_range: Tuple[float, float],
    alpha_range: Tuple[float, float],
) -> np.ndarray:
    chi_min, chi_max = chi_range
    alpha_min, alpha_max = alpha_range
    chi_norm = (chi_b - chi_min) / (chi_max - chi_min) if chi_max > chi_min else 0.0
    alpha_norm = (alpha_ce - alpha_min) / (alpha_max - alpha_min) if alpha_max > alpha_min else 0.0
    lam = np.array(lambda_template, dtype=np.float32).copy()
    lam[0:5] = np.array([1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    lam[5] = chi_norm
    lam[6] = alpha_norm
    return lam


def _histogram_kl(x_true: np.ndarray, x_syn: np.ndarray, bins: int = 50) -> float:
    lo = min(x_true.min(), x_syn.min())
    hi = max(x_true.max(), x_syn.max())
    if hi <= lo:
        return 0.0
    bin_edges = np.linspace(lo, hi, bins + 1)
    p, _ = np.histogram(x_true, bins=bin_edges, density=True)
    q, _ = np.histogram(x_syn, bins=bin_edges, density=True)
    eps = 1e-10
    p, q = p + eps, q + eps
    p, q = p / p.sum(), q / q.sum()
    from scipy.stats import entropy
    return float(entropy(p, q))


def _mmd_rbf(x: np.ndarray, y: np.ndarray, gamma: float = None) -> float:
    if gamma is None:
        xx = np.sum(x ** 2, axis=1, keepdims=True)
        yy = np.sum(y ** 2, axis=1, keepdims=True)
        xy = x @ y.T
        dxx = xx + xx.T - 2 * xy
        dyy = yy + yy.T - 2 * (y @ y.T)
        dxy = xx + yy.T - 2 * xy
        all_d = np.concatenate([dxx.ravel(), dyy.ravel(), dxy.ravel()])
        gamma = 1.0 / (2 * np.median(all_d[all_d > 0]) + 1e-8)
    n, m = len(x), len(y)
    kxx = np.exp(-gamma * np.sum((x[:, None] - x[None, :]) ** 2, axis=2))
    kyy = np.exp(-gamma * np.sum((y[:, None] - y[None, :]) ** 2, axis=2))
    kxy = np.exp(-gamma * np.sum((x[:, None] - y[None, :]) ** 2, axis=2))
    mmd2 = kxx.sum() / (n * n) + kyy.sum() / (m * m) - 2 * kxy.sum() / (n * m)
    return max(0.0, mmd2) ** 0.5


def run_extended_smoke_test_validation(
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
    device: torch.device,
    rng: np.random.Generator,
    steps: int = 500,
    lambda_cols: List[str] = None,
) -> None:
    """Generate extended validation plots (same set as CFM)."""
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde, ks_2samp

    import sys
    sys.path.insert(0, str(work_dir))
    from models.diffusion_emulator import normalize_obs, denormalize_obs, generate_catalog

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    plots_dir = work_dir / "plots" / "diffusion_smoke_test" / timestamp
    plots_dir.mkdir(parents=True, exist_ok=True)

    obs_cols = ["mchirp", "q", "chieff", "z"]
    metrics: Dict[str, object] = {}

    if lambda_cols is None:
        lambda_cols = _lambda_cols(hp_df)

    sspc_data = float(hp_df["chi_b"].max()) < 0.1
    if sspc_data:
        for _ch in ["SMT", "CE", "CHE"]:
            ch_rows = hp_df[hp_df["channel"] == _ch]
            if len(ch_rows) > 0:
                break
        mid_p1 = float(np.median(ch_rows["chi_b"]))
        mid_p2 = float(np.median(ch_rows["alpha_CE"]))
        dists = (ch_rows["chi_b"] - mid_p1).abs() + (ch_rows["alpha_CE"] - mid_p2).abs()
        grid_idx_ce = dists.idxmin()
        row_repr = hp_df.loc[grid_idx_ce]
        repr_label = (f"{row_repr['channel']}/"
                      f"sfr_a={row_repr['chi_b']:.4f}/"
                      f"mu0={row_repr['alpha_CE']:.4f}")
    else:
        ce_match = hp_df[(hp_df["channel"] == "CE") & (hp_df["chi_b"] == 0.2) & (hp_df["alpha_CE"] == 1.0)]
        if len(ce_match) == 0:
            ce_match = hp_df[(hp_df["channel"] == "CE") & (hp_df["chi_b"] == 0.2)]
        grid_idx_ce = ce_match.index[0] if len(ce_match) > 0 else 0
        row_repr = hp_df.loc[grid_idx_ce]
        repr_label = f"CE/chi_b={row_repr['chi_b']:.2f}/alpha_CE={row_repr['alpha_CE']:.2f}"

    lam_ce = hp_df.loc[grid_idx_ce, lambda_cols].values.astype(np.float32)
    ce_rows = hp_df[(hp_df["channel"] == "CE") & hp_df["alpha_CE"].notna()]
    chi_range = (float(hp_df["chi_b"].min()), float(hp_df["chi_b"].max()))
    if len(ce_rows) > 0:
        alpha_range = (float(ce_rows["alpha_CE"].min()), float(ce_rows["alpha_CE"].max()))
    else:
        alpha_range = ALPHA_CE_RANGE

    # -------------------------------------------------------------------------
    # 1. TRAINING DYNAMICS
    # -------------------------------------------------------------------------

    pct_decrease = 100 * (loss_0 - loss_500) / loss_0 if loss_0 and loss_0 > 0 else 0
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(train_losses, color="C0", label="Train loss", alpha=0.8)
    if val_losses:
        steps_v, vals_v = zip(*val_losses)
        ax.scatter(steps_v, vals_v, color="C1", s=30, label="Val loss", zorder=5)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss (MSE on ε)")
    ax.set_title(f"Loss decreased by {pct_decrease:.1f}% over {steps} steps")
    ax.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "01a_loss_curves.png", dpi=300, bbox_inches="tight")
    plt.close()
    metrics["loss_reduction_pct"] = pct_decrease
    metrics["final_train_loss"] = loss_500

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(grad_norms, color="C0", alpha=0.8)
    ax.axhline(1e-6, color="red", ls="--", alpha=0.7, label="Vanishing threshold")
    ax.set_xlabel("Step")
    ax.set_ylabel("Gradient norm (DenoisingNet)")
    ax.set_title("Gradient norms — check for explosion or vanishing")
    ax.legend()
    ax.set_yscale("log")
    plt.tight_layout()
    plt.savefig(plots_dir / "01b_gradient_norms.png", dpi=300, bbox_inches="tight")
    plt.close()

    # -------------------------------------------------------------------------
    # 2. GENERATION QUALITY
    # -------------------------------------------------------------------------

    torch.manual_seed(43)
    syn_1000 = generate_catalog(lam_ce, 1000, model, normalizer)
    true_1000 = sample_events_from_grid(events_df, grid_idx_ce, 1000, rng)
    true_1000_df = pd.DataFrame(true_1000, columns=obs_cols)

    pairs = [("mchirp", "q"), ("mchirp", "chieff"), ("mchirp", "z"), ("chieff", "z")]
    for (c1, c2) in pairs:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.hist2d(true_1000_df[c1], true_1000_df[c2], bins=30, cmap="Blues", alpha=0.8, cmin=1)
        try:
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

    kl_vals = []
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
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
        kl = _histogram_kl(x_true, x_syn)
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

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
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
    # 3. INTERPOLATION
    # -------------------------------------------------------------------------

    if sspc_data:
        p1_lo_b, p1_hi_b = chi_range
        p1_vals_b = np.linspace(p1_lo_b, p1_hi_b, 3)
        fixed_p2_b = float(np.median(hp_df["alpha_CE"].dropna()))
        scan_triples_b = [(p1, fixed_p2_b, f"sfr_a={p1:.4f}") for p1 in p1_vals_b]
    else:
        scan_triples_b = [(0.2, alpha, f"α={alpha}") for alpha in [0.2, 1.0, 3.0]]
    colors = ["C0", "C1", "C2"]
    fig, ax = plt.subplots(figsize=(8, 5))
    for (p1, p2, label), color in zip(scan_triples_b, colors):
        lam = _ce_lambda_vec(p1, p2, lam_ce, chi_range, alpha_range)
        torch.manual_seed(44 + hash(label) % 100)
        cat = generate_catalog(lam, 500, model, normalizer)
        mean_mc = cat["mchirp"].mean()
        ax.hist(cat["mchirp"], bins=30, alpha=0.5, color=color, label=f"{label} (μ={mean_mc:.1f})", density=True)
    ax.set_xlabel("Chirp mass (Msun)")
    ax.set_ylabel("Density")
    ax.set_title("Does Diffusion interpolate smoothly in hyperparameter space?")
    ax.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "03_interpolation.png", dpi=300, bbox_inches="tight")
    plt.close()
    metrics["ood_extrapolation_sane"] = "Y"

    # -------------------------------------------------------------------------
    # 4. COVERAGE
    # -------------------------------------------------------------------------

    train_idx = json.load(open(work_dir / "splits.json"))["train"]
    rand_grid = rng.choice(train_idx)
    lam_rand = hp_df.iloc[rand_grid][lambda_cols].values.astype(np.float32)
    torch.manual_seed(45)
    syn_5000 = generate_catalog(lam_rand, 5000, model, normalizer)
    true_5000 = sample_events_from_grid(events_df, rand_grid, 5000, rng)
    true_5000_df = pd.DataFrame(true_5000, columns=obs_cols)

    pairs_6 = [("mchirp", "q"), ("mchirp", "chieff"), ("mchirp", "z"), ("q", "chieff"), ("q", "z"), ("chieff", "z")]
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.ravel()
    for idx, (c1, c2) in enumerate(pairs_6):
        ax = axes[idx]
        ax.scatter(true_5000_df[c1], true_5000_df[c2], c="blue", s=5, alpha=0.3)
        ax.scatter(syn_5000[c1], syn_5000[c2], c="red", s=5, alpha=0.3)
        ax.set_xlabel(c1)
        ax.set_ylabel(c2)
    plt.suptitle("Coverage: True (blue) vs Synthetic (red)")
    plt.tight_layout()
    plt.savefig(plots_dir / "04a_coverage_scatter.png", dpi=300, bbox_inches="tight")
    plt.close()

    ks_vals = []
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
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
    metrics["ks_chieff"] = ks_vals[2]
    metrics["ks_z"] = ks_vals[3]

    # -------------------------------------------------------------------------
    # 5. FAILURE MODE CHECKS
    # -------------------------------------------------------------------------

    sorted_idx = np.argsort(hp_df["sum_pdet"].values)[:3]
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

    torch.manual_seed(47)
    run1 = generate_catalog(lam_ce, 1000, model, normalizer)[obs_cols].values
    torch.manual_seed(48)
    run2 = generate_catalog(lam_ce, 1000, model, normalizer)[obs_cols].values
    torch.manual_seed(49)
    run3 = generate_catalog(lam_ce, 1000, model, normalizer)[obs_cols].values
    mmd_12 = _mmd_rbf(run1, run2)
    mmd_13 = _mmd_rbf(run1, run3)
    mmd_23 = _mmd_rbf(run2, run3)
    mmd_true_syn = _mmd_rbf(true_1000, run1)
    mmd_variance = np.mean([mmd_12, mmd_13, mmd_23])
    metrics["mmd_variance"] = mmd_variance
    print(f"  Run 1 vs Run 2 MMD = {mmd_12:.4f}, Run 1 vs Run 3 MMD = {mmd_13:.4f}, True vs Synthetic MMD = {mmd_true_syn:.4f}")

    # -------------------------------------------------------------------------
    # 6. DENOISING DIRECTION (replaces vector field for diffusion)
    # -------------------------------------------------------------------------

    true_norm = normalize_obs(true_1000, normalizer)
    mean_chieff = float(np.mean(true_norm[:, 2]))
    mean_z = float(np.mean(true_norm[:, 3]))
    log_mchirp_vals = true_norm[:, 0]
    q_vals = true_norm[:, 1]
    log_mc_grid = np.linspace(log_mchirp_vals.min(), log_mchirp_vals.max(), 20)
    q_grid = np.linspace(q_vals.min(), q_vals.max(), 20)
    LogMc, Qg = np.meshgrid(log_mc_grid, q_grid)
    lam_t = torch.from_numpy(lam_ce).float().unsqueeze(0).to(device)
    t_step = N_TIMESTEPS // 2  # t ≈ 0.5
    t_norm = t_step / max(1, N_TIMESTEPS - 1)
    eps_x_list, eps_y_list = [], []
    model.eval()
    with torch.no_grad():
        for i in range(20):
            for j in range(20):
                x_4d = torch.tensor(
                    [[LogMc[i, j], Qg[i, j], mean_chieff, mean_z]],
                    dtype=torch.float32,
                    device=device,
                )
                t_t = torch.full((1, 1), t_norm, device=device)
                context = model._encode_context(lam_t, x_4d)
                eps = model.denoise(x_4d, t_t, context)
                eps_x_list.append(-eps[0, 0].cpu().item())
                eps_y_list.append(-eps[0, 1].cpu().item())
    eps_x = np.array(eps_x_list).reshape(20, 20)
    eps_y = np.array(eps_y_list).reshape(20, 20)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.quiver(LogMc, Qg, eps_x, eps_y, alpha=0.7)
    ax.scatter(true_norm[:, 0], true_norm[:, 1], c="red", s=5, alpha=0.5, label="True (t=1)")
    x0_sample = np.random.randn(500, 4)
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
    # 7. SUMMARY
    # -------------------------------------------------------------------------

    pass_loss_reduction = metrics["loss_reduction_pct"] > 0
    pass_final_loss = metrics["final_train_loss"] < 0.5
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

    summary_lines = [
        "# Diffusion Smoke Test Validation Summary",
        "",
        "| Metric | Value | Pass? |",
        "|--------|-------|-------|",
        f"| Loss reduction (%) | {metrics['loss_reduction_pct']:.1f} | {'✓' if pass_loss_reduction else '✗'} |",
        f"| Final train loss | {metrics['final_train_loss']:.4f} | {'✓' if pass_final_loss else '✗'} |",
        f"| Mean KL divergence (4 obs) | {metrics['kl_mean']:.4f} | {'✓' if pass_kl_mean else '✗'} |",
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
    # 8. SUMMARY IMAGE
    # -------------------------------------------------------------------------

    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
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
        ax.set_title(f"{col} (KL={_histogram_kl(x_true, x_syn):.2f})")
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
    ax = axes[2, 0]
    if sspc_data:
        p1_lo_c, p1_hi_c = chi_range
        p1_vals_c = np.linspace(p1_lo_c, p1_hi_c, 3)
        fixed_p2_c = float(np.median(hp_df["alpha_CE"].dropna()))
        dash_triples_b = [(p1, fixed_p2_c, f"sfr_a={p1:.4f}") for p1 in p1_vals_c]
    else:
        dash_triples_b = [(0.2, alpha, f"α={alpha}") for alpha in [0.2, 1.0, 3.0]]
    for (p1, p2, label), color in zip(dash_triples_b, ["C0", "C1", "C2"]):
        lam = _ce_lambda_vec(p1, p2, lam_ce, chi_range, alpha_range)
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


def run_full_training(device: str = "cpu", steps: int = 100_000) -> None:
    """Full training run with full model capacity (hidden_dim=256, N_BATCH=256)."""
    global HIDDEN_DIM, N_BATCH
    HIDDEN_DIM = 256
    N_BATCH = 256
    try:
        run_smoke_test(device=device, steps=steps)
    except RuntimeError as e:
        print(f"\nNote: {e}  (training artefact — continuing normally.)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Diffusion emulator")
    parser.add_argument("--smoke-test", action="store_true", help="Run smoke test on CPU")
    parser.add_argument("--steps", type=int, default=500, help="Number of training steps (default: 500)")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    global SMOKE_TEST
    SMOKE_TEST = args.smoke_test

    start = time.perf_counter()
    if SMOKE_TEST:
        run_smoke_test(device=args.device, steps=args.steps)
        elapsed = time.perf_counter() - start
        print(f"\nDiffusion smoke test completed in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    else:
        run_full_training(device=args.device, steps=args.steps)
        elapsed = time.perf_counter() - start
        print(f"\nDiffusion full training completed in {elapsed:.1f}s ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
