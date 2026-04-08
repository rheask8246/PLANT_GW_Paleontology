#!/usr/bin/env python3
"""
Train and validate rate network predicting log10(sum_pdet) from hyperparameters.

Uses hyperparam_table_encoded.csv and splits.json from 02_build_dataset.py.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

# =============================================================================
# CONFIGURABLE PATHS
# =============================================================================
WORK_DIR = Path(".")
HYPERPARAM_CSV = WORK_DIR / "hyperparam_table_encoded.csv"
SPLITS_JSON = WORK_DIR / "splits.json"
CHECKPOINT_DIR = WORK_DIR / "checkpoints"
BEST_MODEL_PATH = CHECKPOINT_DIR / "rate_network_best.pt"
CONFIG_PATH = CHECKPOINT_DIR / "rate_network_config.json"
GP_BASELINE_PATH = CHECKPOINT_DIR / "gp_rate_baseline.pkl"
PLOTS_DIR = WORK_DIR / "plots" / "rate_network"

# Normalization ranges
SSPC_SFRA_RANGE = (0.008, 0.035)   # sfr_a (stored as chi_b in SSPC hyperparam CSV)
SSPC_MU0_RANGE  = (0.005, 0.065)   # mu0   (stored as alpha_CE in SSPC hyperparam CSV)
CHANNEL_COLORS = {"CE": "red", "CHE": "blue", "GC": "green", "NSC": "orange", "SMT": "purple"}


def _is_sspc(df: pd.DataFrame) -> bool:
    """Return True if this is SSPC data (sfr_a/mu0 axes) vs Zenodo (chi_b/alpha_CE)."""
    return float(df["chi_b"].max()) < 0.1   # SSPC sfr_a max ≈ 0.035; Zenodo chi_b max ≈ 0.5


def _find_work_dir() -> Path:
    """Resolve work dir from cwd."""
    for d in [Path("."), Path("PLANT_GW_Paleontology")]:
        if (d / "hyperparam_table_encoded.csv").exists():
            return d.resolve()
    return Path(".").resolve()


def _lambda_cols(df: pd.DataFrame) -> list[str]:
    """Return lambda_* columns sorted by numeric suffix."""
    return sorted(
        [c for c in df.columns if c.startswith("lambda_")],
        key=lambda x: int(x.split("_")[1]),
    )


def _rate_label(df: pd.DataFrame) -> str:
    """Return the y-axis label for the rate network target."""
    return "log₁₀(Σ det_weight)" if _is_sspc(df) else "log₁₀(sum_pdet)"


def load_data(
    hyperparam_csv: Path,
    splits_json: Path,
) -> tuple[np.ndarray, np.ndarray, list[int], list[int], list[int], pd.DataFrame, list[str]]:
    """Load encoded table and splits. Returns X, y, splits, df, lambda_cols."""
    df = pd.read_csv(hyperparam_csv)
    with open(splits_json) as f:
        splits = json.load(f)

    lambda_cols = _lambda_cols(df)
    X = df[lambda_cols].values.astype(np.float32)
    y = np.log10(df["sum_pdet"].values.astype(np.float64) + 1e-300)  # avoid log(0)

    return X, y, splits["train"], splits["val"], splits["test"], df, lambda_cols


def train_rate_network(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 1000,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 100,
    log_every: int = 50,
    device: str = "cpu",
    input_dim: int = 7,
) -> tuple[nn.Module, dict, list[float], list[float], int, float]:
    """Train RateNetwork. Uses val for early stopping, then retrains on train+val for final model.
    Returns (model, config, train_losses, val_losses, best_epoch, best_val_loss)."""
    sys.path.insert(0, str(Path(__file__).parent))
    from models.rate_network import RateNetwork

    model = RateNetwork(input_dim=input_dim, hidden_dims=(64, 64, 32)).to(device)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()

    X_t = torch.from_numpy(X_train).float().to(device)
    y_t = torch.from_numpy(y_train).float().to(device)
    X_v = torch.from_numpy(X_val).float().to(device)
    y_v = torch.from_numpy(y_val).float().to(device)

    best_val_loss = float("inf")
    best_epoch = 0
    best_state = None
    wait = 0
    train_losses: list[float] = []
    val_losses: list[float] = []

    for ep in range(epochs):
        model.train()
        pred = model(X_t)
        loss = criterion(pred, y_t)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        train_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            val_pred = model(X_v)
            val_loss = criterion(val_pred, y_v).item()
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = ep + 1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1

        if (ep + 1) % log_every == 0:
            val_pred_np = val_pred.cpu().numpy()
            print(f"Epoch {ep+1}: train_loss={loss.item():.6f}, val_loss={val_loss:.6f}")
            print("  Val actual vs predicted:")
            for i in range(len(y_val)):
                print(f"    {i}: true={y_val[i]:.4f}, pred={val_pred_np[i]:.4f}")

        if wait >= patience:
            print(f"Early stopping at epoch {ep+1}, best epoch was {best_epoch}")
            break

    # Retrain on train+val for best_epoch steps (more data = better generalization)
    X_all = np.concatenate([X_train, X_val], axis=0)
    y_all = np.concatenate([y_train, y_val], axis=0)
    X_all_t = torch.from_numpy(X_all).float().to(device)
    y_all_t = torch.from_numpy(y_all).float().to(device)

    model = RateNetwork(input_dim=input_dim, hidden_dims=(64, 64, 32)).to(device)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=best_epoch)

    for ep in range(best_epoch):
        model.train()
        pred = model(X_all_t)
        loss = criterion(pred, y_all_t)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

    print(f"  Retrained on train+val for {best_epoch} epochs")

    config = {
        "input_dim": input_dim,
        "hidden_dims": [64, 64, 32],
        "n_params": model.n_params,
    }
    return model, config, train_losses, val_losses, best_epoch, best_val_loss


def train_gp_baseline(X_train: np.ndarray, y_train: np.ndarray) -> "GaussianProcessRate":
    """Fit GP baseline."""
    from models.rate_network import GaussianProcessRate

    gp = GaussianProcessRate()
    gp.fit(X_train, y_train)
    return gp


def plot_evaluation(
    df: pd.DataFrame,
    test_idx: list[int],
    pred_nn: np.ndarray,
    pred_gp: np.ndarray,
    y_test: np.ndarray,
    model,
    gp,
    ckpt_dir: Path,
    lambda_cols: list[str],
) -> None:
    """Generate evaluation plots: scatter, residuals, CE heatmap."""
    import matplotlib.pyplot as plt

    df_test = df.iloc[test_idx].copy()
    df_test["y_true"] = y_test
    df_test["pred_nn"] = pred_nn
    df_test["pred_gp"] = pred_gp
    df_test["resid_nn"] = pred_nn - y_test
    df_test["resid_gp"] = pred_gp - y_test

    _rate_lbl = _rate_label(df)
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    # Predicted vs true (both NN and GP)
    ax = axes[0, 0]
    lo = min(y_test.min(), pred_nn.min(), pred_gp.min())
    hi = max(y_test.max(), pred_nn.max(), pred_gp.max())
    ax.scatter(y_test, pred_nn, alpha=0.7, label="NN", s=50)
    ax.scatter(y_test, pred_gp, alpha=0.7, label="GP", s=50, marker="^")
    ax.plot([lo, hi], [lo, hi], "k--", label="y=x")
    ax.set_xlabel(f"True {_rate_lbl}")
    ax.set_ylabel(f"Predicted {_rate_lbl}")
    ax.legend()
    ax.set_title("Predicted vs True (test set)")
    ax.set_aspect("equal")

    # Residuals vs primary parameter (sfr_a for SSPC, chi_b for Zenodo)
    _sspc = _is_sspc(df)
    _p1, _p2 = ("sfr_a", "mu0") if _sspc else ("chi_b", "alpha_CE")
    ax = axes[0, 1]
    ax.scatter(df_test["chi_b"], df_test["resid_nn"], alpha=0.7, label="NN")
    ax.scatter(df_test["chi_b"], df_test["resid_gp"], alpha=0.7, label="GP", marker="^")
    ax.axhline(0, color="k", ls="--")
    ax.set_xlabel(_p1)
    ax.set_ylabel("Residual")
    ax.legend()
    ax.set_title(f"Residuals vs {_p1}")

    # Residuals vs secondary parameter
    ax = axes[1, 0]
    for ch_m in df_test["channel"].unique():
        sub = df_test[df_test["channel"] == ch_m]
        ax.scatter(sub["alpha_CE"], sub["resid_nn"], alpha=0.7, label=ch_m)
    ax.axhline(0, color="k", ls="--")
    ax.set_xlabel(_p2)
    ax.set_ylabel("Residual")
    ax.legend()
    ax.set_title(f"Residuals vs {_p2}")

    # Heatmap: predicted rate over (p1, p2)
    ax = axes[1, 1]
    hm_ch = next((c for c in ["CE", "SMT", "CHE"] if len(df[df["channel"] == c]) > 0), None)
    hm_df = df[df["channel"] == hm_ch].copy() if hm_ch else pd.DataFrame()
    if len(hm_df) > 0:
        X_hm = hm_df[lambda_cols].values.astype(np.float32)
        with torch.no_grad():
            pred_hm_nn = model(torch.from_numpy(X_hm).float()).numpy()
        true_hm = np.log10(hm_df["sum_pdet"].values + 1e-300)

        p1v = sorted(hm_df["chi_b"].unique())
        p2v = sorted(hm_df["alpha_CE"].unique())
        pg1 = np.linspace(min(p1v), max(p1v), 50)
        pg2 = np.linspace(min(p2v), max(p2v), 50)
        from scipy.interpolate import griddata
        points = hm_df[["chi_b", "alpha_CE"]].values
        Xg, Yg = np.meshgrid(pg1, pg2)
        xi = np.column_stack([Xg.ravel(), Yg.ravel()])
        z_nn = griddata(points, pred_hm_nn, xi, method="linear")
        z_nn = z_nn.reshape(Xg.shape)
        im = ax.imshow(z_nn, extent=[pg1[0], pg1[-1], pg2[0], pg2[-1]], origin="lower", aspect="auto", cmap="viridis")
        ax.scatter(hm_df["chi_b"], hm_df["alpha_CE"], c=true_hm, s=80, cmap="viridis", edgecolor="white", linewidths=1)
        plt.colorbar(im, ax=ax, label=_rate_lbl)
        ax.set_xlabel(_p1)
        ax.set_ylabel(_p2)
        ax.set_title(f"{hm_ch}: predicted heatmap + true")
    plt.tight_layout()
    plt.savefig(ckpt_dir / "rate_evaluation.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   Saved: {ckpt_dir / 'rate_evaluation.png'}")


def _sspc_lambda_vec(
    sfr_a: float,
    mu0: float,
    channel_id: int,
    lambda_template: np.ndarray,
    sfra_range: tuple[float, float] = SSPC_SFRA_RANGE,
    mu0_range: tuple[float, float]  = SSPC_MU0_RANGE,
) -> np.ndarray:
    """Build lambda_vec for SSPC data from (sfr_a, mu0) and channel_id."""
    sfra_norm = (sfr_a - sfra_range[0]) / (sfra_range[1] - sfra_range[0]) if sfra_range[1] > sfra_range[0] else 0.0
    mu0_norm  = (mu0   - mu0_range[0])  / (mu0_range[1]  - mu0_range[0])  if mu0_range[1]  > mu0_range[0]  else 0.0
    lam = np.array(lambda_template, dtype=np.float32).copy()
    if lam.shape[0] < 7:
        raise ValueError(f"Expected >=7 lambda dims, got {lam.shape[0]}")
    lam[0:5] = 0.0
    if channel_id < 5:
        lam[channel_id] = 1.0
    lam[5] = np.float32(sfra_norm)
    lam[6] = np.float32(mu0_norm)
    return lam


def _ce_lambda_vec(
    chi_b: float,
    alpha_ce: float,
    lambda_template: np.ndarray,
    chi_range: tuple[float, float],
    alpha_range: tuple[float, float],
) -> np.ndarray:
    """Build lambda_vec for Zenodo CE channel from (chi_b, alpha_CE). Legacy."""
    chi_min, chi_max = chi_range
    alpha_min, alpha_max = alpha_range
    chi_norm = (chi_b - chi_min) / (chi_max - chi_min) if chi_max > chi_min else 0.0
    alpha_norm = (alpha_ce - alpha_min) / (alpha_max - alpha_min) if alpha_max > alpha_min else 0.0
    lam = np.array(lambda_template, dtype=np.float32).copy()
    if lam.shape[0] < 7:
        raise ValueError(f"Expected >=7 lambda dims, got {lam.shape[0]}")
    lam[0:5] = np.array([1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    lam[5] = chi_norm
    lam[6] = alpha_norm
    return lam


def evaluate(
    model_or_gp,
    X: np.ndarray,
    y: np.ndarray,
    is_gp: bool = False,
) -> tuple[np.ndarray, float, float]:
    """Return predictions, R², MSE."""
    if is_gp:
        pred = model_or_gp.predict(X)
    else:
        with torch.no_grad():
            X_t = torch.from_numpy(X).float()
            pred = model_or_gp(X_t).numpy()
    mse = float(np.mean((pred - y) ** 2))
    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return pred, r2, mse


# =============================================================================
# DETAILED VALIDATION PLOTS
# =============================================================================


def plot_detailed_validation(
    df: pd.DataFrame,
    train_idx: list[int],
    val_idx: list[int],
    test_idx: list[int],
    X: np.ndarray,
    y: np.ndarray,
    pred: np.ndarray,
    model: nn.Module,
    train_losses: list[float],
    val_losses: list[float],
    best_epoch: int,
    best_val_loss: float,
    r2_overall: float,
    plots_dir: Path,
    lambda_cols: list[str],
) -> dict:
    """Generate detailed validation plots. Returns metrics dict for README."""
    import matplotlib.pyplot as plt

    plots_dir.mkdir(parents=True, exist_ok=True)
    device = next(model.parameters()).device
    df_all = df.copy()
    df_all["y_true"] = y
    df_all["pred"] = pred
    df_all["residual"] = pred - y

    df_train = df_all.iloc[train_idx]
    df_val = df_all.iloc[val_idx]
    df_test = df_all.iloc[test_idx]

    # Use test set for prediction quality (primary evaluation)
    y_test = y[test_idx]
    pred_test = pred[test_idx]

    metrics: dict = {"r2_overall": r2_overall, "r2_per_channel": {}, "rmse_per_channel": {}}

    # -------------------------------------------------------------------------
    # 1. PREDICTION QUALITY PLOTS
    # -------------------------------------------------------------------------

    # 1a) Predicted vs True scatter (enhanced)
    rate_lbl = _rate_label(df)
    fig, ax = plt.subplots(figsize=(8, 8))
    lo = min(y_test.min(), pred_test.min())
    hi = max(y_test.max(), pred_test.max())
    margin = 0.1 * (hi - lo) if hi > lo else 0.1
    lo, hi = lo - margin, hi + margin

    # ±0.5 dex shaded band
    ax.fill_between([lo, hi], [lo - 0.5, hi - 0.5], [lo + 0.5, hi + 0.5], alpha=0.2, color="gray", label="±0.5 dex")

    for ch, color in CHANNEL_COLORS.items():
        mask = df_test["channel"] == ch
        if mask.any():
            ax.scatter(
                df_test.loc[mask, "y_true"],
                df_test.loc[mask, "pred"],
                c=color,
                label=ch,
                s=80,
                alpha=0.8,
                edgecolors="black",
                linewidths=0.5,
            )

    # Annotate outliers (residual > 0.5 dex)
    outlier_mask = np.abs(df_test["residual"].values) > 0.5
    for i in np.where(outlier_mask)[0]:
        row = df_test.iloc[i]
        txt = f"χ_b={row['chi_b']:.1f}"
        if pd.notna(row.get("alpha_CE")):
            txt += f", α_CE={row['alpha_CE']:.1f}"
        ax.annotate(txt, (row["y_true"], row["pred"]), fontsize=7, alpha=0.8)

    ax.plot([lo, hi], [lo, hi], "k--", lw=2, label="y=x")
    ax.set_xlabel(f"True {rate_lbl}")
    ax.set_ylabel(f"Predicted {rate_lbl}")
    ax.set_title(f"Predicted vs True (test set) — R² = {r2_overall:.4f}")
    ax.legend(loc="lower right")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(plots_dir / "01a_predicted_vs_true.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 1b) Residual plots
    sspc = _is_sspc(df)
    p1_col   = "chi_b"                      # raw sfr_a for SSPC, chi_b for Zenodo
    p2_col   = "alpha_CE"                   # raw mu0 for SSPC, alpha_CE for Zenodo
    p1_label = "sfr_a" if sspc else "chi_b"
    p2_label = "mu0"   if sspc else "alpha_CE"

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    mean_res = float(np.mean(df_test["residual"]))
    std_res  = float(np.std(df_test["residual"]))

    ax = axes[0]
    for ch, color in CHANNEL_COLORS.items():
        mask = df_test["channel"] == ch
        if mask.any():
            ax.scatter(df_test.loc[mask, p1_col], df_test.loc[mask, "residual"],
                       c=color, label=ch, alpha=0.8, s=50)
    ax.axhline(0, color="k", ls="--")
    ax.axhspan(-0.3, 0.3, alpha=0.2, color="gray")
    ax.set_xlabel(p1_label)
    ax.set_ylabel("Residual (dex)")
    ax.set_title(f"Residuals vs {p1_label} — Mean={mean_res:.4f}, Std={std_res:.4f}")
    ax.legend()

    ax = axes[1]
    for ch, color in CHANNEL_COLORS.items():
        mask = df_test["channel"] == ch
        if mask.any():
            ax.scatter(df_test.loc[mask, p2_col], df_test.loc[mask, "residual"],
                       c=color, label=ch, alpha=0.8, s=50)
    ax.axhline(0, color="k", ls="--")
    ax.axhspan(-0.3, 0.3, alpha=0.2, color="gray")
    ax.set_xlabel(p2_label)
    ax.set_ylabel("Residual (dex)")
    ax.set_title(f"Residuals vs {p2_label} — Mean={mean_res:.4f}, Std={std_res:.4f}")
    ax.legend()

    ax = axes[2]
    for ch, color in CHANNEL_COLORS.items():
        mask = df_test["channel"] == ch
        if mask.any():
            ax.scatter(df_test.loc[mask, "y_true"], df_test.loc[mask, "residual"],
                       c=color, label=ch, alpha=0.8, s=50)
    ax.axhline(0, color="k", ls="--")
    ax.axhspan(-0.3, 0.3, alpha=0.2, color="gray")
    ax.set_xlabel(f"True {rate_lbl}")
    ax.set_ylabel("Residual (dex)")
    ax.set_title(f"Residuals vs true — Mean={mean_res:.4f}, Std={std_res:.4f}")
    ax.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "01b_residuals.png", dpi=300, bbox_inches="tight")
    plt.close()

    # -------------------------------------------------------------------------
    # 2. INTERPOLATION QUALITY  (SSPC: sfr_a × mu0 grid; Zenodo: chi_b × alpha_CE)
    # -------------------------------------------------------------------------

    # Pick the first channel that has data for the heatmap
    heatmap_channel = None
    for _ch in ["CE", "SMT", "CHE", "GC", "NSC"]:
        if len(df[df["channel"] == _ch]) > 0:
            heatmap_channel = _ch
            break

    ch_rows = df[df["channel"] == heatmap_channel].copy() if heatmap_channel else pd.DataFrame()

    if sspc:
        sfra_range = (float(df[p1_col].min()), float(df[p1_col].max()))
        mu0_range  = (float(df[p2_col].min()), float(df[p2_col].max()))
        p1_grid = np.linspace(*sfra_range, 20)
        p2_grid = np.linspace(*mu0_range,  20)
        ch_id = int(ch_rows["channel_id"].iloc[0]) if len(ch_rows) > 0 else 0
        lam_tmpl = ch_rows.iloc[0][lambda_cols].values.astype(np.float32) if len(ch_rows) > 0 else np.zeros(len(lambda_cols), dtype=np.float32)

        P1g, P2g = np.meshgrid(p1_grid, p2_grid)
        X_grid = np.array(
            [_sspc_lambda_vec(p1, p2, ch_id, lam_tmpl, sfra_range, mu0_range)
             for p1, p2 in zip(P1g.ravel(), P2g.ravel())],
            dtype=np.float32,
        )
    else:
        chi_range   = (float(df[p1_col].min()), float(df[p1_col].max()))
        alpha_range = (float(df[p2_col].min()), float(df[p2_col].max()))
        p1_grid = np.linspace(*chi_range,   20)
        p2_grid = np.linspace(*alpha_range, 20)
        lam_tmpl = ch_rows.iloc[0][lambda_cols].values.astype(np.float32) if len(ch_rows) > 0 else np.zeros(len(lambda_cols), dtype=np.float32)

        P1g, P2g = np.meshgrid(p1_grid, p2_grid)
        X_grid = np.array(
            [_ce_lambda_vec(p1, p2, lam_tmpl, chi_range, alpha_range)
             for p1, p2 in zip(P1g.ravel(), P2g.ravel())],
            dtype=np.float32,
        )

    with torch.no_grad():
        pred_grid = model(torch.from_numpy(X_grid).float().to(device)).cpu().numpy()
    Z = pred_grid.reshape(P1g.shape)

    # 2a) heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.pcolormesh(p1_grid, p2_grid, Z, cmap="viridis", shading="auto")
    plt.colorbar(im, ax=ax, label=f"Predicted {rate_lbl}")

    train_ch = ch_rows[ch_rows.index.isin(train_idx)]
    test_ch  = ch_rows[ch_rows.index.isin(test_idx)]
    if len(train_ch) > 0:
        ax.scatter(train_ch[p1_col], train_ch[p2_col], c="white", s=100, marker="o",
                   edgecolors="black", linewidths=2, label="Train", zorder=5)
    if len(test_ch) > 0:
        ax.scatter(test_ch[p1_col], test_ch[p2_col], c="yellow", s=120, marker="x",
                   linewidths=3, label="Test", zorder=5)

    ax.set_xlabel(p1_label)
    ax.set_ylabel(p2_label)
    ax.set_title(f"{heatmap_channel}: Model interpolation on 20×20 grid")
    ax.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "02a_ce_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 2b) 1D slices: predicted log_rate vs p2 for three p1 values
    fig, ax = plt.subplots(figsize=(8, 5))
    p2_slice = np.linspace(p2_grid[0], p2_grid[-1], 50)
    p1_vals = np.quantile(df[p1_col].values, [0.25, 0.5, 0.75])
    for p1_val, color in zip(p1_vals, ["C0", "C1", "C2"]):
        if sspc:
            X_slice = np.array(
                [_sspc_lambda_vec(p1_val, p2, ch_id, lam_tmpl, sfra_range, mu0_range)
                 for p2 in p2_slice],
                dtype=np.float32,
            )
        else:
            X_slice = np.array(
                [_ce_lambda_vec(p1_val, p2, lam_tmpl, chi_range, alpha_range)
                 for p2 in p2_slice],
                dtype=np.float32,
            )
        with torch.no_grad():
            pred_slice = model(torch.from_numpy(X_slice).float().to(device)).cpu().numpy()
        ax.plot(p2_slice, pred_slice, color=color,
                label=f"{p1_label}={p1_val:.4f}" if sspc else f"χ_b={p1_val:.2f}",
                lw=2)

    # Overlay true values for each p1 bucket
    for p1_val, color in zip(p1_vals, ["C0", "C1", "C2"]):
        closest = ch_rows.iloc[(ch_rows[p1_col] - p1_val).abs().argsort()[:4]]
        if len(closest) > 0:
            true_vals = np.log10(closest["sum_pdet"].values + 1e-300)
            ax.scatter(closest[p2_col], true_vals, c=color, s=80, edgecolors="black", zorder=5)

    ax.set_xlabel(p2_label)
    ax.set_ylabel(rate_lbl)
    ax.set_title(f"Does the model capture {p2_label} dependence? ({heatmap_channel})")
    ax.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "02b_ce_1d_slices.png", dpi=300, bbox_inches="tight")
    plt.close()

    # -------------------------------------------------------------------------
    # 3. PER-CHANNEL ANALYSIS
    # -------------------------------------------------------------------------

    for ch in ["CE", "CHE", "GC", "NSC", "SMT"]:
        ch_df = df_all[df_all["channel"] == ch]
        if len(ch_df) == 0:
            continue
        y_ch = ch_df["y_true"].values
        p_ch = ch_df["pred"].values
        ss_res = np.sum((y_ch - p_ch) ** 2)
        ss_tot = np.sum((y_ch - np.mean(y_ch)) ** 2)
        r2_ch = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        rmse_ch = float(np.sqrt(np.mean((y_ch - p_ch) ** 2)))
        metrics["r2_per_channel"][ch] = r2_ch
        metrics["rmse_per_channel"][ch] = rmse_ch

        fig, ax = plt.subplots(figsize=(10, 5))
        x_pos = np.arange(len(ch_df))
        width = 0.35
        p1_vals_ch = ch_df[p1_col].values
        sort_order = np.argsort(p1_vals_ch)
        ch_df_sorted = ch_df.iloc[sort_order]
        y_ch_s = ch_df_sorted["y_true"].values
        p_ch_s = ch_df_sorted["pred"].values
        p1_sorted = ch_df_sorted[p1_col].values
        ax.bar(x_pos - width / 2, y_ch_s, width, label="True", color="steelblue", alpha=0.8)
        ax.bar(x_pos + width / 2, p_ch_s, width, label="Predicted", color="coral", alpha=0.8)
        ax.set_xlabel(f"Grid point (by {p1_label})")
        ax.set_ylabel(rate_lbl)
        ax.set_title(f"{ch} — R²={r2_ch:.4f}, RMSE={rmse_ch:.4f}")
        ax.legend()
        ax.axhline(0, color="k", ls=":", alpha=0.5)
        fmt = ".4f" if sspc else ".1f"
        plt.xticks(x_pos, [f"{c:{fmt}}" for c in p1_sorted], rotation=45)
        plt.tight_layout()
        plt.savefig(plots_dir / f"03_per_channel_{ch}.png", dpi=300, bbox_inches="tight")
        plt.close()

    # -------------------------------------------------------------------------
    # 4. LEARNING CURVES
    # -------------------------------------------------------------------------

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(train_losses, label="Train loss", color="C0")
    ax.plot(val_losses, label="Val loss", color="C1")
    ax.axvline(best_epoch, color="red", ls="--", label=f"Early stop (epoch {best_epoch})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title(f"Learning curves — Stopped at epoch {best_epoch} with val loss = {best_val_loss:.6f}")
    ax.legend()
    ax.set_yscale("log")
    plt.tight_layout()
    plt.savefig(plots_dir / "04_learning_curves.png", dpi=300, bbox_inches="tight")
    plt.close()

    # -------------------------------------------------------------------------
    # 5. UNCERTAINTY QUANTIFICATION — skipped (no ensemble/std)
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    # 6. PHYSICAL CONSISTENCY CHECKS
    # -------------------------------------------------------------------------

    if sspc:
        sfra_lo, sfra_hi = SSPC_SFRA_RANGE
        mu0_lo,  mu0_hi  = SSPC_MU0_RANGE
        mid_sfra = float(np.median(df[p1_col]))
        mid_mu0  = float(np.median(df[p2_col]))

        # 6a) mu0 scan at median sfr_a, with OOD extension
        mu0_scan = np.linspace(max(0.0, mu0_lo * 0.5), mu0_hi * 1.5, 30)
        X_scan = np.array(
            [_sspc_lambda_vec(mid_sfra, m, ch_id, lam_tmpl, sfra_range, mu0_range)
             for m in mu0_scan],
            dtype=np.float32,
        )
        with torch.no_grad():
            pred_scan = model(torch.from_numpy(X_scan).float().to(device)).cpu().numpy()
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(mu0_scan, pred_scan, "o-", color="C0", lw=2, markersize=8)
        ax.axvline(mu0_lo, color="gray", ls=":", alpha=0.7, label="Training boundary")
        ax.axvline(mu0_hi, color="gray", ls=":", alpha=0.7)
        ax.axvspan(0.0, mu0_lo, alpha=0.15, color="red", label="OOD (low mu0)")
        ax.axvspan(mu0_hi, mu0_hi * 1.6, alpha=0.15, color="red", label="OOD (high mu0)")
        ax.set_xlabel("mu0 (mean metallicity at z=0)")
        ax.set_ylabel(f"Predicted {rate_lbl}")
        ax.set_title(f"mu0 scan at sfr_a={mid_sfra:.4f} ({heatmap_channel})")
        ax.legend()
        plt.tight_layout()
        plt.savefig(plots_dir / "06a_ce_efficiency_scan.png", dpi=300, bbox_inches="tight")
        plt.close()

        # 6b) sfr_a scan at median mu0, with OOD extension
        sfra_scan = np.linspace(max(0.001, sfra_lo * 0.5), sfra_hi * 1.5, 30)
        X_spin = np.array(
            [_sspc_lambda_vec(s, mid_mu0, ch_id, lam_tmpl, sfra_range, mu0_range)
             for s in sfra_scan],
            dtype=np.float32,
        )
        with torch.no_grad():
            pred_spin = model(torch.from_numpy(X_spin).float().to(device)).cpu().numpy()
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(sfra_scan, pred_spin, "o-", color="C0", lw=2, markersize=6)
        ax.axvline(sfra_lo, color="gray", ls=":", alpha=0.7, label="Training boundary")
        ax.axvline(sfra_hi, color="gray", ls=":", alpha=0.7)
        ax.axvspan(0.0, sfra_lo, alpha=0.15, color="red", label="OOD (low sfr_a)")
        ax.axvspan(sfra_hi, sfra_hi * 1.6, alpha=0.15, color="red", label="OOD (high sfr_a)")
        ax.set_xlabel("sfr_a (Madau-Dickinson amplitude)")
        ax.set_ylabel(f"Predicted {rate_lbl}")
        ax.set_title(f"sfr_a scan at mu0={mid_mu0:.4f} ({heatmap_channel})")
        ax.legend()
        plt.tight_layout()
        plt.savefig(plots_dir / "06b_spin_scan.png", dpi=300, bbox_inches="tight")
        plt.close()

    else:
        # Legacy Zenodo consistency checks
        alpha_scan = np.array([0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0])
        X_scan = np.array(
            [_ce_lambda_vec(0.2, a, lam_tmpl, chi_range, alpha_range) for a in alpha_scan],
            dtype=np.float32,
        )
        with torch.no_grad():
            pred_scan = model(torch.from_numpy(X_scan).float().to(device)).cpu().numpy()
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(alpha_scan, pred_scan, "o-", color="C0", lw=2, markersize=10)
        ax.axvline(0.2, color="gray", ls=":", alpha=0.7); ax.axvline(5.0, color="gray", ls=":", alpha=0.7)
        ax.axvspan(0.0, 0.2, alpha=0.15, color="red", label="OOD")
        ax.axvspan(5.0, 11, alpha=0.15, color="red")
        ax.set_xlabel("alpha_CE"); ax.set_ylabel(f"Predicted {rate_lbl}")
        ax.set_title("CE efficiency scan (chi_b=0.2)")
        ax.legend(); plt.tight_layout()
        plt.savefig(plots_dir / "06a_ce_efficiency_scan.png", dpi=300, bbox_inches="tight"); plt.close()

        chi_scan = np.linspace(0.0, 1.0, 25)
        X_spin = np.array(
            [_ce_lambda_vec(c, 1.0, lam_tmpl, chi_range, alpha_range) for c in chi_scan],
            dtype=np.float32,
        )
        with torch.no_grad():
            pred_spin = model(torch.from_numpy(X_spin).float().to(device)).cpu().numpy()
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(chi_scan, pred_spin, "o-", color="C0", lw=2, markersize=6)
        ax.axvline(0.5, color="gray", ls=":", alpha=0.7, label="Training boundary")
        ax.axvspan(0.5, 1.05, alpha=0.15, color="red", label="OOD")
        ax.set_xlabel("chi_b"); ax.set_ylabel(f"Predicted {rate_lbl}")
        ax.set_title("Spin scan (alpha_CE=1.0, CE)")
        ax.legend(); plt.tight_layout()
        plt.savefig(plots_dir / "06b_spin_scan.png", dpi=300, bbox_inches="tight"); plt.close()

    return metrics


def _write_plots_readme(plots_dir: Path, metrics: dict, r2: float, mse: float) -> None:
    """Write plots/rate_network/README.md with plot descriptions and key metrics."""
    rmse = float(np.sqrt(mse))
    lines = [
        "# Rate Network Validation Plots",
        "",
        "## Key Metrics",
        f"- **Overall R² (test set):** {r2:.4f}",
        f"- **Overall RMSE (test set):** {rmse:.4f}",
        "",
        "## Per-Channel Metrics",
    ]
    for ch in ["CE", "CHE", "GC", "NSC", "SMT"]:
        if ch in metrics.get("r2_per_channel", {}):
            r2_ch = metrics["r2_per_channel"][ch]
            rmse_ch = metrics["rmse_per_channel"][ch]
            lines.append(f"- **{ch}:** R² = {r2_ch:.4f}, RMSE = {rmse_ch:.4f}")

    lines.extend([
        "",
        "## Plot Descriptions",
        "",
        "### 1. Prediction Quality",
        "- **01a_predicted_vs_true.png** — Predicted vs true log10(sum_pdet) scatter, colored by channel, with ±0.5 dex band and outlier annotations.",
        "- **01b_residuals.png** — Residuals vs chi_b, vs alpha_CE (CE only), and vs true value (heteroscedasticity check).",
        "",
        "### 2. Interpolation Quality",
        "- **02a_ce_heatmap.png** — CE channel: model evaluated on 20×20 dense grid; train (circles) and test (X) points overlaid.",
        "- **02b_ce_1d_slices.png** — 1D slices: predicted log_rate vs alpha_CE for chi_b = 0.1, 0.2, 0.5 with true values overlaid.",
        "",
        "### 3. Per-Channel Analysis",
        "- **03_per_channel_CE.png** — CE: true vs predicted bar plot by chi_b.",
        "- **03_per_channel_CHE.png** — CHE: true vs predicted bar plot.",
        "- **03_per_channel_GC.png** — GC: true vs predicted bar plot.",
        "- **03_per_channel_NSC.png** — NSC: true vs predicted bar plot.",
        "- **03_per_channel_SMT.png** — SMT: true vs predicted bar plot.",
        "",
        "### 4. Learning Curves",
        "- **04_learning_curves.png** — Train and validation loss vs epoch; vertical line marks early stopping.",
        "",
        "### 5. Uncertainty Quantification",
        "- Skipped (no ensemble/uncertainty estimates available).",
        "",
        "### 6. Physical Consistency",
        "- **06a_ce_efficiency_scan.png** — CE: alpha_CE scan (0.1–10) at chi_b=0.2; OOD regions shaded.",
        "- **06b_spin_scan.png** — CE: chi_b scan (0–1) at alpha_CE=1.0; OOD region (chi_b>0.5) shaded.",
        "",
    ])
    readme_path = plots_dir / "README.md"
    readme_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"   Saved: {readme_path}")


def predict_rate(lambda_vec: np.ndarray) -> float:
    """
    Load best model (or GP if better) and return predicted log10(sum_pdet).

    lambda_vec: shape (7,) or (1, 7)
    Returns: scalar float
    """
    lambda_vec = np.asarray(lambda_vec, dtype=np.float32)
    if lambda_vec.ndim == 1:
        lambda_vec = lambda_vec.reshape(1, -1)
    ckpt_dir = _find_work_dir() / "checkpoints"
    config_path = ckpt_dir / "rate_network_config.json"
    best_path = ckpt_dir / "rate_network_best.pt"
    gp_path = ckpt_dir / "gp_rate_baseline.pkl"

    use_gp = False
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        use_gp = config.get("use_gp_baseline", False)

    # Ensure models can be imported
    parent = ckpt_dir.parent
    if str(parent) not in sys.path:
        sys.path.insert(0, str(parent))

    if use_gp and gp_path.exists():
        import joblib
        gp = joblib.load(gp_path)
        return float(gp.predict(lambda_vec)[0])
    elif best_path.exists():
        from models.rate_network import RateNetwork
        with open(config_path) as f:
            config = json.load(f)
        if lambda_vec.shape[1] != int(config["input_dim"]):
            raise ValueError(f"Expected lambda dim {config['input_dim']}, got {lambda_vec.shape[1]}")
        model = RateNetwork(
            input_dim=config["input_dim"],
            hidden_dims=tuple(config["hidden_dims"]),
        )
        model.load_state_dict(torch.load(best_path, map_location="cpu"))
        model.eval()
        with torch.no_grad():
            x = torch.from_numpy(lambda_vec).float()
            return float(model(x).item())
    else:
        raise FileNotFoundError(
            f"No checkpoint found in {ckpt_dir}. Run 03_rate_network.py first."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train rate network")
    parser.add_argument("--hyperparam-csv", type=Path, default=HYPERPARAM_CSV)
    parser.add_argument("--splits-json", type=Path, default=SPLITS_JSON)
    parser.add_argument("--checkpoint-dir", type=Path, default=CHECKPOINT_DIR)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)

    work_dir = _find_work_dir()
    hp_csv = work_dir / args.hyperparam_csv.name if not args.hyperparam_csv.is_absolute() else args.hyperparam_csv
    splits_path = work_dir / args.splits_json.name if not args.splits_json.is_absolute() else args.splits_json
    ckpt_dir = work_dir / args.checkpoint_dir.name if not args.checkpoint_dir.is_absolute() else args.checkpoint_dir
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if not hp_csv.exists():
        raise FileNotFoundError(f"Run 02_build_dataset.py first. Missing: {hp_csv}")
    if not splits_path.exists():
        raise FileNotFoundError(f"Missing: {splits_path}")

    X, y, train_idx, val_idx, test_idx, df, lambda_cols = load_data(hp_csv, splits_path)
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    print("1. Training RateNetwork...")
    model, config, train_losses, val_losses, best_epoch, best_val_loss = train_rate_network(
        X_train, y_train, X_val, y_val,
        epochs=args.epochs, patience=args.patience, device=args.device, input_dim=X.shape[1],
    )
    print(f"   Trainable params: {config['n_params']}")

    print("2. Training GP baseline (on train+val)...")
    X_trainval = np.concatenate([X_train, X_val], axis=0)
    y_trainval = np.concatenate([y_train, y_val], axis=0)
    gp = train_gp_baseline(X_trainval, y_trainval)

    print("3. Evaluating on test set...")
    pred_nn, r2_nn, mse_nn = evaluate(model, X_test, y_test, is_gp=False)
    pred_gp, r2_gp, mse_gp = evaluate(gp, X_test, y_test, is_gp=True)
    print(f"   NN: R²={r2_nn:.4f}, MSE={mse_nn:.6f}")
    print(f"   GP: R²={r2_gp:.4f}, MSE={mse_gp:.6f}")

    print("   Generating evaluation plots...")
    plot_evaluation(df, test_idx, pred_nn, pred_gp, y_test, model, gp, ckpt_dir, lambda_cols)

    # Detailed validation plots (use NN predictions; GP is baseline only)
    plots_dir = work_dir / "plots" / "rate_network"
    device = next(model.parameters()).device
    with torch.no_grad():
        pred_full = model(torch.from_numpy(X).float().to(device)).cpu().numpy()
    metrics = plot_detailed_validation(
        df, train_idx, val_idx, test_idx, X, y, pred_full,
        model, train_losses, val_losses, best_epoch, best_val_loss,
        r2_nn, plots_dir, lambda_cols,
    )
    _write_plots_readme(plots_dir, metrics, r2_nn, mse_nn)

    use_gp = mse_gp < mse_nn
    config["use_gp_baseline"] = use_gp
    if use_gp:
        print("   Using GP baseline (better than NN)")
    else:
        print("   Using NN (better than GP)")

    print("4. Saving checkpoints...")
    torch.save(model.state_dict(), ckpt_dir / "rate_network_best.pt")
    with open(ckpt_dir / "rate_network_config.json", "w") as f:
        json.dump(config, f, indent=2)
    import joblib
    joblib.dump(gp, ckpt_dir / "gp_rate_baseline.pkl")
    print(f"   Saved to {ckpt_dir}")

    print("5. Sanity check: rate sensitivity to primary parameters...")
    sspc_check = _is_sspc(df)
    p1_col_chk = "chi_b"
    if sspc_check:
        p1_vals_sorted = sorted(df[p1_col_chk].unique())
        if len(p1_vals_sorted) >= 2:
            lo_val, hi_val = p1_vals_sorted[0], p1_vals_sorted[-1]
            lam_lo  = df[df[p1_col_chk] == lo_val].iloc[0][lambda_cols].values.astype(np.float32).reshape(1, -1)
            lam_hi  = df[df[p1_col_chk] == hi_val].iloc[0][lambda_cols].values.astype(np.float32).reshape(1, -1)
            r_lo = predict_rate(lam_lo) if not use_gp else float(gp.predict(lam_lo)[0])
            r_hi = predict_rate(lam_hi) if not use_gp else float(gp.predict(lam_hi)[0])
            print(f"   sfr_a={lo_val:.4f} pred: {r_lo:.4f}, sfr_a={hi_val:.4f} pred: {r_hi:.4f}")
            print(f"   Rate changes with sfr_a: {abs(r_hi - r_lo) > 0.01}")
    else:
        ce_df = df[df["channel"] == "CE"]
        ext_lo = ce_df[ce_df["alpha_CE"] == 0.2]
        mid    = ce_df[ce_df["alpha_CE"].isin([1.0, 2.0])]
        if len(ext_lo) and len(mid):
            lam_ext = ext_lo.iloc[0][lambda_cols].values.astype(np.float32).reshape(1, -1)
            lam_mid = mid.iloc[0][lambda_cols].values.astype(np.float32).reshape(1, -1)
            r_ext = predict_rate(lam_ext) if not use_gp else float(gp.predict(lam_ext)[0])
            r_mid = predict_rate(lam_mid) if not use_gp else float(gp.predict(lam_mid)[0])
            print(f"   alpha_CE=0.2 pred: {r_ext:.4f}, alpha_CE=1.0 pred: {r_mid:.4f}")
            print(f"   Extreme < middle (expected): {r_ext < r_mid}")

    print("6. predict_rate() test...")
    lam = X[0:1]
    out = predict_rate(lam)
    print(f"   predict_rate(X[0]) = {out:.4f} (scalar: {np.isscalar(out) or out.ndim==0})")


if __name__ == "__main__":
    main()
