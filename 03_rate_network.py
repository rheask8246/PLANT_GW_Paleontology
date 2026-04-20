#!/usr/bin/env python3
"""
Rate network for SSPC data.

Predicts log10(Σ det_weight) — total observer-frame detection rate — as a
function of all 9 SSPC hyperparameters + channel:

  [CE_ind, CHE_ind, SMT_ind,
   sfr_a, sfr_b, sfr_c, sfr_d,   (Madau-Dickinson SFR shape)
   mu0, muz, sigma0, sigmaz, alpha_skew]   (metallicity distribution)

sfr_a and mu0 are the primary grid axes (set by the user); the other 7 are
nuisance parameters randomly drawn per grid point during data generation,
giving broad coverage of the full parameter space.

Input:  hyperparam_table_encoded.csv  +  splits.json  (from 02_build_dataset.py)
Output: checkpoints/rate_network_best.pt  +  plots/rate_network/<timestamp>/
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent

HYPERPARAM_CSV = _HERE / "hyperparam_table_encoded.csv"
SPLITS_JSON    = _HERE / "splits.json"
CHECKPOINT_DIR = _HERE / "checkpoints"
PLOTS_BASE     = _HERE / "plots" / "rate_network"

LOG_RATE_FLOOR = -5.0

# 9 SSPC parameter columns (mean values from integration)
SSPC_PARAM_COLS = [
    "sspc_sfr_a_mean", "sspc_sfr_b_mean", "sspc_sfr_c_mean", "sspc_sfr_d_mean",
    "sspc_mu0_mean", "sspc_muz_mean", "sspc_sigma0_mean",
    "sspc_sigmaz_mean", "sspc_alpha_skew_mean",
]
SSPC_PARAM_LABELS = [
    "sfr_a", "sfr_b", "sfr_c", "sfr_d",
    "mu0", "muz", "σ₀", "σz", "α_skew",
]

CHANNEL_ORDER  = ["CE", "CHE", "SMT"]
CHANNEL_COLORS = {"CE": "#e6194b", "CHE": "#4363d8", "SMT": "#3cb44b"}

# Median nuisance param values (for default predict_rate calls)
_PARAM_DEFAULTS = {
    "sfr_a": 0.0215, "sfr_b": 2.34, "sfr_c": 3.78, "sfr_d": 5.05,
    "mu0": 0.0350, "muz": -0.228, "sigma0": 0.414,
    "sigmaz": 0.007, "alpha_skew": 0.002,
}


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def build_features(df: pd.DataFrame):
    """
    Return (X, y, norm_params) where:
      X : (N, 12) float32 — [CE, CHE, SMT, sfr_a_n, sfr_b_n, ..., alpha_skew_n]
      y : (N,)    float64 — log10(Σ det_weight), floor at LOG_RATE_FLOOR
      norm_params: dict of {col: (min, max)} for inversion / inference
    """
    missing = [c for c in SSPC_PARAM_COLS if c not in df.columns]
    if missing:
        sys.exit(f"ERROR: Missing columns in CSV: {missing}\nRun 02_build_dataset.py first.")

    X_raw = df[SSPC_PARAM_COLS].values.astype(np.float64)
    norm_params = {col: (float(X_raw[:, i].min()), float(X_raw[:, i].max()))
                   for i, col in enumerate(SSPC_PARAM_COLS)}

    X_norm = np.zeros_like(X_raw, dtype=np.float32)
    for i, col in enumerate(SSPC_PARAM_COLS):
        lo, hi = norm_params[col]
        X_norm[:, i] = (X_raw[:, i] - lo) / max(hi - lo, 1e-12)

    ch = df["channel"].values
    ce  = (ch == "CE" ).astype(np.float32)
    che = (ch == "CHE").astype(np.float32)
    smt = (ch == "SMT").astype(np.float32)

    X = np.column_stack([ce, che, smt, X_norm])

    y_raw = np.log10(df["sum_pdet"].values.astype(np.float64).clip(1e-300))
    n_clip = int((y_raw < LOG_RATE_FLOOR).sum())
    if n_clip:
        print(f"  [rate] Clipping {n_clip} near-zero grid pts (min={y_raw.min():.1f}) to floor={LOG_RATE_FLOOR}")
    y = np.clip(y_raw, LOG_RATE_FLOOR, None)

    return X, y, norm_params


def feature_vec(channel: str, norm_params: dict,
                sfr_a: float, sfr_b: float, sfr_c: float, sfr_d: float,
                mu0: float, muz: float, sigma0: float,
                sigmaz: float, alpha_skew: float) -> np.ndarray:
    """Build a single 12-dim feature from physical parameter values."""
    raw = [sfr_a, sfr_b, sfr_c, sfr_d, mu0, muz, sigma0, sigmaz, alpha_skew]
    norm = []
    for col, val in zip(SSPC_PARAM_COLS, raw):
        lo, hi = norm_params[col]
        norm.append(float((val - lo) / max(hi - lo, 1e-12)))
    onehot = [float(channel == "CE"), float(channel == "CHE"), float(channel == "SMT")]
    return np.array(onehot + norm, dtype=np.float32)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class RateNet(nn.Module):
    """MLP: 12 → 128 → 64 → 32 → 1  with LayerNorm + GELU."""

    def __init__(self, input_dim: int = 12, hidden: tuple = (128, 64, 32)):
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

    def forward(self, x):
        return self.net(x).squeeze(-1)


class NormalizedNet(nn.Module):
    """Wraps RateNet; forward() returns predictions in log10 scale."""

    def __init__(self, base: RateNet, y_mean: float, y_std: float):
        super().__init__()
        self.base = base
        self.register_buffer("y_mean", torch.tensor(float(y_mean)))
        self.register_buffer("y_std",  torch.tensor(float(y_std)))

    def forward(self, x):
        return self.base(x) * self.y_std + self.y_mean


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(X_tr, y_tr, X_va, y_va,
          epochs=3000, patience=300, lr=1e-3):
    y_mean, y_std = float(y_tr.mean()), float(y_tr.std()) + 1e-8
    y_tr_n  = (y_tr  - y_mean) / y_std
    y_va_n  = (y_va  - y_mean) / y_std

    Xtr = torch.from_numpy(X_tr).float()
    ytr = torch.from_numpy(y_tr_n).float()
    Xv  = torch.from_numpy(X_va).float()
    yv  = torch.from_numpy(y_va_n).float()

    base  = RateNet(input_dim=X_tr.shape[1])
    opt   = Adam(base.parameters(), lr=lr, weight_decay=1e-4)
    sched = ReduceLROnPlateau(opt, patience=patience // 4, factor=0.5, min_lr=1e-5)
    loss_fn = nn.HuberLoss(delta=1.5)

    best_val, best_state, best_ep = np.inf, None, 0
    tloss, vloss, no_imp = [], [], 0

    for ep in range(epochs):
        base.train()
        opt.zero_grad()
        l = loss_fn(base(Xtr), ytr)
        l.backward()
        torch.nn.utils.clip_grad_norm_(base.parameters(), 5.0)
        opt.step()

        base.eval()
        with torch.no_grad():
            vl = float(loss_fn(base(Xv), yv))
        sched.step(vl)
        tloss.append(float(l.item()))
        vloss.append(vl)

        if vl < best_val - 1e-6:
            best_val, best_state, best_ep = vl, {k: v.clone() for k, v in base.state_dict().items()}, ep
            no_imp = 0
        else:
            no_imp += 1
        if no_imp >= patience:
            break

    base.load_state_dict(best_state)
    print(f"  Stopped at epoch {best_ep}, best val loss = {best_val:.5f}")
    return NormalizedNet(base, y_mean, y_std), tloss, vloss, best_ep


def fit_gp(X_tr, y_tr):
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, WhiteKernel
    import warnings
    gp = GaussianProcessRegressor(
        kernel=Matern(nu=2.5) + WhiteKernel(noise_level=0.1),
        alpha=1e-6, normalize_y=True, random_state=42,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gp.fit(X_tr, y_tr)
    return gp


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def r2(y_true, y_pred):
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


@torch.no_grad()
def predict(model, X):
    model.eval()
    return model(torch.from_numpy(X.astype(np.float32)).float()).numpy()


def _choose_model(model, gp, X_te, y_te):
    """Return whichever of MLP/GP has higher test R², plus a label."""
    pnn, pgp = predict(model, X_te), gp.predict(X_te)
    r_nn, r_gp = r2(y_te, pnn), r2(y_te, pgp)
    use_gp = r_gp > r_nn
    return (pgp if use_gp else pnn), r_nn, r_gp, use_gp


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_scatter(model, gp, X, y, splits, df, plots_dir):
    """
    Plot 1: predicted vs true scatter per channel on test set.
    Shows both MLP and GP R² values; highlights the better model.
    """
    te = splits["test"]
    X_te, y_te = X[te], y[te]
    ch_te = df["channel"].iloc[te].values

    pred_nn = predict(model, X_te)
    pred_gp = gp.predict(X_te)
    r_nn = r2(y_te, pred_nn)
    r_gp = r2(y_te, pred_gp)
    use_gp = r_gp > r_nn
    pred = pred_gp if use_gp else pred_nn
    tag = "GP" if use_gp else "MLP"

    channels = [ch for ch in CHANNEL_ORDER if ch in ch_te]
    fig, axes = plt.subplots(1, len(channels), figsize=(5 * len(channels), 5), squeeze=False)
    all_v = np.concatenate([y_te, pred])
    lo, hi = all_v.min() - 0.3, all_v.max() + 0.3

    for ax, ch in zip(axes[0], channels):
        m = ch_te == ch
        r2_ch = r2(y_te[m], pred[m]) if m.sum() > 1 else float("nan")
        ax.scatter(y_te[m], pred[m], color=CHANNEL_COLORS.get(ch, "gray"),
                   s=60, alpha=0.85, edgecolors="k", lw=0.5)
        ax.plot([lo, hi], [lo, hi], "k--", lw=1.2)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xlabel(r"True $\log_{10}(\Sigma\dot{N})$")
        ax.set_ylabel(r"Predicted $\log_{10}(\Sigma\dot{N})$")
        ax.set_title(f"{ch}  [{tag}]  R²={r2_ch:.3f}")

    overall_title = (f"Rate network: predicted vs true (test set)\n"
                     f"Overall — MLP R²={r_nn:.3f}  GP R²={r_gp:.3f}  "
                     f"(using {tag})")
    fig.suptitle(overall_title, fontsize=12)
    plt.tight_layout()
    out = plots_dir / "01_scatter.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")
    return {"r2_nn": r_nn, "r2_gp": r_gp, "use_gp": use_gp}


def plot_param_sensitivity(model, df, norm_params, plots_dir):
    """
    Plot 2: 1D sensitivity of predicted log10(rate) vs each SSPC parameter.
    All other parameters held at their data medians.
    One row per channel, one column per parameter (3 × 9 axes, grouped as 3×3 pages).
    """
    channels = [ch for ch in CHANNEL_ORDER if (df["channel"] == ch).any()]
    medians = {col: float(df[col].median()) for col in SSPC_PARAM_COLS}
    n_params = len(SSPC_PARAM_COLS)

    fig, axes = plt.subplots(len(channels), n_params,
                             figsize=(3.0 * n_params, 3.5 * len(channels)),
                             squeeze=False)

    param_keys = ["sfr_a", "sfr_b", "sfr_c", "sfr_d",
                  "mu0", "muz", "sigma0", "sigmaz", "alpha_skew"]
    col_to_key = dict(zip(SSPC_PARAM_COLS, param_keys))

    for row, ch in enumerate(channels):
        for col, (sspc_col, label) in enumerate(zip(SSPC_PARAM_COLS, SSPC_PARAM_LABELS)):
            lo_p, hi_p = norm_params[sspc_col]
            scan = np.linspace(lo_p, hi_p, 50)
            pts = []
            for v in scan:
                kw = {col_to_key[c]: medians[c] for c in SSPC_PARAM_COLS}
                kw[col_to_key[sspc_col]] = v
                pts.append(feature_vec(ch, norm_params, **kw))
            pts = np.stack(pts, dtype=np.float32)
            with torch.no_grad():
                pred = model(torch.from_numpy(pts).float()).numpy()

            ax = axes[row, col]
            ax.plot(scan, pred, color=CHANNEL_COLORS.get(ch, "gray"), lw=2)
            # Overlay actual data for this channel
            ch_df = df[df["channel"] == ch]
            y_ch = np.clip(np.log10(ch_df["sum_pdet"].clip(1e-300)), LOG_RATE_FLOOR, None)
            ax.scatter(ch_df[sspc_col], y_ch.values,
                       color=CHANNEL_COLORS.get(ch, "gray"), s=12, alpha=0.5)
            ax.set_xlabel(label, fontsize=9)
            if col == 0:
                ax.set_ylabel(f"{ch}\n" + r"$\log_{10}(\Sigma\dot{N})$", fontsize=9)
            ax.tick_params(labelsize=8)

    fig.suptitle("Rate sensitivity: model prediction vs each parameter\n"
                 "(other parameters at data median; dots = data)", fontsize=12)
    plt.tight_layout()
    out = plots_dir / "02_param_sensitivity.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def plot_sfra_mu0_heatmap(model, df, norm_params, plots_dir):
    """
    Plot 3: 2D heatmap of predicted log10(rate) on the (sfr_a, mu0) primary grid,
    with other parameters fixed at their data medians.
    One subplot per channel.
    """
    channels = [ch for ch in CHANNEL_ORDER if (df["channel"] == ch).any()]
    medians  = {col: float(df[col].median()) for col in SSPC_PARAM_COLS}

    sfra_u = np.sort(df["sspc_sfr_a_mean"].unique())
    mu0_u  = np.sort(df["sspc_mu0_mean"].unique())
    sfra_scan = np.linspace(sfra_u.min(), sfra_u.max(), 25)
    mu0_scan  = np.linspace(mu0_u.min(),  mu0_u.max(),  25)

    S, M = np.meshgrid(sfra_scan, mu0_scan)

    fig, axes = plt.subplots(1, len(channels), figsize=(6 * len(channels), 5), squeeze=False)

    # Global colour range from actual data
    y_all = np.clip(np.log10(df["sum_pdet"].clip(1e-300)), LOG_RATE_FLOOR, None)
    vmin, vmax = float(y_all.min()), float(y_all.max())

    for ax, ch in zip(axes[0], channels):
        kw_base = {k: medians[c] for k, c in
                   zip(["sfr_b", "sfr_c", "sfr_d", "muz", "sigma0", "sigmaz", "alpha_skew"],
                       ["sspc_sfr_b_mean", "sspc_sfr_c_mean", "sspc_sfr_d_mean",
                        "sspc_muz_mean", "sspc_sigma0_mean", "sspc_sigmaz_mean",
                        "sspc_alpha_skew_mean"])}
        pts = [feature_vec(ch, norm_params, sfr_a=s, mu0=m, **kw_base)
               for s, m in zip(S.ravel(), M.ravel())]
        pts = np.stack(pts, dtype=np.float32)
        with torch.no_grad():
            Z = model(torch.from_numpy(pts).float()).numpy().reshape(S.shape)

        im = ax.contourf(S, M, Z, levels=20, cmap="viridis", vmin=vmin, vmax=vmax)
        plt.colorbar(im, ax=ax, label=r"$\log_{10}(\Sigma\dot{N})$")

        # Overlay true data points (coloured by true rate)
        ch_df = df[df["channel"] == ch]
        y_ch  = np.clip(np.log10(ch_df["sum_pdet"].clip(1e-300)), LOG_RATE_FLOOR, None)
        ax.scatter(ch_df["sspc_sfr_a_mean"], ch_df["sspc_mu0_mean"],
                   c=y_ch.values, cmap="viridis", vmin=vmin, vmax=vmax,
                   s=70, edgecolors="white", lw=0.8, zorder=5)

        ax.set_xlabel("sfr_a  (Madau-Dickinson amplitude)")
        ax.set_ylabel("mu0  (mean metallicity at z=0)")
        ax.set_title(f"{ch}  —  nuisance params at median")

    fig.suptitle(r"Predicted $\log_{10}(\Sigma\dot{N})$ on (sfr_a, mu0) plane"
                 "\n(dots = data; nuisance params held at median)", fontsize=12)
    plt.tight_layout()
    out = plots_dir / "03_sfra_mu0_heatmap.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def plot_learning_curves(tloss, vloss, best_ep, plots_dir):
    """Plot 4: training and validation loss curves."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogy(tloss, label="Train", color="C0")
    ax.semilogy(vloss, label="Val",   color="C1")
    ax.axvline(best_ep, color="red", ls="--", label=f"Best (ep {best_ep})")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Huber loss (normalised)")
    ax.set_title("Rate network learning curves")
    ax.legend()
    plt.tight_layout()
    out = plots_dir / "04_learning_curves.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Checkpoint I/O
# ---------------------------------------------------------------------------

def save_checkpoint(model: NormalizedNet, norm_params: dict, ckpt_dir: Path):
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict":  model.state_dict(),
        "y_mean":      float(model.y_mean),
        "y_std":       float(model.y_std),
        "input_dim":   model.base.net[0].in_features,
        "hidden_dims": [l.out_features for l in model.base.net if isinstance(l, nn.Linear)][:-1],
        "norm_params": norm_params,
    }, ckpt_dir / "rate_network_best.pt")
    with open(ckpt_dir / "rate_network_config.json", "w") as f:
        json.dump({"norm_params": norm_params,
                   "sspc_param_cols": SSPC_PARAM_COLS,
                   "log_rate_floor": LOG_RATE_FLOOR,
                   "feature_order": ["CE","CHE","SMT"] + SSPC_PARAM_COLS}, f, indent=2)
    print(f"  Checkpoint → {ckpt_dir / 'rate_network_best.pt'}")


def save_gp(gp, ckpt_dir: Path):
    with open(ckpt_dir / "gp_rate_baseline.pkl", "wb") as f:
        pickle.dump(gp, f)


# ---------------------------------------------------------------------------
# Public inference API
# ---------------------------------------------------------------------------

_cached_model: NormalizedNet | None = None
_cached_norm:  dict | None = None


def predict_rate(
    channel: str,
    sfr_a: float = _PARAM_DEFAULTS["sfr_a"],
    mu0:   float = _PARAM_DEFAULTS["mu0"],
    sfr_b: float = _PARAM_DEFAULTS["sfr_b"],
    sfr_c: float = _PARAM_DEFAULTS["sfr_c"],
    sfr_d: float = _PARAM_DEFAULTS["sfr_d"],
    muz:   float = _PARAM_DEFAULTS["muz"],
    sigma0: float = _PARAM_DEFAULTS["sigma0"],
    sigmaz: float = _PARAM_DEFAULTS["sigmaz"],
    alpha_skew: float = _PARAM_DEFAULTS["alpha_skew"],
    ckpt_dir: Path | None = None,
) -> float:
    """
    Return predicted log10(Σ det_weight) for the given SSPC parameters.
    Nuisance parameters default to their data-wide medians.
    Loads checkpoint on first call and caches.
    """
    global _cached_model, _cached_norm
    if _cached_model is None:
        if ckpt_dir is None:
            ckpt_dir = _HERE / "checkpoints"
        ckpt = torch.load(ckpt_dir / "rate_network_best.pt", weights_only=False)
        norm  = ckpt["norm_params"]
        base  = RateNet(input_dim=ckpt["input_dim"],
                        hidden=tuple(ckpt["hidden_dims"]))
        wrapped = NormalizedNet(base, ckpt["y_mean"], ckpt["y_std"])
        wrapped.load_state_dict(ckpt["state_dict"])
        wrapped.eval()
        _cached_model = wrapped
        _cached_norm  = norm
    x = feature_vec(channel, _cached_norm,
                    sfr_a=sfr_a, sfr_b=sfr_b, sfr_c=sfr_c, sfr_d=sfr_d,
                    mu0=mu0, muz=muz, sigma0=sigma0,
                    sigmaz=sigmaz, alpha_skew=alpha_skew)[np.newaxis]
    with torch.no_grad():
        return float(_cached_model(torch.from_numpy(x).float()))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SSPC rate network")
    parser.add_argument("--hyperparam-csv", type=Path, default=None)
    parser.add_argument("--splits-json",    type=Path, default=None)
    parser.add_argument("--checkpoint-dir", type=Path, default=CHECKPOINT_DIR)
    parser.add_argument("--epochs",   type=int, default=3000)
    parser.add_argument("--patience", type=int, default=300)
    args = parser.parse_args()

    csv_path  = args.hyperparam_csv or HYPERPARAM_CSV
    json_path = args.splits_json    or SPLITS_JSON
    ckpt_dir  = args.checkpoint_dir
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        sys.exit(f"ERROR: {csv_path} not found. Run 02_build_dataset.py first.")

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    plots_dir = PLOTS_BASE / ts
    plots_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    print("1. Loading data …")
    df = pd.read_csv(csv_path)
    with open(json_path) as f:
        splits = json.load(f)

    X, y, norm_params = build_features(df)
    n_ch = {ch: int((df["channel"] == ch).sum()) for ch in CHANNEL_ORDER}
    print(f"   Grid points : {len(df)}  " +
          "  ".join(f"{ch}={n_ch[ch]}" for ch in CHANNEL_ORDER if n_ch[ch]))
    print(f"   log10(rate) : [{y.min():.2f}, {y.max():.2f}]  (floor={LOG_RATE_FLOOR})")
    print(f"   Splits      : train={len(splits['train'])}  val={len(splits['val'])}  test={len(splits['test'])}")
    print(f"   Features    : 3 channel + 9 SSPC params = {X.shape[1]} dims")

    tr, va, te = splits["train"], splits["val"], splits["test"]
    X_tr, y_tr = X[tr], y[tr]
    X_va, y_va = X[va], y[va]

    # ------------------------------------------------------------------
    print("\n2. Training MLP …")
    model, tloss, vloss, best_ep = train(
        X_tr, y_tr, X_va, y_va,
        epochs=args.epochs, patience=args.patience,
    )

    # ------------------------------------------------------------------
    print("\n3. Training GP baseline …")
    X_tv = np.concatenate([X_tr, X_va])
    y_tv = np.concatenate([y_tr, y_va])
    gp = fit_gp(X_tv, y_tv)

    # ------------------------------------------------------------------
    print("\n4. Evaluating …")
    X_te, y_te = X[te], y[te]
    pred_nn = predict(model, X_te)
    pred_gp = gp.predict(X_te)
    r_nn, r_gp = r2(y_te, pred_nn), r2(y_te, pred_gp)
    print(f"   Test R²  —  MLP: {r_nn:.4f}   GP: {r_gp:.4f}")

    ch_te = df["channel"].iloc[te].values
    for ch in CHANNEL_ORDER:
        m = ch_te == ch
        if m.sum() < 2:
            continue
        r_ch_nn = r2(y_te[m], pred_nn[m])
        r_ch_gp = r2(y_te[m], pred_gp[m])
        print(f"   {ch}: MLP R²={r_ch_nn:.4f}  GP R²={r_ch_gp:.4f}  (n={m.sum()})")

    # ------------------------------------------------------------------
    print("\n5. Saving checkpoint …")
    save_checkpoint(model, norm_params, ckpt_dir)
    save_gp(gp, ckpt_dir)

    # ------------------------------------------------------------------
    print("\n6. Generating plots …")
    plot_scatter(model, gp, X, y, splits, df, plots_dir)
    plot_param_sensitivity(model, df, norm_params, plots_dir)
    plot_sfra_mu0_heatmap(model, df, norm_params, plots_dir)
    plot_learning_curves(tloss, vloss, best_ep, plots_dir)

    print(f"\nDone.  Plots saved to: {plots_dir}")


if __name__ == "__main__":
    main()
