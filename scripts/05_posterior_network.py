#!/usr/bin/env python3
"""
Amortized posterior p(Λ | catalog) over 9 SSPC parameters.

**Training data (PopFlow / proposal):** The loss is **not** computed on batches
read from 02’s `all_events.parquet` (or `all_detected_events.parquet`). For each hyperparameter row Λ, a **synthetic
catalog** is generated on the fly with a **trained, frozen** generative
emulator—either the CFM (04) or the diffusion emulator (04b). The posterior
network learns to invert that forward process (fast synthetic catalogs →
settings), matching the proposal’s Stage-2-then-Stage-4 pipeline: emulator
replaces the expensive simulator; the transformer+flow is trained *through*
synthetic data produced by the emulator, not in parallel to it.

**Required inputs:** `hyperparam_table_encoded.csv`, `splits.json`, and a
**completed** 04/04b checkpoint (`cfm_final.pt` or `diffusion_final.pt`).

**Event features (6-d):** z-scored observables (mchirp, q, z) plus three σ-broadcast channels—
using the `obs_normalizer` **stored inside the emulator checkpoint** (consistent
with 04).

**Checkpoints:** `posterior_network_best.pt`, `posterior_network_config.json`

Use `--num-workers 0` (default); workers cannot hold the ODE/diffusion model.
"""
from __future__ import annotations

import sys
from pathlib import Path as _Path

_PROJECT_ROOT = _Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from plant_paths import (  # noqa: E402
    CHECKPOINT_DIR,
    HYPERPARAM_TABLE_ENCODED_CSV,
    PROJECT_ROOT,
    SPLITS_JSON,
    ensure_paths,
    ml_data_dir,
    plot_run_dir,
)

ensure_paths()

import argparse
import json
import random
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

# -----------------------------------------------------------------------------
# Paths (run from PLANT_GW_Paleontology/)
# -----------------------------------------------------------------------------
HYPERPARAM_CSV = HYPERPARAM_TABLE_ENCODED_CSV
DEFAULT_CFM_CKPT = CHECKPOINT_DIR / "cfm_final.pt"
DEFAULT_DIFF_CKPT = CHECKPOINT_DIR / "diffusion_final.pt"
DEFAULT_NB_CKPT = CHECKPOINT_DIR / "naive_bayes_final.pt"

SSPC_THETA_PARAM_COLS = [
    "sspc_sfr_a_mean",
    "sspc_sfr_b_mean",
    "sspc_sfr_c_mean",
    "sspc_sfr_d_mean",
    "sspc_mu0_mean",
    "sspc_muz_mean",
    "sspc_sigma0_mean",
    "sspc_sigmaz_mean",
    "sspc_alpha_skew_mean",
]





def build_events_6d(
    mchirp: np.ndarray,
    q: np.ndarray,
    z: np.ndarray,
    nrm: Dict[str, Any],
) -> np.ndarray:
    m = np.log10(np.maximum(mchirp.astype(np.float64), 1e-3))
    zr = np.log10(np.maximum(z.astype(np.float64), 0.1))
    qd = q.astype(np.float64)

    x0 = (m - nrm["mchirp"]["mean"]) / nrm["mchirp"]["std"]
    x1 = (qd - nrm["q"]["mean"]) / nrm["q"]["std"]
    x2 = (zr - nrm["z"]["mean"]) / nrm["z"]["std"]

    s0 = float(nrm["mchirp"]["std"])
    s1 = float(nrm["q"]["std"])
    s2 = float(nrm["z"]["std"])
    s = np.broadcast_to(np.array([[s0, s1, s2]], np.float64), (len(mchirp), 3))
    return np.hstack(
        [x0[:, None], x1[:, None], x2[:, None], s]
    ).astype(np.float32)


@dataclass
class ThetaStats:
    mean: np.ndarray
    std: np.ndarray


def compute_theta_stats(
    df: pd.DataFrame, train_idx: List[int]
) -> ThetaStats:
    sub = df.iloc[train_idx]
    t = sub[SSPC_THETA_PARAM_COLS].values.astype(np.float64)
    mean = t.mean(axis=0)
    std = np.clip(t.std(axis=0), 1e-8, None)
    return ThetaStats(mean=mean, std=std)


def load_frozen_emulator(
    ckpt_path: Path,
    device: torch.device,
    kind: str,
) -> Tuple[nn.Module, List[str], Dict[str, Any]]:
    """Load CFM or DiffusionEmulator; freeze weights. Returns (model, lambda_cols, normalizer)."""
    if not ckpt_path.is_file():
        print(
            f"ERROR: emulator checkpoint not found: {ckpt_path}\n"
            f"Train/fit the generative model first: 04_cfm_emulator.py, 04b_diffusion_emulator.py, or 04c_naive_bayes_emulator.py",
            file=sys.stderr,
        )
        sys.exit(1)
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    lambda_cols: List[str] = ck["lambda_cols"]
    nrm: Dict[str, Any] = ck["normalizer"]
    hdim = int(ck.get("hidden_dim", 256))
    ctxd = int(ck.get("context_dim", 128))
    if kind == "cfm":
        from models.cfm_emulator import CFMEmulator

        m = CFMEmulator(lambda_dim=len(lambda_cols), context_dim=ctxd, hidden_dim=hdim)
    elif kind == "diffusion":
        from models.diffusion_emulator import DiffusionEmulator

        nt = int(ck.get("n_timesteps", 100))
        m = DiffusionEmulator(
            lambda_dim=len(lambda_cols),
            context_dim=ctxd,
            hidden_dim=hdim,
            n_timesteps=nt,
        )
        m.load_state_dict(ck["model_state"], strict=True)
    elif kind == "naive_bayes":
        from models.naive_bayes_emulator import load_from_checkpoint

        m, lambda_cols, nrm = load_from_checkpoint(ck, device=device)
        m.eval()
        for p in m.parameters():
            p.requires_grad = False
        return m, lambda_cols, nrm
    else:
        sys.exit(f"ERROR: --emulator must be cfm, diffusion, or naive_bayes, got {kind!r}.")
    m.load_state_dict(ck["model_state"], strict=True)
    m.to(device)
    m.eval()
    for p in m.parameters():
        p.requires_grad = False
    return m, lambda_cols, nrm


def _generate_dataframe(
    emulator: nn.Module, kind: str, lambda_vec: np.ndarray, n_events: int, nrm: Dict[str, Any]
) -> pd.DataFrame:
    if kind == "cfm":
        from models.cfm_emulator import generate_catalog
    elif kind == "diffusion":
        from models.diffusion_emulator import generate_catalog
    elif kind == "naive_bayes":
        from models.naive_bayes_emulator import generate_catalog
    else:
        raise ValueError(f"Unknown emulator kind {kind!r}")
    return generate_catalog(
        np.asarray(lambda_vec, dtype=np.float32), n_events, emulator, nrm
    )


class EmulatorSyntheticCatalogDataset(Dataset):
    """
    (theta, events_6d, mask) for one grid row. Events are **Fresh draws** from
    the **frozen** CFM or diffusion at that row’s Λ (lambda_*).
    """

    def __init__(
        self,
        split_indices: List[int],
        hp_df: pd.DataFrame,
        lambda_cols: List[str],
        emulator: nn.Module,
        emulator_kind: str,
        normalizer: Dict[str, Any],
        n_max_events: int,
        base_seed: int = 0,
    ) -> None:
        self._split = [int(x) for x in split_indices]
        self.hp = hp_df.reset_index(drop=True)
        self.lambda_cols = lambda_cols
        self.emulator = emulator
        self.emulator_kind = emulator_kind
        self.nrm = normalizer
        self.n_max = n_max_events
        self._base = base_seed
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Call each training epoch so CFM ODE / diffusion noise varies across epochs."""
        self._epoch = int(epoch)

    def __len__(self) -> int:
        return len(self._split)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row_i = self._split[i]
        row = self.hp.iloc[row_i]
        theta = torch.from_numpy(
            row[SSPC_THETA_PARAM_COLS].values.astype(np.float32)
        )
        lam = row[self.lambda_cols].values.astype(np.float32)
        # ODE + randn: vary by row index, epoch, and base seed
        s = (self._base + 97 * int(i) + 100_003 * (self._epoch + 1)) % (2**31 - 1)
        torch.manual_seed(int(s))
        with torch.no_grad():
            cat = _generate_dataframe(
                self.emulator, self.emulator_kind, lam, self.n_max, self.nrm
            )
        x6 = build_events_6d(
            cat["mchirp"].values,
            cat["q"].values,
            cat["z"].values,
            self.nrm,
        )
        m = np.ones(self.n_max, dtype=np.float32)
        return theta, torch.from_numpy(x6), torch.from_numpy(m)


def _collate(
    batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    thetas = torch.stack([b[0] for b in batch], dim=0)
    ev = torch.stack([b[1] for b in batch], dim=0)
    m = torch.stack([b[2] for b in batch], dim=0)
    return thetas, ev, m


def _eval_nll(
    model: nn.Module, loader: DataLoader, device: torch.device, use_amp: bool
) -> float:
    model.eval()
    tot, nb = 0.0, 0
    with torch.no_grad():
        for theta, ev, msk in loader:
            theta, ev, msk = theta.to(device), ev.to(device), msk.to(device)
            ctx = (
                torch.amp.autocast("cuda", enabled=use_amp)
                if device.type == "cuda"
                else nullcontext()
            )
            with ctx:
                loss = -model.log_prob(theta, ev, msk).mean()
            tot += float(loss.cpu())
            nb += 1
    return tot / max(nb, 1)


def save_posterior_ckpt(
    model: nn.Module,
    out_dir: Path,
    norm_meta: Dict[str, Any],
    weights_path: Path | None = None,
) -> None:
    """
    Save weights + config. If `weights_path` is set, that file is used and
    `posterior_network_config.json` is written to `weights_path.parent`.
    """
    state = model.state_dict() if not hasattr(model, "module") else model.module.state_dict()
    if weights_path is not None:
        wpath = Path(weights_path)
        wpath.parent.mkdir(parents=True, exist_ok=True)
        path = wpath
        cfg_dir = wpath.parent
    else:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "posterior_network_best.pt"
        cfg_dir = out_dir
    torch.save({"state_dict": state}, path)
    cfg = {
        "sspc_theta_param_cols": SSPC_THETA_PARAM_COLS,
        "input_event_dim": 6,
        "norm_meta": norm_meta,
    }
    with open(cfg_dir / "posterior_network_config.json", "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"  Saved {path}")


def _resolve(p: Path, root: Path) -> Path:
    return p if p.is_absolute() else root / p


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train posterior on emulator-synthetic catalogs (frozen 04/04b)."
    )
    parser.add_argument("--hyperparam-csv", type=Path, default=HYPERPARAM_CSV)
    parser.add_argument("--splits-json", type=Path, default=SPLITS_JSON)
    parser.add_argument("--checkpoint-dir", type=Path, default=CHECKPOINT_DIR)
    parser.add_argument(
        "--emulator",
        type=str,
        default="cfm",
        choices=("cfm", "diffusion", "naive_bayes"),
        help="Which generative model (Step 04, 04b, or 04c baseline) produced the training path.",
    )
    parser.add_argument(
        "--emulator-checkpoint",
        type=Path,
        default=None,
        help="Path to cfm_final.pt or diffusion_final.pt (default: checkpoints/…).",
    )
    parser.add_argument("--model", type=str, default="lite", choices=("lite", "full"))
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--n-max-events", type=int, default=256)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Must be 0 (emulator+ODE runs in main process for each sample).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--amp", action="store_true", help="Use CUDA autocast (mixed precision)."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="auto | cpu | cuda (emulator+posterior on same device).",
    )
    parser.add_argument("--accum-steps", type=int, default=1)
    parser.add_argument(
        "--output-checkpoint-pt",
        type=Path,
        default=None,
        help=(
            "Optional path for the saved .pt file (e.g. checkpoints/posterior_ensemble/2/posterior_network_best.pt). "
            "Config JSON is written next to it. If omitted, uses --checkpoint-dir/posterior_network_best.pt."
        ),
    )
    args = parser.parse_args()

    if args.num_workers != 0:
        print("  Forcing --num-workers 0 (required for on-the-fly emulator generation).", file=sys.stderr)
        args.num_workers = 0

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    data_root = ml_data_dir()
    csv_p = _resolve(args.hyperparam_csv, data_root)
    sp_p = _resolve(args.splits_json, data_root)
    ckpt_dir = _resolve(args.checkpoint_dir, PROJECT_ROOT)

    emu_path = args.emulator_checkpoint
    if emu_path is None:
        if args.emulator == "cfm":
            default_ckpt = DEFAULT_CFM_CKPT
        elif args.emulator == "diffusion":
            default_ckpt = DEFAULT_DIFF_CKPT
        else:
            default_ckpt = DEFAULT_NB_CKPT
        emu_path = _resolve(default_ckpt, PROJECT_ROOT)
    else:
        emu_path = _resolve(emu_path, PROJECT_ROOT)

    print("1. Load hyperparameters + frozen emulator (Step 04/04b)…")
    hp = pd.read_csv(csv_p)
    with open(sp_p) as f:
        sp = json.load(f)

    missing = [c for c in SSPC_THETA_PARAM_COLS if c not in hp.columns]
    if missing:
        sys.exit(
            f"ERROR: expected SSPC mean columns; missing: {missing}. "
            "Re-run 02 with --data-source sspc."
        )

    emulator, lambda_cols, obs_nrm = load_frozen_emulator(emu_path, device, args.emulator)
    for c in lambda_cols:
        if c not in hp.columns:
            sys.exit(
                f"ERROR: column {c!r} in emulator checkpoint but missing from {csv_p}. "
                "Re-run 02 and train 04/04b on the same table."
            )

    print(
        f"   Emulator: {args.emulator}  |  Λ dim={len(lambda_cols)}  |  {emu_path.name}"
    )
    print(
        "   Training posterior on **synthetic** catalogs from this model (emulator frozen)."
    )

    train_i = [int(x) for x in sp["train"]]
    val_i = [int(x) for x in sp["val"]]
    tstats = compute_theta_stats(hp, train_i)

    from models.posterior_network_lite import LitePosteriorNet, PosteriorNet
    from models.posterior_network_full import FullPosteriorNet

    if args.model == "lite":
        model: PosteriorNet = LitePosteriorNet()
    else:
        model = FullPosteriorNet()

    model.set_theta_stats(
        torch.from_numpy(tstats.mean.astype(np.float32)),
        torch.from_numpy(tstats.std.astype(np.float32)),
    )
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"2. Posterior parameters: {n_params:,}  |  device={device}")

    tr_ds = EmulatorSyntheticCatalogDataset(
        train_i,
        hp,
        lambda_cols,
        emulator,
        args.emulator,
        obs_nrm,
        args.n_max_events,
        base_seed=args.seed,
    )
    va_ds = EmulatorSyntheticCatalogDataset(
        val_i,
        hp,
        lambda_cols,
        emulator,
        args.emulator,
        obs_nrm,
        args.n_max_events,
        base_seed=args.seed + 1,
    )
    tr_loader = DataLoader(
        tr_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=_collate,
        pin_memory=device.type == "cuda",
    )
    va_loader = DataLoader(
        va_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=_collate,
    )

    opt = Adam(model.parameters(), lr=args.lr)
    best_val = float("inf")
    bad = 0
    t_loss_hist: List[float] = []
    v_loss_hist: List[float] = []

    use_amp = bool(args.amp and device.type == "cuda")
    print("3. Train posterior (NLL)…")
    for ep in range(1, args.epochs + 1):
        tr_ds.set_epoch(ep)
        model.train()
        tot_loss, nb = 0.0, 0
        opt.zero_grad(set_to_none=True)
        for step, (theta, e, m) in enumerate(tr_loader):
            theta, e, m = theta.to(device), e.to(device), m.to(device)
            ctx = (
                torch.amp.autocast("cuda", enabled=use_amp) if device.type == "cuda" else nullcontext()
            )
            with ctx:
                nll = -model.log_prob(theta, e, m).mean() / float(args.accum_steps)
            nll.backward()
            if (step + 1) % args.accum_steps == 0 or (step + 1) == len(tr_loader):
                if args.max_grad_norm and args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                opt.step()
                opt.zero_grad(set_to_none=True)
            tot_loss += float(nll.detach().cpu()) * args.accum_steps
            nb += 1
        tr_loss = tot_loss / max(nb, 1)
        t_loss_hist.append(tr_loss)
        v_loss = _eval_nll(model, va_loader, device, use_amp)
        v_loss_hist.append(v_loss)
        if ep == 1 or ep % 5 == 0 or ep == args.epochs:
            print(
                f"   epoch {ep:4d}  train_nll {tr_loss:.4f}  val_nll {v_loss:.4f}"
            )
        if v_loss < best_val - 1e-6:
            best_val = v_loss
            bad = 0
            norm_meta = {
                "theta_mean": tstats.mean.tolist(),
                "theta_std": tstats.std.tolist(),
                "obs_normalizer": {k: obs_nrm[k] for k in obs_nrm if isinstance(obs_nrm[k], dict)},
                "emulator": args.emulator,
                "emulator_checkpoint": str(emu_path.resolve()),
                "lambda_columns": lambda_cols,
            }
            save_posterior_ckpt(
                model, ckpt_dir, norm_meta, weights_path=args.output_checkpoint_pt
            )
        else:
            bad += 1
            if bad >= args.patience:
                print(f"   Early stop at epoch {ep} (patience={args.patience}).")
                break

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    plot_d = plot_run_dir(Path(__file__), timestamp=ts)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(t_loss_hist, label="train NLL")
        ax.plot(v_loss_hist, label="val NLL")
        ax.set_xlabel("epoch")
        ax.set_ylabel("NLL")
        ax.legend()
        fig.savefig(plot_d / "learning_curves.png", dpi=120, bbox_inches="tight")
        plt.close()
        print(f"  Saved {plot_d / 'learning_curves.png'}")
    except Exception as ex:  # noqa: BLE001
        print(f"  (no plot) {ex}")

    print("Done.")


if __name__ == "__main__":
    main()
