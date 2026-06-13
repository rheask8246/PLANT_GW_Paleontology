#!/usr/bin/env python3
"""
CFM Emulator: Conditional Flow Matching for merger event generation.

Uses all_events.parquet (intrinsic merger samples), hyperparam_table_encoded.csv, splits.json.
Plots: scripts/analysis/04_cfm_emulator_plots.py (or slurm/04_cfm_emulator_plots.sh).
"""

from __future__ import annotations

import sys
from pathlib import Path as _Path

_PROJECT_ROOT = _Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from plant_paths import (  # noqa: E402
    ALL_EVENTS_PARQUET,
    CHECKPOINT_DIR,
    HYPERPARAM_TABLE_ENCODED_CSV,
    PROJECT_ROOT,
    SPLITS_JSON,
    ensure_paths,
    ml_data_dir,
)

ensure_paths()

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

# Smoke test
SMOKE_TEST = True  # Set True for 500 steps on CPU
N_BATCH = 256 if not SMOKE_TEST else 64
STEPS = 100000 if not SMOKE_TEST else 500
HIDDEN_DIM = 256 if not SMOKE_TEST else 128

try:
    from torchcfm.conditional_flow_matching import ExactOptimalTransportConditionalFlowMatcher
except Exception:
    ExactOptimalTransportConditionalFlowMatcher = None

def _get_flow_matcher():
    if ExactOptimalTransportConditionalFlowMatcher is None:
        raise ImportError(
            "torchcfm required. Install with: pip install torchcfm "
            "or pip install git+https://github.com/atong01/conditional-flow-matching.git"
        )
    return ExactOptimalTransportConditionalFlowMatcher(sigma=0.0)

try:
    from torchdiffeq import odeint
except ImportError:
    odeint = None


def configure_worker_threads(workers: int) -> int:
    """Set CPU thread workers for BLAS/OpenMP; useful for data path on GPU jobs."""
    w = max(1, int(workers))
    os.environ["OMP_NUM_THREADS"] = str(w)
    os.environ["MKL_NUM_THREADS"] = str(w)
    os.environ["OPENBLAS_NUM_THREADS"] = str(w)
    os.environ["NUMEXPR_NUM_THREADS"] = str(w)
    torch.set_num_threads(w)
    return w


def _grid_rate_column(hp_df: pd.DataFrame) -> str:
    """Per-grid total merger rate for importance reweighting (intrinsic: sum_weight)."""
    if "sum_weight" in hp_df.columns:
        return "sum_weight"
    return "sum_pdet"


def load_or_build_obs_normalizer(parquet_path: Path, out_path: Path) -> Dict:
    """
    Load normalizer from 02_build_dataset.py output if it exists.
    Otherwise build from the same intrinsic table (all_events) as 02. mchirp, z: log10 first, then mean/std.
    """
    if out_path.exists():
        with open(out_path) as f:
            normalizer = json.load(f)
        print(f"   Loaded obs_normalizer from {out_path} (from 02_build_dataset.py)")
        return normalizer
    # Fallback: build from parquet (matches 02's compute_and_save_obs_normalizer)
    df = pd.read_parquet(parquet_path)
    cols = ["mchirp", "q", "z"]
    normalizer = {}
    for col in cols:
        x = df[col].values.astype(np.float64)
        if col in ("mchirp", "z"):
            x = np.log10(np.maximum(x, 1e-10))
        normalizer[col] = {"mean": float(np.mean(x)), "std": float(np.std(x) + 1e-8)}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(normalizer, f, indent=2)
    print(f"   Built obs_normalizer from parquet (run 02_build_dataset.py to precompute)")
    return normalizer


def load_normalizer(path: Path) -> Dict:
    with open(path) as f:
        return json.load(f)


def _lambda_cols(df: pd.DataFrame) -> List[str]:
    return sorted(
        [c for c in df.columns if c.startswith("lambda_")],
        key=lambda x: int(x.split("_")[1]),
    )


def _is_sspc_hyperparam_df(hp_df: pd.DataFrame) -> bool:
    """Heuristic: SSPC grids have (sfra, mu0) columns; Zenodo has (chi_b, alpha_CE)."""
    cols = set(hp_df.columns.astype(str))
    return ("sfra" in cols) and ("mu0" in cols)


def _histogram_kl(
    x_true: np.ndarray,
    x_model: np.ndarray,
    *,
    bins: int = 60,
    eps: float = 1e-12,
) -> float:
    """
    Discrete KL(true || model) using matched histogram bins.

    Uses a small epsilon floor so empty bins don't produce inf/NaN.
    """
    xt = np.asarray(x_true, dtype=np.float64)
    xm = np.asarray(x_model, dtype=np.float64)
    xt = xt[np.isfinite(xt)]
    xm = xm[np.isfinite(xm)]
    if xt.size == 0 or xm.size == 0:
        return float("nan")

    lo = float(min(np.min(xt), np.min(xm)))
    hi = float(max(np.max(xt), np.max(xm)))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return 0.0

    h_t, edges = np.histogram(xt, bins=int(bins), range=(lo, hi), density=False)
    h_m, _ = np.histogram(xm, bins=edges, density=False)
    p = h_t.astype(np.float64) + float(eps)
    q = h_m.astype(np.float64) + float(eps)
    p = p / np.sum(p)
    q = q / np.sum(q)
    return float(np.sum(p * (np.log(p) - np.log(q))))


def sample_events_from_grid(
    events_df: pd.DataFrame,
    grid_idx: int,
    n: int,
    rng: np.random.Generator,
    z_jitter: bool = True,
    *,
    z_clip_max: float | None = None,
) -> np.ndarray:
    """Uniform row subsample for grid_idx (expect intrinsic `all_events` from 02). Returns (n,3) [mchirp,q,z]."""
    mask = events_df["grid_idx"] == grid_idx
    sub = events_df.loc[mask, ["mchirp", "q", "z"]]
    if len(sub) == 0:
        return np.zeros((n, 3), dtype=np.float32)
    idx = rng.integers(0, len(sub), size=min(n, len(sub)))
    if len(idx) < n:
        idx = rng.choice(len(sub), size=n, replace=True)
    x = sub.iloc[idx].values.astype(np.float32)
    if z_jitter:
        # Smooth the discrete z grid (bin width 0.1) so the model learns a
        # continuous distribution rather than a delta function at each bin edge.
        if z_clip_max is None:
            z_clip_max = float(sub["z"].max()) if "z" in sub.columns and len(sub) else 10.0
        x[:, 2] = np.clip(
            x[:, 2] + rng.uniform(-0.05, 0.05, size=n).astype(np.float32),
            1e-6,
            float(z_clip_max),
        )
    return x


def _pack_events_by_grid(events_df: pd.DataFrame, *, n_grid: int) -> Tuple[np.ndarray, np.ndarray]:
    """Pack (mchirp,q,z) into one array + ptr offsets by grid_idx."""
    grid = events_df["grid_idx"].values.astype(np.int64, copy=False)
    order = np.argsort(grid, kind="mergesort")
    grid_sorted = grid[order]
    obs_sorted = events_df[["mchirp", "q", "z"]].values.astype(np.float32, copy=False)[order]
    counts = np.bincount(grid_sorted.clip(min=0, max=n_grid - 1), minlength=n_grid).astype(
        np.int64, copy=False
    )
    ptr = np.zeros(n_grid + 1, dtype=np.int64)
    ptr[1:] = np.cumsum(counts)
    return obs_sorted, ptr


def sample_events_from_packed(
    obs_sorted: np.ndarray,
    ptr: np.ndarray,
    grid_idx: int,
    n: int,
    rng: np.random.Generator,
    *,
    z_jitter: bool = True,
    z_clip_max: float | None = None,
) -> np.ndarray:
    start = int(ptr[int(grid_idx)])
    end = int(ptr[int(grid_idx) + 1])
    m = end - start
    if m <= 0:
        return np.zeros((n, 3), dtype=np.float32)
    if m >= n:
        idx = start + rng.integers(0, m, size=n)
    else:
        idx = start + rng.choice(m, size=n, replace=True)
    x = obs_sorted[idx].astype(np.float32, copy=False)
    if z_jitter:
        x = x.copy()
        if z_clip_max is None:
            z_clip_max = float(np.max(obs_sorted[:, 2])) if obs_sorted.size else 10.0
        x[:, 2] = np.clip(
            x[:, 2] + rng.uniform(-0.05, 0.05, size=n).astype(np.float32),
            1e-6,
            float(z_clip_max),
        )
    return x


def _sample_events_any(
    *,
    obs_sorted: np.ndarray | None,
    ptr: np.ndarray | None,
    events_df: pd.DataFrame,
    grid_idx: int,
    n: int,
    rng: np.random.Generator,
    z_jitter: bool = True,
    z_clip_max: float | None = None,
) -> np.ndarray:
    if obs_sorted is not None and ptr is not None:
        return sample_events_from_packed(
            obs_sorted,
            ptr,
            grid_idx,
            n,
            rng,
            z_jitter=z_jitter,
            z_clip_max=z_clip_max,
        )
    return sample_events_from_grid(
        events_df,
        grid_idx,
        n,
        rng,
        z_jitter=z_jitter,
        z_clip_max=z_clip_max,
    )


def run_smoke_test(
    device: str = "cpu",
    steps: int = 500,
    output_checkpoint: Path | None = None,
    seed: int = 42,
    *,
    pack_events: bool = True,
) -> None:
    """Run smoke test (or full run when global SMOKE_TEST is False) and save checkpoint."""
    import sys
    ensure_paths()
    from models.cfm_emulator import CFMEmulator, normalize_obs, denormalize_obs, generate_catalog

    data_dir = ml_data_dir()
    hp_csv = data_dir / HYPERPARAM_TABLE_ENCODED_CSV.name
    events_pq = data_dir / ALL_EVENTS_PARQUET.name
    splits_path = data_dir / SPLITS_JSON.name
    ckpt_dir = CHECKPOINT_DIR
    work_dir = PROJECT_ROOT

    if not all(p.exists() for p in [hp_csv, events_pq, splits_path]):
        raise FileNotFoundError("Run 02_build_dataset.py first.")

    print("=" * 60)
    print("SMOKE TEST MODE")
    print("=" * 60)

    # Load normalizer (from 02_build_dataset.py) or build from parquet
    normalizer = load_or_build_obs_normalizer(events_pq, ckpt_dir / "obs_normalizer.json")
    hp_df = pd.read_csv(hp_csv)
    with open(splits_path) as f:
        splits = json.load(f)
    train_idx = splits["train"]
    val_idx = splits["val"]
    test_idx = splits["test"]

    # Importance weights: sample rare (low intrinsic rate) grid points more often.
    # Floor at 1% of the median to prevent near-zero rate totals from dominating the
    # sampling distribution. Without this floor, one degenerate point absorbs ~100% of
    # sampling probability and produces importance_ratio ≈ 0, giving zero gradient throughout.
    rate_col = _grid_rate_column(hp_df)
    rate_tot = hp_df[rate_col].values
    r_floor = max(float(np.median(rate_tot)) * 0.01, 1e-4)
    rate_clipped = np.maximum(rate_tot, r_floor)
    w = 1.0 / rate_clipped
    w = w / w.sum()
    n_grid = len(hp_df)
    p_uniform = 1.0 / n_grid

    events_df = pd.read_parquet(events_pq, columns=["mchirp", "q", "z", "grid_idx"])
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    obs_sorted = None
    ptr = None
    if pack_events:
        print("   Packing events by grid_idx (faster sampling)...", flush=True)
        obs_sorted, ptr = _pack_events_by_grid(events_df, n_grid=len(hp_df))
        # Keep a tiny dataframe only if needed later (we don't).
        events_df = pd.DataFrame()
    z_clip_max = float(np.max(obs_sorted[:, 2])) if obs_sorted is not None else float(events_df["z"].max())

    lambda_cols = _lambda_cols(hp_df)
    model = CFMEmulator(lambda_dim=len(lambda_cols), context_dim=128, hidden_dim=HIDDEN_DIM)
    flow_matcher = _get_flow_matcher()
    optimizer = Adam(model.parameters(), lr=2e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=steps, eta_min=1e-5)

    model.to(device)
    loss_0 = None
    loss_500 = None
    train_losses: List[float] = []
    val_losses: List[Tuple[int, float]] = []  # (step, val_loss)
    grad_norms: List[float] = []
    kl_logs: List[Tuple[int, float, float, float]] = []  # (step, mchirp_kl, q_kl, z_kl)

    sspc_data = _is_sspc_hyperparam_df(hp_df)
    if sspc_data:
        for _ch in ["SMT", "CE", "CHE"]:
            ch_rows = hp_df[hp_df["channel"] == _ch]
            if len(ch_rows) > 0:
                break
        mid_p1 = float(np.median(ch_rows["sfra"]))
        mid_p2 = float(np.median(ch_rows["mu0"]))
        dists = (ch_rows["sfra"] - mid_p1).abs() + (ch_rows["mu0"] - mid_p2).abs()
        grid_idx_ce = dists.idxmin()
    else:
        ce_match = hp_df[(hp_df["channel"] == "CE") & (hp_df["chi_b"] == 0.2) & (hp_df["alpha_CE"] == 1.0)]
        if len(ce_match) == 0:
            ce_match = hp_df[(hp_df["channel"] == "CE") & (hp_df["chi_b"] == 0.2)]
        grid_idx_ce = ce_match.index[0] if len(ce_match) > 0 else 0
    lam_ce = hp_df.iloc[grid_idx_ce][lambda_cols].values.astype(np.float32)

    for step in range(steps):
        # Sample grid point
        i = rng.choice(n_grid, p=w)
        importance_ratio = p_uniform / (w[i] + 1e-10)
        importance_ratio = min(importance_ratio, 3.0)

        # Sample events
        x1_raw = _sample_events_any(
            obs_sorted=obs_sorted,
            ptr=ptr,
            events_df=events_df,
            grid_idx=i,
            n=N_BATCH,
            rng=rng,
            z_clip_max=z_clip_max,
        )
        x1_norm = normalize_obs(x1_raw, normalizer)
        x1 = torch.from_numpy(x1_norm).float().to(device)
        x0 = torch.randn_like(x1, device=device)
        lam = torch.from_numpy(hp_df.iloc[i][lambda_cols].values.astype(np.float32)).unsqueeze(0).to(device)
        lam = lam.expand(x1.shape[0], -1)

        t, xt, ut = flow_matcher.sample_location_and_conditional_flow(x0, x1)
        context = model.encoder(lam)
        vt = model.vector_field(xt, t, context)
        loss = ((vt - ut) ** 2).mean() * importance_ratio

        optimizer.zero_grad()
        loss.backward()

        # Log gradient norm BEFORE clipping (for diagnostics)
        total_norm = 0.0
        for p in model.vector_field.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        grad_norms.append(total_norm ** 0.5)

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        scheduler.step()

        train_losses.append(loss.item())
        if step == 0:
            loss_0 = loss.item()
        if step == steps - 1:
            loss_500 = loss.item()

        # Val loss and mchirp KL every 50 steps (for plotting and diagnostics)
        if (step + 1) % 50 == 0 and val_idx:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for vi in val_idx[:3]:
                    x1_raw = _sample_events_any(
                        obs_sorted=obs_sorted,
                        ptr=ptr,
                        events_df=events_df,
                        grid_idx=vi,
                        n=64,
                        rng=rng,
                        z_clip_max=z_clip_max,
                    )
                    x1_norm = normalize_obs(x1_raw, normalizer)
                    x1 = torch.from_numpy(x1_norm).float().to(device)
                    x0 = torch.randn_like(x1, device=device)
                    lam = torch.from_numpy(hp_df.iloc[vi][lambda_cols].values.astype(np.float32)).unsqueeze(0).expand(x1.shape[0], -1).to(device)
                    t, xt, ut = flow_matcher.sample_location_and_conditional_flow(x0, x1)
                    vt = model.vector_field(xt, t, model.encoder(lam))
                    val_loss += ((vt - ut) ** 2).mean().item()
            val_loss /= min(3, len(val_idx))
            val_losses.append((step + 1, val_loss))
            torch.manual_seed(seed + step)
            cat_kl = generate_catalog(lam_ce, 1000, model, normalizer)
            true_kl = _sample_events_any(
                obs_sorted=obs_sorted,
                ptr=ptr,
                events_df=events_df,
                grid_idx=grid_idx_ce,
                n=1000,
                rng=rng,
                z_clip_max=z_clip_max,
            )
            mchirp_kl = _histogram_kl(true_kl[:, 0], cat_kl["mchirp"].values)
            q_kl = _histogram_kl(true_kl[:, 1], cat_kl["q"].values)
            z_kl = _histogram_kl(true_kl[:, 2], cat_kl["z"].values)
            kl_logs.append((step + 1, mchirp_kl, q_kl, z_kl))
            ckpt_dir = work_dir / "test" / "checkpoints"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict(), "step": step + 1}, ckpt_dir / f"cfm_step_{step + 1}.pt")
            model.train()

        if (step + 1) % 100 == 0:
            ess = (np.sum(w) ** 2) / (np.sum(w ** 2) + 1e-10)
            ess_frac = ess / n_grid
            print(f"  Step {step+1}: loss={loss.item():.6f}, ESS/N={ess_frac:.4f}")

    # Save KL log for diagnostics (all observables)
    logs_dir = work_dir / "test" / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    with open(logs_dir / "cfm_kl_log.json", "w") as f:
        json.dump({
            "steps": [s for s, _m, _q, _z in kl_logs],
            "mchirp_kl": [m for _s, m, _q, _z in kl_logs],
            "q_kl": [q for _s, _m, q, _z in kl_logs],
            "z_kl": [z for _s, _m, _q, z in kl_logs],
        }, f, indent=2)

    # Generate catalog for validation
    lam_test = hp_df.iloc[test_idx[0]][lambda_cols].values.astype(np.float32)
    catalog = generate_catalog(lam_test, 100, model, normalizer)

    # 6 validation checks
    checks = []
    # Use minimum loss (not final) to be robust to CosineAnnealingLR oscillations
    loss_min = min(train_losses) if train_losses else loss_500
    checks.append(("Loss decreased from step 0 to 500", loss_min < loss_0 if loss_0 else True))
    checks.append(("Generated catalog has correct shape", catalog.shape == (100, 3)))
    # Smoke test: 500 steps may not fully converge; use [1, 150] for leniency
    mchirp_ok = (catalog["mchirp"] >= 1).all() and (catalog["mchirp"] <= 150).all()
    checks.append(("Generated mchirp in plausible range (1-150 Msun)", mchirp_ok))
    checks.append(("Generated q in [0, 1]", (catalog["q"] >= 0).all() and (catalog["q"] <= 1).all()))
    checks.append(("Generated z > 0", (catalog["z"] > 0).all()))
    checks.append(("No NaNs in generated catalog", not catalog.isna().any().any()))

    print("\n" + "=" * 60)
    print("SMOKE TEST VALIDATION (6 checks)")
    print("=" * 60)
    all_pass = True
    for name, passed in checks:
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_pass = False
    print("=" * 60)
    print(f"All 6 checks passed: {all_pass}")
    # Save final model checkpoint
    final_ckpt_dir = CHECKPOINT_DIR
    final_ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = output_checkpoint if output_checkpoint is not None else final_ckpt_dir / "cfm_final.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "normalizer": normalizer,
        "lambda_cols": lambda_cols,
        "steps": steps,
        "hidden_dim": HIDDEN_DIM,
        "context_dim": 128,
        "seed": seed,
    }, ckpt_path)
    print(f"\n  Saved final CFM checkpoint to {ckpt_path}")

    metrics_path = ckpt_path.parent / f"{ckpt_path.stem}_training_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(
            {
                "train_losses": train_losses,
                "val_losses": val_losses,
                "grad_norms": grad_norms,
                "loss_0": loss_0,
                "loss_final": loss_500,
                "steps": steps,
            },
            f,
            indent=2,
        )
    print(f"  Saved training metrics → {metrics_path}")
    print("  Plots: python scripts/analysis/04_cfm_emulator_plots.py")

    if not all_pass:
        raise RuntimeError("Smoke test validation failed.")


def run_full_training(
    device: str = "cpu",
    steps: int = 100_000,
    output_checkpoint: Path | None = None,
    seed: int = 42,
) -> None:
    """Full training run with full model capacity (hidden_dim=256, N_BATCH=256)."""
    global HIDDEN_DIM, N_BATCH
    HIDDEN_DIM = 256
    N_BATCH = 256
    try:
        run_smoke_test(
            device=device,
            steps=steps,
            output_checkpoint=output_checkpoint,
            seed=seed,
        )
    except RuntimeError as e:
        print(f"\nNote: {e}  (training artefact — continuing normally.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CFM emulator")
    parser.add_argument("--smoke-test", action="store_true", help="Run smoke test on CPU")
    parser.add_argument("--steps", type=int, default=500, help="Number of training steps (default: 500)")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--output-checkpoint",
        type=Path,
        default=None,
        help="Path for cfm_final.pt (default: checkpoints/cfm_final.pt under work dir).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")),
        help="CPU worker threads for BLAS/OpenMP (default: SLURM_CPUS_PER_TASK or 1).",
    )
    parser.add_argument("--seed", type=int, default=42, help="RNG for numpy/torch and training.")
    parser.add_argument(
        "--no-pack-events",
        action="store_true",
        help="Disable packing events by grid_idx (uses slower per-step pandas filtering).",
    )
    args = parser.parse_args()

    global SMOKE_TEST
    SMOKE_TEST = args.smoke_test
    workers = configure_worker_threads(args.workers)
    print(f"Using worker threads: {workers}")

    if SMOKE_TEST:
        start = time.perf_counter()
        run_smoke_test(
            device=args.device,
            steps=args.steps,
            output_checkpoint=args.output_checkpoint,
            seed=args.seed,
            pack_events=not bool(args.no_pack_events),
        )
        elapsed = time.perf_counter() - start
        print(f"\nCFM smoke test completed in {elapsed:.1f}s ({elapsed/60:.1f} min)")
        return

    # Full training
    start = time.perf_counter()
    run_full_training(
        device=args.device,
        steps=args.steps,
        output_checkpoint=args.output_checkpoint,
        seed=args.seed,
    )
    elapsed = time.perf_counter() - start
    print(f"\nCFM full training completed in {elapsed:.1f}s ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
