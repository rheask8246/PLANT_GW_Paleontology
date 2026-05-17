#!/usr/bin/env python3
"""
Validate a trained Step-5 posterior on a table of real (or mock) BBH events.

Expects CSV columns (case-insensitive): mass_1 / m1, mass_2 / m2, redshift / z.
Maps to (mchirp, q, z), builds 6-D features with the **emulator** normalizer
from `checkpoints/cfm_final.pt` (or `--emulator-checkpoint`), then runs `PosteriorNet.sample`.

Generates marginal histograms for the nine SSPC `sspc_*_mean` parameters and a summary CSV.
Channel (SMT/CE/CHE) is **not** predicted; the inverse model conditions only on the event set.
"""
from __future__ import annotations

import sys
from pathlib import Path as _Path

_PROJECT_ROOT = _Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from plant_paths import (  # noqa: E402
    PROJECT_ROOT,
    REPO_ROOT,
    SCRIPTS_DIR,
    ensure_paths,
    find_data_dir,
    find_work_dir,
    load_posterior_network_module,
)

ensure_paths()

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch


from models.ensemble_posterior import load_posterior_member
from models.posterior_network_lite import SSPC_THETA_PARAM_COLS

# Reuse featurization and emulator loader from Step 5 driver
_05 = load_posterior_network_module()
build_events_6d = _05.build_events_6d
load_frozen_emulator = _05.load_frozen_emulator

_DEFAULT_OUT = PROJECT_ROOT / "plots" / "gwtc_validation"


def _find_col(df: pd.DataFrame, options: Tuple[str, ...]) -> str:
    lower = {c.lower(): c for c in df.columns}
    for o in options:
        if o.lower() in lower:
            return lower[o.lower()]
    raise ValueError(f"Need one of columns {options}, got {list(df.columns)}")


def _mchirp(m1: np.ndarray, m2: np.ndarray) -> np.ndarray:
    return (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2


def main() -> None:
    p = argparse.ArgumentParser(description="GWTC-style catalog → posterior samples (Step 5 validation).")
    p.add_argument("--events-csv", type=Path, required=True, help="CSV with masses and redshift z")
    p.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=PROJECT_ROOT / "checkpoints",
        help="Directory with posterior_network_best.pt and posterior_network_config.json",
    )
    p.add_argument("--model", type=str, default="lite", choices=("lite", "full"))
    p.add_argument("--emulator", type=str, default="cfm", choices=("cfm", "diffusion"))
    p.add_argument("--emulator-checkpoint", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--num-samples", type=int, default=2000)
    p.add_argument("--device", type=str, default="cpu")
    args = p.parse_args()

    device = torch.device(args.device)
    if args.output_dir is None:
        out = _DEFAULT_OUT / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    else:
        out = args.output_dir
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.events_csv)
    c_m1 = _find_col(df, ("m1", "mass_1", "mass1"))
    c_m2 = _find_col(df, ("m2", "mass_2", "mass2"))
    c_z = _find_col(df, ("z", "redshift"))

    m1 = df[c_m1].values.astype(np.float64)
    m2 = df[c_m2].values.astype(np.float64)
    mask = m1 < m2
    m1[mask], m2[mask] = m2[mask], m1[mask]
    mch = _mchirp(m1, m2)
    qv = m2 / np.maximum(m1, 1e-9)
    zv = df[c_z].values.astype(np.float64)

    emu_path = args.emulator_checkpoint
    if emu_path is None:
        emu_path = PROJECT_ROOT / "checkpoints" / (
            "cfm_final.pt" if args.emulator == "cfm" else "diffusion_final.pt"
        )
    emu_path = Path(emu_path).resolve()
    _, _, nrm = load_frozen_emulator(emu_path, device, args.emulator)

    x6 = build_events_6d(mch, qv, zv, nrm)
    L = x6.shape[0]
    ev = torch.from_numpy(x6).float().view(1, L, 6).to(device)
    mask = torch.ones(1, L, dtype=torch.float32, device=device)

    post = load_posterior_member(
        Path(args.checkpoint_dir).resolve(), args.model, device
    )
    with torch.no_grad():
        samp = post.sample(ev, mask, num_samples=int(args.num_samples))  # (1, S, 9)
    s = samp[0].cpu().numpy()
    # Marginal figures
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    num_draws = s.shape[0]
    ncols = 3
    nrows = (len(SSPC_THETA_PARAM_COLS) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 2.2 * nrows))
    axes = np.array(axes).ravel()
    for i, col in enumerate(SSPC_THETA_PARAM_COLS):
        ax = axes[i]
        ax.hist(s[:, i], bins=40, density=True, alpha=0.85, color="steelblue", edgecolor="none")
        ax.set_title(col.replace("sspc_", "").replace("_mean", ""), fontsize=7)
    for j in range(len(SSPC_THETA_PARAM_COLS), len(axes)):
        fig.delaxes(axes[j])
    fig.suptitle(
        f"GWTC-style posterior samples (N={L} events, {num_draws} draws) — SSPC Λ",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out / "marginal_thetas.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    summary = pd.DataFrame(
        {
            "param": list(SSPC_THETA_PARAM_COLS),
            "mean": s.mean(axis=0),
            "std": s.std(axis=0),
            "p05": np.percentile(s, 5, axis=0),
            "p50": np.percentile(s, 50, axis=0),
            "p95": np.percentile(s, 95, axis=0),
        }
    )
    summary.to_csv(out / "theta_summary.csv", index=False)
    with open(out / "run_meta.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "events_csv": str(args.events_csv),
                "emulator_checkpoint": str(emu_path),
                "posterior_dir": str(Path(args.checkpoint_dir).resolve()),
                "n_events": L,
                "num_samples": int(args.num_samples),
            },
            f,
            indent=2,
        )
    stopgap = """# Channel (SMT / CE / CHE) and real GW events

Step 5 was trained on **synthetic** catalogs where the **formation channel** is fixed by
the hyperparameter table row (implicit in Λ). Real LIGO/Virgo events do not label channel
in the 6-D event vector this script feeds to the posterior.

**v1 in this repository:** a **single** run conditions only on
(mchirp, q, z) and measurement-error structure from the **same** normalizer
as the emulator checkpoint. **This is not a full marginalization over channel.**

A stopgap from the project plan (optional future work) is to compare **K = 3** runs that
differ in **emulator+posterior** pairs (one trained per channel) or to add an explicit
channel dimension to Λ. Neither is required for the CSV-based figures produced here.
"""
    (out / "gwtc_channel_stopgap.md").write_text(stopgap, encoding="utf-8")
    print(f"Wrote: {out / 'marginal_thetas.png'} and {out / 'theta_summary.csv'}")


if __name__ == "__main__":
    main()
