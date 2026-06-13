#!/usr/bin/env python3
"""
Combine K Step-5 posteriors: **log-mean of log p** and/or **mixture** sampling.
"""
from __future__ import annotations

import sys
from pathlib import Path as _Path

_ANALYSIS_DIR = _Path(__file__).resolve().parent
_PROJECT_ROOT = _ANALYSIS_DIR.parents[1]
for _p in (_PROJECT_ROOT, _ANALYSIS_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from plant_paths import (  # noqa: E402
    PROJECT_ROOT,
    REPO_ROOT,
    SCRIPTS_DIR,
    ensure_paths,
    find_data_dir,
    find_work_dir,
    load_posterior_network_module,
    plot_run_dir,
)

ensure_paths()

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch


from models.ensemble_posterior import (  # noqa: E402
    log_mean_of_log_probs,
    load_posterior_member,
    mixture_sample,
)
_05 = load_posterior_network_module()
build_events_6d = _05.build_events_6d
load_frozen_emulator = _05.load_frozen_emulator



def main() -> None:
    p = argparse.ArgumentParser(
        description="Ensemble of K posteriors: geometric-mean log-density and/or mixture samples."
    )
    p.add_argument(
        "--member-dirs",
        type=Path,
        nargs="+",
        required=True,
        help="K directories, each with posterior_network_best.pt + config JSON",
    )
    p.add_argument("--model", type=str, default="lite", choices=("lite", "full"))
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument(
        "--mode",
        type=str,
        default="sample",
        choices=("sample", "logprob", "both"),
    )
    p.add_argument(
        "--synthetic-bag",
        action="store_true",
        help="If set, build a small random event bag (ignores --events-csv) for quick tests.",
    )
    p.add_argument("--events-csv", type=Path, default=None, help="Same columns as 07_gwtc_*")
    p.add_argument("--emulator", type=str, default="cfm", choices=("cfm", "diffusion"))
    p.add_argument("--emulator-checkpoint", type=Path, default=None)
    p.add_argument("--num-samples", type=int, default=512, help="Total mixture samples (split across K).")
    p.add_argument("--output-dir", type=Path, default=None)
    args = p.parse_args()

    device = torch.device(args.device)
    out = args.output_dir
    if out is None:
        out = plot_run_dir(Path(__file__))
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    if args.synthetic_bag:
        L = 4
        mch = np.array([20.0, 35.0, 12.0, 50.0], np.float32)
        qv = np.array([0.7, 0.5, 0.6, 0.8], np.float32)
        zv = np.array([0.3, 0.5, 1.0, 0.2], np.float32)
    else:
        if not args.events_csv:
            sys.exit("Need --events-csv or --synthetic-bag")
        import pandas as pd

        df = pd.read_csv(args.events_csv)
        lower = {c.lower(): c for c in df.columns}

        def col(*names: str) -> str:
            for n in names:
                if n.lower() in lower:
                    return lower[n.lower()]
            sys.exit(f"Need one of {names}, got {list(df.columns)}")

        c_m1, c_m2 = col("m1", "mass_1", "mass1"), col("m2", "mass_2", "mass2")
        c_z = col("z", "redshift")
        m1 = df[c_m1].values.astype(np.float64)
        m2 = df[c_m2].values.astype(np.float64)
        mask = m1 < m2
        m1[mask], m2[mask] = m2[mask], m1[mask]
        mch = (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2
        qv = m2 / np.maximum(m1, 1e-9)
        zv = df[c_z].values.astype(np.float64)
        L = len(mch)

    emu_path = args.emulator_checkpoint
    if emu_path is None:
        emu_path = PROJECT_ROOT / "checkpoints" / (
            "cfm_final.pt" if args.emulator == "cfm" else "diffusion_final.pt"
        )
    emu_path = Path(emu_path).resolve()
    _, _, nrm = load_frozen_emulator(emu_path, device, args.emulator)
    x6 = build_events_6d(
        mch.astype(np.float64), qv.astype(np.float64), zv.astype(np.float64), nrm
    )
    ev = torch.from_numpy(x6).float().view(1, L, 6).to(device)
    msk = torch.ones(1, L, dtype=torch.float32, device=device)

    members = [
        load_posterior_member(d.resolve(), args.model, device) for d in args.member_dirs
    ]
    jmeta: dict = {
        "member_dirs": [str(d.resolve()) for d in args.member_dirs],
        "emulator_checkpoint": str(emu_path),
        "n_events": L,
    }

    if args.mode in ("logprob", "both"):
        # One reference Λ: training-set mean of the first member (physical units)
        tmean = members[0].theta_mean
        theta0 = tmean.view(1, -1)
        lml = log_mean_of_log_probs(members, theta0, ev, msk)
        jmeta["log_mean_log_prob"] = float(lml.item())
        with open(out / "log_mean_log_prob.txt", "w", encoding="utf-8") as f:
            f.write(
                f"(1/K) * sum log p_k(Θ_ref|C) = {float(lml.item()):.6e}\n"
                f"Θ_ref = first member's z-score center (train mean) in physical space.\n"
            )

    if args.mode in ("sample", "both"):
        s = mixture_sample(
            members, ev, msk, num_samples=int(args.num_samples)
        )  # (1, S, 9)
        sp = s[0].cpu().numpy()
        np.savez(out / "mixture_samples.npz", samples=sp)
        jmeta["num_samples"] = int(sp.shape[0])

    with open(out / "run_meta.json", "w", encoding="utf-8") as f:
        json.dump(jmeta, f, indent=2)
    print(f"Wrote: {out}")


if __name__ == "__main__":
    main()
