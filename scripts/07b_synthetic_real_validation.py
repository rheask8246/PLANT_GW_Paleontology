#!/usr/bin/env python3
"""
Compare Step-5 posterior marginals: **synthetic** event CSV vs **real** GW catalog CSV.

Produces the same style of marginal histograms as `07_gwtc_posterior_validate.py`, but
**overlays** two posterior sample sets (e.g. mock / training-style synthetic vs LIGO events).

**About `o4a-astro/data_release/` (and Zenodo):** the GWTC-4.0 population release
([Zenodo 16911563](https://zenodo.org/records/16911563); paper
[arXiv:2508.18083](https://arxiv.org/pdf/2508.18083)) provides **popsummary** HDF5 files
under ``data_release/`` — inferred **population** rate densities and hyperparameters for
collaboration mass/spin/redshift **models**, not a per-event table of
(m1, m2, χeff, z) for every merger. That data drives figures like the o4a-astro scripts;
it is **not** the input format for this posterior net (which expects a **list of events**).

Optional GWTC-3 **comparison curves** from the same record use ``download_gwtc3_data.py`` →
``gwtc3_data/`` (population JSON / auxiliary files), also **not** a per-event CSV.

For ``--real-events-csv``, use a **GW event catalog** (e.g. GWOSC GWTC-3 or GWTC-4 BBH
merger tables with source-frame masses and redshift).

**Synthetic arm (default):** sample a merger catalog from the **CFM or diffusion emulator**
at a row of ``hyperparam_table_encoded.csv``. If ``--synthetic-grid-idx`` is omitted, the
row is resolved from ``--synthetic-hyperparam-key`` (default: **SMT** cell nearest the
TNG100-1 SSPC reference in ``00_sspc_data_generation.py``, ``/SMT/sfra0157/mu00243``).
Optional ``--synthetic-csv`` overrides with a hand-written table (e.g. repo
``data/gwtc_sample_events.csv`` is only a tiny smoke-test placeholder, not tied to a Λ row).

Expected columns (case-insensitive; GWOSC-style aliases supported):

- Masses: ``mass_1``, ``m1``, ``mass_1_source``, …
- Redshift: ``z``, ``redshift``, ``cosmo_redshift``, …

Spin columns in a real CSV, if present, are **ignored** (the pipeline uses mchirp, q, z only).

Channel is not inferred (same caveat as Step 7).
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
    ensure_paths,
    load_posterior_network_module,
)

ensure_paths()

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch


from models.ensemble_posterior import load_posterior_member
from models.posterior_network_lite import SSPC_THETA_PARAM_COLS

_05 = load_posterior_network_module()
build_events_6d = _05.build_events_6d
load_frozen_emulator = _05.load_frozen_emulator

_DEFAULT_OUT = PROJECT_ROOT / "plots" / "synth_real_validation"

# Default emulator row: SMT channel, SSPC grid corner closest to TNG100-1 best-fit
# (sfr_a ≈ 0.017, μ₀ ≈ 0.025) on the 8×8 linspace grid — see 00_sspc_data_generation.py.
_DEFAULT_SMT_TNG_CENTER_KEY = "/SMT/sfra0157/mu00243"

# Column aliases: first match wins (case-insensitive keys in CSV)
_COL_M1 = ("m1", "mass_1", "mass1", "mass_1_source", "m1_source")
_COL_M2 = ("m2", "mass_2", "mass2", "mass_2_source", "m2_source")
_COL_Z = ("z", "redshift", "cosmo_redshift", "z_redshift")


def _find_col(df: pd.DataFrame, options: Tuple[str, ...]) -> str:
    lower = {c.lower(): c for c in df.columns}
    for o in options:
        if o.lower() in lower:
            return lower[o.lower()]
    raise ValueError(f"Need one of columns {options}, got {list(df.columns)}")


def resolve_synthetic_grid_row(
    hyperparam_encoded_csv: Path,
    *,
    grid_idx: Optional[int],
    hyperparam_key: str,
) -> Tuple[int, str, str]:
    """
    Return (row_index, hyperparam_key_for_meta, resolution_note).

    If ``grid_idx`` is not None, it wins and ``hyperparam_key`` is only recorded when it
    matches that row (otherwise stored as the requested key string for metadata).

    If ``grid_idx`` is None, look up ``hyperparam_key`` in column ``key`` (exact match).
    """
    hp = Path(hyperparam_encoded_csv).resolve()
    if grid_idx is not None:
        return int(grid_idx), "", f"explicit_index:{int(grid_idx)}"

    k = str(hyperparam_key).strip()
    if not k:
        raise ValueError("--synthetic-hyperparam-key must be non-empty when --synthetic-grid-idx is omitted")

    dfk = pd.read_csv(hp, usecols=["key"])
    match = dfk.index[dfk["key"] == k].tolist()
    if len(match) == 0:
        raise ValueError(
            f"No row with key={k!r} in {hp}\n"
            "Pass --synthetic-grid-idx explicitly or fix --synthetic-hyperparam-key."
        )
    if len(match) > 1:
        raise ValueError(f"Multiple rows with key={k!r} in {hp}: indices {match[:8]}…")
    return int(match[0]), k, f"key:{k}"


def _mchirp(m1: np.ndarray, m2: np.ndarray) -> np.ndarray:
    return (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2


def _m1_from_mchirp_q(mchirp: np.ndarray, q: np.ndarray) -> np.ndarray:
    # Same map as gwtc4_validation / data_distribution_analysis
    qv = np.clip(q.astype(np.float64), 1e-6, 1.0)
    mc = np.clip(mchirp.astype(np.float64), 1e-6, None)
    return mc * ((1.0 + qv) / np.power(qv, 3.0)) ** 0.2


def _m1m2_from_catalog_df(cat: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    mchirp = cat["mchirp"].values.astype(np.float64)
    q = cat["q"].values.astype(np.float64)
    m1 = _m1_from_mchirp_q(mchirp, q)
    m2 = np.clip(q, 0.0, 1.0) * m1
    swap = m2 > m1
    if np.any(swap):
        m1s = m1.copy()
        m1[swap] = m2[swap]
        m2[swap] = m1s[swap]
    return m1, m2


def _generate_catalog(emulator: str, lambda_vec: np.ndarray, n_events: int, model, normalizer: Dict[str, Any]):
    if emulator == "cfm":
        from models.cfm_emulator import generate_catalog as gc
    elif emulator == "diffusion":
        from models.diffusion_emulator import generate_catalog as gc
    else:
        raise ValueError(f"Unknown emulator {emulator!r}")
    return gc(np.asarray(lambda_vec, dtype=np.float32), int(n_events), model, normalizer)


def synthetic_events_from_emulator(
    *,
    emulator: str,
    emulator_checkpoint: Path,
    device: torch.device,
    hyperparam_encoded_csv: Path,
    grid_idx: int,
    n_events: int,
    seed: int,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    hp = pd.read_csv(hyperparam_encoded_csv)
    n = len(hp)
    if n == 0:
        raise ValueError(f"Empty hyperparameter table: {hyperparam_encoded_csv}")
    if grid_idx < 0 or grid_idx >= n:
        raise IndexError(f"--synthetic-grid-idx {grid_idx} out of range for {n} rows in {hyperparam_encoded_csv}")

    model, lambda_cols, nrm = load_frozen_emulator(emulator_checkpoint, device, emulator)
    missing = [c for c in lambda_cols if c not in hp.columns]
    if missing:
        raise ValueError(
            f"hyperparam CSV missing lambda columns {missing}; have {list(hp.columns)}"
        )

    row = hp.iloc[int(grid_idx)]
    lambda_vec = row[list(lambda_cols)].values.astype(np.float32)

    torch.manual_seed(int(seed))
    cat = _generate_catalog(emulator, lambda_vec, n_events, model, nrm)
    m1, m2 = _m1m2_from_catalog_df(cat)
    df = pd.DataFrame(
        {
            "mass_1": m1,
            "mass_2": m2,
            "z": cat["z"].values.astype(np.float64),
        }
    )
    meta: Dict[str, Any] = {
        "source": "emulator",
        "emulator": emulator,
        "hyperparam_encoded_csv": str(Path(hyperparam_encoded_csv).resolve()),
        "synthetic_grid_idx": int(grid_idx),
        "n_synthetic_events": int(n_events),
        "synthetic_seed": int(seed),
        "lambda_columns": list(lambda_cols),
    }
    return df, meta


def _events_from_dataframe(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    c_m1 = _find_col(df, _COL_M1)
    c_m2 = _find_col(df, _COL_M2)
    c_z = _find_col(df, _COL_Z)
    m1 = df[c_m1].values.astype(np.float64)
    m2 = df[c_m2].values.astype(np.float64)
    swap = m1 < m2
    m1[swap], m2[swap] = m2[swap], m1[swap]
    mch = _mchirp(m1, m2)
    qv = m2 / np.maximum(m1, 1e-9)
    zv = df[c_z].values.astype(np.float64)
    return mch, qv, zv


def sample_theta_posterior(
    df: pd.DataFrame,
    *,
    checkpoint_dir: Path,
    model: str,
    emulator: str,
    emulator_checkpoint: Path,
    device: torch.device,
    num_samples: int,
) -> Tuple[np.ndarray, int]:
    """Return posterior samples (S, 9) and number of events L."""
    mch, qv, zv = _events_from_dataframe(df)
    _, _, nrm = load_frozen_emulator(emulator_checkpoint, device, emulator)
    x6 = build_events_6d(mch, qv, zv, nrm)
    L = x6.shape[0]
    ev = torch.from_numpy(x6).float().view(1, L, 6).to(device)
    mask = torch.ones(1, L, dtype=torch.float32, device=device)
    post = load_posterior_member(checkpoint_dir.resolve(), model, device)
    with torch.no_grad():
        samp = post.sample(ev, mask, num_samples=int(num_samples))
    s = samp[0].cpu().numpy()
    return s, L


def _theta_summary(s: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "param": list(SSPC_THETA_PARAM_COLS),
            "mean": s.mean(axis=0),
            "std": s.std(axis=0),
            "p05": np.percentile(s, 5, axis=0),
            "p50": np.percentile(s, 50, axis=0),
            "p95": np.percentile(s, 95, axis=0),
        }
    )


def _plot_compare(
    s_synth: np.ndarray,
    s_real: np.ndarray,
    *,
    n_synth_events: int,
    n_real_events: int,
    num_samples: int,
    out_path: Path,
    synthetic_label: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ncols = 3
    nrows = (len(SSPC_THETA_PARAM_COLS) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(11, 2.3 * nrows))
    axes = np.array(axes).ravel()

    for i, col in enumerate(SSPC_THETA_PARAM_COLS):
        ax = axes[i]
        lo = min(s_synth[:, i].min(), s_real[:, i].min())
        hi = max(s_synth[:, i].max(), s_real[:, i].max())
        if hi <= lo:
            hi = lo + 1e-6
        bins = np.linspace(lo, hi, 36)
        ax.hist(
            s_synth[:, i],
            bins=bins,
            density=True,
            alpha=0.55,
            color="steelblue",
            edgecolor="none",
            label=synthetic_label,
        )
        ax.hist(
            s_real[:, i],
            bins=bins,
            density=True,
            alpha=0.55,
            color="coral",
            edgecolor="none",
            label="Real GW catalog",
        )
        ax.set_title(col.replace("sspc_", "").replace("_mean", ""), fontsize=8)
        if i == 0:
            ax.legend(fontsize=7, loc="upper right")

    for j in range(len(SSPC_THETA_PARAM_COLS), len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle(
        f"Posterior Λ marginals: synthetic (N={n_synth_events} ev.) vs real (N={n_real_events} ev.), "
        f"{num_samples} draws each",
        fontsize=10,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Overlay Step-5 posterior marginals: synthetic events vs real GW catalog CSV."
    )
    p.add_argument(
        "--synthetic-csv",
        type=Path,
        default=None,
        help="Optional: read synthetic events from CSV (same schema as 07). If omitted, build a catalog "
        "from the emulator (default hyperparam key: TNG-centered SMT; see --synthetic-hyperparam-key).",
    )
    p.add_argument(
        "--synthetic-grid-idx",
        type=int,
        default=None,
        help="Row index into hyperparam_table_encoded.csv. If omitted, "
        "`--synthetic-hyperparam-key` is looked up (default key: TNG-centered SMT cell).",
    )
    p.add_argument(
        "--synthetic-hyperparam-key",
        type=str,
        default=_DEFAULT_SMT_TNG_CENTER_KEY,
        help="Hyperparam table `key` column (e.g. /SMT/sfra0157/mu00243). Used only when "
        "--synthetic-grid-idx is not set.",
    )
    p.add_argument(
        "--hyperparam-encoded-csv",
        type=Path,
        default=HYPERPARAM_TABLE_ENCODED_CSV,
        help="Encoded hyperparameter table (must contain emulator lambda_* columns).",
    )
    p.add_argument(
        "--n-synthetic-events",
        type=int,
        default=256,
        help="Number of mergers to draw from the emulator when not using --synthetic-csv.",
    )
    p.add_argument(
        "--synthetic-seed",
        type=int,
        default=0,
        help="RNG seed for emulator catalog generation.",
    )
    p.add_argument(
        "--real-events-csv",
        type=Path,
        required=True,
        help="Real GW catalog CSV (GWOSC GWTC-3 style). o4a-astro/gwtc3_data/ has population JSONs only.",
    )
    p.add_argument("--checkpoint-dir", type=Path, default=CHECKPOINT_DIR)
    p.add_argument("--model", type=str, default="full", choices=("lite", "full"))
    p.add_argument("--emulator", type=str, default="cfm", choices=("cfm", "diffusion"))
    p.add_argument("--emulator-checkpoint", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--num-samples", type=int, default=2000)
    p.add_argument("--device", type=str, default="cpu")
    args = p.parse_args()

    device = torch.device(args.device)
    real_path = Path(args.real_events_csv).resolve()
    if not real_path.is_file():
        raise FileNotFoundError(
            f"Real events CSV not found: {real_path}\n"
            "Use a GWOSC (or similar) **event catalog** CSV with m1, m2, chi_eff, z.\n"
            "o4a-astro/data_release/ (Zenodo 16911563 popsummary HDF5) is for population\n"
            "inference products — not a substitute for per-event tables for this script."
        )

    emu_path = args.emulator_checkpoint
    if emu_path is None:
        emu_path = CHECKPOINT_DIR / (
            "cfm_final.pt" if args.emulator == "cfm" else "diffusion_final.pt"
        )
    emu_path = Path(emu_path).resolve()
    ckpt_dir = Path(args.checkpoint_dir).resolve()

    out = (
        args.output_dir.resolve()
        if args.output_dir
        else _DEFAULT_OUT / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )
    out.mkdir(parents=True, exist_ok=True)

    synth_meta: Dict[str, Any]
    synthetic_label: str
    if args.synthetic_csv is not None:
        synth_path = Path(args.synthetic_csv).resolve()
        if not synth_path.is_file():
            raise FileNotFoundError(f"Synthetic CSV not found: {synth_path}")
        df_s = pd.read_csv(synth_path)
        synth_meta = {"source": "csv", "synthetic_csv": str(synth_path)}
        synthetic_label = "Synthetic (CSV)"
    else:
        synth_path = None
        hp_enc = Path(args.hyperparam_encoded_csv).resolve()
        if not hp_enc.is_file():
            raise FileNotFoundError(
                f"Hyperparameter table not found: {hp_enc}\n"
                "Pass --hyperparam-encoded-csv or use --synthetic-csv instead."
            )
        if int(args.n_synthetic_events) < 1:
            raise ValueError("--n-synthetic-events must be >= 1")
        row_i, key_used, res_note = resolve_synthetic_grid_row(
            hp_enc,
            grid_idx=args.synthetic_grid_idx,
            hyperparam_key=str(args.synthetic_hyperparam_key),
        )
        loc = f"row {row_i}" + (f" ({key_used})" if key_used else "")
        print(
            f"[07b] Drawing {args.n_synthetic_events} synthetic mergers ({args.emulator}) "
            f"at hyperparam {loc} [{res_note}] …",
            flush=True,
        )
        df_s, synth_meta = synthetic_events_from_emulator(
            emulator=args.emulator,
            emulator_checkpoint=emu_path,
            device=device,
            hyperparam_encoded_csv=hp_enc,
            grid_idx=int(row_i),
            n_events=int(args.n_synthetic_events),
            seed=int(args.synthetic_seed),
        )
        synth_meta["synthetic_grid_resolution"] = res_note
        if key_used:
            synth_meta["synthetic_hyperparam_key"] = key_used

        if key_used:
            synthetic_label = f"{args.emulator.upper()} ({key_used})"
        else:
            synthetic_label = f"{args.emulator.upper()} catalog (Λ row {row_i})"
        df_s.to_csv(out / "synthetic_catalog_used.csv", index=False)

    df_r = pd.read_csv(real_path)

    print("[07b] Sampling posterior for synthetic catalog …", flush=True)
    s_synth, Ls = sample_theta_posterior(
        df_s,
        checkpoint_dir=ckpt_dir,
        model=args.model,
        emulator=args.emulator,
        emulator_checkpoint=emu_path,
        device=device,
        num_samples=args.num_samples,
    )
    print("[07b] Sampling posterior for real catalog …", flush=True)
    s_real, Lr = sample_theta_posterior(
        df_r,
        checkpoint_dir=ckpt_dir,
        model=args.model,
        emulator=args.emulator,
        emulator_checkpoint=emu_path,
        device=device,
        num_samples=args.num_samples,
    )

    _plot_compare(
        s_synth,
        s_real,
        n_synth_events=Ls,
        n_real_events=Lr,
        num_samples=int(args.num_samples),
        out_path=out / "marginal_thetas_synthetic_vs_real.png",
        synthetic_label=synthetic_label,
    )

    _theta_summary(s_synth).to_csv(out / "theta_summary_synthetic.csv", index=False)
    _theta_summary(s_real).to_csv(out / "theta_summary_real.csv", index=False)

    note = """# Data sources vs `07b_synthetic_real_validation.py`

## `o4a-astro/data_release/` (your path on Expanse)

This is where you unpack **GWTC-4.0 population** assets from
[Zenodo 16911563](https://zenodo.org/records/16911563) (see also
[GWTC-4.0 population paper, arXiv:2508.18083](https://arxiv.org/pdf/2508.18083)).
Files there are **`popsummary` HDF5** population results (e.g. `AllCBC_*.h5`,
`BBHMassSpinRedshift_*.h5`) — **not** a CSV of individual events for the amortized
posterior in this repo.

## `gwtc3_data/` (optional, same Zenodo README)

`download_gwtc3_data.py` from that record pulls **GWTC-3 comparison** material (e.g.
PowerLawPeak JSON). Again: **population / plotting aids**, not the per-event catalog
needed here.

## What `07b` needs

**Synthetic (default):** the CFM/diffusion emulator draws a catalog at a row of
`hyperparam_table_encoded.csv`, defaulting to the TNG-centered **SMT** key
`/SMT/sfra0157/mu00243` unless `--synthetic-grid-idx` or `--synthetic-hyperparam-key`
is set. Optional `--synthetic-csv` overrides with a hand-written table.

`--real-events-csv`: a table of **mergers** with (primary mass, secondary mass,
effective spin, redshift), e.g. from **GWOSC** GWTC-3 or GWTC-4 BBH catalogs. Rename
columns to match `mass_1`, `mass_2`, `chi_eff`, `z` if needed, or use GWOSC-style names
listed in the script docstring.
"""
    (out / "data_source_note.md").write_text(note, encoding="utf-8")

    run_meta: Dict[str, Any] = {
        "real_events_csv": str(real_path),
        "n_events_synthetic": Ls,
        "n_events_real": Lr,
        "num_samples": int(args.num_samples),
        "emulator": args.emulator,
        "emulator_checkpoint": str(emu_path),
        "posterior_dir": str(ckpt_dir),
        "model": args.model,
        "synthetic": synth_meta,
    }
    if synth_path is not None:
        run_meta["synthetic_csv"] = str(synth_path)
    with open(out / "run_meta.json", "w", encoding="utf-8") as f:
        json.dump(run_meta, f, indent=2)

    print(f"Wrote: {out / 'marginal_thetas_synthetic_vs_real.png'}", flush=True)
    print(f"Wrote: {out / 'theta_summary_synthetic.csv'}", flush=True)
    print(f"Wrote: {out / 'theta_summary_real.csv'}", flush=True)


if __name__ == "__main__":
    main()
