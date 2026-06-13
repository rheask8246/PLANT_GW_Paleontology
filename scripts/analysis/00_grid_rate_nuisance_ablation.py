#!/usr/bin/env python3
"""
Step 00 — Nuisance ablation heatmaps on the (sfr_a, mu0) grid.

For each of the seven SSPC nuisance parameters, recompute intrinsic merger rate
density (same numerics as ``00_sspc_data_generation.py``) while:

  - varying ``sfr_a`` and ``mu0`` on the Step 00 grid ranges (SLURM default 20×20);
  - holding the other six nuisances at Step-00 fixed best-fit values;
  - either (a) sweeping the ablated nuisance over its ``NUISANCE_RANGES`` from Step 00
    and aggregating (default), or (b) sampling the ablated nuisance once per grid cell
    (``--ablation-sample``), matching Step-00 "random nuisance" style but with only
    one nuisance randomized at a time.

Each (sfr_a, mu0, channel) cell uses the **same rate metric as Step 02** on a
**sampled** mini-catalog (default ``--n-events 5000``, matching ``02_build_dataset``
``N_SAMPLE``): cosmic integration → weighted event draw →
``intrinsic_rate_yr = n_events × Σw² / Σw``, then ``rate_per_gpc3_yr``.

Per-parameter ablation panels (default) average that Step-02 rate over ``--n-nuisance``
values of the swept nuisance (others at TNG100 best-fit). With ``--ablation-sample``,
each cell uses a single uniform draw of the ablated nuisance and no sweep. Baselines:

  - ``all_sampled``: one nuisance draw per cell from ``NUISANCE_RANGES`` (Step 00 random mode).
  - ``all_fixed``: all nuisances at TNG100 best-fit (Step 00 ``--fixed-nuisance-tng100``).

SLURM: ``slurm/00_grid_rate_nuisance_ablation_array.sh`` (9 parallel tasks)
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import multiprocessing as mp
import os
import sys
import warnings
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Literal, Sequence, Tuple, Union, cast

_ANALYSIS_DIR = Path(__file__).resolve().parent
if str(_ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(_ANALYSIS_DIR))

from _bootstrap import setup  # noqa: E402

setup()

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lib.grid_heatmap_plot import pcolormesh_sfra_mu0  # noqa: E402
from plant_paths import resolve_plot_output  # noqa: E402
from sspc_param_ranges import (  # noqa: E402
    DEFAULT_N_MU0,
    DEFAULT_N_SFRA,
    MU0_RANGE,
    SFRA_RANGE,
    linspace_grid,
)

MetricName = Literal["rate", "log_rate"]
AggregateMode = Literal["mean", "std", "min", "max"]
ColorScaleMode = Literal["log", "linear"]
ColormapStyle = Literal["sequential", "diverging"]

CHANNELS = ("SMT", "CE", "CHE")

# Populated in the parent before fork-based worker pools (Linux/HPC).
_POOL: Dict[str, Any] = {}

NUISANCE_LATEX: Dict[str, str] = {
    "sfr_b": r"$b$",
    "sfr_c": r"$c$",
    "sfr_d": r"$d$",
    "muz": r"$\mu_z$",
    "sigma0": r"$\sigma_0$",
    "sigmaz": r"$\sigma_z$",
    "alpha_skew": r"$\alpha$",
}

# Per-parameter ablation targets (one nuisance swept, others at best-fit).
ABLATION_NUISANCES: Tuple[str, ...] = tuple(NUISANCE_LATEX.keys())

# Baseline panels for the 9-task SLURM array.
BASELINE_ALL_SAMPLED = "all_sampled"
BASELINE_ALL_FIXED = "all_fixed"

ABLATION_TARGETS: Tuple[str, ...] = ABLATION_NUISANCES + (
    BASELINE_ALL_SAMPLED,
    BASELINE_ALL_FIXED,
)

ComputeMode = Literal["ablate", "all_sampled", "all_fixed"]


def _load_sspc00_module():
    from plant_paths import PROJECT_ROOT

    path = PROJECT_ROOT / "scripts" / "00_sspc_data_generation.py"
    spec = importlib.util.spec_from_file_location("_plant_sspc00", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_heatmaps_module():
    path = _ANALYSIS_DIR / "00_grid_rate_heatmaps.py"
    spec = importlib.util.spec_from_file_location("_plant_grid_rate_heatmaps", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def nuisance_keys(mod: Any) -> List[str]:
    return list(mod.NUISANCE_RANGES.keys())


def fixed_nuisance_params(mod: Any) -> Dict[str, float]:
    return mod.nuisance_tng100_params()


def _stable_seed(*parts: object) -> int:
    """Process-stable RNG seed (unlike built-in ``hash``)."""
    tokens: List[int] = []
    for part in parts:
        if isinstance(part, float):
            tokens.append(int(round(part * 1_000_000)))
        elif isinstance(part, str):
            tokens.append(sum(ord(c) for c in part) % (2**20))
        else:
            tokens.append(int(part))
    return int(np.random.SeedSequence(tokens).generate_state(1)[0])


def sample_nuisances_for_cell(
    mod: Any,
    sfr_a: float,
    mu0: float,
    *,
    seed: int,
    sample_index: int = 0,
) -> Dict[str, float]:
    """One uniform draw per nuisance on (sfr_a, mu0), matching Step 00 random mode."""
    rng = np.random.default_rng(
        _stable_seed("nuisance", sfr_a, mu0, seed, sample_index)
    )
    return {
        key: float(rng.uniform(*mod.NUISANCE_RANGES[key]))
        for key in nuisance_keys(mod)
    }


def sample_single_nuisance_for_cell(
    mod: Any,
    nuisance: str,
    sfr_a: float,
    mu0: float,
    *,
    seed: int,
) -> float:
    """Uniform draw for one nuisance on (sfr_a, mu0) with stable seeding."""
    rng = np.random.default_rng(_stable_seed("nuisance_one", nuisance, sfr_a, mu0, seed))
    lo, hi = mod.NUISANCE_RANGES[nuisance]
    return float(rng.uniform(lo, hi))


def target_compute_mode(target: str) -> ComputeMode:
    if target == BASELINE_ALL_SAMPLED:
        return "all_sampled"
    if target == BASELINE_ALL_FIXED:
        return "all_fixed"
    if target in ABLATION_NUISANCES:
        return "ablate"
    raise ValueError(f"unknown ablation target {target!r}")


def grid_values(
    mod: Any,
    axis: Literal["sfra", "mu0", "nuisance"],
    n: int,
    *,
    nuisance: str = "",
) -> np.ndarray:
    if axis == "sfra":
        return linspace_grid("sfr_a", n)
    if axis == "mu0":
        return linspace_grid("mu0", n)
    if axis == "nuisance":
        return linspace_grid(nuisance, n)
    raise ValueError(axis)


def channel_rates_like_step02(
    mod: Any,
    bps_ch: Any,
    cosmology: Tuple[np.ndarray, np.ndarray, np.ndarray, float],
    *,
    sfr_a: float,
    mu0: float,
    nuisances: Dict[str, float],
    logZ_bounds: Tuple[float, float],
    n_events: int,
    sample_index: int,
    sample_seed: int,
    chunk_size: int,
    v_gpc3: float,
    channel: str,
) -> Tuple[float, float, float]:
    """
    Match ``02_build_dataset.build_hyperparam_table`` SSPC rate columns on a sampled catalog.

    Same flow as Step 00 per grid cell: ``compute_merger_weights`` → weighted resampling
    of ``n_events`` binaries → ``intrinsic_rate_yr = n × Σw²/Σw``.
    """
    redshifts, times_Myr, shell_volumes, time_first_SF = cosmology
    n_ch = len(bps_ch)
    if n_ch == 0:
        return 0.0, 0.0, 0.0

    sfr_z = mod.find_sfr(
        redshifts,
        float(sfr_a),
        nuisances["sfr_b"],
        nuisances["sfr_c"],
        nuisances["sfr_d"],
    )
    logZ_min, logZ_max = logZ_bounds
    dPdlogZ, met_grid, p_draw = mod.find_metallicity_distribution(
        redshifts,
        logZ_min,
        logZ_max,
        mu0=float(mu0),
        muz=nuisances["muz"],
        sigma0=nuisances["sigma0"],
        sigmaz=nuisances["sigmaz"],
        alpha=nuisances["alpha_skew"],
    )

    weight, _pz = mod.compute_merger_weights(
        bps_ch["delay_time"].values,
        bps_ch["metallicity"].values,
        bps_ch["formation_efficiency_per_solar_mass"].values,
        sfr_z,
        dPdlogZ,
        met_grid,
        p_draw,
        times_Myr,
        redshifts,
        shell_volumes,
        time_first_SF,
        chunk_size=chunk_size,
    )
    weight = np.where(np.isfinite(weight), weight, 0.0)
    total_rate = float(np.sum(weight))
    if total_rate < 1e-30:
        return 0.0, 0.0, 0.0

    rng = np.random.default_rng(
        _stable_seed("sample", channel, sfr_a, mu0, sample_seed, sample_index)
    )
    prob = weight / total_rate
    ev_idx = rng.choice(n_ch, size=int(n_events), replace=True, p=prob)
    w_ev = weight[ev_idx].astype(np.float64)
    sum_weight = float(np.sum(w_ev))
    if sum_weight <= 0.0:
        return 0.0, 0.0, 0.0
    sum_weight_sq = float(np.sum(w_ev * w_ev))
    intrinsic_rate_yr = float(n_events) * sum_weight_sq / sum_weight
    rate_per_gpc3_yr = intrinsic_rate_yr / max(float(v_gpc3), 1e-30)
    return sum_weight, intrinsic_rate_yr, rate_per_gpc3_yr


def mask_edges_inplace(hp: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with the min-mu0 column and min-sfra row set to NaN."""
    out = hp.copy()
    mu0_min = float(out["mu0"].min())
    sfra_min = float(out["sfra"].min())
    edge = (out["mu0"].astype(float) == mu0_min) | (out["sfra"].astype(float) == sfra_min)
    for col in ("rate_per_gpc3_yr", "intrinsic_rate_yr"):
        if col in out.columns:
            out.loc[edge, col] = np.nan
    return out


def aggregate_over_nuisance(values: np.ndarray, mode: AggregateMode) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    if mode == "mean":
        return float(np.mean(finite))
    if mode == "std":
        return float(np.std(finite))
    if mode == "min":
        return float(np.min(finite))
    if mode == "max":
        return float(np.max(finite))
    raise ValueError(mode)


def _worker_init() -> None:
    """One integration thread per process when using a process pool."""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"


def _compute_grid_cell(sfr_a: float, mu0: float, ch: str) -> dict:
    """One (sfr_a, mu0, channel) cell using Step-02-style sampled-catalog rates."""
    pool = _POOL
    mod = pool["mod"]
    mode: ComputeMode = pool["mode"]
    v_gpc3 = pool["v_gpc3"]
    chunk_size = pool["chunk_size"]
    n_events = int(pool["n_events"])
    sample_seed = int(pool["sample_seed"])
    bps_ch = pool["bps_by_channel"][ch]
    cosmology = pool["cosmology"]
    logZ_bounds = pool["logZ_by_channel"][ch]
    target = pool["target"]
    ablation_sample_one = bool(pool.get("ablation_sample_one", False))

    if mode == "all_fixed":
        nuisances = dict(pool["base_nuisances"])
        rate_samples = np.empty(1, dtype=np.float64)
        _sum_w, intrinsic_rate_yr, rate_per_gpc3_yr = channel_rates_like_step02(
            mod,
            bps_ch,
            cosmology,
            sfr_a=float(sfr_a),
            mu0=float(mu0),
            nuisances=nuisances,
            logZ_bounds=logZ_bounds,
            n_events=n_events,
            sample_index=0,
            sample_seed=sample_seed,
            chunk_size=chunk_size,
            v_gpc3=v_gpc3,
            channel=ch,
        )
        rate_samples[0] = rate_per_gpc3_yr
    elif mode == "all_sampled":
        nuisances = sample_nuisances_for_cell(
            mod, sfr_a, mu0, seed=sample_seed, sample_index=0
        )
        rate_samples = np.empty(1, dtype=np.float64)
        _sum_w, intrinsic_rate_yr, rate_per_gpc3_yr = channel_rates_like_step02(
            mod,
            bps_ch,
            cosmology,
            sfr_a=float(sfr_a),
            mu0=float(mu0),
            nuisances=nuisances,
            logZ_bounds=logZ_bounds,
            n_events=n_events,
            sample_index=0,
            sample_seed=sample_seed,
            chunk_size=chunk_size,
            v_gpc3=v_gpc3,
            channel=ch,
        )
        rate_samples[0] = rate_per_gpc3_yr
    else:
        ablated = cast(str, pool["ablated"])
        base_nuisances = pool["base_nuisances"]
        aggregate = pool["aggregate"]

        if ablation_sample_one:
            nuisances = dict(base_nuisances)
            nuisances[ablated] = sample_single_nuisance_for_cell(
                mod, ablated, sfr_a, mu0, seed=sample_seed
            )
            rate_samples = np.empty(1, dtype=np.float64)
            _sum_w, _intr, rate_gpc3 = channel_rates_like_step02(
                mod,
                bps_ch,
                cosmology,
                sfr_a=float(sfr_a),
                mu0=float(mu0),
                nuisances=nuisances,
                logZ_bounds=logZ_bounds,
                n_events=n_events,
                sample_index=0,
                sample_seed=sample_seed,
                chunk_size=chunk_size,
                v_gpc3=v_gpc3,
                channel=ch,
            )
            rate_samples[0] = rate_gpc3
            rate_per_gpc3_yr = float(rate_gpc3)
            intrinsic_rate_yr = rate_per_gpc3_yr * max(float(v_gpc3), 1e-30)
        else:
            nuisance_vals = pool["nuisance_vals"]
            rate_samples = np.empty(len(nuisance_vals), dtype=np.float64)
            for j, nv in enumerate(nuisance_vals):
                nuisances = dict(base_nuisances)
                nuisances[ablated] = float(nv)
                _sum_w, _intr, rate_gpc3 = channel_rates_like_step02(
                    mod,
                    bps_ch,
                    cosmology,
                    sfr_a=float(sfr_a),
                    mu0=float(mu0),
                    nuisances=nuisances,
                    logZ_bounds=logZ_bounds,
                    n_events=n_events,
                    sample_index=int(j),
                    sample_seed=sample_seed,
                    chunk_size=chunk_size,
                    v_gpc3=v_gpc3,
                    channel=ch,
                )
                rate_samples[j] = rate_gpc3
            rate_per_gpc3_yr = aggregate_over_nuisance(rate_samples, aggregate)
            intrinsic_rate_yr = rate_per_gpc3_yr * max(float(v_gpc3), 1e-30)

    return {
        "channel": ch,
        "sfra": float(sfr_a),
        "mu0": float(mu0),
        "ablated_nuisance": target,
        "intrinsic_rate_yr": float(intrinsic_rate_yr),
        "rate_per_gpc3_yr": float(rate_per_gpc3_yr),
        "n_events_sampled": n_events,
    }


def _compute_nuisance_slice(
    sfr_a: float,
    mu0: float,
    ch: str,
    nuisance_index: int,
) -> dict:
    """One ablated-nuisance integration + sample (parallel building block)."""
    pool = _POOL
    mod = pool["mod"]
    v_gpc3 = pool["v_gpc3"]
    chunk_size = pool["chunk_size"]
    n_events = int(pool["n_events"])
    sample_seed = int(pool["sample_seed"])
    bps_ch = pool["bps_by_channel"][ch]
    cosmology = pool["cosmology"]
    logZ_bounds = pool["logZ_by_channel"][ch]
    ablated = pool["ablated"]
    nuisance_vals = pool["nuisance_vals"]
    base_nuisances = pool["base_nuisances"]
    ablation_sample_one = bool(pool.get("ablation_sample_one", False))

    if ablation_sample_one:
        raise RuntimeError(
            "Internal error: _compute_nuisance_slice should not be used when "
            "ablation_sample_one is enabled."
        )
    nuisances = dict(base_nuisances)
    nuisances[ablated] = float(nuisance_vals[nuisance_index])
    _sum_w, _intr, rate_gpc3 = channel_rates_like_step02(
        mod,
        bps_ch,
        cosmology,
        sfr_a=float(sfr_a),
        mu0=float(mu0),
        nuisances=nuisances,
        logZ_bounds=logZ_bounds,
        n_events=n_events,
        sample_index=int(nuisance_index),
        sample_seed=sample_seed,
        chunk_size=chunk_size,
        v_gpc3=v_gpc3,
        channel=ch,
    )
    return {
        "channel": ch,
        "sfra": float(sfr_a),
        "mu0": float(mu0),
        "nuisance_index": int(nuisance_index),
        "rate_per_gpc3_yr": float(rate_gpc3),
    }


def _aggregate_slice_rows(
    partials: List[dict],
    *,
    target: str,
    aggregate: AggregateMode,
    v_gpc3: float,
    n_events: int,
) -> List[dict]:
    """Combine per-nuisance slice rows into one heatmap row per (sfra, mu0, channel)."""
    grouped: Dict[Tuple[str, float, float], List[float]] = defaultdict(list)
    for row in partials:
        key = (str(row["channel"]), float(row["sfra"]), float(row["mu0"]))
        grouped[key].append(float(row["rate_per_gpc3_yr"]))

    out: List[dict] = []
    for (ch, sfr_a, mu0), rates in grouped.items():
        rate_arr = np.asarray(rates, dtype=np.float64)
        rate_per_gpc3_yr = aggregate_over_nuisance(rate_arr, aggregate)
        intrinsic_rate_yr = rate_per_gpc3_yr * max(float(v_gpc3), 1e-30)
        out.append(
            {
                "channel": ch,
                "sfra": sfr_a,
                "mu0": mu0,
                "ablated_nuisance": target,
                "intrinsic_rate_yr": float(intrinsic_rate_yr),
                "rate_per_gpc3_yr": float(rate_per_gpc3_yr),
                "n_events_sampled": int(n_events),
            }
        )
    return out


GridTask = Tuple[float, float, str, int]


def _iter_grid_tasks(
    sfra_vals: Sequence[float],
    mu0_vals: Sequence[float],
    bps_by_channel: Dict[str, Any],
    *,
    mode: ComputeMode,
    n_nuisance_slices: int,
    ablation_sample_one: bool,
) -> List[GridTask]:
    """(sfr_a, mu0, channel, nuisance_index); index -1 = full cell (baselines)."""
    tasks: List[GridTask] = []
    for sfr_a in sfra_vals:
        for mu0 in mu0_vals:
            for ch in CHANNELS:
                if ch not in bps_by_channel or len(bps_by_channel[ch]) == 0:
                    warnings.warn(f"No BPS systems for channel {ch!r}", stacklevel=2)
                    continue
                if mode == "ablate":
                    if ablation_sample_one:
                        tasks.append((float(sfr_a), float(mu0), ch, -1))
                        continue
                    for j in range(n_nuisance_slices):
                        tasks.append((float(sfr_a), float(mu0), ch, int(j)))
                else:
                    tasks.append((float(sfr_a), float(mu0), ch, -1))
    return tasks


def _run_task(task: GridTask) -> Union[dict, List[dict]]:
    sfr_a, mu0, ch, nuisance_index = task
    if nuisance_index < 0:
        return _compute_grid_cell(sfr_a, mu0, ch)
    return _compute_nuisance_slice(sfr_a, mu0, ch, nuisance_index)


def _run_ablation_cells_sequential(
    *,
    target: str,
    mode: ComputeMode,
    sfra_vals: Sequence[float],
    mu0_vals: Sequence[float],
    bps_by_channel: Dict[str, Any],
    ablation_sample_one: bool,
    aggregate: AggregateMode,
    v_gpc3: float,
    n_events: int,
    n_nuisance_slices: int,
    total: int,
) -> List[dict]:
    tasks = _iter_grid_tasks(
        sfra_vals,
        mu0_vals,
        bps_by_channel,
        mode=mode,
        n_nuisance_slices=n_nuisance_slices,
        ablation_sample_one=ablation_sample_one,
    )
    partials: List[dict] = []
    done = 0
    full_rows: List[dict] = []
    for task in tasks:
        result = _run_task(task)
        if isinstance(result, dict) and "nuisance_index" in result:
            partials.append(result)
        else:
            full_rows.append(result)  # type: ignore[arg-type]
        done += 1
        if done % max(1, total // 20) == 0:
            print(f"  [{target}] {100.0 * done / max(total, 1):5.1f}%", flush=True)

    if mode == "ablate" and not ablation_sample_one:
        return _aggregate_slice_rows(
            partials,
            target=target,
            aggregate=aggregate,
            v_gpc3=v_gpc3,
            n_events=n_events,
        )
    return full_rows


def _run_ablation_cells_parallel(
    *,
    target: str,
    mode: ComputeMode,
    sfra_vals: Sequence[float],
    mu0_vals: Sequence[float],
    bps_by_channel: Dict[str, Any],
    workers: int,
    ablation_sample_one: bool,
    aggregate: AggregateMode,
    v_gpc3: float,
    n_events: int,
    n_nuisance_slices: int,
    total: int,
) -> List[dict]:
    tasks = _iter_grid_tasks(
        sfra_vals,
        mu0_vals,
        bps_by_channel,
        mode=mode,
        n_nuisance_slices=n_nuisance_slices,
        ablation_sample_one=ablation_sample_one,
    )
    partials: List[dict] = []
    full_rows: List[dict] = []
    done = 0
    ctx = mp.get_context("fork")
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=ctx,
        initializer=_worker_init,
    ) as ex:
        futures = [ex.submit(_run_task, task) for task in tasks]
        for fut in as_completed(futures):
            result = fut.result()
            if isinstance(result, dict) and "nuisance_index" in result:
                partials.append(result)
            else:
                full_rows.append(result)  # type: ignore[arg-type]
            done += 1
            if done % max(1, total // 20) == 0:
                print(
                    f"  [{target}] {100.0 * done / max(total, 1):5.1f}% "
                    f"({workers} workers)",
                    flush=True,
                )

    if mode == "ablate" and not ablation_sample_one:
        return _aggregate_slice_rows(
            partials,
            target=target,
            aggregate=aggregate,
            v_gpc3=v_gpc3,
            n_events=n_events,
        )
    return full_rows


def build_rate_grid_table(
    mod: Any,
    bps_by_channel: Dict[str, Any],
    cosmology: Tuple[np.ndarray, np.ndarray, np.ndarray, float],
    logZ_by_channel: Dict[str, Tuple[float, float]],
    *,
    target: str,
    sfra_vals: Sequence[float],
    mu0_vals: Sequence[float],
    base_nuisances: Dict[str, float],
    aggregate: AggregateMode,
    v_gpc3: float,
    chunk_size: int,
    workers: int = 1,
    nuisance_vals: Sequence[float] = (),
    sample_seed: int = 42,
    n_events: int = 5000,
    ablation_sample_one: bool = False,
) -> pd.DataFrame:
    global _POOL
    mode = target_compute_mode(target)
    n_sfra = len(sfra_vals)
    n_mu0 = len(mu0_vals)

    _POOL = {
        "mod": mod,
        "bps_by_channel": bps_by_channel,
        "cosmology": cosmology,
        "logZ_by_channel": logZ_by_channel,
        "mode": mode,
        "target": target,
        "base_nuisances": base_nuisances,
        "v_gpc3": v_gpc3,
        "chunk_size": chunk_size,
        "sample_seed": int(sample_seed),
        "n_events": int(n_events),
    }
    if mode == "ablate":
        _POOL["ablated"] = target
        _POOL["nuisance_vals"] = tuple(float(v) for v in nuisance_vals)
        _POOL["aggregate"] = aggregate
        _POOL["ablation_sample_one"] = bool(ablation_sample_one)

    n_ch = sum(
        1 for ch in CHANNELS if ch in bps_by_channel and len(bps_by_channel[ch]) > 0
    )
    ablation_sample_one = bool(_POOL.get("ablation_sample_one", False))
    n_nuisance_slices = (
        1
        if (mode == "ablate" and ablation_sample_one)
        else (len(nuisance_vals) if mode == "ablate" else 1)
    )
    cell_total = n_sfra * n_mu0 * n_ch * n_nuisance_slices

    if workers <= 1:
        rows = _run_ablation_cells_sequential(
            target=target,
            mode=mode,
            sfra_vals=sfra_vals,
            mu0_vals=mu0_vals,
            bps_by_channel=bps_by_channel,
            ablation_sample_one=ablation_sample_one,
            aggregate=aggregate,
            v_gpc3=v_gpc3,
            n_events=int(n_events),
            n_nuisance_slices=n_nuisance_slices,
            total=cell_total,
        )
    else:
        print(
            f"  [{target}] {workers} worker processes, "
            f"{cell_total} integration tasks "
            f"({n_sfra}×{n_mu0}×{n_ch}×{n_nuisance_slices})",
            flush=True,
        )
        rows = _run_ablation_cells_parallel(
            target=target,
            mode=mode,
            sfra_vals=sfra_vals,
            mu0_vals=mu0_vals,
            bps_by_channel=bps_by_channel,
            workers=workers,
            ablation_sample_one=ablation_sample_one,
            aggregate=aggregate,
            v_gpc3=v_gpc3,
            n_events=int(n_events),
            n_nuisance_slices=n_nuisance_slices,
            total=cell_total,
        )

    hp = pd.DataFrame(rows)
    if hp.empty:
        raise ValueError(f"No rows produced for target {target!r}")
    return hp


def pivot_channel(
    hp: pd.DataFrame,
    channel: str,
    metric: MetricName,
    *,
    sfra_axis: np.ndarray | None = None,
    mu0_axis: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    sub = hp.loc[hp["channel"].astype(str) == channel].copy()
    if sub.empty:
        raise ValueError(f"No rows for channel {channel!r}")

    if metric == "rate":
        sub["_metric"] = sub["rate_per_gpc3_yr"].astype(np.float64)
    elif metric == "log_rate":
        r = sub["rate_per_gpc3_yr"].astype(np.float64)
        sub["_metric"] = np.log10(np.maximum(r, 1e-30))
    else:
        raise ValueError(metric)

    if sfra_axis is None:
        sfra_vals = np.sort(sub["sfra"].unique())
    else:
        sfra_vals = np.asarray(sfra_axis, dtype=np.float64)
    if mu0_axis is None:
        mu0_vals = np.sort(sub["mu0"].unique())
    else:
        mu0_vals = np.asarray(mu0_axis, dtype=np.float64)

    z = np.full((len(sfra_vals), len(mu0_vals)), np.nan, dtype=np.float64)
    sfra_to_i = {float(v): i for i, v in enumerate(sfra_vals)}
    mu0_to_j = {float(v): j for j, v in enumerate(mu0_vals)}

    for _, row in sub.iterrows():
        sfra_key = float(row["sfra"])
        mu0_key = float(row["mu0"])
        if sfra_key not in sfra_to_i or mu0_key not in mu0_to_j:
            continue
        i = sfra_to_i[sfra_key]
        j = mu0_to_j[mu0_key]
        z[i, j] = float(row["_metric"])

    return sfra_vals, mu0_vals, z


def _global_color_norm(
    hp: pd.DataFrame,
    metric: MetricName,
    *,
    color_scale: ColorScaleMode,
) -> matplotlib.colors.Normalize:
    pooled: list[np.ndarray] = []
    for ch in CHANNELS:
        _, _, z = pivot_channel(hp, ch, metric)
        fin = z[np.isfinite(z)]
        if fin.size:
            pooled.append(fin)
    if not pooled:
        return matplotlib.colors.Normalize(vmin=0.0, vmax=1.0)

    all_vals = np.concatenate(pooled)
    vmin = float(np.nanmin(all_vals))
    vmax = float(np.nanmax(all_vals))

    if color_scale == "log" and metric != "log_rate":
        nonpos = int(np.sum(all_vals <= 0))
        if nonpos > 0:
            print(
                f"[plot] warning: {nonpos} grid values are <= 0; "
                "log color scale would mask them. Falling back to linear scale.",
                flush=True,
            )
            return matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        pos = all_vals[all_vals > 0]
        if pos.size == 0:
            return matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        return matplotlib.colors.LogNorm(
            vmin=float(np.min(pos)),
            vmax=float(np.max(pos)),
        )
    return matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)


def plot_ablation_heatmaps(
    hp: pd.DataFrame,
    *,
    ablated: str,
    nuisance_range: Tuple[float, float],
    sfra_centers: np.ndarray,
    mu0_centers: np.ndarray,
    aggregate: AggregateMode,
    metric: MetricName,
    color_scale: ColorScaleMode,
    cmap_style: ColormapStyle,
    out_path: Path,
    hm_mod: Any,
    use_tex: bool,
) -> None:
    tex_ok = hm_mod._configure_matplotlib(use_tex=use_tex)
    if use_tex and not tex_ok:
        print(
            "[plot] LaTeX unavailable; using matplotlib mathtext for labels.",
            flush=True,
        )

    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.4), constrained_layout=True)

    label_map = {
        "rate": r"$\mathcal{R}$ [Gpc$^{-3}\,\mathrm{yr}^{-1}$]",
        "log_rate": r"$\log_{10}\mathcal{R}$ [Gpc$^{-3}\,\mathrm{yr}^{-1}$]",
    }
    agg_label = {
        "mean": "mean",
        "std": "std. dev.",
        "min": "minimum",
        "max": "maximum",
    }[aggregate]
    scale_note = ""
    if color_scale == "log" and metric != "log_rate":
        scale_note = " (log intensity)"
    elif color_scale == "linear":
        scale_note = " (linear intensity)"

    n_ev = int(hp["n_events_sampled"].iloc[0]) if "n_events_sampled" in hp.columns else 0
    sample_note = rf" (Step-02 sampled catalog, $N={n_ev}$)" if n_ev > 0 else ""

    if ablated == BASELINE_ALL_FIXED:
        title = (
            r"All nuisances fixed at van Son TNG100 best-fit"
            + sample_note
            + scale_note
        )
    elif ablated == BASELINE_ALL_SAMPLED:
        title = (
            r"All nuisances sampled per cell from Step-00 ranges"
            + sample_note
            + scale_note
        )
    else:
        param_tex = NUISANCE_LATEX.get(ablated, ablated)
        lo, hi = nuisance_range
        title = (
            rf"Nuisance ablation: {agg_label} over {param_tex} $\in [{lo:g}, {hi:g}]$;"
            rf" other nuisances at best-fit"
            + sample_note
            + scale_note
        )

    cmap = hm_mod._heatmap_cmap(cmap_style)
    color_norm = _global_color_norm(hp, metric, color_scale=color_scale)

    ims = []
    for ax, ch in zip(axes, CHANNELS):
        _, _, z = pivot_channel(
            hp,
            ch,
            metric,
            sfra_axis=sfra_centers,
            mu0_axis=mu0_centers,
        )
        z_plot = np.ma.masked_invalid(z)
        im = pcolormesh_sfra_mu0(
            ax,
            mu0_centers,
            sfra_centers,
            z_plot,
            mu0_range=MU0_RANGE,
            sfra_range=SFRA_RANGE,
            norm=color_norm,
            cmap=cmap,
        )
        ax.set_title(ch, fontsize=11)
        ax.set_xlabel(r"$\mu_0$")
        ax.set_ylabel(r"$a_{\mathrm{SF}}$")
        ims.append(im)

    fig.suptitle(title, fontsize=12)
    cbar = fig.colorbar(ims[-1], ax=axes.ravel().tolist(), shrink=0.85, pad=0.02)
    cbar.set_label(label_map[metric], fontsize=11)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Nuisance ablation heatmaps on (sfr_a, mu0) using Step-02-style sampled "
            "catalog rates (default 5000 events per cell, matching 02_build_dataset)."
        )
    )
    p.add_argument(
        "--nuisance",
        choices=ABLATION_TARGETS,
        default=None,
        help=(
            "Run one target: seven per-parameter ablations, or baselines "
            "all_sampled / all_fixed (default: all nine when run without SLURM)."
        ),
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for nuisance draws and event resampling (default: 42).",
    )
    p.add_argument(
        "--n-events",
        type=int,
        default=5000,
        help="Weighted events sampled per grid cell (default: 5000, matches Step 02 N_SAMPLE).",
    )
    p.add_argument(
        "--n-sfra",
        type=int,
        default=DEFAULT_N_SFRA,
        help=f"Grid points in sfr_a (default: {DEFAULT_N_SFRA}).",
    )
    p.add_argument(
        "--n-mu0",
        type=int,
        default=DEFAULT_N_MU0,
        help=f"Grid points in mu0 (default: {DEFAULT_N_MU0}).",
    )
    p.add_argument(
        "--ablation-sample",
        action="store_true",
        help=(
            "For per-parameter ablations only: instead of sweeping the ablated nuisance "
            "over --n-nuisance values and aggregating, draw one uniform sample of the "
            "ablated nuisance per grid cell (others fixed at TNG100 best-fit)."
        ),
    )
    p.add_argument(
        "--n-nuisance",
        type=int,
        default=8,
        help="Grid points for the ablated nuisance range (default: 8).",
    )
    p.add_argument(
        "--aggregate",
        choices=("mean", "std", "min", "max"),
        default="mean",
        help="Reduce ablated nuisance dimension (default: mean).",
    )
    p.add_argument(
        "--metric",
        choices=("rate", "log_rate"),
        default="rate",
        help="rate=Gpc^-3 yr^-1 (default); log_rate=log10(rate).",
    )
    p.add_argument(
        "--color-scale",
        choices=("log", "linear"),
        default="log",
        help="Global intensity mapping: log (default) or linear.",
    )
    p.add_argument(
        "--colormap",
        choices=("sequential", "diverging"),
        default="sequential",
    )
    p.add_argument(
        "--linear-scale",
        action="store_true",
        help="Shorthand for --color-scale linear.",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")),
        help="Parallel worker processes (default: SLURM_CPUS_PER_TASK or 1).",
    )
    p.add_argument(
        "--chunk-size",
        type=int,
        default=20_000,
        help="Binary batch size for cosmic integration (default: 20000; lower if OOM).",
    )
    p.add_argument(
        "--bps-hdf5",
        type=Path,
        default=None,
        help="BPS catalog (default: data/bps_output.h5).",
    )
    p.add_argument(
        "--z-max",
        type=float,
        default=10.0,
        help="Comoving-volume upper limit z_max for rate density (default: 10).",
    )
    p.add_argument(
        "--no-tex",
        action="store_true",
        help="Disable LaTeX text rendering.",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: plots/00_grid_rate_nuisance_ablation/<timestamp>/).",
    )
    p.add_argument(
        "--no-timestamp-subdir",
        action="store_true",
        help="Write directly under plots/00_grid_rate_nuisance_ablation/.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    mod = _load_sspc00_module()
    hm_mod = _load_heatmaps_module()

    import astropy.units as u
    from astropy.cosmology import Planck18

    v_gpc3 = float(Planck18.comoving_volume(float(args.z_max)).to(u.Gpc**3).value)

    bps_path = (args.bps_hdf5 or mod.BPS_PATH_DEFAULT).resolve()
    print(f"Loading BPS: {bps_path}", flush=True)
    bps_full = mod.load_bps(bps_path)
    bps_by_channel = {
        ch: bps_full[bps_full["channel"] == ch].reset_index(drop=True)
        for ch in CHANNELS
    }
    cosmology = mod.build_redshift_grid()
    logZ_by_channel = {
        ch: (
            float(np.log(bps_by_channel[ch]["metallicity"].min())),
            float(np.log(bps_by_channel[ch]["metallicity"].max())),
        )
        for ch in CHANNELS
        if ch in bps_by_channel and len(bps_by_channel[ch]) > 0
    }
    chunk_size = max(1000, int(args.chunk_size))
    workers = max(1, int(args.workers))
    if workers > 1:
        _worker_init()
    print(f"Parallel workers: {workers}", flush=True)

    base_nuisances = fixed_nuisance_params(mod)
    targets = [args.nuisance] if args.nuisance else list(ABLATION_TARGETS)

    sfra_vals = grid_values(mod, "sfra", args.n_sfra)
    mu0_vals = grid_values(mod, "mu0", args.n_mu0)

    color_scale: ColorScaleMode = args.color_scale
    if args.linear_scale:
        if color_scale != "log":
            raise SystemExit("Use only one of --linear-scale and --color-scale (not both).")
        color_scale = "linear"

    metric: MetricName = args.metric
    aggregate: AggregateMode = args.aggregate
    cmap_style: ColormapStyle = args.colormap

    if args.out_dir is not None:
        out_dir = args.out_dir.resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = resolve_plot_output(
            Path(__file__),
            no_timestamp_subdir=args.no_timestamp_subdir,
        )

    all_meta: Dict[str, Any] = {
        "aggregate": aggregate,
        "metric": metric,
        "color_scale": color_scale,
        "colormap_style": cmap_style,
        "n_sfra": int(args.n_sfra),
        "n_mu0": int(args.n_mu0),
        "n_nuisance": int(args.n_nuisance),
        "n_events": int(args.n_events),
        "sfra_values": [float(v) for v in sfra_vals],
        "mu0_values": [float(v) for v in mu0_vals],
        "fixed_nuisances_bestfit": base_nuisances,
        "comoving_volume_gpc3": v_gpc3,
        "z_max": float(args.z_max),
        "bps_hdf5": str(bps_path),
        "chunk_size": chunk_size,
        "workers": workers,
        "plots": {},
    }

    for ablated in targets:
        mode = target_compute_mode(ablated)
        if mode == "ablate":
            lo, hi = mod.NUISANCE_RANGES[ablated]
            nuisance_range = (float(lo), float(hi))
            if args.ablation_sample:
                nuisance_vals = ()
                print(
                    f"\nAblation {ablated}: sample one value per cell from "
                    f"[{lo}, {hi}] (seed={args.seed}); grid {len(sfra_vals)}×{len(mu0_vals)}",
                    flush=True,
                )
            else:
                nuisance_vals = grid_values(
                    mod, "nuisance", args.n_nuisance, nuisance=ablated
                )
                print(
                    f"\nAblation {ablated}: sweep {len(nuisance_vals)} values in "
                    f"[{lo}, {hi}]; grid {len(sfra_vals)}×{len(mu0_vals)}",
                    flush=True,
                )
        elif mode == "all_fixed":
            nuisance_vals = ()
            nuisance_range = (0.0, 0.0)
            print(
                f"\nBaseline {ablated}: all nuisances at TNG100 best-fit; "
                f"grid {len(sfra_vals)}×{len(mu0_vals)}",
                flush=True,
            )
        else:
            nuisance_vals = ()
            nuisance_range = (0.0, 0.0)
            print(
                f"\nBaseline {ablated}: one nuisance draw per cell from "
                f"NUISANCE_RANGES; grid {len(sfra_vals)}×{len(mu0_vals)} "
                f"(seed={args.seed})",
                flush=True,
            )

        hp = build_rate_grid_table(
            mod,
            bps_by_channel,
            cosmology,
            logZ_by_channel,
            target=ablated,
            sfra_vals=sfra_vals,
            mu0_vals=mu0_vals,
            nuisance_vals=nuisance_vals,
            base_nuisances=base_nuisances,
            aggregate=aggregate,
            v_gpc3=v_gpc3,
            chunk_size=chunk_size,
            workers=workers,
            sample_seed=int(args.seed),
            n_events=int(args.n_events),
            ablation_sample_one=bool(args.ablation_sample) if mode == "ablate" else False,
        )

        out_path = out_dir / f"grid_rate_ablation_{ablated}.png"
        table_path = out_dir / f"grid_rate_ablation_{ablated}_table.csv.gz"
        hp.to_csv(table_path, index=False)
        print(f"Saved → {table_path}", flush=True)

        plot_ablation_heatmaps(
            hp,
            ablated=ablated,
            nuisance_range=nuisance_range,
            sfra_centers=sfra_vals,
            mu0_centers=mu0_vals,
            aggregate=aggregate,
            metric=metric,
            color_scale=color_scale,
            cmap_style=cmap_style,
            out_path=out_path,
            hm_mod=hm_mod,
            use_tex=not bool(args.no_tex),
        )

        masked_dir = out_dir / "masked_edges"
        masked_dir.mkdir(parents=True, exist_ok=True)
        masked_path = masked_dir / f"grid_rate_ablation_{ablated}_masked_edges.png"
        hp_masked = mask_edges_inplace(hp)
        plot_ablation_heatmaps(
            hp_masked,
            ablated=ablated,
            nuisance_range=nuisance_range,
            sfra_centers=sfra_vals,
            mu0_centers=mu0_vals,
            aggregate=aggregate,
            metric=metric,
            color_scale=color_scale,
            cmap_style=cmap_style,
            out_path=masked_path,
            hm_mod=hm_mod,
            use_tex=not bool(args.no_tex),
        )

        meta_path = out_path.with_suffix(".json")
        plot_meta: Dict[str, Any] = {
            "target": ablated,
            "compute_mode": mode,
            "ablated_nuisance": ablated,
            "n_rows": int(len(hp)),
            "sfra_range": [float(hp["sfra"].min()), float(hp["sfra"].max())],
            "mu0_range": [float(hp["mu0"].min()), float(hp["mu0"].max())],
            "table_csv_gz": str(table_path),
            "output_png": str(out_path),
            "masked_edges_png": str(masked_path),
        }
        if mode == "ablate":
            plot_meta["nuisance_range"] = [float(nuisance_range[0]), float(nuisance_range[1])]
            plot_meta["nuisance_values"] = [float(v) for v in nuisance_vals]
        elif mode == "all_fixed":
            plot_meta["fixed_nuisances"] = dict(base_nuisances)
        else:
            plot_meta["sample_seed"] = int(args.seed)
            plot_meta["nuisance_ranges"] = {
                k: [float(v) for v in mod.NUISANCE_RANGES[k]]
                for k in nuisance_keys(mod)
            }
        meta_path.write_text(json.dumps(plot_meta, indent=2), encoding="utf-8")
        print(f"Saved → {meta_path}", flush=True)
        all_meta["plots"][ablated] = plot_meta

    summary_path = out_dir / "ablation_run.json"
    if args.nuisance is None:
        summary_path.write_text(json.dumps(all_meta, indent=2), encoding="utf-8")
        print(f"\nSaved run summary → {summary_path}", flush=True)
    else:
        print(f"\nSingle-nuisance run; plot metadata written alongside PNG.", flush=True)


if __name__ == "__main__":
    main()
