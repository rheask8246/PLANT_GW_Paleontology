"""Shared paths for the PLANT GW pipeline (project root vs workspace root)."""
from __future__ import annotations

import sys
from datetime import datetime
from importlib import import_module
from pathlib import Path
from types import ModuleType

# PLANT_GW_Paleontology/ — checkpoints/, models/, plots/, …
PROJECT_ROOT = Path(__file__).resolve().parent

# Workspace root (parent of PLANT_GW_Paleontology/) — syntheticstellarpopconvolve, gwtc4, Fit_SFRD_TNG
REPO_ROOT = PROJECT_ROOT.parent

SCRIPTS_DIR = PROJECT_ROOT / "scripts"
ANALYSIS_DIR = SCRIPTS_DIR / "analysis"

# Step-02 ML artifacts (hyperparam tables, parquets, splits)
ML_DATA_DIR = PROJECT_ROOT / "data"
HYPERPARAM_TABLE_CSV = ML_DATA_DIR / "hyperparam_table.csv"
HYPERPARAM_TABLE_ENCODED_CSV = ML_DATA_DIR / "hyperparam_table_encoded.csv"
ALL_EVENTS_PARQUET = ML_DATA_DIR / "all_events.parquet"
ALL_DETECTED_EVENTS_PARQUET = ML_DATA_DIR / "all_detected_events.parquet"
SPLITS_JSON = ML_DATA_DIR / "splits.json"

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
OBS_NORMALIZER_JSON = CHECKPOINT_DIR / "obs_normalizer.json"

PLOTS_ROOT = PROJECT_ROOT / "plots"


def plot_script_stem(script_path: Path | str) -> str:
    """Python stem used as the plots subdirectory (e.g. ``04_cfm_emulator``)."""
    return Path(script_path).resolve().stem


def plot_run_dir(
    script_path: Path | str,
    *,
    timestamp: str | None = None,
    mkdir: bool = True,
) -> Path:
    """
    Timestamped output folder: ``plots/{script_stem}/{YYYY-MM-DD_HH-MM-SS}/``.

    *script_path* should be ``Path(__file__)`` of the script that produces the figures.
    """
    stem = plot_script_stem(script_path)
    ts = timestamp or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = PLOTS_ROOT / stem / ts
    if mkdir:
        run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def resolve_plot_output(
    script_path: Path | str,
    *,
    out: Path | None = None,
    timestamp: str | None = None,
    no_timestamp_subdir: bool = False,
    filename: str | None = None,
) -> Path:
    """
    Resolve a single output file path under ``plots/{script_stem}/``.

    If *out* is set, use it as-is. Otherwise use a timestamped run directory;
    append *filename* when provided.
    """
    if out is not None:
        path = Path(out).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        return path
    if no_timestamp_subdir:
        base = PLOTS_ROOT / plot_script_stem(script_path)
        base.mkdir(parents=True, exist_ok=True)
        return base / filename if filename else base
    run_dir = plot_run_dir(script_path, timestamp=timestamp, mkdir=True)
    return run_dir / filename if filename else run_dir


def ensure_paths() -> None:
    """Put project root (models/), scripts/, and analysis/ on sys.path."""
    for p in (PROJECT_ROOT, SCRIPTS_DIR, ANALYSIS_DIR):
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)


def find_data_dir() -> Path:
    """Resolve ``data/`` whether run from project root or elsewhere."""
    for d in (PROJECT_ROOT / "data", Path.cwd() / "data", Path("PLANT_GW_Paleontology/data")):
        d = d.resolve()
        if d.is_dir():
            return d
    return (PROJECT_ROOT / "data").resolve()


def ml_data_dir() -> Path:
    """Directory for Step-02 outputs (prefers ``data/``; falls back to project root if legacy)."""
    data = find_data_dir()
    for base in (data, PROJECT_ROOT):
        if (base / "hyperparam_table_encoded.csv").exists():
            return base.resolve()
    return data.resolve()


def find_work_dir() -> Path:
    """Alias for :func:`ml_data_dir` (dataset paths live under ``data/``)."""
    return ml_data_dir()


def load_posterior_network_module() -> ModuleType:
    """Import ``05_posterior_network`` from ``scripts/`` (name starts with a digit)."""
    ensure_paths()
    return import_module("05_posterior_network")
