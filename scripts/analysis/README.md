# Analysis scripts

Diagnostics and figures **after** the training pipeline (`scripts/00` … `scripts/05`).  
Run from the project root, e.g. `python scripts/analysis/02_validation.py`.

## Plot output layout

All figures default to timestamped run folders under `plots/`:

```text
plots/<script_stem>/<YYYY-MM-DD_HH-MM-SS>/   # e.g. plots/04_cfm_emulator/2026-05-17_14-30-00/
```

The subdirectory name matches the **Python script basename** (step prefix included), e.g. `00_distribution_compare`, `03_rate_network`, `04_cfm_emulator`. Override with `--out`, `--output-dir`, or `--out-dir` where supported. Helpers live in `plant_paths.py` (`plot_run_dir`, `resolve_plot_output`).

Step **02** validation reports stay under `test/reports/validation/<timestamp>/`; only its **plots** use `plots/02_validation/<timestamp>/`.

## Layout

| Script | Pipeline step | Purpose |
|--------|---------------|---------|
| `00_population_figures.py` | 00 | Rate vs *z* and *m*₁/*m*₂/*q* slices from `data/sspc/models_sspc.hdf5` |
| `00_distribution_compare.py` | 00 | TNG Figure 5/4 style: SSPC vs simulation mass & *z* distributions |
| `00_fig2_spread.py` | 00 | Figure-2-style SSPC mass marginals on the training grid |
| `00_grid_rate_heatmaps.py` | 00 | Heatmaps of `sum_weight` or `n_systems` on (sfr_a, mu0) per channel |
| `02_validation.py` | 02 | Intrinsic QA on `data/*.parquet`, hyperparam tables, `splits.json` |
| `04_emulator_m1_compare.py` | 04 / 04b | CFM vs diffusion *m*₁ KDE at fixed Λ |
| `04_gwtc4_validation.py` | 04 | GWTC-4 paper figures: emulator grids vs data release |
| `05_gwtc_validate.py` | 05 | Real/mock event CSV → posterior marginals |
| `05_synth_real_compare.py` | 05 | Overlay posterior: emulator synthetic vs real GW catalog |
| `05_ensemble_infer.py` | 05 | Combine *K* trained posteriors (log-mean / mixture) |

**Utilities** (`utils/`):

| Script | Purpose |
|--------|---------|
| `fetch_gwtc40_events.py` | Download GWTC-4.0 default PE table from GWOSC → CSV for `05_synth_real_compare.py` |

**Shared library** (`lib/distribution.py`): TNG/SSPC loaders, Figure 5/4 plotting, emulator *m*₁ comparison helpers.

## SLURM (Expanse)

Every analysis script has a matching job under `slurm/`. Pipeline scripts use `slurm/00_data_gen.sh`, …, `slurm/05_posterior_network.sh`.

| Analysis script | SLURM job |
|-----------------|-----------|
| `00_grid_rate_heatmaps.py` | `slurm/00_grid_rate_heatmaps.sh` |
| `00_population_figures.py` | `slurm/06_population_figures.sh` |
| `00_distribution_compare.py` | `slurm/06a_distribution_analysis.sh` |
| `00_fig2_spread.py` | `slurm/09_fig2_spread.sh` |
| `02_validation.py` | `slurm/02b_data_validation.sh` |
| `04_emulator_m1_compare.py` | `slurm/09_emulator_m1_distribution.sh` |
| `04_gwtc4_validation.py` | `slurm/08_gwtc4_validation.sh` |
| `05_gwtc_validate.py` | `slurm/07_gwtc_validate.sh` |
| `05_synth_real_compare.py` | `slurm/07b_synth_real_validation.sh` |
| `05_ensemble_infer.py` | `slurm/05_ensemble_infer.sh` |
| `utils/fetch_gwtc40_events.py` | `slurm/utils_fetch_gwtc40_events.sh` |

## Adding a new check

1. Pick the pipeline step (`00`, `02`, `04`, `05`, …).
2. Add `scripts/analysis/<step>_<short_name>.py` with the standard bootstrap (copy from `02_validation.py`).
3. Reuse `lib/distribution.py` for SSPC mass/*z* helpers when possible.
4. Add a row to this table and a `slurm/` script if the job is long-running.
