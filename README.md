# PLANT GW Paleontology

End-to-end pipeline for population-level gravitational-wave (GW) inference using Synthetic Stellar Population Convolution (SSPC) data and machine learning emulators (Conditional Flow Matching + Diffusion).

---

## Running on ACCESS Expanse (SLURM)

### One-time setup on Expanse

```bash
ssh <user>@login.expanse.sdsc.edu
cd /expanse/lustre/scratch/$USER/temp_project
git clone <repo> PLANT_GW_Paleontology && cd PLANT_GW_Paleontology

module load cpu
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

mkdir -p logs
```

Edit `slurm/*.sh` and replace `<<PROJECT>>` with your ACCESS allocation ID (find it with `expanse-client user`).

### Full pipeline execution (in order)

```bash
# Step 1 – generate SSPC data (~3h, 48 CPU-h on shared partition)
sbatch slurm/00_data_gen.sh

# Step 2 – build ML dataset (run after Step 1 completes, ~10 min)
sbatch slurm/02_build_dataset.sh

# Step 3 – train rate network (CPU, ~30 min)
sbatch slurm/03_rate_network.sh

# Steps 4/4b – train CFM and Diffusion (can run in parallel, 1 GPU each, ~12–20h)
sbatch slurm/04_cfm.sh
sbatch slurm/04b_diffusion.sh

# Optional: quick sanity check on GPU debug queue before launching full training
sbatch slurm/smoke_test.sh
```

Use `squeue -u $USER` to monitor jobs. Logs go to `logs/`.

### Resource summary

| Step | Script | Partition | CPUs | GPUs | Memory | Est. Wall-time |
|------|--------|-----------|------|------|--------|----------------|
| 00 | Data generation | shared | 16 | — | 64 GB | 4 h |
| 02 | Build dataset | shared | 8 | — | 32 GB | 1 h |
| 03 | Rate network | shared | 4 | — | 16 GB | 1 h |
| 04 | CFM emulator | gpu-shared | 10 | 1×V100 | 96 GB | 24 h |
| 04b | Diffusion emulator | gpu-shared | 10 | 1×V100 | 96 GB | 24 h |

All GPU jobs use `gpu-shared` (not exclusive), saving allocation.

---

## Local quick-start

```bash
cd PLANT_GW_Paleontology
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Tiny smoke-test run (CPU, ~5 min)
python 00_sspc_data_generation.py --n-sfra 4 --n-mu0 4 --n-events 5000
python 02_build_dataset.py --hdf5 data/sspc/models_sspc.hdf5 --data-source sspc
python 03_rate_network.py --epochs 200
python 04_cfm_emulator.py --smoke-test --steps 500
python 04b_diffusion_emulator.py --smoke-test --steps 500
```

---

## Pipeline overview

### Step 00 — SSPC Data Generation (`00_sspc_data_generation.py`)

**What it does:** Performs cosmic integration of Binary Population Synthesis (BPS) output over star formation history and metallicity evolution (Madau-Dickinson / Neijssel+19 models) to produce a grid of predicted GW merger event catalogs representing the **intrinsic** population — no detection-probability weighting is applied.

**Input:**
- `data/bps_output.h5` — COMPAS/GROWL BPS simulation output (3.35M binaries) with columns: `m1`, `m2`, `t_delay`, `metallicity`, `channel` (SMT/CE/CHE).

**Output:** `data/sspc/models_sspc.hdf5` — HDF5 file with keys `/CHANNEL/sfra{NNNN}/mu0{MMMM}`.  
Each key contains a DataFrame with columns: `mchirp`, `q`, `chieff`, `z`, `weight`.

**Parameters scanned (centred on TNG100-1 best-fit values from Briel+):**

| Parameter | Symbol | Range | TNG100-1 best-fit | Grid type |
|-----------|--------|-------|-------------------|-----------|
| SFR amplitude | `sfr_a` | 0.010 – 0.030 | ≈ 0.017 | **primary axis** |
| Mean metallicity (z=0) | `mu0` | 0.010 – 0.060 | ≈ 0.025 | **primary axis** |
| SFR rising slope | `sfr_b` | 1.0 – 3.0 | ≈ 1.46 | nuisance (random draw) |
| SFR turnover redshift | `sfr_c` | 2.0 – 6.0 | ≈ 4.51 | nuisance |
| SFR falling slope | `sfr_d` | 4.0 – 8.0 | ≈ 6.21 | nuisance |
| Metallicity evo. slope | `muz` | −0.5 – 0.1 | ≈ −0.052 | nuisance |
| Metallicity log-spread | `sigma0` | 0.5 – 1.5 | ≈ 1.15 | nuisance |
| Spread redshift evo. | `sigmaz` | −0.1 – 0.1 | ≈ 0.047 | nuisance |
| Log-skew skewness | `alpha_skew` | −2.0 – 2.0 | ≈ −1.85 | nuisance |

**Grid size:** `--n-sfra` × `--n-mu0` grid points × 3 channels (SMT, CE, CHE).  
Full run: 50 × 50 × 3 = 7,500 grid points × 50,000 events ≈ 375M events total.

**Key design:**
- Events are sampled from the **intrinsic merger-rate distribution** (no pdet cut). All mergers across all redshifts z = 0.1 – 10 are included, weighted by the physical merger rate at each redshift.
- `weight` column = per-binary intrinsic merger rate [merger/yr/binary] at the drawn z.
- `z` values clipped to ≥ 0.1 to avoid log10(0) issues.
- `chieff` drawn from channel-dependent Gaussians (no spin info in BPS): CE ~ N(0, 0.10), CHE ~ N(0.25, 0.15), SMT ~ N(0.05, 0.12).

---

### Step 02 — Build ML Dataset (`02_build_dataset.py`)

**What it does:** Reads the HDF5, samples events per grid point, encodes hyperparameters, computes detection rates, writes ML-ready parquet + JSON artifacts.

**Input:** `data/sspc/models_sspc.hdf5`

**Output:**
- `data/all_detected_events.parquet` — all sampled events with columns `mchirp, q, chieff, z, channel_id, lam_sfra, lam_mu0`
- `data/hyperparam_table.csv` — one row per grid point: `chi_b` (=`sfr_a` normalised), `alpha_CE` (=`mu0` normalised), `channel`, `sum_pdet` (=`sum_weight`)
- `data/train_test_splits.json` — train/test grid indices
- `data/obs_normalizer.json` — per-observable z-score stats (log-space for `mchirp`, `z`)

**Encoding:** `sfr_a` and `mu0` are linearly normalised to [−1, 1] using their generation ranges. Channel is one-hot encoded.

**Normalizer:** `mchirp` → `log10(max(x, 1e-3))`, `z` → `log10(max(x, 0.1))` to avoid `log10(0)` corruption.

---

### Step 03 — Rate Network (`03_rate_network.py`)

**What it does:** Trains a small MLP to predict the expected detection rate `log10(Σ det_weight)` as a function of the hyperparameter vector λ = [sfr_a_norm, mu0_norm, channel_1hot×3].

**Input:** `data/hyperparam_table.csv`, `data/train_test_splits.json`

**Architecture:** λ (5D) → [64, 64] → 1 (ReLU activations, dropout 0.1).

**Target:** `log10(sum_weight)` where `sum_weight = Σ pdet × weight` across all events at a grid point.

**Training:** Adam, lr=1e-3, ReduceLROnPlateau, early stopping (patience 100). Default 1000 epochs.

**Output:** `checkpoints/rate_network.pt`, plots in `plots/rate_network/`.

---

### Step 04 — CFM Emulator (`04_cfm_emulator.py`)

**What it does:** Trains a Conditional Flow Matching (CFM) model to learn the conditional distribution `p(mchirp, q, chieff, z | λ)` of GW event observables given hyperparameters.

**Input:** `data/all_detected_events.parquet`, `data/obs_normalizer.json`, `data/hyperparam_table.csv`

**Architecture:**
- Encoder: λ (5D) → [256, 256, 256, 256] → context (128D)
- Vector field: [obs(4) + context(128) + t(1)] → [512, 512, 512, 512] → 4D (full run)
- Smoke test: hidden_dim=128; Full training: hidden_dim=256

**Training (full run):** 100,000 steps, batch=256, AdamW lr=1e-4, CosineAnnealingLR, importance-weighted sampling (cap 3.0), gradient clipping (max_norm=1.0).

**Z-jitter:** ±0.05 uniform noise added to discrete `z` values during training to smooth the distribution.

**Output:** `checkpoints/cfm_final.pt`, plots in `plots/cfm_smoke_test/<timestamp>/`.

---

### Step 04b — Diffusion Emulator (`04b_diffusion_emulator.py`)

**What it does:** Same task as CFM (learn `p(obs | λ)`) using a score-based diffusion model as an alternative/complement.

**Architecture:** Same encoder as CFM. Score network: [noised_obs(4) + context(128) + t_embed(32)] → [256×4] → 4D.

**Training:** Same hyperparameters as CFM (100k steps, batch=256, hidden_dim=256 for full run). Uses a cosine noise schedule with `N_TIMESTEPS=50` diffusion steps.

**Output:** `checkpoints/diffusion_final.pt`, plots in `plots/diffusion_smoke_test/<timestamp>/`.

---

---

### Analysis — BBH Mass Distribution (`data_distribution_analysis.py`)

**What it does:** Reproduces Figure 5 of Briel et al. (Fit_SFRD_TNG paper) and overplots SSPC-generated data for direct comparison. Shows the redshift evolution of the BBH primary-mass distribution dR/dm₁ across three formation channels at merger redshifts z = 0.1 – 0.5.

**Input:**
- `data/sspc/models_sspc.hdf5` — SSPC event catalogs (right column)
- `../Fit_SFRD_TNG/data/Rate_info.h5` — TNG100-1 intrinsic merger rate (left column; optional)
- `../Fit_SFRD_TNG/data/COMPAS_Output_wWeights.h5` — COMPAS DCO table (optional)
- `../Fit_SFRD_TNG/data/BBHMassSpinRedshift_BSplineIID.h5` — GWTC-4 B-Spline overlay (optional, requires `popsummary`)

**Output:** `plots/data_distribution_analysis.png`

**Figure layout:**

| Row | Channel |
|-----|---------|
| 0 | All channels (stable + CE, CHE excluded — matching original Fig. 5) |
| 1 | Stable mass-transfer (SMT) only |
| 2 | Common-envelope (CE) only |

- **Left column**: TNG100-1 intrinsic rate [Gpc⁻³ yr⁻¹ M☉⁻¹] — reproduced from the reference figure
- **Right column**: SSPC intrinsic merger-rate dN/dm₁ aggregated across all grid points, area-normalised per z-slice for shape comparison
- Gray band: GWTC-4 B-Spline posterior (from `popsummary`)
- Colors: `rocket_r` colormap, darkest = z=0.1, lightest = z=0.5

**Note on z range:** The SSPC data now spans z = 0.1 – 10 (intrinsic rate, no detection cut). TNG data in `Rate_info.h5` covers z ≤ 0.45. The original paper uses z up to 8, which requires a full TNG simulation run.

**Usage:**
```bash
python data_distribution_analysis.py

# With custom paths
python data_distribution_analysis.py \
  --tng-data-dir /path/to/Fit_SFRD_TNG/data \
  --sspc-hdf5 data/sspc/models_sspc.hdf5 \
  --output plots/my_comparison.png
```

---

### Intrinsic data validation (`test/validation/run_data_validation.py`)

**What it does:** After `02_build_dataset.py`, this script checks that the **intrinsic** training data (full event range, not detection-weighted) is consistent and usable. It reads `all_events.parquet` by default and does **not** filter on detectability or use `all_detected_events.parquet` for these checks.

**Checks (summary):**

| Check | Purpose |
|-------|---------|
| **Grid coverage** | `lambda_*` ranges, CE occupancy in `(chi_b, alpha_CE)` space, optional coverage plots |
| **Channel health** | Per-channel grid counts and intrinsic `sum_weight` / `log_efficiency` from `hyperparam_table.csv` |
| **Event validity** | NaNs, physical bounds on `mchirp`, `q`, `chieff`, `z`; `weight` is optional (if absent, weight checks are skipped) |
| **Split hygiene** | No overlapping `grid_idx` across train/val/test; nearest train–test distance in λ-space |
| **Rare regions** | Flags low-intrinsic-rate grid points (default: bottom 5% of `sum_weight`) |
| **Distribution sanity** | Train vs test KL / KS / MMD on observables; redshift shape by `channel_id` |

**Outputs:**

- `test/reports/validation/validation_summary.json` — machine-readable pass/warn/fail per check  
- `test/reports/validation/validation_summary.md` — short table  
- `test/reports/validation/*.csv` — e.g. channel summary, rare-event flags, violations sample  
- `test/plots/validation/*.png` — heatmaps, histograms, CDFs  

**When to run:** After Step 02 (same directory as `hyperparam_table.csv`, `hyperparam_table_encoded.csv`, `splits.json`, `all_events.parquet`).

**How to run:**

```bash
cd PLANT_GW_Paleontology
source ../venv/bin/activate   # or your venv

python test/validation/run_data_validation.py
```

**Options:**

| Flag | Meaning |
|------|---------|
| `--project-root PATH` | Root directory (default: parent of `test/validation/`) |
| `--events-parquet PATH` | Override path to intrinsic events (default: `all_events.parquet` under project root) |
| `--rare-quantile FLOAT` | Quantile for “rare” intrinsic grids (default: `0.05`) |
| `--strict` | Exit with code 1 if any check is **warn** or **fail** (for CI / batch jobs) |

---

## File structure

```
PLANT_GW_Paleontology/
├── 00_sspc_data_generation.py   # Step 00: SSPC cosmic integration
├── 02_build_dataset.py          # Step 02: HDF5 → parquet + normalizer
├── 03_rate_network.py           # Step 03: rate MLP
├── 04_cfm_emulator.py           # Step 04: CFM emulator
├── 04b_diffusion_emulator.py    # Step 04b: Diffusion emulator
├── data_distribution_analysis.py  # Figure 5 comparison (TNG vs SSPC mass dist.)
├── selection_effects.py         # pdet computation (SNR grid; used by analysis scripts only)
├── test/
│   ├── validation/
│   │   └── run_data_validation.py   # Intrinsic data validation (after Step 02)
│   ├── reports/validation/          # validation_summary.json, .md, CSVs (generated)
│   └── plots/validation/            # validation plots (generated)
├── requirements.txt
├── slurm/
│   ├── 00_data_gen.sh
│   ├── 02_build_dataset.sh
│   ├── 03_rate_network.sh
│   ├── 04_cfm.sh
│   ├── 04b_diffusion.sh
│   └── smoke_test.sh
├── data/
│   ├── bps_output.h5                              # BPS input (COMPAS/GROWL)
│   └── sspc/
│       └── models_sspc.hdf5                       # generated event catalogs
├── checkpoints/
│   ├── rate_network.pt
│   ├── cfm_final.pt
│   └── diffusion_final.pt
└── plots/
```

---

## Notes

- **CPU vs GPU:** Steps 00–03 are CPU-only. Steps 04 and 04b are GPU-accelerated (`--device cuda`).
- **Lustre constraint:** On Expanse, add `#SBATCH --constraint="lustre"` to any script if you place data on `/expanse/lustre/scratch`.
- **Account ID:** Replace `<<PROJECT>>` in all SLURM scripts with your actual allocation (e.g., `abc123`).
- **Checkpoint loading for inference:** Load saved checkpoints as `torch.load("checkpoints/cfm_final.pt")["model_state"]`.
