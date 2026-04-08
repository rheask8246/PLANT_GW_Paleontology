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

**What it does:** Performs cosmic integration of Binary Population Synthesis (BPS) output over star formation history and metallicity evolution (Madau-Dickinson / Neijssel+19 models) to produce a grid of predicted GW merger event catalogs.

**Input:**
- `data/bps_output.h5` — COMPAS/GROWL BPS simulation output (3.35M binaries) with columns: `m1`, `m2`, `t_delay`, `metallicity`, `channel` (SMT/CE/CHE).
- `data/SNR_Grid_IMRPhenomPv2_FD_all_noise.hdf5` — pre-computed SNR grid for O3 sensitivity.

**Output:** `data/sspc/models_sspc.hdf5` — HDF5 file with keys `/CHANNEL/sfra{NNNN}/mu0{MMMM}`.  
Each key contains a DataFrame with columns: `mchirp`, `q`, `chieff`, `z`, `weight`.

**Parameters scanned:**

| Parameter | Symbol | Range | Description |
|-----------|--------|-------|-------------|
| SFR amplitude | `sfr_a` | 0.008 – 0.035 | Madau-Dickinson aSF (M☉/yr/Mpc³) |
| Mean metallicity (z=0) | `mu0` | 0.005 – 0.065 | Log-skew-normal μ₀ |
| SFR peak redshift | `sfr_c` | 2.0 – 5.5 | (nuisance, sampled per grid point) |
| SFR high-z slope | `sfr_d` | 4.7 – 5.7 | (nuisance) |
| Metallicity evo. | `muz` | −0.5 – 0.1 | Redshift scaling of μ (nuisance) |
| Metallicity spread | `sigma0` | 0.3 – 0.7 | Log-normal σ (nuisance) |

**Grid size:** `--n-sfra` × `--n-mu0` grid points × 3 channels (SMT, CE, CHE).  
Full run: 50 × 50 × 3 = 7,500 grid points × 50,000 events ≈ 375M events total.

**Key design:**
- Events are sampled from the *detection-weighted* population (`det_weight = weight × pdet`) so all generated events are detectable.
- `z` values are clipped to `[0.1, 1.5]` (first physical redshift bin onwards).
- `chieff` drawn from channel-dependent Gaussians: SMT ~ N(0, 0.1), CE ~ N(0, 0.2), CHE ~ N(0, 0.25).

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

## File structure

```
PLANT_GW_Paleontology/
├── 00_sspc_data_generation.py   # Step 00: SSPC cosmic integration
├── 02_build_dataset.py          # Step 02: HDF5 → parquet + normalizer
├── 03_rate_network.py           # Step 03: rate MLP
├── 04_cfm_emulator.py           # Step 04: CFM emulator
├── 04b_diffusion_emulator.py    # Step 04b: Diffusion emulator
├── selection_effects.py         # pdet computation (SNR grid)
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
│   ├── SNR_Grid_IMRPhenomPv2_FD_all_noise.hdf5    # pdet SNR grid (O3)
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
