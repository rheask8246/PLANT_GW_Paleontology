# PLANT GW Paleontology

This repository is a **software pipeline** that helps scientists ask: *When pairs of black holes (or similar compact objects) merge, what can we say about the astrophysical “settings” of the universe that most likely produced them?* The pipeline uses **simulated** mergers and **machine learning** so that, once trained, a computer can answer that kind of question quickly instead of re-running huge simulations every time.

---

## Who this document is for

- **If you are new to the project** and not from a physics or machine-learning background: start with [The big picture (plain language)](#the-big-picture-plain-language) and the [Tiny glossary](#tiny-glossary). Then skim the **In plain English** boxes under each numbered step. You do not need to read file formats or acronyms in depth to follow the *story* of the pipeline.
- **If you will run the code** on a laptop or a cluster: use [Local quick-start](#local-quick-start) and [Running on ACCESS Expanse (SLURM)](#running-on-access-expanse-slurm), then the **command-line** parts of each step.
- **If you are implementing or extending models**: read the same steps, but focus on **Inputs**, **Outputs**, and the detailed bullets (those sections stay technical on purpose).

---

## The big picture (plain language)

**What is a “gravitational wave”?** In rough terms, when very heavy dead stars (black holes) spiral together and merge, they disturb space and time. Instruments on Earth (such as LIGO) can sometimes detect a faint signal—a “gravitational wave”—from a merger.

**What is a “catalog”?** After processing detector data, each candidate merger is summarized by a few numbers you can think of as a **fingerprint**: how massive the system was, how “lopsided” the masses were, how fast it was spinning, and roughly how far away (often expressed using redshift, a stand-in for distance in cosmology). A **catalog** is a list of many such merger summaries.

**What is this pipeline trying to learn?** Astrophysicists have **knobs** that control computer models of how stars are born, evolve, and eventually merge: for example, how many stars form over cosmic time, or how the chemistry of the gas changes with distance. The pipeline’s job is to connect:

1. **Those knobs (hyperparameters / “settings”)**  
2. **To the kinds of merger fingerprints** you would expect to see in a population.

**Why is machine learning here?** A realistic simulation can be extremely slow. The pipeline first **trains a fast “fake universe” (the CFM or diffusion)**, then **trains a fast “inverse” map** (the posterior) **on the fake data that generator would produce**—so at runtime you are not re-running 00 or 04, only small neural nets. Step 3 still **predicts rates**; the “inverse” part is explicitly **chained after** the generator, not an optional parallel track.

**What is “paleontology” doing in the name?** It is a metaphor: like digging up bones and inferring the past, the project is about **inferring the history and physics of stellar populations** from the “fossil record” of mergers, using models and data together.

### Tiny glossary

| Term  | Meaning in this project |
|--------------|-----------------------------------------------|
| **Binary / merger** | Two compact objects (e.g. black holes) orbit and eventually combine; we call that a *merger*. |
| **Hyperparameters (“settings”)** | Numbers that define the astrophysical model: star-formation and metallicity recipe, and related nuisance parameters (see the tables in Step 00). |
| **SSPC** | A particular way in this project to build **synthetic** merger populations by folding stellar-population models over cosmic time (see Step 00). |
| **Intrinsic vs detection-weighted** | *Intrinsic* = full merger distribution weighted by the physical **merger rate** (no LIGO selection). *Detection-weighted* = an optional alternate view (e.g. Zenodo) where columns may include `pdet`; **this pipeline’s main SSPC path trains on the intrinsic view only.** |
| **Emulator (Step 04 *or* 04b)** | A **required** fast **surrogate** (you train one architecture): “given Λ, draw merger-like **fingerprints**.” Step 05 is trained on **batches of those draws** (emulator **frozen**), not on a separate re-read of the 02 parquets. |
| **Naive Bayes emulator (Step 04c)** | **Optional** non-neural **baseline**: fits per-grid statistics from Step 02 in seconds on CPU; same Λ → (mchirp, q, z) API. Use to benchmark CFM/diffusion—not the default production path before Step 05. |
| **Posterior / inverse model (Step 05)** | “**Given a bag of events**, what Λ is plausible?”—trained to invert the *emulator’s* forward map, in line with the proposal’s **Stage 2 → Stage 4** order (CFM or diffusion first, then transformer + flow). |
| **Neural network / “MLP”** | A flexible function approximator; here used for regression, generation, and inverse inference. |
| **Parquet, HDF5, JSON** | Just **file containers** for tables or hierarchical data; you can ignore the format as long as the scripts that expect them are run in order. |
| **SLURM, `sbatch`, cluster** | A **job scheduler** on high-performance computers: you submit a script, and the cluster runs it when resources are free. |
| **GPU vs CPU** | A **GPU** can accelerate certain large matrix operations (typical in deep learning). **CPU** is fine for smaller tests. |

### One-sentence summary of the technical stack (for specialists)

**Technical:** end-to-end SSPC **population** pipeline: MLP for per-grid **intrinsic merger-rate** totals; a **required** generative **emulator** (train **either** the CFM in Step 04 **or** the diffusion in Step 04b) for *p*(observables | Λ) under the same intrinsic sampling as 02; an **optional** Step **04c** Naive Bayes baseline (no gradient training) for comparison; then a **set-transformer + flow** **amortized posterior** in Step 05 for *p*(Λ | catalog), where training catalogs are **synthetic samples from that emulator** (frozen), matching the proposal’s generative-then-inference ordering—not parallel shortcuts.

---

## Running on ACCESS Expanse (SLURM)

[ACCESS Expanse](https://www.sdsc.edu/services/hpc/expanse/) is a **shared research supercomputer** at SDSC. **SLURM** is the software that decides *when* and *on which computer nodes* your job runs. The commands like `sbatch` below are simply **“please run this script on the cluster when you can, using the time limits and memory I asked for.”** You do not need to know SLURM details if you are only following the *meaning* of the pipeline above—this section is for people who will actually press “submit job.”

### One-time setup on ACCESS Expanse

**Python version:** this project needs **Python 3.10 or newer** (`torchcfm` / `pandas>=2.2.2`). The login node’s **`python3` from `/cm/local` (often 3.6)** and **`python37` (3.7)** are **not** enough for `pip install -r requirements.txt`.

This repository’s `slurm/*.sh` files are **pre-configured** for:

- **`#SBATCH --account=sdp153`** — this must match the **PROJECT** column from `expanse-client user` on Expanse (here **`sdp153`**), *not* the ACCESS award code alone. Your TG project **TG-PHY260100** is separate metadata; Slurm charges against **`sdp153`**. If submission still fails, run `expanse-client user` again and use the **PROJECT** value exactly.
- **Conda env `plant`** with Miniconda installed under **`$HOME/miniconda3`**. Each script runs:
  - `CONDA_ROOT="${CONDA_ROOT:-$HOME/miniconda3}"`
  - `source "${CONDA_ROOT}/etc/profile.d/conda.sh"`
  - `conda activate plant`  
  If you installed Miniconda elsewhere, either export `CONDA_ROOT` in the script header or set it when submitting: `CONDA_ROOT=/path/to/miniconda3 sbatch slurm/04_cfm.sh`.

---

### Step-by-step: Miniconda, conda `plant`, and Python packages

Do this **once** on a login node (from your home directory is fine if Lustre scratch is not writable).

**1. Install Miniconda under `$HOME`**

```bash
cd ~
curl -L -o Miniconda3.sh "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
bash Miniconda3.sh -b -p "$HOME/miniconda3"
rm -f Miniconda3.sh
```

**2. Load conda in every new shell** (add to `~/.bashrc` if you like):

```bash
source "$HOME/miniconda3/etc/profile.d/conda.sh"
```

**3. Accept Anaconda channel Terms of Service** (required for recent `conda`; run once):

```bash
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
```

**4. Create the `plant` environment (Python 3.11)**

```bash
conda create -y -n plant python=3.11 pip
conda activate plant
python --version    # expect 3.11.x
```

**5. Install PyTorch and project dependencies** (in the cloned repo)

```bash
pip install -U pip setuptools wheel

# For GPU training (Steps 04 / 05)—adjust cu121 vs cu124 to match https://pytorch.org/get-started/locally/
pip install "torch>=2.1,<2.6" torchvision --index-url https://download.pytorch.org/whl/cu121

# CPU-only testing on login nodes:
# pip install "torch>=2.1,<2.6" torchvision --index-url https://download.pytorch.org/whl/cpu

cd /path/to/PLANT_GW_Paleontology
pip install -r requirements.txt
mkdir -p logs
```

**6. Optional — wider module tree:** if you use site compilers/MPI, run `module load sdsc` and then **`module load cpu` *or* `module load gpu`** (never both). The batch scripts already run the appropriate `module load` before `python`.

---

### Running the pipeline with SLURM (`sbatch`)

1. **`cd`** into **`PLANT_GW_Paleontology/`** (the directory that contains `slurm/`, `scripts/`, and `data/`).
2. Confirm **`conda activate plant`** works interactively and `python -c "import torch; print(torch.__version__)"` succeeds.
3. Submit jobs from § **Full pipeline execution** below (e.g. `sbatch slurm/00_data_gen.sh`). Logs appear under **`logs/`**.
4. Each `slurm/*.sh` exports **`SLURM_CONF=/etc/slurm/slurm.conf`** when that file is readable, so **`sbatch`** avoids DNS “configless” lookup failures on some login nodes.
5. If jobs fail with “conda: command not found” or “Could not find conda environment: plant”, fix **`CONDA_ROOT`** or recreate the `plant` env as above.

**Site conda instead of Miniconda:** if you prefer `module load anaconda3`, create **`plant`** with `python=3.11` there, then either symlink that env or set **`CONDA_ROOT`** to that Anaconda install’s root so `source "${CONDA_ROOT}/etc/profile.d/conda.sh"` works inside the batch scripts.

### Full pipeline execution (in order)

The SLURM scripts under `slurm/` are written for **production** settings: Step **03** uses a long CPU run (`--epochs 2000`), Steps **04** / **05** use **full** GPU training (see `slurm/04_cfm.sh`, `slurm/05_posterior_network.sh`). Submit from **`PLANT_GW_Paleontology/`**; each script **`cd`s** to the submit directory, **activates conda env `plant`**, then runs **`module load`** and **`python`**.

**Core pipeline (00 → 05)**

```bash
# Step 00 — SSPC data generation (CPU, shared)
sbatch slurm/00_data_gen.sh

# Step 02 — build dataset (CPU; run after 00 finishes and data/sspc/models_sspc.hdf5 exists)
sbatch slurm/02_build_dataset.sh

# Step 03 — rate network, full training budget on CPU
sbatch slurm/03_rate_network.sh

# Step 04 or 04b — generative emulator on GPU (need one checkpoint before Step 05)
sbatch slurm/04_cfm.sh
# sbatch slurm/04b_diffusion.sh   # only if Step 05 will use --emulator diffusion (edit 05 script accordingly)

# Step 04c — optional Naive Bayes baseline (CPU, ~minutes; for CFM/diffusion comparison)
# sbatch slurm/04c_naive_bayes.sh

# Step 05 — posterior network, FullPosteriorNet on GPU (after cfm_final.pt or diffusion_final.pt exists)
sbatch slurm/05_posterior_network.sh
```

**After Step 05 — analysis, validation, and figures (CPU unless you change scripts)**

These use the **same** full checkpoints produced above (e.g. `checkpoints/cfm_final.pt`, `checkpoints/posterior_network_best.pt`). Order is flexible except: **02b** needs Step **02** artifacts; **06** / **06a** need `data/sspc/models_sspc.hdf5`; **07** needs trained **04 + 05** and a GWTC-style CSV.

```bash
# Intrinsic data QA on full parquets (after 02)
sbatch slurm/02b_data_validation.sh

# (sfr_a, mu0) merger-rate heatmaps (Step 00 HDF5; optional linear / averaged axes)
sbatch slurm/00_grid_rate_heatmaps.sh
# Fixed-nuisance HDF5 + linear scale:
# SSPC_HDF5=data/sspc/models_sspc_fixed_nuisance.hdf5 EXTRA='--color-scale linear' sbatch slurm/00_grid_rate_heatmaps.sh

# Intrinsic R(z) curves (recompute cosmic integration; TNG100-fixed nuisances)
VARY=sfra sbatch slurm/00_rate_vs_redshift.sh
# VARY=mu0 EXTRA='--log-y' sbatch slurm/00_rate_vs_redshift.sh

# Step 00 with TNG100-fixed nuisances (custom output name):
# SSPC_EXTRA='--fixed-nuisance-tng100 --output-hdf5 data/sspc/models_sspc_fixed_nuisance.hdf5' sbatch slurm/00_data_gen.sh

# Figure 5–style distribution panels + merger-rate vs z (optional TNG paths in script / CLI)
sbatch slurm/06a_distribution_analysis.sh

# Post-training population figures from SSPC HDF5 (rate-weight vs z, M1/M2/q slices)
sbatch slurm/06_population_figures.sh

# GWTC-style catalog → posterior marginals (--model full in the SLURM script)
export EVENTS_CSV=/path/on/expanse/to/gwtc_events.csv   # or rely on data/gwtc_sample_events.csv if present
sbatch slurm/07_gwtc_validate.sh
```

**Optional — K-member epistemic ensemble (full CFM + full posterior per array index)**

Edit `#SBATCH --array=1-3` to `1-K`. Launch **05** ensemble only after the matching **04** ensemble jobs succeed.

```bash
sbatch slurm/04_cfm_ensemble.sh
# Capture job id, then e.g.:
# sbatch --dependency=afterok:<CFM_ENSEMBLE_JOBID> slurm/05_posterior_ensemble.sh
sbatch slurm/05_posterior_ensemble.sh
```

**Optional — chained dependencies (same order, fewer manual waits)**

```bash
J00=$(sbatch --parsable slurm/00_data_gen.sh)
J02=$(sbatch --parsable --dependency=afterok:${J00} slurm/02_build_dataset.sh)
J03=$(sbatch --parsable --dependency=afterok:${J02} slurm/03_rate_network.sh)
J04=$(sbatch --parsable --dependency=afterok:${J02} slurm/04_cfm.sh)
J05=$(sbatch --parsable --dependency=afterok:${J04} slurm/05_posterior_network.sh)
J02b=$(sbatch --parsable --dependency=afterok:${J02} slurm/02b_data_validation.sh)
J06a=$(sbatch --parsable --dependency=afterok:${J00}+${J02} slurm/06a_distribution_analysis.sh)
J06=$(sbatch --parsable --dependency=afterok:${J00} slurm/06_population_figures.sh)
J07=$(sbatch --parsable --dependency=afterok:${J05} slurm/07_gwtc_validate.sh)
```

(Adjust dependencies if you use **04b** instead of **04**, or if **06a** needs paths that exist only after **00**.)

**Optional:** `sbatch slurm/smoke_test.sh` — short GPU sanity pass; not a substitute for the full jobs above.

Use `squeue -u $USER` to monitor jobs. Logs go to `logs/`.

### Resource summary

| Step | Script | Partition | CPUs | GPUs | Memory | Est. Wall-time |
|------|--------|-----------|------|------|--------|----------------|
| 00 | Data generation | shared | 16 | — | 64 GB | 4 h |
| 02 | Build dataset | shared | 8 | — | 32 GB | 1 h |
| 03 | Rate network | shared | 4 | — | 16 GB | 1 h |
| 04 | CFM emulator | gpu-shared | 10 | 1×V100 | 96 GB | 24 h |
| 04b | Diffusion emulator | gpu-shared | 10 | 1×V100 | 96 GB | 24 h |
| 04c | Naive Bayes fit | shared | 4 | 0 | 32 GB | ~1 h |
| 05 | Posterior network | gpu (see script) | 8 | 1×GPU | 64 GB | 6–24 h |
| 04 (ens.) | CFM ensemble member | gpu-shared | 10 | 1×V100 | 96 GB | 24 h (×K) |
| 05 (ens.) | Posterior ensemble | gpu | 8 | 1×GPU | 64 GB | 6–24 h (×K) |
| 00 (heatmaps) | Grid rate heatmaps | shared | 4 | — | 16 GB | 1 h |
| 00 (R(z)) | Rate vs redshift | shared | 4 | — | 32 GB | 2 h |
| 06a | Distribution analysis | shared | 4 | — | 32 GB | 8 h |
| 02b | Data validation (full) | shared | 4 | — | 64 GB | 4 h |
| 06 | Population figures | shared | 4 | — | 32 GB | 4 h |
| 07 | GWTC validate | shared | 4 | — | 16 GB | 2 h |
| 07b | Synth vs real GW | shared | 4 | — | 16 GB | 2 h |
| 08 | GWTC-4 validation figs | shared | 4 | — | 32 GB | 8 h |
| 09 | Fig2 spread / emulator m₁ | shared | 4 | — | 32 GB | 4 h |
| 04 plots | CFM / diff / NB plot suites | gpu-shared | 4–10 | 0–1 | 32–96 GB | 1–4 h |

All GPU jobs use `gpu-shared` (not exclusive), saving allocation.

Step 05: `slurm/05_posterior_network.sh` uses the **full** model on GPU; the **lite** model can be trained on CPU for debugging (see [Step 05 — Posterior network](#step-05--posterior-network-05_posterior_networkpy)). Adjust partition (`gpu`, `gpu-shared`, …) to match your site.

---

## Local quick-start

**Plain language:** the lines below are **typed in a terminal** in order: they (1) create a small isolated Python install, (2) run a *tiny* version of the science steps so you can see that everything is wired, and (3) run a very small training **test** for the “inverse” network. A full research run uses larger settings and more time.

```bash
cd PLANT_GW_Paleontology
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Tiny smoke-test run (CPU, ~5 min)
python scripts/00_sspc_data_generation.py --n-sfra 4 --n-mu0 4 --n-events 5000
python scripts/02_build_dataset.py --hdf5 data/sspc/models_sspc.hdf5 --data-source sspc
python scripts/03_rate_network.py --epochs 200
# Generative emulator (required before Step 5) — at minimum run CFM smoke (writes checkpoints/cfm_final.pt)
python scripts/04_cfm_emulator.py --smoke-test --steps 500
# optional second emulator for comparison, not both required for Step 5 in one go:
# python scripts/04b_diffusion_emulator.py --smoke-test --steps 500

# Optional baseline (CPU, seconds if 02 artifacts exist):
python scripts/04c_naive_bayes_emulator.py

# Posterior: trains on **synthetic catalogs** from the frozen CFM (or diffusion, see --emulator)
python scripts/05_posterior_network.py --emulator cfm --model lite --epochs 10 --device cpu --batch-size 4 --n-max-events 32
# ablation with NB baseline:
# python scripts/05_posterior_network.py --emulator naive_bayes --model lite --epochs 10 --device cpu --batch-size 4 --n-max-events 32

# Smoke test for posterior modules only
python test/test_posterior_network.py

# Analysis / validation (see scripts/analysis/README.md)
python scripts/analysis/02_validation.py
python scripts/analysis/00_population_figures.py --sspc-hdf5 data/sspc/models_sspc.hdf5
python scripts/analysis/05_gwtc_validate.py --events-csv data/gwtc_sample_events.csv --emulator cfm
python test/test_ensemble_posterior.py
```

### Local / laptop vs ACCESS Expanse (same Python entrypoint)

| Step or tool | Local / debug | Production (SLURM) |
|----------------|---------------|---------------------|
| 00 | `--n-sfra 4 --n-mu0 4 --n-events 5000` (README quick-start) | `slurm/00_data_gen.sh` |
| 02 | `python scripts/02_build_dataset.py --hdf5 data/sspc/models_sspc.hdf5 --data-source sspc` | `slurm/02_build_dataset.sh` |
| 02 validation | `python scripts/analysis/02_validation.py` | `slurm/02b_data_validation.sh` |
| 03 | `python scripts/03_rate_network.py --epochs 200` | `slurm/03_rate_network.sh` |
| 04 | `python scripts/04_cfm_emulator.py --smoke-test --steps 500 --device cpu` | `slurm/04_cfm.sh` |
| 04b | `python scripts/04b_diffusion_emulator.py --smoke-test --steps 500` | `slurm/04b_diffusion.sh` |
| 04c | `python scripts/04c_naive_bayes_emulator.py` | `slurm/04c_naive_bayes.sh` |
| 04 ensemble | `bash scripts/train_cfm_ensemble.sh 3` (tune `EXTRA_ARGS`) | `slurm/04_cfm_ensemble.sh` |
| 05 | `python scripts/05_posterior_network.py --model lite --epochs 10 --device cpu` | `slurm/05_posterior_network.sh` |
| 05 ensemble | `bash scripts/train_posterior_ensemble.sh` | `slurm/05_posterior_ensemble.sh` |
| 00 distribution | `python scripts/analysis/00_distribution_compare.py` | `slurm/06a_distribution_analysis.sh` |
| 00 grid heatmaps | `python scripts/analysis/00_grid_rate_heatmaps.py` | `slurm/00_grid_rate_heatmaps.sh` |
| 00 R(z) curves | `python scripts/analysis/00_rate_vs_redshift.py --vary sfra` | `slurm/00_rate_vs_redshift.sh` |
| 00 fixed nuisances | `python scripts/00_sspc_data_generation.py --fixed-nuisance-tng100 …` | `SSPC_EXTRA=… sbatch slurm/00_data_gen.sh` |
| 00 population | `python scripts/analysis/00_population_figures.py` | `slurm/06_population_figures.sh` |
| 05 GWTC | `python scripts/analysis/05_gwtc_validate.py --events-csv data/gwtc_sample_events.csv` | `slurm/07_gwtc_validate.sh` |
| 05 ensemble infer | `python scripts/analysis/05_ensemble_infer.py --synthetic-bag --member-dirs …` | Usually local |

---

## End-to-end flow (SSPC)

**In plain English:** Start from **simulated stellar binaries**, spread them across cosmic time (**00**), and turn the result into **tables and splits** (**02**). **03** regresses the **total intrinsic merger rate** in each grid cell (from `sum_weight` in 02) from the nine SSPC parameters. The **pivot** in the *PopFlow* story is **not** to jump straight from 02 to “inverse physics”: you first train a **fast generator** (**04** CFM *or* **04b** diffusion) that, given the same Λ you stored in 02, **invents synthetic merger *fingerprints*** so that, conditional on Λ, the distribution of observables **matches 02’s intrinsic** training data (`all_events`)—the **full** merger distribution, not “what LIGO would have seen.” **Then** **05** asks: *if the only data I get are bags of *those* synthetic events (the ones the generator would make), can I *recover* the Λ that I fed the generator?* The generator’s weights are **frozen**; only the “inverse” network trains here—exactly the “end-to-end *through* the generative model” design in the research proposal, without pretending 04/04b is optional. (Full proposal *Stage 3*—a separate *selection* network and PE-style noise—can be added later; this codebase matches the *emulator → posterior* spine.)

```mermaid
flowchart TB
  bps[bps_input]
  s00[00 SSPC]
  s02[02 dataset]
  parq[parquet tables]
  hp[encoded CSV splits]
  s03[03 rate MLP]
  s04[04 CFM or 04b Diffusion]
  emu[emulator .pt]
  s05[05 posterior]
  s06[06 pop figures]
  s07[07 GWTC val]
  bps --> s00
  s00 --> s02
  s02 --> parq
  s02 --> hp
  hp --> s03
  parq --> s04
  hp --> s04
  s04 --> emu
  emu --> s05
  hp --> s05
  s05 --> pckpt[posterior .pt]
  s00 --> s06
  pckpt --> s07
  emu --> s07
```

1. **00** — BPS + cosmic integration → `models_sspc.hdf5` (grid of intrinsic catalogs by channel and `(sfr_a, μ₀)` with nuisance draws).  
2. **02** — Per-cell samples + Λ encoding + **train/val/test** + `obs_normalizer` + parquets for **emulator training** (and other analyses).  
3. **03** — MLP: Λ (with channel) → log₁₀ per-grid **intrinsic** merger-rate total (`sum_weight`).  
4. **04 *or* 04b** — **Train one** (or both for comparison) **generative** model on **02’s `all_events`** (intrinsic distribution); **save** `checkpoints/cfm_final.pt` or `diffusion_final.pt` (includes `lambda_*` column list + normalizer). **Step 5 does not use the parquets directly for its training loss**—it uses **fresh draws** from this checkpoint. Optional **04c** fits `naive_bayes_final.pt` (baseline, CPU).  
5. **05** — **Transformer + flow**; each batch, **Λ** from 02, **synthetic** events = **emulator(Λ)** (frozen, **no** gradients into the generator); NLL to recover Λ. *Proposal alignment:* synthetic catalogs come **from the same CFM (or diffusion) you trained in 04/04b**, **after** 04/04b completes—not in parallel to skip them.  
6. **06 (optional, post-train)** — **Forward** **intrinsic** figures from the SSPC HDF5 (rate–weight vs *z*, masses in *z* slices) for paper-ready definitions.  
7. **07 (optional)** — **Validate** the posterior on a **GWTC-style CSV** (masses, spin, *z*); not full PE, not a skymap.

(An optional **epistemic ensemble** trains **K** of **04+05** with distinct seeds and combines inference with `scripts/analysis/05_ensemble_infer.py` — it is not a new pipeline integer, just a **mode**.)

---

## Pipeline overview

**How this section is organized:** For each step you will see, in order, **(1) In plain English** — a story version; **(2) What it does (technical)** — the precise scientific / ML operations; and **(3) inputs, outputs, and commands** as needed.

### Step 00 — SSPC Data Generation (`00_sspc_data_generation.py`)

**In plain English:** Imagine a huge catalog of *possible* star systems that could exist in a galaxy, produced by a detailed stellar-evolution code (COMPAS / GROWL). This step “rolls the dice” in a very structured way: it asks how many mergers would have happened over the history of the universe under different *global* assumptions (how many stars were born over time, how “metal-rich” gas was, and a handful of *random-looking* but physically meaningful *nuisance* numbers drawn for each grid point). The **output** is a library of *synthetic* mergers: each has a few summary numbers and a *weight* saying how *important* that kind of event was in the model. This step does **not** ask “did LIGO see it?”; that comes later. Think of it as the **unfiltered universe the model believes in**.

**What it does (technical):** Performs cosmic integration of Binary Population Synthesis (BPS) output over star formation history and metallicity evolution (Madau-Dickinson / Neijssel+19 models) to produce a grid of predicted GW merger event catalogs representing the **intrinsic** population — no detection-probability weighting is applied.

**Input:**
- `data/bps_output.h5` — COMPAS/GROWL BPS simulation output (3.35M binaries) with columns: `m1`, `m2`, `t_delay`, `metallicity`, `channel` (SMT/CE/CHE).

**Output:** `data/sspc/models_sspc.hdf5` (default) or any path via `--output-hdf5`. HDF5 keys: `/CHANNEL/sfra{NNNN}/mu0{MMMM}` (e.g. `/SMT/sfra0170/mu00250`).

Each key holds **50,000** sampled events (default `--n-events`) with columns:

| Column | Meaning |
|--------|---------|
| `mchirp`, `q`, `z` | Source-frame chirp mass, mass ratio, merger redshift |
| `weight` | Per-binary intrinsic merger-rate weight [merger/yr] (cosmic integration) |
| `intrinsic_rate_yr` | Total intrinsic rate for this grid cell (sum over BPS before sampling) |
| `rate_per_gpc3_yr` | `intrinsic_rate_yr / V_comov(z ≤ 10)` [Gpc⁻³ yr⁻¹] |
| `sspc_sfr_a` … `sspc_alpha_skew` | SSPC parameters used for that cell |

**Parameters scanned (centred on TNG100-1 best-fit values from Briel+):**

| Parameter | Symbol | Range | TNG100-1 best-fit | Grid type |
|-----------|--------|-------|-------------------|-----------|
| SFR amplitude | `sfr_a` | 0.010 – 0.030 | ≈ 0.017 | **primary axis** |
| Mean metallicity (z=0) | `mu0` | 0.010 – 0.060 | ≈ 0.025 | **primary axis** |
| SFR rising slope | `sfr_b` | 1.0 – 3.0 | ≈ 1.456 | nuisance |
| SFR turnover redshift | `sfr_c` | 2.0 – 6.0 | ≈ 4.514 | nuisance |
| SFR falling slope | `sfr_d` | 4.0 – 8.0 | ≈ 6.210 | nuisance |
| Metallicity evo. slope | `muz` | −0.5 – 0.1 | ≈ −0.052 | nuisance |
| Metallicity log-spread | `sigma0` | 0.5 – 1.5 | ≈ 1.151 | nuisance |
| Spread redshift evo. | `sigmaz` | −0.1 – 0.1 | ≈ 0.047 | nuisance |
| Log-skew skewness | `alpha_skew` | −2.0 – 2.0 | ≈ −1.854 | nuisance |

**Grid size:** `--n-sfra` × `--n-mu0` × 3 channels (SMT, CE, CHE).  
Production: **50 × 50 × 3 = 7,500** cells × **50,000** events ≈ **375M** stored mergers (~32 GB per full HDF5).

**Nuisance handling (CLI):**

| Mode | Flag | Behaviour |
|------|------|-----------|
| Default | *(none)* | Random draw of `sfr_b,c,d`, `muz`, `sigma0`, `sigmaz`, `alpha_skew` **per grid cell** |
| TNG100 fixed | `--fixed-nuisance-tng100` | All seven nuisances fixed to TNG100-1 best-fit on every cell; only `sfr_a` and `mu0` vary |
| Fixed merger z | `--fixed-z Z` | Every sampled event gets redshift `Z` (cosmic weights still integrate over all z) |

**Example — fixed nuisances, custom output (still under `data/sspc/`):**

```bash
python scripts/00_sspc_data_generation.py \
  --n-sfra 50 --n-mu0 50 --n-events 50000 \
  --fixed-nuisance-tng100 \
  --output-hdf5 data/sspc/models_sspc_fixed_nuisance.hdf5 \
  --overwrite
```

**SLURM:** `SSPC_EXTRA='--fixed-nuisance-tng100 --output-hdf5 data/sspc/models_sspc_fixed_nuisance.hdf5' sbatch slurm/00_data_gen.sh`

**Step 02 on a non-default HDF5** (isolated under `data/ml_20x20_z02/`, including its own `checkpoints/obs_normalizer.json`):

```bash
HDF5=data/sspc/models_sspc_20x20_z02_fixed_nuisance.hdf5 OUT_DIR=data/ml_20x20_z02 sbatch slurm/02_build_dataset.sh
```

**Example — 20×20, fixed nuisances, events at z = 0.2:**

```bash
SSPC_EXTRA='--n-sfra 20 --n-mu0 20 --fixed-nuisance-tng100 --fixed-z 0.2 --output-hdf5 data/sspc/models_sspc_20x20_z02_fixed_nuisance.hdf5' \
  sbatch slurm/00_data_gen.sh
```

**Key design:**
- Cosmic integration: per-binary weight = Σ_z `feff × SFR(z_form) × dP(metallicity) × ΔV_comov` (Madau–Dickinson + Neijssel+19-style metallicity; see script docstring).
- Events sampled from the **intrinsic** merger-rate distribution (no detection probability).
- `chieff` is **not** in current BPS output; spins are assigned in downstream steps if needed.

---

### Step 02 — Build ML Dataset (`02_build_dataset.py`)

**In plain English:** The huge HDF5 from Step 00 is not yet in a shape that training code can **mini-batch**. This step (a) **draws random subsamples of mergers** from the **intrinsic** population (merger-rate weights), and (optionally) a **separate** table subsampled with detection accept/reject for side analyses; (b) **remembers which universe-settings** (the nine main numbers) went with each subsample; and (c) **writes everything as ordinary tables** on disk, plus a **normalizer** so masses and redshifts are not mixed on wildly different numeric scales. It also **holds out** some settings for *validation* and *testing* so the team can’t accidentally cheat by memorizing the whole universe. A non-expert can think of it as: **“turn the scientific archive from 00 into a clean spreadsheet and train/test groups for the classifiers and generators to come.”**

**What it does (technical):** Reads the HDF5, samples events per grid point (intrinsic by default; optional detection-subsampled copy), encodes hyperparameters, writes ML-ready parquet + JSON under the **current working directory** (default `.`). **Emulator and normalizer** use the **intrinsic** table and `sum_weight` for rates—not detector-weighted paths.

**Input:** SSPC output, e.g. `data/sspc/models_sspc.hdf5` — pass `--hdf5` and `--data-source sspc` (or use `--data-source auto`).

**Outputs (typical names in `PLANT_GW_Paleontology/`):**
- `hyperparam_table.csv` — one row per `(channel, grid key)` with `chi_b` / `alpha_CE` carrying `sfr_a` / `mu0` for SSPC, `sum_weight`, `sum_pdet`, `n_systems`, etc.
- `hyperparam_table_encoded.csv` — extended table with `channel_id`, one-hot, `lambda_*`, and **`sspc_*_mean` / `sspc_*_std`** columns for the nine SSPC parameters (used by 03 and 05).
- `all_events.parquet` — intrinsic merger-rate–weighted samples (`N_SAMPLE` per grid, default 5000); columns include `mchirp`, `q`, `chieff`, `z`, `grid_idx`, `pdet`, `lambda_*`, and per-row `sspc_*` if present. **This is the default table for 04/04b** (and the source for `obs_normalizer.json`).
- `all_detected_events.parquet` — optional detection-subsampled table (`N_DET` per grid, default 2000) for legacy or diagnostic workflows; **not** used to train 03/04/04b/05 in the main pipeline.
- `splits.json` — `train` / `val` / `test` **indices** into the encoded hyperparam table (stratified by channel).
- `checkpoints/obs_normalizer.json` — per-observable mean and `std` after the same transforms as below (used by **04 and 05**).

**Observable transforms in the normalizer (computed from `all_events` rows):**  
`mchirp` → `log10(max(mchirp, 1e-3))`, then z-score; `z` → `log10(max(z, 0.1))`, then z-score; `q` and `chieff` are z-scored in physical space.

**Key flags:** `--n-sample`, `--n-det`, `--out-dir`, `--data-source sspc`.

---

### Step 03 — Rate Network (`03_rate_network.py`)

**In plain English:** A **rate** here is the **total intrinsic merger rate** in each grid cell, in log₁₀ space, for each *combo* of channel (three broad formation stories) and the **nine** astrophysical numbers—**not** “how many events LIGO would count.” This step trains a *small* neural map: **if you hand it the table row describing the settings, it outputs one scalar** summarizing how much merger *activity* that cell carries under the physical weights from 00–02. It is a **regression** problem (predict a number), and it also trains a **Gaussian process** baseline to compare against.

**What it does (technical):** Trains an MLP to predict `log10(sum_weight)` (total **intrinsic** merger rate per grid cell) from **SSPC hyperparameters** (`sum_weight` in `hyperparam_table_encoded.csv`; for Zenodo-only tables without that column, the code can fall back to `sum_pdet`).

**Inputs (defaults are next to the script):** `hyperparam_table_encoded.csv`, `splits.json`. The CSV must contain the nine `sspc_*_mean` columns.

**Feature vector (12-D):** `[CE, CHE, SMT]` one-hot indicators + nine normalized SSPC means (`sfr_a` … `alpha_skew`), each min–max scaled using the training split.

**Architecture:** `RateNet`: 12 → (64, 32) with LayerNorm + GELU → 1; wrapped in `NormalizedNet` for z-scored targets.

**Target:** `log10(sum_weight)` floored at `-5` (per-grid **intrinsic** rate total; SSPC: `sum_pdet` in the table equals this by construction).

**Training:** Adam `lr=1e-3` (default in `train()`), weight decay `1e-4`, Huber loss on z-scored targets, ReduceLROnPlateau, early stopping (`--patience`, default 300), up to `--epochs` (default 3000). Also fits a GP baseline for comparison.

**Outputs:** `checkpoints/rate_network_best.pt`, `checkpoints/rate_network_config.json`, `checkpoints/gp_rate_baseline.pkl`, plots under `plots/03_rate_network/<timestamp>/`.

**CLI example:**

```bash
python 03_rate_network.py --epochs 2000 --patience 200 --checkpoint-dir checkpoints
```

---

### Step 04 — CFM Emulator (`04_cfm_emulator.py`)

**In plain English:** Instead of *one* number like Step 03, you need a model that answers: **“Given these universe-settings, *draw me realistic-looking mergers* (mass, mass ratio, spin, distance proxy).”** *Conditional flow matching* learns a **smooth transformation** from simple random noise to event data—**conditioned** on Λ. This is the **Stage-2 “emulator”** in the PopFlow proposal: **Step 05 is not an optional add-on**—it is trained **on catalogs produced by this model (or by 04b)** with the emulator **frozen**, so you must complete **04 (or 04b) before 05**.

**What it does (technical):** Trains a Conditional Flow Matching (CFM) model to learn the conditional distribution `p(mchirp, q, chieff, z | λ)` of GW event observables given hyperparameters.

**Input:** `all_events.parquet`, `checkpoints/obs_normalizer.json` (ideally from the same 02 run), `hyperparam_table_encoded.csv`, `splits.json` (resolved from the working directory the same way as 05)

**Architecture (typical):**
- Conditioning encoder: all `lambda_*` columns in `hyperparam_table_encoded.csv` (SSPC: extended vector including SSPC means) → MLP → context (128D).
- Vector field: [obs(4) + context(128) + t(1)] → [512, 512, 512, 512] → 4D (full run).
- Smoke test: `hidden_dim=128`; full training: `hidden_dim=256`.

**Training (full run):** 100,000 steps, batch=256, AdamW lr=1e-4, CosineAnnealingLR, importance-weighted sampling (cap 3.0), gradient clipping (max_norm=1.0).

**Z-jitter:** ±0.05 uniform noise added to discrete `z` values during training to smooth the distribution.

**Output:** `checkpoints/cfm_final.pt`, plots in `plots/04_cfm_emulator/<timestamp>/`.

---

### Step 04b — Diffusion Emulator (`04b_diffusion_emulator.py`)

**In plain English:** This is an **alternative** to the CFM for the **same** role: the **only** fast generator you must have **before** Step 05. You typically train **either** 04 **or** 04b, point Step 5 at `diffusion_final.pt`, and reserve running **both** for ablations. **Diffusion** denoises from noise toward data **conditioned** on Λ.

**What it does (technical):** Same task as CFM (learn `p(obs | λ)`) using a score-based diffusion model as an alternative/complement.

**Architecture:** Same encoder as CFM. Score network: [noised_obs(4) + context(128) + t_embed(32)] → [256×4] → 4D.

**Training:** Same hyperparameters as CFM (100k steps, batch=256, hidden_dim=256 for full run). Uses a cosine noise schedule with `N_TIMESTEPS=50` diffusion steps.

**Output:** `checkpoints/diffusion_final.pt`, plots in `plots/04b_diffusion_emulator/<timestamp>/`.

---

### Step 04c — Naive Bayes Emulator (`04c_naive_bayes_emulator.py`)

**In plain English:** The simplest baseline for “given universe-settings, draw mergers.” It does **not** train a neural network. Instead it reads the same Step 02 tables and stores **per-grid statistics** (means and spreads of mass, mass ratio, and redshift). At inference it either **mixes** nearby grid cells with a Gaussian kernel (`mode=gaussian`, default) or **resamples** events from the closest grid row (`mode=nearest`). Use this to see how much CFM/diffusion improve over a classical shortcut.

**What it does (technical):** Fits `NaiveBayesEmulator` from `all_events.parquet` + `hyperparam_table_encoded.csv`: per `grid_idx`, diagonal Gaussian in normalized (mchirp, q, z) space; Λ-kernel weights π_g(Λ) ∝ exp(−‖Λ−Λ_g‖²/2τ²). No gradients; runtime fit is seconds on CPU.

**Input:** Same as Step 04 (`all_events.parquet`, `hyperparam_table_encoded.csv`, `checkpoints/obs_normalizer.json`).

**CLI:**

```bash
python scripts/04c_naive_bayes_emulator.py
python scripts/04c_naive_bayes_emulator.py --mode nearest --bandwidth 0.05
```

**Output:** `checkpoints/naive_bayes_final.pt`, validation plots in `plots/04c_naive_bayes_emulator/<timestamp>/`.

---

### Step 05 — Posterior network (`05_posterior_network.py`)

**In plain English — what this step is for**

- **Goal:** learn a *fast* mapping from a **set of merger “fingerprints”** (a catalog) to a **distribution over the astrophysical settings** Λ—the same nine **SSPC** numbers the pipeline uses everywhere (`sspc_*_mean` in the table), while **channel** is fixed by the data row (it is not predicted in this model).
- **What “posterior” means here:** a **conditional density** *p*(Λ | catalog), not a single point estimate. The network is trained so that, when the catalog was produced from the *true* Λ for that row, the model assigns that Λ **high** log-probability, and the flow can also **draw samples** of plausible Λ for the same catalog.
- **What is *not* happening:** Step 5 does **not** train on a fixed read of `all_events.parquet` or on real detections. Every catalog in the loss is a **new synthetic draw** from the **already trained, frozen** Step 4 model (CFM or diffusion). The emulator is **read-only** here (no backprop into 04/04b), so the inverse map is explicitly the inverse of **what your emulator actually does**, not of the original HDF5 alone.
- **What one training example is:** one **row** of `hyperparam_table_encoded.csv` gives (1) the **label** Λ = nine `sspc_*_mean` values and (2) the `lambda_*` vector fed to the emulator. The emulator generates up to **`n_max_events` mergers**; each merger becomes an **8-D** feature vector. The **task** is to maximize *p*(Λ | that bag of features). The model must be insensitive to the **order** of events in the bag.
- **Amortized:** you train **once**; at use time, **one forward pass** (plus flow sampling) answers “what Λ is plausible for this catalog?” for **any** new catalog, without a separate MCMC run per case.
- **Out of scope in this script:** LIGO **selection** and per-event **parameter estimation (PE)** on strain are *not* implemented; they are natural add-ons, not part of this file.
- **How this sits in the chain:** Step **04/04b** learns a forward model *p*(events | Λ). Step **05** learns its **amortized inverse** *p*(Λ | catalog) **under the same forward process** (same frozen net, same featurization). That is why training never reads a static 02 parquet for the loss—only **fresh** emulator draws match the generator the posterior must invert at test time.

**What it does (technical)**

1. **Load** `hyperparam_table_encoded.csv`, `splits.json`, and a frozen 04/04b checkpoint containing `model_state`, `obs_normalizer`, and `lambda_cols` (names must match the table used when training 04/04b).
2. For each **batch** of table rows, the dataset calls **`generate_catalog(λ, n_max_events, model, normalizer)`** to create a **stochastic** catalog, then `build_events_8d` to tensor `(B, L, 8)` with a padding **mask** (fixed `L = n_max_events` in typical runs). The **supervised label** for the flow is the same row’s **nine** `sspc_*_mean` values, **z-scored** with mean and std from the **train split** only (buffers + JSON config). **Channel** is implicit in the row (it affects which `lambda_*` and emulator draw you get), but the nine predicted means are the Λ **vector** the flow models.
3. A **set encoder** (Transformer, no positional encodings) maps variable-length (masked) event sets to one **context vector** per row. A **conditional RealNVP** models *p*(Λ | context) with exact **log p** and **sampling** via the inverse map.
4. **Optimization** minimizes **NLL = −log *p*(Λ | catalog)** (batch-meaned), with Adam, optional gradient clipping, AMP on CUDA, and **early stopping** on **validation** NLL. **Best** validation `state_dict` is saved. Gradients do **not** flow into the emulator.

**When to run:** After **02** and after **at least one of 04 or 04b** has written `checkpoints/cfm_final.pt` or `diffusion_final.pt`. **Step 03 is not** required.

#### Inputs

| Artifact / object | Role |
|----------|------|
| `hyperparam_table_encoded.csv` | True Λ: nine `sspc_*_mean` + same `lambda_*` columns the emulator was trained on. |
| `splits.json` | Train / val / test row indices. |
| `checkpoints/cfm_final.pt` **or** `diffusion_final.pt` | **Frozen** generator; `05` reads `model_state`, `normalizer`, `lambda_cols` (must **match** the 02 table 04/04b used). **Event featurization** uses the `normalizer` **from this file** so it stays consistent with the emulator. |
| (Context) | Step 04/04b trained on **`all_events.parquet`**; Step 05 does not read that parquet for its loss (only the frozen emulator checkpoint). |

#### What each merger becomes (8-D features)

`build_events_8d` turns each emulated **(mchirp, q, chieff, z)** row into **eight** numbers: the four observables are z-scored using the **`obs_normalizer` inside the 04/04b checkpoint** (same convention as training the emulator); **four extra channels** repeat the per-dimension “σ” from that normalizer so the network sees both value and scale. The dataset builds tensors `(B, L, 8)` with a **mask** for padding. Each training **epoch** nudges the random seed (`set_epoch`) so CFM ODE or diffusion noise does not repeat identically every pass. **`--n-max-events`** sets `L` (default `256`).

**Architectures — what they are in general, and what they do here**

| Component | In general (ML) | In this project |
|-----------|------------------|-----------------|
| **Set Transformer encoder** | A Transformer that treats a **set** of items (here: events) as **unordered**: no positional encodings, and attention is masked so padding is ignored. A permutation of events should not change the output. The stack outputs one vector per event; these are **mean-pooled** (masked) to one **catalog vector**. | **Input:** `(batch, L, 8)` + mask. **Output:** a single **context** vector per catalog, dimension `H` (e.g. 128 in **lite**). This vector summarizes the whole synthetic bag *before* any decision about Λ. |
| **Conditional RealNVP (flow)** | A **normalizing flow**: a composition of **invertible** blocks so the model defines an explicit **density** and can **sample** by pushing Gaussian noise through the inverse map. *Conditional* means scale/translation in each block can depend on an external **context** vector. | **Data** for the flow are the **nine** Λ values **z-scored** with train-split statistics. **Context** = catalog embedding from the encoder. The flow parameterizes *p*(Λ \| catalog); **NLL = −log *p* at the true Λ** is the training loss. **`sample`:** draw **z** ~ 𝒩(0, **I**), then invert the flow to get a catalog-consistent Λ in physical units. |

| Variant | `LitePosteriorNet` | `FullPosteriorNet` |
|--------|---------------------|--------------------|
| Encoder layers / heads / `d_model` | 2 / 4 / 128 | 6 / 8 / 512 |
| FFN dim | 512 | 2048 |
| Flow coupling layers / MLP width | 4 / 64 | 10 / 256 |
| Typical params | ~0.5M | ~20M+ (large GPU) |
| Use case | Smaller grids, CPU or quick GPU | Large grids, long runs on a strong GPU |

**Implementation:** `models/posterior_network_lite.py` and `models/posterior_network_full.py`. Training uses Adam, optional **`--max-grad-norm`**, **`--amp`** (CUDA), and **`--accum-steps`**.

**API (both variants):** `log_prob(theta, events, event_mask)` → per-example scalar; `sample(events, event_mask, num_samples)` → `(B, num_samples, 9)` **physical** Λ; `encode_catalog` → catalog embeddings. **Buffers** `theta_mean` / `theta_std` (train split) are also stored in `posterior_network_config.json`.

#### Outputs

- `checkpoints/posterior_network_best.pt` — `state_dict` at best validation NLL.  
- `checkpoints/posterior_network_config.json` — `sspc_theta_param_cols`, `input_event_dim` (= 8), `norm_meta` (`theta_mean`, `theta_std`, and the `obs_normalizer` snapshot).  
- `plots/05_posterior_network/<timestamp>/learning_curves.png` — train/val NLL.

#### Command-line (run from `PLANT_GW_Paleontology/`)

```bash
# Default: --emulator cfm, load checkpoints/cfm_final.pt, lite posterior, --num-workers 0
python 05_posterior_network.py

# Use diffusion instead (must have trained 04b)
python 05_posterior_network.py --emulator diffusion --emulator-checkpoint checkpoints/diffusion_final.pt

# Full model + GPU
python 05_posterior_network.py --emulator cfm --model full --amp --device cuda \
  --batch-size 8 --n-max-events 256 --epochs 200 --patience 30

# Custom paths
python 05_posterior_network.py \
  --emulator cfm \
  --emulator-checkpoint ./checkpoints/cfm_final.pt \
  --hyperparam-csv ./hyperparam_table_encoded.csv \
  --splits-json ./splits.json \
  --checkpoint-dir ./checkpoints
```

| Flag | Default | Meaning |
|------|---------|---------|
| `--emulator` | `cfm` | `cfm`, `diffusion`, or `naive_bayes` — which generative checkpoint to load. |
| `--emulator-checkpoint` | `checkpoints/cfm_final.pt`, `diffusion_final.pt`, or `naive_bayes_final.pt` | Path to 04/04b/04c **final** save (contains `model_state`, `normalizer`, `lambda_cols`). |
| `--model` | `lite` | `lite` or `full` (posterior). |
| `--epochs` | 200 | Max training epochs. |
| `--patience` | 30 | Early stopping on val NLL. |
| `--batch-size` | 8 | Catalogs per batch (keep modest—each needs ODE/diffusion samples). |
| `--lr` | 1e-4 | Adam learning rate. |
| `--n-max-events` | 256 | Emulated events per catalog per step. |
| `--max-grad-norm` | 1.0 | Gradient clip (set `0` to disable). |
| `--num-workers` | `0` (forced) | **Must** stay 0; on-the-fly generation runs in the main process. |
| `--seed` | 42 | RNG. |
| `--amp` | off | CUDA autocast. |
| `--device` | `auto` | Same device for emulator + posterior. |
| `--accum-steps` | 1 | Gradient accumulation. |
| `--output-checkpoint-pt` | (unset) | Optional explicit path for `posterior_network_best.pt` (e.g. `checkpoints/posterior_ensemble/2/…`); config JSON is written next to it. |

**Cluster:** `slurm/05_posterior_network.sh` should run **after** 04/04b completes. It passes `--emulator cfm` and `CUDA`; set `--emulator` / path if you use diffusion. For **K** ensemble members, use `slurm/05_posterior_ensemble.sh` (array job) with matching `checkpoints/ensemble_cfm/`.

**Unit test (no data required):** `python test/test_posterior_network.py` — shape checks, parameter counts, and RealNVP round-trip on synthetic tensors.

---

## Analysis scripts (`scripts/analysis/`)

**In plain English:** Training stops at Step **05**. Everything under `scripts/analysis/` is for **your** checks: data QA, population figures, paper-style comparisons, and posterior sanity tests. Scripts are named by **which pipeline step they inspect** (`00_…`, `02_…`, `04_…`, `05_…`). Shared plotting code lives in `scripts/analysis/lib/`.

Full index: [`scripts/analysis/README.md`](scripts/analysis/README.md).

| Script | Step | What it does |
|--------|------|----------------|
| `00_population_figures.py` | 00 | Rate vs *z*, *m*₁/*m*₂/*q* → `plots/00_population_figures/<timestamp>/` |
| `00_distribution_compare.py` | 00 | TNG vs SSPC Figure 5 + merger-rate vs *z* → `plots/00_distribution_compare/<timestamp>/` |
| `00_fig2_spread.py` | 00 | Figure-2-style SSPC mass marginals → `plots/00_fig2_spread/<timestamp>/` |
| `00_grid_rate_heatmaps.py` | 00 | Intrinsic rate density heatmaps on (`sfr_a`, `mu0`) per channel → `plots/00_grid_rate_heatmaps/<timestamp>/` |
| `00_rate_vs_redshift.py` | 00 | *R(z)* for several `sfr_a` or `mu0` values (TNG100-fixed nuisances; recomputes integration) → `plots/00_rate_vs_redshift/<timestamp>/` |
| `02_validation.py` | 02 | Intrinsic QA → reports `test/reports/validation/<timestamp>/`, plots `plots/02_validation/<timestamp>/` |
| `04_cfm_emulator_plots.py` | 04 | Full CFM validation plots from `cfm_final.pt` |
| `04b_diffusion_emulator_plots.py` | 04b | Full diffusion validation plots from `diffusion_final.pt` |
| `04c_naive_bayes_emulator_plots.py` | 04c | NB marginal / KDE plots from `naive_bayes_final.pt` |
| `04_emulator_m1_compare.py` | 04/04b/04c | CFM vs diffusion vs optional NB *m*₁ → `plots/04_emulator_m1_compare/<timestamp>/` |
| `04_gwtc4_validation.py` | 04 | GWTC-4 paper figures; optional `--naive-bayes-checkpoint` saves NB grids → `plots/04_gwtc4_validation/<timestamp>/` |
| `05_gwtc_validate.py` | 05 | Event CSV → posterior marginals → `plots/05_gwtc_validate/<timestamp>/` |
| `05_synth_real_compare.py` | 05 | Emulator synthetic catalog vs real GW CSV (overlay marginals) |
| `05_ensemble_infer.py` | 05 | *K* posteriors: log-mean density and/or mixture samples |
| `utils/fetch_gwtc40_events.py` | — | GWOSC API → CSV for `05_synth_real_compare.py` |

**Examples:**

```bash
python scripts/analysis/02_validation.py

# Step 00 HDF5 diagnostics (default: data/sspc/models_sspc.hdf5)
python scripts/analysis/00_grid_rate_heatmaps.py --metric rate
python scripts/analysis/00_grid_rate_heatmaps.py \
  --sspc-hdf5 data/sspc/models_sspc_fixed_nuisance.hdf5 \
  --color-scale linear --average-over mu0

python scripts/analysis/00_rate_vs_redshift.py --vary sfra --log-y
python scripts/analysis/00_rate_vs_redshift.py --vary mu0 --n-curves 7

python scripts/analysis/00_distribution_compare.py --sspc-hdf5 data/sspc/models_sspc.hdf5
python scripts/analysis/00_population_figures.py --z-slices 0.2 1.0
python scripts/analysis/04_cfm_emulator_plots.py
python scripts/analysis/04_emulator_m1_compare.py --device cuda
python scripts/analysis/05_gwtc_validate.py --events-csv data/gwtc_sample_events.csv --emulator cfm
python scripts/analysis/05_ensemble_infer.py --synthetic-bag --member-dirs checkpoints/posterior_ensemble/1 checkpoints/posterior_ensemble/2
```

**`00_grid_rate_heatmaps.py` options:** `--sspc-hdf5`, `--hyperparam-csv` (Step 02 table, not recommended for intrinsic rate), `--metric {rate,count,log_rate,rate_weight}`, `--color-scale {log,linear}`, `--colormap {sequential,diverging}`, `--average-over {none,mu0,sfra}`, `--linear-scale` (alias for linear).

**`00_rate_vs_redshift.py`:** `--vary {sfra,mu0}` (required), `--n-curves`, `--values`, `--sspc-hdf5` (curve locations from keys), `--log-y`, `--channels SMT CE CHE`.

`test/validation/run_data_validation.py` is a thin wrapper that calls `02_validation.py`.

**Epistemic ensemble training** (not analysis): `scripts/train_cfm_ensemble.sh`, `scripts/train_posterior_ensemble.sh` + `slurm/04_cfm_ensemble.sh`, `slurm/05_posterior_ensemble.sh`.

---

## Repository layout

**Plain language:** numbered **`scripts/0*.py`** are the training pipeline in order; **`scripts/analysis/`** is diagnostics and paper figures; **`slurm/`** submits those scripts on Expanse; **`data/`** holds inputs and large HDF5 catalogs; **`checkpoints/`** and **`plots/`** hold trained weights and figures. Path helpers live in **`plant_paths.py`**.

### Top level

| Path | Purpose |
|------|---------|
| `README.md` | This document |
| `plant_paths.py` | `PROJECT_ROOT`, `data/` ML paths, `plots/<script_stem>/<timestamp>/` helpers |
| `requirements.txt` | Core Python dependencies (PyTorch, pandas, torchcfm, …) |
| `requirements-optional.txt` | Optional extras (e.g. `sbi`) |
| `.venv311/` | Local venv (optional; Expanse often uses conda `plant`) |

### Pipeline scripts (`scripts/`)

| Script | Step | Role |
|--------|------|------|
| `00_sspc_data_generation.py` | 00 | BPS + cosmic integration → SSPC HDF5 grid |
| `02_build_dataset.py` | 02 | HDF5 → parquets, CSVs, splits, `obs_normalizer.json` |
| `03_rate_network.py` | 03 | MLP: Λ → log₁₀ intrinsic rate per grid cell |
| `04_cfm_emulator.py` | 04 | Conditional flow matching emulator |
| `04b_diffusion_emulator.py` | 04b | Diffusion emulator (alternative to 04) |
| `04c_naive_bayes_emulator.py` | 04c | Naive Bayes baseline emulator |
| `05_posterior_network.py` | 05 | Set transformer + flow: catalog → Λ |
| `train_cfm_ensemble.sh` | — | Shell helper for *K* CFM seeds |
| `train_posterior_ensemble.sh` | — | Shell helper for *K* posterior seeds |

### Analysis (`scripts/analysis/`)

See [`scripts/analysis/README.md`](scripts/analysis/README.md). Summary:

| Script | Step |
|--------|------|
| `00_population_figures.py`, `00_distribution_compare.py`, `00_fig2_spread.py` | 00 population / paper figures |
| `00_grid_rate_heatmaps.py`, `00_rate_vs_redshift.py` | 00 grid diagnostics |
| `02_validation.py` | 02 data QA |
| `04_cfm_emulator_plots.py`, `04b_diffusion_emulator_plots.py`, `04c_naive_bayes_emulator_plots.py` | 04 emulator validation plots |
| `04_emulator_m1_compare.py`, `04_gwtc4_validation.py` | 04 comparisons / GWTC-4 figs |
| `05_gwtc_validate.py`, `05_synth_real_compare.py`, `05_ensemble_infer.py` | 05 inference / ensemble |
| `lib/distribution.py`, `lib/generative_emulator_plots.py`, … | Shared plotting loaders |
| `utils/fetch_gwtc40_events.py`, `utils/selection_effects.py` | GWOSC fetch, selection helpers |

### Models (`models/`)

| Module | Used by |
|--------|---------|
| `rate_network.py` | Step 03 |
| `cfm_emulator.py` | Step 04 |
| `diffusion_emulator.py` | Step 04b |
| `naive_bayes_emulator.py` | Step 04c |
| `posterior_network_lite.py`, `posterior_network_full.py` | Step 05 |
| `ensemble_posterior.py` | `05_ensemble_infer.py` |

### Data (`data/`)

| File / dir | Produced by | Notes |
|------------|-------------|--------|
| `bps_output.h5` | External (COMPAS/GROWL) | ~3.35M DCO binaries; required for 00 |
| `sspc/models_sspc.hdf5` | Step 00 (default) | 50×50×3 grid; random nuisance per cell |
| `sspc/models_sspc_fixed_nuisance.hdf5` | Step 00 + `--fixed-nuisance-tng100` | Same grid; TNG100-fixed nuisances |
| `hyperparam_table.csv`, `hyperparam_table_encoded.csv` | Step 02 | Per-cell aggregates + encoded Λ |
| `all_events.parquet`, `all_detected_events.parquet` | Step 02 | Training tables for 04/05 |
| `splits.json` | Step 02 | Train/val/test indices |
| `gwtc_sample_events.csv` | Manual / utils | Small catalog for 05 smoke tests |

Step 02 also writes `checkpoints/obs_normalizer.json` (under project root when run from `PLANT_GW_Paleontology/`).

### Checkpoints (`checkpoints/`)

| Artifact | Step |
|----------|------|
| `obs_normalizer.json` | 02 |
| `rate_network_best.pt`, `rate_network_config.json`, `gp_rate_baseline.pkl` | 03 |
| `cfm_final.pt`, `diffusion_final.pt`, `naive_bayes_final.pt` | 04 / 04b / 04c |
| `posterior_network_best.pt`, `posterior_network_config.json` | 05 |
| `ensemble_cfm/`, `posterior_ensemble/` | Optional ensemble runs |

### Plots (`plots/`)

Timestamped subfolders per script: `plots/<script_stem>/<YYYY-MM-DD_HH-MM-SS>/`. Examples: `00_grid_rate_heatmaps/`, `00_rate_vs_redshift/`, `03_rate_network/`, `04_cfm_emulator/`, `05_posterior_network/`, `02_validation/`.

### Tests (`test/`)

| Path | Purpose |
|------|---------|
| `test_posterior_network.py` | Unit tests for posterior modules |
| `test_ensemble_posterior.py` | Ensemble combination tests |
| `validation/run_data_validation.py` | Wrapper → `02_validation.py` |
| `reports/validation/` | JSON/text reports from 02 validation |

### SLURM (`slurm/`)

All jobs assume `conda activate plant` and `#SBATCH --account=sdp153` unless you edit the scripts.

| Script | Runs |
|--------|------|
| `00_data_gen.sh` | Step 00 (`SSPC_EXTRA` for extra CLI flags) |
| `02_build_dataset.sh` | Step 02 |
| `03_rate_network.sh` | Step 03 |
| `04_cfm.sh`, `04b_diffusion.sh`, `04c_naive_bayes.sh` | Steps 04 / 04b / 04c |
| `04_cfm_ensemble.sh`, `05_posterior_ensemble.sh` | Ensemble training |
| `05_posterior_network.sh` | Step 05 |
| `00_grid_rate_heatmaps.sh` | Grid heatmaps (`SSPC_HDF5`, `EXTRA`) |
| `00_rate_vs_redshift.sh` | *R(z)* curves (`VARY=sfra` or `mu0`, `EXTRA`) |
| `02b_data_validation.sh` | Step 02 validation |
| `06_population_figures.sh`, `06a_distribution_analysis.sh`, `09_fig2_spread.sh` | Step 00 figures |
| `04_cfm_emulator_plots.sh`, `04b_diffusion_emulator_plots.sh`, `04c_naive_bayes_emulator_plots.sh` | Emulator plot suites |
| `07_gwtc_validate.sh`, `07b_synth_real_validation.sh` | Step 05 GW validation |
| `08_gwtc4_validation.sh`, `08_smoke_gwtc4_fig2.sh` | GWTC-4 / smoke figs |
| `09_emulator_m1_distribution.sh` | Emulator *m*₁ comparison |
| `05_ensemble_infer.sh` | Ensemble inference |
| `utils_fetch_gwtc40_events.sh` | Download GWTC-4 CSV |
| `smoke_test.sh` | Short GPU sanity check |

### Logs

`logs/<job_script_stem>.<jobid>.out` and `.err` from SLURM.

### External dependency

**`syntheticstellarpopconvolve`** (Hendriks et al.) — Madau–Dickinson SFR and COMPAS metallicity distributions for Step 00. Install per project setup (often from parent repo / `REPO_ROOT` in `plant_paths.py`).

---

## Notes

**In plain English:** *Why do we care about CPU vs GPU?* The **math-heavy** training (Steps 04–05 *full*) is much faster on a **GPU** (a card good at huge parallel matrix work). **Steps 00–03** are usually run on ordinary **CPU** nodes. *What is a checkpoint?* A **saved snapshot** of the learned numbers inside the neural networks so you can **stop and resume**, or **share** results with a collaborator without them re-running training.

- **CPU vs GPU:** Steps 00–03 and **04c** are typically CPU. Steps 04/04b and **05** (full) are best on **GPU** (05 repeatedly calls the frozen emulator’s sampler). **Order:** finish **one** of 04/04b **before** 05 for the main path; **04c** is optional and can run anytime after 02.
- **Lustre constraint:** On Expanse, add `#SBATCH --constraint="lustre"` to any script if you place data on `/expanse/lustre/scratch` (Lustre is a **shared file system** tuned for large parallel reads—only relevant if your site uses it).
- **SLURM account:** Scripts use **`#SBATCH --account=sdp153`**, matching the **PROJECT** field from `expanse-client user` (not the TG id **TG-PHY260100** by itself). If your allocation row shows a different PROJECT name, substitute that in every `slurm/*.sh`.
- **Checkpoint loading (technical):** e.g. `torch.load("checkpoints/cfm_final.pt", weights_only=False)`; posterior weights are in `checkpoints/posterior_network_best.pt` under the key `state_dict`, with Λ normalisation and column order in `posterior_network_config.json`.
