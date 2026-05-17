#!/bin/bash
#SBATCH --job-name=synth_real_07b
#SBATCH --output=logs/07b_synth_real.%j.out
#SBATCH --error=logs/07b_synth_real.%j.err
#SBATCH --partition=gpu-shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --account=sdp153
#SBATCH --export=ALL
#SBATCH --constraint=lustre

# GPU job (ACCESS Expanse): https://www.sdsc.edu/systems/expanse/user_guide.html
# - Do not mix `module load cpu` with `module load gpu`.
# - gpu-shared: 1 GPU; request enough CPUs/mem for dataloader + PyTorch host work.

# Compare posterior marginals: **emulator synthetic catalog** vs real GW catalog (`07b_synthetic_real_validation.py`).
#
# Required: REAL_EVENTS_CSV (e.g. data/gwtc40_o4a_confident_default_pe.csv from fetch_gwtc40_gwosc_csv.py).
#
# Usage:
#   export REAL_EVENTS_CSV=data/gwtc40_o4a_confident_default_pe.csv
#   sbatch slurm/07b_synth_real_validation.sh
#
# Synthetic arm (default): CFM at TNG-centered SMT key /SMT/sfra0157/mu00243 (override with
# SYNTH_HYPERPARAM_KEY or an explicit SYNTH_GRID_IDX).
# Optional:
#   export SYNTH_HYPERPARAM_KEY=/SMT/sfra0157/mu00243
#   export SYNTH_GRID_IDX=<int>  # if set, overrides SYNTH_HYPERPARAM_KEY lookup
#   export N_SYNTHETIC_EVENTS=256
#   export SYNTHETIC_SEED=0
#   export HYPERPARAM_ENCODED_CSV=data/hyperparam_table_encoded.csv
#   export SYNTH_CSV=path/to/table.csv   # if set, overrides emulator path (--synthetic-csv)
#   export DEVICE=cuda          # default; use cpu to force CPU on a GPU node
#   export EXTRA_07B='--num-samples 4000'

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

[[ -r /etc/slurm/slurm.conf ]] && export SLURM_CONF="${SLURM_CONF:-/etc/slurm/slurm.conf}"
[[ -f /etc/profile.d/modules.sh ]] && source /etc/profile.d/modules.sh

module purge
module load gpu

if [[ -f ".venv311/bin/activate" ]]; then
  source ".venv311/bin/activate"
elif [[ -f ".venv/bin/activate" ]]; then
  source ".venv/bin/activate"
else
  CONDA_ROOT="${CONDA_ROOT:-$HOME/miniconda3}"
  source "${CONDA_ROOT}/etc/profile.d/conda.sh"
  conda activate plant
fi

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

REAL="${REAL_EVENTS_CSV:?Set REAL_EVENTS_CSV to your GW merger catalog CSV}"
DEVICE="${DEVICE:-cuda}"

SYNTH_ARGS=()
if [[ -n "${SYNTH_CSV:-}" ]]; then
  SYNTH_ARGS+=( --synthetic-csv "${SYNTH_CSV}" )
else
  SYNTH_ARGS+=(
    --synthetic-hyperparam-key "${SYNTH_HYPERPARAM_KEY:-/SMT/sfra0157/mu00243}"
    --n-synthetic-events "${N_SYNTHETIC_EVENTS:-256}"
    --synthetic-seed "${SYNTHETIC_SEED:-0}"
  )
  if [[ -n "${SYNTH_GRID_IDX:-}" ]]; then
    SYNTH_ARGS+=( --synthetic-grid-idx "${SYNTH_GRID_IDX}" )
  fi
  if [[ -n "${HYPERPARAM_ENCODED_CSV:-}" ]]; then
    SYNTH_ARGS+=( --hyperparam-encoded-csv "${HYPERPARAM_ENCODED_CSV}" )
  fi
fi

EXTRA=()
if [[ -n "${EXTRA_07B:-}" ]]; then
  read -r -a EXTRA <<< "${EXTRA_07B}"
fi

python -u scripts/07b_synthetic_real_validation.py \
  "${SYNTH_ARGS[@]}" \
  --real-events-csv "${REAL}" \
  --checkpoint-dir checkpoints \
  --model full \
  --emulator cfm \
  --emulator-checkpoint checkpoints/cfm_final.pt \
  --num-samples 2000 \
  --device "${DEVICE}" \
  "${EXTRA[@]}"
