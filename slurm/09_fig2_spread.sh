#!/bin/bash
#SBATCH --job-name=fig2_spread
#SBATCH --output=logs/09_fig2_spread.%j.out
#SBATCH --error=logs/09_fig2_spread.%j.err
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --account=sdp153
#SBATCH --export=ALL
#SBATCH --constraint=lustre

# Figure-2-style SSPC marginals (see fig2_spread.py).
# Usage from repo root:
#   sbatch slurm/09_fig2_spread.sh
#
# Optional env overrides (examples):
#   sbatch --export=ALL,FIG2_ZTARGET=0.15,FIG2_NPAIRS=16,FIG2_SMOOTH=1.8,FIG2_EXTRA='--auto-y' slurm/09_fig2_spread.sh

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

[[ -r /etc/slurm/slurm.conf ]] && export SLURM_CONF="${SLURM_CONF:-/etc/slurm/slurm.conf}"
[[ -f /etc/profile.d/modules.sh ]] && source /etc/profile.d/modules.sh

module purge
module load cpu

CONDA_ROOT="${CONDA_ROOT:-$HOME/miniconda3}"
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate plant

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

NPAIRS="${FIG2_NPAIRS:-9}"
SEED="${FIG2_SEED:-0}"
OUT="${FIG2_OUT:-plots/fig2_sspc_spread.pdf}"
SMOOTH="${FIG2_SMOOTH:-}"

EXTRA=()
if [[ -n "${FIG2_EXTRA:-}" ]]; then
  read -r -a EXTRA <<< "${FIG2_EXTRA}"
fi

SMOOTH_FLAG=()
if [[ -n "${SMOOTH}" ]]; then
  SMOOTH_FLAG=(--smooth-sigma "${SMOOTH}")
fi

python -u scripts/fig2_spread.py \
  --z-target "${FIG2_ZTARGET:-0.1}" \
  --n-pairs "${NPAIRS}" \
  --seed "${SEED}" \
  --out "${OUT}" \
  --no-tex \
  "${SMOOTH_FLAG[@]}" \
  "${EXTRA[@]}"
