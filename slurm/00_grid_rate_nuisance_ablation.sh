#!/bin/bash
#SBATCH --job-name=grid_ablation
#SBATCH --output=logs/00_grid_rate_nuisance_ablation.%j.out
#SBATCH --error=logs/00_grid_rate_nuisance_ablation.%j.err
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --account=sdp153
#SBATCH --export=ALL

# Nuisance ablation heatmaps: (sfr_a, mu0) with one nuisance swept over Step 00 range.
# Recomputes cosmic integration (same numerics as 00_sspc_data_generation.py).
# Single nuisance only. For all seven nuisances use the array script (see below).
# Default grid: 20×20.
#
# Usage (from PLANT_GW_Paleontology/):
#   NUISANCE=sfr_b sbatch slurm/00_grid_rate_nuisance_ablation.sh
#
# For all seven nuisances in parallel (recommended):
#   sbatch slurm/00_grid_rate_nuisance_ablation_array.sh

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

[[ -r /etc/slurm/slurm.conf ]] && export SLURM_CONF="${SLURM_CONF:-/etc/slurm/slurm.conf}"
module purge
module load cpu

CONDA_ROOT="${CONDA_ROOT:-$HOME/miniconda3}"
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate plant

NUISANCE="${NUISANCE:-}"
EXTRA="${EXTRA:-}"
WORKERS="${WORKERS:-${SLURM_CPUS_PER_TASK:-16}}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

if [[ -z "${NUISANCE}" ]]; then
  echo "ERROR: This script runs one nuisance at a time. Set NUISANCE=sfr_b (etc.)," >&2
  echo "  or submit all seven in parallel:" >&2
  echo "    sbatch slurm/00_grid_rate_nuisance_ablation_array.sh" >&2
  exit 1
fi

python scripts/analysis/00_grid_rate_nuisance_ablation.py \
    --n-sfra 20 \
    --n-mu0 20 \
    --workers "${WORKERS}" \
    --nuisance "${NUISANCE}" \
    --no-tex \
    ${EXTRA}
