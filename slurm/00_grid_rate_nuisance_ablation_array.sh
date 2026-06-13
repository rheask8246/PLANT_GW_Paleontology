#!/bin/bash
#SBATCH --job-name=grid_abl_arr
#SBATCH --output=logs/00_grid_rate_nuisance_ablation_array.%A_%a.out
#SBATCH --error=logs/00_grid_rate_nuisance_ablation_array.%A_%a.err
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --account=sdp153
#SBATCH --array=0-8
#SBATCH --export=ALL

# Nine heatmaps in parallel (20×20 grid, 5000 sampled events/cell like Step 02).
# Each array task uses 16 CPU worker processes (one integration thread each).
# Ablation panels fan out 8 nuisance slices per grid cell across workers.
#
# Usage (from PLANT_GW_Paleontology/):
#   sbatch slurm/00_grid_rate_nuisance_ablation_array.sh
#
# More CPUs (if your account allows):
#   WORKERS=32 sbatch --cpus-per-task=32 --mem=192G slurm/00_grid_rate_nuisance_ablation_array.sh
#
# Outputs (shared folder per submission):
#   plots/00_grid_rate_nuisance_ablation/array_<JOBID>/

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

[[ -r /etc/slurm/slurm.conf ]] && export SLURM_CONF="${SLURM_CONF:-/etc/slurm/slurm.conf}"
module purge
module load cpu

CONDA_ROOT="${CONDA_ROOT:-$HOME/miniconda3}"
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate plant

NUISANCES=(
  sfr_b sfr_c sfr_d muz sigma0 sigmaz alpha_skew
  all_sampled all_fixed
)
TASK_ID="${SLURM_ARRAY_TASK_ID:?Set #SBATCH --array=0-8}"
NUISANCE="${NUISANCES[$TASK_ID]}"

WORKERS="${WORKERS:-${SLURM_CPUS_PER_TASK:-16}}"
OUT_DIR="${OUT_DIR:-plots/00_grid_rate_nuisance_ablation/array_${SLURM_ARRAY_JOB_ID}}"
EXTRA="${EXTRA:-}"

# Parent process stays single-threaded; worker pool sets 1 thread per child.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "Array task ${TASK_ID}: nuisance=${NUISANCE} workers=${WORKERS} out=${OUT_DIR}"

python scripts/analysis/00_grid_rate_nuisance_ablation.py \
    --n-sfra 20 \
    --n-mu0 20 \
    --nuisance "${NUISANCE}" \
    --workers "${WORKERS}" \
    --out-dir "${OUT_DIR}" \
    --no-tex \
    ${EXTRA}
