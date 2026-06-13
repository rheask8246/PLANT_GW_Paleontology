#!/bin/bash
#SBATCH --job-name=04_kl_hm
#SBATCH --output=logs/04_kl_divergence_heatmaps.%j.out
#SBATCH --error=logs/04_kl_divergence_heatmaps.%j.err
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --account=sdp153
#SBATCH --export=ALL

# KL divergence heatmaps (NB/CFM/Diffusion vs train/test splits).
# Usage: sbatch slurm/04_kl_divergence_heatmaps.sh

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

[[ -r /etc/slurm/slurm.conf ]] && export SLURM_CONF="${SLURM_CONF:-/etc/slurm/slurm.conf}"
module purge
module load cpu

PYTHON="${PYTHON:-${PWD}/.venv311/bin/python}"
[[ -x "$PYTHON" ]] || PYTHON="python"

THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$THREADS}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$THREADS}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-$THREADS}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-$THREADS}"
echo "Using CPU worker threads: ${THREADS}"

$PYTHON scripts/analysis/04_kl_divergence_heatmaps.py --workers "${THREADS}" "$@"
