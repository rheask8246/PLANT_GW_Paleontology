#!/bin/bash
#SBATCH --job-name=rate_vs_z
#SBATCH --output=logs/00_rate_vs_redshift.%j.out
#SBATCH --error=logs/00_rate_vs_redshift.%j.err
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --account=sdp153
#SBATCH --export=ALL

# Intrinsic R(z) with fixed midpoint nuisances (recomputes Step 00 cosmic integration).
#
# Usage (from PLANT_GW_Paleontology/):
#   VARY=sfra sbatch slurm/00_rate_vs_redshift.sh
#   VARY=mu0  sbatch slurm/00_rate_vs_redshift.sh
#   VARY=sfra EXTRA='--log-y --n-curves 7' sbatch slurm/00_rate_vs_redshift.sh

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

[[ -r /etc/slurm/slurm.conf ]] && export SLURM_CONF="${SLURM_CONF:-/etc/slurm/slurm.conf}"
module purge
module load cpu

CONDA_ROOT="${CONDA_ROOT:-$HOME/miniconda3}"
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate plant

VARY="${VARY:?Set VARY=sfra or VARY=mu0}"
EXTRA="${EXTRA:-}"

python scripts/analysis/00_rate_vs_redshift.py \
    --vary "${VARY}" \
    ${EXTRA}
