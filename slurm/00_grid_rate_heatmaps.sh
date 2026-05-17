#!/bin/bash
#SBATCH --job-name=grid_rate_hm
#SBATCH --output=logs/00_grid_rate_heatmaps.%j.out
#SBATCH --error=logs/00_grid_rate_heatmaps.%j.err
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --account=sdp153
#SBATCH --export=ALL

# Step 00 grid diagnostics: heatmaps of sum_weight (rate) or n_systems (count) vs (sfr_a, mu0).
# Requires data/hyperparam_table.csv from Step 02 (or pass --sspc-hdf5 in EXTRA below).
#
# Usage (from PLANT_GW_Paleontology/):
#   sbatch slurm/00_grid_rate_heatmaps.sh
#   sbatch --export=ALL,GRID_METRIC=count slurm/00_grid_rate_heatmaps.sh
#   EXTRA='--log-scale' sbatch slurm/00_grid_rate_heatmaps.sh

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

[[ -r /etc/slurm/slurm.conf ]] && export SLURM_CONF="${SLURM_CONF:-/etc/slurm/slurm.conf}"
module purge
module load cpu

CONDA_ROOT="${CONDA_ROOT:-$HOME/miniconda3}"
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate plant

GRID_METRIC="${GRID_METRIC:-rate}"
EXTRA="${EXTRA:-}"

python scripts/analysis/00_grid_rate_heatmaps.py \
    --metric "${GRID_METRIC}" \
    ${EXTRA}
