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

# Step 00 grid diagnostics: heatmaps of intrinsic rate density [Gpc^-3 yr^-1] vs (sfr_a, mu0).
# Requires data/sspc/models_sspc.hdf5 from Step 00 (00_sspc_data_generation.py).
#
# Usage (from PLANT_GW_Paleontology/):
#   sbatch slurm/00_grid_rate_heatmaps.sh
#   sbatch --export=ALL,GRID_METRIC=count slurm/00_grid_rate_heatmaps.sh
#   EXTRA='--color-scale linear' sbatch slurm/00_grid_rate_heatmaps.sh
#   EXTRA='--colormap diverging' sbatch slurm/00_grid_rate_heatmaps.sh
#   EXTRA='--average-over mu0' sbatch slurm/00_grid_rate_heatmaps.sh
#   EXTRA='--average-over sfra' sbatch slurm/00_grid_rate_heatmaps.sh
#   EXTRA='--linear-scale' sbatch slurm/00_grid_rate_heatmaps.sh   # alias for linear
#   SSPC_HDF5=data/sspc/models_sspc_fixed_nuisance.hdf5 EXTRA='--color-scale linear' sbatch slurm/00_grid_rate_heatmaps.sh
#   SSPC_HDF5=data/sspc/models_sspc_20x20_z02_fixed_nuisance.hdf5 EXTRA='--mark-fiducial-study' sbatch slurm/00_grid_rate_heatmaps.sh

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
SSPC_HDF5="${SSPC_HDF5:-}"

HDF5_ARG=()
if [[ -n "${SSPC_HDF5}" ]]; then
  HDF5_ARG=(--sspc-hdf5 "${SSPC_HDF5}")
fi

python scripts/analysis/00_grid_rate_heatmaps.py \
    --metric "${GRID_METRIC}" \
    "${HDF5_ARG[@]}" \
    ${EXTRA}
