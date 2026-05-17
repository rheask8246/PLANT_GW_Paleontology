#!/bin/bash
#SBATCH --job-name=sspc_data_gen
#SBATCH --output=logs/00_data_gen.%j.out
#SBATCH --error=logs/00_data_gen.%j.err
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --account=sdp153
#SBATCH --export=ALL

# Usage (from PLANT_GW_Paleontology/):  sbatch slurm/00_data_gen.sh
# Generates the full SSPC event HDF5 (50x50 grid per channel, 50k events each).
# ~3h wall-time; ~48 CPU-hours.
#
# Python: uses project .venv311 if present (install SSPC there). Override:  PYTHON=/path/to/python sbatch …

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs data/sspc


# Use cluster slurm.conf when present (avoids configless DNS SRV failures on some nodes).
[[ -r /etc/slurm/slurm.conf ]] && export SLURM_CONF="${SLURM_CONF:-/etc/slurm/slurm.conf}"

module purge
module load cpu

PYTHON="${PYTHON:-${PWD}/.venv311/bin/python}"
[[ -x "$PYTHON" ]] || PYTHON="python"

# Optional: --bps-hdf5 /path/to/bps_output.h5  (default: data/bps_output.h5)

$PYTHON scripts/00_sspc_data_generation.py \
    --n-sfra 50 \
    --n-mu0  50 \
    --n-events 50000 \
    --output-hdf5 data/sspc/models_sspc.hdf5 \
    --overwrite
