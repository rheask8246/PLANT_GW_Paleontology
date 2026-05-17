#!/bin/bash
#SBATCH --job-name=build_dataset
#SBATCH --output=logs/02_build_dataset.%j.out
#SBATCH --error=logs/02_build_dataset.%j.err
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --account=sdp153
#SBATCH --export=ALL

# Usage: sbatch slurm/02_build_dataset.sh
# Reads the SSPC HDF5, samples events, writes parquet + normalizer + splits.
# Run after 00_data_gen.sh completes.
# For 00 at 50×50×3 grid (7.5k keys) + default --n-sample/--n-det: allow several hours (I/O heavy).

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs


# Use cluster slurm.conf when present (avoids configless DNS SRV failures on some nodes).
[[ -r /etc/slurm/slurm.conf ]] && export SLURM_CONF="${SLURM_CONF:-/etc/slurm/slurm.conf}"

module purge
module load cpu

PYTHON="${PYTHON:-${PWD}/.venv311/bin/python}"
[[ -x "$PYTHON" ]] || PYTHON="python"
export PYTHONUNBUFFERED=1

$PYTHON scripts/02_build_dataset.py \
    --hdf5 data/sspc/models_sspc.hdf5 \
    --data-source sspc
