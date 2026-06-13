#!/bin/bash
#SBATCH --job-name=sspc_data_gen
#SBATCH --output=logs/00_data_gen.%j.out
#SBATCH --error=logs/00_data_gen.%j.err
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --account=sdp153
#SBATCH --export=ALL

# Usage (from PLANT_GW_Paleontology/):  sbatch slurm/00_data_gen.sh
# Generates the full SSPC event HDF5 (50x50 grid per channel, 50k events each).
# ~3h wall-time; ~48 CPU-hours.
#
# TNG100-fixed nuisances + custom output:
#   SSPC_EXTRA='--fixed-nuisance-tng100 --output-hdf5 data/sspc/models_sspc_fixed_nuisance.hdf5' sbatch slurm/00_data_gen.sh
#
# 20×20 grid, fixed nuisances, all events at z=0.2:
#   SSPC_EXTRA='--n-sfra 20 --n-mu0 20 --fixed-nuisance-tng100 --fixed-z 0.2 --output-hdf5 data/sspc/models_sspc_20x20_z02_fixed_nuisance.hdf5' sbatch slurm/00_data_gen.sh
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

# Explicitly use all allocated CPU workers for threaded numeric kernels.
THREADS="${SLURM_CPUS_PER_TASK:-16}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$THREADS}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$THREADS}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-$THREADS}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-$THREADS}"
echo "Using CPU worker threads: ${THREADS}"

# Optional: --bps-hdf5 /path/to/bps_output.h5  (default: data/bps_output.h5)

if [[ -n "${SSPC_EXTRA:-}" ]]; then
  # shellcheck disable=SC2206
  EXTRA_ARGS=(${SSPC_EXTRA})
else
  EXTRA_ARGS=(--output-hdf5 data/sspc/models_sspc.hdf5)
fi

$PYTHON scripts/00_sspc_data_generation.py \
    --n-events 50000 \
    --workers "${THREADS}" \
    "${EXTRA_ARGS[@]}"
