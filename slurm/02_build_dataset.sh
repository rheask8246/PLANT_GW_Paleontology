#!/bin/bash
#SBATCH --job-name=build_dataset
#SBATCH --output=logs/02_build_dataset.%j.out
#SBATCH --error=logs/02_build_dataset.%j.err
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --account=sdp153
#SBATCH --export=ALL

# Usage: sbatch slurm/02_build_dataset.sh
# Reads the SSPC HDF5, samples events, writes parquet + normalizer + splits.
# Run after 00_data_gen.sh completes.
# Parallelizes over grid keys (--workers = SLURM_CPUS_PER_TASK, default 16).
# For 50×50×3 grid (~7.5k keys): often ~1–3 h with 16 workers (I/O bound).
#
# Alternate Step-00 HDF5 + isolated dataset directory (does not overwrite data/):
#   HDF5=data/sspc/models_sspc_20x20_z02_fixed_nuisance.hdf5 \
#     OUT_DIR=data/ml_20x20_z02 sbatch slurm/02_build_dataset.sh

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

WORKERS="${WORKERS:-${SLURM_CPUS_PER_TASK:-16}}"
# Parent stays single-threaded; each worker process sets OMP_NUM_THREADS=1.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
echo "Using parallel worker processes: ${WORKERS}"

HDF5="${HDF5:-data/sspc/models_sspc.hdf5}"
OUT_DIR="${OUT_DIR:-data}"

$PYTHON scripts/02_build_dataset.py \
    --hdf5 "${HDF5}" \
    --data-source sspc \
    --out-dir "${OUT_DIR}" \
    --workers "${WORKERS}"
