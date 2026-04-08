#!/bin/bash
#SBATCH --job-name=sspc_data_gen
#SBATCH --output=logs/00_data_gen.%j.out
#SBATCH --error=logs/00_data_gen.%j.err
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --account=<<PROJECT>>
#SBATCH --export=ALL

# Usage: sbatch slurm/00_data_gen.sh
# Generates the full SSPC event HDF5 (50x50 grid per channel, 50k events each).
# ~3h wall-time; ~48 CPU-hours.

module purge
module load cpu
source .venv/bin/activate

mkdir -p logs data/sspc

python 00_sspc_data_generation.py \
    --n-sfra 50 \
    --n-mu0  50 \
    --n-events 50000 \
    --output-hdf5 data/sspc/models_sspc.hdf5 \
    --overwrite
