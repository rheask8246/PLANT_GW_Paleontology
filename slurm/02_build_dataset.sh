#!/bin/bash
#SBATCH --job-name=build_dataset
#SBATCH --output=logs/02_build_dataset.%j.out
#SBATCH --error=logs/02_build_dataset.%j.err
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --account=<<PROJECT>>
#SBATCH --export=ALL

# Usage: sbatch slurm/02_build_dataset.sh
# Reads the SSPC HDF5, samples events, writes parquet + normalizer + splits.
# Run after 00_data_gen.sh completes.  ~10 min wall-time.

module purge
module load cpu
source .venv/bin/activate

python 02_build_dataset.py \
    --hdf5 data/sspc/models_sspc.hdf5 \
    --data-source sspc
