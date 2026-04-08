#!/bin/bash
#SBATCH --job-name=rate_network
#SBATCH --output=logs/03_rate_network.%j.out
#SBATCH --error=logs/03_rate_network.%j.err
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --account=<<PROJECT>>
#SBATCH --export=ALL

# Usage: sbatch slurm/03_rate_network.sh
# Trains the rate network (MLP regressing log10(sum_weight) from hyperparams).
# CPU-only; ~20–40 min wall-time.

module purge
module load cpu
source .venv/bin/activate

python 03_rate_network.py \
    --epochs 2000 \
    --patience 200 \
    --device cpu
