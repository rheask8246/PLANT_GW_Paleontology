#!/bin/bash
#SBATCH --job-name=rate_network
#SBATCH --output=logs/03_rate_network.%j.out
#SBATCH --error=logs/03_rate_network.%j.err
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --account=sdp153
#SBATCH --export=ALL

# Usage: sbatch slurm/03_rate_network.sh
# Trains the rate network (MLP regressing log10(sum_weight) from hyperparams).
# CPU-only; ~20–40 min wall-time.

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs


# Use cluster slurm.conf when present (avoids configless DNS SRV failures on some nodes).
[[ -r /etc/slurm/slurm.conf ]] && export SLURM_CONF="${SLURM_CONF:-/etc/slurm/slurm.conf}"

CONDA_ROOT="${CONDA_ROOT:-$HOME/miniconda3}"
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate plant

module purge
module load cpu

python scripts/03_rate_network.py \
    --epochs 2000 \
    --patience 200 \
    --device cpu
