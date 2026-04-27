#!/bin/bash
#SBATCH --job-name=data_val
#SBATCH --output=logs/02b_data_validation.%j.out
#SBATCH --error=logs/02b_data_validation.%j.err
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --account=sdp153
#SBATCH --export=ALL

# Full-parquet intrinsic validation (CPU). Timestamped under test/reports/validation/ and test/plots/validation/ by default.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs


# Use cluster slurm.conf when present (avoids configless DNS SRV failures on some nodes).
[[ -r /etc/slurm/slurm.conf ]] && export SLURM_CONF="${SLURM_CONF:-/etc/slurm/slurm.conf}"
module purge
module load cpu
CONDA_ROOT="${CONDA_ROOT:-$HOME/miniconda3}"
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate plant

python test/validation/run_data_validation.py
