#!/bin/bash
#SBATCH --job-name=pop_fig
#SBATCH --output=logs/06_population_figures.%j.out
#SBATCH --error=logs/06_population_figures.%j.err
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --account=sdp153
#SBATCH --export=ALL

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

python 06_population_figures.py \
    --sspc-hdf5 data/sspc/models_sspc.hdf5 \
    --z-slices 0.2 1.0
