#!/bin/bash
#SBATCH --job-name=fetch_gwtc40
#SBATCH --output=logs/fetch_gwtc40.%j.out
#SBATCH --error=logs/fetch_gwtc40.%j.err
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --account=sdp153
#SBATCH --export=ALL

# Download GWTC-4.0 default PE parameters from GWOSC (needs outbound network on compute node).
#
#   export GWTC40_OUT=data/gwtc40_o4a_confident_default_pe.csv
#   sbatch slurm/utils_fetch_gwtc40_events.sh

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs data

[[ -r /etc/slurm/slurm.conf ]] && export SLURM_CONF="${SLURM_CONF:-/etc/slurm/slurm.conf}"
module purge
module load cpu

CONDA_ROOT="${CONDA_ROOT:-$HOME/miniconda3}"
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate plant

OUT="${GWTC40_OUT:-data/gwtc40_o4a_confident_default_pe.csv}"

python scripts/analysis/utils/fetch_gwtc40_events.py -o "${OUT}"
