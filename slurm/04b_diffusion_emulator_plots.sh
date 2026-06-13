#!/bin/bash
#SBATCH --job-name=diff_plots
#SBATCH --output=logs/04b_diffusion_emulator_plots.%j.out
#SBATCH --error=logs/04b_diffusion_emulator_plots.%j.err
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --account=sdp153
#SBATCH --export=ALL

# Validation plots from checkpoints/diffusion_final.pt (no retraining).
# Usage: sbatch slurm/04b_diffusion_emulator_plots.sh

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

[[ -r /etc/slurm/slurm.conf ]] && export SLURM_CONF="${SLURM_CONF:-/etc/slurm/slurm.conf}"
module purge
module load cpu

PYTHON="${PYTHON:-${PWD}/.venv311/bin/python}"
[[ -x "$PYTHON" ]] || PYTHON="python"

$PYTHON scripts/analysis/04b_diffusion_emulator_plots.py "$@"
