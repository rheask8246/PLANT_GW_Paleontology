#!/bin/bash
#SBATCH --job-name=nb_plots
#SBATCH --output=logs/04c_naive_bayes_emulator_plots.%j.out
#SBATCH --error=logs/04c_naive_bayes_emulator_plots.%j.err
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --account=sdp153
#SBATCH --export=ALL

# Plots from checkpoints/naive_bayes_final.pt (no refit).
# Usage: sbatch slurm/04c_naive_bayes_emulator_plots.sh

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

[[ -r /etc/slurm/slurm.conf ]] && export SLURM_CONF="${SLURM_CONF:-/etc/slurm/slurm.conf}"
module purge
module load cpu

PYTHON="${PYTHON:-${PWD}/.venv311/bin/python}"
[[ -x "$PYTHON" ]] || PYTHON="python"

$PYTHON scripts/analysis/04c_naive_bayes_emulator_plots.py "$@"
