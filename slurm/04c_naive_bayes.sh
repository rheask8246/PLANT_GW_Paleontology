#!/bin/bash
#SBATCH --job-name=nb_emulator
#SBATCH --output=logs/04c_naive_bayes.%j.out
#SBATCH --error=logs/04c_naive_bayes.%j.err
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --account=sdp153
#SBATCH --no-requeue
#SBATCH --export=ALL

# Usage: sbatch slurm/04c_naive_bayes.sh
# Fit Naive Bayes baseline from Step 02 artifacts (CPU, ~minutes).
# Output: checkpoints/naive_bayes_final.pt
# Plots (separate): sbatch slurm/04c_naive_bayes_emulator_plots.sh

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

[[ -r /etc/slurm/slurm.conf ]] && export SLURM_CONF="${SLURM_CONF:-/etc/slurm/slurm.conf}"

module purge
module load cpu

PYTHON="${PYTHON:-${PWD}/.venv311/bin/python}"
[[ -x "$PYTHON" ]] || PYTHON="python"

THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$THREADS}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$THREADS}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-$THREADS}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-$THREADS}"
echo "Using CPU worker threads: ${THREADS}"

$PYTHON scripts/04c_naive_bayes_emulator.py --workers "${THREADS}" "$@"
