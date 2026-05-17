#!/bin/bash
#SBATCH --job-name=post_ens_inf
#SBATCH --output=logs/05_ensemble_infer.%j.out
#SBATCH --error=logs/05_ensemble_infer.%j.err
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --account=sdp153
#SBATCH --export=ALL

# Combine K trained posteriors (log-mean of log p and/or mixture sampling).
# Set MEMBER_DIRS and MODE before submit, or edit defaults below.
#
#   export MEMBER_DIRS="checkpoints/posterior_ensemble/1 checkpoints/posterior_ensemble/2"
#   export MODE=both
#   sbatch slurm/05_ensemble_infer.sh
#
# Quick synthetic-bag smoke (no CSV):
#   export SYNTHETIC_BAG=1
#   sbatch slurm/05_ensemble_infer.sh

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

[[ -r /etc/slurm/slurm.conf ]] && export SLURM_CONF="${SLURM_CONF:-/etc/slurm/slurm.conf}"
module purge
module load cpu

CONDA_ROOT="${CONDA_ROOT:-$HOME/miniconda3}"
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate plant

MODE="${MODE:-both}"
DEVICE="${DEVICE:-cpu}"
EXTRA="${EXTRA:-}"

if [[ "${SYNTHETIC_BAG:-0}" == "1" ]]; then
  python scripts/analysis/05_ensemble_infer.py \
    --synthetic-bag \
    --member-dirs ${MEMBER_DIRS:-checkpoints/posterior_ensemble/1 checkpoints/posterior_ensemble/2} \
    --mode "${MODE}" \
    --device "${DEVICE}" \
    ${EXTRA}
else
  : "${EVENTS_CSV:?Set EVENTS_CSV to your events table}"
  python scripts/analysis/05_ensemble_infer.py \
    --events-csv "${EVENTS_CSV}" \
    --member-dirs ${MEMBER_DIRS:-checkpoints/posterior_ensemble/1 checkpoints/posterior_ensemble/2} \
    --mode "${MODE}" \
    --device "${DEVICE}" \
    ${EXTRA}
fi
