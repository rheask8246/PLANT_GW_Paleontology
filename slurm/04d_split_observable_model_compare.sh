#!/bin/bash
#SBATCH --job-name=04d_split_cmp
#SBATCH --output=logs/04d_split_observable_model_compare.%j.out
#SBATCH --error=logs/04d_split_observable_model_compare.%j.err
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --account=sdp153
#SBATCH --export=ALL

# Compare NB/CFM/diffusion on mchirp,q,z across train/val/test.
#
# Full comparison (default):
#   sbatch slurm/04d_split_observable_model_compare.sh
#
# NB Λ-kernel τ ablation (easiest — no extra sbatch args):
#   sbatch slurm/04d_nb_kernel_bandwidth_sweep.sh
#
# Or one line (backslash must be immediately before newline, no spaces after \):
#   sbatch slurm/04d_split_observable_model_compare.sh --plot-set nb-kernel-bandwidth --run-tag tau_sweep --nb-kernel-bandwidths 0.05,0.1,0.15,0.2,0.3,0.4,0.5,0.75,1.0 --splits test --max-grids-per-split 200 --events-per-grid 128

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
export PYTHONUNBUFFERED=1

# sbatch may pass a leading "--" (bash end-of-options marker); strip it for Python.
if [[ $# -gt 0 && "$1" == "--" ]]; then
  shift
fi

# Drop empty / whitespace-only tokens (from "\  " after backslashes on the sbatch line).
USER_ARGS=()
for arg in "$@"; do
  trimmed="${arg#"${arg%%[![:space:]]*}"}"
  trimmed="${trimmed%"${trimmed##*[![:space:]]}"}"
  if [[ -n "${trimmed}" ]]; then
    USER_ARGS+=("${trimmed}")
  fi
done

echo "Using CPU worker threads: ${THREADS}"
if [[ ${#USER_ARGS[@]} -eq 0 ]]; then
  echo "Extra args: (none — running full plot set)"
else
  echo "Extra args: ${USER_ARGS[*]}"
fi

$PYTHON scripts/analysis/04d_split_observable_model_compare.py \
  --workers "${THREADS}" "${USER_ARGS[@]}"
