#!/bin/bash
#SBATCH --job-name=diffusion_train
#SBATCH --output=logs/04b_diffusion.%j.out
#SBATCH --error=logs/04b_diffusion.%j.err
#SBATCH --partition=gpu-shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --gpus=1
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --account=sdp153
#SBATCH --no-requeue
#SBATCH --export=ALL

# Usage: sbatch slurm/04b_diffusion.sh
# Full Diffusion training: 100k steps, hidden_dim=256, batch=256 on 1x V100 GPU.
# ~12–20 h wall-time depending on grid size.
# Output: checkpoints/diffusion_final.pt + plots/04b_diffusion_emulator/<timestamp>/

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs


# Use cluster slurm.conf when present (avoids configless DNS SRV failures on some nodes).
[[ -r /etc/slurm/slurm.conf ]] && export SLURM_CONF="${SLURM_CONF:-/etc/slurm/slurm.conf}"

module purge
module load gpu

PYTHON="${PYTHON:-${PWD}/.venv311/bin/python}"
[[ -x "$PYTHON" ]] || PYTHON="python"

THREADS="${SLURM_CPUS_PER_TASK:-10}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$THREADS}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$THREADS}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-$THREADS}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-$THREADS}"
echo "Using CPU worker threads: ${THREADS}"

$PYTHON scripts/04b_diffusion_emulator.py \
    --steps 100000 \
    --device cuda \
    --workers "${THREADS}"
