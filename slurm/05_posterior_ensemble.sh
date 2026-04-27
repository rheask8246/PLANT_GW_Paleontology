#!/bin/bash
#SBATCH --job-name=post_ens
#SBATCH --output=logs/05_posterior_ensemble.%A_%a.out
#SBATCH --error=logs/05_posterior_ensemble.%A_%a.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --account=sdp153
#SBATCH --array=1-3
#SBATCH --export=ALL

# Posterior ensemble member i: pair with checkpoints/ensemble_cfm/i/cfm_final.pt
# Submit after 04_cfm_ensemble.sh (or equivalent paths on $SCRATCH).

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs


# Use cluster slurm.conf when present (avoids configless DNS SRV failures on some nodes).
[[ -r /etc/slurm/slurm.conf ]] && export SLURM_CONF="${SLURM_CONF:-/etc/slurm/slurm.conf}"

CONDA_ROOT="${CONDA_ROOT:-$HOME/miniconda3}"
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate plant

module purge
module load gpu

i="${SLURM_ARRAY_TASK_ID}"
ECHK="checkpoints/ensemble_cfm/${i}/cfm_final.pt"
POUT="checkpoints/posterior_ensemble/${i}/posterior_network_best.pt"

python 05_posterior_network.py \
    --emulator cfm \
    --emulator-checkpoint "$ECHK" \
    --output-checkpoint-pt "$POUT" \
    --model full \
    --epochs 200 \
    --patience 30 \
    --batch-size 8 \
    --n-max-events 256 \
    --lr 1e-4 \
    --seed $((2000 + i)) \
    --amp \
    --device cuda
