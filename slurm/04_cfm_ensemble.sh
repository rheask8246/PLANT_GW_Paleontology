#!/bin/bash
#SBATCH --job-name=cfm_ens
#SBATCH --output=logs/04_cfm_ensemble.%A_%a.out
#SBATCH --error=logs/04_cfm_ensemble.%A_%a.err
#SBATCH --partition=gpu-shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=10
#SBATCH --gpus=1
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --account=PHY260100
#SBATCH --no-requeue
#SBATCH --array=1-3
#SBATCH --export=ALL

# One CFM ensemble member per array task: output checkpoints/ensemble_cfm/${SLURM_ARRAY_TASK_ID}/cfm_final.pt
# Usage: sbatch slurm/04_cfm_ensemble.sh   (edit #SBATCH --array=1-K for K members)

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

CONDA_ROOT="${CONDA_ROOT:-$HOME/miniconda3}"
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate plant

module purge
module load gpu

i="${SLURM_ARRAY_TASK_ID}"
SEED=$((1000 + i * 17))
OUT="checkpoints/ensemble_cfm/${i}/cfm_final.pt"

python 04_cfm_emulator.py \
    --output-checkpoint "$OUT" \
    --seed "$SEED" \
    --device cuda \
    --steps 100000
