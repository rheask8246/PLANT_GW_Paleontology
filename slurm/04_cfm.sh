#!/bin/bash
#SBATCH --job-name=cfm_train
#SBATCH --output=logs/04_cfm.%j.out
#SBATCH --error=logs/04_cfm.%j.err
#SBATCH --partition=gpu-shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=10
#SBATCH --gpus=1
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --account=<<PROJECT>>
#SBATCH --no-requeue
#SBATCH --export=ALL

# Usage: sbatch slurm/04_cfm.sh
# Full CFM training: 100k steps, hidden_dim=256, batch=256 on 1x V100 GPU.
# ~12–20 h wall-time depending on grid size.
# Output: checkpoints/cfm_final.pt + plots/cfm_smoke_test/<date>/

module purge
module load gpu
source .venv/bin/activate

python 04_cfm_emulator.py \
    --steps 100000 \
    --device cuda
