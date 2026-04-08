#!/bin/bash
#SBATCH --job-name=diffusion_train
#SBATCH --output=logs/04b_diffusion.%j.out
#SBATCH --error=logs/04b_diffusion.%j.err
#SBATCH --partition=gpu-shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=10
#SBATCH --gpus=1
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --account=<<PROJECT>>
#SBATCH --no-requeue
#SBATCH --export=ALL

# Usage: sbatch slurm/04b_diffusion.sh
# Full Diffusion training: 100k steps, hidden_dim=256, batch=256 on 1x V100 GPU.
# ~12–20 h wall-time depending on grid size.
# Output: checkpoints/diffusion_final.pt + plots/diffusion_smoke_test/<date>/

module purge
module load gpu
source .venv/bin/activate

python 04b_diffusion_emulator.py \
    --steps 100000 \
    --device cuda
