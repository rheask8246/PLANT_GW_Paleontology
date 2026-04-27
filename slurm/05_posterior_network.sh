#!/bin/bash
#SBATCH --job-name=posterior_net
#SBATCH --output=logs/05_posterior_network.%j.out
#SBATCH --error=logs/05_posterior_network.%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --account=PHY260100
#SBATCH --export=ALL

# Trains the posterior (Step 5) on **synthetic** catalogs from a **trained, frozen** emulator.
# **Submit only after** `04_cfm.sh` or `04b_diffusion.sh` has produced
#   checkpoints/cfm_final.pt  or  checkpoints/diffusion_final.pt
# (and 02 has been run). This is sequential to Stage 2 in the proposal, not parallel.
#
# ACCESS Expanse: prefer project scratch, load CUDA, activate venv.
# `--num-workers` must stay 0 in 05 (on-the-fly CFM/ODE in main process).
#
# For diffusion instead: add  --emulator diffusion --emulator-checkpoint checkpoints/diffusion_final.pt
#
# Usage: sbatch slurm/05_posterior_network.sh

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

mkdir -p logs

CONDA_ROOT="${CONDA_ROOT:-$HOME/miniconda3}"
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate plant

module purge
module load gpu

python 05_posterior_network.py \
    --emulator cfm \
    --emulator-checkpoint checkpoints/cfm_final.pt \
    --model full \
    --epochs 200 \
    --patience 30 \
    --batch-size 8 \
    --n-max-events 256 \
    --lr 1e-4 \
    --amp \
    --device cuda
