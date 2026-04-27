#!/bin/bash
#SBATCH --job-name=smoke_test
#SBATCH --output=logs/smoke_test.%j.out
#SBATCH --error=logs/smoke_test.%j.err
#SBATCH --partition=gpu-debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --account=PHY260100
#SBATCH --export=ALL

# Usage: sbatch slurm/smoke_test.sh
# Quick sanity check for both emulators on a GPU debug node (30 min limit).
# Use this to validate the environment before launching full training.

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

CONDA_ROOT="${CONDA_ROOT:-$HOME/miniconda3}"
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate plant

module purge
module load gpu

echo "=== CFM smoke test ==="
python 04_cfm_emulator.py --smoke-test --steps 500 --device cuda

echo "=== Diffusion smoke test ==="
python 04b_diffusion_emulator.py --smoke-test --steps 500 --device cuda
