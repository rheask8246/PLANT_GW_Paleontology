#!/bin/bash
#SBATCH --job-name=04d_nb_tau
#SBATCH --output=logs/04d_nb_kernel_bandwidth_sweep.%j.out
#SBATCH --error=logs/04d_nb_kernel_bandwidth_sweep.%j.err
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --account=sdp153
#SBATCH --export=ALL

# NB Λ-kernel τ only (legacy). Prefer full ablation:
#   sbatch slurm/04d_nb_ablation.sh
#
# This script — τ sweep only:
#   cd PLANT_GW_Paleontology && sbatch slurm/04d_nb_kernel_bandwidth_sweep.sh

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

bash slurm/04d_split_observable_model_compare.sh \
  --plot-set nb-kernel-bandwidth \
  --run-tag tau_sweep \
  --nb-kernel-bandwidths 0.05,0.1,0.15,0.2,0.3,0.4,0.5,0.75,1.0 \
  --splits test \
  --max-grids-per-split 200 \
  --events-per-grid 128
