#!/bin/bash
#SBATCH --job-name=04d_nb_abl
#SBATCH --output=logs/04d_nb_ablation.%j.out
#SBATCH --error=logs/04d_nb_ablation.%j.err
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --account=sdp153
#SBATCH --export=ALL

# One job → four collated density panels (truth vs NB):
#   nb_ablation_tau.png          — Λ-kernel bandwidth τ
#   nb_ablation_sigma_scale.png  — multiply per-grid grid_sigma at sample time
#   nb_ablation_sigma_floor.png  — refit with different σ_floor
#   nb_ablation_mode.png         — gaussian vs nearest refit
#
# Usage (from PLANT_GW_Paleontology/):
#   sbatch slurm/04d_nb_ablation.sh

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

bash slurm/04d_split_observable_model_compare.sh \
  --plot-set nb-ablation \
  --run-tag nb_ablation \
  --splits test \
  --max-grids-per-split 200 \
  --events-per-grid 128
