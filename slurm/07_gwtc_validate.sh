#!/bin/bash
#SBATCH --job-name=gwtc_val
#SBATCH --output=logs/07_gwtc_validate.%j.out
#SBATCH --error=logs/07_gwtc_validate.%j.err
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --account=sdp153
#SBATCH --export=ALL

# Set EVENTS_CSV to your GWTC export before sbatch, e.g.:
#   export EVENTS_CSV=/path/to/gwtc_events.csv
#   sbatch slurm/07_gwtc_validate.sh

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs


# Use cluster slurm.conf when present (avoids configless DNS SRV failures on some nodes).
[[ -r /etc/slurm/slurm.conf ]] && export SLURM_CONF="${SLURM_CONF:-/etc/slurm/slurm.conf}"
module purge
module load cpu
CONDA_ROOT="${CONDA_ROOT:-$HOME/miniconda3}"
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate plant

CSV="${EVENTS_CSV:-data/gwtc_sample_events.csv}"
if [[ ! -f "$CSV" ]]; then
  echo "Set EVENTS_CSV to a real catalog, or create a small mock CSV at $CSV" >&2
  exit 1
fi

python 07_gwtc_posterior_validate.py \
    --events-csv "$CSV" \
    --checkpoint-dir checkpoints \
    --model full \
    --emulator cfm \
    --emulator-checkpoint checkpoints/cfm_final.pt \
    --num-samples 4000 \
    --device cpu
