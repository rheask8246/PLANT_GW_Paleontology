#!/bin/bash
#SBATCH --job-name=emu_m1_dist
#SBATCH --output=logs/09_emulator_m1_distribution.%j.out
#SBATCH --error=logs/09_emulator_m1_distribution.%j.err
#SBATCH --partition=gpu-shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=16G
#SBATCH --constraint=lustre
#SBATCH --time=00:30:00
#SBATCH --account=sdp153
#SBATCH --export=ALL
#SBATCH --no-requeue

# Fast GPU job: CFM vs diffusion primary-mass KDE at fixed SSPC Λ (three channel rows: all, SMT, CE)
# (`data_distribution_analysis.py --emulator-m1-compare`).
#
# From PLANT_GW_Paleontology/:
#   sbatch slurm/09_emulator_m1_distribution.sh
#
# Optional:
#   export EMU_N_EVENTS=8000
#   export EMU_DEVICE=cuda          # or cpu / auto
#   export EMU_EXTRA='--n-events 5000 --seed 1'

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

[[ -r /etc/slurm/slurm.conf ]] && export SLURM_CONF="${SLURM_CONF:-/etc/slurm/slurm.conf}"
[[ -f /etc/profile.d/modules.sh ]] && source /etc/profile.d/modules.sh

module purge
module load gpu

if [[ -f ".venv311/bin/activate" ]]; then
  source ".venv311/bin/activate"
elif [[ -f ".venv/bin/activate" ]]; then
  source ".venv/bin/activate"
else
  CONDA_ROOT="${CONDA_ROOT:-$HOME/miniconda3}"
  source "${CONDA_ROOT}/etc/profile.d/conda.sh"
  conda activate plant
fi

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

N_EVENTS="${EMU_N_EVENTS:-12000}"
DEVICE="${EMU_DEVICE:-cuda}"
EXTRA=()
if [[ -n "${EMU_EXTRA:-}" ]]; then
  read -r -a EXTRA <<< "${EMU_EXTRA}"
fi

echo "=== emulator m₁ compare: n_events=${N_EVENTS} device=${DEVICE} (3 rows: all, SMT, CE) ==="

python -u scripts/data_distribution_analysis.py \
  --emulator-m1-compare \
  --device "${DEVICE}" \
  --n-events "${N_EVENTS}" \
  "${EXTRA[@]}"

echo "=== done ==="
