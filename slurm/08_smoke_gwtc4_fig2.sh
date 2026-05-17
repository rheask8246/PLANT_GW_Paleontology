#!/bin/bash
#SBATCH --job-name=gwtc4_f2_smoke
#SBATCH --output=logs/08_smoke_gwtc4_fig2.%j.out
#SBATCH --error=logs/08_smoke_gwtc4_fig2.%j.err
#SBATCH --partition=gpu-shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=16G
#SBATCH --constraint=lustre
#SBATCH --time=00:45:00
#SBATCH --account=sdp153
#SBATCH --export=ALL
#SBATCH --no-requeue

# Fast **Figure 2 only** smoke test (low MC → wide / noisy bands; wiring + end-to-end check).
#
# Required:
#   export GWTC4_DATA_RELEASE=/path/to/data_release   # AllCBC_FullPop*.h5 + AllCBC_FullPopBGP*.h5
#
# From PLANT_GW_Paleontology/:
#   export GWTC4_DATA_RELEASE=/path/to/data_release
#   sbatch slurm/08_smoke_gwtc4_fig2.sh
#
# Override smoke defaults (optional):
#   export GWTC4_NBOOT=3 GWTC4_NROWS=48 GWTC4_NEVENTS=128
#   export GWTC4_OUT_DIR=plots/04_gwtc4_validation/my_smoke

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Smoke defaults (~25× fewer emulator catalogs than full 08 defaults for the Fig 2/3 block).
export GWTC4_FIGS="${GWTC4_FIGS:-2}"
export GWTC4_NBOOT="${GWTC4_NBOOT:-2}"
export GWTC4_NROWS="${GWTC4_NROWS:-32}"
export GWTC4_NEVENTS="${GWTC4_NEVENTS:-96}"
export GWTC4_OUT_DIR="${GWTC4_OUT_DIR:-plots/04_gwtc4_validation/smoke_fig2_${SLURM_JOB_ID:-local}}"
export GWTC4_DEVICE="${GWTC4_DEVICE:-cuda}"

if [[ -z "${GWTC4_DATA_RELEASE:-}" ]]; then
  echo "ERROR: export GWTC4_DATA_RELEASE to your GWTC-4.0 Zenodo data_release/ (paper curves for Fig 2)." >&2
  exit 1
fi

echo "=== Smoke Fig 2: figs=${GWTC4_FIGS} n_boot=${GWTC4_NBOOT} n_rows=${GWTC4_NROWS} n_events=${GWTC4_NEVENTS} out=${GWTC4_OUT_DIR} ==="

exec bash "${SCRIPT_DIR}/08_gwtc4_validation.sh"
