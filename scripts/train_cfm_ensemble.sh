#!/usr/bin/env bash
# Train K independent CFM emulators (different --seed, distinct --output-checkpoint).
# Usage:
#   bash scripts/train_cfm_ensemble.sh 3          # K=3
#   K=5 bash scripts/train_cfm_ensemble.sh        # default K=3 if no arg
set -euo pipefail
cd "$(dirname "$0")/.."

CONDA_ROOT="${CONDA_ROOT:-$HOME/miniconda3}"
CONDA_ENV="${CONDA_ENV:-plant}"
if [[ -f "${CONDA_ROOT}/etc/profile.d/conda.sh" ]]; then
  # shellcheck source=/dev/null
  source "${CONDA_ROOT}/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV}"
fi

K="${1:-3}"
for i in $(seq 1 "$K"); do
  SEED=$((1000 + i * 17))
  OUT="checkpoints/ensemble_cfm/${i}/cfm_final.pt"
  echo "=== member ${i}/${K}  seed=${SEED}  -> ${OUT} ==="
  python 04_cfm_emulator.py \
    --output-checkpoint "$OUT" \
    --seed "$SEED" \
    --device "${DEVICE:-cpu}" \
    ${EXTRA_ARGS:-}
done
