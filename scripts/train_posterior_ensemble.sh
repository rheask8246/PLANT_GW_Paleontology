#!/usr/bin/env bash
# Train K posteriors, each with its own --emulator-checkpoint and --output-checkpoint-pt.
# Example: after scripts/train_cfm_ensemble.sh, point EMU to member i.
set -euo pipefail
cd "$(dirname "$0")/.."
K="${1:-3}"
for i in $(seq 1 "$K"); do
  ECHK="checkpoints/ensemble_cfm/${i}/cfm_final.pt"
  POUT="checkpoints/posterior_ensemble/${i}/posterior_network_best.pt"
  echo "=== posterior ${i}/${K}  emulator=${ECHK}  -> ${POUT} ==="
  python 05_posterior_network.py \
    --emulator cfm \
    --emulator-checkpoint "$ECHK" \
    --output-checkpoint-pt "$POUT" \
    --model "${POST_MODEL:-lite}" \
    --seed "$((2000 + i))" \
    --device "${DEVICE:-cpu}" \
    ${POST_EXTRA:-}
done
