#!/usr/bin/env bash
# Train and then evaluate all three methods, one after the other.
#
#   bash scripts/run_all.sh                       # full training on all GPUs
#   STEPS=10 EPOCHS=2 bash scripts/run_all.sh     # quick pass on a subset
#
# Extra arguments are forwarded to every training run (Hydra overrides).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

export LOG_DIR="${LOG_DIR:-./logs}"
mkdir -p "${LOG_DIR}"

for m in llm edm geobfn; do
  echo
  echo "########## training ${m} ##########"
  METHOD="${m}" bash scripts/train.sh "$@"

  CKPT="$(ls -t "${LOG_DIR}/${m}"/*/last.pt 2>/dev/null | head -n 1 || true)"
  if [[ -z "${CKPT}" ]]; then
    echo "[run_all] no checkpoint for ${m}, skipping evaluation" >&2
    continue
  fi
  echo "########## evaluating ${m} (${CKPT}) ##########"
  METHOD="${m}" CKPT="${CKPT}" bash scripts/evaluate.sh
done

echo
echo "[run_all] finished; logs under ${LOG_DIR}"
