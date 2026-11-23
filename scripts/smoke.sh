#!/usr/bin/env bash
# End-to-end smoke test of one method: synthetic data, tiny model, one epoch, then
# a small sampling evaluation.  Runs on CPU in about a minute and needs no QM9.
#
#   bash scripts/smoke.sh            # all three methods
#   METHOD=edm bash scripts/smoke.sh # only one
set -euo pipefail

METHODS=("${METHOD:-geobfn}")
if [[ -z "${METHOD:-}" ]]; then
  METHODS=(geobfn edm llm)
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

for m in "${METHODS[@]}"; do
  RUN_DIR="${LOG_DIR:-./logs}/smoke/${m}"
  rm -rf "${RUN_DIR}"
  echo
  echo "=== smoke: ${m} ==============================================="
  python train.py method="${m}" smoke=true "hydra.run.dir=${RUN_DIR}"

  CKPT="${RUN_DIR}/last.pt"
  if [[ ! -f "${CKPT}" ]]; then
    echo "[smoke] ${m}: no checkpoint was written, aborting" >&2
    exit 1
  fi
  EXTRA=("eval.num=4")
  if [[ "${m}" == "llm" ]]; then
    EXTRA+=("eval.conformers=false")
  fi
  python evaluate.py method="${m}" "eval.ckpt=${CKPT}" "${EXTRA[@]}"
  echo "=== smoke: ${m} OK ==========================================="
done
