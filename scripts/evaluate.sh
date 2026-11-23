#!/usr/bin/env bash
# Evaluate / sample with a trained checkpoint.
#
#   METHOD=geobfn CKPT=/workspace/logs/geobfn/run1/best.pt bash scripts/evaluate.sh
#   METHOD=llm   CKPT=... bash scripts/evaluate.sh eval.num=512 eval.conformers=true
#
# Evaluation runs in one process (reproducible sampling).  Results are written next
# to the checkpoint unless OUT_DIR is given.
set -euo pipefail

METHOD="${METHOD:-geobfn}"
CKPT="${CKPT:?set CKPT to the checkpoint you want to evaluate}"
OUT_DIR="${OUT_DIR:-$(dirname "${CKPT}")/eval_$(date +%Y%m%d-%H%M%S)}"
mkdir -p "${OUT_DIR}"

if [[ "${METHOD}" == "llm" ]]; then
  SAMPLES="${OUT_DIR}/samples.smi"
  CONFORMERS="${OUT_DIR}/conformers.xyz"
else
  SAMPLES="${OUT_DIR}/samples.xyz"
  CONFORMERS=""
fi

echo "[eval] method=${METHOD} ckpt=${CKPT} out=${OUT_DIR}"

EXTRA=()
if [[ -n "${CONFORMERS}" ]]; then
  EXTRA+=("eval.conformers=true" "eval.xyz_out=${CONFORMERS}")
fi

python evaluate.py \
  method="${METHOD}" \
  "eval.ckpt=${CKPT}" \
  "eval.json_out=${OUT_DIR}/metrics.json" \
  "eval.out=${SAMPLES}" \
  "${EXTRA[@]}" \
  "$@"
