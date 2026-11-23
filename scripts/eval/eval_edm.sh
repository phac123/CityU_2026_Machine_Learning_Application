#!/usr/bin/env bash
# Evaluation of the trained edm model on the evaluation set: 1024 samples, every
# metric of the report, raw weights (no EMA).
#
#   bash scripts/eval/eval_edm.sh                    # newest run
#   bash scripts/eval/eval_edm.sh ../other/last.pt   # positional checkpoint
#   CKPT=... NUM=512 bash scripts/eval/eval_edm.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

# add a default only if the caller did not pass that key (Hydra rejects duplicates)
has_override() {
  local key="$1"; shift
  local a
  for a in "$@"; do
    [[ "$a" == "$key"=* ]] && return 0
  done
  return 1
}
EXTRA=()

LOG_ROOT="${LOG_DIR:-./logs}"
if [[ -z "${CKPT:-}" && $# -gt 0 && "$1" == *.pt ]]; then
  CKPT="$1"; shift
fi
if [[ -z "${CKPT:-}" ]]; then
  CKPT="$(ls -t "${LOG_ROOT}/edm"/*/last.pt 2>/dev/null | head -n 1 || true)"
fi
if [[ -z "${CKPT:-}" ]]; then
  echo "[eval] no checkpoint found for edm; pass CKPT=... or a .pt path" >&2
  exit 1
fi

NUM="${NUM:-1024}"
has_override eval.num "$@"      || EXTRA+=("eval.num=${NUM}")
has_override eval.use_ema "$@"  || EXTRA+=("eval.use_ema=false")

METHOD=edm CKPT="${CKPT}" bash scripts/evaluate.sh "${EXTRA[@]}" "$@"
