#!/usr/bin/env bash
# Ten random samples of the trained llm model, plus the per-molecule analysis
# used by the "Molecular generation" section of the report.
#
#   bash scripts/samples/sample_llm.sh                  # newest run
#   bash scripts/samples/sample_llm.sh ../other/last.pt
#
# Writes next to the checkpoint:
#   samples10/samples.smi         the ten samples
#   samples10/metrics.json  the five metrics on those ten samples
#   samples10/analysis.md   one markdown block per model
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
  CKPT="$(ls -t "${LOG_ROOT}/llm"/*/last.pt 2>/dev/null | head -n 1 || true)"
fi
if [[ -z "${CKPT:-}" ]]; then
  echo "[samples] no checkpoint found for llm; pass CKPT=... or a .pt path" >&2
  exit 1
fi

OUT_DIR="${OUT_DIR:-$(dirname "${CKPT}")/samples10}"
mkdir -p "${OUT_DIR}"
NUM="${NUM:-10}"

has_override eval.ckpt "$@"     || EXTRA+=("eval.ckpt=${CKPT}")
has_override eval.num "$@"      || EXTRA+=("eval.num=${NUM}")
has_override eval.use_ema "$@"  || EXTRA+=("eval.use_ema=false")
has_override eval.seed "$@"     || EXTRA+=("eval.seed=${SEED:-0}")
has_override eval.out "$@"      || EXTRA+=("eval.out=${OUT_DIR}/samples.smi")
has_override eval.json_out "$@" || EXTRA+=("eval.json_out=${OUT_DIR}/metrics.json")
has_override eval.xyz_out "$@" || EXTRA+=("eval.xyz_out=${OUT_DIR}/conformers.xyz")
python evaluate.py method=llm "${EXTRA[@]}" "$@"

python tools/analyse_samples.py --method llm \
  --input "${OUT_DIR}/samples.smi" --out "${OUT_DIR}/analysis.md"
echo "[samples] llm: ${OUT_DIR}"
