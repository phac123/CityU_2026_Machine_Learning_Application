#!/usr/bin/env bash
# Formal training run of the edm branch: 10000 optimisation steps, batch
# 64 per GPU, no EMA (the comparison evaluates raw weights).
#
#   bash scripts/train/train_edm.sh                       # the defaults
#   bash scripts/train/train_edm.sh train.lr=0.0003       # CV winner, say
#   STEPS=2000 GPUS=1 bash scripts/train/train_edm.sh     # shorter / one GPU
#
# The fairness of the comparison depends on the three branches using the same
# STEPS/BATCH and the same number of GPUs; run_experiments.sh enforces that.
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

STEPS="${STEPS:-10000}"
BATCH="${BATCH:-64}"
SEED="${SEED:-0}"

has_override train.max_steps "$@" || EXTRA+=("train.max_steps=${STEPS}")
has_override data.batch_size "$@" || EXTRA+=("data.batch_size=${BATCH}")
has_override train.ema_decay "$@" || EXTRA+=("train.ema_decay=0")
has_override train.seed "$@"      || EXTRA+=("train.seed=${SEED}")

METHOD=edm exec bash "${ROOT}/scripts/train.sh" "${EXTRA[@]}" "$@"
