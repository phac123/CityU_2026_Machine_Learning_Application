#!/usr/bin/env bash
# =============================================================================
# debug_run.sh -- a 30-second isolation test: few steps, no workers, no analysis.
#
#   bash scripts/debug_run.sh              # one process (llm, 20 steps)
#   GPUS=8 bash scripts/debug_run.sh       # the same under torchrun -> DDP test
#   METHOD=geobfn STEPS=10 GPUS=2 bash scripts/debug_run.sh
#
# How to read the result:
#   * 1 process completes, 8 processes stop -> the model and the data are fine and
#     the problem is DDP (unused parameters, rank count, collectives);
#   * both stop -> it is the model or the data loader, and the last printed line
#     says where.
#
# Fail fast instead of hanging: TORCH_NCCL_ASYNC_ERROR_HANDLING aborts a stuck
# collective, and ddp.timeout_minutes=2 turns a silent wait into a traceback.
# =============================================================================
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

METHOD="${METHOD:-llm}"
STEPS="${STEPS:-20}"
GPUS="${GPUS:-1}"
TIMEOUT_MIN="${TIMEOUT_MIN:-2}"
PORT="${MASTER_PORT:-29611}"

export PYTHONFAULTHANDLER=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

echo "[debug] method=${METHOD} steps=${STEPS} processes=${GPUS} cwd=$(pwd)"
python - <<'PY'
import torch
print(f"[debug] torch {torch.__version__} | cuda {torch.cuda.is_available()} "
      f"| devices {torch.cuda.device_count()}")
PY

ARGS=(
  method="${METHOD}"
  "train.max_steps=${STEPS}"
  data.num_workers=0
  data.batch_size=8
  train.log_every=1
  train.progress_every=1
  "train.eval_every=${STEPS}"
  train.analysis_every=0
  "train.ckpt_every=${STEPS}"
  ddp.timeout_minutes="${TIMEOUT_MIN}"
  "hydra.run.dir=./logs/debug/${METHOD}"
)

if [[ "${GPUS}" -le 1 ]]; then
  python train.py "${ARGS[@]}"
else
  torchrun --nproc_per_node="${GPUS}" --master_port="${PORT}" train.py "${ARGS[@]}"
fi

echo
echo "[debug] ${METHOD}: ${STEPS} steps with ${GPUS} process(es) finished."
