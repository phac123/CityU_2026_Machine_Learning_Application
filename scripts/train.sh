#!/usr/bin/env bash
# Single-node, multi-GPU training of one method.
#
#   METHOD=geobfn GPUS=8 bash scripts/train.sh
#   METHOD=edm    GPUS=4 bash scripts/train.sh data.batch_size=32 train.lr=1e-4
#
# Any extra argument is forwarded to Hydra, so the whole config can be overridden.
# The run directory is computed once and passed to every rank, so that all ranks
# write into the same folder (only rank 0 actually writes).
set -euo pipefail

# fail fast if a collective desynchronises, instead of hanging forever
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"

METHOD="${METHOD:-geobfn}"
if [[ -z "${GPUS:-}" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    GPUS="$(nvidia-smi -L | wc -l)"
  else
    GPUS=1
  fi
fi
MASTER_PORT="${MASTER_PORT:-29500}"
STAMP="${STAMP:-$(date +%Y%m%d-%H%M%S)}"
export HYDRA_RUN_DIR="${LOG_DIR:-./logs}/${METHOD}/${STAMP}"
mkdir -p "${HYDRA_RUN_DIR}"

# Pre-flight: a mis-packaged conf/method/*.yaml makes every Hydra override fail with
# "Key 'train' is not in struct" on all ranks; catching it here costs 100 ms and turns
# eight stack traces into one sentence.
python tools/check_configs.py || {
  echo "[train] the Hydra configs are inconsistent: fix them with" >&2
  echo "        python tools/check_configs.py --fix" >&2
  exit 1
}

echo "[train] method=${METHOD} gpus=${GPUS} node=$(hostname) port=${MASTER_PORT}"
echo "[train] run dir: ${HYDRA_RUN_DIR}"

torchrun \
  --nproc_per_node="${GPUS}" \
  --master_port="${MASTER_PORT}" \
  train.py \
  method="${METHOD}" \
  "hydra.run.dir=${HYDRA_RUN_DIR}" \
  "$@"
