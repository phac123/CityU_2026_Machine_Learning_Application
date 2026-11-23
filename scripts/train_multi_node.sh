#!/usr/bin/env bash
# Multi-node, multi-GPU training of one method.
#
# Run this script on EVERY node, with the same NNODES/STAMP/MASTER_PORT and a
# different NODE_RANK.  Node 0 hosts the rendezvous (MASTER_ADDR = its address).
#
#   # node 0
#   METHOD=geobfn NNODES=2 NODE_RANK=0 MASTER_ADDR=10.0.0.1 GPUS=8 \
#     STAMP=run42 bash scripts/train_multi_node.sh
#   # node 1
#   METHOD=geobfn NNODES=2 NODE_RANK=1 MASTER_ADDR=10.0.0.1 GPUS=8 \
#     STAMP=run42 bash scripts/train_multi_node.sh
#
# Notes
#   * STAMP must be identical on all nodes, otherwise the ranks write to different
#     run directories (only node 0 writes, but they must agree on the path);
#   * MASTER_PORT must be reachable from every node (open it in the firewall);
#   * NCCL prefers InfiniBand; set NCCL_SOCKET_IFNAME to the cluster interface if
#     the default route is not the one you want.
set -euo pipefail

METHOD="${METHOD:-geobfn}"
GPUS="${GPUS:-8}"
NNODES="${NNODES:-2}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:?set MASTER_ADDR to the address of node 0}"
MASTER_PORT="${MASTER_PORT:-29500}"
STAMP="${STAMP:-$(date +%Y%m%d-%H%M%S)}"
export HYDRA_RUN_DIR="${LOG_DIR:-./logs}/${METHOD}/${STAMP}"

if [[ "${NODE_RANK}" == "0" ]]; then
  mkdir -p "${HYDRA_RUN_DIR}"
fi

echo "[train] method=${METHOD} nodes=${NNODES} rank=${NODE_RANK} gpus=${GPUS}"
echo "[train] rendezvous: ${MASTER_ADDR}:${MASTER_PORT}"
echo "[train] run dir: ${HYDRA_RUN_DIR}"

torchrun \
  --nnodes="${NNODES}" \
  --node_rank="${NODE_RANK}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  --nproc_per_node="${GPUS}" \
  --rdzv_backend=static \
  train.py \
  method="${METHOD}" \
  "hydra.run.dir=${HYDRA_RUN_DIR}" \
  "$@"
