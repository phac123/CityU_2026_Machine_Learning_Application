# =============================================================================
# FILE ROLE
#   A thin wrapper around distributed training: initialize/destroy the process
#   group, expose rank queries, and average across GPUs.
#   The goal is to keep train.py free of any `dist.xxx` boilerplate.
#
# IMPLEMENTATION
#   1) Under torchrun the environment provides RANK / WORLD_SIZE / LOCAL_RANK /
#      MASTER_ADDR / MASTER_PORT. `setup()` detects RANK and calls
#      `init_process_group("env://")`, then `torch.cuda.set_device(LOCAL_RANK)`
#      -- **this step is critical**: each rank may only touch its own GPU,
#      otherwise all-reduce deadlocks.
#   2) Without RANK it degrades to single-process CPU/GPU, with an identical
#      code path (handy for local debugging).
#   3) Backend choice: nccl for CUDA (fastest on GPU, supports multi-node
#      InfiniBand), gloo otherwise. For multi-node multi-GPU, nccl performs its
#      own topology discovery.
#   4) `all_reduce_mean()` averages the per-GPU loss globally, so that the loss
#      in the log keeps the "global batch" semantics.
#
# MULTI-NODE MULTI-GPU NOTES (also in the README and the PDF)
#   - On each node: torchrun --nnodes=N --node_rank=i --master_addr=<node0 IP>
#                          --master_port=29500 --nproc_per_node=<local GPUs> train.py
#   - Only rank0 (the global main process) writes logs, saves checkpoints and
#     runs sampling evaluation; the other ranks wait on barrier() so that the
#     same file is never written concurrently.
# =============================================================================
"""Distributed-training initialization and common utilities."""

from __future__ import annotations

import datetime
import os
from typing import Optional

import torch
import torch.distributed as dist


# --------------------------------------------------------------------------- #
# Initialization / destruction
# --------------------------------------------------------------------------- #
def setup(backend: Optional[str] = None, timeout_minutes: int = 30) -> None:
    """Initialize the process group under torchrun; otherwise do nothing.

    Args:
        backend: explicit backend; when None, nccl/gloo is selected based on
                 CUDA availability.
        timeout_minutes: process-group timeout. Sampling evaluation can be slow;
                         too small a value is mistaken for a failure.

    Notes:
        Uses `env://` by default (torchrun sets MASTER_ADDR/PORT/RANK/WORLD_SIZE).
        If the environment variable `GEOBFN_INIT_METHOD` is set to
        `file:///path/to/store`, file-lock rendezvous (FileStore) is used
        instead -- this switch rescues single-node multi-GPU runs, or Windows
        with the pip build of torch where a missing libuv makes TCPStore
        initialization fail.

        WARNING: the file:// store is **stateful**; it must be deleted before
        the next launch, otherwise rank1 keeps waiting for rank0's stale
        counters (the symptom is training hanging in init_process_group).
        In production prefer torchrun's env:// (or NCCL's own TCPStore).
    """
    if "RANK" not in os.environ or not dist.is_available():
        return
    if dist.is_initialized():
        return

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        backend = backend or "nccl"
    else:
        backend = backend or "gloo"

    init_method = os.environ.get("GEOBFN_INIT_METHOD", "env://")
    kwargs = dict(backend=backend, timeout=datetime.timedelta(minutes=timeout_minutes))
    if init_method == "env://":
        dist.init_process_group(init_method=init_method, **kwargs)
    else:
        # non-env:// requires explicit rank/world_size
        dist.init_process_group(
            init_method=init_method,
            rank=int(os.environ["RANK"]),
            world_size=int(os.environ["WORLD_SIZE"]),
            **kwargs,
        )


def cleanup() -> None:
    """Destroy the process group at the end of training (avoids leaked ports/shared memory)."""
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


# --------------------------------------------------------------------------- #
# Queries
# --------------------------------------------------------------------------- #
def is_distributed() -> bool:
    """Whether we are in multi-process (DDP) mode."""
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    """Global rank (0 in single-process mode)."""
    return dist.get_rank() if is_distributed() else 0


def get_world_size() -> int:
    """Global number of processes (1 in single-process mode)."""
    return dist.get_world_size() if is_distributed() else 1


def get_local_rank() -> int:
    """Rank within this node (determines which GPU to use)."""
    return int(os.environ.get("LOCAL_RANK", 0))


def is_main_process() -> bool:
    """Whether this is the global main process: only it writes files, logs and runs evaluation."""
    return get_rank() == 0


def barrier() -> None:
    """Synchronize all processes (others continue once the main process is done writing)."""
    if is_distributed():
        dist.barrier()


def device() -> torch.device:
    """Device this process should use."""
    if torch.cuda.is_available():
        return torch.device("cuda", get_local_rank())
    return torch.device("cpu")


# --------------------------------------------------------------------------- #
# Communication
# --------------------------------------------------------------------------- #
@torch.no_grad()
def all_reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    """Average the same scalar across all processes; return a new tensor.

    The input is not modified.
    Use case: training logs report a per-GPU mean loss, but we want the
    "global batch" mean.
    """
    if not is_distributed():
        return tensor.detach()
    t = tensor.detach().clone()
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t / get_world_size()


def reduce_dict_mean(d: dict) -> dict:
    """Average all values in a dict[str, float] across processes (for metric logging)."""
    if not is_distributed() or not d:
        return d
    keys = sorted(d)
    t = torch.tensor([float(d[k]) for k in keys], device=device())
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    t = t / get_world_size()
    return {k: float(v) for k, v in zip(keys, t.cpu())}
