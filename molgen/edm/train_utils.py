"""Small training helpers for the EDM branch (kept inside the branch).

Everything here is step-based: the training budget is a fixed number of
optimisation steps, which is what makes the three methods comparable.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset


def make_amp(device: torch.device, amp: str):
    """Return (autocast context factory, GradScaler) for ``none``/``bf16``/``fp16``."""
    if amp == "none" or device.type != "cuda":
        try:
            scaler = torch.amp.GradScaler("cuda", enabled=False)
        except Exception:  # torch < 2.3
            scaler = torch.cuda.amp.GradScaler(enabled=False)
        return (lambda: nullcontext()), scaler
    dtype = torch.bfloat16 if amp == "bf16" else torch.float16
    try:
        from torch.amp import autocast as _autocast

        ctx = lambda: _autocast(device_type="cuda", dtype=dtype)  # noqa: E731
        scaler = torch.amp.GradScaler("cuda", enabled=(amp == "fp16"))
    except Exception:
        from torch.cuda.amp import autocast as _autocast

        ctx = lambda: _autocast(enabled=True, dtype=dtype)  # noqa: E731
        scaler = torch.cuda.amp.GradScaler(enabled=(amp == "fp16"))
    return ctx, scaler


class LossModule(nn.Module):
    """Expose ``model.loss`` as a forward pass.

    ``DistributedDataParallel`` starts its gradient reduction from inside its own
    ``forward``, so the loss has to be *called* through the wrapper.  The per-step
    statistics are stored as plain floats in ``last_stats`` (never as tensors) so
    that DDP's output inspection never sees them.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.last_stats: Dict[str, float] = {}

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        loss, stats = self.model.loss(batch)
        self.last_stats = {k: float(v) for k, v in stats.items()}
        return loss


def split_fold(dataset, fold: int, n_folds: int, seed: int = 0):
    """Deterministic k-fold split of a dataset.

    ``fold < 0`` returns the whole dataset (the final runs validate on the official
    validation split instead); ``fold >= 0`` returns (train, validation) where the
    validation part is every ``n_folds``-th element of a seeded permutation.
    """
    if fold is None or fold < 0:
        return dataset, None
    generator = torch.Generator().manual_seed(int(seed))
    order = torch.randperm(len(dataset), generator=generator).tolist()
    valid_idx = order[fold::n_folds]
    valid_set = set(valid_idx)
    train_idx = [i for i in order if i not in valid_set]
    return Subset(dataset, train_idx), Subset(dataset, valid_idx)


def cap_dataset(dataset, limit: Optional[int]):
    """Keep at most ``limit`` examples (used to make the CV folds cheap)."""
    if limit is None or limit <= 0 or len(dataset) <= limit:
        return dataset
    return Subset(dataset, list(range(int(limit))))


def build_loader(cfg, dataset, shuffle: bool, distributed: bool
                 ) -> Tuple[DataLoader, Optional[object]]:
    """DataLoader plus (optionally) a ``DistributedSampler`` with per-step sharding."""
    sampler = None
    if distributed and shuffle:
        from torch.utils.data.distributed import DistributedSampler

        sampler = DistributedSampler(dataset, shuffle=True, drop_last=True)
    from .data import collate

    loader = DataLoader(
        dataset,
        batch_size=cfg.data.batch_size,
        shuffle=(shuffle and sampler is None),
        sampler=sampler,
        collate_fn=collate,
        num_workers=cfg.data.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=shuffle,
        persistent_workers=cfg.data.num_workers > 0,
    )
    return loader, sampler


@torch.no_grad()
def validation_loss(objective: nn.Module, loader: DataLoader, device: torch.device,
                    max_batches: int = 8) -> float:
    """Mean objective on up to ``max_batches`` validation batches.

    Every rank evaluates the same (unsharded) validation batches and the mean is
    averaged across ranks, so the value is identical everywhere.
    """
    total, count = 0.0, 0
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        total += float(objective(batch))
        count += 1
    return total / max(count, 1)
