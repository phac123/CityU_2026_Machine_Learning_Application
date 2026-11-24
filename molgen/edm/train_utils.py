"""Small training helpers for the EDM branch (kept inside the branch).

Everything here is step-based: the training budget is a fixed number of
optimisation steps, which is what makes the three methods comparable.
"""

from __future__ import annotations

import threading
import time
from collections import deque
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


class EtaEstimator:
    """Steps per second and remaining time, from a rolling window of step stamps.

    A rolling window (default 50 steps) instead of a global average keeps the estimate
    useful: the first steps of a run are slow (cudnn autotuning, lazy initialisation,
    the first dataloader workers), and a global average would keep that spike in the
    number for the whole run.  ``tick`` is O(1) and allocation-free apart from the
    deque append, so it can be called on every step.
    """

    def __init__(self, total_steps: int, window: int = 50):
        self.total_steps = int(total_steps)
        self.window = max(int(window), 2)
        self._stamps: "deque[float]" = deque(maxlen=self.window)

    def tick(self, step: int, since: float = None) -> Dict[str, float]:
        """Register one finished step and return the current speed and ETA.

        With fewer than two samples in the window (the very first steps) it falls back
        to the average since ``since`` (the run start), so the first printed line
        already carries an estimate instead of "unknown"; the rolling window takes
        over as soon as it has data.
        """
        now = time.time()
        self._stamps.append(now)
        remaining = max(self.total_steps - int(step), 0)
        if len(self._stamps) >= 2:
            span = self._stamps[-1] - self._stamps[0]
            done = len(self._stamps) - 1
            speed = done / span if span > 0 else 0.0
        elif since is not None and step > 0 and now > since:
            speed = int(step) / (now - since)
        else:
            speed = 0.0
        eta = remaining / speed if speed > 0 else float("nan")
        return {"steps_per_second": speed, "eta_seconds": eta}

    def warm(self) -> bool:
        """True once the window has enough samples for a meaningful estimate."""
        return len(self._stamps) >= min(10, self.window)


def fmt_duration(seconds: float) -> str:
    """``01:23:45``-style duration; ``--:--:--`` while it is still unknown."""
    if seconds is None or seconds != seconds or seconds < 0:      # NaN guard
        return "--:--:--"
    seconds = int(round(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def progress_line(step: int, total: int, loss: float, speed: Dict[str, float],
                  elapsed: float, tag: str = "") -> str:
    """One compact progress line, including the projected wall-clock finish time."""
    total = max(int(total), 1)
    pct = 100.0 * int(step) / total
    sps = float(speed.get("steps_per_second", 0.0) or 0.0)
    eta_seconds = float(speed.get("eta_seconds", float("nan")))
    if eta_seconds == eta_seconds and eta_seconds >= 0:        # finite
        eta_txt = fmt_duration(eta_seconds)
        ends = time.strftime("%H:%M", time.localtime(time.time() + eta_seconds))
    else:
        eta_txt, ends = "--:--:--", "--:--"
    return (f"{tag} step {int(step):>5d}/{total} {pct:5.1f}% | loss {loss:.4f} | "
            f"{sps:5.2f} step/s | elapsed {fmt_duration(elapsed)} | "
            f"ETA {eta_txt} | ends {ends}")


def print_progress(line: str, tty: bool) -> None:
    """Show a live line: overwrite it on a terminal, append it to a log file."""
    if tty:
        print("\r" + line + "\033[K", end="", flush=True)
    else:
        print(line, flush=True)


def print_line(line: str, tty: bool) -> None:
    """Print a permanent line, clearing whatever the live progress line left behind."""
    if tty:
        print("\r" + line + "\033[K", flush=True)
    else:
        print(line, flush=True)


class Heartbeat:
    """Print a line when no training step has completed for a while.

    A silent stall is the hardest failure to debug in a distributed run: the processes
    are alive, the GPUs may even be busy, and nothing is printed.  This daemon thread
    checks the step counter every ``interval`` seconds and prints exactly one line if
    it has not moved -- so a healthy run stays quiet, and a stalled one says how long
    it has been stuck and which step it last finished.  It never touches the process
    group or the CUDA context, so it cannot itself cause a deadlock.
    """

    def __init__(self, interval: float = 30.0):
        self.interval = float(interval)
        self._step = 0
        self._last = time.time()
        self._stop = threading.Event()
        self._thread = None
        self._tty = False
        self._announced = False

    def start(self, tty: bool = False) -> "Heartbeat":
        self._tty = tty
        if self.interval <= 0:
            return self
        self._thread = threading.Thread(target=self._loop, name="heartbeat", daemon=True)
        self._thread.start()
        return self

    def update(self, step: int) -> None:
        """Called once per finished step: resets the stall timer."""
        self._step = int(step)
        self._last = time.time()
        self._announced = False

    def _loop(self) -> None:
        while not self._stop.wait(self.interval):
            idle = time.time() - self._last
            if idle >= self.interval * 0.9 and not self._announced:
                self._announced = True        # one line per stall, not one per interval
                try:
                    print_line(f"[heartbeat] no step finished for "
                               f"{fmt_duration(idle)} (last step: {self._step}); "
                               f"if this repeats, the run is stuck, not slow",
                               self._tty)
                except Exception:             # a diagnostic must never kill training
                    pass

    def stop(self) -> None:
        self._stop.set()
