"""Distributed, step-budgeted training for the EDM branch (E(3)-equivariant diffusion).

Same protocol as the other two branches: ``train.max_steps`` optimisation steps, one
per-GPU batch size, and ``data.fold >= 0`` for the cross-validation runs (the k-th
random fold of the training split becomes the validation set).  EMA is off by
default, so what is evaluated is exactly what was trained.
"""

from __future__ import annotations

import json
import os
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from . import distributed as dist_utils
from .config import Config, from_hydra
from .data import EvalDataset, QM9Dataset, SyntheticDataset, collate, to_device
from .diffusion import EDMDiffusion
from .ema import EMA
from .analysis import analyse, append_log, write_samples
from .train_utils import (
    LossModule,
    build_loader,
    cap_dataset,
    make_amp,
    split_fold,
    validation_loss,
)


def build_dataset(cfg: Config, split: str):
    """``qm9`` reads the shared npz artefacts; ``synthetic`` is for smoke tests."""
    if cfg.data.name == "synthetic":
        return SyntheticDataset(256 if split == "train" else 32,
                                n_types=len(cfg.data.atomic_nb))
    limit = cfg.data.limit if split == "train" else cfg.data.valid_limit
    return QM9Dataset(cfg.data.root, split=split, pos_scale=cfg.data.pos_scale,
                      atomic_nb=cfg.data.atomic_nb, limit=limit)


def build_data(cfg: Config):
    """Return (train_dataset, validation_dataset, is_fold)."""
    full = build_dataset(cfg, "train")
    train_ds, fold_valid = split_fold(full, cfg.data.fold, cfg.data.n_folds,
                                      seed=cfg.train.seed)
    if fold_valid is None:
        valid_ds = cap_dataset(build_dataset(cfg, "valid"), cfg.data.valid_limit)
        return train_ds, valid_ds, False
    return train_ds, cap_dataset(fold_valid, cfg.data.valid_limit), True


def save_checkpoint(path: str, cfg: Config, model: nn.Module, ema: Optional[EMA],
                    optimizer: torch.optim.Optimizer, step: int,
                    metrics: Dict[str, float]) -> None:
    """Rank-0 checkpoint: weights, EMA, optimiser, step, metrics and config."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "ema": ema.state_dict() if ema is not None else None,
            "optimizer": optimizer.state_dict(),
            "step": step,
            "metrics": metrics,
            "config": cfg.to_dict(),
        },
        path,
    )



def run_analysis(cfg: Config, model: nn.Module, device: torch.device, step: int,
                 tag: str = "analysis") -> Dict[str, float]:
    """Sample, score and log the current weights (all of it lives in analysis.py).

    Called every ``train.analysis_every`` steps and once at the last step, so the
    finished run already contains its own evaluation: the five chemical metrics, the
    position of the samples on the descriptor manifold (PCA) and the samples
    themselves.
    """
    metrics, samples = analyse(
        cfg, model, device,
        num=int(cfg.train.analysis_num),
        steps=int(getattr(cfg.train, "analysis_steps", cfg.train.sample_steps)),
        out_dir=cfg.train.out_dir,
        seed=int(cfg.train.seed),
    )
    record = {"step": int(step), **metrics}
    if dist_utils.is_main_process():
        append_log(os.path.join(cfg.train.out_dir, "eval_log.jsonl"), record)
        if cfg.train.analysis_save_samples:
            write_samples(cfg, samples,
                          os.path.join(cfg.train.out_dir, "samples",
                                       f"{tag}_{step:06d}.xyz"))
        print(f"[edm] step {step} analysis: "
              f"validity={_fmt(metrics, 'validity')} "
              f"stable={_fmt(metrics, 'mol_stable')} "
              f"novel={_fmt(metrics, 'novelty')} "
              f"pca_outside={_fmt(metrics, 'pca_outside')}")
    model.train()
    return metrics


def _fmt(metrics: Dict[str, float], key: str) -> str:
    """Format one metric for the log line (n/a when it was not computed)."""
    value = metrics.get(key)
    return "n/a" if value is None or value != value else f"{value:.3f}"


def train(hydra_cfg: DictConfig) -> Dict[str, float]:
    """Run the step-budgeted training loop; returns the last validation metrics."""
    cfg: Config = from_hydra(hydra_cfg)
    cfg.train.out_dir = str(cfg.train.out_dir)
    max_steps = int(cfg.train.max_steps)

    dist_utils.setup()
    rank, world = dist_utils.get_rank(), dist_utils.get_world_size()
    device = dist_utils.device()
    distributed = dist_utils.is_distributed()

    torch.manual_seed(cfg.train.seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.train.seed + rank)

    if cfg.train.skip_training:
        # evaluation only: evaluate.py rebuilds the model from the checkpoint, so the
        # architecture cannot drift from the weights that are being scored
        metrics: Dict[str, float] = {}
        if dist_utils.is_main_process():
            from .evaluate import evaluate as evaluate_checkpoint

            metrics = evaluate_checkpoint(hydra_cfg)
            os.makedirs(cfg.train.out_dir, exist_ok=True)
            with open(os.path.join(cfg.train.out_dir, "final_metrics.json"), "w",
                      encoding="utf-8") as f:
                payload = {"method": "edm", "skipped_training": True}
                payload.update(metrics)
                json.dump(payload, f, indent=2)
            print("[edm] train.skip_training=true: no training, results in "
                  f"{cfg.train.out_dir}")
        dist_utils.barrier()
        dist_utils.cleanup()
        return metrics

    if dist_utils.is_main_process():
        os.makedirs(cfg.train.out_dir, exist_ok=True)
        from .config import save_config

        save_config(cfg, os.path.join(cfg.train.out_dir, "config.json"))
        print(f"[edm] world_size={world} device={device} out={cfg.train.out_dir}")

    train_ds, valid_ds, is_fold = build_data(cfg)
    train_loader, sampler = build_loader(cfg, train_ds, shuffle=True,
                                         distributed=distributed)
    valid_loader, _ = build_loader(cfg, valid_ds, shuffle=False, distributed=False)
    if dist_utils.is_main_process():
        print(f"[edm] {'fold' if is_fold else 'final'}: train={len(train_ds)} "
              f"valid={len(valid_ds)} per-gpu batch={cfg.data.batch_size} "
              f"global batch={cfg.data.batch_size * world} steps={max_steps}")

    n_types = len(cfg.data.atomic_nb)
    model = EDMDiffusion(cfg.model, cfg.diffusion, n_types=n_types,
                         atomic_nb=cfg.data.atomic_nb).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr,
                                  weight_decay=cfg.train.weight_decay, amsgrad=True)
    warmup = max(1, int(max_steps * cfg.train.warmup_ratio))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min(1.0, (s + 1) / warmup)
    )
    ema = EMA(model, decay=cfg.train.ema_decay) if cfg.train.ema_decay > 0 else None
    amp_ctx, scaler = make_amp(device, cfg.train.amp)

    objective: nn.Module = LossModule(model)
    if distributed:
        objective = nn.parallel.DistributedDataParallel(
            objective,
            device_ids=[dist_utils.get_local_rank()] if device.type == "cuda" else None,
            output_device=dist_utils.get_local_rank() if device.type == "cuda" else None,
            find_unused_parameters=bool(hydra_cfg.ddp.get("find_unused_parameters", False)),
        )
    if dist_utils.is_main_process():
        n_param = sum(p.numel() for p in model.parameters())
        print(f"[edm] parameters={n_param / 1e6:.2f}M")

    train_log = os.path.join(cfg.train.out_dir, "train_log.jsonl")
    metrics_log = os.path.join(cfg.train.out_dir, "metrics.jsonl")
    last_path = os.path.join(cfg.train.out_dir, "last.pt")

    step = 0
    running, seen = torch.zeros((), device=device), 0
    last_metrics: Dict[str, float] = {"valid_loss": float("nan")}
    last_analysis: Dict[str, float] = {}
    t0 = time.time()

    model.train()
    while step < max_steps:
        if sampler is not None:
            sampler.set_epoch(step)
        for batch in train_loader:
            batch = to_device(batch, device)
            with amp_ctx():
                loss = objective(batch)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if cfg.train.grad_clip and cfg.train.grad_clip > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                           cfg.train.grad_clip)
            else:
                grad_norm = torch.zeros((), device=device)
            if not torch.isfinite(grad_norm):
                optimizer.zero_grad(set_to_none=True)
                scaler.update()
                if dist_utils.is_main_process():
                    print(f"[edm] step {step}: non-finite gradient, skipped")
            else:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                if ema is not None:
                    ema.update(model)

            step += 1
            running += loss.detach()
            seen += 1

            if step % cfg.train.log_every == 0 or step == 1:
                mean_loss = dist_utils.all_reduce_mean(running / max(seen, 1))
                if dist_utils.is_main_process():
                    record = {
                        "step": step,
                        "loss": float(mean_loss),
                        "lr": scheduler.get_last_lr()[0],
                        "grad_norm": float(grad_norm),
                        "seconds": round(time.time() - t0, 2),
                    }
                    record.update(getattr(objective, "last_stats", {}))
                    with open(train_log, "a", encoding="utf-8") as f:
                        f.write(json.dumps(record) + "\n")
                    print(f"[edm] step {step}/{max_steps} loss={float(mean_loss):.4f} "
                          f"lr={record['lr']:.2e} |g|={float(grad_norm):.2f}")
                running, seen = torch.zeros((), device=device), 0

            if step % cfg.train.eval_every == 0 or step == max_steps:
                valid = validation_loss(objective, valid_loader, device,
                                        cfg.data.valid_batches)
                valid = float(dist_utils.all_reduce_mean(torch.tensor(valid, device=device)))
                last_metrics = {"valid_loss": valid, "train_loss": float(loss.detach()),
                                "step": step}
                if dist_utils.is_main_process():
                    print(f"[edm] step {step} valid_loss={valid:.4f}")
                    with open(metrics_log, "a", encoding="utf-8") as f:
                        f.write(json.dumps(last_metrics) + "\n")
                    if step % cfg.train.ckpt_every == 0 or step == max_steps:
                        save_checkpoint(last_path, cfg, model, ema, optimizer, step,
                                        last_metrics)
                        if cfg.train.ckpt_keep_all:
                            save_checkpoint(
                                os.path.join(cfg.train.out_dir, "ckpt",
                                             f"step{step:06d}.pt"),
                                cfg, model, ema, optimizer, step, last_metrics)
            if ((cfg.train.analysis_every > 0
                 and step % cfg.train.analysis_every == 0)
                    or step == max_steps):
                if dist_utils.is_main_process():
                    last_analysis = run_analysis(cfg, model, device, step)
                else:
                    last_analysis = {}

            if step >= max_steps:
                break

    if dist_utils.is_main_process():
        save_checkpoint(last_path, cfg, model, ema, optimizer, step, last_metrics)
        with open(os.path.join(cfg.train.out_dir, "final_metrics.json"), "w",
                  encoding="utf-8") as f:
            json.dump({"method": "edm", "analysis": last_analysis, **last_metrics,
                       "fold": int(cfg.data.fold), "max_steps": max_steps,
                       "batch_size": int(cfg.data.batch_size),
                       "world_size": world}, f, indent=2)
        print(f"[edm] done at step {step}, checkpoints in {cfg.train.out_dir}")
    dist_utils.barrier()
    dist_utils.cleanup()
    return last_metrics
