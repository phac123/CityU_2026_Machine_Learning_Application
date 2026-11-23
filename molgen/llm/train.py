"""Distributed, step-budgeted training for the LLM branch (SMILES language model).

Same protocol as the two geometric branches: ``train.max_steps`` optimisation steps,
one per-GPU batch size, ``data.fold >= 0`` for a cross-validation fold, and a
validation loss (teacher-forced cross entropy on held-out molecules) as the selection
criterion.  The branch has no EMA by construction, so the weights that are evaluated
are exactly the weights that were trained.
"""

from __future__ import annotations

import json
import math
import os
import time
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from omegaconf import DictConfig

from . import distributed as dist_utils
from .config import Config, from_hydra
from .dataset import SMOKE_SMILES, SmilesDataset, build_datasets
from .model import MolGPT
from .tokenizer import SmilesTokenizer
from .analysis import analyse, append_log, write_samples
from .train_utils import (
    LossModule,
    build_lm_loader,
    cap_dataset,
    make_amp,
    split_fold,
    validation_loss,
)


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def build_data(cfg: Config):
    """Return (train_dataset, validation_dataset, tokenizer, is_fold)."""
    if cfg.data.name == "synthetic":
        tokenizer = SmilesTokenizer.build(SMOKE_SMILES, min_count=1)
        full = SmilesDataset(SMOKE_SMILES, tokenizer, max_len=cfg.data.max_len,
                             randomize=False)
        train_ds, fold_valid = split_fold(full, cfg.data.fold, cfg.data.n_folds,
                                          seed=cfg.train.seed)
        valid_ds = fold_valid if fold_valid is not None else cap_dataset(full, 32)
        return train_ds, cap_dataset(valid_ds, cfg.data.valid_limit), tokenizer, \
            fold_valid is not None

    train_full, valid_full, tokenizer = build_datasets(cfg.data)
    train_ds, fold_valid = split_fold(train_full, cfg.data.fold, cfg.data.n_folds,
                                      seed=cfg.train.seed)
    if fold_valid is None:
        return train_ds, cap_dataset(valid_full, cfg.data.valid_limit), tokenizer, False
    return train_ds, cap_dataset(fold_valid, cfg.data.valid_limit), tokenizer, True


def validity_of(smiles) -> float:
    """Fraction of strings RDKit can parse (NaN when RDKit is unavailable)."""
    try:
        from .chem import is_valid_smiles

        if not smiles:
            return 0.0
        return sum(1 for s in smiles if is_valid_smiles(s)) / len(smiles)
    except Exception:
        return float("nan")


def save_checkpoint(path: str, cfg: Config, model: MolGPT, tokenizer: SmilesTokenizer,
                    optimizer: torch.optim.Optimizer, step: int,
                    metrics: Dict[str, float]) -> None:
    """Checkpoint including the vocabulary: inference rebuilds identical ids."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "metrics": metrics,
            "config": cfg.to_dict(),
            "itos": tokenizer.itos,
        },
        path,
    )


# --------------------------------------------------------------------------- #
# training
# --------------------------------------------------------------------------- #

def run_analysis(cfg: Config, model: nn.Module, tokenizer: SmilesTokenizer,
                 device: torch.device, step: int, tag: str = "analysis") -> Dict[str, float]:
    """Sample strings, score them and place them on the training manifold.

    Called every ``train.analysis_every`` steps and once at the last step, so the run
    already contains its own evaluation (validity / uniqueness / novelty + PCA).
    """
    metrics, smiles = analyse(
        cfg, model, tokenizer, device,
        num=int(cfg.train.analysis_num),
        out_dir=cfg.train.out_dir,
    )
    record = {"step": int(step), **metrics}
    if dist_utils.is_main_process():
        append_log(os.path.join(cfg.train.out_dir, "eval_log.jsonl"), record)
        if cfg.train.analysis_save_samples:
            write_samples(cfg, smiles,
                          os.path.join(cfg.train.out_dir, "samples",
                                       f"{tag}_{step:06d}.smi"))
        print(f"[llm] step {step} analysis: "
              f"validity={_fmt(metrics, 'validity')} "
              f"unique={_fmt(metrics, 'uniqueness')} "
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
        # evaluation only: evaluate.py rebuilds the tokeniser and the model from the
        # checkpoint, so the vocabulary cannot drift from the weights being scored
        metrics: Dict[str, float] = {}
        if dist_utils.is_main_process():
            from .evaluate import evaluate as evaluate_checkpoint

            metrics = evaluate_checkpoint(hydra_cfg)
            os.makedirs(cfg.train.out_dir, exist_ok=True)
            with open(os.path.join(cfg.train.out_dir, "final_metrics.json"), "w",
                      encoding="utf-8") as f:
                payload = {"method": "llm", "skipped_training": True}
                payload.update(metrics)
                json.dump(payload, f, indent=2)
            print("[llm] train.skip_training=true: no training, results in "
                  f"{cfg.train.out_dir}")
        dist_utils.barrier()
        dist_utils.cleanup()
        return metrics

    if dist_utils.is_main_process():
        os.makedirs(cfg.train.out_dir, exist_ok=True)
        from .config import save_config

        save_config(cfg, os.path.join(cfg.train.out_dir, "config.json"))
        print(f"[llm] world_size={world} device={device} out={cfg.train.out_dir}")

    train_ds, valid_ds, tokenizer, is_fold = build_data(cfg)
    train_loader, sampler = build_lm_loader(cfg, train_ds, tokenizer, shuffle=True,
                                            distributed=distributed)
    valid_loader, _ = build_lm_loader(cfg, valid_ds, tokenizer, shuffle=False,
                                      distributed=False)
    if dist_utils.is_main_process():
        tokenizer.save(os.path.join(cfg.train.out_dir, "vocab.json"))
        print(f"[llm] {'fold' if is_fold else 'final'}: train={len(train_ds)} "
              f"valid={len(valid_ds)} vocab={tokenizer.vocab_size} "
              f"per-gpu batch={cfg.data.batch_size} "
              f"global batch={cfg.data.batch_size * world} steps={max_steps}")

    model = MolGPT(
        vocab_size=tokenizer.vocab_size,
        max_len=cfg.data.max_len,
        d_model=cfg.model.d_model,
        n_head=cfg.model.n_head,
        n_layer=cfg.model.n_layer,
        dropout=cfg.model.dropout,
        n_cond=cfg.data.n_cond_bins,
        tie_embeddings=cfg.model.tie_embeddings,
    ).to(device)
    optimizer = torch.optim.AdamW(_param_groups(model, cfg.train.weight_decay),
                                  lr=cfg.train.lr, betas=(0.9, 0.95))
    warmup = max(1, int(max_steps * cfg.train.warmup_ratio))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min(1.0, (s + 1) / warmup)
    )
    amp_ctx, scaler = make_amp(device, cfg.train.amp)

    objective: nn.Module = LossModule(model)
    if distributed:
        objective = nn.parallel.DistributedDataParallel(
            objective,
            device_ids=[dist_utils.get_local_rank()] if device.type == "cuda" else None,
            output_device=dist_utils.get_local_rank() if device.type == "cuda" else None,
            find_unused_parameters=bool(hydra_cfg.ddp.get("find_unused_parameters", True)),
        )
    if dist_utils.is_main_process():
        print(f"[llm] parameters={model.num_parameters() / 1e6:.2f}M")

    train_log = os.path.join(cfg.train.out_dir, "train_log.jsonl")
    metrics_log = os.path.join(cfg.train.out_dir, "metrics.jsonl")
    last_path = os.path.join(cfg.train.out_dir, "last.pt")

    step = 0
    running, seen = 0.0, 0
    last_metrics: Dict[str, float] = {"valid_loss": float("nan")}
    last_analysis: Dict[str, float] = {}
    t0 = time.time()

    model.train()
    while step < max_steps:
        if sampler is not None:
            sampler.set_epoch(step)
        for batch in train_loader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with amp_ctx():
                loss = objective(batch)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if cfg.train.grad_clip > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                           cfg.train.grad_clip)
            else:
                grad_norm = torch.zeros((), device=device)
            if not torch.isfinite(grad_norm):
                optimizer.zero_grad(set_to_none=True)
                scaler.update()
                if dist_utils.is_main_process():
                    print(f"[llm] step {step}: non-finite gradient, skipped")
            else:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

            step += 1
            running += float(loss.detach())
            seen += 1

            if step % cfg.train.log_every == 0 or step == 1:
                mean_loss = dist_utils.all_reduce_mean(
                    torch.tensor(running / max(seen, 1), device=device))
                if dist_utils.is_main_process():
                    record = {
                        "step": step,
                        "loss": float(mean_loss),
                        "ppl": math.exp(min(float(mean_loss), 20.0)),
                        "lr": scheduler.get_last_lr()[0],
                        "grad_norm": float(grad_norm),
                        "seconds": round(time.time() - t0, 2),
                    }
                    with open(train_log, "a", encoding="utf-8") as f:
                        f.write(json.dumps(record) + "\n")
                    print(f"[llm] step {step}/{max_steps} loss={float(mean_loss):.4f} "
                          f"ppl={record['ppl']:.2f} lr={record['lr']:.2e}")
                running, seen = 0.0, 0

            if step % cfg.train.eval_every == 0 or step == max_steps:
                valid = validation_loss(objective, valid_loader, device,
                                        cfg.data.valid_batches)
                valid = float(dist_utils.all_reduce_mean(torch.tensor(valid, device=device)))
                last_metrics = {"valid_loss": valid,
                                "valid_ppl": math.exp(min(valid, 20.0)),
                                "train_loss": float(loss.detach()), "step": step}
                if dist_utils.is_main_process():
                    print(f"[llm] step {step} valid_loss={valid:.4f} "
                          f"valid_ppl={last_metrics['valid_ppl']:.2f}")
                    with open(metrics_log, "a", encoding="utf-8") as f:
                        f.write(json.dumps(last_metrics) + "\n")
                    if step % cfg.train.ckpt_every == 0 or step == max_steps:
                        save_checkpoint(last_path, cfg, model, tokenizer, optimizer,
                                        step, last_metrics)
                        if cfg.train.ckpt_keep_all:
                            save_checkpoint(
                                os.path.join(cfg.train.out_dir, "ckpt",
                                             f"step{step:06d}.pt"),
                                cfg, model, tokenizer, optimizer, step, last_metrics)
            if ((cfg.train.analysis_every > 0
                 and step % cfg.train.analysis_every == 0)
                    or step == max_steps):
                if dist_utils.is_main_process():
                    last_analysis = run_analysis(cfg, model, tokenizer, device, step)
                else:
                    last_analysis = {}

            if step >= max_steps:
                break

    if dist_utils.is_main_process():
        save_checkpoint(last_path, cfg, model, tokenizer, optimizer, step, last_metrics)
        with open(os.path.join(cfg.train.out_dir, "final_metrics.json"), "w",
                  encoding="utf-8") as f:
            json.dump({"method": "llm", "analysis": last_analysis,
                       **last_metrics,
                       "fold": int(cfg.data.fold), "max_steps": max_steps,
                       "batch_size": int(cfg.data.batch_size),
                       "world_size": world}, f, indent=2)
        print(f"[llm] done at step {step}, checkpoints in {cfg.train.out_dir}")
    dist_utils.barrier()
    dist_utils.cleanup()
    return last_metrics


def _param_groups(model: nn.Module, weight_decay: float):
    """No weight decay on LayerNorm parameters and biases."""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim < 2 or name.endswith(".bias"):
            no_decay.append(p)
        else:
            decay.append(p)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
