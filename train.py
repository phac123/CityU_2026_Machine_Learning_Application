"""Hydra entry point for training one of the three methods.

    python train.py                        # GeoBFN (default), single process
    python train.py method=edm             # switch method
    python train.py method=llm train.lr=3e-4 data.batch_size=64
    torchrun --nproc_per_node=8 train.py method=geobfn      # 1 node, 8 GPUs
    bash scripts/train_multi_node.sh                        # several nodes

The file only dispatches: each method owns its training loop in
``molgen.<method>.train``.  Everything configurable lives in ``conf/``.
"""

from __future__ import annotations

import importlib
from typing import Dict

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

#: method name -> module that exposes ``train(cfg)``
TRAINERS: Dict[str, str] = {
    "geobfn": "molgen.geobfn.train",
    "edm": "molgen.edm.train",
    "llm": "molgen.llm.train",
}


def apply_smoke(cfg: DictConfig) -> DictConfig:
    """Shrink every branch so that the full pipeline runs in about a minute on CPU.

    Used by ``python train.py smoke=true`` (``make smoke``) to check that data
    loading, training, EMA, sampling, metrics and checkpointing all work end to end
    before a long run is launched.
    """
    OmegaConf.set_struct(cfg, False)
    cfg.data.name = "synthetic"
    cfg.data.num_workers = 0
    cfg.data.batch_size = 8
    cfg.data.eval_num = 8
    cfg.data.eval_batch_size = 8
    cfg.train.max_steps = 4
    cfg.train.log_every = 2
    cfg.train.eval_every = 2
    cfg.train.ema_decay = 0.9        # exercises the EMA code path (experiments: 0)
    # exercise the interval features too: a checkpoint and an analysis every 2 steps
    cfg.train.ckpt_every = 2
    cfg.train.ckpt_keep_all = True
    cfg.train.analysis_every = 2
    cfg.train.analysis_num = 4
    cfg.train.analysis_steps = 4
    cfg.train.analysis_pca = False    # synthetic data has no training SMILES to fit on
    cfg.train.analysis_save_samples = True
    cfg.train.amp = "none"
    cfg.train.resume = False
    for key, value in (("n_layers", 2), ("hidden_nf", 32), ("in_node_nf", 8),
                       ("time_embed_dim", 16), ("d_model", 64), ("n_layer", 2),
                       ("n_head", 4), ("dropout", 0.0)):
        if key in cfg.model:
            cfg.model[key] = value
    for key, value in (("sample_steps", 10), ("sample_num", 16), ("ckpt_every", 1)):
        if key in cfg.train:
            cfg.train[key] = value
    if "max_len" in cfg.data:
        cfg.data.max_len = 24
    if "diffusion" in cfg:
        cfg.diffusion.n_timesteps = 100
        cfg.diffusion.sample_steps = 10
    return cfg


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Compose the config, optionally shrink it, then call the branch's trainer."""
    name = str(cfg.name)
    if name not in TRAINERS:
        raise SystemExit(f"unknown method {name!r}; choose one of {sorted(TRAINERS)}")
    if cfg.get("smoke", False):
        cfg = apply_smoke(cfg)
        print("[train] smoke mode: synthetic data, tiny model, one epoch")

    print(f"[train] method={name} run_dir={HydraConfig.get().runtime.output_dir}")
    trainer = importlib.import_module(TRAINERS[name]).train
    trainer(cfg)


if __name__ == "__main__":
    main()
