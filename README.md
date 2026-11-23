[toc]
# Molecular generation with three paradigms — code and experiment manual

All the code behind the report (`../machine_learning_final_report.tex`) and everything
needed to reproduce its experiments: the three independent methods, the shared dataset
builder, the cross-validation that selects hyper-parameters, the training and
evaluation launchers, the exploratory analysis (descriptors, PCA, KMeans, t-SNE) and
the qualitative ten-sample analysis.

The three methods are deliberately isolated: `molgen/llm`, `molgen/edm` and
`molgen/geobfn` share no model, no data loader and no training loop. Only two things
are common, and on purpose:

* the **dataset artefacts** in `$DATA_DIR/qm9/raw/`, built once by
  `molgen/preprocess/`, so the three models see the same molecules and the same split;
* the **comparison protocol**: batch 64 per GPU, 10 000 optimisation steps, the same
  number of GPUs, and no EMA (report §5.1).

**Contents**

- [Molecular generation with three paradigms — code and experiment manual](#molecular-generation-with-three-paradigms--code-and-experiment-manual)
  - [1. What you need](#1-what-you-need)
  - [2. Build the environment once: `make init`](#2-build-the-environment-once-make-init)
  - [3. Enter the environment: `make in`](#3-enter-the-environment-make-in)
  - [4. Prepare the shared dataset (once)](#4-prepare-the-shared-dataset-once)
  - [5. Smoke test](#5-smoke-test)
  - [6. The experiments, in the order of the report](#6-the-experiments-in-the-order-of-the-report)
    - [6.1 Exploratory analysis — descriptors, PCA, KMeans, t-SNE](#61-exploratory-analysis--descriptors-pca-kmeans-t-sne)
    - [6.2 Cross-validation — selecting the hyper-parameters](#62-cross-validation--selecting-the-hyper-parameters)
    - [6.3 Training — the three methods, same budget](#63-training--the-three-methods-same-budget)
    - [6.4 Evaluation — metrics on the evaluation set](#64-evaluation--metrics-on-the-evaluation-set)
    - [6.5 Ten random samples and their analysis](#65-ten-random-samples-and-their-analysis)
    - [6.6 Loss against training step](#66-loss-against-training-step)
  - [7. One command for everything: `run_experiments.sh`](#7-one-command-for-everything-run_experimentssh)
  - [8. Report ↔ command ↔ output](#8-report--command--output)
  - [9. Fairness of the comparison](#9-fairness-of-the-comparison)
  - [10. Runtimes and resources](#10-runtimes-and-resources)
  - [11. Configuration (Hydra)](#11-configuration-hydra)
  - [12. Distributed training](#12-distributed-training)
  - [13. Troubleshooting](#13-troubleshooting)
  - [14. Repository layout](#14-repository-layout)
  - [15. Tests](#15-tests)

---

## 1. What you need

| Requirement | Notes |
|---|---|
| Linux server, **8 GPUs on one node** | the protocol and the timings of §10 assume one node with 8 GPUs; fewer work, but all three methods must then use the same number |
| Docker + NVIDIA Container Toolkit | `docker run --gpus all nvidia-smi` must list the devices |
| GNU `make` | `make init`, `make in`, … (the `Makefile` in this folder) |
| ~80 GB disk | QM9 artefacts (~1 GB), three checkpoints, samples |
| ≥ 16 GB RAM | the dataset builder holds QM9 in memory |
| raw QM9 | `gdb9.sdf` (with coordinates) from the QM9 page, or any `.smi` with one SMILES per line |

Nothing is installed on the host: PyTorch, RDKit, Hydra and scikit-learn live inside
the image.

---

## 2. Build the environment once: `make init`

From this folder:

```bash
make init
```

It asks for four directories, writes them to `.env` (git-ignored; the template is
`.env.example`), creates them and builds the image from
`pytorch/pytorch:2.4.1` (CUDA 12.1, cuDNN 9) plus `docker/requirements.txt`:

| Question | Variable | Mounted at | Contents |
|---|---|---|---|
| code | `CODE_DIR` | `/workspace/code` | this repository (read-write) |
| data | `DATA_DIR` | `/workspace/data` | QM9 `npz`/`smi` artefacts |
| logs | `LOG_DIR` | `/workspace/logs` | one folder per run: config, logs, checkpoints, samples |
| container cache | `CONTAINER_DIR` | `/container` | pip / `TORCH_HOME` / `HF_HOME` caches (survive rebuilds) |

Companions:

```bash
make doctor    # docker, the GPU runtime, the four directories, .env
make status    # docker compose ps
make build     # rebuild after editing docker/requirements.txt
make down      # stop the service
```

Without `make`, every target has a direct equivalent: `python docker.py <command>`
(`init`, `in`, `exec`, `train`, `eval`, `smoke`, `test`, `status`, `doctor`, …).

---

## 3. Enter the environment: `make in`

```bash
make in
```

is `docker compose run --rm dev bash`: a shell inside the container, in
`/workspace/code`, with the four directories mounted, 16 GB of `/dev/shm`, `ipc: host`
(both required by NCCL) and every GPU visible. **Every command below runs in that
shell**, from `/workspace/code`.

A single command from the host, without a shell:

```bash
make exec ARGS="python -m molgen.preprocess.explore --help"
```

---

## 4. Prepare the shared dataset (once)

```bash
make preprocess ARGS="--source /workspace/data/qm9/gdb9.sdf --split random"
```

equivalently, inside the container:

```bash
python -m molgen.preprocess.build_dataset \
    --source /workspace/data/qm9/gdb9.sdf --out /workspace/data/qm9 --split random
```

It writes, under `$DATA_DIR/qm9/`:

```
raw/train.npz  raw/valid.npz  raw/test.npz   # index, num_atoms, charges [N,Kmax], positions [N,Kmax,3]
raw/train.smi  raw/valid.smi  raw/test.smi   # one SMILES per line (LLM branch)
dataset_stats.json                           # source, ratios, split sizes
```

Two split protocols are implemented: `random` (uniform, the protocol of the original
GeoBFN/EDM papers) and `scaffold` (Bemis-Murcko, i.e. generalisation to unseen
frameworks). `dataset_stats.json` is the source of the split-size table of report
§4.2. Options: `--ratios 0.8,0.1,0.1`, `--seed 0`, `--limit N`.

> Everything downstream reads these artefacts and none of the three methods writes to
> them — that is what makes the comparison fair.

`data.name=synthetic` skips the files and generates random molecules; that is what
`make smoke` uses.

---

## 5. Smoke test

```bash
make smoke
```

Runs the three pipelines end to end on synthetic data with a tiny model (about a
minute on CPU, no dataset): dataset → training → checkpoint → sampling → metrics. Do
it after changing the environment, before spending GPU hours.

---

## 6. The experiments, in the order of the report

### 6.1 Exploratory analysis — descriptors, PCA, KMeans, t-SNE

*Report: §4.2, Table 10 (feature maps) and Table 11 (exploratory results).*

```bash
# 12 RDKit descriptors per molecule -> dense array
python -m molgen.preprocess.features /workspace/data/qm9/raw/train.smi \
    --out /workspace/logs/analysis/features_train.npz

# PCA (explained variance, 90 % criterion), KMeans + silhouette, t-SNE
python -m molgen.preprocess.explore \
    --data /workspace/data/qm9/raw/train.smi \
    --out  /workspace/logs/analysis/explore --tsne

# the same on the held-out split, for the train/test overlap
python -m molgen.preprocess.explore \
    --data /workspace/data/qm9/raw/test.smi \
    --out  /workspace/logs/analysis/explore_test
```

Outputs, in each `--out` directory:

| File | Content | Report |
|---|---|---|
| `explore_summary.json` | molecules kept by the element filter, per-descriptor mean/std, PCs needed for 90 % variance, best $k$ and silhouette, cluster profiles, train/test overlap | Table 11 |
| `pca.npz` | PCA scores (first 10 components) and labels | manifold view, §5.4 |
| `tsne.npz` | t-SNE embedding and labels | visual check of the manifold |

Options: `--limit N` (subsample), `--k-max 10`, `--tsne`, `--seed 0`. `features` also
prints the descriptor table to stdout.

### 6.2 Cross-validation — selecting the hyper-parameters

*Report: §5.2, Eq. (46), Table 13 (grid) and Table 14 (result).*

```bash
bash scripts/cv/cv_geobfn.sh      # 5 candidates x 3 folds x 500 steps
bash scripts/cv/cv_edm.sh
bash scripts/cv/cv_llm.sh
```

Each script calls `python cv.py method=<m> +cv=<m>`: for every candidate in
`conf/cv/<method>.yaml` it trains fold `k = 0,1,2` for 500 steps on at most 4 000
training molecules and validates on 512 molecules of the held-out fold. Run folders
are `$LOG_DIR/<method>/cv/<candidate>/fold<k>/`, and the summary is
`$LOG_DIR/<method>/cv/cv_results.json`. The winner, in the exact form the training
scripts accept:

```bash
python tools/cv_best.py /workspace/logs/edm/cv/cv_results.json
# -> train.lr=0.0005
```

Overrides: `bash scripts/cv/cv_edm.sh cv.budget.max_steps=200`, single fold with
`cv.folds=[0]`. Cross-validation runs **one GPU per fold** (it is a 500-step run, so
parallelism buys little and complicates the comparison).

### 6.3 Training — the three methods, same budget

*Report: §5.1 (protocol), §5.3 (logging), Table 16 and Figure 6.*

```bash
bash scripts/train/train_geobfn.sh $(python tools/cv_best.py /workspace/logs/geobfn/cv/cv_results.json)
bash scripts/train/train_edm.sh    $(python tools/cv_best.py /workspace/logs/edm/cv/cv_results.json)
bash scripts/train/train_llm.sh    $(python tools/cv_best.py /workspace/logs/llm/cv/cv_results.json)
```

The scripts add a default only when the key is not already present, so both forms are
valid:

```bash
bash scripts/train/train_edm.sh                       # 10 000 steps, batch 64, no EMA
bash scripts/train/train_edm.sh train.max_steps=2000  # shorter, any override
STEPS=2000 GPUS=4 bash scripts/train/train_edm.sh     # through the environment
```

What a run writes into `$LOG_DIR/<method>/<timestamp>/`:

| File | Content |
|---|---|
| `config.json`, `.hydra/config.yaml` | the composed configuration, written before the first step |
| `train_log.jsonl` | every 50 steps: `loss`, `grad_norm`, `lr`, `seconds`, loss decomposition |
| `metrics.jsonl` | every 1 000 steps: validation loss on the held-out split |
| `last.pt` | weights, optimiser state, step, metrics, config (`ema` is `null`: EMA is off) |
| `final_metrics.json` | final validation loss, step, batch size, world size, fold |

`make train MODEL=geobfn GPUS=8` is the same thing through the Makefile. For several
nodes see §12.

#### Step intervals: when weights and analyses are written

Two step intervals decide what a run produces by itself, and both are ordinary
configuration keys (so they can also be overridden on the command line):

| Key | Default | Meaning |
|---|---|---|
| `train.ckpt_every` | 1000 | write `last.pt` every N optimisation steps |
| `train.ckpt_keep_all` | `false` | also keep `ckpt/step000500.pt`, `ckpt/step001000.pt`, … |
| `train.analysis_every` | 2000 | sample + score the current weights every N steps (`0` disables it) |
| `train.analysis_num` | 128 | molecules (or strings) per analysis |
| `train.analysis_steps` | 100 | reverse steps of the analysis sampler (geometric models) |
| `train.analysis_pca` | `true` | place the samples on the descriptor manifold of the training split |
| `train.analysis_pca_ref` | 8000 | molecules used to fit that manifold |
| `train.analysis_save_samples` | `true` | write `samples/analysis_<step>.xyz` (`.smi` for the LM) |

Every analysis appends one line to `<run>/eval_log.jsonl` with the five chemical
metrics *and* the position of the samples on the training manifold
(`pca_outside`, `pca_mahal_ratio`, `pca_nn_mean`), and **the last step always gets
one**, so a finished run is already evaluated — no separate pass is needed. The
manifold is fitted once from `raw/train.smi` and cached as `<run>/manifold.npz`, so the
metric is comparable across steps and across methods.

```bash
# default: weights every 1 000 steps, analysis every 2 000, analysis at the end
bash scripts/train/train_edm.sh

# weights every 500 steps (kept individually) and an analysis every 1 000
bash scripts/train/train_edm.sh train.ckpt_every=500 train.ckpt_keep_all=true \
     train.analysis_every=1000 train.analysis_num=256

# no periodic analysis at all, just training + the final one
bash scripts/train/train_edm.sh train.analysis_every=0
```

The samples of one analysis can be inspected with
`python tools/analyse_samples.py --method edm --input <run>/samples/analysis_002000.xyz`.

#### Evaluation only (skip training)

`train.skip_training=true` turns the same command into an evaluation-only run: the
model is rebuilt from `eval.ckpt`, sampled and scored exactly as the training loop
would have done it (same `analysis` code, same manifold, same log file), and nothing
is optimised:

```bash
python train.py method=edm train.skip_training=true \
    eval.ckpt=/workspace/logs/edm/2026-01-01_12-00-00/last.pt eval.num=1024
```

It writes `final_metrics.json` (with `skipped_training: true`) into
`train.out_dir` and appends to the checkpoint's `eval_log.jsonl`. The wrapper does the
same thing through `evaluate.py`:

```bash
bash scripts/eval/eval_edm.sh            # 1 024 samples
```

#### Live monitoring while a run is going

```bash
tail -f /workspace/logs/edm/<run>/train_log.jsonl     # loss every 50 steps
tail -f /workspace/logs/edm/<run>/metrics.jsonl       # validation loss every 1 000
tail -f /workspace/logs/edm/<run>/eval_log.jsonl      # chemistry + PCA every 2 000
```

### 6.4 Evaluation — metrics on the evaluation set

*Report: §5.4, Table 15.*

```bash
bash scripts/eval/eval_geobfn.sh      # newest run of that method
bash scripts/eval/eval_edm.sh
bash scripts/eval/eval_llm.sh
```

Each draws 1 024 molecules (or strings) with the **raw weights**
(`eval.use_ema=false`), infers bonds and computes validity, molecule stability, atom
stability, uniqueness and novelty with the branch's own metric module.
`NUM=512 bash scripts/eval/eval_edm.sh` draws fewer; a checkpoint can be given
positionally: `bash scripts/eval/eval_edm.sh /workspace/logs/edm/2026-.../last.pt`.

Outputs: `eval_*/metrics.json` next to the checkpoint, and `eval_*/samples.{xyz,smi}`
when `eval.out` is set.

### 6.5 Ten random samples and their analysis

*Report: §5.4, Figure 7 and the "what each model is expected to do well and badly"
paragraphs.*

```bash
bash scripts/samples/sample_geobfn.sh
bash scripts/samples/sample_edm.sh
bash scripts/samples/sample_llm.sh
```

Ten samples with `eval.seed=0`, written to `$LOG_DIR/<method>/<timestamp>/samples10/`:

```
samples10/samples.xyz | samples.smi   # the ten samples
samples10/metrics.json                # the five metrics on those ten
samples10/analysis.md                 # formula, SMILES, valid, stable, fragments, mean bond
```

The analysis can be re-run on any sample file:

```bash
python tools/analyse_samples.py --method geobfn --input samples10/samples.xyz
```

### 6.6 Loss against training step

*Report: Table 16 and Figure 6.*

```bash
python tools/plot_loss.py --log-root /workspace/logs \
    --out /workspace/logs/analysis/loss_curves.pdf
```

Reads the three `train_log.jsonl` files (newest run per method, or the directories
passed with `--runs a b c`) and draws the smoothed training loss with the 1 000-step
validation points on a log axis.

---

## 7. One command for everything: `run_experiments.sh`

```bash
bash run_experiments.sh                          # 8 GPUs, 10 000 steps, all stages
bash run_experiments.sh --dry-run                # print the commands, run nothing
bash run_experiments.sh --quick                  # ~2 min, synthetic data, no dataset
bash run_experiments.sh --steps 2000 --num 256   # shorter budget
make exp ARGS="--steps 2000"                     # same, from the host
```

| Stage | What runs | Report |
|---|---|---|
| 1 | descriptors + PCA/KMeans/t-SNE on the train and test splits | Table 11 |
| 2 | cross-validation of the three methods | Table 14 |
| 3 | training, 10 000 steps, same budget for the three | Table 16, Figure 6 |
| 4 | evaluation, 1 024 samples per method | Table 15 |
| 5 | ten random samples per method | Figure 7 |
| 6 | a metric summary printed as a table | — |

Before it starts it prints the plan (steps, batch, GPUs, global batch, seed, EMA off),
feeds each method's cross-validation winner into its training run through
`tools/cv_best.py`, and skips a stage gracefully when its inputs are missing
(`--no-analysis`, `--no-cv`, `--no-train`, `--no-eval`, `--no-samples`).

---

## 8. Report ↔ command ↔ output

| Report item | Produced by | Output |
|---|---|---|
| Table 10 — feature maps | design (no run) | — |
| Table 11 — exploratory results | §6.1 `features` + `explore` | `analysis/explore/explore_summary.json` |
| Table 13 — CV grid | the YAML files | `conf/cv/*.yaml` |
| Table 14 — CV result | §6.2 `scripts/cv/cv_*.sh` | `logs/<m>/cv/cv_results.json` |
| Table 16 — loss against step | §6.3 training | `logs/<m>/*/train_log.jsonl` |
| Figure 6 — loss curves | §6.6 `tools/plot_loss.py` | `analysis/loss_curves.pdf` |
| chemistry + PCA per step | §6.3 (in-loop analysis) | `logs/<m>/*/eval_log.jsonl` |
| Table 15 — generation metrics | §6.4 `scripts/eval/eval_*.sh` | `logs/<m>/*/eval_*/metrics.json` |
| Figure 7 — ten samples | §6.5 `scripts/samples/sample_*.sh` | `logs/<m>/*/samples10/` |
| Ablation table §5.5 | §6.3 with one override per row, e.g. `bash scripts/train/train_edm.sh diffusion.loss_weight=none` | one run per row |

---

## 9. Fairness of the comparison

The methods are only comparable if their optimisation budget is identical, so four
things are fixed, and the scripts make a drift visible instead of silent:

| Quantity | Value | Where it is enforced |
|---|---|---|
| per-GPU batch | 64 | `data.batch_size` in `conf/method/*.yaml`, added by `train_*.sh` |
| optimisation steps | 10 000 | `train.max_steps`, added by `train_*.sh` unless overridden |
| GPUs per method | identical for the three | `GPUS=8` for all three in `run_experiments.sh` |
| EMA | disabled | `train.ema_decay=0`, `eval.use_ema=false` |

The global batch is `batch_size × GPUs × nodes`; every run prints it at start-up and
records `batch_size`, `world_size`, `max_steps`, `fold` in `final_metrics.json`, so a
mismatch is visible afterwards:

```bash
grep -H . /workspace/logs/*/*/final_metrics.json
```

If memory forces a smaller batch, lower it **for all three methods** and note it in the
report.

---

## 10. Runtimes and resources

Rough numbers for one node with 8 × A100-40GB and QM9 (~108 k molecules), which is what
the protocol was sized for; the real timings are in `train_log.jsonl` (`seconds`) and
in `final_metrics.json`.

| Stage | Wall clock | Notes |
|---|---|---|
| dataset build | 5–15 min | once; RDKit conformer perception dominates |
| exploratory analysis | 10–20 min | PCA/KMeans/t-SNE on up to 100 k × 12 descriptors |
| cross-validation | 45 min – 2 h | 45 short runs, one GPU each |
| training, geobfn | 20–40 min | 10 000 steps, 512 molecules per step |
| training, edm | 15–30 min | same budget |
| training, llm | 5–15 min | 512 sequences of ≤ 64 tokens per step |
| evaluation, per method | 3–10 min | 1 024 samples; the geometric models do 1 000 network evaluations per sample |
| ten samples, per method | < 1 min | |

Memory: the geometric models fit a 16 GB card at batch 64; the language model needs far
less. If a step runs out of memory, use `--batch 32` for all three.

---

## 11. Configuration (Hydra)

Nothing is hard-coded in the training code:

```
conf/config.yaml          global: seed, ddp, eval block, run directory
conf/method/geobfn.yaml   one complete branch: data / model / train
conf/method/edm.yaml      the same + a `diffusion` block
conf/method/llm.yaml      the language-model branch
conf/cv/<method>.yaml     the cross-validation candidates
```

`method` is a Hydra config group, so switching branch is one key and cannot leave a
parameter of the previous branch behind:

```bash
python train.py                                      # GeoBFN, defaults from conf/
python train.py method=edm                           # switch branch
python train.py method=llm model.n_layer=8 train.lr=3e-4
python train.py -m method=llm model.n_layer=4,8      # sweep, one run per value
```

Runs land in `${LOG_DIR}/<method>/<timestamp>/` (Hydra's run directory). The launchers
pin `hydra.run.dir` so that all ranks agree on the folder and only rank 0 writes.
`hydra.job.chdir=false` keeps relative paths anchored at the project root.

---

## 12. Distributed training

Plain PyTorch DDP: `molgen/<method>/distributed.py` builds the process group from
`RANK` / `WORLD_SIZE` / `LOCAL_RANK`, each package wraps its loss in a small
`LossModule` so that DDP synchronises gradients inside `forward`, and the loader uses a
`DistributedSampler` re-seeded every step-budget pass.

| Scenario | Command |
|---|---|
| 1 GPU | `make train MODEL=edm GPUS=1` |
| 1 node, 8 GPUs | `make train MODEL=edm GPUS=8` |
| several nodes | `bash scripts/train_multi_node.sh` (below) |
| quick check | `make smoke` (CPU, synthetic) |

Multi-node: the same script on every node, same `STAMP`, `NNODES`, `MASTER_PORT` and
`MASTER_ADDR`, different `NODE_RANK`:

```bash
# node 0
METHOD=geobfn NNODES=2 NODE_RANK=0 MASTER_ADDR=10.0.0.1 GPUS=8 STAMP=run42 \
  bash scripts/train_multi_node.sh
# node 1
METHOD=geobfn NNODES=2 NODE_RANK=1 MASTER_ADDR=10.0.0.1 GPUS=8 STAMP=run42 \
  bash scripts/train_multi_node.sh
```

Notes: the global batch is `data.batch_size × world_size`; `MASTER_PORT` must be
reachable between nodes; for InfiniBand set `NCCL_SOCKET_IFNAME` / `NCCL_IB_HCA` in
`.env`; evaluation is single-process on purpose, so a sampled set is reproducible for
a given seed.

---

## 13. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `ERROR: .../raw/train.npz was not found` | run §4 once, or use `--quick` |
| `docker: permission denied` | add your user to the `docker` group, or use a Docker that allows it |
| `could not select device driver "" with capabilities: [[gpu]]` | install the NVIDIA Container Toolkit |
| training hangs at start (NCCL) | `/dev/shm` too small or `ipc` not shared — the service already sets 16 GB and `ipc: host`; check `docker compose config` |
| NCCL hangs on multi-node | check `MASTER_ADDR`, open `MASTER_PORT`, set `NCCL_SOCKET_IFNAME` |
| `CUDA out of memory` | `--batch 32` for the three methods (keep them equal) |
| files in `logs/` owned by root | set `USER_ID`/`GROUP_ID` in `.env` and `make build` |
| `python train.py` cannot import `molgen` | run from `/workspace/code` |
| `no checkpoint found for <method>` | training did not finish, or `LOG_DIR` differs between shells |
| `cv_results.json` missing | the CV stage was skipped; training then uses `conf/method/<m>.yaml` |
| metrics are `nan` | RDKit missing in the image, or a SMILES corpus it rejects |
| duplicate-override error | a key was passed twice; the scripts add defaults only when absent |

---

## 14. Repository layout

```
project/
├── Makefile                  init / in / exec / train / eval / cv / samples / exp / smoke / test
├── docker-compose.yaml       service `dev`: four mounts, 16 GB shm, ipc host, GPUs
├── docker.py                 the implementation behind the Makefile
├── docker/{Dockerfile,requirements.txt}
├── .env.example              template for the four host directories
├── run_experiments.sh        every experiment, end to end (§7)
├── cv.py                     cross-validation driver
├── train.py                  Hydra entry: dispatches on `method`
├── evaluate.py               Hydra entry: sampling + metrics
├── conf/
│   ├── config.yaml
│   ├── method/{geobfn,edm,llm}.yaml
│   └── cv/{geobfn,edm,llm}.yaml
├── molgen/
│   ├── geobfn/               bfn.py, egnn.py, data.py, config.py, metrics.py,
│   │                         distributed.py, ema.py, train_utils.py, train.py, evaluate.py
│   ├── edm/                  the same layout, diffusion.py instead of bfn.py
│   ├── llm/                  + tokenizer.py, dataset.py, sampler.py, conformer.py,
│   │                         chem.py, hf_adapter.py
│   └── preprocess/           build_dataset.py, features.py, explore.py, common.py
├── scripts/
│   ├── cv/ train/ eval/ samples/   one script per method and stage
│   └── train.sh  train_multi_node.sh  evaluate.sh  smoke.sh  run_all.sh
├── tools/
│   ├── plot_loss.py          loss-against-step figure
│   ├── analyse_samples.py    per-molecule table for the ten-sample analysis
│   └── cv_best.py            print the winning CV overrides
└── tests/                    GPU-free checks (math, tokeniser, docker CLI, refs)
```

---

## 15. Tests

```bash
make test
```

runs `tests/test_math.py` (the closed-form BFN/EDM identities), `tests/test_tokenizer.py`
(SMILES tokeniser round-trip) and `tests/test_docker_cli.py` (command construction of
the CLI, no Docker needed). Seconds, no GPU.
