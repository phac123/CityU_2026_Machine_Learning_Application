# Launch scripts

One script per method and per stage. Every script accepts further Hydra overrides and
adds its own defaults only when the corresponding key is not already present, so
`bash scripts/train/train_edm.sh train.lr=1e-4` works without a duplicate-override
error.

| Stage | geobfn | edm | llm |
|---|---|---|---|
| cross-validation | `cv/cv_geobfn.sh` | `cv/cv_edm.sh` | `cv/cv_llm.sh` |
| formal training (10 000 steps) | `train/train_geobfn.sh` | `train/train_edm.sh` | `train/train_llm.sh` |
| evaluation (metrics) | `eval/eval_geobfn.sh` | `eval/eval_edm.sh` | `eval/eval_llm.sh` |
| 10 random samples + analysis | `samples/sample_geobfn.sh` | `samples/sample_edm.sh` | `samples/sample_llm.sh` |

A complete experiment, per method:

```bash
bash scripts/cv/cv_geobfn.sh                    # -> $LOG_DIR/geobfn/cv/cv_results.json
bash scripts/train/train_geobfn.sh $(python tools/cv_best.py $LOG_DIR/geobfn/cv/cv_results.json)
bash scripts/eval/eval_geobfn.sh                # -> metrics.json next to the checkpoint
bash scripts/samples/sample_geobfn.sh           # -> samples10/analysis.md
```

`run_experiments.sh` (in the project root) runs the whole chain for the three methods
with the same budget; `tools/plot_loss.py` draws the loss figure and
`tools/analyse_samples.py` produces the per-molecule tables.

Fairness of the comparison: the same per-GPU batch (64), the same number of
optimisation steps (10 000), the same number of GPUs (hence the same global batch) and
no EMA.
