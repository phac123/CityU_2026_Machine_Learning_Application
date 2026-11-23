#!/usr/bin/env bash
# =============================================================================
# run_experiments.sh -- every experiment of the report, end to end.
#
# Stages (all on by default):
#   1. exploratory analysis : RDKit descriptors, PCA, KMeans + silhouette, t-SNE
#                             on the train and test splits (§4.2 of the report)
#   2. cross-validation     : 5 candidates x 3 folds per method, 500 steps a fold,
#                             validation loss selects the hyper-parameters (§5.2)
#   3. training             : the three methods with the SAME budget -- 10 000
#                             optimisation steps, batch 64 per GPU, no EMA (§5.1)
#   4. evaluation           : 1 024 samples per method, all five metrics (§5.4)
#   5. ten samples          : 10 random samples per method + analysis.md (§5.4)
#   6. figures              : loss against step (Figure 6)
#
# Everything runs on all visible GPUs (8 on the target server). The fairness of the
# comparison is enforced by construction: the three trainings receive the same
# STEPS, BATCH, GPUS and SEED, and the script prints the plan before it starts.
#
#   bash run_experiments.sh                 # the real thing, 8 GPUs, 10 000 steps
#   bash run_experiments.sh --steps 2000    # shorter budget, same protocol
#   bash run_experiments.sh --dry-run       # print the commands, run nothing
#   bash run_experiments.sh --quick         # ~2 min smoke test on synthetic data
#
# Flags (environment variables work too):
#   --gpus N           GPUs per method        (default: all visible GPUs)
#   --steps N          optimisation steps     (default 10000)
#   --batch N          per-GPU batch          (default 64, identical everywhere)
#   --seed N           seed                   (default 0)
#   --num N            molecules per evaluation (default 1024)
#   --quick            tiny budgets, synthetic data, no dataset needed
#   --no-analysis      skip stage 1
#   --no-cv            skip stage 2 (train with the defaults in conf/)
#   --no-train         skip stage 3
#   --no-eval          skip stage 4
#   --no-samples       skip stage 5
#   --dry-run          print the commands without executing them
#   -h, --help         this message
# =============================================================================
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

GH="$(git rev-parse --short HEAD 2>/dev/null || echo nogit)"   # informational only

# ------------------------------------------------------------------ defaults --
GPUS="${GPUS:-}"
STEPS="${STEPS:-10000}"
BATCH="${BATCH:-64}"
SEED="${SEED:-0}"
EVAL_NUM="${EVAL_NUM:-1024}"
CV_STEPS="${CV_STEPS:-500}"
CV_LIMIT="${CV_LIMIT:-4000}"
DO_ANALYSIS=1
DO_CV=1
DO_TRAIN=1
DO_EVAL=1
DO_SAMPLES=1
QUICK=0
DRY_RUN=0
USE_SYNTHETIC=0

# the order matters: the language model is the cheapest, geobfn the most expensive
METHODS=(llm edm geobfn)

usage() { sed -n '3,40p' "$0" | sed -e 's/^# \{0,1\}//'; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpus)        GPUS="$2"; shift 2 ;;
    --steps)       STEPS="$2"; shift 2 ;;
    --batch)       BATCH="$2"; shift 2 ;;
    --seed)        SEED="$2"; shift 2 ;;
    --num)         EVAL_NUM="$2"; shift 2 ;;
    --quick)       QUICK=1; shift ;;
    --no-analysis) DO_ANALYSIS=0; shift ;;
    --no-cv)       DO_CV=0; shift ;;
    --no-train)    DO_TRAIN=0; shift ;;
    --no-eval)     DO_EVAL=0; shift ;;
    --no-samples)  DO_SAMPLES=0; shift ;;
    --dry-run)     DRY_RUN=1; shift ;;
    -h|--help)     usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ "${QUICK}" == "1" ]]; then
  STEPS=200
  BATCH=8
  EVAL_NUM=8
  CV_STEPS=50
  CV_LIMIT=512
  DO_ANALYSIS=0
fi

DATA_ROOT="${DATA_DIR:-./data}/qm9"
LOG_ROOT="${LOG_DIR:-./logs}"

run() {
  echo ""
  echo "+ $*"
  if [[ "${DRY_RUN}" == "1" ]]; then
    return 0
  fi
  "$@"
}

# ------------------------------------------------------------------- checks --
if [[ ! -f "${DATA_ROOT}/raw/train.npz" ]]; then
  if [[ "${QUICK}" == "1" ]]; then
    USE_SYNTHETIC=1
    echo "[plan] no QM9 artefacts found -> quick mode uses synthetic data"
  else
    cat >&2 <<EOF
ERROR: ${DATA_ROOT}/raw/train.npz was not found.

Build the shared artefacts once (the only step that needs the raw QM9 download):

  make preprocess ARGS="--source /workspace/data/qm9/gdb9.sdf --split random"

or, equivalently, inside the container:

  python -m molgen.preprocess.build_dataset \\
      --source /workspace/data/qm9/gdb9.sdf --out ${DATA_ROOT}

Then re-run: bash run_experiments.sh
EOF
    exit 1
  fi
fi

SYNTH_TAG=""
if [[ "${USE_SYNTHETIC}" == "1" ]]; then
  DATA_OVERRIDE=(data.name=synthetic)
  LOG_ROOT="${LOG_ROOT}/quick"
  SYNTH_TAG=" (synthetic)"
else
  DATA_OVERRIDE=()
fi
mkdir -p "${LOG_ROOT}/analysis"

if command -v nvidia-smi >/dev/null 2>&1; then
  VISIBLE_GPUS="$(nvidia-smi -L | wc -l)"
else
  VISIBLE_GPUS=0
fi
GPUS_USED="${GPUS:-${VISIBLE_GPUS}}"
if [[ "${GPUS_USED}" == "0" ]]; then
  echo "[warn] no GPU detected with nvidia-smi: pass --gpus N explicitly" >&2
  GPUS_USED=1
fi

echo "======================================================================"
echo " Experiment plan (run ${GH})"
echo "----------------------------------------------------------------------"
printf "  %-24s %s\n" "methods"          "llm, edm, geobfn"
printf "  %-24s %s\n" "steps per method" "${STEPS}   (identical for the three)"
printf "  %-24s %s\n" "batch per GPU"    "${BATCH}   (identical for the three)"
printf "  %-24s %s\n" "GPUs per method"  "${GPUS_USED}   (identical for the three)"
printf "  %-24s %s\n" "global batch"     "$((BATCH * GPUS_USED))"
printf "  %-24s %s\n" "seed"             "${SEED}"
printf "  %-24s %s\n" "EMA"              "disabled (raw weights are evaluated)"
printf "  %-24s %s\n" "cross-validation" "3 folds x ${CV_STEPS} steps, ${CV_LIMIT} train molecules"
printf "  %-24s %s\n" "evaluation"       "${EVAL_NUM} samples per method"
printf "  %-24s %s\n" "data"             "${DATA_ROOT}${SYNTH_TAG}"
printf "  %-24s %s\n" "logs"             "${LOG_ROOT}"
echo "======================================================================"

# ------------------------------------------------- 1. exploratory analysis ----
if [[ "${DO_ANALYSIS}" == "1" && "${USE_SYNTHETIC}" == "0" ]]; then
  echo ""
  echo "### 1/6 exploratory analysis (descriptors, PCA, KMeans, t-SNE)"
  run python -m molgen.preprocess.features "${DATA_ROOT}/raw/train.smi" \
    --out "${LOG_ROOT}/analysis/features_train.npz"
  run python -m molgen.preprocess.explore \
    --data "${DATA_ROOT}/raw/train.smi" \
    --out "${LOG_ROOT}/analysis/explore" --tsne
  run python -m molgen.preprocess.explore \
    --data "${DATA_ROOT}/raw/test.smi" \
    --out "${LOG_ROOT}/analysis/explore_test"
fi

# ------------------------------------------------------ 2. cross-validation ---
if [[ "${DO_CV}" == "1" ]]; then
  echo ""
  echo "### 2/6 cross-validation (hyper-parameter selection)"
  for m in "${METHODS[@]}"; do
    run bash "scripts/cv/cv_${m}.sh" \
      "cv.budget.max_steps=${CV_STEPS}" \
      "cv.budget.limit=${CV_LIMIT}" \
      "${DATA_OVERRIDE[@]}"
  done
fi

# ------------------------------------------------------------- 3. training ---
if [[ "${DO_TRAIN}" == "1" ]]; then
  echo ""
  echo "### 3/6 training (same budget for the three methods)"
  for m in "${METHODS[@]}"; do
    WIN=""
    CV_JSON="${LOG_ROOT}/${m}/cv/cv_results.json"
    if [[ -f "${CV_JSON}" ]]; then
      if WIN="$(python tools/cv_best.py "${CV_JSON}" 2>/dev/null)"; then
        echo "[${m}] cross-validation winner: ${WIN:-defaults}"
      else
        WIN=""
        echo "[${m}] could not read ${CV_JSON}; training with the defaults"
      fi
    else
      echo "[${m}] no cv_results.json -> training with the defaults of conf/"
    fi
    # shellcheck disable=SC2086  # the overrides must be split into separate words
    run env GPUS="${GPUS}" STEPS="${STEPS}" BATCH="${BATCH}" SEED="${SEED}" \
      bash "scripts/train/train_${m}.sh" ${WIN} "${DATA_OVERRIDE[@]}"
  done

  echo ""
  echo "### loss curves (report Figure 6)"
  run python tools/plot_loss.py --log-root "${LOG_ROOT}" \
    --out "${LOG_ROOT}/analysis/loss_curves.pdf"
fi

# ----------------------------------------------------------- 4. evaluation ---
if [[ "${DO_EVAL}" == "1" ]]; then
  echo ""
  echo "### 4/6 evaluation (${EVAL_NUM} samples per method)"
  for m in "${METHODS[@]}"; do
    run env NUM="${EVAL_NUM}" bash "scripts/eval/eval_${m}.sh" "${DATA_OVERRIDE[@]}"
  done
fi

# -------------------------------------------------------- 5. ten samples -----
if [[ "${DO_SAMPLES}" == "1" ]]; then
  echo ""
  echo "### 5/6 ten random samples per method (report Figure 7)"
  for m in "${METHODS[@]}"; do
    run bash "scripts/samples/sample_${m}.sh" "${DATA_OVERRIDE[@]}"
  done
fi

# ---------------------------------------------------------- 6. collect -------
if [[ "${DO_EVAL}" == "1" && "${DRY_RUN}" != "1" ]]; then
  echo ""
  echo "### 6/6 metric summary over the three methods"
  found=0
  for m in "${METHODS[@]}"; do
    latest="$(ls -t "${LOG_ROOT}/${m}"/*/eval_*/metrics.json 2>/dev/null | head -n 1 || true)"
    if [[ -n "${latest}" ]]; then
      echo "--- ${m}: ${latest}"
      python - "$latest" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))
keys = ["validity", "mol_stable", "atm_stable", "uniqueness", "novelty", "n_valid", "eval_seconds"]
print("   " + "  ".join(f"{k}={m.get(k, float('nan')):.4f}" if isinstance(m.get(k), (int, float)) else f"{k}={m.get(k)}" for k in keys if k in m))
PY
      found=1
    fi
  done
  if [[ "${found}" == "0" ]]; then
    echo "[summary] no metrics.json found yet (evaluation was skipped or failed)"
  fi
fi

# ---------------------------------------------------------------- summary ----
cat <<EOF

======================================================================
 Done
----------------------------------------------------------------------
 checkpoints   ${LOG_ROOT}/<method>/<timestamp>/last.pt
 training logs ${LOG_ROOT}/<method>/<timestamp>/train_log.jsonl
 metrics       ${LOG_ROOT}/<method>/<timestamp>/metrics.jsonl
 CV results    ${LOG_ROOT}/<method>/cv/cv_results.json
 analysis      ${LOG_ROOT}/analysis/{features_train.npz,explore/,explore_test/}
 loss figure   ${LOG_ROOT}/analysis/loss_curves.pdf
 ten samples   ${LOG_ROOT}/<method>/<timestamp>/samples10/

 Everything above maps to a table or a figure of the report:
   Table 10  feature maps            <- stage 1
   Table 11  exploratory results     <- stage 1 (explore_summary.json)
   Table 14  cross-validation        <- stage 2 (cv_results.json)
   Table 16  loss against step       <- stage 3 (train_log.jsonl)
   Figure 6  loss curves             <- stage 3 (tools/plot_loss.py)
   Table 15  generation metrics      <- stage 4 (metrics.json)
   Figure 7  ten samples             <- stage 5 (samples10/)
======================================================================
EOF
