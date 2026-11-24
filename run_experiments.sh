#!/usr/bin/env bash
# =============================================================================
# run_experiments.sh -- train first, evaluate, then analyse the trained models.
#
# Order (the expensive, useful part comes first):
#   1. training        : the three methods, identical budget (10 000 steps, batch 64)
#   2. evaluation      : 1 024 samples per method -> metrics.json
#   3. test-set runs   : each trained model sampled again (256 samples) and compared
#                        with the *test* split (coverage on its descriptor manifold)
#   4. figures         : loss_curves.png, metrics.png, pca_test.png (+ PDF)
#   5. analysis        : descriptor PCA / KMeans / t-SNE of the dataset (after
#                        training; it used to run first and could crash the job)
#   6. ten samples     : the qualitative grid of the report (optional)
#
#   bash run_experiments.sh                  # 8 GPUs, 10 000 steps, stages 1-5
#   bash run_experiments.sh --quick          # ~3 min smoke run, synthetic data
#   bash run_experiments.sh --dry-run        # print every command, run nothing
#   bash run_experiments.sh --with-cv        # cross-validation before training
#
# Flags (environment variables work too):
#   --gpus N        GPUs per method        (default: all visible)
#   --steps N       optimisation steps     (default 10000)
#   --batch N       per-GPU batch          (default 64)
#   --seed N        seed                   (default 0)
#   --num N         samples per evaluation (default 1024)
#   --test-num N    samples for the test-set stage (default 256)
#   --with-cv       run the cross-validation first
#   --no-train --no-eval --no-test --no-analysis --no-samples --no-figures
#   --quick         tiny budgets on synthetic data (no dataset needed)
#   --dry-run       print the commands without executing them
#   -h, --help      this message
# =============================================================================
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

# OpenBLAS/OpenMP pools: on a 100+ core node the default overflows the wheel's thread
# metadata and KMeans/t-SNE die with "Bad memory unallocation" then a double free.
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-8}"

# ------------------------------------------------------------------ defaults --
GPUS="${GPUS:-}"
STEPS="${STEPS:-100000}"
BATCH="${BATCH:-64}"
SEED="${SEED:-0}"
EVAL_NUM="${EVAL_NUM:-1024}"
TEST_NUM="${TEST_NUM:-256}"
CV_STEPS="${CV_STEPS:-500}"
CV_LIMIT="${CV_LIMIT:-4000}"
DO_CV=0
DO_TRAIN=1
DO_EVAL=1
DO_TEST=1
DO_FIGURES=1
DO_ANALYSIS=1
DO_SAMPLES=0
QUICK=0
DRY_RUN=0
USE_SYNTHETIC=0

# cheapest first; geobfn is the most expensive
METHODS=(llm edm geobfn)

usage() { sed -n '3,32p' "$0" | sed -e 's/^# \{0,1\}//'; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpus)         GPUS="$2"; shift 2 ;;
    --steps)        STEPS="$2"; shift 2 ;;
    --batch)        BATCH="$2"; shift 2 ;;
    --seed)         SEED="$2"; shift 2 ;;
    --num)          EVAL_NUM="$2"; shift 2 ;;
    --test-num)     TEST_NUM="$2"; shift 2 ;;
    --with-cv)      DO_CV=1; shift ;;
    --no-cv)        DO_CV=0; shift ;;
    --no-train)     DO_TRAIN=0; shift ;;
    --no-eval)      DO_EVAL=0; shift ;;
    --no-test)      DO_TEST=0; shift ;;
    --no-figures)   DO_FIGURES=0; shift ;;
    --no-analysis)  DO_ANALYSIS=0; shift ;;
    --with-samples) DO_SAMPLES=1; shift ;;
    --no-samples)   DO_SAMPLES=0; shift ;;
    --quick)        QUICK=1; shift ;;
    --dry-run)      DRY_RUN=1; shift ;;
    -h|--help)      usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ "${QUICK}" == "1" ]]; then
  STEPS=200
  BATCH=8
  EVAL_NUM=8
  TEST_NUM=8
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

Build the artefacts once (the converter notices an existing .xyz directory and skips
the download):

  make fetch                       # or: python tools/fetch_qm9.py --out ${DATA_ROOT}

or, from a gdb9.sdf:

  make preprocess ARGS="--source /workspace/data/qm9/gdb9.sdf --split random"

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
FIG_DIR="${LOG_ROOT}/figures"
# every stage writes into one of these two directories; create them up front so a
# missing one cannot fail a stage after minutes of work
mkdir -p "${FIG_DIR}" "${LOG_ROOT}/analysis"

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
echo " Experiment plan"
echo "----------------------------------------------------------------------"
printf "  %-24s %s\n" "order"        "train -> eval -> test-set -> figures -> analysis"
printf "  %-24s %s\n" "methods"      "llm, edm, geobfn"
printf "  %-24s %s\n" "steps/method" "${STEPS}   (identical for the three)"
printf "  %-24s %s\n" "batch/GPU"    "${BATCH}   (identical for the three)"
printf "  %-24s %s\n" "GPUs/method"  "${GPUS_USED}"
printf "  %-24s %s\n" "global batch" "$((BATCH * GPUS_USED))"
printf "  %-24s %s\n" "seed"         "${SEED}"
printf "  %-24s %s\n" "EMA"          "disabled (raw weights are evaluated)"
printf "  %-24s %s\n" "eval samples" "${EVAL_NUM} per method, ${TEST_NUM} for the test set"
printf "  %-24s %s\n" "cross-val"    "$([[ ${DO_CV} == 1 ]] && echo "3 folds x ${CV_STEPS} steps" || echo skipped)"
printf "  %-24s %s\n" "data"         "${DATA_ROOT}${SYNTH_TAG}"
printf "  %-24s %s\n" "logs"         "${LOG_ROOT}"
printf "  %-24s %s\n" "figures"      "${FIG_DIR}"
echo "======================================================================"

# Pre-flight: the configs must compose, or every launch fails on all ranks.
if [[ "${DO_TRAIN}" == "1" ]]; then
  run python tools/check_configs.py
fi

# ------------------------------------------------- 1. training (first!) ------
if [[ "${DO_TRAIN}" == "1" ]]; then
  echo ""
  echo "### 1/6 training (same budget for the three methods)"

  if [[ "${DO_CV}" == "1" ]]; then
    for m in "${METHODS[@]}"; do
      run bash "scripts/cv/cv_${m}.sh" "cv.budget.max_steps=${CV_STEPS}" \
        "cv.budget.limit=${CV_LIMIT}" "${DATA_OVERRIDE[@]}"
    done
  fi

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
    fi
    # shellcheck disable=SC2086  # the overrides must be split into separate words
    run env GPUS="${GPUS}" STEPS="${STEPS}" BATCH="${BATCH}" SEED="${SEED}" \
      bash "scripts/train/train_${m}.sh" ${WIN} "${DATA_OVERRIDE[@]}"
  done
fi

# ------------------------------------------------------------- 2. evaluation --
if [[ "${DO_EVAL}" == "1" ]]; then
  echo ""
  echo "### 2/6 evaluation (${EVAL_NUM} samples per method)"
  for m in "${METHODS[@]}"; do
    run env NUM="${EVAL_NUM}" bash "scripts/eval/eval_${m}.sh" "${DATA_OVERRIDE[@]}"
  done
fi

# --------------------------------------------------------- 3. test-set runs ---
if [[ "${DO_TEST}" == "1" ]]; then
  echo ""
  echo "### 3/6 test-set stage (${TEST_NUM} samples per model, test manifold)"
  for m in "${METHODS[@]}"; do
    CKPT="$(ls -t "${LOG_ROOT}/${m}"/*/last.pt 2>/dev/null | head -n 1 || true)"
    if [[ -z "${CKPT}" ]]; then
      echo "[test] no checkpoint for ${m}, skipped"
      continue
    fi
    RUN_DIR="$(dirname "${CKPT}")"
    run python tools/test_generation.py --method "${m}" --ckpt "${CKPT}" \
      --num "${TEST_NUM}" --seed "${SEED}" \
      --out "${RUN_DIR}/test_eval" "${DATA_OVERRIDE[@]}"
  done
fi

# ---------------------------------------------------------------- 4. figures --
if [[ "${DO_FIGURES}" == "1" ]]; then
  echo ""
  echo "### 4/6 figures (loss, metrics, PCA of the test set)"
  run python tools/make_figures.py --log-root "${LOG_ROOT}" --out "${FIG_DIR}"
fi

# --------------------------------------------------------------- 5. analysis --
if [[ "${DO_ANALYSIS}" == "1" && "${USE_SYNTHETIC}" == "0" ]]; then
  echo ""
  echo "### 5/6 dataset analysis (descriptors, PCA, KMeans, t-SNE)"
  run python -m molgen.preprocess.features "${DATA_ROOT}/raw/train.smi" \
    --out "${LOG_ROOT}/analysis/features_train.npz"
  run python -m molgen.preprocess.explore \
    --data "${DATA_ROOT}/raw/test.smi" \
    --out "${LOG_ROOT}/analysis/explore_test" --tsne
fi

# ----------------------------------------------------------- 6. ten samples ---
if [[ "${DO_SAMPLES}" == "1" ]]; then
  echo ""
  echo "### 6/6 ten random samples per method"
  for m in "${METHODS[@]}"; do
    run bash "scripts/samples/sample_${m}.sh" "${DATA_OVERRIDE[@]}"
  done
fi

# ---------------------------------------------------------------- summary ----
cat <<EOF

======================================================================
 Done
----------------------------------------------------------------------
 checkpoints   ${LOG_ROOT}/<method>/<timestamp>/last.pt
 training logs ${LOG_ROOT}/<method>/<timestamp>/train_log.jsonl
 metrics       ${LOG_ROOT}/<method>/<timestamp>/eval_*/metrics.json
 test analysis ${LOG_ROOT}/<method>/<timestamp>/test_eval/test_analysis.json
 CV results    ${LOG_ROOT}/<method>/cv/cv_results.json
 dataset PCA   ${LOG_ROOT}/analysis/explore_test/
 figures       ${FIG_DIR}/{loss_curves,metrics,pca_test}.png (+ .pdf)

 The three images are written as PNG *and* PDF so they can go straight into
 the report:

   ${FIG_DIR}/loss_curves.png   loss against step, three models
   ${FIG_DIR}/metrics.png       five chemical metrics, grouped bars
   ${FIG_DIR}/pca_test.png      test-set manifold with the samples projected on it
======================================================================
EOF
