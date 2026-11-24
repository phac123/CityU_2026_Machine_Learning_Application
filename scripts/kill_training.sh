#!/usr/bin/env bash
# =============================================================================
# kill_training.sh -- stop every training / evaluation process of this project.
#
#   bash scripts/kill_training.sh              # list what it found, then ask
#   bash scripts/kill_training.sh --yes        # stop them, no question
#   bash scripts/kill_training.sh -n           # only list, kill nothing
#   bash scripts/kill_training.sh -9           # SIGKILL at once (stuck ranks)
#   bash scripts/kill_training.sh -y --gpu     # also show the GPU memory left over
#
# It stops the launchers (run_experiments.sh, scripts/*.sh, torchrun), the entry
# points (train.py, evaluate.py, cv.py) and the helper tools (test_generation.py,
# make_figures.py, plot_loss.py): SIGTERM first so the loops can flush their logs,
# then SIGKILL whatever is still alive after --timeout seconds (default 10).
# Checkpoints are written every `train.ckpt_every` steps, so an interrupted run loses
# at most that many steps.
#
# Run it where the processes are visible: inside the container that started them, or
# on the host (Linux lists container processes in its own ps).  A *second* container
# only sees them because docker-compose.yaml starts the service with `pid: host`.
# =============================================================================
set -uo pipefail

PATTERN='torchrun|train\.py|evaluate\.py|cv\.py|test_generation\.py|make_figures\.py|plot_loss\.py|run_experiments\.sh|scripts/(train|eval|cv|samples)|kill_training\.sh'
SELF='kill_training\.sh'
TIMEOUT=10
DO_KILL=1
FORCE=0
ASSUME_YES=0
SHOW_GPU=0

usage() { sed -n '3,22p' "$0" | sed -e 's/^# \{0,1\}//'; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    -n|--dry-run) DO_KILL=0; shift ;;
    -9|--force)   FORCE=1; ASSUME_YES=1; shift ;;
    -y|--yes)     ASSUME_YES=1; shift ;;
    -t|--timeout) TIMEOUT="$2"; shift 2 ;;
    --gpu)        SHOW_GPU=1; shift ;;
    -h|--help)    usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

if ! command -v ps >/dev/null 2>&1; then
  echo "[kill] 'ps' is not available in this environment" >&2
  exit 1
fi

# ------------------------------------------------------------------- collect --
# `etimes` is nice to have but not universal; fall back to a format without it.
if ps -eo pid=,ppid=,stat=,etimes=,args= >/dev/null 2>&1; then
  PS_ARGS=(-eo "pid=,ppid=,stat=,etimes=,args=")
  WITH_ETIMES=1
else
  PS_ARGS=(-eo "pid=,ppid=,stat=,args=")
  WITH_ETIMES=0
fi

pids=()
descriptions=()
while IFS=$'\t' read -r pid ppid stat etimes cmd; do
  [[ -z "${pid:-}" ]] && continue
  [[ "$pid" == "$$" ]] && continue
  [[ "$pid" == "${PPID:-}" ]] && continue
  [[ "$pid" == "1" ]] && continue
  [[ "$stat" == *Z* ]] && continue                     # zombie: nothing to kill
  [[ "$cmd" == *"$SELF"* ]] && continue                # never kill the helper itself
  [[ "$cmd" == *"ps "* && "$cmd" == *"pid="* ]] && continue
  if [[ "$cmd" =~ $PATTERN ]]; then
    pids+=("$pid")
    descriptions+=("$(printf '%7s %7s %9s  %s' "$pid" "$ppid" "$etimes" "${cmd:0:110}")")
  fi
done < <(
  if [[ "$WITH_ETIMES" == "1" ]]; then
    ps "${PS_ARGS[@]}" 2>/dev/null | while read -r pid ppid stat etimes cmd; do
      printf '%s\t%s\t%s\t%s\t%s\n' "$pid" "$ppid" "$stat" "$etimes" "${cmd//$'\t'/ }"
    done
  else
    ps "${PS_ARGS[@]}" 2>/dev/null | while read -r pid ppid stat cmd; do
      printf '%s\t%s\t%s\t%s\t%s\n' "$pid" "$ppid" "$stat" "-" "${cmd//$'\t'/ }"
    done
  fi
)

echo "======================================================================"
if [[ ${#pids[@]} -eq 0 ]]; then
  echo " No training process of this project is running."
  echo "======================================================================"
  if [[ "$SHOW_GPU" == "1" ]] && command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv || true
  fi
  exit 0
fi

echo " ${#pids[@]} process(es) found:"
echo "----------------------------------------------------------------------"
printf '      PID    PPID   ELAPSED  COMMAND\n'
for line in "${descriptions[@]}"; do
  echo "  $line"
done
echo "======================================================================"

if [[ "$DO_KILL" != "1" ]]; then
  echo "[kill] --dry-run: nothing was stopped"
  exit 0
fi

if [[ "$ASSUME_YES" != "1" ]]; then
  if [[ -t 0 ]]; then
    read -r -p "Stop these ${#pids[@]} process(es)? [y/N] " answer
    case "${answer:-n}" in
      [yY]*) ;;
      *) echo "[kill] aborted"; exit 0 ;;
    esac
  else
    echo "[kill] no terminal to ask on; re-run with --yes" >&2
    exit 1
  fi
fi

# ---------------------------------------------------------------------- kill --
if [[ "$FORCE" == "1" ]]; then
  echo "[kill] SIGKILL ${pids[*]}"
  kill -9 "${pids[@]}" 2>/dev/null || true
else
  echo "[kill] SIGTERM ${pids[*]}"
  kill -TERM "${pids[@]}" 2>/dev/null || true
  alive=()
  for ((i = 0; i < TIMEOUT; i++)); do
    alive=()
    for pid in "${pids[@]}"; do
      kill -0 "$pid" 2>/dev/null && alive+=("$pid")
    done
    [[ ${#alive[@]} -eq 0 ]] && break
    sleep 1
  done
  if [[ ${#alive[@]} -gt 0 ]]; then
    echo "[kill] still alive after ${TIMEOUT}s -> SIGKILL: ${alive[*]}"
    kill -9 "${alive[@]}" 2>/dev/null || true
    sleep 1
  fi
fi

# -------------------------------------------------------------------- report --
remaining=()
for pid in "${pids[@]}"; do
  kill -0 "$pid" 2>/dev/null && remaining+=("$pid")
done
if [[ ${#remaining[@]} -eq 0 ]]; then
  echo "[kill] all stopped."
else
  echo "[kill] still running: ${remaining[*]}" >&2
fi

if [[ "$SHOW_GPU" == "1" ]] && command -v nvidia-smi >/dev/null 2>&1; then
  echo ""
  echo "[kill] GPU state now:"
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv || true
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv || true
fi

echo "[kill] the checkpoints written before the stop are in \$LOG_DIR/<method>/<timestamp>/"
exit 0

