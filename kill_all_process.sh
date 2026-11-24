#!/usr/bin/env bash
# =============================================================================
# kill_gpu_train.sh —— 清理 GPU 上的训练进程（默认只清自己，默认只预演不动手）
#
# 用法:
#   ./kill_gpu_train.sh            预演: 列出将被清理的进程, 不发送任何信号
#   ./kill_gpu_train.sh -y         执行: 先 SIGTERM, 等 10s, 残留的再 SIGKILL
#   ./kill_gpu_train.sh -y -t 30   SIGTERM 后等 30s 再补刀
#   ./kill_gpu_train.sh -y -p      连普通 python 进程一起清 (更激进, 慎用)
#   ./kill_gpu_train.sh -y -u bob  清理指定用户 (仅 root 可用)
#   ./kill_gpu_train.sh -h         帮助
#
# 匹配范围: torchrun / train.py / torch.distributed.run / run_experiments.sh
#           + nvidia-smi 里实际占着显存、且属于目标用户的进程
# 说明: run_experiments.sh 会一起被杀, 防止它继续往下跑 eval/test/figures 阶段
# =============================================================================
set -uo pipefail

ME="$(id -un)"
SELF=$$ ; MY_PPID=$PPID
DRY_RUN=1 ; GRACE=10 ; WIDE=0 ; TARGET_USER=""

while (($#)); do
  case "$1" in
    -y|--yes)     DRY_RUN=0 ;;
    -t|--timeout) GRACE="${2:?缺少数值}"; shift ;;
    -p|--python)  WIDE=1 ;;
    -u|--user)    TARGET_USER="${2:?缺少用户名}"; shift ;;
    -h|--help)    sed -n '2,17p' "$0"; exit 0 ;;
    *) echo "未知参数: $1（用 -h 查看用法）"; exit 2 ;;
  esac
  shift
done

if [[ -n "$TARGET_USER" && "$EUID" -ne 0 ]]; then
  echo "只有 root 才能用 -u 指定其他用户"; exit 2
fi
[[ -z "$TARGET_USER" ]] && TARGET_USER="$ME"

TRAIN_RE='torchrun|train\.py|torch/distributed/run\.py|run_experiments\.sh'

declare -A GPU_MEM=() TARGET=()

# --- 1) nvidia-smi 里占显存的进程 --------------------------------------------
if command -v nvidia-smi >/dev/null 2>&1; then
  while IFS=, read -r pid mem _; do
    pid="${pid// /}"; mem="${mem// /}"
    [[ "$pid" =~ ^[0-9]+$ ]] && GPU_MEM[$pid]="$mem"
  done < <(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null)
else
  echo "警告: 找不到 nvidia-smi，仅按进程名匹配"
fi

# --- 2) 进程表: 目标用户 + 命令行过滤 ----------------------------------------
while read -r pid user args; do
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  [[ "$user" == "$TARGET_USER" ]] || continue
  [[ "$pid" == "$SELF" || "$pid" == "$MY_PPID" ]] && continue
  [[ "$args" == *nvidia-smi* ]] && continue
  if [[ $WIDE -eq 1 ]]; then
    [[ "$args" =~ (python|torchrun) ]] || continue
  else
    [[ "$args" =~ $TRAIN_RE ]] || continue
  fi
  TARGET[$pid]="$args"
done < <(ps -eo pid=,user=,args=)

# --- 3) 占着显存但命令行不匹配的进程也纳入 ----------------------------------
for pid in "${!GPU_MEM[@]}"; do
  [[ "$pid" == "$SELF" || "$pid" == "$MY_PPID" ]] && continue
  u="$(ps -o user= -p "$pid" 2>/dev/null | tr -d ' ')"
  [[ "$u" == "$TARGET_USER" ]] || continue
  if [[ -z "${TARGET[$pid]:-}" ]]; then
    TARGET[$pid]="$(ps -o args= -p "$pid" 2>/dev/null | head -c 100)"
  fi
done

if ((${#TARGET[@]} == 0)); then
  echo "没有找到属于 $TARGET_USER 的 GPU/训练进程，无需清理。"
  exit 0
fi

echo "目标用户: $TARGET_USER   进程数: ${#TARGET[@]}"
printf '%-8s %-10s %s\n' "PID" "GPU显存" "命令"
printf '%-8s %-10s %s\n' "--------" "----------" "----------------------------------------------"
for pid in $(printf '%s\n' "${!TARGET[@]}" | sort -n); do
  cmd="${TARGET[$pid]}"
  printf '%-8s %-10s %s\n' "$pid" "${GPU_MEM[$pid]:-—}" "${cmd:0:110}"
done

if [[ $DRY_RUN -eq 1 ]]; then
  echo
  echo "以上为预演，未发送任何信号。确认无误后执行:  $0 -y"
  exit 0
fi

# --- 4) 先礼后兵 -------------------------------------------------------------
echo
echo "发送 SIGTERM ..."
kill -TERM "${!TARGET[@]}" 2>/dev/null

alive=0
for ((i = 0; i < GRACE; i++)); do
  alive=0
  for pid in "${!TARGET[@]}"; do
    kill -0 "$pid" 2>/dev/null && alive=$((alive + 1))
  done
  ((alive == 0)) && break
  sleep 1
done

if ((alive > 0)); then
  echo "仍有 $alive 个进程未响应 SIGTERM，发送 SIGKILL ..."
  kill -KILL "${!TARGET[@]}" 2>/dev/null
  sleep 1
fi

# --- 5) 结果核对 -------------------------------------------------------------
echo
echo "---- 剩余 GPU 计算进程 ----"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv 2>/dev/null
else
  echo "(无 nvidia-smi)"
fi

unreaped=0
for pid in "${!TARGET[@]}"; do
  if kill -0 "$pid" 2>/dev/null; then
    st="$(ps -o stat= -p "$pid" 2>/dev/null | tr -d ' ')"
    case "$st" in
      D*) echo "PID $pid 卡在不可中断睡眠(D 状态)，信号杀不掉：需要重启容器/宿主机，或 nvidia-smi --gpu-reset" ;;
      *)  echo "PID $pid 仍存活 (stat=$st)" ;;
    esac
    unreaped=$((unreaped + 1))
  fi
done

((unreaped == 0)) && echo "清理完成，无残留进程。"
exit 0

