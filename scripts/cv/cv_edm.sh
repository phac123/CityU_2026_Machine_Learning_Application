#!/usr/bin/env bash
# Cross-validation of the edm hyper-parameters: five candidates in
# conf/cv/edm.yaml, 3 folds, 500 steps per fold on 4000 training
# molecules, validated on 512. The winner is written to
# $LOG_DIR/edm/cv/cv_results.json.
#
#   bash scripts/cv/cv_edm.sh                       # the default budget
#   bash scripts/cv/cv_edm.sh cv.budget.max_steps=200
#   bash scripts/cv/cv_edm.sh cv.folds=[0]
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

# add a default only if the caller did not pass that key (Hydra rejects duplicates)
has_override() {
  local key="$1"; shift
  local a
  for a in "$@"; do
    [[ "$a" == "$key"=* ]] && return 0
  done
  return 1
}
EXTRA=()

python cv.py method=edm +cv=edm "$@"
