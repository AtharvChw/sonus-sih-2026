#!/usr/bin/env bash
# 06_train.sh — fine-tune both runs. RESUMABLE: re-running the same command continues
# from the latest checkpoint, it does not restart. Run in order; each session ends
# by downloading fresh logs (free-tier sessions can end - resume with this same script).
# Usage: bash 06_train.sh [impulse_weighted|no_oversampling|both]   (after: source config.env)
set -u -o pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "$HERE/config.env"
export PYTHONPATH="$DFN_SOURCE/DeepFilterNet${PYTHONPATH:+:$PYTHONPATH}"
cd "$WORK_ROOT"

WHICH="${1:-both}"
# SMOKE GATE: C6b must write successful base reports for the requested runs.
python3 "$HERE/verify_smoke_gate.py" "$WHICH" \
  || { echo "TRAIN BLOCKED: run 06b_smoke.py for each requested run first (C6b)"; exit 1; }
echo "smoke evidence OK - launching trainer"
FAILED=""
run_one() {
  local run="$1" cfg="$2"
  echo "===== train: $run ($cfg) ====="
  python3 -u "$HERE/run_dfn_train.py" "data/$cfg" data "runs/$run" 2>&1 \
    | tee -a "runs/$run/console.log"
  # pipefail: nonzero means the TRAINER failed (tee almost never does) - record, don't continue blind
  [ "${PIPESTATUS[0]}" -eq 0 ] || { echo "TRAIN FAILED for $run - see runs/$run/train.log"; FAILED=1; return 1; }
  echo "----- log check: $run -----"
  grep -E 'Running on device|\[valid\]|\[test\]|nan|inf' "runs/$run/train.log" | tail -n 20
  echo "best checkpoints:"
  ls runs/"$run"/checkpoints/ | grep -E 'model_.*\.ckpt\.best' || echo "(none yet - keep training)"
  bash "$HERE/09_snapshot.sh" "after-$run" || { FAILED=1; return 1; }
}
if [ "$WHICH" = impulse_weighted ] || [ "$WHICH" = both ]; then run_one impulse_weighted dataset.cfg; fi
if [ "$WHICH" = no_oversampling ] || [ "$WHICH" = both ]; then run_one no_oversampling dataset_no_oversampling.cfg; fi

echo "SELECTION RULE: best checkpoint chosen on VALIDATION loss only."
echo "Do NOT score the existing 280-mixture development comparison set for model selection - the sealed final holdout is scored once, later, outside training."
echo "PREREQUISITE: C6b SMOKE ALL PASS for both requested runs must precede full training."
[ -z "$FAILED" ] || { echo "06_train.sh FAILED for at least one run"; exit 1; }
echo "06_train.sh DONE"
