#!/usr/bin/env bash
# 07_export.sh — ONNX export + base-vs-best smoke + packaging. Run AFTER a new
# best checkpoint exists for a run (see 06_train.sh log check).
# Export contract (DeepFilterNet v0.5.6 df/scripts/export.py, verified from source):
#   <export_dir>/<run>_onnx.tar.gz containing EXACTLY enc.onnx, erb_dec.onnx,
#   df_dec.onnx, config.ini. Anything else fails the bundle check below.
# Smoke contract: the SAME fixed probe WAV through starting weights (-e 0) and
# fine-tuned best (-e best); both outputs + probe hash recorded. These run on the
# GPU machine only (needs torch + df) — nothing here is claimed from the laptop.
# Usage: bash 07_export.sh [impulse_weighted|no_oversampling|both]   (after: source config.env)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "$HERE/config.env"
export PYTHONPATH="$DFN_SOURCE/DeepFilterNet${PYTHONPATH:+:$PYTHONPATH}"
cd "$WORK_ROOT"

# Fixed probe: first sorted VoiceBank train WAV. Deterministic for a given extraction;
# its path + SHA-256 go into the smoke manifest so any rerun is comparable.
FIXED_PROBE="$(python3 -c 'from pathlib import Path; print(Path("lists/train_speech.txt").read_text().splitlines()[0])')"
[ -n "$FIXED_PROBE" ] || { echo "EXPORT FAILED: no voicebank WAV for smoke probe"; exit 1; }
python3 -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' \
  "$FIXED_PROBE" > "$WORK_ROOT/smoke_probe_sha256.txt"
echo "$FIXED_PROBE" > "$WORK_ROOT/smoke_probe_path.txt"
echo "smoke probe: $FIXED_PROBE"

export_one() {
  local run="$1"
  echo "===== export: $run ====="
  python3 "$DFN_SOURCE/DeepFilterNet/df/scripts/export.py" \
    -m "runs/$run" -e best "runs/$run/export" 2>&1 | tee "runs/$run/export.log"
  local bundle="runs/$run/export/${run}_onnx.tar.gz"
  [ -f "$bundle" ] || { echo "EXPORT FAILED for $run (no bundle) - save export.log"; return 1; }
  # bundle-content gate: exactly the 4 upstream members
  local members
  members=$(tar -tzf "$bundle" | grep -c -E '(enc\.onnx|erb_dec\.onnx|df_dec\.onnx|config\.ini)$')
  tar -tzf "$bundle"
  [ "$members" -eq 4 ] || { echo "EXPORT FAILED for $run: bundle holds $members/4 expected members"; return 1; }
  sha256sum "$bundle" > "runs/$run/export_sha256.txt"; cat "runs/$run/export_sha256.txt"

  mkdir -p "runs/$run/smoke_base" "runs/$run/smoke_best"
  python3 "$HERE/export_probe.py" --run "runs/$run" --probe "$FIXED_PROBE" || return 1
  sha256sum runs/"$run"/smoke_base/* > "runs/$run/smoke_base_sha256.txt"
  sha256sum runs/"$run"/smoke_best/* > "runs/$run/smoke_best_sha256.txt"
  echo "smoke base+best OK for $run"
}
WHICH="${1:-both}"
FAILED=""
if [ "$WHICH" = impulse_weighted ] || [ "$WHICH" = both ]; then export_one impulse_weighted || FAILED=1; fi
if [ "$WHICH" = no_oversampling ] || [ "$WHICH" = both ]; then export_one no_oversampling || FAILED=1; fi
[ -z "$FAILED" ] || { echo "07_export.sh FAILED for at least one run - fix export.log first"; exit 1; }

python3 "$HERE/08_results_manifest.py" || { echo "07_export.sh FAILED: results manifest incomplete"; exit 1; }
tar --exclude='*/smoke_base' --exclude='*/smoke_best' -czf "$WORK_ROOT/ps26052-gpu-handoff.tar.gz" \
  -C "$WORK_ROOT" runs/impulse_weighted runs/no_oversampling lists \
  data/dataset.cfg data/dataset_no_oversampling.cfg \
  source-archive-sha256.txt environment-pip-freeze.txt results_manifest.json \
  smoke_probe_path.txt smoke_probe_sha256.txt
sha256sum "$WORK_ROOT/ps26052-gpu-handoff.tar.gz" > "$WORK_ROOT/ps26052-gpu-handoff.sha256" \
  || { echo "07_export.sh FAILED: packaging"; exit 1; }
cat "$WORK_ROOT/ps26052-gpu-handoff.sha256"
echo "PACKAGE OK - download ps26052-gpu-handoff.tar.gz + .sha256 from /kaggle/working (see checklist step 8)."
echo "Also download each runs/<run>/smoke_{base,best}/ folder (kept OUT of the tarball - audio stays reviewable, not redistributed)."
