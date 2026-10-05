#!/usr/bin/env bash
# SONUS accepted PF03 live launcher (portable Pi layout).
# Layout expected on the Pi:
#   ~/anc-validation/pi_candidate_v17/   base install: stream_raw + no_oversampling_onnx.tar.gz
#   ~/anc-validation/pi_pf03_clarity/    this overlay: run_pf03.sh + preset reference
#   ~/anc-validation/live-venv/          Python env (numpy, scipy, sounddevice, soundfile, psutil)
# Usage: ./run_pf03.sh [input_device] [output_device] [seconds]
# Defaults: input 1 (USB mic), output 0 (wired headphones), 30 s.
# Stop with: q + Enter. Mixer (Mic 5, AGC off) is set with the commands in
# docs/deployment.md before launch; this script does not change system mixer state.
set -euo pipefail
cd -- "$(dirname -- "$0")"
HERE="$PWD"
REPO_RUNTIME="$HERE/../runtime"
PRESET="$HERE/../configs/preset_pf03.json"
BASE="$HOME/anc-validation/pi_candidate_v17"
PYTHON="$HOME/anc-validation/live-venv/bin/python"
for need in "$BASE/stream_raw" "$BASE/no_oversampling_onnx.tar.gz" "$PRESET" "$REPO_RUNTIME/live_pf03_clarity.py"; do
  [ -e "$need" ] || { echo "missing: $need"; exit 1; }
done
"$PYTHON" - "$PRESET" "$BASE" <<'PY'
import sys, json, hashlib
from pathlib import Path
preset = json.loads(Path(sys.argv[1]).read_text())
base = Path(sys.argv[2])
for key, name in [("processor_sha256", "stream_raw"), ("bundle_sha256", "no_oversampling_onnx.tar.gz")]:
    actual = hashlib.sha256((base / name).read_bytes()).hexdigest()
    assert actual == preset[key], f"identity mismatch for {name}"
print("identity OK: processor + bundle match configs/preset_pf03.json")
PY
export PYTHONPATH="$REPO_RUNTIME${PYTHONPATH:+:$PYTHONPATH}"
exec "$PYTHON" "$REPO_RUNTIME/live_pf03_clarity.py" --native-processor \
  --processor "$BASE/stream_raw" --model-bundle "$BASE/no_oversampling_onnx.tar.gz" \
  --input-device "${1:-1}" --output-device "${2:-0}" --seconds "${3:-30}" \
  --mode on --gain-offset-db -4.5 --device-latency-ms 80 \
  --save-dir "$HERE/pf03_clarity_$(date +%Y%m%d_%H%M%S)"