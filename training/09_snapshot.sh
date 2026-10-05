#!/usr/bin/env bash
# 09_snapshot.sh — pack EVERYTHING needed to resume after a FRESH Kaggle session.
# Run at the END of every session, then download the snapshot tarball + .sha256.
# Contents: full run dirs - checkpoints/* (upstream train.py writes model_<e>.ckpt AND
# opt_<e>.ckpt per epoch plus .best files: weights + optimizer state, verified in
# upstream checkpoint.py/train.py source; 06_train.sh resumes from the latest one),
# config.ini, logs - plus lists/ + dataset cfgs, freeze + source hashes, results manifest.
# Proven by mock (test_resume_mock.sh: 21 files byte-identical).
# Excluded ON PURPOSE: raw/ archives (re-upload), data/*.hdf5 (rebuilt by 04_make_hdf5.sh
# after restore — deterministic from lists/), and paths.env (re-discovered by
# 00_locate_inputs.sh after re-upload, since paths may change between sessions).
# Usage: bash 09_snapshot.sh [tag]   (after: source config.env)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "$HERE/config.env"
cd "$WORK_ROOT"

TAG="${1:-$(date -u +%Y%m%dT%H%M%SZ)}"
SNAP="ps26052-snapshot-${TAG}.tar.gz"
[ -d runs/impulse_weighted ] || { echo "SNAPSHOT FAILED: no runs/ yet (nothing to save)"; exit 1; }
members=()
for m in runs/impulse_weighted runs/no_oversampling lists \
         data/dataset.cfg data/dataset_no_oversampling.cfg \
         source-archive-sha256.txt environment-pip-freeze.txt \
         smoke_probe_path.txt smoke_probe_sha256.txt results_manifest.json; do
  if [ -e "$m" ]; then members+=("$m"); else echo "snapshot note: not yet present (skipped): $m"; fi
done
[ "${#members[@]}" -ge 3 ] || { echo "SNAPSHOT FAILED: almost nothing to save"; exit 1; }
tar -czf "$SNAP" "${members[@]}"
sha256sum "$SNAP" > "$SNAP.sha256"; cat "$SNAP.sha256"
ls -la "$SNAP"
echo "SNAPSHOT OK - download $SNAP + $SNAP.sha256, then follow the checklist 'fresh session resume' section with 10_restore.sh."
