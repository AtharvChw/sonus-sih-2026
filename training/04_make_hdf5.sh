#!/usr/bin/env bash
# 04_make_hdf5.sh — build the nine 48 kHz HDF5 databases + verify them. Resumable:
# re-running skips HDF5 files that already exist; delete one to rebuild it.
# Usage: bash 04_make_hdf5.sh   (after: source config.env; needs lists/ from 03_make_lists.py)
set -u -o pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "$HERE/config.env"
export PYTHONPATH="$DFN_SOURCE/DeepFilterNet${PYTHONPATH:+:$PYTHONPATH}"

[ -f "$WORK_ROOT/lists/split_manifest.csv" ] || { echo "run 03_make_lists.py first"; exit 1; }
mkdir -p "$WORK_ROOT/data"
for split in train valid test; do
  for kind in speech sesa_noise esc_noise; do
    list="$WORK_ROOT/lists/${split}_${kind}.txt"
    out="$WORK_ROOT/data/${split}_${kind}.hdf5"
    if [ -f "$out" ]; then echo "skip (exists): $out"; continue; fi
    sub="speech"; [ "$kind" = "speech" ] || sub="noise"
    python3 "$DFN_SOURCE/DeepFilterNet/df/scripts/prepare_data.py" \
      --sr 48000 --mono --num_workers 4 "$sub" "$list" "$out" || exit 1
  done
done

# dataset configs: impulse trial (SESA x2.0 train) + ablation (x1.0)
python3 - "$WORK_ROOT/data" <<'PY' || exit 1
import json, sys
d = sys.argv[1]
out = {}
for split in ('train', 'valid', 'test'):
    out[split] = [[split + '_speech.hdf5', 1.0],
                  [split + '_sesa_noise.hdf5', 2.0 if split == 'train' else 1.0],
                  [split + '_esc_noise.hdf5', 1.0]]
open(d + '/dataset.cfg', 'w').write(json.dumps(out, indent=2) + '\n')
abl = json.loads(json.dumps(out)); abl['train'][1][1] = 1.0
open(d + '/dataset_no_oversampling.cfg', 'w').write(json.dumps(abl, indent=2) + '\n')
print('wrote dataset.cfg (SESA train x2.0) + dataset_no_oversampling.cfg (x1.0)')
PY

# verify: 48 kHz, nonempty (nonzero exit fails the stage)
python3 - "$WORK_ROOT/data" <<'PY' || exit 1
import sys
from pathlib import Path
import h5py
for p in sorted(Path(sys.argv[1]).glob('*.hdf5')):
    with h5py.File(p) as h:
        kind = 'speech' if 'speech' in p.stem else 'noise'
        assert h.attrs['sr'] == 48000 and len(h[kind]) > 0, p
        print(p.name, 'rate', h.attrs['sr'], 'items', len(h[kind]))
PY
echo "HDF5 STAGE OK"
