# Kaggle runbook — fresh DFN v0.5.6 fine-tune (free GPU)

Goal: two gated runs (impulse-weighted + no-oversampling ablation), validation-only
selection, ONNX bundles + base/best smoke WAVs + hashes to download. Training has NOT
run yet — nothing below claims otherwise. Free-tier quotas/session limits change:
check current terms before a long run, and snapshot after EVERY session.
UI labels shift; pick the obvious equivalent.

## A. Laptop prep (browser)
A1. Download into one local folder (OUTSIDE the source project):
- `clean_trainset_28spk_wav.zip` (~2.32 GB) — https://datashare.ed.ac.uk/items/6ed35425-bf14-4d2b-93a1-0a4984952757
- `SESA.zip` (~26 MB) — https://zenodo.org/records/3519845
- ESC-50: `git clone https://github.com/karolpiczak/ESC-50`, then EITHER keep the folder
  or zip it as `ESC-50.zip` (02_get_data.sh accepts both, wrapper-folder included).
- This bundle: zip the contents of `training/` as `gpu_run.zip`.
A2. Read the VoiceBank/ESC-50/SESA license notes. Research use assumed; do NOT redistribute
raw audio. The existing 280-mixture set is a DEVELOPMENT comparison set (not a sealed
final test): its speakers/noise recordings are excluded from training AND validation
(the builder enforces it). A new speaker- and noise-source-disjoint sealed final
holdout must be built before final acceptance (Stage 3).

## B. Notebook setup (kaggle.com)
B1. Sign in → **Create** → **Notebook**. Attach the GPU accelerator.
B2. Attach data: **+ Add data** → upload the 3 archives (or once: make ONE private
**Dataset** with them and attach it — survives across notebooks, recommended).
B3. Upload `gpu_run.zip` into the working directory.

## C. Cells — run top to bottom. `%%bash` + `set -euo pipefail` means ANY failure
## stops the cell immediately: a silent pass is impossible. Stop and fix on red.
```bash
%%bash
# C1. GPU + bundle present
set -euo pipefail
nvidia-smi --query-gpu=name,memory.total --format=csv
unzip -q -o /kaggle/working/gpu_run.zip -d /kaggle/working/
ls /kaggle/working/gpu_run/
# EXPECT: a GPU name; 00_locate_inputs.sh 01_setup.sh ... 10_restore.sh all listed
```
```bash
%%bash
# C2. setup (aborts on no-CUDA / <20 GB free / bad imports)
set -euo pipefail
source /kaggle/working/gpu_run/config.env
bash /kaggle/working/gpu_run/01_setup.sh
# EXPECT tail: SETUP OK - versions frozen in .../environment-pip-freeze.txt
# Python not 3.10 -> warning recorded, continue only if imports pass.
# libdf wheels missing -> gpu_run/01_setup_notes.txt section 3, re-run this cell.
```
```bash
%%bash
# C3. discover inputs (no placeholders: searches /kaggle/input then raw uploads)
set -euo pipefail
source /kaggle/working/gpu_run/config.env
bash /kaggle/working/gpu_run/00_locate_inputs.sh
source "$WORK_ROOT/paths.env"
bash /kaggle/working/gpu_run/02_get_data.sh
# EXPECT tail: DATA STAGE OK (voicebank wavs 11572; SESA hash OK; ESC rows 2000)
```
```bash
%%bash
# C4. frozen-excluded split lists (reads paths.env: wrapper-dir safe)
set -euo pipefail
source /kaggle/working/gpu_run/config.env
source "$WORK_ROOT/paths.env"
python3 /kaggle/working/gpu_run/03_make_lists.py
# EXPECT: SPLIT BUILD OK (22/3/3 speakers, 340 SESA clips, 9 nonempty lists)
```
```bash
%%bash
# C5. HDF5 databases (longest CPU step; per-file resumable)
set -euo pipefail
source /kaggle/working/gpu_run/config.env
bash /kaggle/working/gpu_run/04_make_hdf5.sh
# EXPECT tail: HDF5 STAGE OK
```
```bash
%%bash
# C6. seed both runs from official weights
set -euo pipefail
source /kaggle/working/gpu_run/config.env
python3 /kaggle/working/gpu_run/05_init_runs.py
# EXPECT tail: INIT OK
```
```bash
%%bash
# C6b. REAL CUDA smoke, BOTH runs, starting weights (proves batch/loss/grads/update/reload/inference)
set -euo pipefail
source /kaggle/working/gpu_run/config.env
PROBE=$(python3 -c 'from pathlib import Path; import os; print(min(Path(os.environ["WORK_ROOT"]).joinpath("lists/train_speech.txt").read_text().splitlines()))')
[ -n "$PROBE" ] || { echo "SMOKE FAILED: no voicebank wav"; exit 1; }
python3 -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' "$PROBE" > "$WORK_ROOT/smoke_probe_sha256.txt"
echo "$PROBE" > "$WORK_ROOT/smoke_probe_path.txt"
python3 /kaggle/working/gpu_run/06b_smoke.py --run-dir "$WORK_ROOT/runs/impulse_weighted" --data-dir "$WORK_ROOT/data" --dataset-cfg "$WORK_ROOT/data/dataset.cfg" --probe "$PROBE" --epoch 0 --tag base --batch-size 2
python3 /kaggle/working/gpu_run/06b_smoke.py --run-dir "$WORK_ROOT/runs/no_oversampling" --data-dir "$WORK_ROOT/data" --dataset-cfg "$WORK_ROOT/data/dataset_no_oversampling.cfg" --probe "$PROBE" --epoch 0 --tag base --batch-size 2
# EXPECT: SMOKE ALL PASS x2 (finite loss, nonzero grads, param delta > 0, reload identical, wav hash).
# The SAME probe rule feeds 07_export smoke later, so base/best outputs stay comparable.
```
```bash
%%bash
# C7. train (same command resumes from latest checkpoint; logs append)
set -euo pipefail
source /kaggle/working/gpu_run/config.env
# Only after owner authorization. This command checks both C6b reports first.
bash /kaggle/working/gpu_run/06_train.sh both
# EXPECT: [valid]/[test] lines; a model_*.ckpt.best per run when done.
# CHECK cuda: grep -E 'Running on device' /kaggle/working/ps26052-gpu/runs/*/train.log
# SELECTION: best checkpoint by VALIDATION loss only. Never the existing 280-mixture
# development comparison set; the sealed final holdout is scored once, later, on the laptop.
```
```bash
%%bash
# C8. end of EVERY session: snapshot + download
set -euo pipefail
source /kaggle/working/gpu_run/config.env
bash /kaggle/working/gpu_run/09_snapshot.sh
# EXPECT: SNAPSHOT OK. Download ps26052-snapshot-<tag>.tar.gz + .sha256 NOW.
```
```bash
%%bash
# C9. fresh-session resume: upload snapshot + re-attach data, then
set -euo pipefail
source /kaggle/working/gpu_run/config.env
export SNAP_TARBALL=/kaggle/working/ps26052-snapshot-<tag>.tar.gz  # <-- fill tag from C8
bash /kaggle/working/gpu_run/10_restore.sh
# Re-uploaded data may sit at new paths, so re-discover + re-stage + rebuild HDF5:
bash /kaggle/working/gpu_run/00_locate_inputs.sh && source "$WORK_ROOT/paths.env"
bash /kaggle/working/gpu_run/02_get_data.sh && source "$WORK_ROOT/paths.env"
python3 /kaggle/working/gpu_run/03_make_lists.py
bash /kaggle/working/gpu_run/04_make_hdf5.sh
PROBE=$(python3 -c 'from pathlib import Path; import os; print(min(Path(os.environ["WORK_ROOT"]).joinpath("lists/train_speech.txt").read_text().splitlines()))')
# Rebuild base gate evidence against THIS session's files. Short-run checkpoint
# retention keeps model_0; the trainer itself resumes from its latest checkpoint.
python3 /kaggle/working/gpu_run/06b_smoke.py --run-dir "$WORK_ROOT/runs/impulse_weighted" --data-dir "$WORK_ROOT/data" --dataset-cfg "$WORK_ROOT/data/dataset.cfg" --probe "$PROBE" --epoch 0 --tag base --batch-size 2
python3 /kaggle/working/gpu_run/06b_smoke.py --run-dir "$WORK_ROOT/runs/no_oversampling" --data-dir "$WORK_ROOT/data" --dataset-cfg "$WORK_ROOT/data/dataset_no_oversampling.cfg" --probe "$PROBE" --epoch 0 --tag base --batch-size 2
bash /kaggle/working/gpu_run/06_train.sh both   # resumes from restored checkpoints
# EXPECT tail: RESTORE OK ... HDF5 STAGE OK ... training continues (not restarts)
```
```bash
%%bash
# C10. export + smoke + package (only after best checkpoints exist)
set -euo pipefail
source /kaggle/working/gpu_run/config.env
PROBE=$(cat "$WORK_ROOT/smoke_probe_path.txt")
python3 /kaggle/working/gpu_run/06b_smoke.py --run-dir "$WORK_ROOT/runs/impulse_weighted" --data-dir "$WORK_ROOT/data" --dataset-cfg "$WORK_ROOT/data/dataset.cfg" --probe "$PROBE" --epoch best --tag best --batch-size 2
python3 /kaggle/working/gpu_run/06b_smoke.py --run-dir "$WORK_ROOT/runs/no_oversampling" --data-dir "$WORK_ROOT/data" --dataset-cfg "$WORK_ROOT/data/dataset_no_oversampling.cfg" --probe "$PROBE" --epoch best --tag best --batch-size 2
bash /kaggle/working/gpu_run/07_export.sh both
# EXPECT: PACKAGE OK; <run>_onnx.tar.gz holds exactly enc/erb_dec/df_dec/config.ini;
# fixed-probe smoke_base (-e 0) + smoke_best (-e best) hashed separately.
```

## D. Bring results home
D1. Download: `ps26052-gpu-handoff.tar.gz` + `.sha256`, each `runs/<run>/smoke_{base,best}/`,
latest snapshot. Verify sha256 locally; copy the tarball to a local evidence folder outside the dataset tree.
D2. Report back: GPU name, wall time per run, chosen validation checkpoints, deviations,
plus every `runs/*/smoke_report_*.json`.
I score the sealed final holdout ONCE per predeclared candidate on the laptop (never for selection),
compare vs the pretrained baseline (D3/D5), and only then touch the Pi plan. The existing
280-mixture development comparison is reported for context only.
