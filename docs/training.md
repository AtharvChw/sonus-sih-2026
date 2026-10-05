# Training

Two six-epoch fine-tunes of pretrained DeepFilterNet3 on real speech mixed with
curated environmental/impulsive noise. The model is fine-tuned from the upstream
pretrained weights, not trained from scratch.

## Data pipeline (reproducible scripts in `training/`)

1. `training/03_make_lists.py` — source-disjoint train/valid/test list builder.
   Excludes every speaker and noise recording of the frozen 280-pair comparison
   set two ways (by identity via `training/frozen_manifest.csv`, and by SHA-256
   content hash of the noise bytes). Gates: 28 VoiceBank-train speakers, SESA
   train extraction of 340 clips, ESC folds 4-5 fully excluded, disjoint speech
   sets across splits. Raw datasets are NOT published; obtain them from their
   distributors (see `THIRD_PARTY_NOTICES.md`) and place them per `training/config.env`.
2. `training/04_make_hdf5.sh` — builds the nine 48 kHz HDF5 databases and writes
   the two dataset configs: `dataset.cfg` (SESA train oversampling x2.0,
   the `impulse_weighted` experiment) and `dataset_no_oversampling.cfg` (x1.0,
   the `no_oversampling` experiment).
3. `training/06_train.sh` + `training/run_dfn_train.py` — fine-tune both runs
   (batch 2, learning rate 0.0001, 6 epochs each; best checkpoint selected on
   validation loss only). Requires the upstream DeepFilterNet v0.5.6 Python
   source (`DFN_SOURCE` in `training/config.env`) and a CUDA GPU machine.
4. `training/07_export.sh` + `training/export_probe.py` — ONNX export of the best
   checkpoint per run via upstream `df/scripts/export.py`. The bundle must contain
   exactly `enc.onnx`, `erb_dec.onnx`, `df_dec.onnx`, `config.ini`, or the runtime
   bundle check (`runtime/src/bin/stream_raw.rs`) rejects it.

`training/config.env` centralises paths and knobs (`MAX_EPOCHS=6`, `BATCH_SIZE=2`,
`LR=0.0001`). Private Kaggle notebooks, dataset archives, credentials, and
checkpoints are excluded from this publication.

## Experiments

| Experiment | Dataset config | Training loss (best, epoch 6) |
|---|---|---|
| `impulse_weighted` | SESA train x2.0 | 0.6610152 |
| `no_oversampling` (selected) | SESA train x1.0 | 0.6613889 |

Losses are training losses, not SNR/STOI/PESQ. `no_oversampling` was selected for
runtime testing because of its speech-preservation tradeoff (fewer material STOI
drops), not declared universally superior. See `docs/decisions.md`.

## Model identity

Selected bundle SHA-256:
`3db028b05a7bb014da2b04310872413be85b31a144feb463311f4be205e190bb`
(8,181,815 bytes; `no_oversampling` epoch-6 best). Weights are NOT published;
see `docs/limitations.md` for the rights reason and for export/build commands.