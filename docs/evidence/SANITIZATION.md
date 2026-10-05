# Evidence sanitization log

All files under `docs/evidence/` are copies of workspace originals with
private recordings, private hashes, and personal machine information removed.
Audio WAVs were excluded entirely (unapproved private recordings).

## Per-file record

| Published file | Workspace source | Scrubbed / changed |
|---|---|---|
| `accepted_pi_report.json` | `evidence/pf03_clarity_20261005_200409/report.json` (runtime-written name; shipped as `accepted_pi_report.json`) | `model_bundle`: `/home/pi/anc-validation/...` rewritten to portable `~/anc-validation/...` (1 occurrence). No hostname, username, or other personal path was present. |
| `accepted_pi_telemetry.csv` | `evidence/pf03_clarity_20261005_200409/telemetry.csv` | No scrub needed: numeric telemetry only, no paths or identifiers found. |
| `accepted_pi_audio_inspection.json` | `evidence/pf03_clarity_20261005_200409/audio_inspection.json` | No scrub needed: sample-rate/peak/RMS figures only. |
| `accepted_pi_listener_acceptance.json` | `evidence/pf03_clarity_20261005_200409/listener_acceptance.json` | No scrub needed: owner verdict text only, no identifiers. |
| `waveform_parity.json` | `evidence/custom_runtime_parity/parity_report.json` (via `project/docs/evidence/`) | No scrub needed: hashes, ready-strings, and numeric diffs only. |
| `paired_validation_review.json` | `evidence/kaggle_v18/paired_review.json` (via `project/docs/evidence/`) | No scrub needed: metric means, CIs, and loss counts only. |
| `validation_review.md` | `handoff/V18_VALIDATION_REVIEW.md` (via `project/docs/evidence/`) | Copied verbatim; no personal paths or identifiers present. |
| `onnx_export_report.json` | `artifacts/dfn_v17_exports/no_oversampling_export_report.json` (via `project/docs/evidence/`) | No scrub needed: tensor deltas, bundle SHA, byte size only. |
| `training_snapshot_verification.json` | `evidence/kaggle_v17/snapshot_verification.json` (via `project/docs/evidence/`) | No scrub needed: archive SHA, byte/member counts, checkpoint hashes only. |
| `arm64_package_report.json` | `evidence/arm64_build/package_report.json` (via `project/docs/evidence/`) | Prior-layout Windows-path source keys rewritten to published-tree-relative `runtime/...` paths (see report `source_key_note`); SHA-256 values kept as build-time records. No hostnames, usernames, or credentials present. |
| `rust_linux_tests.log` | `logs/rust_linux_tests.log` (via `project/docs/evidence/`) | `/mnt/d/PS26052_OpenCode_Workspace/...` build-path prefix stripped to portable relative `project/...` paths (7 occurrences). No hostnames, usernames, or credentials present. |
| `rust_linux_build.log` | `logs/rust_linux_build.log` (via `project/docs/evidence/`) | `/mnt/d/PS26052_OpenCode_Workspace/...` build-path prefix stripped to portable relative `project/...` paths (7 occurrences). No hostnames, usernames, or credentials present. |
| `rust_arm64_build.log` | `logs/rust_arm64_build.log` (via `project/docs/evidence/`) | `/mnt/d/PS26052_OpenCode_Workspace/...` build-path prefix stripped to portable relative `project/...` paths (7 occurrences). No hostnames, usernames, or credentials present. |
| `rust_arm64_elf.txt` | `logs/rust_arm64_elf.txt` (via `project/docs/evidence/`) | `/mnt/d/PS26052_OpenCode_Workspace/...` file-path prefix stripped to portable relative `artifacts/...` path (1 occurrence). No hostnames, usernames, or credentials present. |

## Excluded (never copied)

- `noisy_mic_48k.wav`, `played_output_48k.wav` (accepted PF03 run) — unapproved
  private recordings of the owner. Described in words only; see `docs/deployment.md`.
- Old `manifest.json` / `SHA256SUMS` from the workspace — they hash private files
  and pre-sanitization bytes. A fresh manifest was generated from the published
  files instead (`manifest.json` + `verify_evidence.py` in this folder).
- Training datasets, Kaggle notebooks, credentials, checkpoints, ONNX bundles,
  compiled binaries, caches, `__pycache__`.

## Method

Each copied text file was scanned for `/home/`, `/mnt/`, usernames, `D:\` / `C:\` drive
paths, `Users`, and `DESKTOP`/`LAPTOP` hostnames before and after copying.
Found and rewritten: one `/home/pi` occurrence (`accepted_pi_report.json`) and 22 `/mnt/d/PS26052_OpenCode_Workspace/...` build-path prefixes (7 + 7 + 7 + 1 across the Rust logs and ELF record), stripped to portable relative paths. Post-copy scans of the evidence payload show no remaining live `/home/`, `/mnt/`, drive-letter, hostname, or credential traces (this log necessarily names the removed patterns).