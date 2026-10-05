# Evaluation

Displayed metrics are rounded for readability (same convention as the README);
evidence JSONs keep full precision.

## 1. Development validation: 480 deterministic mixtures (raw model outputs)

Three validation speakers, six utterances, 16 noise recordings across
8 categories, input SNR -5/0/5/10/15 dB. All systems scored on aligned
identities with no EQ, gain, or postfilter. Development validation only:
not a sealed final test, not live performance, not scores for the PF03 preset.

| System | Output SNR (dB) | SI-SNR (dB) | STOI | PESQ-WB |
|---|---|---:|---:|---:|
| Bypass | 5.00 | 5.05 | 0.838 | 1.50 |
| Pretrained DFN3 | 16.37 | 16.50 | 0.917 | 3.00 |
| impulse_weighted | 17.01 | 17.15 | 0.917 | 3.01 |
| no_oversampling (selected) | 16.99 | 17.11 | 0.918 | 3.03 |

Paired deltas vs pretrained (exploratory cluster bootstrap over only 3
speakers / 16 noises): SNR gain +0.62 dB, CI [0.45, 0.82]; STOI/PESQ gain
CIs cross zero. Material regressions vs pretrained for the selected model:
12/480 with STOI drop > 0.02, 92/480 with PESQ drop > 0.1, 3/480 with SNR drop
> 1 dB. No universal-superiority claim. Proof: `docs/evidence/paired_validation_review.json`,
`docs/evidence/validation_review.md`. Recompute from the private v18 archive with
`evaluation/review_v18_results.py` (`SONUS_WORKSPACE` pointing at a full workspace).

Scoring code: `evaluation/evaluate_candidates.py` (export + raw-output scoring;
needs the private training workspace and datasets — kept for provenance, not
runnable from this folder alone).

The frozen 280-pair comparison set (2 speakers, 4 utterances, 14 noise clips) is an older, separate development set: its speakers and noise recordings stay out of training, validation, and selection two ways (by identity via `training/frozen_manifest.csv`, and by SHA-256 content hash of the noise bytes). The 480-mixture set above is the newer v18 development validation for the two fine-tunes. The speaker- and noise-source-disjoint sealed unseen pool stays out of training, validation, and selection; a locked 420-case first-use subset drawn from it is now under evaluation (see §6) and will never be used for training or tuning.

## 2. Export tensor checks

Strict ONNX checks passed per part (`enc` / `erb_dec` / `df_dec`, rtol 1e-4,
atol 1e-5). Proof: `docs/evidence/onnx_export_report.json`.

## 3. Desktop waveform parity: 10/10 passed (two rain recordings only)

File-fed native Rust vs Python references, no fitted gain/lag, gate 0.01:
max relative RMS 0.0007158737216085572 (0.0716%), max absolute 0.0019702650606632233.
Reference scope is ten saved validation cases from two rain recordings, not full
defence coverage. Proof: `docs/evidence/waveform_parity.json`. The 10/10 waveform parity validates the unfiltered-model binary path, not the PF03-preset configuration or quality.
Rerun `evaluation/verify_custom_parity.py` (add `--wsl` on Windows) with `--processor <stream_raw> --bundle <bundle> --validation <dir> --save-dir <dir>`.
(--validation points at a directory of `0000_noisy.wav` + reference outputs; `--save-dir` receives the parity report).
## 4. Accepted 30-second custom-Pi PF03 live run

`evidence/pf03_clarity_20261005_200409` (folder excluded — private recordings; descriptors only): mean 6.23 ms,
max 9.43 ms, p99 8.17 ms per 10 ms hop; zero input/output drops, zero
PortAudio status events, zero roundtrips over 10 ms; processor exit 0;
max temperature 64.7 C, throttled 0x0; peak guard affected 5464 samples;
input peak 0.9346 / output peak 0.9500 with zero near-full-scale samples.
Owner listened to the original and the exact saved live output and called it
"clear and natural". Proof: `docs/evidence/accepted_pi_report.json`,
`docs/evidence/accepted_pi_telemetry.csv`,
`docs/evidence/accepted_pi_audio_inspection.json`,
`docs/evidence/accepted_pi_listener_acceptance.json`.

Processing time is NOT acoustic latency: acoustic end-to-end latency is
unmeasured, as is electrical power. One short run only: no sustained,
field, or recovery validation.

## 5. Supported software proof only

One Rust bundle-validation unit test (`cargo test` in `runtime/`),
the 10/10 parity cases above, and Linux x86_64 / ARM64 release builds
(`docs/evidence/rust_linux_build.log`, `docs/evidence/rust_arm64_build.log`,
`docs/evidence/rust_arm64_elf.txt`, `docs/evidence/rust_linux_tests.log`).
There is no 248-backend/2-frontend test evidence and no frontend production build.

## 6. Final PF03 evaluation (locked 420-case subset): IN PROGRESS

A locked 420-case first-use subset (3 unseen speakers, 14 noise recordings,
7 categories, 5 SNR levels -5..15 dB, 3 s clips; source hashes and
train/validation separation checked, scoring identities locked) is being scored
by a background scorer that appends results/NNNN.json files sharing a single
scoring identity hash. Scored 378/420 as of 5 Oct 2026, 21:46 IST. There is no summary.json,
no per-case metrics rollup, no scores published, and no means extrapolated.
The preset will NOT be tuned on this set.