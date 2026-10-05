# Limitations + remaining work

Every claim in this publication is scoped in `docs/evaluation.md`. The following
are explicitly NOT claimed and remain open.

## Not claimed

- **No reference mic / LMS.** Single-microphone enhancement only. The problem
  statement's primary+reference wording is a documented scope deviation (owner
  chose one mic). Fixed EQ/gain is output conditioning, not adaptive filtering.
- **No physical room cancellation, hearing protection, or field certification.**
  The prototype enhances the headphone/communication signal; it does not cancel
  sound in the room.
- **No universal gains.** 480-mixture means exceed SNR > 15 dB, STOI > 0.85,
  PESQ-WB > 2.5 on development validation only. Per-condition achievement is
  unverified; STOI/PESQ gain CIs vs pretrained cross zero; 12/480 STOI drops >
  0.02 and 92/480 PESQ drops > 0.1 vs pretrained remain under review.
- **Final unseen evaluation IN PROGRESS.** Locked 420-case first-use subset
  (3 unseen speakers, 14 noise recordings, 7 categories, 5 SNR levels -5..15 dB,
  3 s clips); scored 378/420 as of 5 Oct 2026, 21:46 IST; no summary.json, no per-case
  metrics rollup, no scores published, no means extrapolated. The deployment
  preset will NOT be tuned on this set. Drone/artillery/armor real-source
  coverage is not established (7 available categories only).
- **No acoustic latency or power numbers.** Reported 6.23 ms mean / 9.43 ms max
  is processor roundtrip per 10 ms hop, not microphone-to-headphone latency
  (unmeasured, as is electrical power/thermal-battery behavior).
- **Manifest counts are file counts, not test counts.** Supported software proof
  is one Rust bundle-validation unit test, 10/10 parity cases (two rain
  recordings only), and Linux/ARM64 release builds — nothing more.
- **One short live run.** The accepted PF03 evidence is a single ~30 s test with
  phone-played gunfire plus speech. Sustained operation, recovery, broader
  noise/speaker coverage, artillery/armored-vehicle/drone sources, fresh-install
  validation, and redistribution licence clearance are all outstanding.

## Weights / binaries / audio: excluded, with reproduction path

- **Fine-tuned ONNX bundle** (`no_oversampling_onnx.tar.gz`, SHA
  `3db028b0...190bb`) and the **ARM64 `stream_raw` binary** (SHA
  `a0544cd3...0058025`) are NOT published: no affirmative publication right for
  the fine-tuned weights or the built binary was found in the workspace
  (the private release explicitly asserts no permission to publish, and
  redistribution licence clearance is listed as outstanding). Reproduce with:
  `training/06_train.sh` (GPU) → `training/07_export.sh` (bundle) →
  `cargo build --release --bin stream_raw` for the matching target
  (ARM64 Pi build used cargo-zigbuild) → gate with
  `evaluation/verify_custom_parity.py` before any live run.
- **Live audio WAVs** are excluded as unapproved private recordings; the accepted
  run is described by `docs/evidence/accepted_pi_report.json`, `docs/evidence/accepted_pi_telemetry.csv`, `docs/evidence/accepted_pi_audio_inspection.json`, `docs/evidence/accepted_pi_listener_acceptance.json`, and in
  `docs/deployment.md`. The recorded-demo sequence (no new capture needed) is:
  show hardware → play mic-input excerpt → play saved live-output excerpt →
  show the report row (6.23 ms processing, zero drops) → show the 480-mixture
  table labelled "raw model validation". Owner feedback on the saved output:
  "clear and natural" (single listener, not a blind panel).
- **Training datasets / checkpoints / notebooks / credentials** are excluded
  (private). Sources and licences: `THIRD_PARTY_NOTICES.md`.

## Remaining work, in order

1. Fresh unseen held-out evaluation of the locked system + deployment-preset scoring (scoring in progress — see evaluation §6; no scores published yet).
2. Measured microphone-to-headphone latency (declare an acceptance threshold first).
3. Broader noise/speaker trials, sustained PF03 operation, recovery behavior.
4. Measured power/thermal behavior and a sourced bill of materials.
5. Fresh-install validation and redistribution licence clearance.
6. Presentation packaging from this evidence (no PPT/video placeholders are
   included here; no frontend/UI is claimed).
