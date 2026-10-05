# Verification evidence

Copies of workspace reports/logs with private recordings, private hashes, and
personal machine information removed (see `SANITIZATION.md`). File integrity is
not the same as independently rerunning the experiments. Nothing was published
by creating this folder.

## Claim-to-proof map

| Claim | Proof | Actual scope |
|---|---|---|
| Rust bundle-validation test passed | `docs/evidence/rust_linux_tests.log` | ONE test passed (missing/duplicate/extra/incompatible bundle cases). Not 248 backend tests. |
| Linux x86_64 release build succeeded | `docs/evidence/rust_linux_build.log` | Rust audio processor release build. Not a frontend build or deployed production system. |
| Candidate matches Python waveform references | `docs/evidence/waveform_parity.json` | 10/10 file-fed cases; two rain recordings; no acoustic/live validation. |
| ONNX export tensor checks passed | `docs/evidence/onnx_export_report.json` | Conversion checks; interpret thresholds/results in the report. |
| Candidate validation means | `docs/evidence/paired_validation_review.json`, `docs/evidence/validation_review.md` | 480 development mixtures; not sealed final test. Read regression counts and uncertainty. |
| Training checkpoint snapshot preserved | `docs/evidence/training_snapshot_verification.json` | Archive integrity/inventory; not a standalone reproduction of GPU training. |
| ARM64 release/package built | `docs/evidence/rust_arm64_build.log`, `docs/evidence/rust_arm64_elf.txt`, `docs/evidence/arm64_package_report.json` | Architecture/dependencies/package integrity. Build-time snapshot. |
| Accepted custom-Pi PF03 live run | `docs/evidence/accepted_pi_report.json`, `docs/evidence/accepted_pi_telemetry.csv`, `docs/evidence/accepted_pi_audio_inspection.json`, `docs/evidence/accepted_pi_listener_acceptance.json` | One ~30 s run: 6.23 ms mean / 9.43 ms max processing per 10 ms hop, zero drops/events, owner verdict "clear and natural". Processing time is not acoustic latency. Objective 480-mixture scores apply to raw model outputs, not this preset. Runtime `telemetry.csv` is generated per run; only the accepted copy is shipped. |

## Unsupported statement: do not use

"248 backend + 2 frontend tests passed, with successful production build" has no
supporting evidence. Do not put this claim in any submission as verified.

## Recheck evidence-file integrity

From this folder, using Python 3.11+:

```powershell
python verify_evidence.py
```

Expected: `EVIDENCE HASHES PASS: 17 files`. This only checks the preserved files;
it does not rerun training, tests, or live capture.