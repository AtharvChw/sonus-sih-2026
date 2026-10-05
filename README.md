# SONUS — embedded speech enhancement for defence-like noise (PS26052, Team Oblivion)

> Licence: pending owner decision — no licence is asserted for our own code in
> this folder. See `THIRD_PARTY_NOTICES.md`.

Displayed metrics are rounded for readability; evidence JSONs keep full precision.

## Problem

Stationary, changing, and impulsive defence-like noise (gunfire, artillery,
helicopter/drone, vehicles, sirens, wind) destroys speech intelligibility over
communication links, and the enhancement must run in real time on low-power
embedded hardware (Raspberry Pi, no GPU, no cloud).

## Solution

A portable single-microphone enhancer: a USB microphone feeds a persistent,
stateful custom DeepFilterNet3 model running locally on the Pi, followed by the
locked PF03 conditioning chain (post-filter, voice EQ, fixed gain, peak guard),
out to wired headphones. It enhances the listened/communicated signal only: it
does not cancel sound in the room and provides no hearing protection. There is
no reference microphone and no LMS adaptive stage.

## Architecture

```mermaid
flowchart LR
  A["USB microphone"] --> B["48 kHz capture"]
  B --> C["Custom DeepFilterNet3"]
  C --> D["PF03"]
  D --> E["Voice EQ"]
  E --> F["Gain and peak guard"]
  F --> G["Wired headphones"]
```

Details: `docs/architecture.md`. Accepted preset values: `configs/preset_pf03.json`.

## Original contributions (vs upstream DeepFilterNet3)

- **Our fine-tune:** two six-epoch DeepFilterNet3 fine-tunes on real VoiceBank
  speech mixed with curated SESA/ESC noise; selected `no_oversampling` (epoch-6
  best). Upstream pretrained weights were used only to initialise the fine-tune
  and as the paired baseline. Weights are not published.
- **Our runtime:** native ARM64 Rust/libDF persistent inference
  (`runtime/src/bin/stream_raw.rs` + vendored `runtime/libDF/` fork of upstream
  DeepFilterNet v0.5.6), the PF03 live frontend (`runtime/live_pf03_clarity.py`;
  EQ/gain/guard conditioning), and all evaluation tooling. The original
  MetricGAN-style complex U-Net was a research baseline only and is not used here.
- **Our preset:** the owner-accepted PF03 chain — post-filter beta 0.03, natural
  EQ (+3 dB at 4200 Hz Q 0.7; +0.8 dB at 2500 Hz Q 1.1; high-pass 95 Hz; -1 dB
  at 350 Hz; +0.7 dB at 2700 Hz; no bass boost), total gain 9.138518975440501 dB
  (offset -4.5), soft peak guard (0.72/0.95), Mic 5 with AGC off. Exact values:
  `configs/preset_pf03.json`; launcher: `deployment/run_pf03.sh`.

Upstream: DeepFilterNet3 by Rikorose et al.
(https://github.com/Rikorose/DeepFilterNet, MIT/Apache-2.0). See
`THIRD_PARTY_NOTICES.md`.

## Verified results

Development-validation means on 480 deterministic mixtures (raw model outputs,
no EQ, gain, or post-filter) plus one short live test — not per-condition
guarantees, not final-test scores, and not scores of the PF03 preset:

| Item | Verified result | Evidence |
|---|---|---|
| Candidate (`no_oversampling`) output | SNR 16.99 dB, SI-SNR 17.11 dB, STOI 0.918, PESQ-WB 3.03 | `docs/evidence/paired_validation_review.json`, `docs/evidence/validation_review.md` |
| Pretrained DFN3 baseline | 16.37 / 16.50 / 0.917 / 3.00 | same as above |
| `impulse_weighted` (unselected) | 17.01 / 17.15 / 0.917 / 3.01 | same as above |
| Bypass (floor) | 5.00 / 5.05 / 0.838 / 1.50 | same as above |
| Speech regressions, selected vs pretrained | 12/480 STOI drop > 0.02; 92/480 PESQ drop > 0.1; 3/480 SNR drop > 1 dB | same as above |
| Export tensor checks | passed (rtol 1e-4, atol 1e-5) | `docs/evidence/onnx_export_report.json` |
| Desktop waveform parity | 10/10, two rain recordings only (unfiltered-model binary path, not PF03 quality) | `docs/evidence/waveform_parity.json` |
| Accepted 30 s Pi live run | 6.23 ms mean / 9.43 ms max / 8.17 ms p99 per 10 ms hop; zero input/output drops, zero PortAudio events, zero hops over 10 ms; "clear and natural" (single owner-listener, not a blind panel) | `docs/evidence/accepted_pi_report.json`, `docs/evidence/accepted_pi_telemetry.csv`, `docs/evidence/accepted_pi_listener_acceptance.json` |
| Builds + unit test | Linux/ARM64 release builds, one bundle-validation test | `docs/evidence/rust_linux_tests.log`, `docs/evidence/rust_linux_build.log`, `docs/evidence/rust_arm64_build.log`, `docs/evidence/rust_arm64_elf.txt`, `docs/evidence/arm64_package_report.json` |
| Training snapshot | 220 members, checkpoint hashes | `docs/evidence/training_snapshot_verification.json` |

The 6.23 ms figure is processor roundtrip per hop, not microphone-to-headphone
latency: acoustic end-to-end latency is unmeasured, as is electrical power. The
480-mixture set is development raw-output validation, not a final test. Full
scope: `docs/evaluation.md`.

## Setup

Install/build/launch commands below are audited against the reorganized tree
(`runtime/`, `training/`, `configs/`, `deployment/` paths, filenames, cargo bin
name, pip packages) — but a fresh-operator install was NOT tested and remains
unverified.

### Install

```bash
mkdir -p ~/anc-validation/pi_candidate_v17 ~/anc-validation/pi_pf03_clarity
# place the built ARM64 stream_raw + exported bundle into pi_candidate_v17/
# (build/export: docs/deployment.md — binaries/weights excluded)
# copy runtime/, configs/, deployment/ from this publication, e.g. to ~/sonus-sih-2026/
python3 -m venv ~/anc-validation/live-venv
~/anc-validation/live-venv/bin/pip install numpy scipy sounddevice soundfile psutil
cd runtime && cargo build --release --bin stream_raw && cargo test --release
```

Working from a checkout of this folder, the layout above is `<repo>/runtime`,
`<repo>/configs`, `<repo>/deployment`. Package `dfn_stream_probe`, binary target
`stream_raw` (`runtime/src/bin/stream_raw.rs`).

### Launch (accepted PF03 preset)

```bash
amixer -c 3 sset "Auto Gain Control" off   # confirm card number first
amixer -c 3 sset Mic 5
ps -eo pid,comm,%cpu,args --sort=-%cpu | head   # inspect first; stop ONLY a confirmed competing PID, never blind-kill
cd ~/anc-validation/pi_pf03_clarity
./run_pf03.sh 1 0 30   # input 1 = USB mic, output 0 = headphones, 30 s; q+Enter stops
```

A stray `pipewire -c filter-chain.conf` process once cost ~80% CPU and ruined a
run — inspect with `ps` and stop only the confirmed competing process by its
current PID. Full table: `docs/deployment.md`. Press `q` + Enter to stop.

## Final PF03 evaluation: IN PROGRESS

Final PF03 evaluation: IN PROGRESS — locked 420-case first-use subset
(3 unseen speakers, 14 noise recordings, 7 categories, 5 SNR levels -5..15 dB,
3 s clips, source hashes + train/validation separation checked, scoring
identities locked); scored 378/420 as of 5 Oct 2026, 21:46 IST; no summary.json, no per-case
metrics rollup, no scores published, no means extrapolated. The preset will NOT
be tuned on this set.

## Evidence

Sanitized proofs live in `docs/evidence/` (see `docs/evidence/README.md` and
`docs/evidence/SANITIZATION.md`); hashes and provenance are recorded in
`docs/evidence/manifest.json` and checkable with `docs/evidence/verify_evidence.py`.

## Limitations (summary)

- Single microphone only: no reference mic, no LMS adaptive stage. Fixed
  EQ/gain is output conditioning, not adaptive filtering.
- No room cancellation, no hearing protection, no field certification: the
  prototype enhances the headphone/communication signal only.
- 480-mixture means exceed SNR > 15 dB, STOI > 0.85, PESQ-WB > 2.5 on
  development validation only; per-condition achievement is unverified,
  STOI/PESQ gain CIs vs pretrained cross zero, and regressions remain under
  review.
- Final unseen evaluation is in progress (see above); deployment-preset scoring
  is pending until it completes.
- One short (~30 s) custom-PF03 live run only; sustained operation and recovery
  are deferred.
- Acoustic latency and electrical power are unmeasured (6.23 ms is processor
  roundtrip per hop, not microphone-to-headphone latency).
- Limited speakers/noise sources; drone/artillery/armor real-source coverage is
  not established.
- Weights, binaries, audio, and datasets are excluded (reproduce via `training/`
  + `cargo`); fresh-install validation and redistribution licence clearance are
  outstanding.
- No frontend or UI is part of this prototype.

Full list: `docs/limitations.md`.

## Layout

```text
README.md  docs/{architecture,training,evaluation,deployment,limitations,decisions}.md
docs/evidence/ (sanitized proofs + SANITIZATION.md + manifest + verify script)
training/ (data-prep, fine-tune, export)  evaluation/ (scoring + parity + review)
runtime/ (processor, libDF fork, live frontend, launcher helper)
deployment/ (Pi launcher)  configs/ (accepted preset)  tests/ (test pointer)
THIRD_PARTY_NOTICES.md  .gitignore
```