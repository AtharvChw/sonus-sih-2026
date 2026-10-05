# SONUS — embedded speech enhancement for defence-like noise (PS26052, Team Oblivion)

**What.** A portable single-microphone communication enhancer: a USB mic feeds a
persistent, stateful DeepFilterNet3 model running locally on a Raspberry Pi, and
the enhanced signal goes to wired headphones. It reduces interference in the
listened/communicated audio; it does not cancel sound in the room and provides
no hearing protection.

**Problem.** Stationary, changing, and impulsive defence noise (gunfire,
artillery, helicopter/drone, vehicles, sirens, wind) destroys speech
intelligibility over communication links, and the enhancement must run in real
time on low-power embedded hardware.

**Result.** Our custom six-epoch fine-tune (`no_oversampling`), validated on 480
development mixtures (SNR 16.987 dB, SI-SNR 17.108 dB, STOI 0.9183, PESQ-WB 3.0274 — means above
the 15 dB / 0.85 / 2.5 targets), exported to ONNX, and accepted in a 30-second
native Pi live run (mean 6.23 ms / max 9.43 ms processing per 10 ms hop, zero
drops, owner verdict "clear and natural"). All scopes and limits are stated
exactly in `docs/evaluation.md` and `docs/limitations.md` — nothing here is a
sealed final test or a field certification.

> Licence: pending owner decision — no licence is asserted for our own code in
> this folder. See `THIRD_PARTY_NOTICES.md`.

## Architecture

```mermaid
flowchart TD
  "USB microphone (mono 48 kHz)" --> "480-sample frames (10 ms hops)"
  "480-sample frames (10 ms hops)" --> "Persistent DeepFilterNet3 (custom fine-tune)"
  "Persistent DeepFilterNet3 (custom fine-tune)" --> "Post-filter beta 0.03"
  "Post-filter beta 0.03" --> "Natural EQ + presence lift"
  "Natural EQ + presence lift" --> "Fixed gain 9.14 dB"
  "Fixed gain 9.14 dB" --> "Soft peak guard"
  "Soft peak guard" --> "Wired headphones"
```

Details: `docs/architecture.md`. Accepted preset values: `configs/preset_pf03.json`.

## Requirements

- **Hardware:** Raspberry Pi (ARM64, Pi 4 class or better), one external USB
  microphone, wired headphones. No GPU, no cloud, no second microphone.
- **Software:** Linux with ALSA + PortAudio; Rust toolchain (to build `runtime/`);
  Python 3.12 with numpy, scipy, sounddevice, soundfile, psutil (for the live
  frontend); CUDA GPU machine + upstream DeepFilterNet v0.5.6 source only if
  retraining/re-exporting.

## Install

```bash
mkdir -p ~/anc-validation/pi_candidate_v17 ~/anc-validation/pi_pf03_clarity
# place the built ARM64 stream_raw + exported bundle into pi_candidate_v17/
# (build/export instructions in docs/deployment.md — binaries/weights excluded)
# copy runtime/, configs/, deployment/ from this publication, e.g. to ~/sonus-sih-2026/
python3 -m venv ~/anc-validation/live-venv
~/anc-validation/live-venv/bin/pip install numpy scipy sounddevice soundfile psutil
cd runtime && cargo build --release --bin stream_raw && cargo test --release
```

Working from a checkout of this folder, the layout above is
`<repo>/runtime`, `<repo>/configs`, `<repo>/deployment`. Package `dfn_stream_probe`, binary target `stream_raw` (`runtime/src/bin/stream_raw.rs`).

## Launch (accepted PF03 preset)

```bash
amixer -c 3 sset "Auto Gain Control" off   # confirm card number first
amixer -c 3 sset Mic 5
ps -eo pid,comm,%cpu,args --sort=-%cpu | head   # stop only a confirmed stray filter-chain PID
cd ~/anc-validation/pi_pf03_clarity
./run_pf03.sh 1 0 30   # input 1 = USB mic, output 0 = headphones, 30 s; q+Enter stops
```

## Stop / troubleshoot

Press `q` + Enter. If hops exceed 10 ms or drops appear, look for competing
audio processes first (a stray `pipewire -c filter-chain.conf` once cost ~80% CPU).
Full table: `docs/deployment.md`. Acoustic end-to-end latency is unmeasured by
design of the evidence so far — `accepted_pi_report.json` records processor roundtrip only.

## Benchmarks (verified scope only)

| Item | Verified result | SI-SNR (dB, same development scope) | Evidence |
|---|---|---|---|
| Candidate output SNR | 16.986951 dB, 480 development mixtures | 17.107518 | `docs/evidence/paired_validation_review.json` |
| Candidate STOI / PESQ-WB | 0.918306 / 3.027380, same set | — | `docs/evidence/validation_review.md` |
| Pretrained baseline | 16.369689 / 0.916571 / 3.002469 | 16.502343 | same as above |
| Speech regressions | 12/480 STOI drop > 0.02; 92/480 PESQ drop > 0.1 | — | same as above |
| Export tensor checks | passed (rtol 1e-4, atol 1e-5) | — | `docs/evidence/onnx_export_report.json` |
| Desktop waveform parity | 10/10, two rain recordings only | — | `docs/evidence/waveform_parity.json` |
| Accepted Pi live run | 6.23 ms mean / 9.43 ms max, zero drops, "clear and natural" | — | `docs/evidence/accepted_pi_report.json`, `docs/evidence/accepted_pi_telemetry.csv`, `docs/evidence/accepted_pi_listener_acceptance.json` |
| Builds + unit test | Linux/ARM64 release builds, one bundle-validation test | — | `docs/evidence/rust_linux_tests.log`, `docs/evidence/rust_linux_build.log`, `docs/evidence/rust_arm64_build.log`, `docs/evidence/rust_arm64_elf.txt`, `docs/evidence/arm64_package_report.json` |
| Training snapshot | 220 members, checkpoint hashes | — | `docs/evidence/training_snapshot_verification.json` |

These are development-validation means and one short live test — not per-condition
guarantees, not final-test scores, not scores of the PF03 preset, not acoustic latency. The 10/10 waveform parity validates the unfiltered-model binary path, not the PF03-preset configuration or quality.

## Accepted preset + model identity

Custom v17 `no_oversampling` bundle, SHA-256
`3db028b05a7bb014da2b04310872413be85b31a144feb463311f4be205e190bb`;
processor SHA-256
`a0544cd3c9865b7e44862f9849d4e57ce73a1f5edbf8e555ca5370b7c0058025`;
PF beta 0.03 with standard processing; natural EQ +3 dB at 4200 Hz (Q 0.7) and
+0.8 dB at 2500 Hz (Q 1.1); total gain 9.138518975440501 dB; soft peak guard;
Mic 5, AGC off. Exact values: `configs/preset_pf03.json`. Launcher: `deployment/run_pf03.sh`.

## Limitations + remaining work

Single mic (no reference/LMS); no room cancellation or hearing protection;
unmeasured acoustic latency and power; limited speakers/noise sources; one short
custom-PF03 run; sealed final and deployment-preset evaluation pending; weights,
binaries, audio, and datasets excluded (reproduce via `training/` + `cargo` —
see `docs/limitations.md`). No frontend or UI is part of this prototype.

## Attribution (DeepFilterNet3)

- **Upstream pretrained model:** DeepFilterNet3 by Rikorose et al.
  (https://github.com/Rikorose/DeepFilterNet, MIT/Apache-2.0) — used only to
  initialise our fine-tune and as the paired baseline (16.370 / 0.9166 / 3.0025).
- **Our fine-tune:** two six-epoch experiments on real VoiceBank speech +
  SESA/ESC noise; selected `no_oversampling` epoch-6 best. Weights not published.
- **Our runtime:** native ARM64 Rust/libDF persistent inference
  (`runtime/src/bin/stream_raw.rs` + vendored `runtime/libDF/` fork), the PF03
  live frontend, EQ/gain/guard conditioning, and all evaluation tooling.
  The original MetricGAN-style complex U-Net was a research baseline only and is
  not used here.

## Layout

```text
README.md  docs/{architecture,training,evaluation,deployment,limitations,decisions}.md
docs/evidence/ (sanitized proofs + SANITIZATION.md + manifest + verify script)
training/ (data-prep, fine-tune, export)  evaluation/ (scoring + parity + review)
runtime/ (processor, libDF fork, live frontend, launcher helper)
deployment/ (Pi launcher)  configs/ (accepted preset)  tests/ (test pointer)
THIRD_PARTY_NOTICES.md  .gitignore
```