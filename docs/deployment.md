# Deployment (Raspberry Pi, accepted PF03 preset)

Portable Pi layout (no personal machine information):

- Base install: `~/anc-validation/pi_candidate_v17/` holding the native ARM64
  `stream_raw` binary and the `no_oversampling_onnx.tar.gz` bundle.
- This overlay: `~/anc-validation/pi_pf03_clarity/` holding `deployment/run_pf03.sh`
  (run from this publication) and the preset reference `configs/preset_pf03.json`.
- Python env: `~/anc-validation/live-venv/` with numpy, scipy, sounddevice,
  soundfile, psutil.

The ARM64 binary and the ONNX bundle are NOT published (rights unverified —
see `docs/limitations.md`). Build them first: native release build with
`cargo build --release --bin stream_raw` (ARM64 target for the Pi; the accepted
binary was cross-built with cargo-zigbuild), and export the bundle with
`training/07_export.sh`. Verify identities before every launch:

- Bundle SHA-256: `3db028b05a7bb014da2b04310872413be85b31a144feb463311f4be205e190bb`
- Processor SHA-256: `a0544cd3c9865b7e44862f9849d4e57ce73a1f5edbf8e555ca5370b7c0058025`
- Ten-case ARM64 parity must pass with matching hashes
  (`evaluation/verify_custom_parity.py`).

`deployment/run_pf03.sh` performs the SHA check and then launches
`runtime/live_pf03_clarity.py`.

## Hardware / software requirements

- Raspberry Pi (ARM64, Pi 4 class or better) running Linux with `vcgencmd` optional.
- One external USB microphone; wired headphones (speakers risk feedback).
- ALSA + PortAudio (`sounddevice`); 48 kHz mono full-duplex at 10 ms frames.
- Rust toolchain for `runtime/`; Python 3.12 with the packages above for the frontend.

No GPU, cloud connection, Wi-Fi transport, or second microphone is required or used.

## Install

Commands below are audited against this tree (paths, filenames, cargo bin name, pip packages); a fresh-operator install is unverified.

```bash
# on the Pi
mkdir -p ~/anc-validation/pi_candidate_v17 ~/anc-validation/pi_pf03_clarity
# place the built stream_raw + exported bundle into pi_candidate_v17/
# copy runtime/, configs/, deployment/ from this publication, e.g. to ~/sonus-sih-2026/
python3 -m venv ~/anc-validation/live-venv
~/anc-validation/live-venv/bin/pip install numpy scipy sounddevice soundfile psutil
```

## Launch (accepted preset)

```bash
# 1. Mixer: USB mic, no AGC (example: card 3 — confirm with amixer -c 3 info)
amixer -c 3 sset "Auto Gain Control" off
amixer -c 3 sset Mic 5
# 2. Check for competing experimental audio processes first:
ps -eo pid,comm,%cpu,args --sort=-%cpu | head
# A leftover `pipewire -c filter-chain.conf` process once stole ~80% CPU and
# ruined a run; stop only the confirmed experimental process by its current PID.
# 3. Launch (input 1 = USB mic, output 0 = wired headphones, 30 s):
cd ~/anc-validation/pi_pf03_clarity
./run_pf03.sh 1 0 30
# Recheck device numbers with --list-devices if USB topology changed.
```

Each run writes a timestamped folder with the noisy-mic WAV, the exact played-output WAV, `report.json`, and `telemetry.csv` (runtime filenames; only the accepted copies are shipped, as `docs/evidence/accepted_pi_report.json` and `docs/evidence/accepted_pi_telemetry.csv`). Press `q` + Enter to stop.

## Stop

Press `q` + Enter in the launcher terminal. The launcher exits the processor
cleanly (`processor_exit 0` in the report). The base package directory is left
unchanged and serves as rollback (note: its launcher uses a different,
non-PF03 gain preset, so it is not acoustically identical).

## Troubleshoot

| Symptom | Check |
|---|---|
| Identity mismatch at startup | Re-verify SHA-256 of `stream_raw` and the bundle against `configs/preset_pf03.json`; rebuild/re-export if stale. |
| Overruns/underruns or hops over 10 ms | `ps` for competing processors or PipeWire filter chains; stop only the confirmed experimental PID; keep phone/computer load stable. |
| ALSA device busy | Inspect the device owner before touching it; re-enumerate with `--list-devices`. |
| Mixer resets after reboot | Re-apply the two `amixer` lines; startup does not persist system mixer/service changes. |
| Clipped/loud capture | Lower mic capture level (Mic value / card gain), keep 10–15 cm mic distance; the accepted run had zero near-full-scale samples — do not chase the old +23.81 dB capture-gain setting. |
| No measured latency number | Expected: acoustic end-to-end latency has not been measured; `accepted_pi_report.json` records processor roundtrip only. |

## Accepted preset (exact)

Custom v17 `no_oversampling` bundle, standard runtime thresholds, PF beta 0.03
(`full_model_processing=false`), natural EQ (high-pass 95 Hz; -1 dB at 350 Hz;
+0.7 dB at 2700 Hz; +3 dB at 4200 Hz Q 0.7; +0.8 dB at 2500 Hz Q 1.1; no bass boost), total gain
9.138518975440501 dB (offset -4.5), soft peak guard (0.72 / 0.95), USB Mic 5
with AGC off, input 1 / output 0, 48 kHz mono, 10 ms hop, 60 ms jitter buffer,
80 ms requested device latency. Full values: `configs/preset_pf03.json`.