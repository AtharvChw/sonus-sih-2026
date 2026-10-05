# Architecture

Single-microphone embedded speech enhancement for defence-like noise.
The prototype enhances the signal sent to headphones/communication output.
It does not physically cancel ambient sound and provides no hearing protection.

```mermaid
flowchart TD
  "USB microphone (mono 48 kHz)" --> "480-sample frames (10 ms hops)"
  "480-sample frames (10 ms hops)" --> "Persistent DeepFilterNet3 (custom fine-tune, ONNX)"
  "Persistent DeepFilterNet3 (custom fine-tune, ONNX)" --> "Post-filter beta 0.03"
  "Post-filter beta 0.03" --> "Natural EQ + presence lift"
  "Natural EQ + presence lift" --> "Fixed gain 9.14 dB"
  "Fixed gain 9.14 dB" --> "Soft peak guard"
  "Soft peak guard" --> "Wired headphones"
```

## Stages

1. **Capture.** One external USB microphone, 48 kHz mono, 480-sample (10 ms) frames.
   Owner-selected single mic; there is no reference microphone and no LMS stage.
2. **Neural enhancement.** Persistent, stateful DeepFilterNet3 (`deepfilternet3`,
   48 kHz, 480-sample hop) run natively via the Rust/libDF runtime
   (`runtime/src/bin/stream_raw.rs`). State is retained across frames; the model
   bundle is SHA-256 identity-checked at startup (`configs/preset_pf03.json`).
3. **Post-filter.** Upstream runtime post-filter at beta 0.03 with standard
   processing thresholds (accepted PF03 preset; not the full-processing
   parity configuration).
4. **Voice conditioning (fixed, not adaptive).** Natural EQ: high-pass 95 Hz,
   -1 dB at 350 Hz (Q 0.8), +0.7 dB at 2700 Hz (Q 1.0), plus presence lift
   +3 dB at 4200 Hz (Q 0.7) and +0.8 dB at 2500 Hz (Q 1.1). Fixed EQ and gain
   are output conditioning, not an adaptive-filter module.
5. **Gain + guard.** Total fixed gain 9.138518975440501 dB
   (natural 10.638518975440501 + live 3.0 + offset -4.5), then a soft peak guard
   (threshold 0.72, asymptotic peak 0.95) that can alter high-level speech.
6. **Playback.** Wired headphones. A 60 ms output jitter buffer plus requested
   80 ms PortAudio device latency apply; acoustic end-to-end latency is unmeasured.

## Implementation map

| Piece | Location |
|---|---|
| Native processor (bundle check, parity mode, SHA identity) | `runtime/src/bin/stream_raw.rs` |
| Vendored libDF fork (tract inference incl. startup/state parity handling) | `runtime/libDF/` |
| Accepted live frontend (EQ, gain, guard, telemetry, WAV capture) | `runtime/live_pf03_clarity.py` |
| Raw-PCM transport probe helper | `runtime/probe_pipe.py` |
| Accepted preset values | `configs/preset_pf03.json` |
| Pi launcher | `deployment/run_pf03.sh` |

## What this is not

No second/reference microphone, no LMS/NLMS adaptive-filter stage, no physical
room cancellation, no network audio transport (inference is local; the network
was used only for development/file transfer), no quantization, pruning, or
TensorRT. The earlier MetricGAN-style complex U-Net is a research baseline only
and is not part of this runtime.