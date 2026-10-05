"""Experimental microphone -> persistent DFN -> wired output prototype.

The processor retains model state. On Windows it runs under WSL; with
--native-processor it runs directly on Linux, including Raspberry Pi.
Acoustic end-to-end latency must be measured separately.
"""

import argparse
import os
import csv
import json
import math
import queue
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
import psutil
import scipy.signal as signal
import sounddevice as sd
import soundfile as sf

from probe_pipe import read_exact

RATE = 48_000
HOP = 480
NATURAL_GAIN_DB = 10.638518975440501
EXTRA_LIVE_GAIN_DB = 3.0
APPROVED_POSTFILTER_BETA = 0.03


def peak_eq(frequency, q, gain_db):
    amplitude = 10 ** (gain_db / 40)
    w = 2 * np.pi * frequency / RATE
    alpha = np.sin(w) / (2 * q)
    b = np.array([1 + alpha * amplitude, -2 * np.cos(w), 1 - alpha * amplitude])
    a = np.array([1 + alpha / amplitude, -2 * np.cos(w), 1 - alpha / amplitude])
    return np.r_[b / a[0], 1.0, a[1:] / a[0]]


def natural_eq_sections():
    return np.vstack([
        signal.butter(2, 95, btype="highpass", fs=RATE, output="sos"),
        peak_eq(350, 0.8, -1.0)[None, :],
        peak_eq(2700, 1.0, 0.7)[None, :],
        peak_eq(4200, 0.7, 3.0)[None, :],
        peak_eq(2500, 1.1, 0.8)[None, :],
    ])


def soft_peak_limit(audio):
    magnitude = np.abs(audio)
    return np.where(
        magnitude <= 0.72,
        audio,
        np.sign(audio) * (0.72 + 0.23 * np.tanh((magnitude - 0.72) / 0.23)),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument("--input-device", type=int)
    parser.add_argument("--output-device", type=int)
    parser.add_argument("--seconds", type=float, default=20)
    parser.add_argument("--device-latency-ms", type=float,
                        help="requested PortAudio input/output latency in milliseconds; default uses the low-latency device setting")
    parser.add_argument("--mode", choices=["on", "off"], default="on")
    parser.add_argument("--gain-offset-db", type=float, default=-4.5,
                        help="additional dB beyond the current EQ and +3 dB preset")
    parser.add_argument("--save-dir", type=Path, default=Path("reports/pi_audio/live_pc_trial"))
    parser.add_argument("--processor", default=os.path.expanduser("~/anc-validation/pi_candidate_v17/stream_raw"), help="native stream_raw binary (portable default under $HOME; override with --processor)")
    parser.add_argument("--native-processor", action="store_true",
                        help="run the processor directly on the host instead of through WSL")
    parser.add_argument("--model-bundle", help="custom DFN ONNX tar.gz path as seen by the processor")
    parser.add_argument("--full-processing", action="store_true",
                        help="process all frames, including silence, without confidence stage skipping")
    args = parser.parse_args()
    if args.full_processing:
        parser.error("This PF03 preset uses standard processing; omit --full-processing")
    if args.list_devices:
        print(sd.query_devices())
        return
    if args.input_device is None or args.output_device is None:
        parser.error("provide --input-device and --output-device; use --list-devices")
    if not math.isfinite(args.seconds) or args.seconds <= 0:
        parser.error("--seconds must be positive")
    if args.device_latency_ms is not None and (not math.isfinite(args.device_latency_ms) or not 10 <= args.device_latency_ms <= 500):
        parser.error("--device-latency-ms must be between 10 and 500")
    if not math.isfinite(args.gain_offset_db) or not -30 <= args.gain_offset_db <= 20:
        parser.error("--gain-offset-db must be between -30 and +20")
    for idx, key in ((args.input_device, "max_input_channels"),
                     (args.output_device, "max_output_channels")):
        if sd.query_devices(idx)[key] < (1 if key == "max_input_channels" else 2):
            parser.error(f"device {idx} lacks required {key}")

    processor_command = ([args.processor] if args.native_processor else
                         ["wsl.exe", "-d", "Ubuntu", "--", args.processor])
    if args.model_bundle:
        processor_command += ["--model-bundle", args.model_bundle]
    processor_command += ["--pf-beta", str(APPROVED_POSTFILTER_BETA)]
    if args.full_processing:
        processor_command += ["--parity-mode"]
    load_started = time.perf_counter()
    process = subprocess.Popen(processor_command,
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, bufsize=0)
    ready = process.stderr.readline().decode(errors="replace").strip()
    if not ready.startswith("ready sample_rate=48000 hop_samples=480 "):
        process.kill()
        raise RuntimeError(f"processor not ready: {ready}")
    model_ready_seconds = time.perf_counter() - load_started

    incoming = queue.Queue(maxsize=32)
    enhanced = queue.Queue(maxsize=32)
    stop = threading.Event()
    errors = []
    state = {"mode": args.mode, "input_overruns": 0, "output_underruns": 0,
             "processor_ms": [], "portaudio_status": 0, "portaudio_status_details": [],
             "capture_samples": 0, "peak_guard_samples": 0}
    recorded_in, recorded_out = [], []
    sos = natural_eq_sections()
    total_gain_db = NATURAL_GAIN_DB + EXTRA_LIVE_GAIN_DB + args.gain_offset_db
    gain = 10 ** (total_gain_db / 20)

    def worker():
        zi = np.zeros((len(sos), 2), dtype=np.float64)
        try:
            while not stop.is_set() or not incoming.empty():
                try:
                    block = incoming.get(timeout=0.05)
                except queue.Empty:
                    continue
                started = time.perf_counter()
                process.stdin.write(block.astype("<f4", copy=False).tobytes())
                raw = np.frombuffer(read_exact(process.stdout, HOP * 4), dtype="<f4")
                filtered, zi = signal.sosfilt(sos, raw.astype(np.float64), zi=zi)
                amplified = filtered * gain
                state["peak_guard_samples"] += int(np.count_nonzero(np.abs(amplified) > 0.72))
                result = soft_peak_limit(amplified).astype(np.float32)
                state["processor_ms"].append((time.perf_counter() - started) * 1000)
                try:
                    enhanced.put_nowait(result)
                except queue.Full:
                    enhanced.get_nowait()
                    enhanced.put_nowait(result)
        except Exception as exc:
            errors.append(repr(exc))
            stop.set()

    processor_thread = threading.Thread(target=worker, daemon=True)
    processor_thread.start()
    # Six silent frames provide a small jitter buffer; acoustic latency still
    # needs direct measurement, including this buffer and audio-driver queues.
    for _ in range(6):
        enhanced.put_nowait(np.zeros(HOP, dtype=np.float32))

    def callback(indata, outdata, frames, timing, status):
        if status:
            state["portaudio_status"] += 1
            state["portaudio_status_details"].append({
                "capture_sample": state["capture_samples"], "flags": str(status)
            })
        state["capture_samples"] += frames
        mono = indata[:, 0].copy()
        recorded_in.append(mono)
        try:
            incoming.put_nowait(mono)
        except queue.Full:
            state["input_overruns"] += 1
        try:
            processed = enhanced.get_nowait()
        except queue.Empty:
            processed = np.zeros(frames, dtype=np.float32)
            if state["mode"] == "on":
                state["output_underruns"] += 1
        played = processed if state["mode"] == "on" else mono
        outdata[:] = played[:, None]
        recorded_out.append(played.copy())

    def commands():
        while not stop.is_set():
            try:
                key = input().strip().lower()
            except EOFError:
                return
            if key == "a": state["mode"] = "on"
            elif key == "b": state["mode"] = "off"
            elif key == "q": stop.set()

    threading.Thread(target=commands, daemon=True).start()
    started = time.monotonic()
    owner = psutil.Process()
    processor_owner = psutil.Process(process.pid)
    processor_peak_rss = 0
    python_peak_rss = 0
    telemetry = []
    last_hardware_at = -10.0
    last_temp_c = None
    last_throttled = None
    vcgencmd = shutil.which("vcgencmd") if args.native_processor else None
    processor_owner.cpu_percent(interval=None)
    try:
        requested_device_latency = ("low" if args.device_latency_ms is None
                                    else args.device_latency_ms / 1000)
        with sd.Stream(samplerate=RATE, blocksize=HOP, dtype="float32", channels=(1, 2),
                       device=(args.input_device, args.output_device),
                       latency=requested_device_latency,
                       callback=callback) as audio_stream:
            actual_device_latency = audio_stream.latency
            print(f"LIVE TEST gain={total_gain_db:.2f} dB, soft peak guard, PF03; "
                  f"mode={args.mode}; a=ON, b=BYPASS, q=stop. {ready}")
            while not stop.is_set() and time.monotonic() - started < args.seconds:
                time.sleep(1)
                recent = state["processor_ms"][-100:]
                try:
                    current_processor_rss = processor_owner.memory_info().rss
                    processor_peak_rss = max(processor_peak_rss, current_processor_rss)
                    processor_cpu = processor_owner.cpu_percent(interval=None)
                except psutil.NoSuchProcess:
                    current_processor_rss = 0
                    processor_cpu = 0.0
                python_rss = owner.memory_info().rss
                python_peak_rss = max(python_peak_rss, python_rss)
                elapsed = time.monotonic() - started
                if vcgencmd and elapsed - last_hardware_at >= 10:
                    try:
                        temp_text = subprocess.run([vcgencmd, "measure_temp"], capture_output=True,
                                                   text=True, timeout=2, check=True).stdout
                        throttle_text = subprocess.run([vcgencmd, "get_throttled"], capture_output=True,
                                                       text=True, timeout=2, check=True).stdout
                        match = re.search(r"temp=([0-9.]+)", temp_text)
                        last_temp_c = float(match.group(1)) if match else None
                        last_throttled = throttle_text.strip()
                    except (OSError, subprocess.SubprocessError, ValueError):
                        pass
                    last_hardware_at = elapsed
                telemetry.append({
                    "elapsed_seconds": round(elapsed, 3),
                    "processor_cpu_percent": processor_cpu,
                    "processor_rss_bytes": current_processor_rss,
                    "python_rss_bytes": python_rss,
                    "temperature_c": last_temp_c,
                    "throttled": last_throttled,
                    "input_overruns": state["input_overruns"],
                    "output_underruns": state["output_underruns"],
                    "portaudio_status_events": state["portaudio_status"],
                    "recent_mean_processor_ms": float(np.mean(recent)) if recent else None,
                    "recent_max_processor_ms": float(np.max(recent)) if recent else None,
                })
                print(f"mode={state['mode']} process_mean_ms={np.mean(recent) if recent else 0:.1f} "
                      f"process_max_ms={np.max(recent) if recent else 0:.1f} "
                      f"input_overruns={state['input_overruns']} output_underruns={state['output_underruns']} "
                      f"RAM_MiB={owner.memory_info().rss / 2**20:.0f} "
                      f"processor_CPU={processor_cpu:.0f}%")
    finally:
        stop.set()
        processor_thread.join(timeout=3)
        process.stdin.close()
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        stderr_tail = process.stderr.read().decode(errors="replace")[-1000:]
        args.save_dir.mkdir(parents=True, exist_ok=True)
        if recorded_in:
            sf.write(args.save_dir / "noisy_mic_48k.wav", np.concatenate(recorded_in), RATE,
                     subtype="PCM_16")
            sf.write(args.save_dir / "played_output_48k.wav", np.concatenate(recorded_out), RATE,
                     subtype="PCM_16")
        if telemetry:
            with (args.save_dir / "telemetry.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(telemetry[0]))
                writer.writeheader()
                writer.writerows(telemetry)
        roundtrips = np.asarray(state["processor_ms"], dtype=np.float64)
        report = {
            "workflow": "experimental live microphone to output via persistent DFN",
            "processor_transport": "native Linux" if args.native_processor else "Windows to WSL pipe",
            "sampled_processor_peak_rss_bytes": processor_peak_rss if args.native_processor else None,
            "sampled_python_peak_rss_bytes": python_peak_rss,
            "max_sampled_temperature_c": max((row["temperature_c"] for row in telemetry
                                               if row["temperature_c"] is not None), default=None),
            "final_sampled_throttled": last_throttled,
            "external_pretrained_model": None if args.model_bundle else "DeepFilterNet v0.5.6",
            "model_architecture": "DeepFilterNet3",
            "runtime_version": "libDF v0.5.6 isolated fork",
            "trained_project_gan_used": False,
            "custom_dfn_checkpoint_used": bool(args.model_bundle),
            "model_bundle": args.model_bundle,
            "model_bundle_sha256": (re.search(r"model_sha256=([0-9a-f]{64})", ready).group(1)
                                    if re.search(r"model_sha256=([0-9a-f]{64})", ready) else None),
            "full_model_processing": args.full_processing,
            "processor_ready": ready,
            "model_post_filter_beta": APPROVED_POSTFILTER_BETA,
            "output_eq": "highpass 95 Hz; -1 dB at 350 Hz; +0.7 dB at 2700 Hz; +3 dB at 4200 Hz Q0.7; +0.8 dB at 2500 Hz Q1.1",
            "output_gain_db": total_gain_db,
            "gain_offset_db": args.gain_offset_db,
            "peak_guard": "soft threshold 0.72; asymptotic peak 0.95",
            "peak_guard_samples": state["peak_guard_samples"],
            "input_device": args.input_device, "output_device": args.output_device,
            "duration_seconds": time.monotonic() - started,
            "model_hop_ms": 10, "initial_output_jitter_buffer_ms": 60,
            "model_ready_seconds": model_ready_seconds,
            "acoustic_end_to_end_latency_measured": False,
            "requested_device_latency_ms": args.device_latency_ms,
            "reported_portaudio_latency_seconds": actual_device_latency,
            "input_overruns": state["input_overruns"],
            "output_underruns": state["output_underruns"],
            "portaudio_status_events": state["portaudio_status"],
            "portaudio_status_details": state["portaudio_status_details"],
            "mean_processor_roundtrip_ms": float(np.mean(state["processor_ms"])) if state["processor_ms"] else None,
            "max_processor_roundtrip_ms": float(np.max(state["processor_ms"])) if state["processor_ms"] else None,
            "p99_processor_roundtrip_ms": float(np.percentile(roundtrips, 99)) if len(roundtrips) else None,
            "roundtrips_over_10ms": int(np.count_nonzero(roundtrips > 10)),
            "processor_exit": process.returncode, "errors": errors,
            "processor_stderr_tail": stderr_tail,
        }
        (args.save_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
