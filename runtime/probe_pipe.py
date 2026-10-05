"""Verify persistent raw-PCM transport using the approved saved microphone WAV."""

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import soundfile as sf


def read_exact(stream, size):
    chunks = bytearray()
    while len(chunks) < size:
        part = stream.read(size - len(chunks))
        if not part:
            raise RuntimeError(f"processor ended after {len(chunks)}/{size} output bytes")
        chunks.extend(part)
    return bytes(chunks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--processor", required=True,
                        help="Linux stream_raw binary; launched through wsl.exe")
    parser.add_argument("--pf-beta", type=float)
    args = parser.parse_args()
    audio, rate = sf.read(args.input, dtype="float32", always_2d=True)
    if rate != 48000 or audio.shape[1] != 1:
        parser.error("input must be 48 kHz mono")
    hop = 480
    frames = len(audio) // hop
    command = ["wsl.exe", "-d", "Ubuntu", "--", str(args.processor)]
    if args.pf_beta is not None:
        command += ["--pf-beta", str(args.pf_beta)]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, bufsize=0)
    ready = process.stderr.readline().decode("utf-8", errors="replace").strip()
    if not ready.startswith("ready sample_rate=48000 hop_samples=480 "):
        process.kill()
        raise RuntimeError(f"processor did not signal readiness: {ready}")
    frame_times_ms = []
    output = np.empty(frames * hop, dtype=np.float32)
    for frame in range(frames):
        block = np.ascontiguousarray(audio[frame * hop:(frame + 1) * hop, 0])
        started = time.perf_counter()
        process.stdin.write(block.astype("<f4", copy=False).tobytes())
        received = read_exact(process.stdout, hop * 4)
        frame_times_ms.append((time.perf_counter() - started) * 1000)
        output[frame * hop:(frame + 1) * hop] = np.frombuffer(received, dtype="<f4")
    process.stdin.close()
    stderr_tail = process.stderr.read().decode("utf-8", errors="replace")[-1000:]
    exit_code = process.wait(timeout=10)
    if exit_code:
        raise RuntimeError(f"processor exited {exit_code}: {stderr_tail}")
    # Reconstruct file CLI `-D` output only for comparison. A live stream keeps
    # its algorithmic delay and must be measured at the audio devices.
    delay = 1440
    aligned = np.zeros(len(audio) - delay, dtype=np.float32)
    aligned[:len(output) - delay] = output[delay:]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Match libDF's hound writer: multiply by i16::MAX, then truncate to i16.
    pcm16 = np.clip(aligned * 32767.0, -32768, 32767).astype(np.int16)
    sf.write(args.output, pcm16, rate, subtype="PCM_16")
    report = {
        "workflow": "file-fed raw PCM pipe; no live audio devices",
        "input": str(args.input), "output": str(args.output),
        "post_filter_beta": args.pf_beta,
        "processor_ready": ready, "processed_hops": frames,
        "mean_pipe_roundtrip_ms": float(np.mean(frame_times_ms)),
        "max_pipe_roundtrip_ms": float(np.max(frame_times_ms)),
        "pipe_roundtrips_exceeding_10ms": int(np.count_nonzero(np.asarray(frame_times_ms) > 10)),
        "process_exit": exit_code, "process_stderr_tail": stderr_tail,
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
