"""Reject missing, failed or changed smoke evidence before starting training."""
import hashlib
import json
import math
import os
import sys
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    root = Path(os.environ["WORK_ROOT"])
    requested = sys.argv[1] if len(sys.argv) > 1 else "both"
    assert requested in ("both", "impulse_weighted", "no_oversampling")
    runs = ("impulse_weighted", "no_oversampling") if requested == "both" else (requested,)
    for run in runs:
        report = json.loads((root / "runs" / run / "smoke_report_base.json").read_text())
        assert report.get("ok") is True and report.get("run") == run, "no successful CUDA smoke"
        assert report.get("schema_version") == 1 and report.get("tag") == "base"
        assert str(report.get("epoch")) == "0" and report.get("inference_epoch") == 0
        assert report.get("gpu"), "missing GPU identity"
        for name in ("loss", "max_grad", "param_delta"):
            assert math.isfinite(float(report[name])), "nonfinite smoke result"
        assert report["max_grad"] > 0 and report["param_delta"] > 0
        hashes = report["inputs"]
        dataset = "dataset.cfg" if run == "impulse_weighted" else "dataset_no_oversampling.cfg"
        assert hashes["dataset_cfg_sha256"] == sha(root / "data" / dataset), "dataset config changed"
        assert hashes["split_manifest_sha256"] == sha(root / "lists/split_manifest.csv"), "split changed"
        assert hashes["run_config_sha256"] == sha(root / "runs" / run / "config.ini"), "config changed; rerun smoke"
        assert hashes["model_0_sha256"] == sha(root / "runs" / run / "checkpoints/model_0.ckpt"), "starting checkpoint changed"
        expected = {f"{split}_{kind}.hdf5" for split in ("train", "valid", "test")
                    for kind in ("speech", "sesa_noise", "esc_noise")}
        assert set(hashes["hdf5_sha256"]) == expected, "incomplete database evidence"
        for name in expected:
            assert hashes["hdf5_sha256"][name] == sha(root / "data" / name), "database changed: " + name
        wav = root / "runs" / run / "smoke_out_base.wav"
        assert report["smoke_wav"]["sha256"] == sha(wav), "smoke output changed"
    print("SMOKE EVIDENCE VERIFIED")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print("TRAIN BLOCKED: %s" % error, file=sys.stderr)
        sys.exit(1)
