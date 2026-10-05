#!/usr/bin/env python3
"""08_results_manifest.py — collect every result artifact + SHA-256 into results_manifest.json.

Run from anywhere: python3 08_results_manifest.py (uses WORK_ROOT or cwd/ps26052-gpu).
Verifies presence of: per-run config.ini, best checkpoints, train/console/export logs,
pretrained + export hashes, split lists + manifest, dataset cfgs, freeze files.
Prints MISSING entries and exits 1 if anything required is absent.
"""

import hashlib
import json
import os
import sys
from pathlib import Path

RUNS = ("impulse_weighted", "no_oversampling")

REQUIRED_RUN_FILES = ("config.ini", "train.log", "console.log", "export.log",
                      "pretrained_sha256.txt", "export_sha256.txt",
                      "smoke_base_sha256.txt", "smoke_best_sha256.txt")
REQUIRED_TOP = ("lists/split_manifest.csv", "data/dataset.cfg",
                "data/dataset_no_oversampling.cfg", "source-archive-sha256.txt",
                "environment-pip-freeze.txt")


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    work = Path(os.environ.get("WORK_ROOT", "ps26052-gpu"))
    missing, entries = [], {}

    def rec(rel, p, kind):
        if p.is_file():
            entries[rel] = {"sha256": sha256_file(p), "bytes": p.stat().st_size, "kind": kind}
        else:
            missing.append(rel)

    for run in RUNS:
        r = work / "runs" / run
        for name in REQUIRED_RUN_FILES:
            rec("%s/%s" % (run, name), r / name, "log")
        best = sorted((r / "checkpoints").glob("model_*.ckpt.best")) if (r / "checkpoints").is_dir() else []
        if best:
            for b in best:
                rec("%s/checkpoints/%s" % (run, b.name), b, "checkpoint")
        else:
            missing.append("%s/checkpoints/model_*.ckpt.best (none)" % run)
        bundle = r / "export" / ("%s_onnx.tar.gz" % run)
        rec("%s/export/%s_onnx.tar.gz" % (run, run), bundle, "onnx-bundle")
        for lf in ("train_speech", "train_sesa_noise", "train_esc_noise",
                   "valid_speech", "valid_sesa_noise", "valid_esc_noise",
                   "test_speech", "test_sesa_noise", "test_esc_noise"):
            rec("lists/%s.txt" % lf, work / "lists" / ("%s.txt" % lf), "list")
    for name in REQUIRED_TOP:
        rec(name, work / name, "meta")

    manifest = {"runs": list(RUNS), "entries": entries, "missing": missing,
                "selection_rule": "best checkpoint by validation loss only; existing 280-mixture development comparison never used for selection; sealed final holdout scored once later"}
    (work / "results_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("entries:", len(entries), "| missing:", len(missing))
    for m in missing:
        print("MISSING:", m)
    return 0 if not missing else 1


if __name__ == "__main__":
    sys.exit(main())
