#!/usr/bin/env python3
"""Source-disjoint train/valid/test list builder (PS26052, stdlib only).

Excludes EVERY speaker and noise recording in the frozen 280-pair test, driven by
frozen_manifest.csv (bundled next to this script), two independent ways:
  1. by identity: frozen speaker IDs + frozen (noise_source, noise_source_file);
  2. by content: SHA-256 of every selected noise file must not match a frozen noise hash.

Also asserts: 28 VoiceBank-train speakers, no frozen speaker present, SESA train
extraction == 340 clips, ESC folds 4-5 fully excluded, all 9 lists nonempty,
speech sets disjoint across splits.

Layout inputs (override with env):
  WORK_ROOT      default: ~/ps26052-gpu (Kaggle: /kaggle/working/ps26052-gpu)
  VOICEBANK_DIR  default: $WORK_ROOT/raw/voicebank   (extracted clean_trainset_28spk)
  SESA_ZIP       default: $WORK_ROOT/raw/SESA.zip
  ESC_DIR        default: $WORK_ROOT/raw/ESC-50       (cloned repo, has meta/esc50.csv + audio/)
  FROZEN_MANIFEST default: <this script dir>/frozen_manifest.csv
Outputs: $WORK_ROOT/lists/{train,valid,test}_{speech,sesa_noise,esc_noise}.txt + split_manifest.csv

Exit 0 on success; nonzero with a plain-English reason on any gate failure.
"""

import csv
import hashlib
import os
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

SESA_SHA256 = "fdc368afb948052933c6a4cb45449ef19dce6bae43aac929111a4daba5d653ec"
SESA_WANTED = {"gunshot", "explosion", "siren"}
ESC_WANTED = {"helicopter", "engine", "wind", "fireworks", "siren"}


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def load_frozen(manifest_path):
    rows = list(csv.DictReader(open(manifest_path, newline="", encoding="utf-8")))
    if len(rows) != 280:
        raise ValueError("frozen manifest must have 280 rows, got %d" % len(rows))
    speakers = {r["speech_file"].split("_")[0] for r in rows}
    noise_ids = {(r["noise_source"], r["noise_source_file"]) for r in rows}
    noise_hashes = {r["noise_source_sha256"] for r in rows}
    return speakers, noise_ids, noise_hashes


def build(work_root, voicebank_dir, esc_dir, frozen_manifest, out_lists):
    frozen_speakers, frozen_noise_ids, frozen_noise_hashes = load_frozen(frozen_manifest)
    print("frozen speakers:", sorted(frozen_speakers))
    print("frozen noise files:", len(frozen_noise_ids))

    manifest = []
    files = {(s, k): [] for s in ("train", "valid", "test")
             for k in ("speech", "sesa_noise", "esc_noise")}

    def add(path, split, kind, source_id):
        path = Path(path).resolve()
        if not path.is_file():
            raise ValueError("missing file: %s" % path)
        files[(split, kind)].append(str(path))
        manifest.append((split, kind, source_id, str(path), sha256_file(path)))

    # ---- speech: VoiceBank train, by speaker ----
    speech = sorted(Path(voicebank_dir).rglob("*.wav"))
    if not speech:
        raise ValueError("no VoiceBank WAVs in %s" % voicebank_dir)
    speakers = sorted({p.stem.split("_")[0] for p in speech})
    leak = set(speakers) & frozen_speakers
    if leak:
        raise ValueError("FROZEN test speaker in training archive: %s" % sorted(leak))
    if len(speakers) != 28:
        raise ValueError("expected 28 training speakers, got %d" % len(speakers))
    order = sorted(speakers)
    speaker_split = {s: ("train" if i < 22 else "valid" if i < 25 else "test")
                     for i, s in enumerate(order)}
    for p in speech:
        add(p, speaker_split[p.stem.split("_")[0]], "speech", p.stem.split("_")[0])

    # ---- SESA: hash-verified zip, train/ only, wanted classes ----
    # Dir mode (SESA_DIR, no zip): same selection/exclusion/content-hash gates on extracted
    # files. Zip-hash authenticity is replaced by private-dataset provenance + the 340-clip
    # count + frozen-member/content gates below - stated, not silent.
    sesa_zip = Path(os.environ.get("SESA_ZIP", str(Path(work_root) / "raw" / "SESA.zip")))
    sesa_dir = Path(os.environ.get("SESA_DIR", str(Path(work_root) / "raw" / "sesa")))
    sesa_out = Path(work_root) / "raw" / "sesa_train"
    if sesa_zip.is_file():
        actual = sha256_file(sesa_zip)
        if actual != SESA_SHA256:
            raise ValueError("SESA.zip hash mismatch: %s" % actual)
        sesa_out.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(sesa_zip) as archive:
            names = [n for n in archive.namelist()
                     if n.startswith("SESA/train/") and n.endswith(".wav")
                     and Path(n).name.split("_", 1)[0] in SESA_WANTED]
            if len(names) != 340:
                raise ValueError("expected 340 SESA train clips, got %d" % len(names))
            for n in names:
                if ("sesa", n) in frozen_noise_ids or n in {f for (_, f) in frozen_noise_ids}:
                    raise ValueError("FROZEN noise file in SESA train selection: %s" % n)
                (sesa_out / Path(n).name).write_bytes(archive.read(n))
        print("SESA training clips: %d (zip, hash-verified)" % len(names))
    elif sesa_dir.is_dir():
        inner = sesa_dir
        if not (sesa_dir / "train").is_dir():
            found = sorted(p for p in sesa_dir.rglob("train") if p.is_dir())
            if not found:
                raise ValueError("SESA dir has no train/ (%s)" % sesa_dir)
            inner = found[0].parent
            print("SESA wrapper detected, using %s" % inner)
        names = [p for p in sorted((inner / "train").rglob("*.wav"))
                 if p.name.split("_", 1)[0] in SESA_WANTED]
        if len(names) != 340:
            raise ValueError("expected 340 SESA train clips, got %d in %s"
                             % (len(names), inner / "train"))
        for p in names:
            rel = "SESA/train/" + p.name
            if ("sesa", rel) in frozen_noise_ids or rel in {f for (_, f) in frozen_noise_ids}:
                raise ValueError("FROZEN noise file in SESA train selection: %s" % p)
        sesa_out = inner / "train"
        print("SESA training clips: %d (extracted dir, structural gates)" % len(names))
    else:
        raise ValueError("need SESA.zip or SESA_DIR (structurally valid)")
    sesa = defaultdict(list)
    for p in sorted(sesa_out.rglob("*.wav")):
        if p.name.split("_", 1)[0] not in SESA_WANTED:
            continue
        sesa[p.stem.split("_", 1)[0]].append(p)
    if sum(map(len, sesa.values())) != 340:
        raise ValueError("SESA extraction count drift")
    for category, paths in sorted(sesa.items()):
        paths = sorted(paths)
        n_train = round(len(paths) * 0.70)
        n_valid = round(len(paths) * 0.15)
        for i, p in enumerate(paths):
            split = "train" if i < n_train else "valid" if i < n_train + n_valid else "test"
            add(p, split, "sesa_noise", p.stem)

    # ---- ESC-50: folds 1-2 train, fold 3 split by src_file, folds 4-5 excluded ----
    esc_meta = Path(esc_dir) / "meta" / "esc50.csv"
    esc_audio = Path(esc_dir) / "audio"
    esc_rows = list(csv.DictReader(esc_meta.open(newline="", encoding="utf-8")))
    fold3 = defaultdict(lambda: defaultdict(list))
    for row in esc_rows:
        if row["category"] not in ESC_WANTED:
            continue
        fname = row["filename"]
        if ("esc50", fname) in frozen_noise_ids:
            continue  # frozen clip: never usable, whatever its fold
        fold = int(row["fold"])
        path = esc_audio / fname
        if fold in (1, 2):
            add(path, "train", "esc_noise", row["src_file"])
        elif fold == 3:
            fold3[row["category"]][row["src_file"]].append(path)
        # folds 4-5: excluded entirely
    for category, groups in sorted(fold3.items()):
        for i, source_id in enumerate(sorted(groups)):
            split = "valid" if i % 2 == 0 else "test"
            for path in groups[source_id]:
                add(path, split, "esc_noise", source_id)

    # ---- content-hash gate: no selected noise file may match frozen noise bytes ----
    for (split, kind), paths in files.items():
        if "noise" not in kind:
            continue
        for p in paths:
            if sha256_file(p) in frozen_noise_hashes:
                raise ValueError("CONTENT MATCH with frozen noise: %s" % p)

    # ---- disjointness + nonempty gates ----
    for (split, kind), paths in sorted(files.items()):
        if not paths:
            raise ValueError("empty list: %s/%s" % (split, kind))
        print(split, kind, len(paths))
    train_sp = set(files[("train", "speech")])
    if train_sp & set(files[("valid", "speech")]):
        raise ValueError("speech overlap train/valid")
    if train_sp & set(files[("test", "speech")]):
        raise ValueError("speech overlap train/test")

    out_lists = Path(out_lists)
    out_lists.mkdir(parents=True, exist_ok=True)
    for (split, kind), paths in sorted(files.items()):
        (out_lists / ("%s_%s.txt" % (split, kind))).write_text("\n".join(paths) + "\n",
                                                               encoding="utf-8")
    with (out_lists / "split_manifest.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["split", "kind", "source_id", "absolute_path", "sha256"])
        w.writerows(manifest)
    print("wrote", out_lists / "split_manifest.csv", "rows:", len(manifest))
    return {"lists": {k: len(v) for k, v in files.items()}, "manifest_rows": len(manifest)}


def main():
    script_dir = Path(__file__).resolve().parent
    work_root = Path(os.environ.get("WORK_ROOT", str(Path.home() / "ps26052-gpu")))
    voicebank_dir = Path(os.environ.get("VOICEBANK_DIR", str(work_root / "raw" / "voicebank")))
    esc_dir = Path(os.environ.get("ESC_DIR", str(work_root / "raw" / "ESC-50")))
    frozen_manifest = Path(os.environ.get("FROZEN_MANIFEST", str(script_dir / "frozen_manifest.csv")))
    out_lists = Path(os.environ.get("OUT_LISTS", str(work_root / "lists")))
    try:
        build(work_root, voicebank_dir, esc_dir, frozen_manifest, out_lists)
    except Exception as err:
        print("SPLIT BUILD FAILED: %s" % err)
        return 1
    print("SPLIT BUILD OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
