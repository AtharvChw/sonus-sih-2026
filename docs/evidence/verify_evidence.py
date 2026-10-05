"""Verify preserved evidence bytes; does not rerun training or live tests."""
from pathlib import Path
import hashlib, json
root = Path(__file__).resolve().parent
rows = json.loads((root / "manifest.json").read_text(encoding="utf-8"))["files"]
assert rows, "Empty evidence manifest"
for row in rows:
    p = root / row["file"]
    assert p.stat().st_size == row["bytes"], row["file"]
    assert hashlib.file_digest(p.open("rb"), "sha256").hexdigest() == row["sha256"], row["file"]
print(f"EVIDENCE HASHES PASS: {len(rows)} files")