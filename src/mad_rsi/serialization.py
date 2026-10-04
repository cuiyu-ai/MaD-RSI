"""Deterministic identities and atomic JSON checkpoints."""
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def fingerprint(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
    temp = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=path.name + ".", suffix=".tmp", delete=False) as handle:
            temp = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        for attempt in range(5):
            try:
                temp.replace(path)
                break
            except PermissionError:
                if attempt == 4:
                    raise
                time.sleep(.05)
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)
