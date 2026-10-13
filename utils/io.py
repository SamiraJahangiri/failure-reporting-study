"""Small JSON/JSONL and immutable-output helpers shared by the pipelines."""

import hashlib
import json
import os
from pathlib import Path
import tempfile


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path):
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"Blank JSONL record: {path}:{number}")
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Expected an object: {path}:{number}")
            rows.append(row)
    return rows


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_bytes(path, content, overwrite=True):
    """Publish a complete file; exclusive mode never replaces an existing file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        if overwrite:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def atomic_write_json(path, value, overwrite=True):
    content = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    atomic_write_bytes(path, content.encode("utf-8"), overwrite=overwrite)


def atomic_write_jsonl(path, rows, overwrite=True):
    content = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    atomic_write_bytes(path, content.encode("utf-8"), overwrite=overwrite)


def exclusive_write_json(path, value):
    atomic_write_json(path, value, overwrite=False)


def exclusive_write_jsonl(path, rows):
    atomic_write_jsonl(path, rows, overwrite=False)


def append_jsonl(path, row):
    """Append one durable journal record; the generator owns its journal file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = (json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    with os.fdopen(descriptor, "ab") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
