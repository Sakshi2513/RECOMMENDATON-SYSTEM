"""Paths and small helpers shared by the pipeline scripts.
Set INSPECTION_DATASET to the folder that holds the inspection_* folders (default D:/INSPECTION DATASET) and
BUILD_DIR for the outputs (default <INSPECTION_DATASET>/_built). Reports are never stored in this repository."""

import contextlib
import json
import os
from pathlib import Path

ROOT = Path(os.environ.get("INSPECTION_DATASET", r"D:\INSPECTION DATASET"))
OUT = Path(os.environ.get("BUILD_DIR", ROOT / "_built"))
with contextlib.suppress(OSError):  # input drive not available (for example when only importing)
    OUT.mkdir(parents=True, exist_ok=True)


def inspection_dirs():
    return sorted(p for p in ROOT.iterdir() if p.is_dir() and not p.name.startswith("_"))


def jl_write(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def jl_read(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]
