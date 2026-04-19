"""Shared helpers for the TOC-manifest Slurm workflow."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Iterable, List, Optional


ROMAN_MAP = {
    "i": 1,
    "v": 5,
    "x": 10,
    "l": 50,
    "c": 100,
    "d": 500,
    "m": 1000,
}


def discover_pdfs(root: Path) -> List[Path]:
    root = Path(root)
    return sorted(p.resolve() for p in root.rglob("*.pdf") if not p.name.startswith("._"))


def load_pdf_filelist(path: Path) -> List[Path]:
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            rows.append(Path(line).resolve())
    return rows


def write_filelist(root: Path, output: Path) -> int:
    pdfs = discover_pdfs(root)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as fh:
        for pdf in pdfs:
            fh.write(str(pdf) + "\n")
    return len(pdfs)


def write_json_atomic(path: Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", delete=False, dir=str(path.parent), suffix=".tmp") as tmp:
        json.dump(payload, tmp, indent=2)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_path = Path(tmp.name)
    tmp_path.replace(path)


def is_valid_json_output(path: Path) -> bool:
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        with open(path) as fh:
            data = json.load(fh)
    except Exception:
        return False
    return isinstance(data, dict) and data.get("status") in {"ok", "error"}


def stable_doc_id(path: Path) -> str:
    return hashlib.sha1(str(Path(path).resolve()).encode("utf-8")).hexdigest()[:16]


def safe_slug(text: str, max_len: int = 80) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-")
    if not text:
        return "item"
    return text[:max_len]


def parse_page_label(label: Optional[str]) -> Optional[int]:
    if label is None:
        return None
    raw = str(label).strip().strip("()[]")
    if not raw:
        return None
    if raw.isdigit():
        return int(raw)

    roman = raw.lower()
    if not re.fullmatch(r"[ivxlcdm]+", roman):
        return None

    total = 0
    prev = 0
    for ch in reversed(roman):
        value = ROMAN_MAP[ch]
        if value < prev:
            total -= value
        else:
            total += value
            prev = value
    return total if total > 0 else None


def load_json(path: Path) -> dict:
    with open(path) as fh:
        return json.load(fh)


def jsonl_iter(path: Path) -> Iterable[dict]:
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)
