#!/usr/bin/env python3
"""List all source-document folder names from the cluster's digitized_tables.

Each folder is one extracted table; its name is <doc_hash>_t<table_index>_<title_slug>.
Tables sharing the same <doc_hash> came from the same source PDF.

Run on the cluster:
    python3 list_source_folders_cluster.py

Outputs to ~/census_toc/provenance/ :
    source_folders.txt          plain list of every table dir name
    source_folders.csv          (dir_name, doc_hash, table_index, title_slug)
    source_documents.csv        one row per doc_hash with table count
"""
from __future__ import annotations

import csv
import os
import re
from collections import defaultdict
from pathlib import Path

HOME = Path(os.path.expanduser("~"))
SRC_DIR = HOME / "census_toc" / "digitized_tables"
OUT_DIR = HOME / "census_toc" / "provenance"

DIR_RE = re.compile(r"^(?P<doc_hash>[0-9a-f]+)_t(?P<table_idx>\d+)_(?P<title_slug>.*)$")


def main() -> None:
    if not SRC_DIR.exists():
        raise SystemExit(f"Not found: {SRC_DIR}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    folders = sorted(p.name for p in SRC_DIR.iterdir() if p.is_dir())
    print(f"Found {len(folders)} table folders in {SRC_DIR}")

    (OUT_DIR / "source_folders.txt").write_text("\n".join(folders) + "\n", encoding="utf-8")

    rows: list[dict[str, str]] = []
    by_doc: dict[str, list[str]] = defaultdict(list)
    for name in folders:
        m = DIR_RE.match(name)
        if m:
            doc_hash = m.group("doc_hash")
            row = {"dir_name": name, "doc_hash": doc_hash,
                   "table_index": m.group("table_idx"), "title_slug": m.group("title_slug")}
        else:
            doc_hash = ""
            row = {"dir_name": name, "doc_hash": "", "table_index": "", "title_slug": ""}
        rows.append(row)
        by_doc[doc_hash].append(name)

    with (OUT_DIR / "source_folders.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["dir_name", "doc_hash", "table_index", "title_slug"])
        w.writeheader()
        w.writerows(rows)

    doc_rows = [
        {"doc_hash": h, "n_tables": str(len(names)), "table_dirs": ";".join(sorted(names))}
        for h, names in sorted(by_doc.items())
    ]
    with (OUT_DIR / "source_documents.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["doc_hash", "n_tables", "table_dirs"])
        w.writeheader()
        w.writerows(doc_rows)

    print(f"  -> {OUT_DIR / 'source_folders.txt'} ({len(folders)} folders)")
    print(f"  -> {OUT_DIR / 'source_folders.csv'}")
    print(f"  -> {OUT_DIR / 'source_documents.csv'} ({len(doc_rows)} unique source documents)")


if __name__ == "__main__":
    main()
