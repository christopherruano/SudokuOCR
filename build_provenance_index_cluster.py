#!/usr/bin/env python3
"""Build a provenance index on the FASRC cluster.

Walks ~/census_toc/ocr_results/digitized_tables/ and maps each OCR'd table
back to its source PDF using ~/census_toc/organized_manifest.jsonl.

Run on the cluster:
    python3 build_provenance_index_cluster.py

Outputs to ~/census_toc/provenance/ :
    tables_by_pdf.csv   page-level rows
    pdfs_grouped.csv    one row per source PDF
    orphans.csv         tables we couldn't map (newly OCR'd keepers, mostly)
"""
from __future__ import annotations

import csv
import json
import os
import re
from collections import defaultdict
from pathlib import Path

HOME = Path(os.path.expanduser("~"))
RESULTS_DIR = HOME / "census_toc" / "ocr_results" / "digitized_tables"
MANIFEST = HOME / "census_toc" / "organized_manifest.jsonl"
WORKLIST = HOME / "census_toc" / "ocr_worklist.csv"  # optional, scp from local
OUT_DIR = HOME / "census_toc" / "provenance"

DIR_RE = re.compile(r"^(?P<doc_hash>[0-9a-f]+)_t(?P<table_idx>\d+)_(?P<title_slug>.*)$")


def parse_dir_name(name: str) -> dict[str, str]:
    m = DIR_RE.match(name)
    if not m:
        return {"doc_hash": "", "table_idx": "", "title_slug": ""}
    return m.groupdict()


def load_manifest_provenance() -> dict[str, dict[str, str]]:
    index: dict[str, dict[str, str]] = {}
    if not MANIFEST.exists():
        print(f"  (no manifest at {MANIFEST})")
        return index
    with MANIFEST.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            src_img = row.get("source_image", "")
            m = re.search(r"/digitized_tables/([^/]+)/", src_img)
            if not m:
                continue
            doc_hash = parse_dir_name(m.group(1))["doc_hash"]
            if not doc_hash:
                continue
            pdf_path = row.get("digitized_source_file", "")
            index[doc_hash] = {
                "source_pdf": Path(pdf_path).name if pdf_path else "",
                "source_pdf_path": pdf_path,
                "year": str(row.get("year", "")),
                "province": row.get("province", ""),
            }
    return index


def load_worklist_provenance() -> dict[str, dict[str, str]]:
    index: dict[str, dict[str, str]] = {}
    if not WORKLIST.exists():
        print(f"  (no worklist at {WORKLIST} — skipping; orphans expected for keepers)")
        return index
    csv.field_size_limit(10_000_000)
    with WORKLIST.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            dir_name = row.get("dir", "")
            doc_hash = parse_dir_name(dir_name)["doc_hash"]
            if not doc_hash:
                continue
            index[doc_hash] = {
                "source_pdf": row.get("source_pdf", ""),
                "source_pdf_path": "",
                "year": row.get("year", ""),
                "province": "",
            }
    return index


def main() -> None:
    if not RESULTS_DIR.exists():
        raise SystemExit(f"OCR results dir not found: {RESULTS_DIR}")

    print(f"Walking {RESULTS_DIR}")
    manifest_idx = load_manifest_provenance()
    worklist_idx = load_worklist_provenance()
    print(f"  {len(manifest_idx)} doc hashes from manifest")
    print(f"  {len(worklist_idx)} doc hashes from worklist")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tables_rows: list[dict[str, str]] = []
    orphan_rows: list[dict[str, str]] = []
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)

    for table_dir in sorted(RESULTS_DIR.iterdir()):
        if not table_dir.is_dir():
            continue
        parsed = parse_dir_name(table_dir.name)
        doc_hash = parsed["doc_hash"]
        pages = sorted(p.stem.replace("_oneshot", "") for p in table_dir.glob("*_oneshot.json"))
        if not pages:
            continue
        prov = manifest_idx.get(doc_hash) or worklist_idx.get(doc_hash) or {}
        source = "manifest" if doc_hash in manifest_idx else ("worklist" if doc_hash in worklist_idx else "")

        for page_hash in pages:
            row = {
                "table_dir": table_dir.name,
                "doc_hash": doc_hash,
                "table_index": parsed["table_idx"],
                "title_slug": parsed["title_slug"],
                "page_hash": page_hash,
                "source_pdf": prov.get("source_pdf", ""),
                "source_pdf_path": prov.get("source_pdf_path", ""),
                "year": prov.get("year", ""),
                "province": prov.get("province", ""),
                "provenance_source": source,
            }
            tables_rows.append(row)
            if not prov:
                orphan_rows.append(row)
            key = prov.get("source_pdf") or f"UNKNOWN__{doc_hash or table_dir.name}"
            grouped[key].append(row)

    fieldnames = [
        "table_dir", "doc_hash", "table_index", "title_slug",
        "page_hash", "source_pdf", "source_pdf_path",
        "year", "province", "provenance_source",
    ]
    with (OUT_DIR / "tables_by_pdf.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(tables_rows)

    with (OUT_DIR / "orphans.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(orphan_rows)

    grouped_rows = []
    for pdf, rows in sorted(grouped.items()):
        years = sorted({r["year"] for r in rows if r["year"]})
        provinces = sorted({r["province"] for r in rows if r["province"]})
        doc_hashes = sorted({r["doc_hash"] for r in rows if r["doc_hash"]})
        grouped_rows.append({
            "source_pdf": pdf,
            "year": ";".join(years),
            "province": ";".join(provinces),
            "doc_hashes": ";".join(doc_hashes),
            "n_tables": str(len({r["table_dir"] for r in rows})),
            "n_pages": str(len(rows)),
        })

    with (OUT_DIR / "pdfs_grouped.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["source_pdf", "year", "province", "doc_hashes", "n_tables", "n_pages"])
        w.writeheader()
        w.writerows(grouped_rows)

    print(f"\nWrote {len(tables_rows)} page rows -> {OUT_DIR / 'tables_by_pdf.csv'}")
    print(f"Wrote {len(grouped_rows)} PDF rows -> {OUT_DIR / 'pdfs_grouped.csv'}")
    print(f"Orphans: {len(orphan_rows)} -> {OUT_DIR / 'orphans.csv'}")


if __name__ == "__main__":
    main()
