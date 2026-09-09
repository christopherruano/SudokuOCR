#!/usr/bin/env python3
"""Recover real PDF page + provenance for every crop image, using only string
hashing — the PDFs are deleted but we still know each volume's pdf_path,
resolved_start_page, and resolved_end_page from organized_manifest.jsonl.

The original rendering used  imghash = sha1(f"{resolved_pdf_path}:{page}").
So for each PNG we try every page in [start, end] and check which sha1 matches.
No PDFs needed.

Run on the cluster:
    cd ~/census_toc
    python3 cluster_extract_pdf_pages_v2.py \
        --root digitized_tables \
        --manifest organized_manifest.jsonl \
        --out crop_pdf_pages.csv
"""
import argparse
import csv
import hashlib
import json
import os
import re
from collections import defaultdict
from pathlib import Path

HEX = re.compile(r"([0-9a-f]{40})")
DT = re.compile(r"([0-9a-f]{16})_t(\d+)")


def collection_of(pdf: str) -> str:
    b = os.path.basename(pdf or "")
    low = b.lower()
    if "saoa" in low and "crl" in low:
        return "SAOA via CRL"
    if "saoa" in low:
        return "SAOA"
    if low.startswith("crl"):
        return "CRL"
    parent = os.path.basename(os.path.dirname(pdf or ""))
    return (b.split(".")[0] if "." in b else parent) or "unknown"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="digitized_tables")
    ap.add_argument("--manifest", default="organized_manifest.jsonl")
    ap.add_argument("--out", default="crop_pdf_pages.csv")
    ap.add_argument("--max-range-pad", type=int, default=10,
                    help="Pages to search beyond [start, end] in case ranges are tight")
    args = ap.parse_args()

    # 1) Build dochash -> [{pdf, start, end, year, province, title, topic}, ...]
    # Multiple entries per dochash possible (different tables in same volume).
    dh_entries: dict[str, list[dict]] = defaultdict(list)
    seen_pdfs: dict[str, str] = {}  # dochash -> pdf_path (any one)
    with open(args.manifest) as f:
        for line in f:
            row = json.loads(line)
            pdf = row.get("digitized_source_file", "")
            src_img = row.get("source_image", "")
            m = re.search(r"/digitized_tables/([0-9a-f]+)_t\d+_", src_img)
            if not (pdf and m):
                continue
            dh = m.group(1)
            qr = row.get("queue_row", {}) or {}
            dh_entries[dh].append({
                "pdf": pdf,
                "start": qr.get("resolved_start_page"),
                "end": qr.get("resolved_end_page"),
                "year": qr.get("year", ""),
                "province": qr.get("province", row.get("province", "")),
                "title": (qr.get("display_identifier") or qr.get("title") or "")[:120],
                "topic": qr.get("canonical_topic", ""),
                "table_index": qr.get("table_index"),
            })
            seen_pdfs[dh] = pdf
    print(f"Loaded {len(dh_entries)} dochashes from manifest")

    # 2) Walk all crop PNGs
    crops = []  # (imghash, dochash, table_index_str, tdir_name)
    for png in Path(args.root).glob("*/*.png"):
        b = png.name
        ih_m = HEX.search(b)
        dt_m = DT.search(str(png))
        if ih_m and dt_m:
            crops.append((ih_m.group(1), dt_m.group(1), dt_m.group(2), png.parent.name))
    print(f"Found {len(crops)} crop PNGs")

    # 3) For each dochash, build a sha1 -> page lookup over the union of all
    # resolved page ranges for tables in that document (with padding).
    print("Building hash-reversal index per document...")
    hash_to_page_per_doc: dict[str, dict[str, int]] = {}
    for dh, entries in dh_entries.items():
        pdf = seen_pdfs[dh]
        pages = set()
        for e in entries:
            s, ed = e.get("start"), e.get("end")
            if isinstance(s, int) and isinstance(ed, int) and s <= ed:
                lo = max(1, s - args.max_range_pad)
                hi = ed + args.max_range_pad
                pages.update(range(lo, hi + 1))
        if not pages:
            continue
        lookup = {}
        for p in pages:
            h = hashlib.sha1(f"{pdf}:{p}".encode("utf-8")).hexdigest()
            lookup[h] = p
        hash_to_page_per_doc[dh] = lookup

    # 4) Match each crop to a page via hash lookup
    rows = []
    n_matched = n_anchor = n_none = 0
    for imghash, dh, ti, tdir in crops:
        lookup = hash_to_page_per_doc.get(dh) or {}
        pp = lookup.get(imghash)
        # Pick a representative entry for provenance for this dochash
        # (prefer one whose table_index matches the crop's table_index)
        entry = None
        for e in dh_entries.get(dh, []):
            if str(e.get("table_index", "")).zfill(4) == str(ti).zfill(4):
                entry = e
                break
        if not entry and dh_entries.get(dh):
            entry = dh_entries[dh][0]
        if pp is not None:
            page_source = "hash_reversal"
            n_matched += 1
        elif entry and isinstance(entry.get("start"), int):
            pp = entry["start"]
            page_source = "table_anchor"
            n_anchor += 1
        else:
            pp = ""
            page_source = "unknown"
            n_none += 1
        meta = entry or {}
        src = meta.get("pdf", "")
        rows.append({
            "imghash": imghash, "dochash": dh, "table_index": ti,
            "pdf_page": pp, "page_source": page_source,
            "source_pdf": src, "collection": collection_of(src),
            "year": meta.get("year", ""), "province": meta.get("province", ""),
            "title": meta.get("title", ""),
            "table_start_page": meta.get("start", ""), "table_end_page": meta.get("end", ""),
        })

    rows.sort(key=lambda r: (str(r["dochash"]), r["pdf_page"] if isinstance(r["pdf_page"], int) else 1 << 30))

    cols = ["imghash", "dochash", "table_index", "pdf_page", "page_source",
            "source_pdf", "collection", "year", "province", "title",
            "table_start_page", "table_end_page"]
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    total = len(rows)
    print(f"\nWrote {args.out}: {total} crops")
    print(f"  hash_reversal: {n_matched} ({100*n_matched//max(1,total)}%)")
    print(f"  table_anchor:  {n_anchor}")
    print(f"  unknown:       {n_none}")
    distinct_docs = len({r["dochash"] for r in rows if r["dochash"]})
    collections = sorted({r["collection"] for r in rows if r["collection"]})
    print(f"  distinct volumes: {distinct_docs} | collections: {collections}")


if __name__ == "__main__":
    main()
