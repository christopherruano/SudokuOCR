#!/usr/bin/env python3
"""Stitch multi-page tables by (document, table-number) ordered by page, and
report per-PDF page gaps. Then cross-check the missing pages against our image
universe (do we HAVE that page's scan somewhere, just un-placed/un-OCR'd?).

Inputs (sendout_v1/):
  page_reads.json   [{table_id, page_number, page_visible, table_number, section}]
  provenance.csv    (dochash, source_pdf, year, region per table)
  crops.json        (crops with resolvable images)
Outputs:
  stitched_tables.csv   one row per logical multi-page table (doc + table#),
                        page range, ordered pages, missing pages
  page_gaps.csv         per document: captured pages, gaps, gap-in-universe?
"""
import csv, json, os, re, glob
from collections import defaultdict

csv.field_size_limit(10 ** 7)
ROOT = os.path.dirname(os.path.abspath(__file__))
V1 = os.path.join(ROOT, "export_final/sendout_final/sendout_v1")


def load(name):
    return json.load(open(os.path.join(V1, name)))


def main():
    reads = {r["table_id"]: r for r in load("page_reads_fixed.json")}
    prov = {r["table_id"]: r for r in csv.DictReader(open(os.path.join(V1, "provenance.csv")))}
    PAGE_MAX = 1200          # sane upper bound for a census volume page number
    SEG_SPLIT = 12           # split a table's pages into separate logical instances at gaps > this
    # doc-level provenance
    docinfo = {}
    for r in prov.values():
        if r["dochash"]:
            docinfo.setdefault(r["dochash"], {"pdf": r["source_pdf"], "year": r["year"], "region": r["region"]})

    # group captured (table_id) by document, with sanitized arabic page + table_number
    bydoc = defaultdict(list)
    for tid, rd in reads.items():
        pg = str(rd.get("page_number", "")).strip()
        if not rd.get("page_visible") or not pg.isdigit():
            continue
        p = int(pg)
        if not (1 <= p <= PAGE_MAX):     # drop OCR outliers (e.g. "10008")
            continue
        dh = (prov.get(tid) or {}).get("dochash", "")
        if dh:
            bydoc[dh].append({"tid": tid, "page": p,
                              "table": (rd.get("table_number", "") or "?").strip().upper(),
                              "section": rd.get("section", "")})

    # --- stitched logical tables: (doc, table_number) split into contiguous runs ---
    stitched = []
    gaps_rows = []
    for dh, items in bydoc.items():
        info = docinfo.get(dh, {})
        bytab = defaultdict(list)
        for i in items:
            bytab[i["table"]].append(i)
        doc_gap_pages = set()           # union of interior gaps across this doc's tables
        doc_captured = set(i["page"] for i in items)
        for tn, its in bytab.items():
            pages = sorted(set(i["page"] for i in its))
            other = set(i["page"] for i in items if i["table"] != tn)
            # segment into contiguous instances: split where consecutive captured pages jump > SEG_SPLIT
            seg, segs = [pages[0]], []
            for p in pages[1:]:
                if p - seg[-1] > SEG_SPLIT:
                    segs.append(seg); seg = [p]
                else:
                    seg.append(p)
            segs.append(seg)
            for sp in segs:
                lo, hi = sp[0], sp[-1]
                miss = [p for p in range(lo, hi + 1) if p not in set(sp) and p not in other]
                doc_gap_pages |= set(miss)
                stitched.append({"dochash": dh, "source_pdf": (info.get("pdf") or "").split("/")[-1],
                                 "year": info.get("year", ""), "region": info.get("region", ""),
                                 "table_number": tn, "n_pages": len(sp),
                                 "page_range": f"{lo}-{hi}", "pages": ",".join(map(str, sp)),
                                 "missing_pages": ",".join(map(str, miss))})
        doc_missing = sorted(doc_gap_pages)
        gaps_rows.append({"dochash": dh, "source_pdf": (info.get("pdf") or "").split("/")[-1],
                          "year": info.get("year", ""), "region": info.get("region", ""),
                          "captured_pages": len(doc_captured),
                          "page_range": f"{min(doc_captured)}-{max(doc_captured)}" if doc_captured else "-",
                          "n_missing": len(doc_missing), "missing_pages": ",".join(map(str, doc_missing))})

    with open(os.path.join(V1, "stitched_tables.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dochash", "source_pdf", "year", "region", "table_number",
                                          "n_pages", "page_range", "pages", "missing_pages"])
        w.writeheader(); [w.writerow(r) for r in sorted(stitched, key=lambda x: (-x["n_pages"]))]
    with open(os.path.join(V1, "page_gaps.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dochash", "source_pdf", "year", "region",
                                          "captured_pages", "page_range", "n_missing", "missing_pages"])
        w.writeheader(); [w.writerow(r) for r in sorted(gaps_rows, key=lambda x: -x["n_missing"])]

    tot_missing = sum(r["n_missing"] for r in gaps_rows)
    multi = [s for s in stitched if s["n_pages"] > 1]
    print(f"stitched logical tables: {len(stitched)}  (multi-page: {len(multi)})")
    print(f"documents with page gaps: {sum(1 for r in gaps_rows if r['n_missing'])} / {len(gaps_rows)}")
    print(f"total missing page-slots (within captured ranges): {tot_missing}")
    print("wrote stitched_tables.csv, page_gaps.csv")


if __name__ == "__main__":
    main()
