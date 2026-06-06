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
    reads = {r["table_id"]: r for r in load("page_reads.json")}
    prov = {r["table_id"]: r for r in csv.DictReader(open(os.path.join(V1, "provenance.csv")))}
    # doc-level provenance
    docinfo = {}
    for r in prov.values():
        if r["dochash"]:
            docinfo.setdefault(r["dochash"], {"pdf": r["source_pdf"], "year": r["year"], "region": r["region"]})

    # group captured (table_id) by document, with page + table_number
    bydoc = defaultdict(list)
    for tid, rd in reads.items():
        if not rd.get("page_visible") or not str(rd.get("page_number", "")).isdigit():
            continue
        dh = (prov.get(tid) or {}).get("dochash", "")
        if dh:
            bydoc[dh].append({"tid": tid, "page": int(rd["page_number"]),
                              "table": rd.get("table_number", "") or "?", "section": rd.get("section", "")})

    # --- stitched logical tables: (doc, table_number) ordered by page ---
    stitched = []
    gaps_rows = []
    for dh, items in bydoc.items():
        info = docinfo.get(dh, {})
        all_pages = sorted(set(i["page"] for i in items))
        doc_lo, doc_hi = (all_pages[0], all_pages[-1]) if all_pages else (0, 0)
        doc_missing = [p for p in range(doc_lo, doc_hi + 1) if p not in set(all_pages)]
        bytab = defaultdict(list)
        for i in items:
            bytab[i["table"]].append(i)
        for tn, its in bytab.items():
            pages = sorted(set(i["page"] for i in its))
            lo, hi = pages[0], pages[-1]
            # gaps for THIS table: missing within its span, excluding pages owned by other tables
            other = set(i["page"] for i in items if i["table"] != tn)
            miss = [p for p in range(lo, hi + 1) if p not in set(pages) and p not in other]
            stitched.append({"dochash": dh, "source_pdf": (info.get("pdf") or "").split("/")[-1],
                             "year": info.get("year", ""), "region": info.get("region", ""),
                             "table_number": tn, "n_pages": len(pages),
                             "page_range": f"{lo}-{hi}", "pages": ",".join(map(str, pages)),
                             "missing_pages": ",".join(map(str, miss))})
        gaps_rows.append({"dochash": dh, "source_pdf": (info.get("pdf") or "").split("/")[-1],
                          "year": info.get("year", ""), "region": info.get("region", ""),
                          "captured_pages": len(all_pages), "page_range": f"{doc_lo}-{doc_hi}",
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
