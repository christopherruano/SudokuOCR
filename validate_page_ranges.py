#!/usr/bin/env python3
"""Validate + TIGHTEN the Table VII OCR page ranges. The doc-level pdf_page_span
(e.g. 37-177) is far too broad — it re-covers pages we already have. Instead:
  1. per volume, learn the printed->PDF page OFFSET from tables where we have BOTH
     a printed page (page_reads) and a resolved PDF page (manifest),
  2. map each MISSING printed page to its exact PDF page (printed + offset),
  3. report coverage: which volumes have a reliable offset (ranges will work),
     which don't (uncertain), and how many missing pages we still need to read.
Output: tableVII_ocr_targets.csv (volume, pdf, exact PDF pages to OCR, confidence).
"""
import os, re, json, csv, statistics
from collections import defaultdict

csv.field_size_limit(10 ** 7)
ROOT = os.path.dirname(os.path.abspath(__file__))
V1 = os.path.join(ROOT, "export_final/sendout_final/sendout_v1")
DT = re.compile(r"([0-9a-f]{16})_t(\d+)")
DOC = re.compile(r"([0-9a-f]{16})_t")


def main():
    # table_id -> printed page (from our reads)
    printed = {}
    for r in json.load(open(os.path.join(V1, "page_reads_fixed.json"))):
        p = str(r.get("page_number", ""))
        if r.get("page_visible") and p.isdigit():
            printed[r["table_id"]] = int(p)
    # dochash_tNNNN -> resolved PDF start page (from the TOC/manifest)
    pdfpage = {}; doc_pdf = {}; doc_span = {}
    for line in open(os.path.join(ROOT, "export/cluster_sync/organized_manifest.jsonl")):
        try:
            j = json.loads(line)
        except Exception:
            continue
        m = DT.search(j.get("source_image", "") or j.get("result_json", "") or "")
        if not m:
            continue
        key = f"{m.group(1)}_t{m.group(2)}"; dh = m.group(1); qr = j.get("queue_row", {}) or {}
        s = qr.get("resolved_start_page")
        if isinstance(s, int):
            pdfpage[key] = s
            doc_pdf[dh] = (qr.get("filename", "") or doc_pdf.get(dh, ""))
            sp = doc_span.setdefault(dh, [10**9, -1]); sp[0] = min(sp[0], s); sp[1] = max(sp[1], qr.get("resolved_end_page", s))

    # per-volume printed->PDF offset from paired tables
    offsets = defaultdict(list)
    for tid, pr in printed.items():
        m = DT.search(tid)
        if not m:
            continue
        key = f"{m.group(1)}_t{m.group(2)}"
        if key in pdfpage:
            offsets[m.group(1)].append(pdfpage[key] - pr)
    offset = {dh: round(statistics.median(v)) for dh, v in offsets.items() if v}

    # gap docs + missing printed pages
    gaps = {}
    for r in csv.DictReader(open(os.path.join(V1, "newocr_neil_pages.csv"))):
        if r["missing_printed_pages"]:
            gaps[r["dochash"]] = {"pdf": r["source_pdf"], "tables": r["tables"],
                                  "missing": [int(x) for x in r["missing_printed_pages"].split(",")]}

    rows = []; ok = unc = 0; tot_pages = 0
    for dh, g in gaps.items():
        off = offset.get(dh)
        sp = doc_span.get(dh, [None, None])
        if off is not None and sp[0] and sp[0] < 10**9:
            # map printed -> PDF, keep only those inside the doc's resolved span (sanity)
            pdf_pages = sorted({p + off for p in g["missing"] if sp[0] <= p + off <= sp[1] + 2})
            conf = "high" if len(offsets[dh]) >= 3 else "medium"
            ok += 1
        elif sp[0] and sp[0] < 10**9:
            # no offset (too few reads) -> fall back to the TOC-resolved Table VII PDF span
            pdf_pages = list(range(sp[0], sp[1] + 1))
            conf = "span_fallback"
            unc += 1
        else:
            pdf_pages = []
            conf = "no_range"
            unc += 1
        tot_pages += len(pdf_pages)
        rows.append({"dochash": dh, "source_pdf": g["pdf"], "tables": g["tables"],
                     "offset": off if off is not None else "", "confidence": conf,
                     "n_missing_printed": len(g["missing"]), "n_pdf_pages": len(pdf_pages),
                     "pdf_pages_to_ocr": ",".join(map(str, pdf_pages))})

    rows.sort(key=lambda r: -r["n_pdf_pages"])
    with open(os.path.join(V1, "tableVII_ocr_targets.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dochash", "source_pdf", "tables", "offset", "confidence",
                                          "n_missing_printed", "n_pdf_pages", "pdf_pages_to_ocr"])
        w.writeheader(); [w.writerow(r) for r in rows]

    print(f"gap volumes: {len(gaps)}")
    print(f"  with a reliable printed->PDF offset (ranges will work): {ok}")
    print(f"  NO offset (no paired data — ranges uncertain, need manual/TOC check): {unc}")
    print(f"  tight PDF pages to OCR (vs the broad doc-spans): {tot_pages}")
    from collections import Counter
    print("  confidence:", dict(Counter(r["confidence"] for r in rows)))
    print("\nsample (tight targets):")
    for r in rows[:6]:
        print(f"  {r['source_pdf']:24} off={r['offset']!s:4} {r['n_missing_printed']:3} missing -> {r['n_pdf_pages']:3} PDF pages [{r['confidence']}]")
    print("wrote tableVII_ocr_targets.csv")


if __name__ == "__main__":
    main()
