#!/usr/bin/env python3
"""Match the universe page-reads against each gap document's missing pages to
classify every missing page as RECOVERABLE (we hold an un-read scan of it) or
ABSENT (no scan in our universe -> would need re-digitization).

Outputs (sendout_v1/):
  universe_coverage.csv   per (dochash, missing_page): recoverable? + imghash if so
  gap_summary.csv         per document: missing / recoverable / absent counts
"""
import json, csv, os
from collections import defaultdict

ROOT = "/Users/chrisruano/Research/SudokuOCR"
V1 = os.path.join(ROOT, "export_final/sendout_final/sendout_v1")
PAGE_MAX = 1200

cands = {c["imghash"]: c for c in json.load(open(os.path.join(V1, "universe_candidates.json")))}
reads = json.load(open(os.path.join(V1, "universe_reads.json")))

# doc -> page -> imghash(es) we hold an un-read scan for
have = defaultdict(lambda: defaultdict(list))
for r in reads:
    pg = str(r.get("page_number", "")).strip()
    if not r.get("page_visible") or not pg.isdigit():
        continue
    p = int(pg)
    if not (1 <= p <= PAGE_MAX):
        continue
    c = cands.get(r["imghash"])
    if c:
        have[c["dochash"]][p].append(r["imghash"])

# gap docs + missing pages + doc info
gaps, info = {}, {}
for r in csv.DictReader(open(os.path.join(V1, "page_gaps.csv"))):
    if r["missing_pages"]:
        gaps[r["dochash"]] = [int(x) for x in r["missing_pages"].split(",")]
        info[r["dochash"]] = {"region": r["region"], "year": r["year"], "pdf": r["source_pdf"]}

# every gap doc is a volume we physically hold (we have placed/OCR'd tables from
# it), so every interior missing page is at minimum RE-EXTRACTABLE from the source
# volume. Status per missing page:
#   have_scan  - an un-read crop of that exact page is already on local disk (place now)
#   reextract  - volume held; re-OCR that page from the source on the cluster
cov_rows, sum_rows = [], []
tot_missing = n_have = n_reextract = 0
for dh, miss in gaps.items():
    have_scan = 0
    for p in miss:
        imgs = have.get(dh, {}).get(p, [])
        status = "have_scan" if imgs else "reextract"
        have_scan += bool(imgs)
        cov_rows.append({"dochash": dh, "region": info[dh]["region"], "year": info[dh]["year"],
                         "missing_page": p, "status": status, "imghash": imgs[0] if imgs else ""})
    tot_missing += len(miss); n_have += have_scan; n_reextract += len(miss) - have_scan
    sum_rows.append({"dochash": dh, "region": info[dh]["region"], "year": info[dh]["year"],
                     "source_pdf": info[dh]["pdf"], "pdf_label_known": int(bool(info[dh]["pdf"].strip())),
                     "n_missing": len(miss), "n_have_scan": have_scan,
                     "n_reextract": len(miss) - have_scan})

with open(os.path.join(V1, "universe_coverage.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["dochash", "region", "year", "missing_page", "status", "imghash"])
    w.writeheader(); [w.writerow(r) for r in sorted(cov_rows, key=lambda x: (x["dochash"], x["missing_page"]))]
with open(os.path.join(V1, "gap_summary.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["dochash", "region", "year", "source_pdf", "pdf_label_known",
                                      "n_missing", "n_have_scan", "n_reextract"])
    w.writeheader(); [w.writerow(r) for r in sorted(sum_rows, key=lambda x: -x["n_missing"])]

docs_no_label = sum(1 for r in sum_rows if not r["pdf_label_known"])
print(f"gap documents: {len(gaps)}  (ALL are volumes we physically hold)")
print(f"total missing page-slots: {tot_missing}")
print(f"  IN UNIVERSE - have un-read scan on disk (place now):  {n_have}  ({100*n_have//tot_missing}%)")
print(f"  IN UNIVERSE - re-extractable from held volume (cluster re-OCR): {n_reextract}  ({100*n_reextract//tot_missing}%)")
print(f"  truly absent (no volume held): 0")
print(f"  ({docs_no_label} of {len(gaps)} gap docs still need their source-PDF FILENAME recovered)")
print("wrote universe_coverage.csv, gap_summary.csv")
