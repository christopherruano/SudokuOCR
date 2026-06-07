#!/usr/bin/env python3
"""Universe check: which MISSING pages do we actually have a scan for?
Collect every un-read pool image (never_ocrd + referenced, not yet page-read)
that belongs to a gap-bearing document, tag it with that document's missing
pages, and write universe_candidates.json for a targeted Haiku page-read."""
import json, glob, re, os, csv
from collections import defaultdict

csv.field_size_limit(10 ** 7)
ROOT = "/Users/chrisruano/Research/SudokuOCR"
V1 = os.path.join(ROOT, "export_final/sendout_final/sendout_v1")
DOCH = re.compile(r"([0-9a-f]{16})_t\d+")
base = lambda p: os.path.basename(p).rsplit(".", 1)[0]

# pool images
pool = {}
for r in ("export/never_ocrd_images", "export/digitized_tables_referenced"):
    for p in glob.glob(os.path.join(ROOT, r, "**/*.png"), recursive=True) + \
             glob.glob(os.path.join(ROOT, r, "**/*.jpg"), recursive=True):
        pool[base(p)] = p

# already page-read
crops = json.load(open(os.path.join(V1, "crops.json")))
read_ids = set(r["table_id"] for r in json.load(open(os.path.join(V1, "page_reads_fixed.json"))))
captured = set(base(c["img"]) for c in crops if c["table_id"] in read_ids)

# gap docs -> missing pages
gaps, docinfo = {}, {}
for r in csv.DictReader(open(os.path.join(V1, "page_gaps.csv"))):
    if r["missing_pages"]:
        gaps[r["dochash"]] = set(int(x) for x in r["missing_pages"].split(","))
        docinfo[r["dochash"]] = {"region": r["region"], "year": r["year"], "pdf": r["source_pdf"]}

cands = []
for b, p in pool.items():
    if b in captured:
        continue
    m = DOCH.search(p)
    if not m or m.group(1) not in gaps:
        continue
    dh = m.group(1)
    cands.append({"img": p, "dochash": dh, "imghash": b,
                  "missing_pages": sorted(gaps[dh]),
                  "region": docinfo[dh]["region"], "year": docinfo[dh]["year"]})

json.dump(cands, open(os.path.join(V1, "universe_candidates.json"), "w"))
print(f"un-read candidate images in gap docs: {len(cands)}")
print(f"  across {len(set(c['dochash'] for c in cands))} documents")
print("wrote universe_candidates.json")
