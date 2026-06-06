#!/usr/bin/env python3
"""Prep every crop (existing + keeper) in multi-table documents for the Haiku
page-number pass: resolve each table_id's local scan image. Writes crops.json."""
import csv, json, os, glob, re
from collections import defaultdict

csv.field_size_limit(10 ** 7)
ROOT = os.path.dirname(os.path.abspath(__file__))
V1 = os.path.join(ROOT, "export_final/sendout_final/sendout_v1")

# image index (prefer original PNG which keeps full margins/page number)
IMG = {}
for pat in ("export/never_ocrd_images/**/*.png", "export/**/*.png", "organized_results/**/*.png", "census_site/img/*.jpg"):
    for p in glob.glob(os.path.join(ROOT, pat), recursive=True):
        IMG.setdefault(os.path.basename(p).rsplit(".", 1)[0], p)


def resolve(src):
    return IMG.get(os.path.basename(src or "").rsplit(".", 1)[0], "")


def main():
    # table_id -> source_image from both catalogs
    src = {}
    for r in csv.DictReader(open(os.path.join(ROOT, "export/keeper_inference/source_catalog.csv"))):
        src[r["table_id"]] = r["source_image"]
    for r in csv.DictReader(open(os.path.join(V1, "catalog_tables.csv"))):
        src.setdefault(r["table_id"], r["source_image"])

    prov = list(csv.DictReader(open(os.path.join(V1, "provenance.csv"))))
    bydoc = defaultdict(list)
    for r in prov:
        if r["dochash"]:
            bydoc[r["dochash"]].append(r)
    crops, noimg = [], 0
    for dh, ts in bydoc.items():
        if len(ts) < 2:   # only multi-table documents (where stitching matters)
            continue
        for r in ts:
            img = resolve(src.get(r["table_id"], ""))
            if not img:
                noimg += 1
                continue
            crops.append({"table_id": r["table_id"], "dochash": dh, "seq": r["table_seq"],
                          "family": r["family"], "img": img,
                          "page_known": r["page_number"], "title": r["title"]})
    json.dump(crops, open(os.path.join(V1, "crops.json"), "w"))
    print(f"crops in multi-table docs with a resolvable image: {len(crops):,}  (no image: {noimg:,})")
    from collections import Counter
    print("age-family crops:", sum(1 for c in crops if "age" in (c["family"] or "")))
    print("distinct documents:", len(set(c["dochash"] for c in crops)))


if __name__ == "__main__":
    main()
