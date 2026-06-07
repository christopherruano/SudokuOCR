#!/usr/bin/env python3
"""Prep the flagged keeper tables (hard + soft red-flags) for an Opus vision pass:
resolve each table's local source scan + build a compact matrix of what we
extracted, so Opus can compare scan-vs-extraction and verdict on completeness,
column alignment, and whether the flagged pattern is real or a bad read.
Writes opus_adjudicate_candidates.json.
"""
import csv, json, os, glob
from collections import defaultdict

csv.field_size_limit(10 ** 7)
ROOT = os.path.dirname(os.path.abspath(__file__))
V1 = os.path.join(ROOT, "export_final/sendout_final/sendout_v1")

# local image index (prefer the original PNG with full margins)
IMG = {}
for pat in ("export/never_ocrd_images/**/*.png", "export/digitized_tables_referenced/**/*.png",
            "export/**/*.png", "census_site/img/*.jpg"):
    for p in glob.glob(os.path.join(ROOT, pat), recursive=True):
        IMG.setdefault(os.path.basename(p).rsplit(".", 1)[0], p)


def resolve(src):
    return IMG.get(os.path.basename(src or "").rsplit(".", 1)[0], "")


def main():
    flagged = {}
    for r in csv.DictReader(open(os.path.join(V1, "keeper_bad_reads.csv"))):
        rs = r["reasons"]
        tier = "hard" if any(k in rs for k in ("full_zero", "single_value", "mostly_zero")) else \
               ("soft" if any(k in rs for k in ("constant_col", "near_constant")) else None)
        if tier:
            flagged[r["table_id"]] = {"reasons": rs, "detail": r["detail"], "tier": tier,
                                      "title": r["title"], "year": r["year"], "place": r["place"]}

    # source image per keeper table
    src = {r["table_id"]: r["source_image"] for r in
           csv.DictReader(open(os.path.join(ROOT, "export/keeper_inference/source_catalog.csv")))}

    # collect cells for flagged tables (stream the big facts file once)
    cells = defaultdict(list)
    for r in csv.DictReader(open(os.path.join(V1, "keeper_facts.csv"))):
        t = r["table_id"]
        if t in flagged:
            sub = r["sub_column_std"] or r["sex"] or "val"
            cells[t].append((r.get("section", ""), r.get("row_label_std", "") or r.get("resolved_age_label_std", ""),
                             r.get("column_group_std", ""), sub, r.get("value", "")))

    def matrix_text(rows):
        # compact grid: row label -> "col/sub=value" list, capped
        byrow = defaultdict(list)
        cols = []
        for sec, rl, cg, sub, val in rows:
            key = (sec[:18], rl[:22])
            colname = f"{cg[:10]}/{sub[:8]}"
            byrow[key].append(f"{colname}={val}")
            if colname not in cols:
                cols.append(colname)
        lines = [f"columns({len(cols)}): " + ", ".join(cols[:12])]
        for i, (k, vals) in enumerate(byrow.items()):
            if i >= 40:
                lines.append(f"... (+{len(byrow)-40} more rows)"); break
            lines.append(f"[{k[0]}|{k[1]}] " + " ".join(vals[:12]))
        return "\n".join(lines)

    cands, noimg = [], 0
    for t, info in flagged.items():
        img = resolve(src.get(t, ""))
        if not img:
            noimg += 1
            continue
        cands.append({"table_id": t, "img": img, "tier": info["tier"], "reasons": info["reasons"],
                      "detail": info["detail"], "title": info["title"], "year": info["year"],
                      "place": info["place"], "n_cells": len(cells[t]),
                      "extracted_matrix": matrix_text(cells[t])})

    json.dump(cands, open(os.path.join(V1, "opus_adjudicate_candidates.json"), "w"))
    from collections import Counter
    print(f"flagged tables: {len(flagged)}   with resolvable scan: {len(cands)}   (no image: {noimg})")
    print("  by tier:", dict(Counter(c["tier"] for c in cands)))
    print("wrote opus_adjudicate_candidates.json")


if __name__ == "__main__":
    main()
