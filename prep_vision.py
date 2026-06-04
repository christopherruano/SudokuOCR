#!/usr/bin/env python3
"""Prep the UMW-misread vision candidates: for each, the scan image path + the
CURRENT (suspected-wrong) matrix columns + sample rows, so agents can compare."""
import csv, json, os, glob
from collections import defaultdict

csv.field_size_limit(10 ** 7)
ROOT = os.path.dirname(os.path.abspath(__file__))
V1 = os.path.join(ROOT, "export_final/sendout_final/sendout_v1")

IMG = {}
for p in glob.glob(os.path.join(ROOT, "census_site/img/*.jpg")):
    IMG[os.path.basename(p)[:-4]] = p
for pat in ("export/**/*.png", "organized_results/**/*.png"):
    for p in glob.glob(os.path.join(ROOT, pat), recursive=True):
        IMG.setdefault(os.path.basename(p)[:-4], p)


def resolve(src):
    if not src:
        return ""
    return IMG.get(os.path.basename(src).rsplit(".", 1)[0], "")


def main():
    tids = set(json.load(open(os.path.join(V1, "umw_candidates.json"))))
    cat = {r["table_id"]: r for r in csv.DictReader(open(os.path.join(V1, "catalog_tables.csv")))}
    facts = defaultdict(list)
    for r in csv.DictReader(open(os.path.join(V1, "unified_facts.csv"))):
        if r["table_id"] in tids:
            facts[r["table_id"]].append(r)
    out = []
    noimg = 0
    for t in tids:
        r = cat.get(t, {})
        fs = facts.get(t, [])
        cols, rows = [], []
        seen_c, seen_r = set(), set()
        for f in fs:
            sub = f["sub_column_std"] or {"All": "persons", "Male": "males", "Female": "females"}.get(f["sex"], "")
            grp = f["column_group_std"] or ""
            ck = f"{grp}|{sub}".strip("|")
            if ck and ck not in seen_c:
                seen_c.add(ck); cols.append(ck)
            rk = f["row_label_std"] or ""
            if rk and rk not in seen_r and len(rows) < 4:
                seen_r.add(rk); rows.append(rk)
        img = resolve(r.get("source_image", ""))
        if not img:
            noimg += 1
        out.append({"table_id": t, "img": img, "title": (r.get("title") or "")[:120],
                    "measure_class": r.get("measure_class", ""), "year": r.get("year", ""),
                    "place": r.get("canonical_name", ""), "current_columns": cols[:12],
                    "sample_rows": rows, "n_cols": len(cols)})
    json.dump(out, open(os.path.join(V1, "vision_candidates.json"), "w"), indent=1)
    print(f"prepped {len(out)} candidates  ({noimg} with no local image)")


if __name__ == "__main__":
    main()
