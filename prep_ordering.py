#!/usr/bin/env python3
"""Prep multi-page groups for a vision ordering check: each group's member tables
with scan image paths in their current (tNNNN) order, so agents can verify/fix order."""
import csv, json, os, glob

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
    return IMG.get(os.path.basename(src or "").rsplit(".", 1)[0], "")


def main():
    mp = json.load(open(os.path.join(V1, "multipage_groups.json")))
    cat = {r["table_id"]: r for r in csv.DictReader(open(os.path.join(V1, "catalog_tables.csv")))}
    po = mp["page_order"]
    out = []
    for head, members in mp["groups"].items():
        if len(members) < 2:
            continue
        ms = []
        for t in members:
            r = cat.get(t, {})
            ms.append({"table_id": t, "img": resolve(r.get("source_image", "")),
                       "title": (r.get("title") or "")[:110], "seq": po.get(t, {}).get("seq")})
        out.append({"group_head": head, "year": cat.get(head, {}).get("year", ""),
                    "place": cat.get(head, {}).get("canonical_name", ""), "members": ms})
    out.sort(key=lambda g: -len(g["members"]))
    json.dump(out, open(os.path.join(V1, "ordering_candidates.json"), "w"), indent=1)
    miss = sum(1 for g in out for m in g["members"] if not m["img"])
    print(f"multi-page groups to verify: {len(out)}  (member tables: {sum(len(g['members']) for g in out)}, no-image members: {miss})")
    print(f"largest groups: {[len(g['members']) for g in out[:6]]}")


if __name__ == "__main__":
    main()
