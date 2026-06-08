#!/usr/bin/env python3
"""Keeper/assembled catalog rows have empty or cluster-path source_image, so the
explorer/site can't show their scans. Resolve each to its LOCAL crop PNG via the
stable dochash_tNNNN prefix and rewrite source_image + member_images (page-ordered
for assembled). Then rebuild the DB + regenerate the site."""
import os, re, glob, csv, json
from collections import defaultdict

csv.field_size_limit(10 ** 7)
ROOT = os.path.dirname(os.path.abspath(__file__))
V1 = os.path.join(ROOT, "export_final/sendout_final/sendout_v1")
PREF = re.compile(r"([0-9a-f]{16}_t\d+)")


def pref(s):
    m = PREF.search(s or ""); return m.group(1) if m else ""


def main():
    # basename -> local path (SPECIFIC crop) ; prefix -> dir crops (fallback)
    base_idx = {}
    pref_idx = defaultdict(list)
    for p in glob.glob(os.path.join(ROOT, "export/**/*.png"), recursive=True):
        base_idx.setdefault(os.path.basename(p), p)
        m = PREF.search(p)
        if m:
            pref_idx[m.group(1)].append(p)
    for k in pref_idx:
        pref_idx[k] = sorted(set(pref_idx[k]))
    print(f"index: {len(base_idx)} PNGs by basename, {len(pref_idx)} table prefixes")

    # table_id -> its own source_image (specific crop), from the source catalogs
    src = {}
    for r in csv.DictReader(open(os.path.join(ROOT, "export/keeper_inference/source_catalog.csv"))):
        src[r["table_id"]] = r.get("source_image", "")
    for r in csv.DictReader(open(os.path.join(V1, "catalog_tables.csv"))):
        src.setdefault(r["table_id"], r.get("source_image", ""))

    def resolve(tid):
        """specific crop via the table's own source_image basename; else dir fallback."""
        b = os.path.basename(src.get(tid, "") or "")
        if b in base_idx:
            return base_idx[b]
        fb = pref_idx.get(pref(tid), [])
        return fb[0] if fb else ""

    # keeper_catalog: each table -> ITS crop
    p = os.path.join(V1, "keeper_catalog.csv")
    rows = list(csv.DictReader(open(p)))
    cols = rows[0].keys()
    spec = fb = 0
    for r in rows:
        b = os.path.basename(src.get(r["table_id"], "") or "")
        loc = base_idx.get(b) or (pref_idx.get(pref(r["table_id"]), [""])[0])
        if loc:
            r["source_image"] = loc; r["member_images"] = loc
            spec += (b in base_idx); fb += (b not in base_idx)
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); [w.writerow(r) for r in rows]
    print(f"keeper_catalog.csv: specific crop {spec}, dir-fallback {fb}")

    # assembled_catalog: ordered DISTINCT member crops (each member -> its own)
    members = json.load(open(os.path.join(V1, "assembled_members.json")))
    p = os.path.join(V1, "assembled_catalog.csv")
    rows = list(csv.DictReader(open(p)))
    cols = rows[0].keys()
    for r in rows:
        scans = []
        for m in members.get(r["table_id"], []):
            loc = resolve(m)
            if loc and loc not in scans:
                scans.append(loc)
        if scans:
            r["source_image"] = scans[0]
            r["member_images"] = ";".join(scans)
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); [w.writerow(r) for r in rows]
    avg = sum((r["member_images"].count(";") + 1) for r in rows) / max(1, len(rows))
    print(f"assembled_catalog.csv: {len(rows)} tables, avg {avg:.1f} distinct member scans each")


if __name__ == "__main__":
    main()
