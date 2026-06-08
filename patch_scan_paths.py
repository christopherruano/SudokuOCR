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
    idx = defaultdict(list)
    for pool in ("export/never_ocrd_images/digitized_tables", "export/digitized_tables_referenced/digitized_tables"):
        for d in glob.glob(os.path.join(ROOT, pool, "*")):
            if os.path.isdir(d):
                m = PREF.search(os.path.basename(d))
                ps = sorted(glob.glob(d + "/*.png"))
                if m and ps:
                    idx[m.group(1)].append(ps[0])           # one representative crop per table dir
    print(f"prefix->scan index: {len(idx)} table dirs")

    # keeper_catalog: source_image = local PNG
    for fn in ("keeper_catalog.csv",):
        p = os.path.join(V1, fn)
        rows = list(csv.DictReader(open(p)))
        cols = rows[0].keys()
        fixed = 0
        for r in rows:
            scans = idx.get(pref(r["table_id"]), [])
            if scans:
                r["source_image"] = scans[0]
                r["member_images"] = scans[0]
                fixed += 1
        with open(p, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); [w.writerow(r) for r in rows]
        print(f"{fn}: set local scan on {fixed}/{len(rows)}")

    # assembled_catalog: member_images = ordered member scans
    members = json.load(open(os.path.join(V1, "assembled_members.json")))
    p = os.path.join(V1, "assembled_catalog.csv")
    rows = list(csv.DictReader(open(p)))
    cols = rows[0].keys()
    fixed = 0
    for r in rows:
        mem = members.get(r["table_id"], [])
        scans = []
        for m in mem:
            for s in idx.get(pref(m), []):
                if s not in scans:
                    scans.append(s)
        if scans:
            r["source_image"] = scans[0]
            r["member_images"] = ";".join(scans)
            fixed += 1
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); [w.writerow(r) for r in rows]
    print(f"assembled_catalog.csv: set member scans on {fixed}/{len(rows)}")


if __name__ == "__main__":
    main()
