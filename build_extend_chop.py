#!/usr/bin/env python3
"""Fold newly-read Table VII crops into the multi-page stitch structure:
add their page numbers to page_reads_fixed, and attach each to the existing
chop group of the same volume whose page range it fits (else a new per-volume
group). Then assembly stitches them with their siblings."""
import os, re, csv, json
from collections import defaultdict

csv.field_size_limit(10 ** 7)
ROOT = os.path.dirname(os.path.abspath(__file__))
V1 = os.path.join(ROOT, "export_final/sendout_final/sendout_v1")
IMG = re.compile(r"([0-9a-f]{40})")
DOC = re.compile(r"([0-9a-f]{16})_t\d+")


def main():
    # imghash -> keeper table_id + dochash
    by_img = {}
    for r in csv.DictReader(open(os.path.join(ROOT, "export/keeper_inference/source_catalog.csv"))):
        tid = r["table_id"]
        m = IMG.search(r.get("source_image", "") or tid)
        dh = DOC.search(tid)
        if m:
            by_img[m.group(1)] = (tid, dh.group(1) if dh else "")

    reads = json.load(open(os.path.join(V1, "newcrop_pagereads.json")))
    page_reads = json.load(open(os.path.join(V1, "page_reads_fixed.json")))
    have = {r["table_id"] for r in page_reads}
    groups = json.load(open(os.path.join(V1, "chop_groups.json")))

    # index groups by dochash (Table VII-ish only), with their page span
    gby = defaultdict(list)
    for g in groups:
        if re.search(r"VII|civil|age.{0,3}sex", g.get("subject", ""), re.I):
            pgs = g.get("pages", [])
            g["_lo"], g["_hi"] = (min(pgs), max(pgs)) if pgs else (0, 0)
            gby[g["dochash"]].append(g)

    added_reads = 0
    attached = 0
    newgroups = defaultdict(list)   # dochash -> [(tid, page)]
    for rd in reads:
        if not (rd.get("page_visible") and str(rd.get("page_number", "")).isdigit()):
            continue
        info = by_img.get(rd["imghash"])
        if not info:
            continue
        tid, dh = info
        page = int(rd["page_number"])
        if tid not in have:
            page_reads.append({"table_id": tid, "dochash": dh, "page_number": str(page),
                               "page_visible": True, "table_number": rd.get("table_number", ""),
                               "family": "age_civil_condition", "newcrop": True})
            have.add(tid); added_reads += 1
        # attach to the group whose span contains/abuts this page
        cands = gby.get(dh, [])
        best = None
        for g in cands:
            if g["_lo"] - 3 <= page <= g["_hi"] + 3:
                d = 0 if g["_lo"] <= page <= g["_hi"] else min(abs(page - g["_lo"]), abs(page - g["_hi"]))
                if best is None or d < best[0]:
                    best = (d, g)
        if best and tid not in best[1]["member_table_ids"]:
            best[1]["member_table_ids"].append(tid)
            best[1].setdefault("pages", []).append(page)
            attached += 1
        elif not best:
            newgroups[dh].append((tid, page))

    # create new groups for clusters with no existing match
    created = 0
    for dh, items in newgroups.items():
        if len(items) < 1:
            continue
        items.sort(key=lambda x: x[1])
        groups.append({"dochash": dh, "subject": "TABLE VII (new pages)", "table_number": "VII",
                       "anchor_index": 0, "confidence": "medium",
                       "member_table_ids": [t for t, _ in items], "pages": [p for _, p in items],
                       "n": len(items), "newgroup": True})
        created += 1

    json.dump(page_reads, open(os.path.join(V1, "page_reads_fixed.json"), "w"))
    json.dump(groups, open(os.path.join(V1, "chop_groups.json"), "w"))
    print(f"new crops mapped to keeper table_ids: {sum(1 for r in reads if by_img.get(r['imghash']))}")
    print(f"  page reads added: {added_reads}")
    print(f"  attached to existing Table VII groups: {attached}")
    print(f"  new groups created: {created}")


if __name__ == "__main__":
    main()
