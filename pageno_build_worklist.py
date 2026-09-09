#!/usr/bin/env python3
"""Build the page-number reading worklist for fixing assembled Table VII ordering.

For every assembled table (assembled_members.json) we collect its member table_ids,
extract the 40-hex imghash before `_oneshot`, resolve the local scan, and dedupe to
UNIQUE physical scans (one Gemini read per physical page, not per member row).

Each imghash also records every page_reads_fixed.json table_id that shares it, so the
read can later be applied to all matching entries (the page number is a property of the
physical scan, not of an individual member row).

Output: pageno_parts/worklist.json  -> [{imghash, scan_path, table_ids:[...], asm_ids:[...]}]
Also writes pageno_parts/worklist_stats.json for a quick summary.
"""
import json, re, glob, os, sys

ROOT = os.path.dirname(os.path.abspath(__file__))
V1 = os.path.join(ROOT, "export_final/sendout_final/sendout_v1")
PARTS = os.path.join(V1, "pageno_parts")
os.makedirs(PARTS, exist_ok=True)

HASH_RE = re.compile(r"([0-9a-f]{40})_oneshot")


def imghash(tid):
    m = HASH_RE.search(tid or "")
    return m.group(1) if m else None


def build_img_index():
    idx = {}
    for pat in ("export/**/*.png", "organized_results/**/*.png"):
        for p in glob.glob(os.path.join(ROOT, pat), recursive=True):
            idx.setdefault(os.path.basename(p), p)
    return idx


def main():
    members = json.load(open(os.path.join(V1, "assembled_members.json")))
    preads = json.load(open(os.path.join(V1, "page_reads_fixed.json")))

    # every page_reads table_id grouped by imghash (so we can apply reads to all of them)
    pr_by_hash = {}
    for e in preads:
        h = imghash(e["table_id"])
        if h:
            pr_by_hash.setdefault(h, []).append(e["table_id"])

    print(f"indexing local scans ...", flush=True)
    img_index = build_img_index()
    print(f"  {len(img_index):,} images indexed", flush=True)

    # walk all assembled tables, dedupe to unique imghash
    by_hash = {}  # imghash -> {scan_path, asm_ids:set, member_tids:set}
    n_members = n_resolved = n_unresolved = 0
    for asm_id, tids in members.items():
        for tid in tids:
            n_members += 1
            h = imghash(tid)
            scan = img_index.get((h + ".png")) if h else None
            if not scan:
                n_unresolved += 1
                continue
            n_resolved += 1
            rec = by_hash.setdefault(h, {"scan_path": scan, "asm_ids": set(), "member_tids": set()})
            rec["asm_ids"].add(asm_id)
            rec["member_tids"].add(tid)

    worklist = []
    for h, rec in by_hash.items():
        # apply target = union of member tids and ALL page_reads entries with this imghash
        targets = set(rec["member_tids"]) | set(pr_by_hash.get(h, []))
        worklist.append({
            "imghash": h,
            "scan_path": os.path.relpath(rec["scan_path"], ROOT),
            "table_ids": sorted(targets),
            "asm_ids": sorted(rec["asm_ids"]),
        })
    worklist.sort(key=lambda r: r["imghash"])

    out = os.path.join(PARTS, "worklist.json")
    json.dump(worklist, open(out, "w"), indent=0)
    stats = {
        "assembled_tables": len(members),
        "member_rows": n_members,
        "resolved_member_rows": n_resolved,
        "unresolved_member_rows": n_unresolved,
        "unique_scans": len(worklist),
    }
    json.dump(stats, open(os.path.join(PARTS, "worklist_stats.json"), "w"), indent=2)
    print(json.dumps(stats, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
