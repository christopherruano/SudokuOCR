#!/usr/bin/env python3
"""Pilot: verify accepted Opus re-reads against the scans. For a sample of
corrected tables, show Opus the SCAN + the NEW (re-read) matrix + the OLD
(original) matrix, and ask which matches the scan and whether the re-read fixed
the problem areas. Writes pilot_verify_candidates.json."""
import os, re, csv, json, sqlite3, hashlib
from collections import defaultdict
import census_explorer as ce

V1 = os.path.dirname(ce.DB)


def compact(cells):
    by = defaultdict(list); cols = []
    for sec, row, cg, sub, val in cells:
        c = f"{cg[:8]}/{sub[:3]}"
        by[(sec[:12], row[:12])].append(f"{c}={val}")
        if c not in cols:
            cols.append(c)
    lines = ["cols: " + ", ".join(cols[:10])]
    for i, (k, v) in enumerate(by.items()):
        if i >= 16:
            break
        lines.append(f"[{k[0]}|{k[1]}] " + " ".join(v[:10]))
    return "\n".join(lines)


def main():
    corrected = json.load(open(os.path.join(V1, "corrected_tables.json")))["corrected"]
    sample = sorted(corrected, key=lambda t: hashlib.md5(t.encode()).hexdigest())[:16]
    sset = set(sample)
    # new cells (corrected_facts) + old cells (keeper_facts) for the sample
    new = defaultdict(list); old = defaultdict(list)
    for r in csv.DictReader(open(os.path.join(V1, "corrected_facts.csv"))):
        if r["table_id"] in sset:
            new[r["table_id"]].append((r.get("section", ""), r.get("row_label_std", ""),
                                       r.get("column_group_std", ""), r.get("sub_column_std", ""), r.get("value", "")))
    for r in csv.DictReader(open(os.path.join(V1, "keeper_facts.csv"))):
        if r["table_id"] in sset:
            old[r["table_id"]].append((r.get("section", ""), r.get("row_label_std", ""),
                                       r.get("column_group_std", ""), r.get("sub_column_std", ""), r.get("value", "")))
    # scans the Opus re-read actually used (guaranteed resolvable)
    scan_by_tid = {c["table_id"]: c["scans"] for c in json.load(open(os.path.join(V1, "opus_reread_candidates.json")))}

    cands = []
    for tid in sample:
        scans = scan_by_tid.get(tid, [])[:3]
        if not scans:
            continue
        cands.append({"idx": len(cands), "table_id": tid,
                      "scans": scans, "new_matrix": compact(new.get(tid, [])),
                      "old_matrix": compact(old.get(tid, []))})
    json.dump(cands, open(os.path.join(V1, "pilot_verify_candidates.json"), "w"))
    print(f"pilot verify candidates: {len(cands)} (of {len(sample)} sampled accepted re-reads)")


if __name__ == "__main__":
    main()
