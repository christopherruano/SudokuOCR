#!/usr/bin/env python3
"""Idempotent: create assembly runs in tableVII_runs.json for any CONFIRMED /documents range that has
none yet (skips saoas that already have runs; skips ranges with zero OCR'd members — they'll be created
on a later pass once the watcher lands their pages). Safe to re-run forever.
  python3 build_runs_from_confirmed.py
"""
import sys, json, csv, os, shutil
sys.argv = ["x"]
import boundary_apply as B

ROOT = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(ROOT, "export/tableVII_runs.json")


def ranges_of(v):
    if "ranges" in v:
        return [tuple(int(x) for x in r.split("-")) for r in v["ranges"]]
    if "range" in v and v.get("range"):
        return [tuple(v["range"])]
    return []


def main():
    runs = json.load(open(RUNS))
    have = {(str(r["saoa"]), tuple(r["page_range"])) for r in runs}
    have_saoa = {str(r["saoa"]) for r in runs}
    verd = json.load(open(os.path.join(ROOT, "export/review_verdicts/documents.json")))
    cov = {r["saoa"]: r for r in csv.DictReader(open(os.path.join(ROOT, "export/document_coverage.csv")))}
    added = 0
    for k, v in verd.items():
        sa = k.split(":")[1]
        if sa in have_saoa:
            continue                                     # doc already has run(s) — leave manual work alone
        c = cov.get(sa, {})
        dh = c.get("dochash", "")
        if not dh:
            continue
        for s, e in ranges_of(v):
            if (sa, (s, e)) in have:
                continue
            members, missing = [], []
            for pg in range(s, e + 1):
                m, has = B.member_for(dh, pg)
                (members.append({"page": pg, "table_id": m["table_id"]}) if (has and m) else missing.append(pg))
            if not members:
                continue                                 # nothing OCR'd yet — next pass
            runs.append({"saoa": sa, "dochash": dh, "page_range": [s, e], "members": members,
                         "n_pages": len(members), "region": c.get("region", ""), "year": c.get("year", ""),
                         "tnum": "VII", "start_type": "human_confirmed_documents",
                         "end_type": "human_confirmed_documents", "start_clean": True, "end_clean": True,
                         "flags": [], "bridged_pages": [], "missing_before": missing})
            added += 1
    if added:
        shutil.copy(RUNS, RUNS + ".autobuild.bak")
        json.dump(runs, open(RUNS, "w"), indent=1)
    print(f"run-builder: +{added} new runs (total {len(runs)})")


if __name__ == "__main__":
    main()
