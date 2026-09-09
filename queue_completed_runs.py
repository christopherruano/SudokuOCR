#!/usr/bin/env python3
"""After a re-OCR batch finishes, append the affected RUNS to the BACK of the review queue — so you can
review the FULL assembled run with its fresh OCR (not just the single page) — and record them as
completed for the /grow_queue alert. Reads export/reocr_last_batch.txt (imghashes just processed).
"""
import sys, json, csv, os, datetime
sys.argv = ["x"]
import census_review_app as A

ROOT = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(ROOT, "export/tableVII_runs.json")
QUEUE = os.path.join(ROOT, "export_final/sendout_final/sendout_v1/tableVII_imperfect.csv")
BATCH = os.path.join(ROOT, "export/reocr_last_batch.txt")
COMPLETED = os.path.join(ROOT, "export/reocr_completed_runs.json")


def run_fails(run):
    """(representative table_id, fail count) — the run member with the most constraint failures."""
    rep, best = None, -1
    for m in run.get("members", []):
        d = A.load_table(m.get("table_id", ""))[0]
        if not d:
            continue
        f = sum(len(A.check_constraints(s.get("rows", []))) for s in d.get("data", []))
        if f > best:
            best, rep = f, m.get("table_id")
    return rep, max(best, 0)


def main():
    if not os.path.exists(BATCH):
        print("no batch file — nothing to queue"); return
    ihs = [x for x in open(BATCH).read().split() if x.strip()]
    runs = json.load(open(RUNS))
    seen = {}
    for ih in ihs:
        for r in runs:
            if any(ih in m.get("table_id", "") for m in r.get("members", [])):
                seen[f"{r['saoa']}:{r['page_range']}"] = r
                break
    if not os.path.exists(QUEUE):
        print("no queue file"); return
    rows = list(csv.DictReader(open(QUEUE)))
    cols = list(rows[0].keys()) if rows else ["table_id", "year", "province", "status", "issues", "title"]
    have = {r["table_id"] for r in rows}
    completed = {}
    if os.path.exists(COMPLETED):
        try:
            completed = json.load(open(COMPLETED))
        except Exception:
            completed = {}
    ts = datetime.datetime.now().strftime("%m-%d %H:%M")
    added = 0
    for key, run in seen.items():
        rep, fails = run_fails(run)
        if not rep:
            continue
        d = A.load_table(rep)[0]
        title = ((d.get("metadata", {}) or {}).get("title", "") or "")[:60] if d else ""
        completed[key] = {"saoa": str(run["saoa"]), "range": run["page_range"], "ts": ts,
                          "fails": fails, "table_id": rep, "title": title, "reviewed": False}
        if rep not in have:                              # append to the BACK (review the full run later)
            nr = {c: "" for c in cols}
            nr.update({"table_id": rep, "year": str(run.get("year", "")), "province": str(run.get("region", "")),
                       "status": "reocr_done", "issues": f"{fails} fails · re-OCR done · review full run", "title": title})
            rows.append(nr); have.add(rep); added += 1
    with open(QUEUE, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)
    json.dump(completed, open(COMPLETED, "w"), indent=1)
    print(f"appended {added} re-OCR'd run(s) to BACK of queue; {len(seen)} run(s) recorded completed")


if __name__ == "__main__":
    main()
