#!/usr/bin/env python3
"""Precompute a failure-SEVERITY score for every queue item so the review app can sort biggest/most
failures first without loading 450+ JSONs per request. severity = total_failure_magnitude + 30*n_fail_rows
(a linear combination of how BIG the discrepancies are and how MANY there are). Writes:
  export/severity_tables.json  {table_id: {n, mag, max, sev}}   (the /tables queue)
  export/severity_runs.json    {"saoa:[s,e]": {n, mag, sev}}    (the /assembly queue, aggregated)
Re-run after re-OCR / edits to refresh the ordering.
  python3 compute_severity.py
"""
import sys, os, json, re
sys.argv = ["x"]
import census_review_app as A

ROOT = os.path.dirname(os.path.abspath(__file__))
# the reingest writes the authoritative fail count into the issues column: "368 fails on 58/506 rows"
FAILS_RE = re.compile(r"(\d+)\s+fails on\s+(\d+)\s*/\s*(\d+)")


def main():
    # tables the reviewer marked "constraints N/A" (distribution/proportion/rate) — don't rank their fakes
    na = set()
    naf = os.path.join(ROOT, "export/constraints_irrelevant.json")
    if os.path.exists(naf):
        try:
            na = set(json.load(open(naf)))
        except Exception:
            na = set()
    import re as _re
    tbl = {}
    for r in A.imperfect_tables():
        tid = r.get("table_id", "")
        ihm = _re.search(r"([0-9a-f]{40})", tid)
        if ihm and ihm.group(1) in na:
            continue
        issues = r.get("issues") or r.get("category") or ""
        m = FAILS_RE.search(issues)
        iss_fails = int(m.group(1)) if m else 0          # failing identities per the (possibly stale) queue
        iss_rows = int(m.group(2)) if m else 0           # failing rows per the queue
        d, ih, _ = A.load_table(tid)
        # table_severity is now KIND-GATED (constraints_core): transposed layouts are checked properly
        # and proportional/rate tables no longer accrue fake count-identity failures — so when the JSON
        # loads, the LIVE recount is authoritative; the issues-column count is only a fallback for
        # unloadable reads. (The old max(iss, live) kept stale false failures alive forever.)
        if d:
            n_arith, mag, mx, _ = A.table_severity(d)
            n_rows, fails = n_arith, n_arith
        else:
            n_arith, mag, mx = 0, 0, 0
            n_rows, fails = iss_rows, iss_fails
        # severity = magnitude ("biggest", where measurable) + linear combo of failing rows & total fails
        sev = mag + 200 * n_rows + 50 * fails
        if sev or n_rows or fails:
            tbl[tid] = {"n": n_rows, "fails": fails, "mag": mag, "max": mx, "sev": sev, "issue": issues[:48]}
    json.dump(tbl, open(os.path.join(ROOT, "export/severity_tables.json"), "w"))

    runs = {}
    for run in A.assembly_runs():
        key = f"{run['saoa']}:{run['page_range']}"
        n = total = sev = 0
        for m in run.get("members", []):
            d, ih, _ = A.load_table(m.get("table_id", ""))
            if not d:
                continue
            nn, tt, mx, ss = A.table_severity(d)
            n += nn; total += tt; sev += ss
        runs[key] = {"n": n, "mag": total, "sev": sev}
    json.dump(runs, open(os.path.join(ROOT, "export/severity_runs.json"), "w"))

    top = sorted(tbl.items(), key=lambda kv: -kv[1]["sev"])[:8]
    print(f"severity computed: {len(tbl)} tables, {len(runs)} runs")
    print("top tables by severity (sev = mag + 30·n_fail):")
    for tid, s in top:
        print(f"  sev={s['sev']:6}  n_fail={s['n']:3}  total_mag={s['mag']:6}  max={s['max']:5}  {os.path.basename(tid)[:46]}")


if __name__ == "__main__":
    main()
