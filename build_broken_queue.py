#!/usr/bin/env python3
"""Build a severity-ranked review queue from the CORRECTED assembled runs. One row per run that has a
real problem — a constraint failure (M+F≠Persons, U+M+W≠Total) and/or a structural gap (too few age
bands). Worst-first, so the reviewer (using the new paginated /tables view) fixes the most-broken tables
first. Writes tableVII_imperfect.csv (backs up the old one), which the review app reads.
  python3 build_broken_queue.py
"""
import sys, csv, os, re
sys.argv = ["x"]
import census_review_app as A
import constraints_core

V1 = "export_final/sendout_final/sendout_v1"
OUT = os.path.join(V1, "tableVII_imperfect.csv")
# The M+F=P / U+M+W=Total constraints are COUNT identities — they don't hold for derived/proportional
# tables (per 1,000, percentages, subsidiary distributions), so those don't belong in the review queue.
# Gating is now by constraints_core.classify_kind (data_type + title override), which replaces the old
# DERIVED title regex — kind != "count" means the count-identity queue is the wrong venue.
# the unambiguous core count tables (where M+F=P / U+M+W=Total truly apply) — reviewed first
CORE = re.compile(r"\btable\s+(no\.?\s*)?(vii|c-?iii)\b|imperial\s+table\s+vii|"
                  r"age,?\s+sex,?\s+and\s+civil\s+condition", re.I)


def structural(rows):
    if not rows:
        return ["empty"]
    notes = []
    if len(rows) < 3:
        notes.append(f"few_age_bands:{len(rows)}")
    return notes


def main():
    runs = A.assembly_runs()
    out = []
    for run in runs:
        members = sorted(run.get("members", []), key=lambda m: m.get("page", 0))
        if not members:
            continue
        nfail, struct, title = 0, set(), ""
        frows, trows = 0, 0          # failing rows / total rows -> failure DENSITY
        for m in members:
            d, _, _ = A.load_table(m.get("table_id", ""))
            if not d:
                continue
            if not title:
                meta = d.get("metadata", {}) or {}
                title = (meta.get("title", "") or "")
                kind, _base = constraints_core.classify_kind(meta)
                if kind != "count":                 # derived/proportional table -> count identities N/A
                    nfail = -1
                    break
            for s in d.get("data", []):
                rows = s.get("rows", [])
                viol = A.check_constraints(rows)
                _, fails = A.constraint_report(rows)
                nfail += len(fails)
                frows += len(set(k[0] for k in viol))
                trows += len(rows)
                for note in structural(rows):
                    struct.add(note)
        if nfail < 0:               # derived/proportional table — constraints don't apply, skip
            continue
        if nfail == 0 and not struct:
            continue
        density = frows / trows if trows else 0.0
        garbled = density > 0.5     # most rows fail -> OCR shifted rows/cols, not spot-fixable
        issues = (f"{nfail} fails on {frows}/{trows} rows" if nfail else "")
        issues += (" · GARBLED—triage" if garbled else (" · spot-fixable" if nfail else ""))
        if struct:
            issues = (issues + "; " + ";".join(sorted(struct))).strip("; ")
        out.append({
            "table_id": members[0].get("table_id", ""), "year": run.get("year", ""),
            "province": run.get("region", ""), "status": "needs_review",
            "issues": issues, "title": title[:80], "_core": 1 if CORE.search(title) else 0,
            "_n": nfail, "_pages": len(members), "_range": f"{run['page_range'][0]}-{run['page_range'][1]}",
            "_garbled": 1 if garbled else 0,
        })
    # spot-fixable first (productive), core VII first within that, then severity; garbled triage last
    out.sort(key=lambda r: (r["_garbled"], -r["_core"], -r["_n"]))
    if os.path.exists(OUT):
        os.replace(OUT, OUT + ".prequeue")
    cols = ["table_id", "year", "province", "status", "issues", "title"]
    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in out:
            w.writerow({k: r[k] for k in cols})
    nc = sum(1 for r in out if r["_n"] > 0)
    print(f"review queue: {len(out)} broken tables ({nc} with constraint failures), worst-first -> {OUT}")
    for r in out[:12]:
        print(f"    {r['_n']:4d} fails · {r['_pages']}pp pp{r['_range']} · {r['issues'][:42]} · {r['title'][:32]!r}")


if __name__ == "__main__":
    main()
