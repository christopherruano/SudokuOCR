#!/usr/bin/env python3
"""Apply prime-fix re-reads (export/prime_fix_reads/*.json), KEEPING THE BEST version per table:
overlay each read onto its corpus _oneshot.json, recompute fail-rows, and write only when the count
strictly decreases. Backs up to _oneshot.preprimefix.json, sets metadata.human_verified=false and
metadata.primefix_applied=true (plus metadata.source_limited when the read flags it). Modeled on
apply_composite_reads.py. Never worsens a table; safe to re-run."""
import sys, json, glob, shutil, copy, os; sys.argv = ["x"]
sys.path.insert(0, ".")
import census_review_app as A

applied, skipped = 0, []
improved = []
total_before = total_after = 0
for rf in sorted(glob.glob("export/prime_fix_reads/*.json")):
    r = json.load(open(rf))
    ih = r["imghash"]
    hits = glob.glob(f"export/ocr_results_referenced/digitized_tables/*/{ih}_oneshot.json")
    if not hits:
        skipped.append((ih[:8], "no corpus file")); continue
    lp = hits[0]
    d = json.load(open(lp))
    current = A.table_fail_rows(d)
    nd = copy.deepcopy(d)
    ok = True
    for si, sec in enumerate(r["sections"]):
        if si >= len(nd["data"]):
            ok = False; break
        old = nd["data"][si]["rows"]
        rh = next((k for k in old[0] if not isinstance(old[0].get(k), dict)), None) if old else None
        new = sec["rows"]
        if len(new) != len(old):
            ok = False; break
        for i, nrow in enumerate(new):
            rebuilt = {rh: old[i].get(rh, "")} if rh else {}
            for g, v in nrow.items():
                if isinstance(v, dict):
                    rebuilt[g] = v
            old[i] = rebuilt
    if not ok:
        skipped.append((ih[:8], "section/row shape mismatch vs corpus")); continue
    after = A.table_fail_rows(nd)
    total_before += current; total_after += min(after, current)
    # KEEP THE BEST: apply only if this re-read is strictly better than the current corpus version
    if after < current:
        bak = lp.replace("_oneshot.json", "_oneshot.preprimefix.json")
        if not os.path.exists(bak):
            shutil.copy(lp, bak)
        md = nd.setdefault("metadata", {})
        md["human_verified"] = False
        md["primefix_applied"] = True
        md["primefix_fails"] = after
        if r.get("source_limited"):
            md["source_limited"] = True
        json.dump(nd, open(lp, "w"))
        applied += 1
        improved.append((ih[:8], current, after))
    else:
        skipped.append((ih[:8], f"no improvement ({current} -> {after})"))

print(f"kept-best prime-fix applies: {applied}; skipped: {len(skipped)}")
print(f"fail-rows over touched tables: {total_before} -> {total_after}")
for x in sorted(improved, key=lambda x: x[1] - x[2], reverse=True):
    print(f"  {x[0]}: {x[1]} -> {x[2]} fails")
for ih, why in skipped:
    print(f"  SKIP {ih}: {why}")
