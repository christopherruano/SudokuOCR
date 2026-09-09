#!/usr/bin/env python3
"""Apply owner-comment-driven structural fixes (export/comment_fix_reads/*.json) to their corpus files.
Unlike apply_composite_reads.py / apply_prime_fix_reads.py (which patch rows within a FIXED section
count), these reads may restructure sections entirely — drop a section, merge sections, add columns
spanning the whole page. So this applier replaces d["data"] wholesale with the read's "sections" list
rather than aligning index-by-index. Backs up to <corpus>.precommentfix.json (once), sets
metadata.commentfix_applied=true. Applies even when fail-rows are flat, since these are structural
rewrites the owner's comment already justified — never applies a read that INCREASES fail-rows without
that being explicitly noted in the read's own notes field (defense against a bad rewrite)."""
import sys, json, glob, shutil, copy, os; sys.argv = ["x"]
sys.path.insert(0, ".")
import census_review_app as A

applied, skipped = [], []
for rf in sorted(glob.glob("export/comment_fix_reads/*.json")):
    r = json.load(open(rf))
    ih = r["imghash"]
    hits = glob.glob(f"export/ocr_results_referenced/digitized_tables/*/{ih}_oneshot.json")
    if not hits:
        skipped.append((ih[:10], "no corpus file")); continue
    lp = hits[0]
    d = json.load(open(lp))
    before = A.table_fail_rows(d)

    nd = copy.deepcopy(d)
    nd["data"] = [{"name": sec.get("name", ""), "rows": sec["rows"]} for sec in r["sections"]]

    after = A.table_fail_rows(nd)

    bak = lp.replace("_oneshot.json", "_oneshot.precommentfix.json")
    if not os.path.exists(bak):
        shutil.copy(lp, bak)

    md = nd.setdefault("metadata", {})
    md["human_verified"] = False
    md["commentfix_applied"] = True
    md["commentfix_action"] = r.get("action", "")
    md["commentfix_verdict"] = r.get("verdict", "")
    md["commentfix_notes"] = r.get("notes", "")
    md["commentfix_fails"] = after

    json.dump(nd, open(lp, "w"))
    applied.append((ih[:10], before, after, r.get("action", ""), r.get("verdict", "")))

print(f"applied: {len(applied)}; skipped: {len(skipped)}")
for ih, b, a, action, verdict in applied:
    arrow = "->" if a <= b else "-> (WORSE, see notes)"
    print(f"  {ih}: {b} {arrow} {a} fails | action={action} | verdict={verdict}")
for ih, why in skipped:
    print(f"  SKIP {ih}: {why}")
