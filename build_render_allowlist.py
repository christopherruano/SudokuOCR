#!/usr/bin/env python3
"""Pick a deployable subset of tables to render (Cloudflare caps deployments at 20k files).
Priority: ALL relevant Age/Sex/Civil-Condition (Table VII) tables first, then the best general
tables by QA status + fact count, capped at CAP pages so pages+images stay < 20k.
The overflow (lowest-priority extras) is written to static_exclude.txt so build_static_site skips them.
  python3 build_render_allowlist.py [CAP]
"""
import sqlite3, re, os, sys, csv

ROOT = os.path.dirname(os.path.abspath(__file__))
V1 = os.path.join(ROOT, "export_final/sendout_final/sendout_v1")
DB = os.path.join(V1, "census_explorer.sqlite")
EXCL = os.path.join(V1, "static_exclude.txt")
CAP = int(sys.argv[1]) if len(sys.argv) > 1 else 12000
SHOWN = ("confirmed_sent", "recovered", "keeper_accepted", "keeper_review", "assembled")
VII = re.compile(r"civil\s*condition|age,?\s*sex|marital|conjugal|age\s*and\s*sex|sex\s*and\s*age|table\s*vii\b", re.I)
RELFAM = {"age_civil_condition", "age_sex_civil_condition", "age_by_religion", "age_civil_condition_religion",
          "age_sex_marital_status", "age_sex_population", "age_civil_condition_distribution"}
STATUS_RANK = {"keeper_accepted": 0, "confirmed_sent": 0, "recovered": 1, "assembled": 1, "keeper_review": 2}


def main():
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    rows = con.execute("SELECT table_id, title, table_family, status, n_facts, canonical_topic "
                       "FROM tables WHERE status IN (%s)" % ",".join("?" * len(SHOWN)), SHOWN).fetchall()

    def relevant(r):
        return bool(VII.search(r["title"] or "")) or (r["table_family"] in RELFAM) or \
               ("civil_condition" in (r["canonical_topic"] or ""))

    scored = []
    for r in rows:
        rel = relevant(r)
        scored.append((0 if rel else 1, STATUS_RANK.get(r["status"], 3), -(r["n_facts"] or 0),
                       r["table_id"], rel))
    scored.sort()
    keep = set(s[3] for s in scored[:CAP])
    rel_total = sum(1 for s in scored if s[4])
    rel_kept = sum(1 for s in scored[:CAP] if s[4])
    overflow = [s[3] for s in scored[CAP:]]

    # merge with any existing manual excludes, write back
    existing = set()
    if os.path.exists(EXCL):
        existing = {l.strip() for l in open(EXCL) if l.strip()}
    allexcl = existing | set(overflow)
    # don't exclude anything we're keeping
    allexcl -= keep
    with open(EXCL, "w") as fh:
        fh.write("\n".join(sorted(allexcl)) + "\n")

    print(f"total shown tables: {len(rows)} | relevant (Table VII/age-sex-civil): {rel_total}")
    print(f"CAP={CAP} -> rendering {len(keep)} (relevant kept: {rel_kept}/{rel_total}, general: {len(keep)-rel_kept})")
    print(f"excluded (overflow + manual): {len(allexcl)} -> {EXCL}")
    print(f"est files: ~{len(keep)} pages + ~{int(len(keep)*0.65)} images = ~{int(len(keep)*1.65)} (<20000?)")


if __name__ == "__main__":
    main()
