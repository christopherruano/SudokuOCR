#!/usr/bin/env python3
"""SLIM deploy: exclude every NON-relevant (non age/sex/civil-condition) table so build_static_site
renders ONLY the Table VII set + its images. Backs up the original static_exclude.txt so the FULL
build is restorable. Run, then re-run the normal render commands.
  python3 build_slim_exclude.py          # -> static_exclude.txt = original + all non-relevant
  python3 build_slim_exclude.py --full   # -> restore original excludes (back to FULL)
"""
import sqlite3, re, os, sys

ROOT = os.path.dirname(os.path.abspath(__file__))
V1 = os.path.join(ROOT, "export_final/sendout_final/sendout_v1")
DB = os.path.join(V1, "census_explorer.sqlite")
EXCL = os.path.join(V1, "static_exclude.txt")
ORIG = os.path.join(V1, "static_exclude.orig.txt")
SHOWN = ("confirmed_sent", "recovered", "keeper_accepted", "keeper_review", "assembled")
VII = re.compile(r"civil\s*condition|age,?\s*sex|marital|conjugal|age\s*and\s*sex|sex\s*and\s*age|table\s*vii\b", re.I)
RELFAM = {"age_civil_condition", "age_sex_civil_condition", "age_by_religion", "age_civil_condition_religion",
          "age_sex_marital_status", "age_sex_population", "age_civil_condition_distribution"}


def original_excludes():
    # the ORIGINAL not-yet-good excludes (backed up once); never lose them
    if os.path.exists(ORIG):
        return {l.strip() for l in open(ORIG) if l.strip()}
    cur = {l.strip() for l in open(EXCL) if l.strip()} if os.path.exists(EXCL) else set()
    open(ORIG, "w").write("\n".join(sorted(cur)) + "\n")   # snapshot current as the original
    return cur


def main():
    orig = original_excludes()
    if "--full" in sys.argv:
        open(EXCL, "w").write("\n".join(sorted(orig)) + "\n")
        print(f"restored FULL: static_exclude.txt = {len(orig)} original excludes only")
        return
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    rows = con.execute("SELECT table_id, title, table_family, canonical_topic FROM tables "
                       "WHERE status IN (%s)" % ",".join("?" * len(SHOWN)), SHOWN).fetchall()

    def relevant(r):
        return bool(VII.search(r["title"] or "")) or (r["table_family"] in RELFAM) or \
               ("civil_condition" in (r["canonical_topic"] or ""))

    nonrel = {r["table_id"] for r in rows if not relevant(r)}
    rel = sum(1 for r in rows if relevant(r))
    excl = orig | nonrel
    open(EXCL, "w").write("\n".join(sorted(excl)) + "\n")
    print(f"SLIM: keep {rel} relevant Table VII tables | exclude {len(nonrel)} non-relevant (+ {len(orig)} original)")
    print(f"static_exclude.txt now {len(excl)} entries -> re-render to get the slim site")


if __name__ == "__main__":
    main()
