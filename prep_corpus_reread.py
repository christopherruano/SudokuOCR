#!/usr/bin/env python3
"""Candidates for re-reading the corpus-wide imperfect tables in NON-Table-VII
families (education, occupation, caste, age-distribution, etc.). These have
varied column structures, so the re-read is GENERIC: we give Opus the table's
current column headers + issue, and it re-extracts rows x columns. Writes
corpus_reread_candidates.json."""
import os, re, csv, json, sqlite3
import census_explorer as ce

V1 = os.path.dirname(ce.DB)
VII = re.compile(r"\bVII\b|age,?\s*sex.{0,8}civil|civil\s*condition", re.I)
AGEFAM = ("age_civil_condition", "age_sex_civil_condition")


def main():
    imp = {r["table_id"]: r for r in csv.DictReader(open(os.path.join(V1, "corpus_imperfect.csv")))}
    con = sqlite3.connect(f"file:{ce.DB}?mode=ro", uri=True); con.row_factory = sqlite3.Row
    info = {r["table_id"]: r for r in con.execute(
        "SELECT table_id,title,year,canonical_name,table_family,measure_class,source_image,member_images FROM tables").fetchall()}

    cands = []
    for tid, row in imp.items():
        t = info.get(tid)
        if not t:
            continue
        # skip Table VII civil-condition (handled by the dedicated re-read)
        if t["table_family"] in AGEFAM and VII.search(t["title"] or ""):
            continue
        raw = (t["member_images"] or t["source_image"] or "")
        scans = []
        for p in re.split(r"[;,|]", raw):
            loc = ce.resolve_img(p.strip()) if p.strip() else None
            if loc and loc not in scans:
                scans.append(loc)
        if not scans:
            continue
        cols = [r["column_group_std"] for r in con.execute(
            "SELECT DISTINCT column_group_std FROM facts WHERE table_id=? AND column_group_std!='' LIMIT 16", (tid,)).fetchall()]
        cands.append({"idx": len(cands), "table_id": tid, "family": t["table_family"],
                      "measure_class": t["measure_class"], "issues": row["issues"],
                      "title": (t["title"] or "")[:90], "year": t["year"], "province": t["canonical_name"],
                      "scans": scans[:4], "current_columns": cols})
    json.dump(cands, open(os.path.join(V1, "corpus_reread_candidates.json"), "w"))
    from collections import Counter
    print(f"corpus (non-Table-VII) re-read candidates: {len(cands)}")
    print("  by family:", dict(Counter(c["family"] for c in cands).most_common(8)))


if __name__ == "__main__":
    main()
