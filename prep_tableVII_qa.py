#!/usr/bin/env python3
"""Stratified visual-QA sample of Table VII: confirm 'perfect' tables truly match
their scans, and that 'imperfect' flags are real. Resolve scan + compact matrix
per sampled table. Writes tableVII_qa_candidates.json."""
import os, re, csv, json, sqlite3, hashlib
from collections import defaultdict
import census_explorer as ce

ROOT = os.path.dirname(os.path.abspath(__file__))
V1 = os.path.dirname(ce.DB)
VII = re.compile(r"\bVII\b|age,?\s*sex\s*(and|&)?\s*civil|civil\s*condition", re.I)
AGEFAM = ("age_civil_condition", "age_sex_civil_condition", "age_sex_population")


def sample(items, n):
    return sorted(items, key=lambda t: hashlib.md5(t.encode()).hexdigest())[:n]


def main():
    imperfect = {r["table_id"]: r["issues"] for r in csv.DictReader(open(os.path.join(V1, "tableVII_imperfect.csv")))}
    con = sqlite3.connect(f"file:{ce.DB}?mode=ro", uri=True); con.row_factory = sqlite3.Row
    tabs = {r["table_id"]: r for r in con.execute(
        "SELECT table_id,title,year,canonical_name,status,table_family,source_image,member_images FROM tables "
        "WHERE status IN ('confirmed_sent','recovered','keeper_accepted','keeper_review','assembled')").fetchall()
        if r["table_family"] in AGEFAM and VII.search(r["title"] or "")}
    perfect_ids = [t for t in tabs if t not in imperfect]
    imperfect_ids = [t for t in tabs if t in imperfect]

    pick = sample(perfect_ids, 30) + sample(imperfect_ids, 20)
    cands = []
    for tid in pick:
        t = tabs[tid]
        raw = (t["member_images"] or t["source_image"] or "")
        scans = []
        for p in re.split(r"[;,|]", raw):
            loc = ce.resolve_img(p.strip()) if p.strip() else None
            if loc and loc not in scans:
                scans.append(loc)
        if not scans:
            continue
        facts = con.execute("SELECT row_label_std,column_group_std,sub_column_std,value FROM facts WHERE table_id=? LIMIT 400", (tid,)).fetchall()
        byrow = defaultdict(list); cols = []
        for f in facts:
            c = f"{(f['column_group_std'] or '')[:10]}/{(f['sub_column_std'] or '')[:6]}"
            byrow[(f["row_label_std"] or "")[:18]].append(f"{c}={f['value']}")
            if c not in cols: cols.append(c)
        lines = ["cols: " + ", ".join(cols[:10])]
        for i, (rl, vs) in enumerate(byrow.items()):
            if i >= 30: break
            lines.append(f"[{rl}] " + " ".join(vs[:10]))
        cands.append({"idx": len(cands), "table_id": tid, "expected": "imperfect" if tid in imperfect else "perfect",
                      "flagged_issues": imperfect.get(tid, ""), "year": t["year"], "province": t["canonical_name"],
                      "title": (t["title"] or "")[:80], "scans": scans[:4], "matrix": "\n".join(lines)})
    json.dump(cands, open(os.path.join(V1, "tableVII_qa_candidates.json"), "w"))
    from collections import Counter
    print(f"Table VII QA candidates: {len(cands)}  ({dict(Counter(c['expected'] for c in cands))})")


if __name__ == "__main__":
    main()
