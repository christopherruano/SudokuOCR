#!/usr/bin/env python3
"""Build Opus vision re-read candidates for imperfect Table VII tables: each gets
its local scan(s) + the expected schema (sibling template columns/age rows) so
Opus can extract the FULL grid. Validated downstream by U+M+W=Population.
Writes opus_reread_candidates.json."""
import os, re, csv, json, sqlite3
import census_explorer as ce

ROOT = os.path.dirname(os.path.abspath(__file__))
V1 = os.path.dirname(ce.DB)
IMG = re.compile(r"([0-9a-f]{40})")


def main():
    imp = {r["table_id"]: r["issues"] for r in csv.DictReader(open(os.path.join(V1, "tableVII_imperfect.csv")))}
    templ = json.load(open(os.path.join(V1, "group_schema_templates.json")))
    con = sqlite3.connect(f"file:{ce.DB}?mode=ro", uri=True); con.row_factory = sqlite3.Row
    rows = {r["table_id"]: r for r in con.execute(
        "SELECT table_id,title,year,canonical_name,source_image,member_images FROM tables").fetchall()}

    cands = []
    for tid, issues in imp.items():
        t = rows.get(tid)
        if not t:
            continue
        raw = (t["member_images"] or t["source_image"] or "")
        scans = []
        for p in re.split(r"[;,|]", raw):
            loc = ce.resolve_img(p.strip()) if p.strip() else None
            if loc and loc not in scans:
                scans.append(loc)
        if not scans:
            continue
        ih = IMG.search(t["source_image"] or "")
        tm = templ.get(ih.group(1)) if ih else None
        # expected schema: from template if present, else default Table VII
        if tm:
            cols = [c["name"] for c in tm["column_groups"]]
            ages = tm.get("row_labels", [])[:24]
            tier = "templated"
        else:
            cols = ["Population", "Unmarried", "Married", "Widowed"]
            ages = []
            tier = "default"
        cands.append({"idx": len(cands), "table_id": tid, "tier": tier, "issues": issues,
                      "title": (t["title"] or "")[:80], "year": t["year"], "province": t["canonical_name"],
                      "scans": scans[:4], "expected_columns": cols, "expected_age_rows": ages})

    # strongest first (templated)
    cands.sort(key=lambda c: 0 if c["tier"] == "templated" else 1)
    for i, c in enumerate(cands):
        c["idx"] = i
    json.dump(cands, open(os.path.join(V1, "opus_reread_candidates.json"), "w"))
    from collections import Counter
    print(f"Opus re-read candidates: {len(cands)}  ({dict(Counter(c['tier'] for c in cands))})")


if __name__ == "__main__":
    main()
