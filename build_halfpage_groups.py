#!/usr/bin/env python3
"""Build supplementary assembly groups for half-page-split Table VII: tables whose
missing columns sit on an adjacent sibling scan (one crop has Population, the next
has Unmarried/Married/Widowed). We cluster each table with its identified sibling
(union-find) and emit groups so build_assembled_tables unions their columns by
(age, sex) coordinate — recovering the full grid with NO re-OCR.
Writes halfpage_groups.json (consumed by build_assembled_tables)."""
import os, re, csv, json
import census_explorer as ce

csv.field_size_limit(10 ** 7)
V1 = os.path.dirname(ce.DB)
DOC = re.compile(r"([0-9a-f]{16})_t")


def main():
    cats = list(csv.DictReader(open(os.path.join(V1, "imperfect_categorized.csv"))))
    pairs = []
    for r in cats:
        if r["category"] != "half_page_split":
            continue
        m = re.match(r"sibling::(.+?)::seq", r["detail"])
        if m:
            pairs.append((r["table_id"], m.group(1)))

    # union-find over table_ids
    parent = {}
    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    def union(a, b):
        parent[find(a)] = find(b)
    for a, b in pairs:
        parent.setdefault(a, a); parent.setdefault(b, b); union(a, b)

    clusters = {}
    for x in parent:
        clusters.setdefault(find(x), []).append(x)

    # ACCUMULATE: keep prior half-page groups (their crops are already subsumed, so a
    # fresh categorize won't re-see them) and add only genuinely new clusters.
    out_path = os.path.join(V1, "halfpage_groups.json")
    existing = json.load(open(out_path)) if os.path.exists(out_path) else []
    covered = set()
    for g in existing:
        covered.update(g["member_table_ids"])

    groups = list(existing)
    added = 0
    for members in clusters.values():
        if len(members) < 2 or any(m in covered for m in members):
            continue
        m = DOC.search(members[0])
        dh = m.group(1) if m else "halfpage"
        groups.append({"dochash": dh, "subject": "TABLE VII. Age, Sex and Civil Condition (half-page merge)",
                       "table_number": "VII", "anchor_index": -1, "confidence": "halfpage",
                       "member_table_ids": members, "pages": [0] * len(members), "n": len(members)})
        covered.update(members); added += 1

    json.dump(groups, open(out_path, "w"))
    print(f"half-page merge groups: {len(groups)} total (+{added} new) "
          f"covering {sum(len(g['member_table_ids']) for g in groups)} crops")
    return added
    print(f"half-page merge groups: {len(groups)} (covering {sum(len(g['member_table_ids']) for g in groups)} crops)")


if __name__ == "__main__":
    main()
