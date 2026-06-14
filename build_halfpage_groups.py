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
    member_to_group = {}
    for gi, g in enumerate(groups):
        for m in g["member_table_ids"]:
            member_to_group[m] = gi

    added = extended = 0
    for members in clusters.values():
        if len(members) < 2:
            continue
        # if part of this cluster already lives in an existing group, EXTEND that group
        # with the new crops (don't abandon a crop just because its sibling is grouped).
        host = next((member_to_group[m] for m in members if m in member_to_group), None)
        new_members = [m for m in members if m not in covered]
        if not new_members:
            continue
        if host is not None:
            groups[host]["member_table_ids"].extend(new_members)
            groups[host]["pages"] = [0] * len(groups[host]["member_table_ids"])
            groups[host]["n"] = len(groups[host]["member_table_ids"])
            extended += 1
        else:
            m = DOC.search(members[0])
            dh = m.group(1) if m else "halfpage"
            groups.append({"dochash": dh, "subject": "TABLE VII. Age, Sex and Civil Condition (half-page merge)",
                           "table_number": "VII", "anchor_index": -1, "confidence": "halfpage",
                           "member_table_ids": new_members, "pages": [0] * len(new_members), "n": len(new_members)})
            added += 1
        for m in new_members:
            covered.add(m); member_to_group[m] = host if host is not None else len(groups) - 1

    json.dump(groups, open(out_path, "w"))
    print(f"half-page merge groups: {len(groups)} total (+{added} new, {extended} extended) "
          f"covering {sum(len(g['member_table_ids']) for g in groups)} crops")


if __name__ == "__main__":
    main()
