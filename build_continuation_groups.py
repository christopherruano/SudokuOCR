#!/usr/bin/env python3
"""Stitch fragmented multi-page Table VII / civil-condition crops that the chop
pipeline missed: standalone imperfect crops (transposed C-III by-religion pages,
or partial age-range continuation pages) from the SAME volume + SAME table number
are grouped so the assembler unions their age rows and splits religions/places into
sections. Conservative: groups only same (dochash, table_number); the assembler's
per-place split + coordinate dedup provide the safety net.
Writes continuation_groups.json (consumed by build_assembled_tables). ACCUMULATES."""
import os, re, csv, json
from collections import defaultdict
import census_explorer as ce

csv.field_size_limit(10 ** 7)
V1 = os.path.dirname(ce.DB)
DOC = re.compile(r"([0-9a-f]{16})_t")
# table number from a title: C-III / C.III / Imperial VII / TABLE VIII / XIV ...
TNUM = re.compile(r"\b(C[\s.\-]*[IVX]+|[IVX]{1,5})\b", re.I)


def tkey(title):
    m = TNUM.search((title or "").upper().replace("TABLE", " "))
    if not m:
        return ""
    return re.sub(r"[\s.\-]+", "", m.group(1)).upper()


def main():
    cats = [r for r in csv.DictReader(open(os.path.join(V1, "imperfect_categorized.csv")))
            if r["category"] in ("transposed", "partial_page") and not r["table_id"].startswith("asm_")]

    by = defaultdict(list)
    for r in cats:
        m = DOC.search(r["table_id"])
        if not m:
            continue
        k = (m.group(1), tkey(r["title"]))
        by[k].append(r["table_id"])

    out_path = os.path.join(V1, "continuation_groups.json")
    existing = json.load(open(out_path)) if os.path.exists(out_path) else []
    covered = set()
    for g in existing:
        covered.update(g["member_table_ids"])

    groups = list(existing)
    added = 0
    for (dh, tn), members in by.items():
        members = [m for m in members if m not in covered]
        if len(members) < 2 or not tn:
            continue
        groups.append({"dochash": dh, "subject": f"TABLE {tn}. Age, Sex and Civil Condition (continuation stitch)",
                       "table_number": tn, "anchor_index": -1, "confidence": "continuation",
                       "member_table_ids": members, "pages": [0] * len(members), "n": len(members)})
        covered.update(members); added += 1

    json.dump(groups, open(out_path, "w"))
    print(f"continuation stitch groups: {len(groups)} total (+{added} new) "
          f"covering {sum(len(g['member_table_ids']) for g in groups)} crops")


if __name__ == "__main__":
    main()
