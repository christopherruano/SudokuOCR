#!/usr/bin/env python3
"""Context-aware OCR for multi-page Table VII: derive each group's canonical
schema (column groups + sub-columns + age rows + cross-group constraint) from its
BEST-read sibling, and map every member crop's imghash -> that schema. At OCR time
the pipeline can use the sibling template instead of re-discovering structure,
forcing every page to fill the full Population/Unmarried/Married/Widowed x M/F
grid (fixes the silent column drop). Output: group_schema_templates.json {imghash: schema}.
"""
import os, re, csv, json
from collections import defaultdict, Counter

csv.field_size_limit(10 ** 7)
ROOT = os.path.dirname(os.path.abspath(__file__))
V1 = os.path.join(ROOT, "export_final/sendout_final/sendout_v1")
VII = re.compile(r"\bVII\b|age,?\s*sex|civil\s*condition", re.I)
IMG = re.compile(r"([0-9a-f]{40})")
AGEROW = re.compile(r"\d+\s*[-–]\s*\d+|\d+\s*(and over|\+)|under|total|all ages", re.I)


def main():
    groups = json.load(open(os.path.join(V1, "chop_groups.json")))
    vii_groups = [g for g in groups if VII.search(g.get("subject", "") or g.get("table_number", ""))]
    member_ids = set(t for g in vii_groups for t in g["member_table_ids"])

    # cells per member
    cells = defaultdict(list)
    for src in ("unified_facts.csv", "keeper_facts.csv"):
        p = os.path.join(V1, src)
        if not os.path.exists(p):
            continue
        for r in csv.DictReader(open(p)):
            if r["table_id"] in member_ids:
                cells[r["table_id"]].append(r)

    # source image (imghash) per member
    src_img = {}
    for r in csv.DictReader(open(os.path.join(ROOT, "export/keeper_inference/source_catalog.csv"))):
        src_img[r["table_id"]] = r.get("source_image", "")
    for r in csv.DictReader(open(os.path.join(V1, "catalog_tables.csv"))):
        src_img.setdefault(r["table_id"], r.get("source_image", ""))

    CIVIL = {"P": re.compile(r"popul|persons|total", re.I), "U": re.compile(r"unmarr|single", re.I),
             "M": re.compile(r"^marri|^\s*married", re.I), "W": re.compile(r"widow", re.I)}

    def civil_class(name):
        for k, rx in CIVIL.items():
            if rx.search(name or ""):
                return k
        return ""

    def group_template(members):
        # UNION of column groups + their sub-columns across ALL siblings (recovers
        # the full grid even when each page dropped a different part), canonical
        # age rows from the most complete member.
        grp_subs = defaultdict(set)
        best_rows, best_n = [], -1
        for tid in members:
            rows = cells.get(tid, [])
            if not rows:
                continue
            order, seen, nage = [], set(), 0
            for r in rows:
                cg = (r.get("column_group_std") or "").strip()
                sub = (r.get("sub_column_std") or r.get("sex") or "").strip()
                if cg:
                    grp_subs[cg].add(sub or "Persons")
                rl = (r.get("row_label_std") or "").strip()
                if rl and rl not in seen:
                    seen.add(rl); order.append(rl)
                    if AGEROW.search(rl):
                        nage += 1
            if nage > best_n:
                best_n, best_rows = nage, order
        if not grp_subs:
            return None
        # collapse case-variant duplicate groups (POPULATION vs Population), union their subs
        canon = {}
        for g, subs in grp_subs.items():
            key = g.strip().title()
            canon.setdefault(key, set()).update(s for s in subs if s)
        col_groups = [{"name": g, "sub_columns": sorted(s)} for g, s in canon.items()]
        classes = {civil_class(c["name"]) for c in col_groups} - {""}
        # only a TRUSTWORTHY template: must show the civil-condition structure (>=3 of P/U/M/W)
        if len(classes & {"P", "U", "M", "W"}) < 3 or best_n < 4:
            return None
        return {"column_groups": col_groups, "row_labels": best_rows,
                "row_axis_type": "age_rows", "row_label_name": "age", "table_kind": "count",
                "has_persons_column": any("persons" in s.lower() for c in col_groups for s in c["sub_columns"]),
                "cross_group_constraints": {next((c["name"] for c in col_groups if civil_class(c["name"]) == "P"), "Population"):
                                            [c["name"] for c in col_groups if civil_class(c["name"]) in ("U", "M", "W")]}}

    templates = {}
    n_groups = 0
    for g in vii_groups:
        tmpl = group_template(g["member_table_ids"])
        if not tmpl:
            continue
        n_groups += 1
        for tid in g["member_table_ids"]:
            m = IMG.search(src_img.get(tid, "") or "")
            if m:
                templates[m.group(1)] = tmpl

    json.dump(templates, open(os.path.join(V1, "group_schema_templates.json"), "w"))
    print(f"Table VII multi-page groups: {len(vii_groups)}  with a usable sibling template: {n_groups}")
    print(f"member crop imghashes mapped to a template: {len(templates)}")
    avg = sum(len(s['column_groups']) for s in templates.values()) / max(1, len(templates))
    print(f"avg column groups per template: {avg:.1f}")


if __name__ == "__main__":
    main()
