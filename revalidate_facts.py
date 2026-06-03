#!/usr/bin/env python3
"""Deterministic re-validation of confirmed tables under the new type-aware
constraints — NO API. Reconstructs each table from facts_confirmed.csv, infers
count vs proportional from measure_unit, re-derives + re-verifies constraints,
and reports the VERIFIED residue (what genuinely still fails) plus spurious
"persons" columns on rate tables.

Outputs (export_final/sendout_final/review/revalidate/):
  revalidated_tables.csv  - per table: kind, verified pass/fail, fail levels, persons flags
  spurious_persons_rate.csv - rate tables carrying a persons/All column (delete candidates)
"""
import csv, os
from collections import defaultdict
from oneshot import derive_constraints, verify_all_constraints

ROOT = os.path.dirname(os.path.abspath(__file__))
FACTS = os.path.join(ROOT, "export_final/sendout_final/data/facts_confirmed.csv")
OUT = os.path.join(ROOT, "export_final/sendout_final/review/revalidate")
os.makedirs(OUT, exist_ok=True)
csv.field_size_limit(10**7)


def main():
    by_tbl = defaultdict(list)
    for r in csv.DictReader(open(FACTS)):
        by_tbl[r["table_id"]].append(r)

    rows = []
    spurious = []
    for tid, facts in by_tbl.items():
        units = set(f["measure_unit"] for f in facts)
        is_rate = "rate" in units
        kind_hint = "proportional" if is_rate else "absolute"

        # reconstruct parsed: sections -> rows(row_label_std) -> group(column_group_std) -> {sub: value}
        secs = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
        groups = []
        for f in facts:
            sec = f["section"] or "_"
            row = f["row_label_std"] or f["row_label_raw"] or "?"
            grp = f["column_group_std"] or f["column_group_raw"] or "G"
            sub = f["sub_column_std"] or {"All": "persons", "Male": "males", "Female": "females"}.get(f["sex"], "")
            if sub not in ("persons", "males", "females"):
                continue
            try:
                v = float(f["value"]) if f["value"] not in ("", None) else None
            except ValueError:
                v = None
            if v is not None and v == int(v):
                v = int(v)
            secs[sec][row][grp][sub] = v
            if grp not in groups:
                groups.append(grp)

        parsed = {"metadata": {"title": "", "data_type": kind_hint,
                               "column_groups": groups, "row_label_name": "age"},
                  "sections": [{"name": s, "rows": [dict({"age": rl}, **{g: dict(cols) for g, cols in gd.items()})
                                                    for rl, gd in rd.items()]}
                               for s, rd in secs.items()]}
        try:
            cons = derive_constraints(parsed, schema=None, persons_independent=False)
            rep = verify_all_constraints(parsed, cons)
        except Exception as e:
            rows.append({"table_id": tid, "kind": "ERROR", "verified": "error",
                         "fail_levels": str(e)[:60], "n_failures": "", "has_persons": "",
                         "persons_on_rate": ""}); continue

        lv = sorted({f["level"] for f in rep["failures"]})
        has_persons = any(f["sex"] == "All" or f["sub_column_std"] == "persons" for f in facts)
        persons_on_rate = is_rate and has_persons
        if persons_on_rate:
            spurious.append({"table_id": tid, "n_persons_facts":
                             sum(1 for f in facts if f["sub_column_std"] == "persons" or f["sex"] == "All")})
        rows.append({
            "table_id": tid,
            "kind": cons.get("table_kind"),
            "verified": "pass" if not rep["failures"] else "fail",
            "n_failures": len(rep["failures"]),
            "fail_levels": ",".join(lv),
            "has_persons": int(has_persons),
            "persons_on_rate": int(persons_on_rate),
        })

    cols = ["table_id", "kind", "verified", "n_failures", "fail_levels", "has_persons", "persons_on_rate"]
    with open(os.path.join(OUT, "revalidated_tables.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); [w.writerow(r) for r in rows]
    with open(os.path.join(OUT, "spurious_persons_rate.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["table_id", "n_persons_facts"]); w.writeheader(); [w.writerow(r) for r in spurious]

    from collections import Counter
    npass = sum(1 for r in rows if r["verified"] == "pass")
    nfail = sum(1 for r in rows if r["verified"] == "fail")
    print(f"tables re-validated: {len(rows)}")
    print(f"  PASS under new constraints: {npass}")
    print(f"  FAIL (genuine residue):     {nfail}")
    print(f"  errors:                     {sum(1 for r in rows if r['verified']=='error')}")
    print(f"  by kind: {dict(Counter(r['kind'] for r in rows))}")
    print(f"  fail levels: {dict(Counter(l for r in rows if r['verified']=='fail' for l in r['fail_levels'].split(',') if l))}")
    print(f"  spurious persons on RATE tables (delete candidates): {len(spurious)}")
    print(f"  wrote -> {OUT}")


if __name__ == "__main__":
    main()
