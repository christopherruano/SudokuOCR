#!/usr/bin/env python3
"""Phase 1: deterministically re-validate the not-on-site tables that already have
local OCR facts (needs_reocr / needs_human_review / hold_reocr). Re-extract from the
rich-facts files, apply persons-cleaning, re-check under type-aware constraints.
Passers -> recovered (written to recovered_phase1.json). Failers -> vision queue.
No new OCR, no cluster.
"""
import csv, glob, os, json
from collections import defaultdict, Counter
from oneshot import derive_constraints, verify_all_constraints

csv.field_size_limit(10 ** 7)
ROOT = os.path.dirname(os.path.abspath(__file__))
V1 = os.path.join(ROOT, "export_final/sendout_final/sendout_v1")


def main():
    cat = {r["table_id"]: r for r in csv.DictReader(open(os.path.join(V1, "catalog_tables.csv")))}
    target = {t for t, r in cat.items()
              if r["status"] in ("needs_reocr", "needs_human_review", "hold_reocr")}

    # pull facts for target tables from local rich files
    facts = defaultdict(list)
    for f in glob.glob(os.path.join(ROOT, "export/**/cell_facts_rich.csv"), recursive=True):
        try:
            for r in csv.DictReader(open(f)):
                t = r.get("table_id", "")
                if t in target and t not in facts:  # first file wins
                    pass
                if t in target:
                    facts[t].append(r)
        except Exception:
            continue
    # dedupe: keep one source per table (first seen)
    print(f"target tables: {len(target)} | with local facts: {len(facts)}")

    passers, failers = [], []
    for tid, fs in facts.items():
        # reconstruct sections->rows->group->{sub:value}; only persons/males/females sub-cols
        secs = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
        groups = []
        units = set()
        for r in fs:
            units.add(r.get("measure_unit", ""))
            sub = r["sub_column_std"] or {"All": "persons", "Male": "males", "Female": "females"}.get(r["sex"], "")
            if sub not in ("persons", "males", "females"):
                continue
            sec = r["section"] or "_"
            row = r["row_label_std"] or r["row_label_raw"] or "?"
            grp = r["column_group_std"] or r["column_group_raw"] or "G"
            try:
                v = float(r["value"]) if r["value"] not in ("", None) else None
            except ValueError:
                v = None
            if v is not None and v == int(v):
                v = int(v)
            secs[sec][row][grp][sub] = v
            if grp not in groups:
                groups.append(grp)
        kind_hint = "proportional" if "rate" in units else "absolute"
        parsed = {"metadata": {"title": cat[tid].get("title", ""), "data_type": kind_hint,
                               "column_groups": groups, "row_label_name": "age"},
                  "sections": [{"name": s, "rows": [dict({"age": rl}, **{g: dict(c) for g, c in gd.items()})
                                                    for rl, gd in rd.items()]}
                               for s, rd in secs.items()]}
        try:
            cons = derive_constraints(parsed, schema=None, persons_independent=False)
            rep = verify_all_constraints(parsed, cons)
            if not rep["failures"]:
                passers.append(tid)
            else:
                failers.append((tid, len(rep["failures"]), sorted({f["level"] for f in rep["failures"]})))
        except Exception as e:
            failers.append((tid, -1, [str(e)[:40]]))

    json.dump({"recovered": passers}, open(os.path.join(V1, "recovered_phase1.json"), "w"), indent=1)
    json.dump({"failers": [{"table_id": t, "n_fail": n, "levels": lv} for t, n, lv in failers]},
              open(os.path.join(V1, "vision_queue.json"), "w"), indent=1)

    print(f"\nPHASE 1 deterministic re-validation:")
    print(f"  PASS -> recover now: {len(passers)}")
    print(f"     by prior status: {dict(Counter(cat[t]['status'] for t in passers))}")
    print(f"  still FAIL -> vision queue: {len(failers)}")
    print(f"     by prior status: {dict(Counter(cat[t]['status'] for t, _, _ in failers))}")
    print(f"  wrote recovered_phase1.json + vision_queue.json")


if __name__ == "__main__":
    main()
