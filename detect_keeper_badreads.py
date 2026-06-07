#!/usr/bin/env python3
"""Red-flag detector over the merged KEEPER facts (the tables that passed
constraints but may still be badly read). Constraints don't catch these because
0=0+0 and flat columns are arithmetically self-consistent.

Signals:
  full_zero       - every value in the table is 0/blank
  mostly_zero     - >=85% of cells are 0 (sparse misread), >=12 cells
  constant_col    - a column (>=4 rows) is the SAME non-zero value all the way down
  near_constant   - a column (>=6 rows) where one non-zero value covers >=75%
  single_value    - the whole table is one repeated non-zero value
  qc_flag         - extraction's own qc_flags / duplicate-conflict signal

Denominator columns (1000/10000/100 in rate/distribution) are exempted from the
constant/near-constant checks. Writes keeper_bad_reads.csv + prints a summary.
"""
import csv, os, json
from collections import defaultdict, Counter

csv.field_size_limit(10 ** 7)
ROOT = os.path.dirname(os.path.abspath(__file__))
V1 = os.path.join(ROOT, "export_final/sendout_final/sendout_v1")
DENOM = {1000.0, 10000.0, 100.0}


def num(v):
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


def main():
    # extraction's own QC signals per keeper table
    qc = {}
    for r in csv.DictReader(open(os.path.join(ROOT, "export/keeper_inference/source_catalog.csv"))):
        qc[r["table_id"]] = (r.get("qc_flags", "") or "").strip(), \
            num(r.get("duplicate_conflict_cells", "")) or 0, num(r.get("n_constraint_failures", "")) or 0

    cols = defaultdict(lambda: defaultdict(list))   # tid -> colkey -> [values]
    meta = {}
    for r in csv.DictReader(open(os.path.join(V1, "keeper_facts.csv"))):
        t = r["table_id"]
        meta[t] = (r["status"], r["measure_class"], (r["title"] or "")[:70], r["year"],
                   r["canonical_name"], r["table_family"])
        sub = r["sub_column_std"] or {"All": "persons", "Male": "males", "Female": "females"}.get(r["sex"], "")
        ck = (r["section"], r["column_group_std"], sub)
        cols[t][ck].append(num(r["value"]))

    out = []
    for t, cmap in cols.items():
        status, mc = meta[t][0], meta[t][1]
        allvals = [v for vs in cmap.values() for v in vs if v is not None]
        if not allvals:
            continue
        reasons, detail = [], []
        nz_all = [v for v in allvals if v != 0]
        if all(v == 0 for v in allvals):
            reasons.append("full_zero")
        elif len(allvals) >= 12 and (sum(1 for v in allvals if v == 0) / len(allvals)) >= 0.85:
            reasons.append("mostly_zero")
        if nz_all and len(set(nz_all)) == 1 and len(nz_all) >= 6:
            reasons.append("single_value")
        const_hits = near_hits = 0
        for ck, vs in cmap.items():
            nz = [v for v in vs if v is not None and v != 0]
            if len(nz) >= 4 and len(set(nz)) == 1:
                if not (nz[0] in DENOM and mc in ("distribution", "rate")):
                    const_hits += 1
                    if len(detail) < 5:
                        detail.append(f"{ck[1]}/{ck[2]}={nz[0]:g}x{len(nz)}")
            elif len(nz) >= 6:
                mv, mc_ct = Counter(nz).most_common(1)[0]
                if mc_ct / len(nz) >= 0.75 and not (mv in DENOM and mc in ("distribution", "rate")):
                    near_hits += 1
                    if len(detail) < 5:
                        detail.append(f"{ck[1]}/{ck[2]}~{mv:g}({mc_ct}/{len(nz)})")
        if const_hits:
            reasons.append(f"constant_col x{const_hits}")
        if near_hits:
            reasons.append(f"near_constant x{near_hits}")
        qf, dup, ncf = qc.get(t, ("", 0, 0))
        if qf or dup >= 3:
            reasons.append("qc_flag")
            if len(detail) < 6 and qf:
                detail.append(f"qc:{qf[:30]}")
        if reasons:
            out.append({"table_id": t, "reasons": ";".join(reasons), "detail": "; ".join(detail),
                        "status": status, "measure_class": mc, "table_family": meta[t][5],
                        "year": meta[t][3], "place": meta[t][4], "title": meta[t][2]})

    order = ["full_zero", "single_value", "mostly_zero", "constant_col", "near_constant", "qc_flag"]
    out.sort(key=lambda r: next((i for i, k in enumerate(order) if k in r["reasons"]), 9))
    with open(os.path.join(V1, "keeper_bad_reads.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["table_id", "reasons", "detail", "status", "measure_class",
                                          "table_family", "year", "place", "title"])
        w.writeheader(); [w.writerow(r) for r in out]

    c = Counter()
    for r in out:
        for k in order:
            if k in r["reasons"]:
                c[k] += 1
    print(f"keeper tables scanned (with facts): {len(cols):,}")
    print(f"red-flag suspects: {len(out):,}  ({100*len(out)//max(1,len(cols))}%)")
    print("  by signal:", dict(c))
    print("  by current status:", dict(Counter(r["status"] for r in out)))
    print("\nsample full_zero / single_value (most suspicious):")
    for r in out[:10]:
        print(f"   [{r['reasons'][:32]:32}] {r['year']:4} {r['place'][:16]:16} {r['table_family'][:18]:18} {r['title'][:34]}")
    print("\nwrote keeper_bad_reads.csv")


if __name__ == "__main__":
    main()
