#!/usr/bin/env python3
"""Deterministic bad-read detector over the queryable set:
  - full_zero      : every value in the table is 0/blank
  - constant_col   : a column (>=4 rows) is the SAME non-zero value all the way down
                     (a flat column across age bands is a classic OCR mis-read)
Writes sendout_v1/bad_read_suspects.json -> [{table_id, reasons, detail}].
Denominator-style constant columns (1000/10000/100 in rate/distribution tables) are
noted but down-weighted, since those can be legit per-mille bases.
"""
import csv, json, os
from collections import defaultdict

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
    # gather per-table: status, measure_class, and columns -> [values]
    cols = defaultdict(lambda: defaultdict(list))   # tid -> colkey -> [values]
    meta = {}
    for r in csv.DictReader(open(os.path.join(V1, "unified_facts.csv"))):
        t = r["table_id"]
        meta[t] = (r["status"], r["measure_class"], (r["title"] or "")[:70], r["year"], r["canonical_name"])
        sub = r["sub_column_std"] or {"All": "persons", "Male": "males", "Female": "females"}.get(r["sex"], "")
        ck = (r["section"], r["column_group_std"], sub)
        cols[t][ck].append(num(r["value"]))

    out = []
    for t, cmap in cols.items():
        status = meta[t][0]
        if status not in ("confirmed_sent", "recovered"):
            continue
        mc = meta[t][1]
        allvals = [v for vs in cmap.values() for v in vs if v is not None]
        reasons, detail = [], []
        if allvals and all(v == 0 for v in allvals):
            reasons.append("full_zero")
        # constant non-zero columns
        const_hits = 0
        for ck, vs in cmap.items():
            nz = [v for v in vs if v is not None]
            if len(nz) >= 4 and len(set(nz)) == 1 and nz[0] != 0:
                val = nz[0]
                legit_denom = (val in DENOM and mc in ("distribution", "rate"))
                if not legit_denom:
                    const_hits += 1
                    if len(detail) < 4:
                        detail.append(f"{ck[1]}/{ck[2]}={val:g}×{len(nz)}")
        if const_hits:
            reasons.append(f"constant_col×{const_hits}")
        if reasons:
            out.append({"table_id": t, "reasons": ";".join(reasons), "detail": "; ".join(detail),
                        "status": status, "measure_class": mc, "year": meta[t][3],
                        "place": meta[t][4], "title": meta[t][2]})

    out.sort(key=lambda r: (0 if "full_zero" in r["reasons"] else 1, r["reasons"]))
    json.dump(out, open(os.path.join(V1, "bad_read_suspects.json"), "w"), indent=1)
    from collections import Counter
    fz = sum(1 for r in out if "full_zero" in r["reasons"])
    cc = sum(1 for r in out if "constant_col" in r["reasons"])
    print(f"bad-read suspects: {len(out)}  (full_zero {fz}, constant_col {cc})")
    print("by status:", dict(Counter(r["status"] for r in out)))
    print("sample constant-col hits:")
    for r in [x for x in out if "constant_col" in x["reasons"]][:8]:
        print(f"   {r['year']} {r['place'][:16]:16} [{r['detail']}]  {r['title'][:40]}")
    print("wrote bad_read_suspects.json")


if __name__ == "__main__":
    main()
