#!/usr/bin/env python3
"""Sense-check every query-ready table: reconstruct its matrix and flag anything
degenerate or implausible, so we catch stragglers before sharing.

Reads sendout_v1/unified_facts.csv (status in confirmed_sent / recovered).
Writes sendout_v1/sense_check_report.csv (one row per flagged table) + prints a summary.
"""
import csv, os
from collections import defaultdict, Counter

csv.field_size_limit(10 ** 7)
ROOT = os.path.dirname(os.path.abspath(__file__))
V1 = os.path.join(ROOT, "export_final/sendout_final/sendout_v1")
FACTS = os.path.join(V1, "unified_facts.csv")
SEXLAB = {"All": "Persons", "Male": "Males", "Female": "Females"}


def kind(sub):
    s = (sub or "").lower().rstrip(".")
    if s in ("males", "male", "m"): return "M"
    if s in ("females", "female", "f"): return "F"
    if s in ("persons", "total", "all", "p"): return "T"
    return ""


def num(v):
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


def main():
    by_tbl = defaultdict(list)
    meta = {}
    for r in csv.DictReader(open(FACTS)):
        if r["status"] not in ("confirmed_sent", "recovered"):
            continue
        by_tbl[r["table_id"]].append(r)
        meta[r["table_id"]] = r

    flagged = []
    sev_count = Counter()
    for tid, facts in by_tbl.items():
        m = meta[tid]
        secs = defaultdict(list)
        for f in facts:
            secs[f["section"] or ""].append(f)
        n_cells = filled = 0
        max_cols = max_rows = 0
        nonnum = pmf_bad = pmf_chk = neg = dup_cols = 0
        vals = []
        for fs in secs.values():
            cols, rows, grid = [], [], {}
            for f in fs:
                grp = f["column_group_std"] or f["column_group_value"] or ""
                sub = f["sub_column_std"] or SEXLAB.get(f["sex"], "") or ""
                if not grp and not sub:
                    sub = "value"
                ck, rk = (grp, sub), (f["row_label_std"] or f["row_group_value"] or "")
                if ck not in cols: cols.append(ck)
                if rk not in rows: rows.append(rk)
                v = f["value"]
                grid[(rk, ck)] = v
                nv = num(v)
                if v not in ("", None) and nv is None:
                    nonnum += 1
                if nv is not None:
                    vals.append(nv)
                    if nv < 0 and m["measure_class"] != "variation":
                        neg += 1
            max_cols = max(max_cols, len(cols)); max_rows = max(max_rows, len(rows))
            n_cells += len(cols) * len(rows)
            filled += sum(1 for v in grid.values() if v not in ("", None))
            if len(cols) != len(set(cols)): dup_cols += 1
            # P=M+F within each column group — ONLY for count tables (rates/%/distributions don't sum)
            for g in (set(c[0] for c in cols) if m["measure_class"] == "count" else set()):
                gc = {kind(c[1]): c for c in cols if c[0] == g}
                if {"M", "F", "T"} <= set(gc):
                    for rk in rows:
                        pv, mv, fv = (num(grid.get((rk, gc[k]))) for k in ("T", "M", "F"))
                        if None not in (pv, mv, fv):
                            pmf_chk += 1
                            if abs(pv - (mv + fv)) > 0.5:
                                pmf_bad += 1
        fill = filled / n_cells if n_cells else 0
        reasons = []
        if not vals:
            reasons.append("no_numeric_values")
        if max_cols <= 1 and m["measure_class"] == "count" and m["subject"] in ("population", "civil_condition"):
            reasons.append("single_column")
        if vals and all(v == 0 for v in vals):
            reasons.append("all_zero")
        if n_cells >= 12 and fill < 0.35:
            reasons.append(f"sparse_{fill:.0%}")
        if nonnum:
            reasons.append(f"nonnumeric_{nonnum}")
        if pmf_chk and pmf_bad / pmf_chk > 0.10:
            reasons.append(f"pmf_mismatch_{pmf_bad}/{pmf_chk}")
        if neg:
            reasons.append(f"neg_in_{m['measure_class']}_{neg}")
        if dup_cols:
            reasons.append("dup_columns")
        if reasons:
            sev = ("high" if any(x.startswith(("no_numeric", "all_zero", "pmf_mismatch", "single_column")) for x in reasons)
                   else "medium" if any(x.startswith(("sparse", "nonnumeric", "neg_")) for x in reasons) else "low")
            sev_count[sev] += 1
            flagged.append({"table_id": tid, "severity": sev, "reasons": ";".join(reasons),
                            "year": m["year"], "region": m["broad_region"], "place": m["canonical_name"],
                            "subject": m["subject"], "measure_class": m["measure_class"],
                            "rows": max_rows, "cols": max_cols, "fill": f"{fill:.0%}",
                            "n_facts": len(facts), "title": (m["title"] or "")[:80]})

    flagged.sort(key=lambda r: ({"high": 0, "medium": 1, "low": 2}[r["severity"]], r["reasons"]))
    cols = ["table_id", "severity", "reasons", "year", "region", "place", "subject",
            "measure_class", "rows", "cols", "fill", "n_facts", "title"]
    with open(os.path.join(V1, "sense_check_report.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); [w.writerow(r) for r in flagged]

    print(f"tables checked: {len(by_tbl):,}")
    print(f"clean (no flags): {len(by_tbl) - len(flagged):,}  ({100*(len(by_tbl)-len(flagged))/len(by_tbl):.1f}%)")
    print(f"flagged: {len(flagged)}  -> {dict(sev_count)}")
    rc = Counter(x.split("_")[0] + ("_" + x.split("_")[1] if x.startswith(("neg",)) else "") for r in flagged for x in r["reasons"].split(";"))
    print("reason families:", dict(Counter(x.split("_")[0] for r in flagged for x in r["reasons"].split(";")).most_common()))
    print("\ntop HIGH-severity:")
    for r in [x for x in flagged if x["severity"] == "high"][:15]:
        print(f"  [{r['reasons']}] {r['year']} {r['place'][:18]} · {r['subject']}/{r['measure_class']} · {r['title'][:42]}")
    print("\nwrote sense_check_report.csv")


if __name__ == "__main__":
    main()
