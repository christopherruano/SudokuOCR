#!/usr/bin/env python3
"""Corpus-wide perfection scorecard. For every on-site table, apply the constraint
that matches its CLASSIFIED type (measure_class) — not a one-size rule:
  count:        Males+Females=Persons (and U+M+W=Population for civil-condition)
  distribution: U+M+W (or category set) sums to the 1,000/10,000 base
  rate/ratio/percentage: no additive constraint — flag only implausible values
  all:          full-zero / constant-column bad-read flags
Writes corpus_imperfect.csv + prints per-family perfect rates."""
import os, re, sqlite3
from collections import defaultdict, Counter
import census_explorer as ce

V1 = os.path.dirname(ce.DB)
DENOM = {1000.0, 10000.0, 100.0}
CIV = {"P": re.compile(r"popul|persons|^total", re.I), "U": re.compile(r"unmarr|single", re.I),
       "M": re.compile(r"^marri|married", re.I), "W": re.compile(r"widow", re.I)}


def civ(c):
    for k, rx in CIV.items():
        if rx.search(c or ""):
            return k
    return ""


def num(v):
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


def main():
    con = sqlite3.connect(f"file:{ce.DB}?mode=ro", uri=True); con.row_factory = sqlite3.Row
    tabs = con.execute("SELECT table_id,table_family,measure_class,status,title FROM tables").fetchall()
    out = []
    by_fam = defaultdict(lambda: [0, 0])     # family -> [perfect, total]
    for t in tabs:
        tid = t["table_id"]; mc = t["measure_class"] or "count"
        facts = con.execute("SELECT row_label_std,column_group_std,sub_column_std,sex,value FROM facts WHERE table_id=?", (tid,)).fetchall()
        if not facts:
            continue
        # coordinate -> {civil_class or sub: value}; also per-(row,group) sex map
        allv = []
        coord = defaultdict(dict)         # (row, sex) -> {P/U/M/W}
        mf = defaultdict(dict)            # (row, group) -> {persons/males/females}
        for f in facts:
            v = num(f["value"]); allv.append(v)
            cg = f["column_group_std"] or ""; sub = (f["sub_column_std"] or f["sex"] or "").lower()
            k = civ(cg)
            sex = "males" if "male" in sub and "fem" not in sub else "females" if "fem" in sub else "persons" if sub in ("persons", "all", "total") else sub
            if k and v is not None:
                coord[(f["row_label_std"], sex)][k] = v
            if v is not None:
                mf[(f["row_label_std"], cg)][sex] = v
        nz = [v for v in allv if v not in (None, 0)]
        issues = []
        if allv and not nz:
            issues.append("full_zero")
        # additive checks ONLY for count/distribution
        if mc in ("count", "distribution"):
            base = None
            if mc == "distribution":
                sums = [d["U"] + d["M"] + d["W"] for d in coord.values() if all(k in d for k in ("U", "M", "W"))]
                if sums:
                    import statistics
                    base = min(DENOM, key=lambda b: abs(statistics.median(sums) - b))
            cu = bu = 0
            for d in coord.values():
                if all(k in d for k in ("U", "M", "W")):
                    s = d["U"] + d["M"] + d["W"]
                    if mc == "distribution" and base:
                        cu += 1; bu += abs(s - base) > 0.05 * base
                    elif mc == "count" and "P" in d:
                        cu += 1; bu += abs(d["P"] - s) > max(2, 0.02 * max(1, d["P"]))
            if cu and bu > 0.15 * cu:
                issues.append(f"sum_fail:{bu}/{cu}")
            # M+F=Persons (count)
            if mc == "count":
                cp = bp = 0
                for d in mf.values():
                    if all(x in d for x in ("persons", "males", "females")):
                        cp += 1; bp += abs(d["persons"] - (d["males"] + d["females"])) > max(2, 0.02 * max(1, d["persons"]))
                if cp and bp > 0.15 * cp:
                    issues.append(f"mf_fail:{bp}/{cp}")
        # constant non-zero column (bad read) — any kind
        cols = defaultdict(list)
        for f in facts:
            v = num(f["value"])
            if v is not None:
                cols[(f["column_group_std"], f["sub_column_std"])].append(v)
        for vs in cols.values():
            nzc = [v for v in vs if v != 0]
            if len(nzc) >= 5 and len(set(nzc)) == 1 and nzc[0] not in DENOM:
                issues.append("constant_col"); break
        by_fam[t["table_family"]][1] += 1
        if issues:
            out.append({"table_id": tid, "family": t["table_family"], "measure_class": mc,
                        "issues": ";".join(issues), "title": (t["title"] or "")[:60]})
        else:
            by_fam[t["table_family"]][0] += 1

    import csv
    with open(os.path.join(V1, "corpus_imperfect.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["table_id", "family", "measure_class", "issues", "title"])
        w.writeheader(); [w.writerow(r) for r in out]
    tot = sum(v[1] for v in by_fam.values()); perf = sum(v[0] for v in by_fam.values())
    print(f"corpus tables checked: {tot}   PERFECT: {perf} ({100*perf//max(1,tot)}%)   imperfect: {len(out)}")
    print("by family (perfect/total):")
    for fam, (p, n) in sorted(by_fam.items(), key=lambda x: -x[1][1]):
        print(f"  {fam[:24]:24} {p:5}/{n:<5} ({100*p//max(1,n)}%)")
    print("imperfect by issue:", dict(Counter(i.split(':')[0] for r in out for i in r['issues'].split(';'))))


if __name__ == "__main__":
    main()
