#!/usr/bin/env python3
"""Validate every ON-SITE Table VII (Age, Sex & Civil Condition) for PERFECTION:
  - all three civil-condition columns present: Unmarried + Married + Widowed
    (Neil: some UMW tables were read with columns missing)
  - Unmarried + Married + Widowed == Population, per (age, sex)
  - age bands present and reasonably complete
Flags tables that are nominally good but imperfect -> they need re-OCR.
Writes tableVII_imperfect.csv (+ prints a summary)."""
import os, re, sqlite3
from collections import defaultdict

ROOT = os.path.dirname(os.path.abspath(__file__))
import census_explorer as ce
VII = re.compile(r"\bVII\b|age,?\s*sex\s*(and|&)?\s*civil|civil\s*condition", re.I)
AGEFAM = ("age_civil_condition", "age_sex_civil_condition", "age_sex_population")
GOOD = ("confirmed_sent", "recovered", "keeper_accepted", "keeper_review", "assembled")

POP = re.compile(r"popul|persons|^total", re.I)
UNM = re.compile(r"unmarr|single|never", re.I)
MAR = re.compile(r"^marri|^\s*married", re.I)
WID = re.compile(r"widow", re.I)
AGEROW = re.compile(r"\d+\s*[-–]\s*\d+|\d+\s*(and over|\+)|under|all ages|total", re.I)


def cls(c):
    c = c or ""
    if UNM.search(c): return "U"
    if WID.search(c): return "W"
    if MAR.search(c): return "M"
    if POP.search(c): return "P"
    return ""


def num(v):
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


def main():
    con = sqlite3.connect(f"file:{ce.DB}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    q = "SELECT table_id,title,year,canonical_name,status,table_family FROM tables WHERE status IN {}".format(GOOD)
    tabs = [r for r in con.execute(q).fetchall()
            if r["table_family"] in AGEFAM and VII.search(r["title"] or "")]

    out = []
    perfect = 0
    for t in tabs:
        facts = con.execute("SELECT row_label_std,column_group_std,sub_column_std,sex,value "
                            "FROM facts WHERE table_id=?", (t["table_id"],)).fetchall()
        # coordinate: (age_row, sex) -> {class: value}
        coord = defaultdict(dict)
        cols_present = set()
        ages = set()
        for f in facts:
            k = cls(f["column_group_std"]) or cls(f["sub_column_std"])
            if not k:
                continue
            cols_present.add(k)
            sex = f["sex"] or f["sub_column_std"] or ""
            row = f["row_label_std"] or ""
            if AGEROW.search(row):
                ages.add(row)
            v = num(f["value"])
            if v is not None:
                coord[(row, sex)][k] = v
        issues = []
        # U/M/W only expected for civil-condition tables (not plain age x sex population)
        expects_umw = t["table_family"] in ("age_civil_condition", "age_sex_civil_condition")
        if expects_umw and not ({"U", "M", "W"} <= cols_present):
            missing = {"U", "M", "W"} - cols_present
            issues.append("missing_cols:" + "".join(sorted(missing)))
        # U+M+W == P check
        checked = bad = 0
        for (row, sex), d in coord.items():
            if all(k in d for k in ("P", "U", "M", "W")):
                checked += 1
                if abs(d["P"] - (d["U"] + d["M"] + d["W"])) > max(2, 0.02 * d["P"]):
                    bad += 1
        if checked and bad > 0.15 * checked:
            issues.append(f"umw_sum_fail:{bad}/{checked}")
        if len(ages) < 5:
            issues.append(f"few_age_bands:{len(ages)}")
        if issues:
            out.append({"table_id": t["table_id"], "year": t["year"], "province": t["canonical_name"],
                        "status": t["status"], "issues": ";".join(issues), "title": (t["title"] or "")[:70]})
        else:
            perfect += 1

    out.sort(key=lambda r: r["issues"])
    import csv
    with open(os.path.join(os.path.dirname(ce.DB), "tableVII_imperfect.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["table_id", "year", "province", "status", "issues", "title"])
        w.writeheader(); [w.writerow(r) for r in out]

    from collections import Counter
    tag = Counter()
    for r in out:
        for i in r["issues"].split(";"):
            tag[i.split(":")[0]] += 1
    print(f"on-site Table VII validated: {len(tabs)}")
    print(f"  PERFECT (3 cols + U+M+W=P + ages): {perfect}  ({100*perfect//max(1,len(tabs))}%)")
    print(f"  imperfect (need re-OCR): {len(out)}")
    print(f"  by issue: {dict(tag)}")
    print("wrote tableVII_imperfect.csv")


if __name__ == "__main__":
    main()
