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

POP = re.compile(r"popul|persons|total", re.I)
UNM = re.compile(r"unmarr|single|never", re.I)
MAR = re.compile(r"marri", re.I)        # unanchored: civil class often mid-header ("Age 25-34 Married"); UNM is tested first so "unmarried"->U
WID = re.compile(r"widow", re.I)
AGEROW = re.compile(r"\d+\s*[-–]\s*\d+|\d+\s*(and over|\+)|under|all ages|total", re.I)
AGE_EXTRACT = re.compile(r"(\d+\s*[-–]\s*\d+|\d+\s*(?:and over|\+|plus)|under\s*\d+|all\s*ages|not\s*stated)", re.I)


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
    q = "SELECT table_id,title,year,canonical_name,status,table_family,measure_class FROM tables WHERE status IN {}".format(GOOD)
    tabs = [r for r in con.execute(q).fetchall()
            if r["table_family"] in AGEFAM and VII.search(r["title"] or "")]

    # human-verified tables (accepted in the review app) are PERFECT even if the
    # automated gate flags them — the printed source may not sum, or it's a faithful read.
    import csv as _csv
    hv_path = os.path.join(os.path.dirname(ce.DB), "human_verified.csv")
    human_verified = ({r["table_id"] for r in _csv.DictReader(open(hv_path))}
                      if os.path.exists(hv_path) else set())

    out = []
    scorecard = []
    perfect = 0
    for t in tabs:
        if t["table_id"] in human_verified:
            scorecard.append({"table_id": t["table_id"], "year": t["year"], "province": t["canonical_name"],
                              "status": t["status"], "perfect": "1", "issues": "", "title": (t["title"] or "")[:70]})
            perfect += 1
            continue
        facts = con.execute("SELECT row_label_std,column_group_std,sub_column_std,sex,value "
                            "FROM facts WHERE table_id=?", (t["table_id"],)).fetchall()
        # coordinate: (age_row, sex) -> {class: value}
        coord = defaultdict(dict)
        cols_present = set()
        ages = set()
        group_subs = defaultdict(set)   # civil-condition class -> set of sub-columns (M/F/persons) seen
        for f in facts:
            # civil-condition class can live in column_group, sub_column, OR row label
            # (orientation varies: many C-III tables put "Age 25-34 Married" in the header)
            kcol = cls(f["column_group_std"]) or cls(f["sub_column_std"]) or cls(f["row_label_std"])
            for fld in ("column_group_std", "sub_column_std", "row_label_std"):
                kk = cls(f[fld])
                if kk:
                    cols_present.add(kk)
            sex = (f["sex"] or f["sub_column_std"] or "").lower()
            # age can be in the row label OR embedded in the column header (transposed tables)
            am = AGE_EXTRACT.search(f["row_label_std"] or "") or AGE_EXTRACT.search(f["column_group_std"] or "")
            age = re.sub(r"\s+", "", am.group(1)).lower() if am else ""
            if age:
                ages.add(age)
            if kcol:
                group_subs[kcol].add(sex)
            v = num(f["value"])
            if v is not None and kcol and age:
                coord[(age, sex)][kcol] = v
        issues = []
        mc = (t["measure_class"] or "count")
        # U/M/W (count columns) only expected for civil-condition tables that are
        # actually COUNTS or distributions — a rate/ratio/percentage/variation table
        # (e.g. "proportion married per 1,000") has no Unmarried/Married/Widowed counts.
        expects_umw = (t["table_family"] in ("age_civil_condition", "age_sex_civil_condition")
                       and mc in ("count", "distribution"))
        if expects_umw and not ({"U", "M", "W"} <= cols_present):
            missing = {"U", "M", "W"} - cols_present
            issues.append("missing_cols:" + "".join(sorted(missing)))
        # sub-column truncation: a civil-condition group with FEWER sub-cols than the
        # richest group means its M/F (or persons) sub-columns were dropped (wide-table truncation)
        if expects_umw:
            present_subs = {k: {s for s in group_subs[k] if s} for k in ("U", "M", "W", "P") if group_subs[k]}
            if present_subs:
                maxn = max(len(s) for s in present_subs.values())
                short = [k for k, s in present_subs.items() if 0 < len(s) < maxn and maxn >= 2]
                if short:
                    issues.append("dropped_subcols:" + "".join(short))
        # U+M+W constraint gated by the table's CLASSIFIED type (from its title/headers),
        # not inferred from values: only count (sums to Population) and distribution
        # (sums to its 1,000/10,000 base) have this constraint; rate/ratio/percentage do NOT.
        if mc in ("count", "distribution"):
            base = None
            if mc == "distribution":
                sums = [d["U"] + d["M"] + d["W"] for d in coord.values() if all(k in d for k in ("U", "M", "W"))]
                if sums:
                    import statistics
                    med = statistics.median(sums)
                    base = min((1000, 10000, 100), key=lambda b: abs(med - b))   # which base, given it IS a distribution
            checked = bad = 0
            for d in coord.values():
                if not all(k in d for k in ("U", "M", "W")):
                    continue
                s = d["U"] + d["M"] + d["W"]
                if mc == "distribution" and base:
                    checked += 1; bad += abs(s - base) > 0.05 * base
                elif mc == "count" and "P" in d:
                    checked += 1; bad += abs(d["P"] - s) > max(2, 0.02 * max(1, d["P"]))
            if checked and bad > 0.15 * checked:
                issues.append(f"umw_sum_fail:{bad}/{checked}")
        # too-few age bands only signals truncation for tables that should carry a
        # full age structure (counts/distributions); rate/ratio/variation summaries
        # legitimately have few rows.
        if mc in ("count", "distribution") and len(ages) < 5:
            issues.append(f"few_age_bands:{len(ages)}")
        scorecard.append({"table_id": t["table_id"], "year": t["year"], "province": t["canonical_name"],
                          "status": t["status"], "perfect": "" if issues else "1",
                          "issues": ";".join(issues), "title": (t["title"] or "")[:70]})
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
    # full per-table scorecard (perfect + imperfect) for the verdict ledger
    with open(os.path.join(os.path.dirname(ce.DB), "tableVII_scorecard.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["table_id", "year", "province", "status", "perfect", "issues", "title"])
        w.writeheader(); [w.writerow(r) for r in scorecard]

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
