#!/usr/bin/env python3
"""Pull population-pyramid data (total population by age band x sex) for every available place x year from
the normalized facts table. Works PER TABLE (each Table VII has one consistent age scheme + scope), taking
MAX within each (table, age, sex) cell so we get the TOTAL (Population >= civil-condition subsets;
All-Religions >= any single religion) without double-counting. Only good-status COUNT tables.
Writes:
  export/pyramids/pyramids_long.csv   tidy: place,parent,year,entity_type,geo_level,table_id,age_lower,age_upper,age_label,sex,population
  export/pyramids/pyramids_wide.csv   one row per (table) : place,year,geo,#bands, then <age>_M / <age>_F columns
  export/pyramids/coverage.csv        place x year x geo_level availability
  python3 extract_pyramids.py
"""
import os, sqlite3, csv
from collections import defaultdict

ROOT = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(ROOT, "export_final/sendout_final/sendout_v1/census_explorer.sqlite")
OUT = os.path.join(ROOT, "export/pyramids")

GEO_OK = ("british_province", "princely_state", "all_india", "modern_state", "agency", "district", "urban")


def band_label(lo, hi, open_ended):
    if open_ended or hi is None:
        return f"{lo}+"
    if lo == hi:
        return f"{lo}"
    return f"{lo}-{hi}"


def main():
    os.makedirs(OUT, exist_ok=True)
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    q = """
    SELECT table_id, canonical_name, parent_geo, year, entity_type, geo_level,
           resolved_age_lower lo, resolved_age_upper hi, resolved_age_open_ended oe, sex,
           MAX(value) pop
    FROM facts
    WHERE measure_class='count' AND canonical_topic IN ('age_sex_population','age_civil_condition')
      AND sex IN ('Male','Female') AND resolved_age_lower IS NOT NULL
      AND age_aggregate='0'                       -- leaf bands only (drop 0-5 roll-ups & grand totals)
      AND (civil_condition IN ('Population','') OR civil_condition IS NULL)
      AND (religion IN ('All Religions','') OR religion IS NULL)
      AND status IN ('confirmed_sent','recovered','keeper_accepted','keeper_review','assembled','corrected')
    GROUP BY table_id, lo, hi, sex
    """
    rows = list(con.execute(q))
    # group by table
    tables = defaultdict(lambda: {"meta": None, "cells": {}})
    for r in rows:
        t = tables[r["table_id"]]
        if t["meta"] is None:
            t["meta"] = {k: r[k] for k in ("canonical_name", "parent_geo", "year", "entity_type", "geo_level")}
        t["cells"][(r["lo"], r["hi"], r["oe"], r["sex"])] = int(r["pop"] or 0)

    long_rows = []
    wide_rows = []
    cov = defaultdict(set)
    for tid, t in tables.items():
        m = t["meta"]
        if (m.get("entity_type") or "") not in GEO_OK and (m.get("geo_level") or "") not in ("province", "district"):
            continue
        bands = sorted({(lo, hi, oe) for (lo, hi, oe, sx) in t["cells"]},
                       key=lambda b: (b[0] if b[0] is not None else 9999, b[1] if b[1] is not None else 9999, b[2] or 0))
        if len(bands) < 4:
            continue                                    # too few bands to be a pyramid
        place = m["canonical_name"] or "(unknown)"
        for (lo, hi, oe) in bands:
            lbl = band_label(lo, hi, oe)
            for sx in ("Male", "Female"):
                v = t["cells"].get((lo, hi, oe, sx))
                if v is None:
                    continue
                long_rows.append([place, m["parent_geo"] or "", m["year"], m["entity_type"] or "",
                                  m["geo_level"] or "", tid, lo, hi, lbl, sx, v])
        # wide row
        wr = {"place": place, "parent": m["parent_geo"] or "", "year": m["year"],
              "entity_type": m["entity_type"] or "", "geo_level": m["geo_level"] or "",
              "n_bands": len(bands), "table_id": tid}
        for (lo, hi, oe) in bands:
            lbl = band_label(lo, hi, oe)
            wr[f"{lbl}_M"] = t["cells"].get((lo, hi, oe, "Male"), "")
            wr[f"{lbl}_F"] = t["cells"].get((lo, hi, oe, "Female"), "")
        wide_rows.append(wr)
        cov[(place, m["year"])].add(m["geo_level"] or m["entity_type"] or "?")

    # write long
    with open(os.path.join(OUT, "pyramids_long.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["place", "parent", "year", "entity_type", "geo_level", "table_id",
                    "age_lower", "age_upper", "age_label", "sex", "population"])
        w.writerows(long_rows)
    # write wide (union of all band columns, ordered)
    allbands = []
    for wr in wide_rows:
        for k in wr:
            if (k.endswith("_M") or k.endswith("_F")) and k not in allbands:
                allbands.append(k)
    def band_key(k):
        base = k[:-2]
        try:
            return (int(base.split("-")[0].rstrip("+")), k[-1])
        except Exception:
            return (9999, k)
    allbands.sort(key=band_key)
    head = ["place", "parent", "year", "entity_type", "geo_level", "n_bands", "table_id"] + allbands
    with open(os.path.join(OUT, "pyramids_wide.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=head, restval="")
        w.writeheader(); w.writerows(wide_rows)
    # coverage
    with open(os.path.join(OUT, "coverage.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["place", "year", "geo_levels"])
        for (place, year), gls in sorted(cov.items(), key=lambda kv: (str(kv[0][0]), kv[0][1] or 0)):
            w.writerow([place, year, "|".join(sorted(gls))])

    from collections import Counter
    yr = Counter(wr["year"] for wr in wide_rows)
    print(f"pyramids extracted: {len(wide_rows)} tables across {len(cov)} place×year · {len(long_rows):,} age×sex cells")
    print(f"  -> export/pyramids/  (pyramids_long.csv, pyramids_wide.csv, coverage.csv)")
    print("  tables by census year:", dict(sorted(yr.items(), key=lambda x: (x[0] or 0))))
    glc = Counter(wr["geo_level"] or wr["entity_type"] for wr in wide_rows)
    print("  by geo level:", dict(glc))


if __name__ == "__main__":
    main()
