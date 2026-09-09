#!/usr/bin/env python3
"""COMPARABLE Table VII series — harmonized age x sex so tables can be compared ACROSS YEARS for a place
and ACROSS PLACES within a year. Method: take the QC'd pyramid extraction (total population by age band x
sex per table), keep QC-OK tables, canonicalize place names, pick the best table per (place, year), then
aggregate each table's native age scheme onto STANDARD DECADE BANDS (0-10,...,60+) — only when the native
bands tile a target band exactly (no guessing). Emits census_site/data/tableVII_comparable.zip:
  harmonized.csv                 tidy: place, year, band, sex, population, pct_of_sex  (pivot anything)
  by_place/<place>.csv           one place across all years (bands x sex rows, years as columns)
  by_year/<year>.csv             one year across all places (places rows, band x sex + pct columns)
  README.md
  python3 build_comparable.py
"""
import os, csv, io, zipfile, re, datetime
from collections import defaultdict
from canonical_places import canonicalize

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "census_site/data")
BANDS = [(0, 10), (10, 20), (20, 30), (30, 40), (40, 50), (50, 60), (60, None)]


def band_label(b):
    return f"{b[0]}-{b[1]}" if b[1] else f"{b[0]}+"


def scheme_of(cells):
    """native age-scheme granularity: single_year > mixed_fine > quinquennial > decadal > other"""
    widths = []
    for (lo, hi) in cells:
        if hi is None: continue
        widths.append(max(0, hi - lo) or 1)
    ones = sum(1 for w in widths if w <= 1)
    fives = sum(1 for w in widths if 2 <= w <= 5)
    tens = sum(1 for w in widths if 6 <= w <= 10)
    if ones >= 15: return "single_year"
    if ones >= 4 and fives >= 3: return "mixed_fine"
    if fives >= max(3, len(widths) // 2): return "quinquennial"
    if tens >= max(2, len(widths) // 2): return "decadal"
    return "other"


def harmonize(cells):
    """cells: {(lo,hi_or_None): {sex: pop}} native -> {band: {sex: pop}} or None if it doesn't tile."""
    out = {}
    for b in BANDS:
        lo, hi = b
        tot = {"Male": 0, "Female": 0}
        covered = 0
        for (clo, chi), mf in cells.items():
            chi_eff = chi if chi is not None else 999
            if clo >= lo and (chi_eff <= (hi or 999)):
                if b[1] is None and chi is not None and chi_eff < lo:
                    continue
                if clo >= lo and (hi is None or chi_eff <= hi):
                    tot["Male"] += mf.get("Male", 0); tot["Female"] += mf.get("Female", 0)
                    covered += (chi_eff - clo) if chi is not None else (hi or 100) - clo
        span = (hi - lo) if hi else None
        if span is not None and abs(covered - span) > 1:      # native bands must tile the decade (±1yr slack)
            return None
        out[b] = tot
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    qc = {r["table_id"]: r for r in csv.DictReader(open(os.path.join(ROOT, "export/pyramids/pyramids_qc.csv")))}
    tabs = defaultdict(lambda: {"meta": None, "cells": defaultdict(dict)})
    for r in csv.DictReader(open(os.path.join(ROOT, "export/pyramids/pyramids_long.csv"))):
        t = tabs[r["table_id"]]
        if t["meta"] is None:
            t["meta"] = (r["place"], r["year"], r["entity_type"])
        hi = int(r["age_upper"]) if r["age_upper"] else None
        t["cells"][(int(r["age_lower"]), hi)][r["sex"]] = int(r["population"])

    # best QC-OK harmonizable table per (canonical place, year)
    best = {}
    for tid, t in tabs.items():
        place, year, ent = t["meta"]
        v = qc.get(tid, {})
        if v.get("verdict") != "OK" or not year:
            continue
        canon, kind = canonicalize(place)
        if kind == "junk" or not canon:
            continue
        h = harmonize(t["cells"])
        if not h:
            continue
        tot = sum(x["Male"] + x["Female"] for x in h.values())
        sch = scheme_of(t["cells"])
        FINE = {"single_year": 0, "mixed_fine": 1, "quinquennial": 2, "decadal": 3, "other": 4}
        key = (canon, year)
        rank = (FINE.get(sch, 4), -tot)
        if key not in best or rank < best[key][4]:
            best[key] = (h, tot, tid, ent, rank, sch, t["cells"])

    tidy = []
    native_rows = []
    for (place, year), (h, tot, tid, ent, rank, sch, cells) in sorted(best.items()):
        if sch in ("single_year", "mixed_fine"):        # the year-by-year gold: keep UNBINNED too
            for (lo, hi), mf in sorted(cells.items(), key=lambda kv: (kv[0][0], kv[0][1] or 999)):
                lbl = f"{lo}-{hi}" if hi is not None else f"{lo}+"
                for sex in ("Male", "Female"):
                    native_rows.append([place, year, sch, lbl, lo, hi if hi is not None else "", sex, mf.get(sex, 0)])
        tm = sum(x["Male"] for x in h.values()); tf = sum(x["Female"] for x in h.values())
        for b in BANDS:
            for sex, t_ in (("Male", tm), ("Female", tf)):
                pop = h[b][sex]
                tidy.append(dict(place=place, year=year, band=band_label(b), sex=sex, population=pop,
                                 pct_of_sex=round(100 * pop / t_, 2) if t_ else "", source_table=tid[:24],
                                 entity=ent, native_scheme=sch))
    # wide cuts
    by_place = defaultdict(dict)
    by_year = defaultdict(dict)
    for r in tidy:
        by_place[r["place"]].setdefault((r["band"], r["sex"]), {})[r["year"]] = r["population"]
        by_year[r["year"]].setdefault(r["place"], {})[(r["band"], r["sex"])] = (r["population"], r["pct_of_sex"])

    stamp = datetime.datetime.now().strftime("%Y-%m-%d")
    zpath = os.path.join(OUT, "tableVII_comparable.zip")
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        buf = io.StringIO(); w = csv.DictWriter(buf, fieldnames=list(tidy[0].keys())); w.writeheader(); w.writerows(tidy)
        z.writestr("harmonized.csv", buf.getvalue())
        for place, grid in by_place.items():
            years = sorted({y for v in grid.values() for y in v})
            b2 = io.StringIO(); w2 = csv.writer(b2)
            w2.writerow(["band", "sex"] + years)
            for b in BANDS:
                for sex in ("Male", "Female"):
                    w2.writerow([band_label(b), sex] + [grid.get((band_label(b), sex), {}).get(y, "") for y in years])
            fn = re.sub(r"[^A-Za-z0-9]+", "_", place)[:40]
            z.writestr(f"by_place/{fn}.csv", b2.getvalue())
        for year, grid in by_year.items():
            cols = [(band_label(b), s) for b in BANDS for s in ("Male", "Female")]
            b3 = io.StringIO(); w3 = csv.writer(b3)
            w3.writerow(["place"] + [f"{b}_{s}" for b, s in cols] + [f"{b}_{s}_pct" for b, s in cols])
            for place in sorted(grid):
                row = [place]
                row += [grid[place].get(c, ("", ""))[0] for c in cols]
                row += [grid[place].get(c, ("", ""))[1] for c in cols]
                w3.writerow(row)
            z.writestr(f"by_year/{year}.csv", b3.getvalue())
        if native_rows:
            nb = io.StringIO(); nw = csv.writer(nb)
            nw.writerow(["place", "year", "scheme", "age_label", "age_lower", "age_upper", "sex", "population"])
            nw.writerows(native_rows)
            z.writestr("single_year_native.csv", nb.getvalue())
        gb = io.StringIO(); gw = csv.writer(gb)
        gw.writerow(["place", "year", "native_scheme", "highlight"])
        for (place, year), (_h, _t, _tid, _e, _r, sch, _c) in sorted(best.items()):
            gw.writerow([place, year, sch, "YEAR-BY-YEAR ★" if sch in ("single_year", "mixed_fine") else ""])
        z.writestr("granularity.csv", gb.getvalue())
        z.writestr("README.md", f"""# Comparable Table VII series ({stamp})
Total population by AGE BAND x SEX, harmonized to standard decade bands (0-10 ... 60+) so tables are
directly comparable ACROSS census years for a place and ACROSS places within a year.
Only QC-passing tables whose native age scheme tiles the decades exactly are included (no interpolation).
pct_of_sex enables comparison independent of population size. {len(best)} place-years,
{len(by_place)} places, years {min(by_year)}-{max(by_year)}.

## ★ YEAR-BY-YEAR AGE RETURNS (the best data — highlighted, kept unbinned)
Tables whose native scheme is single-year (or fine mixed) are the richest: they support age-heaping,
smoothing and single-age demography that binned data cannot. They are listed in granularity.csv and
their FULL unbinned returns are in single_year_native.csv (the decade bands in harmonized.csv are the
comparison layer; the native file is the analysis layer).
""")
    print(f"comparable series: {len(best)} place-years · {len(by_place)} places · {len(by_year)} years -> {zpath} "
          f"({os.path.getsize(zpath)//1024}KB)")


if __name__ == "__main__":
    main()
