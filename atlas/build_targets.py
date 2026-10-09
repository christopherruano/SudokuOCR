#!/usr/bin/env python3
"""Re-extraction targets: district/state x census year where the district appears in a
whole-population age table but no self-checked age structure could be built (pages of the
table missing or partially read). One row per (place, year, volume) with the tables and the
volume's uncaptured page ranges. Output: census_site/data/reextraction_targets.csv."""
import csv, json, os, re, collections, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from build_index import AGE_FAMILIES, SUBPOP, ALLREL
ROOT = os.path.join(HERE, "..")
SITE = os.path.join(ROOT, "census_site")

def main():
    P = json.load(open(os.path.join(SITE, "data", "places.json")))
    V = json.load(open(os.path.join(SITE, "data", "volumes.json")))
    import sqlite3
    con = sqlite3.connect(os.path.join(ROOT, "export_final/sendout_final/sendout_v1/census_explorer.sqlite"))
    fam = dict(con.execute("select title, table_family from tables"))
    T = P["tables"]
    rows = []
    for pid, p in P["places"].items():
        if p["c"] not in ("district", "princely_state"):
            continue
        have = set((p.get("m") or {}).keys())
        need = collections.defaultdict(list)
        for i, how in p["t"]:
            t = T[i]
            if how == "birthplace" or not t["a"] or not t["y"] or t["y"] in have:
                continue
            title = t["t"] or ""
            if fam.get(title) not in AGE_FAMILIES or (SUBPOP.search(title) and not ALLREL.search(title)):
                continue
            need[(t["y"], t["d"])].append(t)
        for (y, d), ts in need.items():
            v = V["vols"].get(d, {})
            gaps = [it["gap"] for it in v.get("items", []) if "gap" in it]
            rows.append([pid, p["n"], p["c"], p.get("p", ""), y, d, v.get("s", ""), v.get("jstor", ""), len(ts),
                         " | ".join(sorted({(t["t"] or "")[:60] for t in ts}))[:300],
                         "; ".join(sorted({str(t["pg"] or "") for t in ts if t["pg"]}))[:200],
                         "; ".join(gaps)[:300]])
    rows.sort(key=lambda r: (r[4], r[3], r[1]))
    out = os.path.join(SITE, "data", "reextraction_targets.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["place_id", "place", "kind", "parent", "census_year", "volume_id", "volume", "jstor_url",
                    "n_age_tables", "age_table_titles", "printed_pages_of_those_tables", "uncaptured_page_ranges_in_volume"])
        w.writerows(rows)
    by_year = collections.Counter(r[4] for r in rows)
    print(f"{len(rows)} place-year-volume targets; distinct place-years {len({(r[0], r[4]) for r in rows})}; by year {dict(sorted(by_year.items()))}")

if __name__ == "__main__":
    main()
