#!/usr/bin/env python3
"""Historical plausibility: province medians of the verified district figures vs. the
sex ratios printed in the census reports (females per 1,000 males, whole province).
Prints a table; flags provinces whose district median is >60 from the report figure."""
import csv, statistics as st, collections, os
HERE = os.path.dirname(os.path.abspath(__file__))
REPORT = {  # Census of India reports (provincial totals); 1901 & 1911 & 1931
    ("punjab", "1901"): 854, ("punjab", "1911"): 817, ("punjab", "1931"): 831,
    ("frontier", "1901"): 846, ("frontier", "1931"): 843,
    ("united prov", "1901"): 937, ("united prov", "1911"): 915, ("united prov", "1931"): 904,
    ("bengal", "1901"): 960, ("bengal", "1911"): 945, ("bengal", "1931"): 924,
    ("bombay", "1901"): 945, ("bombay", "1911"): 933, ("bombay", "1931"): 909,
    ("madras", "1901"): 1029, ("madras", "1911"): 1032, ("madras", "1921"): 1028, ("madras", "1931"): 1025,
    ("central prov", "1901"): 1019, ("central prov", "1911"): 1008, ("assam", "1901"): 949, ("assam", "1911"): 940,
    ("burma", "1901"): 962, ("burma", "1911"): 955, ("mysore", "1901"): 980, ("mysore", "1911"): 979,
    ("hyderabad", "1901"): 960, ("hyderabad", "1911"): 966, ("baroda", "1901"): 937,
}
KEYS = ['punjab', 'frontier', 'united prov', 'n.-w', 'bengal', 'bihar', 'bombay', 'sind', 'madras', 'central prov',
        'berar', 'burma', 'assam', 'mysore', 'hyderabad', 'travancore', 'rajput', 'baroda', 'gwalior', 'ajmer', 'kashmir']

def main():
    rows = [r for r in csv.DictReader(open(os.path.join(HERE, "..", "census_site/data/district_age_indicators.csv")))
            if r["verified"] == "yes" and r["kind"] == "district"]
    P = {r["id"]: r for r in csv.DictReader(open(os.path.join(HERE, "out", "places.csv")))}
    def prov(pid):
        p = P[pid]["parent"].lower()
        k = next((k for k in KEYS if k in p), p[:14])
        return "united prov" if k == "n.-w" else k
    g = collections.defaultdict(list)
    for r in rows:
        g[(prov(r["place_id"]), r["census_year"])].append(r)
    print(f"{'province':14} year   n  F/1000M  report  %0-9  %60+")
    flags = 0
    for (pv, y), rs in sorted(g.items()):
        if len(rs) < 3:
            continue
        med = lambda k: st.median(float(r[k]) for r in rs if r[k]) if any(r[k] for r in rs) else float("nan")
        sr, rep = med("females_per_1000_males"), REPORT.get((pv, y))
        flag = rep is not None and abs(sr - rep) > 60
        flags += flag
        print(f"{pv:14} {y} {len(rs):3}  {sr:7.0f}  {rep or '':>6}  {med('pct_aged_0_9'):5.1f} {med('pct_aged_60_plus'):5.1f}{'   <-- check' if flag else ''}")
    print(f"\n{flags} province-years differ from the report by more than 60")

if __name__ == "__main__":
    main()
