#!/usr/bin/env python3
"""Historical regression tests for the per-year atlas units (census_site/data/atlas.json).

Each assertion is a documented fact of Indian administrative history; the build must
reproduce it. Run after build_hist_geometry.py:  python3 atlas/test_history.py
"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
A = json.load(open(os.path.join(HERE, "..", "census_site/data/atlas.json")))
H = json.load(open(os.path.join(HERE, "..", "census_site/data/hist_geo.json")))
P = json.load(open(os.path.join(HERE, "out", "districts_svg.json")))["proj"]
lonlat = lambda c: (c[0] / (P["K"] * P["S"]) + P["minx"], P["maxy"] - c[1] / P["S"])

# (unit, census years in which it must be drawn, years in which it must NOT be drawn, why)
PRESENT = [
    ("guntur-district", ["1911", "1931"], ["1901"], "formed 1904 from Kistna"),
    ("kistna-district", ["1901", "1911"], [], ""),
    ("ramnad-district", ["1911"], ["1901"], "formed 1910 from Madura and Tinnevelly"),
    ("attock-district", ["1911", "1921"], ["1901"], "formed 1904 from Rawalpindi"),
    ("lyallpur-district", ["1911"], ["1901"], "formed 1904 (Chenab Colony)"),
    ("sheikhupura-district", ["1921"], ["1911"], "formed 1919"),
    ("wun-district", ["1901"], ["1911"], "renamed Yeotmal 1905"),
    ("yeotmal-district", ["1911"], ["1901"], "Wun renamed 1905"),
    ("basim-district", ["1901"], ["1911"], "abolished 1905"),
    ("ellichpur-district", ["1901"], ["1911"], "abolished 1905"),
    ("amraoti-district", ["1901", "1911"], [], ""),
    ("akola-district", ["1901", "1911"], [], ""),
    ("buldana-district", ["1901", "1911"], [], ""),
    ("narsinghpur-district", ["1901", "1931"], ["1941"], "merged into Hoshangabad 1932"),
    ("ajmer-district", ["1901", "1911"], ["1931"], "separate from Merwara until 1921"),
    ("merwara-district", ["1901"], ["1931"], ""),
    ("ajmer-merwara", ["1931"], ["1901"], "combined from 1921"),
    ("coorg", ["1901", "1931"], [], "Chief Commissioner's province"),
    ("madras-city", ["1901", "1931"], [], "separate district"),
    ("nilgiri-district", ["1901", "1931"], [], "separate district since 1868"),
    ("jullundur-district", ["1901"], [], ""),
    ("tehri-garhwal-state", ["1901", "1931"], [], "princely state throughout"),
    ("pataudi-state", ["1901"], [], "princely state inside Gurgaon"),
    ("patna-district", ["1901", "1931"], [], ""),
    ("patna-state", ["1901", "1931"], [], "Orissa feudatory state (homonym)"),
    ("bilaspur-district", ["1901"], [], ""),
    ("bilaspur-state", ["1901"], [], "Simla Hill state (homonym)"),
    ("partabgarh-district", ["1901"], [], ""),
    ("hyderabad-state", ["1881"], [], "tabulated as one unit in 1881"),
    ("map1941:Benares State", ["1911", "1931"], ["1901"], "created 1911 from Benares District lands"),
    ("mysore-district", ["1901", "1931"], [], "district of Mysore State, not the state itself"),
]
# (unit, year, lon range, lat range) — homonyms must sit in the right province
WHERE = [
    ("nilgiri-district", "1931", (76, 77.5), (11, 12)),
    ("patna-district", "1931", (84.5, 86), (25, 26)),
    ("patna-state", "1931", (82.5, 84), (20, 21.5)),
    ("bilaspur-district", "1901", (81.5, 83), (21.5, 23)),
    ("bilaspur-state", "1901", (76, 77.5), (31, 31.7)),
    ("partabgarh-district", "1901", (81, 82.5), (25.5, 26.5)),
]


def main():
    fails = []
    for pid, yes, no, why in PRESENT:
        for y in yes:
            if pid not in A["hy"].get(y, {}):
                fails.append(f"{pid} missing in {y} ({why})")
        for y in no:
            if pid in A["hy"].get(y, {}):
                fails.append(f"{pid} drawn in {y} but should not be ({why})")
    for pid, y, (lo0, lo1), (la0, la1) in WHERE:
        k = A["hy"].get(y, {}).get(pid)
        if not k:
            fails.append(f"{pid} not drawn in {y}")
            continue
        lo, la = lonlat(H[k]["c"])
        if not (lo0 <= lo <= lo1 and la0 <= la <= la1):
            fails.append(f"{pid} {y} at ({lo:.1f},{la:.1f}), expected lon {lo0}-{lo1} lat {la0}-{la1}")
    n = sum(len(x[1]) + len(x[2]) for x in PRESENT) + len(WHERE)
    print(f"historical assertions: {n - len(fails)}/{n} pass")
    for f in fails:
        print("  FAIL", f)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
