#!/usr/bin/env python3
"""Validate the reconstructed historical-district outlines against an independent
digitized historical map: India State Stories (FLAME University) district boundaries
for 1941, from the Census of India administrative atlas (CC0, via india-geodata,
release census/historical). Coverage: present-day India only (no Sind, NWFP, East
Bengal, Burma). File: mapdata/historical_districts/India-State-Districts-1941.parquet.

For every 1941 unit whose name matches one of our historical places, compare our
reconstructed outline for 1941 (union of the modern successors assigned to that place
in 1941, or 1931 if absent) with the 1941 polygon:
  IoU        area(ours ∩ theirs) / area(ours ∪ theirs)   (equal-area projection)
  coverage   share of the 1941 polygon covered by ours
Writes atlas/out/boundary_validation.csv and prints the summary.
"""
import json, os, re, csv, unicodedata, collections, statistics as st
import geopandas as gpd
from shapely.ops import unary_union
from rapidfuzz import fuzz
HERE = os.path.dirname(os.path.abspath(__file__))
from build_geometry import load_l2
EA = "ESRI:102028"   # Asia South Albers equal-area conic


def nm(s):
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    s = re.sub(r"\b(district|state|agency|the)\b", " ", s)
    return re.sub(r"[^a-z]", "", s)


def main():
    ref = gpd.read_parquet(os.path.join(HERE, "..", "mapdata/historical_districts/India-State-Districts-1941.parquet"))
    ref = ref[ref["Districts"].notna()].to_crs(EA)
    A = json.load(open(os.path.join(HERE, "..", "census_site/data/atlas.json")))
    places = {r["id"]: r for r in csv.DictReader(open(os.path.join(HERE, "out", "places.csv")))}
    l2 = load_l2().to_crs(EA)
    geom = dict(zip(l2["gid"], l2.geometry.buffer(0)))
    groups = {}
    for y in ("1941", "1931", "1921"):
        for pid, key in A["hy"].get(y, {}).items():
            groups.setdefault(pid, (y, key))
    # name index of our districts/states
    idx = collections.defaultdict(list)
    for pid in groups:
        p = places[pid]
        for s in [p["name"]] + [a for a in p["aliases"].split(" | ") if a]:
            idx[nm(s)].append(pid)
    rows = []
    for r in ref.itertuples():
        key = nm(r.Districts)
        cands = idx.get(key, [])
        if not cands:   # fuzzy fallback within spelling distance
            best = max(((fuzz.ratio(key, k), k) for k in idx), default=(0, ""))
            if best[0] >= 92:
                cands = idx[best[1]]
        if not cands:
            rows.append([r.Districts, r._2, r.Type, "", "", "", "", "no matching historical place"])
            continue
        pid = cands[0]
        y, k = groups[pid]
        ours = unary_union([geom[g] for g in k.split("|") if g in geom])
        theirs = r.geometry.buffer(0)
        inter, uni = ours.intersection(theirs).area, ours.union(theirs).area
        iou = inter / uni if uni else 0
        cov = inter / theirs.area if theirs.area else 0
        rows.append([r.Districts, r._2, r.Type, pid, y, round(iou, 3), round(cov, 3), ""])
    out = os.path.join(HERE, "out", "boundary_validation.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["district_1941", "admin_division_1941", "type_1941", "our_place", "our_year", "iou", "coverage_of_1941", "note"])
        w.writerows(rows)
    m = [r for r in rows if r[5] != ""]
    ious = [r[5] for r in m]
    print(f"1941 units: {len(rows)}; matched to our places: {len(m)}")
    for t in ("British districts", "Princely districts", "Princely states"):
        v = [r[5] for r in m if r[2] == t]
        if v:
            print(f"  {t:20} n={len(v):3}  median IoU {st.median(v):.2f}  IoU>=0.7: {sum(x >= .7 for x in v)}/{len(v)}")
    print(f"  all                  n={len(ious):3}  median IoU {st.median(ious):.2f}  IoU>=0.7: {sum(x >= .7 for x in ious)}  IoU<0.4: {sum(x < .4 for x in ious)}")
    print("  worst:", sorted(((r[5], r[0], r[3]) for r in m))[:12])


if __name__ == "__main__":
    main()
