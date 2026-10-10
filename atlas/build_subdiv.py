#!/usr/bin/env python3
"""Subdivision view: historical taluks/tahsils/townships/towns located on GADM 4.1 level-3
units (India tehsils/taluks, Bangladesh upazilas, Myanmar townships; Pakistan has no
finer GADM level than the district).

A historical subdivision is placed on a level-3 unit only if (a) the unit lies inside one
of the subdivision's mapped modern districts and (b) its name, or a recorded spelling,
matches the unit's name or GADM variant at rapidfuzz ratio >= 88 after removing unit
words. Everything else is listed as "not located". Output census_site/data/subdiv.json.
"""
import json, os, re, unicodedata, collections, csv
import geopandas as gpd
from rapidfuzz import fuzz
HERE = os.path.dirname(os.path.abspath(__file__))
from build_geometry import load_l2, projection, svg_path
L3 = os.path.join(HERE, "..", "mapdata", "gadm_l3")
SITE = os.path.join(HERE, "..", "census_site")
UNIT = re.compile(r"(?i)\b(taluk|taluka|taluq|tahsil|tehsil|township|pargana|subdivision|sub-division|thana|circle|"
                  r"mahal|petha|prant|town|city|municipality|cantonment|station|section|sadar|district)\b")


def nm(s):
    s = re.sub(r"([a-z])([A-Z])", r"\1 \2", s or "")
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    return re.sub(r"[^a-z]", "", UNIT.sub(" ", s))


def main():
    proj = projection(load_l2())
    frames = []
    for c in ("IND", "BGD", "MMR"):
        g = gpd.read_file(os.path.join(L3, f"gadm41_{c}_3.json"))
        frames.append(g[["GID_2", "GID_3", "NAME_3", "VARNAME_3", "geometry"]])
    g = gpd.pd.concat(frames, ignore_index=True)
    g["geometry"] = g.geometry.simplify(0.004, preserve_topology=True)
    l2geo = collections.defaultdict(list)
    names = {}
    for r in g.itertuples():
        if r.geometry is None or r.geometry.is_empty:
            continue
        d, c = svg_path(r.geometry, proj, min_area=1e-7)
        label = re.sub(r"([a-z])([A-Z])", r"\1 \2", r.NAME_3 or "")
        if label.startswith("n.a."):
            label = ""
        l2geo[r.GID_2].append([r.GID_3, label, d])
        names[r.GID_3] = (r.GID_2, [nm(r.NAME_3)] + [nm(v) for v in (r.VARNAME_3 or "").split("|") if v])
    geo = json.load(open(os.path.join(HERE, "out", "districts_svg.json")))
    code2gid = {f'{u["country"]}/{u["admin1"]}/{u["name"]}': u["gid"] for u in geo["units"]}
    P = json.load(open(os.path.join(SITE, "data", "places.json")))["places"]
    places = {r["id"]: r for r in csv.DictReader(open(os.path.join(HERE, "out", "places.csv")))}
    match = collections.defaultdict(list)
    unlocated = collections.defaultdict(list)
    n_try = n_ok = 0
    for pid, p in places.items():
        if p["kind"] not in ("sub_district", "town", "city_section") or pid not in P or not p["modern"]:
            continue
        if re.search(r"(?i)\bvillage\b|\bvillage tract\b", p["name"]):
            continue                               # below level-3 resolution
        l2s = [code2gid[m] for m in p["modern"].split(";") if m in code2gid]
        if not l2s:
            continue
        n_try += 1
        keys = {nm(p["name"])} | {nm(a) for a in p["aliases"].split(" | ") if a}
        keys = {k for k in keys if len(k) >= 3}
        best = None
        for g2 in l2s:
            for g3, _, _ in l2geo.get(g2, []):
                for cand in names[g3][1]:
                    if not cand:
                        continue
                    sc = max(fuzz.ratio(k, cand) for k in keys) if keys else 0
                    if sc >= 88 and (best is None or sc > best[0]):
                        best = (sc, g3)
        if best:
            match[best[1]].append(pid)
            n_ok += 1
        else:
            for g2 in l2s[:1]:
                unlocated[g2].append(pid)
    used = {pid for v in match.values() for pid in v} | {pid for v in unlocated.values() for pid in v}
    def anc(pid):
        out, q, seen = [], places[pid]["parent_id"], set()
        while q and q in places and q not in seen:
            seen.add(q)
            if places[q]["kind"] in ("district", "princely_state"):
                out.append(q)
            q = places[q]["parent_id"]
        return out
    pl = {pid: {"n": P[pid]["n"], "c": P[pid]["c"], "cy": P[pid].get("cy", {}), "m": P[pid].get("m", {}),
                "a": anc(pid), "pn": places[pid]["parent"]} for pid in used}
    out = {"l2": l2geo, "match": match, "un": unlocated, "pl": pl}
    path = os.path.join(SITE, "data", "subdiv.json")
    json.dump(out, open(path, "w"), separators=(",", ":"))
    print(f"level-3 units: {len(names)}; historical subdivisions tried {n_try}, located {n_ok} "
          f"({100 * n_ok / max(n_try, 1):.0f}%); file {os.path.getsize(path) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
