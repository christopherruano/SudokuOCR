#!/usr/bin/env python3
"""Historical-unit outlines for each census year, built from historical polygons by overlap.

Base pieces
  * the 1941 Census of India district/princely-state polygons (India State Stories, FLAME
    University; CC0; mapdata/historical_districts/India-State-Districts-1941.parquet),
    which cover the territory of present-day India;
  * GADM 4.1 modern districts only where the 1941 map has no coverage (Pakistan,
    Bangladesh, Myanmar, and slivers).
Ownership in year Y (each piece belongs to exactly one unit)
  1. a 1941 piece whose name matches a historical unit drawn in year Y, *and* which that
     unit's modern-successor area covers by >= 30% (so same-named places elsewhere, e.g.
     Bilaspur State vs Bilaspur District, cannot match), belongs to that unit;
  2. otherwise it belongs to the year-Y unit whose modern-successor area overlaps it most.
A unit's outline is the union of its pieces. For 1931-1941 this is the census map itself;
for earlier years it is the 1941 map regrouped by the year-Y units, so a district later
split (e.g. Kistna before Guntur was carved out in 1904) correctly includes both parts.
Changes that never crossed a 1941 line are not visible. Each outline records its source.

Output census_site/data/hist_geo.json {group_key: {d, c, src}} and atlas/out/hist_owner.json.
"""
import json, os, re, csv, unicodedata, collections
import geopandas as gpd
from shapely.ops import unary_union
from shapely.strtree import STRtree
HERE = os.path.dirname(os.path.abspath(__file__))
from build_geometry import load_l2, projection, svg_path, TOL
REF = os.path.join(HERE, "..", "mapdata/historical_districts/India-State-Districts-1941.parquet")
EA = "ESRI:102028"


def nm(s):
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    s = re.sub(r"\b(district|districts|state|agency|the)\b", " ", s)
    return re.sub(r"[^a-z]", "", s)


def main():
    A = json.load(open(os.path.join(HERE, "..", "census_site/data/atlas.json")))
    places = {r["id"]: r for r in csv.DictReader(open(os.path.join(HERE, "out", "places.csv")))}
    l2 = load_l2()
    proj = projection(l2)
    l2e = l2.to_crs(EA)
    l2e["geometry"] = l2e.geometry.buffer(0)
    ref = gpd.read_parquet(REF)
    # source rows that carry the province name instead of the state's (identified by location; see csv)
    for r in csv.DictReader(open(os.path.join(HERE, "map1941_fixes.csv"))):
        ref.loc[int(r["row"]), "Districts"] = r["name"]
    ref = ref[ref.geometry.notna()].copy()
    ref["geometry"] = ref.geometry.buffer(0)
    refe = ref.to_crs(EA)
    # Base pieces are GADM sub-district units (tehsils/taluks, upazilas, townships; Pakistan has
    # only districts), which carry accurate geometry. Each takes the identity (name, type) of the
    # 1941 census-map unit covering most of it: the 1941 digitization is right about which unit
    # is where but positionally coarse (median ~15 km displacement vs GADM), so it decides
    # composition while GADM draws the lines.
    L3 = os.path.join(HERE, "..", "mapdata", "gadm_l3")
    frames = [gpd.read_file(os.path.join(L3, f"gadm41_{c}_3.json"))[["GID_3", "GID_2", "geometry"]].rename(columns={"GID_3": "pid_"})
              for c in ("IND", "BGD", "MMR")]
    pak = l2[l2["country"] == "PAK"][["gid", "geometry"]].rename(columns={"gid": "pid_"})
    sub = gpd.pd.concat(frames + [pak], ignore_index=True).set_crs(l2.crs, allow_override=True)
    sub["geometry"] = sub.geometry.buffer(0)
    sube = sub.to_crs(EA)
    rtree = STRtree(list(refe.geometry))
    pieces = []      # (piece_id, geom_ea, geom_ll, name_key, source, type, label)
    for r, rl in zip(sube.itertuples(), sub.itertuples()):
        g = r.geometry
        if g is None or g.is_empty:
            continue
        best, ba = None, 0.0
        for h in rtree.query(g):
            a = refe.geometry.iloc[h].intersection(g).area
            if a > ba:
                best, ba = h, a
        if best is not None and ba >= 0.5 * g.area:
            rr = refe.iloc[best]
            pieces.append((f"s{r.pid_}", g, rl.geometry, nm(rr["Districts"]), "census map 1941", rr["Type"] or "", rr["Districts"], int(best)))
        else:
            pieces.append((f"s{r.pid_}", g, rl.geometry, "", "reconstructed from modern districts", "modern", "", None))
    # 1941 units that own no sub-district piece (smaller than, or offset across, every tehsil —
    # e.g. Sangli, Nabha, Malda) are drawn directly from their 1941 polygon, clipped to land,
    # on top of the tehsil-based map
    land = unary_union(list(l2e.geometry))
    has = {pc[7] for pc in pieces if pc[7] is not None}
    n_ov = 0
    for i, r in enumerate(refe.itertuples()):
        if r.Districts and i not in has:
            g = r.geometry.intersection(land)
            if g.is_empty:
                continue
            gl = gpd.GeoSeries([g], crs=EA).to_crs(l2.crs).iloc[0]
            pieces.append((f"o{i}", g, gl, nm(r.Districts), "census map 1941 (drawn directly)", r.Type or "", r.Districts, i))
            n_ov += 1
    print(f"{n_ov} small 1941 units drawn directly from the census map")
    # registry-wide name index (any year) so an untabulated state keeps its own identity
    code2gid = {f'{u["country"]}/{u["admin1"]}/{u["name"]}': u["gid"] for u in json.load(open(os.path.join(HERE, "out", "districts_svg.json")))["units"]}
    gid_geom0 = dict(zip(l2e["gid"], l2e.geometry))
    reg_names, reg_area = collections.defaultdict(list), {}
    for pid, p in places.items():
        if p["kind"] in ("princely_state", "district") and p["modern"]:
            gs = [gid_geom0[code2gid[m]] for m in p["modern"].split(";") if m in code2gid and code2gid[m] in gid_geom0]
            if gs:
                reg_area[pid] = unary_union(gs)
                for s_ in [p["name"]] + [a for a in p["aliases"].split(" | ") if a]:
                    if nm(s_):
                        reg_names[nm(s_)].append(pid)
    gid_geom = dict(zip(l2e["gid"], l2e.geometry))
    out, owner = {}, {}
    changes = {r["district_1941"]: r for r in csv.DictReader(open(os.path.join(HERE, "district_changes.csv")))}
    for y in A["hy"]:
        units = A["hcand"][y]
        # modern-successor area of each candidate unit in year y
        area = {pid: unary_union([gid_geom[g] for g in key.split("|") if g in gid_geom]) for pid, key in units.items()}
        names = collections.defaultdict(list)
        for pid in units:
            p = places.get(pid, {})
            for s in [p.get("name", "")] + [a for a in p.get("aliases", "").split(" | ") if a]:
                if nm(s):
                    names[nm(s)].append(pid)
        uarea = list(area.items())
        utree = STRtree([a for _, a in uarea])
        own = {}
        # decide once per 1941 unit (all its sub-district pieces together), and per piece for modern ones
        decide = collections.OrderedDict()
        for pidx, pc in enumerate(pieces):
            decide.setdefault(("r", pc[7]) if pc[7] is not None else pc[0], []).append(pidx)   # one 1941 record = one unit
        for unit_key, idxs in decide.items():
            pid_piece, _, _, key, src, typ, label, _row = pieces[idxs[0]]
            g = unary_union([pieces[i][1] for i in idxs]) if len(idxs) > 1 else pieces[idxs[0]][1]
            pidx = idxs
            best = None
            ch = changes.get(label) if typ == "British districts" else None
            if ch and int(y if y != "all" else 9999) < int(ch["first_census_as_own_unit"]):
                key = ""                                  # skip name matching before the unit existed
            want = {"Princely states": ("princely_state",), "British districts": ("district",),
                    "Princely districts": ("district", "princely_state")}.get(typ)
            for cand in names.get(key, []) if key else []:
                if want and places.get(cand, {}).get("kind") not in want:
                    continue                      # Benares State must not match Benares District
                if area[cand].intersection(g).area >= 0.3 * g.area:
                    best = cand
                    break
            if best is None and typ in ("Princely states", "Other foreign territories", "Boundaries not demarcated for minor states"):
                # a princely state is never part of another unit: draw it under its own name
                # (its registry place if we have one, otherwise the 1941 map's name), data or not
                reg = [q for q in reg_names.get(key, []) if places[q]["kind"] == "princely_state" and reg_area.get(q) is not None
                       and reg_area[q].intersection(g).area >= 0.3 * g.area]
                best = reg[0] if reg else f"map1941:{label}"
            elif best is None and typ == "British districts":
                # a 1941 district is itself, except before a recorded split/rename (district_changes.csv)
                ch = changes.get(label)
                if ch and int(y if y != "all" else 9999) < int(ch["first_census_as_own_unit"]):
                    best = ch["before_that_part_of"]
                else:
                    reg = [q for q in reg_names.get(key, []) if places[q]["kind"] == "district" and reg_area.get(q) is not None
                           and reg_area[q].intersection(g).area >= 0.3 * g.area]
                    best = reg[0] if reg else f"map1941:{label}"
            elif best is None:
                # districts of a princely state may merge into their state; modern pieces into any unit
                ok_kinds = ("district", "princely_state")
                hits = [h for h in utree.query(g) if places.get(uarea[h][0], {}).get("kind") in ok_kinds]
                # most specific unit wins: intersection over union, so Attock (its own area) beats
                # Rawalpindi (whose modern successors also include Attock)
                ov = []
                for h in hits:
                    inter = uarea[h][1].intersection(g).area
                    if inter > 0.05 * g.area:
                        ov.append((inter / (uarea[h][1].area + g.area - inter), uarea[h][0]))
                if ov:
                    best = max(ov)[1]
                elif label:
                    best = f"map1941:{label}"
            if best is None and ch and int(y if y != "all" else 9999) < int(ch["first_census_as_own_unit"]):
                best = ch["before_that_part_of"]
            if best:
                own.setdefault(best, []).extend(idxs)
        # modern pieces left without an owner (a modern district no historical unit maps to, e.g.
        # Hafizabad) join the nearest owned unit in the same country
        owned = {i for idxs in own.values() for i in idxs}
        cent = {pid: unary_union([pieces[i][1] for i in idxs]) for pid, idxs in own.items()}
        for i, pc in enumerate(pieces):
            if i in owned or pc[4] != "reconstructed from modern districts":
                continue
            c = pc[1].representative_point()
            ctry = pc[0][1:4]
            near = [(g.distance(c), pid) for pid, g in cent.items()
                    if any(pieces[j][0][1:4] == ctry for j in own[pid])]
            if near:
                d, pid = min(near)
                if d < 150000:
                    own[pid].append(i)
        owner[y] = {pid: [pieces[i][0] for i in idxs] for pid, idxs in own.items()}
        for pid, idxs in own.items():
            gk = "|".join(sorted(pieces[i][0] for i in idxs))
            if gk in out:
                A["hy"][y][pid] = gk
                continue
            u = unary_union([pieces[i][2] for i in idxs]).buffer(0.003).buffer(-0.003).simplify(TOL * 0.6, preserve_topology=True)
            d, c = svg_path(u, proj)
            srcs = {pieces[i][4] for i in idxs}
            if srcs == {"census map 1941 (drawn directly)"}:
                out[gk] = {"d": d, "c": c, "src": "census map 1941 (direct)", "ov": 1}
                if pid.startswith("map1941:") or pid not in places:
                    out[gk]["n"] = pid.split(":", 1)[-1] if pid.startswith("map1941:") else pid
                A["hy"][y][pid] = gk
                continue
            srcs = {x.replace(" (drawn directly)", "") for x in srcs}
            out[gk] = {"d": d, "c": c, "src": "census map 1941" if srcs == {"census map 1941"} else
                       "reconstructed from modern districts" if srcs == {"reconstructed from modern districts"} else "mixed"}
            if pid.startswith("map1941:") or pid not in places:
                out[gk]["n"] = pid.split(":", 1)[-1] if pid.startswith("map1941:") else pid
            A["hy"][y][pid] = gk
        # units that own no piece keep no outline (they lie inside another unit's piece)
        for pid in list(A["hy"][y]):
            if pid not in own:
                del A["hy"][y][pid]
        for pid in own:                      # registry states drawn without that year's data
            if pid in places and pid not in A["places"]:
                A["places"][pid] = {"n": places[pid]["name"], "p": places[pid]["parent"], "k": places[pid]["kind"],
                                    "mod": places[pid]["modern"], "tier": ""}
    json.dump(out, open(os.path.join(HERE, "..", "census_site/data/hist_geo.json"), "w"), separators=(",", ":"))
    # keep each drawn unit's modern successors for "today" and the subdivision view
    A["hmod"] = {y: {pid: A["hcand"][y][pid] for pid in A["hy"][y] if pid in A["hcand"][y]} for y in A["hy"]}
    A.pop("hcand", None)
    json.dump(A, open(os.path.join(HERE, "..", "census_site/data/atlas.json"), "w"), separators=(",", ":"))
    json.dump(owner, open(os.path.join(HERE, "out", "hist_owner.json"), "w"))
    src = collections.Counter(v["src"] for v in out.values())
    print(f"{len(out)} outlines from {len(pieces)} sub-district pieces ({sum(p[4] == 'census map 1941' for p in pieces)} identified on the 1941 map); sources {dict(src)}")
    print("units drawn per year:", {y: len(v) for y, v in A["hy"].items()})


if __name__ == "__main__":
    main()
