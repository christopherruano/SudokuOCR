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
    s = re.sub(r"\(.*?\)", " ", s)
    s = re.sub(r"\b(district|districts|state|agency|the|prant|zila|zilla|settlement|estate|chiefship)\b", " ", s)
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
    FIXED_ROWS, FIXED_BEFORE = set(), {}
    for r in csv.DictReader(open(os.path.join(HERE, "map1941_fixes.csv"))):
        ref.loc[int(r["row"]), "Districts"] = r["name"]
        FIXED_ROWS.add(int(r["row"]))       # identity known: skip the lineage lookup for these
        if r.get("first_census"):
            FIXED_BEFORE[int(r["row"])] = (int(r["first_census"]), r["before_that_part_of"])
    ref = ref[ref.geometry.notna()].copy()
    ref["geometry"] = ref.geometry.buffer(0)
    refe = ref.to_crs(EA)
    # Base pieces are GADM sub-district units (tehsils/taluks, upazilas, townships; Pakistan has
    # only districts), which carry accurate geometry. Each takes the identity (name, type) of the
    # 1941 census-map unit covering most of it: the 1941 digitization is right about which unit
    # is where but positionally coarse (median ~15 km displacement vs GADM), so it decides
    # composition while GADM draws the lines.
    L3 = os.path.join(HERE, "..", "mapdata", "gadm_l3")
    frames = [gpd.read_file(os.path.join(L3, f"gadm41_{c}_3.json"))[["GID_3", "GID_2", "NAME_3", "geometry"]].rename(columns={"GID_3": "pid_"})
              for c in ("IND", "BGD", "MMR")]
    named_hits = [0]
    pak = l2[l2["country"] == "PAK"][["gid", "geometry"]].rename(columns={"gid": "pid_"})
    pak["NAME_3"] = ""
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
        ov = []
        for h in rtree.query(g):
            a = refe.geometry.iloc[h].intersection(g).area
            ov.append((a, h))
            if a > ba:
                best, ba = h, a
        # a tehsil named after an overlapping 1941 unit belongs to it (Gurgaon tehsil -> Gurgaon,
        # Tumkur -> Tumkur, Sirohi -> Sirohi State), whatever the map's ~12-15 km positional error
        tn = nm(re.sub(r"([a-z])([A-Z])", r"\1 \2", str(getattr(r, "NAME_3", "") or "")))
        if tn:
            named = [(a, h) for a, h in ov if a > 0.1 * g.area and (nm(refe.iloc[h]["Districts"]) == tn or
                     (len(tn) >= 4 and tn in [nm(w) for w in re.split(r"[\s\-/]+", str(refe.iloc[h]["Districts"]))]))]
            if named:
                best, ba = max(named)[1], max(ba, 0.5 * g.area)
                named_hits[0] += 1
        if best is not None and ba >= 0.5 * g.area:
            rr = refe.iloc[best]
            pieces.append((f"s{r.pid_}", g, rl.geometry, nm(rr["Districts"]), "census map 1941", rr["Type"] or "", rr["Districts"], int(best), rr["Admin Divi"]))
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
            pieces.append((f"o{i}", g, gl, nm(r.Districts), "census map 1941 (drawn directly)", r.Type or "", r.Districts, i, r._2))
            n_ov += 1
    print(f"{n_ov} small 1941 units drawn directly from the census map; {named_hits[0]} tehsils placed by their own name")
    # registry-wide name index (any year) so an untabulated state keeps its own identity
    code2gid = {f'{u["country"]}/{u["admin1"]}/{u["name"]}': u["gid"] for u in json.load(open(os.path.join(HERE, "out", "districts_svg.json")))["units"]}
    gid_geom0 = dict(zip(l2e["gid"], l2e.geometry))
    # every district/state (and city-district) in the registry, keyed by its name, its recorded
    # spellings and, for districts, the GADM name of its modern successor (e.g. "Hugli" -> Hooghly);
    # places with no modern mapping (several large states) are kept, without an area
    reg_names, reg_area = collections.defaultdict(list), {}
    for pid, p in places.items():
        if p["kind"] not in ("princely_state", "district", "town", "province"):
            continue
        gs = [gid_geom0[code2gid[m]] for m in (p["modern"] or "").split(";") if m in code2gid and code2gid[m] in gid_geom0]
        if gs:
            reg_area[pid] = unary_union(gs)
        keys = [p["name"]] + [a for a in p["aliases"].split(" | ") if a]
        if p["kind"] == "district" and len(gs) == 1:
            keys.append(re.sub(r"([a-z])([A-Z])", r"\1 \2", (p["modern"] or "").split("/")[-1]))
        for s_ in keys:
            if nm(s_):
                reg_names[nm(s_)].append(pid)
            w = re.sub(r"(?i)\b(district|state|prant|zilla|zila)\b", "", s_).split()
            if len(w) == 2:                       # "East Godavari" ~ "Godavari East"
                reg_names[nm(w[1] + " " + w[0])].append(pid)
    gid_geom = dict(zip(l2e["gid"], l2e.geometry))
    out, owner = {}, {}
    changes = {r["district_1941"]: r for r in csv.DictReader(open(os.path.join(HERE, "district_changes.csv")))}
    from evolution import ancestors as evo, units_in as evo_units
    carved = collections.Counter(); carved_log = []
    TOVR = list(csv.DictReader(open(os.path.join(HERE, "tehsil_overrides.csv"))))
    # historical taluks located on GADM level-3 units (build_subdiv): gid3 -> [place], place -> its districts/states
    _sd = os.path.join(HERE, "..", "census_site", "data", "subdiv.json")
    _sub = json.load(open(_sd)) if os.path.exists(_sd) else {"match": {}, "pl": {}}
    SUBMATCH = _sub["match"]
    SUBANC = {k: v.get("a", []) for k, v in _sub["pl"].items()}
    evo_used = collections.Counter()
    EVO_KIND = {"british districts": ("district", "town"), "princely states": ("princely_state",), "princely districts": ("district", "princely_state")}

    from rapidfuzz import process as _rp, fuzz as _rf
    reg_keys = list(reg_names)

    def reg_lookup(name, kinds, g, cross=False):
        """registry places for a period spelling. Exact normalized name (else closest spelling
        >= 85: Jullunder -> Jullundur, Ryepoor -> Raipur). A located place within 150 km of the
        polygon always beats one elsewhere or unlocated, and the stated type is preferred but not
        trusted blindly (sources mis-type units: Patna District as a state in 1931, Tehri Garhwal
        as a British district in 1921-31). Homonyms are told apart by location."""
        k = nm(name)
        CITY = ("madras", "calcutta", "bombay", "rangoon", "karachi")
        SMALL_PROVINCE = ("coorg", "delhi", "ajmermerwara", "panthpiploda", "tripura")
        def ok(q, ks):
            kd = places[q]["kind"]
            return kd in ks and (kd != "town" or k in CITY) and (kd != "province" or k in SMALL_PROVINCE)
        exact = list(dict.fromkeys(reg_names.get(k, []) + (reg_names.get(k + "city", []) if k in CITY else [])))
        if not exact and len(k) >= 4:
            for kk, sc, _ in _rp.extract(k, reg_keys, scorer=_rf.ratio, limit=3):
                if sc >= 85:
                    exact += reg_names[kk]
        anyk = ("district", "princely_state", "town", "province")
        near = lambda q: reg_area.get(q) is not None and reg_area[q].distance(g) < 150000
        # kinds are in priority order: a "princely district" is the district (Mysore District),
        # not its state (Mysore State), even though the state's area overlaps more
        # crossing unit types is allowed only for the evolution database's lineage, which
        # mis-types some units; the 1941 map's own types are trusted (Benares State is a state)
        pools = [[q for q in exact if ok(q, (kd,)) and near(q)] for kd in kinds] + ([[q for q in exact if ok(q, anyk) and near(q)]] if cross else [])
        for pool in pools:
            if pool:
                return [max(pool, key=lambda q: reg_area[q].intersection(g).area)]
        unlocated = [q for q in exact if (ok(q, kinds) or ok(q, ("province",))) and reg_area.get(q) is None]
        return unlocated[:1] if len(unlocated) == 1 else []

    def pick_ancestor(cands, g, label):
        """the registry place for the period unit this 1941 unit lay in; among several
        predecessors (a 1941 unit formed from two), the one covering most of the polygon,
        ties going to the predecessor that bears the 1941 unit's own name"""
        scored = []
        for adm, typ_, nme in cands:
            kinds = EVO_KIND.get(str(typ_).lower(), ("district", "princely_state"))
            for q in reg_lookup(nme, kinds, g, cross=True):
                a_ = reg_area[q].intersection(g).area if reg_area.get(q) is not None else 0
                scored.append((round(a_ / max(g.area, 1), 3), nm(nme) == nm(label), q))
        if scored:
            return max(scored)[2]
        same = [c for c in cands if nm(c[2]) == nm(label)]
        c = (same or cands)[0]
        return f"map1941:{c[2]}"
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
            pid_piece, _, _, key, src, typ, label, _row = pieces[idxs[0]][:8]
            g = unary_union([pieces[i][1] for i in idxs]) if len(idxs) > 1 else pieces[idxs[0]][1]
            pidx = idxs
            best = None
            if _row in FIXED_BEFORE and y not in ("all",) and int(y) < FIXED_BEFORE[_row][0]:
                own.setdefault(FIXED_BEFORE[_row][1], []).extend(idxs)   # e.g. Benares State, created 1911
                continue
            # lineage from the India State and District Evolution Database (primary source for
            # present-day India): the period unit this 1941 unit lay in during census year y
            if y not in ("all", "1941", "1951") and label and _row is not None and _row not in FIXED_ROWS:
                anc = evo(label, pieces[idxs[0]][8] if len(pieces[idxs[0]]) > 8 else None)
                if anc and anc.get(int(y)):
                    if len(anc[int(y)]) > 1 and len(idxs) > 1:
                        # several predecessors (e.g. 1901 Amraoti + Ellichpur -> 1941 Amraoti):
                        # each tehsil goes to the predecessor evidenced by a historical taluk located
                        # on it (subdivision matching), else to the one whose area covers it most,
                        # ties to the 1941 unit's own name
                        cand_ids = {q for c in anc[int(y)] for q in reg_names.get(nm(c[2]), [])}
                        for i in idxs:
                            g3 = pieces[i][0][1:]
                            ev = [a for t in SUBMATCH.get(g3, []) for a in SUBANC.get(t, []) if a in cand_ids]
                            q = ev[0] if ev else pick_ancestor(anc[int(y)], pieces[i][1], label)
                            if q:
                                own.setdefault(q, []).append(i)
                        evo_used[y] += 1
                        continue
                    best = pick_ancestor(anc[int(y)], g, label)
                    if best:
                        own.setdefault(best, []).extend(idxs)
                        evo_used[y] += 1
                        continue
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
                reg = reg_lookup(label, ("princely_state",) if typ == "Princely states" else ("district", "princely_state"), g)
                best = reg[0] if reg else f"map1941:{label}"
            elif best is None and typ == "British districts":
                # a 1941 district is itself, except before a recorded split/rename (district_changes.csv)
                ch = changes.get(label)
                if ch and int(y if y != "all" else 9999) < int(ch["first_census_as_own_unit"]):
                    best = ch["before_that_part_of"]
                else:
                    reg = reg_lookup(label, ("district", "town"), g)
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
        # reviewed tehsil-level assignments where the sources cannot separate two units that share a
        # modern successor (tehsil_overrides.csv, e.g. Merwara = Beawar tehsil until 1911)
        if y not in ("all",):
            by_id = {pc[0]: i for i, pc in enumerate(pieces)}
            for r in TOVR:
                if int(r["from_census"]) <= int(y) <= int(r["to_census"]) and "s" + r["gid3"] in by_id:
                    i = by_id["s" + r["gid3"]]
                    for pid in list(own):
                        if i in own[pid]:
                            own[pid].remove(i)
                    own.setdefault(r["place_id"], []).append(i)
            own = {pid: idxs for pid, idxs in own.items() if idxs}
        # carve-out: a British district the evolution database lists in year y but that owns no
        # territory (its 1941 polygon is gone — e.g. Narsinghpur, merged into Hoshangabad in 1932
        # but listed by the database) gets back the tehsils lying in its own mapped area, taken
        # only from districts (never from princely states)
        if y not in ("all", "1941", "1951"):
            for adm, typ_, nme in evo_units(int(y)):
                if str(typ_).lower() != "british districts":
                    continue
                for q in reg_names.get(nm(nme), []):
                    if places[q]["kind"] != "district" or q in own or reg_area.get(q) is None:
                        continue
                    take = []
                    for pid, idxs in own.items():
                        if places.get(pid, {}).get("kind") != "district":
                            continue
                        own_area = reg_area.get(pid)
                        for i in idxs:
                            rp = pieces[i][1].representative_point()
                            # only misassigned land: inside q's area and outside the holder's own
                            if pieces[i][0].startswith("s") and reg_area[q].contains(rp) and not (own_area is not None and own_area.contains(rp)):
                                take.append((pid, i))
                    if take:
                        for pid, i in take:
                            own[pid].remove(i)
                        own[q] = [i for _, i in take]
                        carved[y] += 1
                        carved_log.append((y, q, sorted({pid for pid, _ in take}), len(take)))
                    break
            own = {pid: idxs for pid, idxs in own.items() if idxs}
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
    print("1941 units placed by the evolution-database lineage, per year:", dict(sorted(evo_used.items())))
    print("districts restored by carve-out, per year:", dict(sorted(carved.items())))
    for row in carved_log:
        print("   carved", row)
    print(f"{len(out)} outlines from {len(pieces)} sub-district pieces ({sum(p[4] == 'census map 1941' for p in pieces)} identified on the 1941 map); sources {dict(src)}")
    print("units drawn per year:", {y: len(v) for y, v in A["hy"].items()})


if __name__ == "__main__":
    main()
