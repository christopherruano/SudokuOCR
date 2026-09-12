#!/usr/bin/env python3
"""Phase 2: match historical districts to modern GADM units.

Reads out/district_universe.csv (Phase 1) and GADM 4.1 boundaries
(mapdata/gadm41_*.json). Emits out/concordance_draft.csv with one row per
historical district: best modern candidate(s), score, and evidence tier:
  A  exact normalized name match within the expected modern region
  B  fuzzy match >=90 or documented rename (renames.csv), region-consistent
  C  fuzzy 80-89 or region-inconsistent — needs review
  U  no candidate >=80
Manual decisions go in match_overrides.csv (justified), applied last.
"""
import csv, json, os, re, unicodedata
from rapidfuzz import fuzz, process

HERE = os.path.dirname(os.path.abspath(__file__))
MAPD = os.path.join(HERE, "..", "mapdata")

# historical parent/province -> plausible modern (country, admin1-substring) hints
REGION_HINTS = {
    "Madras": [("IND","TamilNadu"),("IND","AndhraPradesh"),("IND","Telangana"),("IND","Kerala"),("IND","Karnataka"),("IND","Odisha")],
    "Bombay": [("IND","Maharashtra"),("IND","Gujarat"),("IND","Karnataka"),("PAK","Sindh")],
    "Bengal": [("IND","WestBengal"),("BGD",""),("IND","Bihar"),("IND","Odisha"),("IND","Jharkhand"),("IND","Assam")],
    "Punjab": [("IND","Punjab"),("PAK","Punjab"),("IND","Haryana"),("IND","HimachalPradesh"),("IND","NCTofDelhi"),("PAK","KhyberPakhtunkhwa")],
    "United Provinces": [("IND","UttarPradesh"),("IND","Uttarakhand")],
    "N.-W. P": [("IND","UttarPradesh"),("IND","Uttarakhand")],
    "Burma": [("MMR","")], "Assam": [("IND","Assam"),("IND","Meghalaya"),("IND","Nagaland"),("IND","Mizoram"),("BGD","Sylhet")],
    "Central Provinces": [("IND","MadhyaPradesh"),("IND","Chhattisgarh"),("IND","Maharashtra")],
    "Mysore": [("IND","Karnataka")], "Hyderabad": [("IND","Telangana"),("IND","Maharashtra"),("IND","Karnataka")],
    "Travancore": [("IND","Kerala"),("IND","TamilNadu")], "Cochin": [("IND","Kerala")],
    "Baroda": [("IND","Gujarat")], "Rajputana": [("IND","Rajasthan")], "Central India": [("IND","MadhyaPradesh")],
    "Kashmir": [("IND","JammuandKashmir"),("IND","Ladakh"),("PAK","AzadKashmir"),("PAK","Gilgit-Baltistan")],
    "Bihar": [("IND","Bihar"),("IND","Jharkhand"),("IND","Odisha")],
    "Sind": [("PAK","Sindh")], "NWFP": [("PAK","KhyberPakhtunkhwa")],
}

def norm(s):
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    s = re.sub(r"\b(district|division|the)\b", "", s)
    return re.sub(r"[^a-z]", "", s)

def load_units():
    units = []
    for c, lvl in [("IND",2),("PAK",3),("BGD",2),("MMR",2)]:
        d = json.load(open(os.path.join(MAPD, f"gadm41_{c}_{lvl}.json")))
        for f in d["features"]:
            p = f["properties"]
            nm = p[f"NAME_{lvl}"]; var = p.get(f"VARNAME_{lvl}") or ""
            units.append(dict(country=c, admin1=p["NAME_1"], name=nm,
                              variants=[v for v in ([nm]+var.split("|")) if v],
                              gid=p[f"GID_{lvl}"]))
    return units

def main():
    units = load_units()
    lookup = []  # (normname, unit)
    for u in units:
        for v in u["variants"]:
            lookup.append((norm(v), u))
    uni = list(csv.DictReader(open(os.path.join(HERE, "out", "district_universe.csv"))))
    aliases = {}
    ap = os.path.join(HERE, "aliases.csv")
    if os.path.exists(ap):
        for a in csv.DictReader(open(ap)):
            aliases.setdefault(a["canonical_name"], []).append((a["alias"], a["reason"]))
    mov = {}
    mp = os.path.join(HERE, "match_overrides.csv")
    if os.path.exists(mp):
        mov = {m["canonical_name"]: m for m in csv.DictReader(open(mp))}
    rows = []
    for d in uni:
        if d["canonical_name"] in mov:
            m = mov[d["canonical_name"]]
            rows.append(dict(d, tier=m["tier"], match=m["matches"],
                             modern_admin1=m["modern_admin1"], country=m["country"],
                             gid="", score="", method=f'match_override: {m["justification"]}'))
            continue
        hn = norm(d["canonical_name"])
        hints = REGION_HINTS.get(d["parent"].strip(), [])
        def in_region(u):
            return not hints or any(u["country"]==c and (not a or a.lower() in u["admin1"].lower()) for c,a in hints)
        # exact pass (canonical name, then documented aliases)
        tried = [(hn, "exact_norm+region", "A")]
        for alias, reason in aliases.get(d["canonical_name"], []):
            tier = "A" if reason == "translit" else "B"
            tried.append((norm(alias), f"alias:{reason}", tier))
        hit = None
        for nn_try, method, tier in tried:
            exact_in = [u for nn,u in lookup if nn==nn_try and in_region(u)]
            if exact_in:
                hit = (exact_in[0], method, tier); break
        if hit:
            u, method, tier = hit
            rows.append(dict(d, tier=tier, match=u["name"], modern_admin1=u["admin1"],
                             country=u["country"], gid=u["gid"], score=100, method=method))
            continue
        # fuzzy pass
        cands = [(fuzz.ratio(hn, nn), nn, u) for nn,u in lookup if abs(len(nn)-len(hn))<6]
        cands.sort(key=lambda x:-x[0])
        best = None
        for sc, nn, u in cands[:25]:
            if sc < 80: break
            if in_region(u): best = (sc,u); break
        if not best and cands and cands[0][0] >= 80:
            best = (cands[0][0], cands[0][2])  # region-inconsistent
            tier = "C"
        if best:
            sc, u = best
            tier = "B" if sc >= 90 and in_region(u) else "C"
            rows.append(dict(d, tier=tier, match=u["name"], modern_admin1=u["admin1"],
                             country=u["country"], gid=u["gid"], score=sc,
                             method="fuzzy+region" if in_region(u) else "fuzzy_only"))
        else:
            rows.append(dict(d, tier="U", match="", modern_admin1="", country="",
                             gid="", score=0, method="none"))
    cols = list(uni[0].keys())+["tier","match","modern_admin1","country","gid","score","method"]
    with open(os.path.join(HERE,"out","concordance_draft.csv"),"w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=cols); w.writeheader(); w.writerows(rows)
    from collections import Counter
    print(Counter(r["tier"] for r in rows))
    for r in rows:
        if r["tier"] in ("C","U"): print(f'  {r["tier"]} {r["canonical_name"]} ({r["parent"]}) -> {r["match"]} {r["score"]}')
if __name__ == "__main__":
    main()
