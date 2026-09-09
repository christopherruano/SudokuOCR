#!/usr/bin/env python3
"""Deep place tagging for the census explorer: read row labels (facts.row_group_value,
row_dimension='other') and section names per table, resolve them against the geo dict plus an
LLM-built supplemental gazetteer, and emit export/deep_place_tags.csv - one row per table_id
with the places it mentions at each level (province / division / district / taluk / town, plus
foreign countries in comparative tables). Read-only over the sqlite; NEW output files only.

Two-step flow:
  python3 build_place_tags.py --emit-misses   writes export/deep_place_misses.json (labels the
                                              geo dict can't resolve, with province/year context)
  <subagents classify misses>              -> export/deep_place_gazetteer.json
  python3 build_place_tags.py                 writes export/deep_place_tags.csv using both
"""
import argparse, csv, json, os, re, sqlite3, sys, unicodedata
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(ROOT, "export_final/sendout_final/sendout_v1/census_explorer.sqlite")
GAZ = os.path.join(ROOT, "export/deep_place_gazetteer.json")
MISSES = os.path.join(ROOT, "export/deep_place_misses.json")
OUT = os.path.join(ROOT, "export/deep_place_tags.csv")

# geo entity_type -> output level bucket
LEVEL = {"all_india": "national", "british_province": "province", "modern_state": "province",
         "princely_state": "state", "agency": "agency", "native_states_agg": "agency",
         "division": "division", "district": "district", "sub_district": "taluk", "urban": "town"}
DEPTH = {"national": 0, "country": 0, "province": 1, "state": 1, "agency": 1,
         "division": 2, "district": 3, "taluk": 4, "town": 4}


def deaccent(s):
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def norm(s):
    s = deaccent((s or "").strip().lower())
    s = re.sub(r"^\s*[\divxlc]+[\.\)]\s+", "", s)          # leading numbering "1. " / "iv) "
    s = re.sub(r"[\*†‡§]+", "", s)                          # footnote marks
    s = re.sub(r"[\s–—_-]+", " ", s).strip(" .,:;|")
    return s


SUFFIX = re.compile(r"\s+(district|districts|taluk|taluka|taluq|tahsil|tehsil|state|city|town|division|agency|proper)$")
SEXTAIL = re.compile(r"\s*[-–—(]*\s*(males?|females?|persons)[\s.)]*$")
# non-place row labels (totals, ages, religions, layout rows, measures); '\d+ parganas' is a
# real Bengal district, so digits-lead labels get a parganas carve-out
STOP = re.compile(
    r"^(row \d+|total.*|all .*|grand total|urban|rural|males?|females?|persons|"
    r"hindu.*|musalman.*|muslim.*|christian.*|sikh.*|jain.*|buddhist.*|parsi.*|animist.*|tribal.*|"
    r"others?|unspecified|not stated|(?!\d+ parganas)\d+.*|age .*|(?!24 )\S*\d-\d.*|mean .*|number .*|"
    r"proportion .*|percentage .*|variation .*|net .*|actual .*|density.*|population.*|"
    r"literate.*|illiterate.*|married.*|unmarried.*|widowed.*|infirm.*|insane.*|deaf.*|blind.*|leper.*)$")


def load_geo(con):
    """normalized name -> (canonical, level). Skips non-geography and 'unresolved' rows (the
    latter carry no level; the LLM gazetteer re-resolves them)."""
    geo = {}
    for raw, canon, etype, isg in con.execute(
            "SELECT raw_geo_name, canonical_name, entity_type, is_geography FROM geo"):
        if (isg or "") != "1" or not (canon or "").strip() or etype not in LEVEL:
            continue
        for key in {norm(raw), norm(canon)}:
            for k in {key, SUFFIX.sub("", key)}:
                if k and k not in geo:
                    geo[k] = (canon.strip(), LEVEL[etype])
    return geo


def load_gazetteer():
    """LLM supplement: normalized label -> {canonical, level} (level 'not_place' drops it)."""
    if not os.path.exists(GAZ):
        return {}
    out = {}
    for k, v in json.load(open(GAZ)).items():
        lvl = v.get("level")
        if lvl in DEPTH or lvl == "not_place":
            out[norm(k)] = (v.get("canonical") or k, lvl)
    return out


def resolve(label, geo, gaz):
    n = norm(label)
    if not n or STOP.match(n):
        return None
    for cand in (n, SEXTAIL.sub("", n), SUFFIX.sub("", SEXTAIL.sub("", n))):
        if cand in geo:
            return geo[cand]
        if cand in gaz:
            canon, lvl = gaz[cand]
            return None if lvl == "not_place" else (canon, lvl)
    return ("?" + n, "unresolved")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emit-misses", action="store_true",
                    help="write the unresolved-label worklist for subagent classification")
    args = ap.parse_args()

    con = sqlite3.connect(DB)
    geo = load_geo(con)
    gaz = load_gazetteer()
    print(f"geo keys: {len(geo)}, gazetteer keys: {len(gaz)}", file=sys.stderr)

    # per-table metadata straight from facts (canonical_name/year are constant per table)
    tmeta = {tid: (yr, prov or "") for tid, yr, prov in con.execute(
        "SELECT table_id, MAX(year), MAX(canonical_name) FROM facts GROUP BY table_id")}

    pairs = con.execute("SELECT DISTINCT table_id, row_group_value FROM facts "
                        "WHERE row_dimension='other' AND row_group_value != ''").fetchall()
    secs = con.execute("SELECT DISTINCT table_id, section FROM facts WHERE section != ''").fetchall()

    tags = defaultdict(lambda: defaultdict(set))   # table_id -> level -> {canonical}
    miss_ctx = defaultdict(Counter)                # normalized miss -> Counter(province@year)
    for tid, v in pairs:
        r = resolve(v, geo, gaz)
        if not r:
            continue
        canon, lvl = r
        if lvl == "unresolved":
            yr, prov = tmeta.get(tid, ("", ""))
            miss_ctx[canon[1:]][f"{prov}@{yr}"] += 1
        else:
            tags[tid][lvl].add(canon)
    # sections: deterministic hits only (they are mostly religion/sex/layout headers)
    for tid, s in secs:
        n = norm(s)
        if len(n) > 40 or STOP.match(n) or not n:
            continue
        hit = geo.get(n) or geo.get(SUFFIX.sub("", n))
        if hit:
            tags[tid][hit[1]].add(hit[0])

    if args.emit_misses:
        work = [{"label": lbl, "n_tables": sum(ctx.values()),
                 "context": [c for c, _ in ctx.most_common(3)]}
                for lbl, ctx in miss_ctx.items() if sum(ctx.values()) >= 2]
        work.sort(key=lambda w: -w["n_tables"])
        json.dump(work, open(MISSES, "w"), indent=1)
        print(f"wrote {MISSES}: {len(work)} labels (>=2 tables) of {len(miss_ctx)} distinct misses")
        return

    with open(OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["table_id", "year", "volume_place", "national", "provinces", "states",
                    "agencies", "divisions", "districts", "taluks", "towns", "countries",
                    "deepest_level", "n_unresolved_labels"])
        n_deep = Counter()
        unres_per_table = Counter()
        for lbl, ctx in miss_ctx.items():
            pass
        # count unresolved per table in a second cheap pass
        for tid, v in pairs:
            r = resolve(v, geo, gaz)
            if r and r[1] == "unresolved":
                unres_per_table[tid] += 1
        for tid in sorted(tmeta):
            yr, prov = tmeta[tid]
            t = tags.get(tid, {})
            levels_present = [l for l in t if l in DEPTH]
            deepest = max(levels_present, key=lambda l: DEPTH[l], default="")
            n_deep[deepest] += 1
            w.writerow([tid, yr, prov,
                        "; ".join(sorted(t.get("national", []))),
                        "; ".join(sorted(t.get("province", []))),
                        "; ".join(sorted(t.get("state", []))),
                        "; ".join(sorted(t.get("agency", []))),
                        "; ".join(sorted(t.get("division", []))),
                        "; ".join(sorted(t.get("district", []))),
                        "; ".join(sorted(t.get("taluk", []))),
                        "; ".join(sorted(t.get("town", []))),
                        "; ".join(sorted(t.get("country", []))),
                        deepest, unres_per_table.get(tid, 0)])
        print(f"wrote {OUT}: {len(tmeta)} tables; deepest-level counts: "
              f"{dict(sorted(n_deep.items(), key=lambda x: -x[1]))}")


if __name__ == "__main__":
    main()
