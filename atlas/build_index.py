#!/usr/bin/env python3
"""Stage 1: place registry, table->place links, and per-place age indicators.

Inputs : census_explorer.sqlite, atlas/out/gazetteer.csv (curated, validated),
         sendout_v1/provenance.csv + documents.csv (volume/year), GADM unit list.
Outputs (atlas/out/):
  places.csv        one row per historical place: id, name, kind, parent, aliases, modern
  table_places.csv  table_id, place_id, how (tag|title|rows|colg|sect|birthplace), source string
  place_metrics.csv table_id, place_id, child, old, sexratio, total, check, slice
"""
import csv, os, re, sqlite3, collections, unicodedata, json, sys, time, functools
T0 = time.time()
def log(m):
    print(f'[{time.time()-T0:6.0f}s] {m}', flush=True)
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
SV1 = os.path.join(ROOT, "export_final/sendout_final/sendout_v1")
DB = os.path.join(SV1, "census_explorer.sqlite")
OUT = os.path.join(HERE, "out")
sys.path.insert(0, HERE)
import age_metrics as am
from extract_candidates import norm, TITLE_PL

PLACE_KINDS = {"district", "sub_district", "town", "city_section", "princely_state", "province",
               "division", "agency", "natural_division"}
AGE_FAMILIES = {"age_civil_condition", "age_sex_population", "age_distribution", "proportional_age", "other_structured", "age_by_religion"}
SUBPOP = re.compile(r"(?i)insan|blind|deaf|leper|infirm|literat|educat|caste|tribe|race|occupation|livelihood|"
                    r"selected|urban|rural|town|city|europe|anglo|married females|widows|christian|jain|sikh|parsi|"
                    r"buddhist|musalman|muslim|muhammadan|mahomedan|hindu\b|animist|zoroast|jew")
ALLREL = re.compile(r"(?i)all religions|all races|all castes|all classes|\btotal\b")
RELIGION_SLICE = re.compile(r"(?i)\b(hindus?|musalmans?|muslims?|muhammadans?|mahomedans?|christians?|jains?|sikhs?|parsis?|"
                            r"buddhists?|animists?|animistic|tribal|zoroastrians?|jews?|aryas?|brahmos?|others?|minor religions)\b")
DISTRICT_CEILING = 6_000_000   # the largest districts (e.g. Mymensingh, 1931) held under ~5 million
DISTRICT_FLOOR = 30000   # a whole district: the smallest (e.g. Simla, 1901: 40,351) exceed this
BOUNDS = {"child": (10, 45), "old": (0.5, 20), "sexratio": (600, 1300)}
BIRTH = re.compile(r"(?i)birth|born|immigra|emigra|migra|nativity|where enumerated")
STRIP_PARENT = re.compile(r"(?i)\b(state|presidency|province|agency|residency|the)\b")


def slug(s):
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    return re.sub(r"[^a-z0-9]+", "-", s).strip("-")


JUNK_ALIAS = re.compile(r"(?i)\s[-–—·/|]\s|,|\(|\)|\.\s*-|\b(males?|females?|persons|total|hindus?|musalmans?|muslims?|"
                        r"muhammadans?|mahomedans?|christians?|jains?|sikhs?|parsis?|buddhists?|animists?|others?|"
                        r"all religions|urban|rural|continued|contd|concluded|brahmins?|table|part)\b")


TITLE_WORDS = re.compile(r"(?i)\b(in|of|each|the|and|by|sex|condition|population|classes|showing|statement|age|ages)\b")


def clean_aliases(labels, name):
    """Keep strings that are spellings of the place; drop row labels that carry a qualifier
    (caste/religion/sex/age band/title text), e.g. 'Nowgong - Dom', 'Nowgong 0-10',
    'Sex and Civil Condition-CUDDAPAH'. Case variants collapse to one."""
    cand = {}
    for a in labels:
        a = re.sub(r"\s+", " ", a.strip().strip(".:-"))
        if (not a or JUNK_ALIAS.search(a) or re.search(r"\d", a) or TITLE_WORDS.search(a)
                or len(a.split()) > 3 or len(a) > 40):
            continue
        key = a.lower()
        if key not in cand or (a.istitle() and not cand[key].istitle()):
            cand[key] = a if not a.isupper() else a.title()
    keys = sorted(cand, key=len)
    keep = [k for k in keys if not any(k != j and k.startswith(j + " ") for j in keys)]
    bare = re.sub(r"(?i)\s*\b(district|state|taluk|taluka|tahsil|division|agency|town|city|pargana|prant|township)\b\s*", " ", name).strip().lower()
    return sorted(cand[k] for k in keep if k not in (name.lower(), bare))


def build_registry():
    g = list(csv.DictReader(open(os.path.join(OUT, "gazetteer.csv"))))
    by_name = collections.defaultdict(list)
    for r in g:
        if r["kind"] not in PLACE_KINDS:
            continue
        name = (r["hist_name"] or r["label"]).strip()
        by_name[name].append(r)
    places, key2pid = {}, {}
    for name, rs in by_name.items():
        # same name in different modern states = different places (e.g. two Aurangabads)
        clusters = collections.defaultdict(list)
        for r in rs:
            m = (r["modern"] or "").split(";")[0]
            clusters["/".join(m.split("/")[:2]) if m else ""].append(r)
        if "" in clusters and len(clusters) > 1:          # unlocated strings join the biggest cluster
            big = max((k for k in clusters if k), key=lambda k: len(clusters[k]))
            clusters[big] += clusters.pop("")
        for ck, crs in clusters.items():
            parent = collections.Counter(r["hist_parent"] for r in crs if r["hist_parent"]).most_common(1)
            parent = parent[0][0] if parent else ""
            pid = slug(name) if len(clusters) == 1 else f"{slug(name)}--{slug(parent or ck)}"
            kind = collections.Counter(r["kind"] for r in crs).most_common(1)[0][0]
            mods = collections.Counter()
            for r in crs:
                if r["modern"]:
                    mods[r["modern"]] += int(r["n_tables"])
            conf = collections.Counter(r["confidence"] for r in crs if r["modern"]).most_common(1)
            places[pid] = dict(id=pid, name=name, kind=kind, parent=parent,
                               aliases=clean_aliases({r["label"] for r in crs} | {v.strip() for r in crs for v in r["variants"].split(" | ") if v.strip()}, name)[:12],
                               modern=mods.most_common(1)[0][0] if mods else "",
                               confidence=conf[0][0] if conf else "",
                               note=next((r["note"] for r in crs if r["note"]), ""),
                               n_strings=len(crs))
            for r in crs:
                key2pid[r["key"]] = pid
    # reviewed corrections to the curated modern mapping (justified, versioned)
    mo = os.path.join(HERE, "modern_overrides.csv")
    if os.path.exists(mo):
        for r in csv.DictReader(open(mo)):
            if r["place_id"] in places:
                places[r["place_id"]].update(modern=r["modern"], note=r["justification"][:200], confidence="high")
    # parent links: hist_parent text -> a province/state/agency place id
    index = {}
    for p in places.values():
        if p["kind"] in ("province", "princely_state", "agency", "division"):
            for s in [p["name"]] + p["aliases"]:
                index.setdefault(norm(STRIP_PARENT.sub(" ", s)), p["id"])
    for p in places.values():
        p["parent_id"] = index.get(norm(STRIP_PARENT.sub(" ", p["parent"])), "") if p["parent"] else ""
        if p["parent_id"] == p["id"]:
            p["parent_id"] = ""
    return places, key2pid


def doc_years():
    yr = {}
    for r in csv.DictReader(open(os.path.join(SV1, "documents.csv"))):
        yr[r["dochash"]] = r["year"]
    return yr


def main():
    places, key2pid = build_registry()
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    tables = {r["table_id"]: dict(r) for r in con.execute(
        "select table_id,title,canonical_name,year,status,broad_region,entity_type,table_family from tables")}
    dy = doc_years()

    @functools.lru_cache(maxsize=None)
    def resolve(s):
        return key2pid.get(norm(s or ""))

    links = collections.defaultdict(dict)          # tid -> pid -> (how, src)
    PRI = {"tag": 0, "title": 1, "rows": 2, "colg": 3, "sect": 4}

    TOWNS = re.compile(r"(?i)\btowns?\b|municipal|\bcities\b|\bcity\b|cantonment")
    town_of = {}
    for p in places.values():
        if p["kind"] == "town":
            town_of.setdefault(norm(re.sub(r"(?i)\b(town|city|municipality)\b", "", p["name"])), p["id"])

    def link(tid, s, how):
        pid = resolve(s)
        if not pid:
            return
        t = tables.get(tid)
        if how in ("rows", "sect", "colg") and t and TOWNS.search(t["title"] or "") and places[pid]["kind"] == "district":
            # in a table of towns, "Sylhet" is Sylhet town, not Sylhet District
            pid = town_of.get(norm(re.sub(r"(?i)\bdistrict\b", "", places[pid]["name"])))
            if not pid:
                return
        if t and BIRTH.search(t["title"] or "") and how != "tag":
            how = "birthplace"
        cur = links[tid].get(pid)
        if cur is None or PRI.get(how, 9) < PRI.get(cur[0], 9):
            links[tid][pid] = (how, s)

    for tid, t in tables.items():
        link(tid, t["canonical_name"], "tag")
        for m in TITLE_PL.finditer(t["title"] or ""):
            link(tid, m.group(1), "title")
    log("tags/titles linked")
    for tid, s in con.execute("select distinct table_id,row_label_std from facts where row_dimension='other' "
                              "and column_dimension in ('age','civil_condition','sex')"):
        link(tid, s, "rows")
    for tid, s in con.execute("select distinct table_id,column_group_std from facts where row_dimension='age' "
                              "and column_dimension='other'"):
        link(tid, s, "colg")
    for tid, s in con.execute("select distinct table_id,section from facts where row_dimension='age' and section!=''"):
        link(tid, s, "sect")

    log("facts linked")
    with open(os.path.join(OUT, "table_places.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["table_id", "place_id", "how", "source"])
        for tid, d in links.items():
            for pid, (how, s) in d.items():
                w.writerow([tid, pid, how, s])

    # ---- age indicators per (table, place) ----
    cols = ("table_id,row_dimension,row_label_std,section,civil_condition,religion,column_dimension,"
            "column_group_std,sex,resolved_age_lower,resolved_age_upper,resolved_age_label_std,value,measure_unit")
    groups = collections.defaultdict(list)
    for r in con.execute(f"select {cols} from facts where row_dimension='age' or column_dimension='age'"):
        d = dict(r)
        tid = d["table_id"]
        if d["row_dimension"] != "age":
            pid = resolve(d["row_label_std"])
        else:
            pid = resolve(d["section"])
            if pid and places[pid]["kind"] not in ("district", "princely_state", "province", "sub_district", "town", "division"):
                pid = None
            if not pid and d["column_dimension"] != "age":
                pid = resolve(d["column_group_std"])
            if not pid:
                pid = resolve(tables.get(tid, {}).get("canonical_name"))
        if pid and places[pid]["kind"] == "district" and TOWNS.search(tables.get(tid, {}).get("title") or ""):
            pid = None      # a cities/towns table: the "Madura" here is Madura city, not the district
        if pid:
            groups[(tid, pid)].append(d)
    log(f"grouped {len(groups)} table-place groups")
    # ---- multi-page runs: wide tables print their age columns across several pages,
    # so one place's full age distribution exists only across the run. A run is the
    # set of pages in one volume whose titles share a stem ("...-continued" removed).
    from position import positions
    pos = positions(con.execute("select table_id,source_image,member_images from tables"))
    def stem(t):
        t = re.sub(r"(?i)[-—.(\s]*(continued|contd|concluded|concld|cont)\b[.)]*", " ", t or "")
        return re.sub(r"[^a-z]", "", t.lower())[:40]
    cand_runs = collections.defaultdict(list)
    for tid, t in tables.items():
        q = pos.get(tid)
        if q and q["pdf_page"] is not None and stem(t["title"]) and not tid.startswith("asm_"):
            cand_runs[(q["doc"], stem(t["title"]))].append((q["pdf_page"], tid))
    run_of = {}
    for key, mem in cand_runs.items():          # split into runs of consecutive pages (gap <= 3)
        mem.sort()
        cur = [mem[0]]
        for a, b in zip(mem, mem[1:]):
            if b[0] - a[0] <= 3:
                cur.append(b)
            else:
                if len(cur) > 1:
                    for _, tid in cur:
                        run_of[tid] = (key, cur[0][0])
                cur = [b]
        if len(cur) > 1:
            for _, tid in cur:
                run_of[tid] = (key, cur[0][0])
    by_run = collections.defaultdict(list)
    for (tid, pid), fs in list(groups.items()):
        if tid in run_of:
            by_run[(run_of[tid], pid)].append((tid, fs))
    run_groups = {}
    for (rk, pid), parts in by_run.items():
        if len(parts) > 1:
            first = min(parts, key=lambda x: pos[x[0]]["pdf_page"])[0]
            run_groups[(first, pid)] = ([f for _, fs in parts for f in fs], len(parts))
    log(f"{len(run_groups)} multi-page run groups")
    with open(os.path.join(OUT, "place_metrics.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["table_id", "place_id", "child", "old", "sexratio", "total", "check", "slice"])
        n = 0
        rej = collections.Counter()
        items = [((tid, pid), fs, 1) for (tid, pid), fs in groups.items()] + \
                [((tid, pid), fs, n) for (tid, pid), (fs, n) in run_groups.items()]
        done = set()
        for (tid, pid), fs, npages in items:
            if npages == 1 and (tid, pid) in run_groups:
                continue                                  # the run version supersedes the single page
            t = tables.get(tid, {})
            title = t.get("title") or ""
            if BIRTH.search(title):
                continue
            # whole-population age tables only: a district's age structure, not a caste's or the literate's
            if t.get("table_family") not in AGE_FAMILIES or (SUBPOP.search(title) and not ALLREL.search(title)
                                                             and t.get("table_family") != "age_by_religion"):
                rej["not a whole-population age table"] += 1
                continue
            m = am.best_metrics(fs)
            if not m:
                continue
            if RELIGION_SLICE.search(m["slice"]) and not ALLREL.search(m["slice"]):
                rej["single-religion slice"] += 1
                continue
            if SUBPOP.search(title) and not ALLREL.search(title) and not ALLREL.search(m["slice"]):
                rej["sub-population table"] += 1
                continue
            if places[pid]["kind"] == "district" and m.get("unit") == "count" and m["total"] < DISTRICT_FLOOR:
                rej["too small to be the whole district (a town/subset row)"] += 1
                continue
            if places[pid]["kind"] == "district" and m.get("unit") == "count" and m["total"] > DISTRICT_CEILING:
                rej["too large for a district (a provincial table with a wrong tag)"] += 1
                continue
            out = [k for k, (lo, hi) in BOUNDS.items() if m.get(k) is not None and not lo <= m[k] <= hi]
            if out:
                rej["outside plausibility bounds"] += 1
                continue
            if m:
                n += 1
                chk = m["check"] + (f"; merged across {npages} pages of one printed table" if npages > 1 else "")
                w.writerow([tid, pid] + [round(m[k], 3) if m.get(k) is not None else "" for k in ("child", "old", "sexratio")]
                           + [round(m["total"]), chk, m["slice"]])
    with open(os.path.join(OUT, "places.csv"), "w", newline="") as f:
        cols = ["id", "name", "kind", "parent", "parent_id", "aliases", "modern", "confidence", "note", "n_strings"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for p in places.values():
            w.writerow({**p, "aliases": " | ".join(p["aliases"])})
    kinds = collections.Counter(p["kind"] for p in places.values())
    print(f"places: {len(places)} {dict(kinds.most_common())}")
    print(f"table-place links: {sum(len(d) for d in links.values())} over {len(links)} tables; "
          f"by how: {dict(collections.Counter(h for d in links.values() for h, _ in d.values()))}")
    print(f"age-indicator groups: {n} (of {len(groups)} candidate table-place groups); rejected: {dict(rej)}")


if __name__ == "__main__":
    main()
