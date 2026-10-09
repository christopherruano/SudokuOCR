#!/usr/bin/env python3
"""Stage 2: turn the index into the site's data files and downloadable CSVs.

Writes into census_site/data/:
  places.json   place registry + every table each place appears in (place.html, places.html)
  volumes.json  every source volume with its tables in census order + page gaps (volume.html)
  atlas.json    per modern district x census year: coverage class + age indicators (atlas.html)
  place_inventory.csv, district_concordance.csv, district_age_indicators.csv, table_places.csv
"""
import csv, json, os, re, sqlite3, hashlib, collections, math
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
SV1 = os.path.join(ROOT, "export_final/sendout_final/sendout_v1")
DB = os.path.join(SV1, "census_explorer.sqlite")
SITE = os.path.join(ROOT, "census_site")
OUT = os.path.join(HERE, "out")
SITE_STATUSES = ("confirmed_sent", "recovered", "keeper_accepted", "keeper_review", "assembled")
YEARS = ["1872", "1881", "1891", "1901", "1911", "1921", "1931", "1941", "1951"]
LOCAL_KINDS = ("district", "princely_state")          # units whose figures stand for a district
SUB_KINDS = ("sub_district", "town", "city_section")
CLASSES = {"district": "District", "sub_district": "Sub-district (taluk/tahsil/pargana/township)",
           "town": "Town / city", "city_section": "City ward / section", "princely_state": "Princely state",
           "province": "Province / presidency", "division": "Division", "agency": "Agency",
           "natural_division": "Census natural division"}
COUNTRIES = {"IND": "India", "PAK": "Pakistan", "BGD": "Bangladesh", "MMR": "Myanmar"}
AGE_TITLE = re.compile(r"(?i)\bage\b|ages\b|table\s+vii\b|civil condition")


def safe_filename(t):
    base = f"{t['year']}_{t['canonical_name']}_{(t['title'] or '')[:40]}"
    return (re.sub(r"[^A-Za-z0-9._-]+", "_", base).strip("_")[:90]) or "table"


def census_year(y):
    """Map a table year to its census round (1868/1872 -> 1872, 1950 -> 1951 ...)."""
    try:
        y = int(str(y)[:4])
    except (TypeError, ValueError):
        return ""
    best = min(YEARS, key=lambda c: abs(int(c) - y))
    return best if abs(int(best) - y) <= 4 else ""


def main():
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    excl = {l.strip() for l in open(os.path.join(SV1, "static_exclude.txt")) if l.strip()}
    superseded = set(json.load(open(os.path.join(OUT, "superseded_ids.json"))))   # same page re-read as a newer table
    tabs = {}
    for t in con.execute("select table_id,title,canonical_name,year,status,broad_region,entity_type,"
                         "table_family,canonical_topic,dims_present from tables"):
        t = dict(t)
        if t["status"] not in SITE_STATUSES or t["table_id"] in superseded:
            continue
        h = hashlib.md5(t["table_id"].encode()).hexdigest()[:7]
        pg = f"{safe_filename(t)[:70]}_{h}"
        t["page"] = pg if (t["table_id"] not in excl and os.path.exists(os.path.join(SITE, "tables", pg + ".html"))) else ""
        tabs[t["table_id"]] = t

    # volume / order: PDF + PDF page from page_number_status (authoritative), printed page from provenance
    import sys; sys.path.insert(0, HERE)
    from position import positions
    pos = positions(con.execute("select table_id,source_image,member_images from tables"))
    docs = {r["dochash"]: r for r in csv.DictReader(open(os.path.join(SV1, "documents.csv")))}
    vols_meta = {r["filename"]: r for r in csv.DictReader(open(os.path.join(SV1, "cc_volumes.csv")))}
    doc_pdf = {}
    for tid, t in tabs.items():
        q = pos.get(tid) or {}
        t["doc"] = q.get("doc") or (re.search(r"([0-9a-f]{16})", tid).group(1) if re.search(r"([0-9a-f]{16})", tid) else "")
        if q.get("pdf"):
            doc_pdf.setdefault(t["doc"], q["pdf"])
        t["pg"] = q.get("pg") or ""
        t["seq"] = q["pdf_page"] if q.get("pdf_page") is not None else 99999
        y = t["year"] or (docs.get(t["doc"], {}).get("year")) or (vols_meta.get(doc_pdf.get(t["doc"], ""), {}).get("year"))
        t["cy"] = census_year(y)
    for d, f in doc_pdf.items():                        # volumes known only from page_number_status
        if d not in docs:
            vm = vols_meta.get(f, {})
            docs[d] = {"dochash": d, "source_pdf": f, "year": vm.get("year", ""), "region": vm.get("province_or_volume", "")}
        elif not docs[d].get("source_pdf"):
            docs[d]["source_pdf"] = f

    # places + links + metrics
    places = {r["id"]: r for r in csv.DictReader(open(os.path.join(OUT, "places.csv")))}
    links = collections.defaultdict(list)          # pid -> [(tid, how)]
    tplaces = collections.defaultdict(list)
    for r in csv.DictReader(open(os.path.join(OUT, "table_places.csv"))):
        if r["table_id"] in tabs:
            links[r["place_id"]].append((r["table_id"], r["how"]))
            if r["how"] != "birthplace":
                tplaces[r["table_id"]].append(r["place_id"])
    metrics = collections.defaultdict(list)        # (pid, cy) -> [metric rows]
    age_tp = set()
    for r in csv.DictReader(open(os.path.join(OUT, "place_metrics.csv"))):
        t = tabs.get(r["table_id"])
        if not t:
            continue
        age_tp.add((r["table_id"], r["place_id"]))
        metrics[(r["place_id"], t["cy"])].append(r)

    def is_age(tid, pid=None):
        t = tabs[tid]
        if pid is not None and (tid, pid) in age_tp:
            return True
        return bool(AGE_TITLE.search(t["title"] or "")) or "age" in (t["dims_present"] or "")

    # children
    kids = collections.defaultdict(list)
    for p in places.values():
        if p["parent_id"] and p["parent_id"] in places:
            kids[p["parent_id"]].append(p["id"])

    # ---- table list (compact) shared by places.json/volumes.json ----
    tid_list = sorted(tabs, key=lambda x: (tabs[x]["doc"], tabs[x]["seq"]))
    tix = {tid: i for i, tid in enumerate(tid_list)}
    T = [{"p": tabs[t]["page"], "t": (tabs[t]["title"] or "")[:120], "y": tabs[t]["cy"], "d": tabs[t]["doc"],
          "pg": tabs[t]["pg"], "o": f"{tabs[t]['seq']:05d}", "g": tabs[t]["canonical_name"] or "",
          "a": 1 if is_age(t) else 0} for t in tid_list]

    # ---- places.json ----
    P = {}
    for pid, p in places.items():
        L = links.get(pid, [])
        if not L and not kids.get(pid):
            continue
        own = [(tid, how) for tid, how in L if how != "birthplace"]
        cy = collections.defaultdict(lambda: [0, 0])
        for tid, how in own:
            y = tabs[tid]["cy"]
            if y:
                cy[y][0] += 1
                cy[y][1] += 1 if is_age(tid, pid) else 0
        mod = p["modern"]
        P[pid] = {"n": p["name"], "c": p["kind"], "p": p["parent"], "pk": p["parent_id"],
                  "al": [a for a in p["aliases"].split(" | ") if a][:10],
                  "mod": "; ".join(re.sub(r"([a-z])([A-Z])", r"\1 \2", m.split("/")[-1]) + ", " +
                                   re.sub(r"([a-z])([A-Z])", r"\1 \2", m.split("/")[1]) for m in mod.split(";") if m) if mod else "",
                  "tier": {"high": "A", "medium": "B", "low": "C"}.get(p["confidence"], ""),
                  "ev": p["note"][:140], "kids": sorted(kids.get(pid, []), key=lambda k: places[k]["name"])[:200],
                  "r": "", "cy": {y: v for y, v in cy.items()},
                  "t": [[tix[tid], how] for tid, how in sorted(own, key=lambda x: (tabs[x[0]]["cy"], tabs[x[0]]["doc"], tabs[x[0]]["seq"]))]}
        regs = collections.Counter(tabs[tid]["broad_region"] for tid, _ in own if tabs[tid]["broad_region"])
        P[pid]["r"] = regs.most_common(1)[0][0] if regs else ""
    def strength(check):
        return 3 if "printed total" in check else 2 if "sums to base" in check else 1

    for (pid, y), rows in metrics.items():
        if pid in P and y:
            b = max(rows, key=lambda r: (strength(r["check"]), float(r["total"] or 0)))
            f = lambda k: round(float(b[k]), 1) if b[k] != "" else None
            P[pid].setdefault("m", {})[y] = [f("child"), f("old"), f("sexratio"), tabs[b["table_id"]]["page"], b["check"][:80]]
    # same territory under another name/era (e.g. Amraoti 1881-1911 -> Amravati 1951)
    by_modern = collections.defaultdict(list)
    for pid in P:
        if places[pid]["kind"] in LOCAL_KINDS and places[pid]["modern"]:
            by_modern[places[pid]["modern"].split(";")[0]].append(pid)
    for pids in by_modern.values():
        if 1 < len(pids) <= 6:
            for pid in pids:
                P[pid]["sib"] = [q for q in pids if q != pid]
    tag_tables = collections.defaultdict(list)
    for pid, L in links.items():
        for tid, how in L:
            if how == "tag":
                tag_tables[pid].append(tid)
    json.dump({"places": P, "tables": T, "years": YEARS, "classes": CLASSES,
               "vols": {d: {"s": (vols_meta.get(os.path.basename(docs[d]["source_pdf"]), {}).get("province_or_volume") or docs[d]["region"])[:60]}
                        for d in docs}},
              open(os.path.join(SITE, "data", "places.json"), "w"), separators=(",", ":"))

    # ---- volumes.json ----
    gaps = collections.defaultdict(list)
    for r in csv.DictReader(open(os.path.join(SV1, "stitched_tables.csv"))):
        miss = sorted(int(x) for x in r["missing_pages"].split(",") if x.strip().isdigit())
        run = []
        for x in miss + [None]:
            if run and (x is None or x != run[-1] + 1):
                gaps[r["dochash"]].append((run[0], run[-1], r["table_number"]))
                run = []
            if x is not None:
                run.append(x)
    V = {}
    by_doc = collections.defaultdict(list)
    for tid in tid_list:
        by_doc[tabs[tid]["doc"]].append(tid)
    for d, tids in by_doc.items():
        dm = docs.get(d, {})
        vm = vols_meta.get(os.path.basename(dm.get("source_pdf", "")), {})
        items = [{"i": tix[t], "o": tabs[t]["seq"]} for t in tids]
        # printed-page -> PDF-page offset for this volume (median over tables with both)
        offs = sorted(tabs[t]["seq"] - int(tabs[t]["pg"]) for t in tids
                      if str(tabs[t]["pg"]).isdigit() and tabs[t]["seq"] < 99999)
        off = offs[len(offs) // 2] if offs else None
        for a, b, tn in sorted(gaps.get(d, [])):
            g = {"gap": f"{a}" if a == b else f"{a}–{b}", "n": b - a + 1, "tn": tn}
            if off is not None:
                g["o"] = a + off
                pos_ = next((k for k, it in enumerate(items) if it.get("o", 0) > a + off), len(items))
            else:
                pos_ = next((k for k, it in enumerate(items) if "i" in it and str(T[it["i"]]["pg"]).isdigit()
                             and int(T[it["i"]]["pg"]) > b), len(items))
            items.insert(pos_, g)
        V[d] = {"s": (vm.get("province_or_volume") or dm.get("region") or d)[:80], "title": vm.get("title", ""),
                "y": census_year(dm.get("year")) or dm.get("year", ""), "r": dm.get("region", ""),
                "jstor": vm.get("jstor_url", ""), "npages": vm.get("npages_total", ""),
                "n": len(tids), "na": sum(T[tix[t]]["a"] for t in tids), "np": sum(1 for t in tids if tabs[t]["page"]),
                "gap": sum(g[1] - g[0] + 1 for g in gaps.get(d, [])), "items": items}
    for i, tid in enumerate(tid_list):
        T[i]["pl"] = tplaces.get(tid, [])[:6]
    json.dump({"vols": V, "tables": T, "ntables": len(T),
               "pn": {pid: places[pid]["name"] for pid in {p for t in T for p in t["pl"]}}},
              open(os.path.join(SITE, "data", "volumes.json"), "w"), separators=(",", ":"))

    # ---- atlas.json ----
    geo = json.load(open(os.path.join(HERE, "out", "districts_svg.json")))
    code2gid = {f'{u["country"]}/{u["admin1"]}/{u["name"]}': u["gid"] for u in geo["units"]}
    cent = {u["gid"]: (u["cx"], u["cy"], u["country"], u["admin1"]) for u in geo["units"]}
    gid_places = collections.defaultdict(list)     # gid -> [pid] (local kinds first)
    for pid, p in places.items():
        if p["kind"] in LOCAL_KINDS + SUB_KINDS and p["modern"] and pid in P:
            for m in p["modern"].split(";"):
                g = code2gid.get(m.strip())
                if g:
                    gid_places[g].append(pid)
    # province for each gid: most common parent of its local places; else nearest located gid
    gid_prov, how_prov = {}, {}
    for g, pids in gid_places.items():
        c = collections.Counter()
        for pid in pids:
            q, seen = places[pid]["parent_id"], set()
            while q and q in places and places[q]["kind"] not in ("province", "princely_state", "agency") and q not in seen:
                seen.add(q); q = places[q]["parent_id"]
            if q:
                c[q] += 1
        if c:
            gid_prov[g] = c.most_common(1)[0][0]; how_prov[g] = "parent"
    located = list(gid_prov)
    for g, (x, y, ctry, a1) in cent.items():
        if g in gid_prov or not located:
            continue
        same = [h for h in located if cent[h][2] == ctry and cent[h][3] == a1] or [h for h in located if cent[h][2] == ctry] or located
        h = min(same, key=lambda k: (cent[k][0] - x) ** 2 + (cent[k][1] - y) ** 2)
        gid_prov[g] = gid_prov[h]; how_prov[g] = "nearest"

    def strength(check):
        return 3 if "printed total" in check else 2 if "sums to base" in check else 1

    def pick(rows):
        """best indicator row: strongest self-check, then Table VII-like, then largest population"""
        return max(rows, key=lambda r: (strength(r["check"]),
                                        bool(re.search(r"(?i)table\s+vii|civil condition", tabs[r["table_id"]]["title"] or "")),
                                        float(r["total"] or 0)))
    units, summary = {}, {}
    for y in YEARS + ["all"]:
        s = {"hist_ver": set(), "hist_age": set(), "hist_any": set(), "modern_ver": 0, "modern_age": 0, "modern_prov": 0}
        for g in cent:
            pids = gid_places.get(g, [])
            own = [(pid, tid) for pid in pids for tid, how in links.get(pid, [])
                   if how != "birthplace" and (y == "all" or tabs[tid]["cy"] == y)]
            loc = [(pid, tid) for pid, tid in own if places[pid]["kind"] in LOCAL_KINDS]
            na = sum(1 for pid, tid in loc if is_age(tid, pid))
            rec = {"nt": len(own), "na": na}
            for pid, tid in loc:
                s["hist_any"].add(pid)
                if is_age(tid, pid):
                    s["hist_age"].add(pid)
            # indicators: local unit first, then the province
            mrows = [m for pid in pids if places[pid]["kind"] in LOCAL_KINDS
                     for yy in ([y] if y != "all" else YEARS) for m in metrics.get((pid, yy), [])]
            for pid in pids:
                if places[pid]["kind"] in LOCAL_KINDS and any(metrics.get((pid, yy)) for yy in ([y] if y != "all" else YEARS)):
                    s["hist_ver"].add(pid)
            if mrows:
                rec["c"] = "district_verified"
            elif na:
                rec["c"] = "district_age"
            elif own:
                rec["c"] = "district_other"
            lvl = "district"
            if not mrows and g in gid_prov:
                pv = gid_prov[g]
                mrows = [m for yy in ([y] if y != "all" else YEARS) for m in metrics.get((pv, yy), [])]
                lvl = "province"
                if mrows and "c" not in rec:
                    rec["c"] = "province"
            if "c" not in rec and g in gid_prov and any((tabs[t]["cy"] == y or y == "all") and is_age(t)
                                                       for t in tag_tables.get(gid_prov[g], [])):
                rec["c"] = "province"
            if mrows:
                b = pick(mrows)
                rec["m"] = {k: round(float(b[k]), 1) for k in ("child", "old", "sexratio") if b[k] != ""}
                rec["lvl"] = lvl
                rec["src"] = f'{tabs[b["table_id"]]["cy"]} · {(tabs[b["table_id"]]["title"] or "")[:60]}'
                rec["chk"] = b["check"][:70]
                rec["tp"] = tabs[b["table_id"]]["page"]
            if "c" not in rec:
                continue
            if rec["c"] == "district_verified":
                s["modern_ver"] += 1
            elif rec["c"] == "district_age":
                s["modern_age"] += 1
            elif rec["c"] == "province":
                s["modern_prov"] += 1
            units.setdefault(g, {})[y] = rec
        summary[y] = {"hist_ver": len(s["hist_ver"]), "hist_age": len(s["hist_age"]), "hist_any": len(s["hist_any"]),
                      "modern_ver": s["modern_ver"], "modern_age": s["modern_age"], "modern_prov": s["modern_prov"]}
    for g in units:
        if g in gid_prov:
            units[g]["prov"] = gid_prov[g]; units[g]["prov_how"] = how_prov[g]
    hist = {g: sorted({pid for pid in pids if places[pid]["kind"] in LOCAL_KINDS} or set(pids),
                      key=lambda k: -len(links.get(k, [])))[:4] for g, pids in gid_places.items()}
    used = {p for v in hist.values() for p in v} | set(gid_prov.values())
    json.dump({"units": units, "hist": hist, "summary": summary, "years": YEARS, "default_year": "1901",
               "countries": COUNTRIES,
               "places": {pid: {"n": places[pid]["name"], "p": places[pid]["parent"],
                                "tier": {"high": "A", "medium": "B", "low": "C"}.get(places[pid]["confidence"], "")}
                          for pid in used if pid in places}},
              open(os.path.join(SITE, "data", "atlas.json"), "w"), separators=(",", ":"))

    # ---- CSV downloads ----
    D = os.path.join(SITE, "data")
    with open(os.path.join(D, "place_inventory.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["place_id", "place", "kind", "parent", "modern_districts", "match_confidence"] +
                   [f"tables_{y}" for y in YEARS] + [f"age_tables_{y}" for y in YEARS])
        for pid, r in sorted(P.items(), key=lambda kv: kv[1]["n"]):
            w.writerow([pid, r["n"], r["c"], r["p"], places[pid]["modern"], places[pid]["confidence"]] +
                       [r["cy"].get(y, [0, 0])[0] for y in YEARS] + [r["cy"].get(y, [0, 0])[1] for y in YEARS])
    with open(os.path.join(D, "district_concordance.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["place_id", "historical_name", "kind", "historical_parent", "modern_districts_gadm41",
                    "confidence", "evidence_note", "spellings_seen"])
        for pid, p in sorted(places.items(), key=lambda kv: kv[1]["name"]):
            if p["kind"] in LOCAL_KINDS + SUB_KINDS + ("division",) and pid in P:
                w.writerow([pid, p["name"], p["kind"], p["parent"], p["modern"], p["confidence"], p["note"], p["aliases"]])
    with open(os.path.join(D, "district_age_indicators.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["place_id", "place", "kind", "census_year", "pct_aged_0_9", "pct_aged_60_plus",
                    "females_per_1000_males", "population_in_table", "self_check", "slice", "table_title", "table_page"])
        for (pid, y), rows in sorted(metrics.items()):
            if pid not in places:
                continue
            b = pick(rows)
            t = tabs[b["table_id"]]
            w.writerow([pid, places[pid]["name"], places[pid]["kind"], y, b["child"], b["old"], b["sexratio"],
                        b["total"], b["check"], b["slice"], t["title"], t["page"]])
    print("places.json:", len(P), "places |", "volumes.json:", len(V), "volumes |", "tables:", len(T),
          "| published:", sum(1 for t in T if t["p"]))
    for y in YEARS:
        print(" ", y, summary[y])


if __name__ == "__main__":
    main()
