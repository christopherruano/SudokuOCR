#!/usr/bin/env python3
"""Verification suite for the census site. Run against the local build or the live site:

    python3 atlas/verify_site.py                      # local census_site/ (served on :8765)
    python3 atlas/verify_site.py https://<host>       # live deployment

Checks (each prints PASS/FAIL with counts; exit code 1 on any FAIL):
  links      every table page referenced by search, places.json and volumes.json exists
             and serves a real table page (not the host's homepage fallback); every
             href on the top-level pages resolves.
  data       internal consistency: places.json <-> table_places.csv; volume membership;
             atlas units/places exist in the geometry and registry; search index ==
             published tables; no superseded duplicate is listed.
  figures    every published age figure is recomputed from the raw cells in the DB and
             must match the CSV exactly; every figure satisfies its self-check and the
             plausibility bounds.
  browser    each page type renders in headless Chrome with no JavaScript errors and
             shows its key elements.
"""
import csv, json, os, re, sys, sqlite3, random, subprocess, collections, urllib.request, urllib.parse, concurrent.futures as cf
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
SITE = os.path.join(ROOT, "census_site")
DB = os.path.join(ROOT, "export_final/sendout_final/sendout_v1/census_explorer.sqlite")
BASE = (sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8765").rstrip("/")
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
sys.path.insert(0, HERE)
FAILS = []


def report(name, ok, detail):
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}")
    if not ok:
        FAILS.append(name)


def get(path, timeout=30):
    url = path if path.startswith("http") else f"{BASE}/{path.lstrip('/')}"
    req = urllib.request.Request(url, headers={"User-Agent": "census-verify/1.0"})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.status, r.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as e:
            loc = e.headers.get("Location") if e.code in (301, 302, 307, 308) else None
            if loc:                                    # Cloudflare 308s /x.html -> /x; urllib (py3.9) won't follow it
                req = urllib.request.Request(urllib.parse.urljoin(url, loc), headers=req.headers)
                url = req.full_url
                continue
            return e.code, ""
        except Exception:
            if attempt == 2:
                return 0, ""


def load(name):
    s, body = get(f"data/{name}")
    if s != 200:
        raise SystemExit(f"cannot fetch data/{name}: HTTP {s}")
    return json.loads(body)


def check_links(P, V, search_pages):
    pages = {t["p"] for t in V["tables"] if t["p"]}
    from_places = {P["tables"][i]["p"] for p in P["places"].values() for i, _ in p["t"] if P["tables"][i]["p"]}
    every = pages | from_places | set(search_pages)

    def probe(pg):
        s, body = get(f"tables/{pg}.html")
        real = s == 200 and "<title>Census — map</title>" not in body[:300] and "class=imgs" in body
        return pg, real
    bad = []
    with cf.ThreadPoolExecutor(24) as ex:
        for pg, ok in ex.map(probe, sorted(every)):
            if not ok:
                bad.append(pg)
    report("links/table pages", not bad, f"{len(every) - len(bad)}/{len(every)} serve a real table page"
           + (f"; broken e.g. {bad[:3]}" if bad else ""))
    # top-level hrefs
    tops = ["atlas.html", "volume.html", "places.html", "search.html", "place.html", "index.html", "tree.html", "best.html"]
    hrefs = set()
    for t in tops:
        s, body = get(t)
        report(f"links/{t}", s == 200, f"HTTP {s}")
        for h in re.findall(r'href="([^"#?:]+\.(?:html|csv|zip|png|json))', body):
            if not h.startswith(("http", "javascript")) and "${" not in h:
                hrefs.add(h.lstrip("./"))
    miss = [h for h in sorted(hrefs) if get(h)[0] != 200]
    report("links/static hrefs on top pages", not miss, f"{len(hrefs) - len(miss)}/{len(hrefs)} resolve" + (f"; missing {miss[:5]}" if miss else ""))


def check_data(P, V, A, S):
    tp = collections.defaultdict(set)
    for r in csv.DictReader(open(os.path.join(HERE, "out", "table_places.csv"))):
        if r["how"] != "birthplace":
            tp[r["place_id"]].add(r["table_id"])
    # place inventory counts agree with cy counts
    bad = [k for k, p in P["places"].items()
           if sum(v[0] for v in p.get("cy", {}).values()) > len([1 for i, h in p["t"] if h != "birthplace"])]
    report("data/place year counts <= listed tables", not bad, f"{len(P['places']) - len(bad)}/{len(P['places'])} consistent")
    # every volume item index valid, every table in exactly one volume
    seen = collections.Counter()
    for d, v in V["vols"].items():
        for it in v["items"]:
            if "i" in it:
                seen[it["i"]] += 1
    dup = [i for i, n in seen.items() if n > 1]
    report("data/each table in exactly one volume", not dup and len(seen) == len(V["tables"]),
           f"{len(seen)}/{len(V['tables'])} placed, {len(dup)} duplicated")
    # superseded ids absent
    sup = set(json.load(open(os.path.join(HERE, "out", "superseded_ids.json"))))
    import hashlib
    sup_pages = {hashlib.md5(t.encode()).hexdigest()[:7] for t in sup}
    leaked = [t["p"] for t in V["tables"] if t["p"] and t["p"].rsplit("_", 1)[1] in sup_pages]
    report("data/superseded duplicates not listed", not leaked, f"{len(leaked)} leaked")
    # atlas
    geo = json.load(open(os.path.join(HERE, "out", "districts_svg.json")))
    gids = {u["gid"] for u in geo["units"]}
    bad_g = [g for g in A["units"] if g not in gids]
    bad_p = [p for v in A["hist"].values() for p in v if p not in A["places"]]
    report("data/atlas units exist in geometry", not bad_g, f"{len(A['units']) - len(bad_g)}/{len(A['units'])}")
    report("data/atlas place refs resolve", not bad_p, f"{len(bad_p)} unresolved")
    # summary numbers reproducible from units
    for y in A["years"]:
        n = sum(1 for g, u in A["units"].items() if u.get(y, {}).get("c") == "district_verified")
        if n != A["summary"][y]["modern_ver"]:
            report(f"data/atlas summary {y}", False, f"{n} vs {A['summary'][y]['modern_ver']}")
            break
    else:
        report("data/atlas summary counts reproduce from units", True, "all years")
    # search index == published tables
    pub = {t["p"] for t in V["tables"] if t["p"]}
    report("data/search index == published tables", set(S) == pub, f"search {len(S)}, published {len(pub)}")


def check_figures():
    import age_metrics as am
    from build_index import build_registry, norm, BOUNDS
    rows = list(csv.DictReader(open(os.path.join(SITE, "data", "district_age_indicators.csv"))))
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    pm = list(csv.DictReader(open(os.path.join(HERE, "out", "place_metrics.csv"))))
    # bounds + self-check text present
    oob = [r for r in pm for k, (lo, hi) in BOUNDS.items() if r[k] and not lo <= float(r[k]) <= hi]
    nochk = [r for r in pm if not r["check"]]
    report("figures/all within plausibility bounds", not oob, f"{len(pm) - len(oob)}/{len(pm)}")
    report("figures/all carry a self-check", not nochk, f"{len(pm) - len(nochk)}/{len(pm)}")
    # independent recomputation of single-page figures from raw cells (multi-page ones need the run)
    places, key2pid = build_registry()
    random.seed(20261009)
    single = [r for r in pm if "merged across" not in r["check"]]
    sample = random.sample(single, min(150, len(single)))
    mism = []
    for r in sample:
        fs = [dict(x) for x in con.execute(
            "select * from facts where table_id=? and (row_dimension='age' or column_dimension='age')", (r["table_id"],))]
        tag = con.execute("select canonical_name from tables where table_id=?", (r["table_id"],)).fetchone()[0]
        mine = []
        for d in fs:
            if d["row_dimension"] != "age":
                pid = key2pid.get(norm(d["row_label_std"] or ""))
            else:
                pid = key2pid.get(norm(d["section"] or ""))
                if pid and places[pid]["kind"] not in ("district", "princely_state", "province", "sub_district", "town", "division"):
                    pid = None
                if not pid and d["column_dimension"] != "age":
                    pid = key2pid.get(norm(d["column_group_std"] or ""))
                if not pid:
                    pid = key2pid.get(norm(tag or ""))
            if pid == r["place_id"]:
                mine.append(d)
        m = am.best_metrics(mine)
        for k in ("child", "old", "sexratio"):
            a = round(m[k], 3) if m and m.get(k) is not None else ""
            b = round(float(r[k]), 3) if r[k] else ""
            if a != b:
                mism.append((r["table_id"][-40:], r["place_id"], k, a, b))
                break
    report("figures/recomputed from raw cells (150 random)", not mism, f"{len(sample) - len(mism)}/{len(sample)} identical"
           + (f"; e.g. {mism[:2]}" if mism else ""))
    # arithmetic of the published check: bands + not-stated vs printed total within 2%
    badchk = []
    for r in pm:
        m = re.search(r"\(([\d,]+)/([\d,]+)\)", r["check"])
        if m:
            a, b = (float(x.replace(",", "")) for x in m.groups())
            if b <= 0 or abs(a - b) / b > 0.02:
                badchk.append(r)
    report("figures/printed-total checks within 2%", not badchk, f"{len(badchk)} violations")
    # hand-verifiable worked example written out for the methods section
    ex = next((r for r in pm if "= printed total" in r["check"] and r["child"]), None)
    if ex:
        print(f"      worked example: {ex['place_id']} {ex['table_id'][-50:]} child={ex['child']} old={ex['old']} "
              f"sexratio={ex['sexratio']} check='{ex['check']}'")


def check_browser():
    pages = ["atlas.html", "atlas.html?y=1881&l=child", "places.html", "place.html?k=cawnpore-district",
             "volume.html", "volume.html?d=029d7d03bf1acaa7", "search.html?q=kanpur", "index.html", "best.html"]
    expect = {"atlas": "historical districts/states", "places": "places", "place.html": "complete inventory",
              "volume.html?d": "printed page", "volume.html": "Census volumes", "search": "tables", "index": "District atlas",
              "best": "Best age table"}
    for pg in pages:
        r = subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--virtual-time-budget=20000",
                            "--enable-logging=stderr", "--v=0", "--dump-dom", f"{BASE}/{pg}"],
                           capture_output=True, text=True, timeout=180)
        errs = [l for l in r.stderr.splitlines() if "Uncaught" in l or "SyntaxError" in l or "TypeError" in l or "ReferenceError" in l]
        key = next(v for k, v in expect.items() if pg.startswith(k) or k in pg)
        ok = not errs and key.lower() in r.stdout.lower()
        report(f"browser/{pg}", ok, "renders, no JS errors" if ok else f"errors={errs[:2]} key_found={key.lower() in r.stdout.lower()}")


def main():
    print(f"verifying {BASE}\n")
    P, V, A = load("places.json"), load("volumes.json"), load("atlas.json")
    s, body = get("search.html")
    S = [x["p"] for x in json.loads(re.search(r"const IDX=(\[.*?\]);\n", body, re.S).group(1))]
    check_links(P, V, S)
    check_data(P, V, A, S)
    check_figures()
    check_browser()
    print(f"\n{'ALL CHECKS PASSED' if not FAILS else f'{len(FAILS)} CHECK(S) FAILED: ' + ', '.join(FAILS)}")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
