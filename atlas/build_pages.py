#!/usr/bin/env python3
"""Stage 3: write the new views into census_site/ and wire them into every page.

New pages: atlas.html (district map), places.html (every place x year), place.html?k=
(complete per-place inventory), volume.html[?d=] (census order with gaps), search.html
(rebuilt: place-aware, says when a place's inventory is complete).
Patches: shared navigation on every top-level page; on every table page a context strip
(volume, printed page, previous/next table in the volume, places covered).
"""
import json, os, re, html, glob, collections, sqlite3, hashlib, datetime
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
SITE = os.path.join(ROOT, "census_site")
T = os.path.join(HERE, "templates")
STAMP = datetime.datetime.now().strftime("%Y-%m-%d")

NAV_ITEMS = [("atlas.html", "🗺 District atlas"), ("volume.html", "📚 Census order"), ("places.html", "📍 Places A–Z"),
             ("search.html", "🔎 Search"), ("tree.html", "🗂 Columns"), ("index.html", "Regions"), ("best.html", "★ Best Table VII")]
NAVCSS = (".nav{background:#2c3e50;color:#fff;padding:8px 16px;display:flex;flex-wrap:wrap;gap:4px 14px;align-items:center}"
          ".nav a{color:#cfe3f7;text-decoration:none;font-size:13.5px}.nav a.on{color:#fff;font-weight:600;border-bottom:2px solid #9cf}"
          ".nav .brand{color:#fff;font-weight:700;margin-right:6px}.nav .stamp{margin-left:auto;font-size:11px;opacity:.6}")


def nav(cur, prefix=""):
    links = "".join(f'<a href="{prefix}{h}" class="{"on" if h == cur else ""}">{l}</a>' for h, l in NAV_ITEMS)
    return (f'<nav class=nav><a class=brand href="{prefix}atlas.html">Indian Census Age Tables</a>{links}'
            f'<span class=stamp>updated {STAMP}</span></nav>')


def page(tmpl, cur, **subs):
    s = open(os.path.join(T, tmpl)).read().replace("__NAVCSS__", NAVCSS).replace("__NAV__", nav(cur))
    for k, v in subs.items():
        s = s.replace(f"__{k}__", v)
    return s


def write(name, s):
    open(os.path.join(SITE, name), "w").write(s)


def main():
    geo = json.load(open(os.path.join(HERE, "out", "districts_svg.json")))
    atlas = open(os.path.join(SITE, "data", "atlas.json")).read()
    write("atlas.html", page("atlas.html", "atlas.html", VB=geo["viewBox"],
                             GEO=json.dumps(geo, separators=(",", ":")), DATA=atlas))
    write("place.html", page("place.html", "places.html"))
    write("places.html", page("places.html", "places.html"))
    write("volume.html", page("volume.html", "volume.html"))
    build_search()
    patch_top_pages()
    patch_table_pages()


def build_search():
    """Search over every published table; each table also matches the names (and
    spellings) of all places it covers, so a district found only as a row inside a
    provincial table is still found."""
    P = json.load(open(os.path.join(SITE, "data", "places.json")))
    V = json.load(open(os.path.join(SITE, "data", "volumes.json")))
    db = sqlite3.connect(os.path.join(ROOT, "export_final/sendout_final/sendout_v1/census_explorer.sqlite"))
    meta = {}
    for tid, k, m, e, s, r, y, g, t in db.execute(
            "select table_id,subject,measure_class,entity_type,status,broad_region,year,canonical_name,title from tables"):
        meta[tid] = (k, m, e, s, r)
    # page -> table id
    pg2tid = {}
    for (tid,) in db.execute("select table_id from tables"):
        pg2tid[hashlib.md5(tid.encode()).hexdigest()[:7]] = tid
    names = {pid: [p["n"]] + p.get("al", []) for pid, p in P["places"].items()}
    idx = []
    for t in V["tables"]:
        if not t["p"]:
            continue
        tid = pg2tid.get(t["p"].rsplit("_", 1)[1])
        k, m, e, s, r = meta.get(tid, ("", "", "", "", ""))
        x = sorted({n for pid in t.get("pl", []) for n in names.get(pid, [])[:3]})
        idx.append({"p": t["p"], "t": t["t"][:110], "g": t["g"], "y": t["y"], "r": r or "", "k": k or "",
                    "m": m or "", "e": e or "", "s": s or "", "a": t["a"], "x": " · ".join(x)[:200],
                    "v": (V["vols"].get(t["d"], {}).get("s") or "")[:40], "d": t["d"], "pg": t["pg"]})
    pidx = {r["p"]: i for i, r in enumerate(idx)}
    places_small = [[pid, p["n"], p["c"], p.get("p", ""), " ".join(p.get("al", []) + [p.get("mod", "")]),
                     sum(v[0] for v in p.get("cy", {}).values()), sum(v[1] for v in p.get("cy", {}).values()),
                     ",".join(sorted(p.get("cy", {}))),
                     sorted({pidx[P["tables"][i]["p"]] for i, how in p["t"] if how != "birthplace" and P["tables"][i]["p"] in pidx})]
                    for pid, p in P["places"].items() if p.get("cy")]
    s = open(os.path.join(T, "search.html")).read().replace("__NAVCSS__", NAVCSS).replace("__NAV__", nav("search.html"))
    s = s.replace("__IDX__", json.dumps(idx, separators=(",", ":"))).replace("__PLACES__", json.dumps(places_small, separators=(",", ":")))
    write("search.html", s)
    print(f"search.html: {len(idx)} tables, {len(places_small)} places")


HEADER = re.compile(r"<header>(.*?)</header>", re.S)


def patch_top_pages():
    """Prepend the shared nav to legacy top-level pages (keep their own header line)."""
    mine = {"atlas.html", "place.html", "places.html", "volume.html", "search.html"}
    for f in glob.glob(os.path.join(SITE, "*.html")):
        name = os.path.basename(f)
        if name in mine:
            continue
        s = open(f).read()
        if "class=nav" in s:
            s = re.sub(r"<style id=navcss>.*?</style><nav class=nav>.*?</nav>", "", s, flags=re.S)
        block = f"<style id=navcss>{NAVCSS}</style>" + nav(name)
        if re.search(r"<body[^>]*>", s):
            s = re.sub(r"(<body[^>]*>)", lambda m: m.group(1) + block, s, count=1)
        else:                                          # body-less legacy pages: nav goes after the head's </style>
            s = re.sub(r"(</style>)", lambda m: m.group(1) + block, s, count=1)
        if name == "index.html":
            s = add_home_cards(s)
        open(f, "w").write(s)


def add_home_cards(s):
    s = re.sub(r"<div id=homecards[^>]*>.*?</div><!--/homecards-->", "", s, flags=re.S)
    card = ("display:block;flex:1 1 220px;background:#fff;border:1px solid #cdd9e3;border-radius:10px;padding:12px 14px;"
            "text-decoration:none;color:#1a1a1a")
    cards = [("atlas.html", "🗺 District atlas", "Every modern district coloured by what age data exists, census by census — and the age structure where we have it."),
             ("volume.html", "📚 Census order", "Every source volume, its tables in printed page order, with missing pages shown where they fall."),
             ("places.html", "📍 Places A–Z", "Every district, state and town × census year: how many tables, how many with ages."),
             ("search.html", "🔎 Search", "Finds a district even when it is only a row inside a provincial table, and tells you when that is everything we have.")]
    html_ = ("<div id=homecards style='display:flex;flex-wrap:wrap;gap:10px;margin:12px 16px'>" +
             "".join(f"<a href='{h}' style='{card}'><b>{t}</b><br><span style='font-size:12.5px;color:#456'>{d}</span></a>" for h, t, d in cards) +
             "</div><!--/homecards-->")
    return s.replace("</nav>", "</nav>" + html_, 1)


CTX_CSS = ("<style id=ctxcss>header .pill{background:#dbe8f3!important;color:#1f3a52!important}.ctx{background:#eef4f9;border-bottom:1px solid #d5e1ea;padding:6px 16px;font-size:12.5px;"
           "display:flex;flex-wrap:wrap;gap:4px 16px}.ctx a{color:#1a5fa0}</style>")


def patch_table_pages():
    V = json.load(open(os.path.join(SITE, "data", "volumes.json")))
    T, pn = V["tables"], V["pn"]
    order = collections.defaultdict(list)
    for d, v in V["vols"].items():
        for it in v["items"]:
            if "i" in it and T[it["i"]]["p"]:
                order[d].append(it["i"])
    pos = {}
    for d, lst in order.items():
        for k, i in enumerate(lst):
            pos[T[i]["p"]] = (d, k)
    n = 0
    for f in glob.glob(os.path.join(SITE, "tables", "*.html")):
        pgid = os.path.basename(f)[:-5]
        s = open(f).read()
        s = re.sub(r"<style id=ctxcss>.*?</style>", "", s, flags=re.S)
        s = re.sub(r"<div class=ctx>.*?</div><!--/ctx-->", "", s, flags=re.S)
        s = re.sub(r"<style id=navcss>.*?</style><nav class=nav>.*?</nav>", "", s, flags=re.S)
        bits = []
        if pgid in pos:
            d, k = pos[pgid]
            v = V["vols"][d]
            lst = order[d]
            t = T[lst[k]]
            bits.append(f'📚 <a href="../volume.html?d={d}">{html.escape(v["s"])}</a> ({html.escape(str(v["y"]))})'
                        + (f' · printed page {html.escape(str(t["pg"]))}' if t["pg"] else "")
                        + f' · table {k + 1} of {len(lst)} in this volume')
            nb = []
            if k > 0:
                nb.append(f'<a href="{T[lst[k-1]]["p"]}.html">← previous table</a>')
            if k + 1 < len(lst):
                nb.append(f'<a href="{T[lst[k+1]]["p"]}.html">next table →</a>')
            if nb:
                bits.append(" · ".join(nb))
            if t.get("pl"):
                bits.append("📍 covers: " + ", ".join(f'<a href="../place.html?k={p}">{html.escape(pn.get(p, p))}</a>' for p in t["pl"]))
        ctx = f"<div class=ctx>{'<span>' + '</span><span>'.join(bits) + '</span>' if bits else ''}</div><!--/ctx-->"
        s = re.sub(r"(<body[^>]*>)", lambda m: m.group(1) + f"<style id=navcss>{NAVCSS}</style>" + nav("", "../") + CTX_CSS, s, count=1)
        s = s.replace("</header>", "</header>" + ctx, 1)
        open(f, "w").write(s)
        n += 1
    print(f"patched {n} table pages")


if __name__ == "__main__":
    main()
