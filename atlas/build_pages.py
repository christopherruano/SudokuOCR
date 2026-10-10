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
    hgeo = open(os.path.join(SITE, "data", "hist_geo.json")).read()
    write("atlas.html", page("atlas.html", "atlas.html", VB=geo["viewBox"],
                             GEO=json.dumps({k: v for k, v in geo.items() if k != "proj"}, separators=(",", ":")),
                             DATA=atlas, HGEO=hgeo))
    write("place.html", page("place.html", "places.html"))
    write("places.html", page("places.html", "places.html"))
    write("volume.html", page("volume.html", "volume.html"))
    build_search()
    retire_legacy()
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


RETIRED = {  # earlier prototype views superseded by the new pages; old links forward
    "ages.html": ("a", "search.html?q=", "the place pages (age structure per census) and the district atlas"),
    "area.html": ("q", "search.html?q=", "place-aware search and the complete per-place inventories"),
    "highlights.html": (None, "volume.html", "the census-order view of every volume"),
    "coverage.html": (None, "places.html", "Places A–Z (every place × census year) and the district atlas"),
}


def retire_legacy():
    for name, (param, target, what) in RETIRED.items():
        js = (f"const v=new URLSearchParams(location.search).get('{param}');location.replace('{target}'+(v?encodeURIComponent(v):''))"
              if param else f"location.replace('{target}')")
        write(name, f"""<!doctype html><html><head><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>Census — moved</title><style>body{{font:15px system-ui;margin:0;background:#fbfdff}}{NAVCSS}main{{max-width:640px;margin:40px auto;padding:0 16px}}</style></head>
<body>{nav("")}<main><h2>This view has moved</h2><p>It has been replaced by {what}. Taking you there…
If nothing happens, <a href="{target}">continue</a>.</p></main><script>{js}</script></body></html>""")


HEADER = re.compile(r"<header>(.*?)</header>", re.S)


def patch_top_pages():
    """Prepend the shared nav to legacy top-level pages (keep their own header line)."""
    mine = {"atlas.html", "place.html", "places.html", "volume.html", "search.html"} | set(RETIRED)
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


TABLE_ENH = """<meta name=viewport content="width=device-width,initial-scale=1"><style id=tblenh>
 .wrap{height:calc(100vh - var(--top,120px))!important}
 .imgs{position:relative;padding-top:38px!important}
 .imgs img{cursor:zoom-in;transition:width .15s}
 .scanbar{position:absolute;top:6px;left:0;right:0;display:flex;gap:6px;justify-content:center;z-index:2}
 .scanbar button{background:#3a4b5c;color:#fff;border:0;border-radius:5px;padding:3px 9px;font:12px system-ui;cursor:pointer}
 .scanbar button.on{background:#9cf;color:#102030}
 table.mx thead th{position:sticky;top:-12px;z-index:3}
 table.mx thead tr:nth-child(2) th{top:12px}
 table.mx thead th.rh,table.mx thead th:first-child{left:0;z-index:4;background:#eef3f7}
 table.mx td.rh{position:sticky;left:0;z-index:2}
 #lb{position:fixed;inset:0;background:rgba(10,14,18,.94);z-index:99;display:none;overflow:hidden;touch-action:none}
 #lb img{position:absolute;left:0;top:0;transform-origin:0 0;max-width:none;background:#fff;cursor:grab;user-select:none}
 #lb .lbbar{position:absolute;top:10px;right:12px;display:flex;gap:6px;z-index:2}
 #lb .lbbar button{background:#fff;border:0;border-radius:6px;padding:5px 11px;font:13px system-ui;cursor:pointer}
 #lb .lbhint{position:absolute;bottom:10px;left:0;right:0;text-align:center;color:#aab;font:12px system-ui}
 .yrnote{background:#fff3cd;color:#664d03;border-radius:5px;padding:1px 7px;font-size:11.5px;margin-left:6px}
 @media (max-width:820px){.wrap{flex-direction:column;height:auto!important}.imgs{width:100%!important;max-height:none}.data{overflow-x:auto!important;overflow-y:visible!important;max-width:100vw}header{flex-wrap:wrap}}
</style>
<div id=lb><div class=lbbar><button data-z="rot">⟳ rotate</button><button data-z="-">−</button><button data-z="fit">fit</button><button data-z="+">+</button><button data-z="x">✕ close</button></div><img alt=""><div class=lbhint>scroll or pinch to zoom · drag to move · R to rotate · ← → for other pages · Esc to close</div></div>
<script>
(function(){const top=()=>{const w=document.querySelector('.wrap');if(w)document.documentElement.style.setProperty('--top',(w.getBoundingClientRect().top+window.scrollY)+'px')};
addEventListener('load',top);addEventListener('resize',top);
document.addEventListener('DOMContentLoaded',()=>{top();
 // second header row sticks exactly under the first (heights vary with wrapped group labels)
 document.querySelectorAll('table.mx thead').forEach(th=>{const r=th.rows;if(r.length>1){const h=r[0].getBoundingClientRect().height;[...r[1].cells].forEach(c=>c.style.top=(h-12)+'px')}});
 const pane=document.querySelector('.imgs');if(!pane)return;
 const imgs=[...pane.querySelectorAll('img')];const bar=document.createElement('div');bar.className='scanbar';
 bar.innerHTML=['fit','150','200','300'].map(z=>`<button data-w="${z}" class="${z==='fit'?'on':''}">${z==='fit'?'fit':z+'%'}</button>`).join('')+'<button data-full=1>⤢ full screen / rotate</button>';
 pane.prepend(bar);
 bar.onclick=e=>{const b=e.target.closest('button');if(!b)return;if(b.dataset.full){open(0);return}
  [...bar.children].forEach(x=>x.classList.toggle('on',x===b));imgs.forEach(i=>{i.style.maxWidth=b.dataset.w==='fit'?'100%':'none';i.style.width=b.dataset.w==='fit'?'':b.dataset.w+'%'})};
 const lb=document.getElementById('lb'),li=lb.querySelector('img');let k=0,s=1,x=0,y=0,rot=0,drag=null;
 // rotation about the image centre; translate/scale act on the rotated image's bounding box
 const apply=()=>{const w=li.naturalWidth,h=li.naturalHeight;li.style.transform=`translate(${x}px,${y}px) scale(${s}) translate(${(rot%180?h:w)/2}px,${(rot%180?w:h)/2}px) rotate(${rot}deg) translate(${-w/2}px,${-h/2}px)`};
 const fit=()=>{const w=rot%180?li.naturalHeight:li.naturalWidth,h=rot%180?li.naturalWidth:li.naturalHeight;s=Math.min(innerWidth/w,innerHeight/h)*.96;x=(innerWidth-w*s)/2;y=(innerHeight-h*s)/2;apply()};
 const rotate=()=>{rot=(rot+90)%360;try{sessionStorage.setItem('rot',rot)}catch(e){}fit()};
 function open(i){k=(i+imgs.length)%imgs.length;try{rot=+sessionStorage.getItem('rot')||0}catch(e){}li.onload=fit;li.src=imgs[k].src;lb.style.display='block';if(li.complete)fit()}
 imgs.forEach((im,i)=>im.addEventListener('click',()=>open(i)));
 const zoom=(f,cx=innerWidth/2,cy=innerHeight/2)=>{x=cx-(cx-x)*f;y=cy-(cy-y)*f;s*=f;apply()};
 lb.addEventListener('wheel',e=>{e.preventDefault();zoom(e.deltaY<0?1.15:1/1.15,e.clientX,e.clientY)},{passive:false});
 lb.querySelector('.lbbar').onclick=e=>{const z=e.target.dataset.z;if(z==='+')zoom(1.3);else if(z==='-')zoom(1/1.3);else if(z==='fit')fit();else if(z==='rot')rotate();else if(z==='x')lb.style.display='none'};
 li.addEventListener('pointerdown',e=>{drag=[e.clientX-x,e.clientY-y];li.setPointerCapture(e.pointerId)});
 li.addEventListener('pointermove',e=>{if(drag){x=e.clientX-drag[0];y=e.clientY-drag[1];apply()}});
 li.addEventListener('pointerup',()=>drag=null);
 let pinch=null;lb.addEventListener('touchmove',e=>{if(e.touches.length===2){const d=Math.hypot(e.touches[0].clientX-e.touches[1].clientX,e.touches[0].clientY-e.touches[1].clientY);if(pinch)zoom(d/pinch,(e.touches[0].clientX+e.touches[1].clientX)/2,(e.touches[0].clientY+e.touches[1].clientY)/2);pinch=d;e.preventDefault()}},{passive:false});
 lb.addEventListener('touchend',()=>pinch=null);
 addEventListener('keydown',e=>{if(lb.style.display!=='block')return;if(e.key==='Escape')lb.style.display='none';if(e.key==='ArrowRight')open(k+1);if(e.key==='ArrowLeft')open(k-1);if(e.key==='+'||e.key==='=')zoom(1.3);if(e.key==='-')zoom(1/1.3);if(e.key==='r'||e.key==='R')rotate()});
});})();
</script>"""

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
        s = re.sub(r'<meta name=viewport[^>]*><style id=tblenh>.*?</style>\s*<div id=lb>.*?</div></div>\s*<script>\s*\(function\(\)\{const top=.*?\}\)\(\);\s*</script>', "", s, flags=re.S)
        s = re.sub(r"<span class=yrnote>.*?</span>", "", s, flags=re.S)
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
            if v.get("jstor") and str(t.get("o", "")).lstrip("0").isdigit() and int(t["o"]) < 99999:
                bits.append(f'🔍 <a href="{v["jstor"]}?seq={int(t["o"])}" target=_blank rel=noopener>original page at full resolution (JSTOR)</a>')
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
        s = re.sub(r"(<body[^>]*>)", lambda m: m.group(1) + f"<style id=navcss>{NAVCSS}</style>" + nav("", "../") + CTX_CSS + TABLE_ENH, s, count=1)
        if pgid in pos:
            t = T[order[pos[pgid][0]][pos[pgid][1]]]
            if t.get("ry") and t["y"]:
                # the header shows the catalogue year; say when the source volume's census differs
                s = s.replace("</header>", f"<span class=yrnote>census of {t['y']} (from the source volume; catalogue said {t['ry']})</span></header>", 1)
        s = s.replace("</header>", "</header>" + ctx, 1)
        open(f, "w").write(s)
        n += 1
    print(f"patched {n} table pages")


if __name__ == "__main__":
    main()
