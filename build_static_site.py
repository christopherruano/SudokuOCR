#!/usr/bin/env python3
"""Render the Census Explorer to a self-contained static site (no server needed).

Output: census_site/
  index.html         choropleth map + entry
  tree.html          Miller-column navigator (region > province > place > table)
  tables/<page>.html each table's matrix beside its scan image
  img/<hash>.png     the scans (copied, referenced relatively)
  data/<page>.csv/.xlsx  per-table matrix downloads + catalog_tables.csv

Reuses census_explorer's builders. Run after build_explorer_db.py.
"""
import os, sqlite3, shutil, json, html, hashlib
import census_explorer as ce
import census_titles as ct

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "census_site")
SV1 = os.path.dirname(os.path.abspath(ce.DB))   # sendout_v1 dir
for sub in ("", "tables", "img", "data"):
    os.makedirs(os.path.join(OUT, sub), exist_ok=True)
DISTRICTISH = {"district", "sub_district", "urban", "division"}


def esc(s):
    return html.escape(str(s or ""))


def render_matrices_html(matrices):
    out = []
    for m in matrices:
        out.append(f"<h4>{esc(m['section']) or '(table)'}</h4><table class=mx>")
        if m["has_groups"]:
            out.append("<thead><tr><th rowspan=2 class=rh></th>")
            for g, span in m["groupspans"]:
                out.append(f'<th class=grp colspan={span}>{esc(g)}</th>')
            out.append("</tr><tr>")
            for c in m["cols"]:
                out.append(f"<th>{esc(c[1])}</th>")
            out.append("</tr></thead>")
        else:
            out.append("<thead><tr><th class=rh></th>")
            for c in m["cols"]:
                out.append(f"<th>{esc(c[1] or c[0])}</th>")
            out.append("</tr></thead>")
        out.append("<tbody>")
        for r in m["rows"]:
            out.append(f"<tr><td class=rh>{esc(r)}</td>")
            for c in m["cols"]:
                out.append(f"<td>{esc(m['grid'].get((r, c), ''))}</td>")
            out.append("</tr>")
        out.append("</tbody></table>")
    return "".join(out)


TABLE_TMPL = """<!doctype html><html><head><meta charset=utf-8><title>{title}</title>
<style>
 *{{box-sizing:border-box}} body{{font:13px/1.5 system-ui,sans-serif;margin:0;color:#1a1a1a}}
 header{{background:#2c3e50;color:#fff;padding:10px 18px}} header a{{color:#9cf;text-decoration:none}}
 .wrap{{display:flex;height:calc(100vh - 44px)}}
 .imgs{{width:48%;overflow:auto;background:#222;padding:10px;text-align:center}}
 .imgs img{{max-width:100%;margin-bottom:8px;background:#fff;border:1px solid #444}}
 .pagelbl{{color:#9cf;font-size:12px;margin:2px}}
 .data{{flex:1;overflow:auto;padding:12px 16px}} .meta{{color:#456;margin:0 0 10px}}
 h4{{margin:16px 0 4px;color:#1f3a52;font-size:13px}}
 table.mx{{border-collapse:collapse;margin-bottom:8px;font-size:12px}}
 table.mx th,table.mx td{{border:1px solid #d6dde2;padding:2px 7px;text-align:right;white-space:nowrap}}
 table.mx th{{background:#eef3f7;text-align:center}}
 table.mx td.rh{{text-align:left;background:#fafbfc;font-weight:600;white-space:normal;max-width:230px}}
 table.mx thead .grp{{background:#dde7ee}}
 .pill{{display:inline-block;background:#e8eef3;border-radius:10px;padding:1px 8px;font-size:11px}}
 .badges{{margin:3px 0 8px}} .badge{{display:inline-block;background:#eef3f7;border:1px solid #dde7ee;
   border-radius:9px;padding:1px 8px;font-size:11px;color:#345;margin:0 4px 3px 0}}
 .badge.topic{{background:#e7f0ff;border-color:#cfe0ff;font-weight:600;color:#234}}
 .rawt{{color:#789;font-size:11px;font-style:italic}}
 a.dl{{background:#1d6f42;color:#fff;padding:2px 9px;border-radius:4px;text-decoration:none}}
</style></head><body>
<header><a href="../tree.html">← back to tree</a> &nbsp; {place} · {year}
 · <span class=pill>{topic}</span></header>
<div class=wrap>
 <div class=imgs>{imgs}</div>
 <div class=data>
   <p class=meta><b title="{rawtitle}">{title}</b><br>
   <span class=badges>{badges}</span>
   {etype} · {nfacts} cells
   &nbsp; <a class=dl href="../data/{page}.csv">⬇ CSV</a>
   &nbsp; <a class=dl href="../data/{page}.xlsx">⬇ Excel</a>{note}
   <br><span class=rawt>scanned title: “{rawtitle}”</span></p>
   {matrices}
 </div>
</div></body></html>"""


def main():
    con = sqlite3.connect(f"file:{ce.DB}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    qt = con.execute("SELECT * FROM tables WHERE status IN "
                     "('confirmed_sent','recovered','keeper_accepted','keeper_review','assembled') "
                     "ORDER BY broad_region, canonical_name, year").fetchall()
    # "maximally good" public site: exclude tables listed in static_exclude.txt
    # (the failing Table VII not yet verified) so only good tables are published.
    _excl_path = os.path.join(SV1, "static_exclude.txt")
    if os.path.exists(_excl_path):
        _excl = {l.strip() for l in open(_excl_path) if l.strip()}
        before = len(qt)
        qt = [t for t in qt if t["table_id"] not in _excl]
        print(f"  excluded {before - len(qt)} not-yet-good tables (static_exclude.txt)")
    # globally-unique, case-insensitive-safe page names (hash suffix avoids APFS collisions)
    page = {}
    for t in qt:
        h = hashlib.md5(t["table_id"].encode()).hexdigest()[:7]
        page[t["table_id"]] = f"{ce.safe_filename(t)[:70]}_{h}"

    # per-table pages + data + images
    copied = set()
    nav = {}
    for i, t in enumerate(qt):
        pg = page[t["table_id"]]
        tro = dict(t)
        topic = ct.subject_topic(t["title"], tro.get("dims_present"), t["canonical_topic"], t["subject"])
        series = ct.parse_series(t["title"])
        cleant = ct.clean_title(t["title"]) or (t["title"] or "")
        if not os.path.exists(os.path.join(OUT, "tables", pg + ".html")):  # skip already-rendered (fast re-runs)
            facts = con.execute("SELECT * FROM facts WHERE table_id=? ORDER BY rowid", (t["table_id"],)).fetchall()
            mats = ce.build_matrices(facts)
            open(os.path.join(OUT, "data", pg + ".csv"), "w").write(ce.build_table_csv(t, mats))
            with open(os.path.join(OUT, "data", pg + ".xlsx"), "wb") as fh:
                fh.write(ce.build_table_xlsx(t, mats))
            raw = (t["member_images"] if "member_images" in t.keys() and t["member_images"] else t["source_image"]) or ""
            parts = [p for p in raw.replace(";", "|").replace(",", "|").split("|") if p.strip()]
            locs = []
            for p in parts:
                loc = ce.resolve_img(p.strip())
                if loc and os.path.basename(loc) not in [os.path.basename(x) for x in locs]:
                    locs.append(loc)
            imgs_html = []
            for n, loc in enumerate(locs, 1):
                b = os.path.basename(loc)
                dest = os.path.join(OUT, "img", b)
                if not os.path.exists(dest):
                    shutil.copy(loc, dest)
                copied.add(b)
                imgs_html.append(f'<div class=pagelbl>page {n} of {len(locs)}</div><img loading=lazy src="../img/{b}">')
            note = f'<br><span style="color:#a60">⚑ {esc(t["review_note"])}</span>' if t["review_note"] else ""
            chips = ct.badges(tro)
            badges_html = (f'<span class="badge topic">{esc(topic)}</span>'
                           + "".join(f'<span class=badge>{esc(c)}</span>' for c in chips[1:]))
            open(os.path.join(OUT, "tables", pg + ".html"), "w").write(TABLE_TMPL.format(
                title=esc(ct.clean_title(t["title"]) or t["title"]), rawtitle=esc(t["title"]),
                place=esc(t["canonical_name"]), year=esc(t["year"]), topic=esc(topic),
                etype=esc(t["entity_type"]), nfacts=len(facts), page=pg, note=note, badges=badges_html,
                imgs="".join(imgs_html) or '<p style="color:#fbb">scan not found</p>',
                matrices=render_matrices_html(mats)))
        # nav
        reg = t["broad_region"] or "(unspecified)"
        prov = (t["parent_geo"] if t["entity_type"] in DISTRICTISH and t["parent_geo"] else t["canonical_name"]) or "(unknown)"
        node = nav.setdefault(reg, {}).setdefault(prov, {"etype": "province", "places": {}})
        if t["canonical_name"] == prov and t["entity_type"]:
            node["etype"] = t["entity_type"]
        node["places"].setdefault(t["canonical_name"] or "(unknown)", []).append(
            {"page": pg, "year": str(t["year"] or ""), "measure": t["measure_class"],
             "subject": t["subject"], "topic": topic, "series": series,
             "title": ct.shorten(cleant, 90)})
        if (i + 1) % 400 == 0:
            print(f"  {i+1}/{len(qt)} tables rendered")

    # build nav json (sorted, with counts)
    def last(n):
        return (n.startswith("("), n)
    regions = []
    for reg in sorted(nav, key=last):
        provs = []
        for prov in sorted(nav[reg], key=last):
            node = nav[reg][prov]
            places = [{"name": pl, "tables": sorted(node["places"][pl],
                                                    key=lambda x: (x["topic"], x["year"], x["series"]))}
                      for pl in sorted(node["places"], key=last)]
            provs.append({"name": prov, "etype": node["etype"],
                          "n": sum(len(p["tables"]) for p in places), "places": places})
        regions.append({"name": reg, "n": sum(p["n"] for p in provs), "provinces": provs})
    navdata = {"regions": regions}

    shutil.copy(os.path.join(SV1, "catalog_tables.csv"), os.path.join(OUT, "data", "catalog_tables.csv"))
    idx = [{"p": page[t["table_id"]], "t": (t["title"] or "")[:90], "g": t["canonical_name"],
            "y": t["year"], "r": t["broad_region"], "k": t["subject"], "m": t["measure_class"],
            "e": t["entity_type"], "s": t["status"]} for t in qt]
    write_search(idx)
    write_tree(navdata)
    write_index(con)
    write_readme(len(qt), len(copied))
    con.close()
    print(f"\nstatic site -> {OUT}")
    print(f"  {len(qt):,} table pages · {len(copied):,} images · open census_site/index.html")


STATIC_TREE = """<!doctype html><html><head><meta charset=utf-8><title>Census — browse</title>
<style>
 *{box-sizing:border-box} body{font:14px/1.5 system-ui,sans-serif;margin:0;color:#1a1a1a}
 header{background:#2c3e50;color:#fff;padding:10px 20px} header a{color:#9cf;margin-right:14px;text-decoration:none}
 #crumb{padding:6px 20px;color:#456;font-size:13px;background:#fafbfc;border-bottom:1px solid #eee}
 #cols{display:flex;height:calc(100vh - 94px);overflow-x:auto}
 .col{width:250px;border-right:1px solid #e3e7ea;overflow-y:auto;flex:0 0 auto}
 .colh{position:sticky;top:0;background:#eef3f7;padding:6px 10px;font-weight:600;font-size:11px;color:#456;text-transform:uppercase;letter-spacing:.4px;border-bottom:1px solid #dde}
 .item{padding:5px 10px;cursor:pointer;display:flex;justify-content:space-between;gap:8px;border-bottom:1px solid #f2f4f6}
 .item:hover{background:#f0f6ff} .item.sel{background:#dcebff}
 .item .n{color:#8aa;font-size:11px;white-space:nowrap} .chev{color:#9ab}
 .pill{display:inline-block;background:#e8eef3;border-radius:8px;padding:0 6px;font-size:10px;color:#456}
 .tbl{padding:6px 10px;border-bottom:1px solid #f2f4f6;text-decoration:none;color:#163;display:block}
 .tbl:hover{background:#eafff1} .tbl .yr{color:#a60;font-weight:700;margin-right:5px}
 .tbl .m{background:#e8eef3;border-radius:8px;padding:0 6px;font-size:10px;color:#456} .tbl .t{color:#333;font-size:12.5px}
 .tsub{position:sticky;top:27px;background:#eaf0fb;color:#234;font-weight:600;font-size:11px;
   padding:4px 10px;border-top:1px solid #d6e2f5;border-bottom:1px solid #d6e2f5}
 .tbl .ser{background:#e7f0ff;border-radius:8px;padding:0 6px;font-size:10px;color:#246;margin-right:5px}
</style></head><body>
<header><b>Census — browse</b> <a href="tree.html">🗂 Columns</a><a href="index.html">🗺 Map</a><a href="search.html">🔎 Search</a>
 <span style="opacity:.7;font-size:12px">region ▸ province/state ▸ place ▸ table · darker = more · click a table for its matrix + scan</span></header>
<div id=crumb></div><div id=cols></div>
<script>
const DATA=__DATA__;
const cols=document.getElementById('cols'),crumb=document.getElementById('crumb');let path=[];
function esc(s){return (s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;');}
function grn(n,mx){if(!mx)return '#e8f5e9';const t=Math.pow(n/mx,0.6),a=[0xe8,0xf5,0xe9],b=[0x1b,0x5e,0x20];
 return 'rgb('+a.map((v,i)=>Math.round(v+(b[i]-v)*t)).join(',')+')';}
function clearFrom(i){while(cols.children.length>i)cols.removeChild(cols.lastChild);path=path.slice(0,i);drawCrumb();}
function drawCrumb(){crumb.innerHTML='All India'+path.map(p=>' › <b>'+esc(p)+'</b>').join('');}
function sel(c,el){[...c.querySelectorAll('.item')].forEach(x=>x.classList.remove('sel'));el.classList.add('sel');}
function col(h){const c=document.createElement('div');c.className='col';c.innerHTML='<div class=colh>'+h+'</div>';return c;}
function row(label,cnt,onclick,tint){const el=document.createElement('div');el.className='item';
 if(tint)el.style.borderLeft='5px solid '+tint;el.innerHTML='<span>'+label+'</span><span class=n>'+cnt+'</span>';el.onclick=()=>onclick(el);return el;}
function regions(){const c=col('Region · darker = more');const mx=Math.max(...DATA.regions.map(r=>r.n));
 DATA.regions.forEach(r=>c.appendChild(row(esc(r.name),r.n+' <span class=chev>▸</span>',el=>{clearFrom(1);sel(c,el);path=[r.name];drawCrumb();cols.appendChild(provinces(r));},grn(r.n,mx))));cols.appendChild(c);}
function provinces(r){const c=col('Province / State');const mx=Math.max(...r.provinces.map(p=>p.n));
 r.provinces.forEach(p=>c.appendChild(row(esc(p.name)+' <span class=pill>'+esc(p.etype)+'</span>',p.n+' <span class=chev>▸</span>',el=>{clearFrom(2);sel(c,el);path=[r.name,p.name];drawCrumb();cols.appendChild(places(p));},grn(p.n,mx))));return c;}
function places(p){const c=col('Place');const mx=Math.max(...p.places.map(pl=>pl.tables.length));
 p.places.forEach(pl=>c.appendChild(row(esc(pl.name),pl.tables.length+' <span class=chev>▸</span>',el=>{clearFrom(3);sel(c,el);path=path.slice(0,2).concat([pl.name]);drawCrumb();cols.appendChild(tables(pl));},grn(pl.tables.length,mx))));return c;}
function tables(pl){const c=document.createElement('div');c.className='col';c.style.width='360px';c.innerHTML='<div class=colh>Tables</div>';
 pl.tables.forEach(t=>{const a=document.createElement('a');a.className='tbl';a.href='tables/'+t.page+'.html';
  a.innerHTML='<span class=yr>'+t.year+'</span><span class=m>'+esc(t.subject)+'/'+esc(t.measure)+'</span> <span class=t>'+esc(t.title)+'</span>';c.appendChild(a);});return c;}
drawCrumb();regions();
const op=new URLSearchParams(location.search).get('open');
if(op){const e=[...cols.children[0].querySelectorAll('.item')].find(x=>x.textContent.startsWith(op));if(e)e.click();}
</script></body></html>"""


def write_tree(navdata):
    open(os.path.join(OUT, "tree.html"), "w").write(STATIC_TREE.replace("__DATA__", json.dumps(navdata)))


SEARCH_PAGE = """<!doctype html><html><head><meta charset=utf-8><title>Census — search</title>
<style>
 *{box-sizing:border-box} body{font:14px/1.5 system-ui,sans-serif;margin:0;color:#1a1a1a}
 header{background:#2c3e50;color:#fff;padding:10px 20px} header a{color:#9cf;margin-right:14px;text-decoration:none}
 .box{padding:12px 22px;background:#f4f6f8;border-bottom:1px solid #dde;position:sticky;top:0;z-index:5}
 #q{width:100%;max-width:760px;padding:9px 12px;font-size:16px;border:1px solid #bcd;border-radius:6px}
 .facets{display:flex;flex-wrap:wrap;gap:10px;align-items:center;margin-top:9px}
 .facets label{font-size:11px;color:#567;text-transform:uppercase;letter-spacing:.3px;margin-right:3px}
 select,.facets input{padding:5px 7px;border:1px solid #cce;border-radius:5px;font-size:13px}
 .yrwrap{display:flex;align-items:center;gap:5px;background:#fffbe9;border:1px solid #f0d890;border-radius:6px;padding:4px 9px}
 .yrwrap input{width:62px} .yrwrap b{color:#a60}
 #cnt{margin-left:6px;color:#678;font-size:13px} .clear{cursor:pointer;color:#268;font-size:12px}
 #res{padding:6px 0}
 a.row{display:block;padding:7px 22px;border-bottom:1px solid #f0f2f4;text-decoration:none;color:#1a1a1a}
 a.row:hover{background:#f0f6ff}
 .yr{color:#a60;font-weight:700;margin-right:6px} .pl{font-weight:600}
 .pill{display:inline-block;background:#e8eef3;border-radius:8px;padding:0 6px;font-size:11px;color:#456;margin:0 4px}
 .st{font-size:10px;color:#789} .t{color:#333}
</style></head><body>
<header><b>Census — search</b> <a href="tree.html">🗂 Columns</a><a href="index.html">🗺 Map</a><a href="search.html">🔎 Search</a></header>
<div class=box>
 <input id=q autofocus placeholder="type words — title, place, caste, table number…  (e.g. bombay civil condition)">
 <div class=facets>
  <span class=yrwrap><b>📅 Year</b> <input type=number id=y0 placeholder=from> – <input type=number id=y1 placeholder=to>
    <span id=decs></span></span>
  <span><label>Subject</label><select id=fk></select></span>
  <span><label>Measure</label><select id=fm></select></span>
  <span><label>Region</label><select id=fr></select></span>
  <span><label>Geo type</label><select id=fe></select></span>
  <span><label>Status</label><select id=fs></select></span>
  <span class=clear id=clr>✕ clear all</span>
  <span id=cnt></span>
 </div>
</div>
<div id=res></div>
<script>
const IDX=__IDX__;
IDX.forEach(t=>t._h=(t.t+' '+t.g+' '+t.r+' '+t.k+' '+t.m+' '+t.y).toLowerCase());
const $=id=>document.getElementById(id);
function esc(s){return (s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;');}
function fill(sel,key){const vals=[...new Set(IDX.map(t=>t[key]).filter(Boolean))].sort();
  sel.innerHTML='<option value="">(any)</option>'+vals.map(v=>`<option>${esc(v)}</option>`).join('');}
fill($('fk'),'k');fill($('fm'),'m');fill($('fr'),'r');fill($('fe'),'e');fill($('fs'),'s');
// decade quick-buttons
const yrs=IDX.map(t=>+t.y).filter(Boolean), ymin=Math.min(...yrs), ymax=Math.max(...yrs);
$('decs').innerHTML=' '+[1871,1881,1891,1901,1911,1921,1931,1941,1951,1961].map(d=>`<a href="#" data-d="${d}" style="font-size:11px;margin:0 2px;color:#a60">${("'"+String(d).slice(2))}</a>`).join('');
$('decs').querySelectorAll('a').forEach(a=>a.onclick=e=>{e.preventDefault();const d=+a.dataset.d;$('y0').value=d;$('y1').value=d;run();});
function run(){
 const terms=$('q').value.toLowerCase().split(/\\s+/).filter(Boolean);
 const k=$('fk').value,m=$('fm').value,r=$('fr').value,e=$('fe').value,s=$('fs').value;
 const y0=+$('y0').value||ymin, y1=+$('y1').value||ymax;
 const out=[];
 for(const t of IDX){
   if(k&&t.k!==k)continue; if(m&&t.m!==m)continue; if(r&&t.r!==r)continue;
   if(e&&t.e!==e)continue; if(s&&t.s!==s)continue;
   const ty=+t.y; if(ty&&(ty<y0||ty>y1))continue;
   if(!terms.every(w=>t._h.includes(w)))continue;
   out.push(t); if(out.length>=500)break;
 }
 $('cnt').textContent=(out.length>=500?'500+':out.length)+' matches';
 res.innerHTML=out.map(t=>`<a class=row href="tables/${t.p}.html"><span class=yr>${t.y}</span><span class=pl>${esc(t.g)}</span><span class=pill>${esc(t.k)}/${esc(t.m)}</span><span class=t>${esc(t.t)}</span> <span class=st>${esc(t.s)}</span></a>`).join('');
}
['q','y0','y1'].forEach(id=>$(id).addEventListener('input',run));
['fk','fm','fr','fe','fs'].forEach(id=>$(id).addEventListener('change',run));
$('clr').onclick=()=>{['q','y0','y1','fk','fm','fr','fe','fs'].forEach(id=>$(id).value='');run();};
run();
</script></body></html>"""


def write_search(idx):
    open(os.path.join(OUT, "search.html"), "w").write(SEARCH_PAGE.replace("__IDX__", json.dumps(idx)))


def write_index(con):
    rows = con.execute("SELECT broad_region, COUNT(*) n FROM tables WHERE status IN "
                       "('confirmed_sent','recovered','keeper_accepted','keeper_review','assembled') "
                       "AND broad_region!='' GROUP BY broad_region").fetchall()
    counts = {r["broad_region"]: r["n"] for r in rows}
    mx = max([counts.get(r, 0) for r in ce.MAP_GEO["regions"]] + [1])
    ctx = "".join(f'<path class=ctx d="{p}"/>' for p in ce.MAP_GEO["context"])
    regs, lbls = [], []
    for reg, g in ce.MAP_GEO["regions"].items():
        n = counts.get(reg, 0); t = (n / mx) ** 0.6 if n else 0
        fill, txt = ce._choropleth(t)
        regs.append(f'<a class=region href="tree.html?open={esc(reg).replace(" ","%20")}"><title>{esc(reg)}: {n} tables</title>'
                    f'<path d="{g["path"]}" fill="{fill}"/></a>')
        lbls.append(f'<text class=lbl x="{g["cx"]}" y="{g["cy"]}" fill="{txt}">{esc(reg)}'
                    f'<tspan class=c x="{g["cx"]}" dy="14">{n} tables</tspan></text>')
    alln = counts.get("All-India", 0)
    page = (ce.MAP_PAGE.replace('href="/tree"', 'href="tree.html"').replace('href="/map"', 'href="index.html"')
            .replace('href="/search"', 'href="search.html"')
            .replace("__VB__", ce.MAP_GEO["viewBox"]).replace("__CTX__", ctx)
            .replace("__REGIONS__", "".join(regs) + "".join(lbls))
            .replace("__ALL__", f'<a class=all href="tree.html?open=All-India">\U0001f1ee\U0001f1f3 All-India tables · {alln}</a>'))
    open(os.path.join(OUT, "index.html"), "w").write(page)


def write_readme(ntab, nimg):
    open(os.path.join(OUT, "README.txt"), "w").write(
        f"Historical Indian Census — static explorer ({ntab} tables, {nimg} scans)\n\n"
        "Open index.html in any browser (works offline). Click a region on the map or use\n"
        "tree.html to drill Region > Province/State > place > table. Each table page shows\n"
        "the reconstructed matrix beside its source scan, with CSV/Excel downloads.\n\n"
        "To share online: drag this folder onto netlify.com/drop, or host on GitHub Pages.\n")


if __name__ == "__main__":
    main()
