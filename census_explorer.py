#!/usr/bin/env python3
"""Census Explorer — browse the historical Indian Census corpus by topic / region /
year / measure, see which tables match (the MAP), and pull every matching cell
fact into one combined CSV.

Run:  python3 census_explorer.py   then open http://127.0.0.1:5000
Backed by sendout_v1/census_explorer.sqlite (built by build_explorer_db.py).
"""
import csv, io, os, glob, sqlite3, urllib.parse, zipfile, re, xlsxwriter
from flask import Flask, request, Response, render_template_string, send_file, redirect, g, abort

ROOT = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(ROOT, "export_final/sendout_final/sendout_v1/census_explorer.sqlite")
app = Flask(__name__)

# basename(image hash).png -> local path  (source_image paths are cluster paths; resolve locally)
print("indexing source images …")
IMG_INDEX = {}
for _pat in ("export/**/*.png", "organized_results/**/*.png"):
    for _p in glob.glob(os.path.join(ROOT, _pat), recursive=True):
        IMG_INDEX.setdefault(os.path.basename(_p), _p)
print(f"  {len(IMG_INDEX):,} images indexed")


def up(s):
    return urllib.parse.quote(str(s))


def resolve_img(source_image):
    if not source_image:
        return None
    if os.path.exists(source_image):
        return source_image
    return IMG_INDEX.get(os.path.basename(source_image))

# filter field -> facts column
FILTERS = {
    "subject": "subject", "measure_class": "measure_class", "topic": "canonical_topic",
    "region": "broad_region", "entity_type": "entity_type", "geography": "canonical_name",
    "year": "year", "measure": "measure_unit", "additive": "additive",
    "age_aggregate": "age_aggregate", "status": "status", "sex": "sex",
}
FACT_COLS = ["fact_id", "table_id", "year", "canonical_name", "entity_type", "broad_region",
             "subject", "measure_class", "additive", "age_aggregate", "canonical_topic",
             "table_family", "measure_unit", "section", "row_label_std", "column_group_std",
             "sub_column_std", "sex", "civil_condition", "religion", "resolved_age_label_std",
             "resolved_age_lower", "resolved_age_upper", "value", "title", "source_image", "status"]


def db():
    if "db" not in g:
        g.db = sqlite3.connect(DB)
        g.db.row_factory = sqlite3.Row
    return g.db


@app.teardown_appcontext
def _close(_):
    d = g.pop("db", None)
    if d:
        d.close()


def options(col):
    rows = db().execute(f'SELECT DISTINCT "{col}" v FROM facts WHERE "{col}"!="" ORDER BY 1').fetchall()
    return [r["v"] for r in rows]


def where_clause(args):
    clauses, params = [], []
    for key, col in FILTERS.items():
        vals = args.getlist(key)
        vals = [v for v in vals if v]
        if vals:
            clauses.append(f'"{col}" IN ({",".join("?"*len(vals))})')
            params.extend(vals)
    sub = args.get("subtree")  # a province/state node: itself + its districts
    if sub:
        clauses.append("(canonical_name=? OR parent_geo=?)"); params += [sub, sub]
    y0, y1 = args.get("year_from"), args.get("year_to")
    if y0:
        clauses.append("CAST(year AS INTEGER) >= ?"); params.append(int(y0))
    if y1:
        clauses.append("CAST(year AS INTEGER) <= ?"); params.append(int(y1))
    q = args.get("q", "").strip()
    table_ids = None
    if q:
        ft = db().execute("SELECT table_id FROM tables_fts WHERE tables_fts MATCH ?", (q + "*",)).fetchall()
        table_ids = [r["table_id"] for r in ft]
        if table_ids:
            clauses.append(f'table_id IN ({",".join("?"*len(table_ids))})')
            params.extend(table_ids)
        else:
            clauses.append("1=0")
    return (" AND ".join(clauses) if clauses else "1=1"), params


PAGE = """
<!doctype html><html><head><meta charset=utf-8><title>Census Explorer</title>
<style>
 body{font:14px/1.5 system-ui,sans-serif;margin:0;color:#1a1a1a}
 header{background:#2c3e50;color:#fff;padding:14px 22px}
 header h1{margin:0;font-size:19px} header span{opacity:.7;font-size:13px}
 .wrap{display:flex} .side{width:300px;padding:18px;background:#f4f6f8;border-right:1px solid #dde;height:calc(100vh - 52px);overflow:auto}
 .main{flex:1;padding:20px 26px;overflow:auto;height:calc(100vh - 52px)}
 fieldset{border:1px solid #cdd;border-radius:6px;margin:0 0 12px;padding:8px 10px}
 legend{font-weight:600;font-size:12px;color:#456;text-transform:uppercase;letter-spacing:.4px}
 select[multiple]{width:100%;height:120px} input[type=text],input[type=number]{width:100%;padding:5px;border:1px solid #ccd;border-radius:4px;box-sizing:border-box}
 .row{display:flex;gap:6px} label.s{font-size:11px;color:#678}
 button{background:#2980b9;color:#fff;border:0;padding:9px 14px;border-radius:5px;cursor:pointer;font-size:14px}
 button.alt{background:#27ae60} a.btn{display:inline-block;text-decoration:none}
 .stat{display:flex;gap:24px;margin:6px 0 16px;flex-wrap:wrap}
 .stat div{background:#eef3f7;border-radius:6px;padding:10px 16px} .stat b{font-size:22px;display:block}
 table{border-collapse:collapse;width:100%;font-size:13px} th,td{border-bottom:1px solid #e3e7ea;padding:5px 8px;text-align:left}
 th{background:#fafbfc;position:sticky;top:0} tr:hover{background:#f7fafc}
 .pill{display:inline-block;background:#e8eef3;border-radius:10px;padding:1px 8px;font-size:11px;color:#456}
 .muted{color:#789;font-size:12px}
</style></head><body>
<header><h1>Census Explorer <span>· 1872–1961 · {{ '{:,}'.format(total_facts) }} facts · {{ '{:,}'.format(total_tables) }} tables</span>
 &nbsp;&nbsp;<a href="/tree" style="color:#9cf;font-size:14px">🌳 Browse the tree →</a></h1></header>
<form method=get>
<div class=wrap>
 <div class=side>
  <fieldset><legend>Search titles</legend><input type=text name=q value="{{ args.get('q','') }}" placeholder="e.g. literacy, civil condition"></fieldset>
  {% for key,label,opts in facets %}
  <fieldset><legend>{{label}}</legend>
   <select multiple name="{{key}}">
   {% for o in opts %}<option value="{{o}}" {{'selected' if o in args.getlist(key) else ''}}>{{o}}</option>{% endfor %}
   </select></fieldset>
  {% endfor %}
  <fieldset><legend>Year range</legend>
   <div class=row><div><label class=s>from</label><input type=number name=year_from value="{{args.get('year_from','')}}"></div>
   <div><label class=s>to</label><input type=number name=year_to value="{{args.get('year_to','')}}"></div></div></fieldset>
  <button type=submit>Apply filters</button>
 </div>
 <div class=main>
  <div class=stat>
   <div><b>{{ '{:,}'.format(n_facts) }}</b>matching facts</div>
   <div><b>{{ '{:,}'.format(n_tables) }}</b>matching tables</div>
   <div><b>{{ years }}</b>year span</div>
  </div>
  <p>
   <a class=btn href="/matrices.zip?{{ qs }}"><button class=alt type=button>⬇ Download these tables as Excel (.zip)</button></a>
   &nbsp;<span class=muted>Each table as a clean formatted matrix. Or click 🔍 on any row to view it.</span>
  </p>
  <h3>Matching tables <span class=muted>(the map — top {{ tables|length }} of {{ '{:,}'.format(n_tables) }})</span></h3>
  <table><tr><th>Year</th><th>Geography</th><th>Type</th><th>Topic</th><th>Measure</th><th>Facts</th><th>Status</th><th>Title</th></tr>
  {% for t in tables %}
   <tr><td>{{t['year']}}</td><td>{{t['canonical_name']}}</td><td><span class=pill>{{t['entity_type']}}</span></td>
   <td>{{t['canonical_topic']}}</td><td>{{t['measure_category']}}</td><td>{{t['n_facts']}}</td>
   <td><span class=pill>{{t['status']}}</span></td>
   <td title="{{t['title']}}"><a href="/table?tid={{ t['table_id']|urlencode }}" target=_blank>🔍</a> {{ t['title'][:64] }}</td></tr>
  {% endfor %}
  </table>
 </div>
</div></form></body></html>
"""


@app.route("/")
def root():
    return redirect("/tree")


@app.route("/search")
def home():
    where, params = where_clause(request.args)
    d = db()
    n_facts = d.execute(f"SELECT COUNT(*) c FROM facts WHERE {where}", params).fetchone()["c"]
    trows = d.execute(f"SELECT DISTINCT table_id FROM facts WHERE {where}", params).fetchall()
    tids = [r["table_id"] for r in trows]
    n_tables = len(tids)
    yr = d.execute(f"SELECT MIN(CAST(year AS INT)) a, MAX(CAST(year AS INT)) b FROM facts WHERE {where}", params).fetchone()
    years = f"{yr['a']}–{yr['b']}" if yr["a"] else "–"
    tables = []
    if tids:
        sample = tids[:200]
        tables = d.execute(
            f"SELECT * FROM tables WHERE table_id IN ({','.join('?'*len(sample))}) ORDER BY year, canonical_name LIMIT 200",
            sample).fetchall()
    facets = [
        ("subject", "Subject", options("subject")),
        ("measure_class", "Measure class", options("measure_class")),
        ("region", "Broad region", options("broad_region")),
        ("entity_type", "Geography type", options("entity_type")),
        ("measure", "Measure unit", options("measure_unit")),
        ("status", "Data status", options("status")),
        ("sex", "Sex", options("sex")),
    ]
    qs = request.query_string.decode()
    return render_template_string(
        PAGE, args=request.args, facets=facets, n_facts=n_facts, n_tables=n_tables,
        years=years, tables=tables, qs=qs,
        total_facts=d.execute("SELECT COUNT(*) c FROM facts").fetchone()["c"],
        total_tables=d.execute("SELECT COUNT(*) c FROM tables").fetchone()["c"])


TABLE_PAGE = """
<!doctype html><html><head><meta charset=utf-8><title>{{ t['title'][:60] }}</title>
<style>
 body{font:13px/1.5 system-ui,sans-serif;margin:0;color:#1a1a1a}
 header{background:#2c3e50;color:#fff;padding:10px 20px} header a{color:#9cf}
 .wrap{display:flex;height:calc(100vh - 44px)}
 .imgs{width:48%;overflow:auto;background:#222;padding:10px;text-align:center;resize:horizontal}
 .imgs img{max-width:100%;margin-bottom:8px;background:#fff;border:1px solid #444}
 .pagelbl{color:#9cf;font-size:12px;margin:2px 0}
 .data{flex:1;overflow:auto;padding:12px 16px}
 .meta{margin:0 0 10px;color:#456}
 h4{margin:16px 0 4px;color:#1f3a52;font-size:13px}
 table.mx{border-collapse:collapse;margin-bottom:8px;font-size:12px}
 table.mx th,table.mx td{border:1px solid #d6dde2;padding:2px 7px;text-align:right;white-space:nowrap}
 table.mx th{background:#eef3f7;text-align:center;position:sticky;top:0}
 table.mx td.rh{text-align:left;background:#fafbfc;font-weight:600;white-space:normal;max-width:230px}
 table.mx thead .grp{background:#dde7ee}
 .pill{display:inline-block;background:#e8eef3;border-radius:10px;padding:1px 8px;font-size:11px}
 .nofig{color:#fbb} .raw{font-size:11px;color:#789}
</style></head><body>
<header><a href="javascript:history.back()">← back</a> &nbsp; {{ t['canonical_name'] }} · {{ t['year'] }}
 · <span class=pill>{{ t['subject'] }}/{{ t['measure_class'] }}</span> · <span class=pill>{{ t['status'] }}</span>
 {% if n_pages>1 %}· <span class=pill>{{ n_pages }}-page group</span>{% endif %}</header>
<div class=wrap>
 <div class=imgs>
   {% if images %}{% for im in images %}<div class=pagelbl>page {{ loop.index }} of {{ images|length }}</div>
     <img src="/img/{{ im }}" loading=lazy>{% endfor %}
   {% else %}<p class=nofig>Source image not found locally.</p>{% endif %}
 </div>
 <div class=data>
   <p class=meta><b>{{ t['title'] }}</b><br>{{ t['entity_type'] }} · {{ '{:,}'.format(n_facts) }} cells
   &nbsp; <a class=dl style="background:#1d6f42;color:#fff;padding:2px 9px;border-radius:4px;text-decoration:none" href="/matrix.csv?tid={{ t['table_id']|urlencode }}">⬇ CSV</a>
   &nbsp; <a class=dl style="background:#1d6f42;color:#fff;padding:2px 9px;border-radius:4px;text-decoration:none" href="/matrix.xlsx?tid={{ t['table_id']|urlencode }}">⬇ Excel</a>
   {% if t['review_note'] %}<br><span style="color:#a60">⚑ {{ t['review_note'] }}</span>{% endif %}</p>
   {% for m in matrices %}
     <h4>{{ m.section or '(table)' }}</h4>
     <table class=mx>
       {% if m.has_groups %}<thead><tr><th rowspan=2 class=rh></th>
         {% for g,span in m.groupspans %}<th class=grp colspan={{span}}>{{ g }}</th>{% endfor %}</tr>
         <tr>{% for c in m.cols %}<th>{{ c[1] }}</th>{% endfor %}</tr></thead>
       {% else %}<thead><tr><th class=rh></th>{% for c in m.cols %}<th>{{ c[1] or c[0] }}</th>{% endfor %}</tr></thead>{% endif %}
       <tbody>
       {% for r in m.rows %}<tr><td class=rh>{{ r }}</td>
         {% for c in m.cols %}<td>{{ m.grid.get((r,c),'') }}</td>{% endfor %}</tr>{% endfor %}
       </tbody>
     </table>
   {% endfor %}
 </div>
</div></body></html>
"""


def build_matrices(facts):
    """Pivot long facts back into the printed table grid, per section. Robust to
    missing columns and avoids stray/degenerate columns so every table renders clean."""
    SEXLAB = {"All": "Persons", "Male": "Males", "Female": "Females"}

    def gv(f, key):
        try:
            return f[key]
        except (IndexError, KeyError):
            return None

    secs, seen = [], set()
    for f in facts:
        s = gv(f, "section") or ""
        if s not in seen:
            seen.add(s); secs.append(s)
    out = []
    for s in secs:
        fs = [f for f in facts if (gv(f, "section") or "") == s]
        cols, rows, grid = [], [], {}
        for f in fs:
            grp = gv(f, "column_group_std") or gv(f, "column_group_value") or ""
            sub = gv(f, "sub_column_std") or SEXLAB.get(gv(f, "sex") or "", "") or ""
            if not grp and not sub:
                sub = "value"   # single-value-per-row table; never leak civil/religion as a column
            ck = (grp, sub)
            rk = gv(f, "row_label_std") or gv(f, "row_group_value") or ""
            if ck not in cols:
                cols.append(ck)
            if rk not in rows:
                rows.append(rk)
            v = gv(f, "value")
            grid[(rk, ck)] = ("" if v in (None, "") else (int(v) if isinstance(v, float) and v == int(v) else v))

        # Restore the Total/Persons column the persons-cleaning removed: any column group with
        # Males & Females but no total gets Total = Males+Females (exactly the removed value).
        def _kind(sub):
            s = (sub or "").lower().rstrip(".")
            if s in ("males", "male", "m"):
                return "M"
            if s in ("females", "female", "f"):
                return "F"
            if s in ("persons", "total", "all", "p"):
                return "T"
            return ""
        groups_seen = []
        for c in cols:
            if c[0] not in groups_seen:
                groups_seen.append(c[0])
        newcols = []
        for g in groups_seen:
            gcols = [c for c in cols if c[0] == g]
            kinds = {_kind(c[1]) for c in gcols}
            mcol = next((c for c in gcols if _kind(c[1]) == "M"), None)
            fcol = next((c for c in gcols if _kind(c[1]) == "F"), None)
            if mcol and fcol and "T" not in kinds:
                tcol = (g, "Total")
                for rk in rows:
                    mv, fv = grid.get((rk, mcol)), grid.get((rk, fcol))
                    if isinstance(mv, (int, float)) and isinstance(fv, (int, float)):
                        tot = mv + fv
                        grid[(rk, tcol)] = int(tot) if tot == int(tot) else tot
                    else:
                        grid[(rk, tcol)] = ""
                newcols.append(tcol)   # Persons/Total first, matching census layout
            newcols.extend(gcols)
        cols = newcols

        has_groups = any(c[0] for c in cols)
        groupspans = []
        for c in cols:
            g = c[0] or ""
            if groupspans and groupspans[-1][0] == g:
                groupspans[-1][1] += 1
            else:
                groupspans.append([g, 1])
        out.append({"section": s, "cols": cols, "rows": rows, "grid": grid,
                    "has_groups": has_groups, "groupspans": groupspans})
    return out


def col_label(c):
    """Flat, unambiguous column header for CSV: 'Group — Sub' (or just one)."""
    g, sub = c
    if g and sub and sub != "value":
        return f"{g} — {sub}"
    return sub if (sub and sub != "value") else (g or "value")


def build_table_csv(t, matrices):
    out = io.StringIO(); w = csv.writer(out)
    w.writerow([t["title"] or "table"])
    w.writerow([f"{t['canonical_name']} | {t['year']} | {t['subject']}/{t['measure_class']} | {t['status']}"])
    if t["review_note"]:
        w.writerow([f"note: {t['review_note']}"])
    for m in matrices:
        w.writerow([])
        if m["section"]:
            w.writerow([m["section"]])
        w.writerow([""] + [col_label(c) for c in m["cols"]])
        for rk in m["rows"]:
            w.writerow([rk] + [m["grid"].get((rk, c), "") for c in m["cols"]])
    return out.getvalue()


def nav_data(subject, mc):
    """Build the region -> province/state -> place -> tables hierarchy for navigation."""
    d = db()
    where = ["status IN ('confirmed_sent','recovered')", "subject=?"]
    params = [subject]
    if mc:
        where.append("measure_class=?"); params.append(mc)
    rows = d.execute(
        f"SELECT table_id, broad_region, parent_geo, canonical_name, entity_type, year, "
        f"n_facts, measure_class, title FROM tables WHERE {' AND '.join(where)} "
        f"ORDER BY broad_region, canonical_name, year", params).fetchall()
    DISTRICTISH = {"district", "sub_district", "urban", "division"}
    tree = {}
    for r in rows:
        reg = r["broad_region"] or "(unspecified)"
        prov = (r["parent_geo"] if r["entity_type"] in DISTRICTISH and r["parent_geo"] else r["canonical_name"]) or "(unknown)"
        node = tree.setdefault(reg, {}).setdefault(prov, {"etype": "province", "places": {}})
        if r["canonical_name"] == prov and r["entity_type"]:
            node["etype"] = r["entity_type"]
        node["places"].setdefault(r["canonical_name"] or "(unknown)", []).append(r)
    def lastsort(name):  # push "(unspecified)" / "(unknown)" to the bottom
        return (name.startswith("("), name)
    regions = []
    for reg in sorted(tree, key=lastsort):
        provs = []
        for prov in sorted(tree[reg], key=lastsort):
            node = tree[reg][prov]
            places = []
            for place in sorted(node["places"], key=lastsort):
                tabs = sorted(node["places"][place], key=lambda r: str(r["year"]))
                places.append({"name": place, "tables": [
                    {"tid": t["table_id"], "year": str(t["year"] or ""), "measure": t["measure_class"],
                     "title": (t["title"] or "")[:90], "n": t["n_facts"]} for t in tabs]})
            provs.append({"name": prov, "etype": node["etype"],
                          "n": sum(len(p["tables"]) for p in places), "places": places})
        regions.append({"name": reg, "n": sum(p["n"] for p in provs), "provinces": provs})
    return {"regions": regions, "n_tables": len(rows)}


MILLER_PAGE = """
<!doctype html><html><head><meta charset=utf-8><title>Census — browse</title>
<style>
 *{box-sizing:border-box} body{font:14px/1.5 system-ui,sans-serif;margin:0;color:#1a1a1a}
 header{background:#2c3e50;color:#fff;padding:10px 20px} header a{color:#9cf;margin-right:14px;text-decoration:none}
 .bar{padding:8px 20px;background:#f4f6f8;border-bottom:1px solid #dde;display:flex;gap:12px;align-items:center;flex-wrap:wrap}
 select{padding:5px;border:1px solid #cce;border-radius:4px}
 #crumb{padding:6px 20px;color:#456;font-size:13px;background:#fafbfc;border-bottom:1px solid #eee}
 #cols{display:flex;height:calc(100vh - 134px);overflow-x:auto}
 .col{width:250px;border-right:1px solid #e3e7ea;overflow-y:auto;flex:0 0 auto}
 .colh{position:sticky;top:0;background:#eef3f7;padding:6px 10px;font-weight:600;font-size:11px;color:#456;text-transform:uppercase;letter-spacing:.4px;border-bottom:1px solid #dde;display:flex;justify-content:space-between;align-items:center;gap:6px}
 .dl{font-size:10px;background:#1d6f42;color:#fff;padding:1px 6px;border-radius:4px;text-decoration:none;white-space:nowrap}
 .item{padding:5px 10px;cursor:pointer;display:flex;justify-content:space-between;gap:8px;border-bottom:1px solid #f2f4f6}
 .item:hover{background:#f0f6ff} .item.sel{background:#dcebff}
 .item .n{color:#8aa;font-size:11px;font-weight:400;white-space:nowrap} .chev{color:#9ab}
 .pill{display:inline-block;background:#e8eef3;border-radius:8px;padding:0 6px;font-size:10px;color:#456}
 .tbl{padding:6px 10px;border-bottom:1px solid #f2f4f6;text-decoration:none;color:#163;display:block}
 .tbl:hover{background:#eafff1} .tbl .yr{color:#a60;font-weight:700;margin-right:5px}
 .tbl .m{background:#e8eef3;border-radius:8px;padding:0 6px;font-size:10px;color:#456} .tbl .t{color:#333;font-size:12.5px}
</style></head><body>
<header><b>Census — browse</b> <a href="/tree">🗂 Columns</a><a href="/map">🗺 Map</a><a href="/search">🔎 Find</a>
 <span style="opacity:.7;font-size:12px">region ▸ province/state ▸ place ▸ table — click across the columns</span></header>
<div class=bar>
 Subject <select id=subject onchange="reload()">__SUBJ__</select>
 Measure <select id=measure onchange="reload()">__MEAS__</select>
 <span style="color:#789;font-size:12px">🔍 a table opens its matrix beside the scan · ⬇ xlsx downloads that whole branch</span>
</div>
<div id=crumb></div>
<div id=cols></div>
<script>
const DATA=__DATA__, SUBJECT=__SUBJECT__, MC=__MC__;
const cols=document.getElementById('cols'), crumb=document.getElementById('crumb');
let path=[];
function sfx(){return '&subject='+encodeURIComponent(SUBJECT)+(MC?'&measure_class='+encodeURIComponent(MC):'');}
function reload(){location.href='/tree?subject='+encodeURIComponent(subject.value)+'&measure_class='+encodeURIComponent(measure.value);}
function esc(s){return (s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;');}
function dl(kind,name){const k={region:'region',province:'subtree',place:'geography'}[kind];
 return '<a class=dl title="download all as Excel" href="/matrices.zip?'+k+'='+encodeURIComponent(name)+sfx()+'">⬇ xlsx</a>';}
function clearFrom(i){while(cols.children.length>i)cols.removeChild(cols.lastChild);path=path.slice(0,i);drawCrumb();}
function drawCrumb(){crumb.innerHTML='All India'+path.map(p=>' › <b>'+esc(p)+'</b>').join('');}
function sel(col,el){[...col.querySelectorAll('.item')].forEach(x=>x.classList.remove('sel'));el.classList.add('sel');}
function col(headHtml){const c=document.createElement('div');c.className='col';c.innerHTML='<div class=colh>'+headHtml+'</div>';return c;}
function grn(n,mx){if(!mx)return '#e8f5e9';const t=Math.pow(n/mx,0.6),a=[0xe8,0xf5,0xe9],b=[0x1b,0x5e,0x20];
 return 'rgb('+a.map((v,i)=>Math.round(v+(b[i]-v)*t)).join(',')+')';}
function row(label,countHtml,onclick,tint){const el=document.createElement('div');el.className='item';
 if(tint)el.style.borderLeft='5px solid '+tint;
 el.innerHTML='<span>'+label+'</span><span class=n>'+countHtml+'</span>';el.onclick=()=>onclick(el);return el;}

function showRegions(){
 const c=col('Region <span style="font-weight:400;text-transform:none">· darker = more</span>');
 const mx=Math.max(...DATA.regions.map(r=>r.n));
 DATA.regions.forEach(r=>c.appendChild(row(esc(r.name),r.n+' <span class=chev>▸</span>',el=>{
   clearFrom(1);sel(c,el);path=[r.name];drawCrumb();cols.appendChild(showProvinces(r));},grn(r.n,mx))));
 cols.appendChild(c);
}
function showProvinces(r){
 const c=col('Province / State '+dl('region',r.name));
 const mx=Math.max(...r.provinces.map(p=>p.n));
 r.provinces.forEach(p=>c.appendChild(row(esc(p.name)+' <span class=pill>'+esc(p.etype)+'</span>',p.n+' <span class=chev>▸</span>',el=>{
   clearFrom(2);sel(c,el);path=[r.name,p.name];drawCrumb();cols.appendChild(showPlaces(p));},grn(p.n,mx))));
 return c;
}
function showPlaces(p){
 const c=col('Place '+dl('province',p.name));
 const mx=Math.max(...p.places.map(pl=>pl.tables.length));
 p.places.forEach(pl=>c.appendChild(row(esc(pl.name),pl.tables.length+' <span class=chev>▸</span>',el=>{
   clearFrom(3);sel(c,el);path=path.slice(0,2).concat([pl.name]);drawCrumb();cols.appendChild(showTables(pl));},grn(pl.tables.length,mx))));
 return c;
}
function showTables(pl){
 const c=document.createElement('div');c.className='col';c.style.width='360px';
 c.innerHTML='<div class=colh>Tables '+dl('place',pl.name)+'</div>';
 pl.tables.forEach(t=>{const a=document.createElement('a');a.className='tbl';a.target='_blank';
   a.href='/table?tid='+encodeURIComponent(t.tid);
   a.innerHTML='<span class=yr>'+t.year+'</span><span class=m>'+t.measure+'</span> <span class=t>'+esc(t.title)+'</span>';
   c.appendChild(a);});
 return c;
}
drawCrumb();showRegions();
const openReg=new URLSearchParams(location.search).get('open');
if(openReg){const r=DATA.regions.find(x=>x.name===openReg);if(r){[...cols.children[0].querySelectorAll('.item')].find(e=>e.textContent.startsWith(r.name))?.click();}}
</script></body></html>
"""


@app.route("/tree")
def tree():
    subject = request.args.get("subject", "population")
    mc = request.args.get("measure_class", "count" if subject in ("population", "civil_condition", "religion") else "")
    d = db()
    data = nav_data(subject, mc)
    subjects = [r["v"] for r in d.execute("SELECT DISTINCT subject v FROM tables WHERE subject!='' ORDER BY 1")]
    measures = [r["v"] for r in d.execute("SELECT DISTINCT measure_class v FROM tables WHERE measure_class!='' ORDER BY 1")]
    subj_opts = "".join(f'<option {"selected" if s == subject else ""}>{s}</option>' for s in subjects)
    meas_opts = '<option value="">(any)</option>' + "".join(
        f'<option {"selected" if m == mc else ""}>{m}</option>' for m in measures)
    import json as _json
    html = (MILLER_PAGE.replace("__DATA__", _json.dumps(data)).replace("__SUBJECT__", _json.dumps(subject))
            .replace("__MC__", _json.dumps(mc)).replace("__SUBJ__", subj_opts).replace("__MEAS__", meas_opts))
    return Response(html, mimetype="text/html")


with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "mapdata/regions_svg.json")) as _f:
    MAP_GEO = __import__("json").load(_f)


def _choropleth(t):  # t in [0,1] -> light->dark green; returns (fill, text)
    a, b = (0xe8, 0xf5, 0xe9), (0x1b, 0x5e, 0x20)
    rgb = tuple(round(a[i] + (b[i] - a[i]) * t) for i in range(3))
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}", ("#fff" if t > 0.45 else "#14401a")


MAP_PAGE = """
<!doctype html><html><head><meta charset=utf-8><title>Census — map</title>
<style>
 body{font:14px/1.5 system-ui,sans-serif;margin:0;color:#1a1a1a;background:#fbfdff}
 header{background:#2c3e50;color:#fff;padding:10px 20px} header a{color:#9cf;margin-right:14px;text-decoration:none}
 h2{text-align:center;margin:12px 0 2px;color:#234} .sub{text-align:center;color:#789;font-size:13px;margin:0 0 6px}
 .all{display:block;width:max-content;margin:8px auto;background:#1f3a52;color:#fff;padding:7px 16px;border-radius:8px;text-decoration:none}
 svg{display:block;margin:0 auto;max-height:calc(100vh - 175px)}
 .ctx{fill:#eef1f4;stroke:#dde2e6;stroke-width:.7}
 a.region path{stroke:#fff;stroke-width:1.1;transition:.12s;cursor:pointer}
 a.region:hover path{stroke:#103;stroke-width:2;filter:brightness(1.08)}
 .lbl{font:600 13px system-ui;text-anchor:middle;pointer-events:none}
 .lbl .c{font-weight:400;font-size:11px}
 .legend{text-align:center;color:#567;font-size:12px;margin:4px}
 .legend i{display:inline-block;width:22px;height:11px;vertical-align:middle;border:1px solid #ccc}
</style></head><body>
<header><b>Census — map</b> <a href="/tree">🗂 Columns</a><a href="/map">🗺 Map</a><a href="/search">🔎 Find</a>
 <span style="opacity:.7;font-size:12px">darker = more tables · click a region to drill into the columns</span></header>
<h2>British India &amp; the Princely States</h2>
<p class=sub>colour shows how many tables each region has — the darker, the more to explore in the tree</p>
__ALL__
<svg viewBox="__VB__" xmlns="http://www.w3.org/2000/svg">
 __CTX__
 __REGIONS__
</svg>
<p class=legend>fewer <i style="background:#e8f5e9"></i><i style="background:#a5d6a7"></i><i style="background:#66bb6a"></i><i style="background:#388e3c"></i><i style="background:#1b5e20"></i> more tables</p>
</body></html>
"""


@app.route("/map")
def census_map():
    d = db()
    rows = d.execute("SELECT broad_region, COUNT(*) n FROM tables WHERE status IN "
                     "('confirmed_sent','recovered') AND broad_region!='' GROUP BY broad_region").fetchall()
    counts = {r["broad_region"]: r["n"] for r in rows}
    mx = max([counts.get(r, 0) for r in MAP_GEO["regions"]] + [1])
    ctx = "".join(f'<path class=ctx d="{p}"/>' for p in MAP_GEO["context"])
    regs, lbls = [], []
    for reg, g in MAP_GEO["regions"].items():
        n = counts.get(reg, 0)
        t = (n / mx) ** 0.6 if n else 0
        fill, txt = _choropleth(t)
        regs.append(f'<a class=region href="/tree?open={up(reg)}"><title>{reg}: {n} tables</title>'
                    f'<path d="{g["path"]}" fill="{fill}"/></a>')
        lbls.append(f'<text class=lbl x="{g["cx"]}" y="{g["cy"]}" fill="{txt}">{reg}'
                    f'<tspan class=c x="{g["cx"]}" dy="14">{n} tables</tspan></text>')
    alln = counts.get("All-India", 0)
    allbox = f'<a class=all href="/tree?open=All-India">🇮🇳 All-India tables · {alln}</a>'
    html = (MAP_PAGE.replace("__VB__", MAP_GEO["viewBox"]).replace("__CTX__", ctx)
            .replace("__REGIONS__", "".join(regs) + "".join(lbls)).replace("__ALL__", allbox))
    return Response(html, mimetype="text/html")


@app.route("/table")
def table_detail():
    tid = request.args.get("tid", "")
    d = db()
    t = d.execute("SELECT * FROM tables WHERE table_id=?", (tid,)).fetchone()
    if not t:
        abort(404)
    facts = d.execute("SELECT * FROM facts WHERE table_id=? ORDER BY rowid", (tid,)).fetchall()
    matrices = build_matrices(facts)
    # ordered page images: assembled member images in sequence, else the single source image
    raw = (t["member_images"] if "member_images" in t.keys() and t["member_images"] else t["source_image"]) or ""
    parts = [p for p in raw.replace(";", "|").replace(",", "|").split("|") if p.strip()]
    images, seen = [], set()
    for p in parts:
        loc = resolve_img(p.strip())
        if loc:
            b = os.path.basename(loc)
            if b not in seen:
                seen.add(b); images.append(b)
    return render_template_string(TABLE_PAGE, t=t, facts=facts, matrices=matrices, images=images,
                                  n_facts=len(facts), n_pages=len(images))


@app.route("/img/<name>")
def img(name):
    p = IMG_INDEX.get(name)
    if not p or not os.path.exists(p):
        abort(404)
    return send_file(p, mimetype="image/png")


# ---- matrix Excel export (the deliverable: each table as a clean .xlsx grid) ----
def _num(v):
    if v in (None, ""):
        return None
    try:
        f = float(v)
        return int(f) if f == int(f) else f
    except (ValueError, TypeError):
        return v


def safe_filename(t):
    base = f"{t['year']}_{t['canonical_name']}_{(t['title'] or '')[:40]}"
    return (re.sub(r"[^A-Za-z0-9._-]+", "_", base).strip("_")[:90]) or "table"


def build_table_xlsx(t, matrices):
    bio = io.BytesIO()
    wb = xlsxwriter.Workbook(bio, {"in_memory": True})
    F = dict(
        title=wb.add_format({"bold": True, "font_size": 13}),
        meta=wb.add_format({"italic": True, "font_color": "#555555"}),
        sec=wb.add_format({"bold": True, "font_color": "#1f3a52", "bg_color": "#eef3f7"}),
        grp=wb.add_format({"bold": True, "align": "center", "valign": "vcenter", "bg_color": "#dde7ee", "border": 1}),
        hdr=wb.add_format({"bold": True, "align": "center", "valign": "vcenter", "bg_color": "#eef3f7", "border": 1, "text_wrap": True}),
        rh=wb.add_format({"bold": True, "align": "left", "border": 1}),
        num=wb.add_format({"num_format": "#,##0.###", "align": "right", "border": 1}),
        txt=wb.add_format({"align": "right", "border": 1}),
    )
    ws = wb.add_worksheet("table")
    widths = {0: 30}
    r = 0
    ws.write(r, 0, t["title"] or "table", F["title"]); r += 1
    ws.write(r, 0, f"{t['canonical_name']} · {t['year']} · {t['subject']}/{t['measure_class']} · {t['status']}", F["meta"]); r += 1
    if t["review_note"]:
        ws.write(r, 0, "note: " + t["review_note"], F["meta"]); r += 1
    r += 1
    first_data_row = None
    for m in matrices:
        if m["section"]:
            ws.write(r, 0, m["section"], F["sec"]); r += 1
        if m["has_groups"]:
            ws.write(r, 0, "", F["grp"]); c = 1
            for g, span in m["groupspans"]:
                if span > 1:
                    ws.merge_range(r, c, r, c + span - 1, g, F["grp"])
                else:
                    ws.write(r, c, g, F["grp"])
                c += span
            r += 1
        ws.write(r, 0, "", F["hdr"])
        for i, col in enumerate(m["cols"]):
            lab = col[1] or col[0]
            ws.write(r, 1 + i, lab, F["hdr"])
            widths[1 + i] = max(widths.get(1 + i, 9), len(str(lab)) + 2)
        r += 1
        if first_data_row is None:
            first_data_row = r
        for rk in m["rows"]:
            ws.write(r, 0, rk, F["rh"]); widths[0] = max(widths[0], len(str(rk)) + 2)
            for i, col in enumerate(m["cols"]):
                v = _num(m["grid"].get((rk, col), ""))
                if isinstance(v, (int, float)):
                    ws.write_number(r, 1 + i, v, F["num"])
                else:
                    ws.write(r, 1 + i, "" if v is None else v, F["txt"])
            r += 1
        r += 1
    for c, w in widths.items():
        ws.set_column(c, c, min(max(w, 9), 42))
    ws.freeze_panes(first_data_row or 4, 1)
    wb.close(); bio.seek(0)
    return bio.read()


def _table_and_matrices(conn, tid):
    t = conn.execute("SELECT * FROM tables WHERE table_id=?", (tid,)).fetchone()
    if not t:
        return None, None
    facts = conn.execute("SELECT * FROM facts WHERE table_id=? ORDER BY rowid", (tid,)).fetchall()
    return t, build_matrices(facts)


@app.route("/matrix.xlsx")
def matrix_xlsx():
    t, ms = _table_and_matrices(db(), request.args.get("tid", ""))
    if not t:
        abort(404)
    return Response(build_table_xlsx(t, ms),
                    mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    headers={"Content-Disposition": f'attachment; filename="{safe_filename(t)}.xlsx"'})


@app.route("/matrix.csv")
def matrix_csv():
    t, ms = _table_and_matrices(db(), request.args.get("tid", ""))
    if not t:
        abort(404)
    return Response(build_table_csv(t, ms), mimetype="text/csv",
                    headers={"Content-Disposition": f'attachment; filename="{safe_filename(t)}.csv"'})


@app.route("/matrices.zip")
def matrices_zip():
    fmt = "xlsx" if request.args.get("fmt") == "xlsx" else "csv"
    where, params = where_clause(request.args)
    conn = sqlite3.connect(DB); conn.row_factory = sqlite3.Row
    try:
        tids = [r["table_id"] for r in conn.execute(
            f"SELECT DISTINCT table_id FROM facts WHERE {where}", params).fetchall()]
        CAP = 600
        capped, tids = len(tids) > CAP, tids[:CAP]
        bio = io.BytesIO()
        with zipfile.ZipFile(bio, "w", zipfile.ZIP_DEFLATED) as z:
            seen = set()
            for tid in tids:
                t, ms = _table_and_matrices(conn, tid)
                nm = safe_filename(t); base = nm; k = 2
                while nm in seen:
                    nm = f"{base}_{k}"; k += 1
                seen.add(nm)
                if fmt == "xlsx":
                    z.writestr(f"{nm}.xlsx", build_table_xlsx(t, ms))
                else:
                    z.writestr(f"{nm}.csv", build_table_csv(t, ms))
            if capped:
                z.writestr("_NOTE.txt", f"Capped at {CAP} tables; this node has more — narrow the selection.")
    finally:
        conn.close()
    bio.seek(0)
    return Response(bio.read(), mimetype="application/zip",
                    headers={"Content-Disposition": f'attachment; filename="census_matrices_{fmt}.zip"'})


def _stream_csv(where, params, fname="census_facts_filtered.csv"):
    # own connection: a streamed generator outlives the request's teardown-managed db()
    def gen():
        conn = sqlite3.connect(DB)
        try:
            cur = conn.execute(f"SELECT {','.join(FACT_COLS)} FROM facts WHERE {where} ORDER BY year, canonical_name, table_id", params)
            buf = io.StringIO(); w = csv.writer(buf)
            w.writerow(FACT_COLS); yield buf.getvalue(); buf.seek(0); buf.truncate(0)
            for row in cur:
                w.writerow(list(row)); yield buf.getvalue(); buf.seek(0); buf.truncate(0)
        finally:
            conn.close()
    return Response(gen(), mimetype="text/csv",
                    headers={"Content-Disposition": f"attachment; filename={fname}"})


@app.route("/facts.csv")
def facts_csv():
    if request.args.get("table_id"):  # single-table export from the detail page
        return _stream_csv("table_id=?", [request.args.get("table_id")], "census_table_facts.csv")
    where, params = where_clause(request.args)
    return _stream_csv(where, params)


if __name__ == "__main__":
    app.run(debug=False, port=5000)
