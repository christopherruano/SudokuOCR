#!/usr/bin/env python3
"""Localhost review app for the core-money constraint-fails.
Click through each failing page: scan on the left, the pinpointed failing check +
the DETERMINISTIC FIX (trust-the-parts) on the right, and four verdict buttons.
Verdicts persist to core_money_verdicts.json so they can be applied later.

  python3 money_review_app.py      ->  http://localhost:5057
"""
import json, csv, re, os, copy, html
from flask import Flask, send_from_directory, request, jsonify, Response

V1 = "export_final/sendout_final/sendout_v1"
IMGDIR = os.path.expanduser("~/Downloads/final_tableVII")
VERDICTS = os.path.join(V1, "core_money_verdicts.json")
APPLIED = os.path.join(V1, "core_money_applied_fixes.json")
CORRDIR = "export/money_gapfill_ocr/corrected"
FIXRE = re.compile(r"(-?\d+)\s*!=\s*sum\(.*\)\s*=\s*(-?\d+)")
app = Flask(__name__)


def det_fix(detail):
    m = FIXRE.search(detail or "")
    return (int(m.group(1)), int(m.group(2))) if m else None


def _to_int(x):
    try:
        return int(str(x).replace(",", ""))
    except Exception:
        return None


def _get(row, group, col):
    cell = row.get(group)
    return cell.get(col) if isinstance(cell, dict) else cell


def _set(row, group, col, val):
    cell = row.get(group)
    if isinstance(cell, dict) and col in cell:
        cell[col] = val; return True
    if group in row and not isinstance(cell, dict):
        row[group] = val; return True
    return False


def _total_row(rows):
    for i, row in enumerate(rows):
        if any(isinstance(v, str) and re.search(r"\btotal\b", v, re.I) for v in row.values()):
            return i
    return None


def is_num(v):
    return isinstance(v, (int, float)) or (isinstance(v, str) and v.replace(",", "").lstrip("-").isdigit())


def build_columns(d):
    """ordered [(group, sub)] across the table; sub=None for a scalar column group."""
    data = d.get("data", []); order = d.get("metadata", {}).get("column_groups") or []
    subs, groups = {}, []
    for sec in data:
        for row in sec.get("rows", []):
            for k, v in row.items():
                if isinstance(v, dict):
                    subs.setdefault(k, [])
                    for s in v:
                        if s not in subs[k]:
                            subs[k].append(s)
                    if k not in groups:
                        groups.append(k)
                elif is_num(v):
                    subs.setdefault(k, [None])
                    if k not in groups:
                        groups.append(k)
    groups = [g for g in order if g in subs] + [g for g in groups if g not in order]
    return [(g, s) for g in groups for s in subs[g]]


def row_label(row, md):
    rl = md.get("row_label_name")
    if rl and isinstance(row.get(rl), str):
        return row[rl]
    return next((v for v in row.values() if isinstance(v, str) and v.strip()), "")


def apply_fixes(data, fails):
    """Set each failing aggregate = sum of its parts, in a deep copy. Locate the cell by
    matching the wrong value X at (group,col), preferring a 'Total' row; only act on a
    confident (total-row or unique) match. Returns (corrected_data, applied, unlocated)."""
    data = copy.deepcopy(data)
    applied, unlocated = [], []
    for f in fails:
        fx = det_fix(f.get("detail", ""))
        if not fx:
            unlocated.append(f.get("detail", "")); continue
        X, Y = fx; g, c = f.get("group", ""), f.get("col", "")
        done = False
        for sec in data:
            rows = sec.get("rows", [])
            ti = _total_row(rows)
            matches = [i for i in range(len(rows)) if _to_int(_get(rows[i], g, c)) == X]
            pick = ti if (ti is not None and ti in matches) else (matches[0] if len(matches) == 1 else None)
            if pick is not None and _set(rows[pick], g, c, Y):
                applied.append({"group": g, "col": c, "old": X, "new": Y,
                                "row": pick, "via": "total_row" if pick == ti else "unique_value"})
                done = True; break
        if not done:
            unlocated.append(f.get("detail", ""))
    return data, applied, unlocated


def load_items():
    rows = [r for r in csv.DictReader(open(f"{V1}/core_money_ocr_results.csv"))
            if r["category"] == "review_constraints"]
    items = []
    for r in rows:
        d = json.load(open(r["json"]))
        fails = d.get("constraints", {}).get("failures", []) or []
        items.append({"key": f"{r['saoa']}_p{r['pdf_page']}", "saoa": r["saoa"], "page": r["pdf_page"],
                      "n": len(fails), "title": d.get("metadata", {}).get("title", "")[:90], "fails": fails})
    items.sort(key=lambda x: (x["n"], x["saoa"]))
    return items


def verdicts():
    return json.load(open(VERDICTS)) if os.path.exists(VERDICTS) else {}


@app.route("/img/<name>")
def img(name):
    return send_from_directory(IMGDIR, name)


@app.route("/verdict", methods=["POST"])
def verdict():
    v = verdicts(); j = request.get_json()
    v[j["key"]] = j["verdict"]
    json.dump(v, open(VERDICTS, "w"), indent=1)
    return jsonify(ok=True, n=len(v))


@app.route("/apply_fix", methods=["POST"])
def apply_fix():
    key = request.get_json()["key"]
    # find the source OCR json for this page
    src = None
    for r in csv.DictReader(open(f"{V1}/core_money_ocr_results.csv")):
        if f"{r['saoa']}_p{r['pdf_page']}" == key:
            src = r["json"]; break
    if not src:
        return jsonify(error="page not found"), 404
    d = json.load(open(src))
    fails = d.get("constraints", {}).get("failures", []) or []
    cdata, applied, unlocated = apply_fixes(d.get("data", []), fails)
    d["data"] = cdata
    d["corrections_applied"] = applied
    os.makedirs(CORRDIR, exist_ok=True)
    json.dump(d, open(f"{CORRDIR}/{key}_oneshot.json", "w"))
    # log the exact corrections + set the verdict
    log = json.load(open(APPLIED)) if os.path.exists(APPLIED) else {}
    log[key] = {"applied": applied, "unlocated": unlocated, "source": src}
    json.dump(log, open(APPLIED, "w"), indent=1)
    v = verdicts(); v[key] = f"applied {len(applied)} fix(es)" + (f", {len(unlocated)} unlocated" if unlocated else "")
    json.dump(v, open(VERDICTS, "w"), indent=1)
    return jsonify(applied=len(applied), unlocated=len(unlocated),
                   changes=[f"{a['group']}/{a['col']}: {a['old']:,}→{a['new']:,}" for a in applied])


def _src_for(key):
    for r in csv.DictReader(open(f"{V1}/core_money_ocr_results.csv")):
        if f"{r['saoa']}_p{r['pdf_page']}" == key:
            return r["json"]
    return None


def _load_working(key):
    corr = f"{CORRDIR}/{key}_oneshot.json"
    if os.path.exists(corr):
        return json.load(open(corr))
    src = _src_for(key)
    return json.load(open(src)) if src else None


@app.route("/edit", methods=["POST"])
def edit():
    j = request.get_json(); key = j["key"]
    d = _load_working(key)
    if not d:
        return jsonify(error="not found"), 404
    sec, ri, g, s = int(j["sec"]), int(j["row"]), j["g"], j.get("s") or ""
    val = _to_int(j["value"])
    val = val if val is not None else j["value"]
    row = d["data"][sec]["rows"][ri]
    if s and isinstance(row.get(g), dict):
        row[g][s] = val
    else:
        row[g] = val
    os.makedirs(CORRDIR, exist_ok=True)
    json.dump(d, open(f"{CORRDIR}/{key}_oneshot.json", "w"))
    return jsonify(ok=True)


@app.route("/page/<key>")
def page(key):
    d = _load_working(key)
    if not d:
        return "not found", 404
    md = d.get("metadata", {})
    cols = build_columns(d)
    fails = d.get("constraints", {}).get("failures", []) or []
    failcols = {(f.get("group", ""), f.get("col", "") or "") for f in fails}
    keys = [f"{r['saoa']}_p{r['pdf_page']}" for r in
            sorted([r for r in csv.DictReader(open(f"{V1}/core_money_ocr_results.csv")) if r["category"] == "review_constraints"],
                   key=lambda r: int(r["json"] and 0) or 0)]  # order not critical for nav
    try:
        idx = keys.index(key); prv = keys[idx-1] if idx > 0 else ""; nxt = keys[idx+1] if idx < len(keys)-1 else ""
    except ValueError:
        prv = nxt = ""

    # header row
    th = ['<th class="lbl">row</th>']
    for g, s in cols:
        cls = "fc" if (g, s or "") in failcols else ""
        th.append(f'<th class="{cls}"><div class=g>{html.escape(str(g))}</div><div class=s>{html.escape(str(s or ""))}</div></th>')
    body = []
    for si, sec in enumerate(d.get("data", [])):
        rows = sec.get("rows", [])
        ti = _total_row(rows)
        if sec.get("name"):
            body.append(f'<tr class=sec><td colspan="{len(cols)+1}">{html.escape(str(sec["name"]))}</td></tr>')
        for ri, row in enumerate(rows):
            tot = (ri == ti)
            tds = [f'<td class="lbl">{html.escape(str(row_label(row, md)))}</td>']
            for g, s in cols:
                cell = row.get(g)
                val = cell.get(s) if (isinstance(cell, dict) and s) else (cell if not isinstance(cell, dict) else "")
                val = "" if val is None else val
                col = f"{g}||{s or ''}"
                tds.append(f'<td><input value="{html.escape(str(val))}" data-col="{html.escape(col)}" '
                           f'data-sec="{si}" data-row="{ri}" data-g="{html.escape(str(g))}" data-s="{html.escape(str(s or ""))}" '
                           f'oninput="cell(this)"></td>')
            body.append(f'<tr class="{ "total" if tot else "" }" data-total="{1 if tot else 0}">{"".join(tds)}</tr>')

    fchk = "<br>".join(html.escape(f"{f.get('group','')}/{f.get('col','')}: {f.get('detail','')}") for f in fails[:6])
    page_html = f"""<!doctype html><meta charset=utf-8><title>{key} — matrix</title>
    <style>
    body{{font:13px/1.4 -apple-system,Segoe UI,Roboto,sans-serif;margin:0;background:#f4f5f7}}
    header{{position:sticky;top:0;background:#1f3864;color:#fff;padding:8px 14px;z-index:10;display:flex;gap:14px;align-items:center}}
    header a{{color:#cfe0ff}} .wrap{{display:flex;gap:14px;padding:12px;align-items:flex-start}}
    .scan{{position:sticky;top:54px;flex:0 0 540px}}
    .ctrls{{display:flex;gap:6px;margin-bottom:4px}}
    .ctrls button,.ctrls a{{padding:3px 9px;border:0;border-radius:4px;background:#2f5496;color:#fff;cursor:pointer;font-size:12px;text-decoration:none}}
    .viewport{{width:540px;height:760px;overflow:hidden;border:1px solid #ccc;border-radius:4px;background:#333;cursor:grab}}
    #scanimg{{width:540px;transform-origin:center center;user-select:none;-webkit-user-drag:none}}
    .grid{{flex:1;overflow:auto;background:#fff;border:1px solid #ddd;border-radius:6px;padding:6px}}
    table{{border-collapse:collapse}} th,td{{border:1px solid #e3e3e3;padding:0;text-align:center}}
    th{{background:#eef1f7;font-size:10px;position:sticky;top:0}} th.fc{{background:#ffd9dd}}
    th .g{{font-weight:700}} th .s{{color:#666;font-weight:400}}
    td.lbl,th.lbl{{text-align:left;font-size:11px;padding:2px 5px;max-width:150px;background:#fafafa;position:sticky;left:0}}
    tr.sec td{{background:#1f3864;color:#fff;text-align:left;font-size:11px;padding:3px 6px;font-weight:700}}
    tr.total input{{font-weight:700;background:#fffceb}}
    input{{width:56px;border:0;padding:3px 2px;font:12px ui-monospace,monospace;text-align:right}}
    input:focus{{outline:2px solid #2f5496;background:#eef}}
    td.bad input{{background:#ffd9dd}} td.ok input{{background:#e8f5e9}}
    .fail{{font-size:11px;color:#9c0017;margin-left:auto;max-width:50%;text-align:right}}
    </style>
    <header><b>{key}</b>
      <span>{html.escape(md.get('title','')[:60])}</span>
      {'<a href="/page/'+prv+'">&larr; prev</a>' if prv else ''}
      <a href="/">list</a>
      {'<a href="/page/'+nxt+'">next &rarr;</a>' if nxt else ''}
      <span class=fail>failing: {fchk}</span>
    </header>
    <div class=wrap>
      <div class=scan>
        <div class=ctrls>
          <button onclick="zoomb(-.3)">&minus; zoom</button><button onclick="zoomb(.3)">+ zoom</button>
          <button onclick="rot(90)">&#8635; rotate</button><button onclick="rstv()">reset</button>
          <a href="/img/{key}.png" target=_blank>open full</a>
        </div>
        <div class=viewport id=vp><img id=scanimg src="/img/{key}.png"></div>
      </div>
      <div class=grid><table><thead><tr>{''.join(th)}</tr></thead><tbody>{''.join(body)}</tbody></table></div>
    </div>
    <script>
    const KEY="{key}";
    // --- scan zoom / pan / rotate ---
    let z=1,rr=0,tx=0,ty=0,drag=false,sx=0,sy=0;
    const im=document.getElementById('scanimg'),vp=document.getElementById('vp');
    function av(){{im.style.transform=`translate(${{tx}}px,${{ty}}px) scale(${{z}}) rotate(${{rr}}deg)`;}}
    function zoomb(d){{z=Math.max(.2,Math.min(8,z+d));av();}}
    function rot(d){{rr=(rr+d)%360;av();}}
    function rstv(){{z=1;rr=0;tx=0;ty=0;av();}}
    vp.addEventListener('wheel',e=>{{e.preventDefault();zoomb(e.deltaY<0?.25:-.25);}},{{passive:false}});
    vp.addEventListener('mousedown',e=>{{drag=true;sx=e.clientX-tx;sy=e.clientY-ty;vp.style.cursor='grabbing';}});
    window.addEventListener('mousemove',e=>{{if(drag){{tx=e.clientX-sx;ty=e.clientY-sy;av();}}}});
    window.addEventListener('mouseup',()=>{{drag=false;vp.style.cursor='grab';}});
    function num(s){{ s=(s||'').toString().replace(/,/g,''); let n=parseInt(s); return isNaN(n)?null:n; }}
    function balance(){{
      // for each column, sum non-total rows and compare to the total-row cell
      const cols={{}};
      document.querySelectorAll('input[data-col]').forEach(i=>{{
        const c=i.dataset.col, tot=i.closest('tr').dataset.total==='1';
        (cols[c]=cols[c]||{{sum:0,tcell:null}});
        if(tot) cols[c].tcell=i; else {{const v=num(i.value); if(v!==null) cols[c].sum+=v;}}
      }});
      for(const c in cols){{const t=cols[c].tcell; if(!t) continue;
        const tv=num(t.value); const td=t.closest('td');
        td.classList.remove('ok','bad');
        if(tv!==null) td.classList.add(tv===cols[c].sum?'ok':'bad');
      }}
    }}
    async function cell(i){{
      await fetch('/edit',{{method:'POST',headers:{{'Content-Type':'application/json'}},
        body:JSON.stringify({{key:KEY,sec:i.dataset.sec,row:i.dataset.row,g:i.dataset.g,s:i.dataset.s,value:i.value}})}});
      balance();
    }}
    balance();
    </script>"""
    return Response(page_html, mimetype="text/html")


@app.route("/")
def index():
    items = load_items(); v = verdicts()
    n_easy = sum(1 for x in items if x["n"] <= 4)
    cards = []
    for x in items:
        reocr = x["n"] > 30
        cur_v = v.get(x["key"], "")
        chk = []
        for f in x["fails"][:8]:
            where = f"{f.get('group','')}/{f.get('col','')}".strip("/")
            fx = det_fix(f.get("detail", ""))
            fixhtml = ""
            if fx:
                cur, parts = fx
                fixhtml = (f"<div class=fix>deterministic: set total <b>{cur:,} &rarr; {parts:,}</b> "
                           f"(&Delta; {parts-cur:+,})</div>"
                           f"<div class=alt>&hellip;or a cell in this column is off by {abs(parts-cur):,}</div>")
            chk.append(f"<div class=chk><span class=where>{where}</span> "
                       f"<span class=detail>{f.get('detail','')}</span>{fixhtml}</div>")
        tag = "easy" if x["n"] <= 4 else ("mid" if x["n"] <= 30 else "bad")
        cards.append(f"""
        <div class="card {'reocr' if reocr else ''} {'done' if cur_v else ''}" id="c_{x['key']}" data-key="{x['key']}">
          <div class=scan><a href="/img/{x['key']}.png" target=_blank><img loading=lazy src="/img/{x['key']}.png"></a></div>
          <div class=info>
            <h3>{x['key']} <span class="tag {tag}">{x['n']} fail</span>{' &mdash; RE-OCR' if reocr else ''}
                <a href="/page/{x['key']}" style="margin-left:8px;font-size:12px">open matrix editor &rarr;</a>
                <span class=vtag id="v_{x['key']}">{cur_v}</span></h3>
            <div class=ttl>{x['title']}</div>
            {''.join(chk)}
            <div class=btns>
              <button onclick="applyFix('{x['key']}')">total wrong &rarr; apply fix</button>
              <button onclick="vote('{x['key']}','cell_wrong')">a data cell is wrong</button>
              <button onclick="vote('{x['key']}','reocr')">re-OCR</button>
              <button onclick="vote('{x['key']}','skip')">skip</button>
            </div>
          </div>
        </div>""")
    html = f"""<!doctype html><meta charset=utf-8><title>Core money review</title>
    <style>
    body{{font:13px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;margin:0;background:#f4f5f7}}
    header{{position:sticky;top:0;background:#1f3864;color:#fff;padding:10px 16px;z-index:5}}
    .card{{display:flex;gap:16px;background:#fff;margin:10px 16px;border:1px solid #e2e4e8;border-radius:8px;padding:12px}}
    .card.reocr{{border-color:#d99;background:#fff6f6}} .card.done{{opacity:.55}}
    .scan{{flex:0 0 460px}} .scan img{{width:460px;border:1px solid #ccc;border-radius:4px}}
    .info{{flex:1}} h3{{margin:0 0 2px;font-size:14px}} .ttl{{color:#555;font-size:12px;margin-bottom:8px}}
    .chk{{margin:6px 0;padding:8px 10px;border-radius:6px;background:#fbfbfd;border:1px solid #eee}}
    .where{{font-weight:700}} .detail{{font-variant-numeric:tabular-nums}}
    .fix{{margin-top:4px;padding:5px 8px;border-radius:5px;background:#e8f5e9;border:1px solid #b6dcb9;display:inline-block}}
    .fix b{{color:#0a5d1e}} .alt{{color:#7a5b00;font-size:12px;margin-top:3px}}
    .tag{{padding:1px 7px;border-radius:10px;font-size:11px;font-weight:700}}
    .tag.easy{{background:#c6efce;color:#0a5d1e}}.tag.mid{{background:#ffeb9c;color:#7a5b00}}.tag.bad{{background:#ffc7ce;color:#9c0017}}
    .vtag{{color:#0a5d1e;font-weight:700;margin-left:6px}}
    .btns{{margin-top:8px;display:flex;gap:6px;flex-wrap:wrap}}
    .btns button{{padding:5px 10px;border:0;border-radius:5px;background:#2f5496;color:#fff;cursor:pointer}}
    .btns button:hover{{background:#3a64b0}}
    </style>
    <header><b>Core-money constraint-fail review</b> — {len(items)} pages, easiest-first ·
      {n_easy} easy (&le;4) · <span id=done>{len(v)}</span>/{len(items)} reviewed ·
      green = deterministic fix (trust the parts) · amber = a data cell is the mis-read (your call)</header>
    {''.join(cards)}
    <script>
    function mark(key,label){{
      document.getElementById('v_'+key).textContent='✓ '+label;
      document.getElementById('c_'+key).classList.add('done');
      document.getElementById('done').textContent=document.querySelectorAll('.vtag:not(:empty)').length;
    }}
    async function vote(key,verdict){{
      await fetch('/verdict',{{method:'POST',headers:{{'Content-Type':'application/json'}},
        body:JSON.stringify({{key,verdict}})}});
      mark(key,verdict);
    }}
    async function applyFix(key){{
      const r=await (await fetch('/apply_fix',{{method:'POST',headers:{{'Content-Type':'application/json'}},
        body:JSON.stringify({{key}})}})).json();
      mark(key,'applied '+r.applied+(r.unlocated?(' ('+r.unlocated+' unlocated)'):'')
        +(r.changes&&r.changes.length?'  ['+r.changes.join(', ')+']':''));
    }}
    </script>"""
    return Response(html, mimetype="text/html")


if __name__ == "__main__":
    print("Review app -> http://localhost:5057")
    app.run(host="127.0.0.1", port=5057, debug=False)
