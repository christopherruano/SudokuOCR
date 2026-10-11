#!/usr/bin/env python3
"""Local review page for column-structure flags: census_site/review_columns.html.

One card per flagged table: why it was flagged, printed vs digitized column counts, the
digitized column headers, the scan(s), and a link to the full table page. Verdict buttons
(real misread / wrong scan / fine) are kept in the browser and exported as CSV.
Not linked from the site; noindex.
"""
import os, re, csv, html, json, sqlite3, hashlib, glob, collections
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, "..")
DB = os.path.join(ROOT, "export_final/sendout_final/sendout_v1/census_explorer.sqlite")
SITE = os.path.join(ROOT, "census_site"); OUT = os.path.join(HERE, "out")


def load(name):
    p = os.path.join(OUT, name)
    return list(csv.DictReader(open(p))) if os.path.exists(p) else []


def main():
    flags = collections.defaultdict(list)
    counts = {}
    fixlog = os.path.join(OUT, "column_fix_log.csv")
    if os.path.exists(fixlog):     # after the correction pass: review only what is still uncorrected
        for r in csv.DictReader(open(fixlog)):
            if r["decision"] in ("accept", "withdrawn"):
                continue
            g = r["failed_gates"]
            why = ("no second reading could be obtained" if r["decision"] in ("held", "error") else
                   "Persons column could not be verified as Males + Females" if r["check_verdict"] == "persons_only_missing" else
                   "the independent readings disagree on more than 10% of cells" if "two_readings_agree" in g else
                   "the column check still fails on the new reading" if "check" in g else
                   "printed totals do not add up" if "sums" in g else g)
            flags[r["page"]].append(f"{r['check_verdict'].replace('_', ' ')}: not corrected because {why}")
    else:                          # before it: every detector flag
        for r in load("column_count_check.csv"):
            if r["verdict"]:
                flags[r["page"]].append(f"{r['verdict']}: printed {r['printed_per_page']} per page, digitized {r['digitized']}")
        for r in load("column_consistency.csv"):
            flags[r["page"]] += r["issues"].split(" || ")
        for r in load("column_audit_detail.csv"):
            if r["reason"].split(" ")[0] in ("frozen", "duplicate", "dead") and r["level"] == "col":
                flags[r["page"]].append(f"{r['reason'].split(' (')[0]} column: {r['column_or_row'][-60:]}")
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    tid_of = {hashlib.md5(t.encode()).hexdigest()[:7]: t for (t,) in con.execute("select table_id from tables")}
    meta = {t: (fam, title) for t, fam, title in con.execute("select table_id, table_family, title from tables")}
    cards = []
    for page, why in flags.items():
        tid = tid_of.get(page.rsplit("_", 1)[1])
        f = os.path.join(SITE, "tables", page + ".html")
        if not tid or not os.path.exists(f):
            continue
        imgs = re.findall(r'src="\.\./img/([^"]+)"', open(f, errors="ignore").read())
        cols = list(dict.fromkeys((g or "", s or "") for g, s in con.execute(
            "select column_group_std, sub_column_std from facts where table_id=?", (tid,))))
        fam, title = meta.get(tid, ("", ""))
        age = "age" in (fam or "") or "age" in (title or "").lower()
        sev = (0 if any("not read" in w or "invented" in w or "kinds differ" in w for w in why) else 1, 0 if age else 1)
        cards.append((sev, page, title, fam, why, cols, imgs))
    cards.sort(key=lambda c: (c[0], c[1]))
    e = html.escape
    body = []
    for k, (sev, page, title, fam, why, cols, imgs) in enumerate(cards):
        colhtml = "".join(f"<li>{e(g)}{' · ' + e(s) if s and s != g else ''}</li>" for g, s in cols[:60])
        more = f"<li>… {len(cols) - 60} more</li>" if len(cols) > 60 else ""
        scans = "".join(f'<a href="img/{e(i)}" target="_blank"><img loading="lazy" src="img/{e(i)}"></a>' for i in imgs[:4])
        body.append(f'''<section class="card" data-page="{e(page)}">
<header><span class="n">{k + 1}</span><a href="tables/{e(page)}.html" target="_blank">{e(title or page)}</a>
<span class="pg">{e(page)}</span>{'<span class="tag">age</span>' if sev[1] == 0 else ''}</header>
<ul class="why">{''.join(f'<li>{e(w)}</li>' for w in why)}</ul>
<div class="body"><div class="scans">{scans}</div>
<div class="cols"><b>{len(cols)} digitized columns</b><ol>{colhtml}{more}</ol></div></div>
<div class="v"><button data-v="misread">Columns misread</button><button data-v="wrongscan">Wrong scan</button>
<button data-v="fine">Fine</button><button data-v="unsure">Unsure</button><input placeholder="note"></div></section>''')
    page = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="robots" content="noindex"><title>Column Review</title><style>
:root{{--bg:#f6f4ef;--card:#fff;--ink:#222;--mute:#666;--line:#ddd;--acc:#8a3b12;--ok:#2e7d32;--bad:#b3261e;--warn:#a06800}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{--bg:#16171a;--card:#202226;--ink:#e8e6e1;--mute:#9a9a9a;--line:#33363b;--acc:#e0915f}}}}
:root[data-theme="dark"]{{--bg:#16171a;--card:#202226;--ink:#e8e6e1;--mute:#9a9a9a;--line:#33363b;--acc:#e0915f}}
body{{background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,sans-serif;margin:0;padding:16px}}
h1{{font-size:20px;margin:0 0 4px}} .top{{position:sticky;top:0;background:var(--bg);padding:8px 0;z-index:2;border-bottom:1px solid var(--line)}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:8px;margin:14px 0;padding:12px}}
.card.misread{{border-left:5px solid var(--bad)}} .card.wrongscan{{border-left:5px solid var(--warn)}} .card.fine{{border-left:5px solid var(--ok)}}
header a{{font-weight:600;color:var(--acc)}} .n{{color:var(--mute);margin-right:8px}} .pg{{display:block;color:var(--mute);font-size:12px;word-break:break-all}}
.tag{{background:var(--acc);color:#fff;border-radius:4px;padding:0 6px;font-size:11px;margin-left:6px}}
.why{{margin:8px 0;padding-left:18px;color:var(--bad)}} .body{{display:flex;gap:12px;flex-wrap:wrap}}
.scans{{flex:2 1 420px;display:flex;gap:6px;overflow-x:auto}} .scans img{{height:760px;border:1px solid var(--line);background:#fff}}
.cols{{flex:1 1 240px;max-height:760px;overflow:auto;font-size:12.5px}} .cols ol{{padding-left:22px;margin:4px 0}}
.v{{margin-top:8px;display:flex;gap:6px;flex-wrap:wrap}} button{{border:1px solid var(--line);background:var(--bg);color:var(--ink);border-radius:5px;padding:5px 10px;cursor:pointer}}
button.on{{background:var(--acc);color:#fff}} input{{flex:1;min-width:140px;padding:5px;border:1px solid var(--line);background:var(--bg);color:var(--ink);border-radius:5px}}
@media (max-width:600px){{.scans img{{height:360px}}}}
</style></head><body><div class="top"><h1>Column review: {len(cards)} flagged tables</h1>
<span id="prog"></span> · <button id="exp">Export verdicts CSV</button> · <label><input type="checkbox" id="hide" style="flex:none;min-width:0"> hide reviewed</label></div>
{''.join(body)}
<script>
const K='colreview_v1';let S={{}};try{{S=JSON.parse(localStorage.getItem(K)||'{{}}')}}catch(e){{}}
const save=()=>{{try{{localStorage.setItem(K,JSON.stringify(S))}}catch(e){{}};prog()}};
function paint(c){{const s=S[c.dataset.page]||{{}};c.className='card '+(s.v||'');c.querySelectorAll('button').forEach(b=>b.classList.toggle('on',b.dataset.v===s.v));
c.querySelector('input').value=s.note||'';c.style.display=(document.getElementById('hide').checked&&s.v)?'none':''}}
function prog(){{const n=Object.values(S).filter(s=>s.v).length;document.getElementById('prog').textContent=n+' of {len(cards)} reviewed'}}
document.querySelectorAll('.card').forEach(c=>{{c.querySelectorAll('button').forEach(b=>b.onclick=()=>{{const p=c.dataset.page;S[p]=S[p]||{{}};S[p].v=S[p].v===b.dataset.v?'':b.dataset.v;save();paint(c)}});
c.querySelector('input').onchange=ev=>{{const p=c.dataset.page;S[p]=S[p]||{{}};S[p].note=ev.target.value;save()}};paint(c)}});
document.getElementById('hide').onchange=()=>document.querySelectorAll('.card').forEach(paint);
document.getElementById('exp').onclick=()=>{{const rows=[['page','verdict','note']].concat(Object.entries(S).filter(([p,s])=>s.v||s.note).map(([p,s])=>[p,s.v||'',s.note||'']));
const csv=rows.map(r=>r.map(x=>'"'+String(x).replace(/"/g,'""')+'"').join(',')).join('\\n');const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([csv],{{type:'text/csv'}}));a.download='column_review_verdicts.csv';a.click()}};prog();
</script></body></html>'''
    open(os.path.join(SITE, "review_columns.html"), "w").write(page)
    print(f"review page: {len(cards)} flagged tables -> census_site/review_columns.html")


if __name__ == "__main__":
    main()
