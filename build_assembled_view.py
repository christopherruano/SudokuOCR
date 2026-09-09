#!/usr/bin/env python3
"""Build a browsable ASSEMBLED Table VII view for the static site, from export/tableVII_runs.json.
Each run -> one combined table (member-page sections stacked, with the correct grouped/nested rendering
that matches the static site), plus the page range and member scan thumbnails. Honest labels: shows the
PDF page span and that it's auto-assembled (may be partial). Writes:
  census_site/assembled.html            index (by region -> year)
  census_site/assembled/<key>.html      one combined table per run
and adds a homepage link. No reingest needed.
  python3 build_assembled_view.py
"""
import json, os, re, html, glob, hashlib, csv, datetime

STAMP = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
ROOT = os.path.dirname(os.path.abspath(__file__))
SITE = os.path.join(ROOT, "census_site")
CORPUS = os.path.join(ROOT, "export/ocr_results_referenced/digitized_tables")
RUNS = os.path.join(ROOT, "export/tableVII_runs.json")
VERD = os.path.join(ROOT, "export/review_verdicts/assembly.json")
esc = lambda x: html.escape(str(x))


def load_member(table_id):
    tid = table_id.replace("export/ocr_results_referenced/", "").strip("/")
    jf = os.path.join(ROOT, "export/ocr_results_referenced", tid + ".json")
    if not os.path.exists(jf):
        hit = glob.glob(os.path.join(CORPUS, "*", os.path.basename(tid) + ".json"))
        jf = hit[0] if hit else None
    if not jf or not os.path.exists(jf):
        return None, None
    try:
        d = json.load(open(jf))
    except Exception:
        return None, None
    ih = re.search(r"([0-9a-f]{40})_oneshot", os.path.basename(jf))
    return d, (ih.group(1) if ih else None)


def section_table(name, rows, limit=200):
    if not rows:
        return ""
    scalar, dictc = [], []
    for r in rows:
        for k, v in r.items():
            if isinstance(v, dict):
                if k not in dictc:
                    dictc.append(k)
            elif k not in scalar:
                scalar.append(k)
    rh = scalar[0] if scalar else None
    groups = [(sc, [None]) for sc in scalar[1:]]
    for dc in dictc:
        subs = []
        for r in rows:
            v = r.get(dc)
            if isinstance(v, dict):
                for sk in v:
                    if sk not in subs:
                        subs.append(sk)
        groups.append((dc, subs or [None]))
    has = bool(dictc)
    out = [f"<h4>{esc(name)}</h4><table class=mx>"]
    if has:
        out.append("<thead><tr><th rowspan=2 class=rh></th>")
        for g, subs in groups:
            out.append(f"<th class=grp colspan={len(subs)}>{esc(g)}</th>")
        out.append("</tr><tr>")
        for g, subs in groups:
            for sk in subs:
                out.append(f"<th>{esc(sk) if sk else esc(g)}</th>")
        out.append("</tr></thead>")
    else:
        out.append("<thead><tr><th class=rh></th>" + "".join(f"<th>{esc(g)}</th>" for g, _ in groups) + "</tr></thead>")
    out.append("<tbody>")
    for r in rows[:limit]:
        out.append(f"<tr><td class=rh>{esc(r.get(rh, '')) if rh else ''}</td>")
        for g, subs in groups:
            v = r.get(g)
            for sk in subs:
                out.append(f"<td>{esc(v.get(sk, '') if isinstance(v, dict) else (v if sk is None else ''))}</td>")
        out.append("</tr>")
    out.append("</tbody></table>")
    return "".join(out)


CSS = """body{font:14px/1.5 system-ui;margin:0;background:#fbfdff;color:#1a2233}
header{background:#1f3864;color:#fff;padding:12px 20px}header a{color:#cfe0ff}
h1{margin:0;font-size:18px}.sub{opacity:.9;font-size:12px}
h4{margin:10px 0 2px;color:#1f3864}
table.mx{border-collapse:collapse;margin-bottom:12px;font:12px/1.3 Menlo,Consolas,monospace}
table.mx th,table.mx td{border:1px solid #d6dde2;padding:2px 7px;text-align:right;white-space:nowrap}
table.mx th{background:#eef3f7;text-align:center}
table.mx td.rh,table.mx th.rh{text-align:left;background:#fafbfc;font-weight:600;white-space:normal;max-width:220px;position:sticky;left:0}
table.mx thead .grp{background:#dde7ee}
.scans{display:flex;gap:8px;overflow-x:auto;padding:6px 0}.scans img{height:130px;border:1px solid #ccd;border-radius:5px}
.sec{margin:14px 20px 4px;font-weight:700;color:#1f3864;border-bottom:2px solid #e3e9f5}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:8px;margin:6px 20px}
.c{display:block;background:#fff;border:1px solid #e2e6ee;border-left:4px solid #1f3864;border-radius:7px;padding:9px 11px;text-decoration:none;color:inherit}
.c:hover{box-shadow:0 2px 8px rgba(0,0,0,.08)}.c .t{font-weight:600;font-size:13px}.c .m{color:#667;font-size:12px}"""


def main():
    runs = json.load(open(RUNS))
    verds = json.load(open(VERD)) if os.path.exists(VERD) else {}
    # honest data-quality labels from the first-cut catalog (verified/minor-issues/needs-review/incomplete)
    qmap = {}
    _catp = os.path.join(ROOT, "export/first_cut/catalog.csv")
    if os.path.exists(_catp):
        for cr in csv.DictReader(open(_catp)):
            qmap[(cr["doc"], cr["page_range"])] = cr["quality"]
    QBADGE = {"verified": ("✓ verified", "#0a5d1e"), "minor-issues": ("~ minor issues", "#b8860b"),
              "needs-review": ("⚠ needs review", "#9c0006"), "incomplete": ("… incomplete", "#777")}
    os.makedirs(os.path.join(SITE, "assembled"), exist_ok=True)
    by_region = {}
    built = 0
    for run in runs:
        key = re.sub(r"[^A-Za-z0-9]+", "_", f"{run['saoa']}_{run['page_range'][0]}_{run['page_range'][1]}")
        vkey = f"{run['saoa']}:{run['page_range']}"
        vd = verds.get(vkey, {})
        vv = (vd.get("v") if isinstance(vd, dict) else vd) or ""
        title = ""
        scans, body, total = [], [], 0
        for m in run["members"]:
            d, ih = load_member(m["table_id"])
            if d and not title:
                title = (d.get("metadata", {}) or {}).get("title", "") or ""
            if ih and os.path.exists(os.path.join(SITE, "img", ih + ".jpg")):
                scans.append(f"<a href='img/{ih}.jpg' target=_blank><img loading=lazy src='img/{ih}.jpg'></a>")
            for s in (d or {}).get("data", []):
                if s.get("rows"):
                    total += len(s["rows"])
                    body.append(section_table(f"p{m['page']} · {s.get('name', '')}", s["rows"]))
        region = run.get("region", "") or "(unspecified)"
        year = run.get("year", "")
        quality = qmap.get((str(run["saoa"]), f"{run['page_range'][0]}-{run['page_range'][1]}"), "")
        qlabel, qcol = QBADGE.get(quality, ("", "#777"))
        qnote = f" · <b style='color:{qcol}'>{qlabel}</b>" if qlabel else ""
        note = {"wrong": "⚠ flagged: boundaries need extension", "correct": "✓ reviewed",
                "boundary": "⚠ boundary off"}.get(vv, "")
        page = (f"<!doctype html><meta charset=utf-8><title>{esc(title)[:60]}</title><style>{CSS}</style>"
                f"<header><a href='../assembled.html'>← assembled tables</a> &nbsp; "
                f"<b>{esc(title) or 'Table VII'}</b><div class=sub>{esc(region)} · {esc(year)} · "
                f"PDF pages {run['page_range'][0]}–{run['page_range'][1]} ({run['n_pages']} pages, {total} rows) · "
                f"auto-assembled (may be partial){qnote} {note} · <span style='opacity:.7'>updated {STAMP}</span></div></header>"
                f"<div style='padding:6px 20px'><b>Member pages:</b><div class=scans>{''.join(scans)}</div></div>"
                f"<div style='padding:6px 20px'>{''.join(body) or '<i>no extractable rows</i>'}</div>")
        open(os.path.join(SITE, "assembled", key + ".html"), "w").write(page)
        by_region.setdefault(region, {}).setdefault(str(year), []).append(
            (key, title[:60], run['page_range'], run['n_pages'], total, vv, quality))
        built += 1

    # index
    out = [f"<!doctype html><meta charset=utf-8><title>Assembled Table VII</title><style>{CSS}</style>",
           "<header><h1>Assembled Table VII — combined multi-page tables</h1>"
           f"<div class=sub>{built} multi-page Age/Sex/Civil-Condition tables, stitched from their pages · "
           f"auto-assembled (boundaries still being refined) · <b>Last updated {STAMP}</b> · "
           "<a href='index.html'>← explorer</a></div></header>"]
    for region in sorted(by_region):
        out.append(f"<div class=sec>{esc(region)}</div><div class=grid>")
        for year in sorted(by_region[region]):
            for key, t, pr, n, rows, vv, quality in sorted(by_region[region][year]):
                ql, qc = QBADGE.get(quality, ("", "#777"))
                badge = f"<span style='color:{qc};font-size:11px'>{ql}</span>" if ql else ""
                out.append(f"<a class=c href='assembled/{key}.html'><div class=t>{esc(t) or 'Table VII'} {badge}</div>"
                           f"<div class=m>{esc(year)} · pp{pr[0]}–{pr[1]} · {n} pages · {rows} rows</div></a>")
        out.append("</div>")
    open(os.path.join(SITE, "assembled.html"), "w").write("\n".join(out))

    # homepage link
    ip = os.path.join(SITE, "index.html")
    if os.path.exists(ip):
        ih = open(ip, encoding="utf-8").read()
        ih = re.sub(r"<!--ASMVIEW-->.*?<!--/ASMVIEW-->", "", ih, flags=re.S)
        link = ('<!--ASMVIEW--><div style="margin:6px 16px;padding:8px 14px;background:#2c4a7c;color:#fff;border-radius:8px">'
                f'<b>Assembled Table VII</b> &middot; {built} multi-page tables stitched together. '
                '<a href="assembled.html" style="color:#ffd966;font-weight:700;text-decoration:none">Open assembled &rarr;</a></div><!--/ASMVIEW-->')
        ih = ih.replace("</header>", "</header>" + link, 1) if "</header>" in ih else ih + link
        open(ip, "w", encoding="utf-8").write(ih)
    print(f"assembled view: {built} combined tables -> census_site/assembled.html (+ per-run pages)")


if __name__ == "__main__":
    main()
