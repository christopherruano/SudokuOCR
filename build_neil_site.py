#!/usr/bin/env python3
"""Key-tables static site for the Age/Sex/Civil-Condition tables.
Reasonably-good-and-up, honest status badges, Age/Sex/Civil-Condition pinned on top,
matrix-beside-scan. Reads money_assembled.json. Self-contained HTML (scans via file://
for immediate local viewing; run with --bundle to copy downscaled scans for upload).
"""
import json, os, re, html, sys, glob
from collections import defaultdict

V1 = "export_final/sendout_final/sendout_v1"
OUT = os.path.join(V1, "neil_site.html")
BUNDLE = "--bundle" in sys.argv
DIST = os.path.join(V1, "neil_site_dist")
BIGPROV = ["madras", "bengal", "bombay", "punjab", "united provinces", "central provinces",
           "bihar", "assam", "burma", "hyderabad", "mysore", "rajputana", "baroda"]


def scan_path(saoa, page):
    for cand in (os.path.expanduser(f"~/Downloads/core_money_renders/*/{saoa}/{saoa}_p{page}.png"),
                 os.path.expanduser(f"~/Downloads/final_tableVII/{saoa}_p{page}.png")):
        hit = glob.glob(cand)
        if hit:
            return hit[0]
    return ""


def thumb(sp, saoa, page):
    """downscale a scan into the dist folder; return its relative path."""
    if not sp:
        return ""
    rel = f"scans/{saoa}_p{page}.jpg"
    dst = os.path.join(DIST, rel)
    if not os.path.exists(dst):
        try:
            from PIL import Image
            im = Image.open(sp).convert("RGB")
            im.thumbnail((1100, 1600))
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            im.save(dst, "JPEG", quality=80)
        except Exception:
            return ""
    return rel


def matrix_html(d):
    """compact read-only matrix for one page's OCR result."""
    data = d.get("data", [])
    if not data:
        return "<div class=dim>(no table extracted)</div>"
    out = []
    for sec in data[:3]:
        rows = sec.get("rows", [])
        if not rows:
            continue
        # columns = union of dict/numeric fields
        cols = []
        for r in rows:
            for k, v in r.items():
                if isinstance(v, dict):
                    for s in v:
                        if (k, s) not in cols:
                            cols.append((k, s))
                elif (isinstance(v, (int, float)) or (isinstance(v, str) and v.replace(",", "").lstrip("-").isdigit())):
                    if (k, None) not in cols:
                        cols.append((k, None))
        lblkey = next((k for k, v in rows[0].items() if isinstance(v, str)), "")
        th = "<th></th>" + "".join(f"<th>{html.escape(str(g))}<br><span class=s>{html.escape(str(s or ''))}</span></th>" for g, s in cols[:24])
        trs = []
        for r in rows[:40]:
            tds = [f"<td class=lbl>{html.escape(str(r.get(lblkey, '')))[:28]}</td>"]
            for g, s in cols[:24]:
                cell = r.get(g)
                val = cell.get(s) if (isinstance(cell, dict) and s) else (cell if not isinstance(cell, dict) else "")
                tds.append(f"<td>{html.escape(str(val if val is not None else ''))}</td>")
            trs.append("<tr>" + "".join(tds) + "</tr>")
        nm = html.escape(str(sec.get("name", ""))[:80])
        out.append(f"<div class=secname>{nm}</div><table class=mx><tr>{th}</tr>{''.join(trs)}</table>")
    return "".join(out)


BADGE = {"clean": ("verified", "ok"), "minor_issues": ("minor — a few cells flagged", "warn"),
         "ok": ("OK", "ok"), "needs_review": ("needs review", "bad")}


def main():
    tables = json.load(open(f"{V1}/money_assembled.json"))
    keep = [t for t in tables if t["status"] != "empty"]

    def yr(t):
        try:
            return int(re.search(r"(1[89]\d\d)", str(t.get("year", "")) or "").group(1))
        except Exception:
            return 9999

    def rank(t):
        # the most-relevant (Age/Sex/Civil-Condition) tables lead; then ARRANGED BY
        # PROVENANCE: province (alphabetical), year, volume, page.
        prov = (t.get("province", "") or "~~~").strip().lower()
        return (not t["is_tableVII"], prov, yr(t), str(t.get("saoa", "")), t.get("block", [0])[0])
    keep.sort(key=rank)

    css = """
    body{font:13px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;margin:0;background:#f4f5f7;color:#1a1a1a}
    header{position:sticky;top:0;background:#1f3864;color:#fff;padding:12px 18px;z-index:5}
    header h1{margin:0;font-size:18px} .sub{opacity:.9;font-size:12px;margin-top:3px}
    .bar{margin:8px 18px;display:flex;gap:8px} .bar input{flex:1;padding:7px 11px;border:1px solid #ccc;border-radius:6px}
    .feat{background:#fff8e1;padding:6px 18px;font-weight:700;color:#7a5b00;border-bottom:1px solid #f0e0a0}
    .t{background:#fff;margin:10px 18px;border:1px solid #e2e4e8;border-radius:8px;overflow:hidden}
    .t.vii{border-left:4px solid #1f3864}
    .th{display:flex;gap:10px;align-items:center;padding:9px 12px;background:#eef1f7;flex-wrap:wrap}
    .th b{font-size:14px} .prov{color:#555}
    .b{padding:1px 8px;border-radius:10px;font-size:11px;font-weight:700}
    .b.ok{background:#c6efce;color:#0a5d1e}.b.warn{background:#ffeb9c;color:#7a5b00}.b.bad{background:#ffc7ce;color:#9c0017}
    .vt{background:#1f3864;color:#fff;padding:0 6px;border-radius:3px;font-size:10px;font-weight:700}
    .body{display:flex;gap:12px;padding:10px;align-items:flex-start;overflow:auto}
    .pg{flex:0 0 auto;display:flex;gap:8px}
    .pg img{width:300px;border:1px solid #ccc;border-radius:4px}
    .mx{border-collapse:collapse;font-size:10px} .mx td,.mx th{border:1px solid #e3e3e3;padding:1px 3px;text-align:right}
    .mx th{background:#eef1f7;font-size:9px} .mx td.lbl{text-align:left;max-width:130px} .mx .s{color:#888;font-weight:400}
    .secname{font-weight:700;font-size:11px;margin:2px 0;color:#1f3864} .dim{color:#999}
    a{color:#1f4e79} .jstor{font-size:11px}
    """
    out = [f"<!doctype html><meta charset=utf-8><title>Census Age/Sex/Civil-Condition tables</title><style>{css}</style>"]
    nvii = sum(1 for t in keep if t["is_tableVII"])
    out.append(f"<header><h1>Indian Census — Age, Sex &amp; Civil Condition tables</h1>"
               f"<div class=sub>{len(keep)} tables shown · {nvii} Age/Sex/Civil-Condition (pinned first) · "
               f"badges show real state (verified · minor · needs review) — nothing hidden</div></header>"
               "<div class=bar><input id=q placeholder='filter by province, year, title…'></div>"
               "<div class=feat>Key tables — Age, Sex &amp; Civil Condition (most important first)</div>")
    shown_feat = True
    for t in keep:
        if shown_feat and not t["is_tableVII"]:
            out.append("<div class=feat>Other reasonably-good tables</div>"); shown_feat = False
        lab, cls = BADGE.get(t["status"], ("", "ok"))
        s = f"{t.get('year','')} {t.get('province','')} {t.get('title','')}".lower()
        pages_html = []
        for p in t["pages"]:
            if p["status"] == "empty":
                continue
            sp = scan_path(p["saoa"], p["page"])
            src = thumb(sp, p["saoa"], p["page"]) if BUNDLE else (("file://" + sp) if sp else "")
            img = f'<a href="{html.escape(src)}" target=_blank><img loading=lazy src="{html.escape(src)}"></a>' if src else ""
            try:
                d = json.load(open(p["result"]))
                mx = matrix_html(d)
            except Exception:
                mx = "<div class=dim>(unreadable)</div>"
            pages_html.append(f'<div class=pg>{img}<div>{mx}</div></div>')
        jst = t["pages"][0].get("jstor", "") if t["pages"] else ""
        out.append(f'<div class="t {"vii" if t["is_tableVII"] else ""}" data-s="{html.escape(s)}">'
                   f'<div class=th>{"<span class=vt>VII</span>" if t["is_tableVII"] else ""}'
                   f'<b>{html.escape(t.get("title","") or "(untitled)")[:90]}</b>'
                   f'<span class=prov>{html.escape(t.get("province",""))} · {html.escape(str(t.get("year","")))}</span>'
                   f'<span class="b {cls}">{lab}</span>'
                   f'<span class=prov>{t["n_real"]} pg</span></div>'
                   f'<div class=body>{"".join(pages_html) or "<div class=dim>(no readable pages)</div>"}</div></div>')
    out.append("<script>const q=document.getElementById('q');q.addEventListener('input',()=>{"
               "const v=q.value.toLowerCase();document.querySelectorAll('.t').forEach(e=>"
               "e.style.display=(!v||e.dataset.s.includes(v))?'':'none');});</script>")
    outpath = os.path.join(DIST, "index.html") if BUNDLE else OUT
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    open(outpath, "w").write("\n".join(out))
    print(f"wrote {outpath}")
    print(f"  {len(keep)} tables ({nvii} Age/Sex/Civil-Condition pinned first)")
    if BUNDLE:
        nimg = len(glob.glob(os.path.join(DIST, "scans", "*.jpg")))
        sz = sum(os.path.getsize(f) for f in glob.glob(os.path.join(DIST, "**", "*"), recursive=True) if os.path.isfile(f))
        print(f"  bundled {nimg} scans · total {sz//1024//1024} MB · folder ready to deploy: {DIST}")
    print(f"  open: file://{os.path.abspath(outpath)}")


if __name__ == "__main__":
    main()
