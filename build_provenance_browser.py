#!/usr/bin/env python3
"""PROVENANCE BROWSER — a single self-contained HTML file with real collapsible
dropdowns:  VOLUME ▸ TABLE ▸ PAGE, where each PAGE shows its actual image file
(thumbnail + click-to-open). Works in any browser (no Excel needed).

Pages are ordered/labeled by the best DISTINCT page we have: the true per-crop PDF
page when present (crop_pdf_pages.csv), otherwise the printed/book page (which is
already distinct per page), otherwise the manifest table-start page.

  python3 build_master_tracker.py        # refresh data
  python3 build_provenance_browser.py     # -> provenance_browser.html
"""
import os, re, csv, html
from collections import defaultdict
from provenance_status import page_badge, table_status, is_assembled, is_tableVII, load_image_pages

csv.field_size_limit(10 ** 7)
ROOT = os.path.dirname(os.path.abspath(__file__))
V1 = os.path.join(ROOT, "export_final/sendout_final/sendout_v1")
BIG = 10 ** 9


def i(x, default=BIG):
    try:
        return int(float(x))
    except Exception:
        return default


def load():
    rows = list(csv.DictReader(open(os.path.join(V1, "master_tracking.csv"))))
    for r in rows:
        r["_pdf"] = i(r.get("pdf_page", ""))
        r["_printed"] = i(r.get("printed_page", ""))
    return rows


def order_key(r):
    if r.get("pdf_page_source") == "actual" and r["_pdf"] < BIG:
        return r["_pdf"]
    if r["_printed"] < BIG:
        return r["_printed"]
    return r["_pdf"]


def page_label(r):
    if r.get("pdf_page_source") == "actual" and r["_pdf"] < BIG:
        return f"PDF&nbsp;p{r['_pdf']}"
    if r["_printed"] < BIG:
        return f"p.{r['_printed']}&nbsp;<span class=dim>book</span>"
    if r["_pdf"] < BIG:
        return f"PDF&nbsp;p{r['_pdf']}&nbsp;<span class=dim>prov.</span>"
    return "<span class=dim>unpaged</span>"


def clean(s):
    return re.sub(r"^[^0-9A-Za-z]+", "", s or "").strip()


def cls_of(s):
    low = (s or "").lower()
    if "not ocr" in low:
        return "none"
    if "fail" in low or "low/" in low:
        return "bad"
    if "imperfect" in low or "retry" in low:
        return "warn"
    if "complete" in low or "perfect" in low:
        return "ok"
    return "info"


def badge(status):
    t = clean(status)
    return f'<span class="b {cls_of(t)}">{html.escape(t)}</span>' if t else ""


CSS = """
*{box-sizing:border-box} body{font:13px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;margin:0;color:#1a1a1a;background:#f4f5f7}
header{position:sticky;top:0;background:#1f3864;color:#fff;padding:10px 16px;z-index:5;box-shadow:0 2px 6px rgba(0,0,0,.2)}
header h1{margin:0 0 4px;font-size:16px} .tot{font-size:12px;opacity:.9}
.bar{margin-top:8px;display:flex;gap:8px;flex-wrap:wrap;align-items:center}
.bar input{flex:1;min-width:220px;padding:6px 10px;border:0;border-radius:5px}
.bar button{padding:6px 10px;border:0;border-radius:5px;background:#2f5496;color:#fff;cursor:pointer}
.bar button:hover{background:#3a64b0}
main{padding:12px 16px;max-width:1200px;margin:auto}
details{background:#fff;border:1px solid #e2e4e8;border-radius:6px;margin:6px 0}
details>summary{cursor:pointer;padding:8px 12px;list-style:none;display:flex;gap:10px;align-items:center;flex-wrap:wrap}
details>summary::-webkit-details-marker{display:none}
details>summary::before{content:'\\25B8';color:#9aa3b2;font-size:10px;margin-right:2px}
details[open]>summary::before{content:'\\25BE'}
.vii{display:inline-block;padding:0 5px;border-radius:3px;background:#1f3864;color:#fff;font-size:10px;font-weight:700;letter-spacing:.3px}
.vol>summary{font-weight:700;font-size:14px;background:#eaf0fb;border-radius:6px}
.tbl{margin:6px 8px 6px 22px;border-left:3px solid #c8d3ea}
.tbl>summary{font-weight:600}
.range{color:#555;font-variant-numeric:tabular-nums}
.pages{margin:2px 10px 10px 30px;display:flex;flex-wrap:wrap;gap:10px}
.pg{width:150px;border:1px solid #e2e4e8;border-radius:6px;overflow:hidden;background:#fafbfc}
.pg img{width:100%;height:120px;object-fit:cover;display:block;background:#eee}
.pg .meta{padding:5px 7px;font-size:11px}
.pg .pl{font-weight:700;font-variant-numeric:tabular-nums}
.dim{color:#999;font-weight:400;font-size:10px}
.b{display:inline-block;padding:1px 6px;border-radius:10px;font-size:11px;white-space:nowrap}
.b.ok{background:#c6efce;color:#0a5d1e}.b.warn{background:#ffeb9c;color:#7a5b00}
.b.bad{background:#ffc7ce;color:#9c0017}.b.none{background:#e2e2e2;color:#555}.b.info{background:#ddebf7;color:#1f4e79}
.b.img{background:#e4dfec;color:#5b3a7a}
.cnt{color:#666;font-size:11px} .contains{color:#555;font-size:10px;margin-top:2px}
.hide{display:none}
"""

JS = """
const q=document.getElementById('q'), main=document.getElementById('main');
q.addEventListener('input',()=>{const t=q.value.toLowerCase();
 document.querySelectorAll('details.vol').forEach(v=>{
   v.classList.toggle('hide', t && !v.dataset.s.includes(t));});});
function sortBy(k){const vols=[...document.querySelectorAll('details.vol')];
 vols.sort((a,b)=>(k==='year'? (a.dataset.year-b.dataset.year)||a.dataset.prov.localeCompare(b.dataset.prov)
   : (b.dataset[k]-a.dataset[k])));
 vols.forEach(v=>main.appendChild(v));}
"""


def main():
    rows = load() + load_image_pages(V1)
    vols = defaultdict(lambda: defaultdict(list))
    docmeta = {}
    for r in rows:
        ltab = r.get("assembled_table") or r.get("table_id") or f"~{r.get('exact_title','')[:30]}"
        vols[r["volume"]][ltab].append(r)
        docmeta.setdefault(r["volume"], (r.get("year", ""), r.get("province", "")))

    tot_perf = sum(1 for r in rows if r["validation"] == "perfect")
    tot_not = sum(1 for r in rows if r["ocr_status"] == "NOT_OCRd")
    has_actual = any(r.get("pdf_page_source") == "actual" for r in rows)
    n_asm = sum(1 for tabs in vols.values() for mem in tabs.values() if is_assembled(mem))
    n_single_vii = sum(1 for tabs in vols.values() for mem in tabs.values()
                       if is_tableVII(mem) and not is_assembled(mem))

    def vsort(dh):
        y, pr = docmeta.get(dh, ("", ""))
        return (i(y, 9999), pr, dh)

    out = [f"<!doctype html><meta charset=utf-8><title>Provenance browser</title><style>{CSS}</style>"]
    out.append("<header><h1>Census Table VII — provenance browser</h1>"
               f"<div class=tot><b>0 verified-complete tables</b> "
               "(pending content inventory) · "
               f"{n_asm} assembled · {n_single_vii} unassembled VII pages · "
               f"{tot_perf} cell-valid pages · {tot_not} not-OCR'd · {len(vols)} volumes · "
               f"page order: {'TRUE PDF page' if has_actual else 'book page (PDF map pending)'}</div>"
               "<div class=tot style='opacity:.8;font-size:11px'>‘cells valid’ = the page's "
               "arithmetic checks out; it does NOT mean the full table sequence is present or "
               "assembled. ‘Complete’ requires the content inventory + assembly.</div>"
               "<div class=bar><input id=q placeholder='filter volumes — year, province, title…'>"
               "<button onclick=\"sortBy('year')\">sort: year</button>"
               "<button onclick=\"sortBy('perf')\">most perfect</button>"
               "<button onclick=\"sortBy('pages')\">most pages</button></div></header>")
    out.append("<main id=main>")

    for dh in sorted(vols, key=vsort):
        y, pr = docmeta.get(dh, ("", ""))
        tables = vols[dh]
        allr = [m for t in tables.values() for m in t]
        nperf = sum(1 for m in allr if m["validation"] == "perfect")
        npages = len(allr)
        pdfs = sorted(order_key(m) for m in allr if order_key(m) < BIG)
        span = f"p{pdfs[0]}–p{pdfs[-1]}" if pdfs else ""
        s = f"{y} {pr} {dh}".lower()
        out.append(f'<details class=vol data-year="{i(y,9999)}" data-prov="{html.escape(pr)}" '
                   f'data-perf="{nperf}" data-pages="{npages}" data-s="{html.escape(s)}">')
        out.append(f'<summary>{html.escape(y or "—")} · {html.escape(pr or dh[:8])} '
                   f'<span class=range>{span}</span> '
                   f'<span class=cnt>{len(tables)} tables · {npages} pages · {nperf} cell-valid</span></summary>')

        roll = {}
        for t, mem in tables.items():
            ks = sorted(order_key(m) for m in mem if order_key(m) < BIG)
            roll[t] = ks[0] if ks else BIG
        for t in sorted(tables, key=lambda t: (roll[t], t)):
            mem = sorted(tables[t], key=order_key)
            tnum = next((m.get("table_number", "") for m in mem if m.get("table_number")), "")
            fam = next((m.get("family", "") for m in mem if m.get("family")), "")
            title = next((m.get("exact_title", "") for m in mem if m.get("exact_title")), "")
            name = (f"Table {tnum}" if tnum else (fam or (title[:36] if title else "Table")))
            ks = [order_key(m) for m in mem if order_key(m) < BIG]
            rng = f"p{min(ks)}–p{max(ks)}" if ks else ""
            np = sum(1 for m in mem if m["validation"] == "perfect")
            tst, tcls = table_status(mem)
            vii = '<span class=vii>VII</span> ' if any(m.get("is_tableVII") == "Y" for m in mem) else ""
            out.append(f'<details class=tbl><summary>{vii}{html.escape(name)} '
                       f'<span class=range>{rng}</span> <span class="b {tcls}">{html.escape(tst)}</span> '
                       f'<span class=cnt>{len(mem)} pg · {np} cell-valid</span> '
                       f'<span class=contains>{html.escape((title or "")[:70])}</span></summary>')
            out.append('<div class=pages>')
            for m in mem:
                ip = m.get("image_path", "")
                src = ("file://" + os.path.join(ROOT, ip)) if ip else ""
                contains = m.get("contains_sections", "") or m.get("age_range", "")
                out.append('<div class=pg>')
                if src:
                    out.append(f'<a href="{html.escape(src)}" target=_blank>'
                               f'<img loading=lazy src="{html.escape(src)}"></a>')
                pl, pcls = page_badge(m)
                out.append(f'<div class=meta><div class=pl>{page_label(m)}</div>'
                           f'<span class="b {pcls}">{html.escape(pl)}</span>'
                           f'<div class=contains>{html.escape((contains or "")[:46])}</div></div></div>')
            out.append('</div></details>')
        out.append('</details>')

    out.append("</main>")
    out.append(f"<script>{JS}</script>")
    p = os.path.join(V1, "provenance_browser.html")
    open(p, "w").write("\n".join(out))
    print(f"wrote {p}")
    print(f"  {len(vols)} volumes · {len(rows)} pages · {tot_perf} perfect")
    print(f"  open it:  file://{p}")


if __name__ == "__main__":
    main()
