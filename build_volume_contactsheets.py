#!/usr/bin/env python3
"""Per-volume VISUAL contact sheets: for each volume, show its crop PNGs in page order
(by pdf_page = book page + offset), starting where the table begins — so you can SEE
the sequence and spot gaps with your eyes. Each image is labeled with its pdf/book
page, status, sections, and position. Gaps are shown as red markers.
Reads master_tracking.csv. Writes contactsheets/index.html + contactsheets/vol_<dochash>.html."""
import os, csv, html
from collections import defaultdict

csv.field_size_limit(10 ** 7)
V1 = "export_final/sendout_final/sendout_v1"
OUT = "contactsheets"


def esc(s):
    return html.escape(str(s or ""))


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = list(csv.DictReader(open(os.path.join(V1, "master_tracking.csv"))))
    byvol = defaultdict(list)
    for r in rows:
        byvol[r["volume"]].append(r)

    index = []
    for dh, imgs in byvol.items():
        # sort by pdf_page (then book page); unpaged go last
        def key(r):
            pp = r["pdf_page"]
            bp = r["printed_page"]
            return (int(pp) if str(pp).lstrip("-").isdigit() else 99999,
                    int(bp) if str(bp).lstrip("-").isdigit() else 99999)
        imgs.sort(key=key)
        vii = sum(1 for r in imgs if r["is_tableVII"] == "Y")
        if vii == 0:
            continue   # contact sheets for Table-VII-bearing volumes
        yr = imgs[0]["year"]; prov = imgs[0]["province"]; pdf = imgs[0]["source_pdf"]
        pages = [int(r["pdf_page"]) for r in imgs if str(r["pdf_page"]).lstrip("-").isdigit()]
        span = f"{min(pages)}-{max(pages)}" if pages else "?"
        cov = f"{len(set(pages))} pages, span {span}" if pages else "no page numbers yet"

        cards = []
        prev = None
        for r in imgs:
            pp = r["pdf_page"]; bp = r["printed_page"]
            # gap marker
            if prev is not None and str(pp).lstrip("-").isdigit() and str(prev).lstrip("-").isdigit():
                miss = int(pp) - int(prev) - 1
                if miss > 0:
                    cards.append(f'<div class="gap">⚠ missing {miss} page(s)<br>(pdf {int(prev)+1}–{int(pp)-1})</div>')
            if str(pp).lstrip("-").isdigit():
                prev = pp
            st = r["status"]
            cls = ("ok" if st.startswith("✅") else "warn" if st.startswith("⚠") else
                   "todo" if "NOT OCR" in st else "info")
            img = os.path.abspath(r["image_path"])
            cards.append(
                f'<div class="card {cls}">'
                f'<div class="pg">pdf <b>{esc(pp) or "?"}</b> · book {esc(bp) or "?"}</div>'
                f'<a href="file://{esc(img)}" target="_blank"><img loading="lazy" src="file://{esc(img)}"></a>'
                f'<div class="st">{esc(st)[:34]}</div>'
                f'<div class="se">{esc(r["table_number"])} · {esc(r["contains_sections"])[:40]}</div>'
                f'<div class="po">{esc(r["position_in_table"])}</div>'
                f'</div>')
        page = f"""<!doctype html><meta charset=utf-8><title>{esc(prov)} {esc(yr)}</title>
<style>body{{font:13px system-ui;margin:0;background:#1a1a1a;color:#ddd}}
.hd{{padding:10px 16px;background:#000;position:sticky;top:0;z-index:9}}
.row{{display:flex;flex-wrap:wrap;gap:8px;padding:14px}}
.card{{width:200px;background:#262626;border:2px solid #444;border-radius:6px;padding:6px;font-size:11px}}
.card.ok{{border-color:#2a2}}.card.warn{{border-color:#a82}}.card.todo{{border-color:#39c}}
.card img{{width:100%;background:#fff;border:1px solid #555}}
.pg{{font-size:12px}}.st{{color:#fc8;margin-top:3px}}.se{{color:#9bd}}.po{{color:#888}}
.gap{{width:120px;display:flex;align-items:center;justify-content:center;background:#411;border:2px dashed #c44;color:#f99;text-align:center;border-radius:6px;font-size:11px}}
a{{color:#6cf}}</style>
<div class=hd><a href="index.html">← all volumes</a> &nbsp; <b>{esc(prov)} {esc(yr)}</b> · {esc(pdf)} · {esc(cov)} · {len(imgs)} crops, {vii} Table VII
<br><span style="color:#888">images in PDF-page order (pdf = book + offset {esc(imgs[0]['offset'])}); red = a gap (pages we don't have)</span></div>
<div class=row>{''.join(cards)}</div>"""
        fn = f"vol_{dh}.html"
        open(os.path.join(OUT, fn), "w").write(page)
        index.append((dh, prov, yr, pdf, len(imgs), vii, cov, fn))

    index.sort(key=lambda x: (str(x[2]), str(x[1])))
    items = "".join(
        f'<tr><td><a href="{fn}">{esc(prov)} {esc(yr)}</a></td><td>{esc(pdf)}</td>'
        f'<td>{n}</td><td>{vii}</td><td>{esc(cov)}</td></tr>'
        for dh, prov, yr, pdf, n, vii, cov, fn in index)
    idx = f"""<!doctype html><meta charset=utf-8><title>Volume contact sheets</title>
<style>body{{font:14px system-ui;background:#1a1a1a;color:#ddd;padding:16px}}a{{color:#6cf}}
table{{border-collapse:collapse}}td,th{{border:1px solid #444;padding:4px 10px}}th{{background:#000}}</style>
<h2>Volume contact sheets — {len(index)} Table VII volumes</h2>
<p style="color:#888">Click a volume to see its crop PNGs in PDF-page order (from where the table begins), with gaps marked in red.</p>
<table><tr><th>volume</th><th>source PDF</th><th>crops</th><th>Table VII</th><th>coverage</th></tr>{items}</table>"""
    open(os.path.join(OUT, "index.html"), "w").write(idx)
    print(f"volume contact sheets: {len(index)} volumes -> {OUT}/index.html")
    print("Open contactsheets/index.html — click a volume to see its pages in order.")


if __name__ == "__main__":
    main()
