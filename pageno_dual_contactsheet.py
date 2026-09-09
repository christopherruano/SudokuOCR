#!/usr/bin/env python3
"""Self-contained HTML to verify the dual-number reads: for each scan that has an
Arabic volume page, show the TOP crop and the BOTTOM crop side by side, with what we
parsed (volume page = Arabic, usually bottom; section page = Roman). The 86 pages whose
volume page DIFFERS from the earlier Roman-only read are surfaced first (those are the
corrections that matter most).

Usage: python3 pageno_dual_contactsheet.py [--out FILE]
Open the file in a browser (images embedded).
"""
import argparse, base64, io, json, os, html
from PIL import Image

ROOT = os.path.dirname(os.path.abspath(__file__))
V1 = os.path.join(ROOT, "export_final/sendout_final/sendout_v1")
PARTS = os.path.join(V1, "pageno_parts")


def thumb(im, width=480):
    if im.width > width:
        im = im.resize((width, int(im.height * width / im.width)))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=72)
    return base64.b64encode(buf.getvalue()).decode()


def crops(path):
    im = Image.open(path).convert("RGB")
    w, h = im.size
    top = im.crop((0, 0, w, int(h * 0.13)))
    bot = im.crop((0, int(h * 0.86), w, h))
    return thumb(top), thumb(bot)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(PARTS, "dual_contactsheet.html"))
    args = ap.parse_args()

    wl = {w["imghash"]: w for w in json.load(open(os.path.join(PARTS, "worklist.json")))}
    roman = {}
    for line in open(os.path.join(PARTS, "roman_reads.jsonl")):
        r = json.loads(line)
        roman[r["imghash"]] = r.get("page_number_int")
    dual = [json.loads(l) for l in open(os.path.join(PARTS, "dualnum_reads.jsonl"))]

    have_vol = [d for d in dual if d.get("volume_page") is not None]
    corrections = [d for d in have_vol if roman.get(d["imghash"]) not in (None, d["volume_page"])]
    rest_vol = [d for d in have_vol if d not in corrections]
    roman_only = [d for d in dual if d.get("volume_page") is None and d.get("section_page") is not None]

    def card(d, corrected):
        h = d["imghash"]
        scan = os.path.join(ROOT, wl[h]["scan_path"])
        try:
            t, b = crops(scan)
            imgs = (f'<div class="c"><div class="lbl">TOP</div><img src="data:image/jpeg;base64,{t}"></div>'
                    f'<div class="c"><div class="lbl">BOTTOM</div><img src="data:image/jpeg;base64,{b}"></div>')
        except Exception as e:
            imgs = f'<div class="err">{html.escape(str(e)[:80])}</div>'
        was = roman.get(h)
        corr = f'<span class="badge corr">was {was} → {d["volume_page"]}</span>' if corrected else ''
        return f"""<div class="cell{' c2' if corrected else ''}">
          <div class="hd"><span class="vol">p.{d['volume_page']}</span>
            <span class="meta">volume (arabic, bottom)</span> {corr}</div>
          <div class="imgs">{imgs}</div>
          <div class="meta">section(roman top)={d.get('section_page')} · top={html.escape(str(d.get('top_raw')))}({d.get('top_system')}) · bottom={html.escape(str(d.get('bottom_raw')))}({d.get('bottom_system')}) · {d.get('page_type')}</div>
          <div class="meta">{html.escape(os.path.basename(wl[h]['scan_path']))}</div>
        </div>"""

    cells = ([card(d, True) for d in corrections] + [card(d, False) for d in rest_vol])

    doc = f"""<!doctype html><html><head><meta charset="utf-8"><title>Dual-number verification</title>
<style>
 body{{font:13px -apple-system,Segoe UI,sans-serif;margin:14px;background:#0f0f12;color:#eee}}
 h1{{font-size:18px}} .sum{{background:#1b1b1f;border:1px solid #333;border-radius:8px;padding:10px 14px;margin-bottom:14px}}
 .grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(440px,1fr));gap:12px}}
 .cell{{background:#1b1b1f;border:1px solid #333;border-radius:8px;padding:10px}}
 .cell.c2{{border:2px solid #d98}}
 .hd{{display:flex;gap:8px;align-items:center;margin-bottom:6px}}
 .vol{{font-size:22px;font-weight:700;color:#9fe3ff}}
 .imgs{{display:flex;gap:8px}} .c{{flex:1;min-width:0}} .lbl{{font-size:10px;color:#888}}
 .imgs img{{width:100%;border:1px solid #333;border-radius:4px;background:#000}}
 .meta{{color:#9aa;font-size:11px;margin-top:4px;word-break:break-all}}
 .badge{{font-size:11px;padding:2px 8px;border-radius:10px}} .badge.corr{{background:#5a3a1a;color:#fd9}}
</style></head><body>
<h1>Dual-number verification — is the bottom Arabic the right page?</h1>
<div class="sum">
 <b>{len(have_vol)}</b> pages have an Arabic volume page (verify the big <b>p.N</b> matches the BOTTOM crop).
 <b>{len(corrections)}</b> are corrections (orange) where the old Roman-only read was different — these matter most.
 Also {len(roman_only)} pages are Roman/section-only (no Arabic volume found; not shown here).
</div>
<div class="grid">{''.join(cells)}</div>
</body></html>"""
    open(args.out, "w").write(doc)
    print(f"volume pages: {len(have_vol)} | corrections (shown first): {len(corrections)} | roman-only(not shown): {len(roman_only)}")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
