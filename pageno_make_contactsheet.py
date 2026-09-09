#!/usr/bin/env python3
"""Build a self-contained HTML contact sheet so a human can VISUALLY confirm the
Gemini page-number reads against the actual top-of-page strip of each scan.

Surfaces the uncertain reads first (null page number, low/medium confidence,
fallback-recovered), then a random sample of high-confidence reads to confirm the
baseline. Each cell shows the cropped header strip + the read number + flags.

Usage:
  python3 pageno_make_contactsheet.py [--sample 80] [--all] [--out FILE] [--strip-frac 0.30]
Open the resulting HTML file in a browser (it embeds the images, so no server needed).
"""
import argparse, base64, io, json, os, html

from PIL import Image

ROOT = os.path.dirname(os.path.abspath(__file__))
V1 = os.path.join(ROOT, "export_final/sendout_final/sendout_v1")
PARTS = os.path.join(V1, "pageno_parts")
JSONL = os.path.join(PARTS, "gemini_reads.jsonl")


def load_reads():
    by_hash = {}
    for line in open(JSONL):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        by_hash[r["imghash"]] = r  # last write wins
    return list(by_hash.values())


def thumb(path, frac, width=460):
    im = Image.open(path).convert("RGB")
    w, h = im.size
    strip = im.crop((0, 0, w, max(1, int(h * frac))))
    if strip.width > width:
        strip = strip.resize((width, int(strip.height * width / strip.width)))
    buf = io.BytesIO()
    strip.save(buf, format="JPEG", quality=78)
    return base64.b64encode(buf.getvalue()).decode()


def bucket(r):
    if r.get("error"):
        return "error"
    if r.get("page_number") is None:
        return "null"
    c = (r.get("confidence") or "").lower()
    if c not in ("high",):
        return "lowmed"
    if r.get("used_fallback"):
        return "fallback"
    return "high"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=80, help="random high-confidence reads to include")
    ap.add_argument("--all", action="store_true", help="include every read (large file)")
    ap.add_argument("--strip-frac", type=float, default=0.30)
    ap.add_argument("--out", default=os.path.join(PARTS, "contactsheet.html"))
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()

    reads = load_reads()
    buckets = {"error": [], "null": [], "lowmed": [], "fallback": [], "high": []}
    for r in reads:
        buckets[bucket(r)].append(r)

    # deterministic sample of high-confidence reads (no Math.random / Date use)
    import random
    rng = random.Random(args.seed)
    high = list(buckets["high"])
    rng.shuffle(high)
    high_sample = high if args.all else high[: args.sample]

    # order: surface uncertain first
    ordered = (buckets["error"] + buckets["null"] + buckets["lowmed"] +
               buckets["fallback"] + high_sample)

    cells = []
    for i, r in enumerate(ordered):
        scan = os.path.join(ROOT, r["scan_path"])
        b = bucket(r)
        try:
            img = thumb(scan, args.strip_frac)
            imgtag = f'<img src="data:image/jpeg;base64,{img}">'
        except Exception as e:
            imgtag = f'<div class="err">image error: {html.escape(str(e)[:80])}</div>'
        pn = r.get("page_number")
        pn_txt = str(pn) if pn is not None else "—"
        conf = html.escape(r.get("confidence") or "")
        cont = r.get("continuation_marker")
        flags = []
        if r.get("used_fallback"):
            flags.append("fallback")
        if cont:
            flags.append("mark:" + html.escape(str(cont)))
        if r.get("error"):
            flags.append("ERR")
        n_tids = len(r.get("table_ids", []))
        n_asm = len(r.get("asm_ids", []))
        cells.append(f"""
        <div class="cell {b}">
          {imgtag}
          <div class="meta">
            <span class="pn">p.{html.escape(pn_txt)}</span>
            <span class="conf c-{conf}">{conf or 'no-conf'}</span>
          </div>
          <div class="flags">{' · '.join(flags)}</div>
          <div class="hash">{html.escape(r['imghash'][:12])} · in {n_asm} table(s), {n_tids} row(s)</div>
          <div class="path">{html.escape(os.path.basename(r['scan_path']))}</div>
        </div>""")

    summary = {k: len(v) for k, v in buckets.items()}
    summary["TOTAL_READS"] = len(reads)
    summary["SHOWN"] = len(ordered)

    doc = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Page-number read confirmation</title>
<style>
 body{{font:13px -apple-system,Segoe UI,sans-serif;margin:16px;background:#f6f7f9;color:#1a1a1a}}
 h1{{font-size:18px}} .sum{{margin:8px 0 18px;padding:10px 14px;background:#fff;border:1px solid #ddd;border-radius:8px}}
 .sum b{{display:inline-block;min-width:90px}}
 .grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:10px}}
 .cell{{background:#fff;border:1px solid #ddd;border-radius:8px;padding:8px;overflow:hidden}}
 .cell img{{width:100%;display:block;border:1px solid #eee;border-radius:4px;background:#fafafa}}
 .meta{{display:flex;justify-content:space-between;align-items:center;margin-top:6px}}
 .pn{{font-size:20px;font-weight:700}}
 .conf{{font-size:11px;padding:2px 7px;border-radius:10px;background:#eee}}
 .c-high{{background:#d6f5d6}} .c-medium{{background:#ffe9c2}} .c-low{{background:#ffd6d6}}
 .flags{{font-size:11px;color:#a55;min-height:14px}}
 .hash,.path{{font-size:10px;color:#888;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}
 .cell.null{{border:2px solid #e33;background:#fff5f5}}
 .cell.lowmed{{border:2px solid #f0a020;background:#fffaf0}}
 .cell.fallback{{border-left:5px solid #2a7ade}}
 .cell.error{{border:2px solid #900;background:#fee}}
 .legend span{{display:inline-block;margin-right:14px}}
 .sw{{display:inline-block;width:12px;height:12px;border-radius:3px;vertical-align:middle;margin-right:4px}}
</style></head><body>
<h1>Page-number read confirmation — confirm these are right</h1>
<div class="sum">
 <div><b>Total reads:</b> {summary['TOTAL_READS']:,} &nbsp; <b>Shown here:</b> {summary['SHOWN']:,}{' (ALL)' if args.all else f' (uncertain + {len(high_sample)} random high-conf)'}</div>
 <div><b>null:</b> {summary['null']} &nbsp; <b>low/med:</b> {summary['lowmed']} &nbsp; <b>fallback:</b> {summary['fallback']} &nbsp; <b>high-conf:</b> {summary['high']} &nbsp; <b>errors:</b> {summary['error']}</div>
 <div class="legend" style="margin-top:8px">
   <span><span class="sw" style="background:#fff5f5;border:2px solid #e33"></span>null (no number read)</span>
   <span><span class="sw" style="background:#fffaf0;border:2px solid #f0a020"></span>low/medium confidence</span>
   <span><span class="sw" style="border-left:5px solid #2a7ade;background:#fff"></span>recovered via full-image fallback</span>
 </div>
 <div style="margin-top:6px;color:#555">The big <b>p.N</b> is what Gemini read from the strip shown. Check it matches the printed number in the header image.</div>
</div>
<div class="grid">{''.join(cells)}</div>
</body></html>"""

    open(args.out, "w").write(doc)
    print(json.dumps(summary, indent=2))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
