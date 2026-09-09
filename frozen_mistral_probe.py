#!/usr/bin/env python3
"""Probe: does Mistral-OCR read the 'frozen column' pages correctly (varying per-row values) where
Gemini froze them? Renders a sample of frozen-column pages full-page from the drive at 300 DPI, runs
Mistral OCR -> markdown, and prints the markdown table so we can see if the column varies down the rows.
  python3 frozen_mistral_probe.py [ih1 ih2 ...]
"""
import os, re, csv, sys, base64, glob
import fitz
from dotenv import dotenv_values

ROOT = os.path.dirname(os.path.abspath(__file__))
V1 = os.path.join(ROOT, "export_final/sendout_final/sendout_v1")
DRIVE = "/Volumes/Expansion/census_pdfs"
ENV = dotenv_values(os.path.join(ROOT, ".env"))
SAMPLE = sys.argv[1:] or [
    "30648f815df431d57b08adff148b402c10f076d9",   # West India 1901 Hindu — your example
    "ca4284afd1f3", "d882e94e5e68", "ccae38593f2d",
]


def fullhash(p):
    if re.fullmatch(r"[0-9a-f]{40}", p):
        return p
    g = glob.glob(os.path.join(ROOT, "export/ocr_results_referenced/digitized_tables", "*", f"{p}*_oneshot.json"))
    m = re.search(r"([0-9a-f]{40})_oneshot", os.path.basename(g[0])) if g else None
    return m.group(1) if m else p


def page_loc():
    idx = {}
    for r in csv.DictReader(open(os.path.join(V1, "page_number_status.csv"))):
        if r.get("pdf_page", "").lstrip("-").isdigit():
            idx[r["imghash"]] = (r["source_pdf"], int(r["pdf_page"]), r["title"])
    return idx


def render(pdf_base, pg, out):
    hits = glob.glob(os.path.join(DRIVE, "*", pdf_base))
    if not hits:
        return False
    doc = fitz.open(hits[0])
    if not (1 <= pg <= doc.page_count):
        doc.close(); return False
    pix = doc.load_page(pg - 1).get_pixmap(matrix=fitz.Matrix(300 / 72, 300 / 72), colorspace=fitz.csGRAY)
    pix.save(out); doc.close(); return True


def main():
    from mistralai import Mistral
    client = Mistral(api_key=ENV.get("MISTRAL_API_KEY"))
    loc = page_loc()
    for raw in SAMPLE:
        ih = fullhash(raw)
        if ih not in loc:
            print(f"\n### {ih[:12]}: no page location"); continue
        src, pg, title = loc[ih]
        base = os.path.basename(src)
        out = f"/tmp/frozen_{ih[:12]}.png"
        if not render(base, pg, out):
            print(f"\n### {ih[:12]} ({title[:40]}): render failed (drive? page {pg} of {base})"); continue
        try:
            uri = "data:image/png;base64," + base64.b64encode(open(out, "rb").read()).decode()
            resp = client.ocr.process(model="mistral-ocr-latest",
                                      document={"type": "image_url", "image_url": {"url": uri}},
                                      include_image_base64=False, timeout_ms=120000)
            md = "\n".join(getattr(p, "markdown", "") or "" for p in (getattr(resp, "pages", []) or []))
        except Exception as e:
            print(f"\n### {ih[:12]}: Mistral error {e}"); continue
        # show the first markdown table block (rows with several | )
        lines = [l for l in md.splitlines() if l.count("|") >= 4]
        print(f"\n{'='*72}\n### {ih[:12]}  p{pg}  {title[:50]}\n  Mistral markdown table (first ~16 rows):")
        for l in lines[:16]:
            print("   " + l[:150])
        if not lines:
            print("   (no pipe-table found; first 400 chars:)\n   " + md[:400].replace("\n", " "))


if __name__ == "__main__":
    main()
