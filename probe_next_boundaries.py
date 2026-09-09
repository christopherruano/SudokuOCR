#!/usr/bin/env python3
"""Loop round: render the NEXT boundary pages (export/next_boundaries.json) SMALL+grayscale so they
upload fast, for a quick TITLE read (Opus judges boundaries from the title — full table data is only
needed for pages that get confirmed into a run, which we full-OCR at the end). Writes the renders +
export/next_probe_list.txt for Gemini OCR.
  python3 probe_next_boundaries.py
"""
import json, os, re, csv, hashlib, glob

ROOT = os.path.dirname(os.path.abspath(__file__))
V1 = os.path.join(ROOT, "export_final/sendout_final/sendout_v1")
DRIVE = "/Volumes/Expansion/census_pdfs"
OUT = os.path.join(ROOT, "export/boundary_grow_renders")
NB = os.path.join(ROOT, "export/next_boundaries.json")

import fitz
from PIL import Image


def vol_map():
    m = {}
    for fn, col in (("cc_volumes.csv", "cluster_path"), ("documents.csv", "source_pdf"),
                    ("provenance.csv", "source_pdf")):
        p = os.path.join(V1, fn)
        if os.path.exists(p):
            for r in csv.DictReader(open(p)):
                cp = r.get(col)
                if cp and "census_pdfs" in cp:
                    m.setdefault(hashlib.sha1(cp.encode()).hexdigest()[:16], cp)
    return m


def main():
    vol = vol_map()
    nb = json.load(open(NB)) if os.path.exists(NB) else []
    os.makedirs(OUT, exist_ok=True)
    by_cp = {}
    for item in nb:
        cp = vol.get(item["dochash"])
        if cp:
            by_cp.setdefault(cp, []).append((item["saoa"], int(item["page"])))
    lst = open(os.path.join(ROOT, "export/next_probe_list.txt"), "w")
    rendered = skip = nodrv = 0
    for cp, items in by_cp.items():
        ym = re.search(r"/(\d{4})/", cp)
        yr = ym.group(1) if ym else ""
        path = os.path.join(DRIVE, yr, os.path.basename(cp))
        if not os.path.exists(path):
            hits = glob.glob(os.path.join(DRIVE, "*", os.path.basename(cp)))
            path = hits[0] if hits else ""
        if not path:
            nodrv += len(items); continue
        try:
            doc = fitz.open(path)
        except Exception:
            nodrv += len(items); continue
        for sa, pg in items:
            out = os.path.join(OUT, f"{sa}_p{pg}.jpg")
            if os.path.exists(out):
                skip += 1
            elif 1 <= pg <= doc.page_count:
                try:
                    pix = doc.load_page(pg - 1).get_pixmap(matrix=fitz.Matrix(150 / 72, 150 / 72),
                                                           colorspace=fitz.csGRAY)
                    Image.frombytes("L", (pix.width, pix.height), pix.samples).save(
                        out, "JPEG", quality=70, optimize=True)
                    pix = None; rendered += 1
                except Exception:
                    continue
            else:
                continue
            if os.path.exists(out):
                lst.write(os.path.abspath(out) + "\n")
        doc.close()
    lst.close()
    n = sum(1 for _ in open(os.path.join(ROOT, "export/next_probe_list.txt")))
    print(f"probe render: {len(nb)} next-boundaries | rendered {rendered} small | had {skip} | no-drive {nodrv}")
    print(f"  -> export/next_probe_list.txt ({n} pages to fast-OCR)")


if __name__ == "__main__":
    main()
