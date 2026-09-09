#!/usr/bin/env python3
"""Render front-matter pages (where the Table of Contents lives) from volume PDFs on the drive,
memory-safely. Default: a pilot of volumes spread across census years. The TOC vision pass then
reads these to extract the chapter->table->page-range spine.
  python3 build_toc_render.py [--pilot] [--all] [--pages N] [--saoa S1,S2,...]
"""
import csv, os, re, sys, json, glob
from collections import OrderedDict

DRIVE = "/Volumes/Expansion/census_pdfs"
PROV = "export_final/sendout_final/sendout_v1/provenance.csv"
OUT = "export/toc_renders"
NPAGES = int(sys.argv[sys.argv.index("--pages")+1]) if "--pages" in sys.argv else 22
PILOT = "--pilot" in sys.argv or len(sys.argv) == 1
ALLV = "--all" in sys.argv
SAOA = (sys.argv[sys.argv.index("--saoa")+1].split(",")) if "--saoa" in sys.argv else None

try:
    import fitz
except ImportError:
    sys.exit("need pymupdf")


def vol_year():
    """ordered map saoa -> year, from provenance (volumes we actually have tables from)."""
    vy = OrderedDict()
    for r in csv.DictReader(open(PROV)):
        sp = r.get("source_pdf", "")
        m = re.search(r"saoa\.crl\.(\d+)\.pdf", sp)
        y = (r.get("year") or "").strip()
        if m and y:
            vy.setdefault(m.group(1), y)
    return vy


def main():
    vy = vol_year()
    if SAOA:
        vols = [(s, vy.get(s, "")) for s in SAOA]
    elif ALLV:
        vols = list(vy.items())
    else:  # pilot: one volume per year across the spread
        byyear = OrderedDict()
        for sa, y in vy.items():
            byyear.setdefault(y, sa)
        vols = [(sa, y) for y, sa in byyear.items()]
    os.makedirs(OUT, exist_ok=True)
    rendered = missing = 0
    manifest = []
    for sa, y in vols:
        path = os.path.join(DRIVE, y, f"saoa.crl.{sa}.pdf")
        if not os.path.exists(path):
            missing += 1
            continue
        try:
            doc = fitz.open(path)
        except Exception:
            missing += 1
            continue
        n = min(NPAGES, doc.page_count)
        pgs = []
        for pg in range(1, n+1):
            outpng = os.path.join(OUT, f"{sa}_p{pg}.png")
            if not os.path.exists(outpng):
                try:
                    pix = doc.load_page(pg-1).get_pixmap(matrix=fitz.Matrix(200/72, 200/72))
                    pix.save(outpng); pix = None; rendered += 1
                except Exception:
                    continue
            pgs.append(os.path.abspath(outpng))
        doc.close()
        manifest.append({"saoa": sa, "year": y, "front_pages": pgs})
    json.dump(manifest, open(f"{OUT}/manifest.json", "w"), indent=1)
    print(f"volumes: {len(manifest)} | pages rendered: {rendered} | missing PDFs: {missing}")
    print(f"-> {OUT}/  (manifest.json lists front pages per volume)")


if __name__ == "__main__":
    main()
