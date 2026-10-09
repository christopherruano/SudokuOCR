#!/usr/bin/env python3
"""Render scans for published tables whose page images were never placed on the site.
Same output contract as build_static_site._drive_scan: census_site/img/<imghash>.jpg,
grayscale, 150 dpi, longest side <= 1800 px, JPEG q75."""
import json, csv, os, sys, collections
from concurrent.futures import ProcessPoolExecutor
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
IMG = os.path.join(ROOT, "census_site", "img")
PDFS = "/Volumes/Expansion/census_pdfs"

def render(job):
    pdf, items = job
    import fitz
    from PIL import Image
    done = 0
    try:
        doc = fitz.open(pdf)
    except Exception as e:
        return 0, f"open failed {pdf}: {e}"
    for h, pg in sorted(items, key=lambda x: x[1]):
        dest = os.path.join(IMG, h + ".jpg")
        if os.path.exists(dest) or not (1 <= pg <= doc.page_count):
            continue
        pix = doc.load_page(pg - 1).get_pixmap(matrix=fitz.Matrix(150 / 72, 150 / 72), colorspace=fitz.csGRAY)
        im = Image.frombytes("L", (pix.width, pix.height), pix.samples)
        w, hh = im.size
        if max(w, hh) > 1800:
            s = 1800 / max(w, hh); im = im.resize((max(1, int(w * s)), max(1, int(hh * s))))
        im.save(dest, "JPEG", quality=75, optimize=True)
        done += 1
    doc.close()
    return done, ""

def main():
    hs = json.load(open(os.path.join(ROOT, "atlas/out/missing_hashes.json")))
    pns = {r["imghash"]: r for r in csv.DictReader(open(os.path.join(ROOT, "export_final/sendout_final/sendout_v1/page_number_status.csv")))}
    pdfs = {}
    for y in os.listdir(PDFS):
        d = os.path.join(PDFS, y)
        if os.path.isdir(d):
            for f in os.listdir(d):
                if f.endswith(".pdf") and not f.startswith("._"):
                    pdfs[f] = os.path.join(d, f)
    jobs = collections.defaultdict(list)
    for h in hs:
        r = pns.get(h)
        f = os.path.basename(r["source_pdf"]) if r else ""
        if f in pdfs and r["pdf_page"].lstrip("-").isdigit():
            jobs[pdfs[f]].append((h, int(r["pdf_page"])))
    total = 0
    for k, job in enumerate(sorted(jobs.items())):
        n, err = render(job)
        total += n
        print(f"[{k + 1}/{len(jobs)}] {os.path.basename(job[0])}: {n} rendered {err}", flush=True)
    print(f"RENDER_DONE rendered {total} scans from {len(jobs)} PDFs", flush=True)

if __name__ == "__main__":
    main()
