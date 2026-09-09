#!/usr/bin/env python3
"""Pre-render every context page the review app might fetch live from the drive (the page BEFORE and
AFTER each assembled run, used as greyed context cards), into export/context_renders/{dh}_p{pg}.png —
the exact path the /render endpoint caches to. After this, the review app needs ZERO drive access: the
queue scans are already in census_site/img and the context pages are pre-cached. Run with the drive
mounted. Idempotent (skips already-rendered).
  python3 cache_context_pages.py
"""
import json, csv, os, re, hashlib, glob, sys

ROOT = os.path.dirname(os.path.abspath(__file__))
V1 = os.path.join(ROOT, "export_final/sendout_final/sendout_v1")
DRIVE = "/Volumes/Expansion/census_pdfs"
CACHE = os.path.join(ROOT, "export/context_renders")
try:
    import fitz
except Exception:
    sys.exit("need pymupdf")


def main():
    if not os.path.isdir(DRIVE):
        sys.exit("drive not mounted")
    os.makedirs(CACHE, exist_ok=True)
    runs = json.load(open(os.path.join(ROOT, "export/tableVII_runs.json")))
    # dochash -> cluster_path
    vol = {}
    for fn, col in (("cc_volumes.csv", "cluster_path"), ("documents.csv", "source_pdf"), ("provenance.csv", "source_pdf")):
        p = os.path.join(V1, fn)
        if os.path.exists(p):
            for r in csv.DictReader(open(p)):
                cp = r.get(col)
                if cp and "census_pdfs" in cp:
                    vol.setdefault(hashlib.sha1(cp.encode()).hexdigest()[:16], cp)
    # collect (dochash, page) context pages = start-1 and end+1 for every run
    want = {}
    for run in runs:
        dh = run["dochash"]; s, e = run["page_range"]
        for pg in (s - 1, e + 1):
            if pg >= 1 and dh in vol:
                want[(dh, pg)] = vol[dh]
    rendered = skip = nofile = 0
    by_pdf = {}
    for (dh, pg), cp in want.items():
        out = os.path.join(CACHE, f"{dh}_p{pg}.png")
        if os.path.exists(out):
            skip += 1; continue
        by_pdf.setdefault(cp, []).append((dh, pg))
    for cp, items in by_pdf.items():
        ym = re.search(r"/(\d{4})/", cp)
        path = os.path.join(DRIVE, ym.group(1) if ym else "", os.path.basename(cp))
        if not os.path.exists(path):
            hits = glob.glob(os.path.join(DRIVE, "*", os.path.basename(cp)))
            path = hits[0] if hits else ""
        if not path:
            nofile += len(items); continue
        try:
            doc = fitz.open(path)
        except Exception:
            nofile += len(items); continue
        for dh, pg in items:
            out = os.path.join(CACHE, f"{dh}_p{pg}.png")
            if 1 <= pg <= doc.page_count:
                try:
                    pix = doc.load_page(pg - 1).get_pixmap(matrix=fitz.Matrix(150 / 72, 150 / 72))
                    pix.save(out)
                    rendered += 1
                except Exception:
                    nofile += 1
        doc.close()
    print(f"context pages: wanted {len(want)} | already cached {skip} | rendered {rendered} | unreachable {nofile}")
    print(f"  -> review app now needs no drive access for the queue + assembly views")


if __name__ == "__main__":
    main()
