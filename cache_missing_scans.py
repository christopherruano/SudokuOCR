#!/usr/bin/env python3
"""Make the deployment self-contained: every scan referenced by census_site must be a real file in
census_site/img (the static site can't render live from the drive like the review app can). This finds
every img/<imghash> the built pages reference, renders the missing ones from the drive PDF, and writes
them as compressed grayscale JPEGs (matching the build). Run once with the drive mounted.
  python3 cache_missing_scans.py
"""
import os, re, glob, csv, hashlib, sys, json

ROOT = os.path.dirname(os.path.abspath(__file__))
SITE = os.path.join(ROOT, "census_site")
IMG = os.path.join(SITE, "img")
V1 = os.path.join(ROOT, "export_final/sendout_final/sendout_v1")
DRIVE = "/Volumes/Expansion/census_pdfs"
try:
    import fitz
    from PIL import Image
except Exception:
    sys.exit("need pymupdf + pillow")


def main():
    if not os.path.isdir(DRIVE):
        sys.exit("drive not mounted at /Volumes/Expansion/census_pdfs — reconnect it first")
    # 1. every imghash the built site references
    refs = set()
    for f in glob.glob(os.path.join(SITE, "**", "*.html"), recursive=True):
        try:
            for ih in re.findall(r"img/([0-9a-f]{40})\.(?:jpg|png)", open(f, encoding="utf-8", errors="ignore").read()):
                refs.add(ih)
        except Exception:
            pass
    # also every assembly run member — the assembled view skips uncached members, so render them too
    rf = os.path.join(ROOT, "export/tableVII_runs.json")
    if os.path.exists(rf):
        for run in json.load(open(rf)):
            for m in run.get("members", []):
                g = re.search(r"([0-9a-f]{40})", m.get("table_id", ""))
                if g:
                    refs.add(g.group(1))
    present = set(os.path.splitext(x)[0] for x in os.listdir(IMG))
    missing = refs - present
    print(f"  referenced scans: {len(refs)} | already local: {len(refs & present)} | MISSING: {len(missing)}")
    if not missing:
        print("  nothing to render — deployment already self-contained.")
        return
    # 2. imghash -> (dochash, pdf_page) from the master
    idx = {}
    for r in csv.DictReader(open(os.path.join(V1, "page_number_status.csv"))):
        ih = r.get("imghash", "")
        if ih in missing and r.get("pdf_page", "").lstrip("-").isdigit():
            idx[ih] = (r.get("dochash", ""), int(r["pdf_page"]))
    # 3. dochash -> cluster_path
    vol = {}
    for fn, col in (("cc_volumes.csv", "cluster_path"), ("documents.csv", "source_pdf"), ("provenance.csv", "source_pdf")):
        p = os.path.join(V1, fn)
        if os.path.exists(p):
            for r in csv.DictReader(open(p)):
                cp = r.get(col)
                if cp and "census_pdfs" in cp:
                    vol.setdefault(hashlib.sha1(cp.encode()).hexdigest()[:16], cp)
    # 4. render the missing ones from the drive
    rendered = nodrv = noidx = 0
    by_pdf = {}
    for ih in missing:
        if ih not in idx:
            noidx += 1; continue
        dh, pg = idx[ih]
        cp = vol.get(dh)
        if not cp:
            noidx += 1; continue
        by_pdf.setdefault(cp, []).append((ih, pg))
    for cp, items in by_pdf.items():
        ym = re.search(r"/(\d{4})/", cp)
        path = os.path.join(DRIVE, ym.group(1) if ym else "", os.path.basename(cp))
        if not os.path.exists(path):
            hits = glob.glob(os.path.join(DRIVE, "*", os.path.basename(cp)))
            path = hits[0] if hits else ""
        if not path:
            nodrv += len(items); continue
        try:
            doc = fitz.open(path)
        except Exception:
            nodrv += len(items); continue
        for ih, pg in items:
            out = os.path.join(IMG, ih + ".jpg")
            if os.path.exists(out):
                continue
            if 1 <= pg <= doc.page_count:
                try:
                    pix = doc.load_page(pg - 1).get_pixmap(matrix=fitz.Matrix(200 / 72, 200 / 72), colorspace=fitz.csGRAY)
                    im = Image.frombytes("L", (pix.width, pix.height), pix.samples)
                    im.thumbnail((1800, 2400))
                    im.save(out, "JPEG", quality=78, optimize=True)
                    rendered += 1
                except Exception:
                    pass
        doc.close()
    print(f"  rendered {rendered} missing scans into census_site/img | no-master-page {noidx} | no-drive {nodrv}")
    still = sum(1 for ih in missing if not os.path.exists(os.path.join(IMG, ih + ".jpg")))
    print(f"  still missing after pass: {still} (these have no recoverable page/volume — will be absent on the site)")


if __name__ == "__main__":
    main()
