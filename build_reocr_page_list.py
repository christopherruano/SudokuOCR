#!/usr/bin/env python3
"""Build the re-OCR render list from the pages the reviewer flagged in the app
(export/reocr_page_flags.txt — one imghash per line, written by the '✗ re-OCR just this page' button).
Each flagged page is rendered high-res from the drive PDF to export/reocr_flagged_renders/{imghash}.jpg
(named by the canonical imghash so merge_reocr folds the new read back). Writes export/reocr_page_list.txt.
  python3 build_reocr_page_list.py [--dpi 300]
"""
import os, re, csv, hashlib, glob, sys

ROOT = os.path.dirname(os.path.abspath(__file__))
V1 = os.path.join(ROOT, "export_final/sendout_final/sendout_v1")
DRIVE = "/Volumes/Expansion/census_pdfs"
RENDERS = os.path.join(ROOT, "export/reocr_flagged_renders")
FLAGS = os.path.join(ROOT, "export/reocr_page_flags.txt")
LIST = os.path.join(ROOT, "export/reocr_page_list.txt")
DPI = int(sys.argv[sys.argv.index("--dpi") + 1]) if "--dpi" in sys.argv else 300
try:
    import fitz
    from PIL import Image
except Exception:
    sys.exit("need pymupdf + pillow")


def main():
    if not os.path.exists(FLAGS):
        open(LIST, "w").write("")
        print("no flagged pages (export/reocr_page_flags.txt absent)"); return
    ihs = set(x for x in open(FLAGS).read().split() if re.fullmatch(r"[0-9a-f]{40}", x))
    if not ihs:
        open(LIST, "w").write("")
        print("no valid flagged imghashes"); return
    have_drive = os.path.isdir(DRIVE)
    os.makedirs(RENDERS, exist_ok=True)
    idx, vol = {}, {}
    for r in csv.DictReader(open(os.path.join(V1, "page_number_status.csv"))):
        if r.get("imghash") in ihs and r.get("pdf_page", "").lstrip("-").isdigit():
            idx[r["imghash"]] = (r.get("dochash", ""), int(r["pdf_page"]))
    for fn, col in (("cc_volumes.csv", "cluster_path"), ("documents.csv", "source_pdf"), ("provenance.csv", "source_pdf")):
        p = os.path.join(V1, fn)
        if os.path.exists(p):
            for r in csv.DictReader(open(p)):
                cp = r.get(col)
                if cp and "census_pdfs" in cp:
                    vol.setdefault(hashlib.sha1(cp.encode()).hexdigest()[:16], cp)
    by_pdf, noidx = {}, 0
    for ih in ihs:
        if ih not in idx or idx[ih][0] not in vol:
            noidx += 1; continue
        dh, pg = idx[ih]
        by_pdf.setdefault(vol[dh], []).append((ih, pg))
    rendered = 0
    if have_drive:                                   # fresh high-res render from the drive PDF
        for cp, items in by_pdf.items():
            ym = re.search(r"/(\d{4})/", cp)
            path = os.path.join(DRIVE, ym.group(1) if ym else "", os.path.basename(cp))
            if not os.path.exists(path):
                hits = glob.glob(os.path.join(DRIVE, "*", os.path.basename(cp)))
                path = hits[0] if hits else ""
            if not path:
                continue
            try:
                doc = fitz.open(path)
            except Exception:
                continue
            for ih, pg in items:
                out = os.path.join(RENDERS, ih + ".jpg")
                if os.path.exists(out):
                    continue
                if 1 <= pg <= doc.page_count:
                    try:
                        pix = doc.load_page(pg - 1).get_pixmap(matrix=fitz.Matrix(DPI / 72, DPI / 72), colorspace=fitz.csGRAY)
                        Image.frombytes("L", (pix.width, pix.height), pix.samples).save(out, "JPEG", quality=88, optimize=True)
                        rendered += 1
                    except Exception:
                        pass
            doc.close()
    # pick the best image per flagged page: fresh high-res render, else the cached site scan (lower-res)
    CACHE = os.path.join(ROOT, "census_site/img")
    paths, fromcache, missing = [], 0, 0
    for ih in ihs:
        r = os.path.join(RENDERS, ih + ".jpg")
        c = os.path.join(CACHE, ih + ".jpg")
        if os.path.exists(r):
            paths.append(r)
        elif os.path.exists(c):
            paths.append(c); fromcache += 1
        else:
            missing += 1
    open(LIST, "w").write("\n".join(sorted(set(paths))) + ("\n" if paths else ""))
    src = "drive 300dpi" if have_drive else "CACHE only (drive down — lower-res, reconnect for best)"
    print(f"flagged: {len(ihs)} | in list: {len(set(paths))} ({fromcache} from cache) | rendered: {rendered} "
          f"| missing img: {missing} | source: {src} -> {LIST}")


if __name__ == "__main__":
    main()
