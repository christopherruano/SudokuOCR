#!/usr/bin/env python3
"""Build a re-OCR queue from the WORST constraint-failing assembled tables: for the top-N broken runs
(by constraint-failure count), render every member page that actually has a failure to
export/reocr_flagged_renders/{imghash}.jpg at high DPI (named by the canonical imghash so merge_reocr.py
folds the new read back). HUMAN-corrected pages are skipped. Writes export/reocr_constraint_list.txt.
  python3 build_reocr_constraint_list.py [--top N] [--dpi 300]
"""
import sys, os, re, csv, hashlib, glob
sys.argv_top = None
ROOT = os.path.dirname(os.path.abspath(__file__))
V1 = os.path.join(ROOT, "export_final/sendout_final/sendout_v1")
DRIVE = "/Volumes/Expansion/census_pdfs"
RENDERS = os.path.join(ROOT, "export/reocr_flagged_renders")
LIST = os.path.join(ROOT, "export/reocr_constraint_list.txt")
TOP = int(sys.argv[sys.argv.index("--top") + 1]) if "--top" in sys.argv else 40
DPI = int(sys.argv[sys.argv.index("--dpi") + 1]) if "--dpi" in sys.argv else 300

import importlib
A = importlib.import_module("census_review_app")
try:
    import fitz
    from PIL import Image
except Exception:
    sys.exit("need pymupdf + pillow")


def main():
    if not os.path.isdir(DRIVE):
        sys.exit("drive not mounted")
    os.makedirs(RENDERS, exist_ok=True)
    # imghash -> (dochash, page); dochash -> cluster_path
    idx, vol = {}, {}
    for r in csv.DictReader(open(os.path.join(V1, "page_number_status.csv"))):
        if r.get("imghash") and r.get("pdf_page", "").lstrip("-").isdigit():
            idx[r["imghash"]] = (r.get("dochash", ""), int(r["pdf_page"]))
    for fn, col in (("cc_volumes.csv", "cluster_path"), ("documents.csv", "source_pdf"), ("provenance.csv", "source_pdf")):
        p = os.path.join(V1, fn)
        if os.path.exists(p):
            for r in csv.DictReader(open(p)):
                cp = r.get(col)
                if cp and "census_pdfs" in cp:
                    vol.setdefault(hashlib.sha1(cp.encode()).hexdigest()[:16], cp)

    # pages we already re-OCR'd this session — skip them (Gemini gives the same read, wasted time)
    done = set()
    for f in glob.glob(os.path.join(ROOT, "export/reocr_constraint_ocr/**/*_oneshot.json"), recursive=True):
        mm = re.search(r"([0-9a-f]{40})_oneshot", os.path.basename(f))
        if mm:
            done.add(mm.group(1))

    # rank runs by failure count, take top N
    scored = []
    for run in A.assembly_runs():
        members = sorted(run.get("members", []), key=lambda m: m.get("page", 0))
        fails_by_page = []
        nf = 0
        for m in members:
            d, mih, _ = A.load_table(m.get("table_id", ""))
            if not d or (d.get("metadata", {}) or {}).get("human_corrected"):
                continue
            cnt = 0
            for s in d.get("data", []):
                _, fails = A.constraint_report(s.get("rows", []))
                cnt += len(fails)
            if cnt:
                fails_by_page.append((mih, m.get("page"), cnt))
                nf += cnt
        if nf:
            scored.append((nf, run, fails_by_page))
    scored.sort(key=lambda x: -x[0])
    chosen = scored[:TOP]

    rendered, paths, by_pdf = 0, [], {}
    for nf, run, fbp in chosen:
        for mih, page, cnt in fbp:
            if mih not in idx or mih in done:
                continue
            dh, pg = idx[mih]
            cp = vol.get(dh)
            if cp:
                by_pdf.setdefault(cp, []).append((mih, pg))
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
        for mih, pg in items:
            out = os.path.join(RENDERS, mih + ".jpg")
            paths.append(out)
            if os.path.exists(out):
                continue
            if 1 <= pg <= doc.page_count:
                try:
                    pix = doc.load_page(pg - 1).get_pixmap(matrix=fitz.Matrix(DPI / 72, DPI / 72), colorspace=fitz.csGRAY)
                    im = Image.frombytes("L", (pix.width, pix.height), pix.samples)
                    im.save(out, "JPEG", quality=88, optimize=True)
                    rendered += 1
                except Exception:
                    pass
        doc.close()
    open(LIST, "w").write("\n".join(sorted(set(paths))) + "\n")
    print(f"chosen {len(chosen)} worst tables ({sum(x[0] for x in chosen)} total failures) -> "
          f"{len(set(paths))} failing pages | rendered {rendered} new @ {DPI}dpi -> {LIST}")


if __name__ == "__main__":
    main()
