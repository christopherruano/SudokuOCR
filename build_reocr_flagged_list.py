#!/usr/bin/env python3
"""Build the re-OCR queue from the review: tables marked 'needs re-OCR' + thin/empty tables that still
fail extraction. Each is rendered to export/reocr_flagged_renders/{imghash}.jpg (named by the CANONICAL
imghash so merge_reocr can fold the result back), from the read's source_image or a fresh drive render.
HUMAN-corrected tables are skipped. Writes export/reocr_flagged_list.txt.
  python3 build_reocr_flagged_list.py
"""
import json, csv, os, re, glob, hashlib

ROOT = os.path.dirname(os.path.abspath(__file__))
V1 = os.path.join(ROOT, "export_final/sendout_final/sendout_v1")
CORPUS = os.path.join(ROOT, "export/ocr_results_referenced/digitized_tables")
DRIVE = "/Volumes/Expansion/census_pdfs"
RENDERS = os.path.join(ROOT, "export/reocr_flagged_renders")
try:
    import fitz
    from PIL import Image
    HAVE = True
except Exception:
    HAVE = False


def main():
    # indices (built once)
    cidx = {}
    for f in glob.glob(os.path.join(CORPUS, "*", "*_oneshot.json")):
        m = re.search(r"([0-9a-f]{40})_oneshot", os.path.basename(f))
        if m:
            cidx[m.group(1)] = f
    vol = {}
    for fn, col in (("cc_volumes.csv", "cluster_path"), ("documents.csv", "source_pdf"), ("provenance.csv", "source_pdf")):
        p = os.path.join(V1, fn)
        if os.path.exists(p):
            for r in csv.DictReader(open(p)):
                cp = r.get(col)
                if cp and "census_pdfs" in cp:
                    vol.setdefault(hashlib.sha1(cp.encode()).hexdigest()[:16], cp)
    pageidx = {}
    for r in csv.DictReader(open(os.path.join(V1, "page_number_status.csv"))):
        if r.get("imghash") and r.get("pdf_page", "").lstrip("-").isdigit():
            pageidx.setdefault(r["imghash"], int(r["pdf_page"]))

    vf = os.path.join(ROOT, "export/review_verdicts/tables.json")
    verds = json.load(open(vf)) if os.path.exists(vf) else {}
    def verd(t):
        vd = verds.get(t); return (vd.get("v") if isinstance(vd, dict) else vd) or ""
    flagged = {t for t in verds if verd(t) == "reocr"}
    corrected = {t for t in verds if verd(t) == "corrected"}
    thin = set()
    for r in csv.DictReader(open(os.path.join(V1, "tableVII_imperfect.csv"))):
        tid = r.get("table_id", ""); m = re.search(r"([0-9a-f]{40})", tid); jf = cidx.get(m.group(1)) if m else None
        if jf:
            try:
                if sum(len(s.get("rows", [])) for s in (json.load(open(jf)).get("data", []) or [])) < 5:
                    thin.add(tid)
            except Exception:
                pass

    done = set()
    df = os.path.join(ROOT, "export/reocr_flagged_done.txt")
    if os.path.exists(df):
        done = {l.strip() for l in open(df) if l.strip()}
    os.makedirs(RENDERS, exist_ok=True)
    out = []
    for tid in (flagged | thin) - corrected:
        m = re.search(r"([0-9a-f]{40})", tid)
        ih = m.group(1) if m else None
        if not ih or ih in done:        # already re-OCR'd this session; only pick up NEW flags
            continue
        jf = cidx.get(ih)
        if jf:
            try:
                if (json.load(open(jf)).get("metadata", {}) or {}).get("human_corrected"):
                    continue
            except Exception:
                pass
        target = os.path.join(RENDERS, ih + ".jpg")
        if os.path.exists(target):
            out.append(target); continue
        if not HAVE:
            continue
        si = ""
        if jf:
            try:
                si = (json.load(open(jf)).get("source_image") or "").strip()
            except Exception:
                si = ""
        made = False
        if si and os.path.exists(si):
            try:
                im = Image.open(si).convert("L"); im.thumbnail((2100, 2700))
                im.save(target, "JPEG", quality=83, optimize=True); made = True
            except Exception:
                made = False
        if not made:
            dm = re.search(r"([0-9a-f]{16})_t", tid); dh = dm.group(1) if dm else None
            cp = vol.get(dh) if dh else None; pg = pageidx.get(ih)
            if cp and pg:
                ym = re.search(r"/(\d{4})/", cp)
                path = os.path.join(DRIVE, ym.group(1) if ym else "", os.path.basename(cp))
                if not os.path.exists(path):
                    hits = glob.glob(os.path.join(DRIVE, "*", os.path.basename(cp)))
                    path = hits[0] if hits else ""
                if path:
                    try:
                        doc = fitz.open(path)
                        if 1 <= pg <= doc.page_count:
                            pix = doc.load_page(pg - 1).get_pixmap(matrix=fitz.Matrix(220 / 72, 220 / 72), colorspace=fitz.csGRAY)
                            Image.frombytes("L", (pix.width, pix.height), pix.samples).save(target, "JPEG", quality=83, optimize=True)
                            made = True
                        doc.close()
                    except Exception:
                        made = False
        if made and os.path.exists(target):
            out.append(target)
    out = list(dict.fromkeys(out))
    open(os.path.join(ROOT, "export/reocr_flagged_list.txt"), "w").write("\n".join(out) + ("\n" if out else ""))
    print(f"re-OCR queue: {len(flagged)} flagged + {len(thin)} thin -> {len(out)} images named by imghash (skipped {len(corrected)} corrected)")


if __name__ == "__main__":
    main()
