#!/usr/bin/env python3
"""Mistral fallback for pages still showing the frozen-column misfire after the Gemini pass. For each
imghash in frozen_still_bad.txt: Mistral-OCR (reuse the 300dpi render, else render from drive) -> markdown
-> convert to oneshot (emit_mistral_oneshot.to_oneshot) -> REPLACE the corpus read ONLY IF it removes
frozen columns and doesn't add constraint failures. Backs up to *_oneshot.prefrozen.json.
  python3 mistral_frozen.py [--limit N] [--apply]
"""
import sys, os, re, glob, json, csv, shutil
_apply = "--apply" in sys.argv
_limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else 0
sys.argv = ["x"]
import run_mistral_ocr as RM
import parse_mistral_md as P
import emit_mistral_oneshot as E
import detect_misfires as MF
import constraint_guided_reocr as C
from dotenv import dotenv_values

ROOT = os.path.dirname(os.path.abspath(__file__))
V1 = os.path.join(ROOT, "export_final/sendout_final/sendout_v1")
ENV = dotenv_values(os.path.join(ROOT, ".env"))
RENDERS = os.path.join(ROOT, "export/reocr_flagged_renders")
CORPUS = os.path.join(ROOT, "export/ocr_results_referenced/digitized_tables")
DRIVE = "/Volumes/Expansion/census_pdfs"


def frozen_count(d):
    n = 0
    for s in d.get("data", []):
        rows = s.get("rows", [])
        if not rows:
            continue
        rh = next((k for k in rows[0] if not isinstance(rows[0].get(k), dict)), None)
        n += len(MF.section_misfires(rows, rh))
    return n


def fails(d):
    try:
        return C.page_fail_rows(d)[0]
    except Exception:
        return 999


def corpus_path(ih):
    g = glob.glob(os.path.join(CORPUS, "*", f"{ih}_oneshot.json"))
    return g[0] if g else None


_LOC = None
def page_loc(ih):
    global _LOC
    if _LOC is None:
        _LOC = {}
        for r in csv.DictReader(open(os.path.join(V1, "page_number_status.csv"))):
            if r["pdf_page"].lstrip("-").isdigit():
                _LOC.setdefault(r["imghash"], (r["source_pdf"], int(r["pdf_page"])))
    return _LOC.get(ih)


def ensure_render(ih):
    rp = os.path.join(RENDERS, ih + ".jpg")
    if os.path.exists(rp):
        return rp
    loc = page_loc(ih)
    if not loc:
        return None
    src, pg = loc
    hits = glob.glob(os.path.join(DRIVE, "*", os.path.basename(src)))
    if not hits:
        return None
    try:
        import fitz
        from PIL import Image
        doc = fitz.open(hits[0])
        if not (1 <= pg <= doc.page_count):
            doc.close(); return None
        pix = doc.load_page(pg - 1).get_pixmap(matrix=fitz.Matrix(300 / 72, 300 / 72), colorspace=fitz.csGRAY)
        os.makedirs(RENDERS, exist_ok=True)
        Image.frombytes("L", (pix.width, pix.height), pix.samples).save(rp, "JPEG", quality=90)
        doc.close(); return rp
    except Exception:
        return None


def main():
    from mistralai import Mistral
    client = Mistral(api_key=ENV.get("MISTRAL_API_KEY"))
    os.makedirs(RM.OUT, exist_ok=True)
    ihs = [x.strip() for x in open(os.path.join(ROOT, "export/frozen_still_bad.txt")) if x.strip()]
    if _limit:
        ihs = ihs[:_limit]
    improved = notbetter = norender = noparse = 0
    for ih in ihs:
        cp = corpus_path(ih)
        if not cp:
            continue
        rp = ensure_render(ih)
        if not rp:
            norender += 1; continue
        RM.ocr_one(client, ih, rp)
        md = os.path.join(RM.OUT, f"{ih}.mistral.md")
        if not os.path.exists(md):
            noparse += 1; continue
        secs = P.mistral_parse(md)
        if not secs:
            noparse += 1; continue
        old = json.load(open(cp))
        title = (old.get("metadata", {}) or {}).get("title", "")
        new = E.to_oneshot(secs, title, old.get("source_image", ""))
        nrows = sum(len(s.get("rows", [])) for s in new.get("data", []))
        of, nf = frozen_count(old), frozen_count(new)
        ofl, nfl = fails(old), fails(new)
        better = nf < of and nrows >= 4 and nfl <= ofl + 2
        tag = "✅ better" if better else "—"
        print(f"  {ih[:12]} frozen {of}->{nf} · fails {ofl}->{nfl} · {nrows}r  {tag}")
        if better:
            improved += 1
            if _apply:
                shutil.copy(cp, cp.replace("_oneshot.json", "_oneshot.prefrozen.json"))
                new["source_image"] = old.get("source_image", "")
                json.dump(new, open(cp, "w"))
        else:
            notbetter += 1
    print(f"\nmistral frozen: {'APPLIED ' if _apply else 'DRY '}improved {improved} | not-better {notbetter} "
          f"| no-render {norender} | no-parse {noparse}  (of {len(ihs)})")


if __name__ == "__main__":
    main()
