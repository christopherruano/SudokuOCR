#!/usr/bin/env python3
"""Render Table VII gap pages from the flash-drive PDFs, memory-safely (one PDF open
at a time, page-by-page, free each pixmap). Priority = smallest bounding_gap first
(high-confidence Table VII continuation). Writes PNGs + an OCR list for run_local_reocr.sh.
  python3 build_flashdrive_render.py [--priority] [--limit N]
"""
import csv, os, re, sys, glob
from collections import defaultdict

DRIVE = "/Volumes/Expansion/census_pdfs"
L = "export_final/pdf_recovery_incoming_2026-06-16_no_gaps_ledger_v2_refined"
OUT = "export/flashdrive_renders"
LISTF = "export/flashdrive_render_list.txt"
PRIORITY = "--priority" in sys.argv
LIMIT = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None

try:
    import fitz
except ImportError:
    sys.exit("need pymupdf")


def done_set():
    done = set()
    for d in ("export/money_renders_ocr", "export/money_gapfill_ocr", "export/flashdrive_ocr"):
        for f in glob.glob(f"{d}/**/*_oneshot.json", recursive=True):
            m = re.search(r"(\d+)_p(\d+)", os.path.basename(f))
            if m:
                done.add((m.group(1), m.group(2)))
    for f in glob.glob("export_final/gemini_no_gap_ocr_2026-06-16_v2_best/**/*_oneshot.json", recursive=True):
        m = re.search(r"(\d+)_p(\d+)", os.path.basename(f))
        if m:
            done.add((m.group(1), m.group(2)))
    return done


def main():
    done = done_set()
    rows = [r for r in csv.DictReader(open(f"{L}/source_pages_still_need_render.csv"))
            if (r["saoa_id"], r["pdf_page"]) not in done]
    if PRIORITY:
        rows = [r for r in rows if r.get("bounding_gap", "").isdigit() and int(r["bounding_gap"]) <= 4]
    rows.sort(key=lambda r: (int(r.get("bounding_gap", "999") or 999), r["source_pdf"], int(r["pdf_page"])))
    if LIMIT:
        rows = rows[:LIMIT]
    bypdf = defaultdict(list)
    for r in rows:
        bypdf[(r["year"], r["source_pdf"])].append(r)

    os.makedirs(OUT, exist_ok=True)
    rendered = skipped = missing_pdf = 0
    lst = open(LISTF, "w")
    for (year, pdf), rs in bypdf.items():
        path = os.path.join(DRIVE, year, pdf)
        if not os.path.exists(path):
            missing_pdf += 1; continue
        try:
            doc = fitz.open(path)
        except Exception:
            missing_pdf += 1; continue
        n = doc.page_count
        for r in rs:
            pg = int(r["pdf_page"]); sa = r["saoa_id"]
            outpng = os.path.join(OUT, f"{sa}_p{pg}.png")
            if os.path.exists(outpng):
                skipped += 1
            elif 1 <= pg <= n:
                try:
                    pix = doc.load_page(pg - 1).get_pixmap(matrix=fitz.Matrix(300 / 72, 300 / 72))
                    pix.save(outpng); pix = None; rendered += 1
                except Exception:
                    continue
            else:
                continue
            lst.write(os.path.abspath(outpng) + "\n")
        doc.close()
        if rendered and rendered % 200 == 0:
            print(f"  rendered {rendered}...", flush=True)
    lst.close()
    print(f"rendered {rendered} | already-present {skipped} | missing PDFs {missing_pdf}")
    print(f"OCR list: {LISTF}  ({sum(1 for _ in open(LISTF))} pages)")
    print(f"Next: LIST={LISTF} WORKERS=2 RESULTS=export/flashdrive_ocr bash run_local_reocr.sh")


if __name__ == "__main__":
    main()
