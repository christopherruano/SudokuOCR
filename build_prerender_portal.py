#!/usr/bin/env python3
"""
build_prerender_portal.py — warm the /documents portal's page-render cache so browsing is instant.

Pre-renders (and thumbnails) exactly what render_saoa serves on demand:
  - the first FIRST_N pages (default 20) of EVERY drive volume  -> TOC / front-matter strips
  - the full OCR'd / candidate Table VII span of every LOCATED/PARTIAL/ASSEMBLED document -> viewer + candidate strip

Writes to export/context_renders/saoa_<saoa>_p<pg>.png (what render_saoa caches) and the matching
thumb via census_review_app.thumb_of, so cache keys line up 1:1. Resumable (skips already-cached),
parallel (WORKERS env), memory-light (one PDF open per worker).

  python3 build_prerender_portal.py          # FIRST_N=20, WORKERS=5
  FIRST_N=20 WORKERS=5 python3 build_prerender_portal.py
"""
import os, sys, csv, glob, collections
from concurrent.futures import ProcessPoolExecutor

import census_review_app as A  # reuse ROOT / DRIVE / thumb_of so cache paths match render_saoa exactly

ROOT = A.ROOT
DRIVE = A.DRIVE
CR = os.path.join(ROOT, "export/context_renders")
TRUST = {"ASSEMBLED", "PARTIAL", "LOCATED"}   # spans we trust enough to preload in full
FIRST_N = int(os.environ.get("FIRST_N", "20"))


def saoa_path(saoa):
    hits = glob.glob(os.path.join(DRIVE, "**", f"saoa.crl.{saoa}.pdf"), recursive=True)
    return hits[0] if hits else None


def warm(arg):
    saoa, pages, strip = arg   # strip = pages that appear in a thumbnail strip (warm their thumbs too)
    path = saoa_path(saoa)
    if not path:
        return (saoa, 0, "no-pdf")
    import fitz
    made = 0
    try:
        doc = fitz.open(path); n = doc.page_count
        for pg in pages:
            if not (1 <= pg <= n):
                continue
            cache = os.path.join(CR, f"saoa_{saoa}_p{pg}.png")
            if not os.path.exists(cache):
                doc.load_page(pg - 1).get_pixmap(matrix=fitz.Matrix(150 / 72, 150 / 72)).save(cache)
                made += 1
            if pg in strip:
                A.thumb_of(cache, 240)   # warm the exact thumb the strips request (?thumb=240)
        doc.close()
    except Exception as e:
        return (saoa, made, str(e)[:40])
    return (saoa, made, "")


def span_of(r, n):
    s = (r.get("vii_pages") or r.get("scan_candidate") or "").split(";")[0]
    if "-" in s:
        a, b = s.split("-")[:2]
        if a.isdigit() and b.isdigit():
            return {p for p in range(int(a), int(b) + 1) if 1 <= p <= (n or 10 ** 9)}
    return set()


def main():
    cov = list(csv.DictReader(open(os.path.join(ROOT, "export/document_coverage.csv"))))
    tasks = []
    total_pages = 0
    for r in cov:
        n = int(r.get("pages") or 0)
        first = set(range(1, min(FIRST_N, n) + 1)) if n else set()
        pages = set(first)
        cand = span_of(r, n) if r["stage"] in TRUST else set()
        pages |= cand
        if not pages:
            continue
        # thumbs only for strip pages: the TOC strip (first ~10) + the candidate strip (first ~14 of cand)
        strip = set(range(1, min(10, n) + 1))
        if cand:
            cs = min(cand)
            strip |= {p for p in cand if p <= cs + 13}
        tasks.append((r["saoa"], sorted(pages), strip))
        total_pages += len(pages)
    workers = int(os.environ.get("WORKERS", "5"))
    sys.stderr.write(f"pre-render: {len(tasks)} volumes, ~{total_pages} page-renders, {workers} workers "
                     f"(FIRST_N={FIRST_N})\n"); sys.stderr.flush()
    done = made = errs = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for (saoa, m, err) in ex.map(warm, tasks, chunksize=1):
            done += 1; made += m
            if err:
                errs += 1
            if done % 25 == 0:
                sys.stderr.write(f"  {done}/{len(tasks)} volumes · {made} pages rendered · {errs} errs\n")
                sys.stderr.flush()
    print(f"pre-render DONE: {done} volumes, {made} new page-renders warmed, {errs} errors")


if __name__ == "__main__":
    main()
