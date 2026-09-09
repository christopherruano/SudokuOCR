#!/usr/bin/env python3
"""Mistral safety-net for the empty Table VII reads that Gemini returned 0 rows on TWICE (export/
reocr_suspect.txt). Mistral OCR is strong on the dense/transposed layouts Gemini chokes on — so if any
of these 'empty' pages is actually a real data table, Mistral recovers it. For each suspect:
  reuse the 300dpi render -> Mistral OCR -> markdown -> parse -> if it yields a REAL table (>=4 data rows),
  convert to oneshot and REPLACE the empty corpus read (backup *_oneshot.preempty.json). Otherwise it's
  confirmed non-data (title page / prose). Rebuilds affected runs.
  python3 mistral_recover_empty.py [--apply]   (default DRY: reports, places nothing)
"""
import sys, os, glob, json, csv, shutil
from concurrent.futures import ThreadPoolExecutor
_apply = "--apply" in sys.argv
sys.argv = ["x"]
import run_mistral_ocr as RM
import parse_mistral_md as P
import emit_mistral_oneshot as E
import grow_ocr as G
from dotenv import dotenv_values

ROOT = os.path.dirname(os.path.abspath(__file__))
ENV = dotenv_values(os.path.join(ROOT, ".env"))
RENDERS = os.path.join(ROOT, "export/reocr_empty_renders")
CORPUS = os.path.join(ROOT, "export/ocr_results_referenced/digitized_tables")
SUSP = os.path.join(ROOT, "export/reocr_suspect.txt")
PROG = os.path.join(ROOT, "export/mistral_recover_progress.json")


def corpus_path(ih):
    g = glob.glob(os.path.join(CORPUS, "*", f"{ih}_oneshot.json"))
    return g[0] if g else None


def nrows(secs):
    return sum(len(s.get("rows", [])) for s in (secs or []))


def main():
    from mistralai import Mistral
    client = Mistral(api_key=ENV.get("MISTRAL_API_KEY"))
    os.makedirs(RM.OUT, exist_ok=True)
    susp = [x.strip() for x in open(SUSP) if x.strip()]
    work = {w["imghash"]: w for w in json.load(open(os.path.join(ROOT, "export/incomplete_worklist.json")))}
    rps = {ih: os.path.join(RENDERS, ih + ".jpg") for ih in susp if os.path.exists(os.path.join(RENDERS, ih + ".jpg"))}
    json.dump({"state": "ocr", "total": len(rps), "recovered": 0}, open(PROG, "w"))
    print(f"mistral_recover: OCRing {len(rps)} suspect pages …")
    # 1) Mistral OCR all (threaded, modest concurrency for rate limits)
    with ThreadPoolExecutor(max_workers=6) as ex:
        list(ex.map(lambda kv: RM.ocr_one(client, kv[0], kv[1]), rps.items()))
    # 2) parse + place real tables
    recovered, confirmed_nondata, noparse = 0, 0, 0
    affected = set(); recoveries = []
    for ih in rps:
        md = os.path.join(RM.OUT, f"{ih}.mistral.md")
        if not os.path.exists(md):
            noparse += 1; continue
        try:
            secs = P.mistral_parse(md)
        except Exception:
            secs = None
        if not secs or nrows(secs) < 4:
            confirmed_nondata += 1; continue                 # Mistral also finds no real table -> non-data
        cp = corpus_path(ih)
        if not cp:
            continue
        old = json.load(open(cp))
        title = (old.get("metadata", {}) or {}).get("title", "")
        new = E.to_oneshot(secs, title, old.get("source_image", ""))
        nr = sum(len(s.get("rows", [])) for s in new.get("data", []))
        recovered += 1
        w = work.get(ih, {})
        recoveries.append((ih, w.get("saoa"), w.get("page"), nr, title[:46]))
        if _apply:
            shutil.copy(cp, cp.replace("_oneshot.json", "_oneshot.preempty.json"))
            new["source_image"] = old.get("source_image", "")
            json.dump(new, open(cp, "w"))
            if w.get("saoa"):
                affected.add(w["saoa"])
    print(f"\n=== mistral_recover ({'APPLIED' if _apply else 'DRY'}) ===")
    print(f"  REAL tables recovered: {recovered}")
    print(f"  confirmed non-data:    {confirmed_nondata}")
    print(f"  no-parse/err:          {noparse}")
    for ih, sa, pg, nr, t in recoveries:
        print(f"    + {ih[:12]} saoa={sa} p{pg} -> {nr} rows · {t}")
    if _apply:
        for sa in affected:
            try:
                G.rebuild(sa)
            except Exception as e:
                print("  rebuild", sa, "failed:", e)
    json.dump({"state": "done", "total": len(rps), "recovered": recovered,
               "confirmed_nondata": confirmed_nondata, "applied": _apply}, open(PROG, "w"))


if __name__ == "__main__":
    main()
