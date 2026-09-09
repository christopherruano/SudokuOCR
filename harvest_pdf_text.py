#!/usr/bin/env python3
"""Harvest every drive PDF's embedded text layer into a LOCAL cache (one gzip json per doc:
export/pdftext_cache/<saoa>.json.gz = list of per-page text). Everything downstream (autobounds, TOC,
per-town sweeps) then iterates OFFLINE — immune to the drive's constant drops. Skips docs already cached.
  python3 harvest_pdf_text.py            (drive must be mounted; safe to re-run — resumes)
"""
import os, sys, json, glob, gzip, csv

ROOT = os.path.dirname(os.path.abspath(__file__))
DRIVE = "/Volumes/Expansion/census_pdfs"
CACHE = os.path.join(ROOT, "export/pdftext_cache")


def main():
    if not os.path.isdir(DRIVE):
        sys.exit("drive not mounted")
    import fitz
    os.makedirs(CACHE, exist_ok=True)
    cov = list(csv.DictReader(open(os.path.join(ROOT, "export/document_coverage.csv"))))
    done = skip = fail = 0
    for i, r in enumerate(cov, 1):
        sa = r["saoa"]
        out = os.path.join(CACHE, sa + ".json.gz")
        if os.path.exists(out):
            skip += 1; continue
        hits = glob.glob(os.path.join(DRIVE, "*", f"*{sa}*.pdf"))
        if not hits:
            continue
        try:
            doc = fitz.open(hits[0])
            texts = []
            for p in range(doc.page_count):
                try:
                    texts.append(doc.load_page(p).get_text("text") or "")
                except Exception:
                    texts.append("")
            doc.close()
            with gzip.open(out, "wt") as f:
                json.dump(texts, f)
            done += 1
        except Exception as e:
            fail += 1
            print(f"  FAIL {sa}: {str(e)[:50]}")
        if i % 25 == 0:
            print(f"  …{i}/{len(cov)} (new {done} · cached {skip} · fail {fail})")
    print(f"harvest: new {done} · already-cached {skip} · failed {fail} -> {CACHE}")


if __name__ == "__main__":
    main()
