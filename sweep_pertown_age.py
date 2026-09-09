#!/usr/bin/env python3
"""Sweep per-town age tables in town-census volumes (Burma-1891 style): each town/township/district has
its own one-page 'Table showing Population by Sex, Age, Civil Condition, Education…' table. Uses the PDF's
embedded TEXT LAYER (no vision, no API): greps every page for the header pattern, then UPDATES the doc's
vii_per_town verdict ranges with the complete page list (the OCR watcher picks them up automatically).
  python3 sweep_pertown_age.py <saoa> [<saoa> ...]     (drive must be mounted)
"""
import sys, os, json, csv, re, glob

ROOT = os.path.dirname(os.path.abspath(__file__))
DRIVE = "/Volumes/Expansion/census_pdfs"
HDR = re.compile(r"Table\s+showing\s+Population,?\s+by\s+Sex,?\s+Age|Population\s+by\s+Sex,\s*Age,\s*Civil", re.I)


def pdf_for(sa):
    hits = glob.glob(os.path.join(DRIVE, "*", f"*{sa}*.pdf"))
    return hits[0] if hits else None


def main():
    if not os.path.isdir(DRIVE):
        sys.exit("drive not mounted")
    import fitz
    verd = json.load(open(os.path.join(ROOT, "export/review_verdicts/documents.json")))
    for sa in sys.argv[1:]:
        path = pdf_for(sa)
        if not path:
            print(f"{sa}: PDF not found on drive"); continue
        doc = fitz.open(path)
        pages = []
        for i in range(doc.page_count):
            txt = doc.load_page(i).get_text("text") or ""
            if HDR.search(txt):
                pages.append(i + 1)
        doc.close()
        key = f"saoa:{sa}"
        old = verd.get(key, {})
        known = {int(r.split("-")[0]) for r in old.get("ranges", [])}
        allp = sorted(set(pages) | known)
        verd[key] = {**old, "v": "vii_per_town", "ranges": [f"{p}-{p}" for p in allp],
                     "structure": "per_town_scattered",
                     "note": (old.get("note", "").split(" Known pages")[0] +
                              f" FULL text-layer sweep done: {len(allp)} age-table pages.")}
        print(f"{sa}: sweep found {len(pages)} header pages (had {len(known)} known) -> total {len(allp)}")
    json.dump(verd, open(os.path.join(ROOT, "export/review_verdicts/documents.json"), "w"), indent=1)
    print("verdicts updated — the OCR watcher will pick the new pages up on its next round")


if __name__ == "__main__":
    main()
