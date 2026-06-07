#!/usr/bin/env python3
"""Build the cluster queue to FINISH Neil's goal tables (multi-page AGE tables:
general population age structure, age-by-religion, main castes; Imperial Tables
VII/VIII/IX). Two parts:
  A) RE-OCR  - flagged keeper AGE tables not yet re-done -> reocr_neil_images.txt
  B) NEW OCR - AGE multi-page tables with missing interior pages -> newocr_neil_pages.csv
               (per source PDF: the PDF page span to re-extract so the skipped
               pages get captured; gaps are interior to the captured span)
Excludes tables the latest re-OCR run already completed (recent results delta).
"""
import os, re, json, glob, time, csv
from collections import Counter, defaultdict

csv.field_size_limit(10 ** 7)
ROOT = os.path.dirname(os.path.abspath(__file__))
V1 = os.path.join(ROOT, "export_final/sendout_final/sendout_v1")
NEW = os.path.join(ROOT, "export/ocr_results_referenced/digitized_tables")
DT = re.compile(r"([0-9a-f]{16}_t\d+)")
DOC = re.compile(r"([0-9a-f]{16})_t\d+")
AGE = {"age_civil_condition", "age_sex_population", "age_by_religion", "proportional_age",
       "caste_class_age", "single_year_age", "age_sex_civil_condition", "infirmity_age"}


def tkey(s):
    m = DT.search(s or ""); return m.group(1) if m else ""


def main():
    # tables already re-OCR'd in the latest run (don't re-queue)
    now = time.time()
    done = set()
    for p in glob.glob(os.path.join(NEW, "*/*_oneshot.json")):
        if now - os.path.getmtime(p) < 2 * 86400:
            k = tkey(p)
            if k:
                done.add(k)

    # ---- A) re-OCR remaining flagged AGE keeper tables ----
    kp = list(csv.DictReader(open(os.path.join(V1, "reocr_keeper_manifest.csv"))))
    remain = [r for r in kp if r["table_family"] in AGE and tkey(r["source_image"]) not in done]
    with open(os.path.join(V1, "reocr_neil_manifest.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=kp[0].keys()); w.writeheader(); [w.writerow(r) for r in remain]
    imgs = list(dict.fromkeys(r["source_image"] for r in remain if r["source_image"]))
    open(os.path.join(V1, "reocr_neil_images.txt"), "w").write("\n".join(imgs) + "\n")

    # ---- B) new-OCR missing pages of AGE multi-page tables ----
    # PDF page span per doc from the digitization manifest (resolved_start/end_page)
    span = defaultdict(lambda: [10 ** 9, -1]); pdf = {}
    for mf in glob.glob(os.path.join(ROOT, "export/**/organized_manifest.jsonl"), recursive=True):
        for line in open(mf):
            try:
                r = json.loads(line)
            except Exception:
                continue
            m = DOC.search(r.get("source_image", "") or r.get("result_json", "") or "")
            if not m:
                continue
            dh = m.group(1); qr = r.get("queue_row", {}) or {}
            s, e = qr.get("resolved_start_page"), qr.get("resolved_end_page")
            if isinstance(s, int) and isinstance(e, int):
                span[dh][0] = min(span[dh][0], s); span[dh][1] = max(span[dh][1], e)
                pdf[dh] = qr.get("filename", "") or pdf.get(dh, "")

    fam = defaultdict(Counter)
    for r in csv.DictReader(open(os.path.join(V1, "provenance.csv"))):
        if r["dochash"]:
            fam[r["dochash"]][r["family"]] += 1

    st = list(csv.DictReader(open(os.path.join(V1, "stitched_tables.csv"))))
    bydoc = defaultdict(lambda: {"missing": set(), "tables": set()})
    for r in st:
        if not r["missing_pages"]:
            continue
        df = (fam[r["dochash"]].most_common(1) or [("", 0)])[0][0]
        if df in AGE or r["table_number"] in ("VI", "VII", "VIII", "IX"):
            bydoc[r["dochash"]]["missing"].update(int(x) for x in r["missing_pages"].split(","))
            bydoc[r["dochash"]]["tables"].add(r["table_number"])

    rows = []
    for dh, d in bydoc.items():
        sp = span.get(dh, [None, None])
        rows.append({"dochash": dh, "source_pdf": pdf.get(dh, ""),
                     "pdf_page_span": f"{sp[0]}-{sp[1]}" if sp[0] and sp[0] < 10 ** 9 else "",
                     "tables": ",".join(sorted(d["tables"])[:8]),
                     "n_missing": len(d["missing"]),
                     "missing_printed_pages": ",".join(map(str, sorted(d["missing"])))})
    rows.sort(key=lambda x: -x["n_missing"])
    with open(os.path.join(V1, "newocr_neil_pages.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dochash", "source_pdf", "pdf_page_span", "tables",
                                          "n_missing", "missing_printed_pages"])
        w.writeheader(); [w.writerow(r) for r in rows]

    print(f"A) RE-OCR remaining AGE tables: {len(remain)} tables -> {len(imgs)} images (reocr_neil_images.txt)")
    print(f"   by tier: {dict(Counter(r['tier'] for r in remain))}")
    print(f"B) NEW-OCR AGE pages: {sum(r['n_missing'] for r in rows)} missing pages across {len(rows)} PDFs (newocr_neil_pages.csv)")
    print(f"   PDFs with a resolved page span: {sum(1 for r in rows if r['pdf_page_span'])}/{len(rows)}")


if __name__ == "__main__":
    main()
