#!/usr/bin/env python3
"""Build the MISSING-PAGE OCR queue — the real path to COMPLETE (not just internally
consistent) Table VII. We only digitized ~18% of each volume's pages, so multi-page
Table VII have gaps that are genuinely un-OCR'd pages. For each volume:
  1. the Table VII printed pages we HAVE (from page_reads_fixed, now page-ordered),
  2. group into contiguous blocks (a >GAP jump = a different table/section boundary),
  3. missing pages = gaps WITHIN each block (likely Table VII continuation),
  4. learn the printed->PDF offset for the volume (from tables with both a read printed
     page and a manifest resolved PDF page) and map missing printed -> PDF pages.
Writes missing_page_ocr_queue.csv (per volume: source_pdf, missing printed pages, PDF
pages, offset confidence) + missing_pages.jsonl for the cluster OCR runner."""
import os, re, json, csv, statistics
from collections import defaultdict

csv.field_size_limit(10 ** 7)
ROOT = os.path.dirname(os.path.abspath(__file__))
V1 = os.path.join(ROOT, "export_final/sendout_final/sendout_v1")
GAP = 2                                   # within-block tolerance (gaps <=GAP+? are fillable)
DOC = re.compile(r"([0-9a-f]{16})_t(\d+)")
VII = re.compile(r"\bVI[IL]\b|imperial table\s*vi|age,?\s*sex|civil\s*condition", re.I)


def main():
    # source PDF + place/year per dochash
    pdf, meta = {}, {}
    for r in csv.DictReader(open(os.path.join(V1, "documents.csv"))):
        if r.get("source_pdf"):
            pdf[r["dochash"]] = r["source_pdf"]
        meta[r["dochash"]] = {"year": r.get("year", ""), "province": r.get("region", "")}

    # printed page + family per table (page_reads_fixed = accurate after page-ordering)
    reads = json.load(open(os.path.join(V1, "page_reads_fixed.json")))
    doc_vii_pages = defaultdict(set)      # dochash -> set of Table VII printed pages we have
    doc_printed = defaultdict(dict)       # dochash -> {table_id: printed_page} (for offset)
    for r in reads:
        m = DOC.search(r.get("table_id", ""))
        pn = r.get("page_number")
        if not m or not (isinstance(pn, int) or str(pn).isdigit()):
            continue
        pn = int(pn); dh = m.group(1)
        doc_printed[dh][r["table_id"]] = pn
        fam = (r.get("family", "") or ""); ttl = (r.get("table_number", "") or "")
        if VII.search(fam) or VII.search(ttl) or VII.search(r.get("section", "") or ""):
            doc_vii_pages[dh].add(pn)

    # printed->PDF offset per dochash from the organized manifest (resolved_start_page)
    DT = re.compile(r"([0-9a-f]{16})_t(\d+)")
    pdf_start = {}                        # dochash_tNNNN -> resolved PDF start page
    mani = os.path.join(ROOT, "export/cluster_sync/organized_manifest.jsonl")
    if os.path.exists(mani):
        for line in open(mani):
            try:
                j = json.loads(line)
            except Exception:
                continue
            mm = DT.search(j.get("source_image", "") or j.get("result_json", "") or "")
            s = (j.get("queue_row", {}) or {}).get("resolved_start_page")
            if mm and isinstance(s, int):
                pdf_start[f"{mm.group(1)}_t{mm.group(2)}"] = s

    def offset_for(dh):
        diffs = []
        for tid, ppage in doc_printed[dh].items():
            mm = DT.search(tid)
            key = f"{mm.group(1)}_t{mm.group(2)}" if mm else None
            if key in pdf_start:
                diffs.append(pdf_start[key] - ppage)
        if not diffs:
            return None, 0
        return int(statistics.median(diffs)), len(diffs)

    rows, jsonl = [], []
    tot_missing = 0
    for dh, pages in doc_vii_pages.items():
        if len(pages) < 2 or dh not in pdf:
            continue
        sp = sorted(pages)
        # contiguous blocks (gap <= GAP+1 stays in block; bigger = new table boundary)
        blocks, cur = [], [sp[0]]
        for p in sp[1:]:
            if p - cur[-1] <= GAP + 1:
                cur.append(p)
            else:
                blocks.append(cur); cur = [p]
        blocks.append(cur)
        missing = []
        for b in blocks:
            for p in range(b[0], b[-1] + 1):
                if p not in pages:
                    missing.append(p)
        if not missing:
            continue
        off, n = offset_for(dh)
        conf = "high" if n >= 3 else "low" if n >= 1 else "none"
        pdf_pages = [p + off for p in missing] if off is not None else []
        tot_missing += len(missing)
        rows.append({"dochash": dh, "source_pdf": pdf[dh], "n_missing": len(missing),
                     "missing_printed_pages": ";".join(map(str, missing)),
                     "pdf_pages": ";".join(map(str, pdf_pages)),
                     "offset": off if off is not None else "", "offset_confidence": conf})
        # newocr_queue.jsonl: ONE line per gappy block (the sbatch renders the full
        # PDF range, capturing the missing interior pages) — plugs into newocr_digitize.sbatch
        if off is not None and conf in ("high", "low"):
            for b in blocks:
                if any(p not in pages for p in range(b[0], b[-1] + 1)):   # block has a gap
                    jsonl.append({"file": pdf[dh], "filename": os.path.basename(pdf[dh]),
                                  "year": meta.get(dh, {}).get("year", ""),
                                  "province": meta.get(dh, {}).get("province", ""),
                                  "dochash": dh, "title": "Table VII (gap fill)",
                                  "resolved_start_page": b[0] + off, "resolved_end_page": b[-1] + off,
                                  "printed_start": b[0], "printed_end": b[-1],
                                  "range_status": f"gap_fill_{conf}"})

    rows.sort(key=lambda r: (-{"high": 2, "low": 1, "none": 0}[r["offset_confidence"]], -r["n_missing"]))
    with open(os.path.join(V1, "missing_page_ocr_queue.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dochash", "source_pdf", "n_missing",
                                          "missing_printed_pages", "pdf_pages", "offset", "offset_confidence"])
        w.writeheader(); [w.writerow(r) for r in rows]
    with open(os.path.join(V1, "missing_pages.jsonl"), "w") as f:
        for j in jsonl:
            f.write(json.dumps(j) + "\n")
    from collections import Counter
    conf = Counter(r["offset_confidence"] for r in rows)
    fillable = sum(r["n_missing"] for r in rows if r["offset_confidence"] in ("high", "low"))
    print(f"volumes with missing Table VII pages: {len(rows)} | total missing pages: {tot_missing}")
    print(f"  offset confidence: {dict(conf)}")
    print(f"  fillable now (have source PDF + offset): {fillable} pages across "
          f"{sum(1 for r in rows if r['offset_confidence'] in ('high','low'))} volumes")
    print("wrote missing_page_ocr_queue.csv + missing_pages.jsonl (cluster: render PDF pages -> OCR -> ingest)")


if __name__ == "__main__":
    main()
