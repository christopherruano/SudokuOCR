#!/usr/bin/env python3
"""Turn the validated tight per-page targets (tableVII_ocr_targets.csv) into a
cluster digitization queue: contiguous PDF-page windows per volume, with the full
PDF path. Replaces the over-broad doc-span queue. Writes tableVII_tight_newocr.jsonl."""
import os, re, json, csv
from collections import defaultdict

csv.field_size_limit(10 ** 7)
ROOT = os.path.dirname(os.path.abspath(__file__))
V1 = os.path.join(ROOT, "export_final/sendout_final/sendout_v1")
DOC = re.compile(r"([0-9a-f]{16})_t")


def main():
    # full PDF path per dochash (from manifest)
    pdf_path = {}
    for line in open(os.path.join(ROOT, "export/cluster_sync/organized_manifest.jsonl")):
        try:
            j = json.loads(line)
        except Exception:
            continue
        m = DOC.search(j.get("source_image", "") or j.get("result_json", "") or "")
        f = j.get("digitized_source_file", "")
        if m and f and f != "null":
            pdf_path.setdefault(m.group(1), f)

    rows = []
    windows = 0; pages = 0
    for r in csv.DictReader(open(os.path.join(V1, "tableVII_ocr_targets.csv"))):
        if r["confidence"] not in ("high", "medium", "span_fallback") or not r["pdf_pages_to_ocr"]:
            continue
        path = pdf_path.get(r["dochash"])
        if not path:
            continue
        pp = sorted(int(x) for x in r["pdf_pages_to_ocr"].split(","))
        # contiguous windows (merge gaps <=2 to avoid tiny single-page renders)
        runs, cur = [], [pp[0]]
        for p in pp[1:]:
            if p - cur[-1] <= 2:
                cur.append(p)
            else:
                runs.append((cur[0], cur[-1])); cur = [p]
        runs.append((cur[0], cur[-1]))
        for i, (s, e) in enumerate(runs):
            rows.append({"file": path, "resolved_start_page": s, "resolved_end_page": e,
                         "table_index": 900 + i, "title": f"Table VII gap {r['source_pdf']} p{s}-{e}",
                         "dochash": r["dochash"]})
            windows += 1; pages += e - s + 1

    with open(os.path.join(V1, "tableVII_tight_newocr.jsonl"), "w") as f:
        for q in rows:
            f.write(json.dumps(q) + "\n")
    import shutil
    shutil.copy(os.path.join(V1, "tableVII_tight_newocr.jsonl"),
                os.path.join(V1, "cluster_upload", "tableVII_newocr.jsonl"))
    print(f"tight OCR queue: {windows} page-windows across {len(set(r['dochash'] for r in rows))} volumes, {pages} pages")
    print("  (vs the old broad doc-span queue) -> cluster_upload/tableVII_newocr.jsonl")


if __name__ == "__main__":
    main()
