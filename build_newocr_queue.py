#!/usr/bin/env python3
"""Build the Job-B digitization queue: re-render the page windows of the AGE
multi-page tables that have missing interior pages. digitize_table_pages_task.py
renders EVERY page from resolved_start_page..resolved_end_page, so re-running the
gap docs' existing queue rows recaptures the skipped pages (then OCR the new crops).
Writes cluster_upload/newocr_queue.jsonl (queue-row schema the task expects).
"""
import os, re, json, csv
from collections import defaultdict

csv.field_size_limit(10 ** 7)
ROOT = os.path.dirname(os.path.abspath(__file__))
V1 = os.path.join(ROOT, "export_final/sendout_final/sendout_v1")
OUT = os.path.join(V1, "cluster_upload")
MAN = os.path.join(ROOT, "export/cluster_sync/organized_manifest.jsonl")
DOC = re.compile(r"([0-9a-f]{16})_t")
AGEKW = re.compile(r"\b(VI|VII|VIII|IX|X)\b|age|religion|civil|sex|married|caste", re.I)


def main():
    # gap docs that have a resolved PDF (the 98)
    gap_docs = {r["dochash"] for r in csv.DictReader(open(os.path.join(V1, "newocr_neil_pages.csv")))
                if r["source_pdf"].strip() and r["pdf_page_span"].strip()}

    # pull queue_rows for those docs' age tables from the manifest
    rows, seen = [], set()
    for line in open(MAN):
        try:
            r = json.loads(line)
        except Exception:
            continue
        m = DOC.search(r.get("source_image", "") or r.get("result_json", "") or "")
        if not m or m.group(1) not in gap_docs:
            continue
        qr = r.get("queue_row", {}) or {}
        if not (qr.get("file") and isinstance(qr.get("resolved_start_page"), int)
                and isinstance(qr.get("resolved_end_page"), int)):
            continue
        title = (qr.get("title") or "") + " " + (qr.get("canonical_topic") or "")
        if not AGEKW.search(title):
            continue
        key = (m.group(1), qr.get("table_index"), qr.get("resolved_start_page"), qr.get("resolved_end_page"))
        if key in seen:
            continue
        seen.add(key)
        rows.append(qr)

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "newocr_queue.jsonl"), "w") as f:
        for qr in rows:
            f.write(json.dumps(qr) + "\n")

    docs = len(set(DOC.search(q["file"]).group(1) if DOC.search(q.get("file", "")) else q.get("file") for q in rows))
    pages = sum(int(q["resolved_end_page"]) - int(q["resolved_start_page"]) + 1 for q in rows)
    print(f"newocr_queue.jsonl: {len(rows)} table windows across {len(gap_docs)} gap docs, ~{pages} pages to re-render")


if __name__ == "__main__":
    main()
