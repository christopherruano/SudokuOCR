"""Merge per-document page-range outputs and emit digitization/review queues."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from toc_job_utils import load_json


def discover_task_outputs(root: Path):
    return sorted(Path(root).rglob("*.json"))


def write_jsonl(path: Path, rows):
    with open(path, "w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def write_csv(path: Path, rows):
    if not rows:
        with open(path, "w", newline="") as fh:
            pass
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge resolved page-range task outputs")
    parser.add_argument("inputs", nargs="+", help="Range-output directories and/or JSON files")
    parser.add_argument("--output", required=True, help="Merged JSON")
    parser.add_argument("--tables-output", help="Resolved table rows JSONL")
    parser.add_argument("--tables-csv", help="Resolved table rows CSV")
    parser.add_argument("--digitization-queue", help="Core resolved queue JSONL")
    parser.add_argument("--manual-review-queue", help="Core unresolved queue JSONL")
    parser.add_argument("--include-context", action="store_true", help="Include context rows in digitization queue")
    parser.add_argument("--allowed-topics", nargs="+",
                        help="If set, only include these canonical topics in digitization/manual-review queues")
    args = parser.parse_args()

    paths = []
    for raw in args.inputs:
        p = Path(raw)
        if p.is_dir():
            paths.extend(discover_task_outputs(p))
        elif p.is_file():
            paths.append(p)

    documents = []
    errors = []
    rows = []
    for path in sorted(set(paths)):
        payload = load_json(path)
        status = payload.get("status")
        if status == "ok" and payload.get("document_ranges"):
            doc = payload["document_ranges"]
            documents.append(doc)
            rows.extend(doc.get("resolved_tables", []))
        elif status == "error":
            errors.append({
                "task_output": str(path),
                "file": payload.get("file"),
                "error": payload.get("error"),
            })

    rows.sort(key=lambda row: (
        row.get("year") or 0,
        row.get("province") or "",
        row.get("filename") or "",
        row.get("resolved_start_page") or 10**9,
        row.get("table_index") or 10**9,
    ))

    allowed_topics = set(args.allowed_topics or [])

    digitization_queue = []
    manual_review_queue = []
    for row in rows:
        include = row.get("queue_eligible", False)
        if not args.include_context and row.get("relevance_tier") != "core":
            include = False
        if allowed_topics and row.get("canonical_topic") not in allowed_topics:
            include = False
        if include:
            digitization_queue.append({
                "file": row["file"],
                "filename": row["filename"],
                "year": row.get("year"),
                "province": row.get("province"),
                "table_index": row.get("table_index"),
                "display_identifier": row.get("display_identifier"),
                "title": row.get("title"),
                "canonical_topic": row.get("canonical_topic"),
                "query_tags": row.get("query_tags", []),
                "relevance_tier": row.get("relevance_tier"),
                "resolved_start_page": row.get("resolved_start_page"),
                "resolved_end_page": row.get("resolved_end_page"),
                "range_status": row.get("range_status"),
            })
        elif row.get("needs_manual_review") and row.get("relevance_tier") == "core":
            if allowed_topics and row.get("canonical_topic") not in allowed_topics:
                continue
            manual_review_queue.append({
                "file": row["file"],
                "filename": row["filename"],
                "year": row.get("year"),
                "province": row.get("province"),
                "table_index": row.get("table_index"),
                "display_identifier": row.get("display_identifier"),
                "title": row.get("title"),
                "canonical_topic": row.get("canonical_topic"),
                "range_status": row.get("range_status"),
                "probe_window_start": row.get("probe_window_start"),
                "probe_window_end": row.get("probe_window_end"),
            })

    merged = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "document_count": len(documents),
        "error_count": len(errors),
        "resolved_table_count": len(rows),
        "digitization_queue_count": len(digitization_queue),
        "manual_review_queue_count": len(manual_review_queue),
        "summary": {
            "range_status_counts": dict(Counter(row.get("range_status") for row in rows)),
            "canonical_topic_counts": dict(Counter(row.get("canonical_topic") for row in rows)),
        },
        "documents": documents,
        "errors": errors,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as fh:
        json.dump(merged, fh, indent=2)

    if args.tables_output:
        write_jsonl(Path(args.tables_output), rows)
    if args.tables_csv:
        write_csv(Path(args.tables_csv), rows)
    if args.digitization_queue:
        write_jsonl(Path(args.digitization_queue), digitization_queue)
    if args.manual_review_queue:
        write_jsonl(Path(args.manual_review_queue), manual_review_queue)

    print(f"Merged {len(documents)} documents")
    print(f"Resolved table rows: {len(rows)}")
    print(f"Digitization queue: {len(digitization_queue)}")
    print(f"Manual review queue: {len(manual_review_queue)}")
    print(f"Output: {output}")


if __name__ == "__main__":
    main()
