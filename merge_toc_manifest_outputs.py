"""Merge per-task TOC-manifest outputs into one corpus manifest."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from build_toc_manifest import flatten_table_records
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
    parser = argparse.ArgumentParser(description="Merge per-task TOC manifest outputs")
    parser.add_argument("inputs", nargs="+", help="Task-output directories and/or JSON files")
    parser.add_argument("--output", required=True, help="Merged manifest JSON")
    parser.add_argument("--tables-output", help="Flat table-record JSONL")
    parser.add_argument("--tables-csv", help="Flat table-record CSV")
    parser.add_argument("--documents-jsonl", help="Optional documents JSONL")
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
    table_rows = []
    for path in sorted(set(paths)):
        payload = load_json(path)
        status = payload.get("status")
        if status == "ok" and payload.get("document"):
            doc = payload["document"]
            documents.append(doc)
            table_rows.extend(flatten_table_records(doc))
        elif status == "error":
            errors.append({
                "task_output": str(path),
                "pdf": payload.get("pdf"),
                "error": payload.get("error"),
            })

    documents.sort(key=lambda row: (row.get("year") or 0, row.get("province") or "", row.get("filename") or ""))
    table_rows.sort(key=lambda row: (
        row.get("year") or 0,
        row.get("province") or "",
        row.get("filename") or "",
        row.get("best_pdf_page") or 10**9,
        row.get("table_index") or 10**9,
    ))

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "task_output_count": len(set(paths)),
        "document_count": len(documents),
        "error_count": len(errors),
        "table_record_count": len(table_rows),
        "summary": {
            "docs_with_toc": sum(1 for doc in documents if doc.get("toc_detected")),
            "docs_with_relevant_entries": sum(
                1 for doc in documents
                if doc.get("summary", {}).get("relevant_entry_count", 0) > 0
            ),
            "topic_counts": dict(Counter(row.get("canonical_topic") for row in table_rows)),
            "relevance_counts": dict(Counter(row.get("relevance_tier") for row in table_rows)),
        },
        "documents": documents,
        "errors": errors,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as fh:
        json.dump(manifest, fh, indent=2)

    if args.documents_jsonl:
        write_jsonl(Path(args.documents_jsonl), documents)
    if args.tables_output:
        write_jsonl(Path(args.tables_output), table_rows)
    if args.tables_csv:
        write_csv(Path(args.tables_csv), table_rows)

    print(f"Merged {len(documents)} documents from {len(set(paths))} task outputs")
    print(f"Errors: {len(errors)}")
    print(f"Table records: {len(table_rows)}")
    print(f"Manifest: {output}")


if __name__ == "__main__":
    main()
