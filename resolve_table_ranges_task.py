"""Resolve page ranges for one document's table records.

This converts table-level manifest entries into explicit PDF page windows.
It stays conservative: unresolved starts/ends are marked explicitly.
"""

from __future__ import annotations

import argparse
import json
import socket
import statistics
import traceback
from datetime import datetime, timezone
from pathlib import Path

from toc_job_utils import (
    is_valid_json_output,
    jsonl_iter,
    load_json,
    parse_page_label,
    stable_doc_id,
    write_json_atomic,
)


def load_document(args) -> dict:
    if args.document_json:
        return load_json(Path(args.document_json))
    if args.documents_jsonl:
        for idx, row in enumerate(jsonl_iter(Path(args.documents_jsonl))):
            if idx == args.index:
                return row
        raise IndexError(f"Index {args.index} out of range for {args.documents_jsonl}")
    manifest = load_json(Path(args.manifest))
    documents = manifest.get("documents", [])
    if args.index < 0 or args.index >= len(documents):
        raise IndexError(f"Index {args.index} out of range for {len(documents)} documents")
    return documents[args.index]


def build_output_path(args, document: dict) -> Path:
    file_path = Path(document["file"])
    if args.output:
        return Path(args.output)
    if not args.output_dir:
        raise ValueError("Provide either --output or --output-dir")
    if args.index is not None and args.index >= 0:
        return Path(args.output_dir) / f"{args.index:06d}_{stable_doc_id(file_path)}.json"
    return Path(args.output_dir) / f"{stable_doc_id(file_path)}.json"


def doc_page_offset(entries) -> dict:
    offsets = []
    for entry in entries:
        printed = parse_page_label(entry.get("printed_page_label"))
        best = entry.get("best_pdf_page")
        if printed is not None and isinstance(best, int) and best > 0:
            offsets.append(best - printed)
    if not offsets:
        return {"value": None, "evidence_count": 0}
    return {
        "value": int(round(statistics.median(offsets))),
        "evidence_count": len(offsets),
        "min": min(offsets),
        "max": max(offsets),
    }


def estimate_start(entry: dict, offset_info: dict) -> dict:
    body_hits = entry.get("body_page_hits") or []
    body_pages = sorted({
        hit["pdf_page"]
        for hit in body_hits
        if isinstance(hit.get("pdf_page"), int) and hit["pdf_page"] > 0
    })
    if body_pages:
        return {
            "start_page": body_pages[0],
            "observed_last_page": body_pages[-1],
            "start_source": "body_header",
            "start_confidence": "high",
        }

    best = entry.get("best_pdf_page")
    if isinstance(best, int) and best > 0:
        return {
            "start_page": best,
            "observed_last_page": best,
            "start_source": "best_pdf_page",
            "start_confidence": "high",
        }

    printed = parse_page_label(entry.get("printed_page_label"))
    if printed is not None and offset_info.get("value") is not None:
        start = printed + offset_info["value"]
        if start > 0:
            conf = "medium" if offset_info.get("evidence_count", 0) >= 2 else "low"
            return {
                "start_page": start,
                "observed_last_page": start,
                "start_source": "printed_plus_offset",
                "start_confidence": conf,
            }

    return {
        "start_page": None,
        "observed_last_page": None,
        "start_source": None,
        "start_confidence": "none",
    }


def resolve_ranges_for_document(document: dict) -> dict:
    entries = list(document.get("entries", []))
    offset_info = doc_page_offset(entries)

    rows = []
    for index, entry in enumerate(entries, 1):
        row = {
            "file": document["file"],
            "filename": document["filename"],
            "document_title": document.get("title"),
            "year": document.get("year"),
            "province": document.get("province"),
            "pages": document.get("pages"),
            "table_index": index,
            "entry_type": entry.get("entry_type"),
            "display_identifier": entry.get("display_identifier"),
            "identifier": entry.get("identifier"),
            "title": entry.get("title"),
            "canonical_topic": entry.get("canonical_topic"),
            "query_tags": entry.get("query_tags", []),
            "relevance_tier": entry.get("relevance_tier"),
            "age_pipeline_relevant": entry.get("age_pipeline_relevant"),
            "printed_page_label": entry.get("printed_page_label"),
            "toc_pdf_page": entry.get("toc_pdf_page"),
            "best_pdf_page": entry.get("best_pdf_page"),
            "body_page_hits": entry.get("body_page_hits", []),
            "confidence": entry.get("confidence"),
        }
        row.update(estimate_start(entry, offset_info))
        rows.append(row)

    def sort_key(item):
        return (
            item["start_page"] if item["start_page"] is not None else 10**9,
            parse_page_label(item.get("printed_page_label")) or 10**9,
            item["table_index"],
        )

    rows_sorted = sorted(rows, key=sort_key)

    for idx, row in enumerate(rows_sorted):
        next_start = None
        for later in rows_sorted[idx + 1:]:
            if later["start_page"] is not None and later["start_page"] > (row["start_page"] or -1):
                next_start = later["start_page"]
                break

        start = row["start_page"]
        observed_last = row["observed_last_page"]
        pages = row.get("pages") or 0

        if start is None:
            row["resolved_start_page"] = None
            row["resolved_end_page"] = None
            row["range_status"] = "unresolved_missing_start"
            row["needs_manual_review"] = row.get("age_pipeline_relevant", False)
            row["queue_eligible"] = False
            row["probe_window_start"] = None
            row["probe_window_end"] = None
            continue

        if next_start is not None:
            end = next_start - 1
            if observed_last and observed_last > end:
                row["resolved_start_page"] = start
                row["resolved_end_page"] = observed_last
                row["range_status"] = "unresolved_overlapping_headers"
                row["needs_manual_review"] = True
                row["queue_eligible"] = False
                row["probe_window_start"] = start
                row["probe_window_end"] = min(max(start, observed_last), pages or observed_last)
            else:
                end = max(start, end)
                row["resolved_start_page"] = start
                row["resolved_end_page"] = end
                if row["start_source"] == "printed_plus_offset":
                    row["range_status"] = "resolved_from_printed_offset"
                elif observed_last and observed_last > start:
                    row["range_status"] = "resolved_from_body_hits"
                else:
                    row["range_status"] = "resolved_between_headers"
                row["needs_manual_review"] = False
                row["queue_eligible"] = True
                row["probe_window_start"] = start
                row["probe_window_end"] = end
            continue

        if observed_last and observed_last > start:
            row["resolved_start_page"] = start
            row["resolved_end_page"] = observed_last
            row["range_status"] = "resolved_from_body_hits"
            row["needs_manual_review"] = False
            row["queue_eligible"] = True
            row["probe_window_start"] = start
            row["probe_window_end"] = observed_last
            continue

        row["resolved_start_page"] = start
        row["resolved_end_page"] = None
        row["range_status"] = "unresolved_missing_end"
        row["needs_manual_review"] = row.get("age_pipeline_relevant", False)
        row["queue_eligible"] = False
        row["probe_window_start"] = start
        row["probe_window_end"] = min(start + 1, pages) if pages else start + 1

    rows_sorted.sort(key=lambda row: row["table_index"])
    return {
        "file": document["file"],
        "filename": document["filename"],
        "year": document.get("year"),
        "province": document.get("province"),
        "title": document.get("title"),
        "pages": document.get("pages"),
        "document_page_offset": offset_info,
        "resolved_tables": rows_sorted,
        "summary": {
            "table_count": len(rows_sorted),
            "resolved_count": sum(1 for row in rows_sorted if row["queue_eligible"]),
            "manual_review_count": sum(1 for row in rows_sorted if row["needs_manual_review"]),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Resolve page ranges for one document")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--manifest", help="Merged manifest JSON")
    src.add_argument("--documents-jsonl", help="Documents JSONL")
    src.add_argument("--document-json", help="Single document JSON")
    parser.add_argument("--index", type=int, default=-1, help="0-based document index")
    parser.add_argument("--output")
    parser.add_argument("--output-dir")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    document = load_document(args)
    output_path = build_output_path(args, document)
    if args.skip_existing and is_valid_json_output(output_path):
        print(f"SKIP {document['file']} -> {output_path}")
        return

    payload = {
        "status": "error",
        "file": document["file"],
        "output": str(output_path),
        "task": {
            "index": None if args.index < 0 else args.index,
            "host": socket.gethostname(),
            "started_at": datetime.now(timezone.utc).isoformat(),
        },
    }
    try:
        resolved = resolve_ranges_for_document(document)
        payload.update({
            "status": "ok",
            "document_ranges": resolved,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        })
    except Exception as exc:  # pragma: no cover
        payload.update({
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            },
            "completed_at": datetime.now(timezone.utc).isoformat(),
        })

    write_json_atomic(output_path, payload)
    print(f"{payload['status'].upper()} {document['file']} -> {output_path}")


if __name__ == "__main__":
    main()
