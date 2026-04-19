"""Render page windows for one resolved table queue item.

This is the first concrete digitization stage after the TOC/range pipeline:
it converts a resolved table record into page images and sidecar text.
"""

from __future__ import annotations

import argparse
import json
import socket
import traceback
from datetime import datetime, timezone
from pathlib import Path

from build_toc_manifest import ocr_rendered_page, page_text_kind, pdftotext_pages, render_pdf_page
from toc_job_utils import is_valid_json_output, jsonl_iter, safe_slug, stable_doc_id, write_json_atomic


def load_queue_item(queue_path: Path, index: int) -> dict:
    for idx, row in enumerate(jsonl_iter(queue_path)):
        if idx == index:
            return row
    raise IndexError(f"Index {index} out of range for {queue_path}")


def render_task(queue_row: dict, output_root: Path) -> dict:
    file_path = Path(queue_row["file"]).resolve()
    start = int(queue_row["resolved_start_page"])
    end = int(queue_row["resolved_end_page"])
    doc_id = stable_doc_id(file_path)
    label = queue_row.get("display_identifier") or queue_row.get("title") or f"table_{queue_row.get('table_index')}"
    out_dir = output_root / f"{doc_id}_t{int(queue_row['table_index']):04d}_{safe_slug(label)}"
    out_dir.mkdir(parents=True, exist_ok=True)

    page_texts = pdftotext_pages(file_path, start, end, timeout=max(300, (end - start + 1) * 30))
    rendered_pages = []
    for payload in page_texts:
        page_num = payload["pdf_page"]
        png = render_pdf_page(file_path, page_num, out_dir)
        text = payload.get("text") or ""
        source = "pdftotext"
        if page_text_kind(text) != "contentful":
            ocr_text = ocr_rendered_page(png)
            if page_text_kind(ocr_text) == "contentful":
                text = ocr_text
                source = "ocr"
        text_path = out_dir / f"page_{page_num:04d}.txt"
        text_path.write_text(text or "", encoding="utf-8")
        rendered_pages.append({
            "pdf_page": page_num,
            "image_path": str(png),
            "text_path": str(text_path),
            "text_source": source,
            "text_excerpt": (text or "")[:2000],
        })

    return {
        "source_file": str(file_path),
        "output_dir": str(out_dir),
        "queue_row": queue_row,
        "rendered_pages": rendered_pages,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Render one resolved table queue item")
    parser.add_argument("--queue", required=True, help="Digitization queue JSONL")
    parser.add_argument("--index", type=int, required=True, help="0-based line index in queue")
    parser.add_argument("--output-dir", required=True, help="Output root directory")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    queue_row = load_queue_item(Path(args.queue), args.index)
    out_dir = Path(args.output_dir)
    file_path = Path(queue_row["file"]).resolve()
    doc_id = stable_doc_id(file_path)
    label = queue_row.get("display_identifier") or queue_row.get("title") or f"table_{queue_row.get('table_index')}"
    task_dir = out_dir / f"{doc_id}_t{int(queue_row['table_index']):04d}_{safe_slug(label)}"
    metadata_path = task_dir / "metadata.json"

    if args.skip_existing and is_valid_json_output(metadata_path):
        print(f"SKIP {queue_row['file']} table {queue_row['table_index']} -> {metadata_path}")
        return

    payload = {
        "status": "error",
        "task": {
            "queue": str(Path(args.queue).resolve()),
            "index": args.index,
            "host": socket.gethostname(),
            "started_at": datetime.now(timezone.utc).isoformat(),
        },
    }
    try:
        result = render_task(queue_row, out_dir)
        payload.update({
            "status": "ok",
            "digitized_table": result,
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

    write_json_atomic(metadata_path, payload)
    print(f"{payload['status'].upper()} {queue_row['file']} table {queue_row['table_index']} -> {metadata_path}")


if __name__ == "__main__":
    main()
