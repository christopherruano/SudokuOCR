"""Run one TOC-manifest task for one PDF.

Designed for Slurm array jobs. Each task writes one atomic JSON result.
"""

from __future__ import annotations

import argparse
import socket
import traceback
from datetime import datetime, timezone
from pathlib import Path

from build_toc_manifest import process_pdf
from toc_job_utils import (
    is_valid_json_output,
    load_pdf_filelist,
    stable_doc_id,
    write_json_atomic,
)


def resolve_pdf(args) -> Path:
    if args.pdf:
        return Path(args.pdf).resolve()
    filelist = load_pdf_filelist(Path(args.filelist))
    if args.index < 0 or args.index >= len(filelist):
        raise IndexError(f"Index {args.index} out of range for {len(filelist)} PDFs")
    return filelist[args.index]


def build_output_path(args, pdf_path: Path) -> Path:
    if args.output:
        return Path(args.output)
    if not args.output_dir:
        raise ValueError("Provide either --output or --output-dir")
    out_dir = Path(args.output_dir)
    if args.index is not None and args.index >= 0:
        return out_dir / f"{args.index:06d}_{stable_doc_id(pdf_path)}.json"
    return out_dir / f"{stable_doc_id(pdf_path)}.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run TOC-manifest extraction for one PDF")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pdf", help="Single PDF path")
    group.add_argument("--filelist", help="Newline-delimited PDF filelist")
    parser.add_argument("--index", type=int, default=-1, help="0-based line index in --filelist")
    parser.add_argument("--output", help="Single output JSON path")
    parser.add_argument("--output-dir", help="Directory for per-task JSON outputs")
    parser.add_argument("--front-pages", type=int, default=80)
    parser.add_argument("--front-content-pages", type=int, default=25)
    parser.add_argument("--ocr-front", choices=["auto", "always", "never"], default="auto")
    parser.add_argument("--ocr-cache-dir")
    parser.add_argument("--adjudications-dir")
    parser.add_argument("--no-mine-body", action="store_true")
    parser.add_argument("--body-max-pages", type=int, default=0)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    pdf_path = resolve_pdf(args)
    output_path = build_output_path(args, pdf_path)

    if args.skip_existing and is_valid_json_output(output_path):
        print(f"SKIP {pdf_path} -> {output_path}")
        return

    adjudications = {}
    if args.adjudications_dir:
        from build_toc_manifest import load_adjudications
        adjudications = load_adjudications(Path(args.adjudications_dir))

    started = datetime.now(timezone.utc).isoformat()
    payload = {
        "status": "error",
        "pdf": str(pdf_path),
        "output": str(output_path),
        "task": {
            "filelist": str(args.filelist) if args.filelist else None,
            "index": None if args.index < 0 else args.index,
            "host": socket.gethostname(),
            "started_at": started,
        },
    }

    try:
        document = process_pdf(
            pdf_path=pdf_path,
            max_front_pdf_pages=args.front_pages,
            target_front_content_pages=args.front_content_pages,
            ocr_mode=args.ocr_front,
            ocr_cache_dir=Path(args.ocr_cache_dir) if args.ocr_cache_dir else None,
            mine_body=not args.no_mine_body,
            body_max_pages=args.body_max_pages,
            adjudications=adjudications,
        )
        payload.update({
            "status": "ok",
            "document": document,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        })
    except Exception as exc:  # pragma: no cover - defensive runtime path
        payload.update({
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            },
            "completed_at": datetime.now(timezone.utc).isoformat(),
        })

    write_json_atomic(output_path, payload)
    print(f"{payload['status'].upper()} {pdf_path} -> {output_path}")


if __name__ == "__main__":
    main()
