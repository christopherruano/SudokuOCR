"""Inspect rendered digitized table directories using TOC-derived metadata.

This explains what the OCR stage is actually being asked to process.
It reads each table directory's metadata.json, summarizes canonical topics,
relevance tiers, table titles, and optional OCR-result coverage.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def iter_table_dirs(root: Path):
    for path in sorted(root.iterdir()):
        if path.is_dir():
            yield path


def load_metadata(table_dir: Path) -> dict | None:
    metadata_path = table_dir / "metadata.json"
    if not metadata_path.exists():
        return None
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if payload.get("status") != "ok":
        return None
    return payload.get("digitized_table")


def count_pngs(table_dir: Path) -> int:
    return sum(1 for _ in table_dir.glob("*.png"))


def result_status_counts(results_root: Path | None):
    counts = Counter()
    by_dir = Counter()
    if not results_root or not results_root.exists():
        return counts, by_dir

    for path in results_root.rglob("*_oneshot.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            counts["corrupt"] += 1
            continue
        acceptance = payload.get("acceptance") or {}
        constraints = payload.get("constraints") or {}
        if constraints.get("failed", 0) > 0:
            status = "constraint_failures"
        else:
            status = acceptance.get("decision") or "unknown"
        counts[status] += 1
        try:
            table_dir = path.parent.name
            by_dir[table_dir] += 1
        except Exception:
            pass
    return counts, by_dir


def summarize(root: Path, results_root: Path | None = None):
    topic_counts = Counter()
    relevance_counts = Counter()
    title_counts = Counter()
    year_counts = Counter()
    province_counts = Counter()
    topic_pages = Counter()
    topic_dirs = Counter()
    missing_metadata = []
    rows = []

    result_counts, results_by_dir = result_status_counts(results_root)

    for table_dir in iter_table_dirs(root):
        metadata = load_metadata(table_dir)
        if not metadata:
            missing_metadata.append(str(table_dir))
            continue
        queue_row = metadata.get("queue_row") or {}
        topic = queue_row.get("canonical_topic") or "unknown"
        relevance = queue_row.get("relevance_tier") or "unknown"
        title = queue_row.get("display_identifier") or queue_row.get("title") or table_dir.name
        png_count = count_pngs(table_dir)

        topic_counts[topic] += 1
        relevance_counts[relevance] += 1
        title_counts[title] += 1
        year_counts[str(queue_row.get("year") or "unknown")] += 1
        province_counts[queue_row.get("province") or "unknown"] += 1
        topic_pages[topic] += png_count
        topic_dirs[topic] += 1

        rows.append({
            "table_dir": str(table_dir),
            "dir_name": table_dir.name,
            "topic": topic,
            "relevance": relevance,
            "title": title,
            "year": queue_row.get("year"),
            "province": queue_row.get("province"),
            "png_count": png_count,
            "ocr_results": results_by_dir.get(table_dir.name, 0),
        })

    rows.sort(key=lambda row: (row["topic"], row["year"] or 0, row["province"] or "", row["title"]))

    return {
        "table_dir_count": len(rows),
        "png_count": sum(row["png_count"] for row in rows),
        "missing_metadata_count": len(missing_metadata),
        "missing_metadata_samples": missing_metadata[:20],
        "topic_counts": topic_counts,
        "relevance_counts": relevance_counts,
        "title_counts": title_counts,
        "year_counts": year_counts,
        "province_counts": province_counts,
        "topic_pages": topic_pages,
        "ocr_result_status_counts": result_counts,
        "rows": rows,
    }


def print_counter(title: str, counter: Counter, limit: int = 20):
    print(f"\n{title}")
    print("-" * len(title))
    for key, value in counter.most_common(limit):
        print(f"{value:>6}  {key}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect digitized census table directories")
    parser.add_argument("root", help="digitized_tables root")
    parser.add_argument("--results-root", help="Optional OCR results root for coverage stats")
    parser.add_argument("--json-output", help="Write full JSON summary")
    parser.add_argument("--top", type=int, default=20, help="How many top titles/topics to show")
    args = parser.parse_args()

    summary = summarize(Path(args.root).expanduser(), Path(args.results_root).expanduser() if args.results_root else None)

    print(f"Table directories: {summary['table_dir_count']}")
    print(f"Rendered PNGs:     {summary['png_count']}")
    print(f"Missing metadata:  {summary['missing_metadata_count']}")

    print_counter("By Relevance", summary["relevance_counts"], limit=args.top)
    print_counter("By Canonical Topic", summary["topic_counts"], limit=args.top)
    print_counter("By Canonical Topic (pages)", summary["topic_pages"], limit=args.top)
    print_counter("Top Titles", summary["title_counts"], limit=args.top)
    print_counter("By Province", summary["province_counts"], limit=args.top)
    print_counter("By Year", summary["year_counts"], limit=args.top)

    if summary["ocr_result_status_counts"]:
        print_counter("OCR Result Statuses", summary["ocr_result_status_counts"], limit=args.top)

    if args.json_output:
        payload = {
            "table_dir_count": summary["table_dir_count"],
            "png_count": summary["png_count"],
            "missing_metadata_count": summary["missing_metadata_count"],
            "missing_metadata_samples": summary["missing_metadata_samples"],
            "topic_counts": dict(summary["topic_counts"]),
            "relevance_counts": dict(summary["relevance_counts"]),
            "title_counts": dict(summary["title_counts"]),
            "year_counts": dict(summary["year_counts"]),
            "province_counts": dict(summary["province_counts"]),
            "topic_pages": dict(summary["topic_pages"]),
            "ocr_result_status_counts": dict(summary["ocr_result_status_counts"]),
            "rows": summary["rows"],
        }
        Path(args.json_output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nWrote JSON summary: {args.json_output}")


if __name__ == "__main__":
    main()
