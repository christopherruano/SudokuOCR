"""Build an explicit image manifest from digitized table directories.

This lets the OCR stage target only selected TOC-derived table types instead of
recursively scanning every PNG under digitized_tables.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_STRICT_TOPICS = [
    "age_sex_population",
    "age_sex_civil_condition",
    "age_sex_marital_status",
]


def load_digitized_table(table_dir: Path) -> dict | None:
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


def iter_selected_images(root: Path, include_topics: set[str] | None, include_relevance: set[str] | None):
    for table_dir in sorted(root.iterdir()):
        if not table_dir.is_dir():
            continue
        digitized = load_digitized_table(table_dir)
        if not digitized:
            continue
        queue_row = digitized.get("queue_row") or {}
        topic = queue_row.get("canonical_topic") or "unknown"
        relevance = queue_row.get("relevance_tier") or "unknown"
        if include_topics and topic not in include_topics:
            continue
        if include_relevance and relevance not in include_relevance:
            continue
        for rendered in digitized.get("rendered_pages") or []:
            image_path = rendered.get("image_path")
            if image_path:
                yield {
                    "image_path": image_path,
                    "table_dir": str(table_dir),
                    "canonical_topic": topic,
                    "relevance_tier": relevance,
                    "title": queue_row.get("display_identifier") or queue_row.get("title"),
                    "year": queue_row.get("year"),
                    "province": queue_row.get("province"),
                }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a filtered OCR image manifest from digitized_tables")
    parser.add_argument("root", help="digitized_tables root")
    parser.add_argument("--output", required=True, help="Output text file with one image path per line")
    parser.add_argument("--json-output", help="Optional JSONL with per-image metadata")
    parser.add_argument("--topics", nargs="+", help="Canonical topics to include")
    parser.add_argument("--strict-age-only", action="store_true",
                        help="Shortcut for age_sex_population + age_sex_civil_condition + age_sex_marital_status")
    parser.add_argument("--relevance", nargs="+", default=["core"],
                        help="Relevance tiers to include (default: core)")
    args = parser.parse_args()

    include_topics = set(args.topics or [])
    if args.strict_age_only:
        include_topics = set(DEFAULT_STRICT_TOPICS)
    if not include_topics:
        include_topics = None
    include_relevance = set(args.relevance or [])
    if not include_relevance:
        include_relevance = None

    rows = list(iter_selected_images(Path(args.root).expanduser(), include_topics, include_relevance))

    out_path = Path(args.output).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(f"{row['image_path']}\n")

    print(f"Selected images: {len(rows)}")
    print(f"Output list: {out_path}")

    if args.json_output:
        json_path = Path(args.json_output).expanduser()
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")
        print(f"Output JSONL: {json_path}")


if __name__ == "__main__":
    main()
