"""Build an explicit image manifest from digitized table directories.

This lets the OCR stage target only selected TOC-derived table types instead of
recursively scanning every PNG under digitized_tables.
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, Optional, Set


DEFAULT_STRICT_TOPICS = [
    "age_sex_population",
    "age_sex_civil_condition",
    "age_sex_marital_status",
]

POSITIVE_DIR_PATTERNS = [
    re.compile(r"\btable[-_ .]*(?:vii|vii\.|vil|vi1|v1i|7)\b", re.I),
    re.compile(r"\bimperial[-_ .]*table[-_ .]*(?:vii|vil|7)\b", re.I),
    re.compile(r"\bage\b", re.I),
    re.compile(r"age[-_ .]*sex", re.I),
    re.compile(r"age[-_ .]*sex[-_ .]*and[-_ .]*civil[-_ .]*condition", re.I),
    re.compile(r"age[-_ .]*sex[-_ .]*and[-_ .]*marital", re.I),
    re.compile(r"age[_-]?sex", re.I),
]

NEGATIVE_DIR_PATTERNS = [
    re.compile(r"\btable[-_ .]*(?:iv|vi|ix|x|xi|xii)\b", re.I),
    re.compile(r"\beducation(?:al)?\b", re.I),
    re.compile(r"birth[-_ .]*place", re.I),
    re.compile(r"\binfirmit", re.I),
    re.compile(r"\boccupation", re.I),
    re.compile(r"\blanguage\b", re.I),
    re.compile(r"\breligion\b", re.I),
    re.compile(r"\bcaste\b", re.I),
]


def load_digitized_table(table_dir):
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


def strict_age_dir_name_matches(name: str) -> bool:
    text = str(name or "")
    text = re.sub(r"[_./]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    positive = any(pattern.search(text) for pattern in POSITIVE_DIR_PATTERNS)
    negative = any(pattern.search(text) for pattern in NEGATIVE_DIR_PATTERNS)
    return positive and not negative


def rendered_image_rows(table_dir, digitized):
    if digitized:
        for rendered in digitized.get("rendered_pages") or []:
            image_path = rendered.get("image_path")
            if image_path:
                yield {
                    "image_path": image_path,
                    "table_dir": str(table_dir),
                    "digitized_table": digitized,
                }
        return

    for image_path in sorted(table_dir.glob("*.png")):
        yield {
            "image_path": str(image_path),
            "table_dir": str(table_dir),
            "digitized_table": None,
        }


def iter_selected_images(root, include_topics=None, include_relevance=None, strict_age_only=False):
    for table_dir in sorted(root.iterdir()):
        if not table_dir.is_dir():
            continue
        digitized = load_digitized_table(table_dir)
        queue_row = (digitized or {}).get("queue_row") or {}
        topic = queue_row.get("canonical_topic") or "unknown"
        relevance = queue_row.get("relevance_tier") or "unknown"
        dir_matches = strict_age_dir_name_matches(table_dir.name)
        if strict_age_only and not (topic in DEFAULT_STRICT_TOPICS or dir_matches):
            continue
        if include_topics and topic not in include_topics:
            if not (strict_age_only and dir_matches):
                continue
        if include_relevance and relevance not in include_relevance:
            if not (strict_age_only and dir_matches and relevance == "unknown"):
                continue
        for rendered in rendered_image_rows(table_dir, digitized):
            yield {
                "image_path": rendered["image_path"],
                "table_dir": str(table_dir),
                "canonical_topic": topic,
                "relevance_tier": relevance,
                "title": queue_row.get("display_identifier") or queue_row.get("title"),
                "year": queue_row.get("year"),
                "province": queue_row.get("province"),
                "dir_name_match": dir_matches,
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

    rows = list(iter_selected_images(
        Path(args.root).expanduser(),
        include_topics,
        include_relevance,
        strict_age_only=args.strict_age_only,
    ))

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
