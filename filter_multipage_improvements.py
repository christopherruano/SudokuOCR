"""Filter assembled multi-page groups down to the ones that genuinely improve.

Default policy:
  - keep groups with 2+ pages
  - keep only groups where assembled constraint failures < input page failures
  - exclude birth-place themed groups by title

Outputs:
  - improved_groups.csv
  - improved_groups_strong.csv
  - improved_assembled_tables.jsonl
  - improved_member_source_jsons.txt
  - filter_summary.json
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


DEFAULT_EXCLUDE_PATTERNS = [
    r"\bbirth[-\s]*place\b",
    r"\blanguage\b",
]


def normalize_text(value: object) -> str:
    return " ".join(str(value or "").strip().split())


def text_matches(text: str, patterns: list[re.Pattern]) -> bool:
    return any(pattern.search(text or "") for pattern in patterns)


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter assembled multi-page groups to the improved subset.")
    parser.add_argument("--groups-csv", required=True)
    parser.add_argument("--assembled-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--exclude-pattern", action="append", default=[], help="Regex pattern to exclude by title")
    parser.add_argument("--require-all-passed", action="store_true", help="Keep only groups whose assembled constraints fully pass")
    args = parser.parse_args()

    groups_csv = Path(args.groups_csv).expanduser()
    assembled_jsonl = Path(args.assembled_jsonl).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    exclude_patterns = [re.compile(pat, re.I) for pat in (args.exclude_pattern or DEFAULT_EXCLUDE_PATTERNS)]

    group_rows = list(csv.DictReader(groups_csv.open(encoding="utf-8", newline="")))
    assembled_by_id = {}
    with assembled_jsonl.open(encoding="utf-8") as fh:
        for line in fh:
            obj = json.loads(line)
            assembled_by_id[obj["group_id"]] = obj

    improved = []
    strong = []
    excluded = []

    for row in group_rows:
        title = normalize_text(row.get("title"))
        n_pages = int(row.get("n_pages") or 0)
        input_fail = int(row.get("input_constraint_failures") or 0)
        assembled_fail = int(row.get("assembled_constraint_failures") or 0)
        all_passed = str(row.get("assembled_all_passed") or "").strip().lower() in {"1", "true", "yes"}
        delta = input_fail - assembled_fail

        enriched = dict(row)
        enriched["failure_improvement"] = str(delta)
        enriched["is_improved"] = "1" if (n_pages > 1 and delta > 0) else "0"
        enriched["is_strong"] = "1" if (n_pages > 1 and delta > 0 and all_passed) else "0"

        if text_matches(title, exclude_patterns):
            excluded.append(enriched)
            continue
        if n_pages <= 1:
            continue
        if delta <= 0:
            continue
        if args.require_all_passed and not all_passed:
            continue

        improved.append(enriched)
        if all_passed:
            strong.append(enriched)

    fieldnames = list(improved[0].keys()) if improved else list(group_rows[0].keys()) + ["failure_improvement", "is_improved", "is_strong"]
    write_csv(output_dir / "improved_groups.csv", improved, fieldnames)
    write_csv(output_dir / "improved_groups_strong.csv", strong, fieldnames)
    write_csv(output_dir / "excluded_groups.csv", excluded, fieldnames)

    improved_ids = {row["group_id"] for row in improved}
    strong_ids = {row["group_id"] for row in strong}

    improved_members = []
    strong_members = []
    with (output_dir / "improved_assembled_tables.jsonl").open("w", encoding="utf-8") as outf, \
            (output_dir / "strong_assembled_tables.jsonl").open("w", encoding="utf-8") as strongf:
        for group_id, obj in assembled_by_id.items():
            if group_id in improved_ids:
                outf.write(json.dumps(obj, ensure_ascii=False) + "\n")
                improved_members.extend(obj.get("page_source_jsons") or [])
            if group_id in strong_ids:
                strongf.write(json.dumps(obj, ensure_ascii=False) + "\n")
                strong_members.extend(obj.get("page_source_jsons") or [])

    with (output_dir / "improved_member_source_jsons.txt").open("w", encoding="utf-8") as fh:
        for path in sorted(dict.fromkeys(improved_members)):
            fh.write(f"{path}\n")
    with (output_dir / "strong_member_source_jsons.txt").open("w", encoding="utf-8") as fh:
        for path in sorted(dict.fromkeys(strong_members)):
            fh.write(f"{path}\n")

    summary = {
        "input_groups": len(group_rows),
        "improved_groups": len(improved),
        "strong_groups": len(strong),
        "excluded_groups": len(excluded),
        "exclude_patterns": [pat.pattern for pat in exclude_patterns],
        "require_all_passed": args.require_all_passed,
    }
    (output_dir / "filter_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Improved multipage filter")
    print("-------------------------")
    print(f"Input groups:      {len(group_rows)}")
    print(f"Improved groups:   {len(improved)}")
    print(f"Strong groups:     {len(strong)}")
    print(f"Excluded groups:   {len(excluded)}")
    print(f"\nOutputs written to: {output_dir}")


if __name__ == "__main__":
    main()
