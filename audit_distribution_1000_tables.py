#!/usr/bin/env python3
"""Audit proportional "distribution of 1,000" tables in final exports."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


PATTERNS = {
    "civil_condition_of_1000": re.compile(
        r"distribution by civil condition of 1,?000", re.IGNORECASE
    ),
    "age_distribution_of_1000": re.compile(
        r"age distribution of 1,?000", re.IGNORECASE
    ),
    "other_distribution_of_1000": re.compile(
        r"distribution of 1,?000", re.IGNORECASE
    ),
}


def classify_title(title: str) -> str:
    for label in ("civil_condition_of_1000", "age_distribution_of_1000"):
        if PATTERNS[label].search(title or ""):
            return label
    if PATTERNS["other_distribution_of_1000"].search(title or ""):
        return "other_distribution_of_1000"
    return ""


def as_int(value: str) -> int:
    try:
        return int(str(value or "").strip())
    except ValueError:
        return 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-catalog",
        action="append",
        required=True,
        help="CSV path in the form layer_name=path/to/source_catalog.csv",
    )
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    output_rows = []
    summary = {
        "total_rows": 0,
        "by_layer": {},
        "by_layer_and_category": {},
    }

    for spec in args.source_catalog:
        if "=" not in spec:
            raise SystemExit(f"Invalid --source-catalog value: {spec}")
        layer, path_str = spec.split("=", 1)
        path = Path(path_str)
        layer_counts = Counter()
        by_category = defaultdict(Counter)

        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row.get("constraint_regime") != "proportional_distribution":
                    continue
                if row.get("denominator_hint") != "1000":
                    continue

                category = classify_title(row.get("title", ""))
                if not category:
                    continue

                n_checks = as_int(row.get("n_checks", ""))
                failed_checks = as_int(row.get("failed_checks", ""))
                status = row.get("effective_quality_status") or row.get("quality_status") or ""

                audit_row = {
                    "layer": layer,
                    "category": category,
                    "province": row.get("province", ""),
                    "year": row.get("year", ""),
                    "table_family": row.get("table_family", ""),
                    "title": row.get("title", ""),
                    "effective_quality_status": status,
                    "quality_status": row.get("quality_status", ""),
                    "n_checks": str(n_checks),
                    "failed_checks": str(failed_checks),
                    "strict_zero_fail_checks": "1" if n_checks > 0 and failed_checks == 0 else "0",
                    "contextual_unvalidated": "1" if n_checks == 0 else "0",
                    "qc_flags": row.get("qc_flags", ""),
                    "table_id": row.get("table_id", ""),
                    "source_json": row.get("source_json", ""),
                }
                output_rows.append(audit_row)

                layer_counts["total"] += 1
                layer_counts[f"status::{status}"] += 1
                if n_checks > 0 and failed_checks == 0:
                    layer_counts["strict_zero_fail_checks"] += 1
                if n_checks == 0:
                    layer_counts["contextual_unvalidated"] += 1

                by_category[category]["total"] += 1
                by_category[category][f"status::{status}"] += 1
                if n_checks > 0 and failed_checks == 0:
                    by_category[category]["strict_zero_fail_checks"] += 1
                if n_checks == 0:
                    by_category[category]["contextual_unvalidated"] += 1

        summary["by_layer"][layer] = dict(layer_counts)
        summary["by_layer_and_category"][layer] = {
            category: dict(counts) for category, counts in by_category.items()
        }

    output_rows.sort(
        key=lambda row: (
            row["layer"],
            row["category"],
            row["province"],
            row["year"],
            row["title"],
        )
    )
    summary["total_rows"] = len(output_rows)

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0].keys()))
        writer.writeheader()
        writer.writerows(output_rows)

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote {len(output_rows)} rows to {output_csv}")
    print(f"Wrote summary to {output_json}")


if __name__ == "__main__":
    main()
