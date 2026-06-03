#!/usr/bin/env python3
"""Build compact table-by-table packets for LLM plausibility review.

The packets are intentionally offline artifacts. They do not call any API and
they do not duplicate the full facts CSV. Each JSONL row contains enough context
for a model or human reviewer to perform a table-level sense check, while
preserving the deterministic audit as the primary validator.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


ROOT = Path("export_final/final_validated")
TABLES = ROOT / "data" / "tables_all_validated.csv"
FACTS = ROOT / "data" / "facts_all_validated.csv"
REGISTRY = ROOT / "review" / "final_table_registry.csv"
OUT = ROOT / "llm_plausibility_review"

SAMPLE_FIELDS = [
    "section",
    "row_label_raw",
    "row_label_std",
    "row_dimension",
    "row_group_value",
    "column_group_raw",
    "column_group_std",
    "column_dimension",
    "column_group_value",
    "sub_column_raw",
    "sub_column_std",
    "sex",
    "civil_condition",
    "religion",
    "resolved_age_label_std",
    "resolved_age_lower",
    "resolved_age_upper",
    "measure_unit",
    "value",
]


def set_csv_limits() -> None:
    csv.field_size_limit(min(sys.maxsize, 2**31 - 1))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def by_id(rows: Iterable[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {row.get("table_id", ""): row for row in rows if row.get("table_id")}


def to_float(value: object) -> float | None:
    text = str(value or "").strip().replace(",", "")
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def norm(value: object) -> str:
    return str(value or "").strip().lower()


def truthy(value: object) -> bool:
    return norm(value) in {"1", "true", "yes", "y"}


def table_format(row: dict[str, str]) -> str:
    return row.get("format_class") or row.get("measure_category") or row.get("table_family") or "unknown"


def format_review_guidance(table: dict[str, str]) -> dict[str, object]:
    fmt = table.get("format_class", "")
    measure = table.get("measure_category", "")
    denom = table.get("denominator_hint", "")
    title = norm(table.get("title"))
    guidance: dict[str, object] = {
        "format_class": fmt,
        "measure_category": measure,
        "denominator_hint": denom,
        "is_additive_count_table": measure == "counts" and fmt in {
            "additive_age_sex_counts",
            "age_civil_condition_counts",
            "age_by_religion_counts",
            "geography_or_district_counts",
            "education_literacy_table",
            "occupation_caste_tribe_race_table",
        },
        "is_distribution_table": fmt in {"distribution_of_1000", "distribution_of_10000"} or measure == "distribution_of_1000",
        "is_rate_or_ratio_table": fmt in {"rate_or_ratio_table", "mortality_rates"} or measure in {"per_1000_rates", "other_rates"},
        "is_mortality_or_vital_table": any(token in title for token in ["mortality", "death", "deaths", "birth", "births"]),
    }
    if fmt == "distribution_of_1000" or measure == "distribution_of_1000":
        guidance["expected_scale"] = "cells are distribution/per-mille quantities on a 0-1000 scale unless the title says otherwise"
        guidance["primary_checks"] = [
            "bounded distribution values should be between 0 and 1000; 10000-like values are likely scale, decimal, or alignment errors",
            "relevant subcategories may sum to about 1000 within each printed base group",
            "do not expect raw population totals",
        ]
    elif fmt == "distribution_of_10000":
        guidance["expected_scale"] = "cells are distribution quantities on a 0-10000 scale"
        guidance["primary_checks"] = [
            "bounded distribution values should be between 0 and 10000; larger values are likely raw-count leakage or alignment errors",
            "relevant subcategories may sum to about 10000 within each printed base group",
            "do not expect raw population totals",
        ]
    elif fmt == "rate_or_ratio_table":
        guidance["expected_scale"] = "cells are rates, ratios, percentages, sex ratios, or other indices; denominators may vary by title"
        guidance["primary_checks"] = [
            "read the title/headers to identify the denominator before judging magnitude",
            "decimal points are part of rate/ratio/percentage data; dropped decimals or integer-rounded rates should be reviewed",
            "female-per-1000-male and percentage tables are not additive count tables",
            "do not require row/column sums unless the printed table explicitly defines a distribution base",
        ]
    elif fmt == "mortality_rates":
        guidance["expected_scale"] = "cells are mortality/birth/death rates or ratios, not necessarily death counts"
        guidance["primary_checks"] = [
            "confirm whether the source says per mille, per 1000, percentage, ratio, or raw deaths",
            "preserve decimal points in mortality/birth/death rates and flag probable dropped decimals",
            "zero-heavy rows may be valid for small causes/areas but suspicious for general death-rate tables",
            "do not enforce Male+Female=All unless the table gives comparable count cells",
        ]
    elif measure == "counts":
        guidance["expected_scale"] = "cells are raw counts unless the title/headers say otherwise"
        guidance["primary_checks"] = [
            "counts should be non-negative",
            "where explicitly printed, additive totals should match subgroups",
            "large values can be plausible for province-level population counts",
        ]
    else:
        guidance["expected_scale"] = "mixed or descriptive non-count table; infer constraints from title and headers"
        guidance["primary_checks"] = [
            "do not force additive count constraints",
            "identify the printed denominator/base first",
            "use image comparison when the format is ambiguous",
        ]
    return guidance


def sample_row(row: dict[str, str]) -> dict[str, str]:
    return {field: row.get(field, "") for field in SAMPLE_FIELDS if row.get(field, "")}


def is_child_age(row: dict[str, str]) -> bool:
    lower = to_float(row.get("resolved_age_lower"))
    upper = to_float(row.get("resolved_age_upper"))
    label = norm(row.get("resolved_age_label_std") or row.get("row_label_std") or row.get("row_label_raw"))
    if "total" in label or "all age" in label:
        return False
    if lower is not None and lower <= 10:
        return True
    if upper is not None and upper <= 10:
        return True
    return any(token in label for token in ["0-1", "0-5", "1-5", "5-10", "under 5", "under five"])


class TableSummary:
    def __init__(self, max_samples: int) -> None:
        self.fact_rows = 0
        self.nonzero_rows = 0
        self.numeric_rows = 0
        self.zero_rows = 0
        self.min_value: float | None = None
        self.max_value: float | None = None
        self.value_sum = 0.0
        self.sections: Counter[str] = Counter()
        self.row_labels: Counter[str] = Counter()
        self.column_label_counts: Counter[str] = Counter()
        self.sex_values: Counter[str] = Counter()
        self.civil_values: Counter[str] = Counter()
        self.religion_values: Counter[str] = Counter()
        self.age_labels: Counter[str] = Counter()
        self.measure_units: Counter[str] = Counter()
        self.samples: list[dict[str, str]] = []
        self.suspicious_examples: dict[str, dict[str, str]] = {}
        self.heuristic_flags: Counter[str] = Counter()
        self.max_samples = max_samples

    def remember_flag(self, flag: str, row: dict[str, str]) -> None:
        self.heuristic_flags[flag] += 1
        self.suspicious_examples.setdefault(flag, sample_row(row))

    def add(self, row: dict[str, str], table_meta: dict[str, str]) -> None:
        self.fact_rows += 1
        value = to_float(row.get("value"))
        if value is None:
            self.remember_flag("nonnumeric_value", row)
        else:
            self.numeric_rows += 1
            self.value_sum += value
            self.min_value = value if self.min_value is None else min(self.min_value, value)
            self.max_value = value if self.max_value is None else max(self.max_value, value)
            if value == 0:
                self.zero_rows += 1
            else:
                self.nonzero_rows += 1
            if value < 0:
                self.remember_flag("negative_value", row)

            fmt = table_format(table_meta)
            measure = norm(table_meta.get("measure_category"))
            if fmt == "distribution_of_1000" or measure == "distribution_of_1000":
                if value < 0 or value > 1000:
                    self.remember_flag("distribution_1000_out_of_range", row)
            if measure in {"per_1000_rates", "other_rates"} and value > 10000:
                self.remember_flag("rate_value_extremely_large", row)
            if "mortality" in norm(table_meta.get("title")) and value == 0:
                self.remember_flag("mortality_zero_present", row)
            if norm(row.get("civil_condition")) in {"married", "widowed"} and is_child_age(row) and value >= 100:
                self.remember_flag("large_child_married_or_widowed_value", row)

        self.sections[row.get("section", "")] += 1
        self.row_labels[row.get("row_label_std") or row.get("row_label_raw", "")] += 1
        self.remember_column_label(row)
        self.sex_values[row.get("sex", "")] += 1
        self.civil_values[row.get("civil_condition", "")] += 1
        self.religion_values[row.get("religion", "")] += 1
        self.age_labels[row.get("resolved_age_label_std", "")] += 1
        self.measure_units[row.get("measure_unit", "")] += 1

        if len(self.samples) < self.max_samples:
            self.samples.append(sample_row(row))
        elif value is not None and value != 0 and len([s for s in self.samples if s.get("value") not in {"", "0", "0.0"}]) < self.max_samples // 2:
            self.samples.append(sample_row(row))

    def remember_column_label(self, row: dict[str, str]) -> None:
        parts = [
            row.get("column_group_std") or row.get("column_group_raw", ""),
            row.get("sub_column_std") or row.get("sub_column_raw", ""),
        ]
        label = " | ".join(part for part in parts if part).strip()
        self.column_label_counts[label] += 1

    def top(self, counter: Counter[str], limit: int = 20) -> list[str]:
        return [value for value, _ in counter.most_common(limit) if value]

    def as_dict(self) -> dict[str, object]:
        mean = self.value_sum / self.numeric_rows if self.numeric_rows else None
        return {
            "fact_rows": self.fact_rows,
            "numeric_rows": self.numeric_rows,
            "nonzero_rows": self.nonzero_rows,
            "zero_rows": self.zero_rows,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "mean_value": round(mean, 6) if mean is not None and math.isfinite(mean) else None,
            "top_sections": self.top(self.sections),
            "top_row_labels": self.top(self.row_labels),
            "top_column_labels": self.top(self.column_label_counts),
            "sex_values": self.top(self.sex_values),
            "civil_condition_values": self.top(self.civil_values),
            "religion_values": self.top(self.religion_values),
            "age_labels": self.top(self.age_labels),
            "measure_units": self.top(self.measure_units),
            "heuristic_flags": dict(sorted(self.heuristic_flags.items())),
            "suspicious_examples": self.suspicious_examples,
            "sample_facts": self.samples,
        }


def recommended_checks(table: dict[str, str]) -> list[str]:
    checks = [
        "Does the extracted table family match the title and visible source context?",
        "Do row labels and column labels preserve the printed table axes?",
        "Are zeros plausible structural zeros rather than OCR/dropout artifacts?",
        "Are very large values plausible for the province/year/unit?",
    ]
    family = table.get("table_family", "")
    measure = table.get("measure_category", "")
    fmt = table.get("format_class", "")
    title = norm(table.get("title"))
    if family in {"age_sex_population", "age_distribution"}:
        checks.append("Do age bins progress sensibly and avoid impossible age placements?")
    if "civil_condition" in family or "marital" in family:
        checks.append("Are married/widowed counts for young child age bins absent or very small unless the source explicitly shows them?")
    if measure == "counts":
        checks.append("If the printed table explicitly has comparable Male, Female, and Persons/All count cells, do those totals match within expected OCR tolerance?")
    if measure in {"distribution_of_1000", "per_1000_rates"} or fmt == "distribution_of_1000":
        checks.append("Are values interpreted as per-1,000/distribution units rather than raw population counts?")
    if fmt == "distribution_of_10000":
        checks.append("Are values interpreted as distribution-of-10,000 units rather than raw population counts?")
    if fmt == "rate_or_ratio_table":
        checks.append("Does the title define a rate/ratio/percentage/index base, and are additive count constraints avoided?")
    if "mortality" in title or "death" in title or "birth" in title:
        checks.append("For mortality/birth-rate tables, determine whether the values are counts, rates, per-mille, ratios, percentages, or descriptive index values before pooling.")
    if family == "age_by_religion":
        checks.append("Do religion columns remain distinct from race/caste/occupation subcolumns?")
    return checks


def priority_score(table: dict[str, str], registry: dict[str, str], summary: TableSummary) -> tuple[int, list[str]]:
    reasons: list[str] = []
    score = 0
    title = norm(table.get("title"))
    measure = table.get("measure_category", "")
    if summary.heuristic_flags:
        score += 100
        reasons.append("heuristic_flags_present")
    if registry.get("mortality_scale_status") == "scale_review_needed":
        score += 90
        reasons.append("mortality_scale_review_needed")
    if "mortality" in title or "death" in title or "birth" in title:
        score += 50
        reasons.append("mortality_or_vital_rates")
    if measure in {"per_1000_rates", "other_rates", "distribution_of_1000"}:
        score += 30
        reasons.append("rate_or_distribution_unit")
    if int(table.get("failed_checks") or 0) != 0:
        score += 100
        reasons.append("failed_checks_nonzero")
    if int(table.get("facts_rows") or 0) > 1000:
        score += 10
        reasons.append("large_table")
    return score, reasons


def write_prompt(path: Path) -> None:
    prompt = """# LLM Plausibility Review Prompt

You are reviewing OCR-extracted historical census tables. Treat deterministic
constraint checks as primary evidence, and use your judgment only to flag
semantic plausibility problems that arithmetic checks can miss.

For each JSONL packet:

1. Decide one of: `pass`, `needs_image_compare`, `needs_manual_review`, `reject_or_rerun`.
2. Give short reasons grounded in table metadata, sample facts, and the source image path.
3. Do not invent corrections. If exact numbers need confirmation, request image comparison.
4. Be stricter for mortality/rate/distribution tables because units can be easy to misread.
5. Preserve the original table's unit. Do not convert per-1,000, rates, or percentages into counts.

Core plausibility rules:

- Counts cannot be negative.
- Bounded distribution-of-1,000 values should be between 0 and 1,000; 10,000-like values are likely scale, decimal, or alignment errors unless the source clearly says the cell is a ratio/rate that can exceed 1,000.
- Bounded distribution-of-10,000 values should be between 0 and 10,000.
- Per-mille/rate tables should be interpreted as rates, not population counts.
- Percentages, rates, mortality rates, averages, and ratios may have decimal places; decimal points are part of the data and should not be rounded away.
- The normalized sex field is opportunistic metadata extracted from labels like Male, Female, Persons, Total, or All. Do not assume every table should have Male/Female/All categories.
- Only apply Male + Female = Persons/All checks when the printed title, row labels, column labels, or section labels explicitly define a sex axis with comparable count cells.
- Do not flag a table merely because sex_values are blank, only All, or not relevant to the printed table.
- Very young age bins should not have large married/widowed counts unless visibly supported by the source.
- All-zero mortality/rate rows are suspicious if the title implies measured deaths or rates, but they can be valid if the source has structural zeros.
- If a table format is unusual, infer the appropriate constraints from the title, row labels, column labels, and units before judging.

Return compact JSON for each packet:

```json
{
  "table_id": "...",
  "decision": "pass|needs_image_compare|needs_manual_review|reject_or_rerun",
  "confidence": "high|medium|low",
  "flags": ["..."],
  "reason": "...",
  "suggested_next_action": "..."
}
```
"""
    path.write_text(prompt, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tables-csv", type=Path, default=TABLES)
    parser.add_argument("--facts-csv", type=Path, default=FACTS)
    parser.add_argument("--registry-csv", type=Path, default=REGISTRY)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--max-sample-facts", type=int, default=60)
    parser.add_argument("--priority-threshold", type=int, default=30)
    args = parser.parse_args()

    set_csv_limits()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tables = by_id(read_csv(args.tables_csv))
    registry = by_id(read_csv(args.registry_csv))
    summaries = {table_id: TableSummary(args.max_sample_facts) for table_id in tables}

    with args.facts_csv.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for i, row in enumerate(reader, start=1):
            table_id = row.get("table_id", "")
            if table_id in summaries:
                summaries[table_id].add(row, tables[table_id])
            if i % 200000 == 0:
                print(f"scanned {i:,} fact rows")

    all_path = args.output_dir / "validated_table_packets.jsonl"
    priority_path = args.output_dir / "priority_review_packets.jsonl"
    queue_path = args.output_dir / "llm_review_queue.csv"
    results_path = args.output_dir / "llm_review_results_template.csv"

    queue_fields = [
        "table_id",
        "priority_score",
        "priority_reasons",
        "province",
        "year",
        "table_family",
        "measure_category",
        "format_class",
        "facts_rows",
        "heuristic_flags",
        "source_image_local",
        "title",
    ]
    result_fields = [
        "table_id",
        "decision",
        "confidence",
        "flags",
        "reason",
        "suggested_next_action",
        "reviewer",
        "reviewed_at",
    ]

    queue_rows: list[dict[str, str]] = []
    packet_count = 0
    priority_count = 0
    with all_path.open("w", encoding="utf-8") as all_fh, priority_path.open("w", encoding="utf-8") as priority_fh:
        for table_id, table in sorted(tables.items(), key=lambda item: (item[1].get("province", ""), item[1].get("year", ""), item[0])):
            summary = summaries[table_id]
            reg = registry.get(table_id, {})
            score, reasons = priority_score(table, reg, summary)
            packet = {
                "table_id": table_id,
                "metadata": {
                    "province": table.get("province", ""),
                    "year": table.get("year", ""),
                    "title": table.get("title", ""),
                    "table_family": table.get("table_family", ""),
                    "format_class": table.get("format_class", ""),
                    "measure_category": table.get("measure_category", ""),
                    "constraint_regime": table.get("constraint_regime", ""),
                    "denominator_hint": table.get("denominator_hint", ""),
                    "quality_tier": table.get("quality_tier", ""),
                    "effective_quality_status": table.get("effective_quality_status", ""),
                    "n_checks": table.get("n_checks", ""),
                    "failed_checks": table.get("failed_checks", ""),
                    "source_image_local": table.get("source_image_local", ""),
                    "source_json_local": table.get("source_json_local", ""),
                    "excel_path": table.get("excel_path", ""),
                },
                "deterministic_audit": {
                    "plausibility_status": reg.get("plausibility_status", ""),
                    "issue_flags": reg.get("issue_flags", ""),
                    "blocking_reasons": reg.get("blocking_reasons", ""),
                    "mortality_scale_status": reg.get("mortality_scale_status", ""),
                    "mortality_scale_flags": reg.get("mortality_scale_flags", ""),
                },
                "format_review_guidance": format_review_guidance(table),
                "review_priority": {
                    "score": score,
                    "reasons": reasons,
                },
                "recommended_checks": recommended_checks(table),
                "summary": summary.as_dict(),
            }
            all_fh.write(json.dumps(packet, ensure_ascii=False) + "\n")
            packet_count += 1
            if score >= args.priority_threshold:
                priority_fh.write(json.dumps(packet, ensure_ascii=False) + "\n")
                priority_count += 1
            queue_rows.append(
                {
                    "table_id": table_id,
                    "priority_score": str(score),
                    "priority_reasons": ";".join(reasons),
                    "province": table.get("province", ""),
                    "year": table.get("year", ""),
                    "table_family": table.get("table_family", ""),
                    "measure_category": table.get("measure_category", ""),
                    "format_class": table.get("format_class", ""),
                    "facts_rows": table.get("facts_rows", ""),
                    "heuristic_flags": ";".join(summary.heuristic_flags),
                    "source_image_local": table.get("source_image_local", ""),
                    "title": table.get("title", ""),
                }
            )

    queue_rows.sort(key=lambda r: (-int(r["priority_score"]), r["province"], r["year"], r["table_id"]))
    with queue_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=queue_fields)
        writer.writeheader()
        writer.writerows(queue_rows)
    with results_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=result_fields)
        writer.writeheader()
    write_prompt(args.output_dir / "llm_review_prompt.md")
    (args.output_dir / "README.md").write_text(
        "\n".join(
            [
                "# LLM Plausibility Review",
                "",
                "Offline packet export for table-by-table LLM sense checks.",
                "",
                "Files:",
                "",
                "- `validated_table_packets.jsonl`: one compact packet for every currently validated table.",
                "- `priority_review_packets.jsonl`: packets with rate/mortality/unit/heuristic priority signals.",
                "- `llm_review_queue.csv`: sortable table-level queue.",
                "- `llm_review_results_template.csv`: place to store model or human decisions.",
                "- `llm_review_prompt.md`: prompt/schema for review.",
                "",
                "Use this as a triage layer. A `pass` from an LLM is not a replacement for the deterministic audit or visual image comparison.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    report = {
        "validated_table_packets": packet_count,
        "priority_review_packets": priority_count,
        "priority_threshold": args.priority_threshold,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
