from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from build_sendout_final import DATASETS, OUT, TEAM, infer_format, read_csv, to_float, truthy, write_csv, write_json


csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

AUDIT_FIELDS = [
    "table_id",
    "layer",
    "sendout_bucket",
    "province",
    "year",
    "title",
    "table_family",
    "format_class",
    "in_release_scope",
    "quality_tier",
    "failed_checks",
    "needs_manual_review",
    "facts_seen",
    "nonzero_values",
    "plausibility_status",
    "recommendation",
    "flags",
    "flag_count",
    "existing_qc_flags",
]

ISSUE_FIELDS = [
    "table_id",
    "layer",
    "sendout_bucket",
    "format_class",
    "severity",
    "flag",
    "detail",
    "example",
]

REVIEW_FIELDS = [
    "table_id",
    "layer",
    "province",
    "year",
    "title",
    "sendout_bucket",
    "format_class",
    "plausibility_status",
    "recommendation",
    "flags",
    "failed_checks",
    "quality_tier",
    "needs_manual_review",
]


def norm(value: object) -> str:
    return str(value or "").strip().lower()


def hash_key(parts: list[str]) -> str:
    return hashlib.blake2b("\x1f".join(parts).encode("utf-8"), digest_size=12).hexdigest()


def coord_hash(row: dict[str, str], drop_sex: bool = False) -> str:
    fields = [
        "section",
        "row_label_std",
        "row_dimension",
        "row_group_value",
        "column_group_std",
        "column_dimension",
        "column_group_value",
        "sub_column_std",
        "sex",
        "civil_condition",
        "religion",
        "resolved_age_axis",
        "resolved_age_label_std",
    ]
    parts = []
    for field in fields:
        if drop_sex and field == "sex":
            parts.append("")
        else:
            parts.append(norm(row.get(field)))
    return hash_key(parts)


def age_lower_upper(row: dict[str, str]) -> tuple[float | None, float | None]:
    return to_float(row.get("resolved_age_lower")), to_float(row.get("resolved_age_upper"))


def is_total_age(row: dict[str, str]) -> bool:
    label = norm(row.get("resolved_age_label_std"))
    return truthy(row.get("resolved_age_total")) or label in {"total", "all", "all ages"}


def is_child_age(row: dict[str, str]) -> bool:
    lower, upper = age_lower_upper(row)
    if lower is None and upper is None:
        return False
    if lower is not None and lower > 5:
        return False
    return upper is None or upper <= 10


def normalized_sex(value: str) -> str:
    value = norm(value)
    if value in {"persons", "person", "all", "total", "both sexes"}:
        return "persons"
    if value in {"male", "males"}:
        return "male"
    if value in {"female", "females"}:
        return "female"
    return ""


class TableState:
    __slots__ = (
        "facts_seen",
        "nonzero",
        "nonnumeric",
        "negative",
        "duplicate_conflicts",
        "distribution_out_of_range",
        "massive_child_married_widowed",
        "age_labels",
        "civil_repeat_values",
        "coord_values",
        "sex_groups",
        "mortality_small_values",
        "copied_row_values",
        "examples",
    )

    def __init__(self) -> None:
        self.facts_seen = 0
        self.nonzero = 0
        self.nonnumeric = 0
        self.negative = 0
        self.duplicate_conflicts = 0
        self.distribution_out_of_range = 0
        self.massive_child_married_widowed = 0
        self.age_labels: set[str] = set()
        self.civil_repeat_values: dict[str, set[str]] = defaultdict(set)
        self.coord_values: dict[str, str] = {}
        self.sex_groups: dict[str, dict[str, float]] = defaultdict(dict)
        self.mortality_small_values: Counter[float] = Counter()
        self.copied_row_values: dict[str, set[str]] = defaultdict(set)
        self.examples: dict[str, str] = {}

    def remember(self, flag: str, row: dict[str, str]) -> None:
        if flag not in self.examples:
            self.examples[flag] = f"{row.get('row_label_raw','')} | {row.get('column_group_raw','')} | {row.get('sub_column_raw','')} | {row.get('value','')}"


def update_state(state: TableState, row: dict[str, str], fmt: str) -> None:
    state.facts_seen += 1
    number = to_float(row.get("value"))
    if number is None:
        state.nonnumeric += 1
        state.remember("nonnumeric_values", row)
        return
    if number != 0:
        state.nonzero += 1
    if number < 0:
        state.negative += 1
        state.remember("negative_values", row)

    key = coord_hash(row)
    value_text = str(row.get("value", "")).strip()
    prior = state.coord_values.get(key)
    if prior is None:
        state.coord_values[key] = value_text
    elif prior != value_text:
        state.duplicate_conflicts += 1
        state.remember("duplicate_conflicting_coordinates", row)

    age_label = norm(row.get("resolved_age_label_std"))
    if age_label and not is_total_age(row):
        state.age_labels.add(age_label)

    if fmt in {"distribution_of_1000", "distribution_of_10000"}:
        denom = 1000 if fmt == "distribution_of_1000" else 10000
        if number < 0 or number > denom:
            state.distribution_out_of_range += 1
            state.remember("distribution_value_out_of_range", row)

    if fmt in {
        "additive_age_sex_counts",
        "age_civil_condition_counts",
        "age_by_religion_counts",
        "mortality_counts",
        "disease_specific_mortality",
    }:
        sex = normalized_sex(row.get("sex", ""))
        if sex:
            state.sex_groups[coord_hash(row, drop_sex=True)][sex] = number

    if fmt == "age_civil_condition_counts":
        civil = norm(row.get("civil_condition"))
        if civil in {"married", "widowed", "unmarried"} and not is_total_age(row):
            repeat_key = hash_key(
                [
                    norm(row.get("section")),
                    civil,
                    normalized_sex(row.get("sex", "")),
                    norm(row.get("column_group_std")),
                    norm(row.get("sub_column_std")),
                ]
            )
            state.civil_repeat_values[repeat_key].add(str(int(number)) if number.is_integer() else str(number))
        if civil in {"married", "widowed"} and is_child_age(row) and number >= 1000:
            state.massive_child_married_widowed += 1
            state.remember("massive_child_married_widowed", row)

    if fmt in {"mortality_rates", "disease_specific_mortality", "other_mortality"} and number in {0.0, 1.0, 100.0}:
        state.mortality_small_values[number] += 1
        state.remember("repeated_mortality_rate_like_values", row)

    if fmt == "disease_specific_mortality":
        label = norm(row.get("row_label_std") or row.get("row_label_raw"))
        if label and not re.fullmatch(r"\d{4}", label):
            copied_key = hash_key([norm(row.get("column_group_std")), norm(row.get("sub_column_std")), normalized_sex(row.get("sex", ""))])
            state.copied_row_values[copied_key].add(str(int(number)) if number.is_integer() else str(number))


def load_buckets() -> dict[str, str]:
    buckets: dict[str, str] = {}
    for row in read_csv(OUT / "data" / "tables_confirmed.csv"):
        buckets[row["table_id"]] = "confirmed"
    for row in read_csv(OUT / "data" / "review_needed.csv"):
        buckets.setdefault(row["table_id"], "sendout_review_needed")
    for row in read_csv(OUT / "data" / "exclusions.csv"):
        buckets.setdefault(row["table_id"], "sendout_excluded")
    return buckets


def flags_for(row: dict[str, str], state: TableState) -> list[tuple[str, str, str]]:
    fmt = row["format_class"]
    flags: list[tuple[str, str, str]] = []
    if state.facts_seen == 0:
        flags.append(("fail", "no_facts_seen", "No fact rows found in the source fact files."))
    if state.nonnumeric:
        flags.append(("fail", "nonnumeric_values", f"{state.nonnumeric} fact values were not parseable as numbers."))
    if state.negative and fmt not in {"rate_or_ratio_table"}:
        flags.append(("fail", "negative_values", f"{state.negative} negative values in a format where negatives are not expected."))
    if state.nonzero == 0 and state.facts_seen:
        flags.append(("fail", "all_zero_numeric_content", "All numeric values are zero."))
    if state.duplicate_conflicts:
        flags.append(("fail", "duplicate_conflicting_coordinates", f"{state.duplicate_conflicts} duplicate coordinates have conflicting values."))
    if state.distribution_out_of_range:
        flags.append(("fail", "distribution_value_out_of_range", f"{state.distribution_out_of_range} values fall outside the expected distribution denominator."))
    if state.massive_child_married_widowed:
        flags.append(("fail", "massive_child_married_widowed", f"{state.massive_child_married_widowed} child-age married/widowed cells are >= 1000."))

    sex_sum_mismatch = 0
    sex_zero_imbalance = 0
    for values in state.sex_groups.values():
        if {"persons", "male", "female"} <= set(values):
            persons = values["persons"]
            male = values["male"]
            female = values["female"]
            if abs(persons - male - female) > max(1.0, abs(persons) * 0.02):
                sex_sum_mismatch += 1
            if persons >= 10000 and (male == 0 or female == 0):
                sex_zero_imbalance += 1
    if sex_sum_mismatch:
        flags.append(("fail", "sex_sum_mismatch", f"{sex_sum_mismatch} Persons/All values do not match Male + Female within tolerance."))
    title = norm(row.get("title"))
    if sex_zero_imbalance and not re.search(r"\b(female|women|male|men|wife|wives|husband)\b", title):
        flags.append(("fail", "extreme_sex_imbalance", f"{sex_zero_imbalance} large rows have one sex at zero."))

    if fmt == "age_civil_condition_counts" and len(state.age_labels) >= 6:
        constant_civil = sum(1 for values in state.civil_repeat_values.values() if len(values) == 1)
        if constant_civil:
            flags.append(("warn", "constant_civil_condition_columns", f"{constant_civil} civil-condition slices are constant across age-specific rows."))

    if state.mortality_small_values:
        value, count = state.mortality_small_values.most_common(1)[0]
        if count >= 10:
            flags.append(("fail", "repeated_mortality_rate_like_values", f"{count} mortality/rate cells repeat {value}."))

    if fmt == "disease_specific_mortality":
        constant_death_slices = sum(1 for values in state.copied_row_values.values() if len(values) == 1)
        if constant_death_slices and state.facts_seen >= 10:
            flags.append(("warn", "possible_copied_mortality_totals", f"{constant_death_slices} mortality slices are constant across disease/year-like rows."))

    if row.get("layer") == "assembled":
        flags.append(("review", "assembled_requires_visual_review", "Assembled/multipage table requires visual semantic review before confirmation."))
    if truthy(row.get("needs_manual_review")):
        flags.append(("review", "manual_review_signal", "Existing team-ready catalog marks this table for manual review."))
    if row.get("failed_checks") not in {"", "0"}:
        flags.append(("fail", "existing_failed_checks", f"Existing constraint failed_checks={row.get('failed_checks')}."))
    if row.get("quality_tier") not in {"strict_validated", ""}:
        flags.append(("review", "non_strict_quality_tier", f"quality_tier={row.get('quality_tier')}."))
    if not truthy(row.get("in_release_scope")):
        flags.append(("review", "outside_release_scope", "Not in release scope."))
    return flags


def status_and_recommendation(row: dict[str, str], flags: list[tuple[str, str, str]]) -> tuple[str, str]:
    severities = {severity for severity, _, _ in flags}
    flag_names = {flag for _, flag, _ in flags}
    if "fail" in severities:
        status = "fail"
    elif "warn" in severities:
        status = "warn"
    elif "review" in severities:
        status = "review_only"
    else:
        status = "pass"

    if status == "pass":
        recommendation = "keep_confirmed" if row.get("sendout_bucket") == "confirmed" else "plausibility_pass_candidate"
    elif flag_names & {
        "massive_child_married_widowed",
        "sex_sum_mismatch",
        "extreme_sex_imbalance",
        "repeated_mortality_rate_like_values",
        "distribution_value_out_of_range",
        "duplicate_conflicting_coordinates",
        "nonnumeric_values",
        "negative_values",
        "all_zero_numeric_content",
    }:
        recommendation = "exclude_or_rerun_before_use"
    elif "existing_failed_checks" in flag_names:
        recommendation = "repair_or_rerun_before_use"
    elif "assembled_requires_visual_review" in flag_names:
        recommendation = "manual_visual_review_required"
    elif "manual_review_signal" in flag_names:
        recommendation = "manual_review_needed_but_number_sense_ok"
    else:
        recommendation = "keep_out_of_confirmed_until_scope_decision"
    return status, recommendation


def main() -> None:
    master = read_csv(TEAM / "catalogs" / "all_tables_master.csv")
    buckets = load_buckets()
    rows_by_id: dict[str, dict[str, str]] = {}
    states: dict[str, TableState] = {}
    for row in master:
        tid = row["table_id"]
        row = dict(row)
        row["format_class"] = infer_format(row)
        row["sendout_bucket"] = buckets.get(tid, "not_in_sendout_outputs")
        rows_by_id[tid] = row
        states[tid] = TableState()

    for layer, dataset in DATASETS.items():
        layer_ids = {tid for tid, row in rows_by_id.items() if row.get("layer") == layer}
        if not layer_ids:
            continue
        path = dataset / "cell_facts_rich.csv"
        with path.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for fact in reader:
                tid = fact.get("table_id", "")
                if tid in layer_ids:
                    update_state(states[tid], fact, rows_by_id[tid]["format_class"])

    audit_rows = []
    issue_rows = []
    review_rows = []
    for tid, row in sorted(rows_by_id.items(), key=lambda kv: (kv[1].get("sendout_bucket", ""), kv[1].get("layer", ""), kv[1].get("province", ""), kv[1].get("year", ""), kv[1].get("title", ""))):
        state = states[tid]
        flags = flags_for(row, state)
        status, recommendation = status_and_recommendation(row, flags)
        flag_names = [flag for _, flag, _ in flags]
        audit = {
            "table_id": tid,
            "layer": row.get("layer", ""),
            "sendout_bucket": row.get("sendout_bucket", ""),
            "province": row.get("province", ""),
            "year": row.get("year", ""),
            "title": row.get("title", ""),
            "table_family": row.get("table_family", ""),
            "format_class": row.get("format_class", ""),
            "in_release_scope": row.get("in_release_scope", ""),
            "quality_tier": row.get("quality_tier", ""),
            "failed_checks": row.get("failed_checks", ""),
            "needs_manual_review": row.get("needs_manual_review", ""),
            "facts_seen": state.facts_seen,
            "nonzero_values": state.nonzero,
            "plausibility_status": status,
            "recommendation": recommendation,
            "flags": ";".join(flag_names),
            "flag_count": len(flags),
            "existing_qc_flags": row.get("qc_flags", ""),
        }
        audit_rows.append(audit)
        if row.get("sendout_bucket") != "confirmed" or truthy(row.get("needs_manual_review")) or row.get("failed_checks") not in {"", "0"}:
            review_rows.append({field: audit.get(field, row.get(field, "")) for field in REVIEW_FIELDS})
        for severity, flag, detail in flags:
            issue_rows.append(
                {
                    "table_id": tid,
                    "layer": row.get("layer", ""),
                    "sendout_bucket": row.get("sendout_bucket", ""),
                    "format_class": row.get("format_class", ""),
                    "severity": severity,
                    "flag": flag,
                    "detail": detail,
                    "example": state.examples.get(flag, ""),
                }
            )

    qa_dir = OUT / "qa"
    review_dir = OUT / "review"
    write_csv(qa_dir / "all_tables_plausibility_audit.csv", audit_rows, AUDIT_FIELDS)
    write_csv(qa_dir / "all_tables_plausibility_issues.csv", issue_rows, ISSUE_FIELDS)
    write_csv(review_dir / "review_queue_plausibility_triage.csv", review_rows, REVIEW_FIELDS)

    summary = {
        "total_tables_audited": len(audit_rows),
        "facts_seen_total": sum(int(r["facts_seen"]) for r in audit_rows),
        "by_sendout_bucket": dict(sorted(Counter(r["sendout_bucket"] for r in audit_rows).items())),
        "by_status": dict(sorted(Counter(r["plausibility_status"] for r in audit_rows).items())),
        "by_recommendation": dict(sorted(Counter(r["recommendation"] for r in audit_rows).items())),
        "issue_counts": dict(sorted(Counter(r["flag"] for r in issue_rows).items())),
        "confirmed_with_fail_or_warn": sum(1 for r in audit_rows if r["sendout_bucket"] == "confirmed" and r["plausibility_status"] in {"fail", "warn"}),
        "review_queue_number_sense_pass": sum(1 for r in review_rows if r["plausibility_status"] == "pass"),
        "review_queue_repair_or_exclude": sum(1 for r in review_rows if r["recommendation"] in {"exclude_or_rerun_before_use", "repair_or_rerun_before_use"}),
    }
    write_json(qa_dir / "all_tables_plausibility_summary.json", summary)
    lines = ["# All Tables Plausibility Audit", ""]
    for key in ["total_tables_audited", "facts_seen_total", "confirmed_with_fail_or_warn", "review_queue_number_sense_pass", "review_queue_repair_or_exclude"]:
        lines.append(f"- {key}: {summary[key]}")
    lines.append("")
    lines.append("## Status Counts")
    for key, value in summary["by_status"].items():
        lines.append(f"- {key}: {value}")
    lines.append("")
    lines.append("## Recommendation Counts")
    for key, value in summary["by_recommendation"].items():
        lines.append(f"- {key}: {value}")
    (qa_dir / "all_tables_plausibility_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
