"""Export assembled multi-page tables into inference-style source/fact tables.

This sits on top of assemble_multipage_tables.py and preserves page-level
provenance while exposing assembled tables in the same long format used by
export_inference_dataset.py.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from export_inference_dataset import (
    FACT_COLUMNS,
    RICH_FACT_COLUMNS,
    SOURCE_COLUMNS,
    canonical_civil_condition,
    canonical_religion,
    canonical_sex,
    coerce_list,
    dedupe_facts,
    extract_row_label,
    guess_dimension,
    infer_constraint_regime,
    infer_denominator_hint,
    iter_numeric_cells_from_row,
    measure_unit,
    normalize_age_label,
    normalize_col_group,
    normalize_text,
    parse_age_label,
    resolve_age_fields,
    reinterpret_quality_status,
    safe_json,
    source_lookup_keys,
    source_path_metadata,
    summarize_constraint_failures,
)


ASSEMBLED_EXTRA_COLUMNS = [
    "assembled_group_id",
    "assembled_n_pages",
    "assembled_overall_strategy",
    "assembled_signals",
    "assembled_section_strategy_counts",
    "assembled_input_constraint_checks",
    "assembled_input_constraint_failures",
    "assembled_constraint_checks",
    "assembled_constraint_failures",
    "assembled_all_passed",
    "assembled_failure_improvement",
    "assembled_member_source_jsons",
    "assembled_member_source_images",
    "compiled_member_count",
    "compiled_member_titles",
    "compiled_member_queue_titles",
    "compiled_member_table_families",
    "compiled_member_source_files",
    "compiled_member_manifest_classifications",
    "compiled_member_column_groups",
    "compiled_member_provinces",
    "compiled_member_years",
    "compiled_member_quality_statuses",
    "compiled_member_effective_quality_statuses",
    "compiled_section_names",
    "compiled_title_variants_count",
    "compiled_source_file_count",
    "compiled_column_group_count",
]

ASSEMBLED_SOURCE_COLUMNS = SOURCE_COLUMNS + [col for col in ASSEMBLED_EXTRA_COLUMNS if col not in SOURCE_COLUMNS]
FACT_DETAIL_COLUMNS = [
    "section",
    "section_type",
    "row_label_raw",
    "row_label_std",
    "row_dimension",
    "row_group_value",
    "age_label_std",
    "age_lower",
    "age_upper",
    "age_open_ended",
    "age_total",
    "age_not_stated",
    "column_group_raw",
    "column_group_std",
    "column_dimension",
    "column_group_value",
    "column_age_label_std",
    "column_age_lower",
    "column_age_upper",
    "column_age_open_ended",
    "column_age_total",
    "column_age_not_stated",
    "resolved_age_axis",
    "resolved_age_label_std",
    "resolved_age_lower",
    "resolved_age_upper",
    "resolved_age_open_ended",
    "resolved_age_total",
    "resolved_age_not_stated",
    "sub_column_raw",
    "sub_column_std",
    "sex",
    "civil_condition",
    "religion",
    "value",
]
ASSEMBLED_FACT_COLUMNS = ["table_id"] + FACT_DETAIL_COLUMNS
ASSEMBLED_RICH_FACT_COLUMNS = ASSEMBLED_SOURCE_COLUMNS + [col for col in RICH_FACT_COLUMNS if col not in ASSEMBLED_SOURCE_COLUMNS]

MEMBER_PAGE_COLUMNS = [
    "assembled_group_id",
    "table_id",
    "member_index",
    "source_json",
    "source_image",
    "source_file",
    "title",
    "queue_title",
    "table_family",
    "province",
    "year",
    "quality_status",
    "effective_quality_status",
]


def first_nonblank(row: dict[str, Any] | None, keys: list[str]) -> str:
    row = row or {}
    for key in keys:
        value = row.get(key)
        if str(value or "").strip():
            return str(value).strip()
    return ""


def truthy(value: object) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def decode_jsonish(value: object, fallback: Any) -> Any:
    if value in (None, ""):
        return fallback
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return fallback
        try:
            return json.loads(text)
        except Exception:
            return fallback
    return fallback


def load_assembled_groups(path: Path) -> list[dict[str, Any]]:
    groups = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                groups.append(json.loads(line))
    return groups


def add_index_row(index: dict[str, dict[str, Any]], key: str, row: dict[str, Any]) -> None:
    if key and key not in index:
        index[key] = row


def load_page_csv_index(paths: list[Path]) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for path in paths:
        with path.open(encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                for key in source_lookup_keys(first_nonblank(row, ["source_json", "result_json"])):
                    add_index_row(index, key, row)
    return index


def load_page_jsonl_index(paths: list[Path]) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for path in paths:
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                for key in source_lookup_keys(first_nonblank(row, ["source_json", "result_json"])):
                    add_index_row(index, key, row)
    return index


def lookup_row(source_json: str, csv_index: dict[str, dict[str, Any]], jsonl_index: dict[str, dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    for key in source_lookup_keys(source_json):
        csv_row = csv_index.get(key)
        jsonl_row = jsonl_index.get(key)
        if csv_row or jsonl_row:
            return csv_row or {}, jsonl_row or {}
    return {}, {}


def unique_nonblank(values: list[object]) -> list[str]:
    seen = set()
    output = []
    for value in values:
        text = normalize_text(value)
        if not text or text in seen:
            continue
        seen.add(text)
        output.append(text)
    return output


def unique_nonblank_from_lists(values: list[object]) -> list[str]:
    flattened = []
    for value in values:
        if isinstance(value, list):
            flattened.extend(value)
        else:
            flattened.append(value)
    return unique_nonblank(flattened)


def joined_unique(values: list[object], sep: str = " | ") -> str:
    return sep.join(unique_nonblank(values))


def title_quality_score(value: object) -> float:
    text = normalize_text(value)
    if not text:
        return float("-inf")
    lower = text.lower()
    letters = sum(ch.isalpha() for ch in text)
    digits = sum(ch.isdigit() for ch in text)
    weird = sum(not (ch.isalnum() or ch.isspace() or ch in ".,:;()-/&[]'\"") for ch in text)
    punctuation_runs = len(re.findall(r"[`|\\\\]{2,}", text))
    score = 0.0
    score += min(letters, 120)
    score -= weird * 6
    score -= punctuation_runs * 10
    score -= text.count("|") * 2
    score -= text.count("`") * 4
    score -= text.count("\\") * 4
    if "table" in lower:
        score += 20
    if "age" in lower:
        score += 12
    if "civil condition" in lower:
        score += 8
    if "religion" in lower:
        score += 4
    if "continued" in lower or "contd" in lower or "concld" in lower:
        score += 2
    if digits and letters == 0:
        score -= 30
    if len(text) < 12:
        score -= 8
    if len(text) > 220:
        score -= (len(text) - 220) / 4
    return score


def choose_best_title(candidates: list[object], fallback: str = "") -> str:
    usable = unique_nonblank(candidates)
    if not usable:
        return normalize_text(fallback)
    return max(usable, key=title_quality_score)


def numeric_sum(values: list[object]) -> int | float | str:
    total = 0.0
    found = False
    for value in values:
        if value in (None, ""):
            continue
        try:
            total += float(value)
            found = True
        except Exception:
            continue
    if not found:
        return ""
    if total.is_integer():
        return int(total)
    return round(total, 6)


def numeric_avg(values: list[object]) -> int | float | str:
    usable = []
    for value in values:
        if value in (None, ""):
            continue
        try:
            usable.append(float(value))
        except Exception:
            continue
    if not usable:
        return ""
    avg = sum(usable) / len(usable)
    if avg.is_integer():
        return int(avg)
    return round(avg, 6)


def stable_table_id(group_id: str) -> str:
    digest = hashlib.sha1(group_id.encode("utf-8")).hexdigest()[:16]
    return f"assembled::{digest}"


def classify_assembled_quality(report: dict[str, Any]) -> tuple[str, int, int]:
    total_checks = int(report.get("total_checks") or 0)
    failed = int(report.get("failed") or 0)
    if total_checks > 0 and failed == 0:
        return "validated_pass", total_checks, failed
    if total_checks > 0 and failed > 0:
        return "validated_fail", total_checks, failed
    return "unvalidated", total_checks, failed


def page_contexts_for_group(
    group: dict[str, Any],
    page_csv_index: dict[str, dict[str, Any]],
    page_jsonl_index: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    contexts = []
    for idx, source_json in enumerate(group.get("page_source_jsons") or []):
        csv_row, jsonl_row = lookup_row(source_json, page_csv_index, page_jsonl_index)
        contexts.append(
            {
                "assembled_group_id": group.get("group_id") or "",
                "member_index": idx,
                "source_json": source_json,
                "csv_row": csv_row,
                "jsonl_row": jsonl_row,
                "raw_metadata": decode_jsonish(
                    jsonl_row.get("raw_metadata") if jsonl_row else None,
                    decode_jsonish(csv_row.get("metadata_json") if csv_row else None, {}),
                ),
                "raw_acceptance": decode_jsonish(
                    jsonl_row.get("raw_acceptance") if jsonl_row else None,
                    {},
                ),
                "raw_constraints": decode_jsonish(
                    jsonl_row.get("raw_constraints") if jsonl_row else None,
                    decode_jsonish(csv_row.get("constraints_json") if csv_row else None, {}),
                ),
                "raw_queue_row": decode_jsonish(
                    jsonl_row.get("raw_queue_row") if jsonl_row else None,
                    decode_jsonish(csv_row.get("queue_row_json") if csv_row else None, {}),
                ),
                "raw_acceptance_signals": decode_jsonish(
                    csv_row.get("acceptance_signals_json") if csv_row else None,
                    {},
                ),
            }
        )
    return contexts


def aggregate_queue_tags(contexts: list[dict[str, Any]]) -> str:
    values = []
    for ctx in contexts:
        queue_row = ctx["raw_queue_row"] or {}
        if queue_row.get("query_tags"):
            values.extend(coerce_list(queue_row.get("query_tags")))
        else:
            values.extend(coerce_list(ctx["csv_row"].get("queue_tags") if ctx["csv_row"] else None))
    return joined_unique(values)


def aggregate_member_pages_rows(contexts: list[dict[str, Any]], table_id: str) -> list[dict[str, Any]]:
    rows = []
    for ctx in contexts:
        csv_row = ctx["csv_row"] or {}
        rows.append(
            {
                "assembled_group_id": ctx.get("assembled_group_id") or table_id,
                "table_id": table_id,
                "member_index": ctx["member_index"],
                "source_json": ctx["source_json"],
                "source_image": first_nonblank(csv_row, ["source_image", "image_path"]),
                "source_file": first_nonblank(csv_row, ["source_file"]),
                "title": first_nonblank(csv_row, ["title", "queue_title"]),
                "queue_title": first_nonblank(csv_row, ["queue_title"]),
                "table_family": first_nonblank(csv_row, ["table_family", "rescue_family", "canonical_topic"]),
                "province": first_nonblank(csv_row, ["province"]),
                "year": first_nonblank(csv_row, ["year"]),
                "quality_status": first_nonblank(csv_row, ["quality_status"]),
                "effective_quality_status": first_nonblank(csv_row, ["effective_quality_status"]),
            }
        )
    return rows


def build_assembled_source_row(
    group: dict[str, Any],
    contexts: list[dict[str, Any]],
    include_unvalidated: bool,
    include_failed: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    table_id = stable_table_id(group["group_id"])
    primary_csv = next((ctx["csv_row"] for ctx in contexts if ctx["csv_row"]), {})
    assembled_meta = group.get("assembled_metadata") or {}
    assembled_report = group.get("assembled_constraint_report") or {}
    assembled_constraints = group.get("assembled_constraints") or {}
    queue_rows = [ctx["raw_queue_row"] for ctx in contexts if ctx["raw_queue_row"]]
    queue_row_primary = queue_rows[0] if queue_rows else {}
    member_titles = unique_nonblank([first_nonblank(ctx["csv_row"], ["title", "queue_title"]) for ctx in contexts])
    member_queue_titles = unique_nonblank(
        [queue.get("title") for queue in queue_rows]
        + [first_nonblank(ctx["csv_row"], ["queue_title"]) for ctx in contexts]
    )
    title = choose_best_title(
        member_titles
        + member_queue_titles
        + [group.get("title"), assembled_meta.get("title"), first_nonblank(primary_csv, ["title", "queue_title"])],
        fallback=group.get("title") or assembled_meta.get("title") or first_nonblank(primary_csv, ["title", "queue_title"]),
    )
    table_family = group.get("table_family") or first_nonblank(primary_csv, ["table_family"])
    meta_for_measure = {
        "title": title,
        "data_type": assembled_meta.get("data_type") or first_nonblank(primary_csv, ["data_type"]),
        "column_groups": assembled_meta.get("column_groups") or coerce_list(first_nonblank(primary_csv, ["column_groups"])),
        "row_axis_type": assembled_meta.get("row_axis_type") or first_nonblank(primary_csv, ["row_axis_type"]),
        "row_label_name": assembled_meta.get("row_label_name") or first_nonblank(primary_csv, ["row_label_name"]),
    }

    quality_status, n_checks, failed_checks = classify_assembled_quality(assembled_report)
    allowed_quality = {"validated_pass"}
    if include_unvalidated:
        allowed_quality.add("unvalidated")
    if include_failed:
        allowed_quality.add("validated_fail")

    measure_kind = measure_unit(meta_for_measure, table_family)
    denominator_hint = infer_denominator_hint(meta_for_measure)
    constraint_regime = infer_constraint_regime(table_family, measure_kind, meta_for_measure)
    effective_quality_status = reinterpret_quality_status(
        quality_status,
        constraint_regime,
        {"constraints": assembled_report},
        table_family,
    )

    member_source_images = group.get("page_source_images") or [
        first_nonblank(ctx["csv_row"], ["source_image", "image_path"])
        for ctx in contexts
        if first_nonblank(ctx["csv_row"], ["source_image", "image_path"])
    ]
    source_image = member_source_images[0] if member_source_images else ""
    source_image_name, source_table_dir, source_table_dir_name = source_path_metadata(source_image)

    raw_metas = [ctx["raw_metadata"] for ctx in contexts if ctx["raw_metadata"]]
    raw_acceptance = [ctx["raw_acceptance"] for ctx in contexts if ctx["raw_acceptance"]]
    raw_constraints = [ctx["raw_constraints"] for ctx in contexts if ctx["raw_constraints"]]
    raw_acceptance_signals = [ctx["raw_acceptance_signals"] for ctx in contexts if ctx["raw_acceptance_signals"]]

    member_families = unique_nonblank([first_nonblank(ctx["csv_row"], ["table_family"]) for ctx in contexts])
    member_source_files = unique_nonblank([first_nonblank(ctx["csv_row"], ["source_file"]) for ctx in contexts])
    member_manifest_classifications = unique_nonblank([first_nonblank(ctx["csv_row"], ["manifest_classification"]) for ctx in contexts])
    member_column_groups = unique_nonblank_from_lists(
        [assembled_meta.get("column_groups") or []]
        + [coerce_list(first_nonblank(ctx["csv_row"], ["column_groups"])) for ctx in contexts]
    )
    member_provinces = unique_nonblank([group.get("province")] + [first_nonblank(ctx["csv_row"], ["province"]) for ctx in contexts])
    member_years = unique_nonblank([group.get("year")] + [first_nonblank(ctx["csv_row"], ["year"]) for ctx in contexts])
    member_quality_statuses = unique_nonblank([first_nonblank(ctx["csv_row"], ["quality_status"]) for ctx in contexts])
    member_effective_quality_statuses = unique_nonblank([first_nonblank(ctx["csv_row"], ["effective_quality_status"]) for ctx in contexts])

    source_facts = []
    sample_row_labels = []
    nonblank_rows = 0
    blank_rows = 0
    numeric_cells = 0
    zero_cells = 0
    negative_cells = 0
    noninteger_cells = 0
    age_like_rows = 0
    n_rows = 0
    section_names_seen = set()
    row_labels_seen = set()
    column_groups_seen = set()
    sub_columns_seen = set()

    for section in group.get("sections") or []:
        section_name = normalize_text(section.get("name") or "")
        if section_name:
            section_names_seen.add(section_name)
        section_dimension, section_group_value = guess_dimension(section_name, table_family)
        if section_dimension == "religion":
            section_type = "Religion"
        elif section_dimension:
            section_type = section_dimension
        else:
            section_type = "Total" if not section_name or normalize_text(section_name).lower() == normalize_text(title).lower() else "Other"

        for row in section.get("rows") or []:
            n_rows += 1
            row_label_raw = extract_row_label(row)
            row_label_std = normalize_age_label(row_label_raw)
            if row_label_std and len(sample_row_labels) < 5:
                sample_row_labels.append(row_label_std)
            if row_label_std:
                row_labels_seen.add(row_label_std)

            row_dimension, row_group_value = guess_dimension(row_label_std, table_family)
            age_info = parse_age_label(row_label_std)
            if age_info["is_age_like"]:
                age_like_rows += 1

            row_facts = []
            for column_group_raw, sub_column_raw, value in iter_numeric_cells_from_row(row):
                numeric_cells += 1
                column_group_std = normalize_col_group(column_group_raw)
                sub_column_std = normalize_col_group(sub_column_raw) if sub_column_raw else ""
                if column_group_std:
                    column_groups_seen.add(column_group_std)
                if sub_column_std:
                    sub_columns_seen.add(sub_column_std)
                column_dimension, column_group_value = guess_dimension(column_group_std, table_family)
                sex = canonical_sex(sub_column_std)
                civil_condition = canonical_civil_condition(column_group_std)

                if value == 0:
                    zero_cells += 1
                if isinstance(value, (int, float)) and value < 0:
                    negative_cells += 1
                if isinstance(value, float) and not value.is_integer():
                    noninteger_cells += 1

                religion = ""
                for candidate in (
                    row_group_value if row_dimension == "religion" else "",
                    column_group_value if column_dimension == "religion" else "",
                    section_group_value if section_dimension == "religion" else "",
                    canonical_religion(title),
                ):
                    if candidate:
                        religion = candidate
                        break

                column_age_info = parse_age_label(column_group_std)
                resolved_age = resolve_age_fields(age_info, column_age_info)

                row_facts.append(
                    {
                        "table_id": table_id,
                        "source_json": table_id,
                        "source_image": source_image,
                        "source_image_name": source_image_name,
                        "source_table_dir_name": source_table_dir_name,
                        "source_file": group.get("source_file") or first_nonblank(primary_csv, ["source_file"]),
                        "province": group.get("province") or first_nonblank(primary_csv, ["province"]),
                        "year": group.get("year") or first_nonblank(primary_csv, ["year"]),
                        "region": first_nonblank(primary_csv, ["region"]),
                        "geo_name": first_nonblank(primary_csv, ["geo_name"]),
                        "geo_level": first_nonblank(primary_csv, ["geo_level"]),
                        "title": title,
                        "queue_title": queue_row_primary.get("title") or first_nonblank(primary_csv, ["queue_title"]),
                        "queue_display_identifier": queue_row_primary.get("display_identifier") or first_nonblank(primary_csv, ["queue_display_identifier"]),
                        "queue_tags": aggregate_queue_tags(contexts),
                        "canonical_topic": queue_row_primary.get("canonical_topic") or first_nonblank(primary_csv, ["canonical_topic"]),
                        "relevance_tier": queue_row_primary.get("relevance_tier") or first_nonblank(primary_csv, ["relevance_tier"]),
                        "table_family": table_family,
                        "constraint_regime": constraint_regime,
                        "denominator_hint": denominator_hint,
                        "data_type": meta_for_measure.get("data_type") or "",
                        "measure_unit": measure_kind,
                        "row_axis_type": meta_for_measure.get("row_axis_type") or "",
                        "row_label_name": meta_for_measure.get("row_label_name") or "",
                        "extraction_method": "assembled_multipage",
                        "acceptance_decision": "assembled_constraint_pass" if assembled_report.get("all_passed") else ("assembled_constraint_fail" if assembled_report.get("total_checks") else "assembled_unvalidated"),
                        "trust_score": numeric_avg([ctx["csv_row"].get("trust_score") for ctx in contexts if ctx["csv_row"]]),
                        "quality_status": quality_status,
                        "effective_quality_status": effective_quality_status,
                        "section": section_name,
                        "section_type": section_type,
                        "row_label_raw": row_label_raw,
                        "row_label_std": row_label_std,
                        "row_dimension": row_dimension,
                        "row_group_value": row_group_value,
                        "age_label_std": age_info["age_label_std"],
                        "age_lower": age_info["age_lower"],
                        "age_upper": age_info["age_upper"],
                        "age_open_ended": age_info["age_open_ended"],
                        "age_total": age_info["age_total"],
                        "age_not_stated": age_info["age_not_stated"],
                        "column_group_raw": column_group_raw,
                        "column_group_std": column_group_std,
                        "column_dimension": column_dimension,
                        "column_group_value": column_group_value,
                        "column_age_label_std": column_age_info["age_label_std"],
                        "column_age_lower": column_age_info["age_lower"],
                        "column_age_upper": column_age_info["age_upper"],
                        "column_age_open_ended": column_age_info["age_open_ended"],
                        "column_age_total": column_age_info["age_total"],
                        "column_age_not_stated": column_age_info["age_not_stated"],
                        "resolved_age_axis": resolved_age["resolved_age_axis"],
                        "resolved_age_label_std": resolved_age["resolved_age_label_std"],
                        "resolved_age_lower": resolved_age["resolved_age_lower"],
                        "resolved_age_upper": resolved_age["resolved_age_upper"],
                        "resolved_age_open_ended": resolved_age["resolved_age_open_ended"],
                        "resolved_age_total": resolved_age["resolved_age_total"],
                        "resolved_age_not_stated": resolved_age["resolved_age_not_stated"],
                        "sub_column_raw": sub_column_raw,
                        "sub_column_std": sub_column_std,
                        "sex": sex,
                        "civil_condition": civil_condition,
                        "religion": religion,
                        "value": value,
                    }
                )

            if not row_facts:
                blank_rows += 1
                continue

            nonblank_rows += 1
            source_facts.extend(row_facts)

    deduped_facts, duplicate_exact_cells, duplicate_conflict_cells = dedupe_facts(source_facts)
    failure_summary = summarize_constraint_failures(assembled_report.get("failures") or [])

    qc_flags = ["assembled_table", "multipage_group"]
    if quality_status not in allowed_quality:
        qc_flags.append("filtered_quality_status")
    if nonblank_rows == 0 or numeric_cells == 0:
        qc_flags.append("blank_source")
    if len(set(f["row_label_std"] for f in deduped_facts if f["row_label_std"])) <= 1 or len(deduped_facts) <= 2:
        qc_flags.append("degenerate_source")
    if duplicate_conflict_cells > 0:
        qc_flags.append("duplicate_conflict")
    if int(group.get("n_pages") or 0) <= 1:
        qc_flags.append("single_page_group")
    if assembled_report.get("failed"):
        qc_flags.append("assembled_constraint_failures")
    if group.get("overall_strategy") == "mixed":
        qc_flags.append("mixed_strategy")

    keep_for_inference = int(not any(flag in qc_flags for flag in (
        "filtered_quality_status",
        "blank_source",
        "degenerate_source",
        "duplicate_conflict",
        "assembled_constraint_failures",
    )))

    acceptance_reasons = unique_nonblank(
        [
            *[reason for raw in raw_acceptance for reason in (raw.get("reasons") or [])],
            f"assembled_strategy:{group.get('overall_strategy')}",
            f"constraint_improvement:{int(group.get('input_constraint_failures') or 0) - int(assembled_report.get('failed') or 0)}",
        ]
    )

    metadata_json = {
        "assembled_metadata": assembled_meta,
        "member_titles": member_titles,
        "member_queue_titles": member_queue_titles,
        "member_table_families": member_families,
        "member_source_files": member_source_files,
        "member_manifest_classifications": member_manifest_classifications,
        "member_column_groups": member_column_groups,
        "member_provinces": member_provinces,
        "member_years": member_years,
        "member_quality_statuses": member_quality_statuses,
        "member_effective_quality_statuses": member_effective_quality_statuses,
        "compiled_section_names": sorted(section_names_seen),
        "member_raw_metadata": raw_metas,
    }
    acceptance_signals_json = {
        "member_acceptance_signals": raw_acceptance_signals,
        "member_acceptance": raw_acceptance,
        "assembled_signals": group.get("signals") or [],
    }
    constraints_json = {
        "assembled_constraints": assembled_constraints,
        "assembled_constraint_report": assembled_report,
        "member_constraints": raw_constraints,
    }
    queue_row_json = {
        "primary_queue_row": queue_row_primary,
        "member_queue_rows": queue_rows,
    }

    source_row = {
        "table_id": table_id,
        "source_json": table_id,
        "source_image": source_image,
        "source_image_name": source_image_name,
        "source_table_dir": source_table_dir,
        "source_table_dir_name": source_table_dir_name,
        "source_file": group.get("source_file") or first_nonblank(primary_csv, ["source_file"]),
        "manifest_classification": joined_unique([first_nonblank(ctx["csv_row"], ["manifest_classification"]) for ctx in contexts]),
        "province": group.get("province") or first_nonblank(primary_csv, ["province"]),
        "year": group.get("year") or first_nonblank(primary_csv, ["year"]),
        "region": first_nonblank(primary_csv, ["region"]),
        "geo_name": first_nonblank(primary_csv, ["geo_name"]),
        "geo_level": first_nonblank(primary_csv, ["geo_level"]),
        "title": title,
        "queue_title": queue_row_primary.get("title") or first_nonblank(primary_csv, ["queue_title"]),
        "queue_display_identifier": queue_row_primary.get("display_identifier") or first_nonblank(primary_csv, ["queue_display_identifier"]),
        "queue_tags": aggregate_queue_tags(contexts),
        "canonical_topic": queue_row_primary.get("canonical_topic") or first_nonblank(primary_csv, ["canonical_topic"]),
        "relevance_tier": queue_row_primary.get("relevance_tier") or first_nonblank(primary_csv, ["relevance_tier"]),
        "queue_dir_name_match": int(any(truthy(first_nonblank(ctx["csv_row"], ["queue_dir_name_match"])) for ctx in contexts)),
        "table_family": table_family,
        "constraint_regime": constraint_regime,
        "denominator_hint": denominator_hint,
        "data_type": meta_for_measure.get("data_type") or "",
        "measure_unit": measure_kind,
        "row_axis_type": meta_for_measure.get("row_axis_type") or "",
        "row_label_name": meta_for_measure.get("row_label_name") or "",
        "extraction_method": "assembled_multipage",
        "api_calls": numeric_sum([ctx["csv_row"].get("api_calls") for ctx in contexts if ctx["csv_row"]]),
        "elapsed_seconds": numeric_sum([ctx["csv_row"].get("elapsed_seconds") for ctx in contexts if ctx["csv_row"]]),
        "acceptance_decision": "assembled_constraint_pass" if assembled_report.get("all_passed") else ("assembled_constraint_fail" if assembled_report.get("total_checks") else "assembled_unvalidated"),
        "acceptance_next_step": "" if assembled_report.get("all_passed") else "review_assembled",
        "acceptance_version": "assembled_v1",
        "acceptance_reasons": " | ".join(acceptance_reasons),
        "trust_score": numeric_avg([ctx["csv_row"].get("trust_score") for ctx in contexts if ctx["csv_row"]]),
        "quality_status": quality_status,
        "effective_quality_status": effective_quality_status,
        "n_checks": n_checks,
        "failed_checks": failed_checks,
        "n_constraint_failures": failure_summary["n_constraint_failures"],
        "constraint_failure_levels": failure_summary["constraint_failure_levels"],
        "max_constraint_diff": failure_summary["max_constraint_diff"],
        "age_warning_count": 0,
        "repair_api_calls": numeric_sum([ctx["csv_row"].get("repair_api_calls") for ctx in contexts if ctx["csv_row"]]),
        "repair_entries_count": numeric_sum([ctx["csv_row"].get("repair_entries_count") for ctx in contexts if ctx["csv_row"]]),
        "targeted_recheck_used": int(any(truthy(first_nonblank(ctx["csv_row"], ["targeted_recheck_used"])) for ctx in contexts)),
        "cross_group_active": int(any(truthy(first_nonblank(ctx["csv_row"], ["cross_group_active"])) for ctx in contexts)),
        "cross_section_active": int(any(truthy(first_nonblank(ctx["csv_row"], ["cross_section_active"])) for ctx in contexts)),
        "extraction_mode": joined_unique([first_nonblank(ctx["csv_row"], ["extraction_mode"]) for ctx in contexts] + ["assembled"]),
        "n_sections": len(group.get("sections") or []),
        "distinct_sections": len(section_names_seen),
        "n_rows": n_rows,
        "distinct_row_labels": len(row_labels_seen),
        "numeric_cells": numeric_cells,
        "zero_cells": zero_cells,
        "negative_cells": negative_cells,
        "noninteger_cells": noninteger_cells,
        "nonblank_rows": nonblank_rows,
        "blank_rows": blank_rows,
        "age_like_rows": age_like_rows,
        "n_column_groups": len(meta_for_measure.get("column_groups") or []),
        "distinct_column_groups": len(column_groups_seen),
        "distinct_sub_columns": len(sub_columns_seen),
        "duplicate_exact_cells": duplicate_exact_cells,
        "duplicate_conflict_cells": duplicate_conflict_cells,
        "qc_flags": ";".join(sorted(unique_nonblank(qc_flags))),
        "keep_for_inference": keep_for_inference,
        "column_groups": " | ".join(meta_for_measure.get("column_groups") or []),
        "sample_row_labels": " | ".join(sample_row_labels),
        "metadata_json": safe_json(metadata_json),
        "acceptance_signals_json": safe_json(acceptance_signals_json),
        "constraints_json": safe_json(constraints_json),
        "queue_row_json": safe_json(queue_row_json),
        "assembled_group_id": group["group_id"],
        "assembled_n_pages": group.get("n_pages") or 0,
        "assembled_overall_strategy": group.get("overall_strategy") or "",
        "assembled_signals": " | ".join(group.get("signals") or []),
        "assembled_section_strategy_counts": safe_json(group.get("section_strategy_counts") or {}),
        "assembled_input_constraint_checks": group.get("input_constraint_checks") or 0,
        "assembled_input_constraint_failures": group.get("input_constraint_failures") or 0,
        "assembled_constraint_checks": assembled_report.get("total_checks") or 0,
        "assembled_constraint_failures": assembled_report.get("failed") or 0,
        "assembled_all_passed": int(bool(assembled_report.get("all_passed"))),
        "assembled_failure_improvement": int(group.get("input_constraint_failures") or 0) - int(assembled_report.get("failed") or 0),
        "assembled_member_source_jsons": " | ".join(group.get("page_source_jsons") or []),
        "assembled_member_source_images": " | ".join(member_source_images),
        "compiled_member_count": len(contexts),
        "compiled_member_titles": " | ".join(member_titles),
        "compiled_member_queue_titles": " | ".join(member_queue_titles),
        "compiled_member_table_families": " | ".join(member_families),
        "compiled_member_source_files": " | ".join(member_source_files),
        "compiled_member_manifest_classifications": " | ".join(member_manifest_classifications),
        "compiled_member_column_groups": " | ".join(member_column_groups),
        "compiled_member_provinces": " | ".join(member_provinces),
        "compiled_member_years": " | ".join(member_years),
        "compiled_member_quality_statuses": " | ".join(member_quality_statuses),
        "compiled_member_effective_quality_statuses": " | ".join(member_effective_quality_statuses),
        "compiled_section_names": " | ".join(sorted(section_names_seen)),
        "compiled_title_variants_count": len(member_titles),
        "compiled_source_file_count": len(member_source_files),
        "compiled_column_group_count": len(member_column_groups),
    }

    source_jsonl_row = {
        **source_row,
        "raw_assembled_group": group,
        "member_page_csv_rows": [ctx["csv_row"] for ctx in contexts if ctx["csv_row"]],
        "member_page_jsonl_rows": [ctx["jsonl_row"] for ctx in contexts if ctx["jsonl_row"]],
    }
    return source_row, deduped_facts, source_jsonl_row


def write_codebook(path: Path) -> None:
    with path.open("w", encoding="utf-8") as fh:
        fh.write("ASSEMBLED MULTI-PAGE EXPORT CODEBOOK\n")
        fh.write("=" * 60 + "\n\n")
        fh.write("FILES\n")
        fh.write("-" * 40 + "\n")
        fh.write("source_catalog.csv      One row per assembled table with repeated source provenance.\n")
        fh.write("source_catalog.jsonl    One row per assembled table with nested member-page context.\n")
        fh.write("cell_facts.csv          One lean row per numeric cell from assembled tables. Join to source_catalog.csv on table_id for table metadata.\n")
        fh.write("cell_facts_rich.csv     Optional denormalized fact rows with full assembled provenance; written only with --write-rich-facts.\n")
        fh.write("member_pages.csv        Mapping from assembled groups back to member page results.\n\n")
        fh.write("KEY IDEAS\n")
        fh.write("-" * 40 + "\n")
        fh.write("- table_id is a stable assembled-table key derived from group_id.\n")
        fh.write("- source_json is the assembled table id, not a raw page path.\n")
        fh.write("- assembled_* columns preserve group-level merge and constraint metadata.\n")
        fh.write("- compiled_* columns expose aggregate member-page metadata directly in the flat catalog/fact tables.\n")
        fh.write("- age_* columns describe row-axis ages; column_age_* describes column-axis ages; resolved_age_* gives the usable age bin when the axes are not ambiguous.\n")
        fh.write("- member_pages.csv and source_catalog.jsonl preserve traceability to raw page OCR results.\n")
        fh.write("- This export should be kept separate from the page-level core export.\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export assembled multi-page tables into inference-style fact tables.")
    parser.add_argument("--assembled-jsonl", required=True, help="assembled_tables.jsonl or strong_assembled_tables.jsonl")
    parser.add_argument("--page-csv", action="append", required=True, help="Page metadata CSV(s), e.g. source_catalog.csv (can be passed multiple times)")
    parser.add_argument("--page-jsonl", action="append", default=[], help="Optional page metadata JSONL(s), e.g. source_catalog.jsonl")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--include-unvalidated", action="store_true", help="Keep assembled groups with zero assembled checks")
    parser.add_argument("--include-failed", action="store_true", help="Keep assembled groups that still fail assembled constraints")
    parser.add_argument("--write-rich-facts", action="store_true", help="Also write denormalized cell_facts_rich.csv with repeated table-level metadata")
    args = parser.parse_args()

    assembled_jsonl = Path(args.assembled_jsonl).expanduser()
    page_csv_paths = [Path(path).expanduser() for path in args.page_csv]
    page_jsonl_paths = [Path(path).expanduser() for path in args.page_jsonl]
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    page_csv_index = load_page_csv_index(page_csv_paths)
    page_jsonl_index = load_page_jsonl_index(page_jsonl_paths)
    groups = load_assembled_groups(assembled_jsonl)

    progress_every = 100
    if len(groups) <= 100:
        progress_every = 10
    if len(groups) <= 20:
        progress_every = 1

    print("Assembled export setup")
    print("----------------------")
    print(f"assembled groups loaded: {len(groups)}")
    print(f"page csv indexes:        {len(page_csv_paths)}")
    print(f"page jsonl indexes:      {len(page_jsonl_paths)}")
    print(f"page key index size:     {len(page_csv_index)} csv / {len(page_jsonl_index)} jsonl")
    print()

    source_rows = []
    source_jsonl_rows = []
    member_rows = []
    kept_sources = 0
    kept_facts = 0

    facts_path = output_dir / "cell_facts.csv"
    rich_facts_path = output_dir / "cell_facts_rich.csv"
    with facts_path.open("w", newline="", encoding="utf-8") as fact_fh:
        fact_writer = csv.DictWriter(fact_fh, fieldnames=ASSEMBLED_FACT_COLUMNS)
        fact_writer.writeheader()
        rich_fact_fh = rich_facts_path.open("w", newline="", encoding="utf-8") if args.write_rich_facts else None
        rich_fact_writer = None
        if rich_fact_fh:
            rich_fact_writer = csv.DictWriter(rich_fact_fh, fieldnames=ASSEMBLED_RICH_FACT_COLUMNS)
            rich_fact_writer.writeheader()

        try:
            for idx, group in enumerate(groups, 1):
                if idx % progress_every == 0 or idx == len(groups):
                    print(f"[{idx}/{len(groups)}] processed")
                contexts = page_contexts_for_group(group, page_csv_index, page_jsonl_index)
                source_row, facts, source_jsonl_row = build_assembled_source_row(
                    group,
                    contexts,
                    include_unvalidated=args.include_unvalidated,
                    include_failed=args.include_failed,
                )
                source_rows.append(source_row)
                source_jsonl_rows.append(source_jsonl_row)
                member_rows.extend(aggregate_member_pages_rows(contexts, source_row["table_id"]))

                if truthy(source_row["keep_for_inference"]):
                    kept_sources += 1
                    for fact in facts:
                        fact_writer.writerow({column: fact.get(column, "") for column in ASSEMBLED_FACT_COLUMNS})
                        if rich_fact_writer:
                            rich_fact = {**source_row, **fact}
                            rich_fact_writer.writerow({column: rich_fact.get(column, "") for column in ASSEMBLED_RICH_FACT_COLUMNS})
                        kept_facts += 1
        finally:
            if rich_fact_fh:
                rich_fact_fh.close()

    with (output_dir / "source_catalog.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=ASSEMBLED_SOURCE_COLUMNS)
        writer.writeheader()
        writer.writerows(source_rows)

    with (output_dir / "source_catalog.jsonl").open("w", encoding="utf-8") as fh:
        for row in source_jsonl_rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    with (output_dir / "member_pages.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=MEMBER_PAGE_COLUMNS)
        writer.writeheader()
        writer.writerows(member_rows)

    write_codebook(output_dir / "codebook.txt")

    family_counts = Counter(row["table_family"] for row in source_rows if truthy(row["keep_for_inference"]))
    print("Assembled export")
    print("----------------")
    print(f"Groups scanned: {len(source_rows)}")
    print(f"Groups kept:    {kept_sources}")
    print(f"Facts kept:     {kept_facts:,}")
    print("Top kept families:")
    for family, count in family_counts.most_common(10):
        print(f"  {family}: {count}")
    print(f"\nOutputs written to: {output_dir}")


if __name__ == "__main__":
    main()
