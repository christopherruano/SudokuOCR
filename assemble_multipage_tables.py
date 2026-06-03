"""Assemble likely multi-page OCR tables into grouped table artifacts.

This does not replace the page-level export. It creates a second layer:
  - grouped page manifest
  - assembled table JSONL
  - assembly summary CSV

It works with either:
  - source_catalog.csv from an inference export
  - review/decision CSVs that carry result_json + source metadata
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from oneshot import derive_constraints, verify_all_constraints


CONTINUATION_PAT = re.compile(
    r"(?:\bcont(?:d|inued)?\.?\b|\bconcl(?:d|uded)?\.?\b|\bcontinued\b|\bconcluded\b)",
    re.I,
)
PART_PAT = re.compile(r"\bpart\s+[ivxlcdm0-9]+\b", re.I)
TABLE_INDEX_PAT = re.compile(r"_t(\d{4})_")
TABLE_NO_PAT = re.compile(
    r"\b(?:subsidiary\s+)?table\s+([ivxlcdm0-9]+[a-z]?)\b",
    re.I,
)
TRAILING_CONTINUATION_PAT = re.compile(
    r"(?:[-—–,:;]\s*)?(?:cont(?:d|inued)?\.?|concl(?:d|uded)?\.?|continued|concluded)\s*\.?\s*$",
    re.I,
)
TITLE_LABEL_STOPWORDS = {
    "age",
    "ages",
    "all",
    "and",
    "arranged",
    "assam",
    "birth",
    "census",
    "civil",
    "condition",
    "continued",
    "concluded",
    "distribution",
    "final",
    "males",
    "females",
    "hill",
    "married",
    "number",
    "population",
    "province",
    "provinces",
    "religion",
    "sex",
    "showing",
    "single",
    "state",
    "states",
    "table",
    "total",
    "tribes",
    "widowers",
    "widows",
    "widowed",
}


def normalize_text(value: object) -> str:
    return " ".join(str(value or "").strip().split())


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(subvalue) for key, subvalue in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [json_ready(item) for item in value]
    if isinstance(value, set):
        try:
            ordered = sorted(value)
        except Exception:
            ordered = sorted((str(item) for item in value))
        return [json_ready(item) for item in ordered]
    return value


def normalize_title_key(value: object) -> str:
    text = normalize_text(value).lower()
    text = CONTINUATION_PAT.sub("", text)
    text = PART_PAT.sub("", text)
    text = re.sub(r"[^\w\s-]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip(" .-|")
    return text


def continuation_text(row: dict[str, str]) -> str:
    return " | ".join(
        normalize_text(part)
        for part in [
            first_nonblank(row, ["title"]),
            first_nonblank(row, ["title_from_payload"]),
            first_nonblank(row, ["queue_title"]),
            row_title(row),
            row_dir_name(row),
            first_nonblank(row, ["display_identifier", "queue_title"]),
            first_nonblank(row, ["column_groups", "column_groups_from_payload"]),
        ]
        if normalize_text(part)
    )


def has_continuation_marker(row: dict[str, str]) -> bool:
    title = row_title(row)
    if normalize_text(title):
        return bool(TRAILING_CONTINUATION_PAT.search(normalize_text(title)))
    non_title_text = " | ".join(
        normalize_text(part)
        for part in [
            first_nonblank(row, ["title_from_payload"]),
            first_nonblank(row, ["queue_title"]),
            row_dir_name(row),
            first_nonblank(row, ["display_identifier", "queue_title"]),
            first_nonblank(row, ["column_groups", "column_groups_from_payload"]),
        ]
        if normalize_text(part)
    )
    return bool(CONTINUATION_PAT.search(non_title_text))


def title_tokens(value: object) -> set[str]:
    text = normalize_title_key(value)
    tokens = {
        token
        for token in re.findall(r"[a-z0-9]+", text)
        if len(token) >= 3
        and token
        not in {
            "the",
            "and",
            "for",
            "each",
            "sex",
            "table",
            "subsidiary",
            "part",
            "cont",
            "contd",
            "continued",
            "concld",
            "concluded",
        }
    }
    return tokens


def title_similarity(left: object, right: object) -> float:
    ltokens = title_tokens(left)
    rtokens = title_tokens(right)
    if not ltokens or not rtokens:
        return 0.0
    return len(ltokens & rtokens) / max(len(ltokens | rtokens), 1)


def table_number_key(value: object) -> str:
    text = normalize_text(value)
    match = TABLE_NO_PAT.search(text)
    return normalize_text(match.group(1)).lower() if match else ""


def semantic_label_tokens(value: object) -> set[str]:
    return {token for token in title_tokens(value) if token not in TITLE_LABEL_STOPWORDS and not token.isdigit()}


def conflicting_semantic_labels(left: object, right: object) -> bool:
    left_labels = semantic_label_tokens(left)
    right_labels = semantic_label_tokens(right)
    return bool((left_labels or right_labels) and left_labels != right_labels)


def compatible_continuation_candidate(cont_row: dict[str, str], candidate_row: dict[str, str]) -> tuple[bool, str]:
    if row_source_file(cont_row) != row_source_file(candidate_row):
        return False, "different_source_file"
    if row_family(cont_row) != row_family(candidate_row):
        return False, "different_family"

    cont_index = parse_table_index(row_source_json(cont_row) or row_dir_name(cont_row))
    candidate_index = parse_table_index(row_source_json(candidate_row) or row_dir_name(candidate_row))
    if candidate_index > cont_index:
        return False, "after_continuation_page"
    if cont_index - candidate_index > 4:
        return False, "too_far_back"

    cont_title = row_title(cont_row)
    candidate_title = row_title(candidate_row)
    if conflicting_semantic_labels(cont_title, candidate_title):
        return False, "different_semantic_label"

    cont_title_key = normalize_title_key(cont_title)
    candidate_title_key = normalize_title_key(candidate_title)
    if cont_title_key and candidate_title_key and cont_title_key == candidate_title_key:
        return True, "same_title_key"

    cont_table_no = table_number_key(row_title(cont_row)) or table_number_key(row_dir_name(cont_row))
    candidate_table_no = table_number_key(row_title(candidate_row)) or table_number_key(row_dir_name(candidate_row))
    if cont_table_no and candidate_table_no and cont_table_no == candidate_table_no:
        return True, "same_table_number"

    if row_dir_name(cont_row) and row_dir_name(candidate_row) and normalize_title_key(row_dir_name(cont_row)) == normalize_title_key(row_dir_name(candidate_row)):
        return True, "same_table_dir_key"

    sim = title_similarity(row_title(cont_row), row_title(candidate_row))
    if sim >= 0.28:
        return True, f"title_similarity:{sim:.2f}"

    return False, f"title_similarity:{sim:.2f}"


def continuation_candidate_priority(reason: str) -> tuple[int, float]:
    if reason == "same_title_key":
        return (0, 1.0)
    if reason == "same_table_number":
        return (1, 1.0)
    if reason == "same_table_dir_key":
        return (2, 1.0)
    if reason.startswith("title_similarity:"):
        try:
            return (3, -float(reason.rsplit(":", 1)[1]))
        except Exception:
            return (3, 0.0)
    return (4, 0.0)


def find_continuation_candidate(
    idx: int,
    candidate_indices: list[int],
    rows: list[dict[str, str]],
) -> tuple[int | None, str]:
    row = rows[idx]
    cont_index = parse_table_index(row_source_json(row) or row_dir_name(row))
    choices = []
    for candidate_idx in candidate_indices:
        if candidate_idx == idx:
            continue
        candidate_row = rows[candidate_idx]
        ok, reason = compatible_continuation_candidate(row, candidate_row)
        if not ok:
            continue
        candidate_index = parse_table_index(row_source_json(candidate_row) or row_dir_name(candidate_row))
        distance = max(cont_index - candidate_index, 0)
        choices.append(
            (
                int(has_continuation_marker(candidate_row)),
                distance,
                *continuation_candidate_priority(reason),
                candidate_idx,
                reason,
            )
        )
    if not choices:
        return None, "no_compatible_previous_page"
    best = min(choices)
    return int(best[-2]), str(best[-1])


def truthy(value: object) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def first_nonblank(row: dict[str, Any], keys: list[str]) -> str:
    for key in keys:
        value = row.get(key)
        if str(value or "").strip():
            return str(value).strip()
    return ""


def maybe_rewrite_path(path_text: str, rewrite_from: Path | None, rewrite_to: Path | None) -> Path:
    path = Path(str(path_text or "").strip())
    if rewrite_from and rewrite_to:
        try:
            rel = path.relative_to(rewrite_from)
            return rewrite_to / rel
        except Exception:
            return rewrite_to / path.name
    return path


def candidate_json_paths(row: dict[str, str], rewrite_from: Path | None, rewrite_to: Path | None) -> list[Path]:
    candidates = []
    source_json = row_source_json(row)
    if source_json:
        candidates.append(maybe_rewrite_path(source_json, rewrite_from, rewrite_to))

    source_image = row_source_image(row)
    if source_image:
        image_path = maybe_rewrite_path(source_image, rewrite_from, rewrite_to)
        stem = image_path.stem
        candidates.append(image_path.with_name(f"{stem}_oneshot.json"))
        image_text = str(image_path)
        for old, new in [
            ("/census_toc/digitized_tables/", "/census_toc/ocr_results/digitized_tables/"),
            ("/digitized_tables/", "/ocr_results/digitized_tables/"),
        ]:
            if old in image_text:
                candidates.append(Path(image_text.replace(old, new)).with_name(f"{stem}_oneshot.json"))

    seen = set()
    unique = []
    for path in candidates:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def resolve_payload_path(row: dict[str, str], rewrite_from: Path | None, rewrite_to: Path | None) -> Path:
    candidates = candidate_json_paths(row, rewrite_from, rewrite_to)
    for path in candidates:
        if path.exists():
            return path
    return candidates[0] if candidates else Path(row_source_json(row))


def row_identity(row: dict[str, str]) -> str:
    return first_nonblank(
        row,
        [
            "source_json",
            "result_json",
            "table_id",
            "source_image",
            "image_path",
            "title",
        ],
    )


def load_input_rows(
    input_csvs: list[Path],
    only_keep_inference: bool,
    decision_filter: set[str],
) -> list[dict[str, str]]:
    rows = []
    seen = set()
    for input_csv in input_csvs:
        with input_csv.open(encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                if only_keep_inference and "keep_for_inference" in row and not truthy(row.get("keep_for_inference")):
                    continue
                if decision_filter and "review_decision" in row:
                    decision = normalize_text(row.get("review_decision")).lower()
                    if decision not in decision_filter:
                        continue
                identity = row_identity(row)
                if identity and identity in seen:
                    continue
                if identity:
                    seen.add(identity)
                rows.append(row)
    return rows


def row_source_json(row: dict[str, str]) -> str:
    return first_nonblank(row, ["source_json", "result_json"])


def row_source_image(row: dict[str, str]) -> str:
    return first_nonblank(row, ["source_image", "image_path"])


def row_title(row: dict[str, str]) -> str:
    return first_nonblank(row, ["title", "title_from_payload", "queue_title"])


def row_family(row: dict[str, str]) -> str:
    return first_nonblank(row, ["table_family", "rescue_family", "canonical_topic"])


def row_dir_name(row: dict[str, str]) -> str:
    return first_nonblank(row, ["source_table_dir_name", "table_dir_name"])


def row_source_file(row: dict[str, str]) -> str:
    return first_nonblank(row, ["source_file"])


def parse_table_index(text: str) -> int:
    match = TABLE_INDEX_PAT.search(text or "")
    if match:
        return int(match.group(1))
    return 999999


def build_group_keys(rows: list[dict[str, str]]) -> dict[int, str]:
    dir_counts = Counter()
    title_counts = Counter()

    for row in rows:
        family = row_family(row)
        source_file = row_source_file(row)
        dir_name = row_dir_name(row)
        title_key = normalize_title_key(row_title(row))
        if dir_name:
            dir_counts[(source_file, family, dir_name)] += 1
        if title_key:
            title_counts[(source_file, family, title_key)] += 1

    keys = {}
    for idx, row in enumerate(rows):
        family = row_family(row)
        source_file = row_source_file(row)
        dir_name = row_dir_name(row)
        title_key = normalize_title_key(row_title(row))
        if title_key and title_counts[(source_file, family, title_key)] > 1:
            keys[idx] = f"title::{source_file}::{family}::{title_key}"
        elif dir_name and dir_counts[(source_file, family, dir_name)] > 1:
            keys[idx] = f"dir::{source_file}::{family}::{dir_name}"
        else:
            keys[idx] = f"single::{row_source_json(row)}"
    apply_continuation_grouping(rows, keys)
    return keys


def continuation_group_id(row: dict[str, str]) -> str:
    source_file = row_source_file(row)
    family = row_family(row)
    table_no = table_number_key(row_title(row)) or table_number_key(row_dir_name(row))
    if table_no:
        return f"continued::{source_file}::{family}::table-{table_no}"
    base = normalize_title_key(row_title(row)) or normalize_title_key(row_dir_name(row)) or Path(row_source_json(row)).stem
    return f"continued::{source_file}::{family}::{base}"


def apply_continuation_grouping(rows: list[dict[str, str]], keys: dict[int, str]) -> None:
    by_scope: dict[tuple[str, str], list[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        by_scope[(row_source_file(row), row_family(row))].append(idx)

    for scope_indices in by_scope.values():
        candidate_indices = sorted(
            scope_indices,
            key=lambda idx: (
                parse_table_index(row_source_json(rows[idx]) or row_dir_name(rows[idx])),
                row_source_json(rows[idx]),
            ),
        )
        for idx in candidate_indices:
            row = rows[idx]
            if not has_continuation_marker(row):
                continue

            chosen_idx, chosen_reason = find_continuation_candidate(idx, candidate_indices, rows)

            if chosen_idx is not None:
                previous_key = keys.get(chosen_idx, "")
                current_key = keys.get(idx, "")
                if previous_key.startswith("single::") and current_key.startswith("single::"):
                    group_key = continuation_group_id(row)
                    keys[chosen_idx] = group_key
                    keys[idx] = group_key
                elif previous_key.startswith("single::"):
                    keys[chosen_idx] = current_key if not current_key.startswith("single::") else continuation_group_id(row)
                    keys[idx] = keys[chosen_idx]
                else:
                    keys[idx] = previous_key
                row["__continuation_attached_to"] = row_source_json(rows[chosen_idx]) or row_dir_name(rows[chosen_idx])
                row["__continuation_attach_reason"] = chosen_reason
            else:
                row["__continuation_attach_reason"] = "no_compatible_previous_page"


def build_continuation_review_rows(rows: list[dict[str, str]], keys: dict[int, str]) -> list[dict[str, Any]]:
    review_rows = []
    by_scope: dict[tuple[str, str], list[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        by_scope[(row_source_file(row), row_family(row))].append(idx)

    previous_by_idx: dict[int, tuple[int | None, str]] = {}
    for scope_indices in by_scope.values():
        candidate_indices = sorted(
            scope_indices,
            key=lambda idx: (
                parse_table_index(row_source_json(rows[idx]) or row_dir_name(rows[idx])),
                row_source_json(rows[idx]),
            ),
        )
        for idx in candidate_indices:
            row = rows[idx]
            if has_continuation_marker(row):
                candidate, reason = find_continuation_candidate(idx, candidate_indices, rows)
                previous_by_idx[idx] = (candidate, reason)

    for idx, row in enumerate(rows):
        if not has_continuation_marker(row):
            continue
        candidate_idx, reason = previous_by_idx.get(idx, (None, "not_checked"))
        candidate_row = rows[candidate_idx] if candidate_idx is not None else {}
        review_rows.append(
            {
                "table_id": first_nonblank(row, ["table_id"]),
                "source_json": row_source_json(row),
                "source_image": row_source_image(row),
                "province": first_nonblank(row, ["province"]),
                "year": first_nonblank(row, ["year"]),
                "table_family": row_family(row),
                "title": row_title(row),
                "table_dir_name": row_dir_name(row),
                "table_index": parse_table_index(row_source_json(row) or row_dir_name(row)),
                "assigned_group_id": keys.get(idx, ""),
                "auto_attached_to_source_json": row.get("__continuation_attached_to", ""),
                "auto_attach_reason": row.get("__continuation_attach_reason", reason),
                "candidate_source_json": row_source_json(candidate_row) if candidate_row else "",
                "candidate_title": row_title(candidate_row) if candidate_row else "",
                "candidate_table_dir_name": row_dir_name(candidate_row) if candidate_row else "",
                "candidate_table_index": parse_table_index(row_source_json(candidate_row) or row_dir_name(candidate_row)) if candidate_row else "",
                "candidate_group_id": keys.get(candidate_idx, "") if candidate_idx is not None else "",
                "needs_manual_multipage_review": int(candidate_idx is None),
            }
        )
    return review_rows


def detect_group_signals(group_rows: list[dict[str, str]]) -> list[str]:
    signals = []
    titles_blob = " | ".join(
        filter(
            None,
            [
                row_title(row)
                + " | "
                + first_nonblank(row, ["column_groups", "column_groups_from_payload"])
                for row in group_rows
            ],
        )
    )
    if len({row_dir_name(row) for row in group_rows if row_dir_name(row)}) == 1 and len(group_rows) > 1:
        signals.append("same_table_dir")
    if CONTINUATION_PAT.search(titles_blob):
        signals.append("continued_marker")
    if PART_PAT.search(titles_blob):
        signals.append("part_marker")
    if len({normalize_title_key(row_title(row)) for row in group_rows if row_title(row)}) == 1 and len(group_rows) > 1:
        signals.append("same_title_cluster")
    return signals


def extract_row_key(row: dict[str, Any], fallback_index: int) -> tuple[Any, ...]:
    scalar_items = []
    for key, value in row.items():
        if isinstance(value, dict):
            continue
        text = normalize_text(value)
        if text:
            scalar_items.append((key, text))
    if scalar_items:
        return tuple(scalar_items)
    return (("__row_index__", fallback_index),)


def extract_value_cols(row: dict[str, Any]) -> set[str]:
    cols = set()
    for key, value in row.items():
        if isinstance(value, dict):
            cols.add(key)
    return cols


def merge_nested_dict(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    merged = dict(left or {})
    for key, value in (right or {}).items():
        if key not in merged or merged[key] in ("", None):
            merged[key] = value
    return merged


def merge_rows_columnwise(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[tuple[Any, ...], dict[str, Any]] = {}
    order: list[tuple[Any, ...]] = []

    for idx, row in enumerate(rows):
        row_key = extract_row_key(row, idx)
        if row_key not in merged:
            merged[row_key] = {}
            order.append(row_key)
        target = merged[row_key]
        for key, value in row.items():
            if isinstance(value, dict):
                target[key] = merge_nested_dict(target.get(key, {}), value)
            elif key not in target or not normalize_text(target.get(key)):
                target[key] = value
    return [merged[key] for key in order]


def merge_rows_rowwise(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    output = []
    for idx, row in enumerate(rows):
        row_key = extract_row_key(row, idx)
        if row_key in seen:
            continue
        seen.add(row_key)
        output.append(row)
    return output


def assemble_section(section_instances: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
    all_rows: list[dict[str, Any]] = []
    row_key_lists = []
    value_col_sets = []
    for instance in section_instances:
        rows = instance.get("rows") or []
        all_rows.extend(rows)
        row_keys = {extract_row_key(row, i) for i, row in enumerate(rows)}
        row_key_lists.append(row_keys)
        value_cols = set()
        for row in rows:
            value_cols |= extract_value_cols(row)
        value_col_sets.append(value_cols)

    row_overlap_count = 0
    value_overlap_count = 0
    pair_count = 0
    for i in range(len(row_key_lists)):
        for j in range(i + 1, len(row_key_lists)):
            pair_count += 1
            row_overlap_count += len(row_key_lists[i] & row_key_lists[j])
            value_overlap_count += len(value_col_sets[i] & value_col_sets[j])

    if len(section_instances) == 1:
        strategy = "single_page"
        assembled_rows = section_instances[0].get("rows") or []
    elif row_overlap_count > 0 and value_overlap_count > 0:
        strategy = "column_append"
        assembled_rows = merge_rows_columnwise(all_rows)
    elif row_overlap_count == 0:
        strategy = "row_append"
        assembled_rows = merge_rows_rowwise(all_rows)
    else:
        strategy = "page_bundle"
        assembled_rows = []
        for instance in section_instances:
            page_label = instance.get("page_label") or instance.get("source_json")
            for row in instance.get("rows") or []:
                assembled_rows.append({"__page__": page_label, **row})

    stats = {
        "input_pages": len(section_instances),
        "input_rows": len(all_rows),
        "output_rows": len(assembled_rows),
        "row_overlap_pairs": row_overlap_count,
        "value_overlap_pairs": value_overlap_count,
    }
    return strategy, assembled_rows, stats


def load_payload(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        with path.open(encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


def assemble_group(group_id: str, group_rows: list[dict[str, str]], rewrite_from: Path | None, rewrite_to: Path | None) -> dict[str, Any]:
    def primary_row_for_group(rows: list[dict[str, str]]) -> dict[str, str]:
        attached_to = {
            row.get("__continuation_attached_to", "")
            for row in rows
            if row.get("__continuation_attached_to")
        }
        pointed_to_rows = [
            row
            for row in rows
            if row_source_json(row) in attached_to or row_dir_name(row) in attached_to
        ]
        non_continuation_rows = [row for row in rows if not has_continuation_marker(row)]
        candidates = non_continuation_rows or pointed_to_rows or rows
        return min(
            candidates,
            key=lambda row: (
                parse_table_index(row_source_json(row) or row_dir_name(row)),
                row_source_json(row),
            ),
        )

    primary_row = primary_row_for_group(group_rows) if group_rows else {}
    primary_source_json = row_source_json(primary_row) if primary_row else ""
    pages = []
    missing_payloads = []
    for row in sorted(
        group_rows,
        key=lambda r: (
            0 if row_source_json(r) == primary_source_json else 1,
            parse_table_index(row_source_json(r) or row_dir_name(r)),
            row_source_json(r),
        ),
    ):
        source_json = row_source_json(row)
        local_json = resolve_payload_path(row, rewrite_from, rewrite_to)
        payload = load_payload(local_json)
        if payload is None:
            missing_payloads.append(
                {
                    "source_json": source_json,
                    "source_image": row_source_image(row),
                    "tried": [str(path) for path in candidate_json_paths(row, rewrite_from, rewrite_to)],
                }
            )
            continue
        metadata = payload.get("metadata") or {}
        sections = payload.get("data") or []
        pages.append(
            {
                "source_json": source_json,
                "local_json": str(local_json),
                "source_image": row_source_image(row),
                "title": row_title(row),
                "title_key": normalize_title_key(row_title(row)),
                "table_family": row_family(row),
                "province": row.get("province") or "",
                "year": row.get("year") or "",
                "source_file": row_source_file(row),
                "table_dir_name": row_dir_name(row),
                "page_label": Path(source_json).stem,
                "metadata": metadata,
                "sections": sections,
                "row_axis_type": first_nonblank(row, ["row_axis_type"]),
                "row_label_name": first_nonblank(row, ["row_label_name"]),
                "data_type": first_nonblank(row, ["data_type", "measure_unit"]),
                "constraints": payload.get("constraints") or {},
            }
        )

    section_buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for page in pages:
        sections = page.get("sections") or []
        if not sections:
            section_buckets["__empty__"].append({"page_label": page["page_label"], "rows": []})
        for section in sections:
            name = normalize_text(section.get("name") or "__unnamed__")
            section_buckets[name].append(
                {
                    "page_label": page["page_label"],
                    "source_json": page["source_json"],
                    "rows": section.get("rows") or [],
                }
            )

    assembled_sections = []
    section_strategies = Counter()
    unique_page_sections = True
    for section_name, instances in section_buckets.items():
        strategy, rows, stats = assemble_section(instances)
        section_strategies[strategy] += 1
        if len(instances) > 1:
            unique_page_sections = False
        assembled_sections.append(
            {
                "name": section_name,
                "merge_strategy": strategy,
                "stats": stats,
                "rows": rows,
            }
        )

    overall_strategy = "single_page"
    if pages:
        if len(pages) == 1:
            overall_strategy = "single_page"
        else:
            strategy_keys = set(section_strategies)
            if strategy_keys == {"single_page"}:
                overall_strategy = "section_append" if unique_page_sections else "page_bundle"
            elif len(strategy_keys) == 1:
                overall_strategy = next(iter(strategy_keys))
            else:
                overall_strategy = "mixed"

    column_group_union = []
    seen_groups = set()
    for page in pages:
        for group in (page.get("metadata") or {}).get("column_groups", []) or []:
            norm = normalize_text(group)
            if norm and norm not in seen_groups:
                seen_groups.add(norm)
                column_group_union.append(group)

    row_label_name = ""
    row_axis_type = ""
    data_type = ""
    for page in pages:
        if not row_label_name and page.get("row_label_name"):
            row_label_name = page["row_label_name"]
        if not row_axis_type and page.get("row_axis_type"):
            row_axis_type = page["row_axis_type"]
        if not data_type and page.get("data_type"):
            data_type = page["data_type"]

    assembled_parsed = {
        "sections": [{"name": sec["name"], "rows": sec["rows"]} for sec in assembled_sections if sec["rows"]],
        "metadata": {
            "title": row_title(primary_row) if primary_row else "",
            "column_groups": column_group_union,
            "row_label_name": row_label_name,
            "row_axis_type": row_axis_type,
            "data_type": data_type,
        },
    }

    try:
        assembled_constraints = derive_constraints(assembled_parsed)
        assembled_constraint_report = verify_all_constraints(assembled_parsed, assembled_constraints)
    except Exception as exc:
        assembled_constraints = {}
        assembled_constraint_report = {
            "total_checks": 0,
            "passed": 0,
            "failed": 0,
            "all_passed": False,
            "failures": [{"level": "ASSEMBLY", "detail": f"constraint verification error: {exc}"}],
        }

    input_constraint_checks = 0
    input_constraint_failures = 0
    for page in pages:
        report = page.get("constraints") or {}
        input_constraint_checks += int(report.get("total_checks") or 0)
        input_constraint_failures += int(report.get("failed") or 0)

    return {
        "group_id": group_id,
        "province": first_nonblank(primary_row, ["province"]) if primary_row else "",
        "year": first_nonblank(primary_row, ["year"]) if primary_row else "",
        "table_family": row_family(primary_row) if primary_row else "",
        "source_file": row_source_file(primary_row) if primary_row else "",
        "title": row_title(primary_row) if primary_row else "",
        "title_key": normalize_title_key(row_title(primary_row)) if primary_row else "",
        "table_dir_name": row_dir_name(primary_row) if primary_row else "",
        "signals": detect_group_signals(group_rows),
        "n_pages": len(pages),
        "page_source_jsons": [page["source_json"] for page in pages],
        "page_source_images": [page["source_image"] for page in pages],
        "missing_payloads": missing_payloads,
        "overall_strategy": overall_strategy,
        "section_strategy_counts": dict(section_strategies),
        "input_constraint_checks": input_constraint_checks,
        "input_constraint_failures": input_constraint_failures,
        "assembled_metadata": assembled_parsed.get("metadata") or {},
        "assembled_constraints": assembled_constraints,
        "assembled_constraint_report": assembled_constraint_report,
        "sections": assembled_sections,
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Assemble likely multi-page OCR tables into grouped artifacts.")
    parser.add_argument(
        "--input-csv",
        action="append",
        required=True,
        help="CSV with source_json/result_json and source metadata (can be passed multiple times)",
    )
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--only-keep-inference", action="store_true", help="Filter keep_for_inference=1 when present")
    parser.add_argument("--decision-filter", default="", help="Comma-separated review_decision values to include, e.g. promote,hold")
    parser.add_argument("--rewrite-from", default="", help="Optional path prefix to rewrite from")
    parser.add_argument("--rewrite-to", default="", help="Optional path prefix to rewrite to")
    args = parser.parse_args()

    input_csvs = [Path(path).expanduser() for path in args.input_csv]
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    decision_filter = {part.strip().lower() for part in args.decision_filter.split(",") if part.strip()}
    rewrite_from = Path(args.rewrite_from).expanduser() if args.rewrite_from else None
    rewrite_to = Path(args.rewrite_to).expanduser() if args.rewrite_to else None

    rows = load_input_rows(input_csvs, args.only_keep_inference, decision_filter)
    group_keys = build_group_keys(rows)
    continuation_review_rows = build_continuation_review_rows(rows, group_keys)

    grouped_rows: dict[str, list[dict[str, str]]] = defaultdict(list)
    for idx, row in enumerate(rows):
        grouped_rows[group_keys[idx]].append(row)

    assembled_groups = []
    summary_rows = []
    member_rows = []

    for group_id, group in grouped_rows.items():
        assembled = assemble_group(group_id, group, rewrite_from, rewrite_to)
        assembled_groups.append(assembled)

        summary_rows.append(
            {
                "group_id": group_id,
                "province": assembled["province"],
                "year": assembled["year"],
                "table_family": assembled["table_family"],
                "title": assembled["title"],
                "table_dir_name": assembled["table_dir_name"],
                "source_file": assembled["source_file"],
                "n_pages": assembled["n_pages"],
                "overall_strategy": assembled["overall_strategy"],
                "signals": " | ".join(assembled["signals"]),
                "input_constraint_checks": assembled["input_constraint_checks"],
                "input_constraint_failures": assembled["input_constraint_failures"],
                "assembled_constraint_checks": assembled["assembled_constraint_report"].get("total_checks", 0),
                "assembled_constraint_failures": assembled["assembled_constraint_report"].get("failed", 0),
                "assembled_all_passed": assembled["assembled_constraint_report"].get("all_passed", False),
                "section_strategy_counts": json.dumps(assembled["section_strategy_counts"], ensure_ascii=True, sort_keys=True),
            }
        )

        for source_json in assembled["page_source_jsons"]:
            member_rows.append(
                {
                    "group_id": group_id,
                    "source_json": source_json,
                }
            )

    summary_rows.sort(key=lambda row: (-int(row["n_pages"]), row["province"], row["year"], row["title"]))

    with (output_dir / "assembled_tables.jsonl").open("w", encoding="utf-8") as fh:
        for assembled in assembled_groups:
            fh.write(json.dumps(json_ready(assembled), ensure_ascii=False) + "\n")

    write_csv(
        output_dir / "multipage_groups.csv",
        summary_rows,
        [
            "group_id",
            "province",
            "year",
            "table_family",
            "title",
            "table_dir_name",
            "source_file",
            "n_pages",
            "overall_strategy",
            "signals",
            "input_constraint_checks",
            "input_constraint_failures",
            "assembled_constraint_checks",
            "assembled_constraint_failures",
            "assembled_all_passed",
            "section_strategy_counts",
        ],
    )
    write_csv(output_dir / "multipage_group_members.csv", member_rows, ["group_id", "source_json"])
    write_csv(
        output_dir / "continuation_candidates.csv",
        continuation_review_rows,
        [
            "table_id",
            "source_json",
            "source_image",
            "province",
            "year",
            "table_family",
            "title",
            "table_dir_name",
            "table_index",
            "assigned_group_id",
            "auto_attached_to_source_json",
            "auto_attach_reason",
            "candidate_source_json",
            "candidate_title",
            "candidate_table_dir_name",
            "candidate_table_index",
            "candidate_group_id",
            "needs_manual_multipage_review",
        ],
    )

    strategy_counts = Counter(row["overall_strategy"] for row in summary_rows)
    print("Multi-page assembly")
    print("-------------------")
    print(f"Input rows:   {len(rows)}")
    print(f"Groups:       {len(summary_rows)}")
    print(f"Continuation candidates: {len(continuation_review_rows)}")
    print(
        "Continuation needs review: "
        f"{sum(int(row['needs_manual_multipage_review']) for row in continuation_review_rows)}"
    )
    for key, count in sorted(strategy_counts.items()):
        print(f"{key}: {count}")
    print(f"\nOutputs written to: {output_dir}")


if __name__ == "__main__":
    main()
