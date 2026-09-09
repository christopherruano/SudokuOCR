#!/usr/bin/env python3
"""Materialize site-ready physical assemblies into explicit row/cell records.

This layer performs no OCR/API work. It consumes the safe physical edge frontier
and existing CSV artifacts, then writes a concrete assembly record: page order,
row-label source, column order, long-form cells, and per-edge QC. Logical
siblings remain separate; this only covers physical assemblies.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import zipfile
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterable

import xlsxwriter


ROOT = Path(__file__).resolve().parent
DEFAULT_FRONTIER_DIR = ROOT / "export_final/pdf_recovery_incoming_2026-06-16_first_principles_assembly_frontier_smoke_v45_title_guard"
DEFAULT_CANDIDATE_PROMOTION_DIR = ROOT / "export_final/pdf_recovery_incoming_2026-06-16_first_principles_candidate_promotion_smoke_v45_title_guard"
DEFAULT_CONTROL_DIR = ROOT / "export_final/pdf_recovery_incoming_2026-06-16_first_principles_assembly_control_smoke_v45_title_guard"
DEFAULT_OUT_DIR = ROOT / "export_final/pdf_recovery_incoming_2026-06-16_first_principles_materialized_physical_assembly_v46"

SAFE_FRONTIER_CATEGORIES = {
    "site_ready_physical_edge_confirmed_strong",
    "site_ready_physical_edge_confirmed_review_values",
    "site_ready_physical_edge_v42_promoted",
}
MANUAL_PREVIEW_CATEGORY_PREFIX = "manual_visual_review"
MANUAL_PREVIEW_SITE_USE_POLICY = "review_preview_only_no_site_merge"
MANUAL_PREVIEW_RECOMMENDED_ACTION = (
    "Review preview only; do not physically merge or publish unless manually confirmed and promoted."
)
MEASURE_FIELDS = ["persons", "males", "females"]

COMPONENT_FIELDS = [
    "assembly_rank",
    "assembly_id",
    "source_group_id",
    "frontier_category",
    "site_use_policy",
    "physical_merge_allowed",
    "safe_to_publish_values",
    "value_review_required",
    "v5_source_key",
    "source_pdf_id_guess",
    "page_pair",
    "component_pages",
    "n_pages",
    "pages_consecutive",
    "row_pairing_mode",
    "row_label_source_summary",
    "row_count",
    "block_count",
    "cell_count",
    "left_column_groups",
    "right_column_groups",
    "assembled_column_order",
    "row_alignment_evidence",
    "column_order_evidence",
    "zero_check_status",
    "zero_check_failures",
    "left_title",
    "right_title",
    "left_best_csv",
    "right_best_csv",
    "left_image_path",
    "right_image_path",
    "assembly_status",
    "review_flags",
    "recommended_action",
]

BLOCK_FIELDS = [
    "assembly_id",
    "block_index",
    "left_section",
    "right_section",
    "row_count",
    "left_row_count",
    "right_row_count",
    "row_pairing_mode",
    "row_label_status",
    "left_column_groups",
    "right_column_groups",
    "assembled_column_order",
]

ROW_FIELDS = [
    "assembly_id",
    "block_index",
    "assembled_row_index",
    "row_label",
    "row_label_source",
    "left_row_index",
    "right_row_index",
    "left_row_label",
    "right_row_label",
    "row_pair_status",
    "left_section",
    "right_section",
]

CELL_FIELDS = [
    "assembly_id",
    "block_index",
    "assembled_row_index",
    "row_label",
    "row_label_source",
    "column_position",
    "column_side",
    "source_pdf_page",
    "column_group",
    "measure",
    "assembled_column_name",
    "value",
    "source_section",
    "source_row_label",
]

QC_FIELDS = [
    "assembly_id",
    "check",
    "status",
    "detail",
]

MATRIX_BASE_FIELDS = [
    "assembly_id",
    "block_index",
    "assembled_row_index",
    "row_label",
    "row_label_source",
    "row_pair_status",
]

MATRIX_COLUMN_FIELDS = [
    "matrix_column_key",
    "first_seen_order",
    "column_position",
    "column_side",
    "source_pdf_page",
    "column_group",
    "measure",
    "assembled_column_name",
    "assembly_count",
    "assemblies",
]

MATRIX_FILE_FIELDS = [
    "assembly_rank",
    "assembly_id",
    "matrix_csv",
    "row_count",
    "matrix_column_count",
    "nonblank_values",
    "page_pair",
    "row_pairing_mode",
    "row_label_source_summary",
    "assembly_status",
    "review_flags",
    "column_order",
]

SUMMARY_FIELDS = ["metric", "name", "count"]
AUDIT_FIELDS = ["check", "status", "detail"]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def output_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def stable_id(prefix: str, *parts: object) -> str:
    blob = "||".join(str(part or "") for part in parts)
    return f"{prefix}_{hashlib.sha1(blob.encode('utf-8')).hexdigest()[:16]}"


def int_or_none(value: object) -> int | None:
    if value in {None, ""}:
        return None
    try:
        return int(str(value).strip())
    except ValueError:
        return None


def split_semicolon(value: object) -> list[str]:
    return [part.strip() for part in str(value or "").split(";") if part.strip()]


def semi(values: Iterable[object]) -> str:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    return "; ".join(out)


def normalize_text(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower().replace("—", "-").replace("–", "-").rstrip("."))


def label_match_key(value: object) -> str:
    text = normalize_text(value)
    text = re.sub(r"\.{2,}", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def fuzzy_label_score(left: object, right: object) -> float:
    left_key = label_match_key(left)
    right_key = label_match_key(right)
    if not left_key or not right_key:
        return 0.0
    if left_key == right_key:
        return 1.0
    return SequenceMatcher(None, left_key, right_key).ratio()


def labels_fuzzy_match(left: object, right: object, same_row_index: bool = False) -> bool:
    score = fuzzy_label_score(left, right)
    left_key = label_match_key(left)
    right_key = label_match_key(right)
    if score >= 0.84:
        return True
    if same_row_index and score >= 0.72 and max(len(left_key), len(right_key)) <= 8:
        return True
    return False


def parse_page_pair(value: object) -> tuple[int | None, int | None]:
    match = re.fullmatch(r"\s*(\d+)\s*-\s*(\d+)\s*", str(value or ""))
    if not match:
        return None, None
    return int(match.group(1)), int(match.group(2))


def path_from_value(value: object) -> Path:
    path = Path(str(value or "")).expanduser()
    return path if path.is_absolute() else ROOT / path


def json_metadata(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    meta = data.get("metadata", {})
    return meta if isinstance(meta, dict) else {}


def clean_group(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def row_label_from_record(row: dict[str, str]) -> str:
    for key in ["row_label", "age", "district", "DISTRICT.", "DISTRICT", "District", "AGE", "Age"]:
        text = str(row.get(key, "")).strip()
        if text:
            return text
    label_type = str(row.get("row_label_type", "")).strip()
    if label_type:
        for key, value in row.items():
            if normalize_text(key) == normalize_text(label_type):
                text = str(value or "").strip()
                if text:
                    return text
    return ""


def ordered_groups(records: list[dict[str, str]], metadata: dict[str, object]) -> list[str]:
    meta_groups = metadata.get("column_groups", [])
    groups: list[str] = []
    if isinstance(meta_groups, list):
        groups.extend(clean_group(group) for group in meta_groups if clean_group(group))
    for row in records:
        group = clean_group(row.get("group", ""))
        if group and group not in groups:
            groups.append(group)
    return groups


def split_into_table_rows(records: list[dict[str, str]], group_order: list[str]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    current: list[dict[str, str]] = []
    current_groups: set[str] = set()
    current_label = ""
    current_section = ""

    def flush() -> None:
        nonlocal current, current_groups, current_label, current_section
        if not current:
            return
        rows.append(
            {
                "row_index": len(rows) + 1,
                "row_label": current_label,
                "section": current_section,
                "records": current,
            }
        )
        current = []
        current_groups = set()
        current_label = ""
        current_section = ""

    for record in records:
        group = clean_group(record.get("group", "")) or "__value__"
        label = row_label_from_record(record)
        section = str(record.get("section", "")).strip()
        starts_new_by_group = bool(current and group in current_groups)
        starts_new_by_section = bool(current and section and current_section and normalize_text(section) != normalize_text(current_section))
        starts_new_by_label = bool(
            current
            and label
            and current_label
            and normalize_text(label) != normalize_text(current_label)
            and group_order
            and group == group_order[0]
        )
        if starts_new_by_group or starts_new_by_section or starts_new_by_label:
            flush()
        current.append(record)
        current_groups.add(group)
        if not current_label and label:
            current_label = label
        if not current_section and section:
            current_section = section
    flush()
    return rows


def group_rows_by_section(table_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    blocks: list[dict[str, object]] = []
    by_section: dict[str, list[dict[str, object]]] = defaultdict(list)
    order: list[str] = []
    for row in table_rows:
        section = str(row.get("section", "")).strip()
        key = normalize_text(section) or f"__blank_section_{len(order)+1}"
        if key not in by_section:
            order.append(key)
        by_section[key].append(row)
    for key in order:
        rows = by_section[key]
        blocks.append(
            {
                "section": str(rows[0].get("section", "")),
                "rows": rows,
            }
        )
    return blocks


def load_page_table(csv_path: Path, json_path: Path) -> dict[str, object]:
    records = read_csv(csv_path)
    metadata = json_metadata(json_path)
    groups = ordered_groups(records, metadata)
    table_rows = split_into_table_rows(records, groups)
    return {
        "csv_path": csv_path,
        "json_path": json_path,
        "metadata": metadata,
        "groups": groups,
        "rows": table_rows,
        "blocks": group_rows_by_section(table_rows),
    }


def row_records_by_group(row: dict[str, object]) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for record in row.get("records", []):
        if isinstance(record, dict):
            out[clean_group(record.get("group", ""))] = record
    return out


def values_for_record(record: dict[str, str] | None, measure: str) -> str:
    if not record:
        return ""
    return str(record.get(measure, "")).strip()


def slug(value: object, limit: int = 44) -> str:
    text = re.sub(r"[^a-z0-9]+", "_", normalize_text(value)).strip("_")
    return (text[:limit].strip("_") or "blank")


def matrix_column_key(cell: dict[str, object]) -> str:
    position = int_or_none(cell.get("column_position")) or 0
    side = slug(cell.get("column_side", ""), 10)
    group = slug(cell.get("column_group", ""), 50)
    measure = slug(cell.get("measure", ""), 12)
    return f"c{position:03d}_{side}_{group}_{measure}"


def build_matrix_outputs(
    row_rows: list[dict[str, object]],
    cell_rows: list[dict[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[str]]:
    row_lookup: dict[tuple[str, str, str], dict[str, object]] = {}
    for row in row_rows:
        key = (str(row.get("assembly_id", "")), str(row.get("block_index", "")), str(row.get("assembled_row_index", "")))
        row_lookup[key] = row

    matrix_rows_by_key: dict[tuple[str, str, str], dict[str, object]] = {}
    column_meta: dict[str, dict[str, object]] = {}
    column_first_order: dict[str, int] = {}
    column_assemblies: dict[str, set[str]] = defaultdict(set)

    for cell in cell_rows:
        row_key = (str(cell.get("assembly_id", "")), str(cell.get("block_index", "")), str(cell.get("assembled_row_index", "")))
        row_meta = row_lookup.get(row_key, {})
        matrix_row = matrix_rows_by_key.setdefault(
            row_key,
            {
                "assembly_id": row_key[0],
                "block_index": row_key[1],
                "assembled_row_index": row_key[2],
                "row_label": row_meta.get("row_label", cell.get("row_label", "")),
                "row_label_source": row_meta.get("row_label_source", cell.get("row_label_source", "")),
                "row_pair_status": row_meta.get("row_pair_status", ""),
            },
        )
        col_key = matrix_column_key(cell)
        matrix_row[col_key] = cell.get("value", "")
        if col_key not in column_meta:
            column_meta[col_key] = {
                "matrix_column_key": col_key,
                "first_seen_order": len(column_meta) + 1,
                "column_position": cell.get("column_position", ""),
                "column_side": cell.get("column_side", ""),
                "source_pdf_page": cell.get("source_pdf_page", ""),
                "column_group": cell.get("column_group", ""),
                "measure": cell.get("measure", ""),
                "assembled_column_name": cell.get("assembled_column_name", ""),
            }
        column_first_order[col_key] = min(column_first_order.get(col_key, 10**9), int_or_none(cell.get("column_position")) or 10**9)
        column_assemblies[col_key].add(str(cell.get("assembly_id", "")))

    def matrix_row_key(item: tuple[tuple[str, str, str], dict[str, object]]) -> tuple[str, int, int]:
        key, _row = item
        return (key[0], int_or_none(key[1]) or 999999, int_or_none(key[2]) or 999999)

    matrix_column_fields = sorted(
        column_meta,
        key=lambda key: (
            int_or_none(column_meta[key].get("column_position")) or 999999,
            str(column_meta[key].get("column_side", "")),
            str(column_meta[key].get("column_group", "")),
            str(column_meta[key].get("measure", "")),
            key,
        ),
    )
    matrix_rows = [row for _key, row in sorted(matrix_rows_by_key.items(), key=matrix_row_key)]
    column_rows: list[dict[str, object]] = []
    for col_key in matrix_column_fields:
        meta = dict(column_meta[col_key])
        assemblies = sorted(a for a in column_assemblies[col_key] if a)
        meta["assembly_count"] = len(assemblies)
        meta["assemblies"] = semi(assemblies)
        column_rows.append(meta)
    return matrix_rows, column_rows, MATRIX_BASE_FIELDS + matrix_column_fields


def safe_filename(value: object) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "").strip())
    return text.strip("_")[:120] or "assembly"


def write_compact_assembly_matrices(
    out_dir: Path,
    component_rows: list[dict[str, object]],
    row_rows: list[dict[str, object]],
    cell_rows: list[dict[str, object]],
    matrix_dir_name: str = "matrix_by_assembly",
) -> list[dict[str, object]]:
    matrix_dir = out_dir / matrix_dir_name
    matrix_dir.mkdir(parents=True, exist_ok=True)
    rows_by_assembly: dict[str, list[dict[str, object]]] = defaultdict(list)
    cells_by_assembly: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in row_rows:
        rows_by_assembly[str(row.get("assembly_id", ""))].append(row)
    for cell in cell_rows:
        cells_by_assembly[str(cell.get("assembly_id", ""))].append(cell)

    manifest_rows: list[dict[str, object]] = []
    for component in component_rows:
        assembly_id = str(component.get("assembly_id", ""))
        assembly_rows = sorted(
            rows_by_assembly.get(assembly_id, []),
            key=lambda row: (int_or_none(row.get("block_index")) or 999999, int_or_none(row.get("assembled_row_index")) or 999999),
        )
        assembly_cells = cells_by_assembly.get(assembly_id, [])
        column_meta: dict[str, dict[str, object]] = {}
        for cell in assembly_cells:
            key = matrix_column_key(cell)
            column_meta.setdefault(
                key,
                {
                    "position": int_or_none(cell.get("column_position")) or 999999,
                    "side": str(cell.get("column_side", "")),
                    "group": str(cell.get("column_group", "")),
                    "measure": str(cell.get("measure", "")),
                    "name": str(cell.get("assembled_column_name", "")),
                },
            )
        compact_columns = sorted(
            column_meta,
            key=lambda key: (
                column_meta[key]["position"],
                column_meta[key]["side"],
                column_meta[key]["group"],
                column_meta[key]["measure"],
                key,
            ),
        )
        compact_fields = MATRIX_BASE_FIELDS + compact_columns
        row_key_to_matrix: dict[tuple[str, str], dict[str, object]] = {}
        for row in assembly_rows:
            row_key = (str(row.get("block_index", "")), str(row.get("assembled_row_index", "")))
            row_key_to_matrix[row_key] = {
                "assembly_id": assembly_id,
                "block_index": row.get("block_index", ""),
                "assembled_row_index": row.get("assembled_row_index", ""),
                "row_label": row.get("row_label", ""),
                "row_label_source": row.get("row_label_source", ""),
                "row_pair_status": row.get("row_pair_status", ""),
            }
        for cell in assembly_cells:
            row_key = (str(cell.get("block_index", "")), str(cell.get("assembled_row_index", "")))
            matrix_row = row_key_to_matrix.get(row_key)
            if matrix_row is not None:
                matrix_row[matrix_column_key(cell)] = cell.get("value", "")
        compact_rows = [
            row_key_to_matrix[key]
            for key in sorted(row_key_to_matrix, key=lambda key: (int_or_none(key[0]) or 999999, int_or_none(key[1]) or 999999))
        ]
        matrix_path = matrix_dir / f"{int_or_none(component.get('assembly_rank')) or 0:03d}_{safe_filename(assembly_id)}.csv"
        write_csv(matrix_path, compact_rows, compact_fields)
        nonblank_values = sum(1 for row in compact_rows for key in compact_columns if str(row.get(key, "")).strip())
        manifest_rows.append(
            {
                "assembly_rank": component.get("assembly_rank", ""),
                "assembly_id": assembly_id,
                "matrix_csv": output_path(matrix_path),
                "row_count": len(compact_rows),
                "matrix_column_count": len(compact_columns),
                "nonblank_values": nonblank_values,
                "page_pair": component.get("page_pair", ""),
                "row_pairing_mode": component.get("row_pairing_mode", ""),
                "row_label_source_summary": component.get("row_label_source_summary", ""),
                "assembly_status": component.get("assembly_status", ""),
                "review_flags": component.get("review_flags", ""),
                "column_order": semi(column_meta[key]["name"] for key in compact_columns),
            }
        )
    return manifest_rows


def pair_rows(
    left_rows: list[dict[str, object]],
    right_rows: list[dict[str, object]],
) -> tuple[str, list[tuple[dict[str, object] | None, dict[str, object] | None]]]:
    left_labels = [normalize_text(row.get("row_label", "")) for row in left_rows if normalize_text(row.get("row_label", ""))]
    right_labels = [normalize_text(row.get("row_label", "")) for row in right_rows if normalize_text(row.get("row_label", ""))]
    right_by_label = {normalize_text(row.get("row_label", "")): row for row in right_rows if normalize_text(row.get("row_label", ""))}
    if left_labels and right_labels:
        overlap = set(left_labels) & set(right_labels)
        if len(overlap) >= max(1, min(len(set(left_labels)), len(set(right_labels))) // 2):
            paired: list[tuple[dict[str, object] | None, dict[str, object] | None]] = []
            used_right: set[int] = set()
            left_needing_fuzzy: list[dict[str, object]] = []
            for left in left_rows:
                key = normalize_text(left.get("row_label", ""))
                right = right_by_label.get(key)
                if right is not None:
                    used_right.add(int(right.get("row_index") or 0))
                    paired.append((left, right))
                else:
                    left_needing_fuzzy.append(left)
            unused_right = [right for right in right_rows if int(right.get("row_index") or 0) not in used_right]
            for left in left_needing_fuzzy:
                best_idx = -1
                best_score = 0.0
                best_same_row = False
                for idx, right in enumerate(unused_right):
                    score = fuzzy_label_score(left.get("row_label", ""), right.get("row_label", ""))
                    same_row = int_or_none(left.get("row_index")) == int_or_none(right.get("row_index"))
                    if score > best_score:
                        best_idx = idx
                        best_score = score
                        best_same_row = same_row
                if best_idx >= 0 and labels_fuzzy_match(left.get("row_label", ""), unused_right[best_idx].get("row_label", ""), same_row_index=best_same_row):
                    right = unused_right.pop(best_idx)
                    used_right.add(int(right.get("row_index") or 0))
                    paired.append((left, right))
                else:
                    paired.append((left, None))
            for right in right_rows:
                if int(right.get("row_index") or 0) not in used_right:
                    paired.append((None, right))
            return "label_overlap", paired
    max_len = max(len(left_rows), len(right_rows))
    return "position", [
        (left_rows[idx] if idx < len(left_rows) else None, right_rows[idx] if idx < len(right_rows) else None)
        for idx in range(max_len)
    ]


def label_source(left_row: dict[str, object] | None, right_row: dict[str, object] | None) -> tuple[str, str, str, str]:
    left_label = str(left_row.get("row_label", "")).strip() if left_row else ""
    right_label = str(right_row.get("row_label", "")).strip() if right_row else ""
    same_row = bool(left_row and right_row and int_or_none(left_row.get("row_index")) == int_or_none(right_row.get("row_index")))
    if left_label and right_label and normalize_text(left_label) == normalize_text(right_label):
        return left_label, "both_pages", left_label, right_label
    if left_label and right_label and labels_fuzzy_match(left_label, right_label, same_row_index=same_row):
        return left_label, "both_pages_fuzzy_label_match", left_label, right_label
    if left_label and right_label:
        return left_label, "left_page_conflicts_with_right", left_label, right_label
    if left_label:
        return left_label, "left_page_inherited_to_right", left_label, right_label
    if right_label:
        return right_label, "right_page_inherited_to_left", left_label, right_label
    return "", "unresolved", left_label, right_label


def manifest_by_edge(candidate_dir: Path) -> dict[tuple[str, str], dict[str, str]]:
    rows = read_csv(candidate_dir / "canonical_grouping_manifest_v42.csv")
    out: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        if row.get("physical_merge_allowed") != "yes":
            continue
        out[(row.get("v5_source_key", ""), row.get("edge_pairs", "") or row.get("page_range", ""))] = row
    return out


def materialize_edge(
    rank: int,
    edge: dict[str, str],
    manifest_row: dict[str, str] | None,
    preview_only: bool = False,
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    left_page, right_page = parse_page_pair(edge.get("page_pair", ""))
    left_csv = path_from_value(edge.get("left_best_csv", ""))
    right_csv = path_from_value(edge.get("right_best_csv", ""))
    left_json = path_from_value(edge.get("left_best_json", ""))
    right_json = path_from_value(edge.get("right_best_json", ""))
    if preview_only:
        assembly_id = stable_id("v46preview", edge.get("edge_key", ""))
    else:
        assembly_id = (manifest_row or {}).get("group_id") or stable_id("v46phys", edge.get("edge_key", ""))

    qc_rows: list[dict[str, object]] = []
    block_rows: list[dict[str, object]] = []
    assembled_rows: list[dict[str, object]] = []
    cell_rows: list[dict[str, object]] = []

    def qc(check: str, status: str, detail: str) -> None:
        qc_rows.append({"assembly_id": assembly_id, "check": check, "status": status, "detail": detail})

    pages_consecutive = left_page is not None and right_page is not None and right_page == left_page + 1
    qc("pages_are_consecutive", "pass" if pages_consecutive else "fail", edge.get("page_pair", ""))
    if edge.get("frontier_category") in SAFE_FRONTIER_CATEGORIES:
        frontier_status = "pass"
    elif preview_only and edge.get("frontier_category", "").startswith(MANUAL_PREVIEW_CATEGORY_PREFIX):
        frontier_status = "review"
    else:
        frontier_status = "fail"
    qc("frontier_edge_is_site_ready", frontier_status, edge.get("frontier_category", ""))
    qc("csv_artifacts_exist", "pass" if left_csv.exists() and right_csv.exists() else "fail", f"left={left_csv.exists()} right={right_csv.exists()}")

    left_table = load_page_table(left_csv, left_json) if left_csv.exists() else {"groups": [], "blocks": []}
    right_table = load_page_table(right_csv, right_json) if right_csv.exists() else {"groups": [], "blocks": []}
    left_groups = list(left_table.get("groups", []))
    right_groups = list(right_table.get("groups", []))
    left_blocks = list(left_table.get("blocks", []))
    right_blocks = list(right_table.get("blocks", []))
    block_count = max(len(left_blocks), len(right_blocks))
    qc("column_groups_present", "pass" if left_groups and right_groups else "fail", f"left={len(left_groups)} right={len(right_groups)}")
    qc("block_count_compatible", "pass" if len(left_blocks) == len(right_blocks) or min(len(left_blocks), len(right_blocks)) == 1 else "review", f"left={len(left_blocks)} right={len(right_blocks)}")

    row_label_sources: Counter[str] = Counter()
    row_pair_modes: Counter[str] = Counter()
    row_pair_statuses: Counter[str] = Counter()
    row_count_total = 0
    column_position_template: list[tuple[str, int | None, str, str]] = []
    if left_page is not None:
        for group in left_groups:
            for measure in MEASURE_FIELDS:
                column_position_template.append(("left", left_page, group, measure))
    if right_page is not None:
        for group in right_groups:
            for measure in MEASURE_FIELDS:
                column_position_template.append(("right", right_page, group, measure))

    for block_idx in range(block_count):
        left_block = left_blocks[block_idx] if block_idx < len(left_blocks) else {"section": "", "rows": []}
        right_block = right_blocks[block_idx] if block_idx < len(right_blocks) else {"section": "", "rows": []}
        left_rows = list(left_block.get("rows", [])) if isinstance(left_block, dict) else []
        right_rows = list(right_block.get("rows", [])) if isinstance(right_block, dict) else []
        mode, pairs = pair_rows(left_rows, right_rows)
        row_pair_modes[mode] += 1
        label_statuses: Counter[str] = Counter()
        block_start_row = row_count_total
        for left_row, right_row in pairs:
            row_count_total += 1
            row_label, source, left_label, right_label = label_source(left_row, right_row)
            row_label_sources[source] += 1
            if source == "unresolved" or "conflicts" in source:
                label_statuses[source] += 1
            left_records = row_records_by_group(left_row) if left_row else {}
            right_records = row_records_by_group(right_row) if right_row else {}
            row_status = "ok"
            if not row_label:
                row_status = "row_label_unresolved"
            elif "conflicts" in source:
                row_status = "row_label_conflict"
            elif left_row is None or right_row is None:
                row_status = "one_side_missing"
            row_pair_statuses[row_status] += 1
            assembled_rows.append(
                {
                    "assembly_id": assembly_id,
                    "block_index": block_idx + 1,
                    "assembled_row_index": row_count_total,
                    "row_label": row_label,
                    "row_label_source": source,
                    "left_row_index": left_row.get("row_index", "") if left_row else "",
                    "right_row_index": right_row.get("row_index", "") if right_row else "",
                    "left_row_label": left_label,
                    "right_row_label": right_label,
                    "row_pair_status": row_status,
                    "left_section": left_block.get("section", "") if isinstance(left_block, dict) else "",
                    "right_section": right_block.get("section", "") if isinstance(right_block, dict) else "",
                }
            )
            col_pos = 0
            for side, source_page, group, measure in column_position_template:
                col_pos += 1
                source_record = (left_records if side == "left" else right_records).get(group)
                section = ""
                source_row_label = ""
                if source_record:
                    section = source_record.get("section", "")
                    source_row_label = row_label_from_record(source_record)
                cell_rows.append(
                    {
                        "assembly_id": assembly_id,
                        "block_index": block_idx + 1,
                        "assembled_row_index": row_count_total,
                        "row_label": row_label,
                        "row_label_source": source,
                        "column_position": col_pos,
                        "column_side": side,
                        "source_pdf_page": source_page or "",
                        "column_group": group,
                        "measure": measure,
                        "assembled_column_name": f"{side}.{group} / {measure}",
                        "value": values_for_record(source_record, measure),
                        "source_section": section,
                        "source_row_label": source_row_label,
                    }
                )
        block_row_count = row_count_total - block_start_row
        row_label_status = "pass" if not label_statuses else "review"
        block_rows.append(
            {
                "assembly_id": assembly_id,
                "block_index": block_idx + 1,
                "left_section": left_block.get("section", "") if isinstance(left_block, dict) else "",
                "right_section": right_block.get("section", "") if isinstance(right_block, dict) else "",
                "row_count": block_row_count,
                "left_row_count": len(left_rows),
                "right_row_count": len(right_rows),
                "row_pairing_mode": mode,
                "row_label_status": row_label_status,
                "left_column_groups": semi(left_groups),
                "right_column_groups": semi(right_groups),
                "assembled_column_order": semi(f"left.{group} / {measure}" for group in left_groups for measure in MEASURE_FIELDS)
                + ("; " if left_groups and right_groups else "")
                + semi(f"right.{group} / {measure}" for group in right_groups for measure in MEASURE_FIELDS),
            }
        )

    qc("row_labels_resolved", "pass" if row_pair_statuses.get("row_label_unresolved", 0) == 0 else "review", f"unresolved={row_pair_statuses.get('row_label_unresolved', 0)}")
    qc("row_label_conflicts_absent", "pass" if row_pair_statuses.get("row_label_conflict", 0) == 0 else "review", f"conflicts={row_pair_statuses.get('row_label_conflict', 0)}")
    qc("row_counts_compatible", "pass" if row_pair_statuses.get("one_side_missing", 0) == 0 else "review", f"one_side_missing_rows={row_pair_statuses.get('one_side_missing', 0)}")
    expected_cells = row_count_total * len(column_position_template)
    qc("cell_count_matches_rows_times_columns", "pass" if expected_cells == len(cell_rows) else "fail", f"expected={expected_cells} actual={len(cell_rows)}")

    review_flags = []
    if preview_only:
        status = "manual_preview_structure_clean"
        review_flags.extend(["manual_preview_not_promoted", "manual_review_required"])
        if any(row["status"] == "fail" for row in qc_rows):
            status = "manual_preview_materialization_failed"
            review_flags.append("materialization_failure_review")
        elif any(row["status"] == "review" for row in qc_rows):
            status = "manual_preview_structure_review"
            review_flags.append("row_or_block_review")
    else:
        status = "materialized_physical_structure"
        if any(row["status"] == "fail" for row in qc_rows):
            status = "materialization_failed"
        elif any(row["status"] == "review" for row in qc_rows):
            status = "materialized_structure_review"
            review_flags.append("row_or_block_review")
    if edge.get("zero_check_status") != "zero_checks_pass":
        review_flags.append("value_review_required")
    if edge.get("frontier_category") != "site_ready_physical_edge_confirmed_strong":
        review_flags.append("non_strong_values_review")

    if preview_only:
        source_group_id = edge.get("edge_key", "")
        site_use_policy = MANUAL_PREVIEW_SITE_USE_POLICY
        physical_merge_allowed = "no"
        safe_to_publish_values = "no"
        value_review_required = "yes"
        component_pages = edge.get("page_pair", "").replace("-", "; ")
        recommended_action = MANUAL_PREVIEW_RECOMMENDED_ACTION
    else:
        source_group_id = (manifest_row or {}).get("source_record_id", "")
        site_use_policy = (manifest_row or {}).get("site_use_policy", "use_now_physical_component_structure")
        physical_merge_allowed = (manifest_row or {}).get("physical_merge_allowed", "yes")
        safe_to_publish_values = (manifest_row or {}).get("safe_to_publish_values", "no")
        value_review_required = (manifest_row or {}).get("review_required", edge.get("manual_review_required", ""))
        component_pages = (manifest_row or {}).get("component_pages", edge.get("page_pair", "").replace("-", "; "))
        recommended_action = "Use as materialized physical structure; publish values only when value policy permits."

    component = {
        "assembly_rank": rank,
        "assembly_id": assembly_id,
        "source_group_id": source_group_id,
        "frontier_category": edge.get("frontier_category", ""),
        "site_use_policy": site_use_policy,
        "physical_merge_allowed": physical_merge_allowed,
        "safe_to_publish_values": safe_to_publish_values,
        "value_review_required": value_review_required,
        "v5_source_key": edge.get("v5_source_key", ""),
        "source_pdf_id_guess": edge.get("source_pdf_id_guess", ""),
        "page_pair": edge.get("page_pair", ""),
        "component_pages": component_pages,
        "n_pages": (manifest_row or {}).get("n_pages", "2"),
        "pages_consecutive": edge.get("pages_consecutive", ""),
        "row_pairing_mode": semi(f"{key}={value}" for key, value in row_pair_modes.items()),
        "row_label_source_summary": semi(f"{key}={value}" for key, value in row_label_sources.items()),
        "row_count": row_count_total,
        "block_count": block_count,
        "cell_count": len(cell_rows),
        "left_column_groups": semi(left_groups),
        "right_column_groups": semi(right_groups),
        "assembled_column_order": semi(f"left.{group} / {measure}" for group in left_groups for measure in MEASURE_FIELDS)
        + ("; " if left_groups and right_groups else "")
        + semi(f"right.{group} / {measure}" for group in right_groups for measure in MEASURE_FIELDS),
        "row_alignment_evidence": edge.get("row_alignment_evidence", ""),
        "column_order_evidence": edge.get("column_order_evidence", ""),
        "zero_check_status": edge.get("zero_check_status", ""),
        "zero_check_failures": edge.get("zero_check_failures", ""),
        "left_title": edge.get("left_title", ""),
        "right_title": edge.get("right_title", ""),
        "left_best_csv": edge.get("left_best_csv", ""),
        "right_best_csv": edge.get("right_best_csv", ""),
        "left_image_path": edge.get("prev_image_path", ""),
        "right_image_path": edge.get("image_path", ""),
        "assembly_status": status,
        "review_flags": semi(review_flags),
        "recommended_action": recommended_action,
    }
    return component, block_rows, assembled_rows, cell_rows, qc_rows


def write_workbook(
    path: Path,
    summary_rows: list[dict[str, object]],
    audit_rows: list[dict[str, object]],
    component_rows: list[dict[str, object]],
    block_rows: list[dict[str, object]],
    row_rows: list[dict[str, object]],
    matrix_rows: list[dict[str, object]],
    matrix_column_rows: list[dict[str, object]],
    matrix_file_rows: list[dict[str, object]],
    matrix_fields: list[str],
    cell_rows: list[dict[str, object]],
    qc_rows: list[dict[str, object]],
    preview_sheets: list[tuple[str, list[dict[str, object]], list[str]]] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    workbook = xlsxwriter.Workbook(path, {"constant_memory": True})
    header = workbook.add_format({"bold": True, "bg_color": "#F0E6D2", "border": 1})
    wrap = workbook.add_format({"text_wrap": True, "valign": "top"})
    sheets = [
        ("Summary", summary_rows, SUMMARY_FIELDS),
        ("Acceptance Audit", audit_rows, AUDIT_FIELDS),
        ("Assemblies", component_rows, COMPONENT_FIELDS),
        ("Blocks", block_rows, BLOCK_FIELDS),
        ("Rows", row_rows, ROW_FIELDS),
        ("Matrix Files", matrix_file_rows, MATRIX_FILE_FIELDS),
        ("Matrix Rows", matrix_rows, matrix_fields),
        ("Matrix Columns", matrix_column_rows, MATRIX_COLUMN_FIELDS),
        ("Cells", cell_rows, CELL_FIELDS),
        ("QC", qc_rows, QC_FIELDS),
    ]
    if preview_sheets:
        sheets.extend(preview_sheets)
    for name, rows, fields in sheets:
        ws = workbook.add_worksheet(name[:31])
        for col, field in enumerate(fields):
            ws.write(0, col, field, header)
        for row_idx, row in enumerate(rows, start=1):
            for col, field in enumerate(fields):
                fmt = wrap if field in {"assembled_column_order", "recommended_action", "detail", "left_title", "right_title", "left_best_csv", "right_best_csv", "assemblies"} else None
                ws.write(row_idx, col, row.get(field, ""), fmt)
        ws.freeze_panes(1, 0)
        ws.autofilter(0, 0, max(len(rows), 1), max(len(fields) - 1, 0))
        for col, field in enumerate(fields):
            width = 14
            if field in {"assembly_id", "source_group_id"}:
                width = 36
            elif field in {"assembled_column_order", "left_column_groups", "right_column_groups"}:
                width = 58
            elif field.startswith("c") and "_" in field:
                width = 18
            elif field.endswith("_csv") or field.endswith("_path"):
                width = 54
            elif field in {"left_title", "right_title", "detail", "recommended_action"}:
                width = 48
            ws.set_column(col, col, width)
    workbook.close()


def build_audit(
    safe_edges: list[dict[str, str]],
    component_rows: list[dict[str, object]],
    row_rows: list[dict[str, object]],
    cell_rows: list[dict[str, object]],
    matrix_rows: list[dict[str, object]],
    matrix_column_rows: list[dict[str, object]],
    matrix_file_rows: list[dict[str, object]],
    qc_rows: list[dict[str, object]],
    workbook_path: Path,
    manual_preview_edges: list[dict[str, str]] | None = None,
    manual_preview_component_rows: list[dict[str, object]] | None = None,
    manual_preview_matrix_file_rows: list[dict[str, object]] | None = None,
    manual_preview_qc_rows: list[dict[str, object]] | None = None,
) -> list[dict[str, object]]:
    audits: list[dict[str, object]] = []
    manual_preview_edges = manual_preview_edges or []
    manual_preview_component_rows = manual_preview_component_rows or []
    manual_preview_matrix_file_rows = manual_preview_matrix_file_rows or []
    manual_preview_qc_rows = manual_preview_qc_rows or []
    safe_ids = {row.get("edge_key", "") for row in safe_edges}
    materialized_ids = {str(row.get("assembly_id", "")) for row in component_rows}
    audits.append({"check": "all_safe_physical_edges_materialized", "status": "pass" if len(component_rows) == len(safe_edges) else "fail", "detail": f"safe_edges={len(safe_edges)} materialized={len(component_rows)}"})
    bad_consecutive = [row for row in component_rows if row.get("pages_consecutive") != "yes"]
    audits.append({"check": "materialized_edges_are_consecutive", "status": "pass" if not bad_consecutive else "fail", "detail": f"bad={len(bad_consecutive)}"})
    failed_qc = [row for row in qc_rows if row.get("status") == "fail"]
    audits.append({"check": "materialized_qc_has_no_failures", "status": "pass" if not failed_qc else "fail", "detail": f"failures={len(failed_qc)}"})
    no_cells = [row for row in component_rows if int_or_none(row.get("cell_count")) in {None, 0}]
    audits.append({"check": "materialized_assemblies_have_cells", "status": "pass" if not no_cells else "fail", "detail": f"bad={len(no_cells)}"})
    audits.append({"check": "matrix_rows_match_assembled_rows", "status": "pass" if len(matrix_rows) == len(row_rows) else "fail", "detail": f"matrix_rows={len(matrix_rows)} assembled_rows={len(row_rows)}"})
    audits.append({"check": "matrix_columns_present", "status": "pass" if matrix_column_rows and cell_rows else "fail", "detail": f"matrix_columns={len(matrix_column_rows)} cells={len(cell_rows)}"})
    audits.append({"check": "compact_matrix_file_per_assembly", "status": "pass" if len(matrix_file_rows) == len(component_rows) else "fail", "detail": f"matrix_files={len(matrix_file_rows)} assemblies={len(component_rows)}"})
    matrix_value_count = sum(1 for row in matrix_rows for key, value in row.items() if key not in MATRIX_BASE_FIELDS and str(value or "").strip())
    long_value_count = sum(1 for row in cell_rows if str(row.get("value", "")).strip())
    audits.append({"check": "matrix_nonblank_values_match_long_cells", "status": "pass" if matrix_value_count == long_value_count else "fail", "detail": f"matrix_values={matrix_value_count} long_values={long_value_count}"})
    row_labels_unresolved = [row for row in qc_rows if row.get("check") == "row_labels_resolved" and row.get("status") != "pass"]
    audits.append({"check": "row_label_inheritance_is_recorded", "status": "pass" if not row_labels_unresolved else "warn", "detail": f"review={len(row_labels_unresolved)}"})
    if manual_preview_edges or manual_preview_component_rows:
        audits.append({
            "check": "manual_preview_candidates_materialized",
            "status": "pass" if len(manual_preview_component_rows) == len(manual_preview_edges) else "fail",
            "detail": f"manual_candidates={len(manual_preview_edges)} preview_materialized={len(manual_preview_component_rows)}",
        })
        preview_merge_violations = [
            row.get("assembly_id", "")
            for row in manual_preview_component_rows
            if row.get("physical_merge_allowed") != "no" or row.get("safe_to_publish_values") != "no"
        ]
        audits.append({
            "check": "manual_preview_records_are_not_physical_merges",
            "status": "pass" if not preview_merge_violations else "fail",
            "detail": f"bad={len(preview_merge_violations)} examples={'; '.join(preview_merge_violations[:10])}",
        })
        preview_nonconsecutive = [row.get("assembly_id", "") for row in manual_preview_component_rows if row.get("pages_consecutive") != "yes"]
        audits.append({
            "check": "manual_preview_edges_are_consecutive",
            "status": "pass" if not preview_nonconsecutive else "fail",
            "detail": f"bad={len(preview_nonconsecutive)} examples={'; '.join(preview_nonconsecutive[:10])}",
        })
        audits.append({
            "check": "manual_preview_compact_matrix_file_per_candidate",
            "status": "pass" if len(manual_preview_matrix_file_rows) == len(manual_preview_component_rows) else "fail",
            "detail": f"matrix_files={len(manual_preview_matrix_file_rows)} preview_candidates={len(manual_preview_component_rows)}",
        })
        preview_failed_qc = [row for row in manual_preview_qc_rows if row.get("status") == "fail"]
        audits.append({
            "check": "manual_preview_qc_failures_are_review_only",
            "status": "warn" if preview_failed_qc else "pass",
            "detail": f"failures={len(preview_failed_qc)}",
        })
    try:
        with zipfile.ZipFile(workbook_path) as zf:
            bad_file = zf.testzip()
        status = "pass" if bad_file is None else "fail"
        detail = "ok" if bad_file is None else str(bad_file)
    except Exception as exc:  # pragma: no cover
        status = "fail"
        detail = str(exc)
    audits.append({"check": "workbook_zip_integrity", "status": status, "detail": detail})
    return audits


def build_summary(
    component_rows: list[dict[str, object]],
    block_rows: list[dict[str, object]],
    row_rows: list[dict[str, object]],
    matrix_rows: list[dict[str, object]],
    matrix_column_rows: list[dict[str, object]],
    matrix_file_rows: list[dict[str, object]],
    cell_rows: list[dict[str, object]],
    qc_rows: list[dict[str, object]],
    audit_rows: list[dict[str, object]],
    paths: dict[str, Path],
    manual_preview_component_rows: list[dict[str, object]] | None = None,
    manual_preview_block_rows: list[dict[str, object]] | None = None,
    manual_preview_row_rows: list[dict[str, object]] | None = None,
    manual_preview_matrix_rows: list[dict[str, object]] | None = None,
    manual_preview_matrix_file_rows: list[dict[str, object]] | None = None,
    manual_preview_cell_rows: list[dict[str, object]] | None = None,
    manual_preview_qc_rows: list[dict[str, object]] | None = None,
) -> list[dict[str, object]]:
    manual_preview_component_rows = manual_preview_component_rows or []
    manual_preview_block_rows = manual_preview_block_rows or []
    manual_preview_row_rows = manual_preview_row_rows or []
    manual_preview_matrix_rows = manual_preview_matrix_rows or []
    manual_preview_matrix_file_rows = manual_preview_matrix_file_rows or []
    manual_preview_cell_rows = manual_preview_cell_rows or []
    manual_preview_qc_rows = manual_preview_qc_rows or []
    rows: list[dict[str, object]] = [
        {"metric": "counts", "name": "materialized_physical_assemblies", "count": len(component_rows)},
        {"metric": "counts", "name": "materialized_blocks", "count": len(block_rows)},
        {"metric": "counts", "name": "materialized_rows", "count": len(row_rows)},
        {"metric": "counts", "name": "materialized_matrix_rows", "count": len(matrix_rows)},
        {"metric": "counts", "name": "materialized_matrix_columns", "count": len(matrix_column_rows)},
        {"metric": "counts", "name": "compact_matrix_files", "count": len(matrix_file_rows)},
        {"metric": "counts", "name": "materialized_cells", "count": len(cell_rows)},
        {"metric": "counts", "name": "qc_checks", "count": len(qc_rows)},
        {"metric": "counts", "name": "audit_failures", "count": sum(1 for row in audit_rows if row.get("status") == "fail")},
        {"metric": "counts", "name": "audit_warnings", "count": sum(1 for row in audit_rows if row.get("status") == "warn")},
        {"metric": "counts", "name": "manual_preview_candidates", "count": len(manual_preview_component_rows)},
        {"metric": "counts", "name": "manual_preview_blocks", "count": len(manual_preview_block_rows)},
        {"metric": "counts", "name": "manual_preview_rows", "count": len(manual_preview_row_rows)},
        {"metric": "counts", "name": "manual_preview_matrix_rows", "count": len(manual_preview_matrix_rows)},
        {"metric": "counts", "name": "manual_preview_matrix_files", "count": len(manual_preview_matrix_file_rows)},
        {"metric": "counts", "name": "manual_preview_cells", "count": len(manual_preview_cell_rows)},
        {"metric": "counts", "name": "manual_preview_qc_checks", "count": len(manual_preview_qc_rows)},
    ]
    for metric, counter in [
        ("assemblies_by_status", Counter(str(row.get("assembly_status", "")) for row in component_rows)),
        ("assemblies_by_frontier_category", Counter(str(row.get("frontier_category", "")) for row in component_rows)),
        ("qc_by_status", Counter(str(row.get("status", "")) for row in qc_rows)),
        ("row_label_source", Counter(str(row.get("row_label_source", "")) for row in row_rows)),
        ("manual_preview_by_status", Counter(str(row.get("assembly_status", "")) for row in manual_preview_component_rows)),
        ("manual_preview_qc_by_status", Counter(str(row.get("status", "")) for row in manual_preview_qc_rows)),
    ]:
        for name, count in counter.most_common():
            rows.append({"metric": metric, "name": name, "count": count})
    for name, path in paths.items():
        rows.append({"metric": "outputs", "name": name, "count": output_path(path)})
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frontier-dir", type=Path, default=DEFAULT_FRONTIER_DIR)
    parser.add_argument("--candidate-promotion-dir", type=Path, default=DEFAULT_CANDIDATE_PROMOTION_DIR)
    parser.add_argument("--control-dir", type=Path, default=DEFAULT_CONTROL_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    frontier_rows = read_csv(args.frontier_dir / "physical_assembly_frontier.csv")
    safe_edges = [row for row in frontier_rows if row.get("frontier_category") in SAFE_FRONTIER_CATEGORIES]
    manual_preview_edges = [
        row for row in frontier_rows if row.get("frontier_category", "").startswith(MANUAL_PREVIEW_CATEGORY_PREFIX)
    ]
    edge_manifest = manifest_by_edge(args.candidate_promotion_dir)

    component_rows: list[dict[str, object]] = []
    block_rows: list[dict[str, object]] = []
    row_rows: list[dict[str, object]] = []
    cell_rows: list[dict[str, object]] = []
    qc_rows: list[dict[str, object]] = []
    manual_preview_component_rows: list[dict[str, object]] = []
    manual_preview_block_rows: list[dict[str, object]] = []
    manual_preview_row_rows: list[dict[str, object]] = []
    manual_preview_cell_rows: list[dict[str, object]] = []
    manual_preview_qc_rows: list[dict[str, object]] = []

    safe_edges.sort(key=lambda row: (row.get("v5_source_key", ""), int_or_none(str(row.get("page_pair", "")).split("-")[0]) or 999999, row.get("edge_key", "")))
    for rank, edge in enumerate(safe_edges, start=1):
        manifest_row = edge_manifest.get((edge.get("v5_source_key", ""), edge.get("page_pair", "")))
        component, blocks, rows, cells, qc = materialize_edge(rank, edge, manifest_row)
        component_rows.append(component)
        block_rows.extend(blocks)
        row_rows.extend(rows)
        cell_rows.extend(cells)
        qc_rows.extend(qc)

    manual_preview_edges.sort(key=lambda row: (row.get("v5_source_key", ""), int_or_none(str(row.get("page_pair", "")).split("-")[0]) or 999999, row.get("edge_key", "")))
    for rank, edge in enumerate(manual_preview_edges, start=1):
        component, blocks, rows, cells, qc = materialize_edge(rank, edge, None, preview_only=True)
        manual_preview_component_rows.append(component)
        manual_preview_block_rows.extend(blocks)
        manual_preview_row_rows.extend(rows)
        manual_preview_cell_rows.extend(cells)
        manual_preview_qc_rows.extend(qc)

    paths = {
        "materialized_physical_assemblies": out_dir / "materialized_physical_assemblies.csv",
        "materialized_physical_blocks": out_dir / "materialized_physical_blocks.csv",
        "materialized_physical_rows": out_dir / "materialized_physical_rows.csv",
        "materialized_physical_matrix_rows": out_dir / "materialized_physical_matrix_rows.csv",
        "materialized_physical_matrix_columns": out_dir / "materialized_physical_matrix_columns.csv",
        "materialized_physical_matrix_files": out_dir / "materialized_physical_matrix_files.csv",
        "materialized_physical_cells": out_dir / "materialized_physical_cells.csv",
        "materialized_physical_qc": out_dir / "materialized_physical_qc.csv",
        "manual_physical_preview_assemblies": out_dir / "manual_physical_preview_assemblies.csv",
        "manual_physical_preview_blocks": out_dir / "manual_physical_preview_blocks.csv",
        "manual_physical_preview_rows": out_dir / "manual_physical_preview_rows.csv",
        "manual_physical_preview_matrix_rows": out_dir / "manual_physical_preview_matrix_rows.csv",
        "manual_physical_preview_matrix_columns": out_dir / "manual_physical_preview_matrix_columns.csv",
        "manual_physical_preview_matrix_files": out_dir / "manual_physical_preview_matrix_files.csv",
        "manual_physical_preview_cells": out_dir / "manual_physical_preview_cells.csv",
        "manual_physical_preview_qc": out_dir / "manual_physical_preview_qc.csv",
        "acceptance_audit": out_dir / "acceptance_audit.csv",
        "summary": out_dir / "summary.csv",
        "workbook": out_dir / "materialized_physical_assembly_workbook.xlsx",
    }
    write_csv(paths["materialized_physical_assemblies"], component_rows, COMPONENT_FIELDS)
    write_csv(paths["materialized_physical_blocks"], block_rows, BLOCK_FIELDS)
    write_csv(paths["materialized_physical_rows"], row_rows, ROW_FIELDS)
    matrix_rows, matrix_column_rows, matrix_fields = build_matrix_outputs(row_rows, cell_rows)
    matrix_file_rows = write_compact_assembly_matrices(out_dir, component_rows, row_rows, cell_rows)
    write_csv(paths["materialized_physical_matrix_rows"], matrix_rows, matrix_fields)
    write_csv(paths["materialized_physical_matrix_columns"], matrix_column_rows, MATRIX_COLUMN_FIELDS)
    write_csv(paths["materialized_physical_matrix_files"], matrix_file_rows, MATRIX_FILE_FIELDS)
    write_csv(paths["materialized_physical_cells"], cell_rows, CELL_FIELDS)
    write_csv(paths["materialized_physical_qc"], qc_rows, QC_FIELDS)
    write_csv(paths["manual_physical_preview_assemblies"], manual_preview_component_rows, COMPONENT_FIELDS)
    write_csv(paths["manual_physical_preview_blocks"], manual_preview_block_rows, BLOCK_FIELDS)
    write_csv(paths["manual_physical_preview_rows"], manual_preview_row_rows, ROW_FIELDS)
    manual_preview_matrix_rows, manual_preview_matrix_column_rows, manual_preview_matrix_fields = build_matrix_outputs(manual_preview_row_rows, manual_preview_cell_rows)
    manual_preview_matrix_file_rows = write_compact_assembly_matrices(
        out_dir,
        manual_preview_component_rows,
        manual_preview_row_rows,
        manual_preview_cell_rows,
        matrix_dir_name="manual_preview_matrix_by_candidate",
    )
    write_csv(paths["manual_physical_preview_matrix_rows"], manual_preview_matrix_rows, manual_preview_matrix_fields)
    write_csv(paths["manual_physical_preview_matrix_columns"], manual_preview_matrix_column_rows, MATRIX_COLUMN_FIELDS)
    write_csv(paths["manual_physical_preview_matrix_files"], manual_preview_matrix_file_rows, MATRIX_FILE_FIELDS)
    write_csv(paths["manual_physical_preview_cells"], manual_preview_cell_rows, CELL_FIELDS)
    write_csv(paths["manual_physical_preview_qc"], manual_preview_qc_rows, QC_FIELDS)

    preview_sheets = [
        ("Manual Preview Assemblies", manual_preview_component_rows, COMPONENT_FIELDS),
        ("Manual Preview Blocks", manual_preview_block_rows, BLOCK_FIELDS),
        ("Manual Preview Rows", manual_preview_row_rows, ROW_FIELDS),
        ("Manual Preview Matrix Files", manual_preview_matrix_file_rows, MATRIX_FILE_FIELDS),
        ("Manual Preview Matrix Rows", manual_preview_matrix_rows, manual_preview_matrix_fields),
        ("Manual Preview Matrix Columns", manual_preview_matrix_column_rows, MATRIX_COLUMN_FIELDS),
        ("Manual Preview Cells", manual_preview_cell_rows, CELL_FIELDS),
        ("Manual Preview QC", manual_preview_qc_rows, QC_FIELDS),
    ]

    write_workbook(paths["workbook"], [], [], component_rows, block_rows, row_rows, matrix_rows, matrix_column_rows, matrix_file_rows, matrix_fields, cell_rows, qc_rows, preview_sheets)
    audit_rows = build_audit(
        safe_edges,
        component_rows,
        row_rows,
        cell_rows,
        matrix_rows,
        matrix_column_rows,
        matrix_file_rows,
        qc_rows,
        paths["workbook"],
        manual_preview_edges,
        manual_preview_component_rows,
        manual_preview_matrix_file_rows,
        manual_preview_qc_rows,
    )
    summary_rows = build_summary(
        component_rows,
        block_rows,
        row_rows,
        matrix_rows,
        matrix_column_rows,
        matrix_file_rows,
        cell_rows,
        qc_rows,
        audit_rows,
        paths,
        manual_preview_component_rows,
        manual_preview_block_rows,
        manual_preview_row_rows,
        manual_preview_matrix_rows,
        manual_preview_matrix_file_rows,
        manual_preview_cell_rows,
        manual_preview_qc_rows,
    )
    write_workbook(paths["workbook"], summary_rows, audit_rows, component_rows, block_rows, row_rows, matrix_rows, matrix_column_rows, matrix_file_rows, matrix_fields, cell_rows, qc_rows, preview_sheets)
    audit_rows = build_audit(
        safe_edges,
        component_rows,
        row_rows,
        cell_rows,
        matrix_rows,
        matrix_column_rows,
        matrix_file_rows,
        qc_rows,
        paths["workbook"],
        manual_preview_edges,
        manual_preview_component_rows,
        manual_preview_matrix_file_rows,
        manual_preview_qc_rows,
    )
    summary_rows = build_summary(
        component_rows,
        block_rows,
        row_rows,
        matrix_rows,
        matrix_column_rows,
        matrix_file_rows,
        cell_rows,
        qc_rows,
        audit_rows,
        paths,
        manual_preview_component_rows,
        manual_preview_block_rows,
        manual_preview_row_rows,
        manual_preview_matrix_rows,
        manual_preview_matrix_file_rows,
        manual_preview_cell_rows,
        manual_preview_qc_rows,
    )
    write_workbook(paths["workbook"], summary_rows, audit_rows, component_rows, block_rows, row_rows, matrix_rows, matrix_column_rows, matrix_file_rows, matrix_fields, cell_rows, qc_rows, preview_sheets)
    write_csv(paths["acceptance_audit"], audit_rows, AUDIT_FIELDS)
    write_csv(paths["summary"], summary_rows, SUMMARY_FIELDS)

    summary = {
        "materialized_physical_assemblies": len(component_rows),
        "materialized_blocks": len(block_rows),
        "materialized_rows": len(row_rows),
        "materialized_matrix_rows": len(matrix_rows),
        "materialized_matrix_columns": len(matrix_column_rows),
        "compact_matrix_files": len(matrix_file_rows),
        "materialized_cells": len(cell_rows),
        "manual_preview_candidates": len(manual_preview_component_rows),
        "manual_preview_blocks": len(manual_preview_block_rows),
        "manual_preview_rows": len(manual_preview_row_rows),
        "manual_preview_matrix_rows": len(manual_preview_matrix_rows),
        "manual_preview_matrix_columns": len(manual_preview_matrix_column_rows),
        "manual_preview_matrix_files": len(manual_preview_matrix_file_rows),
        "manual_preview_cells": len(manual_preview_cell_rows),
        "manual_preview_qc_checks": len(manual_preview_qc_rows),
        "audit_failure_count": sum(1 for row in audit_rows if row.get("status") == "fail"),
        "audit_warning_count": sum(1 for row in audit_rows if row.get("status") == "warn"),
        "assemblies_by_status": dict(Counter(str(row.get("assembly_status", "")) for row in component_rows)),
        "assemblies_by_frontier_category": dict(Counter(str(row.get("frontier_category", "")) for row in component_rows)),
        "manual_preview_by_status": dict(Counter(str(row.get("assembly_status", "")) for row in manual_preview_component_rows)),
        "manual_preview_by_frontier_category": dict(Counter(str(row.get("frontier_category", "")) for row in manual_preview_component_rows)),
        "qc_by_status": dict(Counter(str(row.get("status", "")) for row in qc_rows)),
        "manual_preview_qc_by_status": dict(Counter(str(row.get("status", "")) for row in manual_preview_qc_rows)),
        "row_label_source": dict(Counter(str(row.get("row_label_source", "")) for row in row_rows)),
        "manual_preview_row_label_source": dict(Counter(str(row.get("row_label_source", "")) for row in manual_preview_row_rows)),
        "outputs": {name: output_path(path) for name, path in paths.items()},
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
