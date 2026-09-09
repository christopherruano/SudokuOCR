#!/usr/bin/env python3
"""Run and publish the first-principles assembly pipeline.

This is the stable user-facing entry point for table assembly. The older
v25-v29 scripts remain as audited implementation stages, but this script emits
one concise output directory with stable filenames and normalized columns.

No OCR/API work is performed here.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import subprocess
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import xlsxwriter


ROOT = Path(__file__).resolve().parent
DEFAULT_OUT_DIR = ROOT / "export_final/pdf_recovery_incoming_2026-06-16_first_principles_assembly_pipeline"
OUT_DIR = DEFAULT_OUT_DIR
MAX_ACTIONABLE_LOGICAL_PAGES = 6

STAGES = [
    ("seed_physical_adjacency", "build_first_principles_seed_assembly_v25.py"),
    ("row_label_inheritance", "build_first_principles_row_assembly_v26.py"),
    ("assembly_manifest", "build_first_principles_assembly_manifest_v27.py"),
    ("operational_filter", "build_first_principles_operational_manifest_v28.py"),
    ("sibling_refinement", "build_first_principles_sibling_refinement_v29.py"),
]

STAGE_OUTPUTS = {
    "v25": ROOT / "export_final/pdf_recovery_incoming_2026-06-16_first_principles_seed_assembly_v25",
    "v26": ROOT / "export_final/pdf_recovery_incoming_2026-06-16_first_principles_row_assembly_v26",
    "v27": ROOT / "export_final/pdf_recovery_incoming_2026-06-16_first_principles_assembly_manifest_v27",
    "v28": ROOT / "export_final/pdf_recovery_incoming_2026-06-16_first_principles_operational_manifest_v28",
    "v29": ROOT / "export_final/pdf_recovery_incoming_2026-06-16_first_principles_sibling_refinement_v29",
}

SOURCE_FILES = {
    "manifest": STAGE_OUTPUTS["v29"] / "v29_operational_manifest.csv",
    "v29_use_now": STAGE_OUTPUTS["v29"] / "v29_use_now_groups.csv",
    "v29_refined_sibling_groups": STAGE_OUTPUTS["v29"] / "v29_refined_sibling_groups.csv",
    "v29_refined_sibling_edges": STAGE_OUTPUTS["v29"] / "v29_refined_sibling_edges.csv",
    "v27_pairwise_edges": STAGE_OUTPUTS["v27"] / "v27_pairwise_edge_manifest.csv",
    "v27_missing_bridge": STAGE_OUTPUTS["v27"] / "v27_missing_bridge_groups.csv",
    "v26_row_attempts": STAGE_OUTPUTS["v26"] / "v26_row_assembly_attempts.csv",
    "v26_constraints": STAGE_OUTPUTS["v26"] / "v26_cross_page_constraints.csv",
    "v26_review_queue": STAGE_OUTPUTS["v26"] / "v26_row_assembly_review_queue.csv",
    "v26_structurally_complete": STAGE_OUTPUTS["v26"] / "v26_structurally_complete_assemblies.csv",
    "v26_partial": STAGE_OUTPUTS["v26"] / "v26_partial_assemblies.csv",
    "v26_rows_jsonl": STAGE_OUTPUTS["v26"] / "v26_assembled_rows.jsonl",
    "v26_artifact_resolution": STAGE_OUTPUTS["v26"] / "v26_artifact_resolution_report.csv",
    "v26_candidate_attempts": STAGE_OUTPUTS["v26"] / "v26_candidate_row_assembly_attempts.csv",
    "v26_candidate_constraints": STAGE_OUTPUTS["v26"] / "v26_candidate_cross_page_constraints.csv",
    "v26_candidate_artifact_resolution": STAGE_OUTPUTS["v26"] / "v26_candidate_artifact_resolution_report.csv",
    "v26_candidate_pass_review": STAGE_OUTPUTS["v26"] / "v26_candidate_test_pass_review_queue.csv",
    "v26_candidate_rows_jsonl": STAGE_OUTPUTS["v26"] / "v26_candidate_assembled_rows.jsonl",
    "v28_required_physical": STAGE_OUTPUTS["v28"] / "v28_required_physical_operational.csv",
    "v28_logical_operational": STAGE_OUTPUTS["v28"] / "v28_logical_operational.csv",
    "v25_seed_edges": STAGE_OUTPUTS["v25"] / "v25_pairwise_seed_edges.csv",
    "v25_seed_components": STAGE_OUTPUTS["v25"] / "v25_seed_components.csv",
    "v25_component_members": STAGE_OUTPUTS["v25"] / "v25_seed_component_members.csv",
    "v25_ironclad_physical": STAGE_OUTPUTS["v25"] / "v25_ironclad_physical_assemblies.csv",
    "v25_growth_candidates": STAGE_OUTPUTS["v25"] / "v25_growth_candidates.csv",
    "v25_logical_candidates": STAGE_OUTPUTS["v25"] / "v25_logical_collection_candidates.csv",
}

MANIFEST_FIELDS = [
    "assembly_group_id",
    "decision",
    "tier",
    "subgroup",
    "assembly_layer",
    "assembly_policy",
    "physical_merge_allowed",
    "logical_collection_only",
    "v5_source_key",
    "source_pdf",
    "source_pdf_id",
    "page_ranges",
    "n_physical_pages",
    "missing_pages",
    "primary_title",
    "reason",
    "review_flags",
    "source_evidence",
    "source_stage_group_id",
    "parent_group_id",
    "segment_kind",
    "segment_variant",
]

PREVIEW_META_FIELDS = [
    "v26_row_id",
    "v25_component_id",
    "block_index",
    "row_index",
    "row_label",
    "left_pdf_page",
    "right_pdf_page",
    "left_section_name",
    "right_section_name",
    "column_order_status",
    "column_order_flags",
    "column_order_rule",
    "left_column_order",
    "right_column_order",
    "assembled_column_order",
    "ordered_values_preview",
    "page_pair",
    "v5_source_key",
    "source_pdf",
    "left_title",
    "right_title",
    "left_image_path",
    "right_image_path",
]

COLUMN_CONTRACT_FIELDS = [
    "v25_component_id",
    "block_index",
    "column_order_status",
    "column_order_flags",
    "row_count",
    "left_pdf_page",
    "right_pdf_page",
    "page_pair",
    "v5_source_key",
    "source_pdf",
    "row_index_first",
    "row_index_last",
    "row_index_monotonic",
    "consistent_column_order",
    "left_then_right_order",
    "adjacent_pdf_pages",
    "left_column_count",
    "right_column_count",
    "assembled_column_count",
    "assembled_column_order",
    "left_title",
    "right_title",
    "left_image_path",
    "right_image_path",
]

REVIEW_QUEUE_FIELDS = [
    "queue_rank",
    "priority_level",
    "queue_bucket",
    "blocker_type",
    "recommended_next_action",
    "promotion_condition",
    "assembly_group_id",
    "decision",
    "tier",
    "subgroup",
    "assembly_layer",
    "assembly_policy",
    "v5_source_key",
    "source_pdf",
    "source_pdf_id",
    "page_ranges",
    "n_physical_pages",
    "missing_pages",
    "primary_title",
    "reason",
    "review_flags",
    "source_evidence",
    "source_stage_group_id",
    "parent_group_id",
    "segment_kind",
    "segment_variant",
    "evidence_paths",
    "image_paths",
]

BAD_PHYSICAL_CANDIDATE_KINDS = {
    "context_or_blank",
    "offtarget_table",
    "derivative_selected_caste_race_tribe",
    "unknown",
}

POTENTIAL_PHYSICAL_FIELDS = [
    "candidate_rank",
    "candidate_id",
    "confidence_tier",
    "candidate_type",
    "safe_to_auto_merge_now",
    "recommended_action",
    "promotion_condition",
    "blocker_type",
    "page_ranges",
    "v5_source_key",
    "prev_pdf_page",
    "pdf_page",
    "prev_pdf_page_parity",
    "pdf_page_parity",
    "odd_unlabeled_right_page_signal",
    "page_gap",
    "prev_physical_page_id",
    "physical_page_id",
    "prev_page_kind",
    "page_kind",
    "prev_table_family_key",
    "table_family_key",
    "prev_table_number_key",
    "table_number_key",
    "title_similarity",
    "core_table_vii_signal",
    "age_civil_signal",
    "row_label_dependency_signal",
    "physical_evidence_status",
    "prev_row_label_ratio",
    "row_label_ratio",
    "left_ocr_artifact",
    "right_ocr_artifact",
    "left_layout_left_stub_candidate",
    "left_layout_right_half_candidate",
    "right_layout_left_stub_candidate",
    "right_layout_right_half_candidate",
    "physical_pair_pattern",
    "prev_first_row_labels",
    "first_row_labels",
    "prev_last_row_labels",
    "last_row_labels",
    "prev_title",
    "title",
    "existing_v25_decision",
    "existing_v25_decision_layer",
    "existing_v25_reason",
    "existing_v25_review_flags",
    "candidate_reason",
    "candidate_review_flags",
    "prev_image_path",
    "image_path",
]

ROW_LABEL_SCAN_FIELDS = [
    "scan_rank",
    "scan_id",
    "scan_scope",
    "candidate_strength",
    "safe_to_auto_merge_now",
    "recommended_action",
    "promotion_condition",
    "row_label_pattern",
    "directional_interpretation",
    "page_ranges",
    "v5_source_key",
    "prev_pdf_page",
    "pdf_page",
    "prev_pdf_page_parity",
    "pdf_page_parity",
    "odd_unlabeled_right_page_signal",
    "page_gap",
    "prev_physical_page_id",
    "physical_page_id",
    "prev_page_kind",
    "page_kind",
    "page_kind_transition",
    "page_kind_blocker",
    "prev_table_family_key",
    "table_family_key",
    "same_table_family",
    "prev_table_number_key",
    "table_number_key",
    "same_table_number",
    "schema_reason",
    "section_reason",
    "title_similarity",
    "title_similarity_band",
    "core_table_vii_signal",
    "age_civil_signal",
    "prev_row_label_ratio",
    "row_label_ratio",
    "prev_row_label_state",
    "row_label_state",
    "left_ocr_artifact",
    "right_ocr_artifact",
    "left_layout_left_stub_candidate",
    "left_layout_right_half_candidate",
    "right_layout_left_stub_candidate",
    "right_layout_right_half_candidate",
    "physical_pair_pattern",
    "existing_v25_decision",
    "v25_edge_id",
    "existing_v25_decision_layer",
    "scan_reason",
    "scan_review_flags",
    "prev_first_row_labels",
    "first_row_labels",
    "prev_last_row_labels",
    "last_row_labels",
    "prev_title",
    "title",
    "prev_image_path",
    "image_path",
]

CANDIDATE_PASS_CONTEXT_FIELDS = [
    "review_rank",
    "v26_attempt_id",
    "source_v25_edge_id",
    "v5_source_key",
    "source_pdf",
    "source_pdf_id",
    "page_pair",
    "candidate_status",
    "safe_to_auto_promote",
    "risk_level",
    "recommended_action",
    "promotion_condition",
    "risk_flags",
    "structural_status",
    "value_constraint_status",
    "left_row_count_total",
    "right_row_count_total",
    "assembled_row_count",
    "adjacency_final_decision_layer",
    "row_label_scan_strength",
    "row_label_scan_pattern",
    "row_label_priority_scan_status",
    "prev_table_family_key",
    "table_family_key",
    "same_table_family",
    "prev_table_number_key",
    "table_number_key",
    "same_table_number",
    "title_similarity",
    "core_table_vii_signal",
    "age_civil_signal",
    "v25_decision",
    "v25_reason",
    "left_title",
    "right_title",
    "left_ocr_artifact",
    "right_ocr_artifact",
    "left_image_path",
    "right_image_path",
]

PHYSICAL_GROUPING_FIELDS = [
    "grouping_rank",
    "v26_attempt_id",
    "v25_component_id",
    "grouping_decision",
    "grouping_confidence",
    "pair_scope",
    "safe_to_group_pages",
    "safe_to_publish_values",
    "value_review_required",
    "recommended_grouping_action",
    "grouping_reason",
    "grouping_review_flags",
    "v26_status",
    "v5_source_key",
    "source_pdf",
    "source_pdf_id",
    "year",
    "province",
    "page_pair",
    "page_gap",
    "edge_layer",
    "edge_reason",
    "left_artifact_type",
    "right_artifact_type",
    "left_block_count",
    "right_block_count",
    "passed_block_count",
    "failed_block_count",
    "structural_status",
    "left_row_count_total",
    "right_row_count_total",
    "left_label_ratio",
    "right_label_ratio",
    "left_numeric_cell_count",
    "right_numeric_cell_count",
    "assembled_row_count",
    "value_constraint_status",
    "value_constraint_reason",
    "constraint_flags",
    "left_ocr_artifact",
    "right_ocr_artifact",
    "left_title",
    "right_title",
    "left_image_path",
    "right_image_path",
]

LOGICAL_GROUPING_FIELDS = [
    "logical_rank",
    "logical_grouping_decision",
    "logical_grouping_confidence",
    "safe_to_use_logical_collection",
    "safe_to_physical_merge",
    "recommended_logical_action",
    "logical_grouping_reason",
    "logical_grouping_review_flags",
    *MANIFEST_FIELDS,
]

MISSING_ARTIFACT_FIELDS = [
    "missing_target_rank",
    "target_key",
    "priority_bucket",
    "blocking_context",
    "recommended_action",
    "promotion_condition",
    "v25_component_id",
    "position_in_component",
    "physical_page_id",
    "v5_source_key",
    "source_pdf",
    "source_pdf_id",
    "pdf_page",
    "page_pair",
    "pair_gap",
    "grouping_decision",
    "grouping_reason",
    "grouping_review_flags",
    "image_path",
    "image_path_exists",
    "image_source_bucket",
    "existing_ocr_artifact",
    "resolved_ocr_artifact",
    "resolution_status",
    "resolution_method",
    "candidate_count",
    "candidate_paths",
    "expected_source_page_globs",
    "expected_image_hash_globs",
    "resolver_drop_roots",
]

ADJACENCY_DECISION_FIELDS = [
    "adjacency_rank",
    "adjacency_key",
    "coverage_scope",
    "final_decision_layer",
    "safe_to_physical_merge",
    "safe_to_logical_group",
    "recommended_action",
    "promotion_condition",
    "decision_reason",
    "decision_review_flags",
    "physical_grouping_decision",
    "physical_grouping_confidence",
    "physical_safe_to_group_pages",
    "physical_safe_to_publish_values",
    "logical_grouping_decision",
    "logical_grouping_confidence",
    "logical_safe_to_use_collection",
    "row_label_watchlist_status",
    "row_label_watchlist_tier",
    "row_label_watchlist_type",
    "row_label_watchlist_blocker",
    "row_label_scan_status",
    "row_label_scan_strength",
    "row_label_scan_pattern",
    "row_label_scan_direction",
    "row_label_priority_scan_status",
    "row_label_scan_review_flags",
    "v25_edge_id",
    "v25_decision",
    "v25_decision_layer",
    "v25_confidence",
    "v25_reason",
    "v25_review_flags",
    "v5_source_key",
    "prev_pdf_page",
    "pdf_page",
    "page_gap",
    "missing_pages",
    "prev_physical_page_id",
    "physical_page_id",
    "prev_page_kind",
    "page_kind",
    "prev_table_family_key",
    "table_family_key",
    "prev_table_number_key",
    "table_number_key",
    "schema_reason",
    "section_reason",
    "title_similarity",
    "collection_candidate",
    "collection_reason",
    "collection_review_flags",
    "prev_row_label_ratio",
    "row_label_ratio",
    "prev_first_row_labels",
    "first_row_labels",
    "prev_last_row_labels",
    "last_row_labels",
    "prev_title",
    "title",
    "prev_image_path",
    "image_path",
]

LOGICAL_COLLECTION_MEMBER_FIELDS = [
    "member_rank",
    "assembly_group_id",
    "logical_grouping_confidence",
    "subgroup",
    "segment_kind",
    "segment_variant",
    "v5_source_key",
    "source_pdf",
    "source_pdf_id",
    "collection_page_ranges",
    "n_collection_pages",
    "member_position",
    "pdf_page",
    "page_is_consecutive_with_previous",
    "title",
    "variant_terms",
    "variant_category_from_previous",
    "source_edge_id_from_previous",
    "image_path",
    "image_path_exists",
    "member_source",
    "physical_merge_allowed",
    "logical_collection_only",
    "review_flags",
]

LOGICAL_COLLECTION_CONTEXT_FIELDS = [
    "context_rank",
    "risk_level",
    "risk_flags",
    "recommended_action",
    "safe_to_use_logical_collection",
    "safe_to_physical_merge",
    "physical_merge_allowed",
    "logical_collection_only",
    "assembly_group_id",
    "logical_grouping_confidence",
    "subgroup",
    "segment_kind",
    "segment_variant",
    "v5_source_key",
    "source_pdf",
    "source_pdf_id",
    "page_ranges",
    "n_collection_pages",
    "member_count",
    "member_pages",
    "member_page_sequence_status",
    "missing_member_count",
    "member_title_count",
    "title_table_numbers",
    "title_table_number_status",
    "member_titles",
    "variant_terms_sequence",
    "variant_category_sequence",
    "variant_coverage_status",
    "image_evidence_status",
    "missing_image_member_count",
    "page_only_member_count",
    "member_source_summary",
    "logical_grouping_reason",
    "logical_grouping_review_flags",
]

LOGICAL_REVIEW_CONTEXT_FIELDS = [
    "review_context_rank",
    "review_priority",
    "review_candidate_tier",
    "review_reason",
    "recommended_action",
    "safe_to_use_logical_collection",
    "safe_to_physical_merge",
    "physical_merge_allowed",
    "logical_collection_only",
    "assembly_group_id",
    "logical_grouping_decision",
    "logical_grouping_confidence",
    "subgroup",
    "tier",
    "segment_kind",
    "v5_source_key",
    "source_pdf",
    "source_pdf_id",
    "page_ranges",
    "n_physical_pages",
    "member_count",
    "member_pages",
    "pages_are_consecutive",
    "bounded_candidate",
    "core_table_vii_signal",
    "age_civil_signal",
    "title_table_numbers",
    "title_table_number_status",
    "image_evidence_status",
    "missing_image_member_count",
    "page_only_member_count",
    "primary_title",
    "logical_grouping_reason",
    "logical_grouping_review_flags",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = union_fields(rows)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def union_fields(rows: Iterable[dict[str, object]]) -> list[str]:
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for field in row:
            if field not in seen:
                fields.append(field)
                seen.add(field)
    return fields


def load_json(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def int_or_none(value: object) -> int | None:
    s = str(value or "").strip()
    if not s:
        return None
    try:
        return int(float(s))
    except ValueError:
        return None


def classify_decision(tier: str) -> str:
    if tier == "use_now":
        return "use_now"
    if tier.startswith("blocked"):
        return "blocked"
    return "review"


def normalize_manifest_row(row: dict[str, str]) -> dict[str, object]:
    tier = row.get("v28_tier", "")
    return {
        "assembly_group_id": row.get("v28_group_id", ""),
        "decision": classify_decision(tier),
        "tier": tier,
        "subgroup": row.get("v28_subgroup", ""),
        "assembly_layer": row.get("assembly_layer", ""),
        "assembly_policy": row.get("assembly_policy", ""),
        "physical_merge_allowed": row.get("physical_merge_allowed", ""),
        "logical_collection_only": row.get("logical_collection_only", ""),
        "v5_source_key": row.get("v5_source_key", ""),
        "source_pdf": row.get("source_pdf", ""),
        "source_pdf_id": row.get("source_pdf_id", ""),
        "page_ranges": row.get("page_ranges", ""),
        "n_physical_pages": row.get("n_physical_pages", ""),
        "missing_pages": row.get("missing_pages", ""),
        "primary_title": row.get("primary_title", ""),
        "reason": row.get("reason", ""),
        "review_flags": row.get("review_flags", ""),
        "source_evidence": row.get("source_evidence", ""),
        "source_stage_group_id": row.get("v28_group_id", ""),
        "parent_group_id": "",
        "segment_kind": "source_group",
        "segment_variant": "",
    }


def stable_id(prefix: str, *parts: object) -> str:
    payload = "||".join(str(part) for part in parts)
    return f"{prefix}_{hashlib.sha1(payload.encode('utf-8')).hexdigest()[:16]}"


def parse_page_tokens(value: object) -> list[int]:
    pages: list[int] = []
    for piece in str(value or "").replace(",", ";").split(";"):
        piece = piece.strip()
        if not piece:
            continue
        if "-" in piece:
            left, right = [int_or_none(part.strip()) for part in piece.split("-", 1)]
            if left is not None and right is not None:
                pages.extend(range(min(left, right), max(left, right) + 1))
        else:
            page = int_or_none(piece)
            if page is not None:
                pages.append(page)
    return sorted(set(pages))


def page_range_from_pages(pages: list[int]) -> str:
    return "; ".join(str(page) for page in sorted(set(pages)))


def compact_text(value: object) -> str:
    return " ".join(str(value or "").split())


def float_or_zero(value: object) -> float:
    try:
        return float(str(value or "").strip() or 0)
    except ValueError:
        return 0.0


def candidate_text(row: dict[str, object]) -> str:
    return " ".join(
        str(row.get(field, ""))
        for field in [
            "prev_title",
            "title",
            "prev_table_number_key",
            "table_number_key",
            "prev_table_family_key",
            "table_family_key",
        ]
    ).lower()


def has_table_vii_signal(row: dict[str, object]) -> bool:
    text = candidate_text(row)
    return bool(re.search(r"\btable\s*vii\b|\bvii\b", text)) or row.get("prev_table_number_key") == "7" or row.get("table_number_key") == "7"


def has_age_civil_signal(row: dict[str, object]) -> bool:
    text = candidate_text(row)
    return "age" in text and "civil" in text


def physical_candidate_blocker(left_ocr: str, right_ocr: str, row: dict[str, object]) -> str:
    if not row.get("prev_image_path") or not row.get("image_path"):
        return "missing_image_path"
    if not left_ocr or not right_ocr:
        return "missing_ocr_artifact_for_row_assembly"
    if row.get("schema_reason") == "missing_schema_profile":
        return "missing_schema_profile"
    return "needs_row_assembly_constraint_test"


def physical_candidate_tier(row: dict[str, object]) -> str:
    title_similarity = float_or_zero(row.get("title_similarity"))
    if has_table_vii_signal(row) and has_age_civil_signal(row) and title_similarity >= 0.55:
        return "high_core_table_vii_row_label_pair"
    if has_table_vii_signal(row) and has_age_civil_signal(row):
        return "medium_core_table_vii_needs_title_review"
    if has_age_civil_signal(row) and title_similarity >= 0.55:
        return "medium_age_civil_row_label_pair"
    return "low_structured_row_label_pair"


def row_label_state(ratio: float) -> str:
    if ratio >= 0.50:
        return "labelled"
    if ratio <= 0.25:
        return "unlabelled_or_placeholder"
    return "partial_or_weak_labels"


def row_label_pattern(prev_state: str, row_state: str) -> str:
    if prev_state == "labelled" and row_state == "unlabelled_or_placeholder":
        return "left_labelled_right_unlabelled"
    if prev_state == "unlabelled_or_placeholder" and row_state == "labelled":
        return "left_unlabelled_right_labelled"
    if prev_state == "unlabelled_or_placeholder" and row_state == "unlabelled_or_placeholder":
        return "both_unlabelled_or_placeholder"
    if prev_state == "labelled" and row_state == "partial_or_weak_labels":
        return "left_labelled_right_weak"
    if prev_state == "partial_or_weak_labels" and row_state == "labelled":
        return "left_weak_right_labelled"
    if prev_state == "partial_or_weak_labels" or row_state == "partial_or_weak_labels":
        return "weak_label_context"
    return "both_labelled"


def title_similarity_band(value: object) -> str:
    similarity = float_or_zero(value)
    if similarity >= 0.80:
        return "high_0.80_plus"
    if similarity >= 0.55:
        return "medium_0.55_plus"
    if similarity > 0:
        return "low_nonzero"
    return "none_or_missing"


def page_parity(value: object) -> str:
    page = int_or_none(value)
    if page is None:
        return "unknown"
    return "odd" if page % 2 else "even"


def row_label_directional_interpretation(pattern: str) -> str:
    if pattern == "left_labelled_right_unlabelled":
        return "possible_right_page_continuation_inherits_left_row_labels"
    if pattern == "left_unlabelled_right_labelled":
        return "possible_left_page_tail_or_previous_missing_context_not_forward_merge"
    if pattern == "both_unlabelled_or_placeholder":
        return "possible_middle_or_right_half_run_needs_bounding_labelled_page"
    if pattern == "left_labelled_right_weak":
        return "possible_partial_right_labels_review"
    if pattern == "left_weak_right_labelled":
        return "possible_partial_left_tail_review"
    return "weak_row_label_context_review"


def row_label_scan_strength(edge: dict[str, object], pattern: str, page_kind_blocker: str, same_family: bool, same_number: bool) -> str:
    if page_kind_blocker:
        return "excluded_context_or_derivative_review_only"
    if pattern == "left_labelled_right_unlabelled" and same_family and has_age_civil_signal(edge) and float_or_zero(edge.get("title_similarity")) >= 0.55:
        if has_table_vii_signal(edge):
            return "high_core_table_vii_row_label_dependency_review"
        return "high_age_civil_row_label_dependency_review"
    if pattern == "left_labelled_right_unlabelled" and same_family and (same_number or has_age_civil_signal(edge)):
        return "medium_row_label_dependency_review"
    if pattern in {"left_unlabelled_right_labelled", "both_unlabelled_or_placeholder"} and same_family:
        return "medium_boundary_or_middle_run_review"
    if same_family or same_number:
        return "low_same_table_low_label_review"
    return "low_low_label_context_review"


def candidate_rank_key(row: dict[str, object]) -> tuple[int, str, int, str]:
    priority = {
        "high_core_table_vii_row_label_pair": 1,
        "medium_core_table_vii_needs_title_review": 2,
        "medium_age_civil_row_label_pair": 3,
        "low_structured_row_label_pair": 4,
    }.get(str(row.get("confidence_tier", "")), 9)
    return (
        priority,
        str(row.get("v5_source_key", "")),
        int_or_none(row.get("prev_pdf_page")) or 10**9,
        str(row.get("candidate_id", "")),
    )


def row_label_scan_rank_key(row: dict[str, object]) -> tuple[int, str, int, str]:
    priority = {
        "high_core_table_vii_row_label_dependency_review": 1,
        "high_age_civil_row_label_dependency_review": 2,
        "medium_row_label_dependency_review": 3,
        "medium_boundary_or_middle_run_review": 4,
        "low_same_table_low_label_review": 5,
        "excluded_context_or_derivative_review_only": 6,
        "low_low_label_context_review": 7,
    }.get(str(row.get("candidate_strength", "")), 9)
    return (
        priority,
        str(row.get("v5_source_key", "")),
        int_or_none(row.get("prev_pdf_page")) or 10**9,
        str(row.get("scan_id", "")),
    )


def page_pair_gap(page_pair: object) -> int | None:
    parts = str(page_pair or "").split("-", 1)
    if len(parts) != 2:
        return None
    left = int_or_none(parts[0])
    right = int_or_none(parts[1])
    if left is None or right is None:
        return None
    return right - left


def physical_grouping_rank_key(row: dict[str, object]) -> tuple[int, str, int, str]:
    priority = {
        "confirmed_pairwise_physical_grouping": 1,
        "partial_pairwise_physical_grouping_review": 2,
        "blocked_missing_ocr_artifact": 3,
        "review_failed_constraints": 4,
        "review_no_safe_block_alignment": 5,
        "blocked_parser": 6,
    }.get(str(row.get("grouping_decision", "")), 9)
    page_pair = str(row.get("page_pair") or "")
    start_page = int_or_none(page_pair.split("-", 1)[0]) or 10**9
    return (
        priority,
        str(row.get("source_pdf") or row.get("v5_source_key") or ""),
        start_page,
        str(row.get("v26_attempt_id") or ""),
    )


def physical_grouping_decision(attempt: dict[str, object]) -> dict[str, str]:
    gap = page_pair_gap(attempt.get("page_pair"))
    structural_status = str(attempt.get("structural_status") or "")
    value_status = str(attempt.get("value_constraint_status") or "")
    status = str(attempt.get("v26_status") or "")
    assembled_rows = int_or_none(attempt.get("assembled_row_count")) or 0
    passed_blocks = int_or_none(attempt.get("passed_block_count")) or 0
    failed_blocks = int_or_none(attempt.get("failed_block_count")) or 0
    flags: list[str] = ["pairwise_only"]
    if gap != 1:
        flags.append(f"not_adjacent_gap_{gap if gap is not None else 'unknown'}")

    if gap == 1 and structural_status == "pass" and assembled_rows:
        safe_values = value_status == "pass"
        if not safe_values:
            flags.append("value_review_required")
        return {
            "grouping_decision": "confirmed_pairwise_physical_grouping",
            "grouping_confidence": "high_structural_pass_value_pass" if safe_values else "high_structural_pass_value_review",
            "pair_scope": "two_consecutive_pdf_pages",
            "safe_to_group_pages": "yes",
            "safe_to_publish_values": "yes" if safe_values else "no",
            "value_review_required": "no" if safe_values else "yes",
            "recommended_grouping_action": "record_grouping_and_publish_values" if safe_values else "record_grouping_but_review_values_before_numeric_publish",
            "grouping_reason": "adjacent PDF pages; row blocks align; right page inherits row labels from left page; column order is left page then right page",
            "grouping_review_flags": "; ".join(flags),
        }

    if gap == 1 and structural_status == "partial" and passed_blocks > 0 and assembled_rows:
        flags.extend(["partial_block_alignment", f"failed_block_count_{failed_blocks}"])
        return {
            "grouping_decision": "partial_pairwise_physical_grouping_review",
            "grouping_confidence": "medium_partial_structural_pass",
            "pair_scope": "two_consecutive_pdf_pages_partial_blocks_only",
            "safe_to_group_pages": "no",
            "safe_to_publish_values": "no",
            "value_review_required": "yes",
            "recommended_grouping_action": "review_passing_blocks_only; do_not_merge_full_page_pair_until_failed_blocks_are_resolved",
            "grouping_reason": "adjacent PDF pages and at least one block aligns, but not every block is safe",
            "grouping_review_flags": "; ".join(flags),
        }

    if status == "blocked_missing_ocr_artifact":
        flags.append("missing_ocr_artifact")
        return {
            "grouping_decision": "blocked_missing_ocr_artifact",
            "grouping_confidence": "blocked",
            "pair_scope": "two_consecutive_pdf_pages_unparsed",
            "safe_to_group_pages": "no",
            "safe_to_publish_values": "no",
            "value_review_required": "yes",
            "recommended_grouping_action": "attach_existing_ocr_artifacts_or_ocr_existing_images_before_grouping",
            "grouping_reason": "pair is a seed physical candidate, but row assembly cannot verify without both OCR artifacts",
            "grouping_review_flags": "; ".join(flags),
        }

    if status == "blocked_parser":
        flags.append("parser_blocked")
        decision = "blocked_parser"
    elif status == "review_no_safe_block_alignment":
        flags.append("no_safe_block_alignment")
        decision = "review_no_safe_block_alignment"
    else:
        flags.append("failed_or_unproven_constraints")
        decision = "review_failed_constraints"
    return {
        "grouping_decision": decision,
        "grouping_confidence": "review_required",
        "pair_scope": "two_consecutive_pdf_pages_unconfirmed",
        "safe_to_group_pages": "no",
        "safe_to_publish_values": "no",
        "value_review_required": "yes",
        "recommended_grouping_action": "manual_review_or_better_ocr_before_grouping",
        "grouping_reason": "consecutive seed pair exists, but current row/block constraints do not prove a safe grouping",
        "grouping_review_flags": "; ".join(flags),
    }


def logical_grouping_rank_key(row: dict[str, object]) -> tuple[int, str, int, str]:
    priority = {
        "confirmed_bounded_logical_collection": 1,
        "review_large_logical_collection": 2,
        "review_same_table_logical_candidate": 3,
        "blocked_missing_pages_logical_candidate": 4,
    }.get(str(row.get("logical_grouping_decision", "")), 9)
    pages = parse_page_tokens(row.get("page_ranges", ""))
    return (
        priority,
        str(row.get("v5_source_key") or row.get("source_pdf") or ""),
        pages[0] if pages else 10**9,
        str(row.get("assembly_group_id") or ""),
    )


def logical_grouping_decision(row: dict[str, object]) -> dict[str, str]:
    n_pages = int_or_none(row.get("n_physical_pages")) or 0
    pages_are_contiguous = contiguous_pages(str(row.get("page_ranges", "")))
    common_flags = ["logical_collection_only", "no_physical_merge"]
    if row.get("missing_pages"):
        return {
            "logical_grouping_decision": "blocked_missing_pages_logical_candidate",
            "logical_grouping_confidence": "blocked_missing_pages",
            "safe_to_use_logical_collection": "no",
            "safe_to_physical_merge": "no",
            "recommended_logical_action": "fill_or_review_missing_pdf_pages_before_using_collection",
            "logical_grouping_reason": "logical candidate crosses missing PDF pages",
            "logical_grouping_review_flags": "; ".join([*common_flags, "missing_pages"]),
        }
    if row.get("decision") == "use_now":
        flags = [*common_flags]
        if n_pages > MAX_ACTIONABLE_LOGICAL_PAGES:
            flags.append("unexpected_large_use_now_logical_group")
        if not pages_are_contiguous:
            flags.append("unexpected_noncontiguous_use_now_logical_group")
        confidence = "high_bounded_logical_collection"
        if row.get("segment_kind") == "variant_child_segment":
            confidence = "high_bounded_child_variant_collection"
        elif "sibling" in str(row.get("subgroup", "")):
            confidence = "high_bounded_sibling_collection"
        elif "self_contained" in str(row.get("subgroup", "")):
            confidence = "high_self_contained_sequence_collection"
        return {
            "logical_grouping_decision": "confirmed_bounded_logical_collection",
            "logical_grouping_confidence": confidence,
            "safe_to_use_logical_collection": "yes" if n_pages <= MAX_ACTIONABLE_LOGICAL_PAGES and pages_are_contiguous else "no",
            "safe_to_physical_merge": "no",
            "recommended_logical_action": "use_as_logical_collection_only",
            "logical_grouping_reason": "bounded consecutive logical collection with explicit no-physical-merge policy",
            "logical_grouping_review_flags": "; ".join(flags),
        }
    if row.get("segment_kind") == "large_parent_context" or row.get("tier") == "review_large_collection":
        return {
            "logical_grouping_decision": "review_large_logical_collection",
            "logical_grouping_confidence": "review_large_or_parent_context",
            "safe_to_use_logical_collection": "no",
            "safe_to_physical_merge": "no",
            "recommended_logical_action": "use_bounded_children_or_refined_siblings_instead_of_parent_group",
            "logical_grouping_reason": "large logical parent/context is retained for provenance but not as a clean use-now collection",
            "logical_grouping_review_flags": "; ".join([*common_flags, "large_or_parent_context"]),
        }
    return {
        "logical_grouping_decision": "review_same_table_logical_candidate",
        "logical_grouping_confidence": "review_required",
        "safe_to_use_logical_collection": "no",
        "safe_to_physical_merge": "no",
        "recommended_logical_action": "manual_review_or_wait_for_refined_sibling_split",
        "logical_grouping_reason": "same-table logical relationship is plausible but not bounded enough for use-now",
        "logical_grouping_review_flags": "; ".join([*common_flags, "not_promoted"]),
    }


def split_variant_terms(value: object) -> list[str]:
    terms: list[str] = []
    seen: set[str] = set()
    for term in str(value or "").split(";"):
        term = compact_text(term)
        if term and term not in seen:
            seen.add(term)
            terms.append(term)
    return terms


def variant_key(value: object) -> str:
    terms = split_variant_terms(value)
    return "+".join(terms) if terms else "unknown_variant"


def build_refined_group_lookup() -> tuple[dict[str, dict[str, str]], dict[str, dict[str, str]]]:
    groups = {
        row.get("v29_group_id", ""): row
        for row in read_csv(OUT_DIR / "refined_sibling_groups.csv")
    }
    edges = {
        row.get("v29_edge_id", ""): row
        for row in read_csv(OUT_DIR / "refined_sibling_edges.csv")
    }
    return groups, edges


def refined_child_segments(parent: dict[str, object], refined_groups: dict[str, dict[str, str]], refined_edges: dict[str, dict[str, str]]) -> list[dict[str, object]]:
    parent_id = str(parent.get("assembly_group_id", ""))
    source_group = refined_groups.get(parent_id)
    if not source_group:
        return []
    edge_ids = [edge_id.strip() for edge_id in source_group.get("source_edge_ids", "").split(";") if edge_id.strip()]
    page_nodes: dict[int, dict[str, str]] = {}
    for edge_id in edge_ids:
        edge = refined_edges.get(edge_id)
        if not edge:
            continue
        prev_page = int_or_none(edge.get("prev_pdf_page"))
        page = int_or_none(edge.get("pdf_page"))
        if prev_page is not None:
            page_nodes.setdefault(
                prev_page,
                {
                    "page": str(prev_page),
                    "title": edge.get("prev_title", ""),
                    "variant": variant_key(edge.get("prev_variant_terms", "")),
                    "image_path": edge.get("prev_image_path", ""),
                },
            )
        if page is not None:
            page_nodes[page] = {
                "page": str(page),
                "title": edge.get("title", ""),
                "variant": variant_key(edge.get("variant_terms", "")),
                "image_path": edge.get("image_path", ""),
            }
    pages = sorted(page_nodes)
    if len(pages) < 2:
        return []

    runs: list[list[int]] = []
    current: list[int] = []
    current_variant = ""
    for page in pages:
        node = page_nodes[page]
        node_variant = node["variant"]
        if (
            not current
            or (page == current[-1] + 1 and node_variant == current_variant and len(current) < MAX_ACTIONABLE_LOGICAL_PAGES)
        ):
            current.append(page)
            current_variant = node_variant
        else:
            runs.append(current)
            current = [page]
            current_variant = node_variant
    if current:
        runs.append(current)

    # A large parent can be a sequence of single-page sibling subtables. Those
    # are useful as child records, but they are not assemblies. Keep them only
    # when they retain explicit variant evidence and parent linkage.
    child_rows: list[dict[str, object]] = []
    for idx, run in enumerate(runs, start=1):
        variant = page_nodes[run[0]]["variant"]
        titles = [page_nodes[page]["title"] for page in run]
        image_paths = [page_nodes[page]["image_path"] for page in run]
        child_rows.append(
            {
                "assembly_group_id": stable_id("logicalchild", parent_id, idx, page_range_from_pages(run), variant),
                "decision": "use_now",
                "tier": "use_now",
                "subgroup": "logical_child_segment",
                "assembly_layer": "logical_collection",
                "assembly_policy": "logical_collection_only",
                "physical_merge_allowed": "no",
                "logical_collection_only": "yes",
                "v5_source_key": parent.get("v5_source_key", ""),
                "source_pdf": parent.get("source_pdf", ""),
                "source_pdf_id": parent.get("source_pdf_id", ""),
                "page_ranges": page_range_from_pages(run),
                "n_physical_pages": len(run),
                "missing_pages": "",
                "primary_title": "; ".join(compact_text(title) for title in titles if compact_text(title)),
                "reason": f"bounded child segment split from large logical parent by variant={variant}",
                "review_flags": "logical_child_segment; no_physical_merge; parent_collection_context",
                "source_evidence": "v29_pairwise_refined_sibling_edges_child_segment",
                "source_stage_group_id": parent.get("source_stage_group_id", ""),
                "parent_group_id": parent_id,
                "segment_kind": "variant_child_segment",
                "segment_variant": variant,
                "image_paths": "; ".join(path for path in image_paths if path),
            }
        )
    return child_rows


def apply_logical_parent_child_policy(manifest: list[dict[str, object]]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    refined_groups, refined_edges = build_refined_group_lookup()
    final_rows: list[dict[str, object]] = []
    child_rows: list[dict[str, object]] = []
    for row in manifest:
        updated = dict(row)
        is_large_logical_use_now = (
            updated.get("decision") == "use_now"
            and updated.get("assembly_layer") == "logical_collection"
            and (int_or_none(updated.get("n_physical_pages")) or 0) > MAX_ACTIONABLE_LOGICAL_PAGES
        )
        if is_large_logical_use_now:
            children = refined_child_segments(updated, refined_groups, refined_edges)
            updated["decision"] = "review"
            updated["tier"] = "review_large_logical_parent"
            updated["subgroup"] = f"{updated.get('subgroup', '')}_parent_context"
            updated["assembly_policy"] = "logical_parent_context_after_review"
            updated["reason"] = f"{updated.get('reason', '')}; demoted because parent has more than {MAX_ACTIONABLE_LOGICAL_PAGES} pages"
            updated["review_flags"] = "; ".join(
                part
                for part in [str(updated.get("review_flags", "")), "large_logical_parent_context_only"]
                if part
            )
            updated["segment_kind"] = "large_parent_context"
            if children:
                updated["reason"] = f"{updated.get('reason', '')}; child segments emitted"
                child_rows.extend(children)
            else:
                updated["reason"] = f"{updated.get('reason', '')}; no safe refined child split available"
        final_rows.append(updated)
    final_rows.extend(child_rows)
    final_rows.sort(
        key=lambda r: (
            str(r.get("decision")),
            str(r.get("assembly_layer")),
            str(r.get("v5_source_key") or r.get("source_pdf")),
            int_or_none(str(r.get("page_ranges")).split(";", 1)[0].split("-", 1)[0]) or 10**9,
            str(r.get("assembly_group_id")),
        )
    )
    return final_rows, child_rows


def run_stages(skip_stages: bool) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for stage_name, script in STAGES:
        path = ROOT / script
        if not path.exists():
            rows.append({"stage": stage_name, "script": script, "status": "missing_script"})
            continue
        if skip_stages:
            rows.append({"stage": stage_name, "script": script, "status": "skipped_existing_outputs"})
            continue
        result = subprocess.run(
            [sys.executable, str(path)],
            cwd=ROOT,
            check=False,
            text=True,
            capture_output=True,
        )
        rows.append(
            {
                "stage": stage_name,
                "script": script,
                "status": "pass" if result.returncode == 0 else "failed",
                "returncode": result.returncode,
                "stdout_tail": result.stdout[-4000:],
                "stderr_tail": result.stderr[-4000:],
            }
        )
        if result.returncode != 0:
            raise RuntimeError(f"{script} failed; see stage_status.csv in {OUT_DIR}")
    return rows


def copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def row_column_order(row: dict[str, object]) -> list[str]:
    order = row.get("column_order")
    values = row.get("values")
    if isinstance(order, list):
        return [str(value) for value in order]
    if isinstance(values, dict):
        return [str(key) for key in values.keys()]
    return []


def left_then_right_order(order: list[str]) -> bool:
    seen_right = False
    has_left = False
    has_right = False
    for key in order:
        if key.startswith("right."):
            seen_right = True
            has_right = True
        elif key.startswith("left."):
            has_left = True
            if seen_right:
                return False
    return has_left and has_right


def row_column_order_flags(row: dict[str, object], order: list[str], values: dict[str, object]) -> list[str]:
    flags: list[str] = []
    if not order:
        flags.append("missing_column_order")
    if order and not left_then_right_order(order):
        flags.append("left_right_column_interleaving_or_missing_side")
    missing_values = [key for key in order if key not in values]
    if missing_values:
        flags.append("column_order_references_missing_values")
    extra_values = [key for key in values if key not in set(order)]
    if extra_values:
        flags.append("values_not_represented_in_column_order")
    left_int = int_or_none(row.get("left_pdf_page"))
    right_int = int_or_none(row.get("right_pdf_page"))
    if left_int is None or right_int is None or right_int - left_int != 1:
        flags.append("pdf_pages_not_adjacent")
    return flags


def ordered_values_preview(values: dict[str, object], order: list[str], limit: int = 12) -> str:
    parts: list[str] = []
    for key in order[:limit]:
        value = values.get(key, "")
        if value in ("", None):
            continue
        parts.append(f"{key}={value}")
    if len(order) > limit:
        parts.append(f"... +{len(order) - limit} cols")
    return " | ".join(parts)


def build_column_order_outputs() -> dict[str, list[dict[str, object]]]:
    attempts = {
        row.get("v25_component_id", ""): row
        for row in read_csv(OUT_DIR / "row_assembly_attempts.csv")
    }
    assembled_rows = read_jsonl(OUT_DIR / "assembled_rows.jsonl")
    preview_rows: list[dict[str, object]] = []
    preview_value_fields: list[str] = []
    preview_value_seen: set[str] = set()
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)

    for row in assembled_rows:
        comp_id = str(row.get("v25_component_id", ""))
        attempt = attempts.get(comp_id, {})
        raw_values = row.get("values")
        values = raw_values if isinstance(raw_values, dict) else {}
        order = row_column_order(row)
        row_flags = row_column_order_flags(row, order, values)
        left_order = [key for key in order if key.startswith("left.")]
        right_order = [key for key in order if key.startswith("right.")]
        for key in order:
            if key not in preview_value_seen:
                preview_value_seen.add(key)
                preview_value_fields.append(key)
        preview = {
            "v26_row_id": row.get("v26_row_id", ""),
            "v25_component_id": comp_id,
            "block_index": row.get("block_index", ""),
            "row_index": row.get("row_index", ""),
            "row_label": row.get("row_label", ""),
            "left_pdf_page": row.get("left_pdf_page", ""),
            "right_pdf_page": row.get("right_pdf_page", ""),
            "left_section_name": row.get("left_section_name", ""),
            "right_section_name": row.get("right_section_name", ""),
            "column_order_status": "pass" if not row_flags else "fail",
            "column_order_flags": "; ".join(row_flags),
            "column_order_rule": row.get("column_order_rule", ""),
            "left_column_order": "; ".join(left_order),
            "right_column_order": "; ".join(right_order),
            "assembled_column_order": "; ".join(order),
            "ordered_values_preview": ordered_values_preview(values, order),
            "page_pair": attempt.get("page_pair", ""),
            "v5_source_key": attempt.get("v5_source_key", ""),
            "source_pdf": attempt.get("source_pdf", ""),
            "left_title": attempt.get("left_title", ""),
            "right_title": attempt.get("right_title", ""),
            "left_image_path": attempt.get("left_image_path", ""),
            "right_image_path": attempt.get("right_image_path", ""),
        }
        for key in order:
            preview[key] = values.get(key, "")
        preview_rows.append(preview)
        grouped[(comp_id, str(row.get("block_index", "")))].append(row)

    contract_rows: list[dict[str, object]] = []
    for (comp_id, block_index), rows in sorted(grouped.items()):
        attempt = attempts.get(comp_id, {})
        row_indexes = [int_or_none(row.get("row_index")) for row in rows]
        numeric_indexes = [idx for idx in row_indexes if idx is not None]
        orders = [row_column_order(row) for row in rows]
        first_order = orders[0] if orders else []
        unique_orders = {json.dumps(order, ensure_ascii=False) for order in orders}
        left_page = rows[0].get("left_pdf_page", "") if rows else ""
        right_page = rows[0].get("right_pdf_page", "") if rows else ""
        left_int = int_or_none(left_page)
        right_int = int_or_none(right_page)
        flags: list[str] = []
        if not first_order:
            flags.append("missing_column_order")
        if len(unique_orders) != 1:
            flags.append("inconsistent_column_order_across_rows")
        if not left_then_right_order(first_order):
            flags.append("left_right_column_interleaving_or_missing_side")
        if not numeric_indexes or numeric_indexes != sorted(numeric_indexes) or len(set(numeric_indexes)) != len(numeric_indexes):
            flags.append("row_index_not_strictly_monotonic")
        if left_int is None or right_int is None or right_int - left_int != 1:
            flags.append("pdf_pages_not_adjacent")
        left_count = sum(1 for key in first_order if key.startswith("left."))
        right_count = sum(1 for key in first_order if key.startswith("right."))
        contract_rows.append(
            {
                "v25_component_id": comp_id,
                "block_index": block_index,
                "column_order_status": "pass" if not flags else "fail",
                "column_order_flags": "; ".join(flags),
                "row_count": len(rows),
                "left_pdf_page": left_page,
                "right_pdf_page": right_page,
                "page_pair": attempt.get("page_pair", ""),
                "v5_source_key": attempt.get("v5_source_key", ""),
                "source_pdf": attempt.get("source_pdf", ""),
                "row_index_first": min(numeric_indexes) if numeric_indexes else "",
                "row_index_last": max(numeric_indexes) if numeric_indexes else "",
                "row_index_monotonic": "yes" if "row_index_not_strictly_monotonic" not in flags else "no",
                "consistent_column_order": "yes" if "inconsistent_column_order_across_rows" not in flags else "no",
                "left_then_right_order": "yes" if "left_right_column_interleaving_or_missing_side" not in flags else "no",
                "adjacent_pdf_pages": "yes" if "pdf_pages_not_adjacent" not in flags else "no",
                "left_column_count": left_count,
                "right_column_count": right_count,
                "assembled_column_count": len(first_order),
                "assembled_column_order": "; ".join(first_order),
                "left_title": attempt.get("left_title", ""),
                "right_title": attempt.get("right_title", ""),
                "left_image_path": attempt.get("left_image_path", ""),
                "right_image_path": attempt.get("right_image_path", ""),
            }
        )

    write_csv(OUT_DIR / "assembled_row_previews.csv", preview_rows, [*PREVIEW_META_FIELDS, *preview_value_fields])
    write_csv(OUT_DIR / "column_order_contract.csv", contract_rows, COLUMN_CONTRACT_FIELDS)
    return {"assembled_preview": preview_rows, "column_contract": contract_rows}


def build_potential_physical_join_candidates() -> dict[str, list[dict[str, object]]]:
    edges = read_csv(OUT_DIR / "seed_pairwise_edges.csv")
    members = {
        row.get("physical_page_id", ""): row
        for row in read_csv(OUT_DIR / "seed_component_members.csv")
    }
    scan_rows: list[dict[str, object]] = []
    watchlist_rows: list[dict[str, object]] = []
    physical_rows: list[dict[str, object]] = []
    for edge in edges:
        if edge.get("page_gap") != "1" or edge.get("missing_pages"):
            continue
        prev_ratio = float_or_zero(edge.get("prev_row_label_ratio"))
        row_ratio = float_or_zero(edge.get("row_label_ratio"))
        prev_state = row_label_state(prev_ratio)
        current_state = row_label_state(row_ratio)
        pattern = row_label_pattern(prev_state, current_state)
        left_member = members.get(edge.get("prev_physical_page_id", ""), {})
        right_member = members.get(edge.get("physical_page_id", ""), {})
        left_stub = str(left_member.get("layout_left_stub_candidate", ""))
        left_half = str(left_member.get("layout_right_half_candidate", ""))
        right_stub = str(right_member.get("layout_left_stub_candidate", ""))
        right_half = str(right_member.get("layout_right_half_candidate", ""))
        physical_pair_pattern = (
            left_stub == "yes"
            and left_half == "no"
            and right_stub == "no"
            and right_half == "yes"
        )
        left_ocr = str(left_member.get("ocr_artifact", ""))
        right_ocr = str(right_member.get("ocr_artifact", ""))
        same_family = bool(edge.get("table_family_key")) and edge.get("prev_table_family_key") == edge.get("table_family_key")
        same_number = bool(edge.get("table_number_key")) and edge.get("prev_table_number_key") == edge.get("table_number_key")
        page_kind_blocker = ""
        if edge.get("prev_page_kind") in {"context_or_blank", "offtarget_table", "derivative_selected_caste_race_tribe"}:
            page_kind_blocker = f"prev_{edge.get('prev_page_kind')}"
        elif edge.get("page_kind") in {"context_or_blank", "offtarget_table", "derivative_selected_caste_race_tribe"}:
            page_kind_blocker = f"current_{edge.get('page_kind')}"

        if prev_state == "unlabelled_or_placeholder" or current_state == "unlabelled_or_placeholder":
            strength = row_label_scan_strength(edge, pattern, page_kind_blocker, same_family, same_number)
            scan_id = stable_id(
                "rlscan",
                edge.get("v5_source_key"),
                edge.get("prev_physical_page_id"),
                edge.get("physical_page_id"),
                edge.get("prev_pdf_page"),
                edge.get("pdf_page"),
            )
            scan_flags = ["scan_only", "pairwise_consecutive", "not_auto_merge"]
            if page_kind_blocker:
                scan_flags.append(page_kind_blocker)
            if edge.get("schema_reason") == "missing_schema_profile":
                scan_flags.append("missing_schema_profile")
            if not same_family:
                scan_flags.append("family_switch_or_missing_family")
            if not same_number:
                scan_flags.append("table_number_switch_or_missing_number")
            if physical_pair_pattern:
                scan_flags.append("stored_layout_signature")
            odd_unlabeled_right_page_signal = (
                pattern == "left_labelled_right_unlabelled"
                and page_parity(edge.get("pdf_page")) == "odd"
            )
            if odd_unlabeled_right_page_signal:
                scan_flags.append("odd_unlabeled_right_page")
            scan_rows.append(
                {
                    "scan_rank": 0,
                    "scan_id": scan_id,
                    "scan_scope": "row_label_low_or_missing_consecutive_edge",
                    "candidate_strength": strength,
                    "safe_to_auto_merge_now": "no",
                    "recommended_action": "Use as a coverage/review lead only; promote to physical assembly only after row-label inheritance, table continuity, and column-order constraints pass.",
                    "promotion_condition": "Pair must remain consecutive, avoid context/derivative page kinds, preserve source PDF order, and prove the unlabeled page belongs to the same table body.",
                    "row_label_pattern": pattern,
                    "directional_interpretation": row_label_directional_interpretation(pattern),
                    "page_ranges": f"{edge.get('prev_pdf_page')}-{edge.get('pdf_page')}",
                    "v5_source_key": edge.get("v5_source_key", ""),
                    "prev_pdf_page": edge.get("prev_pdf_page", ""),
                    "pdf_page": edge.get("pdf_page", ""),
                    "prev_pdf_page_parity": page_parity(edge.get("prev_pdf_page")),
                    "pdf_page_parity": page_parity(edge.get("pdf_page")),
                    "odd_unlabeled_right_page_signal": "yes" if odd_unlabeled_right_page_signal else "no",
                    "page_gap": edge.get("page_gap", ""),
                    "prev_physical_page_id": edge.get("prev_physical_page_id", ""),
                    "physical_page_id": edge.get("physical_page_id", ""),
                    "prev_page_kind": edge.get("prev_page_kind", ""),
                    "page_kind": edge.get("page_kind", ""),
                    "page_kind_transition": f"{edge.get('prev_page_kind', '')}->{edge.get('page_kind', '')}",
                    "page_kind_blocker": page_kind_blocker,
                    "prev_table_family_key": edge.get("prev_table_family_key", ""),
                    "table_family_key": edge.get("table_family_key", ""),
                    "same_table_family": "yes" if same_family else "no",
                    "prev_table_number_key": edge.get("prev_table_number_key", ""),
                    "table_number_key": edge.get("table_number_key", ""),
                    "same_table_number": "yes" if same_number else "no",
                    "schema_reason": edge.get("schema_reason", ""),
                    "section_reason": edge.get("section_reason", ""),
                    "title_similarity": edge.get("title_similarity", ""),
                    "title_similarity_band": title_similarity_band(edge.get("title_similarity")),
                    "core_table_vii_signal": "yes" if has_table_vii_signal(edge) else "no",
                    "age_civil_signal": "yes" if has_age_civil_signal(edge) else "no",
                    "prev_row_label_ratio": edge.get("prev_row_label_ratio", ""),
                    "row_label_ratio": edge.get("row_label_ratio", ""),
                    "prev_row_label_state": prev_state,
                    "row_label_state": current_state,
                    "left_ocr_artifact": left_ocr,
                    "right_ocr_artifact": right_ocr,
                    "left_layout_left_stub_candidate": left_stub,
                    "left_layout_right_half_candidate": left_half,
                    "right_layout_left_stub_candidate": right_stub,
                    "right_layout_right_half_candidate": right_half,
                    "physical_pair_pattern": "yes" if physical_pair_pattern else "no",
                    "existing_v25_decision": edge.get("v25_decision", ""),
                    "v25_edge_id": edge.get("v25_edge_id", ""),
                    "existing_v25_decision_layer": edge.get("v25_decision_layer", ""),
                    "scan_reason": "consecutive source-PDF pages with low or placeholder row labels on at least one page",
                    "scan_review_flags": "; ".join(scan_flags),
                    "prev_first_row_labels": edge.get("prev_first_row_labels", ""),
                    "first_row_labels": edge.get("first_row_labels", ""),
                    "prev_last_row_labels": edge.get("prev_last_row_labels", ""),
                    "last_row_labels": edge.get("last_row_labels", ""),
                    "prev_title": edge.get("prev_title", ""),
                    "title": edge.get("title", ""),
                    "prev_image_path": edge.get("prev_image_path", ""),
                    "image_path": edge.get("image_path", ""),
                }
            )

        if edge.get("v25_decision") == "seed_physical_join":
            continue
        if prev_ratio < 0.50 or row_ratio > 0.25:
            continue
        if edge.get("prev_table_family_key") != edge.get("table_family_key") or not edge.get("table_family_key"):
            continue
        if edge.get("prev_page_kind") in {"context_or_blank", "offtarget_table", "derivative_selected_caste_race_tribe"}:
            continue
        if edge.get("page_kind") in {"context_or_blank", "offtarget_table", "derivative_selected_caste_race_tribe"}:
            continue
        if not has_age_civil_signal(edge):
            continue

        same_kind = edge.get("prev_page_kind") == edge.get("page_kind")
        strict_kind = same_kind and edge.get("page_kind") not in BAD_PHYSICAL_CANDIDATE_KINDS
        candidate_type = (
            "layout_suspect_physical_pair_review_only"
            if physical_pair_pattern
            else "strict_same_kind_family_watchlist"
            if strict_kind
            else "classifier_repair_row_label_pair_review"
        )
        left_ocr = str(left_member.get("ocr_artifact", ""))
        right_ocr = str(right_member.get("ocr_artifact", ""))
        blocker = physical_candidate_blocker(left_ocr, right_ocr, edge)
        confidence_tier = physical_candidate_tier(edge)
        candidate_id = stable_id(
            "physcand",
            edge.get("v5_source_key"),
            edge.get("prev_physical_page_id"),
            edge.get("physical_page_id"),
            edge.get("prev_pdf_page"),
            edge.get("pdf_page"),
        )
        review_flags = [
            "not_promoted",
            "pairwise_only",
            "row_label_dependency_candidate",
        ]
        if physical_pair_pattern:
            review_flags.extend(
                [
                    "stored_layout_signature",
                    "layout_boolean_can_be_false_positive",
                    "needs_visual_or_ocr_confirmation",
                ]
            )
        else:
            review_flags.append("watchlist_not_physical_candidate")
        if not strict_kind:
            review_flags.append("classifier_kind_switch_review")
        if blocker != "needs_row_assembly_constraint_test":
            review_flags.append(blocker)
        row = {
            "candidate_rank": 0,
            "candidate_id": candidate_id,
            "confidence_tier": confidence_tier,
            "candidate_type": candidate_type,
            "safe_to_auto_merge_now": "no",
            "recommended_action": "Run/inspect row-assembly constraints for this adjacent pair; do not physically merge until the right page is confirmed to inherit row labels from the left page.",
            "promotion_condition": "Promote only if images/OCR show same table body, right page has no independent row labels, row counts align or have an explainable placeholder row, and column order is left page then right page.",
            "blocker_type": blocker,
            "page_ranges": f"{edge.get('prev_pdf_page')}-{edge.get('pdf_page')}",
            "v5_source_key": edge.get("v5_source_key", ""),
            "prev_pdf_page": edge.get("prev_pdf_page", ""),
            "pdf_page": edge.get("pdf_page", ""),
            "prev_pdf_page_parity": page_parity(edge.get("prev_pdf_page")),
            "pdf_page_parity": page_parity(edge.get("pdf_page")),
            "odd_unlabeled_right_page_signal": "yes" if page_parity(edge.get("pdf_page")) == "odd" else "no",
            "page_gap": edge.get("page_gap", ""),
            "prev_physical_page_id": edge.get("prev_physical_page_id", ""),
            "physical_page_id": edge.get("physical_page_id", ""),
            "prev_page_kind": edge.get("prev_page_kind", ""),
            "page_kind": edge.get("page_kind", ""),
            "prev_table_family_key": edge.get("prev_table_family_key", ""),
            "table_family_key": edge.get("table_family_key", ""),
            "prev_table_number_key": edge.get("prev_table_number_key", ""),
            "table_number_key": edge.get("table_number_key", ""),
            "title_similarity": edge.get("title_similarity", ""),
            "core_table_vii_signal": "yes" if has_table_vii_signal(edge) else "no",
            "age_civil_signal": "yes",
            "row_label_dependency_signal": "left_labelled_right_unlabelled",
            "physical_evidence_status": (
                "stored_layout_signal_only_not_confirmed"
                if physical_pair_pattern
                else "not_a_physical_layout_candidate"
            ),
            "prev_row_label_ratio": edge.get("prev_row_label_ratio", ""),
            "row_label_ratio": edge.get("row_label_ratio", ""),
            "left_ocr_artifact": left_ocr,
            "right_ocr_artifact": right_ocr,
            "left_layout_left_stub_candidate": left_stub,
            "left_layout_right_half_candidate": left_half,
            "right_layout_left_stub_candidate": right_stub,
            "right_layout_right_half_candidate": right_half,
            "physical_pair_pattern": "yes" if physical_pair_pattern else "no",
            "prev_first_row_labels": edge.get("prev_first_row_labels", ""),
            "first_row_labels": edge.get("first_row_labels", ""),
            "prev_last_row_labels": edge.get("prev_last_row_labels", ""),
            "last_row_labels": edge.get("last_row_labels", ""),
            "prev_title": edge.get("prev_title", ""),
            "title": edge.get("title", ""),
            "existing_v25_decision": edge.get("v25_decision", ""),
            "existing_v25_decision_layer": edge.get("v25_decision_layer", ""),
            "existing_v25_reason": edge.get("v25_reason", ""),
            "existing_v25_review_flags": edge.get("v25_review_flags", ""),
            "candidate_reason": "adjacent pages; same table family; left page has row labels; right page lacks row labels; age/civil table signal present; stored layout flags are review evidence only",
            "candidate_review_flags": "; ".join(review_flags),
            "prev_image_path": edge.get("prev_image_path", ""),
            "image_path": edge.get("image_path", ""),
        }
        watchlist_rows.append(row)
        if physical_pair_pattern:
            physical_rows.append(row)

    scan_rows.sort(key=row_label_scan_rank_key)
    for idx, row in enumerate(scan_rows, start=1):
        row["scan_rank"] = idx
    priority_scan_rows = [
        dict(row)
        for row in scan_rows
        if row.get("row_label_pattern") == "left_labelled_right_unlabelled"
        and row.get("candidate_strength") != "excluded_context_or_derivative_review_only"
        and row.get("existing_v25_decision") != "seed_physical_join"
    ]
    for idx, row in enumerate(priority_scan_rows, start=1):
        row["scan_rank"] = idx
    odd_unlabeled_priority_rows = [
        dict(row)
        for row in priority_scan_rows
        if row.get("odd_unlabeled_right_page_signal") == "yes"
    ]
    for idx, row in enumerate(odd_unlabeled_priority_rows, start=1):
        row["scan_rank"] = idx
    watchlist_rows.sort(key=candidate_rank_key)
    for idx, row in enumerate(watchlist_rows, start=1):
        row["candidate_rank"] = idx
    physical_rows.sort(key=candidate_rank_key)
    for idx, row in enumerate(physical_rows, start=1):
        row["candidate_rank"] = idx

    summary_rows: list[dict[str, object]] = []
    for dataset_name, source_rows in [
        ("potential_physical", physical_rows),
        ("row_label_watchlist", watchlist_rows),
        ("row_label_dependency_scan", scan_rows),
        ("row_label_dependency_priority_scan", priority_scan_rows),
        ("row_label_odd_unlabeled_priority", odd_unlabeled_priority_rows),
    ]:
        summary_rows.append({"summary_type": "dataset_rows", "name": dataset_name, "count": len(source_rows)})
        for field in [
            "confidence_tier",
            "candidate_strength",
            "candidate_type",
            "blocker_type",
            "row_label_pattern",
            "core_table_vii_signal",
            "physical_pair_pattern",
            "physical_evidence_status",
            "same_table_family",
            "same_table_number",
            "page_kind_blocker",
        ]:
            for name, count in sorted(Counter(str(row.get(field, "")) for row in source_rows).items()):
                summary_rows.append({"summary_type": f"{dataset_name}:{field}", "name": name, "count": count})
    write_csv(OUT_DIR / "row_label_dependency_scan.csv", scan_rows, ROW_LABEL_SCAN_FIELDS)
    write_csv(OUT_DIR / "row_label_dependency_priority_scan.csv", priority_scan_rows, ROW_LABEL_SCAN_FIELDS)
    write_csv(OUT_DIR / "row_label_odd_unlabeled_priority.csv", odd_unlabeled_priority_rows, ROW_LABEL_SCAN_FIELDS)
    write_csv(OUT_DIR / "potential_physical_join_candidates.csv", physical_rows, POTENTIAL_PHYSICAL_FIELDS)
    write_csv(OUT_DIR / "layout_suspect_physical_join_candidates.csv", physical_rows, POTENTIAL_PHYSICAL_FIELDS)
    write_csv(OUT_DIR / "row_label_dependency_watchlist.csv", watchlist_rows, POTENTIAL_PHYSICAL_FIELDS)
    write_csv(OUT_DIR / "potential_physical_join_summary.csv", summary_rows, ["summary_type", "name", "count"])
    return {
        "row_label_dependency_scan": scan_rows,
        "row_label_dependency_priority_scan": priority_scan_rows,
        "row_label_odd_unlabeled_priority": odd_unlabeled_priority_rows,
        "potential_physical_candidates": physical_rows,
        "row_label_dependency_watchlist": watchlist_rows,
        "potential_physical_summary": summary_rows,
    }


def build_physical_grouping_outputs() -> dict[str, list[dict[str, object]]]:
    attempts = read_csv(OUT_DIR / "row_assembly_attempts.csv")
    rows: list[dict[str, object]] = []
    for attempt in attempts:
        decision = physical_grouping_decision(attempt)
        page_gap = page_pair_gap(attempt.get("page_pair"))
        row = {
            "grouping_rank": 0,
            **attempt,
            **decision,
            "page_gap": page_gap if page_gap is not None else "",
        }
        rows.append(row)
    rows.sort(key=physical_grouping_rank_key)
    for idx, row in enumerate(rows, start=1):
        row["grouping_rank"] = idx

    confirmed = [row for row in rows if row.get("grouping_decision") == "confirmed_pairwise_physical_grouping"]
    partial = [row for row in rows if row.get("grouping_decision") == "partial_pairwise_physical_grouping_review"]
    review = [row for row in rows if row.get("safe_to_group_pages") != "yes"]
    summary_rows: list[dict[str, object]] = []
    for dataset_name, source_rows in [
        ("all_physical_grouping_decisions", rows),
        ("confirmed_physical_groupings", confirmed),
        ("partial_physical_groupings", partial),
        ("physical_grouping_review", review),
    ]:
        summary_rows.append({"summary_type": "dataset_rows", "name": dataset_name, "count": len(source_rows)})
        for field in [
            "grouping_decision",
            "grouping_confidence",
            "safe_to_group_pages",
            "safe_to_publish_values",
            "value_review_required",
            "structural_status",
            "value_constraint_status",
        ]:
            for name, count in sorted(Counter(str(row.get(field, "")) for row in source_rows).items()):
                summary_rows.append({"summary_type": f"{dataset_name}:{field}", "name": name, "count": count})

    write_csv(OUT_DIR / "physical_grouping_decisions.csv", rows, PHYSICAL_GROUPING_FIELDS)
    write_csv(OUT_DIR / "confirmed_physical_groupings.csv", confirmed, PHYSICAL_GROUPING_FIELDS)
    write_csv(OUT_DIR / "partial_physical_groupings.csv", partial, PHYSICAL_GROUPING_FIELDS)
    write_csv(OUT_DIR / "physical_grouping_review_queue.csv", review, PHYSICAL_GROUPING_FIELDS)
    write_csv(OUT_DIR / "physical_grouping_summary.csv", summary_rows, ["summary_type", "name", "count"])
    return {
        "physical_grouping_decisions": rows,
        "confirmed_physical_groupings": confirmed,
        "partial_physical_groupings": partial,
        "physical_grouping_review": review,
        "physical_grouping_summary": summary_rows,
    }


def local_path_exists(value: object) -> bool:
    path_text = str(value or "").strip()
    if not path_text:
        return False
    path = Path(path_text)
    if not path.is_absolute():
        path = ROOT / path
    return path.exists()


def image_source_bucket(path_text: object) -> str:
    path = str(path_text or "")
    if "core_money_renders" in path:
        return "core_money_render"
    if "final_tableVII" in path:
        return "final_tableVII_render"
    if "never_ocrd_images" in path:
        return "legacy_never_ocrd_image"
    if "digitized_tables_referenced" in path:
        return "legacy_referenced_digitized_table"
    if "digitized_tables" in path:
        return "legacy_digitized_table"
    if path:
        return "other_image_path"
    return "missing_image_path"


def expected_artifact_globs(row: dict[str, object]) -> tuple[str, str]:
    source_id = str(row.get("source_pdf_id") or "").strip()
    page = str(row.get("pdf_page") or "").strip()
    image_path = Path(str(row.get("image_path") or ""))
    image_stem = image_path.stem
    source_globs: list[str] = []
    hash_globs: list[str] = []
    if source_id and page:
        source_globs.extend(
            [
                f"{source_id}_p{page}_*.json",
                f"{source_id}_p{page}.json",
                f"{source_id}_p{page}_*.md",
                f"{source_id}_p{page}.md",
            ]
        )
    if image_stem:
        hash_globs.extend(
            [
                f"{image_stem}.mistral.md",
                f"{image_stem}.md",
                f"{image_stem}.json",
                f"{image_stem}_*.json",
            ]
        )
    return "; ".join(source_globs), "; ".join(hash_globs)


def missing_artifact_rank_key(row: dict[str, object]) -> tuple[int, str, int, int, str]:
    priority = {
        "blocks_pairwise_physical_grouping": 1,
        "unresolved_target_for_review_pair": 2,
        "ambiguous_artifact_match": 3,
        "other_missing_or_unresolved_artifact": 4,
    }.get(str(row.get("priority_bucket", "")), 9)
    return (
        priority,
        str(row.get("source_pdf") or row.get("v5_source_key") or ""),
        int_or_none(row.get("pdf_page")) or 10**9,
        int_or_none(row.get("position_in_component")) or 10**9,
        str(row.get("physical_page_id") or ""),
    )


def build_missing_artifact_targets(
    artifact_resolution: list[dict[str, object]],
    physical_groupings: list[dict[str, object]],
) -> dict[str, list[dict[str, object]]]:
    grouping_by_component = {
        str(row.get("v25_component_id") or ""): row
        for row in physical_groupings
        if row.get("v25_component_id")
    }
    rows: list[dict[str, object]] = []
    resolver_drop_roots = "; ".join(
        [
            "export_final/gemini_no_gap_ocr_2026-06-16_v2_best",
            "export_final/gemini_no_gap_ocr_2026-06-16_v2",
            "export_final/gemini_no_gap_ocr_2026-06-16",
            "export/money_gapfill_ocr",
            "export_final/sendout_final/sendout_v1/mistral_md",
        ]
    )
    for artifact in artifact_resolution:
        status = str(artifact.get("resolution_status") or "")
        if status in {"existing_artifact", "resolved"}:
            continue
        component_id = str(artifact.get("v25_component_id") or "")
        grouping = grouping_by_component.get(component_id, {})
        source_globs, hash_globs = expected_artifact_globs(artifact)
        grouping_decision = str(grouping.get("grouping_decision") or "")
        if status.startswith("ambiguous"):
            priority_bucket = "ambiguous_artifact_match"
            action = "Resolve duplicate OCR artifact candidates manually, then rerun assembly."
        elif grouping_decision == "blocked_missing_ocr_artifact":
            priority_bucket = "blocks_pairwise_physical_grouping"
            action = "OCR or attach this existing image artifact, then rerun first-principles assembly."
        elif grouping_decision:
            priority_bucket = "unresolved_target_for_review_pair"
            action = "Attach OCR artifact if this reviewed pair should be retested."
        else:
            priority_bucket = "other_missing_or_unresolved_artifact"
            action = "Attach OCR artifact if this page remains relevant after review."
        page = str(artifact.get("pdf_page") or "").strip()
        source_id = str(artifact.get("source_pdf_id") or artifact.get("source_pdf") or artifact.get("v5_source_key") or "").strip()
        target_key = f"{source_id}_p{page}" if source_id and page else str(artifact.get("physical_page_id") or "")
        rows.append(
            {
                "missing_target_rank": 0,
                "target_key": target_key,
                "priority_bucket": priority_bucket,
                "blocking_context": grouping_decision or "not_in_physical_grouping_decisions",
                "recommended_action": action,
                "promotion_condition": "Promote only after both pages in the pair have OCR artifacts and row-label inheritance/column-order constraints pass.",
                "v25_component_id": component_id,
                "position_in_component": artifact.get("position_in_component", ""),
                "physical_page_id": artifact.get("physical_page_id", ""),
                "v5_source_key": artifact.get("v5_source_key", ""),
                "source_pdf": artifact.get("source_pdf", ""),
                "source_pdf_id": artifact.get("source_pdf_id", ""),
                "pdf_page": artifact.get("pdf_page", ""),
                "page_pair": grouping.get("page_pair", ""),
                "pair_gap": grouping.get("page_gap", ""),
                "grouping_decision": grouping_decision,
                "grouping_reason": grouping.get("grouping_reason", ""),
                "grouping_review_flags": grouping.get("grouping_review_flags", ""),
                "image_path": artifact.get("image_path", ""),
                "image_path_exists": "yes" if local_path_exists(artifact.get("image_path")) else "no",
                "image_source_bucket": image_source_bucket(artifact.get("image_path")),
                "existing_ocr_artifact": artifact.get("existing_ocr_artifact", ""),
                "resolved_ocr_artifact": artifact.get("resolved_ocr_artifact", ""),
                "resolution_status": status,
                "resolution_method": artifact.get("resolution_method", ""),
                "candidate_count": artifact.get("candidate_count", ""),
                "candidate_paths": artifact.get("candidate_paths", ""),
                "expected_source_page_globs": source_globs,
                "expected_image_hash_globs": hash_globs,
                "resolver_drop_roots": resolver_drop_roots,
            }
        )
    rows.sort(key=missing_artifact_rank_key)
    for idx, row in enumerate(rows, start=1):
        row["missing_target_rank"] = idx
    write_csv(OUT_DIR / "missing_artifact_targets.csv", rows, MISSING_ARTIFACT_FIELDS)
    return {"missing_artifact_targets": rows}


def build_candidate_missing_artifact_targets(
    candidate_artifact_resolution: list[dict[str, object]],
    candidate_attempts: list[dict[str, object]],
) -> dict[str, list[dict[str, object]]]:
    attempts_by_component = {
        str(row.get("v25_component_id") or ""): row
        for row in candidate_attempts
        if row.get("v25_component_id")
    }
    rows: list[dict[str, object]] = []
    resolver_drop_roots = "; ".join(
        [
            "export_final/gemini_no_gap_ocr_2026-06-16_v2_best",
            "export_final/gemini_no_gap_ocr_2026-06-16_v2",
            "export_final/gemini_no_gap_ocr_2026-06-16",
            "export/money_gapfill_ocr",
            "export_final/sendout_final/sendout_v1/mistral_md",
        ]
    )
    for artifact in candidate_artifact_resolution:
        status = str(artifact.get("resolution_status") or "")
        if status in {"existing_artifact", "resolved"}:
            continue
        component_id = str(artifact.get("v25_component_id") or "")
        attempt = attempts_by_component.get(component_id, {})
        source_globs, hash_globs = expected_artifact_globs(artifact)
        if status.startswith("ambiguous"):
            priority_bucket = "candidate_ambiguous_artifact_match"
            action = "Resolve duplicate candidate OCR artifacts manually, then rerun candidate assembly tests."
        else:
            priority_bucket = "candidate_row_label_dependency_missing_ocr"
            action = "OCR or attach this existing image artifact if this row-label dependency candidate remains relevant."
        page = str(artifact.get("pdf_page") or "").strip()
        source_id = str(artifact.get("source_pdf_id") or artifact.get("source_pdf") or artifact.get("v5_source_key") or "").strip()
        target_key = f"{source_id}_p{page}" if source_id and page else str(artifact.get("physical_page_id") or "")
        rows.append(
            {
                "missing_target_rank": 0,
                "target_key": target_key,
                "priority_bucket": priority_bucket,
                "blocking_context": attempt.get("v26_status", "candidate_not_tested_without_ocr"),
                "recommended_action": action,
                "promotion_condition": "Candidate remains review-only; promote only after OCR exists, row-label inheritance passes, and visual/table-number context agrees.",
                "v25_component_id": component_id,
                "position_in_component": artifact.get("position_in_component", ""),
                "physical_page_id": artifact.get("physical_page_id", ""),
                "v5_source_key": artifact.get("v5_source_key", ""),
                "source_pdf": artifact.get("source_pdf", ""),
                "source_pdf_id": artifact.get("source_pdf_id", ""),
                "pdf_page": artifact.get("pdf_page", ""),
                "page_pair": attempt.get("page_pair", ""),
                "pair_gap": attempt.get("page_gap", ""),
                "grouping_decision": "candidate_row_label_dependency_review",
                "grouping_reason": attempt.get("grouping_reason", attempt.get("candidate_scope", "")),
                "grouping_review_flags": attempt.get("grouping_review_flags", attempt.get("constraint_flags", "")),
                "image_path": artifact.get("image_path", ""),
                "image_path_exists": "yes" if local_path_exists(artifact.get("image_path")) else "no",
                "image_source_bucket": image_source_bucket(artifact.get("image_path")),
                "existing_ocr_artifact": artifact.get("existing_ocr_artifact", ""),
                "resolved_ocr_artifact": artifact.get("resolved_ocr_artifact", ""),
                "resolution_status": status,
                "resolution_method": artifact.get("resolution_method", ""),
                "candidate_count": artifact.get("candidate_count", ""),
                "candidate_paths": artifact.get("candidate_paths", ""),
                "expected_source_page_globs": source_globs,
                "expected_image_hash_globs": hash_globs,
                "resolver_drop_roots": resolver_drop_roots,
            }
        )
    rows.sort(key=missing_artifact_rank_key)
    for idx, row in enumerate(rows, start=1):
        row["missing_target_rank"] = idx
    write_csv(OUT_DIR / "candidate_missing_artifact_targets.csv", rows, MISSING_ARTIFACT_FIELDS)
    return {"candidate_missing_artifact_targets": rows}


def split_page_pair(value: object) -> tuple[str, str]:
    parts = str(value or "").split("-", 1)
    if len(parts) != 2:
        return "", ""
    return parts[0].strip(), parts[1].strip()


def pair_key(source: object, left_page: object, right_page: object, left_image: object = "", right_image: object = "") -> tuple[str, str, str, str, str]:
    return (
        str(source or "").strip(),
        str(left_page or "").strip(),
        str(right_page or "").strip(),
        str(left_image or "").strip(),
        str(right_image or "").strip(),
    )


def source_page_key(source: object, left_page: object, right_page: object) -> tuple[str, str, str]:
    return (
        str(source or "").strip(),
        str(left_page or "").strip(),
        str(right_page or "").strip(),
    )


def build_physical_grouping_lookup(rows: list[dict[str, object]]) -> dict[tuple[str, str, str, str, str], dict[str, object]]:
    lookup: dict[tuple[str, str, str, str, str], dict[str, object]] = {}
    for row in rows:
        left_page, right_page = split_page_pair(row.get("page_pair"))
        source = row.get("v5_source_key") or row.get("source_pdf")
        key = pair_key(source, left_page, right_page, row.get("left_image_path"), row.get("right_image_path"))
        if key[0] and key[1] and key[2]:
            lookup[key] = row
    return lookup


def build_watchlist_lookup(rows: list[dict[str, object]]) -> dict[tuple[str, str, str, str, str], dict[str, object]]:
    lookup: dict[tuple[str, str, str, str, str], dict[str, object]] = {}
    for row in rows:
        key = pair_key(
            row.get("v5_source_key"),
            row.get("prev_pdf_page"),
            row.get("pdf_page"),
            row.get("prev_image_path"),
            row.get("image_path"),
        )
        if key[0] and key[1] and key[2]:
            lookup[key] = row
    return lookup


def build_row_label_scan_lookup(rows: list[dict[str, object]]) -> dict[tuple[str, str, str, str, str], dict[str, object]]:
    lookup: dict[tuple[str, str, str, str, str], dict[str, object]] = {}
    for row in rows:
        key = pair_key(
            row.get("v5_source_key"),
            row.get("prev_pdf_page"),
            row.get("pdf_page"),
            row.get("prev_image_path"),
            row.get("image_path"),
        )
        if key[0] and key[1] and key[2]:
            lookup[key] = row
    return lookup


def logical_status_priority(row: dict[str, object]) -> int:
    decision = str(row.get("logical_grouping_decision") or "")
    if decision == "confirmed_bounded_logical_collection":
        return 1
    if decision == "review_large_logical_collection":
        return 2
    if decision == "review_same_table_logical_candidate":
        return 3
    return 9


def build_logical_pair_lookup(rows: list[dict[str, object]]) -> dict[tuple[str, str, str], dict[str, object]]:
    lookup: dict[tuple[str, str, str], dict[str, object]] = {}
    for row in rows:
        source = row.get("v5_source_key") or row.get("source_pdf")
        pages = parse_page_tokens(row.get("page_ranges", ""))
        if len(pages) < 2:
            continue
        for left, right in zip(pages, pages[1:]):
            if right - left != 1:
                continue
            key = source_page_key(source, left, right)
            existing = lookup.get(key)
            if existing is None or logical_status_priority(row) < logical_status_priority(existing):
                lookup[key] = row
    return lookup


def adjacency_rank_key(row: dict[str, object]) -> tuple[int, str, int, str]:
    priority = {
        "confirmed_physical_pair": 1,
        "blocked_physical_missing_ocr_artifact": 2,
        "partial_physical_pair_review": 3,
        "failed_physical_constraints_review": 4,
        "row_label_dependency_review_candidate": 5,
        "logical_collection_only_confirmed": 6,
        "logical_collection_only_review": 7,
        "missing_intermediate_pages_candidate": 8,
        "hard_break_or_not_a_grouping_candidate": 9,
    }.get(str(row.get("final_decision_layer", "")), 99)
    return (
        priority,
        str(row.get("v5_source_key") or ""),
        int_or_none(row.get("prev_pdf_page")) or 10**9,
        str(row.get("adjacency_key") or ""),
    )


def adjacency_decision_for(
    edge: dict[str, object],
    physical: dict[str, object] | None,
    watch: dict[str, object] | None,
    logical: dict[str, object] | None,
) -> dict[str, str]:
    v25_decision = str(edge.get("v25_decision") or "")
    if physical:
        grouping_decision = str(physical.get("grouping_decision") or "")
        if grouping_decision == "confirmed_pairwise_physical_grouping":
            return {
                "final_decision_layer": "confirmed_physical_pair",
                "safe_to_physical_merge": "yes",
                "safe_to_logical_group": "yes",
                "recommended_action": physical.get("recommended_grouping_action", ""),
                "promotion_condition": "Already structurally grouped; publish numeric values only if safe_to_publish_values=yes or value review accepts the pair.",
                "decision_reason": physical.get("grouping_reason", ""),
                "decision_review_flags": physical.get("grouping_review_flags", ""),
            }
        if grouping_decision == "blocked_missing_ocr_artifact":
            return {
                "final_decision_layer": "blocked_physical_missing_ocr_artifact",
                "safe_to_physical_merge": "no",
                "safe_to_logical_group": "no",
                "recommended_action": physical.get("recommended_grouping_action", ""),
                "promotion_condition": "Attach OCR artifacts for both adjacent images, then require row-label inheritance and column-order constraints to pass.",
                "decision_reason": physical.get("grouping_reason", ""),
                "decision_review_flags": physical.get("grouping_review_flags", ""),
            }
        if grouping_decision == "partial_pairwise_physical_grouping_review":
            return {
                "final_decision_layer": "partial_physical_pair_review",
                "safe_to_physical_merge": "no",
                "safe_to_logical_group": "no",
                "recommended_action": physical.get("recommended_grouping_action", ""),
                "promotion_condition": "Use only manually confirmed passing blocks or rerun after better OCR/parsing.",
                "decision_reason": physical.get("grouping_reason", ""),
                "decision_review_flags": physical.get("grouping_review_flags", ""),
            }
        return {
            "final_decision_layer": "failed_physical_constraints_review",
            "safe_to_physical_merge": "no",
            "safe_to_logical_group": "no",
            "recommended_action": physical.get("recommended_grouping_action", ""),
            "promotion_condition": "Do not merge unless stronger OCR/parser evidence satisfies row/block constraints.",
            "decision_reason": physical.get("grouping_reason", ""),
            "decision_review_flags": physical.get("grouping_review_flags", ""),
        }

    if v25_decision == "seed_physical_join":
        return {
            "final_decision_layer": "failed_physical_constraints_review",
            "safe_to_physical_merge": "no",
            "safe_to_logical_group": "no",
            "recommended_action": "Investigate why a seed physical edge has no row-assembly decision.",
            "promotion_condition": "Must appear in physical_grouping_decisions and pass constraints before use.",
            "decision_reason": "seed physical edge lacks downstream physical grouping record",
            "decision_review_flags": "missing_physical_grouping_record",
        }

    if watch:
        return {
            "final_decision_layer": "row_label_dependency_review_candidate",
            "safe_to_physical_merge": "no",
            "safe_to_logical_group": "no",
            "recommended_action": watch.get("recommended_action", ""),
            "promotion_condition": watch.get("promotion_condition", ""),
            "decision_reason": watch.get("candidate_reason", ""),
            "decision_review_flags": watch.get("candidate_review_flags", ""),
        }

    if logical:
        confirmed = logical.get("logical_grouping_decision") == "confirmed_bounded_logical_collection"
        return {
            "final_decision_layer": "logical_collection_only_confirmed" if confirmed else "logical_collection_only_review",
            "safe_to_physical_merge": "no",
            "safe_to_logical_group": "yes" if confirmed else "no",
            "recommended_action": logical.get("recommended_logical_action", ""),
            "promotion_condition": "Use only as a bounded logical collection; never as a physical row merge.",
            "decision_reason": logical.get("logical_grouping_reason", ""),
            "decision_review_flags": logical.get("logical_grouping_review_flags", ""),
        }

    if v25_decision == "missing_bridge_candidate":
        return {
            "final_decision_layer": "missing_intermediate_pages_candidate",
            "safe_to_physical_merge": "no",
            "safe_to_logical_group": "no",
            "recommended_action": "Render/OCR missing intermediate pages before any grouping.",
            "promotion_condition": "Promote only after the complete page sequence is consecutive and compatible.",
            "decision_reason": edge.get("v25_reason", ""),
            "decision_review_flags": edge.get("v25_review_flags", ""),
        }

    if v25_decision == "logical_growth_candidate":
        return {
            "final_decision_layer": "logical_collection_only_review",
            "safe_to_physical_merge": "no",
            "safe_to_logical_group": "no",
            "recommended_action": "Review as a possible bounded logical collection only.",
            "promotion_condition": "Promote only if bounded consecutive sibling/continuation structure is explicit.",
            "decision_reason": edge.get("v25_reason", ""),
            "decision_review_flags": edge.get("v25_review_flags", ""),
        }

    if v25_decision == "growth_review_candidate":
        return {
            "final_decision_layer": "failed_physical_constraints_review",
            "safe_to_physical_merge": "no",
            "safe_to_logical_group": "no",
            "recommended_action": "Manual review or better OCR/parser evidence before considering physical grouping.",
            "promotion_condition": "Promote only if row-label inheritance/column-order constraints pass.",
            "decision_reason": edge.get("v25_reason", ""),
            "decision_review_flags": edge.get("v25_review_flags", ""),
        }

    return {
        "final_decision_layer": "hard_break_or_not_a_grouping_candidate",
        "safe_to_physical_merge": "no",
        "safe_to_logical_group": "no",
        "recommended_action": "Do not group.",
        "promotion_condition": "Only revisit if source classification/provenance changes.",
        "decision_reason": edge.get("v25_reason", ""),
        "decision_review_flags": edge.get("v25_review_flags", ""),
    }


def build_adjacency_decision_outputs(
    physical_groupings: list[dict[str, object]],
    watchlist_rows: list[dict[str, object]],
    row_label_scan_rows: list[dict[str, object]],
    row_label_priority_scan_rows: list[dict[str, object]],
    logical_groupings: list[dict[str, object]],
) -> dict[str, list[dict[str, object]]]:
    seed_edges = read_csv(OUT_DIR / "seed_pairwise_edges.csv")
    physical_lookup = build_physical_grouping_lookup(physical_groupings)
    watch_lookup = build_watchlist_lookup(watchlist_rows)
    row_label_scan_lookup = build_row_label_scan_lookup(row_label_scan_rows)
    row_label_priority_lookup = build_row_label_scan_lookup(row_label_priority_scan_rows)
    logical_lookup = build_logical_pair_lookup(logical_groupings)
    rows: list[dict[str, object]] = []

    for edge in seed_edges:
        include = (
            (edge.get("page_gap") == "1" and not edge.get("missing_pages"))
            or edge.get("v25_decision") == "missing_bridge_candidate"
        )
        if not include:
            continue
        source = edge.get("v5_source_key")
        key = pair_key(
            source,
            edge.get("prev_pdf_page"),
            edge.get("pdf_page"),
            edge.get("prev_image_path"),
            edge.get("image_path"),
        )
        source_key = source_page_key(source, edge.get("prev_pdf_page"), edge.get("pdf_page"))
        physical = physical_lookup.get(key)
        watch = watch_lookup.get(key)
        row_label_scan = row_label_scan_lookup.get(key)
        row_label_priority = row_label_priority_lookup.get(key)
        logical = logical_lookup.get(source_key)
        decision = adjacency_decision_for(edge, physical, watch, logical)
        adjacency_key = stable_id(
            "adj",
            source,
            edge.get("prev_physical_page_id"),
            edge.get("physical_page_id"),
            edge.get("prev_pdf_page"),
            edge.get("pdf_page"),
        )
        coverage_scope = "consecutive_pdf_pages" if edge.get("page_gap") == "1" and not edge.get("missing_pages") else "missing_page_bridge"
        row = {
            "adjacency_rank": 0,
            "adjacency_key": adjacency_key,
            "coverage_scope": coverage_scope,
            **decision,
            "physical_grouping_decision": physical.get("grouping_decision", "") if physical else "",
            "physical_grouping_confidence": physical.get("grouping_confidence", "") if physical else "",
            "physical_safe_to_group_pages": physical.get("safe_to_group_pages", "") if physical else "",
            "physical_safe_to_publish_values": physical.get("safe_to_publish_values", "") if physical else "",
            "logical_grouping_decision": logical.get("logical_grouping_decision", "") if logical else "",
            "logical_grouping_confidence": logical.get("logical_grouping_confidence", "") if logical else "",
            "logical_safe_to_use_collection": logical.get("safe_to_use_logical_collection", "") if logical else "",
            "row_label_watchlist_status": "present" if watch else "",
            "row_label_watchlist_tier": watch.get("confidence_tier", "") if watch else "",
            "row_label_watchlist_type": watch.get("candidate_type", "") if watch else "",
            "row_label_watchlist_blocker": watch.get("blocker_type", "") if watch else "",
            "row_label_scan_status": "present" if row_label_scan else "",
            "row_label_scan_strength": row_label_scan.get("candidate_strength", "") if row_label_scan else "",
            "row_label_scan_pattern": row_label_scan.get("row_label_pattern", "") if row_label_scan else "",
            "row_label_scan_direction": row_label_scan.get("directional_interpretation", "") if row_label_scan else "",
            "row_label_priority_scan_status": "present" if row_label_priority else "",
            "row_label_scan_review_flags": row_label_scan.get("scan_review_flags", "") if row_label_scan else "",
            **{field: edge.get(field, "") for field in ADJACENCY_DECISION_FIELDS if field in edge},
        }
        rows.append(row)

    rows.sort(key=adjacency_rank_key)
    for idx, row in enumerate(rows, start=1):
        row["adjacency_rank"] = idx

    summary_rows: list[dict[str, object]] = []
    summary_rows.append({"summary_type": "dataset_rows", "name": "adjacency_decision_index", "count": len(rows)})
    for field in [
        "coverage_scope",
        "final_decision_layer",
        "safe_to_physical_merge",
        "safe_to_logical_group",
        "v25_decision",
        "v25_decision_layer",
        "row_label_watchlist_status",
        "physical_grouping_decision",
        "logical_grouping_decision",
    ]:
        for name, count in sorted(Counter(str(row.get(field, "")) for row in rows).items()):
            summary_rows.append({"summary_type": f"adjacency:{field}", "name": name, "count": count})

    write_csv(OUT_DIR / "adjacency_decision_index.csv", rows, ADJACENCY_DECISION_FIELDS)
    write_csv(OUT_DIR / "adjacency_decision_summary.csv", summary_rows, ["summary_type", "name", "count"])
    return {
        "adjacency_decision_index": rows,
        "adjacency_decision_summary": summary_rows,
    }


def candidate_attempt_adjacency_key(row: dict[str, object]) -> tuple[str, str, str, str, str]:
    left_page, right_page = split_page_pair(row.get("page_pair"))
    return pair_key(
        row.get("v5_source_key"),
        left_page,
        right_page,
        row.get("left_image_path"),
        row.get("right_image_path"),
    )


def candidate_pass_risk_level(flags: list[str]) -> str:
    high = {
        "table_number_switch_or_missing",
        "table_family_switch_or_missing",
        "not_priority_scan_candidate",
        "not_row_label_dependency_adjacency",
    }
    if any(flag in high for flag in flags):
        return "high_review_risk"
    if flags:
        return "medium_review_risk"
    return "lower_review_risk"


def build_candidate_pass_context_outputs(
    candidate_attempts: list[dict[str, object]],
    adjacency_index: list[dict[str, object]],
) -> dict[str, list[dict[str, object]]]:
    adjacency_by_key = {
        pair_key(
            row.get("v5_source_key"),
            row.get("prev_pdf_page"),
            row.get("pdf_page"),
            row.get("prev_image_path"),
            row.get("image_path"),
        ): row
        for row in adjacency_index
    }
    rows: list[dict[str, object]] = []
    for attempt in candidate_attempts:
        if attempt.get("v26_status") not in {"assembled_pass", "assembled_complete_value_review"}:
            continue
        adjacency = adjacency_by_key.get(candidate_attempt_adjacency_key(attempt), {})
        same_family = (
            "yes"
            if adjacency.get("prev_table_family_key")
            and adjacency.get("prev_table_family_key") == adjacency.get("table_family_key")
            else "no"
        )
        same_number = (
            "yes"
            if adjacency.get("prev_table_number_key")
            and adjacency.get("prev_table_number_key") == adjacency.get("table_number_key")
            else "no"
        )
        flags = ["manual_review_required", "structural_pass_only_not_promoted"]
        if same_family != "yes":
            flags.append("table_family_switch_or_missing")
        if same_number != "yes":
            flags.append("table_number_switch_or_missing")
        if float_or_zero(adjacency.get("title_similarity")) < 0.55:
            flags.append("title_similarity_below_0.55")
        if adjacency.get("row_label_priority_scan_status") != "present":
            flags.append("not_priority_scan_candidate")
        if adjacency.get("final_decision_layer") != "row_label_dependency_review_candidate":
            flags.append("not_row_label_dependency_adjacency")
        if attempt.get("value_constraint_status") not in {"pass", "not_applicable"}:
            flags.append("value_constraint_not_clean")
        rows.append(
            {
                "review_rank": 0,
                "v26_attempt_id": attempt.get("v26_attempt_id", ""),
                "source_v25_edge_id": attempt.get("source_v25_edge_id", ""),
                "v5_source_key": attempt.get("v5_source_key", ""),
                "source_pdf": attempt.get("source_pdf", ""),
                "source_pdf_id": attempt.get("source_pdf_id", ""),
                "page_pair": attempt.get("page_pair", ""),
                "candidate_status": attempt.get("v26_status", ""),
                "safe_to_auto_promote": "no",
                "risk_level": candidate_pass_risk_level(flags),
                "recommended_action": "Manual visual/table-continuity review before promotion; do not treat structural row alignment alone as proof of a physical assembly.",
                "promotion_condition": "Promote only if the two consecutive pages are the same table body, table number/subject continuity is explainable, row labels truly inherit left-to-right, and column order remains image-faithful.",
                "risk_flags": "; ".join(flags),
                "structural_status": attempt.get("structural_status", ""),
                "value_constraint_status": attempt.get("value_constraint_status", ""),
                "left_row_count_total": attempt.get("left_row_count_total", ""),
                "right_row_count_total": attempt.get("right_row_count_total", ""),
                "assembled_row_count": attempt.get("assembled_row_count", ""),
                "adjacency_final_decision_layer": adjacency.get("final_decision_layer", ""),
                "row_label_scan_strength": adjacency.get("row_label_scan_strength", ""),
                "row_label_scan_pattern": adjacency.get("row_label_scan_pattern", ""),
                "row_label_priority_scan_status": adjacency.get("row_label_priority_scan_status", ""),
                "prev_table_family_key": adjacency.get("prev_table_family_key", ""),
                "table_family_key": adjacency.get("table_family_key", ""),
                "same_table_family": same_family,
                "prev_table_number_key": adjacency.get("prev_table_number_key", ""),
                "table_number_key": adjacency.get("table_number_key", ""),
                "same_table_number": same_number,
                "title_similarity": adjacency.get("title_similarity", ""),
                "core_table_vii_signal": adjacency.get("core_table_vii_signal", ""),
                "age_civil_signal": adjacency.get("age_civil_signal", ""),
                "v25_decision": adjacency.get("v25_decision", ""),
                "v25_reason": adjacency.get("v25_reason", ""),
                "left_title": attempt.get("left_title", ""),
                "right_title": attempt.get("right_title", ""),
                "left_ocr_artifact": attempt.get("left_ocr_artifact", ""),
                "right_ocr_artifact": attempt.get("right_ocr_artifact", ""),
                "left_image_path": attempt.get("left_image_path", ""),
                "right_image_path": attempt.get("right_image_path", ""),
            }
        )

    priority = {"high_review_risk": 1, "medium_review_risk": 2, "lower_review_risk": 3}
    rows.sort(
        key=lambda row: (
            priority.get(str(row.get("risk_level") or ""), 9),
            str(row.get("v5_source_key") or ""),
            int_or_none(str(row.get("page_pair") or "").split("-", 1)[0]) or 10**9,
            str(row.get("v26_attempt_id") or ""),
        )
    )
    for idx, row in enumerate(rows, start=1):
        row["review_rank"] = idx
    summary_rows: list[dict[str, object]] = [
        {"summary_type": "dataset_rows", "name": "candidate_pass_context_review", "count": len(rows)}
    ]
    for field in ["risk_level", "same_table_family", "same_table_number", "row_label_scan_strength"]:
        for name, count in sorted(Counter(str(row.get(field, "")) for row in rows).items()):
            summary_rows.append({"summary_type": f"candidate_pass_context:{field}", "name": name, "count": count})
    write_csv(OUT_DIR / "candidate_pass_context_review.csv", rows, CANDIDATE_PASS_CONTEXT_FIELDS)
    write_csv(OUT_DIR / "candidate_pass_context_summary.csv", summary_rows, ["summary_type", "name", "count"])
    return {
        "candidate_pass_context_review": rows,
        "candidate_pass_context_summary": summary_rows,
    }


def build_logical_grouping_outputs(logical_rows: list[dict[str, object]]) -> dict[str, list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    for logical in logical_rows:
        decision = logical_grouping_decision(logical)
        row = {
            "logical_rank": 0,
            **decision,
            **logical,
        }
        rows.append(row)
    rows.sort(key=logical_grouping_rank_key)
    for idx, row in enumerate(rows, start=1):
        row["logical_rank"] = idx

    confirmed = [row for row in rows if row.get("logical_grouping_decision") == "confirmed_bounded_logical_collection"]
    review = [row for row in rows if row.get("safe_to_use_logical_collection") != "yes"]
    summary_rows: list[dict[str, object]] = []
    for dataset_name, source_rows in [
        ("all_logical_grouping_decisions", rows),
        ("confirmed_logical_collections", confirmed),
        ("logical_grouping_review", review),
    ]:
        summary_rows.append({"summary_type": "dataset_rows", "name": dataset_name, "count": len(source_rows)})
        for field in [
            "logical_grouping_decision",
            "logical_grouping_confidence",
            "safe_to_use_logical_collection",
            "safe_to_physical_merge",
            "subgroup",
            "tier",
            "segment_kind",
        ]:
            for name, count in sorted(Counter(str(row.get(field, "")) for row in source_rows).items()):
                summary_rows.append({"summary_type": f"{dataset_name}:{field}", "name": name, "count": count})

    write_csv(OUT_DIR / "logical_grouping_decisions.csv", rows, LOGICAL_GROUPING_FIELDS)
    write_csv(OUT_DIR / "confirmed_logical_collections.csv", confirmed, LOGICAL_GROUPING_FIELDS)
    write_csv(OUT_DIR / "logical_grouping_review_queue.csv", review, LOGICAL_GROUPING_FIELDS)
    write_csv(OUT_DIR / "logical_grouping_summary.csv", summary_rows, ["summary_type", "name", "count"])
    return {
        "logical_grouping_decisions": rows,
        "confirmed_logical_collections": confirmed,
        "logical_grouping_review": review,
        "logical_grouping_summary": summary_rows,
    }


def split_semicolon_values(value: object) -> list[str]:
    values: list[str] = []
    for part in str(value or "").split(";"):
        text = compact_text(part)
        if not text or re.fullmatch(r"\.\.\.\(\+\d+\)", text):
            continue
        values.append(text)
    return values


def source_stage_lookup() -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    for path in [
        OUT_DIR / "source_stage_logical_operational.csv",
        OUT_DIR / "source_stage_operational_manifest.csv",
    ]:
        for row in read_csv(path):
            group_id = row.get("v28_group_id", "")
            if not group_id:
                continue
            existing = rows.get(group_id, {})
            merged = dict(existing)
            for key, value in row.items():
                if value or key not in merged:
                    merged[key] = value
            rows[group_id] = merged
    return rows


def refined_page_nodes_for_group(group_id: str) -> dict[int, dict[str, str]]:
    refined_groups, refined_edges = build_refined_group_lookup()
    group = refined_groups.get(group_id)
    if not group:
        return {}
    nodes: dict[int, dict[str, str]] = {}
    edge_ids = [edge_id.strip() for edge_id in group.get("source_edge_ids", "").split(";") if edge_id.strip()]
    for edge_id in edge_ids:
        edge = refined_edges.get(edge_id)
        if not edge:
            continue
        prev_page = int_or_none(edge.get("prev_pdf_page"))
        page = int_or_none(edge.get("pdf_page"))
        if prev_page is not None:
            nodes.setdefault(
                prev_page,
                {
                    "title": edge.get("prev_title", ""),
                    "variant_terms": edge.get("prev_variant_terms", ""),
                    "image_path": edge.get("prev_image_path", ""),
                    "variant_category_from_previous": "",
                    "source_edge_id_from_previous": "",
                },
            )
        if page is not None:
            nodes[page] = {
                "title": edge.get("title", ""),
                "variant_terms": edge.get("variant_terms", ""),
                "image_path": edge.get("image_path", ""),
                "variant_category_from_previous": edge.get("variant_category", ""),
                "source_edge_id_from_previous": edge.get("v29_edge_id", ""),
            }
    return nodes


def logical_child_lookup(child_rows: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    return {str(row.get("assembly_group_id") or ""): row for row in child_rows if row.get("assembly_group_id")}


def build_logical_collection_members(
    confirmed_logical: list[dict[str, object]],
    child_rows: list[dict[str, object]],
) -> dict[str, list[dict[str, object]]]:
    child_by_id = logical_child_lookup(child_rows)
    stage_by_id = source_stage_lookup()
    rows: list[dict[str, object]] = []

    for collection in confirmed_logical:
        group_id = str(collection.get("assembly_group_id") or "")
        pages = parse_page_tokens(collection.get("page_ranges", ""))
        if not group_id or not pages:
            continue
        source = collection.get("v5_source_key") or collection.get("source_pdf")
        page_nodes = refined_page_nodes_for_group(group_id)
        child = child_by_id.get(group_id, {})
        stage = stage_by_id.get(str(collection.get("source_stage_group_id") or ""), {})
        child_images = split_semicolon_values(child.get("image_paths", ""))
        stage_images = split_semicolon_values(stage.get("image_paths", ""))
        title_parts = split_semicolon_values(collection.get("primary_title", ""))

        for idx, page in enumerate(pages, start=1):
            node = page_nodes.get(page, {})
            member_source = "refined_sibling_edge_page_node" if node else ""
            image_path = node.get("image_path", "")
            title = node.get("title", "")
            variant_terms = node.get("variant_terms", "")
            variant_category = node.get("variant_category_from_previous", "")
            source_edge_id = node.get("source_edge_id_from_previous", "")
            if not image_path and idx <= len(child_images):
                image_path = child_images[idx - 1]
                member_source = member_source or "logical_child_segment_image_list"
            if not image_path and idx <= len(stage_images):
                image_path = stage_images[idx - 1]
                member_source = member_source or "source_stage_logical_image_list"
            if not title and idx <= len(title_parts):
                title = title_parts[idx - 1]
            if not variant_terms and collection.get("segment_variant"):
                variant_terms = str(collection.get("segment_variant") or "")
            if not member_source:
                member_source = "page_only_no_image_evidence"
            rows.append(
                {
                    "member_rank": 0,
                    "assembly_group_id": group_id,
                    "logical_grouping_confidence": collection.get("logical_grouping_confidence", ""),
                    "subgroup": collection.get("subgroup", ""),
                    "segment_kind": collection.get("segment_kind", ""),
                    "segment_variant": collection.get("segment_variant", ""),
                    "v5_source_key": source,
                    "source_pdf": collection.get("source_pdf", ""),
                    "source_pdf_id": collection.get("source_pdf_id", ""),
                    "collection_page_ranges": collection.get("page_ranges", ""),
                    "n_collection_pages": len(pages),
                    "member_position": idx,
                    "pdf_page": page,
                    "page_is_consecutive_with_previous": "yes" if idx == 1 or page == pages[idx - 2] + 1 else "no",
                    "title": title,
                    "variant_terms": variant_terms,
                    "variant_category_from_previous": variant_category,
                    "source_edge_id_from_previous": source_edge_id,
                    "image_path": image_path,
                    "image_path_exists": "yes" if local_path_exists(image_path) else "no" if image_path else "",
                    "member_source": member_source,
                    "physical_merge_allowed": collection.get("physical_merge_allowed", ""),
                    "logical_collection_only": collection.get("logical_collection_only", ""),
                    "review_flags": collection.get("logical_grouping_review_flags", ""),
                }
            )

    rows.sort(
        key=lambda row: (
            str(row.get("v5_source_key") or ""),
            str(row.get("assembly_group_id") or ""),
            int_or_none(row.get("member_position")) or 10**9,
        )
    )
    for idx, row in enumerate(rows, start=1):
        row["member_rank"] = idx
    summary_rows: list[dict[str, object]] = [
        {"summary_type": "dataset_rows", "name": "logical_collection_members", "count": len(rows)}
    ]
    for field in [
        "logical_grouping_confidence",
        "subgroup",
        "segment_kind",
        "member_source",
        "image_path_exists",
        "page_is_consecutive_with_previous",
    ]:
        for name, count in sorted(Counter(str(row.get(field, "")) for row in rows).items()):
            summary_rows.append({"summary_type": f"logical_members:{field}", "name": name, "count": count})

    write_csv(OUT_DIR / "logical_collection_members.csv", rows, LOGICAL_COLLECTION_MEMBER_FIELDS)
    write_csv(OUT_DIR / "logical_collection_member_summary.csv", summary_rows, ["summary_type", "name", "count"])
    return {
        "logical_collection_members": rows,
        "logical_collection_member_summary": summary_rows,
    }


ROMAN_TABLE_VALUES = {
    "i": 1,
    "v": 5,
    "x": 10,
    "l": 50,
    "c": 100,
    "d": 500,
    "m": 1000,
}


def roman_to_int(value: str) -> int | None:
    total = 0
    prev = 0
    for char in reversed(value.lower()):
        current = ROMAN_TABLE_VALUES.get(char)
        if current is None:
            return None
        if current < prev:
            total -= current
        else:
            total += current
            prev = current
    return total if total > 0 else None


def normalize_table_number_token(token: str, suffix: str = "") -> str:
    cleaned = compact_text(token).lower().strip(".")
    if not cleaned:
        return ""
    if cleaned.isdigit():
        number = str(int(cleaned))
    else:
        roman_value = roman_to_int(cleaned)
        number = str(roman_value) if roman_value is not None else cleaned
    suffix = compact_text(suffix).lower()
    return f"{number}{suffix}" if suffix else number


def extract_title_table_numbers(title: object) -> list[str]:
    text = str(title or "")
    numbers: list[str] = []
    for match in re.finditer(
        r"\b(?:table|tab\.?)(?![a-z])\s*(?:[-.:\u2014\u2013]?\s*(?:no|number)\.?)?\s*[-.:\u2014\u2013]?\s*([ivxlcdm]+|\d+)([a-z]?)\b",
        text,
        flags=re.IGNORECASE,
    ):
        normalized = normalize_table_number_token(match.group(1), match.group(2))
        if normalized and normalized not in numbers:
            numbers.append(normalized)
    return numbers


def logical_context_risk_level(flags: list[str]) -> str:
    high_flags = {
        "member_pages_not_consecutive",
        "missing_page_members",
        "table_number_inconsistent",
        "page_only_no_image_evidence",
        "unsafe_physical_merge_flag",
    }
    medium_flags = {
        "large_bounded_collection_review",
        "missing_image_evidence",
        "partial_variant_evidence",
        "missing_variant_evidence",
        "title_missing",
        "no_title_table_number_evidence",
    }
    flag_set = set(flags)
    if flag_set & high_flags:
        return "high_review_risk"
    if flag_set & medium_flags:
        return "medium_review_risk"
    return "lower_review_risk"


def logical_context_recommended_action(risk_level: str) -> str:
    if risk_level == "high_review_risk":
        return "manual_review_before_site_grouping"
    if risk_level == "medium_review_risk":
        return "use_as_logical_collection_with_spot_check"
    return "use_as_logical_collection_only"


def summarize_counter(values: list[str]) -> str:
    return "; ".join(f"{name}={count}" for name, count in sorted(Counter(values).items()) if name)


def build_logical_collection_context_outputs(
    confirmed_logical: list[dict[str, object]],
    logical_members: list[dict[str, object]],
) -> dict[str, list[dict[str, object]]]:
    members_by_group: dict[str, list[dict[str, object]]] = defaultdict(list)
    for member in logical_members:
        members_by_group[str(member.get("assembly_group_id") or "")].append(member)

    rows: list[dict[str, object]] = []
    for collection in confirmed_logical:
        group_id = str(collection.get("assembly_group_id") or "")
        pages = parse_page_tokens(collection.get("page_ranges", ""))
        members = sorted(
            members_by_group.get(group_id, []),
            key=lambda row: int_or_none(row.get("member_position")) or 10**9,
        )
        member_pages = [
            int_or_none(member.get("pdf_page"))
            for member in members
            if int_or_none(member.get("pdf_page")) is not None
        ]
        member_page_sequence_status = (
            "consecutive"
            if member_pages == pages and all(b - a == 1 for a, b in zip(member_pages, member_pages[1:]))
            else "missing_or_nonconsecutive"
        )
        titles = [str(member.get("title") or "").strip() for member in members if str(member.get("title") or "").strip()]
        title_table_numbers: list[str] = []
        for title in titles:
            for number in extract_title_table_numbers(title):
                if number not in title_table_numbers:
                    title_table_numbers.append(number)
        if len(title_table_numbers) == 0:
            title_number_status = "unknown"
        elif len(title_table_numbers) == 1:
            title_number_status = "consistent"
        else:
            title_number_status = "inconsistent"

        variant_terms = [str(member.get("variant_terms") or "").strip() for member in members]
        variant_categories = [str(member.get("variant_category_from_previous") or "").strip() for member in members]
        variant_nonempty = [term for term in variant_terms if term]
        if not members:
            variant_coverage_status = "missing"
        elif len(variant_nonempty) == len(members):
            variant_coverage_status = "complete"
        elif variant_nonempty:
            variant_coverage_status = "partial"
        else:
            variant_coverage_status = "missing"

        missing_image_count = sum(1 for member in members if member.get("image_path_exists") != "yes")
        page_only_count = sum(1 for member in members if member.get("member_source") == "page_only_no_image_evidence")
        if not members:
            image_status = "missing_members"
        elif missing_image_count == 0:
            image_status = "complete"
        elif page_only_count:
            image_status = "page_only_members"
        else:
            image_status = "partial"

        flags = ["logical_collection_only", "no_physical_merge"]
        if member_page_sequence_status != "consecutive":
            flags.append("member_pages_not_consecutive")
        if len(member_pages) != len(pages):
            flags.append("missing_page_members")
        if (int_or_none(collection.get("n_physical_pages")) or len(pages)) >= 5:
            flags.append("large_bounded_collection_review")
        if title_number_status == "inconsistent":
            flags.append("table_number_inconsistent")
        elif title_number_status == "unknown":
            flags.append("no_title_table_number_evidence")
        if len(titles) < len(members):
            flags.append("title_missing")
        if variant_coverage_status == "partial":
            flags.append("partial_variant_evidence")
        elif variant_coverage_status == "missing" and collection.get("logical_grouping_confidence") != "high_self_contained_sequence_collection":
            flags.append("missing_variant_evidence")
        if missing_image_count:
            flags.append("missing_image_evidence")
        if page_only_count:
            flags.append("page_only_no_image_evidence")
        if (
            collection.get("safe_to_physical_merge") != "no"
            or collection.get("physical_merge_allowed") != "no"
            or collection.get("logical_collection_only") != "yes"
        ):
            flags.append("unsafe_physical_merge_flag")

        risk_level = logical_context_risk_level(flags)
        rows.append(
            {
                "context_rank": 0,
                "risk_level": risk_level,
                "risk_flags": "; ".join(flags),
                "recommended_action": logical_context_recommended_action(risk_level),
                "safe_to_use_logical_collection": collection.get("safe_to_use_logical_collection", ""),
                "safe_to_physical_merge": "no",
                "physical_merge_allowed": collection.get("physical_merge_allowed", ""),
                "logical_collection_only": collection.get("logical_collection_only", ""),
                "assembly_group_id": group_id,
                "logical_grouping_confidence": collection.get("logical_grouping_confidence", ""),
                "subgroup": collection.get("subgroup", ""),
                "segment_kind": collection.get("segment_kind", ""),
                "segment_variant": collection.get("segment_variant", ""),
                "v5_source_key": collection.get("v5_source_key", ""),
                "source_pdf": collection.get("source_pdf", ""),
                "source_pdf_id": collection.get("source_pdf_id", ""),
                "page_ranges": collection.get("page_ranges", ""),
                "n_collection_pages": collection.get("n_physical_pages", ""),
                "member_count": len(members),
                "member_pages": "; ".join(str(page) for page in member_pages),
                "member_page_sequence_status": member_page_sequence_status,
                "missing_member_count": max(0, len(pages) - len(member_pages)),
                "member_title_count": len(titles),
                "title_table_numbers": "; ".join(title_table_numbers),
                "title_table_number_status": title_number_status,
                "member_titles": " | ".join(titles),
                "variant_terms_sequence": " | ".join(variant_terms),
                "variant_category_sequence": " | ".join(variant_categories),
                "variant_coverage_status": variant_coverage_status,
                "image_evidence_status": image_status,
                "missing_image_member_count": missing_image_count,
                "page_only_member_count": page_only_count,
                "member_source_summary": summarize_counter([str(member.get("member_source") or "") for member in members]),
                "logical_grouping_reason": collection.get("logical_grouping_reason", ""),
                "logical_grouping_review_flags": collection.get("logical_grouping_review_flags", ""),
            }
        )

    risk_priority = {"high_review_risk": 1, "medium_review_risk": 2, "lower_review_risk": 3}
    rows.sort(
        key=lambda row: (
            risk_priority.get(str(row.get("risk_level") or ""), 9),
            str(row.get("v5_source_key") or row.get("source_pdf") or ""),
            int_or_none(row.get("n_collection_pages")) or 0,
            str(row.get("assembly_group_id") or ""),
        )
    )
    for idx, row in enumerate(rows, start=1):
        row["context_rank"] = idx

    summary_rows: list[dict[str, object]] = [
        {"summary_type": "dataset_rows", "name": "logical_collection_context_review", "count": len(rows)}
    ]
    for field in [
        "risk_level",
        "recommended_action",
        "title_table_number_status",
        "variant_coverage_status",
        "image_evidence_status",
        "logical_grouping_confidence",
        "subgroup",
    ]:
        for name, count in sorted(Counter(str(row.get(field, "")) for row in rows).items()):
            summary_rows.append({"summary_type": f"logical_context:{field}", "name": name, "count": count})

    write_csv(OUT_DIR / "logical_collection_context_review.csv", rows, LOGICAL_COLLECTION_CONTEXT_FIELDS)
    write_csv(OUT_DIR / "logical_collection_context_summary.csv", summary_rows, ["summary_type", "name", "count"])
    return {
        "logical_collection_context_review": rows,
        "logical_collection_context_summary": summary_rows,
    }


def build_logical_review_candidate_members(
    logical_review: list[dict[str, object]],
    child_rows: list[dict[str, object]],
) -> dict[str, list[dict[str, object]]]:
    child_by_id = logical_child_lookup(child_rows)
    stage_by_id = source_stage_lookup()
    rows: list[dict[str, object]] = []

    for collection in logical_review:
        group_id = str(collection.get("assembly_group_id") or "")
        pages = parse_page_tokens(collection.get("page_ranges", ""))
        if not group_id or not pages:
            continue
        source = collection.get("v5_source_key") or collection.get("source_pdf")
        page_nodes = refined_page_nodes_for_group(group_id)
        child = child_by_id.get(group_id, {})
        stage = stage_by_id.get(str(collection.get("source_stage_group_id") or ""), {})
        child_images = split_semicolon_values(child.get("image_paths", ""))
        stage_images = split_semicolon_values(stage.get("image_paths", ""))
        title_parts = split_semicolon_values(collection.get("primary_title", ""))

        for idx, page in enumerate(pages, start=1):
            node = page_nodes.get(page, {})
            member_source = "refined_sibling_edge_page_node" if node else ""
            image_path = node.get("image_path", "")
            title = node.get("title", "")
            variant_terms = node.get("variant_terms", "")
            variant_category = node.get("variant_category_from_previous", "")
            source_edge_id = node.get("source_edge_id_from_previous", "")
            if not image_path and idx <= len(child_images):
                image_path = child_images[idx - 1]
                member_source = member_source or "logical_child_segment_image_list"
            if not image_path and idx <= len(stage_images):
                image_path = stage_images[idx - 1]
                member_source = member_source or "source_stage_logical_image_list"
            if not title and idx <= len(title_parts):
                title = title_parts[idx - 1]
            if not variant_terms and collection.get("segment_variant"):
                variant_terms = str(collection.get("segment_variant") or "")
            if not member_source:
                member_source = "page_only_no_image_evidence"
            rows.append(
                {
                    "member_rank": 0,
                    "assembly_group_id": group_id,
                    "logical_grouping_confidence": collection.get("logical_grouping_confidence", ""),
                    "subgroup": collection.get("subgroup", ""),
                    "segment_kind": collection.get("segment_kind", ""),
                    "segment_variant": collection.get("segment_variant", ""),
                    "v5_source_key": source,
                    "source_pdf": collection.get("source_pdf", ""),
                    "source_pdf_id": collection.get("source_pdf_id", ""),
                    "collection_page_ranges": collection.get("page_ranges", ""),
                    "n_collection_pages": len(pages),
                    "member_position": idx,
                    "pdf_page": page,
                    "page_is_consecutive_with_previous": "yes" if idx == 1 or page == pages[idx - 2] + 1 else "no",
                    "title": title,
                    "variant_terms": variant_terms,
                    "variant_category_from_previous": variant_category,
                    "source_edge_id_from_previous": source_edge_id,
                    "image_path": image_path,
                    "image_path_exists": "yes" if local_path_exists(image_path) else "no" if image_path else "",
                    "member_source": member_source,
                    "physical_merge_allowed": collection.get("physical_merge_allowed", ""),
                    "logical_collection_only": collection.get("logical_collection_only", ""),
                    "review_flags": collection.get("logical_grouping_review_flags", ""),
                }
            )

    rows.sort(
        key=lambda row: (
            str(row.get("v5_source_key") or ""),
            str(row.get("assembly_group_id") or ""),
            int_or_none(row.get("member_position")) or 10**9,
        )
    )
    for idx, row in enumerate(rows, start=1):
        row["member_rank"] = idx

    summary_rows: list[dict[str, object]] = [
        {"summary_type": "dataset_rows", "name": "logical_review_candidate_members", "count": len(rows)}
    ]
    for field in [
        "logical_grouping_confidence",
        "subgroup",
        "segment_kind",
        "member_source",
        "image_path_exists",
        "page_is_consecutive_with_previous",
    ]:
        for name, count in sorted(Counter(str(row.get(field, "")) for row in rows).items()):
            summary_rows.append({"summary_type": f"logical_review_members:{field}", "name": name, "count": count})

    write_csv(OUT_DIR / "logical_review_candidate_members.csv", rows, LOGICAL_COLLECTION_MEMBER_FIELDS)
    write_csv(OUT_DIR / "logical_review_candidate_member_summary.csv", summary_rows, ["summary_type", "name", "count"])
    return {
        "logical_review_candidate_members": rows,
        "logical_review_candidate_member_summary": summary_rows,
    }


def logical_review_priority(row: dict[str, object], pages: list[int], title_numbers: list[str]) -> tuple[str, str, str]:
    n_pages = int_or_none(row.get("n_physical_pages")) or len(pages)
    bounded = n_pages <= MAX_ACTIONABLE_LOGICAL_PAGES and not row.get("missing_pages") and contiguous_pages(str(row.get("page_ranges", "")))
    title = str(row.get("primary_title") or "")
    core = bool(re.search(r"\btable\s*vii\b|\bvii\b", title, flags=re.IGNORECASE) or "7" in title_numbers)
    age_civil = bool(re.search(r"age|civil|religion", title, flags=re.IGNORECASE))
    if row.get("logical_grouping_decision") == "review_large_logical_collection":
        return (
            "large_context_only",
            "large_or_parent_logical_context",
            "Keep as provenance/context only; use bounded child/refined sibling groups instead.",
        )
    if bounded and core and age_civil:
        return (
            "high_bounded_core_table_review",
            "bounded_core_table_same_title_review",
            "Review as a bounded logical collection candidate; do not physically merge.",
        )
    if bounded and age_civil:
        return (
            "medium_bounded_age_civil_review",
            "bounded_age_civil_same_title_review",
            "Review as a bounded same-type logical collection candidate.",
        )
    if bounded:
        return (
            "medium_bounded_same_table_review",
            "bounded_same_table_review",
            "Review title/schema/image evidence before using as a logical collection.",
        )
    return (
        "large_same_table_review",
        "same_table_too_large_for_direct_use",
        "Decompose into smaller consecutive subgroups before use.",
    )


def build_logical_review_context_outputs(
    logical_review: list[dict[str, object]],
    review_members: list[dict[str, object]],
) -> dict[str, list[dict[str, object]]]:
    members_by_group: dict[str, list[dict[str, object]]] = defaultdict(list)
    for member in review_members:
        members_by_group[str(member.get("assembly_group_id") or "")].append(member)

    rows: list[dict[str, object]] = []
    for collection in logical_review:
        group_id = str(collection.get("assembly_group_id") or "")
        pages = parse_page_tokens(collection.get("page_ranges", ""))
        members = sorted(
            members_by_group.get(group_id, []),
            key=lambda row: int_or_none(row.get("member_position")) or 10**9,
        )
        member_pages = [
            int_or_none(member.get("pdf_page"))
            for member in members
            if int_or_none(member.get("pdf_page")) is not None
        ]
        titles = [str(member.get("title") or "").strip() for member in members if str(member.get("title") or "").strip()]
        title_numbers: list[str] = []
        for title in [str(collection.get("primary_title") or ""), *titles]:
            for number in extract_title_table_numbers(title):
                if number not in title_numbers:
                    title_numbers.append(number)
        if len(title_numbers) == 0:
            title_number_status = "unknown"
        elif len(title_numbers) == 1:
            title_number_status = "consistent"
        else:
            title_number_status = "inconsistent"
        priority, candidate_tier, action = logical_review_priority(collection, pages, title_numbers)
        missing_image_count = sum(1 for member in members if member.get("image_path_exists") != "yes")
        page_only_count = sum(1 for member in members if member.get("member_source") == "page_only_no_image_evidence")
        if not members:
            image_status = "missing_members"
        elif missing_image_count == 0:
            image_status = "complete"
        elif page_only_count:
            image_status = "page_only_members"
        else:
            image_status = "partial"
        rows.append(
            {
                "review_context_rank": 0,
                "review_priority": priority,
                "review_candidate_tier": candidate_tier,
                "review_reason": "review_only_candidate; title_or_family_similarity_is_not_enough_for_promotion",
                "recommended_action": action,
                "safe_to_use_logical_collection": collection.get("safe_to_use_logical_collection", ""),
                "safe_to_physical_merge": "no",
                "physical_merge_allowed": collection.get("physical_merge_allowed", ""),
                "logical_collection_only": collection.get("logical_collection_only", ""),
                "assembly_group_id": group_id,
                "logical_grouping_decision": collection.get("logical_grouping_decision", ""),
                "logical_grouping_confidence": collection.get("logical_grouping_confidence", ""),
                "subgroup": collection.get("subgroup", ""),
                "tier": collection.get("tier", ""),
                "segment_kind": collection.get("segment_kind", ""),
                "v5_source_key": collection.get("v5_source_key", ""),
                "source_pdf": collection.get("source_pdf", ""),
                "source_pdf_id": collection.get("source_pdf_id", ""),
                "page_ranges": collection.get("page_ranges", ""),
                "n_physical_pages": collection.get("n_physical_pages", ""),
                "member_count": len(members),
                "member_pages": "; ".join(str(page) for page in member_pages),
                "pages_are_consecutive": "yes" if contiguous_pages(str(collection.get("page_ranges", ""))) else "no",
                "bounded_candidate": "yes" if (int_or_none(collection.get("n_physical_pages")) or len(pages)) <= MAX_ACTIONABLE_LOGICAL_PAGES and contiguous_pages(str(collection.get("page_ranges", ""))) and not collection.get("missing_pages") else "no",
                "core_table_vii_signal": "yes" if "7" in title_numbers or re.search(r"\btable\s*vii\b|\bvii\b", str(collection.get("primary_title") or ""), flags=re.IGNORECASE) else "no",
                "age_civil_signal": "yes" if re.search(r"age|civil|religion", str(collection.get("primary_title") or ""), flags=re.IGNORECASE) else "no",
                "title_table_numbers": "; ".join(title_numbers),
                "title_table_number_status": title_number_status,
                "image_evidence_status": image_status,
                "missing_image_member_count": missing_image_count,
                "page_only_member_count": page_only_count,
                "primary_title": collection.get("primary_title", ""),
                "logical_grouping_reason": collection.get("logical_grouping_reason", ""),
                "logical_grouping_review_flags": collection.get("logical_grouping_review_flags", ""),
            }
        )

    priority_order = {
        "high_bounded_core_table_review": 1,
        "medium_bounded_age_civil_review": 2,
        "medium_bounded_same_table_review": 3,
        "large_same_table_review": 4,
        "large_context_only": 5,
    }
    rows.sort(
        key=lambda row: (
            priority_order.get(str(row.get("review_priority") or ""), 9),
            str(row.get("v5_source_key") or row.get("source_pdf") or ""),
            int_or_none(row.get("n_physical_pages")) or 10**9,
            str(row.get("assembly_group_id") or ""),
        )
    )
    for idx, row in enumerate(rows, start=1):
        row["review_context_rank"] = idx

    summary_rows: list[dict[str, object]] = [
        {"summary_type": "dataset_rows", "name": "logical_review_context", "count": len(rows)}
    ]
    for field in [
        "review_priority",
        "review_candidate_tier",
        "bounded_candidate",
        "core_table_vii_signal",
        "age_civil_signal",
        "title_table_number_status",
        "image_evidence_status",
        "logical_grouping_decision",
    ]:
        for name, count in sorted(Counter(str(row.get(field, "")) for row in rows).items()):
            summary_rows.append({"summary_type": f"logical_review_context:{field}", "name": name, "count": count})

    write_csv(OUT_DIR / "logical_review_context.csv", rows, LOGICAL_REVIEW_CONTEXT_FIELDS)
    write_csv(OUT_DIR / "logical_review_context_summary.csv", summary_rows, ["summary_type", "name", "count"])
    return {
        "logical_review_context": rows,
        "logical_review_context_summary": summary_rows,
    }


def evidence_lookup() -> tuple[dict[str, dict[str, str]], list[dict[str, str]]]:
    by_id: dict[str, dict[str, str]] = {}
    for path in [
        OUT_DIR / "source_stage_required_physical.csv",
        OUT_DIR / "source_stage_logical_operational.csv",
        OUT_DIR / "source_stage_operational_manifest.csv",
    ]:
        for row in read_csv(path):
            group_id = row.get("v28_group_id", "")
            if group_id:
                existing = by_id.get(group_id, {})
                merged = dict(existing)
                for key, value in row.items():
                    if value or key not in merged:
                        merged[key] = value
                by_id[group_id] = merged
    return by_id, read_csv(OUT_DIR / "missing_bridge_groups.csv")


def bridge_images_for(row: dict[str, object], missing_rows: list[dict[str, str]]) -> str:
    source = str(row.get("v5_source_key") or row.get("source_pdf") or "")
    missing = str(row.get("missing_pages") or "")
    title = compact_text(row.get("primary_title", ""))
    matches: list[str] = []
    for candidate in missing_rows:
        if (candidate.get("v5_source_key") or candidate.get("source_pdf") or "") != source:
            continue
        if candidate.get("missing_pages", "") != missing:
            continue
        candidate_title = compact_text(candidate.get("primary_title", ""))
        if title and candidate_title and title[:80] != candidate_title[:80]:
            continue
        if candidate.get("image_paths"):
            matches.append(candidate["image_paths"])
    return "; ".join(dict.fromkeys(matches))


def queue_policy(row: dict[str, object]) -> tuple[int, str, str, str, str]:
    tier = str(row.get("tier", ""))
    subgroup = str(row.get("subgroup", ""))
    layer = str(row.get("assembly_layer", ""))
    n_pages = int_or_none(row.get("n_physical_pages")) or 0

    if tier == "needs_value_review":
        return (
            1,
            "physical_value_review",
            "value_constraint_review",
            "Open Assembled Preview and compare value-check failures against the source images.",
            "Promote only if manual value review accepts the assembled rows.",
        )
    if tier == "partial_review":
        return (
            2,
            "physical_partial_review",
            "partial_row_alignment",
            "Review passing blocks and source images; use only blocks whose row alignment is visually confirmed.",
            "Promote block-level rows only after manual confirmation or improved parsing.",
        )
    if tier == "manual_review" and layer == "required_physical":
        return (
            3,
            "physical_failed_constraints",
            "row_alignment_failed",
            "Inspect source pair for OCR/parser issues; likely needs corrected OCR/parsing before assembly.",
            "Promote only after row-count, label-inheritance, and column-order constraints pass.",
        )
    if tier == "blocked_missing_ocr_artifact":
        return (
            4,
            "physical_missing_ocr_artifact",
            "missing_ocr_artifact",
            "OCR or locate OCR artifact for the existing adjacent images, then rerun assembly pipeline.",
            "Promote only after row assembly constraints pass on the recovered artifact.",
        )
    if tier == "blocked_missing_pages":
        return (
            5,
            "missing_pdf_pages",
            "missing_intermediate_pages",
            "Render/OCR the listed missing PDF pages, then rerun assembly pipeline.",
            "Promote only if the completed page sequence remains consecutive and compatible.",
        )
    if tier == "review_large_logical_parent":
        return (
            6,
            "large_logical_parent_context",
            "parent_too_large_for_actionable_group",
            "Use child segments if emitted; otherwise manually decompose with source images before use.",
            "Promote only as smaller non-overlapping child groups, not as a large flat assembly.",
        )
    if tier == "review_large_collection":
        return (
            7,
            "large_logical_collection_review",
            "large_same_family_collection",
            "Manually identify bounded sibling/continuation subgroups; do not use the whole parent directly.",
            "Promote only after decomposition into small consecutive non-overlapping groups.",
        )
    if subgroup == "logical_same_table_candidate":
        bucket = "small_same_table_logical_review" if n_pages <= MAX_ACTIONABLE_LOGICAL_PAGES else "same_table_logical_review"
        return (
            8 if n_pages <= MAX_ACTIONABLE_LOGICAL_PAGES else 9,
            bucket,
            "same_title_not_enough",
            "Needs visual/schema review; title/family sameness alone is intentionally not sufficient.",
            "Promote only if adjacent images show explicit continuation or identical bounded sibling structure.",
        )
    return (
        10,
        "other_review",
        "unclassified_review_or_blocker",
        "Inspect reason/review_flags and decide whether stronger evidence is available.",
        "Promote only after the relevant assembly invariant is explicitly satisfied.",
    )


def build_review_priority_outputs(rows_by_name: dict[str, list[dict[str, object]]]) -> dict[str, list[dict[str, object]]]:
    by_id, missing_rows = evidence_lookup()
    source_rows = [
        row
        for row in rows_by_name["manifest"]
        if row.get("decision") in {"review", "blocked"}
    ]
    queue_rows: list[dict[str, object]] = []
    for row in source_rows:
        stage = by_id.get(str(row.get("source_stage_group_id", "")), {})
        priority, bucket, blocker, action, condition = queue_policy(row)
        image_paths = stage.get("image_paths", "") or bridge_images_for(row, missing_rows)
        evidence_paths = stage.get("ocr_artifacts", "")
        queue_rows.append(
            {
                "queue_rank": 0,
                "priority_level": priority,
                "queue_bucket": bucket,
                "blocker_type": blocker,
                "recommended_next_action": action,
                "promotion_condition": condition,
                **{field: row.get(field, "") for field in MANIFEST_FIELDS},
                "evidence_paths": evidence_paths,
                "image_paths": image_paths,
            }
        )
    queue_rows.sort(
        key=lambda r: (
            int_or_none(r.get("priority_level")) or 99,
            str(r.get("v5_source_key") or r.get("source_pdf")),
            int_or_none(str(r.get("page_ranges")).split(";", 1)[0].split("-", 1)[0]) or 10**9,
            str(r.get("assembly_group_id")),
        )
    )
    for idx, row in enumerate(queue_rows, start=1):
        row["queue_rank"] = idx

    summary_rows: list[dict[str, object]] = []
    bucket_counts = Counter(str(row.get("queue_bucket")) for row in queue_rows)
    priority_counts = Counter(str(row.get("priority_level")) for row in queue_rows)
    blocker_counts = Counter(str(row.get("blocker_type")) for row in queue_rows)
    for bucket, count in sorted(bucket_counts.items(), key=lambda item: (min(int(row["priority_level"]) for row in queue_rows if row["queue_bucket"] == item[0]), item[0])):
        examples = [row for row in queue_rows if row.get("queue_bucket") == bucket][:3]
        summary_rows.append(
            {
                "summary_type": "queue_bucket",
                "name": bucket,
                "count": count,
                "priority_levels": "; ".join(sorted({str(row.get("priority_level")) for row in queue_rows if row.get("queue_bucket") == bucket})),
                "example_group_ids": "; ".join(str(row.get("assembly_group_id")) for row in examples),
                "recommended_next_action": examples[0].get("recommended_next_action", "") if examples else "",
                "promotion_condition": examples[0].get("promotion_condition", "") if examples else "",
            }
        )
    for priority, count in sorted(priority_counts.items(), key=lambda item: int(item[0])):
        summary_rows.append({"summary_type": "priority_level", "name": priority, "count": count})
    for blocker, count in sorted(blocker_counts.items()):
        summary_rows.append({"summary_type": "blocker_type", "name": blocker, "count": count})

    write_csv(OUT_DIR / "assembly_review_queue.csv", queue_rows, REVIEW_QUEUE_FIELDS)
    write_csv(
        OUT_DIR / "evidence_gap_summary.csv",
        summary_rows,
        ["summary_type", "name", "count", "priority_levels", "example_group_ids", "recommended_next_action", "promotion_condition"],
    )
    return {"review_queue": queue_rows, "evidence_gap_summary": summary_rows}


def publish_stable_artifacts() -> dict[str, list[dict[str, object]]]:
    manifest_raw = read_csv(SOURCE_FILES["manifest"])
    manifest = [normalize_manifest_row(row) for row in manifest_raw]

    copies = {
        "pairwise_edges.csv": SOURCE_FILES["v27_pairwise_edges"],
        "seed_pairwise_edges.csv": SOURCE_FILES["v25_seed_edges"],
        "seed_components.csv": SOURCE_FILES["v25_seed_components"],
        "seed_component_members.csv": SOURCE_FILES["v25_component_members"],
        "ironclad_physical_assemblies.csv": SOURCE_FILES["v25_ironclad_physical"],
        "growth_candidates.csv": SOURCE_FILES["v25_growth_candidates"],
        "logical_collection_candidates.csv": SOURCE_FILES["v25_logical_candidates"],
        "missing_bridge_groups.csv": SOURCE_FILES["v27_missing_bridge"],
        "row_assembly_attempts.csv": SOURCE_FILES["v26_row_attempts"],
        "cross_page_constraints.csv": SOURCE_FILES["v26_constraints"],
        "row_assembly_review_queue.csv": SOURCE_FILES["v26_review_queue"],
        "structurally_complete_row_assemblies.csv": SOURCE_FILES["v26_structurally_complete"],
        "partial_row_assemblies.csv": SOURCE_FILES["v26_partial"],
        "assembled_rows.jsonl": SOURCE_FILES["v26_rows_jsonl"],
        "artifact_resolution_report.csv": SOURCE_FILES["v26_artifact_resolution"],
        "candidate_row_assembly_attempts.csv": SOURCE_FILES["v26_candidate_attempts"],
        "candidate_cross_page_constraints.csv": SOURCE_FILES["v26_candidate_constraints"],
        "candidate_artifact_resolution_report.csv": SOURCE_FILES["v26_candidate_artifact_resolution"],
        "candidate_test_pass_review_queue.csv": SOURCE_FILES["v26_candidate_pass_review"],
        "candidate_assembled_rows.jsonl": SOURCE_FILES["v26_candidate_rows_jsonl"],
        "source_stage_required_physical.csv": SOURCE_FILES["v28_required_physical"],
        "source_stage_logical_operational.csv": SOURCE_FILES["v28_logical_operational"],
        "refined_sibling_groups.csv": SOURCE_FILES["v29_refined_sibling_groups"],
        "refined_sibling_edges.csv": SOURCE_FILES["v29_refined_sibling_edges"],
        "source_stage_operational_manifest.csv": SOURCE_FILES["manifest"],
    }
    for dst_name, src in copies.items():
        copy_if_exists(src, OUT_DIR / dst_name)

    manifest, child_rows = apply_logical_parent_child_policy(manifest)
    use_now = [row for row in manifest if row["decision"] == "use_now"]
    review = [row for row in manifest if row["decision"] == "review"]
    blocked = [row for row in manifest if row["decision"] == "blocked"]
    physical = [row for row in manifest if row["assembly_layer"] == "required_physical"]
    logical = [row for row in manifest if row["assembly_layer"] == "logical_collection"]

    write_csv(OUT_DIR / "assembly_manifest.csv", manifest, MANIFEST_FIELDS)
    write_csv(OUT_DIR / "use_now_groups.csv", use_now, MANIFEST_FIELDS)
    write_csv(OUT_DIR / "review_groups.csv", review, MANIFEST_FIELDS)
    write_csv(OUT_DIR / "blocked_groups.csv", blocked, MANIFEST_FIELDS)
    write_csv(OUT_DIR / "required_physical_groups.csv", physical, MANIFEST_FIELDS)
    write_csv(OUT_DIR / "logical_collection_groups.csv", logical, MANIFEST_FIELDS)
    write_csv(OUT_DIR / "logical_child_segments.csv", child_rows, [*MANIFEST_FIELDS, "image_paths"])

    column_outputs = build_column_order_outputs()
    candidate_outputs = build_potential_physical_join_candidates()
    grouping_outputs = build_physical_grouping_outputs()
    logical_grouping_outputs = build_logical_grouping_outputs(logical)
    logical_member_outputs = build_logical_collection_members(
        logical_grouping_outputs["confirmed_logical_collections"],
        child_rows,
    )
    logical_context_outputs = build_logical_collection_context_outputs(
        logical_grouping_outputs["confirmed_logical_collections"],
        logical_member_outputs["logical_collection_members"],
    )
    logical_review_member_outputs = build_logical_review_candidate_members(
        logical_grouping_outputs["logical_grouping_review"],
        child_rows,
    )
    logical_review_context_outputs = build_logical_review_context_outputs(
        logical_grouping_outputs["logical_grouping_review"],
        logical_review_member_outputs["logical_review_candidate_members"],
    )
    adjacency_outputs = build_adjacency_decision_outputs(
        grouping_outputs["physical_grouping_decisions"],
        candidate_outputs["row_label_dependency_watchlist"],
        candidate_outputs["row_label_dependency_scan"],
        candidate_outputs["row_label_dependency_priority_scan"],
        logical_grouping_outputs["logical_grouping_decisions"],
    )
    artifact_resolution = read_csv(OUT_DIR / "artifact_resolution_report.csv")
    candidate_attempts = read_csv(OUT_DIR / "candidate_row_assembly_attempts.csv")
    candidate_artifact_resolution = read_csv(OUT_DIR / "candidate_artifact_resolution_report.csv")
    candidate_pass_context_outputs = build_candidate_pass_context_outputs(
        candidate_attempts,
        adjacency_outputs["adjacency_decision_index"],
    )
    candidate_missing_artifact_outputs = build_candidate_missing_artifact_targets(
        candidate_artifact_resolution,
        candidate_attempts,
    )
    missing_artifact_outputs = build_missing_artifact_targets(
        artifact_resolution,
        grouping_outputs["physical_grouping_decisions"],
    )
    interim = {
        "manifest": manifest,
        "use_now": use_now,
        "review": review,
        "blocked": blocked,
        "physical": physical,
        "logical": logical,
        "logical_children": child_rows,
        "artifact_resolution": artifact_resolution,
        "candidate_attempts": candidate_attempts,
        "candidate_artifact_resolution": candidate_artifact_resolution,
        **missing_artifact_outputs,
        **column_outputs,
        **candidate_outputs,
        **grouping_outputs,
        **logical_grouping_outputs,
        **logical_member_outputs,
        **logical_context_outputs,
        **logical_review_member_outputs,
        **logical_review_context_outputs,
        **adjacency_outputs,
        **candidate_pass_context_outputs,
        **candidate_missing_artifact_outputs,
    }
    review_outputs = build_review_priority_outputs(interim)

    return {
        "manifest": manifest,
        "use_now": use_now,
        "review": review,
        "blocked": blocked,
        "physical": physical,
        "logical": logical,
        "logical_children": child_rows,
        "artifact_resolution": artifact_resolution,
        "candidate_attempts": candidate_attempts,
        "candidate_artifact_resolution": candidate_artifact_resolution,
        **missing_artifact_outputs,
        **column_outputs,
        **candidate_outputs,
        **grouping_outputs,
        **logical_grouping_outputs,
        **logical_member_outputs,
        **logical_context_outputs,
        **logical_review_member_outputs,
        **logical_review_context_outputs,
        **adjacency_outputs,
        **candidate_pass_context_outputs,
        **candidate_missing_artifact_outputs,
        **review_outputs,
    }


def contiguous_pages(page_ranges: str) -> bool:
    pages: list[int] = []
    for part in str(page_ranges or "").replace(",", ";").split(";"):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            left, right = [int_or_none(p.strip()) for p in part.split("-", 1)]
            if left is None or right is None:
                return False
            pages.extend(range(min(left, right), max(left, right) + 1))
        else:
            value = int_or_none(part)
            if value is None:
                return False
            pages.append(value)
    if len(pages) <= 1:
        return True
    pages = sorted(set(pages))
    return all(b - a == 1 for a, b in zip(pages, pages[1:]))


def overlap_pairs(rows: list[dict[str, object]]) -> list[tuple[str, list[int], str, str]]:
    by_source: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        source = str(row.get("v5_source_key") or row.get("source_pdf") or "")
        by_source[source].append(row)
    overlaps: list[tuple[str, list[int], str, str]] = []
    for source, source_rows in by_source.items():
        for i, left in enumerate(source_rows):
            left_pages = set(parse_page_tokens(left.get("page_ranges", "")))
            if not left_pages:
                continue
            for right in source_rows[i + 1 :]:
                right_pages = set(parse_page_tokens(right.get("page_ranges", "")))
                shared = sorted(left_pages & right_pages)
                if shared:
                    overlaps.append(
                        (
                            source,
                            shared,
                            str(left.get("assembly_group_id", "")),
                            str(right.get("assembly_group_id", "")),
                        )
                    )
    return overlaps


def audit_outputs(rows_by_name: dict[str, list[dict[str, object]]], stage_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    audit: list[dict[str, object]] = []

    def add(check: str, status: str, detail: object = "") -> None:
        audit.append({"check": check, "status": status, "detail": detail})

    for key, src in SOURCE_FILES.items():
        add(f"source_exists:{key}", "pass" if src.exists() else "fail", src)

    add(
        "stage_runs_successful",
        "pass" if all(row.get("status") in {"pass", "skipped_existing_outputs"} for row in stage_rows) else "fail",
        Counter(str(row.get("status")) for row in stage_rows),
    )

    manifest = rows_by_name["manifest"]
    use_now = rows_by_name["use_now"]
    blocked = rows_by_name["blocked"]
    pairwise = read_csv(OUT_DIR / "pairwise_edges.csv")
    seed_edges = read_csv(OUT_DIR / "seed_pairwise_edges.csv")
    row_attempts = read_csv(OUT_DIR / "row_assembly_attempts.csv")
    assembled_rows = read_jsonl(OUT_DIR / "assembled_rows.jsonl")
    preview_rows = rows_by_name.get("assembled_preview", [])
    column_contract = rows_by_name.get("column_contract", [])
    review_queue = rows_by_name.get("review_queue", [])
    potential_candidates = rows_by_name.get("potential_physical_candidates", [])
    row_label_scan = rows_by_name.get("row_label_dependency_scan", [])
    row_label_priority_scan = rows_by_name.get("row_label_dependency_priority_scan", [])
    row_label_odd_unlabeled_priority = rows_by_name.get("row_label_odd_unlabeled_priority", [])
    row_label_watchlist = rows_by_name.get("row_label_dependency_watchlist", [])
    physical_groupings = rows_by_name.get("physical_grouping_decisions", [])
    confirmed_physical_groupings = rows_by_name.get("confirmed_physical_groupings", [])
    logical_groupings = rows_by_name.get("logical_grouping_decisions", [])
    confirmed_logical_collections = rows_by_name.get("confirmed_logical_collections", [])
    logical_collection_members = rows_by_name.get("logical_collection_members", [])
    logical_collection_context = rows_by_name.get("logical_collection_context_review", [])
    logical_review_context = rows_by_name.get("logical_review_context", [])
    logical_review_members = rows_by_name.get("logical_review_candidate_members", [])
    artifact_resolution = rows_by_name.get("artifact_resolution", [])
    missing_artifact_targets = rows_by_name.get("missing_artifact_targets", [])
    candidate_missing_artifact_targets = rows_by_name.get("candidate_missing_artifact_targets", [])
    adjacency_index = rows_by_name.get("adjacency_decision_index", [])
    candidate_attempts = rows_by_name.get("candidate_attempts", [])
    candidate_pass_context = rows_by_name.get("candidate_pass_context_review", [])

    add("manifest_not_empty", "pass" if manifest else "fail", len(manifest))
    add("use_now_not_empty", "pass" if use_now else "fail", len(use_now))
    add("pairwise_edges_present", "pass" if len(pairwise) >= 1 else "fail", len(pairwise))
    add("row_attempts_present", "pass" if len(row_attempts) >= 1 else "fail", len(row_attempts))
    add("assembled_row_previews_match_jsonl", "pass" if len(preview_rows) == len(assembled_rows) else "fail", f"preview={len(preview_rows)} jsonl={len(assembled_rows)}")
    add("column_order_contract_present", "pass" if column_contract else "fail", len(column_contract))
    add("artifact_resolution_report_present", "pass" if artifact_resolution else "fail", len(artifact_resolution))
    expected_queue_count = len(rows_by_name["review"]) + len(rows_by_name["blocked"])
    add("review_queue_covers_all_review_and_blocked_groups", "pass" if len(review_queue) == expected_queue_count else "fail", f"queue={len(review_queue)} expected={expected_queue_count}")

    expected_row_label_scan = [
        edge
        for edge in seed_edges
        if edge.get("page_gap") == "1"
        and not edge.get("missing_pages")
        and (
            row_label_state(float_or_zero(edge.get("prev_row_label_ratio"))) == "unlabelled_or_placeholder"
            or row_label_state(float_or_zero(edge.get("row_label_ratio"))) == "unlabelled_or_placeholder"
        )
    ]
    scan_pair_keys = {
        (
            str(row.get("v5_source_key") or ""),
            str(row.get("prev_physical_page_id") or ""),
            str(row.get("physical_page_id") or ""),
            str(row.get("prev_pdf_page") or ""),
            str(row.get("pdf_page") or ""),
        )
        for row in row_label_scan
    }
    expected_scan_keys = {
        (
            str(edge.get("v5_source_key") or ""),
            str(edge.get("prev_physical_page_id") or ""),
            str(edge.get("physical_page_id") or ""),
            str(edge.get("prev_pdf_page") or ""),
            str(edge.get("pdf_page") or ""),
        )
        for edge in expected_row_label_scan
    }
    add(
        "row_label_dependency_scan_covers_all_low_label_consecutive_edges",
        "pass" if scan_pair_keys == expected_scan_keys else "fail",
        f"scan={len(scan_pair_keys)} expected={len(expected_scan_keys)} missing={len(expected_scan_keys - scan_pair_keys)} extra={len(scan_pair_keys - expected_scan_keys)}",
    )
    watchlist_pair_keys = {
        (
            str(row.get("v5_source_key") or ""),
            str(row.get("prev_physical_page_id") or ""),
            str(row.get("physical_page_id") or ""),
            str(row.get("prev_pdf_page") or ""),
            str(row.get("pdf_page") or ""),
        )
        for row in row_label_watchlist
    }
    add(
        "row_label_dependency_watchlist_is_subset_of_scan",
        "pass" if watchlist_pair_keys <= scan_pair_keys else "fail",
        len(watchlist_pair_keys - scan_pair_keys),
    )
    unsafe_scan_rows = [row for row in row_label_scan if row.get("safe_to_auto_merge_now") == "yes"]
    add("row_label_dependency_scan_is_review_only", "pass" if not unsafe_scan_rows else "fail", len(unsafe_scan_rows))
    row_label_scan_required_columns = {
        "prev_pdf_page_parity",
        "pdf_page_parity",
        "odd_unlabeled_right_page_signal",
    }
    row_label_scan_header = set(read_csv(OUT_DIR / "row_label_dependency_scan.csv")[0].keys()) if row_label_scan else set()
    add(
        "row_label_scan_outputs_include_page_parity_fields",
        "pass" if row_label_scan_required_columns <= row_label_scan_header else "fail",
        json.dumps(sorted(row_label_scan_required_columns - row_label_scan_header), ensure_ascii=False),
    )

    expected_priority_keys = {
        (
            str(row.get("v5_source_key") or ""),
            str(row.get("prev_physical_page_id") or ""),
            str(row.get("physical_page_id") or ""),
            str(row.get("prev_pdf_page") or ""),
            str(row.get("pdf_page") or ""),
        )
        for row in row_label_scan
        if row.get("row_label_pattern") == "left_labelled_right_unlabelled"
        and row.get("candidate_strength") != "excluded_context_or_derivative_review_only"
        and row.get("existing_v25_decision") != "seed_physical_join"
    }
    priority_keys = {
        (
            str(row.get("v5_source_key") or ""),
            str(row.get("prev_physical_page_id") or ""),
            str(row.get("physical_page_id") or ""),
            str(row.get("prev_pdf_page") or ""),
            str(row.get("pdf_page") or ""),
        )
        for row in row_label_priority_scan
    }
    add(
        "row_label_dependency_priority_scan_is_forward_nonexcluded_subset",
        "pass" if priority_keys == expected_priority_keys else "fail",
        f"priority={len(priority_keys)} expected={len(expected_priority_keys)} missing={len(expected_priority_keys - priority_keys)} extra={len(priority_keys - expected_priority_keys)}",
    )
    unsafe_priority_rows = [row for row in row_label_priority_scan if row.get("safe_to_auto_merge_now") == "yes"]
    add("row_label_dependency_priority_scan_is_review_only", "pass" if not unsafe_priority_rows else "fail", len(unsafe_priority_rows))

    expected_odd_unlabeled_keys = {
        (
            str(row.get("v5_source_key") or ""),
            str(row.get("prev_physical_page_id") or ""),
            str(row.get("physical_page_id") or ""),
            str(row.get("prev_pdf_page") or ""),
            str(row.get("pdf_page") or ""),
        )
        for row in row_label_priority_scan
        if row.get("odd_unlabeled_right_page_signal") == "yes"
    }
    odd_unlabeled_keys = {
        (
            str(row.get("v5_source_key") or ""),
            str(row.get("prev_physical_page_id") or ""),
            str(row.get("physical_page_id") or ""),
            str(row.get("prev_pdf_page") or ""),
            str(row.get("pdf_page") or ""),
        )
        for row in row_label_odd_unlabeled_priority
    }
    add(
        "row_label_odd_unlabeled_priority_is_exact_priority_subset",
        "pass" if odd_unlabeled_keys == expected_odd_unlabeled_keys else "fail",
        f"odd_priority={len(odd_unlabeled_keys)} expected={len(expected_odd_unlabeled_keys)} missing={len(expected_odd_unlabeled_keys - odd_unlabeled_keys)} extra={len(odd_unlabeled_keys - expected_odd_unlabeled_keys)}",
    )
    unsafe_odd_unlabeled_rows = [row for row in row_label_odd_unlabeled_priority if row.get("safe_to_auto_merge_now") == "yes"]
    add("row_label_odd_unlabeled_priority_is_review_only", "pass" if not unsafe_odd_unlabeled_rows else "fail", len(unsafe_odd_unlabeled_rows))

    physical_use_now_bad = [
        row
        for row in use_now
        if row.get("assembly_layer") == "required_physical"
        and (
            row.get("physical_merge_allowed") != "yes"
            or (int_or_none(row.get("n_physical_pages")) or 0) > 2
            or not contiguous_pages(str(row.get("page_ranges", "")))
        )
    ]
    add("physical_use_now_is_pairwise_consecutive", "pass" if not physical_use_now_bad else "fail", len(physical_use_now_bad))

    logical_use_now_bad = [
        row
        for row in use_now
        if row.get("assembly_layer") == "logical_collection"
        and (
            row.get("physical_merge_allowed") != "no"
            or row.get("logical_collection_only") != "yes"
            or row.get("missing_pages")
            or not contiguous_pages(str(row.get("page_ranges", "")))
        )
    ]
    add("logical_use_now_is_collection_only_consecutive_no_gaps", "pass" if not logical_use_now_bad else "fail", len(logical_use_now_bad))

    logical_use_now_too_large = [
        row
        for row in use_now
        if row.get("assembly_layer") == "logical_collection"
        and (int_or_none(row.get("n_physical_pages")) or 0) > MAX_ACTIONABLE_LOGICAL_PAGES
    ]
    add("logical_use_now_groups_are_bounded", "pass" if not logical_use_now_too_large else "fail", len(logical_use_now_too_large))

    child_rows = rows_by_name.get("logical_children", [])
    child_bad = [
        row
        for row in child_rows
        if (
            row.get("decision") != "use_now"
            or not row.get("parent_group_id")
            or row.get("physical_merge_allowed") != "no"
            or (int_or_none(row.get("n_physical_pages")) or 0) > MAX_ACTIONABLE_LOGICAL_PAGES
            or not contiguous_pages(str(row.get("page_ranges", "")))
        )
    ]
    add("logical_child_segments_are_bounded_and_linked", "pass" if not child_bad else "fail", len(child_bad))

    logical_use_now_overlaps = overlap_pairs(
        [
            row
            for row in use_now
            if row.get("assembly_layer") == "logical_collection"
        ]
    )
    add(
        "logical_use_now_groups_do_not_overlap",
        "pass" if not logical_use_now_overlaps else "fail",
        json.dumps(logical_use_now_overlaps[:50], ensure_ascii=False),
    )

    queue_bad = [
        row
        for row in review_queue
        if not row.get("queue_bucket") or not row.get("blocker_type") or not row.get("recommended_next_action") or not row.get("promotion_condition")
    ]
    add("review_queue_rows_have_actionable_blockers", "pass" if not queue_bad else "fail", len(queue_bad))

    high_priority_physical_missing_images = [
        row
        for row in review_queue
        if str(row.get("queue_bucket", "")).startswith("physical_")
        and int_or_none(row.get("priority_level")) in {1, 2, 3, 4}
        and not row.get("image_paths")
    ]
    add("physical_review_queue_rows_have_image_paths", "pass" if not high_priority_physical_missing_images else "fail", len(high_priority_physical_missing_images))

    same_table_auto_bad = [
        row
        for row in use_now
        if row.get("subgroup") == "logical_same_table_candidate"
    ]
    add("same_table_title_only_never_use_now", "pass" if not same_table_auto_bad else "fail", len(same_table_auto_bad))

    blocked_missing_bad = [
        row
        for row in manifest
        if row.get("missing_pages") and row.get("decision") != "blocked"
    ]
    add("missing_pages_are_blocked", "pass" if not blocked_missing_bad else "fail", len(blocked_missing_bad))

    pairwise_physical_bad = [
        row
        for row in pairwise
        if row.get("physical_join_allowed") == "yes"
        and (row.get("page_gap") != "1" or row.get("missing_pages"))
    ]
    add("pairwise_physical_join_requires_adjacent_pages", "pass" if not pairwise_physical_bad else "fail", len(pairwise_physical_bad))

    seed_physical_bad = [
        row
        for row in seed_edges
        if row.get("v25_decision") == "seed_physical_join"
        and (row.get("page_gap") != "1" or row.get("missing_pages"))
    ]
    add("seed_physical_edges_are_adjacent_no_gap", "pass" if not seed_physical_bad else "fail", len(seed_physical_bad))

    column_order_bad = [
        row
        for row in column_contract
        if row.get("column_order_status") != "pass"
    ]
    add("assembled_column_order_contract_passes", "pass" if not column_order_bad else "fail", len(column_order_bad))

    assembled_missing_explicit_order = [
        row
        for row in assembled_rows
        if not isinstance(row.get("column_order"), list) or not row.get("column_order")
    ]
    add("assembled_rows_have_explicit_column_order", "pass" if not assembled_missing_explicit_order else "fail", len(assembled_missing_explicit_order))

    preview_order_bad = [
        row
        for row in preview_rows
        if row.get("column_order_status") != "pass"
        or not row.get("assembled_column_order")
        or not row.get("left_column_order")
        or not row.get("right_column_order")
    ]
    add("assembled_preview_preserves_visible_column_order", "pass" if not preview_order_bad else "fail", len(preview_order_bad))

    potential_bad = [
        row
        for row in potential_candidates
        if row.get("safe_to_auto_merge_now") != "no"
        or row.get("page_gap") != "1"
        or row.get("physical_pair_pattern") != "yes"
        or row.get("candidate_type") != "layout_suspect_physical_pair_review_only"
        or row.get("physical_evidence_status") != "stored_layout_signal_only_not_confirmed"
        or not row.get("blocker_type")
        or not row.get("promotion_condition")
    ]
    add("potential_physical_candidates_are_pairwise_nonpromotional", "pass" if not potential_bad else "fail", len(potential_bad))

    potential_bad_kinds = [
        row
        for row in potential_candidates
        if row.get("prev_page_kind") in {"context_or_blank", "offtarget_table", "derivative_selected_caste_race_tribe"}
        or row.get("page_kind") in {"context_or_blank", "offtarget_table", "derivative_selected_caste_race_tribe"}
    ]
    add("potential_physical_candidates_exclude_hard_bad_page_kinds", "pass" if not potential_bad_kinds else "fail", len(potential_bad_kinds))

    row_label_core = [row for row in row_label_watchlist if row.get("core_table_vii_signal") == "yes"]
    add("row_label_watchlist_core_table_vii_candidates_present", "pass" if row_label_core else "fail", len(row_label_core))
    candidate_attempt_keys = set()
    for row in candidate_attempts:
        left_page, right_page = split_page_pair(row.get("page_pair"))
        candidate_attempt_keys.add(
            pair_key(
                row.get("v5_source_key"),
                left_page,
                right_page,
                row.get("left_image_path"),
                row.get("right_image_path"),
            )
        )
    watchlist_missing_candidate_attempt = []
    for row in row_label_watchlist:
        key = pair_key(row.get("v5_source_key"), row.get("prev_pdf_page"), row.get("pdf_page"), row.get("prev_image_path"), row.get("image_path"))
        if key not in candidate_attempt_keys:
            watchlist_missing_candidate_attempt.append(row)
    add(
        "candidate_row_assembly_attempts_cover_row_label_watchlist",
        "pass" if not watchlist_missing_candidate_attempt else "fail",
        f"candidate_attempts={len(candidate_attempts)} watchlist={len(row_label_watchlist)} missing={len(watchlist_missing_candidate_attempt)}",
    )
    priority_scan_missing_candidate_attempt = []
    for row in row_label_priority_scan:
        key = pair_key(row.get("v5_source_key"), row.get("prev_pdf_page"), row.get("pdf_page"), row.get("prev_image_path"), row.get("image_path"))
        if key not in candidate_attempt_keys:
            priority_scan_missing_candidate_attempt.append(row)
    add(
        "candidate_row_assembly_attempts_cover_row_label_priority_scan",
        "pass" if not priority_scan_missing_candidate_attempt else "fail",
        f"candidate_attempts={len(candidate_attempts)} priority_scan={len(row_label_priority_scan)} missing={len(priority_scan_missing_candidate_attempt)}",
    )
    candidate_bad_policy = [
        row
        for row in candidate_attempts
        if row.get("candidate_promotion_policy") != "review_only_never_auto_promote"
    ]
    add("candidate_row_assembly_attempts_are_review_only", "pass" if not candidate_bad_policy else "fail", len(candidate_bad_policy))
    candidate_pass_attempts = [
        row
        for row in candidate_attempts
        if row.get("v26_status") in {"assembled_pass", "assembled_complete_value_review"}
    ]
    add(
        "candidate_pass_context_covers_candidate_test_passes",
        "pass" if len(candidate_pass_context) == len(candidate_pass_attempts) else "fail",
        f"context={len(candidate_pass_context)} pass_attempts={len(candidate_pass_attempts)}",
    )
    unsafe_candidate_pass_context = [
        row for row in candidate_pass_context if row.get("safe_to_auto_promote") == "yes"
    ]
    add("candidate_pass_context_is_review_only", "pass" if not unsafe_candidate_pass_context else "fail", len(unsafe_candidate_pass_context))

    expected_adjacency_edges = [
        row
        for row in seed_edges
        if (row.get("page_gap") == "1" and not row.get("missing_pages"))
        or row.get("v25_decision") == "missing_bridge_candidate"
    ]
    add(
        "adjacency_decision_index_covers_consecutive_and_missing_bridge_edges",
        "pass" if len(adjacency_index) == len(expected_adjacency_edges) else "fail",
        f"index={len(adjacency_index)} expected={len(expected_adjacency_edges)}",
    )
    adjacency_unclassified = [
        row
        for row in adjacency_index
        if not row.get("final_decision_layer")
        or not row.get("recommended_action")
        or not row.get("promotion_condition")
    ]
    add("adjacency_decision_index_has_actionable_decisions", "pass" if not adjacency_unclassified else "fail", len(adjacency_unclassified))
    adjacency_keys = {
        pair_key(
            row.get("v5_source_key"),
            row.get("prev_pdf_page"),
            row.get("pdf_page"),
            row.get("prev_image_path"),
            row.get("image_path"),
        )
        for row in adjacency_index
    }
    missing_physical_adjacency = []
    for row in physical_groupings:
        left_page, right_page = split_page_pair(row.get("page_pair"))
        key = pair_key(row.get("v5_source_key") or row.get("source_pdf"), left_page, right_page, row.get("left_image_path"), row.get("right_image_path"))
        if key not in adjacency_keys:
            missing_physical_adjacency.append(row)
    add("physical_grouping_decisions_covered_by_adjacency_index", "pass" if not missing_physical_adjacency else "fail", len(missing_physical_adjacency))
    missing_watchlist_adjacency = []
    for row in row_label_watchlist:
        key = pair_key(row.get("v5_source_key"), row.get("prev_pdf_page"), row.get("pdf_page"), row.get("prev_image_path"), row.get("image_path"))
        if key not in adjacency_keys:
            missing_watchlist_adjacency.append(row)
    add("row_label_watchlist_covered_by_adjacency_index", "pass" if not missing_watchlist_adjacency else "fail", len(missing_watchlist_adjacency))
    adjacency_by_key = {
        pair_key(
            row.get("v5_source_key"),
            row.get("prev_pdf_page"),
            row.get("pdf_page"),
            row.get("prev_image_path"),
            row.get("image_path"),
        ): row
        for row in adjacency_index
    }
    missing_scan_adjacency = []
    for row in row_label_scan:
        key = pair_key(row.get("v5_source_key"), row.get("prev_pdf_page"), row.get("pdf_page"), row.get("prev_image_path"), row.get("image_path"))
        adjacency = adjacency_by_key.get(key)
        if not adjacency or adjacency.get("row_label_scan_status") != "present":
            missing_scan_adjacency.append(row)
    add("row_label_dependency_scan_covered_by_adjacency_index", "pass" if not missing_scan_adjacency else "fail", len(missing_scan_adjacency))
    missing_priority_scan_adjacency = []
    for row in row_label_priority_scan:
        key = pair_key(row.get("v5_source_key"), row.get("prev_pdf_page"), row.get("pdf_page"), row.get("prev_image_path"), row.get("image_path"))
        adjacency = adjacency_by_key.get(key)
        if (
            not adjacency
            or adjacency.get("row_label_scan_status") != "present"
            or adjacency.get("row_label_priority_scan_status") != "present"
        ):
            missing_priority_scan_adjacency.append(row)
    add("row_label_dependency_priority_scan_covered_by_adjacency_index", "pass" if not missing_priority_scan_adjacency else "fail", len(missing_priority_scan_adjacency))
    unsafe_adjacency_physical_merge = [
        row
        for row in adjacency_index
        if row.get("safe_to_physical_merge") == "yes"
        and (
            row.get("final_decision_layer") != "confirmed_physical_pair"
            or row.get("physical_safe_to_group_pages") != "yes"
            or row.get("page_gap") != "1"
            or row.get("missing_pages")
        )
    ]
    add("adjacency_index_physical_merge_only_when_confirmed_pairwise", "pass" if not unsafe_adjacency_physical_merge else "fail", len(unsafe_adjacency_physical_merge))
    logical_physical_bad = [
        row
        for row in adjacency_index
        if str(row.get("final_decision_layer", "")).startswith("logical_collection")
        and row.get("safe_to_physical_merge") == "yes"
    ]
    add("adjacency_logical_rows_never_allow_physical_merge", "pass" if not logical_physical_bad else "fail", len(logical_physical_bad))

    grouping_bad = [
        row
        for row in confirmed_physical_groupings
        if row.get("safe_to_group_pages") != "yes"
        or str(row.get("page_gap")) != "1"
        or row.get("structural_status") != "pass"
        or not row.get("assembled_row_count")
        or row.get("pair_scope") != "two_consecutive_pdf_pages"
    ]
    add("confirmed_physical_groupings_are_pairwise_structural_pass", "pass" if not grouping_bad and confirmed_physical_groupings else "fail", f"confirmed={len(confirmed_physical_groupings)} bad={len(grouping_bad)}")

    grouping_publish_bad = [
        row
        for row in confirmed_physical_groupings
        if row.get("safe_to_publish_values") == "yes" and row.get("value_constraint_status") != "pass"
    ]
    add("value_review_groupings_not_marked_publishable", "pass" if not grouping_publish_bad else "fail", len(grouping_publish_bad))

    grouping_review_bad = [
        row
        for row in physical_groupings
        if row.get("grouping_decision") != "confirmed_pairwise_physical_grouping"
        and row.get("safe_to_group_pages") == "yes"
    ]
    add("unconfirmed_physical_groupings_not_marked_safe", "pass" if not grouping_review_bad else "fail", len(grouping_review_bad))

    logical_confirmed_bad = [
        row
        for row in confirmed_logical_collections
        if row.get("safe_to_use_logical_collection") != "yes"
        or row.get("safe_to_physical_merge") != "no"
        or row.get("physical_merge_allowed") != "no"
        or row.get("logical_collection_only") != "yes"
        or row.get("missing_pages")
        or not contiguous_pages(str(row.get("page_ranges", "")))
        or (int_or_none(row.get("n_physical_pages")) or 0) > MAX_ACTIONABLE_LOGICAL_PAGES
    ]
    add("confirmed_logical_collections_are_bounded_collection_only", "pass" if not logical_confirmed_bad and confirmed_logical_collections else "fail", f"confirmed={len(confirmed_logical_collections)} bad={len(logical_confirmed_bad)}")

    logical_members_by_group: dict[str, list[dict[str, object]]] = defaultdict(list)
    for member in logical_collection_members:
        logical_members_by_group[str(member.get("assembly_group_id") or "")].append(member)
    missing_logical_member_groups: list[str] = []
    mismatched_logical_member_groups: list[str] = []
    nonconsecutive_logical_member_groups: list[str] = []
    unsafe_logical_member_groups: list[str] = []
    for row in confirmed_logical_collections:
        group_id = str(row.get("assembly_group_id") or "")
        pages = parse_page_tokens(row.get("page_ranges", ""))
        members = logical_members_by_group.get(group_id, [])
        member_pages = [int_or_none(member.get("pdf_page")) for member in sorted(members, key=lambda m: int_or_none(m.get("member_position")) or 10**9)]
        member_pages = [page for page in member_pages if page is not None]
        if not members:
            missing_logical_member_groups.append(group_id)
        if member_pages != pages:
            mismatched_logical_member_groups.append(group_id)
        if pages and any(b - a != 1 for a, b in zip(pages, pages[1:])):
            nonconsecutive_logical_member_groups.append(group_id)
        if any(member.get("physical_merge_allowed") != "no" or member.get("logical_collection_only") != "yes" for member in members):
            unsafe_logical_member_groups.append(group_id)
    add("confirmed_logical_collections_have_page_members", "pass" if not missing_logical_member_groups else "fail", len(missing_logical_member_groups))
    add("logical_collection_members_match_confirmed_page_ranges", "pass" if not mismatched_logical_member_groups else "fail", json.dumps(mismatched_logical_member_groups[:50], ensure_ascii=False))
    add("logical_collection_member_pages_are_consecutive", "pass" if not nonconsecutive_logical_member_groups else "fail", json.dumps(nonconsecutive_logical_member_groups[:50], ensure_ascii=False))
    add("logical_collection_members_never_allow_physical_merge", "pass" if not unsafe_logical_member_groups else "fail", len(unsafe_logical_member_groups))

    context_by_group = {
        str(row.get("assembly_group_id") or ""): row
        for row in logical_collection_context
        if row.get("assembly_group_id")
    }
    missing_logical_context_groups = [
        str(row.get("assembly_group_id") or "")
        for row in confirmed_logical_collections
        if str(row.get("assembly_group_id") or "") not in context_by_group
    ]
    unsafe_logical_context_rows = [
        row
        for row in logical_collection_context
        if row.get("safe_to_physical_merge") != "no"
        or row.get("physical_merge_allowed") != "no"
        or row.get("logical_collection_only") != "yes"
    ]
    logical_context_missing_risk = [
        row
        for row in logical_collection_context
        if row.get("risk_level") not in {"high_review_risk", "medium_review_risk", "lower_review_risk"}
        or not row.get("risk_flags")
        or not row.get("recommended_action")
    ]
    add("logical_collection_context_covers_confirmed_collections", "pass" if not missing_logical_context_groups and len(logical_collection_context) == len(confirmed_logical_collections) else "fail", f"context={len(logical_collection_context)} confirmed={len(confirmed_logical_collections)} missing={missing_logical_context_groups[:50]}")
    add("logical_collection_context_never_allows_physical_merge", "pass" if not unsafe_logical_context_rows else "fail", len(unsafe_logical_context_rows))
    add("logical_collection_context_rows_have_risk_classification", "pass" if not logical_context_missing_risk and logical_collection_context else "fail", len(logical_context_missing_risk))

    logical_review_by_group = {
        str(row.get("assembly_group_id") or ""): row
        for row in logical_review_context
        if row.get("assembly_group_id")
    }
    missing_logical_review_context_groups = [
        str(row.get("assembly_group_id") or "")
        for row in rows_by_name.get("logical_grouping_review", [])
        if str(row.get("assembly_group_id") or "") not in logical_review_by_group
    ]
    unsafe_logical_review_context_rows = [
        row
        for row in logical_review_context
        if row.get("safe_to_physical_merge") != "no"
        or row.get("physical_merge_allowed") != "no"
        or row.get("logical_collection_only") != "yes"
    ]
    logical_review_members_by_group: dict[str, list[dict[str, object]]] = defaultdict(list)
    for member in logical_review_members:
        logical_review_members_by_group[str(member.get("assembly_group_id") or "")].append(member)
    mismatched_logical_review_member_groups: list[str] = []
    for row in rows_by_name.get("logical_grouping_review", []):
        group_id = str(row.get("assembly_group_id") or "")
        pages = parse_page_tokens(row.get("page_ranges", ""))
        members = logical_review_members_by_group.get(group_id, [])
        member_pages = [int_or_none(member.get("pdf_page")) for member in sorted(members, key=lambda m: int_or_none(m.get("member_position")) or 10**9)]
        member_pages = [page for page in member_pages if page is not None]
        if member_pages != pages:
            mismatched_logical_review_member_groups.append(group_id)
    add(
        "logical_review_context_covers_review_queue",
        "pass" if not missing_logical_review_context_groups and len(logical_review_context) == len(rows_by_name.get("logical_grouping_review", [])) else "fail",
        f"context={len(logical_review_context)} review={len(rows_by_name.get('logical_grouping_review', []))} missing={missing_logical_review_context_groups[:50]}",
    )
    add("logical_review_context_never_allows_physical_merge", "pass" if not unsafe_logical_review_context_rows else "fail", len(unsafe_logical_review_context_rows))
    add("logical_review_candidate_members_match_review_page_ranges", "pass" if not mismatched_logical_review_member_groups else "fail", json.dumps(mismatched_logical_review_member_groups[:50], ensure_ascii=False))

    logical_review_bad = [
        row
        for row in logical_groupings
        if row.get("logical_grouping_decision") != "confirmed_bounded_logical_collection"
        and row.get("safe_to_use_logical_collection") == "yes"
    ]
    add("unconfirmed_logical_collections_not_marked_safe", "pass" if not logical_review_bad else "fail", len(logical_review_bad))

    artifact_resolved_bad = []
    artifact_unresolved_used = []
    for row in artifact_resolution:
        status = row.get("resolution_status")
        resolved = row.get("resolved_ocr_artifact")
        existing = row.get("existing_ocr_artifact")
        if status == "resolved":
            path = Path(str(resolved))
            full = path if path.is_absolute() else ROOT / path
            if not resolved or not full.exists():
                artifact_resolved_bad.append(row)
        if str(status).startswith("unresolved") and resolved and resolved != existing:
            artifact_unresolved_used.append(row)
    add("artifact_resolved_paths_exist", "pass" if not artifact_resolved_bad else "fail", len(artifact_resolved_bad))
    add("unresolved_artifacts_not_attached", "pass" if not artifact_unresolved_used else "fail", len(artifact_unresolved_used))

    unresolved_artifact_rows = [
        row
        for row in artifact_resolution
        if row.get("resolution_status") not in {"existing_artifact", "resolved"}
    ]
    add(
        "missing_artifact_targets_cover_unresolved_resolution_rows",
        "pass" if len(missing_artifact_targets) == len(unresolved_artifact_rows) else "fail",
        f"targets={len(missing_artifact_targets)} unresolved={len(unresolved_artifact_rows)}",
    )
    unresolved_candidate_artifact_rows = [
        row
        for row in rows_by_name.get("candidate_artifact_resolution", [])
        if row.get("resolution_status") not in {"existing_artifact", "resolved"}
    ]
    add(
        "candidate_missing_artifact_targets_cover_unresolved_candidate_rows",
        "pass" if len(candidate_missing_artifact_targets) == len(unresolved_candidate_artifact_rows) else "fail",
        f"targets={len(candidate_missing_artifact_targets)} unresolved={len(unresolved_candidate_artifact_rows)}",
    )
    candidate_target_missing_locators = [
        row
        for row in candidate_missing_artifact_targets
        if not row.get("expected_source_page_globs") and not row.get("expected_image_hash_globs")
    ]
    add("candidate_missing_artifact_targets_have_source_or_image_locator", "pass" if not candidate_target_missing_locators else "fail", len(candidate_target_missing_locators))

    blocked_missing_components = {
        str(row.get("v25_component_id") or "")
        for row in physical_groupings
        if row.get("grouping_decision") == "blocked_missing_ocr_artifact"
    }
    target_components = {
        str(row.get("v25_component_id") or "")
        for row in missing_artifact_targets
        if row.get("v25_component_id")
    }
    missing_blocked_components = sorted(blocked_missing_components - target_components)
    add(
        "blocked_missing_ocr_groupings_have_missing_artifact_targets",
        "pass" if not missing_blocked_components else "fail",
        json.dumps(missing_blocked_components[:50], ensure_ascii=False),
    )
    target_missing_images = [
        row
        for row in missing_artifact_targets
        if row.get("image_path_exists") != "yes"
    ]
    add("missing_artifact_target_images_exist", "pass" if not target_missing_images else "fail", len(target_missing_images))

    add("blocked_groups_present", "pass" if blocked else "fail", len(blocked))

    return audit


def safe_excel(value: object) -> object:
    if value is None:
        return ""
    if isinstance(value, (int, float)):
        return value
    text = str(value)
    if len(text) > 32000:
        return text[:32000] + "..."
    return text


def write_sheet(workbook: xlsxwriter.Workbook, name: str, rows: list[dict[str, object]], fields: list[str] | None = None) -> None:
    worksheet = workbook.add_worksheet(name[:31])
    header_fmt = workbook.add_format({"bold": True, "bg_color": "#D9EAF7", "border": 1})
    if fields is None:
        fields = union_fields(rows)
    if not fields:
        worksheet.write(0, 0, "empty")
        return
    for col, field in enumerate(fields):
        worksheet.write(0, col, field, header_fmt)
        worksheet.set_column(col, col, min(max(len(field) + 2, 12), 42))
    for r, row in enumerate(rows, start=1):
        for c, field in enumerate(fields):
            worksheet.write(r, c, safe_excel(row.get(field, "")))
    worksheet.freeze_panes(1, 0)
    worksheet.autofilter(0, 0, max(len(rows), 1), len(fields) - 1)


def write_workbook(rows_by_name: dict[str, list[dict[str, object]]], audit: list[dict[str, object]], summary: dict[str, object]) -> Path:
    path = OUT_DIR / "assembly_workbook.xlsx"
    workbook = xlsxwriter.Workbook(path)
    summary_rows = [{"metric": key, "value": json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value} for key, value in summary.items()]
    write_sheet(workbook, "Summary", summary_rows, ["metric", "value"])
    write_sheet(workbook, "Use Now", rows_by_name["use_now"], MANIFEST_FIELDS)
    write_sheet(workbook, "Review", rows_by_name["review"], MANIFEST_FIELDS)
    write_sheet(workbook, "Blocked", rows_by_name["blocked"], MANIFEST_FIELDS)
    write_sheet(workbook, "Review Queue", read_csv(OUT_DIR / "assembly_review_queue.csv"), REVIEW_QUEUE_FIELDS)
    write_sheet(workbook, "Evidence Gaps", read_csv(OUT_DIR / "evidence_gap_summary.csv"))
    write_sheet(workbook, "Physical", rows_by_name["physical"], MANIFEST_FIELDS)
    write_sheet(workbook, "Logical", rows_by_name["logical"], MANIFEST_FIELDS)
    write_sheet(workbook, "Logical Grouping", read_csv(OUT_DIR / "logical_grouping_decisions.csv"), LOGICAL_GROUPING_FIELDS)
    write_sheet(workbook, "Confirmed Logical", read_csv(OUT_DIR / "confirmed_logical_collections.csv"), LOGICAL_GROUPING_FIELDS)
    write_sheet(workbook, "Logical Summary", read_csv(OUT_DIR / "logical_grouping_summary.csv"))
    write_sheet(workbook, "Logical Children", read_csv(OUT_DIR / "logical_child_segments.csv"))
    write_sheet(workbook, "Logical Members", read_csv(OUT_DIR / "logical_collection_members.csv"), LOGICAL_COLLECTION_MEMBER_FIELDS)
    write_sheet(workbook, "Logical Member Summary", read_csv(OUT_DIR / "logical_collection_member_summary.csv"))
    write_sheet(workbook, "Logical Context", read_csv(OUT_DIR / "logical_collection_context_review.csv"), LOGICAL_COLLECTION_CONTEXT_FIELDS)
    write_sheet(workbook, "Logical Context Summary", read_csv(OUT_DIR / "logical_collection_context_summary.csv"))
    write_sheet(workbook, "Logical Review Context", read_csv(OUT_DIR / "logical_review_context.csv"), LOGICAL_REVIEW_CONTEXT_FIELDS)
    write_sheet(workbook, "Logical Review Members", read_csv(OUT_DIR / "logical_review_candidate_members.csv"), LOGICAL_COLLECTION_MEMBER_FIELDS)
    write_sheet(workbook, "Logical Review Summary", read_csv(OUT_DIR / "logical_review_context_summary.csv"))
    write_sheet(workbook, "Adjacency Index", read_csv(OUT_DIR / "adjacency_decision_index.csv"), ADJACENCY_DECISION_FIELDS)
    write_sheet(workbook, "Adjacency Summary", read_csv(OUT_DIR / "adjacency_decision_summary.csv"))
    write_sheet(workbook, "Pairwise Edges", read_csv(OUT_DIR / "pairwise_edges.csv"))
    write_sheet(workbook, "Row Attempts", read_csv(OUT_DIR / "row_assembly_attempts.csv"))
    write_sheet(workbook, "Artifact Resolution", read_csv(OUT_DIR / "artifact_resolution_report.csv"))
    write_sheet(workbook, "Missing Artifacts", read_csv(OUT_DIR / "missing_artifact_targets.csv"), MISSING_ARTIFACT_FIELDS)
    write_sheet(workbook, "Candidate Attempts", read_csv(OUT_DIR / "candidate_row_assembly_attempts.csv"))
    write_sheet(workbook, "Candidate Pass Review", read_csv(OUT_DIR / "candidate_test_pass_review_queue.csv"))
    write_sheet(workbook, "Candidate Pass Context", read_csv(OUT_DIR / "candidate_pass_context_review.csv"), CANDIDATE_PASS_CONTEXT_FIELDS)
    write_sheet(workbook, "Candidate Pass Summary", read_csv(OUT_DIR / "candidate_pass_context_summary.csv"))
    write_sheet(workbook, "Candidate Artifacts", read_csv(OUT_DIR / "candidate_artifact_resolution_report.csv"))
    write_sheet(workbook, "Candidate Missing", read_csv(OUT_DIR / "candidate_missing_artifact_targets.csv"), MISSING_ARTIFACT_FIELDS)
    write_sheet(workbook, "Physical Grouping", read_csv(OUT_DIR / "physical_grouping_decisions.csv"), PHYSICAL_GROUPING_FIELDS)
    write_sheet(workbook, "Confirmed Physical", read_csv(OUT_DIR / "confirmed_physical_groupings.csv"), PHYSICAL_GROUPING_FIELDS)
    write_sheet(workbook, "Partial Physical", read_csv(OUT_DIR / "partial_physical_groupings.csv"), PHYSICAL_GROUPING_FIELDS)
    write_sheet(workbook, "Grouping Summary", read_csv(OUT_DIR / "physical_grouping_summary.csv"))
    write_sheet(workbook, "Column Order", read_csv(OUT_DIR / "column_order_contract.csv"), COLUMN_CONTRACT_FIELDS)
    write_sheet(workbook, "Assembled Preview", read_csv(OUT_DIR / "assembled_row_previews.csv"))
    write_sheet(workbook, "Layout Suspects", read_csv(OUT_DIR / "layout_suspect_physical_join_candidates.csv"), POTENTIAL_PHYSICAL_FIELDS)
    write_sheet(workbook, "Row Label Watchlist", read_csv(OUT_DIR / "row_label_dependency_watchlist.csv"), POTENTIAL_PHYSICAL_FIELDS)
    write_sheet(workbook, "Odd Unlabeled Priority", read_csv(OUT_DIR / "row_label_odd_unlabeled_priority.csv"), ROW_LABEL_SCAN_FIELDS)
    write_sheet(workbook, "Row Label Priority", read_csv(OUT_DIR / "row_label_dependency_priority_scan.csv"), ROW_LABEL_SCAN_FIELDS)
    write_sheet(workbook, "Row Label Scan", read_csv(OUT_DIR / "row_label_dependency_scan.csv"), ROW_LABEL_SCAN_FIELDS)
    write_sheet(workbook, "Watchlist Summary", read_csv(OUT_DIR / "potential_physical_join_summary.csv"))
    write_sheet(workbook, "Refined Siblings", read_csv(OUT_DIR / "refined_sibling_groups.csv"))
    write_sheet(workbook, "Audit", audit, ["check", "status", "detail"])
    workbook.close()
    return path


def validate_workbook(path: Path) -> str | None:
    with zipfile.ZipFile(path) as zf:
        return zf.testzip()


def write_readme(summary: dict[str, object]) -> None:
    text = f"""# First-Principles Assembly Pipeline

This directory is the stable assembly output. The versioned v25-v29 scripts are
implementation stages; downstream work should read these stable files first.

## Process

1. Sort pages within each source by real `pdf_page`.
2. Decide pairwise adjacency edges only between consecutive pages.
3. Allow physical assembly only for strict pairwise, consecutive pages that need
   row-label/half-spread completion.
4. Verify row-label inheritance and cross-page value constraints from OCR
   artifacts where available.
5. Build logical collection groups separately from physical merges. These can
   group bounded sibling tables such as Muslim/Christian/Jew subtables, but they
   never imply physical row assembly.
6. Block anything crossing missing pages or missing OCR artifacts; review
   title-only/same-table candidates rather than automatically merging them.

## Main Files

- `assembly_manifest.csv`: every final group with stable columns.
- `use_now_groups.csv`: groups safe enough for downstream assembly/static-site use.
- `review_groups.csv`: plausible groups requiring human or stronger automated review.
- `blocked_groups.csv`: groups blocked by missing pages or missing OCR artifacts.
- `assembly_review_queue.csv`: prioritized residual work with blockers/actions.
- `evidence_gap_summary.csv`: counts by blocker and queue bucket.
- `logical_child_segments.csv`: small actionable children split from large logical parents.
- `adjacency_decision_index.csv`: one normalized decision row for every consecutive source edge plus missing-page bridge candidates; this is the compact no-missed-groupings coverage table.
- `adjacency_decision_summary.csv`: counts for the adjacency decision index.
- `pairwise_edges.csv`: explicit edge-level reasons for join/break decisions.
- `row_assembly_attempts.csv`: row-label inheritance and value-check evidence.
- `artifact_resolution_report.csv`: exact-match resolver report showing whether missing row-assembly OCR artifacts were found locally by source/page or image hash.
- `missing_artifact_targets.csv`: unresolved OCR artifact targets by component/page/image, with expected source-page and image-hash filename patterns for incoming OCR drops.
- `candidate_row_assembly_attempts.csv`: conservative row-label-dependency candidate tests outside the seed physical set; passing rows still require manual review before promotion.
- `candidate_test_pass_review_queue.csv`: candidate tests that structurally passed and should be reviewed, not auto-merged.
- `candidate_pass_context_review.csv`: enriched review-only context for structurally passing candidate tests, including table-number/family/title-risk flags so structural row alignment is not mistaken for proof of assembly.
- `candidate_artifact_resolution_report.csv`: resolver report for candidate-test OCR artifacts.
- `candidate_missing_artifact_targets.csv`: unresolved OCR artifact targets for review-only row-label dependency candidates, with expected source-page and image-hash filename patterns.
- `physical_grouping_decisions.csv`: grouping-first decision layer for every seed physical pair; grouping safety is separated from numeric value cleanliness.
- `confirmed_physical_groupings.csv`: consecutive two-page physical groupings with structural row-label inheritance proof. Some still require value review before numeric publication.
- `partial_physical_groupings.csv`: bounded partial page pairs where at least one block aligns, but the full pair must not be merged without review.
- `logical_grouping_decisions.csv`: grouping-first decision layer for logical collections; confirmed rows are bounded, consecutive, collection-only, and never physical merges.
- `confirmed_logical_collections.csv`: use-now logical sibling/child/self-contained collections suitable for grouping on the site without physical row merging.
- `logical_collection_members.csv`: page-level ordered membership for every confirmed logical collection, including PDF page, title/image evidence, and explicit no-physical-merge flags.
- `logical_collection_context_review.csv`: review-only risk context for confirmed logical collections, including table-number consistency, variant coverage, image evidence, and no-physical-merge invariants.
- `logical_review_context.csv`: review-only scoring for unpromoted logical candidates, especially bounded same-table groups that may become site collections after visual/schema review.
- `logical_review_candidate_members.csv`: ordered page-level membership for every unpromoted logical review candidate, preserving page order without allowing physical merge.
- `column_order_contract.csv`: per physical assembly block column-order checks.
- `assembled_row_previews.csv`: flattened left-page then right-page row preview.
- `row_label_dependency_scan.csv`: broad review-only coverage table for every consecutive source edge with low/placeholder row labels on either page; use this to avoid missed row-label-dependency candidates.
- `row_label_dependency_priority_scan.csv`: compact review-only subset of new forward row-label-dependency leads where the left page has row labels, the right page lacks them, neither page is context/offtarget/derivative, and the pair was not already a seed physical join.
- `row_label_odd_unlabeled_priority.csv`: highest-attention review-only subset where the unlabeled continuation page is an odd-numbered PDF page, matching the common right-page/no-row-label spread pattern.
- `layout_suspect_physical_join_candidates.csv` / `potential_physical_join_candidates.csv`: non-promotional adjacent row-label-dependency candidates whose stored layout flags resemble a physical split; these are layout-suspect review leads, not confirmed physical assemblies.
- `row_label_dependency_watchlist.csv`: broader adjacent row-label-deficiency watchlist for classifier/profile misses; these are not physical candidates unless promoted by stronger evidence.
- `refined_sibling_groups.csv`: bounded logical sibling collections recovered after v28.
- `assembly_workbook.xlsx`: spreadsheet version of the same evidence.

## Current Counts

```json
{json.dumps(summary.get("final_counts", {}), indent=2, sort_keys=True)}
```

## When More OCR Artifacts Arrive

The resolver looks for existing OCR artifacts in these roots:

- all local `export_final/gemini_no_gap_ocr_2026-06-16*` directories, including rerun batches from the OCR window
- `export/money_gapfill_ocr`
- `results`
- `export_final/sendout_final/sendout_v1/mistral_md`

Expected filenames are listed per page in `missing_artifact_targets.csv` as
`expected_source_page_globs` and `expected_image_hash_globs`. After new OCR
artifacts land, rerun:

```bash
python3 build_first_principles_assembly_pipeline.py --out-dir export_final/pdf_recovery_incoming_2026-06-16_first_principles_assembly_pipeline_next
```
"""
    (OUT_DIR / "README.md").write_text(text, encoding="utf-8")


def build_summary(rows_by_name: dict[str, list[dict[str, object]]], stage_rows: list[dict[str, object]]) -> dict[str, object]:
    summaries = {
        key: load_json(path / f"{key}_summary.json")
        for key, path in STAGE_OUTPUTS.items()
        if (path / f"{key}_summary.json").exists()
    }
    manifest = rows_by_name["manifest"]
    use_now = rows_by_name["use_now"]
    final_counts = {
        "assembly_manifest_rows": len(manifest),
        "use_now_groups": len(use_now),
        "review_groups": len(rows_by_name["review"]),
        "blocked_groups": len(rows_by_name["blocked"]),
        "required_physical_groups": len(rows_by_name["physical"]),
        "logical_collection_groups": len(rows_by_name["logical"]),
        "logical_child_segments": len(rows_by_name.get("logical_children", [])),
        "review_queue_rows": len(rows_by_name.get("review_queue", [])),
        "review_queue_by_bucket": dict(Counter(str(row.get("queue_bucket")) for row in rows_by_name.get("review_queue", []))),
        "review_queue_by_priority": dict(Counter(str(row.get("priority_level")) for row in rows_by_name.get("review_queue", []))),
        "adjacency_decision_index_rows": len(rows_by_name.get("adjacency_decision_index", [])),
        "adjacency_decisions_by_layer": dict(Counter(str(row.get("final_decision_layer")) for row in rows_by_name.get("adjacency_decision_index", []))),
        "adjacency_decisions_by_scope": dict(Counter(str(row.get("coverage_scope")) for row in rows_by_name.get("adjacency_decision_index", []))),
        "adjacency_decisions_safe_to_physical_merge": sum(1 for row in rows_by_name.get("adjacency_decision_index", []) if row.get("safe_to_physical_merge") == "yes"),
        "adjacency_decisions_safe_to_logical_group": sum(1 for row in rows_by_name.get("adjacency_decision_index", []) if row.get("safe_to_logical_group") == "yes"),
        "adjacency_decisions_with_row_label_scan": sum(1 for row in rows_by_name.get("adjacency_decision_index", []) if row.get("row_label_scan_status") == "present"),
        "adjacency_decisions_with_row_label_priority_scan": sum(1 for row in rows_by_name.get("adjacency_decision_index", []) if row.get("row_label_priority_scan_status") == "present"),
        "assembled_row_preview_rows": len(rows_by_name.get("assembled_preview", [])),
        "column_order_contract_rows": len(rows_by_name.get("column_contract", [])),
        "column_order_contract_by_status": dict(Counter(str(row.get("column_order_status")) for row in rows_by_name.get("column_contract", []))),
        "physical_grouping_decision_rows": len(rows_by_name.get("physical_grouping_decisions", [])),
        "physical_grouping_decisions_by_status": dict(Counter(str(row.get("grouping_decision")) for row in rows_by_name.get("physical_grouping_decisions", []))),
        "physical_grouping_decisions_by_confidence": dict(Counter(str(row.get("grouping_confidence")) for row in rows_by_name.get("physical_grouping_decisions", []))),
        "confirmed_physical_grouping_rows": len(rows_by_name.get("confirmed_physical_groupings", [])),
        "confirmed_physical_groupings_value_review_required": sum(1 for row in rows_by_name.get("confirmed_physical_groupings", []) if row.get("value_review_required") == "yes"),
        "confirmed_physical_groupings_safe_to_publish_values": sum(1 for row in rows_by_name.get("confirmed_physical_groupings", []) if row.get("safe_to_publish_values") == "yes"),
        "partial_physical_grouping_rows": len(rows_by_name.get("partial_physical_groupings", [])),
        "logical_grouping_decision_rows": len(rows_by_name.get("logical_grouping_decisions", [])),
        "logical_grouping_decisions_by_status": dict(Counter(str(row.get("logical_grouping_decision")) for row in rows_by_name.get("logical_grouping_decisions", []))),
        "logical_grouping_decisions_by_confidence": dict(Counter(str(row.get("logical_grouping_confidence")) for row in rows_by_name.get("logical_grouping_decisions", []))),
        "confirmed_logical_collection_rows": len(rows_by_name.get("confirmed_logical_collections", [])),
        "logical_collection_member_rows": len(rows_by_name.get("logical_collection_members", [])),
        "logical_collection_members_by_confidence": dict(Counter(str(row.get("logical_grouping_confidence")) for row in rows_by_name.get("logical_collection_members", []))),
        "logical_collection_members_by_source": dict(Counter(str(row.get("member_source")) for row in rows_by_name.get("logical_collection_members", []))),
        "logical_collection_members_by_image_path_exists": dict(Counter(str(row.get("image_path_exists")) for row in rows_by_name.get("logical_collection_members", []))),
        "logical_collection_members_with_images": sum(1 for row in rows_by_name.get("logical_collection_members", []) if row.get("image_path")),
        "logical_collection_context_review_rows": len(rows_by_name.get("logical_collection_context_review", [])),
        "logical_collection_context_by_risk_level": dict(Counter(str(row.get("risk_level")) for row in rows_by_name.get("logical_collection_context_review", []))),
        "logical_collection_context_by_title_table_number_status": dict(Counter(str(row.get("title_table_number_status")) for row in rows_by_name.get("logical_collection_context_review", []))),
        "logical_collection_context_by_variant_coverage": dict(Counter(str(row.get("variant_coverage_status")) for row in rows_by_name.get("logical_collection_context_review", []))),
        "logical_collection_context_by_image_evidence": dict(Counter(str(row.get("image_evidence_status")) for row in rows_by_name.get("logical_collection_context_review", []))),
        "logical_grouping_review_rows": len(rows_by_name.get("logical_grouping_review", [])),
        "logical_review_context_rows": len(rows_by_name.get("logical_review_context", [])),
        "logical_review_context_by_priority": dict(Counter(str(row.get("review_priority")) for row in rows_by_name.get("logical_review_context", []))),
        "logical_review_context_bounded_candidates": sum(1 for row in rows_by_name.get("logical_review_context", []) if row.get("bounded_candidate") == "yes"),
        "logical_review_context_core_table_vii": sum(1 for row in rows_by_name.get("logical_review_context", []) if row.get("core_table_vii_signal") == "yes"),
        "logical_review_context_age_civil": sum(1 for row in rows_by_name.get("logical_review_context", []) if row.get("age_civil_signal") == "yes"),
        "logical_review_candidate_member_rows": len(rows_by_name.get("logical_review_candidate_members", [])),
        "logical_review_candidate_members_by_source": dict(Counter(str(row.get("member_source")) for row in rows_by_name.get("logical_review_candidate_members", []))),
        "artifact_resolution_report_rows": len(rows_by_name.get("artifact_resolution", [])),
        "artifact_resolution_by_status": dict(Counter(str(row.get("resolution_status")) for row in rows_by_name.get("artifact_resolution", []))),
        "artifact_resolution_by_method": dict(Counter(str(row.get("resolution_method")) for row in rows_by_name.get("artifact_resolution", []) if row.get("resolution_method"))),
        "candidate_row_assembly_attempt_rows": len(rows_by_name.get("candidate_attempts", [])),
        "candidate_row_assembly_attempts_by_status": dict(Counter(str(row.get("v26_status")) for row in rows_by_name.get("candidate_attempts", []))),
        "candidate_row_assembly_test_pass_review_rows": sum(1 for row in rows_by_name.get("candidate_attempts", []) if row.get("v26_status") in {"assembled_pass", "assembled_complete_value_review"}),
        "candidate_pass_context_review_rows": len(rows_by_name.get("candidate_pass_context_review", [])),
        "candidate_pass_context_by_risk_level": dict(Counter(str(row.get("risk_level")) for row in rows_by_name.get("candidate_pass_context_review", []))),
        "candidate_pass_context_same_table_number": dict(Counter(str(row.get("same_table_number")) for row in rows_by_name.get("candidate_pass_context_review", []))),
        "candidate_artifact_resolution_report_rows": len(rows_by_name.get("candidate_artifact_resolution", [])),
        "candidate_artifact_resolution_by_status": dict(Counter(str(row.get("resolution_status")) for row in rows_by_name.get("candidate_artifact_resolution", []))),
        "candidate_artifact_resolution_by_method": dict(Counter(str(row.get("resolution_method")) for row in rows_by_name.get("candidate_artifact_resolution", []) if row.get("resolution_method"))),
        "candidate_missing_artifact_target_rows": len(rows_by_name.get("candidate_missing_artifact_targets", [])),
        "candidate_missing_artifact_targets_by_priority": dict(Counter(str(row.get("priority_bucket")) for row in rows_by_name.get("candidate_missing_artifact_targets", []))),
        "candidate_missing_artifact_targets_by_source_bucket": dict(Counter(str(row.get("image_source_bucket")) for row in rows_by_name.get("candidate_missing_artifact_targets", []))),
        "candidate_missing_artifact_targets_with_existing_images": sum(1 for row in rows_by_name.get("candidate_missing_artifact_targets", []) if row.get("image_path_exists") == "yes"),
        "missing_artifact_target_rows": len(rows_by_name.get("missing_artifact_targets", [])),
        "missing_artifact_targets_by_priority": dict(Counter(str(row.get("priority_bucket")) for row in rows_by_name.get("missing_artifact_targets", []))),
        "missing_artifact_targets_by_source_bucket": dict(Counter(str(row.get("image_source_bucket")) for row in rows_by_name.get("missing_artifact_targets", []))),
        "missing_artifact_targets_with_existing_images": sum(1 for row in rows_by_name.get("missing_artifact_targets", []) if row.get("image_path_exists") == "yes"),
        "potential_physical_candidate_rows": len(rows_by_name.get("potential_physical_candidates", [])),
        "potential_physical_candidates_by_tier": dict(Counter(str(row.get("confidence_tier")) for row in rows_by_name.get("potential_physical_candidates", []))),
        "potential_physical_candidates_by_type": dict(Counter(str(row.get("candidate_type")) for row in rows_by_name.get("potential_physical_candidates", []))),
        "potential_physical_candidates_by_blocker": dict(Counter(str(row.get("blocker_type")) for row in rows_by_name.get("potential_physical_candidates", []))),
        "potential_physical_candidates_by_evidence_status": dict(Counter(str(row.get("physical_evidence_status")) for row in rows_by_name.get("potential_physical_candidates", []))),
        "potential_physical_core_table_vii_candidates": sum(1 for row in rows_by_name.get("potential_physical_candidates", []) if row.get("core_table_vii_signal") == "yes"),
        "row_label_dependency_scan_rows": len(rows_by_name.get("row_label_dependency_scan", [])),
        "row_label_dependency_scan_by_strength": dict(Counter(str(row.get("candidate_strength")) for row in rows_by_name.get("row_label_dependency_scan", []))),
        "row_label_dependency_scan_by_pattern": dict(Counter(str(row.get("row_label_pattern")) for row in rows_by_name.get("row_label_dependency_scan", []))),
        "row_label_dependency_scan_core_table_vii": sum(1 for row in rows_by_name.get("row_label_dependency_scan", []) if row.get("core_table_vii_signal") == "yes"),
        "row_label_dependency_scan_age_civil": sum(1 for row in rows_by_name.get("row_label_dependency_scan", []) if row.get("age_civil_signal") == "yes"),
        "row_label_dependency_priority_scan_rows": len(rows_by_name.get("row_label_dependency_priority_scan", [])),
        "row_label_dependency_priority_scan_by_strength": dict(Counter(str(row.get("candidate_strength")) for row in rows_by_name.get("row_label_dependency_priority_scan", []))),
        "row_label_dependency_priority_scan_core_table_vii": sum(1 for row in rows_by_name.get("row_label_dependency_priority_scan", []) if row.get("core_table_vii_signal") == "yes"),
        "row_label_dependency_priority_scan_age_civil": sum(1 for row in rows_by_name.get("row_label_dependency_priority_scan", []) if row.get("age_civil_signal") == "yes"),
        "row_label_odd_unlabeled_priority_rows": len(rows_by_name.get("row_label_odd_unlabeled_priority", [])),
        "row_label_odd_unlabeled_priority_by_strength": dict(Counter(str(row.get("candidate_strength")) for row in rows_by_name.get("row_label_odd_unlabeled_priority", []))),
        "row_label_odd_unlabeled_priority_core_table_vii": sum(1 for row in rows_by_name.get("row_label_odd_unlabeled_priority", []) if row.get("core_table_vii_signal") == "yes"),
        "row_label_odd_unlabeled_priority_age_civil": sum(1 for row in rows_by_name.get("row_label_odd_unlabeled_priority", []) if row.get("age_civil_signal") == "yes"),
        "row_label_dependency_watchlist_rows": len(rows_by_name.get("row_label_dependency_watchlist", [])),
        "row_label_dependency_watchlist_by_tier": dict(Counter(str(row.get("confidence_tier")) for row in rows_by_name.get("row_label_dependency_watchlist", []))),
        "row_label_dependency_watchlist_by_type": dict(Counter(str(row.get("candidate_type")) for row in rows_by_name.get("row_label_dependency_watchlist", []))),
        "row_label_dependency_watchlist_by_evidence_status": dict(Counter(str(row.get("physical_evidence_status")) for row in rows_by_name.get("row_label_dependency_watchlist", []))),
        "row_label_dependency_watchlist_core_table_vii": sum(1 for row in rows_by_name.get("row_label_dependency_watchlist", []) if row.get("core_table_vii_signal") == "yes"),
        "use_now_by_subgroup": dict(Counter(str(row.get("subgroup")) for row in use_now)),
        "manifest_by_decision": dict(Counter(str(row.get("decision")) for row in manifest)),
        "manifest_by_tier": dict(Counter(str(row.get("tier")) for row in manifest)),
    }
    return {
        "pipeline_output_dir": str(OUT_DIR),
        "stage_status_counts": dict(Counter(str(row.get("status")) for row in stage_rows)),
        "stage_summaries": summaries,
        "final_counts": final_counts,
    }


def main() -> None:
    global OUT_DIR
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-stages",
        action="store_true",
        help="Publish stable outputs from existing v25-v29 artifacts without rerunning stage scripts.",
    )
    parser.add_argument(
        "--out-dir",
        default=str(DEFAULT_OUT_DIR),
        help="Output directory for the published workbook/CSVs. Defaults to the stable first-principles path.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir).expanduser()
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    OUT_DIR = out_dir

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stage_rows = run_stages(skip_stages=args.skip_stages)
    write_csv(OUT_DIR / "stage_status.csv", stage_rows)

    rows_by_name = publish_stable_artifacts()
    audit = audit_outputs(rows_by_name, stage_rows)
    summary = build_summary(rows_by_name, stage_rows)
    workbook = write_workbook(rows_by_name, audit, summary)
    workbook_bad_member = validate_workbook(workbook)
    audit.append({"check": "assembly_workbook_zip_integrity", "status": "pass" if workbook_bad_member is None else "fail", "detail": workbook_bad_member or ""})
    audit_failure_count = sum(1 for row in audit if row.get("status") != "pass")
    summary["audit_failure_count"] = audit_failure_count
    summary["workbook_zip_bad_member"] = workbook_bad_member
    workbook = write_workbook(rows_by_name, audit, summary)
    final_workbook_bad_member = validate_workbook(workbook)
    if final_workbook_bad_member != workbook_bad_member:
        workbook_bad_member = final_workbook_bad_member
        audit[-1] = {
            "check": "assembly_workbook_zip_integrity",
            "status": "pass" if workbook_bad_member is None else "fail",
            "detail": workbook_bad_member or "",
        }
        audit_failure_count = sum(1 for row in audit if row.get("status") != "pass")
        summary["audit_failure_count"] = audit_failure_count
        summary["workbook_zip_bad_member"] = workbook_bad_member
        workbook = write_workbook(rows_by_name, audit, summary)
    write_csv(OUT_DIR / "audit.csv", audit, ["check", "status", "detail"])
    write_readme(summary)
    with (OUT_DIR / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print(json.dumps(summary["final_counts"], indent=2, sort_keys=True))
    if audit_failure_count:
        raise SystemExit(f"assembly pipeline audit failed: {audit_failure_count} failing checks")


if __name__ == "__main__":
    main()
