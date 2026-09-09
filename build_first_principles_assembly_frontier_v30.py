#!/usr/bin/env python3
"""Build a concise first-principles assembly frontier.

This is a control layer over the audited assembly outputs. It does not perform
OCR, does not promote review-only candidates, and does not broaden physical
merge rules. Physical assembly remains pairwise and consecutive-only.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import xlsxwriter


ROOT = Path(__file__).resolve().parent
DEFAULT_ASSEMBLY_DIR = ROOT / "export_final/pdf_recovery_incoming_2026-06-16_first_principles_assembly_pipeline_v29_logical_review_context"
DEFAULT_INTAKE_DIR = ROOT / "export_final/pdf_recovery_incoming_2026-06-16_assembly_ocr_intake_v4_refresh"
DEFAULT_OUT_DIR = ROOT / "export_final/pdf_recovery_incoming_2026-06-16_first_principles_assembly_frontier_v30"

PHYSICAL_FIELDS = [
    "frontier_rank",
    "frontier_layer",
    "frontier_status",
    "action_bucket",
    "safe_to_physical_merge",
    "safe_to_publish_values",
    "review_required",
    "source_pdf",
    "source_pdf_id",
    "v5_source_key",
    "year",
    "province",
    "page_pair",
    "prev_pdf_page",
    "pdf_page",
    "page_gap",
    "pages_consecutive",
    "v25_component_id",
    "v26_attempt_id",
    "v25_edge_id",
    "v26_status",
    "structural_status",
    "value_constraint_status",
    "assembled_row_count",
    "left_label_ratio",
    "right_label_ratio",
    "row_label_pattern",
    "odd_unlabeled_right_page_signal",
    "candidate_strength",
    "core_table_vii_signal",
    "age_civil_signal",
    "ocr_target_count",
    "ocr_targets_with_render",
    "ocr_targets_with_artifact",
    "ocr_unlock_statuses",
    "best_render_paths",
    "best_artifact_paths",
    "left_ocr_artifact",
    "right_ocr_artifact",
    "left_image_path",
    "right_image_path",
    "decision_reason",
    "review_flags",
    "recommended_action",
    "promotion_condition",
]

ROW_LABEL_FIELDS = [
    "frontier_rank",
    "scan_id",
    "frontier_status",
    "action_bucket",
    "v5_source_key",
    "source_pdf",
    "source_pdf_id",
    "page_pair",
    "prev_pdf_page",
    "pdf_page",
    "page_gap",
    "prev_pdf_page_parity",
    "pdf_page_parity",
    "odd_unlabeled_right_page_signal",
    "candidate_strength",
    "row_label_pattern",
    "candidate_attempt_status",
    "candidate_attempt_id",
    "candidate_risk_level",
    "safe_to_auto_merge_now",
    "core_table_vii_signal",
    "age_civil_signal",
    "prev_row_label_ratio",
    "row_label_ratio",
    "prev_table_family_key",
    "table_family_key",
    "same_table_family",
    "prev_table_number_key",
    "table_number_key",
    "same_table_number",
    "ocr_target_count",
    "ocr_targets_with_render",
    "ocr_targets_with_artifact",
    "ocr_unlock_statuses",
    "best_render_paths",
    "best_artifact_paths",
    "decision_reason",
    "review_flags",
    "recommended_action",
    "promotion_condition",
    "prev_title",
    "title",
    "prev_image_path",
    "image_path",
]

LOGICAL_FIELDS = [
    "frontier_rank",
    "frontier_layer",
    "frontier_status",
    "action_bucket",
    "safe_to_use_logical_collection",
    "safe_to_physical_merge",
    "physical_merge_allowed",
    "logical_collection_only",
    "assembly_group_id",
    "review_priority",
    "review_candidate_tier",
    "logical_grouping_decision",
    "logical_grouping_confidence",
    "risk_level",
    "subgroup",
    "segment_kind",
    "segment_variant",
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
    "decision_reason",
    "review_flags",
    "recommended_action",
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


def int_or_none(value: object) -> int | None:
    if value in {None, ""}:
        return None
    try:
        return int(str(value))
    except ValueError:
        return None


def parse_page_pair(pair: str) -> tuple[str, str]:
    nums = re.findall(r"\d+", str(pair or ""))
    if len(nums) >= 2:
        return nums[0], nums[1]
    return "", ""


def page_gap(left: object, right: object) -> str:
    a = int_or_none(left)
    b = int_or_none(right)
    if a is None or b is None:
        return ""
    return str(b - a)


def pages_consecutive(left: object, right: object) -> str:
    return "yes" if page_gap(left, right) == "1" else "no"


def semi(values: Iterable[object]) -> str:
    return "; ".join(str(value) for value in values if value not in {None, ""})


def aggregate_intake(intake_rows: list[dict[str, str]]) -> dict[str, dict[str, object]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in intake_rows:
        comp_id = row.get("v25_component_id", "")
        if comp_id:
            grouped[comp_id].append(row)

    out: dict[str, dict[str, object]] = {}
    for comp_id, rows in grouped.items():
        out[comp_id] = {
            "ocr_target_count": len(rows),
            "ocr_targets_with_render": sum(1 for row in rows if int_or_none(row.get("matched_render_count")) and int(row.get("matched_render_count") or 0) > 0),
            "ocr_targets_with_artifact": sum(1 for row in rows if int_or_none(row.get("matched_artifact_count")) and int(row.get("matched_artifact_count") or 0) > 0),
            "ocr_unlock_statuses": semi(sorted({row.get("unlock_status", "") for row in rows})),
            "best_render_paths": semi(row.get("best_render_path", "") for row in rows),
            "best_artifact_paths": semi(row.get("best_artifact_path", "") for row in rows),
        }
    return out


def intake_status_for(row: dict[str, str], aggregate: dict[str, dict[str, object]]) -> dict[str, object]:
    return aggregate.get(
        row.get("v25_component_id", ""),
        {
            "ocr_target_count": "",
            "ocr_targets_with_render": "",
            "ocr_targets_with_artifact": "",
            "ocr_unlock_statuses": "",
            "best_render_paths": "",
            "best_artifact_paths": "",
        },
    )


def action_for_physical(row: dict[str, object]) -> str:
    status = str(row.get("frontier_status") or "")
    unlock = str(row.get("ocr_unlock_statuses") or "")
    v26_status = str(row.get("v26_status") or "")
    if status == "confirmed_physical_pair":
        if row.get("safe_to_publish_values") == "yes":
            return "use_now_publishable_physical_pair"
        return "use_now_structure_review_values"
    if status == "partial_physical_review":
        return "manual_review_partial_do_not_merge_full_pair"
    if "artifact_available" in unlock:
        return "rerun_row_constraints_now"
    if "render_available_no_ocr_artifact" in unlock:
        return "ocr_existing_render_then_retest"
    if "still_blocked_no_local_artifact" in unlock:
        return "obtain_render_or_artifact_then_retest"
    if v26_status == "assembled_pass":
        return "manual_review_candidate_pass_before_promoting"
    if v26_status in {"review_failed_constraints", "review_no_safe_block_alignment"}:
        return "do_not_merge_unless_constraints_are_repaired"
    if v26_status == "blocked_missing_ocr_artifact":
        return "ocr_missing_candidate_pages_then_retest"
    return "manual_review"


def build_physical_frontier(assembly_dir: Path, intake_aggregate: dict[str, dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    for row in read_csv(assembly_dir / "physical_grouping_decisions.csv"):
        left, right = parse_page_pair(row.get("page_pair", ""))
        decision = row.get("grouping_decision", "")
        if decision == "confirmed_pairwise_physical_grouping":
            status = "confirmed_physical_pair"
        elif decision == "partial_pairwise_physical_grouping_review":
            status = "partial_physical_review"
        elif decision == "blocked_missing_ocr_artifact":
            status = "blocked_physical_missing_ocr"
        else:
            status = "failed_or_review_physical_pair"
        out = {
            "frontier_layer": "physical_pair_decision",
            "frontier_status": status,
            "safe_to_physical_merge": row.get("safe_to_group_pages", "no"),
            "safe_to_publish_values": row.get("safe_to_publish_values", "no"),
            "review_required": "yes" if row.get("value_review_required") == "yes" or status != "confirmed_physical_pair" else "no",
            "source_pdf": row.get("source_pdf", ""),
            "source_pdf_id": row.get("source_pdf_id", ""),
            "v5_source_key": row.get("v5_source_key", ""),
            "year": row.get("year", ""),
            "province": row.get("province", ""),
            "page_pair": row.get("page_pair", ""),
            "prev_pdf_page": left,
            "pdf_page": right,
            "page_gap": page_gap(left, right),
            "pages_consecutive": pages_consecutive(left, right),
            "v25_component_id": row.get("v25_component_id", ""),
            "v26_attempt_id": row.get("v26_attempt_id", ""),
            "v25_edge_id": "",
            "v26_status": row.get("v26_status", ""),
            "structural_status": row.get("structural_status", ""),
            "value_constraint_status": row.get("value_constraint_status", ""),
            "assembled_row_count": row.get("assembled_row_count", ""),
            "left_label_ratio": row.get("left_label_ratio", ""),
            "right_label_ratio": row.get("right_label_ratio", ""),
            "left_ocr_artifact": row.get("left_ocr_artifact", ""),
            "right_ocr_artifact": row.get("right_ocr_artifact", ""),
            "left_image_path": row.get("left_image_path", ""),
            "right_image_path": row.get("right_image_path", ""),
            "decision_reason": row.get("grouping_reason", ""),
            "review_flags": row.get("grouping_review_flags", ""),
            "recommended_action": row.get("recommended_grouping_action", ""),
            "promotion_condition": "Already confirmed only if safe_to_physical_merge=yes; otherwise keep out of physical assembly.",
        }
        out.update(intake_status_for(row, intake_aggregate))
        out["action_bucket"] = action_for_physical(out)
        rows.append(out)

    candidate_context = {
        row.get("v26_attempt_id", ""): row
        for row in read_csv(assembly_dir / "candidate_pass_context_review.csv")
    }
    for row in read_csv(assembly_dir / "candidate_row_assembly_attempts.csv"):
        left, right = parse_page_pair(row.get("page_pair", ""))
        context = candidate_context.get(row.get("v26_attempt_id", ""), {})
        status = "candidate_row_label_review_pass" if row.get("v26_status") in {"assembled_pass", "assembled_complete_value_review"} else "candidate_row_label_review_not_passed"
        out = {
            "frontier_layer": "row_label_dependency_candidate",
            "frontier_status": status,
            "safe_to_physical_merge": "no",
            "safe_to_publish_values": "no",
            "review_required": "yes",
            "source_pdf": row.get("source_pdf", ""),
            "source_pdf_id": row.get("source_pdf_id", ""),
            "v5_source_key": row.get("v5_source_key", ""),
            "year": row.get("year", ""),
            "province": row.get("province", ""),
            "page_pair": row.get("page_pair", ""),
            "prev_pdf_page": left,
            "pdf_page": right,
            "page_gap": page_gap(left, right),
            "pages_consecutive": pages_consecutive(left, right),
            "v25_component_id": row.get("v25_component_id", ""),
            "v26_attempt_id": row.get("v26_attempt_id", ""),
            "v25_edge_id": row.get("source_v25_edge_id", ""),
            "v26_status": row.get("v26_status", ""),
            "structural_status": row.get("structural_status", ""),
            "value_constraint_status": row.get("value_constraint_status", ""),
            "assembled_row_count": row.get("assembled_row_count", ""),
            "left_label_ratio": row.get("left_label_ratio", ""),
            "right_label_ratio": row.get("right_label_ratio", ""),
            "row_label_pattern": context.get("row_label_scan_pattern", ""),
            "odd_unlabeled_right_page_signal": "yes" if context.get("row_label_priority_scan_status") == "yes" else "",
            "candidate_strength": context.get("row_label_scan_strength", ""),
            "core_table_vii_signal": context.get("core_table_vii_signal", ""),
            "age_civil_signal": context.get("age_civil_signal", ""),
            "left_ocr_artifact": row.get("left_ocr_artifact", ""),
            "right_ocr_artifact": row.get("right_ocr_artifact", ""),
            "left_image_path": row.get("left_image_path", ""),
            "right_image_path": row.get("right_image_path", ""),
            "decision_reason": row.get("edge_reason", ""),
            "review_flags": context.get("risk_flags", row.get("constraint_flags", "")),
            "recommended_action": context.get("recommended_action", row.get("recommended_action", "")),
            "promotion_condition": context.get("promotion_condition", "Review-only candidate; never auto-promote from structural alignment alone."),
        }
        out.update(intake_status_for(row, intake_aggregate))
        out["action_bucket"] = action_for_physical(out)
        rows.append(out)

    rows.sort(
        key=lambda item: (
            str(item.get("action_bucket") or ""),
            str(item.get("v5_source_key") or item.get("source_pdf") or ""),
            int_or_none(item.get("prev_pdf_page")) or 10**9,
            int_or_none(item.get("pdf_page")) or 10**9,
            str(item.get("frontier_layer") or ""),
        )
    )
    for idx, row in enumerate(rows, start=1):
        row["frontier_rank"] = idx
    return rows


def build_row_label_frontier(
    assembly_dir: Path,
    intake_aggregate: dict[str, dict[str, object]],
) -> list[dict[str, object]]:
    candidate_by_edge = {
        row.get("source_v25_edge_id", ""): row
        for row in read_csv(assembly_dir / "candidate_row_assembly_attempts.csv")
    }
    candidate_context = {
        row.get("source_v25_edge_id", ""): row
        for row in read_csv(assembly_dir / "candidate_pass_context_review.csv")
    }
    rows: list[dict[str, object]] = []
    for scan in read_csv(assembly_dir / "row_label_odd_unlabeled_priority.csv"):
        edge_id = scan.get("v25_edge_id", "")
        attempt = candidate_by_edge.get(edge_id, {})
        context = candidate_context.get(edge_id, {})
        comp_id = attempt.get("v25_component_id", "")
        intake = intake_aggregate.get(comp_id, {})
        status = "candidate_tested_" + attempt.get("v26_status", "") if attempt else "not_tested_scan_lead"
        if attempt.get("v26_status") in {"assembled_pass", "assembled_complete_value_review"}:
            action = "manual_review_candidate_pass_before_promoting"
        elif attempt.get("v26_status") == "blocked_missing_ocr_artifact":
            unlock = str(intake.get("ocr_unlock_statuses", ""))
            if "render_available_no_ocr_artifact" in unlock:
                action = "ocr_existing_render_then_retest_candidate"
            elif "still_blocked_no_local_artifact" in unlock:
                action = "obtain_render_or_artifact_then_retest_candidate"
            else:
                action = "ocr_missing_candidate_pages_then_retest"
        elif attempt:
            action = "do_not_merge_unless_constraints_are_repaired"
        else:
            action = "review_scan_lead_and_add_to_candidate_tests_if_relevant"
        rows.append(
            {
                "scan_id": scan.get("scan_id", ""),
                "frontier_status": status,
                "action_bucket": action,
                "v5_source_key": scan.get("v5_source_key", ""),
                "source_pdf": attempt.get("source_pdf", ""),
                "source_pdf_id": attempt.get("source_pdf_id", ""),
                "page_pair": scan.get("page_ranges", ""),
                "prev_pdf_page": scan.get("prev_pdf_page", ""),
                "pdf_page": scan.get("pdf_page", ""),
                "page_gap": scan.get("page_gap", ""),
                "prev_pdf_page_parity": scan.get("prev_pdf_page_parity", ""),
                "pdf_page_parity": scan.get("pdf_page_parity", ""),
                "odd_unlabeled_right_page_signal": scan.get("odd_unlabeled_right_page_signal", ""),
                "candidate_strength": scan.get("candidate_strength", ""),
                "row_label_pattern": scan.get("row_label_pattern", ""),
                "candidate_attempt_status": attempt.get("v26_status", ""),
                "candidate_attempt_id": attempt.get("v26_attempt_id", ""),
                "candidate_risk_level": context.get("risk_level", ""),
                "safe_to_auto_merge_now": "no",
                "core_table_vii_signal": scan.get("core_table_vii_signal", ""),
                "age_civil_signal": scan.get("age_civil_signal", ""),
                "prev_row_label_ratio": scan.get("prev_row_label_ratio", ""),
                "row_label_ratio": scan.get("row_label_ratio", ""),
                "prev_table_family_key": scan.get("prev_table_family_key", ""),
                "table_family_key": scan.get("table_family_key", ""),
                "same_table_family": scan.get("same_table_family", ""),
                "prev_table_number_key": scan.get("prev_table_number_key", ""),
                "table_number_key": scan.get("table_number_key", ""),
                "same_table_number": scan.get("same_table_number", ""),
                "ocr_target_count": intake.get("ocr_target_count", ""),
                "ocr_targets_with_render": intake.get("ocr_targets_with_render", ""),
                "ocr_targets_with_artifact": intake.get("ocr_targets_with_artifact", ""),
                "ocr_unlock_statuses": intake.get("ocr_unlock_statuses", ""),
                "best_render_paths": intake.get("best_render_paths", ""),
                "best_artifact_paths": intake.get("best_artifact_paths", ""),
                "decision_reason": scan.get("scan_reason", ""),
                "review_flags": semi([scan.get("scan_review_flags", ""), context.get("risk_flags", "")]),
                "recommended_action": scan.get("recommended_action", ""),
                "promotion_condition": scan.get("promotion_condition", ""),
                "prev_title": scan.get("prev_title", ""),
                "title": scan.get("title", ""),
                "prev_image_path": scan.get("prev_image_path", ""),
                "image_path": scan.get("image_path", ""),
            }
        )
    rows.sort(
        key=lambda item: (
            0 if str(item.get("candidate_strength", "")).startswith("high_core") else 1,
            str(item.get("action_bucket") or ""),
            str(item.get("v5_source_key") or ""),
            int_or_none(item.get("prev_pdf_page")) or 10**9,
        )
    )
    for idx, row in enumerate(rows, start=1):
        row["frontier_rank"] = idx
    return rows


def build_logical_frontier(assembly_dir: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in read_csv(assembly_dir / "confirmed_logical_collections.csv"):
        rows.append(
            {
                "frontier_layer": "confirmed_logical_collection",
                "frontier_status": "confirmed_logical_collection_only",
                "action_bucket": "use_now_logical_collection_only",
                "safe_to_use_logical_collection": row.get("safe_to_use_logical_collection", "yes"),
                "safe_to_physical_merge": "no",
                "physical_merge_allowed": "no",
                "logical_collection_only": "yes",
                "assembly_group_id": row.get("assembly_group_id", ""),
                "logical_grouping_decision": row.get("logical_grouping_decision", ""),
                "logical_grouping_confidence": row.get("logical_grouping_confidence", ""),
                "subgroup": row.get("subgroup", ""),
                "segment_kind": row.get("segment_kind", ""),
                "segment_variant": row.get("segment_variant", ""),
                "v5_source_key": row.get("v5_source_key", ""),
                "source_pdf": row.get("source_pdf", ""),
                "source_pdf_id": row.get("source_pdf_id", ""),
                "page_ranges": row.get("page_ranges", ""),
                "n_physical_pages": row.get("n_physical_pages", ""),
                "primary_title": row.get("primary_title", ""),
                "decision_reason": row.get("logical_grouping_reason", row.get("reason", "")),
                "review_flags": row.get("logical_grouping_review_flags", row.get("review_flags", "")),
                "recommended_action": row.get("recommended_logical_action", "use_as_logical_collection_only"),
            }
        )
    for row in read_csv(assembly_dir / "logical_review_context.csv"):
        rows.append(
            {
                "frontier_layer": "logical_review_candidate",
                "frontier_status": "review_only_logical_candidate",
                "action_bucket": row.get("review_priority", "review_logical_candidate"),
                "safe_to_use_logical_collection": row.get("safe_to_use_logical_collection", "no"),
                "safe_to_physical_merge": "no",
                "physical_merge_allowed": "no",
                "logical_collection_only": "yes",
                "assembly_group_id": row.get("assembly_group_id", ""),
                "review_priority": row.get("review_priority", ""),
                "review_candidate_tier": row.get("review_candidate_tier", ""),
                "logical_grouping_decision": row.get("logical_grouping_decision", ""),
                "logical_grouping_confidence": row.get("logical_grouping_confidence", ""),
                "subgroup": row.get("subgroup", ""),
                "segment_kind": row.get("segment_kind", ""),
                "v5_source_key": row.get("v5_source_key", ""),
                "source_pdf": row.get("source_pdf", ""),
                "source_pdf_id": row.get("source_pdf_id", ""),
                "page_ranges": row.get("page_ranges", ""),
                "n_physical_pages": row.get("n_physical_pages", ""),
                "member_count": row.get("member_count", ""),
                "member_pages": row.get("member_pages", ""),
                "pages_are_consecutive": row.get("pages_are_consecutive", ""),
                "bounded_candidate": row.get("bounded_candidate", ""),
                "core_table_vii_signal": row.get("core_table_vii_signal", ""),
                "age_civil_signal": row.get("age_civil_signal", ""),
                "title_table_numbers": row.get("title_table_numbers", ""),
                "title_table_number_status": row.get("title_table_number_status", ""),
                "image_evidence_status": row.get("image_evidence_status", ""),
                "missing_image_member_count": row.get("missing_image_member_count", ""),
                "page_only_member_count": row.get("page_only_member_count", ""),
                "primary_title": row.get("primary_title", ""),
                "decision_reason": row.get("review_reason", row.get("logical_grouping_reason", "")),
                "review_flags": row.get("logical_grouping_review_flags", ""),
                "recommended_action": row.get("recommended_action", ""),
            }
        )
    rows.sort(
        key=lambda item: (
            0 if item.get("frontier_status") == "confirmed_logical_collection_only" else 1,
            str(item.get("action_bucket") or ""),
            str(item.get("v5_source_key") or item.get("source_pdf") or ""),
            str(item.get("page_ranges") or ""),
        )
    )
    for idx, row in enumerate(rows, start=1):
        row["frontier_rank"] = idx
    return rows


def build_summary(
    physical: list[dict[str, object]],
    row_label: list[dict[str, object]],
    logical: list[dict[str, object]],
    assembly_dir: Path,
    intake_dir: Path,
    out_dir: Path,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = [
        {"metric": "paths", "name": "assembly_dir", "count": output_path(assembly_dir)},
        {"metric": "paths", "name": "intake_dir", "count": output_path(intake_dir)},
        {"metric": "paths", "name": "out_dir", "count": output_path(out_dir)},
        {"metric": "counts", "name": "physical_frontier_rows", "count": len(physical)},
        {"metric": "counts", "name": "row_label_frontier_rows", "count": len(row_label)},
        {"metric": "counts", "name": "logical_frontier_rows", "count": len(logical)},
    ]
    counters = [
        ("physical_by_status", Counter(str(row.get("frontier_status") or "") for row in physical)),
        ("physical_by_action", Counter(str(row.get("action_bucket") or "") for row in physical)),
        ("physical_by_safe_to_merge", Counter(str(row.get("safe_to_physical_merge") or "") for row in physical)),
        ("row_label_by_action", Counter(str(row.get("action_bucket") or "") for row in row_label)),
        ("row_label_by_strength", Counter(str(row.get("candidate_strength") or "") for row in row_label)),
        ("logical_by_status", Counter(str(row.get("frontier_status") or "") for row in logical)),
        ("logical_by_action", Counter(str(row.get("action_bucket") or "") for row in logical)),
    ]
    for metric, counter in counters:
        for name, count in sorted(counter.items()):
            rows.append({"metric": metric, "name": name, "count": count})
    return rows


def audit_outputs(
    physical: list[dict[str, object]],
    row_label: list[dict[str, object]],
    logical: list[dict[str, object]],
) -> list[dict[str, object]]:
    audit: list[dict[str, object]] = []
    bad_confirmed = [
        row for row in physical
        if row.get("frontier_status") == "confirmed_physical_pair" and row.get("pages_consecutive") != "yes"
    ]
    audit.append(
        {
            "check": "confirmed_physical_pairs_are_consecutive",
            "status": "pass" if not bad_confirmed else "fail",
            "detail": f"{len(bad_confirmed)} non-consecutive confirmed pairs",
        }
    )
    unsafe_merge = [
        row for row in physical
        if row.get("safe_to_physical_merge") == "yes" and row.get("frontier_status") != "confirmed_physical_pair"
    ]
    audit.append(
        {
            "check": "only_confirmed_pairs_allow_physical_merge",
            "status": "pass" if not unsafe_merge else "fail",
            "detail": f"{len(unsafe_merge)} non-confirmed physical rows allow merge",
        }
    )
    bad_candidate = [
        row for row in physical
        if row.get("frontier_layer") == "row_label_dependency_candidate" and row.get("safe_to_physical_merge") == "yes"
    ]
    audit.append(
        {
            "check": "candidate_row_label_pairs_are_review_only",
            "status": "pass" if not bad_candidate else "fail",
            "detail": f"{len(bad_candidate)} candidate rows allow physical merge",
        }
    )
    bad_logical = [
        row for row in logical
        if row.get("physical_merge_allowed") == "yes" or row.get("safe_to_physical_merge") == "yes"
    ]
    audit.append(
        {
            "check": "logical_frontier_never_allows_physical_merge",
            "status": "pass" if not bad_logical else "fail",
            "detail": f"{len(bad_logical)} logical rows allow physical merge",
        }
    )
    bad_odd = [
        row for row in row_label
        if row.get("odd_unlabeled_right_page_signal") != "yes" or row.get("pages_consecutive") == "no"
    ]
    # row_label rows do not carry pages_consecutive; enforce page_gap == 1 instead.
    bad_odd = [row for row in row_label if row.get("odd_unlabeled_right_page_signal") != "yes" or str(row.get("page_gap")) != "1"]
    audit.append(
        {
            "check": "odd_unlabeled_frontier_is_adjacent_signal_only",
            "status": "pass" if not bad_odd else "fail",
            "detail": f"{len(bad_odd)} odd/unlabeled rows are not adjacent signal rows",
        }
    )
    return audit


def write_workbook(
    path: Path,
    summary: list[dict[str, object]],
    physical: list[dict[str, object]],
    row_label: list[dict[str, object]],
    logical: list[dict[str, object]],
    audit: list[dict[str, object]],
) -> None:
    workbook = xlsxwriter.Workbook(path)
    header_fmt = workbook.add_format({"bold": True, "bg_color": "#E8EEF7", "border": 1})
    wrap_fmt = workbook.add_format({"text_wrap": True, "valign": "top"})
    sheets = [
        ("Summary", summary, SUMMARY_FIELDS),
        ("Physical Frontier", physical, PHYSICAL_FIELDS),
        ("Odd Row Label Frontier", row_label, ROW_LABEL_FIELDS),
        ("Logical Frontier", logical, LOGICAL_FIELDS),
        ("Audit", audit, AUDIT_FIELDS),
    ]
    for name, rows, fields in sheets:
        ws = workbook.add_worksheet(name[:31])
        for col, field in enumerate(fields):
            ws.write(0, col, field, header_fmt)
        for row_idx, row in enumerate(rows, start=1):
            for col, field in enumerate(fields):
                value = row.get(field, "")
                fmt = wrap_fmt if any(token in field for token in ["reason", "flags", "action", "condition", "path", "title"]) else None
                ws.write(row_idx, col, value, fmt)
        ws.freeze_panes(1, 0)
        ws.autofilter(0, 0, max(len(rows), 1), max(len(fields) - 1, 0))
        for col, field in enumerate(fields):
            ws.set_column(col, col, min(max(len(field) + 2, 12), 56))
    workbook.close()


def validate_workbook(path: Path) -> str | None:
    try:
        with zipfile.ZipFile(path) as zf:
            return zf.testzip()
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"


def build_frontier(assembly_dir: Path, intake_dir: Path, out_dir: Path) -> dict[str, object]:
    intake_rows = read_csv(intake_dir / "ocr_target_unlock_status.csv")
    intake_aggregate = aggregate_intake(intake_rows)
    physical = build_physical_frontier(assembly_dir, intake_aggregate)
    row_label = build_row_label_frontier(assembly_dir, intake_aggregate)
    logical = build_logical_frontier(assembly_dir)
    summary = build_summary(physical, row_label, logical, assembly_dir, intake_dir, out_dir)
    audit = audit_outputs(physical, row_label, logical)

    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "physical_assembly_frontier.csv", physical, PHYSICAL_FIELDS)
    write_csv(out_dir / "odd_unlabeled_row_label_frontier.csv", row_label, ROW_LABEL_FIELDS)
    write_csv(out_dir / "logical_collection_frontier.csv", logical, LOGICAL_FIELDS)
    write_csv(out_dir / "summary.csv", summary, SUMMARY_FIELDS)
    workbook_path = out_dir / "assembly_frontier_workbook.xlsx"
    workbook_bad_member = validate_workbook(workbook_path)
    audit.append(
        {
            "check": "assembly_frontier_workbook_zip_integrity",
            "status": "pass" if workbook_bad_member is None else "fail",
            "detail": workbook_bad_member or "",
        }
    )
    write_csv(out_dir / "audit.csv", audit, AUDIT_FIELDS)
    write_workbook(workbook_path, summary, physical, row_label, logical, audit)
    workbook_bad_member = validate_workbook(workbook_path)
    audit[-1] = {
        "check": "assembly_frontier_workbook_zip_integrity",
        "status": "pass" if workbook_bad_member is None else "fail",
        "detail": workbook_bad_member or "",
    }
    write_csv(out_dir / "audit.csv", audit, AUDIT_FIELDS)
    write_workbook(workbook_path, summary, physical, row_label, logical, audit)
    audit_failure_count = sum(1 for row in audit if row.get("status") != "pass")
    result = {
        "assembly_dir": output_path(assembly_dir),
        "intake_dir": output_path(intake_dir),
        "out_dir": output_path(out_dir),
        "physical_frontier_rows": len(physical),
        "row_label_frontier_rows": len(row_label),
        "logical_frontier_rows": len(logical),
        "physical_by_status": dict(Counter(str(row.get("frontier_status") or "") for row in physical)),
        "physical_by_action": dict(Counter(str(row.get("action_bucket") or "") for row in physical)),
        "row_label_by_action": dict(Counter(str(row.get("action_bucket") or "") for row in row_label)),
        "logical_by_status": dict(Counter(str(row.get("frontier_status") or "") for row in logical)),
        "logical_by_action": dict(Counter(str(row.get("action_bucket") or "") for row in logical)),
        "audit_failure_count": audit_failure_count,
        "workbook_zip_bad_member": workbook_bad_member,
    }
    (out_dir / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    if audit_failure_count:
        raise SystemExit(f"frontier audit failed: {audit_failure_count} failures")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assembly-dir", default=str(DEFAULT_ASSEMBLY_DIR))
    parser.add_argument("--intake-dir", default=str(DEFAULT_INTAKE_DIR))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    args = parser.parse_args()

    assembly_dir = Path(args.assembly_dir).expanduser()
    intake_dir = Path(args.intake_dir).expanduser()
    out_dir = Path(args.out_dir).expanduser()
    if not assembly_dir.is_absolute():
        assembly_dir = ROOT / assembly_dir
    if not intake_dir.is_absolute():
        intake_dir = ROOT / intake_dir
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    result = build_frontier(assembly_dir, intake_dir, out_dir)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
