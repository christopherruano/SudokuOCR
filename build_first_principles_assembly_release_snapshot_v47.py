#!/usr/bin/env python3
"""Build a final no-OCR assembly release snapshot.

This is a thin reporting layer over the first-principles assembly stack. It
does not promote edges or run OCR/API calls. It reconciles the legacy contract
with the later v42-v46 promotion/frontier/materialization layers so downstream
work has one clear, current workbook to inspect.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterable

import xlsxwriter


ROOT = Path(__file__).resolve().parent
DEFAULT_OUT_DIR = ROOT / "export_final/pdf_recovery_incoming_2026-06-16_first_principles_assembly_release_snapshot_v47"

SUMMARY_FIELDS = ["metric", "name", "value", "detail"]
AUDIT_FIELDS = ["check", "status", "detail"]
CHAIN_AUDIT_FIELDS = [
    "component_id",
    "status",
    "source_key",
    "component_pages",
    "component_size",
    "edge_count",
    "edge_keys",
    "issues",
]
DECISION_LEDGER_FIELDS = [
    "ledger_rank",
    "decision_type",
    "decision_status",
    "assembly_or_edge_id",
    "source_key",
    "source_pdf_id",
    "page_range",
    "component_pages",
    "pages_consecutive",
    "grouping_policy",
    "site_use_policy",
    "safe_to_publish_values",
    "review_required",
    "priority",
    "row_label_signal",
    "right_page_no_row_labels_signal",
    "pair_direction_signal",
    "pair_direction_reason",
    "row_pairing_mode",
    "row_label_source_summary",
    "row_count_detail",
    "column_order",
    "column_order_evidence",
    "zero_check_status",
    "artifact_status",
    "title_summary",
    "image_paths",
    "matrix_csv",
    "recommended_action",
    "decision_reason",
    "review_flags",
]
EDGE_DECISION_LEDGER_FIELDS = [
    "edge_rank",
    "edge_id",
    "edge_key",
    "source_key",
    "source_pdf_id",
    "page_pair",
    "pages_consecutive",
    "page_gap",
    "edge_decision_status",
    "grouping_policy",
    "physical_merge_allowed_now",
    "logical_group_allowed_now",
    "requires_manual_review",
    "requires_value_review",
    "likely_after_missing_pages",
    "likely_after_ocr",
    "site_action",
    "promotion_condition",
    "row_label_dependency_signal",
    "right_page_no_row_labels",
    "right_page_parity",
    "left_row_label_ratio",
    "right_row_label_ratio",
    "row_label_scan_strength",
    "row_label_scan_pattern",
    "page_kind_transition",
    "page_class_transition",
    "classifier_break_signal",
    "same_table_family",
    "same_table_number",
    "table_family_transition",
    "table_number_transition",
    "title_switch_signal",
    "decision_reason",
    "rule_flags",
    "source_decision_layer",
    "frontier_statuses",
    "frontier_action_buckets",
    "component_ids",
    "image_paths",
]
PHYSICAL_FRONTIER_ACTION_QUEUE_FIELDS = [
    "queue_rank",
    "queue_bucket",
    "queue_priority",
    "edge_key",
    "source_key",
    "source_pdf_id",
    "page_pair",
    "pages_consecutive",
    "page_gap",
    "missing_intermediate_pages",
    "current_status",
    "physical_merge_allowed_now",
    "safe_physical_edge_now",
    "manual_review_required",
    "action",
    "promotion_condition",
    "queue_row_label_review_signal",
    "row_label_signal",
    "right_page_no_row_labels_signal",
    "right_page_parity",
    "row_label_scan_strength",
    "row_label_scan_pattern",
    "edge_decision_status",
    "row_alignment_evidence",
    "row_counts",
    "row_label_counts",
    "shared_row_label_count",
    "column_order_evidence",
    "combined_column_group_order",
    "zero_check_status",
    "zero_check_failures",
    "title_summary",
    "artifact_status",
    "image_paths",
    "recommended_next_step",
    "rule_reason",
    "review_flags",
]
MANUAL_PREVIEW_TRIAGE_FIELDS = [
    "triage_rank",
    "triage_bucket",
    "suggested_action",
    "promotion_condition",
    "confidence_score",
    "edge_key",
    "assembly_id",
    "source_key",
    "source_pdf_id",
    "page_pair",
    "pages_consecutive",
    "frontier_category",
    "assembly_status",
    "pair_direction_signal",
    "pair_direction_reason",
    "matrix_csv",
    "preview_qc_status",
    "preview_qc_failures",
    "row_label_dependency_signal",
    "right_page_no_row_labels_signal",
    "row_alignment_evidence",
    "row_pairing_mode",
    "row_label_source_summary",
    "row_count",
    "block_count",
    "cell_count",
    "column_order_evidence",
    "zero_check_status",
    "zero_check_failures",
    "left_title",
    "right_title",
    "title_pair_signal",
    "reason",
    "review_checklist",
    "physical_merge_allowed",
    "safe_to_publish_values",
]
GROUPING_PROPOSAL_FIELDS = [
    "proposal_rank",
    "proposal_id",
    "proposal_type",
    "proposal_status",
    "source_key",
    "source_pdf_id",
    "page_range",
    "component_pages",
    "pdf_pages_sorted",
    "pdf_page_start",
    "pdf_page_end",
    "page_span_size",
    "n_pages",
    "pages_consecutive",
    "physical_merge_allowed",
    "logical_collection_only",
    "site_ready_physical_structure",
    "site_ready_logical_collection",
    "review_required",
    "safe_to_publish_values",
    "matrix_csv",
    "member_count",
    "priority_bucket",
    "review_focus",
    "confidence_score",
    "title_summary",
    "row_label_signal",
    "right_page_no_row_labels_signal",
    "pair_direction_signal",
    "pair_direction_reason",
    "row_pairing_mode",
    "row_label_source_summary",
    "column_order_evidence",
    "zero_check_status",
    "artifact_status",
    "decision_source",
    "recommended_action",
    "promotion_condition",
    "reason",
    "review_flags",
]
GROUPING_PROPOSAL_MEMBER_FIELDS = [
    "proposal_id",
    "member_rank",
    "member_role",
    "source_key",
    "source_pdf_id",
    "pdf_page",
    "expanded_pages",
    "image_path",
    "source_csv",
    "source_json",
    "table_number_key",
    "topic_variant_key",
    "religion_terms_detected",
    "title",
    "member_source_id",
]
MISSED_GROUPING_WATCHLIST_FIELDS = [
    "watch_rank",
    "watch_bucket",
    "watch_priority",
    "edge_key",
    "source_key",
    "source_pdf_id",
    "page_pair",
    "pdf_pages_sorted",
    "pdf_page_start",
    "pdf_page_end",
    "pages_consecutive",
    "page_gap",
    "edge_decision_status",
    "grouping_policy",
    "already_in_grouping_proposals",
    "row_label_dependency_signal",
    "right_page_no_row_labels",
    "same_table_family",
    "same_table_number",
    "title_switch_signal",
    "table_family_transition",
    "table_number_transition",
    "page_class_transition",
    "classifier_break_signal",
    "source_decision_layer",
    "component_ids",
    "image_paths",
    "decision_reason",
    "rule_flags",
    "recommended_action",
    "promotion_condition",
]
ASSEMBLY_WORK_QUEUE_FIELDS = [
    "queue_rank",
    "queue_bucket",
    "queue_priority",
    "record_type",
    "proposal_id",
    "proposal_type",
    "proposal_status",
    "source_key",
    "source_pdf_id",
    "page_range",
    "component_pages",
    "pdf_pages_sorted",
    "pdf_page_start",
    "pdf_page_end",
    "page_span_size",
    "n_pages",
    "pages_consecutive",
    "physical_merge_allowed",
    "logical_collection_only",
    "site_ready_physical_structure",
    "site_ready_logical_collection",
    "safe_to_publish_values",
    "matrix_csv",
    "member_count",
    "review_focus",
    "confidence_score",
    "title_summary",
    "row_label_signal",
    "right_page_no_row_labels_signal",
    "column_order_evidence",
    "zero_check_status",
    "artifact_status",
    "recommended_action",
    "promotion_condition",
    "reason",
    "review_flags",
]
ASSEMBLY_PRIORITY_CLASSIFIED_FIELDS = [
    "priority_rank",
    "urgency_rank",
    "core_scope",
    "next_step_type",
    "queue_bucket",
    "queue_priority",
    "record_type",
    "proposal_id",
    "proposal_type",
    "proposal_status",
    "source_key",
    "source_pdf_id",
    "page_range",
    "component_pages",
    "n_pages",
    "pages_consecutive",
    "physical_merge_allowed",
    "logical_collection_only",
    "site_ready_physical_structure",
    "site_ready_logical_collection",
    "right_page_no_row_labels_signal",
    "zero_check_status",
    "artifact_status",
    "title_summary",
    "recommended_action",
    "promotion_condition",
    "reason",
    "review_flags",
    "matrix_csv",
]
WATCHLIST_SOURCE_ROLLUP_FIELDS = [
    "rollup_rank",
    "watch_bucket",
    "watch_priority",
    "source_key",
    "source_pdf_id",
    "edge_count",
    "right_page_no_row_labels_edges",
    "same_table_family_edges",
    "same_table_number_edges",
    "page_pairs_sample",
    "decision_reasons_sample",
    "recommended_action",
    "promotion_condition",
]
WATCHLIST_FAMILY_ROLLUP_FIELDS = [
    "rollup_rank",
    "watch_bucket",
    "watch_priority",
    "source_key",
    "source_pdf_id",
    "page_class_transition",
    "table_family_transition",
    "table_number_transition",
    "title_switch_signal",
    "edge_count",
    "right_page_no_row_labels_edges",
    "same_table_family_edges",
    "same_table_number_edges",
    "page_pairs_sample",
    "decision_reasons_sample",
    "rule_flags_sample",
    "recommended_action",
    "promotion_condition",
]

NON_CORE_LOGICAL_TITLE_RE = re.compile(
    r"\b("
    r"subsidiary|selected|caste|castes|tribe|tribes|race|races|"
    r"education|scholar|literacy|illiteracy|deaf|mute|mutes|leper|lepers|"
    r"infirmity|infirmities|blind|insane|occupation|house|houses|"
    r"sect|sects|language|languages|mother[- ]tongue|parent[- ]tongue"
    r")\b",
    re.IGNORECASE,
)
MAX_SITE_READY_LOGICAL_SIBLING_PAGES = 4


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def csv_fields(rows: list[dict[str, str]]) -> list[str]:
    if not rows:
        return []
    return list(rows[0].keys())


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str] | None = None) -> None:
    fields = fields or (list(rows[0].keys()) if rows else [])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def read_json(path: Path) -> dict[str, object]:
    if not path.exists() or not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def output_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def int_value(value: object) -> int:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return 0


def audit_failure_count(audit_rows: Iterable[dict[str, str]]) -> int:
    return sum(1 for row in audit_rows if row.get("status") == "fail")


def audit_warning_count(audit_rows: Iterable[dict[str, str]]) -> int:
    return sum(1 for row in audit_rows if row.get("status") == "warn")


def counter_rows(metric: str, counter: Counter[str]) -> list[dict[str, object]]:
    return [
        {"metric": metric, "name": key or "(blank)", "value": count, "detail": ""}
        for key, count in sorted(counter.items())
    ]


def intake_status_count(intake_summary: dict[str, object], status: str) -> int:
    counts = intake_summary.get("targets_by_unlock_status")
    if isinstance(counts, dict):
        return int_value(counts.get(status))
    return int_value(intake_summary.get(status))


def split_list(value: str) -> list[str]:
    return [part.strip() for part in str(value or "").split(";") if part.strip()]


def page_sort_key(value: str) -> tuple[int, object]:
    try:
        return (0, int(str(value)))
    except (TypeError, ValueError):
        return (1, str(value))


def is_site_ready_frontier_edge(row: dict[str, str]) -> bool:
    return row.get("safe_physical_edge_now") == "yes" or row.get("frontier_category", "").startswith("site_ready")


def edge_pages(row: dict[str, str]) -> tuple[str, str, str]:
    source_key = row.get("v5_source_key") or row.get("source_pdf_id_guess") or row.get("edge_key", "").split("||")[0]
    pair = row.get("page_pair", "")
    if "-" in pair:
        left, right = pair.split("-", 1)
        return source_key, left.strip(), right.strip()
    parts = row.get("edge_key", "").split("||")
    if len(parts) >= 3:
        return source_key or parts[0], parts[-2].strip(), parts[-1].strip()
    return source_key, "", ""


def pages_are_consecutive(pages: list[str]) -> bool:
    if len(pages) < 2:
        return True
    try:
        nums = sorted(int(page) for page in pages)
    except ValueError:
        return False
    return all(right - left == 1 for left, right in zip(nums, nums[1:]))


def build_physical_chain_audit(frontier_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    """Audit connected chains in the site-ready physical frontier.

    The current materializer is intentionally edge-first. A connected component
    larger than two pages means overlapping safe edges would create a multi-page
    chain, so the release should stop and require explicit multi-page handling
    instead of silently publishing overlapping two-page assemblies.
    """
    safe_edges = [row for row in frontier_rows if is_site_ready_frontier_edge(row)]
    adjacency: dict[tuple[str, str], set[tuple[str, str]]] = defaultdict(set)
    node_edges: dict[tuple[str, str], list[str]] = defaultdict(list)
    edge_lookup: dict[str, dict[str, str]] = {}

    for row in safe_edges:
        source_key, left, right = edge_pages(row)
        edge_key = row.get("edge_key") or f"{source_key}||{left}||{right}"
        edge_lookup[edge_key] = row
        if not source_key or not left or not right:
            continue
        left_node = (source_key, left)
        right_node = (source_key, right)
        adjacency[left_node].add(right_node)
        adjacency[right_node].add(left_node)
        node_edges[left_node].append(edge_key)
        node_edges[right_node].append(edge_key)

    rows: list[dict[str, object]] = []
    seen: set[tuple[str, str]] = set()
    component_rank = 0
    for start in sorted(adjacency, key=lambda node: (node[0], page_sort_key(node[1]))):
        if start in seen:
            continue
        component_rank += 1
        stack = [start]
        seen.add(start)
        nodes: list[tuple[str, str]] = []
        component_edges: set[str] = set()
        while stack:
            node = stack.pop()
            nodes.append(node)
            component_edges.update(node_edges.get(node, []))
            for neighbor in adjacency.get(node, set()):
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)

        source_keys = sorted({source for source, _ in nodes})
        pages = sorted({page for _, page in nodes}, key=page_sort_key)
        issues: list[str] = []
        if len(source_keys) != 1:
            issues.append("multiple_source_keys")
        if not pages_are_consecutive(pages):
            issues.append("component_pages_not_consecutive")
        if len(nodes) > 2:
            issues.append("connected_safe_edge_chain_larger_than_two_requires_explicit_multi_page_materializer")
        if len(component_edges) != max(0, len(nodes) - 1):
            issues.append("safe_edge_component_not_simple_path")
        status = "pass_single_edge_pair" if not issues else "fail_review_before_release"
        rows.append(
            {
                "component_id": f"safe_chain_{component_rank:04d}",
                "status": status,
                "source_key": "; ".join(source_keys),
                "component_pages": "; ".join(pages),
                "component_size": len(nodes),
                "edge_count": len(component_edges),
                "edge_keys": "; ".join(sorted(component_edges)),
                "issues": "; ".join(issues),
            }
        )

    # Preserve malformed safe edges as explicit audit failures instead of
    # dropping them because they could not be added to the graph.
    graphed_edges = {edge for row in rows for edge in split_list(str(row.get("edge_keys", "")))}
    for edge_key, row in sorted(edge_lookup.items()):
        if edge_key in graphed_edges:
            continue
        source_key, left, right = edge_pages(row)
        rows.append(
            {
                "component_id": f"safe_chain_malformed_{len(rows) + 1:04d}",
                "status": "fail_review_before_release",
                "source_key": source_key,
                "component_pages": "; ".join(part for part in [left, right] if part),
                "component_size": 0,
                "edge_count": 1,
                "edge_keys": edge_key,
                "issues": "could_not_parse_safe_edge_pages",
            }
        )
    return rows


def build_grouping_safety_audit(
    frontier_rows: list[dict[str, str]],
    physical_rows: list[dict[str, str]],
    logical_site_ready: list[dict[str, str]],
    chain_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    safe_edges = [row for row in frontier_rows if is_site_ready_frontier_edge(row)]
    nonconsecutive_safe = [row.get("edge_key", "") for row in safe_edges if row.get("pages_consecutive") != "yes"]
    missing_artifact_safe = [
        row.get("edge_key", "")
        for row in safe_edges
        if "missing_ocr_artifact" in row.get("risk_flags", "")
        or row.get("left_artifact_status") == "missing_ocr_artifact"
        or row.get("right_artifact_status") == "missing_ocr_artifact"
    ]
    duplicate_vertical_safe = [
        row.get("edge_key", "")
        for row in safe_edges
        if row.get("column_order_evidence") == "same_column_groups_vertical_or_duplicate_review"
        or "duplicate_or_vertical_column_groups" in row.get("risk_flags", "")
    ]
    physical_nonconsecutive = [
        row.get("assembly_id") or row.get("source_group_id", "")
        for row in physical_rows
        if row.get("pages_consecutive") != "yes"
    ]
    physical_missing_merge = [
        row.get("assembly_id") or row.get("source_group_id", "")
        for row in physical_rows
        if row.get("physical_merge_allowed") != "yes"
    ]
    chain_failures = [row.get("component_id", "") for row in chain_rows if str(row.get("status", "")).startswith("fail")]

    logical_physical_merge = [
        row.get("sibling_set_id", "")
        for row in logical_site_ready
        if row.get("physical_merge_allowed") != "no"
    ]
    logical_nonconsecutive = [
        row.get("sibling_set_id", "")
        for row in logical_site_ready
        if row.get("pages_are_consecutive") != "yes"
    ]
    logical_title_ref_problem: list[str] = []
    logical_noncore_title: list[str] = []
    for row in logical_site_ready:
        refs = set(split_list(row.get("title_table_refs", "")))
        table_number_key = row.get("table_number_key", "")
        if not refs or row.get("nonmatching_title_table_refs") or (table_number_key and table_number_key not in refs):
            logical_title_ref_problem.append(row.get("sibling_set_id", ""))
        if NON_CORE_LOGICAL_TITLE_RE.search(row.get("title_sample", "")):
            logical_noncore_title.append(row.get("sibling_set_id", ""))

    def row(check: str, failures: list[str], detail_prefix: str) -> dict[str, object]:
        detail = f"{detail_prefix}; count={len(failures)}"
        if failures:
            detail += f"; examples={'; '.join(failures[:10])}"
        return {"check": check, "status": "pass" if not failures else "fail", "detail": detail}

    return [
        row("site_ready_frontier_edges_are_consecutive", nonconsecutive_safe, "Safe physical edges must be adjacent in real PDF order"),
        row("site_ready_frontier_edges_have_ocr_artifacts", missing_artifact_safe, "Safe physical edges cannot depend on missing OCR artifacts"),
        row("site_ready_frontier_edges_not_duplicate_vertical", duplicate_vertical_safe, "Duplicate/vertical same-column continuations stay review-only"),
        row("safe_edge_graph_has_no_oversized_chains", chain_failures, "Current release supports isolated pairwise physical components only"),
        row("materialized_physical_pages_are_consecutive", physical_nonconsecutive, "Materialized physical assemblies must remain consecutive"),
        row("materialized_physical_merge_allowed", physical_missing_merge, "Materialized rows must be explicitly physical-merge allowed"),
        row("logical_ready_is_logical_only", logical_physical_merge, "Logical sibling sets must never become physical merges"),
        row("logical_ready_pages_are_consecutive", logical_nonconsecutive, "Logical sibling sets must be bounded to consecutive source pages"),
        row("logical_ready_has_consistent_title_table_refs", logical_title_ref_problem, "Logical-ready title/table references must be visible and consistent"),
        row("logical_ready_excludes_non_core_titles", logical_noncore_title, "Derivative/non-core titles stay review-only"),
    ]


def build_decision_ledger(
    physical_rows: list[dict[str, str]],
    physical_matrix_files: list[dict[str, str]],
    manual_preview_rows: list[dict[str, str]] | None,
    manual_preview_matrix_files: list[dict[str, str]] | None,
    logical_site_ready: list[dict[str, str]],
    logical_review: list[dict[str, str]],
    logical_members: list[dict[str, str]],
    logical_family_quality: list[dict[str, str]],
    religion_sequence_coverage: list[dict[str, str]],
    physical_manual_frontier: list[dict[str, str]],
    physical_missing_ocr: list[dict[str, str]],
    physical_missing_pages: list[dict[str, str]],
    physical_conflicts: list[dict[str, str]],
) -> list[dict[str, object]]:
    """Flatten current grouping decisions into one user-facing ledger.

    The release already carries specialized frontier sheets. This ledger is the
    stable high-level record: what can be assembled now, what is only a logical
    collection, and what must wait for OCR, missing pages, or manual review.
    """
    matrix_by_assembly = {row.get("assembly_id", ""): row for row in physical_matrix_files}
    manual_preview_rows = manual_preview_rows or []
    manual_preview_matrix_files = manual_preview_matrix_files or []
    preview_matrix_by_assembly = {row.get("assembly_id", ""): row for row in manual_preview_matrix_files}
    preview_by_edge = {row.get("source_group_id", ""): row for row in manual_preview_rows}
    rows: list[dict[str, object]] = []

    def append(row: dict[str, object]) -> None:
        row["ledger_rank"] = len(rows) + 1
        rows.append(row)

    for physical in physical_rows:
        matrix = matrix_by_assembly.get(physical.get("assembly_id", ""), {})
        append(
            {
                "decision_type": "physical_materialized_now",
                "decision_status": physical.get("assembly_status", ""),
                "assembly_or_edge_id": physical.get("assembly_id", ""),
                "source_key": physical.get("v5_source_key", ""),
                "source_pdf_id": physical.get("source_pdf_id_guess", ""),
                "page_range": physical.get("page_pair", ""),
                "component_pages": physical.get("component_pages", ""),
                "pages_consecutive": physical.get("pages_consecutive", ""),
                "grouping_policy": "physical_pairwise_component",
                "site_use_policy": physical.get("site_use_policy", ""),
                "safe_to_publish_values": physical.get("safe_to_publish_values", ""),
                "review_required": "yes" if physical.get("assembly_status") != "materialized_physical_structure" or physical.get("value_review_required") == "yes" else "no",
                "priority": physical.get("assembly_rank", ""),
                "row_label_signal": physical.get("row_alignment_evidence", ""),
                "right_page_no_row_labels_signal": "",
                "pair_direction_signal": "",
                "pair_direction_reason": "",
                "row_pairing_mode": physical.get("row_pairing_mode", ""),
                "row_label_source_summary": physical.get("row_label_source_summary", ""),
                "row_count_detail": f"rows={physical.get('row_count', '')}; blocks={physical.get('block_count', '')}; cells={physical.get('cell_count', '')}",
                "column_order": matrix.get("column_order") or physical.get("assembled_column_order", ""),
                "column_order_evidence": physical.get("column_order_evidence", ""),
                "zero_check_status": physical.get("zero_check_status", ""),
                "artifact_status": "left=present; right=present",
                "title_summary": f"left={physical.get('left_title', '')}; right={physical.get('right_title', '')}",
                "image_paths": f"left={physical.get('left_image_path', '')}; right={physical.get('right_image_path', '')}",
                "matrix_csv": matrix.get("matrix_csv", ""),
                "recommended_action": physical.get("recommended_action", ""),
                "decision_reason": physical.get("frontier_category", ""),
                "review_flags": physical.get("review_flags", ""),
            }
        )

    for logical in logical_site_ready:
        append(
            {
                "decision_type": "logical_sibling_collection_now",
                "decision_status": logical.get("sibling_set_status", ""),
                "assembly_or_edge_id": logical.get("sibling_set_id", ""),
                "source_key": logical.get("v5_source_key", ""),
                "source_pdf_id": logical.get("source_pdf_id", ""),
                "page_range": logical.get("page_range", ""),
                "component_pages": logical.get("component_pages", ""),
                "pages_consecutive": logical.get("pages_are_consecutive", ""),
                "grouping_policy": "logical_collection_only_never_physical_merge",
                "site_use_policy": logical.get("site_use_policy", ""),
                "safe_to_publish_values": logical.get("safe_to_publish_values", ""),
                "review_required": logical.get("review_required", ""),
                "priority": logical.get("priority", ""),
                "row_label_signal": "",
                "right_page_no_row_labels_signal": "",
                "row_pairing_mode": "",
                "row_label_source_summary": "",
                "row_count_detail": f"collections={logical.get('collection_count', '')}; member_pages={logical.get('member_page_count', '')}",
                "column_order": "",
                "column_order_evidence": "logical_only_same_parent_table_topic",
                "zero_check_status": "",
                "artifact_status": "",
                "title_summary": logical.get("title_sample", ""),
                "image_paths": "",
                "matrix_csv": "",
                "recommended_action": logical.get("recommended_action", ""),
                "decision_reason": logical.get("review_reason", ""),
                "review_flags": f"religions={logical.get('religion_terms_detected', '')}; title_refs={logical.get('title_table_refs', '')}",
            }
        )

    site_ready_logical_ids = {row.get("parent_family_id", "") for row in logical_site_ready}
    for logical in logical_family_quality:
        audit_bucket = logical.get("audit_bucket", "")
        confirmed = audit_bucket == "confirmed_logical_family"
        review_required = "no" if confirmed and logical.get("family_id", "") in site_ready_logical_ids else "yes"
        append(
            {
                "decision_type": "logical_family_quality_record",
                "decision_status": audit_bucket,
                "assembly_or_edge_id": logical.get("group_id") or logical.get("family_id", ""),
                "source_key": logical.get("v5_source_key", ""),
                "source_pdf_id": logical.get("source_pdf_id_guess", ""),
                "page_range": logical.get("page_range", ""),
                "component_pages": logical.get("page_range", ""),
                "pages_consecutive": logical.get("pages_are_consecutive", ""),
                "grouping_policy": "logical_collection_only_quality_record" if confirmed else "logical_review_only_candidate",
                "site_use_policy": "use_only_if_site_ready_logical_sheet_lists_it" if confirmed else "review_or_split_before_site_use",
                "safe_to_publish_values": "no",
                "review_required": review_required,
                "priority": logical.get("priority", ""),
                "row_label_signal": "",
                "right_page_no_row_labels_signal": "",
                "row_pairing_mode": "",
                "row_label_source_summary": "",
                "row_count_detail": f"pages={logical.get('n_pages', '')}; members={logical.get('member_count', '')}",
                "column_order": "",
                "column_order_evidence": "logical_family_quality_audit",
                "zero_check_status": "",
                "artifact_status": "",
                "title_summary": logical.get("primary_title", ""),
                "image_paths": "",
                "matrix_csv": "",
                "recommended_action": logical.get("recommended_action", ""),
                "decision_reason": audit_bucket,
                "review_flags": f"religions={logical.get('religion_terms_detected', '')}; overbroad={logical.get('overbroad_risk', '')}; missing_family={logical.get('missing_family_risk', '')}",
            }
        )

    for sequence in religion_sequence_coverage:
        covered = sequence.get("coverage_bucket") == "covered_religion_sequence"
        append(
            {
                "decision_type": "religion_sequence_coverage_record",
                "decision_status": sequence.get("coverage_bucket", ""),
                "assembly_or_edge_id": sequence.get("sequence_id", ""),
                "source_key": sequence.get("v5_source_key", ""),
                "source_pdf_id": sequence.get("source_pdf_id_guess", ""),
                "page_range": sequence.get("page_range", ""),
                "component_pages": sequence.get("component_pages", ""),
                "pages_consecutive": "yes",
                "grouping_policy": "covered_logical_religion_sequence" if covered else "logical_religion_sequence_review_only",
                "site_use_policy": "covered_by_existing_logical_group" if covered else "review_before_site_use",
                "safe_to_publish_values": "no",
                "review_required": "no" if covered else "yes",
                "priority": sequence.get("sequence_rank", ""),
                "row_label_signal": "",
                "right_page_no_row_labels_signal": "",
                "row_pairing_mode": "",
                "row_label_source_summary": "",
                "row_count_detail": f"pages={sequence.get('n_pages', '')}; religions={sequence.get('distinct_religion_count', '')}",
                "column_order": "",
                "column_order_evidence": f"same_table_frame={sequence.get('same_table_frame', '')}; table_family={sequence.get('table_family_key', '')}; table_number={sequence.get('table_number_key', '')}",
                "zero_check_status": "",
                "artifact_status": "",
                "title_summary": sequence.get("titles", ""),
                "image_paths": "",
                "matrix_csv": "",
                "recommended_action": sequence.get("recommended_action", ""),
                "decision_reason": sequence.get("coverage_bucket", ""),
                "review_flags": f"religions={sequence.get('religion_terms_detected', '')}; covered_groups={sequence.get('covered_logical_group_ids', '')}",
            }
        )

    def frontier_append(decision_type: str, decision_status: str, grouping_policy: str, frontier: dict[str, str]) -> None:
        preview = preview_by_edge.get(frontier.get("edge_key", ""), {})
        preview_matrix = preview_matrix_by_assembly.get(preview.get("assembly_id", ""), {})
        review_flags = frontier.get("risk_flags") or frontier.get("rule_flags", "")
        if preview.get("review_flags"):
            review_flags = "; ".join(part for part in [review_flags, preview.get("review_flags", "")] if part)
        append(
            {
                "decision_type": decision_type,
                "decision_status": preview.get("assembly_status", "") or decision_status,
                "assembly_or_edge_id": preview.get("assembly_id", "") or frontier.get("edge_key", ""),
                "source_key": frontier.get("v5_source_key", ""),
                "source_pdf_id": frontier.get("source_pdf_id_guess", ""),
                "page_range": frontier.get("page_pair", ""),
                "component_pages": frontier.get("page_pair", "").replace("-", "; "),
                "pages_consecutive": frontier.get("pages_consecutive", ""),
                "grouping_policy": grouping_policy,
                "site_use_policy": preview.get("site_use_policy", "") or "do_not_use_as_physical_group_yet",
                "safe_to_publish_values": "no",
                "review_required": frontier.get("manual_review_required", ""),
                "priority": frontier.get("frontier_priority", ""),
                "row_label_signal": frontier.get("row_label_dependency_signal", ""),
                "right_page_no_row_labels_signal": frontier.get("right_page_no_row_labels_signal", ""),
                "row_pairing_mode": preview.get("row_pairing_mode", "") or frontier.get("row_alignment_evidence", ""),
                "row_label_source_summary": preview.get("row_label_source_summary", "") or f"left_labels={frontier.get('left_row_label_count', '')}; right_labels={frontier.get('right_row_label_count', '')}; shared={frontier.get('shared_row_label_count', '')}",
                "row_count_detail": f"rows={preview.get('row_count', '')}; blocks={preview.get('block_count', '')}; cells={preview.get('cell_count', '')}" if preview else f"left_rows={frontier.get('left_row_count', '')}; right_rows={frontier.get('right_row_count', '')}",
                "column_order": preview_matrix.get("column_order", "") or preview.get("assembled_column_order", "") or frontier.get("combined_column_group_order", ""),
                "column_order_evidence": frontier.get("column_order_evidence", ""),
                "zero_check_status": frontier.get("zero_check_status", ""),
                "artifact_status": f"left={frontier.get('left_artifact_status', '')}; right={frontier.get('right_artifact_status', '')}",
                "title_summary": f"left={frontier.get('left_title', '')}; right={frontier.get('right_title', '')}",
                "image_paths": f"left={frontier.get('prev_image_path', '')}; right={frontier.get('image_path', '')}",
                "matrix_csv": preview_matrix.get("matrix_csv", ""),
                "recommended_action": preview.get("recommended_action", "") or frontier.get("frontier_action") or frontier.get("recommended_next_step", ""),
                "decision_reason": frontier.get("frontier_reason") or frontier.get("rule_decision_reason", ""),
                "review_flags": review_flags,
            }
        )

    for frontier in physical_manual_frontier:
        frontier_append("manual_review_candidate_adjacent_edge", "review_required", "possible_physical_pair_after_manual_adjudication", frontier)
    for frontier in physical_missing_ocr:
        frontier_append("wait_for_ocr_then_retest", "blocked_missing_ocr", "no_physical_merge_until_ocr_artifact_exists_and_edge_retests", frontier)
    for frontier in physical_missing_pages:
        frontier_append("recover_missing_pages_then_retest", "blocked_missing_intermediate_pages", "no_physical_merge_across_gap", frontier)
    for frontier in physical_conflicts:
        frontier_append("do_not_merge_conflict_edge", "blocked_structural_or_semantic_conflict", "do_not_merge_without_explicit_override", frontier)

    return rows


def edge_grouping_policy(edge: dict[str, str]) -> str:
    if edge.get("physical_merge_allowed_now") == "yes":
        return "physical_merge_allowed_now"
    if edge.get("likely_after_missing_pages") == "yes":
        return "no_merge_missing_pages_then_retest"
    if edge.get("likely_after_ocr") == "yes":
        return "no_merge_until_ocr_artifact_then_retest"
    if edge.get("logical_group_allowed_now") == "yes":
        return "logical_only_edge"
    if edge.get("requires_manual_review") == "yes":
        return "manual_review_edge"
    return "do_not_merge"


def build_edge_decision_ledger(rulebook_edges: list[dict[str, str]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for idx, edge in enumerate(rulebook_edges, start=1):
        rows.append(
            {
                "edge_rank": idx,
                "edge_id": edge.get("rule_id", ""),
                "edge_key": edge.get("edge_key", ""),
                "source_key": edge.get("v5_source_key", ""),
                "source_pdf_id": edge.get("source_pdf_id_guess", ""),
                "page_pair": edge.get("page_pair", ""),
                "pages_consecutive": edge.get("pages_consecutive", ""),
                "page_gap": edge.get("page_gap", ""),
                "edge_decision_status": edge.get("physical_rule_status", ""),
                "grouping_policy": edge_grouping_policy(edge),
                "physical_merge_allowed_now": edge.get("physical_merge_allowed_now", ""),
                "logical_group_allowed_now": edge.get("logical_group_allowed_now", ""),
                "requires_manual_review": edge.get("requires_manual_review", ""),
                "requires_value_review": edge.get("requires_value_review", ""),
                "likely_after_missing_pages": edge.get("likely_after_missing_pages", ""),
                "likely_after_ocr": edge.get("likely_after_ocr", ""),
                "site_action": edge.get("site_action", ""),
                "promotion_condition": edge.get("promotion_condition", ""),
                "row_label_dependency_signal": edge.get("row_label_dependency_signal", ""),
                "right_page_no_row_labels": edge.get("right_page_no_row_labels", ""),
                "right_page_parity": edge.get("right_page_parity", ""),
                "left_row_label_ratio": edge.get("left_row_label_ratio", ""),
                "right_row_label_ratio": edge.get("right_row_label_ratio", ""),
                "row_label_scan_strength": edge.get("row_label_scan_strength", ""),
                "row_label_scan_pattern": edge.get("row_label_scan_pattern", ""),
                "page_kind_transition": edge.get("page_kind_transition", ""),
                "page_class_transition": edge.get("page_class_transition", ""),
                "classifier_break_signal": edge.get("classifier_break_signal", ""),
                "same_table_family": edge.get("same_table_family", ""),
                "same_table_number": edge.get("same_table_number", ""),
                "table_family_transition": f"{edge.get('prev_table_family_key', '')} -> {edge.get('table_family_key', '')}",
                "table_number_transition": f"{edge.get('prev_table_number_key', '')} -> {edge.get('table_number_key', '')}",
                "title_switch_signal": edge.get("title_switch_signal", ""),
                "decision_reason": edge.get("decision_reason", ""),
                "rule_flags": edge.get("rule_flags", ""),
                "source_decision_layer": edge.get("source_decision_layer", ""),
                "frontier_statuses": edge.get("frontier_statuses", ""),
                "frontier_action_buckets": edge.get("frontier_action_buckets", ""),
                "component_ids": edge.get("component_ids", ""),
                "image_paths": f"left={edge.get('prev_image_path', '')}; right={edge.get('image_path', '')}",
            }
        )
    return rows


def frontier_queue_bucket(row: dict[str, str]) -> tuple[int, str, str, str]:
    category = row.get("frontier_category", "")
    if category.startswith("site_ready_physical_edge"):
        return (
            1,
            "current_safe_physical_edge",
            "Already safe for structure assembly; values still follow the value-review policy.",
            "use_existing_materialized_structure",
        )
    if category == "manual_visual_review_row_label_dependency_high":
        return (
            2,
            "manual_review_row_label_dependency_high",
            "Visually inspect first; promote only if the right page is the same horizontal/continuation table and column order continues.",
            "visual_confirmation_required_before_physical_merge",
        )
    if category == "manual_visual_review_likely_civil_status_continuation":
        return (
            2,
            "manual_review_likely_civil_status_continuation",
            "High-value visual review; likely Table VII population/unmarried to married/widowed continuation.",
            "visual_confirmation_required_before_physical_merge",
        )
    if category == "manual_visual_review_row_label_dependency_medium":
        return (
            3,
            "manual_review_row_label_dependency_medium",
            "Review row-label inheritance and table continuity; keep separate if the title/table/subtable changes.",
            "visual_confirmation_required_before_physical_merge",
        )
    if category.startswith("manual_visual_review"):
        return (
            4,
            "manual_review_low_confidence_physical_candidate",
            "Review only after higher-priority row-label candidates; promote only with clear same-table evidence.",
            "visual_confirmation_required_before_physical_merge",
        )
    if category == "needs_missing_ocr_artifact":
        return (
            5,
            "wait_for_ocr_then_retest_adjacent_edge",
            "Do not merge now; rerun adjacent-edge tests after the missing OCR artifact is present.",
            "ocr_artifact_required_then_retest",
        )
    if category == "likely_after_missing_pages_no_merge":
        return (
            6,
            "recover_missing_pages_then_retest",
            "Do not bridge the gap; recover intermediate page(s), OCR them, then test only adjacent page-pairs.",
            "missing_intermediate_pages_required_then_retest",
        )
    if category == "do_not_merge_now_structural_or_semantic_conflict":
        return (
            9,
            "do_not_merge_conflict_reference",
            "Keep separate unless a future OCR/manual correction changes the underlying evidence.",
            "no_current_physical_merge",
        )
    return (
        8,
        "review_uncategorized_frontier_edge",
        "Inspect before use; no automatic physical merge is implied.",
        "manual_triage_required",
    )


def build_physical_frontier_action_queue(
    frontier_rows: list[dict[str, str]],
    edge_decision_ledger: list[dict[str, object]],
) -> list[dict[str, object]]:
    edge_by_key = {str(row.get("edge_key", "")): row for row in edge_decision_ledger if row.get("edge_key")}
    keyed_rows: list[tuple[int, int, dict[str, object]]] = []
    for source_rank, row in enumerate(frontier_rows, start=1):
        priority, bucket, action, promotion_condition = frontier_queue_bucket(row)
        edge_row = edge_by_key.get(row.get("edge_key", ""), {})
        row_label_signal = edge_row.get("row_label_dependency_signal") or row.get("row_label_dependency_signal", "")
        right_no_labels = edge_row.get("right_page_no_row_labels") or row.get("right_page_no_row_labels_signal", "")
        queue_row_label_review_signal = (
            "yes"
            if row_label_signal == "yes"
            or right_no_labels == "yes"
            or "row_label_dependency" in bucket
            or "row_label_dependency" in row.get("frontier_category", "")
            else "no"
        )
        record = {
            "queue_rank": 0,
            "queue_bucket": bucket,
            "queue_priority": priority,
            "edge_key": row.get("edge_key", ""),
            "source_key": row.get("v5_source_key", ""),
            "source_pdf_id": row.get("source_pdf_id_guess", ""),
            "page_pair": row.get("page_pair", ""),
            "pages_consecutive": row.get("pages_consecutive", ""),
            "page_gap": row.get("page_gap", ""),
            "missing_intermediate_pages": row.get("missing_intermediate_pages", ""),
            "current_status": row.get("frontier_category", ""),
            "physical_merge_allowed_now": row.get("physical_merge_allowed_now", ""),
            "safe_physical_edge_now": row.get("safe_physical_edge_now", ""),
            "manual_review_required": row.get("manual_review_required", ""),
            "action": action,
            "promotion_condition": promotion_condition,
            "queue_row_label_review_signal": queue_row_label_review_signal,
            "row_label_signal": row_label_signal,
            "right_page_no_row_labels_signal": right_no_labels,
            "right_page_parity": edge_row.get("right_page_parity", ""),
            "row_label_scan_strength": edge_row.get("row_label_scan_strength", ""),
            "row_label_scan_pattern": edge_row.get("row_label_scan_pattern", ""),
            "edge_decision_status": edge_row.get("edge_decision_status", ""),
            "row_alignment_evidence": row.get("row_alignment_evidence", ""),
            "row_counts": f"left={row.get('left_row_count', '')}; right={row.get('right_row_count', '')}",
            "row_label_counts": f"left={row.get('left_row_label_count', '')}; right={row.get('right_row_label_count', '')}",
            "shared_row_label_count": row.get("shared_row_label_count", ""),
            "column_order_evidence": row.get("column_order_evidence", ""),
            "combined_column_group_order": row.get("combined_column_group_order", ""),
            "zero_check_status": row.get("zero_check_status", ""),
            "zero_check_failures": row.get("zero_check_failures", ""),
            "title_summary": f"left={row.get('left_title', '')}; right={row.get('right_title', '')}",
            "artifact_status": f"left={row.get('left_artifact_status', '')}; right={row.get('right_artifact_status', '')}",
            "image_paths": f"left={row.get('prev_image_path', '')}; right={row.get('image_path', '')}",
            "recommended_next_step": row.get("recommended_next_step", ""),
            "rule_reason": row.get("rule_decision_reason", "") or row.get("frontier_reason", ""),
            "review_flags": row.get("risk_flags", "") or row.get("rule_flags", ""),
        }
        keyed_rows.append((priority, source_rank, record))
    rows = [record for _, _, record in sorted(keyed_rows, key=lambda item: (item[0], item[1]))]
    for rank, row in enumerate(rows, start=1):
        row["queue_rank"] = rank
    return rows


def normalize_title_for_compare(value: object) -> str:
    text = re.sub(r"\b(contd|continued|concld|concluded)\b\.?", " ", str(value or "").lower())
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def title_table_ref(value: object) -> str:
    match = re.search(r"\btable\s+([ivxlcdm0-9]+)\b", str(value or ""), re.IGNORECASE)
    return match.group(1).lower() if match else ""


def title_part_ref(value: object) -> str:
    match = re.search(r"\bpart\s*(?:\(|-)?\s*([a-z0-9]+)", str(value or ""), re.IGNORECASE)
    return match.group(1).lower() if match else ""


def title_pair_signal(left_title: object, right_title: object) -> str:
    left = normalize_title_for_compare(left_title)
    right = normalize_title_for_compare(right_title)
    if not left or not right:
        return "missing_title_review"
    left_table = title_table_ref(left_title)
    right_table = title_table_ref(right_title)
    left_part = title_part_ref(left_title)
    right_part = title_part_ref(right_title)
    score = SequenceMatcher(None, left, right).ratio()
    if left == right or score >= 0.90:
        return "same_or_near_same_title"
    if left_table and left_table == right_table and left_part and right_part and left_part != right_part:
        return "same_table_different_subtable_logical"
    if left_table and right_table and left_table != right_table:
        return "table_number_switch_review"
    if "continued" in str(right_title or "").lower() or "concld" in str(right_title or "").lower():
        if score >= 0.55 or (left_table and left_table == right_table):
            return "continued_title"
    if score >= 0.70:
        return "related_title_review"
    return "title_subject_switch_review"


def parse_count_summary(value: object) -> dict[str, int]:
    counts: dict[str, int] = {}
    for part in split_list(str(value or "")):
        if "=" not in part:
            continue
        key, raw = part.split("=", 1)
        counts[key.strip()] = int_value(raw.strip())
    return counts


def build_manual_preview_triage(
    physical_manual_frontier: list[dict[str, str]],
    manual_preview_rows: list[dict[str, str]],
    manual_preview_matrix_files: list[dict[str, str]],
    manual_preview_qc: list[dict[str, str]],
) -> list[dict[str, object]]:
    frontier_by_edge = {row.get("edge_key", ""): row for row in physical_manual_frontier}
    matrix_by_assembly = {row.get("assembly_id", ""): row for row in manual_preview_matrix_files}
    qc_by_assembly: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in manual_preview_qc:
        qc_by_assembly[row.get("assembly_id", "")].append(row)

    triaged: list[tuple[int, int, str, dict[str, object]]] = []
    for preview in manual_preview_rows:
        edge_key = preview.get("source_group_id", "")
        frontier = frontier_by_edge.get(edge_key, {})
        assembly_id = preview.get("assembly_id", "")
        matrix = matrix_by_assembly.get(assembly_id, {})
        qc_rows = qc_by_assembly.get(assembly_id, [])
        qc_failures = [row for row in qc_rows if row.get("status") == "fail"]
        qc_reviews = [row for row in qc_rows if row.get("status") == "review"]
        row_counts = parse_count_summary(preview.get("row_label_source_summary", ""))
        inherited = row_counts.get("left_page_inherited_to_right", 0) + row_counts.get("right_page_inherited_to_left", 0)
        conflicts = row_counts.get("left_page_conflicts_with_right", 0)
        unresolved = row_counts.get("unresolved", 0)
        title_signal = title_pair_signal(preview.get("left_title", ""), preview.get("right_title", ""))
        row_alignment = preview.get("row_alignment_evidence", "")
        column_order = preview.get("column_order_evidence", "")
        zero_status = preview.get("zero_check_status", "")
        category = preview.get("frontier_category", "")
        disjoint_columns = column_order == "left_page_groups_then_right_page_groups_disjoint"
        row_label_signal = frontier.get("row_label_dependency_signal", "")
        right_no_labels = frontier.get("right_page_no_row_labels_signal", "")
        pair_direction = frontier.get("pair_direction_signal", "")
        pair_direction_reason = frontier.get("pair_direction_reason", "")

        if pair_direction == "civil_status_end_to_restart_boundary":
            bucket_rank = 8
            triage_bucket = "likely_boundary_not_physical_merge"
            suggested_action = "Keep separate by default; the column sequence looks like an end-of-table page followed by a new beginning."
            promotion_condition = "Only promote with explicit visual override proving page B is not a new taluk/district/religion segment."
            confidence_score = 10
        elif qc_failures or column_order == "missing_column_group_evidence" or title_signal in {"table_number_switch_review", "title_subject_switch_review"}:
            bucket_rank = 7
            triage_bucket = "reject_or_keep_separate_until_corrected"
            suggested_action = "Keep separate; do not promote unless OCR/title/column evidence is corrected."
            promotion_condition = "Requires corrected OCR or direct visual override proving same physical table."
            confidence_score = 15
        elif column_order == "overlapping_column_groups_review" or title_signal == "same_table_different_subtable_logical":
            bucket_rank = 5
            triage_bucket = "logical_sibling_candidate_not_physical_merge"
            suggested_action = "Treat as possible logical sibling collection; do not physically merge the pages."
            promotion_condition = "Only logical grouping is plausible unless visual review proves horizontal continuation."
            confidence_score = 45
        elif category == "manual_visual_review_row_label_dependency_high" and disjoint_columns and zero_status == "zero_checks_pass":
            bucket_rank = 1
            triage_bucket = "highest_priority_physical_row_label_candidate"
            suggested_action = "Visual-confirm first; if the image shows one horizontal table, promote this consecutive pair."
            promotion_condition = "Promote only after confirming the right page continues columns of the left page with no subject switch."
            confidence_score = 90
        elif category == "manual_visual_review_likely_civil_status_continuation" and disjoint_columns and zero_status == "zero_checks_pass":
            bucket_rank = 1
            triage_bucket = "likely_physical_civil_status_continuation"
            suggested_action = "High-value review: likely two-page Table VII split from population/unmarried columns into married/widowed columns."
            promotion_condition = "Promote if visual review confirms same row universe, same geography/religion, and no table/subtable switch."
            confidence_score = 88
        elif category == "manual_visual_review_row_label_dependency_medium" and disjoint_columns and title_signal in {"same_or_near_same_title", "continued_title", "related_title_review"}:
            bucket_rank = 2
            triage_bucket = "medium_priority_physical_row_label_candidate"
            suggested_action = "Review row-label inheritance and zero-check issues; promote only if the matrix aligns visually."
            promotion_condition = "Confirm row labels and table subject continuity; resolve value/zero-check warnings before publishing values."
            confidence_score = 74 if zero_status == "zero_checks_pass" else 65
        elif disjoint_columns and title_signal in {"same_or_near_same_title", "continued_title"}:
            bucket_rank = 3
            triage_bucket = "possible_physical_same_title_candidate"
            suggested_action = "Review as possible horizontal continuation; do not promote automatically."
            promotion_condition = "Promote only if columns continue and row labels are inherited correctly across the pair."
            confidence_score = 60
        else:
            bucket_rank = 6
            triage_bucket = "low_confidence_manual_review"
            suggested_action = "Inspect after higher-priority candidates; default to separate unless visual evidence is strong."
            promotion_condition = "Needs explicit manual evidence of a consecutive physical table continuation."
            confidence_score = 35

        if right_no_labels == "yes":
            confidence_score += 3
        if inherited:
            confidence_score += min(5, inherited // 25)
        if conflicts or unresolved:
            confidence_score -= min(10, (conflicts + unresolved) // 10)
        confidence_score = max(0, min(99, confidence_score))

        preview_qc_status = "pass"
        if qc_failures:
            preview_qc_status = "fail"
        elif qc_reviews:
            preview_qc_status = "review"
        reason = (
            f"title={title_signal}; row_alignment={row_alignment}; columns={column_order}; "
            f"zero={zero_status}; inherited_labels={inherited}; conflicts={conflicts}; unresolved={unresolved}; "
            f"qc_failures={len(qc_failures)}; qc_reviews={len(qc_reviews)}"
        )
        checklist = (
            "Confirm pages are physically adjacent in the PDF; confirm the right page is column continuation, "
            "not a new subtable; verify row labels inherit from the labeled page; check zero/value warnings before publishing values."
        )
        record = {
            "triage_rank": "",
            "triage_bucket": triage_bucket,
            "suggested_action": suggested_action,
            "promotion_condition": promotion_condition,
            "confidence_score": confidence_score,
            "edge_key": edge_key,
            "assembly_id": assembly_id,
            "source_key": preview.get("v5_source_key", "") or frontier.get("v5_source_key", ""),
            "source_pdf_id": preview.get("source_pdf_id_guess", "") or frontier.get("source_pdf_id_guess", ""),
            "page_pair": preview.get("page_pair", "") or frontier.get("page_pair", ""),
            "pages_consecutive": preview.get("pages_consecutive", "") or frontier.get("pages_consecutive", ""),
            "frontier_category": category,
            "assembly_status": preview.get("assembly_status", ""),
            "pair_direction_signal": pair_direction,
            "pair_direction_reason": pair_direction_reason,
            "matrix_csv": matrix.get("matrix_csv", ""),
            "preview_qc_status": preview_qc_status,
            "preview_qc_failures": "; ".join(f"{row.get('check', '')}: {row.get('detail', '')}" for row in qc_failures),
            "row_label_dependency_signal": row_label_signal,
            "right_page_no_row_labels_signal": right_no_labels,
            "row_alignment_evidence": row_alignment,
            "row_pairing_mode": preview.get("row_pairing_mode", ""),
            "row_label_source_summary": preview.get("row_label_source_summary", ""),
            "row_count": preview.get("row_count", ""),
            "block_count": preview.get("block_count", ""),
            "cell_count": preview.get("cell_count", ""),
            "column_order_evidence": column_order,
            "zero_check_status": zero_status,
            "zero_check_failures": preview.get("zero_check_failures", ""),
            "left_title": preview.get("left_title", ""),
            "right_title": preview.get("right_title", ""),
            "title_pair_signal": title_signal,
            "reason": reason,
            "review_checklist": checklist,
            "physical_merge_allowed": preview.get("physical_merge_allowed", ""),
            "safe_to_publish_values": preview.get("safe_to_publish_values", ""),
        }
        source_rank = int_value(preview.get("assembly_rank"))
        triaged.append((bucket_rank, -confidence_score, f"{record['source_key']}||{record['page_pair']}", record))

    rows = [record for _, _, _, record in sorted(triaged, key=lambda item: (item[0], item[1], item[2]))]
    for rank, row in enumerate(rows, start=1):
        row["triage_rank"] = rank
    return rows


def page_list_from_text(value: object) -> list[str]:
    pages: list[str] = []
    for part in re.split(r"[;,]\s*|\s+", str(value or "").strip()):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            left, right = part.split("-", 1)
            left_i = int_value(left)
            right_i = int_value(right)
            if left_i and right_i and right_i >= left_i and right_i - left_i <= 50:
                pages.extend(str(page) for page in range(left_i, right_i + 1))
                continue
        pages.append(part)
    seen: set[str] = set()
    out: list[str] = []
    for page in pages:
        if page and page not in seen:
            seen.add(page)
            out.append(page)
    return out


def page_metadata(page_range: object, component_pages: object = "") -> dict[str, object]:
    """Return explicit, review-friendly page fields for a proposal or edge."""
    pages = page_list_from_text(component_pages) or page_list_from_text(page_range)
    numeric_pages: list[int] = []
    nonnumeric_pages: list[str] = []
    for page in pages:
        try:
            numeric_pages.append(int(str(page)))
        except (TypeError, ValueError):
            nonnumeric_pages.append(str(page))

    sorted_pages = [str(page) for page in sorted(set(numeric_pages))]
    sorted_pages.extend(page for page in nonnumeric_pages if page not in sorted_pages)
    return {
        "pdf_pages_sorted": "; ".join(sorted_pages),
        "pdf_page_start": str(min(numeric_pages)) if numeric_pages else "",
        "pdf_page_end": str(max(numeric_pages)) if numeric_pages else "",
        "page_span_size": len(sorted_pages),
    }


def proposal_review_focus(proposal: dict[str, object]) -> str:
    proposal_type = str(proposal.get("proposal_type", ""))
    priority_bucket = str(proposal.get("priority_bucket", ""))
    row_signal = proposal.get("row_label_signal") == "yes"
    right_no_label = proposal.get("right_page_no_row_labels_signal") == "yes"

    if proposal_type == "physical_pairwise_assembly":
        if proposal.get("review_required") == "yes":
            return "confirm_materialized_structure_and_values"
        return "site_ready_physical_structure"
    if proposal_type == "manual_preview_physical_candidate":
        if row_signal or right_no_label or "row_label" in priority_bucket:
            return "visual_confirm_row_label_inheritance"
        return "visual_confirm_adjacent_same_table"
    if proposal_type == "manual_preview_review_candidate":
        return "manual_preview_review_only"
    if proposal_type == "blocked_missing_ocr_edge":
        if row_signal or right_no_label:
            return "wait_for_ocr_then_retest_row_label_edge"
        return "wait_for_ocr_then_retest_adjacent_edge"
    if proposal_type == "blocked_missing_page_bridge":
        if row_signal or right_no_label:
            return "recover_intermediate_pages_then_retest_row_label_edge"
        return "recover_intermediate_pages_then_retest_edge"
    if proposal_type == "blocked_conflict_edge":
        return "keep_separate_structural_or_semantic_conflict"
    if proposal_type == "logical_pairwise_edge_candidate":
        return "site_ready_logical_only_pairwise"
    if proposal_type == "logical_sibling_collection":
        if proposal.get("site_ready_logical_collection") == "yes":
            return "site_ready_logical_only_collection"
        return "review_logical_collection_bounds"
    return "low_confidence_review"


def enrich_grouping_proposal(row: dict[str, object]) -> dict[str, object]:
    row.update(page_metadata(row.get("page_range", ""), row.get("component_pages", "")))
    row["review_focus"] = proposal_review_focus(row)
    return row


def page_pair_members(row: dict[str, str]) -> list[tuple[str, str, str, str, str]]:
    left = right = ""
    pair = row.get("page_pair", "")
    if "-" in pair:
        left, right = [part.strip() for part in pair.split("-", 1)]
    return [
        ("left_page", left, row.get("left_image_path", "") or row.get("prev_image_path", ""), row.get("left_best_csv", ""), row.get("left_best_json", "")),
        ("right_page", right, row.get("right_image_path", "") or row.get("image_path", ""), row.get("right_best_csv", ""), row.get("right_best_json", "")),
    ]


def add_pair_members(
    member_rows: list[dict[str, object]],
    proposal_id: str,
    source_key: str,
    source_pdf_id: str,
    row: dict[str, str],
) -> None:
    for role, page, image, csv_path, json_path in page_pair_members(row):
        if not page and not image and not csv_path and not json_path:
            continue
        member_rows.append(
            {
                "proposal_id": proposal_id,
                "member_rank": len([m for m in member_rows if m.get("proposal_id") == proposal_id]) + 1,
                "member_role": role,
                "source_key": source_key,
                "source_pdf_id": source_pdf_id,
                "pdf_page": page,
                "expanded_pages": page,
                "image_path": image,
                "source_csv": csv_path,
                "source_json": json_path,
                "table_number_key": "",
                "topic_variant_key": "",
                "religion_terms_detected": "",
                "title": row.get("left_title", "") if role == "left_page" else row.get("right_title", ""),
                "member_source_id": row.get("edge_key", "") or row.get("source_group_id", "") or row.get("assembly_id", ""),
            }
        )


def build_grouping_proposals(
    physical_rows: list[dict[str, str]],
    physical_matrix_files: list[dict[str, str]],
    manual_preview_rows: list[dict[str, str]],
    manual_preview_triage: list[dict[str, object]],
    logical_site_ready: list[dict[str, str]],
    logical_review: list[dict[str, str]],
    logical_members: list[dict[str, str]],
    physical_missing_ocr: list[dict[str, str]],
    physical_missing_pages: list[dict[str, str]],
    physical_conflicts: list[dict[str, str]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    matrix_by_assembly = {row.get("assembly_id", ""): row for row in physical_matrix_files}
    preview_by_assembly = {row.get("assembly_id", ""): row for row in manual_preview_rows}
    logical_members_by_set: dict[str, list[dict[str, str]]] = defaultdict(list)
    for member in logical_members:
        logical_members_by_set[member.get("sibling_set_id", "")].append(member)

    proposals: list[dict[str, object]] = []
    members: list[dict[str, object]] = []

    def add(row: dict[str, object]) -> None:
        enrich_grouping_proposal(row)
        row["proposal_rank"] = len(proposals) + 1
        proposals.append(row)

    def logical_member_rows(proposal_id: str, logical: dict[str, str]) -> None:
        source_members = logical_members_by_set.get(logical.get("sibling_set_id", ""), [])
        for idx, member in enumerate(source_members, start=1):
            members.append(
                {
                    "proposal_id": proposal_id,
                    "member_rank": idx,
                    "member_role": "logical_member_page",
                    "source_key": member.get("v5_source_key", ""),
                    "source_pdf_id": member.get("source_pdf_id", ""),
                    "pdf_page": member.get("pdf_page", ""),
                    "expanded_pages": member.get("expanded_pages", ""),
                    "image_path": member.get("image_path", ""),
                    "source_csv": "",
                    "source_json": "",
                    "table_number_key": member.get("table_number_key", ""),
                    "topic_variant_key": member.get("topic_variant_key", ""),
                    "religion_terms_detected": member.get("religion_terms_detected", ""),
                    "title": member.get("title", ""),
                    "member_source_id": member.get("logical_collection_id", ""),
                }
            )

    for physical in physical_rows:
        proposal_id = f"phys::{physical.get('assembly_id', '')}"
        matrix = matrix_by_assembly.get(physical.get("assembly_id", ""), {})
        add(
            {
                "proposal_id": proposal_id,
                "proposal_type": "physical_pairwise_assembly",
                "proposal_status": physical.get("assembly_status", ""),
                "source_key": physical.get("v5_source_key", ""),
                "source_pdf_id": physical.get("source_pdf_id_guess", ""),
                "page_range": physical.get("page_pair", ""),
                "component_pages": physical.get("component_pages", ""),
                "n_pages": physical.get("n_pages", ""),
                "pages_consecutive": physical.get("pages_consecutive", ""),
                "physical_merge_allowed": physical.get("physical_merge_allowed", ""),
                "logical_collection_only": "no",
                "site_ready_physical_structure": "yes" if physical.get("physical_merge_allowed") == "yes" else "no",
                "site_ready_logical_collection": "no",
                "review_required": "yes" if physical.get("assembly_status") != "materialized_physical_structure" or physical.get("value_review_required") == "yes" else "no",
                "safe_to_publish_values": physical.get("safe_to_publish_values", ""),
                "matrix_csv": matrix.get("matrix_csv", ""),
                "member_count": int_value(physical.get("n_pages")) or len(page_list_from_text(physical.get("component_pages", ""))),
                "priority_bucket": "site_ready_physical_structure",
                "confidence_score": "",
                "title_summary": f"left={physical.get('left_title', '')}; right={physical.get('right_title', '')}",
                "row_label_signal": physical.get("row_alignment_evidence", ""),
                "right_page_no_row_labels_signal": "",
                "row_pairing_mode": physical.get("row_pairing_mode", ""),
                "row_label_source_summary": physical.get("row_label_source_summary", ""),
                "column_order_evidence": physical.get("column_order_evidence", ""),
                "zero_check_status": physical.get("zero_check_status", ""),
                "artifact_status": "left=present; right=present",
                "decision_source": "materialized_physical_all",
                "recommended_action": physical.get("recommended_action", ""),
                "promotion_condition": "already_materialized_pairwise_consecutive_structure",
                "reason": physical.get("frontier_category", ""),
                "review_flags": physical.get("review_flags", ""),
            }
        )
        add_pair_members(members, proposal_id, physical.get("v5_source_key", ""), physical.get("source_pdf_id_guess", ""), physical)

    for triage in manual_preview_triage:
        preview = preview_by_assembly.get(str(triage.get("assembly_id", "")), {})
        proposal_id = f"preview::{triage.get('assembly_id', '')}"
        bucket = str(triage.get("triage_bucket", ""))
        add(
            {
                "proposal_id": proposal_id,
                "proposal_type": "manual_preview_physical_candidate" if "physical" in bucket else "manual_preview_review_candidate",
                "proposal_status": triage.get("triage_bucket", ""),
                "source_key": triage.get("source_key", ""),
                "source_pdf_id": triage.get("source_pdf_id", ""),
                "page_range": triage.get("page_pair", ""),
                "component_pages": str(triage.get("page_pair", "")).replace("-", "; "),
                "n_pages": "2",
                "pages_consecutive": triage.get("pages_consecutive", ""),
                "physical_merge_allowed": "no",
                "logical_collection_only": "no" if "logical_sibling" not in bucket else "yes",
                "site_ready_physical_structure": "no",
                "site_ready_logical_collection": "no",
                "review_required": "yes",
                "safe_to_publish_values": "no",
                "matrix_csv": triage.get("matrix_csv", ""),
                "member_count": "2",
                "priority_bucket": triage.get("triage_bucket", ""),
                "confidence_score": triage.get("confidence_score", ""),
                "title_summary": f"left={triage.get('left_title', '')}; right={triage.get('right_title', '')}",
                "row_label_signal": triage.get("row_label_dependency_signal", ""),
                "right_page_no_row_labels_signal": triage.get("right_page_no_row_labels_signal", ""),
                "pair_direction_signal": triage.get("pair_direction_signal", ""),
                "pair_direction_reason": triage.get("pair_direction_reason", ""),
                "row_pairing_mode": triage.get("row_pairing_mode", ""),
                "row_label_source_summary": triage.get("row_label_source_summary", ""),
                "column_order_evidence": triage.get("column_order_evidence", ""),
                "zero_check_status": triage.get("zero_check_status", ""),
                "artifact_status": "left=present; right=present",
                "decision_source": "manual_physical_preview_triage",
                "recommended_action": triage.get("suggested_action", ""),
                "promotion_condition": triage.get("promotion_condition", ""),
                "reason": triage.get("reason", ""),
                "review_flags": triage.get("review_checklist", ""),
            }
        )
        add_pair_members(members, proposal_id, str(triage.get("source_key", "")), str(triage.get("source_pdf_id", "")), preview)

    for logical, ready in [(row, True) for row in logical_site_ready] + [(row, False) for row in logical_review]:
        proposal_id = f"logical::{logical.get('sibling_set_id', '')}"
        add(
            {
                "proposal_id": proposal_id,
                "proposal_type": "logical_sibling_collection",
                "proposal_status": logical.get("sibling_set_status", ""),
                "source_key": logical.get("v5_source_key", ""),
                "source_pdf_id": logical.get("source_pdf_id", ""),
                "page_range": logical.get("page_range", ""),
                "component_pages": logical.get("component_pages", ""),
                "n_pages": logical.get("n_pages", ""),
                "pages_consecutive": logical.get("pages_are_consecutive", ""),
                "physical_merge_allowed": "no",
                "logical_collection_only": "yes",
                "site_ready_physical_structure": "no",
                "site_ready_logical_collection": "yes" if ready else "no",
                "review_required": logical.get("review_required", ""),
                "safe_to_publish_values": logical.get("safe_to_publish_values", ""),
                "matrix_csv": "",
                "member_count": logical.get("member_page_count", ""),
                "priority_bucket": "site_ready_logical_sibling" if ready else "review_logical_sibling",
                "confidence_score": "",
                "title_summary": logical.get("title_sample", ""),
                "row_label_signal": "",
                "right_page_no_row_labels_signal": "",
                "row_pairing_mode": "",
                "row_label_source_summary": "",
                "column_order_evidence": "logical_only_same_parent_table_topic",
                "zero_check_status": "",
                "artifact_status": "",
                "decision_source": "logical_sibling_sets_site_ready" if ready else "logical_sibling_sets_review",
                "recommended_action": logical.get("recommended_action", ""),
                "promotion_condition": "logical_collection_only_never_physical_merge",
                "reason": logical.get("review_reason", ""),
                "review_flags": f"religions={logical.get('religion_terms_detected', '')}; title_refs={logical.get('title_table_refs', '')}",
            }
        )
        logical_member_rows(proposal_id, logical)

    blocked_specs = [
        ("blocked_missing_ocr_edge", "physical_frontier_missing_ocr_edges", physical_missing_ocr),
        ("blocked_missing_page_bridge", "physical_frontier_missing_page_bridges", physical_missing_pages),
        ("blocked_conflict_edge", "physical_frontier_conflict_edges", physical_conflicts),
    ]
    for proposal_type, source_name, rows in blocked_specs:
        for row in rows:
            proposal_id = f"{proposal_type}::{row.get('edge_key', '')}"
            add(
                {
                    "proposal_id": proposal_id,
                    "proposal_type": proposal_type,
                    "proposal_status": row.get("frontier_category", ""),
                    "source_key": row.get("v5_source_key", ""),
                    "source_pdf_id": row.get("source_pdf_id_guess", ""),
                    "page_range": row.get("page_pair", ""),
                    "component_pages": row.get("page_pair", "").replace("-", "; "),
                    "n_pages": "2",
                    "pages_consecutive": row.get("pages_consecutive", ""),
                    "physical_merge_allowed": "no",
                    "logical_collection_only": "no",
                    "site_ready_physical_structure": "no",
                    "site_ready_logical_collection": "no",
                    "review_required": row.get("manual_review_required", ""),
                    "safe_to_publish_values": "no",
                    "matrix_csv": "",
                    "member_count": "2",
                    "priority_bucket": row.get("frontier_category", ""),
                    "confidence_score": "",
                    "title_summary": f"left={row.get('left_title', '')}; right={row.get('right_title', '')}",
                    "row_label_signal": row.get("row_label_dependency_signal", ""),
                    "right_page_no_row_labels_signal": row.get("right_page_no_row_labels_signal", ""),
                    "row_pairing_mode": row.get("row_alignment_evidence", ""),
                    "row_label_source_summary": f"left_labels={row.get('left_row_label_count', '')}; right_labels={row.get('right_row_label_count', '')}; shared={row.get('shared_row_label_count', '')}",
                    "column_order_evidence": row.get("column_order_evidence", ""),
                    "zero_check_status": row.get("zero_check_status", ""),
                    "artifact_status": f"left={row.get('left_artifact_status', '')}; right={row.get('right_artifact_status', '')}",
                    "decision_source": source_name,
                    "recommended_action": row.get("frontier_action", "") or row.get("recommended_next_step", ""),
                    "promotion_condition": row.get("rule_decision_reason", "") or row.get("frontier_reason", ""),
                    "reason": row.get("frontier_reason", "") or row.get("rule_decision_reason", ""),
                    "review_flags": row.get("risk_flags", "") or row.get("rule_flags", ""),
                }
            )
            add_pair_members(members, proposal_id, row.get("v5_source_key", ""), row.get("source_pdf_id_guess", ""), row)

    return proposals, members


def labeled_path_map(value: object) -> dict[str, str]:
    out: dict[str, str] = {}
    for part in split_list(str(value or "")):
        if "=" not in part:
            continue
        key, raw = part.split("=", 1)
        out[key.strip()] = raw.strip()
    return out


def append_confirmed_logical_edge_proposals(
    proposals: list[dict[str, object]],
    members: list[dict[str, object]],
    edge_decision_ledger: list[dict[str, object]],
) -> None:
    """Add bounded logical-only pairwise edges not already captured by sibling sets.

    These are deliberately not physical assemblies. They are a safety valve for
    confirmed adjacent logical relationships that failed to form a broader,
    bounded logical sibling set.
    """
    covered = proposal_edge_keys(proposals)
    for edge in edge_decision_ledger:
        edge_key = str(edge.get("edge_key", ""))
        if not edge_key or edge_key in covered:
            continue
        if edge.get("logical_group_allowed_now") != "yes" and edge.get("edge_decision_status") != "confirmed_logical_only_edge":
            continue
        if edge.get("grouping_policy") not in {"logical_only_edge", "logical_collection_only", "manual_review_edge"}:
            continue

        proposal_id = f"logical_edge::{edge_key}"
        page_pair = str(edge.get("page_pair", ""))
        pages = page_list_from_text(page_pair)
        proposal = {
            "proposal_rank": len(proposals) + 1,
            "proposal_id": proposal_id,
            "proposal_type": "logical_pairwise_edge_candidate",
            "proposal_status": "confirmed_logical_only_pairwise_edge",
            "source_key": edge.get("source_key", ""),
            "source_pdf_id": edge.get("source_pdf_id", ""),
            "page_range": page_pair,
            "component_pages": "; ".join(pages),
            "n_pages": len(pages),
            "pages_consecutive": edge.get("pages_consecutive", ""),
            "physical_merge_allowed": "no",
            "logical_collection_only": "yes",
            "site_ready_physical_structure": "no",
            "site_ready_logical_collection": "yes",
            "review_required": "no",
            "safe_to_publish_values": "no",
            "matrix_csv": "",
            "member_count": len(pages),
            "priority_bucket": "confirmed_logical_pairwise_edge",
            "confidence_score": "",
            "title_summary": (
                f"same_family={edge.get('same_table_family', '')}; "
                f"same_table_number={edge.get('same_table_number', '')}; "
                f"title_switch={edge.get('title_switch_signal', '')}"
            ),
            "row_label_signal": edge.get("row_label_dependency_signal", ""),
            "right_page_no_row_labels_signal": edge.get("right_page_no_row_labels", ""),
            "row_pairing_mode": "",
            "row_label_source_summary": "",
            "column_order_evidence": "logical_only_pairwise_no_physical_column_merge",
            "zero_check_status": "",
            "artifact_status": "",
            "decision_source": "rulebook_edge_decision_ledger_confirmed_logical_edge",
            "recommended_action": edge.get("site_action", "") or "Use as logical-only pairwise relationship; never physically merge.",
            "promotion_condition": "Already confirmed as bounded logical-only adjacent edge; never a physical merge.",
            "reason": edge.get("decision_reason", ""),
            "review_flags": edge.get("rule_flags", ""),
        }
        enrich_grouping_proposal(proposal)
        proposals.append(proposal)

        image_paths = labeled_path_map(edge.get("image_paths", ""))
        left_page = pages[0] if pages else ""
        right_page = pages[1] if len(pages) > 1 else ""
        for idx, (role, page, image) in enumerate(
            [
                ("left_logical_edge_page", left_page, image_paths.get("left", "")),
                ("right_logical_edge_page", right_page, image_paths.get("right", "")),
            ],
            start=1,
        ):
            if not page and not image:
                continue
            members.append(
                {
                    "proposal_id": proposal_id,
                    "member_rank": idx,
                    "member_role": role,
                    "source_key": edge.get("source_key", ""),
                    "source_pdf_id": edge.get("source_pdf_id", ""),
                    "pdf_page": page,
                    "expanded_pages": page,
                    "image_path": image,
                    "source_csv": "",
                    "source_json": "",
                    "table_number_key": edge.get("same_table_number", ""),
                    "topic_variant_key": edge.get("title_switch_signal", ""),
                    "religion_terms_detected": "",
                    "title": "",
                    "member_source_id": edge_key,
                }
            )
        covered.add(edge_key)


def proposal_edge_keys(grouping_proposals: list[dict[str, object]]) -> set[str]:
    keys: set[str] = set()
    for proposal in grouping_proposals:
        proposal_id = str(proposal.get("proposal_id", ""))
        if "::" in proposal_id:
            keys.add(proposal_id.split("::", 1)[1])
        source_key = str(proposal.get("source_key", ""))
        page_range = str(proposal.get("page_range", ""))
        if source_key and "-" in page_range:
            left, right = [part.strip() for part in page_range.split("-", 1)]
            keys.add(f"{source_key}||{left}||{right}")
        pages = page_list_from_text(proposal.get("component_pages") or page_range)
        if source_key and len(pages) >= 2:
            try:
                sorted_pages = [str(page) for page in sorted(int(page) for page in pages)]
            except ValueError:
                sorted_pages = pages
            for left, right in zip(sorted_pages, sorted_pages[1:]):
                keys.add(f"{source_key}||{left}||{right}")
    return keys


def build_missed_grouping_watchlist(
    edge_decision_ledger: list[dict[str, object]],
    grouping_proposals: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Expose high-signal rulebook edges not represented as grouping proposals.

    Most rulebook edges are hard breaks. This sheet exists to prevent important
    continuation signals from vanishing just because they are not safe/currently
    bounded enough to become a physical or logical grouping proposal.
    """
    covered = proposal_edge_keys(grouping_proposals)
    rows: list[tuple[int, str, dict[str, object]]] = []

    for edge in edge_decision_ledger:
        edge_key = str(edge.get("edge_key", ""))
        in_proposals = edge_key in covered
        status = str(edge.get("edge_decision_status", ""))
        policy = str(edge.get("grouping_policy", ""))
        row_signal = edge.get("row_label_dependency_signal") == "yes"
        right_no_label = edge.get("right_page_no_row_labels") == "yes"
        likely_after_ocr = edge.get("likely_after_ocr") == "yes"
        likely_after_missing = edge.get("likely_after_missing_pages") == "yes"
        physical_now = edge.get("physical_merge_allowed_now") == "yes"
        logical_review = status == "review_logical_only_edge" or (policy == "manual_review_edge" and edge.get("same_table_family") == "yes")
        logical_confirmed = status == "confirmed_logical_only_edge" or edge.get("logical_group_allowed_now") == "yes"

        if in_proposals and not row_signal:
            continue

        if row_signal and not in_proposals:
            priority = 1
            bucket = "uncovered_row_label_dependency_candidate"
            action = "Investigate immediately; row-label dependency edges should be represented in physical/manual proposal layers."
            condition = "Add to proposal layer or document a hard-break reason."
        elif physical_now and not in_proposals:
            priority = 1
            bucket = "uncovered_physical_merge_allowed_edge"
            action = "Investigate immediately; physical merge allowed edge is missing from proposal manifest."
            condition = "Materialize or block explicitly."
        elif (likely_after_ocr or likely_after_missing) and not in_proposals:
            priority = 2
            bucket = "uncovered_blocked_future_candidate"
            action = "Track as blocked; it should be represented in missing-OCR or missing-page proposal layers."
            condition = "Recover OCR/pages, rerun, then retest adjacent edges."
        elif logical_review and not in_proposals:
            priority = 4 if right_no_label else 5
            bucket = "logical_review_edge_not_bounded_as_sibling_set"
            action = "Review as possible logical-only relationship; do not physically merge."
            condition = "Promote only to logical sibling collection if bounded by same source/table/topic and no incompatible intervening edge."
        elif logical_confirmed and not in_proposals:
            priority = 3
            bucket = "confirmed_logical_edge_not_in_sibling_set"
            action = "Check why confirmed logical edge was not captured by a logical sibling set."
            condition = "Add bounded logical sibling proposal or document exclusion."
        else:
            continue

        record = {
            "watch_rank": "",
            "watch_bucket": bucket,
            "watch_priority": priority,
            "edge_key": edge_key,
            "source_key": edge.get("source_key", ""),
            "source_pdf_id": edge.get("source_pdf_id", ""),
            "page_pair": edge.get("page_pair", ""),
            **page_metadata(edge.get("page_pair", "")),
            "pages_consecutive": edge.get("pages_consecutive", ""),
            "page_gap": edge.get("page_gap", ""),
            "edge_decision_status": status,
            "grouping_policy": policy,
            "already_in_grouping_proposals": "yes" if in_proposals else "no",
            "row_label_dependency_signal": edge.get("row_label_dependency_signal", ""),
            "right_page_no_row_labels": edge.get("right_page_no_row_labels", ""),
            "same_table_family": edge.get("same_table_family", ""),
            "same_table_number": edge.get("same_table_number", ""),
            "title_switch_signal": edge.get("title_switch_signal", ""),
            "table_family_transition": edge.get("table_family_transition", ""),
            "table_number_transition": edge.get("table_number_transition", ""),
            "page_class_transition": edge.get("page_class_transition", ""),
            "classifier_break_signal": edge.get("classifier_break_signal", ""),
            "source_decision_layer": edge.get("source_decision_layer", ""),
            "component_ids": edge.get("component_ids", ""),
            "image_paths": edge.get("image_paths", ""),
            "decision_reason": edge.get("decision_reason", ""),
            "rule_flags": edge.get("rule_flags", ""),
            "recommended_action": action,
            "promotion_condition": condition,
        }
        rows.append((priority, str(edge.get("source_key", "")) + "||" + str(edge.get("page_pair", "")), record))

    out = [record for _, _, record in sorted(rows, key=lambda item: (item[0], item[1]))]
    for rank, row in enumerate(out, start=1):
        row["watch_rank"] = rank
    return out


def grouping_work_priority(proposal: dict[str, object]) -> tuple[int, str]:
    proposal_type = str(proposal.get("proposal_type", ""))
    priority_bucket = str(proposal.get("priority_bucket", ""))
    proposal_status = str(proposal.get("proposal_status", ""))
    right_no_label = proposal.get("right_page_no_row_labels_signal") == "yes"
    row_signal = proposal.get("row_label_signal") == "yes"

    if proposal_type == "physical_pairwise_assembly":
        return 1, "01_site_ready_physical_structure_value_review"
    if priority_bucket == "highest_priority_physical_row_label_candidate":
        return 2, "02_visual_confirm_row_label_physical_high"
    if priority_bucket == "medium_priority_physical_row_label_candidate":
        return 3, "03_visual_confirm_row_label_physical_medium"
    if priority_bucket == "possible_physical_same_title_candidate":
        return 4, "04_visual_confirm_possible_physical_same_title"
    if proposal_type == "blocked_missing_ocr_edge":
        if row_signal or right_no_label:
            return 5, "05_ocr_then_retest_row_label_candidate"
        return 6, "06_ocr_then_retest_adjacent_candidate"
    if proposal_type == "blocked_missing_page_bridge":
        if row_signal or right_no_label:
            return 7, "07_recover_missing_pages_then_retest_row_label_bridge"
        return 8, "08_recover_missing_pages_then_retest_bridge"
    if proposal_type == "logical_sibling_collection" and proposal.get("site_ready_logical_collection") == "yes":
        return 9, "09_site_ready_logical_only_collection"
    if proposal_type == "logical_pairwise_edge_candidate":
        return 10, "10_confirmed_pairwise_logical_only_edge"
    if priority_bucket == "logical_sibling_candidate_not_physical_merge":
        return 11, "11_manual_preview_logical_only_candidate"
    if proposal_type == "logical_sibling_collection":
        return 12, "12_review_logical_only_collection"
    if proposal_type == "blocked_conflict_edge":
        return 13, "13_keep_separate_conflict_documented"
    if proposal_status == "reject_or_keep_separate_until_corrected":
        return 14, "14_reject_or_correct_before_any_grouping"
    return 15, "15_low_confidence_review"


def build_assembly_work_queue(grouping_proposals: list[dict[str, object]]) -> list[dict[str, object]]:
    """One sorted action queue over the complete grouping proposal manifest."""
    rows: list[tuple[int, int, str, dict[str, object]]] = []
    for proposal in grouping_proposals:
        priority, bucket = grouping_work_priority(proposal)
        confidence_sort = -int_value(proposal.get("confidence_score"))
        record = {
            "queue_rank": "",
            "queue_bucket": bucket,
            "queue_priority": priority,
            "record_type": "grouping_proposal",
            "proposal_id": proposal.get("proposal_id", ""),
            "proposal_type": proposal.get("proposal_type", ""),
            "proposal_status": proposal.get("proposal_status", ""),
            "source_key": proposal.get("source_key", ""),
            "source_pdf_id": proposal.get("source_pdf_id", ""),
            "page_range": proposal.get("page_range", ""),
            "component_pages": proposal.get("component_pages", ""),
            "pdf_pages_sorted": proposal.get("pdf_pages_sorted", ""),
            "pdf_page_start": proposal.get("pdf_page_start", ""),
            "pdf_page_end": proposal.get("pdf_page_end", ""),
            "page_span_size": proposal.get("page_span_size", ""),
            "n_pages": proposal.get("n_pages", ""),
            "pages_consecutive": proposal.get("pages_consecutive", ""),
            "physical_merge_allowed": proposal.get("physical_merge_allowed", ""),
            "logical_collection_only": proposal.get("logical_collection_only", ""),
            "site_ready_physical_structure": proposal.get("site_ready_physical_structure", ""),
            "site_ready_logical_collection": proposal.get("site_ready_logical_collection", ""),
            "safe_to_publish_values": proposal.get("safe_to_publish_values", ""),
            "matrix_csv": proposal.get("matrix_csv", ""),
            "member_count": proposal.get("member_count", ""),
            "review_focus": proposal.get("review_focus", ""),
            "confidence_score": proposal.get("confidence_score", ""),
            "title_summary": proposal.get("title_summary", ""),
            "row_label_signal": proposal.get("row_label_signal", ""),
            "right_page_no_row_labels_signal": proposal.get("right_page_no_row_labels_signal", ""),
            "column_order_evidence": proposal.get("column_order_evidence", ""),
            "zero_check_status": proposal.get("zero_check_status", ""),
            "artifact_status": proposal.get("artifact_status", ""),
            "recommended_action": proposal.get("recommended_action", ""),
            "promotion_condition": proposal.get("promotion_condition", ""),
            "reason": proposal.get("reason", ""),
            "review_flags": proposal.get("review_flags", ""),
        }
        sort_key = f"{record['source_key']}||{record['page_range']}||{record['proposal_id']}"
        rows.append((priority, confidence_sort, sort_key, record))

    out = [record for _, _, _, record in sorted(rows, key=lambda item: (item[0], item[1], item[2]))]
    for rank, row in enumerate(out, start=1):
        row["queue_rank"] = rank
    return out


def classify_core_scope(row: dict[str, object]) -> str:
    text = " ".join(
        str(row.get(field, "") or "")
        for field in [
            "proposal_type",
            "proposal_status",
            "title_summary",
            "recommended_action",
            "promotion_condition",
            "reason",
            "review_flags",
        ]
    ).lower()
    has_age = bool(re.search(r"\bages?\b", text))
    has_religion = bool(re.search(r"\breligions?\b", text))
    if NON_CORE_LOGICAL_TITLE_RE.search(text):
        return "noncore_context"
    if re.search(r"\btable[-_\s]*(viii|8)\b", text):
        if "civil condition" in text and (has_age or has_religion):
            return "related_civil_condition"
        return "unclassified_review"
    if re.search(r"\btable[-_\s]*(vii(?!i)|7)\b", text):
        return "core_table_vii"
    if "age by religion" in text or (has_age and has_religion and "civil condition" not in text):
        return "core_table_vii"
    if "civil condition" in text and (has_age or has_religion):
        return "related_civil_condition"
    if row.get("logical_collection_only") == "yes" or "logical" in str(row.get("proposal_type", "")).lower():
        return "logical_only_uncertain_scope"
    return "unclassified_review"


def classify_next_step_type(row: dict[str, object]) -> str:
    bucket = str(row.get("queue_bucket", "") or "")
    proposal_type = str(row.get("proposal_type", "") or "")
    if bucket.startswith("01_"):
        return "review_values_or_publish_structure"
    if bucket.startswith("03_"):
        return "manual_visual_review_before_promotion"
    if bucket.startswith("05_"):
        return "ocr_missing_artifact_then_retest"
    if bucket.startswith(("07_", "08_")):
        return "recover_missing_page_then_adjacent_retest"
    if bucket.startswith(("09_", "10_", "11_", "12_")) or "logical" in proposal_type:
        return "logical_navigation_review_only"
    if bucket.startswith("13_"):
        return "keep_separate_conflict_documented"
    return "lower_priority_review"


def urgency_rank(core_scope: str, next_step_type: str, queue_priority: object) -> int:
    if next_step_type == "review_values_or_publish_structure":
        return 1 if core_scope in {"core_table_vii", "related_civil_condition"} else 4
    if next_step_type == "ocr_missing_artifact_then_retest":
        return {"core_table_vii": 2, "related_civil_condition": 4, "unclassified_review": 6}.get(core_scope, 8)
    if next_step_type == "recover_missing_page_then_adjacent_retest":
        return {"core_table_vii": 3, "related_civil_condition": 5, "unclassified_review": 7}.get(core_scope, 8)
    if next_step_type == "manual_visual_review_before_promotion":
        return {"core_table_vii": 4, "related_civil_condition": 5, "unclassified_review": 7}.get(core_scope, 8)
    if next_step_type == "logical_navigation_review_only":
        return {"core_table_vii": 6, "related_civil_condition": 7, "logical_only_uncertain_scope": 8}.get(core_scope, 9)
    if next_step_type == "keep_separate_conflict_documented":
        return 10
    return 11 + min(max(int_value(queue_priority), 0), 99)


def build_assembly_priority_classified_queue(assembly_work_queue: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[tuple[int, int, str, dict[str, object]]] = []
    for row in assembly_work_queue:
        core_scope = classify_core_scope(row)
        next_step_type = classify_next_step_type(row)
        urgency = urgency_rank(core_scope, next_step_type, row.get("queue_priority"))
        record = {
            "priority_rank": "",
            "urgency_rank": urgency,
            "core_scope": core_scope,
            "next_step_type": next_step_type,
            "queue_bucket": row.get("queue_bucket", ""),
            "queue_priority": row.get("queue_priority", ""),
            "record_type": row.get("record_type", ""),
            "proposal_id": row.get("proposal_id", ""),
            "proposal_type": row.get("proposal_type", ""),
            "proposal_status": row.get("proposal_status", ""),
            "source_key": row.get("source_key", ""),
            "source_pdf_id": row.get("source_pdf_id", ""),
            "page_range": row.get("page_range", ""),
            "component_pages": row.get("component_pages", ""),
            "n_pages": row.get("n_pages", ""),
            "pages_consecutive": row.get("pages_consecutive", ""),
            "physical_merge_allowed": row.get("physical_merge_allowed", ""),
            "logical_collection_only": row.get("logical_collection_only", ""),
            "site_ready_physical_structure": row.get("site_ready_physical_structure", ""),
            "site_ready_logical_collection": row.get("site_ready_logical_collection", ""),
            "right_page_no_row_labels_signal": row.get("right_page_no_row_labels_signal", ""),
            "zero_check_status": row.get("zero_check_status", ""),
            "artifact_status": row.get("artifact_status", ""),
            "title_summary": row.get("title_summary", ""),
            "recommended_action": row.get("recommended_action", ""),
            "promotion_condition": row.get("promotion_condition", ""),
            "reason": row.get("reason", ""),
            "review_flags": row.get("review_flags", ""),
            "matrix_csv": row.get("matrix_csv", ""),
        }
        sort_key = f"{record['source_key']}||{record['page_range']}||{record['proposal_id']}"
        rows.append((urgency, int_value(row.get("queue_priority")), sort_key, record))
    out = [record for _, _, _, record in sorted(rows, key=lambda item: (item[0], item[1], item[2]))]
    for rank, row in enumerate(out, start=1):
        row["priority_rank"] = rank
    return out


def build_assembly_ocr_recovery_priority(assembly_priority_classified: list[dict[str, object]]) -> list[dict[str, object]]:
    actionable_steps = {
        "ocr_missing_artifact_then_retest",
        "recover_missing_page_then_adjacent_retest",
        "manual_visual_review_before_promotion",
    }
    rows = [dict(row) for row in assembly_priority_classified if row.get("next_step_type") in actionable_steps]
    for rank, row in enumerate(rows, start=1):
        row["priority_rank"] = rank
    return rows


def sample_join(values: Iterable[object], limit: int = 8) -> str:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
        if len(out) >= limit:
            break
    return "; ".join(out)


def build_watchlist_source_rollup(missed_grouping_watchlist: list[dict[str, object]]) -> list[dict[str, object]]:
    """Condense logical-only no-missed-grouping watchlist edges by source."""
    grouped: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in missed_grouping_watchlist:
        key = (
            str(row.get("watch_bucket", "")),
            str(row.get("source_key", "")),
            str(row.get("source_pdf_id", "")),
        )
        grouped[key].append(row)

    rows: list[tuple[int, int, str, dict[str, object]]] = []
    for (bucket, source_key, source_pdf_id), group in grouped.items():
        priorities = [int_value(row.get("watch_priority")) or 99 for row in group]
        priority = min(priorities) if priorities else 99
        record = {
            "rollup_rank": "",
            "watch_bucket": bucket,
            "watch_priority": priority,
            "source_key": source_key,
            "source_pdf_id": source_pdf_id,
            "edge_count": len(group),
            "right_page_no_row_labels_edges": sum(1 for row in group if row.get("right_page_no_row_labels") == "yes"),
            "same_table_family_edges": sum(1 for row in group if row.get("same_table_family") == "yes"),
            "same_table_number_edges": sum(1 for row in group if row.get("same_table_number") == "yes"),
            "page_pairs_sample": sample_join((row.get("page_pair", "") for row in group), 12),
            "decision_reasons_sample": sample_join((row.get("decision_reason", "") for row in group), 5),
            "recommended_action": sample_join((row.get("recommended_action", "") for row in group), 2),
            "promotion_condition": sample_join((row.get("promotion_condition", "") for row in group), 2),
        }
        rows.append((priority, -len(group), f"{source_key}||{bucket}", record))

    out = [record for _, _, _, record in sorted(rows, key=lambda item: (item[0], item[1], item[2]))]
    for rank, row in enumerate(out, start=1):
        row["rollup_rank"] = rank
    return out


def build_watchlist_family_rollup(missed_grouping_watchlist: list[dict[str, object]]) -> list[dict[str, object]]:
    """Condense logical-only watchlist edges by source and table-transition evidence."""
    grouped: dict[tuple[str, str, str, str, str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in missed_grouping_watchlist:
        key = (
            str(row.get("watch_bucket", "")),
            str(row.get("source_key", "")),
            str(row.get("source_pdf_id", "")),
            str(row.get("page_class_transition", "")),
            str(row.get("table_family_transition", "")),
            str(row.get("table_number_transition", "")),
            str(row.get("title_switch_signal", "")),
        )
        grouped[key].append(row)

    rows: list[tuple[int, int, str, dict[str, object]]] = []
    for (
        bucket,
        source_key,
        source_pdf_id,
        page_class_transition,
        table_family_transition,
        table_number_transition,
        title_switch_signal,
    ), group in grouped.items():
        priorities = [int_value(row.get("watch_priority")) or 99 for row in group]
        priority = min(priorities) if priorities else 99
        record = {
            "rollup_rank": "",
            "watch_bucket": bucket,
            "watch_priority": priority,
            "source_key": source_key,
            "source_pdf_id": source_pdf_id,
            "page_class_transition": page_class_transition,
            "table_family_transition": table_family_transition,
            "table_number_transition": table_number_transition,
            "title_switch_signal": title_switch_signal,
            "edge_count": len(group),
            "right_page_no_row_labels_edges": sum(1 for row in group if row.get("right_page_no_row_labels") == "yes"),
            "same_table_family_edges": sum(1 for row in group if row.get("same_table_family") == "yes"),
            "same_table_number_edges": sum(1 for row in group if row.get("same_table_number") == "yes"),
            "page_pairs_sample": sample_join((row.get("page_pair", "") for row in group), 14),
            "decision_reasons_sample": sample_join((row.get("decision_reason", "") for row in group), 5),
            "rule_flags_sample": sample_join((row.get("rule_flags", "") for row in group), 5),
            "recommended_action": sample_join((row.get("recommended_action", "") for row in group), 2),
            "promotion_condition": sample_join((row.get("promotion_condition", "") for row in group), 2),
        }
        sort_key = "||".join(
            [source_key, table_family_transition, table_number_transition, page_class_transition, title_switch_signal, bucket]
        )
        rows.append((priority, -len(group), sort_key, record))

    out = [record for _, _, _, record in sorted(rows, key=lambda item: (item[0], item[1], item[2]))]
    for rank, row in enumerate(out, start=1):
        row["rollup_rank"] = rank
    return out


def build_summary(
    paths: dict[str, Path],
    intake_summary: dict[str, object],
    frontier_summary: dict[str, object],
    quality_summary: dict[str, object],
    materialized_summary: dict[str, object],
    logical_summary: dict[str, object],
    manual_summary: dict[str, object],
    contract_summary: dict[str, object],
    rulebook_edges: list[dict[str, str]],
    edge_decision_ledger: list[dict[str, object]],
    physical_rows: list[dict[str, str]],
    physical_matrix_files: list[dict[str, str]],
    physical_matrix_columns: list[dict[str, str]],
    manual_preview_rows: list[dict[str, str]],
    manual_preview_matrix_files: list[dict[str, str]],
    manual_preview_matrix_columns: list[dict[str, str]],
    manual_preview_triage: list[dict[str, object]],
    grouping_proposals: list[dict[str, object]],
    grouping_proposal_members: list[dict[str, object]],
    missed_grouping_watchlist: list[dict[str, object]],
    assembly_work_queue: list[dict[str, object]],
    assembly_priority_classified: list[dict[str, object]],
    watchlist_source_rollup: list[dict[str, object]],
    watchlist_family_rollup: list[dict[str, object]],
    logical_site_ready: list[dict[str, str]],
    logical_review: list[dict[str, str]],
    logical_members: list[dict[str, str]],
    manual_rows: list[dict[str, str]],
    audits: list[dict[str, object]],
    chain_rows: list[dict[str, object]],
    grouping_safety_audit: list[dict[str, object]],
    decision_ledger: list[dict[str, object]],
    physical_missing_ocr: list[dict[str, str]],
    physical_missing_pages: list[dict[str, str]],
    physical_manual_frontier: list[dict[str, str]],
    physical_conflicts: list[dict[str, str]],
    physical_frontier_action_queue: list[dict[str, object]],
) -> list[dict[str, object]]:
    chain_failures = sum(1 for row in chain_rows if str(row.get("status", "")).startswith("fail"))
    grouping_safety_failures = audit_failure_count([dict(row) for row in grouping_safety_audit])
    ocr_recovery_steps = {
        "ocr_missing_artifact_then_retest",
        "recover_missing_page_then_adjacent_retest",
        "manual_visual_review_before_promotion",
    }
    assembly_ocr_recovery_priority = [
        row for row in assembly_priority_classified if row.get("next_step_type") in ocr_recovery_steps
    ]
    rows: list[dict[str, object]] = [
        {"metric": "paths", "name": name, "value": output_path(path), "detail": ""}
        for name, path in paths.items()
    ]
    rows.extend(
        [
            {"metric": "counts", "name": "frontier_safe_physical_edges", "value": int_value(frontier_summary.get("safe_physical_edges_now")), "detail": ""},
            {"metric": "counts", "name": "rulebook_edge_decision_rows", "value": len(edge_decision_ledger), "detail": "One row per rulebook edge considered before physical/logical grouping decisions."},
            {"metric": "counts", "name": "materialized_physical_assemblies", "value": len(physical_rows), "detail": "All v43 safe physical edges materialized by v46."},
            {"metric": "counts", "name": "manual_preview_candidates", "value": len(manual_preview_rows), "detail": "Adjacent manual-review candidates materialized as preview-only matrices; not physical merges."},
            {"metric": "counts", "name": "manual_preview_triage_rows", "value": len(manual_preview_triage), "detail": "Rule-ranked review queue for deciding which preview candidates merit manual promotion, logical grouping, or rejection."},
            {"metric": "counts", "name": "grouping_proposal_rows", "value": len(grouping_proposals), "detail": "Single policy-safe manifest covering physical, preview, logical-only, missing, and rejected/blocker grouping proposals."},
            {"metric": "counts", "name": "grouping_proposal_member_rows", "value": len(grouping_proposal_members), "detail": "Member/page rows for grouping proposals."},
            {"metric": "counts", "name": "missed_grouping_watchlist_rows", "value": len(missed_grouping_watchlist), "detail": "High-signal rulebook edges not represented as grouping proposals, retained for no-missed-grouping review."},
            {"metric": "counts", "name": "assembly_work_queue_rows", "value": len(assembly_work_queue), "detail": "Prioritized dashboard over every grouping proposal; does not replace the full manifest."},
            {"metric": "counts", "name": "assembly_priority_classified_rows", "value": len(assembly_priority_classified), "detail": "Same work queue with derived core/non-core scope and next-step classifications."},
            {"metric": "counts", "name": "assembly_ocr_recovery_priority_rows", "value": len(assembly_ocr_recovery_priority), "detail": "Filtered rows requiring OCR artifact, page recovery, or manual visual review before retest/promotion."},
            {"metric": "counts", "name": "assembly_ocr_recovery_core_table_vii_rows", "value": sum(1 for row in assembly_ocr_recovery_priority if row.get("core_scope") == "core_table_vii"), "detail": "OCR/recovery/manual rows classified as likely core Table VII."},
            {"metric": "counts", "name": "assembly_ocr_recovery_related_civil_condition_rows", "value": sum(1 for row in assembly_ocr_recovery_priority if row.get("core_scope") == "related_civil_condition"), "detail": "OCR/recovery/manual rows classified as related civil-condition material."},
            {"metric": "counts", "name": "assembly_ocr_recovery_noncore_context_rows", "value": sum(1 for row in assembly_ocr_recovery_priority if row.get("core_scope") == "noncore_context"), "detail": "OCR/recovery/manual rows classified as non-core/context."},
            {"metric": "counts", "name": "assembly_priority_core_table_vii_rows", "value": sum(1 for row in assembly_priority_classified if row.get("core_scope") == "core_table_vii"), "detail": "Classified as likely core Table VII work or review."},
            {"metric": "counts", "name": "assembly_priority_related_civil_condition_rows", "value": sum(1 for row in assembly_priority_classified if row.get("core_scope") == "related_civil_condition"), "detail": "Classified as related age/civil-condition work, usually not the primary Table VII target."},
            {"metric": "counts", "name": "assembly_priority_noncore_context_rows", "value": sum(1 for row in assembly_priority_classified if row.get("core_scope") == "noncore_context"), "detail": "Classified as context/off-target for the current Table VII assembly push."},
            {"metric": "counts", "name": "watchlist_source_rollup_rows", "value": len(watchlist_source_rollup), "detail": "Source-level rollup of missed_grouping_watchlist rows for concise logical-only review."},
            {"metric": "counts", "name": "watchlist_family_rollup_rows", "value": len(watchlist_family_rollup), "detail": "Logical-review watchlist rollup by source plus page-class/table/title transition evidence."},
            {"metric": "counts", "name": "assembly_decision_ledger_rows", "value": len(decision_ledger), "detail": "Single consolidated record of safe, logical-only, blocked, and review grouping decisions."},
            {"metric": "counts", "name": "physical_frontier_action_queue_rows", "value": len(physical_frontier_action_queue), "detail": "One prioritized row per physical frontier edge, including safe, review, blocked, and conflict records."},
            {"metric": "counts", "name": "materialized_physical_structure_ready", "value": sum(1 for row in physical_rows if row.get("assembly_status") == "materialized_physical_structure"), "detail": "Structure-ready; values still follow value-review policy."},
            {"metric": "counts", "name": "materialized_physical_review_required", "value": sum(1 for row in physical_rows if row.get("assembly_status") != "materialized_physical_structure"), "detail": "Materialized but not clean enough to treat as fully structure-ready."},
            {"metric": "counts", "name": "materialized_rows", "value": int_value(materialized_summary.get("materialized_rows")), "detail": ""},
            {"metric": "counts", "name": "materialized_matrix_rows", "value": int_value(materialized_summary.get("materialized_matrix_rows")), "detail": "Global matrix rows; one per materialized assembled row."},
            {"metric": "counts", "name": "materialized_matrix_columns", "value": int_value(materialized_summary.get("materialized_matrix_columns")) or len(physical_matrix_columns), "detail": "Global matrix columns across all physical assemblies."},
            {"metric": "counts", "name": "compact_matrix_files", "value": int_value(materialized_summary.get("compact_matrix_files")) or len(physical_matrix_files), "detail": "Dense per-assembly matrix CSVs for direct table review."},
            {"metric": "counts", "name": "materialized_cells", "value": int_value(materialized_summary.get("materialized_cells")), "detail": ""},
            {"metric": "counts", "name": "manual_preview_matrix_files", "value": int_value(materialized_summary.get("manual_preview_matrix_files")) or len(manual_preview_matrix_files), "detail": "Dense per-candidate preview matrix CSVs for manual review only."},
            {"metric": "counts", "name": "manual_preview_matrix_columns", "value": int_value(materialized_summary.get("manual_preview_matrix_columns")) or len(manual_preview_matrix_columns), "detail": ""},
            {"metric": "counts", "name": "manual_preview_rows", "value": int_value(materialized_summary.get("manual_preview_rows")), "detail": ""},
            {"metric": "counts", "name": "manual_preview_cells", "value": int_value(materialized_summary.get("manual_preview_cells")), "detail": ""},
            {"metric": "counts", "name": "site_ready_logical_sibling_sets", "value": len(logical_site_ready), "detail": "Logical-only sibling sets; never physical merges."},
            {"metric": "counts", "name": "review_logical_sibling_sets", "value": len(logical_review), "detail": "Logical-only sibling candidates held for review before site use."},
            {"metric": "counts", "name": "logical_sibling_member_rows", "value": len(logical_members), "detail": "Member-level pages in logical sibling collections."},
            {"metric": "counts", "name": "manual_adjudication_edges", "value": len(manual_rows), "detail": "Adjacent candidates needing visual/manual decision."},
            {"metric": "counts", "name": "missing_ocr_edges", "value": int_value(frontier_summary.get("missing_ocr_edges")), "detail": "Do not merge until OCR artifact exists and edge retests."},
            {"metric": "counts", "name": "no_merge_now_edges", "value": int_value(frontier_summary.get("no_merge_now_edges")), "detail": ""},
            {"metric": "counts", "name": "physical_frontier_missing_ocr_edges", "value": len(physical_missing_ocr), "detail": "Adjacent physical candidates blocked by one or more missing OCR artifacts."},
            {"metric": "counts", "name": "physical_frontier_missing_page_bridges", "value": len(physical_missing_pages), "detail": "Nonconsecutive bridges; recover intermediate pages before retesting adjacent edges."},
            {"metric": "counts", "name": "physical_frontier_manual_review_edges", "value": len(physical_manual_frontier), "detail": "Adjacent row-label/visual candidates not safe enough for automatic physical assembly."},
            {"metric": "counts", "name": "physical_frontier_conflict_edges", "value": len(physical_conflicts), "detail": "Structural/semantic conflicts kept out of physical assembly."},
            {"metric": "counts", "name": "row_label_coverage_audit_rows", "value": int_value(quality_summary.get("row_label_audit_rows")), "detail": "Audit rows explaining row-label inheritance signals and breaks."},
            {"metric": "counts", "name": "row_label_potential_missed", "value": int_value(quality_summary.get("row_label_potential_missed")), "detail": "Must be zero for a clean no-missed-row-label release."},
            {"metric": "counts", "name": "logical_overbroad_risks", "value": int_value(quality_summary.get("logical_overbroad_risks")), "detail": "Logical family risks listed for review; not physical merges."},
            {"metric": "counts", "name": "possible_missing_religion_sequences", "value": int_value(quality_summary.get("possible_missing_religion_sequences")), "detail": "Possible logical-only religion sibling sequences needing review."},
            {"metric": "counts", "name": "safe_edge_graph_components", "value": len(chain_rows), "detail": "Connected components in the current site-ready physical edge graph."},
            {"metric": "counts", "name": "safe_edge_graph_failures", "value": chain_failures, "detail": "Nonconsecutive, malformed, or oversized site-ready physical edge chains."},
            {"metric": "counts", "name": "grouping_safety_failures", "value": grouping_safety_failures, "detail": "Release-blocking grouping invariant failures."},
            {"metric": "counts", "name": "intake_target_count", "value": int_value(intake_summary.get("target_count")), "detail": "No-OCR intake target pages from missing/gap queues."},
            {"metric": "counts", "name": "intake_targets_with_artifact_match", "value": int_value(intake_summary.get("targets_with_artifact_match")), "detail": "Targets already matched to a local OCR artifact."},
            {"metric": "counts", "name": "intake_render_available_no_ocr_artifact", "value": intake_status_count(intake_summary, "render_available_no_ocr_artifact"), "detail": "Rendered pages available locally but still not represented by OCR output."},
            {"metric": "counts", "name": "intake_still_blocked_no_local_artifact", "value": intake_status_count(intake_summary, "still_blocked_no_local_artifact"), "detail": "Targets with no local render/OCR artifact in scanned roots."},
            {"metric": "fingerprints", "name": "intake_artifact_match_fingerprint", "value": intake_summary.get("artifact_match_fingerprint", ""), "detail": "Changes when matched OCR artifact paths/mtimes/sizes/methods change, even if matched count is unchanged."},
            {"metric": "fingerprints", "name": "intake_target_state_fingerprint", "value": intake_summary.get("target_state_fingerprint", ""), "detail": "Changes when target unlock/render/artifact state changes."},
            {"metric": "counts", "name": "legacy_contract_physical_components", "value": int_value(contract_summary.get("site_ready_physical_components")), "detail": "Pre-v42/v46 contract count retained for trace only."},
            {"metric": "counts", "name": "legacy_contract_review_required", "value": int_value(contract_summary.get("review_required")), "detail": ""},
            {"metric": "audits", "name": "release_audit_failures", "value": audit_failure_count([dict(r) for r in audits]), "detail": ""},
            {"metric": "audits", "name": "release_audit_warnings", "value": audit_warning_count([dict(r) for r in audits]), "detail": ""},
        ]
    )
    rows.extend(counter_rows("physical_assembly_status", Counter(row.get("assembly_status", "") for row in physical_rows)))
    rows.extend(counter_rows("physical_frontier_category", Counter(row.get("frontier_category", "") for row in physical_rows)))
    rows.extend(counter_rows("manual_preview_status", Counter(row.get("assembly_status", "") for row in manual_preview_rows)))
    rows.extend(counter_rows("manual_preview_frontier_category", Counter(row.get("frontier_category", "") for row in manual_preview_rows)))
    rows.extend(counter_rows("manual_preview_triage_bucket", Counter(str(row.get("triage_bucket", "")) for row in manual_preview_triage)))
    rows.extend(counter_rows("grouping_proposal_type", Counter(str(row.get("proposal_type", "")) for row in grouping_proposals)))
    rows.extend(counter_rows("grouping_proposal_priority", Counter(str(row.get("priority_bucket", "")) for row in grouping_proposals)))
    rows.extend(counter_rows("assembly_work_queue_bucket", Counter(str(row.get("queue_bucket", "")) for row in assembly_work_queue)))
    rows.extend(counter_rows("assembly_priority_core_scope", Counter(str(row.get("core_scope", "")) for row in assembly_priority_classified)))
    rows.extend(counter_rows("assembly_priority_next_step", Counter(str(row.get("next_step_type", "")) for row in assembly_priority_classified)))
    rows.extend(counter_rows("assembly_ocr_recovery_core_scope", Counter(str(row.get("core_scope", "")) for row in assembly_ocr_recovery_priority)))
    rows.extend(counter_rows("assembly_ocr_recovery_next_step", Counter(str(row.get("next_step_type", "")) for row in assembly_ocr_recovery_priority)))
    rows.extend(counter_rows("missed_grouping_watch_bucket", Counter(str(row.get("watch_bucket", "")) for row in missed_grouping_watchlist)))
    rows.extend(counter_rows("watchlist_source_rollup_bucket", Counter(str(row.get("watch_bucket", "")) for row in watchlist_source_rollup)))
    for key, value in sorted((logical_summary.get("sets_by_status") or {}).items()):
        rows.append({"metric": "logical_sibling_status", "name": key, "value": value, "detail": ""})
    for key, value in sorted((manual_summary.get("adjudication_by_tier") or {}).items()):
        rows.append({"metric": "manual_adjudication_tier", "name": key, "value": value, "detail": ""})
    return rows


def build_audits(
    contract_audit: list[dict[str, str]],
    frontier_audit: list[dict[str, str]],
    logical_audit: list[dict[str, str]],
    manual_audit: list[dict[str, str]],
    materialized_audit: list[dict[str, str]],
    frontier_summary: dict[str, object],
    materialized_summary: dict[str, object],
    contract_summary: dict[str, object],
    quality_summary: dict[str, object],
    grouping_safety_audit: list[dict[str, object]],
    decision_ledger: list[dict[str, object]],
    rulebook_edges: list[dict[str, str]],
    edge_decision_ledger: list[dict[str, object]],
    physical_frontier_action_queue: list[dict[str, object]],
    row_label_coverage: list[dict[str, str]],
    frontier_rows: list[dict[str, str]],
    physical_rows: list[dict[str, str]],
    physical_matrix_files: list[dict[str, str]],
    manual_preview_rows: list[dict[str, str]],
    manual_preview_matrix_files: list[dict[str, str]],
    manual_preview_qc: list[dict[str, str]],
    manual_preview_triage: list[dict[str, object]],
    grouping_proposals: list[dict[str, object]],
    grouping_proposal_members: list[dict[str, object]],
    missed_grouping_watchlist: list[dict[str, object]],
    assembly_work_queue: list[dict[str, object]],
    watchlist_source_rollup: list[dict[str, object]],
    watchlist_family_rollup: list[dict[str, object]],
    logical_site_ready: list[dict[str, str]],
    logical_review: list[dict[str, str]],
    logical_members: list[dict[str, str]],
    logical_family_quality: list[dict[str, str]],
    religion_sequence_coverage: list[dict[str, str]],
    physical_manual_frontier: list[dict[str, str]],
    physical_missing_ocr: list[dict[str, str]],
    physical_missing_pages: list[dict[str, str]],
    physical_conflicts: list[dict[str, str]],
    workbook_path: Path,
) -> list[dict[str, object]]:
    audits: list[dict[str, object]] = []
    audits.append({"check": "legacy_contract_has_no_failures", "status": "pass" if audit_failure_count(contract_audit) == 0 else "fail", "detail": f"failures={audit_failure_count(contract_audit)}"})
    audits.append({"check": "frontier_has_no_failures", "status": "pass" if audit_failure_count(frontier_audit) == 0 else "fail", "detail": f"failures={audit_failure_count(frontier_audit)}"})
    audits.append({"check": "manual_adjudication_has_no_failures", "status": "pass" if audit_failure_count(manual_audit) == 0 else "fail", "detail": f"failures={audit_failure_count(manual_audit)}"})
    audits.append({"check": "logical_siblings_have_no_failures", "status": "pass" if audit_failure_count(logical_audit) == 0 else "fail", "detail": f"failures={audit_failure_count(logical_audit)}"})
    materialized_warnings = audit_warning_count(materialized_audit)
    audits.append({"check": "materialized_physical_has_no_failures", "status": "pass" if audit_failure_count(materialized_audit) == 0 else "fail", "detail": f"failures={audit_failure_count(materialized_audit)} warnings={materialized_warnings}"})
    audits.append({"check": "materialized_physical_warnings_carried_to_review", "status": "warn" if materialized_warnings else "pass", "detail": f"warnings={materialized_warnings}"})

    safe_edges = int_value(frontier_summary.get("safe_physical_edges_now"))
    materialized_count = int_value(materialized_summary.get("materialized_physical_assemblies"))
    audits.append({"check": "frontier_safe_edges_all_materialized", "status": "pass" if safe_edges == materialized_count else "fail", "detail": f"safe_edges={safe_edges} materialized={materialized_count}"})
    audits.append({
        "check": "manual_preview_candidates_match_manual_frontier",
        "status": "pass" if len(manual_preview_rows) == len(physical_manual_frontier) else "fail",
        "detail": f"manual_frontier={len(physical_manual_frontier)} manual_preview={len(manual_preview_rows)}",
    })
    preview_merge_violations = [
        row.get("assembly_id", "")
        for row in manual_preview_rows
        if row.get("physical_merge_allowed") != "no" or row.get("safe_to_publish_values") != "no"
    ]
    audits.append({
        "check": "manual_preview_records_are_not_physical_merges",
        "status": "pass" if not preview_merge_violations else "fail",
        "detail": f"bad={len(preview_merge_violations)} examples={'; '.join(preview_merge_violations[:10])}",
    })
    preview_nonconsecutive = [row.get("assembly_id", "") for row in manual_preview_rows if row.get("pages_consecutive") != "yes"]
    audits.append({
        "check": "manual_preview_edges_are_consecutive",
        "status": "pass" if not preview_nonconsecutive else "fail",
        "detail": f"bad={len(preview_nonconsecutive)} examples={'; '.join(preview_nonconsecutive[:10])}",
    })
    audits.append({
        "check": "manual_preview_matrix_file_per_candidate",
        "status": "pass" if len(manual_preview_matrix_files) == len(manual_preview_rows) else "fail",
        "detail": f"matrix_files={len(manual_preview_matrix_files)} manual_preview={len(manual_preview_rows)}",
    })
    audits.append({
        "check": "manual_preview_triage_covers_candidates",
        "status": "pass" if len(manual_preview_triage) == len(manual_preview_rows) else "fail",
        "detail": f"triage={len(manual_preview_triage)} manual_preview={len(manual_preview_rows)}",
    })
    triage_missing_matrix = [row.get("assembly_id", "") for row in manual_preview_triage if not str(row.get("matrix_csv", "")).strip()]
    audits.append({
        "check": "manual_preview_triage_has_matrix_paths",
        "status": "pass" if not triage_missing_matrix else "fail",
        "detail": f"missing={len(triage_missing_matrix)} examples={'; '.join(triage_missing_matrix[:10])}",
    })
    triage_policy_violations = [
        row.get("assembly_id", "")
        for row in manual_preview_triage
        if row.get("physical_merge_allowed") != "no" or row.get("safe_to_publish_values") != "no"
    ]
    audits.append({
        "check": "manual_preview_triage_remains_review_only",
        "status": "pass" if not triage_policy_violations else "fail",
        "detail": f"bad={len(triage_policy_violations)} examples={'; '.join(triage_policy_violations[:10])}",
    })
    preview_qc_failures = [row for row in manual_preview_qc if row.get("status") == "fail"]
    audits.append({
        "check": "manual_preview_qc_failures_carried_to_review",
        "status": "warn" if preview_qc_failures else "pass",
        "detail": f"failures={len(preview_qc_failures)}",
    })
    expected_proposals = (
        len(physical_rows)
        + len(manual_preview_triage)
        + len(logical_site_ready)
        + len(logical_review)
        + len(physical_missing_ocr)
        + len(physical_missing_pages)
        + len(physical_conflicts)
        + sum(1 for row in grouping_proposals if row.get("proposal_type") == "logical_pairwise_edge_candidate")
    )
    audits.append({
        "check": "grouping_proposals_cover_release_sources",
        "status": "pass" if len(grouping_proposals) == expected_proposals else "fail",
        "detail": f"proposals={len(grouping_proposals)} expected={expected_proposals}",
    })
    member_proposal_ids = {str(row.get("proposal_id", "")) for row in grouping_proposal_members if row.get("proposal_id")}
    proposal_ids = {str(row.get("proposal_id", "")) for row in grouping_proposals if row.get("proposal_id")}
    missing_members = sorted(proposal_ids - member_proposal_ids)
    audits.append({
        "check": "grouping_proposals_have_member_rows",
        "status": "pass" if not missing_members else "fail",
        "detail": f"missing={len(missing_members)} examples={'; '.join(missing_members[:10])}",
    })
    proposal_page_metadata_missing = [
        row.get("proposal_id", "")
        for row in grouping_proposals
        if page_list_from_text(row.get("component_pages") or row.get("page_range"))
        and (
            not str(row.get("pdf_pages_sorted", "")).strip()
            or not str(row.get("pdf_page_start", "")).strip()
            or not str(row.get("pdf_page_end", "")).strip()
        )
    ]
    audits.append({
        "check": "grouping_proposals_have_explicit_pdf_page_fields",
        "status": "pass" if not proposal_page_metadata_missing else "fail",
        "detail": f"missing={len(proposal_page_metadata_missing)} examples={'; '.join(proposal_page_metadata_missing[:10])}",
    })
    proposal_page_count_mismatch = []
    for row in grouping_proposals:
        n_pages = int_value(row.get("n_pages"))
        page_span_size = int_value(row.get("page_span_size"))
        if n_pages and page_span_size and n_pages != page_span_size:
            proposal_page_count_mismatch.append(str(row.get("proposal_id", "")))
    audits.append({
        "check": "grouping_proposal_page_span_matches_n_pages",
        "status": "pass" if not proposal_page_count_mismatch else "fail",
        "detail": f"bad={len(proposal_page_count_mismatch)} examples={'; '.join(proposal_page_count_mismatch[:10])}",
    })
    proposal_physical_policy_bad = [
        row.get("proposal_id", "")
        for row in grouping_proposals
        if row.get("site_ready_physical_structure") == "yes"
        and (row.get("proposal_type") != "physical_pairwise_assembly" or row.get("physical_merge_allowed") != "yes" or row.get("pages_consecutive") != "yes")
    ]
    audits.append({
        "check": "grouping_proposal_physical_site_ready_is_strict_pairwise",
        "status": "pass" if not proposal_physical_policy_bad else "fail",
        "detail": f"bad={len(proposal_physical_policy_bad)} examples={'; '.join(proposal_physical_policy_bad[:10])}",
    })
    proposal_review_merge_bad = [
        row.get("proposal_id", "")
        for row in grouping_proposals
        if row.get("proposal_type") != "physical_pairwise_assembly" and row.get("physical_merge_allowed") == "yes"
    ]
    audits.append({
        "check": "grouping_proposal_review_and_logical_never_physical_merge",
        "status": "pass" if not proposal_review_merge_bad else "fail",
        "detail": f"bad={len(proposal_review_merge_bad)} examples={'; '.join(proposal_review_merge_bad[:10])}",
    })
    logical_policy_bad = [
        row.get("proposal_id", "")
        for row in grouping_proposals
        if row.get("site_ready_logical_collection") == "yes"
        and (row.get("logical_collection_only") != "yes" or row.get("physical_merge_allowed") != "no")
    ]
    audits.append({
        "check": "grouping_proposal_logical_site_ready_is_logical_only",
        "status": "pass" if not logical_policy_bad else "fail",
        "detail": f"bad={len(logical_policy_bad)} examples={'; '.join(logical_policy_bad[:10])}",
    })
    overbroad_site_ready_logical = [
        row.get("proposal_id", "")
        for row in grouping_proposals
        if row.get("proposal_type") == "logical_sibling_collection"
        and row.get("site_ready_logical_collection") == "yes"
        and (int_value(row.get("n_pages")) or 0) > MAX_SITE_READY_LOGICAL_SIBLING_PAGES
    ]
    audits.append({
        "check": "site_ready_logical_sibling_collections_are_conservatively_bounded",
        "status": "pass" if not overbroad_site_ready_logical else "fail",
        "detail": f"max_pages={MAX_SITE_READY_LOGICAL_SIBLING_PAGES} bad={len(overbroad_site_ready_logical)} examples={'; '.join(overbroad_site_ready_logical[:10])}",
    })
    proposal_keys_after_logical_edges = proposal_edge_keys(grouping_proposals)
    uncovered_confirmed_logical_edges = [
        row.get("edge_key", "")
        for row in edge_decision_ledger
        if (row.get("logical_group_allowed_now") == "yes" or row.get("edge_decision_status") == "confirmed_logical_only_edge")
        and row.get("edge_key") not in proposal_keys_after_logical_edges
    ]
    audits.append({
        "check": "confirmed_logical_edges_are_represented_as_grouping_proposals",
        "status": "pass" if not uncovered_confirmed_logical_edges else "fail",
        "detail": f"bad={len(uncovered_confirmed_logical_edges)} examples={'; '.join(uncovered_confirmed_logical_edges[:10])}",
    })
    uncovered_physical_watch = [
        row.get("edge_key", "")
        for row in missed_grouping_watchlist
        if row.get("watch_bucket") in {"uncovered_row_label_dependency_candidate", "uncovered_physical_merge_allowed_edge", "uncovered_blocked_future_candidate"}
    ]
    audits.append({
        "check": "missed_grouping_watchlist_has_no_uncovered_physical_candidates",
        "status": "pass" if not uncovered_physical_watch else "fail",
        "detail": f"bad={len(uncovered_physical_watch)} examples={'; '.join(uncovered_physical_watch[:10])}",
    })
    watchlist_merge_bad = [
        row.get("edge_key", "")
        for row in missed_grouping_watchlist
        if row.get("watch_bucket") != "right_no_label_hard_break_or_context"
        and row.get("recommended_action", "").lower().find("physically merge") >= 0
        and row.get("recommended_action", "").lower().find("do not") < 0
    ]
    audits.append({
        "check": "missed_grouping_watchlist_is_review_only",
        "status": "pass" if not watchlist_merge_bad else "fail",
        "detail": f"bad={len(watchlist_merge_bad)} examples={'; '.join(watchlist_merge_bad[:10])}",
    })
    queue_proposal_ids = {str(row.get("proposal_id", "")) for row in assembly_work_queue if row.get("proposal_id")}
    missing_queue_proposals = sorted(proposal_ids - queue_proposal_ids)
    extra_queue_proposals = sorted(queue_proposal_ids - proposal_ids)
    audits.append({
        "check": "assembly_work_queue_covers_all_grouping_proposals",
        "status": "pass" if not missing_queue_proposals and not extra_queue_proposals else "fail",
        "detail": f"missing={len(missing_queue_proposals)} extra={len(extra_queue_proposals)} examples={'; '.join((missing_queue_proposals + extra_queue_proposals)[:10])}",
    })
    queue_policy_bad = [
        row.get("proposal_id", "")
        for row in assembly_work_queue
        if row.get("record_type") != "grouping_proposal"
        or (row.get("proposal_type") != "physical_pairwise_assembly" and row.get("physical_merge_allowed") == "yes")
    ]
    audits.append({
        "check": "assembly_work_queue_does_not_create_new_physical_merges",
        "status": "pass" if not queue_policy_bad else "fail",
        "detail": f"bad={len(queue_policy_bad)} examples={'; '.join(queue_policy_bad[:10])}",
    })
    rollup_edge_count = sum(int_value(row.get("edge_count")) for row in watchlist_source_rollup)
    audits.append({
        "check": "watchlist_source_rollup_covers_watchlist_edges",
        "status": "pass" if rollup_edge_count == len(missed_grouping_watchlist) else "fail",
        "detail": f"rollup_edges={rollup_edge_count} watchlist_rows={len(missed_grouping_watchlist)}",
    })
    family_rollup_edge_count = sum(int_value(row.get("edge_count")) for row in watchlist_family_rollup)
    audits.append({
        "check": "watchlist_family_rollup_covers_watchlist_edges",
        "status": "pass" if family_rollup_edge_count == len(missed_grouping_watchlist) else "fail",
        "detail": f"rollup_edges={family_rollup_edge_count} watchlist_rows={len(missed_grouping_watchlist)}",
    })

    materialized_audit_map = {row.get("check"): row.get("status") for row in materialized_audit}
    audits.append({"check": "materialized_edges_are_consecutive", "status": "pass" if materialized_audit_map.get("materialized_edges_are_consecutive") == "pass" else "fail", "detail": materialized_audit_map.get("materialized_edges_are_consecutive", "missing")})
    audits.append({"check": "materialized_qc_has_no_failures", "status": "pass" if materialized_audit_map.get("materialized_qc_has_no_failures") == "pass" else "fail", "detail": materialized_audit_map.get("materialized_qc_has_no_failures", "missing")})

    grouping_failures = audit_failure_count([dict(row) for row in grouping_safety_audit])
    audits.append({"check": "grouping_safety_has_no_failures", "status": "pass" if grouping_failures == 0 else "fail", "detail": f"failures={grouping_failures}"})
    quality_failures = int_value(quality_summary.get("audit_failure_count"))
    audits.append({"check": "quality_audit_has_no_failures", "status": "pass" if quality_failures == 0 else "fail", "detail": f"failures={quality_failures}"})
    row_label_missed = int_value(quality_summary.get("row_label_potential_missed"))
    audits.append({"check": "row_label_potential_missed_is_zero", "status": "pass" if row_label_missed == 0 else "fail", "detail": f"potential_missed={row_label_missed}"})

    expected_ledger_rows = (
        len(physical_rows)
        + len(logical_site_ready)
        + len(logical_family_quality)
        + len(religion_sequence_coverage)
        + len(physical_manual_frontier)
        + len(physical_missing_ocr)
        + len(physical_missing_pages)
        + len(physical_conflicts)
    )
    audits.append({"check": "decision_ledger_covers_release_sources", "status": "pass" if len(decision_ledger) == expected_ledger_rows else "fail", "detail": f"ledger={len(decision_ledger)} expected={expected_ledger_rows}"})
    physical_without_matrix = [
        row.get("assembly_or_edge_id", "")
        for row in decision_ledger
        if row.get("decision_type") == "physical_materialized_now" and not str(row.get("matrix_csv", "")).strip()
    ]
    audits.append({"check": "decision_ledger_physical_rows_have_matrix_paths", "status": "pass" if not physical_without_matrix else "fail", "detail": f"missing={len(physical_without_matrix)} examples={'; '.join(physical_without_matrix[:10])}"})
    ledger_nonconsecutive_physical = [
        row.get("assembly_or_edge_id", "")
        for row in decision_ledger
        if row.get("decision_type") == "physical_materialized_now" and row.get("pages_consecutive") != "yes"
    ]
    audits.append({"check": "decision_ledger_physical_rows_are_consecutive", "status": "pass" if not ledger_nonconsecutive_physical else "fail", "detail": f"bad={len(ledger_nonconsecutive_physical)} examples={'; '.join(ledger_nonconsecutive_physical[:10])}"})
    gap_merge_violations = [
        row.get("assembly_or_edge_id", "")
        for row in decision_ledger
        if row.get("decision_type") == "recover_missing_pages_then_retest" and row.get("grouping_policy") != "no_physical_merge_across_gap"
    ]
    audits.append({"check": "decision_ledger_missing_page_edges_never_merge", "status": "pass" if not gap_merge_violations else "fail", "detail": f"bad={len(gap_merge_violations)} examples={'; '.join(gap_merge_violations[:10])}"})
    logical_merge_violations = [
        row.get("assembly_or_edge_id", "")
        for row in decision_ledger
        if row.get("decision_type", "").startswith(("logical_", "religion_sequence_"))
        and "physical" in str(row.get("grouping_policy", ""))
        and "never_physical" not in str(row.get("grouping_policy", ""))
    ]
    audits.append({"check": "decision_ledger_logical_records_are_not_physical_merges", "status": "pass" if not logical_merge_violations else "fail", "detail": f"bad={len(logical_merge_violations)} examples={'; '.join(logical_merge_violations[:10])}"})

    edge_physical_nonconsecutive = [
        row.get("edge_key", "")
        for row in edge_decision_ledger
        if row.get("physical_merge_allowed_now") == "yes" and row.get("pages_consecutive") != "yes"
    ]
    audits.append({"check": "edge_decision_ledger_present", "status": "pass" if edge_decision_ledger else "fail", "detail": f"rows={len(edge_decision_ledger)}"})
    audits.append({"check": "edge_decision_ledger_covers_rulebook_edges", "status": "pass" if len(edge_decision_ledger) == len(rulebook_edges) else "fail", "detail": f"ledger={len(edge_decision_ledger)} rulebook_edges={len(rulebook_edges)}"})
    audits.append({"check": "edge_decision_physical_merges_are_consecutive", "status": "pass" if not edge_physical_nonconsecutive else "fail", "detail": f"bad={len(edge_physical_nonconsecutive)} examples={'; '.join(edge_physical_nonconsecutive[:10])}"})
    edge_missing_page_merge = [
        row.get("edge_key", "")
        for row in edge_decision_ledger
        if row.get("likely_after_missing_pages") == "yes" and row.get("physical_merge_allowed_now") == "yes"
    ]
    audits.append({"check": "edge_decision_missing_page_candidates_never_merge", "status": "pass" if not edge_missing_page_merge else "fail", "detail": f"bad={len(edge_missing_page_merge)} examples={'; '.join(edge_missing_page_merge[:10])}"})
    edge_ocr_wait_merge = [
        row.get("edge_key", "")
        for row in edge_decision_ledger
        if row.get("likely_after_ocr") == "yes" and row.get("physical_merge_allowed_now") == "yes"
    ]
    audits.append({"check": "edge_decision_ocr_wait_candidates_never_merge", "status": "pass" if not edge_ocr_wait_merge else "fail", "detail": f"bad={len(edge_ocr_wait_merge)} examples={'; '.join(edge_ocr_wait_merge[:10])}"})
    edge_row_label_signals = sum(1 for row in edge_decision_ledger if row.get("row_label_dependency_signal") == "yes")
    quality_row_label_signals = sum(1 for row in row_label_coverage if row.get("row_label_dependency_signal") == "yes")
    audits.append({"check": "edge_decision_row_label_signals_match_quality_audit", "status": "pass" if edge_row_label_signals == quality_row_label_signals else "fail", "detail": f"edge_signals={edge_row_label_signals} quality_signals={quality_row_label_signals}"})
    audits.append({"check": "physical_frontier_action_queue_covers_frontier_rows", "status": "pass" if len(physical_frontier_action_queue) == len(frontier_rows) else "fail", "detail": f"queue={len(physical_frontier_action_queue)} frontier={len(frontier_rows)}"})
    queue_by_edge = {str(row.get("edge_key", "")): row for row in physical_frontier_action_queue if row.get("edge_key")}
    frontier_keys = {str(row.get("edge_key", "")) for row in frontier_rows if row.get("edge_key")}
    queue_missing_right_no_label = [
        row.get("edge_key", "")
        for row in edge_decision_ledger
        if row.get("edge_key") in frontier_keys
        and row.get("right_page_no_row_labels") == "yes"
        and queue_by_edge.get(str(row.get("edge_key", "")), {}).get("right_page_no_row_labels_signal") != "yes"
    ]
    audits.append({"check": "physical_action_queue_preserves_rulebook_right_no_label_signal", "status": "pass" if not queue_missing_right_no_label else "fail", "detail": f"bad={len(queue_missing_right_no_label)} examples={'; '.join(queue_missing_right_no_label[:10])}"})
    queue_missing_row_label_review_signal = [
        row.get("edge_key", "")
        for row in physical_frontier_action_queue
        if (
            "row_label_dependency" in str(row.get("queue_bucket", ""))
            or row.get("row_label_signal") == "yes"
            or row.get("right_page_no_row_labels_signal") == "yes"
        )
        and row.get("queue_row_label_review_signal") != "yes"
    ]
    audits.append({"check": "physical_action_queue_marks_row_label_review_signals", "status": "pass" if not queue_missing_row_label_review_signal else "fail", "detail": f"bad={len(queue_missing_row_label_review_signal)} examples={'; '.join(queue_missing_row_label_review_signal[:10])}"})

    logical_physical_merge_violations = [
        row.get("sibling_set_id", "")
        for row in logical_site_ready + logical_review
        if row.get("physical_merge_allowed") != "no" or row.get("logical_collection_only") != "yes"
    ]
    audits.append({"check": "logical_sibling_release_records_are_logical_only", "status": "pass" if not logical_physical_merge_violations else "fail", "detail": f"bad={len(logical_physical_merge_violations)} examples={'; '.join(logical_physical_merge_violations[:10])}"})
    member_set_ids = {row.get("sibling_set_id", "") for row in logical_members if row.get("sibling_set_id")}
    exported_logical_set_ids = {
        row.get("sibling_set_id", "")
        for row in logical_site_ready + logical_review
        if row.get("sibling_set_id")
    }
    missing_member_sets = sorted(exported_logical_set_ids - member_set_ids)
    audits.append({"check": "logical_sibling_release_sets_have_member_rows", "status": "pass" if not missing_member_sets else "fail", "detail": f"missing={len(missing_member_sets)} examples={'; '.join(missing_member_sets[:10])}"})

    legacy_physical = int_value(contract_summary.get("site_ready_physical_components"))
    audits.append({"check": "materialized_count_supersedes_legacy_physical_count", "status": "pass" if materialized_count >= legacy_physical else "fail", "detail": f"legacy_contract={legacy_physical} materialized_current={materialized_count}"})

    try:
        with zipfile.ZipFile(workbook_path) as zf:
            bad_member = zf.testzip()
    except Exception as exc:  # pragma: no cover
        bad_member = f"{type(exc).__name__}: {exc}"
    audits.append({"check": "workbook_zip_integrity", "status": "pass" if bad_member is None else "fail", "detail": str(bad_member or "ok")})
    return audits


def write_workbook(path: Path, sheets: list[tuple[str, list[dict[str, object]], list[str]]]) -> None:
    workbook = xlsxwriter.Workbook(path, {"constant_memory": True})
    header_fmt = workbook.add_format({"bold": True, "bg_color": "#E8EEF7", "border": 1})
    wrap_fmt = workbook.add_format({"text_wrap": True, "valign": "top"})
    for name, rows, fields in sheets:
        ws = workbook.add_worksheet(name[:31])
        for col, field in enumerate(fields):
            ws.write(0, col, field, header_fmt)
        for row_idx, row in enumerate(rows, start=1):
            for col, field in enumerate(fields):
                value = row.get(field, "")
                fmt = wrap_fmt if field in {
                    "detail",
                    "recommended_action",
                    "promotion_condition",
                    "reason",
                    "decision_reason",
                    "decision_reasons_sample",
                    "review_flags",
                    "rule_flags",
                    "assembled_column_order",
                    "title_sample",
                    "title_summary",
                    "page_pairs_sample",
                    "visual_checklist",
                } else None
                ws.write(row_idx, col, value, fmt)
        ws.freeze_panes(1, 0)
        if fields:
            ws.autofilter(0, 0, max(0, len(rows)), len(fields) - 1)
        for col, field in enumerate(fields):
            ws.set_column(col, col, min(max(len(field) + 2, 12), 58))
    workbook.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract-dir", required=True)
    parser.add_argument("--intake-dir", default="")
    parser.add_argument("--rulebook-dir", default="")
    parser.add_argument("--frontier-dir", required=True)
    parser.add_argument("--quality-dir", default="")
    parser.add_argument("--manual-adjudication-dir", required=True)
    parser.add_argument("--logical-sibling-dir", required=True)
    parser.add_argument("--materialized-physical-dir", required=True)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    args = parser.parse_args()

    contract_dir = Path(args.contract_dir).expanduser()
    intake_dir = Path(args.intake_dir).expanduser() if args.intake_dir else None
    rulebook_dir = Path(args.rulebook_dir).expanduser() if args.rulebook_dir else None
    frontier_dir = Path(args.frontier_dir).expanduser()
    quality_dir = Path(args.quality_dir).expanduser() if args.quality_dir else None
    manual_dir = Path(args.manual_adjudication_dir).expanduser()
    logical_dir = Path(args.logical_sibling_dir).expanduser()
    materialized_dir = Path(args.materialized_physical_dir).expanduser()
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "contract_dir": contract_dir,
        "frontier_dir": frontier_dir,
        "manual_adjudication_dir": manual_dir,
        "logical_sibling_dir": logical_dir,
        "materialized_physical_dir": materialized_dir,
        "out_dir": out_dir,
    }
    if intake_dir is not None:
        paths["intake_dir"] = intake_dir
    if rulebook_dir is not None:
        paths["rulebook_dir"] = rulebook_dir
    if quality_dir is not None:
        paths["quality_dir"] = quality_dir

    contract_summary = read_json(contract_dir / "summary.json")
    intake_summary = read_json(intake_dir / "summary.json") if intake_dir is not None else {}
    frontier_summary = read_json(frontier_dir / "summary.json")
    quality_summary = read_json(quality_dir / "summary.json") if quality_dir is not None else {}
    manual_summary = read_json(manual_dir / "summary.json")
    logical_summary = read_json(logical_dir / "summary.json")
    materialized_summary = read_json(materialized_dir / "summary.json")

    rulebook_edges = read_csv(rulebook_dir / "edge_rulebook.csv") if rulebook_dir is not None else []
    frontier_rows = read_csv(frontier_dir / "physical_assembly_frontier.csv")
    physical_rows = read_csv(materialized_dir / "materialized_physical_assemblies.csv")
    physical_qc = read_csv(materialized_dir / "materialized_physical_qc.csv")
    physical_matrix_files = read_csv(materialized_dir / "materialized_physical_matrix_files.csv")
    physical_matrix_columns = read_csv(materialized_dir / "materialized_physical_matrix_columns.csv")
    manual_preview_rows = read_csv(materialized_dir / "manual_physical_preview_assemblies.csv")
    manual_preview_detail_rows = read_csv(materialized_dir / "manual_physical_preview_rows.csv")
    manual_preview_matrix_rows = read_csv(materialized_dir / "manual_physical_preview_matrix_rows.csv")
    manual_preview_cells = read_csv(materialized_dir / "manual_physical_preview_cells.csv")
    manual_preview_qc = read_csv(materialized_dir / "manual_physical_preview_qc.csv")
    manual_preview_matrix_files = read_csv(materialized_dir / "manual_physical_preview_matrix_files.csv")
    manual_preview_matrix_columns = read_csv(materialized_dir / "manual_physical_preview_matrix_columns.csv")
    row_label_coverage = read_csv(quality_dir / "row_label_coverage_audit.csv") if quality_dir is not None else []
    logical_family_quality = read_csv(quality_dir / "logical_family_quality_audit.csv") if quality_dir is not None else []
    religion_sequence_coverage = read_csv(quality_dir / "religion_sequence_coverage_audit.csv") if quality_dir is not None else []
    logical_all = read_csv(logical_dir / "logical_sibling_sets.csv")
    logical_members = read_csv(logical_dir / "logical_sibling_set_members.csv")
    logical_site_ready = read_csv(logical_dir / "site_ready_logical_sibling_sets.csv")
    logical_review = read_csv(logical_dir / "review_logical_sibling_sets.csv")
    manual_rows = read_csv(manual_dir / "manual_adjudication_queue.csv")

    contract_audit = read_csv(contract_dir / "acceptance_audit.csv")
    frontier_audit = read_csv(frontier_dir / "acceptance_audit.csv")
    manual_audit = read_csv(manual_dir / "acceptance_audit.csv")
    logical_audit = read_csv(logical_dir / "acceptance_audit.csv")
    materialized_audit = read_csv(materialized_dir / "acceptance_audit.csv")
    quality_audit = read_csv(quality_dir / "audit.csv") if quality_dir is not None else []

    edge_decision_ledger = build_edge_decision_ledger(rulebook_edges)
    chain_rows = build_physical_chain_audit(frontier_rows)
    grouping_safety_audit = build_grouping_safety_audit(frontier_rows, physical_rows, logical_site_ready, chain_rows)
    physical_frontier_action_queue = build_physical_frontier_action_queue(frontier_rows, edge_decision_ledger)
    physical_missing_ocr = [row for row in frontier_rows if row.get("frontier_category") == "needs_missing_ocr_artifact"]
    physical_missing_pages = [row for row in frontier_rows if row.get("frontier_category") == "likely_after_missing_pages_no_merge"]
    physical_manual_frontier = [row for row in frontier_rows if row.get("frontier_category", "").startswith("manual_visual_review")]
    physical_conflicts = [row for row in frontier_rows if row.get("frontier_category") == "do_not_merge_now_structural_or_semantic_conflict"]
    manual_preview_triage = build_manual_preview_triage(
        physical_manual_frontier,
        manual_preview_rows,
        manual_preview_matrix_files,
        manual_preview_qc,
    )
    decision_ledger = build_decision_ledger(
        physical_rows,
        physical_matrix_files,
        manual_preview_rows,
        manual_preview_matrix_files,
        logical_site_ready,
        logical_review,
        logical_members,
        logical_family_quality,
        religion_sequence_coverage,
        physical_manual_frontier,
        physical_missing_ocr,
        physical_missing_pages,
        physical_conflicts,
    )
    grouping_proposals, grouping_proposal_members = build_grouping_proposals(
        physical_rows,
        physical_matrix_files,
        manual_preview_rows,
        manual_preview_triage,
        logical_site_ready,
        logical_review,
        logical_members,
        physical_missing_ocr,
        physical_missing_pages,
        physical_conflicts,
    )
    append_confirmed_logical_edge_proposals(grouping_proposals, grouping_proposal_members, edge_decision_ledger)
    missed_grouping_watchlist = build_missed_grouping_watchlist(edge_decision_ledger, grouping_proposals)
    assembly_work_queue = build_assembly_work_queue(grouping_proposals)
    assembly_priority_classified = build_assembly_priority_classified_queue(assembly_work_queue)
    assembly_ocr_recovery_priority = build_assembly_ocr_recovery_priority(assembly_priority_classified)
    watchlist_source_rollup = build_watchlist_source_rollup(missed_grouping_watchlist)
    watchlist_family_rollup = build_watchlist_family_rollup(missed_grouping_watchlist)
    structure_ready = [row for row in physical_rows if row.get("assembly_status") == "materialized_physical_structure"]
    structure_review = [row for row in physical_rows if row.get("assembly_status") != "materialized_physical_structure"]
    qc_review = [row for row in physical_qc if row.get("status") != "pass"]

    workbook_path = out_dir / "assembly_release_snapshot_workbook.xlsx"
    write_workbook(workbook_path, [("Placeholder", [], [])])
    audits = build_audits(
        contract_audit,
        frontier_audit,
        logical_audit,
        manual_audit,
        materialized_audit,
        frontier_summary,
        materialized_summary,
        contract_summary,
        quality_summary,
        grouping_safety_audit,
        decision_ledger,
        rulebook_edges,
        edge_decision_ledger,
        physical_frontier_action_queue,
        row_label_coverage,
        frontier_rows,
        physical_rows,
        physical_matrix_files,
        manual_preview_rows,
        manual_preview_matrix_files,
        manual_preview_qc,
        manual_preview_triage,
        grouping_proposals,
        grouping_proposal_members,
        missed_grouping_watchlist,
        assembly_work_queue,
        watchlist_source_rollup,
        watchlist_family_rollup,
        logical_site_ready,
        logical_review,
        logical_members,
        logical_family_quality,
        religion_sequence_coverage,
        physical_manual_frontier,
        physical_missing_ocr,
        physical_missing_pages,
        physical_conflicts,
        workbook_path,
    )
    summary_rows = build_summary(
        paths,
        intake_summary,
        frontier_summary,
        quality_summary,
        materialized_summary,
        logical_summary,
        manual_summary,
        contract_summary,
        rulebook_edges,
        edge_decision_ledger,
        physical_rows,
        physical_matrix_files,
        physical_matrix_columns,
        manual_preview_rows,
        manual_preview_matrix_files,
        manual_preview_matrix_columns,
        manual_preview_triage,
        grouping_proposals,
        grouping_proposal_members,
        missed_grouping_watchlist,
        assembly_work_queue,
        assembly_priority_classified,
        watchlist_source_rollup,
        watchlist_family_rollup,
        logical_site_ready,
        logical_review,
        logical_members,
        manual_rows,
        audits,
        chain_rows,
        grouping_safety_audit,
        decision_ledger,
        physical_missing_ocr,
        physical_missing_pages,
        physical_manual_frontier,
        physical_conflicts,
        physical_frontier_action_queue,
    )

    sheets = [
        ("Summary", summary_rows, SUMMARY_FIELDS),
        ("Acceptance Audit", audits, AUDIT_FIELDS),
        ("Assembly Work Queue", assembly_work_queue, ASSEMBLY_WORK_QUEUE_FIELDS),
        ("Classified Priority", assembly_priority_classified, ASSEMBLY_PRIORITY_CLASSIFIED_FIELDS),
        ("OCR Recovery Priority", assembly_ocr_recovery_priority, ASSEMBLY_PRIORITY_CLASSIFIED_FIELDS),
        ("Watchlist By Source", watchlist_source_rollup, WATCHLIST_SOURCE_ROLLUP_FIELDS),
        ("Watchlist By Family", watchlist_family_rollup, WATCHLIST_FAMILY_ROLLUP_FIELDS),
        ("Grouping Proposals", grouping_proposals, GROUPING_PROPOSAL_FIELDS),
        ("Grouping Proposal Members", grouping_proposal_members, GROUPING_PROPOSAL_MEMBER_FIELDS),
        ("Missed Grouping Watchlist", missed_grouping_watchlist, MISSED_GROUPING_WATCHLIST_FIELDS),
        ("Assembly Decision Ledger", decision_ledger, DECISION_LEDGER_FIELDS),
        ("Rulebook Edge Ledger", edge_decision_ledger, EDGE_DECISION_LEDGER_FIELDS),
        ("Physical Action Queue", physical_frontier_action_queue, PHYSICAL_FRONTIER_ACTION_QUEUE_FIELDS),
        ("Physical All", physical_rows, csv_fields(physical_rows)),
        ("Physical Structure Ready", structure_ready, csv_fields(physical_rows)),
        ("Physical Review Required", structure_review, csv_fields(physical_rows)),
        ("Physical QC Review", qc_review, csv_fields(physical_qc)),
        ("Physical Matrix Files", physical_matrix_files, csv_fields(physical_matrix_files)),
        ("Physical Matrix Columns", physical_matrix_columns, csv_fields(physical_matrix_columns)),
        ("Manual Preview Triage", manual_preview_triage, MANUAL_PREVIEW_TRIAGE_FIELDS),
        ("Manual Preview Assemblies", manual_preview_rows, csv_fields(manual_preview_rows)),
        ("Manual Preview Rows", manual_preview_detail_rows, csv_fields(manual_preview_detail_rows)),
        ("Manual Preview Matrix Rows", manual_preview_matrix_rows, csv_fields(manual_preview_matrix_rows)),
        ("Manual Preview QC", manual_preview_qc, csv_fields(manual_preview_qc)),
        ("Manual Preview Matrix Files", manual_preview_matrix_files, csv_fields(manual_preview_matrix_files)),
        ("Manual Preview Matrix Columns", manual_preview_matrix_columns, csv_fields(manual_preview_matrix_columns)),
        ("Manual Preview Cells", manual_preview_cells, csv_fields(manual_preview_cells)),
        ("Physical Chain Audit", chain_rows, CHAIN_AUDIT_FIELDS),
        ("Grouping Safety Audit", grouping_safety_audit, AUDIT_FIELDS),
        ("Row Label Coverage Audit", row_label_coverage, csv_fields(row_label_coverage)),
        ("Logical Family Quality", logical_family_quality, csv_fields(logical_family_quality)),
        ("Religion Sequence Coverage", religion_sequence_coverage, csv_fields(religion_sequence_coverage)),
        ("Frontier Missing OCR", physical_missing_ocr, csv_fields(frontier_rows)),
        ("Frontier Missing Pages", physical_missing_pages, csv_fields(frontier_rows)),
        ("Frontier Manual Review", physical_manual_frontier, csv_fields(frontier_rows)),
        ("Frontier Conflicts", physical_conflicts, csv_fields(frontier_rows)),
        ("Logical Siblings Ready", logical_site_ready, csv_fields(logical_site_ready)),
        ("Logical Siblings Review", logical_review, csv_fields(logical_review)),
        ("Logical Sibling Members", logical_members, csv_fields(logical_members)),
        ("Logical Siblings All", logical_all, csv_fields(logical_all)),
        ("Manual Edge Review", manual_rows, csv_fields(manual_rows)),
        ("Legacy Contract Audit", contract_audit, csv_fields(contract_audit)),
        ("Materialized Audit", materialized_audit, csv_fields(materialized_audit)),
        ("Quality Audit", quality_audit, csv_fields(quality_audit)),
    ]
    write_workbook(workbook_path, sheets)
    audits = build_audits(
        contract_audit,
        frontier_audit,
        logical_audit,
        manual_audit,
        materialized_audit,
        frontier_summary,
        materialized_summary,
        contract_summary,
        quality_summary,
        grouping_safety_audit,
        decision_ledger,
        rulebook_edges,
        edge_decision_ledger,
        physical_frontier_action_queue,
        row_label_coverage,
        frontier_rows,
        physical_rows,
        physical_matrix_files,
        manual_preview_rows,
        manual_preview_matrix_files,
        manual_preview_qc,
        manual_preview_triage,
        grouping_proposals,
        grouping_proposal_members,
        missed_grouping_watchlist,
        assembly_work_queue,
        watchlist_source_rollup,
        watchlist_family_rollup,
        logical_site_ready,
        logical_review,
        logical_members,
        logical_family_quality,
        religion_sequence_coverage,
        physical_manual_frontier,
        physical_missing_ocr,
        physical_missing_pages,
        physical_conflicts,
        workbook_path,
    )
    summary_rows = build_summary(
        paths,
        intake_summary,
        frontier_summary,
        quality_summary,
        materialized_summary,
        logical_summary,
        manual_summary,
        contract_summary,
        rulebook_edges,
        edge_decision_ledger,
        physical_rows,
        physical_matrix_files,
        physical_matrix_columns,
        manual_preview_rows,
        manual_preview_matrix_files,
        manual_preview_matrix_columns,
        manual_preview_triage,
        grouping_proposals,
        grouping_proposal_members,
        missed_grouping_watchlist,
        assembly_work_queue,
        assembly_priority_classified,
        watchlist_source_rollup,
        watchlist_family_rollup,
        logical_site_ready,
        logical_review,
        logical_members,
        manual_rows,
        audits,
        chain_rows,
        grouping_safety_audit,
        decision_ledger,
        physical_missing_ocr,
        physical_missing_pages,
        physical_manual_frontier,
        physical_conflicts,
        physical_frontier_action_queue,
    )
    sheets[0] = ("Summary", summary_rows, SUMMARY_FIELDS)
    sheets[1] = ("Acceptance Audit", audits, AUDIT_FIELDS)
    write_workbook(workbook_path, sheets)

    write_csv(out_dir / "summary.csv", summary_rows, SUMMARY_FIELDS)
    write_csv(out_dir / "acceptance_audit.csv", audits, AUDIT_FIELDS)
    write_csv(out_dir / "assembly_prioritized_work_queue.csv", assembly_work_queue, ASSEMBLY_WORK_QUEUE_FIELDS)
    write_csv(out_dir / "assembly_priority_classified_work_queue.csv", assembly_priority_classified, ASSEMBLY_PRIORITY_CLASSIFIED_FIELDS)
    write_csv(out_dir / "assembly_ocr_recovery_priority.csv", assembly_ocr_recovery_priority, ASSEMBLY_PRIORITY_CLASSIFIED_FIELDS)
    write_csv(out_dir / "missed_grouping_watchlist_by_source.csv", watchlist_source_rollup, WATCHLIST_SOURCE_ROLLUP_FIELDS)
    write_csv(out_dir / "missed_grouping_watchlist_by_family.csv", watchlist_family_rollup, WATCHLIST_FAMILY_ROLLUP_FIELDS)
    write_csv(out_dir / "assembly_grouping_proposals.csv", grouping_proposals, GROUPING_PROPOSAL_FIELDS)
    write_csv(out_dir / "assembly_grouping_proposal_members.csv", grouping_proposal_members, GROUPING_PROPOSAL_MEMBER_FIELDS)
    write_csv(out_dir / "missed_grouping_watchlist.csv", missed_grouping_watchlist, MISSED_GROUPING_WATCHLIST_FIELDS)
    write_csv(out_dir / "assembly_decision_ledger.csv", decision_ledger, DECISION_LEDGER_FIELDS)
    write_csv(out_dir / "rulebook_edge_decision_ledger.csv", edge_decision_ledger, EDGE_DECISION_LEDGER_FIELDS)
    write_csv(out_dir / "physical_frontier_action_queue.csv", physical_frontier_action_queue, PHYSICAL_FRONTIER_ACTION_QUEUE_FIELDS)
    write_csv(out_dir / "materialized_physical_all.csv", physical_rows, csv_fields(physical_rows))
    write_csv(out_dir / "materialized_physical_structure_ready.csv", structure_ready, csv_fields(physical_rows))
    write_csv(out_dir / "materialized_physical_review_required.csv", structure_review, csv_fields(physical_rows))
    write_csv(out_dir / "materialized_physical_matrix_files.csv", physical_matrix_files, csv_fields(physical_matrix_files))
    write_csv(out_dir / "materialized_physical_matrix_columns.csv", physical_matrix_columns, csv_fields(physical_matrix_columns))
    write_csv(out_dir / "manual_physical_preview_triage.csv", manual_preview_triage, MANUAL_PREVIEW_TRIAGE_FIELDS)
    write_csv(out_dir / "manual_physical_preview_assemblies.csv", manual_preview_rows, csv_fields(manual_preview_rows))
    write_csv(out_dir / "manual_physical_preview_rows.csv", manual_preview_detail_rows, csv_fields(manual_preview_detail_rows))
    write_csv(out_dir / "manual_physical_preview_matrix_rows.csv", manual_preview_matrix_rows, csv_fields(manual_preview_matrix_rows))
    write_csv(out_dir / "manual_physical_preview_qc.csv", manual_preview_qc, csv_fields(manual_preview_qc))
    write_csv(out_dir / "manual_physical_preview_matrix_files.csv", manual_preview_matrix_files, csv_fields(manual_preview_matrix_files))
    write_csv(out_dir / "manual_physical_preview_matrix_columns.csv", manual_preview_matrix_columns, csv_fields(manual_preview_matrix_columns))
    write_csv(out_dir / "manual_physical_preview_cells.csv", manual_preview_cells, csv_fields(manual_preview_cells))
    write_csv(out_dir / "physical_safe_edge_chain_audit.csv", chain_rows, CHAIN_AUDIT_FIELDS)
    write_csv(out_dir / "grouping_safety_audit.csv", grouping_safety_audit, AUDIT_FIELDS)
    write_csv(out_dir / "row_label_coverage_audit.csv", row_label_coverage, csv_fields(row_label_coverage))
    write_csv(out_dir / "logical_family_quality_audit.csv", logical_family_quality, csv_fields(logical_family_quality))
    write_csv(out_dir / "religion_sequence_coverage_audit.csv", religion_sequence_coverage, csv_fields(religion_sequence_coverage))
    write_csv(out_dir / "physical_frontier_missing_ocr_edges.csv", physical_missing_ocr, csv_fields(frontier_rows))
    write_csv(out_dir / "physical_frontier_missing_page_bridges.csv", physical_missing_pages, csv_fields(frontier_rows))
    write_csv(out_dir / "physical_frontier_manual_review_edges.csv", physical_manual_frontier, csv_fields(frontier_rows))
    write_csv(out_dir / "physical_frontier_conflict_edges.csv", physical_conflicts, csv_fields(frontier_rows))
    write_csv(out_dir / "logical_sibling_sets_all.csv", logical_all, csv_fields(logical_all))
    write_csv(out_dir / "logical_sibling_sets_site_ready.csv", logical_site_ready, csv_fields(logical_site_ready))
    write_csv(out_dir / "logical_sibling_sets_review.csv", logical_review, csv_fields(logical_review))
    write_csv(out_dir / "logical_sibling_set_members.csv", logical_members, csv_fields(logical_members))
    write_csv(out_dir / "manual_edge_review_queue.csv", manual_rows, csv_fields(manual_rows))

    snapshot = {
        "schema_version": "assembly_release_snapshot_v47",
        "created_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "inputs": {name: output_path(path) for name, path in paths.items() if name != "out_dir"},
        "outputs": {
            "workbook": output_path(workbook_path),
            "summary": output_path(out_dir / "summary.csv"),
            "acceptance_audit": output_path(out_dir / "acceptance_audit.csv"),
            "assembly_prioritized_work_queue": output_path(out_dir / "assembly_prioritized_work_queue.csv"),
            "assembly_priority_classified_work_queue": output_path(out_dir / "assembly_priority_classified_work_queue.csv"),
            "assembly_ocr_recovery_priority": output_path(out_dir / "assembly_ocr_recovery_priority.csv"),
            "missed_grouping_watchlist_by_source": output_path(out_dir / "missed_grouping_watchlist_by_source.csv"),
            "missed_grouping_watchlist_by_family": output_path(out_dir / "missed_grouping_watchlist_by_family.csv"),
            "assembly_grouping_proposals": output_path(out_dir / "assembly_grouping_proposals.csv"),
            "assembly_grouping_proposal_members": output_path(out_dir / "assembly_grouping_proposal_members.csv"),
            "missed_grouping_watchlist": output_path(out_dir / "missed_grouping_watchlist.csv"),
            "assembly_decision_ledger": output_path(out_dir / "assembly_decision_ledger.csv"),
            "rulebook_edge_decision_ledger": output_path(out_dir / "rulebook_edge_decision_ledger.csv"),
            "physical_frontier_action_queue": output_path(out_dir / "physical_frontier_action_queue.csv"),
            "materialized_physical_all": output_path(out_dir / "materialized_physical_all.csv"),
            "materialized_physical_structure_ready": output_path(out_dir / "materialized_physical_structure_ready.csv"),
            "materialized_physical_review_required": output_path(out_dir / "materialized_physical_review_required.csv"),
            "materialized_physical_matrix_files": output_path(out_dir / "materialized_physical_matrix_files.csv"),
            "materialized_physical_matrix_columns": output_path(out_dir / "materialized_physical_matrix_columns.csv"),
            "manual_physical_preview_assemblies": output_path(out_dir / "manual_physical_preview_assemblies.csv"),
            "manual_physical_preview_triage": output_path(out_dir / "manual_physical_preview_triage.csv"),
            "manual_physical_preview_rows": output_path(out_dir / "manual_physical_preview_rows.csv"),
            "manual_physical_preview_matrix_rows": output_path(out_dir / "manual_physical_preview_matrix_rows.csv"),
            "manual_physical_preview_qc": output_path(out_dir / "manual_physical_preview_qc.csv"),
            "manual_physical_preview_matrix_files": output_path(out_dir / "manual_physical_preview_matrix_files.csv"),
            "manual_physical_preview_matrix_columns": output_path(out_dir / "manual_physical_preview_matrix_columns.csv"),
            "manual_physical_preview_cells": output_path(out_dir / "manual_physical_preview_cells.csv"),
            "physical_safe_edge_chain_audit": output_path(out_dir / "physical_safe_edge_chain_audit.csv"),
            "grouping_safety_audit": output_path(out_dir / "grouping_safety_audit.csv"),
            "row_label_coverage_audit": output_path(out_dir / "row_label_coverage_audit.csv"),
            "logical_family_quality_audit": output_path(out_dir / "logical_family_quality_audit.csv"),
            "religion_sequence_coverage_audit": output_path(out_dir / "religion_sequence_coverage_audit.csv"),
            "physical_frontier_missing_ocr_edges": output_path(out_dir / "physical_frontier_missing_ocr_edges.csv"),
            "physical_frontier_missing_page_bridges": output_path(out_dir / "physical_frontier_missing_page_bridges.csv"),
            "physical_frontier_manual_review_edges": output_path(out_dir / "physical_frontier_manual_review_edges.csv"),
            "physical_frontier_conflict_edges": output_path(out_dir / "physical_frontier_conflict_edges.csv"),
            "logical_sibling_sets_all": output_path(out_dir / "logical_sibling_sets_all.csv"),
            "logical_sibling_sets_site_ready": output_path(out_dir / "logical_sibling_sets_site_ready.csv"),
            "logical_sibling_sets_review": output_path(out_dir / "logical_sibling_sets_review.csv"),
            "logical_sibling_set_members": output_path(out_dir / "logical_sibling_set_members.csv"),
            "manual_edge_review_queue": output_path(out_dir / "manual_edge_review_queue.csv"),
        },
        "counts": {
            "frontier_safe_physical_edges": int_value(frontier_summary.get("safe_physical_edges_now")),
            "rulebook_edge_decision_rows": len(edge_decision_ledger),
            "materialized_physical_assemblies": len(physical_rows),
            "manual_preview_candidates": len(manual_preview_rows),
            "manual_preview_triage_rows": len(manual_preview_triage),
            "grouping_proposal_rows": len(grouping_proposals),
            "grouping_proposal_member_rows": len(grouping_proposal_members),
            "missed_grouping_watchlist_rows": len(missed_grouping_watchlist),
            "assembly_work_queue_rows": len(assembly_work_queue),
            "assembly_priority_classified_rows": len(assembly_priority_classified),
            "assembly_ocr_recovery_priority_rows": len(assembly_ocr_recovery_priority),
            "assembly_ocr_recovery_core_table_vii_rows": sum(1 for row in assembly_ocr_recovery_priority if row.get("core_scope") == "core_table_vii"),
            "assembly_ocr_recovery_related_civil_condition_rows": sum(1 for row in assembly_ocr_recovery_priority if row.get("core_scope") == "related_civil_condition"),
            "assembly_ocr_recovery_noncore_context_rows": sum(1 for row in assembly_ocr_recovery_priority if row.get("core_scope") == "noncore_context"),
            "assembly_priority_core_table_vii_rows": sum(1 for row in assembly_priority_classified if row.get("core_scope") == "core_table_vii"),
            "assembly_priority_related_civil_condition_rows": sum(1 for row in assembly_priority_classified if row.get("core_scope") == "related_civil_condition"),
            "assembly_priority_noncore_context_rows": sum(1 for row in assembly_priority_classified if row.get("core_scope") == "noncore_context"),
            "assembly_priority_ocr_then_retest_rows": sum(1 for row in assembly_priority_classified if row.get("next_step_type") == "ocr_missing_artifact_then_retest"),
            "assembly_priority_recover_missing_page_rows": sum(1 for row in assembly_priority_classified if row.get("next_step_type") == "recover_missing_page_then_adjacent_retest"),
            "watchlist_source_rollup_rows": len(watchlist_source_rollup),
            "watchlist_family_rollup_rows": len(watchlist_family_rollup),
            "assembly_decision_ledger_rows": len(decision_ledger),
            "physical_frontier_action_queue_rows": len(physical_frontier_action_queue),
            "materialized_physical_structure_ready": len(structure_ready),
            "materialized_physical_review_required": len(structure_review),
            "materialized_rows": int_value(materialized_summary.get("materialized_rows")),
            "materialized_matrix_rows": int_value(materialized_summary.get("materialized_matrix_rows")),
            "materialized_matrix_columns": int_value(materialized_summary.get("materialized_matrix_columns")) or len(physical_matrix_columns),
            "compact_matrix_files": int_value(materialized_summary.get("compact_matrix_files")) or len(physical_matrix_files),
            "materialized_cells": int_value(materialized_summary.get("materialized_cells")),
            "manual_preview_matrix_files": int_value(materialized_summary.get("manual_preview_matrix_files")) or len(manual_preview_matrix_files),
            "manual_preview_matrix_columns": int_value(materialized_summary.get("manual_preview_matrix_columns")) or len(manual_preview_matrix_columns),
            "manual_preview_rows": int_value(materialized_summary.get("manual_preview_rows")),
            "manual_preview_cells": int_value(materialized_summary.get("manual_preview_cells")),
            "site_ready_logical_sibling_sets": len(logical_site_ready),
            "review_logical_sibling_sets": len(logical_review),
            "logical_sibling_member_rows": len(logical_members),
            "manual_adjudication_edges": len(manual_rows),
            "missing_ocr_edges": int_value(frontier_summary.get("missing_ocr_edges")),
            "physical_frontier_missing_ocr_edges": len(physical_missing_ocr),
            "physical_frontier_missing_page_bridges": len(physical_missing_pages),
            "physical_frontier_manual_review_edges": len(physical_manual_frontier),
            "physical_frontier_conflict_edges": len(physical_conflicts),
            "row_label_coverage_audit_rows": int_value(quality_summary.get("row_label_audit_rows")),
            "row_label_potential_missed": int_value(quality_summary.get("row_label_potential_missed")),
            "logical_overbroad_risks": int_value(quality_summary.get("logical_overbroad_risks")),
            "possible_missing_religion_sequences": int_value(quality_summary.get("possible_missing_religion_sequences")),
            "safe_edge_graph_components": len(chain_rows),
            "safe_edge_graph_failures": sum(1 for row in chain_rows if str(row.get("status", "")).startswith("fail")),
            "grouping_safety_failures": audit_failure_count([dict(row) for row in grouping_safety_audit]),
            "intake_target_count": int_value(intake_summary.get("target_count")),
            "intake_targets_with_artifact_match": int_value(intake_summary.get("targets_with_artifact_match")),
            "intake_render_available_no_ocr_artifact": intake_status_count(intake_summary, "render_available_no_ocr_artifact"),
            "intake_still_blocked_no_local_artifact": intake_status_count(intake_summary, "still_blocked_no_local_artifact"),
            "intake_artifact_match_fingerprint": str(intake_summary.get("artifact_match_fingerprint", "")),
            "intake_target_state_fingerprint": str(intake_summary.get("target_state_fingerprint", "")),
            "legacy_contract_physical_components": int_value(contract_summary.get("site_ready_physical_components")),
            "audit_failures": audit_failure_count([dict(row) for row in audits]),
            "audit_warnings": audit_warning_count([dict(row) for row in audits]),
        },
        "policy": {
            "physical": "Use materialized_physical_structure_ready.csv for current structure-ready physical assemblies; keep materialized_physical_review_required.csv for manual inspection.",
            "manual_preview": "manual_physical_preview_* files are review-only candidate matrices and must not be treated as physical merges until manually promoted.",
            "values": "Physical structures do not imply publishable values unless the value-review policy says so.",
            "logical": "Logical sibling sets are logical-only collections and must never be physically merged.",
            "future": "Missing-OCR and review candidates stay out of physical assembly until their adjacent edge retests cleanly.",
        },
    }
    (out_dir / "assembly_release_snapshot.json").write_text(json.dumps(snapshot, indent=2, sort_keys=True), encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(snapshot["counts"], indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(snapshot["counts"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
