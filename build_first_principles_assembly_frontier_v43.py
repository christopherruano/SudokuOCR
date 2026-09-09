#!/usr/bin/env python3
"""Build a conservative assembly frontier for the static-site handoff.

This layer performs no OCR/API work and does not promote new physical merges.
It packages the current first-principles evidence into a sortable frontier:

* physical edges already safe to use,
* row-label dependency edges needing visual review,
* candidates blocked by missing OCR artifacts,
* nonconsecutive likely bridges that must not be merged yet, and
* structural/semantic conflicts that should stay out of physical assembly.

The point is to avoid both failure modes: missed plausible groupings and
over-large unsafe groupings.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import zipfile
from collections import Counter
from pathlib import Path
from typing import Iterable

import xlsxwriter


ROOT = Path(__file__).resolve().parent
DEFAULT_PHYSICAL_EVIDENCE_DIR = ROOT / "export_final/pdf_recovery_incoming_2026-06-16_first_principles_assembly_evidence_smoke_v42_promotion"
DEFAULT_CANDIDATE_PROMOTION_DIR = ROOT / "export_final/pdf_recovery_incoming_2026-06-16_first_principles_candidate_promotion_smoke_v42_promotion"
DEFAULT_GROUPING_DIR = ROOT / "export_final/pdf_recovery_incoming_2026-06-16_first_principles_static_grouping_manifest_smoke_v42_promotion"
DEFAULT_OUT_DIR = ROOT / "export_final/pdf_recovery_incoming_2026-06-16_first_principles_assembly_frontier_v43"

FRONTIER_FIELDS = [
    "frontier_rank",
    "edge_key",
    "frontier_category",
    "frontier_action",
    "frontier_reason",
    "frontier_priority",
    "manual_review_required",
    "physical_merge_allowed_now",
    "safe_physical_edge_now",
    "must_remain_edge_only",
    "v42_decision",
    "v42_reason",
    "semantic_column_status",
    "semantic_column_detail",
    "risk_flags",
    "pair_direction_signal",
    "pair_direction_reason",
    "v5_source_key",
    "source_pdf_id_guess",
    "page_pair",
    "page_gap",
    "pages_consecutive",
    "missing_intermediate_pages",
    "physical_rule_status",
    "evidence_bucket",
    "row_label_dependency_signal",
    "right_page_no_row_labels_signal",
    "row_alignment_evidence",
    "left_row_count",
    "right_row_count",
    "left_row_label_count",
    "right_row_label_count",
    "shared_row_label_count",
    "left_first_labels",
    "right_first_labels",
    "column_order_evidence",
    "left_column_groups",
    "right_column_groups",
    "combined_column_group_order",
    "zero_check_status",
    "zero_check_failures",
    "zero_check_cells",
    "left_title",
    "right_title",
    "recommended_next_step",
    "rule_decision_reason",
    "rule_flags",
    "left_artifact_status",
    "right_artifact_status",
    "left_best_json",
    "right_best_json",
    "left_best_csv",
    "right_best_csv",
    "prev_image_path",
    "image_path",
]

LOGICAL_FIELDS = [
    "logical_rank",
    "group_id",
    "group_kind",
    "site_use_policy",
    "review_required",
    "v5_source_key",
    "source_pdf_id",
    "page_range",
    "component_pages",
    "n_pages",
    "pages_are_consecutive",
    "table_number_key",
    "topic_variant_key",
    "religion_terms_detected",
    "title_sample",
    "recommended_action",
]

AUDIT_FIELDS = ["check", "status", "detail"]
SUMMARY_FIELDS = ["metric", "name", "count"]

HARD_SWITCH_FLAGS = {
    "derivative_class_switch_break",
    "offtarget_table_break",
    "page_kind_switch",
    "table_number_switch_or_missing",
}

STRONG_ROW_EVIDENCE = {
    "same_row_labels_exact",
    "same_row_labels_substantial_overlap",
    "right_unlabelled_same_row_count_inherit_left_labels",
}

STRONG_COLUMN_EVIDENCE = {"left_page_groups_then_right_page_groups_disjoint"}


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
        return int(str(value).replace(",", ""))
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


def page_pair_pages(value: object) -> list[int]:
    match = re.fullmatch(r"\s*(\d+)\s*-\s*(\d+)\s*", str(value or ""))
    if not match:
        return []
    return [int(match.group(1)), int(match.group(2))]


def compact_pages(pages: Iterable[int]) -> str:
    values = sorted(set(pages))
    if not values:
        return ""
    parts: list[str] = []
    start = prev = values[0]
    for page in values[1:]:
        if page == prev + 1:
            prev = page
            continue
        parts.append(str(start) if start == prev else f"{start}-{prev}")
        start = prev = page
    parts.append(str(start) if start == prev else f"{start}-{prev}")
    return "; ".join(parts)


def missing_between_pair(value: object) -> str:
    pages = page_pair_pages(value)
    if len(pages) != 2:
        return ""
    start, end = sorted(pages)
    if end <= start + 1:
        return ""
    return compact_pages(range(start + 1, end))


def row_label_dependency(row: dict[str, str]) -> str:
    haystack = " ".join(
        [
            row.get("physical_rule_status", ""),
            row.get("evidence_bucket", ""),
            row.get("row_alignment_evidence", ""),
            row.get("rule_flags", ""),
            row.get("rule_decision_reason", ""),
        ]
    ).lower()
    return "yes" if "row_label" in haystack or "row-label" in haystack or "unlabelled" in haystack else "no"


def right_unlabelled_signal(row: dict[str, str]) -> str:
    if "right_unlabelled" in row.get("row_alignment_evidence", ""):
        return "yes"
    right_count = int_or_none(row.get("right_row_label_count"))
    right_rows = int_or_none(row.get("right_row_count"))
    if right_rows and right_count == 0:
        return "yes"
    return "no"


def rule_flag_set(row: dict[str, str]) -> set[str]:
    return set(split_semicolon(row.get("rule_flags", "")))


def normalize_text(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "").lower().replace("—", "-").replace("–", "-")).strip(" .;:-")


def civil_phase(groups: Iterable[object]) -> set[str]:
    phases: set[str] = set()
    for group in groups:
        text = normalize_text(group)
        if not text:
            continue
        if re.search(r"\bunmarried\b", text):
            phases.add("unmarried")
        elif re.search(r"\bmarried\b", text):
            phases.add("married")
        if re.search(r"\bwidow(?:ed|s)?\b", text):
            phases.add("widowed")
        if re.search(r"\b(population|persons|total)\b", text):
            phases.add("population")
    return phases


def pair_direction_signal(row: dict[str, str]) -> tuple[str, str]:
    left = civil_phase(split_semicolon(row.get("left_column_groups", "")))
    right = civil_phase(split_semicolon(row.get("right_column_groups", "")))
    if not left or not right:
        return "unknown_column_direction", "civil-status column phases unavailable"

    left_begin = bool(left & {"population", "unmarried"})
    left_late = bool(left & {"married", "widowed"})
    right_begin = bool(right & {"population", "unmarried"})
    right_late = bool(right & {"married", "widowed"})

    if left_begin and right_late and not left_late:
        return "civil_status_begin_to_late_continuation", f"left={semi(sorted(left))}; right={semi(sorted(right))}"
    if left_late and right_begin and not right_late:
        return "civil_status_end_to_restart_boundary", f"left={semi(sorted(left))}; right={semi(sorted(right))}"
    if left_begin and left_late and right_begin and right_late:
        return "civil_status_duplicate_or_full_overlap", f"left={semi(sorted(left))}; right={semi(sorted(right))}"
    if left == right:
        return "civil_status_same_phase", f"left={semi(sorted(left))}; right={semi(sorted(right))}"
    return "civil_status_mixed_or_partial_direction", f"left={semi(sorted(left))}; right={semi(sorted(right))}"


def artifact_missing(row: dict[str, str]) -> bool:
    return (
        row.get("evidence_bucket") == "needs_missing_ocr_artifact"
        or row.get("left_artifact_status") == "missing"
        or row.get("right_artifact_status") == "missing"
    )


def risk_flags(row: dict[str, str], decision: dict[str, str]) -> list[str]:
    flags: list[str] = []
    rule_flags = rule_flag_set(row)
    for flag in sorted(HARD_SWITCH_FLAGS & rule_flags):
        flags.append(flag)
    if row.get("pages_consecutive") != "yes":
        flags.append("nonconsecutive_pdf_pages")
    if artifact_missing(row):
        flags.append("missing_ocr_artifact")
    if "failed_or_partial_constraints" in rule_flags:
        flags.append("failed_or_partial_constraints")
    if row.get("zero_check_status") == "no_zero_check_available":
        flags.append("zero_check_missing")
    if row.get("zero_check_status") == "zero_check_failures_review":
        flags.append("zero_check_failures")
    if row.get("column_order_evidence") == "missing_column_group_evidence":
        flags.append("missing_column_group_evidence")
    if row.get("column_order_evidence") == "overlapping_column_groups_review":
        flags.append("overlapping_column_groups")
    if row.get("column_order_evidence") == "same_column_groups_vertical_or_duplicate_review":
        flags.append("duplicate_or_vertical_column_groups")
    if "different_rows" in row.get("row_alignment_evidence", ""):
        flags.append("row_label_mismatch")
    if "right_unlabelled" in row.get("row_alignment_evidence", ""):
        flags.append("right_page_no_row_labels")
    if "both_unlabelled" in row.get("row_alignment_evidence", ""):
        flags.append("both_pages_unlabelled")
    semantic_status = decision.get("semantic_column_status", "")
    if "reversed" in semantic_status:
        flags.append("semantic_column_sequence_reversed")
    if "duplicate" in semantic_status:
        flags.append("semantic_column_sequence_duplicate")
    if "unsupported" in semantic_status:
        flags.append("semantic_column_sequence_unsupported")
    direction, _ = pair_direction_signal(row)
    if direction == "civil_status_end_to_restart_boundary":
        flags.append("likely_table_boundary_end_to_restart")
    if direction == "civil_status_duplicate_or_full_overlap":
        flags.append("civil_status_duplicate_or_full_overlap")
    return flags


def hard_structural_conflict(row: dict[str, str], decision: dict[str, str]) -> bool:
    flags = set(risk_flags(row, decision))
    return bool(
        HARD_SWITCH_FLAGS & rule_flag_set(row)
        or "semantic_column_sequence_reversed" in flags
        or "semantic_column_sequence_duplicate" in flags
        or "duplicate_or_vertical_column_groups" in flags
        or "likely_table_boundary_end_to_restart" in flags
        or "civil_status_duplicate_or_full_overlap" in flags
    )


def classify_edge(row: dict[str, str], decision: dict[str, str]) -> tuple[str, str, str, int]:
    v42_decision = decision.get("decision", "")
    if v42_decision == "promote_physical_structure_review_values":
        return (
            "site_ready_physical_edge_v42_promoted",
            "Use as a two-page physical edge; values still require the normal value policy.",
            "v42 strict promotion passed.",
            1,
        )
    if (
        row.get("physical_rule_status") == "confirmed_physical_edge"
        and row.get("pages_consecutive") == "yes"
        and row.get("physical_merge_allowed_now") == "yes"
    ):
        if row.get("evidence_bucket") == "confirmed_edge_with_strong_ocr_evidence":
            return (
                "site_ready_physical_edge_confirmed_strong",
                "Use as physical structure; value publication still follows value policy.",
                "Confirmed edge has strong OCR evidence.",
                2,
            )
        return (
            "site_ready_physical_edge_confirmed_review_values",
            "Use as physical structure but prioritize visual/value review.",
            "Confirmed edge exists, but OCR evidence is not fully strong.",
            3,
        )
    if row.get("pages_consecutive") != "yes" or row.get("evidence_bucket") == "missing_page_bridge_do_not_merge":
        return (
            "likely_after_missing_pages_no_merge",
            "Do not merge; recover/check missing intermediate pages first.",
            "Candidate spans a PDF page gap.",
            20,
        )
    if artifact_missing(row):
        return (
            "needs_missing_ocr_artifact",
            "Wait for OCR artifact, then rerun v39-v43.",
            "At least one side lacks parseable OCR evidence.",
            10,
        )
    if hard_structural_conflict(row, decision):
        return (
            "do_not_merge_now_structural_or_semantic_conflict",
            "Keep out of physical assembly unless manual review explicitly overrides.",
            "Switch/duplicate/reversed evidence conflicts with a safe physical edge.",
            30,
        )
    strong_rows = row.get("row_alignment_evidence") in STRONG_ROW_EVIDENCE
    strong_cols = row.get("column_order_evidence") in STRONG_COLUMN_EVIDENCE
    zero_ok = row.get("zero_check_status") == "zero_checks_pass"
    if v42_decision == "keep_review_only":
        return (
            "manual_visual_review_strong_candidate_blocked_by_v42",
            "Manual review: inspect image pair before any promotion.",
            "v39 found strong evidence, but v42 kept it review-only.",
            4,
        )
    direction, _ = pair_direction_signal(row)
    if (
        row_label_dependency(row) == "yes"
        and direction == "civil_status_begin_to_late_continuation"
        and row.get("column_order_evidence") in STRONG_COLUMN_EVIDENCE
        and row.get("zero_check_status") == "zero_checks_pass"
    ):
        return (
            "manual_visual_review_likely_civil_status_continuation",
            "Manual review: likely beginning-to-continuation civil-status table split.",
            "Consecutive edge moves from population/unmarried columns to married/widowed columns.",
            5,
        )
    if row_label_dependency(row) == "yes" and strong_rows and strong_cols and zero_ok:
        return (
            "manual_visual_review_row_label_dependency_high",
            "Manual review: likely row-label dependency, but not auto-promoted.",
            "Consecutive edge has strong row/column/zero evidence.",
            5,
        )
    if row_label_dependency(row) == "yes" and (strong_rows or strong_cols):
        return (
            "manual_visual_review_row_label_dependency_medium",
            "Manual review: possible row-label dependency with partial evidence.",
            "Consecutive edge has at least one strong structural signal.",
            6,
        )
    if row.get("evidence_bucket") == "candidate_or_failed_edge_review":
        return (
            "manual_visual_review_low_confidence_candidate",
            "Review only if higher-priority candidates are exhausted.",
            "Candidate lacks enough evidence for safe promotion.",
            40,
        )
    return (
        "manual_visual_review_other",
        "Review before any physical merge.",
        "No safe automatic classification applied.",
        50,
    )


def build_frontier(edge_rows: list[dict[str, str]], decisions: list[dict[str, str]]) -> list[dict[str, object]]:
    decision_by_edge = {row.get("edge_key", ""): row for row in decisions if row.get("edge_key")}
    rows: list[dict[str, object]] = []
    for edge in edge_rows:
        decision = decision_by_edge.get(edge.get("edge_key", ""), {})
        category, action, reason, priority = classify_edge(edge, decision)
        risks = risk_flags(edge, decision)
        direction, direction_reason = pair_direction_signal(edge)
        safe = "yes" if category.startswith("site_ready_physical_edge_") else "no"
        must_edge_only = "yes" if category in {
            "likely_after_missing_pages_no_merge",
            "needs_missing_ocr_artifact",
            "do_not_merge_now_structural_or_semantic_conflict",
        } else "no"
        row = {
            "edge_key": edge.get("edge_key", ""),
            "frontier_category": category,
            "frontier_action": action,
            "frontier_reason": reason,
            "frontier_priority": priority,
            "manual_review_required": "no" if safe == "yes" and category.endswith("_strong") else ("yes" if safe != "yes" or "review" in category else "value_review"),
            "physical_merge_allowed_now": "yes" if safe == "yes" else "no",
            "safe_physical_edge_now": safe,
            "must_remain_edge_only": must_edge_only,
            "v42_decision": decision.get("decision", ""),
            "v42_reason": decision.get("reason", ""),
            "semantic_column_status": decision.get("semantic_column_status", ""),
            "semantic_column_detail": decision.get("semantic_column_detail", ""),
            "risk_flags": semi(risks),
            "pair_direction_signal": direction,
            "pair_direction_reason": direction_reason,
            "missing_intermediate_pages": missing_between_pair(edge.get("page_pair", "")),
            "row_label_dependency_signal": row_label_dependency(edge),
            "right_page_no_row_labels_signal": right_unlabelled_signal(edge),
        }
        row.update(edge)
        rows.append(row)
    rows.sort(
        key=lambda r: (
            int_or_none(r.get("frontier_priority")) or 999,
            r.get("v5_source_key", ""),
            int_or_none(page_pair_pages(r.get("page_pair", ""))[0] if page_pair_pages(r.get("page_pair", "")) else None) or 999999,
            r.get("edge_key", ""),
        )
    )
    for idx, row in enumerate(rows, start=1):
        row["frontier_rank"] = idx
    return rows


def build_logical_frontier(group_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for group in group_rows:
        if group.get("group_layer") != "logical":
            continue
        rows.append(
            {
                "group_id": group.get("group_id", ""),
                "group_kind": group.get("group_kind", ""),
                "site_use_policy": group.get("site_use_policy", ""),
                "review_required": group.get("review_required", ""),
                "v5_source_key": group.get("v5_source_key", ""),
                "source_pdf_id": group.get("source_pdf_id", ""),
                "page_range": group.get("page_range", ""),
                "component_pages": group.get("component_pages", ""),
                "n_pages": group.get("n_pages", ""),
                "pages_are_consecutive": group.get("pages_are_consecutive", ""),
                "table_number_key": group.get("table_number_key", ""),
                "topic_variant_key": group.get("topic_variant_key", ""),
                "religion_terms_detected": group.get("religion_terms_detected", ""),
                "title_sample": group.get("title_sample", ""),
                "recommended_action": group.get("recommended_action", ""),
            }
        )
    rows.sort(key=lambda r: (r.get("review_required", ""), r.get("v5_source_key", ""), r.get("page_range", ""), r.get("group_id", "")))
    for idx, row in enumerate(rows, start=1):
        row["logical_rank"] = idx
    return rows


def ranked_subset(rows: list[dict[str, object]], predicate, rank_field: str) -> list[dict[str, object]]:
    out = [dict(row) for row in rows if predicate(row)]
    for idx, row in enumerate(out, start=1):
        row[rank_field] = idx
    fields = [rank_field, *FRONTIER_FIELDS]
    return [{field: row.get(field, "") for field in fields} for row in out]


def write_workbook(
    path: Path,
    summary_rows: list[dict[str, object]],
    audit_rows: list[dict[str, object]],
    frontier_rows: list[dict[str, object]],
    manual_rows: list[dict[str, object]],
    missing_rows: list[dict[str, object]],
    no_merge_rows: list[dict[str, object]],
    row_label_rows: list[dict[str, object]],
    logical_rows: list[dict[str, object]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    workbook = xlsxwriter.Workbook(path)
    wrap = workbook.add_format({"text_wrap": True, "valign": "top"})
    header = workbook.add_format({"bold": True, "bg_color": "#D9EAF7", "border": 1})
    sheets = [
        ("Summary", summary_rows, SUMMARY_FIELDS),
        ("Audit", audit_rows, AUDIT_FIELDS),
        ("Physical Frontier", frontier_rows, FRONTIER_FIELDS),
        ("Manual Review", manual_rows, ["review_rank", *FRONTIER_FIELDS]),
        ("Missing OCR", missing_rows, ["missing_rank", *FRONTIER_FIELDS]),
        ("No Merge Now", no_merge_rows, ["no_merge_rank", *FRONTIER_FIELDS]),
        ("Row Label Dependency", row_label_rows, ["row_label_rank", *FRONTIER_FIELDS]),
        ("Logical Frontier", logical_rows, LOGICAL_FIELDS),
    ]
    for name, rows, fields in sheets:
        ws = workbook.add_worksheet(name[:31])
        for col, field in enumerate(fields):
            ws.write(0, col, field, header)
        for r_idx, row in enumerate(rows, start=1):
            for c_idx, field in enumerate(fields):
                fmt = wrap if field in {
                    "frontier_action",
                    "frontier_reason",
                    "risk_flags",
                    "v42_reason",
                    "left_title",
                    "right_title",
                    "recommended_next_step",
                    "rule_decision_reason",
                    "left_column_groups",
                    "right_column_groups",
                    "combined_column_group_order",
                    "prev_image_path",
                    "image_path",
                    "title_sample",
                    "recommended_action",
                } else None
                ws.write(r_idx, c_idx, row.get(field, ""), fmt)
        ws.freeze_panes(1, 0)
        ws.autofilter(0, 0, max(len(rows), 1), max(len(fields) - 1, 0))
        for col, field in enumerate(fields):
            width = 14
            if field in {"edge_key", "group_id"}:
                width = 34
            elif field in {"frontier_category", "frontier_action", "frontier_reason", "risk_flags", "recommended_next_step", "title_sample"}:
                width = 48
            elif field.endswith("_path") or field.endswith("_json") or field.endswith("_csv"):
                width = 52
            ws.set_column(col, col, width)
    workbook.close()


def build_audits(
    frontier_rows: list[dict[str, object]],
    decisions: list[dict[str, str]],
    manual_rows: list[dict[str, object]],
    workbook_path: Path,
) -> list[dict[str, object]]:
    audits: list[dict[str, object]] = []
    safe_bad = [
        row for row in frontier_rows
        if row.get("safe_physical_edge_now") == "yes" and row.get("pages_consecutive") != "yes"
    ]
    audits.append({"check": "safe_physical_edges_are_consecutive", "status": "pass" if not safe_bad else "fail", "detail": f"bad={len(safe_bad)}"})
    nonedge_bad = [
        row for row in frontier_rows
        if row.get("frontier_category") in {"likely_after_missing_pages_no_merge", "needs_missing_ocr_artifact"} and row.get("physical_merge_allowed_now") == "yes"
    ]
    audits.append({"check": "blocked_or_gap_edges_not_marked_mergeable", "status": "pass" if not nonedge_bad else "fail", "detail": f"bad={len(nonedge_bad)}"})
    manual_nonconsecutive = [row for row in manual_rows if row.get("pages_consecutive") != "yes"]
    audits.append({"check": "manual_review_edges_are_adjacent_only", "status": "pass" if not manual_nonconsecutive else "fail", "detail": f"bad={len(manual_nonconsecutive)}"})
    promoted = [row for row in frontier_rows if row.get("v42_decision") == "promote_physical_structure_review_values"]
    promoted_bad = [row for row in promoted if row.get("frontier_category") != "site_ready_physical_edge_v42_promoted"]
    audits.append({"check": "v42_promotions_preserved_as_site_ready_edges", "status": "pass" if not promoted_bad else "fail", "detail": f"promoted={len(promoted)} bad={len(promoted_bad)}"})
    blocked_decisions = [row for row in decisions if row.get("decision") == "keep_review_only"]
    blocked_edges = {row.get("edge_key") for row in blocked_decisions}
    retained = [
        row for row in frontier_rows
        if row.get("edge_key") in blocked_edges and row.get("safe_physical_edge_now") != "yes"
    ]
    audits.append({"check": "v42_blocked_candidates_retained_review_only", "status": "pass" if len(retained) == len(blocked_edges) else "fail", "detail": f"blocked={len(blocked_edges)} retained={len(retained)}"})
    row_label_rows = [row for row in frontier_rows if row.get("row_label_dependency_signal") == "yes"]
    audits.append({"check": "row_label_dependency_frontier_present", "status": "pass" if row_label_rows else "fail", "detail": f"rows={len(row_label_rows)}"})
    all_categorized = [row for row in frontier_rows if not row.get("frontier_category")]
    audits.append({"check": "all_physical_edges_categorized", "status": "pass" if not all_categorized else "fail", "detail": f"bad={len(all_categorized)} total={len(frontier_rows)}"})
    try:
        with zipfile.ZipFile(workbook_path) as zf:
            bad_file = zf.testzip()
        status = "pass" if bad_file is None else "fail"
        detail = "ok" if bad_file is None else str(bad_file)
    except Exception as exc:  # pragma: no cover - defensive integrity check
        status = "fail"
        detail = str(exc)
    audits.append({"check": "workbook_zip_integrity", "status": status, "detail": detail})
    return audits


def build_summary(
    frontier_rows: list[dict[str, object]],
    logical_rows: list[dict[str, object]],
    manual_rows: list[dict[str, object]],
    missing_rows: list[dict[str, object]],
    no_merge_rows: list[dict[str, object]],
    row_label_rows: list[dict[str, object]],
    audits: list[dict[str, object]],
    paths: dict[str, Path],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = [
        {"metric": "counts", "name": "physical_frontier_edges", "count": len(frontier_rows)},
        {"metric": "counts", "name": "manual_review_edges", "count": len(manual_rows)},
        {"metric": "counts", "name": "missing_ocr_edges", "count": len(missing_rows)},
        {"metric": "counts", "name": "no_merge_now_edges", "count": len(no_merge_rows)},
        {"metric": "counts", "name": "row_label_dependency_edges", "count": len(row_label_rows)},
        {"metric": "counts", "name": "logical_frontier_groups", "count": len(logical_rows)},
        {"metric": "counts", "name": "safe_physical_edges_now", "count": sum(1 for row in frontier_rows if row.get("safe_physical_edge_now") == "yes")},
        {"metric": "counts", "name": "audit_failure_count", "count": sum(1 for row in audits if row.get("status") != "pass")},
    ]
    for metric, counter in [
        ("frontier_by_category", Counter(str(row.get("frontier_category", "")) for row in frontier_rows)),
        ("frontier_by_rule_status", Counter(str(row.get("physical_rule_status", "")) for row in frontier_rows)),
        ("frontier_by_evidence_bucket", Counter(str(row.get("evidence_bucket", "")) for row in frontier_rows)),
        ("logical_by_group_kind", Counter(str(row.get("group_kind", "")) for row in logical_rows)),
    ]:
        for name, count in counter.most_common():
            rows.append({"metric": metric, "name": name, "count": count})
    for name, path in paths.items():
        rows.append({"metric": "outputs", "name": name, "count": output_path(path)})
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physical-evidence-dir", type=Path, default=DEFAULT_PHYSICAL_EVIDENCE_DIR)
    parser.add_argument("--candidate-promotion-dir", type=Path, default=DEFAULT_CANDIDATE_PROMOTION_DIR)
    parser.add_argument("--grouping-dir", type=Path, default=DEFAULT_GROUPING_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    edge_rows = read_csv(args.physical_evidence_dir / "edge_assembly_evidence.csv")
    decisions = read_csv(args.candidate_promotion_dir / "candidate_promotion_decisions.csv")
    group_rows = read_csv(args.candidate_promotion_dir / "canonical_grouping_manifest_v42.csv")
    if not group_rows:
        group_rows = read_csv(args.grouping_dir / "canonical_grouping_manifest.csv")

    frontier_rows = build_frontier(edge_rows, decisions)
    logical_rows = build_logical_frontier(group_rows)
    manual_rows = ranked_subset(
        frontier_rows,
        lambda row: str(row.get("frontier_category", "")).startswith("manual_visual_review"),
        "review_rank",
    )
    missing_rows = ranked_subset(
        frontier_rows,
        lambda row: row.get("frontier_category") == "needs_missing_ocr_artifact"
        or "missing_ocr_artifact" in str(row.get("risk_flags", "")),
        "missing_rank",
    )
    no_merge_rows = ranked_subset(
        frontier_rows,
        lambda row: row.get("frontier_category") in {
            "likely_after_missing_pages_no_merge",
            "do_not_merge_now_structural_or_semantic_conflict",
        },
        "no_merge_rank",
    )
    row_label_rows = ranked_subset(
        frontier_rows,
        lambda row: row.get("row_label_dependency_signal") == "yes",
        "row_label_rank",
    )

    paths = {
        "physical_frontier": out_dir / "physical_assembly_frontier.csv",
        "manual_review": out_dir / "manual_visual_review_queue.csv",
        "missing_ocr": out_dir / "missing_ocr_frontier.csv",
        "no_merge_now": out_dir / "no_merge_now_frontier.csv",
        "row_label_dependency": out_dir / "row_label_dependency_frontier.csv",
        "logical_frontier": out_dir / "logical_collection_frontier.csv",
        "audit": out_dir / "acceptance_audit.csv",
        "summary": out_dir / "summary.csv",
        "workbook": out_dir / "assembly_frontier_workbook.xlsx",
    }

    write_csv(paths["physical_frontier"], frontier_rows, FRONTIER_FIELDS)
    write_csv(paths["manual_review"], manual_rows, ["review_rank", *FRONTIER_FIELDS])
    write_csv(paths["missing_ocr"], missing_rows, ["missing_rank", *FRONTIER_FIELDS])
    write_csv(paths["no_merge_now"], no_merge_rows, ["no_merge_rank", *FRONTIER_FIELDS])
    write_csv(paths["row_label_dependency"], row_label_rows, ["row_label_rank", *FRONTIER_FIELDS])
    write_csv(paths["logical_frontier"], logical_rows, LOGICAL_FIELDS)

    write_workbook(paths["workbook"], [], [], frontier_rows, manual_rows, missing_rows, no_merge_rows, row_label_rows, logical_rows)
    audits = build_audits(frontier_rows, decisions, manual_rows, paths["workbook"])
    summary_rows = build_summary(frontier_rows, logical_rows, manual_rows, missing_rows, no_merge_rows, row_label_rows, audits, paths)
    write_workbook(paths["workbook"], summary_rows, audits, frontier_rows, manual_rows, missing_rows, no_merge_rows, row_label_rows, logical_rows)
    audits = build_audits(frontier_rows, decisions, manual_rows, paths["workbook"])
    summary_rows = build_summary(frontier_rows, logical_rows, manual_rows, missing_rows, no_merge_rows, row_label_rows, audits, paths)
    write_workbook(paths["workbook"], summary_rows, audits, frontier_rows, manual_rows, missing_rows, no_merge_rows, row_label_rows, logical_rows)

    write_csv(paths["audit"], audits, AUDIT_FIELDS)
    write_csv(paths["summary"], summary_rows, SUMMARY_FIELDS)
    (out_dir / "summary.json").write_text(
        json.dumps(
            {
                "physical_frontier_edges": len(frontier_rows),
                "manual_review_edges": len(manual_rows),
                "missing_ocr_edges": len(missing_rows),
                "no_merge_now_edges": len(no_merge_rows),
                "row_label_dependency_edges": len(row_label_rows),
                "logical_frontier_groups": len(logical_rows),
                "safe_physical_edges_now": sum(1 for row in frontier_rows if row.get("safe_physical_edge_now") == "yes"),
                "frontier_by_category": dict(Counter(str(row.get("frontier_category", "")) for row in frontier_rows)),
                "audit_failure_count": sum(1 for row in audits if row.get("status") != "pass"),
                "outputs": {name: output_path(path) for name, path in paths.items()},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
