#!/usr/bin/env python3
"""Candidate audit layer for missed assembly opportunities.

No OCR/API calls are made. This pass consumes the v17 adjacency edge evidence
and the v21 boundary decisions, then classifies all remaining non-joined
boundaries into actionable review buckets or explicit rejections.

The output is intentionally an audit/worklist layer, not a merge layer: it
keeps the strict invariant that only adjacent physical pages can be joined, and
non-adjacent records become missing-bridge candidates rather than components.
"""

from __future__ import annotations

import csv
import json
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import build_all_ocr_component_assembly_v5_structural as base
import build_all_ocr_component_assembly_v13_scope_refined as v13
import build_all_ocr_component_assembly_v14_variant_refined as v14
import build_all_ocr_component_assembly_v16_left_stub_boundary_hints as v16
import build_all_ocr_component_assembly_v21_boundary_refined_index as v21


ROOT = Path(__file__).resolve().parent
V17_DIR = ROOT / "export_final/pdf_recovery_incoming_2026-06-16_all_ocr_component_assembly_v17_collections"
V21_DIR = ROOT / "export_final/pdf_recovery_incoming_2026-06-16_all_ocr_component_assembly_v21_boundary_refined_index"
OUT_DIR = ROOT / "export_final/pdf_recovery_incoming_2026-06-16_all_ocr_component_assembly_v22_candidate_audit"

V17_EDGES = V17_DIR / "v17_edge_recommendations.csv"
V21_BOUNDARY_DECISIONS = V21_DIR / "v21_boundary_decisions.csv"
V21_SUMMARY = V21_DIR / "v21_summary.json"

CORE_FAMILIES = {
    "main_age_civil",
    "age civil condition",
    "age sex population",
    "age sex civil condition",
    "table_vii",
    "age by religion",
    "single year age",
}
DERIVATIVE_FAMILIES = {"derivative_selected"}
NOISY_FAMILIES = {"context", "offtarget", "unknown"}

JOIN_ACTIONS = {
    "keep_existing_join",
    "join_classifier_core_numeric_form_high",
    "join_classifier_main_tablevii_high",
    "join_classifier_same_part_tablevii_continuation_high",
    "join_classifier_split_title_tablevii_high",
    "join_continuation_same_table_high",
    "join_current_right_half_same_scope_required_high",
    "join_explicit_same_scope_same_table_high",
    "join_plain_same_table_high",
    "join_prev_left_stub_same_table_required_high",
    "join_same_scope_short_title_high",
}

SWITCH_ACTIONS = {
    "hard_break_table_number_switch",
    "hard_break_family_switch",
    "hard_break_scope_entity_switch",
    "hard_break_scope_modifier_switch",
    "hard_break_explicit_scope_switch",
}

CANDIDATE_CLASSES = {
    "possible_missing_bridge_extra_review",
    "review_possible_left_stub_continuation",
    "review_row_label_absent_current",
    "review_continued_same_table",
    "review_high_similarity_same_core_table",
    "low_confidence_same_core_table",
}

CANDIDATE_FIELDS = [
    "sequence_edge_id",
    "candidate_class",
    "recommendation",
    "review_priority",
    "safe_whole_page_join",
    "can_auto_join_now",
    "candidate_layer",
    "v5_source_key",
    "source_pdf",
    "source_pdf_id",
    "prev_physical_run_id",
    "physical_run_id",
    "prev_run_page_range",
    "run_page_range",
    "prev_last_pdf_page",
    "first_pdf_page",
    "page_gap",
    "missing_pages",
    "same_family_and_key",
    "is_core_table_family",
    "max_similarity",
    "strict_title_similarity",
    "scope_similarity",
    "prev_page_kind",
    "page_kind",
    "prev_table_family_key",
    "table_family_key",
    "prev_table_number_key",
    "table_number_key",
    "prev_left_stub",
    "prev_right_half",
    "cur_left_stub",
    "cur_right_half",
    "prev_no_row_label_pages",
    "no_row_label_pages",
    "prev_row_label_deficient_pages",
    "row_label_deficient_pages",
    "v17_recommended_action",
    "v17_recommended_layer",
    "v17_recommended_reason",
    "sequence_edge_action",
    "sequence_join_type",
    "sequence_edge_reason",
    "sequence_review_flags",
    "existing_edge_reason",
    "existing_edge_review_flags",
    "reason",
    "prev_primary_title",
    "primary_title",
    "prev_example_images",
    "example_images",
]

REJECTION_FIELDS = [
    *CANDIDATE_FIELDS,
    "rejection_class",
]

SOURCE_SUMMARY_FIELDS = [
    "v5_source_key",
    "source_pdf",
    "candidate_count",
    "review_candidate_count",
    "possible_missing_bridge_extra_count",
    "classes",
    "page_ranges",
]

SUMMARY_FIELDS = ["metric", "value"]
AUDIT_FIELDS = ["check", "status", "failure_count", "details"]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def as_float(value: object) -> float:
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0


def max_similarity(edge: dict[str, str]) -> float:
    return max(as_float(edge.get("strict_title_similarity")), as_float(edge.get("scope_similarity")))


def has_continuation_marker(value: object) -> bool:
    return bool(re.search(r"\b(contd|continued|concld|concluded|ctd)\b|cont[.-]", str(value or "").lower()))


def is_join_action(action: str) -> bool:
    return action in JOIN_ACTIONS or action.startswith("join_")


def source_pdf_from_key(source_key: object) -> str:
    return v21.source_pdf_from_key(source_key)


def source_pdf_id_from_key(source_key: object) -> str:
    return v21.source_pdf_id_from_key(source_key)


def classify_edge(edge: dict[str, str], boundary_decisions: dict[str, dict[str, str]]) -> tuple[str, str, str, str, str, str, str]:
    edge_id = str(edge.get("sequence_edge_id") or "")
    if edge_id in boundary_decisions:
        decision = boundary_decisions[edge_id].get("decision", "")
        return (
            "already_boundary_decided",
            "already represented in v21 boundary_decisions",
            "",
            "no",
            "no",
            "handled",
            decision,
        )

    action = str(edge.get("v17_recommended_action") or "")
    if is_join_action(action):
        return (
            "already_joined",
            "already represented in physical or logical assembly",
            "",
            "yes",
            "yes",
            "handled",
            action,
        )

    gap = str(edge.get("page_gap") or "")
    same_family = v13.same_family_and_key(edge)
    sim = max_similarity(edge)
    prev_family = str(edge.get("prev_table_family_key") or "")
    cur_family = str(edge.get("table_family_key") or "")
    is_core = prev_family in CORE_FAMILIES or cur_family in CORE_FAMILIES

    if gap and gap != "1":
        if same_family and sim >= 0.72 and cur_family not in DERIVATIVE_FAMILIES and prev_family not in DERIVATIVE_FAMILIES:
            priority = "2" if is_core else "4"
            return (
                "possible_missing_bridge_extra_review",
                "do not join; inspect/render missing pages if this source becomes in-scope",
                priority,
                "no",
                "no",
                "missing_bridge_review",
                f"non_adjacent_same_table_similarity_{sim:.2f}",
            )
        return (
            "not_adjacent_no_group",
            "do not join; non-adjacent and insufficient same-table evidence",
            "",
            "no",
            "no",
            "rejected",
            "not_adjacent",
        )

    if gap != "1":
        return (
            "missing_page_number_or_nonadjacent",
            "do not join until real PDF pages are recovered",
            "3",
            "no",
            "no",
            "rejected",
            "missing_or_invalid_page_gap",
        )

    if prev_family in DERIVATIVE_FAMILIES or cur_family in DERIVATIVE_FAMILIES:
        return (
            "reject_derivative_selected",
            "do not merge selected-caste/race/tribe derivative table with current core table flow",
            "",
            "no",
            "no",
            "rejected",
            "derivative_selected_boundary",
        )

    if action in SWITCH_ACTIONS:
        return (
            "reject_explicit_switch",
            "do not join; v17 detected table/family/scope switch",
            "",
            "no",
            "no",
            "rejected",
            action,
        )

    hard_switch, hard_reason = v14.has_obvious_hard_switch(edge)
    if hard_switch:
        return (
            f"reject_obvious_switch_{hard_reason[:60]}",
            "do not join; obvious title/scope/status switch",
            "",
            "no",
            "no",
            "rejected",
            hard_reason,
        )

    flags = v16.row_layout_flags(edge)
    if same_family and is_core and flags["prev_left_stub"] and not flags["cur_left_stub"] and sim >= 0.55:
        return (
            "review_possible_left_stub_continuation",
            "manual review; previous page is a left/body stub and current page may supply continuation columns",
            "1",
            "unknown",
            "no",
            "adjacent_physical_review",
            f"prev_left_stub_same_table_similarity_{sim:.2f}",
        )

    if same_family and is_core and (
        edge.get("no_row_label_pages") or edge.get("row_label_deficient_pages")
    ) and sim >= 0.60:
        return (
            "review_row_label_absent_current",
            "manual review; current page has absent/deficient row labels and may depend on previous page",
            "1",
            "unknown",
            "no",
            "adjacent_physical_review",
            f"current_row_labels_absent_or_deficient_similarity_{sim:.2f}",
        )

    if same_family and is_core and has_continuation_marker(edge.get("primary_title")) and sim >= 0.65:
        return (
            "review_continued_same_table",
            "manual review; adjacent same-table continuation marker remains ungrouped",
            "2",
            "unknown",
            "no",
            "adjacent_logical_or_physical_review",
            f"continued_marker_same_table_similarity_{sim:.2f}",
        )

    if same_family and is_core and sim >= 0.85:
        return (
            "review_high_similarity_same_core_table",
            "manual review; high title similarity but no structural rule supports automatic join",
            "2",
            "unknown",
            "no",
            "adjacent_logical_review",
            f"high_similarity_same_core_table_{sim:.2f}",
        )

    if same_family and is_core and sim >= 0.60:
        return (
            "low_confidence_same_core_table",
            "low-priority review; same core family/table number but weak or conflicting evidence",
            "4",
            "unknown",
            "no",
            "low_confidence_review",
            f"same_core_table_low_confidence_{sim:.2f}",
        )

    if same_family and sim >= 0.85:
        return (
            "offtarget_or_other_same_table_high_similarity",
            "not current core-table assembly; keep as audit-only if broader tables are later included",
            "",
            "unknown",
            "no",
            "out_of_scope_audit",
            f"non_core_same_table_similarity_{sim:.2f}",
        )

    return (
        "reject_low_signal",
        "do not join; insufficient same-table continuation evidence",
        "",
        "no",
        "no",
        "rejected",
        "low_similarity_or_incompatible_boundary",
    )


def output_row(
    edge: dict[str, str],
    candidate_class: str,
    recommendation: str,
    review_priority: str,
    safe_whole_page_join: str,
    can_auto_join_now: str,
    candidate_layer: str,
    reason: str,
) -> dict[str, object]:
    flags = v16.row_layout_flags(edge)
    source_key = edge.get("v5_source_key", "")
    same_family = v13.same_family_and_key(edge)
    is_core = edge.get("prev_table_family_key") in CORE_FAMILIES or edge.get("table_family_key") in CORE_FAMILIES
    return {
        "sequence_edge_id": edge.get("sequence_edge_id", ""),
        "candidate_class": candidate_class,
        "recommendation": recommendation,
        "review_priority": review_priority,
        "safe_whole_page_join": safe_whole_page_join,
        "can_auto_join_now": can_auto_join_now,
        "candidate_layer": candidate_layer,
        "v5_source_key": source_key,
        "source_pdf": source_pdf_from_key(source_key),
        "source_pdf_id": source_pdf_id_from_key(source_key),
        "prev_physical_run_id": edge.get("prev_physical_run_id", ""),
        "physical_run_id": edge.get("physical_run_id", ""),
        "prev_run_page_range": edge.get("prev_run_page_range", ""),
        "run_page_range": edge.get("run_page_range", ""),
        "prev_last_pdf_page": edge.get("prev_last_pdf_page", ""),
        "first_pdf_page": edge.get("first_pdf_page", ""),
        "page_gap": edge.get("page_gap", ""),
        "missing_pages": edge.get("missing_pages", ""),
        "same_family_and_key": "yes" if same_family else "no",
        "is_core_table_family": "yes" if is_core else "no",
        "max_similarity": f"{max_similarity(edge):.3f}",
        "strict_title_similarity": edge.get("strict_title_similarity", ""),
        "scope_similarity": edge.get("scope_similarity", ""),
        "prev_page_kind": edge.get("prev_page_kind", ""),
        "page_kind": edge.get("page_kind", ""),
        "prev_table_family_key": edge.get("prev_table_family_key", ""),
        "table_family_key": edge.get("table_family_key", ""),
        "prev_table_number_key": edge.get("prev_table_number_key", ""),
        "table_number_key": edge.get("table_number_key", ""),
        "prev_left_stub": "yes" if flags["prev_left_stub"] else "no",
        "prev_right_half": "yes" if flags["prev_right_half"] else "no",
        "cur_left_stub": "yes" if flags["cur_left_stub"] else "no",
        "cur_right_half": "yes" if flags["cur_right_half"] else "no",
        "prev_no_row_label_pages": edge.get("prev_no_row_label_pages", ""),
        "no_row_label_pages": edge.get("no_row_label_pages", ""),
        "prev_row_label_deficient_pages": edge.get("prev_row_label_deficient_pages", ""),
        "row_label_deficient_pages": edge.get("row_label_deficient_pages", ""),
        "v17_recommended_action": edge.get("v17_recommended_action", ""),
        "v17_recommended_layer": edge.get("v17_recommended_layer", ""),
        "v17_recommended_reason": edge.get("v17_recommended_reason", ""),
        "sequence_edge_action": edge.get("sequence_edge_action", ""),
        "sequence_join_type": edge.get("sequence_join_type", ""),
        "sequence_edge_reason": edge.get("sequence_edge_reason", ""),
        "sequence_review_flags": edge.get("sequence_review_flags", ""),
        "existing_edge_reason": edge.get("existing_edge_reason", ""),
        "existing_edge_review_flags": edge.get("existing_edge_review_flags", ""),
        "reason": reason,
        "prev_primary_title": edge.get("prev_primary_title", ""),
        "primary_title": edge.get("primary_title", ""),
        "prev_example_images": edge.get("prev_example_images", ""),
        "example_images": edge.get("example_images", ""),
    }


def build_rows(edges: list[dict[str, str]], boundary_decisions: dict[str, dict[str, str]]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    candidates: list[dict[str, object]] = []
    rejections: list[dict[str, object]] = []
    for edge in edges:
        (
            candidate_class,
            recommendation,
            review_priority,
            safe_whole_page_join,
            can_auto_join_now,
            candidate_layer,
            reason,
        ) = classify_edge(edge, boundary_decisions)
        row = output_row(
            edge,
            candidate_class,
            recommendation,
            review_priority,
            safe_whole_page_join,
            can_auto_join_now,
            candidate_layer,
            reason,
        )
        if candidate_class in CANDIDATE_CLASSES:
            candidates.append(row)
        else:
            rejection = dict(row)
            rejection["rejection_class"] = candidate_class
            rejections.append(rejection)
    candidates.sort(
        key=lambda row: (
            int(str(row.get("review_priority") or "9")),
            str(row.get("candidate_class") or ""),
            str(row.get("v5_source_key") or ""),
            v13.as_int(row.get("prev_last_pdf_page")),
            str(row.get("sequence_edge_id") or ""),
        )
    )
    rejections.sort(
        key=lambda row: (
            str(row.get("rejection_class") or ""),
            str(row.get("v5_source_key") or ""),
            v13.as_int(row.get("prev_last_pdf_page")),
            str(row.get("sequence_edge_id") or ""),
        )
    )
    return candidates, rejections


def source_summary(candidates: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: defaultdict[str, list[dict[str, object]]] = defaultdict(list)
    for row in candidates:
        grouped[str(row.get("v5_source_key") or "")].append(row)
    rows: list[dict[str, object]] = []
    for source, source_rows in sorted(grouped.items()):
        classes = Counter(str(row.get("candidate_class") or "") for row in source_rows)
        page_ranges = base.uniq_join(
            f"{row.get('prev_run_page_range')}->{row.get('run_page_range')}" for row in source_rows
        )
        rows.append(
            {
                "v5_source_key": source,
                "source_pdf": source_pdf_from_key(source),
                "candidate_count": len(source_rows),
                "review_candidate_count": sum(
                    1
                    for row in source_rows
                    if str(row.get("candidate_class") or "") != "possible_missing_bridge_extra_review"
                ),
                "possible_missing_bridge_extra_count": classes.get("possible_missing_bridge_extra_review", 0),
                "classes": json.dumps(dict(classes), sort_keys=True),
                "page_ranges": page_ranges,
            }
        )
    return rows


def validate(candidates: list[dict[str, object]], rejections: list[dict[str, object]], edge_count: int) -> list[dict[str, object]]:
    checks: list[tuple[str, list[object]]] = []
    seen = {str(row.get("sequence_edge_id") or "") for row in candidates}
    rejected = {str(row.get("sequence_edge_id") or "") for row in rejections}
    checks.append(("all_edges_classified_once", [] if len(seen | rejected) == edge_count and not (seen & rejected) else [{"edges": edge_count, "classified": len(seen | rejected), "overlap": len(seen & rejected)}]))
    unsafe_auto = [
        row.get("sequence_edge_id", "")
        for row in candidates
        if str(row.get("can_auto_join_now") or "") == "yes"
    ]
    checks.append(("no_new_auto_joins_in_audit_layer", unsafe_auto))
    candidates_crossing_gaps = [
        row.get("sequence_edge_id", "")
        for row in candidates
        if str(row.get("page_gap") or "") not in {"1", ""}
        and str(row.get("candidate_class") or "") != "possible_missing_bridge_extra_review"
    ]
    checks.append(("non_adjacent_candidates_are_missing_bridges_only", candidates_crossing_gaps))
    return [
        {
            "check": check,
            "status": "pass" if not bad else "fail",
            "failure_count": len(bad),
            "details": json.dumps(bad[:20], sort_keys=True),
        }
        for check, bad in checks
    ]


def write_readme(summary: dict[str, object]) -> None:
    lines = [
        "# v22 Candidate Assembly Audit",
        "",
        "No OCR/API calls are made. This layer audits remaining non-joined boundaries after v21.",
        "",
        "## Rules",
        "",
        "- Existing v17/v21 joins and v21 boundary decisions are treated as already handled.",
        "- Non-adjacent rows are never joins. They can only become `possible_missing_bridge_extra_review`.",
        "- Derivative selected-caste/race/tribe tables, explicit table switches, and obvious scope/status switches are rejected.",
        "- Adjacent core-table rows with left-stub, continuation-title, or high-similarity evidence become review candidates, not automatic joins.",
        "- `can_auto_join_now` is deliberately `no` for every v22 candidate.",
        "",
        "## Summary",
        "",
    ]
    for key in sorted(summary):
        lines.append(f"- `{key}`: {summary[key]}")
    (OUT_DIR / "v22_rules_readme.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    csv.field_size_limit(10**7)
    edges = read_csv(V17_EDGES)
    boundary_decision_rows = read_csv(V21_BOUNDARY_DECISIONS)
    boundary_decisions = {str(row.get("boundary_id") or ""): row for row in boundary_decision_rows}
    if not edges:
        raise SystemExit("Missing v17 edge recommendations")
    if not boundary_decisions:
        raise SystemExit("Missing v21 boundary decisions")

    candidates, rejections = build_rows(edges, boundary_decisions)
    candidate_by_source = source_summary(candidates)
    audit = validate(candidates, rejections, len(edges))
    audit_failures = sum(v13.as_int(row.get("failure_count")) for row in audit)
    candidate_counts = Counter(str(row.get("candidate_class") or "") for row in candidates)
    rejection_counts = Counter(str(row.get("rejection_class") or "") for row in rejections)
    summary = {
        "v22_candidate_edges": len(candidates),
        "v22_rejected_or_handled_edges": len(rejections),
        "v22_total_edges_classified": len(candidates) + len(rejections),
        "v22_input_edges": len(edges),
        "v22_candidate_class_counts": dict(candidate_counts),
        "v22_rejection_class_counts_top20": dict(rejection_counts.most_common(20)),
        "v22_sources_with_candidates": len(candidate_by_source),
        "v22_v21_boundary_decisions_carried": len(boundary_decisions),
        "v22_no_row_label_current_candidates": candidate_counts.get("review_row_label_absent_current", 0),
        "v22_auto_join_candidates": sum(1 for row in candidates if row.get("can_auto_join_now") == "yes"),
        "v22_audit_failure_count": audit_failures,
        "v22_audit_status": "pass" if audit_failures == 0 else "fail",
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base.write_csv(OUT_DIR / "v22_candidate_edges.csv", candidates, CANDIDATE_FIELDS)
    base.write_csv(
        OUT_DIR / "v22_immediate_review_candidates.csv",
        [
            row
            for row in candidates
            if str(row.get("review_priority") or "") in {"1", "2"}
            and row.get("candidate_class") != "possible_missing_bridge_extra_review"
        ],
        CANDIDATE_FIELDS,
    )
    base.write_csv(
        OUT_DIR / "v22_low_confidence_review_candidates.csv",
        [row for row in candidates if str(row.get("review_priority") or "") not in {"1", "2"}],
        CANDIDATE_FIELDS,
    )
    base.write_csv(
        OUT_DIR / "v22_possible_missing_bridge_extra.csv",
        [row for row in candidates if row.get("candidate_class") == "possible_missing_bridge_extra_review"],
        CANDIDATE_FIELDS,
    )
    base.write_csv(
        OUT_DIR / "v22_adjacent_core_review_candidates.csv",
        [row for row in candidates if row.get("candidate_class") != "possible_missing_bridge_extra_review"],
        CANDIDATE_FIELDS,
    )
    base.write_csv(OUT_DIR / "v22_rejected_or_handled_edges.csv", rejections, REJECTION_FIELDS)
    base.write_csv(OUT_DIR / "v22_candidate_summary_by_source.csv", candidate_by_source, SOURCE_SUMMARY_FIELDS)
    base.write_csv(OUT_DIR / "v22_audit.csv", audit, AUDIT_FIELDS)
    summary_rows = [
        {"metric": key, "value": json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value}
        for key, value in summary.items()
    ]
    base.write_csv(OUT_DIR / "v22_summary.csv", summary_rows, SUMMARY_FIELDS)
    with (OUT_DIR / "v22_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    write_readme(summary)
    workbook = OUT_DIR / "v22_candidate_assembly_audit.xlsx"
    base.write_workbook(
        workbook,
        {
            "summary": (summary_rows, SUMMARY_FIELDS),
            "candidate_edges": (candidates, CANDIDATE_FIELDS),
            "immediate_review": (
                [
                    row
                    for row in candidates
                    if str(row.get("review_priority") or "") in {"1", "2"}
                    and row.get("candidate_class") != "possible_missing_bridge_extra_review"
                ],
                CANDIDATE_FIELDS,
            ),
            "adjacent_core_review": (
                [row for row in candidates if row.get("candidate_class") != "possible_missing_bridge_extra_review"],
                CANDIDATE_FIELDS,
            ),
            "missing_bridge_extra": (
                [row for row in candidates if row.get("candidate_class") == "possible_missing_bridge_extra_review"],
                CANDIDATE_FIELDS,
            ),
            "source_summary": (candidate_by_source, SOURCE_SUMMARY_FIELDS),
            "audit": (audit, AUDIT_FIELDS),
        },
    )
    with zipfile.ZipFile(workbook) as z:
        summary["v22_workbook_zip_bad_member"] = z.testzip()
    with (OUT_DIR / "v22_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
