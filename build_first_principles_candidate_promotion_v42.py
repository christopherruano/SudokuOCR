#!/usr/bin/env python3
"""Promote only the safest physical candidate edges into bounded components.

This layer performs no OCR/API work. It reads the v39 edge evidence and the v41
static grouping manifest, then upgrades only candidate row-label dependencies
that pass strict semantic checks:

* adjacent PDF pages,
* candidate row-label dependency status,
* no derivative/off-target/table-switch flags,
* zero checks pass,
* row labels align strongly, and
* the column sequence moves forward rather than backward or duplicating.

Everything else stays review-only with an explicit reason.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import zipfile
from collections import Counter
from pathlib import Path
from typing import Iterable

import xlsxwriter


ROOT = Path(__file__).resolve().parent
DEFAULT_GROUPING_DIR = ROOT / "export_final/pdf_recovery_incoming_2026-06-16_first_principles_static_grouping_manifest_smoke_v41_manifest_colcheck"
DEFAULT_PHYSICAL_EVIDENCE_DIR = ROOT / "export_final/pdf_recovery_incoming_2026-06-16_first_principles_assembly_evidence_smoke_v41_manifest_colcheck"
DEFAULT_OUT_DIR = ROOT / "export_final/pdf_recovery_incoming_2026-06-16_first_principles_candidate_promotion_v42"

GROUP_FIELDS = [
    "group_rank",
    "group_id",
    "group_layer",
    "group_kind",
    "source_record_id",
    "site_use_policy",
    "physical_merge_allowed",
    "logical_collection_only",
    "safe_to_publish_values",
    "review_required",
    "priority",
    "v5_source_key",
    "source_pdf_id",
    "page_range",
    "component_pages",
    "n_pages",
    "edge_pairs",
    "member_count",
    "pages_are_consecutive",
    "table_number_key",
    "topic_variant_key",
    "religion_terms_detected",
    "evidence_bucket",
    "evidence_priority",
    "row_label_dependency_signal",
    "missing_intermediate_pages",
    "value_status",
    "title_sample",
    "recommended_action",
    "evidence_files",
]

MEMBER_FIELDS = [
    "member_rank",
    "group_id",
    "group_kind",
    "member_position",
    "v5_source_key",
    "source_pdf_id",
    "pdf_page",
    "expanded_pages",
    "source_artifact",
    "table_number_key",
    "topic_variant_key",
    "religion_terms_detected",
    "row_label",
    "section",
    "column_group_order",
    "column_group",
    "persons",
    "males",
    "females",
    "image_path",
    "title",
]

DECISION_FIELDS = [
    "decision_rank",
    "edge_key",
    "decision",
    "promoted_group_id",
    "reason",
    "semantic_column_status",
    "semantic_column_detail",
    "v5_source_key",
    "source_pdf_id_guess",
    "page_pair",
    "physical_rule_status",
    "evidence_bucket",
    "zero_check_status",
    "row_alignment_evidence",
    "column_order_evidence",
    "rule_flags",
    "left_title",
    "right_title",
    "left_column_groups",
    "right_column_groups",
    "combined_column_group_order",
    "left_best_json",
    "right_best_json",
    "left_best_csv",
    "right_best_csv",
    "prev_image_path",
    "image_path",
]

AUDIT_FIELDS = ["check", "status", "detail"]
SUMMARY_FIELDS = ["metric", "name", "count"]

FORBIDDEN_FLAGS = {
    "derivative_class_switch_break",
    "offtarget_table_break",
    "page_kind_switch",
    "table_number_switch_or_missing",
}

STRONG_ROWS = {
    "same_row_labels_exact",
    "same_row_labels_substantial_overlap",
    "right_unlabelled_same_row_count_inherit_left_labels",
}

FAILED_PARTIAL_ALLOWED_FLAGS = {
    "failed_or_partial_constraints",
    "pairwise_only",
}


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
        return int(str(value).replace(",", ""))
    except ValueError:
        return None


def semi(values: Iterable[object]) -> str:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    return "; ".join(out)


def split_semicolon(value: object) -> list[str]:
    return [part.strip() for part in str(value or "").split(";") if part.strip()]


def page_pair_pages(value: object) -> list[int]:
    match = re.fullmatch(r"\s*(\d+)\s*-\s*(\d+)\s*", str(value or ""))
    if not match:
        return []
    return [int(match.group(1)), int(match.group(2))]


def normalize_text(value: object) -> str:
    text = str(value or "").lower()
    text = text.replace("\u2013", "-").replace("\u2014", "-").replace("&", " and ")
    text = re.sub(r"\s+", " ", text)
    return text.strip(" .;:-")


def civil_rank(group: str) -> int | None:
    text = normalize_text(group)
    if "unmarried" in text:
        return 20
    if "widow" in text:
        return 40
    if "married" in text:
        return 30
    if "population" in text or "persons" in text or text in {"total", "total."}:
        return 10
    return None


def age_rank(group: str) -> int | None:
    text = normalize_text(group)
    if "unspecified" in text:
        return 999
    if "60" in text and ("over" in text or "+" in text):
        return 60
    if text.startswith("total") or text in {"population", "persons"}:
        return -1
    match = re.search(r"(?<!\d)(\d{1,3})(?:\s*[-. ]|\s+and\s+over|\s*&\s*over|$)", text)
    if not match:
        return None
    return int(match.group(1))


def semantic_column_status(left_groups: list[str], right_groups: list[str]) -> tuple[str, str]:
    if not left_groups or not right_groups:
        return "fail_missing_column_groups", "missing left or right column groups"

    left_civil = [civil_rank(group) for group in left_groups]
    right_civil = [civil_rank(group) for group in right_groups]
    left_civil_vals = [rank for rank in left_civil if rank is not None]
    right_civil_vals = [rank for rank in right_civil if rank is not None]

    left_age = [age_rank(group) for group in left_groups]
    right_age = [age_rank(group) for group in right_groups]
    left_age_vals = [rank for rank in left_age if rank is not None]
    right_age_vals = [rank for rank in right_age if rank is not None]
    left_age_positive = [rank for rank in left_age_vals if rank >= 0]
    right_age_positive = [rank for rank in right_age_vals if rank >= 0]

    if left_age_positive and right_age_positive:
        if max(left_age_positive) < min(right_age_positive):
            return "pass_age_band_sequence_forward", f"left_age_max={max(left_age_positive)} right_age_min={min(right_age_positive)}"
        if min(left_age_positive) >= max(right_age_positive):
            return "fail_age_band_sequence_reversed", f"left_age_min={min(left_age_positive)} right_age_max={max(right_age_positive)}"
        return "fail_age_band_sequence_overlaps", f"left={semi(left_age_positive)} right={semi(right_age_positive)}"

    if left_civil_vals and right_civil_vals:
        left_set = set(left_civil_vals)
        right_set = set(right_civil_vals)
        if left_set == right_set:
            return "fail_duplicate_civil_column_set", f"civil_set={semi(sorted(left_set))}"
        if max(left_civil_vals) <= min(right_civil_vals):
            return "pass_civil_status_sequence_forward", f"left_max={max(left_civil_vals)} right_min={min(right_civil_vals)}"
        if min(left_civil_vals) >= max(right_civil_vals):
            return "fail_civil_status_sequence_reversed", f"left_min={min(left_civil_vals)} right_max={max(right_civil_vals)}"
        return "fail_civil_status_sequence_overlaps", f"left={semi(left_civil_vals)} right={semi(right_civil_vals)}"

    return "fail_no_supported_semantic_sequence", "no age-band or civil-status sequence could be verified"


def strict_failed_partial_candidate(edge: dict[str, str]) -> bool:
    """Allow a tiny subset of seed-layer partials that are structurally exact.

    These are not broad same-title merges. They are adjacent two-page edges where
    the OCR evidence proves row equality, forward disjoint columns, and passing
    zero checks, but the seed layer held the edge because not every visual block
    was independently promoted.
    """
    flags = {flag.strip() for flag in str(edge.get("rule_flags", "")).split(";") if flag.strip()}
    left_rows = int_or_none(edge.get("left_row_count"))
    right_rows = int_or_none(edge.get("right_row_count"))
    left_labels = int_or_none(edge.get("left_row_label_count"))
    right_labels = int_or_none(edge.get("right_row_label_count"))
    zero_failures = int_or_none(edge.get("zero_check_failures"))
    return (
        edge.get("evidence_bucket") == "candidate_or_failed_edge_review"
        and edge.get("physical_rule_status") == "failed_or_partial_physical_review"
        and edge.get("pages_consecutive") == "yes"
        and edge.get("zero_check_status") == "zero_checks_pass"
        and (zero_failures in {None, 0})
        and edge.get("row_alignment_evidence") == "same_row_labels_exact"
        and edge.get("column_order_evidence") == "left_page_groups_then_right_page_groups_disjoint"
        and left_rows is not None
        and left_rows == right_rows
        and left_rows >= 10
        and (left_labels or 0) > 0
        and (right_labels or 0) > 0
        and not (flags & FORBIDDEN_FLAGS)
        and flags <= FAILED_PARTIAL_ALLOWED_FLAGS
    )


def strict_nozero_right_unlabelled_candidate(edge: dict[str, str]) -> bool:
    """Promote structure-only joins where the right page is an unlabeled continuation.

    Some city/district tables continue horizontally by age band and omit row
    labels on the right page. These can be structurally safe even when no zero
    check exists, as long as row counts match exactly, labels inherit from the
    left page, columns move forward semantically, and no hard switch flag exists.
    Values remain review-only.
    """
    flags = {flag.strip() for flag in str(edge.get("rule_flags", "")).split(";") if flag.strip()}
    left_rows = int_or_none(edge.get("left_row_count"))
    right_rows = int_or_none(edge.get("right_row_count"))
    left_labels = int_or_none(edge.get("left_row_label_count"))
    right_labels = int_or_none(edge.get("right_row_label_count"))
    zero_failures = int_or_none(edge.get("zero_check_failures"))
    semantic_status, _ = semantic_column_status(
        split_semicolon(edge.get("left_column_groups", "")),
        split_semicolon(edge.get("right_column_groups", "")),
    )
    return (
        edge.get("physical_rule_status") == "candidate_row_label_dependency"
        and edge.get("pages_consecutive") == "yes"
        and edge.get("zero_check_status") == "no_zero_check_available"
        and (zero_failures in {None, 0})
        and edge.get("row_alignment_evidence") == "right_unlabelled_same_row_count_inherit_left_labels"
        and edge.get("column_order_evidence") == "left_page_groups_then_right_page_groups_disjoint"
        and semantic_status.startswith("pass_")
        and left_rows is not None
        and left_rows == right_rows
        and left_rows >= 10
        and left_labels == left_rows
        and right_labels == 0
        and not (flags & FORBIDDEN_FLAGS)
        and flags <= {"pairwise_only", "row_label_dependency_candidate"}
    )


def decision_for_edge(edge: dict[str, str]) -> dict[str, object]:
    left_groups = split_semicolon(edge.get("left_column_groups", ""))
    right_groups = split_semicolon(edge.get("right_column_groups", ""))
    semantic_status, semantic_detail = semantic_column_status(left_groups, right_groups)
    flags = {flag.strip() for flag in str(edge.get("rule_flags", "")).split(";") if flag.strip()}
    reasons: list[str] = []
    strict_partial = strict_failed_partial_candidate(edge)
    strict_nozero_unlabelled = strict_nozero_right_unlabelled_candidate(edge)

    if edge.get("pages_consecutive") != "yes":
        reasons.append("pages_not_consecutive")
    if edge.get("physical_rule_status") != "candidate_row_label_dependency" and not strict_partial:
        reasons.append(f"status_not_promotable:{edge.get('physical_rule_status', '')}")
    if edge.get("zero_check_status") != "zero_checks_pass" and not strict_nozero_unlabelled:
        reasons.append(f"zero_check_not_pass:{edge.get('zero_check_status', '')}")
    if edge.get("row_alignment_evidence") not in STRONG_ROWS:
        reasons.append(f"row_alignment_not_strong:{edge.get('row_alignment_evidence', '')}")
    if edge.get("column_order_evidence") != "left_page_groups_then_right_page_groups_disjoint":
        reasons.append(f"column_order_not_disjoint_forward_candidate:{edge.get('column_order_evidence', '')}")
    bad_flags = sorted(flags & FORBIDDEN_FLAGS)
    if bad_flags:
        reasons.append(f"forbidden_flags:{semi(bad_flags)}")
    if not semantic_status.startswith("pass_"):
        reasons.append(semantic_status)

    promoted = not reasons
    group_id = stable_id("v42phys", edge.get("edge_key", ""), edge.get("page_pair", "")) if promoted else ""
    return {
        **edge,
        "decision": "promote_physical_structure_review_values" if promoted else "keep_review_only",
        "promoted_group_id": group_id,
        "reason": (
            "strict_nozero_right_unlabelled_rules_passed"
            if strict_nozero_unlabelled
            else "strict_failed_partial_rules_passed"
            if strict_partial
            else "strict_rules_passed"
        ) if promoted else semi(reasons),
        "semantic_column_status": semantic_status,
        "semantic_column_detail": semantic_detail,
    }


def promoted_group(decision: dict[str, object]) -> dict[str, object]:
    edge = decision
    pages = page_pair_pages(edge.get("page_pair", ""))
    return {
        "group_id": edge.get("promoted_group_id", ""),
        "group_layer": "physical",
        "group_kind": "site_ready_physical_component_promoted_by_v42",
        "source_record_id": edge.get("edge_key", ""),
        "site_use_policy": "use_now_physical_component_structure",
        "physical_merge_allowed": "yes",
        "logical_collection_only": "no",
        "safe_to_publish_values": "no",
        "review_required": "yes",
        "priority": "2",
        "v5_source_key": edge.get("v5_source_key", ""),
        "source_pdf_id": edge.get("source_pdf_id_guess", ""),
        "page_range": edge.get("page_pair", ""),
        "component_pages": semi(pages),
        "n_pages": len(set(pages)),
        "edge_pairs": edge.get("page_pair", ""),
        "member_count": len(set(pages)),
        "pages_are_consecutive": "yes",
        "table_number_key": "",
        "topic_variant_key": "",
        "religion_terms_detected": "",
        "evidence_bucket": "promoted_candidate_strict_v42",
        "evidence_priority": "2",
        "row_label_dependency_signal": "yes",
        "missing_intermediate_pages": "",
        "value_status": "structure_only_values_require_review",
        "title_sample": semi([edge.get("left_title", ""), edge.get("right_title", "")]),
        "recommended_action": "Use as physical structure only; values still require review.",
        "evidence_files": "candidate_promotion_decisions.csv; edge_assembly_evidence.csv",
    }


def promoted_members(decision: dict[str, object]) -> list[dict[str, object]]:
    pages = page_pair_pages(decision.get("page_pair", ""))
    images = [decision.get("prev_image_path", ""), decision.get("image_path", "")]
    titles = [decision.get("left_title", ""), decision.get("right_title", "")]
    artifacts = [
        decision.get("left_best_json", "") or decision.get("left_best_csv", ""),
        decision.get("right_best_json", "") or decision.get("right_best_csv", ""),
    ]
    rows: list[dict[str, object]] = []
    for idx, page in enumerate(pages, start=1):
        rows.append(
            {
                "group_id": decision.get("promoted_group_id", ""),
                "group_kind": "site_ready_physical_component_promoted_by_v42",
                "member_position": idx,
                "v5_source_key": decision.get("v5_source_key", ""),
                "source_pdf_id": decision.get("source_pdf_id_guess", ""),
                "pdf_page": page,
                "expanded_pages": page,
                "source_artifact": artifacts[idx - 1] if idx - 1 < len(artifacts) else "",
                "image_path": images[idx - 1] if idx - 1 < len(images) else "",
                "title": titles[idx - 1] if idx - 1 < len(titles) else "",
            }
        )
    return rows


def rerank(rows: list[dict[str, object]], field: str) -> None:
    for idx, row in enumerate(rows, start=1):
        row[field] = idx


def build_outputs(grouping_dir: Path, physical_evidence_dir: Path) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    base_groups = read_csv(grouping_dir / "canonical_grouping_manifest.csv")
    base_members = read_csv(grouping_dir / "canonical_group_members.csv")
    edge_rows = read_csv(physical_evidence_dir / "edge_assembly_evidence.csv")
    strong_edges = [
        row for row in edge_rows
        if row.get("evidence_bucket") == "candidate_has_strong_evidence_review_for_promotion"
        or strict_failed_partial_candidate(row)
        or strict_nozero_right_unlabelled_candidate(row)
    ]
    decisions = [decision_for_edge(row) for row in strong_edges]
    promoted_decisions = [row for row in decisions if row.get("decision") == "promote_physical_structure_review_values"]
    promoted_edge_keys = {str(row.get("edge_key", "")) for row in promoted_decisions}
    promoted_old_group_ids = {row.get("group_id", "") for row in base_groups if row.get("source_record_id") in promoted_edge_keys}

    groups = [dict(row) for row in base_groups if row.get("source_record_id") not in promoted_edge_keys]
    groups.extend(promoted_group(row) for row in promoted_decisions)
    groups.sort(key=lambda row: (int_or_none(row.get("priority")) or 999, str(row.get("group_layer", "")), str(row.get("v5_source_key", "")), str(row.get("page_range", "")), str(row.get("group_id", ""))))
    rerank(groups, "group_rank")

    members = [dict(row) for row in base_members if row.get("group_id") not in promoted_old_group_ids]
    for decision in promoted_decisions:
        members.extend(promoted_members(decision))
    members.sort(key=lambda row: (str(row.get("group_id", "")), int_or_none(row.get("member_position")) or 999, int_or_none(row.get("pdf_page")) or 999))
    rerank(members, "member_rank")

    blocked = [row for row in decisions if row.get("decision") != "promote_physical_structure_review_values"]
    return groups, members, decisions, promoted_decisions, blocked


def build_audits(
    groups: list[dict[str, object]],
    members: list[dict[str, object]],
    decisions: list[dict[str, object]],
    promoted: list[dict[str, object]],
    workbook_path: Path,
) -> list[dict[str, object]]:
    audits: list[dict[str, object]] = []
    promoted_groups = [row for row in groups if row.get("group_kind") == "site_ready_physical_component_promoted_by_v42"]
    bad_promoted = [
        row for row in promoted_groups
        if row.get("physical_merge_allowed") != "yes" or row.get("pages_are_consecutive") != "yes" or int_or_none(row.get("n_pages")) != 2
    ]
    audits.append({"check": "v42_promotions_are_two_page_consecutive_physical_components", "status": "pass" if not bad_promoted else "fail", "detail": f"bad={len(bad_promoted)} total={len(promoted_groups)}"})

    promoted_fail_reasons = [
        row for row in promoted
        if row.get("reason") not in {
            "strict_rules_passed",
            "strict_failed_partial_rules_passed",
            "strict_nozero_right_unlabelled_rules_passed",
        }
    ]
    audits.append({"check": "v42_promotions_pass_strict_rules", "status": "pass" if not promoted_fail_reasons else "fail", "detail": f"bad={len(promoted_fail_reasons)} total={len(promoted)}"})

    duplicate_site_ready_edges = [
        row for row in groups
        if row.get("group_kind") == "site_ready_physical_component_promoted_by_v42"
        and sum(
            1 for other in groups
            if other.get("v5_source_key") == row.get("v5_source_key")
            and other.get("edge_pairs") == row.get("edge_pairs")
            and str(other.get("site_use_policy", "")).startswith("use_now")
        ) > 1
    ]
    audits.append({"check": "v42_promotions_do_not_duplicate_existing_site_ready_edges", "status": "pass" if not duplicate_site_ready_edges else "fail", "detail": f"bad={len(duplicate_site_ready_edges)}"})

    member_ids = {str(row.get("group_id", "")) for row in members}
    missing_members = [row for row in promoted_groups if row.get("group_id") not in member_ids]
    audits.append({"check": "v42_promoted_groups_have_member_rows", "status": "pass" if not missing_members else "fail", "detail": f"missing={len(missing_members)} total={len(promoted_groups)}"})

    reviewed_all_candidates = len(decisions) == len(promoted) + sum(1 for row in decisions if row.get("decision") != "promote_physical_structure_review_values")
    audits.append({"check": "all_v39_promotion_candidates_get_v42_decision", "status": "pass" if reviewed_all_candidates else "fail", "detail": f"decisions={len(decisions)} promoted={len(promoted)}"})

    strict_partial_promotions = [row for row in promoted if row.get("reason") == "strict_failed_partial_rules_passed"]
    bad_strict_partial = [
        row for row in strict_partial_promotions
        if row.get("row_alignment_evidence") != "same_row_labels_exact"
        or row.get("column_order_evidence") != "left_page_groups_then_right_page_groups_disjoint"
        or row.get("zero_check_status") != "zero_checks_pass"
        or row.get("pages_consecutive") != "yes"
    ]
    audits.append({"check": "v42_strict_partial_promotions_are_exact_pairwise_edges", "status": "pass" if not bad_strict_partial else "fail", "detail": f"bad={len(bad_strict_partial)} total={len(strict_partial_promotions)}"})

    strict_nozero_unlabelled_promotions = [row for row in promoted if row.get("reason") == "strict_nozero_right_unlabelled_rules_passed"]
    bad_strict_nozero_unlabelled = [
        row for row in strict_nozero_unlabelled_promotions
        if row.get("row_alignment_evidence") != "right_unlabelled_same_row_count_inherit_left_labels"
        or row.get("column_order_evidence") != "left_page_groups_then_right_page_groups_disjoint"
        or row.get("zero_check_status") != "no_zero_check_available"
        or row.get("pages_consecutive") != "yes"
    ]
    audits.append({"check": "v42_nozero_right_unlabelled_promotions_are_structure_only_pairwise_edges", "status": "pass" if not bad_strict_nozero_unlabelled else "fail", "detail": f"bad={len(bad_strict_nozero_unlabelled)} total={len(strict_nozero_unlabelled_promotions)}"})

    try:
        with zipfile.ZipFile(workbook_path) as zf:
            bad_member = zf.testzip()
    except Exception as exc:
        bad_member = f"{type(exc).__name__}: {exc}"
    audits.append({"check": "workbook_zip_integrity", "status": "pass" if bad_member is None else "fail", "detail": str(bad_member or "ok")})
    return audits


def build_summary(
    groups: list[dict[str, object]],
    members: list[dict[str, object]],
    decisions: list[dict[str, object]],
    promoted: list[dict[str, object]],
    blocked: list[dict[str, object]],
    audits: list[dict[str, object]],
    paths: dict[str, Path],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = [{"metric": "paths", "name": name, "count": output_path(path)} for name, path in paths.items()]
    rows.extend(
        [
            {"metric": "counts", "name": "manifest_groups", "count": len(groups)},
            {"metric": "counts", "name": "manifest_members", "count": len(members)},
            {"metric": "counts", "name": "strong_candidate_decisions", "count": len(decisions)},
            {"metric": "counts", "name": "v42_promoted_candidates", "count": len(promoted)},
            {"metric": "counts", "name": "v42_blocked_strong_candidates", "count": len(blocked)},
            {"metric": "counts", "name": "site_ready_groups", "count": sum(1 for row in groups if str(row.get("site_use_policy", "")).startswith("use_now"))},
            {"metric": "counts", "name": "physical_merge_groups", "count": sum(1 for row in groups if row.get("physical_merge_allowed") == "yes")},
            {"metric": "counts", "name": "audit_failures", "count": sum(1 for row in audits if row.get("status") != "pass")},
        ]
    )
    for metric, counter in [
        ("groups_by_kind", Counter(str(row.get("group_kind", "")) for row in groups)),
        ("groups_by_policy", Counter(str(row.get("site_use_policy", "")) for row in groups)),
        ("decisions_by_status", Counter(str(row.get("decision", "")) for row in decisions)),
        ("blocked_by_reason", Counter(str(row.get("reason", "")) for row in blocked)),
        ("semantic_status", Counter(str(row.get("semantic_column_status", "")) for row in decisions)),
        ("audits_by_status", Counter(str(row.get("status", "")) for row in audits)),
    ]:
        for name, count in sorted(counter.items()):
            rows.append({"metric": metric, "name": name, "count": count})
    return rows


def write_workbook(
    path: Path,
    summary: list[dict[str, object]],
    audits: list[dict[str, object]],
    groups: list[dict[str, object]],
    members: list[dict[str, object]],
    decisions: list[dict[str, object]],
    promoted: list[dict[str, object]],
    blocked: list[dict[str, object]],
) -> None:
    workbook = xlsxwriter.Workbook(path, {"constant_memory": True})
    header_fmt = workbook.add_format({"bold": True, "bg_color": "#E8EEF7", "border": 1})
    wrap_fmt = workbook.add_format({"text_wrap": True, "valign": "top"})
    sheets = [
        ("Summary", summary, SUMMARY_FIELDS),
        ("Audit", audits, AUDIT_FIELDS),
        ("Grouping Manifest v42", groups, GROUP_FIELDS),
        ("Group Members v42", members, MEMBER_FIELDS),
        ("Promotion Decisions", decisions, DECISION_FIELDS),
        ("Promoted", promoted, DECISION_FIELDS),
        ("Blocked Strong", blocked, DECISION_FIELDS),
    ]
    for name, rows, fields in sheets:
        ws = workbook.add_worksheet(name[:31])
        for col, field in enumerate(fields):
            ws.write(0, col, field, header_fmt)
        for row_idx, row in enumerate(rows, start=1):
            for col, field in enumerate(fields):
                value = row.get(field, "")
                fmt = wrap_fmt if field in {"reason", "semantic_column_detail", "recommended_action", "title_sample", "left_column_groups", "right_column_groups", "combined_column_group_order", "left_title", "right_title"} else None
                ws.write(row_idx, col, value, fmt)
        ws.freeze_panes(1, 0)
        ws.autofilter(0, 0, max(0, len(rows)), max(0, len(fields) - 1))
        for col, field in enumerate(fields):
            ws.set_column(col, col, min(max(len(field) + 2, 12), 60))
    workbook.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grouping-dir", default=str(DEFAULT_GROUPING_DIR))
    parser.add_argument("--physical-evidence-dir", default=str(DEFAULT_PHYSICAL_EVIDENCE_DIR))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    args = parser.parse_args()

    grouping_dir = Path(args.grouping_dir).expanduser()
    physical_evidence_dir = Path(args.physical_evidence_dir).expanduser()
    out_dir = Path(args.out_dir).expanduser()
    paths = {"grouping_dir": grouping_dir, "physical_evidence_dir": physical_evidence_dir, "out_dir": out_dir}

    groups, members, decisions, promoted, blocked = build_outputs(grouping_dir, physical_evidence_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    workbook_path = out_dir / "candidate_promotion_workbook.xlsx"
    write_workbook(workbook_path, [], [], groups, members, decisions, promoted, blocked)
    audits = build_audits(groups, members, decisions, promoted, workbook_path)
    summary = build_summary(groups, members, decisions, promoted, blocked, audits, paths)
    write_workbook(workbook_path, summary, audits, groups, members, decisions, promoted, blocked)
    audits = build_audits(groups, members, decisions, promoted, workbook_path)
    summary = build_summary(groups, members, decisions, promoted, blocked, audits, paths)
    write_workbook(workbook_path, summary, audits, groups, members, decisions, promoted, blocked)

    write_csv(out_dir / "canonical_grouping_manifest_v42.csv", groups, GROUP_FIELDS)
    write_csv(out_dir / "canonical_group_members_v42.csv", members, MEMBER_FIELDS)
    write_csv(out_dir / "candidate_promotion_decisions.csv", decisions, DECISION_FIELDS)
    write_csv(out_dir / "promoted_physical_components_v42.csv", promoted, DECISION_FIELDS)
    write_csv(out_dir / "blocked_strong_candidates_v42.csv", blocked, DECISION_FIELDS)
    write_csv(out_dir / "acceptance_audit.csv", audits, AUDIT_FIELDS)
    write_csv(out_dir / "summary.csv", summary, SUMMARY_FIELDS)

    summary_json = {
        "manifest_groups": len(groups),
        "manifest_members": len(members),
        "strong_candidate_decisions": len(decisions),
        "v42_promoted_candidates": len(promoted),
        "v42_blocked_strong_candidates": len(blocked),
        "site_ready_groups": sum(1 for row in groups if str(row.get("site_use_policy", "")).startswith("use_now")),
        "physical_merge_groups": sum(1 for row in groups if row.get("physical_merge_allowed") == "yes"),
        "audit_failure_count": sum(1 for row in audits if row.get("status") != "pass"),
        "groups_by_kind": dict(Counter(str(row.get("group_kind", "")) for row in groups)),
        "decisions_by_status": dict(Counter(str(row.get("decision", "")) for row in decisions)),
        "semantic_status": dict(Counter(str(row.get("semantic_column_status", "")) for row in decisions)),
        "outputs": {
            "workbook": output_path(workbook_path),
            "canonical_grouping_manifest_v42": output_path(out_dir / "canonical_grouping_manifest_v42.csv"),
            "canonical_group_members_v42": output_path(out_dir / "canonical_group_members_v42.csv"),
            "candidate_promotion_decisions": output_path(out_dir / "candidate_promotion_decisions.csv"),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary_json, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary_json, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
