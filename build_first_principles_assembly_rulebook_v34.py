#!/usr/bin/env python3
"""Build a first-principles assembly rulebook from current evidence.

This layer is deliberately conservative. It does not OCR, does not call APIs,
and does not promote broad same-title groups. It turns the existing adjacency
and component evidence into a durable contract:

- every observed PDF-page edge has a rule-level decision;
- physical grouping is allowed only for confirmed consecutive components;
- row-label dependency and missing-page cases are queued, not merged;
- logical families remain logical-only and never imply physical merge.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import xlsxwriter


ROOT = Path(__file__).resolve().parent
DEFAULT_ASSEMBLY_DIR = ROOT / "export_final/pdf_recovery_incoming_2026-06-16_first_principles_assembly_pipeline_v29_logical_review_context"
DEFAULT_FRONTIER_DIR = ROOT / "export_final/pdf_recovery_incoming_2026-06-16_first_principles_assembly_frontier_v30_joined"
DEFAULT_COMPONENT_DIR = ROOT / "export_final/pdf_recovery_incoming_2026-06-16_first_principles_assembly_components_v31"
DEFAULT_LOGICAL_DIR = ROOT / "export_final/pdf_recovery_incoming_2026-06-16_first_principles_logical_family_frontier_v32"
DEFAULT_CONTROL_DIR = ROOT / "export_final/pdf_recovery_incoming_2026-06-16_first_principles_assembly_control_v33"
DEFAULT_CLASSIFIER_CSV = ROOT / "export_final/pdf_recovery_incoming_2026-06-16_no_gaps_ledger_v2_refined/gemini_ocr_worklists/gemini_classify_priority_all.csv"
DEFAULT_OUT_DIR = ROOT / "export_final/pdf_recovery_incoming_2026-06-16_first_principles_assembly_rulebook_v34"


EDGE_FIELDS = [
    "rule_rank",
    "rule_id",
    "edge_key",
    "v5_source_key",
    "source_pdf_id_guess",
    "prev_pdf_page",
    "pdf_page",
    "page_pair",
    "page_gap",
    "pages_consecutive",
    "physical_rule_status",
    "physical_merge_allowed_now",
    "logical_group_allowed_now",
    "site_action",
    "promotion_condition",
    "likely_after_missing_pages",
    "likely_after_ocr",
    "requires_manual_review",
    "requires_value_review",
    "row_label_dependency_signal",
    "right_page_parity",
    "right_page_no_row_labels",
    "left_row_label_ratio",
    "right_row_label_ratio",
    "row_label_scan_strength",
    "row_label_scan_pattern",
    "prev_page_kind",
    "page_kind",
    "page_kind_transition",
    "left_page_class",
    "right_page_class",
    "page_class_transition",
    "classifier_break_signal",
    "same_table_family",
    "same_table_number",
    "prev_table_family_key",
    "table_family_key",
    "prev_table_number_key",
    "table_number_key",
    "title_switch_signal",
    "prev_title",
    "title",
    "decision_reason",
    "rule_flags",
    "source_decision_layer",
    "frontier_statuses",
    "frontier_action_buckets",
    "component_ids",
    "prev_image_path",
    "image_path",
]

GROUP_FIELDS = [
    "group_rank",
    "group_id",
    "group_type",
    "group_status",
    "physical_merge_allowed_now",
    "logical_collection_only",
    "safe_to_publish_values",
    "review_required",
    "v5_source_key",
    "source_pdf_id_guess",
    "page_range",
    "component_pages",
    "n_pages",
    "edge_pairs",
    "member_count",
    "primary_title",
    "religion_terms_detected",
    "reason",
    "rule_flags",
    "recommended_action",
    "evidence_file",
]

PAGE_FIELDS = [
    "page_rank",
    "page_id",
    "v5_source_key",
    "source_pdf_id_guess",
    "pdf_page",
    "page_parity",
    "page_kind",
    "page_class",
    "contains_numeric_grid",
    "table_family_key",
    "table_number_key",
    "row_label_ratio",
    "no_row_labels_signal",
    "incoming_rule_statuses",
    "outgoing_rule_statuses",
    "incoming_edge_count",
    "outgoing_edge_count",
    "title",
    "image_path",
]

QUEUE_FIELDS = [
    "queue_rank",
    "queue_type",
    "priority",
    "v5_source_key",
    "source_pdf_id_guess",
    "page_pair",
    "missing_pages",
    "physical_rule_status",
    "site_action",
    "promotion_condition",
    "row_label_dependency_signal",
    "left_page_class",
    "right_page_class",
    "page_class_transition",
    "rule_flags",
    "decision_reason",
    "prev_image_path",
    "image_path",
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
    payload = "||".join(str(part) for part in parts)
    return f"{prefix}_{hashlib.sha1(payload.encode('utf-8')).hexdigest()[:16]}"


def int_or_none(value: object) -> int | None:
    if value in {None, ""}:
        return None
    try:
        return int(str(value))
    except ValueError:
        return None


def float_or_none(value: object) -> float | None:
    if value in {None, ""}:
        return None
    try:
        return float(str(value))
    except ValueError:
        return None


def semi(values: Iterable[object]) -> str:
    return "; ".join(str(value) for value in values if value not in {None, ""})


def split_semicolon(value: object) -> list[str]:
    return [part.strip() for part in str(value or "").split(";") if part.strip()]


def source_id_from_text(*values: object) -> str:
    for value in values:
        text = str(value or "")
        for pattern in [
            r"saoa\.crl\.(\d{7,8})",
            r"/(\d{7,8})/[^/]*?(\d{7,8})_p\d+",
            r"(\d{7,8})_p\d+",
        ]:
            match = re.search(pattern, text)
            if match:
                return match.group(1)
    return ""


def page_pair(left: object, right: object) -> str:
    if left in {None, ""} or right in {None, ""}:
        return ""
    return f"{left}-{right}"


def missing_pages(left: object, right: object) -> list[int]:
    left_i = int_or_none(left)
    right_i = int_or_none(right)
    if left_i is None or right_i is None or right_i <= left_i + 1:
        return []
    return list(range(left_i + 1, right_i))


def ratio_no_labels(value: object) -> bool:
    ratio = float_or_none(value)
    return ratio is None or ratio <= 0.05


def ratio_has_labels(value: object) -> bool:
    ratio = float_or_none(value)
    return ratio is not None and ratio >= 0.50


def same_nonblank(left: object, right: object) -> str:
    left_s = str(left or "").strip()
    right_s = str(right or "").strip()
    return "yes" if left_s and right_s and left_s == right_s else "no"


def title_switch(prev_title: str, title: str, same_family: str, same_number: str) -> str:
    prev = (prev_title or "").strip().lower()
    cur = (title or "").strip().lower()
    if not prev or not cur:
        return "unknown"
    if prev == cur:
        return "same_title"
    if same_family == "yes" and same_number == "yes":
        return "same_table_different_title_or_subtitle"
    if same_family == "yes":
        return "same_family_title_change"
    return "new_title_or_table_switch"


def classifier_index(rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, str]]:
    by_page: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        source_id = str(row.get("source_id") or source_id_from_text(row.get("image_path"), row.get("image_stem"))).strip()
        pdf_page = str(row.get("pdf_page") or "").strip()
        if source_id and pdf_page:
            by_page[(source_id, pdf_page)] = row
    return by_page


def page_class_transition(left_class: str, right_class: str) -> tuple[str, str]:
    left = left_class or ""
    right = right_class or ""
    if not left and not right:
        return "no_classifier_evidence", "no"
    if "blank_or_empty" in {left, right}:
        return "blank_page_break", "yes"
    if "offtarget_table" in {left, right}:
        return "offtarget_table_break", "yes"
    if left == "tablevii_context_or_title" or right == "tablevii_context_or_title":
        return "context_title_boundary_no_physical_merge", "yes"
    if left.startswith("derivative") or right.startswith("derivative"):
        if left == right:
            return "derivative_same_class_review_only", "yes"
        return "derivative_class_switch_break", "yes"
    if left == "main_tablevii_numeric" and right == "main_tablevii_numeric":
        return "main_tablevii_numeric_compatible", "no"
    if left and right and left == right:
        return "same_classifier_class_review", "no"
    if left or right:
        return "classifier_kind_switch_review", "yes"
    return "no_classifier_evidence", "no"


def page_kind_transition(left_kind: str, right_kind: str) -> str:
    left = left_kind or ""
    right = right_kind or ""
    if left == right and left:
        return "same_page_kind"
    if "context_or_blank" in {left, right}:
        return "context_boundary"
    if "derivative_selected_caste_race_tribe" in {left, right}:
        return "derivative_boundary"
    if "offtarget_table" in {left, right}:
        return "offtarget_boundary"
    if left and right:
        return "page_kind_switch"
    return "unknown"


def classify_edge(row: dict[str, str], left_class: str, right_class: str) -> dict[str, object]:
    layer = row.get("adjacency_final_decision_layer", "")
    gap = int_or_none(row.get("page_gap"))
    consecutive = gap == 1
    same_family = same_nonblank(row.get("prev_table_family_key"), row.get("table_family_key"))
    same_number = same_nonblank(row.get("prev_table_number_key"), row.get("table_number_key"))
    right_page = int_or_none(row.get("pdf_page"))
    right_no_labels = ratio_no_labels(row.get("row_label_ratio"))
    left_has_labels = ratio_has_labels(row.get("prev_row_label_ratio"))
    row_label_dependency = (
        consecutive
        and left_has_labels
        and right_no_labels
        and same_family == "yes"
        and str(row.get("row_label_scan_pattern") or "") == "left_labelled_right_unlabelled"
    )
    class_transition, class_break = page_class_transition(left_class, right_class)
    kind_transition = page_kind_transition(row.get("prev_page_kind", ""), row.get("page_kind", ""))
    title_signal = title_switch(row.get("prev_title", ""), row.get("title", ""), same_family, same_number)

    flags: list[str] = ["pairwise_only"]
    status = "hard_break_or_not_candidate"
    site_action = "do_not_merge"
    promotion = ""
    likely_missing = "no"
    likely_ocr = "no"
    manual_review = "no"
    value_review = "no"

    if not consecutive:
        status = "likely_assemblable_after_missing_pages" if layer == "missing_intermediate_pages_candidate" else "non_consecutive_break"
        site_action = "render_or_attach_missing_intermediate_pages_then_retest" if layer == "missing_intermediate_pages_candidate" else "do_not_merge_non_consecutive"
        promotion = "Only reconsider after every missing intermediate PDF page is rendered/OCRed and adjacent edges are retested."
        likely_missing = "yes" if layer == "missing_intermediate_pages_candidate" else "no"
        flags.append("non_consecutive")
        flags.extend(f"missing_page_{page}" for page in missing_pages(row.get("prev_pdf_page"), row.get("pdf_page")))
    elif layer == "confirmed_physical_pair" and row.get("can_be_physical_edge_now") == "yes":
        status = "confirmed_physical_edge"
        site_action = "use_inside_confirmed_physical_component"
        value_review = "yes"
        flags.append("physical_merge_allowed")
    elif layer == "blocked_physical_missing_ocr_artifact":
        status = "candidate_physical_missing_ocr"
        site_action = "ocr_or_attach_missing_artifacts_then_retest"
        promotion = "Promote only if row assembly constraints pass after both OCR artifacts are present."
        likely_ocr = "yes"
        manual_review = "yes"
        flags.append("missing_ocr_artifact")
    elif layer == "row_label_dependency_review_candidate":
        status = "candidate_row_label_dependency"
        site_action = "test_row_label_inheritance_then_review"
        promotion = "Promote only if row labels from the left page safely align to the right page values and column order passes."
        likely_ocr = "yes" if "missing_ocr" in str(row.get("decision_review_flags", "")) else "no"
        manual_review = "yes"
        flags.append("row_label_dependency_candidate")
    elif layer in {"failed_physical_constraints_review", "partial_physical_pair_review"}:
        status = "failed_or_partial_physical_review"
        site_action = "do_not_merge_until_constraints_are_repaired"
        promotion = "Needs constraint repair or manual review; same title/family is insufficient."
        manual_review = "yes"
        flags.append("failed_or_partial_constraints")
    elif layer == "logical_collection_only_confirmed":
        status = "confirmed_logical_only_edge"
        site_action = "use_as_logical_collection_edge_only"
        flags.extend(["logical_only", "no_physical_merge"])
    elif layer == "logical_collection_only_review":
        status = "review_logical_only_edge"
        site_action = "review_as_logical_collection_edge_only"
        manual_review = "yes"
        flags.extend(["logical_only", "no_physical_merge", "not_promoted"])
    else:
        if row_label_dependency:
            status = "unpromoted_row_label_dependency_signal"
            site_action = "review_row_label_signal_not_promoted"
            manual_review = "yes"
            flags.append("row_label_dependency_signal")
        else:
            status = "hard_break_or_not_candidate"
            site_action = "do_not_merge"
            flags.append("hard_break")

    if class_break == "yes":
        flags.append(class_transition)
    if kind_transition in {"context_boundary", "derivative_boundary", "offtarget_boundary", "page_kind_switch"}:
        flags.append(kind_transition)
    if same_family == "no":
        flags.append("table_family_switch")
    if same_number == "no":
        flags.append("table_number_switch_or_missing")
    if title_signal in {"same_family_title_change", "new_title_or_table_switch"}:
        flags.append(title_signal)

    physical_allowed = "yes" if status == "confirmed_physical_edge" else "no"
    logical_allowed = "yes" if status in {"confirmed_logical_only_edge", "review_logical_only_edge"} and row.get("can_be_logical_group_now") == "yes" else "no"
    return {
        "physical_rule_status": status,
        "physical_merge_allowed_now": physical_allowed,
        "logical_group_allowed_now": logical_allowed,
        "site_action": site_action,
        "promotion_condition": promotion,
        "likely_after_missing_pages": likely_missing,
        "likely_after_ocr": likely_ocr,
        "requires_manual_review": manual_review,
        "requires_value_review": value_review,
        "row_label_dependency_signal": "yes" if row_label_dependency else "no",
        "right_page_parity": "odd" if right_page is not None and right_page % 2 else "even" if right_page is not None else "",
        "right_page_no_row_labels": "yes" if right_no_labels else "no",
        "same_table_family": same_family,
        "same_table_number": same_number,
        "page_kind_transition": kind_transition,
        "page_class_transition": class_transition,
        "classifier_break_signal": class_break,
        "title_switch_signal": title_signal,
        "rule_flags": semi(sorted(set(flags))),
    }


def build_edge_rulebook(edge_rows: list[dict[str, str]], class_by_page: dict[tuple[str, str], dict[str, str]]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for row in edge_rows:
        source_guess = source_id_from_text(
            row.get("source_pdf_id"),
            row.get("source_pdf"),
            row.get("v5_source_key"),
            row.get("prev_image_path"),
            row.get("image_path"),
        )
        left_page = str(row.get("prev_pdf_page") or "")
        right_page = str(row.get("pdf_page") or "")
        left_class_row = class_by_page.get((source_guess, left_page), {})
        right_class_row = class_by_page.get((source_guess, right_page), {})
        left_class = left_class_row.get("page_class", "")
        right_class = right_class_row.get("page_class", "")
        classified = classify_edge(row, left_class, right_class)
        rule_id = stable_id("v34edge", row.get("v5_source_key", ""), left_page, right_page, row.get("adjacency_final_decision_layer", ""))
        output.append(
            {
                "rule_id": rule_id,
                "edge_key": row.get("edge_key", ""),
                "v5_source_key": row.get("v5_source_key", ""),
                "source_pdf_id_guess": source_guess,
                "prev_pdf_page": left_page,
                "pdf_page": right_page,
                "page_pair": page_pair(left_page, right_page),
                "page_gap": row.get("page_gap", ""),
                "pages_consecutive": row.get("pages_consecutive", ""),
                "left_page_class": left_class,
                "right_page_class": right_class,
                "left_row_label_ratio": row.get("prev_row_label_ratio", ""),
                "right_row_label_ratio": row.get("row_label_ratio", ""),
                "row_label_scan_strength": row.get("row_label_scan_strength", ""),
                "row_label_scan_pattern": row.get("row_label_scan_pattern", ""),
                "prev_page_kind": row.get("prev_page_kind", ""),
                "page_kind": row.get("page_kind", ""),
                "prev_table_family_key": row.get("prev_table_family_key", ""),
                "table_family_key": row.get("table_family_key", ""),
                "prev_table_number_key": row.get("prev_table_number_key", ""),
                "table_number_key": row.get("table_number_key", ""),
                "prev_title": row.get("prev_title", ""),
                "title": row.get("title", ""),
                "decision_reason": row.get("decision_reason", ""),
                "source_decision_layer": row.get("adjacency_final_decision_layer", ""),
                "frontier_statuses": row.get("frontier_statuses", ""),
                "frontier_action_buckets": row.get("frontier_action_buckets", ""),
                "component_ids": row.get("component_ids", ""),
                "prev_image_path": row.get("prev_image_path", ""),
                "image_path": row.get("image_path", ""),
                **classified,
            }
        )
    output.sort(
        key=lambda row: (
            str(row.get("v5_source_key", "")),
            int_or_none(row.get("prev_pdf_page")) or 10**9,
            int_or_none(row.get("pdf_page")) or 10**9,
            str(row.get("rule_id", "")),
        )
    )
    for idx, row in enumerate(output, start=1):
        row["rule_rank"] = idx
    return output


def build_group_registry(
    components: list[dict[str, str]],
    edge_rulebook: list[dict[str, object]],
    logical_families: list[dict[str, str]],
) -> list[dict[str, object]]:
    groups: list[dict[str, object]] = []
    for component in components:
        source_guess = source_id_from_text(component.get("source_pdf_id"), component.get("source_pdf"), component.get("v5_source_key"))
        groups.append(
            {
                "group_id": f"physical::{component.get('component_id', '')}",
                "group_type": "physical_component",
                "group_status": "confirmed",
                "physical_merge_allowed_now": "yes",
                "logical_collection_only": "no",
                "safe_to_publish_values": component.get("safe_to_publish_values", ""),
                "review_required": component.get("value_review_required", ""),
                "v5_source_key": component.get("v5_source_key", ""),
                "source_pdf_id_guess": source_guess,
                "page_range": component.get("page_range", ""),
                "component_pages": component.get("component_pages", ""),
                "n_pages": component.get("n_pages", ""),
                "edge_pairs": component.get("edge_pairs", ""),
                "member_count": component.get("n_pages", ""),
                "primary_title": "",
                "religion_terms_detected": "",
                "reason": component.get("component_reason", ""),
                "rule_flags": "confirmed_consecutive_physical_component; physical_merge_allowed",
                "recommended_action": component.get("recommended_action", ""),
                "evidence_file": "confirmed_physical_components.csv",
            }
        )

    candidate_statuses = {
        "candidate_physical_missing_ocr",
        "candidate_row_label_dependency",
        "failed_or_partial_physical_review",
        "likely_assemblable_after_missing_pages",
        "unpromoted_row_label_dependency_signal",
    }
    for edge in edge_rulebook:
        status = str(edge.get("physical_rule_status") or "")
        if status not in candidate_statuses:
            continue
        source_guess = str(edge.get("source_pdf_id_guess") or "")
        page_gap = int_or_none(edge.get("page_gap"))
        pages = [str(edge.get("prev_pdf_page", "")), str(edge.get("pdf_page", ""))]
        if page_gap and page_gap > 1:
            pages = [str(edge.get("prev_pdf_page", "")), *[str(p) for p in missing_pages(edge.get("prev_pdf_page"), edge.get("pdf_page"))], str(edge.get("pdf_page", ""))]
        groups.append(
            {
                "group_id": f"candidate::{edge.get('rule_id', '')}",
                "group_type": "physical_candidate_edge",
                "group_status": status,
                "physical_merge_allowed_now": "no",
                "logical_collection_only": "no",
                "safe_to_publish_values": "no",
                "review_required": "yes",
                "v5_source_key": edge.get("v5_source_key", ""),
                "source_pdf_id_guess": source_guess,
                "page_range": f"{pages[0]}-{pages[-1]}" if pages else "",
                "component_pages": semi(pages),
                "n_pages": len(pages),
                "edge_pairs": edge.get("page_pair", ""),
                "member_count": len(pages),
                "primary_title": edge.get("title", ""),
                "religion_terms_detected": "",
                "reason": edge.get("decision_reason", ""),
                "rule_flags": edge.get("rule_flags", ""),
                "recommended_action": edge.get("promotion_condition") or edge.get("site_action", ""),
                "evidence_file": "edge_rulebook.csv",
            }
        )

    for family in logical_families:
        status = "confirmed" if family.get("frontier_status") == "confirmed_logical_collection" else "review"
        groups.append(
            {
                "group_id": f"logical::{family.get('logical_family_id') or family.get('family_id') or family.get('assembly_group_id') or stable_id('v34logical', family.get('v5_source_key', ''), family.get('member_pages', ''))}",
                "group_type": "logical_family_collection",
                "group_status": status,
                "physical_merge_allowed_now": "no",
                "logical_collection_only": "yes",
                "safe_to_publish_values": "not_applicable",
                "review_required": "no" if status == "confirmed" else "yes",
                "v5_source_key": family.get("v5_source_key", ""),
                "source_pdf_id_guess": source_id_from_text(family.get("source_pdf_id"), family.get("source_pdf"), family.get("v5_source_key")),
                "page_range": family.get("page_ranges", ""),
                "component_pages": family.get("member_pages", ""),
                "n_pages": family.get("n_physical_pages", ""),
                "edge_pairs": "",
                "member_count": family.get("member_count", ""),
                "primary_title": family.get("primary_title", ""),
                "religion_terms_detected": family.get("religion_terms_detected", ""),
                "reason": family.get("decision_reason", ""),
                "rule_flags": "logical_collection_only; no_physical_merge",
                "recommended_action": family.get("recommended_action", ""),
                "evidence_file": "logical_family_frontier.csv",
            }
        )
    groups.sort(
        key=lambda row: (
            {"physical_component": 0, "physical_candidate_edge": 1, "logical_family_collection": 2}.get(str(row.get("group_type")), 9),
            str(row.get("v5_source_key", "")),
            int_or_none(str(row.get("page_range", "")).split("-", 1)[0]) or 10**9,
            str(row.get("group_id", "")),
        )
    )
    for idx, row in enumerate(groups, start=1):
        row["group_rank"] = idx
    return groups


def build_page_signal_index(edge_rulebook: list[dict[str, object]], class_by_page: dict[tuple[str, str], dict[str, str]]) -> list[dict[str, object]]:
    pages: dict[tuple[str, str], dict[str, object]] = {}
    incoming: dict[tuple[str, str], list[str]] = defaultdict(list)
    outgoing: dict[tuple[str, str], list[str]] = defaultdict(list)

    def ensure_page(edge: dict[str, object], side: str) -> None:
        if side == "left":
            pdf_page = str(edge.get("prev_pdf_page", ""))
            ratio = edge.get("left_row_label_ratio", "")
            kind = edge.get("prev_page_kind", "")
            family = edge.get("prev_table_family_key", "")
            number = edge.get("prev_table_number_key", "")
            title = edge.get("prev_title", "")
            image = edge.get("prev_image_path", "")
        else:
            pdf_page = str(edge.get("pdf_page", ""))
            ratio = edge.get("right_row_label_ratio", "")
            kind = edge.get("page_kind", "")
            family = edge.get("table_family_key", "")
            number = edge.get("table_number_key", "")
            title = edge.get("title", "")
            image = edge.get("image_path", "")
        src = str(edge.get("v5_source_key", ""))
        source_guess = str(edge.get("source_pdf_id_guess", ""))
        if not src or not pdf_page:
            return
        classifier = class_by_page.get((source_guess, pdf_page), {})
        key = (src, pdf_page)
        pages.setdefault(
            key,
            {
                "page_id": stable_id("v34page", src, pdf_page),
                "v5_source_key": src,
                "source_pdf_id_guess": source_guess,
                "pdf_page": pdf_page,
                "page_parity": "odd" if (int_or_none(pdf_page) or 0) % 2 else "even" if int_or_none(pdf_page) is not None else "",
                "page_kind": kind,
                "page_class": classifier.get("page_class", ""),
                "contains_numeric_grid": classifier.get("contains_numeric_grid", ""),
                "table_family_key": family,
                "table_number_key": number,
                "row_label_ratio": ratio,
                "no_row_labels_signal": "yes" if ratio_no_labels(ratio) else "no",
                "title": title,
                "image_path": image,
            },
        )

    for edge in edge_rulebook:
        ensure_page(edge, "left")
        ensure_page(edge, "right")
        left_key = (str(edge.get("v5_source_key", "")), str(edge.get("prev_pdf_page", "")))
        right_key = (str(edge.get("v5_source_key", "")), str(edge.get("pdf_page", "")))
        outgoing[left_key].append(str(edge.get("physical_rule_status", "")))
        incoming[right_key].append(str(edge.get("physical_rule_status", "")))

    rows: list[dict[str, object]] = []
    for key, row in pages.items():
        row = dict(row)
        row["incoming_rule_statuses"] = semi(sorted(set(incoming.get(key, []))))
        row["outgoing_rule_statuses"] = semi(sorted(set(outgoing.get(key, []))))
        row["incoming_edge_count"] = len(incoming.get(key, []))
        row["outgoing_edge_count"] = len(outgoing.get(key, []))
        rows.append(row)
    rows.sort(key=lambda row: (str(row.get("v5_source_key", "")), int_or_none(row.get("pdf_page")) or 10**9))
    for idx, row in enumerate(rows, start=1):
        row["page_rank"] = idx
    return rows


def build_promotion_queue(edge_rulebook: list[dict[str, object]]) -> list[dict[str, object]]:
    queue: list[dict[str, object]] = []
    priority_by_status = {
        "candidate_row_label_dependency": 1,
        "candidate_physical_missing_ocr": 2,
        "likely_assemblable_after_missing_pages": 3,
        "unpromoted_row_label_dependency_signal": 4,
        "failed_or_partial_physical_review": 5,
        "review_logical_only_edge": 6,
    }
    for edge in edge_rulebook:
        status = str(edge.get("physical_rule_status") or "")
        if status not in priority_by_status:
            continue
        queue_type = {
            "candidate_row_label_dependency": "row_label_dependency_test",
            "candidate_physical_missing_ocr": "ocr_then_retest",
            "likely_assemblable_after_missing_pages": "missing_pages_then_retest",
            "unpromoted_row_label_dependency_signal": "manual_row_label_signal_review",
            "failed_or_partial_physical_review": "constraint_repair_review",
            "review_logical_only_edge": "logical_edge_review",
        }[status]
        queue.append(
            {
                "queue_type": queue_type,
                "priority": priority_by_status[status],
                "v5_source_key": edge.get("v5_source_key", ""),
                "source_pdf_id_guess": edge.get("source_pdf_id_guess", ""),
                "page_pair": edge.get("page_pair", ""),
                "missing_pages": semi(missing_pages(edge.get("prev_pdf_page"), edge.get("pdf_page"))),
                "physical_rule_status": status,
                "site_action": edge.get("site_action", ""),
                "promotion_condition": edge.get("promotion_condition", ""),
                "row_label_dependency_signal": edge.get("row_label_dependency_signal", ""),
                "left_page_class": edge.get("left_page_class", ""),
                "right_page_class": edge.get("right_page_class", ""),
                "page_class_transition": edge.get("page_class_transition", ""),
                "rule_flags": edge.get("rule_flags", ""),
                "decision_reason": edge.get("decision_reason", ""),
                "prev_image_path": edge.get("prev_image_path", ""),
                "image_path": edge.get("image_path", ""),
            }
        )
    queue.sort(
        key=lambda row: (
            int(row.get("priority") or 99),
            str(row.get("v5_source_key", "")),
            int_or_none(str(row.get("page_pair", "")).split("-", 1)[0]) or 10**9,
        )
    )
    for idx, row in enumerate(queue, start=1):
        row["queue_rank"] = idx
    return queue


def build_summary(
    edge_rulebook: list[dict[str, object]],
    group_registry: list[dict[str, object]],
    page_index: list[dict[str, object]],
    promotion_queue: list[dict[str, object]],
    paths: dict[str, Path],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = [
        {"metric": "paths", "name": name, "count": output_path(path)}
        for name, path in paths.items()
    ]
    rows.extend(
        [
            {"metric": "counts", "name": "edge_rulebook_rows", "count": len(edge_rulebook)},
            {"metric": "counts", "name": "group_registry_rows", "count": len(group_registry)},
            {"metric": "counts", "name": "page_signal_rows", "count": len(page_index)},
            {"metric": "counts", "name": "promotion_queue_rows", "count": len(promotion_queue)},
        ]
    )
    counters = [
        ("edges_by_physical_rule_status", Counter(str(row.get("physical_rule_status", "")) for row in edge_rulebook)),
        ("edges_by_site_action", Counter(str(row.get("site_action", "")) for row in edge_rulebook)),
        ("edges_by_classifier_transition", Counter(str(row.get("page_class_transition", "")) for row in edge_rulebook)),
        ("groups_by_type", Counter(str(row.get("group_type", "")) for row in group_registry)),
        ("groups_by_status", Counter(str(row.get("group_status", "")) for row in group_registry)),
        ("promotion_queue_by_type", Counter(str(row.get("queue_type", "")) for row in promotion_queue)),
        ("pages_by_class", Counter(str(row.get("page_class", "")) for row in page_index)),
        ("pages_by_no_row_labels", Counter(str(row.get("no_row_labels_signal", "")) for row in page_index)),
    ]
    for metric, counter in counters:
        for name, count in sorted(counter.items()):
            rows.append({"metric": metric, "name": name, "count": count})
    return rows


def audit_outputs(
    edge_rulebook: list[dict[str, object]],
    group_registry: list[dict[str, object]],
    promotion_queue: list[dict[str, object]],
    source_edge_count: int,
    workbook_path: Path,
) -> list[dict[str, object]]:
    audits: list[dict[str, object]] = []

    audits.append(
        {
            "check": "edge_rulebook_covers_all_source_edges",
            "status": "pass" if len(edge_rulebook) == source_edge_count else "fail",
            "detail": f"rulebook={len(edge_rulebook)} source={source_edge_count}",
        }
    )
    bad_physical_edges = [
        row for row in edge_rulebook
        if row.get("physical_merge_allowed_now") == "yes"
        and (row.get("pages_consecutive") != "yes" or row.get("physical_rule_status") != "confirmed_physical_edge")
    ]
    audits.append(
        {
            "check": "physical_edges_only_confirmed_consecutive",
            "status": "pass" if not bad_physical_edges else "fail",
            "detail": f"bad={len(bad_physical_edges)}",
        }
    )
    bad_group_merges = [
        row for row in group_registry
        if row.get("physical_merge_allowed_now") == "yes" and row.get("group_type") != "physical_component"
    ]
    audits.append(
        {
            "check": "only_physical_components_allow_physical_merge",
            "status": "pass" if not bad_group_merges else "fail",
            "detail": f"bad={len(bad_group_merges)}",
        }
    )
    bad_logical = [
        row for row in group_registry
        if row.get("logical_collection_only") == "yes" and row.get("physical_merge_allowed_now") == "yes"
    ]
    audits.append(
        {
            "check": "logical_groups_never_allow_physical_merge",
            "status": "pass" if not bad_logical else "fail",
            "detail": f"bad={len(bad_logical)}",
        }
    )
    row_label_edges = [
        row for row in edge_rulebook
        if row.get("physical_rule_status") == "candidate_row_label_dependency"
    ]
    row_label_queued = [
        row for row in promotion_queue
        if row.get("queue_type") == "row_label_dependency_test"
    ]
    audits.append(
        {
            "check": "row_label_dependency_candidates_are_queued",
            "status": "pass" if len(row_label_edges) == len(row_label_queued) else "fail",
            "detail": f"edges={len(row_label_edges)} queued={len(row_label_queued)}",
        }
    )
    bad_missing = [
        row for row in edge_rulebook
        if int_or_none(row.get("page_gap")) and int_or_none(row.get("page_gap")) != 1 and row.get("physical_merge_allowed_now") == "yes"
    ]
    audits.append(
        {
            "check": "missing_page_bridges_never_allow_physical_merge",
            "status": "pass" if not bad_missing else "fail",
            "detail": f"bad={len(bad_missing)}",
        }
    )
    classifier_conflicts = [
        row for row in edge_rulebook
        if row.get("physical_merge_allowed_now") == "yes" and row.get("classifier_break_signal") == "yes"
    ]
    audits.append(
        {
            "check": "confirmed_physical_edges_have_no_classifier_break",
            "status": "pass" if not classifier_conflicts else "fail",
            "detail": f"conflicts={len(classifier_conflicts)}",
        }
    )
    try:
        with zipfile.ZipFile(workbook_path) as zf:
            bad_member = zf.testzip()
    except Exception as exc:  # pragma: no cover - defensive reporting
        bad_member = f"{type(exc).__name__}: {exc}"
    audits.append(
        {
            "check": "workbook_zip_integrity",
            "status": "pass" if bad_member is None else "fail",
            "detail": str(bad_member or "ok"),
        }
    )
    return audits


def write_workbook(
    path: Path,
    summary_rows: list[dict[str, object]],
    edge_rows: list[dict[str, object]],
    group_rows: list[dict[str, object]],
    page_rows: list[dict[str, object]],
    queue_rows: list[dict[str, object]],
    audit_rows: list[dict[str, object]],
) -> None:
    workbook = xlsxwriter.Workbook(path, {"constant_memory": True})
    header_fmt = workbook.add_format({"bold": True, "bg_color": "#E8EEF7", "border": 1})
    wrap_fmt = workbook.add_format({"text_wrap": True, "valign": "top"})
    sheets = [
        ("Summary", summary_rows, SUMMARY_FIELDS),
        ("Audit", audit_rows, AUDIT_FIELDS),
        ("Grouping Registry", group_rows, GROUP_FIELDS),
        ("Promotion Queue", queue_rows, QUEUE_FIELDS),
        ("Edge Rulebook", edge_rows, EDGE_FIELDS),
        ("Page Signals", page_rows, PAGE_FIELDS),
    ]
    for name, rows, fields in sheets:
        ws = workbook.add_worksheet(name[:31])
        for col, field in enumerate(fields):
            ws.write(0, col, field, header_fmt)
        for row_idx, row in enumerate(rows, start=1):
            for col, field in enumerate(fields):
                value = row.get(field, "")
                fmt = wrap_fmt if field in {"decision_reason", "rule_flags", "recommended_action", "promotion_condition", "prev_title", "title"} else None
                ws.write(row_idx, col, value, fmt)
        ws.freeze_panes(1, 0)
        ws.autofilter(0, 0, max(0, len(rows)), max(0, len(fields) - 1))
        for col, field in enumerate(fields):
            width = min(max(len(field) + 2, 12), 54)
            ws.set_column(col, col, width)
    workbook.close()


def write_outputs(
    out_dir: Path,
    edge_rulebook: list[dict[str, object]],
    group_registry: list[dict[str, object]],
    page_index: list[dict[str, object]],
    promotion_queue: list[dict[str, object]],
    summary_rows: list[dict[str, object]],
    audit_rows: list[dict[str, object]],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "edge_rulebook.csv", edge_rulebook, EDGE_FIELDS)
    write_csv(out_dir / "grouping_registry.csv", group_registry, GROUP_FIELDS)
    write_csv(out_dir / "page_signal_index.csv", page_index, PAGE_FIELDS)
    write_csv(out_dir / "promotion_queue.csv", promotion_queue, QUEUE_FIELDS)
    write_csv(out_dir / "summary.csv", summary_rows, SUMMARY_FIELDS)
    write_csv(out_dir / "audit.csv", audit_rows, AUDIT_FIELDS)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assembly-dir", default=str(DEFAULT_ASSEMBLY_DIR))
    parser.add_argument("--frontier-dir", default=str(DEFAULT_FRONTIER_DIR))
    parser.add_argument("--component-dir", default=str(DEFAULT_COMPONENT_DIR))
    parser.add_argument("--logical-dir", default=str(DEFAULT_LOGICAL_DIR))
    parser.add_argument("--control-dir", default=str(DEFAULT_CONTROL_DIR))
    parser.add_argument("--classifier-csv", default=str(DEFAULT_CLASSIFIER_CSV))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    args = parser.parse_args()

    assembly_dir = Path(args.assembly_dir).expanduser()
    frontier_dir = Path(args.frontier_dir).expanduser()
    component_dir = Path(args.component_dir).expanduser()
    logical_dir = Path(args.logical_dir).expanduser()
    control_dir = Path(args.control_dir).expanduser()
    classifier_csv = Path(args.classifier_csv).expanduser()
    out_dir = Path(args.out_dir).expanduser()
    paths = {
        "assembly_dir": assembly_dir,
        "frontier_dir": frontier_dir,
        "component_dir": component_dir,
        "logical_dir": logical_dir,
        "control_dir": control_dir,
        "classifier_csv": classifier_csv,
        "out_dir": out_dir,
    }

    edge_coverage = read_csv(component_dir / "consecutive_edge_coverage.csv")
    components = read_csv(component_dir / "confirmed_physical_components.csv")
    logical_families = read_csv(logical_dir / "logical_family_frontier.csv")
    class_rows = read_csv(classifier_csv)
    class_by_page = classifier_index(class_rows)

    edge_rulebook = build_edge_rulebook(edge_coverage, class_by_page)
    group_registry = build_group_registry(components, edge_rulebook, logical_families)
    page_index = build_page_signal_index(edge_rulebook, class_by_page)
    promotion_queue = build_promotion_queue(edge_rulebook)
    summary_rows = build_summary(edge_rulebook, group_registry, page_index, promotion_queue, paths)

    out_dir.mkdir(parents=True, exist_ok=True)
    workbook_path = out_dir / "assembly_rulebook_workbook.xlsx"
    # Write audits after workbook creation so zip integrity can be tested.
    preliminary_audits: list[dict[str, object]] = []
    write_workbook(workbook_path, summary_rows, edge_rulebook, group_registry, page_index, promotion_queue, preliminary_audits)
    audit_rows = audit_outputs(edge_rulebook, group_registry, promotion_queue, len(edge_coverage), workbook_path)
    summary_rows.extend(
        {"metric": "audits", "name": row["check"], "count": row["status"]}
        for row in audit_rows
    )
    # Rewrite workbook with final summary/audit rows.
    write_workbook(workbook_path, summary_rows, edge_rulebook, group_registry, page_index, promotion_queue, audit_rows)
    audit_rows = audit_outputs(edge_rulebook, group_registry, promotion_queue, len(edge_coverage), workbook_path)
    write_outputs(out_dir, edge_rulebook, group_registry, page_index, promotion_queue, summary_rows, audit_rows)

    summary = {
        "out_dir": output_path(out_dir),
        "edge_rulebook_rows": len(edge_rulebook),
        "grouping_registry_rows": len(group_registry),
        "page_signal_rows": len(page_index),
        "promotion_queue_rows": len(promotion_queue),
        "physical_rule_status_counts": dict(Counter(str(row.get("physical_rule_status", "")) for row in edge_rulebook)),
        "group_type_counts": dict(Counter(str(row.get("group_type", "")) for row in group_registry)),
        "promotion_queue_counts": dict(Counter(str(row.get("queue_type", "")) for row in promotion_queue)),
        "audit_failure_count": sum(1 for row in audit_rows if row.get("status") != "pass"),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
