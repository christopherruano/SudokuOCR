#!/usr/bin/env python3
"""Build a canonical assembly grouping record for a release snapshot.

The release snapshot already contains the raw proposal/work-queue sheets. This
script adds a compact policy layer that downstream review/static-site code can
consume without reinterpreting proposal types. It does not promote, demote, OCR,
or mutate source evidence.
"""

from __future__ import annotations

import argparse
import csv
import json
import zipfile
from collections import Counter
from pathlib import Path
from typing import Iterable

import xlsxwriter


ROOT = Path(__file__).resolve().parent

GROUPING_RECORD_FIELDS = [
    "record_rank",
    "grouping_id",
    "source_proposal_id",
    "grouping_layer",
    "grouping_status",
    "site_use_policy",
    "physical_merge_allowed",
    "logical_collection_only",
    "safe_to_publish_values",
    "review_required",
    "current_action",
    "next_step_type",
    "core_scope",
    "source_key",
    "source_pdf_id",
    "page_range",
    "component_pages",
    "pdf_pages_sorted",
    "pdf_page_start",
    "pdf_page_end",
    "n_pages",
    "pages_consecutive",
    "page_gap",
    "missing_intermediate_pages",
    "table_number_key",
    "topic_variant_key",
    "religion_terms_detected",
    "row_label_dependency_signal",
    "right_page_no_row_labels_signal",
    "pair_direction_signal",
    "pair_direction_reason",
    "row_label_evidence",
    "row_pairing_mode",
    "row_label_source_summary",
    "column_order_evidence",
    "zero_check_status",
    "artifact_status",
    "confidence_score",
    "member_count",
    "matrix_csv",
    "title_summary",
    "reason",
    "review_flags",
    "source_sheet",
    "promotion_condition",
]

RULE_POLICY_FIELDS = [
    "rule_rank",
    "rule_layer",
    "rule_name",
    "physical_merge_allowed",
    "logical_collection_allowed",
    "site_ready_allowed",
    "requirement",
    "why_it_exists",
    "failure_mode",
]

SUMMARY_FIELDS = ["metric", "name", "value", "detail"]
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


def int_value(value: object) -> int:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return 0


def split_list(value: object) -> list[str]:
    return [part.strip() for part in str(value or "").split(";") if part.strip()]


def page_gap(row: dict[str, str]) -> int:
    start = int_value(row.get("pdf_page_start"))
    end = int_value(row.get("pdf_page_end"))
    n_pages = int_value(row.get("n_pages")) or len(split_list(row.get("component_pages")))
    if not start or not end:
        return 0
    return max(0, end - start + 1 - n_pages)


def grouping_layer(proposal_type: str) -> str:
    return {
        "physical_pairwise_assembly": "physical_pairwise",
        "manual_preview_physical_candidate": "manual_physical_review",
        "manual_preview_review_candidate": "manual_review_only",
        "logical_sibling_collection": "logical_only",
        "logical_pairwise_edge_candidate": "logical_only",
        "blocked_missing_ocr_edge": "wait_for_ocr",
        "blocked_missing_page_bridge": "missing_page_bridge",
        "blocked_conflict_edge": "conflict_keep_separate",
    }.get(proposal_type, "unknown_review")


def grouping_status(proposal: dict[str, str], classified: dict[str, str]) -> str:
    proposal_type = proposal.get("proposal_type", "")
    if proposal_type == "physical_pairwise_assembly":
        if proposal.get("pages_consecutive") != "yes":
            return "invalid_physical_nonconsecutive_review"
        if proposal.get("physical_merge_allowed") != "yes":
            return "invalid_physical_policy_review"
        if proposal.get("proposal_status") != "materialized_physical_structure":
            return "physical_structure_review_required"
        if proposal.get("review_required") == "yes" or proposal.get("safe_to_publish_values") != "yes":
            return "physical_structure_ready_value_review"
        return "physical_structure_ready"
    if proposal_type == "manual_preview_physical_candidate":
        return "manual_visual_review_before_physical_promotion"
    if proposal_type == "manual_preview_review_candidate":
        return "manual_review_before_any_grouping"
    if proposal_type in {"logical_sibling_collection", "logical_pairwise_edge_candidate"}:
        if proposal.get("site_ready_logical_collection") == "yes":
            return "site_ready_logical_only"
        return "review_logical_only"
    if proposal_type == "blocked_missing_ocr_edge":
        return "waiting_for_ocr_then_adjacent_retest"
    if proposal_type == "blocked_missing_page_bridge":
        return "waiting_for_missing_pages_then_adjacent_retest"
    if proposal_type == "blocked_conflict_edge":
        return "documented_conflict_keep_separate"
    return classified.get("next_step_type", "") or proposal.get("proposal_status", "") or "unknown_review"


def site_use_policy(proposal: dict[str, str], status: str) -> str:
    if status.startswith("physical_structure_ready"):
        return "use_now_physical_structure_only"
    if status == "physical_structure_review_required":
        return "review_before_site_use"
    if status == "site_ready_logical_only":
        return "use_now_logical_navigation_only"
    if status.startswith("manual_"):
        return "review_before_site_use"
    if status.startswith("waiting_for_"):
        return "recovery_worklist_only"
    if status == "documented_conflict_keep_separate":
        return "keep_separate"
    if proposal.get("logical_collection_only") == "yes":
        return "logical_review_only"
    return "review_only"


def build_grouping_record(
    proposals: list[dict[str, str]],
    members: list[dict[str, str]],
    classified_rows: list[dict[str, str]],
) -> list[dict[str, object]]:
    classified_by_id = {row.get("proposal_id", ""): row for row in classified_rows}
    members_by_id: dict[str, list[dict[str, str]]] = {}
    for member in members:
        members_by_id.setdefault(member.get("proposal_id", ""), []).append(member)

    records: list[dict[str, object]] = []
    for rank, proposal in enumerate(proposals, start=1):
        proposal_id = proposal.get("proposal_id", "")
        classified = classified_by_id.get(proposal_id, {})
        layer = grouping_layer(proposal.get("proposal_type", ""))
        status = grouping_status(proposal, classified)
        member_rows = members_by_id.get(proposal_id, [])
        table_number_key = sample_join(member.get("table_number_key", "") for member in member_rows)
        topic_variant_key = sample_join(member.get("topic_variant_key", "") for member in member_rows)
        religion_terms = sample_join(member.get("religion_terms_detected", "") for member in member_rows)
        gap = page_gap(proposal)
        records.append(
            {
                "record_rank": rank,
                "grouping_id": f"grouping::{rank:04d}",
                "source_proposal_id": proposal_id,
                "grouping_layer": layer,
                "grouping_status": status,
                "site_use_policy": site_use_policy(proposal, status),
                "physical_merge_allowed": proposal.get("physical_merge_allowed", ""),
                "logical_collection_only": proposal.get("logical_collection_only", ""),
                "safe_to_publish_values": proposal.get("safe_to_publish_values", ""),
                "review_required": proposal.get("review_required", ""),
                "current_action": proposal.get("recommended_action", "") or classified.get("recommended_action", ""),
                "next_step_type": classified.get("next_step_type", ""),
                "core_scope": classified.get("core_scope", ""),
                "source_key": proposal.get("source_key", ""),
                "source_pdf_id": proposal.get("source_pdf_id", ""),
                "page_range": proposal.get("page_range", ""),
                "component_pages": proposal.get("component_pages", ""),
                "pdf_pages_sorted": proposal.get("pdf_pages_sorted", ""),
                "pdf_page_start": proposal.get("pdf_page_start", ""),
                "pdf_page_end": proposal.get("pdf_page_end", ""),
                "n_pages": proposal.get("n_pages", ""),
                "pages_consecutive": proposal.get("pages_consecutive", ""),
                "page_gap": gap,
                "missing_intermediate_pages": "; ".join(str(i) for i in missing_intermediate_pages(proposal)),
                "table_number_key": table_number_key,
                "topic_variant_key": topic_variant_key,
                "religion_terms_detected": religion_terms,
                "row_label_dependency_signal": proposal.get("row_label_signal", ""),
                "right_page_no_row_labels_signal": proposal.get("right_page_no_row_labels_signal", ""),
                "pair_direction_signal": proposal.get("pair_direction_signal", ""),
                "pair_direction_reason": proposal.get("pair_direction_reason", ""),
                "row_label_evidence": proposal.get("row_label_source_summary", ""),
                "row_pairing_mode": proposal.get("row_pairing_mode", ""),
                "row_label_source_summary": proposal.get("row_label_source_summary", ""),
                "column_order_evidence": proposal.get("column_order_evidence", ""),
                "zero_check_status": proposal.get("zero_check_status", ""),
                "artifact_status": proposal.get("artifact_status", ""),
                "confidence_score": proposal.get("confidence_score", ""),
                "member_count": proposal.get("member_count", ""),
                "matrix_csv": proposal.get("matrix_csv", ""),
                "title_summary": proposal.get("title_summary", ""),
                "reason": proposal.get("reason", "") or classified.get("reason", ""),
                "review_flags": proposal.get("review_flags", "") or classified.get("review_flags", ""),
                "source_sheet": proposal.get("decision_source", ""),
                "promotion_condition": proposal.get("promotion_condition", "") or classified.get("promotion_condition", ""),
            }
        )
    return records


def sample_join(values: Iterable[object], limit: int = 8) -> str:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        for part in split_list(value):
            if part and part not in seen:
                seen.add(part)
                out.append(part)
                if len(out) >= limit:
                    return "; ".join(out)
    return "; ".join(out)


def missing_intermediate_pages(row: dict[str, str]) -> list[int]:
    if row.get("pages_consecutive") == "yes":
        return []
    start = int_value(row.get("pdf_page_start"))
    end = int_value(row.get("pdf_page_end"))
    pages = {int_value(page) for page in split_list(row.get("component_pages"))}
    if not start or not end or end <= start:
        return []
    return [page for page in range(start, end + 1) if page not in pages]


def build_rule_policy() -> list[dict[str, object]]:
    rules = [
        (
            "physical",
            "same_source",
            "yes",
            "no",
            "yes",
            "Both pages must have the same source PDF/source key.",
            "Prevents cross-volume or same-title cross-source merges.",
            "Same table title in two different books becomes a false assembly.",
        ),
        (
            "physical",
            "consecutive_pdf_pages",
            "yes",
            "no",
            "yes",
            "Physical merge requires adjacent real PDF pages.",
            "The user-requested core invariant; gaps are recovery hints only.",
            "Pages jump across missing/context pages and silently skip data.",
        ),
        (
            "physical",
            "pairwise_first",
            "yes",
            "no",
            "yes",
            "Build one adjacent edge at a time; multi-page chains need every pair verified.",
            "Avoids broad grouping by title/source/family.",
            "Overlarge components swallow unrelated subtables.",
        ),
        (
            "physical",
            "compatible_table_identity",
            "yes",
            "no",
            "yes",
            "No table-number, subject, derivative selected-caste/race/tribe, or context switch.",
            "Same source and adjacency are necessary but not sufficient.",
            "Derivative/context pages get merged into core tables.",
        ),
        (
            "physical",
            "row_label_continuation",
            "yes",
            "no",
            "yes",
            "No-label/right-page rows may inherit labels only when row order/count/schema are compatible.",
            "Captures the essential continuation pattern without guessing.",
            "Wrong row universe inherits labels and corrupts assembled rows.",
        ),
        (
            "physical",
            "column_order_preserved",
            "yes",
            "no",
            "yes",
            "Left-page columns precede right-page continuation columns unless explicit evidence says otherwise.",
            "Keeps matrices intuitive against the scan.",
            "Columns get reordered or duplicated across pages.",
        ),
        (
            "recovery",
            "missing_page_bridge",
            "no",
            "no",
            "no",
            "Nonconsecutive likely bridges remain recovery worklists until missing pages exist.",
            "A same-title bridge across a PDF gap can be useful but is not an assembly.",
            "Missing data is hidden by a false complete assembly.",
        ),
        (
            "recovery",
            "missing_ocr_edge",
            "no",
            "no",
            "no",
            "Adjacent candidate with missing OCR must wait for OCR artifact then retest.",
            "No structure decision should rely on one-sided evidence.",
            "One page's schema is projected onto an unread page.",
        ),
        (
            "logical",
            "logical_sibling_only",
            "no",
            "yes",
            "yes",
            "Same table/topic/religion sibling pages can group for navigation only.",
            "Captures Muslim/Christian/Jew/Hindu sibling sets without physical concatenation.",
            "Related subtables become one fake physical table.",
        ),
        (
            "conflict",
            "conflict_keep_separate",
            "no",
            "no",
            "yes",
            "Structural or semantic conflict edges are kept separate with reason preserved.",
            "Makes negative evidence explicit for static-site and review logic.",
            "Known-bad pair is rediscovered and reconsidered repeatedly.",
        ),
    ]
    return [
        {
            "rule_rank": idx,
            "rule_layer": layer,
            "rule_name": name,
            "physical_merge_allowed": physical,
            "logical_collection_allowed": logical,
            "site_ready_allowed": site_ready,
            "requirement": requirement,
            "why_it_exists": why,
            "failure_mode": failure,
        }
        for idx, (layer, name, physical, logical, site_ready, requirement, why, failure) in enumerate(rules, start=1)
    ]


def build_summary(records: list[dict[str, object]], policies: list[dict[str, object]]) -> list[dict[str, object]]:
    rows = [
        {"metric": "counts", "name": "assembly_grouping_record_rows", "value": len(records), "detail": "One canonical row per grouping proposal."},
        {"metric": "counts", "name": "assembly_rule_policy_rows", "value": len(policies), "detail": "Static rules for interpreting grouping records."},
        {"metric": "counts", "name": "physical_merge_allowed_rows", "value": sum(1 for row in records if row.get("physical_merge_allowed") == "yes"), "detail": "Must be strict pairwise/consecutive."},
        {"metric": "counts", "name": "logical_collection_only_rows", "value": sum(1 for row in records if row.get("logical_collection_only") == "yes"), "detail": "Never physical merges."},
        {"metric": "counts", "name": "missing_page_bridge_rows", "value": sum(1 for row in records if row.get("grouping_layer") == "missing_page_bridge"), "detail": "Recovery worklist only."},
        {"metric": "counts", "name": "wait_for_ocr_rows", "value": sum(1 for row in records if row.get("grouping_layer") == "wait_for_ocr"), "detail": "OCR/retest worklist only."},
        {"metric": "counts", "name": "conflict_keep_separate_rows", "value": sum(1 for row in records if row.get("grouping_layer") == "conflict_keep_separate"), "detail": "Explicit no-merge records."},
    ]
    rows.extend(counter_rows("grouping_layer", Counter(str(row.get("grouping_layer", "")) for row in records)))
    rows.extend(counter_rows("grouping_status", Counter(str(row.get("grouping_status", "")) for row in records)))
    rows.extend(counter_rows("site_use_policy", Counter(str(row.get("site_use_policy", "")) for row in records)))
    rows.extend(counter_rows("core_scope", Counter(str(row.get("core_scope", "")) for row in records)))
    return rows


def counter_rows(metric: str, counter: Counter[str]) -> list[dict[str, object]]:
    return [
        {"metric": metric, "name": key or "(blank)", "value": value, "detail": ""}
        for key, value in sorted(counter.items())
    ]


def build_audits(
    proposals: list[dict[str, str]],
    members: list[dict[str, str]],
    classified_rows: list[dict[str, str]],
    records: list[dict[str, object]],
) -> list[dict[str, object]]:
    proposal_ids = {row.get("proposal_id", "") for row in proposals}
    record_proposal_ids = {str(row.get("source_proposal_id", "")) for row in records}
    member_ids = {row.get("proposal_id", "") for row in members}
    classified_ids = {row.get("proposal_id", "") for row in classified_rows}

    audits: list[dict[str, object]] = []
    missing_records = sorted(proposal_ids - record_proposal_ids)
    extra_records = sorted(record_proposal_ids - proposal_ids)
    audits.append(
        {
            "check": "grouping_record_covers_all_proposals",
            "status": "pass" if not missing_records and not extra_records else "fail",
            "detail": f"missing={len(missing_records)} extra={len(extra_records)} examples={'; '.join((missing_records + extra_records)[:10])}",
        }
    )
    missing_members = sorted(proposal_ids - member_ids)
    audits.append(
        {
            "check": "source_proposals_have_member_rows",
            "status": "pass" if not missing_members else "fail",
            "detail": f"missing={len(missing_members)} examples={'; '.join(missing_members[:10])}",
        }
    )
    missing_classified = sorted(proposal_ids - classified_ids)
    audits.append(
        {
            "check": "source_proposals_have_classified_queue_rows",
            "status": "pass" if not missing_classified else "fail",
            "detail": f"missing={len(missing_classified)} examples={'; '.join(missing_classified[:10])}",
        }
    )
    bad_physical = [
        str(row.get("source_proposal_id", ""))
        for row in records
        if row.get("physical_merge_allowed") == "yes"
        and (
            row.get("grouping_layer") != "physical_pairwise"
            or row.get("pages_consecutive") != "yes"
            or int_value(row.get("n_pages")) != 2
        )
    ]
    audits.append(
        {
            "check": "physical_merges_are_pairwise_consecutive_only",
            "status": "pass" if not bad_physical else "fail",
            "detail": f"bad={len(bad_physical)} examples={'; '.join(bad_physical[:10])}",
        }
    )
    bad_logical = [
        str(row.get("source_proposal_id", ""))
        for row in records
        if row.get("logical_collection_only") == "yes" and row.get("physical_merge_allowed") == "yes"
    ]
    audits.append(
        {
            "check": "logical_records_never_physical_merge",
            "status": "pass" if not bad_logical else "fail",
            "detail": f"bad={len(bad_logical)} examples={'; '.join(bad_logical[:10])}",
        }
    )
    bad_missing_page = [
        str(row.get("source_proposal_id", ""))
        for row in records
        if row.get("grouping_layer") == "missing_page_bridge" and row.get("physical_merge_allowed") == "yes"
    ]
    audits.append(
        {
            "check": "missing_page_bridges_never_merge",
            "status": "pass" if not bad_missing_page else "fail",
            "detail": f"bad={len(bad_missing_page)} examples={'; '.join(bad_missing_page[:10])}",
        }
    )
    bad_missing_ocr = [
        str(row.get("source_proposal_id", ""))
        for row in records
        if row.get("grouping_layer") == "wait_for_ocr" and row.get("physical_merge_allowed") == "yes"
    ]
    audits.append(
        {
            "check": "missing_ocr_edges_never_merge",
            "status": "pass" if not bad_missing_ocr else "fail",
            "detail": f"bad={len(bad_missing_ocr)} examples={'; '.join(bad_missing_ocr[:10])}",
        }
    )
    bad_conflicts = [
        str(row.get("source_proposal_id", ""))
        for row in records
        if row.get("grouping_layer") == "conflict_keep_separate" and row.get("physical_merge_allowed") == "yes"
    ]
    audits.append(
        {
            "check": "conflict_edges_never_merge",
            "status": "pass" if not bad_conflicts else "fail",
            "detail": f"bad={len(bad_conflicts)} examples={'; '.join(bad_conflicts[:10])}",
        }
    )
    oversized_site_ready = [
        str(row.get("source_proposal_id", ""))
        for row in records
        if row.get("physical_merge_allowed") == "yes" and int_value(row.get("n_pages")) > 2
    ]
    audits.append(
        {
            "check": "site_ready_physical_not_oversized",
            "status": "pass" if not oversized_site_ready else "fail",
            "detail": f"bad={len(oversized_site_ready)} examples={'; '.join(oversized_site_ready[:10])}",
        }
    )
    return audits


def workbook_integrity_audit(workbook_path: Path) -> dict[str, object]:
    try:
        with zipfile.ZipFile(workbook_path) as zf:
            bad_member = zf.testzip()
    except Exception as exc:  # pragma: no cover
        bad_member = f"{type(exc).__name__}: {exc}"
    return {
        "check": "grouping_record_workbook_zip_integrity",
        "status": "pass" if bad_member is None else "fail",
        "detail": str(bad_member or "ok"),
    }


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
                fmt = wrap_fmt if field in {"requirement", "why_it_exists", "failure_mode", "current_action", "reason", "review_flags", "promotion_condition", "title_summary"} else None
                ws.write(row_idx, col, row.get(field, ""), fmt)
        ws.freeze_panes(1, 0)
        if fields:
            ws.autofilter(0, 0, max(0, len(rows)), len(fields) - 1)
        for col, field in enumerate(fields):
            ws.set_column(col, col, min(max(len(field) + 2, 12), 60))
    workbook.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-dir", required=True)
    args = parser.parse_args()

    release_dir = Path(args.release_dir).expanduser().resolve()
    proposals = read_csv(release_dir / "assembly_grouping_proposals.csv")
    members = read_csv(release_dir / "assembly_grouping_proposal_members.csv")
    classified = read_csv(release_dir / "assembly_priority_classified_work_queue.csv")
    if not proposals:
        raise SystemExit(f"missing assembly_grouping_proposals.csv in {release_dir}")

    records = build_grouping_record(proposals, members, classified)
    policies = build_rule_policy()
    summary = build_summary(records, policies)
    workbook_path = release_dir / "assembly_grouping_record_workbook.xlsx"
    audits = build_audits(proposals, members, classified, records)

    write_csv(release_dir / "assembly_grouping_record.csv", records, GROUPING_RECORD_FIELDS)
    write_csv(release_dir / "assembly_rule_policy.csv", policies, RULE_POLICY_FIELDS)
    write_csv(release_dir / "assembly_grouping_record_summary.csv", summary, SUMMARY_FIELDS)
    write_workbook(
        workbook_path,
        [
            ("Grouping Record", records, GROUPING_RECORD_FIELDS),
            ("Rule Policy", policies, RULE_POLICY_FIELDS),
            ("Summary", summary, SUMMARY_FIELDS),
            ("Audit", audits, AUDIT_FIELDS),
        ],
    )
    audits.append(workbook_integrity_audit(workbook_path))
    write_csv(release_dir / "assembly_grouping_record_audit.csv", audits, AUDIT_FIELDS)
    write_workbook(
        workbook_path,
        [
            ("Grouping Record", records, GROUPING_RECORD_FIELDS),
            ("Rule Policy", policies, RULE_POLICY_FIELDS),
            ("Summary", summary, SUMMARY_FIELDS),
            ("Audit", audits, AUDIT_FIELDS),
        ],
    )
    print(workbook_path)
    print(json.dumps({"records": len(records), "policies": len(policies), "audit_failures": sum(1 for row in audits if row.get("status") == "fail")}, sort_keys=True))


if __name__ == "__main__":
    main()
