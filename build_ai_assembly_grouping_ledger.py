#!/usr/bin/env python3
"""Build an AI-facing ledger of table groupings to assemble later.

This does not OCR and does not assemble matrices. It packages the current
first-principles grouping record into a clear, ordered ledger for downstream AI
assembly/review: physical groups to assemble, physical groups needing manual
confirmation, logical-only sibling groups, and deferred adjacent retests.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import xlsxwriter


ROOT = Path(__file__).resolve().parent
RELEASE_GLOB = "pdf_recovery_incoming_2026-06-16_first_principles_assembly_release_snapshot_*"

GROUP_FIELDS = [
    "ai_group_rank",
    "ai_group_id",
    "source_grouping_id",
    "source_proposal_id",
    "assembly_scope",
    "grouping_decision",
    "include_for_ai",
    "confidence_tier",
    "physical_merge_allowed_now",
    "logical_collection_only",
    "pages_consecutive",
    "source_key",
    "source_pdf_id",
    "page_range",
    "component_pages",
    "n_pages",
    "core_scope",
    "table_number_key",
    "topic_variant_key",
    "religion_terms_detected",
    "row_label_dependency_signal",
    "right_page_no_row_labels_signal",
    "pair_direction_signal",
    "pair_direction_reason",
    "column_order_evidence",
    "zero_check_status",
    "title_summary",
    "reason",
    "promotion_condition",
    "review_flags",
    "matrix_csv",
    "member_pages_ordered",
    "member_images_ordered",
    "member_ocr_csvs_ordered",
    "member_ocr_jsons_ordered",
    "ai_instruction",
]

MEMBER_FIELDS = [
    "ai_group_id",
    "source_grouping_id",
    "source_proposal_id",
    "member_rank",
    "member_role",
    "source_key",
    "source_pdf_id",
    "pdf_page",
    "expanded_pages",
    "image_path",
    "source_csv",
    "source_json",
    "title",
]

SUMMARY_FIELDS = ["metric", "name", "value"]
AUDIT_FIELDS = ["check", "status", "detail"]
MAX_PHYSICAL_PAGES = 6


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


def latest_release_dir() -> Path:
    candidates = [
        path
        for path in (ROOT / "export_final").glob(RELEASE_GLOB)
        if (path / "assembly_grouping_record.csv").exists()
        and (path / "assembly_grouping_proposal_members.csv").exists()
    ]
    if not candidates:
        raise SystemExit("No assembly release snapshot with grouping record found.")
    return max(candidates, key=lambda p: (p.stat().st_mtime, p.name))


def int_value(value: object, default: int = 0) -> int:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return default


def semi(values: Iterable[object]) -> str:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    return "; ".join(out)


def decision_for_record(row: dict[str, str]) -> tuple[str, str, str, str, str, str]:
    layer = row.get("grouping_layer", "")
    status = row.get("grouping_status", "")
    policy = row.get("site_use_policy", "")
    consecutive = row.get("pages_consecutive", "")
    direction = row.get("pair_direction_signal", "")

    if layer == "physical_pairwise" and status in {"physical_structure_ready", "physical_structure_ready_value_review"}:
        return (
            "physical",
            "assemble_now_physical",
            "yes",
            "confirmed_consecutive_physical",
            "yes",
            "Assemble these consecutive PDF pages as one physical table. Preserve the member page order exactly; do not add non-member pages.",
        )
    if layer == "physical_pairwise" and status == "physical_structure_review_required":
        return (
            "physical",
            "review_then_assemble_physical",
            "review_first",
            "confirmed_edge_but_review_required",
            "yes",
            "Review the scan pair first, then assemble only if it is visibly one physical table continuation.",
        )
    if layer == "manual_physical_review":
        tier = "likely_civil_status_continuation" if direction == "civil_status_begin_to_late_continuation" else "manual_physical_candidate"
        return (
            "physical",
            "review_then_assemble_physical",
            "review_first",
            tier,
            "no",
            "Do not auto-merge. Review the images; assemble only if these adjacent pages are the same physical table and the second page continues the columns/rows of the first.",
        )
    if layer == "logical_only" and policy == "use_now_logical_navigation_only":
        return (
            "logical",
            "group_now_logical_only",
            "logical_only",
            "confirmed_logical_sibling",
            "no",
            "Keep these as separate physical pages/tables, but group them as logical siblings for navigation and interpretation.",
        )
    if layer == "logical_only":
        return (
            "logical",
            "review_logical_only",
            "review_first",
            "logical_candidate_review",
            "no",
            "Review as logical siblings only. Never physically merge unless a separate physical edge proves contiguity and continuation.",
        )
    if layer == "missing_page_bridge":
        return (
            "deferred",
            "recover_missing_pages_then_retest",
            "deferred",
            "likely_after_missing_pages",
            "no",
            "Do not assemble now. Recover the missing intermediate PDF pages, OCR them elsewhere, then retest adjacent pairwise edges.",
        )
    if layer == "wait_for_ocr":
        return (
            "deferred",
            "ocr_available_page_then_retest",
            "deferred",
            "missing_ocr_artifact",
            "no",
            "Do not assemble now. Wait for the missing OCR artifact, then retest the adjacent edge.",
        )
    if layer == "conflict_keep_separate":
        return (
            "negative",
            "keep_separate",
            "no",
            "structural_or_semantic_conflict",
            "no",
            "Record only: do not assemble these pages as one physical group under the current evidence.",
        )
    return (
        "review",
        "review_before_grouping",
        "review_first",
        "uncertain_grouping",
        "no",
        "Review before using this as either a physical or logical grouping.",
    )


def rank_key(row: dict[str, object]) -> tuple[int, int, str, int, str]:
    order = {
        "assemble_now_physical": 1,
        "review_then_assemble_physical": 2,
        "group_now_logical_only": 3,
        "review_logical_only": 4,
        "recover_missing_pages_then_retest": 5,
        "ocr_available_page_then_retest": 6,
        "review_before_grouping": 7,
        "keep_separate": 9,
    }
    core_rank = 0 if row.get("core_scope") == "core_table_vii" else 1
    return (
        order.get(str(row.get("grouping_decision", "")), 99),
        core_rank,
        str(row.get("source_key", "")),
        int_value(row.get("pdf_page_start"), 999999),
        str(row.get("source_grouping_id", "")),
    )


def build_ledgers(release_dir: Path) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    records = read_csv(release_dir / "assembly_grouping_record.csv")
    members = read_csv(release_dir / "assembly_grouping_proposal_members.csv")
    members_by_proposal: dict[str, list[dict[str, str]]] = defaultdict(list)
    for member in members:
        members_by_proposal[member.get("proposal_id", "")].append(member)
    for rows in members_by_proposal.values():
        rows.sort(key=lambda r: (int_value(r.get("member_rank"), 9999), int_value(r.get("pdf_page"), 999999)))

    groups: list[dict[str, object]] = []
    member_rows: list[dict[str, object]] = []
    for record in records:
        scope, decision, include, tier, physical_now, instruction = decision_for_record(record)
        # The primary ledger is intentionally a grouping record, not a negative-edge dump.
        if decision == "keep_separate":
            continue
        proposal_id = record.get("source_proposal_id", "")
        group_members = members_by_proposal.get(proposal_id, [])
        ai_group_id = f"ai::{record.get('grouping_id', '')}"
        groups.append(
            {
                "ai_group_id": ai_group_id,
                "source_grouping_id": record.get("grouping_id", ""),
                "source_proposal_id": proposal_id,
                "assembly_scope": scope,
                "grouping_decision": decision,
                "include_for_ai": include,
                "confidence_tier": tier,
                "physical_merge_allowed_now": physical_now,
                "logical_collection_only": record.get("logical_collection_only", ""),
                "pages_consecutive": consecutive_or_no(record),
                "source_key": record.get("source_key", ""),
                "source_pdf_id": record.get("source_pdf_id", ""),
                "page_range": record.get("page_range", ""),
                "component_pages": record.get("component_pages", ""),
                "n_pages": record.get("n_pages", ""),
                "core_scope": record.get("core_scope", ""),
                "table_number_key": record.get("table_number_key", ""),
                "topic_variant_key": record.get("topic_variant_key", ""),
                "religion_terms_detected": record.get("religion_terms_detected", ""),
                "row_label_dependency_signal": record.get("row_label_dependency_signal", ""),
                "right_page_no_row_labels_signal": record.get("right_page_no_row_labels_signal", ""),
                "pair_direction_signal": record.get("pair_direction_signal", ""),
                "pair_direction_reason": record.get("pair_direction_reason", ""),
                "column_order_evidence": record.get("column_order_evidence", ""),
                "zero_check_status": record.get("zero_check_status", ""),
                "title_summary": record.get("title_summary", ""),
                "reason": record.get("reason", ""),
                "promotion_condition": record.get("promotion_condition", ""),
                "review_flags": record.get("review_flags", ""),
                "matrix_csv": record.get("matrix_csv", ""),
                "member_pages_ordered": semi(member.get("pdf_page", "") for member in group_members),
                "member_images_ordered": semi(member.get("image_path", "") for member in group_members),
                "member_ocr_csvs_ordered": semi(member.get("source_csv", "") for member in group_members),
                "member_ocr_jsons_ordered": semi(member.get("source_json", "") for member in group_members),
                "ai_instruction": instruction,
                "pdf_page_start": record.get("pdf_page_start", ""),
            }
        )
        for member in group_members:
            member_rows.append(
                {
                    "ai_group_id": ai_group_id,
                    "source_grouping_id": record.get("grouping_id", ""),
                    "source_proposal_id": proposal_id,
                    "member_rank": member.get("member_rank", ""),
                    "member_role": member.get("member_role", ""),
                    "source_key": member.get("source_key", ""),
                    "source_pdf_id": member.get("source_pdf_id", ""),
                    "pdf_page": member.get("pdf_page", ""),
                    "expanded_pages": member.get("expanded_pages", ""),
                    "image_path": member.get("image_path", ""),
                    "source_csv": member.get("source_csv", ""),
                    "source_json": member.get("source_json", ""),
                    "title": member.get("title", ""),
                }
            )
    groups.sort(key=rank_key)
    for idx, row in enumerate(groups, start=1):
        row["ai_group_rank"] = idx

    summary: list[dict[str, object]] = [
        {"metric": "path", "name": "release_dir", "value": str(release_dir)},
        {"metric": "counts", "name": "group_rows", "value": len(groups)},
        {"metric": "counts", "name": "member_rows", "value": len(member_rows)},
    ]
    for name, counter in [
        ("grouping_decision", Counter(str(row.get("grouping_decision", "")) for row in groups)),
        ("assembly_scope", Counter(str(row.get("assembly_scope", "")) for row in groups)),
        ("confidence_tier", Counter(str(row.get("confidence_tier", "")) for row in groups)),
        ("core_scope", Counter(str(row.get("core_scope", "")) for row in groups)),
    ]:
        for key, count in sorted(counter.items()):
            summary.append({"metric": name, "name": key, "value": count})
    return groups, member_rows, summary


def consecutive_or_no(record: dict[str, str]) -> str:
    if record.get("pages_consecutive"):
        return record.get("pages_consecutive", "")
    return "yes" if record.get("page_gap") in {"", "0"} else "no"


def parse_page_list(value: object) -> list[int]:
    pages: list[int] = []
    for part in str(value or "").replace(",", ";").split(";"):
        part = part.strip()
        if not part:
            continue
        try:
            pages.append(int(part))
        except ValueError:
            continue
    return pages


def is_consecutive_pages(pages: Iterable[int]) -> bool:
    unique = sorted(set(pages))
    return bool(unique) and unique == list(range(unique[0], unique[-1] + 1))


def build_audits(groups: list[dict[str, object]], members: list[dict[str, object]]) -> list[dict[str, object]]:
    members_by_group: dict[str, list[dict[str, object]]] = defaultdict(list)
    for member in members:
        members_by_group[str(member.get("ai_group_id", ""))].append(member)

    audits: list[dict[str, object]] = []
    physical_groups = [row for row in groups if row.get("assembly_scope") == "physical"]
    physical_bad_contiguity = [
        row
        for row in physical_groups
        if row.get("pages_consecutive") != "yes"
        or not is_consecutive_pages(parse_page_list(row.get("member_pages_ordered") or row.get("component_pages")))
    ]
    audits.append(
        {
            "check": "physical_groups_are_consecutive",
            "status": "pass" if not physical_bad_contiguity else "fail",
            "detail": f"bad={len(physical_bad_contiguity)} total={len(physical_groups)}",
        }
    )

    immediate_physical_bad_policy = [
        row
        for row in groups
        if row.get("grouping_decision") == "assemble_now_physical"
        and row.get("physical_merge_allowed_now") != "yes"
    ]
    audits.append(
        {
            "check": "assemble_now_physical_has_physical_merge_permission",
            "status": "pass" if not immediate_physical_bad_policy else "fail",
            "detail": f"bad={len(immediate_physical_bad_policy)}",
        }
    )

    oversized_physical = [
        row
        for row in physical_groups
        if int_value(row.get("n_pages"), 0) > MAX_PHYSICAL_PAGES
    ]
    audits.append(
        {
            "check": "physical_groups_are_bounded_not_overlarge",
            "status": "pass" if not oversized_physical else "fail",
            "detail": f"bad={len(oversized_physical)} max_allowed_pages={MAX_PHYSICAL_PAGES}",
        }
    )

    logical_physical_leaks = [
        row
        for row in groups
        if row.get("assembly_scope") == "logical"
        and row.get("physical_merge_allowed_now") == "yes"
    ]
    audits.append(
        {
            "check": "logical_groups_never_allow_physical_merge",
            "status": "pass" if not logical_physical_leaks else "fail",
            "detail": f"bad={len(logical_physical_leaks)}",
        }
    )

    deferred_now_leaks = [
        row
        for row in groups
        if row.get("include_for_ai") == "deferred"
        and row.get("grouping_decision") in {"assemble_now_physical", "group_now_logical_only"}
    ]
    audits.append(
        {
            "check": "deferred_groups_are_not_marked_assemble_now",
            "status": "pass" if not deferred_now_leaks else "fail",
            "detail": f"bad={len(deferred_now_leaks)}",
        }
    )

    member_order_bad: list[dict[str, object]] = []
    for group in physical_groups:
        group_members = members_by_group.get(str(group.get("ai_group_id", "")), [])
        ordered_pages = [int_value(member.get("pdf_page"), -1) for member in group_members]
        ordered_pages = [page for page in ordered_pages if page >= 0]
        if ordered_pages and ordered_pages != sorted(ordered_pages):
            member_order_bad.append(group)
    audits.append(
        {
            "check": "physical_member_order_is_pdf_order",
            "status": "pass" if not member_order_bad else "fail",
            "detail": f"bad={len(member_order_bad)} total={len(physical_groups)}",
        }
    )

    groups_without_members = [
        row
        for row in groups
        if row.get("grouping_decision") in {"assemble_now_physical", "review_then_assemble_physical", "group_now_logical_only"}
        and not members_by_group.get(str(row.get("ai_group_id", "")))
    ]
    audits.append(
        {
            "check": "usable_groups_have_member_rows",
            "status": "pass" if not groups_without_members else "fail",
            "detail": f"bad={len(groups_without_members)}",
        }
    )
    return audits


def with_audit_summary(summary: list[dict[str, object]], audits: list[dict[str, object]]) -> list[dict[str, object]]:
    failures = sum(1 for row in audits if row.get("status") == "fail")
    warnings = sum(1 for row in audits if row.get("status") == "warn")
    return summary + [
        {"metric": "audit", "name": "failures", "value": failures},
        {"metric": "audit", "name": "warnings", "value": warnings},
    ]


def write_workbook(path: Path, summary: list[dict[str, object]], audits: list[dict[str, object]], groups: list[dict[str, object]], members: list[dict[str, object]]) -> None:
    workbook = xlsxwriter.Workbook(path)
    header_fmt = workbook.add_format({"bold": True, "bg_color": "#D9EAF7", "border": 1})
    wrap_fmt = workbook.add_format({"text_wrap": True, "valign": "top"})
    best_now = [
        row
        for row in groups
        if row.get("grouping_decision") in {"assemble_now_physical", "review_then_assemble_physical", "group_now_logical_only"}
    ]
    deferred = [row for row in groups if row.get("include_for_ai") == "deferred"]
    logical_review = [row for row in groups if row.get("grouping_decision") == "review_logical_only"]
    sheets = [
        ("Summary", summary, SUMMARY_FIELDS),
        ("Audit", audits, AUDIT_FIELDS),
        ("Best Now", best_now, GROUP_FIELDS),
        ("AI Groupings", groups, GROUP_FIELDS),
        ("Deferred", deferred, GROUP_FIELDS),
        ("Logical Review", logical_review, GROUP_FIELDS),
        ("Members", members, MEMBER_FIELDS),
    ]
    for name, rows, fields in sheets:
        ws = workbook.add_worksheet(name[:31])
        for col, field in enumerate(fields):
            ws.write(0, col, field, header_fmt)
        for row_idx, row in enumerate(rows, start=1):
            for col, field in enumerate(fields):
                ws.write(row_idx, col, row.get(field, ""), wrap_fmt if field in {"title_summary", "reason", "ai_instruction", "image_path"} else None)
        ws.freeze_panes(1, 0)
        ws.autofilter(0, 0, max(len(rows), 1), len(fields) - 1)
        for col, field in enumerate(fields):
            width = 18
            if field in {"title_summary", "reason", "promotion_condition", "review_flags", "ai_instruction"}:
                width = 45
            elif field in {"member_images_ordered", "member_ocr_csvs_ordered", "member_ocr_jsons_ordered", "image_path", "source_csv", "source_json"}:
                width = 55
            ws.set_column(col, col, width)
    workbook.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    release_dir = (args.release_dir or latest_release_dir()).resolve()
    out_dir = (args.out_dir or (release_dir / "ai_assembly_grouping_ledger")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    groups, members, summary = build_ledgers(release_dir)
    audits = build_audits(groups, members)
    summary = with_audit_summary(summary, audits)
    best_now = [
        row
        for row in groups
        if row.get("grouping_decision") in {"assemble_now_physical", "review_then_assemble_physical", "group_now_logical_only"}
    ]
    deferred = [row for row in groups if row.get("include_for_ai") == "deferred"]
    logical_review = [row for row in groups if row.get("grouping_decision") == "review_logical_only"]
    write_csv(out_dir / "ai_assembly_groupings.csv", groups, GROUP_FIELDS)
    write_csv(out_dir / "ai_assembly_groupings_best_now.csv", best_now, GROUP_FIELDS)
    write_csv(out_dir / "ai_assembly_groupings_deferred.csv", deferred, GROUP_FIELDS)
    write_csv(out_dir / "ai_assembly_groupings_logical_review.csv", logical_review, GROUP_FIELDS)
    write_csv(out_dir / "ai_assembly_grouping_members.csv", members, MEMBER_FIELDS)
    write_csv(out_dir / "ai_assembly_grouping_audit.csv", audits, AUDIT_FIELDS)
    write_csv(out_dir / "ai_assembly_grouping_summary.csv", summary, SUMMARY_FIELDS)
    (out_dir / "ai_assembly_grouping_summary.json").write_text(
        json.dumps(
            {
                "counts": {str(row["name"]): row["value"] for row in summary if row["metric"] == "counts"},
                "audit": {str(row["name"]): row["value"] for row in summary if row["metric"] == "audit"},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_workbook(out_dir / "ai_assembly_grouping_ledger.xlsx", summary, audits, groups, members)
    failures = sum(1 for row in audits if row.get("status") == "fail")
    print(json.dumps({"out_dir": str(out_dir), "groups": len(groups), "members": len(members), "audit_failures": failures}, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
