#!/usr/bin/env python3
"""Build a unified first-principles assembly control package.

This packages the current assembly evidence into one operational workbook:
confirmed physical components, row-level previews with column-order evidence,
logical/religion family collections, OCR/retest queues, and review queues.

It does not perform OCR and does not promote any review-only candidate.
"""

from __future__ import annotations

import argparse
import csv
import json
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
DEFAULT_INTAKE_DIR = ROOT / "export_final/pdf_recovery_incoming_2026-06-16_assembly_ocr_intake_v5_join_keys"
DEFAULT_OUT_DIR = ROOT / "export_final/pdf_recovery_incoming_2026-06-16_first_principles_assembly_control_v33"

MANIFEST_FIELDS = [
    "control_rank",
    "control_id",
    "record_type",
    "site_readiness",
    "action_bucket",
    "physical_merge_allowed",
    "logical_collection_only",
    "safe_to_publish_values",
    "review_required",
    "source_pdf",
    "source_pdf_id",
    "v5_source_key",
    "year",
    "province",
    "page_range",
    "component_pages",
    "n_pages",
    "member_count",
    "religion_family_signal",
    "religion_terms_detected",
    "title_table_numbers",
    "primary_title",
    "source_group_id",
    "source_family_id",
    "evidence_files",
    "reason",
    "review_flags",
    "recommended_action",
]

ROW_PREVIEW_FIELDS = [
    "control_component_id",
    "component_rank",
    "component_page_range",
    "v26_row_id",
    "v25_component_id",
    "block_index",
    "row_index",
    "row_label",
    "page_pair",
    "v5_source_key",
    "source_pdf",
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
    "left_title",
    "right_title",
    "left_image_path",
    "right_image_path",
]

COLUMN_CHECK_FIELDS = [
    "control_component_id",
    "component_rank",
    "component_page_range",
    "v25_component_id",
    "block_index",
    "column_order_status",
    "column_order_flags",
    "row_count",
    "page_pair",
    "v5_source_key",
    "source_pdf",
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

QUEUE_FIELDS = [
    "queue_rank",
    "queue_type",
    "action_bucket",
    "target_scope",
    "source_pdf",
    "source_pdf_id",
    "v5_source_key",
    "page_pair",
    "pdf_page",
    "priority_bucket",
    "frontier_status",
    "candidate_strength",
    "matched_render_count",
    "best_render_path",
    "matched_artifact_count",
    "best_artifact_path",
    "unlock_status",
    "recommended_next_step",
    "reason",
    "review_flags",
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


def semi(values: Iterable[object]) -> str:
    return "; ".join(str(value) for value in values if value not in {None, ""})


def split_semicolon(value: object) -> list[str]:
    return [part.strip() for part in str(value or "").split(";") if part.strip()]


def int_or_none(value: object) -> int | None:
    if value in {None, ""}:
        return None
    try:
        return int(str(value))
    except ValueError:
        return None


def build_component_maps(component_rows: list[dict[str, str]], frontier_rows: list[dict[str, str]]) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, str]]]:
    by_control: dict[str, dict[str, str]] = {}
    v25_to_component: dict[str, dict[str, str]] = {}
    frontier_by_attempt = {row.get("v26_attempt_id", ""): row for row in frontier_rows if row.get("v26_attempt_id")}
    for component in component_rows:
        control_id = f"physical::{component.get('component_id', '')}"
        by_control[control_id] = component
        for attempt_id in split_semicolon(component.get("edge_attempt_ids", "")):
            frontier = frontier_by_attempt.get(attempt_id)
            if frontier and frontier.get("v25_component_id"):
                v25_to_component[frontier["v25_component_id"]] = component
    return by_control, v25_to_component


def build_manifest(
    component_rows: list[dict[str, str]],
    logical_rows: list[dict[str, str]],
    paths: dict[str, Path],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for component in component_rows:
        control_id = f"physical::{component.get('component_id', '')}"
        site_readiness = "site_ready_structure_only"
        if component.get("safe_to_publish_values") == "yes":
            site_readiness = "site_ready_publishable_values"
        rows.append(
            {
                "control_id": control_id,
                "record_type": "physical_component",
                "site_readiness": site_readiness,
                "action_bucket": component.get("action_bucket", ""),
                "physical_merge_allowed": component.get("safe_to_physical_merge", "yes"),
                "logical_collection_only": "no",
                "safe_to_publish_values": component.get("safe_to_publish_values", ""),
                "review_required": component.get("value_review_required", ""),
                "source_pdf": component.get("source_pdf", ""),
                "source_pdf_id": component.get("source_pdf_id", ""),
                "v5_source_key": component.get("v5_source_key", ""),
                "year": component.get("year", ""),
                "province": component.get("province", ""),
                "page_range": component.get("page_range", ""),
                "component_pages": component.get("component_pages", ""),
                "n_pages": component.get("n_pages", ""),
                "member_count": component.get("n_pages", ""),
                "source_group_id": component.get("component_id", ""),
                "evidence_files": semi([output_path(paths["components_workbook"]), output_path(paths["component_csv"])]),
                "reason": component.get("component_reason", ""),
                "review_flags": component.get("review_flags", ""),
                "recommended_action": component.get("recommended_action", ""),
            }
        )
    for family in logical_rows:
        control_id = f"logical::{family.get('family_id', '')}"
        site_readiness = "logical_review_only"
        if family.get("frontier_status") == "confirmed_logical_collection" and family.get("action_bucket") == "use_now_logical_family_collection":
            site_readiness = "site_ready_logical_collection"
        rows.append(
            {
                "control_id": control_id,
                "record_type": "logical_family_collection",
                "site_readiness": site_readiness,
                "action_bucket": family.get("action_bucket", ""),
                "physical_merge_allowed": "no",
                "logical_collection_only": "yes",
                "safe_to_publish_values": "not_applicable",
                "review_required": "yes" if "review" in str(family.get("action_bucket", "")) else "no",
                "source_pdf": family.get("source_pdf", ""),
                "source_pdf_id": family.get("source_pdf_id", ""),
                "v5_source_key": family.get("v5_source_key", ""),
                "page_range": family.get("page_ranges", ""),
                "component_pages": family.get("member_pages", ""),
                "n_pages": family.get("n_physical_pages", ""),
                "member_count": family.get("member_count", ""),
                "religion_family_signal": family.get("religion_family_signal", ""),
                "religion_terms_detected": family.get("religion_terms_detected", ""),
                "title_table_numbers": family.get("title_table_numbers", ""),
                "primary_title": family.get("primary_title", ""),
                "source_group_id": family.get("assembly_group_id", ""),
                "source_family_id": family.get("family_id", ""),
                "evidence_files": semi([output_path(paths["logical_workbook"]), output_path(paths["logical_csv"])]),
                "reason": family.get("decision_reason", ""),
                "review_flags": family.get("review_flags", ""),
                "recommended_action": family.get("recommended_action", ""),
            }
        )
    rows.sort(key=lambda row: (str(row.get("record_type")), str(row.get("site_readiness")), str(row.get("v5_source_key")), str(row.get("page_range"))))
    for idx, row in enumerate(rows, start=1):
        row["control_rank"] = idx
    return rows


def build_row_previews(
    assembled_preview_rows: list[dict[str, str]],
    v25_to_component: dict[str, dict[str, str]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for preview in assembled_preview_rows:
        component = v25_to_component.get(preview.get("v25_component_id", ""))
        if not component:
            continue
        rows.append(
            {
                "control_component_id": f"physical::{component.get('component_id', '')}",
                "component_rank": component.get("component_rank", ""),
                "component_page_range": component.get("page_range", ""),
                **preview,
            }
        )
    rows.sort(
        key=lambda row: (
            int_or_none(row.get("component_rank")) or 10**9,
            int_or_none(row.get("block_index")) or 10**9,
            int_or_none(row.get("row_index")) or 10**9,
        )
    )
    return rows


def build_column_checks(
    contract_rows: list[dict[str, str]],
    v25_to_component: dict[str, dict[str, str]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for contract in contract_rows:
        component = v25_to_component.get(contract.get("v25_component_id", ""))
        if not component:
            continue
        rows.append(
            {
                "control_component_id": f"physical::{component.get('component_id', '')}",
                "component_rank": component.get("component_rank", ""),
                "component_page_range": component.get("page_range", ""),
                **contract,
            }
        )
    rows.sort(key=lambda row: (int_or_none(row.get("component_rank")) or 10**9, int_or_none(row.get("block_index")) or 10**9))
    return rows


def build_action_queues(
    intake_rows: list[dict[str, str]],
    physical_frontier_rows: list[dict[str, str]],
    row_label_rows: list[dict[str, str]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for intake in intake_rows:
        if intake.get("unlock_status") == "render_available_no_ocr_artifact":
            queue_type = "ocr_existing_render"
            action = "ocr_existing_render_then_retest"
        elif intake.get("unlock_status") == "still_blocked_no_local_artifact":
            queue_type = "obtain_render_or_artifact"
            action = "obtain_render_or_artifact_then_retest"
        else:
            continue
        rows.append(
            {
                "queue_type": queue_type,
                "action_bucket": action,
                "target_scope": intake.get("target_scope", ""),
                "source_pdf": intake.get("source_pdf", ""),
                "source_pdf_id": intake.get("source_pdf_id", ""),
                "v5_source_key": intake.get("v5_source_key", ""),
                "page_pair": intake.get("page_pair", ""),
                "pdf_page": intake.get("pdf_page", ""),
                "priority_bucket": intake.get("priority_bucket", ""),
                "matched_render_count": intake.get("matched_render_count", ""),
                "best_render_path": intake.get("best_render_path", ""),
                "matched_artifact_count": intake.get("matched_artifact_count", ""),
                "best_artifact_path": intake.get("best_artifact_path", ""),
                "unlock_status": intake.get("unlock_status", ""),
                "recommended_next_step": intake.get("recommended_next_step", ""),
                "reason": intake.get("blocking_context", ""),
                "review_flags": intake.get("grouping_review_flags", ""),
            }
        )
    for row in physical_frontier_rows:
        if row.get("action_bucket") not in {"manual_review_candidate_pass_before_promoting", "manual_review_partial_do_not_merge_full_pair", "do_not_merge_unless_constraints_are_repaired"}:
            continue
        rows.append(
            {
                "queue_type": "manual_or_constraint_review",
                "action_bucket": row.get("action_bucket", ""),
                "source_pdf": row.get("source_pdf", ""),
                "source_pdf_id": row.get("source_pdf_id", ""),
                "v5_source_key": row.get("v5_source_key", ""),
                "page_pair": row.get("page_pair", ""),
                "frontier_status": row.get("frontier_status", ""),
                "reason": row.get("decision_reason", ""),
                "review_flags": row.get("review_flags", ""),
                "recommended_next_step": row.get("recommended_action", ""),
            }
        )
    for row in row_label_rows:
        if row.get("action_bucket") not in {"ocr_existing_render_then_retest_candidate", "obtain_render_or_artifact_then_retest_candidate"}:
            continue
        rows.append(
            {
                "queue_type": "row_label_candidate_retest",
                "action_bucket": row.get("action_bucket", ""),
                "source_pdf": row.get("source_pdf", ""),
                "source_pdf_id": row.get("source_pdf_id", ""),
                "v5_source_key": row.get("v5_source_key", ""),
                "page_pair": row.get("page_pair", ""),
                "frontier_status": row.get("frontier_status", ""),
                "candidate_strength": row.get("candidate_strength", ""),
                "matched_render_count": row.get("ocr_targets_with_render", ""),
                "best_render_path": row.get("best_render_paths", ""),
                "matched_artifact_count": row.get("ocr_targets_with_artifact", ""),
                "unlock_status": row.get("ocr_unlock_statuses", ""),
                "recommended_next_step": row.get("recommended_action", ""),
                "reason": row.get("decision_reason", ""),
                "review_flags": row.get("review_flags", ""),
            }
        )
    priority = {
        "ocr_existing_render": 0,
        "row_label_candidate_retest": 1,
        "manual_or_constraint_review": 2,
        "obtain_render_or_artifact": 3,
    }
    rows.sort(key=lambda row: (priority.get(str(row.get("queue_type")), 9), str(row.get("target_scope")), str(row.get("v5_source_key")), str(row.get("page_pair")), str(row.get("pdf_page"))))
    for idx, row in enumerate(rows, start=1):
        row["queue_rank"] = idx
    return rows


def build_summary(
    manifest: list[dict[str, object]],
    row_previews: list[dict[str, object]],
    column_checks: list[dict[str, object]],
    queues: list[dict[str, object]],
    paths: dict[str, Path],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = [
        {"metric": "paths", "name": key, "count": output_path(path)}
        for key, path in sorted(paths.items())
        if path.exists() or key.endswith("_dir")
    ]
    rows.extend(
        [
            {"metric": "counts", "name": "manifest_rows", "count": len(manifest)},
            {"metric": "counts", "name": "row_preview_rows", "count": len(row_previews)},
            {"metric": "counts", "name": "column_check_rows", "count": len(column_checks)},
            {"metric": "counts", "name": "action_queue_rows", "count": len(queues)},
        ]
    )
    for metric, counter in [
        ("manifest_by_record_type", Counter(str(row.get("record_type", "")) for row in manifest)),
        ("manifest_by_site_readiness", Counter(str(row.get("site_readiness", "")) for row in manifest)),
        ("manifest_by_physical_merge_allowed", Counter(str(row.get("physical_merge_allowed", "")) for row in manifest)),
        ("manifest_by_logical_collection_only", Counter(str(row.get("logical_collection_only", "")) for row in manifest)),
        ("row_previews_by_column_order_status", Counter(str(row.get("column_order_status", "")) for row in row_previews)),
        ("column_checks_by_status", Counter(str(row.get("column_order_status", "")) for row in column_checks)),
        ("column_checks_left_then_right", Counter(str(row.get("left_then_right_order", "")) for row in column_checks)),
        ("queues_by_type", Counter(str(row.get("queue_type", "")) for row in queues)),
        ("queues_by_action", Counter(str(row.get("action_bucket", "")) for row in queues)),
    ]:
        for name, count in sorted(counter.items()):
            rows.append({"metric": metric, "name": name, "count": count})
    return rows


def audit_outputs(
    manifest: list[dict[str, object]],
    row_previews: list[dict[str, object]],
    column_checks: list[dict[str, object]],
    component_rows: list[dict[str, str]],
    logical_rows: list[dict[str, str]],
) -> list[dict[str, object]]:
    audits: list[dict[str, object]] = []
    physical_manifest = [row for row in manifest if row.get("record_type") == "physical_component"]
    logical_manifest = [row for row in manifest if row.get("record_type") == "logical_family_collection"]
    audits.append(
        {
            "check": "all_physical_components_in_manifest",
            "status": "pass" if len(physical_manifest) == len(component_rows) else "fail",
            "detail": f"manifest={len(physical_manifest)} components={len(component_rows)}",
        }
    )
    audits.append(
        {
            "check": "all_logical_families_in_manifest",
            "status": "pass" if len(logical_manifest) == len(logical_rows) else "fail",
            "detail": f"manifest={len(logical_manifest)} logical={len(logical_rows)}",
        }
    )
    bad_logical = [row for row in logical_manifest if row.get("physical_merge_allowed") == "yes"]
    audits.append(
        {
            "check": "logical_manifest_never_allows_physical_merge",
            "status": "pass" if not bad_logical else "fail",
            "detail": f"{len(bad_logical)} logical rows allow physical merge",
        }
    )
    bad_physical = [row for row in physical_manifest if row.get("physical_merge_allowed") != "yes"]
    audits.append(
        {
            "check": "physical_manifest_components_allow_physical_merge",
            "status": "pass" if not bad_physical else "fail",
            "detail": f"{len(bad_physical)} physical rows do not allow physical merge",
        }
    )
    bad_column_checks = [
        row for row in column_checks
        if row.get("column_order_status") != "pass"
        or row.get("left_then_right_order") != "yes"
        or row.get("adjacent_pdf_pages") != "yes"
        or row.get("consistent_column_order") != "yes"
    ]
    audits.append(
        {
            "check": "column_contracts_preserve_visual_left_then_right_order",
            "status": "pass" if not bad_column_checks else "fail",
            "detail": f"{len(bad_column_checks)} column contract rows failed",
        }
    )
    bad_previews = [row for row in row_previews if row.get("column_order_status") != "pass"]
    audits.append(
        {
            "check": "assembled_row_previews_have_passing_column_order",
            "status": "pass" if not bad_previews else "fail",
            "detail": f"{len(bad_previews)} preview rows failed column order",
        }
    )
    preview_components = {row.get("control_component_id") for row in row_previews}
    physical_control_ids = {row.get("control_id") for row in physical_manifest}
    missing_previews = physical_control_ids - preview_components
    audits.append(
        {
            "check": "each_physical_component_has_row_previews",
            "status": "pass" if not missing_previews else "fail",
            "detail": f"{len(missing_previews)} physical components lack row previews",
        }
    )
    return audits


def write_workbook(
    path: Path,
    summary: list[dict[str, object]],
    manifest: list[dict[str, object]],
    row_previews: list[dict[str, object]],
    column_checks: list[dict[str, object]],
    queues: list[dict[str, object]],
    audit: list[dict[str, object]],
) -> None:
    workbook = xlsxwriter.Workbook(path)
    header_fmt = workbook.add_format({"bold": True, "bg_color": "#E8EEF7", "border": 1})
    wrap_fmt = workbook.add_format({"text_wrap": True, "valign": "top"})
    sheets = [
        ("Summary", summary, SUMMARY_FIELDS),
        ("Control Manifest", manifest, MANIFEST_FIELDS),
        ("Physical Row Previews", row_previews, ROW_PREVIEW_FIELDS),
        ("Column Order Checks", column_checks, COLUMN_CHECK_FIELDS),
        ("Action Queues", queues, QUEUE_FIELDS),
        ("Audit", audit, AUDIT_FIELDS),
    ]
    for name, rows, fields in sheets:
        ws = workbook.add_worksheet(name[:31])
        for col, field in enumerate(fields):
            ws.write(0, col, field, header_fmt)
        for row_idx, row in enumerate(rows, start=1):
            for col, field in enumerate(fields):
                value = row.get(field, "")
                fmt = wrap_fmt if any(token in field for token in ["path", "title", "reason", "flags", "action", "order", "preview"]) else None
                ws.write(row_idx, col, value, fmt)
        ws.freeze_panes(1, 0)
        ws.autofilter(0, 0, max(len(rows), 1), max(len(fields) - 1, 0))
        for col, field in enumerate(fields):
            ws.set_column(col, col, min(max(len(field) + 2, 12), 64))
    workbook.close()


def validate_workbook(path: Path) -> str | None:
    try:
        with zipfile.ZipFile(path) as zf:
            return zf.testzip()
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"


def build_control(paths: dict[str, Path]) -> dict[str, object]:
    component_rows = read_csv(paths["component_csv"])
    physical_frontier_rows = read_csv(paths["physical_frontier_csv"])
    logical_rows = read_csv(paths["logical_csv"])
    row_preview_source = read_csv(paths["assembled_previews_csv"])
    column_contract_source = read_csv(paths["column_contract_csv"])
    intake_rows = read_csv(paths["intake_csv"])
    row_label_rows = read_csv(paths["row_label_frontier_csv"])

    _, v25_to_component = build_component_maps(component_rows, physical_frontier_rows)
    manifest = build_manifest(component_rows, logical_rows, paths)
    row_previews = build_row_previews(row_preview_source, v25_to_component)
    column_checks = build_column_checks(column_contract_source, v25_to_component)
    queues = build_action_queues(intake_rows, physical_frontier_rows, row_label_rows)
    summary = build_summary(manifest, row_previews, column_checks, queues, paths)
    audit = audit_outputs(manifest, row_previews, column_checks, component_rows, logical_rows)

    out_dir = paths["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "assembly_control_manifest.csv", manifest, MANIFEST_FIELDS)
    write_csv(out_dir / "physical_row_previews.csv", row_previews, ROW_PREVIEW_FIELDS)
    write_csv(out_dir / "column_order_checks.csv", column_checks, COLUMN_CHECK_FIELDS)
    write_csv(out_dir / "assembly_action_queues.csv", queues, QUEUE_FIELDS)
    write_csv(out_dir / "summary.csv", summary, SUMMARY_FIELDS)
    workbook_path = out_dir / "assembly_control_workbook.xlsx"
    workbook_bad_member = validate_workbook(workbook_path)
    audit.append(
        {
            "check": "assembly_control_workbook_zip_integrity",
            "status": "pass" if workbook_bad_member is None else "fail",
            "detail": workbook_bad_member or "",
        }
    )
    write_csv(out_dir / "audit.csv", audit, AUDIT_FIELDS)
    write_workbook(workbook_path, summary, manifest, row_previews, column_checks, queues, audit)
    workbook_bad_member = validate_workbook(workbook_path)
    audit[-1] = {
        "check": "assembly_control_workbook_zip_integrity",
        "status": "pass" if workbook_bad_member is None else "fail",
        "detail": workbook_bad_member or "",
    }
    write_csv(out_dir / "audit.csv", audit, AUDIT_FIELDS)
    write_workbook(workbook_path, summary, manifest, row_previews, column_checks, queues, audit)
    audit_failure_count = sum(1 for row in audit if row.get("status") != "pass")
    result = {
        "out_dir": output_path(out_dir),
        "manifest_rows": len(manifest),
        "physical_component_rows": sum(1 for row in manifest if row.get("record_type") == "physical_component"),
        "logical_family_rows": sum(1 for row in manifest if row.get("record_type") == "logical_family_collection"),
        "row_preview_rows": len(row_previews),
        "column_check_rows": len(column_checks),
        "action_queue_rows": len(queues),
        "manifest_by_site_readiness": dict(Counter(str(row.get("site_readiness", "")) for row in manifest)),
        "queues_by_type": dict(Counter(str(row.get("queue_type", "")) for row in queues)),
        "column_checks_by_status": dict(Counter(str(row.get("column_order_status", "")) for row in column_checks)),
        "audit_failure_count": audit_failure_count,
        "workbook_zip_bad_member": workbook_bad_member,
    }
    (out_dir / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    if audit_failure_count:
        raise SystemExit(f"control package audit failed: {audit_failure_count} failures")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assembly-dir", default=str(DEFAULT_ASSEMBLY_DIR))
    parser.add_argument("--frontier-dir", default=str(DEFAULT_FRONTIER_DIR))
    parser.add_argument("--component-dir", default=str(DEFAULT_COMPONENT_DIR))
    parser.add_argument("--logical-dir", default=str(DEFAULT_LOGICAL_DIR))
    parser.add_argument("--intake-dir", default=str(DEFAULT_INTAKE_DIR))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    args = parser.parse_args()

    assembly_dir = Path(args.assembly_dir).expanduser()
    frontier_dir = Path(args.frontier_dir).expanduser()
    component_dir = Path(args.component_dir).expanduser()
    logical_dir = Path(args.logical_dir).expanduser()
    intake_dir = Path(args.intake_dir).expanduser()
    out_dir = Path(args.out_dir).expanduser()
    dirs = [assembly_dir, frontier_dir, component_dir, logical_dir, intake_dir, out_dir]
    assembly_dir, frontier_dir, component_dir, logical_dir, intake_dir, out_dir = [
        path if path.is_absolute() else ROOT / path for path in dirs
    ]
    paths = {
        "assembly_dir": assembly_dir,
        "frontier_dir": frontier_dir,
        "component_dir": component_dir,
        "logical_dir": logical_dir,
        "intake_dir": intake_dir,
        "out_dir": out_dir,
        "components_workbook": component_dir / "assembly_components_workbook.xlsx",
        "component_csv": component_dir / "confirmed_physical_components.csv",
        "physical_frontier_csv": frontier_dir / "physical_assembly_frontier.csv",
        "row_label_frontier_csv": frontier_dir / "odd_unlabeled_row_label_frontier.csv",
        "logical_workbook": logical_dir / "logical_family_frontier_workbook.xlsx",
        "logical_csv": logical_dir / "logical_family_frontier.csv",
        "intake_csv": intake_dir / "ocr_target_unlock_status.csv",
        "assembled_previews_csv": assembly_dir / "assembled_row_previews.csv",
        "column_contract_csv": assembly_dir / "column_order_contract.csv",
    }
    result = build_control(paths)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
