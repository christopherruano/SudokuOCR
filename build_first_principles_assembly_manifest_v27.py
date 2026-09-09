#!/usr/bin/env python3
"""Build a scalable first-principles assembly manifest.

v25 gives strict physical seed components. v26 tests whether right-half pages
can actually inherit row labels and assemble into rows. This v27 layer turns
those into downstream-safe records:

* required physical groups: consecutive only, currently max two pages;
* logical collection groups: grouping-only, never a physical merge;
* missing-bridge candidates: likely related pages that cannot be assembled
  because one or more PDF pages are absent.

No OCR/API work is performed here.
"""

from __future__ import annotations

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
V25_DIR = ROOT / "export_final/pdf_recovery_incoming_2026-06-16_first_principles_seed_assembly_v25"
V26_DIR = ROOT / "export_final/pdf_recovery_incoming_2026-06-16_first_principles_row_assembly_v26"
OUT_DIR = ROOT / "export_final/pdf_recovery_incoming_2026-06-16_first_principles_assembly_manifest_v27"


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


def stable_id(prefix: str, *parts: object) -> str:
    payload = "||".join(str(part) for part in parts)
    return f"{prefix}_{hashlib.sha1(payload.encode('utf-8')).hexdigest()[:16]}"


def compact(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def clean_text(value: object) -> str:
    s = str(value or "").lower()
    s = s.replace("&", " and ")
    s = re.sub(r"[\u2010-\u2015]", "-", s)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def int_or_none(value: object) -> int | None:
    s = str(value or "").strip()
    return int(s) if re.fullmatch(r"-?\d+", s) else None


def uniq(values: Iterable[object], limit: int = 20) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        s = compact(value)
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    if len(out) > limit:
        return out[:limit] + [f"...(+{len(out) - limit})"]
    return out


def uniq_join(values: Iterable[object], limit: int = 20) -> str:
    return "; ".join(uniq(values, limit=limit))


def parse_range_piece(piece: str) -> tuple[int, int] | None:
    piece = piece.strip()
    if not piece:
        return None
    if re.fullmatch(r"\d+", piece):
        value = int(piece)
        return value, value
    match = re.fullmatch(r"(\d+)\s*-\s*(\d+)", piece)
    if match:
        start, end = int(match.group(1)), int(match.group(2))
        return (start, end) if start <= end else (end, start)
    return None


def parse_page_ranges(value: object) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    for piece in str(value or "").split(";"):
        parsed = parse_range_piece(piece)
        if parsed:
            ranges.append(parsed)
    return ranges


def ranges_are_consecutive(ranges: list[tuple[int, int]]) -> tuple[bool, str]:
    if not ranges:
        return False, "no_parseable_page_ranges"
    sorted_ranges = sorted(ranges)
    gaps: list[str] = []
    prev_end = sorted_ranges[0][1]
    for start, end in sorted_ranges[1:]:
        if start != prev_end + 1:
            if start > prev_end + 1:
                gaps.append(f"{prev_end + 1}-{start - 1}" if start - prev_end > 2 else str(prev_end + 1))
            else:
                gaps.append(f"overlap_or_out_of_order:{start}-{end}")
        prev_end = max(prev_end, end)
    return not gaps, "; ".join(gaps)


def total_range_pages(ranges: list[tuple[int, int]]) -> int:
    return sum(end - start + 1 for start, end in ranges)


def source_key(row: dict[str, str]) -> str:
    return row.get("source_pdf") or row.get("source_pdf_id") or row.get("v5_source_key") or ""


def v26_lookup() -> dict[str, dict[str, str]]:
    rows = read_csv(V26_DIR / "v26_row_assembly_attempts.csv")
    return {row.get("v25_component_id", ""): row for row in rows}


def physical_status(row: dict[str, str] | None) -> tuple[str, str, str]:
    if not row:
        return "required_physical_not_row_tested", "manual_review_before_static_site_use", "no_v26_row_assembly_attempt"
    status = row.get("v26_status", "")
    mapping = {
        "assembled_pass": ("required_physical_row_assembled_pass", "use_as_seed_row_assembly", ""),
        "assembled_complete_value_review": ("required_physical_row_assembled_value_review", "review_values_before_static_site_use", ""),
        "partial_assembled_review": ("required_physical_partial_row_assembly", "use_only_passing_blocks_after_manual_review", ""),
        "blocked_missing_ocr_artifact": ("required_physical_blocked_missing_ocr_artifact", "OCR_existing_images_before_row_assembly", ""),
        "review_failed_constraints": ("required_physical_failed_row_constraints", "manual_review_or_reocr_before_using", ""),
        "review_no_safe_block_alignment": ("required_physical_no_safe_block_alignment", "manual_review_or_reocr_before_using", ""),
    }
    return mapping.get(status, ("required_physical_row_review", "manual_review_before_static_site_use", status))


def build_required_physical_groups(
    components: list[dict[str, str]],
    members: list[dict[str, str]],
    attempts_by_component: dict[str, dict[str, str]],
) -> list[dict[str, object]]:
    members_by_component: dict[str, list[dict[str, str]]] = {}
    for member in members:
        members_by_component.setdefault(member.get("v25_component_id", ""), []).append(member)

    rows: list[dict[str, object]] = []
    for comp in components:
        if comp.get("component_kind") != "ironclad_physical_assembly":
            continue
        comp_id = comp.get("v25_component_id", "")
        attempt = attempts_by_component.get(comp_id)
        group_type, recommended_action, status_detail = physical_status(attempt)
        comp_members = sorted(members_by_component.get(comp_id, []), key=lambda row: int_or_none(row.get("position_in_component")) or 0)
        pages = [member.get("pdf_page", "") for member in comp_members]
        rows.append(
            {
                "v27_group_id": stable_id("v27phys", comp_id),
                "v27_group_type": group_type,
                "v27_group_level": "required_physical",
                "recommended_action": recommended_action,
                "v25_component_id": comp_id,
                "v26_status": attempt.get("v26_status", "") if attempt else "",
                "v26_structural_status": attempt.get("structural_status", "") if attempt else "",
                "v26_value_constraint_status": attempt.get("value_constraint_status", "") if attempt else "",
                "v26_assembled_row_count": attempt.get("assembled_row_count", "") if attempt else "",
                "v26_passed_block_count": attempt.get("passed_block_count", "") if attempt else "",
                "v26_failed_block_count": attempt.get("failed_block_count", "") if attempt else "",
                "v5_source_key": comp.get("v5_source_key", ""),
                "source_pdf": comp.get("source_pdf", ""),
                "source_pdf_id": comp.get("source_pdf_id", ""),
                "year": comp.get("year", ""),
                "province": comp.get("province", ""),
                "page_ranges": comp.get("page_range", ""),
                "pdf_pages": "; ".join(pages),
                "n_physical_pages": comp.get("n_physical_pages", ""),
                "all_pages_consecutive": comp.get("all_pages_consecutive", ""),
                "physical_merge_allowed": "yes",
                "logical_collection_only": "no",
                "missing_pages": "",
                "source_evidence": "v25_ironclad_seed_physical_join; v26_row_assembly_attempt",
                "review_flags": uniq_join([comp.get("component_review_flags", ""), attempt.get("constraint_flags", "") if attempt else "", status_detail], limit=30),
                "primary_title": comp.get("primary_title", ""),
                "member_physical_page_ids": uniq_join([member.get("physical_page_id", "") for member in comp_members], limit=50),
                "image_paths": uniq_join([member.get("image_path", "") for member in comp_members], limit=8),
                "ocr_artifacts": uniq_join([member.get("ocr_artifact", "") for member in comp_members], limit=8),
            }
        )
    rows.sort(key=lambda row: (str(row.get("v5_source_key") or row.get("source_pdf")), int_or_none(str(row.get("page_ranges", "")).split("-", 1)[0]) or 10**9))
    return rows


def logical_candidate_status(row: dict[str, str]) -> dict[str, object]:
    ranges = parse_page_ranges(row.get("page_ranges"))
    consecutive, gaps = ranges_are_consecutive(ranges)
    flags = row.get("collection_review_flags", "")
    has_missing_flag = "collection_crosses_missing_pages" in flags or bool(gaps)
    n_runs = int_or_none(row.get("n_physical_runs")) or len(ranges)
    n_pages = int_or_none(row.get("n_physical_pages")) or total_range_pages(ranges)
    candidate_type = row.get("v25_candidate_type", "")
    large_review = n_runs > 12 or n_pages > 24
    if has_missing_flag:
        status = "logical_collection_missing_bridge_candidate"
        action = "fill_missing_pdf_pages_before_logical_grouping"
        actual_allowed = "no"
    elif not consecutive:
        status = "logical_collection_nonconsecutive_review"
        action = "manual_review_before_logical_grouping"
        actual_allowed = "no"
    elif large_review:
        status = "logical_collection_consecutive_large_review"
        action = "manual_review_large_collection_before_static_site_use"
        actual_allowed = "review"
    elif candidate_type == "logical_religion_or_part_sibling_candidate":
        status = "logical_religion_or_part_sibling_consecutive"
        action = "safe_logical_collection_only_after_title_check"
        actual_allowed = "yes"
    elif candidate_type == "self_contained_continuation_sequence_candidate":
        status = "logical_self_contained_sequence_consecutive"
        action = "safe_logical_collection_only_not_physical_merge"
        actual_allowed = "yes"
    else:
        status = "logical_same_table_consecutive"
        action = "safe_logical_collection_only_not_physical_merge"
        actual_allowed = "yes"
    return {
        "v27_group_type": status,
        "recommended_action": action,
        "actual_logical_group_allowed": actual_allowed,
        "all_ranges_consecutive": "yes" if consecutive else "no",
        "range_gaps": gaps,
        "n_ranges_parsed": len(ranges),
        "range_page_count": total_range_pages(ranges),
        "large_collection_review": "yes" if large_review else "no",
    }


def build_logical_groups(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for row in rows:
        status = logical_candidate_status(row)
        out.append(
            {
                "v27_group_id": stable_id("v27logic", row.get("collection_group_id", "")),
                "v27_group_level": "logical_collection",
                **status,
                "collection_group_id": row.get("collection_group_id", ""),
                "v25_candidate_type": row.get("v25_candidate_type", ""),
                "collection_basis": row.get("collection_basis", ""),
                "v5_source_key": row.get("v5_source_key", ""),
                "source_pdf": row.get("source_pdf", ""),
                "source_pdf_id": row.get("source_pdf_id", ""),
                "page_ranges": row.get("page_ranges", ""),
                "n_physical_runs": row.get("n_physical_runs", ""),
                "n_physical_pages": row.get("n_physical_pages", ""),
                "physical_merge_allowed": "no",
                "logical_collection_only": "yes",
                "missing_pages": status["range_gaps"],
                "source_evidence": "v25_bounded_same_source_family_schema_collection_candidate",
                "review_flags": row.get("collection_review_flags", ""),
                "primary_title": row.get("primary_title", ""),
                "collection_reasons": row.get("collection_reasons", ""),
                "image_paths": row.get("example_images", ""),
            }
        )
    out.sort(key=lambda row: (str(row.get("v27_group_type")), str(row.get("v5_source_key") or row.get("source_pdf")), int_or_none(str(row.get("page_ranges", "")).split(";", 1)[0].split("-", 1)[0]) or 10**9))
    return out


def build_missing_bridge_groups(growth: list[dict[str, str]], logical_groups: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in growth:
        if row.get("candidate_type") != "missing_bridge_candidate":
            continue
        rows.append(
            {
                "v27_group_id": stable_id("v27miss", row.get("candidate_id", "")),
                "v27_group_level": "missing_bridge",
                "v27_group_type": "physical_or_logical_missing_pdf_pages_candidate",
                "recommended_action": "render_or_locate_missing_pdf_pages_before_assembly",
                "candidate_id": row.get("candidate_id", ""),
                "v5_source_key": row.get("v5_source_key", ""),
                "source_pdf": "",
                "source_pdf_id": "",
                "prev_pdf_page": row.get("prev_pdf_page", ""),
                "pdf_page": row.get("pdf_page", ""),
                "missing_pages": row.get("missing_pages", ""),
                "prev_component_id": row.get("prev_component_id", ""),
                "component_id": row.get("component_id", ""),
                "physical_merge_allowed": "no",
                "logical_collection_only": "candidate_after_gap_fill",
                "source_evidence": "v25_missing_bridge_candidate",
                "review_flags": row.get("review_flags", ""),
                "primary_title": row.get("prev_title", "") or row.get("title", ""),
                "image_paths": uniq_join([row.get("prev_image_path", ""), row.get("image_path", "")], limit=4),
            }
        )
    for row in logical_groups:
        if row.get("v27_group_type") != "logical_collection_missing_bridge_candidate":
            continue
        rows.append(
            {
                "v27_group_id": stable_id("v27logicmiss", row.get("collection_group_id", "")),
                "v27_group_level": "missing_bridge",
                "v27_group_type": "logical_collection_missing_pdf_pages_candidate",
                "recommended_action": "fill_missing_pdf_pages_before_logical_grouping",
                "candidate_id": row.get("collection_group_id", ""),
                "v5_source_key": row.get("v5_source_key", ""),
                "source_pdf": row.get("source_pdf", ""),
                "source_pdf_id": row.get("source_pdf_id", ""),
                "prev_pdf_page": "",
                "pdf_page": "",
                "missing_pages": row.get("missing_pages", ""),
                "prev_component_id": "",
                "component_id": "",
                "physical_merge_allowed": "no",
                "logical_collection_only": "candidate_after_gap_fill",
                "source_evidence": "v25_logical_collection_crosses_missing_pages",
                "review_flags": row.get("review_flags", ""),
                "primary_title": row.get("primary_title", ""),
                "image_paths": row.get("image_paths", ""),
            }
        )
    rows.sort(key=lambda row: (str(row.get("v5_source_key") or row.get("source_pdf")), int_or_none(row.get("prev_pdf_page")) or 10**9, str(row.get("candidate_id"))))
    return rows


def build_edge_manifest(
    edges: list[dict[str, str]],
    members: list[dict[str, str]],
    attempts_by_component: dict[str, dict[str, str]],
) -> list[dict[str, object]]:
    page_to_component = {row.get("physical_page_id", ""): row.get("v25_component_id", "") for row in members}
    rows: list[dict[str, object]] = []
    for edge in edges:
        decision = edge.get("v25_decision", "")
        right_component = page_to_component.get(edge.get("physical_page_id", ""), "")
        attempt = attempts_by_component.get(right_component, {})
        if decision == "seed_physical_join":
            edge_class = "physical_join_allowed"
            physical_allowed = "yes"
            logical_allowed = "no"
            action = "use_v25_seed_join_then_apply_v26_status"
        elif decision == "logical_growth_candidate" and edge.get("page_gap") == "1":
            edge_class = "logical_collection_edge_only"
            physical_allowed = "no"
            logical_allowed = "yes"
            action = "logical_group_only_do_not_physically_merge"
        elif decision == "missing_bridge_candidate":
            edge_class = "missing_pages_bridge_candidate"
            physical_allowed = "no"
            logical_allowed = "candidate_after_gap_fill"
            action = "render_or_locate_missing_pdf_pages_before_grouping"
        elif decision == "growth_review_candidate":
            edge_class = "review_before_join"
            physical_allowed = "no"
            logical_allowed = "review"
            action = "manual_review_before_promoting"
        else:
            edge_class = "do_not_join"
            physical_allowed = "no"
            logical_allowed = "no"
            action = "keep_separate"
        rows.append(
            {
                "v27_edge_id": stable_id("v27edge", edge.get("v25_edge_id", "")),
                "v25_edge_id": edge.get("v25_edge_id", ""),
                "v27_edge_class": edge_class,
                "recommended_action": action,
                "physical_join_allowed": physical_allowed,
                "logical_group_allowed": logical_allowed,
                "v25_decision": decision,
                "v25_decision_layer": edge.get("v25_decision_layer", ""),
                "v25_confidence": edge.get("v25_confidence", ""),
                "v5_source_key": edge.get("v5_source_key", ""),
                "prev_pdf_page": edge.get("prev_pdf_page", ""),
                "pdf_page": edge.get("pdf_page", ""),
                "page_gap": edge.get("page_gap", ""),
                "missing_pages": edge.get("missing_pages", ""),
                "prev_component_id": page_to_component.get(edge.get("prev_physical_page_id", ""), ""),
                "component_id": right_component,
                "v26_status": attempt.get("v26_status", ""),
                "v26_structural_status": attempt.get("structural_status", ""),
                "v26_value_constraint_status": attempt.get("value_constraint_status", ""),
                "title_similarity": edge.get("title_similarity", ""),
                "review_flags": uniq_join([edge.get("v25_review_flags", ""), edge.get("collection_review_flags", ""), attempt.get("constraint_flags", "")], limit=30),
                "reason": uniq_join([edge.get("v25_reason", ""), edge.get("collection_reason", "")], limit=20),
                "prev_title": edge.get("prev_title", ""),
                "title": edge.get("title", ""),
                "prev_image_path": edge.get("prev_image_path", ""),
                "image_path": edge.get("image_path", ""),
            }
        )
    return rows


def build_manifest(required: list[dict[str, object]], logical: list[dict[str, object]], missing: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in required:
        rows.append(
            {
                "v27_group_id": row.get("v27_group_id", ""),
                "v27_group_level": row.get("v27_group_level", ""),
                "v27_group_type": row.get("v27_group_type", ""),
                "recommended_action": row.get("recommended_action", ""),
                "v5_source_key": row.get("v5_source_key", ""),
                "source_pdf": row.get("source_pdf", ""),
                "source_pdf_id": row.get("source_pdf_id", ""),
                "page_ranges": row.get("page_ranges", ""),
                "n_physical_pages": row.get("n_physical_pages", ""),
                "physical_merge_allowed": row.get("physical_merge_allowed", ""),
                "logical_collection_only": row.get("logical_collection_only", ""),
                "actual_logical_group_allowed": "",
                "missing_pages": row.get("missing_pages", ""),
                "review_flags": row.get("review_flags", ""),
                "primary_title": row.get("primary_title", ""),
                "source_evidence": row.get("source_evidence", ""),
            }
        )
    for row in logical:
        if row.get("actual_logical_group_allowed") == "no":
            continue
        rows.append(
            {
                "v27_group_id": row.get("v27_group_id", ""),
                "v27_group_level": row.get("v27_group_level", ""),
                "v27_group_type": row.get("v27_group_type", ""),
                "recommended_action": row.get("recommended_action", ""),
                "v5_source_key": row.get("v5_source_key", ""),
                "source_pdf": row.get("source_pdf", ""),
                "source_pdf_id": row.get("source_pdf_id", ""),
                "page_ranges": row.get("page_ranges", ""),
                "n_physical_pages": row.get("n_physical_pages", ""),
                "physical_merge_allowed": row.get("physical_merge_allowed", ""),
                "logical_collection_only": row.get("logical_collection_only", ""),
                "actual_logical_group_allowed": row.get("actual_logical_group_allowed", ""),
                "missing_pages": row.get("missing_pages", ""),
                "review_flags": row.get("review_flags", ""),
                "primary_title": row.get("primary_title", ""),
                "source_evidence": row.get("source_evidence", ""),
            }
        )
    for row in missing:
        rows.append(
            {
                "v27_group_id": row.get("v27_group_id", ""),
                "v27_group_level": row.get("v27_group_level", ""),
                "v27_group_type": row.get("v27_group_type", ""),
                "recommended_action": row.get("recommended_action", ""),
                "v5_source_key": row.get("v5_source_key", ""),
                "source_pdf": row.get("source_pdf", ""),
                "source_pdf_id": row.get("source_pdf_id", ""),
                "page_ranges": "",
                "n_physical_pages": "",
                "physical_merge_allowed": row.get("physical_merge_allowed", ""),
                "logical_collection_only": row.get("logical_collection_only", ""),
                "actual_logical_group_allowed": "",
                "missing_pages": row.get("missing_pages", ""),
                "review_flags": row.get("review_flags", ""),
                "primary_title": row.get("primary_title", ""),
                "source_evidence": row.get("source_evidence", ""),
            }
        )
    rows.sort(key=lambda row: (str(row.get("v27_group_level")), str(row.get("v5_source_key") or row.get("source_pdf")), int_or_none(str(row.get("page_ranges")).split(";", 1)[0].split("-", 1)[0]) or 10**9))
    return rows


def validate(required: list[dict[str, object]], logical: list[dict[str, object]], manifest: list[dict[str, object]], edges: list[dict[str, object]]) -> list[dict[str, object]]:
    failures: list[tuple[str, list[str]]] = []
    bad_required = [
        str(row.get("v27_group_id"))
        for row in required
        if row.get("physical_merge_allowed") != "yes"
        or row.get("all_pages_consecutive") != "yes"
        or (int_or_none(row.get("n_physical_pages")) or 0) > 2
    ]
    failures.append(("required_physical_groups_are_consecutive_and_pairwise", bad_required))
    bad_missing_in_actual_logical = [
        str(row.get("v27_group_id"))
        for row in logical
        if row.get("actual_logical_group_allowed") in {"yes", "review"}
        and (row.get("all_ranges_consecutive") != "yes" or row.get("missing_pages"))
    ]
    failures.append(("actual_logical_groups_do_not_cross_missing_pages", bad_missing_in_actual_logical))
    bad_physical_merge_in_logical = [
        str(row.get("v27_group_id"))
        for row in logical
        if row.get("physical_merge_allowed") != "no"
    ]
    failures.append(("logical_groups_are_not_physical_merges", bad_physical_merge_in_logical))
    bad_manifest_physical = [
        str(row.get("v27_group_id"))
        for row in manifest
        if row.get("v27_group_level") == "logical_collection" and row.get("physical_merge_allowed") != "no"
    ]
    failures.append(("manifest_logical_rows_keep_physical_merge_disabled", bad_manifest_physical))
    bad_edge_physical = [
        str(row.get("v27_edge_id"))
        for row in edges
        if row.get("physical_join_allowed") == "yes" and row.get("v25_decision") != "seed_physical_join"
    ]
    failures.append(("only_v25_seed_edges_allow_physical_join", bad_edge_physical))
    return [
        {
            "check": name,
            "status": "pass" if not bad else "fail",
            "failure_count": len(bad),
            "details": json.dumps(bad[:50], sort_keys=True),
        }
        for name, bad in failures
    ]


def write_workbook(path: Path, sheets: dict[str, tuple[list[dict[str, object]], list[str]]]) -> None:
    workbook = xlsxwriter.Workbook(str(path))
    header = workbook.add_format({"bold": True, "bg_color": "#D9EAF7", "border": 1})
    wrap = workbook.add_format({"text_wrap": True, "valign": "top"})
    for sheet_name, (rows, fields) in sheets.items():
        ws = workbook.add_worksheet(sheet_name[:31])
        for col, field in enumerate(fields):
            ws.write(0, col, field, header)
        for r_idx, row in enumerate(rows, start=1):
            for c_idx, field in enumerate(fields):
                fmt = wrap if any(token in field for token in ["reason", "flags", "title", "image", "artifact", "evidence"]) else None
                ws.write(r_idx, c_idx, row.get(field, ""), fmt)
        ws.freeze_panes(1, 0)
        ws.autofilter(0, 0, max(1, len(rows)), max(0, len(fields) - 1))
        for col, field in enumerate(fields):
            width = 16
            if any(token in field for token in ["reason", "flags", "title", "image", "artifact", "evidence"]):
                width = 45
            elif "id" in field:
                width = 24
            ws.set_column(col, col, width)
    workbook.close()


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    components = read_csv(V25_DIR / "v25_seed_components.csv")
    members = read_csv(V25_DIR / "v25_seed_component_members.csv")
    edges = read_csv(V25_DIR / "v25_pairwise_seed_edges.csv")
    logical_candidates = read_csv(V25_DIR / "v25_logical_collection_candidates.csv")
    growth_candidates = read_csv(V25_DIR / "v25_growth_candidates.csv")
    attempts = v26_lookup()

    required = build_required_physical_groups(components, members, attempts)
    logical = build_logical_groups(logical_candidates)
    missing = build_missing_bridge_groups(growth_candidates, logical)
    edge_manifest = build_edge_manifest(edges, members, attempts)
    manifest = build_manifest(required, logical, missing)
    audit = validate(required, logical, manifest, edge_manifest)

    required_fields = [
        "v27_group_id",
        "v27_group_level",
        "v27_group_type",
        "recommended_action",
        "v25_component_id",
        "v26_status",
        "v26_structural_status",
        "v26_value_constraint_status",
        "v26_assembled_row_count",
        "v26_passed_block_count",
        "v26_failed_block_count",
        "v5_source_key",
        "source_pdf",
        "source_pdf_id",
        "year",
        "province",
        "page_ranges",
        "pdf_pages",
        "n_physical_pages",
        "all_pages_consecutive",
        "physical_merge_allowed",
        "logical_collection_only",
        "missing_pages",
        "source_evidence",
        "review_flags",
        "primary_title",
        "member_physical_page_ids",
        "image_paths",
        "ocr_artifacts",
    ]
    logical_fields = [
        "v27_group_id",
        "v27_group_level",
        "v27_group_type",
        "recommended_action",
        "actual_logical_group_allowed",
        "all_ranges_consecutive",
        "range_gaps",
        "n_ranges_parsed",
        "range_page_count",
        "large_collection_review",
        "collection_group_id",
        "v25_candidate_type",
        "collection_basis",
        "v5_source_key",
        "source_pdf",
        "source_pdf_id",
        "page_ranges",
        "n_physical_runs",
        "n_physical_pages",
        "physical_merge_allowed",
        "logical_collection_only",
        "missing_pages",
        "source_evidence",
        "review_flags",
        "primary_title",
        "collection_reasons",
        "image_paths",
    ]
    missing_fields = [
        "v27_group_id",
        "v27_group_level",
        "v27_group_type",
        "recommended_action",
        "candidate_id",
        "v5_source_key",
        "source_pdf",
        "source_pdf_id",
        "prev_pdf_page",
        "pdf_page",
        "missing_pages",
        "prev_component_id",
        "component_id",
        "physical_merge_allowed",
        "logical_collection_only",
        "source_evidence",
        "review_flags",
        "primary_title",
        "image_paths",
    ]
    edge_fields = [
        "v27_edge_id",
        "v25_edge_id",
        "v27_edge_class",
        "recommended_action",
        "physical_join_allowed",
        "logical_group_allowed",
        "v25_decision",
        "v25_decision_layer",
        "v25_confidence",
        "v5_source_key",
        "prev_pdf_page",
        "pdf_page",
        "page_gap",
        "missing_pages",
        "prev_component_id",
        "component_id",
        "v26_status",
        "v26_structural_status",
        "v26_value_constraint_status",
        "title_similarity",
        "review_flags",
        "reason",
        "prev_title",
        "title",
        "prev_image_path",
        "image_path",
    ]
    manifest_fields = [
        "v27_group_id",
        "v27_group_level",
        "v27_group_type",
        "recommended_action",
        "v5_source_key",
        "source_pdf",
        "source_pdf_id",
        "page_ranges",
        "n_physical_pages",
        "physical_merge_allowed",
        "logical_collection_only",
        "actual_logical_group_allowed",
        "missing_pages",
        "review_flags",
        "primary_title",
        "source_evidence",
    ]
    audit_fields = ["check", "status", "failure_count", "details"]

    safe_logical = [row for row in logical if row.get("actual_logical_group_allowed") in {"yes", "review"}]
    blocked_logical = [row for row in logical if row.get("actual_logical_group_allowed") == "no"]

    write_csv(OUT_DIR / "v27_required_physical_groups.csv", required, required_fields)
    write_csv(OUT_DIR / "v27_logical_group_candidates.csv", logical, logical_fields)
    write_csv(OUT_DIR / "v27_safe_logical_groups.csv", safe_logical, logical_fields)
    write_csv(OUT_DIR / "v27_blocked_logical_groups.csv", blocked_logical, logical_fields)
    write_csv(OUT_DIR / "v27_missing_bridge_groups.csv", missing, missing_fields)
    write_csv(OUT_DIR / "v27_pairwise_edge_manifest.csv", edge_manifest, edge_fields)
    write_csv(OUT_DIR / "v27_assembly_manifest.csv", manifest, manifest_fields)
    write_csv(OUT_DIR / "v27_audit.csv", audit, audit_fields)

    summary_rows = [
        {"metric": "required_physical_groups", "value": len(required)},
        {"metric": "logical_group_candidates", "value": len(logical)},
        {"metric": "safe_or_review_logical_groups", "value": len(safe_logical)},
        {"metric": "blocked_logical_groups", "value": len(blocked_logical)},
        {"metric": "missing_bridge_groups", "value": len(missing)},
        {"metric": "assembly_manifest_rows", "value": len(manifest)},
        {"metric": "pairwise_edge_manifest_rows", "value": len(edge_manifest)},
        {"metric": "audit_failure_count", "value": sum(int(row["failure_count"]) for row in audit)},
    ]
    for key, value in sorted(Counter(row["v27_group_type"] for row in required).items()):
        summary_rows.append({"metric": f"required_type::{key}", "value": value})
    for key, value in sorted(Counter(row["v27_group_type"] for row in logical).items()):
        summary_rows.append({"metric": f"logical_type::{key}", "value": value})
    write_csv(OUT_DIR / "v27_summary.csv", summary_rows, ["metric", "value"])

    workbook_path = OUT_DIR / "v27_scalable_assembly_manifest.xlsx"
    write_workbook(
        workbook_path,
        {
            "summary": (summary_rows, ["metric", "value"]),
            "assembly_manifest": (manifest, manifest_fields),
            "required_physical": (required, required_fields),
            "safe_logical": (safe_logical, logical_fields),
            "blocked_logical": (blocked_logical, logical_fields),
            "missing_bridge": (missing, missing_fields),
            "pairwise_edges": (edge_manifest[:5000], edge_fields),
            "audit": (audit, audit_fields),
        },
    )

    bad_member = None
    try:
        with zipfile.ZipFile(workbook_path) as zf:
            bad_member = zf.testzip()
    except Exception as exc:
        bad_member = f"{type(exc).__name__}: {exc}"

    summary = {
        "v27_required_physical_groups": len(required),
        "v27_required_group_type_counts": dict(sorted(Counter(row["v27_group_type"] for row in required).items())),
        "v27_logical_group_candidates": len(logical),
        "v27_logical_group_type_counts": dict(sorted(Counter(row["v27_group_type"] for row in logical).items())),
        "v27_safe_or_review_logical_groups": len(safe_logical),
        "v27_blocked_logical_groups": len(blocked_logical),
        "v27_missing_bridge_groups": len(missing),
        "v27_assembly_manifest_rows": len(manifest),
        "v27_pairwise_edge_manifest_rows": len(edge_manifest),
        "v27_audit_failure_count": sum(int(row["failure_count"]) for row in audit),
        "v27_workbook_zip_bad_member": bad_member,
    }
    (OUT_DIR / "v27_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
