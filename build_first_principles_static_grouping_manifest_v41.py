#!/usr/bin/env python3
"""Build a static-site-facing grouping manifest from the no-OCR assembly stack.

This is a packaging layer only. It does not OCR, call APIs, or promote
candidate edges. It combines:

* v38 contract physical recommendations,
* v39 physical edge/component evidence, and
* v40 bounded logical collection evidence.

The output is intentionally conservative: physical merges are allowed only for
confirmed consecutive physical components, while logical religion/table
collections remain logical-only.
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
DEFAULT_CONTRACT_DIR = ROOT / "export_final/pdf_recovery_incoming_2026-06-16_first_principles_assembly_contract_smoke_v40_logical"
DEFAULT_PHYSICAL_EVIDENCE_DIR = ROOT / "export_final/pdf_recovery_incoming_2026-06-16_first_principles_assembly_evidence_smoke_v40_logical"
DEFAULT_LOGICAL_EVIDENCE_DIR = ROOT / "export_final/pdf_recovery_incoming_2026-06-16_first_principles_logical_collection_evidence_smoke_v40_logical"
DEFAULT_OUT_DIR = ROOT / "export_final/pdf_recovery_incoming_2026-06-16_first_principles_static_grouping_manifest_v41"

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

AUDIT_FIELDS = ["check", "status", "detail"]
SUMMARY_FIELDS = ["metric", "name", "count"]
COLUMN_ORDER_FIELDS = ["group_id", "source_page", "section", "row_label", "column_orders", "column_groups", "status", "detail"]


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
        return int(str(value))
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


def expand_pages(value: object) -> list[int]:
    text = str(value or "").strip()
    if not text:
        return []
    pages: list[int] = []
    for part in re.split(r"[;,]\s*", text):
        part = part.strip()
        if not part:
            continue
        match = re.fullmatch(r"(\d+)\s*-\s*(\d+)", part)
        if match:
            start, end = int(match.group(1)), int(match.group(2))
            if start <= end and end - start <= 500:
                pages.extend(range(start, end + 1))
            continue
        page = int_or_none(part)
        if page is not None:
            pages.append(page)
    return pages


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


def pages_consecutive(pages: Iterable[int]) -> bool:
    values = sorted(set(pages))
    return bool(values) and values == list(range(values[0], values[-1] + 1))


def page_pair_pages(value: object) -> list[int]:
    text = str(value or "")
    match = re.fullmatch(r"\s*(\d+)\s*-\s*(\d+)\s*", text)
    if match:
        return [int(match.group(1)), int(match.group(2))]
    return expand_pages(text)


def missing_between_pair(value: object) -> str:
    pages = page_pair_pages(value)
    if len(pages) != 2:
        return ""
    start, end = sorted(pages)
    if end <= start + 1:
        return ""
    return compact_pages(range(start + 1, end))


def source_id(row: dict[str, str]) -> str:
    return row.get("source_pdf_id_guess") or row.get("source_pdf_id") or ""


def row_label_signal(row: dict[str, str]) -> str:
    haystack = " ".join(
        [
            row.get("physical_rule_status", ""),
            row.get("evidence_bucket", ""),
            row.get("row_alignment_evidence", ""),
            row.get("rule_flags", ""),
            row.get("rule_decision_reason", ""),
        ]
    ).lower()
    return "yes" if "row_label" in haystack or "row-label" in haystack or "no row label" in haystack else "no"


def build_page_image_lookup(edges: list[dict[str, str]]) -> dict[tuple[str, str, str], str]:
    lookup: dict[tuple[str, str, str], str] = {}
    for row in edges:
        pages = page_pair_pages(row.get("page_pair", ""))
        if len(pages) != 2:
            continue
        base = (row.get("v5_source_key", ""), source_id(row))
        if row.get("prev_image_path"):
            lookup[(base[0], base[1], str(pages[0]))] = row.get("prev_image_path", "")
        if row.get("image_path"):
            lookup[(base[0], base[1], str(pages[1]))] = row.get("image_path", "")
    return lookup


def physical_component_groups(
    contract_rows: list[dict[str, str]],
    component_rows: list[dict[str, str]],
) -> list[dict[str, object]]:
    contract_by_source = {row.get("source_record_id", ""): row for row in contract_rows}
    groups: list[dict[str, object]] = []
    for row in component_rows:
        contract = contract_by_source.get(row.get("source_record_id", ""), {})
        safe = row.get("safe_to_publish_values", "") == "yes"
        groups.append(
            {
                "group_id": f"physical::{row.get('component_id', '')}",
                "group_layer": "physical",
                "group_kind": "site_ready_physical_component",
                "source_record_id": row.get("source_record_id", ""),
                "site_use_policy": row.get("site_use_policy", "use_now_physical_component_structure"),
                "physical_merge_allowed": row.get("physical_merge_allowed_now", "yes"),
                "logical_collection_only": "no",
                "safe_to_publish_values": row.get("safe_to_publish_values", "no"),
                "review_required": row.get("review_required", "yes"),
                "priority": row.get("component_priority", contract.get("priority", "1")),
                "v5_source_key": row.get("v5_source_key", ""),
                "source_pdf_id": source_id(row),
                "page_range": row.get("page_range", ""),
                "component_pages": row.get("component_pages", ""),
                "n_pages": row.get("n_pages", ""),
                "edge_pairs": row.get("edge_pairs", ""),
                "member_count": row.get("n_pages", ""),
                "pages_are_consecutive": "yes" if pages_consecutive(expand_pages(row.get("component_pages", ""))) else "no",
                "table_number_key": "",
                "topic_variant_key": "",
                "religion_terms_detected": contract.get("religion_terms_detected", ""),
                "evidence_bucket": row.get("component_evidence_bucket", ""),
                "evidence_priority": row.get("component_priority", ""),
                "row_label_dependency_signal": "yes" if "row_label" in str(row.get("row_alignment_summary", "")).lower() else "no",
                "missing_intermediate_pages": "",
                "value_status": "publishable_values" if safe else f"structure_only_{row.get('zero_check_status', '') or 'value_review'}",
                "title_sample": contract.get("primary_title", ""),
                "recommended_action": row.get("recommended_next_step", "") or contract.get("recommended_action", ""),
                "evidence_files": semi([contract.get("evidence_files", ""), "physical_component_evidence.csv", "assembled_physical_preview_long.csv"]),
            }
        )
    return groups


def physical_candidate_groups(edges: list[dict[str, str]]) -> list[dict[str, object]]:
    groups: list[dict[str, object]] = []
    for row in edges:
        if row.get("physical_merge_allowed_now") == "yes":
            continue
        pages = page_pair_pages(row.get("page_pair", ""))
        missing = missing_between_pair(row.get("page_pair", ""))
        bucket = row.get("evidence_bucket", "")
        if missing:
            kind = "likely_physical_after_missing_pages_no_merge"
        elif bucket == "needs_missing_ocr_artifact":
            kind = "physical_candidate_needs_ocr_artifact"
        elif bucket == "candidate_has_strong_evidence_review_for_promotion":
            kind = "strong_physical_candidate_review_for_promotion"
        else:
            kind = "physical_candidate_review"
        groups.append(
            {
                "group_id": stable_id("physical_candidate", row.get("edge_key", ""), row.get("page_pair", "")),
                "group_layer": "physical_candidate",
                "group_kind": kind,
                "source_record_id": row.get("edge_key", ""),
                "site_use_policy": "do_not_merge_until_edge_promotes",
                "physical_merge_allowed": "no",
                "logical_collection_only": "no",
                "safe_to_publish_values": "no",
                "review_required": "yes",
                "priority": row.get("evidence_priority", "5"),
                "v5_source_key": row.get("v5_source_key", ""),
                "source_pdf_id": source_id(row),
                "page_range": row.get("page_pair", ""),
                "component_pages": semi(pages),
                "n_pages": len(set(pages)),
                "edge_pairs": row.get("page_pair", ""),
                "member_count": len(set(pages)),
                "pages_are_consecutive": row.get("pages_consecutive", "no"),
                "table_number_key": "",
                "topic_variant_key": "",
                "religion_terms_detected": "",
                "evidence_bucket": bucket,
                "evidence_priority": row.get("evidence_priority", ""),
                "row_label_dependency_signal": row_label_signal(row),
                "missing_intermediate_pages": missing,
                "value_status": row.get("zero_check_status", ""),
                "title_sample": semi([row.get("left_title", ""), row.get("right_title", "")]),
                "recommended_action": row.get("recommended_next_step", ""),
                "evidence_files": "edge_assembly_evidence.csv",
            }
        )
    return groups


def logical_groups(rows: list[dict[str, str]], site_ready: bool) -> list[dict[str, object]]:
    groups: list[dict[str, object]] = []
    for row in rows:
        groups.append(
            {
                "group_id": f"logical::{row.get('logical_collection_id', '')}",
                "group_layer": "logical",
                "group_kind": "site_ready_bounded_logical_collection" if site_ready else "bounded_logical_collection_review",
                "source_record_id": row.get("logical_collection_id", ""),
                "site_use_policy": row.get("site_use_policy", ""),
                "physical_merge_allowed": "no",
                "logical_collection_only": "yes",
                "safe_to_publish_values": "no",
                "review_required": row.get("review_required", "yes"),
                "priority": row.get("priority", "3"),
                "v5_source_key": row.get("v5_source_key", ""),
                "source_pdf_id": row.get("source_pdf_id", ""),
                "page_range": row.get("page_range", ""),
                "component_pages": row.get("component_pages", ""),
                "n_pages": row.get("n_pages", ""),
                "edge_pairs": "",
                "member_count": row.get("member_count", ""),
                "pages_are_consecutive": row.get("pages_are_consecutive", ""),
                "table_number_key": row.get("table_number_key", ""),
                "topic_variant_key": row.get("topic_variant_key", ""),
                "religion_terms_detected": row.get("religion_terms_detected", ""),
                "evidence_bucket": row.get("evidence_bucket", ""),
                "evidence_priority": row.get("priority", ""),
                "row_label_dependency_signal": "no",
                "missing_intermediate_pages": "",
                "value_status": "logical_collection_only",
                "title_sample": row.get("member_title_sample", ""),
                "recommended_action": row.get("recommended_action", ""),
                "evidence_files": "bounded_logical_collections.csv; logical_collection_members.csv",
            }
        )
    return groups


def physical_members(
    groups: list[dict[str, object]],
    previews: list[dict[str, str]],
    edges: list[dict[str, str]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    image_lookup = build_page_image_lookup(edges)
    artifact_lookup: dict[tuple[str, str, str], str] = {}
    preview_long: list[dict[str, object]] = []
    group_by_component = {
        str(group.get("group_id", "")).removeprefix("physical::"): group
        for group in groups
        if group.get("group_kind") == "site_ready_physical_component"
    }
    for row in previews:
        group = group_by_component.get(row.get("component_id", ""))
        if not group:
            continue
        key = (row.get("v5_source_key", ""), row.get("source_pdf_id_guess", ""), row.get("source_page", ""))
        if row.get("source_artifact"):
            artifact_lookup[key] = row.get("source_artifact", "")
        preview_long.append({"group_id": group.get("group_id", ""), **row})

    members: list[dict[str, object]] = []
    for group in groups:
        if group.get("group_kind") != "site_ready_physical_component":
            continue
        pages = expand_pages(group.get("component_pages", ""))
        for pos, page in enumerate(sorted(set(pages)), start=1):
            key = (str(group.get("v5_source_key", "")), str(group.get("source_pdf_id", "")), str(page))
            members.append(
                {
                    "group_id": group.get("group_id", ""),
                    "group_kind": group.get("group_kind", ""),
                    "member_position": pos,
                    "v5_source_key": group.get("v5_source_key", ""),
                    "source_pdf_id": group.get("source_pdf_id", ""),
                    "pdf_page": page,
                    "expanded_pages": page,
                    "source_artifact": artifact_lookup.get(key, ""),
                    "image_path": image_lookup.get(key, ""),
                    "title": group.get("title_sample", ""),
                }
            )
    return members, preview_long


def candidate_members(groups: list[dict[str, object]], edges: list[dict[str, str]]) -> list[dict[str, object]]:
    by_edge = {row.get("edge_key", ""): row for row in edges}
    members: list[dict[str, object]] = []
    for group in groups:
        if group.get("group_layer") != "physical_candidate":
            continue
        edge = by_edge.get(str(group.get("source_record_id", "")), {})
        pages = page_pair_pages(group.get("edge_pairs", ""))
        images = [edge.get("prev_image_path", ""), edge.get("image_path", "")]
        titles = [edge.get("left_title", ""), edge.get("right_title", "")]
        artifacts = [edge.get("left_best_json", "") or edge.get("left_best_csv", ""), edge.get("right_best_json", "") or edge.get("right_best_csv", "")]
        for pos, page in enumerate(pages, start=1):
            members.append(
                {
                    "group_id": group.get("group_id", ""),
                    "group_kind": group.get("group_kind", ""),
                    "member_position": pos,
                    "v5_source_key": group.get("v5_source_key", ""),
                    "source_pdf_id": group.get("source_pdf_id", ""),
                    "pdf_page": page,
                    "expanded_pages": page,
                    "source_artifact": artifacts[pos - 1] if pos - 1 < len(artifacts) else "",
                    "image_path": images[pos - 1] if pos - 1 < len(images) else "",
                    "title": titles[pos - 1] if pos - 1 < len(titles) else "",
                }
            )
    return members


def logical_members(groups: list[dict[str, object]], members: list[dict[str, str]]) -> list[dict[str, object]]:
    active_ids = {str(group.get("group_id", "")).removeprefix("logical::"): group for group in groups if group.get("group_layer") == "logical"}
    out: list[dict[str, object]] = []
    for row in members:
        group = active_ids.get(row.get("logical_collection_id", ""))
        if not group:
            continue
        out.append(
            {
                "group_id": group.get("group_id", ""),
                "group_kind": group.get("group_kind", ""),
                "member_position": row.get("member_position", ""),
                "v5_source_key": row.get("v5_source_key", ""),
                "source_pdf_id": row.get("source_pdf_id", ""),
                "pdf_page": row.get("pdf_page", ""),
                "expanded_pages": row.get("expanded_pages", ""),
                "table_number_key": row.get("table_number_key", ""),
                "topic_variant_key": row.get("topic_variant_key", ""),
                "religion_terms_detected": row.get("religion_terms_detected", ""),
                "image_path": row.get("image_path", ""),
                "title": row.get("title", ""),
            }
        )
    return out


def build_column_order_audit(physical_preview: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str, str], dict[str, object]] = {}
    for row in physical_preview:
        key = (str(row.get("group_id", "")), str(row.get("source_page", "")), str(row.get("section", "")), str(row.get("row_label", "")))
        bucket = grouped.setdefault(key, {"orders": [], "groups": [], "bad_orders": 0})
        order = int_or_none(row.get("column_group_order", ""))
        if order is None:
            bucket["bad_orders"] = int(bucket["bad_orders"]) + 1
        else:
            bucket["orders"].append(order)
        bucket["groups"].append(row.get("column_group", ""))

    audit_rows: list[dict[str, object]] = []
    for (group_id, source_page, section, row_label), bucket in sorted(grouped.items()):
        orders = sorted(set(bucket["orders"]))
        expected = list(range(1, max(orders) + 1)) if orders else []
        bad_orders = int(bucket["bad_orders"])
        ok = bool(orders) and bad_orders == 0 and orders == expected
        detail_parts = []
        if bad_orders:
            detail_parts.append(f"non_numeric_orders={bad_orders}")
        if orders != expected:
            detail_parts.append(f"expected={semi(expected)}")
        audit_rows.append(
            {
                "group_id": group_id,
                "source_page": source_page,
                "section": section,
                "row_label": row_label,
                "column_orders": semi(orders),
                "column_groups": semi(bucket["groups"]),
                "status": "pass" if ok else "fail",
                "detail": "; ".join(detail_parts) if detail_parts else "contiguous_from_1",
            }
        )
    return audit_rows


def build_audits(
    groups: list[dict[str, object]],
    members: list[dict[str, object]],
    physical_preview: list[dict[str, object]],
    column_order_audit: list[dict[str, object]],
    physical_components: list[dict[str, str]],
    edge_evidence: list[dict[str, str]],
    site_ready_logical: list[dict[str, str]],
    workbook_path: Path,
) -> list[dict[str, object]]:
    audits: list[dict[str, object]] = []
    physical_ready = [row for row in groups if row.get("group_kind") == "site_ready_physical_component"]
    bad_physical = [
        row for row in physical_ready
        if row.get("physical_merge_allowed") != "yes" or row.get("pages_are_consecutive") != "yes"
    ]
    audits.append({"check": "physical_merges_are_only_confirmed_consecutive_components", "status": "pass" if not bad_physical else "fail", "detail": f"bad={len(bad_physical)} total={len(physical_ready)}"})

    logical_bad = [row for row in groups if row.get("group_layer") == "logical" and row.get("physical_merge_allowed") == "yes"]
    audits.append({"check": "logical_collections_never_allow_physical_merge", "status": "pass" if not logical_bad else "fail", "detail": f"bad={len(logical_bad)}"})

    site_logical = [row for row in groups if row.get("group_kind") == "site_ready_bounded_logical_collection"]
    unbounded_logical = [
        row for row in site_logical
        if row.get("pages_are_consecutive") != "yes" or (int_or_none(row.get("n_pages")) or 999) > 6
    ]
    audits.append({"check": "site_ready_logical_groups_are_v40_bounded", "status": "pass" if not unbounded_logical else "fail", "detail": f"bad={len(unbounded_logical)} total={len(site_logical)}"})

    physical_component_ids = {f"physical::{row.get('component_id', '')}" for row in physical_components}
    manifest_physical_ids = {str(row.get("group_id", "")) for row in physical_ready}
    missing_physical = sorted(physical_component_ids - manifest_physical_ids)
    audits.append({"check": "all_v39_physical_components_represented", "status": "pass" if not missing_physical else "fail", "detail": f"missing={len(missing_physical)} total={len(physical_component_ids)}"})

    v40_site_ids = {f"logical::{row.get('logical_collection_id', '')}" for row in site_ready_logical}
    manifest_site_ids = {str(row.get("group_id", "")) for row in site_logical}
    missing_logical = sorted(v40_site_ids - manifest_site_ids)
    audits.append({"check": "all_v40_site_ready_logical_collections_represented", "status": "pass" if not missing_logical else "fail", "detail": f"missing={len(missing_logical)} total={len(v40_site_ids)}"})

    expected_candidates = {row.get("edge_key", "") for row in edge_evidence if row.get("physical_merge_allowed_now") != "yes"}
    manifest_candidates = {str(row.get("source_record_id", "")) for row in groups if row.get("group_layer") == "physical_candidate"}
    missing_candidates = sorted(expected_candidates - manifest_candidates)
    audits.append({"check": "all_nonpromoted_physical_edges_kept_in_review_manifest", "status": "pass" if not missing_candidates else "fail", "detail": f"missing={len(missing_candidates)} total={len(expected_candidates)}"})

    member_group_ids = {str(row.get("group_id", "")) for row in members}
    site_ready_ids = {str(row.get("group_id", "")) for row in groups if str(row.get("site_use_policy", "")).startswith("use_now")}
    missing_members = sorted(site_ready_ids - member_group_ids)
    audits.append({"check": "site_ready_groups_have_member_rows", "status": "pass" if not missing_members else "fail", "detail": f"missing={len(missing_members)} total={len(site_ready_ids)}"})

    preview_group_ids = {str(row.get("group_id", "")) for row in physical_preview}
    physical_ready_ids = {str(row.get("group_id", "")) for row in physical_ready}
    missing_preview = sorted(physical_ready_ids - preview_group_ids)
    audits.append({"check": "physical_preview_exists_for_each_site_ready_physical_component", "status": "pass" if not missing_preview else "fail", "detail": f"missing={len(missing_preview)} total={len(physical_ready_ids)}"})

    column_order_failures = [row for row in column_order_audit if row.get("status") != "pass"]
    audits.append({"check": "physical_preview_column_order_contiguous_per_source_row", "status": "pass" if not column_order_failures else "fail", "detail": f"bad={len(column_order_failures)} total={len(column_order_audit)}"})

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
    physical_preview: list[dict[str, object]],
    audits: list[dict[str, object]],
    paths: dict[str, Path],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = [{"metric": "paths", "name": name, "count": output_path(path)} for name, path in paths.items()]
    rows.extend(
        [
            {"metric": "counts", "name": "manifest_groups", "count": len(groups)},
            {"metric": "counts", "name": "manifest_members", "count": len(members)},
            {"metric": "counts", "name": "physical_preview_rows", "count": len(physical_preview)},
            {"metric": "counts", "name": "audit_failures", "count": sum(1 for row in audits if row.get("status") != "pass")},
            {"metric": "counts", "name": "site_ready_groups", "count": sum(1 for row in groups if str(row.get("site_use_policy", "")).startswith("use_now"))},
            {"metric": "counts", "name": "physical_merge_groups", "count": sum(1 for row in groups if row.get("physical_merge_allowed") == "yes")},
            {"metric": "counts", "name": "review_groups", "count": sum(1 for row in groups if row.get("review_required") == "yes" or not str(row.get("site_use_policy", "")).startswith("use_now"))},
        ]
    )
    for metric, counter in [
        ("groups_by_kind", Counter(str(row.get("group_kind", "")) for row in groups)),
        ("groups_by_policy", Counter(str(row.get("site_use_policy", "")) for row in groups)),
        ("groups_by_evidence_bucket", Counter(str(row.get("evidence_bucket", "")) for row in groups)),
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
    physical_preview: list[dict[str, object]],
    column_order_audit: list[dict[str, object]],
    physical_review: list[dict[str, object]],
    logical_review: list[dict[str, object]],
) -> None:
    workbook = xlsxwriter.Workbook(path, {"constant_memory": True})
    header_fmt = workbook.add_format({"bold": True, "bg_color": "#E8EEF7", "border": 1})
    wrap_fmt = workbook.add_format({"text_wrap": True, "valign": "top"})
    sheets: list[tuple[str, list[dict[str, object]], list[str]]] = [
        ("Summary", summary, SUMMARY_FIELDS),
        ("Audit", audits, AUDIT_FIELDS),
        ("Grouping Manifest", groups, GROUP_FIELDS),
        ("Group Members", members, MEMBER_FIELDS),
        ("Physical Review Queue", physical_review, GROUP_FIELDS),
        ("Logical Review Queue", logical_review, GROUP_FIELDS),
        ("Column Order Audit", column_order_audit, COLUMN_ORDER_FIELDS),
    ]
    if physical_preview:
        preview_fields = list(physical_preview[0].keys())
        sheets.append(("Physical Preview Long", physical_preview, preview_fields))
    for name, rows, fields in sheets:
        ws = workbook.add_worksheet(name[:31])
        for col, field in enumerate(fields):
            ws.write(0, col, field, header_fmt)
        for row_idx, row in enumerate(rows, start=1):
            for col, field in enumerate(fields):
                value = row.get(field, "")
                fmt = wrap_fmt if field in {"recommended_action", "evidence_files", "title_sample", "title", "image_path", "source_artifact"} else None
                ws.write(row_idx, col, value, fmt)
        ws.freeze_panes(1, 0)
        ws.autofilter(0, 0, max(0, len(rows)), max(0, len(fields) - 1))
        for col, field in enumerate(fields):
            ws.set_column(col, col, min(max(len(field) + 2, 12), 58))
    workbook.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract-dir", default=str(DEFAULT_CONTRACT_DIR))
    parser.add_argument("--physical-evidence-dir", default=str(DEFAULT_PHYSICAL_EVIDENCE_DIR))
    parser.add_argument("--logical-evidence-dir", default=str(DEFAULT_LOGICAL_EVIDENCE_DIR))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    args = parser.parse_args()

    contract_dir = Path(args.contract_dir).expanduser()
    physical_evidence_dir = Path(args.physical_evidence_dir).expanduser()
    logical_evidence_dir = Path(args.logical_evidence_dir).expanduser()
    out_dir = Path(args.out_dir).expanduser()
    paths = {
        "contract_dir": contract_dir,
        "physical_evidence_dir": physical_evidence_dir,
        "logical_evidence_dir": logical_evidence_dir,
        "out_dir": out_dir,
    }

    contract_physical = read_csv(contract_dir / "canonical_site_ready_physical_components.csv")
    physical_components = read_csv(physical_evidence_dir / "physical_component_evidence.csv")
    edge_evidence = read_csv(physical_evidence_dir / "edge_assembly_evidence.csv")
    physical_preview_raw = read_csv(physical_evidence_dir / "assembled_physical_preview_long.csv")
    site_ready_logical = read_csv(logical_evidence_dir / "site_ready_bounded_logical_collections.csv")
    review_logical = read_csv(logical_evidence_dir / "review_bounded_logical_candidates.csv")
    logical_member_rows = read_csv(logical_evidence_dir / "logical_collection_members.csv")

    physical_ready_groups = physical_component_groups(contract_physical, physical_components)
    physical_candidate_review_groups = physical_candidate_groups(edge_evidence)
    logical_site_groups = logical_groups(site_ready_logical, site_ready=True)
    logical_review_groups = logical_groups(review_logical, site_ready=False)
    groups = physical_ready_groups + logical_site_groups + physical_candidate_review_groups + logical_review_groups
    groups.sort(key=lambda row: (int_or_none(row.get("priority")) or 999, str(row.get("group_layer", "")), str(row.get("v5_source_key", "")), str(row.get("page_range", "")), str(row.get("group_id", ""))))
    for idx, row in enumerate(groups, start=1):
        row["group_rank"] = idx

    physical_member_rows, physical_preview = physical_members(groups, physical_preview_raw, edge_evidence)
    column_order_audit = build_column_order_audit(physical_preview)
    member_rows = physical_member_rows + logical_members(groups, logical_member_rows) + candidate_members(groups, edge_evidence)
    member_rows.sort(key=lambda row: (str(row.get("group_id", "")), int_or_none(row.get("member_position")) or 999, int_or_none(row.get("pdf_page")) or 999))
    for idx, row in enumerate(member_rows, start=1):
        row["member_rank"] = idx

    physical_review = [row for row in groups if row.get("group_layer") == "physical_candidate"]
    logical_review_manifest = [row for row in groups if row.get("group_kind") == "bounded_logical_collection_review"]

    out_dir.mkdir(parents=True, exist_ok=True)
    workbook_path = out_dir / "static_grouping_manifest_workbook.xlsx"
    write_workbook(workbook_path, [], [], groups, member_rows, physical_preview, column_order_audit, physical_review, logical_review_manifest)
    audits = build_audits(groups, member_rows, physical_preview, column_order_audit, physical_components, edge_evidence, site_ready_logical, workbook_path)
    summary = build_summary(groups, member_rows, physical_preview, audits, paths)
    write_workbook(workbook_path, summary, audits, groups, member_rows, physical_preview, column_order_audit, physical_review, logical_review_manifest)
    audits = build_audits(groups, member_rows, physical_preview, column_order_audit, physical_components, edge_evidence, site_ready_logical, workbook_path)
    summary = build_summary(groups, member_rows, physical_preview, audits, paths)
    write_workbook(workbook_path, summary, audits, groups, member_rows, physical_preview, column_order_audit, physical_review, logical_review_manifest)

    write_csv(out_dir / "canonical_grouping_manifest.csv", groups, GROUP_FIELDS)
    write_csv(out_dir / "canonical_group_members.csv", member_rows, MEMBER_FIELDS)
    write_csv(out_dir / "physical_edge_review_manifest.csv", physical_review, GROUP_FIELDS)
    write_csv(out_dir / "logical_review_manifest.csv", logical_review_manifest, GROUP_FIELDS)
    write_csv(out_dir / "assembled_physical_preview_long_with_group_id.csv", physical_preview, list(physical_preview[0].keys()) if physical_preview else ["group_id"])
    write_csv(out_dir / "physical_column_order_audit.csv", column_order_audit, COLUMN_ORDER_FIELDS)
    write_csv(out_dir / "acceptance_audit.csv", audits, AUDIT_FIELDS)
    write_csv(out_dir / "summary.csv", summary, SUMMARY_FIELDS)

    summary_json = {
        "manifest_groups": len(groups),
        "manifest_members": len(member_rows),
        "physical_preview_rows": len(physical_preview),
        "site_ready_groups": sum(1 for row in groups if str(row.get("site_use_policy", "")).startswith("use_now")),
        "physical_merge_groups": sum(1 for row in groups if row.get("physical_merge_allowed") == "yes"),
        "review_groups": sum(1 for row in groups if row.get("review_required") == "yes" or not str(row.get("site_use_policy", "")).startswith("use_now")),
        "audit_failure_count": sum(1 for row in audits if row.get("status") != "pass"),
        "groups_by_kind": dict(Counter(str(row.get("group_kind", "")) for row in groups)),
        "groups_by_policy": dict(Counter(str(row.get("site_use_policy", "")) for row in groups)),
        "outputs": {
            "workbook": output_path(workbook_path),
            "canonical_grouping_manifest": output_path(out_dir / "canonical_grouping_manifest.csv"),
            "canonical_group_members": output_path(out_dir / "canonical_group_members.csv"),
            "physical_edge_review_manifest": output_path(out_dir / "physical_edge_review_manifest.csv"),
            "logical_review_manifest": output_path(out_dir / "logical_review_manifest.csv"),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary_json, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary_json, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
