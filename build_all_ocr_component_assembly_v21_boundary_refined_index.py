#!/usr/bin/env python3
"""Boundary-refined master index for current assembly artifacts.

No OCR/API calls are made. This pass packages the validated v17 physical
components, v17 logical collections, v18 QA queues, and v19 variant
decomposition into one stable set of CSV/workbook artifacts, then refines the
remaining v18 ambiguous boundaries into explicit section-split or hard-break
decisions for downstream static-site updates and future OCR import checks.
"""

from __future__ import annotations

import csv
import json
import re
import shutil
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import build_all_ocr_component_assembly_v5_structural as base
import build_all_ocr_component_assembly_v11_rule_audit as v11
import build_all_ocr_component_assembly_v13_scope_refined as v13


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "export_final/pdf_recovery_incoming_2026-06-16_all_ocr_component_assembly_v21_boundary_refined_index"

V17_DIR = ROOT / "export_final/pdf_recovery_incoming_2026-06-16_all_ocr_component_assembly_v17_collections"
V18_DIR = ROOT / "export_final/pdf_recovery_incoming_2026-06-16_all_ocr_component_assembly_v18_quality_audit"
V19_DIR = ROOT / "export_final/pdf_recovery_incoming_2026-06-16_all_ocr_component_assembly_v19_variant_decomposition"

V17_COMPONENTS = V17_DIR / "v17_components.csv"
V17_COMPONENT_MEMBERS = V17_DIR / "v17_component_members.csv"
V17_COLLECTIONS = V17_DIR / "v17_collections.csv"
V17_COLLECTION_MEMBERS = V17_DIR / "v17_collection_members.csv"
V17_EDGES = V17_DIR / "v17_edge_recommendations.csv"
V17_COLLECTION_EDGES = V17_DIR / "v17_collection_edges.csv"
V18_INVARIANTS = V18_DIR / "v18_invariant_audit.csv"
V18_REQUIRED = V18_DIR / "v18_required_join_coverage.csv"
V18_MISSING_BRIDGES = V18_DIR / "v18_missing_bridge_review_edges.csv"
V18_UNRESOLVED = V18_DIR / "v18_unresolved_same_table_boundaries.csv"
V19_COLLECTIONS = V19_DIR / "v19_variant_collections.csv"
V19_MEMBERS = V19_DIR / "v19_variant_members.csv"
V19_EDGES = V19_DIR / "v19_variant_edges.csv"

GROUP_FIELDS = [
    "v21_group_id",
    "group_type",
    "source_layer",
    "source_id",
    "v5_source_key",
    "source_pdf",
    "source_pdf_id",
    "page_range",
    "all_pages_consecutive",
    "n_members",
    "n_physical_pages",
    "status",
    "confidence",
    "static_site_action",
    "review_flags",
    "table_number_keys",
    "primary_title",
    "example_images",
    "member_ids",
    "edge_ids",
]

MEMBER_FIELDS = [
    "v21_group_id",
    "group_type",
    "position",
    "member_type",
    "member_id",
    "source_layer",
    "v5_source_key",
    "source_pdf",
    "source_pdf_id",
    "page_range",
    "n_physical_pages",
    "join_from_prev_action",
    "join_from_prev_reason",
    "review_flags",
    "primary_title",
    "example_images",
]

EDGE_FIELDS = [
    "v21_edge_id",
    "edge_type",
    "source_layer",
    "source_edge_id",
    "from_id",
    "to_id",
    "v5_source_key",
    "source_pdf",
    "prev_page_range",
    "page_range",
    "prev_last_pdf_page",
    "first_pdf_page",
    "page_gap",
    "missing_pages",
    "action",
    "confidence",
    "reason",
    "review_flags",
    "strict_title_similarity",
    "scope_similarity",
    "prev_title",
    "title",
    "prev_example_images",
    "example_images",
]

PAGE_INDEX_FIELDS = [
    "v5_source_key",
    "source_pdf",
    "source_pdf_id",
    "pdf_page",
    "physical_component_ids",
    "v17_collection_ids",
    "v19_variant_collection_ids",
    "physical_component_layers",
    "collection_kinds",
    "variant_sources",
    "status_flags",
    "primary_titles",
    "example_images",
]

AUDIT_FIELDS = ["check", "status", "failure_count", "details"]
SUMMARY_FIELDS = ["metric", "value"]
BOUNDARY_DECISION_FIELDS = [
    "boundary_id",
    "boundary_type",
    "decision",
    "safe_whole_page_join",
    "next_action",
    "confidence",
    "v5_source_key",
    "source_pdf",
    "source_pdf_id",
    "prev_run_id",
    "run_id",
    "prev_page_range",
    "run_page_range",
    "page_range",
    "prev_last_pdf_page",
    "first_pdf_page",
    "page_gap",
    "missing_pages",
    "review_flags",
    "reason",
    "visual_evidence",
    "prev_table_family_key",
    "table_family_key",
    "prev_table_number_key",
    "table_number_key",
    "strict_title_similarity",
    "scope_similarity",
    "prev_title",
    "title",
    "prev_example_images",
    "example_images",
]


BOUNDARY_REFINEMENTS = {
    "seqedge_bd2eb9c16730b676": {
        "group_type": "section_split_candidate",
        "edge_type": "section_split_candidate",
        "status": "section_split_required_before_logical_merge",
        "static_site_action": "show_section_split_candidate",
        "review_flags": "mixed_page_boundary; partial_cross_page_continuation; no_whole_page_join",
        "decision": "requires_section_split",
        "next_action": "split page 165 into Balaghat-concluded and South West Madhya Pradesh-start sections before logical assembly",
        "reason": "Page 165 contains Balaghat District conclusion above and starts South West Madhya Pradesh Division below; page 166 continues South West Madhya Pradesh. Whole-page merge would over-associate.",
        "visual_evidence": "export_final/pdf_recovery_incoming_2026-06-16_all_ocr_component_assembly_v21_boundary_review/boundary_1_1951__Madhya Pradesh_165_166.jpg",
    },
    "seqedge_a50de6401622dcc2": {
        "group_type": "section_split_candidate",
        "edge_type": "section_split_candidate",
        "status": "section_split_required_before_logical_merge",
        "static_site_action": "show_section_split_candidate",
        "review_flags": "mixed_page_boundary; partial_cross_page_variant; no_whole_page_join",
        "decision": "requires_section_split",
        "next_action": "split page 85 so Pattanapuram H-Parsi can group with page 84 while Shencottah starts separately",
        "reason": "Page 84 has Pattanapuram Sikh/Jain concluding variants; page 85 top has Pattanapuram H-Parsi but lower page starts Shencottah Taluk. Whole-page merge would attach Shencottah to Pattanapuram.",
        "visual_evidence": "export_final/pdf_recovery_incoming_2026-06-16_all_ocr_component_assembly_v21_boundary_review/boundary_2_saoa.crl.25851854.pdf_84_85.jpg",
    },
    "seqedge_f0b5d09df7ed51fd": {
        "group_type": "boundary_hard_break_confirmed",
        "edge_type": "hard_break_confirmed_by_visual_boundary_review",
        "status": "confirmed_scope_switch_do_not_merge",
        "static_site_action": "do_not_group_confirmed_hard_break",
        "review_flags": "confirmed_table_scope_switch; no_whole_page_join",
        "decision": "confirmed_hard_break",
        "next_action": "keep page 124 in Chengannur and pages 125-126 in Tiruvalla",
        "reason": "Page 124 concludes Chengannur Taluk variants; page 125 starts Tiruvalla Taluk A-Total Population and continues to page 126. This is a clean table-scope switch, not a missed join.",
        "visual_evidence": "export_final/pdf_recovery_incoming_2026-06-16_all_ocr_component_assembly_v21_boundary_review/boundary_3_saoa.crl.25851854.pdf_124_125-126.jpg",
    },
    "seqedge_8901e4d688708d85": {
        "group_type": "section_split_candidate",
        "edge_type": "section_split_candidate",
        "status": "section_split_required_before_logical_merge",
        "static_site_action": "show_section_split_candidate",
        "review_flags": "mixed_page_boundary; partial_cross_page_variant; no_whole_page_join",
        "decision": "requires_section_split",
        "next_action": "split page 161 so Changanachery E-Animistic can group with page 160 while Minachil starts separately",
        "reason": "Page 160 has Changanachery Musalman/Christian variants; page 161 top has Changanachery E-Animistic but lower page starts Minachil Taluk. Whole-page merge would over-associate.",
        "visual_evidence": "export_final/pdf_recovery_incoming_2026-06-16_all_ocr_component_assembly_v21_boundary_review/boundary_4_saoa.crl.25851854.pdf_160_161.jpg",
    },
    "seqedge_d61ca1e0b3b5e9db": {
        "group_type": "section_split_candidate",
        "edge_type": "section_split_candidate",
        "status": "section_split_required_before_logical_merge",
        "static_site_action": "show_section_split_candidate",
        "review_flags": "mixed_page_boundary; partial_cross_page_variant; no_whole_page_join",
        "decision": "requires_section_split",
        "next_action": "split page 171 so Muvattupuzha E-Animistic can group with page 170 while Todupuzha starts separately",
        "reason": "Page 170 has Muvattupuzha Musalman/Christian variants; page 171 top has Muvattupuzha E-Animistic but lower page starts Todupuzha Taluk. Whole-page merge would over-associate.",
        "visual_evidence": "export_final/pdf_recovery_incoming_2026-06-16_all_ocr_component_assembly_v21_boundary_review/boundary_5_saoa.crl.25851854.pdf_170_171.jpg",
    },
}


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


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def pages(value: object) -> list[int]:
    return v11.parse_pages(value)


def page_range_from_edge(edge: dict[str, str]) -> str:
    left = str(edge.get("prev_last_pdf_page") or "")
    right = str(edge.get("first_pdf_page") or "")
    if left and right:
        return f"{left}; {right}"
    return base.uniq_join([edge.get("prev_run_page_range"), edge.get("run_page_range")], limit=2)


def uniq(values: list[object], limit: int = 30) -> str:
    return base.uniq_join(values, limit=limit)


def source_pdf_from_key(source_key: object) -> str:
    key = str(source_key or "")
    if key.lower().endswith(".pdf"):
        return key
    return ""


def source_pdf_id_from_key(source_key: object) -> str:
    match = re.search(r"saoa\.crl\.(\d+)\.pdf$", str(source_key or ""), flags=re.IGNORECASE)
    return match.group(1) if match else ""


def boundary_refinement(edge: dict[str, str]) -> dict[str, str]:
    return BOUNDARY_REFINEMENTS.get(str(edge.get("sequence_edge_id") or ""), {})


def physical_group_id(component_id: object) -> str:
    return f"physical:{component_id}"


def collection_group_id(collection_id: object) -> str:
    return f"collection:{collection_id}"


def variant_group_id(variant_id: object) -> str:
    return f"variant:{variant_id}"


def bridge_group_id(edge_id: object) -> str:
    return f"missing_bridge:{edge_id}"


def unresolved_group_id(edge_id: object) -> str:
    return f"unresolved_boundary:{edge_id}"


def classify_physical_edge(edge: dict[str, str]) -> str:
    refined = boundary_refinement(edge)
    if refined:
        return refined.get("edge_type", "boundary_refined")
    action = str(edge.get("v17_recommended_action") or "")
    if action in JOIN_ACTIONS:
        return "physical_join"
    if action == "keep_missing_bridge":
        return "missing_bridge_candidate"
    if action == "review_same_table_boundary_not_joined":
        return "unresolved_same_table_boundary"
    if action == "keep_logical_variant_collection_boundary":
        return "logical_variant_boundary_nonphysical"
    if action.startswith("hard_break"):
        return "hard_break"
    return "physical_boundary_review"


def build_physical_groups(components: list[dict[str, str]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for component in components:
        component_id = str(component.get("v11_component_id") or "")
        status = "ready_physical_component"
        if "review" in str(component.get("v11_confidence") or "") or str(component.get("v11_review_reasons") or ""):
            status = "review_physical_component"
        rows.append(
            {
                "v21_group_id": physical_group_id(component_id),
                "group_type": "physical_component",
                "source_layer": "v17_components",
                "source_id": component_id,
                "v5_source_key": component.get("v5_source_key", ""),
                "source_pdf": component.get("source_pdf", ""),
                "source_pdf_id": component.get("source_pdf_id", ""),
                "page_range": component.get("page_range", ""),
                "all_pages_consecutive": component.get("all_pages_consecutive", ""),
                "n_members": component.get("n_v10_groups", ""),
                "n_physical_pages": component.get("n_physical_pages", ""),
                "status": status,
                "confidence": component.get("v11_confidence", ""),
                "static_site_action": component.get("v11_static_site_action", ""),
                "review_flags": component.get("v11_review_reasons", ""),
                "table_number_keys": component.get("table_number_keys", ""),
                "primary_title": component.get("primary_title", ""),
                "example_images": component.get("example_images", ""),
                "member_ids": component.get("v10_group_ids", ""),
                "edge_ids": component.get("v11_join_edge_ids", ""),
            }
        )
    return rows


def build_collection_groups(collections: list[dict[str, str]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for collection in collections:
        collection_id = str(collection.get("v17_collection_id") or "")
        kind = str(collection.get("collection_kind") or "collection")
        status = "ready_collection" if kind != "bounded_logical_variant_set" else "review_logical_variant_collection"
        if str(collection.get("collection_review_flags") or ""):
            status = "review_collection"
        rows.append(
            {
                "v21_group_id": collection_group_id(collection_id),
                "group_type": f"v17_{kind}",
                "source_layer": "v17_collections",
                "source_id": collection_id,
                "v5_source_key": collection.get("v5_source_key", ""),
                "source_pdf": collection.get("source_pdf", ""),
                "source_pdf_id": collection.get("source_pdf_id", ""),
                "page_range": collection.get("page_range", ""),
                "all_pages_consecutive": collection.get("all_pages_consecutive", ""),
                "n_members": collection.get("n_v17_components", ""),
                "n_physical_pages": collection.get("n_physical_pages", ""),
                "status": status,
                "confidence": collection.get("collection_confidence", ""),
                "static_site_action": collection.get("static_site_action", ""),
                "review_flags": collection.get("collection_review_flags", ""),
                "table_number_keys": collection.get("table_number_keys", ""),
                "primary_title": collection.get("primary_title", ""),
                "example_images": collection.get("example_images", ""),
                "member_ids": collection.get("v17_component_ids", ""),
                "edge_ids": collection.get("v17_collection_edge_ids", ""),
            }
        )
    return rows


def build_variant_groups(variants: list[dict[str, str]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for variant in variants:
        variant_id = str(variant.get("v19_variant_collection_id") or "")
        rows.append(
            {
                "v21_group_id": variant_group_id(variant_id),
                "group_type": "v19_logical_variant_collection",
                "source_layer": "v19_variant_collections",
                "source_id": variant_id,
                "v5_source_key": variant.get("v5_source_key", ""),
                "source_pdf": variant.get("source_pdf", ""),
                "source_pdf_id": variant.get("source_pdf_id", ""),
                "page_range": variant.get("page_range", ""),
                "all_pages_consecutive": variant.get("all_pages_consecutive", ""),
                "n_members": variant.get("n_members", ""),
                "n_physical_pages": variant.get("n_physical_pages", ""),
                "status": "review_logical_variant_collection" if str(variant.get("review_flags") or "") else "ready_logical_variant_collection",
                "confidence": "review",
                "static_site_action": "show_logical_variant_collection",
                "review_flags": variant.get("review_flags", ""),
                "table_number_keys": variant.get("table_number_keys", ""),
                "primary_title": variant.get("primary_title", ""),
                "example_images": variant.get("example_images", ""),
                "member_ids": variant.get("source_collection_id") or variant.get("v17_collection_id") or variant.get("sequence_group_id", ""),
                "edge_ids": variant.get("sequence_join_types", ""),
            }
        )
    return rows


def build_review_groups(missing_bridges: list[dict[str, str]], unresolved: list[dict[str, str]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for edge in missing_bridges:
        edge_id = str(edge.get("sequence_edge_id") or "")
        source_key = edge.get("v5_source_key", "")
        rows.append(
            {
                "v21_group_id": bridge_group_id(edge_id),
                "group_type": "missing_bridge_candidate",
                "source_layer": "v18_missing_bridge_review_edges",
                "source_id": edge_id,
                "v5_source_key": source_key,
                "source_pdf": source_pdf_from_key(source_key),
                "source_pdf_id": source_pdf_id_from_key(source_key),
                "page_range": page_range_from_edge(edge),
                "all_pages_consecutive": "False",
                "n_members": 2,
                "n_physical_pages": 2,
                "status": "missing_pages_needed_before_join",
                "confidence": "review",
                "static_site_action": "show_missing_bridge_candidate",
                "review_flags": edge.get("sequence_review_flags", ""),
                "table_number_keys": uniq([edge.get("prev_table_number_key"), edge.get("table_number_key")], limit=4),
                "primary_title": edge.get("prev_primary_title") or edge.get("primary_title", ""),
                "example_images": uniq([edge.get("prev_example_images"), edge.get("example_images")], limit=4),
                "member_ids": uniq([edge.get("prev_physical_run_id"), edge.get("physical_run_id")], limit=4),
                "edge_ids": edge_id,
            }
        )
    for edge in unresolved:
        edge_id = str(edge.get("sequence_edge_id") or "")
        source_key = edge.get("v5_source_key", "")
        refined = boundary_refinement(edge)
        rows.append(
            {
                "v21_group_id": unresolved_group_id(edge_id),
                "group_type": refined.get("group_type", "unresolved_same_table_boundary"),
                "source_layer": "v18_unresolved_same_table_boundaries",
                "source_id": edge_id,
                "v5_source_key": source_key,
                "source_pdf": source_pdf_from_key(source_key),
                "source_pdf_id": source_pdf_id_from_key(source_key),
                "page_range": page_range_from_edge(edge),
                "all_pages_consecutive": "True" if str(edge.get("page_gap") or "") == "1" else "False",
                "n_members": 2,
                "n_physical_pages": 2,
                "status": refined.get("status", "manual_review_or_page_slicing_needed"),
                "confidence": "review",
                "static_site_action": refined.get("static_site_action", "show_unresolved_boundary"),
                "review_flags": refined.get("review_flags") or edge.get("v17_boundary_hint_type") or edge.get("sequence_review_flags", ""),
                "table_number_keys": uniq([edge.get("prev_table_number_key"), edge.get("table_number_key")], limit=4),
                "primary_title": edge.get("prev_primary_title") or edge.get("primary_title", ""),
                "example_images": uniq([edge.get("prev_example_images"), edge.get("example_images")], limit=4),
                "member_ids": uniq([edge.get("prev_physical_run_id"), edge.get("physical_run_id")], limit=4),
                "edge_ids": edge_id,
            }
        )
    return rows


def build_members(
    component_members: list[dict[str, str]],
    collection_members: list[dict[str, str]],
    variant_members: list[dict[str, str]],
    missing_bridges: list[dict[str, str]],
    unresolved: list[dict[str, str]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for member in component_members:
        rows.append(
            {
                "v21_group_id": physical_group_id(member.get("v11_component_id")),
                "group_type": "physical_component",
                "position": member.get("position_in_v11_component", ""),
                "member_type": "v10_sequence_group",
                "member_id": member.get("sequence_group_id", ""),
                "source_layer": "v17_component_members",
                "v5_source_key": member.get("v5_source_key", ""),
                "source_pdf": member.get("source_pdf", ""),
                "source_pdf_id": member.get("source_pdf_id", ""),
                "page_range": member.get("page_range", ""),
                "n_physical_pages": member.get("n_physical_pages", ""),
                "join_from_prev_action": member.get("v11_join_from_prev_action", ""),
                "join_from_prev_reason": member.get("v11_join_from_prev_reason", ""),
                "review_flags": member.get("sequence_review_flags", ""),
                "primary_title": member.get("primary_title", ""),
                "example_images": member.get("example_images", ""),
            }
        )
    for member in collection_members:
        rows.append(
            {
                "v21_group_id": collection_group_id(member.get("v17_collection_id")),
                "group_type": "v17_collection",
                "position": member.get("position_in_collection", ""),
                "member_type": "physical_component",
                "member_id": member.get("v11_component_id", ""),
                "source_layer": "v17_collection_members",
                "v5_source_key": member.get("v5_source_key", ""),
                "source_pdf": member.get("source_pdf", ""),
                "source_pdf_id": member.get("source_pdf_id", ""),
                "page_range": member.get("page_range", ""),
                "n_physical_pages": member.get("n_physical_pages", ""),
                "join_from_prev_action": member.get("join_from_prev_action", ""),
                "join_from_prev_reason": member.get("join_from_prev_reason", ""),
                "review_flags": member.get("v11_review_reasons", ""),
                "primary_title": member.get("primary_title", ""),
                "example_images": member.get("example_images", ""),
            }
        )
    for member in variant_members:
        member_id = member.get("physical_run_id") or member.get("v17_component_id") or ""
        rows.append(
            {
                "v21_group_id": variant_group_id(member.get("v19_variant_collection_id")),
                "group_type": "v19_logical_variant_collection",
                "position": member.get("position_in_variant_collection", ""),
                "member_type": member.get("member_type", ""),
                "member_id": member_id,
                "source_layer": "v19_variant_members",
                "v5_source_key": member.get("v5_source_key", ""),
                "source_pdf": member.get("source_pdf", ""),
                "source_pdf_id": member.get("source_pdf_id", ""),
                "page_range": member.get("page_range", ""),
                "n_physical_pages": member.get("n_physical_pages", ""),
                "join_from_prev_action": member.get("sequence_join_from_prev_type", ""),
                "join_from_prev_reason": member.get("sequence_join_from_prev_reason", ""),
                "review_flags": member.get("run_review_flags", ""),
                "primary_title": member.get("primary_title", ""),
                "example_images": member.get("example_images", ""),
            }
        )
    for source_rows, group_builder, default_label in [
        (missing_bridges, bridge_group_id, "missing_bridge_candidate"),
        (unresolved, unresolved_group_id, "unresolved_same_table_boundary"),
    ]:
        for edge in source_rows:
            group_id = group_builder(edge.get("sequence_edge_id"))
            source_key = edge.get("v5_source_key", "")
            refined = boundary_refinement(edge)
            label = refined.get("group_type", default_label)
            review_flags = refined.get("review_flags") or edge.get("sequence_review_flags", "")
            for pos, side in enumerate(["prev", "cur"], start=1):
                rows.append(
                    {
                        "v21_group_id": group_id,
                        "group_type": label,
                        "position": pos,
                        "member_type": "physical_run_boundary",
                        "member_id": edge.get("prev_physical_run_id") if side == "prev" else edge.get("physical_run_id"),
                        "source_layer": "v18_review_edges",
                        "v5_source_key": source_key,
                        "source_pdf": source_pdf_from_key(source_key),
                        "source_pdf_id": source_pdf_id_from_key(source_key),
                        "page_range": edge.get("prev_run_page_range") if side == "prev" else edge.get("run_page_range"),
                        "n_physical_pages": "1",
                        "join_from_prev_action": "" if side == "prev" else edge.get("v17_recommended_action", ""),
                        "join_from_prev_reason": "" if side == "prev" else edge.get("v17_recommended_reason", ""),
                        "review_flags": review_flags,
                        "primary_title": edge.get("prev_primary_title") if side == "prev" else edge.get("primary_title"),
                        "example_images": edge.get("prev_example_images") if side == "prev" else edge.get("example_images"),
                    }
                )
    return rows


def build_edges(
    physical_edges: list[dict[str, str]],
    collection_edges: list[dict[str, str]],
    variant_edges: list[dict[str, str]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for edge in physical_edges:
        source_key = edge.get("v5_source_key", "")
        action = str(edge.get("v17_recommended_action") or "")
        refined = boundary_refinement(edge)
        if refined:
            action = refined.get("decision", action)
        rows.append(
            {
                "v21_edge_id": f"physical_edge:{edge.get('sequence_edge_id')}",
                "edge_type": classify_physical_edge(edge),
                "source_layer": "v17_edge_recommendations",
                "source_edge_id": edge.get("sequence_edge_id", ""),
                "from_id": edge.get("prev_physical_run_id", ""),
                "to_id": edge.get("physical_run_id", ""),
                "v5_source_key": source_key,
                "source_pdf": source_pdf_from_key(source_key),
                "prev_page_range": edge.get("prev_run_page_range", ""),
                "page_range": edge.get("run_page_range", ""),
                "prev_last_pdf_page": edge.get("prev_last_pdf_page", ""),
                "first_pdf_page": edge.get("first_pdf_page", ""),
                "page_gap": edge.get("page_gap", ""),
                "missing_pages": edge.get("missing_pages", ""),
                "action": action,
                "confidence": refined.get("confidence", edge.get("v17_recommended_confidence", "")),
                "reason": refined.get("reason") or edge.get("v17_recommended_reason") or edge.get("sequence_edge_reason", ""),
                "review_flags": refined.get("review_flags") or edge.get("v17_boundary_hint_type") or edge.get("sequence_review_flags", ""),
                "strict_title_similarity": edge.get("strict_title_similarity", ""),
                "scope_similarity": edge.get("scope_similarity", ""),
                "prev_title": edge.get("prev_primary_title", ""),
                "title": edge.get("primary_title", ""),
                "prev_example_images": edge.get("prev_example_images", ""),
                "example_images": edge.get("example_images", ""),
            }
        )
    for edge in collection_edges:
        rows.append(
            {
                "v21_edge_id": f"collection_edge:{edge.get('collection_edge_id')}",
                "edge_type": "collection_boundary",
                "source_layer": "v17_collection_edges",
                "source_edge_id": edge.get("collection_edge_id", ""),
                "from_id": edge.get("prev_v17_component_id", ""),
                "to_id": edge.get("v17_component_id", ""),
                "v5_source_key": edge.get("v5_source_key", ""),
                "source_pdf": edge.get("source_pdf", ""),
                "prev_page_range": edge.get("prev_component_page_range", ""),
                "page_range": edge.get("component_page_range", ""),
                "prev_last_pdf_page": edge.get("prev_last_pdf_page", ""),
                "first_pdf_page": edge.get("first_pdf_page", ""),
                "page_gap": edge.get("page_gap", ""),
                "missing_pages": edge.get("missing_pages", ""),
                "action": edge.get("collection_edge_action", ""),
                "confidence": edge.get("collection_confidence", ""),
                "reason": edge.get("collection_edge_reason", ""),
                "review_flags": edge.get("collection_review_flags", ""),
                "strict_title_similarity": edge.get("strict_title_similarity", ""),
                "scope_similarity": edge.get("scope_similarity", ""),
                "prev_title": edge.get("prev_primary_title", ""),
                "title": edge.get("primary_title", ""),
                "prev_example_images": edge.get("prev_example_images", ""),
                "example_images": edge.get("example_images", ""),
            }
        )
    for edge in variant_edges:
        source_key = edge.get("v5_source_key", "")
        rows.append(
            {
                "v21_edge_id": f"variant_edge:{edge.get('v19_variant_collection_id')}:{edge.get('prev_member_id')}:{edge.get('member_id')}",
                "edge_type": "variant_member_boundary",
                "source_layer": "v19_variant_edges",
                "source_edge_id": edge.get("sequence_edge_id") or edge.get("collection_edge_id") or "",
                "from_id": edge.get("prev_member_id", ""),
                "to_id": edge.get("member_id", ""),
                "v5_source_key": source_key,
                "source_pdf": source_pdf_from_key(source_key),
                "prev_page_range": edge.get("prev_page_range", ""),
                "page_range": edge.get("page_range", ""),
                "prev_last_pdf_page": edge.get("prev_last_pdf_page", ""),
                "first_pdf_page": edge.get("first_pdf_page", ""),
                "page_gap": edge.get("page_gap", ""),
                "missing_pages": edge.get("missing_pages", ""),
                "action": edge.get("edge_action", ""),
                "confidence": "review",
                "reason": edge.get("edge_reason", ""),
                "review_flags": edge.get("review_flags", ""),
                "strict_title_similarity": edge.get("strict_title_similarity", ""),
                "scope_similarity": edge.get("scope_similarity", ""),
                "prev_title": "",
                "title": "",
                "prev_example_images": "",
                "example_images": "",
            }
        )
    return rows


def build_boundary_decisions(
    missing_bridges: list[dict[str, str]],
    unresolved: list[dict[str, str]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for edge in missing_bridges:
        source_key = edge.get("v5_source_key", "")
        rows.append(
            {
                "boundary_id": edge.get("sequence_edge_id", ""),
                "boundary_type": "missing_bridge_candidate",
                "decision": "missing_pages_required_before_join",
                "safe_whole_page_join": "no",
                "next_action": f"render/OCR missing PDF page(s): {edge.get('missing_pages', '')}",
                "confidence": "review",
                "v5_source_key": source_key,
                "source_pdf": source_pdf_from_key(source_key),
                "source_pdf_id": source_pdf_id_from_key(source_key),
                "prev_run_id": edge.get("prev_physical_run_id", ""),
                "run_id": edge.get("physical_run_id", ""),
                "prev_page_range": edge.get("prev_run_page_range", ""),
                "run_page_range": edge.get("run_page_range", ""),
                "page_range": page_range_from_edge(edge),
                "prev_last_pdf_page": edge.get("prev_last_pdf_page", ""),
                "first_pdf_page": edge.get("first_pdf_page", ""),
                "page_gap": edge.get("page_gap", ""),
                "missing_pages": edge.get("missing_pages", ""),
                "review_flags": edge.get("sequence_review_flags", ""),
                "reason": edge.get("sequence_edge_reason", ""),
                "visual_evidence": "",
                "prev_table_family_key": edge.get("prev_table_family_key", ""),
                "table_family_key": edge.get("table_family_key", ""),
                "prev_table_number_key": edge.get("prev_table_number_key", ""),
                "table_number_key": edge.get("table_number_key", ""),
                "strict_title_similarity": edge.get("strict_title_similarity", ""),
                "scope_similarity": edge.get("scope_similarity", ""),
                "prev_title": edge.get("prev_primary_title", ""),
                "title": edge.get("primary_title", ""),
                "prev_example_images": edge.get("prev_example_images", ""),
                "example_images": edge.get("example_images", ""),
            }
        )
    for edge in unresolved:
        source_key = edge.get("v5_source_key", "")
        refined = boundary_refinement(edge)
        rows.append(
            {
                "boundary_id": edge.get("sequence_edge_id", ""),
                "boundary_type": refined.get("group_type", "unresolved_same_table_boundary"),
                "decision": refined.get("decision", "manual_review_required"),
                "safe_whole_page_join": "no",
                "next_action": refined.get("next_action", "inspect visually before grouping"),
                "confidence": "visual_review" if refined else "review",
                "v5_source_key": source_key,
                "source_pdf": source_pdf_from_key(source_key),
                "source_pdf_id": source_pdf_id_from_key(source_key),
                "prev_run_id": edge.get("prev_physical_run_id", ""),
                "run_id": edge.get("physical_run_id", ""),
                "prev_page_range": edge.get("prev_run_page_range", ""),
                "run_page_range": edge.get("run_page_range", ""),
                "page_range": page_range_from_edge(edge),
                "prev_last_pdf_page": edge.get("prev_last_pdf_page", ""),
                "first_pdf_page": edge.get("first_pdf_page", ""),
                "page_gap": edge.get("page_gap", ""),
                "missing_pages": edge.get("missing_pages", ""),
                "review_flags": refined.get("review_flags") or edge.get("v17_boundary_hint_type") or edge.get("sequence_review_flags", ""),
                "reason": refined.get("reason") or edge.get("v17_recommended_reason") or edge.get("sequence_edge_reason", ""),
                "visual_evidence": refined.get("visual_evidence", ""),
                "prev_table_family_key": edge.get("prev_table_family_key", ""),
                "table_family_key": edge.get("table_family_key", ""),
                "prev_table_number_key": edge.get("prev_table_number_key", ""),
                "table_number_key": edge.get("table_number_key", ""),
                "strict_title_similarity": edge.get("strict_title_similarity", ""),
                "scope_similarity": edge.get("scope_similarity", ""),
                "prev_title": edge.get("prev_primary_title", ""),
                "title": edge.get("primary_title", ""),
                "prev_example_images": edge.get("prev_example_images", ""),
                "example_images": edge.get("example_images", ""),
            }
        )
    return rows


def package_visual_evidence(boundary_decisions: list[dict[str, object]]) -> None:
    evidence_dir = OUT_DIR / "visual_boundary_evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    for row in boundary_decisions:
        evidence = str(row.get("visual_evidence") or "")
        if not evidence:
            continue
        src = ROOT / evidence
        if not src.exists():
            continue
        dst = evidence_dir / src.name
        shutil.copy2(src, dst)
        row["visual_evidence"] = str(dst.relative_to(ROOT))


def add_page_mapping(mapping: dict[tuple[str, int], dict[str, list[str]]], source: str, source_pdf: str, source_pdf_id: str, page_range: str, key: str, value: str) -> None:
    for page in pages(page_range):
        row = mapping.setdefault(
            (source, page),
            {
                "v5_source_key": [source],
                "source_pdf": [source_pdf],
                "source_pdf_id": [source_pdf_id],
                "pdf_page": [str(page)],
                "physical_component_ids": [],
                "v17_collection_ids": [],
                "v19_variant_collection_ids": [],
                "physical_component_layers": [],
                "collection_kinds": [],
                "variant_sources": [],
                "status_flags": [],
                "primary_titles": [],
                "example_images": [],
            },
        )
        if value:
            row[key].append(value)


def build_page_index(
    components: list[dict[str, str]],
    collections: list[dict[str, str]],
    variants: list[dict[str, str]],
    groups: list[dict[str, object]],
) -> list[dict[str, object]]:
    mapping: dict[tuple[str, int], dict[str, list[str]]] = {}
    for component in components:
        source = str(component.get("v5_source_key") or "")
        add_page_mapping(mapping, source, component.get("source_pdf", ""), component.get("source_pdf_id", ""), component.get("page_range", ""), "physical_component_ids", component.get("v11_component_id", ""))
        for page in pages(component.get("page_range")):
            row = mapping[(source, page)]
            row["physical_component_layers"].append(str(component.get("v11_component_layer") or ""))
            row["primary_titles"].append(str(component.get("primary_title") or ""))
            row["example_images"].append(str(component.get("example_images") or ""))
    for collection in collections:
        source = str(collection.get("v5_source_key") or "")
        add_page_mapping(mapping, source, collection.get("source_pdf", ""), collection.get("source_pdf_id", ""), collection.get("page_range", ""), "v17_collection_ids", collection.get("v17_collection_id", ""))
        for page in pages(collection.get("page_range")):
            row = mapping[(source, page)]
            row["collection_kinds"].append(str(collection.get("collection_kind") or ""))
    for variant in variants:
        source = str(variant.get("v5_source_key") or "")
        add_page_mapping(mapping, source, variant.get("source_pdf", ""), variant.get("source_pdf_id", ""), variant.get("page_range", ""), "v19_variant_collection_ids", variant.get("v19_variant_collection_id", ""))
        for page in pages(variant.get("page_range")):
            row = mapping[(source, page)]
            row["variant_sources"].append(str(variant.get("variant_source") or ""))
    group_by_source_page: defaultdict[tuple[str, int], list[str]] = defaultdict(list)
    for group in groups:
        status = str(group.get("status") or "")
        group_type = str(group.get("group_type") or "")
        if status.startswith("review") or "needed" in status or "required" in status or group_type in {"section_split_candidate", "missing_bridge_candidate"}:
            for page in pages(group.get("page_range")):
                group_by_source_page[(str(group.get("v5_source_key") or ""), page)].append(group_type)
    rows: list[dict[str, object]] = []
    for key in sorted(mapping, key=lambda item: (item[0], item[1])):
        row = mapping[key]
        row["status_flags"].extend(group_by_source_page.get(key, []))
        rows.append({field: uniq(row.get(field, []), limit=40) for field in PAGE_INDEX_FIELDS})
    return rows


def validate(
    groups: list[dict[str, object]],
    members: list[dict[str, object]],
    edges: list[dict[str, object]],
    page_index: list[dict[str, object]],
    components: list[dict[str, str]],
    collections: list[dict[str, str]],
    variants: list[dict[str, str]],
    v18_invariants: list[dict[str, str]],
) -> list[dict[str, object]]:
    failures: list[tuple[str, int, list[object]]] = []
    group_ids = {str(group.get("v21_group_id") or "") for group in groups}
    physical_groups = {physical_group_id(component.get("v11_component_id")) for component in components}
    collection_groups = {collection_group_id(collection.get("v17_collection_id")) for collection in collections}
    variant_groups = {variant_group_id(variant.get("v19_variant_collection_id")) for variant in variants}
    checks = [
        ("all_physical_components_indexed", sorted(physical_groups - group_ids)),
        ("all_v17_collections_indexed", sorted(collection_groups - group_ids)),
        ("all_v19_variant_collections_indexed", sorted(variant_groups - group_ids)),
        ("members_reference_existing_groups", sorted({str(member.get("v21_group_id") or "") for member in members} - group_ids)),
        ("v18_invariants_still_pass", [row for row in v18_invariants if str(row.get("status") or "") != "pass"]),
    ]
    # Every numeric physical component page should be represented in the page index.
    indexed_pages = {(str(row.get("v5_source_key") or ""), v13.as_int(row.get("pdf_page"))) for row in page_index}
    missing_pages: list[object] = []
    for component in components:
        source = str(component.get("v5_source_key") or "")
        for page in pages(component.get("page_range")):
            if (source, page) not in indexed_pages:
                missing_pages.append({"v5_source_key": source, "pdf_page": page, "component": component.get("v11_component_id")})
    checks.append(("all_numeric_component_pages_indexed", missing_pages))
    for check, bad in checks:
        failures.append((check, len(bad), bad[:20]))
    return [
        {
            "check": check,
            "status": "pass" if count == 0 else "fail",
            "failure_count": count,
            "details": json.dumps(details, sort_keys=True),
        }
        for check, count, details in failures
    ]


def validate_boundary_refinements(
    boundary_decisions: list[dict[str, object]],
    unresolved: list[dict[str, str]],
) -> list[dict[str, object]]:
    checks: list[tuple[str, list[object]]] = []
    unresolved_without_refinement = [
        edge.get("sequence_edge_id", "")
        for edge in unresolved
        if not boundary_refinement(edge)
    ]
    checks.append(("all_current_unresolved_boundaries_have_v21_refinement", unresolved_without_refinement))

    unsafe_join_rows = [
        row.get("boundary_id", "")
        for row in boundary_decisions
        if str(row.get("safe_whole_page_join") or "").lower() != "no"
    ]
    checks.append(("all_boundary_decisions_block_whole_page_join", unsafe_join_rows))

    missing_visual_evidence = []
    for row in boundary_decisions:
        if row.get("decision") != "requires_section_split":
            continue
        evidence = str(row.get("visual_evidence") or "")
        if not evidence or not (ROOT / evidence).exists():
            missing_visual_evidence.append(row.get("boundary_id", ""))
    checks.append(("section_split_candidates_have_visual_contactsheet", missing_visual_evidence))

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
        "# v21 Boundary-Refined Master Assembly Index",
        "",
        "No OCR/API calls are made. This package is the canonical index over current assembly artifacts plus explicit boundary decisions.",
        "",
        "## Layers",
        "",
        "- `physical_component`: strict v17 physical assembly groups.",
        "- `v17_*`: collection/display groups over physical components.",
        "- `v19_logical_variant_collection`: decomposed sibling religion/subtable collections.",
        "- `missing_bridge_candidate`: likely assemblable after missing pages are available.",
        "- `section_split_candidate`: adjacent pages where only part of a page should logically group with its neighbor; do not whole-page merge.",
        "- `boundary_hard_break_confirmed`: visually confirmed table-scope switch; do not merge.",
        "- `unresolved_same_table_boundary`: adjacent mixed/page-slicing boundaries that are not safe whole-page joins and still need classification.",
        "",
        "## Boundary Rule",
        "",
        "A boundary may be a logical continuation without being a safe whole-page join. v21 keeps those as `section_split_candidate` records until a future page-splitting layer can assign page regions to separate logical groups.",
        "",
        "## Summary",
        "",
    ]
    for key in sorted(summary):
        lines.append(f"- `{key}`: {summary[key]}")
    (OUT_DIR / "v21_rules_readme.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    csv.field_size_limit(10**7)
    components = read_csv(V17_COMPONENTS)
    component_members = read_csv(V17_COMPONENT_MEMBERS)
    collections = read_csv(V17_COLLECTIONS)
    collection_members = read_csv(V17_COLLECTION_MEMBERS)
    physical_edges = read_csv(V17_EDGES)
    collection_edges = read_csv(V17_COLLECTION_EDGES)
    v18_invariants = read_csv(V18_INVARIANTS)
    required = read_csv(V18_REQUIRED)
    missing_bridges = read_csv(V18_MISSING_BRIDGES)
    unresolved = read_csv(V18_UNRESOLVED)
    variants = read_csv(V19_COLLECTIONS)
    variant_members = read_csv(V19_MEMBERS)
    variant_edges = read_csv(V19_EDGES)
    if not components or not collections or not variants:
        raise SystemExit("Missing v17/v19 inputs; run prior passes first")

    groups = [
        *build_physical_groups(components),
        *build_collection_groups(collections),
        *build_variant_groups(variants),
        *build_review_groups(missing_bridges, unresolved),
    ]
    members = build_members(component_members, collection_members, variant_members, missing_bridges, unresolved)
    edges = build_edges(physical_edges, collection_edges, variant_edges)
    boundary_decisions = build_boundary_decisions(missing_bridges, unresolved)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    package_visual_evidence(boundary_decisions)
    page_index = build_page_index(components, collections, variants, groups)
    audit = [
        *validate(groups, members, edges, page_index, components, collections, variants, v18_invariants),
        *validate_boundary_refinements(boundary_decisions, unresolved),
    ]
    audit_failures = sum(v13.as_int(row.get("failure_count")) for row in audit)
    group_counts = Counter(str(group.get("group_type") or "") for group in groups)
    edge_counts = Counter(str(edge.get("edge_type") or "") for edge in edges)
    boundary_decision_counts = Counter(str(row.get("decision") or "") for row in boundary_decisions)
    summary = {
        "v21_groups": len(groups),
        "v21_members": len(members),
        "v21_edges": len(edges),
        "v21_boundary_decisions": len(boundary_decisions),
        "v21_page_index_rows": len(page_index),
        "v21_group_type_counts": dict(group_counts),
        "v21_edge_type_counts": dict(edge_counts),
        "v21_boundary_decision_counts": dict(boundary_decision_counts),
        "v21_physical_components_indexed": len(components),
        "v21_v17_collections_indexed": len(collections),
        "v21_v19_variant_collections_indexed": len(variants),
        "v21_required_join_rows_carried": len(required),
        "v21_missing_bridge_groups": len(missing_bridges),
        "v21_section_split_candidate_groups": group_counts.get("section_split_candidate", 0),
        "v21_confirmed_hard_break_boundary_groups": group_counts.get("boundary_hard_break_confirmed", 0),
        "v21_unresolved_boundary_groups": group_counts.get("unresolved_same_table_boundary", 0),
        "v21_audit_failure_count": audit_failures,
        "v21_audit_status": "pass" if audit_failures == 0 else "fail",
    }

    base.write_csv(OUT_DIR / "v21_master_groups.csv", groups, GROUP_FIELDS)
    base.write_csv(OUT_DIR / "v21_master_members.csv", members, MEMBER_FIELDS)
    base.write_csv(OUT_DIR / "v21_master_edges.csv", edges, EDGE_FIELDS)
    base.write_csv(OUT_DIR / "v21_page_index.csv", page_index, PAGE_INDEX_FIELDS)
    base.write_csv(OUT_DIR / "v21_audit.csv", audit, AUDIT_FIELDS)
    base.write_csv(OUT_DIR / "v21_boundary_decisions.csv", boundary_decisions, BOUNDARY_DECISION_FIELDS)
    base.write_csv(OUT_DIR / "v21_missing_bridge_groups.csv", [group for group in groups if group.get("group_type") == "missing_bridge_candidate"], GROUP_FIELDS)
    base.write_csv(OUT_DIR / "v21_section_split_candidate_groups.csv", [group for group in groups if group.get("group_type") == "section_split_candidate"], GROUP_FIELDS)
    base.write_csv(OUT_DIR / "v21_confirmed_hard_break_boundary_groups.csv", [group for group in groups if group.get("group_type") == "boundary_hard_break_confirmed"], GROUP_FIELDS)
    base.write_csv(OUT_DIR / "v21_unresolved_boundary_groups.csv", [group for group in groups if group.get("group_type") == "unresolved_same_table_boundary"], GROUP_FIELDS)
    base.write_csv(OUT_DIR / "v21_variant_groups.csv", [group for group in groups if group.get("group_type") == "v19_logical_variant_collection"], GROUP_FIELDS)
    summary_rows = [{"metric": key, "value": json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value} for key, value in summary.items()]
    base.write_csv(OUT_DIR / "v21_summary.csv", summary_rows, SUMMARY_FIELDS)
    with (OUT_DIR / "v21_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    write_readme(summary)
    workbook = OUT_DIR / "v21_master_assembly_index.xlsx"
    base.write_workbook(
        workbook,
        {
            "summary": (summary_rows, SUMMARY_FIELDS),
            "master_groups": (groups, GROUP_FIELDS),
            "master_members": (members, MEMBER_FIELDS),
            "master_edges": (edges, EDGE_FIELDS),
            "boundary_decisions": (boundary_decisions, BOUNDARY_DECISION_FIELDS),
            "page_index": (page_index, PAGE_INDEX_FIELDS),
            "audit": (audit, AUDIT_FIELDS),
        },
    )
    with zipfile.ZipFile(workbook) as z:
        summary["v21_workbook_zip_bad_member"] = z.testzip()
    with (OUT_DIR / "v21_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
