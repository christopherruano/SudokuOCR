#!/usr/bin/env python3
"""Build a conservative first-principles seed assembly index.

This does not trust current static-site groupings or older component labels.
It starts from physical pages sorted by real PDF page and promotes only
ironclad adjacent page pairs into physical assemblies. Larger assemblies may
emerge later only by adding more accepted adjacent edges.

No OCR/API calls are made. Existing local OCR/profile/layout artifacts are read
only through the v6 page/edge inventory.
"""

from __future__ import annotations

import csv
import json
import re
import zipfile
from collections import Counter, defaultdict, deque
from pathlib import Path

import xlsxwriter

import build_all_ocr_component_assembly_v5_structural as base


ROOT = Path(__file__).resolve().parent
V6_DIR = ROOT / "export_final/pdf_recovery_incoming_2026-06-16_all_ocr_component_assembly_v6_layout_refined"
OUT_DIR = ROOT / "export_final/pdf_recovery_incoming_2026-06-16_first_principles_seed_assembly_v25"

V6_PAGES = V6_DIR / "v6_physical_pages.csv"
V6_EDGES = V6_DIR / "v6_physical_edges.csv"
V6_COLLECTIONS = V6_DIR / "v6_collection_groups.csv"
V6_MISSING = V6_DIR / "v6_missing_gap_candidates.csv"

BAD_PHYSICAL_PAGE_KINDS = {"context_or_blank", "offtarget_table", "unknown", "derivative_selected_caste_race_tribe"}

EDGE_FIELDS = [
    "v25_edge_id",
    "v25_decision",
    "v25_decision_layer",
    "v25_confidence",
    "v25_reason",
    "v25_review_flags",
    "v5_source_key",
    "prev_physical_page_id",
    "physical_page_id",
    "prev_pdf_page",
    "pdf_page",
    "page_gap",
    "missing_pages",
    "prev_page_kind",
    "page_kind",
    "prev_table_family_key",
    "table_family_key",
    "prev_table_number_key",
    "table_number_key",
    "schema_reason",
    "section_reason",
    "title_similarity",
    "collection_candidate",
    "collection_reason",
    "collection_review_flags",
    "prev_title",
    "title",
    "prev_row_label_ratio",
    "row_label_ratio",
    "prev_first_row_labels",
    "first_row_labels",
    "prev_last_row_labels",
    "last_row_labels",
    "prev_image_path",
    "image_path",
]

COMPONENT_FIELDS = [
    "v25_component_id",
    "component_kind",
    "v25_confidence",
    "v5_source_key",
    "source_pdf",
    "source_pdf_id",
    "year",
    "province",
    "page_range",
    "n_physical_pages",
    "all_pages_consecutive",
    "join_edge_ids",
    "page_kind_counts",
    "table_family_counts",
    "table_number_keys",
    "primary_title",
    "component_review_flags",
    "example_images",
]

MEMBER_FIELDS = [
    "v25_component_id",
    "position_in_component",
    "physical_page_id",
    "v5_source_key",
    "source_pdf",
    "source_pdf_id",
    "year",
    "province",
    "pdf_page",
    "page_kind",
    "table_family_key",
    "table_number_key",
    "title",
    "image_path",
    "ocr_artifact",
    "profile_row_label_ratio",
    "profile_first_row_labels",
    "profile_last_row_labels",
    "layout_left_stub_candidate",
    "layout_right_half_candidate",
    "quality_flags",
    "join_from_prev_edge_id",
    "join_from_prev_reason",
]

GROWTH_FIELDS = [
    "candidate_id",
    "candidate_type",
    "recommended_action",
    "v5_source_key",
    "prev_pdf_page",
    "pdf_page",
    "missing_pages",
    "prev_physical_page_id",
    "physical_page_id",
    "prev_component_id",
    "component_id",
    "reason",
    "review_flags",
    "prev_title",
    "title",
    "prev_image_path",
    "image_path",
]

LOGICAL_FIELDS = [
    "collection_group_id",
    "collection_basis",
    "v25_candidate_type",
    "recommended_action",
    "v5_source_key",
    "source_pdf",
    "source_pdf_id",
    "n_physical_runs",
    "n_physical_pages",
    "page_ranges",
    "collection_review_flags",
    "primary_title",
    "collection_reasons",
    "example_images",
]

AUDIT_FIELDS = ["check", "status", "failure_count", "details"]
SUMMARY_FIELDS = ["metric", "value"]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def int_or_none(value: object) -> int | None:
    return base.int_or_none(value)


def uniq_join(values, limit: int = 30) -> str:
    return base.uniq_join(values, limit=limit)


def compact_range(pages: list[int]) -> str:
    values = sorted(set(pages))
    if not values:
        return ""
    ranges: list[str] = []
    start = prev = values[0]
    for value in values[1:]:
        if value == prev + 1:
            prev = value
            continue
        ranges.append(str(start) if start == prev else f"{start}-{prev}")
        start = prev = value
    ranges.append(str(start) if start == prev else f"{start}-{prev}")
    return "; ".join(ranges)


def parse_title_similarity(edge: dict[str, str]) -> float:
    reason = str(edge.get("edge_reason") or "")
    match = re.search(r"title_similarity_([0-9.]+)", reason)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return 0.0
    try:
        return float(edge.get("title_similarity") or 0)
    except ValueError:
        return 0.0


def classify_edge(edge: dict[str, str]) -> dict[str, object]:
    page_gap = int_or_none(edge.get("page_gap"))
    prev_kind = str(edge.get("prev_page_kind") or "")
    cur_kind = str(edge.get("page_kind") or "")
    reason = str(edge.get("edge_reason") or "")
    flags = str(edge.get("edge_review_flags") or "")
    title_similarity = parse_title_similarity(edge)

    if page_gap != 1:
        if edge.get("missing_pages") and edge.get("collection_candidate") == "yes":
            return {
                "v25_decision": "missing_bridge_candidate",
                "v25_decision_layer": "missing_pages",
                "v25_confidence": "review_after_missing_pages_exist",
                "v25_reason": "related pages are not physically adjacent; render/OCR missing pages before any assembly",
                "v25_review_flags": "missing_intermediate_pages",
            }
        return {
            "v25_decision": "break",
            "v25_decision_layer": "hard_boundary",
            "v25_confidence": "not_joinable_now",
            "v25_reason": "not_adjacent_in_real_pdf_page_order",
            "v25_review_flags": "non_adjacent_or_missing_page_order",
        }

    if prev_kind != cur_kind:
        return {
            "v25_decision": "break",
            "v25_decision_layer": "hard_boundary",
            "v25_confidence": "not_physical_join",
            "v25_reason": f"page_kind_switch_{prev_kind}_to_{cur_kind}",
            "v25_review_flags": "page_class_switch",
        }
    if cur_kind in BAD_PHYSICAL_PAGE_KINDS:
        return {
            "v25_decision": "break",
            "v25_decision_layer": "hard_boundary",
            "v25_confidence": "not_physical_join",
            "v25_reason": f"page_kind_not_allowed_for_automatic_physical_assembly:{cur_kind}",
            "v25_review_flags": "context_derivative_or_offtarget_page",
        }

    if edge.get("edge_action") != "join":
        if edge.get("collection_candidate") == "yes":
            return {
                "v25_decision": "logical_growth_candidate",
                "v25_decision_layer": "bounded_collection_candidate",
                "v25_confidence": "review_only",
                "v25_reason": "adjacent related pages, but no ironclad physical dependency",
                "v25_review_flags": edge.get("collection_review_flags", "") or "logical_not_physical",
            }
        return {
            "v25_decision": "break",
            "v25_decision_layer": "hard_boundary",
            "v25_confidence": "not_physical_join",
            "v25_reason": reason,
            "v25_review_flags": flags,
        }

    if "adjacent_image_layout_halfspread" in reason and title_similarity >= 0.88:
        return {
            "v25_decision": "seed_physical_join",
            "v25_decision_layer": "ironclad_layout_halfspread",
            "v25_confidence": "ironclad_seed",
            "v25_reason": f"adjacent PDF pages; same table subject; left/right halfspread geometry; title_similarity={title_similarity:.2f}",
            "v25_review_flags": "",
        }
    if "wide_table_right_half_requires_previous_row_labels" in reason and title_similarity >= 0.55:
        return {
            "v25_decision": "seed_physical_join",
            "v25_decision_layer": "ironclad_row_label_dependency",
            "v25_confidence": "ironclad_seed",
            "v25_reason": "current page lacks row labels and requires previous page row-label column",
            "v25_review_flags": "",
        }
    if "same_schema_current_row_labels_deficient" in reason:
        return {
            "v25_decision": "seed_physical_join",
            "v25_decision_layer": "ironclad_same_schema_row_label_dependency",
            "v25_confidence": "ironclad_seed",
            "v25_reason": "same schema and current page row labels are deficient",
            "v25_review_flags": "",
        }
    if "same_section_age_rows_continue" in reason:
        return {
            "v25_decision": "seed_physical_join",
            "v25_decision_layer": "ironclad_age_row_sequence",
            "v25_confidence": "ironclad_seed",
            "v25_reason": "age row labels continue within same schema/section",
            "v25_review_flags": "",
        }
    return {
        "v25_decision": "growth_review_candidate",
        "v25_decision_layer": "v6_join_not_ironclad_seed",
        "v25_confidence": "review_only",
        "v25_reason": f"v6 proposed a join, but seed layer requires stronger evidence; title_similarity={title_similarity:.2f}",
        "v25_review_flags": flags or "v6_join_demoted_to_review",
    }


def build_seed_edges(edges: list[dict[str, str]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for edge in edges:
        decision = classify_edge(edge)
        edge_id = base.stable_id(
            "v25edge",
            edge.get("v5_source_key"),
            edge.get("prev_physical_page_id"),
            edge.get("physical_page_id"),
            edge.get("prev_pdf_page"),
            edge.get("pdf_page"),
        )
        out.append({"v25_edge_id": edge_id, **decision, **edge})
    out.sort(
        key=lambda row: (
            str(row.get("v5_source_key") or ""),
            int_or_none(row.get("prev_pdf_page")) or 10**9,
            int_or_none(row.get("pdf_page")) or 10**9,
            str(row.get("v25_edge_id") or ""),
        )
    )
    return out


def page_order(row: dict[str, object]) -> tuple[str, int, str]:
    return (
        str(row.get("v5_source_key") or ""),
        int_or_none(row.get("pdf_page")) or 10**9,
        str(row.get("physical_page_id") or ""),
    )


def build_components(
    pages: list[dict[str, str]],
    seed_edges: list[dict[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, str]]:
    page_by_id = {str(row.get("physical_page_id") or ""): row for row in pages}
    adjacency: defaultdict[str, set[str]] = defaultdict(set)
    edge_by_pair: dict[tuple[str, str], dict[str, object]] = {}
    for edge in seed_edges:
        if edge.get("v25_decision") != "seed_physical_join":
            continue
        left = str(edge.get("prev_physical_page_id") or "")
        right = str(edge.get("physical_page_id") or "")
        if left and right:
            adjacency[left].add(right)
            adjacency[right].add(left)
            edge_by_pair[(left, right)] = edge
            edge_by_pair[(right, left)] = edge

    seen: set[str] = set()
    components: list[list[str]] = []
    for page in sorted(pages, key=page_order):
        page_id = str(page.get("physical_page_id") or "")
        if not page_id or page_id in seen:
            continue
        queue = deque([page_id])
        seen.add(page_id)
        group: list[str] = []
        while queue:
            cur = queue.popleft()
            group.append(cur)
            for nxt in sorted(adjacency[cur], key=lambda pid: page_order(page_by_id.get(pid, {}))):
                if nxt not in seen:
                    seen.add(nxt)
                    queue.append(nxt)
        components.append(group)

    component_rows: list[dict[str, object]] = []
    member_rows: list[dict[str, object]] = []
    page_to_component: dict[str, str] = {}
    for group in components:
        group_pages = sorted([page_by_id[pid] for pid in group if pid in page_by_id], key=page_order)
        nums = [int_or_none(row.get("pdf_page")) for row in group_pages]
        page_nums = [n for n in nums if n is not None]
        source = str(group_pages[0].get("v5_source_key") or "") if group_pages else ""
        join_edges: list[dict[str, object]] = []
        for left, right in zip(group_pages, group_pages[1:]):
            edge = edge_by_pair.get((str(left.get("physical_page_id") or ""), str(right.get("physical_page_id") or "")))
            if edge:
                join_edges.append(edge)
        kind = "ironclad_physical_assembly" if len(group_pages) > 1 else "singleton_unassembled_page"
        component_id = base.stable_id(
            "v25comp",
            source,
            compact_range(page_nums),
            len(group_pages),
            group_pages[0].get("physical_page_id", "") if group_pages else "",
        )
        for pos, page in enumerate(group_pages, start=1):
            page_to_component[str(page.get("physical_page_id") or "")] = component_id
            prev_edge = join_edges[pos - 2] if pos > 1 and pos - 2 < len(join_edges) else {}
            member_rows.append(
                {
                    "v25_component_id": component_id,
                    "position_in_component": pos,
                    "physical_page_id": page.get("physical_page_id", ""),
                    "v5_source_key": page.get("v5_source_key", ""),
                    "source_pdf": page.get("source_pdf", ""),
                    "source_pdf_id": page.get("source_pdf_id", ""),
                    "year": page.get("year", ""),
                    "province": page.get("province", ""),
                    "pdf_page": page.get("pdf_page", ""),
                    "page_kind": page.get("page_kind", ""),
                    "table_family_key": page.get("table_family_key", ""),
                    "table_number_key": page.get("table_number_key", ""),
                    "title": page.get("title", ""),
                    "image_path": page.get("image_path", ""),
                    "ocr_artifact": page.get("ocr_artifact", ""),
                    "profile_row_label_ratio": page.get("profile_row_label_ratio", ""),
                    "profile_first_row_labels": page.get("profile_first_row_labels", ""),
                    "profile_last_row_labels": page.get("profile_last_row_labels", ""),
                    "layout_left_stub_candidate": page.get("layout_left_stub_candidate", ""),
                    "layout_right_half_candidate": page.get("layout_right_half_candidate", ""),
                    "quality_flags": page.get("quality_flags", ""),
                    "join_from_prev_edge_id": prev_edge.get("v25_edge_id", ""),
                    "join_from_prev_reason": prev_edge.get("v25_reason", ""),
                }
            )
        component_rows.append(
            {
                "v25_component_id": component_id,
                "component_kind": kind,
                "v25_confidence": "ironclad_seed" if len(group_pages) > 1 else "single_page",
                "v5_source_key": source,
                "source_pdf": uniq_join([row.get("source_pdf", "") for row in group_pages], limit=6),
                "source_pdf_id": uniq_join([row.get("source_pdf_id", "") for row in group_pages], limit=6),
                "year": uniq_join([row.get("year", "") for row in group_pages], limit=6),
                "province": uniq_join([row.get("province", "") for row in group_pages], limit=6),
                "page_range": compact_range(page_nums),
                "n_physical_pages": len(group_pages),
                "all_pages_consecutive": "yes" if page_nums and page_nums == list(range(min(page_nums), max(page_nums) + 1)) else "no" if len(group_pages) > 1 else "yes",
                "join_edge_ids": uniq_join([edge.get("v25_edge_id", "") for edge in join_edges], limit=20),
                "page_kind_counts": json.dumps(dict(Counter(str(row.get("page_kind") or "") for row in group_pages)), sort_keys=True),
                "table_family_counts": json.dumps(dict(Counter(str(row.get("table_family_key") or "") for row in group_pages)), sort_keys=True),
                "table_number_keys": uniq_join([row.get("table_number_key", "") for row in group_pages], limit=10),
                "primary_title": Counter(str(row.get("title") or "") for row in group_pages if row.get("title")).most_common(1)[0][0] if any(row.get("title") for row in group_pages) else "",
                "component_review_flags": uniq_join([row.get("quality_flags", "") for row in group_pages], limit=30),
                "example_images": uniq_join([row.get("image_path", "") for row in group_pages], limit=8),
            }
        )

    component_rows.sort(key=lambda row: (str(row.get("v5_source_key") or ""), int_or_none(str(row.get("page_range") or "").split(";", 1)[0].split("-", 1)[0]) or 10**9, str(row.get("v25_component_id") or "")))
    member_rows.sort(key=lambda row: (str(row.get("v25_component_id") or ""), int(row.get("position_in_component") or 0)))
    return component_rows, member_rows, page_to_component


def build_growth_candidates(
    seed_edges: list[dict[str, object]],
    missing_rows: list[dict[str, str]],
    page_to_component: dict[str, str],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for edge in seed_edges:
        if edge.get("v25_decision") not in {"growth_review_candidate", "logical_growth_candidate"}:
            continue
        rows.append(
            {
                "candidate_id": edge.get("v25_edge_id", ""),
                "candidate_type": edge.get("v25_decision", ""),
                "recommended_action": "manual_review_before_promoting_to_seed_edge",
                "v5_source_key": edge.get("v5_source_key", ""),
                "prev_pdf_page": edge.get("prev_pdf_page", ""),
                "pdf_page": edge.get("pdf_page", ""),
                "missing_pages": edge.get("missing_pages", ""),
                "prev_physical_page_id": edge.get("prev_physical_page_id", ""),
                "physical_page_id": edge.get("physical_page_id", ""),
                "prev_component_id": page_to_component.get(str(edge.get("prev_physical_page_id") or ""), ""),
                "component_id": page_to_component.get(str(edge.get("physical_page_id") or ""), ""),
                "reason": edge.get("v25_reason", ""),
                "review_flags": edge.get("v25_review_flags", ""),
                "prev_title": edge.get("prev_title", ""),
                "title": edge.get("title", ""),
                "prev_image_path": edge.get("prev_image_path", ""),
                "image_path": edge.get("image_path", ""),
            }
        )
    for row in missing_rows:
        rows.append(
            {
                "candidate_id": base.stable_id("v25miss", row.get("v5_source_key"), row.get("prev_physical_page_id"), row.get("physical_page_id"), row.get("missing_pages")),
                "candidate_type": "missing_bridge_candidate",
                "recommended_action": "render_or_OCR_missing_pages_before_assembly",
                "v5_source_key": row.get("v5_source_key", ""),
                "prev_pdf_page": row.get("prev_pdf_page", ""),
                "pdf_page": row.get("pdf_page", ""),
                "missing_pages": row.get("missing_pages", ""),
                "prev_physical_page_id": row.get("prev_physical_page_id", ""),
                "physical_page_id": row.get("physical_page_id", ""),
                "prev_component_id": page_to_component.get(str(row.get("prev_physical_page_id") or ""), ""),
                "component_id": page_to_component.get(str(row.get("physical_page_id") or ""), ""),
                "reason": row.get("collection_reason", "") or row.get("edge_reason", ""),
                "review_flags": "missing_intermediate_pages",
                "prev_title": row.get("prev_title", ""),
                "title": row.get("title", ""),
                "prev_image_path": row.get("prev_image_path", ""),
                "image_path": row.get("image_path", ""),
            }
        )
    rows.sort(key=lambda row: (str(row.get("candidate_type") or ""), str(row.get("v5_source_key") or ""), int_or_none(row.get("prev_pdf_page")) or 10**9))
    return rows


def logical_candidate_rows(collections: list[dict[str, str]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in collections:
        n_runs = int_or_none(row.get("n_physical_runs")) or 0
        if n_runs <= 1:
            continue
        flags = str(row.get("collection_review_flags") or "")
        candidate_type = "logical_same_table_growth_candidate"
        if "religion_variant_same_base_title" in flags:
            candidate_type = "logical_religion_or_part_sibling_candidate"
        elif "self_contained_continued_subtable" in flags:
            candidate_type = "self_contained_continuation_sequence_candidate"
        elif "collection_crosses_missing_pages" in flags:
            candidate_type = "logical_candidate_needs_missing_pages_review"
        rows.append(
            {
                "collection_group_id": row.get("collection_group_id", ""),
                "collection_basis": row.get("collection_basis", ""),
                "v25_candidate_type": candidate_type,
                "recommended_action": "logical_group_only_do_not_physically_merge",
                "v5_source_key": row.get("v5_source_key", ""),
                "source_pdf": row.get("source_pdf", ""),
                "source_pdf_id": row.get("source_pdf_id", ""),
                "n_physical_runs": row.get("n_physical_runs", ""),
                "n_physical_pages": row.get("n_physical_pages", ""),
                "page_ranges": row.get("page_ranges", ""),
                "collection_review_flags": row.get("collection_review_flags", ""),
                "primary_title": row.get("primary_title", ""),
                "collection_reasons": row.get("collection_reasons", ""),
                "example_images": row.get("example_images", ""),
            }
        )
    rows.sort(key=lambda r: (str(r.get("v25_candidate_type") or ""), str(r.get("v5_source_key") or ""), str(r.get("page_ranges") or "")))
    return rows


def validate(
    pages: list[dict[str, str]],
    seed_edges: list[dict[str, object]],
    components: list[dict[str, object]],
    members: list[dict[str, object]],
) -> list[dict[str, object]]:
    checks: list[tuple[str, list[object]]] = []
    bad_seed_edges = [
        edge.get("v25_edge_id")
        for edge in seed_edges
        if edge.get("v25_decision") == "seed_physical_join" and str(edge.get("page_gap")) != "1"
    ]
    checks.append(("seed_physical_edges_are_adjacent_pdf_pages", bad_seed_edges))
    bad_component_gaps = [
        row.get("v25_component_id")
        for row in components
        if row.get("component_kind") == "ironclad_physical_assembly" and row.get("all_pages_consecutive") != "yes"
    ]
    checks.append(("seed_physical_components_have_no_internal_gaps", bad_component_gaps))
    bad_large_without_edges = [
        row.get("v25_component_id")
        for row in components
        if row.get("component_kind") == "ironclad_physical_assembly"
        and int(row.get("n_physical_pages") or 0) > 1
        and len([edge for edge in str(row.get("join_edge_ids") or "").split(";") if edge.strip()]) != int(row.get("n_physical_pages") or 0) - 1
    ]
    checks.append(("each_multi_page_seed_component_has_pairwise_join_edges", bad_large_without_edges))
    page_ids = {str(row.get("physical_page_id") or "") for row in pages if row.get("physical_page_id")}
    member_page_ids = [str(row.get("physical_page_id") or "") for row in members if row.get("physical_page_id")]
    missing_pages = sorted(page_ids - set(member_page_ids))
    duplicate_members = [page for page, count in Counter(member_page_ids).items() if count > 1]
    checks.append(("all_physical_pages_assigned_to_one_seed_component", missing_pages))
    checks.append(("no_physical_page_assigned_to_multiple_seed_components", duplicate_members))
    hard_auto_derivative = [
        row.get("v25_component_id")
        for row in components
        if row.get("component_kind") == "ironclad_physical_assembly"
        and "derivative_selected_caste_race_tribe" in str(row.get("page_kind_counts") or "")
    ]
    checks.append(("no_derivative_pages_in_automatic_seed_physical_assemblies", hard_auto_derivative))
    return [
        {
            "check": name,
            "status": "pass" if not failures else "fail",
            "failure_count": len(failures),
            "details": json.dumps(failures[:50], sort_keys=True),
        }
        for name, failures in checks
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
                ws.write(r_idx, c_idx, row.get(field, ""), wrap if "reason" in field or "flags" in field or field in {"title", "primary_title", "example_images"} else None)
        ws.freeze_panes(1, 0)
        ws.autofilter(0, 0, max(1, len(rows)), max(0, len(fields) - 1))
        for col, field in enumerate(fields):
            width = 18
            if "reason" in field or "flags" in field or field in {"title", "primary_title"}:
                width = 44
            elif "image" in field or field.endswith("ids"):
                width = 58
            ws.set_column(col, col, width)
    workbook.close()


def main() -> None:
    csv.field_size_limit(10**7)
    pages = read_csv(V6_PAGES)
    edges = read_csv(V6_EDGES)
    collections = read_csv(V6_COLLECTIONS)
    missing_rows = read_csv(V6_MISSING)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    seed_edges = build_seed_edges(edges)
    components, members, page_to_component = build_components(pages, seed_edges)
    growth = build_growth_candidates(seed_edges, missing_rows, page_to_component)
    logical = logical_candidate_rows(collections)
    audit = validate(pages, seed_edges, components, members)

    summary = {
        "v25_input_physical_pages": len(pages),
        "v25_input_pairwise_edges": len(edges),
        "v25_seed_physical_join_edges": sum(1 for edge in seed_edges if edge.get("v25_decision") == "seed_physical_join"),
        "v25_seed_components_total": len(components),
        "v25_ironclad_physical_assembly_components": sum(1 for row in components if row.get("component_kind") == "ironclad_physical_assembly"),
        "v25_singleton_components": sum(1 for row in components if row.get("component_kind") == "singleton_unassembled_page"),
        "v25_max_seed_component_pages": max((int(row.get("n_physical_pages") or 0) for row in components), default=0),
        "v25_growth_candidates": len(growth),
        "v25_missing_bridge_candidates": sum(1 for row in growth if row.get("candidate_type") == "missing_bridge_candidate"),
        "v25_logical_collection_candidates": len(logical),
        "v25_edge_decision_counts": dict(Counter(str(edge.get("v25_decision") or "") for edge in seed_edges)),
        "v25_edge_decision_layer_counts": dict(Counter(str(edge.get("v25_decision_layer") or "") for edge in seed_edges)),
        "v25_logical_candidate_type_counts": dict(Counter(str(row.get("v25_candidate_type") or "") for row in logical)),
        "v25_audit_failure_count": sum(int(row.get("failure_count") or 0) for row in audit),
        "v25_audit_status": "pass" if all(row.get("status") == "pass" for row in audit) else "fail",
    }
    summary_rows = [
        {"metric": key, "value": json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value}
        for key, value in summary.items()
    ]

    write_csv(OUT_DIR / "v25_pairwise_seed_edges.csv", seed_edges, EDGE_FIELDS)
    write_csv(OUT_DIR / "v25_seed_components.csv", components, COMPONENT_FIELDS)
    write_csv(OUT_DIR / "v25_seed_component_members.csv", members, MEMBER_FIELDS)
    write_csv(OUT_DIR / "v25_ironclad_physical_assemblies.csv", [row for row in components if row.get("component_kind") == "ironclad_physical_assembly"], COMPONENT_FIELDS)
    write_csv(OUT_DIR / "v25_growth_candidates.csv", growth, GROWTH_FIELDS)
    write_csv(OUT_DIR / "v25_logical_collection_candidates.csv", logical, LOGICAL_FIELDS)
    write_csv(OUT_DIR / "v25_audit.csv", audit, AUDIT_FIELDS)
    write_csv(OUT_DIR / "v25_summary.csv", summary_rows, SUMMARY_FIELDS)
    with (OUT_DIR / "v25_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    workbook = OUT_DIR / "v25_first_principles_seed_assembly.xlsx"
    write_workbook(
        workbook,
        {
            "summary": (summary_rows, SUMMARY_FIELDS),
            "ironclad_assemblies": ([row for row in components if row.get("component_kind") == "ironclad_physical_assembly"], COMPONENT_FIELDS),
            "seed_components": (components, COMPONENT_FIELDS),
            "component_members": (members, MEMBER_FIELDS),
            "pairwise_edges": (seed_edges, EDGE_FIELDS),
            "growth_candidates": (growth, GROWTH_FIELDS),
            "logical_candidates": (logical, LOGICAL_FIELDS),
            "audit": (audit, AUDIT_FIELDS),
        },
    )
    with zipfile.ZipFile(workbook) as zf:
        summary["v25_workbook_zip_bad_member"] = zf.testzip()
    with (OUT_DIR / "v25_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
