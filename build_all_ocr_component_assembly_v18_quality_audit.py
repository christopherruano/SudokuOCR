#!/usr/bin/env python3
"""Quality audit for v17 assembly outputs.

No OCR/API calls are made. This pass does not create new joins; it verifies and
documents whether the current assembly satisfies the key assembly constraints:

* every known required physical row-label/right-half edge is covered inside a
  v17 physical component;
* no promoted physical join is non-adjacent;
* no multi-run physical component or logical collection crosses a page gap;
* unresolved same-table boundaries are explicitly reviewable and not silently
  merged;
* old broad over-associations remain visible for comparison.
"""

from __future__ import annotations

import csv
import json
import zipfile
from collections import Counter
from pathlib import Path

import build_all_ocr_component_assembly_v5_structural as base
import build_all_ocr_component_assembly_v11_rule_audit as v11
import build_all_ocr_component_assembly_v13_scope_refined as v13
import build_all_ocr_component_assembly_v17_collections as v17


ROOT = Path(__file__).resolve().parent
V17_DIR = ROOT / "export_final/pdf_recovery_incoming_2026-06-16_all_ocr_component_assembly_v17_collections"
OUT_DIR = ROOT / "export_final/pdf_recovery_incoming_2026-06-16_all_ocr_component_assembly_v18_quality_audit"

V17_COMPONENTS = V17_DIR / "v17_components.csv"
V17_COLLECTIONS = V17_DIR / "v17_collections.csv"
V17_EDGES = V17_DIR / "v17_edge_recommendations.csv"
V17_QUEUE = V17_DIR / "v17_edge_review_queue.csv"
V17_COLLECTION_EDGES = V17_DIR / "v17_collection_edges.csv"
V17_LOGICAL_VARIANTS = V17_DIR / "v17_logical_variant_collection_edges.csv"
V17_REQUIRED_REVIEW = V17_DIR / "v17_required_physical_review_passthrough.csv"
V17_REQUIRED_EDGES = V17_DIR / "v17_required_physical_edges_passthrough.csv"
V17_OVERASSOC = V17_DIR / "v17_old_overassociation_audit_passthrough.csv"

REQUIRED_JOIN_COVERAGE_FIELDS = [
    "coverage_status",
    "v5_source_key",
    "prev_pdf_page",
    "pdf_page",
    "page_gap",
    "edge_action",
    "edge_reason",
    "edge_review_flags",
    "prev_v17_component_ids",
    "v17_component_ids",
    "shared_v17_component_ids",
    "shared_component_page_ranges",
    "prev_title",
    "title",
    "prev_row_label_ratio",
    "row_label_ratio",
    "prev_first_row_labels",
    "first_row_labels",
]

REQUIRED_CANDIDATE_AUDIT_FIELDS = [
    "candidate_status",
    "candidate_priority",
    "candidate_type",
    "candidate_reason",
    "v5_source_key",
    "prev_pdf_page",
    "pdf_page",
    "page_gap",
    "edge_action",
    "edge_reason",
    "edge_review_flags",
    "v17_edge_action",
    "v17_edge_reason",
    "v17_boundary_hint_type",
    "prev_v17_component_ids",
    "v17_component_ids",
    "shared_v17_component_ids",
    "prev_title",
    "title",
    "prev_row_label_ratio",
    "row_label_ratio",
]

INVARIANT_AUDIT_FIELDS = [
    "check",
    "status",
    "failure_count",
    "details",
]

OVERASSOC_SUMMARY_FIELDS = [
    "metric",
    "value",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def pages(value: object) -> list[int]:
    return v11.parse_pages(value)


def page_range_for_component(component: dict[str, str]) -> str:
    return str(component.get("page_range") or "")


def build_page_index(components: list[dict[str, str]]) -> tuple[dict[str, dict[str, str]], dict[tuple[str, int], set[str]]]:
    component_by_id = {str(component.get("v11_component_id") or ""): component for component in components}
    page_index: dict[tuple[str, int], set[str]] = {}
    for component in components:
        source = str(component.get("v5_source_key") or "")
        component_id = str(component.get("v11_component_id") or "")
        for page in pages(component.get("page_range")):
            page_index.setdefault((source, page), set()).add(component_id)
    return component_by_id, page_index


def component_ids_for_page(page_index: dict[tuple[str, int], set[str]], source: str, page: object) -> set[str]:
    page_num = v13.as_int(page)
    if not source or not page_num:
        return set()
    return page_index.get((source, page_num), set())


def shared_page_ranges(component_by_id: dict[str, dict[str, str]], component_ids: set[str]) -> str:
    return base.uniq_join(
        [page_range_for_component(component_by_id.get(component_id, {})) for component_id in sorted(component_ids)],
        limit=20,
    )


def audit_required_join_coverage(
    required_edges: list[dict[str, str]],
    component_by_id: dict[str, dict[str, str]],
    page_index: dict[tuple[str, int], set[str]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for edge in required_edges:
        if str(edge.get("edge_action") or "") != "join":
            continue
        source = str(edge.get("v5_source_key") or "")
        prev_ids = component_ids_for_page(page_index, source, edge.get("prev_pdf_page"))
        cur_ids = component_ids_for_page(page_index, source, edge.get("pdf_page"))
        shared_ids = prev_ids & cur_ids
        if shared_ids:
            status = "covered_same_v17_component"
        elif not prev_ids or not cur_ids:
            status = "not_covered_page_missing_from_v17_components"
        else:
            status = "not_covered_split_across_v17_components"
        rows.append(
            {
                "coverage_status": status,
                "prev_v17_component_ids": base.uniq_join(sorted(prev_ids), limit=20),
                "v17_component_ids": base.uniq_join(sorted(cur_ids), limit=20),
                "shared_v17_component_ids": base.uniq_join(sorted(shared_ids), limit=20),
                "shared_component_page_ranges": shared_page_ranges(component_by_id, shared_ids),
                **edge,
            }
        )
    return rows


def edge_lookup(edges: list[dict[str, str]]) -> dict[tuple[str, str, str], dict[str, str]]:
    return {
        (str(edge.get("v5_source_key") or ""), str(edge.get("prev_last_pdf_page") or ""), str(edge.get("first_pdf_page") or "")): edge
        for edge in edges
    }


def audit_required_candidates(
    candidates: list[dict[str, str]],
    edges: list[dict[str, str]],
    page_index: dict[tuple[str, int], set[str]],
) -> list[dict[str, object]]:
    by_boundary = edge_lookup(edges)
    rows: list[dict[str, object]] = []
    for candidate in candidates:
        source = str(candidate.get("v5_source_key") or "")
        prev_page = str(candidate.get("prev_pdf_page") or "")
        cur_page = str(candidate.get("pdf_page") or "")
        prev_ids = component_ids_for_page(page_index, source, prev_page)
        cur_ids = component_ids_for_page(page_index, source, cur_page)
        shared_ids = prev_ids & cur_ids
        edge = by_boundary.get((source, prev_page, cur_page), {})
        edge_action = str(edge.get("v17_recommended_action") or "")
        if shared_ids:
            status = "covered_same_v17_component"
        elif edge_action.startswith("hard_break"):
            status = "rejected_by_v17_hard_break"
        elif edge_action == "keep_missing_bridge":
            status = "not_joined_missing_bridge"
        elif edge_action == "review_same_table_boundary_not_joined":
            status = "pending_same_table_review"
        elif edge_action:
            status = f"not_joined_v17_action_{edge_action}"
        else:
            status = "boundary_not_found_in_v17_edges"
        rows.append(
            {
                "candidate_status": status,
                "v17_edge_action": edge_action,
                "v17_edge_reason": edge.get("v17_recommended_reason", ""),
                "v17_boundary_hint_type": edge.get("v17_boundary_hint_type", ""),
                "prev_v17_component_ids": base.uniq_join(sorted(prev_ids), limit=20),
                "v17_component_ids": base.uniq_join(sorted(cur_ids), limit=20),
                "shared_v17_component_ids": base.uniq_join(sorted(shared_ids), limit=20),
                **candidate,
            }
        )
    return rows


def invariant_audit(
    components: list[dict[str, str]],
    collections: list[dict[str, str]],
    edges: list[dict[str, str]],
    collection_edges: list[dict[str, str]],
    required_join_coverage: list[dict[str, object]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    join_actions = {
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
    non_adjacent = [
        edge for edge in edges
        if str(edge.get("v17_recommended_action") or "") in join_actions and str(edge.get("page_gap") or "") not in {"", "1"}
    ]
    component_gaps = [
        component for component in components
        if v13.as_int(component.get("n_v10_groups")) > 1 and str(component.get("all_pages_consecutive") or "") != "True"
    ]
    collection_gaps = [
        collection for collection in collections
        if v13.as_int(collection.get("n_v17_components")) > 1 and str(collection.get("all_pages_consecutive") or "") != "True"
    ]
    mixed_collection_joins = [
        edge for edge in collection_edges
        if str(edge.get("collection_edge_action") or "") == "group_logical_variant_sequence_high"
        and str(edge.get("v17_boundary_hint_type") or "").startswith("mixed")
    ]
    missing_required_joins = [row for row in required_join_coverage if not str(row.get("coverage_status") or "").startswith("covered")]
    checks = [
        ("physical_joins_adjacent", non_adjacent),
        ("physical_components_no_internal_gaps", component_gaps),
        ("logical_collections_no_internal_gaps", collection_gaps),
        ("logical_collections_do_not_cross_mixed_boundaries", mixed_collection_joins),
        ("known_required_physical_joins_covered", missing_required_joins),
    ]
    for check, failures in checks:
        rows.append(
            {
                "check": check,
                "status": "pass" if not failures else "fail",
                "failure_count": len(failures),
                "details": json.dumps(failures[:20], sort_keys=True),
            }
        )
    return rows


def overassoc_summary(overassoc_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    flag_counts: Counter[str] = Counter()
    total_missing = 0
    old_pages = 0
    split_sequence_groups = 0
    for row in overassoc_rows:
        for flag in str(row.get("audit_flags") or "").split(";"):
            flag = flag.strip()
            if flag:
                flag_counts[flag] += 1
        total_missing += v13.as_int(row.get("old_missing_gap_count"))
        old_pages += v13.as_int(row.get("old_n_physical_pages"))
        split_sequence_groups += v13.as_int(row.get("v7_sequence_group_count"))
    return [
        {"metric": "old_overassociated_groups_audited", "value": len(overassoc_rows)},
        {"metric": "old_physical_pages_in_overassociated_groups", "value": old_pages},
        {"metric": "new_adjacency_sequence_groups_after_split", "value": split_sequence_groups},
        {"metric": "total_missing_pages_crossed_by_old_groups", "value": total_missing},
        {"metric": "old_overassociation_flag_counts", "value": json.dumps(dict(flag_counts), sort_keys=True)},
    ]


def write_readme(summary: dict[str, object]) -> None:
    lines = [
        "# v18 Assembly Quality Audit",
        "",
        "No OCR/API calls are made. This pass verifies v17 assembly and records the evidence needed before using it downstream.",
        "",
        "## Gate Checks",
        "",
        "- Physical joins must be adjacent real PDF pages.",
        "- Multi-run physical components must not have internal page gaps.",
        "- Logical collections must not have internal page gaps.",
        "- Logical collections must not cross mixed-page boundaries.",
        "- Known required physical joins from the row-label/right-half passthrough must be covered in v17 components.",
        "",
        "## Summary",
        "",
    ]
    for key in sorted(summary):
        lines.append(f"- `{key}`: {summary[key]}")
    (OUT_DIR / "v18_rules_readme.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    csv.field_size_limit(10**7)
    components = read_csv(V17_COMPONENTS)
    collections = read_csv(V17_COLLECTIONS)
    edges = read_csv(V17_EDGES)
    queue = read_csv(V17_QUEUE)
    collection_edges = read_csv(V17_COLLECTION_EDGES)
    logical_variants = read_csv(V17_LOGICAL_VARIANTS)
    required_edges = read_csv(V17_REQUIRED_EDGES)
    required_candidates = read_csv(V17_REQUIRED_REVIEW)
    overassoc = read_csv(V17_OVERASSOC)
    if not components or not collections or not edges:
        raise SystemExit("Missing v17 inputs; run v17 first")

    component_by_id, page_index = build_page_index(components)
    required_join_coverage = audit_required_join_coverage(required_edges, component_by_id, page_index)
    required_candidate_audit = audit_required_candidates(required_candidates, edges, page_index)
    invariant_rows = invariant_audit(components, collections, edges, collection_edges, required_join_coverage)
    overassoc_rows = overassoc_summary(overassoc)
    unresolved_same_table = [row for row in queue if row.get("v17_recommended_action") == "review_same_table_boundary_not_joined"]
    missing_bridges = [row for row in queue if row.get("v17_recommended_action") == "keep_missing_bridge"]

    invariant_failures = sum(v13.as_int(row.get("failure_count")) for row in invariant_rows)
    coverage_counts = Counter(str(row.get("coverage_status") or "") for row in required_join_coverage)
    candidate_counts = Counter(str(row.get("candidate_status") or "") for row in required_candidate_audit)
    summary = {
        "v18_invariant_failure_count": invariant_failures,
        "v18_invariant_status": "pass" if invariant_failures == 0 else "fail",
        "v18_required_join_edges": len(required_join_coverage),
        "v18_required_join_coverage_counts": dict(coverage_counts),
        "v18_required_candidate_rows": len(required_candidate_audit),
        "v18_required_candidate_status_counts": dict(candidate_counts),
        "v18_unresolved_same_table_boundaries": len(unresolved_same_table),
        "v18_missing_bridge_review_edges": len(missing_bridges),
        "v18_logical_variant_collection_edges": len(logical_variants),
        "v18_overassociation_audit_rows": len(overassoc),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base.write_csv(OUT_DIR / "v18_required_join_coverage.csv", required_join_coverage, REQUIRED_JOIN_COVERAGE_FIELDS)
    base.write_csv(OUT_DIR / "v18_required_candidate_audit.csv", required_candidate_audit, REQUIRED_CANDIDATE_AUDIT_FIELDS)
    base.write_csv(OUT_DIR / "v18_invariant_audit.csv", invariant_rows, INVARIANT_AUDIT_FIELDS)
    base.write_csv(OUT_DIR / "v18_unresolved_same_table_boundaries.csv", unresolved_same_table, v17.V17_REVIEW_QUEUE_FIELDS)
    base.write_csv(OUT_DIR / "v18_missing_bridge_review_edges.csv", missing_bridges, v17.V17_REVIEW_QUEUE_FIELDS)
    base.write_csv(OUT_DIR / "v18_logical_variant_collection_edges.csv", logical_variants, v17.V17_COLLECTION_EDGE_FIELDS)
    base.write_csv(OUT_DIR / "v18_overassociation_summary.csv", overassoc_rows, OVERASSOC_SUMMARY_FIELDS)
    base.write_csv(OUT_DIR / "v18_old_overassociation_audit_passthrough.csv", overassoc, list(overassoc[0].keys()) if overassoc else [])

    with (OUT_DIR / "v18_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    write_readme(summary)

    summary_rows = [
        {"metric": key, "value": json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value}
        for key, value in summary.items()
    ]
    workbook = OUT_DIR / "v18_quality_audit.xlsx"
    base.write_workbook(
        workbook,
        {
            "summary": (summary_rows, ["metric", "value"]),
            "invariant_audit": (invariant_rows, INVARIANT_AUDIT_FIELDS),
            "required_join_coverage": (required_join_coverage, REQUIRED_JOIN_COVERAGE_FIELDS),
            "required_candidate_audit": (required_candidate_audit, REQUIRED_CANDIDATE_AUDIT_FIELDS),
            "unresolved_same_table": (unresolved_same_table, v17.V17_REVIEW_QUEUE_FIELDS),
            "missing_bridges": (missing_bridges, v17.V17_REVIEW_QUEUE_FIELDS),
            "logical_variant_edges": (logical_variants, v17.V17_COLLECTION_EDGE_FIELDS),
            "overassociation_summary": (overassoc_rows, OVERASSOC_SUMMARY_FIELDS),
        },
    )
    with zipfile.ZipFile(workbook) as z:
        summary["v18_workbook_zip_bad_member"] = z.testzip()
    with (OUT_DIR / "v18_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
