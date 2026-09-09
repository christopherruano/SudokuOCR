#!/usr/bin/env python3
"""Physical assembly plus bounded logical collections.

No OCR/API calls are made. This pass consumes v16 outputs and preserves the
strict physical invariant: promoted physical joins must be adjacent real PDF
pages and resulting physical components must have no internal page gaps.

Compared with v16 this pass adds:

* one guarded physical right-half completion rule for a missed adjacent pair;
* a separate top-level collection layer for bounded logical sibling groups,
  such as adjacent religion/subtable variants, without treating those as
  physical row/column dependencies.
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import build_all_ocr_component_assembly_v5_structural as base
import build_all_ocr_component_assembly_v10_layered_manifest as v10
import build_all_ocr_component_assembly_v11_rule_audit as v11
import build_all_ocr_component_assembly_v13_scope_refined as v13
import build_all_ocr_component_assembly_v14_variant_refined as v14
import build_all_ocr_component_assembly_v16_left_stub_boundary_hints as v16


ROOT = Path(__file__).resolve().parent
V16_DIR = ROOT / "export_final/pdf_recovery_incoming_2026-06-16_all_ocr_component_assembly_v16_left_stub_boundary_hints"
OUT_DIR = ROOT / "export_final/pdf_recovery_incoming_2026-06-16_all_ocr_component_assembly_v17_collections"

V16_EDGES = V16_DIR / "v16_edge_recommendations.csv"
V16_MEMBERS = V16_DIR / "v16_member_classifier_overlay_passthrough.csv"
V16_COMPONENTS = V16_DIR / "v16_components.csv"
V16_COMPONENT_MEMBERS = V16_DIR / "v16_component_members.csv"

RIGHT_HALF_JOIN_ACTION = "join_current_right_half_same_scope_required_high"
LOGICAL_VARIANT_ACTION = "group_logical_variant_sequence_high"
JOIN_ACTIONS = {*v16.JOIN_ACTIONS, RIGHT_HALF_JOIN_ACTION}
COLLECTION_JOIN_ACTIONS = {LOGICAL_VARIANT_ACTION}

V17_EDGE_FIELDS = [
    *v16.V16_EDGE_FIELDS,
    "v17_recommended_action",
    "v17_recommended_layer",
    "v17_recommended_confidence",
    "v17_recommended_reason",
    "v17_review_priority",
    "v17_boundary_hint_type",
    "v17_boundary_hint_reason",
]

V17_COMPONENT_FIELDS = [
    *v16.V16_COMPONENT_FIELDS,
    "v17_component_join_actions",
]

V17_COMPONENT_MEMBER_FIELDS = [*v16.V16_COMPONENT_MEMBER_FIELDS]

V17_REVIEW_QUEUE_FIELDS = [
    "v17_review_priority",
    "v17_review_category",
    *[field for field in V17_EDGE_FIELDS if field != "v17_review_priority"],
]

V17_COLLECTION_EDGE_FIELDS = [
    "collection_edge_id",
    "sequence_edge_id",
    "prev_v17_component_id",
    "v17_component_id",
    "v5_source_key",
    "source_pdf",
    "prev_component_page_range",
    "component_page_range",
    "prev_last_pdf_page",
    "first_pdf_page",
    "page_gap",
    "missing_pages",
    "collection_edge_action",
    "collection_join_type",
    "collection_confidence",
    "collection_edge_reason",
    "collection_review_flags",
    "v17_recommended_action",
    "v17_boundary_hint_type",
    "prev_primary_title",
    "primary_title",
    "prev_example_images",
    "example_images",
    "strict_title_similarity",
    "scope_similarity",
]

V17_COLLECTION_FIELDS = [
    "v17_collection_id",
    "collection_kind",
    "static_site_action",
    "collection_confidence",
    "collection_review_flags",
    "v5_source_key",
    "source_pdf",
    "source_pdf_id",
    "n_v17_components",
    "n_v10_groups",
    "n_physical_pages",
    "page_range",
    "all_pages_consecutive",
    "v17_component_ids",
    "v17_component_layers",
    "v17_collection_edge_ids",
    "v17_collection_edge_actions",
    "table_number_keys",
    "primary_title",
    "example_images",
]

V17_COLLECTION_MEMBER_FIELDS = [
    "v17_collection_id",
    "position_in_collection",
    "join_from_prev_collection_edge_id",
    "join_from_prev_action",
    "join_from_prev_reason",
    *V17_COMPONENT_FIELDS,
]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def pages(value: object) -> list[int]:
    return v11.parse_pages(value)


def first_page(row: dict[str, object]) -> int:
    parsed = pages(row.get("page_range"))
    return parsed[0] if parsed else 10**9


def last_page(row: dict[str, object]) -> int:
    parsed = pages(row.get("page_range"))
    return parsed[-1] if parsed else -10**9


def single_component_page(component: dict[str, str] | None) -> bool:
    if not component:
        return False
    return v13.as_int(component.get("n_physical_pages")) == 1 and len(pages(component.get("page_range"))) == 1


def build_v16_context(
    components: list[dict[str, str]],
    component_members: list[dict[str, str]],
    members: list[dict[str, str]],
) -> dict[str, dict[str, str]]:
    component_by_id = {str(component.get("v11_component_id") or ""): component for component in components}
    seq_to_component = {
        str(member.get("sequence_group_id") or ""): str(member.get("v11_component_id") or "")
        for member in component_members
        if member.get("sequence_group_id") and member.get("v11_component_id")
    }
    run_to_seq = {
        str(member.get("physical_run_id") or ""): str(member.get("sequence_group_id") or "")
        for member in members
        if member.get("physical_run_id") and member.get("sequence_group_id")
    }
    run_to_component = {run: seq_to_component.get(seq, "") for run, seq in run_to_seq.items()}
    return {
        "component_by_id": component_by_id,
        "run_to_component": run_to_component,
    }


def current_right_half_same_scope_required_join(edge: dict[str, str], context: dict[str, dict[str, str]]) -> tuple[bool, str]:
    if str(edge.get("v16_recommended_action") or "") != "review_same_table_boundary_not_joined":
        return False, ""
    if not v16.page_gap_is_adjacent(edge):
        return False, "not_adjacent"
    if not v13.same_family_and_key(edge):
        return False, "different_table_family_or_number"

    flags = v16.row_layout_flags(edge)
    if not flags["cur_right_half"]:
        return False, "current_page_not_detected_as_right_half"
    if flags["cur_left_stub"]:
        return False, "current_page_starts_new_left_stub_block"

    component_by_id = context["component_by_id"]
    run_to_component = context["run_to_component"]
    prev_component = component_by_id.get(run_to_component.get(str(edge.get("prev_physical_run_id") or ""), ""))
    cur_component = component_by_id.get(run_to_component.get(str(edge.get("physical_run_id") or ""), ""))
    if not single_component_page(prev_component):
        return False, "previous_component_is_not_single_page_left_source"
    if not cur_component or first_page(cur_component) != v13.as_int(edge.get("first_pdf_page")):
        return False, "current_component_does_not_start_at_boundary_page"
    if not v16.has_local_scope_marker(edge.get("prev_primary_title")):
        return False, "previous_title_lacks_local_scope_marker"
    if v16.max_similarity(edge) < 0.74:
        return False, f"similarity_below_threshold_{v16.max_similarity(edge):.3f}"

    has_switch, switch_reason = v14.has_obvious_hard_switch(edge)
    if has_switch:
        return False, switch_reason

    return True, (
        "current_right_half_adjacent_to_single_page_same_scope_left_source; "
        f"similarity={v16.max_similarity(edge):.3f}; "
        f"prev_component={prev_component.get('v11_component_id') if prev_component else ''}; "
        f"cur_component={cur_component.get('v11_component_id') if cur_component else ''}"
    )


def refine_edge(edge: dict[str, str], context: dict[str, dict[str, str]]) -> dict[str, object]:
    action = str(edge.get("v16_recommended_action") or "")
    layer = str(edge.get("v16_recommended_layer") or "")
    confidence = str(edge.get("v16_recommended_confidence") or "")
    reason = str(edge.get("v16_recommended_reason") or "")
    priority = str(edge.get("v16_review_priority") or "")
    hint_type = str(edge.get("v16_boundary_hint_type") or "")
    hint_reason = str(edge.get("v16_boundary_hint_reason") or "")

    join, join_reason = current_right_half_same_scope_required_join(edge, context)
    if join:
        action = RIGHT_HALF_JOIN_ACTION
        layer = "required_right_half_completion_sequence"
        confidence = "high_review"
        reason = join_reason
        priority = ""
        hint_type = ""
        hint_reason = ""
    elif hint_type == "logical_variant_sequence_candidate" and action == "review_same_table_boundary_not_joined":
        action = "keep_logical_variant_collection_boundary"
        layer = "logical_variant_collection_boundary"
        confidence = "review"
        reason = "resolved_as_logical_variant_collection_not_physical_join; " + reason
        priority = ""

    return {
        **edge,
        "v17_recommended_action": action,
        "v17_recommended_layer": layer,
        "v17_recommended_confidence": confidence,
        "v17_recommended_reason": reason,
        "v17_review_priority": priority,
        "v17_boundary_hint_type": hint_type,
        "v17_boundary_hint_reason": hint_reason,
        # Reuse the v11 component builder with v17-refined physical joins.
        "v11_recommended_action": action,
        "v11_recommended_reason": reason,
    }


def review_queue(edges: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for edge in edges:
        priority = str(edge.get("v17_review_priority") or "")
        action = str(edge.get("v17_recommended_action") or "")
        if not priority:
            continue
        if action in JOIN_ACTIONS or action.startswith("hard_break"):
            continue
        rows.append(
            {
                "v17_review_priority": priority,
                "v17_review_category": action.replace("review_", ""),
                **edge,
            }
        )
    rows.sort(
        key=lambda row: (
            v13.as_int(row.get("v17_review_priority")) or 99,
            str(row.get("v5_source_key") or ""),
            v13.as_int(row.get("prev_last_pdf_page")) or 10**9,
            v13.as_int(row.get("first_pdf_page")) or 10**9,
        )
    )
    return rows


def label_components(components: list[dict[str, object]], edges: list[dict[str, object]]) -> None:
    v16.label_components(components, edges)
    edge_by_id = {str(edge.get("sequence_edge_id") or ""): edge for edge in edges}
    for component in components:
        component["v12_component_source"] = "v17_physical_plus_collections"
        join_edge_ids = [
            edge_id.strip()
            for edge_id in str(component.get("v11_join_edge_ids") or "").split(";")
            if edge_id.strip()
        ]
        actions = {str(edge_by_id.get(edge_id, {}).get("v17_recommended_action") or "") for edge_id in join_edge_ids}
        component["v17_component_join_actions"] = base.uniq_join(sorted(actions), limit=20)
        if RIGHT_HALF_JOIN_ACTION in actions:
            component["v11_component_layer"] = "required_right_half_completion_sequence"
            component["v11_static_site_action"] = "static_group_candidate_required_right_half_completion"
            component["v11_confidence"] = "high_review"


class UnionFind:
    def __init__(self, values: list[str]):
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left != root_right:
            self.parent[root_right] = root_left


def build_run_to_component(
    component_members: list[dict[str, object]],
    members: list[dict[str, str]],
) -> dict[str, str]:
    seq_to_component = {
        str(member.get("sequence_group_id") or ""): str(member.get("v11_component_id") or "")
        for member in component_members
        if member.get("sequence_group_id") and member.get("v11_component_id")
    }
    out: dict[str, str] = {}
    for member in members:
        run_id = str(member.get("physical_run_id") or "")
        seq_id = str(member.get("sequence_group_id") or "")
        if run_id and seq_id:
            out[run_id] = seq_to_component.get(seq_id, "")
    return out


def collection_edge_action(edge: dict[str, object], prev_component: dict[str, object], cur_component: dict[str, object]) -> tuple[str, str, str, str, str]:
    hint_type = str(edge.get("v17_boundary_hint_type") or "")
    action = str(edge.get("v17_recommended_action") or "")
    page_gap = str(edge.get("page_gap") or "")

    if (
        hint_type == "logical_variant_sequence_candidate"
        and page_gap == "1"
        and str(edge.get("v17_recommended_action") or "") == "keep_logical_variant_collection_boundary"
    ):
        return (
            LOGICAL_VARIANT_ACTION,
            "bounded_logical_variant_set",
            "review",
            "adjacent same-table logical variant sequence; keep physical components separate",
            "logical_variant_not_physical_dependency",
        )
    if action == "keep_missing_bridge":
        return (
            "review_missing_bridge_not_grouped",
            "missing_bridge_candidate",
            "review",
            str(edge.get("v17_recommended_reason") or edge.get("v16_recommended_reason") or "missing page blocks grouping"),
            "missing_intermediate_pages; do_not_join_across_gap",
        )
    if hint_type.startswith("mixed"):
        return (
            "review_mixed_page_boundary_not_grouped",
            "mixed_page_boundary",
            "review",
            str(edge.get("v17_boundary_hint_reason") or "mixed page boundary requires page-level slicing"),
            "mixed_page_boundary; do_not_whole_page_group",
        )
    if action == "review_same_table_boundary_not_joined":
        return (
            "review_same_table_boundary_not_grouped",
            "same_table_boundary",
            "review",
            str(edge.get("v17_recommended_reason") or edge.get("v16_recommended_reason") or "same-table boundary not grouped"),
            "same_table_unresolved",
        )
    return (
        "hard_break_or_unrelated_boundary",
        "",
        "high",
        str(edge.get("v17_recommended_reason") or edge.get("v16_recommended_reason") or edge.get("sequence_edge_reason") or ""),
        str(edge.get("sequence_review_flags") or ""),
    )


def build_collection_edges(
    components: list[dict[str, object]],
    component_members: list[dict[str, object]],
    members: list[dict[str, str]],
    edges: list[dict[str, object]],
) -> list[dict[str, object]]:
    component_by_id = {str(component.get("v11_component_id") or ""): component for component in components}
    run_to_component = build_run_to_component(component_members, members)
    rows: list[dict[str, object]] = []
    for edge in edges:
        prev_component_id = run_to_component.get(str(edge.get("prev_physical_run_id") or ""), "")
        cur_component_id = run_to_component.get(str(edge.get("physical_run_id") or ""), "")
        if not prev_component_id or not cur_component_id or prev_component_id == cur_component_id:
            continue
        prev_component = component_by_id.get(prev_component_id)
        cur_component = component_by_id.get(cur_component_id)
        if not prev_component or not cur_component:
            continue
        action, join_type, confidence, reason, flags = collection_edge_action(edge, prev_component, cur_component)
        rows.append(
            {
                "collection_edge_id": base.stable_id("coledge", prev_component_id, cur_component_id, edge.get("sequence_edge_id")),
                "sequence_edge_id": edge.get("sequence_edge_id", ""),
                "prev_v17_component_id": prev_component_id,
                "v17_component_id": cur_component_id,
                "v5_source_key": edge.get("v5_source_key", ""),
                "source_pdf": prev_component.get("source_pdf") or cur_component.get("source_pdf") or "",
                "prev_component_page_range": prev_component.get("page_range", ""),
                "component_page_range": cur_component.get("page_range", ""),
                "prev_last_pdf_page": edge.get("prev_last_pdf_page", ""),
                "first_pdf_page": edge.get("first_pdf_page", ""),
                "page_gap": edge.get("page_gap", ""),
                "missing_pages": edge.get("missing_pages", ""),
                "collection_edge_action": action,
                "collection_join_type": join_type,
                "collection_confidence": confidence,
                "collection_edge_reason": reason,
                "collection_review_flags": flags,
                "v17_recommended_action": edge.get("v17_recommended_action", ""),
                "v17_boundary_hint_type": edge.get("v17_boundary_hint_type", ""),
                "prev_primary_title": edge.get("prev_primary_title", ""),
                "primary_title": edge.get("primary_title", ""),
                "prev_example_images": edge.get("prev_example_images", ""),
                "example_images": edge.get("example_images", ""),
                "strict_title_similarity": edge.get("strict_title_similarity", ""),
                "scope_similarity": edge.get("scope_similarity", ""),
            }
        )
    rows.sort(
        key=lambda row: (
            str(row.get("v5_source_key") or ""),
            v13.as_int(row.get("prev_last_pdf_page")) or 10**9,
            v13.as_int(row.get("first_pdf_page")) or 10**9,
            str(row.get("collection_edge_id") or ""),
        )
    )
    return rows


def build_collections(
    components: list[dict[str, object]],
    collection_edges: list[dict[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    component_ids = [str(component.get("v11_component_id") or "") for component in components]
    component_by_id = {str(component.get("v11_component_id") or ""): component for component in components}
    uf = UnionFind(component_ids)
    joining_edges_by_pair: dict[tuple[str, str], dict[str, object]] = {}
    for edge in collection_edges:
        if str(edge.get("collection_edge_action") or "") not in COLLECTION_JOIN_ACTIONS:
            continue
        left = str(edge.get("prev_v17_component_id") or "")
        right = str(edge.get("v17_component_id") or "")
        if not left or not right or left == right:
            continue
        uf.union(left, right)
        joining_edges_by_pair[(left, right)] = edge

    grouped: dict[str, list[str]] = defaultdict(list)
    for component_id in component_ids:
        grouped[uf.find(component_id)].append(component_id)

    collections: list[dict[str, object]] = []
    members: list[dict[str, object]] = []
    for ids in grouped.values():
        ids = sorted(ids, key=lambda component_id: (str(component_by_id[component_id].get("v5_source_key") or ""), first_page(component_by_id[component_id]), component_id))
        first = component_by_id[ids[0]]
        pages_all: list[int] = []
        join_edges: list[dict[str, object]] = []
        layers: list[str] = []
        review_flags: list[str] = []
        n_v10_groups = 0
        n_physical_pages = 0
        for idx, component_id in enumerate(ids):
            component = component_by_id[component_id]
            pages_all.extend(pages(component.get("page_range")))
            layers.append(str(component.get("v11_component_layer") or ""))
            n_v10_groups += v13.as_int(component.get("n_v10_groups"))
            n_physical_pages += v13.as_int(component.get("n_physical_pages"))
            if str(component.get("v11_review_reasons") or ""):
                review_flags.append(str(component.get("v11_review_reasons") or ""))
            if idx > 0:
                edge = joining_edges_by_pair.get((ids[idx - 1], component_id), {})
                if edge:
                    join_edges.append(edge)
                    if str(edge.get("collection_review_flags") or ""):
                        review_flags.append(str(edge.get("collection_review_flags") or ""))

        pages_all = sorted(set(pages_all))
        all_consecutive = all(b == a + 1 for a, b in zip(pages_all, pages_all[1:]))
        actions = [str(edge.get("collection_edge_action") or "") for edge in join_edges]
        if len(ids) > 1:
            kind = "bounded_logical_variant_set"
            static_action = "show_bounded_logical_variant_set"
            confidence = "review"
        elif str(first.get("v11_component_layer") or "") == "bounded_variant_set":
            kind = "existing_bounded_variant_set"
            static_action = "show_existing_bounded_variant_set"
            confidence = str(first.get("v11_confidence") or "review")
        else:
            kind = "physical_component"
            static_action = str(first.get("v11_static_site_action") or "show_component")
            confidence = str(first.get("v11_confidence") or "")

        collection_id = base.stable_id("v17coll", ids[0], ids[-1], len(ids))
        collections.append(
            {
                "v17_collection_id": collection_id,
                "collection_kind": kind,
                "static_site_action": static_action,
                "collection_confidence": confidence,
                "collection_review_flags": base.uniq_join(review_flags, limit=30),
                "v5_source_key": first.get("v5_source_key", ""),
                "source_pdf": first.get("source_pdf", ""),
                "source_pdf_id": first.get("source_pdf_id", ""),
                "n_v17_components": len(ids),
                "n_v10_groups": n_v10_groups,
                "n_physical_pages": n_physical_pages,
                "page_range": v11.page_range_from_pages(pages_all),
                "all_pages_consecutive": str(all_consecutive),
                "v17_component_ids": base.uniq_join(ids, limit=60),
                "v17_component_layers": base.uniq_join(layers, limit=30),
                "v17_collection_edge_ids": base.uniq_join([edge.get("collection_edge_id") for edge in join_edges], limit=30),
                "v17_collection_edge_actions": base.uniq_join(actions, limit=30),
                "table_number_keys": base.uniq_join([component_by_id[component_id].get("table_number_keys") for component_id in ids], limit=30),
                "primary_title": first.get("primary_title", ""),
                "example_images": base.uniq_join([component_by_id[component_id].get("example_images") for component_id in ids], limit=12),
            }
        )

        for pos, component_id in enumerate(ids, start=1):
            edge = joining_edges_by_pair.get((ids[pos - 2], component_id), {}) if pos > 1 else {}
            members.append(
                {
                    "v17_collection_id": collection_id,
                    "position_in_collection": pos,
                    "join_from_prev_collection_edge_id": edge.get("collection_edge_id", ""),
                    "join_from_prev_action": edge.get("collection_edge_action", ""),
                    "join_from_prev_reason": edge.get("collection_edge_reason", ""),
                    **component_by_id[component_id],
                }
            )

    collections.sort(key=lambda row: (str(row.get("v5_source_key") or ""), first_page(row), str(row.get("v17_collection_id") or "")))
    members.sort(key=lambda row: (str(row.get("v17_collection_id") or ""), v13.as_int(row.get("position_in_collection"))))
    return collections, members


def validate(
    components: list[dict[str, object]],
    edges: list[dict[str, object]],
    collections: list[dict[str, object]],
    collection_edges: list[dict[str, object]],
) -> dict[str, object]:
    failures: list[str] = []
    for component in components:
        if v13.as_int(component.get("n_v10_groups")) > 1 and str(component.get("all_pages_consecutive")) != "True":
            failures.append(f"v17_component_internal_gap:{component.get('v11_component_id')}:{component.get('page_range')}")
    for edge in edges:
        if str(edge.get("v17_recommended_action") or "") in JOIN_ACTIONS and str(edge.get("page_gap") or "") != "1":
            failures.append(f"v17_join_non_adjacent:{edge.get('sequence_edge_id')}:{edge.get('page_gap')}")
    for collection in collections:
        if v13.as_int(collection.get("n_v17_components")) > 1 and str(collection.get("all_pages_consecutive")) != "True":
            failures.append(f"v17_collection_internal_gap:{collection.get('v17_collection_id')}:{collection.get('page_range')}")
    for edge in collection_edges:
        if str(edge.get("collection_edge_action") or "") in COLLECTION_JOIN_ACTIONS and str(edge.get("page_gap") or "") != "1":
            failures.append(f"v17_collection_join_non_adjacent:{edge.get('collection_edge_id')}:{edge.get('page_gap')}")
        if str(edge.get("collection_edge_action") or "") in COLLECTION_JOIN_ACTIONS and str(edge.get("v17_boundary_hint_type") or "").startswith("mixed"):
            failures.append(f"v17_collection_join_crosses_mixed_boundary:{edge.get('collection_edge_id')}")
    return {
        "v17_invariant_failure_count": len(failures),
        "v17_invariant_failures": failures[:100],
    }


def write_readme(summary: dict[str, object]) -> None:
    lines = [
        "# v17 Physical Components and Logical Collections",
        "",
        "No OCR/API calls are made. This pass refines v16 and adds a separate collection layer.",
        "",
        "## Physical Layer",
        "",
        "- Physical joins remain strict: adjacent real PDF pages only, no internal component gaps.",
        "- Added `join_current_right_half_same_scope_required_high` for right-half pages completing an immediately previous single-page same-scope table page.",
        "",
        "## Collection Layer",
        "",
        "- Logical collections can group consecutive sibling components, such as adjacent religion/subtable variants.",
        "- Logical collections do not erase physical component boundaries.",
        "- Mixed-page and missing-page boundaries are recorded but not grouped.",
        "",
        "## Summary",
        "",
    ]
    for key in sorted(summary):
        lines.append(f"- `{key}`: {summary[key]}")
    (OUT_DIR / "v17_rules_readme.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    csv.field_size_limit(10**7)
    groups = read_csv(v13.V10_GROUPS)
    members = read_csv(V16_MEMBERS)
    v16_edges = read_csv(V16_EDGES)
    v16_components = read_csv(V16_COMPONENTS)
    v16_component_members = read_csv(V16_COMPONENT_MEMBERS)
    if not groups or not members or not v16_edges or not v16_components or not v16_component_members:
        raise SystemExit("Missing v10/v16 inputs; run v16 first")

    context = build_v16_context(v16_components, v16_component_members, members)
    refined_edges = [refine_edge(edge, context) for edge in v16_edges]
    components, component_members = v11.build_v11_components(groups, members, refined_edges, join_actions=JOIN_ACTIONS)
    label_components(components, refined_edges)
    queue = review_queue(refined_edges)
    collection_edges = build_collection_edges(components, component_members, members, refined_edges)
    collections, collection_members = build_collections(components, collection_edges)

    edge_actions = Counter(str(edge.get("v17_recommended_action") or "") for edge in refined_edges)
    component_layers = Counter(str(component.get("v11_component_layer") or "") for component in components)
    collection_actions = Counter(str(edge.get("collection_edge_action") or "") for edge in collection_edges)
    collection_kinds = Counter(str(collection.get("collection_kind") or "") for collection in collections)
    summary = {
        "input_v10_groups": len(groups),
        "input_v16_members": len(members),
        "input_v16_edges": len(v16_edges),
        "v17_components": len(components),
        "v17_component_members": len(component_members),
        "v17_edge_rows": len(refined_edges),
        "v17_review_queue_edges": len(queue),
        "v17_collections": len(collections),
        "v17_collection_members": len(collection_members),
        "v17_collection_edges": len(collection_edges),
        "v17_promoted_right_half_joins": edge_actions.get(RIGHT_HALF_JOIN_ACTION, 0),
        "v17_promoted_left_stub_joins": edge_actions.get(v16.LEFT_STUB_JOIN_ACTION, 0),
        "v17_logical_variant_collection_edges": collection_actions.get(LOGICAL_VARIANT_ACTION, 0),
        "v17_logical_variant_boundaries_kept_nonphysical": edge_actions.get("keep_logical_variant_collection_boundary", 0),
        "v17_kept_missing_bridges": edge_actions.get("keep_missing_bridge", 0),
        "v17_remaining_same_table_review_edges": edge_actions.get("review_same_table_boundary_not_joined", 0),
        "v17_component_layer_counts": dict(component_layers),
        "v17_edge_action_counts": dict(edge_actions),
        "v17_collection_action_counts": dict(collection_actions),
        "v17_collection_kind_counts": dict(collection_kinds),
    }
    summary.update(validate(components, refined_edges, collections, collection_edges))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base.write_csv(OUT_DIR / "v17_edge_recommendations.csv", refined_edges, V17_EDGE_FIELDS)
    base.write_csv(OUT_DIR / "v17_components.csv", components, V17_COMPONENT_FIELDS)
    base.write_csv(OUT_DIR / "v17_component_members.csv", component_members, V17_COMPONENT_MEMBER_FIELDS)
    base.write_csv(OUT_DIR / "v17_edge_review_queue.csv", queue, V17_REVIEW_QUEUE_FIELDS)
    base.write_csv(OUT_DIR / "v17_collection_edges.csv", collection_edges, V17_COLLECTION_EDGE_FIELDS)
    base.write_csv(OUT_DIR / "v17_collections.csv", collections, V17_COLLECTION_FIELDS)
    base.write_csv(OUT_DIR / "v17_collection_members.csv", collection_members, V17_COLLECTION_MEMBER_FIELDS)
    logical_variant_edges = [edge for edge in collection_edges if edge.get("collection_edge_action") == LOGICAL_VARIANT_ACTION]
    logical_variant_collections = [collection for collection in collections if collection.get("collection_kind") in {"bounded_logical_variant_set", "existing_bounded_variant_set"}]
    base.write_csv(OUT_DIR / "v17_logical_variant_collection_edges.csv", logical_variant_edges, V17_COLLECTION_EDGE_FIELDS)
    base.write_csv(OUT_DIR / "v17_logical_variant_collections.csv", logical_variant_collections, V17_COLLECTION_FIELDS)
    base.write_csv(OUT_DIR / "v17_member_classifier_overlay_passthrough.csv", members, v11.V11_MEMBER_OVERLAY_FIELDS)
    base.write_csv(OUT_DIR / "v17_required_physical_review_passthrough.csv", read_csv(v13.V10_REQUIRED_CANDIDATES), v10.V10_REQUIRED_CANDIDATE_FIELDS)
    base.write_csv(OUT_DIR / "v17_required_physical_edges_passthrough.csv", read_csv(v13.V10_REQUIRED_EDGES), v10.V10_REQUIRED_EDGE_FIELDS)
    bridges = read_csv(v13.V10_BRIDGES)
    overassoc = read_csv(v13.V10_OVERASSOC)
    base.write_csv(OUT_DIR / "v17_missing_bridge_candidates_passthrough.csv", bridges, list(bridges[0].keys()) if bridges else [])
    base.write_csv(OUT_DIR / "v17_old_overassociation_audit_passthrough.csv", overassoc, list(overassoc[0].keys()) if overassoc else [])

    with (OUT_DIR / "v17_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    write_readme(summary)

    summary_rows = [
        {"metric": key, "value": json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value}
        for key, value in summary.items()
    ]
    base.write_workbook(
        OUT_DIR / "v17_collections.xlsx",
        {
            "summary": (summary_rows, ["metric", "value"]),
            "physical_components": (components, V17_COMPONENT_FIELDS),
            "physical_component_members": (component_members, V17_COMPONENT_MEMBER_FIELDS),
            "physical_edge_recommendations": (refined_edges, V17_EDGE_FIELDS),
            "physical_edge_review_queue": (queue, V17_REVIEW_QUEUE_FIELDS),
            "collections": (collections, V17_COLLECTION_FIELDS),
            "collection_members": (collection_members, V17_COLLECTION_MEMBER_FIELDS),
            "collection_edges": (collection_edges, V17_COLLECTION_EDGE_FIELDS),
            "logical_variant_collections": (logical_variant_collections, V17_COLLECTION_FIELDS),
            "logical_variant_edges": (logical_variant_edges, V17_COLLECTION_EDGE_FIELDS),
            "required_physical_review": (read_csv(v13.V10_REQUIRED_CANDIDATES), v10.V10_REQUIRED_CANDIDATE_FIELDS),
        },
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
