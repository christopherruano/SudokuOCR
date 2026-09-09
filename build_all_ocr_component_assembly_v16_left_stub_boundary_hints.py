#!/usr/bin/env python3
"""Left-stub and mixed-boundary refinement of v15 assembly.

No OCR/API calls are made. This pass consumes v15 outputs and keeps the hard
invariant that promoted joins must be adjacent real PDF pages.

This pass adds one narrow join:

* previous page/run is explicitly detected as a left-stub/body-left page;
* current page is adjacent and same table family/number;
* current page does not itself start a new left-stub block;
* title/scope similarity remains plausible and no hard table/scope switch is
  detected.

Everything else remains separated. Ambiguous adjacent same-table boundaries are
exported as mixed-page or variant-sequence hints for manual/site review.
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path

import build_all_ocr_component_assembly_v5_structural as base
import build_all_ocr_component_assembly_v10_layered_manifest as v10
import build_all_ocr_component_assembly_v11_rule_audit as v11
import build_all_ocr_component_assembly_v13_scope_refined as v13
import build_all_ocr_component_assembly_v14_variant_refined as v14
import build_all_ocr_component_assembly_v15_review_pruned as v15


ROOT = Path(__file__).resolve().parent
V15_DIR = ROOT / "export_final/pdf_recovery_incoming_2026-06-16_all_ocr_component_assembly_v15_review_pruned"
OUT_DIR = ROOT / "export_final/pdf_recovery_incoming_2026-06-16_all_ocr_component_assembly_v16_left_stub_boundary_hints"

V15_EDGES = V15_DIR / "v15_edge_recommendations.csv"
V15_MEMBERS = V15_DIR / "v15_member_classifier_overlay_passthrough.csv"

LEFT_STUB_JOIN_ACTION = "join_prev_left_stub_same_table_required_high"
JOIN_ACTIONS = {*v15.JOIN_ACTIONS, LEFT_STUB_JOIN_ACTION}

V16_EDGE_FIELDS = [
    *v15.V15_EDGE_FIELDS,
    "v16_recommended_action",
    "v16_recommended_layer",
    "v16_recommended_confidence",
    "v16_recommended_reason",
    "v16_review_priority",
    "v16_boundary_hint_type",
    "v16_boundary_hint_reason",
]

V16_COMPONENT_FIELDS = [
    *v15.V15_COMPONENT_FIELDS,
    "v16_component_join_actions",
]

V16_COMPONENT_MEMBER_FIELDS = [*v15.V15_COMPONENT_MEMBER_FIELDS]

V16_REVIEW_QUEUE_FIELDS = [
    "v16_review_priority",
    "v16_review_category",
    *[field for field in V16_EDGE_FIELDS if field != "v16_review_priority"],
]

V16_HINT_FIELDS = [
    "v16_boundary_hint_type",
    "v16_boundary_hint_reason",
    "sequence_edge_id",
    "v5_source_key",
    "prev_physical_run_id",
    "physical_run_id",
    "prev_run_page_range",
    "run_page_range",
    "prev_last_pdf_page",
    "first_pdf_page",
    "page_gap",
    "missing_pages",
    "prev_primary_title",
    "primary_title",
    "prev_example_images",
    "example_images",
    "prev_page_kind",
    "page_kind",
    "prev_table_family_key",
    "table_family_key",
    "prev_table_number_key",
    "table_number_key",
    "v11_prev_classifier_page_classes",
    "v11_classifier_page_classes",
    "v11_prev_classifier_titles",
    "v11_classifier_titles",
    "prev_layout_left_stub_pages",
    "prev_layout_right_half_pages",
    "layout_left_stub_pages",
    "layout_right_half_pages",
    "strict_title_similarity",
    "scope_similarity",
    "v16_recommended_action",
    "v16_review_priority",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def clean(value: object) -> str:
    return base.clean_text(value)


def pages_field_contains(field_value: object, page: object) -> bool:
    return v15.pages_field_contains(field_value, page)


def page_gap_is_adjacent(edge: dict[str, str]) -> bool:
    return str(edge.get("page_gap") or "") == "1"


def max_similarity(edge: dict[str, str]) -> float:
    return max(v13.as_float(edge.get("strict_title_similarity")), v13.as_float(edge.get("scope_similarity")))


def entity_set(value: object) -> set[str]:
    text = clean(value)
    if not text:
        return set()
    return {part.strip() for part in re.split(r"[;,|]+", text) if part.strip()}


def compatible_entity_sets(prev: object, cur: object) -> bool:
    prev_set = entity_set(prev)
    cur_set = entity_set(cur)
    return not prev_set or not cur_set or bool(prev_set & cur_set)


def compatible_entity_signals(edge: dict[str, str]) -> bool:
    return (
        compatible_entity_sets(edge.get("prev_religion_entities"), edge.get("religion_entities"))
        and compatible_entity_sets(edge.get("prev_part_entities"), edge.get("part_entities"))
        and compatible_entity_sets(edge.get("prev_sex_entities"), edge.get("sex_entities"))
    )


def row_layout_flags(edge: dict[str, str]) -> dict[str, bool]:
    prev_page = edge.get("prev_last_pdf_page")
    cur_page = edge.get("first_pdf_page")
    return {
        "prev_left_stub": pages_field_contains(edge.get("prev_layout_left_stub_pages"), prev_page),
        "prev_right_half": pages_field_contains(edge.get("prev_layout_right_half_pages"), prev_page),
        "cur_left_stub": pages_field_contains(edge.get("layout_left_stub_pages"), cur_page),
        "cur_right_half": pages_field_contains(edge.get("layout_right_half_pages"), cur_page),
    }


def title_text(edge: dict[str, str]) -> str:
    return clean(" ".join([str(edge.get("prev_primary_title") or ""), str(edge.get("primary_title") or "")]))


def has_local_scope_marker(value: object) -> bool:
    text = clean(value)
    return bool(
        re.search(r"\(\d+\)\s*[a-z][a-z .'-]*(taluk|district|division|state|city)", text)
        or re.search(r"\b(taluk|district|division|state|city)\b", text)
    )


def continuation_marker(value: object) -> bool:
    text = clean(value)
    return bool(re.search(r"\b(ctd|cld|contd|concld|continued|continuation)\b", text))


def lettered_religion_marker(value: object) -> bool:
    text = clean(value)
    return bool(
        re.search(r"\b[a-h]\s*[-.]\s*(minor|hindu|jain|musalman|christian|animistic|buddhist|sikh|parsi)", text)
        or re.search(r"\b(minor and unspecified religions|general table)\b", text)
    )


def left_stub_same_table_required_join(edge: dict[str, str]) -> tuple[bool, str]:
    if str(edge.get("v15_recommended_action") or "") != "review_same_table_boundary_not_joined":
        return False, ""
    if not page_gap_is_adjacent(edge):
        return False, "not_adjacent"
    if not v13.same_family_and_key(edge):
        return False, "different_table_family_or_number"

    flags = row_layout_flags(edge)
    if not flags["prev_left_stub"]:
        return False, "previous_page_not_detected_as_left_stub"
    if flags["cur_left_stub"]:
        return False, "current_page_starts_new_left_stub_block"
    if not compatible_entity_signals(edge):
        return False, "entity_signal_conflict"

    has_switch, switch_reason = v14.has_obvious_hard_switch(edge)
    if has_switch:
        return False, switch_reason

    sim = max_similarity(edge)
    if sim < 0.72:
        return False, f"similarity_below_threshold_{sim:.3f}"

    right_signal = "current_right_half_detected" if flags["cur_right_half"] else "current_right_half_not_detected_but_previous_left_stub_requires_following_page"
    return True, f"previous_left_stub_adjacent_same_table; {right_signal}; similarity={sim:.3f}"


def boundary_hint(edge: dict[str, str]) -> tuple[str, str]:
    if str(edge.get("v15_recommended_action") or "") != "review_same_table_boundary_not_joined":
        return "", ""
    if not page_gap_is_adjacent(edge):
        return "", ""

    flags = row_layout_flags(edge)
    prev_title = edge.get("prev_primary_title")
    cur_title = edge.get("primary_title")
    text = title_text(edge)

    if flags["cur_left_stub"]:
        return (
            "mixed_page_boundary_current_starts_left_stub",
            "current page starts a new left-stub body block; keep separated unless page-level slicing is available",
        )
    if flags["cur_right_half"] and not flags["prev_left_stub"]:
        return (
            "mixed_or_partial_boundary_current_right_half_without_previous_left_stub",
            "current page is detected as a right-half page but previous boundary is not its left-stub source",
        )
    if flags["prev_right_half"] and flags["cur_right_half"]:
        return (
            "parallel_right_half_boundary",
            "both adjacent pages are right-half/body-continuation pages; this is not a safe whole-page join",
        )
    if lettered_religion_marker(prev_title) or lettered_religion_marker(cur_title):
        return (
            "logical_variant_sequence_candidate",
            "same table/part with lettered religion or general-table blocks; useful as a bounded variant set, not a physical merge",
        )
    if continuation_marker(prev_title) or continuation_marker(cur_title) or has_local_scope_marker(prev_title) or has_local_scope_marker(cur_title):
        return (
            "mixed_page_or_subtable_boundary_candidate",
            "local scope or continuation marker suggests a table segment may start/end within a page",
        )
    if "single year age returns" in text:
        return (
            "mixed_page_boundary_age_return_candidate",
            "same C-V age-return title across adjacent pages, but no left/right layout signal supports a whole-page merge",
        )
    return (
        "same_table_boundary_needs_manual_review",
        "same table number remains adjacent but lacks the layout evidence required for automatic merge",
    )


def refine_edge(edge: dict[str, str]) -> dict[str, object]:
    action = str(edge.get("v15_recommended_action") or "")
    layer = str(edge.get("v15_recommended_layer") or "")
    confidence = str(edge.get("v15_recommended_confidence") or "")
    reason = str(edge.get("v15_recommended_reason") or "")
    priority = str(edge.get("v15_review_priority") or "")
    hint_type, hint_reason = boundary_hint(edge)

    join, join_reason = left_stub_same_table_required_join(edge)
    if join:
        action = LEFT_STUB_JOIN_ACTION
        layer = "required_left_stub_right_half_sequence"
        confidence = "high_review"
        reason = join_reason
        priority = ""
        hint_type = ""
        hint_reason = ""

    return {
        **edge,
        "v16_recommended_action": action,
        "v16_recommended_layer": layer,
        "v16_recommended_confidence": confidence,
        "v16_recommended_reason": reason,
        "v16_review_priority": priority,
        "v16_boundary_hint_type": hint_type,
        "v16_boundary_hint_reason": hint_reason,
        # Reuse the v11 component builder with v16-refined actions.
        "v11_recommended_action": action,
        "v11_recommended_reason": reason,
    }


def review_queue(edges: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for edge in edges:
        priority = str(edge.get("v16_review_priority") or "")
        action = str(edge.get("v16_recommended_action") or "")
        if not priority:
            continue
        if action in JOIN_ACTIONS or action.startswith("hard_break"):
            continue
        rows.append(
            {
                "v16_review_priority": priority,
                "v16_review_category": action.replace("review_", ""),
                **edge,
            }
        )
    rows.sort(
        key=lambda row: (
            v13.as_int(row.get("v16_review_priority")) or 99,
            str(row.get("v5_source_key") or ""),
            v13.as_int(row.get("prev_last_pdf_page")) or 10**9,
            v13.as_int(row.get("first_pdf_page")) or 10**9,
        )
    )
    return rows


def label_components(components: list[dict[str, object]], edges: list[dict[str, object]]) -> None:
    # First preserve the existing v15/v14/v13 component labels.
    v15.label_components(components, edges)

    edge_by_id = {str(edge.get("sequence_edge_id") or ""): edge for edge in edges}
    for component in components:
        component["v12_component_source"] = "v16_left_stub_boundary_hints"
        join_edge_ids = [
            edge_id.strip()
            for edge_id in str(component.get("v11_join_edge_ids") or "").split(";")
            if edge_id.strip()
        ]
        actions = {str(edge_by_id.get(edge_id, {}).get("v16_recommended_action") or "") for edge_id in join_edge_ids}
        component["v16_component_join_actions"] = base.uniq_join(sorted(actions), limit=20)
        if LEFT_STUB_JOIN_ACTION in actions:
            component["v11_component_layer"] = "required_left_stub_right_half_sequence"
            component["v11_static_site_action"] = "static_group_candidate_required_left_stub_right_half"
            component["v11_confidence"] = "high_review"


def hint_rows(edges: list[dict[str, object]], prefix: str = "") -> list[dict[str, object]]:
    rows = []
    for edge in edges:
        hint_type = str(edge.get("v16_boundary_hint_type") or "")
        if not hint_type:
            continue
        if prefix and not hint_type.startswith(prefix):
            continue
        rows.append(edge)
    rows.sort(
        key=lambda row: (
            str(row.get("v5_source_key") or ""),
            v13.as_int(row.get("prev_last_pdf_page")) or 10**9,
            v13.as_int(row.get("first_pdf_page")) or 10**9,
            str(row.get("sequence_edge_id") or ""),
        )
    )
    return rows


def validate(components: list[dict[str, object]], edges: list[dict[str, object]]) -> dict[str, object]:
    failures: list[str] = []
    for component in components:
        if v13.as_int(component.get("n_v10_groups")) > 1 and str(component.get("all_pages_consecutive")) != "True":
            failures.append(f"v16_component_internal_gap:{component.get('v11_component_id')}:{component.get('page_range')}")
    for edge in edges:
        if str(edge.get("v16_recommended_action") or "") in JOIN_ACTIONS and str(edge.get("page_gap") or "") != "1":
            failures.append(f"v16_join_non_adjacent:{edge.get('sequence_edge_id')}:{edge.get('page_gap')}")
        if str(edge.get("v16_recommended_action") or "") == LEFT_STUB_JOIN_ACTION:
            flags = row_layout_flags(edge)
            if not flags["prev_left_stub"] or flags["cur_left_stub"]:
                failures.append(f"v16_left_stub_rule_violation:{edge.get('sequence_edge_id')}")
    return {
        "v16_invariant_failure_count": len(failures),
        "v16_invariant_failures": failures[:100],
    }


def write_readme(summary: dict[str, object]) -> None:
    lines = [
        "# v16 Left-Stub Boundary Hints",
        "",
        "No OCR/API calls are made. This pass refines v15 while preserving the adjacent-real-PDF-page invariant.",
        "",
        "## Added Join",
        "",
        "- `join_prev_left_stub_same_table_required_high`: adjacent same-table boundary where the previous page is detected as the left/body-stub page and the current page does not start a new left-stub block.",
        "",
        "## Added Hints",
        "",
        "- Mixed-page boundaries: adjacent pages where a new body block appears to start or end inside a page.",
        "- Variant-sequence candidates: adjacent same-table/subtable blocks that are useful for review but not safe physical merges.",
        "",
        "## Summary",
        "",
    ]
    for key in sorted(summary):
        lines.append(f"- `{key}`: {summary[key]}")
    (OUT_DIR / "v16_rules_readme.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    csv.field_size_limit(10**7)
    groups = read_csv(v13.V10_GROUPS)
    members = read_csv(V15_MEMBERS)
    v15_edges = read_csv(V15_EDGES)
    if not groups or not members or not v15_edges:
        raise SystemExit("Missing v10/v15 inputs; run v15 first")

    refined_edges = [refine_edge(edge) for edge in v15_edges]
    components, component_members = v11.build_v11_components(groups, members, refined_edges, join_actions=JOIN_ACTIONS)
    label_components(components, refined_edges)
    queue = review_queue(refined_edges)
    mixed_hints = hint_rows(refined_edges, "mixed")
    variant_hints = hint_rows(refined_edges, "logical_variant")
    all_hints = hint_rows(refined_edges)

    edge_actions = Counter(str(edge.get("v16_recommended_action") or "") for edge in refined_edges)
    component_layers = Counter(str(component.get("v11_component_layer") or "") for component in components)
    hint_counts = Counter(str(edge.get("v16_boundary_hint_type") or "") for edge in all_hints)
    summary = {
        "input_v10_groups": len(groups),
        "input_v15_members": len(members),
        "input_v15_edges": len(v15_edges),
        "v16_components": len(components),
        "v16_component_members": len(component_members),
        "v16_edge_rows": len(refined_edges),
        "v16_review_queue_edges": len(queue),
        "v16_promoted_left_stub_joins": edge_actions.get(LEFT_STUB_JOIN_ACTION, 0),
        "v16_kept_missing_bridges": edge_actions.get("keep_missing_bridge", 0),
        "v16_remaining_same_table_review_edges": edge_actions.get("review_same_table_boundary_not_joined", 0),
        "v16_mixed_boundary_hint_edges": len(mixed_hints),
        "v16_variant_sequence_hint_edges": len(variant_hints),
        "v16_all_boundary_hint_edges": len(all_hints),
        "v16_boundary_hint_counts": dict(hint_counts),
        "v16_component_layer_counts": dict(component_layers),
        "v16_edge_action_counts": dict(edge_actions),
    }
    summary.update(validate(components, refined_edges))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base.write_csv(OUT_DIR / "v16_edge_recommendations.csv", refined_edges, V16_EDGE_FIELDS)
    base.write_csv(OUT_DIR / "v16_components.csv", components, V16_COMPONENT_FIELDS)
    base.write_csv(OUT_DIR / "v16_component_members.csv", component_members, V16_COMPONENT_MEMBER_FIELDS)
    base.write_csv(OUT_DIR / "v16_edge_review_queue.csv", queue, V16_REVIEW_QUEUE_FIELDS)
    base.write_csv(OUT_DIR / "v16_mixed_page_boundary_hints.csv", mixed_hints, V16_HINT_FIELDS)
    base.write_csv(OUT_DIR / "v16_variant_sequence_hints.csv", variant_hints, V16_HINT_FIELDS)
    base.write_csv(OUT_DIR / "v16_all_boundary_hints.csv", all_hints, V16_HINT_FIELDS)
    base.write_csv(OUT_DIR / "v16_member_classifier_overlay_passthrough.csv", members, v11.V11_MEMBER_OVERLAY_FIELDS)
    base.write_csv(OUT_DIR / "v16_required_physical_review_passthrough.csv", read_csv(v13.V10_REQUIRED_CANDIDATES), v10.V10_REQUIRED_CANDIDATE_FIELDS)
    base.write_csv(OUT_DIR / "v16_required_physical_edges_passthrough.csv", read_csv(v13.V10_REQUIRED_EDGES), v10.V10_REQUIRED_EDGE_FIELDS)
    bridges = read_csv(v13.V10_BRIDGES)
    overassoc = read_csv(v13.V10_OVERASSOC)
    base.write_csv(OUT_DIR / "v16_missing_bridge_candidates_passthrough.csv", bridges, list(bridges[0].keys()) if bridges else [])
    base.write_csv(OUT_DIR / "v16_old_overassociation_audit_passthrough.csv", overassoc, list(overassoc[0].keys()) if overassoc else [])

    with (OUT_DIR / "v16_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    write_readme(summary)

    summary_rows = [
        {"metric": key, "value": json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value}
        for key, value in summary.items()
    ]
    base.write_workbook(
        OUT_DIR / "v16_left_stub_boundary_hints.xlsx",
        {
            "summary": (summary_rows, ["metric", "value"]),
            "components": (components, V16_COMPONENT_FIELDS),
            "component_members": (component_members, V16_COMPONENT_MEMBER_FIELDS),
            "edge_recommendations": (refined_edges, V16_EDGE_FIELDS),
            "edge_review_queue": (queue, V16_REVIEW_QUEUE_FIELDS),
            "mixed_page_boundary_hints": (mixed_hints, V16_HINT_FIELDS),
            "variant_sequence_hints": (variant_hints, V16_HINT_FIELDS),
            "all_boundary_hints": (all_hints, V16_HINT_FIELDS),
            "required_physical_review": (read_csv(v13.V10_REQUIRED_CANDIDATES), v10.V10_REQUIRED_CANDIDATE_FIELDS),
        },
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
