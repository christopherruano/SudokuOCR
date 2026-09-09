#!/usr/bin/env python3
"""Review-pruned refinement of v14 adjacency-first assembly.

No OCR/API calls are made. This pass consumes v14 outputs and keeps the hard
invariant that promoted joins must be adjacent real PDF pages.

Compared with v14, this pass does not add broad joins. It removes false
positive row-label review edges where:

* the current run already starts with a left-stub page that supplies row labels;
* or the row-label signal came from narrative/context pages without a right-half
  layout signal or matching table title.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import build_all_ocr_component_assembly_v5_structural as base
import build_all_ocr_component_assembly_v10_layered_manifest as v10
import build_all_ocr_component_assembly_v11_rule_audit as v11
import build_all_ocr_component_assembly_v13_scope_refined as v13
import build_all_ocr_component_assembly_v14_variant_refined as v14


ROOT = Path(__file__).resolve().parent
V14_DIR = ROOT / "export_final/pdf_recovery_incoming_2026-06-16_all_ocr_component_assembly_v14_variant_refined"
OUT_DIR = ROOT / "export_final/pdf_recovery_incoming_2026-06-16_all_ocr_component_assembly_v15_review_pruned"

V14_EDGES = V14_DIR / "v14_edge_recommendations.csv"
V14_MEMBERS = V14_DIR / "v14_member_classifier_overlay_passthrough.csv"

JOIN_ACTIONS = set(v14.JOIN_ACTIONS)

V15_EDGE_FIELDS = [
    *v14.V14_EDGE_FIELDS,
    "v15_recommended_action",
    "v15_recommended_layer",
    "v15_recommended_confidence",
    "v15_recommended_reason",
    "v15_review_priority",
]

V15_COMPONENT_FIELDS = [
    *v14.V14_COMPONENT_FIELDS,
    "v15_component_join_actions",
]

V15_COMPONENT_MEMBER_FIELDS = [*v14.V14_COMPONENT_MEMBER_FIELDS]

V15_REVIEW_QUEUE_FIELDS = [
    "v15_review_priority",
    "v15_review_category",
    *[field for field in V15_EDGE_FIELDS if field != "v15_review_priority"],
]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def pages_field_contains(field_value: object, page: object) -> bool:
    target = str(page or "").strip()
    if not target:
        return False
    values = {
        token.strip()
        for token in str(field_value or "").replace(";", ",").split(",")
        if token.strip()
    }
    return target in values


def row_label_candidate_should_demote(edge: dict[str, str]) -> tuple[bool, str]:
    first_page = edge.get("first_pdf_page")
    title_sim = max(v13.as_float(edge.get("strict_title_similarity")), v13.as_float(edge.get("scope_similarity")))
    has_current_right_half = bool(str(edge.get("layout_right_half_pages") or "").strip())
    current_starts_with_left_stub = pages_field_contains(edge.get("layout_left_stub_pages"), first_page)

    if current_starts_with_left_stub:
        return True, "current_run_starts_with_left_stub_so_previous_run_not_required_for_row_labels"

    if not has_current_right_half and title_sim < 0.50:
        return True, "low_similarity_without_right_half_layout_signal_context_or_standalone_table"

    return False, ""


def refine_edge(edge: dict[str, str]) -> dict[str, object]:
    action = str(edge.get("v14_recommended_action") or "")
    v15_action = action
    layer = str(edge.get("v14_recommended_layer") or "")
    confidence = str(edge.get("v14_recommended_confidence") or "")
    reason = str(edge.get("v14_recommended_reason") or "")
    priority = str(edge.get("v14_review_priority") or "")

    if action == "review_required_row_label_join_tablelike":
        should_demote, demote_reason = row_label_candidate_should_demote(edge)
        if should_demote:
            v15_action = "hard_break_pruned_false_row_label_candidate"
            layer = "hard_break"
            confidence = "high"
            reason = demote_reason
            priority = ""

    return {
        **edge,
        "v15_recommended_action": v15_action,
        "v15_recommended_layer": layer,
        "v15_recommended_confidence": confidence,
        "v15_recommended_reason": reason,
        "v15_review_priority": priority,
        # Reuse the v11 component builder with v15-refined actions.
        "v11_recommended_action": v15_action,
        "v11_recommended_reason": reason,
    }


def review_queue(edges: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for edge in edges:
        priority = str(edge.get("v15_review_priority") or "")
        action = str(edge.get("v15_recommended_action") or "")
        if not priority:
            continue
        if action in JOIN_ACTIONS or action.startswith("hard_break"):
            continue
        rows.append(
            {
                "v15_review_priority": priority,
                "v15_review_category": action.replace("review_", ""),
                **edge,
            }
        )
    rows.sort(
        key=lambda row: (
            v13.as_int(row.get("v15_review_priority")) or 99,
            str(row.get("v5_source_key") or ""),
            v13.as_int(row.get("prev_last_pdf_page")) or 10**9,
            v13.as_int(row.get("first_pdf_page")) or 10**9,
        )
    )
    return rows


def label_components(components: list[dict[str, object]], edges: list[dict[str, object]]) -> None:
    edge_by_id = {str(edge.get("sequence_edge_id") or ""): edge for edge in edges}
    for component in components:
        component["v12_component_source"] = "v15_review_pruned_edges"
        join_edge_ids = [
            edge_id.strip()
            for edge_id in str(component.get("v11_join_edge_ids") or "").split(";")
            if edge_id.strip()
        ]
        actions = {str(edge_by_id.get(edge_id, {}).get("v15_recommended_action") or "") for edge_id in join_edge_ids}
        component["v15_component_join_actions"] = base.uniq_join(sorted(actions), limit=20)

        if "join_classifier_same_part_tablevii_continuation_high" in actions:
            component["v11_component_layer"] = "classifier_same_part_tablevii_sequence"
            component["v11_static_site_action"] = "static_group_candidate_classifier_same_part_tablevii"
            component["v11_confidence"] = "high_review"
        elif "join_explicit_same_scope_same_table_high" in actions:
            component["v11_component_layer"] = "explicit_same_scope_body_sequence"
            component["v11_static_site_action"] = "static_group_candidate_explicit_same_scope"
            component["v11_confidence"] = "high_review"
        elif "join_classifier_split_title_tablevii_high" in actions:
            component["v11_component_layer"] = "classifier_split_title_body_sequence"
            component["v11_static_site_action"] = "static_group_candidate_classifier_split_title"
            component["v11_confidence"] = "high_review"
        elif "join_classifier_core_numeric_form_high" in actions:
            component["v11_component_layer"] = "classifier_core_numeric_form_sequence"
            component["v11_static_site_action"] = "static_group_candidate_classifier_core_numeric"
            component["v11_confidence"] = "high_review"
        elif "join_plain_same_table_high" in actions:
            component["v11_component_layer"] = "same_table_refined_body_sequence"
            component["v11_static_site_action"] = "static_group_candidate_same_table_refined"
            component["v11_confidence"] = "high_review"
        elif "join_same_scope_short_title_high" in actions:
            component["v11_component_layer"] = "same_scope_short_title_body_sequence"
            component["v11_static_site_action"] = "static_group_candidate_same_scope_short_title"
            component["v11_confidence"] = "high_review"
        elif "join_continuation_same_table_high" in actions:
            component["v11_component_layer"] = "continuation_refined_body_sequence"
            component["v11_static_site_action"] = "static_group_candidate_continuation_refined"
            component["v11_confidence"] = "high_review"
        elif "join_classifier_main_tablevii_high" in actions:
            component["v11_component_layer"] = "classifier_assisted_body_sequence"
            component["v11_static_site_action"] = "static_group_candidate_classifier_assisted"
            component["v11_confidence"] = "high_review"


def validate(components: list[dict[str, object]], edges: list[dict[str, object]]) -> dict[str, object]:
    failures: list[str] = []
    for component in components:
        if v13.as_int(component.get("n_v10_groups")) > 1 and str(component.get("all_pages_consecutive")) != "True":
            failures.append(f"v15_component_internal_gap:{component.get('v11_component_id')}:{component.get('page_range')}")
    for edge in edges:
        if str(edge.get("v15_recommended_action") or "") in JOIN_ACTIONS and str(edge.get("page_gap") or "") != "1":
            failures.append(f"v15_join_non_adjacent:{edge.get('sequence_edge_id')}:{edge.get('page_gap')}")
    return {
        "v15_invariant_failure_count": len(failures),
        "v15_invariant_failures": failures[:100],
    }


def write_readme(summary: dict[str, object]) -> None:
    lines = [
        "# v15 Review-Pruned Assembly",
        "",
        "No OCR/API calls are made. This pass refines v14 while preserving the adjacent-real-PDF-page invariant.",
        "",
        "## Added Hard Breaks",
        "",
        "- False row-label candidates where the current run already starts with a left-stub row-label page.",
        "- False row-label candidates from narrative/context pages without right-half layout evidence and without a matching table title.",
        "",
        "## Summary",
        "",
    ]
    for key in sorted(summary):
        lines.append(f"- `{key}`: {summary[key]}")
    (OUT_DIR / "v15_rules_readme.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    csv.field_size_limit(10**7)
    groups = read_csv(v13.V10_GROUPS)
    members = read_csv(V14_MEMBERS)
    v14_edges = read_csv(V14_EDGES)
    if not groups or not members or not v14_edges:
        raise SystemExit("Missing v10/v14 inputs; run v14 first")

    refined_edges = [refine_edge(edge) for edge in v14_edges]
    components, component_members = v11.build_v11_components(groups, members, refined_edges, join_actions=JOIN_ACTIONS)
    label_components(components, refined_edges)
    queue = review_queue(refined_edges)

    edge_actions = Counter(str(edge.get("v15_recommended_action") or "") for edge in refined_edges)
    component_layers = Counter(str(component.get("v11_component_layer") or "") for component in components)
    summary = {
        "input_v10_groups": len(groups),
        "input_v14_members": len(members),
        "input_v14_edges": len(v14_edges),
        "v15_components": len(components),
        "v15_component_members": len(component_members),
        "v15_edge_rows": len(refined_edges),
        "v15_review_queue_edges": len(queue),
        "v15_hard_break_pruned_false_row_label_edges": edge_actions.get("hard_break_pruned_false_row_label_candidate", 0),
        "v15_kept_missing_bridges": edge_actions.get("keep_missing_bridge", 0),
        "v15_remaining_required_row_label_review_edges": edge_actions.get("review_required_row_label_join_tablelike", 0),
        "v15_remaining_same_table_review_edges": edge_actions.get("review_same_table_boundary_not_joined", 0),
        "v15_component_layer_counts": dict(component_layers),
        "v15_edge_action_counts": dict(edge_actions),
    }
    summary.update(validate(components, refined_edges))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base.write_csv(OUT_DIR / "v15_edge_recommendations.csv", refined_edges, V15_EDGE_FIELDS)
    base.write_csv(OUT_DIR / "v15_components.csv", components, V15_COMPONENT_FIELDS)
    base.write_csv(OUT_DIR / "v15_component_members.csv", component_members, V15_COMPONENT_MEMBER_FIELDS)
    base.write_csv(OUT_DIR / "v15_edge_review_queue.csv", queue, V15_REVIEW_QUEUE_FIELDS)
    base.write_csv(OUT_DIR / "v15_member_classifier_overlay_passthrough.csv", members, v11.V11_MEMBER_OVERLAY_FIELDS)
    base.write_csv(OUT_DIR / "v15_required_physical_review_passthrough.csv", read_csv(v13.V10_REQUIRED_CANDIDATES), v10.V10_REQUIRED_CANDIDATE_FIELDS)
    base.write_csv(OUT_DIR / "v15_required_physical_edges_passthrough.csv", read_csv(v13.V10_REQUIRED_EDGES), v10.V10_REQUIRED_EDGE_FIELDS)
    bridges = read_csv(v13.V10_BRIDGES)
    overassoc = read_csv(v13.V10_OVERASSOC)
    base.write_csv(OUT_DIR / "v15_missing_bridge_candidates_passthrough.csv", bridges, list(bridges[0].keys()) if bridges else [])
    base.write_csv(OUT_DIR / "v15_old_overassociation_audit_passthrough.csv", overassoc, list(overassoc[0].keys()) if overassoc else [])

    with (OUT_DIR / "v15_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    write_readme(summary)

    summary_rows = [
        {"metric": key, "value": json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value}
        for key, value in summary.items()
    ]
    base.write_workbook(
        OUT_DIR / "v15_review_pruned.xlsx",
        {
            "summary": (summary_rows, ["metric", "value"]),
            "components": (components, V15_COMPONENT_FIELDS),
            "component_members": (component_members, V15_COMPONENT_MEMBER_FIELDS),
            "edge_recommendations": (refined_edges, V15_EDGE_FIELDS),
            "edge_review_queue": (queue, V15_REVIEW_QUEUE_FIELDS),
            "required_physical_review": (read_csv(v13.V10_REQUIRED_CANDIDATES), v10.V10_REQUIRED_CANDIDATE_FIELDS),
        },
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
