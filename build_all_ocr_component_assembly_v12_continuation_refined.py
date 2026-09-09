#!/usr/bin/env python3
"""Refine v11 assembly decisions with bounded continuation-table joins.

This pass performs no OCR/API work. It consumes v10/v11 outputs and makes two
improvements:

* demote obvious review noise, especially context/blank/offtarget/derivative
  and hard entity-switch boundaries;
* promote adjacent same-family/same-table continuation-order edges, such as
  "TABLE C-V ..." followed by "... Contd. TABLE C-V".

Only adjacent PDF pages can be joined. Missing-page bridges remain non-joins.
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


ROOT = Path(__file__).resolve().parent
V10_DIR = ROOT / "export_final/pdf_recovery_incoming_2026-06-16_all_ocr_component_assembly_v10_layered_manifest"
V11_DIR = ROOT / "export_final/pdf_recovery_incoming_2026-06-16_all_ocr_component_assembly_v11_rule_audit"
OUT_DIR = ROOT / "export_final/pdf_recovery_incoming_2026-06-16_all_ocr_component_assembly_v12_continuation_refined"

V10_GROUPS = V10_DIR / "v10_assembly_groups.csv"
V11_MEMBERS = V11_DIR / "v11_member_classifier_overlay.csv"
V11_EDGES = V11_DIR / "v11_edge_recommendations.csv"
V10_REQUIRED_CANDIDATES = V10_DIR / "v10_possible_required_physical_edges_review.csv"
V10_REQUIRED_EDGES = V10_DIR / "v10_required_physical_edges.csv"
V10_BRIDGES = V10_DIR / "v10_missing_bridge_candidates.csv"
V10_OVERASSOC = V10_DIR / "v10_old_overassociation_audit.csv"

BAD_PAGE_KINDS = {"context_or_blank", "offtarget_table", "unknown", ""}
BAD_FAMILIES = {"context", "offtarget", ""}
JOIN_ACTIONS = {"join_classifier_main_tablevii_high", "join_continuation_same_table_high"}

V12_EDGE_FIELDS = [
    *v11.V11_EDGE_FIELDS,
    "v12_recommended_action",
    "v12_recommended_layer",
    "v12_recommended_confidence",
    "v12_recommended_reason",
    "v12_review_priority",
]

V12_COMPONENT_FIELDS = [
    *v11.V11_COMPONENT_FIELDS,
    "v12_component_source",
]

V12_COMPONENT_MEMBER_FIELDS = [
    *v11.V11_COMPONENT_MEMBER_FIELDS,
]

V12_REVIEW_QUEUE_FIELDS = [
    "v12_review_priority",
    "v12_review_category",
    *[field for field in V12_EDGE_FIELDS if field != "v12_review_priority"],
]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def as_float(value: object) -> float:
    try:
        return float(str(value or "").strip())
    except ValueError:
        return 0.0


def as_int(value: object) -> int:
    return base.int_or_none(value) or 0


def normalize_table_key(value: object) -> str:
    key = base.clean_text(value)
    if key in {"7", "vii", "vil", "vi1", "v1i"}:
        return "7"
    return key


def has_continuation_text(*values: object) -> bool:
    blob = " ".join(str(value or "") for value in values).lower()
    return bool(re.search(r"\b(contd|continued|concluded|concld|concl|cont)\b", blob))


def has_bad_boundary(edge: dict[str, str]) -> tuple[bool, str]:
    prev_kind = str(edge.get("prev_page_kind") or "")
    cur_kind = str(edge.get("page_kind") or "")
    prev_family = str(edge.get("prev_table_family_key") or "")
    cur_family = str(edge.get("table_family_key") or "")
    prev_classes = set(part.strip() for part in str(edge.get("v11_prev_classifier_page_classes") or "").split(";") if part.strip())
    cur_classes = set(part.strip() for part in str(edge.get("v11_classifier_page_classes") or "").split(";") if part.strip())
    classifier_bad = bool((prev_classes | cur_classes) & (v11.CONTEXT_CLASSES | v11.DERIVATIVE_CLASSES))
    if classifier_bad:
        return True, "classifier_context_derivative_or_offtarget_boundary"
    if "derivative" in prev_kind or "derivative" in cur_kind:
        return True, "derivative_table_boundary"
    if prev_kind in BAD_PAGE_KINDS or cur_kind in BAD_PAGE_KINDS:
        return True, f"non_table_page_kind_boundary_{prev_kind}_to_{cur_kind}"
    if prev_family in BAD_FAMILIES or cur_family in BAD_FAMILIES:
        return True, f"non_table_family_boundary_{prev_family}_to_{cur_family}"
    return False, ""


def has_scope_entity_switch(edge: dict[str, str]) -> bool:
    return "sequence_scope_entity_switch" in str(edge.get("sequence_review_flags") or "")


def has_table_number_switch(edge: dict[str, str]) -> bool:
    prev_key = normalize_table_key(edge.get("prev_table_number_key"))
    cur_key = normalize_table_key(edge.get("table_number_key"))
    return bool(prev_key and cur_key and prev_key != cur_key)


def same_family_key(edge: dict[str, str]) -> bool:
    return bool(
        str(edge.get("prev_table_family_key") or "")
        and str(edge.get("prev_table_family_key") or "") == str(edge.get("table_family_key") or "")
        and normalize_table_key(edge.get("prev_table_number_key"))
        and normalize_table_key(edge.get("prev_table_number_key")) == normalize_table_key(edge.get("table_number_key"))
    )


def continuation_join_candidate(edge: dict[str, str]) -> tuple[bool, str]:
    if str(edge.get("page_gap") or "") != "1":
        return False, "not_adjacent_pdf_pages"
    bad, bad_reason = has_bad_boundary(edge)
    if bad:
        return False, bad_reason
    if has_scope_entity_switch(edge):
        return False, "scope_entity_switch"
    if has_table_number_switch(edge):
        return False, "table_number_switch"
    if not same_family_key(edge):
        return False, "not_same_table_family_and_key"
    sim = max(as_float(edge.get("strict_title_similarity")), as_float(edge.get("scope_similarity")))
    if sim < 0.74:
        return False, f"title_similarity_below_0.74_{sim:.2f}"
    if not has_continuation_text(edge.get("prev_primary_title"), edge.get("primary_title")):
        return False, "no_continuation_token"
    return True, f"adjacent_same_family_same_table_continuation_order; max_title_similarity_{sim:.2f}"


def refine_edge(edge: dict[str, str]) -> dict[str, object]:
    action = str(edge.get("v11_recommended_action") or "")
    v12_action = action
    layer = str(edge.get("v11_recommended_layer") or "")
    confidence = str(edge.get("v11_recommended_confidence") or "")
    reason = str(edge.get("v11_recommended_reason") or "")
    priority = str(edge.get("v11_review_priority") or "")

    bad, bad_reason = has_bad_boundary(edge)
    if action == "review_required_row_label_join":
        if bad:
            v12_action = "hard_break_demoted_row_label_context_noise"
            layer = "hard_break"
            confidence = "high"
            reason = bad_reason
            priority = ""
        elif has_scope_entity_switch(edge) or has_table_number_switch(edge):
            v12_action = "hard_break_demoted_row_label_scope_switch"
            layer = "hard_break"
            confidence = "high"
            reason = "scope_or_table_number_switch_blocks_required_row_label_join"
            priority = ""
        else:
            v12_action = "review_required_row_label_join_tablelike"
            layer = "possible_required_physical_row_label_assembly"
            confidence = "review"
            priority = "1"
    elif action == "review_related_not_grouped":
        ok, ok_reason = continuation_join_candidate(edge)
        if ok:
            v12_action = "join_continuation_same_table_high"
            layer = "continuation_same_table_body_sequence"
            confidence = "high_review"
            reason = ok_reason
            priority = ""
        elif bad:
            v12_action = "hard_break_demoted_related_context_noise"
            layer = "hard_break"
            confidence = "high"
            reason = bad_reason
            priority = ""
        elif has_scope_entity_switch(edge):
            v12_action = "hard_break_scope_entity_switch"
            layer = "hard_break"
            confidence = "high"
            reason = "part_sex_or_parenthetical_entity_switch"
            priority = ""
        elif has_table_number_switch(edge):
            v12_action = "hard_break_table_number_switch"
            layer = "hard_break"
            confidence = "high"
            reason = "explicit_table_number_switch"
            priority = ""
        elif same_family_key(edge):
            v12_action = "review_same_table_boundary_not_joined"
            layer = "candidate_same_table_not_grouped"
            confidence = "review"
            priority = "3"
        else:
            v12_action = "review_related_not_grouped"
            layer = "candidate_related_not_grouped"
            confidence = "review"
            priority = "4"

    return {
        **edge,
        "v12_recommended_action": v12_action,
        "v12_recommended_layer": layer,
        "v12_recommended_confidence": confidence,
        "v12_recommended_reason": reason,
        "v12_review_priority": priority,
        # Make the v11 component builder see v12's refined action.
        "v11_recommended_action": v12_action,
        "v11_recommended_reason": reason,
    }


def review_queue(edges: list[dict[str, object]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for edge in edges:
        priority = str(edge.get("v12_review_priority") or "")
        action = str(edge.get("v12_recommended_action") or "")
        if not priority:
            continue
        if action in JOIN_ACTIONS or action.startswith("hard_break"):
            continue
        out.append({
            "v12_review_priority": priority,
            "v12_review_category": action.replace("review_", ""),
            **edge,
        })
    out.sort(key=lambda row: (
        as_int(row.get("v12_review_priority")) or 99,
        str(row.get("v5_source_key") or ""),
        as_int(row.get("prev_last_pdf_page")) or 10**9,
        as_int(row.get("first_pdf_page")) or 10**9,
    ))
    return out


def validate(components: list[dict[str, object]], edges: list[dict[str, object]]) -> dict[str, object]:
    failures: list[str] = []
    for component in components:
        if as_int(component.get("n_v10_groups")) > 1 and str(component.get("all_pages_consecutive")) != "True":
            failures.append(f"v12_component_internal_gap:{component.get('v11_component_id')}:{component.get('page_range')}")
    for edge in edges:
        if str(edge.get("v12_recommended_action") or "") in JOIN_ACTIONS and str(edge.get("page_gap") or "") != "1":
            failures.append(f"v12_join_non_adjacent:{edge.get('sequence_edge_id')}:{edge.get('page_gap')}")
    return {
        "v12_invariant_failure_count": len(failures),
        "v12_invariant_failures": failures[:100],
    }


def write_readme(summary: dict[str, object]) -> None:
    lines = [
        "# v12 Continuation-Refined Assembly",
        "",
        "No OCR/API calls are made. This pass refines v11 edge recommendations and rebuilds components.",
        "",
        "## Added Rule",
        "",
        "- Join adjacent pages when they are same table family, same normalized table key, table-like, no scope/entity switch, no derivative/context/offtarget signal, title similarity >= 0.74, and one side has a continuation/concluded token.",
        "",
        "## Demotions",
        "",
        "- Row-label and related-edge review candidates involving context/blank/offtarget/derivative pages are hard breaks.",
        "- Part/sex/parenthetical entity switches and explicit table-number switches are hard breaks, not manual review, unless a future rule explicitly handles them.",
        "",
        "## Summary",
        "",
    ]
    for key in sorted(summary):
        lines.append(f"- `{key}`: {summary[key]}")
    (OUT_DIR / "v12_rules_readme.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    csv.field_size_limit(10 ** 7)
    groups = read_csv(V10_GROUPS)
    members = read_csv(V11_MEMBERS)
    v11_edges = read_csv(V11_EDGES)
    if not groups or not members or not v11_edges:
        raise SystemExit("Missing v10/v11 inputs; run v11 first")

    refined_edges = [refine_edge(edge) for edge in v11_edges]
    components, component_members = v11.build_v11_components(groups, members, refined_edges, join_actions=JOIN_ACTIONS)
    edge_by_id = {str(edge.get("sequence_edge_id") or ""): edge for edge in refined_edges}
    for component in components:
        component["v12_component_source"] = "v12_refined_edges"
        join_edge_ids = [
            edge_id.strip()
            for edge_id in str(component.get("v11_join_edge_ids") or "").split(";")
            if edge_id.strip()
        ]
        join_actions = {str(edge_by_id.get(edge_id, {}).get("v12_recommended_action") or "") for edge_id in join_edge_ids}
        if "join_continuation_same_table_high" in join_actions:
            component["v11_component_layer"] = "continuation_refined_body_sequence"
            component["v11_static_site_action"] = "static_group_candidate_continuation_refined"
            component["v11_confidence"] = "high_review"
        elif "join_classifier_main_tablevii_high" in join_actions:
            component["v11_component_layer"] = "classifier_assisted_body_sequence"
            component["v11_static_site_action"] = "static_group_candidate_classifier_assisted"
            component["v11_confidence"] = "high_review"
    queue = review_queue(refined_edges)

    edge_actions = Counter(str(edge.get("v12_recommended_action") or "") for edge in refined_edges)
    component_layers = Counter(str(component.get("v11_component_layer") or "") for component in components)
    summary = {
        "input_v10_groups": len(groups),
        "input_v11_members": len(members),
        "input_v11_edges": len(v11_edges),
        "v12_components": len(components),
        "v12_component_members": len(component_members),
        "v12_edge_rows": len(refined_edges),
        "v12_review_queue_edges": len(queue),
        "v12_join_classifier_main_edges": edge_actions.get("join_classifier_main_tablevii_high", 0),
        "v12_join_continuation_same_table_edges": edge_actions.get("join_continuation_same_table_high", 0),
        "v12_demoted_row_label_noise_edges": edge_actions.get("hard_break_demoted_row_label_context_noise", 0)
        + edge_actions.get("hard_break_demoted_row_label_scope_switch", 0),
        "v12_demoted_related_noise_edges": edge_actions.get("hard_break_demoted_related_context_noise", 0)
        + edge_actions.get("hard_break_scope_entity_switch", 0)
        + edge_actions.get("hard_break_table_number_switch", 0),
        "v12_remaining_required_row_label_review_edges": edge_actions.get("review_required_row_label_join_tablelike", 0),
        "v12_remaining_same_table_review_edges": edge_actions.get("review_same_table_boundary_not_joined", 0),
        "v12_remaining_related_review_edges": edge_actions.get("review_related_not_grouped", 0),
        "v12_kept_existing_joins": edge_actions.get("keep_existing_join", 0),
        "v12_kept_missing_bridges": edge_actions.get("keep_missing_bridge", 0),
        "v12_component_layer_counts": dict(component_layers),
        "v12_edge_action_counts": dict(edge_actions),
    }
    summary.update(validate(components, refined_edges))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base.write_csv(OUT_DIR / "v12_edge_recommendations.csv", refined_edges, V12_EDGE_FIELDS)
    base.write_csv(OUT_DIR / "v12_components.csv", components, V12_COMPONENT_FIELDS)
    base.write_csv(OUT_DIR / "v12_component_members.csv", component_members, V12_COMPONENT_MEMBER_FIELDS)
    base.write_csv(OUT_DIR / "v12_edge_review_queue.csv", queue, V12_REVIEW_QUEUE_FIELDS)
    base.write_csv(OUT_DIR / "v12_member_classifier_overlay_passthrough.csv", members, v11.V11_MEMBER_OVERLAY_FIELDS)
    base.write_csv(OUT_DIR / "v12_required_physical_review_passthrough.csv", read_csv(V10_REQUIRED_CANDIDATES), v10.V10_REQUIRED_CANDIDATE_FIELDS)
    base.write_csv(OUT_DIR / "v12_required_physical_edges_passthrough.csv", read_csv(V10_REQUIRED_EDGES), v10.V10_REQUIRED_EDGE_FIELDS)
    bridges = read_csv(V10_BRIDGES)
    overassoc = read_csv(V10_OVERASSOC)
    base.write_csv(OUT_DIR / "v12_missing_bridge_candidates_passthrough.csv", bridges, list(bridges[0].keys()) if bridges else [])
    base.write_csv(OUT_DIR / "v12_old_overassociation_audit_passthrough.csv", overassoc, list(overassoc[0].keys()) if overassoc else [])

    with (OUT_DIR / "v12_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    write_readme(summary)

    summary_rows = [
        {"metric": key, "value": json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value}
        for key, value in summary.items()
    ]
    base.write_workbook(
        OUT_DIR / "v12_continuation_refined_review.xlsx",
        {
            "summary": (summary_rows, ["metric", "value"]),
            "components": (components, V12_COMPONENT_FIELDS),
            "component_members": (component_members, V12_COMPONENT_MEMBER_FIELDS),
            "edge_recommendations": (refined_edges, V12_EDGE_FIELDS),
            "edge_review_queue": (queue, V12_REVIEW_QUEUE_FIELDS),
            "required_physical_review": (read_csv(V10_REQUIRED_CANDIDATES), v10.V10_REQUIRED_CANDIDATE_FIELDS),
        },
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
