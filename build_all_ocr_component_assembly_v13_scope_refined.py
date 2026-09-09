#!/usr/bin/env python3
"""Scope-aware refinement of v12 assembly recommendations.

No OCR/API calls are made. This pass consumes v12 and:

* hard-breaks remaining related edges that are table-family switches;
* hard-breaks same-table review edges where only one side carries a scope
  modifier such as fly-sheet, summary, geo detail, special status, or caste/race;
* promotes a small set of adjacent same-table boundaries with strong evidence;
* promotes classifier-confirmed Table VII split-title/core numeric fragments.

All joins remain strictly adjacent real PDF pages.
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
import build_all_ocr_component_assembly_v12_continuation_refined as v12


ROOT = Path(__file__).resolve().parent
V10_DIR = ROOT / "export_final/pdf_recovery_incoming_2026-06-16_all_ocr_component_assembly_v10_layered_manifest"
V12_DIR = ROOT / "export_final/pdf_recovery_incoming_2026-06-16_all_ocr_component_assembly_v12_continuation_refined"
OUT_DIR = ROOT / "export_final/pdf_recovery_incoming_2026-06-16_all_ocr_component_assembly_v13_scope_refined"

V10_GROUPS = V10_DIR / "v10_assembly_groups.csv"
V12_MEMBERS = V12_DIR / "v12_member_classifier_overlay_passthrough.csv"
V12_EDGES = V12_DIR / "v12_edge_recommendations.csv"
V10_REQUIRED_CANDIDATES = V10_DIR / "v10_possible_required_physical_edges_review.csv"
V10_REQUIRED_EDGES = V10_DIR / "v10_required_physical_edges.csv"
V10_BRIDGES = V10_DIR / "v10_missing_bridge_candidates.csv"
V10_OVERASSOC = V10_DIR / "v10_old_overassociation_audit.csv"

JOIN_ACTIONS = {
    "join_classifier_main_tablevii_high",
    "join_continuation_same_table_high",
    "join_classifier_split_title_tablevii_high",
    "join_classifier_core_numeric_form_high",
    "join_plain_same_table_high",
    "join_same_scope_short_title_high",
}

V13_EDGE_FIELDS = [
    *v12.V12_EDGE_FIELDS,
    "v13_recommended_action",
    "v13_recommended_layer",
    "v13_recommended_confidence",
    "v13_recommended_reason",
    "v13_review_priority",
]

V13_COMPONENT_FIELDS = [
    *v12.V12_COMPONENT_FIELDS,
    "v13_component_join_actions",
]

V13_COMPONENT_MEMBER_FIELDS = [*v12.V12_COMPONENT_MEMBER_FIELDS]

V13_REVIEW_QUEUE_FIELDS = [
    "v13_review_priority",
    "v13_review_category",
    *[field for field in V13_EDGE_FIELDS if field != "v13_review_priority"],
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


def title_blob(edge: dict[str, str]) -> str:
    return " ".join([
        str(edge.get("prev_primary_title") or ""),
        str(edge.get("primary_title") or ""),
        str(edge.get("v11_prev_classifier_titles") or ""),
        str(edge.get("v11_classifier_titles") or ""),
    ])


def scope_terms(value: object) -> set[str]:
    text = str(value or "").lower()
    terms: set[str] = set()
    if re.search(r"fly\s*-?\s*sheet", text):
        terms.add("flysheet")
    if re.search(r"\bpart\s+(?:[a-z]|[ivx]+|\d+)\b", text):
        terms.add("part")
    if "summary" in text:
        terms.add("summary")
    if any(token in text for token in ["details for", "district", "state figures", "states", "province", "provincial", "agency"]):
        terms.add("geo_detail")
    if any(token in text for token in ["divorc", "married", "unmarried", "widowed"]):
        terms.add("special_civil_status")
    if any(token in text for token in ["caste", "tribe", "race"]):
        terms.add("caste_race")
    return terms


def prev_cur_scope_terms(edge: dict[str, str]) -> tuple[set[str], set[str]]:
    return scope_terms(edge.get("prev_primary_title")), scope_terms(edge.get("primary_title"))


def same_family_and_key(edge: dict[str, str]) -> bool:
    return bool(
        str(edge.get("prev_table_family_key") or "")
        and str(edge.get("prev_table_family_key") or "") == str(edge.get("table_family_key") or "")
        and normalize_table_key(edge.get("prev_table_number_key"))
        and normalize_table_key(edge.get("prev_table_number_key")) == normalize_table_key(edge.get("table_number_key"))
    )


def table_family_switch(edge: dict[str, str]) -> bool:
    prev_family = str(edge.get("prev_table_family_key") or "")
    cur_family = str(edge.get("table_family_key") or "")
    return bool(prev_family and cur_family and prev_family != cur_family)


def is_classifier_main_pair(edge: dict[str, str]) -> bool:
    return (
        str(edge.get("v11_prev_classifier_page_classes") or "") == v11.MAIN_CLASS
        and str(edge.get("v11_classifier_page_classes") or "") == v11.MAIN_CLASS
    )


def same_normalized_key(edge: dict[str, str]) -> bool:
    prev_key = normalize_table_key(edge.get("prev_table_number_key"))
    cur_key = normalize_table_key(edge.get("table_number_key"))
    return bool(prev_key and cur_key and prev_key == cur_key)


def core_age_civil_subject(edge: dict[str, str]) -> bool:
    text = base.clean_text(title_blob(edge))
    return "age" in text and ("civil condition" in text or "religion" in text)


def split_title_fragment(edge: dict[str, str]) -> bool:
    prev = base.clean_text(edge.get("prev_primary_title"))
    cur = base.clean_text(edge.get("primary_title"))
    return bool(re.match(r"^(by religion|and civil condition)\b", prev) or re.match(r"^(by religion|and civil condition)\b", cur))


def core_numeric_form_fragment(edge: dict[str, str]) -> bool:
    text = base.clean_text(title_blob(edge))
    return (
        ("form via" in text or "distribution of the population by civil condition" in text)
        and "age" in text
        and "civil condition" in text
    )


def plain_same_table_high(edge: dict[str, str]) -> bool:
    prev_terms, cur_terms = prev_cur_scope_terms(edge)
    max_sim = max(as_float(edge.get("strict_title_similarity")), as_float(edge.get("scope_similarity")))
    return (
        not prev_terms
        and not cur_terms
        and str(edge.get("prev_table_family_key") or "") == "main_age_civil"
        and normalize_table_key(edge.get("prev_table_number_key")) == "7"
        and same_family_and_key(edge)
        and max_sim >= 0.85
    )


def same_scope_short_title_high(edge: dict[str, str]) -> bool:
    prev_terms, cur_terms = prev_cur_scope_terms(edge)
    max_sim = max(as_float(edge.get("strict_title_similarity")), as_float(edge.get("scope_similarity")))
    unsafe_terms = {"geo_detail", "special_civil_status", "summary", "flysheet"}
    return (
        prev_terms
        and prev_terms == cur_terms
        and not (prev_terms & unsafe_terms)
        and same_family_and_key(edge)
        and max_sim >= 0.74
    )


def refine_edge(edge: dict[str, str]) -> dict[str, object]:
    action = str(edge.get("v12_recommended_action") or "")
    v13_action = action
    layer = str(edge.get("v12_recommended_layer") or "")
    confidence = str(edge.get("v12_recommended_confidence") or "")
    reason = str(edge.get("v12_recommended_reason") or "")
    priority = str(edge.get("v12_review_priority") or "")

    if action == "review_related_not_grouped" and table_family_switch(edge):
        v13_action = "hard_break_family_switch"
        layer = "hard_break"
        confidence = "high"
        reason = "table_family_switch_blocks_assembly"
        priority = ""
    elif action == "review_same_table_boundary_not_joined":
        prev_terms, cur_terms = prev_cur_scope_terms(edge)
        if prev_terms != cur_terms and (prev_terms or cur_terms):
            v13_action = "hard_break_scope_modifier_switch"
            layer = "hard_break"
            confidence = "high"
            reason = f"scope_modifier_switch_{sorted(prev_terms)}_to_{sorted(cur_terms)}"
            priority = ""
        elif plain_same_table_high(edge):
            v13_action = "join_plain_same_table_high"
            layer = "same_table_body_sequence"
            confidence = "high_review"
            reason = "adjacent_same_table_plain_title_high_similarity"
            priority = ""
        elif same_scope_short_title_high(edge):
            v13_action = "join_same_scope_short_title_high"
            layer = "same_scope_short_title_body_sequence"
            confidence = "high_review"
            reason = f"adjacent_same_scope_short_title_same_table_{sorted(prev_terms)}"
            priority = ""
        else:
            priority = "3"
    elif action == "review_classifier_main_tablevii_low_similarity":
        if is_classifier_main_pair(edge) and same_normalized_key(edge) and core_age_civil_subject(edge) and split_title_fragment(edge):
            v13_action = "join_classifier_split_title_tablevii_high"
            layer = "classifier_split_title_body_sequence"
            confidence = "high_review"
            reason = "classifier_main_tablevii_adjacent_split_title_fragment"
            priority = ""
    elif action == "review_classifier_main_tablevii_boundary":
        if is_classifier_main_pair(edge) and core_age_civil_subject(edge) and core_numeric_form_fragment(edge):
            v13_action = "join_classifier_core_numeric_form_high"
            layer = "classifier_core_numeric_form_sequence"
            confidence = "high_review"
            reason = "classifier_main_numeric_core_form_via_or_civil_condition_fragment"
            priority = ""

    return {
        **edge,
        "v13_recommended_action": v13_action,
        "v13_recommended_layer": layer,
        "v13_recommended_confidence": confidence,
        "v13_recommended_reason": reason,
        "v13_review_priority": priority,
        # Reuse the v11 component builder with v13-refined actions.
        "v11_recommended_action": v13_action,
        "v11_recommended_reason": reason,
    }


def review_queue(edges: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for edge in edges:
        priority = str(edge.get("v13_review_priority") or "")
        action = str(edge.get("v13_recommended_action") or "")
        if not priority:
            continue
        if action in JOIN_ACTIONS or action.startswith("hard_break"):
            continue
        rows.append({
            "v13_review_priority": priority,
            "v13_review_category": action.replace("review_", ""),
            **edge,
        })
    rows.sort(key=lambda row: (
        as_int(row.get("v13_review_priority")) or 99,
        str(row.get("v5_source_key") or ""),
        as_int(row.get("prev_last_pdf_page")) or 10**9,
        as_int(row.get("first_pdf_page")) or 10**9,
    ))
    return rows


def label_components(components: list[dict[str, object]], edges: list[dict[str, object]]) -> None:
    edge_by_id = {str(edge.get("sequence_edge_id") or ""): edge for edge in edges}
    for component in components:
        component["v12_component_source"] = "v13_scope_refined_edges"
        join_edge_ids = [
            edge_id.strip()
            for edge_id in str(component.get("v11_join_edge_ids") or "").split(";")
            if edge_id.strip()
        ]
        actions = {str(edge_by_id.get(edge_id, {}).get("v13_recommended_action") or "") for edge_id in join_edge_ids}
        component["v13_component_join_actions"] = base.uniq_join(sorted(actions), limit=20)
        if "join_classifier_split_title_tablevii_high" in actions:
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
        if as_int(component.get("n_v10_groups")) > 1 and str(component.get("all_pages_consecutive")) != "True":
            failures.append(f"v13_component_internal_gap:{component.get('v11_component_id')}:{component.get('page_range')}")
    for edge in edges:
        if str(edge.get("v13_recommended_action") or "") in JOIN_ACTIONS and str(edge.get("page_gap") or "") != "1":
            failures.append(f"v13_join_non_adjacent:{edge.get('sequence_edge_id')}:{edge.get('page_gap')}")
    return {
        "v13_invariant_failure_count": len(failures),
        "v13_invariant_failures": failures[:100],
    }


def write_readme(summary: dict[str, object]) -> None:
    lines = [
        "# v13 Scope-Refined Assembly",
        "",
        "No OCR/API calls are made. This pass refines v12 using scope modifiers and classifier-confirmed split/core numeric fragments.",
        "",
        "## Added Joins",
        "",
        "- Classifier-confirmed split-title Table VII fragments.",
        "- Classifier-confirmed Form VIA/core civil-condition fragments.",
        "- Plain same-table high-similarity main Table VII boundaries.",
        "- Same-scope short-title boundaries without geo/detail/special-status scope changes.",
        "",
        "## Added Hard Breaks",
        "",
        "- Remaining related-not-grouped table-family switches.",
        "- Same-table boundaries where only one side has a scope modifier such as fly-sheet, summary, geo detail, special civil-status, or caste/race marker.",
        "",
        "## Summary",
        "",
    ]
    for key in sorted(summary):
        lines.append(f"- `{key}`: {summary[key]}")
    (OUT_DIR / "v13_rules_readme.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    csv.field_size_limit(10 ** 7)
    groups = read_csv(V10_GROUPS)
    members = read_csv(V12_MEMBERS)
    v12_edges = read_csv(V12_EDGES)
    if not groups or not members or not v12_edges:
        raise SystemExit("Missing v10/v12 inputs; run v12 first")

    refined_edges = [refine_edge(edge) for edge in v12_edges]
    components, component_members = v11.build_v11_components(groups, members, refined_edges, join_actions=JOIN_ACTIONS)
    label_components(components, refined_edges)
    queue = review_queue(refined_edges)

    edge_actions = Counter(str(edge.get("v13_recommended_action") or "") for edge in refined_edges)
    component_layers = Counter(str(component.get("v11_component_layer") or "") for component in components)
    summary = {
        "input_v10_groups": len(groups),
        "input_v12_members": len(members),
        "input_v12_edges": len(v12_edges),
        "v13_components": len(components),
        "v13_component_members": len(component_members),
        "v13_edge_rows": len(refined_edges),
        "v13_review_queue_edges": len(queue),
        "v13_join_classifier_main_edges": edge_actions.get("join_classifier_main_tablevii_high", 0),
        "v13_join_continuation_same_table_edges": edge_actions.get("join_continuation_same_table_high", 0),
        "v13_join_classifier_split_title_edges": edge_actions.get("join_classifier_split_title_tablevii_high", 0),
        "v13_join_classifier_core_numeric_form_edges": edge_actions.get("join_classifier_core_numeric_form_high", 0),
        "v13_join_plain_same_table_edges": edge_actions.get("join_plain_same_table_high", 0),
        "v13_join_same_scope_short_title_edges": edge_actions.get("join_same_scope_short_title_high", 0),
        "v13_hard_break_family_switch_edges": edge_actions.get("hard_break_family_switch", 0),
        "v13_hard_break_scope_modifier_switch_edges": edge_actions.get("hard_break_scope_modifier_switch", 0),
        "v13_remaining_required_row_label_review_edges": edge_actions.get("review_required_row_label_join_tablelike", 0),
        "v13_remaining_same_table_review_edges": edge_actions.get("review_same_table_boundary_not_joined", 0),
        "v13_remaining_related_review_edges": edge_actions.get("review_related_not_grouped", 0),
        "v13_kept_missing_bridges": edge_actions.get("keep_missing_bridge", 0),
        "v13_component_layer_counts": dict(component_layers),
        "v13_edge_action_counts": dict(edge_actions),
    }
    summary.update(validate(components, refined_edges))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base.write_csv(OUT_DIR / "v13_edge_recommendations.csv", refined_edges, V13_EDGE_FIELDS)
    base.write_csv(OUT_DIR / "v13_components.csv", components, V13_COMPONENT_FIELDS)
    base.write_csv(OUT_DIR / "v13_component_members.csv", component_members, V13_COMPONENT_MEMBER_FIELDS)
    base.write_csv(OUT_DIR / "v13_edge_review_queue.csv", queue, V13_REVIEW_QUEUE_FIELDS)
    base.write_csv(OUT_DIR / "v13_member_classifier_overlay_passthrough.csv", members, v11.V11_MEMBER_OVERLAY_FIELDS)
    base.write_csv(OUT_DIR / "v13_required_physical_review_passthrough.csv", read_csv(V10_REQUIRED_CANDIDATES), v10.V10_REQUIRED_CANDIDATE_FIELDS)
    base.write_csv(OUT_DIR / "v13_required_physical_edges_passthrough.csv", read_csv(V10_REQUIRED_EDGES), v10.V10_REQUIRED_EDGE_FIELDS)
    bridges = read_csv(V10_BRIDGES)
    overassoc = read_csv(V10_OVERASSOC)
    base.write_csv(OUT_DIR / "v13_missing_bridge_candidates_passthrough.csv", bridges, list(bridges[0].keys()) if bridges else [])
    base.write_csv(OUT_DIR / "v13_old_overassociation_audit_passthrough.csv", overassoc, list(overassoc[0].keys()) if overassoc else [])

    with (OUT_DIR / "v13_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    write_readme(summary)

    summary_rows = [
        {"metric": key, "value": json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value}
        for key, value in summary.items()
    ]
    base.write_workbook(
        OUT_DIR / "v13_scope_refined_review.xlsx",
        {
            "summary": (summary_rows, ["metric", "value"]),
            "components": (components, V13_COMPONENT_FIELDS),
            "component_members": (component_members, V13_COMPONENT_MEMBER_FIELDS),
            "edge_recommendations": (refined_edges, V13_EDGE_FIELDS),
            "edge_review_queue": (queue, V13_REVIEW_QUEUE_FIELDS),
            "required_physical_review": (read_csv(V10_REQUIRED_CANDIDATES), v10.V10_REQUIRED_CANDIDATE_FIELDS),
        },
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
