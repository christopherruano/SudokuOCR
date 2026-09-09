#!/usr/bin/env python3
"""Variant-aware refinement of v13 adjacency-first assembly.

No OCR/API calls are made. This pass consumes v13 edge recommendations and
keeps the hard invariant that promoted joins must be adjacent real PDF pages.

Compared with v13, this pass:

* promotes a narrow classifier-confirmed Table VII same-part continuation;
* promotes adjacent same-table edges only when an explicit scope marker matches;
* demotes obvious scope switches, statement-number switches, and subsidiary
  subtable switches to hard breaks instead of review.
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


ROOT = Path(__file__).resolve().parent
V13_DIR = ROOT / "export_final/pdf_recovery_incoming_2026-06-16_all_ocr_component_assembly_v13_scope_refined"
OUT_DIR = ROOT / "export_final/pdf_recovery_incoming_2026-06-16_all_ocr_component_assembly_v14_variant_refined"

V13_EDGES = V13_DIR / "v13_edge_recommendations.csv"
V13_MEMBERS = V13_DIR / "v13_member_classifier_overlay_passthrough.csv"

JOIN_ACTIONS = {
    *v13.JOIN_ACTIONS,
    "join_classifier_same_part_tablevii_continuation_high",
    "join_explicit_same_scope_same_table_high",
}

V14_EDGE_FIELDS = [
    *v13.V13_EDGE_FIELDS,
    "v14_recommended_action",
    "v14_recommended_layer",
    "v14_recommended_confidence",
    "v14_recommended_reason",
    "v14_review_priority",
]

V14_COMPONENT_FIELDS = [
    *v13.V13_COMPONENT_FIELDS,
    "v14_component_join_actions",
]

V14_COMPONENT_MEMBER_FIELDS = [*v13.V13_COMPONENT_MEMBER_FIELDS]

V14_REVIEW_QUEUE_FIELDS = [
    "v14_review_priority",
    "v14_review_category",
    *[field for field in V14_EDGE_FIELDS if field != "v14_review_priority"],
]


def clean(value: object) -> str:
    return base.clean_text(value)


def title_pair_text(edge: dict[str, str]) -> tuple[str, str]:
    return clean(edge.get("prev_primary_title")), clean(edge.get("primary_title"))


def combined_text(edge: dict[str, str]) -> str:
    return clean(
        " ".join(
            [
                str(edge.get("prev_primary_title") or ""),
                str(edge.get("primary_title") or ""),
                str(edge.get("v11_prev_classifier_titles") or ""),
                str(edge.get("v11_classifier_titles") or ""),
            ]
        )
    )


def entity_set(value: object) -> set[str]:
    text = clean(value)
    if not text:
        return set()
    return {part.strip() for part in re.split(r"[;,|]+", text) if part.strip()}


def compatible_entity_sets(prev: object, cur: object) -> bool:
    prev_set = entity_set(prev)
    cur_set = entity_set(cur)
    return not prev_set or not cur_set or prev_set == cur_set


def page_gap_is_adjacent(edge: dict[str, str]) -> bool:
    return str(edge.get("page_gap") or "") == "1"


def selected_derivative_text(text: str) -> bool:
    return bool(re.search(r"\bselected\b.*\b(caste|race|tribe)s?\b", text))


def status_signature(value: object) -> str:
    text = clean(value)
    if "age not stated" in text:
        return "age_not_stated"
    if "unmarried" in text:
        return "unmarried"
    if "divorc" in text or "divorv" in text:
        return "divorced"
    if "widowed" in text or re.search(r"\bwid\b", text) or re.search(r"\bowed\b", text):
        return "widowed"
    if "married" in text:
        return "married"
    if "rural/sample" in text:
        return "rural_sample"
    if "urban/sample" in text:
        return "urban_sample"
    if "general/sample" in text:
        return "general_sample"
    if re.search(r"\bsample\b", text):
        return "sample"
    return ""


def statement_marker(value: object) -> str:
    text = clean(value)
    match = re.search(r"\bstatement\s+no\.?\s*([ivx]+[- ]+\d+|\d+[- ]+\d+)", text)
    return match.group(1).replace(" ", "-") if match else ""


def subsidiary_marker(value: object) -> str:
    raw_text = str(value or "").lower()
    match = re.search(r"\bsubsidiary\s+table\s+[ivx\d]+\s*\(([ivx\d]+)\)", raw_text)
    if match:
        return match.group(1)
    text = clean(value)
    match = re.search(r"\bsubsidiary\s+table\s+[ivx\d]+\s+([ivx\d]+)\b", text)
    return match.group(1) if match else ""


def geo_scope_label(value: object) -> str:
    text = clean(value)
    patterns = [
        r"\bby provinces and states[-\u2014\u2013 ]+([a-z0-9 &'()]+?)(?:\.| age_| table|$)",
        r"\bage, sex and civil condition[-\u2014\u2013 ]+([a-z0-9 &'()]+?)(?:\.| age_| table|$)",
        r"\b(part\s+[abc][-\.\u2014\u2013 ]+details by districts)\b",
        r"\b(state total|towns|rural|urban|details by districts)\b",
        r"\b([a-z &'()]+(?:district|division|state|taluk|city))\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if not match:
            continue
        label = re.sub(r"\s+", " ", match.group(1)).strip(" -\u2014\u2013.;")
        if label:
            return label
    return ""


def explicit_scope(edge: dict[str, str], side: str) -> tuple[str, ...]:
    title = edge.get("prev_primary_title") if side == "prev" else edge.get("primary_title")
    pieces = [
        f"status:{status_signature(title)}" if status_signature(title) else "",
        f"geo:{geo_scope_label(title)}" if geo_scope_label(title) else "",
        f"subsidiary:{subsidiary_marker(title)}" if subsidiary_marker(title) else "",
        f"statement:{statement_marker(title)}" if statement_marker(title) else "",
    ]
    if side == "prev":
        pieces.extend(
            [
                f"part:{clean(edge.get('prev_part_entities'))}" if clean(edge.get("prev_part_entities")) else "",
                f"religion:{clean(edge.get('prev_religion_entities'))}" if clean(edge.get("prev_religion_entities")) else "",
            ]
        )
    else:
        pieces.extend(
            [
                f"part:{clean(edge.get('part_entities'))}" if clean(edge.get("part_entities")) else "",
                f"religion:{clean(edge.get('religion_entities'))}" if clean(edge.get("religion_entities")) else "",
            ]
        )
    return tuple(piece for piece in pieces if piece)


def marker_switch(prev_marker: str, cur_marker: str) -> bool:
    return bool(prev_marker and cur_marker and prev_marker != cur_marker)


def has_obvious_hard_switch(edge: dict[str, str]) -> tuple[bool, str]:
    prev_title, cur_title = title_pair_text(edge)
    prev_status, cur_status = status_signature(prev_title), status_signature(cur_title)
    prev_statement, cur_statement = statement_marker(prev_title), statement_marker(cur_title)
    prev_subsidiary, cur_subsidiary = subsidiary_marker(prev_title), subsidiary_marker(cur_title)
    prev_geo, cur_geo = geo_scope_label(prev_title), geo_scope_label(cur_title)

    if selected_derivative_text(prev_title) or selected_derivative_text(cur_title):
        return True, "selected_caste_race_tribe_derivative_blocks_assembly"
    if prev_status != cur_status and (prev_status or cur_status):
        return True, f"status_scope_switch_{prev_status or 'none'}_to_{cur_status or 'none'}"
    if marker_switch(prev_statement, cur_statement):
        return True, f"statement_number_switch_{prev_statement}_to_{cur_statement}"
    if marker_switch(prev_subsidiary, cur_subsidiary):
        return True, f"subsidiary_subtable_switch_{prev_subsidiary}_to_{cur_subsidiary}"
    if marker_switch(prev_geo, cur_geo):
        return True, f"geo_scope_switch_{prev_geo}_to_{cur_geo}"
    return False, ""


def classifier_same_part_tablevii_continuation(edge: dict[str, str]) -> bool:
    text = combined_text(edge)
    return (
        page_gap_is_adjacent(edge)
        and v13.is_classifier_main_pair(edge)
        and v13.same_normalized_key(edge)
        and v13.normalize_table_key(edge.get("table_number_key")) == "7"
        and str(edge.get("prev_table_family_key") or "") == "main_age_civil"
        and str(edge.get("table_family_key") or "") == "main_age_civil"
        and compatible_entity_sets(edge.get("prev_part_entities"), edge.get("part_entities"))
        and compatible_entity_sets(edge.get("prev_sex_entities"), edge.get("sex_entities"))
        and not selected_derivative_text(text)
        and ("part c" in text or "part b" in text or "part a" in text)
        and ("continued" in text or "contd" in text or "table vii" in text)
    )


def explicit_same_scope_same_table(edge: dict[str, str]) -> bool:
    if not page_gap_is_adjacent(edge) or not v13.same_family_and_key(edge):
        return False
    has_switch, _ = has_obvious_hard_switch(edge)
    if has_switch:
        return False
    prev_scope = set(explicit_scope(edge, "prev"))
    cur_scope = set(explicit_scope(edge, "cur"))
    shared_scope = prev_scope & cur_scope
    if not shared_scope:
        return False
    # Require shared geo/status, or shared part plus shared religion. A shared
    # part alone is too broad and recreates the old over-association failure.
    has_geo_or_status = any(piece.startswith(("geo:", "status:")) for piece in shared_scope)
    has_part_and_religion = (
        any(piece.startswith("part:") for piece in shared_scope)
        and any(piece.startswith("religion:") for piece in shared_scope)
    )
    if not (has_geo_or_status or has_part_and_religion):
        return False
    max_sim = max(v13.as_float(edge.get("strict_title_similarity")), v13.as_float(edge.get("scope_similarity")))
    return max_sim >= 0.72


def refine_edge(edge: dict[str, str]) -> dict[str, object]:
    action = str(edge.get("v13_recommended_action") or "")
    v14_action = action
    layer = str(edge.get("v13_recommended_layer") or "")
    confidence = str(edge.get("v13_recommended_confidence") or "")
    reason = str(edge.get("v13_recommended_reason") or "")
    priority = str(edge.get("v13_review_priority") or "")

    if action == "review_same_table_boundary_not_joined":
        has_switch, switch_reason = has_obvious_hard_switch(edge)
        if has_switch:
            v14_action = "hard_break_explicit_scope_switch"
            layer = "hard_break"
            confidence = "high"
            reason = switch_reason
            priority = ""
        elif explicit_same_scope_same_table(edge):
            v14_action = "join_explicit_same_scope_same_table_high"
            layer = "explicit_same_scope_body_sequence"
            confidence = "high_review"
            reason = f"adjacent_same_table_explicit_scope_match_{';'.join(sorted(set(explicit_scope(edge, 'prev')) & set(explicit_scope(edge, 'cur'))))}"
            priority = ""
        else:
            priority = priority or "3"
    elif action == "review_classifier_main_tablevii_low_similarity":
        if classifier_same_part_tablevii_continuation(edge):
            v14_action = "join_classifier_same_part_tablevii_continuation_high"
            layer = "classifier_same_part_tablevii_sequence"
            confidence = "high_review"
            reason = "classifier_main_tablevii_adjacent_same_part_numeric_continuation"
            priority = ""
        else:
            priority = priority or "2"

    return {
        **edge,
        "v14_recommended_action": v14_action,
        "v14_recommended_layer": layer,
        "v14_recommended_confidence": confidence,
        "v14_recommended_reason": reason,
        "v14_review_priority": priority,
        # Reuse the v11 component builder with v14-refined actions.
        "v11_recommended_action": v14_action,
        "v11_recommended_reason": reason,
    }


def review_queue(edges: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for edge in edges:
        priority = str(edge.get("v14_review_priority") or "")
        action = str(edge.get("v14_recommended_action") or "")
        if not priority:
            continue
        if action in JOIN_ACTIONS or action.startswith("hard_break"):
            continue
        rows.append(
            {
                "v14_review_priority": priority,
                "v14_review_category": action.replace("review_", ""),
                **edge,
            }
        )
    rows.sort(
        key=lambda row: (
            v13.as_int(row.get("v14_review_priority")) or 99,
            str(row.get("v5_source_key") or ""),
            v13.as_int(row.get("prev_last_pdf_page")) or 10**9,
            v13.as_int(row.get("first_pdf_page")) or 10**9,
        )
    )
    return rows


def label_components(components: list[dict[str, object]], edges: list[dict[str, object]]) -> None:
    edge_by_id = {str(edge.get("sequence_edge_id") or ""): edge for edge in edges}
    for component in components:
        component["v12_component_source"] = "v14_variant_refined_edges"
        join_edge_ids = [
            edge_id.strip()
            for edge_id in str(component.get("v11_join_edge_ids") or "").split(";")
            if edge_id.strip()
        ]
        actions = {str(edge_by_id.get(edge_id, {}).get("v14_recommended_action") or "") for edge_id in join_edge_ids}
        component["v14_component_join_actions"] = base.uniq_join(sorted(actions), limit=20)

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
            failures.append(f"v14_component_internal_gap:{component.get('v11_component_id')}:{component.get('page_range')}")
    for edge in edges:
        if str(edge.get("v14_recommended_action") or "") in JOIN_ACTIONS and str(edge.get("page_gap") or "") != "1":
            failures.append(f"v14_join_non_adjacent:{edge.get('sequence_edge_id')}:{edge.get('page_gap')}")
    return {
        "v14_invariant_failure_count": len(failures),
        "v14_invariant_failures": failures[:100],
    }


def write_readme(summary: dict[str, object]) -> None:
    lines = [
        "# v14 Variant-Refined Assembly",
        "",
        "No OCR/API calls are made. This pass refines v13 while preserving the adjacent-real-PDF-page invariant.",
        "",
        "## Added Joins",
        "",
        "- Classifier-confirmed adjacent Table VII same-part numeric continuations.",
        "- Adjacent same-table edges with an explicit matching scope marker such as State Total, Towns, Details by Districts, or the same civil-status/religion/part marker.",
        "",
        "## Added Hard Breaks",
        "",
        "- Special-status switches, including OCR misspellings such as DIVORVED.",
        "- Different explicit province/district/state/division/taluk labels when both sides expose one.",
        "- Statement-number and subsidiary-subtable marker switches.",
        "",
        "## Summary",
        "",
    ]
    for key in sorted(summary):
        lines.append(f"- `{key}`: {summary[key]}")
    (OUT_DIR / "v14_rules_readme.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    csv.field_size_limit(10**7)
    groups = v13.read_csv(v13.V10_GROUPS)
    members = v13.read_csv(V13_MEMBERS)
    v13_edges = v13.read_csv(V13_EDGES)
    if not groups or not members or not v13_edges:
        raise SystemExit("Missing v10/v13 inputs; run v13 first")

    refined_edges = [refine_edge(edge) for edge in v13_edges]
    components, component_members = v11.build_v11_components(groups, members, refined_edges, join_actions=JOIN_ACTIONS)
    label_components(components, refined_edges)
    queue = review_queue(refined_edges)

    edge_actions = Counter(str(edge.get("v14_recommended_action") or "") for edge in refined_edges)
    component_layers = Counter(str(component.get("v11_component_layer") or "") for component in components)
    summary = {
        "input_v10_groups": len(groups),
        "input_v13_members": len(members),
        "input_v13_edges": len(v13_edges),
        "v14_components": len(components),
        "v14_component_members": len(component_members),
        "v14_edge_rows": len(refined_edges),
        "v14_review_queue_edges": len(queue),
        "v14_join_classifier_same_part_tablevii_edges": edge_actions.get("join_classifier_same_part_tablevii_continuation_high", 0),
        "v14_join_explicit_same_scope_edges": edge_actions.get("join_explicit_same_scope_same_table_high", 0),
        "v14_hard_break_explicit_scope_switch_edges": edge_actions.get("hard_break_explicit_scope_switch", 0),
        "v14_kept_missing_bridges": edge_actions.get("keep_missing_bridge", 0),
        "v14_remaining_required_row_label_review_edges": edge_actions.get("review_required_row_label_join_tablelike", 0),
        "v14_remaining_same_table_review_edges": edge_actions.get("review_same_table_boundary_not_joined", 0),
        "v14_remaining_classifier_low_similarity_edges": edge_actions.get("review_classifier_main_tablevii_low_similarity", 0),
        "v14_component_layer_counts": dict(component_layers),
        "v14_edge_action_counts": dict(edge_actions),
    }
    summary.update(validate(components, refined_edges))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base.write_csv(OUT_DIR / "v14_edge_recommendations.csv", refined_edges, V14_EDGE_FIELDS)
    base.write_csv(OUT_DIR / "v14_components.csv", components, V14_COMPONENT_FIELDS)
    base.write_csv(OUT_DIR / "v14_component_members.csv", component_members, V14_COMPONENT_MEMBER_FIELDS)
    base.write_csv(OUT_DIR / "v14_edge_review_queue.csv", queue, V14_REVIEW_QUEUE_FIELDS)
    base.write_csv(OUT_DIR / "v14_member_classifier_overlay_passthrough.csv", members, v11.V11_MEMBER_OVERLAY_FIELDS)
    base.write_csv(OUT_DIR / "v14_required_physical_review_passthrough.csv", v13.read_csv(v13.V10_REQUIRED_CANDIDATES), v10.V10_REQUIRED_CANDIDATE_FIELDS)
    base.write_csv(OUT_DIR / "v14_required_physical_edges_passthrough.csv", v13.read_csv(v13.V10_REQUIRED_EDGES), v10.V10_REQUIRED_EDGE_FIELDS)
    bridges = v13.read_csv(v13.V10_BRIDGES)
    overassoc = v13.read_csv(v13.V10_OVERASSOC)
    base.write_csv(OUT_DIR / "v14_missing_bridge_candidates_passthrough.csv", bridges, list(bridges[0].keys()) if bridges else [])
    base.write_csv(OUT_DIR / "v14_old_overassociation_audit_passthrough.csv", overassoc, list(overassoc[0].keys()) if overassoc else [])

    with (OUT_DIR / "v14_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    write_readme(summary)

    summary_rows = [
        {"metric": key, "value": json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value}
        for key, value in summary.items()
    ]
    base.write_workbook(
        OUT_DIR / "v14_variant_refined_review.xlsx",
        {
            "summary": (summary_rows, ["metric", "value"]),
            "components": (components, V14_COMPONENT_FIELDS),
            "component_members": (component_members, V14_COMPONENT_MEMBER_FIELDS),
            "edge_recommendations": (refined_edges, V14_EDGE_FIELDS),
            "edge_review_queue": (queue, V14_REVIEW_QUEUE_FIELDS),
            "required_physical_review": (v13.read_csv(v13.V10_REQUIRED_CANDIDATES), v10.V10_REQUIRED_CANDIDATE_FIELDS),
        },
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
