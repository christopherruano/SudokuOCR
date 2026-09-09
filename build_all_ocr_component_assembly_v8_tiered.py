#!/usr/bin/env python3
"""Build tiered, adjacency-bounded table assemblies.

This pass keeps v6 physical joins as the required row-label/half-spread layer
and adds a more explicit logical layer on top of v7:

* only adjacent physical runs can join a sequence;
* missing pages become bridge candidates, never joins;
* slightly weaker same-table continuations can join when source/table/entity
  evidence is otherwise stable;
* risky adjacent relationships are exported as candidate edges instead of being
  silently dropped or over-merged.

No OCR/API work is performed here.
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path

import build_all_ocr_component_assembly_v5_structural as base
import build_all_ocr_component_assembly_v7_sequences as v7


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "export_final/pdf_recovery_incoming_2026-06-16_all_ocr_component_assembly_v8_tiered"

ORIGINAL_SEQUENCE_DECISION = v7.sequence_decision

NUMERICISH_PAGE_KINDS = {
    "main_age_civil_numeric",
    "other_table",
    "structured_unknown",
}

CORE_TABLE_FAMILIES = {
    "age civil condition",
    "age_sex_civil_condition",
    "age sex civil condition",
    "age_civil_condition",
    "age_sex_population",
    "age sex population",
    "age_by_religion",
    "age by religion",
}

V8_DETAILED_EDGE_FIELDS = [
    *v7.SEQUENCE_EDGE_FIELDS,
    "v8_assembly_level",
    "v8_join_confidence",
    "v8_rule_family",
    "v8_candidate_related",
    "v8_candidate_reason",
    "prev_run_requires_internal_assembly",
    "run_requires_internal_assembly",
    "prev_row_label_min_ratio",
    "row_label_min_ratio",
    "prev_row_label_deficient_pages",
    "row_label_deficient_pages",
    "prev_no_row_label_pages",
    "no_row_label_pages",
    "prev_layout_left_stub_pages",
    "layout_left_stub_pages",
    "prev_layout_right_half_pages",
    "layout_right_half_pages",
    "prev_internal_join_reasons",
    "internal_join_reasons",
]

V8_CANDIDATE_RELATED_FIELDS = [
    "review_priority",
    "candidate_category",
    *V8_DETAILED_EDGE_FIELDS,
]


def clean(value: object) -> str:
    return base.clean_text(value)


def has_continuation_text(*values: object) -> bool:
    blob = clean(" ".join(str(v or "") for v in values))
    return bool(re.search(r"\b(contd|continued|continuation|concluded|concld|concl)\b", blob))


def numericish_subject_compatible(prev: dict[str, object], cur: dict[str, object]) -> tuple[bool, str]:
    prev_kind = str(prev.get("_page_kind") or "")
    cur_kind = str(cur.get("_page_kind") or "")
    if prev_kind not in NUMERICISH_PAGE_KINDS or cur_kind not in NUMERICISH_PAGE_KINDS:
        return False, f"non_numericish_page_kind_{prev_kind}_to_{cur_kind}"
    if "derivative_selected_caste_race_tribe" in {prev_kind, cur_kind}:
        return False, "derivative_selected_caste_race_tribe_boundary"

    prev_table = str(prev.get("_table_number_key") or "")
    cur_table = str(cur.get("_table_number_key") or "")
    if prev_table and cur_table and prev_table != cur_table:
        return False, f"table_number_switch_{prev_table}_to_{cur_table}"

    prev_family = clean(prev.get("_table_family_key") or "")
    cur_family = clean(cur.get("_table_family_key") or "")
    if prev_family and cur_family and prev_family != cur_family:
        if prev_table and prev_table == cur_table and prev_family in CORE_TABLE_FAMILIES and cur_family in CORE_TABLE_FAMILIES:
            return True, f"compatible_core_table_family_switch_{prev_family}_to_{cur_family}"
        return False, f"table_family_switch_{prev_family}_to_{cur_family}"
    return True, "numericish_same_table_subject"


def same_nonempty_entities(prev: dict[str, object], cur: dict[str, object], field: str) -> bool:
    a = prev.get(field) if isinstance(prev.get(field), set) else set()
    b = cur.get(field) if isinstance(cur.get(field), set) else set()
    return bool(a and b and a == b)


def relaxed_same_table_sequence_decision(
    prev: dict[str, object],
    cur: dict[str, object],
    existing_edge: dict[str, str] | None,
) -> dict[str, object] | None:
    prev_last = v7.last_page_num(prev)
    cur_first = v7.first_page_num(cur)
    if prev_last is None or cur_first is None:
        return None
    page_gap = cur_first - prev_last
    if page_gap <= 0:
        return None

    subject_ok, subject_reason = v7.subject_compatible(prev, cur)
    if not subject_ok:
        subject_ok, subject_reason = numericish_subject_compatible(prev, cur)
        if not subject_ok:
            return None

    entity_conflict, entity_conflicts = v7.has_conflicting_entities(prev, cur)
    if entity_conflict:
        return None

    strict_sim = base.text_similarity(prev.get("_strict_title_key"), cur.get("_strict_title_key"))
    scope_sim = base.text_similarity(prev.get("_scope_without_religion_key"), cur.get("_scope_without_religion_key"))
    prev_table = str(prev.get("_table_number_key") or "")
    cur_table = str(cur.get("_table_number_key") or "")
    same_table_number = bool(prev_table and cur_table and prev_table == cur_table)
    existing_candidate = bool(existing_edge and existing_edge.get("collection_candidate") == "yes")
    continuation = has_continuation_text(prev.get("primary_title"), cur.get("primary_title"))
    same_specific_entity = any(
        same_nonempty_entities(prev, cur, field)
        for field in ["_religion_entities", "_part_entities", "_sex_entities", "_parenthetical_numbers"]
    )

    join_type = ""
    reason = ""
    flags = [
        "v8_relaxed_logical_join_review",
        *v7.missing_entity_flags(prev, cur),
    ]
    if existing_edge and "no_ocr_structural_profile_for_pair" in str(existing_edge.get("edge_review_flags") or ""):
        flags.append("no_ocr_structural_profile_for_sequence_edge")
    if subject_reason != "same_sequence_subject":
        flags.append(subject_reason)

    if same_table_number and strict_sim >= 0.84 and scope_sim >= 0.78 and (continuation or same_specific_entity or existing_candidate):
        join_type = "same_published_table_contiguous_sequence_v8"
        reason = (
            "adjacent_same_published_table_relaxed_threshold; "
            f"strict_title_similarity_{strict_sim:.2f}; scope_similarity_{scope_sim:.2f}"
        )
        flags.append("same_table_number_relaxed_title_threshold")
    elif existing_candidate and strict_sim >= 0.82 and scope_sim >= 0.82:
        join_type = "existing_collection_candidate_sequence_v8"
        reason = (
            "adjacent_existing_collection_candidate_relaxed_threshold; "
            f"strict_title_similarity_{strict_sim:.2f}; scope_similarity_{scope_sim:.2f}"
        )
        flags.append("existing_collection_candidate_review")
    else:
        return None

    missing_pages = ""
    if page_gap > 1:
        missing_pages = f"{prev_last + 1}" if page_gap == 2 else f"{prev_last + 1}-{cur_first - 1}"
        return {
            "page_gap": page_gap,
            "missing_pages": missing_pages,
            "strict_title_similarity": base.float_fmt(strict_sim),
            "scope_similarity": base.float_fmt(scope_sim),
            "sequence_edge_action": "bridge_candidate",
            "sequence_join_type": join_type,
            "sequence_edge_reason": f"missing_pages_block_sequence_join_{missing_pages}; {reason}",
            "sequence_review_flags": base.uniq_join(["missing_intermediate_pages", "not_joined_across_gap", *flags], limit=20),
        }

    if page_gap != 1:
        return None

    return {
        "page_gap": page_gap,
        "missing_pages": "",
        "strict_title_similarity": base.float_fmt(strict_sim),
        "scope_similarity": base.float_fmt(scope_sim),
        "sequence_edge_action": "join_sequence",
        "sequence_join_type": join_type,
        "sequence_edge_reason": reason,
        "sequence_review_flags": base.uniq_join(flags, limit=20),
    }


def sequence_decision_v8(prev: dict[str, object], cur: dict[str, object], existing_edge: dict[str, str] | None) -> dict[str, object]:
    original = ORIGINAL_SEQUENCE_DECISION(prev, cur, existing_edge)
    if original.get("sequence_edge_action") != "break":
        return original
    relaxed = relaxed_same_table_sequence_decision(prev, cur, existing_edge)
    return relaxed or original


def as_float(value: object) -> float | None:
    try:
        return float(str(value or "").strip())
    except ValueError:
        return None


def run_structural_features(run_members: list[dict[str, object]]) -> dict[str, object]:
    ratios = [r for r in (as_float(m.get("profile_row_label_ratio")) for m in run_members) if r is not None]
    deficient = [
        str(m.get("pdf_page") or "")
        for m in run_members
        if (as_float(m.get("profile_row_label_ratio")) is not None and (as_float(m.get("profile_row_label_ratio")) or 0) < 0.55)
    ]
    no_labels = [
        str(m.get("pdf_page") or "")
        for m in run_members
        if str(m.get("profile_loaded") or "") == "True" and not str(m.get("profile_first_row_labels") or "").strip()
    ]
    internal_join_reasons = [
        str(m.get("edge_from_prev_reason") or "")
        for m in run_members
        if str(m.get("edge_from_prev_action") or "") == "join"
    ]
    return {
        "run_requires_internal_assembly": "yes" if internal_join_reasons else "no",
        "row_label_min_ratio": base.float_fmt(min(ratios)) if ratios else "",
        "row_label_deficient_pages": base.uniq_join(deficient, limit=20),
        "no_row_label_pages": base.uniq_join(no_labels, limit=20),
        "layout_left_stub_pages": base.uniq_join((m.get("pdf_page", "") for m in run_members if m.get("layout_left_stub_candidate") == "yes"), limit=20),
        "layout_right_half_pages": base.uniq_join((m.get("pdf_page", "") for m in run_members if m.get("layout_right_half_candidate") == "yes"), limit=20),
        "internal_join_reasons": base.uniq_join(internal_join_reasons, limit=12),
    }


def candidate_related_reason(edge: dict[str, object]) -> tuple[str, str]:
    if edge.get("sequence_edge_action") != "break":
        return "", ""
    if str(edge.get("page_gap") or "") != "1":
        return "", ""
    strict = as_float(edge.get("strict_title_similarity")) or 0.0
    scope = as_float(edge.get("scope_similarity")) or 0.0
    prev_table = str(edge.get("prev_table_number_key") or "")
    table = str(edge.get("table_number_key") or "")
    same_table = bool(prev_table and table and prev_table == table)
    existing_candidate = str(edge.get("existing_collection_candidate") or "") == "yes"
    reason = str(edge.get("sequence_edge_reason") or "")
    subject_switch = "sequence_subject_switch" in str(edge.get("sequence_review_flags") or "")

    if existing_candidate and max(strict, scope) >= 0.72:
        return "yes", f"existing_collection_candidate_not_grouped; strict={strict:.2f}; scope={scope:.2f}"
    if same_table and max(strict, scope) >= 0.72 and "non_sequence_page_kind_context_or_blank" not in reason:
        return "yes", f"same_table_number_adjacent_not_grouped; strict={strict:.2f}; scope={scope:.2f}"
    if subject_switch and same_table and max(strict, scope) >= 0.55:
        return "yes", f"classifier_subject_switch_but_same_table_number; strict={strict:.2f}; scope={scope:.2f}"
    return "", ""


def detailed_edge_rows(
    sequence_edges: list[dict[str, object]],
    members_by_run: dict[str, list[dict[str, object]]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    feature_by_run = {run_id: run_structural_features(rows) for run_id, rows in members_by_run.items()}
    detailed: list[dict[str, object]] = []
    candidates: list[dict[str, object]] = []
    for edge in sequence_edges:
        prev_features = feature_by_run.get(str(edge.get("prev_physical_run_id") or ""), {})
        cur_features = feature_by_run.get(str(edge.get("physical_run_id") or ""), {})
        action = str(edge.get("sequence_edge_action") or "")
        join_type = str(edge.get("sequence_join_type") or "")
        flags = str(edge.get("sequence_review_flags") or "")
        candidate, candidate_reason = candidate_related_reason(edge)
        if action == "bridge_candidate":
            level = "missing_bridge_candidate"
            confidence = "review"
        elif action == "join_sequence" and "religion_variant" in join_type:
            level = "bounded_variant_set"
            confidence = "review" if flags else "medium"
        elif action == "join_sequence" and join_type.endswith("_v8"):
            level = "bounded_same_table_sequence"
            confidence = "review"
        elif action == "join_sequence":
            level = "bounded_same_table_sequence"
            confidence = "review" if flags else "high"
        elif candidate:
            level = "candidate_related_not_grouped"
            confidence = "review"
        else:
            level = "break"
            confidence = ""
        row = {
            **edge,
            "v8_assembly_level": level,
            "v8_join_confidence": confidence,
            "v8_rule_family": join_type or action,
            "v8_candidate_related": candidate,
            "v8_candidate_reason": candidate_reason,
            "prev_run_requires_internal_assembly": prev_features.get("run_requires_internal_assembly", ""),
            "run_requires_internal_assembly": cur_features.get("run_requires_internal_assembly", ""),
            "prev_row_label_min_ratio": prev_features.get("row_label_min_ratio", ""),
            "row_label_min_ratio": cur_features.get("row_label_min_ratio", ""),
            "prev_row_label_deficient_pages": prev_features.get("row_label_deficient_pages", ""),
            "row_label_deficient_pages": cur_features.get("row_label_deficient_pages", ""),
            "prev_no_row_label_pages": prev_features.get("no_row_label_pages", ""),
            "no_row_label_pages": cur_features.get("no_row_label_pages", ""),
            "prev_layout_left_stub_pages": prev_features.get("layout_left_stub_pages", ""),
            "layout_left_stub_pages": cur_features.get("layout_left_stub_pages", ""),
            "prev_layout_right_half_pages": prev_features.get("layout_right_half_pages", ""),
            "layout_right_half_pages": cur_features.get("layout_right_half_pages", ""),
            "prev_internal_join_reasons": prev_features.get("internal_join_reasons", ""),
            "internal_join_reasons": cur_features.get("internal_join_reasons", ""),
        }
        detailed.append(row)
        if candidate:
            priority = 2 if "classifier_subject_switch" in candidate_reason else 3
            candidates.append({"review_priority": priority, "candidate_category": level, **row})
    candidates.sort(key=lambda r: (
        int(r.get("review_priority") or 99),
        str(r.get("v5_source_key") or ""),
        base.int_or_none(r.get("prev_last_pdf_page")) if base.int_or_none(r.get("prev_last_pdf_page")) is not None else 10**9,
    ))
    return detailed, candidates


def core_subsets(
    sequence_groups: list[dict[str, object]],
    sequence_members: list[dict[str, object]],
    sequence_edges: list[dict[str, object]],
    bridge_candidates: list[dict[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    group_ids = {str(group.get("sequence_group_id")) for group in sequence_groups if v7.is_core_table_vii_group(group)}
    return (
        [group for group in sequence_groups if str(group.get("sequence_group_id")) in group_ids],
        [member for member in sequence_members if str(member.get("sequence_group_id")) in group_ids],
        [
            edge
            for edge in sequence_edges
            if v7.is_core_table_vii_edge(edge) and edge.get("sequence_edge_action") in {"join_sequence", "bridge_candidate"}
        ],
        [edge for edge in bridge_candidates if v7.is_core_table_vii_edge(edge)],
    )


def write_rules_readme(path: Path, summary: dict[str, object]) -> None:
    lines = [
        "# v8 Tiered Assembly Rules",
        "",
        "This output is assembly logic only. It performs no OCR/API calls.",
        "",
        "## Layers",
        "",
        "- `required_physical`: inherited from v6 physical runs. These are adjacent pages that must be assembled to interpret the table, usually because a right-side page lacks row labels or image layout shows a half-spread.",
        "- `bounded_same_table_sequence`: adjacent physical runs that appear to be the same published table or continuation. These are logical groupings, not proof that every page physically depends on the prior page.",
        "- `bounded_variant_set`: adjacent same-scope subtables, especially religion variants, where the title scope is compatible and no part/sex/parenthetical-number conflict is detected.",
        "- `missing_bridge_candidate`: related pages separated by missing PDF pages. These are never joined; the missing page range is recorded.",
        "- `candidate_related_not_grouped`: adjacent pages with evidence of relation that remains too risky to group automatically, usually because of classifier/page-kind/family switches or weaker similarity.",
        "- `break`: no safe assembly relation found.",
        "",
        "## Hard Invariants",
        "",
        "- Sequence joins require the same source key and adjacent real PDF pages (`page_gap == 1`).",
        "- Missing intermediate PDF pages are bridge candidates, not joins.",
        "- Non-increasing page order is always a break.",
        "- Explicit table-number switches are always breaks.",
        "- Part, sex, and parenthetical-number conflicts are breaks unless the entity is missing on one side and flagged for review.",
        "",
        "## Relaxed v8 Joins",
        "",
        "- v8 adds reviewed logical joins below the v7 title threshold only when the table number/source/subject evidence is stable.",
        "- These joins carry `v8_relaxed_logical_join_review` and should be treated as reviewable logical sequence edges.",
        "- The relaxed path still cannot cross a missing page; it becomes a bridge candidate instead.",
        "",
        "## Summary",
        "",
    ]
    for key in sorted(summary):
        lines.append(f"- `{key}`: {summary[key]}")
    (path / "v8_rules_readme.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    csv.field_size_limit(10 ** 7)
    runs_raw = v7.read_csv(v7.PHYSICAL_RUNS)
    members_raw = v7.read_csv(v7.RUN_MEMBERS)
    edges_raw = v7.read_csv(v7.PHYSICAL_EDGES)
    required_physical_edges = v7.read_csv(v7.REQUIRED_PHYSICAL_EDGES)
    old_collections = v7.read_csv(v7.OLD_COLLECTION_GROUPS)
    old_collection_members = v7.read_csv(v7.OLD_COLLECTION_MEMBERS)
    if not runs_raw or not members_raw or not edges_raw:
        raise SystemExit(f"Missing v6 strict inputs under {v7.V6_DIR}")

    runs, members_by_run, _features = v7.enrich_runs(runs_raw, members_raw)
    edge_by_run_pair = v7.build_existing_edge_map(edges_raw, members_raw)

    v7.sequence_decision = sequence_decision_v8
    try:
        sequence_groups, sequence_members, sequence_edges, bridge_candidates = v7.build_sequences(runs, edge_by_run_pair)
    finally:
        v7.sequence_decision = ORIGINAL_SEQUENCE_DECISION

    sequence_review_queue = v7.build_sequence_review_queue(sequence_edges, sequence_groups)
    comparison_rows = v7.old_collection_comparison(old_collections, sequence_groups)
    old_overassociation_rows = v7.old_overassociation_audit(
        old_collections,
        old_collection_members,
        sequence_groups,
        sequence_members,
    )
    for row in old_overassociation_rows:
        row["audit_flags"] = str(row.get("audit_flags") or "").replace(
            "old_collection_split_by_adjacency_v7",
            "old_collection_split_by_adjacency_v8",
        )
    detailed_edges, candidate_edges = detailed_edge_rows(sequence_edges, members_by_run)
    core_groups, core_members, core_edges, core_bridges = core_subsets(
        sequence_groups,
        sequence_members,
        sequence_edges,
        bridge_candidates,
    )

    summary = {
        "input_v6_physical_runs": len(runs),
        "input_v6_run_members": len(members_raw),
        "input_v6_required_physical_edges": len(required_physical_edges),
        "sequence_groups": len(sequence_groups),
        "multi_run_sequence_groups": sum(1 for g in sequence_groups if int(g.get("n_physical_runs") or 0) > 1),
        "max_sequence_physical_runs": max((int(g.get("n_physical_runs") or 0) for g in sequence_groups), default=0),
        "max_sequence_physical_pages": max((int(g.get("n_physical_pages") or 0) for g in sequence_groups), default=0),
        "joined_sequence_edges": sum(1 for e in sequence_edges if e.get("sequence_edge_action") == "join_sequence"),
        "v8_relaxed_join_sequence_edges": sum(
            1
            for e in sequence_edges
            if e.get("sequence_edge_action") == "join_sequence" and str(e.get("sequence_join_type") or "").endswith("_v8")
        ),
        "v8_relaxed_bridge_candidate_edges": sum(
            1
            for e in sequence_edges
            if e.get("sequence_edge_action") == "bridge_candidate" and str(e.get("sequence_join_type") or "").endswith("_v8")
        ),
        "bridge_candidate_edges": len(bridge_candidates),
        "break_sequence_edges": sum(1 for e in sequence_edges if e.get("sequence_edge_action") == "break"),
        "candidate_related_not_grouped_edges": len(candidate_edges),
        "sequence_review_queue_rows": len(sequence_review_queue),
        "religion_variant_sequence_groups": sum(1 for g in sequence_groups if g.get("sequence_basis") == "bounded_religion_variant_subtable_sequence"),
        "large_sequence_review_groups": sum(1 for g in sequence_groups if "large_bounded_sequence_review" in str(g.get("sequence_review_flags") or "")),
        "old_v6_multi_run_collection_groups": sum(1 for g in old_collections if int(g.get("n_physical_runs") or 0) > 1),
        "old_v6_max_collection_runs": max((int(g.get("n_physical_runs") or 0) for g in old_collections), default=0),
        "old_v6_overassociation_audit_rows": len(old_overassociation_rows),
        "old_v6_gap_crossing_audit_rows": sum(1 for g in old_overassociation_rows if int(g.get("old_missing_gap_count") or 0) > 0),
        "old_v6_split_by_v8_audit_rows": sum(1 for g in old_overassociation_rows if "old_collection_split_by_adjacency_v8" in str(g.get("audit_flags") or "")),
        "core_table_vii_sequence_groups": len(core_groups),
        "core_table_vii_multi_run_sequence_groups": sum(1 for g in core_groups if int(g.get("n_physical_runs") or 0) > 1),
        "core_table_vii_sequence_members": len(core_members),
        "core_table_vii_sequence_edges": len(core_edges),
        "core_table_vii_bridge_candidates": len(core_bridges),
    }
    summary.update(v7.sequence_invariants(sequence_groups, sequence_edges))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base.write_csv(OUT_DIR / "v8_sequence_groups.csv", sequence_groups, v7.SEQUENCE_GROUP_FIELDS)
    base.write_csv(OUT_DIR / "v8_sequence_members.csv", sequence_members, v7.SEQUENCE_MEMBER_FIELDS)
    base.write_csv(OUT_DIR / "v8_sequence_edges.csv", sequence_edges, v7.SEQUENCE_EDGE_FIELDS)
    base.write_csv(OUT_DIR / "v8_sequence_edges_detailed.csv", detailed_edges, V8_DETAILED_EDGE_FIELDS)
    base.write_csv(OUT_DIR / "v8_candidate_related_edges.csv", candidate_edges, V8_CANDIDATE_RELATED_FIELDS)
    base.write_csv(OUT_DIR / "v8_missing_bridge_candidates.csv", bridge_candidates, v7.SEQUENCE_EDGE_FIELDS)
    base.write_csv(OUT_DIR / "v8_sequence_review_queue.csv", sequence_review_queue, v7.SEQUENCE_REVIEW_FIELDS)
    base.write_csv(OUT_DIR / "v8_collection_comparison.csv", comparison_rows, ["metric", "old_value", "new_value", "note"])
    base.write_csv(OUT_DIR / "v8_old_overassociation_audit.csv", old_overassociation_rows, v7.OLD_OVERASSOCIATION_FIELDS)
    base.write_csv(OUT_DIR / "v8_required_physical_edges_passthrough.csv", required_physical_edges, base.EDGE_FIELDS)
    base.write_csv(OUT_DIR / "v8_core_tableVII_sequence_groups.csv", core_groups, v7.SEQUENCE_GROUP_FIELDS)
    base.write_csv(OUT_DIR / "v8_core_tableVII_sequence_members.csv", core_members, v7.SEQUENCE_MEMBER_FIELDS)
    base.write_csv(OUT_DIR / "v8_core_tableVII_sequence_edges.csv", core_edges, v7.SEQUENCE_EDGE_FIELDS)
    base.write_csv(OUT_DIR / "v8_core_tableVII_bridge_candidates.csv", core_bridges, v7.SEQUENCE_EDGE_FIELDS)

    with (OUT_DIR / "v8_sequence_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    write_rules_readme(OUT_DIR, summary)

    summary_rows = [{"metric": key, "value": json.dumps(value) if isinstance(value, (dict, list)) else value} for key, value in summary.items()]
    base.write_workbook(
        OUT_DIR / "v8_tiered_sequence_review.xlsx",
        {
            "summary": (summary_rows, ["metric", "value"]),
            "sequence_groups": (sequence_groups, v7.SEQUENCE_GROUP_FIELDS),
            "sequence_members": (sequence_members, v7.SEQUENCE_MEMBER_FIELDS),
            "sequence_edges": (sequence_edges, v7.SEQUENCE_EDGE_FIELDS),
            "detailed_edges": (detailed_edges, V8_DETAILED_EDGE_FIELDS),
            "candidate_related": (candidate_edges, V8_CANDIDATE_RELATED_FIELDS),
            "bridge_candidates": (bridge_candidates, v7.SEQUENCE_EDGE_FIELDS),
            "sequence_review_queue": (sequence_review_queue, v7.SEQUENCE_REVIEW_FIELDS),
            "old_overassoc_audit": (old_overassociation_rows, v7.OLD_OVERASSOCIATION_FIELDS),
            "required_physical": (required_physical_edges, base.EDGE_FIELDS),
            "core_vii_groups": (core_groups, v7.SEQUENCE_GROUP_FIELDS),
            "core_vii_members": (core_members, v7.SEQUENCE_MEMBER_FIELDS),
            "core_vii_edges": (core_edges, v7.SEQUENCE_EDGE_FIELDS),
            "core_vii_bridges": (core_bridges, v7.SEQUENCE_EDGE_FIELDS),
            "v6_vs_v8": (comparison_rows, ["metric", "old_value", "new_value", "note"]),
        },
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
