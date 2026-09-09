#!/usr/bin/env python3
"""Build v9 adjacency-bounded assemblies with narrow split-title recovery.

This pass performs no OCR/API work. It reuses the v6 physical-run layer and the
v8 tiered logical rules, then adds one conservative reviewed join:

* adjacent Table VII pages whose printed title is split across the page break,
  usually "TABLE VII - AGE, SEX" followed by "AND CIVIL CONDITION";
* the join still requires same source, adjacent real PDF pages, compatible table
  number evidence, non-context/non-derivative page classes, and no hard entity
  conflict;
* missing pages become bridge candidates, never merged sequence members.
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import build_all_ocr_component_assembly_v5_structural as base
import build_all_ocr_component_assembly_v7_sequences as v7
import build_all_ocr_component_assembly_v8_tiered as v8


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "export_final/pdf_recovery_incoming_2026-06-16_all_ocr_component_assembly_v9_split_title"

CORE_SPLIT_PAGE_KINDS = {
    "main_age_civil_numeric",
    "other_table",
    "structured_unknown",
}

CORE_SPLIT_TABLE_FAMILIES = {
    *v8.CORE_TABLE_FAMILIES,
    "main_age_civil",
    "main age civil",
    "table_vii",
    "table vii",
    "age sex civil condition",
    "age civil condition",
    "age sex population",
    "age by religion",
}

V9_DETAILED_EDGE_FIELDS = [
    *v8.V8_DETAILED_EDGE_FIELDS,
    "v9_rule_added",
    "v9_rule_note",
]

V9_CANDIDATE_RELATED_FIELDS = [
    "review_priority",
    "candidate_category",
    *V9_DETAILED_EDGE_FIELDS,
]

V9_LARGE_GROUP_FIELDS = [
    "large_review_rank",
    "sequence_group_id",
    "v5_source_key",
    "source_pdf",
    "source_pdf_id",
    "is_core_table_vii",
    "sequence_basis",
    "n_physical_runs",
    "n_physical_pages",
    "page_range",
    "page_ranges",
    "all_pages_consecutive",
    "sequence_join_types",
    "sequence_review_flags",
    "split_title_join_edges",
    "v8_relaxed_join_edges",
    "variant_join_edges",
    "internal_required_physical_runs",
    "internal_required_page_ranges",
    "page_kind_counts",
    "table_family_counts",
    "table_number_keys",
    "religion_entities",
    "part_entities",
    "sex_entities",
    "parenthetical_numbers",
    "primary_title",
    "sequence_reasons",
    "example_images",
]

V9_EXTRA_NON_CORE_TABLE_VII_TERMS = [
    "average rain",
    "rain fall",
    "rainfall",
]


def clean(value: object) -> str:
    return base.clean_text(value)


def title_blob(run: dict[str, object]) -> str:
    return strip_classifier_tail(str(run.get("primary_title") or ""))


def strip_classifier_tail(value: str) -> str:
    """Remove classifier suffixes appended to titles without erasing titles."""
    text = re.sub(r"[_]+", " ", value)
    suffixes = [
        r"age\s+sex\s+civil\s+condition",
        r"age\s+civil\s+condition",
        r"age\s+sex\s+population",
        r"age\s+by\s+religion",
        r"main\s+age\s+civil",
        r"other\s+structured",
        r"structured\s+unknown",
        r"table\s+vii",
    ]
    number_tail = r"(?:vii|vil|vi1|v1i|7|[abc])?"
    changed = True
    while changed:
        changed = False
        for suffix in suffixes:
            pattern = rf"\s+{suffix}(?:\s+{number_tail})?\s*$"
            new_text = re.sub(pattern, "", text, flags=re.IGNORECASE).strip()
            if new_text != text.strip():
                text = new_text
                changed = True
    return text.strip()


def explicit_table_number(text: object) -> str:
    explicit = v7.first_explicit_table_number(text)
    if explicit:
        return explicit
    # OCR sometimes turns VII into VIL. Treat it as VII only when it appears
    # after an explicit "Table" token, not as a free-floating table key.
    normalized = clean(text)
    if re.search(r"\btable\s+vi[i1l]\b", normalized):
        return "7"
    return ""


def normalized_table_key(value: object) -> str:
    key = clean(value)
    if key in {"7", "vii", "vil", "vi1", "v1i"}:
        return "7"
    return key


def table_keys_compatible(prev: dict[str, object], cur: dict[str, object], prev_text: str, cur_text: str) -> tuple[bool, str]:
    prev_key = normalized_table_key(prev.get("_table_number_key"))
    cur_key = normalized_table_key(cur.get("_table_number_key"))
    combined = f"{prev_text} {cur_text}"
    combined_explicit = explicit_table_number(combined)

    for side_name, side_text in [("prev", prev_text), ("cur", cur_text)]:
        explicit = explicit_table_number(side_text)
        if explicit and explicit != "7":
            return False, f"{side_name}_explicit_table_number_switch_{explicit}"

    if prev_key and cur_key:
        if prev_key == cur_key:
            if prev_key == "7":
                return True, f"same_table_key_{prev_key}"
            if prev_key in {"a", "b", "c"} and combined_explicit == "7":
                return True, f"same_part_letter_key_{prev_key}_with_explicit_table_vii"
            return False, f"same_non_core_table_key_{prev_key}"
        if prev_key == "7" and cur_key == "7":
            return True, "same_normalized_table_vii_key"
        return False, f"table_number_switch_{prev_key}_to_{cur_key}"
    if combined_explicit == "7":
        return True, "explicit_table_vii_in_combined_title"
    return False, "missing_core_table_vii_number_evidence"


def has_age_sex(text: object) -> bool:
    t = clean(text)
    return bool(re.search(r"\bage(?:s)?\b", t)) and bool(re.search(r"\bsex(?:es)?\b", t))


def has_civil_condition(text: object) -> bool:
    t = clean(text)
    return any(term in t for term in ["civil condition", "marital condition", "conjugal condition"])


def split_title_fragment_like(prev_text: str, cur_text: str) -> bool:
    prev_clean = clean(prev_text)
    cur_clean = clean(cur_text)
    combined = f"{prev_text} {cur_text}"
    if explicit_table_number(combined) != "7":
        return False
    if not has_age_sex(combined) or not has_civil_condition(combined):
        return False

    prev_age = has_age_sex(prev_text)
    cur_age = has_age_sex(cur_text)
    prev_civil = has_civil_condition(prev_text)
    cur_civil = has_civil_condition(cur_text)
    prev_table = explicit_table_number(prev_text) == "7"
    cur_table = explicit_table_number(cur_text) == "7"
    prev_civil_only = prev_civil and not prev_age and not prev_table
    cur_civil_only = cur_civil and not cur_age and not cur_table

    civil_lead = bool(re.search(r"^(and\s+)?civil condition\b", cur_clean)) or bool(
        re.search(r"^(and\s+)?civil condition\b", prev_clean)
    )
    continuation_word = v8.has_continuation_text(prev_text, cur_text)
    title_half = (prev_table and prev_age and not prev_civil and cur_civil) or (
        cur_table and cur_age and not cur_civil and prev_civil
    )
    resumed_full_title = (prev_civil_only and cur_table and cur_age and cur_civil) or (
        cur_civil_only and prev_table and prev_age and prev_civil
    )
    civil_fragment_pair = civil_lead and (
        (prev_table and prev_age and cur_civil_only)
        or (cur_table and cur_age and prev_civil_only)
        or title_half
        or resumed_full_title
    )
    return bool(title_half or resumed_full_title or civil_fragment_pair)


def core_split_subject_compatible(prev: dict[str, object], cur: dict[str, object]) -> tuple[bool, str]:
    prev_kind = str(prev.get("_page_kind") or "")
    cur_kind = str(cur.get("_page_kind") or "")
    if prev_kind not in CORE_SPLIT_PAGE_KINDS or cur_kind not in CORE_SPLIT_PAGE_KINDS:
        return False, f"non_split_title_page_kind_{prev_kind}_to_{cur_kind}"
    if "derivative_selected_caste_race_tribe" in {prev_kind, cur_kind}:
        return False, "derivative_selected_caste_race_tribe_boundary"

    prev_family = clean(prev.get("_table_family_key") or "")
    cur_family = clean(cur.get("_table_family_key") or "")
    if prev_family and prev_family not in CORE_SPLIT_TABLE_FAMILIES:
        return False, f"prev_non_core_split_title_family_{prev_family}"
    if cur_family and cur_family not in CORE_SPLIT_TABLE_FAMILIES:
        return False, f"cur_non_core_split_title_family_{cur_family}"
    return True, f"core_split_title_page_kind_{prev_kind}_to_{cur_kind}"


def has_religion_conflict(prev: dict[str, object], cur: dict[str, object]) -> tuple[bool, str]:
    prev_religion = prev.get("_religion_entities") if isinstance(prev.get("_religion_entities"), set) else set()
    cur_religion = cur.get("_religion_entities") if isinstance(cur.get("_religion_entities"), set) else set()
    if prev_religion and cur_religion and prev_religion != cur_religion:
        return True, f"religion_switch_{sorted(prev_religion)}_to_{sorted(cur_religion)}"
    return False, ""


def has_v9_non_core_table_vii_topic(value: object) -> bool:
    text = clean(value)
    return v7.has_non_core_table_vii_topic(value) or any(term in text for term in V9_EXTRA_NON_CORE_TABLE_VII_TERMS)


def is_core_table_vii_group_v9(group: dict[str, object]) -> bool:
    title_text = " ".join([
        str(group.get("primary_title") or ""),
        str(group.get("strict_title_key") or ""),
        str(group.get("scope_without_religion_key") or ""),
    ])
    return v7.is_core_table_vii_group(group) and not has_v9_non_core_table_vii_topic(title_text)


def is_core_table_vii_edge_v9(edge: dict[str, object]) -> bool:
    prev_text = str(edge.get("prev_primary_title") or "")
    cur_text = str(edge.get("primary_title") or "")
    return (
        v7.is_core_table_vii_edge(edge)
        and not has_v9_non_core_table_vii_topic(prev_text)
        and not has_v9_non_core_table_vii_topic(cur_text)
    )


def core_subsets_v9(
    sequence_groups: list[dict[str, object]],
    sequence_members: list[dict[str, object]],
    sequence_edges: list[dict[str, object]],
    bridge_candidates: list[dict[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    group_ids = {str(group.get("sequence_group_id")) for group in sequence_groups if is_core_table_vii_group_v9(group)}
    return (
        [group for group in sequence_groups if str(group.get("sequence_group_id")) in group_ids],
        [member for member in sequence_members if str(member.get("sequence_group_id")) in group_ids],
        [
            edge
            for edge in sequence_edges
            if is_core_table_vii_edge_v9(edge) and edge.get("sequence_edge_action") in {"join_sequence", "bridge_candidate"}
        ],
        [edge for edge in bridge_candidates if is_core_table_vii_edge_v9(edge)],
    )


def split_title_core_decision(
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

    prev_text = title_blob(prev)
    cur_text = title_blob(cur)
    subject_ok, subject_reason = core_split_subject_compatible(prev, cur)
    if not subject_ok:
        return None
    if v7.has_non_core_table_vii_topic(prev_text) or v7.has_non_core_table_vii_topic(cur_text):
        return None
    table_ok, table_reason = table_keys_compatible(prev, cur, prev_text, cur_text)
    if not table_ok:
        return None
    if not split_title_fragment_like(prev_text, cur_text):
        return None

    entity_conflict, entity_conflicts = v7.has_conflicting_entities(prev, cur)
    if entity_conflict:
        return None
    religion_conflict, religion_reason = has_religion_conflict(prev, cur)
    if religion_conflict:
        return None

    strict_sim = base.text_similarity(prev.get("_strict_title_key"), cur.get("_strict_title_key"))
    scope_sim = base.text_similarity(prev.get("_scope_without_religion_key"), cur.get("_scope_without_religion_key"))
    flags = [
        "v9_split_title_classifier_switch_review",
        subject_reason,
        table_reason,
        *v7.missing_entity_flags(prev, cur),
    ]
    if religion_reason:
        flags.append(religion_reason)
    if existing_edge and "no_ocr_structural_profile_for_pair" in str(existing_edge.get("edge_review_flags") or ""):
        flags.append("no_ocr_structural_profile_for_sequence_edge")
    if existing_edge and existing_edge.get("collection_review_flags"):
        flags.append(str(existing_edge.get("collection_review_flags")))

    reason = (
        "adjacent_core_table_vii_split_title_continuation; "
        f"{subject_reason}; {table_reason}; "
        f"strict_title_similarity_{strict_sim:.2f}; scope_similarity_{scope_sim:.2f}"
    )
    missing_pages = ""
    if page_gap > 1:
        missing_pages = f"{prev_last + 1}" if page_gap == 2 else f"{prev_last + 1}-{cur_first - 1}"
        return {
            "page_gap": page_gap,
            "missing_pages": missing_pages,
            "strict_title_similarity": base.float_fmt(strict_sim),
            "scope_similarity": base.float_fmt(scope_sim),
            "sequence_edge_action": "bridge_candidate",
            "sequence_join_type": "split_title_core_table_contiguous_sequence_v9",
            "sequence_edge_reason": f"missing_pages_block_sequence_join_{missing_pages}; {reason}",
            "sequence_review_flags": base.uniq_join(["missing_intermediate_pages", "not_joined_across_gap", *flags], limit=24),
        }
    if page_gap != 1:
        return None
    return {
        "page_gap": page_gap,
        "missing_pages": "",
        "strict_title_similarity": base.float_fmt(strict_sim),
        "scope_similarity": base.float_fmt(scope_sim),
        "sequence_edge_action": "join_sequence",
        "sequence_join_type": "split_title_core_table_contiguous_sequence_v9",
        "sequence_edge_reason": reason,
        "sequence_review_flags": base.uniq_join(flags, limit=24),
    }


def sequence_decision_v9(prev: dict[str, object], cur: dict[str, object], existing_edge: dict[str, str] | None) -> dict[str, object]:
    original = v8.sequence_decision_v8(prev, cur, existing_edge)
    if original.get("sequence_edge_action") != "break":
        return original
    split_title = split_title_core_decision(prev, cur, existing_edge)
    return split_title or original


def add_v9_edge_columns(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for row in rows:
        join_type = str(row.get("sequence_join_type") or "")
        new_row = dict(row)
        if join_type.endswith("_v9"):
            new_row["v9_rule_added"] = "split_title_core_table_continuation"
            new_row["v9_rule_note"] = "reviewed adjacent join for split printed Table VII title"
        else:
            new_row["v9_rule_added"] = ""
            new_row["v9_rule_note"] = ""
        out.append(new_row)
    return out


def group_members_by_sequence(sequence_members: list[dict[str, object]]) -> dict[str, list[dict[str, object]]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for member in sequence_members:
        grouped[str(member.get("sequence_group_id") or "")].append(member)
    return grouped


def large_group_audit(
    sequence_groups: list[dict[str, object]],
    sequence_members: list[dict[str, object]],
    members_by_run: dict[str, list[dict[str, object]]],
) -> list[dict[str, object]]:
    grouped_members = group_members_by_sequence(sequence_members)
    rows: list[dict[str, object]] = []
    for group in sequence_groups:
        flags = str(group.get("sequence_review_flags") or "")
        n_runs = int(group.get("n_physical_runs") or 0)
        is_large = n_runs > 12 or "large_bounded_sequence_review" in flags
        if not is_large:
            continue
        group_id = str(group.get("sequence_group_id") or "")
        members = grouped_members.get(group_id, [])
        join_types = [str(member.get("sequence_join_from_prev_type") or "") for member in members]
        required_ranges: list[str] = []
        for member in members:
            run_id = str(member.get("physical_run_id") or "")
            features = v8.run_structural_features(members_by_run.get(run_id, []))
            if features.get("run_requires_internal_assembly") == "yes":
                required_ranges.append(str(member.get("_page_range") or member.get("page_range") or ""))
        rows.append({
            "large_review_rank": 0,
            "sequence_group_id": group_id,
            "v5_source_key": group.get("v5_source_key", ""),
            "source_pdf": group.get("source_pdf", ""),
            "source_pdf_id": group.get("source_pdf_id", ""),
            "is_core_table_vii": "yes" if is_core_table_vii_group_v9(group) else "no",
            "sequence_basis": group.get("sequence_basis", ""),
            "n_physical_runs": n_runs,
            "n_physical_pages": group.get("n_physical_pages", ""),
            "page_range": group.get("page_range", ""),
            "page_ranges": group.get("page_ranges", ""),
            "all_pages_consecutive": group.get("all_pages_consecutive", ""),
            "sequence_join_types": group.get("sequence_join_types", ""),
            "sequence_review_flags": flags,
            "split_title_join_edges": sum(1 for jt in join_types if jt.endswith("_v9")),
            "v8_relaxed_join_edges": sum(1 for jt in join_types if jt.endswith("_v8")),
            "variant_join_edges": sum(1 for jt in join_types if "religion_variant" in jt),
            "internal_required_physical_runs": len(required_ranges),
            "internal_required_page_ranges": base.uniq_join(required_ranges, limit=30),
            "page_kind_counts": group.get("page_kind_counts", ""),
            "table_family_counts": group.get("table_family_counts", ""),
            "table_number_keys": group.get("table_number_keys", ""),
            "religion_entities": group.get("religion_entities", ""),
            "part_entities": group.get("part_entities", ""),
            "sex_entities": group.get("sex_entities", ""),
            "parenthetical_numbers": group.get("parenthetical_numbers", ""),
            "primary_title": group.get("primary_title", ""),
            "sequence_reasons": group.get("sequence_reasons", ""),
            "example_images": group.get("example_images", ""),
        })
    rows.sort(key=lambda r: (
        0 if r.get("is_core_table_vii") == "yes" else 1,
        -int(r.get("n_physical_runs") or 0),
        str(r.get("v5_source_key") or ""),
        str(r.get("page_range") or ""),
    ))
    for rank, row in enumerate(rows, start=1):
        row["large_review_rank"] = rank
    return rows


def write_rules_readme(path: Path, summary: dict[str, object]) -> None:
    lines = [
        "# v9 Split-Title Assembly Rules",
        "",
        "This output is assembly logic only. It performs no OCR/API calls.",
        "",
        "## Input Layer",
        "",
        "- Reads v6 strict physical runs through v7/v8 helpers.",
        "- Physical runs remain the only required row-label/half-spread assembly layer.",
        "- Logical sequence groups can only join adjacent physical runs in the same source.",
        "",
        "## Added v9 Rule",
        "",
        "- Adds `split_title_core_table_contiguous_sequence_v9` for adjacent Table VII title fragments.",
        "- Typical pattern: `TABLE VII - AGE, SEX` followed by `AND CIVIL CONDITION`.",
        "- The rule excludes context/blank, off-target, unknown, and derivative selected-caste/race/tribe pages.",
        "- Explicit non-VII table numbers, part/sex/parenthetical conflicts, religion conflicts, and missing pages prevent a direct join.",
        "- Every v9 join carries `v9_split_title_classifier_switch_review`.",
        "",
        "## Review Outputs",
        "",
        "- `v9_sequence_edges_detailed.csv`: every boundary with join/break/bridge reason and row-label/layout context.",
        "- `v9_candidate_related_edges.csv`: adjacent related-looking edges deliberately not grouped.",
        "- `v9_large_group_audit.csv`: large consecutive groups to inspect for over-broad logical grouping.",
        "",
        "## Summary",
        "",
    ]
    for key in sorted(summary):
        lines.append(f"- `{key}`: {summary[key]}")
    (path / "v9_rules_readme.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


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

    original_sequence_decision = v7.sequence_decision
    v7.sequence_decision = sequence_decision_v9
    try:
        sequence_groups, sequence_members, sequence_edges, bridge_candidates = v7.build_sequences(runs, edge_by_run_pair)
    finally:
        v7.sequence_decision = original_sequence_decision

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
            "old_collection_split_by_adjacency_v9",
        )

    detailed_edges_raw, candidate_edges_raw = v8.detailed_edge_rows(sequence_edges, members_by_run)
    detailed_edges = add_v9_edge_columns(detailed_edges_raw)
    candidate_edges = add_v9_edge_columns(candidate_edges_raw)
    large_groups = large_group_audit(sequence_groups, sequence_members, members_by_run)
    core_groups, core_members, core_edges, core_bridges = core_subsets_v9(
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
        "v9_split_title_join_sequence_edges": sum(
            1
            for e in sequence_edges
            if e.get("sequence_edge_action") == "join_sequence" and str(e.get("sequence_join_type") or "").endswith("_v9")
        ),
        "v9_split_title_bridge_candidate_edges": sum(
            1
            for e in sequence_edges
            if e.get("sequence_edge_action") == "bridge_candidate" and str(e.get("sequence_join_type") or "").endswith("_v9")
        ),
        "bridge_candidate_edges": len(bridge_candidates),
        "break_sequence_edges": sum(1 for e in sequence_edges if e.get("sequence_edge_action") == "break"),
        "candidate_related_not_grouped_edges": len(candidate_edges),
        "sequence_review_queue_rows": len(sequence_review_queue),
        "large_group_audit_rows": len(large_groups),
        "large_core_table_vii_group_audit_rows": sum(1 for g in large_groups if g.get("is_core_table_vii") == "yes"),
        "religion_variant_sequence_groups": sum(1 for g in sequence_groups if g.get("sequence_basis") == "bounded_religion_variant_subtable_sequence"),
        "large_sequence_review_groups": sum(1 for g in sequence_groups if "large_bounded_sequence_review" in str(g.get("sequence_review_flags") or "")),
        "old_v6_multi_run_collection_groups": sum(1 for g in old_collections if int(g.get("n_physical_runs") or 0) > 1),
        "old_v6_max_collection_runs": max((int(g.get("n_physical_runs") or 0) for g in old_collections), default=0),
        "old_v6_overassociation_audit_rows": len(old_overassociation_rows),
        "old_v6_gap_crossing_audit_rows": sum(1 for g in old_overassociation_rows if int(g.get("old_missing_gap_count") or 0) > 0),
        "old_v6_split_by_v9_audit_rows": sum(1 for g in old_overassociation_rows if "old_collection_split_by_adjacency_v9" in str(g.get("audit_flags") or "")),
        "core_table_vii_sequence_groups": len(core_groups),
        "core_table_vii_multi_run_sequence_groups": sum(1 for g in core_groups if int(g.get("n_physical_runs") or 0) > 1),
        "core_table_vii_sequence_members": len(core_members),
        "core_table_vii_sequence_edges": len(core_edges),
        "core_table_vii_bridge_candidates": len(core_bridges),
    }
    summary.update(v7.sequence_invariants(sequence_groups, sequence_edges))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base.write_csv(OUT_DIR / "v9_sequence_groups.csv", sequence_groups, v7.SEQUENCE_GROUP_FIELDS)
    base.write_csv(OUT_DIR / "v9_sequence_members.csv", sequence_members, v7.SEQUENCE_MEMBER_FIELDS)
    base.write_csv(OUT_DIR / "v9_sequence_edges.csv", sequence_edges, v7.SEQUENCE_EDGE_FIELDS)
    base.write_csv(OUT_DIR / "v9_sequence_edges_detailed.csv", detailed_edges, V9_DETAILED_EDGE_FIELDS)
    base.write_csv(OUT_DIR / "v9_candidate_related_edges.csv", candidate_edges, V9_CANDIDATE_RELATED_FIELDS)
    base.write_csv(OUT_DIR / "v9_missing_bridge_candidates.csv", bridge_candidates, v7.SEQUENCE_EDGE_FIELDS)
    base.write_csv(OUT_DIR / "v9_sequence_review_queue.csv", sequence_review_queue, v7.SEQUENCE_REVIEW_FIELDS)
    base.write_csv(OUT_DIR / "v9_large_group_audit.csv", large_groups, V9_LARGE_GROUP_FIELDS)
    base.write_csv(OUT_DIR / "v9_collection_comparison.csv", comparison_rows, ["metric", "old_value", "new_value", "note"])
    base.write_csv(OUT_DIR / "v9_old_overassociation_audit.csv", old_overassociation_rows, v7.OLD_OVERASSOCIATION_FIELDS)
    base.write_csv(OUT_DIR / "v9_required_physical_edges_passthrough.csv", required_physical_edges, base.EDGE_FIELDS)
    base.write_csv(OUT_DIR / "v9_core_tableVII_sequence_groups.csv", core_groups, v7.SEQUENCE_GROUP_FIELDS)
    base.write_csv(OUT_DIR / "v9_core_tableVII_sequence_members.csv", core_members, v7.SEQUENCE_MEMBER_FIELDS)
    base.write_csv(OUT_DIR / "v9_core_tableVII_sequence_edges.csv", core_edges, v7.SEQUENCE_EDGE_FIELDS)
    base.write_csv(OUT_DIR / "v9_core_tableVII_bridge_candidates.csv", core_bridges, v7.SEQUENCE_EDGE_FIELDS)

    with (OUT_DIR / "v9_sequence_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    write_rules_readme(OUT_DIR, summary)

    summary_rows = [{"metric": key, "value": json.dumps(value) if isinstance(value, (dict, list)) else value} for key, value in summary.items()]
    base.write_workbook(
        OUT_DIR / "v9_split_title_sequence_review.xlsx",
        {
            "summary": (summary_rows, ["metric", "value"]),
            "sequence_groups": (sequence_groups, v7.SEQUENCE_GROUP_FIELDS),
            "sequence_members": (sequence_members, v7.SEQUENCE_MEMBER_FIELDS),
            "sequence_edges": (sequence_edges, v7.SEQUENCE_EDGE_FIELDS),
            "detailed_edges": (detailed_edges, V9_DETAILED_EDGE_FIELDS),
            "candidate_related": (candidate_edges, V9_CANDIDATE_RELATED_FIELDS),
            "bridge_candidates": (bridge_candidates, v7.SEQUENCE_EDGE_FIELDS),
            "sequence_review_queue": (sequence_review_queue, v7.SEQUENCE_REVIEW_FIELDS),
            "large_group_audit": (large_groups, V9_LARGE_GROUP_FIELDS),
            "old_overassoc_audit": (old_overassociation_rows, v7.OLD_OVERASSOCIATION_FIELDS),
            "required_physical": (required_physical_edges, base.EDGE_FIELDS),
            "core_vii_groups": (core_groups, v7.SEQUENCE_GROUP_FIELDS),
            "core_vii_members": (core_members, v7.SEQUENCE_MEMBER_FIELDS),
            "core_vii_edges": (core_edges, v7.SEQUENCE_EDGE_FIELDS),
            "core_vii_bridges": (core_bridges, v7.SEQUENCE_EDGE_FIELDS),
            "v6_vs_v9": (comparison_rows, ["metric", "old_value", "new_value", "note"]),
        },
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
