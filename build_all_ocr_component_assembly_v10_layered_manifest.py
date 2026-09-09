#!/usr/bin/env python3
"""Build layered assembly manifests for scalable table grouping.

This pass performs no OCR/API work and does not change the v9 grouping
decisions. It makes the assembly semantics explicit:

* required physical assemblies: pages that must be combined to read row labels;
* bounded body sequences: adjacent runs that belong to the same table body;
* bounded variant sets: adjacent same-scope subtables, such as religion variants;
* bridge candidates: related pages separated by missing PDF pages, never joined;
* possible missed required assemblies: adjacent breaks where the next page has
  deficient/no row labels but the previous page has row context.

The output is meant to be stable infrastructure for static-site ingestion and
future OCR batches.
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import build_all_ocr_component_assembly_v5_structural as base
import build_all_ocr_component_assembly_v7_sequences as v7
import build_all_ocr_component_assembly_v8_tiered as v8
import build_all_ocr_component_assembly_v9_split_title as v9


ROOT = Path(__file__).resolve().parent
V6_DIR = ROOT / "export_final/pdf_recovery_incoming_2026-06-16_all_ocr_component_assembly_v6_layout_strict"
V9_DIR = ROOT / "export_final/pdf_recovery_incoming_2026-06-16_all_ocr_component_assembly_v9_split_title"
OUT_DIR = ROOT / "export_final/pdf_recovery_incoming_2026-06-16_all_ocr_component_assembly_v10_layered_manifest"

V6_RUN_MEMBERS = V6_DIR / "v6_run_members.csv"
V6_REQUIRED_EDGES = V6_DIR / "v6_required_assembly_edges.csv"
V6_PHYSICAL_EDGES = V6_DIR / "v6_physical_edges.csv"
V6_LAYOUT_REJECTED_EDGES = V6_DIR / "v6_layout_rejected_edges.csv"

V9_GROUPS = V9_DIR / "v9_sequence_groups.csv"
V9_MEMBERS = V9_DIR / "v9_sequence_members.csv"
V9_DETAILED_EDGES = V9_DIR / "v9_sequence_edges_detailed.csv"
V9_BRIDGES = V9_DIR / "v9_missing_bridge_candidates.csv"
V9_CANDIDATES = V9_DIR / "v9_candidate_related_edges.csv"
V9_OVERASSOC = V9_DIR / "v9_old_overassociation_audit.csv"

V10_GROUP_FIELDS = [
    *v7.SEQUENCE_GROUP_FIELDS,
    "v10_group_layer",
    "v10_static_site_action",
    "v10_group_confidence",
    "v10_group_review_reasons",
    "v10_body_member_count",
    "v10_probable_body_no_profile_count",
    "v10_required_physical_run_count",
    "v10_front_matter_or_context_count",
    "v10_unknown_review_member_count",
    "v10_split_title_edge_count",
    "v10_variant_edge_count",
    "v10_v8_relaxed_edge_count",
    "v10_internal_required_page_ranges",
    "v10_member_roles",
]

V10_MEMBER_FIELDS = [
    *v7.SEQUENCE_MEMBER_FIELDS,
    "v10_member_role",
    "v10_member_role_reason",
    "v10_profile_loaded_pages",
    "v10_profile_row_count_total",
    "v10_profile_numeric_pages",
    "v10_row_label_min_ratio",
    "v10_no_row_label_pages",
    "v10_internal_required_assembly",
    "v10_internal_required_reasons",
]

V10_EDGE_FIELDS = [
    *v9.V9_DETAILED_EDGE_FIELDS,
    "v10_edge_layer",
    "v10_static_site_edge_action",
    "v10_edge_confidence",
    "v10_edge_review_reason",
]

V10_REQUIRED_EDGE_FIELDS = [
    *base.EDGE_FIELDS,
    "v10_required_layer",
    "v10_required_reason",
    "v10_required_confidence",
]

V10_REQUIRED_CANDIDATE_FIELDS = [
    "candidate_priority",
    "candidate_type",
    "candidate_reason",
    *base.EDGE_FIELDS,
]

V10_VARIANT_SET_FIELDS = [
    "variant_set_id",
    "sequence_group_id",
    "v5_source_key",
    "source_pdf",
    "page_range",
    "n_physical_runs",
    "n_physical_pages",
    "religion_entities",
    "part_entities",
    "sex_entities",
    "table_number_keys",
    "primary_title",
    "sequence_join_types",
    "sequence_review_flags",
    "example_images",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def as_float(value: object) -> float | None:
    try:
        return float(str(value or "").strip())
    except ValueError:
        return None


def as_int(value: object) -> int:
    return base.int_or_none(value) or 0


def parse_counter(value: object) -> Counter:
    try:
        raw = json.loads(str(value or "{}"))
    except json.JSONDecodeError:
        return Counter()
    if not isinstance(raw, dict):
        return Counter()
    return Counter({str(k): int(v) for k, v in raw.items() if str(k)})


def run_features(run_members: list[dict[str, str]]) -> dict[str, object]:
    structural = v8.run_structural_features(run_members)
    profile_loaded = [m for m in run_members if str(m.get("profile_loaded") or "") == "True"]
    row_counts = [as_int(m.get("profile_row_count")) for m in run_members]
    numeric_pages = [
        str(m.get("pdf_page") or "")
        for m in run_members
        if str(m.get("profile_has_numeric_cells") or "").lower() in {"yes", "true", "1"}
    ]
    no_label_pages = structural.get("no_row_label_pages", "")
    internal_reasons = structural.get("internal_join_reasons", "")
    return {
        **structural,
        "profile_loaded_pages": len(profile_loaded),
        "profile_row_count_total": sum(row_counts),
        "profile_numeric_pages": base.uniq_join(numeric_pages, limit=20),
        "internal_required_assembly": "yes" if structural.get("run_requires_internal_assembly") == "yes" else "no",
        "internal_required_reasons": internal_reasons,
    }


def tableish_page_kind(page_kind_counts: Counter) -> bool:
    table_kinds = {"main_age_civil_numeric", "other_table", "structured_unknown", "derivative_selected_caste_race_tribe"}
    return bool(table_kinds & set(page_kind_counts))


def front_matterish_title(title: object) -> bool:
    text = base.clean_text(title)
    terms = [
        "synopsis of report",
        "contents",
        "index",
        "introduction",
        "preface",
        "appendix",
        "explanatory note",
        "list of",
    ]
    return any(term in text for term in terms)


def member_role(member: dict[str, object], features: dict[str, object]) -> tuple[str, str]:
    page_kinds = parse_counter(member.get("page_kind_counts"))
    n_pages = as_int(member.get("n_physical_pages"))
    profile_rows = as_int(features.get("profile_row_count_total"))
    has_numeric_profile = bool(str(features.get("profile_numeric_pages") or "").strip())
    has_required = features.get("internal_required_assembly") == "yes" or n_pages > 1
    title = member.get("primary_title", "")

    if has_required:
        return "required_physical_row_label_fragment", "physical_run_has_internal_required_page_join"
    if profile_rows > 0 or has_numeric_profile:
        return "numeric_body_page", "ocr_profile_has_rows_or_numeric_cells"
    if page_kinds.get("main_age_civil_numeric", 0) or page_kinds.get("other_table", 0):
        return "probable_numeric_body_no_profile", "classifier_marks_table_but_no_structural_profile"
    if page_kinds.get("structured_unknown", 0) and not front_matterish_title(title):
        return "logical_table_page_review", "structured_unknown_table_like_page"
    if front_matterish_title(title) or any(k in page_kinds for k in ["context_or_blank", "offtarget_table", "unknown"]):
        return "front_matter_or_context_review", "context_offtarget_unknown_or_front_matter_title"
    if tableish_page_kind(page_kinds):
        return "logical_table_page_review", "table_like_classifier_without_profile"
    return "unclassified_review", "insufficient_member_evidence"


def enrich_members(sequence_members: list[dict[str, str]], members_by_run: dict[str, list[dict[str, str]]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for member in sequence_members:
        run_id = str(member.get("physical_run_id") or "")
        features = run_features(members_by_run.get(run_id, []))
        role, reason = member_role(member, features)
        out.append({
            **member,
            "v10_member_role": role,
            "v10_member_role_reason": reason,
            "v10_profile_loaded_pages": features.get("profile_loaded_pages", ""),
            "v10_profile_row_count_total": features.get("profile_row_count_total", ""),
            "v10_profile_numeric_pages": features.get("profile_numeric_pages", ""),
            "v10_row_label_min_ratio": features.get("row_label_min_ratio", ""),
            "v10_no_row_label_pages": features.get("no_row_label_pages", ""),
            "v10_internal_required_assembly": features.get("internal_required_assembly", ""),
            "v10_internal_required_reasons": features.get("internal_required_reasons", ""),
        })
    return out


def group_edge_counts(sequence_members: list[dict[str, object]]) -> dict[str, Counter]:
    counts: dict[str, Counter] = defaultdict(Counter)
    for member in sequence_members:
        group_id = str(member.get("sequence_group_id") or "")
        join_type = str(member.get("sequence_join_from_prev_type") or "")
        if join_type:
            counts[group_id][join_type] += 1
    return counts


def classify_group(group: dict[str, str], members: list[dict[str, object]], edge_counts: Counter) -> tuple[str, str, str, str]:
    n_runs = as_int(group.get("n_physical_runs"))
    roles = Counter(str(m.get("v10_member_role") or "") for m in members)
    review_reasons: list[str] = []
    flags = str(group.get("sequence_review_flags") or "")
    basis = str(group.get("sequence_basis") or "")

    if roles.get("front_matter_or_context_review"):
        review_reasons.append("contains_front_matter_or_context_member")
    if roles.get("unclassified_review"):
        review_reasons.append("contains_unclassified_member")
    if "large_bounded_sequence_review" in flags or n_runs > 12:
        review_reasons.append("large_consecutive_sequence")
    if "possible_unrelated_subtable" in flags:
        review_reasons.append("possible_unrelated_subtable")
    if "missing_intermediate_pages" in flags:
        # Existing group can still be consecutive; this flag usually comes from
        # inherited page-level quality flags. Keep it visible but not fatal.
        review_reasons.append("has_missing_page_quality_flag")

    if basis == "bounded_religion_variant_subtable_sequence":
        layer = "bounded_variant_set"
    elif n_runs > 1:
        layer = "bounded_body_sequence"
    elif roles.get("required_physical_row_label_fragment"):
        layer = "required_physical_only"
    else:
        layer = "single_physical_run"

    if review_reasons:
        action = "review_before_static_merge" if n_runs > 1 else "show_single_with_review_badge"
        confidence = "review"
    elif layer in {"bounded_body_sequence", "bounded_variant_set"}:
        action = "static_group_candidate"
        confidence = "medium"
    elif layer == "required_physical_only":
        action = "assemble_required_physical"
        confidence = "high"
    else:
        action = "show_single"
        confidence = "high"
    return layer, action, confidence, base.uniq_join(review_reasons, limit=20)


def enrich_groups(groups: list[dict[str, str]], enriched_members: list[dict[str, object]]) -> list[dict[str, object]]:
    members_by_group: dict[str, list[dict[str, object]]] = defaultdict(list)
    for member in enriched_members:
        members_by_group[str(member.get("sequence_group_id") or "")].append(member)
    edge_counts = group_edge_counts(enriched_members)

    out: list[dict[str, object]] = []
    for group in groups:
        group_id = str(group.get("sequence_group_id") or "")
        members = members_by_group.get(group_id, [])
        roles = Counter(str(m.get("v10_member_role") or "") for m in members)
        counts = edge_counts.get(group_id, Counter())
        internal_required_ranges = [
            str(m.get("page_range") or "")
            for m in members
            if str(m.get("v10_internal_required_assembly") or "") == "yes"
        ]
        layer, action, confidence, review_reasons = classify_group(group, members, counts)
        out.append({
            **group,
            "v10_group_layer": layer,
            "v10_static_site_action": action,
            "v10_group_confidence": confidence,
            "v10_group_review_reasons": review_reasons,
            "v10_body_member_count": roles.get("numeric_body_page", 0),
            "v10_probable_body_no_profile_count": roles.get("probable_numeric_body_no_profile", 0),
            "v10_required_physical_run_count": roles.get("required_physical_row_label_fragment", 0),
            "v10_front_matter_or_context_count": roles.get("front_matter_or_context_review", 0),
            "v10_unknown_review_member_count": roles.get("unclassified_review", 0) + roles.get("logical_table_page_review", 0),
            "v10_split_title_edge_count": counts.get("split_title_core_table_contiguous_sequence_v9", 0),
            "v10_variant_edge_count": counts.get("religion_variant_subtable_sequence", 0),
            "v10_v8_relaxed_edge_count": sum(v for k, v in counts.items() if k.endswith("_v8")),
            "v10_internal_required_page_ranges": base.uniq_join(internal_required_ranges, limit=30),
            "v10_member_roles": json.dumps(dict(roles), sort_keys=True),
        })
    return out


def classify_edge(edge: dict[str, str]) -> tuple[str, str, str, str]:
    action = str(edge.get("sequence_edge_action") or "")
    join_type = str(edge.get("sequence_join_type") or "")
    flags = str(edge.get("sequence_review_flags") or "")
    candidate = str(edge.get("v8_candidate_related") or "") == "yes"
    if action == "join_sequence" and join_type == "religion_variant_subtable_sequence":
        return "logical_variant_set", "join", "review" if flags else "medium", flags
    if action == "join_sequence" and join_type.endswith("_v9"):
        return "logical_split_title_sequence", "join", "review", flags
    if action == "join_sequence":
        return "logical_body_sequence", "join", "review" if flags else "high", flags
    if action == "bridge_candidate":
        return "missing_bridge_candidate", "do_not_join_until_missing_pages_present", "review", flags
    if candidate:
        return "candidate_related_not_grouped", "do_not_join_review_candidate", "review", str(edge.get("v8_candidate_reason") or flags)
    return "hard_break", "break", "", flags


def enrich_edges(edges: list[dict[str, str]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for edge in edges:
        layer, action, confidence, reason = classify_edge(edge)
        out.append({
            **edge,
            "v10_edge_layer": layer,
            "v10_static_site_edge_action": action,
            "v10_edge_confidence": confidence,
            "v10_edge_review_reason": reason,
        })
    return out


def enrich_required_edges(edges: list[dict[str, str]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for edge in edges:
        reason = str(edge.get("edge_reason") or "")
        flags = str(edge.get("edge_review_flags") or "")
        if "image_layout_required_join" in flags:
            required_reason = "image_layout_halfspread_requires_previous_row_labels"
            confidence = "high_review"
        elif "current_page_row_labels_deficient" in flags:
            required_reason = "ocr_profile_row_labels_deficient_requires_previous_row_labels"
            confidence = "high"
        else:
            required_reason = reason.split(";", 1)[0]
            confidence = "review"
        out.append({
            **edge,
            "v10_required_layer": "required_physical_row_label_assembly",
            "v10_required_reason": required_reason,
            "v10_required_confidence": confidence,
        })
    return out


def row_label_missing(edge: dict[str, str]) -> bool:
    ratio = as_float(edge.get("row_label_ratio"))
    cur_has_ratio = str(edge.get("row_label_ratio") or "").strip() != ""
    cur_first = str(edge.get("first_row_labels") or "").strip()
    return (ratio is not None and ratio < 0.35) or (cur_has_ratio and not cur_first)


def prev_has_row_context(edge: dict[str, str]) -> bool:
    ratio = as_float(edge.get("prev_row_label_ratio"))
    labels = str(edge.get("prev_first_row_labels") or "") + str(edge.get("prev_last_row_labels") or "")
    return bool(labels.strip()) or (ratio is not None and ratio >= 0.55)


def required_edge_candidates(physical_edges: list[dict[str, str]], layout_rejected_edges: list[dict[str, str]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    seen: set[tuple[str, str, str]] = set()

    def add(priority: int, candidate_type: str, reason: str, edge: dict[str, str]) -> None:
        key = (str(edge.get("v5_source_key") or ""), str(edge.get("prev_pdf_page") or ""), str(edge.get("pdf_page") or ""))
        if key in seen:
            return
        seen.add(key)
        rows.append({
            "candidate_priority": priority,
            "candidate_type": candidate_type,
            "candidate_reason": reason,
            **edge,
        })

    for edge in physical_edges:
        if edge.get("edge_action") != "break" or str(edge.get("page_gap") or "") != "1":
            continue
        if row_label_missing(edge) and prev_has_row_context(edge):
            add(
                1,
                "possible_required_row_label_assembly",
                "current_page_has_deficient_or_missing_row_labels_and_previous_page_has_row_context",
                edge,
            )
    for edge in layout_rejected_edges:
        add(
            2,
            "layout_halfspread_rejected_review",
            str(edge.get("edge_review_flags") or edge.get("edge_reason") or "layout_rejected_candidate"),
            edge,
        )

    rows.sort(key=lambda r: (
        int(r.get("candidate_priority") or 99),
        str(r.get("v5_source_key") or ""),
        base.int_or_none(r.get("prev_pdf_page")) if base.int_or_none(r.get("prev_pdf_page")) is not None else 10**9,
        base.int_or_none(r.get("pdf_page")) if base.int_or_none(r.get("pdf_page")) is not None else 10**9,
    ))
    return rows


def variant_sets(groups: list[dict[str, object]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for group in groups:
        if group.get("v10_group_layer") != "bounded_variant_set":
            continue
        out.append({
            "variant_set_id": base.stable_id("variant", group.get("sequence_group_id")),
            "sequence_group_id": group.get("sequence_group_id", ""),
            "v5_source_key": group.get("v5_source_key", ""),
            "source_pdf": group.get("source_pdf", ""),
            "page_range": group.get("page_range", ""),
            "n_physical_runs": group.get("n_physical_runs", ""),
            "n_physical_pages": group.get("n_physical_pages", ""),
            "religion_entities": group.get("religion_entities", ""),
            "part_entities": group.get("part_entities", ""),
            "sex_entities": group.get("sex_entities", ""),
            "table_number_keys": group.get("table_number_keys", ""),
            "primary_title": group.get("primary_title", ""),
            "sequence_join_types": group.get("sequence_join_types", ""),
            "sequence_review_flags": group.get("sequence_review_flags", ""),
            "example_images": group.get("example_images", ""),
        })
    return out


def validate_outputs(groups: list[dict[str, object]], edges: list[dict[str, object]], required_edges: list[dict[str, object]]) -> dict[str, object]:
    failures: list[str] = []
    for group in groups:
        if as_int(group.get("n_physical_runs")) > 1 and str(group.get("all_pages_consecutive")) != "True":
            failures.append(f"group_internal_gap:{group.get('sequence_group_id')}:{group.get('page_ranges')}")
    for edge in edges:
        if edge.get("sequence_edge_action") == "join_sequence" and str(edge.get("page_gap")) != "1":
            failures.append(f"logical_join_non_adjacent:{edge.get('sequence_edge_id')}:{edge.get('page_gap')}")
        if edge.get("sequence_edge_action") == "bridge_candidate" and str(edge.get("v10_static_site_edge_action")) == "join":
            failures.append(f"bridge_marked_join:{edge.get('sequence_edge_id')}")
    for edge in required_edges:
        if str(edge.get("page_gap")) != "1":
            failures.append(f"required_physical_non_adjacent:{edge.get('v5_source_key')}:{edge.get('prev_pdf_page')}->{edge.get('pdf_page')}")
    return {
        "v10_invariant_failure_count": len(failures),
        "v10_invariant_failures": failures[:100],
    }


def write_rules_readme(path: Path, summary: dict[str, object]) -> None:
    lines = [
        "# v10 Layered Assembly Manifest",
        "",
        "No OCR/API calls are made. This pass reads v6 physical assembly outputs and v9 logical sequence outputs.",
        "",
        "## Layers",
        "",
        "- `required_physical_row_label_assembly`: adjacent pages that must be combined to read a table, usually a right half-page lacking row labels.",
        "- `bounded_body_sequence`: adjacent physical runs that form the same published table body.",
        "- `bounded_variant_set`: adjacent same-scope subtables, e.g. Muslim/Christian/Jew pages for the same table scope.",
        "- `missing_bridge_candidate`: related pages separated by missing PDF pages; never joined automatically.",
        "- `possible_required_row_label_assembly`: review-only candidates where a break may still need row labels from the previous page.",
        "",
        "## Static Site Guidance",
        "",
        "- Use `v10_static_site_action` on groups and `v10_static_site_edge_action` on edges.",
        "- `review_before_static_merge` means the group is consecutive but too uncertain or broad to merge silently.",
        "- Missing bridge candidates should be shown as gaps, not assembled components.",
        "",
        "## Summary",
        "",
    ]
    for key in sorted(summary):
        lines.append(f"- `{key}`: {summary[key]}")
    (path / "v10_rules_readme.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    csv.field_size_limit(10 ** 7)
    groups = read_csv(V9_GROUPS)
    sequence_members = read_csv(V9_MEMBERS)
    detailed_edges = read_csv(V9_DETAILED_EDGES)
    bridges = read_csv(V9_BRIDGES)
    candidates = read_csv(V9_CANDIDATES)
    overassoc = read_csv(V9_OVERASSOC)
    v6_run_members = read_csv(V6_RUN_MEMBERS)
    required_edges_raw = read_csv(V6_REQUIRED_EDGES)
    physical_edges = read_csv(V6_PHYSICAL_EDGES)
    layout_rejected = read_csv(V6_LAYOUT_REJECTED_EDGES)
    if not groups or not sequence_members or not detailed_edges or not v6_run_members:
        raise SystemExit("Missing v9 or v6 inputs; run v9 and v6 strict assembly first")

    members_by_run: dict[str, list[dict[str, str]]] = defaultdict(list)
    for member in v6_run_members:
        members_by_run[str(member.get("physical_run_id") or "")].append(member)

    enriched_members = enrich_members(sequence_members, members_by_run)
    enriched_groups = enrich_groups(groups, enriched_members)
    enriched_edges = enrich_edges(detailed_edges)
    required_edges = enrich_required_edges(required_edges_raw)
    required_candidates = required_edge_candidates(physical_edges, layout_rejected)
    variants = variant_sets(enriched_groups)

    summary = {
        "input_v9_groups": len(groups),
        "input_v9_members": len(sequence_members),
        "input_v9_detailed_edges": len(detailed_edges),
        "v10_groups": len(enriched_groups),
        "v10_members": len(enriched_members),
        "v10_edges": len(enriched_edges),
        "required_physical_edges": len(required_edges),
        "possible_required_edge_review_candidates": len(required_candidates),
        "missing_bridge_candidates": len(bridges),
        "candidate_related_not_grouped_edges": len(candidates),
        "old_overassociation_audit_rows": len(overassoc),
        "variant_sets": len(variants),
        "static_group_candidates": sum(1 for g in enriched_groups if g.get("v10_static_site_action") == "static_group_candidate"),
        "review_before_static_merge_groups": sum(1 for g in enriched_groups if g.get("v10_static_site_action") == "review_before_static_merge"),
        "required_physical_only_groups": sum(1 for g in enriched_groups if g.get("v10_group_layer") == "required_physical_only"),
        "bounded_body_sequence_groups": sum(1 for g in enriched_groups if g.get("v10_group_layer") == "bounded_body_sequence"),
        "bounded_variant_set_groups": sum(1 for g in enriched_groups if g.get("v10_group_layer") == "bounded_variant_set"),
        "groups_with_required_physical_runs": sum(1 for g in enriched_groups if as_int(g.get("v10_required_physical_run_count")) > 0),
        "groups_with_front_matter_or_context": sum(1 for g in enriched_groups if as_int(g.get("v10_front_matter_or_context_count")) > 0),
        "logical_join_edges": sum(1 for e in enriched_edges if e.get("v10_static_site_edge_action") == "join"),
        "bridge_edges_not_joined": sum(1 for e in enriched_edges if e.get("v10_edge_layer") == "missing_bridge_candidate"),
    }
    summary.update(validate_outputs(enriched_groups, enriched_edges, required_edges))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base.write_csv(OUT_DIR / "v10_assembly_groups.csv", enriched_groups, V10_GROUP_FIELDS)
    base.write_csv(OUT_DIR / "v10_assembly_members.csv", enriched_members, V10_MEMBER_FIELDS)
    base.write_csv(OUT_DIR / "v10_assembly_edges.csv", enriched_edges, V10_EDGE_FIELDS)
    base.write_csv(OUT_DIR / "v10_required_physical_edges.csv", required_edges, V10_REQUIRED_EDGE_FIELDS)
    base.write_csv(OUT_DIR / "v10_possible_required_physical_edges_review.csv", required_candidates, V10_REQUIRED_CANDIDATE_FIELDS)
    base.write_csv(OUT_DIR / "v10_missing_bridge_candidates.csv", bridges, v7.SEQUENCE_EDGE_FIELDS)
    base.write_csv(OUT_DIR / "v10_candidate_related_edges.csv", candidates, v9.V9_CANDIDATE_RELATED_FIELDS)
    base.write_csv(OUT_DIR / "v10_variant_sets.csv", variants, V10_VARIANT_SET_FIELDS)
    base.write_csv(OUT_DIR / "v10_old_overassociation_audit.csv", overassoc, v7.OLD_OVERASSOCIATION_FIELDS)

    with (OUT_DIR / "v10_layered_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    write_rules_readme(OUT_DIR, summary)

    summary_rows = [{"metric": key, "value": json.dumps(value) if isinstance(value, (dict, list)) else value} for key, value in summary.items()]
    base.write_workbook(
        OUT_DIR / "v10_layered_assembly_review.xlsx",
        {
            "summary": (summary_rows, ["metric", "value"]),
            "groups": (enriched_groups, V10_GROUP_FIELDS),
            "members": (enriched_members, V10_MEMBER_FIELDS),
            "edges": (enriched_edges, V10_EDGE_FIELDS),
            "required_physical": (required_edges, V10_REQUIRED_EDGE_FIELDS),
            "possible_required_review": (required_candidates, V10_REQUIRED_CANDIDATE_FIELDS),
            "bridges": (bridges, v7.SEQUENCE_EDGE_FIELDS),
            "candidate_related": (candidates, v9.V9_CANDIDATE_RELATED_FIELDS),
            "variant_sets": (variants, V10_VARIANT_SET_FIELDS),
            "old_overassoc": (overassoc, v7.OLD_OVERASSOCIATION_FIELDS),
        },
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
