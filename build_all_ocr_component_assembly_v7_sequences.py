#!/usr/bin/env python3
"""Build bounded logical table sequences on top of strict physical assemblies.

This pass does not OCR or inspect images. It consumes the v6 strict adjacency
output and adds a second grouping layer:

* physical runs remain the only required page assemblies;
* logical table sequences can span adjacent physical runs only;
* missing-page gaps become bridge candidates, never merged groups;
* religion-variant subtables can be grouped when the surrounding table scope is
  the same, while part/sex/location-number switches break the sequence.
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import build_all_ocr_component_assembly_v5_structural as base
import build_all_ocr_component_assembly_v6_layout as layout


ROOT = Path(__file__).resolve().parent
V6_DIR = ROOT / "export_final/pdf_recovery_incoming_2026-06-16_all_ocr_component_assembly_v6_layout_strict"
OUT_DIR = ROOT / "export_final/pdf_recovery_incoming_2026-06-16_all_ocr_component_assembly_v7_sequences"

PHYSICAL_RUNS = V6_DIR / "v6_physical_runs.csv"
RUN_MEMBERS = V6_DIR / "v6_run_members.csv"
PHYSICAL_EDGES = V6_DIR / "v6_physical_edges.csv"
REQUIRED_PHYSICAL_EDGES = V6_DIR / "v6_required_assembly_edges.csv"
OLD_COLLECTION_GROUPS = V6_DIR / "v6_collection_groups.csv"
OLD_COLLECTION_MEMBERS = V6_DIR / "v6_collection_members.csv"


SEQUENCE_GROUP_FIELDS = [
    "sequence_group_id",
    "sequence_basis",
    "review_flag",
    "sequence_review_flags",
    "v5_source_key",
    "source_pdf",
    "source_pdf_id",
    "n_physical_runs",
    "n_physical_pages",
    "page_range",
    "page_ranges",
    "all_pages_consecutive",
    "sequence_join_types",
    "page_kind_counts",
    "table_family_counts",
    "table_number_keys",
    "religion_entities",
    "part_entities",
    "sex_entities",
    "parenthetical_numbers",
    "primary_title",
    "strict_title_key",
    "scope_without_religion_key",
    "sequence_reasons",
    "example_images",
]

SEQUENCE_MEMBER_FIELDS = [
    "sequence_group_id",
    "position_in_sequence",
    "sequence_join_from_prev_type",
    "sequence_join_from_prev_reason",
    "sequence_join_from_prev_flags",
    *base.RUN_FIELDS,
]

SEQUENCE_EDGE_FIELDS = [
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
    "sequence_edge_action",
    "sequence_join_type",
    "sequence_edge_reason",
    "sequence_review_flags",
    "prev_page_kind",
    "page_kind",
    "prev_table_family_key",
    "table_family_key",
    "prev_table_number_key",
    "table_number_key",
    "strict_title_similarity",
    "scope_similarity",
    "existing_edge_action",
    "existing_edge_reason",
    "existing_edge_review_flags",
    "existing_collection_candidate",
    "prev_religion_entities",
    "religion_entities",
    "prev_part_entities",
    "part_entities",
    "prev_sex_entities",
    "sex_entities",
    "prev_parenthetical_numbers",
    "parenthetical_numbers",
    "prev_primary_title",
    "primary_title",
    "prev_example_images",
    "example_images",
]

SEQUENCE_REVIEW_FIELDS = ["review_priority", "review_category", *SEQUENCE_EDGE_FIELDS]

OLD_OVERASSOCIATION_FIELDS = [
    "old_collection_group_id",
    "v5_source_key",
    "old_n_physical_runs",
    "old_n_physical_pages",
    "old_page_ranges",
    "old_missing_gap_count",
    "old_missing_pages",
    "v7_sequence_group_count",
    "v7_sequence_group_ids",
    "v7_page_ranges",
    "audit_flags",
    "primary_title",
    "collection_review_flags",
    "example_images",
]


CLASSIFIER_SUFFIX_PATTERNS = [
    r"\bage civil condition\b",
    r"\bage sex civil condition\b",
    r"\bage sex population\b",
    r"\bage by religion\b",
    r"\bage distribution\b",
    r"\bsingle year age\b",
    r"\bcaste class age\b",
    r"\bother structured\b",
    r"\bmain age civil\b",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def first_page_num(run: dict[str, object]) -> int | None:
    return base.int_or_none(run.get("_first_pdf_page"))


def last_page_num(run: dict[str, object]) -> int | None:
    return base.int_or_none(run.get("_last_pdf_page"))


def page_range_from_pages(pages: list[int]) -> str:
    if not pages:
        return ""
    if len(pages) == 1:
        return str(pages[0])
    return f"{pages[0]}-{pages[-1]}"


def clean_title_key(value: object, *, remove_religion: bool = False) -> str:
    text = base.clean_text(value)
    for pattern in CLASSIFIER_SUFFIX_PATTERNS:
        text = re.sub(pattern, " ", text)
    text = re.sub(r"\b(contd|continued|continuation|concluded|concld|concl)\b", " ", text)
    text = re.sub(r"\btable\s+(viii|vii|vi+a?|iii|iv|ix|x|[0-9]+)\b", " table ", text)
    text = re.sub(r"\bimperial table\b|\bprovincial table\b|\bstate table\b", " table ", text)
    if remove_religion:
        for raw in sorted(layout.RELIGION_ALIASES, key=len, reverse=True):
            text = re.sub(rf"\b{re.escape(raw)}\b", " ", text)
        text = re.sub(r"\ball religions?\b|\bother religions?\b", " ", text)
    text = re.sub(r"\bpage\s*\d+\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def combine_titles(rows: list[dict[str, object]], run: dict[str, object]) -> str:
    values = [run.get("primary_title", ""), run.get("title_norms", "")]
    values.extend(row.get("title", "") for row in rows)
    return " ".join(str(v or "") for v in values if str(v or "").strip())


def entity_sets(text: str) -> dict[str, set[str]]:
    return layout.subtable_entities({"title": text, "title_norm": text})


def sorted_join(values: set[str]) -> str:
    return "; ".join(sorted(v for v in values if v))


def run_feature(run: dict[str, object], members: list[dict[str, object]]) -> dict[str, object]:
    pages = sorted(
        p
        for p in (base.int_or_none(member.get("pdf_page")) for member in members)
        if p is not None
    )
    title_text = combine_titles(members, run)
    entities = entity_sets(title_text)
    page_kind_counts = Counter(str(member.get("page_kind") or "") for member in members if str(member.get("page_kind") or ""))
    family_counts = Counter(str(member.get("table_family_key") or "") for member in members if str(member.get("table_family_key") or ""))
    table_counts = Counter(str(member.get("table_number_key") or "") for member in members if str(member.get("table_number_key") or ""))
    feature = {
        "physical_run_id": run.get("physical_run_id", ""),
        "first_pdf_page": pages[0] if pages else "",
        "last_pdf_page": pages[-1] if pages else "",
        "pages": pages,
        "page_range": page_range_from_pages(pages),
        "page_kind": page_kind_counts.most_common(1)[0][0] if page_kind_counts else "",
        "table_family_key": family_counts.most_common(1)[0][0] if family_counts else "",
        "table_number_key": table_counts.most_common(1)[0][0] if table_counts else "",
        "strict_title_key": clean_title_key(title_text, remove_religion=False),
        "scope_without_religion_key": clean_title_key(title_text, remove_religion=True),
        "religion_entities": entities["religion"],
        "part_entities": entities["part"],
        "sex_entities": entities["sex"],
        "parenthetical_numbers": entities["paren_number"],
        "primary_title": run.get("primary_title", ""),
        "example_images": run.get("example_images", ""),
    }
    return feature


def enrich_runs(runs: list[dict[str, str]], members: list[dict[str, str]]) -> tuple[list[dict[str, object]], dict[str, list[dict[str, object]]], dict[str, dict[str, object]]]:
    members_by_run: dict[str, list[dict[str, object]]] = defaultdict(list)
    for member in members:
        members_by_run[str(member.get("physical_run_id") or "")].append(dict(member))
    features: dict[str, dict[str, object]] = {}
    enriched: list[dict[str, object]] = []
    for run in runs:
        run_obj: dict[str, object] = dict(run)
        run_members = sorted(members_by_run[str(run.get("physical_run_id") or "")], key=lambda r: int(r.get("position_in_physical_run") or 0))
        feature = run_feature(run_obj, run_members)
        features[str(run.get("physical_run_id") or "")] = feature
        run_obj["_first_pdf_page"] = feature["first_pdf_page"]
        run_obj["_last_pdf_page"] = feature["last_pdf_page"]
        run_obj["_page_range"] = feature["page_range"]
        run_obj["_page_kind"] = feature["page_kind"]
        run_obj["_table_family_key"] = feature["table_family_key"]
        run_obj["_table_number_key"] = feature["table_number_key"]
        run_obj["_strict_title_key"] = feature["strict_title_key"]
        run_obj["_scope_without_religion_key"] = feature["scope_without_religion_key"]
        run_obj["_religion_entities"] = feature["religion_entities"]
        run_obj["_part_entities"] = feature["part_entities"]
        run_obj["_sex_entities"] = feature["sex_entities"]
        run_obj["_parenthetical_numbers"] = feature["parenthetical_numbers"]
        enriched.append(run_obj)
    return enriched, members_by_run, features


def build_existing_edge_map(edges: list[dict[str, str]], members: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, str]]:
    page_to_run = {str(member.get("physical_page_id") or ""): str(member.get("physical_run_id") or "") for member in members}
    out: dict[tuple[str, str], dict[str, str]] = {}
    for edge in edges:
        prev_run = page_to_run.get(str(edge.get("prev_physical_page_id") or ""))
        cur_run = page_to_run.get(str(edge.get("physical_page_id") or ""))
        if prev_run and cur_run and prev_run != cur_run:
            out[(prev_run, cur_run)] = edge
    return out


def has_conflicting_entities(prev: dict[str, object], cur: dict[str, object]) -> tuple[bool, list[str]]:
    conflicts: list[str] = []
    for field, label in [
        ("_part_entities", "part_switch"),
        ("_sex_entities", "sex_switch"),
        ("_parenthetical_numbers", "parenthetical_number_switch"),
    ]:
        a = prev.get(field) or set()
        b = cur.get(field) or set()
        if isinstance(a, set) and isinstance(b, set) and a and b and a != b:
            conflicts.append(f"{label}_{sorted(a)}_to_{sorted(b)}")
    return bool(conflicts), conflicts


def missing_entity_flags(prev: dict[str, object], cur: dict[str, object]) -> list[str]:
    flags: list[str] = []
    for field, label in [
        ("_part_entities", "part_entity_incomplete"),
        ("_sex_entities", "sex_entity_incomplete"),
        ("_parenthetical_numbers", "parenthetical_number_incomplete"),
    ]:
        a = prev.get(field) or set()
        b = cur.get(field) or set()
        if isinstance(a, set) and isinstance(b, set) and bool(a) != bool(b):
            flags.append(label)
    return flags


def subject_compatible(prev: dict[str, object], cur: dict[str, object]) -> tuple[bool, str]:
    prev_kind = str(prev.get("_page_kind") or "")
    cur_kind = str(cur.get("_page_kind") or "")
    if prev_kind != cur_kind:
        return False, f"page_kind_switch_{prev_kind}_to_{cur_kind}"
    if prev_kind in {"context_or_blank", "offtarget_table", "unknown", ""}:
        return False, f"non_sequence_page_kind_{prev_kind or 'missing'}"
    if prev_kind == "derivative_selected_caste_race_tribe":
        return False, "derivative_selected_caste_race_tribe_boundary"

    prev_family = str(prev.get("_table_family_key") or "")
    cur_family = str(cur.get("_table_family_key") or "")
    if prev_family and cur_family and prev_family != cur_family:
        return False, f"table_family_switch_{prev_family}_to_{cur_family}"

    prev_table = str(prev.get("_table_number_key") or "")
    cur_table = str(cur.get("_table_number_key") or "")
    if prev_table and cur_table and prev_table != cur_table:
        return False, f"table_number_switch_{prev_table}_to_{cur_table}"

    return True, "same_sequence_subject"


def sequence_decision(prev: dict[str, object], cur: dict[str, object], existing_edge: dict[str, str] | None) -> dict[str, object]:
    prev_last = last_page_num(prev)
    cur_first = first_page_num(cur)
    flags: list[str] = []
    if prev_last is None or cur_first is None:
        return {
            "sequence_edge_action": "break",
            "sequence_join_type": "",
            "sequence_edge_reason": "missing_real_pdf_page_for_sequence_boundary",
            "sequence_review_flags": "missing_pdf_page",
            "page_gap": "",
            "missing_pages": "",
            "strict_title_similarity": "",
            "scope_similarity": "",
        }
    page_gap = cur_first - prev_last
    missing_pages = ""
    if page_gap > 1:
        missing_pages = f"{prev_last + 1}" if page_gap == 2 else f"{prev_last + 1}-{cur_first - 1}"
    if page_gap <= 0:
        return {
            "sequence_edge_action": "break",
            "sequence_join_type": "",
            "sequence_edge_reason": f"non_increasing_run_order_{prev_last}_to_{cur_first}",
            "sequence_review_flags": "non_increasing_page_order",
            "page_gap": page_gap,
            "missing_pages": "",
            "strict_title_similarity": "",
            "scope_similarity": "",
        }

    subject_ok, subject_reason = subject_compatible(prev, cur)
    strict_sim = base.text_similarity(prev.get("_strict_title_key"), cur.get("_strict_title_key"))
    scope_sim = base.text_similarity(prev.get("_scope_without_religion_key"), cur.get("_scope_without_religion_key"))
    entity_conflict, entity_conflicts = has_conflicting_entities(prev, cur)
    religion_prev = prev.get("_religion_entities") if isinstance(prev.get("_religion_entities"), set) else set()
    religion_cur = cur.get("_religion_entities") if isinstance(cur.get("_religion_entities"), set) else set()
    religion_variant = bool(religion_prev and religion_cur and religion_prev != religion_cur)

    base_result = {
        "page_gap": page_gap,
        "missing_pages": missing_pages,
        "strict_title_similarity": base.float_fmt(strict_sim),
        "scope_similarity": base.float_fmt(scope_sim),
    }
    if not subject_ok:
        return {
            **base_result,
            "sequence_edge_action": "break",
            "sequence_join_type": "",
            "sequence_edge_reason": subject_reason,
            "sequence_review_flags": "sequence_subject_switch",
        }
    if entity_conflict:
        return {
            **base_result,
            "sequence_edge_action": "break",
            "sequence_join_type": "",
            "sequence_edge_reason": "; ".join(entity_conflicts),
            "sequence_review_flags": "sequence_scope_entity_switch",
        }

    flags.extend(missing_entity_flags(prev, cur))
    if existing_edge and "no_ocr_structural_profile_for_pair" in str(existing_edge.get("edge_review_flags") or ""):
        flags.append("no_ocr_structural_profile_for_sequence_edge")
    if existing_edge and existing_edge.get("collection_review_flags"):
        flags.append(str(existing_edge.get("collection_review_flags")))

    join_type = ""
    reason = ""
    if religion_variant and scope_sim >= 0.88:
        join_type = "religion_variant_subtable_sequence"
        reason = f"adjacent_same_scope_religion_variant; scope_similarity_{scope_sim:.2f}"
        flags.append("religion_variant_same_table_scope")
    elif strict_sim >= 0.88:
        join_type = "same_title_contiguous_sequence"
        reason = f"adjacent_same_table_title; strict_title_similarity_{strict_sim:.2f}"
    elif scope_sim >= 0.92 and (existing_edge or strict_sim >= 0.72):
        join_type = "same_scope_contiguous_sequence"
        reason = f"adjacent_same_table_scope; scope_similarity_{scope_sim:.2f}; strict_title_similarity_{strict_sim:.2f}"
        if strict_sim < 0.82:
            flags.append("weak_strict_title_same_scope_review")
    elif existing_edge and existing_edge.get("collection_candidate") == "yes" and scope_sim >= 0.86:
        join_type = "existing_collection_candidate_sequence"
        reason = f"adjacent_existing_collection_candidate; scope_similarity_{scope_sim:.2f}; strict_title_similarity_{strict_sim:.2f}"
        flags.append("existing_collection_candidate_review")
    else:
        return {
            **base_result,
            "sequence_edge_action": "break",
            "sequence_join_type": "",
            "sequence_edge_reason": f"not_confident_same_bounded_sequence; {subject_reason}; strict_title_similarity_{strict_sim:.2f}; scope_similarity_{scope_sim:.2f}",
            "sequence_review_flags": "possible_related_but_not_grouped_review" if scope_sim >= 0.78 else "",
        }

    if page_gap > 1:
        return {
            **base_result,
            "sequence_edge_action": "bridge_candidate",
            "sequence_join_type": join_type,
            "sequence_edge_reason": f"missing_pages_block_sequence_join_{missing_pages}; {reason}",
            "sequence_review_flags": base.uniq_join(["missing_intermediate_pages", "not_joined_across_gap", *flags], limit=20),
        }
    if page_gap != 1:
        return {
            **base_result,
            "sequence_edge_action": "break",
            "sequence_join_type": "",
            "sequence_edge_reason": f"non_adjacent_sequence_boundary_gap_{page_gap}",
            "sequence_review_flags": "non_adjacent_sequence_boundary",
        }

    if strict_sim < 0.80 and not religion_variant:
        flags.append("weak_title_sequence_review")
    return {
        **base_result,
        "sequence_edge_action": "join_sequence",
        "sequence_join_type": join_type,
        "sequence_edge_reason": reason,
        "sequence_review_flags": base.uniq_join(flags, limit=20),
    }


def edge_row(prev: dict[str, object], cur: dict[str, object], decision: dict[str, object], existing_edge: dict[str, str] | None) -> dict[str, object]:
    prev_religions = prev.get("_religion_entities") if isinstance(prev.get("_religion_entities"), set) else set()
    cur_religions = cur.get("_religion_entities") if isinstance(cur.get("_religion_entities"), set) else set()
    prev_parts = prev.get("_part_entities") if isinstance(prev.get("_part_entities"), set) else set()
    cur_parts = cur.get("_part_entities") if isinstance(cur.get("_part_entities"), set) else set()
    prev_sexes = prev.get("_sex_entities") if isinstance(prev.get("_sex_entities"), set) else set()
    cur_sexes = cur.get("_sex_entities") if isinstance(cur.get("_sex_entities"), set) else set()
    prev_parens = prev.get("_parenthetical_numbers") if isinstance(prev.get("_parenthetical_numbers"), set) else set()
    cur_parens = cur.get("_parenthetical_numbers") if isinstance(cur.get("_parenthetical_numbers"), set) else set()
    return {
        "sequence_edge_id": base.stable_id("seqedge", prev.get("physical_run_id"), cur.get("physical_run_id")),
        "v5_source_key": prev.get("v5_source_key", ""),
        "prev_physical_run_id": prev.get("physical_run_id", ""),
        "physical_run_id": cur.get("physical_run_id", ""),
        "prev_run_page_range": prev.get("_page_range", prev.get("page_range", "")),
        "run_page_range": cur.get("_page_range", cur.get("page_range", "")),
        "prev_last_pdf_page": prev.get("_last_pdf_page", ""),
        "first_pdf_page": cur.get("_first_pdf_page", ""),
        **decision,
        "prev_page_kind": prev.get("_page_kind", ""),
        "page_kind": cur.get("_page_kind", ""),
        "prev_table_family_key": prev.get("_table_family_key", ""),
        "table_family_key": cur.get("_table_family_key", ""),
        "prev_table_number_key": prev.get("_table_number_key", ""),
        "table_number_key": cur.get("_table_number_key", ""),
        "existing_edge_action": existing_edge.get("edge_action", "") if existing_edge else "",
        "existing_edge_reason": existing_edge.get("edge_reason", "") if existing_edge else "",
        "existing_edge_review_flags": existing_edge.get("edge_review_flags", "") if existing_edge else "",
        "existing_collection_candidate": existing_edge.get("collection_candidate", "") if existing_edge else "",
        "prev_religion_entities": sorted_join(prev_religions),
        "religion_entities": sorted_join(cur_religions),
        "prev_part_entities": sorted_join(prev_parts),
        "part_entities": sorted_join(cur_parts),
        "prev_sex_entities": sorted_join(prev_sexes),
        "sex_entities": sorted_join(cur_sexes),
        "prev_parenthetical_numbers": sorted_join(prev_parens),
        "parenthetical_numbers": sorted_join(cur_parens),
        "prev_primary_title": prev.get("primary_title", ""),
        "primary_title": cur.get("primary_title", ""),
        "prev_example_images": prev.get("example_images", ""),
        "example_images": cur.get("example_images", ""),
    }


def build_sequences(runs: list[dict[str, object]], edge_by_run_pair: dict[tuple[str, str], dict[str, str]]) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    runs_by_source: dict[str, list[dict[str, object]]] = defaultdict(list)
    for run in runs:
        runs_by_source[str(run.get("v5_source_key") or "")].append(run)

    groups: list[dict[str, object]] = []
    members: list[dict[str, object]] = []
    seq_edges: list[dict[str, object]] = []
    bridge_candidates: list[dict[str, object]] = []

    def flush(chain: list[dict[str, object]], join_edges: list[dict[str, object]]) -> None:
        if not chain:
            return
        first = chain[0]
        last = chain[-1]
        pages: list[int] = []
        flags: list[str] = []
        reasons: list[str] = []
        join_types: list[str] = []
        religion_entities: set[str] = set()
        part_entities: set[str] = set()
        sex_entities: set[str] = set()
        paren_entities: set[str] = set()
        for run in chain:
            pages.extend(p for p in (first_page_num(run), last_page_num(run)) if p is not None)
            for flag in str(run.get("run_review_flags") or "").split(";"):
                flag = flag.strip()
                if flag:
                    flags.append(flag)
            religion_entities.update(run.get("_religion_entities") if isinstance(run.get("_religion_entities"), set) else set())
            part_entities.update(run.get("_part_entities") if isinstance(run.get("_part_entities"), set) else set())
            sex_entities.update(run.get("_sex_entities") if isinstance(run.get("_sex_entities"), set) else set())
            paren_entities.update(run.get("_parenthetical_numbers") if isinstance(run.get("_parenthetical_numbers"), set) else set())
        for edge in join_edges:
            join_types.append(str(edge.get("sequence_join_type") or ""))
            reasons.append(str(edge.get("sequence_edge_reason") or ""))
            for flag in str(edge.get("sequence_review_flags") or "").split(";"):
                flag = flag.strip()
                if flag:
                    flags.append(flag)
        unique_pages = sorted(set(pages))
        all_pages_consecutive = all(b == a + 1 for a, b in zip(unique_pages, unique_pages[1:]))
        if not all_pages_consecutive:
            flags.append("sequence_internal_page_gap")
        if len(chain) > 12:
            flags.append("large_bounded_sequence_review")
        basis = "single_physical_run" if len(chain) == 1 else "bounded_adjacent_logical_table_sequence"
        if any(j == "religion_variant_subtable_sequence" for j in join_types):
            basis = "bounded_religion_variant_subtable_sequence"
        sequence_id = base.stable_id("seq", first.get("v5_source_key"), first.get("physical_run_id"), last.get("physical_run_id"), len(chain))
        groups.append({
            "sequence_group_id": sequence_id,
            "sequence_basis": basis,
            "review_flag": "review" if flags or len(chain) > 1 else str(first.get("review_flag") or "ok"),
            "sequence_review_flags": base.uniq_join(flags, limit=30),
            "v5_source_key": first.get("v5_source_key", ""),
            "source_pdf": base.uniq_join((run.get("source_pdf", "") for run in chain), limit=4),
            "source_pdf_id": base.uniq_join((run.get("source_pdf_id", "") for run in chain), limit=4),
            "n_physical_runs": len(chain),
            "n_physical_pages": sum(int(run.get("n_physical_pages") or 0) for run in chain),
            "page_range": page_range_from_pages(unique_pages),
            "page_ranges": "; ".join(str(run.get("_page_range") or run.get("page_range") or "") for run in chain),
            "all_pages_consecutive": str(all_pages_consecutive),
            "sequence_join_types": base.uniq_join(join_types, limit=10),
            "page_kind_counts": base.counter_json((run.get("_page_kind", "") for run in chain)),
            "table_family_counts": base.counter_json((run.get("_table_family_key", "") for run in chain)),
            "table_number_keys": base.uniq_join((run.get("_table_number_key", "") for run in chain), limit=8),
            "religion_entities": sorted_join(religion_entities),
            "part_entities": sorted_join(part_entities),
            "sex_entities": sorted_join(sex_entities),
            "parenthetical_numbers": sorted_join(paren_entities),
            "primary_title": first.get("primary_title", ""),
            "strict_title_key": str(first.get("_strict_title_key") or ""),
            "scope_without_religion_key": str(first.get("_scope_without_religion_key") or ""),
            "sequence_reasons": base.uniq_join(reasons, limit=20),
            "example_images": base.uniq_join((run.get("example_images", "") for run in chain), limit=4),
        })
        for pos, run in enumerate(chain, start=1):
            join_edge = join_edges[pos - 2] if pos > 1 and pos - 2 < len(join_edges) else {}
            members.append({
                "sequence_group_id": sequence_id,
                "position_in_sequence": pos,
                "sequence_join_from_prev_type": join_edge.get("sequence_join_type", "") if join_edge else "",
                "sequence_join_from_prev_reason": join_edge.get("sequence_edge_reason", "") if join_edge else "",
                "sequence_join_from_prev_flags": join_edge.get("sequence_review_flags", "") if join_edge else "",
                **run,
            })

    for source, source_runs in sorted(runs_by_source.items()):
        ordered = sorted(source_runs, key=lambda r: (
            first_page_num(r) if first_page_num(r) is not None else 10**9,
            last_page_num(r) if last_page_num(r) is not None else 10**9,
            str(r.get("physical_run_id") or ""),
        ))
        chain: list[dict[str, object]] = []
        chain_edges: list[dict[str, object]] = []
        prev: dict[str, object] | None = None
        for run in ordered:
            if prev is None:
                chain = [run]
                chain_edges = []
                prev = run
                continue
            existing_edge = edge_by_run_pair.get((str(prev.get("physical_run_id")), str(run.get("physical_run_id"))))
            decision = sequence_decision(prev, run, existing_edge)
            row = edge_row(prev, run, decision, existing_edge)
            seq_edges.append(row)
            if row.get("sequence_edge_action") == "bridge_candidate":
                bridge_candidates.append(row)
            if row.get("sequence_edge_action") == "join_sequence":
                chain.append(run)
                chain_edges.append(row)
            else:
                flush(chain, chain_edges)
                chain = [run]
                chain_edges = []
            prev = run
        flush(chain, chain_edges)

    groups.sort(key=lambda r: (
        str(r.get("v5_source_key") or ""),
        base.int_or_none(str(r.get("page_range") or "").split("-", 1)[0]) if base.int_or_none(str(r.get("page_range") or "").split("-", 1)[0]) is not None else 10**9,
        str(r.get("sequence_group_id") or ""),
    ))
    members.sort(key=lambda r: (
        str(r.get("sequence_group_id") or ""),
        int(r.get("position_in_sequence") or 0),
    ))
    seq_edges.sort(key=lambda r: (
        str(r.get("v5_source_key") or ""),
        base.int_or_none(r.get("prev_last_pdf_page")) if base.int_or_none(r.get("prev_last_pdf_page")) is not None else 10**9,
        base.int_or_none(r.get("first_pdf_page")) if base.int_or_none(r.get("first_pdf_page")) is not None else 10**9,
        str(r.get("sequence_edge_id") or ""),
    ))
    bridge_candidates.sort(key=lambda r: (
        str(r.get("v5_source_key") or ""),
        base.int_or_none(r.get("prev_last_pdf_page")) if base.int_or_none(r.get("prev_last_pdf_page")) is not None else 10**9,
        base.int_or_none(r.get("first_pdf_page")) if base.int_or_none(r.get("first_pdf_page")) is not None else 10**9,
    ))
    return groups, members, seq_edges, bridge_candidates


def build_sequence_review_queue(seq_edges: list[dict[str, object]], groups: list[dict[str, object]]) -> list[dict[str, object]]:
    queue: list[dict[str, object]] = []
    for edge in seq_edges:
        action = str(edge.get("sequence_edge_action") or "")
        flags = str(edge.get("sequence_review_flags") or "")
        category = ""
        priority = 99
        if action == "join_sequence" and flags:
            category = "joined_sequence_review"
            priority = 3
        elif action == "bridge_candidate":
            category = "missing_gap_bridge_candidate"
            priority = 2
        elif "sequence_scope_entity_switch" in flags:
            category = "sequence_scope_entity_switch"
            priority = 4
        elif "sequence_subject_switch" in flags:
            category = "sequence_subject_switch"
            priority = 5
        elif "possible_related_but_not_grouped_review" in flags:
            category = "possible_related_not_grouped"
            priority = 6
        if category:
            queue.append({"review_priority": priority, "review_category": category, **edge})
    queue.sort(key=lambda r: (
        int(r.get("review_priority") or 99),
        str(r.get("v5_source_key") or ""),
        base.int_or_none(r.get("prev_last_pdf_page")) if base.int_or_none(r.get("prev_last_pdf_page")) is not None else 10**9,
    ))
    return queue


def sequence_invariants(groups: list[dict[str, object]], seq_edges: list[dict[str, object]]) -> dict[str, object]:
    failures: list[str] = []
    for group in groups:
        if int(group.get("n_physical_runs") or 0) > 1 and str(group.get("all_pages_consecutive")) != "True":
            failures.append(f"sequence_internal_gap:{group.get('sequence_group_id')}:{group.get('page_ranges')}")
    for edge in seq_edges:
        if edge.get("sequence_edge_action") == "join_sequence" and str(edge.get("page_gap")) != "1":
            failures.append(f"sequence_join_non_adjacent:{edge.get('prev_physical_run_id')}->{edge.get('physical_run_id')}:{edge.get('page_gap')}")
    return {
        "sequence_invariant_failure_count": len(failures),
        "sequence_invariant_failures": failures[:50],
    }


def old_collection_comparison(old_collections: list[dict[str, str]], new_groups: list[dict[str, object]]) -> list[dict[str, object]]:
    old_multi = [r for r in old_collections if int(r.get("n_physical_runs") or 0) > 1]
    new_multi = [r for r in new_groups if int(r.get("n_physical_runs") or 0) > 1]
    rows = [
        {
            "metric": "old_multi_run_collections",
            "old_value": len(old_multi),
            "new_value": "",
            "note": "v6 collection groups inherited from v5 rules",
        },
        {
            "metric": "new_multi_run_sequences",
            "old_value": "",
            "new_value": len(new_multi),
            "note": "v7 bounded adjacent sequence groups",
        },
        {
            "metric": "old_max_collection_runs",
            "old_value": max((int(r.get("n_physical_runs") or 0) for r in old_collections), default=0),
            "new_value": "",
            "note": "",
        },
        {
            "metric": "new_max_sequence_runs",
            "old_value": "",
            "new_value": max((int(r.get("n_physical_runs") or 0) for r in new_groups), default=0),
            "note": "",
        },
    ]
    return rows


def parse_page_range_list(value: object) -> list[int]:
    pages: list[int] = []
    for part in str(value or "").split(";"):
        part = part.strip()
        if not part:
            continue
        if re.fullmatch(r"\d+", part):
            pages.append(int(part))
            continue
        match = re.fullmatch(r"(\d+)\s*-\s*(\d+)", part)
        if match:
            start, end = int(match.group(1)), int(match.group(2))
            if start <= end:
                pages.extend(range(start, end + 1))
    return sorted(set(pages))


def missing_page_segments(pages: list[int]) -> tuple[int, str]:
    segments: list[str] = []
    count = 0
    for left, right in zip(pages, pages[1:]):
        if right <= left + 1:
            continue
        count += right - left - 1
        start, end = left + 1, right - 1
        segments.append(str(start) if start == end else f"{start}-{end}")
    return count, "; ".join(segments)


def old_overassociation_audit(
    old_collections: list[dict[str, str]],
    old_members: list[dict[str, str]],
    sequence_groups: list[dict[str, object]],
    sequence_members: list[dict[str, object]],
) -> list[dict[str, object]]:
    old_members_by_collection: dict[str, list[dict[str, str]]] = defaultdict(list)
    for member in old_members:
        old_members_by_collection[str(member.get("collection_group_id") or "")].append(member)

    run_to_sequence: dict[str, str] = {}
    for member in sequence_members:
        run_id = str(member.get("physical_run_id") or "")
        if run_id:
            run_to_sequence[run_id] = str(member.get("sequence_group_id") or "")
    sequence_by_id = {str(group.get("sequence_group_id") or ""): group for group in sequence_groups}

    rows: list[dict[str, object]] = []
    for group in old_collections:
        n_runs = int(group.get("n_physical_runs") or 0)
        if n_runs <= 1:
            continue
        collection_id = str(group.get("collection_group_id") or "")
        pages = parse_page_range_list(group.get("page_ranges"))
        missing_count, missing_pages = missing_page_segments(pages)
        seq_ids: list[str] = []
        for member in sorted(old_members_by_collection.get(collection_id, []), key=lambda r: int(r.get("position_in_collection") or 0)):
            seq_id = run_to_sequence.get(str(member.get("physical_run_id") or ""))
            if seq_id and seq_id not in seq_ids:
                seq_ids.append(seq_id)
        flags: list[str] = []
        if missing_count:
            flags.append("old_collection_crossed_missing_pages")
        if len(seq_ids) > 1:
            flags.append("old_collection_split_by_adjacency_v7")
        if n_runs > 12:
            flags.append("old_large_broad_collection")
        if "collection_crosses_missing_pages_review" in str(group.get("collection_review_flags") or ""):
            flags.append("old_existing_gap_review_flag")
        if not flags:
            continue
        page_ranges = [str(sequence_by_id.get(seq_id, {}).get("page_range") or "") for seq_id in seq_ids]
        rows.append({
            "old_collection_group_id": collection_id,
            "v5_source_key": group.get("v5_source_key", ""),
            "old_n_physical_runs": n_runs,
            "old_n_physical_pages": group.get("n_physical_pages", ""),
            "old_page_ranges": group.get("page_ranges", ""),
            "old_missing_gap_count": missing_count,
            "old_missing_pages": missing_pages,
            "v7_sequence_group_count": len(seq_ids),
            "v7_sequence_group_ids": base.uniq_join(seq_ids, limit=20),
            "v7_page_ranges": base.uniq_join(page_ranges, limit=20),
            "audit_flags": base.uniq_join(flags, limit=10),
            "primary_title": group.get("primary_title", ""),
            "collection_review_flags": group.get("collection_review_flags", ""),
            "example_images": group.get("example_images", ""),
        })
    rows.sort(key=lambda r: (
        -int(r.get("old_missing_gap_count") or 0),
        -int(r.get("old_n_physical_runs") or 0),
        str(r.get("v5_source_key") or ""),
    ))
    return rows


def is_core_table_vii_text(value: object) -> bool:
    text = base.clean_text(value)
    return re.search(r"\btable\s+vii\b(?!i)", text) is not None


def normalize_explicit_number(value: str) -> str:
    roman = {
        "i": "1",
        "ii": "2",
        "iii": "3",
        "iv": "4",
        "v": "5",
        "vi": "6",
        "via": "6a",
        "vii": "7",
        "viii": "8",
        "ix": "9",
        "x": "10",
    }
    value = base.clean_text(value)
    return roman.get(value, value)


def first_explicit_table_number(value: object) -> str:
    text = base.clean_text(value)
    # Prefer the printed table identifier over trailing classifier suffixes like
    # "age_civil_condition VII"; those suffixes are often copied from nearby
    # OCR metadata and can mislabel Table VIII pages as Table VII.
    pattern = re.compile(
        r"\b(?:imperial|provincial|state|final census)?\s*"
        r"(?:table|form|statement|no|number)\s*"
        r"(?:no|number)?\s*"
        r"(viii|vii|vi+a?|ix|iv|iii|ii|i|x|\d{1,3}|[a-z]{1,3})\b"
    )
    for match in pattern.finditer(text):
        number = normalize_explicit_number(match.group(1))
        # Ignore alpha variants such as "No. VB" rather than treating them as
        # roman V.
        if re.fullmatch(r"\d+[a-z]?", number):
            return number
    return ""


NON_CORE_TABLE_VII_TERMS = [
    "education",
    "literacy",
    "literate",
    "pupil",
    "scholar",
    "school",
    "expenditure",
    "selected caste",
    "selected castes",
    "selected tribe",
    "selected tribes",
    "selected race",
    "selected races",
    "birth place",
    "birth places",
    "birth rate",
    "birth-rate",
    "occupation",
    "language",
    "infirm",
    "blind",
    "deaf",
    "dumb",
    "insane",
    "afflicted",
    "nationalities",
]


def has_non_core_table_vii_topic(value: object) -> bool:
    text = base.clean_text(value)
    return any(term in text for term in NON_CORE_TABLE_VII_TERMS)


def core_vii_topic_like(value: object) -> bool:
    text = base.clean_text(value)
    if has_non_core_table_vii_topic(value):
        return False
    has_age = "age" in text or "ages" in text
    has_core_subject = any(
        token in text
        for token in [
            "civil condition",
            "marital condition",
            "conjugal condition",
            "age sex",
            "age and sex",
            "religion",
            "religions",
        ]
    )
    return has_age and has_core_subject


def is_core_table_vii_side(title: object, table_number_keys: object = "") -> bool:
    title_text = str(title or "")
    explicit = first_explicit_table_number(title_text)
    if explicit:
        if explicit != "7":
            return False
        if has_non_core_table_vii_topic(title_text):
            return False
        # Subsidiary Table VII is often a derivative numbering namespace. Keep it
        # in the core subset only when its subject still matches the money-table
        # Table VII age/religion/civil-condition scope.
        if "subsidiary table" in base.clean_text(title_text):
            return core_vii_topic_like(title_text)
        return True
    numbers = {token.strip() for token in str(table_number_keys or "").split(";") if token.strip()}
    return "7" in numbers and core_vii_topic_like(title_text)


def is_core_table_vii_group(group: dict[str, object]) -> bool:
    title_text = " ".join([
        str(group.get("primary_title") or ""),
        str(group.get("strict_title_key") or ""),
        str(group.get("scope_without_religion_key") or ""),
    ])
    return is_core_table_vii_side(title_text, group.get("table_number_keys"))


def is_core_table_vii_edge(edge: dict[str, object]) -> bool:
    prev_core = is_core_table_vii_side(edge.get("prev_primary_title"), edge.get("prev_table_number_key"))
    cur_core = is_core_table_vii_side(edge.get("primary_title"), edge.get("table_number_key"))
    return prev_core and cur_core


def main() -> None:
    runs_raw = read_csv(PHYSICAL_RUNS)
    members_raw = read_csv(RUN_MEMBERS)
    edges_raw = read_csv(PHYSICAL_EDGES)
    required_physical_edges = read_csv(REQUIRED_PHYSICAL_EDGES)
    if not runs_raw or not members_raw or not edges_raw:
        raise SystemExit(f"Missing v6 strict inputs under {V6_DIR}")

    runs, members_by_run, features = enrich_runs(runs_raw, members_raw)
    edge_by_run_pair = build_existing_edge_map(edges_raw, members_raw)
    sequence_groups, sequence_members, sequence_edges, bridge_candidates = build_sequences(runs, edge_by_run_pair)
    sequence_review_queue = build_sequence_review_queue(sequence_edges, sequence_groups)
    old_collections = read_csv(OLD_COLLECTION_GROUPS)
    old_collection_members = read_csv(OLD_COLLECTION_MEMBERS)
    comparison_rows = old_collection_comparison(old_collections, sequence_groups)
    old_overassociation_rows = old_overassociation_audit(
        old_collections,
        old_collection_members,
        sequence_groups,
        sequence_members,
    )
    core_group_ids = {str(group.get("sequence_group_id")) for group in sequence_groups if is_core_table_vii_group(group)}
    core_sequence_groups = [group for group in sequence_groups if str(group.get("sequence_group_id")) in core_group_ids]
    core_sequence_members = [member for member in sequence_members if str(member.get("sequence_group_id")) in core_group_ids]
    core_bridge_candidates = [edge for edge in bridge_candidates if is_core_table_vii_edge(edge)]
    core_sequence_edges = [
        edge
        for edge in sequence_edges
        if is_core_table_vii_edge(edge) and edge.get("sequence_edge_action") in {"join_sequence", "bridge_candidate"}
    ]

    summary = {
        "input_v6_physical_runs": len(runs),
        "input_v6_run_members": len(members_raw),
        "input_v6_required_physical_edges": len(required_physical_edges),
        "sequence_groups": len(sequence_groups),
        "multi_run_sequence_groups": sum(1 for g in sequence_groups if int(g.get("n_physical_runs") or 0) > 1),
        "max_sequence_physical_runs": max((int(g.get("n_physical_runs") or 0) for g in sequence_groups), default=0),
        "max_sequence_physical_pages": max((int(g.get("n_physical_pages") or 0) for g in sequence_groups), default=0),
        "joined_sequence_edges": sum(1 for e in sequence_edges if e.get("sequence_edge_action") == "join_sequence"),
        "bridge_candidate_edges": len(bridge_candidates),
        "break_sequence_edges": sum(1 for e in sequence_edges if e.get("sequence_edge_action") == "break"),
        "sequence_review_queue_rows": len(sequence_review_queue),
        "religion_variant_sequence_groups": sum(1 for g in sequence_groups if g.get("sequence_basis") == "bounded_religion_variant_subtable_sequence"),
        "large_sequence_review_groups": sum(1 for g in sequence_groups if "large_bounded_sequence_review" in str(g.get("sequence_review_flags") or "")),
        "old_v6_multi_run_collection_groups": sum(1 for g in old_collections if int(g.get("n_physical_runs") or 0) > 1),
        "old_v6_max_collection_runs": max((int(g.get("n_physical_runs") or 0) for g in old_collections), default=0),
        "old_v6_overassociation_audit_rows": len(old_overassociation_rows),
        "old_v6_gap_crossing_audit_rows": sum(1 for g in old_overassociation_rows if int(g.get("old_missing_gap_count") or 0) > 0),
        "old_v6_split_by_v7_audit_rows": sum(1 for g in old_overassociation_rows if "old_collection_split_by_adjacency_v7" in str(g.get("audit_flags") or "")),
        "core_table_vii_sequence_groups": len(core_sequence_groups),
        "core_table_vii_multi_run_sequence_groups": sum(1 for g in core_sequence_groups if int(g.get("n_physical_runs") or 0) > 1),
        "core_table_vii_sequence_members": len(core_sequence_members),
        "core_table_vii_sequence_edges": len(core_sequence_edges),
        "core_table_vii_bridge_candidates": len(core_bridge_candidates),
    }
    summary.update(sequence_invariants(sequence_groups, sequence_edges))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base.write_csv(OUT_DIR / "v7_sequence_groups.csv", sequence_groups, SEQUENCE_GROUP_FIELDS)
    base.write_csv(OUT_DIR / "v7_sequence_members.csv", sequence_members, SEQUENCE_MEMBER_FIELDS)
    base.write_csv(OUT_DIR / "v7_sequence_edges.csv", sequence_edges, SEQUENCE_EDGE_FIELDS)
    base.write_csv(OUT_DIR / "v7_missing_bridge_candidates.csv", bridge_candidates, SEQUENCE_EDGE_FIELDS)
    base.write_csv(OUT_DIR / "v7_sequence_review_queue.csv", sequence_review_queue, SEQUENCE_REVIEW_FIELDS)
    base.write_csv(OUT_DIR / "v7_collection_comparison.csv", comparison_rows, ["metric", "old_value", "new_value", "note"])
    base.write_csv(OUT_DIR / "v7_old_overassociation_audit.csv", old_overassociation_rows, OLD_OVERASSOCIATION_FIELDS)
    base.write_csv(OUT_DIR / "v7_required_physical_edges_passthrough.csv", required_physical_edges, base.EDGE_FIELDS)
    base.write_csv(OUT_DIR / "v7_core_tableVII_sequence_groups.csv", core_sequence_groups, SEQUENCE_GROUP_FIELDS)
    base.write_csv(OUT_DIR / "v7_core_tableVII_sequence_members.csv", core_sequence_members, SEQUENCE_MEMBER_FIELDS)
    base.write_csv(OUT_DIR / "v7_core_tableVII_sequence_edges.csv", core_sequence_edges, SEQUENCE_EDGE_FIELDS)
    base.write_csv(OUT_DIR / "v7_core_tableVII_bridge_candidates.csv", core_bridge_candidates, SEQUENCE_EDGE_FIELDS)

    with (OUT_DIR / "v7_sequence_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    summary_rows = [{"metric": key, "value": json.dumps(value) if isinstance(value, (dict, list)) else value} for key, value in summary.items()]
    base.write_workbook(
        OUT_DIR / "v7_sequence_review.xlsx",
        {
            "summary": (summary_rows, ["metric", "value"]),
            "sequence_groups": (sequence_groups, SEQUENCE_GROUP_FIELDS),
            "sequence_members": (sequence_members, SEQUENCE_MEMBER_FIELDS),
            "sequence_edges": (sequence_edges, SEQUENCE_EDGE_FIELDS),
            "bridge_candidates": (bridge_candidates, SEQUENCE_EDGE_FIELDS),
            "sequence_review_queue": (sequence_review_queue, SEQUENCE_REVIEW_FIELDS),
            "old_overassoc_audit": (old_overassociation_rows, OLD_OVERASSOCIATION_FIELDS),
            "required_physical": (required_physical_edges, base.EDGE_FIELDS),
            "core_vii_groups": (core_sequence_groups, SEQUENCE_GROUP_FIELDS),
            "core_vii_members": (core_sequence_members, SEQUENCE_MEMBER_FIELDS),
            "core_vii_edges": (core_sequence_edges, SEQUENCE_EDGE_FIELDS),
            "core_vii_bridges": (core_bridge_candidates, SEQUENCE_EDGE_FIELDS),
            "v6_vs_v7": (comparison_rows, ["metric", "old_value", "new_value", "note"]),
        },
    )

    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
