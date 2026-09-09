#!/usr/bin/env python3
"""Classifier-assisted assembly rule audit and v11 proposed components.

This pass performs no OCR/API work. It reads the v10 layered assembly manifest
and the Gemini page-class CSV, then adds a conservative rule-audit layer:

* existing v10 joins are preserved;
* missing-page bridges remain non-joins;
* adjacent main Table VII numeric pages can be joined only when the classifier,
  table key, entity checks, and title similarity all agree;
* plausible split-title, variant, or row-label-dependent joins are exported for
  review rather than merged automatically.

The output is intended as scalable static-site input: use v11 components for
safe assembly, and use the review queues to decide the remaining ambiguous
boundaries.
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import build_all_ocr_component_assembly_v5_structural as base
import build_all_ocr_component_assembly_v10_layered_manifest as v10


ROOT = Path(__file__).resolve().parent
V10_DIR = ROOT / "export_final/pdf_recovery_incoming_2026-06-16_all_ocr_component_assembly_v10_layered_manifest"
CLASSIFIER_CSV = ROOT / (
    "export_final/pdf_recovery_incoming_2026-06-16_no_gaps_ledger_v2_refined/"
    "gemini_ocr_worklists/gemini_classify_priority_all.csv"
)
OUT_DIR = ROOT / "export_final/pdf_recovery_incoming_2026-06-16_all_ocr_component_assembly_v11_rule_audit"

V10_GROUPS = V10_DIR / "v10_assembly_groups.csv"
V10_MEMBERS = V10_DIR / "v10_assembly_members.csv"
V10_EDGES = V10_DIR / "v10_assembly_edges.csv"
V10_REQUIRED_CANDIDATES = V10_DIR / "v10_possible_required_physical_edges_review.csv"
V10_REQUIRED_EDGES = V10_DIR / "v10_required_physical_edges.csv"
V10_BRIDGES = V10_DIR / "v10_missing_bridge_candidates.csv"
V10_OVERASSOC = V10_DIR / "v10_old_overassociation_audit.csv"

MAIN_CLASS = "main_tablevii_numeric"
DERIVATIVE_CLASSES = {"derivative_civil_condition"}
CONTEXT_CLASSES = {"tablevii_context_or_title", "blank_or_empty", "offtarget_table"}

V11_MEMBER_OVERLAY_FIELDS = [
    *v10.V10_MEMBER_FIELDS,
    "v11_classifier_matched_pages",
    "v11_classifier_page_classes",
    "v11_classifier_numeric_grid_pages",
    "v11_classifier_printed_titles",
]

V11_EDGE_FIELDS = [
    *v10.V10_EDGE_FIELDS,
    "v11_prev_classifier_page_classes",
    "v11_classifier_page_classes",
    "v11_prev_classifier_titles",
    "v11_classifier_titles",
    "v11_recommended_action",
    "v11_recommended_layer",
    "v11_recommended_confidence",
    "v11_recommended_reason",
    "v11_review_priority",
]

V11_COMPONENT_FIELDS = [
    "v11_component_id",
    "v11_component_layer",
    "v11_static_site_action",
    "v11_confidence",
    "v11_review_reasons",
    "v5_source_key",
    "source_pdf",
    "source_pdf_id",
    "n_v10_groups",
    "n_physical_runs",
    "n_physical_pages",
    "page_range",
    "page_ranges",
    "all_pages_consecutive",
    "v10_group_ids",
    "v10_group_layers",
    "v10_static_site_actions",
    "v11_join_edge_ids",
    "v11_join_reasons",
    "v11_classifier_page_classes",
    "table_number_keys",
    "primary_title",
    "example_images",
]

V11_COMPONENT_MEMBER_FIELDS = [
    "v11_component_id",
    "position_in_v11_component",
    "v11_join_from_prev_edge_id",
    "v11_join_from_prev_action",
    "v11_join_from_prev_reason",
    *v10.V10_GROUP_FIELDS,
]

V11_REVIEW_QUEUE_FIELDS = [
    "v11_review_priority",
    "v11_review_category",
    *[field for field in V11_EDGE_FIELDS if field != "v11_review_priority"],
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


def parse_pages(value: object) -> list[int]:
    pages: list[int] = []
    for part in str(value or "").split(";"):
        part = part.strip()
        if not part:
            continue
        match = re.fullmatch(r"(\d+)\s*-\s*(\d+)", part)
        if match:
            start, end = int(match.group(1)), int(match.group(2))
            if start <= end:
                pages.extend(range(start, end + 1))
            continue
        if part.isdigit():
            pages.append(int(part))
    return sorted(set(pages))


def page_range_from_pages(pages: list[int]) -> str:
    if not pages:
        return ""
    pages = sorted(set(pages))
    spans: list[str] = []
    start = prev = pages[0]
    for page in pages[1:]:
        if page == prev + 1:
            prev = page
            continue
        spans.append(str(start) if start == prev else f"{start}-{prev}")
        start = prev = page
    spans.append(str(start) if start == prev else f"{start}-{prev}")
    return "; ".join(spans)


def infer_source_id(row: dict[str, object]) -> str:
    if str(row.get("source_pdf_id") or "").strip():
        return str(row.get("source_pdf_id") or "").strip()
    blob = " ".join(
        str(row.get(key) or "")
        for key in ["v5_source_key", "source_pdf", "image_path", "example_images", "prev_example_images"]
    )
    match = re.search(r"(?:saoa\.crl\.)?(\d{7,9})(?:\.pdf|[_/\\-]p\d+|/|$)", blob)
    return match.group(1) if match else ""


def classifier_index(rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, str]]:
    return {
        (str(row.get("source_id") or ""), str(row.get("pdf_page") or "")): row
        for row in rows
        if row.get("source_id") and row.get("pdf_page")
    }


def classifier_overlay_for_member(
    member: dict[str, object],
    by_source_page: dict[tuple[str, str], dict[str, str]],
) -> list[dict[str, str]]:
    source_id = infer_source_id(member)
    out: list[dict[str, str]] = []
    for page in parse_pages(member.get("page_range")):
        row = by_source_page.get((source_id, str(page)))
        if row:
            out.append(row)
    return out


def uniq(values: list[object], *, limit: int = 20) -> str:
    return base.uniq_join([str(v or "") for v in values if str(v or "").strip()], limit=limit)


def enrich_members_with_classifier(
    members: list[dict[str, str]],
    by_source_page: dict[tuple[str, str], dict[str, str]],
) -> tuple[list[dict[str, object]], dict[str, list[dict[str, str]]]]:
    out: list[dict[str, object]] = []
    overlay_by_run: dict[str, list[dict[str, str]]] = {}
    for member in members:
        overlay = classifier_overlay_for_member(member, by_source_page)
        run_id = str(member.get("physical_run_id") or "")
        overlay_by_run[run_id] = overlay
        numeric_pages = [
            str(row.get("pdf_page") or "")
            for row in overlay
            if str(row.get("contains_numeric_grid") or "").lower() in {"true", "yes", "1"}
        ]
        out.append({
            **member,
            "v11_classifier_matched_pages": len(overlay),
            "v11_classifier_page_classes": uniq([row.get("page_class") for row in overlay], limit=10),
            "v11_classifier_numeric_grid_pages": uniq(numeric_pages, limit=20),
            "v11_classifier_printed_titles": uniq([row.get("printed_title") for row in overlay], limit=8),
        })
    return out, overlay_by_run


def class_set(rows: list[dict[str, str]]) -> set[str]:
    return {str(row.get("page_class") or "") for row in rows if str(row.get("page_class") or "").strip()}


def printed_titles(rows: list[dict[str, str]]) -> str:
    return " ".join(str(row.get("printed_title") or "") for row in rows if str(row.get("printed_title") or "").strip())


def normalized_title(value: object) -> str:
    text = base.clean_text(value)
    text = re.sub(r"\b(continued|concluded|contd|concld|cont)\b", " ", text)
    text = re.sub(r"\b(table|form)\s+(vii|viia|via|7|6a)\b", " table ", text)
    text = re.sub(r"\bpart\s+[a-zivx0-9-]+\b", " part ", text)
    text = re.sub(r"\bpage\s+\d+\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def title_containment_or_similarity(prev_title: str, cur_title: str) -> tuple[bool, str]:
    prev_norm = normalized_title(prev_title)
    cur_norm = normalized_title(cur_title)
    if not prev_norm or not cur_norm:
        return False, "classifier_title_missing"
    if prev_norm in cur_norm or cur_norm in prev_norm:
        return True, "classifier_printed_title_containment"
    sim = base.text_similarity(prev_norm, cur_norm)
    if sim >= 0.74:
        return True, f"classifier_printed_title_similarity_{sim:.2f}"
    return False, f"classifier_printed_title_similarity_{sim:.2f}"


def has_bad_classifier_boundary(prev_classes: set[str], cur_classes: set[str]) -> tuple[bool, str]:
    if prev_classes & DERIVATIVE_CLASSES or cur_classes & DERIVATIVE_CLASSES:
        return True, "classifier_derivative_table_boundary"
    if prev_classes & CONTEXT_CLASSES or cur_classes & CONTEXT_CLASSES:
        return True, "classifier_context_blank_or_offtarget_boundary"
    return False, ""


def row_label_missing(edge: dict[str, str]) -> bool:
    ratio = as_float(edge.get("row_label_min_ratio"))
    return bool(str(edge.get("no_row_label_pages") or "").strip()) or (ratio > 0 and ratio < 0.35)


def prev_has_row_context(edge: dict[str, str]) -> bool:
    ratio = as_float(edge.get("prev_row_label_min_ratio"))
    return ratio >= 0.55 or bool(str(edge.get("prev_internal_join_reasons") or "").strip())


def has_scope_entity_conflict(edge: dict[str, str]) -> bool:
    return "sequence_scope_entity_switch" in str(edge.get("sequence_review_flags") or "")


def same_normalized_table_key(edge: dict[str, str]) -> bool:
    prev_key = normalize_table_key(edge.get("prev_table_number_key"))
    cur_key = normalize_table_key(edge.get("table_number_key"))
    return bool(prev_key and cur_key and prev_key == cur_key)


def compatible_classifier_main(prev_classes: set[str], cur_classes: set[str]) -> bool:
    return prev_classes == {MAIN_CLASS} and cur_classes == {MAIN_CLASS}


def classify_edge_recommendation(
    edge: dict[str, str],
    overlay_by_run: dict[str, list[dict[str, str]]],
) -> dict[str, object]:
    prev_overlay = overlay_by_run.get(str(edge.get("prev_physical_run_id") or ""), [])
    cur_overlay = overlay_by_run.get(str(edge.get("physical_run_id") or ""), [])
    prev_classes = class_set(prev_overlay)
    cur_classes = class_set(cur_overlay)
    prev_titles = printed_titles(prev_overlay)
    cur_titles = printed_titles(cur_overlay)
    action = str(edge.get("sequence_edge_action") or "")
    page_gap = str(edge.get("page_gap") or "")
    strict_sim = as_float(edge.get("strict_title_similarity"))
    scope_sim = as_float(edge.get("scope_similarity"))
    bad_boundary, bad_reason = has_bad_classifier_boundary(prev_classes, cur_classes)

    recommended_action = "hard_break"
    layer = "hard_break"
    confidence = ""
    reason = str(edge.get("sequence_edge_reason") or "")
    priority = ""

    if action == "join_sequence":
        recommended_action = "keep_existing_join"
        layer = str(edge.get("v10_edge_layer") or "logical_body_sequence")
        confidence = str(edge.get("v10_edge_confidence") or "review")
        reason = str(edge.get("v10_edge_review_reason") or edge.get("sequence_edge_reason") or "existing_v10_join")
    elif action == "bridge_candidate":
        recommended_action = "keep_missing_bridge"
        layer = "missing_bridge_candidate"
        confidence = "review"
        reason = "missing_intermediate_pages_block_join"
        priority = "2"
    elif page_gap != "1":
        recommended_action = "hard_break_or_missing_page"
        layer = "hard_break"
        reason = "not_adjacent_real_pdf_pages"
    elif bad_boundary:
        recommended_action = "hard_break"
        layer = "hard_break"
        reason = bad_reason
    elif row_label_missing(edge) and prev_has_row_context(edge) and not has_scope_entity_conflict(edge):
        recommended_action = "review_required_row_label_join"
        layer = "possible_required_physical_row_label_assembly"
        confidence = "review"
        reason = "current_run_has_missing_or_deficient_row_labels_and_previous_run_has_row_context"
        priority = "1"
    elif compatible_classifier_main(prev_classes, cur_classes) and not has_scope_entity_conflict(edge):
        same_key = same_normalized_table_key(edge)
        title_ok, title_reason = title_containment_or_similarity(prev_titles or edge.get("prev_primary_title", ""), cur_titles or edge.get("primary_title", ""))
        if same_key and max(strict_sim, scope_sim) >= 0.70:
            recommended_action = "join_classifier_main_tablevii_high"
            layer = "classifier_assisted_body_sequence"
            confidence = "high_review"
            reason = (
                "adjacent_classifier_main_tablevii_numeric_same_table_key; "
                f"strict_title_similarity_{strict_sim:.2f}; scope_similarity_{scope_sim:.2f}"
            )
            priority = "1"
        elif same_key and title_ok:
            recommended_action = "review_classifier_main_tablevii_split_or_short_title_join"
            layer = "possible_classifier_assisted_body_sequence"
            confidence = "review"
            reason = f"adjacent_classifier_main_tablevii_numeric_same_table_key; {title_reason}"
            priority = "2"
        elif same_key:
            recommended_action = "review_classifier_main_tablevii_low_similarity"
            layer = "possible_classifier_assisted_body_sequence"
            confidence = "review"
            reason = f"adjacent_classifier_main_tablevii_numeric_same_table_key_but_{title_reason}"
            priority = "3"
        else:
            recommended_action = "review_classifier_main_tablevii_boundary"
            layer = "possible_classifier_assisted_body_sequence"
            confidence = "review"
            reason = "adjacent_classifier_main_tablevii_numeric_but_missing_or_different_table_key"
            priority = "3"
    elif edge.get("v8_candidate_related") == "yes":
        recommended_action = "review_related_not_grouped"
        layer = "candidate_related_not_grouped"
        confidence = "review"
        reason = str(edge.get("v8_candidate_reason") or edge.get("sequence_edge_reason") or "")
        priority = "4"

    return {
        **edge,
        "v11_prev_classifier_page_classes": uniq(sorted(prev_classes), limit=10),
        "v11_classifier_page_classes": uniq(sorted(cur_classes), limit=10),
        "v11_prev_classifier_titles": uniq([row.get("printed_title") for row in prev_overlay], limit=6),
        "v11_classifier_titles": uniq([row.get("printed_title") for row in cur_overlay], limit=6),
        "v11_recommended_action": recommended_action,
        "v11_recommended_layer": layer,
        "v11_recommended_confidence": confidence,
        "v11_recommended_reason": reason,
        "v11_review_priority": priority,
    }


class UnionFind:
    def __init__(self, values: list[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self.parent.setdefault(value, value)
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left != root_right:
            self.parent[root_right] = root_left


def build_v11_components(
    groups: list[dict[str, str]],
    members: list[dict[str, object]],
    edge_rows: list[dict[str, object]],
    join_actions: set[str] | None = None,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    if join_actions is None:
        join_actions = {"join_classifier_main_tablevii_high"}
    groups_by_id = {str(group.get("sequence_group_id") or ""): group for group in groups}
    run_to_group = {
        str(member.get("physical_run_id") or ""): str(member.get("sequence_group_id") or "")
        for member in members
        if member.get("physical_run_id") and member.get("sequence_group_id")
    }
    group_ids = [str(group.get("sequence_group_id") or "") for group in groups]
    uf = UnionFind(group_ids)
    joining_edges_by_pair: dict[tuple[str, str], dict[str, object]] = {}
    for edge in edge_rows:
        if str(edge.get("v11_recommended_action") or "") not in join_actions:
            continue
        left = run_to_group.get(str(edge.get("prev_physical_run_id") or ""))
        right = run_to_group.get(str(edge.get("physical_run_id") or ""))
        if not left or not right or left == right:
            continue
        uf.union(left, right)
        joining_edges_by_pair[(left, right)] = edge

    grouped_ids: dict[str, list[str]] = defaultdict(list)
    for group_id in group_ids:
        grouped_ids[uf.find(group_id)].append(group_id)

    # Sort component atoms in actual page order, not union-find insertion order.
    def group_sort_key(group_id: str) -> tuple[str, int, str]:
        group = groups_by_id[group_id]
        pages = parse_pages(group.get("page_range"))
        return (
            str(group.get("v5_source_key") or ""),
            pages[0] if pages else 10**9,
            group_id,
        )

    component_rows: list[dict[str, object]] = []
    component_members: list[dict[str, object]] = []
    members_by_group: dict[str, list[dict[str, object]]] = defaultdict(list)
    for member in members:
        members_by_group[str(member.get("sequence_group_id") or "")].append(member)

    for atom_ids in grouped_ids.values():
        atom_ids = sorted(atom_ids, key=group_sort_key)
        first = groups_by_id[atom_ids[0]]
        pages: list[int] = []
        all_classes: list[str] = []
        join_edges: list[dict[str, object]] = []
        review_reasons: list[str] = []
        n_physical_runs = 0
        n_physical_pages = 0
        for idx, group_id in enumerate(atom_ids):
            group = groups_by_id[group_id]
            pages.extend(parse_pages(group.get("page_range")))
            n_physical_runs += as_int(group.get("n_physical_runs"))
            n_physical_pages += as_int(group.get("n_physical_pages"))
            if str(group.get("v10_group_review_reasons") or "").strip():
                review_reasons.append(str(group.get("v10_group_review_reasons") or ""))
            for member in members_by_group.get(group_id, []):
                all_classes.extend(str(member.get("v11_classifier_page_classes") or "").split(";"))
            if idx > 0:
                prev_group_id = atom_ids[idx - 1]
                edge = joining_edges_by_pair.get((prev_group_id, group_id), {})
                if edge:
                    join_edges.append(edge)

        pages = sorted(set(pages))
        all_consecutive = all(b == a + 1 for a, b in zip(pages, pages[1:]))
        atom_layers = [str(groups_by_id[group_id].get("v10_group_layer") or "") for group_id in atom_ids]
        atom_actions = [str(groups_by_id[group_id].get("v10_static_site_action") or "") for group_id in atom_ids]
        if len(atom_ids) > 1:
            layer = "classifier_assisted_body_sequence"
            action = "static_group_candidate_classifier_assisted"
            confidence = "high_review"
            review_reasons.append("classifier_assisted_join_added_in_v11")
        else:
            layer = str(first.get("v10_group_layer") or "")
            action = str(first.get("v10_static_site_action") or "")
            confidence = str(first.get("v10_group_confidence") or "")

        component_id = base.stable_id("v11comp", atom_ids[0], atom_ids[-1], len(atom_ids))
        component_rows.append({
            "v11_component_id": component_id,
            "v11_component_layer": layer,
            "v11_static_site_action": action,
            "v11_confidence": confidence,
            "v11_review_reasons": uniq(review_reasons, limit=20),
            "v5_source_key": first.get("v5_source_key", ""),
            "source_pdf": first.get("source_pdf", ""),
            "source_pdf_id": first.get("source_pdf_id", ""),
            "n_v10_groups": len(atom_ids),
            "n_physical_runs": n_physical_runs,
            "n_physical_pages": n_physical_pages,
            "page_range": page_range_from_pages(pages),
            "page_ranges": uniq([groups_by_id[group_id].get("page_ranges") for group_id in atom_ids], limit=40),
            "all_pages_consecutive": str(all_consecutive),
            "v10_group_ids": uniq(atom_ids, limit=50),
            "v10_group_layers": uniq(atom_layers, limit=20),
            "v10_static_site_actions": uniq(atom_actions, limit=20),
            "v11_join_edge_ids": uniq([edge.get("sequence_edge_id") for edge in join_edges], limit=20),
            "v11_join_reasons": uniq([edge.get("v11_recommended_reason") for edge in join_edges], limit=20),
            "v11_classifier_page_classes": uniq(all_classes, limit=20),
            "table_number_keys": uniq([groups_by_id[group_id].get("table_number_keys") for group_id in atom_ids], limit=20),
            "primary_title": first.get("primary_title", ""),
            "example_images": uniq([groups_by_id[group_id].get("example_images") for group_id in atom_ids], limit=10),
        })
        for pos, group_id in enumerate(atom_ids, start=1):
            edge = joining_edges_by_pair.get((atom_ids[pos - 2], group_id), {}) if pos > 1 else {}
            component_members.append({
                "v11_component_id": component_id,
                "position_in_v11_component": pos,
                "v11_join_from_prev_edge_id": edge.get("sequence_edge_id", ""),
                "v11_join_from_prev_action": edge.get("v11_recommended_action", ""),
                "v11_join_from_prev_reason": edge.get("v11_recommended_reason", ""),
                **groups_by_id[group_id],
            })

    component_rows.sort(key=lambda row: (
        str(row.get("v5_source_key") or ""),
        parse_pages(row.get("page_range"))[0] if parse_pages(row.get("page_range")) else 10**9,
        str(row.get("v11_component_id") or ""),
    ))
    component_members.sort(key=lambda row: (
        str(row.get("v11_component_id") or ""),
        as_int(row.get("position_in_v11_component")),
    ))
    return component_rows, component_members


def review_queue(edge_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    queue: list[dict[str, object]] = []
    for edge in edge_rows:
        action = str(edge.get("v11_recommended_action") or "")
        priority = str(edge.get("v11_review_priority") or "")
        if not priority:
            continue
        if action in {"keep_existing_join", "join_classifier_main_tablevii_high"}:
            continue
        category = action.replace("review_", "")
        queue.append({
            "v11_review_priority": priority,
            "v11_review_category": category,
            **edge,
        })
    queue.sort(key=lambda row: (
        as_int(row.get("v11_review_priority")) or 99,
        str(row.get("v5_source_key") or ""),
        as_int(row.get("prev_last_pdf_page")) or 10**9,
        as_int(row.get("first_pdf_page")) or 10**9,
    ))
    return queue


def validate(
    components: list[dict[str, object]],
    edges: list[dict[str, object]],
) -> dict[str, object]:
    failures: list[str] = []
    for component in components:
        if as_int(component.get("n_v10_groups")) > 1 and str(component.get("all_pages_consecutive")) != "True":
            failures.append(f"v11_component_internal_gap:{component.get('v11_component_id')}:{component.get('page_range')}")
    for edge in edges:
        if edge.get("v11_recommended_action") == "join_classifier_main_tablevii_high" and str(edge.get("page_gap")) != "1":
            failures.append(f"v11_high_join_non_adjacent:{edge.get('sequence_edge_id')}:{edge.get('page_gap')}")
    return {
        "v11_invariant_failure_count": len(failures),
        "v11_invariant_failures": failures[:100],
    }


def write_readme(summary: dict[str, object]) -> None:
    lines = [
        "# v11 Rule Audit",
        "",
        "This pass makes no OCR/API calls. It overlays Gemini page classes from the existing classifier CSV onto v10 components.",
        "",
        "## Join Policy",
        "",
        "- Existing v10 joins are preserved.",
        "- Missing PDF-page gaps remain bridge candidates and are not joined.",
        "- Classifier-assisted joins require adjacent PDF pages, both pages classified `main_tablevii_numeric`, matching normalized table key, no scope/entity conflict, and strong title/scope similarity.",
        "- Row-label, split-title, low-similarity main Table VII, and related-but-not-grouped edges are exported for review rather than merged.",
        "",
        "## Summary",
        "",
    ]
    for key in sorted(summary):
        lines.append(f"- `{key}`: {summary[key]}")
    (OUT_DIR / "v11_rules_readme.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    csv.field_size_limit(10 ** 7)
    groups = read_csv(V10_GROUPS)
    members = read_csv(V10_MEMBERS)
    edges = read_csv(V10_EDGES)
    classifier_rows = read_csv(CLASSIFIER_CSV)
    required_candidate_rows = read_csv(V10_REQUIRED_CANDIDATES)
    required_edge_rows = read_csv(V10_REQUIRED_EDGES)
    bridge_rows = read_csv(V10_BRIDGES)
    overassoc_rows = read_csv(V10_OVERASSOC)
    if not groups or not members or not edges:
        raise SystemExit("Missing v10 inputs; run v10 first")

    classifier_by_page = classifier_index(classifier_rows)
    enriched_members, overlay_by_run = enrich_members_with_classifier(members, classifier_by_page)
    edge_rows = [classify_edge_recommendation(edge, overlay_by_run) for edge in edges]
    components, component_members = build_v11_components(groups, enriched_members, edge_rows)
    queue = review_queue(edge_rows)

    action_counts = Counter(str(edge.get("v11_recommended_action") or "") for edge in edge_rows)
    component_layer_counts = Counter(str(component.get("v11_component_layer") or "") for component in components)
    classifier_member_matches = sum(1 for member in enriched_members if as_int(member.get("v11_classifier_matched_pages")) > 0)
    summary = {
        "input_v10_groups": len(groups),
        "input_v10_members": len(members),
        "input_v10_edges": len(edges),
        "classifier_rows": len(classifier_rows),
        "classifier_member_matches": classifier_member_matches,
        "v11_components": len(components),
        "v11_component_members": len(component_members),
        "v11_edge_rows": len(edge_rows),
        "v11_review_queue_edges": len(queue),
        "v11_classifier_high_join_edges": action_counts.get("join_classifier_main_tablevii_high", 0),
        "v11_classifier_review_join_edges": sum(
            count
            for action, count in action_counts.items()
            if action.startswith("review_classifier_main_tablevii")
        ),
        "v11_required_row_label_review_edges": action_counts.get("review_required_row_label_join", 0),
        "v11_kept_existing_joins": action_counts.get("keep_existing_join", 0),
        "v11_kept_missing_bridges": action_counts.get("keep_missing_bridge", 0),
        "v11_hard_breaks": action_counts.get("hard_break", 0) + action_counts.get("hard_break_or_missing_page", 0),
        "v11_component_layer_counts": dict(component_layer_counts),
        "v11_edge_action_counts": dict(action_counts),
    }
    summary.update(validate(components, edge_rows))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base.write_csv(OUT_DIR / "v11_member_classifier_overlay.csv", enriched_members, V11_MEMBER_OVERLAY_FIELDS)
    base.write_csv(OUT_DIR / "v11_edge_recommendations.csv", edge_rows, V11_EDGE_FIELDS)
    base.write_csv(OUT_DIR / "v11_components.csv", components, V11_COMPONENT_FIELDS)
    base.write_csv(OUT_DIR / "v11_component_members.csv", component_members, V11_COMPONENT_MEMBER_FIELDS)
    base.write_csv(OUT_DIR / "v11_edge_review_queue.csv", queue, V11_REVIEW_QUEUE_FIELDS)
    base.write_csv(OUT_DIR / "v11_required_physical_review_passthrough.csv", required_candidate_rows, v10.V10_REQUIRED_CANDIDATE_FIELDS)
    base.write_csv(OUT_DIR / "v11_required_physical_edges_passthrough.csv", required_edge_rows, v10.V10_REQUIRED_EDGE_FIELDS)
    base.write_csv(
        OUT_DIR / "v11_missing_bridge_candidates_passthrough.csv",
        bridge_rows,
        list(bridge_rows[0].keys()) if bridge_rows else [],
    )
    base.write_csv(
        OUT_DIR / "v11_old_overassociation_audit_passthrough.csv",
        overassoc_rows,
        list(overassoc_rows[0].keys()) if overassoc_rows else [],
    )

    with (OUT_DIR / "v11_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    write_readme(summary)

    summary_rows = [
        {"metric": key, "value": json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value}
        for key, value in summary.items()
    ]
    base.write_workbook(
        OUT_DIR / "v11_rule_audit_review.xlsx",
        {
            "summary": (summary_rows, ["metric", "value"]),
            "components": (components, V11_COMPONENT_FIELDS),
            "component_members": (component_members, V11_COMPONENT_MEMBER_FIELDS),
            "edge_recommendations": (edge_rows, V11_EDGE_FIELDS),
            "edge_review_queue": (queue, V11_REVIEW_QUEUE_FIELDS),
            "member_classifier_overlay": (enriched_members, V11_MEMBER_OVERLAY_FIELDS),
            "required_physical_review": (required_candidate_rows, v10.V10_REQUIRED_CANDIDATE_FIELDS),
        },
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
