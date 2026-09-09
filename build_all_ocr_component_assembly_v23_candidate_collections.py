#!/usr/bin/env python3
"""Candidate collection layer over v22 immediate review edges.

No OCR/API calls are made. This pass turns individual v22 candidate edges into
bounded, consecutive review collections. It does not auto-merge anything:
physical joins and logical sibling collections remain review/worklist records.
"""

from __future__ import annotations

import csv
import json
import re
import zipfile
from collections import Counter, defaultdict, deque
from pathlib import Path

from PIL import Image, ImageDraw

import build_all_ocr_component_assembly_v5_structural as base
import build_all_ocr_component_assembly_v11_rule_audit as v11
import build_all_ocr_component_assembly_v13_scope_refined as v13


ROOT = Path(__file__).resolve().parent
V22_DIR = ROOT / "export_final/pdf_recovery_incoming_2026-06-16_all_ocr_component_assembly_v22_candidate_audit"
OUT_DIR = ROOT / "export_final/pdf_recovery_incoming_2026-06-16_all_ocr_component_assembly_v23_candidate_collections"

V22_IMMEDIATE = V22_DIR / "v22_immediate_review_candidates.csv"
V22_SUMMARY = V22_DIR / "v22_summary.json"

COLLECTION_FIELDS = [
    "v23_candidate_collection_id",
    "collection_type",
    "recommended_action",
    "can_auto_apply",
    "review_priority",
    "v5_source_key",
    "source_pdf",
    "source_pdf_id",
    "page_range",
    "all_pages_consecutive",
    "n_edges",
    "n_runs",
    "candidate_classes",
    "candidate_layers",
    "table_family_keys",
    "table_number_keys",
    "max_similarity_min",
    "max_similarity_max",
    "has_left_stub_signal",
    "has_right_half_signal",
    "has_part_switch_signal",
    "has_continuation_marker",
    "edge_ids",
    "run_ids",
    "member_page_ranges",
    "primary_titles",
    "example_images",
    "visual_evidence",
    "reason",
]

MEMBER_FIELDS = [
    "v23_candidate_collection_id",
    "position",
    "member_run_id",
    "v5_source_key",
    "source_pdf",
    "source_pdf_id",
    "page_range",
    "role",
    "candidate_classes_touching",
    "left_stub",
    "right_half",
    "primary_title",
    "example_images",
]

EDGE_FIELDS = [
    "v23_candidate_collection_id",
    "position",
    *[
        "sequence_edge_id",
        "candidate_class",
        "recommendation",
        "review_priority",
        "safe_whole_page_join",
        "can_auto_join_now",
        "candidate_layer",
        "v5_source_key",
        "source_pdf",
        "source_pdf_id",
        "prev_physical_run_id",
        "physical_run_id",
        "prev_run_page_range",
        "run_page_range",
        "prev_last_pdf_page",
        "first_pdf_page",
        "page_gap",
        "missing_pages",
        "same_family_and_key",
        "is_core_table_family",
        "max_similarity",
        "prev_table_family_key",
        "table_family_key",
        "prev_table_number_key",
        "table_number_key",
        "prev_left_stub",
        "prev_right_half",
        "cur_left_stub",
        "cur_right_half",
        "reason",
        "prev_primary_title",
        "primary_title",
        "prev_example_images",
        "example_images",
    ],
]

AUDIT_FIELDS = ["check", "status", "failure_count", "details"]
SUMMARY_FIELDS = ["metric", "value"]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def pages(value: object) -> list[int]:
    return v11.parse_pages(value)


def as_float(value: object) -> float:
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0


def source_pdf_from_key(source_key: object) -> str:
    key = str(source_key or "")
    return key if key.lower().endswith(".pdf") else ""


def source_pdf_id_from_key(source_key: object) -> str:
    match = re.search(r"saoa\.crl\.(\d+)\.pdf$", str(source_key or ""), flags=re.IGNORECASE)
    return match.group(1) if match else ""


def compact_page_range(page_values: list[int]) -> str:
    if not page_values:
        return ""
    values = sorted(set(page_values))
    ranges: list[str] = []
    start = prev = values[0]
    for value in values[1:]:
        if value == prev + 1:
            prev = value
            continue
        ranges.append(str(start) if start == prev else f"{start}-{prev}")
        start = prev = value
    ranges.append(str(start) if start == prev else f"{start}-{prev}")
    return "; ".join(ranges)


def all_consecutive(page_values: list[int]) -> bool:
    values = sorted(set(page_values))
    if not values:
        return False
    return values == list(range(values[0], values[-1] + 1))


def has_continuation_marker(value: object) -> bool:
    return bool(re.search(r"\b(contd|continued|concld|concluded|ctd)\b|cont[.-]", str(value or "").lower()))


def has_part_switch(edge: dict[str, str]) -> bool:
    reason = " ".join(
        [
            str(edge.get("sequence_edge_reason") or ""),
            str(edge.get("v17_recommended_reason") or ""),
            str(edge.get("reason") or ""),
        ]
    ).lower()
    title = " ".join([str(edge.get("prev_primary_title") or ""), str(edge.get("primary_title") or "")]).lower()
    return "part_switch" in reason or bool(re.search(r"\bpart\s*[\(\-]?[a-zivx]+", title))


def classify_collection(edges: list[dict[str, str]]) -> tuple[str, str, str]:
    classes = {str(edge.get("candidate_class") or "") for edge in edges}
    has_left_stub = any(edge.get("prev_left_stub") == "yes" for edge in edges)
    has_right_half = any(edge.get("cur_right_half") == "yes" or edge.get("prev_right_half") == "yes" for edge in edges)
    part_switch = any(has_part_switch(edge) for edge in edges)
    continuation = any(
        has_continuation_marker(edge.get("prev_primary_title")) or has_continuation_marker(edge.get("primary_title"))
        for edge in edges
    )

    if part_switch:
        return (
            "logical_sibling_subtable_review_collection",
            "manual_review_for_logical_sibling_grouping",
            "Adjacent same table has part/religion/status switches; likely logical siblings, not a single physical table body.",
        )
    if "review_possible_left_stub_continuation" in classes and len(edges) <= 2:
        return (
            "physical_dependency_review_collection",
            "manual_review_for_possible_required_page_assembly",
            "left-stub/right-half style evidence may indicate a required physical continuation; keep review-only until confirmed.",
        )
    if continuation and has_right_half:
        return (
            "physical_or_logical_continuation_review_collection",
            "manual_review_for_continuation_type",
            "Adjacent continued same-table pages include right-half/continuation signals but no prior rule was confident enough.",
        )
    if continuation:
        return (
            "logical_continuation_review_collection",
            "manual_review_for_logical_sequence",
            "Adjacent same-table continuation titles remain ungrouped but lack a strong physical dependency signal.",
        )
    return (
        "same_core_table_review_collection",
        "manual_review_for_same_core_table_boundary",
        "Adjacent same-core table pages have high similarity or weak continuation evidence, but no safe automatic rule applies.",
    )


def build_components(rows: list[dict[str, str]]) -> list[list[dict[str, str]]]:
    by_source: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_source[str(row.get("v5_source_key") or "")].append(row)

    components: list[list[dict[str, str]]] = []
    for source_rows in by_source.values():
        adjacency: defaultdict[str, set[str]] = defaultdict(set)
        edge_by_pair: dict[tuple[str, str], dict[str, str]] = {}
        run_ids: set[str] = set()
        for row in source_rows:
            left = str(row.get("prev_physical_run_id") or "")
            right = str(row.get("physical_run_id") or "")
            if not left or not right:
                continue
            run_ids.update([left, right])
            adjacency[left].add(right)
            adjacency[right].add(left)
            edge_by_pair[(left, right)] = row

        seen: set[str] = set()
        for run_id in sorted(run_ids):
            if run_id in seen:
                continue
            queue = deque([run_id])
            seen.add(run_id)
            component_runs: set[str] = set()
            while queue:
                cur = queue.popleft()
                component_runs.add(cur)
                for nxt in adjacency[cur]:
                    if nxt not in seen:
                        seen.add(nxt)
                        queue.append(nxt)
            component_edges = [
                row
                for (left, right), row in edge_by_pair.items()
                if left in component_runs and right in component_runs
            ]
            component_edges.sort(key=lambda edge: (v13.as_int(edge.get("prev_last_pdf_page")), v13.as_int(edge.get("first_pdf_page")), str(edge.get("sequence_edge_id") or "")))
            if component_edges:
                components.append(component_edges)

    components.sort(
        key=lambda edges: (
            str(edges[0].get("v5_source_key") or ""),
            min(v13.as_int(edge.get("prev_last_pdf_page")) for edge in edges),
            str(edges[0].get("sequence_edge_id") or ""),
        )
    )
    return components


def ordered_members(edges: list[dict[str, str]]) -> list[dict[str, object]]:
    runs: dict[str, dict[str, object]] = {}
    touching: defaultdict[str, list[str]] = defaultdict(list)
    for edge in edges:
        for side, run_field, page_field, title_field, image_field, left_field, right_field in [
            ("prev", "prev_physical_run_id", "prev_run_page_range", "prev_primary_title", "prev_example_images", "prev_left_stub", "prev_right_half"),
            ("cur", "physical_run_id", "run_page_range", "primary_title", "example_images", "cur_left_stub", "cur_right_half"),
        ]:
            run_id = str(edge.get(run_field) or "")
            if not run_id:
                continue
            row = runs.setdefault(
                run_id,
                {
                    "member_run_id": run_id,
                    "v5_source_key": edge.get("v5_source_key", ""),
                    "source_pdf": edge.get("source_pdf", "") or source_pdf_from_key(edge.get("v5_source_key")),
                    "source_pdf_id": edge.get("source_pdf_id", "") or source_pdf_id_from_key(edge.get("v5_source_key")),
                    "page_range": edge.get(page_field, ""),
                    "role": side,
                    "left_stub": edge.get(left_field, "no"),
                    "right_half": edge.get(right_field, "no"),
                    "primary_title": edge.get(title_field, ""),
                    "example_images": edge.get(image_field, ""),
                },
            )
            if side == "cur" and row.get("role") == "prev":
                row["role"] = "interior"
            touching[run_id].append(str(edge.get("candidate_class") or ""))
    member_rows = list(runs.values())
    member_rows.sort(key=lambda row: (min(pages(row.get("page_range")) or [10**9]), str(row.get("member_run_id") or "")))
    for row in member_rows:
        row["candidate_classes_touching"] = base.uniq_join(touching[str(row.get("member_run_id") or "")])
    return member_rows


def first_image(value: object) -> Path | None:
    for part in str(value or "").split(";"):
        candidate = ROOT / part.strip()
        if candidate.exists():
            return candidate
    return None


def make_contactsheet(collection_id: str, members: list[dict[str, object]]) -> str:
    evidence_dir = OUT_DIR / "visual_candidate_collections"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    tiles: list[Image.Image] = []
    for index, member in enumerate(members, start=1):
        image_path = first_image(member.get("example_images"))
        if not image_path:
            continue
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception:
            continue
        image.thumbnail((420, 620))
        tile = Image.new("RGB", (460, 700), "white")
        draw = ImageDraw.Draw(tile)
        label = f"{index}: p{member.get('page_range')} {member.get('member_run_id')}"
        draw.text((12, 12), label[:72], fill="black")
        title = str(member.get("primary_title") or "")
        draw.text((12, 34), title[:72], fill="black")
        tile.paste(image, (20, 65))
        tiles.append(tile)
    if not tiles:
        return ""
    width = min(len(tiles), 6) * 460
    rows = (len(tiles) + 5) // 6
    sheet = Image.new("RGB", (width, rows * 700), "white")
    for idx, tile in enumerate(tiles):
        x = (idx % 6) * 460
        y = (idx // 6) * 700
        sheet.paste(tile, (x, y))
    out = evidence_dir / f"{collection_id}.jpg"
    sheet.save(out, quality=88)
    return str(out.relative_to(ROOT))


def build_outputs(components: list[list[dict[str, str]]]) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    collections: list[dict[str, object]] = []
    members_out: list[dict[str, object]] = []
    edges_out: list[dict[str, object]] = []
    for index, edges in enumerate(components, start=1):
        source = str(edges[0].get("v5_source_key") or "")
        collection_id = base.stable_id("v23cand", source, edges[0].get("sequence_edge_id"), edges[-1].get("sequence_edge_id"), len(edges))
        member_rows = ordered_members(edges)
        collection_type, action, reason = classify_collection(edges)
        page_values: list[int] = []
        for member in member_rows:
            page_values.extend(pages(member.get("page_range")))
        similarities = [as_float(edge.get("max_similarity")) for edge in edges]
        visual = make_contactsheet(collection_id, member_rows)
        for pos, member in enumerate(member_rows, start=1):
            out_member = {
                "v23_candidate_collection_id": collection_id,
                "position": pos,
                **member,
            }
            members_out.append(out_member)
        for pos, edge in enumerate(edges, start=1):
            out_edge = {"v23_candidate_collection_id": collection_id, "position": pos}
            for field in EDGE_FIELDS:
                if field in {"v23_candidate_collection_id", "position"}:
                    continue
                out_edge[field] = edge.get(field, "")
            edges_out.append(out_edge)
        collections.append(
            {
                "v23_candidate_collection_id": collection_id,
                "collection_type": collection_type,
                "recommended_action": action,
                "can_auto_apply": "no",
                "review_priority": min(v13.as_int(edge.get("review_priority")) or 9 for edge in edges),
                "v5_source_key": source,
                "source_pdf": edges[0].get("source_pdf", "") or source_pdf_from_key(source),
                "source_pdf_id": edges[0].get("source_pdf_id", "") or source_pdf_id_from_key(source),
                "page_range": compact_page_range(page_values),
                "all_pages_consecutive": "yes" if all_consecutive(page_values) else "no",
                "n_edges": len(edges),
                "n_runs": len(member_rows),
                "candidate_classes": base.uniq_join(edge.get("candidate_class") for edge in edges),
                "candidate_layers": base.uniq_join(edge.get("candidate_layer") for edge in edges),
                "table_family_keys": base.uniq_join(
                    [edge.get("prev_table_family_key", "") for edge in edges]
                    + [edge.get("table_family_key", "") for edge in edges]
                ),
                "table_number_keys": base.uniq_join(
                    [edge.get("prev_table_number_key", "") for edge in edges]
                    + [edge.get("table_number_key", "") for edge in edges]
                ),
                "max_similarity_min": f"{min(similarities):.3f}" if similarities else "",
                "max_similarity_max": f"{max(similarities):.3f}" if similarities else "",
                "has_left_stub_signal": "yes" if any(edge.get("prev_left_stub") == "yes" or edge.get("cur_left_stub") == "yes" for edge in edges) else "no",
                "has_right_half_signal": "yes" if any(edge.get("prev_right_half") == "yes" or edge.get("cur_right_half") == "yes" for edge in edges) else "no",
                "has_part_switch_signal": "yes" if any(has_part_switch(edge) for edge in edges) else "no",
                "has_continuation_marker": "yes" if any(has_continuation_marker(edge.get("prev_primary_title")) or has_continuation_marker(edge.get("primary_title")) for edge in edges) else "no",
                "edge_ids": base.uniq_join((edge.get("sequence_edge_id") for edge in edges), limit=200),
                "run_ids": base.uniq_join((member.get("member_run_id") for member in member_rows), limit=200),
                "member_page_ranges": base.uniq_join((member.get("page_range") for member in member_rows), limit=200),
                "primary_titles": base.uniq_join((member.get("primary_title") for member in member_rows), limit=20),
                "example_images": base.uniq_join((member.get("example_images") for member in member_rows), limit=20),
                "visual_evidence": visual,
                "reason": reason,
            }
        )
    return collections, members_out, edges_out


def validate(collections: list[dict[str, object]], members: list[dict[str, object]], edges: list[dict[str, object]], input_edges: list[dict[str, str]]) -> list[dict[str, object]]:
    checks: list[tuple[str, list[object]]] = []
    edge_ids_in = {str(edge.get("sequence_edge_id") or "") for edge in input_edges}
    edge_ids_out = {str(edge.get("sequence_edge_id") or "") for edge in edges}
    checks.append(("all_immediate_edges_assigned_to_candidate_collection", sorted(edge_ids_in - edge_ids_out)))
    checks.append(("no_extra_candidate_edges_created", sorted(edge_ids_out - edge_ids_in)))
    auto_apply = [row.get("v23_candidate_collection_id") for row in collections if row.get("can_auto_apply") != "no"]
    checks.append(("no_v23_candidate_collection_auto_applies", auto_apply))
    nonconsecutive_physical = [
        row.get("v23_candidate_collection_id")
        for row in collections
        if row.get("collection_type") == "physical_dependency_review_collection" and row.get("all_pages_consecutive") != "yes"
    ]
    checks.append(("physical_dependency_review_collections_are_consecutive", nonconsecutive_physical))
    missing_visual = [
        row.get("v23_candidate_collection_id")
        for row in collections
        if not row.get("visual_evidence") or not (ROOT / str(row.get("visual_evidence"))).exists()
    ]
    checks.append(("candidate_collections_have_visual_evidence", missing_visual))
    collection_ids = {str(row.get("v23_candidate_collection_id") or "") for row in collections}
    member_bad = sorted({str(row.get("v23_candidate_collection_id") or "") for row in members} - collection_ids)
    edge_bad = sorted({str(row.get("v23_candidate_collection_id") or "") for row in edges} - collection_ids)
    checks.append(("members_reference_existing_candidate_collections", member_bad))
    checks.append(("edges_reference_existing_candidate_collections", edge_bad))
    return [
        {
            "check": check,
            "status": "pass" if not bad else "fail",
            "failure_count": len(bad),
            "details": json.dumps(bad[:20], sort_keys=True),
        }
        for check, bad in checks
    ]


def write_readme(summary: dict[str, object]) -> None:
    lines = [
        "# v23 Candidate Collections",
        "",
        "No OCR/API calls are made. v23 groups the v22 immediate review edges into bounded consecutive candidate collections.",
        "",
        "## Semantics",
        "",
        "- `physical_dependency_review_collection`: possible required page assembly, usually left-stub/right-half evidence.",
        "- `logical_sibling_subtable_review_collection`: likely sibling subtables such as adjacent religion/part variants.",
        "- `physical_or_logical_continuation_review_collection`: continuation evidence exists, but review is required to decide whether this is physical assembly or logical grouping.",
        "- `same_core_table_review_collection`: same core table evidence exists but no safe rule supports automatic grouping.",
        "",
        "Every collection has `can_auto_apply=no`. This layer is a review/scalability package, not a merge pass.",
        "",
        "## Summary",
        "",
    ]
    for key in sorted(summary):
        lines.append(f"- `{key}`: {summary[key]}")
    (OUT_DIR / "v23_rules_readme.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    csv.field_size_limit(10**7)
    rows = read_csv(V22_IMMEDIATE)
    if not rows:
        raise SystemExit("Missing v22 immediate review candidates")
    components = build_components(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    collections, members, edges = build_outputs(components)
    audit = validate(collections, members, edges, rows)
    audit_failures = sum(v13.as_int(row.get("failure_count")) for row in audit)
    collection_counts = Counter(str(row.get("collection_type") or "") for row in collections)
    edge_class_counts = Counter(str(row.get("candidate_class") or "") for row in edges)
    summary = {
        "v23_candidate_collections": len(collections),
        "v23_candidate_collection_members": len(members),
        "v23_candidate_collection_edges": len(edges),
        "v23_input_immediate_edges": len(rows),
        "v23_collection_type_counts": dict(collection_counts),
        "v23_edge_candidate_class_counts": dict(edge_class_counts),
        "v23_auto_apply_collections": sum(1 for row in collections if row.get("can_auto_apply") != "no"),
        "v23_audit_failure_count": audit_failures,
        "v23_audit_status": "pass" if audit_failures == 0 else "fail",
    }

    base.write_csv(OUT_DIR / "v23_candidate_collections.csv", collections, COLLECTION_FIELDS)
    base.write_csv(OUT_DIR / "v23_candidate_collection_members.csv", members, MEMBER_FIELDS)
    base.write_csv(OUT_DIR / "v23_candidate_collection_edges.csv", edges, EDGE_FIELDS)
    base.write_csv(
        OUT_DIR / "v23_physical_dependency_review_collections.csv",
        [row for row in collections if row.get("collection_type") == "physical_dependency_review_collection"],
        COLLECTION_FIELDS,
    )
    base.write_csv(
        OUT_DIR / "v23_logical_sibling_review_collections.csv",
        [row for row in collections if row.get("collection_type") == "logical_sibling_subtable_review_collection"],
        COLLECTION_FIELDS,
    )
    base.write_csv(OUT_DIR / "v23_audit.csv", audit, AUDIT_FIELDS)
    summary_rows = [
        {"metric": key, "value": json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value}
        for key, value in summary.items()
    ]
    base.write_csv(OUT_DIR / "v23_summary.csv", summary_rows, SUMMARY_FIELDS)
    with (OUT_DIR / "v23_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    write_readme(summary)
    workbook = OUT_DIR / "v23_candidate_collections.xlsx"
    base.write_workbook(
        workbook,
        {
            "summary": (summary_rows, SUMMARY_FIELDS),
            "candidate_collections": (collections, COLLECTION_FIELDS),
            "members": (members, MEMBER_FIELDS),
            "edges": (edges, EDGE_FIELDS),
            "audit": (audit, AUDIT_FIELDS),
        },
    )
    with zipfile.ZipFile(workbook) as z:
        summary["v23_workbook_zip_bad_member"] = z.testzip()
    with (OUT_DIR / "v23_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
