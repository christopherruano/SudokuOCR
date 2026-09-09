#!/usr/bin/env python3
"""Build layout-aware, adjacency-first OCR page assemblies.

This extends the v5 structural assembler without doing OCR/API work. It reads
existing OCR/metadata, inspects local page images only for non-text layout
signals, and writes a new refined v6 output directory.

The added rule is deliberately narrow: when OCR row profiles are missing, a
physical join can still be made if adjacent pages look like a left/right
half-spread: previous page has a wide left row-label stub, current page's grid
starts at the far-left numeric columns, and title/subtable entities do not
switch. Religion/part/sex/entity switches remain logical collections, not
physical assemblies.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

import build_all_ocr_component_assembly_v5_structural as base


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "export_final/pdf_recovery_incoming_2026-06-16_all_ocr_component_assembly_v6_layout_strict"

LAYOUT_FIELDS = [
    "layout_loaded",
    "layout_error",
    "layout_scaled_width",
    "layout_scaled_height",
    "layout_vertical_line_count",
    "layout_first_vertical_line_x",
    "layout_second_vertical_line_x",
    "layout_initial_vertical_gap",
    "layout_left_stub_candidate",
    "layout_right_half_candidate",
]

PHYSICAL_PAGE_FIELDS = [*base.PHYSICAL_PAGE_FIELDS, *LAYOUT_FIELDS]
RUN_MEMBER_FIELDS = [
    "physical_run_id",
    "position_in_physical_run",
    *PHYSICAL_PAGE_FIELDS,
    "edge_from_prev_action",
    "edge_from_prev_reason",
    "edge_review_flags",
]
RUN_REVIEW_FIELDS = ["review_priority", "review_category", *base.RUN_FIELDS]

RELIGION_ALIASES = {
    "all religions": "all religions",
    "hindu": "hindu",
    "hindus": "hindu",
    "musalman": "musalman",
    "musalmans": "musalman",
    "muslim": "musalman",
    "muslims": "musalman",
    "muhammadan": "musalman",
    "muhammadans": "musalman",
    "christian": "christian",
    "christians": "christian",
    "jew": "jew",
    "jews": "jew",
    "animistic": "animistic",
    "parsi": "parsi",
    "parsis": "parsi",
    "jain": "jain",
    "jains": "jain",
    "sikh": "sikh",
    "sikhs": "sikh",
    "brahmo": "brahmo",
    "brahmos": "brahmo",
    "buddhist": "buddhist",
    "buddhists": "buddhist",
}
SEX_ALIASES = {
    "male": "male",
    "males": "male",
    "female": "female",
    "females": "female",
}

_ORIGINAL_EDGE_DECISION = base.edge_decision
_LAYOUT_CACHE: dict[str, dict[str, object]] = {}


def clean_for_entity(value: object) -> str:
    s = str(value or "").lower()
    s = re.sub(r"[\u2010-\u2015]", "-", s)
    s = re.sub(r"[^a-z0-9()]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def subtable_entities(row: dict[str, object]) -> dict[str, set[str]]:
    text = clean_for_entity(f"{row.get('title') or ''} {row.get('title_norm') or ''}")
    religions = {
        normalized
        for raw, normalized in RELIGION_ALIASES.items()
        if re.search(rf"\b{re.escape(raw)}\b", text)
    }
    sexes = {
        normalized
        for raw, normalized in SEX_ALIASES.items()
        if re.search(rf"\b{re.escape(raw)}\b", text)
    }
    parts = set(re.findall(r"\bpart\s*\(?\s*([a-z0-9]+)\s*\)?", text))
    parenthetical_numbers = set(re.findall(r"\(\s*(\d{1,3})\s*\)", str(row.get("title") or "")))
    return {
        "religion": religions,
        "sex": sexes,
        "part": parts,
        "paren_number": parenthetical_numbers,
    }


def physical_subtable_compatible(prev: dict[str, object], cur: dict[str, object]) -> tuple[bool, str]:
    prev_entities = subtable_entities(prev)
    cur_entities = subtable_entities(cur)
    switches: list[str] = []
    for key in ["religion", "part", "paren_number", "sex"]:
        if prev_entities[key] and cur_entities[key] and prev_entities[key] != cur_entities[key]:
            switches.append(f"{key}_switch_{sorted(prev_entities[key])}_to_{sorted(cur_entities[key])}")
    if switches:
        return False, "; ".join(switches)
    return True, "subtable_entities_compatible"


def core_money_like(prev: dict[str, object], cur: dict[str, object]) -> bool:
    raw_blob = base.clean_text(" ".join([
        str(prev.get("title") or ""),
        str(cur.get("title") or ""),
        str(prev.get("title_norm") or ""),
        str(cur.get("title_norm") or ""),
    ]))
    if any(
        excluded in raw_blob
        for excluded in [
            "selected caste",
            "selected castes",
            "selected castes or tribes",
            "selected race",
            "race tribe",
            "race tribes",
            "birth place",
            "birth places",
        ]
    ):
        return False
    if base.page_kind(prev) != base.page_kind(cur):
        return False
    if base.page_kind(cur) not in {"main_age_civil_numeric", "other_table", "structured_unknown"}:
        return False
    table_vii = (
        str(prev.get("table_number_key") or "") == "7"
        or str(cur.get("table_number_key") or "") == "7"
        or re.search(r"\btable\s+vii\b(?!i)", raw_blob) is not None
    )
    core_topic = (
        any(
            token in raw_blob
            for token in [
                "age civil",
                "age sex",
                "age and sex",
                "civil condition",
                "marital condition",
                "religion and age",
            ]
        )
        and any(token in raw_blob for token in ["age", "sex", "religion"])
    )
    return table_vii and core_topic


def image_layout_features(image_path: object) -> dict[str, object]:
    path = str(image_path or "").strip()
    if not path:
        return {"layout_loaded": "False", "layout_error": "missing_image_path"}
    if path in _LAYOUT_CACHE:
        return _LAYOUT_CACHE[path]

    try:
        full = Path(path)
        if not full.is_absolute():
            full = ROOT / full
        if not full.exists():
            out = {"layout_loaded": "False", "layout_error": "image_not_found"}
            _LAYOUT_CACHE[path] = out
            return out

        image = Image.open(full).convert("L")
        width, height = image.size
        scale = min(1.0, 1000 / max(width, height))
        if scale < 1.0:
            image = image.resize((max(1, int(width * scale)), max(1, int(height * scale))))
        arr = np.array(image)
        blurred = cv2.GaussianBlur(arr, (3, 3), 0)
        _, foreground = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        h, w = foreground.shape
        body = foreground[int(0.14 * h):int(0.86 * h), :]
        vertical_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (1, max(14, body.shape[0] // 20)),
        )
        vertical = cv2.morphologyEx(body, cv2.MORPH_OPEN, vertical_kernel)
        col_density = (vertical > 0).mean(axis=0)
        threshold = max(0.08, min(0.25, float(np.percentile(col_density, 99)) * 0.35))
        xs = np.where(col_density > threshold)[0]

        clusters: list[list[int]] = []
        if len(xs):
            current = [int(xs[0])]
            for x in xs[1:]:
                x = int(x)
                if x - current[-1] <= 3:
                    current.append(x)
                else:
                    clusters.append(current)
                    current = [x]
            clusters.append(current)
        positions = [(sum(cluster) / len(cluster)) / w for cluster in clusters]

        first = positions[0] if positions else None
        second = positions[1] if len(positions) > 1 else None
        initial_gap = second - first if first is not None and second is not None else None
        left_stub = bool(first is not None and first >= 0.14 and len(positions) >= 6)
        right_half = bool(first is not None and first <= 0.07 and len(positions) >= 6)

        out = {
            "layout_loaded": "True",
            "layout_error": "",
            "layout_scaled_width": w,
            "layout_scaled_height": h,
            "layout_vertical_line_count": len(positions),
            "layout_first_vertical_line_x": base.float_fmt(first),
            "layout_second_vertical_line_x": base.float_fmt(second),
            "layout_initial_vertical_gap": base.float_fmt(initial_gap),
            "layout_left_stub_candidate": "yes" if left_stub else "no",
            "layout_right_half_candidate": "yes" if right_half else "no",
        }
    except Exception as exc:
        out = {"layout_loaded": "False", "layout_error": f"{type(exc).__name__}: {exc}"}

    _LAYOUT_CACHE[path] = out
    return out


def ensure_layout(row: dict[str, object]) -> dict[str, object]:
    if row.get("layout_loaded") or row.get("layout_error"):
        return {field: row.get(field, "") for field in LAYOUT_FIELDS}
    features = image_layout_features(row.get("image_path"))
    for field in LAYOUT_FIELDS:
        row[field] = features.get(field, "")
    return features


def layout_required_halfspread(prev: dict[str, object], cur: dict[str, object]) -> tuple[bool, str, str]:
    prev_layout = ensure_layout(prev)
    cur_layout = ensure_layout(cur)
    if prev_layout.get("layout_loaded") != "True" or cur_layout.get("layout_loaded") != "True":
        return False, "layout_unavailable", "layout_unavailable_review"
    if prev_layout.get("layout_left_stub_candidate") != "yes":
        return False, "previous_page_no_wide_row_label_stub", "layout_not_required_join"
    if cur_layout.get("layout_right_half_candidate") != "yes":
        return False, "current_page_grid_not_far_left", "layout_not_required_join"

    subject_ok, subject_reason = base.table_subject_compatible(prev, cur)
    if not subject_ok:
        return False, f"layout_geometry_but_{subject_reason}", "layout_geometry_rejected_subject_switch"
    if not core_money_like(prev, cur):
        return False, "layout_geometry_but_not_core_table_vii_money_topic", "layout_geometry_rejected_non_core_topic"
    subtable_ok, subtable_reason = physical_subtable_compatible(prev, cur)
    if not subtable_ok:
        return False, f"layout_geometry_but_{subtable_reason}", "layout_geometry_rejected_subtable_switch"

    title_sim = base.text_similarity(prev.get("title_norm") or prev.get("title"), cur.get("title_norm") or cur.get("title"))
    if title_sim < 0.80:
        return False, f"layout_geometry_but_weak_title_similarity_{title_sim:.2f}", "layout_geometry_rejected_weak_title"

    reason = (
        "adjacent_image_layout_halfspread_requires_previous_row_labels; "
        f"prev_first_grid_x={prev_layout.get('layout_first_vertical_line_x')}; "
        f"cur_first_grid_x={cur_layout.get('layout_first_vertical_line_x')}; "
        f"title_similarity_{title_sim:.2f}"
    )
    flags = "image_layout_required_join; no_ocr_structural_profile_for_pair; layout_join_review"
    return True, reason, flags


def edge_decision(prev: dict[str, object], cur: dict[str, object]) -> dict[str, object]:
    edge = _ORIGINAL_EDGE_DECISION(prev, cur)
    if edge.get("edge_action") == "join":
        ensure_layout(prev)
        ensure_layout(cur)
        return edge

    reason = str(edge.get("edge_reason") or "")
    if not reason.startswith("adjacent_but_no_structural_profile"):
        return edge

    join, layout_reason, layout_flags = layout_required_halfspread(prev, cur)
    if join:
        return {
            "edge_action": "join",
            "edge_reason": layout_reason,
            "edge_review_flags": layout_flags,
            "page_gap": edge.get("page_gap", ""),
            "missing_pages": "",
            "schema_reason": edge.get("schema_reason", "missing_schema_profile"),
            "section_reason": edge.get("section_reason", "missing_section_profile"),
            "title_similarity": edge.get("title_similarity", ""),
        }

    if layout_flags.startswith("layout_geometry_rejected"):
        edge = dict(edge)
        edge["edge_reason"] = f"{reason}; {layout_reason}"
        edge["edge_review_flags"] = base.uniq_join([edge.get("edge_review_flags", ""), layout_flags], limit=20)
    return edge


def write_outputs(
    records: list[dict[str, object]],
    physical_pages: list[dict[str, object]],
    record_to_page: dict[str, str],
    runs: list[dict[str, object]],
    run_members: list[dict[str, object]],
    edges: list[dict[str, object]],
    missing_gap_candidates: list[dict[str, object]],
) -> None:
    record_rows = base.build_record_rows(records, record_to_page, physical_pages)
    overassociation = base.build_overassociation_report(records, record_rows)
    collection_groups, collection_members = base.build_collection_groups(runs, run_members, edges)

    required_run_ids = {
        str(r.get("physical_run_id"))
        for r in runs
        if int(r.get("n_physical_pages") or 0) > 1
    }
    required_assembly_groups = [r for r in runs if str(r.get("physical_run_id")) in required_run_ids]
    required_assembly_members = [m for m in run_members if str(m.get("physical_run_id")) in required_run_ids]
    required_assembly_edges = [e for e in edges if e.get("edge_action") == "join"]
    layout_required_edges = [
        e for e in required_assembly_edges if "image_layout_required_join" in str(e.get("edge_review_flags") or "")
    ]
    layout_rejected_edges = [
        e for e in edges if "layout_geometry_rejected" in str(e.get("edge_review_flags") or "")
    ]
    edge_review_queue = base.build_edge_review_queue(edges)
    run_review_queue = base.build_run_review_queue(runs)
    layout_rows = [
        {field: page.get(field, "") for field in ["physical_page_id", "v5_source_key", "pdf_page", "image_path", *LAYOUT_FIELDS]}
        for page in physical_pages
        if page.get("layout_loaded") or page.get("layout_error")
    ]

    summary = {
        "input_v4_records": len(records),
        "physical_pages": len(physical_pages),
        "records_with_existing_ocr_artifact": sum(1 for r in records if r.get("ocr_artifact")),
        "records_with_loaded_ocr_profile": sum(1 for r in records if str(r.get("profile_profile_loaded")) == "True"),
        "physical_runs": len(runs),
        "multi_page_physical_runs": sum(1 for r in runs if int(r.get("n_physical_pages") or 0) > 1),
        "max_physical_run_pages": max((int(r.get("n_physical_pages") or 0) for r in runs), default=0),
        "required_assembly_groups": len(required_assembly_groups),
        "required_assembly_edges": len(required_assembly_edges),
        "layout_required_assembly_edges": len(layout_required_edges),
        "layout_geometry_rejected_edges": len(layout_rejected_edges),
        "image_layout_pages_inspected": len(layout_rows),
        "join_edges": sum(1 for e in edges if e.get("edge_action") == "join"),
        "break_edges": sum(1 for e in edges if e.get("edge_action") == "break"),
        "edge_review_queue_rows": len(edge_review_queue),
        "run_review_queue_rows": len(run_review_queue),
        "missing_gap_candidates": len(missing_gap_candidates),
        "collection_groups": len(collection_groups),
        "multi_run_collection_groups": sum(1 for g in collection_groups if int(g.get("n_physical_runs") or 0) > 1),
        "v4_components_split_by_v6": sum(1 for r in overassociation if int(r.get("v5_physical_run_count") or 0) > 1),
        "largest_v4_split_run_count": max((int(r.get("v5_physical_run_count") or 0) for r in overassociation), default=0),
    }
    summary.update(base.run_invariants(runs, edges, run_members))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base.write_csv(OUT_DIR / "v6_page_records.csv", record_rows, base.RECORD_FIELDS)
    base.write_csv(OUT_DIR / "v6_physical_pages.csv", physical_pages, PHYSICAL_PAGE_FIELDS)
    base.write_csv(OUT_DIR / "v6_physical_edges.csv", edges, base.EDGE_FIELDS)
    base.write_csv(OUT_DIR / "v6_physical_runs.csv", runs, base.RUN_FIELDS)
    base.write_csv(OUT_DIR / "v6_run_members.csv", run_members, RUN_MEMBER_FIELDS)
    base.write_csv(OUT_DIR / "v6_required_assembly_groups.csv", required_assembly_groups, base.RUN_FIELDS)
    base.write_csv(OUT_DIR / "v6_required_assembly_members.csv", required_assembly_members, RUN_MEMBER_FIELDS)
    base.write_csv(OUT_DIR / "v6_required_assembly_edges.csv", required_assembly_edges, base.EDGE_FIELDS)
    base.write_csv(OUT_DIR / "v6_layout_required_edges.csv", layout_required_edges, base.EDGE_FIELDS)
    base.write_csv(OUT_DIR / "v6_layout_rejected_edges.csv", layout_rejected_edges, base.EDGE_FIELDS)
    base.write_csv(OUT_DIR / "v6_image_layout_features.csv", layout_rows, ["physical_page_id", "v5_source_key", "pdf_page", "image_path", *LAYOUT_FIELDS])
    base.write_csv(OUT_DIR / "v6_edge_review_queue.csv", edge_review_queue, base.EDGE_REVIEW_FIELDS)
    base.write_csv(OUT_DIR / "v6_run_review_queue.csv", run_review_queue, RUN_REVIEW_FIELDS)
    base.write_csv(OUT_DIR / "v6_missing_gap_candidates.csv", missing_gap_candidates, ["candidate_type", *base.EDGE_FIELDS])
    base.write_csv(OUT_DIR / "v6_collection_groups.csv", collection_groups, base.COLLECTION_FIELDS)
    base.write_csv(OUT_DIR / "v6_collection_members.csv", collection_members, base.COLLECTION_MEMBER_FIELDS)
    base.write_csv(OUT_DIR / "v6_overassociation_examples.csv", overassociation, base.OVERASSOC_FIELDS)

    with (OUT_DIR / "v6_layout_assembly_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    summary_rows = [{"metric": k, "value": json.dumps(v) if isinstance(v, (list, dict)) else v} for k, v in summary.items()]
    base.write_workbook(
        OUT_DIR / "v6_layout_assembly_review.xlsx",
        {
            "summary": (summary_rows, ["metric", "value"]),
            "physical_runs": (runs, base.RUN_FIELDS),
            "run_members": (run_members, RUN_MEMBER_FIELDS),
            "edges": (edges, base.EDGE_FIELDS),
            "required_groups": (required_assembly_groups, base.RUN_FIELDS),
            "required_members": (required_assembly_members, RUN_MEMBER_FIELDS),
            "required_edges": (required_assembly_edges, base.EDGE_FIELDS),
            "layout_required": (layout_required_edges, base.EDGE_FIELDS),
            "layout_rejected": (layout_rejected_edges, base.EDGE_FIELDS),
            "layout_features": (layout_rows, ["physical_page_id", "v5_source_key", "pdf_page", "image_path", *LAYOUT_FIELDS]),
            "edge_review_queue": (edge_review_queue, base.EDGE_REVIEW_FIELDS),
            "run_review_queue": (run_review_queue, RUN_REVIEW_FIELDS),
            "missing_gaps": (missing_gap_candidates, ["candidate_type", *base.EDGE_FIELDS]),
            "collections": (collection_groups, base.COLLECTION_FIELDS),
            "overassociation": (overassociation, base.OVERASSOC_FIELDS),
        },
    )

    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Wrote {OUT_DIR}")


def main() -> None:
    csv.field_size_limit(10 ** 7)
    raw_rows = base.read_csv(base.V4_PAGES)
    if not raw_rows:
        raise SystemExit(f"Missing v4 page inventory: {base.V4_PAGES}")

    base.edge_decision = edge_decision
    base.PHYSICAL_PAGE_FIELDS = PHYSICAL_PAGE_FIELDS
    base.RUN_MEMBER_FIELDS = RUN_MEMBER_FIELDS
    records = base.enrich_records(raw_rows)
    physical_pages, record_to_page, _page_to_records = base.build_physical_pages(records)
    runs, run_members, edges, missing_gap_candidates = base.build_runs(physical_pages)
    write_outputs(records, physical_pages, record_to_page, runs, run_members, edges, missing_gap_candidates)


if __name__ == "__main__":
    main()
