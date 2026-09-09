#!/usr/bin/env python3
"""Build strict physical components and full edge coverage from assembly frontier.

Components are allowed only when every adjacent edge in the run is already a
confirmed physical pair. This layer is intentionally conservative: it records
growth opportunities, but never grows across a missing, failed, candidate, or
logical-only edge.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import zipfile
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Iterable

import xlsxwriter


ROOT = Path(__file__).resolve().parent
DEFAULT_ASSEMBLY_DIR = ROOT / "export_final/pdf_recovery_incoming_2026-06-16_first_principles_assembly_pipeline_v29_logical_review_context"
DEFAULT_FRONTIER_DIR = ROOT / "export_final/pdf_recovery_incoming_2026-06-16_first_principles_assembly_frontier_v30_joined"
DEFAULT_OUT_DIR = ROOT / "export_final/pdf_recovery_incoming_2026-06-16_first_principles_assembly_components_v31"

COMPONENT_FIELDS = [
    "component_rank",
    "component_id",
    "component_status",
    "action_bucket",
    "safe_to_physical_merge",
    "safe_to_publish_values",
    "value_review_required",
    "source_pdf",
    "source_pdf_id",
    "v5_source_key",
    "year",
    "province",
    "page_range",
    "component_pages",
    "n_pages",
    "n_confirmed_edges",
    "all_pages_consecutive",
    "all_internal_edges_confirmed",
    "edge_pairs",
    "edge_frontier_ranks",
    "edge_attempt_ids",
    "assembled_row_count_total",
    "edge_value_statuses",
    "review_flags",
    "component_reason",
    "recommended_action",
]

MEMBER_FIELDS = [
    "component_id",
    "member_rank",
    "source_pdf",
    "source_pdf_id",
    "v5_source_key",
    "pdf_page",
    "page_role",
    "prev_confirmed_edge_pair",
    "next_confirmed_edge_pair",
    "image_path",
    "ocr_artifact",
]

GROWTH_FIELDS = [
    "component_id",
    "direction",
    "source_pdf",
    "source_pdf_id",
    "v5_source_key",
    "component_page_range",
    "candidate_page_pair",
    "candidate_prev_pdf_page",
    "candidate_pdf_page",
    "edge_record_status",
    "adjacency_final_decision_layer",
    "frontier_status",
    "action_bucket",
    "safe_to_physical_merge",
    "safe_to_logical_group",
    "growth_allowed_now",
    "growth_action",
    "decision_reason",
    "review_flags",
    "left_image_path",
    "right_image_path",
]

EDGE_FIELDS = [
    "coverage_rank",
    "edge_key",
    "coverage_scope",
    "source_pdf",
    "source_pdf_id",
    "v5_source_key",
    "prev_pdf_page",
    "pdf_page",
    "page_gap",
    "pages_consecutive",
    "adjacency_final_decision_layer",
    "adjacency_safe_to_physical_merge",
    "adjacency_safe_to_logical_group",
    "frontier_layers",
    "frontier_statuses",
    "frontier_action_buckets",
    "component_ids",
    "covered_by_confirmed_component",
    "can_be_physical_edge_now",
    "can_be_logical_group_now",
    "decision_reason",
    "decision_review_flags",
    "prev_page_kind",
    "page_kind",
    "prev_table_family_key",
    "table_family_key",
    "prev_table_number_key",
    "table_number_key",
    "row_label_scan_strength",
    "row_label_scan_pattern",
    "prev_row_label_ratio",
    "row_label_ratio",
    "prev_title",
    "title",
    "prev_image_path",
    "image_path",
]

SUMMARY_FIELDS = ["metric", "name", "count"]
AUDIT_FIELDS = ["check", "status", "detail"]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def output_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def stable_id(prefix: str, *parts: object) -> str:
    payload = "||".join(str(part) for part in parts)
    return f"{prefix}_{hashlib.sha1(payload.encode('utf-8')).hexdigest()[:16]}"


def int_or_none(value: object) -> int | None:
    if value in {None, ""}:
        return None
    try:
        return int(str(value))
    except ValueError:
        return None


def semi(values: Iterable[object]) -> str:
    return "; ".join(str(value) for value in values if value not in {None, ""})


def source_key(row: dict[str, object]) -> str:
    return str(row.get("v5_source_key") or row.get("source_pdf") or row.get("source_pdf_id") or "")


def edge_key(row: dict[str, object]) -> tuple[str, str, str]:
    return (source_key(row), str(row.get("prev_pdf_page") or ""), str(row.get("pdf_page") or ""))


def page_pair(left: object, right: object) -> str:
    if left in {None, ""} or right in {None, ""}:
        return ""
    return f"{left}-{right}"


def pages_are_consecutive(pages: list[int]) -> bool:
    return bool(pages) and pages == list(range(pages[0], pages[-1] + 1))


def sum_int(rows: Iterable[dict[str, object]], field: str) -> int:
    total = 0
    for row in rows:
        value = int_or_none(row.get(field))
        if value is not None:
            total += value
    return total


def build_frontier_maps(frontier_dir: Path) -> tuple[list[dict[str, str]], dict[tuple[str, str, str], list[dict[str, str]]]]:
    physical = read_csv(frontier_dir / "physical_assembly_frontier.csv")
    by_edge: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in physical:
        key = edge_key(row)
        if all(key):
            by_edge[key].append(row)
    return physical, by_edge


def confirmed_edges(physical_frontier: list[dict[str, str]]) -> list[dict[str, str]]:
    return [
        row for row in physical_frontier
        if row.get("frontier_status") == "confirmed_physical_pair"
        and row.get("safe_to_physical_merge") == "yes"
        and row.get("pages_consecutive") == "yes"
    ]


def page_image_artifact_maps(edge_rows: list[dict[str, str]]) -> tuple[dict[tuple[str, int], str], dict[tuple[str, int], str]]:
    image_by_page: dict[tuple[str, int], str] = {}
    artifact_by_page: dict[tuple[str, int], str] = {}
    for row in edge_rows:
        src = source_key(row)
        left = int_or_none(row.get("prev_pdf_page"))
        right = int_or_none(row.get("pdf_page"))
        if left is not None:
            image_by_page[(src, left)] = row.get("left_image_path", "")
            artifact_by_page[(src, left)] = row.get("left_ocr_artifact", "")
        if right is not None:
            image_by_page[(src, right)] = row.get("right_image_path", "")
            artifact_by_page[(src, right)] = row.get("right_ocr_artifact", "")
    return image_by_page, artifact_by_page


def component_rows_from_confirmed_edges(
    edge_rows: list[dict[str, str]],
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[tuple[str, str, str], str]]:
    graph: dict[str, dict[int, set[int]]] = defaultdict(lambda: defaultdict(set))
    row_by_edge: dict[tuple[str, int, int], dict[str, str]] = {}
    image_by_page, artifact_by_page = page_image_artifact_maps(edge_rows)
    for row in edge_rows:
        src = source_key(row)
        left = int_or_none(row.get("prev_pdf_page"))
        right = int_or_none(row.get("pdf_page"))
        if not src or left is None or right is None:
            continue
        graph[src][left].add(right)
        graph[src][right].add(left)
        row_by_edge[(src, min(left, right), max(left, right))] = row

    components: list[dict[str, object]] = []
    members: list[dict[str, object]] = []
    edge_to_component: dict[tuple[str, str, str], str] = {}
    for src in sorted(graph):
        seen: set[int] = set()
        for start in sorted(graph[src]):
            if start in seen:
                continue
            queue: deque[int] = deque([start])
            seen.add(start)
            pages: list[int] = []
            while queue:
                page = queue.popleft()
                pages.append(page)
                for nxt in sorted(graph[src][page]):
                    if nxt not in seen:
                        seen.add(nxt)
                        queue.append(nxt)
            pages.sort()
            internal_pairs = list(zip(pages, pages[1:]))
            edge_records = [row_by_edge[(src, left, right)] for left, right in internal_pairs if (src, left, right) in row_by_edge]
            component_id = stable_id("v31phys", src, pages[0], pages[-1], ",".join(map(str, pages)))
            all_internal_confirmed = len(edge_records) == max(0, len(pages) - 1)
            all_consecutive = pages_are_consecutive(pages)
            publish_values = "yes" if edge_records and all(row.get("safe_to_publish_values") == "yes" for row in edge_records) else "no"
            value_review = "yes" if any(row.get("review_required") == "yes" or row.get("safe_to_publish_values") != "yes" for row in edge_records) else "no"
            first = edge_records[0] if edge_records else {}
            for left, right in internal_pairs:
                edge_to_component[(src, str(left), str(right))] = component_id
            components.append(
                {
                    "component_id": component_id,
                    "component_status": "confirmed_physical_component",
                    "action_bucket": "use_now_publishable_physical_component" if publish_values == "yes" else "use_now_structure_review_values",
                    "safe_to_physical_merge": "yes" if all_consecutive and all_internal_confirmed else "no",
                    "safe_to_publish_values": publish_values,
                    "value_review_required": value_review,
                    "source_pdf": first.get("source_pdf", ""),
                    "source_pdf_id": first.get("source_pdf_id", ""),
                    "v5_source_key": src,
                    "year": first.get("year", ""),
                    "province": first.get("province", ""),
                    "page_range": f"{pages[0]}-{pages[-1]}",
                    "component_pages": semi(pages),
                    "n_pages": len(pages),
                    "n_confirmed_edges": len(edge_records),
                    "all_pages_consecutive": "yes" if all_consecutive else "no",
                    "all_internal_edges_confirmed": "yes" if all_internal_confirmed else "no",
                    "edge_pairs": semi(page_pair(left, right) for left, right in internal_pairs),
                    "edge_frontier_ranks": semi(row.get("frontier_rank", "") for row in edge_records),
                    "edge_attempt_ids": semi(row.get("v26_attempt_id", "") for row in edge_records),
                    "assembled_row_count_total": sum_int(edge_records, "assembled_row_count"),
                    "edge_value_statuses": semi(sorted({row.get("value_constraint_status", "") for row in edge_records})),
                    "review_flags": semi(sorted({flag.strip() for row in edge_records for flag in str(row.get("review_flags", "")).split(";") if flag.strip()})),
                    "component_reason": "Every internal adjacent edge in this run is a confirmed consecutive physical pair.",
                    "recommended_action": "Use as a physical component; publish numeric values only when value review is clear." if value_review == "yes" else "Use as publishable physical component.",
                }
            )
            for idx, page in enumerate(pages, start=1):
                role = "interior"
                if len(pages) == 1:
                    role = "single"
                elif idx == 1:
                    role = "first"
                elif idx == len(pages):
                    role = "last"
                members.append(
                    {
                        "component_id": component_id,
                        "member_rank": idx,
                        "source_pdf": first.get("source_pdf", ""),
                        "source_pdf_id": first.get("source_pdf_id", ""),
                        "v5_source_key": src,
                        "pdf_page": page,
                        "page_role": role,
                        "prev_confirmed_edge_pair": page_pair(page - 1, page) if (src, page - 1, page) in row_by_edge else "",
                        "next_confirmed_edge_pair": page_pair(page, page + 1) if (src, page, page + 1) in row_by_edge else "",
                        "image_path": image_by_page.get((src, page), ""),
                        "ocr_artifact": artifact_by_page.get((src, page), ""),
                    }
                )
    components.sort(key=lambda row: (str(row.get("v5_source_key")), int(str(row.get("page_range")).split("-", 1)[0])))
    for idx, row in enumerate(components, start=1):
        row["component_rank"] = idx
    return components, members, edge_to_component


def build_edge_coverage(
    adjacency_rows: list[dict[str, str]],
    frontier_by_edge: dict[tuple[str, str, str], list[dict[str, str]]],
    edge_to_component: dict[tuple[str, str, str], str],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for adj in adjacency_rows:
        key = edge_key(adj)
        frontier_rows = frontier_by_edge.get(key, [])
        component_id = edge_to_component.get(key, "")
        frontier_layers = sorted({row.get("frontier_layer", "") for row in frontier_rows if row.get("frontier_layer")})
        frontier_statuses = sorted({row.get("frontier_status", "") for row in frontier_rows if row.get("frontier_status")})
        action_buckets = sorted({row.get("action_bucket", "") for row in frontier_rows if row.get("action_bucket")})
        rows.append(
            {
                "edge_key": "||".join(key),
                "coverage_scope": adj.get("coverage_scope", ""),
                "source_pdf": "",
                "source_pdf_id": "",
                "v5_source_key": adj.get("v5_source_key", ""),
                "prev_pdf_page": adj.get("prev_pdf_page", ""),
                "pdf_page": adj.get("pdf_page", ""),
                "page_gap": adj.get("page_gap", ""),
                "pages_consecutive": "yes" if adj.get("page_gap") == "1" else "no",
                "adjacency_final_decision_layer": adj.get("final_decision_layer", ""),
                "adjacency_safe_to_physical_merge": adj.get("safe_to_physical_merge", ""),
                "adjacency_safe_to_logical_group": adj.get("safe_to_logical_group", ""),
                "frontier_layers": semi(frontier_layers),
                "frontier_statuses": semi(frontier_statuses),
                "frontier_action_buckets": semi(action_buckets),
                "component_ids": component_id,
                "covered_by_confirmed_component": "yes" if component_id else "no",
                "can_be_physical_edge_now": adj.get("safe_to_physical_merge", ""),
                "can_be_logical_group_now": adj.get("safe_to_logical_group", ""),
                "decision_reason": adj.get("decision_reason", ""),
                "decision_review_flags": adj.get("decision_review_flags", ""),
                "prev_page_kind": adj.get("prev_page_kind", ""),
                "page_kind": adj.get("page_kind", ""),
                "prev_table_family_key": adj.get("prev_table_family_key", ""),
                "table_family_key": adj.get("table_family_key", ""),
                "prev_table_number_key": adj.get("prev_table_number_key", ""),
                "table_number_key": adj.get("table_number_key", ""),
                "row_label_scan_strength": adj.get("row_label_scan_strength", ""),
                "row_label_scan_pattern": adj.get("row_label_scan_pattern", ""),
                "prev_row_label_ratio": adj.get("prev_row_label_ratio", ""),
                "row_label_ratio": adj.get("row_label_ratio", ""),
                "prev_title": adj.get("prev_title", ""),
                "title": adj.get("title", ""),
                "prev_image_path": adj.get("prev_image_path", ""),
                "image_path": adj.get("image_path", ""),
            }
        )
    rows.sort(
        key=lambda row: (
            str(row.get("v5_source_key", "")),
            int_or_none(row.get("prev_pdf_page")) or 10**9,
            int_or_none(row.get("pdf_page")) or 10**9,
        )
    )
    for idx, row in enumerate(rows, start=1):
        row["coverage_rank"] = idx
    return rows


def growth_action(edge: dict[str, object]) -> str:
    layer = str(edge.get("adjacency_final_decision_layer") or "")
    action = str(edge.get("frontier_action_buckets") or "")
    if edge.get("safe_to_physical_merge") == "yes" or edge.get("adjacency_safe_to_physical_merge") == "yes":
        return "audit_should_already_be_internal_confirmed_edge"
    if "ocr_existing_render_then_retest" in action:
        return "ocr_existing_render_then_retest_before_growth"
    if "obtain_render_or_artifact_then_retest" in action:
        return "obtain_render_or_artifact_before_growth"
    if layer in {"failed_physical_constraints_review", "partial_physical_pair_review"}:
        return "do_not_grow_until_constraints_repaired"
    if layer == "row_label_dependency_review_candidate":
        return "candidate_only_review_before_growth"
    if layer == "blocked_physical_missing_ocr_artifact":
        return "ocr_or_attach_artifact_before_growth"
    return "do_not_grow_hard_break_or_not_candidate"


def build_growth_frontier(
    components: list[dict[str, object]],
    edge_coverage: list[dict[str, object]],
) -> list[dict[str, object]]:
    coverage_by_key = {
        (str(row.get("v5_source_key", "")), str(row.get("prev_pdf_page", "")), str(row.get("pdf_page", ""))): row
        for row in edge_coverage
    }
    rows: list[dict[str, object]] = []
    for comp in components:
        src = str(comp.get("v5_source_key", ""))
        pages = [int(part) for part in str(comp.get("component_pages", "")).split("; ") if part]
        if not pages:
            continue
        candidates = [
            ("previous", pages[0] - 1, pages[0]),
            ("next", pages[-1], pages[-1] + 1),
        ]
        for direction, left, right in candidates:
            edge = coverage_by_key.get((src, str(left), str(right)), {})
            record_status = "edge_record_found" if edge else "no_adjacent_edge_record"
            action = growth_action(edge) if edge else "no_record_do_not_grow"
            rows.append(
                {
                    "component_id": comp.get("component_id", ""),
                    "direction": direction,
                    "source_pdf": comp.get("source_pdf", ""),
                    "source_pdf_id": comp.get("source_pdf_id", ""),
                    "v5_source_key": src,
                    "component_page_range": comp.get("page_range", ""),
                    "candidate_page_pair": page_pair(left, right),
                    "candidate_prev_pdf_page": left,
                    "candidate_pdf_page": right,
                    "edge_record_status": record_status,
                    "adjacency_final_decision_layer": edge.get("adjacency_final_decision_layer", ""),
                    "frontier_status": edge.get("frontier_statuses", ""),
                    "action_bucket": edge.get("frontier_action_buckets", ""),
                    "safe_to_physical_merge": edge.get("adjacency_safe_to_physical_merge", "no"),
                    "safe_to_logical_group": edge.get("adjacency_safe_to_logical_group", "no"),
                    "growth_allowed_now": "no",
                    "growth_action": action,
                    "decision_reason": edge.get("decision_reason", ""),
                    "review_flags": edge.get("decision_review_flags", ""),
                    "left_image_path": edge.get("prev_image_path", ""),
                    "right_image_path": edge.get("image_path", ""),
                }
            )
    rows.sort(key=lambda row: (str(row.get("component_id")), str(row.get("direction"))))
    return rows


def build_summary(
    components: list[dict[str, object]],
    members: list[dict[str, object]],
    growth: list[dict[str, object]],
    edge_coverage: list[dict[str, object]],
    assembly_dir: Path,
    frontier_dir: Path,
    out_dir: Path,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = [
        {"metric": "paths", "name": "assembly_dir", "count": output_path(assembly_dir)},
        {"metric": "paths", "name": "frontier_dir", "count": output_path(frontier_dir)},
        {"metric": "paths", "name": "out_dir", "count": output_path(out_dir)},
        {"metric": "counts", "name": "confirmed_physical_components", "count": len(components)},
        {"metric": "counts", "name": "confirmed_physical_component_members", "count": len(members)},
        {"metric": "counts", "name": "component_growth_frontier_rows", "count": len(growth)},
        {"metric": "counts", "name": "consecutive_edge_coverage_rows", "count": len(edge_coverage)},
    ]
    for metric, counter in [
        ("components_by_action", Counter(str(row.get("action_bucket", "")) for row in components)),
        ("components_by_n_pages", Counter(str(row.get("n_pages", "")) for row in components)),
        ("growth_by_action", Counter(str(row.get("growth_action", "")) for row in growth)),
        ("edge_coverage_by_layer", Counter(str(row.get("adjacency_final_decision_layer", "")) for row in edge_coverage)),
        ("edge_coverage_by_component", Counter(str(row.get("covered_by_confirmed_component", "")) for row in edge_coverage)),
    ]:
        for name, count in sorted(counter.items()):
            rows.append({"metric": metric, "name": name, "count": count})
    return rows


def audit_outputs(
    components: list[dict[str, object]],
    growth: list[dict[str, object]],
    edge_coverage: list[dict[str, object]],
    adjacency_count: int,
) -> list[dict[str, object]]:
    audits: list[dict[str, object]] = []
    bad_components = [
        row for row in components
        if row.get("all_pages_consecutive") != "yes" or row.get("all_internal_edges_confirmed") != "yes"
    ]
    audits.append(
        {
            "check": "components_are_contiguous_confirmed_edge_runs",
            "status": "pass" if not bad_components else "fail",
            "detail": f"{len(bad_components)} components are not contiguous confirmed runs",
        }
    )
    safe_edges = [row for row in edge_coverage if row.get("adjacency_safe_to_physical_merge") == "yes"]
    uncovered_safe = [row for row in safe_edges if row.get("covered_by_confirmed_component") != "yes"]
    audits.append(
        {
            "check": "all_safe_physical_edges_are_in_components",
            "status": "pass" if not uncovered_safe else "fail",
            "detail": f"{len(uncovered_safe)} safe physical edges are not represented in components",
        }
    )
    unsafe_covered = [
        row for row in edge_coverage
        if row.get("covered_by_confirmed_component") == "yes" and row.get("adjacency_safe_to_physical_merge") != "yes"
    ]
    audits.append(
        {
            "check": "components_do_not_include_unsafe_edges",
            "status": "pass" if not unsafe_covered else "fail",
            "detail": f"{len(unsafe_covered)} unsafe edges are included in components",
        }
    )
    audits.append(
        {
            "check": "edge_coverage_covers_all_adjacency_rows",
            "status": "pass" if len(edge_coverage) == adjacency_count else "fail",
            "detail": f"edge_coverage={len(edge_coverage)} adjacency={adjacency_count}",
        }
    )
    bad_growth = [row for row in growth if row.get("growth_allowed_now") == "yes"]
    audits.append(
        {
            "check": "growth_frontier_never_auto_grows",
            "status": "pass" if not bad_growth else "fail",
            "detail": f"{len(bad_growth)} growth rows allow automatic growth",
        }
    )
    return audits


def write_workbook(
    path: Path,
    summary: list[dict[str, object]],
    components: list[dict[str, object]],
    members: list[dict[str, object]],
    growth: list[dict[str, object]],
    edge_coverage: list[dict[str, object]],
    audit: list[dict[str, object]],
) -> None:
    workbook = xlsxwriter.Workbook(path)
    header_fmt = workbook.add_format({"bold": True, "bg_color": "#E8EEF7", "border": 1})
    wrap_fmt = workbook.add_format({"text_wrap": True, "valign": "top"})
    sheets = [
        ("Summary", summary, SUMMARY_FIELDS),
        ("Physical Components", components, COMPONENT_FIELDS),
        ("Component Members", members, MEMBER_FIELDS),
        ("Growth Frontier", growth, GROWTH_FIELDS),
        ("Edge Coverage", edge_coverage, EDGE_FIELDS),
        ("Audit", audit, AUDIT_FIELDS),
    ]
    for name, rows, fields in sheets:
        ws = workbook.add_worksheet(name[:31])
        for col, field in enumerate(fields):
            ws.write(0, col, field, header_fmt)
        for row_idx, row in enumerate(rows, start=1):
            for col, field in enumerate(fields):
                value = row.get(field, "")
                fmt = wrap_fmt if any(token in field for token in ["reason", "flags", "path", "title", "action"]) else None
                ws.write(row_idx, col, value, fmt)
        ws.freeze_panes(1, 0)
        ws.autofilter(0, 0, max(len(rows), 1), max(len(fields) - 1, 0))
        for col, field in enumerate(fields):
            ws.set_column(col, col, min(max(len(field) + 2, 12), 56))
    workbook.close()


def validate_workbook(path: Path) -> str | None:
    try:
        with zipfile.ZipFile(path) as zf:
            return zf.testzip()
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"


def build_components(assembly_dir: Path, frontier_dir: Path, out_dir: Path) -> dict[str, object]:
    adjacency_rows = read_csv(assembly_dir / "adjacency_decision_index.csv")
    physical_frontier, frontier_by_edge = build_frontier_maps(frontier_dir)
    confirmed = confirmed_edges(physical_frontier)
    components, members, edge_to_component = component_rows_from_confirmed_edges(confirmed)
    edge_coverage = build_edge_coverage(adjacency_rows, frontier_by_edge, edge_to_component)
    growth = build_growth_frontier(components, edge_coverage)
    summary = build_summary(components, members, growth, edge_coverage, assembly_dir, frontier_dir, out_dir)
    audit = audit_outputs(components, growth, edge_coverage, len(adjacency_rows))

    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "confirmed_physical_components.csv", components, COMPONENT_FIELDS)
    write_csv(out_dir / "confirmed_physical_component_members.csv", members, MEMBER_FIELDS)
    write_csv(out_dir / "component_growth_frontier.csv", growth, GROWTH_FIELDS)
    write_csv(out_dir / "consecutive_edge_coverage.csv", edge_coverage, EDGE_FIELDS)
    write_csv(out_dir / "summary.csv", summary, SUMMARY_FIELDS)
    workbook_path = out_dir / "assembly_components_workbook.xlsx"
    workbook_bad_member = validate_workbook(workbook_path)
    audit.append(
        {
            "check": "assembly_components_workbook_zip_integrity",
            "status": "pass" if workbook_bad_member is None else "fail",
            "detail": workbook_bad_member or "",
        }
    )
    write_csv(out_dir / "audit.csv", audit, AUDIT_FIELDS)
    write_workbook(workbook_path, summary, components, members, growth, edge_coverage, audit)
    workbook_bad_member = validate_workbook(workbook_path)
    audit[-1] = {
        "check": "assembly_components_workbook_zip_integrity",
        "status": "pass" if workbook_bad_member is None else "fail",
        "detail": workbook_bad_member or "",
    }
    write_csv(out_dir / "audit.csv", audit, AUDIT_FIELDS)
    write_workbook(workbook_path, summary, components, members, growth, edge_coverage, audit)
    audit_failure_count = sum(1 for row in audit if row.get("status") != "pass")
    result = {
        "assembly_dir": output_path(assembly_dir),
        "frontier_dir": output_path(frontier_dir),
        "out_dir": output_path(out_dir),
        "confirmed_physical_components": len(components),
        "confirmed_physical_component_members": len(members),
        "component_growth_frontier_rows": len(growth),
        "consecutive_edge_coverage_rows": len(edge_coverage),
        "components_by_n_pages": dict(Counter(str(row.get("n_pages", "")) for row in components)),
        "components_by_action": dict(Counter(str(row.get("action_bucket", "")) for row in components)),
        "growth_by_action": dict(Counter(str(row.get("growth_action", "")) for row in growth)),
        "edge_coverage_by_layer": dict(Counter(str(row.get("adjacency_final_decision_layer", "")) for row in edge_coverage)),
        "audit_failure_count": audit_failure_count,
        "workbook_zip_bad_member": workbook_bad_member,
    }
    (out_dir / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    if audit_failure_count:
        raise SystemExit(f"component audit failed: {audit_failure_count} failures")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assembly-dir", default=str(DEFAULT_ASSEMBLY_DIR))
    parser.add_argument("--frontier-dir", default=str(DEFAULT_FRONTIER_DIR))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    args = parser.parse_args()
    assembly_dir = Path(args.assembly_dir).expanduser()
    frontier_dir = Path(args.frontier_dir).expanduser()
    out_dir = Path(args.out_dir).expanduser()
    if not assembly_dir.is_absolute():
        assembly_dir = ROOT / assembly_dir
    if not frontier_dir.is_absolute():
        frontier_dir = ROOT / frontier_dir
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    result = build_components(assembly_dir, frontier_dir, out_dir)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
