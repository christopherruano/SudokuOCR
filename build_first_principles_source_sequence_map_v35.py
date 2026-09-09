#!/usr/bin/env python3
"""Build source-ordered page/edge/run maps from the assembly rulebook.

v34 answers "what is the decision for this edge?" This v35 layer answers
"where does that edge sit in its source PDF, and what larger run would be
formed if every internal edge eventually promotes?"

It does not OCR and does not promote candidates. Runs are interpretability and
planning aids; physical merge remains allowed only where v34/v31 already allow
it.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import xlsxwriter


ROOT = Path(__file__).resolve().parent
DEFAULT_RULEBOOK_DIR = ROOT / "export_final/pdf_recovery_incoming_2026-06-16_first_principles_assembly_rulebook_v34"
DEFAULT_OUT_DIR = ROOT / "export_final/pdf_recovery_incoming_2026-06-16_first_principles_source_sequence_map_v35"


RUN_FIELDS = [
    "run_rank",
    "run_id",
    "run_type",
    "run_status",
    "physical_merge_allowed_now",
    "logical_collection_only",
    "review_required",
    "v5_source_key",
    "source_pdf_id_guess",
    "page_range",
    "component_pages",
    "n_pages",
    "edge_pairs",
    "n_edges",
    "edge_statuses",
    "site_actions",
    "row_label_dependency_edge_count",
    "missing_page_count",
    "missing_pages",
    "classifier_break_count",
    "table_family_keys",
    "table_number_keys",
    "representative_title",
    "run_flags",
    "recommended_action",
]

SOURCE_FIELDS = [
    "source_rank",
    "v5_source_key",
    "source_pdf_id_guess",
    "observed_page_count",
    "observed_page_min",
    "observed_page_max",
    "observed_page_span",
    "internal_observed_page_gaps",
    "confirmed_physical_run_count",
    "confirmed_physical_page_count",
    "physical_candidate_run_count",
    "physical_candidate_edge_count",
    "row_label_dependency_edge_count",
    "missing_bridge_edge_count",
    "logical_only_run_count",
    "hard_break_edge_count",
    "first_priority_queue_type",
    "first_priority_page_pair",
    "source_flags",
]

PAGE_FIELDS = [
    "sequence_rank",
    "v5_source_key",
    "source_pdf_id_guess",
    "pdf_page",
    "page_parity",
    "page_kind",
    "page_class",
    "no_row_labels_signal",
    "row_label_ratio",
    "prev_edge_status",
    "prev_edge_pair",
    "prev_edge_run_type",
    "prev_edge_site_action",
    "next_edge_status",
    "next_edge_pair",
    "next_edge_run_type",
    "next_edge_site_action",
    "current_physical_component_ids",
    "candidate_group_ids",
    "logical_group_ids",
    "title",
    "image_path",
]

EDGE_SEQUENCE_FIELDS = [
    "sequence_rank",
    "run_id",
    "run_type",
    "v5_source_key",
    "source_pdf_id_guess",
    "page_pair",
    "prev_pdf_page",
    "pdf_page",
    "page_gap",
    "physical_rule_status",
    "physical_merge_allowed_now",
    "logical_group_allowed_now",
    "site_action",
    "row_label_dependency_signal",
    "classifier_break_signal",
    "same_table_family",
    "same_table_number",
    "rule_flags",
    "decision_reason",
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


def split_semicolon(value: object) -> list[str]:
    return [part.strip() for part in str(value or "").split(";") if part.strip()]


def uniq(values: Iterable[object]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    return out


def page_pair(left: object, right: object) -> str:
    if left in {None, ""} or right in {None, ""}:
        return ""
    return f"{left}-{right}"


def parse_pages(value: object) -> list[int]:
    pages: list[int] = []
    for part in re.split(r"[;,]\s*|\s+", str(value or "").strip()):
        page = int_or_none(part)
        if page is not None:
            pages.append(page)
    return pages


def missing_pages(left: object, right: object) -> list[int]:
    left_i = int_or_none(left)
    right_i = int_or_none(right)
    if left_i is None or right_i is None or right_i <= left_i + 1:
        return []
    return list(range(left_i + 1, right_i))


def edge_run_type(edge: dict[str, str]) -> str:
    status = edge.get("physical_rule_status", "")
    if status == "confirmed_physical_edge":
        return "confirmed_physical"
    if status in {
        "candidate_row_label_dependency",
        "candidate_physical_missing_ocr",
        "unpromoted_row_label_dependency_signal",
        "failed_or_partial_physical_review",
    }:
        return "physical_candidate"
    if status == "likely_assemblable_after_missing_pages":
        return "missing_page_bridge_candidate"
    if status in {"confirmed_logical_only_edge", "review_logical_only_edge"}:
        return "logical_only"
    return "hard_break_or_other"


def edge_sort_key(edge: dict[str, str]) -> tuple[str, int, int, str]:
    return (
        edge.get("v5_source_key", ""),
        int_or_none(edge.get("prev_pdf_page")) or 10**9,
        int_or_none(edge.get("pdf_page")) or 10**9,
        edge.get("rule_id", ""),
    )


def source_id_guess(rows: Iterable[dict[str, object]]) -> str:
    for row in rows:
        value = str(row.get("source_pdf_id_guess") or "")
        if value:
            return value
    return ""


def build_runs(edges: list[dict[str, str]]) -> tuple[list[dict[str, object]], dict[str, str], list[dict[str, object]]]:
    edges_by_source: dict[str, list[dict[str, str]]] = defaultdict(list)
    for edge in edges:
        edges_by_source[edge.get("v5_source_key", "")].append(edge)

    runs: list[dict[str, object]] = []
    edge_to_run: dict[str, str] = {}
    edge_sequence: list[dict[str, object]] = []

    for src in sorted(edges_by_source):
        source_edges = sorted(edges_by_source[src], key=edge_sort_key)
        current: list[dict[str, str]] = []
        current_type = ""
        previous_right: int | None = None

        def flush() -> None:
            nonlocal current, current_type, previous_right
            if not current:
                return
            run_id = stable_id(
                "v35run",
                src,
                current_type,
                current[0].get("prev_pdf_page", ""),
                current[-1].get("pdf_page", ""),
                semi(edge.get("rule_id", "") for edge in current),
            )
            pages: list[int] = []
            for edge in current:
                left = int_or_none(edge.get("prev_pdf_page"))
                right = int_or_none(edge.get("pdf_page"))
                if left is not None:
                    pages.append(left)
                for missing in missing_pages(edge.get("prev_pdf_page"), edge.get("pdf_page")):
                    pages.append(missing)
                if right is not None:
                    pages.append(right)
            pages = sorted(set(pages))
            missing_all = sorted(set(page for edge in current for page in missing_pages(edge.get("prev_pdf_page"), edge.get("pdf_page"))))
            statuses = uniq(edge.get("physical_rule_status", "") for edge in current)
            actions = uniq(edge.get("site_action", "") for edge in current)
            classifier_break_count = sum(1 for edge in current if edge.get("classifier_break_signal") == "yes")
            row_label_count = sum(1 for edge in current if edge.get("row_label_dependency_signal") == "yes")
            physical_allowed = "yes" if current_type == "confirmed_physical" and all(edge.get("physical_merge_allowed_now") == "yes" for edge in current) else "no"
            logical_only = "yes" if current_type == "logical_only" else "no"
            review_required = "no" if current_type == "confirmed_physical" and physical_allowed == "yes" else "yes"
            if current_type == "confirmed_physical":
                recommended = "Use as a confirmed physical run only if every internal edge remains confirmed."
                status = "confirmed"
            elif current_type == "physical_candidate":
                recommended = "Retest/promote edge-by-edge; if every internal edge confirms later, rebuild components."
                status = "candidate_review"
            elif current_type == "missing_page_bridge_candidate":
                recommended = "Render/OCR all listed missing pages, then retest adjacent edges only."
                status = "blocked_missing_pages"
            elif current_type == "logical_only":
                recommended = "Use/review as logical collection only; never physical merge."
                status = "logical_only"
            else:
                recommended = "Do not merge."
                status = "break"
            run = {
                "run_id": run_id,
                "run_type": current_type,
                "run_status": status,
                "physical_merge_allowed_now": physical_allowed,
                "logical_collection_only": logical_only,
                "review_required": review_required,
                "v5_source_key": src,
                "source_pdf_id_guess": source_id_guess(current),
                "page_range": f"{pages[0]}-{pages[-1]}" if pages else "",
                "component_pages": semi(pages),
                "n_pages": len(pages),
                "edge_pairs": semi(edge.get("page_pair", "") for edge in current),
                "n_edges": len(current),
                "edge_statuses": semi(statuses),
                "site_actions": semi(actions),
                "row_label_dependency_edge_count": row_label_count,
                "missing_page_count": len(missing_all),
                "missing_pages": semi(missing_all),
                "classifier_break_count": classifier_break_count,
                "table_family_keys": semi(uniq([edge.get("prev_table_family_key", "") for edge in current] + [edge.get("table_family_key", "") for edge in current])),
                "table_number_keys": semi(uniq([edge.get("prev_table_number_key", "") for edge in current] + [edge.get("table_number_key", "") for edge in current])),
                "representative_title": current[0].get("prev_title") or current[0].get("title", ""),
                "run_flags": semi(sorted(set(flag for edge in current for flag in split_semicolon(edge.get("rule_flags", ""))))),
                "recommended_action": recommended,
            }
            runs.append(run)
            for edge in current:
                edge_to_run[edge.get("rule_id", "")] = run_id
                edge_sequence.append(
                    {
                        "run_id": run_id,
                        "run_type": current_type,
                        "v5_source_key": src,
                        "source_pdf_id_guess": edge.get("source_pdf_id_guess", ""),
                        "page_pair": edge.get("page_pair", ""),
                        "prev_pdf_page": edge.get("prev_pdf_page", ""),
                        "pdf_page": edge.get("pdf_page", ""),
                        "page_gap": edge.get("page_gap", ""),
                        "physical_rule_status": edge.get("physical_rule_status", ""),
                        "physical_merge_allowed_now": edge.get("physical_merge_allowed_now", ""),
                        "logical_group_allowed_now": edge.get("logical_group_allowed_now", ""),
                        "site_action": edge.get("site_action", ""),
                        "row_label_dependency_signal": edge.get("row_label_dependency_signal", ""),
                        "classifier_break_signal": edge.get("classifier_break_signal", ""),
                        "same_table_family": edge.get("same_table_family", ""),
                        "same_table_number": edge.get("same_table_number", ""),
                        "rule_flags": edge.get("rule_flags", ""),
                        "decision_reason": edge.get("decision_reason", ""),
                    }
                )
            current = []
            current_type = ""
            previous_right = None

        for edge in source_edges:
            left = int_or_none(edge.get("prev_pdf_page"))
            right = int_or_none(edge.get("pdf_page"))
            run_type = edge_run_type(edge)
            adjacent_to_previous = previous_right is not None and left == previous_right
            same_run = current and run_type == current_type and adjacent_to_previous
            if not same_run:
                flush()
                current_type = run_type
            current.append(edge)
            previous_right = right
        flush()

    runs.sort(key=lambda row: (str(row.get("v5_source_key", "")), int_or_none(str(row.get("page_range", "")).split("-", 1)[0]) or 10**9, str(row.get("run_id", ""))))
    for idx, run in enumerate(runs, start=1):
        run["run_rank"] = idx
    edge_sequence.sort(key=lambda row: (str(row.get("v5_source_key", "")), int_or_none(row.get("prev_pdf_page")) or 10**9, int_or_none(row.get("pdf_page")) or 10**9))
    for idx, row in enumerate(edge_sequence, start=1):
        row["sequence_rank"] = idx
    return runs, edge_to_run, edge_sequence


def build_page_group_maps(groups: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, list[str]]]:
    out: dict[tuple[str, str], dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for group in groups:
        src = group.get("v5_source_key", "")
        group_id = group.get("group_id", "")
        group_type = group.get("group_type", "")
        if not src or not group_id:
            continue
        bucket = "candidate_group_ids"
        if group_type == "physical_component":
            bucket = "current_physical_component_ids"
        elif group_type == "logical_family_collection":
            bucket = "logical_group_ids"
        for page in parse_pages(group.get("component_pages", "")):
            out[(src, str(page))][bucket].append(group_id)
    return out


def build_page_sequences(
    pages: list[dict[str, str]],
    edges: list[dict[str, str]],
    edge_to_run: dict[str, str],
    runs: list[dict[str, object]],
    groups: list[dict[str, str]],
) -> list[dict[str, object]]:
    run_by_id = {str(run.get("run_id")): run for run in runs}
    prev_by_page: dict[tuple[str, str], dict[str, str]] = {}
    next_by_page: dict[tuple[str, str], dict[str, str]] = {}
    for edge in edges:
        src = edge.get("v5_source_key", "")
        left = edge.get("prev_pdf_page", "")
        right = edge.get("pdf_page", "")
        if src and left:
            next_by_page[(src, left)] = edge
        if src and right:
            prev_by_page[(src, right)] = edge
    group_map = build_page_group_maps(groups)
    rows: list[dict[str, object]] = []
    for page in pages:
        key = (page.get("v5_source_key", ""), page.get("pdf_page", ""))
        prev_edge = prev_by_page.get(key, {})
        next_edge = next_by_page.get(key, {})
        prev_run = run_by_id.get(edge_to_run.get(prev_edge.get("rule_id", ""), ""), {})
        next_run = run_by_id.get(edge_to_run.get(next_edge.get("rule_id", ""), ""), {})
        page_groups = group_map.get(key, {})
        rows.append(
            {
                "v5_source_key": page.get("v5_source_key", ""),
                "source_pdf_id_guess": page.get("source_pdf_id_guess", ""),
                "pdf_page": page.get("pdf_page", ""),
                "page_parity": page.get("page_parity", ""),
                "page_kind": page.get("page_kind", ""),
                "page_class": page.get("page_class", ""),
                "no_row_labels_signal": page.get("no_row_labels_signal", ""),
                "row_label_ratio": page.get("row_label_ratio", ""),
                "prev_edge_status": prev_edge.get("physical_rule_status", ""),
                "prev_edge_pair": prev_edge.get("page_pair", ""),
                "prev_edge_run_type": prev_run.get("run_type", ""),
                "prev_edge_site_action": prev_edge.get("site_action", ""),
                "next_edge_status": next_edge.get("physical_rule_status", ""),
                "next_edge_pair": next_edge.get("page_pair", ""),
                "next_edge_run_type": next_run.get("run_type", ""),
                "next_edge_site_action": next_edge.get("site_action", ""),
                "current_physical_component_ids": semi(page_groups.get("current_physical_component_ids", [])),
                "candidate_group_ids": semi(page_groups.get("candidate_group_ids", [])),
                "logical_group_ids": semi(page_groups.get("logical_group_ids", [])),
                "title": page.get("title", ""),
                "image_path": page.get("image_path", ""),
            }
        )
    rows.sort(key=lambda row: (str(row.get("v5_source_key", "")), int_or_none(row.get("pdf_page")) or 10**9))
    for idx, row in enumerate(rows, start=1):
        row["sequence_rank"] = idx
    return rows


def source_page_gaps(pages: list[int]) -> list[str]:
    gaps: list[str] = []
    for left, right in zip(pages, pages[1:]):
        if right > left + 1:
            gaps.append(f"{left}-{right} missing {semi(range(left + 1, right))}")
    return gaps


def build_source_summary(page_rows: list[dict[str, object]], run_rows: list[dict[str, object]], edge_rows: list[dict[str, str]], queue_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    pages_by_source: dict[str, list[dict[str, object]]] = defaultdict(list)
    runs_by_source: dict[str, list[dict[str, object]]] = defaultdict(list)
    edges_by_source: dict[str, list[dict[str, str]]] = defaultdict(list)
    queues_by_source: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in page_rows:
        pages_by_source[str(row.get("v5_source_key", ""))].append(row)
    for row in run_rows:
        runs_by_source[str(row.get("v5_source_key", ""))].append(row)
    for row in edge_rows:
        edges_by_source[str(row.get("v5_source_key", ""))].append(row)
    for row in queue_rows:
        queues_by_source[str(row.get("v5_source_key", ""))].append(row)

    out: list[dict[str, object]] = []
    for idx, src in enumerate(sorted(pages_by_source), start=1):
        pages = sorted(int_or_none(row.get("pdf_page")) for row in pages_by_source[src])
        page_nums = [page for page in pages if page is not None]
        runs = runs_by_source.get(src, [])
        edges = edges_by_source.get(src, [])
        queues = sorted(queues_by_source.get(src, []), key=lambda row: int_or_none(row.get("priority")) or 99)
        source_guess = source_id_guess(pages_by_source[src] + runs)
        confirmed_runs = [row for row in runs if row.get("run_type") == "confirmed_physical"]
        candidate_runs = [row for row in runs if row.get("run_type") in {"physical_candidate", "missing_page_bridge_candidate"}]
        logical_runs = [row for row in runs if row.get("run_type") == "logical_only"]
        flags: list[str] = []
        if confirmed_runs:
            flags.append("has_confirmed_physical")
        if candidate_runs:
            flags.append("has_physical_candidates")
        if logical_runs:
            flags.append("has_logical_only")
        if any(edge.get("row_label_dependency_signal") == "yes" for edge in edges):
            flags.append("has_row_label_dependency_signal")
        out.append(
            {
                "source_rank": idx,
                "v5_source_key": src,
                "source_pdf_id_guess": source_guess,
                "observed_page_count": len(page_nums),
                "observed_page_min": min(page_nums) if page_nums else "",
                "observed_page_max": max(page_nums) if page_nums else "",
                "observed_page_span": (max(page_nums) - min(page_nums) + 1) if page_nums else "",
                "internal_observed_page_gaps": semi(source_page_gaps(page_nums)),
                "confirmed_physical_run_count": len(confirmed_runs),
                "confirmed_physical_page_count": sum(int_or_none(row.get("n_pages")) or 0 for row in confirmed_runs),
                "physical_candidate_run_count": len(candidate_runs),
                "physical_candidate_edge_count": sum(1 for edge in edges if edge.get("physical_rule_status") in {"candidate_row_label_dependency", "candidate_physical_missing_ocr", "unpromoted_row_label_dependency_signal", "failed_or_partial_physical_review", "likely_assemblable_after_missing_pages"}),
                "row_label_dependency_edge_count": sum(1 for edge in edges if edge.get("row_label_dependency_signal") == "yes"),
                "missing_bridge_edge_count": sum(1 for edge in edges if edge.get("physical_rule_status") == "likely_assemblable_after_missing_pages"),
                "logical_only_run_count": len(logical_runs),
                "hard_break_edge_count": sum(1 for edge in edges if edge.get("physical_rule_status") == "hard_break_or_not_candidate"),
                "first_priority_queue_type": queues[0].get("queue_type", "") if queues else "",
                "first_priority_page_pair": queues[0].get("page_pair", "") if queues else "",
                "source_flags": semi(flags),
            }
        )
    return out


def build_summary(run_rows: list[dict[str, object]], source_rows: list[dict[str, object]], page_rows: list[dict[str, object]], edge_sequence: list[dict[str, object]], paths: dict[str, Path]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = [
        {"metric": "paths", "name": name, "count": output_path(path)}
        for name, path in paths.items()
    ]
    rows.extend(
        [
            {"metric": "counts", "name": "assembly_runs", "count": len(run_rows)},
            {"metric": "counts", "name": "source_summaries", "count": len(source_rows)},
            {"metric": "counts", "name": "page_sequence_rows", "count": len(page_rows)},
            {"metric": "counts", "name": "edge_sequence_rows", "count": len(edge_sequence)},
        ]
    )
    for metric, counter in [
        ("runs_by_type", Counter(str(row.get("run_type", "")) for row in run_rows)),
        ("runs_by_status", Counter(str(row.get("run_status", "")) for row in run_rows)),
        ("sources_by_flags", Counter(flag for row in source_rows for flag in split_semicolon(row.get("source_flags", "")))),
        ("pages_by_prev_edge_run_type", Counter(str(row.get("prev_edge_run_type", "")) for row in page_rows)),
        ("pages_by_next_edge_run_type", Counter(str(row.get("next_edge_run_type", "")) for row in page_rows)),
    ]:
        for name, count in sorted(counter.items()):
            rows.append({"metric": metric, "name": name, "count": count})
    return rows


def audit_outputs(
    runs: list[dict[str, object]],
    page_rows: list[dict[str, object]],
    edge_sequence: list[dict[str, object]],
    source_edge_count: int,
    source_page_count: int,
    workbook_path: Path,
) -> list[dict[str, object]]:
    audits: list[dict[str, object]] = []
    audits.append(
        {
            "check": "edge_sequence_covers_all_rulebook_edges",
            "status": "pass" if len(edge_sequence) == source_edge_count else "fail",
            "detail": f"sequence={len(edge_sequence)} source={source_edge_count}",
        }
    )
    audits.append(
        {
            "check": "page_sequence_covers_all_rulebook_pages",
            "status": "pass" if len(page_rows) == source_page_count else "fail",
            "detail": f"sequence={len(page_rows)} source={source_page_count}",
        }
    )
    bad_confirmed_runs = [
        run for run in runs
        if run.get("physical_merge_allowed_now") == "yes" and run.get("run_type") != "confirmed_physical"
    ]
    audits.append(
        {
            "check": "only_confirmed_physical_runs_allow_physical_merge",
            "status": "pass" if not bad_confirmed_runs else "fail",
            "detail": f"bad={len(bad_confirmed_runs)}",
        }
    )
    bad_candidate_runs = [
        run for run in runs
        if run.get("run_type") != "confirmed_physical" and run.get("physical_merge_allowed_now") == "yes"
    ]
    audits.append(
        {
            "check": "candidate_runs_never_allow_physical_merge",
            "status": "pass" if not bad_candidate_runs else "fail",
            "detail": f"bad={len(bad_candidate_runs)}",
        }
    )
    non_pairwise_candidate = [
        run for run in runs
        if run.get("run_type") in {"physical_candidate", "missing_page_bridge_candidate"}
        and "Retest/promote edge-by-edge" not in str(run.get("recommended_action"))
        and "Render/OCR all listed missing pages" not in str(run.get("recommended_action"))
    ]
    audits.append(
        {
            "check": "candidate_runs_are_edge_by_edge_or_missing_page_only",
            "status": "pass" if not non_pairwise_candidate else "fail",
            "detail": f"bad={len(non_pairwise_candidate)}",
        }
    )
    try:
        with zipfile.ZipFile(workbook_path) as zf:
            bad_member = zf.testzip()
    except Exception as exc:  # pragma: no cover
        bad_member = f"{type(exc).__name__}: {exc}"
    audits.append(
        {
            "check": "workbook_zip_integrity",
            "status": "pass" if bad_member is None else "fail",
            "detail": str(bad_member or "ok"),
        }
    )
    return audits


def write_workbook(
    path: Path,
    summary_rows: list[dict[str, object]],
    audit_rows: list[dict[str, object]],
    source_rows: list[dict[str, object]],
    run_rows: list[dict[str, object]],
    page_rows: list[dict[str, object]],
    edge_rows: list[dict[str, object]],
) -> None:
    workbook = xlsxwriter.Workbook(path, {"constant_memory": True})
    header_fmt = workbook.add_format({"bold": True, "bg_color": "#E8EEF7", "border": 1})
    wrap_fmt = workbook.add_format({"text_wrap": True, "valign": "top"})
    sheets = [
        ("Summary", summary_rows, SUMMARY_FIELDS),
        ("Audit", audit_rows, AUDIT_FIELDS),
        ("Source Summary", source_rows, SOURCE_FIELDS),
        ("Assembly Runs", run_rows, RUN_FIELDS),
        ("Page Sequence", page_rows, PAGE_FIELDS),
        ("Edge Sequence", edge_rows, EDGE_SEQUENCE_FIELDS),
    ]
    for name, rows, fields in sheets:
        ws = workbook.add_worksheet(name[:31])
        for col, field in enumerate(fields):
            ws.write(0, col, field, header_fmt)
        for row_idx, row in enumerate(rows, start=1):
            for col, field in enumerate(fields):
                value = row.get(field, "")
                fmt = wrap_fmt if field in {"internal_observed_page_gaps", "run_flags", "recommended_action", "title", "decision_reason", "rule_flags"} else None
                ws.write(row_idx, col, value, fmt)
        ws.freeze_panes(1, 0)
        ws.autofilter(0, 0, max(0, len(rows)), max(0, len(fields) - 1))
        for col, field in enumerate(fields):
            ws.set_column(col, col, min(max(len(field) + 2, 12), 54))
    workbook.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rulebook-dir", default=str(DEFAULT_RULEBOOK_DIR))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    args = parser.parse_args()

    rulebook_dir = Path(args.rulebook_dir).expanduser()
    out_dir = Path(args.out_dir).expanduser()
    paths = {"rulebook_dir": rulebook_dir, "out_dir": out_dir}

    edges = read_csv(rulebook_dir / "edge_rulebook.csv")
    pages = read_csv(rulebook_dir / "page_signal_index.csv")
    groups = read_csv(rulebook_dir / "grouping_registry.csv")
    queue = read_csv(rulebook_dir / "promotion_queue.csv")

    runs, edge_to_run, edge_sequence = build_runs(edges)
    page_sequence = build_page_sequences(pages, edges, edge_to_run, runs, groups)
    source_summary = build_source_summary(page_sequence, runs, edges, queue)
    summary_rows = build_summary(runs, source_summary, page_sequence, edge_sequence, paths)

    out_dir.mkdir(parents=True, exist_ok=True)
    workbook_path = out_dir / "source_sequence_map_workbook.xlsx"
    write_workbook(workbook_path, summary_rows, [], source_summary, runs, page_sequence, edge_sequence)
    audit_rows = audit_outputs(runs, page_sequence, edge_sequence, len(edges), len(pages), workbook_path)
    summary_rows.extend({"metric": "audits", "name": row["check"], "count": row["status"]} for row in audit_rows)
    write_workbook(workbook_path, summary_rows, audit_rows, source_summary, runs, page_sequence, edge_sequence)
    audit_rows = audit_outputs(runs, page_sequence, edge_sequence, len(edges), len(pages), workbook_path)

    write_csv(out_dir / "assembly_runs.csv", runs, RUN_FIELDS)
    write_csv(out_dir / "source_sequence_summary.csv", source_summary, SOURCE_FIELDS)
    write_csv(out_dir / "page_sequence_map.csv", page_sequence, PAGE_FIELDS)
    write_csv(out_dir / "edge_sequence_map.csv", edge_sequence, EDGE_SEQUENCE_FIELDS)
    write_csv(out_dir / "summary.csv", summary_rows, SUMMARY_FIELDS)
    write_csv(out_dir / "audit.csv", audit_rows, AUDIT_FIELDS)

    summary = {
        "out_dir": output_path(out_dir),
        "assembly_runs": len(runs),
        "source_summary_rows": len(source_summary),
        "page_sequence_rows": len(page_sequence),
        "edge_sequence_rows": len(edge_sequence),
        "runs_by_type": dict(Counter(str(row.get("run_type", "")) for row in runs)),
        "runs_by_status": dict(Counter(str(row.get("run_status", "")) for row in runs)),
        "audit_failure_count": sum(1 for row in audit_rows if row.get("status") != "pass"),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
