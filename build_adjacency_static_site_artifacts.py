#!/usr/bin/env python3
"""Create static-site-ready artifacts from v4 adjacency components.

The v4 assembly is record-level: a single physical PDF page can appear more
than once when original OCR, core-money renders, and overlap records all point
at the same source page. The static site needs a physical-page view.

This script keeps v4 intact and writes additional v4 artifacts:
  * one row per component/run;
  * one deduped ordered member row per physical PDF page in the run;
  * JSONL group records for future static-site ingestion;
  * focused main age/civil candidate queue.
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parent
V4_DIR = ROOT / "export_final/pdf_recovery_incoming_2026-06-16_all_ocr_component_assembly_v4_adjacency"

COMPONENTS_CSV = V4_DIR / "all_ocr_component_summary.csv"
PAGES_CSV = V4_DIR / "all_ocr_component_pages.csv"
EDGES_CSV = V4_DIR / "all_ocr_component_edges.csv"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fields})


def int_or_none(value: object) -> int | None:
    s = str(value or "").strip()
    if re.fullmatch(r"-?\d+", s):
        return int(s)
    return None


def uniq_join(values: Iterable[object], limit: int = 20) -> str:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        s = str(value or "").strip()
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    if len(out) > limit:
        return "; ".join(out[:limit]) + f"; ...(+{len(out)-limit})"
    return "; ".join(out)


def page_key(row: dict[str, str]) -> tuple[str, str]:
    source_pdf = row.get("source_pdf") or row.get("adj_source_key") or row.get("source_group_key") or ""
    return (source_pdf, row.get("pdf_page") or "")


def page_sort(row: dict[str, str]) -> tuple[int, str, int]:
    page = int_or_none(row.get("pdf_page"))
    return (page if page is not None else 10**9, row.get("page_kind", ""), int_or_none(row.get("position_in_component")) or 0)


def quality_penalty(row: dict[str, str]) -> int:
    flags = row.get("quality_flags", "") + ";" + row.get("edge_review_flags", "")
    penalty = 0
    for flag, points in [
        ("constraint_failures", 200),
        ("zero_constraint_checks", 120),
        ("zero_rows_extracted", 120),
        ("low_quality_ocr", 90),
        ("missing_source_pdf", 70),
        ("missing_pdf_page", 70),
        ("incompatible_page_class_transition", 50),
    ]:
        if flag in flags:
            penalty += points
    return penalty


def representative_score(row: dict[str, str]) -> tuple[int, int, int, int, int]:
    """Higher score wins."""
    has_ocr = 1 if row.get("ocr_artifact") else 0
    checks = int_or_none(row.get("constraint_checks")) or 0
    rows = int_or_none(row.get("rows_extracted")) or int_or_none(row.get("core_rows")) or 0
    has_image = 1 if row.get("image_path") else 0
    return (has_ocr, checks, rows, has_image, -quality_penalty(row))


def choose_representative(rows: list[dict[str, str]]) -> dict[str, str]:
    return sorted(rows, key=representative_score, reverse=True)[0]


def static_priority(component: dict[str, str], rows: list[dict[str, str]], deduped_n: int) -> tuple[int, str, str]:
    basis = component.get("component_basis", "")
    flags = component.get("component_review_flags", "")
    if basis == "adjacency_main_age_civil_numeric_run":
        if component.get("review_flag") == "ok_main_numeric":
            return 1, "main_age_civil_ready", ""
        if "missing_source_pdf" in flags or "missing_intermediate_pages" in flags:
            return 3, "main_age_civil_needs_gap_or_source_review", flags
        if "constraint_failures" in flags or "zero_constraint_checks" in flags or "zero_rows_extracted" in flags or "low_quality_ocr" in flags:
            return 2, "main_age_civil_needs_ocr_quality_review", flags
        return 2, "main_age_civil_needs_light_review", flags
    if basis == "adjacency_derivative_selected_caste_race_tribe_run":
        return 5, "derivative_selected_lower_priority", flags
    if basis == "adjacency_context_or_blank_run":
        return 8, "context_or_blank_not_numeric_body", flags
    if basis == "adjacency_offtarget_table_run":
        return 9, "offtarget_not_for_age_static_site", flags
    if deduped_n == 1:
        return 7, "singleton_or_unsequenced_review", flags
    return 6, "other_structured_review", flags


def consecutive_unique_pages(rows: list[dict[str, str]]) -> tuple[bool, str]:
    nums = sorted({n for n in (int_or_none(r.get("pdf_page")) for r in rows) if n is not None})
    if len(nums) <= 1:
        return True, ""
    gaps = []
    for a, b in zip(nums, nums[1:]):
        if b != a + 1:
            gaps.append(f"{a}->{b}")
    return not gaps, "; ".join(gaps)


def main() -> None:
    components = {r["component_id"]: r for r in read_csv(COMPONENTS_CSV)}
    pages = read_csv(PAGES_CSV)
    by_component: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in pages:
        by_component[row["component_id"]].append(row)

    group_rows: list[dict[str, object]] = []
    member_rows: list[dict[str, object]] = []
    main_candidate_rows: list[dict[str, object]] = []
    jsonl_path = V4_DIR / "static_site_groups.jsonl"

    with jsonl_path.open("w", encoding="utf-8") as jf:
        for cid, component in sorted(components.items()):
            records = sorted(by_component.get(cid, []), key=page_sort)
            physical: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
            for row in records:
                physical[page_key(row)].append(row)
            deduped_pages = []
            for key, rows_for_page in sorted(physical.items(), key=lambda kv: (int_or_none(kv[0][1]) if int_or_none(kv[0][1]) is not None else 10**9, kv[0][0])):
                rep = choose_representative(rows_for_page)
                deduped_pages.append((key, rep, rows_for_page))

            member_start = len(member_rows) + 1
            for idx, (key, rep, rows_for_page) in enumerate(deduped_pages, start=1):
                member = {
                    "component_id": cid,
                    "static_group_id": cid,
                    "page_index": idx,
                    "source_pdf": rep.get("source_pdf", ""),
                    "source_pdf_id": rep.get("source_pdf_id", ""),
                    "pdf_page": rep.get("pdf_page", ""),
                    "page_kind": rep.get("page_kind", ""),
                    "classifier_page_class": rep.get("classifier_page_class", ""),
                    "title": rep.get("classifier_printed_title") or rep.get("title") or rep.get("core_ocr_title") or "",
                    "selected_image_path": rep.get("image_path", ""),
                    "selected_ocr_artifact": rep.get("ocr_artifact", ""),
                    "selected_rows_extracted": rep.get("rows_extracted") or rep.get("core_rows") or "",
                    "selected_constraint_checks": rep.get("constraint_checks", ""),
                    "selected_constraint_failed": rep.get("constraint_failed", ""),
                    "selected_quality_flags": rep.get("quality_flags", ""),
                    "edge_from_prev_action": rep.get("edge_from_prev_action", ""),
                    "edge_from_prev_reason": rep.get("edge_from_prev_reason", ""),
                    "edge_review_flags": rep.get("edge_review_flags", ""),
                    "duplicate_record_count": len(rows_for_page),
                    "duplicate_image_paths": uniq_join([r.get("image_path", "") for r in rows_for_page], limit=8),
                    "duplicate_v3_component_ids": uniq_join([r.get("v3_component_id", "") for r in rows_for_page], limit=8),
                    "representative_score": json.dumps(representative_score(rep)),
                }
                member_rows.append(member)

            is_consecutive, page_gap_edges = consecutive_unique_pages([rep for _, rep, _ in deduped_pages])
            priority, priority_label, priority_reason = static_priority(component, records, len(deduped_pages))
            group = {
                "static_group_id": cid,
                "component_id": cid,
                "component_basis": component.get("component_basis", ""),
                "static_priority": priority,
                "static_priority_label": priority_label,
                "static_priority_reason": priority_reason,
                "review_flag": component.get("review_flag", ""),
                "component_review_flags": component.get("component_review_flags", ""),
                "n_unique_physical_pages": len(deduped_pages),
                "n_source_records": len(records),
                "duplicate_record_count": len(records) - len(deduped_pages),
                "unique_pages_are_consecutive": is_consecutive,
                "unique_page_gap_edges": page_gap_edges,
                "source_pdfs": component.get("source_pdfs", ""),
                "source_pdf_ids": component.get("source_pdf_ids", ""),
                "page_range": component.get("page_range", ""),
                "page_kind_counts": component.get("page_kind_counts", ""),
                "classifier_page_class_counts": component.get("classifier_page_class_counts", ""),
                "primary_title": component.get("primary_title", ""),
                "row_records_from_json_pages": component.get("row_records_from_json_pages", ""),
                "constraint_checks_total": component.get("constraint_checks_total", ""),
                "constraint_failed_total": component.get("constraint_failed_total", ""),
                "member_start_row_1based": member_start,
                "member_end_row_1based": len(member_rows),
                "example_images": component.get("example_images", ""),
            }
            group_rows.append(group)
            jf.write(json.dumps({
                **group,
                "members": [
                    {
                        "page_index": idx,
                        "source_pdf": rep.get("source_pdf", ""),
                        "pdf_page": rep.get("pdf_page", ""),
                        "page_kind": rep.get("page_kind", ""),
                        "title": rep.get("classifier_printed_title") or rep.get("title") or rep.get("core_ocr_title") or "",
                        "image_path": rep.get("image_path", ""),
                        "ocr_artifact": rep.get("ocr_artifact", ""),
                        "duplicate_record_count": len(rows_for_page),
                    }
                    for idx, (_, rep, rows_for_page) in enumerate(deduped_pages, start=1)
                ],
            }, ensure_ascii=False) + "\n")

            if component.get("component_basis") == "adjacency_main_age_civil_numeric_run":
                main_candidate_rows.append(group)

    group_fields = [
        "static_priority", "static_priority_label", "static_group_id", "component_id", "component_basis",
        "review_flag", "component_review_flags", "n_unique_physical_pages", "n_source_records",
        "duplicate_record_count", "unique_pages_are_consecutive", "unique_page_gap_edges",
        "source_pdfs", "source_pdf_ids", "page_range", "page_kind_counts",
        "classifier_page_class_counts", "primary_title", "row_records_from_json_pages",
        "constraint_checks_total", "constraint_failed_total", "member_start_row_1based",
        "member_end_row_1based", "example_images", "static_priority_reason",
    ]
    member_fields = [
        "static_group_id", "component_id", "page_index", "source_pdf", "source_pdf_id", "pdf_page",
        "page_kind", "classifier_page_class", "title", "selected_image_path",
        "selected_ocr_artifact", "selected_rows_extracted", "selected_constraint_checks",
        "selected_constraint_failed", "selected_quality_flags", "edge_from_prev_action",
        "edge_from_prev_reason", "edge_review_flags", "duplicate_record_count",
        "duplicate_image_paths", "duplicate_v3_component_ids", "representative_score",
    ]

    group_rows.sort(key=lambda r: (int(r["static_priority"]), -int(r["n_unique_physical_pages"]), str(r["source_pdfs"]), str(r["page_range"])))
    main_candidate_rows.sort(key=lambda r: (int(r["static_priority"]), -int(r["n_unique_physical_pages"]), str(r["source_pdfs"]), str(r["page_range"])))
    write_csv(V4_DIR / "static_site_groups.csv", group_rows, group_fields)
    write_csv(V4_DIR / "static_site_group_members.csv", member_rows, member_fields)
    write_csv(V4_DIR / "static_site_main_age_civil_candidates.csv", main_candidate_rows, group_fields)

    summary = {
        "static_groups_total": len(group_rows),
        "static_members_total_unique_physical_pages": len(member_rows),
        "main_age_civil_groups": len(main_candidate_rows),
        "main_age_civil_unique_physical_pages": sum(int(r["n_unique_physical_pages"]) for r in main_candidate_rows),
        "priority_counts": dict(Counter(str(r["static_priority"]) + "_" + str(r["static_priority_label"]) for r in group_rows)),
        "nonconsecutive_group_count": sum(1 for r in group_rows if not r["unique_pages_are_consecutive"]),
        "groups_with_duplicate_source_records": sum(1 for r in group_rows if int(r["duplicate_record_count"]) > 0),
        "outputs": {
            "static_site_groups_csv": str(V4_DIR / "static_site_groups.csv"),
            "static_site_group_members_csv": str(V4_DIR / "static_site_group_members.csv"),
            "static_site_main_age_civil_candidates_csv": str(V4_DIR / "static_site_main_age_civil_candidates.csv"),
            "static_site_groups_jsonl": str(jsonl_path),
        },
    }
    (V4_DIR / "static_site_artifacts_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
