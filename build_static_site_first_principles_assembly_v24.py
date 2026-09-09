#!/usr/bin/env python3
"""Audit current static-site assemblies against first-principles physical runs.

This pass does not OCR anything. It treats the current static-site assembled
tables as untrusted input, maps their member images back to the strict v6
physical-page/run inventory, and writes replacement grouping records.

Physical assembly rule: a proposed physical group can only be one v6 physical
run, which itself is built from adjacent real PDF pages. Multi-run relationships
are exported only as logical/review candidates.
"""

from __future__ import annotations

import csv
import json
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import xlsxwriter

import build_all_ocr_component_assembly_v5_structural as base


ROOT = Path(__file__).resolve().parent
V1_DIR = ROOT / "export_final/sendout_final/sendout_v1"
V6_DIR = ROOT / "export_final/pdf_recovery_incoming_2026-06-16_all_ocr_component_assembly_v6_layout_refined"
OUT_DIR = ROOT / "export_final/pdf_recovery_incoming_2026-06-16_static_site_first_principles_assembly_v24"

ASSEMBLED_MEMBERS = V1_DIR / "assembled_members.json"
ASSEMBLED_CATALOG = V1_DIR / "assembled_catalog.csv"
V6_PAGES = V6_DIR / "v6_physical_pages.csv"
V6_RUNS = V6_DIR / "v6_physical_runs.csv"
V6_RUN_MEMBERS = V6_DIR / "v6_run_members.csv"
V6_COLLECTIONS = V6_DIR / "v6_collection_groups.csv"
V6_COLLECTION_MEMBERS = V6_DIR / "v6_collection_members.csv"

HEX40 = re.compile(r"[0-9a-f]{40}", flags=re.IGNORECASE)


SITE_MEMBER_FIELDS = [
    "site_table_id",
    "member_position",
    "member_ref",
    "image_hash",
    "hash_match_count",
    "match_status",
    "physical_page_id",
    "physical_run_id",
    "collection_group_ids",
    "v5_source_key",
    "source_pdf",
    "source_pdf_id",
    "pdf_page",
    "page_kind",
    "table_family_key",
    "table_number_key",
    "title",
    "image_path",
    "row_label_ratio",
    "layout_left_stub_candidate",
    "layout_right_half_candidate",
    "quality_flags",
]

AUDIT_FIELDS = [
    "site_table_id",
    "status",
    "year",
    "canonical_name",
    "canonical_topic",
    "title",
    "n_current_member_refs",
    "n_member_hashes",
    "n_matched_member_refs",
    "n_unmatched_member_refs",
    "n_ambiguous_member_refs",
    "n_unique_matched_pages",
    "n_distinct_sources",
    "n_v6_physical_runs",
    "n_v6_collection_groups",
    "n_proposed_physical_groups",
    "current_member_pdf_order",
    "sorted_pdf_order",
    "page_order_status",
    "page_gap_edges",
    "incomplete_required_run_ids",
    "logical_collection_candidate_ids",
    "issue_flags",
    "recommended_action",
    "proposed_physical_group_ids",
    "example_images",
]

PROPOSED_FIELDS = [
    "proposed_physical_group_id",
    "site_table_id",
    "group_sequence",
    "recommended_action",
    "v6_physical_run_id",
    "v6_run_full_page_range",
    "included_page_range",
    "included_pages_are_consecutive",
    "included_page_count",
    "full_run_page_count",
    "run_included_completely",
    "v5_source_key",
    "source_pdf",
    "source_pdf_id",
    "collection_group_ids",
    "page_kind_counts",
    "table_family_counts",
    "primary_title",
    "included_member_positions",
    "included_image_hashes",
    "example_images",
    "review_flags",
]

LOGICAL_FIELDS = [
    "site_table_id",
    "collection_group_id",
    "collection_basis",
    "n_site_physical_runs_in_collection",
    "n_collection_physical_runs_total",
    "site_run_ids",
    "collection_page_ranges",
    "collection_review_flags",
    "primary_title",
    "recommended_action",
]

SUMMARY_FIELDS = ["metric", "value"]
AUDIT_CHECK_FIELDS = ["check", "status", "failure_count", "details"]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def int_or_none(value: object) -> int | None:
    return base.int_or_none(value)


def uniq(values: list[object] | tuple[object, ...] | set[object]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    return out


def uniq_join(values, limit: int = 25) -> str:
    return base.uniq_join(values, limit=limit)


def hashes_from_text(value: object) -> list[str]:
    return [h.lower() for h in HEX40.findall(str(value or ""))]


def compact_page_range(pages: list[int]) -> str:
    values = sorted(set(pages))
    if not values:
        return ""
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


def pages_are_consecutive(pages: list[int]) -> bool:
    values = sorted(set(pages))
    return bool(values) and values == list(range(values[0], values[-1] + 1))


def page_sort_key(row: dict[str, object]) -> tuple[str, int, str]:
    page = int_or_none(row.get("pdf_page"))
    return (
        str(row.get("v5_source_key") or ""),
        page if page is not None else 10**9,
        str(row.get("physical_page_id") or ""),
    )


def load_site_catalog() -> tuple[list[dict[str, str]], dict[str, list[str]]]:
    catalog = read_csv(ASSEMBLED_CATALOG)
    member_map = json.loads(ASSEMBLED_MEMBERS.read_text(encoding="utf-8"))
    catalog_by_id = {row["table_id"]: row for row in catalog}
    for table_id, row in catalog_by_id.items():
        if table_id not in member_map:
            member_map[table_id] = []
        # The catalog sometimes carries absolute image paths even when the
        # member map has a normalized OCR id. Include both as matchable refs.
        for image in str(row.get("member_images") or "").split(";"):
            image = image.strip()
            if image:
                member_map[table_id].append(image)
    return catalog, {str(k): [str(v) for v in vals] for k, vals in member_map.items()}


def build_hash_index(pages: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    index: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
    for page in pages:
        for field in ["image_path", "duplicate_image_paths"]:
            for image_hash in hashes_from_text(page.get(field, "")):
                index[image_hash].append(page)
    # Deduplicate repeated references to the same physical page.
    out: dict[str, list[dict[str, str]]] = {}
    for image_hash, rows in index.items():
        seen: set[str] = set()
        deduped: list[dict[str, str]] = []
        for row in sorted(rows, key=page_sort_key):
            page_id = str(row.get("physical_page_id") or "")
            if page_id and page_id not in seen:
                seen.add(page_id)
                deduped.append(row)
        out[image_hash] = deduped
    return out


def build_run_indexes(
    runs: list[dict[str, str]],
    run_members: list[dict[str, str]],
    collection_groups: list[dict[str, str]],
    collection_members: list[dict[str, str]],
) -> tuple[dict[str, dict[str, str]], dict[str, list[dict[str, str]]], dict[str, list[str]], dict[str, dict[str, str]], dict[str, list[str]]]:
    run_by_id = {str(row.get("physical_run_id") or ""): row for row in runs}
    members_by_run: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
    for row in run_members:
        members_by_run[str(row.get("physical_run_id") or "")].append(row)
    for rows in members_by_run.values():
        rows.sort(key=lambda r: int_or_none(r.get("pdf_page")) or 10**9)
    collections_by_id = {str(row.get("collection_group_id") or ""): row for row in collection_groups}
    collections_by_run: defaultdict[str, list[str]] = defaultdict(list)
    runs_by_collection: defaultdict[str, list[str]] = defaultdict(list)
    for row in collection_members:
        run_id = str(row.get("physical_run_id") or "")
        collection_id = str(row.get("collection_group_id") or "")
        if run_id and collection_id:
            collections_by_run[run_id].append(collection_id)
            runs_by_collection[collection_id].append(run_id)
    return run_by_id, members_by_run, collections_by_run, collections_by_id, runs_by_collection


def choose_match(matches: list[dict[str, str]]) -> dict[str, str] | None:
    if not matches:
        return None
    # Prefer records with real source_pdf/page and representative image/OCR.
    return sorted(
        matches,
        key=lambda row: (
            1 if row.get("source_pdf") else 0,
            1 if row.get("pdf_page") else 0,
            int_or_none(row.get("constraint_checks")) or 0,
            int_or_none(row.get("rows_extracted")) or 0,
            str(row.get("physical_page_id") or ""),
        ),
        reverse=True,
    )[0]


def site_member_rows(
    catalog: list[dict[str, str]],
    member_map: dict[str, list[str]],
    hash_index: dict[str, list[dict[str, str]]],
    collections_by_run: dict[str, list[str]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for cat in catalog:
        table_id = cat["table_id"]
        refs = member_map.get(table_id, [])
        for pos, ref in enumerate(refs, start=1):
            hashes = hashes_from_text(ref)
            image_hash = hashes[-1] if hashes else ""
            matches = hash_index.get(image_hash, []) if image_hash else []
            match = choose_match(matches)
            if not image_hash:
                status = "no_image_hash_in_member_ref"
            elif not matches:
                status = "unmatched_hash"
            elif len(matches) > 1:
                status = "ambiguous_hash_chose_best_match"
            else:
                status = "matched"
            run_id = str(match.get("physical_run_id") or "") if match else ""
            rows.append(
                {
                    "site_table_id": table_id,
                    "member_position": pos,
                    "member_ref": ref,
                    "image_hash": image_hash,
                    "hash_match_count": len(matches),
                    "match_status": status,
                    "physical_page_id": match.get("physical_page_id", "") if match else "",
                    "physical_run_id": run_id,
                    "collection_group_ids": uniq_join(collections_by_run.get(run_id, []), limit=10),
                    "v5_source_key": match.get("v5_source_key", "") if match else "",
                    "source_pdf": match.get("source_pdf", "") if match else "",
                    "source_pdf_id": match.get("source_pdf_id", "") if match else "",
                    "pdf_page": match.get("pdf_page", "") if match else "",
                    "page_kind": match.get("page_kind", "") if match else "",
                    "table_family_key": match.get("table_family_key", "") if match else "",
                    "table_number_key": match.get("table_number_key", "") if match else "",
                    "title": match.get("title", "") if match else "",
                    "image_path": match.get("image_path", "") if match else "",
                    "row_label_ratio": match.get("profile_row_label_ratio", "") if match else "",
                    "layout_left_stub_candidate": match.get("layout_left_stub_candidate", "") if match else "",
                    "layout_right_half_candidate": match.get("layout_right_half_candidate", "") if match else "",
                    "quality_flags": match.get("quality_flags", "") if match else "",
                }
            )
    return rows


def audit_site_tables(
    catalog: list[dict[str, str]],
    member_rows: list[dict[str, object]],
    run_by_id: dict[str, dict[str, str]],
    members_by_run: dict[str, list[dict[str, str]]],
    collections_by_run: dict[str, list[str]],
    collections_by_id: dict[str, dict[str, str]],
    runs_by_collection: dict[str, list[str]],
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    rows_by_table: defaultdict[str, list[dict[str, object]]] = defaultdict(list)
    for row in member_rows:
        rows_by_table[str(row.get("site_table_id") or "")].append(row)

    audit_rows: list[dict[str, object]] = []
    proposed_rows: list[dict[str, object]] = []
    logical_rows: list[dict[str, object]] = []

    for cat in catalog:
        table_id = cat["table_id"]
        rows = rows_by_table.get(table_id, [])
        matched = [row for row in rows if row.get("physical_page_id")]
        unique_pages: dict[str, dict[str, object]] = {}
        for row in matched:
            unique_pages.setdefault(str(row.get("physical_page_id") or ""), row)
        unique_page_rows = list(unique_pages.values())
        unique_page_rows_sorted = sorted(unique_page_rows, key=page_sort_key)

        sources = uniq([row.get("v5_source_key") for row in unique_page_rows])
        run_ids = uniq([row.get("physical_run_id") for row in unique_page_rows])
        collection_ids = uniq(
            [
                cid
                for run_id in run_ids
                for cid in collections_by_run.get(str(run_id), [])
                if cid and collections_by_id.get(cid, {}).get("collection_basis") != "single_physical_run"
            ]
        )
        member_order = [
            f"{row.get('v5_source_key')}@{row.get('pdf_page')}"
            for row in matched
            if row.get("v5_source_key") or row.get("pdf_page")
        ]
        sorted_order = [
            f"{row.get('v5_source_key')}@{row.get('pdf_page')}"
            for row in unique_page_rows_sorted
            if row.get("v5_source_key") or row.get("pdf_page")
        ]
        member_order_pages = [
            (str(row.get("v5_source_key") or ""), int_or_none(row.get("pdf_page")))
            for row in matched
            if int_or_none(row.get("pdf_page")) is not None
        ]
        page_order_status = "unknown_no_matched_pages"
        if member_order_pages:
            page_order_status = "ok_pdf_order" if member_order_pages == sorted(member_order_pages) else "wrong_current_member_order"

        gap_edges: list[str] = []
        by_source: defaultdict[str, list[int]] = defaultdict(list)
        for row in unique_page_rows_sorted:
            page = int_or_none(row.get("pdf_page"))
            if page is not None:
                by_source[str(row.get("v5_source_key") or "")].append(page)
        for source, pages in by_source.items():
            for a, b in zip(sorted(set(pages)), sorted(set(pages))[1:]):
                if b > a + 1:
                    gap_edges.append(f"{source}:{a}->{b}")

        incomplete_required: list[str] = []
        for run_id in run_ids:
            full_members = members_by_run.get(str(run_id), [])
            full_pages = {str(row.get("physical_page_id") or "") for row in full_members}
            included_pages = {
                str(row.get("physical_page_id") or "")
                for row in unique_page_rows
                if str(row.get("physical_run_id") or "") == str(run_id)
            }
            if len(full_pages) > 1 and included_pages and included_pages != full_pages:
                incomplete_required.append(str(run_id))

        flags: list[str] = []
        if any(row.get("match_status") == "unmatched_hash" for row in rows):
            flags.append("unmatched_member_images")
        if any("ambiguous" in str(row.get("match_status") or "") for row in rows):
            flags.append("ambiguous_member_image_hashes")
        if len(sources) > 1:
            flags.append("multiple_sources_in_current_site_table")
        if page_order_status == "wrong_current_member_order":
            flags.append("current_site_member_order_not_pdf_order")
        if gap_edges:
            flags.append("current_site_table_has_pdf_page_gaps")
        if len(run_ids) > 1:
            flags.append("current_site_table_spans_multiple_strict_physical_runs")
        if incomplete_required:
            flags.append("current_site_table_splits_required_physical_run")
        if len(unique_page_rows) >= 4 and len(run_ids) >= 4:
            flags.append("current_site_table_too_large_for_single_physical_assembly")
        if any(str(row.get("page_kind") or "").startswith("derivative") for row in unique_page_rows):
            flags.append("contains_derivative_selected_table_page")
        if any(str(row.get("page_kind") or "") in {"context_or_blank", "offtarget_table", "unknown"} for row in unique_page_rows):
            flags.append("contains_context_offtarget_or_unknown_page")

        if not matched:
            action = "manual_review_unmatched_current_site_table"
        elif len(sources) > 1:
            action = "split_by_source_then_review"
        elif len(run_ids) == 1 and not incomplete_required and page_order_status != "wrong_current_member_order":
            action = "keep_as_single_strict_physical_group"
        elif collection_ids and len(run_ids) > 1:
            action = "split_into_strict_physical_runs_keep_logical_collection_candidate"
        else:
            action = "split_into_strict_physical_runs"
        if page_order_status == "wrong_current_member_order" and action.startswith("keep"):
            action = "reorder_within_single_strict_physical_group"

        proposed_ids: list[str] = []
        grouped_by_run: defaultdict[str, list[dict[str, object]]] = defaultdict(list)
        for row in unique_page_rows_sorted:
            grouped_by_run[str(row.get("physical_run_id") or "")].append(row)
        for seq, run_id in enumerate([rid for rid in run_ids if rid in grouped_by_run], start=1):
            group_members = sorted(grouped_by_run[run_id], key=page_sort_key)
            page_nums = [int_or_none(row.get("pdf_page")) for row in group_members]
            pages = [p for p in page_nums if p is not None]
            full_run = run_by_id.get(run_id, {})
            full_members = members_by_run.get(run_id, [])
            full_page_ids = {str(row.get("physical_page_id") or "") for row in full_members}
            included_page_ids = {str(row.get("physical_page_id") or "") for row in group_members}
            member_positions = [
                str(row.get("member_position"))
                for row in rows
                if str(row.get("physical_run_id") or "") == run_id and row.get("member_position")
            ]
            image_hashes = [
                str(row.get("image_hash") or "")
                for row in rows
                if str(row.get("physical_run_id") or "") == run_id and row.get("image_hash")
            ]
            group_id = base.stable_id("v24phys", table_id, run_id, ",".join(sorted(included_page_ids)))
            proposed_ids.append(group_id)
            run_collection_ids = collections_by_run.get(run_id, [])
            proposed_rows.append(
                {
                    "proposed_physical_group_id": group_id,
                    "site_table_id": table_id,
                    "group_sequence": seq,
                    "recommended_action": "physical_component_for_static_site",
                    "v6_physical_run_id": run_id,
                    "v6_run_full_page_range": full_run.get("page_range", ""),
                    "included_page_range": compact_page_range(pages),
                    "included_pages_are_consecutive": "yes" if pages_are_consecutive(pages) else "no",
                    "included_page_count": len(group_members),
                    "full_run_page_count": len(full_members),
                    "run_included_completely": "yes" if included_page_ids == full_page_ids else "no",
                    "v5_source_key": group_members[0].get("v5_source_key", "") if group_members else "",
                    "source_pdf": group_members[0].get("source_pdf", "") if group_members else "",
                    "source_pdf_id": group_members[0].get("source_pdf_id", "") if group_members else "",
                    "collection_group_ids": uniq_join(run_collection_ids, limit=12),
                    "page_kind_counts": json.dumps(dict(Counter(str(row.get("page_kind") or "") for row in group_members)), sort_keys=True),
                    "table_family_counts": json.dumps(dict(Counter(str(row.get("table_family_key") or "") for row in group_members)), sort_keys=True),
                    "primary_title": full_run.get("primary_title", "") or (group_members[0].get("title", "") if group_members else ""),
                    "included_member_positions": uniq_join(member_positions, limit=60),
                    "included_image_hashes": uniq_join(image_hashes, limit=60),
                    "example_images": uniq_join([row.get("image_path", "") for row in group_members], limit=8),
                    "review_flags": uniq_join(
                        [
                            "incomplete_required_run" if included_page_ids != full_page_ids and len(full_page_ids) > 1 else "",
                            "nonconsecutive_included_pages" if not pages_are_consecutive(pages) else "",
                            full_run.get("run_review_flags", ""),
                        ],
                        limit=30,
                    ),
                }
            )

        for collection_id in collection_ids:
            collection = collections_by_id.get(collection_id, {})
            site_runs = [run_id for run_id in run_ids if collection_id in collections_by_run.get(str(run_id), [])]
            if len(site_runs) <= 1:
                continue
            logical_rows.append(
                {
                    "site_table_id": table_id,
                    "collection_group_id": collection_id,
                    "collection_basis": collection.get("collection_basis", ""),
                    "n_site_physical_runs_in_collection": len(site_runs),
                    "n_collection_physical_runs_total": len(set(runs_by_collection.get(collection_id, []))),
                    "site_run_ids": uniq_join(site_runs, limit=80),
                    "collection_page_ranges": collection.get("page_ranges", ""),
                    "collection_review_flags": collection.get("collection_review_flags", ""),
                    "primary_title": collection.get("primary_title", ""),
                    "recommended_action": "logical_group_candidate_only_not_physical_merge",
                }
            )

        audit_rows.append(
            {
                "site_table_id": table_id,
                "status": cat.get("status", ""),
                "year": cat.get("year", ""),
                "canonical_name": cat.get("canonical_name", ""),
                "canonical_topic": cat.get("canonical_topic", ""),
                "title": cat.get("title", ""),
                "n_current_member_refs": len(rows),
                "n_member_hashes": len({row.get("image_hash") for row in rows if row.get("image_hash")}),
                "n_matched_member_refs": len(matched),
                "n_unmatched_member_refs": sum(1 for row in rows if row.get("match_status") == "unmatched_hash"),
                "n_ambiguous_member_refs": sum(1 for row in rows if "ambiguous" in str(row.get("match_status") or "")),
                "n_unique_matched_pages": len(unique_page_rows),
                "n_distinct_sources": len(sources),
                "n_v6_physical_runs": len(run_ids),
                "n_v6_collection_groups": len(collection_ids),
                "n_proposed_physical_groups": len(proposed_ids),
                "current_member_pdf_order": " | ".join(member_order[:120]),
                "sorted_pdf_order": " | ".join(sorted_order[:120]),
                "page_order_status": page_order_status,
                "page_gap_edges": "; ".join(gap_edges),
                "incomplete_required_run_ids": uniq_join(incomplete_required, limit=50),
                "logical_collection_candidate_ids": uniq_join(collection_ids, limit=40),
                "issue_flags": uniq_join(flags, limit=40),
                "recommended_action": action,
                "proposed_physical_group_ids": uniq_join(proposed_ids, limit=80),
                "example_images": uniq_join([row.get("image_path", "") for row in unique_page_rows_sorted], limit=8),
            }
        )

    return audit_rows, proposed_rows, logical_rows


def validate(
    catalog: list[dict[str, str]],
    site_members: list[dict[str, object]],
    proposed: list[dict[str, object]],
    audit_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    checks: list[tuple[str, list[object]]] = []
    proposed_bad_nonconsecutive = [
        row.get("proposed_physical_group_id")
        for row in proposed
        if int(row.get("included_page_count") or 0) > 1 and row.get("included_pages_are_consecutive") != "yes"
    ]
    checks.append(("proposed_physical_groups_are_internally_consecutive", proposed_bad_nonconsecutive))
    proposed_bad_cross_run = [
        row.get("proposed_physical_group_id")
        for row in proposed
        if not row.get("v6_physical_run_id")
    ]
    checks.append(("proposed_physical_groups_reference_one_v6_run", proposed_bad_cross_run))
    auto_oversized = [
        row.get("site_table_id")
        for row in audit_rows
        if row.get("recommended_action") == "keep_as_single_strict_physical_group" and int(row.get("n_v6_physical_runs") or 0) != 1
    ]
    checks.append(("keep_recommendations_have_exactly_one_strict_run", auto_oversized))
    missing_catalog_rows = sorted(
        {row["table_id"] for row in catalog} - {str(row.get("site_table_id") or "") for row in audit_rows}
    )
    checks.append(("all_catalog_tables_audited", missing_catalog_rows))
    matched_without_proposal = [
        row.get("site_table_id")
        for row in audit_rows
        if int(row.get("n_unique_matched_pages") or 0) > 0 and int(row.get("n_proposed_physical_groups") or 0) == 0
    ]
    checks.append(("matched_tables_have_proposed_physical_groups", matched_without_proposal))
    duplicate_member_rows = len(site_members) - len({(row.get("site_table_id"), row.get("member_position")) for row in site_members})
    checks.append(("site_member_rows_have_unique_table_position", ["duplicates"] * duplicate_member_rows))
    return [
        {
            "check": check,
            "status": "pass" if not failures else "fail",
            "failure_count": len(failures),
            "details": json.dumps(failures[:30], sort_keys=True),
        }
        for check, failures in checks
    ]


def write_workbook(path: Path, sheets: dict[str, tuple[list[dict[str, object]], list[str]]]) -> None:
    workbook = xlsxwriter.Workbook(str(path))
    header = workbook.add_format({"bold": True, "bg_color": "#D9EAF7", "border": 1})
    wrap = workbook.add_format({"text_wrap": True, "valign": "top"})
    for sheet_name, (rows, fields) in sheets.items():
        ws = workbook.add_worksheet(sheet_name[:31])
        for col, field in enumerate(fields):
            ws.write(0, col, field, header)
        for r_idx, row in enumerate(rows, start=1):
            for c_idx, field in enumerate(fields):
                value = row.get(field, "")
                ws.write(r_idx, c_idx, value, wrap if field in {"title", "issue_flags", "recommended_action", "example_images", "current_member_pdf_order", "sorted_pdf_order"} or "reason" in field or "flags" in field else None)
        ws.freeze_panes(1, 0)
        ws.autofilter(0, 0, max(1, len(rows)), max(0, len(fields) - 1))
        for col, field in enumerate(fields):
            width = 18
            if field in {"title", "issue_flags", "recommended_action"} or "flags" in field:
                width = 42
            elif "image" in field or "order" in field or field.endswith("ids"):
                width = 56
            ws.set_column(col, col, width)
    workbook.close()


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    catalog, member_map = load_site_catalog()
    pages = read_csv(V6_PAGES)
    runs = read_csv(V6_RUNS)
    run_members = read_csv(V6_RUN_MEMBERS)
    collections = read_csv(V6_COLLECTIONS)
    collection_members = read_csv(V6_COLLECTION_MEMBERS)
    hash_index = build_hash_index(pages)
    run_by_id, members_by_run, collections_by_run, collections_by_id, runs_by_collection = build_run_indexes(
        runs, run_members, collections, collection_members
    )
    member_rows = site_member_rows(catalog, member_map, hash_index, collections_by_run)
    audit_rows, proposed_rows, logical_rows = audit_site_tables(
        catalog,
        member_rows,
        run_by_id,
        members_by_run,
        collections_by_run,
        collections_by_id,
        runs_by_collection,
    )
    audit_checks = validate(catalog, member_rows, proposed_rows, audit_rows)
    summary = {
        "v24_site_tables_audited": len(audit_rows),
        "v24_site_member_rows": len(member_rows),
        "v24_site_member_refs_matched": sum(1 for row in member_rows if row.get("physical_page_id")),
        "v24_site_member_refs_unmatched": sum(1 for row in member_rows if row.get("match_status") == "unmatched_hash"),
        "v24_site_member_refs_ambiguous": sum(1 for row in member_rows if "ambiguous" in str(row.get("match_status") or "")),
        "v24_proposed_strict_physical_groups": len(proposed_rows),
        "v24_logical_collection_candidates_for_site": len(logical_rows),
        "v24_recommended_action_counts": dict(Counter(str(row.get("recommended_action") or "") for row in audit_rows)),
        "v24_issue_flag_counts": dict(Counter(flag.strip() for row in audit_rows for flag in str(row.get("issue_flags") or "").split(";") if flag.strip())),
        "v24_audit_failure_count": sum(int(row["failure_count"]) for row in audit_checks),
        "v24_audit_status": "pass" if all(row["status"] == "pass" for row in audit_checks) else "fail",
    }

    summary_rows = [
        {"metric": key, "value": json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value}
        for key, value in summary.items()
    ]

    write_csv(OUT_DIR / "v24_current_static_site_member_map.csv", member_rows, SITE_MEMBER_FIELDS)
    write_csv(OUT_DIR / "v24_current_static_site_group_audit.csv", audit_rows, AUDIT_FIELDS)
    write_csv(OUT_DIR / "v24_proposed_static_physical_groups.csv", proposed_rows, PROPOSED_FIELDS)
    write_csv(OUT_DIR / "v24_static_logical_collection_candidates.csv", logical_rows, LOGICAL_FIELDS)
    write_csv(OUT_DIR / "v24_audit_checks.csv", audit_checks, AUDIT_CHECK_FIELDS)
    write_csv(OUT_DIR / "v24_summary.csv", summary_rows, SUMMARY_FIELDS)
    with (OUT_DIR / "v24_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    workbook = OUT_DIR / "v24_static_site_first_principles_assembly.xlsx"
    write_workbook(
        workbook,
        {
            "summary": (summary_rows, SUMMARY_FIELDS),
            "site_group_audit": (audit_rows, AUDIT_FIELDS),
            "proposed_physical_groups": (proposed_rows, PROPOSED_FIELDS),
            "member_map": (member_rows, SITE_MEMBER_FIELDS),
            "logical_candidates": (logical_rows, LOGICAL_FIELDS),
            "audit_checks": (audit_checks, AUDIT_CHECK_FIELDS),
        },
    )
    with zipfile.ZipFile(workbook) as zf:
        summary["v24_workbook_zip_bad_member"] = zf.testzip()
    with (OUT_DIR / "v24_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
