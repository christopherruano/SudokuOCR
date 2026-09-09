#!/usr/bin/env python3
"""Publish logical-only family collections, with religion sibling focus.

This makes the non-physical "same family / same table" layer explicit. It is
for grouping related subtables for navigation/review, not for row or page
stitching. Every output row keeps physical_merge_allowed=no.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import xlsxwriter


ROOT = Path(__file__).resolve().parent
DEFAULT_ASSEMBLY_DIR = ROOT / "export_final/pdf_recovery_incoming_2026-06-16_first_principles_assembly_pipeline_v29_logical_review_context"
DEFAULT_OUT_DIR = ROOT / "export_final/pdf_recovery_incoming_2026-06-16_first_principles_logical_family_frontier_v32"

RELIGION_PATTERNS = [
    ("all_religions", r"\ball religions?\b"),
    ("hindu", r"\bhindus?\b|\bhindu\b"),
    (
        "muslim",
        r"\bmuslims?\b|\bmusalman(?:s)?\b|\bmussalman(?:s)?\b|\bmussulman(?:s)?\b|"
        r"\bmuhammadan(?:s)?\b|\bmohammadan(?:s)?\b|\bmohammedan(?:s)?\b",
    ),
    ("christian", r"\bchristians?\b|\bchristian\b"),
    ("jew", r"\bjews?\b|\bjewish\b|\bjew\b"),
    ("parsi", r"\bparsis?\b|\bparsi\b"),
    ("sikh", r"\bsikhs?\b|\bsikh\b"),
    ("jain", r"\bjains?\b|\bjain\b"),
    ("buddhist", r"\bbuddh?ists?\b|\bbuddhist\b|\bbudhist\b"),
    ("animistic", r"\banimistic\b|\banimist\b|\btribal\b|\baboriginal\b"),
    ("brahmo", r"\bbrahmos?\b|\bbrahmo\b"),
    ("unspecified_religion", r"\bunspecified religion\b|\bother religions?\b"),
]

FAMILY_FIELDS = [
    "family_rank",
    "family_id",
    "frontier_status",
    "action_bucket",
    "review_priority",
    "risk_level",
    "safe_to_use_logical_collection",
    "safe_to_physical_merge",
    "physical_merge_allowed",
    "logical_collection_only",
    "assembly_group_id",
    "logical_grouping_decision",
    "logical_grouping_confidence",
    "subgroup",
    "segment_kind",
    "segment_variant",
    "v5_source_key",
    "source_pdf",
    "source_pdf_id",
    "page_ranges",
    "member_pages",
    "n_physical_pages",
    "member_count",
    "member_rows_found",
    "pages_are_consecutive",
    "bounded_candidate",
    "core_table_vii_signal",
    "age_civil_signal",
    "religion_family_signal",
    "religion_terms_detected",
    "religion_term_count",
    "variant_terms_sequence",
    "variant_category_sequence",
    "title_table_numbers",
    "title_table_number_status",
    "image_evidence_status",
    "missing_image_member_count",
    "page_only_member_count",
    "primary_title",
    "member_title_sample",
    "decision_reason",
    "review_flags",
    "recommended_action",
]

MEMBER_FIELDS = [
    "family_id",
    "family_rank",
    "member_rank",
    "frontier_status",
    "assembly_group_id",
    "v5_source_key",
    "source_pdf",
    "source_pdf_id",
    "collection_page_ranges",
    "member_position",
    "pdf_page",
    "page_is_consecutive_with_previous",
    "title",
    "religion_terms_detected",
    "variant_terms",
    "variant_category_from_previous",
    "segment_variant",
    "image_path",
    "image_path_exists",
    "member_source",
    "physical_merge_allowed",
    "logical_collection_only",
    "review_flags",
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


def semi(values: Iterable[object]) -> str:
    return "; ".join(str(value) for value in values if value not in {None, ""})


def stable_family_id(status: str, group_id: str) -> str:
    prefix = "v32fam_confirmed" if status == "confirmed_logical_collection" else "v32fam_review"
    return f"{prefix}_{group_id}"


def religion_terms(text: str) -> list[str]:
    found: list[str] = []
    lowered = text.lower()
    for term, pattern in RELIGION_PATTERNS:
        if re.search(pattern, lowered):
            found.append(term)
    return sorted(set(found))


def parse_pages(value: str) -> list[int]:
    return [int(part) for part in re.findall(r"\d+", str(value or ""))]


def pages_consecutive(value: str) -> str:
    pages = parse_pages(value)
    if len(pages) <= 1:
        return "yes" if pages else ""
    return "yes" if pages == list(range(pages[0], pages[-1] + 1)) else "no"


def member_map(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row.get("assembly_group_id", "")].append(row)
    for group_rows in grouped.values():
        group_rows.sort(key=lambda row: int(row.get("member_rank") or row.get("member_position") or 0))
    return grouped


def action_bucket(row: dict[str, object]) -> str:
    status = str(row.get("frontier_status") or "")
    religion = str(row.get("religion_family_signal") or "")
    priority = str(row.get("review_priority") or "")
    safe = str(row.get("safe_to_use_logical_collection") or "")
    risk = str(row.get("risk_level") or "")
    if status == "confirmed_logical_collection" and safe == "yes" and risk != "high_review_risk":
        return "use_now_logical_family_collection"
    if status == "confirmed_logical_collection":
        return "review_confirmed_logical_family_before_site_use"
    if religion == "yes" and priority == "high_bounded_core_table_review":
        return "high_priority_review_religion_table_family"
    if religion == "yes":
        return "review_religion_table_family_candidate"
    if priority:
        return priority
    return "review_logical_family_candidate"


def build_family_rows(assembly_dir: Path) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    confirmed_context = read_csv(assembly_dir / "logical_collection_context_review.csv")
    review_context = read_csv(assembly_dir / "logical_review_context.csv")
    confirmed_members = member_map(read_csv(assembly_dir / "logical_collection_members.csv"))
    review_members = member_map(read_csv(assembly_dir / "logical_review_candidate_members.csv"))

    families: list[dict[str, object]] = []
    members_out: list[dict[str, object]] = []

    context_specs = [
        ("confirmed_logical_collection", confirmed_context, confirmed_members),
        ("review_logical_candidate", review_context, review_members),
    ]
    for status, contexts, grouped_members in context_specs:
        for context in contexts:
            group_id = context.get("assembly_group_id", "")
            group_members = grouped_members.get(group_id, [])
            text_parts = [
                context.get("primary_title", ""),
                context.get("member_titles", ""),
                context.get("variant_terms_sequence", ""),
                context.get("segment_variant", ""),
                context.get("subgroup", ""),
                context.get("logical_grouping_reason", ""),
            ]
            text_parts.extend(member.get("title", "") for member in group_members)
            text_parts.extend(member.get("variant_terms", "") for member in group_members)
            terms = religion_terms(" ".join(text_parts))
            family_id = stable_family_id(status, group_id)
            page_ranges = context.get("page_ranges", "")
            member_pages = context.get("member_pages", page_ranges)
            title_sample = semi(member.get("title", "") for member in group_members[:4])
            family = {
                "family_id": family_id,
                "frontier_status": status,
                "review_priority": context.get("review_priority", ""),
                "risk_level": context.get("risk_level", ""),
                "safe_to_use_logical_collection": context.get("safe_to_use_logical_collection", "yes" if status == "confirmed_logical_collection" else "no"),
                "safe_to_physical_merge": "no",
                "physical_merge_allowed": "no",
                "logical_collection_only": "yes",
                "assembly_group_id": group_id,
                "logical_grouping_decision": context.get("logical_grouping_decision", ""),
                "logical_grouping_confidence": context.get("logical_grouping_confidence", ""),
                "subgroup": context.get("subgroup", ""),
                "segment_kind": context.get("segment_kind", ""),
                "segment_variant": context.get("segment_variant", ""),
                "v5_source_key": context.get("v5_source_key", ""),
                "source_pdf": context.get("source_pdf", ""),
                "source_pdf_id": context.get("source_pdf_id", ""),
                "page_ranges": page_ranges,
                "member_pages": member_pages,
                "n_physical_pages": context.get("n_physical_pages", context.get("n_collection_pages", "")),
                "member_count": context.get("member_count", ""),
                "member_rows_found": len(group_members),
                "pages_are_consecutive": context.get("pages_are_consecutive", pages_consecutive(member_pages)),
                "bounded_candidate": context.get("bounded_candidate", "yes" if status == "confirmed_logical_collection" else ""),
                "core_table_vii_signal": context.get("core_table_vii_signal", ""),
                "age_civil_signal": context.get("age_civil_signal", ""),
                "religion_family_signal": "yes" if terms else "no",
                "religion_terms_detected": semi(terms),
                "religion_term_count": len(terms),
                "variant_terms_sequence": context.get("variant_terms_sequence", ""),
                "variant_category_sequence": context.get("variant_category_sequence", ""),
                "title_table_numbers": context.get("title_table_numbers", ""),
                "title_table_number_status": context.get("title_table_number_status", ""),
                "image_evidence_status": context.get("image_evidence_status", ""),
                "missing_image_member_count": context.get("missing_image_member_count", ""),
                "page_only_member_count": context.get("page_only_member_count", ""),
                "primary_title": context.get("primary_title", ""),
                "member_title_sample": title_sample,
                "decision_reason": context.get("review_reason", context.get("logical_grouping_reason", "")),
                "review_flags": context.get("risk_flags", context.get("logical_grouping_review_flags", "")),
                "recommended_action": context.get("recommended_action", context.get("recommended_logical_action", "")),
            }
            family["action_bucket"] = action_bucket(family)
            families.append(family)
            for member in group_members:
                member_text = " ".join([member.get("title", ""), member.get("variant_terms", ""), member.get("segment_variant", "")])
                members_out.append(
                    {
                        "family_id": family_id,
                        "member_rank": member.get("member_rank", ""),
                        "frontier_status": status,
                        "assembly_group_id": group_id,
                        "v5_source_key": member.get("v5_source_key", context.get("v5_source_key", "")),
                        "source_pdf": member.get("source_pdf", context.get("source_pdf", "")),
                        "source_pdf_id": member.get("source_pdf_id", context.get("source_pdf_id", "")),
                        "collection_page_ranges": member.get("collection_page_ranges", page_ranges),
                        "member_position": member.get("member_position", ""),
                        "pdf_page": member.get("pdf_page", ""),
                        "page_is_consecutive_with_previous": member.get("page_is_consecutive_with_previous", ""),
                        "title": member.get("title", ""),
                        "religion_terms_detected": semi(religion_terms(member_text)),
                        "variant_terms": member.get("variant_terms", ""),
                        "variant_category_from_previous": member.get("variant_category_from_previous", ""),
                        "segment_variant": member.get("segment_variant", context.get("segment_variant", "")),
                        "image_path": member.get("image_path", ""),
                        "image_path_exists": member.get("image_path_exists", ""),
                        "member_source": member.get("member_source", ""),
                        "physical_merge_allowed": "no",
                        "logical_collection_only": "yes",
                        "review_flags": member.get("review_flags", ""),
                    }
                )

    families.sort(
        key=lambda row: (
            0 if row.get("frontier_status") == "confirmed_logical_collection" else 1,
            0 if row.get("religion_family_signal") == "yes" else 1,
            str(row.get("action_bucket", "")),
            str(row.get("v5_source_key", "")),
            str(row.get("page_ranges", "")),
        )
    )
    rank_by_family: dict[str, int] = {}
    for idx, row in enumerate(families, start=1):
        row["family_rank"] = idx
        rank_by_family[str(row.get("family_id"))] = idx
    for row in members_out:
        row["family_rank"] = rank_by_family.get(str(row.get("family_id")), "")
    members_out.sort(key=lambda row: (int(row.get("family_rank") or 10**9), int(row.get("member_rank") or 10**9)))
    return families, members_out


def build_summary(families: list[dict[str, object]], members: list[dict[str, object]], assembly_dir: Path, out_dir: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = [
        {"metric": "paths", "name": "assembly_dir", "count": output_path(assembly_dir)},
        {"metric": "paths", "name": "out_dir", "count": output_path(out_dir)},
        {"metric": "counts", "name": "logical_family_rows", "count": len(families)},
        {"metric": "counts", "name": "logical_family_member_rows", "count": len(members)},
        {"metric": "counts", "name": "religion_family_rows", "count": sum(1 for row in families if row.get("religion_family_signal") == "yes")},
        {"metric": "counts", "name": "confirmed_religion_family_rows", "count": sum(1 for row in families if row.get("religion_family_signal") == "yes" and row.get("frontier_status") == "confirmed_logical_collection")},
        {"metric": "counts", "name": "review_religion_family_rows", "count": sum(1 for row in families if row.get("religion_family_signal") == "yes" and row.get("frontier_status") == "review_logical_candidate")},
    ]
    for metric, counter in [
        ("families_by_status", Counter(str(row.get("frontier_status", "")) for row in families)),
        ("families_by_action", Counter(str(row.get("action_bucket", "")) for row in families)),
        ("families_by_religion_signal", Counter(str(row.get("religion_family_signal", "")) for row in families)),
        ("families_by_terms", Counter(term for row in families for term in str(row.get("religion_terms_detected", "")).split("; ") if term)),
        ("members_by_status", Counter(str(row.get("frontier_status", "")) for row in members)),
        ("members_by_image_exists", Counter(str(row.get("image_path_exists", "")) for row in members)),
    ]:
        for name, count in sorted(counter.items()):
            rows.append({"metric": metric, "name": name, "count": count})
    return rows


def audit_outputs(families: list[dict[str, object]], members: list[dict[str, object]]) -> list[dict[str, object]]:
    audits: list[dict[str, object]] = []
    bad_physical = [
        row for row in families
        if row.get("physical_merge_allowed") == "yes" or row.get("safe_to_physical_merge") == "yes"
    ]
    audits.append(
        {
            "check": "logical_families_never_allow_physical_merge",
            "status": "pass" if not bad_physical else "fail",
            "detail": f"{len(bad_physical)} family rows allow physical merge",
        }
    )
    bad_members = [row for row in members if row.get("physical_merge_allowed") == "yes"]
    audits.append(
        {
            "check": "logical_family_members_never_allow_physical_merge",
            "status": "pass" if not bad_members else "fail",
            "detail": f"{len(bad_members)} member rows allow physical merge",
        }
    )
    confirmed_missing_members = [
        row for row in families
        if row.get("frontier_status") == "confirmed_logical_collection" and int(row.get("member_rows_found") or 0) == 0
    ]
    audits.append(
        {
            "check": "confirmed_logical_families_have_member_rows",
            "status": "pass" if not confirmed_missing_members else "fail",
            "detail": f"{len(confirmed_missing_members)} confirmed logical families have no member rows",
        }
    )
    religion_without_terms = [
        row for row in families
        if row.get("religion_family_signal") == "yes" and not row.get("religion_terms_detected")
    ]
    audits.append(
        {
            "check": "religion_family_rows_have_terms",
            "status": "pass" if not religion_without_terms else "fail",
            "detail": f"{len(religion_without_terms)} religion family rows have no terms",
        }
    )
    return audits


def write_workbook(
    path: Path,
    summary: list[dict[str, object]],
    families: list[dict[str, object]],
    religion_families: list[dict[str, object]],
    members: list[dict[str, object]],
    audit: list[dict[str, object]],
) -> None:
    workbook = xlsxwriter.Workbook(path)
    header_fmt = workbook.add_format({"bold": True, "bg_color": "#E8EEF7", "border": 1})
    wrap_fmt = workbook.add_format({"text_wrap": True, "valign": "top"})
    sheets = [
        ("Summary", summary, SUMMARY_FIELDS),
        ("Logical Families", families, FAMILY_FIELDS),
        ("Religion Families", religion_families, FAMILY_FIELDS),
        ("Family Members", members, MEMBER_FIELDS),
        ("Audit", audit, AUDIT_FIELDS),
    ]
    for name, rows, fields in sheets:
        ws = workbook.add_worksheet(name[:31])
        for col, field in enumerate(fields):
            ws.write(0, col, field, header_fmt)
        for row_idx, row in enumerate(rows, start=1):
            for col, field in enumerate(fields):
                value = row.get(field, "")
                fmt = wrap_fmt if any(token in field for token in ["title", "reason", "flags", "path", "action", "sequence"]) else None
                ws.write(row_idx, col, value, fmt)
        ws.freeze_panes(1, 0)
        ws.autofilter(0, 0, max(len(rows), 1), max(len(fields) - 1, 0))
        for col, field in enumerate(fields):
            ws.set_column(col, col, min(max(len(field) + 2, 12), 60))
    workbook.close()


def validate_workbook(path: Path) -> str | None:
    try:
        with zipfile.ZipFile(path) as zf:
            return zf.testzip()
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"


def build_logical_family_frontier(assembly_dir: Path, out_dir: Path) -> dict[str, object]:
    families, members = build_family_rows(assembly_dir)
    religion_families = [row for row in families if row.get("religion_family_signal") == "yes"]
    summary = build_summary(families, members, assembly_dir, out_dir)
    audit = audit_outputs(families, members)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "logical_family_frontier.csv", families, FAMILY_FIELDS)
    write_csv(out_dir / "religion_family_frontier.csv", religion_families, FAMILY_FIELDS)
    write_csv(out_dir / "logical_family_members.csv", members, MEMBER_FIELDS)
    write_csv(out_dir / "summary.csv", summary, SUMMARY_FIELDS)
    workbook_path = out_dir / "logical_family_frontier_workbook.xlsx"
    workbook_bad_member = validate_workbook(workbook_path)
    audit.append(
        {
            "check": "logical_family_frontier_workbook_zip_integrity",
            "status": "pass" if workbook_bad_member is None else "fail",
            "detail": workbook_bad_member or "",
        }
    )
    write_csv(out_dir / "audit.csv", audit, AUDIT_FIELDS)
    write_workbook(workbook_path, summary, families, religion_families, members, audit)
    workbook_bad_member = validate_workbook(workbook_path)
    audit[-1] = {
        "check": "logical_family_frontier_workbook_zip_integrity",
        "status": "pass" if workbook_bad_member is None else "fail",
        "detail": workbook_bad_member or "",
    }
    write_csv(out_dir / "audit.csv", audit, AUDIT_FIELDS)
    write_workbook(workbook_path, summary, families, religion_families, members, audit)
    audit_failure_count = sum(1 for row in audit if row.get("status") != "pass")
    result = {
        "assembly_dir": output_path(assembly_dir),
        "out_dir": output_path(out_dir),
        "logical_family_rows": len(families),
        "logical_family_member_rows": len(members),
        "religion_family_rows": len(religion_families),
        "confirmed_religion_family_rows": sum(1 for row in religion_families if row.get("frontier_status") == "confirmed_logical_collection"),
        "review_religion_family_rows": sum(1 for row in religion_families if row.get("frontier_status") == "review_logical_candidate"),
        "families_by_status": dict(Counter(str(row.get("frontier_status", "")) for row in families)),
        "families_by_action": dict(Counter(str(row.get("action_bucket", "")) for row in families)),
        "families_by_religion_terms": dict(Counter(term for row in families for term in str(row.get("religion_terms_detected", "")).split("; ") if term)),
        "audit_failure_count": audit_failure_count,
        "workbook_zip_bad_member": workbook_bad_member,
    }
    (out_dir / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    if audit_failure_count:
        raise SystemExit(f"logical family audit failed: {audit_failure_count} failures")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assembly-dir", default=str(DEFAULT_ASSEMBLY_DIR))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    args = parser.parse_args()
    assembly_dir = Path(args.assembly_dir).expanduser()
    out_dir = Path(args.out_dir).expanduser()
    if not assembly_dir.is_absolute():
        assembly_dir = ROOT / assembly_dir
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    result = build_logical_family_frontier(assembly_dir, out_dir)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
