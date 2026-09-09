#!/usr/bin/env python3
"""Build logical-only sibling sets across bounded religion/table collections.

This layer performs no OCR/API work and never creates physical merges. It groups
bounded logical collections into higher-level sibling sets only when they are
consecutive, same-source, same-table, same-topic runs. This is for cases like
adjacent Muslim/Christian/Jew/Hindu variants of the same table family.
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
DEFAULT_LOGICAL_EVIDENCE_DIR = ROOT / "export_final/pdf_recovery_incoming_2026-06-16_first_principles_logical_collection_evidence_smoke_v44_adjudication"
DEFAULT_OUT_DIR = ROOT / "export_final/pdf_recovery_incoming_2026-06-16_first_principles_logical_sibling_sets_v45"

MAX_SITE_READY_PAGES = 4
REVIEW_RISK_BUCKETS = {
    "large_review_family_overbreadth_risk",
    "single_variant_large_review_family",
    "possible_missing_religion_family",
}

SET_FIELDS = [
    "sibling_rank",
    "sibling_set_id",
    "sibling_set_status",
    "site_use_policy",
    "logical_collection_only",
    "physical_merge_allowed",
    "safe_to_publish_values",
    "review_required",
    "priority",
    "recommended_action",
    "review_reason",
    "v5_source_key",
    "source_pdf_id",
    "parent_family_id",
    "parent_audit_bucket",
    "parent_statuses",
    "source_kinds",
    "page_range",
    "component_pages",
    "n_pages",
    "pages_are_consecutive",
    "collection_count",
    "member_page_count",
    "table_number_key",
    "topic_variant_key",
    "religion_terms_detected",
    "distinct_religion_count",
    "collection_ids",
    "evidence_buckets",
    "title_table_refs",
    "nonmatching_title_table_refs",
    "title_sample",
]

MEMBER_FIELDS = [
    "member_rank",
    "sibling_set_id",
    "logical_collection_id",
    "collection_position",
    "v5_source_key",
    "source_pdf_id",
    "parent_family_id",
    "pdf_page",
    "expanded_pages",
    "table_number_key",
    "topic_variant_key",
    "religion_terms_detected",
    "title",
    "image_path",
]

AUDIT_FIELDS = ["check", "status", "detail"]
SUMMARY_FIELDS = ["metric", "name", "count"]


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
    blob = "||".join(str(part or "") for part in parts)
    return f"{prefix}_{hashlib.sha1(blob.encode('utf-8')).hexdigest()[:16]}"


def int_or_none(value: object) -> int | None:
    if value in {None, ""}:
        return None
    try:
        return int(str(value).replace(",", ""))
    except ValueError:
        return None


def expand_pages(value: object) -> list[int]:
    pages: list[int] = []
    for part in re.split(r"[;,]\s*", str(value or "").strip()):
        part = part.strip()
        if not part:
            continue
        match = re.fullmatch(r"(\d+)\s*-\s*(\d+)", part)
        if match:
            start, end = int(match.group(1)), int(match.group(2))
            if start <= end and end - start <= 500:
                pages.extend(range(start, end + 1))
            continue
        page = int_or_none(part)
        if page is not None:
            pages.append(page)
    return pages


def compact_pages(pages: Iterable[int]) -> str:
    values = sorted(set(pages))
    if not values:
        return ""
    parts: list[str] = []
    start = prev = values[0]
    for page in values[1:]:
        if page == prev + 1:
            prev = page
            continue
        parts.append(str(start) if start == prev else f"{start}-{prev}")
        start = prev = page
    parts.append(str(start) if start == prev else f"{start}-{prev}")
    return "; ".join(parts)


def pages_consecutive(pages: Iterable[int]) -> bool:
    values = sorted(set(pages))
    return bool(values) and values == list(range(values[0], values[-1] + 1))


def split_semicolon(value: object) -> list[str]:
    return [part.strip() for part in str(value or "").split(";") if part.strip()]


def semi(values: Iterable[object]) -> str:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    return "; ".join(out)


def first_nonempty(values: Iterable[object]) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""


ROMAN_TABLES = {
    "I": "1",
    "II": "2",
    "III": "3",
    "IV": "4",
    "V": "5",
    "VI": "6",
    "VII": "7",
    "VIII": "8",
    "IX": "9",
    "X": "10",
    "XI": "11",
    "XII": "12",
    "XIII": "13",
    "XIV": "14",
    "XV": "15",
    "XVI": "16",
    "XVII": "17",
    "XVIII": "18",
    "XIX": "19",
    "XX": "20",
}
TABLE_REF_RE = re.compile(
    r"(?:\bTABLE\b|No\.?|NO\.?|age_[a-z_]+)\s*(?:[-–—: ]+)?(?:No\.?\s*)?"
    r"(X(?:VIII|VII|VI|IV|IX|V?I{0,3})|IX|IV|V?I{1,3}|[1-9]|1[0-9]|20)\b",
    re.IGNORECASE,
)
NON_CORE_LOGICAL_RE = re.compile(
    r"\b(subsidiary|selected|caste|castes|tribe|tribes|race|races|"
    r"education|scholar|literacy|illiteracy|deaf|mute|mutes|leper|lepers|"
    r"infirmity|infirmities|blind|insane|occupation|house|houses)\b",
    re.IGNORECASE,
)
CORE_LOGICAL_RE = re.compile(
    r"\b(age|ages|religion|religions|civil condition|sex|population|persons|"
    r"unmarried|married|widowed|widow)\b",
    re.IGNORECASE,
)


def normalize_table_ref(value: str) -> str:
    text = value.strip().upper().rstrip(".")
    return ROMAN_TABLES.get(text, text)


def extract_title_table_refs(value: object) -> list[str]:
    refs: list[str] = []
    seen: set[str] = set()
    for match in TABLE_REF_RE.finditer(str(value or "")):
        ref = normalize_table_ref(match.group(1))
        if ref and ref not in seen:
            seen.add(ref)
            refs.append(ref)
    return refs


def has_core_logical_title_evidence(value: object) -> bool:
    text = str(value or "")
    if NON_CORE_LOGICAL_RE.search(text):
        return False
    return bool(CORE_LOGICAL_RE.search(text))


def collection_key(row: dict[str, str]) -> tuple[str, str, str, str, str]:
    return (
        row.get("v5_source_key", ""),
        row.get("source_pdf_id", ""),
        row.get("parent_family_id", ""),
        row.get("table_number_key", ""),
        row.get("topic_variant_key", ""),
    )


def valid_collection(row: dict[str, str]) -> bool:
    return (
        row.get("logical_collection_only") == "yes"
        and row.get("physical_merge_allowed") == "no"
        and row.get("pages_are_consecutive") == "yes"
        and bool(row.get("table_number_key"))
        and row.get("topic_variant_key") != "same_title_unspecified_variant"
        and bool(row.get("religion_terms_detected"))
    )


def build_runs(collections: list[dict[str, str]]) -> list[list[dict[str, object]]]:
    prepared: list[dict[str, object]] = []
    for row in collections:
        if not valid_collection(row):
            continue
        pages = expand_pages(row.get("component_pages") or row.get("page_range"))
        if not pages:
            continue
        item: dict[str, object] = dict(row)
        item["_pages"] = sorted(set(pages))
        item["_min_page"] = min(pages)
        item["_max_page"] = max(pages)
        prepared.append(item)

    grouped: dict[tuple[str, str, str, str, str], list[dict[str, object]]] = defaultdict(list)
    for item in prepared:
        grouped[collection_key(item)].append(item)  # type: ignore[arg-type]

    runs: list[list[dict[str, object]]] = []
    for items in grouped.values():
        items.sort(key=lambda item: (int(item["_min_page"]), str(item.get("logical_collection_id", ""))))
        current: list[dict[str, object]] = []
        current_max: int | None = None
        for item in items:
            min_page = int(item["_min_page"])
            max_page = int(item["_max_page"])
            if not current:
                current = [item]
                current_max = max_page
                continue
            if current_max is not None and min_page <= current_max + 1:
                current.append(item)
                current_max = max(current_max, max_page)
                continue
            runs.append(current)
            current = [item]
            current_max = max_page
        if current:
            runs.append(current)
    return runs


def classify_set(
    items: list[dict[str, object]],
    religions: list[str],
    pages: list[int],
    title_refs: list[str],
    nonmatching_title_refs: list[str],
) -> tuple[str, str, str, str, int, str]:
    parent_buckets = {str(item.get("parent_audit_bucket", "")) for item in items}
    source_kinds = {str(item.get("source_kind", "")) for item in items}
    evidence_buckets = {str(item.get("evidence_bucket", "")) for item in items}
    table = first_nonempty(item.get("table_number_key", "") for item in items)
    topic = first_nonempty(item.get("topic_variant_key", "") for item in items)

    if len(items) < 2 and len(religions) < 2:
        return (
            "single_collection_not_sibling_review",
            "review_before_site_use_logical_only",
            "yes",
            "Single collection with one religion term; keep as logical collection, not a sibling set.",
            6,
            "not_enough_sibling_members",
        )
    if not table or not topic or topic == "same_title_unspecified_variant":
        return (
            "logical_sibling_review_missing_table_or_topic",
            "review_before_site_use_logical_only",
            "yes",
            "Review only; table/topic evidence is incomplete.",
            5,
            "missing_table_or_topic",
        )
    if not pages_consecutive(pages):
        return (
            "logical_sibling_gap_no_merge",
            "review_before_site_use_logical_only",
            "yes",
            "Do not group for site use until missing logical pages are understood.",
            5,
            "nonconsecutive_pages",
        )
    if nonmatching_title_refs:
        return (
            "logical_sibling_review_title_table_conflict",
            "review_before_site_use_logical_only",
            "yes",
            "Review only; normalized table key conflicts with visible title/table-number evidence.",
            3,
            f"title_table_refs_conflict={semi(nonmatching_title_refs)}",
        )
    if not title_refs:
        return (
            "logical_sibling_review_missing_title_table_refs",
            "review_before_site_use_logical_only",
            "yes",
            "Review only; visible title/table-number evidence could not be verified.",
            3,
            "missing_visible_title_table_refs",
        )
    title_text = " ".join(str(item.get("member_title_sample", "")) for item in items)
    if not has_core_logical_title_evidence(title_text):
        return (
            "logical_sibling_review_non_core_title_terms",
            "review_before_site_use_logical_only",
            "yes",
            "Review only; title terms suggest a subsidiary, derivative, or non-core table family.",
            3,
            "non_core_logical_title_terms",
        )
    if parent_buckets & REVIEW_RISK_BUCKETS or "qa_missing_religion_sequence" in source_kinds:
        return (
            "logical_sibling_review_only_risk",
            "review_before_site_use_logical_only",
            "yes",
            "Review sibling relationship; source parent is overbroad, missing-religion, or QA-derived.",
            4,
            semi(sorted((parent_buckets & REVIEW_RISK_BUCKETS) | ({"qa_missing_religion_sequence"} if "qa_missing_religion_sequence" in source_kinds else set()))),
        )
    if len(religions) >= 2 and len(pages) <= MAX_SITE_READY_PAGES and not any("missing_religion" in bucket for bucket in evidence_buckets):
        return (
            "site_ready_logical_sibling_set",
            "use_now_logical_sibling_collection_only",
            "no",
            "Use as logical-only sibling set; never physically merge these pages.",
            1,
            "same_source_parent_table_topic_consecutive_multi_religion",
        )
    if len(religions) >= 2:
        return (
            "logical_sibling_review_multi_religion",
            "review_before_site_use_logical_only",
            "yes",
            "Review multi-religion logical sibling set before site use.",
            3,
            "multi_religion_but_risk_or_size_limit",
        )
    return (
        "logical_sibling_review_single_religion_run",
        "review_before_site_use_logical_only",
        "yes",
        "Single-religion run; useful for navigation but not a complete sibling set.",
        6,
        "single_religion_run",
    )


def build_sets(collections: list[dict[str, str]], members: list[dict[str, str]]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    member_by_collection: dict[str, list[dict[str, str]]] = defaultdict(list)
    for member in members:
        member_by_collection[member.get("logical_collection_id", "")].append(member)

    set_rows: list[dict[str, object]] = []
    member_rows: list[dict[str, object]] = []
    runs = build_runs(collections)
    for run in runs:
        collection_ids = [str(item.get("logical_collection_id", "")) for item in run]
        pages = sorted({page for item in run for page in item.get("_pages", [])})  # type: ignore[union-attr]
        religions = sorted({term for item in run for term in split_semicolon(item.get("religion_terms_detected", ""))})
        table_key = first_nonempty(item.get("table_number_key", "") for item in run)
        title_sample = semi(str(item.get("member_title_sample", "")) for item in run)[:1000]
        title_refs = sorted({ref for item in run for ref in extract_title_table_refs(item.get("member_title_sample", ""))})
        nonmatching_title_refs = [ref for ref in title_refs if table_key and ref != str(table_key)]
        status, policy, review_required, action, priority, review_reason = classify_set(run, religions, pages, title_refs, nonmatching_title_refs)
        sibling_id = stable_id("v45sib", collection_key(run[0]), compact_pages(pages), semi(collection_ids))
        set_rows.append(
            {
                "sibling_set_id": sibling_id,
                "sibling_set_status": status,
                "site_use_policy": policy,
                "logical_collection_only": "yes",
                "physical_merge_allowed": "no",
                "safe_to_publish_values": "no",
                "review_required": review_required,
                "priority": priority,
                "recommended_action": action,
                "review_reason": review_reason,
                "v5_source_key": run[0].get("v5_source_key", ""),
                "source_pdf_id": run[0].get("source_pdf_id", ""),
                "parent_family_id": run[0].get("parent_family_id", ""),
                "parent_audit_bucket": semi(item.get("parent_audit_bucket", "") for item in run),
                "parent_statuses": semi(item.get("parent_status", "") for item in run),
                "source_kinds": semi(item.get("source_kind", "") for item in run),
                "page_range": compact_pages(pages),
                "component_pages": semi(item.get("component_pages", "") for item in run),
                "n_pages": len(pages),
                "pages_are_consecutive": "yes" if pages_consecutive(pages) else "no",
                "collection_count": len(run),
                "member_page_count": sum(len(member_by_collection.get(cid, [])) for cid in collection_ids),
                "table_number_key": run[0].get("table_number_key", ""),
                "topic_variant_key": run[0].get("topic_variant_key", ""),
                "religion_terms_detected": semi(religions),
                "distinct_religion_count": len(religions),
                "collection_ids": semi(collection_ids),
                "evidence_buckets": semi(item.get("evidence_bucket", "") for item in run),
                "title_table_refs": semi(title_refs),
                "nonmatching_title_table_refs": semi(nonmatching_title_refs),
                "title_sample": title_sample,
            }
        )
        collection_position = 0
        for item in run:
            collection_position += 1
            collection_id = str(item.get("logical_collection_id", ""))
            collection_members = member_by_collection.get(collection_id, [])
            if not collection_members:
                member_rows.append(
                    {
                        "sibling_set_id": sibling_id,
                        "logical_collection_id": collection_id,
                        "collection_position": collection_position,
                        "v5_source_key": item.get("v5_source_key", ""),
                        "source_pdf_id": item.get("source_pdf_id", ""),
                        "parent_family_id": item.get("parent_family_id", ""),
                        "pdf_page": item.get("page_range", ""),
                        "expanded_pages": item.get("component_pages", ""),
                        "table_number_key": item.get("table_number_key", ""),
                        "topic_variant_key": item.get("topic_variant_key", ""),
                        "religion_terms_detected": item.get("religion_terms_detected", ""),
                        "title": item.get("member_title_sample", ""),
                        "image_path": "",
                    }
                )
                continue
            for member in collection_members:
                member_rows.append(
                    {
                        "sibling_set_id": sibling_id,
                        "logical_collection_id": collection_id,
                        "collection_position": collection_position,
                        "v5_source_key": member.get("v5_source_key", ""),
                        "source_pdf_id": member.get("source_pdf_id", ""),
                        "parent_family_id": member.get("parent_family_id", ""),
                        "pdf_page": member.get("pdf_page", ""),
                        "expanded_pages": member.get("expanded_pages", ""),
                        "table_number_key": member.get("table_number_key", ""),
                        "topic_variant_key": member.get("topic_variant_key", ""),
                        "religion_terms_detected": member.get("religion_terms_detected", ""),
                        "title": member.get("title", ""),
                        "image_path": member.get("image_path", ""),
                    }
                )

    set_rows.sort(
        key=lambda row: (
            int_or_none(row.get("priority")) or 999,
            row.get("v5_source_key", ""),
            int_or_none(str(row.get("page_range", "")).split("-")[0].split(";")[0]) or 999999,
            row.get("sibling_set_id", ""),
        )
    )
    rank_by_id: dict[str, int] = {}
    for idx, row in enumerate(set_rows, start=1):
        row["sibling_rank"] = idx
        rank_by_id[str(row.get("sibling_set_id", ""))] = idx
    member_rows.sort(
        key=lambda row: (
            rank_by_id.get(str(row.get("sibling_set_id", "")), 999999),
            int_or_none(row.get("collection_position")) or 999,
            int_or_none(row.get("pdf_page")) or 999999,
            row.get("logical_collection_id", ""),
        )
    )
    for idx, row in enumerate(member_rows, start=1):
        row["member_rank"] = idx
    return set_rows, member_rows


def write_workbook(
    path: Path,
    summary_rows: list[dict[str, object]],
    audits: list[dict[str, object]],
    set_rows: list[dict[str, object]],
    member_rows: list[dict[str, object]],
    site_ready_rows: list[dict[str, object]],
    review_rows: list[dict[str, object]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    workbook = xlsxwriter.Workbook(path)
    header = workbook.add_format({"bold": True, "bg_color": "#DCE6D5", "border": 1})
    wrap = workbook.add_format({"text_wrap": True, "valign": "top"})
    sheets = [
        ("Summary", summary_rows, SUMMARY_FIELDS),
        ("Audit", audits, AUDIT_FIELDS),
        ("Sibling Sets", set_rows, SET_FIELDS),
        ("Site Ready Sets", site_ready_rows, SET_FIELDS),
        ("Review Sets", review_rows, SET_FIELDS),
        ("Members", member_rows, MEMBER_FIELDS),
    ]
    for name, rows, fields in sheets:
        ws = workbook.add_worksheet(name[:31])
        for col, field in enumerate(fields):
            ws.write(0, col, field, header)
        for r_idx, row in enumerate(rows, start=1):
            for c_idx, field in enumerate(fields):
                fmt = wrap if field in {
                    "recommended_action",
                    "review_reason",
                    "title_sample",
                    "title",
                    "collection_ids",
                    "component_pages",
                    "image_path",
                } else None
                ws.write(r_idx, c_idx, row.get(field, ""), fmt)
        ws.freeze_panes(1, 0)
        ws.autofilter(0, 0, max(len(rows), 1), max(len(fields) - 1, 0))
        for col, field in enumerate(fields):
            width = 16
            if field in {"sibling_set_id", "parent_family_id", "logical_collection_id"}:
                width = 34
            elif field in {"recommended_action", "review_reason", "title_sample", "title"}:
                width = 52
            elif field in {"collection_ids", "image_path"}:
                width = 48
            ws.set_column(col, col, width)
    workbook.close()


def build_audits(set_rows: list[dict[str, object]], member_rows: list[dict[str, object]], workbook_path: Path) -> list[dict[str, object]]:
    audits: list[dict[str, object]] = []
    physical_bad = [row for row in set_rows if row.get("physical_merge_allowed") != "no" or row.get("logical_collection_only") != "yes"]
    audits.append({"check": "sibling_sets_are_logical_only", "status": "pass" if not physical_bad else "fail", "detail": f"bad={len(physical_bad)}"})
    nonconsecutive_site = [row for row in set_rows if row.get("site_use_policy") == "use_now_logical_sibling_collection_only" and row.get("pages_are_consecutive") != "yes"]
    audits.append({"check": "site_ready_sibling_sets_are_consecutive", "status": "pass" if not nonconsecutive_site else "fail", "detail": f"bad={len(nonconsecutive_site)}"})
    weak_site = [
        row for row in set_rows
        if row.get("site_use_policy") == "use_now_logical_sibling_collection_only"
        and (
            int_or_none(row.get("distinct_religion_count")) is None
            or (int_or_none(row.get("distinct_religion_count")) or 0) < 2
            or not row.get("table_number_key")
            or row.get("topic_variant_key") == "same_title_unspecified_variant"
            or (int_or_none(row.get("n_pages")) or 0) > MAX_SITE_READY_PAGES
        )
    ]
    audits.append({"check": "site_ready_sibling_sets_have_strong_logical_evidence", "status": "pass" if not weak_site else "fail", "detail": f"bad={len(weak_site)}"})
    risky_site = [row for row in set_rows if row.get("site_use_policy") == "use_now_logical_sibling_collection_only" and row.get("parent_audit_bucket") in REVIEW_RISK_BUCKETS]
    audits.append({"check": "site_ready_sibling_sets_exclude_overbroad_or_missing_religion_parents", "status": "pass" if not risky_site else "fail", "detail": f"bad={len(risky_site)}"})
    title_conflict_site = [row for row in set_rows if row.get("site_use_policy") == "use_now_logical_sibling_collection_only" and row.get("nonmatching_title_table_refs")]
    audits.append({"check": "site_ready_sibling_sets_have_no_title_table_conflicts", "status": "pass" if not title_conflict_site else "fail", "detail": f"bad={len(title_conflict_site)}"})
    member_set_ids = {str(row.get("sibling_set_id", "")) for row in member_rows}
    missing_members = [row for row in set_rows if str(row.get("sibling_set_id", "")) not in member_set_ids]
    audits.append({"check": "sibling_sets_have_member_rows", "status": "pass" if not missing_members else "fail", "detail": f"missing={len(missing_members)} total={len(set_rows)}"})
    try:
        with zipfile.ZipFile(workbook_path) as zf:
            bad_file = zf.testzip()
        status = "pass" if bad_file is None else "fail"
        detail = "ok" if bad_file is None else str(bad_file)
    except Exception as exc:  # pragma: no cover
        status = "fail"
        detail = str(exc)
    audits.append({"check": "workbook_zip_integrity", "status": status, "detail": detail})
    return audits


def build_summary(set_rows: list[dict[str, object]], member_rows: list[dict[str, object]], audits: list[dict[str, object]], paths: dict[str, Path]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = [
        {"metric": "counts", "name": "logical_sibling_sets", "count": len(set_rows)},
        {"metric": "counts", "name": "logical_sibling_members", "count": len(member_rows)},
        {"metric": "counts", "name": "site_ready_logical_sibling_sets", "count": sum(1 for row in set_rows if row.get("site_use_policy") == "use_now_logical_sibling_collection_only")},
        {"metric": "counts", "name": "review_logical_sibling_sets", "count": sum(1 for row in set_rows if row.get("review_required") == "yes")},
        {"metric": "counts", "name": "audit_failure_count", "count": sum(1 for row in audits if row.get("status") != "pass")},
    ]
    for metric, counter in [
        ("sets_by_status", Counter(str(row.get("sibling_set_status", "")) for row in set_rows)),
        ("sets_by_policy", Counter(str(row.get("site_use_policy", "")) for row in set_rows)),
        ("sets_by_table", Counter(str(row.get("table_number_key", "")) for row in set_rows)),
        ("sets_by_topic", Counter(str(row.get("topic_variant_key", "")) for row in set_rows)),
    ]:
        for name, count in counter.most_common():
            rows.append({"metric": metric, "name": name, "count": count})
    for name, path in paths.items():
        rows.append({"metric": "outputs", "name": name, "count": output_path(path)})
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logical-evidence-dir", type=Path, default=DEFAULT_LOGICAL_EVIDENCE_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    collections = read_csv(args.logical_evidence_dir / "bounded_logical_collections.csv")
    members = read_csv(args.logical_evidence_dir / "logical_collection_members.csv")
    set_rows, member_rows = build_sets(collections, members)
    site_ready_rows = [row for row in set_rows if row.get("site_use_policy") == "use_now_logical_sibling_collection_only"]
    review_rows = [row for row in set_rows if row.get("review_required") == "yes"]

    paths = {
        "logical_sibling_sets": out_dir / "logical_sibling_sets.csv",
        "logical_sibling_set_members": out_dir / "logical_sibling_set_members.csv",
        "site_ready_logical_sibling_sets": out_dir / "site_ready_logical_sibling_sets.csv",
        "review_logical_sibling_sets": out_dir / "review_logical_sibling_sets.csv",
        "audit": out_dir / "acceptance_audit.csv",
        "summary": out_dir / "summary.csv",
        "workbook": out_dir / "logical_sibling_sets_workbook.xlsx",
    }
    write_csv(paths["logical_sibling_sets"], set_rows, SET_FIELDS)
    write_csv(paths["logical_sibling_set_members"], member_rows, MEMBER_FIELDS)
    write_csv(paths["site_ready_logical_sibling_sets"], site_ready_rows, SET_FIELDS)
    write_csv(paths["review_logical_sibling_sets"], review_rows, SET_FIELDS)

    write_workbook(paths["workbook"], [], [], set_rows, member_rows, site_ready_rows, review_rows)
    audits = build_audits(set_rows, member_rows, paths["workbook"])
    summary = build_summary(set_rows, member_rows, audits, paths)
    write_workbook(paths["workbook"], summary, audits, set_rows, member_rows, site_ready_rows, review_rows)
    audits = build_audits(set_rows, member_rows, paths["workbook"])
    summary = build_summary(set_rows, member_rows, audits, paths)
    write_workbook(paths["workbook"], summary, audits, set_rows, member_rows, site_ready_rows, review_rows)
    write_csv(paths["audit"], audits, AUDIT_FIELDS)
    write_csv(paths["summary"], summary, SUMMARY_FIELDS)
    (out_dir / "summary.json").write_text(
        json.dumps(
            {
                "logical_sibling_sets": len(set_rows),
                "logical_sibling_members": len(member_rows),
                "site_ready_logical_sibling_sets": len(site_ready_rows),
                "review_logical_sibling_sets": len(review_rows),
                "sets_by_status": dict(Counter(str(row.get("sibling_set_status", "")) for row in set_rows)),
                "sets_by_policy": dict(Counter(str(row.get("site_use_policy", "")) for row in set_rows)),
                "audit_failure_count": sum(1 for row in audits if row.get("status") != "pass"),
                "outputs": {name: output_path(path) for name, path in paths.items()},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
