#!/usr/bin/env python3
"""Build bounded logical collection evidence and split suggestions.

This layer performs no OCR/API work. It refines logical-only groupings so the
site can use small, bounded sibling collections while keeping overbroad parents
out of publishable grouping outputs.
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
DEFAULT_LOGICAL_DIR = ROOT / "export_final/pdf_recovery_incoming_2026-06-16_first_principles_logical_family_frontier_v32"
DEFAULT_QUALITY_DIR = ROOT / "export_final/pdf_recovery_incoming_2026-06-16_first_principles_assembly_quality_audit_v36"
DEFAULT_RULEBOOK_DIR = ROOT / "export_final/pdf_recovery_incoming_2026-06-16_first_principles_assembly_rulebook_v34"
DEFAULT_CONTRACT_DIR = ROOT / "export_final/pdf_recovery_incoming_2026-06-16_first_principles_assembly_contract_v38"
DEFAULT_OUT_DIR = ROOT / "export_final/pdf_recovery_incoming_2026-06-16_first_principles_logical_collection_evidence_v40"

MAX_BOUNDED_PAGES = 6
RELIGION_TERMS = [
    "all_religions",
    "hindu",
    "muslim",
    "musalman",
    "mussalman",
    "mussulman",
    "muhammadan",
    "mohammadan",
    "mohammedan",
    "christian",
    "jew",
    "jain",
    "parsi",
    "sikh",
    "buddhist",
    "brahmo",
    "animistic",
    "animist",
]
RELIGION_NORMALIZATION = {
    "musalman": "muslim",
    "mussalman": "muslim",
    "mussulman": "muslim",
    "muhammadan": "muslim",
    "mohammadan": "muslim",
    "mohammedan": "muslim",
    "animist": "animistic",
}

RELIGION_ALIASES = [
    ("all_religions", r"\ball\s+religions?\b"),
    ("hindu", r"\bhindus?\b|\bhindu\b"),
    (
        "muslim",
        r"\bmuslims?\b|\bmusalman(?:s)?\b|\bmussalman(?:s)?\b|\bmussulman(?:s)?\b|"
        r"\bmuhammadan(?:s)?\b|\bmohammadan(?:s)?\b|\bmohammedan(?:s)?\b",
    ),
    ("christian", r"\bchristians?\b|\bchristian\b"),
    ("jew", r"\bjews?\b|\bjewish\b|\bjew\b"),
    ("jain", r"\bjains?\b|\bjain\b"),
    ("parsi", r"\bparsis?\b|\bparsi\b"),
    ("sikh", r"\bsikhs?\b|\bsikh\b"),
    ("buddhist", r"\bbuddh?ists?\b|\bbuddhist\b|\bbudhist\b"),
    ("brahmo", r"\bbrahmos?\b|\bbrahmo\b"),
    ("animistic", r"\banimistic\b|\banimist\b|\btribal\b|\baboriginal\b"),
]

COLLECTION_FIELDS = [
    "logical_rank",
    "logical_collection_id",
    "source_kind",
    "parent_family_id",
    "parent_status",
    "parent_audit_bucket",
    "site_use_policy",
    "logical_collection_only",
    "physical_merge_allowed",
    "evidence_bucket",
    "review_required",
    "priority",
    "v5_source_key",
    "source_pdf_id",
    "page_range",
    "component_pages",
    "n_pages",
    "member_count",
    "pages_are_consecutive",
    "table_number_key",
    "topic_variant_key",
    "religion_terms_detected",
    "distinct_religion_count",
    "member_title_sample",
    "parent_page_range",
    "parent_n_pages",
    "parent_risk_level",
    "recommended_action",
]

MEMBER_FIELDS = [
    "member_row_rank",
    "logical_collection_id",
    "parent_family_id",
    "source_kind",
    "v5_source_key",
    "source_pdf_id",
    "member_position",
    "pdf_page",
    "expanded_pages",
    "table_number_key",
    "topic_variant_key",
    "religion_terms_detected",
    "title",
    "image_path",
    "source_member_key",
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


def semi(values: Iterable[object]) -> str:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    return "; ".join(out)


def stable_id(prefix: str, *parts: object) -> str:
    blob = "||".join(str(part or "") for part in parts)
    return f"{prefix}_{hashlib.sha1(blob.encode('utf-8')).hexdigest()[:16]}"


def int_or_none(value: object) -> int | None:
    if value in {None, ""}:
        return None
    try:
        return int(str(value))
    except ValueError:
        return None


def expand_page_token(value: object) -> list[int]:
    text = str(value or "").strip()
    if not text:
        return []
    pages: list[int] = []
    for part in re.split(r"[;,]\s*", text):
        part = part.strip()
        if not part:
            continue
        match = re.fullmatch(r"(\d+)\s*-\s*(\d+)", part)
        if match:
            start, end = int(match.group(1)), int(match.group(2))
            if start <= end and end - start <= 100:
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
    ranges: list[str] = []
    start = prev = values[0]
    for page in values[1:]:
        if page == prev + 1:
            prev = page
            continue
        ranges.append(str(start) if start == prev else f"{start}-{prev}")
        start = prev = page
    ranges.append(str(start) if start == prev else f"{start}-{prev}")
    return "; ".join(ranges)


def pages_consecutive(pages: Iterable[int]) -> bool:
    values = sorted(set(pages))
    return bool(values) and values == list(range(values[0], values[-1] + 1))


def roman_to_int(value: str) -> str:
    roman = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
    text = value.upper()
    if not text or any(ch not in roman for ch in text):
        return value
    total = 0
    prev = 0
    for ch in reversed(text):
        cur = roman[ch]
        if cur < prev:
            total -= cur
        else:
            total += cur
            prev = cur
    return str(total)


def table_number(title: object, fallback: object = "") -> str:
    text = str(title or "")
    match = re.search(r"\bTABLE\s+([IVXLCDM]+|\d+)\b", text, flags=re.IGNORECASE)
    if match:
        token = match.group(1).upper()
        return roman_to_int(token)
    fallback_text = str(fallback or "")
    match = re.search(r"\b([IVXLCDM]+|\d+)\b", fallback_text, flags=re.IGNORECASE)
    if match:
        return roman_to_int(match.group(1).upper())
    return ""


def topic_variant(title: object, segment_variant: object = "", variant_category: object = "") -> str:
    text = " ".join([str(title or ""), str(segment_variant or ""), str(variant_category or "")]).lower()
    text = text.replace("\u2014", "-").replace("\u2013", "-")
    checks = [
        ("widowed", [r"\bwidow", r"\bwidowed"]),
        ("unmarried", [r"\bunmarried", r"\bsingle\b"]),
        ("married", [r"\bmarried"]),
        ("population", [r"\bpopulation", r"\bpersons"]),
        ("age_by_religion", [r"age.*religion", r"religion.*age"]),
        ("civil_condition", [r"civil condition"]),
        ("literacy", [r"\blitera", r"\billiterate"]),
        ("education", [r"\beducat", r"\bscholar"]),
        ("occupation", [r"\boccupation", r"\bworker"]),
        ("caste_tribe_race", [r"\bcaste", r"\btribe", r"\brace"]),
        ("houses", [r"\bhouse"]),
    ]
    for key, patterns in checks:
        if any(re.search(pattern, text) for pattern in patterns):
            return key
    return "same_title_unspecified_variant"


def religion_terms(*values: object) -> list[str]:
    text = " ".join(str(value or "") for value in values).lower()
    found: list[str] = []
    for normalized, pattern in RELIGION_ALIASES:
        if re.search(pattern, text) and normalized not in found:
            found.append(normalized)
    for term in RELIGION_TERMS:
        needle = term.replace("_", " ")
        if term in text or needle in text:
            normalized = RELIGION_NORMALIZATION.get(term, term)
            if normalized not in found:
                found.append(normalized)
    return found


def page_min(row: dict[str, object]) -> int:
    pages = row.get("_pages", [])
    return min(pages) if isinstance(pages, list) and pages else 10**9


def page_max(row: dict[str, object]) -> int:
    pages = row.get("_pages", [])
    return max(pages) if isinstance(pages, list) and pages else -1


def page_signal_lookup(page_rows: list[dict[str, str]]) -> dict[tuple[str, str, str], dict[str, str]]:
    lookup: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in page_rows:
        lookup[(row.get("v5_source_key", ""), row.get("source_pdf_id_guess", ""), row.get("pdf_page", ""))] = row
    return lookup


def prepare_member(row: dict[str, str], parent: dict[str, str] | None = None) -> dict[str, object]:
    title = row.get("title", "")
    fallback_table = parent.get("title_table_numbers", "") if parent else ""
    terms = religion_terms(row.get("religion_terms_detected", ""), row.get("variant_terms", ""), title)
    pages = expand_page_token(row.get("pdf_page", ""))
    if not pages:
        pages = expand_page_token(row.get("member_pages", ""))
    return {
        **row,
        "_pages": pages,
        "_table_number": table_number(title, fallback_table),
        "_topic_variant": topic_variant(title, row.get("segment_variant", ""), row.get("variant_category_from_previous", "")),
        "_religion_terms": terms,
    }


def close_chunk(
    parent: dict[str, str],
    chunk: list[dict[str, object]],
    source_kind: str,
    parent_audit_bucket: str,
    parent_status: str,
    force_review: bool,
) -> tuple[dict[str, object], list[dict[str, object]]] | None:
    if not chunk:
        return None
    pages: list[int] = []
    terms: list[str] = []
    titles: list[str] = []
    table_numbers: list[str] = []
    variants: list[str] = []
    for member in chunk:
        pages.extend(member.get("_pages", []))
        terms.extend(member.get("_religion_terms", []))
        if member.get("title"):
            titles.append(str(member.get("title")))
        if member.get("_table_number"):
            table_numbers.append(str(member.get("_table_number")))
        if member.get("_topic_variant"):
            variants.append(str(member.get("_topic_variant")))
    consecutive = pages_consecutive(pages)
    distinct_terms = sorted(set(terms))
    table_key = semi(table_numbers)
    variant_key = semi(variants)
    n_pages = len(set(pages))
    is_confirmed_parent = parent_status == "confirmed_logical_collection"
    has_specific_title_evidence = bool(titles) and variant_key != "same_title_unspecified_variant"
    has_specific_religion_evidence = len(distinct_terms) >= 2
    can_site_use = (
        is_confirmed_parent
        and not force_review
        and consecutive
        and n_pages <= MAX_BOUNDED_PAGES
        and (has_specific_title_evidence or has_specific_religion_evidence)
    )
    if can_site_use:
        evidence_bucket = "site_ready_bounded_logical_collection"
        site_use_policy = "use_now_logical_collection_only"
        review_required = "no"
        priority = 1
        action = "Use as logical-only sibling collection; never physically merge pages."
    elif len(distinct_terms) >= 2 and consecutive and n_pages <= MAX_BOUNDED_PAGES:
        evidence_bucket = "bounded_multi_religion_same_variant_review"
        site_use_policy = "review_before_site_use_logical_only"
        review_required = "yes"
        priority = 2
        action = "Review as bounded same-table/same-variant religion sibling collection."
    elif consecutive and n_pages <= MAX_BOUNDED_PAGES:
        evidence_bucket = "bounded_child_logical_review"
        site_use_policy = "review_before_site_use_logical_only"
        review_required = "yes"
        priority = 3
        action = "Review as bounded child collection; parent remains too broad or unconfirmed."
    else:
        evidence_bucket = "unbounded_logical_review_only"
        site_use_policy = "do_not_use_as_single_group_until_split"
        review_required = "yes"
        priority = 9
        action = "Do not use as one collection; split further or inspect manually."
    child_id = stable_id("v40log", source_kind, parent.get("family_id", parent.get("sequence_id", "")), compact_pages(pages), table_key, variant_key, semi(distinct_terms))
    row = {
        "logical_collection_id": child_id,
        "source_kind": source_kind,
        "parent_family_id": parent.get("family_id", parent.get("sequence_id", "")),
        "parent_status": parent_status,
        "parent_audit_bucket": parent_audit_bucket,
        "site_use_policy": site_use_policy,
        "logical_collection_only": "yes",
        "physical_merge_allowed": "no",
        "evidence_bucket": evidence_bucket,
        "review_required": review_required,
        "priority": priority,
        "v5_source_key": parent.get("v5_source_key", ""),
        "source_pdf_id": parent.get("source_pdf_id", parent.get("source_pdf_id_guess", "")),
        "page_range": compact_pages(pages),
        "component_pages": semi(sorted(set(pages))),
        "n_pages": n_pages,
        "member_count": len(chunk),
        "pages_are_consecutive": "yes" if consecutive else "no",
        "table_number_key": table_key,
        "topic_variant_key": variant_key,
        "religion_terms_detected": semi(distinct_terms),
        "distinct_religion_count": len(distinct_terms),
        "member_title_sample": semi(titles[:5]),
        "parent_page_range": parent.get("page_ranges", parent.get("page_range", "")),
        "parent_n_pages": parent.get("n_physical_pages", parent.get("n_pages", "")),
        "parent_risk_level": parent.get("risk_level", ""),
        "recommended_action": action,
    }
    member_rows: list[dict[str, object]] = []
    for pos, member in enumerate(chunk, start=1):
        member_rows.append(
            {
                "logical_collection_id": child_id,
                "parent_family_id": row["parent_family_id"],
                "source_kind": source_kind,
                "v5_source_key": parent.get("v5_source_key", ""),
                "source_pdf_id": parent.get("source_pdf_id", parent.get("source_pdf_id_guess", "")),
                "member_position": pos,
                "pdf_page": member.get("pdf_page", ""),
                "expanded_pages": semi(member.get("_pages", [])),
                "table_number_key": member.get("_table_number", ""),
                "topic_variant_key": member.get("_topic_variant", ""),
                "religion_terms_detected": semi(member.get("_religion_terms", [])),
                "title": member.get("title", ""),
                "image_path": member.get("image_path", ""),
                "source_member_key": member.get("member_rank", member.get("sequence_rank", "")),
            }
        )
    return row, member_rows


def split_family(
    parent: dict[str, str],
    raw_members: list[dict[str, str]],
    audit_bucket: str,
    force_review: bool = False,
    source_kind: str = "logical_family_frontier",
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    members = [prepare_member(member, parent) for member in raw_members]
    members.sort(key=lambda row: (page_min(row), page_max(row), str(row.get("member_position", ""))))
    chunks: list[dict[str, object]] = []
    chunk_members: list[dict[str, object]] = []
    current: list[dict[str, object]] = []
    last_page = None
    current_sig = None
    parent_status = parent.get("frontier_status", parent.get("coverage_bucket", "review"))
    for member in members:
        pages = member.get("_pages", [])
        if not isinstance(pages, list) or not pages:
            continue
        sig = (member.get("_table_number", ""), member.get("_topic_variant", ""))
        gap_break = last_page is not None and min(pages) > last_page + 1
        sig_break = current_sig is not None and sig != current_sig and current
        size_break = current and len(set([page for item in current for page in item.get("_pages", [])] + pages)) > MAX_BOUNDED_PAGES
        if gap_break or sig_break or size_break:
            closed = close_chunk(parent, current, source_kind, audit_bucket, parent_status, force_review)
            if closed:
                row, member_rows = closed
                chunks.append(row)
                chunk_members.extend(member_rows)
            current = []
            current_sig = None
        current.append(member)
        current_sig = sig
        last_page = max(pages)
    closed = close_chunk(parent, current, source_kind, audit_bucket, parent_status, force_review)
    if closed:
        row, member_rows = closed
        chunks.append(row)
        chunk_members.extend(member_rows)
    return chunks, chunk_members


def synthetic_members_from_sequence(seq: dict[str, str], page_lookup: dict[tuple[str, str, str], dict[str, str]]) -> list[dict[str, str]]:
    pages = expand_page_token(seq.get("component_pages", ""))
    fallback_titles = [part.strip() for part in str(seq.get("titles", "")).split(";") if part.strip()]
    members: list[dict[str, str]] = []
    for idx, page in enumerate(pages, start=1):
        signal = page_lookup.get((seq.get("v5_source_key", ""), seq.get("source_pdf_id_guess", ""), str(page)), {})
        title = signal.get("title", "") or (fallback_titles[idx - 1] if idx - 1 < len(fallback_titles) else "")
        page_terms = religion_terms(title)
        members.append(
            {
                "member_rank": str(idx),
                "pdf_page": str(page),
                "title": title,
                "religion_terms_detected": semi(page_terms) if page_terms else seq.get("religion_terms_detected", ""),
                "image_path": signal.get("image_path", ""),
                "segment_variant": "",
                "variant_category_from_previous": "",
            }
        )
    return members


def build_outputs(
    logical_dir: Path,
    quality_dir: Path,
    rulebook_dir: Path,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    families = read_csv(logical_dir / "logical_family_frontier.csv")
    members = read_csv(logical_dir / "logical_family_members.csv")
    quality = read_csv(quality_dir / "logical_family_quality_audit.csv")
    religion_sequences = read_csv(quality_dir / "religion_sequence_coverage_audit.csv")
    page_signals = read_csv(rulebook_dir / "page_signal_index.csv")
    page_lookup = page_signal_lookup(page_signals)
    quality_by_family = {row.get("family_id", ""): row for row in quality}
    members_by_family: dict[str, list[dict[str, str]]] = defaultdict(list)
    for member in members:
        members_by_family[member.get("family_id", "")].append(member)

    collections: list[dict[str, object]] = []
    collection_members: list[dict[str, object]] = []
    split_suggestions: list[dict[str, object]] = []
    missing_religion_splits: list[dict[str, object]] = []

    for family in families:
        family_id = family.get("family_id", "")
        audit = quality_by_family.get(family_id, {})
        audit_bucket = audit.get("audit_bucket", "")
        force_review = audit_bucket in {
            "large_review_family_overbreadth_risk",
            "single_variant_large_review_family",
            "religion_family_review_candidate",
            "logical_review_candidate",
        } or family.get("frontier_status") != "confirmed_logical_collection"
        chunks, chunk_members = split_family(family, members_by_family.get(family_id, []), audit_bucket, force_review=force_review)
        collections.extend(chunks)
        collection_members.extend(chunk_members)
        if audit.get("overbroad_risk") == "yes" or audit_bucket in {"large_review_family_overbreadth_risk", "single_variant_large_review_family", "religion_family_review_candidate"}:
            split_suggestions.extend(chunks)

    for seq in religion_sequences:
        if seq.get("coverage_bucket") != "possible_missing_religion_family":
            continue
        parent = {
            "sequence_id": seq.get("sequence_id", ""),
            "coverage_bucket": seq.get("coverage_bucket", ""),
            "frontier_status": "review_missing_religion_family",
            "v5_source_key": seq.get("v5_source_key", ""),
            "source_pdf_id_guess": seq.get("source_pdf_id_guess", ""),
            "page_range": seq.get("page_range", ""),
            "n_pages": seq.get("n_pages", ""),
            "risk_level": "review_missing_core_religion_family",
        }
        raw_members = synthetic_members_from_sequence(seq, page_lookup)
        chunks, chunk_members = split_family(parent, raw_members, seq.get("coverage_bucket", ""), force_review=True, source_kind="qa_missing_religion_sequence")
        for row in chunks:
            row["evidence_bucket"] = "missing_religion_family_bounded_review" if row.get("n_pages", 0) <= MAX_BOUNDED_PAGES else row.get("evidence_bucket", "")
            row["recommended_action"] = "Review as missing bounded religion-family candidate; logical-only, no physical merge."
        collections.extend(chunks)
        collection_members.extend(chunk_members)
        missing_religion_splits.extend(chunks)

    for idx, row in enumerate(collections, start=1):
        row["logical_rank"] = idx
    for idx, row in enumerate(collection_members, start=1):
        row["member_row_rank"] = idx

    site_ready = [row for row in collections if row.get("site_use_policy") == "use_now_logical_collection_only"]
    review = [row for row in collections if row.get("site_use_policy") != "use_now_logical_collection_only"]
    return collections, site_ready, review, split_suggestions, missing_religion_splits, collection_members


def build_audits(
    collections: list[dict[str, object]],
    site_ready: list[dict[str, object]],
    split_suggestions: list[dict[str, object]],
    missing_religion_splits: list[dict[str, object]],
    quality_rows: list[dict[str, str]],
    religion_sequences: list[dict[str, str]],
    workbook_path: Path,
) -> list[dict[str, object]]:
    audits: list[dict[str, object]] = []
    physical_bad = [row for row in collections if row.get("physical_merge_allowed") != "no" or row.get("logical_collection_only") != "yes"]
    audits.append({"check": "logical_collections_are_never_physical_merges", "status": "pass" if not physical_bad else "fail", "detail": f"bad={len(physical_bad)}"})
    unbounded_site_ready = [
        row for row in site_ready
        if row.get("pages_are_consecutive") != "yes" or (int_or_none(row.get("n_pages")) or 999) > MAX_BOUNDED_PAGES
    ]
    audits.append({"check": "site_ready_logical_collections_are_bounded", "status": "pass" if not unbounded_site_ready else "fail", "detail": f"bad={len(unbounded_site_ready)} total={len(site_ready)} max_pages={MAX_BOUNDED_PAGES}"})
    overbroad_families = {row.get("family_id", "") for row in quality_rows if row.get("overbroad_risk") == "yes"}
    split_parent_ids = {str(row.get("parent_family_id", "")) for row in split_suggestions}
    missing_splits = sorted(overbroad_families - split_parent_ids)
    audits.append({"check": "overbroad_families_have_split_suggestions", "status": "pass" if not missing_splits else "fail", "detail": f"missing={len(missing_splits)} overbroad={len(overbroad_families)}"})
    large_children = [
        row for row in collections
        if row.get("evidence_bucket") != "unbounded_logical_review_only" and (int_or_none(row.get("n_pages")) or 999) > MAX_BOUNDED_PAGES
    ]
    audits.append({"check": "bounded_children_respect_page_limit", "status": "pass" if not large_children else "fail", "detail": f"bad={len(large_children)} max_pages={MAX_BOUNDED_PAGES}"})
    expected_missing = [row for row in religion_sequences if row.get("coverage_bucket") == "possible_missing_religion_family"]
    covered_missing = {row.get("parent_family_id", "") for row in missing_religion_splits}
    audits.append({"check": "missing_religion_sequences_have_bounded_candidates", "status": "pass" if len(covered_missing) == len(expected_missing) else "fail", "detail": f"sequences={len(expected_missing)} covered={len(covered_missing)}"})
    try:
        with zipfile.ZipFile(workbook_path) as zf:
            bad_member = zf.testzip()
    except Exception as exc:
        bad_member = f"{type(exc).__name__}: {exc}"
    audits.append({"check": "workbook_zip_integrity", "status": "pass" if bad_member is None else "fail", "detail": str(bad_member or "ok")})
    return audits


def build_summary(
    collections: list[dict[str, object]],
    site_ready: list[dict[str, object]],
    review: list[dict[str, object]],
    split_suggestions: list[dict[str, object]],
    missing_religion_splits: list[dict[str, object]],
    members: list[dict[str, object]],
    audits: list[dict[str, object]],
    paths: dict[str, Path],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = [{"metric": "paths", "name": name, "count": output_path(path)} for name, path in paths.items()]
    rows.extend(
        [
            {"metric": "counts", "name": "bounded_logical_collections", "count": len(collections)},
            {"metric": "counts", "name": "site_ready_logical_collections", "count": len(site_ready)},
            {"metric": "counts", "name": "review_logical_candidates", "count": len(review)},
            {"metric": "counts", "name": "overbroad_split_suggestions", "count": len(split_suggestions)},
            {"metric": "counts", "name": "missing_religion_split_candidates", "count": len(missing_religion_splits)},
            {"metric": "counts", "name": "logical_collection_members", "count": len(members)},
            {"metric": "counts", "name": "audit_failures", "count": sum(1 for row in audits if row.get("status") != "pass")},
        ]
    )
    for metric, counter in [
        ("collections_by_evidence_bucket", Counter(str(row.get("evidence_bucket", "")) for row in collections)),
        ("collections_by_policy", Counter(str(row.get("site_use_policy", "")) for row in collections)),
        ("collections_by_topic_variant", Counter(str(row.get("topic_variant_key", "")) for row in collections)),
        ("audit_by_status", Counter(str(row.get("status", "")) for row in audits)),
    ]:
        for name, count in sorted(counter.items()):
            rows.append({"metric": metric, "name": name, "count": count})
    return rows


def write_workbook(
    path: Path,
    summary_rows: list[dict[str, object]],
    audits: list[dict[str, object]],
    collections: list[dict[str, object]],
    site_ready: list[dict[str, object]],
    review: list[dict[str, object]],
    split_suggestions: list[dict[str, object]],
    missing_religion_splits: list[dict[str, object]],
    members: list[dict[str, object]],
) -> None:
    workbook = xlsxwriter.Workbook(path, {"constant_memory": True})
    header_fmt = workbook.add_format({"bold": True, "bg_color": "#E8EEF7", "border": 1})
    wrap_fmt = workbook.add_format({"text_wrap": True, "valign": "top"})
    sheets = [
        ("Summary", summary_rows, SUMMARY_FIELDS),
        ("Audit", audits, AUDIT_FIELDS),
        ("Bounded Collections", collections, COLLECTION_FIELDS),
        ("Site Ready Logical", site_ready, COLLECTION_FIELDS),
        ("Review Logical", review, COLLECTION_FIELDS),
        ("Overbroad Splits", split_suggestions, COLLECTION_FIELDS),
        ("Missing Religion", missing_religion_splits, COLLECTION_FIELDS),
        ("Members", members, MEMBER_FIELDS),
    ]
    for name, rows, fields in sheets:
        ws = workbook.add_worksheet(name[:31])
        for col, field in enumerate(fields):
            ws.write(0, col, field, header_fmt)
        for row_idx, row in enumerate(rows, start=1):
            for col, field in enumerate(fields):
                value = row.get(field, "")
                fmt = wrap_fmt if field in {"member_title_sample", "recommended_action", "title", "image_path"} else None
                ws.write(row_idx, col, value, fmt)
        ws.freeze_panes(1, 0)
        ws.autofilter(0, 0, max(0, len(rows)), max(0, len(fields) - 1))
        for col, field in enumerate(fields):
            ws.set_column(col, col, min(max(len(field) + 2, 12), 60))
    workbook.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logical-dir", default=str(DEFAULT_LOGICAL_DIR))
    parser.add_argument("--quality-dir", default=str(DEFAULT_QUALITY_DIR))
    parser.add_argument("--rulebook-dir", default=str(DEFAULT_RULEBOOK_DIR))
    parser.add_argument("--contract-dir", default=str(DEFAULT_CONTRACT_DIR))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    args = parser.parse_args()

    logical_dir = Path(args.logical_dir).expanduser()
    quality_dir = Path(args.quality_dir).expanduser()
    rulebook_dir = Path(args.rulebook_dir).expanduser()
    contract_dir = Path(args.contract_dir).expanduser()
    out_dir = Path(args.out_dir).expanduser()
    paths = {
        "logical_dir": logical_dir,
        "quality_dir": quality_dir,
        "rulebook_dir": rulebook_dir,
        "contract_dir": contract_dir,
        "out_dir": out_dir,
    }

    collections, site_ready, review, split_suggestions, missing_religion_splits, members = build_outputs(logical_dir, quality_dir, rulebook_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    workbook_path = out_dir / "logical_collection_evidence_workbook.xlsx"
    write_workbook(workbook_path, [], [], collections, site_ready, review, split_suggestions, missing_religion_splits, members)
    quality_rows = read_csv(quality_dir / "logical_family_quality_audit.csv")
    religion_sequences = read_csv(quality_dir / "religion_sequence_coverage_audit.csv")
    audits = build_audits(collections, site_ready, split_suggestions, missing_religion_splits, quality_rows, religion_sequences, workbook_path)
    summary_rows = build_summary(collections, site_ready, review, split_suggestions, missing_religion_splits, members, audits, paths)
    write_workbook(workbook_path, summary_rows, audits, collections, site_ready, review, split_suggestions, missing_religion_splits, members)
    audits = build_audits(collections, site_ready, split_suggestions, missing_religion_splits, quality_rows, religion_sequences, workbook_path)
    summary_rows = build_summary(collections, site_ready, review, split_suggestions, missing_religion_splits, members, audits, paths)
    write_workbook(workbook_path, summary_rows, audits, collections, site_ready, review, split_suggestions, missing_religion_splits, members)

    write_csv(out_dir / "bounded_logical_collections.csv", collections, COLLECTION_FIELDS)
    write_csv(out_dir / "site_ready_bounded_logical_collections.csv", site_ready, COLLECTION_FIELDS)
    write_csv(out_dir / "review_bounded_logical_candidates.csv", review, COLLECTION_FIELDS)
    write_csv(out_dir / "overbroad_logical_split_suggestions.csv", split_suggestions, COLLECTION_FIELDS)
    write_csv(out_dir / "missing_religion_split_candidates.csv", missing_religion_splits, COLLECTION_FIELDS)
    write_csv(out_dir / "logical_collection_members.csv", members, MEMBER_FIELDS)
    write_csv(out_dir / "audit.csv", audits, AUDIT_FIELDS)
    write_csv(out_dir / "summary.csv", summary_rows, SUMMARY_FIELDS)
    summary = {
        "bounded_logical_collections": len(collections),
        "site_ready_logical_collections": len(site_ready),
        "review_logical_candidates": len(review),
        "overbroad_split_suggestions": len(split_suggestions),
        "missing_religion_split_candidates": len(missing_religion_splits),
        "logical_collection_members": len(members),
        "audit_failure_count": sum(1 for row in audits if row.get("status") != "pass"),
        "collections_by_evidence_bucket": dict(Counter(str(row.get("evidence_bucket", "")) for row in collections)),
        "collections_by_policy": dict(Counter(str(row.get("site_use_policy", "")) for row in collections)),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
