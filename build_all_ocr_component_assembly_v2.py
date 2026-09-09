#!/usr/bin/env python3
"""Build a broad review map of all OCRed census pages into components.

This is intentionally an analysis/output layer. It reads the conservative v1
component page map and reassigns the leftover singleton/weak records into
reviewable heuristic groups using source PDF + family + OCR table directory.
It does not modify the portal database or any OCR payloads.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import xlsxwriter


ROOT = Path(__file__).resolve().parent
V1_DIR = ROOT / "export_final/pdf_recovery_incoming_2026-06-16_all_ocr_component_assembly"
OUT_DIR = ROOT / "export_final/pdf_recovery_incoming_2026-06-16_all_ocr_component_assembly_v3_refined"
LEDGER_DIR = ROOT / "export_final/pdf_recovery_incoming_2026-06-16_no_gaps_ledger_v2_refined"

V1_PAGES = V1_DIR / "all_ocr_component_pages.csv"
V1_UNOCRD = V1_DIR / "all_ocr_unocrd_master_pages.csv"
STATIC_DIR = ROOT / "export_final/pdf_recovery_incoming_2026-06-16_static_site_after_core_money"
STATIC_MISSING = STATIC_DIR / "static_site_remaining_missing_pages_after_core_money.csv"
STATIC_NOW_LOCAL = STATIC_DIR / "static_site_pages_now_local_from_core_money.csv"
CORE_READY = ROOT / "export_final/pdf_recovery_incoming_2026-06-16_core_money_renders_check/core_money_NEW_RENDERS_READY_FOR_OCR.csv"
CORE_PAGES = ROOT / "export_final/pdf_recovery_incoming_2026-06-16_core_money_assembly_first_pass/core_money_tablekey_pages.csv"


PREASSIGNED_BASES = {
    "core_money_table_key",
    "portal_assembled_table",
    "proposed_allread_group",
}

HEURISTIC_INPUT_BASES = {
    "singleton_original_ocr",
    "fallback_source_page",
}

MISSING_OCR_STATUSES = {
    "no_ocr_json",
    "NO_OCR_JSON",
    "missing_ocr_artifact",
}

OCR_REVIEW_STATUSES = {
    "ocrd_rejected_constraints_failed",
    "ocrd_with_risk",
    "many_off_or_unchecked",
    "many_off",
    "few_off",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fields})


def stable_id(prefix: str, *parts: object) -> str:
    h = hashlib.sha1("||".join(str(p) for p in parts).encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{h}"


def norm_path(s: str) -> str:
    return (s or "").replace("\\", "/")


def table_dir_key(row: dict[str, str]) -> str:
    """Extract the OCR table directory from table_id or image_path."""
    value = norm_path(row.get("table_id") or row.get("image_path") or "")
    if not value:
        return ""
    parts = [p for p in value.split("/") if p]
    if "digitized_tables" in parts:
        i = parts.index("digitized_tables")
        if i + 1 < len(parts):
            return parts[i + 1]
    base = parts[-1] if parts else value
    if "__" in base:
        return base.split("__", 1)[0]
    if base.endswith("_oneshot") or base.endswith(".png") or base.endswith(".json"):
        if len(parts) >= 2:
            return parts[-2]
    return ""


def source_key(row: dict[str, str]) -> str:
    source_pdf = row.get("source_pdf") or ""
    if source_pdf:
        return source_pdf
    year = row.get("year") or ""
    province = row.get("province") or ""
    volume = row.get("source_group_key") or ""
    if year or province:
        return f"{year}::{province}"
    return volume or "unknown_source"


def int_or_none(value: str) -> int | None:
    value = (value or "").strip()
    if re.fullmatch(r"-?\d+", value):
        return int(value)
    return None


def page_sort_tuple(row: dict[str, object]) -> tuple[int, int, str]:
    p = int_or_none(str(row.get("pdf_page", "")))
    printed = int_or_none(str(row.get("printed_page", "")))
    if p is not None:
        return (0, p, str(row.get("imghash") or row.get("image_path") or ""))
    if printed is not None:
        return (1, printed, str(row.get("imghash") or row.get("image_path") or ""))
    return (2, 0, str(row.get("imghash") or row.get("image_path") or ""))


def uniq_join(values: list[str], limit: int = 8) -> str:
    vals = []
    seen = set()
    for v in values:
        v = (v or "").strip()
        if v and v not in seen:
            seen.add(v)
            vals.append(v)
    if len(vals) > limit:
        return "; ".join(vals[:limit]) + f"; ...(+{len(vals)-limit})"
    return "; ".join(vals)


def counter_json(values: list[str]) -> str:
    return json.dumps(dict(Counter(v or "" for v in values)), sort_keys=True)


def page_gap_summary(rows: list[dict[str, object]]) -> tuple[str, int, str]:
    nums = sorted(n for n in {int_or_none(str(r.get("pdf_page", ""))) for r in rows} if n is not None)
    if not nums:
        return "", 0, ""
    if len(nums) == 1:
        return str(nums[0]), 0, ""
    gaps: list[tuple[int, int]] = []
    for a, b in zip(nums, nums[1:]):
        if b > a + 1:
            gaps.append((a + 1, b - 1))
    gap_count = sum(b - a + 1 for a, b in gaps)
    gap_labels = [str(a) if a == b else f"{a}-{b}" for a, b in gaps[:20]]
    return f"{nums[0]}-{nums[-1]}", gap_count, "; ".join(gap_labels)


def review_flag(component_basis: str, rows: list[dict[str, object]]) -> str:
    statuses = {str(r.get("ocr_status") or r.get("page_status") or "") for r in rows}
    lower_statuses = {s.lower() for s in statuses}
    if lower_statuses & {s.lower() for s in MISSING_OCR_STATUSES}:
        return "missing_ocr_artifact"
    if lower_statuses & {s.lower() for s in OCR_REVIEW_STATUSES}:
        return "ocr_review"
    if component_basis.startswith("heuristic_"):
        return "heuristic_review"
    if component_basis == "singleton_original_ocr":
        return "singleton"
    if component_basis == "fallback_source_page":
        return "weak_grouping"
    return "ok"


def row_has_missing_ocr(row: dict[str, object]) -> bool:
    status = str(row.get("ocr_status") or row.get("page_status") or "").lower()
    return status in {s.lower() for s in MISSING_OCR_STATUSES}


def row_has_ocr_review(row: dict[str, object]) -> bool:
    status = str(row.get("ocr_status") or row.get("page_status") or "").lower()
    return status in {s.lower() for s in OCR_REVIEW_STATUSES}


def component_range_review_flag(
    component_basis: str,
    rows: list[dict[str, object]],
    titles: list[str],
    ocr_statuses: list[str],
    gap_n: int,
    duplicate_pages: list[str],
) -> str:
    flags: list[str] = []
    n = len(rows)
    title_set = {t for t in titles if t}
    title_blob = " ".join(titles[:20]).lower()
    status_counts = Counter(s.lower() for s in ocr_statuses)
    empty_ratio = status_counts.get("empty", 0) / n if n else 0

    if component_basis == "core_money_table_key":
        chapterish = bool(re.search(r"\b(chapter|introduction|no table present|unknown|contents|corrigenda)\b", title_blob))
        if n >= 20 and (empty_ratio >= 0.25 or len(title_set) >= 10 or chapterish):
            flags.append("core_money_range_mixed_or_overbroad")
    if component_basis.startswith("heuristic_image_dir_group") and n >= 5:
        flags.append("image_dir_group_low_confidence")
    if component_basis == "heuristic_mixed_dir_group":
        flags.append("mixed_prior_basis_group")
    if duplicate_pages:
        flags.append("duplicate_pdf_pages")
    if gap_n:
        flags.append("internal_numeric_page_gaps")
    return "; ".join(flags)


def build_components(v1_rows: list[dict[str, str]]) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    """Return component summaries, reassigned pages, and heuristic groups."""
    heuristic_candidates: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in v1_rows:
        if row.get("component_scope") == "original_corpus" and row.get("component_basis") in HEURISTIC_INPUT_BASES:
            key = (source_key(row), row.get("family") or "", table_dir_key(row))
            if key[2]:
                heuristic_candidates[key].append(row)

    heuristic_multi = {k for k, rows in heuristic_candidates.items() if len(rows) >= 2}

    component_rows: dict[str, list[dict[str, object]]] = defaultdict(list)
    heuristic_group_rows: list[dict[str, object]] = []

    for row in v1_rows:
        new = dict(row)
        new["v1_component_id"] = row.get("component_id", "")
        new["v1_component_basis"] = row.get("component_basis", "")
        new["table_dir_key"] = table_dir_key(row)
        new["heuristic_source_key"] = source_key(row)
        basis = row.get("component_basis", "")

        if row.get("component_scope") == "original_corpus" and basis in HEURISTIC_INPUT_BASES:
            hkey = (source_key(row), row.get("family") or "", table_dir_key(row))
            if hkey in heuristic_multi:
                prior_bases = {r.get("component_basis", "") for r in heuristic_candidates[hkey]}
                if prior_bases == {"fallback_source_page"}:
                    nbasis = "heuristic_image_dir_group"
                    prefix = "imgdir"
                elif "fallback_source_page" in prior_bases:
                    nbasis = "heuristic_mixed_dir_group"
                    prefix = "mixdir"
                else:
                    nbasis = "heuristic_table_dir_group"
                    prefix = "tabledir"
                component_id = stable_id(prefix, *hkey)
                source_group = "||".join(hkey)
                new["component_id"] = component_id
                new["component_basis"] = nbasis
                new["source_group_key"] = source_group
            else:
                # Preserve the existing singleton/weak assignment.
                new["component_id"] = row.get("component_id", "")
                new["component_basis"] = basis
                new["source_group_key"] = row.get("source_group_key", "")
        elif basis in PREASSIGNED_BASES:
            new["component_id"] = row.get("component_id", "")
            new["component_basis"] = basis
            new["source_group_key"] = row.get("source_group_key", "")
        else:
            # Defensive fallback for unexpected rows.
            new["component_id"] = row.get("component_id") or stable_id("component", source_key(row), table_dir_key(row), row.get("imghash", ""))
            new["component_basis"] = basis or "unclassified_component"
            new["source_group_key"] = row.get("source_group_key", "")

        component_rows[str(new["component_id"])].append(new)

    for key in sorted(heuristic_multi):
        rows = heuristic_candidates[key]
        bases = Counter(r.get("component_basis", "") for r in rows)
        pages = [r.get("pdf_page", "") for r in rows]
        titles = [r.get("title", "") for r in rows]
        families = [r.get("family", "") for r in rows]
        prange, gap_n, gaps = page_gap_summary(rows)
        heuristic_group_rows.append({
            "heuristic_source_key": key[0],
            "family": key[1],
            "table_dir_key": key[2],
            "n_records": len(rows),
            "n_unique_pdf_pages": len({p for p in pages if p}),
            "page_range": prange,
            "numeric_gap_count_within_group": gap_n,
            "numeric_gaps_within_group": gaps,
            "n_distinct_titles": len({t for t in titles if t}),
            "mixed_titles": len({t for t in titles if t}) > 1,
            "n_distinct_families": len({f for f in families if f}),
            "prior_basis_counts": json.dumps(dict(bases), sort_keys=True),
            "primary_title": uniq_join(titles, limit=2),
            "example_pages": uniq_join(pages, limit=12),
        })

    summaries: list[dict[str, object]] = []
    pages_out: list[dict[str, object]] = []
    for cid, rows in component_rows.items():
        rows = sorted(rows, key=page_sort_tuple)
        for i, row in enumerate(rows, start=1):
            row["position_in_component"] = i
            pages_out.append(row)

        basis = str(rows[0].get("component_basis", ""))
        scope = str(rows[0].get("component_scope", ""))
        source_group = str(rows[0].get("source_group_key", ""))
        titles = [str(r.get("title", "")) for r in rows]
        families = [str(r.get("family", "")) for r in rows]
        source_pdfs = [str(r.get("source_pdf", "")) for r in rows]
        provinces = [str(r.get("province", "")) for r in rows]
        years = [str(r.get("year", "")) for r in rows]
        pages = [str(r.get("pdf_page", "")) for r in rows]
        table_dirs = [str(r.get("table_dir_key", "")) for r in rows]
        prange, gap_n, gaps = page_gap_summary(rows)
        page_counter = Counter(p for p in pages if p)
        duplicate_pages = [p for p, n in page_counter.items() if n > 1]
        ocr_statuses = [str(r.get("ocr_status") or r.get("page_status") or "") for r in rows]
        source_types = [str(r.get("source_record_type", "")) for r in rows]
        prior_bases = [str(r.get("v1_component_basis", "")) for r in rows]
        row_records = sum(int_or_none(str(r.get("rows_extracted", ""))) or 0 for r in rows)
        n_artifacts = sum(1 for r in rows if str(r.get("ocr_artifact", "")).strip())
        n_covered = sum(1 for r in rows if str(r.get("ocr_status", "")) == "covered_by_original_corpus_ocr")
        n_overlap = sum(1 for r in rows if str(r.get("overlaps_original_ocr_source_page", "")).lower() == "true")

        summaries.append({
            "component_id": cid,
            "component_scope": scope,
            "component_basis": basis,
            "review_flag": review_flag(basis, rows),
            "range_review_flag": component_range_review_flag(basis, rows, titles, ocr_statuses, gap_n, duplicate_pages),
            "source_group_key": source_group,
            "n_records": len(rows),
            "n_unique_source_pages": len({(r.get("source_pdf", ""), r.get("pdf_page", "")) for r in rows if r.get("pdf_page")}),
            "n_ocr_artifacts": n_artifacts,
            "n_covered_by_original_ocr": n_covered,
            "n_overlap_with_original_ocr_source_page": n_overlap,
            "row_records_from_json_pages": row_records,
            "years": uniq_join(years),
            "provinces": uniq_join(provinces),
            "source_pdfs": uniq_join(source_pdfs),
            "primary_family": Counter(families).most_common(1)[0][0] if families else "",
            "families": uniq_join(families),
            "primary_title": Counter(titles).most_common(1)[0][0] if titles else "",
            "n_distinct_titles": len({t for t in titles if t}),
            "mixed_titles": len({t for t in titles if t}) > 1,
            "table_dir_keys": uniq_join(table_dirs, limit=3),
            "page_range": prange,
            "numeric_gap_count_within_component": gap_n,
            "numeric_gaps_within_component": gaps,
            "duplicate_pdf_pages": uniq_join(duplicate_pages, limit=20),
            "ocr_status_counts": counter_json(ocr_statuses),
            "source_record_type_counts": counter_json(source_types),
            "v1_component_basis_counts": counter_json(prior_bases),
            "example_images": uniq_join([str(r.get("image_path", "")) for r in rows], limit=3),
        })

    summaries.sort(key=lambda r: (
        str(r["review_flag"]) not in {"missing_ocr_artifact", "ocr_review", "heuristic_review", "weak_grouping", "singleton"},
        str(r["review_flag"]),
        -int(r["n_records"]),
        str(r["component_id"]),
    ))
    pages_out.sort(key=lambda r: (str(r.get("component_id", "")), int(r.get("position_in_component", 0))))
    return summaries, pages_out, heuristic_group_rows


def autosize(ws, fields: list[str], rows: list[dict[str, object]], max_width: int = 60) -> None:
    widths = {field: min(max(len(field) + 2, 10), max_width) for field in fields}
    for row in rows[:2000]:
        for field in fields:
            widths[field] = min(max(widths[field], len(str(row.get(field, ""))) + 2), max_width)
    for i, field in enumerate(fields):
        ws.set_column(i, i, widths[field])


def add_sheet(wb, name: str, rows: list[dict[str, object]], fields: list[str]) -> None:
    ws = wb.add_worksheet(name[:31])
    header_fmt = wb.add_format({"bold": True, "bg_color": "#EDEDED", "border": 1})
    text_fmt = wb.add_format({"text_wrap": False, "valign": "top"})
    for c, field in enumerate(fields):
        ws.write(0, c, field, header_fmt)
    for r, row in enumerate(rows, start=1):
        for c, field in enumerate(fields):
            value = row.get(field, "")
            if isinstance(value, (dict, list, tuple, set)):
                value = json.dumps(value, sort_keys=True)
            ws.write(r, c, value, text_fmt)
    ws.freeze_panes(1, 0)
    if rows:
        ws.autofilter(0, 0, len(rows), len(fields) - 1)
    autosize(ws, fields, rows)


def write_workbook(path: Path, summary_rows: list[dict[str, object]], page_rows: list[dict[str, object]], heuristic_rows: list[dict[str, object]], refined_core_rows: list[dict[str, object]], unocrd_rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    wb = xlsxwriter.Workbook(str(path), {"constant_memory": True})
    ws = wb.add_worksheet("README")
    lines = [
        ("purpose", "Review map that assembles all OCRed pages into the strongest available components without changing corpus data."),
        ("component priority", "portal assembled table > all-read proposed group > heuristic source/family/table-dir group > singleton/weak page."),
        ("heuristic_review", "Pages share source, family, and OCR table directory; visually review before treating as final assembly."),
        ("missing_ocr_artifact", "A page is local/provenance-known but has no OCR JSON and no original OCR coverage."),
        ("ocr_review", "OCR exists but prior constraints/risk flags say it needs inspection."),
        ("singleton", "OCRed page could not be grouped confidently with local metadata."),
        ("weak_grouping", "OCRed page lacks enough table metadata; directory/title may still help manual assembly."),
    ]
    for r, (k, v) in enumerate(lines):
        ws.write(r, 0, k)
        ws.write(r, 1, v)

    component_fields = [
        "component_id", "component_scope", "component_basis", "review_flag", "range_review_flag", "source_group_key",
        "n_records", "n_unique_source_pages", "n_ocr_artifacts", "n_covered_by_original_ocr",
        "n_overlap_with_original_ocr_source_page", "row_records_from_json_pages", "years",
        "provinces", "source_pdfs", "primary_family", "families", "primary_title",
        "n_distinct_titles", "mixed_titles", "table_dir_keys", "page_range",
        "numeric_gap_count_within_component", "numeric_gaps_within_component",
        "duplicate_pdf_pages", "ocr_status_counts", "source_record_type_counts",
        "v1_component_basis_counts", "example_images",
    ]
    page_fields = [
        "component_id", "position_in_component", "component_scope", "component_basis", "component_review_flag", "range_review_flag",
        "source_record_type", "year", "province", "source_pdf", "pdf_page", "printed_page",
        "family", "title", "table_number", "table_dir_key", "table_id", "imghash",
        "image_path", "ocr_artifact", "ocr_status", "page_status", "validation", "verdict",
        "assembled_table", "position_in_table", "proposed_allread_group_id", "improved_group_id",
        "rows_extracted", "sections_extracted", "constraint_checks", "constraint_failed",
        "overlaps_original_ocr_source_page", "linked_original_component_ids", "linked_original_table_ids",
        "v1_component_id", "v1_component_basis", "parent_component_id", "parent_source_group_key",
        "core_page_class", "core_ocr_title", "core_ocr_status", "core_ocr_decision",
        "core_source_kinds", "core_rows", "core_sections", "core_constraint_failed",
        "original_overlap_titles", "original_overlap_families", "original_overlap_component_ids",
        "original_overlap_ocr_statuses",
    ]
    # Add component-level review flag to page rows for filtering.
    flag_by_id = {str(r["component_id"]): str(r["review_flag"]) for r in summary_rows}
    range_flag_by_id = {str(r["component_id"]): str(r.get("range_review_flag", "")) for r in summary_rows}
    for row in page_rows:
        row["component_review_flag"] = flag_by_id.get(str(row.get("component_id", "")), "")
        row["range_review_flag"] = range_flag_by_id.get(str(row.get("component_id", "")), "")

    heuristic_fields = [
        "heuristic_source_key", "family", "table_dir_key", "n_records", "n_unique_pdf_pages",
        "page_range", "numeric_gap_count_within_group", "numeric_gaps_within_group",
        "n_distinct_titles", "mixed_titles", "n_distinct_families", "prior_basis_counts",
        "primary_title", "example_pages",
    ]
    missing_rows = [r for r in page_rows if row_has_missing_ocr(r)]
    review_rows = [r for r in page_rows if row_has_ocr_review(r)]

    add_sheet(wb, "Components", summary_rows, component_fields)
    add_sheet(wb, "Pages", page_rows, page_fields)
    add_sheet(wb, "HeuristicGroups", heuristic_rows, heuristic_fields)
    if refined_core_rows:
        add_sheet(wb, "RefinedCoreRuns", refined_core_rows, list(refined_core_rows[0].keys()))
    add_sheet(wb, "MissingOCR", missing_rows, page_fields)
    add_sheet(wb, "OCRReviewPages", review_rows, page_fields)
    if unocrd_rows:
        add_sheet(wb, "UnOCRdOriginal", unocrd_rows, list(unocrd_rows[0].keys()))
    wb.close()


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_original_source_page_lookup(page_rows: list[dict[str, object]]) -> dict[tuple[str, str], dict[str, str]]:
    lookup: dict[tuple[str, str], dict[str, str]] = {}
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in page_rows:
        if row.get("source_record_type") == "master_tracking_page" and row.get("source_pdf") and row.get("pdf_page"):
            grouped[(str(row.get("source_pdf")), str(row.get("pdf_page")))].append(row)
    for key, rows in grouped.items():
        lookup[key] = {
            "original_overlap_titles": uniq_join([str(r.get("title", "")) for r in rows], limit=5),
            "original_overlap_families": uniq_join([str(r.get("family", "")) for r in rows], limit=5),
            "original_overlap_component_ids": uniq_join([str(r.get("component_id", "")) for r in rows], limit=5),
            "original_overlap_ocr_statuses": counter_json([str(r.get("ocr_status", "")) for r in rows]),
        }
    return lookup


def load_core_page_lookup() -> dict[tuple[str, str], dict[str, str]]:
    lookup: dict[tuple[str, str], dict[str, str]] = {}
    for row in read_csv(CORE_PAGES):
        lookup[(row.get("table_key", ""), row.get("pdf_page", ""))] = row
    return lookup


def classify_core_page(row: dict[str, object]) -> str:
    status = str(row.get("ocr_status", "")).lower()
    title = str(row.get("core_ocr_title") or row.get("original_overlap_titles") or row.get("title") or "")
    t = title.lower()
    rows_extracted = int_or_none(str(row.get("core_rows") or row.get("rows_extracted") or "0")) or 0

    if status in {"no_ocr_json", "NO_OCR_JSON".lower()}:
        return "missing_ocr"
    if status == "empty" or re.search(r"\b(blank|no table present|no table found|contents|corrigenda|map showing|diagram showing)\b", t):
        return "empty_or_non_table"

    target_terms = r"civil condition|age,?\s*sex|age and civil|table\s+vii|c\.?\s*iii|c-iii|married|widowed|unmarried"
    if re.search(target_terms, t):
        if re.search(r"\bchapter\b", t) and "table" not in t:
            return "target_context"
        return "target_money_table"
    if "table" in t:
        return "other_table"
    if rows_extracted > 0:
        return "structured_context"
    if status == "covered_by_original_corpus_ocr":
        return "covered_original_context"
    return "context_unknown"


def assign_refined_core_basis(page_class: str) -> str:
    if page_class == "target_money_table":
        return "core_money_refined_target_table_run"
    if page_class in {"other_table"}:
        return "core_money_refined_other_table_run"
    if page_class in {"missing_ocr"}:
        return "core_money_refined_missing_ocr_run"
    if page_class in {"empty_or_non_table"}:
        return "core_money_refined_empty_or_non_table_run"
    return "core_money_refined_context_run"


def refine_overbroad_core_money_ranges(
    component_rows: list[dict[str, object]],
    page_rows: list[dict[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    """Split only core-money ranges flagged as mixed/overbroad into typed page runs."""
    core_lookup = load_core_page_lookup()
    original_lookup = build_original_source_page_lookup(page_rows)
    overbroad_ids = {
        str(r["component_id"])
        for r in component_rows
        if "core_money_range_mixed_or_overbroad" in str(r.get("range_review_flag", ""))
    }
    if not overbroad_ids:
        return component_rows, page_rows, []

    enriched: list[dict[str, object]] = []
    for row in page_rows:
        new = dict(row)
        if row.get("component_basis") == "core_money_table_key":
            meta = core_lookup.get((str(row.get("source_group_key", "")), str(row.get("pdf_page", ""))), {})
            new["core_ocr_title"] = meta.get("ocr_title", "")
            new["core_ocr_status"] = meta.get("ocr_status", "")
            new["core_ocr_decision"] = meta.get("ocr_decision", "")
            new["core_source_kinds"] = meta.get("source_kinds", "")
            new["core_rows"] = meta.get("rows", "")
            new["core_sections"] = meta.get("sections", "")
            new["core_constraint_failed"] = meta.get("constraint_failed", "")
            overlap = original_lookup.get((str(row.get("source_pdf", "")), str(row.get("pdf_page", ""))), {})
            new.update(overlap)
            new["core_page_class"] = classify_core_page(new)
        enriched.append(new)

    refined_runs: list[dict[str, object]] = []
    output_pages: list[dict[str, object]] = []
    by_component: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in enriched:
        by_component[str(row.get("component_id", ""))].append(row)

    for old_component_id, rows in by_component.items():
        if old_component_id not in overbroad_ids:
            output_pages.extend(rows)
            continue

        rows = sorted(rows, key=page_sort_tuple)
        run: list[dict[str, object]] = []
        run_class = ""
        last_page: int | None = None

        def flush() -> None:
            nonlocal run, run_class
            if not run:
                return
            pages = [str(r.get("pdf_page", "")) for r in run]
            start = pages[0] if pages else ""
            end = pages[-1] if pages else ""
            parent_key = str(run[0].get("source_group_key", ""))
            basis = assign_refined_core_basis(run_class)
            component_id = stable_id("corefine", parent_key, run_class, start, end, len(run))
            source_group = f"{parent_key}||{run_class}||{start}-{end}"
            for idx, r in enumerate(run, start=1):
                r["parent_component_id"] = old_component_id
                r["parent_source_group_key"] = parent_key
                r["component_id"] = component_id
                r["component_basis"] = basis
                r["source_group_key"] = source_group
                r["position_in_component"] = idx
                output_pages.append(r)
            refined_runs.append({
                "parent_component_id": old_component_id,
                "parent_source_group_key": parent_key,
                "component_id": component_id,
                "component_basis": basis,
                "core_page_class": run_class,
                "n_records": len(run),
                "source_pdf": run[0].get("source_pdf", ""),
                "year": run[0].get("year", ""),
                "province": run[0].get("province", ""),
                "page_range": f"{start}-{end}",
                "ocr_status_counts": counter_json([str(r.get("ocr_status", "")) for r in run]),
                "core_titles": uniq_join([str(r.get("core_ocr_title") or r.get("original_overlap_titles") or r.get("title") or "") for r in run], limit=4),
            })
            run = []
            run_class = ""

        for row in rows:
            page_class = str(row.get("core_page_class", "context_unknown"))
            page_num = int_or_none(str(row.get("pdf_page", "")))
            is_contiguous = last_page is None or page_num is None or page_num == last_page + 1
            if run and (page_class != run_class or not is_contiguous):
                flush()
            run.append(row)
            run_class = page_class
            if page_num is not None:
                last_page = page_num
        flush()

    summaries = summarize_components_from_pages(output_pages)
    return summaries, sorted(output_pages, key=lambda r: (str(r.get("component_id", "")), int(r.get("position_in_component") or 0))), refined_runs


def summarize_components_from_pages(page_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in page_rows:
        grouped[str(row.get("component_id", ""))].append(row)

    summaries: list[dict[str, object]] = []
    for cid, rows in grouped.items():
        rows = sorted(rows, key=page_sort_tuple)
        for i, row in enumerate(rows, start=1):
            row["position_in_component"] = i
        basis = str(rows[0].get("component_basis", ""))
        scope = str(rows[0].get("component_scope", ""))
        source_group = str(rows[0].get("source_group_key", ""))
        titles = [str(r.get("core_ocr_title") or r.get("original_overlap_titles") or r.get("title", "")) for r in rows]
        families = [str(r.get("family", "")) for r in rows]
        source_pdfs = [str(r.get("source_pdf", "")) for r in rows]
        provinces = [str(r.get("province", "")) for r in rows]
        years = [str(r.get("year", "")) for r in rows]
        pages = [str(r.get("pdf_page", "")) for r in rows]
        table_dirs = [str(r.get("table_dir_key", "")) for r in rows]
        prange, gap_n, gaps = page_gap_summary(rows)
        page_counter = Counter(p for p in pages if p)
        duplicate_pages = [p for p, n in page_counter.items() if n > 1]
        ocr_statuses = [str(r.get("ocr_status") or r.get("page_status") or "") for r in rows]
        source_types = [str(r.get("source_record_type", "")) for r in rows]
        prior_bases = [str(r.get("v1_component_basis", "")) for r in rows]
        row_records = sum(int_or_none(str(r.get("rows_extracted", ""))) or 0 for r in rows)
        n_artifacts = sum(1 for r in rows if str(r.get("ocr_artifact", "")).strip())
        n_covered = sum(1 for r in rows if str(r.get("ocr_status", "")) == "covered_by_original_corpus_ocr")
        n_overlap = sum(1 for r in rows if str(r.get("overlaps_original_ocr_source_page", "")).lower() == "true")
        range_flag = component_range_review_flag(basis, rows, titles, ocr_statuses, gap_n, duplicate_pages)
        if basis.startswith("core_money_refined_"):
            # The refinement itself resolves the overbroad-range flag; keep only true page-order warnings.
            range_flag = "; ".join(f for f in range_flag.split("; ") if f and f != "core_money_range_mixed_or_overbroad")

        summaries.append({
            "component_id": cid,
            "component_scope": scope,
            "component_basis": basis,
            "review_flag": review_flag(basis, rows),
            "range_review_flag": range_flag,
            "source_group_key": source_group,
            "n_records": len(rows),
            "n_unique_source_pages": len({(r.get("source_pdf", ""), r.get("pdf_page", "")) for r in rows if r.get("pdf_page")}),
            "n_ocr_artifacts": n_artifacts,
            "n_covered_by_original_ocr": n_covered,
            "n_overlap_with_original_ocr_source_page": n_overlap,
            "row_records_from_json_pages": row_records,
            "years": uniq_join(years),
            "provinces": uniq_join(provinces),
            "source_pdfs": uniq_join(source_pdfs),
            "primary_family": Counter(families).most_common(1)[0][0] if families else "",
            "families": uniq_join(families),
            "primary_title": Counter(titles).most_common(1)[0][0] if titles else "",
            "n_distinct_titles": len({t for t in titles if t}),
            "mixed_titles": len({t for t in titles if t}) > 1,
            "table_dir_keys": uniq_join(table_dirs, limit=3),
            "page_range": prange,
            "numeric_gap_count_within_component": gap_n,
            "numeric_gaps_within_component": gaps,
            "duplicate_pdf_pages": uniq_join(duplicate_pages, limit=20),
            "ocr_status_counts": counter_json(ocr_statuses),
            "source_record_type_counts": counter_json(source_types),
            "v1_component_basis_counts": counter_json(prior_bases),
            "example_images": uniq_join([str(r.get("image_path", "")) for r in rows], limit=3),
        })

    summaries.sort(key=lambda r: (
        str(r["review_flag"]) not in {"missing_ocr_artifact", "ocr_review", "heuristic_review", "weak_grouping", "singleton"},
        str(r["review_flag"]),
        str(r.get("range_review_flag", "")) == "",
        -int(r["n_records"]),
        str(r["component_id"]),
    ))
    return summaries


def build_no_gaps_ledger(summary: dict[str, object], component_rows: list[dict[str, object]], page_rows: list[dict[str, object]], unocrd_rows: list[dict[str, str]]) -> None:
    LEDGER_DIR.mkdir(parents=True, exist_ok=True)
    source_missing = read_csv(STATIC_MISSING)
    source_now_local = read_csv(STATIC_NOW_LOCAL)
    core_ready = read_csv(CORE_READY)

    flag_by_id = {str(r["component_id"]): str(r["review_flag"]) for r in component_rows}
    range_flag_by_id = {str(r["component_id"]): str(r.get("range_review_flag", "")) for r in component_rows}
    for row in page_rows:
        row["component_review_flag"] = flag_by_id.get(str(row.get("component_id", "")), "")
        row["range_review_flag"] = range_flag_by_id.get(str(row.get("component_id", "")), "")

    missing_ocr = [r for r in page_rows if row_has_missing_ocr(r)]
    ocr_review = [r for r in page_rows if row_has_ocr_review(r)]
    assembly_review = [
        r for r in component_rows
        if str(r.get("review_flag", "")) in {"heuristic_review", "singleton", "weak_grouping", "missing_ocr_artifact", "ocr_review"}
        or str(r.get("range_review_flag", ""))
    ]

    write_csv(LEDGER_DIR / "source_pages_still_need_render.csv", source_missing, list(source_missing[0].keys()) if source_missing else [])
    write_csv(LEDGER_DIR / "source_pages_now_local_from_core_money.csv", source_now_local, list(source_now_local[0].keys()) if source_now_local else [])
    write_csv(LEDGER_DIR / "local_core_money_renders_ready_for_ocr.csv", core_ready, list(core_ready[0].keys()) if core_ready else [])
    write_csv(LEDGER_DIR / "ocr_missing_or_not_covered.csv", missing_ocr, list(page_rows[0].keys()) if page_rows else [])
    write_csv(LEDGER_DIR / "ocr_quality_review_pages.csv", ocr_review, list(page_rows[0].keys()) if page_rows else [])
    write_csv(LEDGER_DIR / "assembly_components_needing_review.csv", assembly_review, list(component_rows[0].keys()) if component_rows else [])
    if unocrd_rows:
        shutil.copyfile(V1_UNOCRD, LEDGER_DIR / "original_master_pages_not_ocrd.csv")

    ledger_summary = dict(summary)
    ledger_summary.update({
        "static_site_source_pages_still_need_render": len(source_missing),
        "static_site_source_pages_now_local_from_core_money": len(source_now_local),
        "local_core_money_renders_ready_for_ocr": len(core_ready),
        "component_pages_missing_ocr_or_not_covered": len(missing_ocr),
        "component_pages_with_ocr_quality_review": len(ocr_review),
        "components_needing_assembly_or_quality_review": len(assembly_review),
        "components_with_range_review_flag": sum(1 for r in component_rows if str(r.get("range_review_flag", ""))),
        "original_master_pages_not_ocrd": len(unocrd_rows),
    })
    (LEDGER_DIR / "no_gaps_summary.json").write_text(json.dumps(ledger_summary, indent=2), encoding="utf-8")

    wb = xlsxwriter.Workbook(str(LEDGER_DIR / "no_gaps_ledger.xlsx"), {"constant_memory": True})
    summary_rows = [{"metric": k, "value": v} for k, v in ledger_summary.items()]
    add_sheet(wb, "Summary", summary_rows, ["metric", "value"])
    if source_missing:
        add_sheet(wb, "NeedRender", source_missing, list(source_missing[0].keys()))
    if source_now_local:
        add_sheet(wb, "NowLocal", source_now_local, list(source_now_local[0].keys()))
    if core_ready:
        add_sheet(wb, "ReadyForOCR", core_ready, list(core_ready[0].keys()))
    if missing_ocr:
        add_sheet(wb, "MissingOCR", missing_ocr, list(page_rows[0].keys()))
    if ocr_review:
        add_sheet(wb, "OCRReview", ocr_review, list(page_rows[0].keys()))
    if assembly_review:
        add_sheet(wb, "AssemblyReview", assembly_review, list(component_rows[0].keys()))
    if unocrd_rows:
        add_sheet(wb, "OriginalNotOCRd", unocrd_rows, list(unocrd_rows[0].keys()))
    wb.close()


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = read_csv(V1_PAGES)
    unocrd_rows = read_csv(V1_UNOCRD)
    component_rows, page_rows, heuristic_rows = build_components(rows)
    component_rows, page_rows, refined_core_rows = refine_overbroad_core_money_ranges(component_rows, page_rows)
    flag_by_id = {str(r["component_id"]): str(r["review_flag"]) for r in component_rows}
    range_flag_by_id = {str(r["component_id"]): str(r.get("range_review_flag", "")) for r in component_rows}
    for row in page_rows:
        row["component_review_flag"] = flag_by_id.get(str(row.get("component_id", "")), "")
        row["range_review_flag"] = range_flag_by_id.get(str(row.get("component_id", "")), "")

    component_fields = [
        "component_id", "component_scope", "component_basis", "review_flag", "range_review_flag", "source_group_key",
        "n_records", "n_unique_source_pages", "n_ocr_artifacts", "n_covered_by_original_ocr",
        "n_overlap_with_original_ocr_source_page", "row_records_from_json_pages", "years",
        "provinces", "source_pdfs", "primary_family", "families", "primary_title",
        "n_distinct_titles", "mixed_titles", "table_dir_keys", "page_range",
        "numeric_gap_count_within_component", "numeric_gaps_within_component",
        "duplicate_pdf_pages", "ocr_status_counts", "source_record_type_counts",
        "v1_component_basis_counts", "example_images",
    ]
    page_fields = [
        "component_id", "position_in_component", "component_scope", "component_basis", "component_review_flag", "range_review_flag", "source_group_key",
        "source_record_type", "year", "province", "source_pdf", "pdf_page", "printed_page",
        "page_sort_key", "family", "title", "table_number", "table_dir_key", "table_id", "imghash",
        "image_path", "ocr_artifact", "ocr_status", "page_status", "validation", "verdict",
        "assembled_table", "position_in_table", "proposed_allread_group_id", "improved_group_id",
        "rows_extracted", "sections_extracted", "constraint_checks", "constraint_failed",
        "overlaps_original_ocr_source_page", "linked_original_component_ids", "linked_original_table_ids",
        "v1_component_id", "v1_component_basis", "heuristic_source_key",
        "parent_component_id", "parent_source_group_key", "core_page_class",
        "core_ocr_title", "core_ocr_status", "core_ocr_decision", "core_source_kinds",
        "core_rows", "core_sections", "core_constraint_failed",
        "original_overlap_titles", "original_overlap_families", "original_overlap_component_ids",
        "original_overlap_ocr_statuses",
    ]
    refined_core_fields = [
        "parent_component_id", "parent_source_group_key", "component_id", "component_basis",
        "core_page_class", "n_records", "source_pdf", "year", "province", "page_range",
        "ocr_status_counts", "core_titles",
    ]
    heuristic_fields = [
        "heuristic_source_key", "family", "table_dir_key", "n_records", "n_unique_pdf_pages",
        "page_range", "numeric_gap_count_within_group", "numeric_gaps_within_group",
        "n_distinct_titles", "mixed_titles", "n_distinct_families", "prior_basis_counts",
        "primary_title", "example_pages",
    ]

    write_csv(OUT_DIR / "all_ocr_component_summary.csv", component_rows, component_fields)
    write_csv(OUT_DIR / "all_ocr_component_pages.csv", page_rows, page_fields)
    write_csv(OUT_DIR / "heuristic_group_candidates.csv", heuristic_rows, heuristic_fields)
    write_csv(OUT_DIR / "refined_core_money_runs.csv", refined_core_rows, refined_core_fields)
    write_jsonl(OUT_DIR / "all_ocr_components.jsonl", component_rows)
    if V1_UNOCRD.exists():
        shutil.copyfile(V1_UNOCRD, OUT_DIR / "all_ocr_unocrd_master_pages.csv")

    summary = {
        "input_v1_page_records": len(rows),
        "component_page_records_total": len(page_rows),
        "components_total": len(component_rows),
        "heuristic_group_candidates": len(heuristic_rows),
        "heuristic_grouped_records": sum(int(r["n_records"]) for r in heuristic_rows),
        "refined_core_money_runs": len(refined_core_rows),
        "refined_core_money_run_class_counts": dict(Counter(str(r.get("core_page_class", "")) for r in refined_core_rows)),
        "component_basis_counts": dict(Counter(str(r["component_basis"]) for r in component_rows)),
        "component_scope_counts": dict(Counter(str(r["component_scope"]) for r in component_rows)),
        "review_flag_counts": dict(Counter(str(r["review_flag"]) for r in component_rows)),
        "range_review_flag_counts": dict(Counter(str(r.get("range_review_flag", "")) for r in component_rows if str(r.get("range_review_flag", "")))),
        "page_source_record_type_counts": dict(Counter(str(r.get("source_record_type", "")) for r in page_rows)),
        "v1_basis_counts_on_pages": dict(Counter(str(r.get("v1_component_basis", "")) for r in page_rows)),
        "v2_basis_counts_on_pages": dict(Counter(str(r.get("component_basis", "")) for r in page_rows)),
    }
    (OUT_DIR / "all_ocr_component_assembly_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    write_workbook(OUT_DIR / "all_ocr_component_assembly_review.xlsx", component_rows, page_rows, heuristic_rows, refined_core_rows, unocrd_rows)
    build_no_gaps_ledger(summary, component_rows, page_rows, unocrd_rows)

    print(json.dumps(summary, indent=2))
    print(f"wrote {OUT_DIR}")
    print(f"wrote {LEDGER_DIR}")


if __name__ == "__main__":
    main()
