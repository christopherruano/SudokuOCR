#!/usr/bin/env python3
"""Audit likely missed/overbroad assembly opportunities.

This is a QA layer over v34/v35. It does not OCR, does not promote groups, and
does not change existing outputs. It answers two questions:

1. Are odd/no-row-label continuation signals captured or explained?
2. Are logical/religion families bounded, or are there missing/overbroad risks?
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
DEFAULT_SEQUENCE_DIR = ROOT / "export_final/pdf_recovery_incoming_2026-06-16_first_principles_source_sequence_map_v35"
DEFAULT_LOGICAL_DIR = ROOT / "export_final/pdf_recovery_incoming_2026-06-16_first_principles_logical_family_frontier_v32"
DEFAULT_OUT_DIR = ROOT / "export_final/pdf_recovery_incoming_2026-06-16_first_principles_assembly_quality_audit_v36"


RELIGION_PATTERNS = [
    ("all_religions", r"\ball\s+religions?\b"),
    ("hindu", r"\bhindus?\b|\bhindu\b|\bbrahmanic\b"),
    ("muslim", r"\bmusalman(s)?\b|\bmuhamm?adan(s)?\b|\bmuslim(s)?\b"),
    ("christian", r"\bchristian(s)?\b"),
    ("animistic", r"\banimistic\b|\banimist(s)?\b"),
    ("jew", r"\bjew(s)?\b|\bjewish\b"),
    ("jain", r"\bjain(s)?\b"),
    ("sikh", r"\bsikh(s)?\b"),
    ("buddhist", r"\bbuddhist(s)?\b|\bbudhist\b"),
    ("parsi", r"\bparsi(s)?\b|\bzoroastrian(s)?\b"),
    ("brahmo", r"\bbrahmo\b|\bbramho\b"),
    ("unspecified_religion", r"\bunspecified\b"),
]


ROW_LABEL_FIELDS = [
    "audit_rank",
    "audit_id",
    "coverage_bucket",
    "priority",
    "v5_source_key",
    "source_pdf_id_guess",
    "page_pair",
    "prev_pdf_page",
    "pdf_page",
    "right_page_parity",
    "physical_rule_status",
    "site_action",
    "row_label_dependency_signal",
    "left_row_label_ratio",
    "right_row_label_ratio",
    "row_label_scan_pattern",
    "same_table_family",
    "same_table_number",
    "page_kind_transition",
    "page_class_transition",
    "classifier_break_signal",
    "title_switch_signal",
    "potential_missed_signal",
    "explanation",
    "rule_flags",
    "prev_title",
    "title",
    "prev_image_path",
    "image_path",
]

LOGICAL_FIELDS = [
    "audit_rank",
    "audit_id",
    "audit_bucket",
    "priority",
    "family_id",
    "group_id",
    "group_status",
    "v5_source_key",
    "source_pdf_id_guess",
    "page_range",
    "n_pages",
    "member_count",
    "religion_terms_detected",
    "distinct_religion_count",
    "page_terms_detected",
    "pages_are_consecutive",
    "has_page_ranges",
    "overbroad_risk",
    "missing_family_risk",
    "recommended_action",
    "primary_title",
]

RELIGION_SEQUENCE_FIELDS = [
    "sequence_rank",
    "sequence_id",
    "coverage_bucket",
    "v5_source_key",
    "source_pdf_id_guess",
    "page_range",
    "component_pages",
    "n_pages",
    "table_family_key",
    "table_number_key",
    "religion_terms_detected",
    "distinct_religion_count",
    "covered_logical_group_ids",
    "covered_family_ids",
    "same_table_frame",
    "recommended_action",
    "titles",
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


def float_or_none(value: object) -> float | None:
    if value in {None, ""}:
        return None
    try:
        return float(str(value))
    except ValueError:
        return None


def semi(values: Iterable[object]) -> str:
    return "; ".join(str(value) for value in values if value not in {None, ""})


def split_semicolon(value: object) -> list[str]:
    return [part.strip() for part in str(value or "").split(";") if part.strip()]


def clean_text(value: object) -> str:
    text = str(value or "").lower().replace("&", " and ")
    text = re.sub(r"[\u2010-\u2015]", "-", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def extract_religions(title: object) -> set[str]:
    text = clean_text(title)
    out: set[str] = set()
    for key, pattern in RELIGION_PATTERNS:
        if re.search(pattern, text):
            out.add(key)
    return out


def parse_pages(value: object) -> list[int]:
    pages: list[int] = []
    for token in re.findall(r"\d+", str(value or "")):
        page = int_or_none(token)
        if page is not None:
            pages.append(page)
    return pages


def pages_consecutive(pages: list[int]) -> bool:
    if not pages:
        return False
    pages = sorted(set(pages))
    return pages == list(range(pages[0], pages[-1] + 1))


def ratio_has_labels(value: object) -> bool:
    ratio = float_or_none(value)
    return ratio is not None and ratio >= 0.50


def edge_considered_row_label(edge: dict[str, str]) -> bool:
    if edge.get("right_page_no_row_labels") != "yes":
        return False
    if edge.get("right_page_parity") == "odd":
        return True
    if edge.get("row_label_scan_pattern") == "left_labelled_right_unlabelled":
        return True
    if edge.get("row_label_dependency_signal") == "yes":
        return True
    if edge.get("same_table_family") == "yes" and ratio_has_labels(edge.get("left_row_label_ratio")):
        return True
    return False


def row_label_bucket(edge: dict[str, str]) -> tuple[str, int, str, str]:
    status = edge.get("physical_rule_status", "")
    flags = set(split_semicolon(edge.get("rule_flags", "")))
    if edge.get("row_label_dependency_signal") == "yes" and status == "candidate_row_label_dependency":
        return "captured_candidate_row_label_dependency", 1, "no", "Captured as the primary row-label dependency queue."
    if edge.get("row_label_dependency_signal") == "yes" and status == "unpromoted_row_label_dependency_signal":
        return "captured_manual_row_label_signal", 2, "no", "Captured as an unpromoted row-label signal for manual review."
    if edge.get("row_label_dependency_signal") == "yes" and status == "failed_or_partial_physical_review":
        return "captured_constraint_repair_row_label_signal", 3, "no", "Captured but current constraints failed or are partial."
    if status == "confirmed_physical_edge":
        return "already_confirmed_physical_edge", 9, "no", "Already inside a confirmed physical edge."
    if status in {"confirmed_logical_only_edge", "review_logical_only_edge"}:
        return "logical_only_no_physical_merge", 7, "no", "Logical relationship only; not a row-label physical assembly."
    if status == "likely_assemblable_after_missing_pages":
        return "blocked_by_missing_intermediate_pages", 4, "no", "Non-consecutive bridge; missing intermediate pages must be restored first."
    if status == "candidate_physical_missing_ocr":
        return "physical_candidate_missing_ocr_no_rowlabel_signal", 5, "no", "Candidate is blocked on OCR artifacts, not row-label inheritance evidence."
    if "context_boundary" in flags or "derivative_boundary" in flags or "offtarget_boundary" in flags:
        return "explained_context_derivative_or_offtarget_break", 9, "no", "Break is explained by context, derivative, or off-target page type."
    if edge.get("classifier_break_signal") == "yes":
        return "explained_classifier_break", 9, "no", "Classifier marks a page-class break."
    if edge.get("same_table_family") != "yes":
        return "explained_table_family_switch", 8, "no", "Adjacent pages do not share a table-family key."
    if not ratio_has_labels(edge.get("left_row_label_ratio")):
        return "explained_no_left_row_labels_to_inherit", 8, "no", "Left page does not provide a clear row-label source."
    if edge.get("title_switch_signal") in {"new_title_or_table_switch", "same_family_title_change"}:
        return "review_possible_title_switch", 6, "yes", "Potential row-label signal, but title/subtitle changed."
    return "possible_missed_row_label_signal", 1, "yes", "Right page lacks labels, left appears labelled, and no hard exclusion explains it."


def build_row_label_audit(edges: list[dict[str, str]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for edge in edges:
        if not edge_considered_row_label(edge):
            continue
        bucket, priority, missed, explanation = row_label_bucket(edge)
        rows.append(
            {
                "audit_id": stable_id("v36row", edge.get("rule_id", "")),
                "coverage_bucket": bucket,
                "priority": priority,
                "v5_source_key": edge.get("v5_source_key", ""),
                "source_pdf_id_guess": edge.get("source_pdf_id_guess", ""),
                "page_pair": edge.get("page_pair", ""),
                "prev_pdf_page": edge.get("prev_pdf_page", ""),
                "pdf_page": edge.get("pdf_page", ""),
                "right_page_parity": edge.get("right_page_parity", ""),
                "physical_rule_status": edge.get("physical_rule_status", ""),
                "site_action": edge.get("site_action", ""),
                "row_label_dependency_signal": edge.get("row_label_dependency_signal", ""),
                "left_row_label_ratio": edge.get("left_row_label_ratio", ""),
                "right_row_label_ratio": edge.get("right_row_label_ratio", ""),
                "row_label_scan_pattern": edge.get("row_label_scan_pattern", ""),
                "same_table_family": edge.get("same_table_family", ""),
                "same_table_number": edge.get("same_table_number", ""),
                "page_kind_transition": edge.get("page_kind_transition", ""),
                "page_class_transition": edge.get("page_class_transition", ""),
                "classifier_break_signal": edge.get("classifier_break_signal", ""),
                "title_switch_signal": edge.get("title_switch_signal", ""),
                "potential_missed_signal": missed,
                "explanation": explanation,
                "rule_flags": edge.get("rule_flags", ""),
                "prev_title": edge.get("prev_title", ""),
                "title": edge.get("title", ""),
                "prev_image_path": edge.get("prev_image_path", ""),
                "image_path": edge.get("image_path", ""),
            }
        )
    rows.sort(key=lambda row: (int(row["priority"]), str(row["v5_source_key"]), int_or_none(row["prev_pdf_page"]) or 10**9))
    for idx, row in enumerate(rows, start=1):
        row["audit_rank"] = idx
    return rows


def logical_family_page_index(members: list[dict[str, str]]) -> dict[tuple[str, str], set[str]]:
    index: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in members:
        src = row.get("v5_source_key", "")
        page = row.get("pdf_page", "")
        family_id = row.get("family_id", "")
        if src and page and family_id:
            index[(src, page)].add(family_id)
    return index


def build_logical_family_audit(groups: list[dict[str, str]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for group in groups:
        if group.get("group_type") != "logical_family_collection":
            continue
        pages = parse_pages(group.get("component_pages") or group.get("page_range"))
        terms = set(split_semicolon(group.get("religion_terms_detected", "")))
        page_terms = extract_religions(group.get("primary_title", ""))
        distinct_count = len(terms or page_terms)
        consecutive = pages_consecutive(pages)
        n_pages = int_or_none(group.get("n_pages")) or len(set(pages))
        overbroad = "no"
        bucket = "logical_family_ok"
        priority = 9
        action = group.get("recommended_action", "")
        if not consecutive:
            overbroad = "yes"
            bucket = "logical_family_nonconsecutive_review"
            priority = 2
            action = "Split or review: logical families should be bounded by consecutive observed pages."
        elif group.get("group_status") == "review" and n_pages >= 15:
            overbroad = "yes"
            bucket = "large_review_family_overbreadth_risk"
            priority = 3
            action = "Review/split before using; large same-title families are high overmerge risk."
        elif group.get("group_status") == "review" and distinct_count <= 1 and n_pages >= 8:
            overbroad = "yes"
            bucket = "single_variant_large_review_family"
            priority = 4
            action = "Review whether this is one table body, repeated subtable pages, or overbroad title grouping."
        elif group.get("group_status") == "confirmed":
            bucket = "confirmed_logical_family"
            priority = 8
        elif distinct_count >= 2:
            bucket = "religion_family_review_candidate"
            priority = 5
        else:
            bucket = "logical_review_candidate"
            priority = 7
        rows.append(
            {
                "audit_id": stable_id("v36log", group.get("group_id", "")),
                "audit_bucket": bucket,
                "priority": priority,
                "family_id": group.get("group_id", "").replace("logical::", ""),
                "group_id": group.get("group_id", ""),
                "group_status": group.get("group_status", ""),
                "v5_source_key": group.get("v5_source_key", ""),
                "source_pdf_id_guess": group.get("source_pdf_id_guess", ""),
                "page_range": group.get("page_range", ""),
                "n_pages": n_pages,
                "member_count": group.get("member_count", ""),
                "religion_terms_detected": group.get("religion_terms_detected", ""),
                "distinct_religion_count": distinct_count,
                "page_terms_detected": semi(sorted(page_terms)),
                "pages_are_consecutive": "yes" if consecutive else "no",
                "has_page_ranges": "yes" if "-" in group.get("page_range", "") else "no",
                "overbroad_risk": overbroad,
                "missing_family_risk": "no",
                "recommended_action": action,
                "primary_title": group.get("primary_title", ""),
            }
        )
    rows.sort(key=lambda row: (int(row["priority"]), str(row["v5_source_key"]), str(row["page_range"])))
    for idx, row in enumerate(rows, start=1):
        row["audit_rank"] = idx
    return rows


def build_religion_sequences(pages: list[dict[str, str]], member_index: dict[tuple[str, str], set[str]], groups: list[dict[str, str]]) -> list[dict[str, object]]:
    group_by_id = {row.get("group_id", "").replace("logical::", ""): row for row in groups if row.get("group_type") == "logical_family_collection"}
    pages_by_source: dict[str, list[dict[str, str]]] = defaultdict(list)
    for page in pages:
        pages_by_source[page.get("v5_source_key", "")].append(page)
    rows: list[dict[str, object]] = []
    for src in sorted(pages_by_source):
        ordered = sorted(pages_by_source[src], key=lambda row: int_or_none(row.get("pdf_page")) or 10**9)
        current: list[dict[str, str]] = []
        current_frame = ("", "")

        def flush() -> None:
            nonlocal current, current_frame
            if len(current) < 2:
                current = []
                current_frame = ("", "")
                return
            pages_int = [int_or_none(row.get("pdf_page")) for row in current]
            pages_clean = [page for page in pages_int if page is not None]
            terms = sorted(set(term for row in current for term in extract_religions(row.get("title", ""))))
            if len(terms) < 2:
                current = []
                current_frame = ("", "")
                return
            covered_families = sorted(set(fid for row in current for fid in member_index.get((src, row.get("pdf_page", "")), set())))
            covered_groups = sorted(set(f"logical::{fid}" for fid in covered_families))
            family_key = current_frame[0]
            table_number = current_frame[1]
            core_table_signal = (
                "age_civil" in family_key
                or "main_age_civil" in family_key
                or "table_vii" in family_key
                or table_number == "7"
            )
            off_target_signal = "offtarget" in family_key or family_key == "unknown"
            if covered_families:
                bucket = "covered_religion_sequence"
                action = "Already covered by logical family member index; review group status before site use."
            elif off_target_signal or not core_table_signal:
                bucket = "non_core_religion_sequence_review"
                action = "Recorded for completeness, but do not treat as a missing core Table VII/age-civil family without manual review."
            else:
                bucket = "possible_missing_religion_family"
                action = "Review as a bounded logical-only religion family candidate."
            rows.append(
                {
                    "sequence_id": stable_id("v36rel", src, semi(pages_clean), semi(terms)),
                    "coverage_bucket": bucket,
                    "v5_source_key": src,
                    "source_pdf_id_guess": current[0].get("source_pdf_id_guess", ""),
                    "page_range": f"{pages_clean[0]}-{pages_clean[-1]}" if pages_clean else "",
                    "component_pages": semi(pages_clean),
                    "n_pages": len(pages_clean),
                    "table_family_key": current_frame[0],
                    "table_number_key": current_frame[1],
                    "religion_terms_detected": semi(terms),
                    "distinct_religion_count": len(terms),
                    "covered_logical_group_ids": semi(covered_groups),
                    "covered_family_ids": semi(covered_families),
                    "same_table_frame": "yes",
                    "recommended_action": action,
                    "titles": semi(row.get("title", "") for row in current),
                }
            )
            current = []
            current_frame = ("", "")

        previous_page: int | None = None
        for page in ordered:
            page_num = int_or_none(page.get("pdf_page"))
            terms = extract_religions(page.get("title", ""))
            frame = (page.get("table_family_key", ""), page.get("table_number_key", ""))
            eligible = bool(terms) and bool(frame[0])
            consecutive = previous_page is not None and page_num == previous_page + 1
            if not eligible or (current and (not consecutive or frame != current_frame)):
                flush()
            if eligible:
                if not current:
                    current_frame = frame
                current.append(page)
            previous_page = page_num
        flush()

    rows.sort(key=lambda row: (0 if row["coverage_bucket"] == "possible_missing_religion_family" else 1, str(row["v5_source_key"]), int_or_none(str(row["page_range"]).split("-", 1)[0]) or 10**9))
    for idx, row in enumerate(rows, start=1):
        row["sequence_rank"] = idx
    return rows


def build_summary(
    row_label_rows: list[dict[str, object]],
    logical_rows: list[dict[str, object]],
    religion_rows: list[dict[str, object]],
    paths: dict[str, Path],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = [
        {"metric": "paths", "name": name, "count": output_path(path)}
        for name, path in paths.items()
    ]
    rows.extend(
        [
            {"metric": "counts", "name": "row_label_audit_rows", "count": len(row_label_rows)},
            {"metric": "counts", "name": "logical_family_audit_rows", "count": len(logical_rows)},
            {"metric": "counts", "name": "religion_sequence_rows", "count": len(religion_rows)},
        ]
    )
    for metric, counter in [
        ("row_label_by_bucket", Counter(str(row.get("coverage_bucket", "")) for row in row_label_rows)),
        ("row_label_potential_missed", Counter(str(row.get("potential_missed_signal", "")) for row in row_label_rows)),
        ("logical_by_bucket", Counter(str(row.get("audit_bucket", "")) for row in logical_rows)),
        ("logical_overbroad_risk", Counter(str(row.get("overbroad_risk", "")) for row in logical_rows)),
        ("religion_sequence_by_bucket", Counter(str(row.get("coverage_bucket", "")) for row in religion_rows)),
    ]:
        for name, count in sorted(counter.items()):
            rows.append({"metric": metric, "name": name, "count": count})
    return rows


def audit_outputs(
    row_label_rows: list[dict[str, object]],
    logical_rows: list[dict[str, object]],
    religion_rows: list[dict[str, object]],
    workbook_path: Path,
) -> list[dict[str, object]]:
    audits: list[dict[str, object]] = []
    uncaptured = [row for row in row_label_rows if row.get("potential_missed_signal") == "yes"]
    audits.append(
        {
            "check": "row_label_missed_signals_are_explicitly_listed",
            "status": "pass",
            "detail": f"potential_missed={len(uncaptured)}",
        }
    )
    overbroad = [row for row in logical_rows if row.get("overbroad_risk") == "yes"]
    audits.append(
        {
            "check": "logical_overbroad_risks_are_explicitly_listed",
            "status": "pass",
            "detail": f"overbroad={len(overbroad)}",
        }
    )
    missing_religion = [row for row in religion_rows if row.get("coverage_bucket") == "possible_missing_religion_family"]
    audits.append(
        {
            "check": "possible_missing_religion_sequences_are_explicitly_listed",
            "status": "pass",
            "detail": f"possible_missing={len(missing_religion)}",
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
    row_label_rows: list[dict[str, object]],
    logical_rows: list[dict[str, object]],
    religion_rows: list[dict[str, object]],
) -> None:
    workbook = xlsxwriter.Workbook(path, {"constant_memory": True})
    header_fmt = workbook.add_format({"bold": True, "bg_color": "#E8EEF7", "border": 1})
    wrap_fmt = workbook.add_format({"text_wrap": True, "valign": "top"})
    sheets = [
        ("Summary", summary_rows, SUMMARY_FIELDS),
        ("Audit", audit_rows, AUDIT_FIELDS),
        ("Row Label Coverage", row_label_rows, ROW_LABEL_FIELDS),
        ("Logical Family Audit", logical_rows, LOGICAL_FIELDS),
        ("Religion Sequences", religion_rows, RELIGION_SEQUENCE_FIELDS),
    ]
    for name, rows, fields in sheets:
        ws = workbook.add_worksheet(name[:31])
        for col, field in enumerate(fields):
            ws.write(0, col, field, header_fmt)
        for row_idx, row in enumerate(rows, start=1):
            for col, field in enumerate(fields):
                value = row.get(field, "")
                fmt = wrap_fmt if field in {"explanation", "recommended_action", "prev_title", "title", "primary_title", "titles", "rule_flags"} else None
                ws.write(row_idx, col, value, fmt)
        ws.freeze_panes(1, 0)
        ws.autofilter(0, 0, max(0, len(rows)), max(0, len(fields) - 1))
        for col, field in enumerate(fields):
            ws.set_column(col, col, min(max(len(field) + 2, 12), 54))
    workbook.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rulebook-dir", default=str(DEFAULT_RULEBOOK_DIR))
    parser.add_argument("--sequence-dir", default=str(DEFAULT_SEQUENCE_DIR))
    parser.add_argument("--logical-dir", default=str(DEFAULT_LOGICAL_DIR))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    args = parser.parse_args()

    rulebook_dir = Path(args.rulebook_dir).expanduser()
    sequence_dir = Path(args.sequence_dir).expanduser()
    logical_dir = Path(args.logical_dir).expanduser()
    out_dir = Path(args.out_dir).expanduser()
    paths = {"rulebook_dir": rulebook_dir, "sequence_dir": sequence_dir, "logical_dir": logical_dir, "out_dir": out_dir}

    edges = read_csv(rulebook_dir / "edge_rulebook.csv")
    groups = read_csv(rulebook_dir / "grouping_registry.csv")
    pages = read_csv(rulebook_dir / "page_signal_index.csv")
    logical_members = read_csv(logical_dir / "logical_family_members.csv")

    member_index = logical_family_page_index(logical_members)
    row_label_rows = build_row_label_audit(edges)
    logical_rows = build_logical_family_audit(groups)
    religion_rows = build_religion_sequences(pages, member_index, groups)
    summary_rows = build_summary(row_label_rows, logical_rows, religion_rows, paths)

    out_dir.mkdir(parents=True, exist_ok=True)
    workbook_path = out_dir / "assembly_quality_audit_workbook.xlsx"
    write_workbook(workbook_path, summary_rows, [], row_label_rows, logical_rows, religion_rows)
    audit_rows = audit_outputs(row_label_rows, logical_rows, religion_rows, workbook_path)
    summary_rows.extend({"metric": "audits", "name": row["check"], "count": row["status"]} for row in audit_rows)
    write_workbook(workbook_path, summary_rows, audit_rows, row_label_rows, logical_rows, religion_rows)
    audit_rows = audit_outputs(row_label_rows, logical_rows, religion_rows, workbook_path)

    write_csv(out_dir / "row_label_coverage_audit.csv", row_label_rows, ROW_LABEL_FIELDS)
    write_csv(out_dir / "logical_family_quality_audit.csv", logical_rows, LOGICAL_FIELDS)
    write_csv(out_dir / "religion_sequence_coverage_audit.csv", religion_rows, RELIGION_SEQUENCE_FIELDS)
    write_csv(out_dir / "summary.csv", summary_rows, SUMMARY_FIELDS)
    write_csv(out_dir / "audit.csv", audit_rows, AUDIT_FIELDS)

    summary = {
        "out_dir": output_path(out_dir),
        "row_label_audit_rows": len(row_label_rows),
        "row_label_potential_missed": sum(1 for row in row_label_rows if row.get("potential_missed_signal") == "yes"),
        "logical_family_audit_rows": len(logical_rows),
        "logical_overbroad_risks": sum(1 for row in logical_rows if row.get("overbroad_risk") == "yes"),
        "religion_sequence_rows": len(religion_rows),
        "possible_missing_religion_sequences": sum(1 for row in religion_rows if row.get("coverage_bucket") == "possible_missing_religion_family"),
        "row_label_buckets": dict(Counter(str(row.get("coverage_bucket", "")) for row in row_label_rows)),
        "logical_buckets": dict(Counter(str(row.get("audit_bucket", "")) for row in logical_rows)),
        "religion_sequence_buckets": dict(Counter(str(row.get("coverage_bucket", "")) for row in religion_rows)),
        "audit_failure_count": sum(1 for row in audit_rows if row.get("status") != "pass"),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
