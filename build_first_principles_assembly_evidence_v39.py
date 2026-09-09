#!/usr/bin/env python3
"""Build OCR-backed evidence for first-principles physical assembly decisions.

This layer performs no OCR/API work. It reads existing OCR CSV/JSON artifacts,
checks whether page-pair/component assembly is supported by row labels, column
group order, and zero checks, and writes reviewable evidence without promoting
any edge.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import xlsxwriter


ROOT = Path(__file__).resolve().parent
DEFAULT_RULEBOOK_DIR = ROOT / "export_final/pdf_recovery_incoming_2026-06-16_first_principles_assembly_rulebook_v34"
DEFAULT_CONTRACT_DIR = ROOT / "export_final/pdf_recovery_incoming_2026-06-16_first_principles_assembly_contract_v38"
DEFAULT_OUT_DIR = ROOT / "export_final/pdf_recovery_incoming_2026-06-16_first_principles_assembly_evidence_v39"

PHYSICAL_RELEVANT_STATUSES = {
    "confirmed_physical_edge",
    "candidate_row_label_dependency",
    "candidate_physical_missing_ocr",
    "likely_assemblable_after_missing_pages",
    "unpromoted_row_label_dependency_signal",
    "failed_or_partial_physical_review",
}

EDGE_FIELDS = [
    "evidence_rank",
    "edge_key",
    "v5_source_key",
    "source_pdf_id_guess",
    "page_pair",
    "page_gap",
    "pages_consecutive",
    "physical_rule_status",
    "physical_merge_allowed_now",
    "evidence_bucket",
    "evidence_priority",
    "recommended_next_step",
    "left_artifact_status",
    "right_artifact_status",
    "left_best_json",
    "right_best_json",
    "left_best_csv",
    "right_best_csv",
    "left_title",
    "right_title",
    "left_region",
    "right_region",
    "left_column_groups",
    "right_column_groups",
    "combined_column_group_order",
    "column_order_evidence",
    "row_alignment_evidence",
    "left_row_count",
    "right_row_count",
    "left_row_label_count",
    "right_row_label_count",
    "shared_row_label_count",
    "left_first_labels",
    "right_first_labels",
    "zero_check_status",
    "zero_check_failures",
    "zero_check_cells",
    "rule_decision_reason",
    "rule_flags",
    "prev_image_path",
    "image_path",
]

COMPONENT_FIELDS = [
    "component_rank",
    "component_id",
    "source_record_id",
    "v5_source_key",
    "source_pdf_id_guess",
    "page_range",
    "component_pages",
    "n_pages",
    "edge_pairs",
    "site_use_policy",
    "physical_merge_allowed_now",
    "safe_to_publish_values",
    "review_required",
    "component_evidence_bucket",
    "component_priority",
    "recommended_next_step",
    "edge_evidence_buckets",
    "artifact_statuses",
    "combined_column_group_order",
    "row_alignment_summary",
    "zero_check_status",
    "zero_check_failures",
    "zero_check_cells",
    "preview_row_count",
]

PREVIEW_FIELDS = [
    "preview_rank",
    "component_id",
    "v5_source_key",
    "source_pdf_id_guess",
    "component_pages",
    "source_page",
    "source_artifact",
    "section",
    "row_label",
    "column_group_order",
    "column_group",
    "persons",
    "males",
    "females",
]

CANDIDATE_FIELDS = [
    "queue_rank",
    "edge_key",
    "v5_source_key",
    "source_pdf_id_guess",
    "page_pair",
    "physical_rule_status",
    "evidence_bucket",
    "evidence_priority",
    "recommended_next_step",
    "left_artifact_status",
    "right_artifact_status",
    "column_order_evidence",
    "row_alignment_evidence",
    "zero_check_status",
    "rule_decision_reason",
]

AUDIT_FIELDS = ["check", "status", "detail"]
SUMMARY_FIELDS = ["metric", "name", "count"]


@dataclass(frozen=True)
class Artifact:
    path: Path
    kind: str
    source_pdf_id: str
    pdf_page: str
    stem: str
    mtime: float


@dataclass
class PageModel:
    artifact_json: Path | None = None
    artifact_csv: Path | None = None
    title: str = ""
    region: str = ""
    column_groups: list[str] | None = None
    rows: list[dict[str, object]] | None = None
    row_labels: list[str] | None = None
    zero_failures: int = 0
    zero_cells: int = 0
    parse_error: str = ""

    def __post_init__(self) -> None:
        if self.column_groups is None:
            self.column_groups = []
        if self.rows is None:
            self.rows = []
        if self.row_labels is None:
            self.row_labels = []


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_csv_rows(path: Path) -> list[list[str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.reader(f))


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def output_path(path: Path | None) -> str:
    if path is None:
        return ""
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


def int_or_none(value: object) -> int | None:
    if value in {None, ""}:
        return None
    try:
        return int(str(value).replace(",", ""))
    except ValueError:
        return None


def source_page_from_name(path: Path) -> tuple[str, str]:
    match = re.search(r"(\d{7,8})_p(\d+)", path.name)
    if not match:
        return "", ""
    return match.group(1), match.group(2)


def image_hash_or_stem(path_or_value: object) -> str:
    text = str(path_or_value or "")
    if not text:
        return ""
    stem = Path(text).stem
    for suffix in ["_oneshot", ".mistral"]:
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    hash_match = re.search(r"([0-9a-f]{40})", stem, flags=re.IGNORECASE)
    if hash_match:
        return hash_match.group(1).lower()
    source_pdf_id, pdf_page = source_page_from_name(Path(text))
    if source_pdf_id and pdf_page:
        return f"{source_pdf_id}_p{pdf_page}"
    return stem.lower()


def artifact_kind(path: Path) -> str:
    if path.suffix.lower() == ".json":
        return "json"
    if path.suffix.lower() == ".csv":
        return "csv"
    if "".join(path.suffixes).lower().endswith(".mistral.md"):
        return "mistral_md"
    return path.suffix.lower().lstrip(".")


def artifact_priority(path: Path) -> tuple[int, float, str]:
    text = str(path).lower()
    if "manual_visual_corrections" in text or "/corrected/" in text:
        tier = 0
    elif "v2_best" in text:
        tier = 1
    elif "money_gapfill_ocr" in text or "money_renders_ocr" in text:
        tier = 2
    elif "ocr_results_referenced" in text:
        tier = 3
    elif "gemini_no_gap_ocr_2026-06-16_v2" in text:
        tier = 4
    elif "gemini_no_gap_ocr_2026-06-16" in text:
        tier = 5
    elif "/results/" in text:
        tier = 6
    else:
        tier = 9
    return tier, -path.stat().st_mtime, output_path(path)


def artifact_roots(extra_roots: Iterable[str]) -> list[Path]:
    roots: list[Path] = [
        ROOT / "export/ocr_results_referenced",
        ROOT / "export/money_gapfill_ocr",
        ROOT / "export/money_renders_ocr",
        ROOT / "results",
    ]
    roots.extend(sorted((ROOT / "export_final").glob("gemini_no_gap_ocr_2026-06-16*")))
    roots.extend(Path(root).expanduser() for root in extra_roots)
    unique: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        if not root.exists():
            continue
        resolved = root.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(root)
    return unique


def index_artifacts(roots: list[Path]) -> tuple[dict[tuple[str, str, str], list[Artifact]], dict[tuple[str, str], list[Artifact]], dict[str, list[Artifact]]]:
    by_source_kind: dict[tuple[str, str, str], list[Artifact]] = defaultdict(list)
    by_source_page: dict[tuple[str, str], list[Artifact]] = defaultdict(list)
    by_stem_kind: dict[str, list[Artifact]] = defaultdict(list)
    for root in roots:
        for pattern in ("*.json", "*.csv"):
            for path in root.rglob(pattern):
                if path.name.startswith("_"):
                    continue
                try:
                    stat = path.stat()
                except OSError:
                    continue
                kind = artifact_kind(path)
                source_pdf_id, pdf_page = source_page_from_name(path)
                stem = image_hash_or_stem(path)
                artifact = Artifact(path, kind, source_pdf_id, pdf_page, stem, stat.st_mtime)
                if source_pdf_id and pdf_page:
                    by_source_page[(source_pdf_id, pdf_page)].append(artifact)
                    by_source_kind[(source_pdf_id, pdf_page, kind)].append(artifact)
                if stem:
                    by_stem_kind[f"{stem}|{kind}"].append(artifact)
    for bucket in list(by_source_kind.values()) + list(by_source_page.values()) + list(by_stem_kind.values()):
        bucket.sort(key=lambda item: artifact_priority(item.path))
    return by_source_kind, by_source_page, by_stem_kind


def best_artifact(
    source_pdf_id: str,
    pdf_page: str,
    image_path: str,
    kind: str,
    by_source_kind: dict[tuple[str, str, str], list[Artifact]],
    by_stem_kind: dict[str, list[Artifact]],
) -> Artifact | None:
    hits: list[Artifact] = []
    if source_pdf_id and pdf_page:
        hits.extend(by_source_kind.get((source_pdf_id, pdf_page, kind), []))
    stem = image_hash_or_stem(image_path)
    if stem:
        hits.extend(by_stem_kind.get(f"{stem}|{kind}", []))
    unique = {artifact.path.resolve(): artifact for artifact in hits}
    if not unique:
        return None
    return sorted(unique.values(), key=lambda item: artifact_priority(item.path))[0]


def normalize_label(value: object) -> str:
    text = str(value or "").strip().lower()
    text = text.replace("\u2013", "-").replace("\u2014", "-").replace("\u2212", "-")
    text = re.sub(r"\s+", " ", text)
    return text.strip(" .:-")


def normalize_column_group(value: object) -> str:
    text = normalize_label(value)
    text = re.sub(r"[^a-z0-9&+-]+", "", text)
    return text


def display_label(value: object) -> str:
    return str(value or "").replace("\u2013", "-").replace("\u2014", "-").strip()


def possible_row_label_key(row: dict[str, object]) -> str:
    for key in ["age", "Age.", "Age", "row_label", "district", "District", "area", "Area", "name"]:
        value = row.get(key)
        if value not in {None, ""} and not isinstance(value, dict):
            return display_label(value)
    for key, value in row.items():
        if isinstance(value, dict):
            continue
        lower = str(key).lower()
        if any(token in lower for token in ["age", "district", "row", "area"]):
            return display_label(value)
    return ""


def numeric_triplet_failures(rows: list[dict[str, object]], groups: list[str]) -> tuple[int, int]:
    failures = 0
    cells = 0
    for row in rows:
        for group in groups:
            value = row.get(group)
            if not isinstance(value, dict):
                continue
            persons = int_or_none(value.get("persons"))
            males = int_or_none(value.get("males"))
            females = int_or_none(value.get("females"))
            if persons is None or males is None or females is None:
                continue
            cells += 1
            if persons != males + females:
                failures += 1
    return failures, cells


def parse_json_model(path: Path | None) -> PageModel:
    model = PageModel(artifact_json=path)
    if path is None:
        return model
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        model.parse_error = f"{type(exc).__name__}: {exc}"
        return model
    if not isinstance(raw, dict):
        model.parse_error = "json_root_not_object"
        return model
    meta = raw.get("metadata") if isinstance(raw.get("metadata"), dict) else {}
    model.title = str(meta.get("title") or raw.get("title") or "")
    model.region = str(meta.get("region") or raw.get("region") or "")
    groups = [display_label(value) for value in meta.get("column_groups", []) if display_label(value)] if isinstance(meta.get("column_groups"), list) else []
    rows_out: list[dict[str, object]] = []
    labels: list[str] = []
    data = raw.get("data", [])
    if isinstance(data, list):
        for section in data:
            if not isinstance(section, dict):
                continue
            section_name = display_label(section.get("name") or section.get("section") or "")
            section_rows = section.get("rows", [])
            if not isinstance(section_rows, list):
                continue
            for source_row in section_rows:
                if not isinstance(source_row, dict):
                    continue
                row = dict(source_row)
                row["_section"] = section_name
                label = possible_row_label_key(row)
                if label:
                    labels.append(label)
                for key, value in row.items():
                    if isinstance(value, dict) and display_label(key) not in groups:
                        groups.append(display_label(key))
                rows_out.append(row)
    model.column_groups = groups
    model.rows = rows_out
    model.row_labels = labels
    model.zero_failures, model.zero_cells = numeric_triplet_failures(rows_out, groups)
    return model


def parse_csv_model(path: Path | None) -> PageModel:
    model = PageModel(artifact_csv=path)
    if path is None:
        return model
    try:
        rows = read_csv_rows(path)
    except Exception as exc:
        model.parse_error = f"{type(exc).__name__}: {exc}"
        return model
    if not rows:
        return model
    header = [str(value or "").strip() for value in rows[0]]
    dict_rows: list[dict[str, object]] = []
    labels: list[str] = []
    groups: list[str] = []
    for raw_row in rows[1:]:
        row = {header[i]: raw_row[i] if i < len(raw_row) else "" for i in range(len(header))}
        group = display_label(row.get("group", ""))
        if group:
            groups.append(group)
        label = display_label(row.get("row_label") or row.get("age") or row.get("district") or "")
        if label:
            labels.append(label)
        dict_rows.append(row)
    if not groups:
        for field in header:
            if field and field.lower() not in {"section", "row_label", "row_label_type", "age", "district"}:
                groups.append(field)
    model.column_groups = list(dict.fromkeys(groups))
    model.rows = dict_rows
    model.row_labels = labels
    # CSV zero checks only apply to normalized group/person/males/females rows.
    failures = cells = 0
    if {"persons", "males", "females"}.issubset(set(header)):
        for row in dict_rows:
            persons = int_or_none(row.get("persons"))
            males = int_or_none(row.get("males"))
            females = int_or_none(row.get("females"))
            if persons is None or males is None or females is None:
                continue
            cells += 1
            if persons != males + females:
                failures += 1
    model.zero_failures = failures
    model.zero_cells = cells
    return model


def merged_page_model(json_artifact: Artifact | None, csv_artifact: Artifact | None) -> PageModel:
    json_model = parse_json_model(json_artifact.path if json_artifact else None)
    csv_model = parse_csv_model(csv_artifact.path if csv_artifact else None)
    if json_model.rows:
        json_model.artifact_csv = csv_artifact.path if csv_artifact else None
        if not json_model.row_labels and csv_model.row_labels:
            json_model.row_labels = csv_model.row_labels
        if not json_model.column_groups and csv_model.column_groups:
            json_model.column_groups = csv_model.column_groups
        if json_model.zero_cells == 0 and csv_model.zero_cells:
            json_model.zero_failures = csv_model.zero_failures
            json_model.zero_cells = csv_model.zero_cells
        return json_model
    csv_model.artifact_json = json_artifact.path if json_artifact else None
    if json_model.parse_error and not csv_model.parse_error:
        csv_model.parse_error = json_model.parse_error
    return csv_model


def first_labels(labels: list[str], limit: int = 8) -> str:
    return semi(labels[:limit])


def label_alignment(left: PageModel, right: PageModel, rule_right_no_labels: bool) -> tuple[str, int]:
    left_norm = [normalize_label(x) for x in left.row_labels if normalize_label(x)]
    right_norm = [normalize_label(x) for x in right.row_labels if normalize_label(x)]
    shared = len(set(left_norm) & set(right_norm))
    if not left.rows or not right.rows:
        return "artifact_missing_or_unparsed", shared
    if left_norm and right_norm:
        if left_norm == right_norm:
            return "same_row_labels_exact", shared
        if shared >= min(len(left_norm), len(right_norm)) * 0.65:
            return "same_row_labels_substantial_overlap", shared
        return "both_pages_labelled_different_rows_review", shared
    if left_norm and not right_norm:
        if len(left.rows) == len(right.rows):
            return "right_unlabelled_same_row_count_inherit_left_labels", shared
        return "right_unlabelled_row_count_mismatch_review", shared
    if not left_norm and right_norm:
        return "left_unlabelled_right_labelled_no_inheritance_needed" if rule_right_no_labels else "left_unlabelled_right_labelled_review", shared
    if rule_right_no_labels and len(left.rows) == len(right.rows) and left.rows:
        return "both_unlabelled_but_equal_row_count_rule_review", shared
    return "no_row_label_evidence", shared


def column_order(left: PageModel, right: PageModel) -> tuple[str, list[str]]:
    left_groups = [g for g in left.column_groups if g]
    right_groups = [g for g in right.column_groups if g]
    left_norm = [normalize_column_group(g) for g in left_groups]
    right_norm = [normalize_column_group(g) for g in right_groups]
    left_norm_set = {g for g in left_norm if g}
    right_norm_set = {g for g in right_norm if g}
    combined = list(left_groups)
    seen_norm = set(left_norm_set)
    for group, norm in zip(right_groups, right_norm):
        if norm and norm in seen_norm:
            continue
        combined.append(group)
        if norm:
            seen_norm.add(norm)
    if not left_groups or not right_groups:
        return "missing_column_group_evidence", combined
    if left_norm_set and left_norm_set == right_norm_set:
        return "same_column_groups_vertical_or_duplicate_review", combined
    if left_norm_set.isdisjoint(right_norm_set):
        return "left_page_groups_then_right_page_groups_disjoint", combined
    return "overlapping_column_groups_review", combined


def zero_status(left: PageModel, right: PageModel) -> tuple[str, int, int]:
    failures = left.zero_failures + right.zero_failures
    cells = left.zero_cells + right.zero_cells
    if cells == 0:
        return "no_zero_check_available", failures, cells
    if failures == 0:
        return "zero_checks_pass", failures, cells
    return "zero_check_failures_review", failures, cells


def evidence_bucket(edge: dict[str, str], left: PageModel, right: PageModel, row_evidence: str, col_evidence: str, z_status: str) -> tuple[str, int, str]:
    status = edge.get("physical_rule_status", "")
    both_artifacts = bool(left.artifact_json or left.artifact_csv) and bool(right.artifact_json or right.artifact_csv)
    consecutive = edge.get("pages_consecutive") == "yes"
    strong_row = row_evidence in {
        "same_row_labels_exact",
        "same_row_labels_substantial_overlap",
        "right_unlabelled_same_row_count_inherit_left_labels",
        "left_unlabelled_right_labelled_no_inheritance_needed",
    }
    strong_col = col_evidence == "left_page_groups_then_right_page_groups_disjoint"
    zero_ok = z_status in {"zero_checks_pass", "no_zero_check_available"}
    if status == "confirmed_physical_edge" and both_artifacts and consecutive and strong_row and strong_col and zero_ok:
        return "confirmed_edge_with_strong_ocr_evidence", 1, "Keep as physical structure; values publish only if value-review policy allows."
    if status == "confirmed_physical_edge" and both_artifacts:
        return "confirmed_edge_needs_value_or_visual_review", 2, "Keep structural grouping but review column/row/value evidence before publishing values."
    if not both_artifacts:
        return "needs_missing_ocr_artifact", 3, "Wait for or attach OCR artifacts for both pages, then rerun this evidence layer."
    if status in {"candidate_row_label_dependency", "unpromoted_row_label_dependency_signal", "candidate_physical_missing_ocr"} and consecutive and strong_row and strong_col and zero_ok:
        return "candidate_has_strong_evidence_review_for_promotion", 4, "Manual review candidate; promote only after image/table context confirms no switch."
    if status == "likely_assemblable_after_missing_pages":
        return "missing_page_bridge_do_not_merge", 5, "Do not merge now; OCR missing intermediate pages and retest as adjacent edges."
    return "candidate_or_failed_edge_review", 6, "Keep out of site-ready physical groupings until constraints are repaired or reviewed."


def artifact_status(model: PageModel) -> str:
    if model.artifact_json and model.artifact_csv:
        return "json_and_csv"
    if model.artifact_json:
        return "json_only"
    if model.artifact_csv:
        return "csv_only"
    return "missing"


def build_edge_evidence(
    edges: list[dict[str, str]],
    by_source_kind: dict[tuple[str, str, str], list[Artifact]],
    by_stem_kind: dict[str, list[Artifact]],
) -> tuple[list[dict[str, object]], dict[str, tuple[PageModel, PageModel]]]:
    rows: list[dict[str, object]] = []
    model_cache: dict[tuple[str, str, str], PageModel] = {}
    edge_models: dict[str, tuple[PageModel, PageModel]] = {}

    def get_model(source_id: str, page: str, image_path: str) -> PageModel:
        key = (source_id, page, image_path)
        if key in model_cache:
            return model_cache[key]
        json_artifact = best_artifact(source_id, page, image_path, "json", by_source_kind, by_stem_kind)
        csv_artifact = best_artifact(source_id, page, image_path, "csv", by_source_kind, by_stem_kind)
        model = merged_page_model(json_artifact, csv_artifact)
        model_cache[key] = model
        return model

    relevant = [edge for edge in edges if edge.get("physical_rule_status") in PHYSICAL_RELEVANT_STATUSES]
    for idx, edge in enumerate(relevant, start=1):
        source_id = edge.get("source_pdf_id_guess", "")
        left = get_model(source_id, edge.get("prev_pdf_page", ""), edge.get("prev_image_path", ""))
        right = get_model(source_id, edge.get("pdf_page", ""), edge.get("image_path", ""))
        row_evidence, shared_labels = label_alignment(left, right, edge.get("right_page_no_row_labels") == "yes")
        col_evidence, combined_groups = column_order(left, right)
        z_status, z_failures, z_cells = zero_status(left, right)
        bucket, priority, next_step = evidence_bucket(edge, left, right, row_evidence, col_evidence, z_status)
        edge_models[edge.get("edge_key", "")] = (left, right)
        rows.append(
            {
                "evidence_rank": idx,
                "edge_key": edge.get("edge_key", ""),
                "v5_source_key": edge.get("v5_source_key", ""),
                "source_pdf_id_guess": source_id,
                "page_pair": edge.get("page_pair", ""),
                "page_gap": edge.get("page_gap", ""),
                "pages_consecutive": edge.get("pages_consecutive", ""),
                "physical_rule_status": edge.get("physical_rule_status", ""),
                "physical_merge_allowed_now": edge.get("physical_merge_allowed_now", ""),
                "evidence_bucket": bucket,
                "evidence_priority": priority,
                "recommended_next_step": next_step,
                "left_artifact_status": artifact_status(left),
                "right_artifact_status": artifact_status(right),
                "left_best_json": output_path(left.artifact_json),
                "right_best_json": output_path(right.artifact_json),
                "left_best_csv": output_path(left.artifact_csv),
                "right_best_csv": output_path(right.artifact_csv),
                "left_title": left.title,
                "right_title": right.title,
                "left_region": left.region,
                "right_region": right.region,
                "left_column_groups": semi(left.column_groups),
                "right_column_groups": semi(right.column_groups),
                "combined_column_group_order": semi(combined_groups),
                "column_order_evidence": col_evidence,
                "row_alignment_evidence": row_evidence,
                "left_row_count": len(left.rows),
                "right_row_count": len(right.rows),
                "left_row_label_count": len(left.row_labels),
                "right_row_label_count": len(right.row_labels),
                "shared_row_label_count": shared_labels,
                "left_first_labels": first_labels(left.row_labels),
                "right_first_labels": first_labels(right.row_labels),
                "zero_check_status": z_status,
                "zero_check_failures": z_failures,
                "zero_check_cells": z_cells,
                "rule_decision_reason": edge.get("decision_reason", ""),
                "rule_flags": edge.get("rule_flags", ""),
                "prev_image_path": edge.get("prev_image_path", ""),
                "image_path": edge.get("image_path", ""),
            }
        )
    return rows, edge_models


def edge_key_for(source_key: str, source_id: str, left_page: str, right_page: str) -> str:
    return f"{source_key}||{left_page}||{right_page}"


def build_component_evidence(
    components: list[dict[str, str]],
    edge_evidence: list[dict[str, object]],
    edge_models: dict[str, tuple[PageModel, PageModel]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    edge_by_key = {str(row.get("edge_key", "")): row for row in edge_evidence}
    component_rows: list[dict[str, object]] = []
    preview_rows: list[dict[str, object]] = []
    preview_rank = 0
    for rank, component in enumerate(components, start=1):
        pages = [part.strip() for part in str(component.get("component_pages", "")).split(";") if part.strip()]
        edge_rows: list[dict[str, object]] = []
        for left_page, right_page in zip(pages, pages[1:]):
            key = edge_key_for(component.get("v5_source_key", ""), component.get("source_pdf_id_guess", ""), left_page, right_page)
            if key in edge_by_key:
                edge_rows.append(edge_by_key[key])
        buckets = [str(row.get("evidence_bucket", "")) for row in edge_rows]
        artifact_statuses = [f"{row.get('page_pair')}:{row.get('left_artifact_status')}/{row.get('right_artifact_status')}" for row in edge_rows]
        zero_failures = sum(int_or_none(row.get("zero_check_failures")) or 0 for row in edge_rows)
        zero_cells = sum(int_or_none(row.get("zero_check_cells")) or 0 for row in edge_rows)
        zero_status = "zero_checks_pass" if zero_cells and zero_failures == 0 else ("zero_check_failures_review" if zero_failures else "no_zero_check_available")
        if edge_rows and all(row.get("evidence_bucket") == "confirmed_edge_with_strong_ocr_evidence" for row in edge_rows):
            bucket = "component_has_strong_ocr_evidence"
            priority = 1
            next_step = "Use as physical structure; value publication still follows value-review policy."
        elif edge_rows and any("missing" in str(row.get("left_artifact_status")) + str(row.get("right_artifact_status")) for row in edge_rows):
            bucket = "component_missing_ocr_evidence"
            priority = 3
            next_step = "Keep structural grouping but fill missing OCR evidence before publishing values."
        else:
            bucket = "component_needs_value_or_visual_review"
            priority = 2
            next_step = "Review row/column/value evidence before publishing values."
        combined_groups = semi(row.get("combined_column_group_order", "") for row in edge_rows)
        row_summary = semi(row.get("row_alignment_evidence", "") for row in edge_rows)
        component_id = component.get("source_record_id", "").replace("physical::", "") or component.get("recommendation_id", "")
        for edge_row in edge_rows:
            models = edge_models.get(str(edge_row.get("edge_key", "")))
            if not models:
                continue
            left, right = models
            source_pages = [str(edge_row.get("page_pair", "")).split("-")[0], str(edge_row.get("page_pair", "")).split("-")[-1]]
            for page, model in zip(source_pages, [left, right]):
                for row in model.rows:
                    section = display_label(row.get("_section", ""))
                    label = possible_row_label_key(row)
                    for order, group in enumerate(model.column_groups, start=1):
                        value = row.get(group)
                        if not isinstance(value, dict):
                            continue
                        preview_rank += 1
                        preview_rows.append(
                            {
                                "preview_rank": preview_rank,
                                "component_id": component_id,
                                "v5_source_key": component.get("v5_source_key", ""),
                                "source_pdf_id_guess": component.get("source_pdf_id_guess", ""),
                                "component_pages": component.get("component_pages", ""),
                                "source_page": page,
                                "source_artifact": output_path(model.artifact_json or model.artifact_csv),
                                "section": section,
                                "row_label": label,
                                "column_group_order": order,
                                "column_group": group,
                                "persons": value.get("persons", ""),
                                "males": value.get("males", ""),
                                "females": value.get("females", ""),
                            }
                        )
        component_rows.append(
            {
                "component_rank": rank,
                "component_id": component_id,
                "source_record_id": component.get("source_record_id", ""),
                "v5_source_key": component.get("v5_source_key", ""),
                "source_pdf_id_guess": component.get("source_pdf_id_guess", ""),
                "page_range": component.get("page_range", ""),
                "component_pages": component.get("component_pages", ""),
                "n_pages": component.get("n_pages", ""),
                "edge_pairs": component.get("edge_pairs", ""),
                "site_use_policy": component.get("site_use_policy", ""),
                "physical_merge_allowed_now": component.get("physical_merge_allowed_now", ""),
                "safe_to_publish_values": component.get("safe_to_publish_values", ""),
                "review_required": component.get("review_required", ""),
                "component_evidence_bucket": bucket,
                "component_priority": priority,
                "recommended_next_step": next_step,
                "edge_evidence_buckets": semi(buckets),
                "artifact_statuses": semi(artifact_statuses),
                "combined_column_group_order": combined_groups,
                "row_alignment_summary": row_summary,
                "zero_check_status": zero_status,
                "zero_check_failures": zero_failures,
                "zero_check_cells": zero_cells,
                "preview_row_count": sum(1 for row in preview_rows if row.get("component_id") == component_id),
            }
        )
    return component_rows, preview_rows


def build_candidate_queue(edge_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    candidates = [
        row for row in edge_rows
        if row.get("physical_rule_status") != "confirmed_physical_edge"
        or row.get("evidence_bucket") in {"confirmed_edge_needs_value_or_visual_review", "needs_missing_ocr_artifact"}
    ]
    candidates.sort(key=lambda row: (int_or_none(row.get("evidence_priority")) or 99, str(row.get("v5_source_key")), str(row.get("page_pair"))))
    out: list[dict[str, object]] = []
    for idx, row in enumerate(candidates, start=1):
        out.append({"queue_rank": idx, **row})
    return out


def ranked_rows(rows: list[dict[str, object]], rank_field: str) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for idx, row in enumerate(rows, start=1):
        out.append({rank_field: idx, **row})
    return out


def build_audits(
    edge_rows: list[dict[str, object]],
    component_rows: list[dict[str, object]],
    preview_rows: list[dict[str, object]],
    workbook_path: Path,
) -> list[dict[str, object]]:
    audits: list[dict[str, object]] = []
    unsafe = [
        row for row in edge_rows
        if row.get("physical_merge_allowed_now") == "yes" and row.get("pages_consecutive") != "yes"
    ]
    audits.append({"check": "physical_merge_edges_are_consecutive", "status": "pass" if not unsafe else "fail", "detail": f"bad={len(unsafe)}"})
    promoted_candidates = [
        row for row in edge_rows
        if row.get("physical_rule_status") != "confirmed_physical_edge" and row.get("physical_merge_allowed_now") == "yes"
    ]
    audits.append({"check": "candidate_edges_not_promoted_by_evidence_layer", "status": "pass" if not promoted_candidates else "fail", "detail": f"bad={len(promoted_candidates)}"})
    confirmed_without_evidence = [
        row for row in edge_rows
        if row.get("physical_rule_status") == "confirmed_physical_edge" and row.get("evidence_bucket") == "needs_missing_ocr_artifact"
    ]
    audits.append({"check": "confirmed_edges_have_some_ocr_evidence", "status": "pass" if not confirmed_without_evidence else "fail", "detail": f"missing={len(confirmed_without_evidence)}"})
    component_missing_edges = [
        row for row in component_rows
        if not row.get("edge_evidence_buckets")
    ]
    audits.append({"check": "site_ready_components_have_edge_evidence", "status": "pass" if not component_missing_edges else "fail", "detail": f"missing={len(component_missing_edges)}"})
    audits.append({"check": "assembled_preview_exists_for_components", "status": "pass" if preview_rows else "fail", "detail": f"rows={len(preview_rows)}"})
    try:
        with zipfile.ZipFile(workbook_path) as zf:
            bad_member = zf.testzip()
    except Exception as exc:
        bad_member = f"{type(exc).__name__}: {exc}"
    audits.append({"check": "workbook_zip_integrity", "status": "pass" if bad_member is None else "fail", "detail": str(bad_member or "ok")})
    return audits


def build_summary(
    edge_rows: list[dict[str, object]],
    component_rows: list[dict[str, object]],
    preview_rows: list[dict[str, object]],
    candidate_rows: list[dict[str, object]],
    strong_candidate_rows: list[dict[str, object]],
    missing_ocr_rows: list[dict[str, object]],
    component_review_rows: list[dict[str, object]],
    audits: list[dict[str, object]],
    paths: dict[str, Path],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = [{"metric": "paths", "name": name, "count": output_path(path)} for name, path in paths.items()]
    rows.extend(
        [
            {"metric": "counts", "name": "edge_evidence_rows", "count": len(edge_rows)},
            {"metric": "counts", "name": "component_evidence_rows", "count": len(component_rows)},
            {"metric": "counts", "name": "assembled_preview_rows", "count": len(preview_rows)},
            {"metric": "counts", "name": "candidate_review_queue_rows", "count": len(candidate_rows)},
            {"metric": "counts", "name": "strong_candidate_promotion_queue_rows", "count": len(strong_candidate_rows)},
            {"metric": "counts", "name": "missing_ocr_evidence_queue_rows", "count": len(missing_ocr_rows)},
            {"metric": "counts", "name": "component_review_queue_rows", "count": len(component_review_rows)},
            {"metric": "counts", "name": "audit_failures", "count": sum(1 for row in audits if row.get("status") != "pass")},
        ]
    )
    for metric, counter in [
        ("edge_by_rule_status", Counter(str(row.get("physical_rule_status", "")) for row in edge_rows)),
        ("edge_by_evidence_bucket", Counter(str(row.get("evidence_bucket", "")) for row in edge_rows)),
        ("component_by_evidence_bucket", Counter(str(row.get("component_evidence_bucket", "")) for row in component_rows)),
        ("audit_by_status", Counter(str(row.get("status", "")) for row in audits)),
    ]:
        for name, count in sorted(counter.items()):
            rows.append({"metric": metric, "name": name, "count": count})
    return rows


def write_workbook(
    path: Path,
    summary_rows: list[dict[str, object]],
    audits: list[dict[str, object]],
    edge_rows: list[dict[str, object]],
    component_rows: list[dict[str, object]],
    preview_rows: list[dict[str, object]],
    candidate_rows: list[dict[str, object]],
    strong_candidate_rows: list[dict[str, object]],
    missing_ocr_rows: list[dict[str, object]],
    component_review_rows: list[dict[str, object]],
) -> None:
    workbook = xlsxwriter.Workbook(path, {"constant_memory": True})
    header_fmt = workbook.add_format({"bold": True, "bg_color": "#E8EEF7", "border": 1})
    wrap_fmt = workbook.add_format({"text_wrap": True, "valign": "top"})
    sheets = [
        ("Summary", summary_rows, SUMMARY_FIELDS),
        ("Audit", audits, AUDIT_FIELDS),
        ("Edge Evidence", edge_rows, EDGE_FIELDS),
        ("Component Evidence", component_rows, COMPONENT_FIELDS),
        ("Assembled Preview", preview_rows, PREVIEW_FIELDS),
        ("Candidate Queue", candidate_rows, CANDIDATE_FIELDS),
        ("Strong Candidates", strong_candidate_rows, EDGE_FIELDS),
        ("Missing OCR Evidence", missing_ocr_rows, EDGE_FIELDS),
        ("Component Review", component_review_rows, COMPONENT_FIELDS),
    ]
    for name, rows, fields in sheets:
        ws = workbook.add_worksheet(name[:31])
        for col, field in enumerate(fields):
            ws.write(0, col, field, header_fmt)
        for row_idx, row in enumerate(rows, start=1):
            for col, field in enumerate(fields):
                value = row.get(field, "")
                fmt = wrap_fmt if field in {"recommended_next_step", "rule_decision_reason", "combined_column_group_order", "prev_image_path", "image_path"} else None
                ws.write(row_idx, col, value, fmt)
        ws.freeze_panes(1, 0)
        ws.autofilter(0, 0, max(0, len(rows)), max(0, len(fields) - 1))
        for col, field in enumerate(fields):
            ws.set_column(col, col, min(max(len(field) + 2, 12), 60))
    workbook.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rulebook-dir", default=str(DEFAULT_RULEBOOK_DIR))
    parser.add_argument("--contract-dir", default=str(DEFAULT_CONTRACT_DIR))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--extra-root", action="append", default=[])
    args = parser.parse_args()

    rulebook_dir = Path(args.rulebook_dir).expanduser()
    contract_dir = Path(args.contract_dir).expanduser()
    out_dir = Path(args.out_dir).expanduser()
    paths = {"rulebook_dir": rulebook_dir, "contract_dir": contract_dir, "out_dir": out_dir}

    roots = artifact_roots(args.extra_root)
    by_source_kind, _, by_stem_kind = index_artifacts(roots)
    edges = read_csv(rulebook_dir / "edge_rulebook.csv")
    components = read_csv(contract_dir / "canonical_site_ready_physical_components.csv")
    edge_rows, edge_models = build_edge_evidence(edges, by_source_kind, by_stem_kind)
    component_rows, preview_rows = build_component_evidence(components, edge_rows, edge_models)
    candidate_rows = build_candidate_queue(edge_rows)
    strong_candidate_rows = [
        row for row in edge_rows
        if row.get("evidence_bucket") == "candidate_has_strong_evidence_review_for_promotion"
    ]
    strong_candidate_rows.sort(key=lambda row: (str(row.get("v5_source_key")), str(row.get("page_pair"))))
    missing_ocr_rows = [
        row for row in edge_rows
        if row.get("evidence_bucket") == "needs_missing_ocr_artifact"
    ]
    missing_ocr_rows.sort(key=lambda row: (str(row.get("physical_rule_status")), str(row.get("v5_source_key")), str(row.get("page_pair"))))
    component_review_rows = [
        row for row in component_rows
        if row.get("component_evidence_bucket") != "component_has_strong_ocr_evidence"
        or row.get("review_required") == "yes"
        or row.get("safe_to_publish_values") != "yes"
    ]

    out_dir.mkdir(parents=True, exist_ok=True)
    workbook_path = out_dir / "assembly_evidence_workbook.xlsx"
    write_workbook(workbook_path, [], [], edge_rows, component_rows, preview_rows, candidate_rows, strong_candidate_rows, missing_ocr_rows, component_review_rows)
    audits = build_audits(edge_rows, component_rows, preview_rows, workbook_path)
    summary_rows = build_summary(edge_rows, component_rows, preview_rows, candidate_rows, strong_candidate_rows, missing_ocr_rows, component_review_rows, audits, paths)
    write_workbook(workbook_path, summary_rows, audits, edge_rows, component_rows, preview_rows, candidate_rows, strong_candidate_rows, missing_ocr_rows, component_review_rows)
    audits = build_audits(edge_rows, component_rows, preview_rows, workbook_path)
    summary_rows = build_summary(edge_rows, component_rows, preview_rows, candidate_rows, strong_candidate_rows, missing_ocr_rows, component_review_rows, audits, paths)
    write_workbook(workbook_path, summary_rows, audits, edge_rows, component_rows, preview_rows, candidate_rows, strong_candidate_rows, missing_ocr_rows, component_review_rows)

    write_csv(out_dir / "edge_assembly_evidence.csv", edge_rows, EDGE_FIELDS)
    write_csv(out_dir / "physical_component_evidence.csv", component_rows, COMPONENT_FIELDS)
    write_csv(out_dir / "assembled_physical_preview_long.csv", preview_rows, PREVIEW_FIELDS)
    write_csv(out_dir / "candidate_edge_review_queue.csv", candidate_rows, CANDIDATE_FIELDS)
    write_csv(out_dir / "strong_candidate_promotion_queue.csv", ranked_rows(strong_candidate_rows, "promotion_rank"), ["promotion_rank", *EDGE_FIELDS])
    write_csv(out_dir / "missing_ocr_evidence_queue.csv", ranked_rows(missing_ocr_rows, "missing_rank"), ["missing_rank", *EDGE_FIELDS])
    write_csv(out_dir / "component_review_queue.csv", ranked_rows(component_review_rows, "review_rank"), ["review_rank", *COMPONENT_FIELDS])
    write_csv(out_dir / "audit.csv", audits, AUDIT_FIELDS)
    write_csv(out_dir / "summary.csv", summary_rows, SUMMARY_FIELDS)
    summary = {
        "edge_evidence_rows": len(edge_rows),
        "component_evidence_rows": len(component_rows),
        "assembled_preview_rows": len(preview_rows),
        "candidate_review_queue_rows": len(candidate_rows),
        "strong_candidate_promotion_queue_rows": len(strong_candidate_rows),
        "missing_ocr_evidence_queue_rows": len(missing_ocr_rows),
        "component_review_queue_rows": len(component_review_rows),
        "audit_failure_count": sum(1 for row in audits if row.get("status") != "pass"),
        "edge_by_evidence_bucket": dict(Counter(str(row.get("evidence_bucket", "")) for row in edge_rows)),
        "component_by_evidence_bucket": dict(Counter(str(row.get("component_evidence_bucket", "")) for row in component_rows)),
        "artifact_roots_scanned": [output_path(root) for root in roots],
        "created_unix": int(time.time()),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
