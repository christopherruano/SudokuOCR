#!/usr/bin/env python3
"""Build structural, adjacency-first OCR page assemblies.

This is intentionally stricter than v4:
  * one physical page per source/pdf_page is selected before run-building;
  * physical runs only join adjacent pages with structural continuation evidence;
  * same title/source/family can create collection candidates, not physical merges;
  * gaps are recorded as missing-page candidates and never crossed by a run.

No OCR/API work is performed here. Existing OCR JSON artifacts are read only to
extract lightweight row-label/schema signals.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterable

import xlsxwriter


ROOT = Path(__file__).resolve().parent
V4_DIR = ROOT / "export_final/pdf_recovery_incoming_2026-06-16_all_ocr_component_assembly_v4_adjacency"
OUT_DIR = ROOT / "export_final/pdf_recovery_incoming_2026-06-16_all_ocr_component_assembly_v5_structural"

V4_PAGES = V4_DIR / "all_ocr_component_pages.csv"


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


def stable_id(prefix: str, *parts: object) -> str:
    data = "||".join(str(part) for part in parts)
    return f"{prefix}_{hashlib.sha1(data.encode('utf-8')).hexdigest()[:16]}"


def int_or_none(value: object) -> int | None:
    s = str(value or "").strip()
    return int(s) if re.fullmatch(r"-?\d+", s) else None


def float_fmt(value: float | None) -> str:
    if value is None or math.isnan(value):
        return ""
    return f"{value:.3f}"


def uniq(values: Iterable[object], limit: int = 12) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        s = str(value or "").strip()
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    if len(out) > limit:
        return out[:limit] + [f"...(+{len(out) - limit})"]
    return out


def uniq_join(values: Iterable[object], limit: int = 12) -> str:
    return "; ".join(uniq(values, limit=limit))


def counter_json(values: Iterable[object]) -> str:
    cleaned = [str(v or "").strip() for v in values if str(v or "").strip()]
    return json.dumps(dict(Counter(cleaned)), sort_keys=True)


def clean_text(value: object) -> str:
    s = str(value or "").lower()
    s = s.replace("&", " and ")
    s = re.sub(r"[\u2010-\u2015]", "-", s)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def compact_text(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def token_set(value: object) -> set[str]:
    return {t for t in clean_text(value).split() if len(t) > 1}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def text_similarity(a: object, b: object) -> float:
    ta = strip_continuation_words(a)
    tb = strip_continuation_words(b)
    if not ta or not tb:
        return 0.0
    return SequenceMatcher(None, ta, tb).ratio()


def strip_continuation_words(value: object) -> str:
    s = clean_text(value)
    s = re.sub(r"\b(contd|continued|continuation|concluded|concld|concl|part|pt)\b", " ", s)
    s = re.sub(r"\b[abcdeh]\b", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def title_blob(row: dict[str, object]) -> str:
    values = [
        row.get("classifier_printed_title"),
        row.get("core_ocr_title"),
        row.get("title"),
        row.get("family"),
        row.get("table_number"),
    ]
    return " ".join(str(v or "") for v in values if str(v or "").strip())


def has_continuation_signal(row: dict[str, object]) -> bool:
    return bool(re.search(r"\b(contd|continued|continuation|concluded|concld|concl)\b", clean_text(title_blob(row))))


RELIGION_TERMS = {
    "all religions",
    "religions",
    "religion",
    "hindu",
    "hindus",
    "brahmanic",
    "brahmo",
    "arya",
    "musalman",
    "musalmans",
    "muhammadan",
    "muhammadans",
    "muslim",
    "muslims",
    "christian",
    "christians",
    "native christian",
    "jew",
    "jews",
    "animistic",
    "animists",
    "sikh",
    "sikhs",
    "jain",
    "jains",
    "buddhist",
    "buddhists",
    "parsi",
    "parsis",
    "zoroastrian",
    "zoroastrians",
    "other religions",
}


def collection_base_text(value: object) -> str:
    s = strip_continuation_words(value)
    s = re.sub(r"\b(table|imperial table|state table|provincial table)\b\s*(viii|vii|vi+a?|iii|[0-9]+)\b", " table ", s)
    for term in sorted(RELIGION_TERMS, key=len, reverse=True):
        s = re.sub(rf"\b{re.escape(term)}\b", " ", s)
    s = re.sub(r"\b(part|pt)\s*[a-z0-9]+\b", " ", s)
    s = re.sub(r"\b(unmarried|married|widowed|single|population|persons|males|females)\b", " ", s)
    s = re.sub(r"\b(concld|concluded|continued|contd)\b", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def religion_variant_similarity(prev: dict[str, object], cur: dict[str, object]) -> float:
    return text_similarity(collection_base_text(prev.get("title_norm") or prev.get("title")), collection_base_text(cur.get("title_norm") or cur.get("title")))


def source_key(row: dict[str, object]) -> str:
    for field in ["source_pdf", "source_pdf_id", "adj_source_key", "source_group_key"]:
        value = str(row.get(field) or "").strip()
        if value:
            return value.split("||", 1)[0]
    year = str(row.get("year") or "").strip()
    province = str(row.get("province") or "").strip()
    return f"{year}::{province}" if year or province else "unknown_source"


def table_family(row: dict[str, object]) -> str:
    return str(row.get("table_family_key") or row.get("family") or row.get("page_kind") or "").strip()


def table_number_key(row: dict[str, object]) -> str:
    explicit = clean_text(row.get("table_number"))
    blob = clean_text(title_blob(row))
    def normalize(value: str) -> str:
        value = re.sub(r"^\b(table|form|statement|c)\b\s*", "", value).strip()
        value = re.sub(r"\s+", " ", value)
        roman = {
            "i": "1",
            "ii": "2",
            "iii": "3",
            "iv": "4",
            "v": "5",
            "vi": "6",
            "via": "6a",
            "vii": "7",
            "viii": "8",
            "ix": "9",
            "x": "10",
        }
        return roman.get(value, value)
    if explicit:
        return normalize(explicit)
    m = re.search(r"\b(table|form|statement|c)\s*(viii|vii|vi+a?|iii|[0-9]+)\b", blob)
    if m:
        return normalize(m.group(2))
    return ""


def page_kind(row: dict[str, object]) -> str:
    return str(row.get("page_kind") or "").strip() or "unknown"


def is_numeric_like(row: dict[str, object]) -> bool:
    return page_kind(row) in {"main_age_civil_numeric", "derivative_selected_caste_race_tribe", "structured_unknown", "other_table"}


def is_context_or_offtarget(row: dict[str, object]) -> bool:
    return page_kind(row) in {"context_or_blank", "offtarget_table", "unknown"}


def quality_flags(row: dict[str, object], profile: dict[str, object] | None = None) -> list[str]:
    flags: list[str] = []
    for flag in str(row.get("quality_flags") or "").split(";"):
        flag = flag.strip()
        if flag:
            flags.append(flag)
    checks = int_or_none(row.get("constraint_checks")) or 0
    failed = int_or_none(row.get("constraint_failed")) or 0
    rows = int_or_none(row.get("rows_extracted")) or 0
    if row.get("ocr_artifact") and checks == 0:
        flags.append("zero_constraint_checks")
    if failed:
        flags.append("constraint_failures")
    if row.get("ocr_artifact") and rows == 0:
        flags.append("zero_rows_extracted")
    status = clean_text(f"{row.get('ocr_status', '')} {row.get('page_status', '')} {row.get('validation', '')} {row.get('verdict', '')}")
    if re.search(r"\b(risk|rejected|failed|unchecked|low confidence)\b", status):
        flags.append("low_quality_ocr")
    if profile:
        if profile.get("ocr_exists") and not profile.get("profile_loaded"):
            flags.append("ocr_profile_read_error")
        if profile.get("profile_loaded") and int(profile.get("row_count") or 0) == 0:
            flags.append("zero_profile_rows")
        ratio = profile.get("row_label_ratio")
        if isinstance(ratio, float) and ratio < 0.55 and int(profile.get("row_count") or 0) >= 4:
            flags.append("row_labels_deficient")
    return uniq(flags, limit=30)


def age_ordinal(value: object) -> float | None:
    s = clean_text(value)
    if not s or re.search(r"\b(total|all religions|all ages|grand total)\b", s):
        return None
    # 0-1, 1-2, 5-10, 10 and over, under 1, 60+
    if re.search(r"\bunder\s*1\b", s):
        return 0.0
    m = re.search(r"\b(\d{1,3})\s*[-to]+\s*(\d{1,3})\b", s)
    if m:
        return float(m.group(1))
    m = re.search(r"\b(\d{1,3})\s*(and over|and above|plus|over|\+)\b", s)
    if m:
        return float(m.group(1))
    # Accept bare single-year age labels only when the whole label is numeric.
    # This avoids treating caste/occupation serial numbers like "6. Jain" or
    # "222 Horoscope casters" as age continuations.
    if re.fullmatch(r"\d{1,3}", s):
        n = float(s)
        return n if n <= 120 else None
    return None


LABEL_KEY_HINTS = {
    "age",
    "ages",
    "category",
    "class",
    "condition",
    "civil condition",
    "district",
    "division",
    "religion",
    "name",
}


def is_scalar(value: object) -> bool:
    return not isinstance(value, (dict, list))


def normalize_schema_token(value: object) -> str:
    return clean_text(value).replace(" ", "_")


def extract_row_label(row: dict[str, object]) -> tuple[str, str]:
    scalar_items = [(str(k), v) for k, v in row.items() if is_scalar(v)]
    for key, value in scalar_items:
        nk = clean_text(key)
        if nk in LABEL_KEY_HINTS or "age" in nk:
            return key, compact_text(value)
    for key, value in scalar_items:
        text = compact_text(value)
        if text and not re.fullmatch(r"[-+]?\d+(\.\d+)?", text):
            return key, text
    if scalar_items:
        key, value = scalar_items[0]
        return key, compact_text(value)
    return "", ""


def load_ocr_profile(path_value: object) -> dict[str, object]:
    path = Path(str(path_value or ""))
    profile: dict[str, object] = {
        "ocr_exists": bool(str(path_value or "").strip()),
        "profile_loaded": False,
        "profile_error": "",
        "ocr_title": "",
        "region": "",
        "metadata_column_groups": "",
        "section_count": 0,
        "row_count": 0,
        "row_label_count": 0,
        "row_label_ratio": None,
        "row_label_keys": "",
        "first_row_labels": "",
        "last_row_labels": "",
        "first_section": "",
        "last_section": "",
        "section_names": "",
        "section_name_signature": "",
        "schema_signature": "",
        "schema_tokens": "",
        "schema_token_count": 0,
        "first_age_ordinal": "",
        "last_age_ordinal": "",
        "starts_with_total_or_all": "",
        "has_numeric_cells": "",
    }
    if not path_value:
        return profile
    if not path.exists():
        profile["profile_error"] = "ocr_artifact_missing_on_disk"
        return profile
    try:
        with path.open(encoding="utf-8") as f:
            obj = json.load(f)
    except Exception as exc:  # noqa: BLE001 - keep processing other pages.
        profile["profile_error"] = f"{type(exc).__name__}: {exc}"
        return profile

    metadata = obj.get("metadata", {}) if isinstance(obj, dict) else {}
    data = obj.get("data", []) if isinstance(obj, dict) else obj
    if not isinstance(data, list):
        data = []

    section_names: list[str] = []
    row_labels: list[str] = []
    row_label_keys: list[str] = []
    schema_tokens: set[str] = set()
    age_values: list[float] = []
    numeric_cells = 0

    for cg in metadata.get("column_groups", []) if isinstance(metadata.get("column_groups"), list) else []:
        token = normalize_schema_token(cg)
        if token:
            schema_tokens.add(token)

    for section in data:
        if not isinstance(section, dict):
            continue
        section_name = compact_text(section.get("name") or section.get("title") or "")
        if section_name:
            section_names.append(section_name)
        rows = section.get("rows", [])
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            key, label = extract_row_label(row)
            if key:
                row_label_keys.append(key)
            if label:
                row_labels.append(label)
                age = age_ordinal(label)
                if age is not None:
                    age_values.append(age)
            for col_key, value in row.items():
                if key and col_key == key:
                    continue
                if isinstance(value, dict):
                    group = normalize_schema_token(col_key)
                    if group:
                        schema_tokens.add(group)
                    for nested_key, nested_value in value.items():
                        nested = normalize_schema_token(nested_key)
                        if group and nested:
                            schema_tokens.add(f"{group}.{nested}")
                        if isinstance(nested_value, (int, float)) or re.fullmatch(r"[-+]?\d+(\.\d+)?", str(nested_value or "").replace(",", "")):
                            numeric_cells += 1
                else:
                    token = normalize_schema_token(col_key)
                    if token:
                        schema_tokens.add(token)
                    if isinstance(value, (int, float)) or re.fullmatch(r"[-+]?\d+(\.\d+)?", str(value or "").replace(",", "")):
                        numeric_cells += 1

    row_count = sum(
        len(section.get("rows", []))
        for section in data
        if isinstance(section, dict) and isinstance(section.get("rows", []), list)
    )
    section_names_norm = [clean_text(s) for s in section_names if clean_text(s)]
    schema_list = sorted(t for t in schema_tokens if t)
    row_label_ratio = len(row_labels) / row_count if row_count else None
    first_label = row_labels[0] if row_labels else ""
    starts_total = bool(re.search(r"\b(total|all religions|all ages|grand total)\b", clean_text(first_label)))

    profile.update({
        "profile_loaded": True,
        "ocr_title": compact_text(metadata.get("title", "")),
        "region": compact_text(metadata.get("region", "")),
        "metadata_column_groups": "; ".join(str(v) for v in metadata.get("column_groups", []) if str(v).strip()) if isinstance(metadata.get("column_groups"), list) else "",
        "section_count": len(section_names),
        "row_count": row_count,
        "row_label_count": len(row_labels),
        "row_label_ratio": row_label_ratio,
        "row_label_keys": uniq_join(row_label_keys, limit=6),
        "first_row_labels": "; ".join(row_labels[:5]),
        "last_row_labels": "; ".join(row_labels[-5:]),
        "first_section": section_names[0] if section_names else "",
        "last_section": section_names[-1] if section_names else "",
        "section_names": "; ".join(uniq(section_names, limit=10)),
        "section_name_signature": stable_id("sect", *sorted(set(section_names_norm))) if section_names_norm else "",
        "schema_signature": stable_id("schema", *schema_list) if schema_list else "",
        "schema_tokens": "; ".join(schema_list[:30]),
        "schema_token_count": len(schema_list),
        "first_age_ordinal": float_fmt(age_values[0]) if age_values else "",
        "last_age_ordinal": float_fmt(age_values[-1]) if age_values else "",
        "starts_with_total_or_all": "yes" if starts_total else "no" if row_labels else "",
        "has_numeric_cells": "yes" if numeric_cells else "no",
    })
    return profile


def representative_score(row: dict[str, object]) -> tuple[int, int, int, int, int, str]:
    rows = int_or_none(row.get("rows_extracted")) or 0
    checks = int_or_none(row.get("constraint_checks")) or 0
    failed = int_or_none(row.get("constraint_failed")) or 0
    has_ocr = 1 if row.get("ocr_artifact") else 0
    has_image = 1 if row.get("image_path") else 0
    source_rank = 3 if "core_money_renders" in str(row.get("image_path") or "") else 2 if "final_tableVII" in str(row.get("image_path") or "") else 1
    return (has_ocr, rows, checks - failed * 25, source_rank, has_image, str(row.get("image_path") or ""))


def enrich_records(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    profile_cache: dict[str, dict[str, object]] = {}
    enriched: list[dict[str, object]] = []
    for idx, row in enumerate(rows, start=1):
        out: dict[str, object] = dict(row)
        out["input_record_id"] = stable_id("rec", idx, row.get("source_pdf"), row.get("pdf_page"), row.get("image_path"), row.get("ocr_artifact"), row.get("v3_component_id"))
        out["v5_source_key"] = source_key(out)
        out["pdf_page_num"] = int_or_none(out.get("pdf_page"))
        out["v5_table_number_key"] = table_number_key(out)
        out["v5_title_norm"] = strip_continuation_words(title_blob(out))
        out["v5_has_continuation_signal"] = "yes" if has_continuation_signal(out) else "no"
        art = str(out.get("ocr_artifact") or "")
        if art not in profile_cache:
            profile_cache[art] = load_ocr_profile(art)
        profile = profile_cache[art]
        for key, value in profile.items():
            out[f"profile_{key}"] = float_fmt(value) if isinstance(value, float) else value
        out["v5_quality_flags"] = uniq_join(quality_flags(out, profile), limit=30)
        enriched.append(out)
    return enriched


def build_physical_pages(records: list[dict[str, object]]) -> tuple[list[dict[str, object]], dict[str, str], dict[str, list[dict[str, object]]]]:
    by_page: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for record in records:
        page = str(record.get("pdf_page") or "").strip()
        if not page:
            page = f"missing_pdf_page::{record.get('input_record_id')}"
        by_page[(str(record.get("v5_source_key") or ""), page)].append(record)

    physical_pages: list[dict[str, object]] = []
    record_to_page: dict[str, str] = {}
    page_to_records: dict[str, list[dict[str, object]]] = {}

    for (source, page), page_records in by_page.items():
        rep = max(page_records, key=representative_score)
        page_num = int_or_none(rep.get("pdf_page"))
        physical_page_id = stable_id("phys", source, page)
        page_to_records[physical_page_id] = page_records
        for record in page_records:
            record_to_page[str(record["input_record_id"])] = physical_page_id

        duplicate_flags: list[str] = []
        kind_counts = Counter(page_kind(r) for r in page_records)
        family_counts = Counter(table_family(r) for r in page_records)
        title_norms = {str(r.get("v5_title_norm") or "") for r in page_records if r.get("v5_title_norm")}
        if len(page_records) > 1:
            duplicate_flags.append("duplicate_records_for_same_physical_page")
        if len(kind_counts) > 1:
            duplicate_flags.append("same_pdf_page_conflicting_page_kind")
        if len(family_counts) > 1:
            duplicate_flags.append("same_pdf_page_conflicting_table_family")
        if len(title_norms) > 1:
            duplicate_flags.append("same_pdf_page_multiple_titles")
        for flag in str(rep.get("v5_quality_flags") or "").split(";"):
            flag = flag.strip()
            if flag:
                duplicate_flags.append(flag)

        row_ratio = rep.get("profile_row_label_ratio")
        physical_pages.append({
            "physical_page_id": physical_page_id,
            "v5_source_key": source,
            "source_pdf": rep.get("source_pdf", ""),
            "source_pdf_id": rep.get("source_pdf_id", ""),
            "year": rep.get("year", ""),
            "province": rep.get("province", ""),
            "pdf_page": rep.get("pdf_page", ""),
            "pdf_page_num": page_num if page_num is not None else "",
            "representative_record_id": rep.get("input_record_id", ""),
            "record_count": len(page_records),
            "duplicate_record_count": max(0, len(page_records) - 1),
            "page_kind": page_kind(rep),
            "page_kind_counts": counter_json(page_kind(r) for r in page_records),
            "table_family_key": table_family(rep),
            "table_family_counts": counter_json(table_family(r) for r in page_records),
            "table_number_key": rep.get("v5_table_number_key", ""),
            "title": title_blob(rep),
            "title_norm": rep.get("v5_title_norm", ""),
            "has_continuation_signal": rep.get("v5_has_continuation_signal", ""),
            "classifier_page_class": rep.get("classifier_page_class", ""),
            "image_path": rep.get("image_path", ""),
            "ocr_artifact": rep.get("ocr_artifact", ""),
            "rows_extracted": rep.get("rows_extracted", ""),
            "constraint_checks": rep.get("constraint_checks", ""),
            "constraint_failed": rep.get("constraint_failed", ""),
            "profile_loaded": rep.get("profile_profile_loaded", ""),
            "profile_error": rep.get("profile_profile_error", ""),
            "profile_row_count": rep.get("profile_row_count", ""),
            "profile_section_count": rep.get("profile_section_count", ""),
            "profile_row_label_ratio": row_ratio,
            "profile_row_label_keys": rep.get("profile_row_label_keys", ""),
            "profile_first_row_labels": rep.get("profile_first_row_labels", ""),
            "profile_last_row_labels": rep.get("profile_last_row_labels", ""),
            "profile_first_section": rep.get("profile_first_section", ""),
            "profile_last_section": rep.get("profile_last_section", ""),
            "profile_section_names": rep.get("profile_section_names", ""),
            "profile_section_name_signature": rep.get("profile_section_name_signature", ""),
            "profile_schema_signature": rep.get("profile_schema_signature", ""),
            "profile_schema_tokens": rep.get("profile_schema_tokens", ""),
            "profile_first_age_ordinal": rep.get("profile_first_age_ordinal", ""),
            "profile_last_age_ordinal": rep.get("profile_last_age_ordinal", ""),
            "profile_starts_with_total_or_all": rep.get("profile_starts_with_total_or_all", ""),
            "profile_has_numeric_cells": rep.get("profile_has_numeric_cells", ""),
            "quality_flags": uniq_join(duplicate_flags, limit=30),
            "duplicate_image_paths": uniq_join((r.get("image_path", "") for r in page_records), limit=10),
            "duplicate_ocr_artifacts": uniq_join((r.get("ocr_artifact", "") for r in page_records), limit=10),
            "duplicate_v4_component_ids": uniq_join((r.get("component_id", "") for r in page_records), limit=10),
            "representative_score": "|".join(map(str, representative_score(rep)[:5])),
        })

    physical_pages.sort(key=lambda r: (
        str(r.get("v5_source_key") or ""),
        int(r.get("pdf_page_num")) if str(r.get("pdf_page_num") or "").isdigit() else 10**9,
        str(r.get("physical_page_id") or ""),
    ))
    return physical_pages, record_to_page, page_to_records


def schema_compatible(prev: dict[str, object], cur: dict[str, object]) -> tuple[bool, str]:
    a = str(prev.get("profile_schema_signature") or "")
    b = str(cur.get("profile_schema_signature") or "")
    if a and b and a == b:
        return True, "same_schema_signature"
    at = set(str(prev.get("profile_schema_tokens") or "").split("; "))
    bt = set(str(cur.get("profile_schema_tokens") or "").split("; "))
    at.discard("")
    bt.discard("")
    if not at and not bt:
        return False, "missing_schema_profile"
    score = jaccard(at, bt)
    if score >= 0.72:
        return True, f"schema_token_overlap_{score:.2f}"
    if not a or not b:
        return False, "missing_schema_profile"
    return False, f"schema_token_overlap_{score:.2f}"


def section_overlap(prev: dict[str, object], cur: dict[str, object]) -> tuple[bool, str]:
    pa = token_set(prev.get("profile_section_names"))
    ca = token_set(cur.get("profile_section_names"))
    score = jaccard(pa, ca)
    return score >= 0.35 and bool(pa and ca), f"section_overlap_{score:.2f}"


def row_label_ratio(row: dict[str, object]) -> float | None:
    value = str(row.get("profile_row_label_ratio") or "").strip()
    try:
        return float(value)
    except ValueError:
        return None


def is_row_label_deficient(row: dict[str, object]) -> bool:
    ratio = row_label_ratio(row)
    count = int_or_none(row.get("profile_row_count")) or 0
    return ratio is not None and count >= 4 and ratio < 0.55


def age_block_continues(prev: dict[str, object], cur: dict[str, object]) -> tuple[bool, str]:
    prev_last = str(prev.get("profile_last_age_ordinal") or "")
    cur_first = str(cur.get("profile_first_age_ordinal") or "")
    try:
        prev_last_n = float(prev_last)
        cur_first_n = float(cur_first)
    except ValueError:
        return False, "missing_age_sequence"
    if cur_first_n > prev_last_n and cur_first_n - prev_last_n <= 15:
        return True, f"age_labels_continue_{prev_last_n:g}_to_{cur_first_n:g}"
    return False, f"age_labels_do_not_continue_{prev_last}_to_{cur_first}"


def table_subject_compatible(prev: dict[str, object], cur: dict[str, object]) -> tuple[bool, str]:
    prev_kind = page_kind(prev)
    cur_kind = page_kind(cur)
    if prev_kind != cur_kind:
        return False, f"page_kind_switch_{prev_kind}_to_{cur_kind}"
    if prev_kind in {"context_or_blank", "offtarget_table", "unknown"}:
        return False, f"non_numeric_boundary_{prev_kind}"

    prev_family = table_family(prev)
    cur_family = table_family(cur)
    if prev_family and cur_family and prev_family != cur_family:
        return False, f"table_family_switch_{prev_family}_to_{cur_family}"

    prev_table = str(prev.get("table_number_key") or "")
    cur_table = str(cur.get("table_number_key") or "")
    if prev_table and cur_table and prev_table != cur_table:
        return False, f"table_number_switch_{prev_table}_to_{cur_table}"

    return True, "same_page_kind_table_family"


def edge_decision(prev: dict[str, object], cur: dict[str, object]) -> dict[str, object]:
    prev_page = int_or_none(prev.get("pdf_page"))
    cur_page = int_or_none(cur.get("pdf_page"))
    flags: list[str] = []
    if prev.get("v5_source_key") != cur.get("v5_source_key"):
        return {
            "edge_action": "break",
            "edge_reason": "source_key_changed",
            "edge_review_flags": "source_changed",
            "page_gap": "",
            "missing_pages": "",
        }
    if prev_page is None or cur_page is None:
        return {
            "edge_action": "break",
            "edge_reason": "missing_real_pdf_page_for_pairwise_order",
            "edge_review_flags": "missing_pdf_page",
            "page_gap": "",
            "missing_pages": "",
        }

    page_gap = cur_page - prev_page
    if page_gap <= 0:
        return {
            "edge_action": "break",
            "edge_reason": f"non_increasing_pdf_page_order_{prev_page}_to_{cur_page}",
            "edge_review_flags": "non_increasing_page_order",
            "page_gap": page_gap,
            "missing_pages": "",
        }
    if page_gap > 1:
        missing = f"{prev_page + 1}" if page_gap == 2 else f"{prev_page + 1}-{cur_page - 1}"
        return {
            "edge_action": "break",
            "edge_reason": f"missing_intermediate_pdf_pages_{missing}",
            "edge_review_flags": "missing_intermediate_pages",
            "page_gap": page_gap,
            "missing_pages": missing,
        }

    subject_ok, subject_reason = table_subject_compatible(prev, cur)
    schema_ok, schema_reason = schema_compatible(prev, cur)
    sections_ok, sections_reason = section_overlap(prev, cur)
    age_ok, age_reason = age_block_continues(prev, cur)
    title_sim = text_similarity(prev.get("title_norm") or prev.get("title"), cur.get("title_norm") or cur.get("title"))
    cur_row_deficient = is_row_label_deficient(cur)
    prev_row_deficient = is_row_label_deficient(prev)
    cur_has_profile = str(cur.get("profile_loaded") or "") == "True"
    prev_has_profile = str(prev.get("profile_loaded") or "") == "True"
    cur_starts_total = str(cur.get("profile_starts_with_total_or_all") or "") == "yes"

    if not subject_ok:
        return {
            "edge_action": "break",
            "edge_reason": subject_reason,
            "edge_review_flags": "incompatible_page_class_or_table_subject",
            "page_gap": page_gap,
            "missing_pages": "",
            "schema_reason": schema_reason,
            "section_reason": sections_reason,
            "title_similarity": float_fmt(title_sim),
        }

    row_label_missing = cur_has_profile and int_or_none(cur.get("profile_row_count")) and (
        cur_row_deficient or not str(cur.get("profile_first_row_labels") or "").strip()
    )
    prev_has_row_context = prev_has_profile and (
        not prev_row_deficient
        or bool(str(prev.get("profile_first_row_labels") or "").strip())
        or bool(str(prev.get("profile_last_row_labels") or "").strip())
    )
    if row_label_missing and prev_has_row_context and (has_continuation_signal(cur) or title_sim >= 0.78):
        flags.append("current_page_row_labels_deficient")
        if not schema_ok:
            flags.append("schema_switch_allowed_for_wide_table_right_half")
        if title_sim < 0.86:
            flags.append("title_similarity_review")
        return {
            "edge_action": "join",
            "edge_reason": f"adjacent_wide_table_right_half_requires_previous_row_labels; {schema_reason}; {sections_reason}",
            "edge_review_flags": uniq_join(flags, limit=20),
            "page_gap": page_gap,
            "missing_pages": "",
            "schema_reason": schema_reason,
            "section_reason": sections_reason,
            "title_similarity": float_fmt(title_sim),
        }

    if not schema_ok and cur_has_profile and prev_has_profile:
        return {
            "edge_action": "break",
            "edge_reason": f"column_schema_switch_{schema_reason}",
            "edge_review_flags": "column_schema_or_subtable_switch",
            "page_gap": page_gap,
            "missing_pages": "",
            "schema_reason": schema_reason,
            "section_reason": sections_reason,
            "title_similarity": float_fmt(title_sim),
        }

    if not cur_has_profile or not prev_has_profile:
        flags.append("no_ocr_structural_profile_for_pair")
        # Do not physically merge pages without structure. They can still be
        # surfaced as collection candidates if titles/schemas look close.
        return {
            "edge_action": "break",
            "edge_reason": f"adjacent_but_no_structural_profile_title_similarity_{title_sim:.2f}",
            "edge_review_flags": uniq_join(flags + ["uncertain_without_ocr_profile"], limit=20),
            "page_gap": page_gap,
            "missing_pages": "",
            "schema_reason": schema_reason,
            "section_reason": sections_reason,
            "title_similarity": float_fmt(title_sim),
        }

    if cur_row_deficient and schema_ok and (sections_ok or title_sim >= 0.55 or has_continuation_signal(cur)):
        flags.append("current_page_row_labels_deficient")
        if not sections_ok:
            flags.append("weak_section_overlap_review")
        return {
            "edge_action": "join",
            "edge_reason": f"adjacent_same_schema_current_row_labels_deficient_requires_previous_row_titles; {schema_reason}; {sections_reason}",
            "edge_review_flags": uniq_join(flags, limit=20),
            "page_gap": page_gap,
            "missing_pages": "",
            "schema_reason": schema_reason,
            "section_reason": sections_reason,
            "title_similarity": float_fmt(title_sim),
        }

    if age_ok and schema_ok and sections_ok and not cur_starts_total:
        return {
            "edge_action": "join",
            "edge_reason": f"adjacent_same_section_age_rows_continue; {age_reason}; {schema_reason}",
            "edge_review_flags": "",
            "page_gap": page_gap,
            "missing_pages": "",
            "schema_reason": schema_reason,
            "section_reason": sections_reason,
            "title_similarity": float_fmt(title_sim),
        }

    if has_continuation_signal(cur) and schema_ok and sections_ok and (cur_row_deficient or age_ok):
        if cur_row_deficient:
            flags.append("current_page_row_labels_deficient")
        return {
            "edge_action": "join",
            "edge_reason": f"adjacent_continued_heading_with_structural_continuation; {schema_reason}; {sections_reason}; {age_reason}",
            "edge_review_flags": uniq_join(flags, limit=20),
            "page_gap": page_gap,
            "missing_pages": "",
            "schema_reason": schema_reason,
            "section_reason": sections_reason,
            "title_similarity": float_fmt(title_sim),
        }

    break_flags = []
    if has_continuation_signal(cur):
        break_flags.append("continued_title_but_rows_look_self_contained")
    if prev_row_deficient:
        break_flags.append("previous_page_row_labels_deficient_unresolved")
    if cur_starts_total:
        break_flags.append("current_page_starts_new_total_or_section")
    if title_sim < 0.55:
        break_flags.append("title_switch")
    else:
        break_flags.append("possible_same_title_collection_not_physical_join")
    return {
        "edge_action": "break",
        "edge_reason": f"adjacent_but_no_required_structural_continuation; {schema_reason}; {sections_reason}; {age_reason}",
        "edge_review_flags": uniq_join(break_flags, limit=20),
        "page_gap": page_gap,
        "missing_pages": "",
        "schema_reason": schema_reason,
        "section_reason": sections_reason,
        "title_similarity": float_fmt(title_sim),
    }


def likely_collection_edge(prev: dict[str, object], cur: dict[str, object], edge: dict[str, object]) -> tuple[bool, str, str]:
    if prev.get("v5_source_key") != cur.get("v5_source_key"):
        return False, "source_changed", "source_changed"
    prev_page = int_or_none(prev.get("pdf_page"))
    cur_page = int_or_none(cur.get("pdf_page"))
    if prev_page is None or cur_page is None:
        return False, "missing_real_pdf_page_collection_not_bounded", "missing_pdf_page"
    if cur_page <= prev_page:
        return False, "non_increasing_pdf_page_collection_not_bounded", "non_increasing_page_order"
    if page_kind(prev) != page_kind(cur):
        return False, "page_kind_switch", "incompatible_collection_boundary"
    if page_kind(prev) in {"context_or_blank", "offtarget_table", "unknown"}:
        return False, "non_numeric_pages_not_collected", "non_numeric_boundary"
    if table_family(prev) != table_family(cur):
        return False, "table_family_switch", "table_subject_switch"
    schema_ok, schema_reason = schema_compatible(prev, cur)
    title_sim = text_similarity(prev.get("title_norm") or prev.get("title"), cur.get("title_norm") or cur.get("title"))
    variant_sim = religion_variant_similarity(prev, cur)
    table_prev = str(prev.get("table_number_key") or "")
    table_cur = str(cur.get("table_number_key") or "")
    same_table_number = bool(table_prev and table_cur and table_prev == table_cur)
    weak_no_profile_collection = schema_reason == "missing_schema_profile" and (
        title_sim >= 0.88 or (same_table_number and title_sim >= 0.70 and variant_sim >= 0.88)
    )
    if (schema_ok and (title_sim >= 0.62 or same_table_number or has_continuation_signal(cur))) or weak_no_profile_collection:
        flags = []
        if weak_no_profile_collection:
            flags.append("collection_no_ocr_structural_profile")
        if edge.get("missing_pages"):
            flags.append("collection_crosses_missing_pages_review")
        if title_sim < 0.75:
            flags.append("weak_title_collection_review")
        if variant_sim >= 0.88 and title_sim < 0.88:
            flags.append("religion_variant_same_base_title")
        if "continued_title_but_rows_look_self_contained" in str(edge.get("edge_review_flags") or ""):
            flags.append("self_contained_continued_subtable")
        return True, f"same_source_family_collection_candidate; {schema_reason}; title_similarity_{title_sim:.2f}; variant_similarity_{variant_sim:.2f}", uniq_join(flags, limit=10)
    return False, f"collection_not_confident; {schema_reason}; title_similarity_{title_sim:.2f}; variant_similarity_{variant_sim:.2f}", "possible_unrelated_subtable"


def build_runs(physical_pages: list[dict[str, object]]) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    pages_by_source: dict[str, list[dict[str, object]]] = defaultdict(list)
    for page in physical_pages:
        pages_by_source[str(page.get("v5_source_key") or "")].append(page)

    all_edges: list[dict[str, object]] = []
    run_members: list[dict[str, object]] = []
    runs: list[dict[str, object]] = []
    missing_gap_candidates: list[dict[str, object]] = []

    def flush(run: list[dict[str, object]]) -> None:
        if not run:
            return
        source = str(run[0].get("v5_source_key") or "")
        pages = [int_or_none(r.get("pdf_page")) for r in run if int_or_none(r.get("pdf_page")) is not None]
        first = min(pages) if pages else ""
        last = max(pages) if pages else ""
        basis_kind = Counter(page_kind(r) for r in run).most_common(1)[0][0] if run else "unknown"
        run_id = stable_id("run", source, first, last, len(run), basis_kind, run[0].get("physical_page_id"))
        component_flags: list[str] = []
        for pos, page in enumerate(run, start=1):
            page["physical_run_id"] = run_id
            page["position_in_physical_run"] = pos
            run_members.append({
                "physical_run_id": run_id,
                "position_in_physical_run": pos,
                **{k: page.get(k, "") for k in PHYSICAL_PAGE_FIELDS},
                "edge_from_prev_action": page.get("edge_from_prev_action", ""),
                "edge_from_prev_reason": page.get("edge_from_prev_reason", ""),
                "edge_review_flags": page.get("edge_review_flags", ""),
            })
            for field in ["quality_flags", "edge_review_flags"]:
                for flag in str(page.get(field) or "").split(";"):
                    flag = flag.strip()
                    if flag:
                        component_flags.append(flag)
        unique_pages_are_consecutive = True
        gap_edges = []
        for a, b in zip(pages, pages[1:]):
            if b != a + 1:
                unique_pages_are_consecutive = False
                gap_edges.append(f"{a}->{b}")
        runs.append({
            "physical_run_id": run_id,
            "run_basis": f"structural_{basis_kind}_run",
            "review_flag": "review" if component_flags else "ok",
            "run_review_flags": uniq_join(component_flags, limit=30),
            "v5_source_key": source,
            "source_pdf": uniq_join((r.get("source_pdf", "") for r in run), limit=4),
            "source_pdf_id": uniq_join((r.get("source_pdf_id", "") for r in run), limit=4),
            "year": uniq_join((r.get("year", "") for r in run), limit=4),
            "province": uniq_join((r.get("province", "") for r in run), limit=4),
            "n_physical_pages": len(run),
            "page_range": f"{first}-{last}" if first != "" and last != "" and first != last else str(first),
            "unique_pages_are_consecutive": str(unique_pages_are_consecutive),
            "unique_page_gap_edges": "; ".join(gap_edges),
            "page_kind_counts": counter_json(page_kind(r) for r in run),
            "table_family_counts": counter_json(table_family(r) for r in run),
            "primary_title": Counter(str(r.get("title") or "") for r in run if r.get("title")).most_common(1)[0][0] if any(r.get("title") for r in run) else "",
            "title_norms": uniq_join((r.get("title_norm", "") for r in run), limit=6),
            "schema_signatures": uniq_join((r.get("profile_schema_signature", "") for r in run), limit=8),
            "section_signatures": uniq_join((r.get("profile_section_name_signature", "") for r in run), limit=8),
            "total_source_records": sum(int(r.get("record_count") or 0) for r in run),
            "duplicate_record_count": sum(int(r.get("duplicate_record_count") or 0) for r in run),
            "row_records_from_json_pages": sum(int_or_none(r.get("profile_row_count")) or 0 for r in run),
            "constraint_checks_total": sum(int_or_none(r.get("constraint_checks")) or 0 for r in run),
            "constraint_failed_total": sum(int_or_none(r.get("constraint_failed")) or 0 for r in run),
            "example_images": uniq_join((r.get("image_path", "") for r in run), limit=4),
        })

    for source, source_pages in sorted(pages_by_source.items()):
        ordered = sorted(source_pages, key=lambda r: (
            int(r.get("pdf_page_num")) if str(r.get("pdf_page_num") or "").isdigit() else 10**9,
            str(r.get("physical_page_id") or ""),
        ))
        run: list[dict[str, object]] = []
        prev: dict[str, object] | None = None
        for page in ordered:
            if prev is None or not run:
                page["edge_from_prev_action"] = "start"
                page["edge_from_prev_reason"] = "first_physical_page_in_source"
                page["edge_review_flags"] = page.get("quality_flags", "")
                run = [page]
                prev = page
                continue

            edge = edge_decision(prev, page)
            edge_row = {
                "v5_source_key": source,
                "prev_physical_page_id": prev.get("physical_page_id", ""),
                "physical_page_id": page.get("physical_page_id", ""),
                "prev_pdf_page": prev.get("pdf_page", ""),
                "pdf_page": page.get("pdf_page", ""),
                "page_gap": edge.get("page_gap", ""),
                "missing_pages": edge.get("missing_pages", ""),
                "edge_action": edge.get("edge_action", ""),
                "edge_reason": edge.get("edge_reason", ""),
                "edge_review_flags": edge.get("edge_review_flags", ""),
                "prev_page_kind": page_kind(prev),
                "page_kind": page_kind(page),
                "prev_table_family_key": table_family(prev),
                "table_family_key": table_family(page),
                "prev_table_number_key": prev.get("table_number_key", ""),
                "table_number_key": page.get("table_number_key", ""),
                "schema_reason": edge.get("schema_reason", ""),
                "section_reason": edge.get("section_reason", ""),
                "title_similarity": edge.get("title_similarity", ""),
                "prev_title": prev.get("title", ""),
                "title": page.get("title", ""),
                "prev_row_label_ratio": prev.get("profile_row_label_ratio", ""),
                "row_label_ratio": page.get("profile_row_label_ratio", ""),
                "prev_first_row_labels": prev.get("profile_first_row_labels", ""),
                "first_row_labels": page.get("profile_first_row_labels", ""),
                "prev_last_row_labels": prev.get("profile_last_row_labels", ""),
                "last_row_labels": page.get("profile_last_row_labels", ""),
                "prev_image_path": prev.get("image_path", ""),
                "image_path": page.get("image_path", ""),
            }
            coll_ok, coll_reason, coll_flags = likely_collection_edge(prev, page, edge)
            edge_row["collection_candidate"] = "yes" if coll_ok else "no"
            edge_row["collection_reason"] = coll_reason
            edge_row["collection_review_flags"] = coll_flags
            all_edges.append(edge_row)

            if edge.get("missing_pages") and coll_ok:
                missing_gap_candidates.append({
                    **edge_row,
                    "candidate_type": "missing_bridge_between_related_pages",
                })

            if edge.get("edge_action") == "join":
                page["edge_from_prev_action"] = "join"
                page["edge_from_prev_reason"] = edge.get("edge_reason", "")
                page["edge_review_flags"] = uniq_join([edge.get("edge_review_flags", ""), page.get("quality_flags", "")], limit=30)
                run.append(page)
            else:
                page["edge_from_prev_action"] = "break_start"
                page["edge_from_prev_reason"] = edge.get("edge_reason", "")
                page["edge_review_flags"] = uniq_join([edge.get("edge_review_flags", ""), page.get("quality_flags", "")], limit=30)
                flush(run)
                run = [page]
            prev = page
        flush(run)

    runs.sort(key=lambda r: (str(r.get("v5_source_key", "")), str(r.get("page_range", "")), str(r.get("physical_run_id", ""))))
    run_members.sort(key=lambda r: (str(r.get("v5_source_key", "")), str(r.get("physical_run_id", "")), int(r.get("position_in_physical_run") or 0)))
    return runs, run_members, all_edges, missing_gap_candidates


def build_collection_groups(runs: list[dict[str, object]], run_members: list[dict[str, object]], edges: list[dict[str, object]]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    first_page_by_run: dict[str, dict[str, object]] = {}
    last_page_by_run: dict[str, dict[str, object]] = {}
    members_by_run: dict[str, list[dict[str, object]]] = defaultdict(list)
    for member in run_members:
        members_by_run[str(member.get("physical_run_id"))].append(member)
    for run_id, members in members_by_run.items():
        ordered = sorted(members, key=lambda r: int(r.get("position_in_physical_run") or 0))
        first_page_by_run[run_id] = ordered[0]
        last_page_by_run[run_id] = ordered[-1]

    page_to_run: dict[str, str] = {}
    for member in run_members:
        page_to_run[str(member.get("physical_page_id"))] = str(member.get("physical_run_id"))

    edge_by_run_pair: dict[tuple[str, str], dict[str, object]] = {}
    for edge in edges:
        prev_run = page_to_run.get(str(edge.get("prev_physical_page_id") or ""))
        cur_run = page_to_run.get(str(edge.get("physical_page_id") or ""))
        if prev_run and cur_run and prev_run != cur_run:
            edge_by_run_pair[(prev_run, cur_run)] = edge

    runs_by_source: dict[str, list[dict[str, object]]] = defaultdict(list)
    for run in runs:
        runs_by_source[str(run.get("v5_source_key") or "")].append(run)

    collection_groups: list[dict[str, object]] = []
    collection_members: list[dict[str, object]] = []

    def flush_collection(chain: list[dict[str, object]], reasons: list[str], flags: list[str]) -> None:
        if not chain:
            return
        first = chain[0]
        last = chain[-1]
        if len(chain) == 1:
            collection_id = stable_id("coll", first.get("physical_run_id"))
            basis = "single_physical_run"
        else:
            collection_id = stable_id("coll", first.get("v5_source_key"), first.get("physical_run_id"), last.get("physical_run_id"), len(chain))
            basis = "bounded_same_source_family_schema_collection_candidate"
        page_ranges = [str(r.get("page_range") or "") for r in chain]
        collection_groups.append({
            "collection_group_id": collection_id,
            "collection_basis": basis,
            "review_flag": "review" if flags or len(chain) > 1 else str(first.get("review_flag") or "ok"),
            "collection_review_flags": uniq_join(flags, limit=30),
            "v5_source_key": first.get("v5_source_key", ""),
            "source_pdf": uniq_join((r.get("source_pdf", "") for r in chain), limit=4),
            "source_pdf_id": uniq_join((r.get("source_pdf_id", "") for r in chain), limit=4),
            "n_physical_runs": len(chain),
            "n_physical_pages": sum(int(r.get("n_physical_pages") or 0) for r in chain),
            "page_ranges": "; ".join(page_ranges),
            "page_kind_counts": counter_json((r.get("page_kind_counts", "") for r in chain)),
            "primary_title": first.get("primary_title", ""),
            "collection_reasons": uniq_join(reasons, limit=20),
            "example_images": uniq_join((r.get("example_images", "") for r in chain), limit=4),
        })
        for pos, run in enumerate(chain, start=1):
            collection_members.append({
                "collection_group_id": collection_id,
                "position_in_collection": pos,
                **run,
            })

    for source, source_runs in sorted(runs_by_source.items()):
        ordered = sorted(source_runs, key=lambda r: (
            int(str(r.get("page_range", "")).split("-", 1)[0]) if re.match(r"^\d+", str(r.get("page_range", ""))) else 10**9,
            str(r.get("physical_run_id") or ""),
        ))
        chain: list[dict[str, object]] = []
        reasons: list[str] = []
        flags: list[str] = []
        prev_run: dict[str, object] | None = None
        for run in ordered:
            if prev_run is None:
                chain = [run]
                prev_run = run
                continue
            edge = edge_by_run_pair.get((str(prev_run.get("physical_run_id")), str(run.get("physical_run_id"))))
            if edge and edge.get("collection_candidate") == "yes":
                chain.append(run)
                reasons.append(str(edge.get("collection_reason") or ""))
                if edge.get("collection_review_flags"):
                    flags.append(str(edge.get("collection_review_flags")))
            else:
                flush_collection(chain, reasons, flags)
                chain = [run]
                reasons = []
                flags = []
            prev_run = run
        flush_collection(chain, reasons, flags)

    return collection_groups, collection_members


def build_edge_review_queue(edges: list[dict[str, object]]) -> list[dict[str, object]]:
    queue: list[dict[str, object]] = []
    for edge in edges:
        flags = str(edge.get("edge_review_flags") or "")
        reason = str(edge.get("edge_reason") or "")
        collection_flags = str(edge.get("collection_review_flags") or "")
        title_similarity = float(edge.get("title_similarity") or 0)
        category = ""
        priority = 99

        if edge.get("edge_action") == "join":
            category = "required_physical_join"
            priority = 1
        elif "missing_intermediate_pages" in flags:
            category = "missing_intermediate_pages"
            priority = 2
        elif "column_schema_or_subtable_switch" in flags:
            category = "schema_or_subtable_switch"
            priority = 3
        elif "incompatible_page_class_or_table_subject" in flags:
            category = "incompatible_page_class_or_table_subject"
            priority = 4
        elif "table_number_switch" in reason or "table_family_switch" in reason:
            category = "table_subject_switch"
            priority = 4
        elif "title_switch" in flags:
            category = "title_switch"
            priority = 5
        elif edge.get("collection_candidate") == "yes" and collection_flags:
            category = "collection_candidate_review"
            priority = 6
        elif "no_ocr_structural_profile_for_pair" in flags and title_similarity >= 0.88:
            category = "high_title_similarity_no_structural_profile"
            priority = 7

        if category:
            queue.append({
                "review_priority": priority,
                "review_category": category,
                **edge,
            })

    queue.sort(key=lambda r: (
        int(r.get("review_priority") or 99),
        str(r.get("v5_source_key") or ""),
        int_or_none(r.get("prev_pdf_page")) if int_or_none(r.get("prev_pdf_page")) is not None else 10**9,
        int_or_none(r.get("pdf_page")) if int_or_none(r.get("pdf_page")) is not None else 10**9,
    ))
    return queue


def build_run_review_queue(runs: list[dict[str, object]]) -> list[dict[str, object]]:
    queue: list[dict[str, object]] = []
    quality_flags = {
        "constraint_failures",
        "low_quality_ocr",
        "ocr_profile_read_error",
        "row_labels_deficient",
        "zero_constraint_checks",
        "zero_profile_rows",
        "zero_rows_extracted",
    }
    dedupe_flags = {
        "duplicate_records_for_same_physical_page",
        "same_pdf_page_conflicting_page_kind",
        "same_pdf_page_conflicting_table_family",
        "same_pdf_page_multiple_titles",
    }
    edge_flags = {
        "column_schema_or_subtable_switch",
        "current_page_row_labels_deficient",
        "incompatible_page_class_or_table_subject",
        "missing_intermediate_pages",
        "missing_pdf_page",
        "non_increasing_page_order",
        "previous_page_row_labels_deficient_unresolved",
        "schema_switch_allowed_for_wide_table_right_half",
        "title_similarity_review",
    }

    for run in runs:
        flags = {f.strip() for f in str(run.get("run_review_flags") or "").split(";") if f.strip()}
        n_pages = int(run.get("n_physical_pages") or 0)
        category = ""
        priority = 99

        if n_pages > 1:
            category = "required_physical_assembly_run"
            priority = 1
        elif flags & edge_flags:
            category = "edge_transition_review"
            priority = 3
        elif flags & quality_flags:
            category = "ocr_quality_or_zero_check_review"
            priority = 4
        elif flags & dedupe_flags:
            category = "same_page_dedupe_conflict_review"
            priority = 5

        if category:
            queue.append({
                "review_priority": priority,
                "review_category": category,
                **run,
            })

    queue.sort(key=lambda r: (
        int(r.get("review_priority") or 99),
        str(r.get("v5_source_key") or ""),
        str(r.get("page_range") or ""),
        str(r.get("physical_run_id") or ""),
    ))
    return queue


PHYSICAL_PAGE_FIELDS = [
    "physical_page_id",
    "v5_source_key",
    "source_pdf",
    "source_pdf_id",
    "year",
    "province",
    "pdf_page",
    "pdf_page_num",
    "representative_record_id",
    "record_count",
    "duplicate_record_count",
    "page_kind",
    "page_kind_counts",
    "table_family_key",
    "table_family_counts",
    "table_number_key",
    "title",
    "title_norm",
    "has_continuation_signal",
    "classifier_page_class",
    "image_path",
    "ocr_artifact",
    "rows_extracted",
    "constraint_checks",
    "constraint_failed",
    "profile_loaded",
    "profile_error",
    "profile_row_count",
    "profile_section_count",
    "profile_row_label_ratio",
    "profile_row_label_keys",
    "profile_first_row_labels",
    "profile_last_row_labels",
    "profile_first_section",
    "profile_last_section",
    "profile_section_names",
    "profile_section_name_signature",
    "profile_schema_signature",
    "profile_schema_tokens",
    "profile_first_age_ordinal",
    "profile_last_age_ordinal",
    "profile_starts_with_total_or_all",
    "profile_has_numeric_cells",
    "quality_flags",
    "duplicate_image_paths",
    "duplicate_ocr_artifacts",
    "duplicate_v4_component_ids",
    "representative_score",
]


def build_record_rows(records: list[dict[str, object]], record_to_page: dict[str, str], physical_pages: list[dict[str, object]]) -> list[dict[str, object]]:
    page_by_id = {str(p["physical_page_id"]): p for p in physical_pages}
    rows: list[dict[str, object]] = []
    for record in records:
        page_id = record_to_page.get(str(record.get("input_record_id") or ""), "")
        page = page_by_id.get(page_id, {})
        rows.append({
            "input_record_id": record.get("input_record_id", ""),
            "physical_page_id": page_id,
            "physical_run_id": page.get("physical_run_id", ""),
            "position_in_physical_run": page.get("position_in_physical_run", ""),
            "record_is_representative": "yes" if record.get("input_record_id") == page.get("representative_record_id") else "no",
            "v5_source_key": record.get("v5_source_key", ""),
            "source_pdf": record.get("source_pdf", ""),
            "source_pdf_id": record.get("source_pdf_id", ""),
            "year": record.get("year", ""),
            "province": record.get("province", ""),
            "pdf_page": record.get("pdf_page", ""),
            "page_kind": record.get("page_kind", ""),
            "table_family_key": record.get("table_family_key", ""),
            "table_number_key": record.get("v5_table_number_key", ""),
            "title": title_blob(record),
            "image_path": record.get("image_path", ""),
            "ocr_artifact": record.get("ocr_artifact", ""),
            "rows_extracted": record.get("rows_extracted", ""),
            "constraint_checks": record.get("constraint_checks", ""),
            "constraint_failed": record.get("constraint_failed", ""),
            "v4_component_id": record.get("component_id", ""),
            "v4_component_basis": record.get("component_basis", ""),
            "v4_edge_from_prev_action": record.get("edge_from_prev_action", ""),
            "v4_edge_from_prev_reason": record.get("edge_from_prev_reason", ""),
            "v5_quality_flags": record.get("v5_quality_flags", ""),
            "profile_loaded": record.get("profile_profile_loaded", ""),
            "profile_row_label_ratio": record.get("profile_row_label_ratio", ""),
            "profile_schema_signature": record.get("profile_schema_signature", ""),
            "profile_section_names": record.get("profile_section_names", ""),
        })
    return rows


def build_overassociation_report(records: list[dict[str, object]], record_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    run_by_record = {str(r.get("input_record_id")): str(r.get("physical_run_id") or "") for r in record_rows}
    page_by_record = {str(r.get("input_record_id")): str(r.get("physical_page_id") or "") for r in record_rows}
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record in records:
        grouped[str(record.get("component_id") or "missing_v4_component")].append(record)

    out: list[dict[str, object]] = []
    for v4_component, rows in grouped.items():
        runs = {run_by_record.get(str(r.get("input_record_id"))) for r in rows if run_by_record.get(str(r.get("input_record_id")))}
        pages = {page_by_record.get(str(r.get("input_record_id"))) for r in rows if page_by_record.get(str(r.get("input_record_id")))}
        if len(runs) <= 1 and len(rows) <= 1:
            continue
        pdf_pages = sorted({int_or_none(r.get("pdf_page")) for r in rows if int_or_none(r.get("pdf_page")) is not None})
        gap_count = 0
        gaps: list[str] = []
        for a, b in zip(pdf_pages, pdf_pages[1:]):
            if b > a + 1:
                gap_count += b - a - 1
                gaps.append(f"{a + 1}" if b == a + 2 else f"{a + 1}-{b - 1}")
        if len(runs) > 1 or gap_count:
            out.append({
                "v4_component_id": v4_component,
                "v4_component_basis": uniq_join((r.get("component_basis", "") for r in rows), limit=4),
                "v4_record_count": len(rows),
                "v4_unique_physical_pages": len(pages),
                "v5_physical_run_count": len(runs),
                "v5_physical_runs": uniq_join(sorted(runs), limit=12),
                "v4_page_range": f"{pdf_pages[0]}-{pdf_pages[-1]}" if pdf_pages else "",
                "v4_gap_count": gap_count,
                "v4_gaps": "; ".join(gaps),
                "page_kind_counts": counter_json((r.get("page_kind", "") for r in rows)),
                "title_count": len({str(r.get("v5_title_norm") or "") for r in rows if r.get("v5_title_norm")}),
                "titles": uniq_join((title_blob(r) for r in rows), limit=6),
                "source_pdfs": uniq_join((r.get("source_pdf", "") for r in rows), limit=6),
                "example_images": uniq_join((r.get("image_path", "") for r in rows), limit=4),
            })
    out.sort(key=lambda r: (int(r.get("v5_physical_run_count") or 0), int(r.get("v4_record_count") or 0)), reverse=True)
    return out


def write_workbook(path: Path, sheets: dict[str, tuple[list[dict[str, object]], list[str]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    workbook = xlsxwriter.Workbook(str(path))
    header_fmt = workbook.add_format({"bold": True, "bg_color": "#D9EAF7", "border": 1})
    wrap_fmt = workbook.add_format({"text_wrap": True, "valign": "top"})
    for sheet_name, (rows, fields) in sheets.items():
        ws = workbook.add_worksheet(sheet_name[:31])
        for col, field in enumerate(fields):
            ws.write(0, col, field, header_fmt)
        for r_idx, row in enumerate(rows, start=1):
            for c_idx, field in enumerate(fields):
                ws.write(r_idx, c_idx, row.get(field, ""), wrap_fmt if field.endswith("reason") or field.endswith("flags") or field in {"title", "example_images"} else None)
        ws.freeze_panes(1, 0)
        ws.autofilter(0, 0, max(1, len(rows)), max(0, len(fields) - 1))
        for col, field in enumerate(fields):
            width = 18
            if "reason" in field or "flags" in field or "title" in field:
                width = 42
            elif field in {"image_path", "ocr_artifact", "example_images"}:
                width = 55
            ws.set_column(col, col, width)
    workbook.close()


RUN_FIELDS = [
    "physical_run_id",
    "run_basis",
    "review_flag",
    "run_review_flags",
    "v5_source_key",
    "source_pdf",
    "source_pdf_id",
    "year",
    "province",
    "n_physical_pages",
    "page_range",
    "unique_pages_are_consecutive",
    "unique_page_gap_edges",
    "page_kind_counts",
    "table_family_counts",
    "primary_title",
    "title_norms",
    "schema_signatures",
    "section_signatures",
    "total_source_records",
    "duplicate_record_count",
    "row_records_from_json_pages",
    "constraint_checks_total",
    "constraint_failed_total",
    "example_images",
]

RUN_MEMBER_FIELDS = [
    "physical_run_id",
    "position_in_physical_run",
    *PHYSICAL_PAGE_FIELDS,
    "edge_from_prev_action",
    "edge_from_prev_reason",
    "edge_review_flags",
]

EDGE_FIELDS = [
    "v5_source_key",
    "prev_physical_page_id",
    "physical_page_id",
    "prev_pdf_page",
    "pdf_page",
    "page_gap",
    "missing_pages",
    "edge_action",
    "edge_reason",
    "edge_review_flags",
    "prev_page_kind",
    "page_kind",
    "prev_table_family_key",
    "table_family_key",
    "prev_table_number_key",
    "table_number_key",
    "schema_reason",
    "section_reason",
    "title_similarity",
    "collection_candidate",
    "collection_reason",
    "collection_review_flags",
    "prev_title",
    "title",
    "prev_row_label_ratio",
    "row_label_ratio",
    "prev_first_row_labels",
    "first_row_labels",
    "prev_last_row_labels",
    "last_row_labels",
    "prev_image_path",
    "image_path",
]

EDGE_REVIEW_FIELDS = ["review_priority", "review_category", *EDGE_FIELDS]
RUN_REVIEW_FIELDS = ["review_priority", "review_category", *RUN_FIELDS]

RECORD_FIELDS = [
    "input_record_id",
    "physical_page_id",
    "physical_run_id",
    "position_in_physical_run",
    "record_is_representative",
    "v5_source_key",
    "source_pdf",
    "source_pdf_id",
    "year",
    "province",
    "pdf_page",
    "page_kind",
    "table_family_key",
    "table_number_key",
    "title",
    "image_path",
    "ocr_artifact",
    "rows_extracted",
    "constraint_checks",
    "constraint_failed",
    "v4_component_id",
    "v4_component_basis",
    "v4_edge_from_prev_action",
    "v4_edge_from_prev_reason",
    "v5_quality_flags",
    "profile_loaded",
    "profile_row_label_ratio",
    "profile_schema_signature",
    "profile_section_names",
]

OVERASSOC_FIELDS = [
    "v4_component_id",
    "v4_component_basis",
    "v4_record_count",
    "v4_unique_physical_pages",
    "v5_physical_run_count",
    "v5_physical_runs",
    "v4_page_range",
    "v4_gap_count",
    "v4_gaps",
    "page_kind_counts",
    "title_count",
    "titles",
    "source_pdfs",
    "example_images",
]

COLLECTION_FIELDS = [
    "collection_group_id",
    "collection_basis",
    "review_flag",
    "collection_review_flags",
    "v5_source_key",
    "source_pdf",
    "source_pdf_id",
    "n_physical_runs",
    "n_physical_pages",
    "page_ranges",
    "page_kind_counts",
    "primary_title",
    "collection_reasons",
    "example_images",
]

COLLECTION_MEMBER_FIELDS = [
    "collection_group_id",
    "position_in_collection",
    *RUN_FIELDS,
]


def run_invariants(runs: list[dict[str, object]], edges: list[dict[str, object]], run_members: list[dict[str, object]]) -> dict[str, object]:
    failures: list[str] = []
    for edge in edges:
        if edge.get("edge_action") == "join":
            if str(edge.get("page_gap")) != "1":
                failures.append(f"join_non_adjacent:{edge.get('prev_pdf_page')}->{edge.get('pdf_page')}")
            if edge.get("prev_page_kind") != edge.get("page_kind"):
                failures.append(f"join_page_kind_switch:{edge.get('prev_page_kind')}->{edge.get('page_kind')}")
    members_by_run: dict[str, list[dict[str, object]]] = defaultdict(list)
    for member in run_members:
        members_by_run[str(member.get("physical_run_id") or "")].append(member)
    for run_id, members in members_by_run.items():
        pages = sorted(int_or_none(m.get("pdf_page")) for m in members if int_or_none(m.get("pdf_page")) is not None)
        for a, b in zip(pages, pages[1:]):
            if b != a + 1:
                failures.append(f"run_internal_gap:{run_id}:{a}->{b}")
    return {
        "invariant_failure_count": len(failures),
        "invariant_failures": failures[:50],
        "join_edges": sum(1 for e in edges if e.get("edge_action") == "join"),
        "break_edges": sum(1 for e in edges if e.get("edge_action") == "break"),
        "multi_page_physical_runs": sum(1 for r in runs if int(r.get("n_physical_pages") or 0) > 1),
        "max_physical_run_pages": max((int(r.get("n_physical_pages") or 0) for r in runs), default=0),
    }


def main() -> None:
    raw_rows = read_csv(V4_PAGES)
    if not raw_rows:
        raise SystemExit(f"Missing v4 page inventory: {V4_PAGES}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    records = enrich_records(raw_rows)
    physical_pages, record_to_page, page_to_records = build_physical_pages(records)
    runs, run_members, edges, missing_gap_candidates = build_runs(physical_pages)
    record_rows = build_record_rows(records, record_to_page, physical_pages)
    overassociation = build_overassociation_report(records, record_rows)
    collection_groups, collection_members = build_collection_groups(runs, run_members, edges)
    required_run_ids = {
        str(r.get("physical_run_id"))
        for r in runs
        if int(r.get("n_physical_pages") or 0) > 1
    }
    required_assembly_groups = [
        r for r in runs if str(r.get("physical_run_id")) in required_run_ids
    ]
    required_assembly_members = [
        m for m in run_members if str(m.get("physical_run_id")) in required_run_ids
    ]
    required_assembly_edges = [e for e in edges if e.get("edge_action") == "join"]
    edge_review_queue = build_edge_review_queue(edges)
    run_review_queue = build_run_review_queue(runs)

    summary = {
        "input_v4_records": len(records),
        "physical_pages": len(physical_pages),
        "records_with_existing_ocr_artifact": sum(1 for r in records if r.get("ocr_artifact")),
        "records_with_loaded_ocr_profile": sum(1 for r in records if str(r.get("profile_profile_loaded")) == "True"),
        "physical_runs": len(runs),
        "multi_page_physical_runs": sum(1 for r in runs if int(r.get("n_physical_pages") or 0) > 1),
        "max_physical_run_pages": max((int(r.get("n_physical_pages") or 0) for r in runs), default=0),
        "required_assembly_groups": len(required_assembly_groups),
        "required_assembly_edges": len(required_assembly_edges),
        "join_edges": sum(1 for e in edges if e.get("edge_action") == "join"),
        "break_edges": sum(1 for e in edges if e.get("edge_action") == "break"),
        "edge_review_queue_rows": len(edge_review_queue),
        "run_review_queue_rows": len(run_review_queue),
        "missing_gap_candidates": len(missing_gap_candidates),
        "collection_groups": len(collection_groups),
        "multi_run_collection_groups": sum(1 for g in collection_groups if int(g.get("n_physical_runs") or 0) > 1),
        "v4_components_split_by_v5": sum(1 for r in overassociation if int(r.get("v5_physical_run_count") or 0) > 1),
        "largest_v4_split_run_count": max((int(r.get("v5_physical_run_count") or 0) for r in overassociation), default=0),
    }
    summary.update(run_invariants(runs, edges, run_members))

    write_csv(OUT_DIR / "v5_page_records.csv", record_rows, RECORD_FIELDS)
    write_csv(OUT_DIR / "v5_physical_pages.csv", physical_pages, PHYSICAL_PAGE_FIELDS)
    write_csv(OUT_DIR / "v5_physical_edges.csv", edges, EDGE_FIELDS)
    write_csv(OUT_DIR / "v5_physical_runs.csv", runs, RUN_FIELDS)
    write_csv(OUT_DIR / "v5_run_members.csv", run_members, RUN_MEMBER_FIELDS)
    write_csv(OUT_DIR / "v5_required_assembly_groups.csv", required_assembly_groups, RUN_FIELDS)
    write_csv(OUT_DIR / "v5_required_assembly_members.csv", required_assembly_members, RUN_MEMBER_FIELDS)
    write_csv(OUT_DIR / "v5_required_assembly_edges.csv", required_assembly_edges, EDGE_FIELDS)
    write_csv(OUT_DIR / "v5_edge_review_queue.csv", edge_review_queue, EDGE_REVIEW_FIELDS)
    write_csv(OUT_DIR / "v5_run_review_queue.csv", run_review_queue, RUN_REVIEW_FIELDS)
    write_csv(OUT_DIR / "v5_missing_gap_candidates.csv", missing_gap_candidates, ["candidate_type", *EDGE_FIELDS])
    write_csv(OUT_DIR / "v5_collection_groups.csv", collection_groups, COLLECTION_FIELDS)
    write_csv(OUT_DIR / "v5_collection_members.csv", collection_members, COLLECTION_MEMBER_FIELDS)
    write_csv(OUT_DIR / "v5_overassociation_examples.csv", overassociation, OVERASSOC_FIELDS)

    with (OUT_DIR / "v5_structural_assembly_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    summary_rows = [{"metric": k, "value": json.dumps(v) if isinstance(v, (list, dict)) else v} for k, v in summary.items()]
    write_workbook(
        OUT_DIR / "v5_structural_assembly_review.xlsx",
        {
            "summary": (summary_rows, ["metric", "value"]),
            "physical_runs": (runs, RUN_FIELDS),
            "run_members": (run_members, RUN_MEMBER_FIELDS),
            "edges": (edges, EDGE_FIELDS),
            "required_groups": (required_assembly_groups, RUN_FIELDS),
            "required_members": (required_assembly_members, RUN_MEMBER_FIELDS),
            "required_edges": (required_assembly_edges, EDGE_FIELDS),
            "edge_review_queue": (edge_review_queue, EDGE_REVIEW_FIELDS),
            "run_review_queue": (run_review_queue, RUN_REVIEW_FIELDS),
            "missing_gaps": (missing_gap_candidates, ["candidate_type", *EDGE_FIELDS]),
            "collections": (collection_groups, COLLECTION_FIELDS),
            "overassociation": (overassociation, OVERASSOC_FIELDS),
        },
    )

    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
