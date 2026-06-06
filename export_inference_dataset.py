"""Build inference-oriented exports with stricter QC and richer provenance.

Outputs:
  - source_catalog.csv      one row per OCR table/result
  - source_catalog.jsonl    one row per OCR table/result with nested raw context
  - cell_facts.csv          normalized long table of numeric cells
  - cell_facts_rich.csv     denormalized numeric cells with full source provenance
  - quarantine_sources.csv  source-level QC failures / warnings
  - quarantine_rows.csv     row-level blank/degenerate rows
  - codebook.txt            dataset documentation

The exporter is intentionally stricter than export_standardized.py:
  - removes fully blank rows
  - quarantines fully blank / degenerate sources
  - flags duplicate analytical coordinates
  - recovers TOC / digitization context when available
  - preserves richer dimensions for downstream harmonization
"""

import argparse
import csv
import glob
import json
import os
import re
from collections import Counter
from pathlib import Path

from build_digitized_ocr_manifest import load_digitized_table


RELIGION_MAP = {
    "hindu": "Hindu",
    "hindus": "Hindu",
    "muslim": "Muslim",
    "muslims": "Muslim",
    "musalman": "Muslim",
    "musalmans": "Muslim",
    "muhammadan": "Muslim",
    "muhammadans": "Muslim",
    "mohammedan": "Muslim",
    "mohammedans": "Muslim",
    "christian": "Christian",
    "christians": "Christian",
    "sikh": "Sikh",
    "sikhs": "Sikh",
    "jain": "Jain",
    "jains": "Jain",
    "buddhist": "Buddhist",
    "buddhists": "Buddhist",
    "parsi": "Parsi",
    "parsis": "Parsi",
    "zoroastrian": "Parsi",
    "zoroastrians": "Parsi",
    "arya": "Arya",
    "arya samaj": "Arya",
    "brahmo": "Brahmo",
    "animist": "Animist",
    "animists": "Animist",
    "tribal": "Tribal",
    "tribals": "Tribal",
    "jewish": "Jewish",
    "jews": "Jewish",
    "others": "Others",
    "other religions": "Others",
    "minor and unspecified religions": "Others",
    "all religions": "All Religions",
}

CIVIL_CONDITION_MAP = {
    "population": "Population",
    "total": "Population",
    "total population": "Population",
    "unmarried": "Unmarried",
    "single": "Unmarried",
    "married": "Married",
    "widowed": "Widowed",
    "widowers": "Widowed",
    "widows": "Widowed",
    "divorced": "Divorced",
}

SEX_MAP = {
    "persons": "All",
    "total": "All",
    "males": "Male",
    "male": "Male",
    "females": "Female",
    "female": "Female",
}

OFF_TARGET_PATTERNS = [
    re.compile(r"\bcataract\b", re.I),
    re.compile(r"\bsuccessful operations\b", re.I),
    re.compile(r"\beye diseases?\b", re.I),
    re.compile(r"\bjail population\b", re.I),
]

AGE_TOPIC_FAMILIES = {
    "age_sex_civil_condition": "age_civil_condition",
    "age_sex_population": "age_sex_population",
    "age_sex_marital_status": "age_sex_marital_status",
    "age_distribution": "age_distribution",
    "age_by_religion": "age_by_religion",
    "age_civil_condition": "age_civil_condition_distribution",
}

SOURCE_COLUMNS = [
    "table_id",
    "source_json",
    "source_image",
    "source_image_name",
    "source_table_dir",
    "source_table_dir_name",
    "source_file",
    "manifest_classification",
    "province",
    "year",
    "region",
    "geo_name",
    "geo_level",
    "title",
    "queue_title",
    "queue_display_identifier",
    "queue_tags",
    "canonical_topic",
    "relevance_tier",
    "queue_dir_name_match",
    "table_family",
    "constraint_regime",
    "denominator_hint",
    "data_type",
    "measure_unit",
    "row_axis_type",
    "row_label_name",
    "extraction_method",
    "api_calls",
    "elapsed_seconds",
    "acceptance_decision",
    "acceptance_next_step",
    "acceptance_version",
    "acceptance_reasons",
    "trust_score",
    "quality_status",
    "effective_quality_status",
    "n_checks",
    "failed_checks",
    "n_constraint_failures",
    "constraint_failure_levels",
    "max_constraint_diff",
    "age_warning_count",
    "repair_api_calls",
    "repair_entries_count",
    "targeted_recheck_used",
    "cross_group_active",
    "cross_section_active",
    "extraction_mode",
    "n_sections",
    "distinct_sections",
    "n_rows",
    "distinct_row_labels",
    "numeric_cells",
    "zero_cells",
    "negative_cells",
    "noninteger_cells",
    "nonblank_rows",
    "blank_rows",
    "age_like_rows",
    "n_column_groups",
    "distinct_column_groups",
    "distinct_sub_columns",
    "duplicate_exact_cells",
    "duplicate_conflict_cells",
    "qc_flags",
    "keep_for_inference",
    "column_groups",
    "sample_row_labels",
    "metadata_json",
    "acceptance_signals_json",
    "constraints_json",
    "queue_row_json",
]

FACT_COLUMNS = [
    "table_id",
    "source_json",
    "source_image",
    "source_image_name",
    "source_table_dir_name",
    "source_file",
    "province",
    "year",
    "region",
    "geo_name",
    "geo_level",
    "title",
    "queue_title",
    "queue_display_identifier",
    "queue_tags",
    "canonical_topic",
    "relevance_tier",
    "table_family",
    "constraint_regime",
    "denominator_hint",
    "data_type",
    "measure_unit",
    "row_axis_type",
    "row_label_name",
    "extraction_method",
    "api_calls",
    "elapsed_seconds",
    "acceptance_decision",
    "acceptance_next_step",
    "acceptance_version",
    "acceptance_reasons",
    "trust_score",
    "quality_status",
    "effective_quality_status",
    "n_checks",
    "failed_checks",
    "n_constraint_failures",
    "constraint_failure_levels",
    "max_constraint_diff",
    "age_warning_count",
    "repair_api_calls",
    "repair_entries_count",
    "targeted_recheck_used",
    "cross_group_active",
    "cross_section_active",
    "extraction_mode",
    "source_table_dir",
    "manifest_classification",
    "queue_dir_name_match",
    "n_sections",
    "distinct_sections",
    "n_rows",
    "distinct_row_labels",
    "numeric_cells",
    "zero_cells",
    "negative_cells",
    "noninteger_cells",
    "nonblank_rows",
    "blank_rows",
    "age_like_rows",
    "n_column_groups",
    "distinct_column_groups",
    "distinct_sub_columns",
    "duplicate_exact_cells",
    "duplicate_conflict_cells",
    "qc_flags",
    "keep_for_inference",
    "column_groups",
    "sample_row_labels",
    "metadata_json",
    "acceptance_signals_json",
    "constraints_json",
    "queue_row_json",
    "section",
    "section_type",
    "row_label_raw",
    "row_label_std",
    "row_dimension",
    "row_group_value",
    "age_label_std",
    "age_lower",
    "age_upper",
    "age_open_ended",
    "age_total",
    "age_not_stated",
    "column_group_raw",
    "column_group_std",
    "column_dimension",
    "column_group_value",
    "column_age_label_std",
    "column_age_lower",
    "column_age_upper",
    "column_age_open_ended",
    "column_age_total",
    "column_age_not_stated",
    "resolved_age_axis",
    "resolved_age_label_std",
    "resolved_age_lower",
    "resolved_age_upper",
    "resolved_age_open_ended",
    "resolved_age_total",
    "resolved_age_not_stated",
    "sub_column_raw",
    "sub_column_std",
    "sex",
    "civil_condition",
    "religion",
    "value",
]

RICH_FACT_COLUMNS = SOURCE_COLUMNS + [column for column in FACT_COLUMNS if column not in SOURCE_COLUMNS]

QUARANTINE_SOURCE_COLUMNS = [
    "table_id",
    "source_json",
    "title",
    "province",
    "year",
    "table_family",
    "quality_status",
    "qc_flags",
    "n_rows",
    "numeric_cells",
    "duplicate_conflict_cells",
]

QUARANTINE_ROW_COLUMNS = [
    "table_id",
    "source_json",
    "section",
    "row_label_raw",
    "reason",
]


def normalize_text(value):
    text = str(value or "").strip()
    text = text.replace("&", " and ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_token(value):
    value = normalize_text(value).lower()
    value = value.replace("_", " ")
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return " ".join(value.split())


def coerce_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            loaded = json.loads(text)
        except Exception:
            loaded = None
        if isinstance(loaded, list):
            return loaded
        return [part.strip() for part in re.split(r"[|;,]", text) if part.strip()]
    return [str(value)]


def normalize_region(region):
    if not region:
        return ""
    region = normalize_text(region).rstrip(".")
    region = region.replace("-concld.", "").replace("-concld", "")
    region = region.replace("-concluded.", "").replace("-concluded", "")
    region = region.replace("-contd.", "").replace("-contd", "")
    region = region.replace("-continued.", "").replace("-continued", "")
    region = region.strip().rstrip(".-– ")
    return region.title() if region.isupper() else region


def normalize_col_group(name):
    return normalize_text(name).rstrip(".")


def safe_json(value):
    return json.dumps(value or {}, ensure_ascii=False, sort_keys=True)


def normalize_age_label(label):
    if label is None:
        return ""
    label = normalize_text(label).rstrip(".")
    label = label.replace("—", "-").replace("–", "-")
    low = normalize_token(label)
    if low in ("total", "all ages", "all age", "grand total", "total population"):
        return "Total"
    if low in ("age not stated", "age not returned", "not stated", "unspecified", "age unspecified"):
        return "Age not stated"
    if low == "all religions":
        return "All Religions"
    if low == "all races":
        return "All Races"
    if low.startswith("total "):
        tail = label[6:].strip()
        if tail:
            return "Total " + tail
    label = re.sub(r"\b[Yy]ears?\b", "", label).strip()
    label = re.sub(r"\s+", " ", label).strip()
    return label


def is_plausible_age_number(value):
    try:
        number = int(str(value).strip())
    except Exception:
        return False
    return 0 <= number <= 130


def parse_age_label(label):
    raw = normalize_age_label(label)
    info = {
        "age_label_std": raw,
        "age_lower": "",
        "age_upper": "",
        "age_open_ended": "",
        "age_total": "",
        "age_not_stated": "",
        "is_age_like": False,
    }
    if not raw:
        return info

    low = normalize_text(raw).lower()
    if low == "total":
        info["age_total"] = "1"
        info["is_age_like"] = True
        return info
    if low == "age not stated":
        info["age_not_stated"] = "1"
        info["is_age_like"] = True
        return info

    match = re.match(r"^total\s+(\d+)\s*(?:-|to)\s*(\d+)$", low)
    if match and is_plausible_age_number(match.group(1)) and is_plausible_age_number(match.group(2)):
        info["age_lower"] = match.group(1)
        info["age_upper"] = match.group(2)
        info["age_total"] = "1"
        info["is_age_like"] = True
        return info

    match = re.match(r"^under\s+(\d+)$", low)
    if match and is_plausible_age_number(match.group(1)):
        info["age_lower"] = "0"
        try:
            info["age_upper"] = str(max(int(match.group(1)) - 1, 0))
        except Exception:
            info["age_upper"] = ""
        info["is_age_like"] = True
        return info

    match = re.match(r"^(\d+)\s*(?:-|to)\s*(\d+)$", low)
    if match and is_plausible_age_number(match.group(1)) and is_plausible_age_number(match.group(2)):
        info["age_lower"] = match.group(1)
        info["age_upper"] = match.group(2)
        info["is_age_like"] = True
        return info

    match = re.match(r"^(\d+)\s*(?:and|&)\s*(?:over|above)$", low)
    if match and is_plausible_age_number(match.group(1)):
        info["age_lower"] = match.group(1)
        info["age_open_ended"] = "1"
        info["is_age_like"] = True
        return info

    match = re.match(r"^(\d+)\s*\+$", low)
    if match and is_plausible_age_number(match.group(1)):
        info["age_lower"] = match.group(1)
        info["age_open_ended"] = "1"
        info["is_age_like"] = True
        return info

    match = re.match(r"^(\d+)\s+and\s+over$", low)
    if match and is_plausible_age_number(match.group(1)):
        info["age_lower"] = match.group(1)
        info["age_open_ended"] = "1"
        info["is_age_like"] = True
        return info

    match = re.match(r"^(\d+)\s*$", low)
    if match and is_plausible_age_number(match.group(1)):
        info["age_lower"] = match.group(1)
        info["age_upper"] = match.group(1)
        info["is_age_like"] = True
        return info

    return info


def age_specificity(info):
    if not (info or {}).get("is_age_like"):
        return 0
    if info.get("age_lower") or info.get("age_upper") or info.get("age_open_ended"):
        return 2
    if info.get("age_total") or info.get("age_not_stated"):
        return 1
    return 1


def resolve_age_fields(row_age_info, column_age_info):
    resolved = {
        "resolved_age_axis": "",
        "resolved_age_label_std": "",
        "resolved_age_lower": "",
        "resolved_age_upper": "",
        "resolved_age_open_ended": "",
        "resolved_age_total": "",
        "resolved_age_not_stated": "",
    }
    row_score = age_specificity(row_age_info)
    column_score = age_specificity(column_age_info)
    if not row_score and not column_score:
        return resolved

    chosen = None
    if row_score and not column_score:
        resolved["resolved_age_axis"] = "row"
        chosen = row_age_info
    elif column_score and not row_score:
        resolved["resolved_age_axis"] = "column"
        chosen = column_age_info
    elif row_age_info.get("age_label_std") == column_age_info.get("age_label_std"):
        resolved["resolved_age_axis"] = "both"
        chosen = row_age_info
    elif row_score > column_score:
        resolved["resolved_age_axis"] = "row"
        chosen = row_age_info
    elif column_score > row_score:
        resolved["resolved_age_axis"] = "column"
        chosen = column_age_info
    else:
        resolved["resolved_age_axis"] = "both"

    if chosen:
        resolved["resolved_age_label_std"] = chosen.get("age_label_std") or ""
        resolved["resolved_age_lower"] = chosen.get("age_lower") or ""
        resolved["resolved_age_upper"] = chosen.get("age_upper") or ""
        resolved["resolved_age_open_ended"] = chosen.get("age_open_ended") or ""
        resolved["resolved_age_total"] = chosen.get("age_total") or ""
        resolved["resolved_age_not_stated"] = chosen.get("age_not_stated") or ""
    return resolved


def canonical_religion(text):
    token = normalize_token(text)
    if token in RELIGION_MAP:
        return RELIGION_MAP[token]
    return ""


def canonical_civil_condition(text):
    token = normalize_token(text)
    if token in CIVIL_CONDITION_MAP:
        return CIVIL_CONDITION_MAP[token]
    return ""


def canonical_sex(text):
    token = normalize_token(text)
    if token in SEX_MAP:
        return SEX_MAP[token]
    return ""


def guess_dimension(label, table_family=""):
    label_std = normalize_age_label(label)
    label_token = normalize_token(label_std)
    if not label_token:
        return "", ""

    age_info = parse_age_label(label_std)
    if age_info["is_age_like"]:
        return "age", label_std

    religion = canonical_religion(label_std)
    if religion:
        return "religion", religion

    civil = canonical_civil_condition(label_std)
    if civil:
        return "civil_condition", civil

    sex = canonical_sex(label_std)
    if sex:
        return "sex", sex

    if "caste" in label_token or "tribe" in label_token or "class" in label_token or "community" in label_token:
        return "caste_class", label_std
    if "occupation" in label_token or "industry" in label_token or "worker" in label_token:
        return "occupation", label_std
    if "literacy" in label_token or "literate" in label_token or "illiterate" in label_token or "education" in label_token:
        return "education", label_std
    if "language" in label_token or "bilingual" in label_token:
        return "language", label_std
    if any(term in label_token for term in ("blind", "deaf", "leper", "insane", "infirm")):
        return "infirmity", label_std

    if table_family == "caste_class_age":
        return "caste_class", label_std
    if table_family == "occupation_age":
        return "occupation", label_std
    if table_family in ("education_age", "education_religion"):
        return "education", label_std
    if table_family == "language_age":
        return "language", label_std
    if table_family == "infirmity_age":
        return "infirmity", label_std

    return "other", label_std


def classify_quality(payload):
    constraints = payload.get("constraints", {}) or {}
    total_checks = constraints.get("total_checks", 0) or 0
    failed = constraints.get("failed", 0) or 0
    if total_checks > 0 and failed == 0:
        return "validated_pass", total_checks, failed
    if total_checks > 0 and failed > 0:
        return "validated_fail", total_checks, failed
    return "unvalidated", total_checks, failed


def source_lookup_keys(path_value):
    path_value = str(path_value or "").strip()
    if not path_value:
        return []
    path_obj = Path(path_value)
    keys = [path_value, path_obj.name]
    if len(path_obj.parts) >= 2:
        keys.append("/".join(path_obj.parts[-2:]))
    return keys


def load_manifest_index(path):
    index = {}
    if not path:
        return index
    path_obj = Path(path).expanduser()
    if not path_obj.exists():
        return index
    with path_obj.open(encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            source_image = row.get("source_image")
            if not source_image:
                continue
            for key in source_lookup_keys(source_image):
                if key not in index:
                    index[key] = row
    return index


def build_digitized_index(root):
    index = {}
    if not root:
        return index
    root_path = Path(root).expanduser()
    if not root_path.exists():
        return index
    for table_dir in sorted(root_path.iterdir()):
        if not table_dir.is_dir():
            continue
        digitized = load_digitized_table(table_dir)
        if not digitized:
            continue
        queue_row = digitized.get("queue_row") or {}
        source_file = digitized.get("source_file") or ""
        rendered_pages = digitized.get("rendered_pages") or []
        for page in rendered_pages:
            image_path = page.get("image_path")
            if not image_path:
                continue
            context = {
                "queue_row": queue_row,
                "digitized_source_file": source_file,
            }
            for key in source_lookup_keys(image_path):
                if key not in index:
                    index[key] = context
    return index


def resolve_context(source_image, manifest_index, digitized_index):
    for key in source_lookup_keys(source_image):
        if key in manifest_index:
            row = manifest_index[key]
            return {
                "queue_row": row.get("queue_row") or {},
                "digitized_source_file": row.get("digitized_source_file") or "",
                "classification": row.get("classification") or "",
            }
    for key in source_lookup_keys(source_image):
        if key in digitized_index:
            return digitized_index[key]
    return {"queue_row": {}, "digitized_source_file": "", "classification": ""}


def infer_geo(meta, queue_row, source_json):
    province = queue_row.get("province") or ""
    year = queue_row.get("year") or meta.get("year") or ""
    region = meta.get("region") or ""

    if not province:
        source_parts = Path(source_json).parts
        if len(source_parts) >= 3:
            province = source_parts[0].replace("_", " ")
    if not year:
        try:
            year = int(Path(source_json).parts[1])
        except Exception:
            year = ""

    province = normalize_region(province)
    region = normalize_region(region)
    geo_name = region or province
    geo_level = "district" if region and normalize_token(region) != normalize_token(province) else "province"
    try:
        year = int(year)
    except Exception:
        year = ""
    return province, year, region, geo_name, geo_level


def classify_table_family(payload, context):
    meta = payload.get("metadata", {}) or {}
    title = meta.get("title") or ""
    title_token = normalize_token(title)
    col_groups = " ".join(normalize_token(cg) for cg in meta.get("column_groups", []) or [])
    queue_row = context.get("queue_row") or {}
    topic = queue_row.get("canonical_topic") or ""
    data_type = meta.get("data_type") or ""

    if topic in AGE_TOPIC_FAMILIES:
        family = AGE_TOPIC_FAMILIES[topic]
        if family == "age_civil_condition" and canonical_religion(title):
            return "age_civil_condition_religion"
        return family

    if any(pattern.search(title) for pattern in OFF_TARGET_PATTERNS):
        return "off_target"

    if "life table" in title_token:
        return "life_table"
    if "occupation" in title_token or "industri" in title_token or "worker" in title_token:
        return "occupation_age"
    if "education" in title_token or "literate" in title_token or "illiterate" in title_token or "literacy" in title_token:
        return "education_religion" if canonical_religion(title) else "education_age"
    if "language" in title_token or "bilingual" in title_token:
        return "language_age"
    if "caste" in title_token or "tribe" in title_token or "class" in title_token or "community" in title_token:
        return "caste_class_age"
    if any(term in title_token for term in ("blind", "deaf", "leper", "insane", "infirm")):
        return "infirmity_age"
    if any(term in title_token for term in ("birth", "death", "mortality", "fertility")):
        return "birth_death"
    if "single year" in title_token:
        return "single_year_age"
    if "civil condition" in title_token or any(term in col_groups for term in ("unmarried", "married", "widowed", "divorced")):
        return "age_civil_condition_religion" if canonical_religion(title) else "age_civil_condition"
    if "age" in title_token and "religion" in title_token:
        return "age_by_religion"
    if data_type == "proportional" or "per mille" in title_token or "10 000" in title_token or "100 000" in title_token:
        return "proportional_age"
    if "age" in title_token and "sex" in title_token:
        return "age_sex_population"
    return "other_structured"


def requires_age_axis(table_family):
    return table_family in {
        "age_civil_condition",
        "age_civil_condition_religion",
        "age_sex_population",
        "age_sex_marital_status",
        "age_distribution",
        "age_by_religion",
        "age_civil_condition_distribution",
        "single_year_age",
        "proportional_age",
    }


def measure_unit(meta, table_family):
    title = normalize_token(meta.get("title") or "")
    data_type = meta.get("data_type") or ""
    if data_type == "proportional" or "per cent" in title or "percent" in title:
        return "rate"
    if "10 000" in title or "10000" in title:
        return "per_10000"
    if "100 000" in title or "100000" in title:
        return "per_100000"
    return "count"


def infer_denominator_hint(meta):
    title = normalize_token(meta.get("title") or "")
    if "100000" in title or "100 000" in title:
        return "100000"
    if "10000" in title or "10 000" in title:
        return "10000"
    if "1000" in title or "1 000" in title or "per mille" in title:
        return "1000"
    if "per cent" in title or "percent" in title:
        return "100"
    return ""


def infer_constraint_regime(table_family, measure_kind, meta):
    if measure_kind in {"per_10000", "per_100000", "rate"}:
        return "proportional_distribution"
    if table_family in {
        "age_civil_condition",
        "age_civil_condition_religion",
        "age_sex_population",
        "age_sex_marital_status",
        "single_year_age",
    }:
        return "strict_additive_counts"
    if table_family in {
        "age_distribution",
        "age_by_religion",
        "age_civil_condition_distribution",
        "proportional_age",
    }:
        return "proportional_distribution"
    if table_family in {
        "education_age",
        "education_religion",
        "occupation_age",
        "language_age",
        "caste_class_age",
        "infirmity_age",
        "birth_death",
        "life_table",
    }:
        return "descriptive_nonadditive"
    if table_family == "other_structured":
        return "review_needed"
    return "descriptive_nonadditive"


def reinterpret_quality_status(quality_status, constraint_regime, payload, table_family):
    constraints = payload.get("constraints", {}) or {}
    failures = constraints.get("failures") or []

    if quality_status == "validated_pass":
        return "accepted_strict"

    if quality_status == "validated_fail":
        if constraint_regime == "proportional_distribution":
            tiny = 0
            for failure in failures:
                diff = abs(failure.get("diff", 0) or 0)
                if diff <= 1:
                    tiny += 1
            if failures and tiny == len(failures):
                return "accepted_proportional_tolerance"
            return "review_proportional_failures"
        return "rejected_constraint_failures"

    if quality_status == "unvalidated":
        if constraint_regime in {"proportional_distribution", "descriptive_nonadditive"}:
            return "accepted_contextual_unvalidated"
        if table_family == "other_structured":
            return "review_family_context"
        return "review_unvalidated_counts"

    return "review_needed"


def extract_row_label(row):
    for _, value in row.items():
        if not isinstance(value, dict):
            return str(value or "")
    return ""


def coerce_numeric_cell(value):
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return value
    text = str(value).strip()
    if text in ("..", ".", "-", "—"):
        return 0
    text = text.replace(",", "").replace(" ", "").replace("·", ".")
    if text.startswith(".") and len(text) > 1 and text[1].isdigit():
        text = "0" + text
    if text.startswith("-.") and len(text) > 2 and text[2].isdigit():
        text = "-0" + text[1:]
    if re.fullmatch(r"-?\d+", text):
        return int(text)
    if re.fullmatch(r"-?(?:\d+\.\d*|\d*\.\d+)", text):
        return float(text)
    return None


def iter_numeric_cells_from_row(row):
    row_label_key_seen = False
    for key, value in row.items():
        if not isinstance(value, dict):
            if not row_label_key_seen:
                row_label_key_seen = True
            continue
        for sub_key, sub_value in value.items():
            parsed = coerce_numeric_cell(sub_value)
            if parsed is None:
                continue
            yield key, sub_key, parsed
    for key, value in row.items():
        if isinstance(value, dict):
            continue
        if row_label_key_seen:
            row_label_key_seen = False
            continue
        parsed = coerce_numeric_cell(value)
        if parsed is None:
            continue
        yield key, "", parsed


def dedupe_facts(facts):
    deduped = []
    exact_duplicates = 0
    conflicts = 0
    seen = {}
    for fact in facts:
        key = (
            fact["section"],
            fact["row_label_std"],
            fact["column_group_std"],
            fact["sub_column_std"],
        )
        prev = seen.get(key)
        if prev is None:
            seen[key] = fact
            deduped.append(fact)
            continue
        if prev["value"] == fact["value"]:
            exact_duplicates += 1
            continue
        conflicts += 1
    return deduped, exact_duplicates, conflicts


def source_path_metadata(source_image):
    source_image = str(source_image or "").strip()
    if not source_image:
        return "", "", ""
    path = Path(source_image)
    source_image_name = path.name
    source_table_dir = str(path.parent) if str(path.parent) != "." else ""
    source_table_dir_name = path.parent.name if path.parent and str(path.parent) != "." else ""
    return source_image_name, source_table_dir, source_table_dir_name


def summarize_constraint_failures(failures):
    levels = sorted({
        str(failure.get("level") or "").strip()
        for failure in (failures or [])
        if str(failure.get("level") or "").strip()
    })
    max_diff = ""
    for failure in failures or []:
        diff = failure.get("diff")
        if diff in (None, ""):
            continue
        try:
            numeric = abs(float(diff))
        except Exception:
            continue
        if max_diff == "" or numeric > max_diff:
            max_diff = numeric
    if isinstance(max_diff, float) and max_diff.is_integer():
        max_diff = int(max_diff)
    return {
        "n_constraint_failures": len(failures or []),
        "constraint_failure_levels": ";".join(levels),
        "max_constraint_diff": max_diff,
    }


def write_codebook(path):
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("INFERENCE EXPORT CODEBOOK\n")
        fh.write("=" * 60 + "\n\n")
        fh.write("FILES\n")
        fh.write("-" * 40 + "\n")
        fh.write("source_catalog.csv      One row per OCR table/result with QC flags and provenance.\n")
        fh.write("source_catalog.jsonl    One row per OCR table/result with nested raw metadata/context.\n")
        fh.write("cell_facts.csv          One row per numeric cell for kept sources.\n")
        fh.write("cell_facts_rich.csv     One row per numeric cell with full source provenance repeated.\n")
        fh.write("quarantine_sources.csv  Source-level QC failures/warnings.\n")
        fh.write("quarantine_rows.csv     Blank/degenerate rows removed before export.\n\n")
        fh.write("KEY IDEAS\n")
        fh.write("-" * 40 + "\n")
        fh.write("- table_id is the stable source-level key for joining facts back to the catalog.\n")
        fh.write("- quality_status reflects OCR validation status before inference QC.\n")
        fh.write("- effective_quality_status reinterprets those readings under the right constraint regime.\n")
        fh.write("- qc_flags records blank, degenerate, duplicate, or off-target problems.\n")
        fh.write("- canonical_topic / relevance_tier come from TOC/digitization context when available.\n")
        fh.write("- row_dimension / column_dimension preserve axis semantics even when they are not age.\n")
        fh.write("- age_* columns describe row-axis ages; column_age_* describes column-axis ages when present.\n")
        fh.write("- resolved_age_* picks the usable age bin for the fact when age is present on exactly one axis or one axis is clearly more specific.\n")
        fh.write("- measure_unit distinguishes counts from proportional tables.\n")
        fh.write("- constraint_regime distinguishes strict additive counts from proportional or descriptive tables.\n")
        fh.write("- source_catalog.jsonl preserves raw metadata / acceptance / constraint / queue context for later reinterpretation.\n")


def load_extra_result_paths(paths):
    extra = []
    for raw_path in paths or []:
        path_obj = Path(raw_path).expanduser()
        if not path_obj.exists():
            continue
        with path_obj.open(encoding="utf-8") as fh:
            for line in fh:
                text = line.strip()
                if text and text.endswith(".json"):
                    extra.append(str(Path(text).expanduser()))
    return extra


def main():
    parser = argparse.ArgumentParser(description="Export stricter inference-ready datasets from OCR JSON.")
    parser.add_argument("--organized-root", default="organized_results",
                        help="Root directory containing organized *_oneshot.json files")
    parser.add_argument("--output-dir", default="export/inference",
                        help="Output directory")
    parser.add_argument("--manifest-jsonl", default="",
                        help="Optional organized manifest JSONL with queue_row context")
    parser.add_argument("--digitized-root", default="",
                        help="Optional digitized_tables root for queue_row/source_file lookup")
    parser.add_argument("--extra-result-json-list", nargs="*", default=[],
                        help="Optional text file(s) containing absolute *_oneshot.json paths to include in addition to --organized-root")
    parser.add_argument("--include-unvalidated", action="store_true",
                        help="Include unvalidated sources in inference exports if they pass QC")
    parser.add_argument("--include-failed", action="store_true",
                        help="Include validated-fail sources in inference exports if they pass QC")
    args = parser.parse_args()

    organized_root = Path(args.organized_root).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_index = load_manifest_index(args.manifest_jsonl)
    digitized_index = build_digitized_index(args.digitized_root)

    allowed_quality = {"validated_pass"}
    if args.include_unvalidated:
        allowed_quality.add("unvalidated")
    if args.include_failed:
        allowed_quality.add("validated_fail")

    source_paths = sorted(glob.glob(str(organized_root / "**" / "*_oneshot.json"), recursive=True))
    source_paths = sorted(set(source_paths + load_extra_result_paths(args.extra_result_json_list)))
    print(f"Scanning {len(source_paths)} organized result JSONs...")

    source_catalog_rows = []
    source_catalog_jsonl_rows = []
    quarantine_source_rows = []
    kept_source_count = 0
    kept_fact_count = 0

    facts_path = output_dir / "cell_facts.csv"
    rich_facts_path = output_dir / "cell_facts_rich.csv"
    quarantine_rows_path = output_dir / "quarantine_rows.csv"

    with facts_path.open("w", newline="", encoding="utf-8") as fact_fh, \
            rich_facts_path.open("w", newline="", encoding="utf-8") as rich_fact_fh, \
            quarantine_rows_path.open("w", newline="", encoding="utf-8") as qrow_fh:
        fact_writer = csv.DictWriter(fact_fh, fieldnames=FACT_COLUMNS)
        fact_writer.writeheader()
        rich_fact_writer = csv.DictWriter(rich_fact_fh, fieldnames=RICH_FACT_COLUMNS)
        rich_fact_writer.writeheader()
        qrow_writer = csv.DictWriter(qrow_fh, fieldnames=QUARANTINE_ROW_COLUMNS)
        qrow_writer.writeheader()

        _bad_json = 0
        for idx, json_path in enumerate(source_paths, 1):
            if idx % 250 == 0:
                print(f"[{idx}/{len(source_paths)}] processed")

            try:
                payload = json.load(open(json_path, encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as _e:
                _bad_json += 1
                if _bad_json <= 20:
                    print(f"  SKIP malformed JSON ({_bad_json}): {os.path.basename(json_path)} — {str(_e)[:60]}")
                continue
            meta = payload.get("metadata", {}) or {}
            source_image = payload.get("source_image") or ""
            source_image_name, source_table_dir, source_table_dir_name = source_path_metadata(source_image)
            try:
                source_json_rel = os.path.relpath(json_path, organized_root)
            except Exception:
                source_json_rel = str(json_path)
            if source_json_rel.startswith(".."):
                source_json_rel = str(json_path)

            context = resolve_context(source_image, manifest_index, digitized_index)
            queue_row = context.get("queue_row") or {}
            acceptance = payload.get("acceptance") or {}
            acceptance_signals = acceptance.get("signals") or {}
            constraints = payload.get("constraints") or {}
            failures = constraints.get("failures") or []

            province, year, region, geo_name, geo_level = infer_geo(meta, queue_row, source_json_rel)
            table_family = classify_table_family(payload, context)
            quality_status, n_checks, failed_checks = classify_quality(payload)
            acceptance_decision = acceptance.get("decision") or ""
            trust_score = acceptance.get("trust_score") or ""
            title = meta.get("title") or ""
            table_id = Path(source_json_rel).with_suffix("").as_posix()
            measure_kind = measure_unit(meta, table_family)
            denominator_hint = infer_denominator_hint(meta)
            constraint_regime = infer_constraint_regime(table_family, measure_kind, meta)
            effective_quality_status = reinterpret_quality_status(
                quality_status, constraint_regime, payload, table_family)

            source_facts = []
            sample_row_labels = []
            nonblank_rows = 0
            blank_rows = 0
            numeric_cells = 0
            zero_cells = 0
            negative_cells = 0
            noninteger_cells = 0
            age_like_rows = 0
            n_rows = 0
            section_names_seen = set()
            row_labels_seen = set()
            column_groups_seen = set()
            sub_columns_seen = set()

            for section in payload.get("data", []) or []:
                section_name = normalize_text(section.get("name") or "")
                if section_name:
                    section_names_seen.add(section_name)
                section_dimension, section_group_value = guess_dimension(section_name, table_family)
                if section_dimension == "religion":
                    section_type = "Religion"
                elif section_dimension:
                    section_type = section_dimension
                else:
                    section_type = "Total" if not section_name or normalize_token(section_name) == normalize_token(title) else "Other"

                for row in section.get("rows", []) or []:
                    n_rows += 1
                    row_label_raw = extract_row_label(row)
                    row_label_std = normalize_age_label(row_label_raw)
                    if row_label_std and len(sample_row_labels) < 5:
                        sample_row_labels.append(row_label_std)
                    if row_label_std:
                        row_labels_seen.add(row_label_std)

                    row_dimension, row_group_value = guess_dimension(row_label_std, table_family)
                    age_info = parse_age_label(row_label_std)
                    if age_info["is_age_like"]:
                        age_like_rows += 1

                    row_facts = []
                    for column_group_raw, sub_column_raw, value in iter_numeric_cells_from_row(row):
                        numeric_cells += 1
                        column_group_std = normalize_col_group(column_group_raw)
                        sub_column_std = normalize_col_group(sub_column_raw) if sub_column_raw else ""
                        if column_group_std:
                            column_groups_seen.add(column_group_std)
                        if sub_column_std:
                            sub_columns_seen.add(sub_column_std)
                        column_dimension, column_group_value = guess_dimension(column_group_std, table_family)
                        sex = canonical_sex(sub_column_std)
                        civil_condition = canonical_civil_condition(column_group_std)

                        if value == 0:
                            zero_cells += 1
                        if isinstance(value, (int, float)) and value < 0:
                            negative_cells += 1
                        if isinstance(value, float) and not value.is_integer():
                            noninteger_cells += 1

                        religion = ""
                        for candidate in (
                                row_group_value if row_dimension == "religion" else "",
                                column_group_value if column_dimension == "religion" else "",
                                section_group_value if section_dimension == "religion" else "",
                                canonical_religion(title),
                        ):
                            if candidate:
                                religion = candidate
                                break

                        column_age_info = parse_age_label(column_group_std)
                        resolved_age = resolve_age_fields(age_info, column_age_info)

                        row_facts.append({
                            "table_id": table_id,
                            "source_json": source_json_rel,
                            "source_image": source_image,
                            "source_image_name": source_image_name,
                            "source_table_dir_name": source_table_dir_name,
                            "source_file": context.get("digitized_source_file") or "",
                            "province": province,
                            "year": year,
                            "region": region,
                            "geo_name": geo_name,
                            "geo_level": geo_level,
                            "title": title,
                            "canonical_topic": queue_row.get("canonical_topic") or "",
                            "relevance_tier": queue_row.get("relevance_tier") or "",
                            "table_family": table_family,
                            "constraint_regime": constraint_regime,
                            "denominator_hint": denominator_hint,
                            "data_type": meta.get("data_type") or "",
                            "measure_unit": measure_kind,
                            "row_axis_type": meta.get("row_axis_type") or "",
                            "row_label_name": meta.get("row_label_name") or "",
                            "extraction_method": payload.get("extraction_method") or "",
                            "acceptance_decision": acceptance_decision,
                            "trust_score": trust_score,
                            "quality_status": quality_status,
                            "effective_quality_status": effective_quality_status,
                            "section": section_name,
                            "section_type": section_type,
                            "row_label_raw": row_label_raw,
                            "row_label_std": row_label_std,
                            "row_dimension": row_dimension,
                            "row_group_value": row_group_value,
                            "age_label_std": age_info["age_label_std"],
                            "age_lower": age_info["age_lower"],
                            "age_upper": age_info["age_upper"],
                            "age_open_ended": age_info["age_open_ended"],
                            "age_total": age_info["age_total"],
                            "age_not_stated": age_info["age_not_stated"],
                            "column_group_raw": column_group_raw,
                            "column_group_std": column_group_std,
                            "column_dimension": column_dimension,
                            "column_group_value": column_group_value,
                            "column_age_label_std": column_age_info["age_label_std"],
                            "column_age_lower": column_age_info["age_lower"],
                            "column_age_upper": column_age_info["age_upper"],
                            "column_age_open_ended": column_age_info["age_open_ended"],
                            "column_age_total": column_age_info["age_total"],
                            "column_age_not_stated": column_age_info["age_not_stated"],
                            "resolved_age_axis": resolved_age["resolved_age_axis"],
                            "resolved_age_label_std": resolved_age["resolved_age_label_std"],
                            "resolved_age_lower": resolved_age["resolved_age_lower"],
                            "resolved_age_upper": resolved_age["resolved_age_upper"],
                            "resolved_age_open_ended": resolved_age["resolved_age_open_ended"],
                            "resolved_age_total": resolved_age["resolved_age_total"],
                            "resolved_age_not_stated": resolved_age["resolved_age_not_stated"],
                            "sub_column_raw": sub_column_raw,
                            "sub_column_std": sub_column_std,
                            "sex": sex,
                            "civil_condition": civil_condition,
                            "religion": religion,
                            "value": value,
                        })

                    if not row_facts:
                        blank_rows += 1
                        qrow_writer.writerow({
                            "table_id": table_id,
                            "source_json": source_json_rel,
                            "section": section_name,
                            "row_label_raw": row_label_raw,
                            "reason": "blank_row",
                        })
                        continue

                    nonblank_rows += 1
                    source_facts.extend(row_facts)

            deduped_facts, duplicate_exact_cells, duplicate_conflict_cells = dedupe_facts(source_facts)
            failure_summary = summarize_constraint_failures(failures)

            qc_flags = []
            if quality_status not in allowed_quality:
                qc_flags.append("filtered_quality_status")
            if table_family == "off_target":
                qc_flags.append("off_target_family")
            if nonblank_rows == 0 or numeric_cells == 0:
                qc_flags.append("blank_source")
            if len(set(f["row_label_std"] for f in deduped_facts if f["row_label_std"])) <= 1 or len(deduped_facts) <= 2:
                qc_flags.append("degenerate_source")
            if duplicate_conflict_cells > 0:
                qc_flags.append("duplicate_conflict")
            if requires_age_axis(table_family):
                age_ratio = (age_like_rows / nonblank_rows) if nonblank_rows else 0.0
                if age_like_rows < 3 or age_ratio < 0.3:
                    qc_flags.append("invalid_age_axis")

            keep_for_inference = int(not any(flag in qc_flags for flag in (
                "filtered_quality_status",
                "off_target_family",
                "blank_source",
                "degenerate_source",
                "duplicate_conflict",
            )))

            source_row = {
                "table_id": table_id,
                "source_json": source_json_rel,
                "source_image": source_image,
                "source_image_name": source_image_name,
                "source_table_dir": source_table_dir,
                "source_table_dir_name": source_table_dir_name,
                "source_file": context.get("digitized_source_file") or "",
                "manifest_classification": context.get("classification") or "",
                "province": province,
                "year": year,
                "region": region,
                "geo_name": geo_name,
                "geo_level": geo_level,
                "title": title,
                "queue_title": queue_row.get("title") or "",
                "queue_display_identifier": queue_row.get("display_identifier") or "",
                "queue_tags": " | ".join(coerce_list(queue_row.get("query_tags"))),
                "canonical_topic": queue_row.get("canonical_topic") or "",
                "relevance_tier": queue_row.get("relevance_tier") or "",
                "queue_dir_name_match": int(bool(queue_row.get("dir_name_match"))),
                "table_family": table_family,
                "constraint_regime": constraint_regime,
                "denominator_hint": denominator_hint,
                "data_type": meta.get("data_type") or "",
                "measure_unit": measure_kind,
                "row_axis_type": meta.get("row_axis_type") or "",
                "row_label_name": meta.get("row_label_name") or "",
                "extraction_method": payload.get("extraction_method") or "",
                "api_calls": payload.get("api_calls") or 0,
                "elapsed_seconds": payload.get("elapsed_seconds") or "",
                "acceptance_decision": acceptance_decision,
                "acceptance_next_step": acceptance.get("next_step") or "",
                "acceptance_version": acceptance.get("version") or "",
                "acceptance_reasons": " | ".join(acceptance.get("reasons") or []),
                "trust_score": trust_score,
                "quality_status": quality_status,
                "effective_quality_status": effective_quality_status,
                "n_checks": n_checks,
                "failed_checks": failed_checks,
                "n_constraint_failures": failure_summary["n_constraint_failures"],
                "constraint_failure_levels": failure_summary["constraint_failure_levels"],
                "max_constraint_diff": failure_summary["max_constraint_diff"],
                "age_warning_count": len(payload.get("age_ordering_warnings") or []),
                "repair_api_calls": acceptance_signals.get("repair_api_calls") or 0,
                "repair_entries_count": acceptance_signals.get("repair_entries_count") or 0,
                "targeted_recheck_used": int(bool(acceptance_signals.get("targeted_recheck_used"))),
                "cross_group_active": int(bool(acceptance_signals.get("cross_group_active"))),
                "cross_section_active": int(bool(acceptance_signals.get("cross_section_active"))),
                "extraction_mode": acceptance_signals.get("extraction_mode") or "",
                "n_sections": len(payload.get("data") or []),
                "distinct_sections": len(section_names_seen),
                "n_rows": n_rows,
                "distinct_row_labels": len(row_labels_seen),
                "numeric_cells": numeric_cells,
                "zero_cells": zero_cells,
                "negative_cells": negative_cells,
                "noninteger_cells": noninteger_cells,
                "nonblank_rows": nonblank_rows,
                "blank_rows": blank_rows,
                "age_like_rows": age_like_rows,
                "n_column_groups": len(meta.get("column_groups") or []),
                "distinct_column_groups": len(column_groups_seen),
                "distinct_sub_columns": len(sub_columns_seen),
                "duplicate_exact_cells": duplicate_exact_cells,
                "duplicate_conflict_cells": duplicate_conflict_cells,
                "qc_flags": ";".join(sorted(qc_flags)),
                "keep_for_inference": keep_for_inference,
                "column_groups": " | ".join(meta.get("column_groups") or []),
                "sample_row_labels": " | ".join(sample_row_labels),
                "metadata_json": safe_json(meta),
                "acceptance_signals_json": safe_json(acceptance_signals),
                "constraints_json": safe_json(constraints),
                "queue_row_json": safe_json(queue_row),
            }
            source_catalog_rows.append(source_row)
            source_catalog_jsonl_rows.append({
                **source_row,
                "raw_metadata": meta,
                "raw_acceptance": acceptance,
                "raw_constraints": constraints,
                "raw_queue_row": queue_row,
                "age_ordering_warnings": payload.get("age_ordering_warnings") or [],
                "sample_constraint_failures": failures[:10],
            })

            if qc_flags:
                quarantine_source_rows.append({
                    "table_id": table_id,
                    "source_json": source_json_rel,
                    "title": title,
                    "province": province,
                    "year": year,
                    "table_family": table_family,
                    "quality_status": quality_status,
                    "qc_flags": ";".join(sorted(qc_flags)),
                    "n_rows": n_rows,
                    "numeric_cells": numeric_cells,
                    "duplicate_conflict_cells": duplicate_conflict_cells,
                })

            if keep_for_inference:
                kept_source_count += 1
                for fact in deduped_facts:
                    rich_fact = {**source_row, **fact}
                    fact_writer.writerow({column: rich_fact.get(column, "") for column in FACT_COLUMNS})
                    rich_fact_writer.writerow({column: rich_fact.get(column, "") for column in RICH_FACT_COLUMNS})
                    kept_fact_count += 1

    with (output_dir / "source_catalog.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=SOURCE_COLUMNS)
        writer.writeheader()
        writer.writerows(source_catalog_rows)

    with (output_dir / "source_catalog.jsonl").open("w", encoding="utf-8") as fh:
        for row in source_catalog_jsonl_rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    with (output_dir / "quarantine_sources.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=QUARANTINE_SOURCE_COLUMNS)
        writer.writeheader()
        writer.writerows(quarantine_source_rows)

    write_codebook(output_dir / "codebook.txt")

    family_counts = Counter(row["table_family"] for row in source_catalog_rows if row["keep_for_inference"])
    print(f"\nWrote outputs to {output_dir}")
    print(f"Sources scanned: {len(source_catalog_rows)}")
    print(f"Sources kept:    {kept_source_count}")
    print(f"Facts kept:      {kept_fact_count:,}")
    print("Top kept families:")
    for family, count in family_counts.most_common(10):
        print(f"  {family}: {count}")


if __name__ == "__main__":
    main()
