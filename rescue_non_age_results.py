"""Rescue age-relevant tables from the OCR non-age bucket without rereading images.

Inputs:
  - age_recall_audit/ocr_non_age_candidates.csv from audit_age_recall.py
  - raw OCR results root (for loading the existing *_oneshot.json payloads)

Outputs:
  - rescue_summary.json
  - rescue_all.csv
  - core_rescue.csv
  - tail_rescue.csv
  - still_non_age.csv
  - core_rescue_results.txt / core_rescue_images.txt
  - tail_rescue_results.txt / tail_rescue_images.txt

The script uses existing OCR JSON structure plus TOC-derived candidate metadata to
reinterpret tables under a broader age-structured ontology. It does not OCR again.
"""

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

from export_inference_dataset import (
    canonical_civil_condition,
    canonical_religion,
    canonical_sex,
    normalize_text,
    normalize_token,
    parse_age_label,
)


PRIMARY_TOPICS = {
    "age_sex_population",
    "age_sex_civil_condition",
    "age_sex_marital_status",
}

TAIL_TOPICS = {
    "age_distribution",
    "age_by_religion",
    "age_civil_condition",
    "age_literacy_education",
}

CORE_TITLE_PATTERNS = [
    re.compile(r"\b(?:imperial\s+)?table\s+(?:vii|vil|vi1|v1i|7)\b", re.I),
    re.compile(r"\bage[-\s]*sex[-\s]*and[-\s]*civil[-\s]*condition\b", re.I),
    re.compile(r"\bage[-\s]*sex[-\s]*and[-\s]*marital", re.I),
    re.compile(r"\bpopulation\b.*\bage\b.*\bsex\b", re.I),
    re.compile(r"\bage\b.*\bsex\b.*\bpopulation\b", re.I),
]

STRONG_CORE_TITLE_PATTERNS = [
    re.compile(r"\bage[\W_]*sex[\W_]*and[\W_]*civil[\W_]*condition\b", re.I),
    re.compile(r"\bage[\W_]*sex[\W_]*and[\W_]*marital", re.I),
    re.compile(r"\bdistribution\s+of\s+the\s+population\s+by\s+(?:civil\s+condition[\W_]*)?age(?:[\W_]*|\s+and\s+)sex\b", re.I),
    re.compile(r"\bpopulation\b.*\bage\b.*\bsex\b", re.I),
    re.compile(r"\bage\b.*\bsex\b.*\bpopulation\b", re.I),
    re.compile(r"\bage\s+returns?\b", re.I),
]

TAIL_TITLE_PATTERNS = [
    re.compile(r"\bage[-\s]*distribution\b", re.I),
    re.compile(r"\bage\b.*\breligion\b", re.I),
    re.compile(r"\bcivil\s+condition\b.*\bage\b", re.I),
    re.compile(r"\bage\b.*\bcivil\s+condition\b", re.I),
    re.compile(r"\bliteracy\b.*\bage\b", re.I),
    re.compile(r"\beducation\b.*\bage\b", re.I),
    re.compile(r"\boccupation\b.*\bage\b", re.I),
    re.compile(r"\bage\b.*\boccupation\b", re.I),
    re.compile(r"\blanguage\b.*\bage\b", re.I),
    re.compile(r"\bage\b.*\blanguage\b", re.I),
    re.compile(r"\bcaste\b.*\bage\b", re.I),
    re.compile(r"\btribe\b.*\bage\b", re.I),
    re.compile(r"\b10,?000\b", re.I),
    re.compile(r"\b1,?000\b", re.I),
    re.compile(r"\bper\s+(?:10,?000|1,?000|cent)\b", re.I),
]

NON_AGE_BLOCK_PATTERNS = [
    re.compile(r"\bbirth[-\s]*place\b", re.I),
    re.compile(r"\blanguage\b", re.I),
]

NON_CORE_PATTERNS = [
    re.compile(r"\bbirth\s+rates?\b", re.I),
    re.compile(r"\bcharacter\s+of\s+labou?r\b", re.I),
    re.compile(r"\blabou?r\b", re.I),
    re.compile(r"\blivelihood\b", re.I),
    re.compile(r"\blepers?\b", re.I),
    re.compile(r"\bchildless\b", re.I),
    re.compile(r"\boccupations?\b", re.I),
    re.compile(r"\beducation\b", re.I),
    re.compile(r"\bliteracy\b", re.I),
    re.compile(r"\bcaste\b", re.I),
    re.compile(r"\btribe\b", re.I),
    re.compile(r"\brace\b", re.I),
]

DOMAIN_PATTERNS = {
    "religion": re.compile(r"\breligion|hindu|musalm|mohammed|christian|jain|sikh|parsi|tribal|all religions\b", re.I),
    "civil_condition": re.compile(r"\bcivil\s+condition|unmarried|married|widowed|divorced\b", re.I),
    "marital_status": re.compile(r"\bmarital|unmarried|married|widowed|divorced\b", re.I),
    "sex": re.compile(r"\bsex|male|female|males|females|persons\b", re.I),
    "population": re.compile(r"\bpopulation|persons\b", re.I),
    "education": re.compile(r"\beducation|literacy|literate|illiterate\b", re.I),
    "occupation": re.compile(r"\boccupation|industry|worker|employment\b", re.I),
    "caste_class": re.compile(r"\bcaste|tribe|class|community\b", re.I),
    "language": re.compile(r"\blanguage|bilingual\b", re.I),
    "infirmity": re.compile(r"\bblind|deaf|leper|insane|infirm\b", re.I),
}


def path_keys(path_value):
    path_value = str(path_value or "").strip()
    if not path_value:
        return []
    path_obj = Path(path_value)
    keys = [path_value, path_obj.name]
    if len(path_obj.parts) >= 2:
        keys.append("/".join(path_obj.parts[-2:]))
    if len(path_obj.parts) >= 3:
        keys.append("/".join(path_obj.parts[-3:]))
    return keys


def load_result_index(results_root):
    results_root = Path(results_root).expanduser()
    index = {}
    files = sorted(results_root.rglob("*_oneshot.json"))
    for idx, path in enumerate(files, 1):
        if idx % 500 == 0:
            print(f"[results {idx}/{len(files)}] indexed")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        source_image = payload.get("source_image") or ""
        if not source_image:
            continue
        record = {
            "result_json": str(path),
            "payload": payload,
        }
        for key in path_keys(source_image):
            index[key] = record
    return index


def text_has_age_fragment(text):
    token = normalize_token(text)
    if not token:
        return False
    if parse_age_label(text).get("is_age_like"):
        return True
    age_fragment_patterns = [
        r"\b\d+\s*-\s*\d+\b",
        r"\bunder\s+\d+\b",
        r"\b\d+\s*(?:and|&)\s*(?:over|above)\b",
        r"\bnot\s*stated\b",
        r"\ball\s+ages\b",
        r"\bage\b",
    ]
    return any(re.search(pattern, token) for pattern in age_fragment_patterns)


def collect_payload_features(payload):
    meta = payload.get("metadata") or {}
    title = meta.get("title") or ""
    row_axis_type = meta.get("row_axis_type") or ""
    row_label_name = meta.get("row_label_name") or ""
    column_groups = [normalize_text(group) for group in (meta.get("column_groups") or [])]
    sections = payload.get("data") or []
    data_type = meta.get("data_type") or ""

    total_rows = 0
    age_rows = 0
    age_row_samples = []
    age_section_hits = 0
    age_colgroup_hits = 0
    age_inner_key_hits = 0
    sex_hits = 0
    civil_hits = 0
    religion_hits = 0
    occupation_hits = 0
    education_hits = 0
    caste_hits = 0
    language_hits = 0
    infirmity_hits = 0
    nonnull_numeric_cells = 0

    for section in sections:
        section_name = normalize_text(section.get("name") or "")
        if text_has_age_fragment(section_name):
            age_section_hits += 1
        for domain, pattern in DOMAIN_PATTERNS.items():
            if pattern.search(section_name):
                if domain == "sex":
                    sex_hits += 1
                elif domain == "civil_condition":
                    civil_hits += 1
                elif domain == "religion":
                    religion_hits += 1
                elif domain == "occupation":
                    occupation_hits += 1
                elif domain == "education":
                    education_hits += 1
                elif domain == "caste_class":
                    caste_hits += 1
                elif domain == "language":
                    language_hits += 1
                elif domain == "infirmity":
                    infirmity_hits += 1

        for row in section.get("rows") or []:
            row_label = ""
            total_rows += 1
            for key, value in row.items():
                if not isinstance(value, dict):
                    row_label = normalize_text(value)
                    break
            if text_has_age_fragment(row_label):
                age_rows += 1
                if row_label and len(age_row_samples) < 5:
                    age_row_samples.append(row_label)

            for domain, pattern in DOMAIN_PATTERNS.items():
                if pattern.search(row_label):
                    if domain == "sex":
                        sex_hits += 1
                    elif domain == "civil_condition":
                        civil_hits += 1
                    elif domain == "religion":
                        religion_hits += 1
                    elif domain == "occupation":
                        occupation_hits += 1
                    elif domain == "education":
                        education_hits += 1
                    elif domain == "caste_class":
                        caste_hits += 1
                    elif domain == "language":
                        language_hits += 1
                    elif domain == "infirmity":
                        infirmity_hits += 1

            for group_name, group_value in row.items():
                if not isinstance(group_value, dict):
                    continue
                if text_has_age_fragment(group_name):
                    age_colgroup_hits += 1
                if DOMAIN_PATTERNS["sex"].search(group_name):
                    sex_hits += 1
                if DOMAIN_PATTERNS["civil_condition"].search(group_name):
                    civil_hits += 1
                if DOMAIN_PATTERNS["religion"].search(group_name):
                    religion_hits += 1
                for inner_key, inner_value in group_value.items():
                    if text_has_age_fragment(inner_key):
                        age_inner_key_hits += 1
                    if inner_value not in (None, ""):
                        nonnull_numeric_cells += 1
                    if canonical_sex(inner_key):
                        sex_hits += 1
                    if canonical_civil_condition(inner_key) or canonical_civil_condition(group_name):
                        civil_hits += 1
                    if canonical_religion(inner_key) or canonical_religion(group_name):
                        religion_hits += 1

    for group_name in column_groups:
        if text_has_age_fragment(group_name):
            age_colgroup_hits += 1
        if DOMAIN_PATTERNS["sex"].search(group_name):
            sex_hits += 1
        if DOMAIN_PATTERNS["civil_condition"].search(group_name):
            civil_hits += 1
        if DOMAIN_PATTERNS["religion"].search(group_name):
            religion_hits += 1
        if DOMAIN_PATTERNS["occupation"].search(group_name):
            occupation_hits += 1
        if DOMAIN_PATTERNS["education"].search(group_name):
            education_hits += 1
        if DOMAIN_PATTERNS["caste_class"].search(group_name):
            caste_hits += 1
        if DOMAIN_PATTERNS["language"].search(group_name):
            language_hits += 1
        if DOMAIN_PATTERNS["infirmity"].search(group_name):
            infirmity_hits += 1

    title_hits = {
        "core_title": int(any(pattern.search(title) for pattern in CORE_TITLE_PATTERNS)),
        "strong_core_title": int(any(pattern.search(title) for pattern in STRONG_CORE_TITLE_PATTERNS)),
        "tail_title": int(any(pattern.search(title) for pattern in TAIL_TITLE_PATTERNS)),
        "blocked_non_age_title": int(any(pattern.search(title) for pattern in NON_AGE_BLOCK_PATTERNS)),
        "blocked_core_title": int(any(pattern.search(title) for pattern in NON_CORE_PATTERNS)),
    }

    age_evidence_score = 0
    age_evidence_score += 5 if row_axis_type == "age_rows" else 0
    age_evidence_score += 4 if normalize_token(row_label_name) == "age" else 0
    age_evidence_score += 4 if age_rows >= 3 else 0
    age_evidence_score += 3 if age_inner_key_hits >= 3 else 0
    age_evidence_score += 2 if age_colgroup_hits >= 2 else 0
    age_evidence_score += 2 if title_hits["core_title"] or title_hits["tail_title"] else 0
    age_evidence_score += 1 if age_section_hits >= 1 else 0

    return {
        "title": title,
        "row_axis_type": row_axis_type,
        "row_label_name": row_label_name,
        "column_groups": column_groups,
        "data_type": data_type,
        "total_rows": total_rows,
        "age_rows": age_rows,
        "age_row_ratio": (age_rows / total_rows) if total_rows else 0.0,
        "age_row_samples": age_row_samples,
        "age_section_hits": age_section_hits,
        "age_colgroup_hits": age_colgroup_hits,
        "age_inner_key_hits": age_inner_key_hits,
        "age_evidence_score": age_evidence_score,
        "sex_hits": sex_hits,
        "civil_hits": civil_hits,
        "religion_hits": religion_hits,
        "occupation_hits": occupation_hits,
        "education_hits": education_hits,
        "caste_hits": caste_hits,
        "language_hits": language_hits,
        "infirmity_hits": infirmity_hits,
        "nonnull_numeric_cells": nonnull_numeric_cells,
        **title_hits,
    }


def decide_rescue(candidate_row, features):
    topic = candidate_row.get("canonical_topic") or ""
    query_tags = normalize_token(candidate_row.get("query_tags") or "")
    title = features["title"]
    dir_name_match = int(candidate_row.get("dir_name_match") or 0)

    reasons = []

    structural_age_rows = (
        features["row_axis_type"] == "age_rows"
        or normalize_token(features["row_label_name"]) == "age"
        or features["age_rows"] >= 3
    )
    structural_age_columns = (
        features["age_colgroup_hits"] >= 2
        or features["age_inner_key_hits"] >= 2
    )
    structural_age_evidence = structural_age_rows or structural_age_columns
    title_only_age = bool(features["core_title"] or features["tail_title"] or topic.startswith("age_"))

    if not structural_age_evidence:
        if title_only_age:
            return "still_non_age", "other", 1, ["title_or_topic_without_structural_age_evidence"]
        return "still_non_age", "other", 0, ["weak_age_evidence"]

    if features["blocked_non_age_title"] and not topic.startswith("age_"):
        return "still_non_age", "other", 0, ["blocked_non_age_title"]
    if features["nonnull_numeric_cells"] < 6:
        return "still_non_age", "other", 0, ["too_few_numeric_cells"]

    core_score = 0
    tail_score = 0
    specialized_domain = any([
        features["occupation_hits"],
        features["education_hits"],
        features["caste_hits"],
        features["language_hits"],
        features["infirmity_hits"],
    ])
    core_blocked = bool(features["blocked_core_title"] or specialized_domain or features["data_type"] == "proportional")
    core_primary_signal = bool(features["core_title"] or (dir_name_match and topic in PRIMARY_TOPICS))
    core_structure = bool(
        structural_age_rows
        or (
            structural_age_columns
            and (features["sex_hits"] > 0 or features["civil_hits"] > 0 or DOMAIN_PATTERNS["population"].search(title))
        )
    )
    core_semantic_signal = bool(
        features["sex_hits"] > 0
        or features["civil_hits"] > 0
        or topic == "age_sex_marital_status"
        or DOMAIN_PATTERNS["population"].search(title)
    )
    opaque_title = len(normalize_token(title)) < 12
    religion_only_age = bool(
        features["religion_hits"] > 0
        and features["civil_hits"] == 0
        and features["sex_hits"] == 0
    )

    if topic in PRIMARY_TOPICS:
        core_score += 3
        reasons.append(f"primary_topic:{topic}")
    if features["core_title"]:
        core_score += 5
        reasons.append("core_title_signal")
    if dir_name_match and ("table vii" in normalize_token(title) or topic in PRIMARY_TOPICS):
        core_score += 2
        reasons.append("dir_name_primary_support")
    if features["sex_hits"] > 0:
        core_score += 1
    if features["civil_hits"] > 0:
        core_score += 2
    if "marital" in query_tags or topic == "age_sex_marital_status":
        core_score += 2
    if DOMAIN_PATTERNS["population"].search(title):
        core_score += 1
    if structural_age_rows:
        core_score += 2
        reasons.append("age_rows_structure")
    if structural_age_columns:
        core_score += 1
        reasons.append("age_columns_structure")
    if specialized_domain:
        core_score -= 6
        reasons.append("specialized_domain_penalty")
    if features["data_type"] == "proportional":
        core_score -= 4
        reasons.append("core_proportional_penalty")
    if features["blocked_core_title"]:
        core_score -= 6
        reasons.append("blocked_core_title")

    if topic in TAIL_TOPICS:
        tail_score += 4
        reasons.append(f"tail_topic:{topic}")
    if features["tail_title"]:
        tail_score += 3
        reasons.append("tail_title_signal")
    if features["religion_hits"] > 0:
        tail_score += 2
    if features["civil_hits"] > 0 and topic not in PRIMARY_TOPICS:
        tail_score += 2
    if features["occupation_hits"] > 0:
        tail_score += 3
    if features["education_hits"] > 0:
        tail_score += 3
    if features["caste_hits"] > 0:
        tail_score += 3
    if features["language_hits"] > 0:
        tail_score += 2
    if features["infirmity_hits"] > 0:
        tail_score += 1
    if features["data_type"] == "proportional":
        tail_score += 2
        reasons.append("proportional_table")
    if features["age_colgroup_hits"] >= 2 or features["age_inner_key_hits"] >= 2:
        tail_score += 2
        reasons.append("age_in_columns_or_cells")
    if structural_age_rows:
        tail_score += 2
    if structural_age_columns:
        tail_score += 1

    if core_score >= max(tail_score, 8):
        if core_blocked:
            return "still_non_age", "other", core_score, sorted(set(reasons)) + ["core_blocked"]
        if not core_primary_signal:
            return "still_non_age", "other", core_score, sorted(set(reasons)) + ["insufficient_primary_signal"]
        if not core_structure:
            return "still_non_age", "other", core_score, sorted(set(reasons)) + ["insufficient_primary_structure"]
        if not core_semantic_signal:
            return "still_non_age", "other", core_score, sorted(set(reasons)) + ["insufficient_primary_semantics"]
        if topic == "age_sex_marital_status":
            rescue_family = "age_sex_marital_status"
        elif features["civil_hits"] > 0 or topic == "age_sex_civil_condition":
            rescue_family = "age_civil_condition"
        else:
            rescue_family = "age_sex_population"
        if not features["strong_core_title"]:
            return "core_review", rescue_family, core_score, sorted(set(reasons)) + ["needs_primary_review"]
        if religion_only_age:
            return "core_review", rescue_family, core_score, sorted(set(reasons)) + ["religion_only_age_table"]
        if opaque_title and not features["core_title"]:
            return "core_review", rescue_family, core_score, sorted(set(reasons)) + ["opaque_title"]
        return "core_rescue", rescue_family, core_score, sorted(set(reasons))

    if tail_score >= 6:
        if not any([
            features["religion_hits"] > 0,
            features["civil_hits"] > 0,
            features["occupation_hits"] > 0,
            features["education_hits"] > 0,
            features["caste_hits"] > 0,
            features["language_hits"] > 0,
            features["infirmity_hits"] > 0,
            features["data_type"] == "proportional",
        ]):
            return "still_non_age", "other", tail_score, sorted(set(reasons)) + ["insufficient_tail_signal"]
        if features["occupation_hits"] > 0:
            rescue_family = "occupation_age"
        elif features["education_hits"] > 0:
            rescue_family = "education_age"
        elif features["caste_hits"] > 0:
            rescue_family = "caste_class_age"
        elif features["language_hits"] > 0:
            rescue_family = "language_age"
        elif features["infirmity_hits"] > 0:
            rescue_family = "infirmity_age"
        elif features["religion_hits"] > 0 and features["civil_hits"] > 0:
            rescue_family = "age_civil_condition_religion"
        elif features["religion_hits"] > 0:
            rescue_family = "age_by_religion"
        elif features["civil_hits"] > 0:
            rescue_family = "age_civil_condition_distribution"
        elif features["data_type"] == "proportional":
            rescue_family = "proportional_age"
        else:
            rescue_family = "age_distribution"
        return "tail_rescue", rescue_family, tail_score, sorted(set(reasons))

    return "still_non_age", "other", max(core_score, tail_score), sorted(set(reasons)) or ["insufficient_specificity"]


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_list(path, rows, key):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(f"{row[key]}\n")


def main():
    parser = argparse.ArgumentParser(description="Rescue age-relevant tables from the OCR non-age bucket")
    parser.add_argument("--candidates-csv", default="~/census_toc/age_recall_audit/ocr_non_age_candidates.csv",
                        help="CSV produced by audit_age_recall.py")
    parser.add_argument("--results-root", default="~/census_toc/ocr_results",
                        help="Raw OCR results root containing *_oneshot.json")
    parser.add_argument("--output-dir", default="~/census_toc/non_age_rescue",
                        help="Output directory")
    args = parser.parse_args()

    candidates_csv = Path(args.candidates_csv).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    result_index = load_result_index(args.results_root)
    print(f"Result index size: {len(result_index)}")

    rows = []
    with candidates_csv.open(encoding="utf-8", newline="") as fh:
        for idx, candidate_row in enumerate(csv.DictReader(fh), 1):
            if idx % 500 == 0:
                print(f"[candidates {idx}] processed")
            image_path = candidate_row.get("image_path") or ""
            record = next((result_index[key] for key in path_keys(image_path) if key in result_index), None)
            if not record:
                rows.append({
                    **candidate_row,
                    "result_json": "",
                    "rescue_bucket": "missing_result",
                    "rescue_family": "",
                    "rescue_score": "",
                    "rescue_reasons": "missing_result_json",
                })
                continue

            payload = record["payload"]
            features = collect_payload_features(payload)
            rescue_bucket, rescue_family, rescue_score, rescue_reasons = decide_rescue(candidate_row, features)
            rows.append({
                **candidate_row,
                "result_json": record["result_json"],
                "title_from_payload": features["title"],
                "row_axis_type": features["row_axis_type"],
                "row_label_name": features["row_label_name"],
                "data_type": features["data_type"],
                "age_rows": features["age_rows"],
                "total_rows": features["total_rows"],
                "age_row_ratio": round(features["age_row_ratio"], 3),
                "age_colgroup_hits": features["age_colgroup_hits"],
                "age_inner_key_hits": features["age_inner_key_hits"],
                "age_section_hits": features["age_section_hits"],
                "age_evidence_score": features["age_evidence_score"],
                "sex_hits": features["sex_hits"],
                "civil_hits": features["civil_hits"],
                "religion_hits": features["religion_hits"],
                "occupation_hits": features["occupation_hits"],
                "education_hits": features["education_hits"],
                "caste_hits": features["caste_hits"],
                "language_hits": features["language_hits"],
                "infirmity_hits": features["infirmity_hits"],
                "nonnull_numeric_cells": features["nonnull_numeric_cells"],
                "column_groups_from_payload": " | ".join(features["column_groups"]),
                "age_row_samples": " | ".join(features["age_row_samples"]),
                "rescue_bucket": rescue_bucket,
                "rescue_family": rescue_family,
                "rescue_score": rescue_score,
                "rescue_reasons": " | ".join(rescue_reasons),
            })

    rows.sort(key=lambda row: (-int(row.get("rescue_score") or 0), row.get("province") or "", row.get("year") or "", row.get("image_path") or ""))

    fieldnames = list(rows[0].keys()) if rows else []
    write_csv(output_dir / "rescue_all.csv", rows, fieldnames)

    core_rows = [row for row in rows if row["rescue_bucket"] == "core_rescue"]
    core_review_rows = [row for row in rows if row["rescue_bucket"] == "core_review"]
    tail_rows = [row for row in rows if row["rescue_bucket"] == "tail_rescue"]
    still_rows = [row for row in rows if row["rescue_bucket"] not in {"core_rescue", "core_review", "tail_rescue"}]

    write_csv(output_dir / "core_rescue.csv", core_rows, fieldnames)
    write_csv(output_dir / "core_review.csv", core_review_rows, fieldnames)
    write_csv(output_dir / "tail_rescue.csv", tail_rows, fieldnames)
    write_csv(output_dir / "still_non_age.csv", still_rows, fieldnames)
    write_list(output_dir / "core_rescue_results.txt", core_rows, "result_json")
    write_list(output_dir / "core_rescue_images.txt", core_rows, "image_path")
    write_list(output_dir / "core_review_results.txt", core_review_rows, "result_json")
    write_list(output_dir / "core_review_images.txt", core_review_rows, "image_path")
    write_list(output_dir / "tail_rescue_results.txt", tail_rows, "result_json")
    write_list(output_dir / "tail_rescue_images.txt", tail_rows, "image_path")

    summary = {
        "candidates_scanned": len(rows),
        "core_rescue": len(core_rows),
        "core_review": len(core_review_rows),
        "tail_rescue": len(tail_rows),
        "still_non_age": len(still_rows),
        "top_core_families": Counter(row["rescue_family"] for row in core_rows).most_common(),
        "top_core_review_families": Counter(row["rescue_family"] for row in core_review_rows).most_common(),
        "top_tail_families": Counter(row["rescue_family"] for row in tail_rows).most_common(),
        "top_topics_core": Counter(row["canonical_topic"] for row in core_rows).most_common(20),
        "top_topics_core_review": Counter(row["canonical_topic"] for row in core_review_rows).most_common(20),
        "top_topics_tail": Counter(row["canonical_topic"] for row in tail_rows).most_common(20),
    }
    with (output_dir / "rescue_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print("\nNon-age rescue summary")
    print("----------------------")
    print(f"Candidates scanned: {len(rows)}")
    print(f"Core rescue:        {len(core_rows)}")
    print(f"Core review:        {len(core_review_rows)}")
    print(f"Tail rescue:        {len(tail_rows)}")
    print(f"Still non-age:      {len(still_rows)}")
    print(f"\nOutputs written to: {output_dir}")


if __name__ == "__main__":
    main()
