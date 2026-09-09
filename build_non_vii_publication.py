#!/usr/bin/env python3
"""Curate publishable non-VII OCR results without mutating source artifacts.

Scope comes from the existing non-age rescue ledgers plus frozen_nonvii.csv.
Explicit core/VII candidates are removed before any OCR JSON is opened. The
builder processes one payload at a time and writes curated copies only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
from collections import Counter, defaultdict
from copy import deepcopy
from difflib import SequenceMatcher
from pathlib import Path


HASH_RE = re.compile(r"^[0-9a-f]{40}$")
CURRENT_JSON_RE = re.compile(r"^([0-9a-f]{40})_oneshot\.json$")
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
NUMBER_FIELDS = ("persons", "males", "females")
SERIAL_ROW_AXES = {"no", "no.", "number", "serial", "serial no", "serial no.", "serial number"}
AGE_LABEL_RE = re.compile(
    r"(?:under\s+)?\d+(?:\s*(?:-|to)\s*\d+|\s*years?)?"
    r"(?:\s+and\s+(?:over|above))?\.?",
    re.I,
)
MEASURE_AXIS_RE = re.compile(
    r"\b(?:acre(?:s)?|area|book(?:s)?|enumeration book(?:s)?|house(?:s)?|"
    r"hotel(?:s)?|mile(?:s)?|room(?:s)?|saloon(?:s)?|ship(?:s)?|shop(?:s)?|"
    r"steamer(?:s)?|tavern(?:s)?|tonnage|town(?:s)?|vessel(?:s)?|village(?:s)?)\b|"
    r"\bproprietors?\s+and\s+tenants?\b",
    re.I,
)
PROTECTED_RE = re.compile(
    r"(?<![A-Za-z0-9])table(?:[\W_]*no\.?)?[\W_]*"
    r"(?:vii(?:[a-h])?|viil|vil|vll|v1i|vi1|7)\b|"
    r"(?<![A-Za-z0-9])age[\W_]*sex[\W_]*(?:and|&)[\W_]*"
    r"(?:civil[\W_]*condition|marital)",
    re.I,
)

CATEGORY_RULES = (
    ("health_mortality", "Health and mortality", (
        "death", "mortality", "infirm", "blind", "deaf", "insane", "leper",
        "disease", "birth rate", "vital statistics", "infanticide",
    )),
    ("fertility_family", "Fertility and family", (
        "fertility", "childless", "children born", "births", "family size",
    )),
    ("household_family", "Households and families", (
        "household", "family composition", "size and sex constitution of families",
        "relationship", "families by size",
    )),
    ("education_literacy", "Education and literacy", (
        "education", "literacy", "literate", "illiterate", "learning",
        "school", "scholar",
    )),
    ("civil_condition", "Marriage and civil condition", (
        "civil condition", "marital", "married", "unmarried", "widowed",
        "divorced", "wife", "wives", "husband",
    )),
    ("occupation_livelihood", "Occupation and livelihood", (
        "occupation", "livelihood", "worker", "employment", "industry",
        "labour", "labor", "profession", "raw material",
    )),
    ("trade_transport", "Trade and transport", (
        "ships and tonnage", "shipping", "port of", "imports", "exports",
        "trade", "railway", "road traffic", "transport",
    )),
    ("land_agriculture", "Land and agriculture", (
        "agriculture", "agricultural", "cultivation", "crops", "land tenure",
        "proprietors and tenants", "holdings", "irrigation",
    )),
    ("migration_birthplace", "Migration and birthplace", (
        "migration", "migrant", "immigrant", "emigrant", "birth-place",
        "birthplace", "born in", "place of birth",
    )),
    ("caste_community", "Caste, tribe, and community", (
        "caste", "tribe", "race", "community", "selected class", "social group",
    )),
    ("language", "Language", ("language", "bilingual", "mother tongue", "vernacular")),
    ("religion", "Religion", (
        "religion", "religious", "hindu", "muslim", "muhammadan", "jain",
        "sikh", "christian", "animistic", "parsi", "parsee", "brahmo",
        "arya", "sect",
    )),
    ("demographic_distribution", "Demographic distributions", (
        "age distribution", "sex ratio", "females per", "males per", "females to",
        "number of females", "age periods", "age-period", "age groups",
        "population by age", "age return", "age-return", "adjusted returns",
        "mean age", "life table", "age statistics", "proportion",
    )),
    ("population_settlement", "Population and settlement", (
        "area, houses and population", "area houses and population", "population of",
        "distribution of the people", "density", "movement of population",
        "variation in population", "towns and villages", "village", "taluq",
        "districts and states", "blocks and circles",
    )),
    ("administration_services", "Administration and public services", (
        "political agency", "licensed", "public service", "administration",
        "municipal", "police", "prison", "revenue", "taxation",
    )),
)

FACET_RULES = {
    "age": ("age", "years", "under ", "and over"),
    "sex": ("sex", "male", "female", "persons"),
    "religion": ("religion", "hindu", "muslim", "muhammadan", "jain", "sikh", "christian"),
    "civil_condition": ("civil condition", "married", "unmarried", "widowed", "divorced"),
    "education": ("education", "literacy", "literate", "illiterate", "learning"),
    "occupation": ("occupation", "livelihood", "worker", "industry"),
    "caste_community": ("caste", "tribe", "race", "community"),
    "migration": ("migration", "migrant", "immigrant", "birthplace", "birth-place", "born in"),
    "health": ("death", "mortality", "infirm", "blind", "deaf", "insane", "leper"),
    "language": ("language", "bilingual", "mother tongue"),
}


def normalize(value) -> str:
    return " ".join(str(value or "").lower().replace("—", "-").replace("–", "-").split())


def contains_phrase(text: str, phrase: str) -> bool:
    phrase = normalize(phrase)
    return bool(phrase and re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", text))


def as_int(value, default=0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def valid_census_year(value):
    year = as_int(value, -1)
    return year if 1870 <= year <= 1962 and year % 10 in {1, 2} else None


def read_csv(path: Path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        yield from csv.DictReader(handle)


def load_json(path: Path):
    try:
        with path.open(encoding="utf-8") as handle:
            return json.load(handle), None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def image_hash(row: dict) -> str:
    value = (row.get("image_name") or row.get("imghash") or "").split(".")[0]
    return value if HASH_RE.fullmatch(value) else ""


def load_scope(rescue_dir: Path, frozen_path: Path, fixes_path: Path):
    core_hashes = set()
    for name in ("core_rescue.csv", "core_review.csv"):
        for row in read_csv(rescue_dir / name):
            if image_hash(row):
                core_hashes.add(image_hash(row))

    frozen_rows = {row["imghash"]: row for row in read_csv(frozen_path)}
    fixes = defaultdict(list)
    if fixes_path.exists():
        for row in read_csv(fixes_path):
            fixes[row["imghash"]].append(row)

    scope = {}
    exclusions = []
    for name in ("tail_rescue.csv", "still_non_age.csv"):
        for row in read_csv(rescue_dir / name):
            imghash = image_hash(row)
            if not imghash or imghash in core_hashes:
                continue
            guard_text = " ".join(row.get(key) or "" for key in ("title", "table_dir_name", "query_tags"))
            if PROTECTED_RE.search(guard_text):
                exclusions.append({"imghash": imghash, "reason": "protected_metadata_guard"})
                continue
            scope[imghash] = {
                "imghash": imghash,
                "title_hint": row.get("title") or "",
                "province": row.get("province") or "",
                "year": valid_census_year(row.get("year")),
                "rescue_bucket": row.get("rescue_bucket") or "",
                "rescue_family": row.get("rescue_family") or "",
                "canonical_topic": row.get("canonical_topic") or "",
                "source_file": row.get("source_file") or "",
            }

    for imghash, row in frozen_rows.items():
        if imghash in core_hashes:
            exclusions.append({"imghash": imghash, "reason": "explicit_core_hash"})
            continue
        if PROTECTED_RE.search(row.get("title") or ""):
            exclusions.append({"imghash": imghash, "reason": "protected_frozen_title_guard"})
            continue
        scope.setdefault(imghash, {
            "imghash": imghash,
            "title_hint": row.get("title") or "",
            "province": "",
            "year": None,
            "rescue_bucket": "frozen_nonvii",
            "rescue_family": "",
            "canonical_topic": "",
            "source_file": "",
        })

    return scope, frozen_rows, fixes, exclusions


def index_files(roots: list[Path], hashes: set[str]):
    index = defaultdict(list)
    for root in roots:
        if not root.exists():
            continue
        for directory, _, filenames in os.walk(root):
            for filename in filenames:
                prefix = filename[:40].lower()
                if prefix in hashes and (len(filename) == 40 or filename[40] in "._"):
                    index[prefix].append(Path(directory) / filename)
    return index


def sections(payload: dict) -> list[dict]:
    value = payload.get("data") or payload.get("sections") or []
    return value if isinstance(value, list) else []


def row_label(row: dict) -> str:
    for _, value in row.items():
        if not isinstance(value, dict) and value not in (None, ""):
            return str(value).strip()
    return ""


def number(value):
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def table_kind(payload: dict) -> str:
    metadata = payload.get("metadata") or {}
    title = normalize(metadata.get("title"))
    dtype = normalize(metadata.get("data_type"))
    if any(term in title for term in ("per cent", "percentage", "percent", "%")):
        return "percentage"
    if any(term in title for term in ("death rate", "birth rate", "death-rate", "birth-rate", "rate per", "mortality rate")):
        return "rate"
    if re.search(r"\bratio\b", title) or any(term in title for term in ("females per", "males per", "to 1,000")):
        return "ratio"
    if dtype == "proportional":
        return "distribution"
    return "count"


def payload_stats(payload: dict):
    labels = []
    groups = []
    numeric_cells = zero_cells = non_numeric_cells = duplicate_labels = 0
    section_names = []
    for section in sections(payload):
        section_names.append(str(section.get("name") or ""))
        seen = Counter()
        for row in section.get("rows") or []:
            label = row_label(row)
            labels.append(label)
            seen[normalize(label)] += 1
            for group, values in row.items():
                if not isinstance(values, dict):
                    continue
                groups.append(str(group))
                for value in values.values():
                    if value is None:
                        continue
                    if number(value) is None:
                        non_numeric_cells += 1
                    else:
                        numeric_cells += 1
                        zero_cells += value == 0
        duplicate_labels += sum(count - 1 for label, count in seen.items() if label and count > 1)
    rows = len(labels)
    generic = sum(bool(re.fullmatch(r"row\s*\d+", normalize(label))) for label in labels)
    age_like = sum(bool(AGE_LABEL_RE.fullmatch(normalize(label))) for label in labels)
    sex_group_names = {
        "male", "males", "female", "females", "both sexes", "both sex", "persons"
    }
    sex_axis_groups = [group for group in dict.fromkeys(groups) if normalize(group).strip(".:") in sex_group_names]
    sex_axis_collision = False
    if len(sex_axis_groups) >= 2:
        for section in sections(payload):
            for row in section.get("rows") or []:
                nested = [row.get(group) for group in sex_axis_groups if isinstance(row.get(group), dict)]
                if any(sum(field in values for field in NUMBER_FIELDS) >= 2 for values in nested):
                    sex_axis_collision = True
                    break
            if sex_axis_collision:
                break
    metadata = payload.get("metadata") or {}
    row_axis = normalize(metadata.get("row_label_name")).strip(":")
    repeated_serial_axis = row_axis in SERIAL_ROW_AXES and duplicate_labels > 0

    populated_triples = degenerate_triples = categorical_triplets = 0
    number_field_groups = set()
    metric_triplets = []
    for section in sections(payload):
        for row in section.get("rows") or []:
            for group, values in row.items():
                if not isinstance(values, dict):
                    continue
                present = {field: number(values.get(field)) for field in NUMBER_FIELDS}
                populated = [value for value in present.values() if value is not None]
                if len(populated) >= 2 and MEASURE_AXIS_RE.search(str(group)):
                    metric_triplets.append(str(group))
                if len(populated) >= 2:
                    number_field_groups.add(str(group))
                if len(populated) == 3:
                    populated_triples += 1
                    if present["persons"] == present["males"] and present["females"] == 0:
                        degenerate_triples += 1
                    total = sum(populated)
                    if total in {100, 1000, 10000} and present["persons"] != present["males"] + present["females"]:
                        categorical_triplets += 1
    semantic_text = " ".join([
        str(metadata.get("title") or ""),
        str(metadata.get("row_label_name") or ""),
        *section_names,
    ])
    measure_axis_collision = bool(metric_triplets) or bool(
        populated_triples and MEASURE_AXIS_RE.search(semantic_text)
    )
    degenerate_sex_axis = populated_triples >= 12 and degenerate_triples / populated_triples >= 0.95
    sex_row_markers = {
        normalize(label).strip(" .:-") for label in labels
    }.intersection(sex_group_names)
    sex_axis_collision = sex_axis_collision or bool(sex_row_markers and number_field_groups)
    categorical_triplet_axis = (
        populated_triples >= 5 and categorical_triplets / populated_triples >= 0.8
    )
    return {
        "section_count": len(sections(payload)),
        "row_count": rows,
        "numeric_cells": numeric_cells,
        "zero_ratio": zero_cells / numeric_cells if numeric_cells else None,
        "non_numeric_cells": non_numeric_cells,
        "duplicate_labels": duplicate_labels,
        "generic_label_ratio": generic / rows if rows else None,
        "age_label_ratio": age_like / rows if rows else None,
        "column_groups": list(dict.fromkeys(groups)),
        "section_names": section_names,
        "sex_axis_collision": sex_axis_collision,
        "repeated_serial_axis": repeated_serial_axis,
        "measure_axis_collision": measure_axis_collision,
        "measure_axis_examples": list(dict.fromkeys(metric_triplets))[:5],
        "degenerate_sex_axis": degenerate_sex_axis,
        "categorical_triplet_axis": categorical_triplet_axis,
    }


def total_group_relation(payload: dict):
    groups = list((payload.get("metadata") or {}).get("column_groups") or [])
    if not groups:
        for section in sections(payload):
            for row in section.get("rows") or []:
                groups = [key for key, value in row.items() if isinstance(value, dict)]
                if groups:
                    break
            if groups:
                break
    total_candidates = [group for group in groups if normalize(group).strip(".:") in {"total", "population", "total population"}]
    if len(total_candidates) != 1 or len(groups) < 3:
        return None
    total_group = total_candidates[0]
    components = [group for group in groups if group != total_group]
    opportunities = person_matches = 0
    for section in sections(payload):
        for row in section.get("rows") or []:
            total = row.get(total_group)
            component_values = [row.get(group) for group in components]
            if not isinstance(total, dict) or not all(isinstance(value, dict) for value in component_values):
                continue
            total_persons = number(total.get("persons"))
            persons = [number(value.get("persons")) for value in component_values]
            if total_persons is None or not all(value is not None for value in persons):
                continue
            opportunities += 1
            person_matches += total_persons == sum(persons)
    if opportunities >= 3 and person_matches / opportunities >= 0.8:
        return {"total_group": total_group, "component_groups": components, "confirmation_checks": opportunities, "confirmation_matches": person_matches}
    return None


def independent_constraints(payload: dict):
    checks = Counter()
    failures = Counter()
    examples = []
    kind = table_kind(payload)
    relation = total_group_relation(payload)

    def record(level, ok, detail):
        checks[level] += 1
        if not ok:
            failures[level] += 1
            if len(examples) < 30:
                examples.append({"level": level, "detail": detail})

    for section in sections(payload):
        for row in section.get("rows") or []:
            label = row_label(row)
            for group, values in row.items():
                if not isinstance(values, dict):
                    continue
                p, m, f = (number(values.get(field)) for field in NUMBER_FIELDS)
                if kind == "count" and p is not None and m is not None and f is not None:
                    record("persons_equals_sexes", p == m + f, f"{label}/{group}: {p} != {m}+{f}")
                for field, value in values.items():
                    value = number(value)
                    if value is not None:
                        record("non_negative", value >= 0, f"{label}/{group}/{field}: {value}")

            if relation:
                total = row.get(relation["total_group"])
                components = [row.get(group) for group in relation["component_groups"]]
                if isinstance(total, dict) and all(isinstance(value, dict) for value in components):
                    for field in NUMBER_FIELDS:
                        total_value = number(total.get(field))
                        component_values = [number(value.get(field)) for value in components]
                        if total_value is not None and all(value is not None for value in component_values):
                            record("cross_group_sum", total_value == sum(component_values), f"{label}/{field}")

    additive_levels = {"persons_equals_sexes", "cross_group_sum"}
    return {
        "table_kind": kind,
        "checks": dict(checks),
        "failures": dict(failures),
        "total_checks": sum(checks.values()),
        "total_failures": sum(failures.values()),
        "additive_checks": sum(checks[level] for level in additive_levels),
        "additive_failures": sum(failures[level] for level in additive_levels),
        "failure_examples": examples,
        "cross_group_relation": relation,
    }


def repair_determined_totals(payload: dict):
    """Repair total-group sex cells only when component groups prove both values."""
    repaired = deepcopy(payload)
    relation = total_group_relation(repaired)
    repairs = []
    if not relation:
        return repaired, repairs
    total_group = relation["total_group"]
    components = relation["component_groups"]
    for section in sections(repaired):
        for row in section.get("rows") or []:
            total = row.get(total_group)
            component_values = [row.get(group) for group in components]
            if not isinstance(total, dict) or not all(isinstance(value, dict) for value in component_values):
                continue
            total_p = number(total.get("persons"))
            component_p = [number(value.get("persons")) for value in component_values]
            component_m = [number(value.get("males")) for value in component_values]
            component_f = [number(value.get("females")) for value in component_values]
            if total_p is None or not all(value is not None for value in component_p + component_m + component_f):
                continue
            sum_p, sum_m, sum_f = sum(component_p), sum(component_m), sum(component_f)
            if total_p != sum_p or sum_p != sum_m + sum_f:
                continue
            for field, expected in (("males", sum_m), ("females", sum_f)):
                old = number(total.get(field))
                if old is not None and old != expected:
                    total[field] = expected
                    repairs.append({
                        "type": "cross_group_determined_total",
                        "section": section.get("name") or "",
                        "row_label": row_label(row),
                        "group": total_group,
                        "field": field,
                        "old": old,
                        "new": expected,
                        "component_groups": components,
                    })
    return repaired, repairs


def semantic_category(title: str, stats: dict, scope_row: dict):
    signals = [title, *stats["column_groups"], *stats["section_names"], scope_row.get("rescue_family") or ""]
    if (stats.get("age_label_ratio") or 0) >= 0.5:
        signals.append("age distribution")
    text = normalize(" ".join(signals))
    for slug, label, terms in CATEGORY_RULES:
        if any(term in text for term in terms):
            facets = sorted(name for name, needles in FACET_RULES.items() if any(contains_phrase(text, needle) for needle in needles))
            return slug, label, facets
    facets = sorted(name for name, needles in FACET_RULES.items() if any(contains_phrase(text, needle) for needle in needles))
    return "other", "Other historical tables", facets


def title_similarity(left: str, right: str) -> float:
    return SequenceMatcher(None, normalize(left), normalize(right)).ratio()


def candidate_evaluation(path: Path, payload: dict, scope_row: dict, fix_dirs: set[str]):
    stats = payload_stats(payload)
    independent = independent_constraints(payload)
    metadata = payload.get("metadata") or {}
    stored = payload.get("constraints") or {}
    acceptance = payload.get("acceptance") or {}
    stored_checks = as_int(stored.get("total_checks"))
    stored_failures = as_int(stored.get("failed"))
    score = 0.0
    score += 40 if stats["row_count"] and stats["numeric_cells"] else -200
    score += min(stats["numeric_cells"] / 100, 15)
    score += 25 if metadata.get("row_label_name") else 0
    score += 20 if (stats["generic_label_ratio"] or 0) <= 0.1 else -35
    if stored_checks:
        fail_rate = stored_failures / stored_checks
        score += 65 if stored_failures == 0 else 45 if fail_rate <= 0.01 else -min(80, fail_rate * 400)
    if independent["additive_checks"]:
        score += 30 if independent["additive_failures"] == 0 else -min(60, independent["additive_failures"] * 3)
    decision = acceptance.get("decision")
    score += 20 if decision == "accepted_high_confidence" else 5 if decision == "accepted_with_risk" else -20 if decision == "rejected_constraints_failed" else 0
    score += 20 if path.parent.name in fix_dirs else 0
    score += 10 if valid_census_year(metadata.get("year")) else 0
    score += 10 * title_similarity(scope_row.get("title_hint") or "", metadata.get("title") or "")
    return {"path": path, "payload": payload, "score": round(score, 3), "stats": stats, "independent": independent}


def choose_best_json(paths: list[Path], scope_row: dict, fix_dirs: set[str]):
    candidates = []
    errors = []
    for path in paths:
        if not CURRENT_JSON_RE.fullmatch(path.name):
            continue
        payload, error = load_json(path)
        if payload:
            candidates.append(candidate_evaluation(path, payload, scope_row, fix_dirs))
        else:
            errors.append({"path": str(path), "error": error})
    if not candidates:
        return None, [], errors
    candidates.sort(key=lambda item: (item["score"], item["path"].stat().st_mtime), reverse=True)
    return candidates[0], candidates, errors


def data_fingerprint(payload: dict) -> str:
    raw = json.dumps(payload.get("data") or payload.get("sections") or [], sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(raw.encode()).hexdigest()


def publication_status(record: dict):
    issues = []
    if not record["selected_json"]:
        return "reread_required", ["missing_local_json"]
    if not record["source_image"]:
        issues.append("missing_source_image")
    if record["frozen_status"] == "unfixed":
        issues.append("frozen_columns_not_reread")
    elif record["remaining_frozen_columns"]:
        issues.append("frozen_columns_remain")
    if record["row_count"] == 0 or record["numeric_cells"] == 0:
        issues.append("empty_or_non_numeric")
    if (record["generic_label_ratio"] or 0) > 0.25:
        issues.append("generic_row_labels")
    if record["duplicate_labels"]:
        issues.append("duplicate_row_labels")
    if record["sex_axis_collision"]:
        issues.append("sex_axis_schema_collision")
    if record["repeated_serial_axis"]:
        issues.append("repeated_serial_number_axis")
    if record["measure_axis_collision"]:
        issues.append("measure_axis_schema_collision")
    if record["degenerate_sex_axis"]:
        issues.append("degenerate_sex_axis")
    if record["categorical_triplet_axis"]:
        issues.append("categorical_triplet_schema_collision")
    if record["independent_additive_failures"]:
        issues.append("additive_constraints_fail")
    if record["unresolved_stored_failures"]:
        issues.append("stored_constraints_unresolved")
    if record["category"] == "other":
        issues.append("uncategorized_domain")
    if record["generic_title"]:
        issues.append("generic_or_continuation_title")
    source_repaired = record["deterministic_repairs"] > 0 and not record["unresolved_stored_failures"]
    if str(record["acceptance_decision"]).startswith("rejected") and not source_repaired:
        issues.append("source_read_rejected")
    if record["acceptance_decision"] == "accepted_with_risk" and (record["trust_score"] or 0) < 80:
        issues.append("low_source_trust")

    hard = {
        "frozen_columns_not_reread", "frozen_columns_remain", "empty_or_non_numeric",
        "generic_row_labels", "additive_constraints_fail", "stored_constraints_unresolved",
        "sex_axis_schema_collision",
        "repeated_serial_number_axis", "measure_axis_schema_collision", "degenerate_sex_axis",
        "categorical_triplet_schema_collision",
    }
    if hard.intersection(issues):
        return "reread_required", issues
    if "missing_source_image" in issues:
        return "review_required", issues
    if "uncategorized_domain" in issues:
        return "review_required", issues
    if "generic_or_continuation_title" in issues:
        return "review_required", issues
    if {"source_read_rejected", "low_source_trust"}.intersection(issues):
        return "review_required", issues
    if record["independent_additive_checks"] >= 3:
        return "publishable_verified", issues
    if record["table_kind"] in {"rate", "ratio", "percentage", "distribution"} and record["acceptance_decision"] == "accepted_high_confidence":
        return "publishable_limited", issues + ["constraints_not_applicable_or_limited"]
    return "review_required", issues + ["insufficient_independent_validation"]


def remediation(status: str, issues: list[str]):
    issue_set = set(issues)
    if status.startswith("publishable"):
        return 0, "none"
    if "missing_source_image" in issue_set:
        return 1, "recover_source_image"
    if issue_set.intersection({
        "sex_axis_schema_collision", "measure_axis_schema_collision",
        "degenerate_sex_axis", "categorical_triplet_schema_collision",
    }):
        return 2, "reextract_semantic_axes"
    if issue_set.intersection({
        "generic_row_labels", "repeated_serial_number_axis",
        "frozen_columns_not_reread", "frozen_columns_remain",
    }):
        return 3, "targeted_structure_reread"
    if issue_set.intersection({"additive_constraints_fail", "stored_constraints_unresolved"}):
        return 4, "constraint_guided_reread"
    if "empty_or_non_numeric" in issue_set:
        return 5, "full_table_reread"
    if "uncategorized_domain" in issue_set:
        return 6, "classify_domain"
    if "generic_or_continuation_title" in issue_set:
        return 7, "restore_title_context"
    return 8, "human_spot_check"


def select_image(paths: list[Path], imghash: str):
    candidates = [path for path in paths if path.suffix.lower() in IMAGE_SUFFIXES and path.stem == imghash]
    if not candidates:
        return None
    candidates.sort(key=lambda path: ("digitized_tables_referenced" in str(path), path.suffix.lower() == ".png", path.stat().st_size), reverse=True)
    return candidates[0]


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_tidy_csv(payload: dict, path: Path):
    fields = ["section", "row_group", "row_label", "group", "persons", "males", "females", "values_json"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for section in sections(payload):
            current_row_group = ""
            section_rows = section.get("rows") or []
            for index, row in enumerate(section_rows):
                label = row_label(row)
                mappings = [values for values in row.values() if isinstance(values, dict)]
                next_label = row_label(section_rows[index + 1]) if index + 1 < len(section_rows) else ""
                starts_age_block = (
                    not AGE_LABEL_RE.fullmatch(normalize(label))
                    and bool(AGE_LABEL_RE.fullmatch(normalize(next_label)))
                )
                if starts_age_block or (
                    mappings and not any(value is not None for values in mappings for value in values.values())
                ):
                    current_row_group = label
                for group, values in row.items():
                    if not isinstance(values, dict):
                        continue
                    extra = {key: value for key, value in values.items() if key not in NUMBER_FIELDS}
                    writer.writerow({
                        "section": section.get("name") or "",
                        "row_group": current_row_group,
                        "row_label": label,
                        "group": group,
                        "persons": values.get("persons"),
                        "males": values.get("males"),
                        "females": values.get("females"),
                        "values_json": json.dumps(extra, sort_keys=True) if extra else "",
                    })


def relative_or_blank(path):
    return str(path) if path else ""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rescue-dir", type=Path, default=Path("export/local_research_bundle/non_age_rescue_v5"))
    parser.add_argument("--frozen", type=Path, default=Path("export/frozen_nonvii.csv"))
    parser.add_argument("--fixes", type=Path, default=Path("export/frozen_nonvii_fixes.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("export/non_vii_publication"))
    parser.add_argument("--limit", type=int, help="Process only the first N hashes for a smoke test")
    parser.add_argument("--catalog-only", action="store_true", help="Do not snapshot publishable JSON/CSV files")
    args = parser.parse_args()

    scope, frozen_rows, fixes, exclusions = load_scope(args.rescue_dir, args.frozen, args.fixes)
    if args.limit:
        scope = dict(list(sorted(scope.items()))[:args.limit])
    hashes = set(scope)
    roots = [
        Path("export/ocr_results_referenced/digitized_tables"),
        Path("export/digitized_tables_referenced/digitized_tables"),
        Path("export/reocr_constraint_ocr/export/reocr_flagged_renders"),
        Path("export/reocr_flagged_renders"),
    ]
    file_index = index_files(roots, hashes)

    sentinel = args.output_dir / ".non_vii_publication_output"
    if args.output_dir.exists():
        if not sentinel.exists():
            raise SystemExit(f"Refusing to replace non-builder directory: {args.output_dir}")
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True)
    sentinel.write_text("generated by build_non_vii_publication.py\n", encoding="ascii")

    catalog = []
    detail = []
    all_repairs = []
    for position, imghash in enumerate(sorted(scope), 1):
        if position % 500 == 0:
            print(f"[{position}/{len(scope)}] validated")
        scope_row = scope[imghash]
        paths = file_index.get(imghash, [])
        protected_paths = [path for path in paths if PROTECTED_RE.search(path.parent.name)]
        paths = [path for path in paths if path not in protected_paths]
        if protected_paths and not any(CURRENT_JSON_RE.fullmatch(path.name) for path in paths):
            exclusions.append({"imghash": imghash, "reason": "protected_path_guard"})
            continue
        fix_dirs = {Path(row.get("table_dir") or "").name for row in fixes.get(imghash, [])}
        selected, candidates, errors = choose_best_json(paths, scope_row, fix_dirs)
        image_path = select_image(paths, imghash)

        frozen_row = frozen_rows.get(imghash)
        best_fix = min(fixes.get(imghash, []), key=lambda row: as_int(row.get("new_frozen"), 10**9), default=None)
        if not frozen_row:
            frozen_status, remaining_frozen = "not_flagged", 0
        elif not best_fix:
            frozen_status, remaining_frozen = "unfixed", as_int(frozen_row.get("n_frozen_cols"))
        else:
            remaining_frozen = as_int(best_fix.get("new_frozen"))
            frozen_status = "corrected" if remaining_frozen == 0 else "partial"

        if selected:
            source_payload = selected["payload"]
            curated_payload, repairs = repair_determined_totals(source_payload)
            curated_stats = payload_stats(curated_payload)
            independent = independent_constraints(curated_payload)
            metadata = curated_payload.setdefault("metadata", {})
            metadata["year"] = valid_census_year(metadata.get("year")) or scope_row.get("year")
            if not metadata.get("region") and scope_row.get("province"):
                metadata["region"] = scope_row["province"]
            title = metadata.get("title") or scope_row.get("title_hint") or "Untitled table"
            if PROTECTED_RE.search(title):
                exclusions.append({"imghash": imghash, "reason": "protected_payload_title_guard"})
                continue
            category, category_label, facets = semantic_category(title, curated_stats, scope_row)
            stored = source_payload.get("constraints") or {}
            stored_failures = as_int(stored.get("failed"))
            resolved_by_repairs = min(stored_failures, len(repairs)) if all(item["type"] == "cross_group_determined_total" for item in repairs) else 0
            unresolved_stored = max(0, stored_failures - resolved_by_repairs)
            acceptance = source_payload.get("acceptance") or {}
            variant_fingerprints = Counter(data_fingerprint(candidate["payload"]) for candidate in candidates)
        else:
            curated_payload = None
            repairs = []
            curated_stats = {
                "row_count": 0, "numeric_cells": 0, "generic_label_ratio": None,
                "duplicate_labels": 0, "zero_ratio": None, "column_groups": [],
                "section_names": [], "age_label_ratio": None, "section_count": 0,
                "sex_axis_collision": False, "repeated_serial_axis": False,
                "measure_axis_collision": False, "measure_axis_examples": [],
                "degenerate_sex_axis": False, "categorical_triplet_axis": False,
            }
            independent = {"table_kind": "unknown", "additive_checks": 0, "additive_failures": 0, "total_checks": 0, "total_failures": 0, "failure_examples": [], "checks": {}, "failures": {}, "cross_group_relation": None}
            title = scope_row.get("title_hint") or "Untitled table"
            category, category_label, facets = semantic_category(title, curated_stats, scope_row)
            stored = {}
            stored_failures = unresolved_stored = 0
            acceptance = {}
            variant_fingerprints = Counter()

        record = {
            "imghash": imghash,
            "title": title,
            "category": category,
            "category_label": category_label,
            "facets": "|".join(facets),
            "province": scope_row.get("province") or "",
            "year": (curated_payload.get("metadata") or {}).get("year") if curated_payload else scope_row.get("year"),
            "rescue_bucket": scope_row.get("rescue_bucket") or "",
            "rescue_family": scope_row.get("rescue_family") or "",
            "selected_json": relative_or_blank(selected["path"] if selected else None),
            "source_image": relative_or_blank(image_path),
            "candidate_readings": len(candidates),
            "agreeing_variants": max(variant_fingerprints.values(), default=0),
            "selection_score": selected["score"] if selected else None,
            "frozen_status": frozen_status,
            "remaining_frozen_columns": remaining_frozen,
            "section_count": curated_stats["section_count"],
            "row_count": curated_stats["row_count"],
            "numeric_cells": curated_stats["numeric_cells"],
            "zero_ratio": round(curated_stats["zero_ratio"], 5) if curated_stats["zero_ratio"] is not None else None,
            "generic_label_ratio": round(curated_stats["generic_label_ratio"], 5) if curated_stats["generic_label_ratio"] is not None else None,
            "duplicate_labels": curated_stats["duplicate_labels"],
            "sex_axis_collision": int(curated_stats["sex_axis_collision"]),
            "repeated_serial_axis": int(curated_stats["repeated_serial_axis"]),
            "measure_axis_collision": int(curated_stats["measure_axis_collision"]),
            "measure_axis_examples": "|".join(curated_stats["measure_axis_examples"]),
            "degenerate_sex_axis": int(curated_stats["degenerate_sex_axis"]),
            "categorical_triplet_axis": int(curated_stats["categorical_triplet_axis"]),
            "table_kind": independent["table_kind"],
            "stored_constraint_checks": as_int(stored.get("total_checks")),
            "stored_constraint_failures": stored_failures,
            "unresolved_stored_failures": unresolved_stored,
            "independent_checks": independent["total_checks"],
            "independent_failures": independent["total_failures"],
            "independent_additive_checks": independent["additive_checks"],
            "independent_additive_failures": independent["additive_failures"],
            "deterministic_repairs": len(repairs),
            "acceptance_decision": acceptance.get("decision") or "unknown",
            "trust_score": acceptance.get("trust_score"),
            "generic_title": int(bool(re.fullmatch(
                r"(?:(?:table|statement|abstract|appendix|no\.?)[\s.:-]*)+"
                r"[ivxlcdm\d]+(?:[\s.:-]*(?:continued|concluded|contd))?\.?",
                str(title).strip(), re.I,
            ))),
        }
        status, issues = publication_status(record)
        record["site_status"] = status
        record["qa_issues"] = "|".join(issues)
        record["qa_priority"], record["recommended_action"] = remediation(status, issues)
        catalog.append(record)

        detail.append({
            **record,
            "facets": facets,
            "qa_issues": issues,
            "candidate_paths": [str(candidate["path"]) for candidate in candidates],
            "candidate_scores": [{"path": str(candidate["path"]), "score": candidate["score"]} for candidate in candidates],
            "candidate_errors": errors,
            "independent_validation": independent,
            "repairs": repairs,
        })
        for repair in repairs:
            all_repairs.append({"imghash": imghash, "source_json": record["selected_json"], **repair})

        if curated_payload and status.startswith("publishable") and not args.catalog_only:
            destination = args.output_dir / "readings" / category / imghash
            destination.mkdir(parents=True, exist_ok=True)
            curated_payload["publication"] = {
                "source_json": record["selected_json"],
                "source_image": record["source_image"],
                "site_status": status,
                "category": category,
                "facets": facets,
                "deterministic_repairs": repairs,
                "independent_validation": independent,
            }
            json_path = destination / f"{imghash}_oneshot.json"
            with json_path.open("w", encoding="utf-8") as handle:
                json.dump(curated_payload, handle, indent=2, ensure_ascii=True)
            write_tidy_csv(curated_payload, destination / f"{imghash}_oneshot.csv")

    catalog.sort(key=lambda row: (row["category_label"], str(row["year"] or ""), row["title"], row["imghash"]))
    detail.sort(key=lambda row: (row["category_label"], str(row["year"] or ""), row["title"], row["imghash"]))
    fields = list(catalog[0]) if catalog else []
    write_csv(args.output_dir / "catalog.csv", catalog, fields)
    write_csv(args.output_dir / "publishable.csv", [row for row in catalog if row["site_status"].startswith("publishable")], fields)
    queue_key = lambda row: (row["qa_priority"], row["category_label"], row["title"], row["imghash"])
    write_csv(args.output_dir / "review_queue.csv", sorted(
        (row for row in catalog if row["site_status"] == "review_required"), key=queue_key,
    ), fields)
    write_csv(args.output_dir / "reread_queue.csv", sorted(
        (row for row in catalog if row["site_status"] == "reread_required"), key=queue_key,
    ), fields)
    with (args.output_dir / "catalog.json").open("w", encoding="utf-8") as handle:
        json.dump(detail, handle, indent=2, ensure_ascii=True)
    with (args.output_dir / "repairs.jsonl").open("w", encoding="utf-8") as handle:
        for repair in all_repairs:
            handle.write(json.dumps(repair, ensure_ascii=True) + "\n")
    with (args.output_dir / "excluded_protected.jsonl").open("w", encoding="utf-8") as handle:
        for exclusion in exclusions:
            handle.write(json.dumps(exclusion) + "\n")

    site_index = [{
        "id": row["imghash"],
        "title": row["title"],
        "category": row["category"],
        "category_label": row["category_label"],
        "facets": row["facets"].split("|") if row["facets"] else [],
        "province": row["province"] or None,
        "year": row["year"],
        "validation": row["site_status"],
        "json": f"readings/{row['category']}/{row['imghash']}/{row['imghash']}_oneshot.json",
        "csv": f"readings/{row['category']}/{row['imghash']}/{row['imghash']}_oneshot.csv",
        "source_image": row["source_image"],
    } for row in catalog if row["site_status"].startswith("publishable")]
    with (args.output_dir / "site_index.json").open("w", encoding="utf-8") as handle:
        json.dump(site_index, handle, separators=(",", ":"), ensure_ascii=True)

    summary = {
        "scope_candidates": len(scope),
        "protected_exclusions": len(exclusions),
        "by_status": Counter(row["site_status"] for row in catalog),
        "by_category": Counter(row["category_label"] for row in catalog),
        "by_table_kind": Counter(row["table_kind"] for row in catalog),
        "local_json_found": sum(bool(row["selected_json"]) for row in catalog),
        "source_images_found": sum(bool(row["source_image"]) for row in catalog),
        "deterministic_repairs": len(all_repairs),
    }
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    (args.output_dir / "README.md").write_text(
        """# Non-VII Publication Dataset

This output is built only from explicit non-VII rescue/frozen ledgers. Protected
VII/core hashes and metadata matches are excluded before OCR JSON is opened.
Source OCR artifacts are immutable; publishable readings are curated copies.

- `site_index.json` is the public-site ingestion contract.
- `publishable.csv` contains verified and limited-constraint publication rows.
- `review_queue.csv` needs human confirmation.
- `reread_queue.csv` has structural, frozen-column, or arithmetic failures.
- Queue rows include `qa_priority` and `recommended_action` for staged repair.
- `repairs.jsonl` records every deterministic cell repair and its evidence.
- `catalog.json` contains full provenance and validation details.

Tidy reading CSVs retain blank-valued hierarchy headers in `row_group`, so
repeated age or category labels remain identifiable outside the source JSON.

`publishable_verified` has independent additive checks. `publishable_limited`
is reserved for rates, ratios, percentages, or distributions where additive
constraints are not semantically applicable and the OCR acceptance is high.
""",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
