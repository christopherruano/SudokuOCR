from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import xlsxwriter


ROOT = Path("export_final")
OUT = ROOT / "sendout_final"
TEAM = ROOT / "team_ready"
DATASETS = {
    "core": ROOT / "datasets" / "core_curated_v2_final",
    "tail": ROOT / "datasets" / "tail_exploratory_v2",
    "assembled": ROOT / "datasets" / "assembled_strong_v2",
}
LOCAL_IMAGE_ROOT = Path("export/digitized_tables_referenced/digitized_tables")

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

FACT_FIELDS = [
    "fact_id",
    "table_id",
    "section",
    "row_label_raw",
    "row_label_std",
    "row_dimension",
    "row_group_value",
    "column_group_raw",
    "column_group_std",
    "column_dimension",
    "column_group_value",
    "sub_column_raw",
    "sub_column_std",
    "sex",
    "civil_condition",
    "religion",
    "resolved_age_axis",
    "resolved_age_label_std",
    "resolved_age_lower",
    "resolved_age_upper",
    "resolved_age_open_ended",
    "resolved_age_total",
    "resolved_age_not_stated",
    "measure_unit",
    "value",
]

TABLE_FIELDS = [
    "table_id",
    "layer",
    "province",
    "year",
    "title",
    "table_family",
    "format_class",
    "measure_category",
    "constraint_regime",
    "denominator_hint",
    "effective_quality_status",
    "quality_tier",
    "confidence_level",
    "n_checks",
    "failed_checks",
    "qc_flags",
    "source_image_local",
    "source_json_local",
    "source_image_original",
    "source_json_original",
    "facts_rows",
    "excel_path",
]

EXCLUSION_FIELDS = [
    "table_id",
    "layer",
    "province",
    "year",
    "title",
    "reason_category",
    "reason_detail",
    "source_image_local",
    "source_json_local",
]

REVIEW_FIELDS = EXCLUSION_FIELDS + ["recommendation"]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def write_csv(path: Path, rows: Iterable[dict[str, object]], fieldnames: list[str]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
            count += 1
    return count


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def truthy(value: object) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def to_float(value: object) -> float | None:
    text = str(value or "").strip().replace(",", "")
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def to_int(value: object) -> int:
    number = to_float(value)
    if number is None or math.isnan(number):
        return 0
    return int(number)


def stable_slug(text: str, limit: int = 96) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text.strip())
    text = re.sub(r"_+", "_", text).strip("._-")
    return (text or "table")[:limit]


def table_hash(table_id: str) -> str:
    return hashlib.sha1(table_id.encode("utf-8")).hexdigest()[:12]


def localize_path(path: str) -> str:
    if not path:
        return ""
    p = Path(path)
    if p.exists():
        return str(p)
    marker = "/digitized_tables/"
    if marker in path:
        rel = path.split(marker, 1)[1]
        candidate = LOCAL_IMAGE_ROOT / rel
        if candidate.exists():
            return str(candidate)
    return ""


def source_json_local(path: str) -> str:
    if path and Path(path).exists():
        return path
    return ""


def infer_format(row: dict[str, str]) -> str:
    title = (row.get("title", "") + " " + row.get("qc_flags", "")).lower()
    family = row.get("table_family", "")
    unit = row.get("measure_unit", "").lower()
    denom = row.get("denominator_hint", "")
    mortality = truthy(row.get("mortality_topic", ""))
    if mortality:
        kind = row.get("mortality_kind", "").lower()
        if "rate" in kind or unit in {"rate", "ratio"}:
            return "mortality_rates"
        if "disease" in title or "cholera" in title or "fever" in title:
            return "disease_specific_mortality"
        return "mortality_counts" if unit == "count" else "other_mortality"
    if denom == "1000" and unit in {"rate", "ratio"}:
        if "female" in title and "1,000 male" in title:
            return "rate_or_ratio_table"
        return "distribution_of_1000"
    if denom == "10000":
        return "distribution_of_10000"
    if unit in {"rate", "ratio"}:
        return "rate_or_ratio_table"
    if family in {"age_civil_condition", "age_sex_marital_status", "age_civil_condition_religion", "age_civil_condition_distribution"}:
        return "age_civil_condition_counts"
    if family == "age_by_religion":
        return "age_by_religion_counts"
    if "district" in title or "town" in title or "village" in title:
        return "geography_or_district_counts"
    if "education" in title or "literacy" in title or "literate" in title:
        return "education_literacy_table"
    if any(word in title for word in ("occupation", "caste", "tribe", "race")):
        return "occupation_caste_tribe_race_table"
    if family == "age_sex_population":
        return "additive_age_sex_counts"
    if family == "age_distribution":
        return "additive_age_sex_counts" if unit == "count" else "rate_or_ratio_table"
    return "malformed_or_ambiguous"


def lean_fact(row: dict[str, str], fact_id: int) -> dict[str, str]:
    out = {field: row.get(field, "") for field in FACT_FIELDS}
    out["fact_id"] = f"F{fact_id:09d}"
    return out


def coord_key(row: dict[str, str], drop: set[str] | None = None) -> tuple[str, ...]:
    drop = drop or set()
    fields = [
        "section",
        "row_label_std",
        "row_dimension",
        "row_group_value",
        "column_group_std",
        "column_dimension",
        "column_group_value",
        "sub_column_std",
        "sex",
        "civil_condition",
        "religion",
        "resolved_age_axis",
        "resolved_age_label_std",
    ]
    return tuple("" if f in drop else str(row.get(f, "")).strip().lower() for f in fields)


def is_total_age(row: dict[str, str]) -> bool:
    return truthy(row.get("resolved_age_total")) or str(row.get("resolved_age_label_std", "")).strip().lower() in {
        "total",
        "all",
        "all ages",
    }


def is_child_age(row: dict[str, str]) -> bool:
    lower = to_float(row.get("resolved_age_lower"))
    upper = to_float(row.get("resolved_age_upper"))
    if lower is None and upper is None:
        return False
    if lower is not None and lower > 5:
        return False
    return upper is None or upper <= 10


def audit_table(row: dict[str, str], facts: list[dict[str, str]]) -> tuple[list[str], list[dict[str, str]]]:
    flags: list[str] = []
    audit_rows: list[dict[str, str]] = []
    fmt = row["format_class"]
    values_by_coord: dict[tuple[str, ...], set[str]] = defaultdict(set)
    nonzero = 0
    negative = 0
    nonnumeric = 0
    for fact in facts:
        number = to_float(fact.get("value"))
        if number is None:
            nonnumeric += 1
            continue
        if number != 0:
            nonzero += 1
        if number < 0:
            negative += 1
        values_by_coord[coord_key(fact)].add(str(fact.get("value", "")))

    duplicate_conflicts = sum(1 for vals in values_by_coord.values() if len(vals) > 1)
    if nonnumeric:
        flags.append("nonnumeric_values")
    if negative and fmt not in {"rate_or_ratio_table"}:
        flags.append("negative_values")
    if nonzero == 0:
        flags.append("all_zero_numeric_content")
    if duplicate_conflicts:
        flags.append("duplicate_conflicting_coordinates")

    if fmt in {
        "additive_age_sex_counts",
        "age_civil_condition_counts",
        "age_by_religion_counts",
        "mortality_counts",
        "disease_specific_mortality",
    }:
        grouped: dict[tuple[str, ...], dict[str, float]] = defaultdict(dict)
        for fact in facts:
            number = to_float(fact.get("value"))
            if number is None:
                continue
            sex = str(fact.get("sex", "")).strip().lower()
            if sex in {"persons", "person", "all", "total", "both sexes"}:
                sex = "persons"
            elif sex in {"male", "males"}:
                sex = "male"
            elif sex in {"female", "females"}:
                sex = "female"
            else:
                continue
            grouped[coord_key(fact, {"sex"})][sex] = number
        bad_sex_sums = 0
        bad_imbalance = 0
        for vals in grouped.values():
            if {"persons", "male", "female"} <= set(vals):
                if abs(vals["persons"] - vals["male"] - vals["female"]) > max(1, abs(vals["persons"]) * 0.02):
                    bad_sex_sums += 1
                if vals["persons"] >= 10000 and (vals["male"] == 0 or vals["female"] == 0):
                    bad_imbalance += 1
        if bad_sex_sums:
            flags.append(f"sex_sum_mismatch:{bad_sex_sums}")
        if bad_imbalance and not re.search(r"\b(female|women|male|men|wife|wives|husband)\b", row.get("title", "").lower()):
            flags.append(f"extreme_sex_imbalance:{bad_imbalance}")

    if fmt == "age_civil_condition_counts":
        child_artifacts = 0
        repeats: dict[tuple[str, str, str, str], set[str]] = defaultdict(set)
        for fact in facts:
            number = to_float(fact.get("value"))
            if number is None or is_total_age(fact):
                continue
            civil = str(fact.get("civil_condition", "")).strip().lower()
            if civil in {"married", "widowed", "unmarried"}:
                repeats[
                    (
                        str(fact.get("section", "")),
                        civil,
                        str(fact.get("sex", "")).strip().lower(),
                        str(fact.get("column_group_std", "")).strip().lower(),
                    )
                ].add(str(int(number)) if number.is_integer() else str(number))
            if civil in {"married", "widowed"} and is_child_age(fact) and number >= 1000:
                child_artifacts += 1
        if child_artifacts:
            flags.append(f"massive_child_married_widowed:{child_artifacts}")
        constant_suspects = sum(1 for vals in repeats.values() if len(vals) == 1)
        if constant_suspects and len({f.get("resolved_age_label_std", "") for f in facts if not is_total_age(f)}) >= 6:
            # This is intentionally conservative: only exact constants across age-specific rows are flagged.
            flags.append(f"constant_civil_condition_columns:{constant_suspects}")

    if fmt in {"distribution_of_1000", "distribution_of_10000"}:
        denom = 1000 if fmt == "distribution_of_1000" else 10000
        out_of_range = sum(
            1
            for fact in facts
            if (n := to_float(fact.get("value"))) is not None and (n < 0 or n > denom)
        )
        if out_of_range:
            flags.append(f"distribution_value_out_of_range:{out_of_range}")

    if fmt in {"mortality_rates", "disease_specific_mortality", "other_mortality"}:
        repeated_small = Counter()
        for fact in facts:
            label = str(fact.get("row_label_std", "") or fact.get("row_label_raw", "")).strip().lower()
            number = to_float(fact.get("value"))
            if label and number in {0.0, 1.0, 100.0}:
                repeated_small[number] += 1
        if repeated_small and repeated_small.most_common(1)[0][1] >= 10:
            flags.append(f"repeated_mortality_rate_like_values:{repeated_small.most_common(1)[0]}")

    for flag in flags:
        audit_rows.append(
            {
                "table_id": row["table_id"],
                "layer": row["layer"],
                "format_class": fmt,
                "severity": "exclude" if flag else "ok",
                "flag": flag,
                "detail": "Extra sendout plausibility check",
            }
        )
    if not flags:
        audit_rows.append(
            {
                "table_id": row["table_id"],
                "layer": row["layer"],
                "format_class": fmt,
                "severity": "pass",
                "flag": "",
                "detail": "No additional number-sense flags",
            }
        )
    return flags, audit_rows


def table_metadata(row: dict[str, str]) -> dict[str, str]:
    source_image = row.get("source_image", "")
    source_json = row.get("source_json", "")
    return {
        "source_image_local": localize_path(source_image),
        "source_json_local": source_json_local(source_json),
        "source_image_original": source_image,
        "source_json_original": source_json,
    }


def build_candidates(master_rows: list[dict[str, str]]) -> tuple[dict[str, dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    candidates: dict[str, dict[str, str]] = {}
    exclusions: list[dict[str, str]] = []
    review_needed: list[dict[str, str]] = []
    for row in master_rows:
        meta = table_metadata(row)
        base = {**row, **meta}
        base["format_class"] = infer_format(row)
        clean_policy = (
            truthy(row.get("in_release_scope"))
            and row.get("quality_tier") == "strict_validated"
            and row.get("failed_checks") == "0"
            and not truthy(row.get("needs_manual_review"))
            and row.get("layer") != "assembled"
        )
        if clean_policy:
            candidates[row["table_id"]] = base
            continue
        reason_category = "outside_confirmed_policy"
        detail = []
        if not truthy(row.get("in_release_scope")):
            reason_category = "out_of_scope"
            detail.append("in_release_scope != 1")
        if row.get("quality_tier") != "strict_validated":
            reason_category = "quality_not_strict"
            detail.append(f"quality_tier={row.get('quality_tier')}")
        if row.get("failed_checks") not in {"", "0"}:
            reason_category = "failed_existing_checks"
            detail.append(f"failed_checks={row.get('failed_checks')}")
        if truthy(row.get("needs_manual_review")):
            reason_category = "manual_review_unresolved"
            detail.append("needs_manual_review=1")
        if row.get("layer") == "assembled":
            reason_category = "assembled_requires_visual_review"
            detail.append("assembled layer excluded from automatic confirmation")
        target = review_needed if truthy(row.get("in_release_scope")) else exclusions
        out = {
            **base,
            "reason_category": reason_category,
            "reason_detail": "; ".join(detail) or "Not eligible for confirmed sendout set",
            "recommendation": "rerun_ocr_or_manual_review" if target is review_needed else "reference_only",
        }
        target.append(out)
    return candidates, exclusions, review_needed


def collect_candidate_facts(candidates: dict[str, dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    facts: dict[str, list[dict[str, str]]] = defaultdict(list)
    fact_id = 1
    remaining = set(candidates)
    for layer, dataset in DATASETS.items():
        path = dataset / "cell_facts_rich.csv"
        if not path.exists() or not remaining:
            continue
        layer_ids = {tid for tid, row in candidates.items() if row["layer"] == layer}
        if not layer_ids:
            continue
        with path.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                tid = row.get("table_id", "")
                if tid in layer_ids:
                    facts[tid].append(lean_fact(row, fact_id))
                    fact_id += 1
    return facts


def workbook_path(row: dict[str, str]) -> Path:
    name = f"{row['layer']}_{stable_slug(row.get('province',''))}_{row.get('year','')}_{stable_slug(row.get('table_family',''))}_{table_hash(row['table_id'])}.xlsx"
    return OUT / "excel_tables" / row["layer"] / name


def sanitize_sheet(name: str) -> str:
    return re.sub(r"[\[\]:*?/\\]", "_", name)[:31] or "Sheet"


def write_workbook(path: Path, table: dict[str, str], facts: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    workbook = xlsxwriter.Workbook(str(path), {"constant_memory": True, "strings_to_urls": False})
    header_fmt = workbook.add_format({"bold": True, "bg_color": "#E8EEF6"})
    text_fmt = workbook.add_format({"text_wrap": False})

    summary = workbook.add_worksheet(sanitize_sheet("Summary"))
    for r, key in enumerate(TABLE_FIELDS):
        summary.write(r, 0, key, header_fmt)
        summary.write(r, 1, table.get(key, ""), text_fmt)

    matrix = workbook.add_worksheet(sanitize_sheet("Matrix"))
    row_labels = sorted({(f.get("section", ""), f.get("row_label_raw", ""), f.get("row_label_std", "")) for f in facts})
    col_labels = sorted({(f.get("column_group_raw", ""), f.get("sub_column_raw", ""), f.get("sex", ""), f.get("civil_condition", ""), f.get("religion", "")) for f in facts})
    if len(row_labels) <= 2000 and len(col_labels) <= 250:
        matrix.write_row(0, 0, ["section", "row_label_raw", "row_label_std"], header_fmt)
        for c, label in enumerate(col_labels, start=3):
            matrix.write(0, c, " | ".join(x for x in label if x), header_fmt)
        row_index = {key: i + 1 for i, key in enumerate(row_labels)}
        col_index = {key: i + 3 for i, key in enumerate(col_labels)}
        for key, r in row_index.items():
            matrix.write_row(r, 0, list(key))
        for f in facts:
            r = row_index[(f.get("section", ""), f.get("row_label_raw", ""), f.get("row_label_std", ""))]
            c = col_index[(f.get("column_group_raw", ""), f.get("sub_column_raw", ""), f.get("sex", ""), f.get("civil_condition", ""), f.get("religion", ""))]
            value = to_float(f.get("value"))
            matrix.write(r, c, value if value is not None else f.get("value", ""))
    else:
        matrix.write(0, 0, "not suitable for reconstruction")
        matrix.write(1, 0, f"rows={len(row_labels)} columns={len(col_labels)}")

    facts_ws = workbook.add_worksheet(sanitize_sheet("Facts"))
    facts_ws.write_row(0, 0, FACT_FIELDS, header_fmt)
    for r, fact in enumerate(facts, start=1):
        facts_ws.write_row(r, 0, [fact.get(field, "") for field in FACT_FIELDS])

    source = workbook.add_worksheet(sanitize_sheet("Source"))
    rows = [
        ("source_image_local", table.get("source_image_local", "")),
        ("source_json_local", table.get("source_json_local", "")),
        ("source_image_original", table.get("source_image_original", "")),
        ("source_json_original", table.get("source_json_original", "")),
        ("review_notes", "No manual override used in sendout_final."),
    ]
    for r, (key, value) in enumerate(rows):
        source.write(r, 0, key, header_fmt)
        source.write(r, 1, value, text_fmt)
    workbook.close()


def rel(path: Path) -> str:
    return str(path.relative_to(OUT))


def ensure_symlink(link: Path, target: Path) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.exists() or link.is_symlink():
        link.unlink()
    link.symlink_to(os.path.relpath(target.resolve(), link.parent.resolve()))


def build_showcase(confirmed: list[dict[str, str]]) -> None:
    showcase = OUT / "showcase"
    buckets = {
        "age_civil_condition": [r for r in confirmed if r["format_class"] == "age_civil_condition_counts"][:5],
        "age_sex_population": [r for r in confirmed if r["table_family"] == "age_sex_population"][:5],
        "distribution_1000": [r for r in confirmed if r["format_class"] == "distribution_of_1000"][:5],
        "mortality": [r for r in confirmed if r["format_class"].startswith("mortality") or "mortality" in r["format_class"]][:5],
    }
    for bucket, rows in buckets.items():
        bucket_dir = showcase / bucket
        bucket_dir.mkdir(parents=True, exist_ok=True)
        for row in rows:
            stem = f"{stable_slug(row.get('province',''))}_{row.get('year','')}_{table_hash(row['table_id'])}"
            excel = OUT / row["excel_path"]
            ensure_symlink(bucket_dir / f"{stem}.xlsx", excel)
            if row.get("source_image_local"):
                src = Path(row["source_image_local"])
                if src.exists():
                    ensure_symlink(bucket_dir / f"{stem}{src.suffix}", src)
            (bucket_dir / f"{stem}.txt").write_text(
                "\n".join(
                    [
                        row.get("title", ""),
                        f"Table: {row['table_id']}",
                        f"Why trustworthy: strict_validated, failed_checks=0, no unresolved manual-review flag, and passed sendout number-sense audit as {row['format_class']}.",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )


def write_docs(counts: dict[str, object]) -> None:
    readme = f"""# SudokuOCR Sendout Final

This folder is the collaborator-ready local release. Start with `data/tables_confirmed.csv` for the one-row-per-table catalog and `data/facts_confirmed.csv` for the lean numeric cell facts.

One row in `facts_confirmed.csv` is one confirmed numeric cell from one confirmed table. Table-level context such as title, province, year, source paths, confidence, and workbook location lives in `data/tables_confirmed.csv` so the fact table stays compact and analytical.

Excel workbooks are under `excel_tables/`. The layer folders (`core/`, `tail/`) contain the physical workbooks; `excel_tables/by_family/` and `excel_tables/mortality/` contain symlinks for browsing.

Confidence levels come from the team-ready quality layer. This sendout includes only strict validated, in-scope, zero-failure, non-assembled tables with no unresolved manual-review signal, then removes tables that fail extra format-aware number-sense checks.

Mortality is separated under `mortality/`, with its own confirmed table/fact extracts and mortality-focused index.

Excluded and review-needed tables are documented in `data/exclusions.csv`, `data/review_needed.csv`, `excluded/`, and `review/`. Review-needed rows are not in `facts_confirmed.csv`.

Counts:
- confirmed tables: {counts['confirmed_tables']}
- confirmed facts: {counts['confirmed_facts']}
- mortality tables: {counts['mortality_tables']}
- excluded tables: {counts['excluded_tables']}
- review-needed tables: {counts['review_needed_tables']}
"""
    (OUT / "README.md").write_text(readme, encoding="utf-8")

    dictionary_sections = {
        "facts_confirmed.csv": FACT_FIELDS,
        "tables_confirmed.csv": TABLE_FIELDS,
        "table_members.csv": ["table_id", "member_index", "member_source_image", "member_source_json", "member_title", "member_page_status"],
        "exclusions.csv": EXCLUSION_FIELDS,
        "review_needed.csv": REVIEW_FIELDS,
    }
    definitions = {
        "fact_id": "Stable row identifier assigned in this sendout build.",
        "table_id": "Stable source table identifier.",
        "layer": "Source dataset layer: core, tail, or assembled.",
        "format_class": "Sendout-inferred table format used for plausibility checks.",
        "excel_path": "Path to the per-table workbook relative to sendout_final.",
        "reason_category": "Machine-readable reason the table is not confirmed.",
        "reason_detail": "Human-readable detail for the exclusion or review decision.",
    }
    dd_lines = ["# Data Dictionary", ""]
    for filename, fields in dictionary_sections.items():
        dd_lines.extend([f"## {filename}", ""])
        for field in fields:
            dd_lines.append(f"- `{field}`: {definitions.get(field, field.replace('_', ' '))}.")
        dd_lines.append("")
    (OUT / "DATA_DICTIONARY.md").write_text("\n".join(dd_lines), encoding="utf-8")

    notes = ["# Release Notes", ""]
    notes.append(f"- Confirmed tables: {counts['confirmed_tables']}")
    notes.append(f"- Confirmed facts: {counts['confirmed_facts']}")
    notes.append(f"- Mortality confirmed tables: {counts['mortality_tables']}")
    notes.append(f"- Excluded tables: {counts['excluded_tables']}")
    notes.append(f"- Review-needed tables: {counts['review_needed_tables']}")
    notes.append("")
    notes.append("## Included By Layer")
    for key, value in counts["by_layer"].items():
        notes.append(f"- {key}: {value}")
    notes.append("")
    notes.append("## Included By Family")
    for key, value in counts["by_family"].items():
        notes.append(f"- {key}: {value}")
    notes.append("")
    notes.append("## Caveats")
    notes.append("- No assembled tables are automatically included; they remain review-needed unless separately visually confirmed.")
    notes.append("- Manual image review notes are empty because this sendout avoids manual overrides.")
    notes.append("- Source JSON local paths are populated only when the original JSON exists locally.")
    (OUT / "RELEASE_NOTES.md").write_text("\n".join(notes) + "\n", encoding="utf-8")


def file_integrity(paths: list[Path]) -> list[dict[str, str]]:
    rows = []
    for path in paths:
        if not path.exists() and not path.is_symlink():
            rows.append({"path": rel(path), "exists": "0", "bytes": "", "sha256": ""})
            continue
        digest = hashlib.sha256()
        size = 0
        target = path
        if path.is_symlink():
            target = path.resolve()
        if target.is_file():
            with target.open("rb") as fh:
                for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                    size += len(chunk)
                    digest.update(chunk)
            sha = digest.hexdigest()
        else:
            sha = ""
        rows.append({"path": rel(path), "exists": "1", "bytes": str(size), "sha256": sha})
    return rows


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    for name in ["catalogs", "data", "excel_tables/core", "excel_tables/tail", "excel_tables/mortality", "mortality", "review", "excluded", "qa", "showcase"]:
        (OUT / name).mkdir(parents=True, exist_ok=True)

    master_rows = read_csv(TEAM / "catalogs" / "all_tables_master.csv")
    candidates, exclusions, review_needed = build_candidates(master_rows)
    candidate_facts = collect_candidate_facts(candidates)

    confirmed: list[dict[str, str]] = []
    audit_rows: list[dict[str, str]] = []
    for tid, row in sorted(candidates.items(), key=lambda kv: (kv[1]["layer"], kv[1]["province"], kv[1]["year"], kv[1]["title"])):
        facts = candidate_facts.get(tid, [])
        flags, rows = audit_table(row, facts)
        audit_rows.extend(rows)
        if flags:
            review_needed.append(
                {
                    **row,
                    "reason_category": "sendout_plausibility_flag",
                    "reason_detail": "; ".join(flags),
                    "recommendation": "manual_image_review_before_inclusion",
                }
            )
            continue
        row["facts_rows"] = str(len(facts))
        confirmed.append(row)

    fact_count = 0
    with (OUT / "data" / "facts_confirmed.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FACT_FIELDS)
        writer.writeheader()
        for table in confirmed:
            for fact in candidate_facts.get(table["table_id"], []):
                writer.writerow(fact)
                fact_count += 1

    excel_index = []
    for table in confirmed:
        path = workbook_path(table)
        table["excel_path"] = rel(path)
        write_workbook(path, table, candidate_facts.get(table["table_id"], []))
        excel_index.append({"table_id": table["table_id"], "excel_path": table["excel_path"], "layer": table["layer"], "table_family": table["table_family"], "format_class": table["format_class"]})
        family_link = OUT / "excel_tables" / "by_family" / stable_slug(table["table_family"]) / path.name
        ensure_symlink(family_link, path)
        if table["format_class"].startswith("mortality") or "mortality" in table["format_class"]:
            ensure_symlink(OUT / "excel_tables" / "mortality" / path.name, path)

    table_rows = [{field: row.get(field, "") for field in TABLE_FIELDS} for row in confirmed]
    write_csv(OUT / "data" / "tables_confirmed.csv", table_rows, TABLE_FIELDS)
    write_csv(OUT / "catalogs" / "excel_index.csv", excel_index, ["table_id", "excel_path", "layer", "table_family", "format_class"])
    write_csv(OUT / "data" / "exclusions.csv", exclusions, EXCLUSION_FIELDS)
    write_csv(OUT / "data" / "review_needed.csv", review_needed, REVIEW_FIELDS)

    members = []
    for row in read_csv(DATASETS["assembled"] / "member_pages.csv"):
        if row.get("table_id") in {r["table_id"] for r in confirmed}:
            members.append(
                {
                    "table_id": row.get("table_id", ""),
                    "member_index": row.get("member_index", ""),
                    "member_source_image": localize_path(row.get("source_image", "")) or row.get("source_image", ""),
                    "member_source_json": source_json_local(row.get("source_json", "")) or row.get("source_json", ""),
                    "member_title": row.get("title", ""),
                    "member_page_status": row.get("effective_quality_status", "") or row.get("quality_status", ""),
                }
            )
    write_csv(OUT / "data" / "table_members.csv", members, ["table_id", "member_index", "member_source_image", "member_source_json", "member_title", "member_page_status"])

    mortality_tables = [r for r in table_rows if r["format_class"].startswith("mortality") or "mortality" in r["format_class"]]
    mortality_ids = {r["table_id"] for r in mortality_tables}
    write_csv(OUT / "mortality" / "tables_mortality_confirmed.csv", mortality_tables, TABLE_FIELDS)
    write_csv(
        OUT / "mortality" / "facts_mortality_confirmed.csv",
        (fact for tid in mortality_ids for fact in candidate_facts.get(tid, [])),
        FACT_FIELDS,
    )
    mortality_splits = {
        "death_counts": {"mortality_counts"},
        "death_rates": {"mortality_rates"},
        "disease_specific": {"disease_specific_mortality"},
        "other_mortality": {"other_mortality"},
    }
    for kind, classes in mortality_splits.items():
        rows = [r for r in mortality_tables if r["format_class"] in classes]
        write_csv(OUT / "mortality" / f"{kind}.csv", rows, TABLE_FIELDS)

    write_csv(OUT / "qa" / "number_sense_audit.csv", audit_rows, ["table_id", "layer", "format_class", "severity", "flag", "detail"])
    write_json(OUT / "qa" / "number_sense_summary.json", dict(Counter(r["flag"] or "pass" for r in audit_rows)))
    write_json(
        OUT / "qa" / "schema_audit.json",
        {
            "facts_confirmed_fields": FACT_FIELDS,
            "tables_confirmed_fields": TABLE_FIELDS,
            "forbidden_fact_metadata_absent": all(field not in FACT_FIELDS for field in ["title", "source_file", "source_json", "source_image", "metadata_json", "constraints_json", "queue_row_json", "acceptance_signals_json"]),
            "tables_without_workbooks": [r["table_id"] for r in table_rows if not (OUT / r["excel_path"]).exists()],
        },
    )
    write_csv(OUT / "qa" / "sample_rows_preview.csv", (fact for tid in list({r["table_id"] for r in confirmed})[:25] for fact in candidate_facts.get(tid, [])[:4]), FACT_FIELDS)
    write_csv(OUT / "qa" / "manual_image_review_notes.csv", [], ["table_id", "inspected_image_path", "inspected_json_path", "issue_seen", "decision", "notes"])

    counts = {
        "confirmed_tables": len(table_rows),
        "confirmed_facts": fact_count,
        "mortality_tables": len(mortality_tables),
        "excluded_tables": len(exclusions),
        "review_needed_tables": len(review_needed),
        "by_layer": dict(sorted(Counter(r["layer"] for r in table_rows).items())),
        "by_family": dict(sorted(Counter(r["table_family"] for r in table_rows).items())),
        "by_year": dict(sorted(Counter(r["year"] for r in table_rows).items())),
        "by_province": dict(sorted(Counter(r["province"] for r in table_rows).items())),
        "excluded_by_reason": dict(sorted(Counter(r["reason_category"] for r in exclusions).items())),
        "review_needed_by_reason": dict(sorted(Counter(r["reason_category"] for r in review_needed).items())),
    }
    write_json(OUT / "qa" / "final_counts.json", counts)
    md = ["# Final Counts", ""]
    for key in ["confirmed_tables", "confirmed_facts", "mortality_tables", "excluded_tables", "review_needed_tables"]:
        md.append(f"- {key}: {counts[key]}")
    (OUT / "qa" / "final_counts.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    build_showcase(table_rows)
    write_docs(counts)
    (OUT / "review" / "README.md").write_text(
        "Review-needed table metadata is in `../data/review_needed.csv`. These tables are not present in `facts_confirmed.csv`.\n",
        encoding="utf-8",
    )
    (OUT / "excluded" / "README.md").write_text(
        "Excluded table metadata is in `../data/exclusions.csv`. These tables are outside the confirmed sendout set.\n",
        encoding="utf-8",
    )

    integrity_paths = [
        OUT / "README.md",
        OUT / "DATA_DICTIONARY.md",
        OUT / "RELEASE_NOTES.md",
        OUT / "data" / "facts_confirmed.csv",
        OUT / "data" / "tables_confirmed.csv",
        OUT / "data" / "table_members.csv",
        OUT / "data" / "exclusions.csv",
        OUT / "data" / "review_needed.csv",
        OUT / "catalogs" / "excel_index.csv",
        OUT / "qa" / "final_counts.json",
        OUT / "qa" / "number_sense_audit.csv",
        OUT / "qa" / "schema_audit.json",
    ]
    write_csv(OUT / "qa" / "file_integrity_report.csv", file_integrity(integrity_paths), ["path", "exists", "bytes", "sha256"])
    print(json.dumps(counts, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
