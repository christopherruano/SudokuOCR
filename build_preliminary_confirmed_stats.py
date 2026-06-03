from __future__ import annotations

import csv
import json
import math
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

import xlsxwriter


SENDOUT = Path("export_final/sendout_final")
DATA_DIR = SENDOUT / "data"
OUT_DIR = SENDOUT / "analysis" / "preliminary_stats"

TABLES_CSV = DATA_DIR / "tables_confirmed.csv"
FACTS_CSV = DATA_DIR / "facts_confirmed.csv"

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

BROAD_AGE_FIELDS = [
    "child_0_14",
    "working_15_59",
    "older_60_plus",
]

SPECIAL_TOPIC_RE = re.compile(
    r"religion|civil condition|caste|tribe|race|occupation|infirm|deaf|blind|"
    r"leper|insane|literacy|education|vaccinated|small[- ]?pox|life table|"
    r"livelihood|जीविका|proportion|percentage|per\s*1[, ]?000|rate|ratio",
    re.IGNORECASE,
)


def read_tables() -> dict[str, dict[str, str]]:
    with TABLES_CSV.open(newline="", encoding="utf-8") as fh:
        return {row["table_id"]: row for row in csv.DictReader(fh)}


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def to_number(value: str) -> float | None:
    text = str(value or "").replace(",", "").strip()
    if not text:
        return None
    try:
        value_f = float(text)
    except ValueError:
        return None
    if not math.isfinite(value_f):
        return None
    return value_f


def to_intish(value: str) -> float | None:
    number = to_number(value)
    if number is None:
        return None
    return number


def truthy(value: str) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def age_bucket(row: dict[str, str]) -> str:
    if truthy(row.get("resolved_age_total", "")) or truthy(row.get("resolved_age_not_stated", "")):
        return ""
    lower = to_intish(row.get("resolved_age_lower", ""))
    upper = to_intish(row.get("resolved_age_upper", ""))
    if lower is None:
        return ""

    open_ended = truthy(row.get("resolved_age_open_ended", ""))
    if open_ended or upper is None:
        return "older_60_plus" if lower >= 60 else ""

    # Most historical labels like 10-15 mean "10 and under 15"; treat ranges
    # as half-open for broad bucket assignment. Single-year ages use the lower.
    if upper <= lower:
        if lower < 15:
            return "child_0_14"
        if lower < 60:
            return "working_15_59"
        return "older_60_plus"
    if lower >= 0 and upper <= 15:
        return "child_0_14"
    if lower >= 15 and upper <= 60:
        return "working_15_59"
    if lower >= 60:
        return "older_60_plus"
    return ""


def confirmed_strict(row: dict[str, str]) -> bool:
    return (
        row.get("quality_tier") == "strict_validated"
        and str(row.get("failed_checks", "")).strip() in {"0", "0.0", ""}
        and row.get("effective_quality_status") == "accepted_strict"
    )


def write_counter(path: Path, counter: Counter, key_name: str, value_name: str = "tables") -> list[dict]:
    rows = [{key_name: key, value_name: value} for key, value in counter.most_common()]
    write_csv(path, rows, [key_name, value_name])
    return rows


def build_coverage(tables: dict[str, dict[str, str]]) -> dict[str, list[dict]]:
    family = Counter()
    format_class = Counter()
    measure = Counter()
    year = Counter()
    province = Counter()
    by_year_family = Counter()

    for row in tables.values():
        family[row.get("table_family", "")] += 1
        format_class[row.get("format_class", "")] += 1
        measure[row.get("measure_category", "")] += 1
        year[row.get("year", "")] += 1
        province[row.get("province", "")] += 1
        by_year_family[(row.get("year", ""), row.get("table_family", ""))] += 1

    rows_by_year_family = [
        {"year": year, "table_family": family, "tables": count}
        for (year, family), count in sorted(by_year_family.items())
    ]
    write_csv(OUT_DIR / "coverage_by_year_family.csv", rows_by_year_family, ["year", "table_family", "tables"])

    return {
        "by_family": write_counter(OUT_DIR / "coverage_by_table_family.csv", family, "table_family"),
        "by_format": write_counter(OUT_DIR / "coverage_by_format_class.csv", format_class, "format_class"),
        "by_measure": write_counter(OUT_DIR / "coverage_by_measure_category.csv", measure, "measure_category"),
        "by_year": write_counter(OUT_DIR / "coverage_by_year.csv", year, "year"),
        "by_province": write_counter(OUT_DIR / "coverage_by_province.csv", province, "province"),
        "by_year_family": rows_by_year_family,
    }


def primary_age_fact(meta: dict[str, str], fact: dict[str, str]) -> bool:
    if not confirmed_strict(meta):
        return False
    if meta.get("measure_category") != "counts":
        return False
    if meta.get("table_family") != "age_sex_population":
        return False
    if meta.get("format_class") != "additive_age_sex_counts":
        return False
    if fact.get("measure_unit") != "count":
        return False
    if fact.get("sex") not in {"All", "Male", "Female"}:
        return False
    if not fact.get("resolved_age_lower"):
        return False
    if truthy(fact.get("resolved_age_total", "")) or truthy(fact.get("resolved_age_not_stated", "")):
        return False
    row_dim = fact.get("row_dimension", "")
    col_dim = fact.get("column_dimension", "")
    allowed_dims = {"age", "sex", "other", ""}
    if row_dim not in allowed_dims or col_dim not in allowed_dims:
        return False
    if (row_dim == "age") + (col_dim == "age") != 1:
        return False
    return True


def profile_entity(fact: dict[str, str]) -> str:
    section = fact.get("section", "")
    if fact.get("resolved_age_axis") == "column" and fact.get("row_dimension") != "age":
        entity = fact.get("row_label_std") or fact.get("row_label_raw") or ""
        return " | ".join(part for part in [section, entity] if part)
    return section or "table"


def profile_key(meta: dict[str, str], fact: dict[str, str]) -> tuple[str, str, str, str]:
    return (
        fact["table_id"],
        fact.get("section", ""),
        profile_entity(fact),
        fact.get("sex", ""),
    )


def build_age_profiles(tables: dict[str, dict[str, str]]) -> tuple[list[dict], dict[str, int]]:
    profiles: dict[tuple[str, str, str, str], dict] = {}
    stats = Counter()

    with FACTS_CSV.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for fact in reader:
            meta = tables.get(fact.get("table_id", ""))
            if not meta:
                continue
            stats["facts_scanned"] += 1
            if not primary_age_fact(meta, fact):
                continue
            stats["primary_age_facts"] += 1
            value = to_number(fact.get("value", ""))
            if value is None:
                stats["non_numeric_primary_age_facts"] += 1
                continue
            key = profile_key(meta, fact)
            bucket = age_bucket(fact)
            row = profiles.setdefault(
                key,
                {
                    "table_id": fact.get("table_id", ""),
                    "province": meta.get("province", ""),
                    "year": meta.get("year", ""),
                    "title": meta.get("title", ""),
                    "table_family": meta.get("table_family", ""),
                    "format_class": meta.get("format_class", ""),
                    "section": fact.get("section", ""),
                    "entity_label": profile_entity(fact),
                    "sex": fact.get("sex", ""),
                    "age_total_observed": 0.0,
                    "age_total_assigned": 0.0,
                    "n_age_facts": 0,
                    "n_unassigned_age_facts": 0,
                    "n_child_bins": 0,
                    "n_working_bins": 0,
                    "n_older_bins": 0,
                    "child_0_14": 0.0,
                    "working_15_59": 0.0,
                    "older_60_plus": 0.0,
                    "source_image_local": meta.get("source_image_local", ""),
                },
            )
            row["age_total_observed"] += value
            row["n_age_facts"] += 1
            if bucket:
                row[bucket] += value
                row["age_total_assigned"] += value
                row[f"n_{bucket.replace('_0_14', '').replace('_15_59', '').replace('_60_plus', '')}_bins"] += 1
            else:
                row["n_unassigned_age_facts"] += 1

    output = []
    for row in profiles.values():
        observed = row["age_total_observed"]
        assigned = row["age_total_assigned"]
        row["assigned_share"] = assigned / observed if observed else 0.0
        for field in BROAD_AGE_FIELDS:
            row[f"share_{field}"] = row[field] / assigned if assigned else ""
        row["complete_broad_profile"] = (
            assigned > 0
            and row["assigned_share"] >= 0.95
            and row["n_child_bins"] > 0
            and row["n_working_bins"] > 0
            and row["n_older_bins"] > 0
        )
        flags = []
        if assigned and assigned < 1000:
            flags.append("small_profile_total_lt_1000")
        if row["complete_broad_profile"]:
            child_share = float(row["share_child_0_14"])
            older_share = float(row["share_older_60_plus"])
            if child_share < 0.10:
                flags.append("very_low_child_share_lt_10pct")
            if child_share > 0.60:
                flags.append("very_high_child_share_gt_60pct")
            if older_share > 0.25:
                flags.append("very_high_older_share_gt_25pct")
        if SPECIAL_TOPIC_RE.search(f"{row.get('title', '')} {row.get('entity_label', '')}"):
            flags.append("special_topic_or_non_population_keyword")
        row["analysis_flags"] = ";".join(flags)
        row["quick_descriptive_ready"] = (
            row["complete_broad_profile"]
            and assigned >= 1000
            and "very_low_child_share_lt_10pct" not in flags
            and "very_high_child_share_gt_60pct" not in flags
            and "very_high_older_share_gt_25pct" not in flags
            and "special_topic_or_non_population_keyword" not in flags
        )
        output.append(row)

    output.sort(key=lambda r: (r["year"], r["province"], r["table_id"], r["section"], r["entity_label"], r["sex"]))
    return output, dict(stats)


def percentile(values: list[float], q: float) -> float | str:
    if not values:
        return ""
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    pos = (len(values) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return values[lo]
    return values[lo] + (values[hi] - values[lo]) * (pos - lo)


def summarize_profiles(rows: list[dict], keys: list[str], include_sex: bool = False) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        if not row.get("complete_broad_profile"):
            continue
        if not include_sex and row.get("sex") != "All":
            continue
        grouped[tuple(row.get(key, "") for key in keys)].append(row)

    output = []
    for key_values, group in sorted(grouped.items()):
        total_weight = sum(float(row["age_total_assigned"]) for row in group)
        out = {key: value for key, value in zip(keys, key_values)}
        out["profile_units"] = len(group)
        out["tables"] = len({row["table_id"] for row in group})
        out["provinces"] = len({row["province"] for row in group})
        out["assigned_population_observed"] = total_weight
        for field in BROAD_AGE_FIELDS:
            shares = [float(row[f"share_{field}"]) for row in group if row.get(f"share_{field}") != ""]
            out[f"median_share_{field}"] = statistics.median(shares) if shares else ""
            out[f"mean_share_{field}"] = statistics.mean(shares) if shares else ""
            out[f"p25_share_{field}"] = percentile(shares, 0.25)
            out[f"p75_share_{field}"] = percentile(shares, 0.75)
            out[f"weighted_share_{field}"] = (
                sum(float(row[field]) for row in group) / total_weight if total_weight else ""
            )
        output.append(out)
    return output


def summarize_filtered_profiles(rows: list[dict], keys: list[str], flag_field: str, include_sex: bool = False) -> list[dict]:
    return summarize_profiles([row for row in rows if row.get(flag_field)], keys, include_sex=include_sex)


def fmt_pct(value) -> str:
    if value == "" or value is None:
        return ""
    return f"{float(value) * 100:.1f}%"


def write_workbook(paths: dict[str, Path], sheet_rows: dict[str, tuple[list[dict], list[str]]]) -> None:
    workbook = xlsxwriter.Workbook(str(paths["workbook"]), {"constant_memory": True, "strings_to_urls": False})
    header = workbook.add_format({"bold": True, "bg_color": "#E8EEF6", "border": 1})
    normal = workbook.add_format({"border": 1})
    for sheet_name, (rows, fields) in sheet_rows.items():
        ws = workbook.add_worksheet(sheet_name[:31])
        ws.freeze_panes(1, 0)
        ws.write_row(0, 0, fields, header)
        for r, row in enumerate(rows, start=1):
            ws.write_row(r, 0, [row.get(field, "") for field in fields], normal)
        ws.autofilter(0, 0, len(rows), max(len(fields) - 1, 0))
        ws.set_column(0, min(len(fields), 12), 16)
    workbook.close()


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tables = read_tables()
    coverage = build_coverage(tables)
    profiles, stream_stats = build_age_profiles(tables)

    profile_fields = [
        "table_id",
        "province",
        "year",
        "title",
        "table_family",
        "format_class",
        "section",
        "entity_label",
        "sex",
        "age_total_observed",
        "age_total_assigned",
        "assigned_share",
        "complete_broad_profile",
        "quick_descriptive_ready",
        "analysis_flags",
        "n_age_facts",
        "n_unassigned_age_facts",
        "n_child_bins",
        "n_working_bins",
        "n_older_bins",
        "child_0_14",
        "working_15_59",
        "older_60_plus",
        "share_child_0_14",
        "share_working_15_59",
        "share_older_60_plus",
        "source_image_local",
    ]
    write_csv(OUT_DIR / "age_profile_units.csv", profiles, profile_fields)

    by_year = summarize_profiles(profiles, ["year"])
    by_year_sex = summarize_profiles(profiles, ["year", "sex"], include_sex=True)
    by_province_year = summarize_profiles(profiles, ["province", "year"])
    by_year_quick = summarize_filtered_profiles(profiles, ["year"], "quick_descriptive_ready")
    flagged_profiles = [row for row in profiles if row.get("analysis_flags")]

    summary_fields = [
        "year",
        "profile_units",
        "tables",
        "provinces",
        "assigned_population_observed",
        "median_share_child_0_14",
        "median_share_working_15_59",
        "median_share_older_60_plus",
        "weighted_share_child_0_14",
        "weighted_share_working_15_59",
        "weighted_share_older_60_plus",
        "p25_share_child_0_14",
        "p75_share_child_0_14",
        "p25_share_older_60_plus",
        "p75_share_older_60_plus",
    ]
    year_sex_fields = ["year", "sex"] + [field for field in summary_fields if field != "year"]
    province_year_fields = ["province"] + summary_fields

    write_csv(OUT_DIR / "age_over_time_by_year.csv", by_year, summary_fields)
    write_csv(OUT_DIR / "age_over_time_quick_descriptive_by_year.csv", by_year_quick, summary_fields)
    write_csv(OUT_DIR / "age_over_time_by_year_sex.csv", by_year_sex, year_sex_fields)
    write_csv(OUT_DIR / "age_over_time_by_province_year.csv", by_province_year, province_year_fields)
    write_csv(OUT_DIR / "age_profile_plausibility_flags.csv", flagged_profiles, profile_fields)

    stats = {
        "confirmed_tables": len(tables),
        "confirmed_facts_scanned": stream_stats.get("facts_scanned", 0),
        "primary_age_facts": stream_stats.get("primary_age_facts", 0),
        "age_profile_units": len(profiles),
        "complete_all_sex_profile_units": sum(
            1 for row in profiles if row.get("complete_broad_profile") and row.get("sex") == "All"
        ),
        "quick_descriptive_all_sex_profile_units": sum(
            1 for row in profiles if row.get("quick_descriptive_ready") and row.get("sex") == "All"
        ),
        "age_years_with_complete_profiles": len(by_year),
        "age_years_with_quick_descriptive_profiles": len(by_year_quick),
        "notes": [
            "Primary age-over-time series uses strict validated additive age/sex count tables only.",
            "Rows are table/profile units, not a deduplicated population census.",
            "Do not sum counts across profile units without deduplication by source/table geography.",
        ],
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    top_year_lines = []
    for row in by_year:
        top_year_lines.append(
            "| {year} | {profile_units} | {tables} | {child} | {working} | {older} |".format(
                year=row["year"],
                profile_units=row["profile_units"],
                tables=row["tables"],
                child=fmt_pct(row["median_share_child_0_14"]),
                working=fmt_pct(row["median_share_working_15_59"]),
                older=fmt_pct(row["median_share_older_60_plus"]),
            )
        )

    readme = f"""# Preliminary Confirmed-Table Stats

These files are generated from `data/tables_confirmed.csv` and `data/facts_confirmed.csv`.

## Scope

- Confirmed tables: {stats["confirmed_tables"]:,}
- Confirmed facts scanned: {stats["confirmed_facts_scanned"]:,}
- Primary age-over-time facts used: {stats["primary_age_facts"]:,}
- Age profile units built: {stats["age_profile_units"]:,}
- Complete `All`-sex broad age profiles: {stats["complete_all_sex_profile_units"]:,}

## Important Interpretation Note

The age-over-time summaries are preliminary profile-level summaries. They are useful for
seeing broad temporal patterns in the confirmed age tables, but they are **not** a
deduplicated population series. Multiple confirmed tables can cover overlapping geographies
or subgroup sections, so do not sum `assigned_population_observed` across rows as if it were
an all-India total.

The primary series uses only strict-validated additive age/sex count tables, excludes rate
and per-1,000 tables, and keeps only age bins that can be assigned cleanly to:

- `child_0_14`
- `working_15_59`
- `older_60_plus`

## Median Broad Age Shares By Year

| year | profile units | tables | child 0-14 | working 15-59 | older 60+ |
|---|---:|---:|---:|---:|---:|
{chr(10).join(top_year_lines)}

## Files

- `age_profile_units.csv`: one table/section/entity/sex profile with broad age counts and shares.
- `age_over_time_by_year.csv`: median/mean/weighted broad age shares by census year for `sex=All`.
- `age_over_time_quick_descriptive_by_year.csv`: stricter quick-look version excluding tiny, extreme, and special-topic profiles.
- `age_over_time_by_year_sex.csv`: same summary split by sex.
- `age_over_time_by_province_year.csv`: same summary by province-year.
- `age_profile_plausibility_flags.csv`: profile units with small totals, extreme shares, or special-topic keywords.
- `coverage_by_*.csv`: confirmed table coverage by family, format, measure, year, and province.
- `preliminary_confirmed_stats.xlsx`: browseable workbook version.
"""
    (OUT_DIR / "README.md").write_text(readme, encoding="utf-8")

    workbook_path = OUT_DIR / "preliminary_confirmed_stats.xlsx"
    write_workbook(
        {"workbook": workbook_path},
        {
            "age_by_year": (by_year, summary_fields),
            "quick_age_year": (by_year_quick, summary_fields),
            "age_by_year_sex": (by_year_sex, year_sex_fields),
            "province_year": (by_province_year, province_year_fields),
            "age_profile_units": (profiles, profile_fields),
            "profile_flags": (flagged_profiles, profile_fields),
            "coverage_family": (coverage["by_family"], ["table_family", "tables"]),
            "coverage_format": (coverage["by_format"], ["format_class", "tables"]),
            "coverage_year": (coverage["by_year"], ["year", "tables"]),
        },
    )

    print(f"Wrote preliminary stats to {OUT_DIR}")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
