from __future__ import annotations

import csv
import hashlib
import re
import sys
from collections import defaultdict
from pathlib import Path

import xlsxwriter


ROOT = Path("export_final")
SENDOUT = ROOT / "sendout_final"
APP_DIR = SENDOUT / "review" / "plausibility_review_app"
REVIEW_CSV = APP_DIR / "review_has_entries.csv"
DECISIONS_CSV = APP_DIR / "review_decisions.csv"
OUT_DIR = APP_DIR / "excel_review_tables"
DATASETS = {
    "core": ROOT / "datasets" / "core_curated_v2_final" / "cell_facts_rich.csv",
    "tail": ROOT / "datasets" / "tail_exploratory_v2" / "cell_facts_rich.csv",
}

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

SUMMARY_FIELDS = [
    "table_id",
    "layer",
    "province",
    "year",
    "title",
    "sendout_bucket",
    "format_class",
    "plausibility_status",
    "recommendation",
    "flags",
    "issue_flags",
    "issue_details",
    "failed_checks",
    "quality_tier",
    "facts_seen",
    "source_image_local",
    "source_json_local",
]

FACT_FIELDS = [
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
    "measure_unit",
    "value",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def slug(text: str, limit: int = 80) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text.strip())
    text = re.sub(r"_+", "_", text).strip("._-")
    return (text or "table")[:limit]


def table_hash(table_id: str) -> str:
    return hashlib.sha1(table_id.encode("utf-8")).hexdigest()[:12]


def workbook_path(row: dict[str, str]) -> Path:
    name = "_".join(
        [
            slug(row.get("layer", "")),
            slug(row.get("province", "")),
            slug(row.get("year", "")),
            slug(row.get("format_class", "")),
            table_hash(row.get("table_id", "")),
        ]
    )
    return OUT_DIR / row.get("layer", "unknown") / f"{name}.xlsx"


def to_number(value: str):
    text = str(value or "").replace(",", "").strip()
    if not text:
        return ""
    try:
        number = float(text)
    except ValueError:
        return value
    if number.is_integer():
        return int(number)
    return number


def matrix_shape(facts: list[dict[str, str]]) -> tuple[list[tuple[str, str, str]], list[tuple[str, str, str, str, str]]]:
    rows = sorted({(f.get("section", ""), f.get("row_label_raw", ""), f.get("row_label_std", "")) for f in facts})
    cols = sorted(
        {
            (
                f.get("column_group_raw", ""),
                f.get("sub_column_raw", ""),
                f.get("sex", ""),
                f.get("civil_condition", ""),
                f.get("religion", ""),
            )
            for f in facts
        }
    )
    return rows, cols


def write_workbook(path: Path, review_row: dict[str, str], facts: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    workbook = xlsxwriter.Workbook(str(path), {"constant_memory": True, "strings_to_urls": False})
    header = workbook.add_format({"bold": True, "bg_color": "#E8EEF6"})
    wrap = workbook.add_format({"text_wrap": True, "valign": "top"})

    summary = workbook.add_worksheet("Summary")
    for idx, field in enumerate(SUMMARY_FIELDS):
        summary.write(idx, 0, field, header)
        summary.write(idx, 1, review_row.get(field, ""), wrap)
    summary.set_column(0, 0, 24)
    summary.set_column(1, 1, 90)

    matrix = workbook.add_worksheet("Matrix")
    row_labels, col_labels = matrix_shape(facts)
    if len(row_labels) <= 1800 and len(col_labels) <= 240:
        matrix.write_row(0, 0, ["section", "row_label_raw", "row_label_std"], header)
        for c, label in enumerate(col_labels, start=3):
            matrix.write(0, c, " | ".join(part for part in label if part), header)
        row_index = {key: idx + 1 for idx, key in enumerate(row_labels)}
        col_index = {key: idx + 3 for idx, key in enumerate(col_labels)}
        for key, r in row_index.items():
            matrix.write_row(r, 0, list(key))
        for fact in facts:
            r = row_index[(fact.get("section", ""), fact.get("row_label_raw", ""), fact.get("row_label_std", ""))]
            c = col_index[
                (
                    fact.get("column_group_raw", ""),
                    fact.get("sub_column_raw", ""),
                    fact.get("sex", ""),
                    fact.get("civil_condition", ""),
                    fact.get("religion", ""),
                )
            ]
            matrix.write(r, c, to_number(fact.get("value", "")))
        matrix.set_column(0, 2, 22)
    else:
        matrix.write(0, 0, "not suitable for compact reconstruction", header)
        matrix.write(1, 0, f"rows={len(row_labels)} columns={len(col_labels)}")

    facts_ws = workbook.add_worksheet("Facts")
    facts_ws.write_row(0, 0, FACT_FIELDS, header)
    for idx, fact in enumerate(facts, start=1):
        facts_ws.write_row(idx, 0, [fact.get(field, "") for field in FACT_FIELDS])

    source = workbook.add_worksheet("Source")
    source_rows = [
        ("source_image_local", review_row.get("source_image_local", "")),
        ("source_json_local", review_row.get("source_json_local", "")),
        ("source_image_original", review_row.get("source_image_original", "")),
        ("source_json_original", review_row.get("source_json_original", "")),
        ("preview_matrix", review_row.get("preview_matrix", "")),
        ("preview_facts", review_row.get("preview_facts", "")),
    ]
    for idx, (key, value) in enumerate(source_rows):
        source.write(idx, 0, key, header)
        source.write(idx, 1, value, wrap)
    source.set_column(0, 0, 24)
    source.set_column(1, 1, 100)
    workbook.close()


def collect_facts(review_rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    wanted_by_layer: dict[str, set[str]] = defaultdict(set)
    for row in review_rows:
        wanted_by_layer[row.get("layer", "")].add(row.get("table_id", ""))

    facts: dict[str, list[dict[str, str]]] = defaultdict(list)
    for layer, path in DATASETS.items():
        wanted = wanted_by_layer.get(layer, set())
        if not wanted or not path.exists():
            continue
        with path.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                table_id = row.get("table_id", "")
                if table_id in wanted:
                    facts[table_id].append({field: row.get(field, "") for field in FACT_FIELDS})
    return facts


def main() -> None:
    all_decision_rows = read_csv(DECISIONS_CSV)
    review_rows = [row for row in all_decision_rows if int(float(row.get("facts_seen") or 0)) > 0]
    facts = collect_facts(review_rows)
    index_rows: list[dict[str, str]] = []
    updated_rows: list[dict[str, str]] = []
    missing_rows: list[dict[str, str]] = []
    for idx, row in enumerate(review_rows, start=1):
        table_facts = facts.get(row.get("table_id", ""), [])
        if not table_facts:
            row = dict(row)
            row["excel_missing_reason"] = "facts_not_available_in_local_source_csv"
            missing_rows.append(row)
            updated_rows.append(row)
            continue
        path = workbook_path(row)
        write_workbook(path, row, table_facts)
        rel_path = str(path.relative_to(SENDOUT))
        row = dict(row)
        row["excel_path"] = rel_path
        row["excel_abs"] = str(path.resolve())
        row["excel_missing_reason"] = ""
        updated_rows.append(row)
        index_rows.append(
            {
                "table_id": row.get("table_id", ""),
                "layer": row.get("layer", ""),
                "province": row.get("province", ""),
                "year": row.get("year", ""),
                "title": row.get("title", ""),
                "recommendation": row.get("recommendation", ""),
                "facts_rows": str(len(table_facts)),
                "excel_path": rel_path,
            }
        )
        if idx % 100 == 0:
            print(f"[{idx}/{len(review_rows)}] processed")

    write_csv(APP_DIR / "review_excel_index.csv", index_rows, list(index_rows[0].keys()) if index_rows else ["table_id", "excel_path"])

    updated_by_id = {r["table_id"]: r for r in updated_rows}
    merged = [updated_by_id.get(r.get("table_id", ""), r) for r in all_decision_rows]
    write_csv(DECISIONS_CSV, merged, list(merged[0].keys()) if merged else [])
    write_csv(APP_DIR / "review_has_entries.csv", updated_rows, list(updated_rows[0].keys()) if updated_rows else [])
    write_csv(
        APP_DIR / "review_has_entries_missing_excel.csv",
        missing_rows,
        list(missing_rows[0].keys()) if missing_rows else ["table_id", "excel_missing_reason"],
    )

    print(f"Wrote {len(index_rows)} review workbooks to {OUT_DIR}")
    print(f"Rows with entries but no local workbook source: {len(missing_rows)}")


if __name__ == "__main__":
    main()
