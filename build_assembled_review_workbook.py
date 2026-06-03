from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path
from typing import Iterable

import xlsxwriter


SENDOUT = Path("export_final/sendout_final")
APP_DIR = SENDOUT / "review" / "plausibility_review_app"
REVIEW_CSV = APP_DIR / "review_decisions.csv"
ASSEMBLED_CATALOG = Path("export_final/team_ready/catalogs/by_layer/assembled_tables.csv")
STRICT_ASSEMBLED_CATALOG = Path("export_final/browse/by_quality/strict_validated/assembled_tables.csv")
PREVIEW_ROOT = Path("export/final_project_exports/preview_pack")
ASSEMBLED_PREVIEW_DIR = PREVIEW_ROOT / "assembled_reviewed"
OUT_XLSX = APP_DIR / "assembled_tables_review_all.xlsx"
OUT_INDEX = APP_DIR / "assembled_tables_review_all_index.csv"

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

INDEX_FIELDS = [
    "seq",
    "sheet",
    "table_id",
    "province",
    "year",
    "table_family",
    "title",
    "facts_seen",
    "quality_tier",
    "recommendation",
    "n_checks",
    "failed_checks",
    "assembled_review_decision",
    "preview_folder",
    "source_image_local",
    "member_count",
    "section_count",
]

SUMMARY_FIELDS = [
    "table_id",
    "province",
    "year",
    "title",
    "table_family",
    "facts_seen",
    "quality_tier",
    "confidence_level",
    "effective_quality_status",
    "quality_status",
    "n_checks",
    "failed_checks",
    "qc_flags",
    "measure_category",
    "measure_unit",
    "constraint_regime",
    "denominator_hint",
    "recommendation",
    "issue_flags",
    "issue_details",
    "assembled_review_decision",
    "manual_final_decision",
    "preview_folder",
    "source_image_local",
    "source_json_local",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except json.JSONDecodeError:
        return {}


def write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def safe_sheet_name(seq: int) -> str:
    return f"A{seq:03d}"


def to_excel_value(value: str):
    text = str(value or "").strip()
    if not text:
        return ""
    cleaned = text.replace(",", "")
    try:
        number = float(cleaned)
    except ValueError:
        return text
    if number.is_integer():
        return int(number)
    return number


def load_preview_dirs() -> dict[str, Path]:
    out: dict[str, Path] = {}
    if not ASSEMBLED_PREVIEW_DIR.exists():
        return out
    for source_row_path in ASSEMBLED_PREVIEW_DIR.glob("*/source_catalog_row.json"):
        row = read_json(source_row_path)
        table_id = row.get("table_id")
        if table_id:
            out[table_id] = source_row_path.parent
    return out


def load_catalog_rows() -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    for path in [ASSEMBLED_CATALOG, STRICT_ASSEMBLED_CATALOG]:
        for row in read_csv(path):
            table_id = row.get("table_id", "")
            if not table_id:
                continue
            merged = dict(rows.get(table_id, {}))
            merged.update({k: v for k, v in row.items() if v not in ("", None)})
            rows[table_id] = merged
    return rows


def load_review_rows() -> list[dict[str, str]]:
    rows = []
    for row in read_csv(REVIEW_CSV):
        if row.get("layer") == "assembled" or row.get("table_id", "").startswith("assembled::"):
            rows.append(row)
    return rows


def merge_row(*rows: dict[str, str]) -> dict[str, str]:
    merged: dict[str, str] = {}
    for row in rows:
        for key, value in row.items():
            if value not in ("", None) or key not in merged:
                merged[key] = value
    return merged


def path_from_preview(preview_dir: Path | None, relative: str) -> Path | None:
    if not preview_dir or not relative:
        return None
    path = preview_dir / relative
    return path if path.exists() else None


def sheet_link(sheet: str, cell: str = "A1") -> str:
    return f"internal:'{sheet}'!{cell}"


def write_kv(worksheet, rows: Iterable[tuple[str, str]], start_row: int, key_fmt, value_fmt) -> int:
    row_num = start_row
    for key, value in rows:
        worksheet.write(row_num, 0, key, key_fmt)
        worksheet.write(row_num, 1, value or "", value_fmt)
        row_num += 1
    return row_num


def write_csv_table(worksheet, path: Path, start_row: int, header_fmt, normal_fmt, max_cols: int = 240) -> int:
    if not path.exists():
        worksheet.write(start_row, 0, f"Missing CSV: {path}")
        return start_row + 2
    with path.open(newline="", encoding="utf-8", errors="replace") as fh:
        reader = csv.reader(fh)
        for offset, row in enumerate(reader):
            row_num = start_row + offset
            for col_num, value in enumerate(row[:max_cols]):
                fmt = header_fmt if offset == 0 else normal_fmt
                worksheet.write(row_num, col_num, to_excel_value(value), fmt)
        return start_row + offset + 2 if "offset" in locals() else start_row + 2


def write_member_table(worksheet, member_path: Path | None, start_row: int, header_fmt, normal_fmt) -> int:
    if not member_path or not member_path.exists():
        worksheet.write(start_row, 0, "No member_pages.csv available.")
        return start_row + 2
    rows = read_csv(member_path)
    if not rows:
        worksheet.write(start_row, 0, "member_pages.csv is empty.")
        return start_row + 2
    fields = list(rows[0].keys())
    worksheet.write_row(start_row, 0, fields, header_fmt)
    for r, row in enumerate(rows, start=start_row + 1):
        for c, field in enumerate(fields):
            worksheet.write(r, c, row.get(field, ""), normal_fmt)
    return start_row + len(rows) + 3


def main() -> None:
    review_rows = load_review_rows()
    catalog_by_id = load_catalog_rows()
    preview_by_id = load_preview_dirs()

    table_ids = []
    for row in review_rows + list(catalog_by_id.values()):
        table_id = row.get("table_id", "")
        if table_id.startswith("assembled::") and table_id not in table_ids:
            table_ids.append(table_id)

    rows: list[dict[str, str]] = []
    for seq, table_id in enumerate(table_ids, start=1):
        review = next((row for row in review_rows if row.get("table_id") == table_id), {})
        catalog = catalog_by_id.get(table_id, {})
        preview_dir = preview_by_id.get(table_id)
        preview_source = read_json(preview_dir / "source_catalog_row.json") if preview_dir else {}
        row = merge_row(review, catalog, preview_source)
        row["table_id"] = table_id
        row["seq"] = str(seq)
        row["sheet"] = safe_sheet_name(seq)
        row["preview_folder"] = str(preview_dir) if preview_dir else row.get("preview_folder", "")
        row["source_image_local"] = row.get("source_image_local") or row.get("source_image", "")
        rows.append(row)

    OUT_XLSX.parent.mkdir(parents=True, exist_ok=True)
    workbook = xlsxwriter.Workbook(str(OUT_XLSX), {"constant_memory": True, "strings_to_urls": False})
    title_fmt = workbook.add_format({"bold": True, "font_size": 14})
    header_fmt = workbook.add_format({"bold": True, "bg_color": "#E8EEF6", "border": 1})
    key_fmt = workbook.add_format({"bold": True, "bg_color": "#F4F4F4", "border": 1})
    normal_fmt = workbook.add_format({"text_wrap": False, "valign": "top", "border": 1})
    wrap_fmt = workbook.add_format({"text_wrap": True, "valign": "top", "border": 1})
    muted_fmt = workbook.add_format({"font_color": "#666666", "italic": True})
    link_fmt = workbook.add_format({"font_color": "blue", "underline": 1})

    index_ws = workbook.add_worksheet("Index")
    index_ws.freeze_panes(1, 0)
    index_ws.write_row(0, 0, INDEX_FIELDS, header_fmt)
    index_rows: list[dict[str, str]] = []

    for idx, row in enumerate(rows, start=1):
        sheet = row["sheet"]
        preview_dir = Path(row["preview_folder"]) if row.get("preview_folder") else None
        sections_index_path = path_from_preview(preview_dir, "sections_index.csv")
        sections = read_csv(sections_index_path) if sections_index_path else []
        member_path = path_from_preview(preview_dir, "member_pages.csv")
        member_count = len(read_csv(member_path)) if member_path else 0

        index_row = {field: row.get(field, "") for field in INDEX_FIELDS}
        index_row["member_count"] = str(member_count)
        index_row["section_count"] = str(len(sections))
        index_rows.append(index_row)

        for col, field in enumerate(INDEX_FIELDS):
            value = index_row.get(field, "")
            if field == "sheet":
                index_ws.write_url(idx, col, sheet_link(sheet), link_fmt, string=value)
            else:
                index_ws.write(idx, col, value)

        ws = workbook.add_worksheet(sheet)
        ws.freeze_panes(8, 0)
        ws.set_column(0, 0, 26)
        ws.set_column(1, 1, 90)
        ws.set_column(2, 40, 14)
        ws.write(0, 0, f"{idx}. {row.get('province', '')} {row.get('year', '')}", title_fmt)
        ws.write(1, 0, row.get("title", ""), wrap_fmt)
        ws.write_url(2, 0, sheet_link("Index"), link_fmt, string="Back to Index")
        if idx > 1:
            ws.write_url(2, 1, sheet_link(safe_sheet_name(idx - 1)), link_fmt, string="Previous table")
        if idx < len(rows):
            ws.write_url(2, 2, sheet_link(safe_sheet_name(idx + 1)), link_fmt, string="Next table")

        r = 4
        summary_pairs = [(field, row.get(field, "")) for field in SUMMARY_FIELDS]
        r = write_kv(ws, summary_pairs, r, key_fmt, wrap_fmt)
        r += 1

        image_path = row.get("source_image_local", "")
        if image_path:
            ws.write(r, 0, "source image path", key_fmt)
            ws.write(r, 1, image_path, wrap_fmt)
            r += 1
        if preview_dir:
            ws.write(r, 0, "preview folder", key_fmt)
            ws.write(r, 1, str(preview_dir), wrap_fmt)
            r += 2
        else:
            ws.write(r, 0, "No assembled preview folder found locally.", muted_fmt)
            r += 2

        ws.write(r, 0, "Member pages", title_fmt)
        r += 1
        r = write_member_table(ws, member_path, r, header_fmt, normal_fmt)

        if sections:
            ws.write(r, 0, "Sections", title_fmt)
            r += 1
            ws.write_row(r, 0, list(sections[0].keys()), header_fmt)
            for s_offset, section in enumerate(sections, start=1):
                for c, field in enumerate(sections[0].keys()):
                    ws.write(r + s_offset, c, section.get(field, ""), normal_fmt)
            r += len(sections) + 3

            for section in sections:
                section_name = section.get("section_name", "")
                csv_file = section.get("csv_file", "")
                csv_path = path_from_preview(preview_dir, csv_file)
                ws.write(r, 0, f"Section {section.get('section_index', '')}: {section_name}", title_fmt)
                ws.write(r, 1, f"strategy={section.get('merge_strategy', '')}; input_pages={section.get('input_pages', '')}; output_rows={section.get('output_rows', '')}", muted_fmt)
                r += 1
                if csv_path:
                    r = write_csv_table(ws, csv_path, r, header_fmt, normal_fmt)
                else:
                    ws.write(r, 0, f"Missing section CSV: {csv_file}", muted_fmt)
                    r += 2
        else:
            ws.write(r, 0, "No section matrix available; this row is likely a rerun/no-output assembled candidate.", muted_fmt)

    index_ws.autofilter(0, 0, len(rows), len(INDEX_FIELDS) - 1)
    index_ws.set_column(0, 1, 10)
    index_ws.set_column(2, 5, 18)
    index_ws.set_column(6, 6, 70)
    index_ws.set_column(7, len(INDEX_FIELDS) - 1, 18)
    workbook.close()

    write_csv(OUT_INDEX, index_rows, INDEX_FIELDS)
    print(f"Assembled workbook written: {OUT_XLSX}")
    print(f"Assembled tables included: {len(rows)}")
    print(f"Tables with preview sections: {sum(1 for r in index_rows if int(r.get('section_count') or 0) > 0)}")
    print(f"Index written: {OUT_INDEX}")


if __name__ == "__main__":
    main()
