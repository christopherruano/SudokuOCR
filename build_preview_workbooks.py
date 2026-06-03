#!/usr/bin/env python3
"""Build Excel workbooks from preview_pack."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable, List

import xlsxwriter


ROOT = Path("/Users/chrisruano/Research/SudokuOCR/export/final_project_exports/preview_pack")


def slug_sheet_name(text: str, used: set[str]) -> str:
    safe = "".join(ch for ch in text if ch not in "[]:*?/\\")
    safe = safe.strip() or "Sheet"
    safe = safe[:31]
    base = safe
    i = 2
    while safe in used:
        suffix = f"_{i}"
        safe = (base[: 31 - len(suffix)] + suffix)[:31]
        i += 1
    used.add(safe)
    return safe


def read_csv(path: Path) -> List[dict]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def read_summary(path: Path) -> List[str]:
    return path.read_text(encoding="utf-8").splitlines()


def write_key_value_lines(ws, start_row: int, lines: Iterable[str], bold_fmt, mono_fmt) -> int:
    row = start_row
    for line in lines:
        if ":" in line:
            k, v = line.split(":", 1)
            ws.write(row, 0, k.strip(), bold_fmt)
            ws.write(row, 1, v.strip(), mono_fmt)
        else:
            ws.write(row, 0, line, mono_fmt)
        row += 1
    return row


def write_table(ws, start_row: int, rows: List[dict], header_fmt, mono_fmt) -> int:
    if not rows:
        ws.write(start_row, 0, "(no rows)", mono_fmt)
        return start_row + 1
    fieldnames = list(rows[0].keys())
    for c, name in enumerate(fieldnames):
        ws.write(start_row, c, name, header_fmt)
    row_ix = start_row + 1
    for row in rows:
        for c, name in enumerate(fieldnames):
            ws.write(row_ix, c, row.get(name, ""), mono_fmt)
        row_ix += 1
    return row_ix


def autosize(ws, rows: List[List[str]], max_width: int = 40) -> None:
    widths: Dict[int, int] = {}
    for row in rows:
        for c, val in enumerate(row):
            widths[c] = min(max(widths.get(c, 0), len(str(val))), max_width)
    for c, width in widths.items():
        ws.set_column(c, c, width + 2)


def build_page_level_workbook(dataset_dir: Path, output_path: Path) -> None:
    workbook = xlsxwriter.Workbook(str(output_path))
    header_fmt = workbook.add_format({"bold": True, "bg_color": "#D9EAF7", "border": 1})
    bold_fmt = workbook.add_format({"bold": True})
    mono_fmt = workbook.add_format({"font_name": "Menlo", "font_size": 10})
    link_fmt = workbook.add_format({"font_color": "blue", "underline": 1})

    used: set[str] = set()
    index = workbook.add_worksheet("Index")
    index.write_row(0, 0, ["sheet", "folder", "title", "province", "year", "family"])

    folders = sorted([p for p in dataset_dir.iterdir() if p.is_dir()])
    for i, folder in enumerate(folders, start=1):
        summary_lines = read_summary(folder / "summary.txt")
        source_row = __import__("json").loads((folder / "source_catalog_row.json").read_text(encoding="utf-8"))
        matrix_rows = read_csv(folder / "matrix.csv")
        sheet_name = slug_sheet_name(f"{source_row.get('province','')}_{source_row.get('year','')}_{source_row.get('table_family','')}", used)

        index.write_url(i, 0, f"internal:'{sheet_name}'!A1", link_fmt, string=sheet_name)
        index.write(i, 1, str(folder.relative_to(ROOT)))
        index.write(i, 2, source_row.get("title", ""))
        index.write(i, 3, source_row.get("province", ""))
        index.write(i, 4, source_row.get("year", ""))
        index.write(i, 5, source_row.get("table_family", ""))

        ws = workbook.add_worksheet(sheet_name)
        ws.freeze_panes(1, 0)
        row = 0
        ws.write(row, 0, "Summary", header_fmt)
        row = write_key_value_lines(ws, row + 1, summary_lines, bold_fmt, mono_fmt) + 1
        ws.write(row, 0, "Matrix View", header_fmt)
        row += 1
        row = write_table(ws, row, matrix_rows, header_fmt, mono_fmt)
        autosize(ws, [list(r.keys()) for r in matrix_rows[:1]] + [list(r.values()) for r in matrix_rows[:25]])

    workbook.close()


def build_assembled_workbook(dataset_dir: Path, output_path: Path) -> None:
    workbook = xlsxwriter.Workbook(str(output_path))
    header_fmt = workbook.add_format({"bold": True, "bg_color": "#F6E4D2", "border": 1})
    bold_fmt = workbook.add_format({"bold": True})
    mono_fmt = workbook.add_format({"font_name": "Menlo", "font_size": 10})
    link_fmt = workbook.add_format({"font_color": "blue", "underline": 1})

    used: set[str] = set()
    index = workbook.add_worksheet("Index")
    index.write_row(0, 0, ["sheet", "folder", "title", "province", "year", "family", "compiled_member_count"])

    folders = sorted([p for p in dataset_dir.iterdir() if p.is_dir()])
    for i, folder in enumerate(folders, start=1):
        summary_lines = read_summary(folder / "summary.txt")
        source_row = __import__("json").loads((folder / "source_catalog_row.json").read_text(encoding="utf-8"))
        sections_index = read_csv(folder / "sections_index.csv")
        member_pages = read_csv(folder / "member_pages.csv") if (folder / "member_pages.csv").exists() else []
        section_files = sorted(folder.glob("section_*.csv"))
        sheet_name = slug_sheet_name(f"{source_row.get('province','')}_{source_row.get('year','')}_{source_row.get('table_family','')}", used)

        index.write_url(i, 0, f"internal:'{sheet_name}'!A1", link_fmt, string=sheet_name)
        index.write(i, 1, str(folder.relative_to(ROOT)))
        index.write(i, 2, source_row.get("title", ""))
        index.write(i, 3, source_row.get("province", ""))
        index.write(i, 4, source_row.get("year", ""))
        index.write(i, 5, source_row.get("table_family", ""))
        index.write(i, 6, source_row.get("compiled_member_count", ""))

        ws = workbook.add_worksheet(sheet_name)
        row = 0
        ws.write(row, 0, "Summary", header_fmt)
        row = write_key_value_lines(ws, row + 1, summary_lines, bold_fmt, mono_fmt) + 1

        ws.write(row, 0, "Sections Index", header_fmt)
        row = write_table(ws, row + 1, sections_index, header_fmt, mono_fmt) + 1

        if member_pages:
            ws.write(row, 0, "Member Pages", header_fmt)
            row = write_table(ws, row + 1, member_pages, header_fmt, mono_fmt) + 2

        for section_file in section_files[:6]:
            section_rows = read_csv(section_file)
            ws.write(row, 0, section_file.name, header_fmt)
            row = write_table(ws, row + 1, section_rows, header_fmt, mono_fmt) + 2
        autosize(ws, [["x"] * 10])

    workbook.close()


def main() -> None:
    build_page_level_workbook(ROOT / "core_example", ROOT / "core_examples.xlsx")
    build_page_level_workbook(ROOT / "tail_example", ROOT / "tail_examples.xlsx")
    build_assembled_workbook(ROOT / "assembled_reviewed", ROOT / "assembled_reviewed.xlsx")
    print("Wrote:")
    print(ROOT / "core_examples.xlsx")
    print(ROOT / "tail_examples.xlsx")
    print(ROOT / "assembled_reviewed.xlsx")


if __name__ == "__main__":
    main()
