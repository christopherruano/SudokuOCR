#!/usr/bin/env python3
"""Build a human-readable workbook for compiled multi-page assembled tables."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

import xlsxwriter


DEFAULT_ASSEMBLED_JSONL = Path("export_final/assembled_all_read_improved/strong_assembled_tables.jsonl")
DEFAULT_OUTPUT_DIR = Path("export_final/assembled_all_read_review")
DEFAULT_DIGITIZED_LOCAL_ROOT = Path("export/digitized_tables_referenced/digitized_tables")
CLUSTER_DIGITIZED_PREFIX = "/n/home11/cruano/census_toc/digitized_tables/"

INDEX_FIELDS = [
    "seq",
    "sheet",
    "group_id",
    "province",
    "year",
    "table_family",
    "title",
    "n_pages",
    "overall_strategy",
    "signals",
    "assembled_constraint_checks",
    "assembled_constraint_failures",
    "assembled_all_passed",
    "section_count",
    "member_page_count",
]


def normalize_text(value: object) -> str:
    return " ".join(str(value or "").strip().split())


def safe_sheet_name(seq: int) -> str:
    return f"A{seq:03d}"


def safe_text(value: object, limit: int = 32000) -> str:
    text = normalize_text(value)
    if len(text) > limit:
        return text[: limit - 20] + " ... [truncated]"
    return text


def to_excel_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (int, float)):
        return value
    text = str(value).strip()
    if not text:
        return ""
    cleaned = text.replace(",", "")
    try:
        number = float(cleaned)
    except ValueError:
        return safe_text(text)
    if number.is_integer():
        return int(number)
    return number


def local_image_path(path_text: object, local_root: Path) -> str:
    text = str(path_text or "").strip()
    if not text:
        return ""
    path = Path(text)
    if path.exists():
        return str(path)
    if CLUSTER_DIGITIZED_PREFIX in text:
        rel = text.split(CLUSTER_DIGITIZED_PREFIX, 1)[1]
        candidate = local_root / rel
        if candidate.exists():
            return str(candidate)
    return text


def load_groups(path: Path) -> list[dict[str, Any]]:
    groups = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                groups.append(json.loads(line))
    return groups


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def member_rows(group: dict[str, Any], local_root: Path) -> list[dict[str, Any]]:
    source_jsons = group.get("page_source_jsons") or []
    source_images = group.get("page_source_images") or []
    rows = []
    max_len = max(len(source_jsons), len(source_images), 1)
    for idx in range(max_len):
        image = source_images[idx] if idx < len(source_images) else ""
        rows.append(
            {
                "member_index": idx + 1,
                "source_json": source_jsons[idx] if idx < len(source_jsons) else "",
                "source_image": image,
                "local_image": local_image_path(image, local_root),
            }
        )
    return rows


def flatten_section_rows(rows: list[dict[str, Any]]) -> list[list[Any]]:
    scalar_keys: list[str] = []
    value_keys: list[tuple[str, str]] = []
    scalar_seen = set()
    value_seen = set()

    for row in rows:
        for key, value in row.items():
            if isinstance(value, dict):
                for subkey in value:
                    pair = (str(key), str(subkey))
                    if pair not in value_seen:
                        value_seen.add(pair)
                        value_keys.append(pair)
            elif key not in scalar_seen:
                scalar_seen.add(key)
                scalar_keys.append(str(key))

    # Put likely row labels first while preserving all scalar context.
    preferred = [
        key
        for key in [
            "row_label",
            "age",
            "AGE",
            "Age",
            "RELIGION",
            "Religion",
            "Caste",
            "CASTE",
            "PROVINCE OR STATE.",
            "District",
            "__page__",
        ]
        if key in scalar_keys
    ]
    scalar_keys = preferred + [key for key in scalar_keys if key not in preferred]

    header = scalar_keys + [f"{key} | {subkey}" for key, subkey in value_keys]
    output = [header]
    for row in rows:
        output.append(
            [row.get(key, "") for key in scalar_keys]
            + [
                (row.get(key) or {}).get(subkey, "") if isinstance(row.get(key), dict) else ""
                for key, subkey in value_keys
            ]
        )
    return output


def write_table(ws, start_row: int, rows: list[list[Any]], header_fmt, normal_fmt, max_cols: int) -> int:
    if not rows:
        ws.write(start_row, 0, "No rows.")
        return start_row + 2
    for r_offset, row in enumerate(rows):
        fmt = header_fmt if r_offset == 0 else normal_fmt
        for col, value in enumerate(row[:max_cols]):
            ws.write(start_row + r_offset, col, to_excel_value(value), fmt)
    return start_row + len(rows) + 2


def write_kv(ws, start_row: int, rows: list[tuple[str, Any]], key_fmt, value_fmt) -> int:
    row_num = start_row
    for key, value in rows:
        ws.write(row_num, 0, key, key_fmt)
        ws.write(row_num, 1, safe_text(value), value_fmt)
        row_num += 1
    return row_num


def sheet_link(sheet: str, cell: str = "A1") -> str:
    return f"internal:'{sheet}'!{cell}"


def autosize_widths(ws, widths: dict[int, int], cap: int = 80) -> None:
    for col, width in widths.items():
        ws.set_column(col, col, min(max(width + 2, 10), cap))


def build_workbook(groups: list[dict[str, Any]], output_xlsx: Path, output_index: Path, local_root: Path, max_cols: int) -> None:
    output_xlsx.parent.mkdir(parents=True, exist_ok=True)
    workbook = xlsxwriter.Workbook(str(output_xlsx), {"constant_memory": True, "strings_to_urls": False})

    title_fmt = workbook.add_format({"bold": True, "font_size": 14})
    header_fmt = workbook.add_format({"bold": True, "bg_color": "#E8EEF6", "border": 1})
    key_fmt = workbook.add_format({"bold": True, "bg_color": "#F3F3F3", "border": 1})
    normal_fmt = workbook.add_format({"border": 1, "valign": "top"})
    wrap_fmt = workbook.add_format({"border": 1, "valign": "top", "text_wrap": True})
    muted_fmt = workbook.add_format({"font_color": "#666666", "italic": True})
    link_fmt = workbook.add_format({"font_color": "blue", "underline": 1})

    index_ws = workbook.add_worksheet("Index")
    index_ws.freeze_panes(1, 0)
    index_ws.write_row(0, 0, INDEX_FIELDS, header_fmt)
    index_rows: list[dict[str, Any]] = []

    sorted_groups = sorted(
        groups,
        key=lambda g: (
            str(g.get("province") or ""),
            str(g.get("year") or ""),
            str(g.get("table_family") or ""),
            str(g.get("title") or ""),
        ),
    )

    for seq, group in enumerate(sorted_groups, start=1):
        sheet = safe_sheet_name(seq)
        sections = group.get("sections") or []
        report = group.get("assembled_constraint_report") or {}
        row = {
            "seq": seq,
            "sheet": sheet,
            "group_id": group.get("group_id", ""),
            "province": group.get("province", ""),
            "year": group.get("year", ""),
            "table_family": group.get("table_family", ""),
            "title": group.get("title", ""),
            "n_pages": group.get("n_pages", ""),
            "overall_strategy": group.get("overall_strategy", ""),
            "signals": " | ".join(group.get("signals") or []),
            "assembled_constraint_checks": report.get("total_checks", ""),
            "assembled_constraint_failures": report.get("failed", ""),
            "assembled_all_passed": report.get("all_passed", ""),
            "section_count": len(sections),
            "member_page_count": len(group.get("page_source_jsons") or []),
        }
        index_rows.append(row)
        for col, field in enumerate(INDEX_FIELDS):
            value = row.get(field, "")
            if field == "sheet":
                index_ws.write_url(seq, col, sheet_link(sheet), link_fmt, string=str(value))
            else:
                index_ws.write(seq, col, safe_text(value))

        ws = workbook.add_worksheet(sheet)
        ws.freeze_panes(8, 0)
        ws.set_column(0, 0, 28)
        ws.set_column(1, 1, 90)
        ws.set_column(2, max_cols, 14)
        ws.write(0, 0, f"{seq}. {group.get('province', '')} {group.get('year', '')}", title_fmt)
        ws.write(1, 0, safe_text(group.get("title", "")), wrap_fmt)
        ws.write_url(2, 0, sheet_link("Index"), link_fmt, string="Back to Index")
        if seq > 1:
            ws.write_url(2, 1, sheet_link(safe_sheet_name(seq - 1)), link_fmt, string="Previous table")
        if seq < len(sorted_groups):
            ws.write_url(2, 2, sheet_link(safe_sheet_name(seq + 1)), link_fmt, string="Next table")

        r = 4
        r = write_kv(
            ws,
            r,
            [
                ("group_id", group.get("group_id", "")),
                ("table_family", group.get("table_family", "")),
                ("source_file", group.get("source_file", "")),
                ("n_pages", group.get("n_pages", "")),
                ("overall_strategy", group.get("overall_strategy", "")),
                ("signals", " | ".join(group.get("signals") or [])),
                ("constraint checks", report.get("total_checks", "")),
                ("constraint failures", report.get("failed", "")),
                ("all passed", report.get("all_passed", "")),
                ("section strategies", json.dumps(group.get("section_strategy_counts") or {}, sort_keys=True)),
            ],
            key_fmt,
            wrap_fmt,
        )
        r += 1

        members = member_rows(group, local_root)
        ws.write(r, 0, "Member pages", title_fmt)
        r += 1
        member_fields = ["member_index", "source_json", "local_image", "source_image"]
        r = write_table(ws, r, [member_fields] + [[m.get(field, "") for field in member_fields] for m in members], header_fmt, normal_fmt, max_cols)

        for section_idx, section in enumerate(sections, start=1):
            stats = section.get("stats") or {}
            rows = section.get("rows") or []
            ws.write(r, 0, f"Section {section_idx}: {section.get('name', '')}", title_fmt)
            ws.write(
                r,
                1,
                f"strategy={section.get('merge_strategy', '')}; input_pages={stats.get('input_pages', '')}; input_rows={stats.get('input_rows', '')}; output_rows={stats.get('output_rows', '')}",
                muted_fmt,
            )
            r += 1
            matrix = flatten_section_rows(rows)
            r = write_table(ws, r, matrix, header_fmt, normal_fmt, max_cols)

    index_ws.autofilter(0, 0, len(index_rows), len(INDEX_FIELDS) - 1)
    index_ws.set_column(0, 1, 10)
    index_ws.set_column(2, 5, 20)
    index_ws.set_column(6, 6, 80)
    index_ws.set_column(7, len(INDEX_FIELDS) - 1, 18)
    workbook.close()

    write_csv(output_index, index_rows, INDEX_FIELDS)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build one workbook for reviewing compiled multi-page assembled tables.")
    parser.add_argument("--assembled-jsonl", type=Path, default=DEFAULT_ASSEMBLED_JSONL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--name", default="assembled_strong_review_all")
    parser.add_argument("--local-image-root", type=Path, default=DEFAULT_DIGITIZED_LOCAL_ROOT)
    parser.add_argument("--max-cols", type=int, default=240)
    args = parser.parse_args()

    groups = load_groups(args.assembled_jsonl)
    output_xlsx = args.output_dir / f"{args.name}.xlsx"
    output_index = args.output_dir / f"{args.name}_index.csv"
    build_workbook(groups, output_xlsx, output_index, args.local_image_root, args.max_cols)
    print(f"Assembled review workbook written: {output_xlsx}")
    print(f"Index written: {output_index}")
    print(f"Groups included: {len(groups)}")


if __name__ == "__main__":
    main()
