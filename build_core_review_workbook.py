"""Build an Excel workbook for local manual review of core_review rows.

The workbook is designed for the current local bundle workflow:
  - review the original image
  - read the OCR output preview
  - set review_decision / review_notes in Excel
  - import the edited workbook back into core_review_decisions.csv

Expected flow:
  1. sync_core_review_assets.sh
  2. build_core_review_workbook.py
  3. edit the workbook
  4. import_core_review_workbook.py
  5. apply_core_review_decisions.py --apply
"""

from __future__ import annotations

import argparse
import csv
import json
import textwrap
from pathlib import Path
from tempfile import TemporaryDirectory

import xlsxwriter
from PIL import Image


DEFAULT_DECISIONS_CSV = "export/local_research_bundle/final_core_prep/core_review_decisions.csv"
DEFAULT_ASSET_ROOT = "export/local_research_bundle/final_core_prep/review_assets"
DEFAULT_OUTPUT_XLSX = "export/local_research_bundle/final_core_prep/core_review_workbook.xlsx"


def normalize_text(value: object) -> str:
    text = str(value or "").strip()
    return " ".join(text.split())


def remote_to_local_path(remote_path: str, remote_root: Path, local_asset_root: Path) -> Path:
    path = Path(str(remote_path or "").strip())
    try:
        rel = path.relative_to(remote_root)
        return local_asset_root / rel
    except Exception:
        return local_asset_root / path.name


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def write_summary_sheet(workbook, rows, output_path: Path):
    sheet = workbook.add_worksheet("Summary")
    title_fmt = workbook.add_format({"bold": True, "font_size": 16})
    header_fmt = workbook.add_format({"bold": True, "bg_color": "#EFE3CA", "border": 1})
    text_fmt = workbook.add_format({"text_wrap": True, "valign": "top"})

    counts = {}
    for key in ("suggested_decision", "review_decision", "final_decision", "priority_bucket"):
        vals = {}
        for row in rows:
            value = normalize_text(row.get(key, "")) or "(blank)"
            vals[value] = vals.get(value, 0) + 1
        counts[key] = vals

    sheet.write("A1", "Core Review Workbook", title_fmt)
    sheet.write("A3", "Workbook path")
    sheet.write("B3", str(output_path))
    sheet.write("A5", "How to use", header_fmt)
    instructions = (
        "1. Review the image thumbnail and use the local image/JSON hyperlinks for full detail.\n"
        "2. Set review_decision to promote, reject, or hold.\n"
        "3. Enter review_notes when a decision needs justification or a fix description.\n"
        "4. Save the workbook, then run import_core_review_workbook.py.\n"
        "5. After import, run apply_core_review_decisions.py --apply to build the curated final list."
    )
    sheet.write("A6", instructions, text_fmt)
    sheet.set_row(5, 86)

    row_idx = 8
    for label, vals in counts.items():
        sheet.write(row_idx, 0, label, header_fmt)
        row_idx += 1
        sheet.write_row(row_idx, 0, ["value", "count"], header_fmt)
        row_idx += 1
        for key, count in sorted(vals.items()):
            sheet.write(row_idx, 0, key)
            sheet.write(row_idx, 1, count)
            row_idx += 1
        row_idx += 1

    sheet.set_column("A:A", 24)
    sheet.set_column("B:B", 18)


def flatten_nested_dict(data: dict) -> str:
    parts = []
    for key, value in data.items():
        if isinstance(value, dict):
            inner = ", ".join(f"{normalize_text(k)}={v}" for k, v in list(value.items())[:8])
            parts.append(f"{normalize_text(key)}[{inner}]")
        else:
            parts.append(f"{normalize_text(key)}={value}")
    return "; ".join(parts)


def build_ocr_preview(result_json_path: Path, max_sections: int = 3, max_rows_per_section: int = 6) -> str:
    if not result_json_path.exists():
        return "(result JSON not synced locally)"
    try:
        obj = json.load(result_json_path.open(encoding="utf-8"))
    except Exception as exc:
        return f"(could not read JSON: {exc})"

    parts = []
    title = normalize_text(obj.get("title") or "")
    if title:
        parts.append(f"TITLE: {title}")

    acceptance = obj.get("acceptance") or {}
    decision = normalize_text(acceptance.get("decision") or "")
    if decision:
        parts.append(f"ACCEPTANCE: {decision}")

    sections = obj.get("data") or []
    if not isinstance(sections, list):
        return "\n".join(parts + [json.dumps(sections, ensure_ascii=True)[:1500]])

    for section in sections[:max_sections]:
        section_name = normalize_text(section.get("name") or "Section")
        parts.append(f"\nSECTION: {section_name}")
        rows = section.get("rows") or []
        for row in rows[:max_rows_per_section]:
            if isinstance(row, dict):
                parts.append(flatten_nested_dict(row))
            else:
                parts.append(str(row))
        if len(rows) > max_rows_per_section:
            parts.append(f"... {len(rows) - max_rows_per_section} more rows")
    return "\n".join(parts)[:12000]


def make_thumbnail(src: Path, dest: Path, max_size=(240, 320)) -> bool:
    if not src.exists():
        return False
    try:
        with Image.open(src) as img:
            img = img.convert("RGB")
            img.thumbnail(max_size)
            dest.parent.mkdir(parents=True, exist_ok=True)
            img.save(dest, format="PNG")
        return True
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser(description="Build an Excel workbook for core_review manual review.")
    parser.add_argument("--decisions-csv", default=DEFAULT_DECISIONS_CSV)
    parser.add_argument("--remote-root", default="/n/home11/cruano/census_toc")
    parser.add_argument("--local-asset-root", default=DEFAULT_ASSET_ROOT)
    parser.add_argument("--output-xlsx", default=DEFAULT_OUTPUT_XLSX)
    args = parser.parse_args()

    decisions_csv = Path(args.decisions_csv).expanduser()
    remote_root = Path(args.remote_root).expanduser()
    local_asset_root = Path(args.local_asset_root).expanduser()
    output_xlsx = Path(args.output_xlsx).expanduser()
    output_xlsx.parent.mkdir(parents=True, exist_ok=True)

    rows = read_rows(decisions_csv)

    workbook = xlsxwriter.Workbook(str(output_xlsx))
    sheet = workbook.add_worksheet("Review")
    write_summary_sheet(workbook, rows, output_xlsx)

    header_fmt = workbook.add_format({"bold": True, "bg_color": "#EFE3CA", "border": 1, "text_wrap": True, "valign": "top"})
    wrap_fmt = workbook.add_format({"text_wrap": True, "valign": "top", "border": 1})
    note_fmt = workbook.add_format({"text_wrap": True, "valign": "top", "border": 1, "bg_color": "#FFFDF8"})
    hidden_fmt = workbook.add_format({"font_color": "#888888", "border": 1})
    decision_fmt = workbook.add_format({"border": 1, "valign": "top"})

    headers = [
        "row_id",
        "thumbnail",
        "review_decision",
        "review_notes",
        "suggested_decision",
        "province",
        "year",
        "title_from_payload",
        "rescue_family",
        "canonical_topic",
        "rescue_score",
        "priority_bucket",
        "priority_reasons",
        "ocr_preview",
        "column_groups_from_payload",
        "age_row_samples",
        "row_axis_type",
        "row_label_name",
        "nonnull_numeric_cells",
        "remote_image_path",
        "local_image_path",
        "remote_result_json",
        "local_result_json",
    ]

    for col_idx, header in enumerate(headers):
        sheet.write(0, col_idx, header, header_fmt)

    widths = {
        "A:A": 10,
        "B:B": 34,
        "C:C": 14,
        "D:D": 32,
        "E:E": 16,
        "F:F": 18,
        "G:G": 8,
        "H:H": 48,
        "I:I": 24,
        "J:J": 20,
        "K:K": 10,
        "L:M": 20,
        "N:N": 90,
        "O:O": 48,
        "P:P": 32,
        "Q:R": 18,
        "S:V": 40,
    }
    for key, value in widths.items():
        sheet.set_column(key, value)

    sheet.freeze_panes(1, 2)
    sheet.autofilter(0, 0, max(len(rows), 1), len(headers) - 1)
    sheet.data_validation(1, 2, max(len(rows), 1), 2, {"validate": "list", "source": ["promote", "reject", "hold"]})

    # Hide the path columns but keep them in the workbook for round-trip import.
    sheet.set_column("T:W", None, hidden_fmt, {"hidden": True})

    with TemporaryDirectory() as tmpdir:
        thumb_root = Path(tmpdir)
        for row_idx, row in enumerate(rows, start=1):
            remote_img = Path(row.get("image_path") or "")
            remote_json = Path(row.get("result_json") or "")
            local_img = remote_to_local_path(str(remote_img), remote_root, local_asset_root)
            local_json = remote_to_local_path(str(remote_json), remote_root, local_asset_root)

            preview = build_ocr_preview(local_json)

            row_values = {
                "row_id": row_idx,
                "review_decision": normalize_text(row.get("review_decision") or ""),
                "review_notes": row.get("review_notes") or "",
                "suggested_decision": normalize_text(row.get("suggested_decision") or ""),
                "province": row.get("province") or "",
                "year": row.get("year") or "",
                "title_from_payload": row.get("title_from_payload") or row.get("title") or "",
                "rescue_family": row.get("rescue_family") or "",
                "canonical_topic": row.get("canonical_topic") or "",
                "rescue_score": row.get("rescue_score") or "",
                "priority_bucket": row.get("priority_bucket") or "",
                "priority_reasons": row.get("priority_reasons") or "",
                "ocr_preview": preview,
                "column_groups_from_payload": row.get("column_groups_from_payload") or "",
                "age_row_samples": row.get("age_row_samples") or "",
                "row_axis_type": row.get("row_axis_type") or "",
                "row_label_name": row.get("row_label_name") or "",
                "nonnull_numeric_cells": row.get("nonnull_numeric_cells") or "",
                "remote_image_path": str(remote_img),
                "local_image_path": str(local_img),
                "remote_result_json": str(remote_json),
                "local_result_json": str(local_json),
            }

            sheet.set_row(row_idx, 180)
            sheet.write(row_idx, 0, row_values["row_id"], wrap_fmt)
            sheet.write(row_idx, 2, row_values["review_decision"], decision_fmt)
            sheet.write(row_idx, 3, row_values["review_notes"], note_fmt)
            for col_idx, header in enumerate(headers[4:], start=4):
                fmt = wrap_fmt
                sheet.write(row_idx, col_idx, row_values[header], fmt)

            if local_img.exists():
                sheet.write_url(row_idx, 20, f"external:{local_img}", string=str(local_img))
                thumb_path = thumb_root / f"thumb_{row_idx}.png"
                if make_thumbnail(local_img, thumb_path):
                    sheet.insert_image(
                        row_idx,
                        1,
                        str(thumb_path),
                        {
                            "x_scale": 1.0,
                            "y_scale": 1.0,
                            "x_offset": 4,
                            "y_offset": 4,
                            "object_position": 1,
                            "url": f"external:{local_img}",
                        },
                    )
                else:
                    sheet.write(row_idx, 1, "(image unreadable)", wrap_fmt)
            else:
                sheet.write(row_idx, 1, "(image not synced locally)", wrap_fmt)

            if local_json.exists():
                sheet.write_url(row_idx, 22, f"external:{local_json}", string=str(local_json))
            else:
                sheet.write(row_idx, 22, str(local_json), wrap_fmt)

        workbook.close()
    print(f"Wrote workbook: {output_xlsx}")


if __name__ == "__main__":
    main()
