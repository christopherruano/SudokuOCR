#!/usr/bin/env python3
"""Build a full final-submission review queue for all `needs_review` tables."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

import xlsxwriter

from build_preview_workbooks import autosize, slug_sheet_name, write_key_value_lines, write_table
from build_table_preview_pack import (
    build_interpretation_lines,
    build_matrix,
    load_csv_rows,
    set_csv_limits,
    slugify,
    stream_selected_facts,
    truthy,
    write_csv,
    write_json,
)


EXPORT_FINAL = Path("/Users/chrisruano/Research/SudokuOCR/export_final")
DATASETS_ROOT = EXPORT_FINAL / "datasets"
OUT_ROOT = EXPORT_FINAL / "browse" / "final_submission_review"
NEEDS_REVIEW_CSV = EXPORT_FINAL / "browse" / "by_quality" / "needs_review" / "tables.csv"
FAILING_ASSESSMENT_CSV = EXPORT_FINAL / "browse" / "failing_tables" / "assistant_assessment.csv"

DATASET_SPECS = {
    "core": {
        "catalog": DATASETS_ROOT / "core_curated_v2_final" / "source_catalog.csv",
        "facts": DATASETS_ROOT / "core_curated_v2_final" / "cell_facts_rich.csv",
    },
    "tail": {
        "catalog": DATASETS_ROOT / "tail_exploratory_v2" / "source_catalog.csv",
        "facts": DATASETS_ROOT / "tail_exploratory_v2" / "cell_facts_rich.csv",
    },
}


def to_int(value: str) -> int:
    try:
        return int(float(value))
    except Exception:
        return 0


def load_catalog_index(path: Path) -> dict[str, dict]:
    rows = load_csv_rows(path)
    return {row.get("table_id", ""): row for row in rows}


def load_assistant_suggestions() -> dict[str, str]:
    if not FAILING_ASSESSMENT_CSV.exists():
        return {}
    out: dict[str, str] = {}
    for row in load_csv_rows(FAILING_ASSESSMENT_CSV):
        action = row.get("recommended_action", "")
        if action == "drop":
            out[row["table_id"]] = "drop"
        elif action == "keep_with_caution":
            out[row["table_id"]] = "keep"
        else:
            out[row["table_id"]] = "hold"
    return out


def write_page_preview_folder(base_dir: Path, row: dict, facts_fieldnames: list[str], facts_rows: list[dict]) -> dict:
    table_id = row.get("table_id", "")
    suffix = hashlib.sha1(table_id.encode("utf-8")).hexdigest()[:10] if table_id else "noid"
    base_name = slugify(
        f"{row.get('layer','')}_{row.get('province','')}_{row.get('year','')}_{row.get('table_family','')}_{row.get('title','')}",
        fallback="review_table",
    )
    folder_name = f"{base_name[:105]}_{suffix}"
    table_dir = base_dir / folder_name
    table_dir.mkdir(parents=True, exist_ok=True)

    write_json(table_dir / "source_catalog_row.json", row)
    write_csv(table_dir / "facts.csv", facts_fieldnames, facts_rows)
    matrix_fields, matrix_rows = build_matrix(facts_rows)
    write_csv(table_dir / "matrix.csv", matrix_fields, matrix_rows)
    (table_dir / "interpretation.txt").write_text(
        "\n".join(build_interpretation_lines(row, facts_rows)) + "\n",
        encoding="utf-8",
    )

    summary = [
        f"layer: {row.get('layer', '')}",
        f"quality_tier: {row.get('quality_tier', '')}",
        f"title: {row.get('title', '')}",
        f"table_id: {row.get('table_id', '')}",
        f"province: {row.get('province', '')}",
        f"year: {row.get('year', '')}",
        f"table_family: {row.get('table_family', '')}",
        f"effective_quality_status: {row.get('effective_quality_status', '')}",
        f"n_checks: {row.get('n_checks', '')}",
        f"failed_checks: {row.get('failed_checks', '')}",
        f"facts_rows: {len(facts_rows)}",
        f"source_image: {row.get('source_image', '')}",
        f"source_json: {row.get('source_json', '')}",
    ]
    (table_dir / "summary.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")

    return {
        "preview_folder": str(table_dir.relative_to(OUT_ROOT)),
        "facts_rows": str(len(facts_rows)),
    }


def build_workbook(rows: list[dict], out_path: Path) -> None:
    workbook = xlsxwriter.Workbook(str(out_path))
    header_fmt = workbook.add_format({"bold": True, "bg_color": "#D9EAF7", "border": 1})
    bold_fmt = workbook.add_format({"bold": True})
    mono_fmt = workbook.add_format({"font_name": "Menlo", "font_size": 10})
    link_fmt = workbook.add_format({"font_color": "blue", "underline": 1})

    used: set[str] = set()
    index = workbook.add_worksheet("Index")
    index.write_row(0, 0, ["sheet", "layer", "province", "year", "family", "status", "failed_checks", "title"])

    for i, row in enumerate(rows, start=1):
        folder = OUT_ROOT / row["preview_folder"]
        summary_lines = (folder / "summary.txt").read_text(encoding="utf-8").splitlines()
        interpretation_lines = (folder / "interpretation.txt").read_text(encoding="utf-8").splitlines()
        matrix_rows = load_csv_rows(folder / "matrix.csv")
        sheet_name = slug_sheet_name(
            f"{row.get('province','')}_{row.get('year','')}_{row.get('table_family','')}",
            used,
        )

        index.write_url(i, 0, f"internal:'{sheet_name}'!A1", link_fmt, string=sheet_name)
        index.write(i, 1, row.get("layer", ""))
        index.write(i, 2, row.get("province", ""))
        index.write(i, 3, row.get("year", ""))
        index.write(i, 4, row.get("table_family", ""))
        index.write(i, 5, row.get("effective_quality_status", ""))
        index.write(i, 6, row.get("failed_checks", ""))
        index.write(i, 7, row.get("title", ""))

        ws = workbook.add_worksheet(sheet_name)
        row_ix = 0
        ws.write(row_ix, 0, "Final Submission Review Queue", header_fmt)
        row_ix += 1
        row_ix = write_key_value_lines(ws, row_ix, summary_lines, bold_fmt, mono_fmt) + 1
        ws.write(row_ix, 0, "Interpretation", header_fmt)
        row_ix += 1
        row_ix = write_key_value_lines(ws, row_ix, interpretation_lines, bold_fmt, mono_fmt) + 1
        ws.write(row_ix, 0, "Matrix View", header_fmt)
        row_ix += 1
        row_ix = write_table(ws, row_ix, matrix_rows, header_fmt, mono_fmt)
        autosize(ws, [list(r.keys()) for r in matrix_rows[:1]] + [list(r.values()) for r in matrix_rows[:25]])

    workbook.close()


def build() -> None:
    set_csv_limits()
    csv.field_size_limit(sys.maxsize)
    needs_rows = load_csv_rows(NEEDS_REVIEW_CSV)
    catalog_index = {}
    for layer, spec in DATASET_SPECS.items():
        catalog_index[layer] = load_catalog_index(spec["catalog"])
    assistant_suggestions = load_assistant_suggestions()

    if OUT_ROOT.exists():
        shutil.rmtree(OUT_ROOT)
    (OUT_ROOT / "previews" / "tables").mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for row in needs_rows:
        layer = row.get("layer", "")
        table_id = row.get("table_id", "")
        source_row = catalog_index[layer].get(table_id, {})
        merged = dict(source_row)
        merged.update(row)
        merged["source_image"] = source_row.get("source_image", "")
        merged["source_json"] = source_row.get("source_json", "")
        suggested = assistant_suggestions.get(table_id, "hold")
        merged["suggested_decision"] = suggested
        rows.append(merged)

    rows.sort(
        key=lambda r: (
            {"drop": 0, "hold": 1, "keep": 2}.get(r["suggested_decision"], 1),
            {"rejected_constraint_failures": 0, "review_proportional_failures": 1, "review_unvalidated_counts": 2}.get(r.get("effective_quality_status", ""), 9),
            -to_int(r.get("failed_checks", "")),
            r.get("layer", ""),
            r.get("province", ""),
            r.get("year", ""),
            r.get("title", ""),
        )
    )

    preview_lookup: dict[str, dict] = {}
    for layer, spec in DATASET_SPECS.items():
        layer_rows = [r for r in rows if r["layer"] == layer]
        selected_ids = {r["table_id"] for r in layer_rows}
        facts_fieldnames, facts_by_table = stream_selected_facts(spec["facts"], selected_ids, f"submission_{layer}")
        for row in layer_rows:
            preview_lookup[row["table_id"]] = write_page_preview_folder(
                OUT_ROOT / "previews" / "tables",
                row,
                facts_fieldnames,
                facts_by_table.get(row["table_id"], []),
            )

    fieldnames = [
        "table_id",
        "preview_folder",
        "layer",
        "quality_tier",
        "province",
        "year",
        "table_family",
        "effective_quality_status",
        "n_checks",
        "failed_checks",
        "qc_flags",
        "title",
        "source_image",
        "source_json",
        "suggested_decision",
        "review_decision",
        "review_notes",
        "final_decision",
        "facts_rows",
    ]
    review_rows = []
    for row in rows:
        preview = preview_lookup[row["table_id"]]
        review_rows.append(
            {
                "table_id": row.get("table_id", ""),
                "preview_folder": preview["preview_folder"],
                "layer": row.get("layer", ""),
                "quality_tier": row.get("quality_tier", ""),
                "province": row.get("province", ""),
                "year": row.get("year", ""),
                "table_family": row.get("table_family", ""),
                "effective_quality_status": row.get("effective_quality_status", ""),
                "n_checks": row.get("n_checks", ""),
                "failed_checks": row.get("failed_checks", ""),
                "qc_flags": row.get("qc_flags", ""),
                "title": row.get("title", ""),
                "source_image": row.get("source_image", ""),
                "source_json": row.get("source_json", ""),
                "suggested_decision": row.get("suggested_decision", "hold"),
                "review_decision": "",
                "review_notes": "",
                "final_decision": "",
                "facts_rows": preview["facts_rows"],
            }
        )

    write_csv(OUT_ROOT / "review_decisions.csv", fieldnames, review_rows)
    write_csv(OUT_ROOT / "all_tables.csv", fieldnames, review_rows)
    write_csv(OUT_ROOT / "core_tables.csv", fieldnames, [r for r in review_rows if r["layer"] == "core"])
    write_csv(OUT_ROOT / "tail_tables.csv", fieldnames, [r for r in review_rows if r["layer"] == "tail"])
    for decision in ["drop", "hold", "keep"]:
        write_csv(OUT_ROOT / f"suggested_{decision}.csv", fieldnames, [r for r in review_rows if r["suggested_decision"] == decision])

    build_workbook(review_rows, OUT_ROOT / "final_submission_review.xlsx")

    summary = {
        "total_tables": len(review_rows),
        "by_layer": dict(Counter(r["layer"] for r in review_rows)),
        "by_effective_quality_status": dict(Counter(r["effective_quality_status"] for r in review_rows)),
        "by_suggested_decision": dict(Counter(r["suggested_decision"] for r in review_rows)),
    }
    (OUT_ROOT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    readme = [
        "# Final Submission Review Queue",
        "",
        "This queue contains every table currently in `browse/by_quality/needs_review`.",
        "",
        "Files:",
        "- `review_decisions.csv`: canonical editable review sheet",
        "- `all_tables.csv`: same rows, for reference",
        "- `core_tables.csv`, `tail_tables.csv`: layer-specific subsets",
        "- `suggested_drop.csv`, `suggested_hold.csv`, `suggested_keep.csv`: assistant-seeded buckets",
        "- `final_submission_review.xlsx`: browse workbook",
        "- `previews/tables/`: one preview folder per table",
        "",
        "Decision semantics:",
        "- `keep`: include in the final project-facing submission",
        "- `drop`: exclude from the final project-facing submission",
        "- `hold`: unresolved / not ready to include",
        "",
        "Suggested decisions are pre-seeded from the assistant failing-table triage where available.",
        "",
        f"Total tables: {len(review_rows)}",
        f"Core: {summary['by_layer'].get('core', 0)}",
        f"Tail: {summary['by_layer'].get('tail', 0)}",
    ]
    (OUT_ROOT / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")


if __name__ == "__main__":
    build()
