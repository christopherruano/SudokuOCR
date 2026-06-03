#!/usr/bin/env python3
"""Build a dedicated mortality subcollection inside export_final."""

from __future__ import annotations

import csv
import json
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import xlsxwriter

from build_preview_workbooks import (
    autosize,
    slug_sheet_name,
    write_key_value_lines,
    write_table,
)
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
OUT_ROOT = EXPORT_FINAL / "browse" / "by_topic" / "mortality"

DATASET_SPECS = {
    "core": {
        "catalog": DATASETS_ROOT / "core_curated_v2_final" / "source_catalog.csv",
        "facts": DATASETS_ROOT / "core_curated_v2_final" / "cell_facts_rich.csv",
    },
    "tail": {
        "catalog": DATASETS_ROOT / "tail_exploratory_v2" / "source_catalog.csv",
        "facts": DATASETS_ROOT / "tail_exploratory_v2" / "cell_facts_rich.csv",
    },
    "assembled": {
        "catalog": DATASETS_ROOT / "assembled_strong_v2" / "source_catalog.csv",
        "facts": DATASETS_ROOT / "assembled_strong_v2" / "cell_facts_rich.csv",
    },
}

RX_ANY = re.compile(r"\bmortality\b|\bdeaths?\b|\bdeath-rate\b|\bdeath rate\b", re.I)
RX_RATE = re.compile(
    r"\bdeath-rate\b|\bdeath rate\b|per mille living|female deaths per 1,000 male deaths|"
    r"deaths of females to 1,000 males|deaths per mille|age specific death rate|"
    r"standardized death rates|corrected death rates|birth rate and death rate",
    re.I,
)
RX_COUNTS = re.compile(
    r"actual number of births and deaths|number of deaths|deaths registered|"
    r"reported deaths from certain diseases|deaths of each sex at different ages|"
    r"births and deaths|deaths annually reported",
    re.I,
)
RX_MORTALITY = re.compile(r"\bmortality\b", re.I)


def classify_quality_tier(row: dict) -> str:
    status = row.get("effective_quality_status", "")
    if status in {"accepted_strict", "accepted_proportional_tolerance"}:
        return "strict_validated"
    if status == "accepted_contextual_unvalidated":
        return "contextual_unvalidated"
    return "needs_review"


def classify_mortality_kind(title: str) -> str:
    if RX_RATE.search(title):
        return "death_rate"
    if RX_COUNTS.search(title):
        return "death_counts"
    if RX_MORTALITY.search(title):
        return "mortality_table"
    return "other"


def mortality_rows() -> list[dict]:
    rows: list[dict] = []
    for layer, spec in DATASET_SPECS.items():
        for row in load_csv_rows(spec["catalog"]):
            if not truthy(row.get("keep_for_inference")):
                continue
            title = str(row.get("title", "") or "")
            if not RX_ANY.search(title):
                continue
            out = dict(row)
            out["layer"] = layer
            out["quality_tier"] = classify_quality_tier(row)
            out["mortality_kind"] = classify_mortality_kind(title)
            rows.append(out)
    rows.sort(
        key=lambda r: (
            {"strict_validated": 0, "contextual_unvalidated": 1, "needs_review": 2}[r["quality_tier"]],
            r["layer"],
            r.get("province", ""),
            r.get("year", ""),
            r.get("title", ""),
        )
    )
    return rows


def write_page_preview_folder(base_dir: Path, row: dict, facts_fieldnames: list[str], facts_rows: list[dict]) -> dict:
    folder_name = slugify(
        f"{row.get('layer','')}_{row.get('quality_tier','')}_{row.get('province','')}_{row.get('year','')}_{row.get('table_family','')}_{row.get('title','')}",
        fallback="mortality_table",
    )
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
        f"mortality_kind: {row.get('mortality_kind', '')}",
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
        "layer": row.get("layer", ""),
        "quality_tier": row.get("quality_tier", ""),
        "mortality_kind": row.get("mortality_kind", ""),
        "table_id": row.get("table_id", ""),
        "province": row.get("province", ""),
        "year": row.get("year", ""),
        "table_family": row.get("table_family", ""),
        "effective_quality_status": row.get("effective_quality_status", ""),
        "failed_checks": row.get("failed_checks", ""),
        "title": row.get("title", ""),
        "facts_rows": str(len(facts_rows)),
        "preview_folder": str(table_dir.relative_to(OUT_ROOT)),
        "source_image": row.get("source_image", ""),
        "source_json": row.get("source_json", ""),
    }


def safe_symlink(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(src.resolve())


def write_subset_csvs(rows: list[dict], base: Path, preview_lookup: dict[str, dict]) -> None:
    fieldnames = [
        "layer",
        "quality_tier",
        "mortality_kind",
        "table_id",
        "province",
        "year",
        "table_family",
        "effective_quality_status",
        "failed_checks",
        "title",
        "facts_rows",
        "preview_folder",
        "source_image",
        "source_json",
    ]
    enriched = []
    for row in rows:
        p = preview_lookup[row["table_id"]]
        enriched.append({**{k: row.get(k, "") for k in fieldnames}, **p})
    write_csv(base, fieldnames, enriched)


def build_workbook(rows: list[dict], preview_lookup: dict[str, dict], out_path: Path, title: str) -> None:
    workbook = xlsxwriter.Workbook(str(out_path))
    header_fmt = workbook.add_format({"bold": True, "bg_color": "#D9EAF7", "border": 1})
    bold_fmt = workbook.add_format({"bold": True})
    mono_fmt = workbook.add_format({"font_name": "Menlo", "font_size": 10})
    link_fmt = workbook.add_format({"font_color": "blue", "underline": 1})

    used: set[str] = set()
    index = workbook.add_worksheet("Index")
    index.write_row(0, 0, ["sheet", "layer", "tier", "kind", "province", "year", "family", "title"])

    for i, row in enumerate(rows, start=1):
        preview_rel = preview_lookup[row["table_id"]]["preview_folder"]
        folder = OUT_ROOT / preview_rel
        summary_lines = (folder / "summary.txt").read_text(encoding="utf-8").splitlines()
        interpretation_lines = (folder / "interpretation.txt").read_text(encoding="utf-8").splitlines()
        matrix_rows = load_csv_rows(folder / "matrix.csv")
        sheet_name = slug_sheet_name(
            f"{row.get('province','')}_{row.get('year','')}_{row.get('mortality_kind','')}",
            used,
        )

        index.write_url(i, 0, f"internal:'{sheet_name}'!A1", link_fmt, string=sheet_name)
        index.write(i, 1, row.get("layer", ""))
        index.write(i, 2, row.get("quality_tier", ""))
        index.write(i, 3, row.get("mortality_kind", ""))
        index.write(i, 4, row.get("province", ""))
        index.write(i, 5, row.get("year", ""))
        index.write(i, 6, row.get("table_family", ""))
        index.write(i, 7, row.get("title", ""))

        ws = workbook.add_worksheet(sheet_name)
        row_ix = 0
        ws.write(row_ix, 0, title, header_fmt)
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


def build_review_csv(rows: list[dict], preview_lookup: dict[str, dict], out_path: Path) -> None:
    fieldnames = [
        "table_id",
        "preview_folder",
        "layer",
        "quality_tier",
        "mortality_kind",
        "province",
        "year",
        "table_family",
        "effective_quality_status",
        "failed_checks",
        "title",
        "source_image",
        "source_json",
        "suggested_decision",
        "review_decision",
        "review_notes",
        "final_decision",
    ]
    review_rows = []
    for row in rows:
        suggested = "hold" if row["quality_tier"] == "needs_review" else "keep"
        review_rows.append(
            {
                "table_id": row["table_id"],
                "preview_folder": preview_lookup[row["table_id"]]["preview_folder"],
                "layer": row["layer"],
                "quality_tier": row["quality_tier"],
                "mortality_kind": row["mortality_kind"],
                "province": row.get("province", ""),
                "year": row.get("year", ""),
                "table_family": row.get("table_family", ""),
                "effective_quality_status": row.get("effective_quality_status", ""),
                "failed_checks": row.get("failed_checks", ""),
                "title": row.get("title", ""),
                "source_image": row.get("source_image", ""),
                "source_json": row.get("source_json", ""),
                "suggested_decision": suggested,
                "review_decision": "",
                "review_notes": "",
                "final_decision": "",
            }
        )
    write_csv(out_path, fieldnames, review_rows)


def build() -> None:
    set_csv_limits()
    csv.field_size_limit(sys.maxsize)
    rows = mortality_rows()

    if OUT_ROOT.exists():
        shutil.rmtree(OUT_ROOT)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    preview_tables = OUT_ROOT / "previews" / "tables"
    preview_tables.mkdir(parents=True, exist_ok=True)

    preview_lookup: dict[str, dict] = {}

    for layer, spec in DATASET_SPECS.items():
        layer_rows = [r for r in rows if r["layer"] == layer]
        if not layer_rows:
            continue
        selected_ids = {r["table_id"] for r in layer_rows}
        facts_fieldnames, facts_by_table = stream_selected_facts(spec["facts"], selected_ids, f"mortality_{layer}")
        for row in layer_rows:
            preview_lookup[row["table_id"]] = write_page_preview_folder(
                preview_tables,
                row,
                facts_fieldnames,
                facts_by_table.get(row["table_id"], []),
            )

    # Main catalogs.
    write_subset_csvs(rows, OUT_ROOT / "all_tables.csv", preview_lookup)
    for tier in ["strict_validated", "contextual_unvalidated", "needs_review"]:
        subset = [r for r in rows if r["quality_tier"] == tier]
        write_subset_csvs(subset, OUT_ROOT / f"{tier}.csv", preview_lookup)
    for kind in ["death_rate", "death_counts", "mortality_table", "other"]:
        subset = [r for r in rows if r["mortality_kind"] == kind]
        write_subset_csvs(subset, OUT_ROOT / f"{kind}.csv", preview_lookup)

    # Symlink views.
    for group_name, values in {
        "by_quality": ["strict_validated", "contextual_unvalidated", "needs_review"],
        "by_kind": ["death_rate", "death_counts", "mortality_table", "other"],
    }.items():
        for value in values:
            links_dir = OUT_ROOT / "preview_links" / group_name / value
            links_dir.mkdir(parents=True, exist_ok=True)
            subset = [r for r in rows if r.get("quality_tier") == value or r.get("mortality_kind") == value]
            for row in subset:
                src = OUT_ROOT / preview_lookup[row["table_id"]]["preview_folder"]
                safe_symlink(src, links_dir / src.name)

    # Review queue for the non-strict slice.
    review_rows = [r for r in rows if r["quality_tier"] in {"contextual_unvalidated", "needs_review"}]
    write_subset_csvs(review_rows, OUT_ROOT / "review_queue.csv", preview_lookup)
    build_review_csv(review_rows, preview_lookup, OUT_ROOT / "review_decisions.csv")
    review_links = OUT_ROOT / "preview_links" / "review_queue"
    review_links.mkdir(parents=True, exist_ok=True)
    for row in review_rows:
        src = OUT_ROOT / preview_lookup[row["table_id"]]["preview_folder"]
        safe_symlink(src, review_links / src.name)

    # Workbooks.
    build_workbook(rows, preview_lookup, OUT_ROOT / "mortality_all.xlsx", "Mortality Subcollection")
    build_workbook(review_rows, preview_lookup, OUT_ROOT / "mortality_review_queue.xlsx", "Mortality Review Queue")

    # Summary and readme.
    summary = {
        "total_tables": len(rows),
        "by_layer": dict(Counter(r["layer"] for r in rows)),
        "by_quality_tier": dict(Counter(r["quality_tier"] for r in rows)),
        "by_mortality_kind": dict(Counter(r["mortality_kind"] for r in rows)),
        "review_queue": len(review_rows),
    }
    (OUT_ROOT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    readme = [
        "# Mortality Subcollection",
        "",
        "Dedicated browseable slice for death-rate, deaths, and mortality-titled tables in the final release.",
        "",
        "Files:",
        "- `all_tables.csv`: every mortality-like kept table in the release",
        "- `strict_validated.csv`: strong mortality subset",
        "- `contextual_unvalidated.csv`: usable but not fully proven subset",
        "- `needs_review.csv`: mortality tables that still need explicit review",
        "- `death_rate.csv`, `death_counts.csv`, `mortality_table.csv`, `other.csv`: semantic splits",
        "- `review_queue.csv`: non-strict mortality tables that are worth human review",
        "- `review_decisions.csv`: editable decision sheet for the local review app",
        "- `mortality_all.xlsx`: browse workbook for the whole subcollection",
        "- `mortality_review_queue.xlsx`: smaller workbook for the review subset",
        "",
        "Preview folders live under `previews/tables/` and are linked through `preview_links/`.",
        "",
        "Counts:",
        f"- total: {summary['total_tables']}",
        f"- strict_validated: {summary['by_quality_tier'].get('strict_validated', 0)}",
        f"- contextual_unvalidated: {summary['by_quality_tier'].get('contextual_unvalidated', 0)}",
        f"- needs_review: {summary['by_quality_tier'].get('needs_review', 0)}",
        f"- review_queue: {summary['review_queue']}",
    ]
    (OUT_ROOT / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")


if __name__ == "__main__":
    build()
