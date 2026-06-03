#!/usr/bin/env python3
"""Build a dedicated preview pack for kept tables with failed shipped checks."""

from __future__ import annotations

import csv
import hashlib
import shutil
import sys
from pathlib import Path

from build_table_preview_pack import (
    ROOT,
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


OUT_ROOT = Path("export_final/browse/failing_tables")


def to_int(value: str) -> int:
    try:
        return int(float(value))
    except Exception:
        return 0


def write_page_level_table_folder(
    out_dir: Path,
    dataset_label: str,
    source_row: dict,
    facts_fieldnames: list[str],
    facts_rows: list[dict],
) -> dict:
    table_id = source_row.get("table_id", "")
    unique_suffix = hashlib.sha1(table_id.encode("utf-8")).hexdigest()[:10] if table_id else "noid"
    base_name = slugify(
        f"{dataset_label}_{source_row.get('province', '')}_{source_row.get('year', '')}_{source_row.get('table_family', '')}_{source_row.get('title', '')}",
        fallback="table",
    )
    folder_name = f"{base_name[:105]}_{unique_suffix}"
    table_dir = out_dir / folder_name
    table_dir.mkdir(parents=True, exist_ok=True)

    write_json(table_dir / "source_catalog_row.json", source_row)
    write_csv(table_dir / "facts.csv", facts_fieldnames, facts_rows)
    matrix_fields, matrix_rows = build_matrix(facts_rows)
    write_csv(table_dir / "matrix.csv", matrix_fields, matrix_rows)
    (table_dir / "interpretation.txt").write_text(
        "\n".join(build_interpretation_lines(source_row, facts_rows)) + "\n",
        encoding="utf-8",
    )

    summary = [
        f"dataset: {dataset_label}",
        f"title: {source_row.get('title', '')}",
        f"table_id: {source_row.get('table_id', '')}",
        f"province: {source_row.get('province', '')}",
        f"year: {source_row.get('year', '')}",
        f"table_family: {source_row.get('table_family', '')}",
        f"effective_quality_status: {source_row.get('effective_quality_status', '')}",
        f"n_checks: {source_row.get('n_checks', '')}",
        f"failed_checks: {source_row.get('failed_checks', '')}",
        f"facts_rows: {len(facts_rows)}",
        f"source_image: {source_row.get('source_image', '')}",
        f"source_json: {source_row.get('source_json', '')}",
    ]
    (table_dir / "summary.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")

    return {
        "dataset": dataset_label,
        "table_id": source_row.get("table_id", ""),
        "province": source_row.get("province", ""),
        "year": source_row.get("year", ""),
        "table_family": source_row.get("table_family", ""),
        "effective_quality_status": source_row.get("effective_quality_status", ""),
        "n_checks": source_row.get("n_checks", ""),
        "failed_checks": source_row.get("failed_checks", ""),
        "qc_flags": source_row.get("qc_flags", ""),
        "title": source_row.get("title", ""),
        "facts_rows": str(len(facts_rows)),
        "folder": str(table_dir.relative_to(OUT_ROOT)),
        "source_image": source_row.get("source_image", ""),
        "source_json": source_row.get("source_json", ""),
    }


def choose_failed_rows(source_catalog: Path, layer_name: str) -> list[dict]:
    rows = []
    for row in load_csv_rows(source_catalog):
        if not truthy(row.get("keep_for_inference")):
            continue
        if to_int(row.get("failed_checks", "")) <= 0:
            continue
        row["_layer"] = layer_name
        rows.append(row)
    rows.sort(
        key=lambda r: (
            -to_int(r.get("failed_checks", "")),
            -to_int(r.get("n_checks", "")),
            r.get("province", ""),
            r.get("year", ""),
            r.get("title", ""),
        )
    )
    return rows


def main() -> None:
    set_csv_limits()
    if OUT_ROOT.exists():
        shutil.rmtree(OUT_ROOT)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    index_rows: list[dict] = []
    all_summary_rows: list[dict] = []

    datasets = {
        "core_failed": (
            ROOT / "core_curated_v2_final" / "source_catalog.csv",
            ROOT / "core_curated_v2_final" / "cell_facts_rich.csv",
        ),
        "tail_failed": (
            ROOT / "tail_exploratory_v2" / "source_catalog.csv",
            ROOT / "tail_exploratory_v2" / "cell_facts_rich.csv",
        ),
    }

    for label, (source_catalog, facts_csv) in datasets.items():
        chosen_rows = choose_failed_rows(source_catalog, label)
        selected_ids = {row["table_id"] for row in chosen_rows}
        print(f"{label}: selected {len(chosen_rows)} failing kept tables")
        facts_fieldnames, facts_by_table = stream_selected_facts(facts_csv, selected_ids, label)
        out_dir = OUT_ROOT / label
        out_dir.mkdir(parents=True, exist_ok=True)
        for row in chosen_rows:
            summary_row = write_page_level_table_folder(
                out_dir,
                label,
                row,
                facts_fieldnames,
                facts_by_table.get(row["table_id"], []),
            )
            index_rows.append(summary_row)
            all_summary_rows.append(summary_row)

    index_rows.sort(
        key=lambda r: (
            r["dataset"],
            -to_int(r["failed_checks"]),
            -to_int(r["n_checks"]),
            r["province"],
            r["year"],
            r["title"],
        )
    )
    fieldnames = [
        "dataset",
        "table_id",
        "province",
        "year",
        "table_family",
        "effective_quality_status",
        "n_checks",
        "failed_checks",
        "qc_flags",
        "title",
        "facts_rows",
        "folder",
        "source_image",
        "source_json",
    ]
    write_csv(OUT_ROOT / "index.csv", fieldnames, index_rows)
    write_csv(OUT_ROOT / "all_failing_tables.csv", fieldnames, all_summary_rows)
    write_csv(OUT_ROOT / "top_50_worst.csv", fieldnames, index_rows[:50])

    counts_by_layer = {}
    for label in datasets:
        layer_rows = [r for r in index_rows if r["dataset"] == label]
        counts_by_layer[label] = len(layer_rows)
        write_csv(OUT_ROOT / f"{label}.csv", fieldnames, layer_rows)
        (OUT_ROOT / f"{label}_table_ids.txt").write_text(
            "\n".join(r["table_id"] for r in layer_rows) + ("\n" if layer_rows else ""),
            encoding="utf-8",
        )

    readme_lines = [
        "# Failing Tables Review Pack",
        "",
        "This directory contains only kept release tables with nonzero shipped `failed_checks`.",
        "",
        "Files:",
        "- `index.csv`: all failing kept tables",
        "- `top_50_worst.csv`: highest-failure subset",
        "- `core_failed.csv`, `tail_failed.csv`: layer-specific lists",
        "- one folder per failing table with `facts.csv`, `matrix.csv`, `source_catalog_row.json`, `interpretation.txt`, `summary.txt`",
        "",
        "Notes:",
        "- These are not all 'bad tables'. Some are close misses or proportional-tolerance issues.",
        "- Use `effective_quality_status`, `failed_checks`, and `qc_flags` to interpret severity.",
        f"- core failing kept tables: {counts_by_layer.get('core_failed', 0)}",
        f"- tail failing kept tables: {counts_by_layer.get('tail_failed', 0)}",
    ]
    (OUT_ROOT / "README.md").write_text("\n".join(readme_lines) + "\n", encoding="utf-8")

    print(f"Wrote failing review pack to {OUT_ROOT}")


if __name__ == "__main__":
    main()
