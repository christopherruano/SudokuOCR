#!/usr/bin/env python3
"""Build a lightweight, browsable preview pack for the final release."""

from __future__ import annotations

import csv
import json
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, List, Tuple


ROOT = Path("/Users/chrisruano/Research/SudokuOCR/export/final_project_exports")
CLUSTER_SYNC = Path("/Users/chrisruano/Research/SudokuOCR/export/cluster_sync")
PREVIEW_DIR = ROOT / "preview_pack"


def set_csv_limits() -> None:
    csv.field_size_limit(100_000_000)


def truthy(value: str) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes"}


def to_int(value: str) -> int:
    try:
        return int(float(value))
    except Exception:
        return 0


def slugify(text: str, fallback: str = "table") -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text or "").strip("_")
    return text[:120] or fallback


def first_nonblank(row: dict, keys: Iterable[str]) -> str:
    for key in keys:
        value = str(row.get(key, "")).strip()
        if value:
            return value
    return ""


def load_csv_rows(path: Path) -> List[dict]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def write_csv(path: Path, fieldnames: List[str], rows: Iterable[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path: Path, obj: object) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def choose_page_level_examples(source_catalog: Path, dataset_name: str) -> List[dict]:
    rows = [r for r in load_csv_rows(source_catalog) if truthy(r.get("keep_for_inference"))]
    if dataset_name == "core":
        rows = [r for r in rows if r.get("effective_quality_status") == "accepted_strict"]

    rows.sort(
        key=lambda r: (
            1 if r.get("effective_quality_status") == "accepted_strict" else 0,
            to_int(r.get("trust_score")),
            to_int(r.get("numeric_cells")),
            r.get("title", ""),
        ),
        reverse=True,
    )
    chosen: List[dict] = []
    seen_family: set[str] = set()
    for row in rows:
        family = row.get("table_family", "") or "unknown"
        if family in seen_family:
            continue
        seen_family.add(family)
        chosen.append(row)
    return chosen


def load_assembled_keep_ids() -> set[str]:
    path = ROOT / "overlays" / "assembled_keep_table_ids.txt"
    if not path.exists():
        return set()
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def choose_assembled_rows(source_catalog: Path) -> List[dict]:
    keep_ids = load_assembled_keep_ids()
    rows = [r for r in load_csv_rows(source_catalog) if truthy(r.get("keep_for_inference"))]
    if keep_ids:
        rows = [r for r in rows if r.get("table_id") in keep_ids]
    rows.sort(
        key=lambda r: (
            to_int(r.get("assembled_failure_improvement")),
            to_int(r.get("compiled_member_count")),
            r.get("title", ""),
        ),
        reverse=True,
    )
    return rows


def row_label(row: dict) -> str:
    return first_nonblank(
        row,
        ["row_group_value", "row_label_std", "row_label_raw", "resolved_age_label_std", "geo_name"],
    ) or "row"


def combined_column_label(row: dict) -> str:
    parts = []
    base = str(row.get("column_group_value", "")).strip()
    if base:
        parts.append(base)

    # Prefer the raw/standardized subcolumn label over inferred semantic axes
    # in previews. It is more honest for tricky families like
    # `age_sex_marital_status`, where generic inference can over-interpret the
    # subcolumns.
    sub = str(row.get("sub_column_std", "")).strip()
    if sub and sub.lower() not in {"", "none"}:
        parts.append(sub)
    else:
        for key in ["civil_condition", "religion", "sex"]:
            value = str(row.get(key, "")).strip()
            if value and value not in {"All"}:
                parts.append(value)
    return " | ".join(parts) or "value"


def build_interpretation_lines(source_row: dict, facts_rows: List[dict]) -> List[str]:
    sections = sorted({str(r.get("section", "")).strip() for r in facts_rows if str(r.get("section", "")).strip()})
    column_groups = sorted({str(r.get("column_group_value", "")).strip() for r in facts_rows if str(r.get("column_group_value", "")).strip()})
    subcolumns = sorted({str(r.get("sub_column_std", "")).strip() for r in facts_rows if str(r.get("sub_column_std", "")).strip()})
    sex_values = sorted({str(r.get("sex", "")).strip() for r in facts_rows if str(r.get("sex", "")).strip()})
    civil_values = sorted({str(r.get("civil_condition", "")).strip() for r in facts_rows if str(r.get("civil_condition", "")).strip()})
    religion_values = sorted({str(r.get("religion", "")).strip() for r in facts_rows if str(r.get("religion", "")).strip()})

    lines = [
        f"table_family: {source_row.get('table_family', '')}",
        f"row_axis: {source_row.get('row_label_name', '') or 'row labels'}",
        f"section_values: {' | '.join(sections[:20]) if sections else '(none)'}",
        f"column_groups: {' | '.join(column_groups[:20]) if column_groups else '(none)'}",
        f"subcolumns: {' | '.join(subcolumns[:20]) if subcolumns else '(none)'}",
        f"sex_values: {' | '.join(sex_values[:20]) if sex_values else '(none)'}",
        f"civil_condition_values: {' | '.join(civil_values[:20]) if civil_values else '(none)'}",
        f"religion_values: {' | '.join(religion_values[:20]) if religion_values else '(none)'}",
    ]

    table_family = source_row.get("table_family", "")
    if table_family == "age_sex_marital_status":
        lines.append(
            "caution: this family is easy to misread in a flat matrix. The section often already encodes sex/religion, "
            "while the three within-year subcolumns are the distribution buckets that sum to 1,000. Treat the raw "
            "subcolumn labels and the section context as authoritative; do not read the preview matrix as a polished "
            "printed reconstruction."
        )
    if sections and any("MALES" in s or "FEMALES" in s for s in sections) and {"All", "Male", "Female"} <= set(sex_values):
        lines.append(
            "caution: section labels already carry a sex axis, so any additional All/Male/Female decomposition in the "
            "preview should be treated cautiously. Use facts.csv plus section context for interpretation."
        )
    return lines


def build_matrix(rows: List[dict]) -> Tuple[List[str], List[dict]]:
    ordered_row_keys: List[Tuple[str, str]] = []
    ordered_col_labels: List[str] = []
    row_seen: set[Tuple[str, str]] = set()
    col_seen: set[str] = set()
    matrix: Dict[Tuple[str, str], Dict[str, str]] = defaultdict(dict)

    for row in rows:
        section = str(row.get("section", "")).strip()
        rlabel = row_label(row)
        rkey = (section, rlabel)
        clabel = combined_column_label(row)
        if rkey not in row_seen:
            row_seen.add(rkey)
            ordered_row_keys.append(rkey)
        if clabel not in col_seen:
            col_seen.add(clabel)
            ordered_col_labels.append(clabel)
        matrix[rkey][clabel] = row.get("value", "")

    fieldnames = ["section", "row_label"] + ordered_col_labels
    out_rows: List[dict] = []
    for section, rlabel in ordered_row_keys:
        out = {"section": section, "row_label": rlabel}
        for clabel in ordered_col_labels:
            out[clabel] = matrix[(section, rlabel)].get(clabel, "")
        out_rows.append(out)
    return fieldnames, out_rows


def stream_selected_facts(facts_csv: Path, selected_ids: set[str], label: str) -> Tuple[List[str], Dict[str, List[dict]]]:
    rows_by_table: DefaultDict[str, List[dict]] = defaultdict(list)
    with facts_csv.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fieldnames = list(reader.fieldnames or [])
        for i, row in enumerate(reader, start=1):
            tid = row.get("table_id", "")
            if tid in selected_ids:
                rows_by_table[tid].append(row)
            if i % 200000 == 0:
                print(f"{label}: scanned {i:,} fact rows")
    return fieldnames, rows_by_table


def load_member_pages(member_pages_csv: Path, selected_ids: set[str]) -> Dict[str, List[dict]]:
    rows_by_table: DefaultDict[str, List[dict]] = defaultdict(list)
    with member_pages_csv.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            tid = row.get("table_id", "")
            if tid in selected_ids:
                rows_by_table[tid].append(row)
    return rows_by_table


def load_assembled_groups(assembled_jsonl: Path, selected_group_ids: set[str]) -> Dict[str, dict]:
    groups: Dict[str, dict] = {}
    with assembled_jsonl.open(encoding="utf-8") as fh:
        for i, line in enumerate(fh, start=1):
            obj = json.loads(line)
            gid = obj.get("group_id", "")
            if gid in selected_group_ids:
                groups[gid] = obj
            if i % 10 == 0:
                print(f"assembled: scanned {i} groups")
    return groups


def write_page_level_table_folder(
    out_dir: Path,
    dataset_label: str,
    source_row: dict,
    facts_fieldnames: List[str],
    facts_rows: List[dict],
) -> dict:
    folder_name = slugify(
        f"{dataset_label}_{source_row.get('province', '')}_{source_row.get('year', '')}_{source_row.get('table_family', '')}_{source_row.get('title', '')}"
    )
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
        "title": source_row.get("title", ""),
        "facts_rows": str(len(facts_rows)),
        "folder": str(table_dir.relative_to(PREVIEW_DIR)),
        "source_image": source_row.get("source_image", ""),
        "source_json": source_row.get("source_json", ""),
        "compiled_member_count": "",
    }


def write_assembled_table_folder(
    out_dir: Path,
    dataset_label: str,
    source_row: dict,
    assembled_group: dict,
    member_rows: List[dict],
) -> dict:
    folder_name = slugify(
        f"{dataset_label}_{source_row.get('province', '')}_{source_row.get('year', '')}_{source_row.get('table_family', '')}_{source_row.get('title', '')}"
    )
    table_dir = out_dir / folder_name
    table_dir.mkdir(parents=True, exist_ok=True)

    write_json(table_dir / "source_catalog_row.json", source_row)
    write_json(table_dir / "assembled_group.json", assembled_group)
    if member_rows:
        write_csv(table_dir / "member_pages.csv", list(member_rows[0].keys()), member_rows)

    section_index_rows: List[dict] = []
    for idx, section in enumerate(assembled_group.get("sections", []), start=1):
        sec_name = section.get("name", f"section_{idx}")
        sec_slug = slugify(sec_name, f"section_{idx}")
        rows = section.get("rows", [])
        if rows:
            fieldnames: List[str] = []
            seen: set[str] = set()
            for row in rows:
                for key in row.keys():
                    if key not in seen:
                        seen.add(key)
                        fieldnames.append(key)
            write_csv(table_dir / f"section_{idx:02d}_{sec_slug}.csv", fieldnames, rows)
        section_index_rows.append(
            {
                "section_index": idx,
                "section_name": sec_name,
                "merge_strategy": section.get("merge_strategy", ""),
                "input_pages": section.get("stats", {}).get("input_pages", ""),
                "input_rows": section.get("stats", {}).get("input_rows", ""),
                "output_rows": section.get("stats", {}).get("output_rows", ""),
                "csv_file": f"section_{idx:02d}_{sec_slug}.csv",
            }
        )
    write_csv(
        table_dir / "sections_index.csv",
        ["section_index", "section_name", "merge_strategy", "input_pages", "input_rows", "output_rows", "csv_file"],
        section_index_rows,
    )

    summary = [
        f"dataset: {dataset_label}",
        f"title: {source_row.get('title', '')}",
        f"table_id: {source_row.get('table_id', '')}",
        f"province: {source_row.get('province', '')}",
        f"year: {source_row.get('year', '')}",
        f"table_family: {source_row.get('table_family', '')}",
        f"assembled_group_id: {source_row.get('assembled_group_id', '')}",
        f"compiled_member_count: {source_row.get('compiled_member_count', '')}",
        f"assembled_failure_improvement: {source_row.get('assembled_failure_improvement', '')}",
        f"member_pages: {len(member_rows)}",
        f"sections: {len(section_index_rows)}",
        f"source_image: {source_row.get('source_image', '')}",
    ]
    (table_dir / "summary.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")

    return {
        "dataset": dataset_label,
        "table_id": source_row.get("table_id", ""),
        "province": source_row.get("province", ""),
        "year": source_row.get("year", ""),
        "table_family": source_row.get("table_family", ""),
        "title": source_row.get("title", ""),
        "facts_rows": "",
        "folder": str(table_dir.relative_to(PREVIEW_DIR)),
        "source_image": source_row.get("source_image", ""),
        "source_json": source_row.get("source_json", ""),
        "compiled_member_count": source_row.get("compiled_member_count", ""),
    }


def build_index_markdown(index_rows: List[dict]) -> str:
    lines = [
        "# Preview Pack",
        "",
        "Small, sendable per-table extracts from the final release.",
        "",
        "For page-level tables, each folder includes `facts.csv`, `matrix.csv`, and `source_catalog_row.json`.",
        "For assembled tables, each folder includes `assembled_group.json`, `sections_index.csv`, per-section CSVs, and `member_pages.csv`.",
        "",
        "| Dataset | Province | Year | Family | Title | Folder |",
        "|---|---|---:|---|---|---|",
    ]
    for row in index_rows:
        title = row["title"].replace("|", "/")
        lines.append(
            f"| {row['dataset']} | {row['province']} | {row['year']} | {row['table_family']} | {title} | `{row['folder']}` |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    set_csv_limits()
    if PREVIEW_DIR.exists():
        shutil.rmtree(PREVIEW_DIR)
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

    index_rows: List[dict] = []

    # Core and tail representatives.
    for dataset_name, label in [("core", "core_example"), ("tail", "tail_example")]:
        source_catalog = ROOT / ("core_curated_v2_final" if dataset_name == "core" else "tail_exploratory_v2") / "source_catalog.csv"
        facts_csv = ROOT / ("core_curated_v2_final" if dataset_name == "core" else "tail_exploratory_v2") / "cell_facts_rich.csv"
        chosen_rows = choose_page_level_examples(source_catalog, dataset_name)
        selected_ids = {row["table_id"] for row in chosen_rows}
        print(f"{label}: selected {len(chosen_rows)} representative tables")
        facts_fieldnames, facts_by_table = stream_selected_facts(facts_csv, selected_ids, label)
        out_dir = PREVIEW_DIR / label
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

    # Reviewed assembled tables.
    assembled_source_catalog = ROOT / "assembled_strong_v2" / "source_catalog.csv"
    assembled_member_pages = ROOT / "assembled_strong_v2" / "member_pages.csv"
    assembled_jsonl = CLUSTER_SYNC / "assembled_final_core_plus_tail_v1_improved" / "strong_assembled_tables.jsonl"
    chosen_rows = choose_assembled_rows(assembled_source_catalog)
    selected_ids = {row["table_id"] for row in chosen_rows}
    print(f"assembled_reviewed: selected {len(chosen_rows)} reviewed assembled tables")
    member_rows_by_table = load_member_pages(assembled_member_pages, selected_ids)
    selected_group_ids = {row.get("assembled_group_id", "") for row in chosen_rows}
    assembled_groups = load_assembled_groups(assembled_jsonl, selected_group_ids)
    out_dir = PREVIEW_DIR / "assembled_reviewed"
    out_dir.mkdir(parents=True, exist_ok=True)
    for row in chosen_rows:
        tid = row["table_id"]
        summary_row = write_assembled_table_folder(
            out_dir,
            "assembled_reviewed",
            row,
            assembled_groups[row.get("assembled_group_id", "")],
            member_rows_by_table.get(tid, []),
        )
        index_rows.append(summary_row)

    index_rows.sort(key=lambda r: (r["dataset"], r["table_family"], r["province"], r["title"]))
    write_csv(
        PREVIEW_DIR / "index.csv",
        [
            "dataset",
            "table_id",
            "province",
            "year",
            "table_family",
            "title",
            "facts_rows",
            "folder",
            "source_image",
            "source_json",
            "compiled_member_count",
        ],
        index_rows,
    )
    (PREVIEW_DIR / "README.md").write_text(build_index_markdown(index_rows), encoding="utf-8")
    print(f"Wrote preview pack: {PREVIEW_DIR}")
    print(f"Tables included: {len(index_rows)}")


if __name__ == "__main__":
    main()
