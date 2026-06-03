#!/usr/bin/env python3
"""Build a MECE column map for the final cell_facts_rich schemas."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List


ROOT = Path("/Users/chrisruano/Research/SudokuOCR/export/final_project_exports")
OUT_DIR = ROOT / "overlays"


GROUPS: Dict[str, Dict[str, object]] = {
    "identity_provenance": {
        "description": "Stable identifiers and raw source paths for the originating OCR table/image/PDF.",
        "columns": [
            "table_id",
            "source_json",
            "source_image",
            "source_image_name",
            "source_table_dir",
            "source_table_dir_name",
            "source_file",
            "manifest_classification",
        ],
    },
    "geo_time_context": {
        "description": "Geographic and temporal context attached to the table or fact row.",
        "columns": ["province", "year", "region", "geo_name", "geo_level"],
    },
    "title_queue_context": {
        "description": "Human-readable titles and upstream queue/context labels.",
        "columns": [
            "title",
            "queue_title",
            "queue_display_identifier",
            "queue_tags",
            "canonical_topic",
            "relevance_tier",
            "queue_dir_name_match",
        ],
    },
    "table_semantics": {
        "description": "High-level table-family semantics and denominator/measurement interpretation.",
        "columns": [
            "table_family",
            "constraint_regime",
            "denominator_hint",
            "data_type",
            "measure_unit",
            "row_axis_type",
            "row_label_name",
        ],
    },
    "extraction_process": {
        "description": "Operational extraction metadata from OCR/table processing.",
        "columns": [
            "extraction_method",
            "api_calls",
            "elapsed_seconds",
            "repair_api_calls",
            "repair_entries_count",
            "targeted_recheck_used",
            "extraction_mode",
        ],
    },
    "acceptance_quality": {
        "description": "Acceptance, quality, and inclusion decisions used in the release.",
        "columns": [
            "acceptance_decision",
            "acceptance_next_step",
            "acceptance_version",
            "acceptance_reasons",
            "trust_score",
            "quality_status",
            "effective_quality_status",
            "qc_flags",
            "keep_for_inference",
        ],
    },
    "constraint_summaries": {
        "description": "Compact summaries of validation/constraint behavior.",
        "columns": [
            "n_checks",
            "failed_checks",
            "n_constraint_failures",
            "constraint_failure_levels",
            "max_constraint_diff",
            "age_warning_count",
            "cross_group_active",
            "cross_section_active",
        ],
    },
    "table_structure_summaries": {
        "description": "Counts and structural summaries of extracted rows, columns, and duplicates.",
        "columns": [
            "n_sections",
            "distinct_sections",
            "n_rows",
            "distinct_row_labels",
            "numeric_cells",
            "zero_cells",
            "negative_cells",
            "noninteger_cells",
            "nonblank_rows",
            "blank_rows",
            "age_like_rows",
            "n_column_groups",
            "distinct_column_groups",
            "distinct_sub_columns",
            "duplicate_exact_cells",
            "duplicate_conflict_cells",
            "column_groups",
            "sample_row_labels",
        ],
    },
    "nested_payloads": {
        "description": "Serialized raw metadata and validation payloads preserved for reinterpretation.",
        "columns": [
            "metadata_json",
            "acceptance_signals_json",
            "constraints_json",
            "queue_row_json",
        ],
    },
    "assembled_validation": {
        "description": "Assembled-table grouping, merge, and validation metadata.",
        "columns": [
            "assembled_group_id",
            "assembled_n_pages",
            "assembled_overall_strategy",
            "assembled_signals",
            "assembled_section_strategy_counts",
            "assembled_input_constraint_checks",
            "assembled_input_constraint_failures",
            "assembled_constraint_checks",
            "assembled_constraint_failures",
            "assembled_all_passed",
            "assembled_failure_improvement",
        ],
    },
    "assembled_member_provenance": {
        "description": "Backreferences from assembled tables to their member page-level tables.",
        "columns": [
            "assembled_member_source_jsons",
            "assembled_member_source_images",
        ],
    },
    "compiled_member_metadata": {
        "description": "Flattened summary metadata compiled from assembled member pages.",
        "columns": [
            "compiled_member_count",
            "compiled_member_titles",
            "compiled_member_queue_titles",
            "compiled_member_table_families",
            "compiled_member_source_files",
            "compiled_member_manifest_classifications",
            "compiled_member_column_groups",
            "compiled_member_provinces",
            "compiled_member_years",
            "compiled_member_quality_statuses",
            "compiled_member_effective_quality_statuses",
            "compiled_section_names",
            "compiled_title_variants_count",
            "compiled_source_file_count",
            "compiled_column_group_count",
        ],
    },
    "fact_section_row_axis": {
        "description": "Section and row-axis labels attached to each fact row.",
        "columns": [
            "section",
            "section_type",
            "row_label_raw",
            "row_label_std",
            "row_dimension",
            "row_group_value",
        ],
    },
    "row_axis_age": {
        "description": "Age parsing derived from the row axis.",
        "columns": [
            "age_label_std",
            "age_lower",
            "age_upper",
            "age_open_ended",
            "age_total",
            "age_not_stated",
        ],
    },
    "column_axis_normalization": {
        "description": "Raw and standardized column-group labels plus column-axis type.",
        "columns": [
            "column_group_raw",
            "column_group_std",
            "column_dimension",
            "column_group_value",
        ],
    },
    "column_axis_age": {
        "description": "Age parsing derived from the column axis.",
        "columns": [
            "column_age_label_std",
            "column_age_lower",
            "column_age_upper",
            "column_age_open_ended",
            "column_age_total",
            "column_age_not_stated",
        ],
    },
    "resolved_age": {
        "description": "Canonical resolved age interpretation used for analysis.",
        "columns": [
            "resolved_age_axis",
            "resolved_age_label_std",
            "resolved_age_lower",
            "resolved_age_upper",
            "resolved_age_open_ended",
            "resolved_age_total",
            "resolved_age_not_stated",
        ],
    },
    "subcolumn_semantics": {
        "description": "Lower-level subcolumn labels and semantic decomposition of the fact.",
        "columns": [
            "sub_column_raw",
            "sub_column_std",
            "sex",
            "civil_condition",
            "religion",
        ],
    },
    "numeric_payload": {
        "description": "The numeric fact value itself.",
        "columns": ["value"],
    },
}


def load_header(path: Path) -> List[str]:
    with path.open(newline="", encoding="utf-8") as fh:
        return next(csv.reader(fh))


def build_column_to_group() -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    dupes: Dict[str, List[str]] = {}
    for group_name, spec in GROUPS.items():
        for col in spec["columns"]:
            if col in mapping:
                dupes.setdefault(col, []).append(group_name)
            else:
                mapping[col] = group_name
    if dupes:
        raise SystemExit(f"Duplicate column assignments in GROUPS: {dupes}")
    return mapping


def main() -> None:
    core_header = load_header(ROOT / "core_curated_v2_final" / "cell_facts_rich.csv")
    tail_header = load_header(ROOT / "tail_exploratory_v2" / "cell_facts_rich.csv")
    assembled_header = load_header(ROOT / "assembled_strong_v2" / "cell_facts_rich.csv")

    if core_header != tail_header:
        raise SystemExit("Core and tail cell_facts_rich headers differ unexpectedly.")

    shared_header = core_header
    shared_set = set(shared_header)
    assembled_set = set(assembled_header)
    assembled_only = assembled_set - shared_set

    column_to_group = build_column_to_group()

    missing_shared = [c for c in shared_header if c not in column_to_group]
    missing_assembled = [c for c in assembled_header if c not in column_to_group]
    extra_mapped = [c for c in column_to_group if c not in assembled_set]

    if missing_shared or missing_assembled or extra_mapped:
        raise SystemExit(
            json.dumps(
                {
                    "missing_shared": missing_shared,
                    "missing_assembled": missing_assembled,
                    "extra_mapped": extra_mapped,
                },
                indent=2,
            )
        )

    rows = []
    for col in assembled_header:
        group = column_to_group[col]
        applies_to = "shared" if col in shared_set else "assembled_only"
        rows.append(
            {
                "column_name": col,
                "group_name": group,
                "group_description": GROUPS[group]["description"],
                "applies_to": applies_to,
            }
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    map_path = OUT_DIR / "cell_facts_rich_column_map.csv"
    with map_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["column_name", "group_name", "group_description", "applies_to"],
        )
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "shared_column_count": len(shared_header),
        "assembled_column_count": len(assembled_header),
        "assembled_only_column_count": len(assembled_only),
        "group_count": len(GROUPS),
        "all_columns_assigned_exactly_once": True,
        "shared_columns_fully_covered": True,
        "assembled_columns_fully_covered": True,
        "groups": {
            group: {
                "description": spec["description"],
                "count": len(spec["columns"]),
                "columns": spec["columns"],
            }
            for group, spec in GROUPS.items()
        },
    }
    (OUT_DIR / "cell_facts_rich_schema_audit.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Wrote {map_path}")
    print(f"Wrote {OUT_DIR / 'cell_facts_rich_schema_audit.json'}")
    print(f"Shared columns: {len(shared_header)}")
    print(f"Assembled columns: {len(assembled_header)}")


if __name__ == "__main__":
    main()
