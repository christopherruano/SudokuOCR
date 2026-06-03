#!/usr/bin/env python3
"""Build a human-browsable final export shell around final_project_exports."""

from __future__ import annotations

import csv
import json
import shutil
import sys
from pathlib import Path


SRC = Path("export/final_project_exports")
OUT = Path("export_final")

DOC_FILES = [
    "README.md",
    "PROJECT_CODEBOOK.md",
    "COLUMN_CODEBOOK.md",
    "DATA_RELEASE_WRITEUP.md",
    "RELEASE_SUMMARY.md",
    "UNIFIED_PAPER_DATASET_POLICY.md",
    "EMAIL_BLURB.txt",
]

DATASETS = {
    "core_curated_v2_final": SRC / "core_curated_v2_final",
    "tail_exploratory_v2": SRC / "tail_exploratory_v2",
    "assembled_strong_v2": SRC / "assembled_strong_v2",
}

BROWSE_LINKS = {
    "preview_pack": SRC / "preview_pack",
    "typed_views": SRC / "typed_views",
}


def ensure_clean_dir(path: Path) -> None:
    if path.exists() or path.is_symlink():
        if path.is_symlink() or path.is_file():
            path.unlink()
        else:
            shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def make_symlink(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    dst.symlink_to(src.resolve())


def kept_rows(source_catalog: Path) -> int:
    csv.field_size_limit(sys.maxsize)
    total = 0
    with source_catalog.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("keep_for_inference", "")).lower() in {"1", "true", "yes"}:
                total += 1
    return total


def fact_rows(cell_facts: Path) -> int:
    csv.field_size_limit(sys.maxsize)
    with cell_facts.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        next(reader)
        return sum(1 for _ in reader)


def load_release_counts() -> dict[str, dict[str, int]]:
    path = SRC / "release_summary.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, dict[str, int]] = {}
    for key in ["core_curated_v2_final", "tail_exploratory_v2", "assembled_strong_v2"]:
        if key in data:
            out[key] = {
                "kept_sources": int(data[key].get("kept_sources", 0)),
                "fact_rows": int(data[key].get("fact_rows", 0)),
            }
    return out


def main() -> None:
    if not SRC.exists():
        raise SystemExit(f"Missing source package: {SRC}")

    ensure_clean_dir(OUT)
    docs = OUT / "docs"
    data = OUT / "datasets"
    browse = OUT / "browse"
    indexes = OUT / "indexes"
    qa = OUT / "qa"
    overlays = OUT / "overlays"
    docs.mkdir()
    data.mkdir()
    browse.mkdir()
    indexes.mkdir()
    qa.mkdir()
    overlays.mkdir()

    for name in DOC_FILES:
        make_symlink(SRC / name, docs / name)

    for name, path in DATASETS.items():
        make_symlink(path, data / name)

    for name, path in BROWSE_LINKS.items():
        make_symlink(path, browse / name)

    excel_dir = browse / "excel_previews"
    excel_dir.mkdir()
    make_symlink(SRC / "preview_pack" / "core_examples.xlsx", excel_dir / "core_examples.xlsx")
    make_symlink(SRC / "preview_pack" / "tail_examples.xlsx", excel_dir / "tail_examples.xlsx")
    make_symlink(SRC / "preview_pack" / "assembled_reviewed.xlsx", excel_dir / "assembled_reviewed.xlsx")

    by_type = browse / "by_type"
    by_type.mkdir()
    for category in [
        "counts",
        "distribution_of_1000",
        "per_1000_rates",
        "other_rates",
        "per_10000",
        "per_100000",
    ]:
        make_symlink(SRC / "typed_views" / category, by_type / category)

    # overlays / QA symlinks
    for name in [
        "core_curation_actions.csv",
        "assembled_group_decisions_all.csv",
        "assembled_keep_table_ids.txt",
        "assembled_drop_table_ids.txt",
        "distribution_1000_audit.csv",
        "distribution_1000_summary.json",
        "distribution_1000_validated_strict.csv",
        "distribution_1000_needs_caution.csv",
        "cell_facts_rich_column_map.csv",
        "cell_facts_rich_schema_audit.json",
    ]:
        src = SRC / "overlays" / name
        if src.exists():
            make_symlink(src, overlays / name)

    dataset_rows = []
    release_counts = load_release_counts()
    for name, path in DATASETS.items():
        counts = release_counts.get(name, {})
        dataset_rows.append(
            {
                "dataset": name,
                "source_catalog": str((path / "source_catalog.csv").resolve()),
                "cell_facts_rich": str((path / "cell_facts_rich.csv").resolve()),
                "kept_tables": str(counts.get("kept_sources") or kept_rows(path / "source_catalog.csv")),
                "fact_rows": str(counts.get("fact_rows") or fact_rows(path / "cell_facts_rich.csv")),
            }
        )

    with (indexes / "dataset_entrypoints.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["dataset", "source_catalog", "cell_facts_rich", "kept_tables", "fact_rows"],
        )
        writer.writeheader()
        writer.writerows(dataset_rows)

    typed_summary = json.loads((SRC / "typed_views" / "summary.json").read_text(encoding="utf-8"))
    type_rows = []
    for category, info in typed_summary.get("categories", {}).items():
        type_rows.append(
            {
                "category": category,
                "kept_tables": str(info.get("kept_tables", 0)),
                "core_tables": str(info.get("by_layer", {}).get("core", 0)),
                "tail_tables": str(info.get("by_layer", {}).get("tail", 0)),
                "assembled_tables": str(info.get("by_layer", {}).get("assembled", 0)),
                "preview_tables": str(info.get("preview_tables", 0)),
                "tables_csv": str((SRC / "typed_views" / category / "tables.csv").resolve())
                if (SRC / "typed_views" / category / "tables.csv").exists()
                else "",
            }
        )
    with (indexes / "typed_category_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "category",
                "kept_tables",
                "core_tables",
                "tail_tables",
                "assembled_tables",
                "preview_tables",
                "tables_csv",
            ],
        )
        writer.writeheader()
        writer.writerows(type_rows)

    preview_rows = []
    with (SRC / "preview_pack" / "index.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            preview_rows.append(
                {
                    "dataset": row["dataset"],
                    "table_id": row["table_id"],
                    "province": row["province"],
                    "year": row["year"],
                    "table_family": row["table_family"],
                    "title": row["title"],
                    "folder": row["folder"],
                    "local_folder": str((SRC / "preview_pack" / row["folder"]).resolve()),
                    "source_image": row["source_image"],
                    "source_json_reference": row["source_json"],
                    "compiled_member_count": row.get("compiled_member_count", ""),
                }
            )
    with (indexes / "preview_inventory.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "dataset",
                "table_id",
                "province",
                "year",
                "table_family",
                "title",
                "folder",
                "local_folder",
                "source_image",
                "source_json_reference",
                "compiled_member_count",
            ],
        )
        writer.writeheader()
        writer.writerows(preview_rows)

    readme = OUT / "README.md"
    readme.write_text(
        "\n".join(
            [
                "# export_final",
                "",
                "This is the human-browsable final package for exploration and analysis.",
                "",
                "## Start Here",
                "- `docs/README.md`: package overview",
                "- `docs/PROJECT_CODEBOOK.md`: practical usage guide",
                "- `indexes/dataset_entrypoints.csv`: direct paths to the three canonical datasets",
                "- `browse/by_type/`: browse tables by measurement semantics",
                "- `browse/preview_pack/`: per-table previews",
                "- `browse/excel_previews/`: Excel workbooks for fast inspection",
                "- `qa/`: package QA reports",
                "",
                "## Canonical datasets",
                "- `datasets/core_curated_v2_final/`: primary paper dataset",
                "- `datasets/tail_exploratory_v2/`: broader secondary layer",
                "- `datasets/assembled_strong_v2/`: reviewed multipage assembled layer",
                "",
                "## Measurement-semantic browsing",
                "- `browse/by_type/counts/`: additive count tables",
                "- `browse/by_type/distribution_of_1000/`: compositional `out of 1,000` tables",
                "- `browse/by_type/per_1000_rates/`: generic per-1,000 rate tables",
                "- `browse/by_type/other_rates/`: percent/rate tables outside the 1,000-distribution bucket",
                "",
                "## Important note on raw OCR page JSON",
                "- The release ships table-level metadata JSON and preview JSON, not the full raw OCR page JSON corpus.",
                "- `source_json` fields are references to the original OCR outputs, typically on the cluster path.",
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    manifest = {
        "source_package": str(SRC.resolve()),
        "datasets": dataset_rows,
        "typed_categories": type_rows,
        "preview_tables": len(preview_rows),
        "notes": {
            "uses_symlinks": True,
            "raw_ocr_page_json_not_fully_bundled": True,
        },
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Built {OUT}")


if __name__ == "__main__":
    main()
