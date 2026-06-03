#!/usr/bin/env python3
"""Build type-based views over the final project exports without duplicating data."""

from __future__ import annotations

import csv
import json
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path("export/final_project_exports")
OUTPUT = ROOT / "typed_views"
PREVIEW_INDEX = ROOT / "preview_pack" / "index.csv"

LAYERS = {
    "core": ROOT / "core_curated_v2_final" / "source_catalog.csv",
    "tail": ROOT / "tail_exploratory_v2" / "source_catalog.csv",
    "assembled": ROOT / "assembled_strong_v2" / "source_catalog.csv",
}

CATEGORY_ORDER = [
    "counts",
    "distribution_of_1000",
    "per_1000_rates",
    "other_rates",
    "per_10000",
    "per_100000",
    "other",
]

CATEGORY_DESCRIPTIONS = {
    "counts": "Absolute-count age tables. These are the standard additive tables.",
    "distribution_of_1000": "Compositional proportional tables where a slice sums to 1,000.",
    "per_1000_rates": "Per-1,000 rate/ratio tables that are not compositional distributions.",
    "other_rates": "Rate / ratio tables that are not treated as compositional 1,000-distribution tables.",
    "per_10000": "Rare tables measured per 10,000.",
    "per_100000": "Rare tables measured per 100,000.",
    "other": "Anything that did not fit the standard measurement buckets.",
}

THIN_FIELDS = [
    "layer",
    "category",
    "table_id",
    "province",
    "year",
    "table_family",
    "measure_unit",
    "denominator_hint",
    "constraint_regime",
    "quality_status",
    "effective_quality_status",
    "n_checks",
    "failed_checks",
    "qc_flags",
    "title",
    "source_json",
    "source_image",
    "preview_available",
    "preview_folder",
]


def keep_row(row: dict[str, str]) -> bool:
    return str(row.get("keep_for_inference", "")).lower() in {"1", "true", "yes"}


def classify_row(row: dict[str, str]) -> str:
    unit = row.get("measure_unit", "")
    denom = row.get("denominator_hint", "")
    regime = row.get("constraint_regime", "")
    title = row.get("title", "")
    title_l = title.lower()
    is_distribution_1000 = (
        "distribution by civil condition of 1,000" in title_l
        or "distribution by civil condition of 1000" in title_l
        or "age distribution of 1,000" in title_l
        or "age distribution of 1000" in title_l
        or "distribution of 1,000" in title_l
        or "distribution of 1000" in title_l
        or "out of 1,000" in title_l
        or "out of 1000" in title_l
    )
    if is_distribution_1000 and denom == "1000" and regime == "proportional_distribution":
        return "distribution_of_1000"
    if denom == "1000" and unit == "rate":
        return "per_1000_rates"
    if unit == "count":
        return "counts"
    if unit == "rate":
        return "other_rates"
    if unit == "per_10000":
        return "per_10000"
    if unit == "per_100000":
        return "per_100000"
    return "other"


def load_preview_index() -> dict[str, dict[str, str]]:
    preview_map: dict[str, dict[str, str]] = {}
    if not PREVIEW_INDEX.exists():
        return preview_map
    with PREVIEW_INDEX.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            preview_map[row["table_id"]] = row
    return preview_map


def ensure_clean_dir(path: Path) -> None:
    if path.exists() or path.is_symlink():
        if path.is_symlink() or path.is_file():
            path.unlink()
        else:
            shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_symlink(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    dst.symlink_to(src.resolve())


def main() -> None:
    csv.field_size_limit(sys.maxsize)
    preview_map = load_preview_index()
    ensure_clean_dir(OUTPUT)

    by_category: dict[str, list[dict[str, str]]] = {cat: [] for cat in CATEGORY_ORDER}
    summary: dict[str, dict[str, object]] = {
        "categories": {},
        "layers": {},
    }

    for layer, csv_path in LAYERS.items():
        layer_rows = 0
        layer_counts = Counter()
        with csv_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if not keep_row(row):
                    continue
                layer_rows += 1
                category = classify_row(row)
                layer_counts[category] += 1
                preview = preview_map.get(row["table_id"])
                thin = {
                    "layer": layer,
                    "category": category,
                    "table_id": row["table_id"],
                    "province": row.get("province", ""),
                    "year": row.get("year", ""),
                    "table_family": row.get("table_family", ""),
                    "measure_unit": row.get("measure_unit", ""),
                    "denominator_hint": row.get("denominator_hint", ""),
                    "constraint_regime": row.get("constraint_regime", ""),
                    "quality_status": row.get("quality_status", ""),
                    "effective_quality_status": row.get("effective_quality_status", ""),
                    "n_checks": row.get("n_checks", ""),
                    "failed_checks": row.get("failed_checks", ""),
                    "qc_flags": row.get("qc_flags", ""),
                    "title": row.get("title", ""),
                    "source_json": row.get("source_json", ""),
                    "source_image": row.get("source_image", ""),
                    "preview_available": "1" if preview else "0",
                    "preview_folder": preview.get("folder", "") if preview else "",
                }
                by_category[category].append(thin)
        summary["layers"][layer] = {
            "kept_tables": layer_rows,
            "by_category": dict(layer_counts),
        }

    overall_rows = []
    for category in CATEGORY_ORDER:
        rows = sorted(
            by_category[category],
            key=lambda row: (
                row["layer"],
                row["province"],
                row["year"],
                row["table_family"],
                row["title"],
            ),
        )
        overall_rows.extend(rows)
        cat_dir = OUTPUT / category
        cat_dir.mkdir(parents=True, exist_ok=True)

        write_csv(cat_dir / "tables.csv", rows, THIN_FIELDS)

        layer_groups = defaultdict(list)
        fam_counts: Counter[tuple[str, str]] = Counter()
        preview_rows = []
        for row in rows:
            layer_groups[row["layer"]].append(row)
            fam_counts[(row["layer"], row["table_family"])] += 1
            if row["preview_available"] == "1":
                preview_rows.append(row)

        for layer, subset in layer_groups.items():
            write_csv(cat_dir / f"{layer}_tables.csv", subset, THIN_FIELDS)
            table_ids = [r["table_id"] for r in subset]
            (cat_dir / f"{layer}_table_ids.txt").write_text(
                "\n".join(table_ids) + ("\n" if table_ids else ""),
                encoding="utf-8",
            )

        fam_rows = [
            {
                "layer": layer,
                "table_family": family,
                "n_tables": str(count),
            }
            for (layer, family), count in sorted(
                fam_counts.items(), key=lambda kv: (kv[0][0], -kv[1], kv[0][1])
            )
        ]
        write_csv(cat_dir / "family_counts.csv", fam_rows, ["layer", "table_family", "n_tables"])

        by_family_dir = cat_dir / "by_family"
        by_family_dir.mkdir(exist_ok=True)
        fam_groups = defaultdict(list)
        for row in rows:
            fam_groups[(row["layer"], row["table_family"])].append(row)
        for (layer, family), subset in sorted(fam_groups.items()):
            stem = f"{layer}__{family}"
            write_csv(by_family_dir / f"{stem}.csv", subset, THIN_FIELDS)
            (by_family_dir / f"{stem}_table_ids.txt").write_text(
                "\n".join(r["table_id"] for r in subset) + ("\n" if subset else ""),
                encoding="utf-8",
            )

        preview_dir = cat_dir / "preview_links"
        preview_dir.mkdir(exist_ok=True)
        preview_index_rows = []
        for row in preview_rows:
            target = ROOT / "preview_pack" / row["preview_folder"]
            if target.exists():
                link_name = Path(row["preview_folder"]).name
                link_path = preview_dir / link_name
                make_symlink(target, link_path)
                preview_index_rows.append(
                    {
                        "table_id": row["table_id"],
                        "layer": row["layer"],
                        "table_family": row["table_family"],
                        "title": row["title"],
                        "preview_folder": row["preview_folder"],
                        "preview_link": str(link_path.relative_to(ROOT)),
                    }
                )
        if preview_index_rows:
            write_csv(
                cat_dir / "preview_index.csv",
                preview_index_rows,
                ["table_id", "layer", "table_family", "title", "preview_folder", "preview_link"],
            )

        readme_lines = [
            f"# {category}",
            "",
            CATEGORY_DESCRIPTIONS[category],
            "",
            f"Kept tables: {len(rows)}",
            "",
            "Files:",
            "- `tables.csv`: all kept tables in this category",
            "- `core_tables.csv`, `tail_tables.csv`, `assembled_tables.csv`: layer-specific filtered indexes",
            "- `*_table_ids.txt`: exact table ids to filter in analysis",
            "- `family_counts.csv`: counts by layer and table family",
            "- `by_family/`: family-level filtered indexes and table-id lists",
            "- `preview_links/`: symlinks to the preview folders for the subset that has previews",
            "",
            "Interpretation:",
            "- `counts` are the standard additive age tables.",
            "- `distribution_of_1000` are compositional tables where slices sum to 1,000.",
            "- `other_rates` are rate/ratio tables and should not be treated like additive counts.",
        ]
        (cat_dir / "README.md").write_text("\n".join(readme_lines) + "\n", encoding="utf-8")

        summary["categories"][category] = {
            "kept_tables": len(rows),
            "by_layer": {layer: len(subset) for layer, subset in layer_groups.items()},
            "preview_tables": len(preview_rows),
        }

    write_csv(OUTPUT / "all_tables.csv", overall_rows, THIN_FIELDS)
    (OUTPUT / "SUMMARY.md").write_text(
        "\n".join(
            [
                "# Typed Views",
                "",
                "This directory reorganizes the final release by measurement semantics without copying the large data files.",
                "",
                "Main categories:",
                "- `counts`: plain additive age tables",
                "- `distribution_of_1000`: compositional proportional tables",
                "- `per_1000_rates`: per-1,000 rates that are not compositional distributions",
                "- `other_rates`: other rates/ratios",
                "- `per_10000` and `per_100000`: rare higher-denominator rate tables",
                "",
                "Use the per-category `tables.csv` and `*_table_ids.txt` files to filter the main release tables.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote typed views to {OUTPUT}")


if __name__ == "__main__":
    main()
