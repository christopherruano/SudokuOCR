#!/usr/bin/env python3
"""Build quality-tier views for export_final."""

from __future__ import annotations

import csv
import json
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path("export/final_project_exports")
OUT = Path("export_final")
PREVIEW_INDEX = ROOT / "preview_pack" / "index.csv"

LAYERS = {
    "core": ROOT / "core_curated_v2_final" / "source_catalog.csv",
    "tail": ROOT / "tail_exploratory_v2" / "source_catalog.csv",
    "assembled": ROOT / "assembled_strong_v2" / "source_catalog.csv",
}

TIERS = [
    "strict_validated",
    "contextual_unvalidated",
    "needs_review",
]


def keep_row(row: dict[str, str]) -> bool:
    return str(row.get("keep_for_inference", "")).lower() in {"1", "true", "yes"}


def tier_for(row: dict[str, str]) -> str:
    status = row.get("effective_quality_status", "")
    if status in {"accepted_strict", "accepted_proportional_tolerance"}:
        return "strict_validated"
    if status == "accepted_contextual_unvalidated":
        return "contextual_unvalidated"
    return "needs_review"


def load_preview_map() -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    with PREVIEW_INDEX.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            out[row["table_id"]] = row
    return out


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


def write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    csv.field_size_limit(sys.maxsize)
    preview_map = load_preview_map()

    root = OUT / "browse" / "by_quality"
    ensure_clean_dir(root)

    fields = [
        "layer",
        "quality_tier",
        "table_id",
        "province",
        "year",
        "table_family",
        "effective_quality_status",
        "quality_status",
        "measure_unit",
        "constraint_regime",
        "n_checks",
        "failed_checks",
        "qc_flags",
        "title",
        "preview_available",
        "preview_folder",
    ]

    by_tier: dict[str, list[dict[str, str]]] = {tier: [] for tier in TIERS}
    summary = {"tiers": {}, "layers": {}}

    for layer, path in LAYERS.items():
        layer_counts = Counter()
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if not keep_row(row):
                    continue
                tier = tier_for(row)
                layer_counts[tier] += 1
                preview = preview_map.get(row["table_id"])
                by_tier[tier].append(
                    {
                        "layer": layer,
                        "quality_tier": tier,
                        "table_id": row["table_id"],
                        "province": row.get("province", ""),
                        "year": row.get("year", ""),
                        "table_family": row.get("table_family", ""),
                        "effective_quality_status": row.get("effective_quality_status", ""),
                        "quality_status": row.get("quality_status", ""),
                        "measure_unit": row.get("measure_unit", ""),
                        "constraint_regime": row.get("constraint_regime", ""),
                        "n_checks": row.get("n_checks", ""),
                        "failed_checks": row.get("failed_checks", ""),
                        "qc_flags": row.get("qc_flags", ""),
                        "title": row.get("title", ""),
                        "preview_available": "1" if preview else "0",
                        "preview_folder": preview.get("folder", "") if preview else "",
                    }
                )
        summary["layers"][layer] = dict(layer_counts)

    for tier in TIERS:
        rows = sorted(
            by_tier[tier],
            key=lambda row: (
                row["layer"],
                row["province"],
                row["year"],
                row["table_family"],
                row["title"],
            ),
        )
        tier_dir = root / tier
        tier_dir.mkdir(parents=True, exist_ok=True)
        write_csv(tier_dir / "tables.csv", rows, fields)

        layer_groups = defaultdict(list)
        for row in rows:
            layer_groups[row["layer"]].append(row)
        for layer, subset in layer_groups.items():
            write_csv(tier_dir / f"{layer}_tables.csv", subset, fields)
            (tier_dir / f"{layer}_table_ids.txt").write_text(
                "\n".join(r["table_id"] for r in subset) + ("\n" if subset else ""),
                encoding="utf-8",
            )

        preview_dir = tier_dir / "preview_links"
        preview_dir.mkdir(exist_ok=True)
        preview_index = []
        for row in rows:
            if row["preview_available"] != "1":
                continue
            target = ROOT / "preview_pack" / row["preview_folder"]
            if not target.exists():
                continue
            link = preview_dir / Path(row["preview_folder"]).name
            make_symlink(target, link)
            preview_index.append(
                {
                    "table_id": row["table_id"],
                    "layer": row["layer"],
                    "table_family": row["table_family"],
                    "title": row["title"],
                    "preview_link": str(link.relative_to(OUT)),
                }
            )
        if preview_index:
            write_csv(
                tier_dir / "preview_index.csv",
                preview_index,
                ["table_id", "layer", "table_family", "title", "preview_link"],
            )

        (tier_dir / "README.md").write_text(
            "\n".join(
                [
                    f"# {tier}",
                    "",
                    {
                        "strict_validated": "Tables that passed shipped validation checks cleanly.",
                        "contextual_unvalidated": "Tables kept for context but not automatically validated.",
                        "needs_review": "Tables kept in the release that still have review-oriented statuses or failed checks.",
                    }[tier],
                    "",
                    "- `tables.csv`: all kept tables in this tier",
                    "- `core_tables.csv`, `tail_tables.csv`, `assembled_tables.csv`: per-layer splits",
                    "- `preview_links/`: symlinks to matching preview folders when available",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        summary["tiers"][tier] = {
            "kept_tables": len(rows),
            "by_layer": {layer: len(subset) for layer, subset in layer_groups.items()},
        }

    write_csv(
        OUT / "indexes" / "quality_tier_summary.csv",
        [
            {
                "quality_tier": tier,
                "kept_tables": str(summary["tiers"][tier]["kept_tables"]),
                "core_tables": str(summary["tiers"][tier]["by_layer"].get("core", 0)),
                "tail_tables": str(summary["tiers"][tier]["by_layer"].get("tail", 0)),
                "assembled_tables": str(summary["tiers"][tier]["by_layer"].get("assembled", 0)),
            }
            for tier in TIERS
        ],
        ["quality_tier", "kept_tables", "core_tables", "tail_tables", "assembled_tables"],
    )

    (root / "README.md").write_text(
        "\n".join(
            [
                "# by_quality",
                "",
                "This view separates the shipped kept tables by trust tier.",
                "",
                "- `strict_validated/`: best subset for headline analysis and examples",
                "- `contextual_unvalidated/`: usable but not automatically validated",
                "- `needs_review/`: exploratory tables that should not be presented as equally trusted",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    (root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote quality views to {root}")


if __name__ == "__main__":
    main()
