from __future__ import annotations

import csv
import json
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path("export_final")
OUT = ROOT / "team_ready"

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

DATASET_SPECS = [
    ("core", ROOT / "datasets" / "core_curated_v2_final"),
    ("tail", ROOT / "datasets" / "tail_exploratory_v2"),
    ("assembled", ROOT / "datasets" / "assembled_strong_v2"),
]

QUALITY_DIR = ROOT / "browse" / "by_quality"
TYPE_DIR = ROOT / "browse" / "by_type"
MORTALITY_DIR = ROOT / "browse" / "by_topic" / "mortality"
FINAL_REVIEW_DIR = ROOT / "browse" / "final_submission_review"
OVERLAYS_DIR = ROOT / "overlays"


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def safe_unlink(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def ensure_symlink(dst: Path, target: Path) -> None:
    if dst.exists() or dst.is_symlink():
        safe_unlink(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.symlink_to(target.resolve())


def load_quality_map() -> dict[str, str]:
    quality_map: dict[str, str] = {}
    for tier in ("strict_validated", "contextual_unvalidated", "needs_review"):
        for row in read_csv(QUALITY_DIR / tier / "tables.csv"):
            table_id = row.get("table_id", "")
            if table_id:
                quality_map[table_id] = tier
    return quality_map


def load_type_map() -> dict[str, str]:
    type_map: dict[str, str] = {}
    for category_dir in sorted(TYPE_DIR.iterdir()):
        if not category_dir.is_dir():
            continue
        for row in read_csv(category_dir / "tables.csv"):
            table_id = row.get("table_id", "")
            if table_id:
                type_map[table_id] = category_dir.name
    return type_map


def load_mortality_map() -> dict[str, dict]:
    out: dict[str, dict] = {}
    for row in read_csv(MORTALITY_DIR / "all_tables.csv"):
        table_id = row.get("table_id", "")
        if table_id:
            out[table_id] = row
    return out


def load_final_review_map() -> dict[str, dict]:
    out: dict[str, dict] = {}
    for row in read_csv(FINAL_REVIEW_DIR / "review_decisions.csv"):
        table_id = row.get("table_id", "")
        if table_id:
            out[table_id] = row
    return out


def load_assembled_review_map() -> dict[str, dict]:
    out: dict[str, dict] = {}
    for row in read_csv(OVERLAYS_DIR / "assembled_group_decisions_all.csv"):
        table_id = row.get("table_id", "")
        if table_id:
            out[table_id] = row
    return out


def confidence_for_tier(tier: str) -> str:
    return {
        "strict_validated": "high",
        "contextual_unvalidated": "medium",
        "needs_review": "low",
        "not_in_release": "reference_only",
    }.get(tier, "unknown")


def truthy(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def build_rows() -> tuple[list[dict], list[dict]]:
    quality_map = load_quality_map()
    type_map = load_type_map()
    mortality_map = load_mortality_map()
    final_review_map = load_final_review_map()
    assembled_review_map = load_assembled_review_map()

    full_rows: list[dict] = []
    master_rows: list[dict] = []

    for layer, dataset_dir in DATASET_SPECS:
        source_catalog_path = dataset_dir / "source_catalog.csv"
        for row in read_csv(source_catalog_path):
            table_id = row.get("table_id", "")
            in_release_scope = truthy(row.get("keep_for_inference", ""))
            tier = quality_map.get(table_id, "")
            if not tier:
                tier = "not_in_release" if not in_release_scope else "unknown"
            confidence = confidence_for_tier(tier)
            measure_category = type_map.get(table_id, "") or "unclassified"
            mortality = mortality_map.get(table_id, {})
            final_review = final_review_map.get(table_id, {})
            assembled_review = assembled_review_map.get(table_id, {})
            facts_rows = row.get("numeric_cells", "")
            preview_folder = final_review.get("preview_folder", "")
            preview_available = "1" if preview_folder else "0"

            derived = {
                "layer": layer,
                "dataset_dir": str(dataset_dir),
                "source_catalog_path": str(source_catalog_path),
                "facts_path": str(dataset_dir / "cell_facts_rich.csv"),
                "quality_tier": tier,
                "confidence_level": confidence,
                "measure_category": measure_category,
                "facts_rows": facts_rows,
                "preview_available": preview_available,
                "preview_folder": preview_folder,
                "in_release_scope": "1" if in_release_scope else "0",
                "mortality_topic": "1" if mortality else "0",
                "mortality_kind": mortality.get("mortality_kind", ""),
                "in_final_submission_review_queue": "1" if final_review else "0",
                "assistant_suggested_decision": final_review.get("suggested_decision", ""),
                "manual_review_decision": final_review.get("review_decision", ""),
                "manual_review_notes": final_review.get("review_notes", ""),
                "manual_final_decision": final_review.get("final_decision", ""),
                "in_assembled_review": "1" if assembled_review else "0",
                "assembled_review_decision": assembled_review.get("final_decision", ""),
                "assembled_review_notes": assembled_review.get("review_notes", ""),
                "needs_manual_review": "1"
                if tier == "needs_review" or final_review or assembled_review
                else "0",
            }

            full = dict(row)
            full.update(derived)
            full_rows.append(full)

            master_rows.append(
                {
                    "layer": layer,
                    "table_id": table_id,
                    "province": row.get("province", ""),
                    "year": row.get("year", ""),
                    "title": row.get("title", ""),
                    "table_family": row.get("table_family", ""),
                    "facts_rows": facts_rows,
                    "in_release_scope": derived["in_release_scope"],
                    "quality_tier": tier,
                    "confidence_level": confidence,
                    "effective_quality_status": row.get("effective_quality_status", ""),
                    "quality_status": row.get("quality_status", ""),
                    "n_checks": row.get("n_checks", ""),
                    "failed_checks": row.get("failed_checks", ""),
                    "qc_flags": row.get("qc_flags", ""),
                    "measure_category": measure_category,
                    "measure_unit": row.get("measure_unit", ""),
                    "constraint_regime": row.get("constraint_regime", ""),
                    "denominator_hint": row.get("denominator_hint", ""),
                    "mortality_topic": derived["mortality_topic"],
                    "mortality_kind": derived["mortality_kind"],
                    "needs_manual_review": derived["needs_manual_review"],
                    "assistant_suggested_decision": derived["assistant_suggested_decision"],
                    "manual_review_decision": derived["manual_review_decision"],
                    "manual_final_decision": derived["manual_final_decision"],
                    "assembled_review_decision": derived["assembled_review_decision"],
                    "preview_available": preview_available,
                    "preview_folder": preview_folder,
                    "source_image": row.get("source_image", ""),
                    "source_json": row.get("source_json", ""),
                }
            )

    return full_rows, master_rows


def build_outputs() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)

    (OUT / "catalogs").mkdir(parents=True, exist_ok=True)
    (OUT / "indexes").mkdir(parents=True, exist_ok=True)

    for name in ("docs", "qa", "overlays"):
        ensure_symlink(OUT / name, ROOT / name)

    for name in ("datasets", "browse"):
        (OUT / name).mkdir(parents=True, exist_ok=True)

    for layer, dataset_dir in DATASET_SPECS:
        ensure_symlink(OUT / "datasets" / dataset_dir.name, dataset_dir)

    ensure_symlink(OUT / "browse" / "preview_pack", ROOT / "browse" / "preview_pack")
    ensure_symlink(OUT / "browse" / "by_type", ROOT / "browse" / "by_type")
    ensure_symlink(OUT / "browse" / "by_quality", ROOT / "browse" / "by_quality")
    ensure_symlink(OUT / "browse" / "mortality", MORTALITY_DIR)
    ensure_symlink(OUT / "browse" / "review_queue_all", FINAL_REVIEW_DIR)
    ensure_symlink(OUT / "browse" / "failing_tables", ROOT / "browse" / "failing_tables")

    (OUT / "apps").mkdir(parents=True, exist_ok=True)
    ensure_symlink(OUT / "apps" / "final_submission_review_app.py", Path("final_submission_review_app.py"))
    ensure_symlink(OUT / "apps" / "mortality_review_app.py", Path("mortality_review_app.py"))
    ensure_symlink(OUT / "apps" / "sync_release_review_assets.sh", Path("sync_release_review_assets.sh"))
    ensure_symlink(OUT / "apps" / "sync_mortality_review_assets.sh", Path("sync_mortality_review_assets.sh"))
    launch_final = OUT / "apps" / "launch_final_submission_review.sh"
    launch_final.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "cd /Users/chrisruano/Research/SudokuOCR\n"
        "python3 final_submission_review_app.py\n",
        encoding="utf-8",
    )
    launch_final.chmod(0o755)
    launch_mortality = OUT / "apps" / "launch_mortality_review.sh"
    launch_mortality.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "cd /Users/chrisruano/Research/SudokuOCR\n"
        "python3 mortality_review_app.py\n",
        encoding="utf-8",
    )
    launch_mortality.chmod(0o755)

    full_rows, master_rows = build_rows()
    master_rows.sort(key=lambda r: (r["layer"], r["province"], r["year"], r["title"]))
    full_rows.sort(key=lambda r: (r["layer"], r["province"], r["year"], r["title"]))

    master_fields = [
        "layer",
        "table_id",
        "province",
        "year",
        "title",
        "table_family",
        "facts_rows",
        "in_release_scope",
        "quality_tier",
        "confidence_level",
        "effective_quality_status",
        "quality_status",
        "n_checks",
        "failed_checks",
        "qc_flags",
        "measure_category",
        "measure_unit",
        "constraint_regime",
        "denominator_hint",
        "mortality_topic",
        "mortality_kind",
        "needs_manual_review",
        "assistant_suggested_decision",
        "manual_review_decision",
        "manual_final_decision",
        "assembled_review_decision",
        "preview_available",
        "preview_folder",
        "source_image",
        "source_json",
    ]
    write_csv(OUT / "catalogs" / "all_tables_master.csv", master_rows, master_fields)

    full_fields = list(full_rows[0].keys()) if full_rows else []
    write_csv(OUT / "catalogs" / "all_tables_full_metadata.csv", full_rows, full_fields)

    by_layer: dict[str, list[dict]] = defaultdict(list)
    by_conf: dict[str, list[dict]] = defaultdict(list)
    by_quality: dict[str, list[dict]] = defaultdict(list)
    manual_review_rows: list[dict] = []
    mortality_rows: list[dict] = []
    release_scope_rows: list[dict] = []
    reference_only_rows: list[dict] = []
    for row in master_rows:
        by_layer[row["layer"]].append(row)
        by_conf[row["confidence_level"]].append(row)
        by_quality[row["quality_tier"]].append(row)
        if truthy(row["needs_manual_review"]):
            manual_review_rows.append(row)
        if truthy(row["mortality_topic"]):
            mortality_rows.append(row)
        if truthy(row["in_release_scope"]):
            release_scope_rows.append(row)
        else:
            reference_only_rows.append(row)

    for layer, rows in by_layer.items():
        write_csv(OUT / "catalogs" / "by_layer" / f"{layer}_tables.csv", rows, master_fields)
    for conf, rows in by_conf.items():
        write_csv(OUT / "catalogs" / "by_confidence" / f"{conf}_confidence.csv", rows, master_fields)
    for quality, rows in by_quality.items():
        write_csv(OUT / "catalogs" / "by_quality" / f"{quality}.csv", rows, master_fields)

    write_csv(OUT / "catalogs" / "manual_review_queue.csv", manual_review_rows, master_fields)
    write_csv(OUT / "catalogs" / "mortality_tables.csv", mortality_rows, master_fields)
    write_csv(OUT / "catalogs" / "release_scope_tables.csv", release_scope_rows, master_fields)
    write_csv(OUT / "catalogs" / "reference_only_tables.csv", reference_only_rows, master_fields)
    final_clean_rows = [
        r
        for r in release_scope_rows
        if r["quality_tier"] == "strict_validated" and not truthy(r["needs_manual_review"])
    ]
    assembled_review_required_rows = [
        r for r in release_scope_rows if r["layer"] == "assembled" and truthy(r["needs_manual_review"])
    ]
    non_clean_release_rows = [
        r
        for r in release_scope_rows
        if r["quality_tier"] != "strict_validated" or truthy(r["needs_manual_review"])
    ]
    write_csv(OUT / "catalogs" / "final_clean_tables.csv", final_clean_rows, master_fields)
    write_csv(OUT / "catalogs" / "assembled_review_required.csv", assembled_review_required_rows, master_fields)
    write_csv(OUT / "catalogs" / "non_clean_release_scope.csv", non_clean_release_rows, master_fields)
    write_csv(
        OUT / "catalogs" / "mortality_needs_review.csv",
        [r for r in mortality_rows if r["quality_tier"] != "strict_validated"],
        master_fields,
    )
    write_csv(
        OUT / "catalogs" / "final_clean_mortality.csv",
        [r for r in final_clean_rows if truthy(r["mortality_topic"])],
        master_fields,
    )

    final_clean_dir = OUT / "final_clean"
    final_clean_dir.mkdir(parents=True, exist_ok=True)
    ensure_symlink(final_clean_dir / "docs", ROOT / "docs")
    ensure_symlink(final_clean_dir / "qa", ROOT / "qa")
    (final_clean_dir / "browse").mkdir(parents=True, exist_ok=True)
    ensure_symlink(final_clean_dir / "browse" / "preview_pack", ROOT / "browse" / "preview_pack")
    ensure_symlink(final_clean_dir / "browse" / "mortality", MORTALITY_DIR)
    ensure_symlink(final_clean_dir / "browse" / "by_quality", ROOT / "browse" / "by_quality" / "strict_validated")
    (final_clean_dir / "catalogs").mkdir(parents=True, exist_ok=True)
    ensure_symlink(final_clean_dir / "catalogs" / "final_clean_tables.csv", OUT / "catalogs" / "final_clean_tables.csv")
    ensure_symlink(final_clean_dir / "catalogs" / "final_clean_mortality.csv", OUT / "catalogs" / "final_clean_mortality.csv")
    ensure_symlink(final_clean_dir / "catalogs" / "assembled_review_required.csv", OUT / "catalogs" / "assembled_review_required.csv")
    ensure_symlink(final_clean_dir / "catalogs" / "non_clean_release_scope.csv", OUT / "catalogs" / "non_clean_release_scope.csv")
    final_clean_readme = f"""# Final Clean Demo Set

This folder is the no-embarrassment subset for team presentation.

Inclusion rule:
- in shipped release scope
- `quality_tier = strict_validated`
- not still flagged for manual review

Counts:
- total clean tables: {len(final_clean_rows)}
- core: {sum(1 for r in final_clean_rows if r['layer'] == 'core')}
- tail: {sum(1 for r in final_clean_rows if r['layer'] == 'tail')}
- assembled: {sum(1 for r in final_clean_rows if r['layer'] == 'assembled')}

Use:
- `catalogs/final_clean_tables.csv` for the main demo table list
- `catalogs/final_clean_mortality.csv` for the mortality-only clean subset

Excluded from this clean set:
- all non-strict release tables
- all tables still needing manual review
- all assembled tables for now, because the assembled layer still needs semantic review even when constraint checks pass
"""
    (final_clean_dir / "README.md").write_text(final_clean_readme, encoding="utf-8")

    layer_counts = Counter(r["layer"] for r in master_rows)
    quality_counts = Counter(r["quality_tier"] for r in master_rows)
    confidence_counts = Counter(r["confidence_level"] for r in master_rows)
    measure_counts = Counter(r["measure_category"] for r in master_rows)
    mortality_kind_counts = Counter(r["mortality_kind"] for r in mortality_rows)

    write_csv(
        OUT / "indexes" / "layer_summary.csv",
        [{"layer": k, "tables": v} for k, v in sorted(layer_counts.items())],
        ["layer", "tables"],
    )
    write_csv(
        OUT / "indexes" / "quality_summary.csv",
        [{"quality_tier": k, "tables": v} for k, v in sorted(quality_counts.items())],
        ["quality_tier", "tables"],
    )
    write_csv(
        OUT / "indexes" / "confidence_summary.csv",
        [{"confidence_level": k, "tables": v} for k, v in sorted(confidence_counts.items())],
        ["confidence_level", "tables"],
    )
    write_csv(
        OUT / "indexes" / "measure_category_summary.csv",
        [{"measure_category": k, "tables": v} for k, v in sorted(measure_counts.items())],
        ["measure_category", "tables"],
    )
    write_csv(
        OUT / "indexes" / "mortality_kind_summary.csv",
        [{"mortality_kind": k, "tables": v} for k, v in sorted(mortality_kind_counts.items())],
        ["mortality_kind", "tables"],
    )

    payload = {
        "total_tables": len(master_rows),
        "layers": dict(sorted(layer_counts.items())),
        "quality_tiers": dict(sorted(quality_counts.items())),
        "confidence_levels": dict(sorted(confidence_counts.items())),
        "measure_categories": dict(sorted(measure_counts.items())),
        "release_scope_tables": len(release_scope_rows),
        "reference_only_tables": len(reference_only_rows),
        "final_clean_tables": len(final_clean_rows),
        "manual_review_tables": len(manual_review_rows),
        "mortality_tables": len(mortality_rows),
        "mortality_non_strict": sum(1 for r in mortality_rows if r["quality_tier"] != "strict_validated"),
    }
    write_json(OUT / "indexes" / "summary.json", payload)

    readme = f"""# Team-Ready Unified Export

This folder is the fast handoff layer for the full release.

It includes:
- the three shipped release datasets
- one unified catalog covering all read tables
- separate release-scope and reference-only splits

Release datasets:
- `datasets/core_curated_v2_final`
- `datasets/tail_exploratory_v2`
- `datasets/assembled_strong_v2`

Use these entrypoints:
- `catalogs/all_tables_master.csv`: concise one-row-per-table index for the whole release
- `catalogs/all_tables_full_metadata.csv`: full source-catalog metadata plus derived QC/review fields
- `catalogs/release_scope_tables.csv`: the shipped release-scope tables
- `catalogs/reference_only_tables.csv`: read tables outside the final release scope
- `catalogs/final_clean_tables.csv`: strict-only, no-manual-review demo subset
- `catalogs/manual_review_queue.csv`: all tables that still warrant later manual review
- `catalogs/mortality_tables.csv`: mortality/death-related subcollection
- `browse/mortality`: dedicated mortality browsing folder
- `browse/review_queue_all`: full review queue with preview folders and app-backed review flow
- `apps/launch_final_submission_review.sh`: launch the general review app from the repo root
- `apps/launch_mortality_review.sh`: launch the mortality review app from the repo root

Confidence tiers:
- `high`: strict validated
- `medium`: contextual unvalidated
- `low`: needs review
- `reference_only`: not in the shipped release scope, but still included here for browsing and follow-up

Manual review signals:
- `needs_manual_review=1` marks tables that are in the review queue or assembled-review overlay
- `assistant_suggested_decision`, `manual_review_decision`, and `manual_final_decision` preserve the current review state

Important note:
- This is a showable unified export, not a deduplicated analytical union.
- Assembled tables can overlap page-level core/tail tables.
- Keep `layer` in any team presentation so people do not assume the three layers are already merged.
- For the safest team demo, start in `final_clean/`.
"""
    (OUT / "README.md").write_text(readme, encoding="utf-8")


if __name__ == "__main__":
    build_outputs()
