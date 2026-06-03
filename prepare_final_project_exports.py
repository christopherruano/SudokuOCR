#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import shutil
import sys
from pathlib import Path

from export_assembled_cell_facts import choose_best_title, title_quality_score


ROOT = Path("export/cluster_sync")
OUT_ROOT = Path("export/final_project_exports")

CORE_SRC = ROOT / "export_inference_final_core_curated_v2"
TAIL_SRC = ROOT / "export_inference_tail_rescue_unreviewed_v2"
ASSEMBLED_SRC = ROOT / "export_assembled_final_core_plus_tail_strong_v2_facts"

CORE_OUT = OUT_ROOT / "core_curated_v2_final"
TAIL_LINK = OUT_ROOT / "tail_exploratory_v2"
ASSEMBLED_LINK = OUT_ROOT / "assembled_strong_v2"
OVERLAYS = OUT_ROOT / "overlays"

CORE_EXCLUDE = {
    "/n/home11/cruano/census_toc/ocr_results/digitized_tables/983ca59647a32810_t0015_Flg.6.2-Growth-Rate-of-ST-population-by-broad-age-group-1981-200-DO--6.6-Age---g/22ba99747e0b79a99591d9457c9383d4fe75e966_oneshot.json": {
        "action": "exclude",
        "reason": "Manifest says non_age_table; province and year unresolved; off-target modern appendix material.",
    }
}

CORE_YEAR_OVERRIDES = {
    "Bengal/unsorted/1926b44e7df6b5e4_t0005_0-i4ind-55-and-above-and-those-in-the-aee-eroup-15-5-.-STATEMENT-6-2-Oepondcncy-__21ec4d0f3771a527a77dff204e8cd4d8c6797b4c_oneshot.json": {
        "year": "1961",
        "reason": "Recovered from source_file path /census_1961/.../Census-Of-Hooghly_text.pdf",
    },
    "Bengal/unsorted/1926b44e7df6b5e4_t0005_0-i4ind-55-and-above-and-those-in-the-aee-eroup-15-5-.-STATEMENT-6-2-Oepondcncy-__ec45b474522416c1c2edd0f4de630171dab29872_oneshot.json": {
        "year": "1961",
        "reason": "Recovered from source_file path /census_1961/.../Census-Of-Hooghly_text.pdf",
    },
}

ASSEMBLED_SUSPICIOUS_PATTERNS = [
    "towns and villages classified by population",
    "population density (persons per square mile)",
    "literate by caste",
    "comparison of combined figures of literate",
    "reported deai",
]

ASSEMBLED_SUSPICIOUS_MEMBER_PATTERNS = [
    "infirmities by castes, etc.",
    "general distribution of the population by religions",
]


def ensure_clean_dir(path: Path) -> None:
    if path.exists() or path.is_symlink():
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        else:
            path.unlink()
    path.mkdir(parents=True, exist_ok=True)


def maybe_symlink(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    dst.symlink_to(src.resolve())


def patch_core_source_row(row: dict[str, str]) -> dict[str, str] | None:
    source_json = row.get("source_json") or ""
    if source_json in CORE_EXCLUDE:
        return None
    if source_json in CORE_YEAR_OVERRIDES:
        row["year"] = CORE_YEAR_OVERRIDES[source_json]["year"]
        if row.get("raw_metadata") == "":
            pass
    return row


def patch_jsonish_year(container: object, year_value: str) -> object:
    if not isinstance(container, dict):
        return container
    for key in ("year",):
        if key in container:
            container[key] = year_value
    return container


def rewrite_core_export() -> dict[str, int]:
    ensure_clean_dir(CORE_OUT)
    stats = {
        "source_rows_in": 0,
        "source_rows_out": 0,
        "fact_rows_in": 0,
        "fact_rows_out": 0,
        "excluded_source_rows": 0,
        "patched_year_source_rows": 0,
        "patched_year_fact_rows": 0,
    }

    # small files copied directly
    for name in ["codebook.txt", "quarantine_rows.csv", "quarantine_sources.csv"]:
        src = CORE_SRC / name
        if src.exists():
            dst = CORE_OUT / name
            if dst.exists():
                dst.unlink()
            shutil.copy2(src, dst)

    # source_catalog.csv
    src = CORE_SRC / "source_catalog.csv"
    dst = CORE_OUT / "source_catalog.csv"
    with src.open(newline="", encoding="utf-8") as f_in, dst.open("w", newline="", encoding="utf-8") as f_out:
        reader = csv.DictReader(f_in)
        writer = csv.DictWriter(f_out, fieldnames=reader.fieldnames)
        writer.writeheader()
        for row in reader:
            stats["source_rows_in"] += 1
            source_json = row.get("source_json") or ""
            patched = patch_core_source_row(dict(row))
            if patched is None:
                stats["excluded_source_rows"] += 1
                continue
            if source_json in CORE_YEAR_OVERRIDES:
                stats["patched_year_source_rows"] += 1
            writer.writerow(patched)
            stats["source_rows_out"] += 1

    # source_catalog.jsonl
    src = CORE_SRC / "source_catalog.jsonl"
    dst = CORE_OUT / "source_catalog.jsonl"
    with src.open(encoding="utf-8") as f_in, dst.open("w", encoding="utf-8") as f_out:
        for line in f_in:
            row = json.loads(line)
            source_json = row.get("source_json") or ""
            if source_json in CORE_EXCLUDE:
                continue
            if source_json in CORE_YEAR_OVERRIDES:
                year_value = CORE_YEAR_OVERRIDES[source_json]["year"]
                row["year"] = year_value
                row["raw_metadata"] = patch_jsonish_year(row.get("raw_metadata"), year_value)
                row["raw_queue_row"] = patch_jsonish_year(row.get("raw_queue_row"), year_value)
            f_out.write(json.dumps(row, ensure_ascii=False) + "\n")

    # fact tables
    for name in ["cell_facts.csv", "cell_facts_rich.csv"]:
        src = CORE_SRC / name
        dst = CORE_OUT / name
        with src.open(newline="", encoding="utf-8") as f_in, dst.open("w", newline="", encoding="utf-8") as f_out:
            reader = csv.DictReader(f_in)
            writer = csv.DictWriter(f_out, fieldnames=reader.fieldnames)
            writer.writeheader()
            for row in reader:
                stats["fact_rows_in"] += 1 if name == "cell_facts.csv" else 0
                source_json = row.get("source_json") or ""
                if source_json in CORE_EXCLUDE:
                    continue
                if source_json in CORE_YEAR_OVERRIDES:
                    row["year"] = CORE_YEAR_OVERRIDES[source_json]["year"]
                    stats["patched_year_fact_rows"] += 1 if name == "cell_facts.csv" else 0
                writer.writerow(row)
                stats["fact_rows_out"] += 1 if name == "cell_facts.csv" else 0

    return stats


def build_core_actions_csv() -> None:
    path = OVERLAYS / "core_curation_actions.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["source_json", "action", "new_year", "reason"],
        )
        writer.writeheader()
        for source_json, info in CORE_EXCLUDE.items():
            writer.writerow(
                {
                    "source_json": source_json,
                    "action": info["action"],
                    "new_year": "",
                    "reason": info["reason"],
                }
            )
        for source_json, info in CORE_YEAR_OVERRIDES.items():
            writer.writerow(
                {
                    "source_json": source_json,
                    "action": "patch_year",
                    "new_year": info["year"],
                    "reason": info["reason"],
                }
            )


def assembled_suspicious_row(row: dict[str, str]) -> bool:
    title = (row.get("title") or "").lower()
    members = (row.get("compiled_member_titles") or "").lower()
    if any(pat in title for pat in ASSEMBLED_SUSPICIOUS_PATTERNS):
        return True
    if any(pat in members for pat in ASSEMBLED_SUSPICIOUS_MEMBER_PATTERNS):
        return True
    return False


def build_assembled_review_files() -> tuple[int, int]:
    suspicious_path = OVERLAYS / "assembled_suspicious_groups.csv"
    review_path = OVERLAYS / "assembled_title_review.csv"
    suspicious_count = 0
    override_suggestions = 0
    csv.field_size_limit(sys.maxsize)
    with ASSEMBLED_SRC.joinpath("source_catalog.csv").open(newline="", encoding="utf-8") as f_in, \
            suspicious_path.open("w", newline="", encoding="utf-8") as f_susp, \
            review_path.open("w", newline="", encoding="utf-8") as f_rev:
        reader = csv.DictReader(f_in)
        susp_fields = [
            "table_id",
            "table_family",
            "title",
            "compiled_member_count",
            "assembled_failure_improvement",
            "compiled_member_titles",
            "compiled_member_queue_titles",
            "review_note",
        ]
        rev_fields = [
            "table_id",
            "table_family",
            "current_title",
            "current_title_score",
            "suggested_title",
            "suggested_title_score",
            "suggest_override",
            "compiled_member_titles",
            "compiled_member_queue_titles",
        ]
        susp_writer = csv.DictWriter(f_susp, fieldnames=susp_fields)
        rev_writer = csv.DictWriter(f_rev, fieldnames=rev_fields)
        susp_writer.writeheader()
        rev_writer.writeheader()

        for row in reader:
            if str(row.get("keep_for_inference") or "").lower() not in {"1", "true", "yes"}:
                continue
            cur = row.get("title") or ""
            candidates = []
            for field in ["compiled_member_titles", "compiled_member_queue_titles"]:
                txt = row.get(field) or ""
                candidates.extend([x.strip() for x in txt.split(" | ") if x.strip()])
            best = choose_best_title(candidates + [cur], fallback=cur)
            cur_score = title_quality_score(cur)
            best_score = title_quality_score(best)
            suggest_override = int(best != cur and best_score >= cur_score + 15 and best_score >= 100)
            if suggest_override:
                override_suggestions += 1
            rev_writer.writerow(
                {
                    "table_id": row.get("table_id"),
                    "table_family": row.get("table_family"),
                    "current_title": cur,
                    "current_title_score": f"{cur_score:.1f}",
                    "suggested_title": best,
                    "suggested_title_score": f"{best_score:.1f}",
                    "suggest_override": suggest_override,
                    "compiled_member_titles": row.get("compiled_member_titles"),
                    "compiled_member_queue_titles": row.get("compiled_member_queue_titles"),
                }
            )
            if assembled_suspicious_row(row):
                suspicious_count += 1
                susp_writer.writerow(
                    {
                        "table_id": row.get("table_id"),
                        "table_family": row.get("table_family"),
                        "title": cur,
                        "compiled_member_count": row.get("compiled_member_count"),
                        "assembled_failure_improvement": row.get("assembled_failure_improvement"),
                        "compiled_member_titles": row.get("compiled_member_titles"),
                        "compiled_member_queue_titles": row.get("compiled_member_queue_titles"),
                        "review_note": "Likely needs human review before using assembled version in analysis.",
                    }
                )
    return suspicious_count, override_suggestions


def write_readme(core_stats: dict[str, int], suspicious_count: int, override_suggestions: int) -> None:
    path = OUT_ROOT / "README.md"
    with path.open("w", encoding="utf-8") as f:
        f.write("# Final Project Exports\n\n")
        f.write("This directory is the project-facing export package.\n\n")
        f.write("## Authoritative layers\n")
        f.write(f"- `core_curated_v2_final/`: cleaned core export. One obvious off-target source was removed and two Hooghly/Bengal sources had `year=1961` patched from the source PDF path.\n")
        f.write(f"- `tail_exploratory_v2/`: symlink to the tail rescue export. Use `resolved_age_*` for analysis.\n")
        f.write(f"- `assembled_strong_v2/`: symlink to the strong assembled multipage export. Use with the review overlays below.\n\n")
        f.write("## Recommended columns\n")
        f.write("- Use `resolved_age_*` as the canonical age fields.\n")
        f.write("- Use `source_catalog.csv` as the authoritative table-level metadata, especially for assembled tables.\n")
        f.write("- Use `table_id` for joins.\n\n")
        f.write("## Overlay / review files\n")
        f.write("- `overlays/core_curation_actions.csv`: source-level exclusions and metadata patches applied to the cleaned core export.\n")
        f.write(f"- `overlays/assembled_suspicious_groups.csv`: {suspicious_count} assembled groups that should be treated cautiously.\n")
        f.write(f"- `overlays/assembled_title_review.csv`: all 34 assembled groups with title scores and {override_suggestions} automatic override suggestions.\n\n")
        f.write("## Core export summary\n")
        f.write(f"- source rows in/out: {core_stats['source_rows_in']} -> {core_stats['source_rows_out']}\n")
        f.write(f"- excluded source rows: {core_stats['excluded_source_rows']}\n")
        f.write(f"- year-patched source rows: {core_stats['patched_year_source_rows']}\n")
        f.write(f"- fact rows in/out: {core_stats['fact_rows_in']} -> {core_stats['fact_rows_out']}\n")


def write_manifest() -> None:
    manifest = {
        "core": str(CORE_OUT),
        "tail": str(TAIL_LINK),
        "assembled": str(ASSEMBLED_LINK),
        "overlays": str(OVERLAYS),
    }
    (OUT_ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    ensure_clean_dir(OUT_ROOT)
    OVERLAYS.mkdir(parents=True, exist_ok=True)
    core_stats = rewrite_core_export()
    maybe_symlink(TAIL_SRC, TAIL_LINK)
    maybe_symlink(ASSEMBLED_SRC, ASSEMBLED_LINK)
    build_core_actions_csv()
    suspicious_count, override_suggestions = build_assembled_review_files()
    write_readme(core_stats, suspicious_count, override_suggestions)
    write_manifest()
    print("Wrote final project exports to:", OUT_ROOT)
    print(json.dumps(
        {
            "core_source_rows_out": core_stats["source_rows_out"],
            "core_fact_rows_out": core_stats["fact_rows_out"],
            "assembled_suspicious_groups": suspicious_count,
            "assembled_override_suggestions": override_suggestions,
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
