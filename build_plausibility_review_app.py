from __future__ import annotations

import csv
import json
import shutil
from collections import Counter
from pathlib import Path


ROOT = Path("export_final/sendout_final")
OUT = ROOT / "review" / "plausibility_review_app"
TRIAGE = ROOT / "review" / "review_queue_plausibility_triage.csv"
TABLES_CONFIRMED = ROOT / "data" / "tables_confirmed.csv"
REVIEW_NEEDED = ROOT / "data" / "review_needed.csv"
EXCLUSIONS = ROOT / "data" / "exclusions.csv"
AUDIT = ROOT / "qa" / "all_tables_plausibility_audit.csv"
ISSUES = ROOT / "qa" / "all_tables_plausibility_issues.csv"
FINAL_SUBMISSION_REVIEW = Path("export_final/browse/final_submission_review/review_decisions.csv")
FINAL_SUBMISSION_PREVIEW_ROOT = Path("export_final/browse/final_submission_review")
FAILING_INDEX = Path("export_final/browse/failing_tables/index.csv")
FAILING_PREVIEW_ROOT = Path("export_final/browse/failing_tables")


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


def by_id(rows: list[dict]) -> dict[str, dict]:
    return {r.get("table_id", ""): r for r in rows if r.get("table_id", "")}


def issues_by_id(rows: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for row in rows:
        out.setdefault(row.get("table_id", ""), []).append(row)
    return out


def load_preview_map() -> dict[str, str]:
    preview_map: dict[str, str] = {}
    for row in read_csv(FINAL_SUBMISSION_REVIEW):
        table_id = row.get("table_id", "")
        folder = row.get("preview_folder", "")
        if table_id and folder:
            preview_map[table_id] = str(FINAL_SUBMISSION_PREVIEW_ROOT / folder)
    for row in read_csv(FAILING_INDEX):
        table_id = row.get("table_id", "")
        folder = row.get("folder", "")
        if table_id and folder and table_id not in preview_map:
            preview_map[table_id] = str(FAILING_PREVIEW_ROOT / folder)
    return preview_map


def existing(path: str) -> str:
    if not path:
        return ""
    p = Path(path)
    return str(p) if p.exists() else ""


def build() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    confirmed = by_id(read_csv(TABLES_CONFIRMED))
    review_needed = by_id(read_csv(REVIEW_NEEDED))
    exclusions = by_id(read_csv(EXCLUSIONS))
    audit = by_id(read_csv(AUDIT))
    issues = issues_by_id(read_csv(ISSUES))
    preview_map = load_preview_map()
    triage_rows = read_csv(TRIAGE)

    rows: list[dict] = []
    for idx, row in enumerate(triage_rows):
        table_id = row.get("table_id", "")
        source = confirmed.get(table_id) or review_needed.get(table_id) or exclusions.get(table_id) or {}
        audit_row = audit.get(table_id, {})
        issue_rows = issues.get(table_id, [])
        issue_flags = " | ".join(
            f"{i.get('severity','')}:{i.get('flag','')}" for i in issue_rows[:12]
        )
        issue_details = " || ".join(
            f"{i.get('flag','')}: {i.get('detail','')} {i.get('example','')}".strip()
            for i in issue_rows[:8]
        )
        facts_seen = int(audit_row.get("facts_seen") or 0)
        recommendation = row.get("recommendation", "")
        flags = row.get("flags", "")
        issue_flags_for_row = issue_flags
        issue_details_for_row = issue_details
        if facts_seen == 0:
            recommendation = "rerun_required_no_output"
            flags = ";".join(part for part in [flags, "rerun_required_no_output"] if part)
            issue_flags_for_row = "fail:rerun_required_no_output" + (
                f" | {issue_flags}" if issue_flags else ""
            )
            issue_details_for_row = "rerun_required_no_output: No extracted fact rows were found for this table; treat as due for rerun before any use." + (
                f" || {issue_details}" if issue_details else ""
            )
        excel_path = source.get("excel_path", "")
        excel_abs = str((ROOT / excel_path).resolve()) if excel_path else ""
        preview_dir = preview_map.get(table_id, "")
        source_image_local = existing(source.get("source_image_local", ""))
        source_json_local = existing(source.get("source_json_local", ""))
        rows.append(
            {
                "review_index": idx,
                "table_id": table_id,
                "layer": row.get("layer", ""),
                "province": row.get("province", ""),
                "year": row.get("year", ""),
                "title": row.get("title", ""),
                "sendout_bucket": row.get("sendout_bucket", ""),
                "format_class": row.get("format_class", ""),
                "plausibility_status": row.get("plausibility_status", ""),
                "recommendation": recommendation,
                "flags": flags,
                "issue_flags": issue_flags_for_row,
                "issue_details": issue_details_for_row,
                "failed_checks": row.get("failed_checks", ""),
                "quality_tier": row.get("quality_tier", ""),
                "needs_manual_review": row.get("needs_manual_review", ""),
                "facts_seen": audit_row.get("facts_seen", ""),
                "nonzero_values": audit_row.get("nonzero_values", ""),
                "existing_qc_flags": audit_row.get("existing_qc_flags", ""),
                "source_image_local": source_image_local,
                "source_json_local": source_json_local,
                "source_image_original": source.get("source_image_original", ""),
                "source_json_original": source.get("source_json_original", ""),
                "excel_path": excel_path,
                "excel_abs": excel_abs if excel_abs and Path(excel_abs).exists() else "",
                "preview_dir": preview_dir if preview_dir and Path(preview_dir).exists() else "",
                "preview_matrix": str(Path(preview_dir) / "matrix.csv")
                if preview_dir and (Path(preview_dir) / "matrix.csv").exists()
                else "",
                "preview_facts": str(Path(preview_dir) / "facts.csv")
                if preview_dir and (Path(preview_dir) / "facts.csv").exists()
                else "",
                "preview_summary": str(Path(preview_dir) / "summary.txt")
                if preview_dir and (Path(preview_dir) / "summary.txt").exists()
                else "",
                "review_decision": "",
                "review_notes": "",
            }
        )

    fieldnames = list(rows[0].keys()) if rows else []
    write_csv(OUT / "review_queue.csv", rows, fieldnames)
    write_csv(OUT / "review_decisions.csv", rows, fieldnames)
    write_csv(
        OUT / "review_has_entries.csv",
        [r for r in rows if int(r.get("facts_seen") or 0) > 0],
        fieldnames,
    )
    write_csv(
        OUT / "review_has_entries_no_preview.csv",
        [
            r
            for r in rows
            if int(r.get("facts_seen") or 0) > 0 and not r.get("preview_matrix")
        ],
        fieldnames,
    )
    write_csv(
        OUT / "review_no_facts_regenerate_candidates.csv",
        [
            r
            for r in rows
            if int(r.get("facts_seen") or 0) == 0
        ],
        fieldnames,
    )
    write_csv(
        OUT / "review_no_facts_reference_only.csv",
        [
            r
            for r in rows
            if int(r.get("facts_seen") or 0) == 0
            and r.get("recommendation") == "keep_out_of_confirmed_until_scope_decision"
        ],
        fieldnames,
    )

    counts = {
        "rows": len(rows),
        "by_recommendation": dict(Counter(r["recommendation"] for r in rows)),
        "by_status": dict(Counter(r["plausibility_status"] for r in rows)),
        "with_source_image_local": sum(1 for r in rows if r["source_image_local"]),
        "with_excel": sum(1 for r in rows if r["excel_abs"]),
        "with_preview_matrix": sum(1 for r in rows if r["preview_matrix"]),
        "with_fact_entries": sum(1 for r in rows if int(r.get("facts_seen") or 0) > 0),
        "fact_entries_but_no_preview": sum(
            1 for r in rows if int(r.get("facts_seen") or 0) > 0 and not r["preview_matrix"]
        ),
        "no_fact_regenerate_candidates": sum(1 for r in rows if int(r.get("facts_seen") or 0) == 0),
    }
    (OUT / "summary.json").write_text(json.dumps(counts, indent=2) + "\n", encoding="utf-8")
    (OUT / "README.md").write_text(
        "# Plausibility Review App\n\n"
        "Run from the repo root:\n\n"
        "```bash\npython3 plausibility_review_app.py\n```\n\n"
        "Then open http://127.0.0.1:8768.\n\n"
        "This app reads `review_decisions.csv` and lets you compare the table metadata, "
        "local source image, plausibility issues, generated matrix preview, fact rows, "
        "and generated Excel workbook paths where available.\n\n"
        "Current index summary is in `summary.json`. Decisions are written back into "
        "`review_decisions.csv`.\n\n"
        "Useful queue files:\n"
        "- `review_has_entries.csv`: rows with fact entries to inspect\n"
        "- `review_has_entries_no_preview.csv`: rows that have facts but need preview generation\n"
        "- `review_no_facts_regenerate_candidates.csv`: rows with no facts that are plausible rerun candidates\n"
        "- `review_no_facts_reference_only.csv`: rows with no facts kept only as reference/scope material\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    build()
