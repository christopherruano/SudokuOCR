from __future__ import annotations

import csv
import json
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import xlsxwriter


ROOT = Path("export_final")
SENDOUT = ROOT / "sendout_final"
OUT = ROOT / "final_validated"

SENDOUT_DATA = SENDOUT / "data"
SENDOUT_QA = SENDOUT / "qa"
REVIEW_APP = SENDOUT / "review" / "plausibility_review_app"
EDA_TABLES = SENDOUT / "analysis" / "eda_slideshow" / "tables"
PRELIM = SENDOUT / "analysis" / "preliminary_stats"

TABLES_CONFIRMED = SENDOUT_DATA / "tables_confirmed.csv"
FACTS_CONFIRMED = SENDOUT_DATA / "facts_confirmed.csv"
REVIEW_NEEDED = SENDOUT_DATA / "review_needed.csv"
EXCLUSIONS = SENDOUT_DATA / "exclusions.csv"
AUDIT = SENDOUT_QA / "all_tables_plausibility_audit.csv"
ISSUES = SENDOUT_QA / "all_tables_plausibility_issues.csv"
REVIEW_DECISIONS = REVIEW_APP / "review_decisions.csv"
AGE_PROFILE_UNITS = PRELIM / "age_profile_units.csv"
MORTALITY_DIAGNOSTICS = EDA_TABLES / "mortality_table_diagnostics.csv"

STALE_OUTPUTS = [
    OUT / "data" / "tables_paper_grade.csv",
    OUT / "data" / "facts_paper_grade.csv",
    OUT / "review" / "queues" / "paper_grade_panel_candidates.csv",
]

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

FACT_FIELDS = [
    "fact_id",
    "table_id",
    "section",
    "row_label_raw",
    "row_label_std",
    "row_dimension",
    "row_group_value",
    "column_group_raw",
    "column_group_std",
    "column_dimension",
    "column_group_value",
    "sub_column_raw",
    "sub_column_std",
    "sex",
    "civil_condition",
    "religion",
    "resolved_age_axis",
    "resolved_age_label_std",
    "resolved_age_lower",
    "resolved_age_upper",
    "resolved_age_open_ended",
    "resolved_age_total",
    "resolved_age_not_stated",
    "measure_unit",
    "value",
]

TABLE_FIELDS = [
    "table_id",
    "layer",
    "province",
    "year",
    "title",
    "table_family",
    "format_class",
    "measure_category",
    "constraint_regime",
    "denominator_hint",
    "effective_quality_status",
    "quality_tier",
    "confidence_level",
    "n_checks",
    "failed_checks",
    "qc_flags",
    "source_image_local",
    "source_json_local",
    "source_image_original",
    "source_json_original",
    "facts_rows",
    "excel_path",
]

REGISTRY_FIELDS = [
    "table_id",
    "layer",
    "sendout_bucket",
    "province",
    "year",
    "title",
    "table_family",
    "format_class",
    "measure_category",
    "constraint_regime",
    "denominator_hint",
    "current_quality_tier",
    "effective_quality_status",
    "failed_checks",
    "facts_seen",
    "nonzero_values",
    "plausibility_status",
    "current_recommendation",
    "review_decision",
    "review_notes",
    "issue_flags",
    "existing_qc_flags",
    "source_image_local",
    "source_json_local",
    "excel_path",
    "excel_abs",
    "final_tier",
    "final_disposition",
    "final_action",
    "blocking_reasons",
    "validated_for_fact_export",
    "age_profile_panel_candidate",
    "age_profile_panel_reason",
    "mortality_scale_status",
    "mortality_scale_flags",
    "rerun_priority",
    "review_priority",
]

SUMMARY_FIELDS = ["metric", "value"]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def write_csv(path: Path, rows, fieldnames: list[str]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
            count += 1
    return count


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def remove_stale_outputs() -> None:
    for path in STALE_OUTPUTS:
        if path.exists():
            path.unlink()


def by_id(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {row.get("table_id", ""): row for row in rows if row.get("table_id", "")}


def first_nonblank(*values: object) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""


def truthy(value: object) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def intish(value: object) -> int:
    text = str(value or "").strip()
    if not text:
        return 0
    try:
        return int(float(text))
    except ValueError:
        return 0


def merge_issue_flags(issues_by_id: dict[str, list[dict[str, str]]], table_id: str) -> str:
    flags = []
    for row in issues_by_id.get(table_id, []):
        severity = row.get("severity", "")
        flag = row.get("flag", "")
        detail = row.get("detail", "")
        parts = [part for part in [severity, flag, detail] if part]
        if parts:
            flags.append(":".join(parts))
    return " | ".join(flags)


def load_age_profile_panel_ids() -> set[str]:
    ids = set()
    for row in read_csv(AGE_PROFILE_UNITS):
        if row.get("sex") == "All" and truthy(row.get("quick_descriptive_ready", "")):
            ids.add(row.get("table_id", ""))
    return ids


def mortality_scale_status(mortality_row: dict[str, str] | None) -> tuple[str, str]:
    if not mortality_row:
        return "", ""
    flags = mortality_row.get("flags", "")
    if flags:
        return "scale_review_needed", flags
    measure = mortality_row.get("measure_category", "")
    if measure == "counts":
        return "count", ""
    if measure in {"per_1000_rates", "other_rates"}:
        return "scale_ready_descriptive", ""
    return "unknown", ""


def decide_registry_row(
    table_id: str,
    audit: dict[str, str],
    confirmed: dict[str, str],
    review: dict[str, str],
    review_needed: dict[str, str],
    exclusion: dict[str, str],
    issue_flags: str,
    mortality: dict[str, str] | None,
    age_profile_panel_ids: set[str],
) -> dict[str, str]:
    source = confirmed or review or review_needed or exclusion or audit
    sendout_bucket = first_nonblank(audit.get("sendout_bucket"), review.get("sendout_bucket"), "unknown")
    facts_seen = first_nonblank(audit.get("facts_seen"), review.get("facts_seen"), confirmed.get("facts_rows"), "0")
    recommendation = first_nonblank(review.get("recommendation"), audit.get("recommendation"), review_needed.get("recommendation"))
    review_decision = review.get("review_decision", "")
    mortality_status, mortality_flags = mortality_scale_status(mortality)

    final_tier = "pending_review"
    final_disposition = "pending"
    final_action = "review"
    blocking_reasons: list[str] = []
    validated_for_fact_export = "0"
    age_profile_panel_candidate = "0"
    age_profile_panel_reason = ""
    rerun_priority = ""
    review_priority = ""

    if sendout_bucket == "confirmed":
        final_tier = "validated_confirmed"
        final_disposition = "validated_current_confirmed"
        final_action = "none"
        validated_for_fact_export = "1"
        if table_id in age_profile_panel_ids:
            age_profile_panel_candidate = "1"
            age_profile_panel_reason = "strict_quick_descriptive_age_profile"
        if mortality_flags:
            blocking_reasons.append("mortality_scale_review_needed_for_pooled_analysis")
    else:
        if intish(facts_seen) == 0 or "no_facts_seen" in (audit.get("flags", "") + ";" + review.get("flags", "")):
            final_tier = "pending_rerun_no_output"
            final_disposition = "pending_rerun"
            final_action = "rerun_ocr_or_extraction"
            rerun_priority = "1"
            blocking_reasons.append("no_facts_seen")
        elif review_decision in {"keep", "manual_check_done"}:
            if first_nonblank(audit.get("plausibility_status"), review.get("plausibility_status")) in {"pass", "review_only", "warn"}:
                final_tier = "validated_manual_accept"
                final_disposition = "validated_manual_accept"
                final_action = "none"
                validated_for_fact_export = "1"
            else:
                final_tier = "pending_repair_or_rerun"
                final_disposition = "manual_keep_but_blocked"
                final_action = "repair_or_rerun_then_revalidate"
                blocking_reasons.append("manual_keep_with_blocking_plausibility_status")
        elif review_decision in {"drop", "scope_out"}:
            final_tier = "excluded_manual"
            final_disposition = "manual_excluded"
            final_action = "none"
        elif review_decision == "rerun":
            final_tier = "pending_rerun_requested"
            final_disposition = "pending_rerun"
            final_action = "rerun_ocr_or_extraction"
            rerun_priority = "1"
        elif review_decision == "repair":
            final_tier = "pending_repair_requested"
            final_disposition = "pending_repair"
            final_action = "repair_then_revalidate"
            review_priority = "2"
        elif recommendation == "manual_review_needed_but_number_sense_ok":
            final_tier = "pending_manual_review"
            final_disposition = "pending_visual_acceptance"
            final_action = "manual_visual_review"
            review_priority = "1"
        elif recommendation == "manual_visual_review_required":
            final_tier = "pending_assembled_visual_review"
            final_disposition = "pending_assembled_acceptance"
            final_action = "review_assembled_table"
            review_priority = "1"
        elif recommendation == "repair_or_rerun_before_use":
            final_tier = "pending_repair_or_rerun"
            final_disposition = "pending_repair_or_rerun"
            final_action = "repair_or_rerun_then_revalidate"
            review_priority = "2"
        elif recommendation == "keep_out_of_confirmed_until_scope_decision":
            final_tier = "pending_scope_decision"
            final_disposition = "pending_scope_decision"
            final_action = "decide_scope_then_tier"
            review_priority = "3"
        elif recommendation == "exclude_or_rerun_before_use":
            final_tier = "pending_exclude_or_rerun_decision"
            final_disposition = "pending_exclude_or_rerun"
            final_action = "inspect_then_exclude_or_rerun"
            review_priority = "4"
        else:
            final_tier = "pending_review"
            final_disposition = "pending_review"
            final_action = "review"
            review_priority = "5"
        if issue_flags:
            blocking_reasons.append("plausibility_issues_present")
        if mortality_flags:
            blocking_reasons.append("mortality_scale_review_needed")

    return {
        "table_id": table_id,
        "layer": first_nonblank(source.get("layer"), audit.get("layer")),
        "sendout_bucket": sendout_bucket,
        "province": first_nonblank(source.get("province"), audit.get("province")),
        "year": first_nonblank(source.get("year"), audit.get("year")),
        "title": first_nonblank(source.get("title"), audit.get("title")),
        "table_family": first_nonblank(source.get("table_family"), audit.get("table_family")),
        "format_class": first_nonblank(source.get("format_class"), audit.get("format_class")),
        "measure_category": source.get("measure_category", ""),
        "constraint_regime": source.get("constraint_regime", ""),
        "denominator_hint": source.get("denominator_hint", ""),
        "current_quality_tier": first_nonblank(audit.get("quality_tier"), source.get("quality_tier")),
        "effective_quality_status": source.get("effective_quality_status", ""),
        "failed_checks": first_nonblank(audit.get("failed_checks"), source.get("failed_checks")),
        "facts_seen": facts_seen,
        "nonzero_values": first_nonblank(audit.get("nonzero_values"), review.get("nonzero_values")),
        "plausibility_status": first_nonblank(audit.get("plausibility_status"), review.get("plausibility_status")),
        "current_recommendation": recommendation,
        "review_decision": review_decision,
        "review_notes": review.get("review_notes", ""),
        "issue_flags": first_nonblank(review.get("issue_flags"), issue_flags),
        "existing_qc_flags": first_nonblank(audit.get("existing_qc_flags"), source.get("qc_flags")),
        "source_image_local": first_nonblank(source.get("source_image_local"), review.get("source_image_local")),
        "source_json_local": first_nonblank(source.get("source_json_local"), review.get("source_json_local")),
        "excel_path": first_nonblank(source.get("excel_path"), review.get("excel_path")),
        "excel_abs": review.get("excel_abs", ""),
        "final_tier": final_tier,
        "final_disposition": final_disposition,
        "final_action": final_action,
        "blocking_reasons": ";".join(dict.fromkeys(blocking_reasons)),
        "validated_for_fact_export": validated_for_fact_export,
        "age_profile_panel_candidate": age_profile_panel_candidate,
        "age_profile_panel_reason": age_profile_panel_reason,
        "mortality_scale_status": mortality_status,
        "mortality_scale_flags": mortality_flags,
        "rerun_priority": rerun_priority,
        "review_priority": review_priority,
    }


def build_registry() -> list[dict[str, str]]:
    audit_rows = read_csv(AUDIT)
    confirmed = by_id(read_csv(TABLES_CONFIRMED))
    review = by_id(read_csv(REVIEW_DECISIONS))
    review_needed = by_id(read_csv(REVIEW_NEEDED))
    exclusions = by_id(read_csv(EXCLUSIONS))
    mortality = by_id(read_csv(MORTALITY_DIAGNOSTICS))
    age_profile_panel_ids = load_age_profile_panel_ids()

    issues_by_id: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in read_csv(ISSUES):
        issues_by_id[row.get("table_id", "")].append(row)

    all_ids = []
    seen = set()
    for rows in [audit_rows, list(confirmed.values()), list(review.values()), list(review_needed.values()), list(exclusions.values())]:
        for row in rows:
            table_id = row.get("table_id", "")
            if table_id and table_id not in seen:
                all_ids.append(table_id)
                seen.add(table_id)

    audit_by_id = by_id(audit_rows)
    registry = []
    for table_id in all_ids:
        registry.append(
            decide_registry_row(
                table_id=table_id,
                audit=audit_by_id.get(table_id, {}),
                confirmed=confirmed.get(table_id, {}),
                review=review.get(table_id, {}),
                review_needed=review_needed.get(table_id, {}),
                exclusion=exclusions.get(table_id, {}),
                issue_flags=merge_issue_flags(issues_by_id, table_id),
                mortality=mortality.get(table_id),
                age_profile_panel_ids=age_profile_panel_ids,
            )
        )
    registry.sort(key=lambda r: (r["sendout_bucket"], r["layer"], r["province"], r["year"], r["table_id"]))
    return registry


def stream_facts(validated_ids: set[str], age_profile_panel_ids: set[str]) -> dict[str, int]:
    all_path = OUT / "data" / "facts_all_validated.csv"
    panel_path = OUT / "analysis" / "age_profile_panel" / "facts_age_profile_panel_candidates.csv"
    all_path.parent.mkdir(parents=True, exist_ok=True)
    panel_path.parent.mkdir(parents=True, exist_ok=True)
    counts = Counter()

    with FACTS_CONFIRMED.open(newline="", encoding="utf-8") as src, all_path.open("w", newline="", encoding="utf-8") as all_fh, panel_path.open("w", newline="", encoding="utf-8") as panel_fh:
        reader = csv.DictReader(src)
        all_writer = csv.DictWriter(all_fh, fieldnames=FACT_FIELDS, extrasaction="ignore")
        panel_writer = csv.DictWriter(panel_fh, fieldnames=FACT_FIELDS, extrasaction="ignore")
        all_writer.writeheader()
        panel_writer.writeheader()
        for row in reader:
            table_id = row.get("table_id", "")
            if table_id in validated_ids:
                all_writer.writerow(row)
                counts["facts_all_validated"] += 1
            if table_id in age_profile_panel_ids:
                panel_writer.writerow(row)
                counts["facts_age_profile_panel_candidates"] += 1
    return dict(counts)


def write_registry_outputs(registry: list[dict[str, str]]) -> dict[str, int]:
    counts = Counter()
    write_csv(OUT / "review" / "final_table_registry.csv", registry, REGISTRY_FIELDS)
    counts["registry_tables"] = len(registry)

    queue_specs = {
        "no_output_rerun_queue.csv": lambda r: r["final_tier"] == "pending_rerun_no_output",
        "manual_review_queue.csv": lambda r: r["final_tier"] == "pending_manual_review",
        "assembled_visual_review_queue.csv": lambda r: r["final_tier"] == "pending_assembled_visual_review",
        "repair_or_rerun_queue.csv": lambda r: r["final_tier"] in {"pending_repair_or_rerun", "pending_repair_requested"},
        "scope_decision_queue.csv": lambda r: r["final_tier"] == "pending_scope_decision",
        "exclude_or_rerun_queue.csv": lambda r: r["final_tier"] == "pending_exclude_or_rerun_decision",
        "mortality_scale_review_queue.csv": lambda r: r["mortality_scale_status"] == "scale_review_needed",
        "age_profile_panel_candidates.csv": lambda r: truthy(r["age_profile_panel_candidate"]),
        "tables_pending_action.csv": lambda r: r["final_tier"].startswith("pending_"),
        "tables_excluded_or_unextractable.csv": lambda r: r["final_tier"].startswith("excluded_"),
    }
    for name, predicate in queue_specs.items():
        rows = [r for r in registry if predicate(r)]
        counts[name.replace(".csv", "")] = write_csv(OUT / "review" / "queues" / name, rows, REGISTRY_FIELDS)
    return dict(counts)


def write_table_exports(registry: list[dict[str, str]]) -> tuple[set[str], set[str], dict[str, int]]:
    confirmed_rows = read_csv(TABLES_CONFIRMED)
    confirmed_by_id = by_id(confirmed_rows)
    validated_ids = {r["table_id"] for r in registry if truthy(r["validated_for_fact_export"])}
    age_profile_panel_ids = {r["table_id"] for r in registry if truthy(r["age_profile_panel_candidate"])}

    validated_rows = [confirmed_by_id[table_id] for table_id in sorted(validated_ids) if table_id in confirmed_by_id]
    panel_rows = []
    for table_id in sorted(age_profile_panel_ids):
        if table_id in confirmed_by_id:
            row = dict(confirmed_by_id[table_id])
            row["age_profile_panel_reason"] = "strict_quick_descriptive_age_profile"
            panel_rows.append(row)

    write_csv(OUT / "data" / "tables_all_validated.csv", validated_rows, TABLE_FIELDS)
    write_csv(OUT / "analysis" / "age_profile_panel" / "tables_age_profile_panel_candidates.csv", panel_rows, TABLE_FIELDS + ["age_profile_panel_reason"])

    pending_rows = [r for r in registry if r["final_tier"].startswith("pending_")]
    excluded_rows = [r for r in registry if r["final_tier"].startswith("excluded_")]
    write_csv(OUT / "data" / "tables_pending_action.csv", pending_rows, REGISTRY_FIELDS)
    write_csv(OUT / "data" / "tables_excluded_or_unextractable.csv", excluded_rows, REGISTRY_FIELDS)

    counts = {
        "tables_all_validated": len(validated_rows),
        "tables_age_profile_panel_candidates": len(panel_rows),
        "tables_pending_action": len(pending_rows),
        "tables_excluded_or_unextractable": len(excluded_rows),
    }
    return validated_ids, age_profile_panel_ids, counts


def write_mortality_annotation_template(registry: list[dict[str, str]]) -> int:
    fields = [
        "table_id",
        "province",
        "year",
        "title",
        "format_class",
        "measure_category",
        "denominator_hint",
        "mortality_scale_status",
        "mortality_scale_flags",
        "scale_annotation",
        "annotation_notes",
    ]
    rows = []
    for row in registry:
        if row["mortality_scale_status"] == "scale_review_needed":
            rows.append({field: row.get(field, "") for field in fields})
    return write_csv(OUT / "review" / "mortality_scale_annotation_template.csv", rows, fields)


def write_workbook(registry: list[dict[str, str]], summary_rows: list[dict[str, object]]) -> None:
    path = OUT / "review" / "final_validation_workbook.xlsx"
    path.parent.mkdir(parents=True, exist_ok=True)
    workbook = xlsxwriter.Workbook(str(path), {"constant_memory": True, "strings_to_urls": False})
    header = workbook.add_format({"bold": True, "bg_color": "#E8EEF6", "border": 1})
    normal = workbook.add_format({"border": 1})

    sheets = {
        "summary": (summary_rows, SUMMARY_FIELDS),
        "registry": (registry, REGISTRY_FIELDS),
        "pending": ([r for r in registry if r["final_tier"].startswith("pending_")], REGISTRY_FIELDS),
        "age_profile_panel": ([r for r in registry if truthy(r["age_profile_panel_candidate"])], REGISTRY_FIELDS),
        "mortality_review": ([r for r in registry if r["mortality_scale_status"] == "scale_review_needed"], REGISTRY_FIELDS),
    }
    for sheet_name, (rows, fields) in sheets.items():
        ws = workbook.add_worksheet(sheet_name[:31])
        ws.freeze_panes(1, 0)
        ws.write_row(0, 0, fields, header)
        for r_idx, row in enumerate(rows, start=1):
            ws.write_row(r_idx, 0, [row.get(field, "") for field in fields], normal)
        ws.autofilter(0, 0, len(rows), len(fields) - 1)
        ws.set_column(0, min(len(fields) - 1, 10), 18)
    workbook.close()


def write_docs(report: dict[str, object]) -> None:
    readme = f"""# final_validated

This package is the current implementation of the full-validation export plan.

## Readiness status

`readiness_status`: **{report['readiness_status']}**

The package is intentionally honest: current confirmed facts are exported as validated,
but tables still requiring rerun, repair, scope decisions, assembled review, or mortality
scale annotation remain in explicit queues.

## Main files

- `data/tables_all_validated.csv`: currently validated table catalog.
- `data/facts_all_validated.csv`: facts for currently validated tables.
- `analysis/age_profile_panel/tables_age_profile_panel_candidates.csv`: a small conservative subset used for quick age-profile EDA panels. This is not a correctness boundary.
- `analysis/age_profile_panel/facts_age_profile_panel_candidates.csv`: facts for that EDA subset.
- `data/tables_pending_action.csv`: all tables blocking full readiness.
- `review/final_table_registry.csv`: one row for every audited table.
- `review/queues/`: action-specific queues for rerun, manual review, repair, scope, assembled review, and mortality scale review.
- `qa/final_validation_report.json`: machine-readable readiness report.
- `qa/final_validation_report.md`: human-readable readiness report.

## Tier meanings

- `validated_confirmed`: current validated facts, source-faithful and passing release checks.
- `validated_manual_accept`: manually accepted facts after visual review.
- `excluded_*`: manually excluded or unextractable after process closure.
- `pending_*`: not fully ready yet; see `final_action` and `blocking_reasons`.

The age-profile panel candidate files are intentionally narrow, because they are meant
for one clean descriptive plot family. They do not mean that only those tables are correct.
The full validated export is `data/tables_all_validated.csv` plus `data/facts_all_validated.csv`.
"""
    (OUT / "README.md").write_text(readme, encoding="utf-8")

    lines = [
        "# Final Validation Report",
        "",
        f"- readiness_status: `{report['readiness_status']}`",
        f"- audited_tables_total: `{report['audited_tables_total']}`",
        f"- tables_all_validated: `{report['tables_all_validated']}`",
        f"- facts_all_validated: `{report['facts_all_validated']}`",
        f"- tables_age_profile_panel_candidates: `{report['tables_age_profile_panel_candidates']}`",
        f"- facts_age_profile_panel_candidates: `{report['facts_age_profile_panel_candidates']}`",
        f"- tables_pending_action: `{report['tables_pending_action']}`",
        f"- tables_excluded_or_unextractable: `{report['tables_excluded_or_unextractable']}`",
        "",
        "## Pending By Tier",
        "",
    ]
    for key, value in sorted(report["by_final_tier"].items()):
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Pending By Action", ""])
    for key, value in sorted(report["by_final_action"].items()):
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This package is not yet fully ready for a claim that every audited table has validated facts. It is ready as a controlled validation workspace: all current validated facts are exported, all pending tables are accounted for, and the remaining blockers are explicit.",
        ]
    )
    (OUT / "qa" / "final_validation_report.md").parent.mkdir(parents=True, exist_ok=True)
    (OUT / "qa" / "final_validation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    next_steps = f"""# Next Actions To Reach Full Readiness

Current status: `{report['readiness_status']}`.

## Required closure order

1. Rerun no-output tables.
   - Queue: `review/queues/no_output_rerun_queue.csv`
   - Count: `{report['blocking_summary']['no_output_rerun']}`
   - Run on the cluster/API environment, then rebuild sendout and this package.

2. Fast manual review.
   - Queue: `review/queues/manual_review_queue.csv`
   - Count: `{report['blocking_summary']['manual_review']}`
   - Use the review app and mark `keep` or `drop`.

3. Assembled review.
   - Queue: `review/queues/assembled_visual_review_queue.csv`
   - Count: `{report['blocking_summary']['assembled_visual_review']}`
   - Use the assembled renderer/workbook; accept or reject whole assembled tables.

4. Repair/rerun candidates.
   - Queue: `review/queues/repair_or_rerun_queue.csv`
   - Count: `{report['blocking_summary']['repair_or_rerun']}`
   - Repair extraction or rerun, then revalidate constraints.

5. Scope decisions.
   - Queue: `review/queues/scope_decision_queue.csv`
   - Count: `{report['blocking_summary']['scope_decision']}`
   - Decide whether each table belongs in the project. In-scope tables need validation; out-of-scope tables move to an `excluded_*` tier.

6. Exclude-or-rerun decisions.
   - Queue: `review/queues/exclude_or_rerun_queue.csv`
   - Count: `{report['blocking_summary']['exclude_or_rerun_decision']}`
   - Inspect and either rerun or permanently exclude.

7. Mortality scale annotations.
   - Template: `review/mortality_scale_annotation_template.csv`
   - Count: `{report['blocking_summary']['mortality_scale_review']}`
   - Fill `scale_annotation` before using mortality rates in pooled analysis.

## Rebuild command

```bash
python3 build_final_validated_export.py
```

## Done condition

Full readiness requires:

- `tables_pending_action == 0`
- every audited table has a terminal tier: `validated_confirmed`, `validated_manual_accept`, an `excluded_*` tier, or another explicit validated/excluded terminal status
- no facts refer to excluded tables
- full plausibility audit has zero unresolved fail/warn rows for validated tables
- mortality scale review queue is empty for any mortality table used in pooled rate analysis
"""
    (OUT / "NEXT_ACTIONS.md").write_text(next_steps, encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    remove_stale_outputs()
    registry = build_registry()
    registry_counts = write_registry_outputs(registry)
    validated_ids, age_profile_panel_ids, table_counts = write_table_exports(registry)
    fact_counts = stream_facts(validated_ids, age_profile_panel_ids)
    mortality_annotation_rows = write_mortality_annotation_template(registry)

    by_final_tier = Counter(r["final_tier"] for r in registry)
    by_final_action = Counter(r["final_action"] for r in registry)
    pending_count = sum(1 for r in registry if r["final_tier"].startswith("pending_"))
    readiness_status = "ready" if pending_count == 0 else "not_ready_pending_actions"
    report = {
        "readiness_status": readiness_status,
        "audited_tables_total": len(registry),
        **table_counts,
        **fact_counts,
        "mortality_scale_annotation_template_rows": mortality_annotation_rows,
        "by_final_tier": dict(sorted(by_final_tier.items())),
        "by_final_action": dict(sorted(by_final_action.items())),
        "registry_counts": registry_counts,
        "blocking_summary": {
            "pending_tables": pending_count,
            "no_output_rerun": by_final_tier.get("pending_rerun_no_output", 0),
            "manual_review": by_final_tier.get("pending_manual_review", 0),
            "assembled_visual_review": by_final_tier.get("pending_assembled_visual_review", 0),
            "repair_or_rerun": by_final_tier.get("pending_repair_or_rerun", 0),
            "scope_decision": by_final_tier.get("pending_scope_decision", 0),
            "exclude_or_rerun_decision": by_final_tier.get("pending_exclude_or_rerun_decision", 0),
            "mortality_scale_review": mortality_annotation_rows,
        },
    }
    write_json(OUT / "qa" / "final_validation_report.json", report)
    summary_rows = [{"metric": key, "value": value} for key, value in report.items() if not isinstance(value, (dict, list))]
    write_csv(OUT / "qa" / "final_validation_summary.csv", summary_rows, SUMMARY_FIELDS)
    write_workbook(registry, summary_rows)
    write_docs(report)

    # Preserve the most relevant upstream QA summaries for traceability.
    trace_dir = OUT / "qa" / "upstream"
    trace_dir.mkdir(parents=True, exist_ok=True)
    for path in [
        SENDOUT_QA / "final_counts.json",
        SENDOUT_QA / "all_tables_plausibility_summary.json",
        EDA_TABLES / "table_usage_gap_summary.csv",
    ]:
        if path.exists():
            shutil.copy2(path, trace_dir / path.name)

    print(f"Wrote final validation workspace to {OUT}")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
