#!/usr/bin/env python3
"""Apply assembly review decisions to a release snapshot.

Inputs:
- assembly_grouping_record.csv from the release snapshot.
- optional decisions exported from assembly_review_page.html as JSON or CSV.

Outputs are overlay manifests only. This does not mutate OCR, rulebook evidence,
or the underlying automatic grouping records.
"""

from __future__ import annotations

import argparse
import csv
import json
import zipfile
from collections import Counter
from pathlib import Path

import xlsxwriter


OVERLAY_FIELDS = [
    "overlay_rank",
    "review_key",
    "decision_choice",
    "decision_status",
    "review_queue",
    "source_proposal_id",
    "grouping_id",
    "grouping_layer",
    "current_grouping_status",
    "overlay_status",
    "physical_merge_allowed_after_review",
    "site_use_policy_after_review",
    "source_key",
    "source_pdf_id",
    "page_range",
    "pages_consecutive",
    "core_scope",
    "next_step_type",
    "matrix_csv",
    "title_summary",
    "decision_note",
    "decision_updated_at",
]

PHYSICAL_MANIFEST_FIELDS = [
    "manifest_rank",
    "manifest_status",
    "source_proposal_id",
    "grouping_id",
    "source_key",
    "source_pdf_id",
    "page_range",
    "component_pages",
    "pages_consecutive",
    "matrix_csv",
    "values_policy",
    "decision_choice",
    "decision_note",
    "title_summary",
    "row_pairing_mode",
    "row_label_source_summary",
    "column_order_evidence",
    "zero_check_status",
]

LOGICAL_MANIFEST_FIELDS = [
    "manifest_rank",
    "manifest_status",
    "source_proposal_id",
    "grouping_id",
    "source_key",
    "source_pdf_id",
    "page_range",
    "component_pages",
    "pages_consecutive",
    "logical_collection_only",
    "core_scope",
    "table_number_key",
    "topic_variant_key",
    "religion_terms_detected",
    "title_summary",
    "reason",
    "decision_choice",
    "decision_note",
]

RECOVERY_FIELDS = [
    "recovery_rank",
    "source_proposal_id",
    "grouping_layer",
    "source_key",
    "source_pdf_id",
    "page_range",
    "component_pages",
    "page_gap",
    "missing_intermediate_pages",
    "core_scope",
    "next_step_type",
    "current_action",
    "reason",
    "review_flags",
]

PENDING_FIELDS = [
    "pending_rank",
    "review_key",
    "review_queue",
    "source_proposal_id",
    "source_key",
    "page_range",
    "current_grouping_status",
    "core_scope",
    "next_step_type",
    "matrix_csv",
    "title_summary",
    "reason",
]

UNMATCHED_FIELDS = [
    "unmatched_rank",
    "review_key",
    "decision_choice",
    "source",
    "pages",
    "queue",
    "note",
    "updated_at",
]

SUMMARY_FIELDS = ["metric", "name", "value", "detail"]
AUDIT_FIELDS = ["check", "status", "detail"]

VALID_CHOICES = {
    "promote_structure",
    "keep_separate",
    "needs_image_review",
    "approve_logical",
    "reject_logical",
    "",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def int_value(value: object) -> int:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return 0


def proposal_short_id(source_proposal_id: str) -> str:
    return source_proposal_id.split("::", 1)[1] if "::" in source_proposal_id else source_proposal_id


def review_queue_for_record(record: dict[str, str]) -> str:
    status = record.get("grouping_status", "")
    if status == "physical_structure_review_required":
        return "materialized_review_required"
    if (
        record.get("grouping_layer") == "manual_physical_review"
        and record.get("grouping_status") == "manual_visual_review_before_physical_promotion"
    ):
        if record.get("core_scope") == "core_table_vii":
            return "core_table_vii_manual_preview"
        return "manual_physical_preview"
    if record.get("grouping_layer") == "logical_only" and status == "review_logical_only":
        if record.get("core_scope") == "core_table_vii":
            return "core_table_vii_logical_sibling_review"
        return "logical_sibling_review"
    return ""


def review_key(record: dict[str, str]) -> str:
    queue = review_queue_for_record(record)
    return "::".join(
        [
            queue,
            proposal_short_id(record.get("source_proposal_id", "")),
            record.get("source_key", ""),
            record.get("page_range", ""),
        ]
    )


def load_decisions(path: Path | None) -> tuple[dict[str, dict[str, str]], list[dict[str, str]]]:
    if path is None or not path.exists():
        return {}, []
    if path.suffix.lower() == ".json":
        raw = json.loads(path.read_text(encoding="utf-8"))
        rows = raw if isinstance(raw, list) else raw.get("decisions", [])
    else:
        rows = read_csv(path)

    decisions: dict[str, dict[str, str]] = {}
    for row in rows:
        key = str(row.get("key") or "::".join([
            str(row.get("queue", "")),
            str(row.get("id", "")),
            str(row.get("source", "")),
            str(row.get("pages", "")),
        ]))
        choice = str(row.get("choice", "")).strip()
        decisions[key] = {
            "key": key,
            "choice": choice,
            "queue": str(row.get("queue", "")),
            "source": str(row.get("source", "")),
            "pages": str(row.get("pages", "")),
            "note": str(row.get("note", "")),
            "updated_at": str(row.get("updated_at", "")),
        }
    return decisions, list(decisions.values())


def overlay_status(record: dict[str, str], choice: str) -> tuple[str, str, str]:
    if not choice:
        return "pending_manual_review", "no", "review_before_site_use"
    if choice in {"keep_separate", "reject_logical"}:
        return "manual_rejected_keep_separate", "no", "keep_separate"
    if choice == "needs_image_review":
        return "needs_image_review", "no", "review_before_site_use"
    if choice == "approve_logical":
        if record.get("grouping_layer") == "logical_only" and record.get("logical_collection_only") == "yes":
            return "manual_approved_logical_collection", "no", "use_now_logical_collection_only"
        return "invalid_logical_approval_not_logical_only", "no", "review_before_site_use"
    if choice == "promote_structure":
        if record.get("grouping_layer") in {"physical_pairwise", "manual_physical_review"} and record.get("pages_consecutive") == "yes":
            return "manual_promoted_physical_structure", "yes", "use_now_physical_structure_only"
        return "invalid_promotion_not_physical_pairwise", "no", "review_before_site_use"
    return "invalid_decision_choice", "no", "review_before_site_use"


def values_policy(record: dict[str, str]) -> str:
    return "publish_structure_and_values" if record.get("safe_to_publish_values") == "yes" else "publish_structure_only_values_review"


def build_overlay(
    records: list[dict[str, str]],
    decisions: dict[str, dict[str, str]],
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    reviewable = [record for record in records if review_queue_for_record(record)]
    reviewable_keys = {review_key(record) for record in reviewable}
    overlay_rows: list[dict[str, object]] = []
    pending_rows: list[dict[str, object]] = []
    physical_manifest: list[dict[str, object]] = []
    logical_manifest: list[dict[str, object]] = []
    recovery_manifest: list[dict[str, object]] = []
    unmatched_rows: list[dict[str, object]] = []

    for record in records:
        if record.get("grouping_status") in {"physical_structure_ready", "physical_structure_ready_value_review"}:
            physical_manifest.append(physical_manifest_row(record, "automatic_structure_ready", "", ""))
        if record.get("grouping_status") == "site_ready_logical_only":
            logical_manifest.append(logical_manifest_row(record, "automatic_site_ready_logical", "", ""))
        if record.get("grouping_layer") in {"missing_page_bridge", "wait_for_ocr"}:
            recovery_manifest.append(recovery_row(record))

    for idx, record in enumerate(reviewable, start=1):
        key = review_key(record)
        decision = decisions.get(key, {})
        choice = decision.get("choice", "")
        status, physical_after, site_policy_after = overlay_status(record, choice)
        row = {
            "overlay_rank": idx,
            "review_key": key,
            "decision_choice": choice,
            "decision_status": "decided" if choice else "pending",
            "review_queue": review_queue_for_record(record),
            "source_proposal_id": record.get("source_proposal_id", ""),
            "grouping_id": record.get("grouping_id", ""),
            "grouping_layer": record.get("grouping_layer", ""),
            "current_grouping_status": record.get("grouping_status", ""),
            "overlay_status": status,
            "physical_merge_allowed_after_review": physical_after,
            "site_use_policy_after_review": site_policy_after,
            "source_key": record.get("source_key", ""),
            "source_pdf_id": record.get("source_pdf_id", ""),
            "page_range": record.get("page_range", ""),
            "pages_consecutive": record.get("pages_consecutive", ""),
            "core_scope": record.get("core_scope", ""),
            "next_step_type": record.get("next_step_type", ""),
            "matrix_csv": record.get("matrix_csv", ""),
            "title_summary": record.get("title_summary", ""),
            "decision_note": decision.get("note", ""),
            "decision_updated_at": decision.get("updated_at", ""),
        }
        overlay_rows.append(row)
        if choice == "promote_structure" and physical_after == "yes":
            physical_manifest.append(physical_manifest_row(record, status, choice, decision.get("note", "")))
        elif choice == "approve_logical" and status == "manual_approved_logical_collection":
            logical_manifest.append(logical_manifest_row(record, status, choice, decision.get("note", "")))
        elif not choice or choice == "needs_image_review":
            pending_rows.append(pending_row(record, key, review_queue_for_record(record)))

    for idx, decision in enumerate(decisions.values(), start=1):
        if decision.get("key") not in reviewable_keys:
            unmatched_rows.append(
                {
                    "unmatched_rank": idx,
                    "review_key": decision.get("key", ""),
                    "decision_choice": decision.get("choice", ""),
                    "source": decision.get("source", ""),
                    "pages": decision.get("pages", ""),
                    "queue": decision.get("queue", ""),
                    "note": decision.get("note", ""),
                    "updated_at": decision.get("updated_at", ""),
                }
            )

    for rank, row in enumerate(physical_manifest, start=1):
        row["manifest_rank"] = rank
    for rank, row in enumerate(logical_manifest, start=1):
        row["manifest_rank"] = rank
    for rank, row in enumerate(recovery_manifest, start=1):
        row["recovery_rank"] = rank
    for rank, row in enumerate(pending_rows, start=1):
        row["pending_rank"] = rank
    return overlay_rows, physical_manifest, logical_manifest, recovery_manifest, pending_rows, unmatched_rows


def physical_manifest_row(record: dict[str, str], status: str, choice: str, note: str) -> dict[str, object]:
    return {
        "manifest_rank": "",
        "manifest_status": status,
        "source_proposal_id": record.get("source_proposal_id", ""),
        "grouping_id": record.get("grouping_id", ""),
        "source_key": record.get("source_key", ""),
        "source_pdf_id": record.get("source_pdf_id", ""),
        "page_range": record.get("page_range", ""),
        "component_pages": record.get("component_pages", ""),
        "pages_consecutive": record.get("pages_consecutive", ""),
        "matrix_csv": record.get("matrix_csv", ""),
        "values_policy": values_policy(record),
        "decision_choice": choice,
        "decision_note": note,
        "title_summary": record.get("title_summary", ""),
        "row_pairing_mode": record.get("row_pairing_mode", ""),
        "row_label_source_summary": record.get("row_label_source_summary", ""),
        "column_order_evidence": record.get("column_order_evidence", ""),
        "zero_check_status": record.get("zero_check_status", ""),
    }


def logical_manifest_row(record: dict[str, str], status: str, choice: str, note: str) -> dict[str, object]:
    return {
        "manifest_rank": "",
        "manifest_status": status,
        "source_proposal_id": record.get("source_proposal_id", ""),
        "grouping_id": record.get("grouping_id", ""),
        "source_key": record.get("source_key", ""),
        "source_pdf_id": record.get("source_pdf_id", ""),
        "page_range": record.get("page_range", ""),
        "component_pages": record.get("component_pages", ""),
        "pages_consecutive": record.get("pages_consecutive", ""),
        "logical_collection_only": record.get("logical_collection_only", ""),
        "core_scope": record.get("core_scope", ""),
        "table_number_key": record.get("table_number_key", ""),
        "topic_variant_key": record.get("topic_variant_key", ""),
        "religion_terms_detected": record.get("religion_terms_detected", ""),
        "title_summary": record.get("title_summary", ""),
        "reason": record.get("reason", ""),
        "decision_choice": choice,
        "decision_note": note,
    }


def recovery_row(record: dict[str, str]) -> dict[str, object]:
    return {
        "recovery_rank": "",
        "source_proposal_id": record.get("source_proposal_id", ""),
        "grouping_layer": record.get("grouping_layer", ""),
        "source_key": record.get("source_key", ""),
        "source_pdf_id": record.get("source_pdf_id", ""),
        "page_range": record.get("page_range", ""),
        "component_pages": record.get("component_pages", ""),
        "page_gap": record.get("page_gap", ""),
        "missing_intermediate_pages": record.get("missing_intermediate_pages", ""),
        "core_scope": record.get("core_scope", ""),
        "next_step_type": record.get("next_step_type", ""),
        "current_action": record.get("current_action", ""),
        "reason": record.get("reason", ""),
        "review_flags": record.get("review_flags", ""),
    }


def pending_row(record: dict[str, str], key: str, queue: str) -> dict[str, object]:
    return {
        "pending_rank": "",
        "review_key": key,
        "review_queue": queue,
        "source_proposal_id": record.get("source_proposal_id", ""),
        "source_key": record.get("source_key", ""),
        "page_range": record.get("page_range", ""),
        "current_grouping_status": record.get("grouping_status", ""),
        "core_scope": record.get("core_scope", ""),
        "next_step_type": record.get("next_step_type", ""),
        "matrix_csv": record.get("matrix_csv", ""),
        "title_summary": record.get("title_summary", ""),
        "reason": record.get("reason", ""),
    }


def build_summary(
    records: list[dict[str, str]],
    decisions: dict[str, dict[str, str]],
    overlay_rows: list[dict[str, object]],
    physical_manifest: list[dict[str, object]],
    logical_manifest: list[dict[str, object]],
    recovery_manifest: list[dict[str, object]],
    pending_rows: list[dict[str, object]],
    unmatched_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    rows = [
        {"metric": "counts", "name": "grouping_record_rows", "value": len(records), "detail": ""},
        {"metric": "counts", "name": "reviewable_rows", "value": len(overlay_rows), "detail": "Rows where human decision can alter site use."},
        {"metric": "counts", "name": "decision_rows_loaded", "value": len(decisions), "detail": ""},
        {"metric": "counts", "name": "pending_review_rows", "value": len(pending_rows), "detail": ""},
        {"metric": "counts", "name": "physical_manifest_rows", "value": len(physical_manifest), "detail": "Automatic structure-ready plus manually promoted physical structures."},
        {"metric": "counts", "name": "logical_manifest_rows", "value": len(logical_manifest), "detail": "Logical-only site-ready navigation groups."},
        {"metric": "counts", "name": "recovery_manifest_rows", "value": len(recovery_manifest), "detail": "Missing page/OCR retest worklist."},
        {"metric": "counts", "name": "unmatched_decision_rows", "value": len(unmatched_rows), "detail": "Should be zero when decisions match this snapshot."},
    ]
    rows.extend(counter_rows("decision_choice", Counter(row.get("decision_choice", "") or "pending" for row in overlay_rows)))
    rows.extend(counter_rows("overlay_status", Counter(str(row.get("overlay_status", "")) for row in overlay_rows)))
    rows.extend(counter_rows("physical_manifest_status", Counter(str(row.get("manifest_status", "")) for row in physical_manifest)))
    return rows


def counter_rows(metric: str, counter: Counter[str]) -> list[dict[str, object]]:
    return [
        {"metric": metric, "name": key or "(blank)", "value": value, "detail": ""}
        for key, value in sorted(counter.items())
    ]


def build_audit(
    overlay_rows: list[dict[str, object]],
    physical_manifest: list[dict[str, object]],
    logical_manifest: list[dict[str, object]],
    recovery_manifest: list[dict[str, object]],
    unmatched_rows: list[dict[str, object]],
    workbook_path: Path,
) -> list[dict[str, object]]:
    audits = []
    invalid_choices = [
        str(row.get("review_key", ""))
        for row in overlay_rows
        if str(row.get("decision_choice", "")) not in VALID_CHOICES
    ]
    audits.append({"check": "decision_choices_are_known", "status": "pass" if not invalid_choices else "fail", "detail": f"bad={len(invalid_choices)} examples={'; '.join(invalid_choices[:10])}"})
    audits.append({"check": "decision_rows_match_reviewable_records", "status": "pass" if not unmatched_rows else "fail", "detail": f"unmatched={len(unmatched_rows)} examples={'; '.join(str(row.get('review_key','')) for row in unmatched_rows[:10])}"})
    bad_physical = [
        str(row.get("source_proposal_id", ""))
        for row in physical_manifest
        if row.get("pages_consecutive") != "yes"
    ]
    audits.append({"check": "physical_manifest_rows_are_consecutive", "status": "pass" if not bad_physical else "fail", "detail": f"bad={len(bad_physical)} examples={'; '.join(bad_physical[:10])}"})
    bad_logical = [
        str(row.get("source_proposal_id", ""))
        for row in logical_manifest
        if row.get("logical_collection_only") != "yes"
    ]
    audits.append({"check": "logical_manifest_is_logical_only", "status": "pass" if not bad_logical else "fail", "detail": f"bad={len(bad_logical)} examples={'; '.join(bad_logical[:10])}"})
    bad_recovery = [
        str(row.get("source_proposal_id", ""))
        for row in recovery_manifest
        if row.get("grouping_layer") not in {"missing_page_bridge", "wait_for_ocr"}
    ]
    audits.append({"check": "recovery_manifest_contains_only_retest_work", "status": "pass" if not bad_recovery else "fail", "detail": f"bad={len(bad_recovery)} examples={'; '.join(bad_recovery[:10])}"})
    try:
        with zipfile.ZipFile(workbook_path) as zf:
            bad_member = zf.testzip()
    except Exception as exc:  # pragma: no cover
        bad_member = f"{type(exc).__name__}: {exc}"
    audits.append({"check": "overlay_workbook_zip_integrity", "status": "pass" if bad_member is None else "fail", "detail": str(bad_member or "ok")})
    return audits


def write_workbook(path: Path, sheets: list[tuple[str, list[dict[str, object]], list[str]]]) -> None:
    workbook = xlsxwriter.Workbook(path, {"constant_memory": True})
    header_fmt = workbook.add_format({"bold": True, "bg_color": "#E8EEF7", "border": 1})
    wrap_fmt = workbook.add_format({"text_wrap": True, "valign": "top"})
    for name, rows, fields in sheets:
        ws = workbook.add_worksheet(name[:31])
        for col, field in enumerate(fields):
            ws.write(0, col, field, header_fmt)
        for row_idx, row in enumerate(rows, start=1):
            for col, field in enumerate(fields):
                fmt = wrap_fmt if field in {"title_summary", "reason", "review_flags", "decision_note", "current_action"} else None
                ws.write(row_idx, col, row.get(field, ""), fmt)
        ws.freeze_panes(1, 0)
        if fields:
            ws.autofilter(0, 0, max(0, len(rows)), len(fields) - 1)
        for col, field in enumerate(fields):
            ws.set_column(col, col, min(max(len(field) + 2, 12), 60))
    workbook.close()


def default_decision_path(release_dir: Path) -> Path | None:
    for name in ["assembly_review_decisions.json", "assembly_review_decisions.csv"]:
        path = release_dir / name
        if path.exists():
            return path
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-dir", required=True)
    parser.add_argument("--decisions", default="")
    args = parser.parse_args()

    release_dir = Path(args.release_dir).expanduser().resolve()
    decision_path = Path(args.decisions).expanduser().resolve() if args.decisions else default_decision_path(release_dir)
    records = read_csv(release_dir / "assembly_grouping_record.csv")
    if not records:
        raise SystemExit(f"missing assembly_grouping_record.csv in {release_dir}; run build_assembly_grouping_record.py first")

    decisions, decision_rows = load_decisions(decision_path)
    overlay_rows, physical_manifest, logical_manifest, recovery_manifest, pending_rows, unmatched_rows = build_overlay(records, decisions)
    summary = build_summary(records, decisions, overlay_rows, physical_manifest, logical_manifest, recovery_manifest, pending_rows, unmatched_rows)
    workbook_path = release_dir / "assembly_manual_decision_overlay_workbook.xlsx"
    audit = build_audit(overlay_rows, physical_manifest, logical_manifest, recovery_manifest, unmatched_rows, workbook_path)

    write_csv(release_dir / "assembly_review_decision_overlay.csv", overlay_rows, OVERLAY_FIELDS)
    write_csv(release_dir / "assembly_static_physical_manifest.csv", physical_manifest, PHYSICAL_MANIFEST_FIELDS)
    write_csv(release_dir / "assembly_static_logical_manifest.csv", logical_manifest, LOGICAL_MANIFEST_FIELDS)
    write_csv(release_dir / "assembly_recovery_retest_manifest.csv", recovery_manifest, RECOVERY_FIELDS)
    write_csv(release_dir / "assembly_pending_manual_review.csv", pending_rows, PENDING_FIELDS)
    write_csv(release_dir / "assembly_unmatched_review_decisions.csv", unmatched_rows, UNMATCHED_FIELDS)
    write_csv(release_dir / "assembly_manual_decision_overlay_summary.csv", summary, SUMMARY_FIELDS)
    write_workbook(
        workbook_path,
        [
            ("Decision Overlay", overlay_rows, OVERLAY_FIELDS),
            ("Physical Manifest", physical_manifest, PHYSICAL_MANIFEST_FIELDS),
            ("Logical Manifest", logical_manifest, LOGICAL_MANIFEST_FIELDS),
            ("Recovery Retest", recovery_manifest, RECOVERY_FIELDS),
            ("Pending Review", pending_rows, PENDING_FIELDS),
            ("Unmatched Decisions", unmatched_rows, UNMATCHED_FIELDS),
            ("Summary", summary, SUMMARY_FIELDS),
            ("Audit", audit, AUDIT_FIELDS),
        ],
    )
    audit = build_audit(overlay_rows, physical_manifest, logical_manifest, recovery_manifest, unmatched_rows, workbook_path)
    write_csv(release_dir / "assembly_manual_decision_overlay_audit.csv", audit, AUDIT_FIELDS)
    write_workbook(
        workbook_path,
        [
            ("Decision Overlay", overlay_rows, OVERLAY_FIELDS),
            ("Physical Manifest", physical_manifest, PHYSICAL_MANIFEST_FIELDS),
            ("Logical Manifest", logical_manifest, LOGICAL_MANIFEST_FIELDS),
            ("Recovery Retest", recovery_manifest, RECOVERY_FIELDS),
            ("Pending Review", pending_rows, PENDING_FIELDS),
            ("Unmatched Decisions", unmatched_rows, UNMATCHED_FIELDS),
            ("Summary", summary, SUMMARY_FIELDS),
            ("Audit", audit, AUDIT_FIELDS),
        ],
    )
    print(workbook_path)
    print(json.dumps({
        "decision_file": str(decision_path or ""),
        "reviewable": len(overlay_rows),
        "physical_manifest": len(physical_manifest),
        "logical_manifest": len(logical_manifest),
        "recovery_manifest": len(recovery_manifest),
        "pending_review": len(pending_rows),
        "unmatched_decisions": len(unmatched_rows),
        "audit_failures": sum(1 for row in audit if row.get("status") == "fail"),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
