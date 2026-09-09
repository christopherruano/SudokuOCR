#!/usr/bin/env python3
"""Build the consolidated assembly recommendations manifest.

This is the operational layer for consumers such as the static site. It combines
the strict rulebook, source sequence map, and QA audits into explicit use
policies. It does not OCR and does not promote candidates into physical merges.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import zipfile
from collections import Counter
from pathlib import Path
from typing import Iterable

import xlsxwriter


ROOT = Path(__file__).resolve().parent
DEFAULT_RULEBOOK_DIR = ROOT / "export_final/pdf_recovery_incoming_2026-06-16_first_principles_assembly_rulebook_v34"
DEFAULT_SEQUENCE_DIR = ROOT / "export_final/pdf_recovery_incoming_2026-06-16_first_principles_source_sequence_map_v35"
DEFAULT_QUALITY_DIR = ROOT / "export_final/pdf_recovery_incoming_2026-06-16_first_principles_assembly_quality_audit_v36"
DEFAULT_OUT_DIR = ROOT / "export_final/pdf_recovery_incoming_2026-06-16_first_principles_assembly_recommendations_v37"


RECOMMENDATION_FIELDS = [
    "recommendation_rank",
    "recommendation_id",
    "source_record_id",
    "record_type",
    "recommendation_status",
    "site_use_policy",
    "physical_merge_allowed_now",
    "logical_collection_only",
    "safe_to_publish_values",
    "review_required",
    "priority",
    "v5_source_key",
    "source_pdf_id_guess",
    "page_range",
    "component_pages",
    "n_pages",
    "edge_pairs",
    "member_count",
    "religion_terms_detected",
    "qa_bucket",
    "qa_flags",
    "primary_title",
    "reason",
    "recommended_action",
    "evidence_files",
]

RUN_CONTEXT_FIELDS = [
    "context_rank",
    "run_id",
    "run_type",
    "run_status",
    "site_use_policy",
    "physical_merge_allowed_now",
    "v5_source_key",
    "source_pdf_id_guess",
    "page_range",
    "component_pages",
    "n_pages",
    "edge_pairs",
    "n_edges",
    "edge_statuses",
    "row_label_dependency_edge_count",
    "missing_page_count",
    "missing_pages",
    "recommended_action",
]

SUMMARY_FIELDS = ["metric", "name", "count"]
AUDIT_FIELDS = ["check", "status", "detail"]


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


def output_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def stable_id(prefix: str, *parts: object) -> str:
    payload = "||".join(str(part) for part in parts)
    return f"{prefix}_{hashlib.sha1(payload.encode('utf-8')).hexdigest()[:16]}"


def int_or_none(value: object) -> int | None:
    if value in {None, ""}:
        return None
    try:
        return int(str(value))
    except ValueError:
        return None


def semi(values: Iterable[object]) -> str:
    return "; ".join(str(value) for value in values if value not in {None, ""})


def recommendation_for_group(group: dict[str, str], qa_by_group: dict[str, dict[str, str]]) -> dict[str, object]:
    group_id = group.get("group_id", "")
    group_type = group.get("group_type", "")
    group_status = group.get("group_status", "")
    qa = qa_by_group.get(group_id, {})
    qa_bucket = qa.get("audit_bucket", "")
    overbroad = qa.get("overbroad_risk") == "yes"

    status = "review_only"
    policy = "review_before_site_use"
    priority = 50
    review_required = group.get("review_required", "yes")
    qa_flags: list[str] = []
    action = group.get("recommended_action", "")
    reason = group.get("reason", "")

    if group_type == "physical_component":
        status = "site_ready_physical_component"
        policy = "use_now_physical_component_structure"
        priority = 1
        review_required = group.get("review_required", "")
        qa_flags.append("confirmed_consecutive_physical")
    elif group_type == "physical_candidate_edge":
        status = f"future_{group_status}"
        policy = "do_not_merge_until_edge_promotes"
        priority = 20
        review_required = "yes"
        qa_flags.append("pairwise_candidate_only")
    elif group_type == "logical_family_collection":
        if overbroad:
            status = "manual_split_required_logical_family"
            policy = "do_not_use_as_single_group_until_split"
            priority = 5
            review_required = "yes"
            qa_flags.extend(["overbroad_risk", qa_bucket])
            action = qa.get("recommended_action", action)
        elif group_status == "confirmed":
            status = "site_ready_logical_collection"
            policy = "use_now_logical_collection_only"
            priority = 3
            review_required = "no"
            qa_flags.append("confirmed_logical_only")
        elif qa_bucket == "religion_family_review_candidate":
            status = "review_bounded_religion_family"
            policy = "review_before_site_use_logical_only"
            priority = 7
            review_required = "yes"
            qa_flags.append("bounded_religion_review")
        else:
            status = "review_logical_collection"
            policy = "review_before_site_use_logical_only"
            priority = 10
            review_required = "yes"
            qa_flags.append(qa_bucket or "logical_review_candidate")
    else:
        qa_flags.append("unknown_record_type")

    return {
        "recommendation_id": stable_id("v37rec", group_id, status),
        "source_record_id": group_id,
        "record_type": group_type,
        "recommendation_status": status,
        "site_use_policy": policy,
        "physical_merge_allowed_now": group.get("physical_merge_allowed_now", "no") if group_type == "physical_component" else "no",
        "logical_collection_only": group.get("logical_collection_only", ""),
        "safe_to_publish_values": group.get("safe_to_publish_values", ""),
        "review_required": review_required,
        "priority": priority,
        "v5_source_key": group.get("v5_source_key", ""),
        "source_pdf_id_guess": group.get("source_pdf_id_guess", ""),
        "page_range": group.get("page_range", ""),
        "component_pages": group.get("component_pages", ""),
        "n_pages": group.get("n_pages", ""),
        "edge_pairs": group.get("edge_pairs", ""),
        "member_count": group.get("member_count", ""),
        "religion_terms_detected": group.get("religion_terms_detected", ""),
        "qa_bucket": qa_bucket,
        "qa_flags": semi(qa_flags),
        "primary_title": group.get("primary_title", ""),
        "reason": reason,
        "recommended_action": action,
        "evidence_files": "grouping_registry.csv; logical_family_quality_audit.csv",
    }


def recommendation_for_missing_religion(row: dict[str, str]) -> dict[str, object]:
    return {
        "recommendation_id": f"logical_missing::{row.get('sequence_id', '')}",
        "source_record_id": row.get("sequence_id", ""),
        "record_type": "logical_family_candidate_added_by_qa",
        "recommendation_status": "review_missing_core_religion_family",
        "site_use_policy": "review_before_site_use_logical_only",
        "physical_merge_allowed_now": "no",
        "logical_collection_only": "yes",
        "safe_to_publish_values": "not_applicable",
        "review_required": "yes",
        "priority": 6,
        "v5_source_key": row.get("v5_source_key", ""),
        "source_pdf_id_guess": row.get("source_pdf_id_guess", ""),
        "page_range": row.get("page_range", ""),
        "component_pages": row.get("component_pages", ""),
        "n_pages": row.get("n_pages", ""),
        "edge_pairs": "",
        "member_count": row.get("n_pages", ""),
        "religion_terms_detected": row.get("religion_terms_detected", ""),
        "qa_bucket": row.get("coverage_bucket", ""),
        "qa_flags": "possible_missing_core_religion_family; logical_only; no_physical_merge",
        "primary_title": row.get("titles", ""),
        "reason": "QA found a bounded same-frame core Table VII/age-civil religion sequence not covered by the logical family index.",
        "recommended_action": row.get("recommended_action", ""),
        "evidence_files": "religion_sequence_coverage_audit.csv",
    }


def build_recommendations(groups: list[dict[str, str]], logical_qa: list[dict[str, str]], religion_qa: list[dict[str, str]]) -> list[dict[str, object]]:
    qa_by_group = {row.get("group_id", ""): row for row in logical_qa if row.get("group_id")}
    rows = [recommendation_for_group(group, qa_by_group) for group in groups]
    rows.extend(
        recommendation_for_missing_religion(row)
        for row in religion_qa
        if row.get("coverage_bucket") == "possible_missing_religion_family"
    )
    rows.sort(
        key=lambda row: (
            int_or_none(row.get("priority")) or 99,
            str(row.get("v5_source_key", "")),
            int_or_none(str(row.get("page_range", "")).split("-", 1)[0]) or 10**9,
            str(row.get("recommendation_id", "")),
        )
    )
    for idx, row in enumerate(rows, start=1):
        row["recommendation_rank"] = idx
    return rows


def build_run_context(runs: list[dict[str, str]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for run in runs:
        run_type = run.get("run_type", "")
        if run_type in {"hard_break_or_other", "logical_only"}:
            continue
        policy = "do_not_merge_until_edge_promotes"
        if run_type == "confirmed_physical":
            policy = "use_now_physical_component_structure"
        elif run_type == "missing_page_bridge_candidate":
            policy = "restore_missing_pages_then_retest_edges"
        rows.append(
            {
                "run_id": run.get("run_id", ""),
                "run_type": run_type,
                "run_status": run.get("run_status", ""),
                "site_use_policy": policy,
                "physical_merge_allowed_now": run.get("physical_merge_allowed_now", "no") if run_type == "confirmed_physical" else "no",
                "v5_source_key": run.get("v5_source_key", ""),
                "source_pdf_id_guess": run.get("source_pdf_id_guess", ""),
                "page_range": run.get("page_range", ""),
                "component_pages": run.get("component_pages", ""),
                "n_pages": run.get("n_pages", ""),
                "edge_pairs": run.get("edge_pairs", ""),
                "n_edges": run.get("n_edges", ""),
                "edge_statuses": run.get("edge_statuses", ""),
                "row_label_dependency_edge_count": run.get("row_label_dependency_edge_count", ""),
                "missing_page_count": run.get("missing_page_count", ""),
                "missing_pages": run.get("missing_pages", ""),
                "recommended_action": run.get("recommended_action", ""),
            }
        )
    rows.sort(key=lambda row: (str(row.get("v5_source_key", "")), int_or_none(str(row.get("page_range", "")).split("-", 1)[0]) or 10**9))
    for idx, row in enumerate(rows, start=1):
        row["context_rank"] = idx
    return rows


def build_summary(recs: list[dict[str, object]], run_context: list[dict[str, object]], paths: dict[str, Path]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = [{"metric": "paths", "name": name, "count": output_path(path)} for name, path in paths.items()]
    rows.extend(
        [
            {"metric": "counts", "name": "recommendations", "count": len(recs)},
            {"metric": "counts", "name": "run_context_rows", "count": len(run_context)},
        ]
    )
    for metric, counter in [
        ("recommendations_by_status", Counter(str(row.get("recommendation_status", "")) for row in recs)),
        ("recommendations_by_policy", Counter(str(row.get("site_use_policy", "")) for row in recs)),
        ("recommendations_by_record_type", Counter(str(row.get("record_type", "")) for row in recs)),
        ("run_context_by_type", Counter(str(row.get("run_type", "")) for row in run_context)),
    ]:
        for name, count in sorted(counter.items()):
            rows.append({"metric": metric, "name": name, "count": count})
    return rows


def audit_outputs(recs: list[dict[str, object]], workbook_path: Path) -> list[dict[str, object]]:
    audits: list[dict[str, object]] = []
    unsafe_physical = [
        row for row in recs
        if row.get("physical_merge_allowed_now") == "yes" and row.get("recommendation_status") != "site_ready_physical_component"
    ]
    audits.append({"check": "only_site_ready_physical_components_allow_physical_merge", "status": "pass" if not unsafe_physical else "fail", "detail": f"bad={len(unsafe_physical)}"})
    overbroad_site = [
        row for row in recs
        if "overbroad_risk" in str(row.get("qa_flags", "")) and str(row.get("site_use_policy", "")).startswith("use_now")
    ]
    audits.append({"check": "overbroad_logical_families_not_site_ready", "status": "pass" if not overbroad_site else "fail", "detail": f"bad={len(overbroad_site)}"})
    missing_added = [row for row in recs if row.get("record_type") == "logical_family_candidate_added_by_qa"]
    bad_missing_added = [
        row for row in missing_added
        if row.get("physical_merge_allowed_now") == "yes"
        or row.get("logical_collection_only") != "yes"
        or not str(row.get("site_use_policy", "")).startswith("review_before")
    ]
    audits.append({"check": "qa_added_religion_candidates_are_review_only", "status": "pass" if not bad_missing_added else "fail", "detail": f"added={len(missing_added)} bad={len(bad_missing_added)}"})
    candidate_ready = [
        row for row in recs
        if row.get("record_type") == "physical_candidate_edge" and str(row.get("site_use_policy", "")).startswith("use_now")
    ]
    audits.append({"check": "physical_candidates_not_site_ready", "status": "pass" if not candidate_ready else "fail", "detail": f"bad={len(candidate_ready)}"})
    try:
        with zipfile.ZipFile(workbook_path) as zf:
            bad_member = zf.testzip()
    except Exception as exc:  # pragma: no cover
        bad_member = f"{type(exc).__name__}: {exc}"
    audits.append({"check": "workbook_zip_integrity", "status": "pass" if bad_member is None else "fail", "detail": str(bad_member or "ok")})
    return audits


def write_workbook(path: Path, summary: list[dict[str, object]], audits: list[dict[str, object]], recs: list[dict[str, object]], run_context: list[dict[str, object]]) -> None:
    workbook = xlsxwriter.Workbook(path, {"constant_memory": True})
    header_fmt = workbook.add_format({"bold": True, "bg_color": "#E8EEF7", "border": 1})
    wrap_fmt = workbook.add_format({"text_wrap": True, "valign": "top"})
    sheets = [
        ("Summary", summary, SUMMARY_FIELDS),
        ("Audit", audits, AUDIT_FIELDS),
        ("All Recommendations", recs, RECOMMENDATION_FIELDS),
        ("Site Ready", [row for row in recs if str(row.get("site_use_policy", "")).startswith("use_now")], RECOMMENDATION_FIELDS),
        ("Review Required", [row for row in recs if row.get("review_required") == "yes"], RECOMMENDATION_FIELDS),
        ("Run Context", run_context, RUN_CONTEXT_FIELDS),
    ]
    for name, rows, fields in sheets:
        ws = workbook.add_worksheet(name[:31])
        for col, field in enumerate(fields):
            ws.write(0, col, field, header_fmt)
        for row_idx, row in enumerate(rows, start=1):
            for col, field in enumerate(fields):
                value = row.get(field, "")
                fmt = wrap_fmt if field in {"primary_title", "reason", "recommended_action", "qa_flags"} else None
                ws.write(row_idx, col, value, fmt)
        ws.freeze_panes(1, 0)
        ws.autofilter(0, 0, max(0, len(rows)), max(0, len(fields) - 1))
        for col, field in enumerate(fields):
            ws.set_column(col, col, min(max(len(field) + 2, 12), 54))
    workbook.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rulebook-dir", default=str(DEFAULT_RULEBOOK_DIR))
    parser.add_argument("--sequence-dir", default=str(DEFAULT_SEQUENCE_DIR))
    parser.add_argument("--quality-dir", default=str(DEFAULT_QUALITY_DIR))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    args = parser.parse_args()

    rulebook_dir = Path(args.rulebook_dir).expanduser()
    sequence_dir = Path(args.sequence_dir).expanduser()
    quality_dir = Path(args.quality_dir).expanduser()
    out_dir = Path(args.out_dir).expanduser()
    paths = {"rulebook_dir": rulebook_dir, "sequence_dir": sequence_dir, "quality_dir": quality_dir, "out_dir": out_dir}

    groups = read_csv(rulebook_dir / "grouping_registry.csv")
    runs = read_csv(sequence_dir / "assembly_runs.csv")
    logical_qa = read_csv(quality_dir / "logical_family_quality_audit.csv")
    religion_qa = read_csv(quality_dir / "religion_sequence_coverage_audit.csv")
    recommendations = build_recommendations(groups, logical_qa, religion_qa)
    run_context = build_run_context(runs)
    summary_rows = build_summary(recommendations, run_context, paths)

    out_dir.mkdir(parents=True, exist_ok=True)
    workbook_path = out_dir / "assembly_recommendations_workbook.xlsx"
    write_workbook(workbook_path, summary_rows, [], recommendations, run_context)
    audit_rows = audit_outputs(recommendations, workbook_path)
    summary_rows.extend({"metric": "audits", "name": row["check"], "count": row["status"]} for row in audit_rows)
    write_workbook(workbook_path, summary_rows, audit_rows, recommendations, run_context)
    audit_rows = audit_outputs(recommendations, workbook_path)

    write_csv(out_dir / "assembly_recommendations.csv", recommendations, RECOMMENDATION_FIELDS)
    write_csv(out_dir / "site_ready_groupings.csv", [row for row in recommendations if str(row.get("site_use_policy", "")).startswith("use_now")], RECOMMENDATION_FIELDS)
    write_csv(out_dir / "manual_review_groupings.csv", [row for row in recommendations if row.get("review_required") == "yes"], RECOMMENDATION_FIELDS)
    write_csv(out_dir / "future_physical_candidate_runs.csv", run_context, RUN_CONTEXT_FIELDS)
    write_csv(out_dir / "summary.csv", summary_rows, SUMMARY_FIELDS)
    write_csv(out_dir / "audit.csv", audit_rows, AUDIT_FIELDS)

    summary = {
        "out_dir": output_path(out_dir),
        "recommendations": len(recommendations),
        "site_ready_groupings": sum(1 for row in recommendations if str(row.get("site_use_policy", "")).startswith("use_now")),
        "manual_review_groupings": sum(1 for row in recommendations if row.get("review_required") == "yes"),
        "recommendations_by_status": dict(Counter(str(row.get("recommendation_status", "")) for row in recommendations)),
        "recommendations_by_policy": dict(Counter(str(row.get("site_use_policy", "")) for row in recommendations)),
        "audit_failure_count": sum(1 for row in audit_rows if row.get("status") != "pass"),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
