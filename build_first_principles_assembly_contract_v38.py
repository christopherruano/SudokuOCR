#!/usr/bin/env python3
"""Build the canonical assembly contract for downstream consumers.

This packages the recommendation stack into explicit, low-ambiguity outputs:
site-ready physical structure, publishable physical values, site-ready logical
collections, review queues, future physical candidates, and a requirement trace.

It performs no OCR/API work and does not promote candidates.
"""

from __future__ import annotations

import argparse
import csv
import json
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import xlsxwriter


ROOT = Path(__file__).resolve().parent
DEFAULT_RULEBOOK_DIR = ROOT / "export_final/pdf_recovery_incoming_2026-06-16_first_principles_assembly_rulebook_v34"
DEFAULT_SEQUENCE_DIR = ROOT / "export_final/pdf_recovery_incoming_2026-06-16_first_principles_source_sequence_map_v35"
DEFAULT_QUALITY_DIR = ROOT / "export_final/pdf_recovery_incoming_2026-06-16_first_principles_assembly_quality_audit_v36"
DEFAULT_RECOMMENDATIONS_DIR = ROOT / "export_final/pdf_recovery_incoming_2026-06-16_first_principles_assembly_recommendations_v37"
DEFAULT_REFRESH_SCRIPT = ROOT / "run_first_principles_assembly_rulebook_refresh.sh"
DEFAULT_OUT_DIR = ROOT / "export_final/pdf_recovery_incoming_2026-06-16_first_principles_assembly_contract_v38"


REC_FIELDS = [
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

TRACE_FIELDS = [
    "requirement_id",
    "requirement",
    "status",
    "evidence",
    "details",
]

AUDIT_FIELDS = ["check", "status", "detail"]
SUMMARY_FIELDS = ["metric", "name", "count"]


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


def int_or_none(value: object) -> int | None:
    if value in {None, ""}:
        return None
    try:
        return int(str(value))
    except ValueError:
        return None


def parse_pages(value: object) -> list[int]:
    pages: list[int] = []
    for part in str(value or "").replace(",", ";").split(";"):
        page = int_or_none(part.strip())
        if page is not None:
            pages.append(page)
    return pages


def pages_consecutive(pages: list[int]) -> bool:
    if not pages:
        return False
    pages = sorted(set(pages))
    return pages == list(range(pages[0], pages[-1] + 1))


def semi(values: Iterable[object]) -> str:
    return "; ".join(str(value) for value in values if value not in {None, ""})


def split_semicolon(value: object) -> list[str]:
    return [part.strip() for part in str(value or "").split(";") if part.strip()]


def canonical_slices(recs: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    site_ready = [row for row in recs if str(row.get("site_use_policy", "")).startswith("use_now")]
    physical_structure = [
        row for row in site_ready
        if row.get("record_type") == "physical_component" and row.get("physical_merge_allowed_now") == "yes"
    ]
    physical_values = [
        row for row in physical_structure
        if row.get("safe_to_publish_values") == "yes" and row.get("review_required") != "yes"
    ]
    logical_collections = [
        row for row in site_ready
        if row.get("record_type") == "logical_family_collection" and row.get("logical_collection_only") == "yes"
    ]
    physical_candidates = [row for row in recs if row.get("record_type") == "physical_candidate_edge"]
    review_required = [
        row for row in recs
        if row.get("review_required") == "yes" or not str(row.get("site_use_policy", "")).startswith("use_now")
    ]
    overbroad = [row for row in recs if row.get("recommendation_status") == "manual_split_required_logical_family"]
    missing_religion = [row for row in recs if row.get("record_type") == "logical_family_candidate_added_by_qa"]
    return {
        "site_ready": site_ready,
        "site_ready_physical_structure": physical_structure,
        "site_ready_physical_values": physical_values,
        "site_ready_logical_collections": logical_collections,
        "future_physical_candidates": physical_candidates,
        "review_required": review_required,
        "overbroad_logical_families": overbroad,
        "missing_religion_family_candidates": missing_religion,
    }


def build_audits(
    recs: list[dict[str, str]],
    edge_rows: list[dict[str, str]],
    runs: list[dict[str, str]],
    row_label_audit: list[dict[str, str]],
    religion_audit: list[dict[str, str]],
    slices: dict[str, list[dict[str, str]]],
    refresh_script: Path,
    workbook_path: Path,
) -> list[dict[str, object]]:
    audits: list[dict[str, object]] = []
    physical = slices["site_ready_physical_structure"]
    bad_physical = [
        row for row in physical
        if row.get("physical_merge_allowed_now") != "yes" or not pages_consecutive(parse_pages(row.get("component_pages")))
    ]
    audits.append({"check": "site_ready_physical_components_are_consecutive", "status": "pass" if not bad_physical else "fail", "detail": f"bad={len(bad_physical)} total={len(physical)}"})

    unsafe_physical = [
        row for row in recs
        if row.get("physical_merge_allowed_now") == "yes" and row.get("record_type") != "physical_component"
    ]
    audits.append({"check": "only_physical_components_have_physical_merge", "status": "pass" if not unsafe_physical else "fail", "detail": f"bad={len(unsafe_physical)}"})

    bad_candidates = [
        row for row in slices["future_physical_candidates"]
        if row.get("physical_merge_allowed_now") == "yes" or str(row.get("site_use_policy", "")).startswith("use_now")
    ]
    audits.append({"check": "future_physical_candidates_are_not_site_ready", "status": "pass" if not bad_candidates else "fail", "detail": f"bad={len(bad_candidates)} total={len(slices['future_physical_candidates'])}"})

    bad_logical = [
        row for row in slices["site_ready_logical_collections"]
        if row.get("logical_collection_only") != "yes" or row.get("physical_merge_allowed_now") == "yes"
    ]
    audits.append({"check": "site_ready_logical_collections_are_logical_only", "status": "pass" if not bad_logical else "fail", "detail": f"bad={len(bad_logical)} total={len(slices['site_ready_logical_collections'])}"})

    overbroad_ready = [
        row for row in slices["overbroad_logical_families"]
        if str(row.get("site_use_policy", "")).startswith("use_now")
    ]
    audits.append({"check": "overbroad_logical_families_are_not_site_ready", "status": "pass" if not overbroad_ready else "fail", "detail": f"bad={len(overbroad_ready)} total={len(slices['overbroad_logical_families'])}"})

    missed_row_labels = [row for row in row_label_audit if row.get("potential_missed_signal") == "yes"]
    audits.append({"check": "row_label_audit_has_no_unexplained_missed_signals", "status": "pass" if not missed_row_labels else "fail", "detail": f"potential_missed={len(missed_row_labels)}"})

    expected_missing = [row for row in religion_audit if row.get("coverage_bucket") == "possible_missing_religion_family"]
    audits.append({"check": "missing_religion_candidates_carried_to_contract", "status": "pass" if len(expected_missing) == len(slices["missing_religion_family_candidates"]) else "fail", "detail": f"qa={len(expected_missing)} contract={len(slices['missing_religion_family_candidates'])}"})

    edge_run_count_match = len(edge_rows) == sum(int_or_none(row.get("n_edges")) or 0 for row in runs)
    audits.append({"check": "run_edge_counts_cover_rulebook_edges", "status": "pass" if edge_run_count_match else "fail", "detail": f"edges={len(edge_rows)} run_edge_sum={sum(int_or_none(row.get('n_edges')) or 0 for row in runs)}"})

    script_text = refresh_script.read_text(encoding="utf-8") if refresh_script.exists() else ""
    required_scripts = [
        "build_assembly_ocr_intake_report.py",
        "build_first_principles_assembly_frontier_v30.py",
        "build_first_principles_assembly_components_v31.py",
        "build_first_principles_logical_family_frontier_v32.py",
        "build_first_principles_assembly_control_v33.py",
        "build_first_principles_assembly_rulebook_v34.py",
        "build_first_principles_source_sequence_map_v35.py",
        "build_first_principles_assembly_quality_audit_v36.py",
        "build_first_principles_assembly_recommendations_v37.py",
        "build_first_principles_assembly_contract_v38.py",
        "build_first_principles_assembly_evidence_v39.py",
        "build_first_principles_logical_collection_evidence_v40.py",
        "build_first_principles_static_grouping_manifest_v41.py",
        "build_first_principles_candidate_promotion_v42.py",
        "build_first_principles_assembly_frontier_v43.py",
        "build_first_principles_manual_adjudication_v44.py",
        "build_first_principles_logical_sibling_sets_v45.py",
        "build_first_principles_materialized_physical_assembly_v46.py",
        "build_first_principles_assembly_release_snapshot_v47.py",
    ]
    missing_scripts = [name for name in required_scripts if name not in script_text]
    audits.append({"check": "refresh_driver_rebuilds_full_no_ocr_stack", "status": "pass" if not missing_scripts else "fail", "detail": f"missing={semi(missing_scripts) or 'none'}"})

    try:
        with zipfile.ZipFile(workbook_path) as zf:
            bad_member = zf.testzip()
    except Exception as exc:  # pragma: no cover
        bad_member = f"{type(exc).__name__}: {exc}"
    audits.append({"check": "workbook_zip_integrity", "status": "pass" if bad_member is None else "fail", "detail": str(bad_member or "ok")})
    return audits


def build_trace(audits: list[dict[str, object]], slices: dict[str, list[dict[str, str]]], row_label_audit: list[dict[str, str]], religion_audit: list[dict[str, str]]) -> list[dict[str, object]]:
    audit_status = {row["check"]: row["status"] for row in audits}
    return [
        {
            "requirement_id": "R1",
            "requirement": "Physical assemblies must be consecutive and only confirmed adjacent edges may physically merge.",
            "status": "pass" if audit_status.get("site_ready_physical_components_are_consecutive") == "pass" and audit_status.get("only_physical_components_have_physical_merge") == "pass" else "fail",
            "evidence": "canonical_site_ready_physical_components.csv; acceptance_audit.csv",
            "details": f"site_ready_physical_components={len(slices['site_ready_physical_structure'])}",
        },
        {
            "requirement_id": "R2",
            "requirement": "Likely assemblies across missing pages may be marked but not physically merged.",
            "status": "pass" if audit_status.get("future_physical_candidates_are_not_site_ready") == "pass" else "fail",
            "evidence": "canonical_future_physical_candidates.csv; future_physical_candidate_runs.csv",
            "details": f"future_physical_candidates={len(slices['future_physical_candidates'])}",
        },
        {
            "requirement_id": "R3",
            "requirement": "Odd/no-row-label continuation signals must be captured or explicitly explained.",
            "status": "pass" if audit_status.get("row_label_audit_has_no_unexplained_missed_signals") == "pass" else "fail",
            "evidence": "row_label_coverage_audit.csv",
            "details": f"row_label_audit_rows={len(row_label_audit)} potential_missed={sum(1 for row in row_label_audit if row.get('potential_missed_signal') == 'yes')}",
        },
        {
            "requirement_id": "R4",
            "requirement": "Logical religion/table families can be grouped only as logical-only collections, never physical merges.",
            "status": "pass" if audit_status.get("site_ready_logical_collections_are_logical_only") == "pass" else "fail",
            "evidence": "canonical_site_ready_logical_collections.csv; canonical_missing_religion_family_candidates.csv",
            "details": f"site_ready_logical={len(slices['site_ready_logical_collections'])} missing_religion_candidates={len(slices['missing_religion_family_candidates'])}",
        },
        {
            "requirement_id": "R5",
            "requirement": "Overbroad logical groups must be demoted for review/splitting rather than used as single site-ready groups.",
            "status": "pass" if audit_status.get("overbroad_logical_families_are_not_site_ready") == "pass" else "fail",
            "evidence": "canonical_overbroad_logical_families.csv",
            "details": f"overbroad_logical_families={len(slices['overbroad_logical_families'])}",
        },
        {
            "requirement_id": "R6",
            "requirement": "The process must be refreshable as new OCR artifacts arrive without OCR/API calls in this layer.",
            "status": "pass" if audit_status.get("refresh_driver_rebuilds_full_no_ocr_stack") == "pass" else "fail",
            "evidence": "run_first_principles_assembly_rulebook_refresh.sh",
            "details": "refresh driver includes intake plus v30-v47 builders",
        },
    ]


def build_summary(slices: dict[str, list[dict[str, str]]], recs: list[dict[str, str]], audits: list[dict[str, object]], paths: dict[str, Path]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = [{"metric": "paths", "name": name, "count": output_path(path)} for name, path in paths.items()]
    rows.extend(
        [
            {"metric": "counts", "name": "recommendations_total", "count": len(recs)},
            {"metric": "counts", "name": "canonical_site_ready_physical_components", "count": len(slices["site_ready_physical_structure"])},
            {"metric": "counts", "name": "canonical_site_ready_physical_values", "count": len(slices["site_ready_physical_values"])},
            {"metric": "counts", "name": "canonical_site_ready_logical_collections", "count": len(slices["site_ready_logical_collections"])},
            {"metric": "counts", "name": "canonical_future_physical_candidates", "count": len(slices["future_physical_candidates"])},
            {"metric": "counts", "name": "canonical_review_required", "count": len(slices["review_required"])},
            {"metric": "counts", "name": "canonical_overbroad_logical_families", "count": len(slices["overbroad_logical_families"])},
            {"metric": "counts", "name": "canonical_missing_religion_family_candidates", "count": len(slices["missing_religion_family_candidates"])},
        ]
    )
    for metric, counter in [
        ("recommendations_by_status", Counter(str(row.get("recommendation_status", "")) for row in recs)),
        ("recommendations_by_policy", Counter(str(row.get("site_use_policy", "")) for row in recs)),
        ("audits_by_status", Counter(str(row.get("status", "")) for row in audits)),
    ]:
        for name, count in sorted(counter.items()):
            rows.append({"metric": metric, "name": name, "count": count})
    return rows


def write_workbook(
    path: Path,
    summary_rows: list[dict[str, object]],
    audits: list[dict[str, object]],
    trace: list[dict[str, object]],
    slices: dict[str, list[dict[str, str]]],
) -> None:
    workbook = xlsxwriter.Workbook(path, {"constant_memory": True})
    header_fmt = workbook.add_format({"bold": True, "bg_color": "#E8EEF7", "border": 1})
    wrap_fmt = workbook.add_format({"text_wrap": True, "valign": "top"})
    sheets = [
        ("Summary", summary_rows, SUMMARY_FIELDS),
        ("Acceptance Audit", audits, AUDIT_FIELDS),
        ("Requirement Trace", trace, TRACE_FIELDS),
        ("Physical Structure", slices["site_ready_physical_structure"], REC_FIELDS),
        ("Physical Values", slices["site_ready_physical_values"], REC_FIELDS),
        ("Logical Collections", slices["site_ready_logical_collections"], REC_FIELDS),
        ("Future Physical", slices["future_physical_candidates"], REC_FIELDS),
        ("Review Required", slices["review_required"], REC_FIELDS),
        ("Overbroad Logical", slices["overbroad_logical_families"], REC_FIELDS),
        ("Missing Religion", slices["missing_religion_family_candidates"], REC_FIELDS),
    ]
    for name, rows, fields in sheets:
        ws = workbook.add_worksheet(name[:31])
        for col, field in enumerate(fields):
            ws.write(0, col, field, header_fmt)
        for row_idx, row in enumerate(rows, start=1):
            for col, field in enumerate(fields):
                value = row.get(field, "")
                fmt = wrap_fmt if field in {"requirement", "evidence", "details", "primary_title", "reason", "recommended_action", "qa_flags"} else None
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
    parser.add_argument("--recommendations-dir", default=str(DEFAULT_RECOMMENDATIONS_DIR))
    parser.add_argument("--refresh-script", default=str(DEFAULT_REFRESH_SCRIPT))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    args = parser.parse_args()

    rulebook_dir = Path(args.rulebook_dir).expanduser()
    sequence_dir = Path(args.sequence_dir).expanduser()
    quality_dir = Path(args.quality_dir).expanduser()
    recommendations_dir = Path(args.recommendations_dir).expanduser()
    refresh_script = Path(args.refresh_script).expanduser()
    out_dir = Path(args.out_dir).expanduser()
    paths = {
        "rulebook_dir": rulebook_dir,
        "sequence_dir": sequence_dir,
        "quality_dir": quality_dir,
        "recommendations_dir": recommendations_dir,
        "refresh_script": refresh_script,
        "out_dir": out_dir,
    }

    recs = read_csv(recommendations_dir / "assembly_recommendations.csv")
    edge_rows = read_csv(rulebook_dir / "edge_rulebook.csv")
    runs = read_csv(sequence_dir / "assembly_runs.csv")
    row_label_audit = read_csv(quality_dir / "row_label_coverage_audit.csv")
    religion_audit = read_csv(quality_dir / "religion_sequence_coverage_audit.csv")
    slices = canonical_slices(recs)

    out_dir.mkdir(parents=True, exist_ok=True)
    workbook_path = out_dir / "assembly_contract_workbook.xlsx"
    write_workbook(workbook_path, [], [], [], slices)
    audits = build_audits(recs, edge_rows, runs, row_label_audit, religion_audit, slices, refresh_script, workbook_path)
    trace = build_trace(audits, slices, row_label_audit, religion_audit)
    summary_rows = build_summary(slices, recs, audits, paths)
    write_workbook(workbook_path, summary_rows, audits, trace, slices)
    audits = build_audits(recs, edge_rows, runs, row_label_audit, religion_audit, slices, refresh_script, workbook_path)
    summary_rows = build_summary(slices, recs, audits, paths)
    trace = build_trace(audits, slices, row_label_audit, religion_audit)
    write_workbook(workbook_path, summary_rows, audits, trace, slices)

    write_csv(out_dir / "canonical_site_ready_physical_components.csv", slices["site_ready_physical_structure"], REC_FIELDS)
    write_csv(out_dir / "canonical_site_ready_physical_values.csv", slices["site_ready_physical_values"], REC_FIELDS)
    write_csv(out_dir / "canonical_site_ready_logical_collections.csv", slices["site_ready_logical_collections"], REC_FIELDS)
    write_csv(out_dir / "canonical_future_physical_candidates.csv", slices["future_physical_candidates"], REC_FIELDS)
    write_csv(out_dir / "canonical_review_required.csv", slices["review_required"], REC_FIELDS)
    write_csv(out_dir / "canonical_overbroad_logical_families.csv", slices["overbroad_logical_families"], REC_FIELDS)
    write_csv(out_dir / "canonical_missing_religion_family_candidates.csv", slices["missing_religion_family_candidates"], REC_FIELDS)
    write_csv(out_dir / "acceptance_audit.csv", audits, AUDIT_FIELDS)
    write_csv(out_dir / "requirement_trace.csv", trace, TRACE_FIELDS)
    write_csv(out_dir / "summary.csv", summary_rows, SUMMARY_FIELDS)

    contract = {
        "schema_version": "assembly_contract_v38",
        "created_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "inputs": {name: output_path(path) for name, path in paths.items()},
        "outputs": {
            "canonical_site_ready_physical_components": output_path(out_dir / "canonical_site_ready_physical_components.csv"),
            "canonical_site_ready_physical_values": output_path(out_dir / "canonical_site_ready_physical_values.csv"),
            "canonical_site_ready_logical_collections": output_path(out_dir / "canonical_site_ready_logical_collections.csv"),
            "canonical_future_physical_candidates": output_path(out_dir / "canonical_future_physical_candidates.csv"),
            "canonical_review_required": output_path(out_dir / "canonical_review_required.csv"),
            "canonical_overbroad_logical_families": output_path(out_dir / "canonical_overbroad_logical_families.csv"),
            "canonical_missing_religion_family_candidates": output_path(out_dir / "canonical_missing_religion_family_candidates.csv"),
            "acceptance_audit": output_path(out_dir / "acceptance_audit.csv"),
            "requirement_trace": output_path(out_dir / "requirement_trace.csv"),
            "workbook": output_path(workbook_path),
        },
        "counts": {
            "recommendations_total": len(recs),
            "site_ready_physical_components": len(slices["site_ready_physical_structure"]),
            "site_ready_physical_values": len(slices["site_ready_physical_values"]),
            "site_ready_logical_collections": len(slices["site_ready_logical_collections"]),
            "future_physical_candidates": len(slices["future_physical_candidates"]),
            "review_required": len(slices["review_required"]),
            "overbroad_logical_families": len(slices["overbroad_logical_families"]),
            "missing_religion_family_candidates": len(slices["missing_religion_family_candidates"]),
            "audit_failures": sum(1 for row in audits if row.get("status") != "pass"),
        },
        "policies": {
            "physical_merge": "Only records in canonical_site_ready_physical_components.csv may be physically merged now.",
            "physical_values": "Only records in canonical_site_ready_physical_values.csv may publish values without value review.",
            "logical_groups": "Logical collections are never physical merges.",
            "future_candidates": "Future physical candidates require edge-level promotion after OCR/missing-page/constraint resolution.",
        },
    }
    (out_dir / "assembly_contract.json").write_text(json.dumps(contract, indent=2, sort_keys=True), encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(contract["counts"], indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(contract["counts"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
