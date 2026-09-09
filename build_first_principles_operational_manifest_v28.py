#!/usr/bin/env python3
"""Create the operational assembly manifest from v27 grouping records.

v27 intentionally preserves many possible logical collections. This v28 layer is
stricter about what should be used downstream without manual review:

* physical joins remain only the v25/v26 required physical groups;
* logical groups are usable only for bounded sibling/part or self-contained
  continuation sequences that are consecutive and not too large;
* generic same-table continuation runs are kept as review-only candidates;
* gap-crossing groups stay missing-bridge candidates.

No OCR/API work is performed here.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import zipfile
from collections import Counter
from pathlib import Path
from typing import Iterable

import xlsxwriter


ROOT = Path(__file__).resolve().parent
V27_DIR = ROOT / "export_final/pdf_recovery_incoming_2026-06-16_first_principles_assembly_manifest_v27"
OUT_DIR = ROOT / "export_final/pdf_recovery_incoming_2026-06-16_first_principles_operational_manifest_v28"


MAX_USABLE_LOGICAL_PAGES = 12
MAX_USABLE_LOGICAL_RUNS = 12


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


def stable_id(prefix: str, *parts: object) -> str:
    payload = "||".join(str(part) for part in parts)
    return f"{prefix}_{hashlib.sha1(payload.encode('utf-8')).hexdigest()[:16]}"


def compact(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def clean_text(value: object) -> str:
    s = str(value or "").lower()
    s = s.replace("&", " and ")
    s = re.sub(r"[\u2010-\u2015]", "-", s)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def int_or_none(value: object) -> int | None:
    s = str(value or "").strip()
    return int(s) if re.fullmatch(r"-?\d+", s) else None


def uniq(values: Iterable[object], limit: int = 20) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        s = compact(value)
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    if len(out) > limit:
        return out[:limit] + [f"...(+{len(out) - limit})"]
    return out


def uniq_join(values: Iterable[object], limit: int = 20) -> str:
    return "; ".join(uniq(values, limit=limit))


def has_row_label_dependency(row: dict[str, str]) -> bool:
    blob = " ".join(
        [
            row.get("v27_group_type", ""),
            row.get("review_flags", ""),
            row.get("source_evidence", ""),
            row.get("primary_title", ""),
        ]
    )
    return any(
        marker in blob
        for marker in [
            "row_label",
            "right_half",
            "halfspread",
            "imputed_missing_right_total_0_5",
            "requires_previous_row_labels",
        ]
    )


def physical_operational_status(row: dict[str, str]) -> tuple[str, str, str, str]:
    group_type = row.get("v27_group_type", "")
    if group_type == "required_physical_row_assembled_pass":
        return "use_now", "physical_required_row_assembled", "physical_join", "row assembly and value checks pass"
    if group_type == "required_physical_row_assembled_value_review":
        return "needs_value_review", "physical_required_structurally_complete", "physical_join_after_review", "row structure aligns; value checks need review"
    if group_type == "required_physical_partial_row_assembly":
        return "partial_review", "physical_required_partial_rows", "physical_join_after_review", "some blocks align; others need review"
    if group_type == "required_physical_blocked_missing_ocr_artifact":
        return "blocked_missing_ocr_artifact", "physical_required_not_row_tested", "physical_join_candidate", "physical pair is consecutive, but no row artifact is available"
    if group_type in {"required_physical_failed_row_constraints", "required_physical_no_safe_block_alignment"}:
        return "manual_review", "physical_required_failed_row_checks", "physical_join_after_review", "row alignment constraints failed or could not align blocks"
    return "manual_review", "physical_required_review", "physical_join_after_review", "unmapped physical status"


def logical_operational_status(row: dict[str, str]) -> tuple[str, str, str, str]:
    group_type = row.get("v27_group_type", "")
    n_pages = int_or_none(row.get("n_physical_pages")) or int_or_none(row.get("range_page_count")) or 0
    n_runs = int_or_none(row.get("n_physical_runs")) or int_or_none(row.get("n_ranges_parsed")) or 0
    if row.get("all_ranges_consecutive") != "yes" or row.get("missing_pages"):
        return "blocked_missing_pages", "logical_gap_crossing_candidate", "logical_collection_after_gap_fill", "logical group crosses missing PDF pages"
    if n_pages > MAX_USABLE_LOGICAL_PAGES or n_runs > MAX_USABLE_LOGICAL_RUNS:
        return "review_large_collection", "logical_consecutive_large_review", "logical_collection_after_review", f"consecutive but large: {n_runs} runs / {n_pages} pages"
    if group_type == "logical_religion_or_part_sibling_consecutive":
        return "use_now", "logical_sibling_bounded", "logical_collection_only", "bounded religion/part sibling sequence, not a physical merge"
    if group_type == "logical_self_contained_sequence_consecutive":
        return "use_now", "logical_self_contained_sequence", "logical_collection_only", "bounded self-contained continuation sequence"
    if group_type == "logical_same_table_consecutive":
        # Keep these visible but not automatically usable: title sameness alone is
        # explicitly not enough for physical or logical assembly.
        return "review_only", "logical_same_table_candidate", "logical_collection_after_review", "same-title/family consecutive candidate; needs title/schema review"
    if group_type == "logical_collection_consecutive_large_review":
        return "review_large_collection", "logical_consecutive_large_review", "logical_collection_after_review", "large consecutive logical collection"
    return "review_only", "logical_unmapped_review", "logical_collection_after_review", "unmapped logical candidate"


def build_operational_required(required_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in required_rows:
        tier, subgroup, policy, reason = physical_operational_status(row)
        rows.append(
            {
                "v28_group_id": stable_id("v28req", row.get("v27_group_id", "")),
                "v28_tier": tier,
                "v28_subgroup": subgroup,
                "assembly_policy": policy,
                "assembly_layer": "required_physical",
                "reason": reason,
                "row_label_dependency": "yes" if has_row_label_dependency(row) else "unknown_or_not_applicable",
                "v27_group_id": row.get("v27_group_id", ""),
                "v25_component_id": row.get("v25_component_id", ""),
                "v26_status": row.get("v26_status", ""),
                "v5_source_key": row.get("v5_source_key", ""),
                "source_pdf": row.get("source_pdf", ""),
                "source_pdf_id": row.get("source_pdf_id", ""),
                "page_ranges": row.get("page_ranges", ""),
                "n_physical_pages": row.get("n_physical_pages", ""),
                "physical_merge_allowed": row.get("physical_merge_allowed", ""),
                "logical_collection_only": row.get("logical_collection_only", ""),
                "missing_pages": row.get("missing_pages", ""),
                "recommended_action": row.get("recommended_action", ""),
                "review_flags": row.get("review_flags", ""),
                "primary_title": row.get("primary_title", ""),
                "source_evidence": row.get("source_evidence", ""),
                "image_paths": row.get("image_paths", ""),
                "ocr_artifacts": row.get("ocr_artifacts", ""),
            }
        )
    rows.sort(key=lambda r: (str(r.get("v28_tier")), str(r.get("v5_source_key") or r.get("source_pdf")), int_or_none(str(r.get("page_ranges")).split("-", 1)[0]) or 10**9))
    return rows


def build_operational_logical(logical_rows: list[dict[str, str]], blocked_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in [*logical_rows, *blocked_rows]:
        tier, subgroup, policy, reason = logical_operational_status(row)
        rows.append(
            {
                "v28_group_id": stable_id("v28log", row.get("v27_group_id", "")),
                "v28_tier": tier,
                "v28_subgroup": subgroup,
                "assembly_policy": policy,
                "assembly_layer": "logical_collection",
                "reason": reason,
                "row_label_dependency": "not_applicable_logical_collection",
                "v27_group_id": row.get("v27_group_id", ""),
                "collection_group_id": row.get("collection_group_id", ""),
                "v25_candidate_type": row.get("v25_candidate_type", ""),
                "v5_source_key": row.get("v5_source_key", ""),
                "source_pdf": row.get("source_pdf", ""),
                "source_pdf_id": row.get("source_pdf_id", ""),
                "page_ranges": row.get("page_ranges", ""),
                "n_physical_runs": row.get("n_physical_runs", ""),
                "n_physical_pages": row.get("n_physical_pages", ""),
                "all_ranges_consecutive": row.get("all_ranges_consecutive", ""),
                "range_gaps": row.get("range_gaps", ""),
                "large_collection_review": row.get("large_collection_review", ""),
                "physical_merge_allowed": row.get("physical_merge_allowed", ""),
                "logical_collection_only": row.get("logical_collection_only", ""),
                "missing_pages": row.get("missing_pages", ""),
                "recommended_action": row.get("recommended_action", ""),
                "review_flags": row.get("review_flags", ""),
                "primary_title": row.get("primary_title", ""),
                "source_evidence": row.get("source_evidence", ""),
                "collection_reasons": row.get("collection_reasons", ""),
                "image_paths": row.get("image_paths", ""),
            }
        )
    rows.sort(key=lambda r: (str(r.get("v28_tier")), str(r.get("v28_subgroup")), str(r.get("v5_source_key") or r.get("source_pdf")), int_or_none(str(r.get("page_ranges")).split(";", 1)[0].split("-", 1)[0]) or 10**9))
    return rows


def build_operational_manifest(required: list[dict[str, object]], logical: list[dict[str, object]], missing_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in required:
        rows.append(
            {
                "v28_group_id": row.get("v28_group_id", ""),
                "v28_tier": row.get("v28_tier", ""),
                "v28_subgroup": row.get("v28_subgroup", ""),
                "assembly_layer": row.get("assembly_layer", ""),
                "assembly_policy": row.get("assembly_policy", ""),
                "reason": row.get("reason", ""),
                "v5_source_key": row.get("v5_source_key", ""),
                "source_pdf": row.get("source_pdf", ""),
                "source_pdf_id": row.get("source_pdf_id", ""),
                "page_ranges": row.get("page_ranges", ""),
                "n_physical_pages": row.get("n_physical_pages", ""),
                "physical_merge_allowed": row.get("physical_merge_allowed", ""),
                "logical_collection_only": row.get("logical_collection_only", ""),
                "missing_pages": row.get("missing_pages", ""),
                "review_flags": row.get("review_flags", ""),
                "primary_title": row.get("primary_title", ""),
                "source_evidence": row.get("source_evidence", ""),
            }
        )
    for row in logical:
        if row.get("v28_tier") == "blocked_missing_pages":
            # Gap-crossing logical candidates are still preserved in
            # v28_logical_operational.csv, but the actionable manifest should
            # point to the de-duplicated missing-bridge rows instead.
            continue
        rows.append(
            {
                "v28_group_id": row.get("v28_group_id", ""),
                "v28_tier": row.get("v28_tier", ""),
                "v28_subgroup": row.get("v28_subgroup", ""),
                "assembly_layer": row.get("assembly_layer", ""),
                "assembly_policy": row.get("assembly_policy", ""),
                "reason": row.get("reason", ""),
                "v5_source_key": row.get("v5_source_key", ""),
                "source_pdf": row.get("source_pdf", ""),
                "source_pdf_id": row.get("source_pdf_id", ""),
                "page_ranges": row.get("page_ranges", ""),
                "n_physical_pages": row.get("n_physical_pages", ""),
                "physical_merge_allowed": row.get("physical_merge_allowed", ""),
                "logical_collection_only": row.get("logical_collection_only", ""),
                "missing_pages": row.get("missing_pages", ""),
                "review_flags": row.get("review_flags", ""),
                "primary_title": row.get("primary_title", ""),
                "source_evidence": row.get("source_evidence", ""),
            }
        )
    for row in missing_rows:
        rows.append(
            {
                "v28_group_id": stable_id("v28miss", row.get("v27_group_id", "")),
                "v28_tier": "blocked_missing_pages",
                "v28_subgroup": row.get("v27_group_type", "missing_bridge"),
                "assembly_layer": "missing_bridge",
                "assembly_policy": "assemble_after_gap_fill_only",
                "reason": "requires missing PDF pages before grouping",
                "v5_source_key": row.get("v5_source_key", ""),
                "source_pdf": row.get("source_pdf", ""),
                "source_pdf_id": row.get("source_pdf_id", ""),
                "page_ranges": "",
                "n_physical_pages": "",
                "physical_merge_allowed": "no",
                "logical_collection_only": row.get("logical_collection_only", ""),
                "missing_pages": row.get("missing_pages", ""),
                "review_flags": row.get("review_flags", ""),
                "primary_title": row.get("primary_title", ""),
                "source_evidence": row.get("source_evidence", ""),
            }
        )
    rows.sort(key=lambda r: (str(r.get("assembly_layer")), str(r.get("v28_tier")), str(r.get("v5_source_key") or r.get("source_pdf")), int_or_none(str(r.get("page_ranges")).split(";", 1)[0].split("-", 1)[0]) or 10**9))
    return rows


def validate(required: list[dict[str, object]], logical: list[dict[str, object]], manifest: list[dict[str, object]]) -> list[dict[str, object]]:
    failures: list[tuple[str, list[str]]] = []
    physical_bad = [
        str(row.get("v28_group_id"))
        for row in required
        if row.get("assembly_policy") in {"physical_join", "physical_join_after_review", "physical_join_candidate"}
        and row.get("physical_merge_allowed") != "yes"
    ]
    failures.append(("physical_policy_only_on_required_physical_groups", physical_bad))
    logical_physical_bad = [
        str(row.get("v28_group_id"))
        for row in logical
        if row.get("physical_merge_allowed") != "no"
    ]
    failures.append(("logical_groups_never_allow_physical_merge", logical_physical_bad))
    usable_logical_bad = [
        str(row.get("v28_group_id"))
        for row in logical
        if row.get("v28_tier") == "use_now"
        and (
            row.get("all_ranges_consecutive") != "yes"
            or row.get("missing_pages")
            or (int_or_none(row.get("n_physical_pages")) or 0) > MAX_USABLE_LOGICAL_PAGES
            or row.get("v28_subgroup") not in {"logical_sibling_bounded", "logical_self_contained_sequence"}
        )
    ]
    failures.append(("use_now_logical_groups_are_bounded_consecutive_siblings_or_sequences", usable_logical_bad))
    generic_same_table_usable = [
        str(row.get("v28_group_id"))
        for row in logical
        if row.get("v28_subgroup") == "logical_same_table_candidate" and row.get("v28_tier") == "use_now"
    ]
    failures.append(("generic_same_table_candidates_are_not_use_now", generic_same_table_usable))
    missing_use_now = [
        str(row.get("v28_group_id"))
        for row in manifest
        if row.get("v28_tier") == "use_now" and row.get("missing_pages")
    ]
    failures.append(("use_now_manifest_rows_have_no_missing_pages", missing_use_now))
    return [
        {
            "check": name,
            "status": "pass" if not bad else "fail",
            "failure_count": len(bad),
            "details": json.dumps(bad[:50], sort_keys=True),
        }
        for name, bad in failures
    ]


def write_workbook(path: Path, sheets: dict[str, tuple[list[dict[str, object]], list[str]]]) -> None:
    workbook = xlsxwriter.Workbook(str(path))
    header = workbook.add_format({"bold": True, "bg_color": "#D9EAF7", "border": 1})
    wrap = workbook.add_format({"text_wrap": True, "valign": "top"})
    for sheet_name, (rows, fields) in sheets.items():
        ws = workbook.add_worksheet(sheet_name[:31])
        for col, field in enumerate(fields):
            ws.write(0, col, field, header)
        for r_idx, row in enumerate(rows, start=1):
            for c_idx, field in enumerate(fields):
                fmt = wrap if any(token in field for token in ["reason", "flags", "title", "image", "artifact", "evidence"]) else None
                ws.write(r_idx, c_idx, row.get(field, ""), fmt)
        ws.freeze_panes(1, 0)
        ws.autofilter(0, 0, max(1, len(rows)), max(0, len(fields) - 1))
        for col, field in enumerate(fields):
            width = 16
            if any(token in field for token in ["reason", "flags", "title", "image", "artifact", "evidence"]):
                width = 45
            elif "id" in field:
                width = 24
            ws.set_column(col, col, width)
    workbook.close()


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    required_v27 = read_csv(V27_DIR / "v27_required_physical_groups.csv")
    safe_logical_v27 = read_csv(V27_DIR / "v27_safe_logical_groups.csv")
    blocked_logical_v27 = read_csv(V27_DIR / "v27_blocked_logical_groups.csv")
    missing_v27 = read_csv(V27_DIR / "v27_missing_bridge_groups.csv")

    required = build_operational_required(required_v27)
    logical = build_operational_logical(safe_logical_v27, blocked_logical_v27)
    manifest = build_operational_manifest(required, logical, missing_v27)
    audit = validate(required, logical, manifest)

    required_fields = [
        "v28_group_id",
        "v28_tier",
        "v28_subgroup",
        "assembly_policy",
        "assembly_layer",
        "reason",
        "row_label_dependency",
        "v27_group_id",
        "v25_component_id",
        "v26_status",
        "v5_source_key",
        "source_pdf",
        "source_pdf_id",
        "page_ranges",
        "n_physical_pages",
        "physical_merge_allowed",
        "logical_collection_only",
        "missing_pages",
        "recommended_action",
        "review_flags",
        "primary_title",
        "source_evidence",
        "image_paths",
        "ocr_artifacts",
    ]
    logical_fields = [
        "v28_group_id",
        "v28_tier",
        "v28_subgroup",
        "assembly_policy",
        "assembly_layer",
        "reason",
        "row_label_dependency",
        "v27_group_id",
        "collection_group_id",
        "v25_candidate_type",
        "v5_source_key",
        "source_pdf",
        "source_pdf_id",
        "page_ranges",
        "n_physical_runs",
        "n_physical_pages",
        "all_ranges_consecutive",
        "range_gaps",
        "large_collection_review",
        "physical_merge_allowed",
        "logical_collection_only",
        "missing_pages",
        "recommended_action",
        "review_flags",
        "primary_title",
        "source_evidence",
        "collection_reasons",
        "image_paths",
    ]
    manifest_fields = [
        "v28_group_id",
        "v28_tier",
        "v28_subgroup",
        "assembly_layer",
        "assembly_policy",
        "reason",
        "v5_source_key",
        "source_pdf",
        "source_pdf_id",
        "page_ranges",
        "n_physical_pages",
        "physical_merge_allowed",
        "logical_collection_only",
        "missing_pages",
        "review_flags",
        "primary_title",
        "source_evidence",
    ]
    audit_fields = ["check", "status", "failure_count", "details"]

    use_now = [row for row in manifest if row.get("v28_tier") == "use_now"]
    review = [row for row in manifest if row.get("v28_tier") in {"needs_value_review", "partial_review", "manual_review", "review_only", "review_large_collection"}]
    blocked = [row for row in manifest if str(row.get("v28_tier", "")).startswith("blocked")]

    write_csv(OUT_DIR / "v28_required_physical_operational.csv", required, required_fields)
    write_csv(OUT_DIR / "v28_logical_operational.csv", logical, logical_fields)
    write_csv(OUT_DIR / "v28_operational_manifest.csv", manifest, manifest_fields)
    write_csv(OUT_DIR / "v28_use_now_groups.csv", use_now, manifest_fields)
    write_csv(OUT_DIR / "v28_review_groups.csv", review, manifest_fields)
    write_csv(OUT_DIR / "v28_blocked_groups.csv", blocked, manifest_fields)
    write_csv(OUT_DIR / "v28_audit.csv", audit, audit_fields)

    summary_rows = [
        {"metric": "required_physical_operational_rows", "value": len(required)},
        {"metric": "logical_operational_rows", "value": len(logical)},
        {"metric": "operational_manifest_rows", "value": len(manifest)},
        {"metric": "use_now_groups", "value": len(use_now)},
        {"metric": "review_groups", "value": len(review)},
        {"metric": "blocked_groups", "value": len(blocked)},
        {"metric": "audit_failure_count", "value": sum(int(row["failure_count"]) for row in audit)},
    ]
    for key, value in sorted(Counter(row["v28_tier"] for row in manifest).items()):
        summary_rows.append({"metric": f"tier::{key}", "value": value})
    for key, value in sorted(Counter(row["v28_subgroup"] for row in manifest).items()):
        summary_rows.append({"metric": f"subgroup::{key}", "value": value})
    write_csv(OUT_DIR / "v28_summary.csv", summary_rows, ["metric", "value"])

    workbook_path = OUT_DIR / "v28_operational_assembly_manifest.xlsx"
    write_workbook(
        workbook_path,
        {
            "summary": (summary_rows, ["metric", "value"]),
            "use_now": (use_now, manifest_fields),
            "review": (review, manifest_fields),
            "blocked": (blocked, manifest_fields),
            "required_physical": (required, required_fields),
            "logical_operational": (logical, logical_fields),
            "audit": (audit, audit_fields),
        },
    )

    bad_member = None
    try:
        with zipfile.ZipFile(workbook_path) as zf:
            bad_member = zf.testzip()
    except Exception as exc:
        bad_member = f"{type(exc).__name__}: {exc}"

    summary = {
        "v28_required_physical_operational_rows": len(required),
        "v28_logical_operational_rows": len(logical),
        "v28_operational_manifest_rows": len(manifest),
        "v28_use_now_groups": len(use_now),
        "v28_review_groups": len(review),
        "v28_blocked_groups": len(blocked),
        "v28_tier_counts": dict(sorted(Counter(row["v28_tier"] for row in manifest).items())),
        "v28_subgroup_counts": dict(sorted(Counter(row["v28_subgroup"] for row in manifest).items())),
        "v28_audit_failure_count": sum(int(row["failure_count"]) for row in audit),
        "v28_workbook_zip_bad_member": bad_member,
    }
    (OUT_DIR / "v28_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
