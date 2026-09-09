#!/usr/bin/env python3
"""Build a conservative static-site update manifest from v4 adjacency crosswalks.

The manifest is intentionally non-destructive. It does not edit the explorer DB,
does not rebuild the site, and does not infer that page-level splits already have
clean fact rows. It records the safest next action for each current/v4 age-table
candidate so the later static-site update can be reviewed and replayed.
"""

from __future__ import annotations

import csv
import json
import os
from collections import Counter, defaultdict

try:
    import xlsxwriter
except ImportError:  # pragma: no cover
    xlsxwriter = None


ROOT = os.path.dirname(os.path.abspath(__file__))
V4_DIR = os.path.join(
    ROOT,
    "export_final",
    "pdf_recovery_incoming_2026-06-16_all_ocr_component_assembly_v4_adjacency",
)

QUEUE = os.path.join(V4_DIR, "main_age_civil_static_site_update_queue.csv")
SPLIT_PLAN = os.path.join(V4_DIR, "current_site_split_plan_exploded.csv")
GROUPS = os.path.join(V4_DIR, "static_site_groups.csv")
MEMBERS = os.path.join(V4_DIR, "static_site_group_members.csv")

OUT_MANIFEST_CSV = os.path.join(V4_DIR, "static_site_update_manifest.csv")
OUT_MANIFEST_JSON = os.path.join(V4_DIR, "static_site_update_manifest.json")
OUT_REVIEW_CSV = os.path.join(V4_DIR, "static_site_update_review_queue.csv")
OUT_BY_YEAR_REGION = os.path.join(V4_DIR, "static_site_update_summary_by_year_region.csv")
OUT_SUMMARY = os.path.join(V4_DIR, "static_site_update_manifest_summary.json")
OUT_XLSX = os.path.join(V4_DIR, "static_site_update_manifest_review.xlsx")

AUTO_ACTIONS = {"keep_current_table"}


def read_csv(path: str) -> list[dict[str, str]]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: str, rows: list[dict], fieldnames: list[str]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def unique_join(vals) -> str:
    out = []
    seen = set()
    for v in vals:
        s = str(v or "").strip()
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return " | ".join(out)


def intish(v, default=0) -> int:
    try:
        return int(float(str(v).strip()))
    except Exception:
        return default


def stable_part_id(site_table_id: str, idx: str, gid: str) -> str:
    base = site_table_id.strip() or gid
    safe = "".join(c if c.isalnum() else "_" for c in base)[-80:].strip("_")
    return f"{safe}__adj_part_{intish(idx):02d}_{gid[-8:]}"


def review_priority(manifest_action: str, priority_label: str) -> tuple[str, str]:
    label = priority_label.lower()
    if manifest_action == "keep_current_table":
        return "99", "auto_keep_not_for_manual_review"
    if manifest_action == "split_current_table_by_adjacency":
        return "01", "review_splits_first"
    if manifest_action == "promote_or_add_v4_adjacency_group":
        if "ready" in label:
            return "02", "promote_ready_v4_age_tables"
        if "light_review" in label:
            return "03", "promote_light_review_v4_age_tables"
        return "04", "promote_v4_age_tables_needing_quality_or_source_review"
    if manifest_action in {"review_no_v4_match", "review_partial_mapping"}:
        return "05", "current_site_age_tables_not_cleanly_mapped_to_v4"
    if manifest_action == "demote_or_remove_from_main_age_view":
        return "06", "demote_derivative_or_offtarget_from_main_age_view"
    return "98", "manual_review_other"


def finalize_record(rec: dict) -> dict:
    rank, label = review_priority(rec.get("manifest_action", ""), rec.get("adjacency_priority_label", ""))
    rec["review_priority_rank"] = rank
    rec["review_priority_label"] = label
    return rec


def load_inputs():
    queue = read_csv(QUEUE)
    split_plan = read_csv(SPLIT_PLAN)
    groups = {r["static_group_id"]: r for r in read_csv(GROUPS)}
    members_by_group: dict[str, list[dict]] = defaultdict(list)
    for r in read_csv(MEMBERS):
        members_by_group[r["static_group_id"]].append(r)
    split_by_site: dict[str, list[dict]] = defaultdict(list)
    for r in split_plan:
        split_by_site[r["site_table_id"]].append(r)
    return queue, split_plan, groups, members_by_group, split_by_site


def member_rollup(gid: str, members_by_group: dict[str, list[dict]]) -> dict[str, str]:
    members = members_by_group.get(gid, [])
    return {
        "member_count": str(len(members)),
        "member_pdf_pages": ";".join(
            f"{m.get('source_pdf_id') or m.get('source_pdf')}@{m.get('pdf_page')}" for m in members
        ),
        "member_images": ";".join(m.get("selected_image_path", "") for m in members if m.get("selected_image_path")),
        "member_ocr_artifacts": ";".join(
            m.get("selected_ocr_artifact", "") for m in members if m.get("selected_ocr_artifact")
        ),
        "member_quality_flags": unique_join(m.get("selected_quality_flags", "") for m in members),
        "member_edge_review_flags": unique_join(m.get("edge_review_flags", "") for m in members),
        "member_rows_extracted_total": str(sum(intish(m.get("selected_rows_extracted")) for m in members)),
        "member_constraint_failed_total": str(sum(intish(m.get("selected_constraint_failed")) for m in members)),
    }


def manifest_base_from_queue(row: dict, groups: dict, members_by_group: dict) -> dict:
    gid = (row.get("adjacency_group_ids") or "").split(";")[0]
    g = groups.get(gid, {})
    out = {
        "current_site_table_id": row.get("site_table_id", ""),
        "proposed_table_id": row.get("site_table_id", ""),
        "adjacency_group_id": gid,
        "current_status": row.get("status", ""),
        "year": row.get("year", ""),
        "canonical_name": row.get("canonical_name", ""),
        "parent_geo": row.get("parent_geo", ""),
        "broad_region": row.get("broad_region", ""),
        "current_topic": row.get("canonical_topic", ""),
        "current_subject": row.get("subject", ""),
        "current_measure_class": row.get("measure_class", ""),
        "current_n_facts": row.get("n_facts", ""),
        "current_title": row.get("title", ""),
        "adjacency_priority_label": row.get("adjacency_priorities", "") or g.get("static_priority_label", ""),
        "adjacency_page_range": row.get("adjacency_page_ranges", "") or g.get("page_range", ""),
        "adjacency_page_kinds": row.get("adjacency_page_kinds", "") or g.get("page_kind_counts", ""),
        "adjacency_review_flags": row.get("adjacency_review_flags", "") or g.get("component_review_flags", ""),
        "adjacency_primary_title": row.get("adjacency_primary_titles", "") or g.get("primary_title", ""),
        "source_pdf_ids": row.get("adjacency_source_pdf_ids", "") or g.get("source_pdf_ids", ""),
        "source_pdfs": row.get("adjacency_source_pdfs", "") or g.get("source_pdfs", ""),
        "source_queue_action": row.get("recommended_static_site_action", ""),
        "crosswalk_assessment": row.get("crosswalk_assessment", ""),
    }
    out.update(member_rollup(gid, members_by_group))
    return out


def build_manifest(queue, groups, members_by_group, split_by_site):
    rows = []
    review = []

    for q in queue:
        src_action = q.get("recommended_static_site_action", "")
        if src_action == "likely_keep_current_grouping":
            rec = manifest_base_from_queue(q, groups, members_by_group)
            rec.update(
                {
                    "manifest_action": "keep_current_table",
                    "publish_decision": "auto_keep",
                    "requires_fact_rebuild": "no",
                    "requires_manual_review": "no",
                    "reason": "Current published table maps to one compatible v4 adjacency group.",
                }
            )
            rows.append(finalize_record(rec))
            continue

        if src_action == "split_current_site_table_by_adjacency_groups":
            parts = split_by_site.get(q.get("site_table_id", ""), [])
            for part in parts:
                gid = part.get("adjacency_group_id", "")
                rec = {
                    "current_site_table_id": part.get("site_table_id", ""),
                    "proposed_table_id": stable_part_id(part.get("site_table_id", ""), part.get("split_index", ""), gid),
                    "adjacency_group_id": gid,
                    "current_status": part.get("current_status", ""),
                    "year": part.get("year", ""),
                    "canonical_name": part.get("canonical_name", ""),
                    "parent_geo": q.get("parent_geo", ""),
                    "broad_region": q.get("broad_region", ""),
                    "current_topic": q.get("canonical_topic", ""),
                    "current_subject": q.get("subject", ""),
                    "current_measure_class": q.get("measure_class", ""),
                    "current_n_facts": q.get("n_facts", ""),
                    "current_title": part.get("current_title", ""),
                    "adjacency_priority_label": part.get("adjacency_priority_label", ""),
                    "adjacency_page_range": part.get("adjacency_page_range", ""),
                    "adjacency_page_kinds": part.get("adjacency_page_kind_counts", ""),
                    "adjacency_review_flags": part.get("adjacency_review_flags", ""),
                    "adjacency_primary_title": part.get("adjacency_primary_title", ""),
                    "source_pdf_ids": part.get("source_pdf_ids", ""),
                    "source_pdfs": "",
                    "source_queue_action": src_action,
                    "crosswalk_assessment": q.get("crosswalk_assessment", ""),
                    "manifest_action": "split_current_table_by_adjacency",
                    "publish_decision": "review_before_publish",
                    "requires_fact_rebuild": "yes",
                    "requires_manual_review": "yes",
                    "reason": "Current published table spans multiple adjacency groups; page grouping is split, but fact rows must be rebuilt or verified per split.",
                }
                rec.update(member_rollup(gid, members_by_group))
                rec = finalize_record(rec)
                rows.append(rec)
                review.append(rec)
            continue

        rec = manifest_base_from_queue(q, groups, members_by_group)
        if src_action == "remove_or_demote_from_main_age_static_site":
            manifest_action = "demote_or_remove_from_main_age_view"
            decision = "review_before_demote"
            fact_rebuild = "no"
            reason = "Current main age/civil signal maps to derivative or off-target v4 page kind."
        elif src_action == "review_partial_image_mapping_before_site_update":
            manifest_action = "review_partial_mapping"
            decision = "review_before_publish"
            fact_rebuild = "unknown"
            reason = "Only part of the current table image set matched v4 adjacency groups."
        elif src_action == "review_no_adjacency_match":
            manifest_action = "review_no_v4_match"
            decision = "review_before_publish"
            fact_rebuild = "unknown"
            reason = "Current published main age/civil table did not match v4 adjacency by image hash."
        elif src_action == "candidate_promote_or_add_from_v4_not_in_current_published_main_age_civil":
            rec["proposed_table_id"] = f"v4_{rec['adjacency_group_id']}"
            manifest_action = "promote_or_add_v4_adjacency_group"
            decision = "review_before_publish"
            fact_rebuild = "yes"
            reason = "V4 has a main age/civil adjacency group not represented by a current published main age/civil table."
        else:
            manifest_action = "review_unknown_action"
            decision = "review_before_publish"
            fact_rebuild = "unknown"
            reason = f"Unhandled queue action: {src_action}"

        rec.update(
            {
                "manifest_action": manifest_action,
                "publish_decision": decision,
                "requires_fact_rebuild": fact_rebuild,
                "requires_manual_review": "yes",
                "reason": reason,
            }
        )
        rec = finalize_record(rec)
        rows.append(rec)
        review.append(rec)

    fields = [
        "review_priority_rank",
        "review_priority_label",
        "manifest_action",
        "publish_decision",
        "requires_manual_review",
        "requires_fact_rebuild",
        "reason",
        "current_site_table_id",
        "proposed_table_id",
        "adjacency_group_id",
        "year",
        "canonical_name",
        "parent_geo",
        "broad_region",
        "current_status",
        "current_topic",
        "current_subject",
        "current_measure_class",
        "current_n_facts",
        "current_title",
        "adjacency_priority_label",
        "adjacency_page_range",
        "adjacency_page_kinds",
        "adjacency_review_flags",
        "adjacency_primary_title",
        "source_pdf_ids",
        "source_pdfs",
        "member_count",
        "member_pdf_pages",
        "member_rows_extracted_total",
        "member_constraint_failed_total",
        "member_quality_flags",
        "member_edge_review_flags",
        "member_ocr_artifacts",
        "member_images",
        "source_queue_action",
        "crosswalk_assessment",
    ]
    return rows, review, fields


def build_year_region_summary(rows: list[dict]) -> list[dict]:
    grouped = defaultdict(lambda: Counter())
    for r in rows:
        key = (
            r.get("year", ""),
            r.get("broad_region", ""),
            r.get("manifest_action", ""),
            r.get("review_priority_label", ""),
        )
        c = grouped[key]
        c["rows"] += 1
        if r.get("requires_manual_review") == "yes":
            c["requires_review"] += 1
        if r.get("requires_fact_rebuild") == "yes":
            c["requires_fact_rebuild"] += 1
        c["member_pages"] += intish(r.get("member_count"))
        c["source_current_tables"] += 1 if r.get("current_site_table_id") else 0

    out = []
    for (year, region, action, priority), c in grouped.items():
        out.append(
            {
                "year": year,
                "broad_region": region,
                "manifest_action": action,
                "review_priority_label": priority,
                "rows": c["rows"],
                "requires_review": c["requires_review"],
                "requires_fact_rebuild": c["requires_fact_rebuild"],
                "member_pages": c["member_pages"],
                "source_current_table_rows": c["source_current_tables"],
            }
        )
    return sorted(
        out,
        key=lambda r: (
            r["review_priority_label"],
            str(r["year"]),
            str(r["broad_region"]),
            str(r["manifest_action"]),
        ),
    )


def write_json_manifest(rows: list[dict], summary: dict) -> None:
    by_action = defaultdict(list)
    for r in rows:
        by_action[r["manifest_action"]].append(r)
    payload = {
        "summary": summary,
        "actions": {k: v for k, v in sorted(by_action.items())},
    }
    with open(OUT_MANIFEST_JSON, "w") as f:
        json.dump(payload, f, indent=2)


def write_workbook(
    rows: list[dict],
    review: list[dict],
    fields: list[str],
    summary: dict,
    year_region_summary: list[dict],
) -> None:
    if xlsxwriter is None:
        return
    wb = xlsxwriter.Workbook(OUT_XLSX)
    header = wb.add_format({"bold": True, "bg_color": "#DDEBF7", "border": 1})
    wrap = wb.add_format({"text_wrap": True, "valign": "top"})

    ws = wb.add_worksheet("README")
    ws.write(0, 0, "Purpose", header)
    ws.write(0, 1, "Conservative manifest for later static-site update from v4 adjacency grouping.")
    ws.write(2, 0, "Rule", header)
    ws.write(2, 1, "Only keep_current_table is auto. Splits, demotions, no-matches, partial matches, and v4 promote/add candidates require review before publish.")
    r = 4
    for k, v in summary.items():
        if not isinstance(v, dict):
            ws.write(r, 0, k)
            ws.write(r, 1, v)
            r += 1

    sheets = [
        ("Manifest", rows, fields),
        ("ReviewQueue", review, fields),
        (
            "SummaryByYearRegion",
            year_region_summary,
            list(year_region_summary[0].keys()) if year_region_summary else [],
        ),
    ]
    for name, data, sheet_fields in sheets:
        sh = wb.add_worksheet(name)
        if not sheet_fields:
            continue
        for c, col in enumerate(sheet_fields):
            sh.write(0, c, col, header)
        for i, row in enumerate(data, 1):
            for c, col in enumerate(sheet_fields):
                sh.write(i, c, row.get(col, ""), wrap)
        sh.freeze_panes(1, 0)
        sh.autofilter(0, 0, len(data), len(sheet_fields) - 1)
        for c, col in enumerate(sheet_fields):
            width = 16
            if col in {"reason", "current_title", "adjacency_primary_title", "member_images", "member_ocr_artifacts", "member_pdf_pages"}:
                width = 48
            elif col.endswith("_id") or col.endswith("_ids") or col in {"current_site_table_id", "proposed_table_id"}:
                width = 34
            sh.set_column(c, c, width)

    wb.close()


def main() -> None:
    queue, split_plan, groups, members_by_group, split_by_site = load_inputs()
    rows, review, fields = build_manifest(queue, groups, members_by_group, split_by_site)
    year_region_summary = build_year_region_summary(rows)

    action_counts = Counter(r["manifest_action"] for r in rows)
    review_counts = Counter(r["manifest_action"] for r in review)
    summary = {
        "manifest_rows": len(rows),
        "review_rows": len(review),
        "source_update_queue_rows": len(queue),
        "source_split_plan_rows": len(split_plan),
        "auto_keep_rows": action_counts.get("keep_current_table", 0),
        "requires_review_rows": sum(1 for r in rows if r["requires_manual_review"] == "yes"),
        "requires_fact_rebuild_rows": sum(1 for r in rows if r["requires_fact_rebuild"] == "yes"),
        "manifest_action_counts": dict(action_counts),
        "review_action_counts": dict(review_counts),
        "outputs": {
            "manifest_csv": OUT_MANIFEST_CSV,
            "manifest_json": OUT_MANIFEST_JSON,
            "review_csv": OUT_REVIEW_CSV,
            "summary_by_year_region": OUT_BY_YEAR_REGION,
            "workbook": OUT_XLSX if xlsxwriter is not None else "",
        },
    }

    write_csv(OUT_MANIFEST_CSV, rows, fields)
    write_csv(OUT_REVIEW_CSV, review, fields + ["human_decision", "human_notes"])
    write_csv(OUT_BY_YEAR_REGION, year_region_summary, list(year_region_summary[0].keys()) if year_region_summary else [])
    write_json_manifest(rows, summary)
    write_workbook(rows, review, fields, summary, year_region_summary)
    with open(OUT_SUMMARY, "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
