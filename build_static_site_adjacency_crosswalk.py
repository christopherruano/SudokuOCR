#!/usr/bin/env python3
"""Compare current Census Explorer/static-site table grouping to v4 adjacency groups.

This is intentionally read-only against the existing site/DB. It writes review
artifacts into the v4 adjacency output directory so we can decide what the static
site should split, keep, or review before changing the public build.
"""

from __future__ import annotations

import csv
import json
import os
import re
import sqlite3
from collections import Counter, defaultdict

try:
    import xlsxwriter
except ImportError:  # pragma: no cover - CSV/JSON outputs still work.
    xlsxwriter = None


ROOT = os.path.dirname(os.path.abspath(__file__))
SENDOUT = os.path.join(ROOT, "export_final", "sendout_final", "sendout_v1")
DB = os.path.join(SENDOUT, "census_explorer.sqlite")
STATIC_EXCLUDE = os.path.join(SENDOUT, "static_exclude.txt")

V4_DIR = os.path.join(
    ROOT,
    "export_final",
    "pdf_recovery_incoming_2026-06-16_all_ocr_component_assembly_v4_adjacency",
)
V4_GROUPS = os.path.join(V4_DIR, "static_site_groups.csv")
V4_MEMBERS = os.path.join(V4_DIR, "static_site_group_members.csv")

OUT_SITE_TO_ADJ = os.path.join(V4_DIR, "current_site_to_adjacency_crosswalk.csv")
OUT_ADJ_TO_SITE = os.path.join(V4_DIR, "adjacency_to_current_site_crosswalk.csv")
OUT_SPLITS = os.path.join(V4_DIR, "current_site_split_by_adjacency.csv")
OUT_SPLIT_PLAN = os.path.join(V4_DIR, "current_site_split_plan_exploded.csv")
OUT_UPDATE_QUEUE = os.path.join(V4_DIR, "main_age_civil_static_site_update_queue.csv")
OUT_SUMMARY = os.path.join(V4_DIR, "site_adjacency_crosswalk_summary.json")
OUT_XLSX = os.path.join(V4_DIR, "site_adjacency_crosswalk_review.xlsx")

PUBLISH_STATUSES = {
    "confirmed_sent",
    "recovered",
    "keeper_accepted",
    "keeper_review",
    "assembled",
}
HEX40_RE = re.compile(r"(?i)([0-9a-f]{40})(?:_oneshot)?")


def read_csv(path: str) -> list[dict[str, str]]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: str, rows: list[dict], fieldnames: list[str]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def split_paths(raw: str) -> list[str]:
    if not raw:
        return []
    # Current tables use semicolons; older assembly records sometimes use pipes
    # or commas. Keep this deliberately permissive for review matching.
    return [p.strip() for p in re.split(r"[;|,]", raw) if p.strip()]


def image_hashes(raw: str) -> list[str]:
    seen = set()
    out = []
    for part in split_paths(raw):
        for m in HEX40_RE.finditer(part):
            h = m.group(1).lower()
            if h not in seen:
                seen.add(h)
                out.append(h)
    return out


def norm_join(values) -> str:
    vals = []
    seen = set()
    for v in values:
        if v is None:
            continue
        s = str(v).strip()
        if not s or s in seen:
            continue
        seen.add(s)
        vals.append(s)
    return " | ".join(vals)


def intish(v, default=0) -> int:
    try:
        return int(float(str(v).strip()))
    except Exception:
        return default


def is_main_age_civil_table(row: dict[str, str]) -> bool:
    text = " ".join(
        str(row.get(k, "") or "")
        for k in ("canonical_topic", "subject", "table_family", "title")
    ).lower()
    return (
        "civil_condition" in text
        or "civil condition" in text
        or "table vii" in text
        or "table vil" in text
    ) and ("selected caste" not in text and "selected tribe" not in text)


def load_current_site_tables() -> list[dict[str, str]]:
    exclude = set()
    if os.path.exists(STATIC_EXCLUDE):
        with open(STATIC_EXCLUDE) as f:
            exclude = {line.strip() for line in f if line.strip()}

    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    rows = []
    for r in con.execute("SELECT * FROM tables ORDER BY broad_region, canonical_name, year, table_id"):
        row = dict(r)
        status = row.get("status")
        row["would_publish_current_static_site"] = (
            "yes" if status in PUBLISH_STATUSES and row["table_id"] not in exclude else "no"
        )
        row["in_static_exclude"] = "yes" if row["table_id"] in exclude else "no"
        raw_images = ";".join(
            p for p in (row.get("member_images"), row.get("source_image")) if p
        )
        hashes = image_hashes(raw_images)
        row["image_hashes"] = ";".join(hashes)
        row["n_image_hashes"] = len(hashes)
        row["main_age_civil_signal"] = "yes" if is_main_age_civil_table(row) else "no"
        rows.append(row)
    con.close()
    return rows


def load_v4() -> tuple[dict[str, dict], dict[str, list[dict]], dict[str, list[dict]]]:
    groups = {r["static_group_id"]: r for r in read_csv(V4_GROUPS)}
    members_by_group: dict[str, list[dict]] = defaultdict(list)
    image_to_members: dict[str, list[dict]] = defaultdict(list)

    for r in read_csv(V4_MEMBERS):
        gid = r["static_group_id"]
        members_by_group[gid].append(r)
        paths = [r.get("selected_image_path", ""), r.get("duplicate_image_paths", "")]
        for h in image_hashes(";".join(paths)):
            image_to_members[h].append(r)

    return groups, members_by_group, image_to_members


def summarize_groups(group_ids: list[str], groups: dict[str, dict]) -> dict[str, str]:
    rows = [groups[g] for g in group_ids if g in groups]
    return {
        "adjacency_group_ids": ";".join(group_ids),
        "adjacency_priorities": norm_join(r.get("static_priority_label") for r in rows),
        "adjacency_page_ranges": norm_join(r.get("page_range") for r in rows),
        "adjacency_page_kinds": norm_join(r.get("page_kind_counts") for r in rows),
        "adjacency_review_flags": norm_join(r.get("component_review_flags") for r in rows),
        "adjacency_primary_titles": norm_join(r.get("primary_title") for r in rows),
        "adjacency_source_pdfs": norm_join(r.get("source_pdfs") for r in rows),
        "adjacency_source_pdf_ids": norm_join(r.get("source_pdf_ids") for r in rows),
    }


def classify_site_row(row: dict, group_ids: list[str], matched_hashes: set[str], groups: dict[str, dict]) -> str:
    if intish(row.get("n_image_hashes")) == 0:
        return "no_image_hashes_in_current_table"
    if not group_ids:
        return "no_v4_image_match"
    if len(matched_hashes) < intish(row.get("n_image_hashes")):
        return "partial_v4_image_match"
    if len(group_ids) == 1:
        gid = group_ids[0]
        label = groups.get(gid, {}).get("static_priority_label", "")
        if "offtarget" in label:
            return "single_v4_group_but_offtarget"
        if "derivative_selected" in label:
            return "single_v4_group_but_derivative"
        return "single_v4_group"
    return "split_by_adjacency"


def build_site_crosswalk(
    site_rows: list[dict[str, str]],
    groups: dict[str, dict],
    image_to_members: dict[str, list[dict]],
) -> tuple[list[dict], dict[str, list[dict]]]:
    adjacency_to_site: dict[str, list[dict]] = defaultdict(list)
    out = []

    for row in site_rows:
        hashes = [h for h in row.get("image_hashes", "").split(";") if h]
        matched_hashes = set()
        group_ids = []
        seen_groups = set()
        matched_member_pages = []
        for h in hashes:
            for m in image_to_members.get(h, []):
                matched_hashes.add(h)
                gid = m["static_group_id"]
                if gid not in seen_groups:
                    seen_groups.add(gid)
                    group_ids.append(gid)
                matched_member_pages.append(
                    f"{gid}:{m.get('source_pdf_id') or m.get('source_pdf')}@{m.get('pdf_page')}"
                )

        assess = classify_site_row(row, group_ids, matched_hashes, groups)
        for gid in group_ids:
            adjacency_to_site[gid].append(row)

        rec = {
            "site_table_id": row.get("table_id", ""),
            "would_publish_current_static_site": row.get("would_publish_current_static_site", ""),
            "in_static_exclude": row.get("in_static_exclude", ""),
            "crosswalk_assessment": assess,
            "main_age_civil_signal": row.get("main_age_civil_signal", ""),
            "status": row.get("status", ""),
            "year": row.get("year", ""),
            "canonical_name": row.get("canonical_name", ""),
            "parent_geo": row.get("parent_geo", ""),
            "broad_region": row.get("broad_region", ""),
            "canonical_topic": row.get("canonical_topic", ""),
            "subject": row.get("subject", ""),
            "measure_class": row.get("measure_class", ""),
            "n_facts": row.get("n_facts", ""),
            "title": row.get("title", ""),
            "n_current_image_hashes": len(hashes),
            "n_matched_image_hashes": len(matched_hashes),
            "n_adjacency_groups": len(group_ids),
            "current_image_hashes": ";".join(hashes),
            "matched_image_hashes": ";".join(sorted(matched_hashes)),
            "matched_member_pages": norm_join(matched_member_pages),
        }
        rec.update(summarize_groups(group_ids, groups))
        out.append(rec)

    return out, adjacency_to_site


def build_adjacency_crosswalk(
    groups: dict[str, dict],
    members_by_group: dict[str, list[dict]],
    adjacency_to_site: dict[str, list[dict]],
) -> list[dict]:
    out = []
    for gid, g in groups.items():
        site = adjacency_to_site.get(gid, [])
        site_ids = [r.get("table_id", "") for r in site]
        publish = [r for r in site if r.get("would_publish_current_static_site") == "yes"]
        main_site = [r for r in site if r.get("main_age_civil_signal") == "yes"]
        members = members_by_group.get(gid, [])
        out.append(
            {
                "static_group_id": gid,
                "static_priority": g.get("static_priority", ""),
                "static_priority_label": g.get("static_priority_label", ""),
                "component_basis": g.get("component_basis", ""),
                "review_flag": g.get("review_flag", ""),
                "component_review_flags": g.get("component_review_flags", ""),
                "n_unique_physical_pages": g.get("n_unique_physical_pages", ""),
                "n_source_records": g.get("n_source_records", ""),
                "duplicate_record_count": g.get("duplicate_record_count", ""),
                "page_range": g.get("page_range", ""),
                "page_kind_counts": g.get("page_kind_counts", ""),
                "primary_title": g.get("primary_title", ""),
                "source_pdfs": g.get("source_pdfs", ""),
                "source_pdf_ids": g.get("source_pdf_ids", ""),
                "n_current_site_tables_matched": len(site),
                "n_current_published_site_tables_matched": len(publish),
                "n_current_main_age_civil_site_tables_matched": len(main_site),
                "site_match_assessment": (
                    "not_in_current_site"
                    if not site
                    else "matched_one_site_table"
                    if len(site) == 1
                    else "shared_by_multiple_site_tables"
                ),
                "current_site_table_ids": ";".join(site_ids),
                "current_site_titles": norm_join(r.get("title", "") for r in site),
                "member_pdf_pages": ";".join(
                    f"{m.get('source_pdf_id') or m.get('source_pdf')}@{m.get('pdf_page')}" for m in members
                ),
            }
        )
    return out


def build_update_queue(site_to_adj: list[dict], adj_to_site: list[dict]) -> list[dict]:
    queue = []
    for r in site_to_adj:
        if r["would_publish_current_static_site"] != "yes":
            continue
        if r["main_age_civil_signal"] != "yes":
            continue
        assessment = r["crosswalk_assessment"]
        if assessment == "single_v4_group":
            action = "likely_keep_current_grouping"
        elif assessment == "split_by_adjacency":
            action = "split_current_site_table_by_adjacency_groups"
        elif assessment == "partial_v4_image_match":
            action = "review_partial_image_mapping_before_site_update"
        elif assessment in {"single_v4_group_but_offtarget", "single_v4_group_but_derivative"}:
            action = "remove_or_demote_from_main_age_static_site"
        else:
            action = "review_no_adjacency_match"
        q = dict(r)
        q["recommended_static_site_action"] = action
        queue.append(q)

    # Also include v4 main candidates that are not represented by a current
    # published main age/civil table. Some may already exist elsewhere in the
    # DB under a different topic/status, so avoid calling them universally absent.
    represented = set()
    for r in queue:
        for gid in r.get("adjacency_group_ids", "").split(";"):
            if gid:
                represented.add(gid)
    for r in adj_to_site:
        label = r.get("static_priority_label", "")
        if "main_age_civil" not in label:
            continue
        if r["static_group_id"] in represented:
            continue
        q = {
            "site_table_id": "",
            "would_publish_current_static_site": "no",
            "in_static_exclude": "",
            "crosswalk_assessment": r.get("site_match_assessment", ""),
            "main_age_civil_signal": "v4_main_age_civil",
            "status": "",
            "year": "",
            "canonical_name": "",
            "parent_geo": "",
            "broad_region": "",
            "canonical_topic": "",
            "subject": "",
            "measure_class": "",
            "n_facts": "",
            "title": r.get("primary_title", ""),
            "n_current_image_hashes": "",
            "n_matched_image_hashes": "",
            "n_adjacency_groups": "1",
            "current_image_hashes": "",
            "matched_image_hashes": "",
            "matched_member_pages": r.get("member_pdf_pages", ""),
            "adjacency_group_ids": r["static_group_id"],
            "adjacency_priorities": label,
            "adjacency_page_ranges": r.get("page_range", ""),
            "adjacency_page_kinds": r.get("page_kind_counts", ""),
            "adjacency_review_flags": r.get("component_review_flags", ""),
            "adjacency_primary_titles": r.get("primary_title", ""),
            "adjacency_source_pdfs": r.get("source_pdfs", ""),
            "adjacency_source_pdf_ids": r.get("source_pdf_ids", ""),
            "recommended_static_site_action": "candidate_promote_or_add_from_v4_not_in_current_published_main_age_civil",
        }
        queue.append(q)

    def sort_key(r):
        action_rank = {
            "split_current_site_table_by_adjacency_groups": 0,
            "remove_or_demote_from_main_age_static_site": 1,
            "review_partial_image_mapping_before_site_update": 2,
            "review_no_adjacency_match": 3,
            "candidate_promote_or_add_from_v4_not_in_current_published_main_age_civil": 4,
            "likely_keep_current_grouping": 5,
        }.get(r.get("recommended_static_site_action"), 9)
        return (action_rank, str(r.get("year", "")), str(r.get("canonical_name", "")), str(r.get("site_table_id", "")))

    return sorted(queue, key=sort_key)


def build_split_plan(splits: list[dict], groups: dict[str, dict], members_by_group: dict[str, list[dict]]) -> list[dict]:
    out = []
    for r in splits:
        gids = [g for g in r.get("adjacency_group_ids", "").split(";") if g]
        for idx, gid in enumerate(gids, 1):
            g = groups.get(gid, {})
            members = members_by_group.get(gid, [])
            member_hashes = []
            member_images = []
            member_pages = []
            for m in members:
                img = m.get("selected_image_path", "")
                member_images.append(img)
                member_pages.append(f"{m.get('source_pdf_id') or m.get('source_pdf')}@{m.get('pdf_page')}")
                member_hashes.extend(image_hashes(img))
            out.append(
                {
                    "site_table_id": r.get("site_table_id", ""),
                    "split_index": idx,
                    "n_splits_for_current_table": len(gids),
                    "main_age_civil_signal": r.get("main_age_civil_signal", ""),
                    "current_status": r.get("status", ""),
                    "year": r.get("year", ""),
                    "canonical_name": r.get("canonical_name", ""),
                    "current_title": r.get("title", ""),
                    "adjacency_group_id": gid,
                    "adjacency_priority_label": g.get("static_priority_label", ""),
                    "adjacency_component_basis": g.get("component_basis", ""),
                    "adjacency_review_flags": g.get("component_review_flags", ""),
                    "adjacency_page_range": g.get("page_range", ""),
                    "adjacency_page_kind_counts": g.get("page_kind_counts", ""),
                    "adjacency_primary_title": g.get("primary_title", ""),
                    "source_pdf_ids": g.get("source_pdf_ids", ""),
                    "member_pdf_pages": ";".join(member_pages),
                    "member_image_hashes": ";".join(dict.fromkeys(member_hashes)),
                    "member_selected_images": ";".join(member_images),
                }
            )
    return out


def write_workbook(
    site_to_adj: list[dict],
    adj_to_site: list[dict],
    splits: list[dict],
    split_plan: list[dict],
    update_queue: list[dict],
) -> None:
    if xlsxwriter is None:
        return
    wb = xlsxwriter.Workbook(OUT_XLSX)
    header_fmt = wb.add_format({"bold": True, "bg_color": "#DDEBF7", "border": 1})
    wrap_fmt = wb.add_format({"text_wrap": True, "valign": "top"})
    readme = wb.add_worksheet("README")
    readme.write(0, 0, "Purpose")
    readme.write(0, 1, "Compare current Census Explorer/static-site table grouping to v4 adjacency-first physical-page groups.")
    readme.write(2, 0, "Use")
    readme.write(2, 1, "Start with MainAgeCivilQueue. Splits are likely current over-associations; candidate promote/add rows are v4 main age/civil groups not represented by a current published main age/civil table.")

    sheets = [
        ("MainAgeCivilQueue", update_queue),
        ("CurrentSiteToAdjacency", site_to_adj),
        ("Splits", splits),
        ("SplitPlan", split_plan),
        ("AdjacencyToCurrentSite", adj_to_site),
    ]
    for name, rows in sheets:
        ws = wb.add_worksheet(name[:31])
        if not rows:
            continue
        cols = list(rows[0].keys())
        for c, col in enumerate(cols):
            ws.write(0, c, col, header_fmt)
        for r_i, row in enumerate(rows, 1):
            for c, col in enumerate(cols):
                ws.write(r_i, c, row.get(col, ""), wrap_fmt)
        ws.freeze_panes(1, 0)
        ws.autofilter(0, 0, len(rows), len(cols) - 1)
        for c, col in enumerate(cols):
            width = 14
            if col in {"title", "adjacency_primary_titles", "current_site_titles", "matched_member_pages"}:
                width = 48
            elif col.endswith("_ids") or col.endswith("_hashes") or col in {"site_table_id", "adjacency_group_ids"}:
                width = 32
            ws.set_column(c, c, width)
    wb.close()


def main() -> None:
    os.makedirs(V4_DIR, exist_ok=True)
    site_rows = load_current_site_tables()
    groups, members_by_group, image_to_members = load_v4()

    site_to_adj, adjacency_to_site = build_site_crosswalk(site_rows, groups, image_to_members)
    adj_to_site = build_adjacency_crosswalk(groups, members_by_group, adjacency_to_site)
    splits = [
        r
        for r in site_to_adj
        if r["would_publish_current_static_site"] == "yes"
        and r["crosswalk_assessment"] == "split_by_adjacency"
    ]
    split_plan = build_split_plan(splits, groups, members_by_group)
    update_queue = build_update_queue(site_to_adj, adj_to_site)

    site_fields = list(site_to_adj[0].keys()) if site_to_adj else []
    adj_fields = list(adj_to_site[0].keys()) if adj_to_site else []
    queue_fields = list(update_queue[0].keys()) if update_queue else []
    write_csv(OUT_SITE_TO_ADJ, site_to_adj, site_fields)
    write_csv(OUT_ADJ_TO_SITE, adj_to_site, adj_fields)
    write_csv(OUT_SPLITS, splits, site_fields)
    write_csv(OUT_SPLIT_PLAN, split_plan, list(split_plan[0].keys()) if split_plan else [])
    write_csv(OUT_UPDATE_QUEUE, update_queue, queue_fields)
    write_workbook(site_to_adj, adj_to_site, splits, split_plan, update_queue)

    publish_rows = [r for r in site_to_adj if r["would_publish_current_static_site"] == "yes"]
    main_publish = [r for r in publish_rows if r["main_age_civil_signal"] == "yes"]
    main_queue_counts = Counter(r["recommended_static_site_action"] for r in update_queue)
    summary = {
        "current_db_tables_total": len(site_rows),
        "current_static_site_publishable_tables_after_exclude": len(publish_rows),
        "current_static_site_publishable_main_age_civil_tables": len(main_publish),
        "current_site_crosswalk_assessment_counts": dict(Counter(r["crosswalk_assessment"] for r in publish_rows)),
        "current_main_age_civil_crosswalk_assessment_counts": dict(Counter(r["crosswalk_assessment"] for r in main_publish)),
        "current_publishable_tables_split_by_adjacency": len(splits),
        "current_publishable_main_age_civil_split_by_adjacency": sum(1 for r in splits if r["main_age_civil_signal"] == "yes"),
        "current_publishable_split_plan_rows": len(split_plan),
        "v4_adjacency_groups_total": len(groups),
        "v4_adjacency_groups_matched_to_current_site": sum(1 for r in adj_to_site if intish(r["n_current_site_tables_matched"]) > 0),
        "v4_adjacency_groups_not_in_current_site": sum(1 for r in adj_to_site if intish(r["n_current_site_tables_matched"]) == 0),
        "main_age_civil_update_queue_rows": len(update_queue),
        "main_age_civil_update_queue_action_counts": dict(main_queue_counts),
        "outputs": {
            "site_to_adjacency": OUT_SITE_TO_ADJ,
            "adjacency_to_site": OUT_ADJ_TO_SITE,
            "splits": OUT_SPLITS,
            "split_plan_exploded": OUT_SPLIT_PLAN,
            "main_age_civil_update_queue": OUT_UPDATE_QUEUE,
            "workbook": OUT_XLSX if xlsxwriter is not None else "",
        },
    }
    with open(OUT_SUMMARY, "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
