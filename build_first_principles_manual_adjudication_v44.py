#!/usr/bin/env python3
"""Build a focused manual adjudication packet for remaining assembly candidates.

This layer performs no OCR/API work and does not promote any edge. It reads the
v43 assembly frontier and turns adjacent manual-review candidates into a small,
sortable review queue with explicit suggested actions and visual checklists.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import zipfile
from collections import Counter
from pathlib import Path
from typing import Iterable

import xlsxwriter


ROOT = Path(__file__).resolve().parent
DEFAULT_FRONTIER_DIR = ROOT / "export_final/pdf_recovery_incoming_2026-06-16_first_principles_assembly_frontier_smoke_v43_frontier"
DEFAULT_OUT_DIR = ROOT / "export_final/pdf_recovery_incoming_2026-06-16_first_principles_manual_adjudication_v44"

ADJUDICATION_FIELDS = [
    "adjudication_rank",
    "edge_key",
    "adjudication_tier",
    "default_decision",
    "suggested_action",
    "visual_review_question",
    "promotion_if_confirmed",
    "do_not_promote_if",
    "visual_checklist",
    "safe_to_auto_promote",
    "frontier_category",
    "risk_flags",
    "pair_direction_signal",
    "pair_direction_reason",
    "v5_source_key",
    "source_pdf_id_guess",
    "page_pair",
    "pages_consecutive",
    "physical_rule_status",
    "row_alignment_evidence",
    "column_order_evidence",
    "zero_check_status",
    "zero_check_failures",
    "left_row_count",
    "right_row_count",
    "left_row_label_count",
    "right_row_label_count",
    "shared_row_label_count",
    "left_first_labels",
    "right_first_labels",
    "left_column_groups",
    "right_column_groups",
    "combined_column_group_order",
    "left_title",
    "right_title",
    "rule_decision_reason",
    "left_best_json",
    "right_best_json",
    "left_best_csv",
    "right_best_csv",
    "prev_image_path",
    "image_path",
    "prev_image_exists",
    "image_exists",
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


def int_or_none(value: object) -> int | None:
    if value in {None, ""}:
        return None
    try:
        return int(str(value).replace(",", ""))
    except ValueError:
        return None


def split_semicolon(value: object) -> list[str]:
    return [part.strip() for part in str(value or "").split(";") if part.strip()]


def semi(values: Iterable[object]) -> str:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    return "; ".join(out)


def resolve_path(value: object) -> Path:
    text = str(value or "").strip()
    if not text:
        return Path("")
    path = Path(text).expanduser()
    if path.is_absolute():
        return path
    return ROOT / path


def path_exists(value: object) -> str:
    path = resolve_path(value)
    return "yes" if str(path) and path.exists() else "no"


def path_uri(value: object) -> str:
    path = resolve_path(value)
    if not str(path):
        return ""
    try:
        return path.resolve(strict=False).as_uri()
    except ValueError:
        return html.escape(str(path))


def classify(row: dict[str, str]) -> tuple[str, str, str, str, str, str, str]:
    risks = set(split_semicolon(row.get("risk_flags", "")))
    strong_row = row.get("row_alignment_evidence") in {"same_row_labels_exact", "same_row_labels_substantial_overlap"}
    strong_col = row.get("column_order_evidence") == "left_page_groups_then_right_page_groups_disjoint"
    zero_pass = row.get("zero_check_status") == "zero_checks_pass"
    direction = row.get("pair_direction_signal", "")

    if direction == "civil_status_end_to_restart_boundary":
        tier = "tier_9_likely_boundary_keep_separate"
        default = "keep_separate_by_default"
        action = "Treat as a table-boundary crossing: page A appears to end a civil-status segment and page B restarts the next one."
        question = "Is there any visual evidence overriding the end-to-restart column sequence?"
        promote = "Only promote with explicit manual override and notes; the rule stack says this is probably not a physical join."
        reject = "Keep separate if page B begins a new taluk/district/religion or restarts population/unmarried columns."
    elif row.get("frontier_category") == "manual_visual_review_likely_civil_status_continuation":
        tier = "tier_1_likely_civil_status_continuation"
        default = "visual_review_then_promote_structure_if_confirmed"
        action = "High-value review: likely two-page Table VII split from population/unmarried into married/widowed columns."
        question = "Does page B continue the exact same physical table from page A, adding married/widowed columns for the same row universe?"
        promote = "If yes: promote as a two-page physical structure; keep values under normal review policy."
        reject = "Do not promote if page B changes geography, religion, table subject, or row universe."
    elif row.get("frontier_category") == "manual_visual_review_row_label_dependency_high" and strong_row and strong_col and zero_pass:
        tier = "tier_1_promotable_if_visual_continuation"
        default = "visual_review_then_promote_structure_if_confirmed"
        action = "Inspect the image pair first; promote only if page B is the same table continuation with no title, geography, or subtable switch."
        question = "Does the right page continue the same physical table horizontally, preserving the same row universe and adding later column groups?"
        promote = "If yes: promote as a two-page physical structure and keep values under review policy."
        reject = "Do not promote if the right page starts a new district/taluk, repeats the same column block vertically, or changes table/subtable."
    elif "zero_check_failures" in risks:
        tier = "tier_2_value_anomaly_structure_review"
        default = "usually_keep_values_under_review_even_if_structure_promotes"
        action = "Review structure and zero-check failures together; structure can only be promoted if the image is unambiguous."
        question = "Are the zero failures OCR/value noise on a visually clear continuation, or do they indicate a bad assembly?"
        promote = "If visual continuation is exact: promote structure only, with values flagged for correction/review."
        reject = "Do not promote if the zero issue comes from mixed row universes, duplicate columns, or a table switch."
    elif "row_label_mismatch" in risks and strong_col:
        tier = "tier_3_row_block_switch_review"
        default = "usually_keep_separate_unless_next_row_block"
        action = "Confirm whether the row labels are the next block of the same table, not an unrelated table with similar columns."
        question = "Are page B's row titles a legitimate next block of the same table, or a new geography/subtable?"
        promote = "If the row block is visibly the next block of the same printed table: keep as a bounded continuation candidate."
        reject = "Do not promote if row labels represent a new district/taluk/religion/table segment rather than continuation."
    elif "duplicate_or_vertical_column_groups" in risks:
        tier = "tier_4_duplicate_or_vertical_columns_review"
        default = "keep_separate_by_default"
        action = "Treat as vertical/duplicate unless the image clearly shows a horizontal split not captured by OCR."
        question = "Are the same column groups repeated because this is a vertical continuation, duplicate extraction, or a separate table?"
        promote = "Only promote after explicit visual confirmation that the OCR column groups are misleading."
        reject = "Do not promote when the same column groups repeat normally down the page/table."
    else:
        tier = "tier_5_low_confidence_keep_separate"
        default = "keep_separate_pending_manual_override"
        action = "Keep separate unless visual review finds a compelling continuation that the rules missed."
        question = "Is there direct visual evidence strong enough to override the conservative rule stack?"
        promote = "Only promote with an explicit reviewer note."
        reject = "Default is no physical merge."

    checklist = semi(
        [
            "PDF pages are adjacent",
            "same source PDF/source key",
            "no new table number or title switch",
            "right page is continuation/concluded, not a new table",
            "row universe is same or next legitimate block",
            "column groups continue in left-to-right order",
            "values/zero checks do not contradict the structure",
        ]
    )
    return tier, default, action, question, promote, reject, checklist


def build_rows(manual_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in manual_rows:
        tier, default, action, question, promote, reject, checklist = classify(row)
        out = dict(row)
        out.update(
            {
                "adjudication_tier": tier,
                "default_decision": default,
                "suggested_action": action,
                "visual_review_question": question,
                "promotion_if_confirmed": promote,
                "do_not_promote_if": reject,
                "visual_checklist": checklist,
                "safe_to_auto_promote": "no",
                "prev_image_exists": path_exists(row.get("prev_image_path", "")),
                "image_exists": path_exists(row.get("image_path", "")),
            }
        )
        rows.append(out)
    rows.sort(
        key=lambda r: (
            r.get("adjudication_tier", ""),
            r.get("v5_source_key", ""),
            int_or_none(str(r.get("page_pair", "")).split("-")[0]) or 999999,
            r.get("edge_key", ""),
        )
    )
    for idx, row in enumerate(rows, start=1):
        row["adjudication_rank"] = idx
    return rows


def build_html(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cards: list[str] = []
    for row in rows:
        left_uri = path_uri(row.get("prev_image_path", ""))
        right_uri = path_uri(row.get("image_path", ""))
        cards.append(
            f"""
<section class="card">
  <header>
    <h2>{html.escape(str(row.get('adjudication_rank', '')))}. {html.escape(str(row.get('edge_key', '')))}</h2>
    <p><b>{html.escape(str(row.get('adjudication_tier', '')))}</b> · pages {html.escape(str(row.get('page_pair', '')))} · {html.escape(str(row.get('default_decision', '')))}</p>
  </header>
  <div class="meta">
    <p><b>Question:</b> {html.escape(str(row.get('visual_review_question', '')))}</p>
    <p><b>Promote if:</b> {html.escape(str(row.get('promotion_if_confirmed', '')))}</p>
    <p><b>Do not promote if:</b> {html.escape(str(row.get('do_not_promote_if', '')))}</p>
    <p><b>Risk:</b> {html.escape(str(row.get('risk_flags', '')))}</p>
    <p><b>Rows:</b> {html.escape(str(row.get('row_alignment_evidence', '')))} · <b>Columns:</b> {html.escape(str(row.get('column_order_evidence', '')))} · <b>Zero:</b> {html.escape(str(row.get('zero_check_status', '')))}</p>
    <p><b>Left title:</b> {html.escape(str(row.get('left_title', '')))}</p>
    <p><b>Right title:</b> {html.escape(str(row.get('right_title', '')))}</p>
  </div>
  <div class="images">
    <figure><figcaption>Left page</figcaption><img src="{left_uri}" loading="lazy" /></figure>
    <figure><figcaption>Right page</figcaption><img src="{right_uri}" loading="lazy" /></figure>
  </div>
</section>
"""
        )
    document = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Manual Assembly Adjudication</title>
  <style>
    :root {{ color-scheme: light; --ink:#1e1b18; --paper:#f7f1e8; --line:#cabca8; --accent:#1f5e5a; }}
    body {{ margin:0; padding:24px; font-family: Georgia, 'Times New Roman', serif; background:var(--paper); color:var(--ink); }}
    h1 {{ margin:0 0 8px; font-size:32px; }}
    .intro {{ max-width:1100px; margin-bottom:20px; line-height:1.45; }}
    .card {{ background:#fffaf1; border:1px solid var(--line); border-radius:14px; padding:18px; margin:18px 0; box-shadow:0 8px 24px rgba(70,48,22,.08); }}
    .card h2 {{ margin:0 0 4px; font-size:20px; color:var(--accent); }}
    .meta {{ line-height:1.35; font-size:14px; }}
    .images {{ display:grid; grid-template-columns:1fr 1fr; gap:14px; align-items:start; }}
    figure {{ margin:0; }}
    figcaption {{ font-weight:bold; margin:0 0 6px; }}
    img {{ width:100%; max-height:900px; object-fit:contain; background:white; border:1px solid var(--line); }}
    @media (max-width: 900px) {{ .images {{ grid-template-columns:1fr; }} body {{ padding:12px; }} }}
  </style>
</head>
<body>
  <h1>Manual Assembly Adjudication</h1>
  <p class="intro">These are adjacent manual-review candidates only. Nothing here is auto-promoted. Use the question and checklist to decide whether an edge should become a bounded two-page physical structure or remain separate.</p>
  {''.join(cards)}
</body>
</html>
"""
    path.write_text(document, encoding="utf-8")


def write_workbook(path: Path, summary_rows: list[dict[str, object]], audits: list[dict[str, object]], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    workbook = xlsxwriter.Workbook(path)
    header = workbook.add_format({"bold": True, "bg_color": "#E6D9C8", "border": 1})
    wrap = workbook.add_format({"text_wrap": True, "valign": "top"})
    sheets = [
        ("Summary", summary_rows, SUMMARY_FIELDS),
        ("Audit", audits, AUDIT_FIELDS),
        ("Adjudication Queue", rows, ADJUDICATION_FIELDS),
    ]
    for name, sheet_rows, fields in sheets:
        ws = workbook.add_worksheet(name[:31])
        for col, field in enumerate(fields):
            ws.write(0, col, field, header)
        for r_idx, row in enumerate(sheet_rows, start=1):
            for c_idx, field in enumerate(fields):
                fmt = wrap if field in {
                    "suggested_action",
                    "visual_review_question",
                    "promotion_if_confirmed",
                    "do_not_promote_if",
                    "visual_checklist",
                    "risk_flags",
                    "left_title",
                    "right_title",
                    "rule_decision_reason",
                    "left_column_groups",
                    "right_column_groups",
                    "combined_column_group_order",
                    "prev_image_path",
                    "image_path",
                } else None
                ws.write(r_idx, c_idx, row.get(field, ""), fmt)
        ws.freeze_panes(1, 0)
        ws.autofilter(0, 0, max(len(sheet_rows), 1), max(len(fields) - 1, 0))
        for col, field in enumerate(fields):
            width = 16
            if field in {"edge_key"}:
                width = 36
            elif field in {"suggested_action", "visual_review_question", "promotion_if_confirmed", "do_not_promote_if", "visual_checklist"}:
                width = 52
            elif field.endswith("_path") or field.endswith("_json") or field.endswith("_csv"):
                width = 48
            elif field in {"left_title", "right_title", "risk_flags"}:
                width = 42
            ws.set_column(col, col, width)
    workbook.close()


def build_audits(rows: list[dict[str, object]], manual_input_count: int, workbook_path: Path, html_path: Path) -> list[dict[str, object]]:
    audits: list[dict[str, object]] = []
    audits.append({"check": "all_manual_input_rows_carried", "status": "pass" if len(rows) == manual_input_count else "fail", "detail": f"input={manual_input_count} output={len(rows)}"})
    non_adjacent = [row for row in rows if row.get("pages_consecutive") != "yes"]
    audits.append({"check": "adjudication_edges_are_adjacent", "status": "pass" if not non_adjacent else "fail", "detail": f"bad={len(non_adjacent)}"})
    auto_promoted = [row for row in rows if row.get("safe_to_auto_promote") == "yes"]
    audits.append({"check": "adjudication_does_not_auto_promote", "status": "pass" if not auto_promoted else "fail", "detail": f"bad={len(auto_promoted)}"})
    missing_leaks = [row for row in rows if "missing_ocr_artifact" in str(row.get("risk_flags", ""))]
    audits.append({"check": "manual_adjudication_excludes_missing_ocr_edges", "status": "pass" if not missing_leaks else "fail", "detail": f"bad={len(missing_leaks)}"})
    tier1_bad = [
        row for row in rows
        if row.get("adjudication_tier") == "tier_1_promotable_if_visual_continuation"
        and not (
            row.get("row_alignment_evidence") in {"same_row_labels_exact", "same_row_labels_substantial_overlap"}
            and row.get("column_order_evidence") == "left_page_groups_then_right_page_groups_disjoint"
            and row.get("zero_check_status") == "zero_checks_pass"
        )
    ]
    audits.append({"check": "tier1_candidates_have_strong_structural_evidence", "status": "pass" if not tier1_bad else "fail", "detail": f"bad={len(tier1_bad)}"})
    missing_images = [row for row in rows if row.get("prev_image_exists") != "yes" or row.get("image_exists") != "yes"]
    audits.append({"check": "adjudication_image_paths_exist", "status": "pass" if not missing_images else "warn", "detail": f"missing_or_unresolved={len(missing_images)} total={len(rows)}"})
    audits.append({"check": "html_gallery_written", "status": "pass" if html_path.exists() and html_path.stat().st_size > 0 else "fail", "detail": output_path(html_path) if html_path.exists() else "missing"})
    try:
        with zipfile.ZipFile(workbook_path) as zf:
            bad_file = zf.testzip()
        status = "pass" if bad_file is None else "fail"
        detail = "ok" if bad_file is None else str(bad_file)
    except Exception as exc:  # pragma: no cover
        status = "fail"
        detail = str(exc)
    audits.append({"check": "workbook_zip_integrity", "status": status, "detail": detail})
    return audits


def build_summary(rows: list[dict[str, object]], audits: list[dict[str, object]], paths: dict[str, Path]) -> list[dict[str, object]]:
    summary: list[dict[str, object]] = [
        {"metric": "counts", "name": "adjudication_rows", "count": len(rows)},
        {"metric": "counts", "name": "tier1_promotable_if_visual_continuation", "count": sum(1 for row in rows if row.get("adjudication_tier") == "tier_1_promotable_if_visual_continuation")},
        {"metric": "counts", "name": "audit_failure_count", "count": sum(1 for row in audits if row.get("status") == "fail")},
        {"metric": "counts", "name": "audit_warning_count", "count": sum(1 for row in audits if row.get("status") == "warn")},
    ]
    for metric, counter in [
        ("adjudication_by_tier", Counter(str(row.get("adjudication_tier", "")) for row in rows)),
        ("adjudication_by_default_decision", Counter(str(row.get("default_decision", "")) for row in rows)),
    ]:
        for name, count in counter.most_common():
            summary.append({"metric": metric, "name": name, "count": count})
    for name, path in paths.items():
        summary.append({"metric": "outputs", "name": name, "count": output_path(path)})
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frontier-dir", type=Path, default=DEFAULT_FRONTIER_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    manual_rows = read_csv(args.frontier_dir / "manual_visual_review_queue.csv")
    rows = build_rows(manual_rows)
    paths = {
        "adjudication_queue": out_dir / "manual_adjudication_queue.csv",
        "html_gallery": out_dir / "manual_adjudication_gallery.html",
        "workbook": out_dir / "manual_adjudication_workbook.xlsx",
        "audit": out_dir / "acceptance_audit.csv",
        "summary": out_dir / "summary.csv",
    }
    write_csv(paths["adjudication_queue"], rows, ADJUDICATION_FIELDS)
    build_html(paths["html_gallery"], rows)
    write_workbook(paths["workbook"], [], [], rows)
    audits = build_audits(rows, len(manual_rows), paths["workbook"], paths["html_gallery"])
    summary = build_summary(rows, audits, paths)
    write_workbook(paths["workbook"], summary, audits, rows)
    audits = build_audits(rows, len(manual_rows), paths["workbook"], paths["html_gallery"])
    summary = build_summary(rows, audits, paths)
    write_workbook(paths["workbook"], summary, audits, rows)
    write_csv(paths["audit"], audits, AUDIT_FIELDS)
    write_csv(paths["summary"], summary, SUMMARY_FIELDS)
    (out_dir / "summary.json").write_text(
        json.dumps(
            {
                "adjudication_rows": len(rows),
                "adjudication_by_tier": dict(Counter(str(row.get("adjudication_tier", "")) for row in rows)),
                "adjudication_by_default_decision": dict(Counter(str(row.get("default_decision", "")) for row in rows)),
                "audit_failure_count": sum(1 for row in audits if row.get("status") == "fail"),
                "audit_warning_count": sum(1 for row in audits if row.get("status") == "warn"),
                "outputs": {name: output_path(path) for name, path in paths.items()},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
