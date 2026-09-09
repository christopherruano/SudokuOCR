#!/usr/bin/env python3
"""Local first-principles assembly review app.

This is intentionally scoped to physical assembly review. It mirrors the older
group-review workflow but uses the new adjacency-first grouping record.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import subprocess
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from flask import Flask, abort, redirect, render_template_string, request, send_file, url_for


ROOT = Path(__file__).resolve().parent
RELEASE_GLOB = "pdf_recovery_incoming_2026-06-16_first_principles_assembly_release_snapshot_*"

DECISION_JSON = "assembly_review_decisions.json"
DECISION_CSV = "assembly_physical_review_decisions.csv"
DECISION_FIELDS = [
    "review_key",
    "human_decision",
    "human_notes",
    "reviewer",
    "reviewed_at",
    "source_proposal_id",
    "grouping_id",
    "grouping_layer",
    "grouping_status",
    "core_scope",
    "source_key",
    "source_pdf_id",
    "page_range",
    "component_pages",
    "pages_consecutive",
    "approved_member_order_source_ids",
    "approved_member_order_pages",
    "approved_member_order_images",
    "approved_matrix_csv",
    "row_pairing_mode",
    "row_label_source_summary",
    "column_order_evidence",
    "zero_check_status",
    "review_flags",
    "reason",
]

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


def write_csv_atomic(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})
    tmp.replace(path)


def write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def latest_release_dir() -> Path:
    candidates = [
        path
        for path in (ROOT / "export_final").glob(RELEASE_GLOB)
        if (path / "assembly_grouping_record.csv").exists()
    ]
    if not candidates:
        raise SystemExit("No first-principles assembly release snapshot found.")
    return max(candidates, key=lambda p: (p.stat().st_mtime, p.name))


def resolve_path(value: str | Path) -> Path | None:
    text = str(value or "").strip()
    if not text:
        return None
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    try:
        return path.resolve()
    except OSError:
        return path


def path_exists(value: str | Path) -> bool:
    path = resolve_path(value)
    return bool(path and path.exists())


def proposal_short_id(source_proposal_id: str) -> str:
    return source_proposal_id.split("::", 1)[1] if "::" in source_proposal_id else source_proposal_id


def review_queue_for_record(record: dict[str, str]) -> str:
    if record.get("grouping_status") == "physical_structure_review_required":
        return "materialized_review_required"
    if (
        record.get("grouping_layer") == "manual_physical_review"
        and record.get("grouping_status") == "manual_visual_review_before_physical_promotion"
    ):
        if record.get("core_scope") == "core_table_vii":
            return "core_table_vii_manual_preview"
        return "manual_physical_preview"
    if record.get("grouping_layer") == "logical_only" and record.get("grouping_status") == "review_logical_only":
        if record.get("core_scope") == "core_table_vii":
            return "core_table_vii_logical_sibling_review"
        return "logical_sibling_review"
    return ""


def review_key(record: dict[str, str]) -> str:
    return "::".join(
        [
            review_queue_for_record(record),
            proposal_short_id(record.get("source_proposal_id", "")),
            record.get("source_key", ""),
            record.get("page_range", ""),
        ]
    )


def split_semicolon(value: str) -> list[str]:
    return [part.strip() for part in str(value or "").split(";") if part.strip()]


def split_pipe(value: str) -> list[str]:
    return [part.strip() for part in str(value or "").split("|") if part.strip()]


def int_value(value: object) -> int | None:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def load_decisions_json(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        rows = raw.get("decisions", [])
    elif isinstance(raw, list):
        rows = raw
    else:
        rows = []
    decisions: dict[str, dict[str, str]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        key = str(row.get("key") or "")
        if not key:
            continue
        decisions[key] = {str(k): str(v) for k, v in row.items()}
    return decisions


def decision_class(choice: str) -> str:
    if choice in {"promote_structure", "approve_logical"}:
        return "ok"
    if choice in {"keep_separate", "reject_logical"}:
        return "bad"
    if choice == "needs_image_review":
        return "warn"
    return "warn"


def proposal_class(record: dict[str, str]) -> str:
    status = record.get("grouping_status", "")
    if status == "physical_structure_review_required":
        return "warn"
    if record.get("grouping_layer") == "manual_physical_review":
        return "warn"
    if status in {"physical_structure_ready", "physical_structure_ready_value_review"}:
        return "ok"
    return "warn"


def is_physical_structure(record: dict[str, str]) -> bool:
    return record.get("grouping_layer") == "physical_pairwise"


def is_core_physical_structure(record: dict[str, str]) -> bool:
    return is_physical_structure(record) and record.get("core_scope") == "core_table_vii"


def matrix_html(matrix_csv: str, max_rows: int = 120, max_cols: int = 80) -> str:
    path = resolve_path(matrix_csv)
    if not path or not path.exists():
        return '<div class="muted">No matrix CSV available for this candidate.</div>'
    rows: list[list[str]] = []
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        for idx, row in enumerate(reader):
            if idx >= max_rows:
                break
            rows.append(row[:max_cols])
    if not rows:
        return '<div class="muted">Matrix CSV is empty.</div>'
    out = ["<table class=\"matrix mx\">"]
    for idx, row in enumerate(rows):
        tag = "th" if idx == 0 else "td"
        out.append("<tr>")
        for col_idx, cell in enumerate(row):
            cls = ' class="lbl"' if col_idx == 0 else ""
            out.append(f"<{tag}{cls}>{html.escape(str(cell))}</{tag}>")
        out.append("</tr>")
    out.append("</table>")
    extra = ""
    if len(rows) == max_rows:
        extra = f'<div class="muted">Preview capped at {max_rows} rows.</div>'
    return "".join(out) + extra


def candidate_context_paths(source_pdf_id: str, page: int) -> list[Path]:
    if not source_pdf_id or page < 1:
        return []
    names = [
        ROOT / "export" / "never_ocrd_images" / f"{source_pdf_id}_p{page}.png",
        Path.home() / "Downloads" / "final_tableVII" / f"{source_pdf_id}_p{page}.png",
    ]
    names.extend((Path.home() / "Downloads" / "core_money_renders").glob(f"*/{source_pdf_id}/{source_pdf_id}_p{page}.png"))
    return [path.resolve() for path in names if path.exists()]


class ReviewAppData:
    def __init__(self, release_dir: Path):
        self.release_dir = release_dir.resolve()
        self.records = read_csv(self.release_dir / "assembly_grouping_record.csv")
        self.members = read_csv(self.release_dir / "assembly_grouping_proposal_members.csv")
        self.members_by_proposal: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in self.members:
            self.members_by_proposal[row.get("proposal_id", "")].append(row)
        for rows in self.members_by_proposal.values():
            rows.sort(key=lambda r: (int_value(r.get("member_rank")) or 9999, int_value(r.get("pdf_page")) or 999999))
        self.decisions_path = self.release_dir / DECISION_JSON
        self.decision_csv_path = self.release_dir / DECISION_CSV
        self.decisions = load_decisions_json(self.decisions_path)

    def reload_decisions(self) -> None:
        self.decisions = load_decisions_json(self.decisions_path)

    def reviewable_records(self) -> list[dict[str, str]]:
        return [row for row in self.records if review_queue_for_record(row)]

    def ready_records(self) -> list[dict[str, str]]:
        return [
            row
            for row in self.records
            if is_physical_structure(row)
            and row.get("grouping_status") in {"physical_structure_ready", "physical_structure_ready_value_review"}
        ]

    def filtered_indices(self, mode: str) -> list[int]:
        mode = mode or "unreviewed"
        output: list[int] = []
        for idx, row in enumerate(self.records):
            queue = review_queue_for_record(row)
            key = review_key(row) if queue else ""
            choice = self.decisions.get(key, {}).get("choice", "")
            if mode == "core-physical" and is_core_physical_structure(row):
                output.append(idx)
            elif mode == "all-physical-structures" and is_physical_structure(row):
                output.append(idx)
            elif mode == "all-reviewable" and queue:
                output.append(idx)
            elif mode == "unreviewed" and queue and not choice:
                output.append(idx)
            elif mode == "decided" and queue and choice:
                output.append(idx)
            elif mode == "core" and queue and row.get("core_scope") == "core_table_vii":
                output.append(idx)
            elif mode == "physical" and queue and row.get("grouping_layer") in {"physical_pairwise", "manual_physical_review"}:
                output.append(idx)
            elif mode == "logical" and queue and row.get("grouping_layer") == "logical_only":
                output.append(idx)
            elif mode == "manual" and row.get("grouping_layer") == "manual_physical_review":
                output.append(idx)
            elif mode == "review-required" and row.get("grouping_status") == "physical_structure_review_required":
                output.append(idx)
            elif mode == "ready-samples" and row in self.ready_records():
                output.append(idx)
        if not output and mode not in {"unreviewed", "core-physical", "all-physical-structures"}:
            return self.filtered_indices("unreviewed")
        if not output and mode == "unreviewed":
            return [idx for idx, row in enumerate(self.records) if review_queue_for_record(row)]
        return output

    def first_index(self, mode: str) -> int:
        indices = self.filtered_indices(mode)
        return indices[0] if indices else 0

    def nav(self, idx: int, mode: str) -> tuple[int | None, int | None]:
        valid = self.filtered_indices(mode)
        if not valid or idx not in valid:
            return None, None
        pos = valid.index(idx)
        prev_idx = valid[pos - 1] if pos > 0 else None
        next_idx = valid[pos + 1] if pos + 1 < len(valid) else None
        return prev_idx, next_idx

    def next_target_before_save(self, idx: int, mode: str) -> int:
        valid = self.filtered_indices(mode)
        if not valid:
            return idx
        if idx in valid:
            pos = valid.index(idx)
            if pos + 1 < len(valid):
                return valid[pos + 1]
            if pos > 0:
                return valid[pos - 1]
        return valid[0]

    def member_images(self, row: dict[str, str]) -> dict[str, list[dict[str, Any]]]:
        members = self.members_by_proposal.get(row.get("source_proposal_id", ""), [])
        current: list[dict[str, Any]] = []
        for idx, member in enumerate(members):
            image_path = resolve_path(member.get("image_path", ""))
            current.append(
                {
                    "label": member.get("member_role") or f"Page {idx + 1}",
                    "role": "proposed",
                    "source_id": member.get("member_source_id", ""),
                    "source_json": member.get("source_json", ""),
                    "source_csv": member.get("source_csv", ""),
                    "image_path": str(image_path or member.get("image_path", "")),
                    "exists": bool(image_path and image_path.exists()),
                    "pdf_page": member.get("pdf_page", ""),
                    "title": member.get("title", ""),
                }
            )
        context: list[dict[str, Any]] = []
        pages = [int_value(page) for page in split_semicolon(row.get("component_pages", ""))]
        pages = [page for page in pages if page is not None]
        source_pdf_id = row.get("source_pdf_id", "")
        for label, page in [("Previous context", min(pages) - 1 if pages else None), ("Next context", max(pages) + 1 if pages else None)]:
            if page is None:
                continue
            for path in candidate_context_paths(source_pdf_id, page)[:1]:
                context.append(
                    {
                        "label": f"{label}: PDF page {page}",
                        "role": "context",
                        "source_id": "",
                        "source_json": "",
                        "source_csv": "",
                        "image_path": str(path),
                        "exists": True,
                        "pdf_page": str(page),
                        "title": "Not part of proposed group; context only.",
                    }
                )
        all_images = current + context
        for view_idx, image in enumerate(all_images):
            image["view_index"] = view_idx
        return {"current": current, "context": context, "all": all_images}

    def decision_rows(self) -> list[dict[str, str]]:
        self.reload_decisions()
        rows = []
        by_key = {review_key(row): row for row in self.records if review_queue_for_record(row)}
        for key, decision in sorted(self.decisions.items()):
            record = by_key.get(key, {})
            members = self.members_by_proposal.get(record.get("source_proposal_id", ""), [])
            rows.append(
                {
                    "review_key": key,
                    "human_decision": decision.get("choice", ""),
                    "human_notes": decision.get("note", ""),
                    "reviewer": "local",
                    "reviewed_at": decision.get("updated_at", ""),
                    "source_proposal_id": record.get("source_proposal_id", ""),
                    "grouping_id": record.get("grouping_id", ""),
                    "grouping_layer": record.get("grouping_layer", ""),
                    "grouping_status": record.get("grouping_status", ""),
                    "core_scope": record.get("core_scope", ""),
                    "source_key": record.get("source_key", decision.get("source", "")),
                    "source_pdf_id": record.get("source_pdf_id", ""),
                    "page_range": record.get("page_range", decision.get("pages", "")),
                    "component_pages": record.get("component_pages", ""),
                    "pages_consecutive": record.get("pages_consecutive", ""),
                    "approved_member_order_source_ids": "|".join(member.get("member_source_id", "") for member in members),
                    "approved_member_order_pages": "|".join(member.get("pdf_page", "") for member in members),
                    "approved_member_order_images": "|".join(member.get("image_path", "") for member in members),
                    "approved_matrix_csv": record.get("matrix_csv", ""),
                    "row_pairing_mode": record.get("row_pairing_mode", ""),
                    "row_label_source_summary": record.get("row_label_source_summary", ""),
                    "column_order_evidence": record.get("column_order_evidence", ""),
                    "zero_check_status": record.get("zero_check_status", ""),
                    "review_flags": record.get("review_flags", ""),
                    "reason": record.get("reason", ""),
                }
            )
        return rows

    def save_decision(self, row: dict[str, str], choice: str, note: str) -> None:
        if choice not in VALID_CHOICES:
            raise ValueError(f"unknown decision choice: {choice}")
        key = review_key(row)
        if not key.strip(":"):
            raise ValueError("row is not reviewable")
        now = datetime.now().isoformat(timespec="seconds")
        if choice:
            self.decisions[key] = {
                "key": key,
                "choice": choice,
                "queue": review_queue_for_record(row),
                "id": proposal_short_id(row.get("source_proposal_id", "")),
                "source": row.get("source_key", ""),
                "pages": row.get("page_range", ""),
                "note": note,
                "updated_at": now,
            }
        else:
            self.decisions.pop(key, None)
        write_json_atomic(self.decisions_path, {"decisions": list(self.decisions.values())})
        write_csv_atomic(self.decision_csv_path, self.decision_rows(), DECISION_FIELDS)
        self.run_overlay()

    def run_overlay(self) -> str:
        cmd = [
            "python3",
            str(ROOT / "build_assembly_manual_decision_overlay.py"),
            "--release-dir",
            str(self.release_dir),
            "--decisions",
            str(self.decisions_path),
        ]
        result = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=False)
        return (result.stdout + result.stderr).strip()

    def counts(self) -> dict[str, int]:
        self.reload_decisions()
        reviewable = self.reviewable_records()
        return {
            "reviewable": len(reviewable),
            "unreviewed": sum(1 for row in reviewable if not self.decisions.get(review_key(row), {}).get("choice", "")),
            "decided": sum(1 for row in reviewable if self.decisions.get(review_key(row), {}).get("choice", "")),
            "core": sum(1 for row in reviewable if row.get("core_scope") == "core_table_vii"),
            "core_physical": sum(1 for row in self.records if is_core_physical_structure(row)),
            "all_physical_structures": sum(1 for row in self.records if is_physical_structure(row)),
            "manual": sum(1 for row in reviewable if row.get("grouping_layer") == "manual_physical_review"),
            "physical": sum(1 for row in reviewable if row.get("grouping_layer") in {"physical_pairwise", "manual_physical_review"}),
            "logical": sum(1 for row in reviewable if row.get("grouping_layer") == "logical_only"),
            "review_required": sum(1 for row in reviewable if row.get("grouping_status") == "physical_structure_review_required"),
            "ready_samples": len(self.ready_records()),
        }

    def decision_counts(self) -> dict[str, int]:
        self.reload_decisions()
        return dict(Counter(decision.get("choice", "") for decision in self.decisions.values()))


APP_HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Assembly Review</title>
  <style>
    :root {
      --ink:#191715;
      --muted:#706a61;
      --bg:#ece7dc;
      --paper:#fffaf0;
      --line:#d6cbbb;
      --accent:#263d35;
      --accent2:#b0612d;
      --bad:#8f2828;
      --warn:#8a650e;
      --ok:#27603a;
    }
    * { box-sizing:border-box; }
    body { margin:0; padding-bottom:176px; color:var(--ink); background:radial-gradient(circle at top left, #fff7df 0, var(--bg) 42%, #e6ddd0 100%); font-family:"Iowan Old Style", Georgia, serif; }
    header { position:sticky; top:0; z-index:4; background:rgba(255,250,240,.97); backdrop-filter:blur(8px); border-bottom:1px solid var(--line); padding:12px 16px; display:flex; justify-content:space-between; gap:12px; align-items:flex-start; }
    h1 { font-size:20px; margin:0 0 4px 0; line-height:1.2; }
    h2 { font-size:17px; margin:0; }
    h3 { font-size:15px; margin:0 0 8px 0; }
    main { display:grid; grid-template-columns:minmax(390px,34vw) minmax(720px,1fr); gap:14px; padding:14px; align-items:start; }
    .card { background:var(--paper); border:1px solid var(--line); border-radius:16px; box-shadow:0 10px 28px rgba(49,34,16,.07); overflow:hidden; margin-bottom:14px; }
    .pad { padding:14px; }
    .muted { color:var(--muted); font-size:12px; }
    .mono { font-family:Menlo, Consolas, monospace; }
    .toolbar { display:flex; gap:8px; flex-wrap:wrap; align-items:center; justify-content:flex-end; }
    .btn, button { appearance:none; border:0; border-radius:999px; padding:9px 13px; background:var(--accent); color:white; text-decoration:none; font-weight:700; cursor:pointer; font-family:inherit; }
    .btn.secondary, button.secondary { background:#d8c7ac; color:var(--ink); }
    .btn.ghost { background:transparent; color:var(--ink); border:1px solid var(--line); }
    .btn.bad, button.bad { background:var(--bad); }
    .btn.warn, button.warn { background:var(--accent2); }
    .pill { display:inline-block; border:1px solid var(--line); border-radius:999px; padding:4px 9px; background:#f5eddf; margin:2px; font-size:12px; }
    .pill.ok { color:var(--ok); border-color:#9cbea4; background:#eef8ef; }
    .pill.warn { color:var(--warn); border-color:#d8bf78; background:#fff6d8; }
    .pill.bad { color:var(--bad); border-color:#dca2a2; background:#fff0ef; }
    .pages { position:sticky; top:86px; max-height:calc(100vh - 112px); overflow:auto; padding-right:2px; }
    .image-card { background:#26231f; color:#f7eedf; border-radius:14px; padding:10px; margin-bottom:12px; border:1px solid #514a42; }
    .image-card.proposed { background:#202b27; border-color:#536b5f; }
    .image-card.context { background:#382d20; border-color:#7d6344; }
    .image-card img { width:100%; display:block; border:1px solid #514a42; background:white; }
    .caption { margin-top:8px; font-size:12px; color:#dfd1bd; overflow-wrap:anywhere; }
    .meta { display:grid; grid-template-columns:178px minmax(0,1fr); gap:7px 12px; font-size:13px; }
    .callout { border-left:4px solid var(--accent2); background:#fff5df; padding:11px 13px; line-height:1.35; overflow-wrap:anywhere; }
    .callout.ok { border-color:var(--ok); background:#eef8ef; }
    .callout.bad { border-color:var(--bad); background:#fff0ef; }
    pre { white-space:pre-wrap; word-break:break-word; overflow:auto; max-height:230px; background:#fffdf8; border:1px solid var(--line); border-radius:12px; padding:12px; font:12px/1.4 Menlo, Consolas, monospace; }
    textarea { width:100%; min-height:98px; resize:vertical; border:1px solid var(--line); border-radius:12px; padding:10px; background:#fffdf8; font:14px/1.35 Menlo, Consolas, monospace; }
    .table-wrap { overflow:auto; max-height:72vh; background:#fff; border-top:1px solid #e2e4e8; }
    table.matrix { border-collapse:collapse; width:max-content; min-width:100%; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; font-size:11px; line-height:1.25; }
    table.matrix th { position:sticky; top:0; z-index:1; background:#eef1f7; color:#1f3864; font-size:10px; font-weight:700; }
    table.matrix th, table.matrix td { border:1px solid #e3e3e3; padding:3px 5px; white-space:nowrap; text-align:right; }
    table.matrix .lbl, table.matrix th:first-child, table.matrix td:first-child { position:sticky; left:0; text-align:left; background:#fff; z-index:2; max-width:190px; white-space:normal; }
    table.matrix th:first-child { z-index:3; background:#eef1f7; }
    .decision-bar { position:fixed; left:0; right:0; bottom:0; z-index:5; background:rgba(255,250,240,.98); backdrop-filter:blur(10px); border-top:1px solid var(--line); padding:10px 14px; box-shadow:0 -10px 30px rgba(49,34,16,.12); }
    .decision-form { display:grid; grid-template-columns:minmax(420px,1fr) minmax(620px,1.35fr); gap:12px; align-items:start; }
    .quick-actions { display:grid; grid-template-columns:repeat(4,minmax(130px,1fr)); gap:8px; }
    .quick-actions button { border-radius:14px; min-height:58px; font-size:15px; }
    .quick-actions small { display:block; opacity:.82; font-weight:400; margin-top:3px; }
    @media (max-width:1120px) {
      body { padding-bottom:270px; }
      header { display:block; }
      main, .decision-form { grid-template-columns:1fr; }
      .pages { position:relative; top:0; max-height:none; }
    }
  </style>
</head>
<body>
  <header>
    <div>
      <div class="muted">First-principles assembly review | {{ position }} / {{ filtered_total }} | filter={{ filter_mode }}</div>
      <h1>{{ row.source_key }} | pages {{ row.page_range }} | {{ row.core_scope }}</h1>
      <div class="mono muted">{{ row.grouping_id }} · {{ row.source_proposal_id }}</div>
    </div>
    <div class="toolbar">
      {% if prev_idx is not none %}<a class="btn secondary" data-prev href="{{ url_for('review_page', idx=prev_idx, filter=filter_mode) }}">Prev</a>{% endif %}
      {% if next_idx is not none %}<a class="btn secondary" data-next href="{{ url_for('review_page', idx=next_idx, filter=filter_mode) }}">Next</a>{% endif %}
      {% for mode,label,count in filters %}
        <a class="btn ghost" href="{{ url_for('review_page', idx=first_index(mode), filter=mode) }}">{{ label }} ({{ count }})</a>
      {% endfor %}
    </div>
  </header>

  <main>
    <aside class="pages">
      <div class="card">
        <div class="pad">
          <h2>{{ review_mode_label }}</h2>
          <div class="muted">{{ review_instruction }}</div>
        </div>
      </div>
      {% for image in member_images %}
        <div class="image-card proposed">
          {% if image.exists %}
            <a href="{{ url_for('image_file', idx=index, image_idx=image.view_index) }}" target="_blank">
              <img src="{{ url_for('image_file', idx=index, image_idx=image.view_index) }}">
            </a>
          {% else %}
            <div class="muted">Image not found locally.</div>
          {% endif %}
          <div class="caption">
            <strong>{{ image.label }} · PDF page {{ image.pdf_page }}</strong><br>
            {{ image.title }}<br>
            {{ image.image_path }}<br>
            {{ image.source_csv }}
          </div>
        </div>
      {% endfor %}

      {% if context_images %}
        <div class="card"><div class="pad"><h3>Neighbor Context, Not Included</h3></div></div>
        {% for image in context_images %}
          <div class="image-card context">
            <a href="{{ url_for('image_file', idx=index, image_idx=image.view_index) }}" target="_blank">
              <img src="{{ url_for('image_file', idx=index, image_idx=image.view_index) }}">
            </a>
            <div class="caption"><strong>{{ image.label }}</strong><br>{{ image.image_path }}</div>
          </div>
        {% endfor %}
      {% endif %}
    </aside>

    <section>
      <div class="card">
        <div class="pad">
          <h2>Assembly Proposal</h2>
          <div>
            <span class="pill {{ proposal_class }}">{{ row.grouping_status }}</span>
            <span class="pill">{{ row.grouping_layer }}</span>
            <span class="pill">{{ row.pages_consecutive }} consecutive</span>
            <span class="pill {{ decision_class }}">{{ decision.choice or "unreviewed" }}</span>
          </div>
        </div>
        <div class="pad meta">
          <strong>Source PDF</strong><span>{{ row.source_key }} / {{ row.source_pdf_id }}</span>
          <strong>Component pages</strong><span>{{ row.component_pages }}</span>
          <strong>Queue</strong><span>{{ queue }}</span>
          <strong>Decision mode</strong><span>{{ review_mode_label }}</span>
          <strong>Row labels</strong><span>{{ row.row_label_source_summary }}</span>
          <strong>Pairing mode</strong><span>{{ row.row_pairing_mode }}</span>
          <strong>Right no labels</strong><span>{{ row.right_page_no_row_labels_signal }}</span>
          <strong>Pair direction</strong><span>{{ row.pair_direction_signal }}{% if row.pair_direction_reason %} | {{ row.pair_direction_reason }}{% endif %}</span>
          <strong>Column evidence</strong><span>{{ row.column_order_evidence }}</span>
          <strong>Zero checks</strong><span>{{ row.zero_check_status }}</span>
          <strong>Flags</strong><span>{{ row.review_flags }}</span>
        </div>
        <div class="pad">
          <div class="callout {{ proposal_class }}">
            <strong>Reason:</strong> {{ row.reason }}<br>
            <strong>Promotion condition:</strong> {{ row.promotion_condition }}<br>
            <strong>Title summary:</strong> {{ row.title_summary }}
          </div>
        </div>
      </div>

      <div class="card matrix-card">
        <div class="pad">
          <h2>Physical Table View</h2>
          <div class="muted mono">{{ row.matrix_csv }}</div>
        </div>
        <div class="table-wrap">{{ matrix_html|safe }}</div>
      </div>

      <div class="card">
        <div class="pad">
          <h2>Decision Ledger</h2>
          <div class="meta">
            <strong>Decision file</strong><span class="mono">{{ decision_path }}</span>
            <strong>CSV export</strong><span class="mono">{{ decision_csv_path }}</span>
            <strong>Counts</strong><span>{{ counts }}</span>
            <strong>By decision</strong><span>{{ decision_counts }}</span>
          </div>
        </div>
      </div>
    </section>
  </main>

  <form class="decision-bar" method="post" action="{{ url_for('save_review_page', idx=index, filter=filter_mode) }}">
    <input type="hidden" name="quick_decision" value="">
    <div class="decision-form">
      <div>
        <label><strong>Human notes / corrections</strong></label>
        <textarea name="human_notes" placeholder="If not approved, write what breaks: title switch, row-label mismatch, wrong column order, missing page, split point, logical sibling mismatch, etc.">{{ decision.note }}</textarea>
      </div>
      <div>
        <label><strong>Decision</strong></label>
        <div class="quick-actions">
          {% if is_logical %}
            <button type="button" onclick="quickSubmit('approve_logical')">Approve logical only<small>1</small></button>
            <button class="bad" type="button" onclick="quickSubmit('reject_logical')">Reject logical set<small>2</small></button>
            <button class="warn" type="button" onclick="quickSubmit('needs_image_review')">Needs review / split<small>3</small></button>
            <button class="secondary" type="button" onclick="quickSubmit('')">Clear decision<small>0</small></button>
          {% else %}
            <button type="button" onclick="quickSubmit('promote_structure')">Approve physical join<small>1</small></button>
            <button class="bad" type="button" onclick="quickSubmit('keep_separate')">Keep separate<small>2</small></button>
            <button class="warn" type="button" onclick="quickSubmit('needs_image_review')">Needs review / split<small>3</small></button>
            <button class="secondary" type="button" onclick="quickSubmit('')">Clear decision<small>0</small></button>
          {% endif %}
        </div>
      </div>
    </div>
  </form>
  <script>
    function quickSubmit(decision) {
      const form = document.querySelector('form');
      const quick = document.querySelector('input[name="quick_decision"]');
      if (quick) quick.value = decision;
      if (form) form.submit();
    }
    document.addEventListener('keydown', function(evt) {
      if (evt.target && ['TEXTAREA','INPUT','SELECT'].includes(evt.target.tagName)) return;
      if (evt.key === 'j') { const el=document.querySelector('a[data-next]'); if (el) window.location=el.href; }
      if (evt.key === 'k') { const el=document.querySelector('a[data-prev]'); if (el) window.location=el.href; }
      if (evt.key === '1') quickSubmit('{{ "approve_logical" if is_logical else "promote_structure" }}');
      if (evt.key === '2') quickSubmit('{{ "reject_logical" if is_logical else "keep_separate" }}');
      if (evt.key === '3') quickSubmit('needs_image_review');
      if (evt.key === '0') quickSubmit('');
    });
  </script>
</body>
</html>
"""


FINAL_HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Physical Assembly Matrix</title>
  <style>
    body{font:13px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;margin:0;background:#f4f5f7;color:#1a1a1a}
    header{position:sticky;top:0;background:#1f3864;color:#fff;padding:12px 18px;z-index:5;box-shadow:0 2px 10px rgba(0,0,0,.15)}
    header h1{margin:0;font-size:18px;line-height:1.25}.sub{opacity:.9;font-size:12px;margin-top:3px}
    .bar{display:flex;gap:8px;flex-wrap:wrap;margin-top:10px}.btn{background:#fff;color:#1f3864;border:1px solid rgba(255,255,255,.5);padding:6px 10px;border-radius:6px;text-decoration:none;font-weight:700}
    .btn.ghost{background:transparent;color:#fff}.btn.warn{background:#fff8e1;color:#7a5b00}
    .t{background:#fff;margin:10px 18px;border:1px solid #e2e4e8;border-radius:8px;overflow:hidden}
    .t.vii{border-left:4px solid #1f3864}.th{display:flex;gap:10px;align-items:center;padding:9px 12px;background:#eef1f7;flex-wrap:wrap}
    .th b{font-size:14px}.prov{color:#555}.b{padding:1px 8px;border-radius:10px;font-size:11px;font-weight:700}
    .b.ok{background:#c6efce;color:#0a5d1e}.b.warn{background:#ffeb9c;color:#7a5b00}.b.bad{background:#ffc7ce;color:#9c0017}
    .vt{background:#1f3864;color:#fff;padding:0 6px;border-radius:3px;font-size:10px;font-weight:700}.dim{color:#777}
    .body{display:grid;grid-template-columns:minmax(640px,1fr) minmax(320px,38vw);gap:12px;padding:10px;align-items:start;overflow:auto}
    .matrix-wrap{overflow:auto;max-height:calc(100vh - 205px);border:1px solid #e2e4e8;border-radius:6px;background:#fff}
    .mx{border-collapse:collapse;font-size:10px;width:max-content;min-width:100%}.mx td,.mx th{border:1px solid #e3e3e3;padding:2px 4px;text-align:right;white-space:nowrap}
    .mx th{position:sticky;top:0;background:#eef1f7;font-size:9px;z-index:2}.mx td:first-child,.mx th:first-child{position:sticky;left:0;text-align:left;background:#fff;z-index:3;max-width:180px;white-space:normal}
    .mx th:first-child{background:#eef1f7;z-index:4}.secname{font-weight:700;font-size:11px;margin:2px 0;color:#1f3864}
    .scans{display:flex;gap:8px;overflow:auto;align-items:flex-start}.scan{flex:0 0 300px;background:#fff;border:1px solid #ddd;border-radius:6px;padding:6px}
    .scan img{width:100%;display:block;border:1px solid #ccc;border-radius:4px}.cap{font-size:11px;color:#555;margin-top:5px;overflow-wrap:anywhere}
    .meta{display:grid;grid-template-columns:150px minmax(0,1fr);gap:5px 10px;padding:10px 12px;border-top:1px solid #e2e4e8;font-size:12px}.meta b{color:#333}
    .note{margin:8px 18px;background:#fff8e1;border:1px solid #f0e0a0;border-radius:8px;padding:8px 12px;color:#634900}
    @media(max-width:1050px){.body{grid-template-columns:1fr}.matrix-wrap{max-height:none}.scan{flex-basis:260px}}
  </style>
</head>
<body>
  <header>
    <div class="sub">Research partner view | physical assemblies only | {{ position }} / {{ filtered_total }} | filter={{ filter_mode }}</div>
    <h1>{{ row.source_key }} | PDF pages {{ row.page_range }}</h1>
    <div class="sub">{{ row.grouping_id }} · {{ row.source_proposal_id }}</div>
    <div class="bar">
      {% if prev_idx is not none %}<a class="btn" href="{{ url_for('final_page', idx=prev_idx, filter=filter_mode) }}">Prev</a>{% endif %}
      {% if next_idx is not none %}<a class="btn" href="{{ url_for('final_page', idx=next_idx, filter=filter_mode) }}">Next</a>{% endif %}
      <a class="btn ghost" href="{{ url_for('final_page', idx=first_index('core-physical'), filter='core-physical') }}">Core VII physical ({{ counts.core_physical }})</a>
      <a class="btn ghost" href="{{ url_for('final_page', idx=first_index('all-physical-structures'), filter='all-physical-structures') }}">All physical ({{ counts.all_physical_structures }})</a>
      <a class="btn warn" href="{{ url_for('review_page', idx=index, filter=review_filter) }}">Review/evidence view</a>
      <a class="btn ghost" href="{{ url_for('review_page', idx=first_index('unreviewed'), filter='unreviewed') }}">Review queue ({{ counts.unreviewed }})</a>
    </div>
  </header>

  {% if row.core_scope != 'core_table_vii' %}
    <div class="note">This is not marked core Table VII. It is included only because the current filter is broader than core physical assemblies.</div>
  {% endif %}

  <section class="t vii">
    <div class="th">
      <span class="vt">VII</span>
      <b>{{ row.title_summary }}</b>
      <span class="prov">{{ row.source_pdf_id }} · pages {{ row.component_pages }}</span>
      <span class="b {{ proposal_class }}">{{ row.grouping_status }}</span>
      <span class="b {{ zero_class }}">{{ row.zero_check_status }}</span>
    </div>
    <div class="body">
      <div>
        <div class="matrix-wrap">{{ matrix_html|safe }}</div>
      </div>
      <div>
        <div class="scans">
          {% for image in member_images %}
            <div class="scan">
              {% if image.exists %}
                <a href="{{ url_for('image_file', idx=index, image_idx=image.view_index) }}" target="_blank">
                  <img src="{{ url_for('image_file', idx=index, image_idx=image.view_index) }}">
                </a>
              {% else %}
                <div class="dim">Image not found locally.</div>
              {% endif %}
              <div class="cap"><b>{{ image.label }} · PDF page {{ image.pdf_page }}</b><br>{{ image.title }}<br>{{ image.image_path }}</div>
            </div>
          {% endfor %}
        </div>
      </div>
    </div>
    <div class="meta">
      <b>Source PDF</b><span>{{ row.source_key }} / {{ row.source_pdf_id }}</span>
      <b>Physical rule</b><span>{{ row.pages_consecutive }} consecutive; {{ row.reason }}</span>
      <b>Row labels</b><span>{{ row.row_label_source_summary }} | {{ row.row_pairing_mode }}</span>
      <b>Columns</b><span>{{ row.column_order_evidence }}</span>
      <b>Flags</b><span>{{ row.review_flags or 'none' }}</span>
      <b>Matrix CSV</b><span class="dim">{{ row.matrix_csv }}</span>
    </div>
  </section>
</body>
</html>
"""


def create_app(data: ReviewAppData) -> Flask:
    app = Flask(__name__)

    filters = [
        ("unreviewed", "Unreviewed", lambda c: c["unreviewed"]),
        ("core", "Core VII", lambda c: c["core"]),
        ("physical", "Physical", lambda c: c["physical"]),
        ("logical", "Logical", lambda c: c["logical"]),
        ("manual", "Manual", lambda c: c["manual"]),
        ("review-required", "Review required", lambda c: c["review_required"]),
        ("decided", "Decided", lambda c: c["decided"]),
        ("all-reviewable", "All reviewable", lambda c: c["reviewable"]),
        ("ready-samples", "Ready samples", lambda c: c["ready_samples"]),
    ]

    @app.route("/")
    def home():
        mode = request.args.get("filter", "core")
        return redirect(url_for("review_page", idx=data.first_index(mode), filter=mode))

    @app.route("/final")
    def final_home():
        mode = request.args.get("filter", "core-physical")
        return redirect(url_for("final_page", idx=data.first_index(mode), filter=mode))

    @app.route("/final/<int:idx>")
    def final_page(idx: int):
        data.reload_decisions()
        if idx < 0 or idx >= len(data.records):
            abort(404)
        filter_mode = request.args.get("filter", "core-physical")
        valid = data.filtered_indices(filter_mode)
        if valid and idx not in valid:
            idx = valid[0]
        row = data.records[idx]
        images = data.member_images(row)
        prev_idx, next_idx = data.nav(idx, filter_mode)
        counts = data.counts()
        position = (valid.index(idx) + 1) if idx in valid else 1
        zero_status = row.get("zero_check_status", "")
        zero_class = "ok" if "pass" in zero_status else ("bad" if "fail" in zero_status else "warn")
        review_filter = "review-required" if review_queue_for_record(row) else "ready-samples"
        return render_template_string(
            FINAL_HTML,
            row=row,
            index=idx,
            position=position,
            filtered_total=len(valid),
            filter_mode=filter_mode,
            first_index=data.first_index,
            prev_idx=prev_idx,
            next_idx=next_idx,
            counts=counts,
            member_images=images["current"],
            matrix_html=matrix_html(row.get("matrix_csv", ""), max_rows=600, max_cols=160),
            proposal_class=proposal_class(row),
            zero_class=zero_class,
            review_filter=review_filter,
        )

    @app.route("/review/<int:idx>")
    def review_page(idx: int):
        data.reload_decisions()
        if idx < 0 or idx >= len(data.records):
            abort(404)
        filter_mode = request.args.get("filter", "unreviewed")
        valid = data.filtered_indices(filter_mode)
        if valid and idx not in valid:
            idx = valid[0]
        row = data.records[idx]
        queue = review_queue_for_record(row)
        key = review_key(row) if queue else ""
        decision = data.decisions.get(key, {})
        is_logical = row.get("grouping_layer") == "logical_only"
        review_mode_label = "Logical Sibling Collection" if is_logical else "Proposed Physical Order"
        review_instruction = (
            "Approve only as a logical/navigation grouping. These pages must never be physically merged; they should share source, table number/topic, and bounded sibling structure."
            if is_logical
            else "Approve only if these are consecutive PDF pages and the right/next page truly continues the same table structure."
        )
        images = data.member_images(row)
        prev_idx, next_idx = data.nav(idx, filter_mode)
        counts = data.counts()
        decorated_filters = [(mode, label, get_count(counts)) for mode, label, get_count in filters]
        position = (valid.index(idx) + 1) if idx in valid else 1
        return render_template_string(
            APP_HTML,
            row=row,
            index=idx,
            position=position,
            filtered_total=len(valid),
            filter_mode=filter_mode,
            filters=decorated_filters,
            first_index=data.first_index,
            prev_idx=prev_idx,
            next_idx=next_idx,
            queue=queue,
            decision=decision,
            decision_class=decision_class(decision.get("choice", "")),
            proposal_class=proposal_class(row),
            is_logical=is_logical,
            review_mode_label=review_mode_label,
            review_instruction=review_instruction,
            member_images=images["current"],
            context_images=images["context"],
            matrix_html=matrix_html(row.get("matrix_csv", "")),
            counts=counts,
            decision_counts=data.decision_counts(),
            decision_path=str(data.decisions_path),
            decision_csv_path=str(data.decision_csv_path),
        )

    @app.post("/review/<int:idx>/save")
    def save_review_page(idx: int):
        if idx < 0 or idx >= len(data.records):
            abort(404)
        row = data.records[idx]
        if not review_queue_for_record(row):
            abort(400, "This row is not a physical review candidate.")
        filter_mode = request.args.get("filter", "unreviewed")
        target = data.next_target_before_save(idx, filter_mode)
        choice = request.form.get("quick_decision") if "quick_decision" in request.form else request.form.get("human_decision", "")
        if choice is None:
            choice = ""
        note = request.form.get("human_notes", "")
        try:
            data.save_decision(row, choice, note)
        except ValueError as exc:
            abort(400, str(exc))
        return redirect(url_for("review_page", idx=target, filter=filter_mode))

    @app.route("/image/<int:idx>/<int:image_idx>")
    def image_file(idx: int, image_idx: int):
        if idx < 0 or idx >= len(data.records):
            abort(404)
        images = data.member_images(data.records[idx])["all"]
        if image_idx < 0 or image_idx >= len(images):
            abort(404)
        path = resolve_path(images[image_idx].get("image_path", ""))
        if not path or not path.exists():
            abort(404)
        return send_file(path)

    @app.route("/decisions.csv")
    def decisions_csv():
        if not data.decision_csv_path.exists():
            write_csv_atomic(data.decision_csv_path, data.decision_rows(), DECISION_FIELDS)
        return send_file(data.decision_csv_path, as_attachment=False)

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-dir", default="", help="Release snapshot dir. Defaults to latest first-principles release.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5011)
    args = parser.parse_args()

    release_dir = Path(args.release_dir).expanduser() if args.release_dir else latest_release_dir()
    data = ReviewAppData(release_dir)
    data.run_overlay()
    app = create_app(data)
    print(f"Release snapshot: {data.release_dir}")
    print(f"Decision JSON: {data.decisions_path}")
    print(f"Decision CSV: {data.decision_csv_path}")
    print(f"Open http://{args.host}:{args.port}/")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
