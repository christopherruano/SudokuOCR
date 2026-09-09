#!/usr/bin/env python3
"""Build a static review page for first-principles assembly decisions.

This is a reporting-only helper. It reads existing assembly CSV outputs and
does not OCR, call APIs, or mutate any assembly decisions.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path
from urllib.parse import quote


ROOT = Path(__file__).resolve().parent
DEFAULT_RELEASE_DIR = ROOT / (
    "export_final/pdf_recovery_incoming_2026-06-16_first_principles_assembly_release_snapshot_"
    "overnight_priority_20260617c_20260617_085848"
)
MAX_MATRIX_ROWS = 80
MAX_MATRIX_COLS = 80


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def project_path(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return ROOT / path


def file_url(path_text: str) -> str:
    if not path_text:
        return ""
    return project_path(path_text).resolve().as_uri()


def rel_label(path_text: str) -> str:
    if not path_text:
        return ""
    path = project_path(path_text).resolve()
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def csv_matrix(path_text: str) -> dict[str, object]:
    if not path_text:
        return {"path": "", "exists": False, "rows": [], "truncated": False}
    path = project_path(path_text)
    if not path.exists():
        return {"path": path_text, "exists": False, "rows": [], "truncated": False}
    rows: list[list[str]] = []
    truncated = False
    with path.open(newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        for idx, row in enumerate(reader):
            if idx >= MAX_MATRIX_ROWS:
                truncated = True
                break
            if len(row) > MAX_MATRIX_COLS:
                row = row[:MAX_MATRIX_COLS] + ["..."]
                truncated = True
            rows.append(row)
    return {
        "path": rel_label(path_text),
        "url": file_url(path_text),
        "exists": True,
        "rows": rows,
        "truncated": truncated,
    }


def first_value(row: dict[str, str], *keys: str) -> str:
    for key in keys:
        value = row.get(key, "")
        if value:
            return value
    return ""


def summary_values(release_dir: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for row in read_csv(release_dir / "summary.csv"):
        name = row.get("name", "")
        if name and name not in out:
            out[name] = row.get("value", "")
    return out


def build_review_required(release_dir: Path, matrix_by_assembly: dict[str, str]) -> list[dict[str, object]]:
    rows = []
    for row in read_csv(release_dir / "materialized_physical_review_required.csv"):
        matrix_csv = matrix_by_assembly.get(row.get("assembly_id", ""), "")
        rows.append(
            {
                "id": row.get("assembly_id", ""),
                "queue": "materialized_review_required",
                "rank": row.get("assembly_rank", ""),
                "source": row.get("v5_source_key", ""),
                "source_pdf_id": row.get("source_pdf_id_guess", ""),
                "pages": row.get("page_pair", ""),
                "left_title": row.get("left_title", ""),
                "right_title": row.get("right_title", ""),
                "evidence": {
                    "row_pairing_mode": row.get("row_pairing_mode", ""),
                    "row_label_source_summary": row.get("row_label_source_summary", ""),
                    "column_order_evidence": row.get("column_order_evidence", ""),
                    "row_alignment_evidence": row.get("row_alignment_evidence", ""),
                    "zero_check_status": row.get("zero_check_status", ""),
                    "zero_check_failures": row.get("zero_check_failures", ""),
                    "review_flags": row.get("review_flags", ""),
                    "recommended_action": row.get("recommended_action", ""),
                },
                "links": {
                    "left_csv": file_url(row.get("left_best_csv", "")),
                    "right_csv": file_url(row.get("right_best_csv", "")),
                    "matrix_csv": file_url(matrix_csv),
                    "left_image": file_url(row.get("left_image_path", "")),
                    "right_image": file_url(row.get("right_image_path", "")),
                },
                "labels": {
                    "left_csv": rel_label(row.get("left_best_csv", "")),
                    "right_csv": rel_label(row.get("right_best_csv", "")),
                    "matrix_csv": rel_label(matrix_csv),
                    "left_image": rel_label(row.get("left_image_path", "")),
                    "right_image": rel_label(row.get("right_image_path", "")),
                },
                "matrix": csv_matrix(matrix_csv),
            }
        )
    return rows


def core_manual_ids(release_dir: Path) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for row in read_csv(release_dir / "assembly_ocr_recovery_priority.csv"):
        if row.get("core_scope") != "core_table_vii":
            continue
        if row.get("next_step_type") != "manual_visual_review_before_promotion":
            continue
        out[row.get("proposal_id", "")] = row
    return out


def build_manual_preview(release_dir: Path, core_only: bool) -> list[dict[str, object]]:
    core_rows = core_manual_ids(release_dir)
    rows = []
    for row in read_csv(release_dir / "manual_physical_preview_triage.csv"):
        proposal_id = f"preview::{row.get('assembly_id', '')}"
        recovery = core_rows.get(proposal_id)
        if core_only and recovery is None:
            continue
        rows.append(
            {
                "id": row.get("assembly_id", ""),
                "proposal_id": proposal_id,
                "queue": "core_table_vii_manual_preview" if recovery else "manual_preview_other",
                "rank": recovery.get("priority_rank", "") if recovery else row.get("triage_rank", ""),
                "source": row.get("source_key", ""),
                "source_pdf_id": row.get("source_pdf_id", ""),
                "pages": row.get("page_pair", ""),
                "left_title": row.get("left_title", ""),
                "right_title": row.get("right_title", ""),
                "evidence": {
                    "triage_bucket": row.get("triage_bucket", ""),
                    "confidence_score": row.get("confidence_score", ""),
                    "row_label_dependency_signal": row.get("row_label_dependency_signal", ""),
                    "right_page_no_row_labels_signal": row.get("right_page_no_row_labels_signal", ""),
                    "row_alignment_evidence": row.get("row_alignment_evidence", ""),
                    "row_pairing_mode": row.get("row_pairing_mode", ""),
                    "row_label_source_summary": row.get("row_label_source_summary", ""),
                    "column_order_evidence": row.get("column_order_evidence", ""),
                    "zero_check_status": row.get("zero_check_status", ""),
                    "zero_check_failures": row.get("zero_check_failures", ""),
                    "suggested_action": row.get("suggested_action", ""),
                    "promotion_condition": row.get("promotion_condition", ""),
                    "review_checklist": row.get("review_checklist", ""),
                    "reason": row.get("reason", ""),
                },
                "links": {"matrix_csv": file_url(row.get("matrix_csv", ""))},
                "labels": {"matrix_csv": rel_label(row.get("matrix_csv", ""))},
                "matrix": csv_matrix(row.get("matrix_csv", "")),
            }
        )
    return sorted(rows, key=lambda item: (int(str(item.get("rank") or "9999")), str(item.get("source")), str(item.get("pages"))))


def build_blocked_summary(release_dir: Path) -> dict[str, object]:
    recovery = read_csv(release_dir / "assembly_ocr_recovery_priority.csv")
    core = [row for row in recovery if row.get("core_scope") == "core_table_vii"]
    by_step: dict[str, int] = {}
    examples: dict[str, list[dict[str, str]]] = {}
    for row in core:
        step = row.get("next_step_type", "")
        by_step[step] = by_step.get(step, 0) + 1
        examples.setdefault(step, [])
        if len(examples[step]) < 8:
            examples[step].append(
                {
                    "priority": row.get("priority_rank", ""),
                    "source": row.get("source_key", ""),
                    "pages": row.get("page_range", ""),
                    "reason": row.get("reason", ""),
                    "flags": row.get("review_flags", ""),
                }
            )
    return {"core_by_step": by_step, "examples": examples}


def html_page(data: dict[str, object]) -> str:
    payload = json.dumps(data, ensure_ascii=False)
    title = "Assembly Review Queue"
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>
:root {{
  --ink: #1b211c;
  --muted: #617064;
  --paper: #f7f3ea;
  --card: #fffdf8;
  --line: #d7cdbb;
  --accent: #1f6f5b;
  --warn: #aa5a12;
  --bad: #8f2e24;
  --good: #26724d;
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0;
  color: var(--ink);
  background:
    radial-gradient(circle at 0 0, rgba(31,111,91,.16), transparent 32rem),
    linear-gradient(135deg, #f8f4ea 0%, #ebe0cc 100%);
  font: 15px/1.45 Georgia, "Iowan Old Style", "Palatino Linotype", serif;
}}
header {{
  position: sticky;
  top: 0;
  z-index: 10;
  padding: 18px 22px;
  background: rgba(247,243,234,.94);
  border-bottom: 1px solid var(--line);
  backdrop-filter: blur(10px);
}}
h1 {{ margin: 0 0 8px; font-size: 28px; letter-spacing: -0.02em; }}
.sub {{ color: var(--muted); max-width: 1200px; }}
.counts {{ display: flex; flex-wrap: wrap; gap: 8px; margin-top: 14px; }}
.pill {{
  border: 1px solid var(--line);
  background: var(--card);
  border-radius: 999px;
  padding: 6px 10px;
  font-size: 13px;
}}
main {{ display: grid; grid-template-columns: 340px 1fr; gap: 16px; padding: 16px; }}
aside {{
  position: sticky;
  top: 114px;
  align-self: start;
  max-height: calc(100vh - 130px);
  overflow: auto;
  border: 1px solid var(--line);
  background: rgba(255,253,248,.92);
  border-radius: 16px;
  padding: 12px;
}}
.toolbar {{ display: grid; gap: 8px; margin-bottom: 12px; }}
input, select, textarea {{
  width: 100%;
  border: 1px solid var(--line);
  background: #fffdf9;
  color: var(--ink);
  border-radius: 10px;
  padding: 8px 10px;
  font: inherit;
}}
button {{
  border: 1px solid var(--line);
  border-radius: 999px;
  padding: 7px 10px;
  background: #fffdf8;
  color: var(--ink);
  cursor: pointer;
  font: 13px/1.2 Georgia, "Iowan Old Style", serif;
}}
button:hover {{ border-color: var(--accent); color: var(--accent); }}
.queue-list {{ display: grid; gap: 8px; }}
.queue-item {{
  text-align: left;
  border: 1px solid var(--line);
  background: #fffdf8;
  border-radius: 12px;
  padding: 10px;
  cursor: pointer;
}}
.queue-item.active {{ outline: 2px solid var(--accent); }}
.queue-item .meta {{ color: var(--muted); font-size: 12px; }}
.queue-item .status {{ float: right; font-size: 12px; color: var(--accent); }}
.detail {{
  min-width: 0;
  border: 1px solid var(--line);
  background: rgba(255,253,248,.94);
  border-radius: 18px;
  overflow: hidden;
}}
.detail-head {{ padding: 18px 20px; border-bottom: 1px solid var(--line); }}
.detail-head h2 {{ margin: 0 0 8px; font-size: 24px; }}
.badge-row {{ display: flex; flex-wrap: wrap; gap: 8px; margin-top: 10px; }}
.badge {{ border-radius: 999px; padding: 4px 8px; border: 1px solid var(--line); background: #f8f0df; font-size: 12px; }}
.badge.good {{ color: var(--good); }}
.badge.warn {{ color: var(--warn); }}
.content {{ padding: 16px 20px 24px; display: grid; gap: 16px; }}
.decision {{
  display: grid;
  grid-template-columns: repeat(4, minmax(120px, 1fr));
  gap: 8px;
}}
.decision button[data-choice="promote_structure"] {{ border-color: rgba(38,114,77,.45); }}
.decision button[data-choice="keep_separate"] {{ border-color: rgba(143,46,36,.45); }}
.decision button[data-choice="needs_image_review"] {{ border-color: rgba(170,90,18,.45); }}
.decision button.selected {{ background: var(--accent); color: white; border-color: var(--accent); }}
.notes {{ min-height: 72px; }}
.grid2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }}
.scan {{
  border: 1px solid var(--line);
  border-radius: 14px;
  overflow: hidden;
  background: #f8f4eb;
}}
.scan h3, .panel h3 {{ margin: 0; padding: 10px 12px; border-bottom: 1px solid var(--line); font-size: 15px; }}
.scan img {{ width: 100%; max-height: 640px; object-fit: contain; display: block; background: #ede7da; }}
.scan a, .links a {{ word-break: break-all; color: var(--accent); }}
.scan .path {{ padding: 8px 10px; font-size: 12px; color: var(--muted); }}
.panel {{ border: 1px solid var(--line); border-radius: 14px; overflow: hidden; }}
.facts {{ display: grid; grid-template-columns: 210px 1fr; gap: 0; }}
.facts div {{ padding: 7px 10px; border-bottom: 1px solid #eee5d6; }}
.facts div:nth-child(odd) {{ color: var(--muted); background: #faf5ea; }}
.matrix-wrap {{ overflow: auto; max-height: 560px; }}
table.matrix {{ border-collapse: collapse; min-width: 100%; font-size: 12px; font-family: "Menlo", "Consolas", monospace; }}
.matrix td, .matrix th {{ border: 1px solid #d9cfbf; padding: 4px 6px; white-space: nowrap; }}
.matrix tr:first-child td {{ position: sticky; top: 0; background: #efe5d4; font-weight: 700; }}
.blocked {{ padding: 12px; display: grid; gap: 8px; }}
.blocked-row {{ border: 1px solid var(--line); border-radius: 10px; padding: 8px 10px; background: #fffdf8; }}
.small {{ font-size: 12px; color: var(--muted); }}
@media (max-width: 980px) {{
  main {{ grid-template-columns: 1fr; }}
  aside {{ position: relative; top: 0; max-height: none; }}
  .grid2, .decision {{ grid-template-columns: 1fr; }}
}}
</style>
</head>
<body>
<header>
  <h1>Assembly Review Queue</h1>
  <div class="sub" id="snapshot"></div>
  <div class="counts" id="counts"></div>
</header>
<main>
  <aside>
    <div class="toolbar">
      <select id="queueFilter">
        <option value="review_required">Materialized review-required</option>
        <option value="core_manual">Core Table VII manual previews</option>
        <option value="all_review">All reviewable</option>
        <option value="blocked">Blocked summary</option>
      </select>
      <select id="decisionFilter">
        <option value="all">All decisions</option>
        <option value="pending">Pending only</option>
        <option value="promote_structure">Promote structure</option>
        <option value="keep_separate">Keep separate</option>
        <option value="needs_image_review">Needs image review</option>
      </select>
      <input id="search" placeholder="Search source, page, title, flags">
      <button id="exportJson">Export decisions JSON</button>
      <button id="exportCsv">Export decisions CSV</button>
    </div>
    <div class="queue-list" id="queueList"></div>
  </aside>
  <section class="detail" id="detail"></section>
</main>
<script id="payload" type="application/json">{html.escape(payload)}</script>
<script>
const data = JSON.parse(document.getElementById('payload').textContent);
const storeKey = 'assembly-review-decisions:' + data.snapshot_id;
let decisions = JSON.parse(localStorage.getItem(storeKey) || '{{}}');
let selectedId = null;

function save() {{
  localStorage.setItem(storeKey, JSON.stringify(decisions));
}}
function allReviewable() {{
  return [...data.review_required, ...data.core_manual];
}}
function currentRows() {{
  const queue = document.getElementById('queueFilter').value;
  if (queue === 'review_required') return data.review_required;
  if (queue === 'core_manual') return data.core_manual;
  if (queue === 'blocked') return [];
  return allReviewable();
}}
function decisionFor(row) {{
  return decisions[rowKey(row)] || {{}};
}}
function rowKey(row) {{
  return `${{row.queue}}::${{row.id}}::${{row.source}}::${{row.pages}}`;
}}
function rowText(row) {{
  return [
    row.source, row.pages, row.left_title, row.right_title,
    Object.values(row.evidence || {{}}).join(' ')
  ].join(' ').toLowerCase();
}}
function filteredRows() {{
  let rows = currentRows();
  const term = document.getElementById('search').value.trim().toLowerCase();
  const decisionFilter = document.getElementById('decisionFilter').value;
  if (term) rows = rows.filter(row => rowText(row).includes(term));
  if (decisionFilter !== 'all') {{
    rows = rows.filter(row => (decisionFor(row).choice || 'pending') === decisionFilter);
  }}
  return rows;
}}
function renderCounts() {{
  const s = data.summary;
  const parts = [
    ['physical', s.materialized_physical_assemblies],
    ['structure-ready', s.materialized_physical_structure_ready],
    ['review-required', s.materialized_physical_review_required],
    ['core manual', data.core_manual.length],
    ['missing OCR edges', s.physical_frontier_missing_ocr_edges],
    ['missing page bridges', s.physical_frontier_missing_page_bridges],
    ['audit failures', s.release_audit_failures],
  ];
  document.getElementById('snapshot').textContent = data.snapshot_label;
  document.getElementById('counts').innerHTML = parts.map(([k,v]) => `<span class="pill"><b>${{v}}</b> ${{k}}</span>`).join('');
}}
function renderList() {{
  const q = document.getElementById('queueFilter').value;
  if (q === 'blocked') {{
    selectedId = 'blocked';
    renderBlocked();
    document.getElementById('queueList').innerHTML = '<div class="small">Blocked items are worklists, not reviewable assembly decisions yet.</div>';
    return;
  }}
  const rows = filteredRows();
  if (!selectedId || !rows.some(row => rowKey(row) === selectedId)) selectedId = rows[0] ? rowKey(rows[0]) : null;
  document.getElementById('queueList').innerHTML = rows.map(row => {{
    const key = rowKey(row);
    const d = decisionFor(row);
    return `<div class="queue-item ${{key === selectedId ? 'active' : ''}}" data-key="${{key}}">
      <span class="status">${{d.choice || 'pending'}}</span>
      <b>${{escapeHtml(row.source)}} ${{escapeHtml(row.pages)}}</b>
      <div class="meta">${{escapeHtml(row.queue)}} · rank ${{escapeHtml(row.rank || '')}}</div>
      <div class="meta">${{escapeHtml(row.left_title || '')}}</div>
    </div>`;
  }}).join('') || '<div class="small">No rows match the current filters.</div>';
  document.querySelectorAll('.queue-item').forEach(el => el.addEventListener('click', () => {{
    selectedId = el.dataset.key;
    renderList();
    renderDetail();
  }}));
  renderDetail();
}}
function findSelected() {{
  return allReviewable().find(row => rowKey(row) === selectedId);
}}
function renderDetail() {{
  const row = findSelected();
  if (!row) {{
    document.getElementById('detail').innerHTML = '<div class="content"><p>No review row selected.</p></div>';
    return;
  }}
  const key = rowKey(row);
  const d = decisionFor(row);
  const facts = Object.entries(row.evidence || {{}})
    .filter(([,v]) => String(v || '').trim())
    .map(([k,v]) => `<div>${{escapeHtml(k)}}</div><div>${{escapeHtml(String(v))}}</div>`).join('');
  const buttons = ['promote_structure','keep_separate','needs_image_review','clear'].map(choice => {{
    const label = choice === 'clear' ? 'Clear' : choice.replaceAll('_', ' ');
    return `<button data-choice="${{choice}}" class="${{d.choice === choice ? 'selected' : ''}}">${{label}}</button>`;
  }}).join('');
  document.getElementById('detail').innerHTML = `
    <div class="detail-head">
      <h2>${{escapeHtml(row.source)}} · pages ${{escapeHtml(row.pages)}}</h2>
      <div>${{escapeHtml(row.left_title || '')}}</div>
      <div class="small">${{escapeHtml(row.right_title || '')}}</div>
      <div class="badge-row">
        <span class="badge">${{escapeHtml(row.queue)}}</span>
        <span class="badge ${{String(row.evidence.zero_check_status || '').includes('pass') ? 'good' : 'warn'}}">${{escapeHtml(row.evidence.zero_check_status || 'zero unknown')}}</span>
        <span class="badge">${{escapeHtml(row.evidence.row_pairing_mode || row.evidence.triage_bucket || '')}}</span>
      </div>
    </div>
    <div class="content">
      <div class="panel">
        <h3>Decision</h3>
        <div style="padding:12px">
          <div class="decision">${{buttons}}</div>
          <textarea class="notes" id="notes" placeholder="Optional note: why promote/keep separate/needs image review">${{escapeHtml(d.note || '')}}</textarea>
        </div>
      </div>
      <div class="grid2">${{renderScan(row, 'left')}}${{renderScan(row, 'right')}}</div>
      <div class="panel">
        <h3>Evidence</h3>
        <div class="facts">${{facts}}</div>
      </div>
      <div class="panel">
        <h3>Matrix Preview <span class="small">${{escapeHtml(row.matrix.path || '')}}</span></h3>
        ${{renderMatrix(row.matrix)}}
      </div>
      <div class="links small">${{renderLinks(row)}}</div>
    </div>`;
  document.querySelectorAll('.decision button').forEach(btn => btn.addEventListener('click', () => {{
    const choice = btn.dataset.choice;
    if (choice === 'clear') delete decisions[key];
    else decisions[key] = {{...(decisions[key] || {{}}), choice, updated_at: new Date().toISOString()}};
    const notes = document.getElementById('notes');
    if (notes && decisions[key]) decisions[key].note = notes.value;
    save();
    renderList();
  }}));
  document.getElementById('notes').addEventListener('input', e => {{
    decisions[key] = {{...(decisions[key] || {{}}), note: e.target.value, updated_at: new Date().toISOString()}};
    save();
  }});
}}
function renderScan(row, side) {{
  const url = row.links?.[side + '_image'];
  const label = row.labels?.[side + '_image'];
  if (!url) return `<div class="scan"><h3>${{side}} scan</h3><div class="path">No image path in this queue row.</div></div>`;
  return `<div class="scan"><h3>${{side}} scan</h3><a href="${{url}}" target="_blank"><img src="${{url}}" loading="lazy"></a><div class="path">${{escapeHtml(label || url)}}</div></div>`;
}}
function renderMatrix(matrix) {{
  if (!matrix || !matrix.exists) return '<div class="blocked">No matrix preview available.</div>';
  const rows = matrix.rows || [];
  return `<div class="matrix-wrap"><table class="matrix">${{rows.map(r => '<tr>' + r.map(c => `<td>${{escapeHtml(String(c ?? ''))}}</td>`).join('') + '</tr>').join('')}}</table></div>
    ${{matrix.truncated ? '<div class="small" style="padding:8px 10px">Matrix preview truncated for browser performance.</div>' : ''}}`;
}}
function renderLinks(row) {{
  const links = [];
  for (const [key, url] of Object.entries(row.links || {{}})) {{
    if (!url) continue;
    links.push(`<a href="${{url}}" target="_blank">${{escapeHtml(key)}}: ${{escapeHtml(row.labels?.[key] || url)}}</a>`);
  }}
  return links.join('<br>');
}}
function renderBlocked() {{
  const b = data.blocked;
  const chunks = Object.entries(b.core_by_step || {{}}).map(([step, count]) => {{
    const examples = (b.examples[step] || []).map(ex => `<div class="blocked-row"><b>${{escapeHtml(ex.source)}} ${{escapeHtml(ex.pages)}}</b><div class="small">${{escapeHtml(ex.flags || ex.reason || '')}}</div></div>`).join('');
    return `<div class="panel"><h3>${{escapeHtml(step)}} · ${{count}}</h3><div class="blocked">${{examples}}</div></div>`;
  }}).join('');
  document.getElementById('detail').innerHTML = `<div class="detail-head"><h2>Blocked / Not Reviewable Yet</h2><div class="small">These are recovery or OCR queues. They should not be manually promoted until missing pages/artifacts exist and adjacent edges retest.</div></div><div class="content">${{chunks}}</div>`;
}}
function escapeHtml(value) {{
  return String(value ?? '').replace(/[&<>"']/g, c => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));
}}
function download(filename, text, type) {{
  const blob = new Blob([text], {{type}});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = filename;
  a.click();
  URL.revokeObjectURL(a.href);
}}
function exportRows() {{
  return allReviewable().map(row => {{
    const d = decisionFor(row);
    return {{
      key: rowKey(row),
      queue: row.queue,
      source: row.source,
      pages: row.pages,
      choice: d.choice || '',
      note: d.note || '',
      updated_at: d.updated_at || '',
      left_title: row.left_title || '',
      right_title: row.right_title || ''
    }};
  }});
}}
document.getElementById('exportJson').addEventListener('click', () => download('assembly_review_decisions.json', JSON.stringify(exportRows(), null, 2), 'application/json'));
document.getElementById('exportCsv').addEventListener('click', () => {{
  const rows = exportRows();
  const fields = Object.keys(rows[0] || {{key:''}});
  const csv = [fields.join(',')].concat(rows.map(r => fields.map(f => JSON.stringify(r[f] ?? '')).join(','))).join('\\n');
  download('assembly_review_decisions.csv', csv, 'text/csv');
}});
['queueFilter','decisionFilter','search'].forEach(id => document.getElementById(id).addEventListener('input', renderList));
renderCounts();
renderList();
</script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-dir", default=str(DEFAULT_RELEASE_DIR))
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    release_dir = Path(args.release_dir).expanduser().resolve()
    out = Path(args.out).expanduser() if args.out else release_dir / "assembly_review_page.html"
    matrix_by_assembly = {
        row.get("assembly_id", ""): row.get("matrix_csv", "")
        for row in read_csv(release_dir / "materialized_physical_matrix_files.csv")
    }
    summary = summary_values(release_dir)
    data = {
        "snapshot_id": release_dir.name,
        "snapshot_label": f"Snapshot: {release_dir.name}",
        "summary": summary,
        "review_required": build_review_required(release_dir, matrix_by_assembly),
        "core_manual": build_manual_preview(release_dir, core_only=True),
        "blocked": build_blocked_summary(release_dir),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html_page(data), encoding="utf-8")
    print(out)
    print(f"review_required={len(data['review_required'])} core_manual={len(data['core_manual'])}")


if __name__ == "__main__":
    main()
