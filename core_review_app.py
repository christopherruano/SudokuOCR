"""Local web app for reviewing core_review candidates at full image resolution.

This app is tailored to the local bundle workflow:
  - decision CSV in export/local_research_bundle/final_core_prep
  - synced review assets in export/local_research_bundle/final_core_prep/review_assets
  - high-resolution original image on the left
  - OCR preview / parsed sections / raw JSON on the right
  - decisions saved directly back to core_review_decisions.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from html import escape
from pathlib import Path

from flask import Flask, abort, redirect, render_template_string, request, send_file, url_for


DEFAULT_DECISIONS_CSV = "export/local_research_bundle/final_core_prep/core_review_decisions.csv"
DEFAULT_LOCAL_ASSET_ROOT = "export/local_research_bundle/final_core_prep/review_assets"
DEFAULT_REMOTE_ROOT = "/n/home11/cruano/census_toc"

CONTINUATION_PAT = re.compile(r"\b(continued|contd|concld|concluded)\b", re.I)
PART_PAT = re.compile(r"\bpart\s+[ivxlcdm0-9]+\b", re.I)


APP_HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Core Review</title>
  <style>
    :root {
      --bg: #f3efe4;
      --panel: #fffaf0;
      --ink: #1d1b18;
      --muted: #6d665d;
      --line: #d8cdb9;
      --accent: #8c3d1f;
      --ok: #2f6b3d;
      --warn: #946516;
      --bad: #8c2b2b;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Georgia, "Iowan Old Style", serif;
      background: linear-gradient(180deg, #efe7d7 0%, var(--bg) 100%);
      color: var(--ink);
    }
    header {
      padding: 14px 18px;
      border-bottom: 1px solid var(--line);
      display: flex;
      justify-content: space-between;
      gap: 20px;
      align-items: baseline;
    }
    main {
      padding: 16px;
      display: grid;
      grid-template-columns: 1.25fr 1fr;
      gap: 16px;
      align-items: start;
    }
    .card {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 14px;
      box-shadow: 0 8px 24px rgba(40, 28, 10, 0.06);
    }
    .meta {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 8px 16px;
      margin-bottom: 10px;
      font-size: 14px;
    }
    .muted { color: var(--muted); }
    .pill {
      display: inline-block;
      border: 1px solid var(--line);
      background: #f7f1e6;
      border-radius: 999px;
      padding: 4px 8px;
      font-size: 12px;
      margin-right: 6px;
    }
    .ok { color: var(--ok); }
    .warn { color: var(--warn); }
    .bad { color: var(--bad); }
    .toolbar, .nav, .savebar {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      align-items: center;
      margin-bottom: 10px;
    }
    .btn, button {
      border: 0;
      border-radius: 999px;
      padding: 10px 14px;
      background: var(--accent);
      color: white;
      cursor: pointer;
      text-decoration: none;
      font-weight: 600;
    }
    .btn.secondary, button.secondary {
      background: #d8ccb4;
      color: var(--ink);
    }
    .btn.ghost {
      background: transparent;
      color: var(--ink);
      border: 1px solid var(--line);
    }
    .viewer {
      border: 1px solid var(--line);
      border-radius: 12px;
      background: white;
      height: calc(100vh - 220px);
      min-height: 620px;
      overflow: auto;
      padding: 6px;
    }
    .viewer img {
      display: block;
      width: auto;
      max-width: none;
      height: auto;
      transform-origin: top left;
    }
    .stack {
      display: grid;
      gap: 14px;
    }
    .field {
      display: grid;
      gap: 6px;
      margin-bottom: 10px;
    }
    select, textarea, input {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 10px;
      background: #fffdf8;
      font: 14px/1.35 Menlo, Consolas, monospace;
    }
    textarea { min-height: 120px; }
    .ocr-preview, pre {
      white-space: pre-wrap;
      word-break: break-word;
      background: #fffdf8;
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 12px;
      overflow: auto;
      max-height: 320px;
      font: 13px/1.4 Menlo, Consolas, monospace;
    }
    .table-wrap {
      overflow: auto;
      border: 1px solid var(--line);
      border-radius: 10px;
      background: #fffdf8;
    }
    table.data {
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }
    table.data th, table.data td {
      border: 1px solid var(--line);
      padding: 6px 8px;
      vertical-align: top;
      text-align: left;
    }
    table.data th {
      background: #f1e5cf;
      position: sticky;
      top: 0;
    }
    details > summary {
      cursor: pointer;
      font-weight: 600;
      margin-bottom: 8px;
    }
    .filterbar {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      align-items: center;
    }
    @media (max-width: 1200px) {
      main { grid-template-columns: 1fr; }
      .viewer { height: 68vh; min-height: 440px; }
    }
  </style>
</head>
<body>
  <header>
    <div>
      <div class="muted">Core Review App</div>
      <h2 style="margin:4px 0 0 0;">{{ row.province }} {{ row.year }} | {{ row.title_from_payload or row.title }}</h2>
    </div>
    <div class="muted">Row {{ index + 1 }} / {{ total }}{% if filter_mode %} | filter={{ filter_mode }}{% endif %}</div>
  </header>

  <main>
    <section class="card">
      <div class="meta">
        <div><strong>Suggested</strong><br><span class="pill">{{ row.suggested_decision or 'hold' }}</span></div>
        <div><strong>Current</strong><br>
          {% set d = row.review_decision or '(blank)' %}
          <span class="pill {{ 'ok' if d == 'promote' else 'bad' if d == 'reject' else 'warn' }}">{{ d }}</span>
        </div>
        <div><strong>Family</strong><br><span class="muted">{{ row.rescue_family }}</span></div>
        <div><strong>Score</strong><br><span class="muted">{{ row.rescue_score }}</span></div>
      </div>

      <div class="card" style="margin-bottom:10px;padding:10px 12px;">
        <strong>Likely multi-page:</strong>
        {% if multipage.is_multipage %}
          <span class="pill warn">yes</span>
        {% else %}
          <span class="pill">no strong signal</span>
        {% endif %}
        {% for signal in multipage.signals %}
          <span class="pill">{{ signal }}</span>
        {% endfor %}
      </div>

      <div class="nav">
        {% if prev_idx is not none %}
          <a class="btn secondary" href="{{ url_for('review_page', idx=prev_idx, filter=filter_mode) }}">Previous</a>
        {% endif %}
        {% if next_idx is not none %}
          <a class="btn secondary" href="{{ url_for('review_page', idx=next_idx, filter=filter_mode) }}">Next</a>
        {% endif %}
        {% if next_unresolved_idx is not none %}
          <a class="btn ghost" href="{{ url_for('review_page', idx=next_unresolved_idx, filter=filter_mode) }}">Next Unresolved</a>
        {% endif %}
      </div>

      <div class="filterbar">
        <a class="btn ghost" href="{{ url_for('review_page', idx=0, filter='all') }}">All</a>
        <a class="btn ghost" href="{{ url_for('review_page', idx=first_index('unresolved'), filter='unresolved') }}">Unresolved</a>
        <a class="btn ghost" href="{{ url_for('review_page', idx=first_index('promote'), filter='promote') }}">Promote</a>
        <a class="btn ghost" href="{{ url_for('review_page', idx=first_index('reject'), filter='reject') }}">Reject</a>
        <a class="btn ghost" href="{{ url_for('review_page', idx=first_index('hold'), filter='hold') }}">Hold</a>
      </div>

      <div class="toolbar">
        <button type="button" class="secondary" onclick="setZoom(1)">100%</button>
        <button type="button" class="secondary" onclick="setZoom(1.5)">150%</button>
        <button type="button" class="secondary" onclick="setZoom(2)">200%</button>
        <button type="button" class="secondary" onclick="adjustZoom(0.8)">- Zoom</button>
        <button type="button" class="secondary" onclick="adjustZoom(1.25)">+ Zoom</button>
        <button type="button" class="secondary" onclick="fitWidth()">Fit Width</button>
        <a class="btn ghost" target="_blank" href="{{ url_for('image_file', idx=index) }}">Open Image</a>
      </div>

      <div class="viewer" id="viewer">
        {% if row.local_image_exists %}
          <img id="reviewImage" src="{{ url_for('image_file', idx=index) }}" alt="review image">
        {% else %}
          <div class="muted">Local image not found. Sync review assets first.</div>
        {% endif %}
      </div>

      {% if multipage.related_pages %}
      <div class="card" style="margin-top:10px;">
        <h3 style="margin-top:0;">Related Pages</h3>
        <div class="table-wrap">
          <table class="data">
            <tr>
              <th>Idx</th>
              <th>Decision</th>
              <th>Why linked</th>
              <th>Title</th>
              <th>Open</th>
            </tr>
            {% for rel in multipage.related_pages %}
            <tr>
              <td>{{ rel.idx + 1 }}</td>
              <td>{{ rel.review_decision or '(blank)' }}</td>
              <td>{{ rel.link_reason }}</td>
              <td>{{ rel.title }}</td>
              <td><a class="btn ghost" href="{{ url_for('review_page', idx=rel.idx, filter='all') }}">Go</a></td>
            </tr>
            {% endfor %}
          </table>
        </div>
      </div>
      {% endif %}
    </section>

    <section class="stack">
      <div class="card">
        <form method="post" action="{{ url_for('save_review', idx=index, filter=filter_mode) }}">
          <div class="field">
            <label><strong>Review decision</strong></label>
            <select name="review_decision">
              <option value="" {% if not row.review_decision %}selected{% endif %}></option>
              <option value="promote" {% if row.review_decision == 'promote' %}selected{% endif %}>promote</option>
              <option value="reject" {% if row.review_decision == 'reject' %}selected{% endif %}>reject</option>
              <option value="hold" {% if row.review_decision == 'hold' %}selected{% endif %}>hold</option>
            </select>
          </div>
          <div class="field">
            <label><strong>Review notes</strong></label>
            <textarea name="review_notes">{{ row.review_notes or '' }}</textarea>
          </div>
          <div class="savebar">
            <button type="submit">Save</button>
            {% if row.local_result_exists %}
              <a class="btn ghost" target="_blank" href="{{ url_for('json_file', idx=index) }}">Open JSON</a>
            {% endif %}
          </div>
        </form>
      </div>

      <div class="card">
        <div class="meta">
          <div><strong>Priority</strong><br><span class="muted">{{ row.priority_bucket }}</span></div>
          <div><strong>Reasons</strong><br><span class="muted">{{ row.priority_reasons }}</span></div>
          <div><strong>Axis</strong><br><span class="muted">{{ row.row_axis_type or '(blank)' }}</span></div>
          <div><strong>Row label</strong><br><span class="muted">{{ row.row_label_name or '(blank)' }}</span></div>
        </div>
        <div class="field">
          <label><strong>OCR preview</strong></label>
          <div class="ocr-preview">{{ ocr_preview }}</div>
        </div>
      </div>

      <div class="card">
        <details open>
          <summary>Parsed OCR sections</summary>
          {{ parsed_html|safe }}
        </details>
      </div>

      <div class="card">
        <details>
          <summary>Raw JSON</summary>
          <pre>{{ raw_json }}</pre>
        </details>
      </div>
    </section>
  </main>

  <script>
    let zoom = 1.0;
    function imageEl() { return document.getElementById('reviewImage'); }
    function viewerEl() { return document.getElementById('viewer'); }
    function applyZoom() {
      const img = imageEl();
      if (!img) return;
      img.style.transform = `scale(${zoom})`;
    }
    function setZoom(value) {
      zoom = value;
      applyZoom();
    }
    function adjustZoom(mult) {
      zoom = Math.max(0.3, Math.min(6, zoom * mult));
      applyZoom();
    }
    function fitWidth() {
      const img = imageEl();
      const viewer = viewerEl();
      if (!img || !viewer || !img.naturalWidth) return;
      zoom = Math.max(0.1, (viewer.clientWidth - 20) / img.naturalWidth);
      applyZoom();
    }
    window.addEventListener('load', function() {
      const img = imageEl();
      if (img) {
        if (img.complete) fitWidth();
        else img.onload = fitWidth;
      }
    });
    document.addEventListener('keydown', function(evt) {
      if (evt.target && ['TEXTAREA', 'INPUT', 'SELECT'].includes(evt.target.tagName)) return;
      if (evt.key === 'j') {
        const next = document.querySelector('a[href*="/review/"][class*="secondary"]:last-of-type');
        if (next) window.location = next.href;
      } else if (evt.key === 'k') {
        const prev = document.querySelector('a[href*="/review/"][class*="secondary"]');
        if (prev) window.location = prev.href;
      } else if (evt.key === '1') {
        document.querySelector('select[name="review_decision"]').value = 'promote';
      } else if (evt.key === '2') {
        document.querySelector('select[name="review_decision"]').value = 'reject';
      } else if (evt.key === '3') {
        document.querySelector('select[name="review_decision"]').value = 'hold';
      } else if (evt.key === 's') {
        document.querySelector('form').submit();
      }
    });
  </script>
</body>
</html>
"""


def normalize_text(value: object) -> str:
    return " ".join(str(value or "").strip().split())


def normalized_group_title(row: dict[str, str]) -> str:
    text = normalize_text(row.get("title_from_payload") or row.get("title") or row.get("table_dir_name") or "")
    text = CONTINUATION_PAT.sub("", text)
    text = re.sub(r"\(?(?:contd|continued|concld|concluded)\.?\)?", "", text, flags=re.I)
    text = re.sub(r"\s+", " ", text)
    return text.strip(" .-|").lower()


def remote_to_local_path(remote_path: str, remote_root: Path, local_asset_root: Path) -> Path:
    path = Path(str(remote_path or "").strip())
    try:
        rel = path.relative_to(remote_root)
        return local_asset_root / rel
    except Exception:
        return local_asset_root / path.name


def load_rows(decisions_csv: Path, remote_root: Path, local_asset_root: Path) -> list[dict[str, str]]:
    rows = []
    with decisions_csv.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            local_img = remote_to_local_path(row.get("image_path", ""), remote_root, local_asset_root)
            local_json = remote_to_local_path(row.get("result_json", ""), remote_root, local_asset_root)
            row["local_image_path"] = str(local_img)
            row["local_result_json"] = str(local_json)
            row["local_image_exists"] = "1" if local_img.exists() else ""
            row["local_result_exists"] = "1" if local_json.exists() else ""
            row["_title_key"] = normalized_group_title(row)
            rows.append(row)
    return rows


def write_rows(decisions_csv: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = [k for k in rows[0].keys() if not k.startswith("local_")] if rows else []
    with decisions_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def flatten_value(value) -> str:
    if isinstance(value, dict):
        parts = []
        for key, inner in value.items():
            parts.append(f"{normalize_text(key)}={inner}")
        return "; ".join(parts)
    if value is None:
        return ""
    return str(value)


def build_ocr_preview(result_json_path: Path) -> tuple[str, str, str]:
    if not result_json_path.exists():
        return "(result JSON not synced locally)", "<div class='muted'>No local JSON synced.</div>", "(missing JSON)"
    try:
        obj = json.load(result_json_path.open(encoding="utf-8"))
    except Exception as exc:
        msg = f"(could not read JSON: {exc})"
        return msg, f"<div class='muted'>{escape(msg)}</div>", msg

    preview_parts = []
    title = normalize_text(obj.get("title") or "")
    if title:
        preview_parts.append(f"TITLE: {title}")
    acceptance = obj.get("acceptance") or {}
    decision = normalize_text(acceptance.get("decision") or "")
    if decision:
        preview_parts.append(f"ACCEPTANCE: {decision}")

    parsed_parts = []
    sections = obj.get("data") or []
    if isinstance(sections, list):
        for section in sections[:6]:
            section_name = normalize_text(section.get("name") or "Section")
            rows = section.get("rows") or []
            preview_parts.append(f"\nSECTION: {section_name}")
            for row in rows[:6]:
                if isinstance(row, dict):
                    bits = [f"{normalize_text(k)}={flatten_value(v)}" for k, v in row.items()]
                    preview_parts.append("; ".join(bits))
                else:
                    preview_parts.append(str(row))
            if len(rows) > 6:
                preview_parts.append(f"... {len(rows) - 6} more rows")

            columns = []
            for row in rows[:20]:
                if isinstance(row, dict):
                    for key in row.keys():
                        if key not in columns:
                            columns.append(key)
            table = [f"<h4>{escape(section_name)}</h4>"]
            if not columns or not rows:
                table.append("<div class='muted'>No row data.</div>")
            else:
                table.append("<div class='table-wrap'><table class='data'><tr>")
                for col in columns:
                    table.append(f"<th>{escape(str(col))}</th>")
                table.append("</tr>")
                for row in rows[:30]:
                    table.append("<tr>")
                    if isinstance(row, dict):
                        for col in columns:
                            table.append(f"<td>{escape(flatten_value(row.get(col)))}</td>")
                    else:
                        table.append(f"<td colspan='{len(columns)}'>{escape(str(row))}</td>")
                    table.append("</tr>")
                table.append("</table></div>")
            parsed_parts.append("".join(table))
    else:
        preview_parts.append(json.dumps(sections, ensure_ascii=True)[:4000])
        parsed_parts.append(f"<pre>{escape(json.dumps(sections, indent=2, ensure_ascii=True)[:12000])}</pre>")

    raw_json = json.dumps(obj, indent=2, ensure_ascii=True)[:30000]
    return "\n".join(preview_parts)[:12000], "".join(parsed_parts) or "<div class='muted'>No parsed sections.</div>", raw_json


def build_index(rows: list[dict[str, str]], filter_mode: str) -> list[int]:
    if filter_mode == "unresolved":
        return [i for i, row in enumerate(rows) if not normalize_text(row.get("review_decision"))]
    if filter_mode in {"promote", "reject", "hold"}:
        return [i for i, row in enumerate(rows) if normalize_text(row.get("review_decision")) == filter_mode]
    return list(range(len(rows)))


def create_app(decisions_csv: Path, remote_root: Path, local_asset_root: Path):
    app = Flask(__name__)

    def get_rows():
        return load_rows(decisions_csv, remote_root, local_asset_root)

    def first_index(filter_mode: str) -> int:
        rows = get_rows()
        indices = build_index(rows, filter_mode)
        return indices[0] if indices else 0

    def multipage_context(rows: list[dict[str, str]], idx: int) -> dict[str, object]:
        row = rows[idx]
        text_blob = " | ".join(
            [
                row.get("title", ""),
                row.get("title_from_payload", ""),
                row.get("column_groups_from_payload", ""),
                row.get("table_dir_name", ""),
            ]
        )

        same_dir = [i for i, other in enumerate(rows) if i != idx and other.get("table_dir_name") == row.get("table_dir_name")]
        same_source_title = [
            i for i, other in enumerate(rows)
            if i != idx
            and other.get("source_file") == row.get("source_file")
            and other.get("_title_key") == row.get("_title_key")
            and row.get("_title_key")
        ]

        signals = []
        if same_dir:
            signals.append(f"same table dir ({len(same_dir) + 1} pages)")
        if CONTINUATION_PAT.search(text_blob):
            signals.append("continued/concluded marker")
        if PART_PAT.search(text_blob):
            signals.append("part marker")
        if same_source_title:
            signals.append(f"same source+title cluster ({len(same_source_title) + 1})")

        related = []
        added = set()
        for rel_idx in same_dir:
            added.add(rel_idx)
            rel = rows[rel_idx]
            related.append(
                {
                    "idx": rel_idx,
                    "review_decision": rel.get("review_decision", ""),
                    "title": rel.get("title_from_payload") or rel.get("title") or rel.get("table_dir_name"),
                    "link_reason": "same table dir",
                }
            )
        for rel_idx in same_source_title:
            if rel_idx in added:
                continue
            added.add(rel_idx)
            rel = rows[rel_idx]
            related.append(
                {
                    "idx": rel_idx,
                    "review_decision": rel.get("review_decision", ""),
                    "title": rel.get("title_from_payload") or rel.get("title") or rel.get("table_dir_name"),
                    "link_reason": "same source + title",
                }
            )
        related.sort(key=lambda item: item["idx"])
        return {
            "is_multipage": bool(signals),
            "signals": signals,
            "related_pages": related,
        }

    @app.route("/")
    def home():
        return redirect(url_for("review_page", idx=first_index("unresolved"), filter="unresolved"))

    @app.route("/image/<int:idx>")
    def image_file(idx: int):
        rows = get_rows()
        if idx < 0 or idx >= len(rows):
            abort(404)
        path = Path(rows[idx]["local_image_path"])
        if not path.exists():
            abort(404)
        return send_file(path)

    @app.route("/json/<int:idx>")
    def json_file(idx: int):
        rows = get_rows()
        if idx < 0 or idx >= len(rows):
            abort(404)
        path = Path(rows[idx]["local_result_json"])
        if not path.exists():
            abort(404)
        return send_file(path)

    @app.route("/review/<int:idx>")
    def review_page(idx: int):
        filter_mode = request.args.get("filter", "all")
        rows = get_rows()
        if idx < 0 or idx >= len(rows):
            abort(404)

        visible = build_index(rows, filter_mode)
        if idx not in visible:
            if visible:
                return redirect(url_for("review_page", idx=visible[0], filter=filter_mode))
            filter_mode = "all"
            visible = build_index(rows, filter_mode)

        pos = visible.index(idx) if idx in visible else 0
        prev_idx = visible[pos - 1] if pos > 0 else None
        next_idx = visible[pos + 1] if pos + 1 < len(visible) else None

        next_unresolved_idx = None
        for candidate in visible[pos + 1:]:
            if not normalize_text(rows[candidate].get("review_decision")):
                next_unresolved_idx = candidate
                break

        row = rows[idx]
        ocr_preview, parsed_html, raw_json = build_ocr_preview(Path(row["local_result_json"]))
        multipage = multipage_context(rows, idx)

        return render_template_string(
            APP_HTML,
            row=row,
            index=idx,
            total=len(rows),
            filter_mode=filter_mode,
            prev_idx=prev_idx,
            next_idx=next_idx,
            next_unresolved_idx=next_unresolved_idx,
            multipage=multipage,
            ocr_preview=ocr_preview,
            parsed_html=parsed_html,
            raw_json=raw_json,
            first_index=first_index,
        )

    @app.route("/save/<int:idx>", methods=["POST"])
    def save_review(idx: int):
        filter_mode = request.args.get("filter", "all")
        rows = get_rows()
        if idx < 0 or idx >= len(rows):
            abort(404)

        decision = normalize_text(request.form.get("review_decision", ""))
        if decision not in {"", "promote", "reject", "hold"}:
            decision = ""
        note = str(request.form.get("review_notes", "") or "").strip()

        rows[idx]["review_decision"] = decision
        rows[idx]["review_notes"] = note
        rows[idx]["final_decision"] = decision if decision in {"promote", "reject", "hold"} else ""
        write_rows(decisions_csv, rows)
        return redirect(url_for("review_page", idx=idx, filter=filter_mode))

    return app


def main():
    parser = argparse.ArgumentParser(description="Local web app for reviewing core_review rows.")
    parser.add_argument("--decisions-csv", default=DEFAULT_DECISIONS_CSV)
    parser.add_argument("--local-asset-root", default=DEFAULT_LOCAL_ASSET_ROOT)
    parser.add_argument("--remote-root", default=DEFAULT_REMOTE_ROOT)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    decisions_csv = Path(args.decisions_csv).expanduser()
    local_asset_root = Path(args.local_asset_root).expanduser()
    remote_root = Path(args.remote_root).expanduser()

    app = create_app(decisions_csv, remote_root, local_asset_root)
    print(f"Starting core review app on http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
