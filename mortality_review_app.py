#!/usr/bin/env python3
"""Local review app for mortality subcollection tables."""

from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path

from flask import Flask, abort, redirect, render_template_string, request, send_file, url_for


DEFAULT_DECISIONS_CSV = "export_final/browse/by_topic/mortality/review_decisions.csv"
DEFAULT_BASE_DIR = "export_final/browse/by_topic/mortality"
DEFAULT_ASSET_ROOT = "export_final/browse/by_topic/mortality/review_assets"
DEFAULT_REMOTE_ROOT = "/n/home11/cruano/census_toc"


APP_HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Mortality Review</title>
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
      grid-template-columns: 1.15fr 1fr;
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
    .toolbar, .nav, .savebar, .filterbar {
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
    .stack { display: grid; gap: 14px; }
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
    pre {
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
      max-height: 72vh;
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
    @media (max-width: 1200px) {
      main { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <header>
    <div>
      <div class="muted">Mortality Review App</div>
      <h2 style="margin:4px 0 0 0;">{{ row.province }} {{ row.year }} | {{ row.title }}</h2>
    </div>
    <div class="muted">Row {{ index + 1 }} / {{ total }}{% if filter_mode %} | filter={{ filter_mode }}{% endif %}</div>
  </header>

  <main>
    <section class="stack">
      <div class="card">
        <div class="meta">
          <div><strong>Layer</strong><br><span class="pill">{{ row.layer }}</span></div>
          <div><strong>Tier</strong><br><span class="pill {{ 'ok' if row.quality_tier == 'strict_validated' else 'warn' if row.quality_tier == 'contextual_unvalidated' else 'bad' }}">{{ row.quality_tier }}</span></div>
          <div><strong>Kind</strong><br><span class="pill">{{ row.mortality_kind }}</span></div>
          <div><strong>Status</strong><br><span class="pill">{{ row.effective_quality_status }}</span></div>
          <div><strong>Family</strong><br><span class="muted">{{ row.table_family }}</span></div>
          <div><strong>Failed checks</strong><br><span class="muted">{{ row.failed_checks or '0' }}</span></div>
          <div><strong>Preview folder</strong><br><span class="muted">{{ row.preview_folder }}</span></div>
          <div><strong>Source image</strong><br><span class="muted">{{ source_image_status }}</span></div>
        </div>

        <div class="nav">
          {% if prev_idx is not none %}
            <a class="btn secondary" data-prev="1" href="{{ url_for('review_page', idx=prev_idx, filter=filter_mode) }}">Previous</a>
          {% endif %}
          {% if next_idx is not none %}
            <a class="btn secondary" data-next="1" href="{{ url_for('review_page', idx=next_idx, filter=filter_mode) }}">Next</a>
          {% endif %}
          {% if next_unresolved_idx is not none %}
            <a class="btn ghost" href="{{ url_for('review_page', idx=next_unresolved_idx, filter=filter_mode) }}">Next Unresolved</a>
          {% endif %}
        </div>

        <div class="filterbar">
          <a class="btn ghost" href="{{ url_for('review_page', idx=first_index('all'), filter='all') }}">All</a>
          <a class="btn ghost" href="{{ url_for('review_page', idx=first_index('unresolved'), filter='unresolved') }}">Unresolved</a>
          <a class="btn ghost" href="{{ url_for('review_page', idx=first_index('contextual_unvalidated'), filter='contextual_unvalidated') }}">Contextual</a>
          <a class="btn ghost" href="{{ url_for('review_page', idx=first_index('needs_review'), filter='needs_review') }}">Needs Review</a>
          <a class="btn ghost" href="{{ url_for('review_page', idx=first_index('keep'), filter='keep') }}">Keep</a>
          <a class="btn ghost" href="{{ url_for('review_page', idx=first_index('drop'), filter='drop') }}">Drop</a>
          <a class="btn ghost" href="{{ url_for('review_page', idx=first_index('hold'), filter='hold') }}">Hold</a>
        </div>

        <div class="toolbar">
          {% if has_local_image %}
            <a class="btn secondary" href="{{ url_for('source_image_file', idx=index) }}">Open Source Image</a>
          {% endif %}
          {% if has_local_source_json %}
            <a class="btn secondary" href="{{ url_for('source_json_file', idx=index) }}">Open Source OCR JSON</a>
          {% endif %}
          <a class="btn ghost" href="{{ url_for('preview_file', idx=index, name='summary.txt') }}">Open Summary</a>
          <a class="btn ghost" href="{{ url_for('preview_file', idx=index, name='interpretation.txt') }}">Open Interpretation</a>
          <a class="btn ghost" href="{{ url_for('preview_file', idx=index, name='matrix.csv') }}">Open Matrix CSV</a>
          <a class="btn ghost" href="{{ url_for('preview_file', idx=index, name='facts.csv') }}">Open Facts CSV</a>
          <a class="btn ghost" href="{{ url_for('preview_file', idx=index, name='source_catalog_row.json') }}">Open Source JSON</a>
        </div>
      </div>

      {% if has_local_image %}
      <div class="card">
        <h3 style="margin-top:0;">Source Image</h3>
        <div class="table-wrap" style="background:white; padding: 8px;">
          <img src="{{ url_for('source_image_file', idx=index) }}" alt="source image" style="display:block; max-height:80vh; max-width:100%; width:auto; height:auto; margin:0 auto;">
        </div>
      </div>
      {% endif %}

      <div class="card">
        <h3 style="margin-top:0;">Matrix View</h3>
        <div class="table-wrap">{{ matrix_html|safe }}</div>
      </div>

      <div class="card">
        <h3 style="margin-top:0;">Facts Preview</h3>
        <div class="table-wrap">{{ facts_html|safe }}</div>
      </div>
    </section>

    <section class="stack">
      <div class="card">
        <h3 style="margin-top:0;">Review</h3>
        <form method="post" action="{{ url_for('save_page', idx=index, filter=filter_mode) }}">
          <div class="field">
            <label>Suggested</label>
            <input value="{{ row.suggested_decision }}" readonly>
          </div>
          <div class="field">
            <label>Decision</label>
            <select name="review_decision">
              <option value="" {% if not row.review_decision %}selected{% endif %}></option>
              <option value="keep" {% if row.review_decision == 'keep' %}selected{% endif %}>keep</option>
              <option value="drop" {% if row.review_decision == 'drop' %}selected{% endif %}>drop</option>
              <option value="hold" {% if row.review_decision == 'hold' %}selected{% endif %}>hold</option>
            </select>
          </div>
          <div class="field">
            <label>Notes</label>
            <textarea name="review_notes">{{ row.review_notes }}</textarea>
          </div>
          <div class="savebar">
            <button type="submit">Save</button>
          </div>
        </form>
      </div>

      <div class="card">
        <h3 style="margin-top:0;">Summary</h3>
        <pre>{{ summary_text }}</pre>
      </div>

      <div class="card">
        <h3 style="margin-top:0;">Interpretation</h3>
        <pre>{{ interpretation_text }}</pre>
      </div>

      <div class="card">
        <h3 style="margin-top:0;">Source Catalog Row</h3>
        <pre>{{ source_json }}</pre>
      </div>
    </section>
  </main>
  <script>
    document.addEventListener('keydown', function(evt) {
      if (evt.target && ['TEXTAREA', 'INPUT', 'SELECT'].includes(evt.target.tagName)) return;
      if (evt.key === 'j') {
        const el = document.querySelector('a[href*="/row/"][href*="filter="][data-next]') || document.querySelector('a[href*="/row/"]');
        if (el) window.location = el.href;
      } else if (evt.key === 'k') {
        const el = document.querySelector('a[data-prev]');
        if (el) window.location = el.href;
      } else if (evt.key === '1') {
        document.querySelector('select[name="review_decision"]').value = 'keep';
      } else if (evt.key === '2') {
        document.querySelector('select[name="review_decision"]').value = 'drop';
      } else if (evt.key === '3') {
        document.querySelector('select[name="review_decision"]').value = 'hold';
      } else if (evt.key === 's') {
        const form = document.querySelector('form');
        if (form) form.submit();
      }
    });
  </script>
</body>
</html>
"""


def normalize_text(value: str) -> str:
    return str(value or "").strip()


def load_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def save_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def filter_indices(rows: list[dict], filter_mode: str) -> list[int]:
    mode = normalize_text(filter_mode) or "all"
    if mode == "all":
        return list(range(len(rows)))
    if mode == "unresolved":
        return [i for i, row in enumerate(rows) if not normalize_text(row.get("review_decision"))]
    if mode in {"keep", "drop", "hold"}:
        return [i for i, row in enumerate(rows) if normalize_text(row.get("review_decision")) == mode]
    if mode in {"strict_validated", "contextual_unvalidated", "needs_review"}:
        return [i for i, row in enumerate(rows) if normalize_text(row.get("quality_tier")) == mode]
    if mode in {"death_rate", "death_counts", "mortality_table", "other"}:
        return [i for i, row in enumerate(rows) if normalize_text(row.get("mortality_kind")) == mode]
    return list(range(len(rows)))


def first_index(rows: list[dict], filter_mode: str) -> int:
    filtered = filter_indices(rows, filter_mode)
    return filtered[0] if filtered else 0


def preview_dir(base_dir: Path, row: dict) -> Path:
    return base_dir / normalize_text(row.get("preview_folder"))


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def read_csv(path: Path, limit: int | None = None) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if limit is None:
            return list(reader)
        rows = []
        for i, row in enumerate(reader):
            if i >= limit:
                break
            rows.append(row)
        return rows


def table_html(rows: list[dict]) -> str:
    if not rows:
        return "<div class='muted'>(no rows)</div>"
    fieldnames = list(rows[0].keys())
    parts = ["<table class='data'><thead><tr>"]
    for name in fieldnames:
        parts.append(f"<th>{html.escape(name)}</th>")
    parts.append("</tr></thead><tbody>")
    for row in rows:
        parts.append("<tr>")
        for name in fieldnames:
            parts.append(f"<td>{html.escape(str(row.get(name, '')))}</td>")
        parts.append("</tr>")
    parts.append("</tbody></table>")
    return "".join(parts)


def map_remote_to_local(path_str: str, asset_root: Path, remote_root: Path) -> Path | None:
    raw = normalize_text(path_str)
    if not raw:
        return None
    path = Path(raw)
    candidates: list[Path] = []
    if path.is_absolute():
        try:
            candidates.append(asset_root / path.relative_to(remote_root))
        except Exception:
            candidates.append(path)
    else:
        candidates.append(asset_root / path)
        candidates.append(path)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def source_path_from_row_or_preview(row: dict, base_dir: Path, key: str) -> str:
    direct = normalize_text(row.get(key, ""))
    if direct:
        return direct
    preview = preview_dir(base_dir, row) / "source_catalog_row.json"
    if preview.exists():
        try:
            payload = json.loads(preview.read_text(encoding="utf-8"))
            return normalize_text(payload.get(key, ""))
        except Exception:
            return ""
    return ""


def create_app(rows: list[dict], decisions_csv: Path, base_dir: Path, asset_root: Path, remote_root: Path) -> Flask:
    app = Flask(__name__)

    @app.context_processor
    def inject_helpers():
        return {"first_index": lambda mode: first_index(rows, mode)}

    @app.route("/")
    def home():
        idx = first_index(rows, "unresolved")
        return redirect(url_for("review_page", idx=idx, filter="unresolved"))

    @app.route("/source-image/<int:idx>")
    def source_image_file(idx: int):
        if idx < 0 or idx >= len(rows):
            abort(404)
        remote = source_path_from_row_or_preview(rows[idx], base_dir, "source_image")
        path = map_remote_to_local(remote, asset_root, remote_root)
        if not path:
            abort(404)
        return send_file(path.resolve())

    @app.route("/source-json/<int:idx>")
    def source_json_file(idx: int):
        if idx < 0 or idx >= len(rows):
            abort(404)
        remote = source_path_from_row_or_preview(rows[idx], base_dir, "source_json")
        path = map_remote_to_local(remote, asset_root, remote_root)
        if not path:
            abort(404)
        return send_file(path.resolve())

    @app.route("/row/<int:idx>")
    def review_page(idx: int):
        if idx < 0 or idx >= len(rows):
            abort(404)
        filter_mode = normalize_text(request.args.get("filter", "unresolved")) or "unresolved"
        valid = filter_indices(rows, filter_mode)
        if idx not in valid:
            idx = valid[0] if valid else 0
        row = rows[idx]
        cur_preview = preview_dir(base_dir, row)

        prev_idx = None
        next_idx = None
        next_unresolved_idx = None
        if valid:
            pos = valid.index(idx)
            if pos > 0:
                prev_idx = valid[pos - 1]
            if pos + 1 < len(valid):
                next_idx = valid[pos + 1]
        unresolved = [i for i, r in enumerate(rows) if not normalize_text(r.get("review_decision"))]
        for candidate in unresolved:
            if candidate > idx:
                next_unresolved_idx = candidate
                break

        summary_text = read_text(cur_preview / "summary.txt")
        interpretation_text = read_text(cur_preview / "interpretation.txt")
        matrix_rows = read_csv(cur_preview / "matrix.csv")
        facts_rows = read_csv(cur_preview / "facts.csv", limit=120)
        source_json_text = read_text(cur_preview / "source_catalog_row.json")
        remote_image = source_path_from_row_or_preview(row, base_dir, "source_image")
        remote_source_json = source_path_from_row_or_preview(row, base_dir, "source_json")
        local_image = map_remote_to_local(remote_image, asset_root, remote_root)
        local_source_json = map_remote_to_local(remote_source_json, asset_root, remote_root)
        source_image_status = str(local_image) if local_image else "not bundled locally"

        return render_template_string(
            APP_HTML,
            row=row,
            index=idx,
            total=len(rows),
            filter_mode=filter_mode,
            prev_idx=prev_idx,
            next_idx=next_idx,
            next_unresolved_idx=next_unresolved_idx,
            summary_text=summary_text,
            interpretation_text=interpretation_text,
            matrix_html=table_html(matrix_rows),
            facts_html=table_html(facts_rows),
            source_json=source_json_text,
            source_image_status=source_image_status,
            has_local_image=bool(local_image),
            has_local_source_json=bool(local_source_json),
        )

    @app.route("/save/<int:idx>", methods=["POST"])
    def save_page(idx: int):
        if idx < 0 or idx >= len(rows):
            abort(404)
        filter_mode = normalize_text(request.args.get("filter", "unresolved")) or "unresolved"
        decision = normalize_text(request.form.get("review_decision", "")).lower()
        if decision not in {"", "keep", "drop", "hold"}:
            decision = ""
        rows[idx]["review_decision"] = decision
        rows[idx]["review_notes"] = normalize_text(request.form.get("review_notes", ""))
        rows[idx]["final_decision"] = decision if decision in {"keep", "drop", "hold"} else ""
        save_rows(decisions_csv, rows)
        return redirect(url_for("review_page", idx=idx, filter=filter_mode))

    @app.route("/file/<int:idx>/<path:name>")
    def preview_file(idx: int, name: str):
        if idx < 0 or idx >= len(rows):
            abort(404)
        allowed = {"summary.txt", "interpretation.txt", "matrix.csv", "facts.csv", "source_catalog_row.json"}
        if name not in allowed:
            abort(404)
        path = preview_dir(base_dir, rows[idx]) / name
        if not path.exists():
            abort(404)
        return send_file(path)

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Local mortality review app")
    parser.add_argument("--decisions-csv", default=DEFAULT_DECISIONS_CSV)
    parser.add_argument("--base-dir", default=DEFAULT_BASE_DIR)
    parser.add_argument("--asset-root", default=DEFAULT_ASSET_ROOT)
    parser.add_argument("--remote-root", default=DEFAULT_REMOTE_ROOT)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()

    decisions_csv = Path(args.decisions_csv)
    base_dir = Path(args.base_dir)
    asset_root = Path(args.asset_root)
    remote_root = Path(args.remote_root)
    rows = load_rows(decisions_csv)
    if not rows:
        raise SystemExit("No mortality review rows found.")

    app = create_app(rows, decisions_csv, base_dir, asset_root, remote_root)
    print(f"Starting mortality review app on http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
