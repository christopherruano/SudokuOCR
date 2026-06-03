from __future__ import annotations

import csv
import html
import json
from pathlib import Path

from flask import Flask, abort, redirect, render_template_string, request, send_file, url_for


DEFAULT_REVIEW_CSV = "export_final/sendout_final/review/plausibility_review_app/review_decisions.csv"
DEFAULT_ASSEMBLED_WORKBOOK = "export_final/sendout_final/review/plausibility_review_app/assembled_tables_review_all.xlsx"
ASSEMBLED_PREVIEW_DIR = "export/final_project_exports/preview_pack/assembled_reviewed"
APP_PORT = 8768


APP_HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Plausibility Review</title>
  <style>
    :root { color-scheme: light; --border:#d8d8d8; --muted:#666; --bg:#f7f7f5; --ink:#191919; }
    * { box-sizing: border-box; }
    body { margin:0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color:var(--ink); background:var(--bg); }
    header { position:sticky; top:0; z-index:2; background:white; border-bottom:1px solid var(--border); padding:10px 14px; display:flex; align-items:center; gap:10px; flex-wrap:wrap; }
    main { display:grid; grid-template-columns: 420px minmax(420px, 1fr); gap:12px; padding:12px; }
    aside, section { background:white; border:1px solid var(--border); border-radius:6px; overflow:hidden; }
    .panel { padding:12px; border-bottom:1px solid var(--border); }
    .panel:last-child { border-bottom:0; }
    .title { font-weight:700; line-height:1.25; }
    .muted { color:var(--muted); font-size:12px; }
    .kv { display:grid; grid-template-columns: 130px 1fr; gap:5px 8px; font-size:13px; }
    .kv strong { color:#333; }
    .btn { appearance:none; border:1px solid var(--border); background:#fff; color:#111; padding:6px 9px; border-radius:5px; text-decoration:none; font-size:13px; cursor:pointer; display:inline-block; }
    .btn.primary { background:#111; color:white; border-color:#111; }
    .btn.warn { background:#fff7e5; border-color:#d9ad4f; }
    .btn.danger { background:#fff0f0; border-color:#d88484; }
    .btn.active { outline:2px solid #111; }
    .row { display:flex; gap:8px; align-items:center; flex-wrap:wrap; }
    textarea { width:100%; min-height:82px; resize:vertical; font-family:inherit; font-size:13px; }
    select { font-size:13px; padding:5px; }
    .image-wrap { padding:10px; display:flex; justify-content:center; align-items:flex-start; min-height:360px; background:#fafafa; overflow:auto; }
    img.source { display:block; max-width:100%; max-height:82vh; width:auto; height:auto; border:1px solid var(--border); background:white; }
    .issue { border-left:3px solid #b44; padding-left:8px; white-space:pre-wrap; font-size:13px; line-height:1.35; }
    pre { white-space:pre-wrap; overflow:auto; max-height:320px; font-size:12px; background:#fafafa; padding:10px; border:1px solid var(--border); }
    .table-scroll { overflow:auto; max-height:520px; border-top:1px solid var(--border); }
    table { border-collapse:collapse; width:100%; font-size:12px; }
    td, th { border:1px solid var(--border); padding:4px 6px; vertical-align:top; }
    th { background:#f1f1ef; position:sticky; top:0; }
    @media (max-width: 980px) { main { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
  <header>
    <strong>Plausibility Review</strong>
    <span class="muted">{{ index + 1 }} / {{ total }}</span>
    {% if has_assembled_workbook %}<a class="btn primary" href="{{ url_for('assembled_workbook_file') }}">Open assembled workbook</a>{% endif %}
    {% for mode,label in filters %}
      <a class="btn {% if filter_mode == mode %}active{% endif %}" href="{{ url_for('review_page', idx=first_index(mode), filter=mode) }}">{{ label }}</a>
    {% endfor %}
  </header>
  <main>
    <aside>
      <div class="panel">
        <div class="title">{{ row.title }}</div>
        <div class="muted">{{ row.table_id }}</div>
      </div>
      <div class="panel kv">
        <strong>Layer</strong><span>{{ row.layer }}</span>
        <strong>Place/year</strong><span>{{ row.province }} / {{ row.year }}</span>
        <strong>Bucket</strong><span>{{ row.sendout_bucket }}</span>
        <strong>Format</strong><span>{{ row.format_class }}</span>
        <strong>Status</strong><span>{{ row.plausibility_status }}</span>
        <strong>Recommendation</strong><span>{{ row.recommendation }}</span>
        <strong>Quality</strong><span>{{ row.quality_tier }}</span>
        <strong>Checks failed</strong><span>{{ row.failed_checks }}</span>
        <strong>Facts seen</strong><span>{{ row.facts_seen }}</span>
      </div>
      <div class="panel">
        <div class="row">
          {% if prev_idx is not none %}<a class="btn" href="{{ url_for('review_page', idx=prev_idx, filter=filter_mode) }}">Prev</a>{% endif %}
          {% if next_idx is not none %}<a class="btn" href="{{ url_for('review_page', idx=next_idx, filter=filter_mode) }}">Next</a>{% endif %}
          {% if next_unresolved_idx is not none %}<a class="btn primary" href="{{ url_for('review_page', idx=next_unresolved_idx, filter='unresolved') }}">Next unresolved</a>{% endif %}
        </div>
      </div>
      <div class="panel">
        <div class="row">
          {% if has_image %}<a class="btn" href="{{ url_for('source_image_file', idx=index) }}" target="_blank">Open image</a>{% endif %}
          {% if has_json %}<a class="btn" href="{{ url_for('source_json_file', idx=index) }}" target="_blank">Open JSON</a>{% endif %}
          {% if has_excel %}<a class="btn" href="{{ url_for('excel_file', idx=index) }}">Open Excel</a>{% endif %}
          {% if is_assembled %}<a class="btn primary" href="{{ url_for('assembled_review_page', idx=index, filter=filter_mode) }}">Review assembled table</a>{% endif %}
          {% if has_matrix %}<a class="btn" href="{{ url_for('matrix_file', idx=index) }}" target="_blank">Open matrix CSV</a>{% endif %}
          {% if has_facts %}<a class="btn" href="{{ url_for('facts_file', idx=index) }}" target="_blank">Open facts CSV</a>{% endif %}
        </div>
        <p class="muted">Image: {{ row.source_image_local or "not local" }}</p>
        <p class="muted">Excel: {{ row.excel_path or "not available" }}</p>
      </div>
      <form class="panel" method="post" action="{{ url_for('save_page', idx=index, filter=filter_mode) }}">
        <label class="muted">Decision</label><br>
        <select name="review_decision">
          {% for opt in ["", "keep", "drop", "rerun", "manual_check_done", "hold"] %}
            <option value="{{ opt }}" {% if row.review_decision == opt %}selected{% endif %}>{{ opt or "(unset)" }}</option>
          {% endfor %}
        </select>
        <p><textarea name="review_notes" placeholder="Review notes">{{ row.review_notes }}</textarea></p>
        <button class="btn primary" type="submit">Save</button>
      </form>
      <div class="panel">
        <div class="muted">Flags</div>
        <pre>{{ row.flags }}</pre>
        <div class="muted">Issue details</div>
        <div class="issue">{{ row.issue_details or row.issue_flags }}</div>
      </div>
    </aside>
    <section>
      <div class="panel">
        <strong>Source image</strong>
      </div>
      {% if has_image %}
        <div class="image-wrap"><img class="source" src="{{ url_for('source_image_file', idx=index) }}"></div>
      {% else %}
        <div class="panel muted">No local image found for this row.</div>
      {% endif %}
      <div class="panel">
        <strong>Preview summary</strong>
        <pre>{{ preview_summary }}</pre>
      </div>
      <div class="panel">
        <strong>Generated table preview</strong>
      </div>
      <div class="table-scroll">{{ matrix_html|safe }}</div>
      <div class="panel">
        <strong>Fact rows preview</strong>
      </div>
      <div class="table-scroll">{{ facts_html|safe }}</div>
    </section>
  </main>
</body>
</html>
"""


ASSEMBLED_HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Assembled Table Review</title>
  <style>
    :root { color-scheme: light; --border:#d8d8d8; --muted:#666; --bg:#f7f7f5; --ink:#191919; }
    * { box-sizing: border-box; }
    body { margin:0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color:var(--ink); background:var(--bg); }
    header { position:sticky; top:0; z-index:2; background:white; border-bottom:1px solid var(--border); padding:10px 14px; display:flex; align-items:center; gap:10px; flex-wrap:wrap; }
    main { padding:12px; display:grid; grid-template-columns: 390px minmax(520px, 1fr); gap:12px; }
    aside, section, .card { background:white; border:1px solid var(--border); border-radius:6px; overflow:hidden; }
    .panel { padding:12px; border-bottom:1px solid var(--border); }
    .panel:last-child { border-bottom:0; }
    .title { font-weight:700; line-height:1.25; }
    .muted { color:var(--muted); font-size:12px; }
    .kv { display:grid; grid-template-columns: 135px 1fr; gap:5px 8px; font-size:13px; }
    .btn { appearance:none; border:1px solid var(--border); background:#fff; color:#111; padding:6px 9px; border-radius:5px; text-decoration:none; font-size:13px; cursor:pointer; display:inline-block; }
    .btn.primary { background:#111; color:white; border-color:#111; }
    .btn.danger { background:#fff0f0; border-color:#d88484; }
    .btn.warn { background:#fff7e5; border-color:#d9ad4f; }
    .row { display:flex; gap:8px; align-items:center; flex-wrap:wrap; }
    textarea { width:100%; min-height:82px; resize:vertical; font-family:inherit; font-size:13px; }
    select { font-size:13px; padding:5px; }
    pre { white-space:pre-wrap; overflow:auto; max-height:240px; font-size:12px; background:#fafafa; padding:10px; border:1px solid var(--border); }
    .section-block { margin-bottom:18px; background:white; border:1px solid var(--border); border-radius:6px; overflow:hidden; }
    .image-strip { display:flex; gap:10px; overflow:auto; padding:10px; background:#fafafa; border-top:1px solid var(--border); }
    .member-image { min-width:260px; max-width:420px; }
    .member-image img { max-width:100%; max-height:420px; border:1px solid var(--border); background:white; display:block; }
    .table-scroll { overflow:auto; max-height:620px; border-top:1px solid var(--border); }
    table { border-collapse:collapse; width:100%; font-size:12px; }
    td, th { border:1px solid var(--border); padding:4px 6px; vertical-align:top; }
    th { background:#f1f1ef; position:sticky; top:0; }
    @media (max-width: 980px) { main { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
  <header>
    <strong>Assembled Table Review</strong>
    <span class="muted">{{ index + 1 }} / {{ total }}</span>
    <a class="btn" href="{{ url_for('review_page', idx=index, filter=filter_mode) }}">Back to normal review</a>
    {% if prev_idx is not none %}<a class="btn" href="{{ url_for('assembled_review_page', idx=prev_idx, filter=filter_mode) }}">Prev assembled</a>{% endif %}
    {% if next_idx is not none %}<a class="btn" href="{{ url_for('assembled_review_page', idx=next_idx, filter=filter_mode) }}">Next assembled</a>{% endif %}
    {% if has_assembled_workbook %}<a class="btn" href="{{ url_for('assembled_workbook_file') }}">Open combined workbook</a>{% endif %}
  </header>
  <main>
    <aside>
      <div class="panel">
        <div class="title">{{ row.title }}</div>
        <div class="muted">{{ row.table_id }}</div>
      </div>
      <div class="panel kv">
        <strong>Place/year</strong><span>{{ row.province }} / {{ row.year }}</span>
        <strong>Family</strong><span>{{ row.table_family or row.format_class }}</span>
        <strong>Facts seen</strong><span>{{ row.facts_seen }}</span>
        <strong>Recommendation</strong><span>{{ row.recommendation }}</span>
        <strong>Quality</strong><span>{{ row.quality_tier }}</span>
        <strong>Checks failed</strong><span>{{ row.failed_checks }}</span>
        <strong>Preview</strong><span>{{ preview_dir or "not available" }}</span>
        <strong>Sections</strong><span>{{ sections|length }}</span>
        <strong>Members</strong><span>{{ members|length }}</span>
      </div>
      <form class="panel" method="post" action="{{ url_for('save_assembled_page', idx=index, filter=filter_mode) }}">
        <div class="muted">One decision applies to the whole assembled table.</div>
        <p>
          <select name="review_decision">
            {% for opt in ["", "keep", "drop", "rerun", "manual_check_done", "hold"] %}
              <option value="{{ opt }}" {% if row.review_decision == opt %}selected{% endif %}>{{ opt or "(unset)" }}</option>
            {% endfor %}
          </select>
        </p>
        <p><textarea name="review_notes" placeholder="Decision notes for the whole assembled table">{{ row.review_notes }}</textarea></p>
        <div class="row">
          <button class="btn primary" name="review_decision" value="keep" type="submit">Accept whole assembled table</button>
          <button class="btn danger" name="review_decision" value="drop" type="submit">Reject whole assembled table</button>
          <button class="btn warn" name="review_decision" value="rerun" type="submit">Mark for rerun</button>
        </div>
      </form>
      <div class="panel">
        <div class="muted">Issue details</div>
        <pre>{{ row.issue_details or row.issue_flags or row.flags }}</pre>
      </div>
      <div class="panel">
        <strong>Member pages</strong>
      </div>
      {% if member_images %}
        <div class="image-strip">
          {% for image in member_images %}
            <div class="member-image">
              <div class="muted">member {{ image.member_index }}</div>
              <a href="{{ url_for('assembled_member_image_file', idx=index, member_idx=image.member_index) }}" target="_blank">
                <img src="{{ url_for('assembled_member_image_file', idx=index, member_idx=image.member_index) }}">
              </a>
            </div>
          {% endfor %}
        </div>
      {% endif %}
      <div class="table-scroll">{{ members_html|safe }}</div>
    </aside>
    <section>
      {% if not sections %}
        <div class="panel muted">No local assembled section preview is available for this row. Use rerun/hold unless the metadata is enough for your decision.</div>
      {% endif %}
      {% for section in sections %}
        <div class="section-block">
          <div class="panel">
            <strong>Section {{ section.meta.section_index }}: {{ section.meta.section_name }}</strong>
            <div class="muted">strategy={{ section.meta.merge_strategy }}; input_pages={{ section.meta.input_pages }}; output_rows={{ section.meta.output_rows }}; file={{ section.meta.csv_file }}</div>
          </div>
          <div class="table-scroll">{{ section.html|safe }}</div>
        </div>
      {% endfor %}
      <form class="panel" method="post" action="{{ url_for('save_assembled_page', idx=index, filter=filter_mode) }}">
        <p><textarea name="review_notes" placeholder="Decision notes for the whole assembled table">{{ row.review_notes }}</textarea></p>
        <div class="row">
          <button class="btn primary" name="review_decision" value="keep" type="submit">Accept whole assembled table</button>
          <button class="btn danger" name="review_decision" value="drop" type="submit">Reject whole assembled table</button>
          <button class="btn warn" name="review_decision" value="rerun" type="submit">Mark for rerun</button>
        </div>
      </form>
    </section>
  </main>
</body>
</html>
"""


FILTERS = [
    ("unresolved", "Unresolved"),
    ("all", "All"),
    ("repair", "Repair/rerun"),
    ("exclude", "Exclude/rerun"),
    ("manual_ok", "Manual OK"),
    ("assembled", "Assembled"),
    ("mortality", "Mortality"),
    ("has_image", "Has image"),
    ("has_entries", "Has entries"),
    ("entries_no_preview", "Entries no preview"),
    ("no_facts_rerun", "No facts rerun"),
    ("keep", "Kept"),
    ("drop", "Dropped"),
]


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except json.JSONDecodeError:
        return {}


def read_text(path_str: str) -> str:
    if not path_str:
        return ""
    path = Path(path_str)
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def read_csv_limited(path_str: str, limit: int = 80) -> list[dict]:
    if not path_str:
        return []
    path = Path(path_str)
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        out = []
        for idx, row in enumerate(reader):
            if idx >= limit:
                break
            out.append(row)
        return out


def read_csv_path_limited(path: Path, limit: int = 160) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8", errors="replace") as fh:
        reader = csv.DictReader(fh)
        out = []
        for idx, row in enumerate(reader):
            if idx >= limit:
                break
            out.append(row)
        return out


def table_html(rows: list[dict]) -> str:
    if not rows:
        return "<div class='panel muted'>No preview rows available.</div>"
    fields = list(rows[0].keys())
    parts = ["<table><thead><tr>"]
    for field in fields:
        parts.append(f"<th>{html.escape(field)}</th>")
    parts.append("</tr></thead><tbody>")
    for row in rows:
        parts.append("<tr>")
        for field in fields:
            parts.append(f"<td>{html.escape(str(row.get(field, '')))}</td>")
        parts.append("</tr>")
    parts.append("</tbody></table>")
    return "".join(parts)


def write_rows(path: Path, rows: list[dict]) -> None:
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def truthy(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def path_exists(value: str) -> bool:
    return bool(value) and Path(value).exists()


def localize_cluster_path(value: str) -> str:
    text = value or ""
    replacements = {
        "/n/home11/cruano/census_toc/digitized_tables/": "export/digitized_tables_referenced/digitized_tables/",
        "/n/home11/cruano/census_toc/ocr_results/digitized_tables/": "export/digitized_tables_referenced/digitized_tables/",
    }
    for old, new in replacements.items():
        if text.startswith(old):
            candidate = text.replace(old, new, 1)
            if Path(candidate).exists():
                return candidate
    return text


def is_assembled_row(row: dict) -> bool:
    return row.get("layer") == "assembled" or row.get("table_id", "").startswith("assembled::")


def assembled_preview_dirs() -> dict[str, Path]:
    root = Path(ASSEMBLED_PREVIEW_DIR)
    out = {}
    if not root.exists():
        return out
    for source_catalog in root.glob("*/source_catalog_row.json"):
        row = read_json(source_catalog)
        table_id = row.get("table_id")
        if table_id:
            out[table_id] = source_catalog.parent
    return out


def assembled_preview_dir(row: dict) -> Path | None:
    preview_dir = row.get("preview_folder", "")
    if preview_dir:
        path = Path(preview_dir)
        if path.exists():
            return path
        candidate = Path("export/final_project_exports/preview_pack") / preview_dir
        if candidate.exists():
            return candidate
    return assembled_preview_dirs().get(row.get("table_id", ""))


def assembled_sections(preview_dir: Path | None) -> list[dict]:
    if not preview_dir:
        return []
    section_index = preview_dir / "sections_index.csv"
    sections = read_csv_path_limited(section_index, 1000)
    out = []
    for section in sections:
        csv_file = section.get("csv_file", "")
        section_path = preview_dir / csv_file
        out.append(
            {
                "meta": section,
                "html": table_html(read_csv_path_limited(section_path, 220)),
            }
        )
    return out


def assembled_member_image_path(row: dict, member_idx: int) -> Path | None:
    preview_dir = assembled_preview_dir(row)
    if not preview_dir:
        return None
    members = read_csv_path_limited(preview_dir / "member_pages.csv", 1000)
    for member in members:
        if str(member.get("member_index", "")) != str(member_idx):
            continue
        path = Path(localize_cluster_path(member.get("source_image", "")))
        if path.exists():
            return path
    return None


def assembled_member_images(members: list[dict]) -> list[dict]:
    out = []
    for member in members:
        path = Path(localize_cluster_path(member.get("source_image", "")))
        if path.exists():
            out.append({"member_index": member.get("member_index", ""), "path": str(path)})
    return out


def filter_indices(rows: list[dict], mode: str) -> list[int]:
    out = []
    for i, row in enumerate(rows):
        rec = row.get("recommendation", "")
        decision = row.get("review_decision", "")
        title = f"{row.get('title','')} {row.get('format_class','')}".lower()
        ok = False
        if mode == "all":
            ok = True
        elif mode == "unresolved":
            ok = not decision
        elif mode == "repair":
            ok = "repair_or_rerun" in rec
        elif mode == "exclude":
            ok = "exclude_or_rerun" in rec
        elif mode == "manual_ok":
            ok = "manual_review_signal_but_number_sense_ok" in rec
        elif mode == "assembled":
            ok = is_assembled_row(row)
        elif mode == "mortality":
            ok = "mortality" in title or "death" in title
        elif mode == "has_image":
            ok = path_exists(row.get("source_image_local", ""))
        elif mode == "has_entries":
            ok = int(row.get("facts_seen") or 0) > 0
        elif mode == "entries_no_preview":
            ok = int(row.get("facts_seen") or 0) > 0 and not row.get("preview_matrix", "")
        elif mode == "no_facts_rerun":
            ok = int(row.get("facts_seen") or 0) == 0
        elif mode == "keep":
            ok = decision == "keep"
        elif mode == "drop":
            ok = decision == "drop"
        if ok:
            out.append(i)
    return out or list(range(len(rows)))


def first_index(rows: list[dict], mode: str) -> int:
    vals = filter_indices(rows, mode)
    return vals[0] if vals else 0


def create_app(rows: list[dict], review_csv: Path) -> Flask:
    app = Flask(__name__)

    @app.context_processor
    def helpers():
        return {"first_index": lambda mode: first_index(rows, mode)}

    @app.route("/")
    def home():
        return redirect(url_for("review_page", idx=first_index(rows, "unresolved"), filter="unresolved"))

    @app.route("/source-image/<int:idx>")
    def source_image_file(idx: int):
        if idx < 0 or idx >= len(rows):
            abort(404)
        path = Path(rows[idx].get("source_image_local", ""))
        if not path.exists():
            abort(404)
        return send_file(path.resolve())

    @app.route("/source-json/<int:idx>")
    def source_json_file(idx: int):
        if idx < 0 or idx >= len(rows):
            abort(404)
        path = Path(rows[idx].get("source_json_local", ""))
        if not path.exists():
            abort(404)
        return send_file(path.resolve())

    @app.route("/excel/<int:idx>")
    def excel_file(idx: int):
        if idx < 0 or idx >= len(rows):
            abort(404)
        path = Path(rows[idx].get("excel_abs", ""))
        if not path.exists():
            abort(404)
        return send_file(path.resolve(), as_attachment=False)

    @app.route("/assembled-workbook")
    def assembled_workbook_file():
        path = Path(DEFAULT_ASSEMBLED_WORKBOOK)
        if not path.exists():
            abort(404)
        return send_file(path.resolve(), as_attachment=False)

    @app.route("/matrix/<int:idx>")
    def matrix_file(idx: int):
        if idx < 0 or idx >= len(rows):
            abort(404)
        path = Path(rows[idx].get("preview_matrix", ""))
        if not path.exists():
            abort(404)
        return send_file(path.resolve())

    @app.route("/facts/<int:idx>")
    def facts_file(idx: int):
        if idx < 0 or idx >= len(rows):
            abort(404)
        path = Path(rows[idx].get("preview_facts", ""))
        if not path.exists():
            abort(404)
        return send_file(path.resolve())

    @app.route("/assembled-member-image/<int:idx>/<int:member_idx>")
    def assembled_member_image_file(idx: int, member_idx: int):
        if idx < 0 or idx >= len(rows):
            abort(404)
        if not is_assembled_row(rows[idx]):
            abort(404)
        path = assembled_member_image_path(rows[idx], member_idx)
        if not path or not path.exists():
            abort(404)
        return send_file(path.resolve())

    @app.route("/assembled/<int:idx>")
    def assembled_review_page(idx: int):
        if idx < 0 or idx >= len(rows):
            abort(404)
        if not is_assembled_row(rows[idx]):
            abort(404)
        mode = request.args.get("filter", "assembled")
        valid = [i for i in filter_indices(rows, mode) if is_assembled_row(rows[i])]
        if idx not in valid:
            idx = valid[0]
        row = rows[idx]
        pos = valid.index(idx)
        prev_idx = valid[pos - 1] if pos > 0 else None
        next_idx = valid[pos + 1] if pos + 1 < len(valid) else None
        preview_dir = assembled_preview_dir(row)
        members = read_csv_path_limited(preview_dir / "member_pages.csv", 1000) if preview_dir else []
        sections = assembled_sections(preview_dir)
        return render_template_string(
            ASSEMBLED_HTML,
            row=row,
            index=idx,
            total=len(rows),
            filter_mode=mode,
            prev_idx=prev_idx,
            next_idx=next_idx,
            preview_dir=str(preview_dir) if preview_dir else "",
            members=members,
            sections=sections,
            member_images=assembled_member_images(members),
            members_html=table_html(members),
            has_assembled_workbook=Path(DEFAULT_ASSEMBLED_WORKBOOK).exists(),
        )

    @app.route("/row/<int:idx>")
    def review_page(idx: int):
        if idx < 0 or idx >= len(rows):
            abort(404)
        mode = request.args.get("filter", "unresolved")
        valid = filter_indices(rows, mode)
        if idx not in valid:
            idx = valid[0]
        row = rows[idx]
        pos = valid.index(idx)
        prev_idx = valid[pos - 1] if pos > 0 else None
        next_idx = valid[pos + 1] if pos + 1 < len(valid) else None
        unresolved = [i for i, r in enumerate(rows) if not r.get("review_decision", "")]
        next_unresolved = next((i for i in unresolved if i > idx), None)
        return render_template_string(
            APP_HTML,
            row=row,
            index=idx,
            total=len(rows),
            filter_mode=mode,
            filters=FILTERS,
            has_assembled_workbook=Path(DEFAULT_ASSEMBLED_WORKBOOK).exists(),
            prev_idx=prev_idx,
            next_idx=next_idx,
            next_unresolved_idx=next_unresolved,
            has_image=path_exists(row.get("source_image_local", "")),
            has_json=path_exists(row.get("source_json_local", "")),
            has_excel=path_exists(row.get("excel_abs", "")),
            is_assembled=is_assembled_row(row),
            has_matrix=path_exists(row.get("preview_matrix", "")),
            has_facts=path_exists(row.get("preview_facts", "")),
            preview_summary=read_text(row.get("preview_summary", "")),
            matrix_html=table_html(read_csv_limited(row.get("preview_matrix", ""), 80)),
            facts_html=table_html(read_csv_limited(row.get("preview_facts", ""), 120)),
        )

    @app.route("/save/<int:idx>", methods=["POST"])
    def save_page(idx: int):
        if idx < 0 or idx >= len(rows):
            abort(404)
        mode = request.args.get("filter", "unresolved")
        rows[idx]["review_decision"] = request.form.get("review_decision", "")
        rows[idx]["review_notes"] = request.form.get("review_notes", "")
        write_rows(review_csv, rows)
        valid = filter_indices(rows, mode)
        next_idx = next((i for i in valid if i > idx), idx)
        return redirect(url_for("review_page", idx=next_idx, filter=mode))

    @app.route("/save-assembled/<int:idx>", methods=["POST"])
    def save_assembled_page(idx: int):
        if idx < 0 or idx >= len(rows):
            abort(404)
        if not is_assembled_row(rows[idx]):
            abort(404)
        mode = request.args.get("filter", "assembled")
        rows[idx]["review_decision"] = request.form.get("review_decision", "")
        rows[idx]["review_notes"] = request.form.get("review_notes", "")
        write_rows(review_csv, rows)
        valid = [i for i in filter_indices(rows, mode) if is_assembled_row(rows[i])]
        next_idx = next((i for i in valid if i > idx), idx)
        return redirect(url_for("assembled_review_page", idx=next_idx, filter=mode))

    return app


def main() -> None:
    review_csv = Path(DEFAULT_REVIEW_CSV)
    rows = read_rows(review_csv)
    app = create_app(rows, review_csv)
    print(f"Open http://127.0.0.1:{APP_PORT}")
    app.run(host="127.0.0.1", port=APP_PORT, debug=False)


if __name__ == "__main__":
    main()
