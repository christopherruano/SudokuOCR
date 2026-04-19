"""Local review app for census image verification.

Purpose:
- Show the source image beside the current extracted result
- Let a human mark the page as verified / corrected / flagged
- Save reviewed output in a separate annotations directory

Usage:
    python3 review_app.py
    python3 review_app.py --root age_tables/Berar/1901
    python3 review_app.py --review-queue results/_review_queue.jsonl
"""

import argparse
import json
from html import escape
from pathlib import Path

from flask import Flask, abort, redirect, render_template_string, request, send_file, url_for

from batch_production import discover_images
from result_lookup import find_all_results_for_image, parse_results_roots


APP_HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Census Review</title>
  <style>
    :root {
      --bg: #f3efe4;
      --card: #fffaf0;
      --ink: #1d1b18;
      --muted: #6b655c;
      --accent: #8c3d1f;
      --line: #d8cdb9;
      --ok: #2f6b3d;
      --warn: #9a6a12;
      --bad: #8e2b2b;
    }
    body {
      margin: 0;
      font-family: Georgia, "Iowan Old Style", serif;
      background: linear-gradient(180deg, #efe7d7 0%, var(--bg) 100%);
      color: var(--ink);
    }
    header, main {
      max-width: 1500px;
      margin: 0 auto;
      padding: 18px 22px;
    }
    header {
      display: flex;
      justify-content: space-between;
      align-items: baseline;
      border-bottom: 1px solid var(--line);
    }
    .muted { color: var(--muted); }
    .grid {
      display: grid;
      grid-template-columns: 1.15fr 1fr;
      gap: 18px;
      align-items: start;
    }
    .card {
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 14px;
      box-shadow: 0 8px 24px rgba(40, 28, 10, 0.06);
    }
    .image-toolbar {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      margin-bottom: 10px;
    }
    .image-viewport {
      max-height: 82vh;
      overflow: auto;
      border: 1px solid var(--line);
      border-radius: 10px;
      background: white;
      padding: 8px;
    }
    .image-viewport img {
      display: block;
      width: 160%;
      max-width: none;
      height: auto;
      border-radius: 8px;
      background: white;
      transform-origin: top left;
    }
    textarea, select, input {
      width: 100%;
      box-sizing: border-box;
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 10px;
      font: 14px/1.35 Menlo, Consolas, monospace;
      background: #fffdf8;
    }
    textarea { min-height: 260px; }
    .meta {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 8px 16px;
      margin-bottom: 12px;
      font-size: 14px;
    }
    .pill {
      display: inline-block;
      padding: 4px 8px;
      border-radius: 999px;
      font-size: 12px;
      border: 1px solid var(--line);
      background: #f7f1e6;
    }
    .ok { color: var(--ok); }
    .warn { color: var(--warn); }
    .bad { color: var(--bad); }
    .actions {
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      margin-top: 12px;
    }
    button, .btn {
      border: 0;
      border-radius: 999px;
      padding: 10px 16px;
      background: var(--accent);
      color: white;
      cursor: pointer;
      text-decoration: none;
      font-weight: 600;
    }
    .btn.secondary {
      background: #d9ccb4;
      color: var(--ink);
    }
    pre {
      white-space: pre-wrap;
      word-break: break-word;
      background: #fffdf8;
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 12px;
      overflow: auto;
      max-height: 260px;
    }
    .stack { display: grid; gap: 12px; }
    .issues {
      display: grid;
      gap: 8px;
    }
    .issues table {
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
      background: #fffdf8;
    }
    .issues th, .issues td {
      border: 1px solid var(--line);
      padding: 6px;
      vertical-align: top;
    }
    @media (max-width: 1100px) {
      .grid {
        grid-template-columns: 1fr;
      }
      .image-viewport {
        max-height: 65vh;
      }
      .image-viewport img {
        width: 130%;
      }
    }
  </style>
</head>
<body>
  <header>
    <div>
      <div class="muted">Census Review Tool</div>
      <h2 style="margin: 4px 0 0 0;">{{ task.label }}</h2>
    </div>
    <div class="muted">Task {{ index + 1 }} / {{ total }}</div>
  </header>
  <main class="grid">
    <section class="card">
      <div class="meta">
        <div><strong>Image</strong><br><span class="muted">{{ task.image_path }}</span></div>
        <div><strong>Result JSON</strong><br><span class="muted">{{ task.result_path or 'none' }}</span></div>
        <div><strong>Acceptance</strong><br>
          {% if task.acceptance_decision %}
            <span class="pill {{ 'ok' if task.acceptance_decision == 'accepted_high_confidence' else 'warn' }}">
              {{ task.acceptance_decision }}
            </span>
          {% else %}
            <span class="pill bad">missing</span>
          {% endif %}
        </div>
        <div><strong>Reviewed</strong><br>
          {% if saved.status %}
            <span class="pill {{ 'ok' if saved.status == 'verified' else 'warn' if saved.status == 'corrected' else 'bad' }}">{{ saved.status }}</span>
          {% else %}
            <span class="pill">unreviewed</span>
          {% endif %}
        </div>
      </div>
      <div class="image-toolbar">
        <button type="button" class="btn secondary" onclick="adjustZoom(0.8)">- Zoom</button>
        <button type="button" class="btn secondary" onclick="adjustZoom(1.25)">+ Zoom</button>
        <button type="button" class="btn secondary" onclick="resetZoom()">Reset</button>
      </div>
      <div class="image-viewport" id="imageViewport">
        <img id="reviewImage" src="{{ url_for('image_file', task_id=index) }}" alt="census image">
      </div>
      <div class="actions">
        {% if index > 0 %}
          <a class="btn secondary" href="{{ url_for('review_task', task_id=index-1) }}">Previous</a>
        {% endif %}
        {% if index + 1 < total %}
          <a class="btn secondary" href="{{ url_for('review_task', task_id=index+1) }}">Next</a>
        {% endif %}
      </div>
    </section>
    <section class="stack">
      {% for block in result_blocks %}
        <div class="card">
          <h3 style="margin-top:0;">{{ block.title }}</h3>
          <div class="meta">
            <div><strong>Decision</strong><br>
              {% if block.acceptance_decision %}
                <span class="pill {{ 'ok' if block.acceptance_decision == 'accepted_high_confidence' else 'warn' }}">{{ block.acceptance_decision }}</span>
              {% else %}
                <span class="pill">none</span>
              {% endif %}
            </div>
            <div><strong>Source</strong><br><span class="muted">{{ block.path or 'none' }}</span></div>
          </div>
          <div class="actions">
            {% if block.files.json %}
              <a class="btn secondary" href="{{ url_for('result_file', task_id=index, kind=block.kind, ext='json') }}">Open JSON</a>
            {% endif %}
            {% if block.files.csv %}
              <a class="btn secondary" href="{{ url_for('result_file', task_id=index, kind=block.kind, ext='csv') }}">Open CSV</a>
            {% endif %}
            {% if block.files.xlsx %}
              <a class="btn secondary" href="{{ url_for('result_file', task_id=index, kind=block.kind, ext='xlsx') }}">Open XLSX</a>
            {% endif %}
          </div>
          <div>{{ block.table_html|safe }}</div>
          <details style="margin-top:12px;">
            <summary>Raw JSON</summary>
            <pre>{{ block.raw_json }}</pre>
          </details>
        </div>
      {% endfor %}
      <div class="card">
        <h3 style="margin-top:0;">Review</h3>
        <form method="post" action="{{ url_for('save_task', task_id=index) }}">
          <label>Status</label>
          <select name="status">
            {% for choice in ['unreviewed', 'verified', 'corrected', 'flagged'] %}
              <option value="{{ choice }}" {% if saved.status == choice %}selected{% endif %}>{{ choice }}</option>
            {% endfor %}
          </select>
          <div style="height:10px;"></div>
          <label>Notes</label>
          <textarea name="notes" style="min-height:120px;">{{ saved.notes }}</textarea>
          <div style="height:10px;"></div>
          <label>Issues Found</label>
          <div class="issues">
            {% if saved.issues %}
              <table>
                <tr>
                  <th>Section</th>
                  <th>Row</th>
                  <th>Group</th>
                  <th>Column</th>
                  <th>Observed</th>
                  <th>Correct</th>
                  <th>Notes</th>
                </tr>
                {% for issue in saved.issues %}
                  <tr>
                    <td>{{ issue.section }}</td>
                    <td>{{ issue.row_label }}</td>
                    <td>{{ issue.group_name }}</td>
                    <td>{{ issue.column_name }}</td>
                    <td>{{ issue.observed_value }}</td>
                    <td>{{ issue.correct_value }}</td>
                    <td>{{ issue.notes }}</td>
                  </tr>
                {% endfor %}
              </table>
            {% else %}
              <div class="muted">No issue reports saved yet.</div>
            {% endif %}
            <input name="issue_section" placeholder="Section">
            <input name="issue_row_label" placeholder="Row / district / age label">
            <input name="issue_group_name" placeholder="Group">
            <input name="issue_column_name" placeholder="Column (persons/males/females)">
            <input name="issue_observed_value" placeholder="Observed value">
            <input name="issue_correct_value" placeholder="Correct value">
            <textarea name="issue_notes" style="min-height:90px;" placeholder="Why this is wrong / what you want fixed"></textarea>
          </div>
          <div style="height:10px;"></div>
          <label>Reviewed JSON/Data</label>
          <textarea name="reviewed_json">{{ reviewed_json }}</textarea>
          <div class="actions">
            <button type="submit">Save Review</button>
          </div>
        </form>
      </div>
    </section>
  </main>
  <script>
    let imageZoom = 1;
    function applyZoom() {
      const img = document.getElementById("reviewImage");
      if (!img) return;
      img.style.transform = `scale(${imageZoom})`;
    }
    function adjustZoom(multiplier) {
      imageZoom = Math.max(0.5, Math.min(4, imageZoom * multiplier));
      applyZoom();
    }
    function resetZoom() {
      imageZoom = 1;
      applyZoom();
    }
    window.addEventListener("load", applyZoom);
  </script>
</body>
</html>
"""


def build_tasks(root, results_roots, review_queue_path=None, name_filter=""):
    tasks = []
    if review_queue_path:
        seen = set()
        with open(review_queue_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                image_path = item.get("image")
                if not image_path or image_path in seen:
                    continue
                seen.add(image_path)
                tasks.append(_task_from_image(Path(image_path), results_roots))
    else:
        for image in discover_images(root):
            tasks.append(_task_from_image(image, results_roots))

    if name_filter:
        nf = name_filter.lower()
        tasks = [t for t in tasks if nf in t["label"].lower() or nf in t["image_path"].lower()]
    return tasks


def _result_files_from_json(result_path):
    if not result_path:
        return {"json": "", "csv": "", "xlsx": ""}
    result_path = Path(result_path)
    stem = result_path.name.replace("_oneshot.json", "_oneshot")
    base = result_path.parent / stem
    files = {"json": str(result_path)}
    for ext in ("csv", "xlsx"):
        candidate = base.with_suffix(f".{ext}")
        files[ext] = str(candidate) if candidate.exists() else ""
    return files


def _task_from_image(image_path, results_roots):
    results = find_all_results_for_image(image_path, results_roots)
    primary = results.get("oneshot")
    result = primary["data"] if primary else None
    result_path = primary["path"] if primary else ""
    acceptance = result.get("acceptance", {}) if result else {}
    result_files = {}
    for kind, payload in results.items():
        result_files[kind] = _result_files_from_json(payload["path"])
    return {
        "label": str(image_path.relative_to(Path.cwd())) if image_path.is_relative_to(Path.cwd()) else image_path.name,
        "image_path": str(image_path),
        "result_path": str(result_path) if result_path else "",
        "result_files": result_files,
        "results": results,
        "result": result,
        "acceptance_decision": acceptance.get("decision"),
        "acceptance": acceptance,
    }


def render_result_tables(result):
    if not result:
        return "<p class='muted'>No existing parsed output found for this image.</p>"

    sections = result.get("data", [])
    if not sections:
        return "<p class='muted'>Result JSON exists, but it has no parsed sections.</p>"

    parts = []
    for section in sections:
        rows = section.get("rows", [])
        if not rows:
            continue
        group_names = []
        for row in rows:
            for key, val in row.items():
                if key == "age":
                    continue
                if key == "district":
                    continue
                if isinstance(val, dict) and key not in group_names:
                    group_names.append(key)

        parts.append(f"<h4>{escape(section.get('name', 'Section'))}</h4>")
        table = ["<div style='overflow:auto'><table style='border-collapse:collapse;width:100%;font-size:13px'>"]
        header = ["<tr>"]
        if any("district" in row for row in rows):
            header.append("<th style='border:1px solid #d8cdb9;padding:6px;background:#f7f1e6'>District</th>")
        if any("age" in row for row in rows):
            header.append("<th style='border:1px solid #d8cdb9;padding:6px;background:#f7f1e6'>Age/Row</th>")
        for group in group_names:
            header.append(f"<th colspan='3' style='border:1px solid #d8cdb9;padding:6px;background:#efe3ca'>{escape(group)}</th>")
        header.append("</tr>")
        table.append("".join(header))

        sub = ["<tr>"]
        if any("district" in row for row in rows):
            sub.append("<th style='border:1px solid #d8cdb9;padding:6px;background:#fbf7ef'></th>")
        if any("age" in row for row in rows):
            sub.append("<th style='border:1px solid #d8cdb9;padding:6px;background:#fbf7ef'></th>")
        for _ in group_names:
            for col in ("Persons", "Males", "Females"):
                sub.append(f"<th style='border:1px solid #d8cdb9;padding:6px;background:#fbf7ef'>{col}</th>")
        sub.append("</tr>")
        table.append("".join(sub))

        for row in rows:
            body = ["<tr>"]
            if any("district" in r for r in rows):
                body.append(f"<td style='border:1px solid #d8cdb9;padding:6px'>{escape(str(row.get('district', '') or ''))}</td>")
            if any("age" in r for r in rows):
                body.append(f"<td style='border:1px solid #d8cdb9;padding:6px'>{escape(str(row.get('age', '') or ''))}</td>")
            for group in group_names:
                gdata = row.get(group, {})
                for key in ("persons", "males", "females"):
                    val = gdata.get(key) if isinstance(gdata, dict) else ""
                    body.append(f"<td style='border:1px solid #d8cdb9;padding:6px;text-align:right'>{escape(str(val if val is not None else ''))}</td>")
            body.append("</tr>")
            table.append("".join(body))
        table.append("</table></div>")
        parts.append("".join(table))

    return "".join(parts) if parts else "<p class='muted'>Could not render result as a table.</p>"


def annotation_path(annotations_root, task):
    image_path = Path(task["image_path"])
    rel = image_path.relative_to(Path.cwd() / "age_tables") if image_path.is_relative_to(Path.cwd() / "age_tables") else Path(image_path.name)
    return Path(annotations_root) / rel.with_suffix(".review.json")


def load_saved_annotation(annotations_root, task):
    path = annotation_path(annotations_root, task)
    if not path.exists():
        return {"status": "", "notes": "", "reviewed_json": "", "issues": []}
    with open(path) as f:
        data = json.load(f)
    return {
        "status": data.get("status", ""),
        "notes": data.get("notes", ""),
        "reviewed_json": json.dumps(data.get("reviewed_data"), indent=2, ensure_ascii=True) if data.get("reviewed_data") is not None else "",
        "issues": data.get("issues", []),
    }


def create_app(tasks, annotations_root):
    app = Flask(__name__)
    annotations_root = Path(annotations_root)
    annotations_root.mkdir(parents=True, exist_ok=True)

    @app.route("/")
    def home():
        return redirect(url_for("review_task", task_id=0))

    @app.route("/image/<int:task_id>")
    def image_file(task_id):
        if task_id < 0 or task_id >= len(tasks):
            abort(404)
        return send_file(tasks[task_id]["image_path"])

    @app.route("/result/<int:task_id>/<kind>/<ext>")
    def result_file(task_id, kind, ext):
        if task_id < 0 or task_id >= len(tasks):
            abort(404)
        path = tasks[task_id].get("result_files", {}).get(kind, {}).get(ext)
        if not path or not Path(path).exists():
            abort(404)
        return send_file(path)

    @app.route("/task/<int:task_id>")
    def review_task(task_id):
        if task_id < 0 or task_id >= len(tasks):
            abort(404)
        task = tasks[task_id]
        saved = load_saved_annotation(annotations_root, task)
        result_blocks = []
        payload = task.get("results", {}).get("oneshot")
        if payload:
            data = payload["data"]
            acceptance = data.get("acceptance", {})
            result_blocks.append({
                "kind": "oneshot",
                "title": "Pipeline Output: One-Shot",
                "path": payload["path"],
                "files": task.get("result_files", {}).get("oneshot", {}),
                "acceptance_decision": acceptance.get("decision"),
                "table_html": render_result_tables(data),
                "raw_json": json.dumps(data, indent=2, ensure_ascii=True),
            })
        reviewed_json = saved["reviewed_json"]
        if not reviewed_json and task.get("result"):
            reviewed_json = json.dumps(task["result"].get("data"), indent=2, ensure_ascii=True)
        return render_template_string(
            APP_HTML,
            task=task,
            index=task_id,
            total=len(tasks),
            saved=saved,
            result_blocks=result_blocks,
            reviewed_json=reviewed_json,
        )

    @app.route("/save/<int:task_id>", methods=["POST"])
    def save_task(task_id):
        if task_id < 0 or task_id >= len(tasks):
            abort(404)
        task = tasks[task_id]
        raw_reviewed = request.form.get("reviewed_json", "").strip()
        reviewed_data = None
        parse_error = None
        if raw_reviewed:
            try:
                reviewed_data = json.loads(raw_reviewed)
            except json.JSONDecodeError as exc:
                parse_error = str(exc)
        existing = load_saved_annotation(annotations_root, task)
        issues = list(existing.get("issues", []))
        issue_fields = {
            "section": request.form.get("issue_section", "").strip(),
            "row_label": request.form.get("issue_row_label", "").strip(),
            "group_name": request.form.get("issue_group_name", "").strip(),
            "column_name": request.form.get("issue_column_name", "").strip(),
            "observed_value": request.form.get("issue_observed_value", "").strip(),
            "correct_value": request.form.get("issue_correct_value", "").strip(),
            "notes": request.form.get("issue_notes", "").strip(),
        }
        if any(issue_fields.values()):
            issues.append(issue_fields)

        payload = {
            "image_path": task["image_path"],
            "result_path": task["result_path"],
            "status": request.form.get("status", "unreviewed"),
            "notes": request.form.get("notes", ""),
            "reviewed_data": reviewed_data,
            "parse_error": parse_error,
            "acceptance": task.get("acceptance"),
            "issues": issues,
        }
        out_path = annotation_path(annotations_root, task)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)
        return redirect(url_for("review_task", task_id=task_id))

    return app


def main():
    parser = argparse.ArgumentParser(description="Local census review app")
    parser.add_argument("--root", default="age_tables",
                        help="Image root to review")
    parser.add_argument("--results", default="results",
                        help="Comma-separated results roots used to locate extracted JSON")
    parser.add_argument("--annotations", default="review_annotations",
                        help="Directory for saved human reviews")
    parser.add_argument("--review-queue",
                        help="Optional JSONL review queue from batch_production")
    parser.add_argument("--filter", default="",
                        help="Substring filter for image/task names")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5001)
    args = parser.parse_args()

    tasks = build_tasks(args.root, parse_results_roots(args.results), args.review_queue, args.filter)
    if not tasks:
        raise SystemExit("No review tasks found.")

    app = create_app(tasks, args.annotations)
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
