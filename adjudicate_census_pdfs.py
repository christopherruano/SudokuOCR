"""Manual adjudication app for census PDFs.

Reads an audit manifest, renders probe pages for each PDF, and lets a human
mark whether the volume has age-table content.

Usage:
    python3 adjudicate_census_pdfs.py --manifest /tmp/census_audit_full.json
    python3 adjudicate_census_pdfs.py --manifest /tmp/census_audit_full.json --only unresolved
"""

import argparse
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

from flask import Flask, abort, redirect, render_template_string, request, send_file, url_for


APP_HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>PDF Adjudication</title>
  <style>
    :root {
      --bg: #f2efe8;
      --card: #fffdf8;
      --ink: #201c17;
      --muted: #746d63;
      --line: #ddd2c1;
      --accent: #8a3b1e;
      --ok: #2c6b3b;
      --bad: #8a2b2b;
      --warn: #9a6c16;
    }
    body {
      margin: 0;
      font-family: Georgia, serif;
      background: linear-gradient(180deg, #ece3d2 0%, var(--bg) 100%);
      color: var(--ink);
    }
    header, main {
      max-width: 1600px;
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
      grid-template-columns: 360px 1fr;
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
    .stack { display: grid; gap: 14px; }
    .meta {
      display: grid;
      gap: 8px;
      font-size: 14px;
      line-height: 1.35;
    }
    .pill {
      display: inline-block;
      padding: 4px 8px;
      border-radius: 999px;
      border: 1px solid var(--line);
      background: #f7f1e6;
      font-size: 12px;
    }
    .page-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 14px;
    }
    .page-card img {
      width: 100%;
      height: auto;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: white;
    }
    textarea, input, select {
      width: 100%;
      box-sizing: border-box;
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 10px;
      background: #fffdf8;
      font: 14px/1.35 Menlo, monospace;
    }
    textarea { min-height: 120px; }
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
    .btn.secondary { background: #d9ccb4; color: var(--ink); }
    pre {
      white-space: pre-wrap;
      word-break: break-word;
      max-height: 220px;
      overflow: auto;
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 10px;
      background: #fffdf8;
      font: 12px/1.35 Menlo, monospace;
    }
    @media (max-width: 1100px) {
      .grid { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <header>
    <div>
      <div class="muted">Census PDF Adjudication</div>
      <h2 style="margin:4px 0 0 0;">{{ item.filename }}</h2>
    </div>
    <div class="muted">Item {{ index + 1 }} / {{ total }}</div>
  </header>
  <main class="grid">
    <section class="stack">
      <div class="card">
        <div class="meta">
          <div><strong>Path</strong><br><span class="muted">{{ item.file }}</span></div>
          <div><strong>Initial Classification</strong><br><span class="pill">{{ item.classification }}</span></div>
          <div><strong>Year / Province</strong><br>{{ item.year or '?' }} / {{ item.province or '?' }}</div>
          <div><strong>Pages / Size</strong><br>{{ item.pages or '?' }} pages / {{ item.size_mb or '?' }} MB</div>
          <div><strong>Saved Decision</strong><br>
            {% if saved.decision %}
              <span class="pill">{{ saved.decision }}</span>
            {% else %}
              <span class="pill">unreviewed</span>
            {% endif %}
          </div>
        </div>
        <div class="actions">
          {% if index > 0 %}
            <a class="btn secondary" href="{{ url_for('review_item', item_id=index-1) }}">Previous</a>
          {% endif %}
          {% if index + 1 < total %}
            <a class="btn secondary" href="{{ url_for('review_item', item_id=index+1) }}">Next</a>
          {% endif %}
        </div>
      </div>

      <div class="card">
        <h3 style="margin-top:0;">Audit Signals</h3>
        <pre>{{ pretty_hits }}</pre>
      </div>

      <div class="card">
        <h3 style="margin-top:0;">Title Excerpt</h3>
        <pre>{{ item.title_excerpt or '' }}</pre>
      </div>

      <div class="card">
        <h3 style="margin-top:0;">Adjudicate</h3>
        <form method="post" action="{{ url_for('save_item', item_id=index) }}">
          <label>Decision</label>
          <select name="decision">
            {% for option in ['has_age_tables', 'no_age_tables', 'uncertain'] %}
              <option value="{{ option }}" {% if saved.decision == option %}selected{% endif %}>{{ option }}</option>
            {% endfor %}
          </select>
          <div style="height:10px;"></div>
          <label>Notes</label>
          <textarea name="notes">{{ saved.notes or '' }}</textarea>
          <div class="actions">
            <button type="submit">Save</button>
          </div>
        </form>
      </div>
    </section>

    <section class="stack">
      <div class="card">
        <h3 style="margin-top:0;">Probe Pages</h3>
        <div class="page-grid">
          {% for page in pages %}
            <div class="page-card">
              <div class="muted" style="margin-bottom:6px;">Page {{ page }}</div>
              <a href="{{ url_for('page_image', item_id=index, page=page) }}" target="_blank">
                <img src="{{ url_for('page_image', item_id=index, page=page) }}" alt="page {{ page }}">
              </a>
            </div>
          {% endfor %}
        </div>
      </div>
    </section>
  </main>
</body>
</html>
"""


def probe_pages(page_count):
    if not page_count or page_count <= 0:
        return [1]
    pages = list(range(1, min(page_count, 3) + 1))
    pages.extend(p for p in (8, 10, 12) if p <= page_count)
    if page_count > 20:
        pages.extend([max(1, page_count // 4), max(1, page_count // 2), max(1, (3 * page_count) // 4)])
    return sorted(set(pages))


def render_page(pdf_path, page_number, cache_dir):
    pdf_path = Path(pdf_path)
    digest = hashlib.sha1(f"{pdf_path}:{page_number}".encode("utf-8")).hexdigest()[:16]
    prefix = Path(cache_dir) / digest
    png = Path(str(prefix) + ".png")
    if png.exists():
        return png
    cmd = [
        "pdftoppm", "-png", "-singlefile",
        "-f", str(page_number), "-l", str(page_number),
        str(pdf_path), str(prefix),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0 or not png.exists():
        raise RuntimeError(result.stderr.strip() or f"Failed to render page {page_number}")
    return png


def load_manifest(path):
    with open(path) as fh:
        data = json.load(fh)
    return data["pdfs"]


def select_items(rows, mode):
    if mode == "all":
        return rows
    if mode == "unresolved":
        return [r for r in rows if r["classification"] in {"unreadable", "no_age_signal", "table_volume_only", "age_table_candidate"}]
    if mode == "candidates":
        return [r for r in rows if r["classification"] in {"age_table_candidate", "table_volume_only"}]
    raise ValueError(mode)


def annotation_path(item, annotations_dir):
    key = hashlib.sha1(item["file"].encode("utf-8")).hexdigest()
    return Path(annotations_dir) / f"{key}.review.json"


def load_annotation(item, annotations_dir):
    path = annotation_path(item, annotations_dir)
    if not path.exists():
        return {}
    with open(path) as fh:
        return json.load(fh)


def save_annotation(item, annotations_dir, payload):
    path = annotation_path(item, annotations_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Manual adjudication app for census PDFs")
    parser.add_argument("--manifest", required=True, help="Audit manifest JSON")
    parser.add_argument("--only", default="unresolved", choices=["all", "unresolved", "candidates"],
                        help="Which manifest subset to review")
    parser.add_argument("--annotations", default="review_annotations_pdf",
                        help="Where to save adjudication files")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5001)
    args = parser.parse_args()

    rows = select_items(load_manifest(args.manifest), args.only)
    cache_dir = Path(tempfile.gettempdir()) / "census_pdf_review_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    app = Flask(__name__)

    @app.route("/")
    def home():
        return redirect(url_for("review_item", item_id=0))

    @app.route("/item/<int:item_id>")
    def review_item(item_id):
        if item_id < 0 or item_id >= len(rows):
            abort(404)
        item = rows[item_id]
        saved = load_annotation(item, args.annotations)
        pages = probe_pages(item.get("pages") or 0)
        pretty_hits = json.dumps({k: v for k, v in item.get("keyword_hits", {}).items() if v}, indent=2)
        return render_template_string(
            APP_HTML,
            item=item,
            saved=saved,
            pages=pages,
            pretty_hits=pretty_hits,
            index=item_id,
            total=len(rows),
        )

    @app.route("/item/<int:item_id>/save", methods=["POST"])
    def save_item(item_id):
        if item_id < 0 or item_id >= len(rows):
            abort(404)
        item = rows[item_id]
        payload = {
            "file": item["file"],
            "filename": item["filename"],
            "decision": request.form.get("decision", "uncertain"),
            "notes": request.form.get("notes", "").strip(),
        }
        save_annotation(item, args.annotations, payload)
        next_id = min(item_id + 1, len(rows) - 1)
        return redirect(url_for("review_item", item_id=next_id))

    @app.route("/item/<int:item_id>/page/<int:page>.png")
    def page_image(item_id, page):
        if item_id < 0 or item_id >= len(rows):
            abort(404)
        item = rows[item_id]
        png = render_page(item["file"], page, cache_dir)
        return send_file(png, mimetype="image/png")

    print(f"Loaded {len(rows)} items from {args.manifest}")
    print(f"Annotations: {args.annotations}")
    print(f"Open: http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
