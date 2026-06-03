#!/usr/bin/env python3
"""Build browser pages for reviewing assembled tables against member images."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import os
import shutil
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps


DEFAULT_ASSEMBLED_JSONL = Path("export_final/assembled_all_read_improved/strong_assembled_tables.jsonl")
DEFAULT_OUTPUT_DIR = Path("export_final/assembled_all_read_review/image_table_review_strong")
DEFAULT_DIGITIZED_LOCAL_ROOT = Path("export/digitized_tables_referenced/digitized_tables")
CLUSTER_DIGITIZED_PREFIX = "/n/home11/cruano/census_toc/digitized_tables/"


def normalize_text(value: object) -> str:
    return " ".join(str(value or "").strip().split())


def slugify(value: object, max_len: int = 80) -> str:
    text = normalize_text(value).lower()
    text = "".join(ch if ch.isalnum() else "-" for ch in text)
    text = "-".join(part for part in text.split("-") if part)
    return (text or "table")[:max_len]


def short_hash(value: object) -> str:
    return hashlib.sha1(str(value or "").encode("utf-8")).hexdigest()[:12]


def load_groups(path: Path) -> list[dict[str, Any]]:
    groups = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                groups.append(json.loads(line))
    return groups


def local_image_path(path_text: object, local_root: Path) -> Path | None:
    text = str(path_text or "").strip()
    if not text:
        return None
    path = Path(text)
    if path.exists():
        return path
    if CLUSTER_DIGITIZED_PREFIX in text:
        rel = text.split(CLUSTER_DIGITIZED_PREFIX, 1)[1]
        candidate = local_root / rel
        if candidate.exists():
            return candidate
    if "/digitized_tables/" in text:
        rel = text.split("/digitized_tables/", 1)[1]
        candidate = local_root / rel
        if candidate.exists():
            return candidate
    return None


def relpath(path: Path, start: Path) -> str:
    return Path(os.path.relpath(Path(path).resolve(), start.resolve())).as_posix()


def make_thumbnail(src: Path, dest: Path, max_width: int, quality: int) -> None:
    if dest.exists() and dest.stat().st_mtime >= src.stat().st_mtime:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as img:
        img = ImageOps.exif_transpose(img)
        if img.mode not in {"RGB", "L"}:
            img = img.convert("RGB")
        if img.width > max_width:
            ratio = max_width / img.width
            new_size = (max_width, max(1, int(img.height * ratio)))
            img = img.resize(new_size, Image.Resampling.LANCZOS)
        img.save(dest, "JPEG", quality=quality, optimize=True)


def html_escape(value: object) -> str:
    return html.escape(str(value if value is not None else ""))


def value_display(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def flatten_section_rows(rows: list[dict[str, Any]]) -> list[list[Any]]:
    scalar_keys: list[str] = []
    value_keys: list[tuple[str, str]] = []
    scalar_seen = set()
    value_seen = set()

    for row in rows:
        for key, value in row.items():
            if isinstance(value, dict):
                for subkey in value:
                    pair = (str(key), str(subkey))
                    if pair not in value_seen:
                        value_seen.add(pair)
                        value_keys.append(pair)
            elif key not in scalar_seen:
                scalar_seen.add(key)
                scalar_keys.append(str(key))

    preferred_candidates = [
        "row_label",
        "age",
        "AGE",
        "Age",
        "RELIGION",
        "Religion",
        "Caste",
        "CASTE",
        "PROVINCE OR STATE.",
        "District",
        "__page__",
    ]
    preferred = [key for key in preferred_candidates if key in scalar_keys]
    scalar_keys = preferred + [key for key in scalar_keys if key not in preferred]

    header = scalar_keys + [f"{key} | {subkey}" for key, subkey in value_keys]
    output = [header]
    for row in rows:
        output.append(
            [row.get(key, "") for key in scalar_keys]
            + [
                (row.get(key) or {}).get(subkey, "") if isinstance(row.get(key), dict) else ""
                for key, subkey in value_keys
            ]
        )
    return output


def render_matrix(rows: list[list[Any]]) -> str:
    if not rows:
        return "<p class='muted'>No compiled rows.</p>"
    out = ["<div class='table-scroll'><table>"]
    for idx, row in enumerate(rows):
        tag = "th" if idx == 0 else "td"
        out.append("<tr>" + "".join(f"<{tag}>{html_escape(value_display(cell))}</{tag}>" for cell in row) + "</tr>")
    out.append("</table></div>")
    return "\n".join(out)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


CSS = """
:root { --ink:#181715; --muted:#746d62; --line:#ddd4c7; --paper:#fffaf0; --panel:#f3eadb; --accent:#245b4d; }
* { box-sizing:border-box; }
body { margin:0; font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color:var(--ink); background:#efe4d2; }
a { color:#174f82; }
.top { position:sticky; top:0; z-index:5; padding:12px 18px; background:#1d271f; color:#fff; display:flex; gap:16px; align-items:center; justify-content:space-between; }
.top a { color:#d9f2e9; }
.shell { display:grid; grid-template-columns:minmax(360px, 42vw) minmax(520px, 1fr); gap:0; min-height:calc(100vh - 52px); }
.images { background:#211f1b; color:#f8f0e5; padding:14px; overflow:auto; max-height:calc(100vh - 52px); position:sticky; top:52px; }
.compiled { padding:18px; overflow:auto; }
.card { background:var(--paper); border:1px solid var(--line); border-radius:14px; padding:14px; margin-bottom:16px; box-shadow:0 1px 4px rgba(0,0,0,.08); }
.image-card { background:#302d27; border:1px solid #555047; border-radius:12px; padding:10px; margin-bottom:14px; }
.image-card img { width:100%; height:auto; display:block; background:#fff; border:1px solid #4b463d; }
.caption { margin-top:8px; font-size:12px; color:#d5cab9; overflow-wrap:anywhere; }
.meta-grid { display:grid; grid-template-columns:180px minmax(0,1fr); gap:6px 12px; font-size:14px; }
.key { color:var(--muted); font-weight:700; }
.muted { color:var(--muted); }
.badge { display:inline-block; border:1px solid var(--line); border-radius:999px; padding:3px 8px; margin:2px 4px 2px 0; background:#fff; font-size:12px; }
h1 { font-size:22px; margin:0; }
h2 { font-size:18px; margin:0 0 10px; }
h3 { font-size:15px; margin:0 0 8px; }
.section-head { display:flex; justify-content:space-between; gap:12px; align-items:baseline; }
.table-scroll { max-height:70vh; overflow:auto; border:1px solid var(--line); background:#fff; }
table { border-collapse:collapse; font-size:12px; min-width:100%; }
th, td { border:1px solid #ded8cc; padding:4px 6px; white-space:nowrap; vertical-align:top; }
th { position:sticky; top:0; background:#e9f0eb; z-index:1; }
td:first-child, th:first-child { position:sticky; left:0; background:#faf6ee; z-index:2; }
th:first-child { z-index:3; background:#dfe9e2; }
.index { padding:18px; }
.index table { background:#fff; width:100%; }
.index th, .index td { white-space:normal; }
.summary { font-size:13px; color:#e9dfcf; }
@media (max-width: 1100px) { .shell { grid-template-columns:1fr; } .images { position:relative; top:0; max-height:none; } }
"""


def group_page_name(seq: int, group: dict[str, Any]) -> str:
    digest = short_hash(group.get("group_id", ""))
    return f"{seq:03d}_{slugify(group.get('province'))}_{slugify(group.get('year'))}_{slugify(group.get('table_family'))}_{digest}.html"


def render_group_page(
    group: dict[str, Any],
    seq: int,
    total: int,
    page_name: str,
    prev_name: str | None,
    next_name: str | None,
    output_dir: Path,
    thumb_dir: Path,
    local_root: Path,
    thumb_width: int,
    thumb_quality: int,
) -> dict[str, Any]:
    source_images = group.get("page_source_images") or []
    source_jsons = group.get("page_source_jsons") or []
    image_blocks = []
    for idx, image_text in enumerate(source_images, start=1):
        local = local_image_path(image_text, local_root)
        thumb_rel = ""
        original_rel = ""
        if local:
            thumb_path = thumb_dir / f"{short_hash(local)}.jpg"
            make_thumbnail(local, thumb_path, thumb_width, thumb_quality)
            thumb_rel = relpath(thumb_path, output_dir)
            original_rel = relpath(local, output_dir)
        caption_parts = [
            f"Page {idx}",
            source_jsons[idx - 1] if idx - 1 < len(source_jsons) else "",
        ]
        caption = " | ".join(part for part in caption_parts if part)
        if thumb_rel:
            image_html = f"<a href='{html_escape(original_rel)}' target='_blank'><img src='{html_escape(thumb_rel)}'></a>"
            open_link = f"<a href='{html_escape(original_rel)}' target='_blank'>Open original image</a>"
        else:
            image_html = "<div class='muted'>Image not found locally.</div>"
            open_link = html_escape(str(image_text))
        image_blocks.append(
            f"<div class='image-card'>{image_html}<div class='caption'>{html_escape(caption)}<br>{open_link}</div></div>"
        )

    report = group.get("assembled_constraint_report") or {}
    section_blocks = []
    for section_idx, section in enumerate(group.get("sections") or [], start=1):
        stats = section.get("stats") or {}
        matrix = flatten_section_rows(section.get("rows") or [])
        section_blocks.append(
            "<div class='card'>"
            "<div class='section-head'>"
            f"<h2>Section {section_idx}: {html_escape(section.get('name', ''))}</h2>"
            f"<span class='muted'>strategy={html_escape(section.get('merge_strategy', ''))}; "
            f"input_pages={html_escape(stats.get('input_pages', ''))}; "
            f"output_rows={html_escape(stats.get('output_rows', ''))}</span>"
            "</div>"
            f"{render_matrix(matrix)}"
            "</div>"
        )

    nav = ["<a href='index.html'>Index</a>"]
    if prev_name:
        nav.append(f"<a href='{html_escape(prev_name)}'>Previous</a>")
    if next_name:
        nav.append(f"<a href='{html_escape(next_name)}'>Next</a>")

    meta_pairs = [
        ("group_id", group.get("group_id", "")),
        ("province", group.get("province", "")),
        ("year", group.get("year", "")),
        ("table_family", group.get("table_family", "")),
        ("n_pages", group.get("n_pages", "")),
        ("overall_strategy", group.get("overall_strategy", "")),
        ("signals", " | ".join(group.get("signals") or [])),
        ("constraint checks", report.get("total_checks", "")),
        ("constraint failures", report.get("failed", "")),
        ("all passed", report.get("all_passed", "")),
        ("source_file", group.get("source_file", "")),
    ]
    meta_html = "\n".join(
        f"<div class='key'>{html_escape(key)}</div><div>{html_escape(value)}</div>"
        for key, value in meta_pairs
    )

    page_html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>{seq}. {html_escape(group.get("title", ""))}</title>
  <style>{CSS}</style>
</head>
<body>
  <div class="top">
    <div><strong>Assembled Table {seq}/{total}</strong></div>
    <div>{' · '.join(nav)}</div>
  </div>
  <div class="shell">
    <aside class="images">
      <h1>Scanned Pages</h1>
      <div class="summary">{len(source_images)} page image(s). Click an image to open the original PNG.</div>
      <br>
      {''.join(image_blocks) if image_blocks else "<p>No images listed for this group.</p>"}
    </aside>
    <main class="compiled">
      <div class="card">
        <h1>{html_escape(group.get("title", ""))}</h1>
        <p class="muted">This is the compiled output from all member pages in the assembled group.</p>
        <div class="meta-grid">{meta_html}</div>
      </div>
      {''.join(section_blocks) if section_blocks else "<div class='card'>No assembled sections.</div>"}
    </main>
  </div>
</body>
</html>
"""
    (output_dir / page_name).write_text(page_html, encoding="utf-8")

    return {
        "seq": seq,
        "page": page_name,
        "group_id": group.get("group_id", ""),
        "province": group.get("province", ""),
        "year": group.get("year", ""),
        "table_family": group.get("table_family", ""),
        "title": group.get("title", ""),
        "n_pages": group.get("n_pages", ""),
        "overall_strategy": group.get("overall_strategy", ""),
        "assembled_constraint_checks": report.get("total_checks", ""),
        "assembled_constraint_failures": report.get("failed", ""),
        "assembled_all_passed": report.get("all_passed", ""),
        "section_count": len(group.get("sections") or []),
        "image_count": len(source_images),
    }


def render_index(index_rows: list[dict[str, Any]], output_dir: Path) -> None:
    rows_html = []
    for row in index_rows:
        rows_html.append(
            "<tr>"
            f"<td><a href='{html_escape(row['page'])}'>{html_escape(row['seq'])}</a></td>"
            f"<td>{html_escape(row.get('province', ''))}</td>"
            f"<td>{html_escape(row.get('year', ''))}</td>"
            f"<td>{html_escape(row.get('table_family', ''))}</td>"
            f"<td><a href='{html_escape(row['page'])}'>{html_escape(row.get('title', ''))}</a></td>"
            f"<td>{html_escape(row.get('n_pages', ''))}</td>"
            f"<td>{html_escape(row.get('overall_strategy', ''))}</td>"
            f"<td>{html_escape(row.get('assembled_constraint_failures', ''))}</td>"
            f"<td>{html_escape(row.get('section_count', ''))}</td>"
            f"<td>{html_escape(row.get('image_count', ''))}</td>"
            "</tr>"
        )
    index_html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Assembled Table Image Review</title>
  <style>{CSS}</style>
</head>
<body>
  <div class="top"><strong>Assembled Table Image Review</strong><span>{len(index_rows)} groups</span></div>
  <main class="index">
    <div class="card">
      <h1>Compiled Multipage Tables</h1>
      <p class="muted">Open a row to see all member scanned pages beside the compiled assembled table.</p>
    </div>
    <table>
      <tr>
        <th>#</th><th>Province</th><th>Year</th><th>Family</th><th>Title</th>
        <th>Pages</th><th>Strategy</th><th>Failures</th><th>Sections</th><th>Images</th>
      </tr>
      {''.join(rows_html)}
    </table>
  </main>
</body>
</html>
"""
    (output_dir / "index.html").write_text(index_html, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build HTML pages for reviewing compiled assembled tables with all member images.")
    parser.add_argument("--assembled-jsonl", type=Path, default=DEFAULT_ASSEMBLED_JSONL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--local-image-root", type=Path, default=DEFAULT_DIGITIZED_LOCAL_ROOT)
    parser.add_argument("--thumbnail-width", type=int, default=1200)
    parser.add_argument("--thumbnail-quality", type=int, default=72)
    args = parser.parse_args()

    groups = sorted(
        load_groups(args.assembled_jsonl),
        key=lambda g: (
            str(g.get("province") or ""),
            str(g.get("year") or ""),
            str(g.get("table_family") or ""),
            str(g.get("title") or ""),
        ),
    )
    if args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    thumb_dir = args.output_dir / "thumbnails"

    page_names = [group_page_name(idx, group) for idx, group in enumerate(groups, start=1)]
    index_rows = []
    for idx, group in enumerate(groups, start=1):
        index_rows.append(
            render_group_page(
                group,
                idx,
                len(groups),
                page_names[idx - 1],
                page_names[idx - 2] if idx > 1 else None,
                page_names[idx] if idx < len(groups) else None,
                args.output_dir,
                thumb_dir,
                args.local_image_root,
                args.thumbnail_width,
                args.thumbnail_quality,
            )
        )

    render_index(index_rows, args.output_dir)
    write_csv(
        args.output_dir / "index.csv",
        index_rows,
        [
            "seq",
            "page",
            "group_id",
            "province",
            "year",
            "table_family",
            "title",
            "n_pages",
            "overall_strategy",
            "assembled_constraint_checks",
            "assembled_constraint_failures",
            "assembled_all_passed",
            "section_count",
            "image_count",
        ],
    )
    print(f"HTML review written: {args.output_dir / 'index.html'}")
    print(f"Groups included: {len(groups)}")
    print(f"Thumbnails: {thumb_dir}")


if __name__ == "__main__":
    main()
