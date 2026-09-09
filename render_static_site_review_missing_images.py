#!/usr/bin/env python3
"""Render exact PDF pages for static-site update review rows missing images.

This is rendering only, not OCR. It reads the missing-image review rows, locates
the original PDFs on the expansion drive, renders the exact pdf_page values to
PNG, and writes a backfill manifest consumed by the local review browser.
"""

from __future__ import annotations

import csv
import glob
import json
import os
from collections import Counter

import fitz


ROOT = os.path.dirname(os.path.abspath(__file__))
PDF_ROOTS = [
    "/Volumes/Expansion/census_pdfs",
    os.path.expanduser("~/Downloads/census_pdfs"),
]
V4_DIR = os.path.join(
    ROOT,
    "export_final",
    "pdf_recovery_incoming_2026-06-16_all_ocr_component_assembly_v4_adjacency",
)
MISSING_CSV = os.path.join(V4_DIR, "static_site_update_review_rows_missing_images.csv")
OUT_DIR = os.path.join(V4_DIR, "review_missing_page_renders")
OUT_BACKFILL = os.path.join(V4_DIR, "static_site_update_review_image_backfill.csv")
OUT_SUMMARY = os.path.join(V4_DIR, "static_site_update_review_image_backfill_summary.json")

DPI = 200


def read_csv(path: str) -> list[dict[str, str]]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: str, rows: list[dict], fieldnames: list[str]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def parse_member_pages(raw: str) -> list[tuple[str, int]]:
    out = []
    for token in (raw or "").split(";"):
        if "@" not in token:
            continue
        sid, page = token.split("@", 1)
        sid = sid.strip()
        try:
            page_i = int(float(page.strip()))
        except Exception:
            continue
        if sid and page_i > 0:
            out.append((sid, page_i))
    return out


def find_pdf(source_id: str) -> str:
    name = f"saoa.crl.{source_id}.pdf"
    for root in PDF_ROOTS:
        if not os.path.exists(root):
            continue
        direct = glob.glob(os.path.join(root, "*", name))
        direct.extend(glob.glob(os.path.join(root, name)))
        if direct:
            return sorted(direct)[0]
    return ""


def render_page(pdf_path: str, source_id: str, pdf_page: int) -> tuple[str, str]:
    out_subdir = os.path.join(OUT_DIR, source_id)
    os.makedirs(out_subdir, exist_ok=True)
    out_path = os.path.join(out_subdir, f"{source_id}_p{pdf_page}.png")
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        return out_path, "already_present"

    with fitz.open(pdf_path) as doc:
        if pdf_page < 1 or pdf_page > doc.page_count:
            return "", f"page_out_of_range_pdf_has_{doc.page_count}"
        page = doc.load_page(pdf_page - 1)
        pix = page.get_pixmap(matrix=fitz.Matrix(DPI / 72, DPI / 72), alpha=False)
        pix.save(out_path)
    return out_path, "rendered"


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = read_csv(MISSING_CSV)
    pdf_cache: dict[str, str] = {}
    out_rows = []
    status_counts = Counter()
    rendered_pages = set()

    for row in rows:
        rendered_paths = []
        statuses = []
        pdf_paths = []
        for source_id, pdf_page in parse_member_pages(row.get("member_pdf_pages", "")):
            key = (source_id, pdf_page)
            pdf_path = pdf_cache.get(source_id)
            if pdf_path is None:
                pdf_path = find_pdf(source_id)
                pdf_cache[source_id] = pdf_path
            if not pdf_path:
                statuses.append(f"{source_id}@{pdf_page}:pdf_not_found")
                status_counts["pdf_not_found"] += 1
                continue
            pdf_paths.append(pdf_path)
            out_path, status = render_page(pdf_path, source_id, pdf_page)
            statuses.append(f"{source_id}@{pdf_page}:{status}")
            status_counts[status] += 1
            if out_path:
                rendered_paths.append(out_path)
                rendered_pages.add(key)

        status = "ok" if rendered_paths and all(":rendered" in s or ":already_present" in s for s in statuses) else "incomplete"
        out_rows.append(
            {
                "review_index": row.get("review_index", ""),
                "proposed_table_id": row.get("proposed_table_id", ""),
                "adjacency_group_id": row.get("adjacency_group_id", ""),
                "member_pdf_pages": row.get("member_pdf_pages", ""),
                "rendered_image_paths": ";".join(rendered_paths),
                "source_pdf_paths": ";".join(dict.fromkeys(pdf_paths)),
                "render_status": status,
                "render_details": ";".join(statuses),
                "adjacency_primary_title": row.get("adjacency_primary_title", ""),
            }
        )

    fields = [
        "review_index",
        "proposed_table_id",
        "adjacency_group_id",
        "member_pdf_pages",
        "rendered_image_paths",
        "source_pdf_paths",
        "render_status",
        "render_details",
        "adjacency_primary_title",
    ]
    write_csv(OUT_BACKFILL, out_rows, fields)
    summary = {
        "missing_review_rows": len(rows),
        "backfill_rows": len(out_rows),
        "rows_with_rendered_images": sum(1 for r in out_rows if r["rendered_image_paths"]),
        "unique_rendered_pages": len(rendered_pages),
        "status_counts": dict(status_counts),
        "output_dir": OUT_DIR,
        "backfill_csv": OUT_BACKFILL,
    }
    with open(OUT_SUMMARY, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
