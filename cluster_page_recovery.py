#!/usr/bin/env python3
"""Resolve and render missing census PDF pages on the machine with PDF access.

Use this on FASRC or another machine that can read the original PDFs. It has two
jobs:

1. Resolve unplaced local image hashes by brute-forcing
   sha1(canonical_pdf_path + ":" + pdf_page_number).
2. Render requested missing PDF pages into PNGs named with the same deterministic
   image hash, preserving the existing pipeline convention.

If running somewhere other than the original cluster path, pass both:
  --pdf-root /actual/local/pdf/root
  --canonical-pdf-root /n/home11/cruano/census_pdfs

The relative path below --pdf-root is appended to --canonical-pdf-root for hash
calculation.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import re
import shutil
import subprocess
import sys
from pathlib import Path
from posixpath import join as posix_join
from typing import Any

csv.field_size_limit(10**8)


def sha_page(canonical_pdf_path: str, page: int) -> str:
    return hashlib.sha1(f"{canonical_pdf_path}:{page}".encode("utf-8")).hexdigest()


def safe_slug(text: str, limit: int = 80) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "-", text or "").strip("-")
    return text[:limit] or "table"


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def command_exists(name: str) -> bool:
    return shutil.which(name) is not None


def pdf_page_count(pdf_path: Path) -> int | None:
    if not command_exists("pdfinfo"):
        return None
    proc = subprocess.run(["pdfinfo", str(pdf_path)], capture_output=True, text=True)
    if proc.returncode != 0:
        return None
    for line in proc.stdout.splitlines():
        if line.lower().startswith("pages:"):
            try:
                return int(line.split(":", 1)[1].strip())
            except Exception:
                return None
    return None


def discover_pdfs(pdf_root: Path, canonical_pdf_root: str | None) -> list[dict[str, Any]]:
    rows = []
    for actual in sorted(pdf_root.rglob("*.pdf")):
        if actual.name.startswith("._"):
            continue
        if canonical_pdf_root:
            try:
                canonical = posix_join(canonical_pdf_root.rstrip("/"), *actual.relative_to(pdf_root).parts)
            except ValueError:
                canonical = str(actual)
        else:
            canonical = str(actual)
        rows.append({
            "actual_pdf_path": str(actual.resolve()),
            "canonical_pdf_path": str(canonical),
            "source_pdf_filename": actual.name,
            "page_count": pdf_page_count(actual),
        })
    return rows


def build_pdf_lookup(pdf_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    lookup = {}
    for row in pdf_rows:
        lookup[row["canonical_pdf_path"]] = row
        lookup.setdefault(row["source_pdf_filename"], row)
    return lookup


def render_page(actual_pdf: Path, page: int, out_png: Path) -> tuple[bool, str]:
    if out_png.exists():
        return True, "exists"
    if not command_exists("pdftoppm"):
        return False, "pdftoppm_not_found"
    out_png.parent.mkdir(parents=True, exist_ok=True)
    prefix = out_png.with_suffix("")
    proc = subprocess.run(
        ["pdftoppm", "-png", "-singlefile", "-f", str(page), "-l", str(page), str(actual_pdf), str(prefix)],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0 or not out_png.exists():
        return False, (proc.stderr.strip() or "render_failed")
    return True, "rendered"


def render_missing_pages(rows: list[dict[str, str]], pdf_lookup: dict[str, dict[str, Any]], out_root: Path) -> list[dict[str, Any]]:
    out_rows = []
    for row in rows:
        canonical_pdf = row.get("source_pdf_original", "")
        page_text = row.get("pdf_page_number", "")
        table_key = row.get("table_keys", "") or row.get("table_key", "")
        title = row.get("titles", "") or row.get("title", "")
        try:
            page = int(float(page_text))
        except Exception:
            out_rows.append({**row, "render_status": "skipped_no_page", "rendered_png": ""})
            continue
        pdf_row = pdf_lookup.get(canonical_pdf) or pdf_lookup.get(Path(canonical_pdf).name)
        if not pdf_row:
            out_rows.append({**row, "render_status": "skipped_pdf_not_found", "rendered_png": ""})
            continue
        image_hash = sha_page(canonical_pdf, page)
        table_dir = f"{table_key}_{safe_slug(title)}" if table_key else f"{Path(canonical_pdf).stem}_p{page}"
        out_png = out_root / "digitized_tables" / table_dir / f"{image_hash}.png"
        ok, status = render_page(Path(pdf_row["actual_pdf_path"]), page, out_png)
        out_rows.append({
            **row,
            "image_hash": image_hash,
            "actual_pdf_path": pdf_row["actual_pdf_path"],
            "canonical_pdf_path": canonical_pdf,
            "render_status": status if ok else f"failed:{status}",
            "rendered_png": str(out_png),
        })
    return out_rows


def resolve_unplaced_images(rows: list[dict[str, str]], pdf_rows: list[dict[str, Any]], max_pages: int) -> list[dict[str, Any]]:
    targets = {row.get("image_hash", "") for row in rows if row.get("image_hash")}
    targets.discard("")
    if not targets:
        return []

    hash_hits: dict[str, dict[str, Any]] = {}
    for pdf_row in pdf_rows:
        page_count = pdf_row.get("page_count") or max_pages
        page_count = min(int(page_count), max_pages)
        canonical = pdf_row["canonical_pdf_path"]
        for page in range(1, page_count + 1):
            h = sha_page(canonical, page)
            if h in targets:
                hash_hits[h] = {
                    "resolved_source_pdf_original": canonical,
                    "resolved_actual_pdf_path": pdf_row["actual_pdf_path"],
                    "resolved_source_pdf_filename": pdf_row["source_pdf_filename"],
                    "resolved_pdf_page_number": page,
                    "resolve_status": "matched_by_hash",
                }
    out = []
    for row in rows:
        h = row.get("image_hash", "")
        hit = hash_hits.get(h)
        if hit:
            out.append({**row, **hit})
        else:
            out.append({**row, "resolve_status": "not_matched"})
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Render missing PDF pages and resolve unplaced image hashes.")
    parser.add_argument("--pdf-root", type=Path, required=True, help="Actual directory containing original PDFs.")
    parser.add_argument("--canonical-pdf-root", help="Canonical root used for hashing, e.g. /n/home11/cruano/census_pdfs.")
    parser.add_argument("--missing-pages-csv", type=Path, help="CSV like table_vii_missing_pages.csv or missing_pdf_pages.csv.")
    parser.add_argument("--page-provenance-csv", type=Path, help="page_provenance_dictionary.csv for resolving unplaced image hashes.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--render-missing", action="store_true")
    parser.add_argument("--resolve-unplaced", action="store_true")
    parser.add_argument("--max-pages", type=int, default=2000)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pdf_rows = discover_pdfs(args.pdf_root, args.canonical_pdf_root)
    if not pdf_rows:
        print(f"No PDFs found under {args.pdf_root}", file=sys.stderr)
        sys.exit(2)
    write_csv(args.out_dir / "pdf_inventory.csv", pdf_rows, ["actual_pdf_path", "canonical_pdf_path", "source_pdf_filename", "page_count"])
    print(f"PDFs indexed: {len(pdf_rows)}")
    print(f"PDF inventory: {args.out_dir / 'pdf_inventory.csv'}")

    pdf_lookup = build_pdf_lookup(pdf_rows)
    if args.render_missing:
        if not args.missing_pages_csv:
            raise SystemExit("--render-missing requires --missing-pages-csv")
        missing_rows = read_csv(args.missing_pages_csv)
        rendered = render_missing_pages(missing_rows, pdf_lookup, args.out_dir / "rendered_missing_pages")
        fields = list(rendered[0].keys()) if rendered else []
        write_csv(args.out_dir / "rendered_missing_pages_manifest.csv", rendered, fields)
        print(f"Missing-page rows requested: {len(missing_rows)}")
        print(f"Rendered manifest: {args.out_dir / 'rendered_missing_pages_manifest.csv'}")

    if args.resolve_unplaced:
        if not args.page_provenance_csv:
            raise SystemExit("--resolve-unplaced requires --page-provenance-csv")
        prov_rows = [r for r in read_csv(args.page_provenance_csv) if r.get("provenance_confidence") != "exact"]
        resolved = resolve_unplaced_images(prov_rows, pdf_rows, args.max_pages)
        fields = list(resolved[0].keys()) if resolved else []
        write_csv(args.out_dir / "resolved_unplaced_images.csv", resolved, fields)
        print(f"Unplaced image rows tested: {len(prov_rows)}")
        print(f"Resolved unplaced manifest: {args.out_dir / 'resolved_unplaced_images.csv'}")


if __name__ == "__main__":
    main()
