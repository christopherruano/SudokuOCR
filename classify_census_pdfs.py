"""Classify census PDFs from CRL/JSTOR archive by reading title pages.

Reads page 1 of each PDF to extract:
  - Census year, province/state, volume number
  - Whether it likely contains age tables (Imperial Tables, Part II)

No LLM calls — pure text extraction and pattern matching.

Usage:
    python3 classify_census_pdfs.py /Volumes/Expansion/census_pdfs/
    python3 classify_census_pdfs.py /Volumes/Expansion/census_pdfs/1901/
    python3 classify_census_pdfs.py /Volumes/Expansion/census_pdfs/ --output manifest.json
"""

import os
import re
import json
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed


def read_title_page(pdf_path):
    """Read the first few pages of a PDF using pdftotext (poppler).

    Much faster than pypdf for large scanned PDFs on USB drives —
    doesn't load the entire file into memory.
    Returns (text, n_pages) or (None, 0).
    """
    import subprocess

    # Get page count
    n_pages = 0
    try:
        info = subprocess.run(
            ["pdfinfo", str(pdf_path)],
            capture_output=True, text=True, timeout=30)
        for line in info.stdout.splitlines():
            if line.startswith("Pages:"):
                n_pages = int(line.split(":")[1].strip())
                break
    except Exception:
        pass

    # Extract first 5 pages of text
    try:
        result = subprocess.run(
            ["pdftotext", "-f", "1", "-l", "5", str(pdf_path), "-"],
            capture_output=True, text=True, timeout=60)
        text = result.stdout
        if text and text.strip():
            return text, n_pages
    except Exception:
        pass

    return None, n_pages


def classify_pdf(pdf_path):
    """Classify a single PDF by reading its title page."""
    text, n_pages = read_title_page(pdf_path)

    info = {
        "file": str(pdf_path),
        "filename": Path(pdf_path).name,
        "size_mb": round(Path(pdf_path).stat().st_size / (1024 * 1024), 1),
        "pages": n_pages,
        "year": None,
        "province": None,
        "volume": None,
        "title": None,
        "has_age_tables": False,
        "has_imperial_tables": False,
        "classification": "unknown",
        "raw_title": None,
    }

    if not text:
        info["classification"] = "unreadable"
        return info

    # The JSTOR/CRL first page has a standardized format:
    # "Census of India 1901. Vol. 14, Coorg : report and tables."
    info["raw_title"] = text[:500]

    # Extract title line
    title_match = re.search(
        r'Census\s+of\s+India\s+(\d{4})[.,]?\s*(.*?)(?:\n|Source:)',
        text, re.IGNORECASE | re.DOTALL)

    if title_match:
        info["year"] = int(title_match.group(1))
        rest = title_match.group(2).strip()
        info["title"] = f"Census of India {info['year']}. {rest}"

        # Extract volume number
        vol_match = re.search(r'Vol\.?\s*([IVXLC\d]+[-A-Za-z]*)', rest, re.IGNORECASE)
        if vol_match:
            info["volume"] = vol_match.group(1).strip()

        # Extract province/region name
        # Pattern: "Vol. XX, PROVINCE : description" or "Vol. XX-A PROVINCE"
        prov_match = re.search(
            r'Vol\.?\s*[IVXLC\d]+[-A-Za-z]*[,.\s]+([A-Z][A-Za-z\s&-]+?)(?:\s*:|$)',
            rest)
        if prov_match:
            info["province"] = prov_match.group(1).strip()
        else:
            # Try after volume: everything before the colon
            colon_match = re.search(r',\s*([^:]+?):', rest)
            if colon_match:
                info["province"] = colon_match.group(1).strip()

    # Fallback: try directory name as year
    parent = Path(pdf_path).parent.name
    if info["year"] is None and parent.isdigit():
        info["year"] = int(parent)

    # Check for age table indicators in title and table of contents
    text_lower = text.lower()

    # Imperial Tables typically contain the age tables (Table VIII)
    if any(kw in text_lower for kw in [
        "imperial tables", "imperial table",
        "part ii", "part 2",
        "tables", "report and tables",
    ]):
        info["has_imperial_tables"] = True

    # Direct age table indicators
    if any(kw in text_lower for kw in [
        "age, sex and civil condition",
        "age and sex",
        "table viii", "table no. viii",
        "table 8", "table no. 8",
        "age-distribution", "age distribution",
        "age, sex, and civil",
    ]):
        info["has_age_tables"] = True

    # Classify
    if info["has_age_tables"]:
        info["classification"] = "age_table_confirmed"
    elif info["has_imperial_tables"]:
        info["classification"] = "likely_has_age_tables"
    elif info["title"]:
        info["classification"] = "census_other"
    else:
        info["classification"] = "unknown"

    return info


def main():
    parser = argparse.ArgumentParser(
        description="Classify census PDFs by reading title pages (no LLM calls)")
    parser.add_argument("root", help="Directory containing census PDFs")
    parser.add_argument("--output", default="census_manifest.json",
                        help="Output manifest file (default: census_manifest.json)")
    parser.add_argument("--workers", type=int, default=4,
                        help="Concurrent PDF readers (default: 4)")
    args = parser.parse_args()

    # Discover all PDFs
    root = Path(args.root)
    pdfs = sorted(root.rglob("*.pdf"))
    print(f"Found {len(pdfs)} PDFs under {root}")

    if not pdfs:
        return

    # Classify in parallel
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(classify_pdf, p): p for p in pdfs}
        for i, future in enumerate(as_completed(futures)):
            pdf_path = futures[future]
            try:
                info = future.result()
                results.append(info)
                status = info["classification"]
                province = info.get("province") or "?"
                year = info.get("year") or "?"
                print(f"  [{i+1}/{len(pdfs)}] {pdf_path.name}: "
                      f"{year} {province} — {status}")
            except Exception as e:
                print(f"  [{i+1}/{len(pdfs)}] {pdf_path.name}: ERROR {e}")
                results.append({
                    "file": str(pdf_path),
                    "filename": pdf_path.name,
                    "classification": "error",
                    "error": str(e),
                })

    # Sort by year, province
    results.sort(key=lambda r: (r.get("year") or 0, r.get("province") or ""))

    # Summary
    by_class = {}
    for r in results:
        c = r["classification"]
        by_class.setdefault(c, []).append(r)

    print(f"\n{'='*60}")
    print("CLASSIFICATION SUMMARY")
    print(f"{'='*60}")
    for cls in ["age_table_confirmed", "likely_has_age_tables",
                "census_other", "unknown", "unreadable", "error"]:
        items = by_class.get(cls, [])
        if items:
            print(f"\n  {cls}: {len(items)} PDFs")
            for r in items[:10]:
                prov = r.get("province") or "?"
                yr = r.get("year") or "?"
                pg = r.get("pages") or "?"
                sz = r.get("size_mb") or "?"
                print(f"    {yr} {prov:<30} {pg:>5} pages  {sz:>8} MB  {r['filename']}")
            if len(items) > 10:
                print(f"    ... and {len(items) - 10} more")

    # Write manifest
    manifest = {
        "total_pdfs": len(results),
        "age_table_confirmed": len(by_class.get("age_table_confirmed", [])),
        "likely_has_age_tables": len(by_class.get("likely_has_age_tables", [])),
        "census_other": len(by_class.get("census_other", [])),
        "unknown": len(by_class.get("unknown", [])),
        "pdfs": results,
    }

    out_path = Path(args.output)
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest written to {out_path}")
    print(f"Age table candidates: "
          f"{manifest['age_table_confirmed'] + manifest['likely_has_age_tables']} PDFs")


if __name__ == "__main__":
    main()
