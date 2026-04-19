"""Audit pipeline for census PDF discovery.

Combines:
  1. Full-text PDF extraction with normalized regex matching
  2. Province/year inventory cross-checks
  3. Optional Mistral OCR fallback for unreadable/unresolved PDFs

Usage:
    python3 audit_census_pipeline.py /Volumes/Expansion/census_pdfs
    python3 audit_census_pipeline.py /Volumes/Expansion/census_pdfs --mistral-fallback
"""

import argparse
import base64
import json
import os
import re
import subprocess
import tempfile
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

try:
    from mistralai import Mistral
except Exception:  # pragma: no cover - optional at runtime
    Mistral = None


EXPECTED_PROVINCES = {
    1901: [
        "Ajmer Merwara",
        "Andaman & Nicobar",
        "Assam",
        "Baluchistan",
        "Baroda",
        "Bengal",
        "Berar",
        "Bombay",
        "Burma",
        "Central India",
        "Central Provinces",
        "Cochin",
        "Coorg",
        "Gwalior",
        "Hyderabad",
        "Madras",
        "Mysore",
        "North Western Provinces Oudh",
        "Punjab",
        "Rajputana",
        "Travancore",
        "United Provinces",
    ],
}

PROVINCE_ALIASES = {
    "Ajmer-Merwara": "Ajmer Merwara",
    "Ajmer Merwara": "Ajmer Merwara",
    "Andaman and Nicobar": "Andaman & Nicobar",
    "Andaman & Nicobar": "Andaman & Nicobar",
    "Central India": "Central India",
    "Central Provinces": "Central Provinces",
    "Central Provinces and Berar": "Central Provinces",
    "Cochin": "Cochin",
    "Coorg": "Coorg",
    "Gwalior": "Gwalior",
    "Hyderabad": "Hyderabad",
    "Madras": "Madras",
    "Mysore": "Mysore",
    "North-Western Provinces and Oudh": "North Western Provinces Oudh",
    "North Western Provinces Oudh": "North Western Provinces Oudh",
    "North-Western Provinces Oudh": "North Western Provinces Oudh",
    "North Western Provinces and Oudh": "North Western Provinces Oudh",
    "N.W. Provinces and Oudh": "North Western Provinces Oudh",
    "Punjab": "Punjab",
    "Rajputana": "Rajputana",
    "Travancore": "Travancore",
    "United Provinces": "United Provinces",
    "United Provinces of Agra and Oudh": "United Provinces",
    "Bombay": "Bombay",
    "Burma": "Burma",
    "Bengal": "Bengal",
    "Baroda": "Baroda",
    "Berar": "Berar",
    "Assam": "Assam",
    "Baluchistan": "Baluchistan",
}


KEYWORD_GROUPS = {
    "strong_phrase": [
        re.compile(r"\bage\s*(?:,|\s|&|and|-)*sex\s*(?:,|\s|&|and|-)*(?:civil|marital)\b", re.I),
        re.compile(r"\bage\s*(?:,|\s|&|and|-)*civil\s+condition\b", re.I),
        re.compile(r"\bage\s*(?:,|\s|&|and|-)*marital\s+status\b", re.I),
        re.compile(r"\bcivil\s+condition\s+by\s+(?:religion\s+and\s+)?age\b", re.I),
        re.compile(r"\bcivil\s+condition\s+by\s+religion\b", re.I),
        re.compile(r"\bage\s+and\s+civil\s+condition\s+by\s+districts\b", re.I),
        re.compile(r"\bages?\s+of\s+the\s+population\b", re.I),
        re.compile(r"\bage\s+by\s+religion\b", re.I),
        re.compile(r"\bages?\s+by\s+(?:all\s+)?religions?\b", re.I),
        re.compile(r"\bage\s+distribution\b", re.I),
        re.compile(r"\bage\s*-\s*distribution\b", re.I),
    ],
    "table_number": [
        re.compile(r"\btable\s*(?:no\.\s*)?(?:v|vi|vii|viii|8)\b", re.I),
        re.compile(r"\btable\s*(?:vil|vir|vi1|v1i|vit|vln|vll)\b", re.I),
    ],
    "series_marker": [
        re.compile(r"\bc[\s\-]?(?:i|ii|iii)\b", re.I),
        re.compile(r"\bc[\s\-]?series\b", re.I),
        re.compile(r"\bdistrict\s+census\s+handbook\b", re.I),
    ],
    "broad_volume": [
        re.compile(r"\bimperial\s+tables?\b", re.I),
        re.compile(r"\breport\s+and\s+tables\b", re.I),
        re.compile(r"\bsubsidiary\s+tables\b", re.I),
        re.compile(r"\bpart\s+ii\b", re.I),
    ],
    "semantics": [
        re.compile(r"\bages?\b", re.I),
        re.compile(r"\bsex\b", re.I),
        re.compile(r"\bcivil\s+condition\b", re.I),
        re.compile(r"\bmarital\s+status\b", re.I),
        re.compile(r"\bpopulation\b", re.I),
        re.compile(r"\bmales?\b", re.I),
        re.compile(r"\bfemales?\b", re.I),
    ],
}


def run_command(args, timeout):
    return subprocess.run(args, capture_output=True, text=True, timeout=timeout)


def get_page_count(pdf_path):
    try:
        result = run_command(["pdfinfo", str(pdf_path)], timeout=30)
    except Exception:
        return 0
    for line in result.stdout.splitlines():
        if line.startswith("Pages:"):
            try:
                return int(line.split(":", 1)[1].strip())
            except ValueError:
                return 0
    return 0


def pdftotext_extract(pdf_path, first_page=None, last_page=None, timeout=300):
    cmd = ["pdftotext"]
    if first_page is not None:
        cmd += ["-f", str(first_page)]
    if last_page is not None:
        cmd += ["-l", str(last_page)]
    cmd += [str(pdf_path), "-"]
    try:
        result = run_command(cmd, timeout=timeout)
    except Exception:
        return ""
    return result.stdout or ""


def ascii_fold(text):
    return "".join(
        ch for ch in unicodedata.normalize("NFKD", text)
        if not unicodedata.combining(ch)
    )


def normalize_text(text):
    folded = ascii_fold(text)
    folded = folded.replace("&", " and ")
    folded = folded.lower()
    folded = folded.replace("1", "1").replace("0", "0")
    folded = re.sub(r"[\u2010-\u2015]", "-", folded)
    folded = re.sub(r"[^a-z0-9\-]+", " ", folded)
    folded = re.sub(r"\s+", " ", folded).strip()
    return folded


def dedupe_keep_order(values):
    seen = set()
    out = []
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def title_rest_from_text(text):
    match = re.search(
        r"Census\s+of\s+India\s+(\d{4})[.,]?\s*(.*?)(?:\n|Source:)",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    if not match:
        return None, None
    return int(match.group(1)), match.group(2).strip()


def canonicalize_province(name):
    candidate = re.sub(r"\s+", " ", name.replace("_", " ")).strip(" .,:;")
    candidate = candidate.replace("Pt. 1", "").replace("Pt. 2", "").strip(" .,:;")
    for alias, canonical in PROVINCE_ALIASES.items():
        if candidate.lower() == alias.lower():
            return canonical
    return candidate if candidate else None


def extract_province(text, pdf_path):
    year, rest = title_rest_from_text(text)
    haystacks = []
    if rest:
        haystacks.append(rest)
    haystacks.append(pdf_path.stem.replace("_", " "))
    haystacks.extend(part.replace("_", " ") for part in pdf_path.parts[-4:-1])

    for hay in haystacks:
        for alias, canonical in PROVINCE_ALIASES.items():
            if re.search(rf"\b{re.escape(alias)}\b", hay, re.I):
                return canonical

    if rest:
        patterns = [
            r"Vols?\.?\s*[IVXLC\d\-A-Za-z]+\s*,\s*([A-Z][A-Za-z&\-\s]+?)(?:\.\s*Pt\.|\s*:|\.?$)",
            r"Vols?\.?\s*[IVXLC\d\-A-Za-z]+\s+([A-Z][A-Za-z&\-\s]+?)(?:\.\s*Pt\.|\s*:|\.?$)",
            r"District\s+Census\s+Handbook\.\s*([A-Z][A-Za-z&\-\s]+?)(?:\.\s+[A-Z][A-Za-z&\-\s]+)?\.?$",
        ]
        for pattern in patterns:
            match = re.search(pattern, rest)
            if match:
                return canonicalize_province(match.group(1))
    return None


def extract_year(text, pdf_path):
    year, _ = title_rest_from_text(text)
    if year:
        return year
    for part in reversed(pdf_path.parts):
        if part.isdigit() and len(part) == 4:
            return int(part)
    match = re.search(r"(18\d{2}|19\d{2})", pdf_path.name)
    if match:
        return int(match.group(1))
    return None


def collect_keyword_hits(text):
    hits = {}
    for label, patterns in KEYWORD_GROUPS.items():
        found = []
        for pattern in patterns:
            for match in pattern.finditer(text):
                found.append(match.group(0))
        hits[label] = dedupe_keep_order(found)
    return hits


def classify_from_hits(hits):
    strong = bool(hits["strong_phrase"])
    numbered = bool(hits["table_number"])
    series = bool(hits["series_marker"])
    broad = bool(hits["broad_volume"])
    semantics = len(hits["semantics"])

    if strong:
        return "age_table_evidence"
    if numbered and semantics >= 2:
        return "age_table_evidence"
    if series and semantics >= 2:
        return "age_table_evidence"
    if semantics >= 3:
        return "age_table_candidate"
    if broad and semantics >= 2:
        return "age_table_candidate"
    if broad:
        return "table_volume_only"
    return "no_age_signal"


def analyze_pdf(pdf_path):
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        return {
            "file": str(pdf_path),
            "filename": pdf_path.name,
            "pages": 0,
            "size_mb": 0,
            "year": extract_year("", pdf_path),
            "province": extract_province("", pdf_path),
            "title_excerpt": "",
            "text_bytes": 0,
            "keyword_hits": {},
            "classification": "missing_file",
            "needs_mistral": False,
        }
    pages = get_page_count(pdf_path)
    full_text = pdftotext_extract(pdf_path, timeout=300)
    first_text = full_text[:4000] if full_text else pdftotext_extract(pdf_path, first_page=1, last_page=15, timeout=120)

    result = {
        "file": str(pdf_path),
        "filename": pdf_path.name,
        "pages": pages,
        "size_mb": round(pdf_path.stat().st_size / (1024 * 1024), 1),
        "year": extract_year(first_text or full_text or "", pdf_path),
        "province": extract_province(first_text or full_text or "", pdf_path),
        "title_excerpt": (first_text or full_text or "")[:500],
        "text_bytes": len(full_text.encode("utf-8")) if full_text else 0,
        "keyword_hits": {},
        "classification": "unreadable",
        "needs_mistral": False,
    }

    if not full_text.strip():
        result["classification"] = "unreadable"
        result["needs_mistral"] = True
        return result

    normalized = normalize_text(full_text)
    hits = collect_keyword_hits(normalized)
    result["keyword_hits"] = hits
    result["classification"] = classify_from_hits(hits)
    result["needs_mistral"] = result["classification"] in {"table_volume_only", "no_age_signal"}
    return result


def select_probe_pages(page_count, max_pages):
    if page_count <= 0:
        return []
    pages = list(range(1, min(page_count, 5) + 1))
    pages.extend(p for p in (8, 10, 12) if p <= page_count)
    if page_count > 20:
        pages.extend([max(1, page_count // 4), max(1, page_count // 2), max(1, (3 * page_count) // 4)])
    if page_count > 200:
        pages.append(200)
    if page_count > 1:
        pages.append(page_count)
    pages = sorted(set(pages))
    return pages[:max_pages]


def render_page_png(pdf_path, page_number, out_dir):
    prefix = Path(out_dir) / f"{Path(pdf_path).stem}_p{page_number}"
    cmd = ["pdftoppm", "-png", "-singlefile", "-f", str(page_number), "-l", str(page_number), str(pdf_path), str(prefix)]
    result = run_command(cmd, timeout=120)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or f"pdftoppm failed for {pdf_path}")
    return Path(str(prefix) + ".png")


def mistral_markdown_from_image(client, image_path):
    with open(image_path, "rb") as fh:
        data_uri = "data:image/png;base64," + base64.b64encode(fh.read()).decode("ascii")
    response = client.ocr.process(
        model="mistral-ocr-latest",
        document={"type": "image_url", "image_url": {"url": data_uri}},
        include_image_base64=False,
        timeout_ms=120000,
    )
    page_texts = []
    for page in getattr(response, "pages", []) or []:
        markdown = getattr(page, "markdown", None)
        if markdown:
            page_texts.append(markdown)
    return "\n".join(page_texts)


def run_mistral_fallback(pdf_path, page_count, max_probe_pages):
    if Mistral is None:
        raise RuntimeError("mistralai package is not available")
    api_key = os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        raise RuntimeError("MISTRAL_API_KEY is not set")

    client = Mistral(api_key=api_key)
    probe_pages = select_probe_pages(page_count, max_probe_pages)
    if not probe_pages:
        return {
            "status": "skipped",
            "reason": "no_pages",
            "pages_checked": [],
            "classification": "unreadable",
            "keyword_hits": {},
        }

    combined_text = []
    checked = []
    with tempfile.TemporaryDirectory(prefix="census_audit_") as tmpdir:
        for page in probe_pages:
            try:
                png_path = render_page_png(pdf_path, page, tmpdir)
            except Exception:
                continue
            if not png_path.exists():
                continue
            markdown = mistral_markdown_from_image(client, png_path)
            checked.append(page)
            if markdown:
                combined_text.append(markdown)
                hits = collect_keyword_hits(normalize_text("\n".join(combined_text)))
                classification = classify_from_hits(hits)
                if classification == "age_table_evidence":
                    return {
                        "status": "ok",
                        "pages_checked": checked,
                        "classification": classification,
                        "keyword_hits": hits,
                    }

    full_text = "\n".join(combined_text)
    hits = collect_keyword_hits(normalize_text(full_text)) if full_text else {}
    classification = classify_from_hits(hits) if hits else "no_age_signal"
    return {
        "status": "ok",
        "pages_checked": checked,
        "classification": classification,
        "keyword_hits": hits,
    }


def inventory_report(results):
    by_year = {}
    for entry in results:
        year = entry.get("year")
        province = entry.get("province")
        if not year or not province:
            continue
        by_year.setdefault(year, set()).add(province)

    report = {}
    for year, expected in EXPECTED_PROVINCES.items():
        found = by_year.get(year, set())
        report[str(year)] = {
            "expected_count": len(expected),
            "found_count": len(found),
            "missing": sorted(set(expected) - found),
            "found": sorted(found),
        }
    return report


def summarize(results):
    counts = {}
    for entry in results:
        counts[entry["classification"]] = counts.get(entry["classification"], 0) + 1
    unresolved = [
        entry for entry in results
        if entry["classification"] in {"unreadable", "table_volume_only", "no_age_signal"}
    ]
    return {
        "counts": counts,
        "unresolved_count": len(unresolved),
        "inventory": inventory_report(results),
    }


def main():
    parser = argparse.ArgumentParser(description="Audit census PDF classification completeness.")
    parser.add_argument("root", help="Directory containing PDFs")
    parser.add_argument("--output", default="census_audit_manifest.json", help="Output JSON path")
    parser.add_argument("--workers", type=int, default=4, help="Parallel workers for text pass")
    parser.add_argument("--mistral-fallback", action="store_true", help="Run Mistral OCR on unresolved PDFs")
    parser.add_argument("--mistral-limit", type=int, default=0, help="Max unresolved PDFs to send to Mistral (0 = all)")
    parser.add_argument("--probe-pages", type=int, default=8, help="Max probe pages per unresolved PDF for Mistral")
    args = parser.parse_args()

    root = Path(args.root)
    pdfs = sorted(root.rglob("*.pdf"))
    pdfs = [pdf for pdf in pdfs if not pdf.name.startswith("._")]
    print(f"Found {len(pdfs)} PDFs under {root}")
    if not pdfs:
        return

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_map = {executor.submit(analyze_pdf, pdf): pdf for pdf in pdfs}
        for i, future in enumerate(as_completed(future_map), 1):
            entry = future.result()
            results.append(entry)
            print(f"[{i}/{len(pdfs)}] {entry['filename']}: {entry['classification']}")

    results.sort(key=lambda row: (row.get("year") or 0, row.get("province") or "", row["filename"]))

    if args.mistral_fallback:
        unresolved = [
            row for row in results
            if row["classification"] in {"unreadable", "table_volume_only", "no_age_signal"}
        ]
        if args.mistral_limit:
            unresolved = unresolved[:args.mistral_limit]
        print(f"Running Mistral fallback on {len(unresolved)} PDFs")
        for idx, row in enumerate(unresolved, 1):
            print(f"  [Mistral {idx}/{len(unresolved)}] {row['filename']}")
            try:
                fallback = run_mistral_fallback(row["file"], row.get("pages") or 0, args.probe_pages)
                row["mistral_fallback"] = fallback
                if fallback.get("classification") in {"age_table_evidence", "age_table_candidate"}:
                    row["classification"] = fallback["classification"]
                    row["keyword_hits"] = fallback.get("keyword_hits", row.get("keyword_hits", {}))
            except Exception as exc:
                row["mistral_fallback"] = {
                    "status": "error",
                    "error": str(exc),
                }

    summary = summarize(results)
    manifest = {
        "root": str(root),
        "total_pdfs": len(results),
        "summary": summary,
        "pdfs": results,
    }

    out_path = Path(args.output)
    with open(out_path, "w") as fh:
        json.dump(manifest, fh, indent=2)

    print("\nSummary")
    for key, value in sorted(summary["counts"].items()):
        print(f"  {key}: {value}")
    print(f"  unresolved_count: {summary['unresolved_count']}")
    for year, inv in summary["inventory"].items():
        missing = inv["missing"]
        print(f"  inventory {year}: found {inv['found_count']}/{inv['expected_count']} ; missing {len(missing)}")
        if missing:
            print(f"    missing: {', '.join(missing[:12])}")
    print(f"\nManifest written to {out_path}")


if __name__ == "__main__":
    main()
