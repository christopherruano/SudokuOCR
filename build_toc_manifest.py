"""Build a TOC-centered document manifest for census PDFs.

This script treats the table of contents as the primary organizer for a
document, but it does not trust TOC OCR alone. It combines:

1. Front-matter extraction with `pdftotext`
2. OCR fallback for blank/boilerplate front pages
3. Conservative TOC/list-of-tables parsing
4. Body-header mining to recover table identity when TOC parsing is weak

The output manifest is designed to drive downstream table extraction and to
support later queryable corpus construction.

Usage:
    python3 build_toc_manifest.py /Volumes/Expansion/census_pdfs
    python3 build_toc_manifest.py /Volumes/Expansion/census_pdfs --workers 8
    python3 build_toc_manifest.py /Volumes/Expansion/census_pdfs \\
        --adjudications-dir review_annotations_pdf \\
        --tables-output toc_tables.jsonl
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
import tempfile
import unicodedata
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional

try:
    from PIL import Image
    import pytesseract
except Exception:  # pragma: no cover - optional dependency at runtime
    Image = None
    pytesseract = None


STOPWORDS = {
    "a", "all", "and", "appendix", "area", "areas", "by", "census", "chapter",
    "continued", "concluded", "district", "for", "general", "in", "india",
    "imperial", "of", "part", "population", "province", "report", "section",
    "showing", "state", "subsidiary", "table", "tables", "the", "to", "vol",
    "volume", "with",
}

LOW_VALUE_PATTERNS = [
    re.compile(r"^\s*$"),
    re.compile(r"this content downloaded from", re.I),
    re.compile(r"all use subject to https?://about\.jstor\.org/terms", re.I),
    re.compile(r"stable url:\s*https?://www\.jstor\.org/stable", re.I),
    re.compile(r"south asia open archives", re.I),
]

ENTRY_START_RE = re.compile(
    r"^(?:"
    r"subsidiary\s+table|imperial\s+table|table(?:\s+no\.)?|"
    r"section|part|chapter|appendix|"
    r"[abcdehst]\s*-\s*[ivxlcdm0-9]+(?:[\.\-][0-9a-z]+)*(?:\s+part\s+[ab])?"
    r")\b",
    re.I,
)

IDENTIFIER_PATTERNS = [
    ("subsidiary_table", re.compile(
        r"^(subsidiary\s+table)\s+([a-z0-9\-\.\(\)]+)\s*[:\-]?\s*(.*)$",
        re.I,
    )),
    ("imperial_table", re.compile(
        r"^(imperial\s+table)\s+([a-z0-9\-\.\(\)]+)\s*[:\-]?\s*(.*)$",
        re.I,
    )),
    ("table", re.compile(
        r"^(table(?:\s+no\.)?)\s+([a-z0-9\-\.\(\)]+)\s*[:\-]?\s*(.*)$",
        re.I,
    )),
    ("part", re.compile(
        r"^(part)\s+([a-z0-9\-\.\(\)]+)\s*[:\-]?\s*(.*)$",
        re.I,
    )),
    ("section", re.compile(
        r"^(section)\s+([a-z0-9\-\.\(\)]+)\s*[:\-]?\s*(.*)$",
        re.I,
    )),
    ("chapter", re.compile(
        r"^(chapter)\s+([a-z0-9\-\.\(\)]+)\s*[:\-]?\s*(.*)$",
        re.I,
    )),
]

PRINTED_PAGE_RE = re.compile(r"(?:\.{2,}|\s)\(?([0-9]{1,4}|[ivxlcdm]{1,12})\)?\s*$", re.I)
STRONG_TOPIC_RE = re.compile(
    r"\b(age|sex|civil condition|marital|religion|language|literacy|"
    r"migration|workers?|occupation|housing|household|mother tongue|"
    r"scheduled castes?|scheduled tribes?)\b",
    re.I,
)

TOPIC_PATTERNS = {
    "age": [re.compile(r"\bage\b", re.I), re.compile(r"\bages\b", re.I), re.compile(r"\bage[- ]group", re.I)],
    "sex": [re.compile(r"\bsex\b", re.I), re.compile(r"\bmales?\b", re.I), re.compile(r"\bfemales?\b", re.I)],
    "civil_condition": [re.compile(r"\bcivil\s+condition\b", re.I)],
    "marital_status": [re.compile(r"\bmarital\b", re.I), re.compile(r"\bmarriage\b", re.I)],
    "religion": [re.compile(r"\breligion\b", re.I), re.compile(r"\breligions\b", re.I)],
    "population": [re.compile(r"\bpopulation\b", re.I), re.compile(r"\bpersons\b", re.I)],
    "district": [re.compile(r"\bdistrict\b", re.I), re.compile(r"\btehsil\b", re.I)],
    "language": [re.compile(r"\blanguage\b", re.I), re.compile(r"\bmother[- ]tongue\b", re.I)],
    "literacy": [re.compile(r"\bliteracy\b", re.I), re.compile(r"\bliterate\b", re.I)],
    "education": [re.compile(r"\beducation", re.I), re.compile(r"\beducational\b", re.I)],
    "migration": [re.compile(r"\bmigration\b", re.I), re.compile(r"\bmigrant\b", re.I)],
    "workers": [re.compile(r"\bworkers?\b", re.I), re.compile(r"\bnon-workers?\b", re.I)],
    "occupation": [re.compile(r"\boccupation\b", re.I), re.compile(r"\boccupational\b", re.I)],
    "housing": [re.compile(r"\bhousing\b", re.I), re.compile(r"\bhousehold\b", re.I), re.compile(r"\bhouseholds\b", re.I)],
    "caste_tribe": [re.compile(r"\bscheduled castes?\b", re.I), re.compile(r"\bscheduled tribes?\b", re.I)],
    "general_population_tables": [re.compile(r"\bgeneral population tables?\b", re.I)],
    "c_series": [re.compile(r"\bc[\s\-]?series\b", re.I), re.compile(r"\bc[\s\-]?(?:i|ii|iii|iv|v|vi|vii|viii)\b", re.I)],
}

PROVINCE_ALIASES = {
    "Ajmer-Merwara": "Ajmer Merwara",
    "Ajmer Merwara": "Ajmer Merwara",
    "Andaman and Nicobar": "Andaman & Nicobar",
    "Andaman & Nicobar": "Andaman & Nicobar",
    "Baluchistan": "Baluchistan",
    "Baroda": "Baroda",
    "Bengal": "Bengal",
    "Berar": "Berar",
    "Bombay": "Bombay",
    "Burma": "Burma",
    "Cochin": "Cochin",
    "Coorg": "Coorg",
    "Central India": "Central India",
    "Central Provinces": "Central Provinces",
    "Central Provinces and Berar": "Central Provinces",
    "Gwalior": "Gwalior",
    "Hyderabad": "Hyderabad",
    "Jammu and Kashmir": "Jammu and Kashmir",
    "Madras": "Madras",
    "Mysore": "Mysore",
    "North-Western Provinces and Oudh": "North Western Provinces Oudh",
    "North-Western Provinces Oudh": "North Western Provinces Oudh",
    "North Western Provinces and Oudh": "North Western Provinces Oudh",
    "North Western Provinces Oudh": "North Western Provinces Oudh",
    "North-West Frontier Province": "North-West Frontier Province",
    "Punjab": "Punjab",
    "Rajputana": "Rajputana",
    "Rajasthan": "Rajasthan",
    "Travancore": "Travancore",
    "United Provinces": "United Provinces",
    "United Provinces of Agra and Oudh": "United Provinces",
}


def run_command(args: List[str], timeout: int = 120) -> subprocess.CompletedProcess:
    return subprocess.run(args, capture_output=True, text=True, timeout=timeout)


def ascii_fold(text: str) -> str:
    return "".join(
        ch for ch in unicodedata.normalize("NFKD", text)
        if not unicodedata.combining(ch)
    )


def normalize_text(text: str) -> str:
    text = ascii_fold(text or "")
    text = text.replace("&", " and ")
    text = text.lower()
    text = re.sub(r"[\u2010-\u2015]", "-", text)
    text = re.sub(r"[^a-z0-9\-\.\(\)]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def pdf_page_count(pdf_path: Path) -> int:
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


def pdftotext_pages(pdf_path: Path, start_page: int, end_page: int, timeout: int = 300) -> List[dict]:
    result = run_command(
        ["pdftotext", "-f", str(start_page), "-l", str(end_page), str(pdf_path), "-"],
        timeout=timeout,
    )
    text = result.stdout or ""
    parts = text.split("\f")
    pages = []
    for idx in range(start_page, end_page + 1):
        part_idx = idx - start_page
        pages.append({
            "pdf_page": idx,
            "text": parts[part_idx] if part_idx < len(parts) else "",
            "text_source": "pdftotext",
        })
    return pages


def page_text_kind(text: str) -> str:
    compact = normalize_text(text)
    if not compact:
        return "blank"
    token_count = len(compact.split())
    if token_count < 8:
        return "blank"
    if any(pattern.search(text or "") for pattern in LOW_VALUE_PATTERNS):
        contentful_terms = sum(bool(re.search(term, text or "", re.I)) for term in [
            r"\bcensus\b", r"\bcontents\b", r"\btable\b", r"\bpart\b", r"\bvolume\b",
            r"\brajasthan\b", r"\bpopulation\b", r"\bage\b", r"\bseries\b",
        ])
        if contentful_terms < 2:
            return "boilerplate"
    return "contentful"


def should_ocr_page(text: str, ocr_mode: str) -> bool:
    if ocr_mode == "always":
        return True
    if ocr_mode == "never":
        return False
    return page_text_kind(text) in {"blank", "boilerplate"}


def render_pdf_page(pdf_path: Path, page_number: int, cache_dir: Path) -> Path:
    digest = hashlib.sha1(f"{pdf_path}:{page_number}".encode("utf-8")).hexdigest()
    prefix = cache_dir / digest
    png = Path(str(prefix) + ".png")
    if png.exists():
        return png
    cmd = [
        "pdftoppm", "-png", "-singlefile",
        "-f", str(page_number), "-l", str(page_number),
        str(pdf_path), str(prefix),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    if result.returncode != 0 or not png.exists():
        raise RuntimeError(result.stderr.strip() or f"Failed to render page {page_number}")
    return png


def ocr_rendered_page(png_path: Path) -> str:
    if Image is None or pytesseract is None:
        return ""
    image = Image.open(png_path)
    return pytesseract.image_to_string(image, config="--psm 6")


def extract_front_pages(
    pdf_path: Path,
    page_count: int,
    max_pdf_pages: int,
    target_content_pages: int,
    ocr_mode: str,
    ocr_cache_dir: Optional[Path],
) -> List[dict]:
    limit = min(page_count, max_pdf_pages)
    raw_pages = pdftotext_pages(pdf_path, 1, limit, timeout=max(300, limit * 8))
    cache_root_ctx = None
    cache_root = ocr_cache_dir
    if cache_root is None:
        cache_root_ctx = tempfile.TemporaryDirectory()
        cache_root = Path(cache_root_ctx.name)
    cache_root.mkdir(parents=True, exist_ok=True)

    payloads = []
    content_pages = 0
    try:
        for payload in raw_pages:
            text = payload["text"]
            source = "pdftotext"
            kind = page_text_kind(text)
            if should_ocr_page(text, ocr_mode):
                try:
                    png = render_pdf_page(pdf_path, payload["pdf_page"], cache_root)
                    ocr_text = ocr_rendered_page(png)
                except Exception:
                    ocr_text = ""
                if page_text_kind(ocr_text) == "contentful" or (ocr_mode == "always" and ocr_text.strip()):
                    text = ocr_text
                    source = "ocr"
                    kind = page_text_kind(text)
            row = {
                "pdf_page": payload["pdf_page"],
                "text": text,
                "text_source": source,
                "text_kind": kind,
            }
            payloads.append(row)
            if kind == "contentful":
                content_pages += 1
            if content_pages >= target_content_pages:
                break
    finally:
        if cache_root_ctx is not None:
            cache_root_ctx.cleanup()
    return payloads


def extract_year(text: str, pdf_path: Path) -> Optional[int]:
    match = re.search(r"Census\s+of\s+India\s+(\d{4})", text, re.I)
    if match:
        return int(match.group(1))
    for part in reversed(pdf_path.parts):
        if part.isdigit() and len(part) == 4:
            return int(part)
    match = re.search(r"(18\d{2}|19\d{2})", pdf_path.name)
    if match:
        return int(match.group(1))
    return None


def canonicalize_province(name: str) -> Optional[str]:
    candidate = re.sub(r"\s+", " ", name.replace("_", " ")).strip(" .,:;")
    candidate = re.sub(r"\bPt\.?\s*[0-9A-Za-z]+\b", "", candidate, flags=re.I).strip(" .,:;")
    for alias, canonical in PROVINCE_ALIASES.items():
        if candidate.lower() == alias.lower():
            return canonical
    return candidate or None


def extract_province(text: str, pdf_path: Path) -> Optional[str]:
    haystacks = [text, pdf_path.name, " ".join(pdf_path.parts[-4:])]
    for hay in haystacks:
        for alias, canonical in PROVINCE_ALIASES.items():
            if re.search(rf"\b{re.escape(alias)}\b", hay, re.I):
                return canonical
    match = re.search(
        r"Vols?\.?\s*[IVXLC\d\-A-Za-z]+\s*,\s*([A-Z][A-Za-z&\-\s]+?)(?:\.\s*Pt\.|\s*:|\.?$)",
        text,
        re.I,
    )
    if match:
        return canonicalize_province(match.group(1))
    return None


def extract_title(front_pages: List[dict], pdf_path: Path) -> str:
    lines = []
    for payload in front_pages[:10]:
        for raw in payload["text"].splitlines():
            line = raw.strip()
            if not line:
                continue
            if re.search(r"\bCensus of India\b", line, re.I):
                return re.sub(r"\s+", " ", line)
            if re.search(r"\bVolume\b|\bVol\.\b|\bPart\b", line, re.I) and len(line) < 180:
                lines.append(line)
    if lines:
        return re.sub(r"\s+", " ", " | ".join(lines[:2]))
    return pdf_path.stem.replace("_", " ")


def clean_lines(text: str) -> List[str]:
    out = []
    for raw in (text or "").splitlines():
        line = re.sub(r"\s+", " ", raw).strip()
        if not line:
            continue
        if re.fullmatch(r"[\(\)\[\]\-–—_\.]+", line):
            continue
        if re.fullmatch(r"[ivxlcdm0-9]+", line, re.I):
            continue
        if re.search(r"this content downloaded from|all use subject to", line, re.I):
            continue
        out.append(line)
    return out


def line_has_dense_numeric_table(line: str) -> bool:
    numeric_chunks = re.findall(r"\d[\d,\.]*", line)
    return len(numeric_chunks) >= 4 and not ENTRY_START_RE.match(line)


def line_looks_like_toc_entry(line: str) -> bool:
    if len(line) > 260:
        return False
    if line_has_dense_numeric_table(line):
        return False
    if ENTRY_START_RE.match(line) and PRINTED_PAGE_RE.search(line):
        return True
    if STRONG_TOPIC_RE.search(line) and PRINTED_PAGE_RE.search(line):
        return True
    if ENTRY_START_RE.match(line):
        return True
    return False


def toc_page_score(payload: dict) -> dict:
    text = payload["text"]
    lines = clean_lines(text)
    normalized = normalize_text(text)
    score = 0
    reasons = {}

    dotted_lines = sum(bool(PRINTED_PAGE_RE.search(line)) for line in lines)
    strong_entries = sum(bool(ENTRY_START_RE.match(line)) for line in lines)
    topical_entries = sum(bool(STRONG_TOPIC_RE.search(line) and PRINTED_PAGE_RE.search(line)) for line in lines)
    numeric_penalty = sum(bool(line_has_dense_numeric_table(line)) for line in lines)

    if "contents" in normalized or "table of contents" in normalized:
        reasons["contents"] = 6
    if "list of tables" in normalized:
        reasons["list_of_tables"] = 6
    if "1961 census publications" in normalized:
        reasons["publication_schedule"] = 3
    if strong_entries >= 3:
        reasons["strong_entries"] = 3
    if dotted_lines >= 4:
        reasons["page_leaders"] = 3
    if topical_entries >= 2:
        reasons["topic_lines"] = 3
    if any(re.search(r"\bpage\b", line, re.I) for line in lines[:4]):
        reasons["page_header"] = 1
    if numeric_penalty >= 5:
        reasons["dense_numeric_penalty"] = -5
    elif numeric_penalty >= 2:
        reasons["dense_numeric_penalty"] = -2

    score = sum(reasons.values())
    return {"score": score, "reasons": reasons}


def merge_wrapped_entries(lines: List[str]) -> List[str]:
    entries = []
    buffer: List[str] = []

    def flush():
        if buffer:
            candidate = " ".join(buffer).strip()
            if candidate:
                entries.append(candidate)
            buffer.clear()

    for line in lines:
        if re.match(r"^(contents|page|preface)\b", line, re.I):
            continue
        if len(line) > 260:
            flush()
            continue

        if buffer and (PRINTED_PAGE_RE.search(" ".join(buffer)) or (ENTRY_START_RE.match(line) and not line.startswith("("))):
            flush()

        if not buffer and not line_looks_like_toc_entry(line):
            continue

        if buffer:
            joined = " ".join(buffer + [line])
            if line_has_dense_numeric_table(joined):
                flush()
                continue
            buffer.append(line)
        else:
            buffer.append(line)

        if PRINTED_PAGE_RE.search(" ".join(buffer)):
            flush()

    flush()
    return entries


def parse_identifier(left: str) -> tuple:
    for entry_type, pattern in IDENTIFIER_PATTERNS:
        match = pattern.match(left)
        if match:
            label = match.group(1)
            identifier = match.group(2)
            title = match.group(3).strip(" .:-")
            return entry_type, f"{label} {identifier}".strip(), identifier, title
    if ENTRY_START_RE.match(left):
        return "entry", None, None, left.strip(" .:-")
    return "entry", None, None, left.strip(" .:-")


def classify_entry(text: str) -> dict:
    facets = []
    for facet, patterns in TOPIC_PATTERNS.items():
        if any(pattern.search(text or "") for pattern in patterns):
            facets.append(facet)

    facet_set = set(facets)
    canonical = "other"
    relevance = "none"

    if {"age", "sex", "civil_condition"} <= facet_set:
        canonical = "age_sex_civil_condition"
        relevance = "core"
    elif {"age", "sex", "marital_status"} <= facet_set:
        canonical = "age_sex_marital_status"
        relevance = "core"
    elif {"age", "sex"} <= facet_set:
        canonical = "age_sex_population"
        relevance = "core"
    elif {"age", "civil_condition"} <= facet_set:
        canonical = "age_civil_condition"
        relevance = "core"
    elif {"age", "religion"} <= facet_set:
        canonical = "age_by_religion"
        relevance = "core"
    elif {"sex", "marital_status"} <= facet_set:
        canonical = "sex_marital_status"
        relevance = "context"
    elif {"age", "literacy"} <= facet_set or {"age", "education"} <= facet_set:
        canonical = "age_literacy_education"
        relevance = "context"
    elif "age" in facet_set:
        canonical = "age_distribution"
        relevance = "core"
    elif {"general_population_tables", "c_series"} <= facet_set:
        canonical = "social_cultural_series"
        relevance = "context"
    elif "migration" in facet_set:
        canonical = "migration_tables"
        relevance = "context"
    elif "language" in facet_set:
        canonical = "language_tables"
        relevance = "context"
    elif "literacy" in facet_set or "education" in facet_set:
        canonical = "literacy_education_tables"
        relevance = "context"
    elif "workers" in facet_set or "occupation" in facet_set:
        canonical = "economic_tables"
        relevance = "context"
    elif "housing" in facet_set:
        canonical = "housing_tables"
        relevance = "context"
    elif "caste_tribe" in facet_set:
        canonical = "caste_tribe_tables"
        relevance = "context"

    return {
        "canonical_topic": canonical,
        "query_tags": facets,
        "age_pipeline_relevant": relevance == "core",
        "relevance_tier": relevance,
        "confidence": "high" if relevance == "core" else ("medium" if canonical != "other" else "low"),
    }


def parse_toc_entries(toc_payloads: List[dict]) -> List[dict]:
    entries = []
    for payload in toc_payloads:
        merged = merge_wrapped_entries(clean_lines(payload["text"]))
        for raw in merged:
            page_match = PRINTED_PAGE_RE.search(raw)
            if not page_match:
                continue
            left = raw[:page_match.start()].strip(" .")
            if not left:
                continue
            if line_has_dense_numeric_table(left):
                continue
            entry_type, display_id, identifier, title = parse_identifier(left)
            title_text = title or left
            if not STRONG_TOPIC_RE.search(title_text) and entry_type == "entry":
                continue
            tags = classify_entry(title_text)
            entries.append({
                "source": "toc",
                "toc_pdf_page": payload["pdf_page"],
                "text_source": payload["text_source"],
                "raw_entry": raw,
                "entry_type": entry_type,
                "display_identifier": display_id,
                "identifier": identifier,
                "title": title_text,
                "printed_page_label": page_match.group(1),
                **tags,
            })
    return dedupe_entries(entries)


def unique_tokens(text: str) -> List[str]:
    tokens = []
    for token in normalize_text(text).split():
        if len(token) < 3 or token in STOPWORDS:
            continue
        if token not in tokens:
            tokens.append(token)
    return tokens


def extract_header_candidate(page_text: str) -> Optional[str]:
    lines = clean_lines(page_text)
    header_lines = []
    for line in lines[:14]:
        if line_has_dense_numeric_table(line):
            break
        header_lines.append(line)
    joined = " ".join(header_lines)
    if re.search(r"https?://|stable\s+url|jstor", joined, re.I) and not STRONG_TOPIC_RE.search(joined):
        return None
    match = re.search(
        r"(\b(?:subsidiary\s+table|imperial\s+table|table(?:\s+no\.)?)\b\s+[A-Z0-9\-\.\(\)]+(?:\s*[:\-]\s*|\s+).{0,220})",
        joined,
        re.I,
    )
    if match:
        return re.sub(r"\s+", " ", match.group(1)).strip(" .")
    if re.search(r"\bage\b", joined, re.I) and re.search(r"\bsex\b", joined, re.I):
        return re.sub(r"\s+", " ", joined[:240]).strip(" .")
    return None


def mine_headers_from_payloads(page_payloads: List[dict], source_name: str) -> List[dict]:
    mined = []
    for payload in page_payloads:
        candidate = extract_header_candidate(payload["text"])
        if not candidate:
            continue
        entry_type, display_id, identifier, title = parse_identifier(candidate)
        tags = classify_entry(title or candidate)
        mined.append({
            "source": source_name,
            "body_pdf_page": payload["pdf_page"],
            "entry_type": entry_type,
            "display_identifier": display_id,
            "identifier": identifier,
            "title": title or candidate,
            **tags,
        })
    return dedupe_entries(mined)


def mine_body_headers(pdf_path: Path, page_count: int, body_max_pages: int) -> List[dict]:
    if page_count <= 0:
        return []
    limit = page_count if body_max_pages <= 0 else min(page_count, body_max_pages)
    pages = pdftotext_pages(pdf_path, 1, limit, timeout=max(600, limit * 4))
    return mine_headers_from_payloads(pages, "body_header")


def entry_key(entry: dict) -> str:
    if entry.get("identifier"):
        return f"id::{normalize_text(entry['identifier'])}"
    return f"title::{normalize_text(entry.get('title', ''))}"


def dedupe_entries(entries: List[dict]) -> List[dict]:
    merged: Dict[str, dict] = {}
    for entry in entries:
        key = entry_key(entry)
        current = merged.get(key)
        if not current:
            merged[key] = dict(entry)
            continue
        if entry.get("source") in {"body_header", "front_header"}:
            current.setdefault("body_page_hits", [])
            current["body_page_hits"].append({
                "pdf_page": entry["body_pdf_page"],
                "score": 1,
                "source": entry["source"],
            })
            if "body_pdf_page" not in current:
                current["body_pdf_page"] = entry["body_pdf_page"]
        if entry.get("source") == "toc" and "toc_pdf_page" not in current:
            current["toc_pdf_page"] = entry.get("toc_pdf_page")
            current["printed_page_label"] = entry.get("printed_page_label")
        current["query_tags"] = sorted(set(current.get("query_tags", [])) | set(entry.get("query_tags", [])))
        if current.get("relevance_tier") != "core" and entry.get("relevance_tier") == "core":
            current["relevance_tier"] = "core"
            current["age_pipeline_relevant"] = True
            current["canonical_topic"] = entry.get("canonical_topic")
            current["confidence"] = entry.get("confidence")
    rows = list(merged.values())
    rows.sort(key=lambda row: (
        0 if row.get("relevance_tier") == "core" else 1,
        row.get("body_pdf_page") or 10**9,
        row.get("toc_pdf_page") or 10**9,
        row.get("display_identifier") or "",
        row.get("title") or "",
    ))
    return rows


def merge_toc_and_body_entries(toc_entries: List[dict], body_entries: List[dict]) -> List[dict]:
    merged = {entry_key(entry): dict(entry) for entry in toc_entries}
    for entry in body_entries:
        key = entry_key(entry)
        current = merged.get(key)
        if not current:
            current = dict(entry)
            current["body_page_hits"] = [{"pdf_page": entry["body_pdf_page"], "score": 1, "source": entry["source"]}]
            merged[key] = current
            continue
        current.setdefault("body_page_hits", [])
        current["body_page_hits"].append({"pdf_page": entry["body_pdf_page"], "score": 1, "source": entry["source"]})
        current.setdefault("body_pdf_page", entry["body_pdf_page"])
        current["query_tags"] = sorted(set(current.get("query_tags", [])) | set(entry.get("query_tags", [])))
        if current.get("canonical_topic") == "other" and entry.get("canonical_topic") != "other":
            current["canonical_topic"] = entry["canonical_topic"]
            current["relevance_tier"] = entry["relevance_tier"]
            current["age_pipeline_relevant"] = entry["age_pipeline_relevant"]
            current["confidence"] = entry["confidence"]

    rows = list(merged.values())
    for row in rows:
        if row.get("body_page_hits"):
            row["body_page_hits"].sort(key=lambda hit: (hit["pdf_page"], hit["score"]))
            row["best_pdf_page"] = row["body_page_hits"][0]["pdf_page"]
        elif row.get("body_pdf_page"):
            row["best_pdf_page"] = row["body_pdf_page"]
    rows.sort(key=lambda row: (
        row.get("best_pdf_page") or 10**9,
        row.get("printed_page_label") or "",
        row.get("display_identifier") or "",
        row.get("title") or "",
    ))
    return rows


def summarize_entries(entries: List[dict]) -> dict:
    topic_counts = Counter(entry["canonical_topic"] for entry in entries)
    relevance_counts = Counter(entry.get("relevance_tier") or "none" for entry in entries)
    return {
        "entry_count": len(entries),
        "relevant_entry_count": sum(1 for entry in entries if entry.get("age_pipeline_relevant")),
        "topics": dict(topic_counts),
        "relevance": dict(relevance_counts),
    }


def load_adjudications(adjudications_dir: Optional[Path]) -> Dict[str, dict]:
    if not adjudications_dir or not adjudications_dir.exists():
        return {}
    rows = {}
    for path in adjudications_dir.glob("*.review.json"):
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        file_path = data.get("file")
        if file_path:
            rows[file_path] = {
                "decision": data.get("decision"),
                "notes": data.get("notes", ""),
            }
    return rows


def flatten_table_records(doc: dict) -> List[dict]:
    out = []
    for idx, entry in enumerate(doc.get("entries", []), 1):
        out.append({
            "file": doc["file"],
            "filename": doc["filename"],
            "year": doc.get("year"),
            "province": doc.get("province"),
            "document_title": doc.get("title"),
            "human_review_decision": (doc.get("human_review") or {}).get("decision"),
            "table_index": idx,
            "entry_type": entry.get("entry_type"),
            "display_identifier": entry.get("display_identifier"),
            "identifier": entry.get("identifier"),
            "title": entry.get("title"),
            "canonical_topic": entry.get("canonical_topic"),
            "query_tags": ";".join(entry.get("query_tags", [])),
            "relevance_tier": entry.get("relevance_tier"),
            "age_pipeline_relevant": entry.get("age_pipeline_relevant"),
            "printed_page_label": entry.get("printed_page_label"),
            "best_pdf_page": entry.get("best_pdf_page"),
            "toc_pdf_page": entry.get("toc_pdf_page"),
            "body_pdf_page": entry.get("body_pdf_page"),
            "source": entry.get("source"),
            "confidence": entry.get("confidence"),
        })
    return out


def process_pdf(
    pdf_path: Path,
    max_front_pdf_pages: int,
    target_front_content_pages: int,
    ocr_mode: str,
    ocr_cache_dir: Optional[Path],
    mine_body: bool,
    body_max_pages: int,
    adjudications: Dict[str, dict],
) -> dict:
    pdf_path = Path(pdf_path)
    page_count = pdf_page_count(pdf_path)
    if page_count <= 0:
        return {
            "file": str(pdf_path),
            "filename": pdf_path.name,
            "pages": 0,
            "classification": "unreadable",
            "toc_detected": False,
            "entries": [],
        }

    front_pages = extract_front_pages(
        pdf_path,
        page_count=page_count,
        max_pdf_pages=max_front_pdf_pages,
        target_content_pages=target_front_content_pages,
        ocr_mode=ocr_mode,
        ocr_cache_dir=ocr_cache_dir,
    )
    front_excerpt = "\n".join(payload["text"] for payload in front_pages[:6])[:3000]
    year = extract_year(front_excerpt, pdf_path)
    province = extract_province(front_excerpt, pdf_path)
    title = extract_title(front_pages, pdf_path)

    toc_payloads = []
    toc_pages = []
    for payload in front_pages:
        score_payload = toc_page_score(payload)
        if score_payload["score"] >= 6:
            toc_pages.append({
                "pdf_page": payload["pdf_page"],
                "score": score_payload["score"],
                "reasons": score_payload["reasons"],
                "text_source": payload["text_source"],
            })
            toc_payloads.append(payload)

    toc_entries = parse_toc_entries(toc_payloads)
    toc_page_numbers = {payload["pdf_page"] for payload in toc_payloads}
    front_body_payloads = [payload for payload in front_pages if payload["pdf_page"] not in toc_page_numbers]
    front_header_entries = mine_headers_from_payloads(front_body_payloads, "front_header")
    body_entries = mine_body_headers(pdf_path, page_count, body_max_pages) if mine_body else []
    entries = merge_toc_and_body_entries(toc_entries, front_header_entries + body_entries)
    summary = summarize_entries(entries)

    return {
        "file": str(pdf_path),
        "filename": pdf_path.name,
        "pages": page_count,
        "year": year,
        "province": province,
        "title": title,
        "human_review": adjudications.get(str(pdf_path)),
        "toc_detected": bool(toc_pages),
        "toc_pages": toc_pages,
        "front_pages": [{
            "pdf_page": payload["pdf_page"],
            "text_source": payload["text_source"],
            "text_kind": payload["text_kind"],
        } for payload in front_pages],
        "front_matter_excerpt": front_excerpt,
        "entries": entries,
        "summary": summary,
    }


def write_tables_jsonl(path: Path, table_rows: List[dict]) -> None:
    with open(path, "w") as fh:
        for row in table_rows:
            fh.write(json.dumps(row) + "\n")


def write_tables_csv(path: Path, table_rows: List[dict]) -> None:
    fieldnames = [
        "file", "filename", "year", "province", "document_title", "human_review_decision",
        "table_index", "entry_type", "display_identifier", "identifier", "title",
        "canonical_topic", "query_tags", "relevance_tier", "age_pipeline_relevant",
        "printed_page_label", "best_pdf_page", "toc_pdf_page", "body_pdf_page",
        "source", "confidence",
    ]
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(table_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a TOC-centered census manifest")
    parser.add_argument("root", help="Directory containing PDFs")
    parser.add_argument("--output", default="toc_manifest.json", help="Output JSON path")
    parser.add_argument("--tables-output", help="Optional JSONL file for flat table records")
    parser.add_argument("--tables-csv", help="Optional CSV file for flat table records")
    parser.add_argument("--workers", type=int, default=4, help="Parallel workers")
    parser.add_argument("--front-pages", type=int, default=80, help="Maximum front PDF pages to scan")
    parser.add_argument("--front-content-pages", type=int, default=25, help="Stop after this many contentful front pages")
    parser.add_argument(
        "--ocr-front",
        choices=["auto", "always", "never"],
        default="auto",
        help="OCR policy for front matter",
    )
    parser.add_argument("--ocr-cache-dir", help="Optional persistent OCR cache directory")
    parser.add_argument("--adjudications-dir", help="Optional directory of *.review.json files")
    parser.add_argument("--no-mine-body", action="store_true", help="Disable body-header mining")
    parser.add_argument("--body-max-pages", type=int, default=0, help="Limit body-header mining to N pages (0 = all)")
    args = parser.parse_args()

    root = Path(args.root)
    pdfs = sorted(p for p in root.rglob("*.pdf") if not p.name.startswith("._"))
    print(f"Found {len(pdfs)} PDFs under {root}")
    if not pdfs:
        return

    adjudications = load_adjudications(Path(args.adjudications_dir)) if args.adjudications_dir else {}
    ocr_cache_dir = Path(args.ocr_cache_dir) if args.ocr_cache_dir else None

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_map = {
            executor.submit(
                process_pdf,
                pdf,
                max_front_pdf_pages=args.front_pages,
                target_front_content_pages=args.front_content_pages,
                ocr_mode=args.ocr_front,
                ocr_cache_dir=ocr_cache_dir,
                mine_body=not args.no_mine_body,
                body_max_pages=args.body_max_pages,
                adjudications=adjudications,
            ): pdf
            for pdf in pdfs
        }
        for idx, future in enumerate(as_completed(future_map), 1):
            row = future.result()
            results.append(row)
            summary = row.get("summary", {})
            review = (row.get("human_review") or {}).get("decision")
            print(
                f"[{idx}/{len(pdfs)}] {row['filename']}: "
                f"toc={row.get('toc_detected')} entries={summary.get('entry_count', 0)} "
                f"relevant={summary.get('relevant_entry_count', 0)} review={review or '-'}"
            )

    results.sort(key=lambda r: (r.get("year") or 0, r.get("province") or "", r["filename"]))
    table_rows = []
    for doc in results:
        table_rows.extend(flatten_table_records(doc))

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "total_pdfs": len(results),
        "options": {
            "front_pages": args.front_pages,
            "front_content_pages": args.front_content_pages,
            "ocr_front": args.ocr_front,
            "mine_body": not args.no_mine_body,
            "body_max_pages": args.body_max_pages,
        },
        "documents": results,
        "table_record_count": len(table_rows),
    }
    out = Path(args.output)
    with open(out, "w") as fh:
        json.dump(manifest, fh, indent=2)

    if args.tables_output:
        write_tables_jsonl(Path(args.tables_output), table_rows)
    if args.tables_csv:
        write_tables_csv(Path(args.tables_csv), table_rows)

    docs_with_toc = sum(1 for doc in results if doc.get("toc_detected"))
    docs_with_relevant = sum(1 for doc in results if doc.get("summary", {}).get("relevant_entry_count", 0) > 0)
    print("\nSummary")
    print(f"  docs_with_toc: {docs_with_toc}")
    print(f"  docs_with_relevant_entries: {docs_with_relevant}")
    print(f"  table_record_count: {len(table_rows)}")
    print(f"Manifest written to {out}")


if __name__ == "__main__":
    main()
