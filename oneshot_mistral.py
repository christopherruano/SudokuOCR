"""
Mistral OCR + constraint-repair hybrid pipeline for historical Indian census tables.

Architecture:
  1. Mistral OCR ($0.002/page) → structured markdown table
  2. Parse markdown → standard JSON format (same as oneshot.py)
  3. Derive + verify constraints (L1-L5)
  4. If failures: free repair (Phase A: M/F swap, Phase B: digit deduction)
  5. If still failing: Gemini targeted recheck (1 API call)
  6. If structural failure: full Gemini fallback (extract_and_verify)

Usage:
    python3 oneshot_mistral.py IMAGE                     # single image
    python3 oneshot_mistral.py --batch DIR               # all PNGs in directory
    python3 oneshot_mistral.py IMAGE --full-fallback      # allow full Gemini fallback
"""

import os
import re
import sys
import json
import time
import base64
import logging
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(PROJECT_ROOT))
from pipeline import normalize_age, export_multigroup_to_excel, export_multigroup_to_json
from oneshot import (
    derive_constraints,
    verify_all_constraints,
    constraint_repair,
    identify_suspicious_cells,
    targeted_recheck,
    to_tidy_dataframe,
    to_legacy_format,
    extract_and_verify,
    _detect_and_fix_mf_swaps,
    _deductive_digit_fix,
    _normalize_value,
)
from schema_discovery import TableSchema


# ---------------------------------------------------------------------------
# Mistral OCR API caller
# ---------------------------------------------------------------------------

def call_mistral_ocr(image_path):
    """Call Mistral OCR API on an image, return raw markdown text.

    Returns:
        (markdown_text, elapsed_seconds)
    """
    from mistralai import Mistral

    api_key = os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        raise RuntimeError("MISTRAL_API_KEY not set")

    client = Mistral(api_key=api_key)

    suffix = Path(image_path).suffix.lower()
    mime = "image/png" if suffix == ".png" else "image/jpeg"
    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")
    data_uri = f"data:{mime};base64,{img_b64}"

    t0 = time.time()
    response = client.ocr.process(
        model="mistral-ocr-latest",
        document={"type": "image_url", "image_url": data_uri},
    )
    elapsed = time.time() - t0

    raw_text = "\n\n".join(page.markdown for page in response.pages)
    return raw_text, elapsed


# ---------------------------------------------------------------------------
# Markdown table parser → standard parsed dict
# ---------------------------------------------------------------------------

def _parse_cell_value(cell_text):
    """Parse a single cell from markdown table into int or None."""
    s = cell_text.strip()
    if not s or s == ".." or s == "." or s == "—" or s == "-" or s == "…":
        return 0
    # Remove commas and spaces in numbers
    s = s.replace(",", "").replace(" ", "")
    # Handle middle dot decimal (proportional tables like "1,384·8")
    if "·" in s:
        # Proportional table with decimal — round to nearest int
        s = s.replace("·", ".")
        try:
            return int(round(float(s)))
        except ValueError:
            return None
    try:
        return int(s)
    except ValueError:
        return None


def _extract_markdown_tables(raw_text):
    """Split raw Mistral markdown into individual table blocks.

    Each table block is a list of rows, where each row is a list of cell strings.
    Returns list of table blocks.
    """
    lines = raw_text.split("\n")
    tables = []
    current_table = []

    for line in lines:
        stripped = line.strip()
        if not stripped.startswith("|"):
            if current_table:
                tables.append(current_table)
                current_table = []
            continue

        # Parse pipe-separated cells
        cells = [c.strip() for c in stripped.split("|")]
        # Remove empty first/last from leading/trailing pipes
        if cells and cells[0] == "":
            cells = cells[1:]
        if cells and cells[-1] == "":
            cells = cells[:-1]

        # Skip separator rows (| --- | --- |)
        if all(re.match(r'^-+$', c.strip()) for c in cells if c.strip()):
            continue

        current_table.append(cells)

    if current_table:
        tables.append(current_table)

    return tables


def _is_number_row(cells):
    """Check if a row is the column-number row (e.g. | 1 | 2 | 3 | ...)."""
    nums = 0
    for c in cells:
        s = c.strip()
        if s and re.match(r'^\d{1,2}$', s):
            nums += 1
    return nums >= len(cells) * 0.6


def _classify_pmf(s):
    """Classify a sub-header label as persons/males/females or None."""
    s = s.strip().rstrip(".").lower()
    if s in ("persons", "p", "total"):
        return "persons"
    if s in ("males", "m", "male"):
        return "males"
    if s in ("females", "f", "female"):
        return "females"
    return None


def _detect_header_groups(header_row, subheader_row):
    """Detect column groups and P/M/F mapping from header rows.

    Handles header/subheader column misalignment (common in 1941 tables where
    the header row has fewer prefix columns than the subheader).

    Returns:
        groups: list of group names (e.g. ["POPULATION", "UNMARRIED", ...])
        col_map: list of (group_name, "persons"|"males"|"females") for each data column
        has_persons: whether any group has a Persons column
        data_col_start: column index where data begins in the subheader
    """
    # Build P/M/F pattern from subheader
    pmf_pattern = [_classify_pmf(c) for c in subheader_row]

    # Find where data columns start in subheader
    data_start = None
    for i, p in enumerate(pmf_pattern):
        if p is not None:
            data_start = i
            break
    if data_start is None:
        return [], [], False, 0

    # Detect group size (3=P/M/F or 2=M/F)
    data_cols = pmf_pattern[data_start:]
    has_persons = "persons" in data_cols[:4]
    group_size = 3 if has_persons else 2

    # Count number of P/M/F columns (strip trailing Nones)
    n_pmf = 0
    for p in data_cols:
        if p is not None:
            n_pmf += 1
    expected_groups = n_pmf // group_size

    # Extract group names from the ENTIRE header row (not just from data_start).
    # The header may have group names at different column positions than the
    # subheader's data columns (e.g. POPULATION at col 1 but Persons at col 2).
    # Filter out label-like cells (District, Age, Srl. No., etc.)
    # Instead of filtering out label words (fragile with OCR noise),
    # positively match known census group names.
    _KNOWN_GROUPS = {
        "population", "unmarried", "married", "widowed", "divorced",
    }
    header_labels = [c.strip().replace("&amp;", "&") for c in header_row]
    group_names = []
    for label in header_labels:
        if not label or re.match(r'^\s*$', label):
            continue
        label_clean = label.upper().strip(". ")
        # Check exact match or word-boundary containment of known group names
        # Order matters: check longer names first to avoid "MARRIED" matching "UNMARRIED"
        matched = False
        for known in sorted(_KNOWN_GROUPS, key=len, reverse=True):
            if known.upper() == label_clean:
                group_names.append(known.upper())
                matched = True
                break
            # Word-boundary match: "UNMARRIED" contains "MARRIED" but should match "UNMARRIED"
            if known.upper() in label_clean:
                # Check it's not a substring of a longer known group
                longer_match = False
                for other in _KNOWN_GROUPS:
                    if len(other) > len(known) and other.upper() in label_clean:
                        longer_match = True
                        break
                if not longer_match:
                    group_names.append(known.upper())
                    matched = True
                    break
        if not matched:
            # Check if it looks like a group name (all caps, not a label)
            # but only if not a known label-like word
            label_norm = re.sub(r'[^a-zA-Z]', '', label).lower()
            if label_norm in ("district", "religion", "age", "srl", "no",
                              "communities", "and", "ages", "sex", "civil",
                              "condition", "ommunities", "sno"):
                continue
            # Skip column numbers
            if re.match(r'^\d{1,2}$', label.strip()):
                continue
            if _classify_pmf(label) is not None:
                continue
            # Only add if it looks substantive (not garbled OCR)
            if len(label_clean) > 3 and not any(
                    c in label_clean.lower() for c in
                    ["communit", "religion", "district", "srl", "age"]):
                group_names.append(label_clean)

    # Truncate to expected number of groups
    group_names = group_names[:expected_groups]

    # Pad with generic names if needed
    while len(group_names) < expected_groups:
        group_names.append(f"GROUP_{len(group_names)+1}")

    # Build col_map
    col_map = []
    for gname in group_names:
        if has_persons:
            col_map.append((gname, "persons"))
            col_map.append((gname, "males"))
            col_map.append((gname, "females"))
        else:
            col_map.append((gname, "males"))
            col_map.append((gname, "females"))

    return group_names, col_map, has_persons, data_start


def _find_data_start_in_row(row, n_expected_data_cols, sub_data_start):
    """Dynamically detect where numeric data starts in a data row.

    Strategy:
    1. Scan from right counting numeric cells to find data boundary
    2. Use subheader data_start as reference to resolve ambiguity
    3. Handle ".." placeholders in label columns

    Returns the column index where data begins.
    """
    n_cells = len(row)
    if n_cells <= n_expected_data_cols:
        return 0

    # Count how many trailing cells look numeric (or ".." / empty)
    numeric_from_right = 0
    for i in range(n_cells - 1, -1, -1):
        cell = row[i].strip()
        if not cell or cell in ("..", ".", "—", "-", "…"):
            numeric_from_right += 1
        elif re.match(r'^[\d,]+$', cell.replace(" ", "")):
            numeric_from_right += 1
        elif "·" in cell:  # Proportional table decimal
            numeric_from_right += 1
        else:
            break

    candidate = n_cells - numeric_from_right

    # The right-scan might include ".." label placeholders as data.
    # Use sub_data_start as a reference: data shouldn't start before the
    # first real number. Scan forward from candidate to find first real number.
    if candidate < sub_data_start:
        # Check if cells between candidate and sub_data_start are just ".." placeholders
        for i in range(candidate, min(sub_data_start + 1, n_cells)):
            cell = row[i].strip()
            if cell and re.match(r'^[\d,]+$', cell.replace(" ", "")):
                # Found a real number — data starts here
                return i
        # All were ".." — data probably starts at sub_data_start
        return sub_data_start

    return candidate


def _is_age_label(text):
    """Check if text looks like an age label or age-related row label."""
    t = text.strip()
    return bool(
        re.match(r'^\d+[-—–]\d+$', t) or
        re.match(r'^\d+\s*[-—–]\s*\d+$', t) or
        re.match(r'^\d+\s*(and|&)\s*over', t, re.I) or
        re.match(r'^(total|grand\s*total)', t, re.I) or
        re.match(r'^TOTAL\s+\d', t, re.I) or
        t.lower() in ("mean age", "not stated", "age not stated")
    )


def parse_mistral_markdown(raw_text):
    """Parse Mistral OCR markdown output into standard parsed dict.

    Handles format variations across 1891-1941 census tables:
    - 1931 Hyderabad: 2 prefix cols (District, Religion/Age), 4 groups (P/M/F each)
    - 1941 Hyderabad: 2 prefix cols (Srl.No., Communities/Age), 5 groups including DIVORCED
    - 1901 Travancore: 1 prefix col (Age), 2+ groups
    - Proportional tables: decimal values with middle dot (·)

    Returns:
        dict with "metadata" and "sections" keys (same format as oneshot.py),
        or None on parse failure.
    """
    tables = _extract_markdown_tables(raw_text)
    if not tables:
        logger.warning("No markdown tables found")
        return None

    # Use the largest table (most rows)
    table = max(tables, key=len)

    if len(table) < 3:
        logger.warning("Table too small: %d rows", len(table))
        return None

    # Find the sub-header row (the one with persons/males/females)
    subheader_idx = None
    header_idx = None
    for i, row in enumerate(table[:6]):
        pmf_count = sum(1 for c in row if _classify_pmf(c) is not None)
        if pmf_count >= 3:
            subheader_idx = i
            header_idx = max(0, i - 1)
            break

    if subheader_idx is None:
        logger.warning("Could not find P/M/F sub-header row")
        return None

    # Parse headers
    groups, col_map, has_persons, sub_data_start = _detect_header_groups(
        table[header_idx], table[subheader_idx])

    if not groups or not col_map:
        logger.warning("Could not detect column groups")
        return None

    n_data_cols = len(col_map)

    # Find where data rows start (skip column-number rows)
    data_start_row = subheader_idx + 1
    while data_start_row < len(table) and _is_number_row(table[data_start_row]):
        data_start_row += 1

    # Parse data rows into sections
    sections = []
    current_section = None
    current_rows = []

    for row_idx in range(data_start_row, len(table)):
        row = table[row_idx]
        if not row:
            continue

        # Skip rows that are mostly empty
        non_empty = [c for c in row if c.strip()]
        if len(non_empty) < 2:
            continue

        # Dynamically find where data starts in THIS row
        row_data_start = _find_data_start_in_row(row, n_data_cols, sub_data_start)

        # Extract label text (everything before data columns)
        label_parts = []
        for i in range(min(row_data_start, len(row))):
            cell = row[i].strip()
            if cell and cell not in ("..", ".", "—", "-"):
                label_parts.append(cell)
        label_text = " ".join(label_parts).strip()

        # Clean up label
        label_text = re.sub(r'^\d+\.\s*', '', label_text)  # Remove "1." prefix
        label_text = label_text.replace("&amp;", "&")
        # Remove hyphenation artifacts from OCR (e.g. "COM-\nMUNITIES" → "COMMUNITIES")
        label_text = re.sub(r'-\s+', '', label_text) if "COM-" in label_text else label_text

        # Extract data values
        data_values = []
        for i in range(row_data_start, min(row_data_start + n_data_cols, len(row))):
            data_values.append(row[i])

        # Classify row: age data row vs section header
        is_age = _is_age_label(label_text)

        if not is_age:
            # Check if this is a section header
            if not label_text:
                continue

            # Save previous section
            if current_section is not None and current_rows:
                sections.append({"name": current_section, "rows": current_rows})
            current_section = label_text
            current_rows = []

            # Section header with totals — add as "Total" row
            if data_values:
                has_real_numbers = any(
                    _parse_cell_value(v) is not None and
                    _parse_cell_value(v) != 0 and
                    v.strip() not in ("..", ".", "—", "-", "")
                    for v in data_values[:6])
                if has_real_numbers:
                    total_row = {"age": "Total"}
                    for ci, (gname, col) in enumerate(col_map):
                        if gname not in total_row:
                            total_row[gname] = {}
                        total_row[gname][col] = (
                            _parse_cell_value(data_values[ci])
                            if ci < len(data_values) else None)
                    current_rows.append(total_row)
        else:
            # Age data row
            if current_section is None:
                current_section = "All"

            age_label = label_text.replace("—", "-").replace("–", "-").strip()
            if not age_label:
                continue

            data_row = {"age": age_label}
            for ci, (gname, col) in enumerate(col_map):
                if gname not in data_row:
                    data_row[gname] = {}
                data_row[gname][col] = (
                    _parse_cell_value(data_values[ci])
                    if ci < len(data_values) else None)

            # Compute persons = males + females if no persons column
            if not has_persons:
                for gname in groups:
                    gdata = data_row.get(gname, {})
                    m = gdata.get("males")
                    f = gdata.get("females")
                    if m is not None and f is not None:
                        gdata["persons"] = m + f

            current_rows.append(data_row)

    # Save last section
    if current_section is not None and current_rows:
        sections.append({"name": current_section, "rows": current_rows})

    if not sections:
        logger.warning("No sections parsed from markdown")
        return None

    # Build metadata
    metadata = {
        "column_groups": groups,
        "data_type": "absolute",
    }

    # Detect proportional tables (denominator in title like "1,000 of each sex")
    title_match = re.search(r'(\d[,\d]*)\s+of\s+each\s+sex', raw_text, re.I)
    if title_match:
        denom = int(title_match.group(1).replace(",", ""))
        metadata["data_type"] = "proportional"
        metadata["denominator"] = denom
        metadata["title"] = raw_text.split("\n")[0][:200]

    parsed = {"metadata": metadata, "sections": sections}
    return parsed


# ---------------------------------------------------------------------------
# Schema inference from parsed data (no Gemini call needed)
# ---------------------------------------------------------------------------

def infer_schema_from_parsed(parsed):
    """Build a minimal TableSchema from parsed data for constraint derivation.

    This avoids the Gemini schema discovery call entirely.
    """
    meta = parsed.get("metadata", {})
    groups = meta.get("column_groups", [])
    sections = parsed.get("sections", [])

    # Detect persons column
    has_persons = False
    if sections:
        first_row = sections[0].get("rows", [{}])[0]
        for g in groups:
            gdata = first_row.get(g, {})
            if isinstance(gdata, dict) and "persons" in gdata:
                has_persons = True
                break

    # Detect data type
    data_type = meta.get("data_type", "absolute")
    denominator = meta.get("denominator", 0)

    # Build row labels from first section
    row_labels = []
    if sections:
        for row in sections[0].get("rows", []):
            row_labels.append(row.get("age", ""))

    # Detect subtotal hierarchy from row labels
    subtotal_hierarchy = {}
    # Look for "Total X-Y" type labels
    for label in row_labels:
        m = re.match(r'^Total\s+(.+)', label, re.I)
        if m:
            sub_range = m.group(1).strip()
            # This is a subtotal — find its components
            # (derive_constraints handles this automatically)

    schema = TableSchema(
        title=meta.get("title", ""),
        region="",
        year=0,
        data_type=data_type,
        denominator=denominator,
        column_groups=[{"name": g} for g in groups],
        has_persons_column=has_persons,
        row_labels=row_labels,
        sections=[{"name": s.get("name", "")} for s in sections],
    )

    return schema


# ---------------------------------------------------------------------------
# Parse quality validation
# ---------------------------------------------------------------------------

def _validate_parse(parsed):
    """Check if the parsed data looks reasonable.

    Returns (is_valid, reason) tuple.
    """
    if parsed is None:
        return False, "parse returned None"

    sections = parsed.get("sections", [])
    if not sections:
        return False, "no sections"

    groups = parsed.get("metadata", {}).get("column_groups", [])
    if not groups:
        return False, "no column groups"

    # Check minimum row count
    total_rows = sum(len(s.get("rows", [])) for s in sections)
    if total_rows < 3:
        return False, f"too few rows ({total_rows})"

    # Check that data rows have actual numeric values
    non_null_count = 0
    total_cells = 0
    for sec in sections:
        for row in sec.get("rows", []):
            for g in groups:
                gdata = row.get(g, {})
                if isinstance(gdata, dict):
                    for col in ("persons", "males", "females"):
                        val = gdata.get(col)
                        total_cells += 1
                        if val is not None:
                            non_null_count += 1

    if total_cells == 0:
        return False, "no data cells"

    fill_rate = non_null_count / total_cells
    if fill_rate < 0.3:
        return False, f"low fill rate ({fill_rate:.0%})"

    return True, "ok"


# ---------------------------------------------------------------------------
# Main hybrid pipeline
# ---------------------------------------------------------------------------

def extract_and_verify_mistral(image_path, full_fallback=False):
    """Hybrid Mistral OCR + constraint repair + optional Gemini fallback.

    Args:
        image_path: Path to census table image.
        full_fallback: If True, allow full Gemini fallback on structural failure.

    Returns:
        dict with keys: parsed, schema, constraints, report, method, api_calls, cost, elapsed
    """
    image_path = Path(image_path)
    if not image_path.exists():
        print(f"ERROR: Image not found: {image_path}")
        return None

    # Output stem (match oneshot.py naming convention)
    parent = image_path.parent.name
    if parent.isdigit() and not image_path.stem.endswith(f"_{parent}"):
        stem = f"{image_path.stem}_{parent}"
    else:
        stem = image_path.stem

    print(f"\n{'='*70}")
    print(f"Mistral Hybrid: {image_path.name}")
    print(f"{'='*70}")

    t0 = time.time()
    mistral_calls = 0
    gemini_calls = 0
    cost = 0.0

    # Step 1: Mistral OCR
    print("\nStep 1: Mistral OCR")
    try:
        raw_text, ocr_elapsed = call_mistral_ocr(image_path)
        mistral_calls += 1
        cost += 0.002
        print(f"  OCR done in {ocr_elapsed:.1f}s ({len(raw_text)} chars)")
    except Exception as e:
        print(f"  Mistral OCR failed: {e}")
        if full_fallback:
            print("  Falling back to full Gemini pipeline...")
            return _gemini_fallback(image_path, t0)
        return None

    # Save raw text for debugging
    raw_path = RESULTS_DIR / f"{stem}_mistral_hybrid_raw.txt"
    raw_path.write_text(raw_text)

    # Step 2: Parse markdown → structured JSON
    print("\nStep 2: Parse markdown")
    parsed = parse_mistral_markdown(raw_text)

    is_valid, reason = _validate_parse(parsed)
    if not is_valid:
        print(f"  Parse failed: {reason}")
        if full_fallback:
            print("  Falling back to full Gemini pipeline...")
            return _gemini_fallback(image_path, t0, mistral_calls, cost)
        return None

    groups = parsed["metadata"].get("column_groups", [])
    n_sections = len(parsed["sections"])
    n_rows = sum(len(s.get("rows", [])) for s in parsed["sections"])
    print(f"  Groups: {groups}")
    print(f"  Sections: {n_sections}, Rows: {n_rows}")

    # Step 3: Infer schema + derive constraints
    print("\nStep 3: Derive constraints")
    schema = infer_schema_from_parsed(parsed)

    # Detect persons_independent
    persons_independent = False
    if schema.data_type == "proportional":
        title_lower = (schema.title or "").lower()
        if "of each sex" in title_lower or "each sex" in title_lower:
            persons_independent = True
            print("  Detected: persons_independent (proportional 'of each sex')")

    constraints = derive_constraints(parsed, schema=schema,
                                     persons_independent=persons_independent)

    report = verify_all_constraints(parsed, constraints)
    print(f"  Constraints: {report['total_checks']} checks, "
          f"{report['passed']} passed, {report['failed']} failed")

    if report["all_passed"]:
        print("  ALL CONSTRAINTS PASS — done!")
        return _build_result(parsed, schema, constraints, report,
                           "mistral_only", stem, image_path,
                           mistral_calls, gemini_calls, cost, t0)

    # Step 4: Free repair (Phases A + B only)
    print("\nStep 4: Constraint repair (free phases)")

    # Phase A: M/F swap
    parsed, log_a = _detect_and_fix_mf_swaps(
        parsed, constraints, persons_independent)
    if log_a:
        constraints = derive_constraints(parsed, schema=schema,
                                        persons_independent=persons_independent)
        report = verify_all_constraints(parsed, constraints)
        print(f"  Phase A: {len(log_a)} swaps → {report['failed']} failures")
        if report["all_passed"]:
            return _build_result(parsed, schema, constraints, report,
                               "mistral+swap", stem, image_path,
                               mistral_calls, gemini_calls, cost, t0)

    # Phase B: Digit deduction
    parsed, log_b = _deductive_digit_fix(
        parsed, constraints, persons_independent)
    if log_b:
        constraints = derive_constraints(parsed, schema=schema,
                                        persons_independent=persons_independent)
        report = verify_all_constraints(parsed, constraints)
        print(f"  Phase B: {len(log_b)} digit fixes → {report['failed']} failures")
        if report["all_passed"]:
            return _build_result(parsed, schema, constraints, report,
                               "mistral+repair", stem, image_path,
                               mistral_calls, gemini_calls, cost, t0)

    # Second round of A+B (repairs can unlock more)
    for extra_round in range(2):
        parsed, log_a2 = _detect_and_fix_mf_swaps(
            parsed, constraints, persons_independent)
        parsed, log_b2 = _deductive_digit_fix(
            parsed, constraints, persons_independent)
        if log_a2 or log_b2:
            constraints = derive_constraints(parsed, schema=schema,
                                            persons_independent=persons_independent)
            report = verify_all_constraints(parsed, constraints)
            if log_a2:
                print(f"  Phase A (round {extra_round+2}): {len(log_a2)} swaps")
            if log_b2:
                print(f"  Phase B (round {extra_round+2}): {len(log_b2)} fixes")
            print(f"  → {report['failed']} failures remaining")
            if report["all_passed"]:
                return _build_result(parsed, schema, constraints, report,
                                   "mistral+repair", stem, image_path,
                                   mistral_calls, gemini_calls, cost, t0)
        else:
            break

    n_failures = report["failed"]
    print(f"\n  {n_failures} failures remain after free repair")

    # Step 5: Gemini targeted recheck (if few failures)
    if n_failures <= 20 and n_failures > 0:
        print("\nStep 5: Gemini targeted recheck")
        try:
            suspicious = identify_suspicious_cells(parsed, report["failures"])
            if suspicious:
                parsed_new = targeted_recheck(image_path, parsed, suspicious)
                gemini_calls += 1
                cost += 0.03
                if parsed_new:
                    parsed = parsed_new
                    constraints = derive_constraints(
                        parsed, schema=schema,
                        persons_independent=persons_independent)
                    report = verify_all_constraints(parsed, constraints)
                    print(f"  After recheck: {report['failed']} failures")

                    if not report["all_passed"]:
                        # Try repair again after recheck
                        for _ in range(2):
                            parsed, log_ar = _detect_and_fix_mf_swaps(
                                parsed, constraints, persons_independent)
                            parsed, log_br = _deductive_digit_fix(
                                parsed, constraints, persons_independent)
                            if log_ar or log_br:
                                constraints = derive_constraints(
                                    parsed, schema=schema,
                                    persons_independent=persons_independent)
                                report = verify_all_constraints(parsed, constraints)
                                if report["all_passed"]:
                                    break
                            else:
                                break

                    if report["all_passed"]:
                        print("  ALL CONSTRAINTS PASS after recheck!")
                        return _build_result(parsed, schema, constraints, report,
                                           "mistral+gemini_recheck", stem, image_path,
                                           mistral_calls, gemini_calls, cost, t0)
        except Exception as e:
            print(f"  Recheck failed: {e}")

    # Step 6: Full Gemini fallback
    if full_fallback and not report["all_passed"]:
        print("\nStep 6: Full Gemini fallback")
        return _gemini_fallback(image_path, t0, mistral_calls, cost)

    # Return best effort
    method = "mistral+partial_repair"
    if gemini_calls > 0:
        method = "mistral+gemini_partial"
    return _build_result(parsed, schema, constraints, report,
                        method, stem, image_path,
                        mistral_calls, gemini_calls, cost, t0)


def _gemini_fallback(image_path, t0, mistral_calls=0, mistral_cost=0.0):
    """Full Gemini fallback using the standard oneshot pipeline."""
    result = extract_and_verify(image_path)
    if result is None:
        return None

    elapsed = time.time() - t0
    gemini_calls = result.get("parsed", {}) and 3  # typical
    return {
        "parsed": result["parsed"],
        "schema": result["schema"],
        "constraints": result["constraints"],
        "report": result["report"],
        "method": "gemini_fallback",
        "mistral_calls": mistral_calls,
        "gemini_calls": gemini_calls,
        "cost": mistral_cost + gemini_calls * 0.03,
        "elapsed": elapsed,
    }


def _build_result(parsed, schema, constraints, report, method, stem,
                  image_path, mistral_calls, gemini_calls, cost, t0):
    """Build standard result dict and save outputs."""
    elapsed = time.time() - t0

    # Export
    output_paths = {}

    # JSON
    json_path = RESULTS_DIR / f"{stem}_mistral_hybrid.json"
    json_output = {
        "source_image": str(image_path),
        "extraction_method": method,
        "mistral_calls": mistral_calls,
        "gemini_calls": gemini_calls,
        "api_calls": mistral_calls + gemini_calls,
        "cost_estimate": round(cost, 4),
        "elapsed_seconds": round(elapsed, 1),
        "metadata": parsed.get("metadata", {}),
        "data": parsed["sections"],
        "constraints": {
            "total_checks": report["total_checks"],
            "passed": report["passed"],
            "failed": report["failed"],
            "all_passed": report["all_passed"],
            "failures": report["failures"],
        },
    }
    with open(json_path, "w") as f:
        json.dump(json_output, f, indent=2)
    output_paths["json"] = str(json_path)

    print(f"\nDone in {elapsed:.1f}s | method={method} | "
          f"mistral={mistral_calls} gemini={gemini_calls} | "
          f"cost=${cost:.3f} | "
          f"{report['total_checks']} checks, {report['failed']} failures")

    return {
        "parsed": parsed,
        "schema": schema,
        "constraints": constraints,
        "report": report,
        "method": method,
        "mistral_calls": mistral_calls,
        "gemini_calls": gemini_calls,
        "cost": cost,
        "elapsed": elapsed,
        "output_paths": output_paths,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Mistral OCR hybrid pipeline")
    parser.add_argument("image", nargs="?", help="Image path")
    parser.add_argument("--batch", help="Process all PNGs in directory")
    parser.add_argument("--full-fallback", action="store_true",
                       help="Allow full Gemini fallback on failure")
    args = parser.parse_args()

    if args.batch:
        directory = Path(args.batch)
        for img in sorted(directory.glob("*.png")):
            extract_and_verify_mistral(img, full_fallback=args.full_fallback)
    elif args.image:
        extract_and_verify_mistral(args.image, full_fallback=args.full_fallback)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
