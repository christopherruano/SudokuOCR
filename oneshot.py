"""
Single-shot extraction pipeline for historical Indian census tables.

Architecture: 1 Gemini API call + comprehensive Python-only constraint
verification + data-science-friendly exports (tidy CSV, Excel, JSON, Parquet).

Usage:
    python3 oneshot.py IMAGE                           # single image
    python3 oneshot.py --batch DIR                     # all PNGs in directory
    python3 oneshot.py IMAGE --parquet                 # also emit .parquet
    python3 oneshot.py IMAGE --fallback                # auto-fallback to MoE on failure
"""

import os
import re
import sys
import json
import time
import logging
import argparse
from pathlib import Path
from collections import Counter

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(PROJECT_ROOT))
from pipeline import (call_gemini, encode_image, parse_json_response,
                       normalize_age, export_multigroup_to_excel,
                       export_multigroup_to_json)
from schema_discovery import (TableSchema, SCHEMA_DISCOVERY_PROMPT,
                               _parse_schema_json, _dict_to_schema)


# ---------------------------------------------------------------------------
# Schema-aware extraction: discover_schema_single + build_oneshot_extraction_prompt
# ---------------------------------------------------------------------------

def discover_schema_single(image_path):
    """Single-model schema discovery using Gemini (Call 1).

    Sends SCHEMA_DISCOVERY_PROMPT to Gemini only — no multi-model consensus,
    no reconciliation. Fast (~10s) and cheap.

    Returns:
        TableSchema instance, or None on failure.
    """
    b64 = encode_image(str(image_path))
    for attempt in range(3):
        try:
            raw = call_gemini(b64, SCHEMA_DISCOVERY_PROMPT)
            d = _parse_schema_json(raw)
            if d is not None:
                return _dict_to_schema(d)
            logger.warning("Schema discovery: parse failed")
            return None
        except Exception as e:
            wait = 2 ** attempt
            logger.warning("Schema discovery attempt %d failed: %s", attempt + 1, e)
            if attempt < 2:
                time.sleep(wait)
    return None


def build_oneshot_extraction_prompt(schema, persons_independent=False,
                                    target_section=None, target_group=None):
    """Generate a schema-tailored prompt for full multi-group extraction (Call 2).

    Unlike build_extraction_prompt() in schema_discovery.py (which returns a flat
    JSON array for a single group), this returns the full nested structure for
    ALL groups at once — matching the oneshot multi-section JSON shape.

    Args:
        schema: TableSchema instance.
        persons_independent: If True, P/M/F are each independent distributions
            (e.g. "of each sex" tables) — do NOT enforce P=M+F.
        target_section: If set, only extract this one section (by name).
            Used for multi-section tables to avoid timeout.
        target_group: If set, only extract this one column group (by name).
            Used when all-at-once extraction fails (e.g. large output).
    """
    import re as _re

    # --- Column groups ---
    all_group_names = [cg.get("name", f"Group_{i}")
                       for i, cg in enumerate(schema.column_groups)]
    if target_group:
        group_names = [target_group]
    else:
        group_names = all_group_names
    groups_str = ", ".join(f'"{g}"' for g in group_names)

    # --- Persons column ---
    if persons_independent:
        pmf_instruction = (
            'Each group has sub-columns: "persons", "males", "females".\n'
            "IMPORTANT: Persons, Males, and Females are each INDEPENDENT distributions "
            f"(each sums to {schema.denominator:,} across all age rows). "
            "Persons does NOT equal Males + Females."
        )
        pmf_keys = '"persons": N, "males": N, "females": N'
    elif schema.has_persons_column:
        pmf_instruction = (
            'Each group has sub-columns: "persons", "males", "females".\n'
            "Persons MUST equal Males + Females — verify each row as you read."
        )
        pmf_keys = '"persons": N, "males": N, "females": N'
    else:
        pmf_instruction = (
            'Each group has sub-columns: "males", "females" (no Persons column in the image).\n'
            "Compute persons = males + females for each row and include it in your output."
        )
        pmf_keys = '"persons": N, "males": N, "females": N'

    # --- Row labels ---
    def _clean_label(label):
        if _re.match(r'(?i)^total\s*[.\-…]+$', label.strip()):
            return "Total"
        return label

    if schema.row_labels:
        clean_labels = [_clean_label(l) for l in schema.row_labels]
        row_list = "\n".join(f"  - {l}" for l in clean_labels)
        row_instruction = f"The table contains these rows (in order):\n{row_list}"
    else:
        row_instruction = (
            "Extract all age-group rows visible in the table, including "
            "any subtotal rows and the grand Total row."
        )

    # --- Subtotal hierarchy ---
    subtotal_instruction = ""
    if schema.subtotal_hierarchy:
        parts = []
        for sub_label, comp_labels in schema.subtotal_hierarchy.items():
            clean_sub = _clean_label(sub_label)
            clean_comps = [_clean_label(c) for c in comp_labels]
            parts.append(f"  - {clean_sub} = {' + '.join(clean_comps)}")
        subtotal_instruction = (
            "\n\nSubtotal relationships (use to verify your readings):\n"
            + "\n".join(parts)
            + "\nIf a subtotal doesn't match the sum of its components, "
            "re-examine the ambiguous digits."
        )

    # --- Data type ---
    if schema.data_type == "proportional" and schema.denominator > 0:
        denom = schema.denominator
        data_instruction = (
            f"\n\nIMPORTANT: This table shows proportions per {denom:,}, "
            f"NOT absolute population counts. The Total row should equal exactly "
            f"{denom:,} for Males and Females. "
            "Use this constraint to verify your readings."
        )
    else:
        data_instruction = ""

    # --- Cross-group constraints (skip when extracting a single group) ---
    cross_group_instruction = ""
    if schema.cross_group_constraints and not target_group:
        parts = []
        for total_g, comp_gs in schema.cross_group_constraints.items():
            parts.append(f"  - {total_g} = {' + '.join(comp_gs)}")
        cross_group_instruction = (
            "\n\nCross-group constraints (must hold for every row and sex):\n"
            + "\n".join(parts)
            + "\nVerify this for each row. If it fails, re-read the conflicting digits."
        )

    # --- Sections ---
    if target_section:
        section_instruction = (
            f'\n\nThis table has multiple sections. Extract ONLY the section '
            f'labeled "{target_section}". Ignore all other sections.'
        )
    elif schema.sections:
        sec_names = [s.get("name", "?") for s in schema.sections]
        section_instruction = (
            f"\n\nThis table has {len(schema.sections)} sections: "
            + ", ".join(f'"{n}"' for n in sec_names)
            + ".\nExtract each section separately."
        )
    else:
        section_instruction = ""

    # --- Target group (per-group extraction) ---
    group_instruction = ""
    if target_group:
        group_instruction = (
            f'\n\nExtract ONLY the column group labeled "{target_group}". '
            f"Ignore all other column groups in the table."
        )

    # Build the example structure
    example_group_data = ", ".join(
        f'"{g}": {{{pmf_keys}}}' for g in group_names[:2]
    )
    if len(group_names) > 2:
        example_group_data += ", ..."

    prompt = f"""You are an expert at reading historical Indian census tables from scanned images.

Extract ALL data from this table into structured JSON.

Column groups in this table: [{groups_str}]
{pmf_instruction}

{row_instruction}{subtotal_instruction}{data_instruction}{cross_group_instruction}{section_instruction}{group_instruction}

Return a JSON object with this structure:
{{
  "metadata": {{
    "title": "...",
    "region": "...",
    "year": {schema.year or 'NNNN'},
    "data_type": "{schema.data_type}",
    "column_groups": [{groups_str}]
  }},
  "sections": [
    {{
      "name": "...",
      "rows": [
        {{"age": "0-1", {example_group_data}}},
        ...
        {{"age": "Total", {example_group_data}}}
      ]
    }}
  ]
}}

RULES:
1. All values as integers — NO commas, NO spaces in numbers
2. ".." or "." in the table means 0
3. Use null for truly illegible/missing values
4. Include ALL age rows AND the Total row for each section
5. If there is only one section, use the table title as the section name
6. Be extremely careful with similar digits: 3 vs 8, 5 vs 6, 0 vs 9

VERIFY YOUR WORK before returning:
{f"- Each row: Persons, Males, Females are INDEPENDENT — do NOT cross-check them" if persons_independent else "- Each row: Persons MUST equal Males + Females"}
- Each section: Total row MUST equal the sum of all age rows below it
- Cross-group sums must hold if applicable
Re-examine any digits involved in a failed check.

Return ONLY valid JSON. No explanation."""

    return prompt


# ---------------------------------------------------------------------------
# parse_response() — normalize raw Gemini JSON
# ---------------------------------------------------------------------------

def parse_response(raw_text, schema=None, persons_independent=False):
    """Parse and normalize raw Gemini JSON response into internal format.

    Handles:
    - Markdown code fences
    - Commas in numbers (e.g. "823,622" -> 823622)
    - ".." or "." entries -> 0
    - null values preserved as None
    - Underscore separators in numbers (e.g. 823_622)
    - M/F-only tables: computes persons = males + females when schema
      indicates has_persons_column=False (and persons_independent=False)

    Returns:
        dict with "metadata" and "sections" keys, or None on failure.
    """
    text = raw_text.strip()

    # Strip markdown code fences
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)

    # Remove underscore digit separators
    text = re.sub(r'(?<=\d)_(?=\d)', '', text)

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        # Try to find JSON object
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                parsed = json.loads(text[start:end])
            except json.JSONDecodeError:
                return None
        else:
            return None

    if not isinstance(parsed, dict):
        return None

    # Normalize: ensure we have metadata and sections
    if "metadata" not in parsed:
        parsed["metadata"] = {}
    if "sections" not in parsed:
        # Maybe the model returned a flat structure — try to adapt
        if "rows" in parsed:
            parsed["sections"] = [{"name": "All", "rows": parsed["rows"]}]
        else:
            return None

    # Normalize all numeric values in sections
    for section in parsed["sections"]:
        for row in section.get("rows", []):
            for key, val in row.items():
                if key == "age":
                    continue
                if isinstance(val, dict):
                    # Column group dict: {"persons": N, "males": N, "females": N}
                    for col in ("persons", "males", "females"):
                        if col in val:
                            val[col] = _normalize_value(val[col])
                elif isinstance(val, (int, float, str)):
                    row[key] = _normalize_value(val)

    # If schema says no persons column AND persons are not independent,
    # compute persons = males + females.  For "of each sex" tables
    # (persons_independent=True), P is its own per-N distribution ≠ M+F.
    if schema is not None and not schema.has_persons_column and not persons_independent:
        for section in parsed["sections"]:
            for row in section.get("rows", []):
                for key, val in row.items():
                    if key == "age":
                        continue
                    if isinstance(val, dict):
                        m = val.get("males")
                        f = val.get("females")
                        if m is not None and f is not None:
                            if val.get("persons") is None:
                                val["persons"] = m + f

    return parsed


def _normalize_value(val):
    """Normalize a single cell value to int, 0, or None."""
    if val is None:
        return None
    if isinstance(val, int):
        return val
    if isinstance(val, float):
        return int(round(val))
    s = str(val).strip()
    if s in ("..", ".", "-", "—", ""):
        return 0
    # Remove commas and spaces
    s = s.replace(",", "").replace(" ", "")
    try:
        return int(s)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# derive_constraints() — auto-discover constraints from data shape
# ---------------------------------------------------------------------------

def derive_constraints(parsed, schema=None, persons_independent=False):
    """Auto-discover all constraint levels from the extracted data structure.

    When schema is provided, enriches constraints with:
    - Subtotal hierarchy (intermediate L2 checks, not just Total = sum)
    - Known totals for proportional tables (Total M=1000, F=1000)
    - Cross-group constraints from schema (even if auto-detection misses)

    Returns a dict describing all constraints:
    {
        "L1_row": [(section, age, group)],       # P = M + F
        "L2_vertical": [(section, group, col)],   # Total = sum(age rows)
        "L2_subtotal": [...],                      # intermediate subtotals
        "L3_cross_group": {                        # Pop = sum(other groups)
            "total_group": str,
            "component_groups": [str],
            "cells": [(section, age, col)],
        } or None,
        "L4_cross_section": {                      # ALL = sum(community sections)
            "total_section": str,
            "component_sections": [str],
            "cells": [(age, group, col)],
        } or None,
        "L5_non_negative": [(section, age, group, col)],
        "known_totals": {...} or None,
    }
    """
    sections = parsed.get("sections", [])
    meta = parsed.get("metadata", {})
    groups = meta.get("column_groups", [])

    # If groups not in metadata, discover from first section's first row
    if not groups and sections:
        first_row = sections[0].get("rows", [{}])[0]
        groups = [k for k in first_row.keys()
                  if k != "age" and isinstance(first_row.get(k), dict)]

    # Track whether the table is proportional (rounding tolerance for L2 checks)
    is_proportional = (schema is not None and schema.data_type == "proportional")
    if not is_proportional and meta.get("data_type") == "proportional":
        is_proportional = True

    # Auto-detect persons_independent from title when not explicitly set
    if not persons_independent:
        title_lower = (meta.get("title") or "").lower()
        if "of each sex" in title_lower or "each sex" in title_lower:
            persons_independent = True

    constraints = {
        "L1_row": [],
        "L2_vertical": [],
        "L2_subtotal": [],
        "is_proportional": is_proportional,
        "L3_cross_group": None,
        "L4_cross_section": None,
        "L5_non_negative": [],
        "known_totals": None,
        "column_groups": groups,
    }

    for si, section in enumerate(sections):
        sec_name = section.get("name", f"Section_{si}")
        rows = section.get("rows", [])

        for row in rows:
            age = row.get("age", "")
            for group in groups:
                gdata = row.get(group)
                if not isinstance(gdata, dict):
                    continue

                # L1: P = M + F (skip for independent-column tables)
                if not persons_independent:
                    p = gdata.get("persons")
                    m = gdata.get("males")
                    f = gdata.get("females")
                    if p is not None and m is not None and f is not None:
                        constraints["L1_row"].append((sec_name, age, group))

                # L5: Non-negativity
                for col in ("persons", "males", "females"):
                    v = gdata.get(col)
                    if v is not None:
                        constraints["L5_non_negative"].append(
                            (sec_name, age, group, col))

        # L2: Vertical sum — find Total row
        # Exclude non-age rows like "Mean Age" from summation
        _NON_AGE_KEYS = {"mean age", "median age", "average age"}
        total_row = None
        grandtotal_row = None
        age_rows = []
        # Detect intermediate subtotals: rows whose raw label starts
        # with "Total " or "TOTAL " followed by an age range (e.g.
        # "Total 0-15", "TOTAL 40-60").  After normalize_age() these
        # become "0-15", "40-60" which look like leaf rows but aren't.
        _intermediate_subtotal_keys = set()
        for row in rows:
            raw = row.get("age", "")
            import re as _re2
            if _re2.match(r'(?i)^total\s+\S', raw):
                # "Total 0-15" → normalize → "0-15"; mark as subtotal
                _intermediate_subtotal_keys.add(normalize_age(raw))

        # Detect range-based subtotals: "0-5" after "0-1","1-2","2-3","3-4","4-5"
        # A row is a range subtotal if its numeric range exactly spans
        # the union of 2+ contiguous preceding rows' ranges.
        _ordered_ranges = []  # (norm_age, (start, end) or None)
        for row in rows:
            raw = row.get("age", "")
            age_norm = normalize_age(raw)
            if age_norm in ("total", "grandtotal") or age_norm in _NON_AGE_KEYS:
                _ordered_ranges.append((age_norm, None))
                continue
            m = _re2.match(r'^(\d+)-(\d+)$', age_norm)
            if m:
                _ordered_ranges.append(
                    (age_norm, (int(m.group(1)), int(m.group(2)))))
            else:
                _ordered_ranges.append((age_norm, None))

        for i in range(1, len(_ordered_ranges)):
            norm_i, rng_i = _ordered_ranges[i]
            if (rng_i is None
                    or norm_i in _intermediate_subtotal_keys):
                continue
            # Only ranges wider than single-year can be subtotals
            if rng_i[1] - rng_i[0] <= 1:
                continue
            # Walk backwards checking contiguous preceding rows
            j = i - 1
            acc_start = None
            count = 0
            while j >= 0:
                norm_j, rng_j = _ordered_ranges[j]
                if (rng_j is None
                        or norm_j in ("total", "grandtotal")
                        or norm_j in _NON_AGE_KEYS):
                    break
                if acc_start is None:
                    if rng_j[1] != rng_i[1]:
                        break  # End doesn't align
                    acc_start = rng_j[0]
                    count = 1
                else:
                    if rng_j[1] != acc_start:
                        break  # Not contiguous
                    acc_start = rng_j[0]
                    count += 1
                if acc_start == rng_i[0] and count >= 2:
                    # Rows j..i-1 tile exactly to rng_i
                    _intermediate_subtotal_keys.add(norm_i)
                    comp_norms = [_ordered_ranges[k][0]
                                  for k in range(j, i)]
                    for group in groups:
                        for col in ("persons", "males", "females"):
                            constraints["L2_subtotal"].append(
                                (sec_name, group, col,
                                 norm_i, comp_norms))
                    break
                j -= 1

        # Track row ordering: rows before Total are its components.
        # Rows between Total and Grand Total are Grand Total's extra
        # components (e.g. "Civil & Military Station" in Mysore 1891).
        pre_total_keys = []
        post_total_keys = []
        seen_total = False
        for row in rows:
            age_norm = normalize_age(row.get("age", ""))
            if age_norm == "total":
                total_row = row
                seen_total = True
            elif age_norm == "grandtotal":
                grandtotal_row = row
            elif age_norm in _NON_AGE_KEYS:
                continue  # Skip derived statistics
            elif age_norm in _intermediate_subtotal_keys:
                continue  # Skip intermediate subtotals (e.g. "Total 0-15")
            else:
                age_rows.append(row)
                if seen_total:
                    post_total_keys.append(age_norm)
                else:
                    pre_total_keys.append(age_norm)

        # Register auto-detected intermediate subtotals so they are
        # excluded from the L2 vertical leaf sum in verify_all_constraints
        if _intermediate_subtotal_keys:
            existing = constraints.get("_auto_subtotal_keys", set())
            existing.update(_intermediate_subtotal_keys)
            constraints["_auto_subtotal_keys"] = existing

        # When both Total and Grand Total exist, Total should sum only
        # the rows before it (districts), not the rows between Total
        # and Grand Total (e.g. Civil & Military Station).
        if total_row and grandtotal_row and post_total_keys:
            if "total_components" not in constraints or \
               not constraints["total_components"]:
                constraints["total_components"] = pre_total_keys
            # Also add L2 subtotal for Grand Total = Total + post_total rows
            gt_comps = ["total"] + post_total_keys
            for group in groups:
                for col in ("persons", "males", "females"):
                    constraints["L2_subtotal"].append(
                        (sec_name, group, col, "grandtotal", gt_comps))

        # Handle case where Grand Total appears BEFORE Total in row order
        # (e.g. Mysore 1891: districts, C&M Station, GRAND TOTAL, Total).
        # Here post_total_keys is empty but pre_total_keys includes extra
        # rows that shouldn't be in Total's sum.
        elif total_row and grandtotal_row and not post_total_keys:
            if "total_components" not in constraints or \
               not constraints["total_components"]:
                # Detect Total's true components by value matching:
                # try removing trailing pre_total rows until sum == Total
                rmap = {}
                for row in rows:
                    rmap[normalize_age(row.get("age", ""))] = row
                detected = False
                for group in groups:
                    if detected:
                        break
                    gt_data = total_row.get(group, {})
                    if not isinstance(gt_data, dict):
                        continue
                    for col in ("persons", "males", "females"):
                        t_val = gt_data.get(col)
                        if t_val is None or t_val == 0:
                            continue
                        for n_remove in range(1, min(len(pre_total_keys), 4)):
                            trial_keys = pre_total_keys[:-n_remove]
                            trial_sum = sum(
                                rmap.get(k, {}).get(group, {}).get(col, 0)
                                for k in trial_keys)
                            if trial_sum == t_val and trial_keys:
                                constraints["total_components"] = trial_keys
                                extra_keys = pre_total_keys[-n_remove:]
                                gt_comps = ["total"] + extra_keys
                                for g in groups:
                                    for c in ("persons", "males", "females"):
                                        constraints["L2_subtotal"].append(
                                            (sec_name, g, c,
                                             "grandtotal", gt_comps))
                                detected = True
                                break
                        if detected:
                            break

        if total_row and age_rows:
            for group in groups:
                for col in ("persons", "males", "females"):
                    gt = total_row.get(group, {})
                    if not isinstance(gt, dict):
                        continue
                    total_val = gt.get(col)
                    if total_val is not None:
                        constraints["L2_vertical"].append(
                            (sec_name, group, col))

    # L3: Cross-group auto-detection
    # Check if first group's totals equal sum of all other groups' totals
    # Skip for independent-column tables (year-groups are never additive)
    if len(groups) > 1 and sections and not persons_independent:
        first_group = groups[0]
        other_groups = groups[1:]
        match_count = 0
        close_count = 0   # matches within small tolerance
        check_count = 0

        for section in sections:
            total_row = None
            for row in section.get("rows", []):
                if normalize_age(row.get("age", "")) == "total":
                    total_row = row
                    break
            if not total_row:
                continue

            fg_data = total_row.get(first_group, {})
            if not isinstance(fg_data, dict):
                continue

            for col in ("persons", "males", "females"):
                fg_val = fg_data.get(col)
                if fg_val is None:
                    continue
                other_sum = 0
                all_present = True
                for og in other_groups:
                    og_data = total_row.get(og, {})
                    if not isinstance(og_data, dict):
                        all_present = False
                        break
                    og_val = og_data.get(col)
                    if og_val is None:
                        all_present = False
                        break
                    other_sum += og_val
                if all_present:
                    check_count += 1
                    if fg_val == other_sum:
                        match_count += 1
                        close_count += 1
                    elif abs(fg_val - other_sum) <= max(fg_val * 0.001, 10):
                        # Within 0.1% or 10 — likely L3 holds with a
                        # small extraction error in a minor section
                        close_count += 1

        # Activate L3 if at least 75% of checks match exactly AND all
        # remaining checks are within tolerance.  This catches the common
        # case where L3 holds structurally but a tiny digit error in one
        # section prevents exact detection.
        if (check_count > 0
                and close_count == check_count
                and match_count >= check_count * 0.75):
            # Cross-group constraint confirmed
            xg_cells = []
            for section in sections:
                sec_name = section.get("name", "")
                for row in section.get("rows", []):
                    age = row.get("age", "")
                    for col in ("persons", "males", "females"):
                        fg_data = row.get(first_group, {})
                        if isinstance(fg_data, dict) and fg_data.get(col) is not None:
                            xg_cells.append((sec_name, age, col))
            constraints["L3_cross_group"] = {
                "total_group": first_group,
                "component_groups": other_groups,
                "cells": xg_cells,
            }

    # L4: Cross-section auto-detection
    # Check if first section's totals equal sum of other sections' totals
    if len(sections) > 1:
        first_sec = sections[0]
        other_secs = sections[1:]

        # Build total row lookups
        def _get_total(sec):
            for row in sec.get("rows", []):
                if normalize_age(row.get("age", "")) == "total":
                    return row
            return None

        first_total = _get_total(first_sec)
        other_totals = [_get_total(s) for s in other_secs]

        if first_total and all(t is not None for t in other_totals):
            xs_match = 0
            xs_close = 0
            xs_check = 0
            for group in groups:
                fg_data = first_total.get(group, {})
                if not isinstance(fg_data, dict):
                    continue
                for col in ("persons", "males", "females"):
                    fg_val = fg_data.get(col)
                    if fg_val is None:
                        continue
                    other_sum = 0
                    all_present = True
                    for ot in other_totals:
                        ot_data = ot.get(group, {})
                        if not isinstance(ot_data, dict):
                            all_present = False
                            break
                        ot_val = ot_data.get(col)
                        if ot_val is None:
                            all_present = False
                            break
                        other_sum += ot_val
                    if all_present:
                        xs_check += 1
                        if fg_val == other_sum:
                            xs_match += 1
                            xs_close += 1
                        elif abs(fg_val - other_sum) <= max(fg_val * 0.001, 10):
                            xs_close += 1

            if (xs_check > 0 and xs_close == xs_check
                    and xs_match >= xs_check * 0.75):
                xs_cells = []
                for row in first_sec.get("rows", []):
                    age = row.get("age", "")
                    for group in groups:
                        for col in ("persons", "males", "females"):
                            gdata = row.get(group, {})
                            if isinstance(gdata, dict) and gdata.get(col) is not None:
                                xs_cells.append((age, group, col))
                constraints["L4_cross_section"] = {
                    "total_section": first_sec.get("name", ""),
                    "component_sections": [s.get("name", "")
                                           for s in other_secs],
                    "cells": xs_cells,
                }
            elif xs_check > 0:
                logger.info("Cross-section: partial coverage (%d/%d match) "
                            "— not an error, sections may not be exhaustive",
                            xs_match, xs_check)

    # --- Schema-enriched constraints ---
    if schema is not None:
        # Subtotal hierarchy → L2_subtotal checks
        if schema.subtotal_hierarchy:
            from schema_discovery import _clean_age_key
            for sub_label, comp_labels in schema.subtotal_hierarchy.items():
                norm_sub = _clean_age_key(normalize_age(sub_label))
                if not norm_sub:
                    continue
                norm_comps = [_clean_age_key(normalize_age(c))
                              for c in comp_labels]
                norm_comps = [c for c in norm_comps if c]
                if norm_sub == "total":
                    # Save Total's components so L2 vertical can use them
                    # instead of summing all non-total rows (which might
                    # include rows like "Civil & Military Station" that
                    # belong to a higher-level Grand Total, not to Total).
                    if norm_comps:
                        constraints["total_components"] = norm_comps
                    continue  # L2_vertical already covers Total
                if norm_comps:
                    for section in sections:
                        sec_name = section.get("name", "")
                        for group in groups:
                            for col in ("persons", "males", "females"):
                                constraints["L2_subtotal"].append(
                                    (sec_name, group, col, norm_sub, norm_comps))

        # Known totals for proportional tables
        if schema.data_type == "proportional" and schema.denominator > 0:
            denom = schema.denominator
            if persons_independent:
                # "of each sex" tables: P, M, F each independently sum to denom
                kt = {"total": {"persons": denom, "males": denom, "females": denom}}
            else:
                kt = {"total": {"males": denom, "females": denom}}
                if schema.has_persons_column:
                    kt["total"]["persons"] = denom
            constraints["known_totals"] = kt

        # Cross-group constraints from schema (if auto-detection missed)
        if schema.cross_group_constraints and constraints["L3_cross_group"] is None:
            for total_g, comp_gs in schema.cross_group_constraints.items():
                # Find the matching group names in the extracted data
                xg_cells = []
                for section in sections:
                    sec_name = section.get("name", "")
                    for row in section.get("rows", []):
                        age = row.get("age", "")
                        for col in ("persons", "males", "females"):
                            fg_data = row.get(total_g, {})
                            if isinstance(fg_data, dict) and fg_data.get(col) is not None:
                                xg_cells.append((sec_name, age, col))
                if xg_cells:
                    constraints["L3_cross_group"] = {
                        "total_group": total_g,
                        "component_groups": comp_gs,
                        "cells": xg_cells,
                    }
                break  # Only use the first cross-group constraint

    return constraints


# ---------------------------------------------------------------------------
# verify_all_constraints() — comprehensive multi-level checker
# ---------------------------------------------------------------------------

def verify_all_constraints(parsed, constraints):
    """Run all constraint checks, return structured report.

    Returns:
        dict with keys: total_checks, passed, failed, failures (list of dicts)
    """
    sections = parsed.get("sections", [])
    groups = constraints.get("column_groups", [])
    failures = []
    total_checks = 0

    # Build lookup: sec_name -> {norm_age -> row}
    sec_lookup = {}
    for section in sections:
        sec_name = section.get("name", "")
        rmap = {}
        for row in section.get("rows", []):
            age_norm = normalize_age(row.get("age", ""))
            rmap[age_norm] = row
        sec_lookup[sec_name] = rmap

    # L1: P = M + F
    for sec_name, age, group in constraints.get("L1_row", []):
        rmap = sec_lookup.get(sec_name, {})
        age_norm = normalize_age(age)
        row = rmap.get(age_norm)
        if not row:
            continue
        gdata = row.get(group, {})
        if not isinstance(gdata, dict):
            continue
        p = gdata.get("persons")
        m = gdata.get("males")
        f = gdata.get("females")
        if p is not None and m is not None and f is not None:
            total_checks += 1
            if p != m + f:
                failures.append({
                    "level": "L1",
                    "section": sec_name,
                    "age": age,
                    "group": group,
                    "detail": f"P={p} != M={m}+F={f}={m+f}",
                    "diff": p - (m + f),
                })

    # L2: Vertical sum (Total = sum of age rows)
    # Build set of intermediate subtotal keys to exclude from leaf sum.
    # These are the keys (normalized ages) of subtotal rows, so we only sum
    # leaf rows when computing Total = sum(rows).
    subtotal_keys = set()
    for (_, _, _, sub_key, _) in constraints.get("L2_subtotal", []):
        subtotal_keys.add(sub_key)
    # Include auto-detected intermediate subtotals (e.g. "Total 0-15")
    subtotal_keys.update(constraints.get("_auto_subtotal_keys", set()))

    # If the schema provided explicit components for Total, use those
    # instead of "all rows minus exclusions".  This avoids including rows
    # like "Civil & Military Station" that belong to a higher-level
    # Grand Total but not to the intermediate Total.
    total_components = constraints.get("total_components")  # list or None

    # Non-age rows to exclude from vertical sum (derived statistics, not population)
    _NON_AGE_KEYS = {"mean age", "median age", "average age"}

    for sec_name, group, col in constraints.get("L2_vertical", []):
        rmap = sec_lookup.get(sec_name, {})
        total_row = rmap.get("total")
        if not total_row:
            continue
        gt = total_row.get(group, {})
        if not isinstance(gt, dict):
            continue
        total_val = gt.get(col)
        if total_val is None:
            continue

        comp_sum = 0
        all_present = True

        if total_components:
            # Use explicit component list from subtotal hierarchy
            for comp_key in total_components:
                row = rmap.get(comp_key)
                if not row:
                    all_present = False
                    break
                gdata = row.get(group, {})
                if not isinstance(gdata, dict):
                    all_present = False
                    break
                val = gdata.get(col)
                if val is None:
                    all_present = False
                    break
                comp_sum += val
        else:
            # Fallback: sum all rows except total, subtotals, non-age
            for age_norm, row in rmap.items():
                if age_norm == "total":
                    continue
                # Skip intermediate subtotals to avoid double-counting
                if age_norm in subtotal_keys:
                    continue
                # Skip derived statistics like "Mean Age"
                if age_norm in _NON_AGE_KEYS:
                    continue
                gdata = row.get(group, {})
                if not isinstance(gdata, dict):
                    all_present = False
                    break
                val = gdata.get(col)
                if val is None:
                    all_present = False
                    break
                comp_sum += val

        if all_present:
            total_checks += 1
            diff = total_val - comp_sum
            # For proportional tables, allow rounding tolerance:
            # each leaf row can have ±0.5 rounding error from independent
            # rounding in the original hand-computed table.  Use 2*N to
            # account for imperfect rounding in 100+ year old sources.
            is_prop = constraints.get("is_proportional", False)
            n_leaf = (len(total_components) if total_components
                      else sum(1 for a in rmap if a != "total"
                               and a not in subtotal_keys
                               and a not in _NON_AGE_KEYS))
            tol = n_leaf * 2 if is_prop else 0
            if abs(diff) > tol:
                failures.append({
                    "level": "L2",
                    "section": sec_name,
                    "group": group,
                    "col": col,
                    "detail": f"Total={total_val} != sum(rows)={comp_sum}",
                    "diff": diff,
                })

    # L2_subtotal: Intermediate subtotal checks (from schema hierarchy)
    for (sec_name, group, col, sub_key, comp_keys) in constraints.get("L2_subtotal", []):
        rmap = sec_lookup.get(sec_name, {})
        sub_row = rmap.get(sub_key)
        if not sub_row:
            continue
        gdata = sub_row.get(group, {})
        if not isinstance(gdata, dict):
            continue
        sub_val = gdata.get(col)
        if sub_val is None:
            continue

        comp_sum = 0
        all_present = True
        for ck in comp_keys:
            ck_row = rmap.get(ck)
            if not ck_row:
                all_present = False
                break
            ck_data = ck_row.get(group, {})
            if not isinstance(ck_data, dict):
                all_present = False
                break
            val = ck_data.get(col)
            if val is None:
                all_present = False
                break
            comp_sum += val

        if all_present:
            total_checks += 1
            diff = sub_val - comp_sum
            is_prop = constraints.get("is_proportional", False)
            tol = len(comp_keys) if is_prop else 0
            if abs(diff) > tol:
                failures.append({
                    "level": "L2",
                    "section": sec_name,
                    "group": group,
                    "col": col,
                    "detail": f"{sub_key}={sub_val} != sum({comp_keys})={comp_sum}",
                    "diff": diff,
                })

    # Known totals check (e.g. proportional tables: Total M=1000, F=1000)
    kt = constraints.get("known_totals")
    if kt:
        for age_key, col_vals in kt.items():
            for sec_name, rmap in sec_lookup.items():
                kt_row = rmap.get(age_key)
                if not kt_row:
                    continue
                for group in groups:
                    gdata = kt_row.get(group, {})
                    if not isinstance(gdata, dict):
                        continue
                    for col, expected in col_vals.items():
                        actual = gdata.get(col)
                        if actual is not None:
                            total_checks += 1
                            if actual != expected:
                                failures.append({
                                    "level": "L2",
                                    "section": sec_name,
                                    "group": group,
                                    "col": col,
                                    "detail": f"Known total {age_key}.{col}: "
                                              f"expected={expected}, got={actual}",
                                    "diff": actual - expected,
                                })

    # L3: Cross-group
    xg = constraints.get("L3_cross_group")
    if xg:
        total_group = xg["total_group"]
        comp_groups = xg["component_groups"]
        for section in sections:
            sec_name = section.get("name", "")
            for row in section.get("rows", []):
                age = row.get("age", "")
                for col in ("persons", "males", "females"):
                    fg_data = row.get(total_group, {})
                    if not isinstance(fg_data, dict):
                        continue
                    fg_val = fg_data.get(col)
                    if fg_val is None:
                        continue
                    comp_sum = 0
                    all_present = True
                    for cg in comp_groups:
                        cg_data = row.get(cg, {})
                        if not isinstance(cg_data, dict):
                            all_present = False
                            break
                        cg_val = cg_data.get(col)
                        if cg_val is None:
                            all_present = False
                            break
                        comp_sum += cg_val
                    if all_present:
                        total_checks += 1
                        if fg_val != comp_sum:
                            failures.append({
                                "level": "L3",
                                "section": sec_name,
                                "age": age,
                                "col": col,
                                "detail": (f"{total_group}.{col}={fg_val} != "
                                           f"sum({comp_groups})={comp_sum}"),
                                "diff": fg_val - comp_sum,
                            })

    # L4: Cross-section
    xs = constraints.get("L4_cross_section")
    if xs:
        total_sec = xs["total_section"]
        comp_secs = xs["component_sections"]
        total_sec_lookup = sec_lookup.get(total_sec, {})

        comp_sec_lookups = [sec_lookup.get(cs, {}) for cs in comp_secs]

        for age_norm, total_row in total_sec_lookup.items():
            for group in groups:
                gt = total_row.get(group, {})
                if not isinstance(gt, dict):
                    continue
                for col in ("persons", "males", "females"):
                    total_val = gt.get(col)
                    if total_val is None:
                        continue
                    comp_sum = 0
                    all_present = True
                    for csl in comp_sec_lookups:
                        comp_row = csl.get(age_norm)
                        if not comp_row:
                            all_present = False
                            break
                        cg_data = comp_row.get(group, {})
                        if not isinstance(cg_data, dict):
                            all_present = False
                            break
                        cg_val = cg_data.get(col)
                        if cg_val is None:
                            all_present = False
                            break
                        comp_sum += cg_val
                    if all_present:
                        total_checks += 1
                        if total_val != comp_sum:
                            failures.append({
                                "level": "L4",
                                "age": age_norm,
                                "group": group,
                                "col": col,
                                "detail": (f"{total_sec}={total_val} != "
                                           f"sum({comp_secs})={comp_sum}"),
                                "diff": total_val - comp_sum,
                            })

    # L5: Non-negativity
    for sec_name, age, group, col in constraints.get("L5_non_negative", []):
        rmap = sec_lookup.get(sec_name, {})
        age_norm = normalize_age(age)
        row = rmap.get(age_norm)
        if not row:
            continue
        gdata = row.get(group, {})
        if not isinstance(gdata, dict):
            continue
        val = gdata.get(col)
        if val is not None:
            total_checks += 1
            if val < 0:
                failures.append({
                    "level": "L5",
                    "section": sec_name,
                    "age": age,
                    "group": group,
                    "col": col,
                    "detail": f"Negative value: {val}",
                })

    return {
        "total_checks": total_checks,
        "passed": total_checks - len(failures),
        "failed": len(failures),
        "all_passed": len(failures) == 0,
        "failures": failures,
        "summary": {
            "L1_row": sum(1 for c in constraints.get("L1_row", [])
                          if not any(f["level"] == "L1"
                                     and f.get("age") == c[1]
                                     and f.get("group") == c[2]
                                     for f in failures)),
            "L2_vertical": len(constraints.get("L2_vertical", [])),
            "L3_cross_group": "active" if xg else "not detected",
            "L4_cross_section": "active" if xs else "not detected",
        },
    }


# ---------------------------------------------------------------------------
# attempt_repair() — minimal single-digit correction for L1 failures
# ---------------------------------------------------------------------------

def attempt_repair(parsed, failures):
    """Attempt minimal repairs for L1 (P != M+F) failures.

    For each failure, find the smallest single-value correction
    that restores P = M + F.

    Returns:
        (repaired_parsed, repair_log)
    """
    import copy
    repaired = copy.deepcopy(parsed)
    repair_log = []

    # Build lookup
    sec_lookup = {}
    for section in repaired.get("sections", []):
        sec_name = section.get("name", "")
        rmap = {}
        for row in section.get("rows", []):
            age_norm = normalize_age(row.get("age", ""))
            rmap[age_norm] = row
        sec_lookup[sec_name] = rmap

    for fail in failures:
        if fail["level"] != "L1":
            continue
        sec_name = fail.get("section", "")
        age = fail.get("age", "")
        group = fail.get("group", "")
        age_norm = normalize_age(age)

        row = sec_lookup.get(sec_name, {}).get(age_norm)
        if not row:
            continue
        gdata = row.get(group, {})
        if not isinstance(gdata, dict):
            continue

        p = gdata.get("persons")
        m = gdata.get("males")
        f = gdata.get("females")
        if p is None or m is None or f is None:
            continue

        expected_p = m + f
        diff_p = abs(p - expected_p)

        candidate_m = p - f
        diff_m = abs(m - candidate_m)

        candidate_f = p - m
        diff_f = abs(f - candidate_f)

        candidates = []
        if diff_p > 0:
            candidates.append((diff_p, "persons", expected_p, p))
        if candidate_m >= 0 and diff_m > 0:
            candidates.append((diff_m, "males", candidate_m, m))
        if candidate_f >= 0 and diff_f > 0:
            candidates.append((diff_f, "females", candidate_f, f))

        candidates.sort(key=lambda c: c[0])
        if candidates:
            _, fix_col, new_val, old_val = candidates[0]
            gdata[fix_col] = new_val
            repair_log.append({
                "section": sec_name,
                "age": age,
                "group": group,
                "col": fix_col,
                "old": old_val,
                "new": new_val,
                "diff": abs(new_val - old_val),
            })

    return repaired, repair_log


# ---------------------------------------------------------------------------
# identify_suspicious_cells() — blame scoring from constraint failures
# ---------------------------------------------------------------------------

def identify_suspicious_cells(parsed, failures):
    """From constraint failures, find root-cause cells via blame scoring.

    For each failure, enumerate all cells that *could* be wrong:
    - L1 (P!=M+F): the P, M, F cells for that (section, age, group)
    - L2 (Total!=sum): Total cell + all component age cells for that (section, group, col)
    - L3 (cross-group): total-group cell + all component-group cells for that (section, age, col)
    - L4 (cross-section): total-section cell + all component-section cells for that (age, group, col)

    Returns list of (blame_score, cell_key) sorted descending by blame,
    where cell_key = (section, age, group, col).
    """
    sections = parsed.get("sections", [])
    groups = parsed.get("metadata", {}).get("column_groups", [])

    # Build section lookup for L2 age enumeration
    sec_lookup = {}
    for section in sections:
        sec_name = section.get("name", "")
        rmap = {}
        for row in section.get("rows", []):
            rmap[normalize_age(row.get("age", ""))] = row
        sec_lookup[sec_name] = rmap

    blame = Counter()

    for f in failures:
        level = f["level"]

        if level == "L1":
            sec = f.get("section", "")
            age = f.get("age", "")
            group = f.get("group", "")
            for col in ("persons", "males", "females"):
                blame[(sec, age, group, col)] += 1

        elif level == "L2":
            sec = f.get("section", "")
            group = f.get("group", "")
            col = f.get("col", "")
            # Total cell
            blame[(sec, "Total", group, col)] += 1
            # All age rows
            rmap = sec_lookup.get(sec, {})
            for age_norm in rmap:
                if age_norm != "total":
                    blame[(sec, age_norm, group, col)] += 1

        elif level == "L3":
            sec = f.get("section", "")
            age = f.get("age", "")
            col = f.get("col", "")
            # All groups for this row
            for group in groups:
                blame[(sec, age, group, col)] += 1

        elif level == "L4":
            age = f.get("age", "")
            group = f.get("group", "")
            col = f.get("col", "")
            # All sections for this cell
            for section in sections:
                sec_name = section.get("name", "")
                blame[(sec_name, age, group, col)] += 1

    if not blame:
        return []

    # Sort by blame score descending, return top cells
    ranked = sorted(blame.items(), key=lambda x: -x[1])
    return ranked


# ---------------------------------------------------------------------------
# targeted_recheck() — one extra API call for suspicious cells
# ---------------------------------------------------------------------------

RECHECK_PROMPT = """I extracted data from this census table but some values
failed consistency checks. Please re-read ONLY the following cells very
carefully, examining each digit one by one.

Cells to verify:
{cells_list}

For each cell, return your reading as JSON array:
[
  {{"section": "...", "age": "...", "group": "...", "persons": N, "males": N, "females": N}}
]

CRITICAL: Look at each digit individually. Common confusions: 3↔8, 5↔6, 0↔9.
Return ONLY valid JSON array."""


def targeted_recheck(image_path, parsed, suspicious_cells):
    """One extra API call with focused verification prompt for suspicious cells.

    Args:
        image_path: Path to the census image.
        parsed: Current parsed data dict.
        suspicious_cells: List of (blame_score, (section, age, group, col)) tuples.

    Returns:
        Updated parsed dict (modified in place only if recheck improves constraints).
    """
    import copy

    # Deduplicate to (section, age, group) level — we re-read entire PMF triples
    seen = set()
    cells_to_check = []
    for (sec, age, group, _col), _score in suspicious_cells:
        key = (sec, age, group)
        if key not in seen:
            seen.add(key)
            cells_to_check.append(key)
        if len(cells_to_check) >= 10:  # cap to keep prompt focused
            break

    # Format the cells list for the prompt
    cells_desc = []
    for sec, age, group in cells_to_check:
        cells_desc.append(f"  - Section: \"{sec}\", Age: \"{age}\", Group: \"{group}\" → read Persons, Males, Females")

    prompt = RECHECK_PROMPT.format(cells_list="\n".join(cells_desc))

    # Make the API call
    b64 = encode_image(str(image_path))
    try:
        raw = call_gemini(b64, prompt)
    except Exception as e:
        print(f"    Recheck API call failed: {e}")
        return parsed

    # Parse the response
    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)

    try:
        recheck_data = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("[")
        end = text.rfind("]") + 1
        if start >= 0 and end > start:
            try:
                recheck_data = json.loads(text[start:end])
            except json.JSONDecodeError:
                print("    Could not parse recheck response")
                return parsed
        else:
            print("    Could not parse recheck response")
            return parsed

    if not isinstance(recheck_data, list):
        recheck_data = [recheck_data]

    # Build lookup for current data
    sec_lookup = {}
    for section in parsed.get("sections", []):
        sec_name = section.get("name", "")
        rmap = {}
        for row in section.get("rows", []):
            rmap[normalize_age(row.get("age", ""))] = row
        sec_lookup[sec_name] = rmap

    # Try each recheck reading: accept only if it reduces failures
    constraints_before = derive_constraints(parsed)
    report_before = verify_all_constraints(parsed, constraints_before)
    failures_before = report_before["failed"]

    for item in recheck_data:
        if not isinstance(item, dict):
            continue
        sec = item.get("section", "")
        age = item.get("age", "")
        group = item.get("group", "")
        age_norm = normalize_age(age)

        row = sec_lookup.get(sec, {}).get(age_norm)
        if not row:
            continue
        gdata = row.get(group)
        if not isinstance(gdata, dict):
            continue

        # Save originals
        orig = {col: gdata.get(col) for col in ("persons", "males", "females")}

        # Check if recheck differs
        new_vals = {}
        any_diff = False
        for col in ("persons", "males", "females"):
            new_val = _normalize_value(item.get(col))
            new_vals[col] = new_val
            if new_val is not None and new_val != orig[col]:
                any_diff = True

        if not any_diff:
            continue

        # Tentatively apply
        for col in ("persons", "males", "females"):
            if new_vals[col] is not None:
                gdata[col] = new_vals[col]

        # Re-verify
        constraints_new = derive_constraints(parsed)
        report_new = verify_all_constraints(parsed, constraints_new)

        if report_new["failed"] < failures_before:
            # Accept the change
            print(f"    ACCEPTED recheck: {sec}/{age}/{group} "
                  f"{orig} -> {new_vals} (failures {failures_before} -> {report_new['failed']})")
            failures_before = report_new["failed"]
        else:
            # Revert
            for col in ("persons", "males", "females"):
                gdata[col] = orig[col]

    return parsed


# ---------------------------------------------------------------------------
# Constraint-driven repair cascade (Step 4c)
# ---------------------------------------------------------------------------

# Known OCR digit confusion pairs: (wrong_digit, correct_digit)
_DIGIT_CONFUSIONS = [
    (3, 8), (8, 3),  # 3↔8 (diff=5)
    (5, 6), (6, 5),  # 5↔6 (diff=1)
    (0, 9), (9, 0),  # 0↔9 (diff=9)
    (0, 8), (8, 0),  # 0↔8 (diff=8)
    (1, 7), (7, 1),  # 1↔7 (diff=6)
    (6, 8), (8, 6),  # 6↔8 (diff=2)
    (3, 5), (5, 3),  # 3↔5 (diff=2)
    (0, 6), (6, 0),  # 0↔6 (diff=6)
    (6, 9), (9, 6),  # 6↔9 (diff=3)
    (4, 9), (9, 4),  # 4↔9 (diff=5)
    (3, 9), (9, 3),  # 3↔9 (diff=6)
    (1, 4), (4, 1),  # 1↔4 (diff=3)
    (2, 7), (7, 2),  # 2↔7 (diff=5)
    (5, 9), (9, 5),  # 5↔9 (diff=4)
    (0, 5), (5, 0),  # 0↔5 (diff=5)
    (3, 6), (6, 3),  # 3↔6 (diff=3, curved digits)
    (4, 5), (5, 4),  # 4↔5 (diff=1)
    (1, 0), (0, 1),  # 1↔0 (diff=1, thin stroke)
    (5, 8), (8, 5),  # 5↔8 (diff=3, curved digits)
    (2, 5), (5, 2),  # 2↔5 (diff=3, curved bottom)
    (0, 3), (3, 0),  # 0↔3 (diff=3, closed vs open curve)
    (4, 7), (7, 4),  # 4↔7 (diff=3, angular strokes)
    (8, 9), (9, 8),  # 8↔9 (diff=1, closed loops)
    (2, 3), (3, 2),  # 2↔3 (diff=1, curved digits)
    (6, 7), (7, 6),  # 6↔7 (diff=1)
]


def _find_digit_fix(value, target_change):
    """Find a single-digit replacement that changes value by exactly target_change.

    Checks all OCR digit confusion patterns at every digit position.

    Returns (new_value, position, old_digit, new_digit) or None.
    """
    if value is None or target_change == 0:
        return None

    digits = list(str(abs(value)))
    n_digits = len(digits)

    for pos in range(n_digits):
        place_value = 10 ** (n_digits - 1 - pos)
        current_digit = int(digits[pos])

        for wrong_d, correct_d in _DIGIT_CONFUSIONS:
            if current_digit != wrong_d:
                continue
            change = (correct_d - wrong_d) * place_value
            if change == target_change:
                new_digits = digits.copy()
                new_digits[pos] = str(correct_d)
                new_value = int("".join(new_digits))
                if value < 0:
                    new_value = -new_value
                return (new_value, n_digits - 1 - pos, wrong_d, correct_d)

    return None


def _find_two_digit_fix(value, target_change):
    """Find a TWO-digit replacement that changes value by exactly target_change.

    For when single-digit fix fails: tries all pairs of OCR confusions.
    Returns (new_value, [(pos1, old1, new1), (pos2, old2, new2)]) or None.
    """
    if value is None or target_change == 0:
        return None

    digits = list(str(abs(value)))
    n_digits = len(digits)

    # Precompute all single-digit changes at each position
    changes = []  # (pos, change, wrong_d, correct_d)
    for pos in range(n_digits):
        place_value = 10 ** (n_digits - 1 - pos)
        current_digit = int(digits[pos])
        for wrong_d, correct_d in _DIGIT_CONFUSIONS:
            if current_digit != wrong_d:
                continue
            change = (correct_d - wrong_d) * place_value
            changes.append((pos, change, wrong_d, correct_d))

    # Try all pairs of changes at DIFFERENT positions
    for i in range(len(changes)):
        for j in range(i + 1, len(changes)):
            p1, c1, w1, r1 = changes[i]
            p2, c2, w2, r2 = changes[j]
            if p1 == p2:
                continue  # same position
            if c1 + c2 == target_change:
                new_digits = digits.copy()
                new_digits[p1] = str(r1)
                new_digits[p2] = str(r2)
                new_value = int("".join(new_digits))
                if value < 0:
                    new_value = -new_value
                return (new_value, [
                    (n_digits - 1 - p1, w1, r1),
                    (n_digits - 1 - p2, w2, r2),
                ])

    return None


def _detect_and_fix_mf_swaps(parsed, constraints, persons_independent):
    """Phase A: Detect and fix M/F swaps (0 API calls).

    When the OCR model reads males from the females column and vice versa,
    L2 vertical sums show diff_males = -diff_females. L3 cross-group checks
    also reveal swap patterns where diff_males = -diff_females for a specific
    age row. This phase detects such patterns and swaps values back.

    Returns (parsed, repair_log).
    """
    if persons_independent:
        return parsed, []

    repair_log = []
    from collections import defaultdict

    for iteration in range(50):
        report = verify_all_constraints(parsed, constraints)
        if report["all_passed"]:
            break

        total_before = report["failed"]
        failures = report["failures"]
        found_any = False

        # Identify L2 swap patterns: (section, group) where diff_m + diff_f == 0
        l2_by_sg = defaultdict(dict)
        for f in failures:
            if f["level"] == "L2":
                l2_by_sg[(f["section"], f["group"])][f.get("col", "")] = f["diff"]

        l2_swap_sgs = set()
        for (sec, grp), diffs in l2_by_sg.items():
            m = diffs.get("males", 0)
            f_diff = diffs.get("females", 0)
            if m + f_diff == 0 and m != 0:
                l2_swap_sgs.add((sec, grp))

        # Identify L3 swap patterns: (section, age) where diff_m + diff_f == 0
        l3_by_sa = defaultdict(dict)
        for f in failures:
            if f["level"] == "L3":
                l3_by_sa[(f["section"], f.get("age", ""))][f.get("col", "")] = f["diff"]

        l3_swap_rows = set()
        for (sec, age), diffs in l3_by_sa.items():
            m = diffs.get("males", 0)
            f_diff = diffs.get("females", 0)
            if m + f_diff == 0 and m != 0:
                l3_swap_rows.add((sec, age))

        if not l2_swap_sgs and not l3_swap_rows:
            break

        groups = constraints.get("column_groups", [])

        # --- Multi-group swap: swap M/F across ALL groups at a row ---
        # When L3 identifies a row-level MF-swap, single-group swaps often
        # fail because fixing L2 breaks L3 (or vice versa). Swapping ALL
        # groups simultaneously at the affected row preserves cross-group
        # relationships.
        for (sec, age) in l3_swap_rows:
            if found_any:
                break
            age_norm = normalize_age(age)
            for section in parsed["sections"]:
                if found_any or section["name"] != sec:
                    continue
                for row in section["rows"]:
                    if normalize_age(row.get("age", "")) != age_norm:
                        continue
                    # Collect all swappable groups at this row
                    swap_candidates = []
                    for grp in groups:
                        gdata = row.get(grp)
                        if not isinstance(gdata, dict):
                            continue
                        m_val = gdata.get("males")
                        f_val = gdata.get("females")
                        if m_val is not None and f_val is not None and m_val != f_val:
                            swap_candidates.append((grp, gdata, m_val, f_val))
                    if len(swap_candidates) < 2:
                        continue

                    # Try swapping ALL groups at this row
                    for _, gdata, m_val, f_val in swap_candidates:
                        gdata["males"] = f_val
                        gdata["females"] = m_val
                    new_report = verify_all_constraints(parsed, constraints)
                    if new_report["failed"] < total_before:
                        for grp, gdata, m_val, f_val in swap_candidates:
                            repair_log.append({
                                "phase": "A", "action": "mf_swap",
                                "section": sec, "age": age_norm,
                                "group": grp,
                                "old_m": m_val, "old_f": f_val,
                                "new_m": f_val, "new_f": m_val,
                            })
                        found_any = True
                        break
                    else:
                        for _, gdata, m_val, f_val in swap_candidates:
                            gdata["males"] = m_val
                            gdata["females"] = f_val

                    # If all-group swap didn't work, try subsets:
                    # swap only groups that show L2 MF-swap pattern
                    l2_grps = {grp for (s, grp) in l2_swap_sgs if s == sec}
                    subset = [(g, gd, m, f) for g, gd, m, f in swap_candidates
                              if g in l2_grps]
                    if 1 < len(subset) < len(swap_candidates):
                        for _, gdata, m_val, f_val in subset:
                            gdata["males"] = f_val
                            gdata["females"] = m_val
                        new_report = verify_all_constraints(parsed, constraints)
                        if new_report["failed"] < total_before:
                            for grp, gdata, m_val, f_val in subset:
                                repair_log.append({
                                    "phase": "A", "action": "mf_swap",
                                    "section": sec, "age": age_norm,
                                    "group": grp,
                                    "old_m": m_val, "old_f": f_val,
                                    "new_m": f_val, "new_f": m_val,
                                })
                            found_any = True
                            break
                        else:
                            for _, gdata, m_val, f_val in subset:
                                gdata["males"] = m_val
                                gdata["females"] = f_val

        # --- Single-group swap (original approach) ---
        if not found_any:
            candidates = []

            # L3-guided: specific (section, age) with each group
            for (sec, age) in l3_swap_rows:
                age_norm = normalize_age(age)
                for grp in groups:
                    candidates.append((sec, age_norm, grp))

            # L2-guided: try each non-Total row in the affected (section, group)
            for (sec, grp) in l2_swap_sgs:
                for section in parsed["sections"]:
                    if section["name"] != sec:
                        continue
                    for row in section["rows"]:
                        age_norm = normalize_age(row.get("age", ""))
                        if age_norm == "total":
                            continue
                        candidates.append((sec, age_norm, grp))

            # Deduplicate while preserving order
            seen = set()
            unique = []
            for c in candidates:
                if c not in seen:
                    seen.add(c)
                    unique.append(c)

            for (sec, age_norm, grp) in unique:
                if found_any:
                    break
                for section in parsed["sections"]:
                    if found_any or section["name"] != sec:
                        continue
                    for row in section["rows"]:
                        if normalize_age(row.get("age", "")) != age_norm:
                            continue
                        gdata = row.get(grp)
                        if not isinstance(gdata, dict):
                            continue
                        m_val = gdata.get("males")
                        f_val = gdata.get("females")
                        if m_val is None or f_val is None or m_val == f_val:
                            continue

                        gdata["males"] = f_val
                        gdata["females"] = m_val

                        new_report = verify_all_constraints(parsed, constraints)
                        if new_report["failed"] < total_before:
                            repair_log.append({
                                "phase": "A", "action": "mf_swap",
                                "section": sec, "age": age_norm,
                                "group": grp,
                                "old_m": m_val, "old_f": f_val,
                                "new_m": f_val, "new_f": m_val,
                            })
                            found_any = True
                            break
                        else:
                            gdata["males"] = m_val
                            gdata["females"] = f_val

        if not found_any:
            break

    return parsed, repair_log


def _deductive_digit_fix(parsed, constraints, persons_independent):
    """Phase B: Fix single-digit OCR confusions deductively (0 API calls).

    Handles three fix strategies:
    1. Single-cell: one digit confusion in one cell fixes the constraint.
    2. Paired M/F: when L2 diff_males = -diff_females, fix both males and
       females digits simultaneously (prevents L1 breakage).
    3. L3-targeted: use cross-group failure to identify the exact (section, age)
       and try each group's cells.

    Returns (parsed, repair_log).
    """
    from collections import defaultdict

    repair_log = []

    for iteration in range(50):
        report = verify_all_constraints(parsed, constraints)
        if report["all_passed"]:
            break

        total_before = report["failed"]
        failures = report["failures"]
        found_any = False

        # --- Strategy 1: Single-cell L1 fix ---
        for f in failures:
            if found_any:
                break
            if f["level"] != "L1" or persons_independent:
                continue
            diff = f["diff"]  # P - (M+F)
            sec, age, grp = f["section"], f["age"], f["group"]

            for section in parsed["sections"]:
                if section["name"] != sec:
                    continue
                for row in section["rows"]:
                    if normalize_age(row.get("age", "")) != normalize_age(age):
                        continue
                    gdata = row.get(grp)
                    if not isinstance(gdata, dict):
                        continue
                    for col, tc in [("persons", -diff),
                                    ("males", diff),
                                    ("females", diff)]:
                        val = gdata.get(col)
                        if val is None:
                            continue
                        fix = _find_digit_fix(val, tc)
                        if not fix:
                            continue
                        old_val = gdata[col]
                        gdata[col] = fix[0]
                        new_report = verify_all_constraints(parsed, constraints)
                        if new_report["failed"] < total_before:
                            repair_log.append({
                                "phase": "B", "action": "digit_fix",
                                "section": sec, "age": age,
                                "group": grp, "col": col,
                                "old": old_val, "new": fix[0],
                                "digit_pos": fix[1],
                                "old_digit": fix[2], "new_digit": fix[3],
                            })
                            found_any = True
                            break
                        else:
                            gdata[col] = old_val
                    break
                break
            if found_any:
                continue

        # --- Strategy 2: Single-cell L2 fix (with P propagation) ---
        for f in failures:
            if found_any:
                break
            if f["level"] != "L2":
                continue
            diff = f["diff"]
            sec, grp, col = f["section"], f["group"], f["col"]

            for section in parsed["sections"]:
                if section["name"] != sec:
                    continue
                for row in section["rows"]:
                    if found_any:
                        break
                    age_norm = normalize_age(row.get("age", ""))
                    gdata = row.get(grp)
                    if not isinstance(gdata, dict):
                        continue
                    val = gdata.get(col)
                    if val is None:
                        continue
                    tc = -diff if age_norm == "total" else diff
                    fix = _find_digit_fix(val, tc)
                    if not fix:
                        continue
                    old_val = gdata[col]
                    old_p = gdata.get("persons")
                    gdata[col] = fix[0]

                    # Try with P = M+F propagation first (for M/F fixes)
                    p_propagated = False
                    if (col in ("males", "females")
                            and not persons_independent
                            and old_p is not None):
                        new_p = (gdata.get("males", 0)
                                 + gdata.get("females", 0))
                        if new_p != old_p:
                            gdata["persons"] = new_p
                            p_propagated = True

                    new_report = verify_all_constraints(parsed, constraints)
                    if new_report["failed"] < total_before:
                        repair_log.append({
                            "phase": "B", "action": "digit_fix",
                            "section": sec, "age": row.get("age", ""),
                            "group": grp, "col": col,
                            "old": old_val, "new": fix[0],
                            "digit_pos": fix[1],
                            "old_digit": fix[2], "new_digit": fix[3],
                        })
                        if p_propagated:
                            repair_log.append({
                                "phase": "B", "action": "p_propagate",
                                "section": sec, "age": row.get("age", ""),
                                "group": grp, "col": "persons",
                                "old": old_p, "new": gdata["persons"],
                            })
                        found_any = True
                    else:
                        # If P propagation didn't help, try without
                        if p_propagated:
                            gdata["persons"] = old_p
                            new_report = verify_all_constraints(
                                parsed, constraints)
                            if new_report["failed"] < total_before:
                                repair_log.append({
                                    "phase": "B", "action": "digit_fix",
                                    "section": sec,
                                    "age": row.get("age", ""),
                                    "group": grp, "col": col,
                                    "old": old_val, "new": fix[0],
                                    "digit_pos": fix[1],
                                    "old_digit": fix[2],
                                    "new_digit": fix[3],
                                })
                                found_any = True
                            else:
                                gdata[col] = old_val
                        else:
                            gdata[col] = old_val
                break
            if found_any:
                continue

        # --- Strategy 3: Paired M/F L2 fix (same or different rows) ---
        # When diff_males + diff_females == 0, fix both columns.
        # Same-row: fix M and F on same row (no P breakage).
        # Multi-row: fix M on row A, F on row B, propagate P on both
        # (net P change cancels out since m_diff + f_diff == 0).
        if not found_any:
            l2_by_sg = defaultdict(dict)
            for f in failures:
                if f["level"] == "L2":
                    l2_by_sg[(f["section"], f["group"])][f["col"]] = f["diff"]

            for (sec, grp), diffs in l2_by_sg.items():
                if found_any:
                    break
                m_diff = diffs.get("males")
                f_diff = diffs.get("females")
                if m_diff is None or f_diff is None:
                    continue
                if m_diff + f_diff != 0 or m_diff == 0:
                    continue

                # Collect candidate fixes for males and females
                m_cands = []  # (gdata, fix, row)
                f_cands = []
                for section in parsed["sections"]:
                    if section["name"] != sec:
                        continue
                    for row in section["rows"]:
                        age_norm = normalize_age(row.get("age", ""))
                        gdata = row.get(grp)
                        if not isinstance(gdata, dict):
                            continue
                        m_val = gdata.get("males")
                        f_val = gdata.get("females")

                        tc_m = -m_diff if age_norm == "total" else m_diff
                        tc_f = -f_diff if age_norm == "total" else f_diff

                        if m_val is not None:
                            mf = _find_digit_fix(m_val, tc_m)
                            if mf:
                                m_cands.append((gdata, mf, row))
                        if f_val is not None:
                            ff = _find_digit_fix(f_val, tc_f)
                            if ff:
                                f_cands.append((gdata, ff, row))

                # Try all (m_cand, f_cand) pairs
                for mg, mf, mrow in m_cands:
                    if found_any:
                        break
                    for fg, ff, frow in f_cands:
                        if found_any:
                            break
                        old_m = mg["males"]
                        old_f = fg["females"]
                        old_pm = mg.get("persons")
                        old_pf = fg.get("persons")
                        same_row = (mg is fg)

                        mg["males"] = mf[0]
                        fg["females"] = ff[0]

                        # Propagate P = M + F on affected rows
                        if not persons_independent:
                            mg["persons"] = (mg.get("males", 0)
                                             + mg.get("females", 0))
                            if not same_row:
                                fg["persons"] = (fg.get("males", 0)
                                                 + fg.get("females", 0))

                        new_report = verify_all_constraints(
                            parsed, constraints)
                        if new_report["failed"] < total_before:
                            for gd, fx, rw, cn in [
                                (mg, mf, mrow, "males"),
                                (fg, ff, frow, "females"),
                            ]:
                                repair_log.append({
                                    "phase": "B", "action": "digit_fix",
                                    "section": sec,
                                    "age": rw.get("age", ""),
                                    "group": grp, "col": cn,
                                    "old": old_m if cn == "males" else old_f,
                                    "new": fx[0],
                                    "digit_pos": fx[1],
                                    "old_digit": fx[2],
                                    "new_digit": fx[3],
                                })
                            if not persons_independent:
                                if old_pm != mg.get("persons"):
                                    repair_log.append({
                                        "phase": "B",
                                        "action": "p_propagate",
                                        "section": sec,
                                        "age": mrow.get("age", ""),
                                        "group": grp, "col": "persons",
                                        "old": old_pm,
                                        "new": mg["persons"],
                                    })
                                if (not same_row
                                        and old_pf != fg.get("persons")):
                                    repair_log.append({
                                        "phase": "B",
                                        "action": "p_propagate",
                                        "section": sec,
                                        "age": frow.get("age", ""),
                                        "group": grp, "col": "persons",
                                        "old": old_pf,
                                        "new": fg["persons"],
                                    })
                            found_any = True
                        else:
                            mg["males"] = old_m
                            fg["females"] = old_f
                            if old_pm is not None:
                                mg["persons"] = old_pm
                            if not same_row and old_pf is not None:
                                fg["persons"] = old_pf

        # --- Strategy 4: L3-targeted paired fix ---
        # L3 cross-group failures pinpoint exact (section, age) rows.
        # When diff_males = -diff_females, the error is in one component group.
        if not found_any:
            l3_by_sa = defaultdict(dict)
            for f in failures:
                if f["level"] == "L3":
                    l3_by_sa[(f["section"], f.get("age", ""))][f.get("col", "")] = f["diff"]

            groups = constraints.get("column_groups", [])
            xg = constraints.get("L3_cross_group")
            component_groups = xg["component_groups"] if xg else groups[1:]

            for (sec, age), diffs in l3_by_sa.items():
                if found_any:
                    break
                m_diff = diffs.get("males", 0)
                f_diff = diffs.get("females", 0)
                if m_diff + f_diff != 0 or m_diff == 0:
                    continue
                age_norm = normalize_age(age)

                # Try each component group — find which one has the confused digits
                for section in parsed["sections"]:
                    if found_any or section["name"] != sec:
                        continue
                    for row in section["rows"]:
                        if found_any:
                            break
                        if normalize_age(row.get("age", "")) != age_norm:
                            continue

                        for grp in component_groups:
                            if found_any:
                                break
                            gdata = row.get(grp)
                            if not isinstance(gdata, dict):
                                continue
                            m_val = gdata.get("males")
                            f_val = gdata.get("females")
                            if m_val is None or f_val is None:
                                continue

                            # The L3 diff tells us the cross-group error.
                            # The component group cell needs to change by
                            # -diff to fix the cross-group sum.
                            m_fix = _find_digit_fix(m_val, -m_diff)
                            f_fix = _find_digit_fix(f_val, -f_diff)

                            if m_fix and f_fix:
                                old_m, old_f = m_val, f_val
                                gdata["males"] = m_fix[0]
                                gdata["females"] = f_fix[0]
                                new_report = verify_all_constraints(
                                    parsed, constraints)
                                if new_report["failed"] < total_before:
                                    for col_name, fix, old_v in [
                                        ("males", m_fix, old_m),
                                        ("females", f_fix, old_f)
                                    ]:
                                        repair_log.append({
                                            "phase": "B",
                                            "action": "digit_fix",
                                            "section": sec,
                                            "age": row.get("age", ""),
                                            "group": grp, "col": col_name,
                                            "old": old_v, "new": fix[0],
                                            "digit_pos": fix[1],
                                            "old_digit": fix[2],
                                            "new_digit": fix[3],
                                        })
                                    found_any = True
                                else:
                                    gdata["males"] = old_m
                                    gdata["females"] = old_f

        # --- Strategy 5: Component digit fix + L3 propagation ---
        # When L3 cross-group exists (e.g. POPULATION = UNMARRIED + MARRIED +
        # WIDOWED) and both the total group and a component group share the
        # same L2 diff pattern (m+f=0), the root cause is a digit error in the
        # COMPONENT group. Fix the component, then propagate via L3: set
        # total_group(age) = sum(component_groups(age)) for affected rows.
        if not found_any:
            xg = constraints.get("L3_cross_group")
            if xg:
                total_grp = xg["total_group"]
                comp_grps = xg["component_groups"]

                l2_sg2 = defaultdict(dict)
                for f in failures:
                    if f["level"] == "L2":
                        key = (f["section"], f["group"])
                        l2_sg2[key][f["col"]] = f["diff"]

                # Find component groups with same diff as total group
                for sec_grp_total, diffs_total in l2_sg2.items():
                    if found_any:
                        break
                    sec, grp_t = sec_grp_total
                    if grp_t != total_grp:
                        continue
                    md_t = diffs_total.get("males")
                    fd_t = diffs_total.get("females")
                    if (md_t is None or fd_t is None
                            or md_t + fd_t != 0 or md_t == 0):
                        continue

                    # Which component groups share this diff?
                    matching_comps = []
                    for cg in comp_grps:
                        cd = l2_sg2.get((sec, cg), {})
                        if (cd.get("males") == md_t
                                and cd.get("females") == fd_t):
                            matching_comps.append(cg)

                    if not matching_comps:
                        continue

                    # Try fixing each matching component group
                    for comp in matching_comps:
                        if found_any:
                            break
                        mc, fc = [], []
                        for section in parsed["sections"]:
                            if section["name"] != sec:
                                continue
                            for row in section["rows"]:
                                an = normalize_age(row.get("age", ""))
                                gd = row.get(comp)
                                if not isinstance(gd, dict):
                                    continue
                                tc_m = (-md_t if an == "total"
                                        else md_t)
                                tc_f = (-fd_t if an == "total"
                                        else fd_t)
                                mv = gd.get("males")
                                if mv is not None:
                                    mf = _find_digit_fix(mv, tc_m)
                                    if mf:
                                        mc.append((gd, mf, row))
                                fv = gd.get("females")
                                if fv is not None:
                                    ff = _find_digit_fix(fv, tc_f)
                                    if ff:
                                        fc.append((gd, ff, row))

                        # Try each (m_cand, f_cand) pair for this
                        # component, then propagate via L3
                        import itertools
                        for mg_t, mf_t, mrow in mc[:5]:
                            if found_any:
                                break
                            for fg_t, ff_t, frow in fc[:5]:
                                if found_any:
                                    break
                                # Save state for component
                                old_cm = mg_t["males"]
                                old_cf = fg_t["females"]
                                old_cp = mg_t.get("persons")
                                old_cfp = fg_t.get("persons")
                                sr = mg_t is fg_t

                                # Apply component digit fix
                                mg_t["males"] = mf_t[0]
                                fg_t["females"] = ff_t[0]
                                if not persons_independent:
                                    mg_t["persons"] = (
                                        mg_t.get("males", 0)
                                        + mg_t.get("females", 0))
                                    if not sr:
                                        fg_t["persons"] = (
                                            fg_t.get("males", 0)
                                            + fg_t.get("females", 0))

                                # Propagate via L3: update total group
                                # at affected age rows
                                l3_saved = []
                                mrow_age = normalize_age(
                                    mrow.get("age", ""))
                                frow_age = normalize_age(
                                    frow.get("age", ""))
                                affected = {mrow_age}
                                if not sr:
                                    affected.add(frow_age)

                                for section in parsed["sections"]:
                                    if section["name"] != sec:
                                        continue
                                    for row in section["rows"]:
                                        an = normalize_age(
                                            row.get("age", ""))
                                        if an not in affected:
                                            continue
                                        tgd = row.get(total_grp)
                                        if not isinstance(tgd, dict):
                                            continue
                                        old_vals = {
                                            c: tgd.get(c)
                                            for c in ("persons",
                                                      "males",
                                                      "females")
                                        }
                                        # Recompute from components
                                        for col in ("males", "females"):
                                            s = 0
                                            for cg in comp_grps:
                                                cgd = row.get(cg, {})
                                                s += (cgd.get(col, 0)
                                                      if isinstance(
                                                          cgd, dict)
                                                      else 0)
                                            tgd[col] = s
                                        if not persons_independent:
                                            tgd["persons"] = (
                                                tgd.get("males", 0)
                                                + tgd.get("females", 0)
                                            )
                                        l3_saved.append(
                                            (tgd, old_vals, row))

                                nr = verify_all_constraints(
                                    parsed, constraints)
                                if nr["failed"] < total_before:
                                    # Log component digit fixes
                                    for cn, fx, ov in [
                                        ("males", mf_t, old_cm),
                                        ("females", ff_t, old_cf),
                                    ]:
                                        repair_log.append({
                                            "phase": "B",
                                            "action": "digit_fix",
                                            "section": sec,
                                            "age": (
                                                mrow if cn == "males"
                                                else frow
                                            ).get("age", ""),
                                            "group": comp,
                                            "col": cn,
                                            "old": ov,
                                            "new": fx[0],
                                            "digit_pos": fx[1],
                                            "old_digit": fx[2],
                                            "new_digit": fx[3],
                                        })
                                    # Log L3 propagation
                                    for tgd, ov, row in l3_saved:
                                        for col in ("males",
                                                    "females",
                                                    "persons"):
                                            if ov[col] != tgd.get(col):
                                                repair_log.append({
                                                    "phase": "B",
                                                    "action":
                                                        "l3_propagate",
                                                    "section": sec,
                                                    "age": row.get(
                                                        "age", ""),
                                                    "group": total_grp,
                                                    "col": col,
                                                    "old": ov[col],
                                                    "new": tgd[col],
                                                })
                                    found_any = True
                                else:
                                    # Revert component
                                    mg_t["males"] = old_cm
                                    fg_t["females"] = old_cf
                                    if old_cp is not None:
                                        mg_t["persons"] = old_cp
                                    if (not sr
                                            and old_cfp is not None):
                                        fg_t["persons"] = old_cfp
                                    # Revert L3 propagation
                                    for tgd, ov, _ in l3_saved:
                                        for col in ("persons",
                                                    "males",
                                                    "females"):
                                            if ov[col] is not None:
                                                tgd[col] = ov[col]

        # --- Strategy 6: L3-guided single-cell fix with L1 propagation ---
        # When L3 pinpoints (section, age, col) and L2/L1 corroborate,
        # fix the specific cell in one component group and propagate P=M+F.
        if not found_any:
            l3_by_sa = defaultdict(dict)
            for f in failures:
                if f["level"] == "L3":
                    l3_by_sa[(f["section"], f.get("age", ""))][f.get("col", "")] = f["diff"]

            xg = constraints.get("L3_cross_group")
            component_groups = xg["component_groups"] if xg else groups[1:]

            for (sec, age), diffs in l3_by_sa.items():
                if found_any:
                    break
                for err_col in ("males", "females", "persons"):
                    if found_any:
                        break
                    d = diffs.get(err_col)
                    if d is None or d == 0:
                        continue
                    age_norm = normalize_age(age)
                    for section in parsed["sections"]:
                        if found_any or section["name"] != sec:
                            continue
                        for row in section["rows"]:
                            if found_any:
                                break
                            if normalize_age(row.get("age", "")) != age_norm:
                                continue
                            for grp in component_groups:
                                if found_any:
                                    break
                                gdata = row.get(grp)
                                if not isinstance(gdata, dict):
                                    continue
                                val = gdata.get(err_col)
                                if val is None:
                                    continue
                                # Try single-digit fix first, then two-digit
                                fix = _find_digit_fix(val, -d)
                                fix2 = None
                                if not fix:
                                    fix2 = _find_two_digit_fix(val, -d)
                                if not fix and not fix2:
                                    continue
                                old_val = gdata[err_col]
                                old_p = gdata.get("persons")
                                gdata[err_col] = (fix[0] if fix
                                                  else fix2[0])
                                if (err_col in ("males", "females")
                                        and not persons_independent
                                        and old_p is not None):
                                    gdata["persons"] = (
                                        gdata.get("males", 0)
                                        + gdata.get("females", 0))
                                nr = verify_all_constraints(
                                    parsed, constraints)
                                if nr["failed"] < total_before:
                                    if fix:
                                        repair_log.append({
                                            "phase": "B",
                                            "action": "digit_fix",
                                            "section": sec,
                                            "age": row.get("age", ""),
                                            "group": grp, "col": err_col,
                                            "old": old_val,
                                            "new": fix[0],
                                            "digit_pos": fix[1],
                                            "old_digit": fix[2],
                                            "new_digit": fix[3],
                                            "strategy": "S6_l3_single",
                                        })
                                    else:
                                        repair_log.append({
                                            "phase": "B",
                                            "action": "two_digit_fix",
                                            "section": sec,
                                            "age": row.get("age", ""),
                                            "group": grp, "col": err_col,
                                            "old": old_val,
                                            "new": fix2[0],
                                            "fixes": fix2[1],
                                            "strategy": "S6_l3_2digit",
                                        })
                                    if old_p != gdata.get("persons"):
                                        repair_log.append({
                                            "phase": "B",
                                            "action": "p_propagate",
                                            "section": sec,
                                            "age": row.get("age", ""),
                                            "group": grp,
                                            "col": "persons",
                                            "old": old_p,
                                            "new": gdata["persons"],
                                        })
                                    found_any = True
                                else:
                                    gdata[err_col] = old_val
                                    if old_p is not None:
                                        gdata["persons"] = old_p

        # --- Strategy 7: Two-digit L2 fix ---
        # When single-digit fix fails, try two simultaneous OCR confusions.
        if not found_any:
            for f in failures:
                if found_any:
                    break
                if f["level"] != "L2":
                    continue
                diff = f["diff"]
                sec, grp, col = f["section"], f["group"], f["col"]

                for section in parsed["sections"]:
                    if found_any or section["name"] != sec:
                        continue
                    for row in section["rows"]:
                        if found_any:
                            break
                        age_norm = normalize_age(row.get("age", ""))
                        gdata = row.get(grp)
                        if not isinstance(gdata, dict):
                            continue
                        val = gdata.get(col)
                        if val is None:
                            continue
                        tc = -diff if age_norm == "total" else diff
                        fix2 = _find_two_digit_fix(val, tc)
                        if not fix2:
                            continue
                        old_val = gdata[col]
                        old_p = gdata.get("persons")
                        gdata[col] = fix2[0]
                        if (col in ("males", "females")
                                and not persons_independent
                                and old_p is not None):
                            gdata["persons"] = (
                                gdata.get("males", 0)
                                + gdata.get("females", 0))
                        nr = verify_all_constraints(parsed, constraints)
                        if nr["failed"] < total_before:
                            repair_log.append({
                                "phase": "B",
                                "action": "two_digit_fix",
                                "section": sec,
                                "age": row.get("age", ""),
                                "group": grp, "col": col,
                                "old": old_val, "new": fix2[0],
                                "fixes": fix2[1],
                                "strategy": "S7_two_digit",
                            })
                            if old_p != gdata.get("persons"):
                                repair_log.append({
                                    "phase": "B",
                                    "action": "p_propagate",
                                    "section": sec,
                                    "age": row.get("age", ""),
                                    "group": grp, "col": "persons",
                                    "old": old_p,
                                    "new": gdata["persons"],
                                })
                            found_any = True
                        else:
                            gdata[col] = old_val
                            if old_p is not None:
                                gdata["persons"] = old_p

        # --- Strategy 8: Paired Total-row M/F two-digit fix ---
        # When L2 failures come in M/F pairs (diff_males = -diff_females)
        # at a group's Total row, fix BOTH simultaneously so persons stays
        # consistent and net failures decrease.
        if not found_any:
            # Group L2 failures by (section, group) and find MF pairs
            from collections import defaultdict
            l2_by_sg = defaultdict(dict)
            for f in failures:
                if f["level"] != "L2":
                    continue
                key = (f["section"], f["group"])
                l2_by_sg[key][f["col"]] = f

            for (sec, grp), col_map in l2_by_sg.items():
                if found_any:
                    break
                if "males" not in col_map or "females" not in col_map:
                    continue
                dm = col_map["males"]["diff"]
                df = col_map["females"]["diff"]
                if dm + df != 0 or dm == 0:
                    continue  # Not an MF-swap pattern

                # Find the Total row for this section/group
                for section in parsed["sections"]:
                    if found_any or section["name"] != sec:
                        continue
                    for row in section["rows"]:
                        if normalize_age(row.get("age", "")) != "total":
                            continue
                        gdata = row.get(grp)
                        if not isinstance(gdata, dict):
                            continue
                        m_val = gdata.get("males")
                        f_val = gdata.get("females")
                        if m_val is None or f_val is None:
                            continue

                        # Try two-digit fix on both columns
                        # For Total: tc = -diff (to make Total match sum)
                        fix_m = _find_two_digit_fix(m_val, -dm)
                        fix_f = _find_two_digit_fix(f_val, -df)
                        if fix_m is None or fix_f is None:
                            # Also try single-digit
                            if fix_m is None:
                                fix_m1 = _find_digit_fix(m_val, -dm)
                                if fix_m1:
                                    fix_m = (fix_m1[0], [fix_m1[1]])
                            if fix_f is None:
                                fix_f1 = _find_digit_fix(f_val, -df)
                                if fix_f1:
                                    fix_f = (fix_f1[0], [fix_f1[1]])
                        if fix_m is None or fix_f is None:
                            continue

                        # Apply both fixes simultaneously
                        old_m = m_val
                        old_f = f_val
                        old_p = gdata.get("persons")
                        gdata["males"] = fix_m[0]
                        gdata["females"] = fix_f[0]
                        if not persons_independent and old_p is not None:
                            gdata["persons"] = fix_m[0] + fix_f[0]

                        nr = verify_all_constraints(parsed, constraints)
                        if nr["failed"] < total_before:
                            repair_log.append({
                                "phase": "B",
                                "action": "paired_total_mf_fix",
                                "section": sec,
                                "age": "Total",
                                "group": grp,
                                "col": "males",
                                "old": old_m, "new": fix_m[0],
                                "strategy": "S8_paired_total",
                            })
                            repair_log.append({
                                "phase": "B",
                                "action": "paired_total_mf_fix",
                                "section": sec,
                                "age": "Total",
                                "group": grp,
                                "col": "females",
                                "old": old_f, "new": fix_f[0],
                                "strategy": "S8_paired_total",
                            })
                            found_any = True
                        else:
                            gdata["males"] = old_m
                            gdata["females"] = old_f
                            if old_p is not None:
                                gdata["persons"] = old_p

        # --- Strategy 9: Paired cross-group L2 fix ---
        # When two groups (e.g., MARRIED + WIDOWED) have symmetric L2 diffs
        # (G1.M=+D, G1.F=-D, G2.M=-D, G2.F=+D), fix all 4 values at one row.
        # Preserves L1 (M+F unchanged) and L3 (group changes cancel).
        if not found_any:
            l2_by_sg = defaultdict(dict)
            for f in failures:
                if f["level"] == "L2":
                    key = (f["section"], f["group"])
                    l2_by_sg[key][f["col"]] = f

            # Find pairs of groups with symmetric MF diffs
            sg_pairs = []
            keys = list(l2_by_sg.keys())
            for i in range(len(keys)):
                for j in range(i + 1, len(keys)):
                    s1, g1 = keys[i]
                    s2, g2 = keys[j]
                    if s1 != s2:
                        continue
                    c1 = l2_by_sg[keys[i]]
                    c2 = l2_by_sg[keys[j]]
                    if not all(k in c1 for k in ("males", "females")):
                        continue
                    if not all(k in c2 for k in ("males", "females")):
                        continue
                    d1m = c1["males"]["diff"]
                    d1f = c1["females"]["diff"]
                    d2m = c2["males"]["diff"]
                    d2f = c2["females"]["diff"]
                    # Check symmetric pattern: d1m=-d1f, d2m=-d2f, d1m=-d2m
                    if (d1m + d1f == 0 and d2m + d2f == 0
                            and d1m + d2m == 0 and d1m != 0):
                        sg_pairs.append((s1, g1, g2, d1m))

            for sec, grp1, grp2, D in sg_pairs:
                if found_any:
                    break
                # D is the L2 diff for grp1.males
                # At the error row: grp1.M is D too high (needs -D),
                #   grp1.F is D too low (needs +D),
                #   grp2.M is D too low (needs +D),
                #   grp2.F is D too high (needs -D)
                for section in parsed["sections"]:
                    if found_any or section["name"] != sec:
                        continue
                    for row in section["rows"]:
                        age = normalize_age(row.get("age", ""))
                        if age in ("total", "grandtotal"):
                            continue
                        gd1 = row.get(grp1)
                        gd2 = row.get(grp2)
                        if not isinstance(gd1, dict) or not isinstance(gd2, dict):
                            continue
                        # D = L2 diff for grp1.males = Total - sum(rows).
                        # If D<0, sum is too high, row needs decrease (change=D).
                        # grp1.M change=D, grp1.F change=-D,
                        # grp2.M change=-D, grp2.F change=D
                        fix1m = _find_digit_fix(gd1.get("males"), D)
                        fix1f = _find_digit_fix(gd1.get("females"), -D)
                        fix2m = _find_digit_fix(gd2.get("males"), -D)
                        fix2f = _find_digit_fix(gd2.get("females"), D)
                        if not all([fix1m, fix1f, fix2m, fix2f]):
                            continue

                        # Apply all 4 fixes
                        old = {
                            (grp1, "males"): gd1["males"],
                            (grp1, "females"): gd1["females"],
                            (grp2, "males"): gd2["males"],
                            (grp2, "females"): gd2["females"],
                        }
                        gd1["males"] = fix1m[0]
                        gd1["females"] = fix1f[0]
                        gd2["males"] = fix2m[0]
                        gd2["females"] = fix2f[0]

                        nr = verify_all_constraints(parsed, constraints)
                        if nr["failed"] < total_before:
                            for g, c, fix in [
                                (grp1, "males", fix1m),
                                (grp1, "females", fix1f),
                                (grp2, "males", fix2m),
                                (grp2, "females", fix2f),
                            ]:
                                repair_log.append({
                                    "phase": "B",
                                    "action": "cross_group_l2_fix",
                                    "section": sec,
                                    "age": row.get("age", ""),
                                    "group": g,
                                    "col": c,
                                    "old": old[(g, c)],
                                    "new": fix[0],
                                    "strategy": "S9_cross_group",
                                })
                            found_any = True
                        else:
                            # Revert
                            gd1["males"] = old[(grp1, "males")]
                            gd1["females"] = old[(grp1, "females")]
                            gd2["males"] = old[(grp2, "males")]
                            gd2["females"] = old[(grp2, "females")]

        # --- Strategy 10: Paired L3 age-swap fix ---
        # When L3 diffs at two ages are equal and opposite for the same columns,
        # fix the total-group values to match component sums. Preserves L2
        # because +D and -D cancel in the column sum.
        if not found_any:
            # Group L3 failures by (section, col) and find opposing-age pairs
            l3_by_sc = defaultdict(list)
            for f in failures:
                if f["level"] == "L3":
                    l3_by_sc[(f["section"], f["col"])].append(f)

            # Look for pairs where diff at age1 = -diff at age2
            paired_l3 = {}  # (section, age1, age2) -> {col: diff}
            for (sec, col), flist in l3_by_sc.items():
                for i in range(len(flist)):
                    for j in range(i + 1, len(flist)):
                        if flist[i]["diff"] + flist[j]["diff"] == 0:
                            a1 = flist[i]["age"]
                            a2 = flist[j]["age"]
                            key = (sec, min(a1, a2), max(a1, a2))
                            if key not in paired_l3:
                                paired_l3[key] = {}
                            paired_l3[key][col] = flist[i]["diff"] if a1 < a2 else flist[j]["diff"]

            for (sec, age1, age2), col_diffs in paired_l3.items():
                if found_any:
                    break
                if "persons" not in col_diffs:
                    continue  # Need at least persons
                # The total group is the one referenced in L3
                # Identify from the failure detail which group is the total
                total_group = None
                for f in failures:
                    if f["level"] == "L3" and f["section"] == sec:
                        detail = f.get("detail", "")
                        # e.g. "POPULATION.persons=103801 != sum(..."
                        if "." in detail:
                            total_group = detail.split(".")[0]
                            break
                if not total_group:
                    continue

                # Find the rows for age1 and age2
                for section in parsed["sections"]:
                    if found_any or section["name"] != sec:
                        continue
                    rows_map = {}
                    for row in section["rows"]:
                        na = normalize_age(row.get("age", ""))
                        if na in (normalize_age(age1), normalize_age(age2)):
                            rows_map[na] = row

                    na1 = normalize_age(age1)
                    na2 = normalize_age(age2)
                    if na1 not in rows_map or na2 not in rows_map:
                        continue
                    r1 = rows_map[na1]
                    r2 = rows_map[na2]
                    gd1 = r1.get(total_group)
                    gd2 = r2.get(total_group)
                    if not isinstance(gd1, dict) or not isinstance(gd2, dict):
                        continue

                    # For each col with a paired diff, fix total group
                    # at both ages. diff at age1 means:
                    # total_group.col(age1) - sum(components) = diff
                    # Fix: total_group.col(age1) -= diff (make it match)
                    old_vals = {}
                    fixes_valid = True
                    for col, diff_a1 in col_diffs.items():
                        v1 = gd1.get(col)
                        v2 = gd2.get(col)
                        if v1 is None or v2 is None:
                            fixes_valid = False
                            break
                        # Verify it's an OCR-plausible fix
                        fix1 = _find_digit_fix(v1, -diff_a1)
                        fix2 = _find_digit_fix(v2, diff_a1)  # opposite diff at age2
                        if not fix1 or not fix2:
                            fixes_valid = False
                            break
                        old_vals[(na1, col)] = v1
                        old_vals[(na2, col)] = v2

                    if not fixes_valid:
                        continue

                    # Apply all fixes
                    for col, diff_a1 in col_diffs.items():
                        gd1[col] = gd1[col] - diff_a1
                        gd2[col] = gd2[col] + diff_a1

                    # Recompute persons if needed (for cols males/females
                    # we need P=M+F)
                    if not persons_independent:
                        for gd in (gd1, gd2):
                            m = gd.get("males")
                            f = gd.get("females")
                            if m is not None and f is not None:
                                gd["persons"] = m + f

                    nr = verify_all_constraints(parsed, constraints)
                    if nr["failed"] < total_before:
                        for col, diff_a1 in col_diffs.items():
                            for na, old_v, new_v, age_label in [
                                (na1, old_vals[(na1, col)],
                                 old_vals[(na1, col)] - diff_a1, age1),
                                (na2, old_vals[(na2, col)],
                                 old_vals[(na2, col)] + diff_a1, age2),
                            ]:
                                repair_log.append({
                                    "phase": "B",
                                    "action": "paired_l3_age_fix",
                                    "section": sec,
                                    "age": age_label,
                                    "group": total_group,
                                    "col": col,
                                    "old": old_v, "new": new_v,
                                    "strategy": "S10_paired_l3",
                                })
                        found_any = True
                    else:
                        # Revert
                        for col, diff_a1 in col_diffs.items():
                            gd1[col] = old_vals[(na1, col)]
                            gd2[col] = old_vals[(na2, col)]
                        # Revert persons too
                        if not persons_independent:
                            for gd, na in [(gd1, na1), (gd2, na2)]:
                                m_old = old_vals.get((na, "males"))
                                f_old = old_vals.get((na, "females"))
                                if m_old is not None and f_old is not None:
                                    gd["persons"] = m_old + f_old

        if not found_any:
            break

    return parsed, repair_log


def _repair_truncated(image_path, parsed, schema, constraints, persons_independent):
    """Phase C: Fill truncated sections via 1 API call per section.

    Detects sections with fewer rows than siblings (missing age rows at the
    end), identifies the missing ages, and re-extracts only those rows.

    Returns (parsed, repair_log, api_calls).
    """
    if image_path is None:
        return parsed, [], 0
    sections = parsed.get("sections", [])
    if len(sections) < 2:
        return parsed, [], 0

    # Count non-Total, non-subtotal age rows per section
    row_counts = {}
    row_ages = {}
    for section in sections:
        sec_name = section.get("name", "")
        ages = set()
        for row in section.get("rows", []):
            age_norm = normalize_age(row.get("age", ""))
            if age_norm != "total":
                ages.add(age_norm)
        row_counts[sec_name] = len(ages)
        row_ages[sec_name] = ages

    max_rows = max(row_counts.values())
    repair_log = []
    api_calls = 0

    for sec_name, count in row_counts.items():
        if count >= max_rows - 1:
            continue
        # This section has significantly fewer rows — check for large L2 diffs
        report = verify_all_constraints(parsed, constraints)
        sec_failures = [f for f in report["failures"]
                        if f.get("section") == sec_name and f["level"] == "L2"]
        if not sec_failures:
            continue
        max_diff = max(abs(f["diff"]) for f in sec_failures)
        if max_diff < 1000:
            continue  # Small diffs, not a truncation issue

        # Identify missing ages from the largest sibling section
        largest_sec = max(row_ages, key=lambda s: len(row_ages[s]))
        missing_ages = row_ages[largest_sec] - row_ages[sec_name]
        missing_ages.discard("total")

        if not missing_ages:
            continue

        logger.info("Phase C: Section '%s' missing %d ages: %s",
                     sec_name, len(missing_ages), sorted(missing_ages))

        # Build prompt for missing rows
        groups = parsed.get("metadata", {}).get("column_groups", [])
        groups_str = ", ".join(f'"{g}"' for g in groups)
        missing_str = ", ".join(f'"{a}"' for a in sorted(missing_ages))

        prompt = f"""I extracted data from section "{sec_name}" of this census table but missed some age rows.

Please read ONLY these age rows from section "{sec_name}":
{missing_str}

For each age row, read ALL column groups: [{groups_str}]
Each group has: persons, males, females.

Return a JSON array:
[
  {{"age": "...", {", ".join(f'"{g}": {{"persons": N, "males": N, "females": N}}' for g in groups[:2])}}}
]

CRITICAL: Read each digit carefully. Common confusions: 3↔8, 5↔6, 0↔9.
Return ONLY valid JSON array."""

        b64 = encode_image(str(image_path))
        try:
            raw = call_gemini(b64, prompt)
            api_calls += 1
        except Exception as e:
            logger.warning("Phase C API call failed: %s", e)
            continue

        # Parse response
        text = raw.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines)

        try:
            new_rows = json.loads(text)
        except json.JSONDecodeError:
            start = text.find("[")
            end = text.rfind("]") + 1
            if start >= 0 and end > start:
                try:
                    new_rows = json.loads(text[start:end])
                except json.JSONDecodeError:
                    logger.warning("Phase C: Could not parse response")
                    continue
            else:
                continue

        if not isinstance(new_rows, list):
            new_rows = [new_rows]

        # Normalize values in new rows
        for row in new_rows:
            for key, val in row.items():
                if key == "age":
                    continue
                if isinstance(val, dict):
                    for col in ("persons", "males", "females"):
                        if col in val:
                            val[col] = _normalize_value(val[col])

        # Merge into the section — insert before Total row
        for section in parsed["sections"]:
            if section["name"] != sec_name:
                continue
            # Find Total row index
            total_idx = None
            for i, row in enumerate(section["rows"]):
                if normalize_age(row.get("age", "")) == "total":
                    total_idx = i
                    break

            for new_row in new_rows:
                age_norm = normalize_age(new_row.get("age", ""))
                if age_norm in row_ages[sec_name]:
                    continue  # Already exists
                if total_idx is not None:
                    section["rows"].insert(total_idx, new_row)
                    total_idx += 1
                else:
                    section["rows"].append(new_row)
                repair_log.append({
                    "phase": "C", "action": "add_missing_row",
                    "section": sec_name, "age": new_row.get("age", ""),
                })

    return parsed, repair_log, api_calls


def _repair_structural(image_path, parsed, schema, constraints, persons_independent):
    """Phase D: Re-extract sections with structural misreads (1-2 API calls).

    Detects sections where the model read from wrong columns entirely
    (L3 failures where component sum is a tiny fraction of total, or all
    age labels are empty).

    Returns (parsed, repair_log, api_calls).
    """
    if image_path is None:
        return parsed, [], 0
    report = verify_all_constraints(parsed, constraints)
    if report["all_passed"]:
        return parsed, [], 0

    sections = parsed.get("sections", [])
    groups = parsed.get("metadata", {}).get("column_groups", [])
    repair_log = []
    api_calls = 0

    # Detect structurally broken sections
    broken_sections = set()

    # Check 1: All L3 failures in one section with enormous relative diffs
    xg = constraints.get("L3_cross_group")
    if xg:
        from collections import Counter
        l3_sec_counts = Counter()
        l3_sec_huge = Counter()
        for f in report["failures"]:
            if f["level"] == "L3":
                sec = f.get("section", "")
                l3_sec_counts[sec] += 1
                detail = f.get("detail", "")
                diff = abs(f.get("diff", 0))
                # Check if diff is > 50% of the total group value
                if "!=" in detail:
                    parts = detail.split("!=")
                    try:
                        total_val = int(parts[0].split("=")[-1])
                        if total_val > 0 and diff / total_val > 0.5:
                            l3_sec_huge[sec] += 1
                    except (ValueError, IndexError):
                        pass

        for sec, count in l3_sec_counts.items():
            if count >= 10 and l3_sec_huge.get(sec, 0) >= count // 2:
                broken_sections.add(sec)

    # Check 2: Sections where all age labels are empty
    for section in sections:
        sec_name = section.get("name", "")
        rows = section.get("rows", [])
        if rows and all(not row.get("age", "").strip() for row in rows):
            broken_sections.add(sec_name)

    if not broken_sections:
        return parsed, [], 0

    b64 = encode_image(str(image_path))

    for sec_name in broken_sections:
        logger.info("Phase D: Re-extracting structurally broken section '%s'", sec_name)

        # Re-extract with explicit section targeting
        extraction_prompt = build_oneshot_extraction_prompt(
            schema, persons_independent=persons_independent,
            target_section=sec_name)

        try:
            raw = call_gemini(b64, extraction_prompt)
            api_calls += 1
        except Exception as e:
            logger.warning("Phase D API call failed for '%s': %s", sec_name, e)
            continue

        new_parsed = parse_response(raw, schema=schema,
                                    persons_independent=persons_independent)
        if not new_parsed or not new_parsed.get("sections"):
            logger.warning("Phase D: Could not parse re-extraction for '%s'", sec_name)
            continue

        new_sec_data = new_parsed["sections"][0]
        new_sec_data["name"] = sec_name

        # Replace the broken section
        for i, section in enumerate(parsed["sections"]):
            if section["name"] == sec_name:
                old_row_count = len(section.get("rows", []))
                parsed["sections"][i] = new_sec_data
                new_row_count = len(new_sec_data.get("rows", []))
                repair_log.append({
                    "phase": "D", "action": "re_extract_section",
                    "section": sec_name,
                    "old_rows": old_row_count,
                    "new_rows": new_row_count,
                })
                break

    return parsed, repair_log, api_calls


def _multi_reading_repair(image_path, parsed, schema, constraints,
                          persons_independent):
    """Phase E: Multi-reading vote for persistent failures (2-3 API calls).

    For cells still failing after phases A-D, make 2 additional independent
    readings with different prompt emphasis and use digit_level_vote() to
    resolve ambiguous digits.

    Returns (parsed, repair_log, api_calls).
    """
    if image_path is None:
        return parsed, [], 0
    from ensemble import digit_level_vote

    report = verify_all_constraints(parsed, constraints)
    if report["all_passed"]:
        return parsed, [], 0

    repair_log = []
    api_calls = 0

    # Identify suspicious cells from remaining failures
    suspicious = identify_suspicious_cells(parsed, report["failures"])
    if not suspicious:
        return parsed, [], 0

    # Deduplicate to (section, age, group) level
    seen = set()
    cells_to_read = []
    for (sec, age, group, _col), _score in suspicious:
        key = (sec, age, group)
        if key not in seen:
            seen.add(key)
            cells_to_read.append(key)
        if len(cells_to_read) >= 15:
            break

    if not cells_to_read:
        return parsed, [], 0

    # Format cells for prompt
    cells_desc = []
    for sec, age, group in cells_to_read:
        cells_desc.append(
            f'  - Section: "{sec}", Age: "{age}", Group: "{group}"'
            f' → read Persons, Males, Females')

    # Two additional readings with different emphasis
    prompts = [
        "Read each digit ONE AT A TIME from left to right. For multi-digit "
        "numbers, say each digit aloud to yourself before writing the number. "
        "Pay special attention to: 3 vs 8, 5 vs 6, 0 vs 9.",

        "Focus on the ONES digit first (rightmost), then work leftward. "
        "Numbers in census tables follow patterns — if a digit seems off, "
        "check the surrounding rows for context.",
    ]

    all_readings = []  # list of dicts: {(sec, age, group): {col: value}}

    # Collect original reading
    orig_reading = {}
    for sec, age, group in cells_to_read:
        for section in parsed["sections"]:
            if section["name"] != sec:
                continue
            for row in section["rows"]:
                if normalize_age(row.get("age", "")) != normalize_age(age):
                    continue
                gdata = row.get(group, {})
                if isinstance(gdata, dict):
                    orig_reading[(sec, age, group)] = {
                        col: gdata.get(col) for col in ("persons", "males", "females")
                    }
    all_readings.append(orig_reading)

    b64 = encode_image(str(image_path))

    for prompt_extra in prompts:
        full_prompt = f"""{prompt_extra}

Re-read ONLY these specific cells from the census table image:

{chr(10).join(cells_desc)}

Return a JSON array:
[
  {{"section": "...", "age": "...", "group": "...", "persons": N, "males": N, "females": N}}
]

Return ONLY valid JSON array."""

        try:
            raw = call_gemini(b64, full_prompt)
            api_calls += 1
        except Exception as e:
            logger.warning("Phase E API call failed: %s", e)
            continue

        text = raw.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines)

        try:
            recheck_data = json.loads(text)
        except json.JSONDecodeError:
            start = text.find("[")
            end = text.rfind("]") + 1
            if start >= 0 and end > start:
                try:
                    recheck_data = json.loads(text[start:end])
                except json.JSONDecodeError:
                    continue
            else:
                continue

        if not isinstance(recheck_data, list):
            recheck_data = [recheck_data]

        reading = {}
        for item in recheck_data:
            if not isinstance(item, dict):
                continue
            sec = item.get("section", "")
            age = item.get("age", "")
            group = item.get("group", "")
            reading[(sec, age, group)] = {
                col: _normalize_value(item.get(col))
                for col in ("persons", "males", "females")
            }
        all_readings.append(reading)

    if len(all_readings) < 2:
        return parsed, repair_log, api_calls

    # Digit-level vote across all readings
    total_before = report["failed"]

    for sec, age, group in cells_to_read:
        for section in parsed["sections"]:
            if section["name"] != sec:
                continue
            for row in section["rows"]:
                if normalize_age(row.get("age", "")) != normalize_age(age):
                    continue
                gdata = row.get(group)
                if not isinstance(gdata, dict):
                    continue

                for col in ("persons", "males", "females"):
                    values = []
                    for reading in all_readings:
                        val = reading.get((sec, age, group), {}).get(col)
                        if val is not None:
                            values.append(val)

                    if len(values) < 2:
                        continue

                    voted = digit_level_vote(values)
                    if voted is None or voted == gdata.get(col):
                        continue

                    old_val = gdata[col]
                    gdata[col] = voted
                    new_report = verify_all_constraints(parsed, constraints)
                    if new_report["failed"] < total_before:
                        repair_log.append({
                            "phase": "E", "action": "voted_fix",
                            "section": sec, "age": age,
                            "group": group, "col": col,
                            "old": old_val, "new": voted,
                            "readings": values,
                        })
                        total_before = new_report["failed"]
                    else:
                        gdata[col] = old_val

    return parsed, repair_log, api_calls


def _upscale_reextract(image_path, parsed, schema, constraints,
                       persons_independent):
    """Phase F: Re-extract from upscaled image for persistent failures.

    When constraint failures remain after Phases A-E, the image may simply
    be too low-resolution for reliable digit/column reading.  This phase:

    1. Upscales the image 2-3× (based on width)
    2. Re-extracts each failing column group individually (targeted zoom)
    3. Merges re-extracted values into the main result
    4. Runs Phase A+B repair on the merged result

    Using per-group extraction from the upscaled image gives the model both
    high resolution AND reduced column-attribution ambiguity, since it only
    needs to locate 3 columns (P/M/F) for one group at a time.

    Returns (parsed, repair_log, api_calls).
    """
    if image_path is None:
        return parsed, [], 0

    from PIL import Image as PILImage

    report = verify_all_constraints(parsed, constraints)
    if report["all_passed"]:
        return parsed, [], 0

    # Determine upscale factor from image width
    img = PILImage.open(str(image_path))
    w, h = img.size
    if min(w, h) >= 3000:
        # Already large — upscaling won't help
        return parsed, [], 0
    factor = 3 if max(w, h) < 2000 else 2

    new_w, new_h = w * factor, h * factor
    img_up = img.resize((new_w, new_h), PILImage.LANCZOS)
    upscaled_path = image_path.parent / f"{image_path.stem}_up{factor}x.png"
    img_up.save(str(upscaled_path))
    b64_up = encode_image(str(upscaled_path))
    logger.info("Phase F: Upscaled %dx%d → %dx%d (%d×)", w, h, new_w, new_h, factor)

    # Identify which (section, group) pairs have failures
    failing_sec_groups = {}  # sec_name -> set of group_names
    for f in report["failures"]:
        sec = f.get("section", "")
        grp = f.get("group", "")
        if grp:
            failing_sec_groups.setdefault(sec, set()).add(grp)

    if not failing_sec_groups:
        return parsed, [], 0

    repair_log = []
    api_calls = 0

    # Build lookup: sec_name -> section dict (for row-level merge)
    sec_lookup = {}
    for section in parsed["sections"]:
        sec_lookup[section["name"]] = section

    n_groups = sum(len(gs) for gs in failing_sec_groups.values())
    logger.info("Phase F: Re-extracting %d group(s) from upscaled image", n_groups)

    for sec_name, groups in failing_sec_groups.items():
        section = sec_lookup.get(sec_name)
        if section is None:
            continue

        is_multi = (len(schema.sections or []) > 1)

        for gname in groups:
            prompt = build_oneshot_extraction_prompt(
                schema, persons_independent=persons_independent,
                target_group=gname,
                target_section=sec_name if is_multi else None)

            raw = None
            for attempt in range(3):
                try:
                    raw = call_gemini(b64_up, prompt)
                    break
                except Exception as e:
                    wait = 2 ** attempt
                    logger.warning("Phase F API error (attempt %d): %s",
                                   attempt + 1, e)
                    if attempt < 2:
                        time.sleep(wait)
            if raw is None:
                logger.warning("Phase F: Failed to extract group '%s'", gname)
                continue
            api_calls += 1

            g_parsed = parse_response(
                raw, schema=schema,
                persons_independent=persons_independent)
            if not g_parsed or not g_parsed.get("sections"):
                logger.warning("Phase F: Could not parse group '%s'", gname)
                continue

            # Merge: update each row's group data from the upscaled reading
            new_rows = g_parsed["sections"][0].get("rows", [])
            new_by_age = {}
            for row in new_rows:
                age_norm = normalize_age(row.get("age", ""))
                # The group data may be under gname or as flat P/M/F
                gdata = row.get(gname, {})
                if not isinstance(gdata, dict) or not gdata:
                    gdata = {}
                    for k in ("persons", "males", "females"):
                        if k in row and row[k] is not None:
                            gdata[k] = row[k]
                if gdata:
                    new_by_age[age_norm] = gdata

            if not new_by_age:
                continue

            # Apply the merge: for each row, replace the group's values
            # with the upscaled reading — but only if it reduces failures.
            # Save originals so we can revert if it makes things worse.
            originals = {}
            for row in section["rows"]:
                age_norm = normalize_age(row.get("age", ""))
                if age_norm in new_by_age and gname in row:
                    originals[age_norm] = dict(row[gname])
                    row[gname] = new_by_age[age_norm]

            # Check if the merge helped
            new_constraints = derive_constraints(
                parsed, schema=schema,
                persons_independent=persons_independent)
            new_report = verify_all_constraints(parsed, new_constraints)

            if new_report["failed"] < report["failed"]:
                # Improvement — keep the merge
                improved = report["failed"] - new_report["failed"]
                logger.info("  Phase F: group '%s' in '%s': -%d failures",
                            gname, sec_name, improved)
                repair_log.append({
                    "phase": "F", "action": "upscale_reextract",
                    "section": sec_name, "group": gname,
                    "old_failures": report["failed"],
                    "new_failures": new_report["failed"],
                })
                report = new_report
                constraints = new_constraints
            else:
                # No improvement or worse — revert
                for row in section["rows"]:
                    age_norm = normalize_age(row.get("age", ""))
                    if age_norm in originals and gname in row:
                        row[gname] = originals[age_norm]

            if report["all_passed"]:
                break
        if report["all_passed"]:
            break

    # Run Phase A+B on the merged result (free cleanup)
    if not report["all_passed"]:
        parsed, log_ab = _detect_and_fix_mf_swaps(
            parsed, constraints, persons_independent)
        repair_log.extend(log_ab)
        if log_ab:
            constraints = derive_constraints(
                parsed, schema=schema,
                persons_independent=persons_independent)
        parsed, log_b = _deductive_digit_fix(parsed, constraints,
                                              persons_independent)
        repair_log.extend(log_b)

    # Clean up upscaled image
    try:
        upscaled_path.unlink()
    except OSError:
        pass

    return parsed, repair_log, api_calls


def constraint_repair(image_path, parsed, schema, constraints,
                      persons_independent):
    """Constraint-driven repair cascade (Step 4c).

    Runs repair phases in order of cost:
      A: M/F swap detection (FREE)
      B: Deductive digit fix (FREE)
      C: Truncated section fill (1 API call per section)
      D: Structural re-extract (1-2 API calls per section)
      E: Multi-reading vote (2-3 API calls)
      F: Upscale + per-group re-extract (1 API call per failing group)

    Each phase re-verifies constraints after fixes. The cascade loops up to
    3 rounds, breaking early if all constraints pass or no progress is made.

    Returns (parsed, repair_log, api_calls_used).
    """
    repair_log = []
    api_calls = 0

    for round_num in range(3):
        report = verify_all_constraints(parsed, constraints)
        if report["all_passed"]:
            break

        failures_before = report["failed"]
        logger.info("Repair round %d: %d failures", round_num + 1, failures_before)

        # Phase A: M/F swap (FREE)
        parsed, log_a = _detect_and_fix_mf_swaps(
            parsed, constraints, persons_independent)
        repair_log.extend(log_a)
        if log_a:
            constraints = derive_constraints(
                parsed, schema=schema, persons_independent=persons_independent)
            report = verify_all_constraints(parsed, constraints)
            logger.info("  Phase A: %d swaps fixed → %d failures",
                        len(log_a), report["failed"])
            if report["all_passed"]:
                break

        # Phase B: Digit deduction (FREE)
        parsed, log_b = _deductive_digit_fix(
            parsed, constraints, persons_independent)
        repair_log.extend(log_b)
        if log_b:
            constraints = derive_constraints(
                parsed, schema=schema, persons_independent=persons_independent)
            report = verify_all_constraints(parsed, constraints)
            logger.info("  Phase B: %d digit fixes → %d failures",
                        len(log_b), report["failed"])
            if report["all_passed"]:
                break

        # Phase C: Truncated section fill (API calls)
        parsed, log_c, calls_c = _repair_truncated(
            image_path, parsed, schema, constraints, persons_independent)
        repair_log.extend(log_c)
        api_calls += calls_c
        if log_c:
            constraints = derive_constraints(
                parsed, schema=schema, persons_independent=persons_independent)
            report = verify_all_constraints(parsed, constraints)
            logger.info("  Phase C: %d rows added → %d failures",
                        len(log_c), report["failed"])
            if report["all_passed"]:
                break

        # Phase D: Structural re-extract (API calls)
        parsed, log_d, calls_d = _repair_structural(
            image_path, parsed, schema, constraints, persons_independent)
        repair_log.extend(log_d)
        api_calls += calls_d
        if log_d:
            constraints = derive_constraints(
                parsed, schema=schema, persons_independent=persons_independent)
            report = verify_all_constraints(parsed, constraints)
            logger.info("  Phase D: %d sections re-extracted → %d failures",
                        len(log_d), report["failed"])
            if report["all_passed"]:
                break

        # Phase E: Multi-reading vote (API calls)
        parsed, log_e, calls_e = _multi_reading_repair(
            image_path, parsed, schema, constraints, persons_independent)
        repair_log.extend(log_e)
        api_calls += calls_e
        if log_e:
            constraints = derive_constraints(
                parsed, schema=schema, persons_independent=persons_independent)
            report = verify_all_constraints(parsed, constraints)
            logger.info("  Phase E: %d voted fixes → %d failures",
                        len(log_e), report["failed"])
            if report["all_passed"]:
                break

        # Phase F: Upscale + targeted re-extract (API calls, last resort)
        parsed, log_f, calls_f = _upscale_reextract(
            image_path, parsed, schema, constraints, persons_independent)
        repair_log.extend(log_f)
        api_calls += calls_f
        if log_f:
            constraints = derive_constraints(
                parsed, schema=schema, persons_independent=persons_independent)
            report = verify_all_constraints(parsed, constraints)
            logger.info("  Phase F: %d upscale fixes → %d failures",
                        len(log_f), report["failed"])
            if report["all_passed"]:
                break

        # Check progress — break if no improvement
        if report["failed"] >= failures_before:
            logger.info("  No progress in round %d, stopping", round_num + 1)
            break

    return parsed, repair_log, api_calls


# ---------------------------------------------------------------------------
# to_tidy_dataframe() — pandas DataFrame for data science
# ---------------------------------------------------------------------------

def to_tidy_dataframe(parsed):
    """Convert parsed extraction to a tidy pandas DataFrame.

    One row per (section, age, group) observation:
        section | age | group | persons | males | females
    """
    import pandas as pd

    records = []
    for section in parsed.get("sections", []):
        sec_name = section.get("name", "")
        for row in section.get("rows", []):
            age = row.get("age", "")
            for key, val in row.items():
                if key == "age":
                    continue
                if isinstance(val, dict):
                    records.append({
                        "section": sec_name,
                        "age": age,
                        "group": key,
                        "persons": val.get("persons"),
                        "males": val.get("males"),
                        "females": val.get("females"),
                    })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# to_legacy_format() — adapter for existing Excel/JSON export
# ---------------------------------------------------------------------------

def to_legacy_format(parsed):
    """Convert parsed oneshot data to the legacy all_results dict shape.

    Returns:
        (all_results, schema) tuple compatible with
        export_multigroup_to_excel() and export_multigroup_to_json().
    """
    meta = parsed.get("metadata", {})
    groups = meta.get("column_groups", [])
    sections = parsed.get("sections", [])

    # Build TableSchema for export
    schema = TableSchema(
        title=meta.get("title", ""),
        region=meta.get("region", ""),
        year=meta.get("year", 0),
        data_type=meta.get("data_type", "absolute"),
        column_groups=[{"name": g} for g in groups],
        sections=[{"name": s.get("name", "")} for s in sections],
    )

    # Detect cross-group constraints for schema
    constraints = derive_constraints(parsed)
    xg = constraints.get("L3_cross_group")
    if xg:
        schema.cross_group_constraints = {
            xg["total_group"]: xg["component_groups"]
        }

    # Build all_results: {section_name: {group_name: {"rows": [...]}}}
    all_results = {}
    for section in sections:
        sec_name = section.get("name", "")
        all_results[sec_name] = {}
        for group in groups:
            rows = []
            for row in section.get("rows", []):
                gdata = row.get(group, {})
                if not isinstance(gdata, dict):
                    continue
                rows.append({
                    "age": row.get("age", ""),
                    "persons": gdata.get("persons"),
                    "males": gdata.get("males"),
                    "females": gdata.get("females"),
                })
            all_results[sec_name][group] = {"rows": rows}

    return all_results, schema


# ---------------------------------------------------------------------------
# extract_and_verify() — main orchestrator
# ---------------------------------------------------------------------------

def validate_age_ordering(parsed):
    """Check that age labels appear in ascending numeric order within each section.

    Detects potential row-swap errors that pass all constraint checks
    (vertical sums and P=M+F are preserved even when rows are swapped).

    Returns list of warning strings, empty if ordering looks correct.
    """
    warnings = []
    for section in parsed.get("sections", []):
        sec_name = section.get("name", "")
        numeric_ages = []
        for row in section.get("rows", []):
            age = row.get("age", "")
            age_norm = normalize_age(age)
            if age_norm in ("total", "grandtotal"):
                continue
            if any(kw in age_norm for kw in ("mean", "median", "average")):
                continue
            m = re.match(r'^(\d+)', age_norm)
            if m:
                numeric_ages.append((int(m.group(1)), age))

        for i in range(1, len(numeric_ages)):
            if numeric_ages[i][0] < numeric_ages[i - 1][0]:
                warnings.append(
                    f"Section '{sec_name}': '{numeric_ages[i][1]}' "
                    f"(starts at {numeric_ages[i][0]}) appears after "
                    f"'{numeric_ages[i-1][1]}' (starts at {numeric_ages[i-1][0]}) "
                    f"— possible row swap")
    return warnings


def extract_and_verify(image_path, retry_on_fail=False, fallback=False,
                       output_dir=None):
    """Main orchestrator: 3-call max pipeline.

    Call 1: Schema discovery (Gemini) — understand table structure
    Call 2: Tailored extraction (Gemini) — read digits with schema-aware prompt
    Call 3: Targeted recheck (Gemini, optional) — fix constraint failures

    Args:
        image_path: Path to census table image.
        retry_on_fail: If True, retry once with slightly different prompt.
        fallback: If True, fall back to MoE pipeline on constraint failure.
        output_dir: Directory for output files. If None, uses default RESULTS_DIR.

    Returns:
        dict with keys: parsed, schema, constraints, report, df, output_paths, elapsed
    """
    image_path = Path(image_path)
    if not image_path.exists():
        print(f"ERROR: Image not found: {image_path}")
        return None

    effective_output_dir = Path(output_dir) if output_dir else RESULTS_DIR
    effective_output_dir.mkdir(parents=True, exist_ok=True)

    # Build unique output stem: include parent dir if it's a year (e.g. "1931")
    # to avoid collisions when multiple years have identical image names.
    original_image_path = image_path          # preserve for JSON output
    parent = image_path.parent.name
    if parent.isdigit() and not image_path.stem.endswith(f"_{parent}"):
        stem = f"{image_path.stem}_{parent}"
    else:
        stem = image_path.stem
    print(f"\n{'='*70}")
    print(f"One-Shot Extraction: {image_path.name}")
    print(f"{'='*70}")

    t0 = time.time()

    # Step 1a: Structure discovery (Call 1)
    print("\nStep 1a: Schema discovery (Call 1)")
    schema = discover_schema_single(image_path)
    api_calls = 1
    schema_elapsed = time.time() - t0

    if schema is None:
        print("  FAILED: Schema discovery returned None")
        if fallback:
            return _fallback_to_moe(image_path)
        return None

    # Step 1b: Auto-rotate if schema says image is sideways
    if schema.rotation in (90, 180, 270):
        from PIL import Image as PILImage
        rot_deg = schema.rotation
        print(f"  Image needs {rot_deg}° clockwise rotation — rotating and "
              "re-running schema discovery")
        img = PILImage.open(str(image_path))
        # PIL rotate() is counter-clockwise, so negate
        img = img.rotate(-rot_deg, expand=True)
        rotated_path = image_path.parent / f"{image_path.stem}_rot{rot_deg}.png"
        img.save(str(rotated_path))
        image_path = rotated_path
        # Re-run schema discovery on the upright image
        schema = discover_schema_single(image_path)
        api_calls += 1
        if schema is None:
            print("  FAILED: Schema discovery on rotated image returned None")
            if fallback:
                return _fallback_to_moe(image_path)
            return None
        schema_elapsed = time.time() - t0

    # Detect independent-column tables ("of each sex" → P, M, F are
    # each independently per-N, not P=M+F)
    persons_independent = False
    if schema.data_type == "proportional":
        title_lower = (schema.title or "").lower()
        if "of each sex" in title_lower or "each sex" in title_lower:
            persons_independent = True

    print(f"  Schema discovered in {schema_elapsed:.1f}s:")
    print(f"    Data type: {schema.data_type}"
          + (f" (per {schema.denominator})" if schema.denominator else ""))
    print(f"    Groups: {[cg.get('name', '?') for cg in schema.column_groups]}")
    print(f"    Rows: {len(schema.row_labels)}")
    print(f"    Has persons: {schema.has_persons_column}"
          + (" (INDEPENDENT — P≠M+F)" if persons_independent else ""))
    if schema.subtotal_hierarchy:
        print(f"    Subtotal hierarchy: {len(schema.subtotal_hierarchy)} groups")
    if schema.cross_group_constraints:
        for tg, cgs in schema.cross_group_constraints.items():
            print(f"    Cross-group: {tg} = {' + '.join(cgs)}")

    # Step 2: Tailored extraction (Call 2+)
    # For multi-section tables, extract one section at a time to avoid timeouts.
    b64 = encode_image(str(image_path))
    n_sections = len(schema.sections) if schema.sections else 1
    use_per_section = n_sections > 1

    if use_per_section:
        # Per-section extraction: one call per section
        sec_names = [s.get("name", "?") for s in schema.sections]
        print(f"\nStep 1b: Per-section extraction ({n_sections} sections)")
        all_section_data = []
        t1 = time.time()

        for si, sec_name in enumerate(sec_names):
            print(f"  [{si+1}/{n_sections}] Section: {sec_name}")
            extraction_prompt = build_oneshot_extraction_prompt(
                schema, persons_independent=persons_independent,
                target_section=sec_name)

            raw = None
            for attempt in range(3):
                try:
                    raw = call_gemini(b64, extraction_prompt)
                    break
                except Exception as e:
                    wait = 2 ** attempt
                    print(f"    Attempt {attempt+1} failed: {e}")
                    if attempt < 2:
                        print(f"    Retrying in {wait}s...")
                        time.sleep(wait)
            if raw is None:
                print(f"    FAILED: Section {sec_name}")
                continue  # Skip this section, try others
            api_calls += 1

            sec_parsed = parse_response(raw, schema=schema,
                                        persons_independent=persons_independent)
            if sec_parsed and sec_parsed.get("sections"):
                # Take the first (and only) section from the response
                sec_data = sec_parsed["sections"][0]
                sec_data["name"] = sec_name  # Ensure name matches
                all_section_data.append(sec_data)
                n_rows = len(sec_data.get("rows", []))
                print(f"    Extracted: {n_rows} rows")
            else:
                print(f"    FAILED: Could not parse section {sec_name}")

        extract_elapsed = time.time() - t1
        print(f"  Per-section extraction: {extract_elapsed:.1f}s")

        if not all_section_data:
            print("  FAILED: No sections extracted")
            if fallback:
                return _fallback_to_moe(image_path)
            return None

        # Build combined parsed structure
        meta = {"title": schema.title, "region": schema.region,
                "year": schema.year, "data_type": schema.data_type,
                "column_groups": [cg.get("name", "?")
                                  for cg in schema.column_groups]}
        parsed = {"metadata": meta, "sections": all_section_data}
    else:
        # Single-section: try one call for all groups first
        print("\nStep 1b: Tailored extraction (Call 2)")
        t1 = time.time()
        extraction_prompt = build_oneshot_extraction_prompt(
            schema, persons_independent=persons_independent)

        raw = None
        for attempt in range(3):
            try:
                raw = call_gemini(b64, extraction_prompt)
                break
            except Exception as e:
                wait = 2 ** attempt
                print(f"  Attempt {attempt+1} failed: {e}")
                if attempt < 2:
                    print(f"  Retrying in {wait}s...")
                    time.sleep(wait)

        if raw is None and len(schema.column_groups) > 1:
            # All-at-once failed — fall back to per-group extraction
            all_group_names = [cg.get("name", f"Group_{i}")
                               for i, cg in enumerate(schema.column_groups)]
            print(f"  All-at-once failed; switching to per-group extraction "
                  f"({len(all_group_names)} groups)")
            merged_rows_by_age = {}  # age -> {group_name: {p, m, f}}
            for gi, gname in enumerate(all_group_names):
                print(f"  [{gi+1}/{len(all_group_names)}] Group: {gname}")
                gp = build_oneshot_extraction_prompt(
                    schema, persons_independent=persons_independent,
                    target_group=gname)
                g_raw = None
                for attempt in range(3):
                    try:
                        g_raw = call_gemini(b64, gp)
                        break
                    except Exception as e:
                        wait = 2 ** attempt
                        print(f"    Attempt {attempt+1} failed: {e}")
                        if attempt < 2:
                            print(f"    Retrying in {wait}s...")
                            time.sleep(wait)
                if g_raw is None:
                    print(f"    FAILED: Could not extract group {gname}")
                    continue
                api_calls += 1
                g_parsed = parse_response(g_raw, schema=schema,
                                          persons_independent=persons_independent)
                if g_parsed and g_parsed.get("sections"):
                    g_rows = g_parsed["sections"][0].get("rows", [])
                    n_ok = 0
                    for row in g_rows:
                        age = row.get("age", "")
                        if age not in merged_rows_by_age:
                            merged_rows_by_age[age] = {"age": age}
                        # Extract group data — might be under gname or as flat fields
                        gdata = row.get(gname, {})
                        if isinstance(gdata, dict) and gdata:
                            merged_rows_by_age[age][gname] = gdata
                        else:
                            # Flat response: persons/males/females at top level
                            vals = {}
                            for k in ("persons", "males", "females"):
                                if k in row and row[k] is not None:
                                    vals[k] = row[k]
                            if vals:
                                merged_rows_by_age[age][gname] = vals
                        n_ok += 1
                    print(f"    Extracted: {n_ok} rows")
                else:
                    print(f"    FAILED: Could not parse group {gname}")

            extract_elapsed = time.time() - t1
            print(f"  Per-group extraction: {extract_elapsed:.1f}s")

            if not merged_rows_by_age:
                print("  FAILED: No groups extracted")
                if fallback:
                    return _fallback_to_moe(image_path)
                return None

            # Build parsed structure from merged per-group results
            # Preserve row order from schema
            ordered_rows = []
            for label in (schema.row_labels or []):
                for age_key, row_data in merged_rows_by_age.items():
                    if age_key == label or age_key.strip() == label.strip():
                        ordered_rows.append(row_data)
                        break
            # Add any rows not in schema labels (shouldn't happen, but safe)
            found_ages = {r["age"] for r in ordered_rows}
            for age_key in merged_rows_by_age:
                if age_key not in found_ages:
                    ordered_rows.append(merged_rows_by_age[age_key])

            meta = {"title": schema.title, "region": schema.region,
                    "year": schema.year, "data_type": schema.data_type,
                    "column_groups": [cg.get("name", "?")
                                      for cg in schema.column_groups]}
            sec_name = schema.title or "Section 1"
            parsed = {"metadata": meta,
                      "sections": [{"name": sec_name, "rows": ordered_rows}]}
            raw = None  # signal that we used per-group path
        elif raw is None:
            print("  FAILED: All API attempts failed")
            if fallback:
                return _fallback_to_moe(image_path)
            return None

        if raw is not None:
            api_calls += 1
            extract_elapsed = time.time() - t1
            print(f"  Extraction call: {extract_elapsed:.1f}s")
            parsed = parse_response(raw, schema=schema,
                                    persons_independent=persons_independent)

    # Step 2: Verify parse
    print("\nStep 2: Parse response")
    if parsed is None:
        print("  FAILED: Could not parse Gemini response")
        if fallback:
            return _fallback_to_moe(image_path)
        return None

    n_sections = len(parsed.get("sections", []))
    n_groups = len(parsed.get("metadata", {}).get("column_groups", []))
    n_rows = sum(len(s.get("rows", []))
                 for s in parsed.get("sections", []))
    print(f"  Parsed: {n_sections} sections, {n_groups} groups, {n_rows} rows")

    meta = parsed.get("metadata", {})
    print(f"  Title: {meta.get('title', '?')}")
    print(f"  Region: {meta.get('region', '?')}, Year: {meta.get('year', '?')}")
    print(f"  Groups: {meta.get('column_groups', [])}")

    # Step 3: Derive constraints (schema-enriched)
    print("\nStep 3: Derive constraints")
    constraints = derive_constraints(parsed, schema=schema,
                                     persons_independent=persons_independent)
    n_l1 = len(constraints.get("L1_row", []))
    n_l2 = len(constraints.get("L2_vertical", []))
    n_l2s = len(constraints.get("L2_subtotal", []))
    xg = constraints.get("L3_cross_group")
    xs = constraints.get("L4_cross_section")
    n_l5 = len(constraints.get("L5_non_negative", []))
    print(f"  L1 (P=M+F): {n_l1} checks")
    print(f"  L2 (vertical sum): {n_l2} checks"
          + (f" + {n_l2s} subtotal checks" if n_l2s else ""))
    print(f"  L3 (cross-group): "
          f"{'active — ' + xg['total_group'] + ' = sum(' + ', '.join(xg['component_groups']) + ')' if xg else 'not detected'}")
    print(f"  L4 (cross-section): "
          f"{'active — ' + xs['total_section'] + ' = sum(' + ', '.join(xs['component_sections']) + ')' if xs else 'not detected'}")
    print(f"  L5 (non-negative): {n_l5} checks")
    if constraints.get("known_totals"):
        print(f"  Known totals: {constraints['known_totals']}")

    # Step 4: Verify
    print("\nStep 4: Verify all constraints")
    report = verify_all_constraints(parsed, constraints)
    print(f"  Total checks: {report['total_checks']}")
    print(f"  Passed: {report['passed']}")
    print(f"  Failed: {report['failed']}")

    if report["all_passed"]:
        print("  ALL CONSTRAINTS PASSED")
    else:
        print(f"\n  FAILURES ({report['failed']}):")
        for f in report["failures"]:
            print(f"    [{f['level']}] {f.get('section', '')}/{f.get('age', '')} "
                  f"{f.get('group', '')} {f.get('col', '')}: {f['detail']}")

        # Attempt repair for L1 failures
        l1_failures = [f for f in report["failures"] if f["level"] == "L1"]
        if l1_failures:
            print(f"\n  Attempting repair of {len(l1_failures)} L1 failures...")
            parsed, repair_log = attempt_repair(parsed, report["failures"])
            for r in repair_log:
                print(f"    FIX {r['section']}/{r['age']} {r['group']}.{r['col']}: "
                      f"{r['old']} -> {r['new']} (diff={r['diff']})")

            # Re-derive and re-verify
            constraints = derive_constraints(parsed, schema=schema,
                                             persons_independent=persons_independent)
            report = verify_all_constraints(parsed, constraints)
            print(f"  After repair: {report['passed']}/{report['total_checks']} passed")
            if report["all_passed"]:
                print("  ALL CONSTRAINTS NOW PASS")

        # Step 4b: Targeted re-check of suspicious cells (Call 3, optional)
        if not report["all_passed"]:
            suspicious = identify_suspicious_cells(parsed, report["failures"])
            if suspicious:
                print(f"\n  Step 4b: Targeted re-check — Call 3 ({len(suspicious)} suspicious cells)")
                api_calls += 1
                parsed = targeted_recheck(image_path, parsed, suspicious)
                constraints = derive_constraints(parsed, schema=schema,
                                                 persons_independent=persons_independent)
                report = verify_all_constraints(parsed, constraints)
                print(f"  After recheck: {report['passed']}/{report['total_checks']} passed")
                if report["all_passed"]:
                    print("  ALL CONSTRAINTS NOW PASS")

        # Step 4c: Constraint-driven repair cascade
        if not report["all_passed"]:
            print(f"\n  Step 4c: Constraint-driven repair ({report['failed']} failures)")
            parsed, repair_cascade_log, repair_calls = constraint_repair(
                image_path, parsed, schema, constraints, persons_independent)
            api_calls += repair_calls
            constraints = derive_constraints(
                parsed, schema=schema, persons_independent=persons_independent)
            report = verify_all_constraints(parsed, constraints)

            if repair_cascade_log:
                phase_counts = {}
                for entry in repair_cascade_log:
                    p = entry.get("phase", "?")
                    phase_counts[p] = phase_counts.get(p, 0) + 1
                phase_str = ", ".join(f"{p}:{n}" for p, n in sorted(phase_counts.items()))
                print(f"  Repairs applied: {len(repair_cascade_log)} ({phase_str})")
                for entry in repair_cascade_log:
                    if entry.get("action") == "mf_swap":
                        print(f"    [A] SWAP {entry['section']}/{entry['age']} "
                              f"{entry['group']}: M={entry['old_m']}↔F={entry['old_f']}")
                    elif entry.get("action") == "digit_fix":
                        print(f"    [B] DIGIT {entry['section']}/{entry['age']} "
                              f"{entry['group']}.{entry['col']}: "
                              f"{entry['old']}→{entry['new']} "
                              f"({entry['old_digit']}→{entry['new_digit']})")
                    elif entry.get("action") == "add_missing_row":
                        print(f"    [C] ADD ROW {entry['section']}/{entry['age']}")
                    elif entry.get("action") == "re_extract_section":
                        print(f"    [D] RE-EXTRACT {entry['section']} "
                              f"({entry['old_rows']}→{entry['new_rows']} rows)")
                    elif entry.get("action") == "voted_fix":
                        print(f"    [E] VOTE {entry['section']}/{entry['age']} "
                              f"{entry['group']}.{entry['col']}: "
                              f"{entry['old']}→{entry['new']}")
                    elif entry.get("action") == "upscale_reextract":
                        print(f"    [F] UPSCALE {entry['section']}/{entry['group']}: "
                              f"{entry['old_failures']}→{entry['new_failures']} failures")

            print(f"  After repair: {report['passed']}/{report['total_checks']} passed"
                  f" (+{repair_calls} API calls)")
            if report["all_passed"]:
                print("  ALL CONSTRAINTS NOW PASS")

        if not report["all_passed"] and fallback:
            print("\n  Falling back to MoE pipeline...")
            return _fallback_to_moe(image_path)

    # Step 4d: Age ordering sanity check
    age_warnings = validate_age_ordering(parsed)
    if age_warnings:
        print(f"\n  AGE ORDERING WARNINGS ({len(age_warnings)}):")
        for w in age_warnings:
            print(f"    {w}")

    # Step 5: Export
    print("\nStep 5: Export")
    output_paths = {}

    # Tidy CSV
    df = to_tidy_dataframe(parsed)
    csv_path = effective_output_dir / f"{stem}_oneshot.csv"
    df.to_csv(csv_path, index=False)
    output_paths["csv"] = str(csv_path)
    print(f"  CSV: {csv_path} ({len(df)} rows)")

    # Excel (via legacy format adapter)
    all_results, export_schema = to_legacy_format(parsed)
    xlsx_path = effective_output_dir / f"{stem}_oneshot.xlsx"
    export_multigroup_to_excel(all_results, export_schema, xlsx_path)
    output_paths["xlsx"] = str(xlsx_path)

    # JSON with constraint report
    json_path = effective_output_dir / f"{stem}_oneshot.json"
    json_output = {
        "source_image": str(original_image_path),
        "extraction_method": "oneshot_3call",
        "api_calls": api_calls,
        "elapsed_seconds": round(time.time() - t0, 1),
        "metadata": meta,
        "data": parsed["sections"],
        "constraints": {
            "total_checks": report["total_checks"],
            "passed": report["passed"],
            "failed": report["failed"],
            "all_passed": report["all_passed"],
            "failures": report["failures"],
        },
        "age_ordering_warnings": age_warnings,
    }
    with open(json_path, "w") as f:
        json.dump(json_output, f, indent=2)
    output_paths["json"] = str(json_path)
    print(f"  JSON: {json_path}")

    # Clean up temporary rotated image (auto-rotation creates these)
    if image_path != original_image_path:
        try:
            image_path.unlink()
            logger.info("Cleaned up rotated image: %s", image_path.name)
        except OSError:
            pass

    total_elapsed = time.time() - t0
    print(f"\nDone in {total_elapsed:.1f}s "
          f"({api_calls} API calls, "
          f"{report['total_checks']} checks, "
          f"{report['failed']} failures)")

    return {
        "parsed": parsed,
        "schema": schema,
        "constraints": constraints,
        "report": report,
        "df": df,
        "output_paths": output_paths,
        "elapsed": total_elapsed,
        "age_ordering_warnings": age_warnings,
    }


def _fallback_to_moe(image_path):
    """Fall back to the full MoE pipeline."""
    print("\n  --- Fallback to MoE pipeline ---")
    from schema_discovery import extract_table
    result = extract_table(str(image_path), extract_all_groups=True,
                            fast_mode=True)
    if result is None:
        return None
    all_results, schema, passed, failures = result
    return {
        "method": "moe_fallback",
        "all_results": all_results,
        "schema": schema,
        "passed": passed,
        "failures": failures,
    }


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------

def batch_extract(directory, parquet=False):
    """Process all PNG images in a directory.

    Returns list of result dicts.
    """
    directory = Path(directory)
    images = sorted(directory.glob("*.png"))
    if not images:
        # Try recursive
        images = sorted(directory.rglob("*.png"))

    print(f"\nBatch: {len(images)} images in {directory}")
    results = []
    all_dfs = []

    for i, img in enumerate(images):
        print(f"\n[{i+1}/{len(images)}] {img.name}")
        result = extract_and_verify(img)
        if result and "df" in result:
            results.append(result)
            df = result["df"]
            df["source_image"] = img.name
            all_dfs.append(df)

    # Combined exports
    if all_dfs:
        import pandas as pd
        combined = pd.concat(all_dfs, ignore_index=True)

        csv_path = RESULTS_DIR / f"{directory.name}_batch.csv"
        combined.to_csv(csv_path, index=False)
        print(f"\nCombined CSV: {csv_path} ({len(combined)} rows)")

        if parquet:
            pq_path = RESULTS_DIR / f"{directory.name}_batch.parquet"
            combined.to_parquet(pq_path, index=False)
            print(f"Combined Parquet: {pq_path}")

    # Summary
    total = len(images)
    succeeded = len(results)
    all_passed = sum(1 for r in results
                     if r.get("report", {}).get("all_passed", False))
    print(f"\n{'='*70}")
    print(f"Batch Summary: {succeeded}/{total} extracted, "
          f"{all_passed}/{succeeded} all constraints passed")
    print(f"{'='*70}")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Single-shot census table extraction pipeline")
    parser.add_argument("image", nargs="?", help="Path to image file")
    parser.add_argument("--batch", type=str, default=None,
                        help="Process all PNGs in directory")
    parser.add_argument("--parquet", action="store_true",
                        help="Also emit Parquet output")
    parser.add_argument("--fallback", action="store_true",
                        help="Fall back to MoE pipeline on constraint failure")
    parser.add_argument("--retry", action="store_true",
                        help="Retry once on constraint failure")
    args = parser.parse_args()

    if args.batch:
        batch_extract(args.batch, parquet=args.parquet)
    elif args.image:
        result = extract_and_verify(args.image,
                                     retry_on_fail=args.retry,
                                     fallback=args.fallback)
        if result and args.parquet and "df" in result:
            pq_path = RESULTS_DIR / f"{Path(args.image).stem}_oneshot.parquet"
            result["df"].to_parquet(pq_path, index=False)
            print(f"Parquet: {pq_path}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
