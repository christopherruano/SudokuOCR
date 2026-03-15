"""
Auto-discovery pipeline for historical Indian census tables.

Analyzes a raw table image with an LLM to discover its structure (columns,
rows, subtotal hierarchy, data type), then drives the existing MoE pipeline
with zero manual configuration.

Usage:
    from schema_discovery import extract_table
    result = extract_table("age_tables/Cochin/1901/Cochin_age_1891_01.png")

    # Or step by step:
    schema = discover_schema("image.png")
    config = schema_to_config(schema)
    prompt = build_extraction_prompt(schema)
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# TableSchema dataclass
# ---------------------------------------------------------------------------

@dataclass
class TableSchema:
    """Structured representation of a discovered census table."""
    title: str = ""
    region: str = ""
    year: int = 0
    data_type: str = "absolute"  # "absolute" | "proportional"
    denominator: int = 0  # For proportional: 1000, 10000, etc. 0 = absolute.
    column_groups: list = field(default_factory=list)
    # Each: {"name": str, "sub_columns": list[str], "left_frac": float, "right_frac": float}
    target_group_index: int = 0
    row_labels: list = field(default_factory=list)
    subtotal_hierarchy: dict = field(default_factory=dict)
    # e.g. {"Total 0-5": ["0-1","1-2","2-3","3-4","4-5"], ...}
    has_persons_column: bool = True
    known_totals: dict = field(default_factory=dict)
    multi_year: bool = False
    years_present: list = field(default_factory=list)
    multi_region: bool = False
    sections: list = field(default_factory=list)
    # Each: {"name": str, "row_labels": list[str], "subtotal_hierarchy": dict}
    cross_group_constraints: dict = field(default_factory=dict)
    # e.g. {"POPULATION": ["UNMARRIED", "MARRIED", "WIDOWED", "DIVORCED"]}
    # Empty dict if no cross-group relationship exists.
    rotation: int = 0
    # Degrees clockwise to rotate the image so text reads normally (0, 90, 180, 270).


# ---------------------------------------------------------------------------
# Schema discovery prompt
# ---------------------------------------------------------------------------

SCHEMA_DISCOVERY_PROMPT = """You are an expert at analyzing the STRUCTURE of historical census tables. This is a scanned page from an Indian census report.

Your job is to describe the table's layout — NOT to read the actual numbers. Focus on structure only.

Analyze the image and return a JSON object with these fields:

1. "title": The title or header text visible on the page (string).
2. "region": The region/province/state name if visible (string).
3. "year": The census year if visible (integer, e.g. 1901). Use 0 if unclear.
4. "data_type": Either "absolute" (raw population counts, typically large numbers like 169509) or "proportional" (proportions per N, where the total row equals exactly N).
4b. "denominator": If data_type is "proportional", the denominator (e.g. 1000 for "per 1,000", 10000 for "per 10,000"). Read this from the table title. If data_type is "absolute", set to 0.
5. "column_groups": An array of column group objects. Each group represents a major section of the table (e.g. "Population", "Unmarried", "Married", etc., or years like "1891", "1901"). Each object has:
   - "name": The group header text (string)
   - "sub_columns": Array of sub-column names within this group (e.g. ["Persons", "Males", "Females"])
   - "left_frac": Approximate left edge as a fraction of total image width (0.0 to 1.0)
   - "right_frac": Approximate right edge as a fraction of total image width (0.0 to 1.0)
6. "target_group_index": Which column_group (0-indexed) contains the PRIMARY population data to extract. Usually the first group with Persons/Males/Females columns, or the group matching the census year.
7. "row_labels": Array of ALL row labels visible in the age column, exactly as printed. Include individual ages (e.g. "0-1", "1-2"), summary groups (e.g. "0-5", "5-10"), subtotal rows (e.g. "Total 0-5", "Total 0-15"), and the grand "Total" row.
8. "subtotal_hierarchy": Object mapping subtotal row labels to arrays of their component row labels. Build the COMPLETE hierarchy:
   - Individual subtotals: e.g. {"Total 0-5": ["0-1", "1-2", "2-3", "3-4", "4-5"]}
   - Intermediate subtotals: e.g. {"Total 0-15": ["Total 0-5", "5-10", "10-15"]}
   - Grand total: e.g. {"Total": ["Total 0-15", "Total 15-40", "Total 40-60", "60 and over"]}
   IMPORTANT: If there is a grand "Total" row AND intermediate subtotal rows, the Total should map to its direct children (the intermediate subtotals + any ungrouped rows like "60 and over"), NOT to all individual rows.
   Only include subtotals that are explicitly visible as rows in the table. If there are no subtotal rows, return an empty object {}.
9. "has_persons_column": Boolean — true if there is an explicit "Persons" (or "Total"/"Both sexes") column, false if only Males and Females columns are present.
10. "multi_year": Boolean — true if the table shows data for multiple census years side by side.
11. "years_present": Array of census years visible in the table (e.g. [1891, 1901]).
12. "multi_region": Boolean — true if the table shows data for multiple regions/districts.
13. "sections": Array of section objects. Many census tables have multiple numbered sections on the same page, each with its own set of age rows. For example:
   - "1. ALL COMMUNITIES" with age rows 0-1, 1-2, ..., Total
   - "2. Brahmanic Hindus" with its own age rows 0-1, 1-2, ..., Total
   - "3. Other Hindus" with its own age rows
   Each section object has:
   - "name": The section header text (e.g. "All Communities", "Brahmanic Hindus")
   - "row_labels": Array of row labels specific to this section (same format as top-level row_labels)
   - "subtotal_hierarchy": Subtotal hierarchy for this section (same format as top-level)
   If the table has only ONE section (no numbered sub-tables), return an empty array [] and use the top-level row_labels and subtotal_hierarchy as before.
   If the table HAS multiple sections, the top-level row_labels and subtotal_hierarchy should describe the FIRST section, and each section should be listed in the sections array.
14. "cross_group_constraints": Object mapping a total column group name to an array of its component group names, IF one column group represents the sum of others. For example, if a table has "Population", "Unmarried", "Married", "Widowed", and "Divorced" column groups where Population = Unmarried + Married + Widowed + Divorced, return: {"Population": ["Unmarried", "Married", "Widowed", "Divorced"]}. Similarly for religion breakdowns: {"Total": ["Hindu", "Muslim", "Christian"]}. Return an empty object {} if no such relationship exists.
15. "rotation": Integer — how many degrees CLOCKWISE the image must be rotated so that the text reads normally left-to-right. Return 0 if the image is already upright. Return 90 if the text currently reads bottom-to-top (image needs 90° clockwise rotation). Return 270 if the text reads top-to-bottom. Return 180 if the image is upside-down. Most scanned pages will be 0.

IMPORTANT:
- For column positions (left_frac, right_frac), estimate carefully by looking at where column headers and data begin/end relative to the full image width.
- The first column group is usually the age labels column — skip that and start with the first DATA column group.
- Be precise about row labels: list them in the exact order they appear top to bottom.
- For subtotal_hierarchy, only include relationships you can verify from the row labels. If "Total 0-5" appears and individual rows "0-1" through "4-5" also appear, include that mapping.

Return ONLY valid JSON. No explanation or commentary."""


# ---------------------------------------------------------------------------
# discover_schema()
# ---------------------------------------------------------------------------

def _discover_single(b64, model_name, caller):
    """Run schema discovery with a single model, with retries.

    Returns (TableSchema, raw_dict) or (None, None) on failure.
    """
    import time as _time

    for attempt in range(3):
        label = f"{model_name}" + (f" (retry {attempt})" if attempt else "")
        print(f"    {label}...", end=" ", flush=True)
        try:
            raw = caller(b64, SCHEMA_DISCOVERY_PROMPT)
            parsed = _parse_schema_json(raw)
            if parsed is not None:
                schema = _dict_to_schema(parsed)
                print("OK")
                return schema, parsed
            print("parse failed")
            return None, None  # Parse failure — don't retry
        except Exception as e:
            print(f"error: {e}")
            if attempt < 2:
                _time.sleep(2 ** attempt)

    return None, None


def discover_schema(image_path, model="gemini"):
    """Multi-model schema discovery with consensus reconciliation.

    Runs all available models in parallel (conceptually), then merges:
      - Structural fields (data_type, hierarchy, row_labels): majority vote
      - Row labels: union of all models (don't miss any)
      - Subtotal hierarchy: union (take all relationships)
      - Crop positions: most conservative (widest) estimate

    Falls back to single-model if only one model is available.

    Args:
        image_path: Path to the census table image.
        model: Preferred model (used as tiebreaker).

    Returns:
        TableSchema instance, or None on failure.
    """
    import os
    from collections import Counter
    from pipeline import encode_image, MODELS

    b64 = encode_image(str(image_path))

    env_keys = {"openai": "OPENAI_API_KEY", "gemini": "GEMINI_API_KEY",
                "claude": "ANTHROPIC_API_KEY"}
    available = [m for m in MODELS if os.environ.get(env_keys[m])]

    if not available:
        logger.error("No models available for schema discovery")
        return None

    # Run all available models
    print(f"  Schema discovery: querying {len(available)} models...")
    model_schemas = {}  # model_name -> TableSchema
    model_dicts = {}    # model_name -> raw dict

    for m in available:
        schema, raw_dict = _discover_single(b64, m, MODELS[m])
        if schema is not None:
            model_schemas[m] = schema
            model_dicts[m] = raw_dict

    if not model_schemas:
        logger.error("Schema discovery failed on all models")
        return None

    # If only one model succeeded, use it directly
    if len(model_schemas) == 1:
        only_model = list(model_schemas.keys())[0]
        print(f"  Only {only_model} succeeded — using single-model result")
        return model_schemas[only_model]

    # Merge multiple schemas
    print(f"  Reconciling {len(model_schemas)} model results...")
    merged = _merge_schemas(model_schemas, preferred=model)
    return merged


def _merge_schemas(model_schemas, preferred="gemini"):
    """Merge multiple TableSchema results into a consensus schema.

    Strategy:
      - Scalar fields: majority vote, tiebreak with preferred model
      - row_labels: union (ordered by first appearance across models)
      - subtotal_hierarchy: union of all relationships
      - column_groups: use preferred model's groups, but adjust crop to
        be most conservative (widest right_frac)
    """
    from collections import Counter
    from pipeline import normalize_age

    models = list(model_schemas.keys())
    schemas = list(model_schemas.values())

    def _majority(values, prefer_idx=0):
        """Majority vote with tiebreaker."""
        counts = Counter(str(v) for v in values)
        top_count = counts.most_common(1)[0][1]
        winners = [v for v in values if counts[str(v)] == top_count]
        if len(set(str(w) for w in winners)) == 1:
            return winners[0]
        # Tiebreak: prefer the preferred model's value
        return values[prefer_idx]

    prefer_idx = models.index(preferred) if preferred in models else 0

    merged = TableSchema()

    # --- Title: prefer longer (more informative) ---
    titles = [s.title for s in schemas if s.title]
    merged.title = max(titles, key=len) if titles else ""

    # --- Region: majority vote ---
    regions = [s.region for s in schemas if s.region]
    if regions:
        merged.region = _majority(regions, prefer_idx)

    # --- Year: majority vote (ignore 0) ---
    years = [s.year for s in schemas if s.year > 0]
    if years:
        merged.year = _majority(years, prefer_idx)

    # --- Data type: majority vote ---
    merged.data_type = _majority([s.data_type for s in schemas], prefer_idx)

    # --- Denominator: majority vote (ignore 0) ---
    denoms = [s.denominator for s in schemas if s.denominator > 0]
    if denoms:
        merged.denominator = _majority(denoms, prefer_idx)
    elif merged.data_type == "proportional":
        merged.denominator = 1000

    # --- Has persons column: majority vote ---
    merged.has_persons_column = _majority(
        [s.has_persons_column for s in schemas], prefer_idx)

    # --- Multi-year: majority vote ---
    merged.multi_year = _majority([s.multi_year for s in schemas], prefer_idx)

    # --- Years present: union ---
    all_years = set()
    for s in schemas:
        all_years.update(s.years_present)
    merged.years_present = sorted(all_years)

    # --- Multi-region: majority vote ---
    merged.multi_region = _majority([s.multi_region for s in schemas], prefer_idx)

    # --- Preferred model schema (used as base for several fields) ---
    pref_schema = model_schemas.get(preferred, schemas[0])

    # --- Sections: use preferred model's sections (structural choice) ---
    pref_sections = pref_schema.sections
    if pref_sections:
        merged.sections = pref_sections
    else:
        # Check if any model found sections
        for s in schemas:
            if s.sections:
                merged.sections = s.sections
                break

    # --- Row labels: union, preserving order from first appearance ---
    seen = set()
    ordered_labels = []
    # Process preferred model first to establish base ordering
    for label in pref_schema.row_labels:
        norm = normalize_age(label)
        if norm and norm not in seen:
            seen.add(norm)
            ordered_labels.append(label)
    # Then add any labels found only by other models
    for s in schemas:
        for label in s.row_labels:
            norm = normalize_age(label)
            if norm and norm not in seen:
                seen.add(norm)
                ordered_labels.append(label)
    merged.row_labels = ordered_labels

    # --- Subtotal hierarchy: union of all relationships ---
    # Use _clean_age_key to dedup "Total" vs "Total ..." → both become "total"
    merged_hier = {}
    for s in schemas:
        for sub_label, comp_labels in s.subtotal_hierarchy.items():
            norm_sub = _clean_age_key(normalize_age(sub_label)) or normalize_age(sub_label)
            if norm_sub not in merged_hier:
                merged_hier[norm_sub] = (sub_label, comp_labels)
            else:
                # Take the version with more components (more detailed)
                existing_comps = merged_hier[norm_sub][1]
                if len(comp_labels) > len(existing_comps):
                    merged_hier[norm_sub] = (sub_label, comp_labels)
    # Reconstruct with original labels — prefer clean labels
    merged.subtotal_hierarchy = {}
    for norm_key, (label, comps) in merged_hier.items():
        # Clean up "Total ..." → "Total" in the label
        if _re.match(r'(?i)^total\s*[.\-…]+$', label.strip()):
            label = "Total"
        merged.subtotal_hierarchy[label] = comps

    # --- Cross-group constraints: majority vote on presence, union of keys ---
    # Take the version with most models agreeing
    cgc_candidates = [s.cross_group_constraints for s in schemas
                      if s.cross_group_constraints]
    if cgc_candidates:
        # Pick the constraint dict that appears most often (by key set)
        from collections import Counter as _Counter
        key_sets = _Counter(
            frozenset(c.keys()) for c in cgc_candidates)
        best_keys = key_sets.most_common(1)[0][0]
        for c in cgc_candidates:
            if frozenset(c.keys()) == best_keys:
                merged.cross_group_constraints = c
                break
    else:
        merged.cross_group_constraints = {}

    # --- Column groups: use preferred model's structure, but adjust crop ---
    # Take preferred model's column_groups as the base
    pref_cg = pref_schema.column_groups
    merged.column_groups = pref_cg
    merged.target_group_index = pref_schema.target_group_index

    # Find the most conservative (widest) right_frac across all models'
    # target groups, so we don't accidentally crop out relevant data
    target_rights = []
    for s in schemas:
        if s.column_groups and 0 <= s.target_group_index < len(s.column_groups):
            tgt = s.column_groups[s.target_group_index]
            target_rights.append(tgt.get("right_frac", 1.0))
    if target_rights and merged.column_groups:
        widest_right = max(target_rights)
        idx = merged.target_group_index
        if 0 <= idx < len(merged.column_groups):
            merged.column_groups[idx]["right_frac"] = widest_right

    # Log the merge
    agree_count = 0
    total_fields = 0
    for field in ["data_type", "denominator", "has_persons_column",
                   "multi_year", "multi_region"]:
        total_fields += 1
        vals = set(str(getattr(s, field)) for s in schemas)
        if len(vals) == 1:
            agree_count += 1
    print(f"    Structural consensus: {agree_count}/{total_fields} fields unanimous")
    print(f"    Row labels: {len(merged.row_labels)} (union of all models)")
    print(f"    Hierarchy groups: {len(merged.subtotal_hierarchy)}")
    if target_rights:
        print(f"    Crop right_frac: {max(target_rights):.2f} "
              f"(max of {[round(r, 2) for r in target_rights]})")

    return merged


def _parse_schema_json(text):
    """Extract and parse a JSON object from LLM response text."""
    import re
    text = text.strip()
    # Strip markdown code fences
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)

    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to find a JSON object
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass

    return None


def _dict_to_schema(d):
    """Convert a parsed JSON dict to a TableSchema."""
    schema = TableSchema()
    schema.title = d.get("title", "")
    schema.region = d.get("region", "")
    schema.year = d.get("year", 0)
    raw_dtype = d.get("data_type", "absolute")
    # Normalize: accept "per_mille" as a legacy alias for "proportional"
    schema.data_type = "proportional" if raw_dtype in ("per_mille", "proportional") else "absolute"
    schema.denominator = d.get("denominator", 0)
    # Infer denominator from data_type if not explicitly set
    if schema.data_type == "proportional" and schema.denominator == 0:
        schema.denominator = 1000  # default assumption
    schema.column_groups = d.get("column_groups", [])
    schema.target_group_index = d.get("target_group_index", 0)
    schema.row_labels = d.get("row_labels", [])
    schema.subtotal_hierarchy = d.get("subtotal_hierarchy", {})
    schema.has_persons_column = d.get("has_persons_column", True)
    schema.known_totals = d.get("known_totals", {})
    schema.multi_year = d.get("multi_year", False)
    schema.years_present = d.get("years_present", [])
    schema.multi_region = d.get("multi_region", False)
    schema.sections = d.get("sections", [])
    schema.cross_group_constraints = d.get("cross_group_constraints", {})
    rot = d.get("rotation", 0)
    schema.rotation = rot if rot in (0, 90, 180, 270) else 0
    return schema


# ---------------------------------------------------------------------------
# schema_to_config()
# ---------------------------------------------------------------------------

import re as _re

def _clean_age_key(normalized):
    """Fix degenerate age keys from LLM output.

    After normalize_age(), labels like "Total ..." become "...".
    This function detects and fixes such cases:
      - "..." or other punctuation-only → "total"
      - Empty string → None (will be filtered out)
    """
    if not normalized:
        return None
    # If it's only punctuation/dots, treat as grand total
    if _re.fullmatch(r'[.\-_…]+', normalized):
        return "total"
    return normalized


def schema_to_config(schema):
    """Translate a TableSchema into pipeline configuration parameters.

    Returns a dict with keys matching strategy_moe_pipeline() params:
        - preprocessing: dict with crop config
        - prompt_type: "full" or "subtotals"
        - constraint_groups: dict or None
        - known_totals: dict or None
        - custom_prompt: str (tailored extraction prompt)
    """
    from pipeline import normalize_age

    config = {}

    # --- Preprocessing: derive crop from column positions ---
    crop = _derive_crop(schema)
    config["preprocessing"] = {"crop": crop} if crop else {"crop": None}

    # --- Prompt type ---
    has_subtotals = bool(schema.subtotal_hierarchy)
    config["prompt_type"] = "subtotals" if has_subtotals else "full"

    # --- Constraint groups ---
    if schema.subtotal_hierarchy:
        constraint_groups = {}
        for subtotal_label, component_labels in schema.subtotal_hierarchy.items():
            norm_key = _clean_age_key(normalize_age(subtotal_label))
            norm_components = [_clean_age_key(normalize_age(c))
                               for c in component_labels]
            if norm_key:  # Skip entries that normalize to empty/invalid
                constraint_groups[norm_key] = [c for c in norm_components if c]
        config["constraint_groups"] = constraint_groups if constraint_groups else None
    else:
        config["constraint_groups"] = None

    # --- Known totals ---
    if schema.data_type == "proportional" and schema.denominator > 0:
        denom = schema.denominator
        config["known_totals"] = {
            "total": {"males": denom, "females": denom},
        }
        # Also set persons total if has_persons_column
        if schema.has_persons_column:
            config["known_totals"]["total"]["persons"] = denom
    else:
        config["known_totals"] = None

    # --- Custom extraction prompt ---
    config["custom_prompt"] = build_extraction_prompt(schema)

    return config


def _derive_crop_for_group(schema, group_index):
    """Derive crop fractions for a specific column group.

    For the first group (index 0): only crops the right side, keeping
    age labels visible. Same as original behavior.

    For subsequent groups (index > 0): returns composite crop info
    (left_frac, right_frac, label_right_frac) so the caller can build
    a composite image: [age labels] | [target P/M/F columns].
    This eliminates column confusion by making adjacent groups invisible.

    Args:
        schema: TableSchema instance.
        group_index: Index into schema.column_groups.

    Returns a dict like {"right_frac": 0.33} or
    {"right_frac": 0.82, "left_frac": 0.60, "label_right_frac": 0.15}
    or None.
    """
    if not schema.column_groups:
        return None

    if group_index < 0 or group_index >= len(schema.column_groups):
        return None

    target = schema.column_groups[group_index]
    right = target.get("right_frac", 1.0)
    left = target.get("left_frac", 0.0)

    crop = {}

    # Composite crop for non-first groups: provide left_frac and
    # label_right_frac so the caller can build a composite image.
    # This eliminates column confusion by hiding adjacent groups.
    is_composite = group_index > 0 and left > 0.10

    # Right crop: tighter margin for composite crops (3%) to avoid
    # bleeding into adjacent group; wider margin (10%) for simple crops
    right_margin = 0.03 if is_composite else 0.10
    right_with_margin = min(1.0, right + right_margin)
    if right_with_margin < 0.98:
        crop["right_frac"] = right_with_margin

    if is_composite:
        # Left edge of target group (with 2% margin — tight to avoid
        # bleeding in adjacent column data; 5% was too generous)
        crop["left_frac"] = max(0.0, left - 0.02)

        # Right edge of the age label column (≈ first data group's left_frac)
        # Must NOT extend into data columns — even a sliver of POPULATION
        # numbers causes the model to read them instead of the target group
        first_group = schema.column_groups[0]
        label_right = first_group.get("left_frac", 0.15)
        crop["label_right_frac"] = label_right

        # Ensure we have a right_frac even for the rightmost group
        if "right_frac" not in crop:
            crop["right_frac"] = 1.0

    return crop if crop else None


def _derive_crop(schema):
    """Derive crop fractions from column group positions (target group).

    Convenience wrapper around _derive_crop_for_group using the schema's
    target_group_index.
    """
    return _derive_crop_for_group(schema, schema.target_group_index)


# ---------------------------------------------------------------------------
# build_extraction_prompt()
# ---------------------------------------------------------------------------

def build_extraction_prompt(schema):
    """Generate a custom extraction prompt tailored to the discovered table structure.

    Better than generic PROMPT_FULL because it tells the model exactly
    what rows and columns to expect.
    """
    has_subtotals = bool(schema.subtotal_hierarchy)
    has_persons = schema.has_persons_column

    # Build column instruction
    if has_persons:
        col_instruction = "Extract the Persons, Males, and Females columns."
    else:
        col_instruction = (
            "This table has Males and Females columns but no Persons column. "
            "Compute Persons = Males + Females for each row."
        )

    # Build row label list — clean up degenerate labels
    if schema.row_labels:
        clean_labels = []
        for label in schema.row_labels:
            # Fix "Total ..." → "Total"
            if _re.match(r'(?i)^total\s*[.\-…]+$', label.strip()):
                clean_labels.append("Total")
            else:
                clean_labels.append(label)
        row_list = "\n".join(f"  - {label}" for label in clean_labels)
        row_instruction = f"The table contains these rows (in order):\n{row_list}"
    else:
        row_instruction = (
            "Extract all age-group rows visible in the table, including "
            "any subtotal rows and the grand Total row."
        )

    # Build subtotal instruction — clean up degenerate labels
    def _clean_label(label):
        if _re.match(r'(?i)^total\s*[.\-…]+$', label.strip()):
            return "Total"
        return label

    subtotal_instruction = ""
    if has_subtotals:
        parts = []
        for sub_label, comp_labels in schema.subtotal_hierarchy.items():
            clean_sub = _clean_label(sub_label)
            clean_comps = [_clean_label(c) for c in comp_labels]
            comp_str = " + ".join(clean_comps)
            parts.append(f"  - {clean_sub} = {comp_str}")
        subtotal_instruction = (
            "\n\nThis table has subtotal rows. The following relationships hold:\n"
            + "\n".join(parts)
            + "\n\nUse these relationships to verify your readings. "
            "If a subtotal doesn't match the sum of its components, "
            "re-examine the ambiguous digits."
        )

    # Build data type instruction
    if schema.data_type == "proportional" and schema.denominator > 0:
        denom = schema.denominator
        data_instruction = (
            f"\n\nIMPORTANT: This table shows proportions per {denom:,}, "
            f"NOT absolute population counts. The Total row should equal exactly {denom:,} "
            f"for both Males and Females. "
            "Use this constraint to verify your readings."
        )
    else:
        data_instruction = ""

    prompt = f"""You are an expert at reading historical census tables. This is a scanned page from an Indian census report.

{col_instruction}

{row_instruction}{subtotal_instruction}{data_instruction}

CRITICAL: Read each digit extremely carefully. Common OCR errors on old typefaces:
- The digit 3 has angular notches/points at its curves; 8 has smooth continuous curves. These look very similar — examine closely.
- Also watch for: 6 vs 5, 0 vs 9, 1 vs 7
- Verify by checking: Persons should equal Males + Females for each row
- If a digit is ambiguous, use the M+F=Persons constraint to determine the correct reading

Return ONLY a JSON array. Each object: {{"age": "...", "persons": N, "males": N, "females": N}}
For subtotal rows, use the label as-is (e.g. "Total 0-5", "Total 15-40", "Total").
All values are integers, no commas. Return ONLY valid JSON."""

    return prompt


def build_extraction_prompt_for_group(schema, group_name, section_name=None,
                                      row_labels=None, subtotal_hierarchy=None):
    """Generate a custom extraction prompt for a specific column group.

    Like build_extraction_prompt() but explicitly names the target column group
    and optionally scopes to a section of the table.

    Args:
        schema: TableSchema instance.
        group_name: Name of the column group (e.g. "Unmarried", "Married").
        section_name: Optional section name (e.g. "All Communities").
        row_labels: Override row labels (for section-specific labels).
        subtotal_hierarchy: Override hierarchy (for section-specific hierarchy).
    """
    has_persons = schema.has_persons_column
    use_row_labels = row_labels if row_labels is not None else schema.row_labels
    use_hierarchy = subtotal_hierarchy if subtotal_hierarchy is not None else schema.subtotal_hierarchy
    has_subtotals = bool(use_hierarchy)

    # Build column instruction
    if has_persons:
        col_instruction = (
            f"Extract the **{group_name}** Persons, Males, and Females columns.\n"
            f"IMPORTANT: Read from the {group_name} column group ONLY, not from other groups."
        )
    else:
        col_instruction = (
            f"This table has Males and Females columns but no Persons column "
            f"in the {group_name} group. "
            "Compute Persons = Males + Females for each row."
        )

    # Section instruction
    section_instruction = ""
    if section_name:
        section_instruction = (
            f"\n\nFocus ONLY on the section labeled \"{section_name}\". "
            f"Ignore rows from other sections of the table."
        )

    # Build row label list
    if use_row_labels:
        clean_labels = []
        for label in use_row_labels:
            if _re.match(r'(?i)^total\s*[.\-…]+$', label.strip()):
                clean_labels.append("Total")
            else:
                clean_labels.append(label)
        row_list = "\n".join(f"  - {label}" for label in clean_labels)
        row_instruction = f"The table contains these rows (in order):\n{row_list}"
    else:
        row_instruction = (
            "Extract all age-group rows visible in the table, including "
            "any subtotal rows and the grand Total row."
        )

    # Build subtotal instruction
    def _clean_label(label):
        if _re.match(r'(?i)^total\s*[.\-…]+$', label.strip()):
            return "Total"
        return label

    subtotal_instruction = ""
    if has_subtotals:
        parts = []
        for sub_label, comp_labels in use_hierarchy.items():
            clean_sub = _clean_label(sub_label)
            clean_comps = [_clean_label(c) for c in comp_labels]
            comp_str = " + ".join(clean_comps)
            parts.append(f"  - {clean_sub} = {comp_str}")
        subtotal_instruction = (
            "\n\nThis table has subtotal rows. The following relationships hold:\n"
            + "\n".join(parts)
            + "\n\nUse these relationships to verify your readings. "
            "If a subtotal doesn't match the sum of its components, "
            "re-examine the ambiguous digits."
        )

    # Build data type instruction
    if schema.data_type == "proportional" and schema.denominator > 0:
        denom = schema.denominator
        data_instruction = (
            f"\n\nIMPORTANT: This table shows proportions per {denom:,}, "
            f"NOT absolute population counts. The Total row should equal exactly {denom:,} "
            f"for both Males and Females. "
            "Use this constraint to verify your readings."
        )
    else:
        data_instruction = ""

    prompt = f"""You are an expert at reading historical census tables. This is a scanned page from an Indian census report.

{col_instruction}{section_instruction}

{row_instruction}{subtotal_instruction}{data_instruction}

CRITICAL: Read each digit extremely carefully. Common OCR errors on old typefaces:
- The digit 3 has angular notches/points at its curves; 8 has smooth continuous curves. These look very similar — examine closely.
- Also watch for: 6 vs 5, 0 vs 9, 1 vs 7
- Verify by checking: Persons should equal Males + Females for each row
- If a digit is ambiguous, use the M+F=Persons constraint to determine the correct reading

Return ONLY a JSON array. Each object: {{"age": "...", "persons": N, "males": N, "females": N}}
For subtotal rows, use the label as-is (e.g. "Total 0-5", "Total 15-40", "Total").
All values are integers, no commas. Return ONLY valid JSON."""

    return prompt


# ---------------------------------------------------------------------------
# verify_constraints()
# ---------------------------------------------------------------------------

def verify_constraints(rows, config):
    """Post-extraction check that all constraints pass.

    Checks:
    1. P = M + F for every row
    2. Group sum constraints (subtotal = sum of components)
    3. Known totals match

    Args:
        rows: List of row dicts from the pipeline.
        config: Dict from schema_to_config().

    Returns:
        (passed: bool, failures: list of str)
    """
    from pipeline import normalize_age

    failures = []

    if not rows:
        return False, ["No rows to verify"]

    # Build age -> row lookup
    row_by_age = {}
    for row in rows:
        age_norm = normalize_age(row.get("age", ""))
        row_by_age[age_norm] = row

    # Check 1: P = M + F
    for row in rows:
        p = row.get("persons")
        m = row.get("males")
        f = row.get("females")
        if p is not None and m is not None and f is not None:
            if p != m + f:
                failures.append(
                    f"{row.get('age')}: P={p} != M={m}+F={f}={m+f}"
                )

    # Check 2: Group sum constraints
    constraint_groups = config.get("constraint_groups") or {}
    for subtotal_key, comp_keys in constraint_groups.items():
        sub_row = row_by_age.get(subtotal_key)
        if sub_row is None:
            failures.append(f"Subtotal row '{subtotal_key}' not found in output")
            continue
        for col in ("persons", "males", "females"):
            sub_val = sub_row.get(col)
            if sub_val is None:
                continue
            comp_sum = 0
            all_present = True
            for ck in comp_keys:
                ck_row = row_by_age.get(ck)
                if ck_row is None or ck_row.get(col) is None:
                    all_present = False
                    break
                comp_sum += ck_row[col]
            if all_present and sub_val != comp_sum:
                failures.append(
                    f"{subtotal_key}.{col}: subtotal={sub_val} != "
                    f"sum({comp_keys})={comp_sum}"
                )

    # Check 3: Known totals
    known_totals = config.get("known_totals") or {}
    for age_key, col_vals in known_totals.items():
        kt_row = row_by_age.get(age_key)
        if kt_row is None:
            failures.append(f"Known total row '{age_key}' not found in output")
            continue
        for col, expected in col_vals.items():
            actual = kt_row.get(col)
            if actual is not None and actual != expected:
                failures.append(
                    f"Known total {age_key}.{col}: expected={expected}, "
                    f"got={actual}"
                )

    passed = len(failures) == 0
    return passed, failures


# ---------------------------------------------------------------------------
# extract_table() — end-to-end orchestrator
# ---------------------------------------------------------------------------

def _config_for_group(schema, group_index):
    """Build pipeline config for a specific column group.

    Like schema_to_config() but targets a specific group index.
    """
    from pipeline import normalize_age

    config = {}

    crop = _derive_crop_for_group(schema, group_index)
    config["preprocessing"] = {"crop": crop} if crop else {"crop": None}

    has_subtotals = bool(schema.subtotal_hierarchy)
    config["prompt_type"] = "subtotals" if has_subtotals else "full"

    if schema.subtotal_hierarchy:
        constraint_groups = {}
        for subtotal_label, component_labels in schema.subtotal_hierarchy.items():
            norm_key = _clean_age_key(normalize_age(subtotal_label))
            norm_components = [_clean_age_key(normalize_age(c))
                               for c in component_labels]
            if norm_key:
                constraint_groups[norm_key] = [c for c in norm_components if c]
        config["constraint_groups"] = constraint_groups if constraint_groups else None
    else:
        config["constraint_groups"] = None

    if schema.data_type == "proportional" and schema.denominator > 0:
        denom = schema.denominator
        config["known_totals"] = {
            "total": {"males": denom, "females": denom},
        }
        if schema.has_persons_column:
            config["known_totals"]["total"]["persons"] = denom
    else:
        config["known_totals"] = None

    group_name = ""
    if schema.column_groups and 0 <= group_index < len(schema.column_groups):
        group_name = schema.column_groups[group_index].get("name", "")

    config["custom_prompt"] = build_extraction_prompt_for_group(
        schema, group_name or "Population")

    return config


def _config_for_group_section(schema, group_index, section):
    """Build pipeline config for a specific column group + section."""
    from pipeline import normalize_age

    config = {}

    crop = _derive_crop_for_group(schema, group_index)
    config["preprocessing"] = {"crop": crop} if crop else {"crop": None}

    sec_hierarchy = section.get("subtotal_hierarchy", {})
    has_subtotals = bool(sec_hierarchy)
    config["prompt_type"] = "subtotals" if has_subtotals else "full"

    if sec_hierarchy:
        constraint_groups = {}
        for subtotal_label, component_labels in sec_hierarchy.items():
            norm_key = _clean_age_key(normalize_age(subtotal_label))
            norm_components = [_clean_age_key(normalize_age(c))
                               for c in component_labels]
            if norm_key:
                constraint_groups[norm_key] = [c for c in norm_components if c]
        config["constraint_groups"] = constraint_groups if constraint_groups else None
    else:
        config["constraint_groups"] = None

    if schema.data_type == "proportional" and schema.denominator > 0:
        denom = schema.denominator
        config["known_totals"] = {
            "total": {"males": denom, "females": denom},
        }
        if schema.has_persons_column:
            config["known_totals"]["total"]["persons"] = denom
    else:
        config["known_totals"] = None

    group_name = ""
    if schema.column_groups and 0 <= group_index < len(schema.column_groups):
        group_name = schema.column_groups[group_index].get("name", "")

    config["custom_prompt"] = build_extraction_prompt_for_group(
        schema, group_name or "Population",
        section_name=section.get("name"),
        row_labels=section.get("row_labels"),
        subtotal_hierarchy=sec_hierarchy)

    return config


def extract_table(image_path, output_path=None, discovery_model="gemini",
                  extract_all_groups=False, fast_mode=False):
    """End-to-end auto-discovery extraction pipeline.

    1. discover_schema() — LLM analyzes table structure
    2. schema_to_config() — translate to pipeline params
    3. build_extraction_prompt() — tailored prompt
    4. strategy_moe_pipeline() — variants x models
    5. verify_constraints() — check all constraints pass
    6. export_results_to_excel() — Excel with constraint formulas

    Args:
        image_path: Path to the census table image.
        output_path: Optional output Excel path.
        discovery_model: Model for schema discovery (default "gemini").
        extract_all_groups: If True, extract all column groups (Population,
            Unmarried, Married, etc.) and all sections. If False, extract
            only the target (Population) group.
        fast_mode: If True, use 2 variants x 2 models instead of 4x3.
            ~3x faster, ~60% cheaper.

    Returns:
        When extract_all_groups=False:
            (rows, schema, config, passed, failures) tuple.
        When extract_all_groups=True:
            (all_results, schema, passed, failures) tuple where all_results
            is a dict: {section_name: {group_name: {"rows": [...], "config": {...}}}}
    """
    import time
    from pipeline import (strategy_moe_pipeline, export_results_to_excel,
                          RESULTS_DIR)

    image_path = Path(image_path)
    if not image_path.exists():
        print(f"ERROR: Image not found: {image_path}")
        return None, None, None, False, ["Image not found"]

    print(f"\n{'='*70}")
    print(f"Auto-Discovery Pipeline: {image_path.name}")
    print(f"{'='*70}")

    # Step 1: Discover schema
    print("\nStep 1: Schema Discovery")
    t0 = time.time()
    schema = discover_schema(image_path, model=discovery_model)
    elapsed = time.time() - t0

    if schema is None:
        print("  FAILED: Could not discover table schema")
        if extract_all_groups:
            return None, None, False, ["Schema discovery failed"]
        return None, None, None, False, ["Schema discovery failed"]

    print(f"  Discovered in {elapsed:.1f}s:")
    print(f"    Title: {schema.title}")
    print(f"    Region: {schema.region}, Year: {schema.year}")
    print(f"    Data type: {schema.data_type}")
    print(f"    Column groups: {len(schema.column_groups)}")
    for i, cg in enumerate(schema.column_groups):
        marker = " <-- TARGET" if i == schema.target_group_index else ""
        print(f"      [{i}] {cg.get('name', '?')}: "
              f"{cg.get('sub_columns', [])} "
              f"({cg.get('left_frac', '?')}-{cg.get('right_frac', '?')}){marker}")
    print(f"    Row labels: {len(schema.row_labels)} rows")
    print(f"    Subtotal hierarchy: {len(schema.subtotal_hierarchy)} groups")
    if schema.subtotal_hierarchy:
        for k, v in schema.subtotal_hierarchy.items():
            print(f"      {k} = sum({v})")
    print(f"    Has persons column: {schema.has_persons_column}")
    print(f"    Multi-year: {schema.multi_year}, Years: {schema.years_present}")
    if schema.sections:
        print(f"    Sections: {len(schema.sections)}")
        for sec in schema.sections:
            print(f"      - {sec.get('name', '?')} ({len(sec.get('row_labels', []))} rows)")
    if schema.cross_group_constraints:
        print(f"    Cross-group constraints:")
        for total_g, comp_gs in schema.cross_group_constraints.items():
            print(f"      {total_g} = {' + '.join(comp_gs)}")
    if fast_mode:
        print(f"    Mode: FAST (2 variants × 2 models)")

    variant_mode = "fast" if fast_mode else "full"

    # ── All-groups extraction path ────────────────────────────────────
    if extract_all_groups:
        return _extract_all_groups(image_path, schema, output_path,
                                   strategy_moe_pipeline, RESULTS_DIR,
                                   variant_mode=variant_mode)

    # ── Single-group extraction path (original behavior) ─────────────
    # Step 2: Translate to config
    print("\nStep 2: Schema → Config")
    config = schema_to_config(schema)
    print(f"    Preprocessing: {config['preprocessing']}")
    print(f"    Prompt type: {config['prompt_type']}")
    if config['constraint_groups']:
        print(f"    Constraint groups: {list(config['constraint_groups'].keys())}")
    if config['known_totals']:
        print(f"    Known totals: {config['known_totals']}")

    # Step 3: Run MoE pipeline
    print("\nStep 3: MoE Pipeline")
    t0 = time.time()
    resolved_rows, resolution_log, _ = strategy_moe_pipeline(
        str(image_path),
        preprocessing_config=config["preprocessing"],
        prompt_type=config["prompt_type"],
        constraint_groups=config["constraint_groups"],
        known_totals=config["known_totals"],
        custom_prompt=config["custom_prompt"],
        variant_mode=variant_mode,
    )
    elapsed = time.time() - t0
    print(f"  Pipeline completed in {elapsed:.1f}s")

    if resolved_rows is None:
        print("  FAILED: Pipeline produced no results")
        return None, schema, config, False, ["Pipeline produced no results"]

    print(f"  Extracted {len(resolved_rows)} rows")

    # Step 4: Verify constraints
    print("\nStep 4: Constraint Verification")
    passed, failures = verify_constraints(resolved_rows, config)
    if passed:
        print("  ALL CONSTRAINTS PASSED")
    else:
        print(f"  FAILURES ({len(failures)}):")
        for f in failures:
            print(f"    - {f}")

    # Step 5: Export results
    print("\nStep 5: Export")
    test_id = image_path.stem

    # Build a name for the sheet
    sheet_name = schema.region or schema.title or image_path.stem
    if schema.year:
        sheet_name = f"{sheet_name} {schema.year}"

    test_results = {
        test_id: {
            "name": sheet_name,
            "rows": resolved_rows,
            "constraint_groups": config["constraint_groups"],
            "known_totals": config["known_totals"],
        }
    }

    if output_path is None:
        output_path = RESULTS_DIR / f"{test_id}_extracted.xlsx"
    output_path = Path(output_path)

    export_results_to_excel(test_results, output_path=output_path)

    # Save structured JSON alongside the Excel
    json_path = output_path.with_suffix(".json")
    clean_rows = []
    for row in resolved_rows:
        clean = {k: v for k, v in row.items() if not k.startswith("_")}
        clean_rows.append(clean)

    json_output = {
        "source_image": str(image_path),
        "discovery": {
            "title": schema.title,
            "region": schema.region,
            "year": schema.year,
            "data_type": schema.data_type,
            "denominator": schema.denominator,
            "has_persons_column": schema.has_persons_column,
            "multi_year": schema.multi_year,
            "years_present": schema.years_present,
            "target_columns": (schema.column_groups[schema.target_group_index]
                               if schema.column_groups else {}),
            "subtotal_hierarchy": schema.subtotal_hierarchy,
            "row_labels": schema.row_labels,
        },
        "pipeline_config": {k: v for k, v in config.items()
                            if k != "custom_prompt"},
        "results": {
            "rows": clean_rows,
            "n_rows": len(clean_rows),
            "constraints_passed": passed,
            "constraint_failures": failures,
        },
    }
    with open(json_path, "w") as f:
        json.dump(json_output, f, indent=2)
    print(f"  JSON: {json_path}")

    return resolved_rows, schema, config, passed, failures


def _extract_all_groups(image_path, schema, output_path,
                        strategy_moe_pipeline, RESULTS_DIR,
                        variant_mode="full"):
    """Extract all column groups (and sections) from a table image.

    Loops over sections × column groups, running the MoE pipeline for each.
    When cross_group_constraints exist, runs cross-group reconciliation
    after each section completes.

    Args:
        variant_mode: "full" (4 variants × 3 models) or "fast" (2 × 2).

    Returns:
        (all_results, schema, passed, failures) tuple.
    """
    import time
    from ensemble import cross_group_reconcile

    # Determine sections to process
    if schema.sections:
        sections = schema.sections
    else:
        # Single section — use top-level row_labels / hierarchy
        sections = [{
            "name": "All",
            "row_labels": schema.row_labels,
            "subtotal_hierarchy": schema.subtotal_hierarchy,
        }]

    # Determine column groups to extract (all of them)
    groups = schema.column_groups
    if not groups:
        print("  ERROR: No column groups discovered")
        return None, schema, False, ["No column groups discovered"]

    has_cross_group = bool(schema.cross_group_constraints)
    all_results = {}  # section_name -> {group_name -> {"rows": [...], "config": {...}}}
    all_raw_readings = {}  # section_name -> {group_name -> list[list[row_dicts]]}
    all_failures = []
    total_groups_extracted = 0

    for sec in sections:
        sec_name = sec.get("name", "All")
        print(f"\n{'─'*70}")
        print(f"Section: {sec_name}")
        print(f"{'─'*70}")

        all_results[sec_name] = {}
        all_raw_readings[sec_name] = {}

        for gi, cg in enumerate(groups):
            group_name = cg.get("name", f"Group_{gi}")
            print(f"\n  Column Group [{gi}]: {group_name}")

            # Build config for this group + section
            if len(sections) > 1 or schema.sections:
                config = _config_for_group_section(schema, gi, sec)
            else:
                config = _config_for_group(schema, gi)

            print(f"    Crop: {config['preprocessing'].get('crop')}")

            # Run MoE pipeline
            t0 = time.time()
            try:
                result = strategy_moe_pipeline(
                    str(image_path),
                    preprocessing_config=config["preprocessing"],
                    prompt_type=config["prompt_type"],
                    constraint_groups=config["constraint_groups"],
                    known_totals=config["known_totals"],
                    custom_prompt=config["custom_prompt"],
                    variant_mode=variant_mode,
                    return_raw_readings=has_cross_group,
                )
                if has_cross_group:
                    resolved_rows, resolution_log, _, raw_readings = result
                else:
                    resolved_rows, resolution_log, _ = result
                    raw_readings = None
                elapsed = time.time() - t0
            except Exception as e:
                print(f"    ERROR: {e}")
                all_failures.append(f"{sec_name}/{group_name}: {e}")
                continue

            if resolved_rows is None:
                print(f"    FAILED: No results ({time.time()-t0:.1f}s)")
                all_failures.append(f"{sec_name}/{group_name}: No results")
                continue

            print(f"    Extracted {len(resolved_rows)} rows ({elapsed:.1f}s)")

            # Verify constraints
            passed, failures = verify_constraints(resolved_rows, config)
            if passed:
                print(f"    Constraints: PASS")
            else:
                print(f"    Constraints: {len(failures)} failures")
                for fl in failures:
                    print(f"      - {fl}")
                all_failures.extend(
                    f"{sec_name}/{group_name}: {fl}" for fl in failures)

            all_results[sec_name][group_name] = {
                "rows": resolved_rows,
                "config": {k: v for k, v in config.items()
                           if k != "custom_prompt"},
                "constraints_passed": passed,
                "constraint_failures": failures,
            }
            if raw_readings is not None:
                all_raw_readings[sec_name][group_name] = raw_readings
            total_groups_extracted += 1

        # ── Cross-group reconciliation for this section ──────────────
        if has_cross_group and all_results[sec_name]:
            print(f"\n  Cross-group reconciliation: {sec_name}")
            # Build group_results: {group_name: [row_dicts]}
            sec_group_results = {
                gname: gdata["rows"]
                for gname, gdata in all_results[sec_name].items()
            }

            # Detect column confusion before reconciliation
            from ensemble import detect_column_confusion
            confused_groups = detect_column_confusion(
                sec_group_results, schema.cross_group_constraints)
            if confused_groups:
                print(f"    Column confusion detected: {confused_groups}")
                print(f"    Excluding confused groups from reconciliation")
                # Remove confused groups so reconciliation doesn't
                # "fix" their garbage values or trust them
                for cg in confused_groups:
                    sec_group_results.pop(cg, None)

            # Build raw readings for candidate lookup
            sec_raw = all_raw_readings.get(sec_name, {})
            sec_raw_readings = sec_raw if sec_raw else None

            corrected_groups, reconcile_log = cross_group_reconcile(
                sec_group_results,
                schema.cross_group_constraints,
                all_raw_readings=sec_raw_readings,
            )

            # Report reconciliation results
            summary = [e for e in reconcile_log if e.get("phase") == "summary"]
            corrections = [e for e in reconcile_log if "new" in e]
            if summary:
                s = summary[-1]
                initial = s.get("initial_discrepancies", 0)
                remaining = s.get("remaining_discrepancies", 0)
                n_fixes = s.get("corrections", 0)
                print(f"    Initial discrepancies: {initial}")
                print(f"    Corrections made: {n_fixes}")
                print(f"    Remaining: {remaining}")
            for c in corrections:
                print(f"    FIX {c.get('age', '?')}.{c.get('col', '?')} "
                      f"[{c.get('group', '?')}]: "
                      f"{c.get('old')} → {c.get('new')} "
                      f"({c.get('method', '?')})")

            # Update all_results with corrected rows
            for gname, corrected_rows in corrected_groups.items():
                if gname in all_results[sec_name]:
                    all_results[sec_name][gname]["rows"] = corrected_rows

                    # Re-verify constraints after reconciliation
                    gconfig = all_results[sec_name][gname].get("config", {})
                    passed, failures = verify_constraints(corrected_rows,
                                                          gconfig)
                    all_results[sec_name][gname]["constraints_passed"] = passed
                    all_results[sec_name][gname]["constraint_failures"] = failures
                    if failures:
                        all_failures.extend(
                            f"{sec_name}/{gname} (post-reconcile): {fl}"
                            for fl in failures)

    # Summary
    overall_passed = len(all_failures) == 0
    print(f"\n{'='*70}")
    print(f"Extraction Complete: {total_groups_extracted} group(s) across "
          f"{len(sections)} section(s)")
    if overall_passed:
        print("ALL CONSTRAINTS PASSED")
    else:
        print(f"FAILURES ({len(all_failures)}):")
        for fl in all_failures:
            print(f"  - {fl}")
    print(f"{'='*70}")

    # Export
    if output_path is None:
        output_path = RESULTS_DIR / f"{image_path.stem}_all_groups.xlsx"
    output_path = Path(output_path)

    from pipeline import export_multigroup_to_excel, export_multigroup_to_json
    export_multigroup_to_excel(all_results, schema, output_path)
    json_path = output_path.with_suffix(".json")
    export_multigroup_to_json(all_results, schema, str(image_path), json_path)

    return all_results, schema, overall_passed, all_failures
