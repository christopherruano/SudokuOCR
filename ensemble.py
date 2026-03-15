"""
Ensemble methods for OCR results from historical census tables.

Provides multi-pass voting, cross-model ensemble, constraint enforcement,
and digit-level voting to improve extraction accuracy from scanned tables.

Input format: lists of parsed row dicts like
    [{"age": "0-5", "persons": 169509, "males": 81224, "females": 88285}, ...]

Usage:
    from ensemble import majority_vote, cross_model_ensemble, enforce_constraints

    # Multi-pass voting across repeated extractions
    merged = majority_vote([pass1_rows, pass2_rows, pass3_rows])

    # Cross-model ensemble
    merged = cross_model_ensemble({
        "openai": openai_rows,
        "gemini": gemini_rows,
        "claude": claude_rows,
    })

    # Constraint enforcement
    corrected, log = enforce_constraints(merged)
"""

import re
import math
import logging
from collections import Counter
from statistics import median

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Age-group normalization and row alignment
# ---------------------------------------------------------------------------

def _normalize_age(age_str):
    """Normalize an age-group string for matching across extractions.

    Handles en-dashes vs hyphens, whitespace, 'Total' prefix, case, and
    common OCR artifacts like curly quotes or stray periods.

    Examples:
        "0\u20135"     -> "0-5"
        "Total 0-5"  -> "0-5"
        " 60 and over " -> "60andover"
    """
    s = str(age_str)
    # Replace en-dash, em-dash, and other dash-like characters with hyphen
    s = re.sub(r"[\u2013\u2014\u2012\u2015]", "-", s)
    # Remove 'total' prefix (case-insensitive) but keep if the whole label
    # is literally "total" (grand total row)
    stripped = re.sub(r"(?i)^total\s+", "", s).strip()
    if stripped:
        s = stripped
    # Collapse whitespace, lowercase
    s = re.sub(r"\s+", "", s).lower()
    # Remove stray punctuation that OCR might introduce
    s = re.sub(r"[.,;:'\"\u2018\u2019\u201c\u201d]", "", s)
    return s


def _align_rows(results_list):
    """Align multiple extraction results by normalized age-group label.

    Returns:
        age_keys: ordered list of normalized age keys (union of all extractions)
        aligned:  list (one per extraction) of dicts mapping
                  normalized_age -> original row dict.
                  Missing rows map to None.
    """
    # Collect all age keys in order of first appearance across extractions
    seen_order = []
    seen_set = set()
    aligned = []

    for rows in results_list:
        row_map = {}
        for row in rows:
            age_raw = row.get("age", "")
            key = _normalize_age(age_raw)
            if not key:
                continue
            # If the same key appears twice in one extraction, keep the first
            if key not in row_map:
                row_map[key] = row
            if key not in seen_set:
                seen_set.add(key)
                seen_order.append(key)
        aligned.append(row_map)

    return seen_order, aligned


def _safe_int(value):
    """Coerce a value to int, tolerating strings with commas/spaces.

    Returns None if conversion is not possible.
    """
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return int(round(value))
    try:
        cleaned = str(value).replace(",", "").replace(" ", "").strip()
        if not cleaned:
            return None
        return int(cleaned)
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Majority voting helpers
# ---------------------------------------------------------------------------

def _vote_single_value(values):
    """Pick the majority value from a list.  Tie-break with median for numerics.

    Args:
        values: list of values (ints or strings). None values are excluded.

    Returns:
        (winner, confidence) where confidence is the fraction of votes
        that agreed with the winner.
    """
    # Filter out Nones
    clean = [v for v in values if v is not None]
    if not clean:
        return None, 0.0

    counts = Counter(clean)
    max_count = counts.most_common(1)[0][1]
    top = [val for val, cnt in counts.items() if cnt == max_count]

    if len(top) == 1:
        winner = top[0]
    else:
        # Tie: try median for numeric values, else pick the first alphabetically
        nums = []
        for v in top:
            n = _safe_int(v)
            if n is not None:
                nums.append(n)
        if nums:
            winner = int(median(nums))
        else:
            winner = sorted(top, key=str)[0]

    confidence = counts.get(winner, max_count) / len(clean)
    return winner, confidence


def _numeric_columns(rows):
    """Detect which keys in row dicts hold numeric data (excluding 'age')."""
    cols = set()
    for row in rows:
        for k, v in row.items():
            if k == "age":
                continue
            if _safe_int(v) is not None:
                cols.add(k)
    return cols


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def majority_vote(results_list):
    """Produce a single best result by majority-voting across multiple extractions.

    For each age group and each column, collects all predicted values and
    takes the most common one. Ties among numeric values are broken by
    taking the median.

    Args:
        results_list: List of extraction results. Each element is a
                      list of row dicts, e.g.
                      [{"age": "0-5", "persons": 169509, ...}, ...]

    Returns:
        Single merged list of row dicts with the voted values.
    """
    if not results_list:
        return []
    if len(results_list) == 1:
        return list(results_list[0])

    age_keys, aligned = _align_rows(results_list)

    # Determine the superset of numeric columns across all extractions
    all_num_cols = set()
    for rows in results_list:
        all_num_cols |= _numeric_columns(rows)

    merged = []
    for age_key in age_keys:
        # Collect the original age label (prefer the most common spelling)
        raw_ages = []
        for amap in aligned:
            row = amap.get(age_key)
            if row is not None:
                raw_ages.append(row.get("age", ""))
        age_label, _ = _vote_single_value(raw_ages)

        voted_row = {"age": age_label}
        for col in sorted(all_num_cols):
            vals = []
            for amap in aligned:
                row = amap.get(age_key)
                if row is not None:
                    vals.append(_safe_int(row.get(col)))
                # If this extraction has no row for this age group, skip
                # (do not append None -- absence is different from observing None)
            winner, conf = _vote_single_value(vals)
            voted_row[col] = winner
            voted_row[f"_conf_{col}"] = round(conf, 4)

        merged.append(voted_row)

    logger.info(
        "majority_vote: merged %d extractions into %d rows",
        len(results_list), len(merged),
    )
    return merged


def cross_model_ensemble(model_results_dict):
    """Ensemble across different models using per-cell majority voting.

    This is a thin wrapper around majority_vote that accepts a dict keyed
    by model name.

    Args:
        model_results_dict: {"openai": [...], "gemini": [...], "claude": [...]}
            Each value is a list of row dicts.

    Returns:
        Single merged list of row dicts.
    """
    if not model_results_dict:
        return []

    results_list = list(model_results_dict.values())
    merged = majority_vote(results_list)

    # Annotate which models contributed to each row
    age_keys, aligned = _align_rows(results_list)
    model_names = list(model_results_dict.keys())

    for i, row in enumerate(merged):
        if i < len(age_keys):
            age_key = age_keys[i]
            contributors = []
            for j, amap in enumerate(aligned):
                if amap.get(age_key) is not None:
                    contributors.append(model_names[j])
            row["_models"] = contributors

    logger.info(
        "cross_model_ensemble: ensembled %d models (%s) -> %d rows",
        len(model_results_dict),
        ", ".join(model_results_dict.keys()),
        len(merged),
    )
    return merged


def enforce_constraints(rows, constraint_type="population"):
    """Apply mathematical constraints to detect and fix errors.

    Constraints applied for constraint_type="population":
      1. persons == males + females  (for each row)
      2. If a "Total" row exists, it should equal the sum of component rows
         for each numeric column.
      3. All numeric values must be non-negative integers.

    For constraint 1, if persons != males + females, the function identifies
    which single value correction is smallest (in absolute terms) and applies
    it.  The heuristic: compute the two candidate corrections --
        a) set persons = males + females
        b) adjust males or females so that males + females = persons
    and pick whichever requires the smallest absolute change.

    Args:
        rows: list of row dicts with at least "persons", "males", "females"
              keys. May also contain confidence annotations from voting
              (keys starting with '_conf_').
        constraint_type: "population" (the only type currently supported).

    Returns:
        (corrected_rows, corrections_log)
        corrected_rows: a new list of row dicts with fixes applied.
        corrections_log: list of dicts describing each correction, e.g.
            {"age": "0-5", "field": "persons", "old": 169510,
             "new": 169509, "rule": "males+females=persons",
             "confidence": "low"}
    """
    corrections = []

    # Deep-copy so we don't mutate the original
    fixed = []
    for row in rows:
        fixed.append(dict(row))

    if constraint_type != "population":
        logger.warning("Unknown constraint_type %r; returning rows unchanged.", constraint_type)
        return fixed, corrections

    # ── Constraint 3: non-negative integers ──────────────────────────────
    num_cols = {"persons", "males", "females"}
    for row in fixed:
        for col in num_cols:
            val = row.get(col)
            as_int = _safe_int(val)
            if as_int is None:
                continue
            if as_int < 0:
                corrections.append({
                    "age": row.get("age"),
                    "field": col,
                    "old": as_int,
                    "new": abs(as_int),
                    "rule": "non_negative",
                    "confidence": "medium",
                })
                row[col] = abs(as_int)
            elif val != as_int:
                # Coerce to clean int (e.g. from float or string)
                row[col] = as_int

    # ── Constraint 1: persons == males + females ─────────────────────────
    for row in fixed:
        p = _safe_int(row.get("persons"))
        m = _safe_int(row.get("males"))
        f = _safe_int(row.get("females"))

        if p is None or m is None or f is None:
            continue
        if p == m + f:
            continue

        expected_p = m + f
        diff_p = abs(p - expected_p)  # cost of fixing persons

        # Cost of fixing males: set males = persons - females
        candidate_m = p - f
        diff_m = abs(m - candidate_m)

        # Cost of fixing females: set females = persons - males
        candidate_f = p - m
        diff_f = abs(f - candidate_f)

        # Use confidence annotations if available to break ties
        conf_p = row.get("_conf_persons", 1.0)
        conf_m = row.get("_conf_males", 1.0)
        conf_f = row.get("_conf_females", 1.0)

        # Build candidates: (absolute_change, negative_confidence, field_name, new_value)
        # We prefer to change the field with the smallest absolute change;
        # among ties, the one with the lowest voting confidence.
        candidates = [
            (diff_p, -conf_p, "persons", expected_p),
            (diff_m, -conf_m, "males", candidate_m),
            (diff_f, -conf_f, "females", candidate_f),
        ]
        # Filter out candidates that would produce a negative value
        candidates = [c for c in candidates if c[3] >= 0]

        if not candidates:
            # All candidate fixes produce a negative -- log and skip
            corrections.append({
                "age": row.get("age"),
                "field": None,
                "old": {"persons": p, "males": m, "females": f},
                "new": None,
                "rule": "males+females=persons",
                "confidence": "unfixable",
            })
            continue

        candidates.sort(key=lambda c: (c[0], c[1]))
        best = candidates[0]
        _, _, field, new_val = best

        old_val = _safe_int(row[field])
        if old_val == new_val:
            continue

        # Classify confidence of correction
        change_pct = abs(new_val - old_val) / max(abs(old_val), 1)
        if change_pct < 0.001:
            conf_label = "high"
        elif change_pct < 0.01:
            conf_label = "medium"
        else:
            conf_label = "low"

        corrections.append({
            "age": row.get("age"),
            "field": field,
            "old": old_val,
            "new": new_val,
            "rule": "males+females=persons",
            "confidence": conf_label,
        })
        row[field] = new_val

    # ── Constraint 2: Total row == sum of component rows ─────────────────
    total_key = None
    total_idx = None
    component_indices = []
    for i, row in enumerate(fixed):
        age_norm = _normalize_age(row.get("age", ""))
        if age_norm == "total":
            total_key = i
            total_idx = i
        else:
            component_indices.append(i)

    if total_idx is not None and component_indices:
        total_row = fixed[total_idx]
        for col in sorted(num_cols):
            total_val = _safe_int(total_row.get(col))
            if total_val is None:
                continue
            component_sum = 0
            any_missing = False
            for ci in component_indices:
                v = _safe_int(fixed[ci].get(col))
                if v is None:
                    any_missing = True
                    break
                component_sum += v
            if any_missing:
                continue
            if total_val != component_sum:
                # Decide: fix the total, or is the discrepancy too large?
                diff = abs(total_val - component_sum)
                pct = diff / max(abs(component_sum), 1)

                if pct < 0.01:
                    # Small discrepancy -- likely a rounding/OCR error in total
                    conf_label = "high" if pct < 0.001 else "medium"
                    corrections.append({
                        "age": total_row.get("age"),
                        "field": col,
                        "old": total_val,
                        "new": component_sum,
                        "rule": "total=sum(components)",
                        "confidence": conf_label,
                    })
                    total_row[col] = component_sum
                else:
                    # Large discrepancy -- log but don't auto-fix
                    corrections.append({
                        "age": total_row.get("age"),
                        "field": col,
                        "old": total_val,
                        "new": component_sum,
                        "rule": "total=sum(components)",
                        "confidence": "low",
                        "note": (
                            f"Large discrepancy ({diff:,}, {pct:.2%}) between "
                            f"total ({total_val:,}) and component sum "
                            f"({component_sum:,}). NOT auto-corrected."
                        ),
                        "auto_corrected": False,
                    })

    if corrections:
        logger.info("enforce_constraints: made %d corrections", len(corrections))
        for c in corrections:
            logger.debug("  %s", c)
    else:
        logger.info("enforce_constraints: all constraints satisfied, no corrections needed")

    return fixed, corrections


def digit_level_vote(values):
    """Digit-level majority voting across multiple numeric predictions.

    Given several OCR readings of the same number, aligns them by
    right-justifying (since the ones digit is most reliably aligned)
    and votes on each digit position independently.

    Examples:
        digit_level_vote([81234, 81224, 81224])  -> 81224
        digit_level_vote([1234, 234])             -> 1234  (leading absence ignored)

    Args:
        values: list of integers (or values coercible to int).

    Returns:
        Voted integer, or None if no valid integers were provided.
    """
    # Coerce to ints, filter out failures
    ints = []
    for v in values:
        n = _safe_int(v)
        if n is not None:
            ints.append(n)
    if not ints:
        return None
    if len(ints) == 1:
        return ints[0]

    # Convert to digit strings, preserving sign
    # (Historical census data should always be positive, but handle negatives
    # gracefully by stripping the sign, voting on magnitude, then restoring.)
    signs = [1 if n >= 0 else -1 for n in ints]
    digit_strs = [str(abs(n)) for n in ints]

    max_len = max(len(d) for d in digit_strs)

    # Right-align: pad on the left with a sentinel character
    SENTINEL = "_"
    padded = [SENTINEL * (max_len - len(d)) + d for d in digit_strs]

    voted_digits = []
    for pos in range(max_len):
        col_digits = [p[pos] for p in padded]
        # Count real digits (ignore sentinel)
        real = [c for c in col_digits if c != SENTINEL]
        if not real:
            # All sentinels at this position -- skip (leading position
            # where shorter numbers have no digit)
            continue
        counts = Counter(real)
        max_count = counts.most_common(1)[0][1]
        top = [d for d, cnt in counts.items() if cnt == max_count]
        if len(top) == 1:
            voted_digits.append(top[0])
        else:
            # Tie among digits at this position: pick the median digit
            digit_vals = sorted(int(d) for d in top)
            med = digit_vals[len(digit_vals) // 2]
            voted_digits.append(str(med))

    if not voted_digits:
        return None

    result = int("".join(voted_digits))

    # Restore sign by majority vote
    sign_vote, _ = _vote_single_value(signs)
    if sign_vote == -1:
        result = -result

    return result


# ---------------------------------------------------------------------------
# Multi-enhancement digit-level ensemble (legacy voting approach)
# ---------------------------------------------------------------------------

def digit_level_ensemble(results_list):
    """Ensemble multiple extraction results using digit-level voting.

    Given many independent readings (e.g. 12 from 4 variants × 3 models),
    aligns rows by age group and applies digit_level_vote() on each cell
    independently. This resolves single-digit confusions (5↔6, 0↔9, etc.)
    that are the dominant error mode in census OCR.

    Args:
        results_list: List of extraction results. Each element is a list of
                      row dicts like [{"age": "0-5", "persons": N, ...}, ...].

    Returns:
        (voted_rows, confidence_metadata)
        voted_rows: list of row dicts with digit-level-voted values.
        confidence_metadata: dict mapping (age_key, column) -> {
            "agreement": float,  # fraction of readings matching the voted value
            "n_readings": int,   # how many readings contributed
            "voted_value": int,  # the winning value
            "all_values": list,  # all readings for this cell
        }
    """
    if not results_list:
        return [], {}
    # Filter out None/empty results
    results_list = [r for r in results_list if r]
    if not results_list:
        return [], {}
    if len(results_list) == 1:
        rows = list(results_list[0])
        meta = {}
        for row in rows:
            age_key = _normalize_age(row.get("age", ""))
            for col in ("persons", "males", "females"):
                val = _safe_int(row.get(col))
                if val is not None:
                    meta[(age_key, col)] = {
                        "agreement": 1.0,
                        "n_readings": 1,
                        "voted_value": val,
                        "all_values": [val],
                    }
        return rows, meta

    age_keys, aligned = _align_rows(results_list)

    # Determine numeric columns
    all_num_cols = set()
    for rows in results_list:
        all_num_cols |= _numeric_columns(rows)
    # Focus on the standard population columns
    target_cols = sorted(all_num_cols & {"persons", "males", "females"})
    if not target_cols:
        target_cols = sorted(all_num_cols)

    voted_rows = []
    confidence_meta = {}

    for age_key in age_keys:
        # Collect original age labels and vote on spelling
        raw_ages = []
        for amap in aligned:
            row = amap.get(age_key)
            if row is not None:
                raw_ages.append(row.get("age", ""))
        age_label, _ = _vote_single_value(raw_ages)

        voted_row = {"age": age_label}

        for col in target_cols:
            # Collect all values for this cell across all readings
            cell_values = []
            for amap in aligned:
                row = amap.get(age_key)
                if row is not None:
                    val = _safe_int(row.get(col))
                    if val is not None:
                        cell_values.append(val)

            if not cell_values:
                voted_row[col] = None
                continue

            # Apply digit-level voting
            voted_val = digit_level_vote(cell_values)
            voted_row[col] = voted_val

            # Compute agreement: what fraction of readings exactly match the voted value
            n_agree = sum(1 for v in cell_values if v == voted_val)
            agreement = n_agree / len(cell_values)

            confidence_meta[(age_key, col)] = {
                "agreement": round(agreement, 4),
                "n_readings": len(cell_values),
                "voted_value": voted_val,
                "all_values": cell_values,
            }
            voted_row[f"_conf_{col}"] = round(agreement, 4)

        voted_rows.append(voted_row)

    logger.info(
        "digit_level_ensemble: ensembled %d readings into %d rows",
        len(results_list), len(voted_rows),
    )
    return voted_rows, confidence_meta


# ---------------------------------------------------------------------------
# Constraint-propagation ensemble (deductive solver)
# ---------------------------------------------------------------------------

COLS = ("persons", "males", "females")


def _collect_cell_candidates(aligned, age_key, col):
    """Collect all unique candidate values for a cell, with vote counts.

    Returns:
        List of (value, count) tuples sorted by count descending.
    """
    counts = Counter()
    for amap in aligned:
        row = amap.get(age_key)
        if row is not None:
            val = _safe_int(row.get(col))
            if val is not None:
                counts[val] += 1
    return sorted(counts.items(), key=lambda x: -x[1])


def constraint_ensemble(results_list, constraint_groups=None, known_totals=None):
    """Constraint-propagation ensemble.

    Solves the table like a Sudoku: start with what's certain, then
    deduce everything else through chains of constraints.

    Phase 0.5 — INJECT: Lock cells from known_totals (e.g. grand total = 1000).
    Phase 1 — LOCK: For every cell where all N readings agree, lock it.
    Phase 2 — PROPAGATE: Iteratively apply constraint types until no more progress:
        Row constraint:   P = M + F  (2 known → deduce the 3rd)
        Group constraints: subtotal = sum(components) for each group
                           (all-but-one known → deduce the missing one)
      Deductions cascade: resolving one cell may unlock another constraint.
    Phase 3 — RESOLVE REMAINING: For any cells still unresolved after
              propagation, pick from candidates using constraint
              satisfaction (find (P,M,F) triple from candidates where
              P=M+F, preferring higher vote counts).

    Args:
        results_list: List of extraction results (list of list of row dicts).
        constraint_groups: Dict mapping subtotal_key -> [component_keys].
            Each defines a sum constraint: subtotal = sum(components).
            Keys are normalized age strings matching _normalize_age() output.
            If None, auto-detects from the "total" row.
        known_totals: Dict mapping age_key -> {col: value} for known fixed
            values (e.g. {"total": {"males": 1000, "females": 1000}}).

    Returns:
        (resolved_rows, resolution_log)
    """
    if not results_list:
        return [], []
    results_list = [r for r in results_list if r]
    if not results_list:
        return [], []

    age_keys, aligned = _align_rows(results_list)
    n_readings = len(results_list)

    # ── Build the grid ────────────────────────────────────────────────
    # grid[age_key][col] = resolved value (int) or None
    # candidates[age_key][col] = [(value, count), ...]
    # status[age_key][col] = "locked" | "deduced" | "unresolved"

    grid = {}
    candidates = {}
    status = {}
    age_labels = {}  # age_key -> display label
    log = []

    for age_key in age_keys:
        # Vote on age label
        raw_ages = []
        for amap in aligned:
            row = amap.get(age_key)
            if row is not None:
                raw_ages.append(row.get("age", ""))
        age_labels[age_key], _ = _vote_single_value(raw_ages)

        grid[age_key] = {}
        candidates[age_key] = {}
        status[age_key] = {}

        for col in COLS:
            cands = _collect_cell_candidates(aligned, age_key, col)
            candidates[age_key][col] = cands
            grid[age_key][col] = None
            status[age_key][col] = "unresolved"

    # ── Build constraint groups ──────────────────────────────────────
    # Auto-detect if not provided: "total" row sums all other rows
    if constraint_groups is None:
        total_key = "total" if "total" in grid else None
        if total_key is not None:
            component_keys = [k for k in age_keys if k != total_key]
            constraint_groups = {total_key: component_keys}
        else:
            constraint_groups = {}

    # Filter to groups where key and at least some components exist in grid
    active_groups = {}
    for subtotal_key, comp_keys in constraint_groups.items():
        if subtotal_key in grid:
            valid_comps = [k for k in comp_keys if k in grid]
            if valid_comps:
                active_groups[subtotal_key] = valid_comps

    # ── Phase 0.5: Inject known totals ───────────────────────────────
    if known_totals:
        for age_key, col_vals in known_totals.items():
            if age_key in grid:
                for col, val in col_vals.items():
                    if col in COLS:
                        grid[age_key][col] = val
                        status[age_key][col] = "locked"
                        log.append({
                            "age": age_labels.get(age_key, age_key),
                            "col": col, "value": val,
                            "method": "known_total",
                            "votes": "fixed",
                        })

    # ── Phase 1: Lock high-agreement cells ─────────────────────────────
    # Phase 1a: unanimous (100%) — highest confidence
    # Phase 1b: supermajority (≥75%) — still very strong for 12 readings

    supermajority_threshold = 0.75

    for age_key in age_keys:
        for col in COLS:
            if grid[age_key][col] is not None:
                continue  # Already locked by known_totals
            cands = candidates[age_key][col]
            if not cands:
                continue
            total_votes = sum(c for _, c in cands)
            top_val, top_count = cands[0]  # Most popular
            agreement = top_count / total_votes if total_votes else 0

            if len(cands) == 1 and total_votes == n_readings:
                grid[age_key][col] = top_val
                status[age_key][col] = "locked"
                log.append({
                    "age": age_labels[age_key], "col": col,
                    "value": top_val,
                    "method": "locked_100pct",
                    "votes": f"{top_count}/{total_votes}",
                })
            elif agreement >= supermajority_threshold:
                grid[age_key][col] = top_val
                status[age_key][col] = "locked"
                log.append({
                    "age": age_labels[age_key], "col": col,
                    "value": top_val,
                    "method": "locked_supermajority",
                    "votes": f"{top_count}/{total_votes}",
                })

    # Log Phase 1 summary
    n_locked = sum(1 for ak in age_keys for c in COLS
                   if status[ak][c] == "locked")
    n_total_cells = len(age_keys) * len(COLS)
    logger.info("Phase 1: locked %d/%d cells. Active groups: %s",
                n_locked, n_total_cells, list(active_groups.keys()))

    # ── Phase 2: Constraint propagation ───────────────────────────────

    def _resolve_cell(age_key, col, value, method, note=""):
        """Mark a cell as deduced and record it."""
        grid[age_key][col] = value
        status[age_key][col] = "deduced"
        # Check if value was observed
        cand_values = [v for v, _ in candidates[age_key][col]]
        observed = value in cand_values
        log.append({
            "age": age_labels[age_key], "col": col,
            "value": value,
            "method": method,
            "observed": observed,
            "note": note,
        })

    changed = True
    iteration = 0
    while changed:
        changed = False
        iteration += 1

        # Row constraint: P = M + F
        for age_key in age_keys:
            resolved = {c: grid[age_key][c] for c in COLS
                        if grid[age_key][c] is not None}
            unresolved = [c for c in COLS if grid[age_key][c] is None]

            if len(resolved) == 2 and len(unresolved) == 1:
                missing_col = unresolved[0]
                if missing_col == "persons":
                    implied = resolved["males"] + resolved["females"]
                elif missing_col == "males":
                    implied = resolved["persons"] - resolved["females"]
                else:
                    implied = resolved["persons"] - resolved["males"]

                _resolve_cell(age_key, missing_col, implied,
                              f"row_constraint_iter{iteration}",
                              f"P=M+F, implied={implied}")
                changed = True

        # Group sum constraints: subtotal = sum(components) for each group
        for subtotal_key, comp_keys in active_groups.items():
            for col in COLS:
                subtotal_val = grid[subtotal_key][col]
                comp_vals = [(ck, grid[ck][col]) for ck in comp_keys]
                resolved_comps = [(ck, v) for ck, v in comp_vals
                                  if v is not None]
                unresolved_comps = [ck for ck, v in comp_vals if v is None]

                if subtotal_val is not None and len(unresolved_comps) == 1:
                    # All but one component known → deduce the missing one
                    missing_key = unresolved_comps[0]
                    comp_sum = sum(v for _, v in resolved_comps)
                    implied = subtotal_val - comp_sum
                    logger.debug(
                        "Group 1-unknown: %s.%s, missing=%s → %s",
                        subtotal_key, col, missing_key, implied)
                    _resolve_cell(missing_key, col, implied,
                                  f"group_constraint_iter{iteration}",
                                  f"{subtotal_key}-others="
                                  f"{subtotal_val}-{comp_sum}={implied}")
                    changed = True

                elif subtotal_val is not None and len(unresolved_comps) == 2:
                    # Two unknowns — try candidate pairs that sum correctly
                    comp_sum = sum(v for _, v in resolved_comps)
                    target = subtotal_val - comp_sum
                    k1, k2 = unresolved_comps
                    cands1 = candidates[k1][col]
                    cands2 = candidates[k2][col]
                    valid_pairs = []
                    for v1, c1 in cands1:
                        for v2, c2 in cands2:
                            if v1 + v2 == target:
                                valid_pairs.append((v1, v2, c1 + c2))
                    if len(valid_pairs) == 1:
                        # Uniquely determined!
                        v1, v2, _ = valid_pairs[0]
                        _resolve_cell(k1, col, v1,
                                      f"group_pair_iter{iteration}",
                                      f"{k1}+{k2}={target}, unique pair")
                        _resolve_cell(k2, col, v2,
                                      f"group_pair_iter{iteration}",
                                      f"{k1}+{k2}={target}, unique pair")
                        changed = True
                    elif len(valid_pairs) > 1:
                        # Multiple valid pairs — pick highest vote total
                        valid_pairs.sort(key=lambda x: -x[2])
                        best = valid_pairs[0]
                        # Only use if significantly better than runner-up
                        if len(valid_pairs) == 1 or best[2] > valid_pairs[1][2]:
                            _resolve_cell(k1, col, best[0],
                                          f"group_pair_best_iter{iteration}",
                                          f"{k1}+{k2}={target}, best of "
                                          f"{len(valid_pairs)} pairs")
                            _resolve_cell(k2, col, best[1],
                                          f"group_pair_best_iter{iteration}",
                                          f"{k1}+{k2}={target}, best of "
                                          f"{len(valid_pairs)} pairs")
                            changed = True

                elif subtotal_val is None and len(unresolved_comps) == 0:
                    # All components known → deduce the subtotal
                    comp_sum = sum(v for _, v in resolved_comps)
                    _resolve_cell(subtotal_key, col, comp_sum,
                                  f"group_sum_iter{iteration}",
                                  f"sum({','.join(comp_keys)})={comp_sum}")
                    changed = True

    # ── Phase 2.5: Hierarchical column CSP ──────────────────────────────
    # For each column with a known total, solve the constraint hierarchy
    # top-down.  At each level, compute possible values for each sub-group
    # from its component candidates (top 2 per leaf cell).  Find the unique
    # combination that matches the known total, then backtrack to determine
    # individual leaf cells.  This exploits the group structure to reduce
    # the search space at each level.

    if known_totals:
        import itertools

        # Phase 2.5a: Lock high-confidence intermediate subtotals.
        # If a group key has a direct subtotal reading with strict majority
        # (>50%), inject it as a known value.  Models read subtotal rows
        # more reliably than individual cells, so even moderate agreement
        # provides strong anchor points that dramatically narrow the CSP
        # search space.
        subtotal_lock_threshold = 0.50
        csp_known = dict(known_totals)
        for grp_key in active_groups:
            if grp_key in csp_known:
                continue
            for col in COLS:
                cands = candidates.get(grp_key, {}).get(col, [])
                if not cands:
                    continue
                total_votes = sum(c for _, c in cands)
                if total_votes > 0:
                    top_val, top_count = cands[0]
                    agreement = top_count / total_votes
                    if agreement > subtotal_lock_threshold:
                        if grp_key not in csp_known:
                            csp_known[grp_key] = {}
                        csp_known[grp_key][col] = top_val
                        logger.debug("CSP: locked intermediate subtotal "
                                     "%s.%s=%d (%.0f%% agreement)",
                                     grp_key, col, top_val,
                                     agreement * 100)

        def _cell_values(key, col):
            """Possible values for a cell: always top-2 candidates.
            Let the constraint determine the answer; votes are tiebreakers.
            For subtotals locked by csp_known, use the fixed value.
            Otherwise: enumerate sums from component values."""
            # If this group key is locked by csp_known, treat as fixed
            if key in csp_known and col in csp_known[key]:
                return {csp_known[key][col]}
            if key not in active_groups:
                cands = candidates.get(key, {}).get(col, [])
                if not cands:
                    return set()
                if len(cands) == 1:
                    return {cands[0][0]}
                return {v for v, _ in cands[:2]}
            result = set()
            comp_vals = [sorted(_cell_values(ck, col))
                         for ck in active_groups[key]]
            if any(not s for s in comp_vals):
                return set()
            for combo in itertools.product(*comp_vals):
                result.add(sum(combo))
            return result

        def _solve(key, col, target, _top_key=None):
            """Find best leaf assignment summing to target, preferring
            higher vote counts. Returns (dict {leaf_key: value}, score)
            or None.  _top_key is the key being solved at the outermost
            level — skip the csp_known shortcut for it."""
            # Locked intermediate subtotal — treat as pseudo-leaf
            # (but not for the key we're actively solving at the top)
            if key != _top_key and key in csp_known \
                    and col in csp_known[key]:
                if csp_known[key][col] == target:
                    cands = candidates.get(key, {}).get(col, [])
                    score = dict(cands).get(target, 0)
                    return ({key: target}, score)
                return None
            if key not in active_groups:
                cands = candidates.get(key, {}).get(col, [])
                # Check top-2 candidates (matching _cell_values)
                for v, c in cands[:2]:
                    if v == target:
                        return ({key: v}, c)
                return None
            comp_keys = active_groups[key]
            comp_vals = [sorted(_cell_values(ck, col)) for ck in comp_keys]
            if any(not s for s in comp_vals):
                return None
            valid = [c for c in itertools.product(*comp_vals)
                     if sum(c) == target]
            if not valid:
                return None
            # Try each valid combo, pick the one with highest leaf vote score
            best_result = None
            best_score = -1
            for combo in valid:
                result = {}
                total_score = 0
                ok = True
                for i, ck in enumerate(comp_keys):
                    sub = _solve(ck, col, combo[i], _top_key)
                    if sub is None:
                        ok = False
                        break
                    sub_dict, sub_score = sub
                    result.update(sub_dict)
                    total_score += sub_score
                if ok and total_score > best_score:
                    best_result = result
                    best_score = total_score
            if best_result is None:
                return None
            if len(valid) > 1:
                logger.debug("CSP %s.%s: %d valid combos for target=%d, "
                             "picked best (score=%d)",
                             key, col, len(valid), target, best_score)
            return (best_result, best_score)

        for total_key_kt, col_vals in csp_known.items():
            if total_key_kt not in active_groups:
                continue
            for col, target_total in col_vals.items():
                if col not in COLS:
                    continue
                logger.debug("Hierarchical CSP: %s.%s = %d",
                             total_key_kt, col, target_total)

                result = _solve(total_key_kt, col, target_total,
                                _top_key=total_key_kt)
                if result:
                    solution, score = result
                    logger.info("Hierarchical CSP solved %s.%s: %d cells "
                                "(score=%d)",
                                total_key_kt, col, len(solution), score)
                    for lk, new_val in solution.items():
                        if lk not in grid:
                            continue
                        old_val = grid[lk][col]
                        if old_val != new_val or old_val is None:
                            grid[lk][col] = new_val
                            status[lk][col] = "deduced"
                            log.append({
                                "age": age_labels.get(lk, lk), "col": col,
                                "value": new_val,
                                "method": "hierarchical_csp",
                                "observed": new_val in [
                                    v for v, _ in
                                    candidates.get(lk, {}).get(col, [])],
                                "note": f"was {old_val}",
                            })
                        if col in ("males", "females"):
                            m = grid[lk].get("males")
                            f = grid[lk].get("females")
                            if m is not None and f is not None:
                                implied_p = m + f
                                if grid[lk].get("persons") != implied_p:
                                    grid[lk]["persons"] = implied_p
                                    status[lk]["persons"] = "deduced"
                                    log.append({
                                        "age": age_labels.get(lk, lk),
                                        "col": "persons",
                                        "value": implied_p,
                                        "method": "hierarchical_csp_row",
                                        "observed": implied_p in [
                                            v for v, _ in
                                            candidates[lk]["persons"]],
                                        "note": "P=M+F after CSP",
                                    })
                else:
                    logger.debug("Hierarchical CSP: %s.%s not uniquely solvable",
                                 total_key_kt, col)

    # ── Phase 3: Resolve remaining via candidate constraint search ────

    for age_key in age_keys:
        unresolved = [c for c in COLS if grid[age_key][c] is None]
        if not unresolved:
            continue

        resolved = {c: grid[age_key][c] for c in COLS
                    if grid[age_key][c] is not None}
        n_resolved = len(resolved)

        if n_resolved == 2:
            # One missing — constraint determines it
            missing_col = unresolved[0]
            if missing_col == "persons":
                implied = resolved["males"] + resolved["females"]
            elif missing_col == "males":
                implied = resolved["persons"] - resolved["females"]
            else:
                implied = resolved["persons"] - resolved["males"]

            cand_vals = [v for v, _ in candidates[age_key][missing_col]]
            if implied in cand_vals:
                _resolve_cell(age_key, missing_col, implied,
                              "phase3_constraint_observed",
                              f"implied={implied}")
            else:
                _resolve_cell(age_key, missing_col, implied,
                              "phase3_constraint_unobserved",
                              f"implied={implied}, NOT in candidates {cand_vals}")

        elif n_resolved == 1:
            # Two missing — find best pair from candidates
            locked_col = list(resolved.keys())[0]
            locked_val = resolved[locked_col]
            cols_to_solve = unresolved

            cands_a = candidates[age_key][cols_to_solve[0]]
            cands_b = candidates[age_key][cols_to_solve[1]]
            result = _find_best_pair_for_constraint(
                locked_col, locked_val,
                cols_to_solve[0], cands_a,
                cols_to_solve[1], cands_b)
            for col, val, method in result:
                _resolve_cell(age_key, col, val, f"phase3_{method}")

        else:
            # Zero resolved — find best triple from candidates
            cands_p = candidates[age_key]["persons"]
            cands_m = candidates[age_key]["males"]
            cands_f = candidates[age_key]["females"]
            if cands_p and cands_m and cands_f:
                result = _find_best_triple(cands_p, cands_m, cands_f)
                for col, val, method in result:
                    _resolve_cell(age_key, col, val, f"phase3_{method}")
            else:
                # Absolute fallback: most popular candidate
                for col in COLS:
                    cands = candidates[age_key][col]
                    if cands and grid[age_key][col] is None:
                        _resolve_cell(age_key, col, cands[0][0],
                                      "phase3_fallback_popular")

    # ── Build output rows ─────────────────────────────────────────────

    resolved_rows = []
    for age_key in age_keys:
        row = {"age": age_labels[age_key]}
        for col in COLS:
            row[col] = grid[age_key][col]
            # Confidence: fraction of readings that match the resolved value
            cands = candidates[age_key][col]
            total_votes = sum(c for _, c in cands) if cands else 0
            val = grid[age_key][col]
            if val is not None and total_votes > 0:
                n_agree = sum(c for v, c in cands if v == val)
                row[f"_conf_{col}"] = round(n_agree / total_votes, 4)
            row[f"_status_{col}"] = status[age_key][col]
        resolved_rows.append(row)

    logger.info(
        "constraint_ensemble: resolved %d rows from %d readings "
        "(%d iterations of propagation)",
        len(resolved_rows), n_readings, iteration,
    )
    return resolved_rows, log


def _find_best_pair_for_constraint(locked_col, locked_val,
                                   col_a, cands_a, col_b, cands_b):
    """Find the best (val_a, val_b) pair satisfying M+F=P given one locked column.

    Returns list of (col, value, method) tuples.
    """
    def _check_constraint(va, vb):
        vals = {locked_col: locked_val, col_a: va, col_b: vb}
        return vals["persons"] == vals["males"] + vals["females"]

    # Try all pairs, prefer exact match with highest total votes
    best = None
    best_score = -1
    for va, ca in cands_a:
        for vb, cb in cands_b:
            if _check_constraint(va, vb):
                score = ca + cb
                if score > best_score:
                    best = (va, vb)
                    best_score = score
    if best is not None:
        return [
            (col_a, best[0], "pair_exact"),
            (col_b, best[1], "pair_exact"),
        ]

    # No exact match — smallest residual
    best = None
    best_residual = float("inf")
    best_score = -1
    for va, ca in cands_a:
        for vb, cb in cands_b:
            vals = {locked_col: locked_val, col_a: va, col_b: vb}
            residual = abs(vals["persons"] - vals["males"] - vals["females"])
            score = ca + cb
            if (residual < best_residual
                    or (residual == best_residual and score > best_score)):
                best = (va, vb)
                best_residual = residual
                best_score = score
    if best is not None:
        return [
            (col_a, best[0], "pair_approx"),
            (col_b, best[1], "pair_approx"),
        ]
    return [
        (col_a, cands_a[0][0] if cands_a else 0, "fallback"),
        (col_b, cands_b[0][0] if cands_b else 0, "fallback"),
    ]


def _find_best_triple(cands_p, cands_m, cands_f):
    """Find the best (P, M, F) triple where P = M + F.

    Returns list of (col, value, method) tuples.
    """
    best = None
    best_score = -1
    for vp, cp in cands_p:
        for vm, cm in cands_m:
            for vf, cf in cands_f:
                if vp == vm + vf:
                    score = cp + cm + cf
                    if score > best_score:
                        best = (vp, vm, vf)
                        best_score = score
    if best is not None:
        return [
            ("persons", best[0], "triple_exact"),
            ("males", best[1], "triple_exact"),
            ("females", best[2], "triple_exact"),
        ]

    # No exact — smallest residual
    best = None
    best_residual = float("inf")
    best_score = -1
    for vp, cp in cands_p:
        for vm, cm in cands_m:
            for vf, cf in cands_f:
                residual = abs(vp - vm - vf)
                score = cp + cm + cf
                if (residual < best_residual
                        or (residual == best_residual and score > best_score)):
                    best = (vp, vm, vf)
                    best_residual = residual
                    best_score = score
    if best is not None:
        return [
            ("persons", best[0], "triple_approx"),
            ("males", best[1], "triple_approx"),
            ("females", best[2], "triple_approx"),
        ]
    return [
        ("persons", cands_p[0][0] if cands_p else 0, "fallback"),
        ("males", cands_m[0][0] if cands_m else 0, "fallback"),
        ("females", cands_f[0][0] if cands_f else 0, "fallback"),
    ]


# ---------------------------------------------------------------------------
# Convenience: full pipeline
# ---------------------------------------------------------------------------

def ensemble_pipeline(model_results_dict, n_passes_per_model=None):
    """Run the full ensemble pipeline: multi-pass vote per model, cross-model
    ensemble, then constraint enforcement.

    Args:
        model_results_dict: Either
            - {"model_name": [rows]}  for single-pass per model, or
            - {"model_name": [[rows_pass1], [rows_pass2], ...]} for multi-pass.
        n_passes_per_model: If set, indicates each model value is a list of
            lists (multi-pass). If None, auto-detect based on whether values
            are lists of lists.

    Returns:
        (final_rows, corrections_log, metadata)
    """
    # Determine if multi-pass
    per_model_voted = {}
    for model, data in model_results_dict.items():
        if not data:
            continue
        # Auto-detect: if data is a list whose first element is also a list
        # of dicts, it is multi-pass.
        is_multipass = (
            isinstance(data, list)
            and len(data) > 0
            and isinstance(data[0], list)
        )
        if n_passes_per_model is not None:
            is_multipass = n_passes_per_model > 1

        if is_multipass and isinstance(data[0], list):
            per_model_voted[model] = majority_vote(data)
        else:
            # Single pass -- use as-is
            per_model_voted[model] = data if isinstance(data, list) else []

    # Cross-model ensemble
    if len(per_model_voted) > 1:
        ensembled = cross_model_ensemble(per_model_voted)
    elif per_model_voted:
        ensembled = list(per_model_voted.values())[0]
    else:
        return [], [], {"models": 0}

    # Constraint enforcement
    corrected, corrections = enforce_constraints(ensembled)

    metadata = {
        "models": list(model_results_dict.keys()),
        "num_models": len(per_model_voted),
        "num_rows": len(corrected),
        "num_corrections": len(corrections),
    }
    return corrected, corrections, metadata


# ---------------------------------------------------------------------------
# Cross-group reconciliation
# ---------------------------------------------------------------------------

def detect_column_confusion(group_results, cross_group_constraints):
    """Flag groups whose values duplicate an adjacent group.

    For each pair of adjacent groups (in the order they appear in
    cross_group_constraints), count how many (age, col) cells match
    exactly.  If >50% match, the rightward group likely read data from
    the leftward group's columns — classic column confusion.

    Args:
        group_results: {group_name: [row_dicts]} — resolved output per group.
        cross_group_constraints: {total_group: [component_groups]}.

    Returns:
        Set of confused group names that should be excluded from
        reconciliation.
    """
    confused = set()

    for total_group, comp_groups in cross_group_constraints.items():
        # Check total group + components in order
        all_groups = [total_group] + list(comp_groups)

        # Build per-group lookups: group_name -> {norm_age: row_dict}
        lookups = {}
        for gname in all_groups:
            if gname not in group_results:
                continue
            rmap = {}
            for row in group_results[gname]:
                key = _normalize_age(row.get("age", ""))
                if key:
                    rmap[key] = row
            lookups[gname] = rmap

        # Compare each pair of adjacent groups
        for i in range(len(all_groups) - 1):
            left_g = all_groups[i]
            right_g = all_groups[i + 1]

            if left_g not in lookups or right_g not in lookups:
                continue

            left_map = lookups[left_g]
            right_map = lookups[right_g]

            total_cells = 0
            matching_cells = 0
            for age_key in left_map:
                if age_key not in right_map:
                    continue
                for col in COLS:
                    left_val = _safe_int(left_map[age_key].get(col))
                    right_val = _safe_int(right_map[age_key].get(col))
                    if left_val is not None and right_val is not None:
                        total_cells += 1
                        if left_val == right_val:
                            matching_cells += 1

            if total_cells > 0:
                match_rate = matching_cells / total_cells
                if match_rate > 0.50:
                    confused.add(right_g)
                    logger.warning(
                        "Column confusion detected: %s matches %s at %.0f%% "
                        "(%d/%d cells) — excluding from reconciliation",
                        right_g, left_g, match_rate * 100,
                        matching_cells, total_cells)

    return confused


def _is_plausible_correction(old_val, new_val, total_val=None):
    """Check if a cross-group correction is plausible.

    Guards against overcorrection when input data is garbage (e.g. column
    confusion fed wrong numbers entirely).

    Returns (is_plausible, reason) tuple.
    """
    if new_val < 0:
        return False, "negative value"

    # Component can't exceed the population total (if known)
    if total_val is not None and new_val > total_val:
        return False, f"exceeds population ({new_val} > {total_val})"

    # Change can't be >50% of original (unless original is very small)
    if old_val is not None and old_val > 10:
        change_pct = abs(new_val - old_val) / old_val
        if change_pct > 0.50:
            return False, f"change too large ({change_pct:.0%} of original)"

    return True, ""


def cross_group_reconcile(group_results, cross_group_constraints,
                          all_raw_readings=None):
    """Reconcile results across column groups using sum constraints.

    Generalizable: works for any {total_group: [component_groups]} structure.
    For example:
      {"POPULATION": ["UNMARRIED", "MARRIED", "WIDOWED", "DIVORCED"]}
    means Population = Unmarried + Married + Widowed + Divorced for each
    (age, col) cell.

    Args:
        group_results: {group_name: [row_dicts]} — resolved output per group.
        cross_group_constraints: {total_group: [component_groups]}.
        all_raw_readings: {group_name: list[list[row_dicts]]} — raw OCR
            readings for candidate lookup. Optional.

    Returns:
        (corrected_group_results, reconciliation_log)
        corrected_group_results has the same shape as group_results.
    """
    if not cross_group_constraints:
        return group_results, []

    log = []

    # Deep-copy group_results so we don't mutate the original
    corrected = {}
    for gname, rows in group_results.items():
        corrected[gname] = [dict(r) for r in rows]

    # Build per-group row lookup: group_name -> {norm_age: row_dict}
    def _build_lookup(results_dict):
        lookups = {}
        for gname, rows in results_dict.items():
            rmap = {}
            for row in rows:
                key = _normalize_age(row.get("age", ""))
                if key:
                    rmap[key] = row
            lookups[gname] = rmap
        return lookups

    lookups = _build_lookup(corrected)

    # Build raw candidate lookups if available
    raw_lookups = {}  # group_name -> {norm_age -> {col -> Counter}}
    if all_raw_readings:
        for gname, readings_list in all_raw_readings.items():
            age_col_counts = {}
            for reading in readings_list:
                if not reading:
                    continue
                for row in reading:
                    key = _normalize_age(row.get("age", ""))
                    if not key:
                        continue
                    if key not in age_col_counts:
                        age_col_counts[key] = {c: Counter() for c in COLS}
                    for col in COLS:
                        val = _safe_int(row.get(col))
                        if val is not None:
                            age_col_counts[key][col][val] += 1
            raw_lookups[gname] = age_col_counts

    # Collect all age keys (union across groups, preserving order)
    all_age_keys = []
    seen = set()
    for gname in group_results:
        for row in group_results[gname]:
            key = _normalize_age(row.get("age", ""))
            if key and key not in seen:
                seen.add(key)
                all_age_keys.append(key)

    # ── Pre-check: verify Total row satisfies cross-group constraint ──
    for total_group, comp_groups in cross_group_constraints.items():
        total_lookup = lookups.get(total_group, {})
        if not total_lookup:
            log.append({"phase": "precheck", "note":
                        f"Total group '{total_group}' not found, skipping"})
            continue

        total_row = total_lookup.get("total")
        if total_row is None:
            continue

        for col in COLS:
            total_val = _safe_int(total_row.get(col))
            if total_val is None:
                continue
            comp_sum = 0
            all_present = True
            for cg in comp_groups:
                cg_lookup = lookups.get(cg, {})
                cg_total = cg_lookup.get("total")
                if cg_total is None or _safe_int(cg_total.get(col)) is None:
                    all_present = False
                    break
                comp_sum += _safe_int(cg_total[col])
            if all_present and total_val != comp_sum:
                log.append({
                    "phase": "precheck",
                    "note": f"Total row {col}: {total_group}={total_val} != "
                            f"sum({comp_groups})={comp_sum}, "
                            f"diff={total_val - comp_sum}",
                })

    # ── Phase 1: Find all discrepancies ──────────────────────────────
    discrepancies = []

    for total_group, comp_groups in cross_group_constraints.items():
        total_lookup = lookups.get(total_group, {})
        if not total_lookup:
            continue

        for age_key in all_age_keys:
            total_row = total_lookup.get(age_key)
            if total_row is None:
                continue
            for col in COLS:
                total_val = _safe_int(total_row.get(col))
                if total_val is None:
                    continue
                comp_sum = 0
                all_present = True
                for cg in comp_groups:
                    cg_lookup = lookups.get(cg, {})
                    cg_row = cg_lookup.get(age_key)
                    if cg_row is None or _safe_int(cg_row.get(col)) is None:
                        all_present = False
                        break
                    comp_sum += _safe_int(cg_row[col])
                if all_present and total_val != comp_sum:
                    discrepancies.append((age_key, col, total_group,
                                          total_val, comp_sum))

    if not discrepancies:
        log.append({"phase": "summary", "note": "No cross-group discrepancies"})
        return corrected, log

    log.append({"phase": "phase1",
                "note": f"Found {len(discrepancies)} discrepancies"})

    # ── Phase 2: Confidence-based deduction (iterative) ──────────────
    changed = True
    iteration = 0
    max_iterations = 10

    while changed and iteration < max_iterations:
        changed = False
        iteration += 1

        for total_group, comp_groups in cross_group_constraints.items():
            total_lookup = lookups.get(total_group, {})
            if not total_lookup:
                continue

            for age_key in all_age_keys:
                for col in COLS:
                    total_row = total_lookup.get(age_key)
                    if total_row is None:
                        continue
                    total_val = _safe_int(total_row.get(col))
                    if total_val is None:
                        continue

                    # Gather component values and confidence
                    comp_vals = []
                    for cg in comp_groups:
                        cg_row = lookups.get(cg, {}).get(age_key)
                        if cg_row is None:
                            comp_vals.append((cg, None, 0.0))
                        else:
                            val = _safe_int(cg_row.get(col))
                            conf = cg_row.get(f"_conf_{col}", 0.5)
                            comp_vals.append((cg, val, conf))

                    comp_sum = sum(v for _, v, _ in comp_vals
                                   if v is not None)
                    n_present = sum(1 for _, v, _ in comp_vals
                                    if v is not None)

                    if n_present < len(comp_groups):
                        continue
                    if total_val == comp_sum:
                        continue

                    diff = total_val - comp_sum
                    HIGH_CONF = 0.75
                    low_conf_groups = [(cg, val, conf) for cg, val, conf
                                       in comp_vals
                                       if conf < HIGH_CONF and val is not None]

                    total_conf = total_row.get(f"_conf_{col}", 0.5)

                    if len(low_conf_groups) == 1 and total_conf >= HIGH_CONF:
                        # One low-confidence component — fix it
                        fix_group, old_val, old_conf = low_conf_groups[0]
                        new_val = old_val + diff
                        plausible, reason = _is_plausible_correction(
                            old_val, new_val, total_val)
                        if new_val >= 0 and plausible:
                            fix_row = lookups[fix_group][age_key]
                            fix_row[col] = new_val
                            log.append({
                                "phase": "phase2",
                                "age": age_key, "col": col,
                                "group": fix_group,
                                "old": old_val, "new": new_val,
                                "method": "confidence_deduction",
                                "note": f"diff={diff}, conf={old_conf:.2f}",
                            })
                            changed = True
                            _cascade_row_constraint(fix_row, col, log,
                                                    age_key, fix_group)
                        elif new_val >= 0 and not plausible:
                            log.append({
                                "phase": "phase2",
                                "age": age_key, "col": col,
                                "group": fix_group,
                                "old": old_val, "new": new_val,
                                "method": "plausibility_blocked",
                                "note": reason,
                            })

                    elif total_conf < HIGH_CONF and all(
                            conf >= HIGH_CONF for _, _, conf in comp_vals
                            if _ != total_group):
                        # Total is low confidence, components are high —
                        # fix the total
                        new_total = comp_sum
                        total_row[col] = new_total
                        log.append({
                            "phase": "phase2",
                            "age": age_key, "col": col,
                            "group": total_group,
                            "old": total_val, "new": new_total,
                            "method": "total_deduction",
                            "note": f"total conf={total_conf:.2f}",
                        })
                        changed = True
                        _cascade_row_constraint(total_row, col, log,
                                                age_key, total_group)

    # ── Phase 3: Candidate search ────────────────────────────────────
    if raw_lookups:
        for total_group, comp_groups in cross_group_constraints.items():
            total_lookup = lookups.get(total_group, {})
            if not total_lookup:
                continue

            for age_key in all_age_keys:
                for col in COLS:
                    total_row = total_lookup.get(age_key)
                    if total_row is None:
                        continue
                    total_val = _safe_int(total_row.get(col))
                    if total_val is None:
                        continue

                    comp_sum = 0
                    all_present = True
                    for cg in comp_groups:
                        cg_row = lookups.get(cg, {}).get(age_key)
                        if cg_row is None or _safe_int(cg_row.get(col)) is None:
                            all_present = False
                            break
                        comp_sum += _safe_int(cg_row[col])

                    if not all_present or total_val == comp_sum:
                        continue

                    diff = total_val - comp_sum
                    fixed = False

                    # Try each component group
                    for cg in comp_groups:
                        cg_row = lookups.get(cg, {}).get(age_key)
                        if cg_row is None:
                            continue
                        current_val = _safe_int(cg_row.get(col))
                        if current_val is None:
                            continue
                        needed = current_val + diff
                        if needed < 0:
                            continue

                        raw_cands = (raw_lookups.get(cg, {})
                                     .get(age_key, {})
                                     .get(col, Counter()))
                        if needed in raw_cands and needed != current_val:
                            plausible, reason = _is_plausible_correction(
                                current_val, needed, total_val)
                            if not plausible:
                                log.append({
                                    "phase": "phase3",
                                    "age": age_key, "col": col,
                                    "group": cg,
                                    "old": current_val, "new": needed,
                                    "method": "plausibility_blocked",
                                    "note": reason,
                                })
                                continue
                            old_val = current_val
                            cg_row[col] = needed
                            log.append({
                                "phase": "phase3",
                                "age": age_key, "col": col,
                                "group": cg,
                                "old": old_val, "new": needed,
                                "method": "candidate_search",
                                "note": (f"found in {raw_cands[needed]} "
                                         "readings"),
                            })
                            _cascade_row_constraint(cg_row, col, log,
                                                    age_key, cg)
                            fixed = True
                            break

                    # Also try the total group
                    if not fixed:
                        current_total = total_val
                        needed_total = comp_sum
                        raw_total_cands = (raw_lookups.get(total_group, {})
                                           .get(age_key, {})
                                           .get(col, Counter()))
                        if (needed_total in raw_total_cands
                                and needed_total != current_total):
                            total_row[col] = needed_total
                            log.append({
                                "phase": "phase3",
                                "age": age_key, "col": col,
                                "group": total_group,
                                "old": current_total, "new": needed_total,
                                "method": "candidate_search_total",
                                "note": (f"found in "
                                         f"{raw_total_cands[needed_total]} "
                                         "readings"),
                            })
                            _cascade_row_constraint(total_row, col, log,
                                                    age_key, total_group)
                            fixed = True

                    if not fixed:
                        log.append({
                            "phase": "phase3",
                            "age": age_key, "col": col,
                            "note": (f"unresolved: {total_group}="
                                     f"{total_val} != sum={comp_sum}, "
                                     f"diff={diff}"),
                        })

    # ── Phase 4: Re-verify and count remaining ───────────────────────
    remaining = 0
    for total_group, comp_groups in cross_group_constraints.items():
        total_lookup = lookups.get(total_group, {})
        if not total_lookup:
            continue
        for age_key in all_age_keys:
            for col in COLS:
                total_row = total_lookup.get(age_key)
                if total_row is None:
                    continue
                total_val = _safe_int(total_row.get(col))
                if total_val is None:
                    continue
                comp_sum = 0
                all_present = True
                for cg in comp_groups:
                    cg_row = lookups.get(cg, {}).get(age_key)
                    if cg_row is None or _safe_int(cg_row.get(col)) is None:
                        all_present = False
                        break
                    comp_sum += _safe_int(cg_row[col])
                if all_present and total_val != comp_sum:
                    remaining += 1

    log.append({
        "phase": "summary",
        "initial_discrepancies": len(discrepancies),
        "remaining_discrepancies": remaining,
        "corrections": sum(1 for e in log if "new" in e),
    })

    # Rebuild corrected from lookups
    for gname, rmap in lookups.items():
        if gname in corrected:
            for i, row in enumerate(corrected[gname]):
                key = _normalize_age(row.get("age", ""))
                if key and key in rmap:
                    corrected[gname][i] = rmap[key]

    return corrected, log


def _cascade_row_constraint(row, changed_col, log, age_key, group_name):
    """After changing M or F, cascade P = M + F within the row."""
    m = _safe_int(row.get("males"))
    f = _safe_int(row.get("females"))
    p = _safe_int(row.get("persons"))

    if changed_col in ("males", "females") and m is not None and f is not None:
        implied_p = m + f
        if p != implied_p:
            row["persons"] = implied_p
            log.append({
                "phase": "cascade",
                "age": age_key, "col": "persons",
                "group": group_name,
                "old": p, "new": implied_p,
                "method": "row_cascade",
                "note": "P=M+F after cross-group fix",
            })
    elif changed_col == "persons" and p is not None:
        if m is not None and f is None:
            implied_f = p - m
            if implied_f >= 0:
                row["females"] = implied_f
        elif f is not None and m is None:
            implied_m = p - f
            if implied_m >= 0:
                row["males"] = implied_m


# ---------------------------------------------------------------------------
# Self-test / demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")

    # --- Demo: majority_vote ---
    pass1 = [
        {"age": "0-5", "persons": 169509, "males": 81224, "females": 88285},
        {"age": "5-10", "persons": 140000, "males": 70000, "females": 70000},
    ]
    pass2 = [
        {"age": "0-5", "persons": 169509, "males": 81234, "females": 88285},
        {"age": "5-10", "persons": 140000, "males": 70000, "females": 70000},
    ]
    pass3 = [
        {"age": "0\u20135", "persons": 169509, "males": 81224, "females": 88285},
        {"age": "5-10", "persons": 140001, "males": 70000, "females": 70000},
    ]

    print("=== majority_vote ===")
    voted = majority_vote([pass1, pass2, pass3])
    for r in voted:
        print(f"  {r}")

    # --- Demo: digit_level_vote ---
    print("\n=== digit_level_vote ===")
    result = digit_level_vote([81234, 81224, 81224])
    print(f"  [81234, 81224, 81224] -> {result}")

    result2 = digit_level_vote([1234, 234, 1234])
    print(f"  [1234, 234, 1234] -> {result2}")

    # --- Demo: enforce_constraints ---
    print("\n=== enforce_constraints ===")
    bad_rows = [
        {"age": "0-5", "persons": 169510, "males": 81224, "females": 88285},
        {"age": "5-10", "persons": 140000, "males": 70000, "females": 70000},
        {"age": "Total", "persons": 309509, "males": 151224, "females": 158285},
    ]
    corrected, log = enforce_constraints(bad_rows)
    print("  Corrected rows:")
    for r in corrected:
        print(f"    {r}")
    print("  Corrections log:")
    for entry in log:
        print(f"    {entry}")

    # --- Demo: cross_model_ensemble ---
    print("\n=== cross_model_ensemble ===")
    openai_rows = [
        {"age": "0-5", "persons": 169509, "males": 81224, "females": 88285},
    ]
    gemini_rows = [
        {"age": "0-5", "persons": 169509, "males": 81234, "females": 88285},
    ]
    claude_rows = [
        {"age": "0-5", "persons": 169509, "males": 81224, "females": 88285},
    ]
    ensembled = cross_model_ensemble({
        "openai": openai_rows,
        "gemini": gemini_rows,
        "claude": claude_rows,
    })
    for r in ensembled:
        print(f"  {r}")

    print("\nAll demos passed.")
