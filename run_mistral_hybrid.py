"""Batch runner for Mistral OCR hybrid pipeline.

Loops over all 52 *_oneshot.json results, extracts via Mistral hybrid pipeline,
and scores against the existing oneshot results (ground truth).

Usage:
    python3 run_mistral_hybrid.py                  # run all 52
    python3 run_mistral_hybrid.py --dry-run         # parse-only from cached raw text
    python3 run_mistral_hybrid.py --filter 1931     # only tables matching pattern
    python3 run_mistral_hybrid.py --full-fallback    # enable Gemini fallback
"""

import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from pipeline import normalize_age

PROJECT_ROOT = Path(__file__).parent
RESULTS_DIR = PROJECT_ROOT / "results"
BASELINE_DIR = RESULTS_DIR / "baselines"
BASELINE_DIR.mkdir(exist_ok=True)


def load_gt_cells(oneshot_json):
    """Extract ground truth cells from an existing oneshot result JSON.

    Returns list of dicts: {section, age_norm, group, col, value}
    """
    cells = []
    for sec in oneshot_json.get("data", []):
        sec_name = sec.get("name", "")
        for row in sec.get("rows", []):
            age = row.get("age", "")
            age_norm = normalize_age(age)
            for k, v in row.items():
                if k == "age" or not isinstance(v, dict):
                    continue
                for col, val in v.items():
                    if val is not None:
                        cells.append({
                            "section": sec_name,
                            "age_norm": age_norm,
                            "group": k,
                            "col": col,
                            "value": int(val),
                        })
    return cells


def extract_predicted_cells(hybrid_json):
    """Extract predicted cells from hybrid pipeline result JSON.

    Returns dict: (section_norm, age_norm, group_norm, col) → value
    """
    cells = {}
    for sec in hybrid_json.get("data", []):
        sec_name = sec.get("name", "")
        for row in sec.get("rows", []):
            age = row.get("age", "")
            age_norm = normalize_age(age)
            for k, v in row.items():
                if k == "age" or not isinstance(v, dict):
                    continue
                for col, val in v.items():
                    if val is not None:
                        cells[(sec_name, age_norm, k, col)] = int(val)
    return cells


def score_hybrid(gt_cells, pred_cells):
    """Score predicted cells against ground truth.

    Uses two matching strategies:
    1. Exact match: (section, age_norm, group, col) matches exactly
    2. Generous match: (age_norm, group_norm, col) matches (ignoring section name differences)

    Returns (exact_matched, generous_matched, total, mismatches)
    """
    exact = 0
    generous = 0
    total = len(gt_cells)
    mismatches = []

    def _norm_group(g):
        return g.lower().strip(". ")

    # Build generous lookup: (age_norm, group_norm, col) → set of values
    generous_lookup = {}
    for key, val in pred_cells.items():
        _, age_norm, group, col = key
        gkey = (age_norm, _norm_group(group), col)
        if gkey not in generous_lookup:
            generous_lookup[gkey] = set()
        generous_lookup[gkey].add(val)

    for gt in gt_cells:
        # Exact match
        exact_key = (gt["section"], gt["age_norm"], gt["group"], gt["col"])
        if exact_key in pred_cells and pred_cells[exact_key] == gt["value"]:
            exact += 1
            generous += 1
            continue

        # Generous match (ignore section name mismatch, normalize group name)
        gkey = (gt["age_norm"], _norm_group(gt["group"]), gt["col"])
        if gkey in generous_lookup and gt["value"] in generous_lookup[gkey]:
            generous += 1
            continue

        mismatches.append(gt)

    return exact, generous, total, mismatches


def run_parse_only(name, raw_text_path):
    """Parse cached Mistral raw text + run constraints (no API calls)."""
    from oneshot_mistral import (
        parse_mistral_markdown,
        infer_schema_from_parsed,
        _validate_parse,
    )
    from oneshot import (
        derive_constraints,
        verify_all_constraints,
        _detect_and_fix_mf_swaps,
        _deductive_digit_fix,
    )

    raw_text = raw_text_path.read_text()
    parsed = parse_mistral_markdown(raw_text)

    is_valid, reason = _validate_parse(parsed)
    if not is_valid:
        return None, f"parse failed: {reason}"

    schema = infer_schema_from_parsed(parsed)

    persons_independent = False
    if schema.data_type == "proportional":
        title_lower = (schema.title or "").lower()
        if "of each sex" in title_lower or "each sex" in title_lower:
            persons_independent = True

    constraints = derive_constraints(parsed, schema=schema,
                                     persons_independent=persons_independent)
    report = verify_all_constraints(parsed, constraints)

    # Try free repair
    if not report["all_passed"]:
        for _ in range(3):
            parsed, _ = _detect_and_fix_mf_swaps(
                parsed, constraints, persons_independent)
            parsed, _ = _deductive_digit_fix(
                parsed, constraints, persons_independent)
            constraints = derive_constraints(parsed, schema=schema,
                                            persons_independent=persons_independent)
            report = verify_all_constraints(parsed, constraints)
            if report["all_passed"]:
                break

    result = {
        "metadata": parsed.get("metadata", {}),
        "data": parsed["sections"],
        "constraints": {
            "total_checks": report["total_checks"],
            "passed": report["passed"],
            "failed": report["failed"],
            "all_passed": report["all_passed"],
        },
    }
    return result, "ok"


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Batch Mistral hybrid runner")
    parser.add_argument("--dry-run", action="store_true",
                       help="Parse-only from cached raw text (no API calls)")
    parser.add_argument("--filter", default="",
                       help="Only process tables matching pattern")
    parser.add_argument("--full-fallback", action="store_true",
                       help="Enable full Gemini fallback")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Show per-cell mismatches")
    args = parser.parse_args()

    # Collect all oneshot results as ground truth
    gt_files = sorted(RESULTS_DIR.glob("*_oneshot.json"))
    if not gt_files:
        print("ERROR: No *_oneshot.json files found in results/")
        return

    print(f"Found {len(gt_files)} oneshot result files\n")

    total_gt_cells = 0
    total_exact = 0
    total_generous = 0
    total_constraints_passed = 0
    total_constraints_checked = 0
    total_cost = 0.0
    total_mistral = 0
    total_gemini = 0
    results_summary = []

    perfect_count = 0
    failed_tables = []

    for fi, gt_file in enumerate(gt_files):
        name = gt_file.stem.replace("_oneshot", "")

        if args.filter and args.filter not in name:
            continue

        # Load ground truth
        with open(gt_file) as f:
            gt_json = json.load(f)
        gt_cells = load_gt_cells(gt_json)
        if not gt_cells:
            continue

        # Get source image path
        src = gt_json.get("source_image", "")
        if not src:
            continue
        img_path = Path(src) if Path(src).is_absolute() else PROJECT_ROOT / src
        if not img_path.exists():
            img_path = PROJECT_ROOT / src
        if not img_path.exists():
            print(f"  [{fi+1:2d}/{len(gt_files)}] {name}: IMAGE NOT FOUND")
            continue

        if args.dry_run:
            # Use cached Mistral raw text
            raw_path = BASELINE_DIR / f"{name}_mistral_raw.txt"
            if not raw_path.exists():
                # Also check the hybrid raw text
                raw_path = RESULTS_DIR / f"{name}_mistral_hybrid_raw.txt"
            if not raw_path.exists():
                print(f"  [{fi+1:2d}/{len(gt_files)}] {name}: NO CACHED RAW TEXT")
                results_summary.append({
                    "name": name, "status": "no_cache",
                    "exact": 0, "generous": 0, "total": len(gt_cells),
                })
                total_gt_cells += len(gt_cells)
                failed_tables.append(name)
                continue

            t0 = time.time()
            hybrid_result, status = run_parse_only(name, raw_path)
            elapsed = time.time() - t0

            if hybrid_result is None:
                print(f"  [{fi+1:2d}/{len(gt_files)}] {name}: {status}")
                results_summary.append({
                    "name": name, "status": status,
                    "exact": 0, "generous": 0, "total": len(gt_cells),
                })
                total_gt_cells += len(gt_cells)
                failed_tables.append(name)
                continue

            pred_cells = extract_predicted_cells(hybrid_result)
            exact, generous, total, mismatches = score_hybrid(gt_cells, pred_cells)
            cpass = hybrid_result["constraints"]["all_passed"]
            cfailed = hybrid_result["constraints"]["failed"]
            cchecked = hybrid_result["constraints"]["total_checks"]

        else:
            # Full pipeline with API calls
            from oneshot_mistral import extract_and_verify_mistral

            t0 = time.time()
            result = extract_and_verify_mistral(
                img_path, full_fallback=args.full_fallback)
            elapsed = time.time() - t0

            if result is None:
                print(f"  [{fi+1:2d}/{len(gt_files)}] {name}: EXTRACTION FAILED")
                results_summary.append({
                    "name": name, "status": "failed",
                    "exact": 0, "generous": 0, "total": len(gt_cells),
                })
                total_gt_cells += len(gt_cells)
                failed_tables.append(name)
                continue

            # Load the saved hybrid JSON for scoring
            hybrid_json_path = RESULTS_DIR / f"{name}_mistral_hybrid.json"
            if hybrid_json_path.exists():
                with open(hybrid_json_path) as f:
                    hybrid_result = json.load(f)
            else:
                hybrid_result = {
                    "metadata": result["parsed"].get("metadata", {}),
                    "data": result["parsed"]["sections"],
                    "constraints": {
                        "total_checks": result["report"]["total_checks"],
                        "passed": result["report"]["passed"],
                        "failed": result["report"]["failed"],
                        "all_passed": result["report"]["all_passed"],
                    },
                }

            pred_cells = extract_predicted_cells(hybrid_result)
            exact, generous, total, mismatches = score_hybrid(gt_cells, pred_cells)
            cpass = result["report"]["all_passed"]
            cfailed = result["report"]["failed"]
            cchecked = result["report"]["total_checks"]
            total_mistral += result.get("mistral_calls", 0)
            total_gemini += result.get("gemini_calls", 0)
            total_cost += result.get("cost", 0)

        # Update totals
        total_gt_cells += total
        total_exact += exact
        total_generous += generous
        total_constraints_checked += cchecked
        if cpass:
            total_constraints_passed += 1

        exact_pct = exact / total * 100 if total else 0
        generous_pct = generous / total * 100 if total else 0

        is_perfect = generous == total
        if is_perfect:
            perfect_count += 1
        else:
            failed_tables.append(name)

        status_str = "PERFECT" if is_perfect else f"MISS {total - generous}"
        constraint_str = "PASS" if cpass else f"FAIL({cfailed})"

        print(f"  [{fi+1:2d}/{len(gt_files)}] {name}: "
              f"{generous}/{total} ({generous_pct:.1f}%) "
              f"[{constraint_str}] "
              f"{status_str} "
              f"[{elapsed:.1f}s]")

        if args.verbose and mismatches:
            for m in mismatches[:10]:
                print(f"    MISS: {m['section']}/{m['age_norm']} "
                      f"{m['group']}.{m['col']}={m['value']}")
            if len(mismatches) > 10:
                print(f"    ... and {len(mismatches)-10} more")

        results_summary.append({
            "name": name,
            "status": "perfect" if is_perfect else "partial",
            "exact": exact,
            "generous": generous,
            "total": total,
            "exact_pct": round(exact_pct, 1),
            "generous_pct": round(generous_pct, 1),
            "constraints_pass": cpass,
            "constraints_failed": cfailed,
            "elapsed": round(elapsed, 1),
        })

    # Summary
    print(f"\n{'='*70}")
    print(f"MISTRAL HYBRID SUMMARY")
    print(f"{'='*70}")
    exact_pct = total_exact / total_gt_cells * 100 if total_gt_cells else 0
    generous_pct = total_generous / total_gt_cells * 100 if total_gt_cells else 0
    n_tables = len(results_summary)
    print(f"Tables: {n_tables}")
    print(f"Perfect: {perfect_count}/{n_tables} "
          f"({perfect_count/n_tables*100:.0f}%)" if n_tables else "")
    print(f"Exact accuracy:    {total_exact}/{total_gt_cells} ({exact_pct:.1f}%)")
    print(f"Generous accuracy: {total_generous}/{total_gt_cells} ({generous_pct:.1f}%)")
    print(f"Constraints: {total_constraints_passed}/{n_tables} tables all-pass")

    if not args.dry_run:
        print(f"API calls: Mistral={total_mistral}, Gemini={total_gemini}")
        print(f"Est. cost: ${total_cost:.2f}")

    if failed_tables:
        print(f"\nFailed tables ({len(failed_tables)}):")
        for t in failed_tables:
            entry = next((r for r in results_summary if r["name"] == t), None)
            if entry:
                print(f"  {t}: {entry.get('generous', 0)}/{entry.get('total', '?')} "
                      f"({entry.get('generous_pct', 0):.1f}%)")

    # Save summary
    summary_path = BASELINE_DIR / "_mistral_hybrid_summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "total_tables": n_tables,
            "perfect_tables": perfect_count,
            "total_exact": total_exact,
            "total_generous": total_generous,
            "total_cells": total_gt_cells,
            "exact_pct": round(exact_pct, 2),
            "generous_pct": round(generous_pct, 2),
            "per_table": results_summary,
        }, f, indent=2)
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
