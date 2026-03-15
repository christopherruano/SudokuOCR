"""
Test schema discovery robustness: run multiple models on each image,
compare outputs, identify disagreements.

Usage:
    python3 test_schema_robustness.py                   # all 3 test cases
    python3 test_schema_robustness.py 0                 # just test case 0
    python3 test_schema_robustness.py --image PATH      # arbitrary image
"""

import sys
import os
import json
import time
import dataclasses
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from dotenv import load_dotenv
load_dotenv()

from schema_discovery import (
    discover_schema, schema_to_config, TableSchema,
    SCHEMA_DISCOVERY_PROMPT, _parse_schema_json, _dict_to_schema
)
from pipeline import TEST_CASES, MODELS, encode_image, normalize_age

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def run_discovery_all_models(image_path, label=""):
    """Run schema discovery with each available model independently."""
    b64 = encode_image(str(image_path))

    env_keys = {"openai": "OPENAI_API_KEY", "gemini": "GEMINI_API_KEY",
                "claude": "ANTHROPIC_API_KEY"}
    available = [m for m in MODELS if os.environ.get(env_keys[m])]

    results = {}
    for model in available:
        print(f"\n  [{model}] Discovering schema...")
        t0 = time.time()
        for attempt in range(3):
            try:
                raw = MODELS[model](b64, SCHEMA_DISCOVERY_PROMPT)
                parsed = _parse_schema_json(raw)
                if parsed is not None:
                    schema = _dict_to_schema(parsed)
                    elapsed = time.time() - t0
                    results[model] = {
                        "schema": schema,
                        "raw_dict": parsed,
                        "time": round(elapsed, 1),
                    }
                    print(f"    OK ({elapsed:.1f}s)")
                    break
                else:
                    print(f"    Parse failed, attempt {attempt+1}")
            except Exception as e:
                print(f"    Error attempt {attempt+1}: {e}")
                if attempt < 2:
                    time.sleep(2 ** attempt)
        else:
            print(f"    FAILED after 3 attempts")

    return results


def compare_schemas(model_results):
    """Compare schema outputs across models, identify agreements and disagreements."""
    models = list(model_results.keys())
    if len(models) < 2:
        print("  Only 1 model available, nothing to compare")
        return {}

    comparisons = {}

    # Compare key fields
    fields_to_compare = [
        ("data_type", "Data type"),
        ("denominator", "Denominator"),
        ("has_persons_column", "Has persons column"),
        ("multi_year", "Multi-year"),
        ("target_group_index", "Target group index"),
    ]

    for field, label in fields_to_compare:
        values = {}
        for m in models:
            schema = model_results[m]["schema"]
            values[m] = getattr(schema, field, None)

        unique = set(str(v) for v in values.values())
        if len(unique) == 1:
            comparisons[field] = {"status": "AGREE", "value": list(values.values())[0]}
        else:
            comparisons[field] = {"status": "DISAGREE", "values": values}
            print(f"  DISAGREE on {label}: {values}")

    # Compare row labels
    row_sets = {}
    for m in models:
        labels = model_results[m]["schema"].row_labels
        # Normalize for comparison
        norm = [normalize_age(l) for l in labels if normalize_age(l)]
        row_sets[m] = set(norm)

    all_rows = set()
    for s in row_sets.values():
        all_rows |= s

    # Find rows present in all vs some
    unanimous = all_rows.copy()
    for s in row_sets.values():
        unanimous &= s

    partial = all_rows - unanimous
    if partial:
        comparisons["row_labels"] = {
            "status": "PARTIAL_DISAGREE",
            "unanimous": sorted(unanimous),
            "disputed": {r: [m for m in models if r in row_sets[m]]
                         for r in sorted(partial)},
        }
        print(f"  Row labels: {len(unanimous)} unanimous, {len(partial)} disputed")
        for r, ms in sorted(comparisons["row_labels"]["disputed"].items()):
            print(f"    '{r}': only in {ms}")
    else:
        comparisons["row_labels"] = {"status": "AGREE", "count": len(unanimous)}

    # Compare subtotal hierarchy
    hier_sets = {}
    for m in models:
        hier = model_results[m]["schema"].subtotal_hierarchy
        # Normalize keys and values
        norm_hier = {}
        for k, v in hier.items():
            nk = normalize_age(k)
            nv = tuple(sorted(normalize_age(c) for c in v))
            norm_hier[nk] = nv
        hier_sets[m] = norm_hier

    all_keys = set()
    for h in hier_sets.values():
        all_keys |= set(h.keys())

    hier_agree = {}
    hier_disagree = {}
    for key in sorted(all_keys):
        values = {}
        for m in models:
            if key in hier_sets[m]:
                values[m] = hier_sets[m][key]

        if len(values) < len(models):
            # Not all models found this group
            hier_disagree[key] = {"present_in": list(values.keys()),
                                   "components": {m: list(v) for m, v in values.items()}}
        else:
            unique_vals = set(values.values())
            if len(unique_vals) == 1:
                hier_agree[key] = list(list(values.values())[0])
            else:
                hier_disagree[key] = {"components": {m: list(v) for m, v in values.items()}}

    if hier_disagree:
        comparisons["hierarchy"] = {
            "status": "PARTIAL_DISAGREE",
            "agreed": hier_agree,
            "disagreed": hier_disagree,
        }
        print(f"  Hierarchy: {len(hier_agree)} groups agree, {len(hier_disagree)} disagree")
        for k, info in hier_disagree.items():
            print(f"    '{k}': {info}")
    else:
        comparisons["hierarchy"] = {"status": "AGREE", "groups": hier_agree}

    # Compare column groups (crop positions)
    crop_configs = {}
    for m in models:
        config = schema_to_config(model_results[m]["schema"])
        crop = (config.get("preprocessing") or {}).get("crop")
        crop_configs[m] = crop

    if all(c is None for c in crop_configs.values()):
        comparisons["crop"] = {"status": "AGREE", "value": "no crop"}
    elif all(c is not None for c in crop_configs.values()):
        rights = {m: c.get("right_frac", 1.0) for m, c in crop_configs.items()}
        spread = max(rights.values()) - min(rights.values())
        if spread < 0.10:
            comparisons["crop"] = {"status": "AGREE", "right_fracs": rights,
                                   "spread": round(spread, 3)}
        else:
            comparisons["crop"] = {"status": "DISAGREE", "right_fracs": rights,
                                   "spread": round(spread, 3)}
            print(f"  DISAGREE on crop: {rights} (spread={spread:.3f})")
    else:
        comparisons["crop"] = {"status": "DISAGREE", "configs": {
            m: str(c) for m, c in crop_configs.items()}}
        print(f"  DISAGREE on crop: some crop, some don't")

    return comparisons


def run_test(image_path, label=""):
    print(f"\n{'='*70}")
    print(f"ROBUSTNESS TEST: {label or image_path}")
    print(f"{'='*70}")

    results = run_discovery_all_models(image_path, label)

    if len(results) < 2:
        print("\n  Cannot compare — fewer than 2 models succeeded")
        return results, {}

    print(f"\n--- Cross-Model Comparison ---")
    comparisons = compare_schemas(results)

    # Summary
    n_agree = sum(1 for v in comparisons.values() if v.get("status") == "AGREE")
    n_total = len(comparisons)
    print(f"\n  SUMMARY: {n_agree}/{n_total} fields agree across all models")

    return results, comparisons


def main():
    args = sys.argv[1:]

    if "--image" in args:
        idx = args.index("--image")
        image_path = args[idx + 1]
        run_test(image_path)
        return

    if args:
        indices = [int(a) for a in args]
    else:
        indices = list(range(len(TEST_CASES)))

    all_comparisons = {}
    for i in indices:
        tc = TEST_CASES[i]
        results, comparisons = run_test(tc["image_path"], tc["name"])
        all_comparisons[tc["name"]] = comparisons

    # Final summary
    print(f"\n\n{'='*70}")
    print("CROSS-IMAGE SUMMARY")
    print(f"{'='*70}")
    for name, comp in all_comparisons.items():
        disagrees = [k for k, v in comp.items() if v.get("status") != "AGREE"]
        if disagrees:
            print(f"  {name}: DISAGREEMENTS on {disagrees}")
        else:
            print(f"  {name}: ALL AGREE")


if __name__ == "__main__":
    main()
