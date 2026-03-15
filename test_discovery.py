"""
Test script for schema discovery phase — runs discover_schema() on each
test case and dumps the full TableSchema as pretty JSON for inspection.

Usage:
    python3 test_discovery.py                   # all 3 known test cases
    python3 test_discovery.py 0                 # just Travancore
    python3 test_discovery.py 0 1 2             # specific indices
    python3 test_discovery.py --image PATH      # any arbitrary image
"""

import sys
import json
import time
import dataclasses
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

from schema_discovery import discover_schema, schema_to_config, build_extraction_prompt, TableSchema
from pipeline import TEST_CASES, normalize_age

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def schema_to_dict(schema):
    """Convert TableSchema to a clean dict for JSON serialization."""
    return dataclasses.asdict(schema)


def test_discovery_on_image(image_path, label=""):
    """Run schema discovery on one image and return detailed results."""
    print(f"\n{'='*70}")
    print(f"SCHEMA DISCOVERY: {label or image_path}")
    print(f"{'='*70}")

    t0 = time.time()
    schema = discover_schema(image_path, model="gemini")
    elapsed = time.time() - t0

    if schema is None:
        print("  FAILED: Could not discover schema")
        return None

    print(f"\n  Completed in {elapsed:.1f}s")

    # Pretty-print the full schema
    schema_dict = schema_to_dict(schema)
    print(f"\n  --- Raw Schema JSON ---")
    print(json.dumps(schema_dict, indent=2))

    # Now show the derived config
    print(f"\n  --- Derived Config ---")
    config = schema_to_config(schema)
    config_display = {k: v for k, v in config.items() if k != "custom_prompt"}
    print(json.dumps(config_display, indent=2))

    # Show the custom prompt
    print(f"\n  --- Custom Extraction Prompt ---")
    print(config["custom_prompt"])

    # Save to results
    out_path = RESULTS_DIR / f"schema_{Path(image_path).stem}.json"
    output = {
        "image": str(image_path),
        "label": label,
        "discovery_time_s": round(elapsed, 1),
        "schema": schema_dict,
        "config": config_display,
        "custom_prompt": config["custom_prompt"],
    }
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved to {out_path}")

    return schema, config


def validate_against_known(schema, config, tc):
    """Compare discovered config against the hand-configured test case."""
    print(f"\n  --- Validation Against Known Config ---")
    issues = []

    # Check crop
    known_crop = (tc.get("preprocessing") or {}).get("crop")
    discovered_crop = (config.get("preprocessing") or {}).get("crop")

    if known_crop is None and discovered_crop is None:
        print(f"  Crop: MATCH (no crop needed)")
    elif known_crop is None and discovered_crop is not None:
        print(f"  Crop: EXTRA — discovered {discovered_crop} but none expected")
        issues.append("unnecessary crop")
    elif known_crop is not None and discovered_crop is None:
        print(f"  Crop: MISSING — expected {known_crop} but none discovered")
        issues.append("missing crop")
    else:
        # Both present — compare approximately
        for key in ("left_frac", "right_frac"):
            known_val = known_crop.get(key, 0.0 if "left" in key else 1.0)
            disc_val = discovered_crop.get(key, 0.0 if "left" in key else 1.0)
            diff = abs(known_val - disc_val)
            status = "OK" if diff < 0.15 else "MISMATCH"
            print(f"  Crop {key}: known={known_val:.2f} discovered={disc_val:.2f} "
                  f"(diff={diff:.2f}) {status}")
            if status == "MISMATCH":
                issues.append(f"crop {key} off by {diff:.2f}")

    # Check prompt type
    known_prompt = tc.get("prompt", "full")
    disc_prompt = config.get("prompt_type", "full")
    if known_prompt == disc_prompt:
        print(f"  Prompt type: MATCH ({disc_prompt})")
    else:
        print(f"  Prompt type: MISMATCH — known={known_prompt} discovered={disc_prompt}")
        issues.append(f"prompt type mismatch")

    # Check constraint groups
    known_groups = tc.get("constraint_groups")
    disc_groups = config.get("constraint_groups")
    if known_groups is None and disc_groups is None:
        print(f"  Constraint groups: MATCH (none)")
    elif known_groups is None and disc_groups is not None:
        print(f"  Constraint groups: EXTRA — discovered {list(disc_groups.keys())}")
        issues.append("unnecessary constraint groups")
    elif known_groups is not None and disc_groups is None:
        print(f"  Constraint groups: MISSING — expected {list(known_groups.keys())}")
        issues.append("missing constraint groups")
    else:
        known_keys = set(known_groups.keys())
        disc_keys = set(disc_groups.keys())
        matched = known_keys & disc_keys
        missing = known_keys - disc_keys
        extra = disc_keys - known_keys
        print(f"  Constraint groups: {len(matched)} matched, "
              f"{len(missing)} missing, {len(extra)} extra")
        if missing:
            print(f"    Missing: {missing}")
            issues.append(f"missing constraint groups: {missing}")
        if extra:
            print(f"    Extra: {extra}")
        # Check components for matched groups
        for key in sorted(matched):
            known_comps = set(known_groups[key])
            disc_comps = set(disc_groups[key])
            if known_comps == disc_comps:
                print(f"    {key}: components MATCH")
            else:
                diff = known_comps.symmetric_difference(disc_comps)
                print(f"    {key}: components DIFFER — delta={diff}")
                issues.append(f"constraint group {key} components differ")

    # Check known totals
    known_kt = tc.get("known_totals")
    disc_kt = config.get("known_totals")
    if known_kt is None and disc_kt is None:
        print(f"  Known totals: MATCH (none)")
    elif known_kt is not None and disc_kt is not None:
        if known_kt == disc_kt:
            print(f"  Known totals: MATCH ({disc_kt})")
        else:
            print(f"  Known totals: DIFFER — known={known_kt} discovered={disc_kt}")
            issues.append("known totals differ")
    else:
        print(f"  Known totals: MISMATCH — known={known_kt} discovered={disc_kt}")
        issues.append("known totals mismatch")

    # Summary
    if issues:
        print(f"\n  ISSUES ({len(issues)}):")
        for issue in issues:
            print(f"    - {issue}")
    else:
        print(f"\n  ALL CHECKS PASSED")

    return issues


def main():
    args = sys.argv[1:]

    if "--image" in args:
        idx = args.index("--image")
        image_path = args[idx + 1]
        test_discovery_on_image(image_path)
        return

    if args:
        indices = [int(a) for a in args]
    else:
        indices = list(range(len(TEST_CASES)))

    all_issues = {}
    for i in indices:
        tc = TEST_CASES[i]
        result = test_discovery_on_image(tc["image_path"], tc["name"])
        if result is not None:
            schema, config = result
            issues = validate_against_known(schema, config, tc)
            all_issues[tc["name"]] = issues

    # Final summary
    print(f"\n\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    for name, issues in all_issues.items():
        status = "PASS" if not issues else f"FAIL ({len(issues)} issues)"
        print(f"  {name}: {status}")


if __name__ == "__main__":
    main()
