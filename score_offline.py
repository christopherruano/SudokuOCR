"""Offline scorer: compare saved oneshot JSON results against ground truth.

No API calls — reads existing result files from results/ directory.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from pipeline import normalize_age, score
from test_oneshot_gt import load_gt_sheet

PROJECT_ROOT = Path(__file__).parent
RESULTS_DIR = PROJECT_ROOT / "results"


def load_result_json(filename):
    """Load a oneshot result JSON file."""
    path = RESULTS_DIR / filename
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def extract_predicted_for_group(result, group_name):
    """Extract predicted rows for a group from result JSON.

    The JSON has: data[section_idx].rows[i].{group_name}.{persons,males,females}
    """
    data = result.get("data", [])
    if not data:
        return []

    section = data[0]  # first section
    groups = result.get("metadata", {}).get("column_groups", [])

    # Find matching group name (case-insensitive, strip punctuation)
    target_group = None
    gn_clean = str(group_name).lower().strip(". ")

    for g in groups:
        if g.lower().strip(". ") == gn_clean:
            target_group = g
            break

    # Try year-as-int match
    if target_group is None:
        for g in groups:
            if str(group_name).strip() == str(g).strip(". "):
                target_group = g
                break

    if target_group is None:
        return []

    rows = []
    for row in section.get("rows", []):
        gdata = row.get(target_group, {})
        if not isinstance(gdata, dict):
            continue
        rows.append({
            "age": row.get("age", ""),
            "persons": gdata.get("persons"),
            "males": gdata.get("males"),
            "females": gdata.get("females"),
        })
    return rows


EVAL_CASES = [
    {
        "name": "Travancore Eastern 1901",
        "result_file": "Eastern_division_age_1901_oneshot.json",
        "gt_sheet": "Sheet1",
        "has_persons": True,
    },
    {
        "name": "Hyderabad State 1901",
        "result_file": "Hyderabad_state_summary_age_1901_oneshot.json",
        "gt_sheet": "Sheet2",
        "has_persons": True,
    },
    {
        "name": "Coorg 1901",
        "result_file": "Coorg_age_1901_oneshot.json",
        "gt_sheet": "Sheet3",
        "has_persons": False,
    },
]


def main():
    results = []

    for tc in EVAL_CASES:
        print(f"\n{'#'*70}")
        print(f"# EVAL: {tc['name']}")
        print(f"{'#'*70}")

        result = load_result_json(tc["result_file"])
        if result is None:
            print(f"  RESULT FILE NOT FOUND: {tc['result_file']}")
            results.append({"name": tc["name"], "status": "MISSING"})
            continue

        # Show constraint results from the file
        constraints = result.get("constraints", {})
        print(f"  Constraints: {constraints.get('passed', '?')}/{constraints.get('total_checks', '?')} "
              f"({'ALL PASS' if constraints.get('all_passed') else 'FAILURES'})")
        if constraints.get("failures"):
            for f in constraints["failures"][:5]:
                print(f"    FAIL: {f}")

        # Load GT
        gt_all = load_gt_sheet(tc["gt_sheet"], has_persons=tc["has_persons"])

        # Score each group
        total_exact = 0
        total_cells = 0
        all_errors = []
        group_scores = {}

        for gt_group_name, gt_rows in gt_all.items():
            predicted = extract_predicted_for_group(result, gt_group_name)

            if not predicted:
                print(f"  WARNING: No predicted rows for group '{gt_group_name}'")
                for row in gt_rows:
                    for col in ("persons", "males", "females"):
                        if row.get(col) is not None:
                            total_cells += 1
                group_scores[gt_group_name] = {"exact": 0, "total": 0, "pct": "N/A"}
                continue

            sc = score(predicted, gt_rows)
            total_exact += sc["exact"]
            total_cells += sc["total"]
            all_errors.extend(
                {**e, "group": gt_group_name} for e in sc.get("errors", [])
            )
            group_scores[gt_group_name] = {
                "exact": sc["exact"],
                "total": sc["total"],
                "pct": f"{sc['exact_match_rate']:.1%}",
            }

        overall_pct = total_exact / total_cells if total_cells else 0

        print(f"  GT SCORING: {total_exact}/{total_cells} = {overall_pct:.1%}")
        for gname, gs in group_scores.items():
            print(f"    {gname}: {gs['exact']}/{gs['total']} ({gs['pct']})")

        if all_errors:
            print(f"  ERRORS ({len(all_errors)}):")
            for e in all_errors[:20]:
                print(f"    [{e.get('group', '?')}] {e['age']} {e['col']}: "
                      f"pred={e['pred']} gt={e['gt']} err={e['err']}")

        results.append({
            "name": tc["name"],
            "gt_score": f"{total_exact}/{total_cells}",
            "gt_pct": f"{overall_pct:.1%}",
            "constraints": f"{constraints.get('passed', '?')}/{constraints.get('total_checks', '?')}",
            "all_passed": constraints.get("all_passed"),
            "errors": all_errors,
        })

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    grand_exact = 0
    grand_total = 0
    for r in results:
        if r.get("status") == "MISSING":
            print(f"  {r['name']}: RESULT FILE MISSING")
        else:
            err_count = len(r.get("errors", []))
            status = "PERFECT" if r["gt_pct"] == "100.0%" else f"{err_count} errors"
            print(f"  {r['name']}: {r['gt_score']} ({r['gt_pct']}) "
                  f"constraints={r['constraints']} [{status}]")
            parts = r["gt_score"].split("/")
            grand_exact += int(parts[0])
            grand_total += int(parts[1])

    if grand_total:
        print(f"\n  OVERALL: {grand_exact}/{grand_total} "
              f"({grand_exact/grand_total:.1%})")


if __name__ == "__main__":
    main()
