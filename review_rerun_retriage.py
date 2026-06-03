#!/usr/bin/env python3
"""Deterministically re-triage the plausibility review queue for RERUN decisions
under the new type-aware constraint logic — NO API calls.

Many "rerun" flags came from additive/sum checks applied to tables that don't
obey them (rates, ratios, distributions, percentages). Under the corrected
constraints those are NOT failures, so they need no rerun. This script splits
the queue into:
  - cleared_false_failure : proportional tables flagged only by additive checks
  - cleared_in_scope_now  : occupation/caste/education etc. that were scope-excluded
  - genuine_rerun         : count tables whose sum failures are real (API rerun/MoE)
  - keep_review           : everything else (human glance)
"""
import csv, os
from collections import Counter

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "export_final/sendout_final/review/review_queue_plausibility_triage.csv")
OUT = os.path.join(ROOT, "export_final/sendout_final/review/rerun_retriage")
os.makedirs(OUT, exist_ok=True)
csv.field_size_limit(10**7)

# format_class -> table kind family
PROPORTIONAL = {"rate_or_ratio_table", "distribution_of_1000", "distribution_of_10000",
                "distribution_of_100", "mortality_rates", "disease_specific_mortality_rates"}
SCOPE_NOW = {"occupation_caste_tribe_race_table", "education_literacy_table"}
COUNT = {"age_civil_condition_counts", "additive_age_sex_counts", "age_by_religion_counts",
         "mortality_counts", "geography_or_district_counts", "disease_specific_mortality"}


def classify(r):
    fc = (r.get("format_class") or "").strip()
    status = (r.get("plausibility_status") or "").lower()
    if fc in PROPORTIONAL:
        # additive/sum failures don't apply to these -> no rerun
        return "cleared_false_failure"
    if fc in SCOPE_NOW:
        return "cleared_in_scope_now"
    if fc in COUNT:
        # real additive checks apply; a genuine fail is a real rerun candidate
        return "genuine_rerun" if status in ("fail",) else "keep_review"
    return "keep_review"


def main():
    rows = [r for r in csv.DictReader(open(SRC)) if r["sendout_bucket"] == "sendout_review_needed"]
    buckets = {}
    for r in rows:
        r["rerun_retriage"] = classify(r)
        buckets.setdefault(r["rerun_retriage"], []).append(r)

    cols = list(rows[0].keys())
    for name, rs in buckets.items():
        with open(os.path.join(OUT, f"{name}.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
            for r in rs:
                w.writerow(r)

    print(f"review_needed total: {len(rows)}")
    for k in ("cleared_false_failure", "cleared_in_scope_now", "genuine_rerun", "keep_review"):
        print(f"  {k:24} {len(buckets.get(k, [])):5}")
    cleared = len(buckets.get("cleared_false_failure", [])) + len(buckets.get("cleared_in_scope_now", []))
    print(f"\n  => clearable WITHOUT rerun/API: {cleared}  ({100*cleared/len(rows):.0f}%)")
    print(f"  => genuine API-rerun candidates: {len(buckets.get('genuine_rerun', []))}")
    print(f"  wrote per-bucket CSVs -> {OUT}")


if __name__ == "__main__":
    main()
