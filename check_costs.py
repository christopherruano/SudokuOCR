"""Tally estimated API costs from batch logs and OCR results.

Run on FASRC:
    python3 check_costs.py
"""

import json
import os
import glob

RESULTS_ROOT = os.path.expanduser("~/census_toc/ocr_results")
COST_PER_CALL = 0.0025  # USD per Gemini API call (rough)

# 1. Sum from batch logs (most accurate — tracks actual API calls)
log_pattern = os.path.join(RESULTS_ROOT, "_batch_log*.jsonl")
log_files = sorted(glob.glob(log_pattern))

total_api_calls = 0
total_images = 0
by_status = {}

for lf in log_files:
    with open(lf) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            total_images += 1
            calls = rec.get("api_calls", 0)
            if not calls:
                s = rec.get("status", "")
                calls = 2 if s in ("accepted_high_confidence",
                                    "accepted_with_risk",
                                    "has_failures") else 1
            total_api_calls += calls
            st = rec.get("status", "unknown")
            by_status[st] = by_status.get(st, 0) + 1

print("=" * 60)
print("COST ESTIMATE FROM BATCH LOGS")
print("=" * 60)
print(f"Batch log files found: {len(log_files)}")
print(f"Total image records:   {total_images:,}")
print(f"Total API calls:       {total_api_calls:,}")
print(f"Cost per call:         ${COST_PER_CALL}")
print(f"Estimated spend:       ${total_api_calls * COST_PER_CALL:,.2f}")
print()
print("By status:")
for st in sorted(by_status, key=by_status.get, reverse=True):
    print(f"  {st:<35} {by_status[st]:>6}")

# 2. Count actual result JSONs on disk
print()
print("=" * 60)
print("RESULT FILES ON DISK")
print("=" * 60)

n_results = 0
n_validated = 0
n_failed = 0
n_unvalidated = 0

for dirpath, _, filenames in os.walk(RESULTS_ROOT):
    for fn in filenames:
        if not fn.endswith("_oneshot.json"):
            continue
        n_results += 1
        try:
            d = json.load(open(os.path.join(dirpath, fn)))
            c = d.get("constraints", {})
            checks = c.get("total_checks", 0)
            failed = c.get("failed", 0)
            if checks > 0 and failed == 0:
                n_validated += 1
            elif failed > 0:
                n_failed += 1
            else:
                n_unvalidated += 1
        except Exception:
            pass

print(f"Total result JSONs:    {n_results:,}")
print(f"  Validated:           {n_validated:,}")
print(f"  Constraint failures: {n_failed:,}")
print(f"  Unvalidated:         {n_unvalidated:,}")

# 3. Overall summary
print()
print("=" * 60)
print("SUMMARY")
print("=" * 60)
# If batch logs undercount (e.g., missing chunk logs), estimate from results
if n_results > total_images:
    est_calls = n_results * 2.5
    print(f"Batch logs may be incomplete ({total_images:,} logged vs "
          f"{n_results:,} results)")
    print(f"Estimate from result count: {n_results:,} images x 2.5 calls "
          f"= ~${est_calls * COST_PER_CALL:,.2f}")
else:
    print(f"Estimated total spend: ${total_api_calls * COST_PER_CALL:,.2f}")
print(f"Cost per table: ~${COST_PER_CALL * 2.5:.4f}")
