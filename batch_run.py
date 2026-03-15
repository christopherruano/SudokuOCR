"""Batch runner: process all census PDF pages through the oneshot pipeline.

Usage:
    python3 batch_run.py                    # run all unprocessed pages
    python3 batch_run.py --province Mysore  # run just one province
    python3 batch_run.py --dry-run          # show what would run
"""

import os
import sys
import json
import time
import argparse
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
RESULTS_DIR = PROJECT_ROOT / "results"
AGE_TABLES = PROJECT_ROOT / "age_tables"

# High-value targets ordered by GT cells per image (descending)
TARGETS = [
    # 1891 multi-group (4 groups = 156 GT cells/district)
    {"province": "Mysore", "year": "1891", "gt_cells_per_district": 156},
    {"province": "Assam", "year": "1891", "gt_cells_per_district": 156},
    {"province": "Bombay", "year": "1891", "gt_cells_per_district": 156},
    {"province": "Burma", "year": "1891", "gt_cells_per_district": 156},
    {"province": "Berar", "year": "1901", "gt_cells_per_district": 84},
    # 1901 population-only
    {"province": "Madras", "year": "1891", "gt_cells_per_district": 39},
    {"province": "Bengal", "year": "1901", "gt_cells_per_district": 21},
    {"province": "North_Western_Provinces_Oudh", "year": "1891", "gt_cells_per_district": 39},
    {"province": "Punjab", "year": "1891", "gt_cells_per_district": 39},
    {"province": "Burma", "year": "1901", "gt_cells_per_district": 21},
    {"province": "Assam", "year": "1901", "gt_cells_per_district": 24},
    {"province": "Mysore", "year": "1901", "gt_cells_per_district": 39},
    {"province": "Central_Provinces", "year": "1901", "gt_cells_per_district": 21},
    {"province": "Baroda", "year": "1901", "gt_cells_per_district": 57},
    {"province": "Rajputana", "year": "1901", "gt_cells_per_district": 21},
    {"province": "Gwalior", "year": "1901", "gt_cells_per_district": 39},
    {"province": "Madras", "year": "1901", "gt_cells_per_district": 21},
    {"province": "Central_India", "year": "1891", "gt_cells_per_district": 39},
    {"province": "Central_India", "year": "1901", "gt_cells_per_district": 21},
    {"province": "Bombay", "year": "1901", "gt_cells_per_district": 21},
    {"province": "Ajmer_Merwara", "year": "1901", "gt_cells_per_district": 60},
]


def find_pages(province, year):
    """Find all PNG pages for a province/year."""
    base_dir = AGE_TABLES / province / year
    if not base_dir.exists():
        # Try without underscore
        for d in AGE_TABLES.iterdir():
            if d.name.replace("_", " ").lower() == province.replace("_", " ").lower():
                base_dir = d / year
                break
    if not base_dir.exists():
        return []
    pages = sorted(base_dir.glob("*.png"))
    # Exclude any that are the original Hyderabad-style single-table images
    # (those don't have page numbers in the name)
    return pages


def result_exists(page_path):
    """Check if a result JSON already exists for this page."""
    stem = page_path.stem
    # oneshot.py names results as: {stem}_{year}_oneshot.json
    # Try common patterns
    for pattern in [
        f"{stem}_*_oneshot.json",
        f"{stem}_oneshot.json",
    ]:
        if list(RESULTS_DIR.glob(pattern)):
            return True
    return False


def run_page(page_path, timeout=600):
    """Run oneshot.py on a single page."""
    cmd = [sys.executable, str(PROJECT_ROOT / "oneshot.py"), str(page_path)]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
            cwd=str(PROJECT_ROOT)
        )
        return result.returncode == 0, result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT"
    except Exception as e:
        return False, str(e)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--province", help="Run only this province")
    parser.add_argument("--year", help="Run only this year")
    parser.add_argument("--dry-run", action="store_true", help="Show what would run")
    parser.add_argument("--max-pages", type=int, default=0, help="Stop after N pages")
    args = parser.parse_args()

    targets = TARGETS
    if args.province:
        targets = [t for t in targets if t["province"].lower() == args.province.lower()]
    if args.year:
        targets = [t for t in targets if t["year"] == args.year]

    # Collect all pages to process
    all_pages = []
    for target in targets:
        pages = find_pages(target["province"], target["year"])
        new_pages = [p for p in pages if not result_exists(p)]
        all_pages.extend([(target, p) for p in new_pages])
        status = f"{len(new_pages)}/{len(pages)} new"
        print(f"  {target['province']}/{target['year']}: {len(pages)} pages ({status})")

    print(f"\nTotal pages to process: {len(all_pages)}")
    if args.max_pages:
        all_pages = all_pages[:args.max_pages]
        print(f"  (limited to {args.max_pages})")

    if args.dry_run:
        for target, page in all_pages:
            print(f"  WOULD RUN: {page.name}")
        return

    # Process pages
    successes = 0
    failures = 0
    start_time = time.time()

    for i, (target, page) in enumerate(all_pages):
        elapsed = time.time() - start_time
        rate = (i / elapsed * 60) if elapsed > 0 and i > 0 else 0
        eta = ((len(all_pages) - i) / rate) if rate > 0 else 0

        print(f"\n[{i+1}/{len(all_pages)}] {target['province']}/{target['year']}/{page.name}"
              f"  ({rate:.1f} pages/min, ETA: {eta:.0f} min)")

        ok, output = run_page(page)
        if ok:
            successes += 1
            # Extract key stats from output
            for line in output.split("\n"):
                if "Done in" in line:
                    print(f"  {line.strip()}")
                    break
        else:
            failures += 1
            # Show last few lines of error
            lines = output.strip().split("\n")
            print(f"  FAILED: {lines[-1][:100]}")

    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"BATCH COMPLETE: {successes} succeeded, {failures} failed")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Average: {total_time/max(successes+failures,1):.0f}s per page")


if __name__ == "__main__":
    main()
