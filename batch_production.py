"""Production batch runner for large-scale census OCR processing.

Designed for processing 380 GB of Indian census images at scale.
Key features over batch_run.py:
  - Auto-discovers all images (no hardcoded target list)
  - Province/year-namespaced output prevents filename collisions
  - Structured JSONL logging for post-hoc analysis
  - Resume from last run (validates existing results on restart)
  - Configurable rate limiting via GEMINI_MIN_INTERVAL env var
  - Global summary report with failure breakdown
  - Graceful per-image failure handling

Usage:
    python3 batch_production.py age_tables/                    # process all
    python3 batch_production.py age_tables/Mysore/             # one province
    python3 batch_production.py age_tables/Mysore/1891/        # one year
    python3 batch_production.py age_tables/ --dry-run           # scan only
    python3 batch_production.py age_tables/ --rerun-failures    # redo failures
    python3 batch_production.py age_tables/ --max-failures 5    # stop after N failures
    python3 batch_production.py age_tables/ --timeout 900       # per-image timeout

Environment variables:
    GEMINI_MODEL            Primary model (default: gemini-3.1-pro-preview)
    GEMINI_ALLOW_FALLBACK   Set to "1" to allow fallback to gemini-2.5-pro
    GEMINI_MIN_INTERVAL     Minimum seconds between API calls (default: 0)
"""

import os
import sys
import json
import time
import signal
import logging
import argparse
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger("batch")


# ---------------------------------------------------------------------------
# Image discovery
# ---------------------------------------------------------------------------

def discover_images(root_dir, extensions=(".png", ".jpg", ".jpeg")):
    """Recursively find all image files under root_dir.

    Returns list of Path objects sorted by (province, year, filename).
    """
    root = Path(root_dir)
    if not root.exists():
        logger.error("Root directory does not exist: %s", root)
        return []

    images = []
    for ext in extensions:
        images.extend(root.rglob(f"*{ext}"))
        if ext != ext.upper():
            images.extend(root.rglob(f"*{ext.upper()}"))

    # Deduplicate (case-insensitive extensions could overlap on some OS)
    images = list({p.resolve(): p for p in images}.values())

    # Sort by path components for predictable ordering
    images.sort(key=lambda p: p.parts)
    return images


# ---------------------------------------------------------------------------
# Output path computation
# ---------------------------------------------------------------------------

def compute_output_dir(image_path, results_root):
    """Compute province/year-namespaced output directory for an image.

    Maps: age_tables/Mysore/1891/page_01.png -> results/Mysore/1891/
    Maps: age_tables/Mysore/some_image.png   -> results/Mysore/
    Maps: standalone_image.png               -> results/
    """
    image_path = Path(image_path).resolve()
    parts = image_path.parts

    # Find 'age_tables' in the path to determine relative structure
    try:
        at_idx = parts.index("age_tables")
        # Everything between age_tables and the filename is the namespace
        namespace_parts = parts[at_idx + 1:-1]  # province, year, etc.
    except ValueError:
        # Not under age_tables — use parent dirs up to 2 levels
        namespace_parts = image_path.parent.parts[-2:]

    return Path(results_root).joinpath(*namespace_parts)


def compute_result_json_path(image_path, output_dir):
    """Compute the expected result JSON path for an image."""
    parent = Path(image_path).parent.name
    stem = Path(image_path).stem
    if parent.isdigit() and not stem.endswith(f"_{parent}"):
        stem = f"{stem}_{parent}"
    return Path(output_dir) / f"{stem}_oneshot.json"


# ---------------------------------------------------------------------------
# Result validation
# ---------------------------------------------------------------------------

def validate_existing_result(json_path):
    """Check if an existing result JSON is valid and complete.

    Returns:
        (status, n_failures, n_checks, age_warnings)
        status: "perfect" | "has_failures" | "corrupt" | "missing"
    """
    json_path = Path(json_path)
    if not json_path.exists():
        return "missing", 0, 0, []

    try:
        with open(json_path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return "corrupt", 0, 0, []

    constraints = data.get("constraints")
    if not isinstance(constraints, dict):
        return "corrupt", 0, 0, []

    n_checks = constraints.get("total_checks", 0)
    n_failures = constraints.get("failed", 0)
    age_warnings = data.get("age_ordering_warnings", [])

    if n_checks == 0:
        return "corrupt", 0, 0, age_warnings

    if n_failures == 0:
        return "perfect", 0, n_checks, age_warnings
    return "has_failures", n_failures, n_checks, age_warnings


# ---------------------------------------------------------------------------
# JSONL structured logging
# ---------------------------------------------------------------------------

class BatchLogger:
    """Append-only JSONL logger for batch processing results."""

    def __init__(self, log_path):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, entry):
        """Append a JSON object as one line."""
        entry["timestamp"] = datetime.now().isoformat()
        with open(self.log_path, "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------

def process_one(image_path, output_dir, timeout=600):
    """Process a single image through the oneshot pipeline.

    Returns a result dict for logging.
    """
    from oneshot import extract_and_verify

    image_path = Path(image_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    result_entry = {
        "image": str(image_path),
        "output_dir": str(output_dir),
        "status": "unknown",
    }

    # Set an alarm for timeout (Unix only)
    timed_out = False
    old_handler = None
    if hasattr(signal, "SIGALRM"):
        def _timeout_handler(signum, frame):
            raise TimeoutError(f"Processing exceeded {timeout}s timeout")
        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(timeout)

    try:
        result = extract_and_verify(
            image_path, fallback=False, output_dir=str(output_dir))

        if result is None:
            result_entry["status"] = "extraction_failed"
        elif result.get("report", {}).get("all_passed", False):
            result_entry["status"] = "perfect"
            result_entry["checks"] = result["report"]["total_checks"]
            result_entry["api_calls"] = result.get("elapsed", 0)
        else:
            result_entry["status"] = "has_failures"
            result_entry["checks"] = result["report"]["total_checks"]
            result_entry["failures"] = result["report"]["failed"]
            result_entry["failure_details"] = result["report"].get("failures", [])

        age_warnings = result.get("age_ordering_warnings", []) if result else []
        if age_warnings:
            result_entry["age_ordering_warnings"] = age_warnings

    except TimeoutError as e:
        result_entry["status"] = "timeout"
        result_entry["error"] = str(e)
        timed_out = True
    except KeyboardInterrupt:
        result_entry["status"] = "interrupted"
        raise
    except Exception as e:
        result_entry["status"] = "error"
        result_entry["error"] = f"{type(e).__name__}: {str(e)[:200]}"

    finally:
        if hasattr(signal, "SIGALRM"):
            signal.alarm(0)  # Cancel the alarm
            if old_handler is not None:
                signal.signal(signal.SIGALRM, old_handler)

    result_entry["elapsed_seconds"] = round(time.time() - t0, 1)
    return result_entry


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------

def write_summary(results_root, batch_log_path, images_total, counters,
                  elapsed, failures_list):
    """Write a human-readable summary and machine-readable JSON."""
    summary = {
        "timestamp": datetime.now().isoformat(),
        "total_images": images_total,
        "processed": counters["processed"],
        "skipped_existing_perfect": counters["skipped_perfect"],
        "skipped_existing_failures": counters.get("skipped_failures", 0),
        "perfect": counters["perfect"],
        "has_failures": counters["has_failures"],
        "extraction_failed": counters["extraction_failed"],
        "errors": counters["errors"],
        "timeouts": counters["timeouts"],
        "elapsed_seconds": round(elapsed, 1),
        "elapsed_minutes": round(elapsed / 60, 1),
        "batch_log": str(batch_log_path),
    }

    if counters["processed"] > 0:
        summary["seconds_per_image"] = round(
            elapsed / counters["processed"], 1)

    if failures_list:
        summary["images_with_failures"] = failures_list[:50]

    summary_path = Path(results_root) / "_batch_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Print human-readable summary
    print(f"\n{'=' * 70}")
    print("BATCH SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Total images discovered: {images_total}")
    print(f"  Skipped (already perfect): {counters['skipped_perfect']}")
    if counters.get("skipped_failures", 0):
        print(f"  Skipped (existing with failures): {counters['skipped_failures']}")
    print(f"  Processed this run: {counters['processed']}")
    print(f"    Perfect (0 failures): {counters['perfect']}")
    print(f"    Has failures: {counters['has_failures']}")
    print(f"    Extraction failed: {counters['extraction_failed']}")
    print(f"    Errors/crashes: {counters['errors']}")
    print(f"    Timeouts: {counters['timeouts']}")
    print(f"  Total time: {elapsed / 60:.1f} minutes")
    if counters["processed"] > 0:
        rate = counters["processed"] / (elapsed / 60)
        print(f"  Rate: {rate:.1f} images/minute")
        remaining = images_total - counters["skipped_perfect"] - counters["processed"]
        if remaining > 0 and rate > 0:
            print(f"  Estimated remaining: {remaining / rate:.0f} minutes "
                  f"({remaining} images)")

    if failures_list:
        print(f"\n  Images with constraint failures ({len(failures_list)}):")
        for item in failures_list[:20]:
            print(f"    {item['failures']} failures: {item['image']}")
        if len(failures_list) > 20:
            print(f"    ... and {len(failures_list) - 20} more "
                  f"(see {summary_path})")

    print(f"\n  Summary: {summary_path}")
    print(f"  Full log: {batch_log_path}")
    return summary


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Production batch runner for large-scale census OCR",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    parser.add_argument("root", help="Root directory containing images "
                        "(e.g., age_tables/ or age_tables/Mysore/1891/)")
    parser.add_argument("--results", default="results",
                        help="Output root directory (default: results/)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Scan and report — don't process anything")
    parser.add_argument("--rerun-failures", action="store_true",
                        help="Re-process images that have constraint failures")
    parser.add_argument("--rerun-errors", action="store_true",
                        help="Re-process images that errored or timed out")
    parser.add_argument("--max-images", type=int, default=0,
                        help="Stop after processing N images (0 = no limit)")
    parser.add_argument("--max-failures", type=int, default=0,
                        help="Stop after N consecutive failures (0 = no limit)")
    parser.add_argument("--timeout", type=int, default=600,
                        help="Per-image timeout in seconds (default: 600)")
    parser.add_argument("--min-interval", type=float, default=0,
                        help="Minimum seconds between Gemini API calls "
                        "(overrides GEMINI_MIN_INTERVAL env var)")
    args = parser.parse_args()

    # Configure rate limiting
    if args.min_interval > 0:
        os.environ["GEMINI_MIN_INTERVAL"] = str(args.min_interval)

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s: %(message)s",
        datefmt="%H:%M:%S")

    results_root = Path(args.results)
    results_root.mkdir(parents=True, exist_ok=True)

    # Discover images
    print(f"Scanning {args.root} for images...")
    images = discover_images(args.root)
    print(f"Found {len(images)} images")

    if not images:
        print("No images found. Check the path.")
        return

    # Classify existing results
    to_process = []
    counters = {
        "processed": 0, "skipped_perfect": 0, "skipped_failures": 0,
        "perfect": 0, "has_failures": 0, "extraction_failed": 0,
        "errors": 0, "timeouts": 0,
    }

    for img_path in images:
        out_dir = compute_output_dir(img_path, results_root)
        json_path = compute_result_json_path(img_path, out_dir)
        status, n_fail, n_checks, age_warns = validate_existing_result(json_path)

        # Fallback: also check flat results/ directory (pre-namespaced runs)
        if status == "missing":
            flat_json = compute_result_json_path(img_path, results_root)
            if flat_json != json_path:
                status, n_fail, n_checks, age_warns = validate_existing_result(
                    flat_json)

        if status == "perfect" and not args.rerun_failures:
            counters["skipped_perfect"] += 1
            continue
        elif status == "has_failures" and not args.rerun_failures:
            counters["skipped_failures"] += 1
            continue
        elif status == "corrupt":
            # Always reprocess corrupt results
            to_process.append((img_path, out_dir))
        elif status == "missing":
            to_process.append((img_path, out_dir))
        elif args.rerun_failures and status in ("has_failures", "corrupt"):
            to_process.append((img_path, out_dir))
        elif args.rerun_failures and status == "perfect":
            counters["skipped_perfect"] += 1
            continue

    if args.max_images > 0:
        to_process = to_process[:args.max_images]

    print(f"\nTo process: {len(to_process)} images")
    print(f"Already perfect: {counters['skipped_perfect']}")
    if counters["skipped_failures"]:
        print(f"Existing with failures (use --rerun-failures): "
              f"{counters['skipped_failures']}")

    # Dry run: show stats and exit
    if args.dry_run:
        # Size analysis
        total_bytes = sum(p.stat().st_size for p, _ in to_process)
        print(f"\nTotal data to process: {total_bytes / (1024**3):.1f} GB")

        # Group by province
        by_province = {}
        for img_path, _ in to_process:
            parts = img_path.parts
            try:
                at_idx = parts.index("age_tables")
                province = parts[at_idx + 1] if at_idx + 1 < len(parts) - 1 else "unknown"
            except ValueError:
                province = "unknown"
            by_province.setdefault(province, []).append(img_path)

        print(f"\nBy province ({len(by_province)} provinces):")
        for prov in sorted(by_province):
            n = len(by_province[prov])
            sz = sum(p.stat().st_size for p in by_province[prov])
            print(f"  {prov:<35} {n:>5} images  {sz/(1024**2):>8.1f} MB")

        # Cost estimate (rough: ~2-5 API calls per image)
        avg_calls = 3
        total_calls = len(to_process) * avg_calls
        # Gemini pricing: ~$0.0025 per image call (rough estimate)
        est_cost = total_calls * 0.0025
        print(f"\nEstimated API calls: ~{total_calls:,} "
              f"(~{avg_calls} per image)")
        print(f"Rough cost estimate: ~${est_cost:,.0f}")

        avg_time = 20  # seconds per image
        est_hours = len(to_process) * avg_time / 3600
        print(f"Estimated time: ~{est_hours:.1f} hours "
              f"(at ~{avg_time}s/image)")
        return

    # Process images
    batch_log_path = results_root / "_batch_log.jsonl"
    batch_logger = BatchLogger(batch_log_path)
    failures_list = []
    consecutive_failures = 0
    t_batch_start = time.time()

    print(f"\nStarting batch at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Log: {batch_log_path}")
    print()

    try:
        for i, (img_path, out_dir) in enumerate(to_process):
            elapsed_batch = time.time() - t_batch_start
            rate = (i / elapsed_batch * 60) if elapsed_batch > 0 and i > 0 else 0
            remaining = len(to_process) - i
            eta_min = (remaining / rate) if rate > 0 else 0

            print(f"\n[{i + 1}/{len(to_process)}] "
                  f"{img_path.relative_to(Path(args.root).parent) if args.root in str(img_path) else img_path.name}"
                  f"  ({rate:.1f}/min, ETA: {eta_min:.0f}min)")

            result = process_one(img_path, out_dir, timeout=args.timeout)
            batch_logger.log(result)

            status = result["status"]
            counters["processed"] += 1

            if status == "perfect":
                counters["perfect"] += 1
                consecutive_failures = 0
                print(f"  PERFECT ({result.get('checks', '?')} checks, "
                      f"{result['elapsed_seconds']}s)")
            elif status == "has_failures":
                counters["has_failures"] += 1
                consecutive_failures += 1
                n_fail = result.get("failures", "?")
                print(f"  {n_fail} FAILURES ({result['elapsed_seconds']}s)")
                failures_list.append({
                    "image": str(img_path),
                    "failures": n_fail,
                })
            elif status == "extraction_failed":
                counters["extraction_failed"] += 1
                consecutive_failures += 1
                print(f"  EXTRACTION FAILED ({result['elapsed_seconds']}s)")
            elif status == "timeout":
                counters["timeouts"] += 1
                consecutive_failures += 1
                print(f"  TIMEOUT after {args.timeout}s")
            elif status == "error":
                counters["errors"] += 1
                consecutive_failures += 1
                print(f"  ERROR: {result.get('error', '?')[:100]}")
            else:
                counters["errors"] += 1
                consecutive_failures += 1
                print(f"  UNKNOWN STATUS: {status}")

            # Age ordering warnings
            if result.get("age_ordering_warnings"):
                print(f"  AGE ORDER WARNING: "
                      f"{len(result['age_ordering_warnings'])} issues")

            # Check consecutive failure limit
            if (args.max_failures > 0
                    and consecutive_failures >= args.max_failures):
                print(f"\n  STOPPING: {consecutive_failures} consecutive "
                      f"failures (limit: {args.max_failures})")
                break

    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Writing summary...")

    elapsed_total = time.time() - t_batch_start
    write_summary(results_root, batch_log_path, len(images), counters,
                  elapsed_total, failures_list)


if __name__ == "__main__":
    main()
