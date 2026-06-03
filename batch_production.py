"""Production batch runner for large-scale census OCR processing.

Designed for processing 380 GB of Indian census images at scale.
Key features:
  - Auto-discovers all images (no hardcoded target list)
  - Province/year-namespaced output prevents filename collisions
  - Structured JSONL logging for post-hoc analysis
  - Resume from last run (validates existing results on restart)
  - Configurable rate limiting via GEMINI_MIN_INTERVAL env var
  - Global summary report with failure breakdown
  - Concurrent processing via --workers thread pool
  - Slurm array job support via --chunk / --total-chunks
  - Graceful SIGTERM/SIGINT shutdown with checkpoint
  - Periodic checkpoint writes (every --checkpoint-interval images)
  - Per-image API call cap via --max-api-calls

Usage:
    python3 batch_production.py age_tables/                    # process all
    python3 batch_production.py age_tables/Mysore/             # one province
    python3 batch_production.py age_tables/Mysore/1891/        # one year
    python3 batch_production.py age_tables/ --dry-run           # scan only
    python3 batch_production.py age_tables/ --rerun-failures    # redo failures
    python3 batch_production.py age_tables/ --max-failures 5    # stop after N failures
    python3 batch_production.py age_tables/ --timeout 900       # per-image timeout
    python3 batch_production.py age_tables/ --workers 4         # 4 concurrent workers
    python3 batch_production.py age_tables/ --max-api-calls 6   # cap API calls/image

Slurm array jobs:
    #SBATCH --array=0-19
    python3 batch_production.py age_tables/ \\
        --chunk $SLURM_ARRAY_TASK_ID --total-chunks 20 --workers 4

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
import threading
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger("batch")


# ---------------------------------------------------------------------------
# Graceful shutdown
# ---------------------------------------------------------------------------

_shutdown_requested = threading.Event()


def _request_shutdown(signum, frame):
    """Signal handler for SIGTERM/SIGINT — request graceful shutdown."""
    sig_name = signal.Signals(signum).name if hasattr(signal, 'Signals') else signum
    print(f"\n\nReceived {sig_name} — shutting down gracefully "
          "(finishing in-flight work)...")
    _shutdown_requested.set()


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

    # Ignore pipeline-generated artifacts that may live alongside source scans.
    # These are temporary or derived images, not distinct census pages.
    filtered = []
    for p in images:
        stem = p.stem.lower()
        if "_rot" in stem or "_up" in stem:
            continue
        filtered.append(p)
    images = filtered

    # Sort by path components for predictable ordering
    images.sort(key=lambda p: p.parts)
    return images


def discover_images_from_list(list_path):
    """Load explicit image paths from a text manifest.

    The file may contain one absolute or relative image path per line.
    Blank lines and comment lines beginning with '#' are ignored.
    """
    manifest = Path(list_path)
    if not manifest.exists():
        logger.error("Image list does not exist: %s", manifest)
        return []

    images = []
    seen = set()
    with open(manifest, encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            path = Path(line).expanduser()
            if not path.is_absolute():
                path = (manifest.parent / path).resolve()
            else:
                path = path.resolve()
            if not path.exists():
                logger.warning("Skipping missing image from list: %s", path)
                continue
            if path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
                logger.warning("Skipping non-image path from list: %s", path)
                continue
            if path in seen:
                continue
            seen.add(path)
            images.append(path)

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
        (status, n_failures, n_checks, age_warnings, acceptance)
        status:
          "accepted_high_confidence" | "accepted_with_risk" |
          "has_failures" | "corrupt" | "missing"
    """
    json_path = Path(json_path)
    if not json_path.exists():
        return "missing", 0, 0, [], None

    try:
        with open(json_path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return "corrupt", 0, 0, [], None

    constraints = data.get("constraints")
    if not isinstance(constraints, dict):
        return "corrupt", 0, 0, [], None

    n_checks = constraints.get("total_checks", 0)
    n_failures = constraints.get("failed", 0)
    age_warnings = data.get("age_ordering_warnings", [])
    acceptance = data.get("acceptance")

    if n_checks == 0:
        return "corrupt", 0, 0, age_warnings, acceptance

    if n_failures > 0:
        return "has_failures", n_failures, n_checks, age_warnings, acceptance

    decision = None
    if isinstance(acceptance, dict):
        decision = acceptance.get("decision")

    if decision == "accepted_high_confidence":
        return "accepted_high_confidence", 0, n_checks, age_warnings, acceptance
    return "accepted_with_risk", 0, n_checks, age_warnings, acceptance


# ---------------------------------------------------------------------------
# JSONL structured logging (thread-safe)
# ---------------------------------------------------------------------------

class BatchLogger:
    """Append-only JSONL logger for batch processing results."""

    def __init__(self, log_path):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def log(self, entry):
        """Append a JSON object as one line (thread-safe)."""
        entry["timestamp"] = datetime.now().isoformat()
        line = json.dumps(entry, default=str) + "\n"
        with self._lock:
            with open(self.log_path, "a") as f:
                f.write(line)


# ---------------------------------------------------------------------------
# Thread-safe counters
# ---------------------------------------------------------------------------

class AtomicCounters:
    """Thread-safe counters for batch processing stats."""

    def __init__(self):
        self._lock = threading.Lock()
        self._counts = {
            "processed": 0, "skipped_high_confidence": 0,
            "accepted_with_risk": 0, "skipped_failures": 0,
            "accepted_high_confidence": 0, "accepted_with_risk_processed": 0,
            "has_failures": 0, "extraction_failed": 0,
            "errors": 0, "timeouts": 0,
            "total_api_calls": 0,
        }

    def increment(self, key, n=1):
        with self._lock:
            self._counts[key] = self._counts.get(key, 0) + n

    def snapshot(self):
        with self._lock:
            return dict(self._counts)

    def __getitem__(self, key):
        with self._lock:
            return self._counts[key]


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------

def process_one(image_path, output_dir, timeout=600, max_api_calls=0):
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

    # Set an alarm for timeout (Unix only).
    # SIGALRM is per-process and not safe with threads — only use it
    # when running single-threaded (workers=1).
    use_alarm = (hasattr(signal, "SIGALRM")
                 and threading.active_count() <= 2)  # main + 1 worker at most
    old_handler = None
    if use_alarm:
        def _timeout_handler(signum, frame):
            raise TimeoutError(f"Processing exceeded {timeout}s timeout")
        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(timeout)

    try:
        kwargs = dict(fallback=os.environ.get("SUDOKUOCR_MOE_FALLBACK") == "1",
                      output_dir=str(output_dir))
        if max_api_calls > 0:
            kwargs["max_api_calls"] = max_api_calls
        result = extract_and_verify(image_path, **kwargs)

        if result is None:
            result_entry["status"] = "extraction_failed"
        else:
            api_calls = result.get("api_calls", 0)
            if not api_calls:
                api_calls = result.get("acceptance", {}).get(
                    "signals", {}).get("api_calls", 0)
            if api_calls:
                result_entry["api_calls"] = api_calls

        if result is None:
            pass
        elif result.get("report", {}).get("all_passed", False):
            acceptance = result.get("acceptance", {})
            decision = acceptance.get("decision", "accepted_with_risk")
            result_entry["status"] = decision
            result_entry["checks"] = result["report"]["total_checks"]
            result_entry["acceptance"] = acceptance
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
    except KeyboardInterrupt:
        result_entry["status"] = "interrupted"
        raise
    except Exception as e:
        result_entry["status"] = "error"
        result_entry["error"] = f"{type(e).__name__}: {str(e)[:200]}"

    finally:
        if use_alarm:
            signal.alarm(0)  # Cancel the alarm
            if old_handler is not None:
                signal.signal(signal.SIGALRM, old_handler)

    result_entry["elapsed_seconds"] = round(time.time() - t0, 1)
    return result_entry


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------

def write_summary(results_root, batch_log_path, images_total, counters,
                  elapsed, failures_list, is_checkpoint=False):
    """Write a human-readable summary and machine-readable JSON."""
    if isinstance(counters, AtomicCounters):
        counters = counters.snapshot()

    label = "CHECKPOINT" if is_checkpoint else "BATCH SUMMARY"

    summary = {
        "timestamp": datetime.now().isoformat(),
        "is_checkpoint": is_checkpoint,
        "total_images": images_total,
        "processed": counters["processed"],
        "skipped_existing_high_confidence": counters["skipped_high_confidence"],
        "existing_accepted_with_risk": counters.get("accepted_with_risk", 0),
        "skipped_existing_failures": counters.get("skipped_failures", 0),
        "accepted_high_confidence": counters["accepted_high_confidence"],
        "accepted_with_risk": counters["accepted_with_risk_processed"],
        "has_failures": counters["has_failures"],
        "extraction_failed": counters["extraction_failed"],
        "errors": counters["errors"],
        "timeouts": counters["timeouts"],
        "elapsed_seconds": round(elapsed, 1),
        "elapsed_minutes": round(elapsed / 60, 1),
        "batch_log": str(batch_log_path),
    }

    total_api_calls = counters.get("total_api_calls", 0)
    est_cost = total_api_calls * 0.0025
    summary["total_api_calls"] = total_api_calls
    summary["estimated_cost_usd"] = round(est_cost, 2)

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
    print(label)
    print(f"{'=' * 70}")
    print(f"  Total images discovered: {images_total}")
    print(f"  Skipped (accepted_high_confidence): {counters['skipped_high_confidence']}")
    if counters.get("accepted_with_risk", 0):
        print(f"  Existing accepted_with_risk reprocessed: {counters['accepted_with_risk']}")
    if counters.get("skipped_failures", 0):
        print(f"  Skipped (existing with failures): {counters['skipped_failures']}")
    print(f"  Processed this run: {counters['processed']}")
    print(f"    Accepted high confidence: {counters['accepted_high_confidence']}")
    print(f"    Accepted with risk: {counters['accepted_with_risk_processed']}")
    print(f"    Has failures: {counters['has_failures']}")
    print(f"    Extraction failed: {counters['extraction_failed']}")
    print(f"    Errors/crashes: {counters['errors']}")
    print(f"    Timeouts: {counters['timeouts']}")
    print(f"  Total time: {elapsed / 60:.1f} minutes")
    print(f"  API calls: {counters.get('total_api_calls', 0)} "
          f"(~${counters.get('total_api_calls', 0) * 0.0025:.2f})")
    if counters["processed"] > 0:
        rate = counters["processed"] / (elapsed / 60)
        print(f"  Rate: {rate:.1f} images/minute")
        remaining = images_total - counters["skipped_high_confidence"] - counters["processed"]
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
    parser.add_argument("root", nargs="?", help="Root directory containing images "
                        "(e.g., age_tables/ or age_tables/Mysore/1891/)")
    parser.add_argument("--results", default="results",
                        help="Output root directory (default: results/)")
    parser.add_argument("--image-list",
                        help="Text file containing explicit image paths, one per line. "
                             "If set, this is used instead of recursively scanning root.")
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
    parser.add_argument("--chunk", type=int, default=None,
                        help="Which chunk to process (0-indexed, for Slurm "
                        "array jobs). Use with --total-chunks.")
    parser.add_argument("--total-chunks", type=int, default=None,
                        help="Total number of chunks to split work into")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of concurrent worker threads (default: 1). "
                        "Each worker processes one image at a time via API calls.")
    parser.add_argument("--max-api-calls", type=int, default=0,
                        help="Max API calls per image (0 = no limit). "
                        "Caps the repair cascade to control cost.")
    parser.add_argument("--checkpoint-interval", type=int, default=50,
                        help="Write checkpoint summary every N images "
                        "(default: 50)")
    parser.add_argument("--max-cost", type=float, default=0,
                        help="Stop after estimated spend exceeds this amount "
                        "in USD (0 = no limit). Uses ~$0.0025/API call.")
    args = parser.parse_args()

    # Configure rate limiting
    if args.min_interval > 0:
        os.environ["GEMINI_MIN_INTERVAL"] = str(args.min_interval)

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s [%(threadName)s]: %(message)s",
        datefmt="%H:%M:%S")

    results_root = Path(args.results)
    results_root.mkdir(parents=True, exist_ok=True)

    # Install graceful shutdown handlers
    signal.signal(signal.SIGTERM, _request_shutdown)
    # SIGINT (Ctrl+C) — also handle gracefully
    signal.signal(signal.SIGINT, _request_shutdown)

    # Discover images
    if args.image_list:
        print(f"Loading image list from {args.image_list}...")
        images = discover_images_from_list(args.image_list)
    elif args.root:
        print(f"Scanning {args.root} for images...")
        images = discover_images(args.root)
    else:
        parser.error("Provide either ROOT or --image-list")
    print(f"Found {len(images)} images")

    if not images:
        print("No images found. Check the path.")
        return

    # Classify existing results
    to_process = []
    counters = AtomicCounters()

    for img_path in images:
        out_dir = compute_output_dir(img_path, results_root)
        json_path = compute_result_json_path(img_path, out_dir)
        status, n_fail, n_checks, age_warns, acceptance = validate_existing_result(json_path)

        # Fallback: also check flat results/ directory (pre-namespaced runs)
        if status == "missing":
            flat_json = compute_result_json_path(img_path, results_root)
            if flat_json != json_path:
                status, n_fail, n_checks, age_warns, acceptance = validate_existing_result(
                    flat_json)

        if status == "accepted_high_confidence" and not args.rerun_failures:
            counters.increment("skipped_high_confidence")
            continue
        elif status == "accepted_with_risk":
            counters.increment("accepted_with_risk")
            to_process.append((img_path, out_dir))
            continue
        elif status == "has_failures" and not args.rerun_failures:
            counters.increment("skipped_failures")
            continue
        elif status == "corrupt":
            # Always reprocess corrupt results
            to_process.append((img_path, out_dir))
        elif status == "missing":
            to_process.append((img_path, out_dir))
        elif args.rerun_failures and status in ("has_failures", "corrupt"):
            to_process.append((img_path, out_dir))
        elif args.rerun_failures and status == "accepted_high_confidence":
            counters.increment("skipped_high_confidence")
            continue

    # Chunking for Slurm array jobs
    if args.chunk is not None and args.total_chunks is not None:
        if args.chunk < 0 or args.chunk >= args.total_chunks:
            print(f"ERROR: --chunk must be 0..{args.total_chunks - 1}")
            return
        chunk_size = (len(to_process) + args.total_chunks - 1) // args.total_chunks
        start = args.chunk * chunk_size
        end = min(start + chunk_size, len(to_process))
        print(f"\nChunk {args.chunk}/{args.total_chunks}: "
              f"images {start}-{end - 1} of {len(to_process)}")
        to_process = to_process[start:end]
    elif args.chunk is not None or args.total_chunks is not None:
        print("ERROR: --chunk and --total-chunks must be used together")
        return

    if args.max_images > 0:
        to_process = to_process[:args.max_images]

    print(f"\nTo process: {len(to_process)} images")
    print(f"Already accepted_high_confidence: {counters['skipped_high_confidence']}")
    if counters["accepted_with_risk"]:
        print(f"Existing accepted_with_risk (will be reprocessed): "
              f"{counters['accepted_with_risk']}")
    if counters["skipped_failures"]:
        print(f"Existing with failures (use --rerun-failures): "
              f"{counters['skipped_failures']}")
    if args.workers > 1:
        print(f"Workers: {args.workers} concurrent threads")
    if args.max_api_calls > 0:
        print(f"API call cap: {args.max_api_calls} per image")

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
        avg_calls = min(3, args.max_api_calls) if args.max_api_calls > 0 else 3
        total_calls = len(to_process) * avg_calls
        # Gemini pricing: ~$0.0025 per image call (rough estimate)
        est_cost = total_calls * 0.0025
        print(f"\nEstimated API calls: ~{total_calls:,} "
              f"(~{avg_calls} per image)")
        print(f"Rough cost estimate: ~${est_cost:,.0f}")

        avg_time = 20  # seconds per image
        effective_workers = max(1, args.workers)
        est_hours = len(to_process) * avg_time / 3600 / effective_workers
        print(f"Estimated time: ~{est_hours:.1f} hours "
              f"(at ~{avg_time}s/image, {effective_workers} workers)")
        return

    # Process images
    batch_log_path = results_root / "_batch_log.jsonl"
    if args.chunk is not None:
        # Per-chunk log to avoid write contention across Slurm jobs
        batch_log_path = results_root / f"_batch_log_chunk{args.chunk}.jsonl"
    batch_logger = BatchLogger(batch_log_path)
    failures_list = []
    failures_lock = threading.Lock()
    consecutive_failures = 0
    consecutive_lock = threading.Lock()
    t_batch_start = time.time()
    completed_count = 0
    completed_lock = threading.Lock()

    print(f"\nStarting batch at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Log: {batch_log_path}")
    print()

    COST_PER_CALL = 0.0025  # rough estimate USD per Gemini API call

    def _handle_result(result, img_path, idx):
        """Process a completed result (called from main thread or callback)."""
        nonlocal consecutive_failures, completed_count

        batch_logger.log(result)
        status = result["status"]
        counters.increment("processed")

        # Track API calls for cost estimation
        n_calls = result.get("api_calls", 0)
        if not n_calls:
            # Estimate from status: successful images use 2-3 calls
            n_calls = 2 if status in ("accepted_high_confidence",
                                       "accepted_with_risk",
                                       "has_failures") else 1
        counters.increment("total_api_calls", n_calls)

        with completed_lock:
            completed_count += 1
            current_count = completed_count

        if status == "accepted_high_confidence":
            counters.increment("accepted_high_confidence")
            with consecutive_lock:
                consecutive_failures = 0
            print(f"  [{current_count}/{len(to_process)}] "
                  f"{img_path.name} "
                  f"ACCEPTED_HIGH_CONFIDENCE ({result.get('checks', '?')} checks, "
                  f"{result['elapsed_seconds']}s)")
        elif status == "accepted_with_risk":
            counters.increment("accepted_with_risk_processed")
            with consecutive_lock:
                consecutive_failures = 0
            trust = result.get("acceptance", {}).get("trust_score", "?")
            print(f"  [{current_count}/{len(to_process)}] "
                  f"{img_path.name} "
                  f"ACCEPTED_WITH_RISK (trust={trust}, "
                  f"{result['elapsed_seconds']}s)")
        elif status == "has_failures":
            counters.increment("has_failures")
            with consecutive_lock:
                consecutive_failures += 1
            n_fail = result.get("failures", "?")
            print(f"  [{current_count}/{len(to_process)}] "
                  f"{img_path.name} "
                  f"{n_fail} FAILURES ({result['elapsed_seconds']}s)")
            with failures_lock:
                failures_list.append({
                    "image": str(img_path),
                    "failures": n_fail,
                })
        elif status == "extraction_failed":
            counters.increment("extraction_failed")
            with consecutive_lock:
                consecutive_failures += 1
            print(f"  [{current_count}/{len(to_process)}] "
                  f"{img_path.name} "
                  f"EXTRACTION FAILED ({result['elapsed_seconds']}s)")
        elif status == "timeout":
            counters.increment("timeouts")
            with consecutive_lock:
                consecutive_failures += 1
            print(f"  [{current_count}/{len(to_process)}] "
                  f"{img_path.name} TIMEOUT")
        elif status == "error":
            counters.increment("errors")
            with consecutive_lock:
                consecutive_failures += 1
            print(f"  [{current_count}/{len(to_process)}] "
                  f"{img_path.name} "
                  f"ERROR: {result.get('error', '?')[:100]}")
        else:
            counters.increment("errors")
            with consecutive_lock:
                consecutive_failures += 1
            print(f"  [{current_count}/{len(to_process)}] "
                  f"{img_path.name} "
                  f"UNKNOWN STATUS: {status}")

        if result.get("age_ordering_warnings"):
            print(f"    AGE ORDER WARNING: "
                  f"{len(result['age_ordering_warnings'])} issues")

        # Periodic checkpoint
        if (args.checkpoint_interval > 0
                and current_count % args.checkpoint_interval == 0):
            elapsed = time.time() - t_batch_start
            print(f"\n  --- Checkpoint at {current_count} images ---")
            write_summary(results_root, batch_log_path, len(images),
                          counters, elapsed, failures_list,
                          is_checkpoint=True)

    def _should_stop():
        """Check if we should stop processing."""
        if _shutdown_requested.is_set():
            return True
        if args.max_failures > 0:
            with consecutive_lock:
                if consecutive_failures >= args.max_failures:
                    return True
        if args.max_cost > 0:
            est_cost = counters["total_api_calls"] * COST_PER_CALL
            if est_cost >= args.max_cost:
                print(f"\n  COST CAP: ~${est_cost:.2f} estimated "
                      f"(limit: ${args.max_cost:.2f})")
                return True
        return False

    try:
        if args.workers <= 1:
            # Sequential mode (original behavior, with SIGALRM timeout)
            for i, (img_path, out_dir) in enumerate(to_process):
                if _should_stop():
                    with consecutive_lock:
                        cf = consecutive_failures
                    if _shutdown_requested.is_set():
                        print("\nShutdown requested. Writing summary...")
                    else:
                        print(f"\n  STOPPING: {cf} consecutive "
                              f"failures (limit: {args.max_failures})")
                    break

                elapsed_batch = time.time() - t_batch_start
                rate = (i / elapsed_batch * 60) if elapsed_batch > 0 and i > 0 else 0
                remaining = len(to_process) - i
                eta_min = (remaining / rate) if rate > 0 else 0

                print(f"\n[{i + 1}/{len(to_process)}] "
                      f"{img_path.relative_to(Path(args.root).parent) if args.root and args.root in str(img_path) else img_path.name}"
                      f"  ({rate:.1f}/min, ETA: {eta_min:.0f}min)")

                result = process_one(img_path, out_dir,
                                     timeout=args.timeout,
                                     max_api_calls=args.max_api_calls)
                _handle_result(result, img_path, i)
        else:
            # Concurrent mode — thread pool
            # Note: SIGALRM doesn't work with threads, so we enforce
            # per-image timeout via future.result(timeout=...).
            with ThreadPoolExecutor(max_workers=args.workers,
                                    thread_name_prefix="ocr") as executor:
                futures = {}
                submitted = 0
                # Submit initial batch up to pool size
                for i, (img_path, out_dir) in enumerate(to_process):
                    if _should_stop():
                        break
                    future = executor.submit(
                        process_one, img_path, out_dir,
                        timeout=args.timeout,
                        max_api_calls=args.max_api_calls)
                    futures[future] = (img_path, i)
                    submitted = i + 1

                for future in as_completed(futures):
                    if _should_stop():
                        for f in futures:
                            f.cancel()
                        break
                    img_path, idx = futures[future]
                    try:
                        result = future.result(timeout=args.timeout + 60)
                    except TimeoutError:
                        result = {
                            "image": str(img_path),
                            "status": "timeout",
                            "error": f"Thread timeout after {args.timeout + 60}s",
                            "elapsed_seconds": args.timeout + 60,
                        }
                    except Exception as e:
                        result = {
                            "image": str(img_path),
                            "status": "error",
                            "error": f"{type(e).__name__}: {str(e)[:200]}",
                            "elapsed_seconds": 0,
                        }
                    _handle_result(result, img_path, idx)

    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Writing summary...")

    if _shutdown_requested.is_set():
        print("\nGraceful shutdown complete.")

    elapsed_total = time.time() - t_batch_start
    write_summary(results_root, batch_log_path, len(images), counters,
                  elapsed_total, failures_list)


if __name__ == "__main__":
    main()
