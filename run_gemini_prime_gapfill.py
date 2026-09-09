#!/usr/bin/env python3
"""Targeted Harvard-Gemini OCR gap filler for prime-age Table VII pages.

This is intentionally conservative on local memory: it launches one
batch_production.py subprocess per Gemini key by default, with each subprocess
using one image worker. The script builds an auditable queue first, then can
launch the work across all available Harvard keys.
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from batch_production import (
    compute_output_dir,
    compute_result_json_path,
    validate_existing_result,
)


PROJECT_ROOT = Path(__file__).resolve().parent
WORKLIST_DIR = PROJECT_ROOT / (
    "export_final/pdf_recovery_incoming_2026-06-16_no_gaps_ledger_v2_refined/"
    "gemini_ocr_worklists"
)
DEFAULT_RESULTS = PROJECT_ROOT / "export_final/gemini_no_gap_ocr_2026-06-16_v2_best"

CLASS_MANIFESTS = {
    "main": WORKLIST_DIR / "gemini_manifest_fullocr_main_tablevii_numeric_all.txt",
    "derivative": WORKLIST_DIR / "gemini_manifest_derivative_civil_condition.txt",
    "context": WORKLIST_DIR / "gemini_manifest_context_or_blank.txt",
    "strict_core": WORKLIST_DIR / "gemini_manifest_fullocr_core_money_strict_all.txt",
    "target_or_missing": WORKLIST_DIR / "gemini_priority_manifest_target_or_missing.txt",
}

CLASS_PRIORITY = {
    "main": 0,
    "strict_core": 0,
    "derivative": 1,
    "context": 2,
    "target_or_missing": 3,
    "exact": -1,
}


@dataclass(frozen=True)
class QueueRow:
    priority_class: str
    image_path: Path
    status: str
    failures: int
    checks: int
    result_json: Path


def load_harvard_keys(env_path: Path) -> list[str]:
    raw = os.environ.get("HARVARD_GEMINI_API_KEYS", "").strip()
    if not raw and env_path.exists():
        raw = load_env_value(env_path, "HARVARD_GEMINI_API_KEYS")
    return [k.strip() for k in raw.split(",") if k.strip()]


def load_env_value(env_path: Path, name: str) -> str:
    if not env_path.exists():
        return ""
    prefix = name + "="
    for line in env_path.read_text(errors="ignore").splitlines():
        if line.startswith(prefix):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    return ""


def read_manifest(path: Path) -> list[Path]:
    if not path.exists():
        raise FileNotFoundError(f"Missing manifest: {path}")
    out: list[Path] = []
    seen: set[Path] = set()
    with path.open(encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            p = Path(line).expanduser()
            if not p.is_absolute():
                p = (path.parent / p).resolve()
            else:
                p = p.resolve()
            if p not in seen:
                seen.add(p)
                out.append(p)
    return out


def audit_image(image: Path, results_root: Path) -> tuple[str, int, int, Path]:
    out_dir = compute_output_dir(image, results_root)
    json_path = compute_result_json_path(image, out_dir)
    status, failures, checks, _, _ = validate_existing_result(json_path)
    if status == "missing":
        flat_json = compute_result_json_path(image, results_root)
        if flat_json != json_path:
            status, failures, checks, _, _ = validate_existing_result(flat_json)
            if status != "missing":
                json_path = flat_json
    return status, int(failures or 0), int(checks or 0), json_path


def page_sort_key(path: Path) -> tuple[str, int, str]:
    stem = path.stem
    page = -1
    if "_p" in stem:
        tail = stem.rsplit("_p", 1)[-1]
        digits = ""
        for ch in tail:
            if ch.isdigit():
                digits += ch
            else:
                break
        if digits:
            page = int(digits)
    source = path.parent.name
    if not source.isdigit() and "_" in stem:
        source = stem.split("_", 1)[0]
    return source, page, str(path)


def build_queue(
    classes: Iterable[str],
    results_root: Path,
    include_statuses: set[str],
    include_failures_max: int | None,
    exclude_images: set[Path],
    limit: int,
) -> list[QueueRow]:
    rows: list[QueueRow] = []
    seen: set[Path] = set()
    for cls in classes:
        manifest = CLASS_MANIFESTS[cls]
        for image in read_manifest(manifest):
            if image in exclude_images:
                continue
            if image in seen:
                continue
            seen.add(image)
            status, failures, checks, result_json = audit_image(image, results_root)
            include = status in include_statuses
            if include_failures_max is not None and status == "has_failures":
                include = failures <= include_failures_max
            if include:
                rows.append(QueueRow(cls, image, status, failures, checks, result_json))

    rows.sort(
        key=lambda r: (
            CLASS_PRIORITY.get(r.priority_class, 99),
            {"missing": 0, "corrupt": 1, "has_failures": 2, "accepted_with_risk": 3}.get(
                r.status, 9
            ),
            r.failures,
            page_sort_key(r.image_path),
        )
    )
    if limit > 0:
        rows = rows[:limit]
    return rows


def build_exact_queue(manifest: Path, results_root: Path, limit: int) -> list[QueueRow]:
    rows: list[QueueRow] = []
    seen: set[Path] = set()
    for image in read_manifest(manifest):
        if image in seen:
            continue
        seen.add(image)
        status, failures, checks, result_json = audit_image(image, results_root)
        rows.append(QueueRow("exact", image, status, failures, checks, result_json))
    rows.sort(key=lambda r: page_sort_key(r.image_path))
    if limit > 0:
        rows = rows[:limit]
    return rows


def write_queue_files(rows: list[QueueRow], prefix: Path) -> tuple[Path, Path]:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    manifest = prefix.with_suffix(".txt")
    audit_csv = prefix.with_suffix(".csv")

    with manifest.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(str(row.image_path) + "\n")

    with audit_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "priority_class",
                "image_path",
                "status",
                "failures",
                "checks",
                "result_json",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "priority_class": row.priority_class,
                    "image_path": row.image_path,
                    "status": row.status,
                    "failures": row.failures,
                    "checks": row.checks,
                    "result_json": row.result_json,
                }
            )
    return manifest, audit_csv


def summarize(rows: Iterable[QueueRow]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        counts[row.status] = counts.get(row.status, 0) + 1
        key = f"class:{row.priority_class}"
        counts[key] = counts.get(key, 0) + 1
    return counts


def launch_batches(
    manifest: Path,
    rows_count: int,
    results_root: Path,
    key_pairs: list[tuple[int, str]],
    log_dir: Path,
    workers_per_key: int,
    timeout: int,
    min_interval: float,
    max_api_calls: int,
    max_cost: float,
    stagger_seconds: float,
    rerun_failures: bool,
) -> int:
    if rows_count == 0:
        print("No queued images to run.")
        return 0
    n = min(len(key_pairs), rows_count)
    log_dir.mkdir(parents=True, exist_ok=True)
    procs: list[tuple[int, subprocess.Popen]] = []
    for i, (key_index, key) in enumerate(key_pairs[:n]):
        env = os.environ.copy()
        env_path = PROJECT_ROOT / ".env"
        env["HARVARD_GEMINI_API_KEY"] = key
        env["HARVARD_GEMINI_BASE_URL"] = (
            env.get("HARVARD_GEMINI_BASE_URL")
            or load_env_value(env_path, "HARVARD_GEMINI_BASE_URL")
        )
        env["OPENAI_API_KEY"] = ""
        env["ANTHROPIC_API_KEY"] = ""
        env["MISTRAL_API_KEY"] = ""
        env["GEMINI_API_KEY"] = ""
        env["GEMINI_ALLOW_FALLBACK"] = "0"
        env["SUDOKUOCR_TABLEVII_HINT"] = "1"
        env["SUDOKUOCR_MOE_FALLBACK"] = "0"
        env["GEMINI_MIN_INTERVAL"] = str(min_interval)
        env["PYTHONUNBUFFERED"] = "1"

        log_path = log_dir / f"prime_gapfill_key{key_index:02d}.log"
        cmd = [
            sys.executable,
            "batch_production.py",
            "--image-list",
            str(manifest),
            "--results",
            str(results_root),
            "--chunk",
            str(i),
            "--total-chunks",
            str(n),
            "--workers",
            str(workers_per_key),
            "--timeout",
            str(timeout),
            "--min-interval",
            str(min_interval),
            "--max-api-calls",
            str(max_api_calls),
            "--max-cost",
            str(max_cost),
            "--checkpoint-interval",
            "10",
        ]
        if rerun_failures:
            cmd.append("--rerun-failures")
        with log_path.open("wb") as log_fh:
            proc = subprocess.Popen(
                cmd,
                cwd=PROJECT_ROOT,
                env=env,
                stdout=log_fh,
                stderr=subprocess.STDOUT,
            )
        procs.append((key_index, proc))
        print(f"launched key {key_index:02d} pid={proc.pid} log={log_path}")
        if stagger_seconds > 0 and i + 1 < n:
            time.sleep(stagger_seconds)

    failures = 0
    for key_index, proc in procs:
        rc = proc.wait()
        if rc != 0:
            failures += 1
            print(f"key {key_index:02d} exited rc={rc}")
    print(f"batch workers complete: {len(procs)} launched, {failures} failed processes")
    return failures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--classes",
        default="main,derivative",
        help="Comma-separated classes: main, derivative, context, strict_core, target_or_missing",
    )
    parser.add_argument(
        "--exact-manifest",
        default="",
        help="Optional explicit image manifest to run exactly as listed, bypassing class/status filtering.",
    )
    parser.add_argument("--results", default=str(DEFAULT_RESULTS), help="OCR result root")
    parser.add_argument(
        "--queue-prefix",
        default=str(WORKLIST_DIR / "gemini_prime_age_gapfill_queue"),
        help="Output prefix for queue .txt and .csv files",
    )
    parser.add_argument(
        "--statuses",
        default="missing,corrupt",
        help="Comma-separated statuses to include directly",
    )
    parser.add_argument(
        "--include-failures-max",
        type=int,
        default=None,
        help="Also include has_failures pages with failures <= N",
    )
    parser.add_argument(
        "--exclude-manifest",
        action="append",
        default=[],
        help="Manifest of image paths to exclude from this queue. May be repeated.",
    )
    parser.add_argument(
        "--skip-key-indexes",
        default="",
        help="Comma-separated Harvard key indexes to leave idle because they are already active.",
    )
    parser.add_argument(
        "--only-key-indexes",
        default="",
        help="Comma-separated Harvard key indexes to use; applied before skip/max-keys.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Limit queued images")
    parser.add_argument("--run", action="store_true", help="Launch OCR after queue build")
    parser.add_argument("--max-keys", type=int, default=0, help="Limit number of keys used")
    parser.add_argument("--workers-per-key", type=int, default=1)
    parser.add_argument(
        "--slots-per-key",
        type=int,
        default=1,
        help="Reserved safety option; must remain 1 so each key has one active slot.",
    )
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--min-interval", type=float, default=1.5)
    parser.add_argument("--max-api-calls", type=int, default=12)
    parser.add_argument("--max-cost", type=float, default=20.0)
    parser.add_argument("--stagger-seconds", type=float, default=1.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    classes = [c.strip() for c in args.classes.split(",") if c.strip()]
    if not args.exact_manifest:
        bad = [c for c in classes if c not in CLASS_MANIFESTS]
        if bad:
            raise SystemExit(f"Unknown class names: {', '.join(bad)}")

    results_root = Path(args.results).resolve()
    if args.exact_manifest:
        rows = build_exact_queue(Path(args.exact_manifest), results_root, args.limit)
    else:
        include_statuses = {s.strip() for s in args.statuses.split(",") if s.strip()}
        exclude_images: set[Path] = set()
        for exclude_manifest in args.exclude_manifest:
            exclude_images.update(read_manifest(Path(exclude_manifest)))
        rows = build_queue(
            classes=classes,
            results_root=results_root,
            include_statuses=include_statuses,
            include_failures_max=args.include_failures_max,
            exclude_images=exclude_images,
            limit=args.limit,
        )
    manifest, audit_csv = write_queue_files(rows, Path(args.queue_prefix))
    print(f"queue_images={len(rows)}")
    print(f"queue_counts={summarize(rows)}")
    print(f"queue_manifest={manifest}")
    print(f"queue_audit={audit_csv}")

    keys = load_harvard_keys(PROJECT_ROOT / ".env")
    key_pairs = list(enumerate(keys))
    only_indexes = {
        int(x.strip())
        for x in args.only_key_indexes.split(",")
        if x.strip()
    }
    if only_indexes:
        key_pairs = [(idx, key) for idx, key in key_pairs if idx in only_indexes]
    skip_indexes = {
        int(x.strip())
        for x in args.skip_key_indexes.split(",")
        if x.strip()
    }
    if skip_indexes:
        key_pairs = [(idx, key) for idx, key in key_pairs if idx not in skip_indexes]
    if args.max_keys > 0:
        key_pairs = key_pairs[: args.max_keys]
    print(f"harvard_keys_available={len(key_pairs)}")
    print(f"harvard_key_indexes={[idx for idx, _ in key_pairs]}")
    if args.run:
        if args.slots_per_key != 1:
            raise SystemExit(
                "--slots-per-key must be 1; Harvard keys are not split across "
                "multiple concurrent OCR subprocesses."
            )
        if not key_pairs:
            raise SystemExit("No HARVARD_GEMINI_API_KEYS available")
        log_dir = Path(args.queue_prefix).parent / (Path(args.queue_prefix).name + "_logs")
        return launch_batches(
            manifest=manifest,
            rows_count=len(rows),
            results_root=results_root,
            key_pairs=key_pairs,
            log_dir=log_dir,
            workers_per_key=args.workers_per_key,
            timeout=args.timeout,
            min_interval=args.min_interval,
            max_api_calls=args.max_api_calls,
            max_cost=args.max_cost,
            stagger_seconds=args.stagger_seconds,
            rerun_failures=any(row.status == "has_failures" for row in rows),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
