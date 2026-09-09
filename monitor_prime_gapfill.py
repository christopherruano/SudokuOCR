#!/usr/bin/env python3
"""Conservative monitor/recycler for prime-age Gemini OCR gap filling.

This wrapper is intentionally small and auditable. It does not change the OCR
engine; it coordinates the existing audit and exact-manifest runner so idle
Harvard Gemini keys are used without duplicating pages already in flight.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from run_gemini_prime_gapfill import PROJECT_ROOT, WORKLIST_DIR, load_harvard_keys


AUDIT_PREFIX = WORKLIST_DIR / "gemini_best_of_main_derivative_live"
SUMMARY_JSON = AUDIT_PREFIX.with_name(AUDIT_PREFIX.name + "_summary.json")
REMAINING_CSV = AUDIT_PREFIX.with_name(AUDIT_PREFIX.name + "_remaining_failures.csv")
ACTIVE_MANIFEST = WORKLIST_DIR / "gemini_active_precise_current.txt"
TRIAGE_CSV = WORKLIST_DIR / "gemini_best_of_main_derivative_live_failure_triage.csv"
MANUAL_TRIAGE_CSV = WORKLIST_DIR / "gemini_prime_manual_triage.csv"


@dataclass(frozen=True)
class ActiveBatch:
    pid: int
    parent_pid: int
    manifest: Path
    chunk: int | None
    total_chunks: int | None
    images: list[str]
    key_index: int | None


def run_text(cmd: list[str]) -> str:
    return subprocess.check_output(cmd, cwd=PROJECT_ROOT, text=True)


def shell_text(cmd: str) -> str:
    return subprocess.check_output(cmd, cwd=PROJECT_ROOT, shell=True, text=True)


def parse_index_list(raw: str) -> list[int]:
    return sorted(int(x.strip()) for x in raw.split(",") if x.strip())


def extract_arg(command: str, name: str) -> str | None:
    m = re.search(rf"{re.escape(name)}\s+(\S+)", command)
    return m.group(1) if m else None


def slice_chunk(items: list[str], chunk: int | None, total: int | None) -> list[str]:
    if chunk is None or total is None:
        return items
    n = len(items)
    base = n // total
    remainder = n % total
    if chunk < remainder:
        start = chunk * (base + 1)
        end = start + base + 1
    else:
        start = remainder * (base + 1) + (chunk - remainder) * base
        end = start + base
    return items[start:end]


def load_manifest(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text(errors="ignore").splitlines() if line.strip()]


def ps_rows() -> list[tuple[int, int, str]]:
    raw = shell_text("ps -axo pid,ppid,command -ww")
    rows: list[tuple[int, int, str]] = []
    for line in raw.splitlines()[1:]:
        parts = line.strip().split(None, 2)
        if len(parts) < 3:
            continue
        try:
            rows.append((int(parts[0]), int(parts[1]), parts[2]))
        except ValueError:
            continue
    return rows


def active_batches() -> list[ActiveBatch]:
    rows = ps_rows()
    commands_by_pid = {pid: command for pid, _, command in rows}
    out: list[ActiveBatch] = []
    for pid, ppid, command in rows:
        if "batch_production.py" not in command:
            continue
        manifest_raw = extract_arg(command, "--image-list")
        if not manifest_raw:
            continue
        manifest = Path(manifest_raw)
        if not manifest.is_absolute():
            manifest = (PROJECT_ROOT / manifest).resolve()
        chunk_raw = extract_arg(command, "--chunk")
        total_raw = extract_arg(command, "--total-chunks")
        chunk = int(chunk_raw) if chunk_raw is not None else None
        total = int(total_raw) if total_raw is not None else None
        all_images = load_manifest(manifest)
        images = slice_chunk(all_images, chunk, total)

        key_index: int | None = None
        parent_command = commands_by_pid.get(ppid, "")
        only_raw = extract_arg(parent_command, "--only-key-indexes")
        if only_raw and chunk is not None:
            indexes = parse_index_list(only_raw)
            if 0 <= chunk < len(indexes):
                key_index = indexes[chunk]

        out.append(
            ActiveBatch(
                pid=pid,
                parent_pid=ppid,
                manifest=manifest,
                chunk=chunk,
                total_chunks=total,
                images=images,
                key_index=key_index,
            )
        )
    return out


def write_active_manifest(batches: list[ActiveBatch]) -> set[str]:
    seen: dict[str, None] = {}
    for batch in batches:
        for image in batch.images:
            seen.setdefault(image, None)
    ACTIVE_MANIFEST.write_text("\n".join(seen) + ("\n" if seen else ""), encoding="utf-8")
    return set(seen)


def memory_throttled_pages() -> int:
    raw = shell_text("vm_stat | rg '^Pages throttled:' || true")
    m = re.search(r"Pages throttled:\s+([0-9]+)", raw)
    return int(m.group(1)) if m else 0


def run_audit() -> dict[str, Any]:
    subprocess.check_call(
        [
            sys.executable,
            "audit_gemini_best_of.py",
            "--classes",
            "main,derivative",
            "--output-prefix",
            str(AUDIT_PREFIX),
        ],
        cwd=PROJECT_ROOT,
    )
    return json.loads(SUMMARY_JSON.read_text(encoding="utf-8"))


def remaining_rows() -> list[dict[str, str]]:
    if not REMAINING_CSV.exists():
        return []
    with REMAINING_CSV.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def best_failure_details(path: str, max_items: int = 3) -> str:
    p = Path(path)
    if not p.exists():
        return ""
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:
        return f"json_error={exc}"
    failures = (data.get("constraints") or {}).get("failures") or []
    details: list[str] = []
    for failure in failures[:max_items]:
        pieces = [
            str(failure.get("level", "")),
            str(failure.get("section", "")),
            str(failure.get("age", "")),
            str(failure.get("group", "")),
            str(failure.get("detail", "")),
            f"diff={failure.get('diff', '')}",
        ]
        details.append(" | ".join(pieces))
    return " ; ".join(details)


def attempt_count(row: dict[str, str]) -> int:
    statuses = row.get("candidate_statuses", "")
    image_statuses = [
        x for x in statuses.split(";") if ":has_failures:" in x or ":accepted_" in x
    ]
    return len(image_statuses)


def classify_failure(row: dict[str, str]) -> str:
    details = best_failure_details(row.get("best_result_json", ""), max_items=2).lower()
    failures = int(row.get("best_failures") or 0)
    if "table viii" in details:
        return "off_target_table_viii_or_derivative"
    if "percentage" in details or "subsidiary table" in details:
        return "derivative_or_percentage_semantics"
    if failures <= 2:
        return "low_residual_manual_or_printed_inconsistency"
    if failures <= 12:
        return "medium_residual_recheck_or_structure"
    return "structural_or_table_shape"


def write_triage(rows: list[dict[str, str]], active_images: set[str]) -> None:
    manual = load_manual_triage()
    fields = [
        "priority_class",
        "image_path",
        "source",
        "page",
        "best_failures",
        "best_checks",
        "best_root",
        "attempt_count",
        "active",
        "manual_decision",
        "neil_blocking",
        "auto_rerun",
        "triage_bucket",
        "first_failures",
        "best_result_json",
    ]
    with TRIAGE_CSV.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            manual_row = manual.get((row.get("source", ""), row.get("page", "")), {})
            writer.writerow(
                {
                    "priority_class": row.get("priority_class", ""),
                    "image_path": row.get("image_path", ""),
                    "source": row.get("source", ""),
                    "page": row.get("page", ""),
                    "best_failures": row.get("best_failures", ""),
                    "best_checks": row.get("best_checks", ""),
                    "best_root": row.get("best_root", ""),
                    "attempt_count": attempt_count(row),
                    "active": row.get("image_path", "") in active_images,
                    "manual_decision": manual_row.get("decision", ""),
                    "neil_blocking": manual_row.get("neil_blocking", ""),
                    "auto_rerun": manual_row.get("auto_rerun", ""),
                    "triage_bucket": classify_failure(row),
                    "first_failures": best_failure_details(row.get("best_result_json", "")),
                    "best_result_json": row.get("best_result_json", ""),
                }
            )


def load_manual_triage() -> dict[tuple[str, str], dict[str, str]]:
    if not MANUAL_TRIAGE_CSV.exists():
        return {}
    with MANUAL_TRIAGE_CSV.open(encoding="utf-8", newline="") as fh:
        return {
            (row.get("source", ""), row.get("page", "")): row
            for row in csv.DictReader(fh)
        }


def manual_priority(row: dict[str, str], manual: dict[tuple[str, str], dict[str, str]]) -> int:
    decision = manual.get((row.get("source", ""), row.get("page", "")), {}).get("decision", "")
    if decision == "true_prime_table":
        return 0
    if decision == "manual_review_if_still_fails":
        return 1
    if decision == "lower_priority_derivative":
        return 5
    if row.get("priority_class") == "derivative":
        return 6
    return 2


def triage_summary(rows: list[dict[str, str]]) -> dict[str, int]:
    manual = load_manual_triage()
    summary = {
        "neil_blocking": 0,
        "manual_review": 0,
        "lower_priority_derivative": 0,
        "not_neil": 0,
        "unclassified": 0,
        "auto_rerun_no": 0,
    }
    for row in rows:
        manual_row = manual.get((row.get("source", ""), row.get("page", "")), {})
        decision = manual_row.get("decision", "")
        auto_rerun = manual_row.get("auto_rerun", "yes")
        if auto_rerun == "no":
            summary["auto_rerun_no"] += 1
        if decision == "not_neil_table":
            summary["not_neil"] += 1
        elif decision == "lower_priority_derivative":
            summary["lower_priority_derivative"] += 1
        elif decision == "manual_review_if_still_fails":
            summary["manual_review"] += 1
            summary["neil_blocking"] += 1
        elif decision == "true_prime_table":
            summary["neil_blocking"] += 1
        else:
            summary["unclassified"] += 1
            if row.get("priority_class") == "main":
                summary["neil_blocking"] += 1
            else:
                summary["lower_priority_derivative"] += 1
    return summary


def launch_uncovered(
    rows: list[dict[str, str]],
    active_images: set[str],
    idle_keys: list[int],
    *,
    max_attempts: int,
    run_name: str,
    timeout: int,
) -> bool:
    if not idle_keys:
        return False
    manual = load_manual_triage()
    uncovered = [r for r in rows if r.get("image_path") not in active_images]
    eligible = [
        r
        for r in uncovered
        if attempt_count(r) < max_attempts
        and manual.get((r.get("source", ""), r.get("page", "")), {}).get("auto_rerun", "yes") != "no"
    ]
    if not eligible:
        return False
    eligible.sort(
        key=lambda r: (
            manual_priority(r, manual),
            int(r.get("best_failures") or 9999),
            r.get("priority_class", ""),
            r.get("source", ""),
            int(r.get("page") or -1),
        )
    )
    selected = eligible[: len(idle_keys)]
    key_slice = idle_keys[: len(selected)]
    prefix = WORKLIST_DIR / run_name
    manifest = prefix.with_suffix(".txt")
    manifest.write_text(
        "\n".join(r["image_path"] for r in selected) + "\n",
        encoding="utf-8",
    )
    result_root = PROJECT_ROOT / "export_final" / f"gemini_no_gap_ocr_2026-06-16_v2_rerun_{run_name}"
    cmd = [
        sys.executable,
        "run_gemini_prime_gapfill.py",
        "--exact-manifest",
        str(manifest),
        "--results",
        str(result_root),
        "--queue-prefix",
        str(prefix),
        "--only-key-indexes",
        ",".join(str(k) for k in key_slice),
        "--workers-per-key",
        "1",
        "--slots-per-key",
        "1",
        "--timeout",
        str(timeout),
        "--run",
    ]
    launch_log = prefix.with_suffix(".launch.log")
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    with launch_log.open("ab") as log_fh:
        log_fh.write(("\n" + "=" * 80 + "\n").encode("utf-8"))
        log_fh.write((f"launch_time={time.strftime('%Y-%m-%d %H:%M:%S')}\n").encode("utf-8"))
        log_fh.write(("cmd=" + " ".join(cmd) + "\n").encode("utf-8"))
        for row in selected:
            log_fh.write((f"selected failures={row['best_failures']} image={row['image_path']}\n").encode("utf-8"))
        log_fh.flush()
        subprocess.Popen(
            cmd,
            cwd=PROJECT_ROOT,
            env=env,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    print(f"launched run_name={run_name} images={len(selected)} keys={key_slice} launch_log={launch_log}")
    for row in selected:
        print(f"  {row['best_failures']} failures {row['image_path']}")
    return True


def one_cycle(args: argparse.Namespace) -> dict[str, Any]:
    throttled = memory_throttled_pages()
    batches = active_batches()
    active_images = write_active_manifest(batches)
    active_keys = sorted({b.key_index for b in batches if b.key_index is not None})
    all_keys = list(range(len(load_harvard_keys(PROJECT_ROOT / ".env"))))
    idle_keys = [k for k in all_keys if k not in active_keys]
    summary = run_audit()
    rows = remaining_rows()
    write_triage(rows, active_images)
    uncovered = [r for r in rows if r.get("image_path") not in active_images]
    triage = triage_summary(rows)

    print(
        "status "
        f"accepted={summary.get('accepted')} remaining={summary.get('remaining')} "
        f"active_batches={len(batches)} active_keys={active_keys} idle_keys={idle_keys} "
        f"active_remaining={len(rows) - len(uncovered)} uncovered_remaining={len(uncovered)} "
        f"pages_throttled={throttled} "
        f"neil_blocking={triage['neil_blocking']} manual_review={triage['manual_review']} "
        f"lower_derivative={triage['lower_priority_derivative']} not_neil={triage['not_neil']} "
        f"auto_rerun_no={triage['auto_rerun_no']}"
    )
    print(f"active_manifest={ACTIVE_MANIFEST}")
    print(f"triage_csv={TRIAGE_CSV}")

    if args.launch and throttled <= args.max_pages_throttled:
        run_name = f"gemini_prime_monitor_{time.strftime('%Y%m%d_%H%M%S')}"
        launch_uncovered(
            rows,
            active_images,
            idle_keys,
            max_attempts=args.max_attempts,
            run_name=run_name,
            timeout=args.timeout,
        )
    elif args.launch:
        print(f"skip_launch memory throttled pages {throttled} > {args.max_pages_throttled}")

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch", action="store_true", help="Launch uncovered remaining pages on idle keys")
    parser.add_argument("--loop", action="store_true", help="Repeat cycles until interrupted")
    parser.add_argument("--interval", type=int, default=300)
    parser.add_argument("--timeout", type=int, default=2700)
    parser.add_argument("--max-attempts", type=int, default=8)
    parser.add_argument("--max-pages-throttled", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    while True:
        one_cycle(args)
        if not args.loop:
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
