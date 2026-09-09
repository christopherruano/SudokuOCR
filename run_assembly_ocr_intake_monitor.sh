#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

MONITOR_DIR="export_final/pdf_recovery_incoming_2026-06-16_assembly_ocr_intake_monitor"
ASSEMBLY_DIR="export_final/pdf_recovery_incoming_2026-06-16_first_principles_assembly_pipeline_v29_logical_review_context"
ITERATIONS="${ASSEMBLY_OCR_INTAKE_ITERATIONS:-72}"
SLEEP_SECONDS="${ASSEMBLY_OCR_INTAKE_SLEEP_SECONDS:-300}"

mkdir -p "$MONITOR_DIR"

for ((i = 1; i <= ITERATIONS; i++)); do
  ts="$(date +%Y%m%d_%H%M%S)"
  out_dir="$MONITOR_DIR/run_${ts}"
  log_path="$MONITOR_DIR/run_${ts}.log"
  python3 build_assembly_ocr_intake_report.py \
    --assembly-dir "$ASSEMBLY_DIR" \
    --extra-root "$HOME/Downloads" \
    --out-dir "$out_dir" \
    > "$log_path" 2>&1
  if [[ "$i" -lt "$ITERATIONS" ]]; then
    sleep "$SLEEP_SECONDS"
  fi
done
