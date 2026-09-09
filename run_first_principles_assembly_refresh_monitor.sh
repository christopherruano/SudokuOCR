#!/usr/bin/env bash
set -euo pipefail

# Monitor newly arriving OCR artifacts and refresh the first-principles assembly
# outputs only when intake evidence changes. This performs no OCR/API calls.
#
# Environment overrides:
#   ASSEMBLY_REFRESH_MONITOR_ITERATIONS  Number of polling cycles. Default: 24.
#   ASSEMBLY_REFRESH_MONITOR_SLEEP       Seconds between cycles. Default: 300.
#   ASSEMBLY_REFRESH_MONITOR_TAG_PREFIX  Output tag prefix. Default: monitor.
#   ASSEMBLY_DIR                         Existing first-principles assembly source.
#   EXTRA_RENDER_ROOT                    Render/artifact root. Default: ~/Downloads.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  sed -n '1,18p' "$0"
  exit 0
fi

ITERATIONS="${ASSEMBLY_REFRESH_MONITOR_ITERATIONS:-24}"
SLEEP_SECONDS="${ASSEMBLY_REFRESH_MONITOR_SLEEP:-300}"
TAG_PREFIX="${ASSEMBLY_REFRESH_MONITOR_TAG_PREFIX:-monitor}"
ASSEMBLY_DIR="${ASSEMBLY_DIR:-export_final/pdf_recovery_incoming_2026-06-16_first_principles_assembly_pipeline_v29_logical_review_context}"
EXTRA_RENDER_ROOT="${EXTRA_RENDER_ROOT:-$HOME/Downloads}"
LOG_DIR="export_final/assembly_overnight_monitor_logs"

mkdir -p "$LOG_DIR"

latest_release_summary() {
  python3 - <<'PY'
from pathlib import Path
paths = sorted(
    Path("export_final").glob("pdf_recovery_incoming_2026-06-16_first_principles_assembly_release_snapshot_*/summary.json"),
    key=lambda p: p.stat().st_mtime,
)
print(paths[-1] if paths else "")
PY
}

summary_metric() {
  local summary_path="$1"
  local key="$2"
  python3 - "$summary_path" "$key" <<'PY'
import json
import sys
from pathlib import Path
path = Path(sys.argv[1])
key = sys.argv[2]
if not path.exists():
    print(0)
else:
    print(json.loads(path.read_text()).get(key, 0) or 0)
PY
}

summary_value() {
  local summary_path="$1"
  local key="$2"
  python3 - "$summary_path" "$key" <<'PY'
import json
import sys
from pathlib import Path
path = Path(sys.argv[1])
key = sys.argv[2]
if not path.exists():
    print("")
else:
    value = json.loads(path.read_text()).get(key, "")
    print("" if value is None else value)
PY
}

for ((i = 1; i <= ITERATIONS; i++)); do
  ts="$(date +%Y%m%d_%H%M%S)"
  intake_dir="export_final/pdf_recovery_incoming_2026-06-16_assembly_ocr_intake_${TAG_PREFIX}_${ts}"
  intake_log="$LOG_DIR/intake_${ts}.log"

  python3 build_assembly_ocr_intake_report.py \
    --assembly-dir "$ASSEMBLY_DIR" \
    --extra-root "$EXTRA_RENDER_ROOT" \
    --out-dir "$intake_dir" \
    > "$intake_log" 2>&1

  latest_summary="$(latest_release_summary)"
  current_matches="$(summary_metric "$intake_dir/summary.json" targets_with_artifact_match)"
  previous_matches="$(summary_metric "$latest_summary" intake_targets_with_artifact_match)"
  current_artifact_fingerprint="$(summary_value "$intake_dir/summary.json" artifact_match_fingerprint)"
  previous_artifact_fingerprint="$(summary_value "$latest_summary" intake_artifact_match_fingerprint)"
  current_target_fingerprint="$(summary_value "$intake_dir/summary.json" target_state_fingerprint)"
  previous_target_fingerprint="$(summary_value "$latest_summary" intake_target_state_fingerprint)"

  {
    printf 'timestamp=%s\n' "$ts"
    printf 'iteration=%s/%s\n' "$i" "$ITERATIONS"
    printf 'intake_dir=%s\n' "$intake_dir"
    printf 'latest_release_summary=%s\n' "$latest_summary"
    printf 'current_targets_with_artifact_match=%s\n' "$current_matches"
    printf 'previous_release_targets_with_artifact_match=%s\n' "$previous_matches"
    printf 'current_artifact_match_fingerprint=%s\n' "$current_artifact_fingerprint"
    printf 'previous_release_artifact_match_fingerprint=%s\n' "$previous_artifact_fingerprint"
    printf 'current_target_state_fingerprint=%s\n' "$current_target_fingerprint"
    printf 'previous_release_target_state_fingerprint=%s\n' "$previous_target_fingerprint"
  } >> "$LOG_DIR/monitor_refresh.log"

  if (( current_matches > previous_matches )) \
    || [[ -n "$current_artifact_fingerprint" && "$current_artifact_fingerprint" != "$previous_artifact_fingerprint" ]] \
    || [[ -n "$current_target_fingerprint" && "$current_target_fingerprint" != "$previous_target_fingerprint" ]]; then
    refresh_tag="${TAG_PREFIX}_${ts}"
    refresh_log="$LOG_DIR/refresh_${ts}.log"
    ASSEMBLY_DIR="$ASSEMBLY_DIR" EXTRA_RENDER_ROOT="$EXTRA_RENDER_ROOT" \
      ./run_first_principles_assembly_rulebook_refresh.sh "$refresh_tag" \
      > "$refresh_log" 2>&1
    printf 'refresh_tag=%s\n' "$refresh_tag" >> "$LOG_DIR/monitor_refresh.log"
  else
    printf 'refresh_tag=skipped_no_new_matches\n' >> "$LOG_DIR/monitor_refresh.log"
  fi

  if [[ "$i" -lt "$ITERATIONS" ]]; then
    sleep "$SLEEP_SECONDS"
  fi
done
