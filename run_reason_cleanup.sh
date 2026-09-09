#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")"
LOG=fasrc/logs/reason_cleanup.log
exec >> "$LOG" 2>&1
echo "===== $(date '+%F %T') reasoning cleanup on de-frozen tables (max-fail-rows 16) ====="
python3 constraint_guided_reocr.py --reason --apply --max-fail-rows 16 2>&1 | grep -vE "NotOpenSSL|warnings|urllib3|v2 only|non-text|thought" | tail -50
echo "----- rebuild -----"
python3 detect_misfires.py 2>&1 | head -1
python3 build_broken_queue.py 2>&1 | head -1
echo "===== DONE $(date '+%F %T') ====="
