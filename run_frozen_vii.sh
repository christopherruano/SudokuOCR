#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")"
LOG=fasrc/logs/frozen_vii.log
exec >> "$LOG" 2>&1
echo "===== $(date '+%F %T') de-freeze Table VII (Mistral->structure map) ====="
python3 fix_frozen_vii.py --apply 2>&1 | grep -vE "NotOpenSSL|warnings|urllib3|v2 only|non-text|thought_sig"
echo "----- re-detect + rebuild -----"
python3 detect_misfires.py 2>&1 | head -1
python3 build_broken_queue.py 2>&1 | head -1
python3 build_assembled_view.py 2>&1 | tail -1
echo "===== DONE $(date '+%F %T') ====="
