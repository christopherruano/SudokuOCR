#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")"
LOG=fasrc/logs/mistral_frozen.log
exec >> "$LOG" 2>&1
echo "===== $(date '+%F %T') Mistral fallback on Table VII frozen pages ====="
python3 mistral_frozen.py --apply 2>&1 | grep -vE "NotOpenSSL|warnings.warn|urllib3|v2 only" | tail -45
echo "----- frozen-aware merge already done inline; rebuild queue + re-detect -----"
python3 detect_misfires.py 2>&1 | head -1
python3 build_broken_queue.py 2>&1 | head -1
python3 build_assembled_view.py 2>&1 | tail -1
echo "===== DONE $(date '+%F %T') ====="
