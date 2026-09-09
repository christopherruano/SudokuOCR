#!/usr/bin/env bash
cd "$(dirname "$0")"
while pgrep -f "results export/boundary_gap_ocr" >/dev/null 2>&1; do sleep 30; done
echo "$(date '+%H:%M:%S') boundary-gap OCR done; re-running detection" >> fasrc/logs/finalize_boundaries.log
python3 build_tableVII_runs.py >> fasrc/logs/finalize_boundaries.log 2>&1
pkill -f census_review_app.py 2>/dev/null; sleep 1
nohup python3 census_review_app.py > fasrc/logs/review_app.log 2>&1 &
echo "$(date '+%H:%M:%S') re-detection done, review app restarted" >> fasrc/logs/finalize_boundaries.log
