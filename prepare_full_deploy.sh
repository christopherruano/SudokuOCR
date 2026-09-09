#!/usr/bin/env bash
# Build the FULL deployable census_site (all pages, not slim) from the current corpus: rebuild the DB
# via the reingest, render the site, then overlay the AI-supervised assembled view. Idempotent-ish.
set -uo pipefail
cd "$(dirname "$0")"
export SUDOKUOCR_SKIP_PLAIN_FACTS=1
LOG=fasrc/logs/full_deploy_build.log
exec > "$LOG" 2>&1
say(){ echo; echo "$(date '+%H:%M:%S') ===== $* ====="; }

say "FULL DEPLOY BUILD START (disk $(df -h / | awk 'NR==2{print $4}') free)"
say "restore FULL page set (drop slim exclusions)"
python3 build_slim_exclude.py --full || true

say "reingest: rebuild DB from the current (deduped) corpus"
if ! bash reingest.sh; then echo "BUILD_STATUS=FAILED:reingest"; exit 1; fi

say "build static site (all pages) + highlights"
if ! bash finish_and_build.sh; then echo "BUILD_STATUS=FAILED:finish_and_build"; exit 1; fi

say "overlay the AI-supervised assembled view + coverage nav"
python3 build_assembled_view.py || true
python3 build_coverage_nav.py 2>/dev/null || true

N=$(find census_site -type f | wc -l | tr -d ' ')
say "DONE — census_site: $N files, $(du -sh census_site | cut -f1)"
[ "$N" -le 20000 ] && echo "BUILD_STATUS=READY files=$N" || echo "BUILD_STATUS=OVER_20k_CAP files=$N"
