#!/usr/bin/env bash
# The DB is already rebuilt (full, 10k+ tables, slim exclusions dropped). Render the FULL static site
# directly — skipping finish_and_build.sh's OCR-drain wrapper (which aborts when no OCR is running).
set -uo pipefail
cd "$(dirname "$0")"
LOG=fasrc/logs/full_deploy_build.log
exec >> "$LOG" 2>&1
say(){ echo; echo "$(date '+%H:%M:%S') ===== $* ====="; }

say "FINISH FULL SITE — clean render of every non-excluded page"
rm -rf census_site/tables census_site/data && mkdir -p census_site/tables census_site/data
python3 prune_orphan_pages.py --apply 2>/dev/null || true

say "build_static_site (ALL pages)"
if ! python3 build_static_site.py; then echo "BUILD_STATUS=FAILED:build_static_site"; exit 1; fi

say "highlights + coverage + AI-supervised assembled view"
python3 build_highlights.py 2>/dev/null || true
python3 build_coverage_nav.py 2>/dev/null || true
python3 build_assembled_view.py 2>/dev/null || true

N=$(find census_site -type f | wc -l | tr -d ' ')
say "DONE — census_site: $N files, $(du -sh census_site | cut -f1), tables $(ls census_site/tables | wc -l | tr -d ' ')"
[ "$N" -le 20000 ] && echo "BUILD_STATUS=READY files=$N" || echo "BUILD_STATUS=OVER_20k_CAP files=$N"
