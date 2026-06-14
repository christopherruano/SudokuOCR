#!/usr/bin/env bash
# Re-ingest loop: run AFTER rsync'ing the cluster re-OCR + new-OCR results into
#   export/ocr_results_referenced/   (re-OCR'd crops, overwritten in place)
# Refreshes the keeper extract -> re-merges -> re-QAs -> re-assembles multi-page
# tables -> rebuilds the explorer -> re-validates Table VII (measures the lift).
# Idempotent; stops on first error. Each step timed.
set -euo pipefail
cd "$(dirname "$0")"
say(){ echo; echo "=== $* === ($(date +%H:%M:%S))"; }

say "1/8 re-extract keeper cell facts from the (re-OCR'd) results"
python3 export_inference_dataset.py \
  --organized-root export/ocr_results_referenced \
  --output-dir export/keeper_inference \
  --include-unvalidated --include-failed

say "2/8 rebuild provenance backbone (dochash -> PDF, page numbers)"
python3 build_provenance.py

say "3/8 merge keepers into explorer-format facts + catalog"
python3 merge_keepers.py

say "4/8 deterministic red-flag detection on the fresh reads"
python3 detect_keeper_badreads.py
python3 apply_keeper_flags.py

say "5/8 re-derive sibling schema templates (now from improved reads)"
python3 build_group_schema_templates.py

say "6/8 re-assemble multi-page tables + make sections cohesive + wire local scans"
python3 build_assembled_tables.py
python3 clean_assembled_sections.py
python3 patch_scan_paths.py

say "7/8 rebuild the explorer SQLite (good-only)"
python3 build_explorer_db.py

say "8/10 re-validate Table VII (the scorecard)"
python3 build_tableVII_plan.py
python3 validate_tableVII.py

say "9/10 categorize imperfect by ROOT CAUSE + merge half-page splits & stitch continuations (free, no re-OCR)"
python3 categorize_imperfect.py
python3 build_halfpage_groups.py
python3 build_continuation_groups.py
# re-assemble so half-page column-splits are unioned + fragmented pages stitched, then rebuild + re-validate
python3 build_assembled_tables.py
python3 clean_assembled_sections.py
python3 patch_scan_paths.py
python3 build_explorer_db.py
python3 validate_tableVII.py
python3 categorize_imperfect.py

say "10/10 build the PRECISE re-OCR queue + FINAL VERDICT ledger (perfect / unrecoverable / retrying)"
python3 build_genuine_reocr_queue.py
python3 build_verdicts.py

say "DONE — review validate_tableVII output above for the perfect-rate lift."
echo "Next: 'python3 build_static_site.py' to refresh the shareable site."
echo "Cluster: upload tableVII_genuine_reocr.txt (+ tableVII_genuine_schema.json) and re-OCR."
