#!/usr/bin/env bash
# ONE Table VII finish job: NEW-OCR (missing pages) first, then RE-OCR (fix existing).
# Run from repo root. Single combined OCR list, new pages at the top.
# Uses rsync (not scp) for everything so each multi-file step shares one SSH
# session — FASRC's login nodes drop fresh scp sessions too aggressively.
set -euo pipefail
CL=cruano@login.rc.fas.harvard.edu
SUDOKU=/n/home11/cruano/SudokuOCR
CENSUS=/n/home11/cruano/census_toc
B=export_final/sendout_final/sendout_v1/cluster_upload

# 1) push the (improved) prompts + digitize task to ~/SudokuOCR/
rsync -av oneshot.py schema_discovery.py batch_production.py digitize_table_pages_task.py $CL:$SUDOKU/

# 2) push targets + schema templates to ~/census_toc/
rsync -av \
  $B/tableVII_newocr.jsonl \
  $B/tableVII_reocr.txt \
  $B/group_schema_templates.json \
  $CL:$CENSUS/

cat <<'REMOTE'

==== ON THE CLUSTER, run in order ====
# PHASE 1 — NEW-OCR: render the missing Table VII pages from the PDFs
QUEUE=$HOME/census_toc/tableVII_newocr.jsonl OUTPUT_DIR=$HOME/census_toc/tableVII_newpages \
  sbatch --wait fasrc/newocr_digitize.sbatch

# PHASE 2 — build ONE combined image list (NEW pages first, then re-OCR crops) and OCR it
find $HOME/census_toc/tableVII_newpages -name '*.png' | sort  >  $HOME/census_toc/tableVII_all.txt
cat $HOME/census_toc/tableVII_reocr.txt                       >> $HOME/census_toc/tableVII_all.txt
# Context-aware OCR: crops with a sibling schema template use the FULL known grid
# (no rediscovery, no silent drops); the rest fall back to the TABLEVII_HINT prime.
# Single initial pass, NO MoE (SUDOKUOCR_MOE_FALLBACK unset).
IMAGE_LIST=$HOME/census_toc/tableVII_all.txt \
  sbatch --export=ALL,SUDOKUOCR_TABLEVII_HINT=1,SUDOKUOCR_SCHEMA_TEMPLATES=$HOME/census_toc/group_schema_templates.json \
  fasrc/rerun_reocr.sbatch
REMOTE
