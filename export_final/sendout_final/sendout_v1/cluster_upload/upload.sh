#!/usr/bin/env bash
# Upload bundle to finish the corpus. Run from repo root.
# Uses rsync (not scp) for everything so each multi-file step shares one SSH
# session — FASRC's login nodes drop fresh scp sessions too aggressively.
set -euo pipefail
CL=cruano@login.rc.fas.harvard.edu
SUDOKU=/n/home11/cruano/SudokuOCR
CENSUS=/n/home11/cruano/census_toc
BUNDLE=export_final/sendout_final/sendout_v1/cluster_upload

# 1) CRITICAL: push the fixed prompts FIRST (fixes dropped-cols / zeros)
rsync -av oneshot.py schema_discovery.py batch_production.py digitize_table_pages_task.py $CL:$SUDOKU/

# 2) JOB A re-OCR lists + JOB B new-OCR queue (all to ~/census_toc/)
rsync -av \
  $BUNDLE/reocr_neil_age.txt \
  $BUNDLE/reocr_all_remaining.txt \
  $BUNDLE/newocr_queue.jsonl \
  $BUNDLE/newocr_neil_pages.csv \
  $CL:$CENSUS/

# 3) sbatch scripts -> repo on cluster
rsync -av \
  $BUNDLE/rerun_reocr.sbatch \
  $BUNDLE/newocr_digitize.sbatch \
  $CL:$SUDOKU/fasrc/

echo ''
echo 'LAUNCH (on the cluster):'
echo '  JOB A re-OCR (Neil first):  IMAGE_LIST=$HOME/census_toc/reocr_neil_age.txt sbatch fasrc/rerun_reocr.sbatch'
echo '  JOB A full set:             IMAGE_LIST=$HOME/census_toc/reocr_all_remaining.txt sbatch fasrc/rerun_reocr.sbatch'
echo '  JOB B new-OCR render:       QUEUE=$HOME/census_toc/newocr_queue.jsonl sbatch fasrc/newocr_digitize.sbatch'
