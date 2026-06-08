#!/usr/bin/env bash
# Upload bundle to finish the corpus. Run from repo root.
set -euo pipefail
CL=cruano@login.rc.fas.harvard.edu
DEST=/n/home11/cruano/census_toc

# 1) CRITICAL: push the fixed prompts FIRST (fixes dropped-cols / zeros)
rsync -av oneshot.py schema_discovery.py batch_production.py $CL:$DEST/

# 2) JOB A re-OCR lists + JOB B new-OCR queue
scp /Users/chrisruano/Research/SudokuOCR/export_final/sendout_final/sendout_v1/cluster_upload/reocr_neil_age.txt        $CL:$DEST/reocr_neil_age.txt
scp /Users/chrisruano/Research/SudokuOCR/export_final/sendout_final/sendout_v1/cluster_upload/reocr_all_remaining.txt   $CL:$DEST/reocr_all_remaining.txt
scp /Users/chrisruano/Research/SudokuOCR/export_final/sendout_final/sendout_v1/cluster_upload/newocr_queue.jsonl        $CL:$DEST/newocr_queue.jsonl
scp /Users/chrisruano/Research/SudokuOCR/export_final/sendout_final/sendout_v1/cluster_upload/newocr_neil_pages.csv     $CL:$DEST/newocr_neil_pages.csv

# 3) sbatch scripts -> repo on cluster
scp /Users/chrisruano/Research/SudokuOCR/export_final/sendout_final/sendout_v1/cluster_upload/rerun_reocr.sbatch /Users/chrisruano/Research/SudokuOCR/export_final/sendout_final/sendout_v1/cluster_upload/newocr_digitize.sbatch $CL:$DEST/../SudokuOCR/fasrc/ 2>/dev/null || true

echo ''
echo 'LAUNCH (on the cluster):'
echo '  JOB A re-OCR (Neil first):  IMAGE_LIST=$HOME/census_toc/reocr_neil_age.txt sbatch fasrc/rerun_reocr.sbatch'
echo '  JOB A full set:             IMAGE_LIST=$HOME/census_toc/reocr_all_remaining.txt sbatch fasrc/rerun_reocr.sbatch'
echo '  JOB B new-OCR render:       QUEUE=$HOME/census_toc/newocr_queue.jsonl sbatch fasrc/newocr_digitize.sbatch'
