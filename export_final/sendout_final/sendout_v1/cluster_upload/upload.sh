#!/usr/bin/env bash
# Upload bundle to finish the corpus. Run from repo root.
set -euo pipefail
CL=cruano@login.rc.fas.harvard.edu
DEST=/n/home11/cruano/census_toc

# 1) CRITICAL: push the fixed prompts FIRST (fixes dropped-cols / zeros)
rsync -av oneshot.py schema_discovery.py batch_production.py $CL:$DEST/

# 2) image lists + page targets + sbatch
scp /Users/chrisruano/Research/SudokuOCR/export_final/sendout_final/sendout_v1/cluster_upload/reocr_neil_age.txt        $CL:$DEST/reocr_neil_age.txt
scp /Users/chrisruano/Research/SudokuOCR/export_final/sendout_final/sendout_v1/cluster_upload/reocr_all_remaining.txt   $CL:$DEST/reocr_all_remaining.txt
scp /Users/chrisruano/Research/SudokuOCR/export_final/sendout_final/sendout_v1/cluster_upload/newocr_neil_pages.csv     $CL:$DEST/newocr_neil_pages.csv
scp /Users/chrisruano/Research/SudokuOCR/export_final/sendout_final/sendout_v1/cluster_upload/rerun_reocr.sbatch        $CL:$DEST/../SudokuOCR/fasrc/ 2>/dev/null || true

# 3) launch (on the cluster):
echo 'on cluster (Neil first):  IMAGE_LIST=$HOME/census_toc/reocr_neil_age.txt sbatch fasrc/rerun_reocr.sbatch'
echo 'then the full set:        IMAGE_LIST=$HOME/census_toc/reocr_all_remaining.txt sbatch fasrc/rerun_reocr.sbatch'
