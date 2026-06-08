#!/usr/bin/env bash
# PRIORITY #1 — perfect every Table VII (Age, Sex & Civil Condition). Run from repo root.
set -euo pipefail
CL=cruano@login.rc.fas.harvard.edu
DEST=/n/home11/cruano/census_toc
# 1) fixed prompts FIRST (fixes dropped U/M/W columns + zeros)
rsync -av oneshot.py schema_discovery.py batch_production.py $CL:$DEST/
# 2) Table VII targets
scp export_final/sendout_final/sendout_v1/cluster_upload/tableVII_reocr.txt   $CL:$DEST/tableVII_reocr.txt
scp export_final/sendout_final/sendout_v1/cluster_upload/tableVII_newocr.jsonl $CL:$DEST/tableVII_newocr.jsonl
echo ''
echo 'LAUNCH on cluster (Table VII first):'
echo '  re-OCR 962 imperfect/flagged:  IMAGE_LIST=$HOME/census_toc/tableVII_reocr.txt sbatch fasrc/rerun_reocr.sbatch'
echo '  new-OCR missing VII pages:      QUEUE=$HOME/census_toc/tableVII_newocr.jsonl sbatch fasrc/newocr_digitize.sbatch'
