#!/bin/bash
# Robust runner: waits for the Expansion drive, runs the confirmed-range drive OCR pass, retries until
# nothing is left (the script skips already-done pages, so re-runs are idempotent). Survives drive drops.
cd /Users/chrisruano/Research/SudokuOCR
for round in $(seq 1 48); do
  if [ ! -d /Volumes/Expansion/census_pdfs ]; then
    echo "[$(date +%H:%M)] round $round: drive not mounted — waiting 120s"
    sleep 120; continue
  fi
  echo "[$(date +%H:%M)] round $round: drive OK — running confirmed-range drive pass"
  python3 ocr_confirmed_drive.py 2>&1 | grep -vE "indexing|images indexed|NotOpenSSL|warnings.warn"
  # done when no page in the need-drive list is still missing
  left=$(python3 - <<'PY' 2>/dev/null | tail -1
import sys; sys.argv=["x"]
import boundary_apply as B
n=0
for line in open("export/docs_need_drive.txt"):
    p=line.strip().split("\t")
    if len(p)>=3 and p[2].isdigit() and not B.member_for(p[1],int(p[2]))[1]: n+=1
print(n)
PY
)
  echo "[$(date +%H:%M)] still missing: $left"
  [ "$left" = "0" ] && { echo "ALL CONFIRMED-RANGE PAGES OCR'D ✅"; break; }
  sleep 60
done
