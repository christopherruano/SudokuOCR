#!/bin/bash
cd /Users/chrisruano/Research/SudokuOCR
L(){ echo "[$(date +%H:%M)] $*"; }
python3 compute_severity.py 2>&1 | grep -vE "indexing|images" | tail -1
python3 export_first_cut.py 2>&1 | grep quality | tail -1
python3 build_best_dataset.py 2>&1 | grep -E "best dataset|source kinds" | tail -2
python3 build_assembled_view.py 2>&1 | tail -1
python3 build_static_site.py 2>&1 | tail -2
[ -f census_site/data/deep_area_index.json ] && L "deep_area_index OK" || L "WARN deep index missing"
N=$(find census_site -type f | wc -l | tr -d ' '); L "files: $N"
for try in $(seq 1 10); do
  OUT=$(npx wrangler pages deploy census_site --project-name=heb91r-5a20dde1 --branch=main --commit-dirty=true 2>&1 | tail -3)
  echo "$OUT"; echo "$OUT" | grep -q "Deployment complete" && { L "DEPLOYED (try $try)"; break; }
  sleep 120
done
L "=== PRIMEFIX DEPLOY COMPLETE ==="
