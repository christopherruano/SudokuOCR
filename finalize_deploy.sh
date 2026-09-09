#!/bin/bash
cd /Users/chrisruano/Research/SudokuOCR
L(){ echo "[$(date +%H:%M)] $*"; }
L "=== FINALIZATION DEPLOY START ==="
L "regenerate data products on the corrected corpus"
python3 compute_severity.py 2>&1 | grep -vE "indexing|images" | tail -1
python3 export_first_cut.py 2>&1 | grep quality | tail -1
python3 build_best_dataset.py 2>&1 | grep -E "best dataset|source kinds" | tail -2
python3 build_assembled_view.py 2>&1 | tail -1
python3 build_single_age.py 2>&1 | grep -vE "indexing|images" | tail -1
python3 build_comparable.py 2>&1 | tail -1
python3 build_static_site.py 2>&1 | tail -3
L "file count check"
N=$(find census_site -type f | wc -l | tr -d ' ')
L "files: $N"
if [ "$N" -gt 20000 ]; then L "WARN >20k files — Cloudflare cap; aborting deploy"; exit 1; fi
L "deploy (accumulating retries — CF caches uploaded assets between attempts)"
for try in $(seq 1 10); do
  OUT=$(npx wrangler pages deploy census_site --project-name=heb91r-5a20dde1 --branch=main --commit-dirty=true 2>&1 | tail -4)
  echo "$OUT"
  echo "$OUT" | grep -q "Deployment complete" && { L "DEPLOYED (attempt $try)"; break; }
  L "retry $try after timeout…"; sleep 120
done
L "=== FINALIZATION DEPLOY COMPLETE ==="
