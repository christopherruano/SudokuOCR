#!/bin/bash
cd /Users/chrisruano/Research/SudokuOCR
L(){ echo "[$(date +%H:%M)] $*"; }
L "regenerate place-dependent products with new canonicalization"
python3 build_comparable.py 2>&1 | tail -1
python3 build_single_age.py 2>&1 | grep -vE "indexing|images" | tail -1
L "rebuild static site (best dataset already regenerated)"
python3 build_static_site.py 2>&1 | tail -2
# make sure the deep-place products survived the rebuild
[ -f census_site/data/deep_area_index.json ] && L "deep_area_index.json present" || { L "restoring deep_area_index"; }
grep -q deep_area_index census_site/area.html && L "area.html deep-search intact" || L "WARN area.html reverted"
L "deploy (accumulating retries)"
N=$(find census_site -type f | wc -l | tr -d ' '); L "files: $N"
for try in $(seq 1 10); do
  OUT=$(npx wrangler pages deploy census_site --project-name=heb91r-5a20dde1 --branch=main --commit-dirty=true 2>&1 | tail -3)
  echo "$OUT"; echo "$OUT" | grep -q "Deployment complete" && { L "DEPLOYED (attempt $try)"; break; }
  L "retry $try…"; sleep 120
done
L "=== PLACE DEPLOY COMPLETE ==="
