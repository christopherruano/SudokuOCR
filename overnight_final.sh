#!/bin/bash
cd /Users/chrisruano/Research/SudokuOCR
L(){ echo "[$(date +%H:%M)] $*"; }
L "waiting for subagent re-read fleet (target: 52 verdicts, max 150 min)"
for i in $(seq 1 60); do
  n=$(ls export/agent_rereads/*.json 2>/dev/null | wc -l | tr -d ' ')
  L "  verdicts: $n/52"
  [ "$n" -ge 52 ] && break
  sleep 150
done
L "applying corrections (gated)"
python3 apply_agent_rereads.py 2>&1 | grep -vE "indexing|images" | tail -1
L "revalidate + regenerate everything"
python3 compute_severity.py 2>&1 | head -1
python3 export_first_cut.py 2>&1 | grep quality | tail -1
python3 build_best_dataset.py 2>&1 | grep -vE "indexing|images" | tail -2
python3 extract_pyramids.py 2>&1 | tail -1
python3 validate_pyramids.py 2>&1 | grep verdicts | head -1
python3 build_comparable.py 2>&1 | tail -1
python3 build_single_age.py 2>&1 | grep -vE "indexing|images" | tail -1
python3 build_assembled_view.py 2>&1 | tail -1
python3 update_provenance_safe.py 2>&1 | head -2
python3 build_provenance_full.py 2>&1 | grep -vE "indexing|images" | tail -3
L "rebuild site (corrected tables re-render)"
rm -f census_site/tables/*.html
python3 build_static_site.py 2>&1 | tail -3
L "deploy (accumulating retries)"
n=$(find census_site -type f | wc -l | tr -d ' ')
L "files: $n"
if [ "$n" -lt 20000 ]; then
  for try in $(seq 1 12); do
    OUT=$(npx wrangler pages deploy census_site --project-name=heb91r-5a20dde1 --branch=main --commit-dirty=true 2>&1 | tail -4)
    echo "$OUT"
    echo "$OUT" | grep -q "Deployment complete" && { L "DEPLOYED (attempt $try)"; break; }
    sleep 120
  done
fi
L "=== OVERNIGHT FINALIZATION COMPLETE ==="
