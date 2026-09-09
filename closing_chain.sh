#!/bin/bash
# THE CLOSING CHAIN — runs after the user finished all document verification (222 verdicts).
# Drive-dependent stages first; every stage logs; failures don't kill later independent stages.
cd /Users/chrisruano/Research/SudokuOCR
L(){ echo "[$(date +%H:%M)] $*"; }

L "[1/8] pre-cache all referenced + assembly-member scans from drive"
python3 cache_missing_scans.py 2>&1 | tail -2

L "[2/8] wait for the OCR watcher to clear the last confirmed pages (max ~40min)"
for i in $(seq 1 16); do
  n=$(python3 - <<'PY' 2>/dev/null | tail -1
import sys, json, csv; sys.argv=["x"]
import boundary_apply as B
verd=json.load(open("export/review_verdicts/documents.json"))
cov={r["saoa"]: r.get("dochash","") for r in csv.DictReader(open("export/document_coverage.csv"))}
def rng(v):
    if "ranges" in v: return [tuple(int(x) for x in r.split("-")) for r in v["ranges"]]
    if "range" in v and v.get("range"): return [tuple(v["range"])]
    return []
m=0
for k,v in verd.items():
    dh=cov.get(k.split(":")[1])
    if not dh: continue
    for s,e in rng(v):
        for pg in range(s,e+1):
            if not B.member_for(dh,pg)[1]: m+=1
print(m)
PY
)
  L "  missing confirmed pages: $n"
  [ "$n" = "0" ] && break
  sleep 150
done

L "[3/8] final assembly runs from confirmed ranges"
python3 build_runs_from_confirmed.py 2>&1 | tail -1

L "[4/8] recovery: stitch re-discovery + apply (validation-gated)"
python3 stitch_pilot.py --apply 2>&1 | grep -E "CLEAN|no valid" | head -3

L "[5/8] recovery: constraint-guided re-OCR (self-validating)"
python3 constraint_guided_reocr.py --max-fail-rows 8 --apply 2>&1 | grep -E "resolved|applied" | tail -2

L "[6/8] FULL reingest (drive attached; keeper_inference reachable)"
bash reingest.sh > export/logs/reingest_close.log 2>&1 && L "reingest OK" || L "REINGEST FAILED (continuing — site builds from last good DB)"
python3 -c "import json; a=json.load(open('export/tableVII_runs.json')); b=json.load(open('export/tableVII_runs.json.preclose.bak')); print('runs preserved:', len(a)>=len(b))"

L "[7/8] rebuild everything"
rm -f census_site/tables/*.html
python3 build_static_site.py 2>&1 | tail -3
python3 export_first_cut.py 2>&1 | grep -E "quality|tables \|" | tail -2
python3 build_assembled_view.py 2>&1 | tail -1
python3 compute_severity.py 2>&1 | head -1
python3 update_provenance_safe.py 2>&1 | head -2
python3 build_provenance_full.py 2>&1 | tail -4

L "[8/8] deploy (accumulating retries)"
n=$(find census_site -type f | wc -l | tr -d " ")
L "files: $n"
if [ "$n" -lt 20000 ]; then
  for try in $(seq 1 12); do
    OUT=$(npx wrangler pages deploy census_site --project-name=heb91r-5a20dde1 --branch=main --commit-dirty=true 2>&1 | tail -4)
    echo "$OUT"
    echo "$OUT" | grep -q "Deployment complete" && { L "DEPLOYED on attempt $try"; break; }
    sleep 90
  done
else
  L "OVER FILE LIMIT — deploy skipped"
fi
L "=== CLOSING CHAIN COMPLETE ==="
