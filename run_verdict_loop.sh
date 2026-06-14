#!/usr/bin/env bash
# Drive every Table VII to a FINAL VERDICT (perfect / unrecoverable) by repeated
# escalating re-OCR. Run AFTER an initial base re-OCR has populated fresh reads.
# Each round: reingest (re-extract -> merge -> half-page+continuation assembly ->
# rebuild -> validate -> categorize -> verdicts) then, if any 'retrying' tables
# remain, re-OCR just those with MoE (3 Gemini + Mistral) and loop.
set -uo pipefail
cd "$(dirname "$0")"
V1=export_final/sendout_final/sendout_v1
MAX_ROUNDS=${MAX_ROUNDS:-3}

say(){ echo; echo "############ $* ($(date +%H:%M:%S)) ############"; }

for round in $(seq 0 "$MAX_ROUNDS"); do
  say "VERDICT ROUND $round — reingest + validate + verdicts"
  bash reingest.sh
  n=$(grep -c . "$V1/requery_local.txt" 2>/dev/null || echo 0)
  say "ROUND $round RESULT: $n scans still retrying"
  python3 - <<'PY'
import csv,os
from collections import Counter
V1="export_final/sendout_final/sendout_v1"
led=list(csv.DictReader(open(f"{V1}/tableVII_verdicts.csv")))
t=Counter(r["status"] for r in led)
print(f"  perfect={t['perfect']}  unrecoverable={t['unrecoverable']}  retrying={t['retrying']}  total={len(led)}")
PY
  if [ "$n" -le 1 ] || [ "$round" -ge "$MAX_ROUNDS" ]; then
    say "STOP — converged or max rounds reached"
    break
  fi
  say "RE-OCR requery set with MoE (round $round -> $((round+1)))"
  LIST="$V1/requery_local.txt" MOE=1 bash run_local_reocr.sh
done

say "FINAL VERDICT LEDGER -> $V1/tableVII_verdicts.csv"
python3 - <<'PY'
import csv
from collections import Counter
V1="export_final/sendout_final/sendout_v1"
led=list(csv.DictReader(open(f"{V1}/tableVII_verdicts.csv")))
t=Counter(r["status"] for r in led)
print(f"TOTAL Table VII: {len(led)}")
print(f"  PERFECT:        {t['perfect']} ({100*t['perfect']//max(1,len(led))}%)")
print(f"  UNRECOVERABLE:  {t['unrecoverable']}")
print(f"  still retrying: {t['retrying']}")
rs=Counter(r["reason"][:60] for r in led if r["status"]=="unrecoverable")
for r,n in rs.most_common(): print(f"     {n:3d}  {r}")
PY
