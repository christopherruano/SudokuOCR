#!/usr/bin/env bash
# Drop on the cluster and run a few minutes into the Table VII OCR to confirm health.
echo "== 1. does the code that the sbatch runs have the template support? =="
for d in $HOME/census_toc/oneshot.py $HOME/SudokuOCR/oneshot.py; do
  [ -f "$d" ] && { grep -q _load_schema_templates "$d" && echo "  UPDATED: $d" || echo "  OLD:     $d"; }
done
echo "  (the sbatch's working dir must use the UPDATED one, else no templates/prime/fix)"

echo "== 2. is the sibling-template injection firing? =="
grep -rm5 "Using sibling schema template" fasrc/logs/slurm_reocr_*.out 2>/dev/null | head -5 \
  || echo "  (none yet — wait for first results, or templates not loading)"

echo "== 3. do fresh outputs have all 4 civil-condition columns? =="
latest=$(ls -t $HOME/census_toc/ocr_results/digitized_tables/*/*_oneshot.json 2>/dev/null | head -1)
[ -n "$latest" ] && python3 - "$latest" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
cols=set()
def walk(o):
    if isinstance(o,dict):
        for k,v in o.items():
            if 'group' in k.lower() or 'column' in k.lower(): cols.add(str(v)[:40])
            walk(v)
    elif isinstance(o,list):
        [walk(x) for x in o]
walk(d.get('data',d))
print("  newest:",sys.argv[1].split('/')[-2][:40]); print("  column-ish:",list(cols)[:8])
PY

echo "== 4. API key exhaustion / errors? =="
grep -rhoE "RESOURCE_EXHAUSTED|429|quota" fasrc/logs/slurm_*.err 2>/dev/null | sort | uniq -c | head \
  || echo "  none seen"
