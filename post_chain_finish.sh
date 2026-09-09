#!/bin/bash
cd /Users/chrisruano/Research/SudokuOCR
# wait for the closing chain to end
while pgrep -f closing_chain.sh >/dev/null 2>&1; do sleep 300; done
echo "[$(date +%H:%M)] closing chain done — refreshing best-dataset on fresh data"
python3 build_best_dataset.py 2>&1 | tail -2
python3 - <<'PY'
# rebuild ONLY the homepage so the button reflects the fresh zip (full site already rebuilt by the chain)
import importlib, sys; sys.argv=["x"]
import build_static_site as B
# cheap: run main homepage section via full build? too heavy. index.html was built by chain WITH button
# (builder patched before its step 7). Just confirm button present:
h=open("census_site/index.html").read()
print("button in homepage:", "tableVII_best.zip" in h)
PY
echo "[$(date +%H:%M)] redeploying (delta — most files cached)"
for try in $(seq 1 10); do
  OUT=$(npx wrangler pages deploy census_site --project-name=heb91r-5a20dde1 --branch=main --commit-dirty=true 2>&1 | tail -4)
  echo "$OUT"
  echo "$OUT" | grep -q "Deployment complete" && { echo "FINAL DEPLOY OK (attempt $try)"; break; }
  sleep 90
done
