#!/bin/bash
# Re-run all tables with constraint failures using gemini-3.1-pro-preview
cd /Users/chrisruano/Research/HEB_91r
export GEMINI_MODEL=gemini-3.1-pro-preview

echo "=== Re-running failed tables with gemini-3.1-pro-preview ==="
date

# Extract source image paths from failed results and re-run
python3 -c "
import json
from pathlib import Path

results_dir = Path('results')
failed_images = []
for f in sorted(results_dir.glob('*_oneshot.json')):
    with open(f) as fp:
        data = json.load(fp)
    c = data.get('constraints', {})
    if not c.get('all_passed', False):
        src = data.get('source_image', '')
        if src:
            failed_images.append((f.name, src))

for name, src in failed_images:
    print(f'{name}|{src}')
" | while IFS='|' read -r result_name source_image; do
    echo ""
    echo "--- Re-running: $result_name ---"
    echo "  Source: $source_image"

    # Delete old result files
    stem="${result_name%_oneshot.json}"
    rm -f "results/${stem}_oneshot.json" "results/${stem}_oneshot.xlsx" "results/${stem}_oneshot.csv"

    # Re-run
    python3 oneshot.py "$source_image"

    # Check result
    if [ -f "results/${stem}_oneshot.json" ]; then
        passed=$(python3 -c "import json; d=json.load(open('results/${stem}_oneshot.json')); c=d.get('constraints',{}); print(f'{c.get(\"passed\",0)}/{c.get(\"total_checks\",0)} {\"PASS\" if c.get(\"all_passed\") else \"FAIL\"}')")
        echo "  Result: $passed"
    else
        echo "  FAILED: No result produced"
    fi
done

echo ""
echo "=== DONE ==="
date
