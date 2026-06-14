#!/usr/bin/env bash
# After you place the cluster's crop_pdf_pages.csv (imghash,pdf_page — extracted from
# the digitization metadata.json files) into export_final/sendout_final/sendout_v1/,
# run this to give every crop its TRUE PDF page and rebuild all trackers + contact sheets.
set -uo pipefail
cd "$(dirname "$0")"
V1=export_final/sendout_final/sendout_v1
CSV="$V1/crop_pdf_pages.csv"
[ -f "$CSV" ] || { echo "Missing $CSV — sync it from the cluster first (see MISSING_PAGES_TASK / the extract command)."; exit 1; }
echo "crop_pdf_pages.csv: $(($(wc -l < "$CSV")-1)) imghash->pdf_page mappings"
python3 - <<'PY'
import csv, glob, os, re
csv.field_size_limit(10**7)
V1="export_final/sendout_final/sendout_v1"
HEX=re.compile(r"([0-9a-f]{40})")
have={r["imghash"] for r in csv.DictReader(open(f"{V1}/crop_pdf_pages.csv"))}
crops={HEX.search(os.path.basename(p)).group(1) for p in glob.glob("export/*/digitized_tables/*/*.png") if HEX.search(os.path.basename(p))}
cov=len(crops & have)
print(f"coverage: {cov}/{len(crops)} crops now have a real PDF page ({100*cov//max(1,len(crops))}%)")
PY
echo "Rebuilding trackers..."
python3 build_master_tracker.py
python3 build_tableVII_completeness.py
python3 build_volume_contactsheets.py
echo "Done — open tableVII_completeness.xlsx (now on real PDF pages)."
