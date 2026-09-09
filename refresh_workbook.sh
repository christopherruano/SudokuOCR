#!/usr/bin/env bash
# Run this AFTER the flash-drive CSVs land in export_final/sendout_final/sendout_v1/.
# It folds in the true per-page PDF numbers + authoritative provenance and rebuilds
# the provenance workbook in real PDF-page order.
#
#   place:  have_pdf_pages.csv   (imghash, saoa_id, pdf_page)   <- HAVE, true pages
#           volumes.csv          (saoa_id, year, title, province_or_volume)  <- provenance
#           tableVII_inventory.csv  (optional, for the gap list)
#   then:   ./refresh_workbook.sh
set -uo pipefail
cd "$(dirname "$0")"
V1=export_final/sendout_final/sendout_v1

if [ -f "$V1/have_pdf_pages.csv" ]; then
  echo "folding have_pdf_pages.csv -> crop_pdf_pages.csv (authoritative per-page PDF pages)"
  python3 - <<'PY'
import csv
V1="export_final/sendout_final/sendout_v1"
rows=list(csv.DictReader(open(f"{V1}/have_pdf_pages.csv")))
n=0
with open(f"{V1}/crop_pdf_pages.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["imghash","pdf_page"])
    for r in rows:
        if str(r.get("pdf_page","")).lstrip("-").isdigit():
            w.writerow([r["imghash"], r["pdf_page"]]); n+=1
print(f"  wrote crop_pdf_pages.csv: {n} crops now have a true PDF page")
PY
else
  echo "(no have_pdf_pages.csv yet — rebuilding on current/manifest pages)"
fi

echo "rebuilding tracker + workbook..."
python3 build_master_tracker.py
python3 build_provenance_workbook.py
echo "Done -> open $V1/provenance_workbook.xlsx"
