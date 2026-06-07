# Cluster upload bundle

Re-OCR (have images, fixed prompt re-reads them):
  reocr_neil_age.txt        619 Neil age tables (PRIMARY)
  reocr_all_remaining.txt   1141 all remaining flagged keepers (full)

New-OCR (extract skipped interior pages from PDFs):
  newocr_neil_pages.csv     139 docs / 2,688 pages (98 with resolved page spans)

Steps: ./upload.sh  (pushes prompt fixes first, then lists + sbatch).
Open: 41 of 139 new-OCR docs need a source-PDF filename -> grab the cluster's
full organized_manifest.jsonl (has digitized_source_file + resolved pages).
