# Cluster upload bundle

Re-OCR (have images, fixed prompt re-reads them):
  reocr_neil_age.txt        687 Neil age tables (PRIMARY)
  reocr_all_remaining.txt   1255 all remaining flagged keepers (full)

New-OCR (Job B - re-render skipped interior pages, then OCR the new crops):
  newocr_queue.jsonl        716 table windows across 98 docs (~5,797 pages)
  newocr_neil_pages.csv     reference: per-doc missing-page list

Steps: ./upload.sh  (pushes prompt fixes first, then lists + queue + sbatch).
After Job B render: run the normal extract->OCR path over $HOME/census_toc/newocr_pages.
Open: 41 of 139 new-OCR docs are keeper-only (no PDF reference anywhere) -> excluded
from Job B until their source PDF is recovered.
