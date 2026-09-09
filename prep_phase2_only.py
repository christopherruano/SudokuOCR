#!/usr/bin/env python3
"""Build Phase 2's image list (Phase 1 skipped — pages already extracted).

For each queue entry, map its PDF filename to the document hash via
organized_manifest.jsonl, then find PNGs in that SPECIFIC table directory
(not every table-N across all documents). Merge with tableVII_reocr.txt.
Wipe old OCR results so Phase 2 re-processes them with the fixed prompts.

Usage:
    python3 prep_phase2_only.py --dry-run    # safe: prints counts only
    python3 prep_phase2_only.py              # actually wipes + writes list
"""
import argparse
import json
import re
import sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--dry-run", action="store_true", help="Print counts only, don't wipe or write")
args = parser.parse_args()

HOME = Path.home()
DIG = HOME / "census_toc" / "digitized_tables"
QUEUE = HOME / "census_toc" / "tableVII_newocr.jsonl"
REOCR_TXT = HOME / "census_toc" / "tableVII_reocr.txt"
MANIFEST = HOME / "census_toc" / "organized_manifest.jsonl"
OUT = HOME / "census_toc" / "tableVII_all.txt"

DOC_HASH_RE = re.compile(r"/digitized_tables/([0-9a-f]+)_t\d+_")

# Build filename -> doc_hash from the manifest
fn_to_hash = {}
with MANIFEST.open() as f:
    for line in f:
        row = json.loads(line)
        pdf_path = row.get("digitized_source_file", "")
        src_img = row.get("source_image", "")
        if not pdf_path or not src_img:
            continue
        m = DOC_HASH_RE.search(src_img)
        if not m:
            continue
        fn_to_hash[Path(pdf_path).name] = m.group(1)
print(f"Built {len(fn_to_hash)} filename -> doc_hash mappings from manifest")

# For each queue entry, get the SPECIFIC table directory
new_pngs = set()
unmatched = 0
with QUEUE.open() as f:
    for line in f:
        entry = json.loads(line)
        fn = entry.get("filename", "")
        ti = entry.get("table_index", 0)
        doc_hash = fn_to_hash.get(fn)
        if not doc_hash:
            unmatched += 1
            continue
        # Match the exact <doc_hash>_t<padded>_* directory
        for dir_match in DIG.glob(f"{doc_hash}_t{ti:04d}_*"):
            for png in dir_match.glob("*.png"):
                new_pngs.add(str(png))
print(f"Queue entries unmatched to manifest: {unmatched}")
print(f"Scoped PNGs from queue's tables: {len(new_pngs)}")

# Safety: queue is 149 entries, each table has ~10 PNGs on average -> expect ~1,500.
# If we're way above that, the glob is over-matching again — abort before wiping.
MAX_EXPECTED = 3000
if len(new_pngs) > MAX_EXPECTED:
    raise SystemExit(
        f"ABORT: {len(new_pngs)} PNGs is way more than expected (~1,500). "
        f"Refusing to wipe — fix the scoping bug first."
    )

reocr_existing = set()
if REOCR_TXT.exists():
    with REOCR_TXT.open() as f:
        for line in f:
            p = line.strip()
            if p:
                reocr_existing.add(p)

print(f"PNGs from tableVII_reocr.txt: {len(reocr_existing)}")

# Sample some paths so user can sanity-check the scope
print("\nSample of PNGs to be processed (first 5):")
for p in sorted(new_pngs)[:5]:
    print(f"  {p}")
print("\nSample of result files that would be wiped (first 5):")
sample_wipes = []
for png_path in sorted(new_pngs)[:20]:
    p = png_path.replace("/census_toc/digitized_tables/", "/census_toc/ocr_results/digitized_tables/")
    stem = p[:-4]
    for ext in (".json", ".csv", ".xlsx"):
        fp = Path(stem + "_oneshot" + ext)
        if fp.exists():
            sample_wipes.append(str(fp))
            if len(sample_wipes) >= 5:
                break
    if len(sample_wipes) >= 5:
        break
for w in sample_wipes:
    print(f"  {w}")

if args.dry_run:
    print("\n[DRY RUN] No files were wiped or written.")
    print("Re-run without --dry-run to actually do the wipe and create the list.")
    sys.exit(0)

# Wipe old OCR results so Phase 2 re-processes them
wiped = 0
for png_path in new_pngs:
    p = png_path.replace("/census_toc/digitized_tables/", "/census_toc/ocr_results/digitized_tables/")
    stem = p[:-4]
    for ext in (".json", ".csv", ".xlsx"):
        fp = Path(stem + "_oneshot" + ext)
        if fp.exists():
            fp.unlink()
            wiped += 1

ordered = list(sorted(new_pngs)) + [p for p in sorted(reocr_existing) if p not in new_pngs]
with OUT.open("w") as f:
    for p in ordered:
        f.write(p + "\n")

print(f"\nCombined (deduped): {len(ordered)}")
print(f"Order: Phase 1 set first, then re-OCR crops")
print(f"Wiped old OCR result files: {wiped}")
print(f"Wrote: {OUT}")
