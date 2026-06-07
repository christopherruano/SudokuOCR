#!/usr/bin/env python3
"""Assemble the cluster upload bundle to complete the corpus (Neil's age tables
first, then everything). Stages image lists + page targets + sbatch + an upload
script into export_final/sendout_final/sendout_v1/cluster_upload/.
"""
import os, re, glob, time, csv, shutil

csv.field_size_limit(10 ** 7)
ROOT = os.path.dirname(os.path.abspath(__file__))
V1 = os.path.join(ROOT, "export_final/sendout_final/sendout_v1")
NEW = os.path.join(ROOT, "export/ocr_results_referenced/digitized_tables")
OUT = os.path.join(V1, "cluster_upload")
DT = re.compile(r"([0-9a-f]{16}_t\d+)")
AGE = {"age_civil_condition", "age_sex_population", "age_by_religion", "proportional_age",
       "caste_class_age", "single_year_age", "age_sex_civil_condition", "infirmity_age"}


def k(s):
    m = DT.search(s or ""); return m.group(1) if m else ""


def main():
    os.makedirs(OUT, exist_ok=True)
    now = time.time()
    done = {k(p) for p in glob.glob(os.path.join(NEW, "*/*_oneshot.json"))
            if now - os.path.getmtime(p) < 2 * 86400 and k(p)}

    kp = list(csv.DictReader(open(os.path.join(V1, "reocr_keeper_manifest.csv"))))
    remain = [r for r in kp if k(r["source_image"]) not in done]
    age = [r for r in remain if r["table_family"] in AGE]

    # image lists (Neil-first; full set)
    def write_imgs(name, rows):
        imgs = list(dict.fromkeys(r["source_image"] for r in rows if r["source_image"]))
        open(os.path.join(OUT, name), "w").write("\n".join(imgs) + "\n")
        return len(imgs)
    n_age = write_imgs("reocr_neil_age.txt", age)
    n_all = write_imgs("reocr_all_remaining.txt", remain)

    # new-OCR page targets
    shutil.copy(os.path.join(V1, "newocr_neil_pages.csv"), os.path.join(OUT, "newocr_neil_pages.csv"))
    # sbatch
    for sb in ("rerun_reocr.sbatch", "ocr_never_ocrd_keepers.sbatch"):
        p = os.path.join(ROOT, "fasrc", sb)
        if os.path.exists(p):
            shutil.copy(p, os.path.join(OUT, sb))

    upload = f"""#!/usr/bin/env bash
# Upload bundle to finish the corpus. Run from repo root.
set -euo pipefail
CL=cruano@login.rc.fas.harvard.edu
DEST=/n/home11/cruano/census_toc

# 1) CRITICAL: push the fixed prompts FIRST (fixes dropped-cols / zeros)
rsync -av oneshot.py schema_discovery.py batch_production.py $CL:$DEST/

# 2) image lists + page targets + sbatch
scp {OUT}/reocr_neil_age.txt        $CL:$DEST/reocr_neil_age.txt
scp {OUT}/reocr_all_remaining.txt   $CL:$DEST/reocr_all_remaining.txt
scp {OUT}/newocr_neil_pages.csv     $CL:$DEST/newocr_neil_pages.csv
scp {OUT}/rerun_reocr.sbatch        $CL:$DEST/../SudokuOCR/fasrc/ 2>/dev/null || true

# 3) launch (on the cluster):
echo 'on cluster (Neil first):  IMAGE_LIST=$HOME/census_toc/reocr_neil_age.txt sbatch fasrc/rerun_reocr.sbatch'
echo 'then the full set:        IMAGE_LIST=$HOME/census_toc/reocr_all_remaining.txt sbatch fasrc/rerun_reocr.sbatch'
"""
    open(os.path.join(OUT, "upload.sh"), "w").write(upload)
    os.chmod(os.path.join(OUT, "upload.sh"), 0o755)

    open(os.path.join(OUT, "README.md"), "w").write(f"""# Cluster upload bundle

Re-OCR (have images, fixed prompt re-reads them):
  reocr_neil_age.txt        {n_age} Neil age tables (PRIMARY)
  reocr_all_remaining.txt   {n_all} all remaining flagged keepers (full)

New-OCR (extract skipped interior pages from PDFs):
  newocr_neil_pages.csv     139 docs / 2,688 pages (98 with resolved page spans)

Steps: ./upload.sh  (pushes prompt fixes first, then lists + sbatch).
Open: 41 of 139 new-OCR docs need a source-PDF filename -> grab the cluster's
full organized_manifest.jsonl (has digitized_source_file + resolved pages).
""")
    print(f"bundle -> {OUT}")
    print(f"  reocr_neil_age.txt: {n_age}   reocr_all_remaining.txt: {n_all}")
    print(f"  newocr_neil_pages.csv + sbatch + upload.sh + README.md")


if __name__ == "__main__":
    main()
