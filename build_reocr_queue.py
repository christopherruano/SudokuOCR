#!/usr/bin/env python3
"""Build the cluster re-OCR package for all tables marked needs_reocr / hold_reocr.

Outputs:
  sendout_v1/reocr_images.txt    distinct cluster image paths to re-OCR (--image-list)
  sendout_v1/reocr_manifest.csv  table_id, source_image, status, reason  (for re-ingest)
  fasrc/rerun_reocr.sbatch       Slurm array job (new decimal/sub-column prompt)
Prints the upload+run commands (incl. the REQUIRED prompt rsync).
"""
import csv, os
from collections import Counter

csv.field_size_limit(10 ** 7)
ROOT = os.path.dirname(os.path.abspath(__file__))
V1 = os.path.join(ROOT, "export_final/sendout_final/sendout_v1")


def main():
    rows = [r for r in csv.DictReader(open(os.path.join(V1, "catalog_tables.csv")))
            if r["status"] in ("needs_reocr", "hold_reocr")]
    # distinct images (preserve cluster paths), + manifest
    imgs = []
    seen = set()
    with open(os.path.join(V1, "reocr_manifest.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["table_id", "source_image", "status", "reason"])
        for r in rows:
            img = r["source_image"]
            w.writerow([r["table_id"], img, r["status"], (r.get("review_note") or "")[:120]])
            if img and img not in seen:
                seen.add(img); imgs.append(img)
    open(os.path.join(V1, "reocr_images.txt"), "w").write("\n".join(imgs) + "\n")

    n_chunks = 40
    sbatch = f"""#!/bin/bash
#SBATCH --job-name=sudokuocr_reocr
#SBATCH --partition=shared
#SBATCH --time=24:00:00
#SBATCH --mem=1G
#SBATCH --cpus-per-task=1
#SBATCH --array=0-{n_chunks-1}%4
#SBATCH --output=fasrc/logs/slurm_reocr_%A_%a.out
#SBATCH --error=fasrc/logs/slurm_reocr_%A_%a.err
#
# Re-OCR the {len(rows)} review/QA-flagged tables ({len(imgs)} distinct page images)
# with the UPDATED prompt (decimal preservation + reads real sub-columns incl.
# Unmarried/Married/Widowed, multi-table split, rotation). %4 = 4 concurrent
# workers, one per active Harvard Gemini key.
set -euo pipefail
cd ~/SudokuOCR
module load python/3.12.5-fasrc01 2>/dev/null || module load python/3.10.13-fasrc01
source ~/.envs/sudokuocr/bin/activate
set -a; source ~/SudokuOCR/.env; set +a
unset OPENAI_API_KEY; unset ANTHROPIC_API_KEY
export PYTHONUNBUFFERED=1
export GEMINI_MIN_INTERVAL=2.0
# MoE fallback ON: this is a small set of KNOWN-hard tables, worth the extra readings.
export SUDOKUOCR_MOE_FALLBACK=1

IMAGE_LIST=${{IMAGE_LIST:-$HOME/census_toc/reocr_images.txt}}
RESULTS_DIR=${{RESULTS_DIR:-$HOME/census_toc/ocr_results}}

if [ -n "${{HARVARD_GEMINI_API_KEYS:-}}" ]; then
  IFS=',' read -ra KEYS <<< "$HARVARD_GEMINI_API_KEYS"
  N_KEYS=${{#KEYS[@]}}
  export HARVARD_GEMINI_API_KEY="${{KEYS[$((SLURM_ARRAY_TASK_ID % N_KEYS))]}}"
  echo "Using key $((SLURM_ARRAY_TASK_ID % N_KEYS + 1))/${{N_KEYS}}"
fi

echo "=== reocr worker ${{SLURM_ARRAY_TASK_ID}}/${{SLURM_ARRAY_TASK_COUNT}} on $(hostname) at $(date) ==="
python3 batch_production.py \\
    --image-list "$IMAGE_LIST" \\
    --results "$RESULTS_DIR" \\
    --chunk "$SLURM_ARRAY_TASK_ID" \\
    --total-chunks "$SLURM_ARRAY_TASK_COUNT" \\
    --workers 1 --timeout 1800 --min-interval 2.0 --max-api-calls 12 --max-cost 60
echo "=== worker ${{SLURM_ARRAY_TASK_ID}} done at $(date) ==="
"""
    os.makedirs(os.path.join(ROOT, "fasrc"), exist_ok=True)
    open(os.path.join(ROOT, "fasrc/rerun_reocr.sbatch"), "w").write(sbatch)

    print(f"re-OCR tables: {len(rows)}  ({dict(Counter(r['status'] for r in rows))})")
    print(f"distinct page images: {len(imgs)}")
    print("wrote: sendout_v1/reocr_images.txt, sendout_v1/reocr_manifest.csv, fasrc/rerun_reocr.sbatch")
    print("\n" + "=" * 70)
    print("UPLOAD + RUN (paste in your terminal — CL = login.rc.fas.harvard.edu):")
    print("=" * 70)
    print("""
CL=cruano@login.rc.fas.harvard.edu

# 1) CRITICAL: push the updated prompts (decimal + sub-column) to the cluster
rsync -avz oneshot.py schema_discovery.py batch_production.py $CL:~/SudokuOCR/

# 2) upload the image list + the sbatch
scp export_final/sendout_final/sendout_v1/reocr_images.txt $CL:~/census_toc/reocr_images.txt
scp fasrc/rerun_reocr.sbatch $CL:~/SudokuOCR/fasrc/

# 3) submit
ssh $CL 'cd ~/SudokuOCR && mkdir -p fasrc/logs && sbatch fasrc/rerun_reocr.sbatch'
# watch:  ssh $CL 'squeue -u cruano'
""")


if __name__ == "__main__":
    main()
