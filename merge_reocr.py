#!/usr/bin/env python3
"""Merge improved re-OCR reads back into the corpus, matched by imghash.
For each export/reocr_results/.../{imghash}_oneshot.json, find the corpus table with that imghash and
OVERLAY it IF the new read is better (resolves constraint failures and/or extracts >= as many rows).
Keeps the corpus source_image (so scans resolve). Backs up the original to *_oneshot.prereocr.json.
  python3 merge_reocr.py [--dry]
"""
import glob, json, os, re, sys, shutil

ROOT = os.path.dirname(os.path.abspath(__file__))
CORPUS = os.path.join(ROOT, "export/ocr_results_referenced/digitized_tables")
DRY = "--dry" in sys.argv

_argv = sys.argv
sys.argv = ["x"]
import constraint_guided_reocr as C   # authoritative constraint-failure count (page_fail_rows)
import detect_misfires as MF          # frozen-column misfire detector
sys.argv = _argv


def nrows(d):
    return sum(len(s.get("rows", [])) for s in d.get("data", []))


def frozen(d):
    """count of 'frozen column' misfires (a non-zero value repeated down most age rows) — a hard
    bad-read signal the constraint count can miss."""
    n = 0
    for s in d.get("data", []):
        rows = s.get("rows", [])
        if not rows:
            continue
        rh = next((k for k in rows[0] if not isinstance(rows[0].get(k), dict)), None)
        n += len(MF.section_misfires(rows, rh))
    return n


def failures(d):
    """Actual count of rows that violate Table VII constraints — same metric the review app uses,
    NOT the stale stored acceptance.final_failures (which was missing/optimistic on fresh reads)."""
    try:
        return C.page_fail_rows(d)[0]
    except Exception:
        return 999


def main():
    corpus_by_ih = {}
    for f in glob.glob(os.path.join(CORPUS, "*/*_oneshot.json")):
        m = re.search(r"([0-9a-f]{40})_oneshot", os.path.basename(f))
        if m:
            corpus_by_ih[m.group(1)] = f

    improved = same = worse = nomatch = 0
    result_files = (glob.glob(os.path.join(ROOT, "export/reocr_results/**/*_oneshot.json"), recursive=True)
                    + glob.glob(os.path.join(ROOT, "export/empty_vii_reocr/**/*_oneshot.json"), recursive=True)
                    + glob.glob(os.path.join(ROOT, "export/reocr_constraint_ocr/**/*_oneshot.json"), recursive=True))
    for f in result_files:
        m = re.search(r"([0-9a-f]{40})_oneshot", os.path.basename(f))
        if not m:
            continue
        ih = m.group(1)
        cf = corpus_by_ih.get(ih)
        if not cf:
            nomatch += 1
            continue
        try:
            new = json.load(open(f))
            old = json.load(open(cf))
        except Exception:
            continue
        if (old.get("metadata", {}) or {}).get("human_corrected"):
            continue                       # never clobber a hand-corrected table
        nn, on = nrows(new), nrows(old)
        nf, of = failures(new), failures(old)
        nzf, ozf = frozen(new), frozen(old)
        # better = removes frozen-column misfires (keeping most rows), OR — at equal frozen count —
        # fixes constraint failures while keeping rows, OR adds rows with no new failures
        better = (nn > 0) and (
            (nzf < ozf and nn >= on * 0.8)
            or (nzf == ozf and ((nf < of and nn >= on * 0.9) or (nn > on and nf <= of)))
        )
        if not better:
            same += 1 if nn == on else 0
            worse += 1 if nn < on else 0
            continue
        improved += 1
        if not DRY:
            shutil.copy(cf, cf.replace("_oneshot.json", "_oneshot.prereocr.json"))
            new["source_image"] = old.get("source_image", new.get("source_image", ""))
            json.dump(new, open(cf, "w"))
    print(f"re-OCR merge: improved {improved} | no-change {same} | not-better {worse} | no-corpus-match {nomatch}")
    print("(originals backed up to *_oneshot.prereocr.json)" if not DRY else "(--dry: nothing written)")


if __name__ == "__main__":
    main()
