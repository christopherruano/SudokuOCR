#!/usr/bin/env python3
"""Same physical page (same imghash) is sometimes OCR'd into >1 corpus dir with divergent quality —
the review app/queue can show the WORSE copy while a clean read already exists elsewhere. For each
imghash with multiple copies, propagate the BEST read's cell `data` (fewest constraint-fail rows, then
most rows) into the worse copies, preserving each copy's own source_image + metadata. Never overwrites a
human_corrected copy. Only propagates when the copies are the same table (section-name sets overlap).
Backs up overwritten copies to *_oneshot.prededupe.json.

  python3 dedupe_corpus_best.py [--apply]   # default dry-run
"""
import sys, glob, json, os, re, collections, shutil
sys.argv_b = sys.argv
APPLY = "--apply" in sys.argv
sys.argv = ["x"]
import constraint_guided_reocr as C
sys.argv = sys.argv_b

CORPUS = "export/ocr_results_referenced/digitized_tables"


def load(f):
    try:
        return json.load(open(f))
    except Exception:
        return None


def fails(d):
    try:
        return C.page_fail_rows(d)[0]
    except Exception:
        return 999


def nrows(d):
    return sum(len(s.get("rows", [])) for s in (d or {}).get("data", []))


def secnames(d):
    return {(s.get("name", "") or "").strip().lower() for s in (d or {}).get("data", [])}


def main():
    by_ih = collections.defaultdict(list)
    for f in glob.glob(os.path.join(CORPUS, "*", "*_oneshot.json")):
        m = re.search(r"([0-9a-f]{40})_oneshot", os.path.basename(f))
        if m:
            by_ih[m.group(1)].append(f)

    applied = skipped_struct = skipped_human = groups = 0
    recovered = 0
    for ih, fs in by_ih.items():
        if len(fs) < 2:
            continue
        infos = []
        for f in fs:
            d = load(f)
            if d is None:
                continue
            infos.append({"f": f, "d": d, "fails": fails(d), "n": nrows(d),
                          "human": bool((d.get("metadata", {}) or {}).get("human_corrected")),
                          "secs": secnames(d)})
        if len(infos) < 2:
            continue
        fmin = min(i["fails"] for i in infos)
        if fmin == max(i["fails"] for i in infos):
            continue                                   # all equal quality — nothing to do
        groups += 1
        best = sorted(infos, key=lambda i: (i["fails"], -i["n"]))[0]
        for t in infos:
            if t["f"] == best["f"] or t["fails"] <= best["fails"]:
                continue
            if t["human"]:
                skipped_human += 1
                continue
            if best["secs"] and t["secs"] and not (best["secs"] & t["secs"]):
                skipped_struct += 1                    # disjoint sections -> not the same table, skip
                continue
            recovered += t["fails"] - best["fails"]
            applied += 1
            if APPLY:
                shutil.copy(t["f"], t["f"].replace("_oneshot.json", "_oneshot.prededupe.json"))
                merged = dict(t["d"])
                merged["data"] = best["d"].get("data", [])
                if "acceptance" in best["d"]:
                    merged["acceptance"] = best["d"]["acceptance"]
                json.dump(merged, open(t["f"], "w"))
    print(f"divergent groups: {groups} | copies overwritten: {applied} | fail-rows recovered: {recovered}")
    print(f"  skipped (human_corrected target): {skipped_human} | skipped (different table/sections): {skipped_struct}")
    print("APPLIED (backups: *_oneshot.prededupe.json)" if APPLY else "DRY-RUN (use --apply)")


if __name__ == "__main__":
    main()
