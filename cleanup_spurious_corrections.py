#!/usr/bin/env python3
"""Undo the save-bug damage: pages that got marked human_corrected by a Save even though no cell actually
changed (their data is identical to the *_oneshot.preedit.json backup). For each such page, strip the
human_corrected flag and remove the spurious backup, so the page is treated as un-reviewed again (and
re-OCR/merge may still help it). Pages with a REAL change (data differs from backup) are left intact.
  python3 cleanup_spurious_corrections.py [--apply]
"""
import json, glob, os, sys

APPLY = "--apply" in sys.argv
CORPUS = "export/ocr_results_referenced/digitized_tables"


def datastr(d):
    return json.dumps(d.get("data", []), sort_keys=True)


def main():
    spurious = real = 0
    examples = []
    for bak in glob.glob(f"{CORPUS}/*/*_oneshot.preedit.json"):
        cur = bak.replace("_oneshot.preedit.json", "_oneshot.json")
        if not os.path.exists(cur):
            continue
        try:
            pd, cd = json.load(open(bak)), json.load(open(cur))
        except Exception:
            continue
        if datastr(pd) == datastr(cd):           # no actual cell change -> spurious mark
            spurious += 1
            if len(examples) < 8:
                examples.append(os.path.basename(cur)[:20])
            if APPLY:
                meta = cd.get("metadata", {}) or {}
                meta.pop("human_corrected", None)
                cd["metadata"] = meta
                json.dump(cd, open(cur, "w"))
                os.remove(bak)
        else:
            real += 1
    print(f"{'APPLIED — ' if APPLY else 'DRY RUN — '}spurious human_corrected (unchanged) pages: {spurious} "
          f"| genuine edits kept: {real}")
    if not APPLY:
        print("  sample spurious:", ", ".join(examples))
        print("  re-run with --apply to strip the flag + remove the spurious backups")
    else:
        print("  stripped the flag and removed spurious backups; genuine edits untouched")


if __name__ == "__main__":
    main()
