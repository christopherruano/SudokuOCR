#!/usr/bin/env python3
"""Verification follow-up: the generic Mistral re-OCR fixed frozen columns but, for some
tables, Mistral's markdown dropped the left row-label column (or read the table in a
different/transposed orientation), so real row labels (age bands '0-5'/'5-12', caste &
occupation names, language groups) got replaced with synthetic 'Row N'. For a labelled
census table that loses row identity — worse than a single flaggable frozen column. So:
REVERT any applied fix where the ORIGINAL had real meaningful labels and the new read lost
them (restore *_oneshot.prefrozen.json). Keep every fix that preserved real labels (or whose
original was already synthetic). Dry-run by default.
  python3 revert_label_loss.py [--apply]
"""
import sys, json, glob, os, re, csv, shutil

APPLY = "--apply" in sys.argv
CORPUS = "export/ocr_results_referenced/digitized_tables"
SYNTH = re.compile(r"^(row|unknown|col|column|cell|item|line|section|subtotal|sub-total|group)\s*\d+$", re.I)
TOTAL = re.compile(r"^(total|grand total|all ages?|all religions?|none)\.?$", re.I)


def meaningful(d):
    out = []
    for s in d.get("data", []):
        for r in s.get("rows", []):
            rh = next((k for k in r if not isinstance(r.get(k), dict)), None)
            l = str(r.get(rh, "")).strip() if rh else ""
            if not l or SYNTH.match(l) or TOTAL.match(l):
                continue
            if re.search(r"[A-Za-z]", l) or re.search(r"\d+\s*[-–—]\s*\d+", l):
                out.append(l)
    return out


def main():
    rows = list(csv.DictReader(open("export/frozen_nonvii_fixes.csv")))
    revert, keep = [], 0
    for r in rows:
        ih, td = r["imghash"], r["table_dir"]
        cur = f"{CORPUS}/{td}/{ih}_oneshot.json"
        bak = f"{CORPUS}/{td}/{ih}_oneshot.prefrozen.json"
        if not (os.path.exists(cur) and os.path.exists(bak)):
            continue
        om = meaningful(json.load(open(bak)))
        nm = meaningful(json.load(open(cur)))
        if len(om) >= 4 and len(nm) <= 0.3 * len(om):
            revert.append((ih, td, r["title"][:42], len(om), len(nm)))
        else:
            keep += 1
    print(f"applied fixes: {len(rows)} | KEEP (labels ok): {keep} | REVERT (label loss): {len(revert)}")
    for ih, td, t, no, nn in revert:
        print(f"  revert {ih[:12]} old_labels={no} new={nn}  {t}")
        if APPLY:
            cur = f"{CORPUS}/{td}/{ih}_oneshot.json"
            bak = f"{CORPUS}/{td}/{ih}_oneshot.prefrozen.json"
            shutil.copy(bak, cur)            # restore original (labels intact, still frozen)
    print(f"\n{'REVERTED' if APPLY else 'DRY-RUN'} {len(revert)} copies"
          + ("" if APPLY else "  (use --apply)"))


if __name__ == "__main__":
    main()
