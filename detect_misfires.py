#!/usr/bin/env python3
"""Detect the 'frozen column' OCR misfire the reviewer flagged: in an age-distribution Table VII section,
a numeric column (Population/Unmarried/Married/Widowed × persons/males/females) repeats the SAME non-zero
value down most of the age-band rows — physically impossible for an age distribution, so it's a bad read.
Zeros are legitimate (small religions), so only NON-ZERO constants count. Reports affected tables (and,
optionally, flags them for re-OCR).

  python3 detect_misfires.py                 # report + write export/misfire_tables.csv
  python3 detect_misfires.py --flag          # also append imghashes to export/reocr_page_flags.txt
"""
import glob, json, os, re, sys, csv, collections

ROOT = os.path.dirname(os.path.abspath(__file__))
CORPUS = os.path.join(ROOT, "export/ocr_results_referenced/digitized_tables")
FLAG = "--flag" in sys.argv
TOTAL_RE = re.compile(r"total|grand|all ages|^\s*$", re.I)


def is_age_label(lbl):
    s = str(lbl)
    return bool(re.search(r"\d", s)) and not TOTAL_RE.search(s)


def section_misfires(rows, rh):
    """Return list of 'group/subkey' columns frozen at a non-zero value across most age rows."""
    age_rows = [r for r in rows if is_age_label(r.get(rh, ""))]
    if len(age_rows) < 5:
        return []
    cols = collections.defaultdict(list)
    for row in age_rows:
        for g, sub in row.items():
            if isinstance(sub, dict):
                for k, v in sub.items():
                    if isinstance(v, (int, float)):
                        cols[f"{g}/{k}"].append(v)
    out = []
    for col, vals in cols.items():
        nz = [v for v in vals if v]                      # ignore zeros (legit for sparse religions)
        if len(nz) < 5:
            continue
        val, cnt = collections.Counter(nz).most_common(1)[0]
        if val != 0 and cnt >= 0.7 * len(vals):          # same non-zero value in >=70% of age rows
            out.append((col, val, cnt, len(vals)))
    return out


def main():
    hits = []
    for f in glob.glob(os.path.join(CORPUS, "*", "*_oneshot.json")):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        if (d.get("metadata", {}) or {}).get("human_corrected"):
            continue
        ih = re.search(r"([0-9a-f]{40})_oneshot", os.path.basename(f))
        ih = ih.group(1) if ih else ""
        bad = []
        for s in d.get("data", []):
            rows = s.get("rows", [])
            if not rows:
                continue
            rh = next((k for k in rows[0] if not isinstance(rows[0].get(k), dict)), None)
            mf = section_misfires(rows, rh)
            if mf:
                bad.append((s.get("name", "")[:34], mf))
        if bad:
            ttl = ((d.get("metadata", {}) or {}).get("title", "") or "")[:42]
            ncols = sum(len(m) for _, m in bad)
            hits.append((ih, ttl, bad, ncols, f))
    hits.sort(key=lambda h: -h[3])
    seen = set()
    uniq = [h for h in hits if not (h[0] in seen or seen.add(h[0]))]
    out = os.path.join(ROOT, "export/misfire_tables.csv")
    with open(out, "w", newline="") as fo:
        w = csv.writer(fo)
        w.writerow(["imghash", "title", "n_frozen_cols", "sections"])
        for ih, ttl, bad, ncols, f in uniq:
            secs = " | ".join(f"{nm}: " + ",".join(f"{c}={v}({cnt}/{tot})" for c, v, cnt, tot in m) for nm, m in bad)
            w.writerow([ih, ttl, ncols, secs[:300]])
    print(f"frozen-column misfires: {len(uniq)} tables (-> {out})")
    print("worst 12:")
    for ih, ttl, bad, ncols, f in uniq[:12]:
        worst = bad[0]
        print(f"  {ih[:12]} [{ncols} frozen cols] {ttl}  ·  e.g. {worst[0]}: {worst[1][0][0]}={worst[1][0][1]} in {worst[1][0][2]}/{worst[1][0][3]} rows")
    if FLAG:
        flags = os.path.join(ROOT, "export/reocr_page_flags.txt")
        existing = set(open(flags).read().split()) if os.path.exists(flags) else set()
        new = [h[0] for h in uniq if h[0] and h[0] not in existing]
        with open(flags, "a") as fa:
            for ih in new:
                fa.write(ih + "\n")
        print(f"flagged {len(new)} new pages for re-OCR -> {flags}")


if __name__ == "__main__":
    main()
