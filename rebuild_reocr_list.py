#!/usr/bin/env python3
"""Rebuild export/reocr_image_list.txt to point at the TRUE full-res source images.

The old list pointed at census_site/img/<ih>.png, but those were compressed to .jpg during the slim
build, so every re-OCR worker found nothing and crash-looped. The real full-res original of each
imperfect table is recorded in its corpus result JSON as `source_image` (e.g. the core_money_renders
PNG). This rebuilds the list from those, skipping tables already re-OCR'd, falling back to the
compressed census_site/img/<ih>.jpg only if the original is truly gone.
  python3 rebuild_reocr_list.py
"""
import os, re, json, glob

ROOT = os.path.dirname(os.path.abspath(__file__))
CORPUS = os.path.join(ROOT, "export/ocr_results_referenced/digitized_tables")
OLD = os.path.join(ROOT, "export/reocr_image_list.txt")
RESULTS = os.path.join(ROOT, "export/reocr_results")
IMG = os.path.join(ROOT, "census_site/img")
OUT = OLD
IH = re.compile(r"([0-9a-f]{40})")


def hashes_from(path_glob, anchor="_oneshot"):
    out = set()
    for f in glob.glob(path_glob, recursive=True):
        m = re.search(r"([0-9a-f]{40})" + anchor, os.path.basename(f))
        if m:
            out.add(m.group(1))
    return out


def main():
    # 1. the imghashes we intended to re-OCR (from the old list), order-preserving
    want, seen = [], set()
    if os.path.exists(OLD):
        for line in open(OLD):
            m = IH.search(line)
            if m and m.group(1) not in seen:
                seen.add(m.group(1)); want.append(m.group(1))

    # 2. already re-OCR'd -> skip
    done = hashes_from(os.path.join(RESULTS, "**", "*_oneshot.json")) | hashes_from(os.path.join(RESULTS, "*_oneshot.json"))

    # 3. index ih -> corpus json (one walk)
    idx = {}
    for f in glob.glob(os.path.join(CORPUS, "*", "*_oneshot.json")):
        m = re.search(r"([0-9a-f]{40})_oneshot", os.path.basename(f))
        if m:
            idx.setdefault(m.group(1), f)

    out, no_src, full, compressed = [], 0, 0, 0
    for ih in want:
        if ih in done:
            continue
        src = None
        jf = idx.get(ih)
        if jf:
            try:
                si = (json.load(open(jf)).get("source_image") or "").strip()
            except Exception:
                si = ""
            if si and os.path.exists(si):
                src = si; full += 1
        if not src:
            for ext in (".jpg", ".png"):
                p = os.path.join(IMG, ih + ext)
                if os.path.exists(p):
                    src = p; compressed += 1; break
        if src:
            out.append(src)
        else:
            no_src += 1

    with open(OUT, "w") as fh:
        fh.write("\n".join(out) + ("\n" if out else ""))
    print(f"REBUILT {OUT}")
    print(f"  wanted={len(want)}  already_done={len(done & set(want))}  remaining={len(out)}")
    print(f"  sources: {full} full-res original, {compressed} compressed fallback, {no_src} NO source (skipped)")


if __name__ == "__main__":
    main()
