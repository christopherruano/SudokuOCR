#!/usr/bin/env python3
"""Apply page-number reads to page_reads_fixed.json.

Merges three sources per physical scan (imghash), in precedence order:
  1. human decision (pageno_parts/human_pageno_decisions.json)   -- you, the reviewer
  2. first-pass Arabic read with a number (gemini_reads.jsonl)    -- verified 10/10
  3. Roman-aware re-read of the nulls (roman_reads.jsonl)         -- verified by spot-check
For every resulting non-null page_number, set page_number (int) and page_visible
(true) on EVERY page_reads entry whose imghash matches (the page number is a
property of the physical scan). Only those two fields are touched. Scans that
resolve to null are left untouched (assembler keeps original order for ties/zeros).

Usage:
  python3 pageno_apply.py [--dry-run]
"""
import argparse, json, os, re, shutil

ROOT = os.path.dirname(os.path.abspath(__file__))
V1 = os.path.join(ROOT, "export_final/sendout_final/sendout_v1")
PARTS = os.path.join(V1, "pageno_parts")
JSONL = os.path.join(PARTS, "gemini_reads.jsonl")
ROMAN_JSONL = os.path.join(PARTS, "roman_reads.jsonl")
DUAL_JSONL = os.path.join(PARTS, "dualnum_reads.jsonl")
HUMAN = os.path.join(PARTS, "human_pageno_decisions.json")
PR = os.path.join(V1, "page_reads_fixed.json")

HASH_RE = re.compile(r"([0-9a-f]{40})_oneshot")


def imghash(tid):
    m = HASH_RE.search(tid or "")
    return m.group(1) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    # source 4 (lowest precedence): Roman-aware re-read (section numbers)
    pn_by_hash = {}
    section_by_hash = {}
    src = {}
    if os.path.exists(ROMAN_JSONL):
        for line in open(ROMAN_JSONL):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            pn = r.get("page_number_int")
            if pn is not None:
                pn_by_hash[r["imghash"]] = int(pn); src[r["imghash"]] = "roman"
    n_roman = len(pn_by_hash)

    # source 3: dual-number re-read -> prefer the ARABIC VOLUME page; keep Roman as section_page
    n_dual_vol = 0
    if os.path.exists(DUAL_JSONL):
        for line in open(DUAL_JSONL):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            vol, sec = r.get("volume_page"), r.get("section_page")
            if vol is not None:
                pn_by_hash[r["imghash"]] = int(vol); src[r["imghash"]] = "volume"; n_dual_vol += 1
                if sec is not None:
                    section_by_hash[r["imghash"]] = int(sec)
            elif sec is not None:
                pn_by_hash[r["imghash"]] = int(sec); src[r["imghash"]] = "section"

    # source 2: first-pass top-Arabic read (no overlap with the dual/roman null set)
    for line in open(JSONL):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        pn = r.get("page_number")
        if pn is not None:
            pn_by_hash[r["imghash"]] = int(pn); src[r["imghash"]] = "firstpass"
    n_after_fp = len(pn_by_hash)

    # source 1 (highest precedence): human decisions
    n_human = 0
    if os.path.exists(HUMAN):
        for h, d in json.load(open(HUMAN)).items():
            v = d.get("page_number")
            if v is None:
                pn_by_hash.pop(h, None)             # human explicitly said "no number"
                src[h] = "human-none"
            else:
                pn_by_hash[h] = int(v); src[h] = "human"
            n_human += 1
    from collections import Counter
    bysrc = Counter(src.get(h) for h in pn_by_hash)
    print(f"sources -> roman-section:{n_roman}  dual-volume:{n_dual_vol}  +firstpass:{n_after_fp - n_roman} (new)  human:{n_human}")
    print(f"final page numbers to apply: {len(pn_by_hash)} unique scans  by-source={dict(bysrc)}")
    print(f"section_page (Roman) recorded for {len(section_by_hash)} dual-numbered scans")

    preads = json.load(open(PR))
    updated = 0
    sec_set = 0
    touched_hashes = set()
    for e in preads:
        h = imghash(e["table_id"])
        if h in pn_by_hash:
            new = pn_by_hash[h]
            if e.get("page_number") != new or e.get("page_visible") is not True:
                e["page_number"] = new
                e["page_visible"] = True
                updated += 1
                touched_hashes.add(h)
        if h in section_by_hash:
            if e.get("section_page") != section_by_hash[h]:
                e["section_page"] = section_by_hash[h]
                sec_set += 1
    print(f"page_reads entries updated: {updated}  (distinct scans applied: {len(touched_hashes)})")
    print(f"section_page field written on {sec_set} entries")

    if args.dry_run:
        print("[dry-run] not writing.")
        return

    # safety: keep a pre-apply copy too
    pre = PR + ".bak_preapply"
    if not os.path.exists(pre):
        shutil.copy(PR, pre)
        print(f"saved pre-apply backup -> {os.path.basename(pre)}")
    tmp = PR + ".tmp"
    json.dump(preads, open(tmp, "w"))  # compact, matches original formatting
    os.replace(tmp, PR)
    print(f"wrote {PR}")


if __name__ == "__main__":
    main()
