#!/usr/bin/env python3
"""Find the page-number reads that are PROBLEMATIC FOR ORDERING after merging all
sources (human > first-pass Arabic > Roman-aware). Writes a queue of just those.

A scan is flagged when, within an assembled table it belongs to:
  - dup     : its page number is shared by another physical page in the same table
              (one of them is almost certainly a misread) -> highest priority
  - null    : no page number resolved for it at all
  - outlier : its page number sits far outside the rest of the table's run
              (|p - median| > 40 and the table has >=3 numbered pages)

Output: export_final/sendout_final/sendout_v1/pageno_parts/problems.json
        [{imghash, reasons:[...], page, asm_id, table_pages:[...]}]  (dedup by imghash)
Usage : python3 pageno_find_problems.py
"""
import json, os, re
from collections import defaultdict
from statistics import median

ROOT = os.path.dirname(os.path.abspath(__file__))
V1 = os.path.join(ROOT, "export_final/sendout_final/sendout_v1")
PARTS = os.path.join(V1, "pageno_parts")
HASH_RE = re.compile(r"([0-9a-f]{40})_oneshot")


def imghash(tid):
    m = HASH_RE.search(tid or "")
    return m.group(1) if m else None


def merged_pageno():
    """imghash -> (int page number, confidence) with precedence
    human > first-pass top-Arabic > dual Arabic-volume > Roman section."""
    pn, conf = {}, {}
    rp = os.path.join(PARTS, "roman_reads.jsonl")
    if os.path.exists(rp):
        for line in open(rp):
            line = line.strip()
            if line:
                r = json.loads(line)
                if r.get("page_number_int") is not None:
                    pn[r["imghash"]] = int(r["page_number_int"])
                    conf[r["imghash"]] = (r.get("confidence") or "").lower()
    # dual-number reads override the Roman-only reads with the Arabic volume page
    dp = os.path.join(PARTS, "dualnum_reads.jsonl")
    if os.path.exists(dp):
        for line in open(dp):
            line = line.strip()
            if line:
                r = json.loads(line)
                vp = r.get("volume_page")
                if vp is None:
                    vp = r.get("section_page")
                if vp is not None:
                    pn[r["imghash"]] = int(vp)
                    conf[r["imghash"]] = (r.get("confidence") or "").lower()
    for line in open(os.path.join(PARTS, "gemini_reads.jsonl")):
        line = line.strip()
        if line:
            r = json.loads(line)
            if r.get("page_number") is not None:
                pn[r["imghash"]] = int(r["page_number"])
                conf[r["imghash"]] = (r.get("confidence") or "").lower()
    hp = os.path.join(PARTS, "human_pageno_decisions.json")
    if os.path.exists(hp):
        for h, d in json.load(open(hp)).items():
            if d.get("page_number") is None:
                pn.pop(h, None); conf.pop(h, None)
            else:
                pn[h] = int(d["page_number"]); conf[h] = "human"
    return pn, conf


def main():
    members = json.load(open(os.path.join(V1, "assembled_members.json")))
    # which scans we actually have in the worklist (resolvable)
    worklist = json.load(open(os.path.join(PARTS, "worklist.json")))
    known_hashes = {w["imghash"] for w in worklist}
    pn, conf = merged_pageno()

    problems = {}  # imghash -> {reasons:set, page, asm_ids:set, table_pages}

    def flag(h, reason, asm, pages):
        rec = problems.setdefault(h, {"reasons": set(), "page": pn.get(h), "asm_ids": set(), "table_pages": pages})
        rec["reasons"].add(reason)
        rec["asm_ids"].add(asm)
        rec["table_pages"] = pages

    for asm, tids in members.items():
        # physical pages of this table (dedup by imghash), only resolvable ones
        hs = []
        for t in tids:
            h = imghash(t)
            if h and h in known_hashes and h not in hs:
                hs.append(h)
        if not hs:
            continue
        pages = sorted([pn[h] for h in hs if h in pn])
        # duplicates within table
        seen = defaultdict(list)
        for h in hs:
            if h in pn:
                seen[pn[h]].append(h)
        for val, hh in seen.items():
            if len(hh) > 1:
                for h in hh:
                    flag(h, "dup", asm, pages)
        # nulls
        for h in hs:
            if h not in pn:
                flag(h, "null", asm, pages)
        # low / medium confidence
        for h in hs:
            if h in pn and conf.get(h) in {"low", "medium"}:
                flag(h, "lowconf", asm, pages)
        # outliers
        nums = [pn[h] for h in hs if h in pn]
        if len(nums) >= 3:
            md = median(nums)
            for h in hs:
                if h in pn and abs(pn[h] - md) > 40 and len(seen[pn[h]]) == 1:
                    flag(h, "outlier", asm, pages)

    # order: dup first, then null, then lowconf, then outlier; group by table
    def rank(reasons):
        if "dup" in reasons:
            return 0
        if "null" in reasons:
            return 1
        if "lowconf" in reasons:
            return 2
        return 3

    out = []
    for h, rec in problems.items():
        out.append({
            "imghash": h, "reasons": sorted(rec["reasons"]), "page": rec["page"],
            "asm_ids": sorted(rec["asm_ids"]), "table_pages": rec["table_pages"],
        })
    out.sort(key=lambda r: (rank(r["reasons"]), r["asm_ids"][0]))

    json.dump(out, open(os.path.join(PARTS, "problems.json"), "w"), indent=1)
    from collections import Counter
    c = Counter()
    for r in out:
        for reason in r["reasons"]:
            c[reason] += 1
    n_tables_with_dups = len({a for r in out if "dup" in r["reasons"] for a in r["asm_ids"]})
    print(f"problem scans: {len(out)}")
    print(f"  by reason (a scan can have several): {dict(c)}")
    print(f"  assembled tables with a duplicate page number: {n_tables_with_dups}")
    print(f"wrote {os.path.join(PARTS, 'problems.json')}")


if __name__ == "__main__":
    main()
