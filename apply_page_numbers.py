#!/usr/bin/env python3
"""Apply the page numbers read by the page-ordering window into page_reads_fixed.json,
so the assembler orders multi-page tables correctly. Merge priority per imghash:
  human_pageno_decisions.json (human-confirmed)  >  roman_reads.jsonl (Opus, roman
  numerals)  >  gemini_reads.jsonl (Gemini, all 2,492 scans).
page_reads_fixed.json is keyed by table_id, and each table_id embeds the imghash, so
we match by imghash. Backs up first (keep old copies). Read-then-write only this one
JSON; the caller then re-runs the assembly/reingest."""
import os, re, json, shutil
from datetime import datetime

V1 = "export_final/sendout_final/sendout_v1"
P = os.path.join(V1, "pageno_parts")
HEX = re.compile(r"([0-9a-f]{40})")


def main():
    # 1) merge page numbers by priority into imghash -> (page, source)
    page = {}
    # gemini (base, all scans)
    for line in open(os.path.join(P, "gemini_reads.jsonl")):
        d = json.loads(line)
        pn = d.get("page_number")
        if isinstance(pn, int) and d.get("page_visible"):
            page[d["imghash"]] = (pn, "gemini")
    # roman/opus (override)
    for line in open(os.path.join(P, "roman_reads.jsonl")):
        d = json.loads(line)
        pn = d.get("page_number_int")
        if isinstance(pn, int) and d.get("confidence") in ("high", "medium"):
            page[d["imghash"]] = (pn, "opus_roman")
    # human (highest)
    human = json.load(open(os.path.join(P, "human_pageno_decisions.json")))
    for ih, v in human.items():
        if isinstance(v.get("page_number"), int):
            page[ih] = (v["page_number"], "human")
    print(f"merged page numbers: {len(page)} scans  "
          f"(human {sum(1 for v in page.values() if v[1]=='human')}, "
          f"opus_roman {sum(1 for v in page.values() if v[1]=='opus_roman')}, "
          f"gemini {sum(1 for v in page.values() if v[1]=='gemini')})")

    # 2) back up + apply to page_reads_fixed.json (keyed by table_id; imghash embedded)
    prf = os.path.join(V1, "page_reads_fixed.json")
    bak = prf + ".bak_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    shutil.copy(prf, bak)
    reads = json.load(open(prf))
    updated = same = nohit = 0
    for r in reads:
        m = HEX.search(r.get("table_id", ""))
        if not m:
            nohit += 1; continue
        hit = page.get(m.group(1))
        if not hit:
            nohit += 1; continue
        pn = hit[0]
        if str(r.get("page_number", "")) == str(pn):
            same += 1
        else:
            r["page_number"] = pn
            updated += 1
        r["page_visible"] = True
        r["page_known"] = "1"
    json.dump(reads, open(prf, "w"))
    print(f"page_reads_fixed.json updated: {updated} changed, {same} already-correct, {nohit} no page read")
    print(f"  backup: {bak}")
    print("Next: re-assemble + reingest so multi-page tables order by these page numbers.")


if __name__ == "__main__":
    main()
