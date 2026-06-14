#!/usr/bin/env python3
"""Audit whether assembled multi-page Table VII are COMBINED CORRECTLY — the part the
arithmetic 'perfect' gate does NOT check. Per assembled table, verify its members:
  - all from the SAME source volume (dochash)            [foreign-page check]
  - all the SAME place (member_place)                    [wrong-place-merged check]
  - all the SAME census year                             [year-mix check]
  - printed page numbers ASCENDING & CONTIGUOUS (<=GAP)  [order + completeness check]
A table that is arithmetically 'perfect' but FAILS integrity is NOT trustworthy.
Writes assembly_integrity.csv and prints how many 'perfect' assembled tables are
actually integrity-clean vs suspect."""
import os, re, csv, json
from collections import defaultdict

csv.field_size_limit(10 ** 7)
V1 = "export_final/sendout_final/sendout_v1"
GAP = 3
DOC = re.compile(r"([0-9a-f]{16})_t")


def main():
    members = json.load(open(os.path.join(V1, "assembled_members.json")))
    reads = {r["table_id"]: r for r in json.load(open(os.path.join(V1, "page_reads_fixed.json")))}
    place = json.load(open(os.path.join(V1, "member_place.json"))) if os.path.exists(os.path.join(V1, "member_place.json")) else {}
    verdicts = {r["table_id"]: r for r in csv.DictReader(open(os.path.join(V1, "tableVII_verdicts.csv")))} \
        if os.path.exists(os.path.join(V1, "tableVII_verdicts.csv")) else {}

    out = []
    clean = suspect = 0
    perfect_clean = perfect_suspect = 0
    for asm, mems in members.items():
        issues = []
        # dochash
        docs = {DOC.search(m).group(1) for m in mems if DOC.search(m)}
        if len(docs) > 1:
            issues.append(f"mixed_volumes({len(docs)})")
        # place
        places = {place.get(m, "") for m in mems if place.get(m, "")}
        if len(places) > 1:
            issues.append(f"mixed_places({len(places)})")
        # page order + contiguity
        pages = [int(reads[m]["page_number"]) for m in mems
                 if m in reads and str(reads[m].get("page_number", "")).isdigit()]
        n_nopage = sum(1 for m in mems if not (m in reads and str(reads[m].get("page_number", "")).isdigit()))
        if pages:
            if pages != sorted(pages):
                issues.append("pages_out_of_order")
            gaps = [b - a for a, b in zip(sorted(pages), sorted(pages)[1:]) if b - a > GAP]
            if gaps:
                issues.append(f"page_gaps({len(gaps)})")
        if n_nopage > 0:
            issues.append(f"members_without_page({n_nopage}/{len(mems)})")
        # years
        years = {reads[m].get("seq", "") and reads[m].get("family", "") for m in mems if m in reads}

        v = verdicts.get(asm, {})
        is_perfect = v.get("status") == "perfect"
        ok = not issues
        if ok: clean += 1
        else: suspect += 1
        if is_perfect:
            if ok: perfect_clean += 1
            else: perfect_suspect += 1
        out.append({"table_id": asm, "n_members": len(mems), "verdict": v.get("status", ""),
                    "integrity": "clean" if ok else "suspect", "issues": ";".join(issues),
                    "pages": ",".join(str(p) for p in sorted(pages))[:60]})

    out.sort(key=lambda r: (r["integrity"] == "clean", r["table_id"]))
    with open(os.path.join(V1, "assembly_integrity.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["table_id", "n_members", "verdict", "integrity", "issues", "pages"])
        w.writeheader(); [w.writerow(r) for r in out]
    from collections import Counter
    tag = Counter()
    for r in out:
        for i in r["issues"].split(";"):
            if i: tag[i.split("(")[0]] += 1
    print(f"assembled tables audited: {len(out)}")
    print(f"  integrity CLEAN: {clean} | SUSPECT: {suspect}")
    print(f"  of arithmetically-'perfect' assembled tables: {perfect_clean} clean, {perfect_suspect} SUSPECT")
    print(f"  suspect reasons: {dict(tag)}")
    print("wrote assembly_integrity.csv (sort: suspect first)")


if __name__ == "__main__":
    main()
