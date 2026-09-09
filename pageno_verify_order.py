#!/usr/bin/env python3
"""Measure ordering health across the GENERAL POPULATION of assembled tables.

For each assembled table (assembled_members.json), look at its member scans
(deduped by physical page / imghash) and ask: do they have distinct, known
printed page numbers so the assembler can put them in unambiguous ascending order?

Classifies every assembled table:
  ordered       all (>1) physical pages have distinct known page numbers
  singleton     exactly one physical page (trivially ordered)
  has_dups      page numbers known but with duplicate values (ambiguous)
  partial       some pages have numbers, some don't
  none_known    no page number on any member

Run before AND after applying reads to see the lift. Also prints a random sample
of tables with their resolved member page-number sequence for spot-checking.

Usage: python3 pageno_verify_order.py [--sample 12] [--seed 7]
"""
import argparse, json, os, re, random

ROOT = os.path.dirname(os.path.abspath(__file__))
V1 = os.path.join(ROOT, "export_final/sendout_final/sendout_v1")
HASH_RE = re.compile(r"([0-9a-f]{40})_oneshot")


def imghash(tid):
    m = HASH_RE.search(tid or "")
    return m.group(1) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=12)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    members = json.load(open(os.path.join(V1, "assembled_members.json")))
    preads = json.load(open(os.path.join(V1, "page_reads_fixed.json")))
    pn = {}
    for e in preads:
        v = e.get("page_number")
        pn[e["table_id"]] = int(v) if str(v).isdigit() else 0

    classes = {"ordered": [], "singleton": [], "has_dups": [], "partial": [], "none_known": []}
    detail = {}
    for asm, tids in members.items():
        # dedupe to physical pages; a page's number is the max known across its member rows
        bypage = {}
        for t in tids:
            h = imghash(t)
            if not h:
                continue
            bypage[h] = max(bypage.get(h, 0), pn.get(t, 0))
        nums = list(bypage.values())
        n = len(nums)
        if n == 0:
            continue
        known = [x for x in nums if x > 0]
        detail[asm] = sorted(nums)
        if n == 1:
            classes["singleton"].append(asm)
        elif len(known) == 0:
            classes["none_known"].append(asm)
        elif len(known) < n:
            classes["partial"].append(asm)
        elif len(set(known)) < len(known):
            classes["has_dups"].append(asm)
        else:
            classes["ordered"].append(asm)

    total = sum(len(v) for v in classes.values())
    print(f"=== ordering health over {total} assembled tables ===")
    for k in ["ordered", "singleton", "has_dups", "partial", "none_known"]:
        c = len(classes[k])
        print(f"  {k:12s} {c:5d}  ({100*c/total:5.1f}%)")
    multi = total - len(classes["singleton"])
    good_multi = len(classes["ordered"])
    print(f"\n  multi-page tables: {multi}  | fully-ordered multi-page: {good_multi} "
          f"({100*good_multi/multi:.1f}% of multi-page)" if multi else "")

    rng = random.Random(args.seed)
    multi_ids = [a for a in members if a not in set(classes["singleton"]) and a in detail]
    rng.shuffle(multi_ids)
    print(f"\n=== random sample of {min(args.sample, len(multi_ids))} multi-page tables (page-number sequence) ===")
    for a in multi_ids[: args.sample]:
        cls = next(k for k in classes if a in classes[k])
        print(f"  [{cls:10s}] {a}  pages={detail[a]}")


if __name__ == "__main__":
    main()
