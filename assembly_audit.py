#!/usr/bin/env python3
"""Comprehensive health check of EVERY assembled run. Read-only. Flags the four ways an assembly can be
wrong, using the per-page titles in the master:

  MIXED_FAMILY  : the run's pages aren't all the same table (e.g. a TABLE VII run that runs forward into
                  TABLE VIII) -> over-extended across a table boundary; the minority pages should be cut.
  CONT_START    : the first page is marked 'Continued' -> the table began earlier; opening pages missing.
  CONT_END_RESUME: the last page is 'Continued' AND a same-family page reappears within a few pages after
                  a blank -> the table may continue (flag only; the next same-numbered table is often a
                  *different district*, so this needs a human, not an auto-merge).
  ORPHAN_VII    : a page the master marks Table VII that is in NO run at all.

Usage: python3 assembly_audit.py [--csv out.csv]
"""
import json, csv, os, re, sys
from collections import Counter

ROOT = os.path.dirname(os.path.abspath(__file__))
V1 = os.path.join(ROOT, "export_final/sendout_final/sendout_v1")
RUNS = os.path.join(ROOT, "export/tableVII_runs.json")
MASTER = os.path.join(V1, "page_number_status.csv")

CONT = re.compile(r"contd|continued|\(cont|concld|conclud|-cont\b", re.I)
TAB = re.compile(r"TABLE\s+(?:No\.?\s*)?((?:[A-Z]-)?[IVXLC]+|[A-Z])\b", re.I)
BLANK = re.compile(r"blank|no table|unknown|^\s*$", re.I)


def family(title):
    t = re.sub(r"[-.\s]*(cont(?:d|inued)?|concl(?:d|uded)?)\.?", " ", title or "", flags=re.I)
    m = TAB.search(t)
    return m.group(1).upper() if m else ""


def main():
    runs = json.load(open(RUNS))
    title, vii = {}, {}
    for r in csv.DictReader(open(MASTER)):
        dh, pg = r.get("dochash", ""), r.get("pdf_page", "")
        if dh and pg.lstrip("-").isdigit():
            k = (dh, int(pg))
            if k not in title:
                title[k] = r.get("title", "")
            vii[k] = vii.get(k, False) or r.get("is_tableVII") == "1"

    owned = set()
    for run in runs:
        for p in range(run["page_range"][0], run["page_range"][1] + 1):
            owned.add((run["dochash"], p))

    mixed, cstart, cend, short = [], [], [], []
    for run in runs:
        dh = run["dochash"]; s, e = run["page_range"]
        fams = Counter()
        for p in range(s, e + 1):
            t = title.get((dh, p))
            if t is None or BLANK.search(t):
                continue
            f = family(t)
            if f:
                fams[f] += 1
        dom = fams.most_common(1)[0][0] if fams else ""
        off = [(p, family(title.get((dh, p), "")), title.get((dh, p), "")[:34])
               for p in range(s, e + 1)
               if title.get((dh, p)) and not BLANK.search(title.get((dh, p), ""))
               and family(title.get((dh, p), "")) and family(title.get((dh, p), "")) != dom]
        if off and len(fams) > 1:
            mixed.append((run["saoa"], dh, [s, e], dict(fams), off))
        st = title.get((dh, s), "")
        if CONT.search(st):
            cstart.append((run["saoa"], dh, [s, e], st[:40]))
        et = title.get((dh, e), "")
        if CONT.search(et):
            # same-family page within 4 after a blank?
            for p in range(e + 1, e + 5):
                t = title.get((dh, p))
                if t and not BLANK.search(t) and family(t) == dom:
                    cend.append((run["saoa"], dh, [s, e], p, t[:34]))
                    break
        if e - s == 0:
            short.append((run["saoa"], dh, s))

    orphans = [(dh, p) for (dh, p), isv in vii.items() if isv and (dh, p) not in owned]
    orph_by_doc = Counter(dh for dh, p in orphans)

    print(f"=== ASSEMBLY AUDIT — {len(runs)} runs ===\n")
    print(f"[MIXED_FAMILY] {len(mixed)} runs span >1 table (over-extended across a boundary):")
    for sa, dh, rng, fams, off in mixed[:40]:
        print(f"   {sa} {rng} families={fams}")
        for p, f, t in off[:6]:
            print(f"        off p{p} [{f}]: {t!r}")
    print(f"\n[CONT_START] {len(cstart)} runs start on a 'Continued' page (missing opening pages):")
    for sa, dh, rng, t in cstart[:40]:
        print(f"   {sa} {rng}  start={t!r}")
    print(f"\n[CONT_END_RESUME] {len(cend)} runs end 'Continued' with a same-family page soon after (review):")
    for sa, dh, rng, p, t in cend[:40]:
        print(f"   {sa} {rng}  -> resumes p{p} {t!r}")
    print(f"\n[ORPHAN_VII] {len(orphans)} Table VII pages in NO run, across {len(orph_by_doc)} volumes:")
    for dh, n in orph_by_doc.most_common(15):
        print(f"   {dh}: {n} orphan VII pages")
    print(f"\n[SHORT] {len(short)} single-page runs")
    print(f"\nSUMMARY: mixed={len(mixed)} cont_start={len(cstart)} cont_end={len(cend)} "
          f"orphan_vii={len(orphans)} single_page={len(short)}")

    if "--csv" in sys.argv:
        out = sys.argv[sys.argv.index("--csv") + 1]
        with open(out, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["issue", "saoa", "dochash", "range", "detail"])
            for sa, dh, rng, fams, off in mixed:
                w.writerow(["MIXED_FAMILY", sa, dh, rng, f"{fams}; off={[(p,f) for p,f,_ in off]}"])
            for sa, dh, rng, t in cstart:
                w.writerow(["CONT_START", sa, dh, rng, t])
            for sa, dh, rng, p, t in cend:
                w.writerow(["CONT_END_RESUME", sa, dh, rng, f"resume p{p}: {t}"])
            for dh, n in orph_by_doc.most_common():
                w.writerow(["ORPHAN_VII", "", dh, "", f"{n} orphan pages"])
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
