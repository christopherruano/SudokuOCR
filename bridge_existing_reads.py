#!/usr/bin/env python3
"""Backward bridging for runs that START on a 'continued' page (Phase 1, no OCR).

A run whose first page is marked 'Continued/Contd' is, by definition, missing the table's opening pages
— the table began earlier. For each such run we walk BACKWARD, absorbing contiguous same-family pages
(TABLE C-III stays with C-III), bridging up to 2 blank pages, and STOPPING the moment we hit a page that
is NOT the same family (a region divider like 'Kolar Gold Fields.', a different table, or prose). That
stop rule is what keeps per-region VII tables (Bangalore vs Mysore vs Kolar) from merging into each other.

We only look backward and only for continued-START runs — NOT forward, because a run ENDING on
'continued' is ambiguous (the next same-numbered table is usually a *different district*, not a
continuation, so forward-merging would corrupt the data).

Default = dry run (prints proposed additions with page titles). Pass --apply to write. After applying,
re-run apply_boundary_verdicts.py (rebuild members) then build_assembled_view.py.
  python3 bridge_existing_reads.py [--apply] [--reach N] [--max-blank K]
"""
import json, csv, os, re, sys

ROOT = os.path.dirname(os.path.abspath(__file__))
V1 = os.path.join(ROOT, "export_final/sendout_final/sendout_v1")
RUNS = os.path.join(ROOT, "export/tableVII_runs.json")
MASTER = os.path.join(V1, "page_number_status.csv")
APPLY = "--apply" in sys.argv
REACH = int(sys.argv[sys.argv.index("--reach") + 1]) if "--reach" in sys.argv else 8
MAXBLANK = int(sys.argv[sys.argv.index("--max-blank") + 1]) if "--max-blank" in sys.argv else 2

CONT = re.compile(r"contd|continued|\(cont|concld|-cont\b", re.I)
TAB = re.compile(r"TABLE\s+(?:No\.?\s*)?([A-Z]{0,3}-?[IVXLC0-9]+(?:-[A-Z0-9]+)?)", re.I)


def family(title):
    # strip continuation markers FIRST, else 'VII-contd' parses as family 'VII-C' and 'VII- contd' as
    # 'VII' — the same table reads as two families and the backward walk stops early.
    t = re.sub(r"[-.\s]*(cont(?:d|inued)?|concl(?:d|uded)?)\.?", " ", title or "", flags=re.I)
    m = TAB.search(t)
    return re.sub(r"[^A-Z0-9-]", "", m.group(1).upper()).strip("-") if m else ""


def main():
    runs = json.load(open(RUNS))
    page = {}
    for r in csv.DictReader(open(MASTER)):
        dh, pg = r.get("dochash", ""), r.get("pdf_page", "")
        if dh and pg.lstrip("-").isdigit():
            k = (dh, int(pg))
            t, v = page.get(k, ("", False))
            page[k] = (t or r.get("title", ""), v or r.get("is_tableVII") == "1")

    # which (dochash,page) already belongs to some run — so we can flag/merge overlaps
    owner = {}
    for i, run in enumerate(runs):
        for p in range(run["page_range"][0], run["page_range"][1] + 1):
            owner[(run["dochash"], p)] = i

    proposals = []  # (run_idx, new_start, [(page,title), ...], overlap_run_idx_or_None)
    for i, run in enumerate(runs):
        dh = run["dochash"]
        s, e = run["page_range"]
        st = page.get((dh, s), ("", False))[0]
        if not CONT.search(st):
            continue
        fam = family(st)
        if not fam:
            continue
        added = []
        ns = s
        blanks = 0
        p = s - 1
        while p >= 1 and (s - p) <= REACH:
            rec = page.get((dh, p))
            if rec is None:                       # unread / dropped page — treat as a bridgeable blank
                blanks += 1
                if blanks > MAXBLANK:
                    break
                p -= 1
                continue
            if family(rec[0]) == fam:             # same table family -> absorb
                ns = p
                added.append((p, rec[0]))
                blanks = 0
                p -= 1
                continue
            break                                  # different family / region divider / prose -> STOP
        if ns < s:
            ov = owner.get((dh, ns))               # did we walk into another run of this dochash?
            proposals.append((i, ns, list(reversed(added)), ov if ov != i else None))

    print(f"=== {len(proposals)} continued-START runs would extend backward "
          f"(same-family only, bridge<= {MAXBLANK} blanks) ===\n")
    for ri, ns, added, ov in proposals:
        r = runs[ri]
        s, e = r["page_range"]
        tag = f"  [overlaps run #{ov} {runs[ov]['saoa']} {runs[ov]['page_range']} -> MERGE]" if ov is not None else ""
        print(f"{r['saoa']}  {s}-{e}  ->  {ns}-{e}   (+{s-ns} pages){tag}")
        for p, t in added:
            print(f"      + p{p}: {t[:48]!r}")

    if not APPLY:
        print("\n(dry run — re-run with --apply to write these backward extensions)")
        return

    # apply: extend start, then DROP runs now fully contained in a larger same-dochash run (the intended
    # merges). We only drop strict containments — never merge merely-adjacent runs, which could be
    # different tables (a VII ending p50 next to a VIII starting p51 must stay separate).
    for ri, ns, added, ov in proposals:
        runs[ri]["page_range"][0] = ns
    keep, partial = [], []
    for i, r in enumerate(runs):
        dh, (s, e) = r["dochash"], r["page_range"]
        sz = e - s
        contained = any(
            j != i and runs[j]["dochash"] == dh
            and runs[j]["page_range"][0] <= s and e <= runs[j]["page_range"][1]
            and (runs[j]["page_range"][1] - runs[j]["page_range"][0]) > sz
            for j in range(len(runs)))
        if contained:
            continue
        # flag any non-contained partial overlap for eyeballing (shouldn't happen with this data)
        for j in range(len(runs)):
            if j != i and runs[j]["dochash"] == dh:
                os_, oe = runs[j]["page_range"]
                if s < oe and os_ < e and not (os_ <= s and e <= oe) and not (s <= os_ and oe <= e):
                    partial.append((r["saoa"], [s, e], runs[j]["page_range"]))
        keep.append(r)
    json.dump(keep, open(RUNS, "w"), indent=0)
    print(f"\nAPPLIED. runs: {len(runs)} -> {len(keep)} (dropped {len(runs)-len(keep)} fully-contained stubs).")
    if partial:
        print(f"  ⚠ {len(partial)} partial overlaps to eyeball:")
        for sa, a, b in partial[:10]:
            print(f"     {sa}: {a} vs {b}")
    print("  next: python3 apply_boundary_verdicts.py   (rebuild members)  then  build_assembled_view.py")


if __name__ == "__main__":
    main()
