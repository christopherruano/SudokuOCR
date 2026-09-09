#!/usr/bin/env python3
"""CSP cell-guesser using THREE constraints per table section:
   (H1) Males + Females = Persons         within each civil-condition group (per row)
   (H2) Unmarried + Married + Widowed = Total   per sub-column (per row)
   (V)  sum of the age-band rows = the 'All ages'/Total row    per (group, sub-column)
A single misread cell breaks the constraints it sits in. We find the one cell whose correction — a value
triangulated from those constraints — makes the ENTIRE section consistent (zero remaining failures across
all rows AND all column sums). That whole-section verification is what makes a guess trustworthy: a row
with several (possibly compensating) errors has no single-cell fix and is correctly left alone.

(Cross-table totals — page subtotals rolling into grand totals — would extend this further; deferred.)

  python3 build_cell_guesses.py                 # report over the broken queue -> export/cell_guesses.csv
  python3 build_cell_guesses.py --validate IH   # run on a table's *.preedit.json, compare to your fix
"""
import sys, re, json, csv, glob, os, copy
VALIDATE = sys.argv[sys.argv.index("--validate") + 1] if "--validate" in sys.argv else None
sys.argv = ["x"]
import census_review_app as A

UNM = re.compile(r"unmarried|single|never", re.I)
MAR = re.compile(r"married", re.I)
WID = re.compile(r"widow|divorc", re.I)
TOT = re.compile(r"total|population|all ages|grand", re.I)
PERSON = re.compile(r"person|total|both", re.I)
MALE = re.compile(r"\bmale", re.I)
FEMALE = re.compile(r"female", re.I)
ALLAGES = re.compile(r"all\s*ages|grand\s*total|^\s*total\s*$|all\s*$", re.I)
num = A._num


def pmf_keys(v):
    return (next((k for k in v if PERSON.search(k)), None),
            next((k for k in v if MALE.search(k) and not FEMALE.search(k)), None),
            next((k for k in v if FEMALE.search(k)), None))


def rowlabel(row):
    return next((str(v) for k, v in row.items() if not isinstance(v, dict)), "")


def total_row_idx(rows):
    for i, r in enumerate(rows):
        if ALLAGES.search(rowlabel(r)):
            return i
    return None


def section_failures(rows, tot_idx):
    """Total constraint failures: per-row H1+H2, plus per-column V (age bands sum to the total row)."""
    f = 0
    for r in rows:
        f += len(A.check_constraints([r]))
    if tot_idx is not None:
        T = rows[tot_idx]
        for g, v in T.items():
            if not isinstance(v, dict):
                continue
            for sk in v:
                tv = num(v.get(sk))
                if tv is None:
                    continue
                s, ok = 0, True
                for i, r in enumerate(rows):
                    if i == tot_idx:
                        continue
                    gv = r.get(g)
                    if isinstance(gv, dict):
                        x = num(gv.get(sk))
                        if x is None:
                            ok = False; break
                        s += x
                if ok and s != tv:
                    f += 1
    return f


def candidates(rows, ri, g, sk, tot_idx):
    """Triangulated candidate values for cell (ri,g,sk) from H1, H2 and V."""
    row = rows[ri]; v = row.get(g)
    if not isinstance(v, dict):
        return []
    out = []
    pk, mk, fk = pmf_keys(v)
    p, m, fv = num(v.get(pk)), num(v.get(mk)), num(v.get(fk))
    if sk == pk and None not in (m, fv):
        out.append(m + fv)
    elif sk == mk and None not in (p, fv):
        out.append(p - fv)
    elif sk == fk and None not in (p, m):
        out.append(p - m)
    groups = {gg: vv for gg, vv in row.items() if isinstance(vv, dict)}
    unm = next((x for x in groups if UNM.search(x)), None)
    mar = next((x for x in groups if MAR.search(x) and not UNM.search(x)), None)
    wid = next((x for x in groups if WID.search(x)), None)
    tt = next((x for x in groups if TOT.search(x)), None)
    if unm and mar and wid and tt:
        if g == tt:
            vals = [num(groups[x].get(sk)) for x in (unm, mar, wid)]
            if None not in vals:
                out.append(sum(vals))
        elif g in (unm, mar, wid):
            tv = num(groups[tt].get(sk))
            ov = [num(groups[x].get(sk)) for x in (unm, mar, wid) if x != g]
            if tv is not None and None not in ov:
                out.append(tv - sum(ov))
    if tot_idx is not None:
        T = rows[tot_idx]
        if isinstance(T.get(g), dict):
            if ri == tot_idx:
                s, ok = 0, True
                for i, r in enumerate(rows):
                    if i == tot_idx:
                        continue
                    x = num(r.get(g, {}).get(sk)) if isinstance(r.get(g), dict) else None
                    if x is None:
                        ok = False; break
                    s += x
                if ok:
                    out.append(s)
            else:
                tv = num(T[g].get(sk)); s, ok = 0, True
                for i, r in enumerate(rows):
                    if i in (tot_idx, ri):
                        continue
                    x = num(r.get(g, {}).get(sk)) if isinstance(r.get(g), dict) else None
                    if x is None:
                        ok = False; break
                    s += x
                if tv is not None and ok:
                    out.append(tv - s)
    return [c for c in out if c is not None]   # full list incl. duplicates -> support count


def guess_section(rows):
    """Return [(row_idx, group, subkey, current, guess)] — single cells whose correction clears the WHOLE
    section. If more than one distinct cell can do it, the section is ambiguous -> return []."""
    tot_idx = total_row_idx(rows)
    if section_failures(rows, tot_idx) == 0:
        return []
    # suspect = cells in a row that has a horizontal failure, or in the total row (column failures)
    suspect = set()
    for ri, row in enumerate(rows):
        if A.check_constraints([row]) or ri == tot_idx:
            for g, v in row.items():
                if isinstance(v, dict):
                    for sk in v:
                        suspect.add((ri, g, sk))
    fixes = []
    for (ri, g, sk) in suspect:
        cur = num(rows[ri].get(g, {}).get(sk))
        cands = candidates(rows, ri, g, sk, tot_idx)   # list incl. duplicates -> support count
        vals = set(cands)
        if len(vals) != 1:               # the constraints triangulate to DIFFERENT values -> not clean
            continue
        cand = next(iter(vals))
        if cand == cur:                  # every constraint says the current value is fine
            continue
        support = len(cands)             # how many independent constraints AGREE on the new value
        test = copy.deepcopy(rows)
        test[ri][g][sk] = cand
        if section_failures(test, tot_idx) == 0:
            fixes.append((ri, g, sk, cur, cand, support))
    cells = {(ri, g, sk) for ri, g, sk, _, _, _ in fixes}
    return fixes if len(cells) == 1 else []   # unique single-cell fix only


def validate(ih):
    pre = glob.glob(f"export/ocr_results_referenced/digitized_tables/*/{ih}_oneshot.preedit.json")
    if not pre:
        print("  no preedit backup for", ih); return
    pd = json.load(open(pre[0]))
    cur = json.load(open(pre[0].replace(".preedit", "")))
    print(f"VALIDATION on {ih}: guesser on the ORIGINAL vs YOUR saved fix")
    n = 0
    for si, s in enumerate(pd.get("data", [])):
        crows = cur.get("data", [])[si].get("rows", []) if si < len(cur.get("data", [])) else []
        for ri, g, sk, c0, guess, support in guess_section(s.get("rows", [])):
            yours = num(crows[ri].get(g, {}).get(sk)) if ri < len(crows) and isinstance(crows[ri].get(g), dict) else None
            n += 1
            tag = {3: "GOLD 3-constraint", 2: "silver 2-constraint", 1: "bronze 1-constraint"}.get(support, f"{support}-constraint")
            print(f"   row '{rowlabel(s['rows'][ri])[:14]}' {g}.{sk}: read={c0} → guess {guess} [{tag}] | "
                  + ("✅ MATCHES your fix" if yours == guess else f"❌ you put {yours}"))
    if n == 0:
        print("   (no uniquely-solvable single-cell errors — correctly left for manual fix)")


def main():
    if VALIDATE:
        validate(VALIDATE); return
    queue = list(csv.DictReader(open("export_final/sendout_final/sendout_v1/tableVII_imperfect.csv")))
    out, nsolv = [], 0
    for q in queue:
        run = A.run_of(q["table_id"]) or A.neighborhood_of(q["table_id"])
        for m in sorted(run.get("members", []), key=lambda m: m.get("page", 0)) if run else []:
            d, _, _ = A.load_table(m.get("table_id", ""))
            if not d:
                continue
            for s in d.get("data", []):
                rows = s.get("rows", [])
                for ri, g, sk, c0, guess, support in guess_section(rows):
                    nsolv += 1
                    out.append({"table": q["title"][:42], "page": m.get("page", ""),
                                "row": rowlabel(rows[ri])[:16], "cell": f"{g}.{sk}",
                                "read": c0, "likely": guess, "support": support,
                                "grade": {3: "gold", 2: "silver", 1: "bronze"}.get(support, str(support)),
                                "table_id": m.get("table_id", "")})
    out.sort(key=lambda r: -r["support"])
    with open("export/cell_guesses.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["table", "page", "row", "cell", "read", "likely", "support", "grade", "table_id"])
        w.writeheader()
        for r in out:
            w.writerow(r)
    gold = [r for r in out if r["support"] >= 3]
    silver = [r for r in out if r["support"] == 2]
    print(f"single-cell guesses: {nsolv} total | GOLD (all 3 constraints agree): {len(gold)} | "
          f"silver (2 agree): {len(silver)} | bronze (1): {nsolv - len(gold) - len(silver)} -> export/cell_guesses.csv\n")
    print("  GOLD guesses (H1 + H2 + column-sum all agree — trust these):")
    for r in gold[:18]:
        print(f"    {r['table'][:28]:28} p{r['page']} '{r['row'][:12]}' {r['cell']}: {r['read']} → {r['likely']}")


if __name__ == "__main__":
    main()
