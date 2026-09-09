#!/usr/bin/env python3
"""Deterministic 'blame the most-failing cell' repair (the reviewer's heuristic, encoded). Where a single
cell sits in BOTH a failing M+F=Persons (its group) AND a failing Unmarried+Married+Widowed=Population
(its column), and both identities AGREE on its corrected value, set it. No model — pure arithmetic, so
it's free and certain. Gated: a table's change is kept only if it REDUCES total constraint failures.
Never touches human_corrected or transposed tables. Backs up *_oneshot.preblame.json.
  python3 repair_blame.py [--apply] [--limit N]
"""
import sys, json, glob, os, shutil
_apply = "--apply" in sys.argv
_limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else 0
sys.argv = ["x"]
import census_review_app as A

ROOT = os.path.dirname(os.path.abspath(__file__))
CORPUS = os.path.join(ROOT, "export/ocr_results_referenced/digitized_tables")


def fails(d):
    return sum(len(A.check_constraints(s.get("rows", []))) for s in d.get("data", []))


def repair(d):
    """Apply blame-repairs in place; return number of cells changed."""
    n = 0
    for s in d.get("data", []):
        rows = s.get("rows", [])
        if not rows or A.is_transposed(rows):
            continue
        for row in rows:
            for g, sk, cur, sug, nf in A.blame_cell(row):
                if nf >= 2 and isinstance(row.get(g), dict):   # in BOTH failing modes = the culprit
                    row[g][sk] = int(sug); n += 1
                    break                                      # one repair per row per pass (re-run to cascade)
    return n


def main():
    files = glob.glob(os.path.join(CORPUS, "*", "*_oneshot.json"))
    if _limit:
        files = files[:_limit]
    tables = cells = 0
    examples = []
    for f in files:
        try:
            d = json.load(open(f))
        except Exception:
            continue
        if (d.get("metadata", {}) or {}).get("human_corrected"):
            continue
        of = fails(d)
        if of == 0:
            continue
        before = json.dumps(d)
        # cascade a couple passes (fixing one cell can unblock the next)
        changed = 0
        for _ in range(3):
            c = repair(d)
            changed += c
            if c == 0:
                break
        nf = fails(d)
        if changed and nf < of:
            tables += 1; cells += changed
            if len(examples) < 12:
                examples.append((os.path.basename(os.path.dirname(f))[:22], of, nf, changed))
            if _apply:
                shutil.copy(f, f.replace("_oneshot.json", "_oneshot.preblame.json"))
                json.dump(d, open(f, "w"))
        # else: not better -> discard (we only wrote on apply+better)
    print(f"{'APPLIED' if _apply else 'DRY'}: blame-repaired {cells} cell(s) across {tables} table(s)")
    for nm, o, n, c in examples:
        print(f"  {nm}: {c} cell(s), failures {o}->{n}")


if __name__ == "__main__":
    main()
