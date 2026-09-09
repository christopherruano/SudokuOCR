#!/usr/bin/env python3
"""PILOT: stitch left/right half-page Table VII spreads. A LEFT half has the age labels + Population +
Unmarried columns; the facing RIGHT half has Married + Widowed (often with blank labels). They were
scanned as two pages, so neither alone passes U+M+W=Population. This merges each pair row-by-row into a
full table and validates with the real constraints. DRY by default (reports; writes nothing).
  python3 stitch_pilot.py [--apply] [--limit N]
"""
import sys, os, glob, json, re, csv
_apply = "--apply" in sys.argv
_limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else 0
sys.argv = ["x"]
import census_review_app as A

CORPUS = "export/ocr_results_referenced/digitized_tables"
PAIRS = "export/stitch_pairs.json"


def path_for(ih):
    g = glob.glob(os.path.join(CORPUS, "*", f"{ih}_oneshot.json"))
    return g[0] if g else None


def rh_of(rows):
    return next((k for k in rows[0] if not isinstance(rows[0].get(k), dict)), None) if rows else None


def merge(left, right):
    """Merge right's dict-groups into left's rows, by section index then row index. Returns merged data or None."""
    ld, rd = left.get("data", []), right.get("data", [])
    if len(ld) != len(rd):
        return None
    out = []
    for ls, rs in zip(ld, rd):
        lr, rr = ls.get("rows", []), rs.get("rows", [])
        if len(lr) != len(rr) or not lr:
            return None
        rh = rh_of(lr)
        merged_rows = []
        for a, b in zip(lr, rr):
            row = dict(a)                                  # left: label + Population + Unmarried
            for g, v in b.items():                         # right: add Married + Widowed groups
                if isinstance(v, dict):
                    row[g] = v
            merged_rows.append(row)
        out.append({"name": ls.get("name", ""), "rows": merged_rows})
    return out


def fails(data):
    """REAL stitched-table gate: normalized-name U+M+W=P per sex per row. Returns violation count,
    or a huge number when NOTHING was checkable (un-checkable merges must never pass)."""
    import re as _re
    def norm(g): return _re.sub(r"[^a-z]", "", g.lower())
    def cls(g):
        n = norm(g)
        if "unmarried" in n or n == "single": return "U"
        if "widow" in n or "divorc" in n: return "W"
        if "married" in n: return "M"
        if "population" in n or n == "total": return "P"
        return None
    def num(v, k):
        try: return int(str(v.get(k, 0)).replace(",", ""))
        except Exception: return None
    bad = checked = 0
    for s in data:
        for r in s.get("rows", []):
            parts = {"U": None, "M": None, "W": None, "P": None}
            for g, v in r.items():
                if isinstance(v, dict):
                    c = cls(g)
                    if c and parts[c] is None: parts[c] = v
            if not (parts["P"] and parts["M"]):
                continue
            for sex in ("males", "females"):
                p = num(parts["P"], sex); m = num(parts["M"], sex)
                u = num(parts["U"], sex) if parts["U"] else 0
                w = num(parts["W"], sex) if parts["W"] else 0
                if None in (p, m):
                    continue
                checked += 1
                if abs((u or 0) + (m or 0) + (w or 0) - p) > 0: bad += 1
    return bad if checked else 10 ** 6


def main():
    halves = json.load(open(PAIRS + ".halves")) if os.path.exists(PAIRS + ".halves") else None
    lefts = json.load(open("export/stitch_lefts.json"))
    rights = json.load(open("export/stitch_rights.json"))
    if _limit:
        lefts = lefts[:_limit]
    # index right halves by dochash
    by_doc = {}
    for R in rights:
        by_doc.setdefault(R["dochash"], []).append(R)
    clean = nopair = 0
    results = []
    for L in lefts:
        lp = path_for(L["ih"])
        if not lp:
            continue
        left = json.load(open(lp))
        # candidate right halves: same doc, within 2 pages — pick the merge that VALIDATES best
        best = None
        for R in by_doc.get(L["dochash"], []):
            if R["page"] - L["page"] != 1:                 # stitches are CONSECUTIVE pages, left first (user rule)
                continue
            rp = path_for(R["ih"])
            if not rp:
                continue
            merged = merge(left, json.load(open(rp)))
            if merged is None:
                continue
            fm = fails(merged)
            if best is None or fm < best[0]:
                best = (fm, R, merged, rp)
        if best is None:
            nopair += 1; continue
        fm, R, merged, rp = best
        nrows = sum(len(s["rows"]) for s in merged)
        if fm == 0:
            clean += 1
        results.append((L["dochash"][:10], L["page"], R["page"], nrows, fm))
        if _apply and fm == 0:
            import shutil
            shutil.copy(lp, lp.replace("_oneshot.json", "_oneshot.prestitch.json"))
            left["data"] = merged
            left.setdefault("metadata", {})["stitched_halfpage"] = R["ih"]
            json.dump(left, open(lp, "w"))
    print(f"=== stitch pilot ({'APPLIED' if _apply else 'DRY'}) — {len(lefts)} left halves, validation-driven pairing ===")
    print(f"  CLEAN stitch (0 constraint fails): {clean}")
    print(f"  no valid right-half partner found: {nopair}")
    print(f"  merged-but-imperfect:              {len(results) - clean}")
    print(f"\n  doc        L+R      rows  fails")
    for dh, lpg, rpg, nr, fm in sorted(results, key=lambda x: x[4])[:30]:
        tag = "  ✅" if fm == 0 else ""
        print(f"  {dh:10} p{lpg}+p{rpg:<4} {nr:4}  {fm:4}{tag}")


if __name__ == "__main__":
    main()
