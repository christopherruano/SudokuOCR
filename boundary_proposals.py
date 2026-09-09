#!/usr/bin/env python3
"""For every human assembly verdict that carries a note, compute a PROPOSED true page-range using the
title_role signal (start -> continued -> concluded), absorbing interleaved right-half / district pages
(is_tableVII=0 between a start and its concluded). Reports: is the user's comment ALREADY satisfied by
the current run bounds, or does the run still need extending/splitting? Writes a review CSV; applies
nothing.

  python3 boundary_proposals.py            # report only
Output: export/boundary_proposals.csv  (saoa, note_verdict, note, cur_ranges, proposed, status, evidence)
"""
import json, os, csv, re
from collections import defaultdict

ROOT = os.path.dirname(os.path.abspath(__file__))
V1 = os.path.join(ROOT, "export_final/sendout_final/sendout_v1")
RUNS = os.path.join(ROOT, "export/tableVII_runs.json")
VERD = os.path.join(ROOT, "export/review_verdicts/assembly.json")
PNS = os.path.join(V1, "page_number_status.csv")


def parse_key(k):
    s, rest = k.split(":[", 1)
    a, b = rest.rstrip("]").split(",")
    return s, int(a), int(b)


def load_pages():
    """dochash -> sorted list of {pg, vii, role, title}."""
    by = defaultdict(dict)
    for r in csv.DictReader(open(PNS)):
        dh = r["dochash"]
        try:
            pg = int(r["pdf_page"])
        except Exception:
            continue
        by[dh][pg] = {"pg": pg, "vii": r["is_tableVII"] == "1",
                      "role": (r["title_role"] or "").strip(), "title": (r["title"] or "").strip()}
    return {dh: dict(sorted(d.items())) for dh, d in by.items()}


def is_table_page(p):
    """A page that belongs to a table spread: VII content OR an interleaved right-half/district page
    (no title_role, blank/short title) sitting between table pages. We treat role!='' as a table page,
    and absorb '' pages too; only a NEW 'start' breaks a run forward."""
    return p["vii"] or p["role"] in ("continued", "concluded", "start")


def propose(pages, cur_s, cur_e):
    """Walk outward from the current range over the contiguous table block. Backward: extend to the
    nearest 'start' role (the table's title page). Forward: extend through 'concluded', stopping before
    the next independent 'start'. Absorb interleaved non-VII (right-half/district) pages within."""
    if not pages:
        return cur_s, cur_e, "no-page-data"
    keys = sorted(pages)
    lo, hi = keys[0], keys[-1]

    # ---- backward: find the table's start page ----
    ps = cur_s
    p = cur_s
    seen_start = pages.get(cur_s, {}).get("role") == "start"
    while p - 1 >= lo:
        prev = pages.get(p - 1)
        if prev is None:
            break
        # stop if we'd cross into a clearly different *concluded* table (a concluded then gap)
        if prev["role"] == "concluded" and pages.get(cur_s, {}).get("role") != "continued":
            break
        ps = p - 1
        p -= 1
        if prev["role"] == "start":
            seen_start = True
            break
        # don't run away across many non-table pages
        if not is_table_page(prev) and not prev["vii"]:
            # absorb at most a single divider; if two blanks in a row, stop
            pp = pages.get(p - 1)
            if pp is None or (not is_table_page(pp) and not pp["vii"]):
                break

    # ---- forward: extend through concluded / up to before next start ----
    pe = cur_e
    p = cur_e
    while p + 1 <= hi:
        nxt = pages.get(p + 1)
        if nxt is None:
            break
        if nxt["role"] == "start":
            break                      # next table begins
        if pages.get(p, {}).get("role") == "concluded":
            break                      # this table already ended
        pe = p + 1
        p += 1
        if nxt["role"] == "concluded":
            break
        if not is_table_page(nxt) and not nxt["vii"]:
            nn = pages.get(p + 1)
            if nn is None or (not is_table_page(nn) and not nn["vii"]):
                break
    return ps, pe, ("start-found" if seen_start else "no-start-header")


def roleseq(pages, s, e, pad=2):
    out = []
    for pg in range(s - pad, e + pad + 1):
        p = pages.get(pg)
        if not p:
            continue
        mark = {"start": "S", "continued": "c", "concluded": "E", "": ("v" if p["vii"] else ".")}.get(p["role"], "?")
        inrange = "[" if pg == s else ("]" if pg == e else "")
        out.append(f"{pg}{mark}{inrange}")
    return " ".join(out)


def main():
    runs = json.load(open(RUNS))
    verds = json.load(open(VERD))
    pages_by = load_pages()
    saoa2dh = {}
    bysaoa = defaultdict(list)
    for r in runs:
        bysaoa[str(r["saoa"])].append(r)
        saoa2dh[str(r["saoa"])] = r["dochash"]

    rows = []
    for k, v in verds.items():
        note = (v.get("note") or "").strip()
        if not note:
            continue
        try:
            saoa, a, b = parse_key(k)
        except Exception:
            continue
        dh = saoa2dh.get(saoa)
        pages = pages_by.get(dh, {})
        cur = bysaoa.get(saoa, [])
        overlap = [r for r in cur if not (r["page_range"][1] < a or r["page_range"][0] > b)]
        if overlap:
            cs = min(r["page_range"][0] for r in overlap)
            ce = max(r["page_range"][1] for r in overlap)
        else:
            cs, ce = a, b
        ps, pe, flag = propose(pages, cs, ce)
        if ps == cs and pe == ce:
            status = "SATISFIED"
        elif v.get("v") == "wrong" and "cut" in note.lower():
            status = "CUT"
        else:
            status = "EXTEND"
        rows.append({
            "saoa": saoa, "verdict": v.get("v"), "note_range": f"{a}-{b}",
            "cur": f"{cs}-{ce}" if overlap else "(none)", "proposed": f"{ps}-{pe}",
            "delta_before": cs - ps, "delta_after": pe - ce,
            "status": status, "flag": flag,
            "evidence": roleseq(pages, ps, pe) if pages else "",
            "note": note,
        })

    out = os.path.join(ROOT, "export/boundary_proposals.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    from collections import Counter
    c = Counter(r["status"] for r in rows)
    print(f"{len(rows)} commented runs -> {out}")
    print("status:", dict(c))
    print("EXTEND with a real delta:", sum(1 for r in rows if r["status"] == "EXTEND" and (r["delta_before"] or r["delta_after"])))
    print("\nSample EXTEND proposals (legend: S=start c=continued E=concluded v=VII-only .=non-VII; [ ]=range):")
    n = 0
    for r in rows:
        if r["status"] == "EXTEND" and (r["delta_before"] or r["delta_after"]):
            print(f"\n  {r['saoa']} cur {r['cur']} -> proposed {r['proposed']}  (+{r['delta_before']}before/+{r['delta_after']}after) [{r['flag']}]")
            print(f"    note: {r['note'][:90]}")
            print(f"    {r['evidence'][:140]}")
            n += 1
        if n >= 12:
            break


if __name__ == "__main__":
    main()
