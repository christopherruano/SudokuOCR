#!/usr/bin/env python3
"""Consolidate every ASSEMBLY-GROWTH / boundary comment from BOTH review surfaces — /assembly
(review_verdicts/assembly.json) and /tables (review_verdicts/tables.json) — into one worklist, and for
each tell whether the indicated extension lands on a page that's ALREADY OCR'd (just re-include via
bounds) or needs a fresh render+OCR. Writes export/growth_worklist.csv.
  python3 harvest_growth.py
"""
import json, re, csv, glob, os, collections

ROOT = os.path.dirname(os.path.abspath(__file__))
V1 = os.path.join(ROOT, "export_final/sendout_final/sendout_v1")
RUNS = json.load(open(os.path.join(ROOT, "export/tableVII_runs.json")))
GROWTH = re.compile(r"extend|continu|earlier|later|\bbefore\b|\bafter\b|grow|onitnu|more page|goes (back|further)", re.I)

# page index: dochash -> {pdf_page: (imghash, is_VII, ocrd)}
PIDX = collections.defaultdict(dict)
_ocr = set()
for f in glob.glob(os.path.join(ROOT, "export/ocr_results_referenced/digitized_tables/*/*_oneshot.json")):
    m = re.search(r"([0-9a-f]{40})_oneshot", os.path.basename(f))
    if m:
        _ocr.add(m.group(1))
for r in csv.DictReader(open(os.path.join(V1, "page_number_status.csv"))):
    if not r["pdf_page"].lstrip("-").isdigit():
        continue
    p = int(r["pdf_page"])
    cur = PIDX[r["dochash"]].get(p)
    if cur is None or r["is_tableVII"] == "1":
        PIDX[r["dochash"]][p] = (r["imghash"], r["is_tableVII"] == "1", r["imghash"] in _ocr)


def run_by_saoa_range(saoa, a, b):
    cand = [r for r in RUNS if str(r["saoa"]) == str(saoa)
            and not (r["page_range"][1] < a or r["page_range"][0] > b)]
    return cand[0] if cand else None


def run_by_tid(tid):
    dh = re.search(r"([0-9a-f]{16})_t", tid)
    dh = dh.group(1) if dh else None
    ih = re.search(r"([0-9a-f]{40})", tid)
    ih = ih.group(1) if ih else None
    for r in RUNS:
        if dh and r["dochash"] == dh:
            if ih and any(ih in m.get("table_id", "") for m in r["members"]):
                return r
    for r in RUNS:
        if dh and r["dochash"] == dh:
            return r
    return None


def coverage(run, side):
    """what's just beyond the run on `side` ('before'/'after')? is it VII and OCR'd?"""
    if not run:
        return "no-run"
    dh = run["dochash"]; s, e = run["page_range"]
    p = s - 1 if side == "before" else e + 1
    rec = PIDX.get(dh, {}).get(p)
    if rec is None:
        return f"p{p}: not in PDF (true edge)"
    ih, vii, ocrd = rec
    tag = "Table VII" if vii else "non-VII"
    return f"p{p}: {tag}, " + ("already OCR'd → just extend bounds" if ocrd else "NOT OCR'd → needs render+OCR")


def direction(note):
    n = note.lower()
    b = bool(re.search(r"earlier|before|back|begin", n))
    a = bool(re.search(r"later|after|continu|onitnu|past|end", n))
    return ("both" if a and b else "before" if b else "after" if a else "?")


def main():
    out = []
    for surface, path, keyed_by_saoa in (
        ("assembly", "export/review_verdicts/assembly.json", True),
        ("tables", "export/review_verdicts/tables.json", False)):
        p = os.path.join(ROOT, path)
        if not os.path.exists(p):
            continue
        v = json.load(open(p))
        for k, val in v.items():
            if not isinstance(val, dict):
                continue
            note = (val.get("note") or "").strip()
            if not note or not GROWTH.search(note):
                continue
            if keyed_by_saoa:
                m = re.match(r"(\w+):\[(\d+),\s*(\d+)\]", k)
                run = run_by_saoa_range(m.group(1), int(m.group(2)), int(m.group(3))) if m else None
            else:
                run = run_by_tid(k)
            d = direction(note)
            cov = " | ".join(coverage(run, s) for s in (["before", "after"] if d == "both" else [d] if d in ("before", "after") else ["before", "after"]))
            out.append({"surface": surface, "saoa": (run or {}).get("saoa", "?"),
                        "bounds": (run or {}).get("page_range", "?"), "direction": d,
                        "coverage": cov, "note": note[:80]})
    out.sort(key=lambda r: ("needs render" not in r["coverage"], r["surface"]))
    w = csv.DictWriter(open(os.path.join(ROOT, "export/growth_worklist.csv"), "w", newline=""),
                       fieldnames=["surface", "saoa", "bounds", "direction", "coverage", "note"])
    w.writeheader(); w.writerows(out)
    need = [r for r in out if "needs render" in r["coverage"]]
    print(f"{len(out)} growth comments ({sum(1 for r in out if r['surface']=='tables')} from /tables, "
          f"{sum(1 for r in out if r['surface']=='assembly')} from /assembly)")
    print(f"  -> {len(need)} reach a NOT-yet-OCR'd page (real OCR work); rest already covered.")
    print("  worklist -> export/growth_worklist.csv\n")
    for r in out:
        flag = "🆕OCR" if "needs render" in r["coverage"] else "✓have"
        print(f"  [{r['surface'][:4]} {flag}] {r['saoa']} {r['bounds']} {r['direction']:6} — {r['coverage']}")
        print(f"            «{r['note']}»")


if __name__ == "__main__":
    main()
