#!/usr/bin/env python3
"""AUTO-BOUNDS v2 — offline (reads export/pdftext_cache, immune to drive drops). Fix for v1's failure:
text layers catch VII headers only sporadically, so blocks fragmented. v2 adds the user's real
continuation signal: table-BODY pages are digit-dense. Page classes:
  V  imperial-VII header (era-aware)      X  excluded header (subsidiary/appendix/other tables)
  T  table-ish body (digit-dense)         P  prose (word-dense)          B  blank/near-empty
Growth: blocks start at V; extend over T/B freely and >=1 P only if brief; stop at X or sustained prose.
  python3 vii_autobounds2.py --validate | --predict
"""
import sys, os, json, csv, re, gzip
from collections import defaultdict

ROOT = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(ROOT, "export/pdftext_cache")

IMP = re.compile(r"\bTABLE\s+(No\.?\s*)?V[Il1!][Il1!.]?\b|\bSTATE\s+TABLE\s+VII\b|\bC[-–\s]?III\b[^I]|"
                 r"Age,?\s*(Sex)?\s*(and|&)?\s*Civil\s*Cond|"
                 r"Table\s+showing\s+Population,?\s+by\s+Sex,?\s+Age", re.I)
EXCL = re.compile(r"SUBSIDIARY\s+TABLE|APPENDIX|STATEMENT\s+No|proportion\s+per|"
                  r"per\s+10,?000|per\s+mille|TABLE\s+(No\.?\s*)?(VIII|IX|X{1,3}I{0,3}|XI?V|XV I{0,3})\b|"
                  r"C[-–\s]?IV\b|C[-–\s]?V\b|OCCUPATION|BIRTH-?PLACE|LITERACY|INFIRMIT|LANGUAGE\s+TABLE|"
                  r"CASTE,?\s+TRIBE|MEANS\s+OF\s+SUBSIST", re.I)


BANNER = re.compile(r"This content downloaded.*?terms", re.S | re.I)


def classify(txt):
    txt = BANNER.sub(" ", txt)
    head = txt[:900]
    if IMP.search(head):
        return "V"
    if EXCL.search(head):
        return "X"
    toks = txt.split()
    if len(toks) < 12:
        return "B"
    digits = sum(1 for t in toks if any(ch.isdigit() for ch in t) or (len(t) <= 4 and not t.isalpha()))
    return "T" if digits / len(toks) >= 0.28 else "P"


def grow(labels):
    n = len(labels)
    blocks = []
    i = 0
    while i < n:
        if labels[i] != "V":
            i += 1; continue
        s = i
        e = i
        j = i + 1
        prose_run = 0
        while j < n:
            c = labels[j]
            if c == "X":
                break
            if c == "V":
                e = j; prose_run = 0
            elif c == "T":
                e = j; prose_run = 0
            elif c == "B":
                prose_run += 0            # blanks are free
            else:                          # P
                prose_run += 1
                if prose_run >= 2:
                    break                  # sustained prose = the table ended
            j += 1
        blocks.append((s + 1, e + 1))
        i = max(j, e + 1)
    # merge blocks separated by tiny gaps
    merged = []
    for b in blocks:
        if merged and b[0] - merged[-1][1] <= 3:
            merged[-1] = (merged[-1][0], b[1])
        else:
            merged.append(b)
    return [b for b in merged if b[1] > b[0]] or merged


def score(pred, true):
    P = set(); T = set()
    for s, e in pred: P.update(range(s, e + 1))
    for s, e in true: T.update(range(s, e + 1))
    if not T: return None
    return len(P & T) / len(T), len(P - T) / max(1, len(P))


def ranges_of(v):
    if "ranges" in v: return [tuple(int(x) for x in r.split("-")) for r in v["ranges"]]
    if "range" in v and v.get("range"): return [tuple(v["range"])]
    return []


def main():
    mode = "--validate" if "--validate" in sys.argv else "--predict"
    verd = json.load(open(os.path.join(ROOT, "export/review_verdicts/documents.json")))
    conf = {k.split(":")[1]: ranges_of(v) for k, v in verd.items()}
    cov = {r["saoa"]: r for r in csv.DictReader(open(os.path.join(ROOT, "export/document_coverage.csv")))}
    targets = [sa for sa in conf if conf[sa]] if mode == "--validate" else \
              [sa for sa in cov if sa not in conf]
    rows = []; ex = ne = pa = ba = 0; nocache = 0
    for sa in targets:
        cp = os.path.join(CACHE, sa + ".json.gz")
        if not os.path.exists(cp):
            nocache += 1; continue
        texts = json.load(gzip.open(cp, "rt"))
        labels = [classify(t) for t in texts]
        pred = grow(labels)
        nV = sum(1 for c in labels if c == "V")
        span = sum(e - s + 1 for s, e in pred)
        dens = (sum(1 for i in range(len(labels)) if labels[i] in "VT" and
                    any(a - 1 <= i <= b - 1 for a, b in pred)) / span) if span else 0
        confid = "AUTO" if (pred and dens >= 0.7 and len(pred) <= 4) else ("REVIEW" if pred else "NO_SIGNAL")
        row = dict(saoa=sa, predicted=";".join(f"{s}-{e}" for s, e in pred),
                   n_header_hits=nV, block_density=round(dens, 2), confidence=confid,
                   region=cov.get(sa, {}).get("region", ""), year=cov.get(sa, {}).get("year", ""))
        if mode == "--validate":
            r = score(pred, conf[sa])
            if r:
                c, s = r
                row.update(coverage=round(c, 2), spill=round(s, 2),
                           true=";".join(f"{a}-{b}" for a, b in conf[sa]))
                if c >= 0.95 and s <= 0.1: ex += 1; row["grade"] = "EXACT"
                elif c >= 0.85 and s <= 0.25: ne += 1; row["grade"] = "NEAR"
                elif c >= 0.6: pa += 1; row["grade"] = "PARTIAL"
                else: ba += 1; row["grade"] = "BAD"
        rows.append(row)
    out = os.path.join(ROOT, f"export/vii_autobounds2_{'validation' if mode=='--validate' else 'predictions'}.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["saoa"]); w.writeheader(); w.writerows(rows)
    print(f"{mode}: {len(rows)} docs (no cache: {nocache}) -> {out}")
    if mode == "--validate":
        print(f"  EXACT {ex} · NEAR {ne} · PARTIAL {pa} · BAD {ba}")
        from collections import Counter
        g = Counter((r.get("grade"), r["confidence"]) for r in rows if r.get("grade"))
        auto_ok = sum(v for (gr, cf), v in g.items() if cf == "AUTO" and gr in ("EXACT", "NEAR"))
        auto_all = sum(v for (gr, cf), v in g.items() if cf == "AUTO")
        print(f"  AUTO-tier precision: {auto_ok}/{auto_all} EXACT+NEAR" if auto_all else "  (no AUTO rows)")
    else:
        from collections import Counter
        print("  confidence:", dict(Counter(r["confidence"] for r in rows)))


if __name__ == "__main__":
    main()
