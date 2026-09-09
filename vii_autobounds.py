#!/usr/bin/env python3
"""AUTO-BOUNDS: replicate the user's manual /documents decisions — find each volume's true Imperial
Table VII range(s) from the PDF itself. Three independent evidence streams, all from the embedded TEXT
LAYER (no API):
  1. per-page header classification (era-aware VII patterns vs the learned exclusion list vs neutral)
  2. seed-and-grow blocks (bridge <=GAP neutral pages, hard-stop on excluded pages)
  3. TOC parse ("Table VII ... <printed page>") + printed-folio -> PDF-page offset, as a cross-check
Confidence: AUTO when TOC and seed-grow agree (or seed-grow is dense and clean), REVIEW otherwise.
Modes:
  python3 vii_autobounds.py --validate           # run on the confirmed docs, score vs the user's ranges
  python3 vii_autobounds.py --predict            # emit predictions for unconfirmed docs
Writes export/vii_autobounds_{validation,predictions}.csv
"""
import sys, os, json, csv, re, glob
from collections import defaultdict

ROOT = os.path.dirname(os.path.abspath(__file__))
DRIVE = "/Volumes/Expansion/census_pdfs"
GAP = 5

IMP = re.compile(r"\bTABLE\s+(No\.?\s*)?VII\b|\bSTATE\s+TABLE\s+VII\b|\bC[-–\s]?III\b[^I]|"
                 r"Age(,?\s*Sex)?\s*(and|&)\s*Civil\s*Cond|"
                 r"Table\s+showing\s+Population,?\s+by\s+Sex,?\s+Age", re.I)
EXCL = re.compile(r"SUBSIDIARY\s+TABLE|APPENDIX|STATEMENT\s+No|CHAPTER\s+[IVX]|proportion\s+per|"
                  r"per\s+10,?000|per\s+mille|TABLE\s+VIII|TABLE\s+IX\b|TABLE\s+XI{0,2}\b|TABLE\s+XV|"
                  r"TABLE\s+VI\b|TABLE\s+V\b|TABLE\s+IV\b|TABLE\s+X\b|C[-–\s]?IV\b|C[-–\s]?V\b|"
                  r"OCCUPATION|BIRTHPLACE|LITERACY|INFIRMIT|LANGUAGE|CASTE,?\s+TRIBE|RELIGION\.\s*$", re.I)
TOCLINE = re.compile(r"(?:TABLE\s+(?:No\.?\s*)?VII\b|C[-–\s]?III\b|Age,?\s*Sex\s*and\s*Civil)[^\d]{0,80}?(\d{1,4})\s*$", re.I | re.M)
FOLIO = re.compile(r"^\s*(\d{1,4})\s*$", re.M)


def pdf_for(sa):
    hits = glob.glob(os.path.join(DRIVE, "*", f"*{sa}*.pdf"))
    return hits[0] if hits else None


def page_texts(path):
    import fitz
    doc = fitz.open(path)
    out = []
    for i in range(doc.page_count):
        try:
            out.append(doc.load_page(i).get_text("text") or "")
        except Exception:
            out.append("")
    doc.close()
    return out


def classify(txt):
    head = txt[:900]                                     # headers live at the top of the page
    if EXCL.search(head):
        return "X"
    if IMP.search(head):
        return "V"
    return "."


def seed_grow(labels):
    seeds = [i for i, c in enumerate(labels) if c == "V"]
    if not seeds:
        return []
    blocks = [[seeds[0], seeds[0]]]
    for p in seeds[1:]:
        gap = p - blocks[-1][1] - 1
        ok = gap <= GAP and all(labels[q] != "X" for q in range(blocks[-1][1] + 1, p))
        if ok:
            blocks[-1][1] = p
        else:
            blocks.append([p, p])
    return [(a + 1, b + 1) for a, b in blocks]           # 1-based pages


def toc_start(texts):
    """(printed_start_page, pdf_offset) from the TOC + folio calibration, or None."""
    printed = None
    for i in range(min(30, len(texts))):
        m = TOCLINE.search(texts[i])
        if m:
            printed = int(m.group(1)); break
    if printed is None:
        return None
    # calibrate offset: find pages whose text contains a small folio number; offset = pdf_idx+1 - folio
    votes = defaultdict(int)
    for i in range(len(texts)):
        for fm in FOLIO.finditer(texts[i][:200] + "\n" + texts[i][-200:]):
            f = int(fm.group(1))
            if 1 <= f <= len(texts):
                off = (i + 1) - f
                if -5 <= off <= 60:
                    votes[off] += 1
    if not votes:
        return None
    off = max(votes, key=votes.get)
    return printed + off


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
    if not os.path.isdir(DRIVE):
        sys.exit("drive not mounted")
    verd = json.load(open(os.path.join(ROOT, "export/review_verdicts/documents.json")))
    conf = {k.split(":")[1]: ranges_of(v) for k, v in verd.items()}
    cov = {r["saoa"]: r for r in csv.DictReader(open(os.path.join(ROOT, "export/document_coverage.csv")))}
    targets = [sa for sa in conf if conf[sa]] if mode == "--validate" else [sa for sa in cov if sa not in conf]
    rows = []
    ex = ne = pa = ba = 0
    for n, sa in enumerate(targets, 1):
        path = pdf_for(sa)
        if not path:
            continue
        try:
            texts = page_texts(path)
        except Exception:
            continue
        labels = [classify(t) for t in texts]
        pred = seed_grow(labels)
        tstart = toc_start(texts)
        toc_agree = bool(pred and tstart and any(abs(s - tstart) <= 3 for s, _ in pred))
        dens = sum(1 for c in labels if c == "V") / max(1, sum(e - s + 1 for s, e in pred)) if pred else 0
        confid = "AUTO" if (toc_agree or (dens >= 0.5 and pred and len(pred) <= 4)) else ("REVIEW" if pred else "NO_SIGNAL")
        row = dict(saoa=sa, predicted=";".join(f"{s}-{e}" for s, e in pred), toc_start=tstart or "",
                   toc_agree=int(toc_agree), seed_density=round(dens, 2), confidence=confid)
        if mode == "--validate":
            r = score(pred, conf[sa])
            if r:
                c, s = r
                row.update(coverage=round(c, 2), spill=round(s, 2), true=";".join(f"{a}-{b}" for a, b in conf[sa]))
                if c >= 0.95 and s <= 0.1: ex += 1; row["grade"] = "EXACT"
                elif c >= 0.85 and s <= 0.25: ne += 1; row["grade"] = "NEAR"
                elif c >= 0.6: pa += 1; row["grade"] = "PARTIAL"
                else: ba += 1; row["grade"] = "BAD"
        rows.append(row)
        if n % 20 == 0:
            print(f"  …{n}/{len(targets)}")
    out = os.path.join(ROOT, f"export/vii_autobounds_{'validation' if mode=='--validate' else 'predictions'}.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["saoa"]); w.writeheader(); w.writerows(rows)
    print(f"\n{mode} -> {out} ({len(rows)} docs)")
    if mode == "--validate":
        print(f"  EXACT {ex} · NEAR {ne} · PARTIAL {pa} · BAD {ba}")
    else:
        from collections import Counter
        print("  confidence:", dict(Counter(r['confidence'] for r in rows)))


if __name__ == "__main__":
    main()
