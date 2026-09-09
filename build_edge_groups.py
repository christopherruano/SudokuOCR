#!/usr/bin/env python3
"""Edge-first, axis-aware grouping for multi-page Table VII (pilot on the flashdrive OCR set).

PRINCIPLE: only consecutive real PDF pages of the same volume can form a physical edge.
Titles label groups; they never create a physical merge by themselves. For each adjacent
pair (page N, N+1) we classify the MERGE AXIS deterministically:

  stub_inherit  : N+1 has NO row-label stub (all-numeric rows), N does -> N+1 inherits N's stub.
                  (rare, but ironclad must-merge: the page is uninterpretable alone)
  column_union  : same stub values, COMPLEMENTARY data columns -> horizontal merge (wide table split)
  row_stack     : same data columns, N+1's stub continues N's sequence -> vertical stack (continuation)
  no_merge      : different table (schema/family mismatch) -> not a physical edge

Then we emit the four operational buckets:
  assemble_now_physical : edges with a clear axis + strong signals (chain into runs)
  group_logical_only    : same family/title but no physical axis (navigation group, never a merge)
  deferred              : a neighbor page is MISSING (gap) -> wait for gap-fill/OCR
  review                : signals conflict -> needs the vision pass / human eye

The 'assemble_now_physical' and 'review' edges are written to a VISION-CANDIDATE list:
the bounded set worth a printed-page-number read to seal the edge. Everything else is free.
  python3 build_edge_groups.py [--corpus]   (--corpus also pulls existing tables via provenance; TODO)
"""
import glob, os, re, json, collections

OCR_GLOB = "export/flashdrive_ocr/**/*_oneshot.json"
OUT = "export/edge_groups"
VII = re.compile(r"table\s*vii|ages?\s*by\s*relig|age,?\s*sex|civil\s*condit|marital|conjugal", re.I)
FAMILY = re.compile(r"(ages?\s*by\s*relig|age,?\s*sex\s*and\s*civil|civil\s*condition|table\s*v?i{1,3}\b)", re.I)


def is_num(v):
    s = str(v).strip().replace(",", "").replace(".", "").replace("-", "").replace("–", "").replace("+", "")
    return s == "" or s.isdigit()


def age_band(v):
    """parse a stub value into a (lo,hi) age band if it looks like one, else None."""
    s = str(v).strip().lower()
    m = re.match(r"(\d{1,3})\s*[-–—to ]+\s*(\d{1,3})", s)
    if m:
        return (int(m.group(1)), int(m.group(2)))
    m = re.match(r"(\d{1,3})\s*\+?$", s)
    if m:
        return (int(m.group(1)), int(m.group(1)))
    return None


def load_page(f):
    try:
        d = json.load(open(f))
    except Exception:
        return None
    rows = [r for s in d.get("data", []) for r in s.get("rows", [])]
    if not rows:
        return None
    title = ((d.get("metadata") or {}).get("title") or "").strip()
    cols = []
    for r in rows:
        for k in r.keys():
            if k not in cols:
                cols.append(k)
    # stub column = the key whose values are majority non-numeric text labels
    stub_col = None
    for c in cols:
        vals = [r.get(c) for r in rows if c in r]
        nonnum = [v for v in vals if v not in (None, "") and not is_num(v)]
        if len(nonnum) >= max(2, 0.5 * len(vals)):
            stub_col = c
            break
    data_cols = [c for c in cols if c != stub_col]
    stub_vals = [str(r.get(stub_col)).strip() for r in rows] if stub_col else []
    fam = FAMILY.search(title)
    return {
        "title": title, "is_vii": bool(VII.search(title)),
        "family": (fam.group(1).lower() if fam else ""),
        "cols": set(data_cols), "stub_col": stub_col, "stub_vals": stub_vals,
        "n_rows": len(rows), "has_stub": stub_col is not None,
    }


def table_num(title):
    """extract the table identity token, e.g. VII / XII / 3, for the identity gate."""
    m = re.search(r"\btable\s*(?:no\.?\s*)?([ivxl]+|\d+)\b", title or "", re.I)
    return m.group(1).upper() if m else None


CONTD = re.compile(r"contd|continued|\(cont", re.I)


def classify(a, b):
    """classify the merge axis of edge a->b (consecutive pages). returns (axis, confidence, why)."""
    # IDENTITY GATE: different numbered tables never physically merge, however similar their schema.
    an, bn = table_num(a["title"]), table_num(b["title"])
    if an and bn and an != bn:
        return "no_merge", 0.92, f"different table numbers ({an} vs {bn})"
    contd = bool(CONTD.search(a["title"]) or CONTD.search(b["title"]))
    fam_match = a["family"] and a["family"] == b["family"]
    inter = a["cols"] & b["cols"]
    union = a["cols"] | b["cols"] or {1}
    jac = len(inter) / len(union)
    stub_match = a["has_stub"] and b["has_stub"] and set(a["stub_vals"]) == set(b["stub_vals"])

    # stub_inherit: b is all-numeric (no stub), a has one, columns broadly compatible
    if a["has_stub"] and not b["has_stub"] and jac >= 0.4:
        return "stub_inherit", 0.9, "b lacks stub; inherits a's row labels"
    # column_union: same rows (stub), complementary data columns
    if stub_match and jac < 0.5 and (a["cols"] - b["cols"]) and (b["cols"] - a["cols"]):
        return "column_union", 0.85, "same stub, complementary columns -> horizontal merge"
    # row_stack: same columns, b's age bands continue a's
    if jac >= 0.7:
        a_bands = [age_band(v) for v in a["stub_vals"] if age_band(v)]
        b_bands = [age_band(v) for v in b["stub_vals"] if age_band(v)]
        if a_bands and b_bands and max(x[1] for x in a_bands) <= min(x[0] for x in b_bands) + 1:
            return "row_stack", 0.8, "same columns, b continues a's age sequence"
        if stub_match:
            return "row_stack", 0.5, "same columns+stub (possible duplicate/continuation)"
    # related but no physical axis
    if fam_match:
        return "logical_only", 0.6, "same family, no physical-merge signal"
    return "no_merge", 0.7, "different table"


def main():
    os.makedirs(OUT, exist_ok=True)
    # load pages keyed by (saoa, pdf_page)
    pages = {}
    for f in glob.glob(OCR_GLOB, recursive=True):
        m = re.search(r"(\d+)_p(\d+)", os.path.basename(f))
        if not m:
            continue
        pg = load_page(f)
        if pg:
            pg["saoa"], pg["page"] = m.group(1), int(m.group(2))
            pages[(m.group(1), int(m.group(2)))] = pg

    by_vol = collections.defaultdict(set)
    for (sa, p) in pages:
        by_vol[sa].add(p)

    edges = []
    buckets = collections.Counter()
    vision_candidates = []
    for sa, ps in by_vol.items():
        for p in sorted(ps):
            a = pages[(sa, p)]
            if not a["is_vii"]:
                continue
            nb = (sa, p + 1)
            if nb not in pages:
                # is the next page a known gap (expected but missing)?  -> deferred
                buckets["deferred(neighbor_missing)"] += 1
                edges.append({"saoa": sa, "a": p, "b": p + 1, "axis": "deferred",
                              "why": "next page not OCR'd/rendered", "conf": 0})
                continue
            b = pages[nb]
            axis, conf, why = classify(a, b)
            e = {"saoa": sa, "a": p, "b": p + 1, "axis": axis, "conf": conf, "why": why,
                 "a_title": a["title"][:50], "b_title": b["title"][:50]}
            edges.append(e)
            if axis in ("stub_inherit", "column_union", "row_stack"):
                bucket = "assemble_now_physical" if conf >= 0.7 else "review"
                vision_candidates.append(f"{sa}_p{p}|{sa}_p{p+1}|{axis}")
            elif axis == "logical_only":
                bucket = "group_logical_only"
            else:
                bucket = "no_merge"
            buckets[bucket] += 1
            if bucket == "review":
                vision_candidates.append(f"{sa}_p{p}|{sa}_p{p+1}|{axis}")

    json.dump(edges, open(f"{OUT}/edges.json", "w"), indent=1)
    open(f"{OUT}/vision_candidates.txt", "w").write("\n".join(sorted(set(vision_candidates))))
    print(f"Table VII pages considered: {sum(1 for p in pages.values() if p['is_vii'])}")
    print(f"edges classified: {len(edges)}")
    print("\n=== BUCKETS ===")
    for k, v in buckets.most_common():
        print(f"  {v:5d}  {k}")
    print("\n=== AXIS distribution (physical-merge candidates) ===")
    ax = collections.Counter(e["axis"] for e in edges)
    for k, v in ax.most_common():
        print(f"  {v:5d}  {k}")
    print(f"\nvision-candidate edges (need page-number seal): {len(set(vision_candidates))}")
    print(f"-> {OUT}/edges.json , {OUT}/vision_candidates.txt")
    print("\n--- sample physical-merge edges ---")
    for e in edges:
        if e["axis"] in ("stub_inherit", "column_union") and e.get("conf", 0) >= 0.7:
            print(f"  {e['saoa']}_p{e['a']}->p{e['b']}  {e['axis']:13} | '{e['a_title']}' + '{e['b_title']}'")
    pass


if __name__ == "__main__":
    main()
