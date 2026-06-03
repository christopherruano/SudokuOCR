#!/usr/bin/env python3
"""
Bottom-up table reconstruction for a census year.

Instead of patching the (fragmented) v2 assembled groups, this rebuilds the
"true" tables from EVERY locally digitized page:
  1. Index all local page images for the year's source PDFs.
  2. Attach vision metadata (page_number, is_head, table_identity, markers).
  3. Within each PDF, assign every page a table identity (explicit, else the
     nearest preceding head in page-number order, else contiguity).
  4. A reconstructed table = (PDF, identity). Order its pages by page number.
  5. "truly_missing" = page numbers inside the table's span that are absent from
     the ENTIRE PDF's digitized page set (not just this table) -> candidates for
     pages never digitized. Pages present in the PDF but under another identity
     are "interleaved" (expected), NOT missing.
  6. Map each reconstructed table to the v2 group_ids its pages currently sit in
     (split/merge/add actions for the review queue).

Usage: python3 reconstruct_tables.py --year 1901
"""
import json, os, glob, re, csv, argparse
from collections import defaultdict, Counter

ROOT = os.path.dirname(os.path.abspath(__file__))
JSONL = os.path.join(ROOT, "export_final/assembled_all_read_v2/assembled_tables.jsonl")
IMGROOT = os.path.join(ROOT, "export/digitized_tables_referenced/digitized_tables")
csv.field_size_limit(10**7)

TABLE_ID = re.compile(r"(SUBSIDIARY\s+|IMPERIAL\s+)?TABLE\s*(?:NO\.?\s*)?([IVXLC]+|\d+)\b", re.I)


def norm_identity(s):
    if not s:
        return ""
    m = TABLE_ID.search(s)
    if not m:
        return ""
    pre = (m.group(1) or "").strip().upper()
    pre = "SUBTABLE" if pre.startswith("SUB") else ("IMP" if pre.startswith("IMP") else "TABLE")
    return f"{pre} {m.group(2).upper()}"


def load_vision():
    V = {}
    for f in sorted(glob.glob(os.path.join(ROOT, "export_final/assembled_all_read_v2/vbatch*/result_*.json"))):
        try:
            V.update(json.load(open(f)))
        except Exception:
            pass
    return V


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--year", required=True)
    a = ap.parse_args()
    V = load_vision()

    # prefix(PDF content hash) -> source pdf path ; hash -> group_id (v2)
    prefix2pdf = {}
    hash2group = {}
    for line in open(JSONL):
        o = json.loads(line)
        if str(o.get("year")) != str(a.year):
            continue
        for j in o.get("page_source_jsons", []):
            m = re.search(r"/digitized_tables/([0-9a-f]+)_t\d+_", j)
            if m:
                prefix2pdf[m.group(1)] = os.path.basename(o["source_file"])
            hash2group[j.split("/")[-1].replace("_oneshot.json", "")] = o["group_id"]

    # all local pages for those PDFs
    pages = []  # {hash, pdf, prefix, dir, pn, head, tid, end, sec, hdr, group}
    for prefix, pdf in prefix2pdf.items():
        for d in glob.glob(f"{IMGROOT}/{prefix}_t*"):
            for p in glob.glob(d + "/*.png"):
                h = os.path.basename(p)[:-4]
                v = V.get(h, {})
                tid = norm_identity(v.get("table_identity") or "") or norm_identity(v.get("header_text") or "")
                pages.append({
                    "hash": h, "pdf": pdf, "prefix": prefix,
                    "dir": os.path.basename(d),
                    "pn": v.get("page_number"), "head": bool(v.get("is_head")),
                    "tid": tid, "end": v.get("end_marker") or "none",
                    "sec": v.get("section") or "", "hdr": v.get("header_text") or "",
                    "group": hash2group.get(h, ""),
                })

    # per-PDF: full set of digitized page numbers (for true-missing test)
    pdf_pageset = defaultdict(set)
    for pg in pages:
        if pg["pn"] is not None:
            pdf_pageset[pg["pdf"]].add(pg["pn"])

    # assign identity to pages lacking one: nearest preceding head by page number
    by_pdf = defaultdict(list)
    for pg in pages:
        by_pdf[pg["pdf"]].append(pg)
    for pdf, pl in by_pdf.items():
        numbered = sorted([p for p in pl if p["pn"] is not None], key=lambda x: x["pn"])
        last_head_id = None
        for p in numbered:
            if p["head"] and p["tid"]:
                last_head_id = p["tid"]
            if not p["tid"] and last_head_id:
                p["tid"] = last_head_id
                p["tid_inferred"] = True

    # reconstructed tables keyed by (pdf, tid), then SEGMENTED into contiguous
    # page-number runs (a repeated table number across chapters, or a misread
    # page number, must not lump into one giant span).
    SEG_GAP = 12   # consecutive pages within this distance = same physical run
    tables = defaultdict(list)
    for pg in pages:
        key = (pg["pdf"], pg["tid"] or "UNIDENTIFIED")
        tables[key].append(pg)

    rows = []
    for (pdf, tid), pl in sorted(tables.items()):
        numbered = sorted([p for p in pl if p["pn"] is not None], key=lambda x: x["pn"])
        nonum_pages = [p for p in pl if p["pn"] is None]
        # drop outlier/misread page numbers (>60 from every neighbour)
        allpn = sorted(p["pn"] for p in numbered)
        suspects = set()
        if len(allpn) >= 3:
            for p in numbered:
                if min(abs(p["pn"] - x) for x in allpn if x != p["pn"]) > 60:
                    suspects.add(p["hash"])
        clean = [p for p in numbered if p["hash"] not in suspects]

        # segment into contiguous runs
        segs = []
        cur = []
        for p in clean:
            if cur and p["pn"] - cur[-1]["pn"] > SEG_GAP:
                segs.append(cur); cur = []
            cur.append(p)
        if cur:
            segs.append(cur)
        if not segs:  # no usable page numbers; keep as one segment for grouping info
            segs = [pl]

        for si, seg in enumerate(segs):
            segpns = [p["pn"] for p in seg if p["pn"] is not None]
            uniq = sorted(set(segpns))
            internal_gaps = []
            if len(uniq) >= 2:
                internal_gaps = sorted(set(range(uniq[0], uniq[-1] + 1)) - set(uniq))
            dup_pns = sorted(p for p, c in Counter(segpns).items() if c > 1)
            groups_covering = sorted(set(p["group"] for p in seg if p["group"]))
            ungrouped = sum(1 for p in seg if not p["group"])
            rows.append({
                "pdf": pdf,
                "table_identity": tid + (f" [run{si+1}]" if len(segs) > 1 else ""),
                "n_pages_local": len(seg),
                "page_range": f"{uniq[0]}-{uniq[-1]}" if uniq else "",
                "page_numbers": ",".join(map(str, uniq)),
                "pages_without_number": sum(1 for p in seg if p["pn"] is None),
                "has_head": int(any(p["head"] for p in seg)),
                "has_concluded": int(any(p["end"] == "concluded" for p in seg)),
                "duplicate_page_numbers": ",".join(map(str, dup_pns)),
                "internal_gaps": ",".join(map(str, internal_gaps)),
                "n_internal_gaps": len(internal_gaps),
                "v2_groups_covering": " | ".join(g.split("::")[-1][:40] for g in groups_covering),
                "n_v2_groups_covering": len(groups_covering),
                "ungrouped_pages": ungrouped,
                "suspect_page_hashes": ",".join(sorted(suspects)) if si == 0 else "",
                "action": ("MERGE" if len(groups_covering) > 1 else
                           "ADD_UNGROUPED" if groups_covering and ungrouped else
                           "NEW" if not groups_covering else "OK"),
            })

    out = os.path.join(ROOT, f"export_final/assembled_all_read_v2/reconstructed_tables_{a.year}.csv")
    cols = ["pdf", "table_identity", "n_pages_local", "page_range", "page_numbers",
            "pages_without_number", "has_head", "has_concluded",
            "duplicate_page_numbers", "internal_gaps", "n_internal_gaps",
            "n_v2_groups_covering", "v2_groups_covering", "ungrouped_pages",
            "suspect_page_hashes", "action"]
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in sorted(rows, key=lambda x: (x["pdf"], x["table_identity"])):
            w.writerow(r)

    import sys
    print(f"year {a.year}", file=sys.stderr)
    print(f"local pages indexed: {len(pages)}  PDFs: {len(prefix2pdf)}", file=sys.stderr)
    print(f"reconstructed table runs: {len(rows)}", file=sys.stderr)
    print("action mix:", dict(Counter(r["action"] for r in rows)), file=sys.stderr)
    print("runs with internal gaps (<=seg span; may be un-OCR'd OR missing):",
          sum(1 for r in rows if r["n_internal_gaps"]), file=sys.stderr)
    print("  total internal-gap page#s:", sum(r["n_internal_gaps"] for r in rows), file=sys.stderr)
    print("runs spanning >1 v2 group (fragmented -> merge):", sum(1 for r in rows if r["n_v2_groups_covering"] > 1), file=sys.stderr)
    print("runs entirely ungrouped (NEW, digitized but never assembled):", sum(1 for r in rows if r["action"] == "NEW"), file=sys.stderr)
    print("UNIDENTIFIED runs:", sum(1 for r in rows if r["table_identity"].startswith("UNIDENTIFIED")), file=sys.stderr)
    print("wrote", out, file=sys.stderr)


if __name__ == "__main__":
    main()
