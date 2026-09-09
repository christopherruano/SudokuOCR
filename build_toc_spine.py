#!/usr/bin/env python3
"""Merge per-volume TOC vision output (export/toc_spine/{saoa}.json) into a clean spine:
  chapter -> table -> {num, title, start_page, end_page}, sorted by printed page.
Deterministic post-processing (normalize table numbers, parse pages, derive ranges). The
page->PDF-page offset is applied later from a few printed-page reads; here we keep printed pages.
  python3 build_toc_spine.py
Output: export/toc_spine_merged.json + a readable summary.
"""
import os, glob, json, re

IN = "export/toc_spine"
OUT = "export/toc_spine_merged.json"
ROMAN = re.compile(r"\b([IVXLC]+)\b")


def norm_num(s):
    """normalize a table identifier token for comparison/grouping."""
    s = (s or "").strip().upper()
    s = re.sub(r"\s+", " ", s)
    s = s.replace("IMPERIAL TABLE", "TABLE").replace("PROVINCIAL TABLE", "TABLE")
    return s


def to_int(p):
    try:
        return int(re.sub(r"[^\d]", "", str(p)))
    except Exception:
        return None


def main():
    merged = {}
    n_vol = n_tab = 0
    for f in sorted(glob.glob(f"{IN}/*.json")):
        d = json.load(open(f))
        ents = []
        for e in d.get("entries", []):
            pg = to_int(e.get("page"))
            ents.append({
                "chapter": (e.get("chapter") or "").strip(),
                "num": norm_num(e.get("table_num")),
                "num_raw": (e.get("table_num") or "").strip(),
                "title": (e.get("title") or "").strip(),
                "start_page": pg,
            })
        # keep entries with a usable page, sort by page, derive end_page = next start - 1
        paged = sorted([e for e in ents if e["start_page"] is not None], key=lambda x: x["start_page"])
        for i, e in enumerate(paged):
            e["end_page"] = (paged[i + 1]["start_page"] - 1) if i + 1 < len(paged) else None
        # group by chapter (preserve order of first appearance)
        chapters = []
        ch_index = {}
        for e in paged:
            ch = e["chapter"] or "(unchaptered)"
            if ch not in ch_index:
                ch_index[ch] = len(chapters)
                chapters.append({"chapter": ch, "tables": []})
            chapters[ch_index[ch]]["tables"].append(
                {"num": e["num"], "num_raw": e["num_raw"], "title": e["title"],
                 "start_page": e["start_page"], "end_page": e["end_page"]})
        merged[d["saoa"]] = {"year": d.get("year"), "n_toc_pages": len(d.get("toc_pages", [])),
                             "chapters": chapters, "n_tables": len(paged)}
        n_vol += 1
        n_tab += len(paged)
    json.dump(merged, open(OUT, "w"), indent=1)
    print(f"merged {n_vol} volumes, {n_tab} tables with pages -> {OUT}")
    # readable summary
    for sa, v in list(merged.items())[:4]:
        print(f"\n  {sa} ({v['year']}): {len(v['chapters'])} chapters, {v['n_tables']} tables")
        for ch in v["chapters"][:3]:
            print(f"    [{ch['chapter'][:34]}] {len(ch['tables'])} tables")
            for t in ch["tables"][:3]:
                print(f"       {t['num'][:22]:<22} p{t['start_page']}-{t['end_page']}  {t['title'][:34]}")


if __name__ == "__main__":
    main()
