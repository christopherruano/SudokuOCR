#!/usr/bin/env python3
"""For AI boundary review: for each Table VII run, show the title of the page just OUTSIDE each end
(where the crawl stopped) so a human/LLM can judge SAME-table-continues (extend) vs DIFFERENT-table
(correct boundary). Flags AMBIGUOUS stops — the page reads Table-VII-ish or is a continued/part/
district/chapter title — which is exactly where a regex is unreliable and AI title-reading helps.

Pulls titles from the crawl's boundary reads (export/boundary_grow_ocr) + the master. Writes
export/boundary_review.csv and prints the ambiguous ones.
  python3 boundary_titles_report.py
"""
import json, os, re, csv, glob, collections

ROOT = os.path.dirname(os.path.abspath(__file__))
V1 = os.path.join(ROOT, "export_final/sendout_final/sendout_v1")
RUNS = os.path.join(ROOT, "export/tableVII_runs.json")
MASTER = os.path.join(V1, "page_number_status.csv")
BG = os.path.join(ROOT, "export/boundary_grow_ocr")

VII = re.compile(r"\bvii\b|age|sex|civil|marital|conjugal|religion|caste", re.I)
DIFF = re.compile(r"education|literacy|language|birth|occupation|infirm|\bviii\b|\bvi\b|\bv\b|\bix\b|\bx\b|"
                  r"livelihood|housing|density|migration|variation|town|village", re.I)
CONT = re.compile(r"cont|contd|continued|part|district|concl|taluk|division", re.I)


def main():
    runs = json.load(open(RUNS))
    # probe reads: (saoa, page) -> (title, nrows)
    probe = {}
    for f in glob.glob(os.path.join(BG, "**", "*_oneshot.json"), recursive=True):
        m = re.search(r"(\d+|[0-9a-f]{16})_p(\d+)_oneshot", os.path.basename(f))
        if not m:
            continue
        try:
            d = json.load(open(f))
        except Exception:
            continue
        title = ((d.get("metadata", {}) or {}).get("title", "") or "").strip()
        nrows = sum(len(s.get("rows", [])) for s in d.get("data", []))
        probe[(m.group(1), int(m.group(2)))] = (title, nrows)
    # master titles by (dochash, page)
    mtitle = {}
    for r in csv.DictReader(open(MASTER)):
        if r.get("dochash") and r.get("pdf_page", "").lstrip("-").isdigit():
            mtitle.setdefault((r["dochash"], int(r["pdf_page"])), (r.get("title", ""), r.get("is_tableVII") == "1"))

    def title_at(saoa, dh, pg):
        if (saoa, pg) in probe:
            return probe[(saoa, pg)]  # (title, nrows)
        if (dh, pg) in mtitle:
            t, v = mtitle[(dh, pg)]
            return (t, 1 if v else 0)
        return (None, None)

    rows = []
    ambiguous = []
    for run in runs:
        sa, dh = run["saoa"], run["dochash"]
        s, e = run["page_range"]
        rtitle = ""
        for m in run["members"]:
            pass
        # run identity = most common member title
        mems = []
        for m in run["members"]:
            im = m.get("title", "")
        for side, pg in (("before", s - 1), ("after", e + 1)):
            t, nr = title_at(sa, dh, pg)
            if t is None:
                cls = "unread/edge"
            elif not t.strip() or "blank" in t.lower():
                cls = "blank->stop"
            elif DIFF.search(t) and not VII.search(t):
                cls = "different-table->stop"
            elif VII.search(t) or CONT.search(t):
                cls = "AMBIGUOUS"
            else:
                cls = "other->stop"
            row = {"run": f"{run['saoa']}:{s}-{e}", "region": run.get("region", ""),
                   "year": run.get("year", ""), "n_pages": run["n_pages"], "side": side,
                   "boundary_page": pg, "boundary_title": (t or "")[:75], "rows": nr, "class": cls}
            rows.append(row)
            if cls == "AMBIGUOUS":
                ambiguous.append(row)

    with open(os.path.join(ROOT, "export/boundary_review.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["run", "region", "year", "n_pages", "side", "boundary_page", "boundary_title", "rows", "class"])
        w.writeheader(); w.writerows(rows)

    cc = collections.Counter(r["class"] for r in rows)
    print(f"boundary review: {len(runs)} runs, {len(rows)} ends examined")
    print(f"  {dict(cc)}")
    print(f"\nAMBIGUOUS stops to AI-judge ({len(ambiguous)}) — title where the crawl stopped:")
    for r in ambiguous[:60]:
        print(f"  [{r['side']:6s}] {r['region'][:16]:16s} {r['year']} pp{r['boundary_page']:>4} rows={r['rows']}  {r['boundary_title']!r}")


if __name__ == "__main__":
    main()
