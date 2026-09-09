#!/usr/bin/env python3
"""Prepare the AMBIGUOUS boundary decisions for AI judgment (fan-out). For each ambiguous stop, emit the
context an LLM needs to decide SAME-table-continues (extend) vs DIFFERENT-table (stop):
  run identity (region/year/its own title + the title of its first & last page), the boundary page's
  title + row count, and the title one page BEYOND it (does the table keep going?).
Writes export/boundary_judgments.json (a list the workflow slices into agent batches).
  python3 build_boundary_judgments.py
"""
import json, os, re, csv, glob

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
    probe = {}
    for f in glob.glob(os.path.join(BG, "**", "*_oneshot.json"), recursive=True):
        m = re.search(r"(\d+|[0-9a-f]{16})_p(\d+)_oneshot", os.path.basename(f))
        if not m:
            continue
        try:
            d = json.load(open(f))
        except Exception:
            continue
        t = ((d.get("metadata", {}) or {}).get("title", "") or "").strip()
        nr = sum(len(s.get("rows", [])) for s in d.get("data", []))
        probe[(m.group(1), int(m.group(2)))] = (t, nr)
    mtitle = {}
    for r in csv.DictReader(open(MASTER)):
        if r.get("dochash") and r.get("pdf_page", "").lstrip("-").isdigit():
            mtitle.setdefault((r["dochash"], int(r["pdf_page"])), (r.get("title", ""), r.get("is_tableVII") == "1"))

    def title_at(sa, dh, pg):
        if (sa, pg) in probe:
            return probe[(sa, pg)]
        if (dh, pg) in mtitle:
            t, v = mtitle[(dh, pg)]
            return (t, 1 if v else 0)
        return (None, None)

    todo = []
    for run in runs:
        sa, dh = run["saoa"], run["dochash"]
        s, e = run["page_range"]
        first_t = title_at(sa, dh, s)[0] or mtitle.get((dh, s), ("", 0))[0]
        last_t = title_at(sa, dh, e)[0] or mtitle.get((dh, e), ("", 0))[0]
        for side, pg, beyond in (("before", s - 1, s - 2), ("after", e + 1, e + 2)):
            t, nr = title_at(sa, dh, pg)
            if t is None or not t.strip() or "blank" in t.lower():
                continue
            if DIFF.search(t) and not VII.search(t):
                continue   # clearly a different table -> auto-stop, no AI needed
            if not (VII.search(t) or CONT.search(t)):
                continue   # not VII-ish, not continuation -> auto-stop
            bt = title_at(sa, dh, beyond)[0]
            todo.append({
                "run": f"{run['saoa']}:{s}-{e}", "region": run.get("region", ""), "year": run.get("year", ""),
                "run_pages": f"{s}-{e}", "run_first_page_title": (first_t or "")[:90],
                "run_last_page_title": (last_t or "")[:90],
                "side": side, "boundary_page": pg, "boundary_title": (t or "")[:90], "boundary_rows": nr,
                "one_page_beyond_title": (bt or "")[:90],
            })
    json.dump(todo, open(os.path.join(ROOT, "export/boundary_judgments.json"), "w"), indent=0)
    print(f"wrote export/boundary_judgments.json: {len(todo)} ambiguous boundaries for AI judgment")


if __name__ == "__main__":
    main()
