#!/usr/bin/env python3
"""SAFE provenance refresh — updates the page master (page_number_status.csv) to reflect WHERE WE ARE,
WITHOUT re-running build_tableVII_runs.py (which would wipe the manual bounds/title-pages). It re-derives
the assembly columns (assembly_run, run_range, run_position, sub_group) from the CURRENT
export/tableVII_runs.json (the authoritative manual version), marks title pages, and appends any newly
OCR'd member pages (grow-OCR additions) that weren't in the CSV. Everything else is preserved.
  python3 update_provenance_safe.py
Then: python3 refresh_provenance_workbook.py   (mirrors the CSV into the xlsx)
"""
import sys, os, json, csv, re
sys.argv = ["x"]
import census_review_app as A

ROOT = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(ROOT, "export_final/sendout_final/sendout_v1/page_number_status.csv")
RUNS = json.load(open(os.path.join(ROOT, "export/tableVII_runs.json")))


def ih_of(tid):
    m = re.search(r"([0-9a-f]{40})_oneshot", tid)
    return m.group(1) if m else None


def main():
    rows = list(csv.DictReader(open(CSV)))
    cols = list(rows[0].keys())
    by_ih = {r["imghash"]: r for r in rows}

    # build: imghash -> assembly info, from the CURRENT (manual) runs
    asm = {}
    title_ihs = set()
    for run in RUNS:
        saoa = str(run["saoa"]); rng = run["page_range"]; dh = run["dochash"]
        tp = set(run.get("title_pages", []))
        members = sorted(run.get("members", []), key=lambda m: m.get("page", 0))
        for pos, m in enumerate(members):
            ih = ih_of(m.get("table_id", ""))
            if not ih:
                continue
            asm[ih] = {"assembly_run": saoa, "run_range": f"{rng[0]}-{rng[1]}",
                       "run_position": str(pos), "pdf_page": str(m.get("page", "")), "dochash": dh}
        # title-page member imghashes (resolve via member_for to be safe)
        import boundary_apply as B
        for pg in tp:
            mm, _ = B.member_for(dh, pg)
            if mm:
                tih = ih_of(mm["table_id"])
                if tih:
                    title_ihs.add(tih)

    updated = added = 0
    for ih, info in asm.items():
        r = by_ih.get(ih)
        if r is None:                                   # newly OCR'd page not in the CSV -> append it
            nr = {c: "" for c in cols}
            nr.update({"imghash": ih, "dochash": info["dochash"], "pdf_page": info["pdf_page"],
                       "is_tableVII": "1", "page_status": "grow_ocr",
                       "assembly_run": info["assembly_run"], "run_range": info["run_range"],
                       "run_position": info["run_position"]})
            rows.append(nr); by_ih[ih] = nr; added += 1
        else:
            r["assembly_run"] = info["assembly_run"]; r["run_range"] = info["run_range"]
            r["run_position"] = info["run_position"]; updated += 1
    for ih in title_ihs:
        if ih in by_ih:
            by_ih[ih]["title_role"] = "title_page"; by_ih[ih]["is_tableVII"] = "0"

    with open(CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)
    print(f"page master updated from CURRENT runs (manual bounds preserved):")
    print(f"  assembly membership refreshed: {updated} pages | newly-added member pages: {added} | "
          f"title pages marked: {len(title_ihs)}")
    print(f"  total runs reflected: {len(RUNS)} | total page rows: {len(rows)}")


if __name__ == "__main__":
    main()
