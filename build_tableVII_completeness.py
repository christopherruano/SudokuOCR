#!/usr/bin/env python3
"""PER-VOLUME TABLE VII COMPLETENESS — the master view for getting ALL Table VII from
ALL volumes. For each volume: the Table VII pages we have in PDF-page order, an
ordering-consistency check (read page should track PDF page at the fixed offset), the
gaps in the Table VII section (missing PDF pages to OCR/render), and a per-volume
status. Reads master_tracking.csv. Writes tableVII_completeness.csv + .xlsx
(Volumes summary + Pages detail) and a missing-pages action list."""
import os, re, csv
from collections import defaultdict

csv.field_size_limit(10 ** 7)
V1 = "export_final/sendout_final/sendout_v1"
GAP = 2   # within-section tolerance: a jump > GAP+1 is a section/table boundary
VII = re.compile(r"VI[IL]|civil|age.{0,3}sex|C-?III", re.I)


def isint(x):
    return str(x).lstrip("-").isdigit()


def main():
    rows = list(csv.DictReader(open(os.path.join(V1, "master_tracking.csv"))))
    byvol = defaultdict(list)
    for r in rows:
        is_vii = r["is_tableVII"] == "Y" or VII.search(r["table_number"] or "")
        if is_vii:
            byvol[r["volume"]].append(r)

    vol_rows, page_rows, missing_actions = [], [], []
    for dh, imgs in byvol.items():
        paged = [r for r in imgs if isint(r["pdf_page"])]
        paged.sort(key=lambda r: int(r["pdf_page"]))
        meta0 = imgs[0]
        offset = meta0["offset"] if meta0["offset"] not in ("", None) else None
        all_pages = sorted(int(r["pdf_page"]) for r in paged)
        n_negative = sum(1 for p in all_pages if p < 1)
        have = sorted({p for p in all_pages if p >= 1})
        # contiguous Table VII blocks (sections)
        blocks, cur = [], ([have[0]] if have else [])
        for p in have[1:]:
            if p - cur[-1] <= GAP + 1:
                cur.append(p)
            else:
                blocks.append(cur); cur = [p]
        if cur:
            blocks.append(cur)
        # the REAL Table VII section = substantial blocks (>=2 pages); isolated 1-page
        # blocks far from the cluster are suspect/misread reads, not real gaps.
        main_blocks = [b for b in blocks if len(b) >= 2]
        outlier_pages = n_negative + sum(len(b) for b in blocks if len(b) < 2)
        main_have = sorted({p for b in main_blocks for p in b})
        missing = []
        for b in main_blocks:
            missing += [p for p in range(b[0], b[-1] + 1) if p not in have]
        blocks = main_blocks   # report on the real sections only
        have = main_have if main_have else have
        # ordering check: read should equal pdf - offset
        bad_order = 0
        for r in paged:
            page_rows.append({"volume": dh, "year": meta0["year"], "province": meta0["province"],
                              "pdf_page": r["pdf_page"], "read_page": r["printed_page"],
                              "expected_read": (int(r["pdf_page"]) - int(offset)) if offset is not None and isint(r["pdf_page"]) else "",
                              "table_number": r["table_number"], "status": r["status"][:30],
                              "sections": r["contains_sections"][:40]})
            if offset is not None and isint(r["printed_page"]) and isint(r["pdf_page"]):
                if int(r["printed_page"]) != int(r["pdf_page"]) - int(offset):
                    bad_order += 1
        # status counts
        sc = defaultdict(int)
        for r in imgs:
            s = r["status"]
            k = ("perfect" if s.startswith("✅") else "imperfect" if s.startswith("⚠")
                 else "not_ocrd" if "NOT OCR" in s else "failed" if "❌" in s else "other")
            sc[k] += 1
        notocrd_unpaged = sum(1 for r in imgs if not isint(r["pdf_page"]))
        complete = (not missing) and (sc["not_ocrd"] == 0) and (sc["failed"] == 0)
        vol_rows.append({
            "volume": dh, "year": meta0["year"], "province": meta0["province"],
            "source_pdf": meta0["source_pdf"], "offset": offset if offset is not None else "",
            "tableVII_pages_have": len(have),
            "tableVII_pdf_range": f"{have[0]}-{have[-1]}" if have else "",
            "blocks": " | ".join(f"{b[0]}-{b[-1]}" for b in blocks),
            "missing_pdf_pages": ";".join(map(str, missing)),
            "n_missing": len(missing),
            "suspect_page_reads": outlier_pages,
            "unpaged_crops": notocrd_unpaged,
            "perfect": sc["perfect"], "imperfect": sc["imperfect"],
            "not_ocrd": sc["not_ocrd"], "failed": sc["failed"],
            "COMPLETE": "YES" if complete else "NO",
        })
        if missing or sc["not_ocrd"] or sc["failed"]:
            missing_actions.append({"volume": dh, "province": meta0["province"], "year": meta0["year"],
                                    "source_pdf": meta0["source_pdf"], "missing_pdf_pages": ";".join(map(str, missing)),
                                    "never_ocrd_crops": sc["not_ocrd"] + notocrd_unpaged, "failed_crops": sc["failed"]})

    vol_rows.sort(key=lambda r: (r["COMPLETE"] == "YES", -r["n_missing"], -r["not_ocrd"]))
    vcols = ["COMPLETE", "volume", "year", "province", "source_pdf", "offset", "tableVII_pages_have",
             "tableVII_pdf_range", "blocks", "missing_pdf_pages", "n_missing", "suspect_page_reads",
             "unpaged_crops", "perfect", "imperfect", "not_ocrd", "failed"]
    with open(os.path.join(V1, "tableVII_completeness.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=vcols, extrasaction="ignore")
        w.writeheader(); [w.writerow(r) for r in vol_rows]
    pcols = ["volume", "year", "province", "pdf_page", "read_page", "expected_read",
             "table_number", "status", "sections"]
    page_rows.sort(key=lambda r: (str(r["volume"]), int(r["pdf_page"]) if isint(r["pdf_page"]) else 9999))
    with open(os.path.join(V1, "tableVII_pages_detail.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=pcols, extrasaction="ignore")
        w.writeheader(); [w.writerow(r) for r in page_rows]

    try:
        from openpyxl import Workbook
        from openpyxl.styles import Font, PatternFill
        wb = Workbook(); ws = wb.active; ws.title = "Volumes"
        ws.append(vcols)
        for c in ws[1]:
            c.font = Font(bold=True)
        red = PatternFill("solid", fgColor="FFCCCC"); grn = PatternFill("solid", fgColor="CCFFCC")
        for r in vol_rows:
            ws.append([r.get(k, "") for k in vcols])
            ws.cell(ws.max_row, 1).fill = grn if r["COMPLETE"] == "YES" else red
        ws.freeze_panes = "A2"
        ws2 = wb.create_sheet("Pages")
        ws2.append(pcols)
        for c in ws2[1]:
            c.font = Font(bold=True)
        for r in page_rows:
            ws2.append([r.get(k, "") for k in pcols])
        ws2.freeze_panes = "A2"
        wb.save(os.path.join(V1, "tableVII_completeness.xlsx"))
        xlsx = " + tableVII_completeness.xlsx"
    except Exception as e:
        xlsx = f" (xlsx skipped: {e})"

    nvol = len(vol_rows); ncomplete = sum(1 for r in vol_rows if r["COMPLETE"] == "YES")
    print(f"Table VII volumes: {nvol} | COMPLETE: {ncomplete} | incomplete: {nvol - ncomplete}")
    print(f"  total missing PDF pages (gaps in Table VII sections): {sum(r['n_missing'] for r in vol_rows)}")
    print(f"  total never-OCR'd Table VII crops: {sum(r['not_ocrd'] for r in vol_rows)}")
    print(f"  volumes with SUSPECT page reads (negative/outlier pages to re-read): {sum(1 for r in vol_rows if r['suspect_page_reads'])}")
    print(f"wrote tableVII_completeness.csv + tableVII_pages_detail.csv{xlsx}")


if __name__ == "__main__":
    main()
