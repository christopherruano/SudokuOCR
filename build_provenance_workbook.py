#!/usr/bin/env python3
"""PROVENANCE WORKBOOK — a nice, browsable Excel of everything we have.

Three-level collapsible outline (the +/- dropdowns in Excel's left margin):
    VOLUME  ▸  TABLE  ▸  PAGE
ordered by PDF page number at every level. Each row shows provenance + how it's
doing (OCR / validation / assembled / complete). Reads master_tracking.csv (built by
build_master_tracker.py), so it upgrades to TRUE per-page PDF order automatically the
moment crop_pdf_pages.csv (the cluster/flash-drive page map) is present.

  python3 build_master_tracker.py          # refresh the data
  python3 build_provenance_workbook.py      # -> provenance_workbook.xlsx
"""
import os, re, csv
from collections import defaultdict
from provenance_status import table_status, page_badge, FILL, load_image_pages

csv.field_size_limit(10 ** 7)
ROOT = os.path.dirname(os.path.abspath(__file__))
V1 = os.path.join(ROOT, "export_final/sendout_final/sendout_v1")
HEX = re.compile(r"([0-9a-f]{40})")


def as_int(x, default=10 ** 9):
    try:
        return int(float(x))
    except Exception:
        return default


def load_rows():
    p = os.path.join(V1, "master_tracking.csv")
    rows = list(csv.DictReader(open(p)))
    for r in rows:
        r["_pdf"] = as_int(r.get("pdf_page", ""))
        r["_printed"] = as_int(r.get("printed_page", ""))
    return rows


def load_completeness():
    p = os.path.join(V1, "tableVII_completeness.csv")
    out = {}
    if os.path.exists(p):
        for r in csv.DictReader(open(p)):
            out[r.get("volume", "")] = r
    return out


def load_provenance():
    """authoritative volume identity, preferring volumes.csv (from the flash drive)
    keyed by saoa_id, falling back to documents.csv keyed by dochash."""
    prov = {}                       # dochash -> {year, province, title, saoa}
    docs = os.path.join(V1, "documents.csv")
    saoa_to_doc = {}
    if os.path.exists(docs):
        for r in csv.DictReader(open(docs)):
            dh = r.get("dochash", "")
            saoa = (re.search(r"(\d{6,})", r.get("source_pdf", "") or "") or [None, ""])[1] \
                if re.search(r"(\d{6,})", r.get("source_pdf", "") or "") else ""
            prov[dh] = {"year": r.get("year", ""), "province": r.get("region", ""),
                        "title": r.get("title", "") or r.get("source_pdf", ""), "saoa": saoa}
            if saoa:
                saoa_to_doc[saoa] = dh
    vols = os.path.join(V1, "volumes.csv")          # arrives from the flash-drive CC
    if os.path.exists(vols):
        for r in csv.DictReader(open(vols)):
            dh = saoa_to_doc.get(r.get("saoa_id", ""))
            if dh:
                prov.setdefault(dh, {})
                prov[dh].update({"year": r.get("year", "") or prov[dh].get("year", ""),
                                 "province": r.get("province_or_volume", "") or prov[dh].get("province", ""),
                                 "title": r.get("title", "") or prov[dh].get("title", "")})
    return prov


# ---- status -> fill colour (keyword-based, no emoji) ------------------------
GREEN, YELLOW, RED, GREY, BLUE = "C6EFCE", "FFEB9C", "FFC7CE", "E2E2E2", "DDEBF7"
def clean(s):
    return re.sub(r"^[^0-9A-Za-z]+", "", s or "").strip()
def status_fill(s):
    low = (s or "").lower()
    if "not ocr" in low:
        return GREY
    if "fail" in low or "low/" in low:
        return RED
    if "imperfect" in low or "retry" in low:
        return YELLOW
    if "complete" in low or "perfect" in low:
        return GREEN
    return BLUE


def table_rollup(members):
    """one summary line for a logical table from its page rows."""
    pdfs = sorted(m["_pdf"] for m in members if m["_pdf"] < 10 ** 9)
    rng = f"p{pdfs[0]}–p{pdfs[-1]}" if pdfs else ""
    n = len(members)
    nperf = sum(1 for m in members if m.get("validation") == "perfect")
    nbad = sum(1 for m in members if m.get("validation") and m.get("validation") != "perfect")
    vii = any(m.get("is_tableVII") == "Y" for m in members)
    st, cls = table_status(members)            # honest: assembly + sequence, never "complete" yet
    # representative title / number / family
    title = ""
    for m in members:
        if m.get("exact_title"):
            title = m["exact_title"]; break
    fam = next((m.get("family", "") for m in members if m.get("family")), "")
    tnum = next((m.get("table_number", "") for m in members if m.get("table_number")), "")
    return {"range": rng, "n": n, "nperf": nperf, "nbad": nbad, "vii": vii,
            "status": st, "cls": cls, "title": title, "family": fam, "tnum": tnum,
            "minpdf": pdfs[0] if pdfs else 10 ** 9}


def build():
    rows = load_rows() + load_image_pages(V1)
    comp = load_completeness()
    prov = load_provenance()
    # group: volume -> logical_table -> [page rows]
    vols = defaultdict(lambda: defaultdict(list))
    for r in rows:
        ltab = r.get("assembled_table") or r.get("table_id") or f"~untabled:{r.get('exact_title','')[:40]}"
        vols[r["volume"]][ltab].append(r)

    try:
        from openpyxl import Workbook
        from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
        from openpyxl.utils import get_column_letter
    except Exception as e:
        raise SystemExit(f"need openpyxl: pip install openpyxl ({e})")

    wb = Workbook()
    ws = wb.active; ws.title = "Browse"
    ws.sheet_properties.outlinePr.summaryBelow = False   # summary row ABOVE its detail

    COLS = ["Item", "PDF pg", "Printed", "Status", "Validation", "VII",
            "Contains (sections)", "Age", "Columns", "Rows", "Title / provenance",
            "Source PDF", "ID", "Image"]
    ws.append(COLS)
    hdr_fill = PatternFill("solid", fgColor="1F3864")
    for c in ws[1]:
        c.font = Font(bold=True, color="FFFFFF"); c.fill = hdr_fill
        c.alignment = Alignment(vertical="center")
    ws.freeze_panes = "A2"
    thin = Side(style="thin", color="BFBFBF")
    border = Border(bottom=thin)

    def put(values, level, fill=None, bold=False, hidden=False, link=None, italic=False):
        ws.append(values)
        r = ws.max_row
        for c in ws[r]:
            c.alignment = Alignment(vertical="center", wrap_text=False)
            if fill:
                c.fill = PatternFill("solid", fgColor=fill)
            if bold:
                c.font = Font(bold=True, color="FFFFFF" if level == 0 else "000000")
            elif italic:
                c.font = Font(italic=True, color="595959")
        if level:
            ws.row_dimensions[r].outline_level = level
        if hidden:
            ws.row_dimensions[r].hidden = True
        if link:
            cell = ws.cell(row=r, column=len(COLS))
            cell.hyperlink = link; cell.value = "open"; cell.font = Font(color="0563C1", underline="single")
        return r

    # order volumes by (has-provenance, year, province) so populated ones lead
    def vkey(dh):
        pr = prov.get(dh, {})
        y = pr.get("year", "")
        has = 0 if (y or pr.get("province")) else 1
        return (has, as_int(y, 9999), str(pr.get("province", "")), dh)

    for dh in sorted(vols, key=vkey):
        pr = prov.get(dh, {})
        tables = vols[dh]
        # volume rollup
        allrows = [m for t in tables.values() for m in t]
        npages = len(allrows)
        ntab = len(tables)
        nperf = sum(1 for m in allrows if m.get("validation") == "perfect")
        cm = comp.get(dh, {})
        miss = cm.get("n_missing", "")
        comp_flag = cm.get("COMPLETE", "")
        span = ""
        pdfs = sorted(m["_pdf"] for m in allrows if m["_pdf"] < 10 ** 9)
        if pdfs:
            span = f"p{pdfs[0]}–p{pdfs[-1]}"
        n_asm = sum(1 for t in tables.values() if any(m.get("assembled_table") for m in t))
        vstat = [f"{ntab} tables · {npages} pages", f"{n_asm} assembled", f"{nperf} cell-valid"]
        head = f"{pr.get('year','') or '—'}   ·   {pr.get('province','') or pr.get('saoa','') or dh[:8]}"
        put([head, span, "", " · ".join(vstat), "", "", "", "", "", "",
             (pr.get("title", "") or "")[:90], pr.get("saoa", "") or "", dh, ""],
            level=0, fill="2F5496", bold=True)

        # order tables by min pdf page
        roll = {t: table_rollup(m) for t, m in tables.items()}
        for t in sorted(tables, key=lambda t: (roll[t]["minpdf"], t)):
            R = roll[t]
            tname = (f"Table {R['tnum']}" if R["tnum"] else
                     (R["family"] or (R["title"][:34] if R["title"] else "Table")))
            tlabel = "    " + tname + (f" · {R['family']}" if R["tnum"] and R["family"] else "")
            put([tlabel, R["range"], "", R["status"], f"{R['nperf']}/{R['n']} cell-valid" if R["n"] else "",
                 "VII" if R["vii"] else "", "", "", "", R["n"],
                 (R["title"] or "")[:90], "", t[-12:], ""],
                level=1, fill=FILL.get(R["cls"], "DDEBF7"), bold=False)
            # pages within table, ordered by pdf page then printed
            for m in sorted(tables[t], key=lambda r: (r["_pdf"], r["_printed"])):
                pdf = m.get("pdf_page", ""); psrc = m.get("pdf_page_source", "")
                # lead with the DISTINCT page we actually have: true per-crop PDF page
                # when present, else the printed/book page (distinct per page now),
                # else the provisional manifest table-start page.
                if psrc == "actual" and pdf not in ("", None):
                    plabel = f"        PDF p{pdf}"
                elif m["_printed"] < 10 ** 9:
                    plabel = f"        p.{m['_printed']} (book)"
                elif pdf not in ("", None):
                    plabel = f"        PDF p{pdf} (prov)"
                else:
                    plabel = f"        {m.get('imghash','')[:8]}"
                link = None
                ip = m.get("image_path", "")
                if ip:
                    link = "file://" + os.path.join(ROOT, ip)
                pb_label, pb_cls = page_badge(m)
                put([plabel, pdf, m.get("printed_page", ""), pb_label,
                     m.get("validation", ""), "VII" if m.get("is_tableVII") == "Y" else "",
                     (m.get("contains_sections", "") or "")[:50], m.get("age_range", ""),
                     (m.get("columns", "") or "")[:40], m.get("n_rows", ""),
                     (m.get("exact_title", "") or "")[:90], m.get("source_pdf", ""),
                     m.get("imghash", "")[:10], ""],
                    level=2, fill=FILL.get(pb_cls, "DDEBF7"), hidden=True, link=link)

    widths = [34, 8, 8, 26, 14, 5, 30, 9, 26, 6, 40, 24, 14, 7]
    for i, w in enumerate(widths, 1):
        ws.column_dimensions[get_column_letter(i)].width = w
    ws.auto_filter.ref = f"A1:{get_column_letter(len(COLS))}1"

    # ---- Volumes overview sheet ----
    ws2 = wb.create_sheet("Volumes")
    vp = os.path.join(V1, "master_tracking_volumes.csv")
    if os.path.exists(vp):
        vr = list(csv.DictReader(open(vp)))
        vcols = list(vr[0].keys()) if vr else []
        ws2.append(vcols)
        for c in ws2[1]:
            c.font = Font(bold=True, color="FFFFFF"); c.fill = hdr_fill
        # sort by year then province
        for r in sorted(vr, key=lambda r: (str(r.get("year", "")), str(r.get("province", "")))):
            ws2.append([r.get(k, "") for k in vcols])
        ws2.freeze_panes = "A2"
        for i in range(1, len(vcols) + 1):
            ws2.column_dimensions[get_column_letter(i)].width = 16

    out = os.path.join(V1, "provenance_workbook.xlsx")
    wb.save(out)
    nvol = len(vols)
    npage = sum(len(m) for t in vols.values() for m in t.values())
    print(f"wrote {out}")
    print(f"  {nvol} volumes · {sum(len(t) for t in vols.values())} tables · {npage} pages")
    print(f"  ordered by PDF page; outline: VOLUME ▸ TABLE ▸ PAGE (pages collapsed by default)")
    has_real = any(r.get("pdf_page_source") == "actual" for r in rows)
    print("  PDF pages: " + ("AUTHORITATIVE (crop_pdf_pages.csv present)" if has_real
                             else "provisional (manifest/est) — drop in crop_pdf_pages.csv for true per-page order"))


if __name__ == "__main__":
    build()
