#!/usr/bin/env python3
"""Honest status model shared by the provenance workbook + browser.

A table is NOT "complete" just because its pages pass the cell-level arithmetic
check. Complete = the FULL content-detected page sequence is present, in PDF-page
order, ASSEMBLED into one table, and cell-valid. Until the flash-drive content
inventory (SHOULD) confirms the full sequence, completeness is UNVERIFIED — so we
never label a table "Complete" yet; cell-validation is shown only as a sub-attribute.
"""
import re

FILL = {"ok": "C6EFCE", "warn": "FFEB9C", "bad": "FFC7CE", "none": "E2E2E2",
        "info": "DDEBF7", "img": "E4DFEC"}


def clean(s):
    return re.sub(r"^[^0-9A-Za-z]+", "", s or "").strip()


def page_badge(m):
    """(label, class) for a single page/crop."""
    if m.get("ocr_status") == "image_only":
        return ("image present — not OCR'd", "img")
    if m.get("ocr_status") == "NOT_OCRd":
        return ("not OCR'd", "none")
    v = m.get("validation", "")
    if v == "perfect":
        return ("cells valid", "ok")
    if v:
        return (v, "warn")
    st = (m.get("status", "") or "").lower()
    if "fail" in st or "low/" in st:
        return ("OCR low/failed", "bad")
    return ("OCR'd", "info")


def table_status(members):
    """(label, class) for a logical table — leads with assembly + sequence state.
    'Complete' is intentionally NOT emitted here; it requires the content inventory."""
    n = len(members)
    notocr = sum(1 for m in members if m.get("ocr_status") == "NOT_OCRd")
    nbad = sum(1 for m in members if m.get("validation") and m.get("validation") != "perfect")
    nperf = sum(1 for m in members if m.get("validation") == "perfect")
    vii = any(m.get("is_tableVII") == "Y" for m in members)
    is_asm = any(m.get("assembled_table") for m in members)
    n_img = sum(1 for m in members if m.get("ocr_status") == "image_only")
    if n_img == n:
        return (f"imaged {n}pg · not OCR'd", "img")
    if notocr == n:
        return (f"not OCR'd ({n} pg)", "none")
    if is_asm:
        if nbad:
            return (f"assembled {n}pg · {nbad} imperfect cells", "warn")
        tag = " · cells valid" if nperf == n and n else ""
        return (f"assembled {n}pg{tag} · sequence UNVERIFIED", "info")
    if vii:
        return ("single page · NOT assembled", "warn")
    return ("OCR'd · non-Table VII", "info")


def load_image_pages(V1):
    """Synthetic page rows for downloaded Table VII images that aren't OCR'd yet
    (status 'gap' in tableVII_images.csv). Grouped per volume so the workbook/browser
    shows 'newly imaged — not OCR'd' pages slotted by PDF page."""
    import os, csv, hashlib
    out = []
    img = os.path.join(V1, "tableVII_images.csv"); ccv = os.path.join(V1, "cc_volumes.csv")
    if not (os.path.exists(img) and os.path.exists(ccv)):
        return out
    meta = {}
    for r in csv.DictReader(open(ccv)):
        cp = r.get("cluster_path", "")
        if cp:
            meta[r["saoa_id"]] = {"dh": hashlib.sha1(cp.encode()).hexdigest()[:16],
                                  "year": r.get("year", ""), "province": r.get("province_or_volume", ""),
                                  "pdf": cp.split("/")[-1]}
    for r in csv.DictReader(open(img)):
        if r.get("status") != "gap":
            continue
        m = meta.get(r["saoa_id"])
        if not m:
            continue
        pg = int(r["pdf_page"])
        out.append({"status": "image present - not OCR'd", "volume": m["dh"], "year": m["year"],
                    "province": m["province"], "pdf_page": str(pg), "pdf_page_source": "actual",
                    "printed_page": "", "table_number": "", "exact_title": "newly imaged - not OCR'd",
                    "contains_sections": "", "age_range": "", "columns": "", "n_rows": "",
                    "is_tableVII": "Y", "family": "", "assembled_table": f"newimg_{m['dh']}",
                    "position_in_table": "", "ocr_status": "image_only", "validation": "", "verdict": "",
                    "table_id": f"img_{r['saoa_id']}_p{pg}", "imghash": r.get("imghash", ""),
                    "image_path": r["image_path"], "source_pdf": m["pdf"], "_pdf": pg, "_printed": 10 ** 9})
    return out


def is_assembled(members):
    return any(m.get("assembled_table") for m in members)


def is_tableVII(members):
    return any(m.get("is_tableVII") == "Y" for m in members)
