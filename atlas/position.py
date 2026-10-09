#!/usr/bin/env python3
"""Where each table sits: volume (dochash + PDF), PDF page, printed page.

page_number_status.csv maps every page image to its PDF and PDF page; provenance.csv
adds the printed page number where it was read. Assembled tables take their first
member's position.
"""
import csv, os, re, json
HERE = os.path.dirname(os.path.abspath(__file__))
SV1 = os.path.join(HERE, "..", "export_final/sendout_final/sendout_v1")


def positions(table_rows):
    """table_rows: iterable of (table_id, source_image, member_images).
    Returns {table_id: {"doc", "pdf", "pdf_page", "pg"}}."""
    pns = {r["imghash"]: r for r in csv.DictReader(open(os.path.join(SV1, "page_number_status.csv")))}
    prov = {r["table_id"]: r for r in csv.DictReader(open(os.path.join(SV1, "provenance.csv")))}
    members = json.load(open(os.path.join(SV1, "assembled_members.json")))
    out = {}
    for tid, si, mi in table_rows:
        hs = re.findall(r"[0-9a-f]{40}", " ".join([tid, si or "", mi or ""] + members.get(tid, [])))
        hits = [pns[h] for h in hs if h in pns and pns[h]["pdf_page"].lstrip("-").isdigit()]
        p = prov.get(tid) or {}
        if hits:
            h = min(hits, key=lambda r: int(r["pdf_page"]))
            out[tid] = {"doc": h["dochash"], "pdf": os.path.basename(h["source_pdf"]),
                        "pdf_page": int(h["pdf_page"]), "pg": p.get("page_number", "") or ""}
        elif p.get("dochash"):
            out[tid] = {"doc": p["dochash"], "pdf": os.path.basename(p.get("source_pdf", "")),
                        "pdf_page": None, "pg": p.get("page_number", "") or ""}
    return out
