#!/usr/bin/env python3
"""Provenance + document backbone for stitching (Neil's requirement):
tie every table (existing on-site + keeper) to its source PDF/document, so
multi-page tables can be grouped per volume and ordered by page.

Outputs (sendout_v1/):
  provenance.csv   one row per table: table_id, source, dochash, source_pdf,
                   year, region, table_seq, family, title
  documents.csv    one row per source document: dochash, source_pdf, year,
                   region, n_tables (existing/keeper), table_seq range
"""
import csv, json, re, glob, os
from collections import defaultdict, Counter

csv.field_size_limit(10 ** 7)
ROOT = os.path.dirname(os.path.abspath(__file__))
V1 = os.path.join(ROOT, "export_final/sendout_final/sendout_v1")
SEQ = re.compile(r"([0-9a-f]{16})_t(\d+)_")


def parse(img):
    m = SEQ.search(img or "")
    return (m.group(1), int(m.group(2))) if m else ("", -1)


def main():
    # dochash -> source PDF (from digitization manifests)
    doc2pdf = {}
    for mf in glob.glob(os.path.join(ROOT, "export/**/organized_manifest.jsonl"), recursive=True):
        for line in open(mf):
            try:
                r = json.loads(line)
            except Exception:
                continue
            dh, _ = parse(r.get("source_image", ""))
            pdf = r.get("digitized_source_file", "")
            if dh and pdf:
                doc2pdf.setdefault(dh, pdf)

    # deterministic page numbers from the never-OCR'd triage (vision-read), keyed by dochash_tNNNN
    dt = re.compile(r"([0-9a-f]{16}_t\d+)")
    tri_page = {}
    for fn in ("keepers.csv", "triage_all.csv"):
        p = os.path.join(ROOT, "export/never_ocrd_triage", fn)
        if not os.path.exists(p):
            continue
        for r in csv.DictReader(open(p)):
            m = dt.search(r.get("dir", ""))
            if m and (r.get("page_number") or "").strip():
                tri_page.setdefault(m.group(1), {"page": r["page_number"].strip(), "ident": r.get("table_identity", "")})

    # AUTHORITATIVE per-table PDF page: crop_pdf_pages.csv (imghash -> pdf_page) joined to each
    # table via its crop in digitized_tables/{table_id}_*/{imghash}_oneshot.json. Wins over triage.
    import glob as _glob
    crop_pdf = {}
    _cpp = os.path.join(V1, "crop_pdf_pages.csv")
    if os.path.exists(_cpp):
        for r in csv.DictReader(open(_cpp)):
            if str(r.get("pdf_page", "")).lstrip("-").isdigit():
                crop_pdf[r["imghash"]] = r["pdf_page"]
    auth_page = {}   # table_id (dochash_tNNNN) -> authoritative pdf_page
    for d in _glob.glob(os.path.join(ROOT, "export/ocr_results_referenced/digitized_tables/*")):
        mt = re.match(r"([0-9a-f]{16}_t\d+)", os.path.basename(d))
        if not mt:
            continue
        for jf in _glob.glob(os.path.join(d, "*_oneshot.json")):
            ih = re.search(r"([0-9a-f]{40})_oneshot", os.path.basename(jf))
            if ih and ih.group(1) in crop_pdf:
                auth_page[mt.group(1)] = crop_pdf[ih.group(1)]
                break

    def page_of(img_or_dir):
        m = dt.search(img_or_dir or "")
        if not m:
            return {}
        tid = m.group(1)
        d = dict(tri_page.get(tid, {}))
        if tid in auth_page:                 # PDF page is authoritative
            d["page"] = auth_page[tid]
        return d

    rows = []
    # existing on-site/queue tables
    for r in csv.DictReader(open(os.path.join(V1, "catalog_tables.csv"))):
        dh, seq = parse(r["source_image"])
        rows.append({"table_id": r["table_id"], "source": "existing", "dochash": dh,
                     "source_pdf": doc2pdf.get(dh, ""), "year": r["year"], "region": r["broad_region"] or r["canonical_name"],
                     "table_seq": seq, "family": r["table_family"], "status": r["status"],
                     "page_number": page_of(r["source_image"]).get("page",""),
                     "table_ident": page_of(r["source_image"]).get("ident",""),
                     "title": (r["title"] or "")[:90]})
    # keeper tables
    for r in csv.DictReader(open(os.path.join(ROOT, "export/keeper_inference/source_catalog.csv"))):
        dh, seq = parse(r["source_image"])
        rows.append({"table_id": r["table_id"], "source": "keeper", "dochash": dh,
                     "source_pdf": doc2pdf.get(dh, ""), "year": r["year"], "region": r.get("region", ""),
                     "table_seq": seq, "family": r.get("table_family", ""),
                     "status": r.get("effective_quality_status", ""),
                     "page_number": page_of(r["source_table_dir_name"] or r["source_image"]).get("page",""),
                     "table_ident": page_of(r["source_table_dir_name"] or r["source_image"]).get("ident",""),
                     "title": (r.get("title") or "")[:90]})

    with open(os.path.join(V1, "provenance.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["table_id", "source", "dochash", "source_pdf", "year",
                                          "region", "table_seq", "page_number", "table_ident", "family", "status", "title"])
        w.writeheader(); [w.writerow(r) for r in rows]

    # per-document summary
    docs = defaultdict(list)
    for r in rows:
        if r["dochash"]:
            docs[r["dochash"]].append(r)
    with open(os.path.join(V1, "documents.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dochash", "source_pdf", "year", "region", "n_existing", "n_keeper", "seq_min", "seq_max"])
        for dh, ts in sorted(docs.items(), key=lambda x: -len(x[1])):
            seqs = [t["table_seq"] for t in ts if t["table_seq"] >= 0]
            pdf = next((t["source_pdf"] for t in ts if t["source_pdf"]), "")
            yr = next((t["year"] for t in ts if t["year"]), "")
            reg = next((t["region"] for t in ts if t["region"]), "")
            w.writerow([dh, pdf, yr, reg,
                        sum(1 for t in ts if t["source"] == "existing"),
                        sum(1 for t in ts if t["source"] == "keeper"),
                        min(seqs) if seqs else "", max(seqs) if seqs else ""])

    print(f"provenance rows: {len(rows):,} ({Counter(r['source'] for r in rows)})")
    print(f"distinct source documents: {len(docs)}")
    print(f"  with source PDF resolved: {sum(1 for dh in docs if doc2pdf.get(dh))}")
    print(f"  documents with BOTH existing+keeper tables (to stitch): "
          f"{sum(1 for ts in docs.values() if any(t['source']=='existing' for t in ts) and any(t['source']=='keeper' for t in ts))}")
    print("wrote provenance.csv, documents.csv")


if __name__ == "__main__":
    main()
