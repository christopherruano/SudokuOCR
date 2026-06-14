#!/usr/bin/env python3
"""RUN ON THE CLUSTER, from ~/census_toc (where digitized_tables/ + the PDFs live).

Recovers, for EVERY crop image we have, its exact PDF page and full source provenance,
and writes one file (crop_pdf_pages.csv) to sync back to the laptop.

How the page number is known (the rendering is deterministic):
  during digitization each page was rendered as sha1(f"{resolved_pdf_path}:{page}").png
  and recorded in that table's metadata.json -> rendered_pages: [{pdf_page, image_path}].
So we recover the page number three ways, in priority order:
  (1) DIRECT  : metadata.json rendered_pages maps the crop's imghash -> pdf_page.
  (2) REVERSE : for any crop not in rendered_pages, recompute sha1(f"{source_file}:{p}")
                for every page p of that volume's PDF and match the imghash.
  (3) TABLE   : fall back to the table's resolved_start_page (page range anchor).

Provenance per image: source_pdf, collection, year, province, title, table page range.

Usage:
  cd ~/census_toc
  python3 cluster_extract_pdf_pages.py --root digitized_tables --out crop_pdf_pages.csv
Then: rsync crop_pdf_pages.csv back to <laptop>/export_final/sendout_final/sendout_v1/
"""
import os, re, csv, json, glob, hashlib, argparse, subprocess
from collections import defaultdict

HEX = re.compile(r"([0-9a-f]{40})")
DT = re.compile(r"([0-9a-f]{16})_t(\d+)")


def page_count(pdf):
    try:
        out = subprocess.run(["pdfinfo", pdf], capture_output=True, text=True, timeout=60).stdout
        m = re.search(r"Pages:\s+(\d+)", out)
        return int(m.group(1)) if m else 0
    except Exception:
        return 0


def collection_of(pdf):
    b = os.path.basename(pdf or "")
    low = b.lower()
    if "saoa" in low and "crl" in low:
        return "SAOA via CRL"
    if "saoa" in low:
        return "SAOA"
    if low.startswith("crl"):
        return "CRL"
    # fall back to the leading token of the filename / the parent dir
    parent = os.path.basename(os.path.dirname(pdf or ""))
    return (b.split(".")[0] if "." in b else parent) or "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="digitized_tables", help="dir of {dochash}_tNNNN_*/ crop folders")
    ap.add_argument("--out", default="crop_pdf_pages.csv")
    ap.add_argument("--no-reverse", action="store_true", help="skip the hash-reversal fallback (faster)")
    args = ap.parse_args()

    # ---- pass 1: read every metadata.json (direct mapping + per-table provenance) ----
    direct = {}              # imghash -> pdf_page
    tdir_meta = {}           # tdir basename -> provenance dict (incl. source_file)
    for md in glob.glob(os.path.join(args.root, "*", "metadata.json")):
        try:
            d = json.load(open(md))
        except Exception:
            continue
        tdir = os.path.basename(os.path.dirname(md))
        qr = d.get("queue_row", {}) or {}
        src = d.get("source_file") or qr.get("file") or ""
        tdir_meta[tdir] = {
            "source_file": src,
            "year": qr.get("year", ""), "province": qr.get("province", ""),
            "title": qr.get("display_identifier") or qr.get("title", ""),
            "topic": qr.get("canonical_topic", ""),
            "start": qr.get("resolved_start_page", ""), "end": qr.get("resolved_end_page", ""),
        }
        for rp in d.get("rendered_pages", []) or []:
            ih = HEX.search(os.path.basename(rp.get("image_path", "") or ""))
            if ih and rp.get("pdf_page") is not None:
                direct[ih.group(1)] = rp["pdf_page"]

    # ---- pass 2: enumerate every crop image (the read photos) ----
    crop = {}                # imghash -> (dochash, table_index, tdir)
    for png in glob.glob(os.path.join(args.root, "*", "*.png")):
        b = os.path.basename(png)
        ih = HEX.search(b)
        m = DT.search(png)
        if ih and m:
            crop[ih.group(1)] = (m.group(1), m.group(2), os.path.basename(os.path.dirname(png)))
    print(f"crops found: {len(crop)} | direct (metadata) page numbers: {sum(1 for ih in crop if ih in direct)}")

    # ---- pass 3: hash-reversal for crops the metadata didn't cover ----
    recovered = {}
    if not args.no_reverse:
        # group uncovered crops by their table dir's source PDF
        need_by_pdf = defaultdict(set)
        for ih, (dh, ti, tdir) in crop.items():
            if ih in direct:
                continue
            src = tdir_meta.get(tdir, {}).get("source_file", "")
            if src:
                need_by_pdf[src].add(ih)
        print(f"hash-reversal: {sum(len(v) for v in need_by_pdf.values())} crops across {len(need_by_pdf)} PDFs ...")
        for pdf, want in need_by_pdf.items():
            n = page_count(pdf)
            if not n:
                continue
            remaining = set(want)
            for p in range(1, n + 1):
                h = hashlib.sha1(f"{pdf}:{p}".encode("utf-8")).hexdigest()
                if h in remaining:
                    recovered[h] = p
                    remaining.discard(h)
                    if not remaining:
                        break
        print(f"  recovered via hash-reversal: {len(recovered)}")

    # ---- write the master CSV ----
    cols = ["imghash", "dochash", "table_index", "pdf_page", "page_source",
            "source_pdf", "collection", "year", "province", "title",
            "table_start_page", "table_end_page"]
    rows = []
    n_direct = n_rev = n_anchor = n_none = 0
    for ih, (dh, ti, tdir) in crop.items():
        meta = tdir_meta.get(tdir, {})
        if ih in direct:
            pp, psrc = direct[ih], "direct"; n_direct += 1
        elif ih in recovered:
            pp, psrc = recovered[ih], "hash_reversal"; n_rev += 1
        elif str(meta.get("start", "")).isdigit():
            pp, psrc = int(meta["start"]), "table_anchor"; n_anchor += 1
        else:
            pp, psrc = "", "unknown"; n_none += 1
        src = meta.get("source_file", "")
        rows.append({"imghash": ih, "dochash": dh, "table_index": ti, "pdf_page": pp,
                     "page_source": psrc, "source_pdf": src, "collection": collection_of(src),
                     "year": meta.get("year", ""), "province": meta.get("province", ""),
                     "title": (meta.get("title", "") or "")[:120],
                     "table_start_page": meta.get("start", ""), "table_end_page": meta.get("end", "")})
    rows.sort(key=lambda r: (str(r["dochash"]), r["pdf_page"] if isinstance(r["pdf_page"], int) else 1 << 30))
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader(); [w.writerow(r) for r in rows]
    print(f"\nwrote {args.out}: {len(rows)} crops")
    print(f"  page number: direct {n_direct} | hash-reversal {n_rev} | table-anchor {n_anchor} | NONE {n_none}")
    print(f"  -> {100*(len(rows)-n_none)//max(1,len(rows))}% of crops have a real PDF page")
    print(f"  distinct volumes: {len({r['dochash'] for r in rows})} | collections: "
          f"{sorted({r['collection'] for r in rows})}")
    print(f"Now: rsync {args.out} back to the laptop's export_final/sendout_final/sendout_v1/ and run ./sync_cluster_manifest.sh")


if __name__ == "__main__":
    main()
