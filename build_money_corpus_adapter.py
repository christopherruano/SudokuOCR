#!/usr/bin/env python3
"""Adapter: rewrite the flat money OCR (saoa,page) results into the corpus structure
   export/ocr_results_referenced/digitized_tables/{dochash}_t{NNNN}_{label}/{imghash}_oneshot.json
   + the scan PNG as {imghash}.png alongside, so the ORIGINAL reingest pipeline ingests
them exactly like the existing tables (table_id = {dochash}_t{NNNN}; provenance from the
cluster path + OCR metadata). Idempotent. --limit N for a test batch.
"""
import json, csv, re, os, glob, hashlib, sys, shutil

V1 = "export_final/sendout_final/sendout_v1"
DEST = "export/ocr_results_referenced/digitized_tables"
SRC_DIRS = ["export/money_renders_ocr", "export/money_gapfill_ocr",
            "export/flashdrive_ocr",
            "export/documents_ocr_results",   # /documents portal: human-confirmed Table VII ranges
            "export/tableVII_final_gaps_ocr",
            "export/boundary_gap_ocr",
            "export/boundary_grow_ocr",
            "export/empty_vii_reocr",
            "export_final/gemini_no_gap_ocr_2026-06-16_v2_best"]
CORR = "export/money_gapfill_ocr/corrected"
LIMIT = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None
# AI-confirmed Table VII continuation/divider pages (from the boundary fan-out): fold these even when
# they read 0-row, so the run detector can include them and the runs reach their AI-judged true extent.
CONFIRMED = set()
if os.path.exists("export/confirmed_vii_pages.json"):
    for _c in json.load(open("export/confirmed_vii_pages.json")):
        CONFIRMED.add((str(_c["saoa"]), int(_c["page"])))


def slug(s):
    return (re.sub(r"[^A-Za-z0-9]+", "-", s or "").strip("-")[:60]) or "table"


def main():
    cp_of = {}
    for r in csv.DictReader(open(f"{V1}/cc_volumes.csv")):
        if r.get("cluster_path"):
            cp_of[r["saoa_id"]] = r["cluster_path"]
    # next free table index per dochash (avoid colliding with existing tNNNN dirs)
    nextidx = {}
    DT = re.compile(r"([0-9a-f]{16})_t(\d+)")
    for d in glob.glob(f"{DEST}/*"):
        m = DT.search(os.path.basename(d))
        if m:
            nextidx[m.group(1)] = max(nextidx.get(m.group(1), 0), int(m.group(2)) + 1)
    # IDEMPOTENCY: skip pages whose imghash is ALREADY anywhere in the corpus. Without this, every re-run
    # created a fresh tNNNN dir per page -> the 8x duplicate-folder explosion (59GB) we just had to move.
    EXISTING_IH = set()
    for f in glob.glob(f"{DEST}/*/*_oneshot.json"):
        mm = re.search(r"([0-9a-f]{40})_oneshot", os.path.basename(f))
        if mm:
            EXISTING_IH.add(mm.group(1))

    def scan_for(sa, pg):
        for cand in (os.path.expanduser(f"~/Downloads/core_money_renders/*/{sa}/{sa}_p{pg}.png"),
                     os.path.expanduser(f"~/Downloads/final_tableVII/{sa}_p{pg}.png"),
                     f"export/flashdrive_renders/{sa}_p{pg}.png",
                     f"export/tableVII_final_gaps/{sa}_p{pg}.png",
                     f"export/boundary_gap_renders/{sa}_p{pg}.png",
                     f"export/boundary_grow_renders/{sa}_p{pg}.png",
                     f"export/empty_vii_renders/{sa}_p{pg}.png"):
            hit = glob.glob(cand)
            if hit:
                return hit[0]
        return ""

    seen = set(); written = 0; skipped = 0; noscan = 0
    for sd in SRC_DIRS:
        for f in sorted(glob.glob(f"{sd}/**/*_oneshot.json", recursive=True)):
            m = re.search(r"(\d+)_p(\d+)", os.path.basename(f))
            if not m:
                continue
            sa, pg = m.group(1), int(m.group(2))
            if (sa, pg) in seen:
                continue
            seen.add((sa, pg))
            cp = cp_of.get(sa)
            if not cp:
                continue
            # prefer human-corrected copy
            corr = os.path.join(CORR, f"{sa}_p{pg}_oneshot.json")
            srcf = corr if os.path.exists(corr) else f
            try:
                d = json.load(open(srcf))
            except Exception:
                continue
            nrows = sum(len(s.get("rows", [])) for s in d.get("data", []))
            if nrows == 0 and (sa, pg) not in CONFIRMED:
                skipped += 1; continue          # don't import empties/prose into the corpus
                                                # (but DO import AI-confirmed VII continuation pages)
            dochash = hashlib.sha1(cp.encode()).hexdigest()[:16]
            imghash = hashlib.sha1(f"{cp}:{pg}".encode()).hexdigest()
            if imghash in EXISTING_IH:        # already folded once — don't create a duplicate folder
                continue
            md = d.get("metadata", {}) or {}
            title = md.get("title") or (d.get("data", [{}])[0].get("name", "")) or f"page {pg}"
            idx = nextidx.get(dochash, 1); nextidx[dochash] = idx + 1
            tdir = os.path.join(DEST, f"{dochash}_t{idx:04d}_{slug(title)}")
            outjson = os.path.join(tdir, f"{imghash}_oneshot.json")
            if os.path.exists(outjson):
                continue
            # place scan as {imghash}.png so patch_scan_paths/static site can find it.
            # Prefer the OCR JSON's own source_image (absolute render path) when it exists.
            orig_si = d.get("source_image")
            sp = orig_si if (orig_si and os.path.exists(orig_si)) else scan_for(sa, pg)
            os.makedirs(tdir, exist_ok=True)
            # SUDOKUOCR_ADAPTER_NOIMG=1 -> fold the JSON read only, DON'T copy the source image into the
            # corpus. The crawl/detector only need the JSON; the image copies are for the deploy reingest
            # (blocked) and were filling the disk (Errno 28). source_image still points at the real render.
            if sp and not os.environ.get("SUDOKUOCR_ADAPTER_NOIMG"):
                shutil.copy(sp, os.path.join(tdir, f"{imghash}.png"))
            elif not sp:
                noscan += 1
            d.setdefault("source_image", os.path.join(tdir, f"{imghash}.png"))
            json.dump(d, open(outjson, "w"))
            written += 1
            if LIMIT and written >= LIMIT:
                print(f"[--limit {LIMIT}] stopping early")
                print(f"wrote {written} tables | skipped empties {skipped} | no-scan {noscan}")
                return
    print(f"adapted money tables into corpus: {written} | skipped empties {skipped} | no-scan {noscan}")
    print(f"-> {DEST}  (table_id = dochash_tNNNN; ingest via reingest.sh)")


if __name__ == "__main__":
    main()
