#!/usr/bin/env python3
"""Mistral fallback for the ~211 NON-Table-VII frozen-column tables (Education/Literacy by
Religion&Age, Civil Condition by Religion, Occupations, Subsidiary, Infirmities, ...). The
Table VII civil-condition converter (parse_mistral_md) can't read these layouts, so this uses
the GENERIC markdown->oneshot converter (parse_mistral_generic) which infers the column
groups/subkeys from the markdown's multi-row header.

For each imghash in export/frozen_nonvii.txt:
  render (reuse 300dpi cache, else from drive) -> Mistral-OCR (cached md) -> generic parse ->
  for EACH currently-frozen corpus copy of that imghash: REPLACE its read ONLY IF the new read
  removes frozen columns for that copy AND is a sane grid (>=4 rows, >=3 group/subkey columns,
  >=12 numeric cells). Backs up each replaced copy to *_oneshot.prefrozen.json (never clobbers
  an existing backup). VII arithmetic identities (U+M+W=Population) do NOT apply here, so they
  are intentionally not gated on.

  python3 mistral_frozen_generic.py [--limit N] [--apply] [--workers K]
"""
import sys, os, re, glob, json, csv, shutil
from concurrent.futures import ThreadPoolExecutor, as_completed

_apply = "--apply" in sys.argv
_limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else 0
_workers = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 2
sys.argv = ["x"]
import run_mistral_ocr as RM
import parse_mistral_generic as G
import detect_misfires as MF
import mistral_frozen as MZ          # reuse ensure_render / page_loc / ENV
from dotenv import dotenv_values

ROOT = os.path.dirname(os.path.abspath(__file__))
ENV = dotenv_values(os.path.join(ROOT, ".env"))
CORPUS = os.path.join(ROOT, "export/ocr_results_referenced/digitized_tables")
WORKLIST = os.path.join(ROOT, "export/frozen_nonvii.txt")
VERIFY = os.path.join(ROOT, "export/frozen_nonvii_fixes.csv")


def frozen_count(d):
    n = 0
    for s in d.get("data", []):
        rows = s.get("rows", [])
        if not rows:
            continue
        rh = next((k for k in rows[0] if not isinstance(rows[0].get(k), dict)), None)
        n += len(MF.section_misfires(rows, rh))
    return n


def grid_stats(d):
    """(#rows, #distinct group/subkey columns, #numeric cells)."""
    nrows = ncells = 0
    cols = set()
    for s in d.get("data", []):
        for r in s.get("rows", []):
            nrows += 1
            for g, sub in r.items():
                if isinstance(sub, dict):
                    for k, v in sub.items():
                        if isinstance(v, (int, float)):
                            cols.add(f"{g}/{k}"); ncells += 1
    return nrows, len(cols), ncells


def copies_of(ih):
    return glob.glob(os.path.join(CORPUS, "*", f"{ih}_oneshot.json"))


def rowkey_of(d):
    for s in d.get("data", []):
        if s.get("rows"):
            return next((k for k in s["rows"][0] if not isinstance(s["rows"][0].get(k), dict)), None)
    return None


def main():
    ihs = [x.strip() for x in open(WORKLIST) if x.strip()]
    if _limit:
        ihs = ihs[:_limit]
    print(f"non-VII frozen worklist: {len(ihs)} imghashes  ({'APPLY' if _apply else 'DRY-RUN'})")

    # --- Phase 1: ensure 300dpi renders (serial; only a few are missing) ---
    renders = {}
    miss = 0
    for ih in ihs:
        rp = MZ.ensure_render(ih)
        renders[ih] = rp
        if not rp:
            miss += 1
    print(f"renders ready: {sum(1 for v in renders.values() if v)}/{len(ihs)}  (no-render: {miss})")

    # --- Phase 2: Mistral-OCR -> markdown (cached; small concurrency) ---
    from mistralai import Mistral
    client = Mistral(api_key=ENV.get("MISTRAL_API_KEY"))
    os.makedirs(RM.OUT, exist_ok=True)
    todo = [(ih, renders[ih]) for ih in ihs if renders.get(ih)]
    ocr_done = ocr_err = 0
    with ThreadPoolExecutor(max_workers=_workers) as ex:
        futs = {ex.submit(RM.ocr_one, client, ih, rp): ih for ih, rp in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            ih, st = fut.result()
            if st.startswith("ERR"):
                ocr_err += 1
                print(f"  ocr ERR {ih[:12]}: {st}")
            else:
                ocr_done += 1
            if i % 25 == 0:
                print(f"  ...ocr {i}/{len(todo)} (ok={ocr_done} err={ocr_err})")
    print(f"mistral ocr: ok/cached {ocr_done}, err {ocr_err}")

    # --- Phase 3: parse + gate + apply (serial) ---
    fixed_ih = badparse = norender = noimprove = parseerr = 0
    copies_fixed = 0
    vrows = []
    for ih in ihs:
      try:
        rp = renders.get(ih)
        md = os.path.join(RM.OUT, f"{ih}.mistral.md")
        copies = copies_of(ih)
        if not copies:
            continue
        # frozen, non-human copies
        frz = []
        for cp in copies:
            try:
                d = json.load(open(cp))
            except Exception:
                continue
            if (d.get("metadata", {}) or {}).get("human_corrected"):
                continue
            of = frozen_count(d)
            if of > 0:
                frz.append((cp, d, of))
        if not frz:
            continue
        if not rp:
            norender += 1; continue
        if not os.path.exists(md):
            badparse += 1; continue
        rowkey = rowkey_of(frz[0][1]) or "row"
        secs = G.parse_sections(md, rowkey=rowkey)
        title0 = (frz[0][1].get("metadata", {}) or {}).get("title", "")
        new = G.to_oneshot(secs, title0, frz[0][1].get("source_image", ""))
        nf = frozen_count(new)
        nrows, ncols, ncells = grid_stats(new)
        sane = nrows >= 4 and ncols >= 3 and ncells >= 12
        if not sane:
            badparse += 1
            print(f"  {ih[:12]} BAD-PARSE nf={nf} rows={nrows} cols={ncols} cells={ncells}")
            continue
        any_fix = False
        for cp, d, of in frz:
            if nf >= of:
                continue
            any_fix = True
            tdir = os.path.basename(os.path.dirname(cp))
            vrows.append([ih, tdir, title0[:50], of, nf, nrows, ncols, ncells])
            if _apply:
                bak = cp.replace("_oneshot.json", "_oneshot.prefrozen.json")
                if not os.path.exists(bak):
                    shutil.copy(cp, bak)
                out = dict(new)
                out["source_image"] = d.get("source_image", "")
                out["metadata"] = dict(new["metadata"])
                out["metadata"]["title"] = (d.get("metadata", {}) or {}).get("title", title0)
                json.dump(out, open(cp, "w"))
            copies_fixed += 1
        if any_fix:
            fixed_ih += 1
            print(f"  {ih[:12]} FIX frozen->{nf}  rows={nrows} cols={ncols}  copies={len([1 for cp,d,of in frz if of>nf])}  {title0[:42]}")
        else:
            noimprove += 1
      except Exception as e:
        parseerr += 1
        print(f"  {ih[:12]} PARSE-ERR {type(e).__name__}: {str(e)[:80]}")

    with open(VERIFY, "w", newline="") as fo:
        w = csv.writer(fo)
        w.writerow(["imghash", "table_dir", "title", "old_frozen", "new_frozen", "new_rows", "new_cols", "new_cells"])
        w.writerows(vrows)

    print(f"\n{'APPLIED' if _apply else 'DRY-RUN'}: imghashes fixed {fixed_ih} ({copies_fixed} corpus copies) | "
          f"no-improve {noimprove} | bad-parse {badparse} | no-render {norender} | parse-err {parseerr}")
    print(f"verify CSV -> {VERIFY}")


if __name__ == "__main__":
    main()
