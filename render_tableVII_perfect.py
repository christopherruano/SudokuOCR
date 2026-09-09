#!/usr/bin/env python3
r"""
Render EVERY Imperial Table VII (Age, Sex & Civil Condition) page from the census
PDFs on this drive to 300-DPI PNGs -- by reading each page's OWN header, never a
table-of-contents. Pure Python; works on Windows / macOS / Linux.

WHY content-detection (and not the TOC): the first pass used TOC page ranges and
that is precisely what produced gaps + wrong ranges. This script ignores the TOC
completely. A page is Table VII iff its printed header says so. For image-only
scans (no embedded text) it OCRs just the top band of the page to read the header.

Dependencies:
    pip install pymupdf pillow
    # OCR fallback (only used for pages that have NO embedded text layer):
    pip install rapidocr_onnxruntime        # preferred: pip-only, no system binary
    #   ...or...  pip install pytesseract    # needs the Tesseract program installed

Usage (run it from the drive so output lands on the drive):
    python render_tableVII_perfect.py D:\census_pdfs
    # optional: limit to one year to probe first:  python render_tableVII_perfect.py D:\census_pdfs\1901

Output (folder "tableVII_renders" in the current directory):
    <key>_p<page>.png            one 300-DPI PNG per detected Table VII page
    render_manifest.csv          one row per PNG (the map back to the main machine)
    render_log.txt               per-volume summary (pages, hits, text-layer, OCR use)
    needs_attention.txt          volumes that need OCR but no OCR engine was available
It is RESUMABLE: re-run after an interruption and it skips finished volumes.
"""
import os, re, sys, csv, json, time

try:
    import fitz  # PyMuPDF
except ImportError:
    sys.exit("Need PyMuPDF.  Run:  pip install pymupdf pillow   then re-run this script.")

ARGV  = sys.argv[1:]
PROBE = "--probe" in ARGV                       # diagnose-only: detect + print, render nothing
LIMIT = None
if "--limit" in ARGV:
    LIMIT = int(ARGV[ARGV.index("--limit") + 1])
POS  = [a for a in ARGV if not a.startswith("--") and not a.isdigit()]
ROOT = POS[0] if POS else "."
OUT  = os.path.abspath(os.environ.get("TVII_OUT", "tableVII_renders"))
os.makedirs(OUT, exist_ok=True)
RENDER_DPI = 300       # full-page render quality (good for re-OCR of the table)
HEADER_DPI = 170       # header-band OCR quality (only for text-less pages)
HEADER_FRAC = 0.30     # top 30% of the page = the header region we read
GAP = 6                # VII pages within this many pages = same table block
EXT = 4                # include up to this many unlabeled continuation pages each side

# ---- header patterns -------------------------------------------------------
VII = [re.compile(p, re.I) for p in [
    r"TABLE\s*VII\b",                                  # \b stops it matching TABLE VIII
    r"AGE,?\s*SEX\s*(?:AND|&)?\s*CIVIL\s*CONDITION",
    r"CIVIL\s*CONDITION\s*(?:BY|AND)\s*AGE",
    r"AGE\s*AND\s*CIVIL\s*CONDITION",
    r"\bC\s*[-–.\s]?\s*III\b",                    # post-1931 "C-III" age/marital table
]]
OTHER_TABLE = re.compile(r"\b(?:SUBSIDIARY\s+TABLE|IMPERIAL\s+TABLE|TABLE)\s+([IVXLC]+)\b", re.I)
OTHER_C     = re.compile(r"\bC\s*[-–.\s]?\s*(I|II|IV|V|VI|VIII|IX|X)\b", re.I)

def classify(text):
    """Return 'VII' | 'OTHER' | 'CONT' for a page from its header text."""
    h = text[:1800]
    if any(p.search(h) for p in VII):
        return "VII"
    m = OTHER_TABLE.search(h)
    if m and m.group(1).upper() != "VII":
        return "OTHER"
    if OTHER_C.search(h):
        return "OTHER"
    return "CONT"

# ---- OCR engine (lazy; only for text-less pages) ---------------------------
_OCR = {"kind": None, "obj": None, "ready": False}
def ocr_ready():
    if _OCR["ready"]:
        return _OCR["kind"]
    _OCR["ready"] = True
    try:
        from rapidocr_onnxruntime import RapidOCR
        _OCR.update(kind="rapid", obj=RapidOCR())
        return "rapid"
    except Exception:
        pass
    try:
        import pytesseract  # noqa
        _OCR.update(kind="tess", obj=pytesseract)
        return "tess"
    except Exception:
        pass
    _OCR.update(kind=None)
    return None

def ocr_header(page):
    from PIL import Image
    r = page.rect
    band = fitz.Rect(r.x0, r.y0, r.x1, r.y0 + r.height * HEADER_FRAC)
    pix = page.get_pixmap(matrix=fitz.Matrix(HEADER_DPI/72, HEADER_DPI/72), clip=band)
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    if _OCR["kind"] == "rapid":
        import numpy as np
        res, _ = _OCR["obj"](np.array(img))
        return " ".join(t[1] for t in res) if res else ""
    if _OCR["kind"] == "tess":
        return _OCR["obj"].image_to_string(img)
    return ""

# ---- file discovery --------------------------------------------------------
def base_key(fn):
    b = os.path.splitext(os.path.basename(fn))[0]
    for suf in ("_text", "_bw", "_encrypted"):
        if b.lower().endswith(suf):
            b = b[: -len(suf)]
    return b

def year_of(path):
    rel = os.path.relpath(path, ROOT)
    top = rel.split(os.sep)[0]
    m = re.search(r"(18|19|20)\d\d", top)        # top-level folder = the census-year bucket
    return m.group(0) if m else ""

def is_core(path):
    """Enable OCR-fallback only for the Imperial series (saoa.crl.* / 1871-1951)."""
    if re.search(r"saoa\.crl\.", os.path.basename(path), re.I):
        return True
    y = year_of(path)
    return y.isdigit() and 1871 <= int(y) <= 1951

# prefer the *_text.pdf twin (it has a real text layer) when both exist
pdfs = {}
for dp, _, fns in os.walk(ROOT):
    for fn in fns:
        if not fn.lower().endswith(".pdf"):
            continue
        key = base_key(fn)
        full = os.path.join(dp, fn)
        if key not in pdfs or fn.lower().endswith("_text.pdf"):
            pdfs[key] = full

# ---- resume state ----------------------------------------------------------
done_path = os.path.join(OUT, "_done.json")
done = set(json.load(open(done_path)) if os.path.exists(done_path) else [])
man_path = os.path.join(OUT, "render_manifest.csv")
log_path = os.path.join(OUT, "render_log.txt")
att_path = os.path.join(OUT, "needs_attention.txt")
COLS = ["filename","saoa_id","pdf_key","pdf_page","block_id","source_pdf","year","detect","header"]
if not os.path.exists(man_path):
    with open(man_path, "w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=COLS).writeheader()

def log(line):
    print(line, flush=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(line + "\n")

mode = "PROBE (diagnose only, no rendering)" if PROBE else "RENDER"
log(f"\n=== run start [{mode}]: {len(pdfs)} PDFs under {ROOT} | output -> {OUT} ===")

# ---- main loop -------------------------------------------------------------
total_pngs = 0
processed = 0
for key in sorted(pdfs):
    path = pdfs[key]
    if not PROBE and key in done:
        continue
    if LIMIT is not None and processed >= LIMIT:
        break
    processed += 1
    sid_m = re.search(r"(\d{6,})", os.path.basename(path))
    saoa  = sid_m.group(1) if sid_m else key
    year  = year_of(path)
    core  = is_core(path)
    try:
        doc = fitz.open(path)
        if doc.needs_pass:
            doc.authenticate("")  # try empty password for "encrypted" archive scans
        n = doc.page_count
    except Exception as e:
        log(f"SKIP open-failed  {key}: {e}")
        done.add(key); json.dump(sorted(done), open(done_path, "w")); continue

    labels = []            # per page: VII / OTHER / CONT
    detect = []            # per page: how header was read (embedded / ocr / none)
    headers = []           # short header snippet per page (for verification)
    any_embed = False
    n_ocr = 0
    for i in range(n):
        pg = doc.load_page(i)
        t = pg.get_text("text") or ""
        if t.strip():
            any_embed = True
            d = "embedded"
        elif core and ocr_ready():
            t = ocr_header(pg); d = "ocr"; n_ocr += 1
        else:
            d = "none"
        labels.append(classify(t))
        detect.append(d)
        headers.append(re.sub(r"\s+", " ", t[:90]).strip())

    # build Table VII blocks from VII pages: cluster (gap<=GAP), extend over CONT (walls=OTHER)
    vii = [i for i, l in enumerate(labels) if l == "VII"]
    blocks = []
    if vii:
        cl = [[vii[0]]]
        for x in vii[1:]:
            (cl[-1].append(x) if x - cl[-1][-1] <= GAP else cl.append([x]))
        for c in cl:
            a, b = c[0], c[-1]
            k = 0
            while a - 1 >= 0 and labels[a-1] != "OTHER" and k < EXT:
                a -= 1; k += 1
            k = 0
            while b + 1 < n and labels[b+1] != "OTHER" and k < EXT:
                b += 1; k += 1
            blocks.append((a, b))

    tl = "embed" if any_embed else ("ocr" if n_ocr else "NONE")
    if PROBE:
        # diagnose only: show what we'd render + the actual headers, so we can verify
        # the detector catches real Table VII pages BEFORE committing to a full render.
        ranges = ", ".join(f"p{a+1}-p{b+1}" for a, b in blocks) or "(none found)"
        log(f"\n{saoa} [{year}] pages={n} text={tl} core={int(core)} "
            f"VII_pages={len(vii)} blocks={len(blocks)}  ->  {ranges}")
        for bi, (a, b) in enumerate(blocks):
            log(f"    block {bi}: p{a+1}-p{b+1}  first-header: \"{headers[a]}\"")
        if vii:
            log(f"    sample VII header: \"{headers[vii[0]]}\"")
        # a couple of non-VII headers so we can see the page-header style for this year
        others = [i for i, l in enumerate(labels) if l != "CONT"][:0] or \
                 [i for i in range(min(n, 40)) if headers[i]][:2]
        for i in others:
            log(f"    sample page header [{labels[i]}] p{i+1}: \"{headers[i]}\"")
        doc.close()
        continue

    # render
    rows = []
    for bi, (a, b) in enumerate(blocks):
        for i in range(a, b + 1):
            p = i + 1
            png = os.path.join(OUT, f"{key}_p{p}.png")
            if not os.path.exists(png):
                try:
                    doc.load_page(i).get_pixmap(matrix=fitz.Matrix(RENDER_DPI/72, RENDER_DPI/72)).save(png)
                except Exception as e:
                    log(f"  render-fail {key} p{p}: {e}"); continue
            rows.append({"filename": os.path.basename(png), "saoa_id": saoa, "pdf_key": key,
                         "pdf_page": p, "block_id": bi, "source_pdf": os.path.basename(path),
                         "year": year, "detect": detect[i], "header": headers[i]})
    if rows:
        with open(man_path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=COLS)
            for r in rows:
                w.writerow(r)
    total_pngs += len(rows)

    # flag image-only core volumes we could not detect (don't silently fall back to TOC!)
    if core and not any_embed and not ocr_ready():
        with open(att_path, "a", encoding="utf-8") as f:
            f.write(f"{saoa}\t{path}\tNO TEXT LAYER and no OCR engine installed\n")

    log(f"{saoa} [{year}] pages={n} VII_pages={len(vii)} blocks={len(blocks)} "
        f"rendered={len(rows)} text={tl} core={int(core)}")
    doc.close()
    done.add(key); json.dump(sorted(done), open(done_path, "w"))

log(f"=== DONE: rendered {total_pngs} pages this run; "
    f"manifest -> {man_path} ===")
print(f"\nCarry the whole folder back to the main machine:\n  {OUT}\n"
      f"(PNGs + render_manifest.csv). If needs_attention.txt is non-empty, tell me — "
      f"those volumes need the OCR engine installed (pip install rapidocr_onnxruntime).")
