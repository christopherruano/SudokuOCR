#!/usr/bin/env python3
"""Vision check of every published table's COLUMN structure against its scan (Gemini).

Same task and verdict vocabulary as the Claude reviewers (out/column_review_sample.jsonl):
ok | persons_only_missing | columns_missing | columns_invented | columns_shifted |
columns_mislabelled | wrong_scan | unsure.

Scans: the page image embedded in the source PDF (native resolution, ~1400x2300, bytes
passed through unchanged); the site JPEG only when the page has no PDF match.

  python3 atlas/gemini_column_check.py --model gemini-3.1-pro-preview --pages sample  # calibration
  python3 atlas/gemini_column_check.py --model ... --pages all --workers 48           # everything
Output: out/gemini_column_check_<model>.jsonl (resumable; one line per table).
"""
import os, re, sys, csv, json, time, glob, sqlite3, hashlib, argparse, threading, random, collections
from concurrent.futures import ThreadPoolExecutor, as_completed
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, "..")
DB = os.path.join(ROOT, "export_final/sendout_final/sendout_v1/census_explorer.sqlite")
SITE = os.path.join(ROOT, "census_site"); PDFS = "/Volumes/Expansion/census_pdfs"
NATIVE = os.path.join(HERE, "out", "native_scans")
MAX_IMAGES = 8

PROMPT = """You are checking whether a digitized census table has its COLUMNS read correctly from the scanned page(s). Digit-level accuracy does not matter; only column structure does.

Table: "{title}" ({year}, {place}). The scan has {n_images} page(s){shown}.
Digitized: {n_rows} rows, {n_columns} columns. Digitized columns in reading order, as "column group | sub-column  [first: row=value; ...]":
{columns}

Look at the printed table header and body. Compare the printed value columns with the digitized columns, and spot-check that the first values sit under the right printed header. Choose ONE verdict:
- ok: digitized columns correspond to the printed value columns (ignore serial-number / column-number rows and the row-label stub; minor wording/standardization differences are fine; a table legitimately digitized from only part of a page holding several tables is ok if that part's columns are right)
- persons_only_missing: the only problem is that printed Persons/Total/Both-sexes columns were not digitized, while the Males/Females columns are right
- columns_missing: other printed value columns were not digitized
- columns_invented: digitized columns that are not printed on these pages
- columns_shifted: right headers but values sit under the wrong header (offset, males/females swapped)
- columns_mislabelled: right columns but wrong headers (e.g. printed Married/Unmarried/Widowed digitized as persons/males/females)
- wrong_scan: the scan is not the table that was digitized (different table/title/data)
- unsure: illegible or cannot tell
If more pages exist than are shown, judge only what is shown."""

SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["ok", "persons_only_missing", "columns_missing", "columns_invented",
                                               "columns_shifted", "columns_mislabelled", "wrong_scan", "unsure"]},
        "printed_columns": {"type": "string"},
        "problem_columns": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        "note": {"type": "string"},
    },
    "required": ["verdict", "printed_columns", "problem_columns", "confidence", "note"],
}


KEY_NAME = "GEMINI_CREDITS_KEY"     # the only key these scripts may use (user-designated credits key)


def env_key(name=KEY_NAME):
    if name != KEY_NAME:
        raise SystemExit(f"refusing to use {name}: only {KEY_NAME} is authorised for API calls")
    for p in (os.path.join(ROOT, ".env"), "/Volumes/Expansion/SudokuOCR/.env"):
        if os.path.exists(p):
            for l in open(p):
                if l.startswith(name + "="):
                    return l.split("=", 1)[1].strip().strip('"').strip("'")
    return os.environ.get(name)


_pdf_index, _lock = None, threading.Lock()


def pdf_index():
    global _pdf_index
    with _lock:
        if _pdf_index is None:
            pdfs = {}
            for y in os.listdir(PDFS):
                d = os.path.join(PDFS, y)
                if os.path.isdir(d):
                    for f in os.listdir(d):
                        if f.endswith(".pdf") and not f.startswith("._"):
                            pdfs[f] = os.path.join(d, f)
            idx = {}
            for r in csv.DictReader(open(os.path.join(ROOT, "export_final/sendout_final/sendout_v1/page_number_status.csv"))):
                f = os.path.basename(r["source_pdf"])
                if f in pdfs and r["pdf_page"].lstrip("-").isdigit():
                    idx[r["imghash"]] = (pdfs[f], int(r["pdf_page"]))
            _pdf_index = idx
    return _pdf_index


def native_scan(img):
    """bytes + mime of the best available scan for a site image name"""
    import fitz
    h = img.rsplit(".", 1)[0]
    os.makedirs(NATIVE, exist_ok=True)
    for ext, mime in ((".jpg", "image/jpeg"), (".png", "image/png")):
        c = os.path.join(NATIVE, h + ext)
        if os.path.exists(c):
            return open(c, "rb").read(), mime, "native"
    src = pdf_index().get(h)
    if src:
        try:
            doc = fitz.open(src[0]); page = doc.load_page(src[1] - 1)
            ims = page.get_images()
            if len(ims) == 1:                       # a scanned page: one embedded image
                x = doc.extract_image(ims[0][0])
                if x["ext"] in ("jpeg", "jpg", "png") and x["width"] >= 1000:
                    ext = ".png" if x["ext"] == "png" else ".jpg"
                    open(os.path.join(NATIVE, h + ext), "wb").write(x["image"])
                    return x["image"], "image/png" if ext == ".png" else "image/jpeg", "native"
            pix = page.get_pixmap(matrix=fitz.Matrix(220 / 72, 220 / 72), colorspace=fitz.csGRAY)
            b = pix.tobytes("png"); open(os.path.join(NATIVE, h + ".png"), "wb").write(b)
            return b, "image/png", "rendered220"
        except Exception:
            pass
    return open(os.path.join(SITE, "img", img), "rb").read(), "image/jpeg", "site"


def table_items(pages=None):
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    tid_of = {hashlib.md5(t.encode()).hexdigest()[:7]: t for (t,) in con.execute("select table_id from tables")}
    meta = {t: (ti, y, pl) for t, ti, y, pl in con.execute("select table_id,title,year,canonical_name from tables")}
    seen, out = set(), []
    allp = sorted(os.path.basename(f)[:-5] for f in glob.glob(os.path.join(SITE, "tables", "*.html")))
    for p in (pages if pages is not None else allp):
        h = p.rsplit("_", 1)[1]
        if h not in tid_of or h in seen:
            continue
        seen.add(h)
        f = os.path.join(SITE, "tables", p + ".html")
        if not os.path.exists(f):
            continue
        imgs = re.findall(r'src="\.\./img/([^"]+)"', open(f, errors="ignore").read())
        if not imgs:
            continue
        tid = tid_of[h]
        cols, rows = collections.OrderedDict(), set()
        for sec, row, g, s, v in con.execute("select section,row_label_std,column_group_std,sub_column_std,value from facts where table_id=? order by rowid", (tid,)):
            k = f"{g or ''} | {s or ''}"; cols.setdefault(k, [])
            if len(cols[k]) < 3:
                cols[k].append(f"{row}={v:g}" if v is not None else f"{row}=blank")
            rows.add((sec, row))
        ti, y, pl = meta[tid]
        out.append(dict(page=p, title=ti, year=y, place=pl, imgs=imgs, n_rows=len(rows), n_columns=len(cols),
                        columns=[f"{k}  [first: {'; '.join(v)}]" for k, v in list(cols.items())[:120]]))
    return out


def check(client, model, it):
    from google.genai import types
    shown = f", the first {MAX_IMAGES} are shown" if len(it["imgs"]) > MAX_IMAGES else ""
    text = PROMPT.format(title=it["title"], year=it["year"], place=it["place"], n_images=len(it["imgs"]), shown=shown,
                         n_rows=it["n_rows"], n_columns=it["n_columns"], columns="\n".join(it["columns"]))
    parts, srcs = [], []
    for img in it["imgs"][:MAX_IMAGES]:
        b, mime, src = native_scan(img)
        parts.append(types.Part.from_bytes(data=b, mime_type=mime)); srcs.append(src)
    parts.append(types.Part.from_text(text=text))
    cfg = types.GenerateContentConfig(response_mime_type="application/json", response_schema=SCHEMA, temperature=0)
    for attempt in range(8):
        try:
            r = client.models.generate_content(model=model, contents=parts, config=cfg)
            d = json.loads(r.text)
            d.update(page=it["page"], model=model, scan_source=",".join(sorted(set(srcs))))
            return d
        except Exception as e:
            msg = str(e)
            if attempt == 7:
                return dict(page=it["page"], model=model, verdict="error", note=msg[:200])
            time.sleep(min(60, 2 ** attempt + random.random() * 2) if ("429" in msg or "503" in msg or "500" in msg) else 2)


def main():
    from google import genai
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemini-3.1-pro-preview")
    ap.add_argument("--pages", default="sample", help="sample | all | path to a file of page ids")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--key", default="GEMINI_CREDITS_KEY")
    a = ap.parse_args()
    if a.pages == "sample":
        pages = [json.loads(l)["page"] for l in open(os.path.join(HERE, "out", "column_review_sample.jsonl"))]
    elif a.pages == "all":
        pages = None
    else:
        pages = [l.strip() for l in open(a.pages) if l.strip()]
    items = table_items(pages)
    out = os.path.join(HERE, "out", f"gemini_column_check_{a.model}.jsonl")
    done = set()
    if os.path.exists(out):
        done = {d["page"] for d in map(json.loads, open(out)) if d.get("verdict") != "error"}
    todo = [i for i in items if i["page"] not in done]
    print(f"{len(items)} tables, {len(todo)} to check with {a.model}", flush=True)
    client = genai.Client(api_key=env_key(a.key))
    t0, n, c = time.time(), 0, collections.Counter()
    with open(out, "a") as f, ThreadPoolExecutor(a.workers) as ex:
        futs = [ex.submit(check, client, a.model, i) for i in todo]
        for fu in as_completed(futs):
            d = fu.result(); f.write(json.dumps(d) + "\n"); f.flush()
            n += 1; c[d.get("verdict")] += 1
            if n % 100 == 0 or n == len(todo):
                print(f"{n}/{len(todo)} {time.time() - t0:.0f}s {dict(c)}", flush=True)
    print("GEMINI_CHECK_DONE", flush=True)


if __name__ == "__main__":
    main()
