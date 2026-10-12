#!/usr/bin/env python3
"""Find the PDF pages a table's numbers were actually read from (for withdrawn wrong-scan tables and
tables noted for missing facing pages), by searching the source volume on the drive.

For each target table: its distinctive numbers (>= 3 significant digits, distinct) from the digitized
data; every page within +/-WINDOW of the linked page in the same PDF, read via the PDF text layer when
it has one, else Tesseract digits (atlas/grounding.py); a greedy cover picks the pages holding most of
the table's numbers. Local only, no API.

  python3 atlas/find_pages.py            -> out/found_pages.jsonl (resumable)
"""
import os, re, sys, json, sqlite3, hashlib, collections, csv
from concurrent.futures import ProcessPoolExecutor
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
ROOT = os.path.join(HERE, "..")
import gemini_column_check as g
V1 = os.path.join(ROOT, "export_final/sendout_final/sendout_v1")
CACHE = os.path.join(HERE, "out", "pdf_page_numbers.jsonl")
OUT = os.path.join(HERE, "out", "found_pages.jsonl")
WINDOW = int(os.environ.get("FP_WINDOW", "12"))


def digits(t):
    return re.sub(r"\D", "", str(t)).lstrip("0")


def sig(v):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    d = digits(("%f" % x).rstrip("0").rstrip("."))
    return d if len(d) >= 3 else None


def page_numbers(job):
    """numbers printed on one PDF page: text layer if usable, else Tesseract digits"""
    pdf, pg = job
    import fitz
    from PIL import Image
    import grounding
    doc = fitz.open(pdf)
    if not (1 <= pg <= doc.page_count):
        return pdf, pg, []
    page = doc.load_page(pg - 1)
    toks = [digits(t) for t in re.findall(r"\d[\d,.·]*", page.get_text())]
    toks = [t for t in toks if len(t) >= 2]
    src = "text"
    if len(toks) < 20:
        pix = page.get_pixmap(matrix=fitz.Matrix(200 / 72, 200 / 72), colorspace=fitz.csGRAY)
        tmp = f"/tmp/_fp_{os.getpid()}.png"
        Image.frombytes("L", (pix.width, pix.height), pix.samples).save(tmp)
        toks = [digits(k) for k in grounding.page_numbers(tmp)]
        toks = [t for t in toks if len(t) >= 2]
        src = "ocr"
    return pdf, pg, sorted(set(toks)), src


def targets():
    log = list(csv.DictReader(open(os.path.join(HERE, "out", "column_fix_log.csv"))))
    ov = json.load(open(os.path.join(V1, "column_fix_overlay.json")))
    tids = set(ov["withdrawn"]) | {r["table_id"] for r in log if r["decision"] in ("reject", "held", "error")}
    return tids


def main():
    con = sqlite3.connect(f"file:{g.DB}?mode=ro", uri=True)
    tids = targets()
    idx = g.pdf_index()
    jobs, tables = set(), {}
    for tid in tids:
        r = con.execute("select member_images, source_image from tables where table_id=?", (tid,)).fetchone()
        if not r:
            continue
        hs = re.findall(r"([0-9a-f]{40})", (r[0] or "") + " " + (r[1] or "") + " " + tid)
        srcs = [idx[h] for h in hs if h in idx]
        if not srcs:
            continue
        pdf = srcs[0][0]; pages = sorted({p for f, p in srcs if f == pdf})
        nums = {sig(v) for (v,) in con.execute("select value from facts where table_id=?", (tid,))} - {None}
        if len(nums) < 5:
            continue
        tables[tid] = (pdf, pages, nums)
        for p in range(max(1, pages[0] - WINDOW), pages[-1] + WINDOW + 1):
            jobs.add((pdf, p))
    have = {}
    if os.path.exists(CACHE):
        for l in open(CACHE):
            d = json.loads(l); have[(d["pdf"], d["page"])] = set(d["nums"])
    todo = sorted(j for j in jobs if j not in have)
    print(f"{len(tables)} target tables; {len(jobs)} pages to search ({len(todo)} not yet read)", flush=True)
    with open(CACHE, "a") as fh, ProcessPoolExecutor(max(1, os.cpu_count() - 1)) as ex:
        for k, res in enumerate(ex.map(page_numbers, todo, chunksize=4)):
            pdf, pg, toks = res[0], res[1], res[2]
            have[(pdf, pg)] = set(toks)
            fh.write(json.dumps({"pdf": pdf, "page": pg, "nums": toks, "src": res[3] if len(res) > 3 else ""}) + "\n")
            if k % 500 == 0:
                print(f"  read {k}/{len(todo)}", flush=True); fh.flush()
    out = []
    for tid, (pdf, linked, nums) in tables.items():
        cand = {p: have.get((pdf, p), set()) for p in range(max(1, linked[0] - WINDOW), linked[-1] + WINDOW + 1)}
        def hit(pageset, n):
            return n in pageset or (len(n) >= 4 and any(n in t for t in pageset if len(t) >= 7))
        cover, chosen, left = 0, [], set(nums)
        for _ in range(8):
            best = max(cand, key=lambda p: sum(hit(cand[p], n) for n in left)) if cand else None
            if best is None:
                break
            gain = sum(hit(cand[best], n) for n in left)
            if gain < max(3, 0.08 * len(nums)):
                break
            chosen.append(best); left = {n for n in left if not hit(cand[best], n)}; cand.pop(best)
        cov = 1 - len(left) / len(nums)
        lin_cov = sum(1 for n in nums if any(hit(have.get((pdf, p), set()), n) for p in linked)) / len(nums)
        out.append({"table_id": tid, "pdf": pdf, "linked_pages": linked, "found_pages": sorted(chosen),
                    "coverage": round(cov, 3), "linked_coverage": round(lin_cov, 3), "n_numbers": len(nums)})
    with open(OUT, "w") as fh:
        for r in out:
            fh.write(json.dumps(r) + "\n")
    c = collections.Counter("found" if r["coverage"] >= 0.6 and r["coverage"] > r["linked_coverage"] + 0.2 else
                            "linked already ok" if r["linked_coverage"] >= 0.6 else "not found" for r in out)
    print(dict(c), flush=True)
    print("FIND_DONE", flush=True)


if __name__ == "__main__":
    main()
