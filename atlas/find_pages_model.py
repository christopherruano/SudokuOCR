#!/usr/bin/env python3
"""Model search for the pages a withdrawn table was read from (for the tables find_pages.py could not
place: most of these volumes have no text layer and local OCR misses their numbers).

Gemini 3.8 Flash (credits key only) is shown the pages within +/-WINDOW of the table's linked scan in
the same PDF, in labelled chunks, with the table's title, columns and first values, and names the pages
whose printed data match. Matches are merged into out/found_pages.jsonl (method "model") so that
verify_found_pages.py / reread_found.py / apply_column_fixes.py treat them like the local matches.

  python3 atlas/find_pages_model.py    -> out/found_pages_model.jsonl (resumable), merged into found_pages.jsonl
"""
import os, re, sys, json, time, random, sqlite3, collections
from concurrent.futures import ThreadPoolExecutor, as_completed
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import gemini_column_check as g
from find_right_scan import render
from verify_found_pages import item
import find_pages as F

MODEL = "gemini-3.8-flash"
WINDOW, CHUNK = 12, 9
OUT = os.path.join(HERE, "out", "found_pages_model.jsonl")
PROMPT = """Below are consecutive scanned pages from one census volume, each preceded by its label. A digitized table's data was read from one or more pages of this volume, possibly among these. Name every page shown whose printed data is this table: the title or column headers must correspond AND the listed first values must actually be printed there (one page can hold the left half of a wide table and the next page the right half; list both).

Table: "{title}" ({year}, {place}); {n_rows} rows, {n_columns} columns.
Digitized columns with first values:
{columns}

Return the matching page labels (empty if none of these pages holds this table's data)."""
SCHEMA = {"type": "object", "properties": {
    "pages": {"type": "array", "items": {"type": "integer"}},
    "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
    "note": {"type": "string"}}, "required": ["pages", "confidence", "note"]}


def small(pdf, p):
    """page image downscaled for the search (long side 1800 px, JPEG); reading uses full resolution"""
    import io
    from PIL import Image
    r = render(pdf, p)
    if not r:
        return None
    im = Image.open(io.BytesIO(r[0])).convert("L"); im.thumbnail((1800, 1800))
    buf = io.BytesIO(); im.save(buf, "JPEG", quality=80)
    return buf.getvalue(), "image/jpeg"


def ask(client, it, pdf, pages):
    from google.genai import types
    parts = []
    for p in pages:
        r = small(pdf, p)
        if r:
            parts += [types.Part.from_text(text=f"Page {p}:"), types.Part.from_bytes(data=r[0], mime_type=r[1])]
    parts.append(types.Part.from_text(text=PROMPT.format(title=it["title"], year=it["year"], place=it["place"],
                 n_rows=it["n_rows"], n_columns=it["n_columns"], columns="\n".join(it["columns"][:40]))))
    cfg = types.GenerateContentConfig(response_mime_type="application/json", response_schema=SCHEMA, temperature=0)
    for attempt in range(8):
        try:
            d = json.loads(client.models.generate_content(model=MODEL, contents=parts, config=cfg).text)
            d["pages"] = [p for p in d.get("pages") or [] if p in pages]
            return d
        except Exception as e:
            msg = str(e); print("  call failed:", msg[:160], flush=True)
            time.sleep(min(60, 2 ** attempt + random.random() * 2) if ("429" in msg or "500" in msg or "503" in msg) else 2)
    return {"pages": [], "confidence": "low", "note": "error " + msg[:150]}


def coverage(pdf, pages, nums, have):
    hit = lambda s, n: n in s or (len(n) >= 4 and any(n in t for t in s if len(t) >= 7))
    return round(sum(1 for n in nums if any(hit(have.get((pdf, p), set()), n) for p in pages)) / max(1, len(nums)), 3)


def main():
    from google import genai
    con = sqlite3.connect(f"file:{g.DB}?mode=ro", uri=True)
    ov = json.load(open(os.path.join(F.V1, "column_fix_overlay.json")))
    fp = [json.loads(l) for l in open(os.path.join(HERE, "out", "found_pages.jsonl"))]
    todo = [r for r in fp if r["table_id"] in set(ov["withdrawn"]) and r.get("method") != "model"]
    done = {json.loads(l)["table_id"] for l in open(OUT)} if os.path.exists(OUT) else set()
    todo = [r for r in todo if r["table_id"] not in done]
    print(f"{len(todo)} withdrawn tables to search", flush=True)
    client = genai.Client(api_key=g.env_key())

    def job(r):
        it = item(sqlite3.connect(f"file:{g.DB}?mode=ro", uri=True), dict(r, found_pages=[]))
        lo, hi = max(1, r["linked_pages"][0] - WINDOW), r["linked_pages"][-1] + WINDOW
        import fitz
        hi = min(hi, fitz.open(r["pdf"]).page_count)
        allp = list(range(lo, hi + 1))
        found, notes = [], []
        for k in range(0, len(allp), CHUNK):
            t0 = time.time(); d = ask(client, it, r["pdf"], allp[k:k + CHUNK])
            print(f"  chunk {r['table_id'][:30]} {allp[k]}: {d['pages']} {d['confidence']} {time.time() - t0:.0f}s", flush=True)
            if d["confidence"] != "low":
                found += d["pages"]
            if d["pages"]:
                notes.append(f"{d['pages']} {d['confidence']}: {d['note'][:150]}")
        return {"table_id": r["table_id"], "pdf": r["pdf"], "linked_pages": r["linked_pages"],
                "found_pages": sorted(set(found)), "notes": notes}
    c = collections.Counter()
    with open(OUT, "a") as f, ThreadPoolExecutor(12) as ex:
        for k, fu in enumerate(as_completed([ex.submit(job, r) for r in todo]), 1):
            d = fu.result(); f.write(json.dumps(d) + "\n"); f.flush()
            c["found" if d["found_pages"] else "none"] += 1
            if k % 25 == 0:
                print(f"  {k}/{len(todo)} {dict(c)}", flush=True)
    # merge into found_pages.jsonl (model matches replace the local search's row for that table)
    have = {}
    for l in open(F.CACHE):
        d = json.loads(l); have[(d["pdf"], d["page"])] = set(d["nums"])
    model = {json.loads(l)["table_id"]: json.loads(l) for l in open(OUT)}
    merged = []
    for r in fp:
        m = model.get(r["table_id"])
        if m and m["found_pages"] and set(m["found_pages"]) != set(r["linked_pages"]):
            nums = {F.sig(v) for (v,) in con.execute("select value from facts where table_id=?", (r["table_id"],))} - {None}
            r = dict(r, found_pages=m["found_pages"], coverage=coverage(r["pdf"], m["found_pages"], nums, have),
                     method="model", model_notes=m["notes"])
        merged.append(r)
    with open(os.path.join(HERE, "out", "found_pages.jsonl"), "w") as f:
        for r in merged:
            f.write(json.dumps(r) + "\n")
    print(dict(c), "merged:", sum(1 for r in merged if r.get("method") == "model"), "\nMODEL_FIND_DONE", flush=True)


if __name__ == "__main__":
    main()
