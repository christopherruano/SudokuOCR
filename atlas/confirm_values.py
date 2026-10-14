#!/usr/bin/env python3
"""Value test for pages found by find_pages_model.py: Gemini 3.8 Flash (credits key only) is given 12
digitized cells of the table (row, column, value; at least 3 significant digits, spread over columns)
and the found pages at full resolution, and reports for each cell the value printed at that row and
column. A cell is confirmed when the printed value equals the digitized one. The same test on the
table's linked (wrong) scan is the control.

  python3 atlas/confirm_values.py [--control]   -> out/confirm_values.jsonl
"""
import os, sys, json, time, random, sqlite3, collections
from concurrent.futures import ThreadPoolExecutor, as_completed
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import gemini_column_check as g
from find_right_scan import render
import find_pages as F

MODEL = "gemini-3.8-flash"
OUT = os.path.join(HERE, "out", "confirm_values.jsonl")
N_CELLS = 12
PROMPT = """These scanned census pages may hold the table "{title}" ({year}, {place}). For each numbered cell below (row label, column header), find that row and column on the pages and report the number printed there exactly as printed (digits only), or null if that row/column is not on these pages. Do not guess and do not copy the value given; read the page.

{cells}"""
SCHEMA = {"type": "object", "properties": {"cells": {"type": "array", "items": {"type": "object", "properties": {
    "n": {"type": "integer"}, "printed": {"type": "string", "nullable": True}}, "required": ["n", "printed"]}}},
    "required": ["cells"]}


def pick_cells(con, tid):
    rows = [(r, gr, s, v) for r, gr, s, v in con.execute(
        "select row_label_std, column_group_std, sub_column_std, value from facts where table_id=?", (tid,)) if F.sig(v)]
    by = collections.defaultdict(list)
    for x in rows:
        by[(x[1], x[2])].append(x)
    rnd = random.Random(tid); out = []
    cols = list(by); rnd.shuffle(cols)
    while len(out) < N_CELLS and any(by[c] for c in cols):
        for c in cols:
            if by[c] and len(out) < N_CELLS:
                out.append(by[c].pop(rnd.randrange(len(by[c]))))
    return out


def ask(client, title, year, place, pdf, pages, cells):
    from google.genai import types
    parts = []
    for p in pages[:6]:
        r = render(pdf, p)
        if r:
            parts += [types.Part.from_text(text=f"Page {p}:"), types.Part.from_bytes(data=r[0], mime_type=r[1])]
    # the value is deliberately NOT shown: the model reads it, we compare
    lines = "\n".join(f"{k}. row \"{r}\", column \"{gr or ''} | {s or ''}\"" for k, (r, gr, s, v) in enumerate(cells, 1))
    parts.append(types.Part.from_text(text=PROMPT.format(title=title, year=year, place=place, cells=lines)))
    cfg = types.GenerateContentConfig(response_mime_type="application/json", response_schema=SCHEMA, temperature=0)
    for attempt in range(8):
        try:
            return json.loads(client.models.generate_content(model=MODEL, contents=parts, config=cfg).text)["cells"]
        except Exception as e:
            msg = str(e); time.sleep(min(60, 2 ** attempt + random.random() * 2))
    return None


def score(cells, answer):
    got = {a["n"]: a.get("printed") for a in answer or []}
    ok = sum(1 for k, (r, gr, s, v) in enumerate(cells, 1)
             if got.get(k) and F.digits(got[k]) == F.digits(("%f" % float(v)).rstrip("0").rstrip(".")))
    return ok


def main():
    from google import genai
    control = "--control" in sys.argv
    con = sqlite3.connect(f"file:{g.DB}?mode=ro", uri=True)
    rs = [json.loads(l) for l in open(os.path.join(HERE, "out", "found_pages.jsonl"))]
    rs = [r for r in rs if r.get("method") == "model"]
    if control:
        rs = random.Random(1).sample(rs, 30)
    kind = "linked" if control else "found"
    done = {(d["table_id"], d["kind"], tuple(d["pages"])) for d in map(json.loads, open(OUT))} if os.path.exists(OUT) else set()
    client = genai.Client(api_key=g.env_key())

    def job(r):
        pages = r["linked_pages"] if control else r["found_pages"]
        if (r["table_id"], kind, tuple(pages)) in done:
            return None
        c2 = sqlite3.connect(f"file:{g.DB}?mode=ro", uri=True)
        ti, y, pl = c2.execute("select title,year,canonical_name from tables where table_id=?", (r["table_id"],)).fetchone()
        cells = pick_cells(c2, r["table_id"])
        ans = ask(client, ti, y, pl, r["pdf"], pages, cells)
        return {"table_id": r["table_id"], "kind": kind, "pages": pages, "n": len(cells),
                "confirmed": score(cells, ans) if ans is not None else None,
                "answers": ans, "cells": [[a, b, c, d] for a, b, c, d in cells]}
    c = collections.Counter()
    with open(OUT, "a") as f, ThreadPoolExecutor(12) as ex:
        for fu in as_completed([ex.submit(job, r) for r in rs]):
            d = fu.result()
            if d:
                f.write(json.dumps(d) + "\n"); f.flush()
                c[None if d["confirmed"] is None else round(d["confirmed"] / max(1, d["n"]), 1)] += 1
    print(kind, sorted(c.items(), key=lambda x: (x[0] is None, x[0] or 0)), "\nCONFIRM_DONE", flush=True)


if __name__ == "__main__":
    main()
