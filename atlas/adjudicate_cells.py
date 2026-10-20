#!/usr/bin/env python3
"""Forced-choice adjudication of second-reading disputes by Gemini 3.1 Pro (credits key only).

For a cell where the published value and the agreed second reading differ, Pro is shown the table's
scan(s), the cell's position (section, row with its neighbours, column with the full column list) and
the two candidate values in random order -- without saying which is published -- and reports which one
is printed (or neither), using the page's own sums where it can.

  python3 atlas/adjudicate_cells.py calibrate   # the 100 cells reviewed by Claude (review/batch_*.json)
  python3 atlas/adjudicate_cells.py run         # all undecided disputes -> out/second_reading/adjudicate.jsonl
"""
import os, sys, json, glob, random, collections, sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import gemini_column_check as g
import second_reading as S
import audit_cells as A

MODEL = "gemini-3.1-pro-preview"
OD = S.OD
PROMPT = """These are the scanned page(s) of the census table "{title}" ({year}, {place}). One cell of a digitized copy of this table is disputed.

Cell:
  section: {section}
  row: "{row}" (row {row_position}); neighbouring row labels: {nearby}
  column: "{column}" (column {column_position}); all columns of this section: {columns}

Two candidate values: A = {a}, B = {b}. Exactly one, or neither, is printed in this cell.
Find the cell on the page and read it carefully. Where the page prints totals, use them (row or column sums, Persons = Males + Females) to settle doubtful digits. If the table spans several pages or blocks (e.g. different religions or sexes), use the section name to pick the right one.
Answer "A", "B", "neither" (give the printed value) or "cannot" (the cell cannot be located with confidence)."""
SCHEMA = {"type": "object", "properties": {
    "answer": {"type": "string", "enum": ["A", "B", "neither", "cannot"]},
    "printed": {"type": "string", "nullable": True},
    "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
    "note": {"type": "string"}}, "required": ["answer", "confidence", "note"]}


def ask(client, p, imgs):
    from google.genai import types
    parts = []
    for i in imgs[:8]:
        b, mime = A.image(i); parts.append(types.Part.from_bytes(data=b, mime_type=mime))
    parts.append(types.Part.from_text(text=PROMPT.format(
        title=p["title"], year=p["year"], place=p["place"], section=p["section"] or "(main table)", row=p["row"],
        row_position=p["row_position"], nearby="; ".join(p["row_labels_in_section"]), column=p["column"],
        column_position=p["column_position"], columns="; ".join(p["columns_in_section"]),
        a=p["candidates"][0], b=p["candidates"][1])))
    cfg = types.GenerateContentConfig(response_mime_type="application/json", response_schema=SCHEMA, temperature=0)
    try:
        return json.loads(S.call(client, model=MODEL, contents=parts, config=cfg).text)
    except Exception as e:
        return {"answer": "error", "note": str(e)[:200]}


def packet(con, pub, d, rnd, undo):
    t = d["table_id"]
    ti, y, pl = con.execute("select title,year,canonical_name from tables where table_id=?", (t,)).fetchone()
    G, cell = S.grid(con, t, undo)
    pos = next(p for p, f in cell.items() if f["fact_id"] == d["fact_id"])
    sec = G["sections"][pos[0]]
    cands = ["%g" % float(d["old"]), "%g" % float(d["new"])]; rnd.shuffle(cands)
    return {"title": ti, "year": y, "place": pl, "section": sec["name"], "row": d["row"],
            "row_position": f"{pos[1] + 1} of {len(sec['rows'])}", "column": " / ".join(x for x in (d["group"], d["sub"]) if x),
            "column_position": f"{pos[2] + 1} of {len(sec['columns'])}",
            "row_labels_in_section": [r["label"] for r in sec["rows"]][max(0, pos[1] - 4):pos[1] + 5],
            "columns_in_section": [" / ".join(x for x in (c["group"], c["sub"]) if x) for c in sec["columns"]][:60],
            "candidates": cands}, pub[t][1]


def verdict(p, r):
    """'old', 'new', 'neither', 'cannot' from Pro's answer"""
    if r.get("answer") in ("A", "B"):
        v = p["candidates"][0 if r["answer"] == "A" else 1]
        return v, r.get("confidence")
    return r.get("answer"), r.get("confidence")


def calibrate():
    from google import genai
    client = genai.Client(api_key=g.env_key())
    R = os.path.join(OD, "review")
    K = {k["n"]: k for k in json.load(open(os.path.join(R, "key.json")))}
    P = [p for f in sorted(glob.glob(os.path.join(R, "batch_*.json"))) for p in json.load(open(f))]
    C = {}
    for f in glob.glob(os.path.join(R, "out_*.jsonl")):
        for l in open(f):
            d = json.loads(l); C[d["n"]] = d
    pub = A.published()

    def job(p):
        imgs = pub[K[p["n"]]["table_id"]][1]
        return p, ask(client, p, imgs)
    out = open(os.path.join(R, "gemini_adjudication.jsonl"), "w")
    tab = collections.Counter()
    with ThreadPoolExecutor(25) as ex:
        for fu in as_completed([ex.submit(job, p) for p in P]):
            p, r = fu.result(); out.write(json.dumps(dict(r, n=p["n"])) + "\n")
            k, c = K[p["n"]], C.get(p["n"])
            if not c or c["choice"] not in (0, 1):
                continue
            truth = "published" if c["choice"] == k["old"] else "new"
            old = p["candidates"][k["old"]]
            v, conf = verdict(p, r)
            g_ = "published" if v == old else "new" if v in p["candidates"] else v
            tab[(k["stratum"][:24], truth, g_, conf)] += 1
    for x in sorted(tab.items(), key=str):
        print(x)


def run():
    import csv
    from google import genai
    client = genai.Client(api_key=g.env_key())
    con = sqlite3.connect(f"file:{g.DB}?mode=ro", uri=True)
    pub = A.published(); undo = S.undo_map()
    D = [d for d in csv.DictReader(open(os.path.join(OD, "decisions.csv")))
         if d["decision"] == "kept: three readers, adjudication pending"]
    out = os.path.join(OD, "adjudicate.jsonl")
    done = {json.loads(l)["fact_id"] for l in open(out)} if os.path.exists(out) else set()
    D = [d for d in D if d["fact_id"] not in done]
    print(f"{len(D)} cells to adjudicate", flush=True)
    rnd = random.Random(7)

    def job(d):
        p, imgs = packet(con if False else sqlite3.connect(f"file:{g.DB}?mode=ro", uri=True), pub, d, random.Random(d["fact_id"]), undo)
        r = ask(client, p, imgs)
        v, conf = verdict(p, r)
        return dict(r, fact_id=d["fact_id"], table_id=d["table_id"], verdict=v, old="%g" % float(d["old"]), new="%g" % float(d["new"]))
    with open(out, "a") as f, ThreadPoolExecutor(S.WORKERS) as ex:
        for k, fu in enumerate(as_completed([ex.submit(job, d) for d in D]), 1):
            r = fu.result()
            if r.get("answer") != "error":
                f.write(json.dumps(r) + "\n"); f.flush()
            if k % 500 == 0:
                print(f"  adjudicated {k}/{len(D)}", flush=True)
    print("ADJ_DONE", flush=True)


if __name__ == "__main__":
    {"calibrate": calibrate, "run": run}.get(sys.argv[1] if len(sys.argv) > 1 else "", lambda: print(__doc__))()
