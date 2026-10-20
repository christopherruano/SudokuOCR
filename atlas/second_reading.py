#!/usr/bin/env python3
"""Second, independent reading of every published table still carrying its original transcription
(tables corrected by the column pass already have a two/three-model vote and are skipped).

1. run: Gemini 3.8 Flash and Gemini 3.1 Pro (credits key only) are each given the table's published
   scan(s) and its layout -- sections, column headers and row labels in order -- and fill in the number
   printed in every cell. They are NOT shown the digitized values. Large tables go in row chunks.
   -> out/second_reading/<model>/<page>.json  (resumable)
2. third: a cell where both readers agree with each other but not with the published value, and the
   table's printed arithmetic cannot decide, is read once more on its own by Gemini 3.7 Flash (blind).
   -> out/second_reading/third.jsonl
3. decide + apply: a published cell is changed only when both readers give the same new value AND
     (a) the table's arithmetic holds with the new value and not with the old one (Total row/column,
         a 100/1,000/10,000 base, or Persons = Males + Females), or
     (b) the arithmetic cannot decide (none applies, or it fails with both), the third reader gives the same new
       value too, and Gemini 3.1 Pro, shown both candidates unlabelled, picks the new one (adjudicate_cells.py);
   and never when the arithmetic holds with the old value and fails with the new one. A table where
   the readers differ from the published values on more than MAX_CHANGE of the cells they share is not
   changed cell by cell (the layout was probably read differently); it is listed for review.
   -> sendout_v1/cell_fixes.csv (overlay: fact_id, old, new, evidence), out/second_reading/decisions.csv,
      out/second_reading/table_stats.csv; the explorer DB is patched in place and each changed table's
      review note says how many cells the second reading corrected.

  python3 atlas/second_reading.py run [--limit N]
  python3 atlas/second_reading.py third
  python3 atlas/second_reading.py decide        # dry: counts only
  python3 atlas/second_reading.py apply
"""
import os, re, sys, csv, json, time, random, sqlite3, hashlib, collections
from concurrent.futures import ThreadPoolExecutor, as_completed
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import gemini_column_check as g
import audit_cells as A
import vote_readings as V
import reread_columns as RC
import threading
STATS = collections.Counter(); _sl = threading.Lock()


def call(client, **kw):
    """reread_columns.call, counting calls, rate-limit pauses and time per call (printed in progress lines)"""
    t0 = time.time()
    try:
        return RC.call(client, **kw)
    finally:
        with _sl:
            STATS["calls"] += 1; STATS["secs"] += time.time() - t0

V1 = os.path.join(HERE, "..", "export_final/sendout_final/sendout_v1")
OD = os.path.join(HERE, "out", "second_reading")
MODELS = ["gemini-3.8-flash", "gemini-3.1-pro-preview"]
THIRD = "gemini-3.7-flash"
CHUNK_CELLS = int(os.environ.get("CHUNK_CELLS", 700))   # cells per call for large tables (smaller on retry
                                                        # of replies cut off mid-JSON)
MAX_CHANGE = 0.2
WORKERS = 48              # calls are slow (~100 s) and not rate-limited
TAG = "second reading 2026-10"

PROMPT = """These are the scanned page(s) of the census table "{title}" ({year}, {place}).

Below is the table's layout: its section(s), its value columns (numbered) and its row labels, in printed order. For every row listed, read from the page the number printed in each numbered column, exactly as printed.

Rules:
- Numbers without thousands separators; decimals as printed.
- A cell printed as a dash, dots ("...") or "nil" -> 0.
- A cell that is blank, or a row/column you cannot find on the pages -> null. Do not guess.
- Return one entry per listed row, in the same order, with exactly {{n}} values per row where n is the number of columns of its section.
- Read the page; values you report will be compared with an independent transcription.

{layout}"""
SCHEMA = {"type": "object", "properties": {"sections": {"type": "array", "items": {"type": "object", "properties": {
    "k": {"type": "integer"}, "rows": {"type": "array", "items": {"type": "object", "properties": {
        "r": {"type": "integer"}, "values": {"type": "array", "items": {"type": "number", "nullable": True}}},
        "required": ["r", "values"]}}}, "required": ["k", "rows"]}}}, "required": ["sections"]}


def undo_map():
    """fact_id -> value before this script's fixes, so decisions are always made against the original"""
    p = os.path.join(V1, "cell_fixes.csv")
    return {r["fact_id"]: float(r["old"]) for r in csv.DictReader(open(p))} if os.path.exists(p) else {}


def grid(con, tid, undo=None):
    """the published table as a read structure (sections -> columns, rows -> values) plus, per cell,
    (section index, row index, column index) -> fact (fact_id, value, sex, group)"""
    secs, cell = [], {}
    cur_sec = cur_row = None
    for fid, sec, row, grp, sub, sex, v in con.execute(
            "select fact_id, section, row_label_std, column_group_std, sub_column_std, sex, value from facts "
            "where table_id=? order by rowid", (tid,)):
        if undo and fid in undo:
            v = undo[fid]
        if cur_sec is None or sec != cur_sec["name"]:
            cur_sec = {"name": sec, "columns": [], "rows": [], "_ci": {}}; secs.append(cur_sec); cur_row = None
        ck = (grp or "", sub or "")
        if ck not in cur_sec["_ci"]:
            cur_sec["_ci"][ck] = len(cur_sec["columns"]); cur_sec["columns"].append({"group": grp or "", "sub": sub or ""})
        ci = cur_sec["_ci"][ck]
        if cur_row is None or row != cur_row["label"] or ci in cur_row["_v"]:
            cur_row = {"label": row, "_v": {}}; cur_sec["rows"].append(cur_row)
        cur_row["_v"][ci] = v
        cell[(len(secs) - 1, len(cur_sec["rows"]) - 1, ci)] = {"fact_id": fid, "value": v, "sex": sex, "group": grp or ""}
    for s in secs:
        n = len(s["columns"])
        for r in s["rows"]:
            r["values"] = [r["_v"].get(j) for j in range(n)]
    return {"sections": [{"name": s["name"], "columns": s["columns"],
                          "rows": [{"label": r["label"], "values": r["values"]} for r in s["rows"]]} for s in secs]}, cell


def chunks(G):
    """[(section index, row start, row end)] so that each call holds at most CHUNK_CELLS cells"""
    out = []
    for si, s in enumerate(G["sections"]):
        per = max(1, CHUNK_CELLS // max(1, len(s["columns"])))
        for a in range(0, len(s["rows"]), per):
            out.append((si, a, min(len(s["rows"]), a + per)))
    # merge small consecutive pieces of different sections into one call
    calls, cur, size = [], [], 0
    for si, a, b in out:
        n = (b - a) * len(G["sections"][si]["columns"])
        if cur and size + n > CHUNK_CELLS:
            calls.append(cur); cur, size = [], 0
        cur.append((si, a, b)); size += n
    if cur:
        calls.append(cur)
    return calls


def layout(G, part):
    txt = []
    for k, (si, a, b) in enumerate(part):
        s = G["sections"][si]
        txt.append(f"SECTION k={k}: {s['name'] or '(main table)'}" + (f" (rows {a + 1}-{b} of {len(s['rows'])})" if b - a < len(s["rows"]) else ""))
        txt.append("  columns: " + "; ".join(f"{j + 1}. {' / '.join(x for x in (c['group'], c['sub']) if x)}"
                                            for j, c in enumerate(s["columns"])))
        txt.append("  rows: " + "; ".join(f"r={i}: {s['rows'][i]['label']}" for i in range(a, b)))
    return "\n".join(txt)


def read_table(client, model, page, imgs, meta, G):
    from google.genai import types
    pics = []
    for i in imgs[:8]:
        b, mime = A.image(i); pics.append(types.Part.from_bytes(data=b, mime_type=mime))
    vals = {}
    for part in chunks(G):
        prompt = PROMPT.format(title=meta[0], year=meta[1], place=meta[2], layout=layout(G, part))
        cfg = types.GenerateContentConfig(response_mime_type="application/json", response_schema=SCHEMA, temperature=0,
                                          max_output_tokens=65535)
        try:
            d = json.loads(call(client, model=model, contents=pics + [types.Part.from_text(text=prompt)], config=cfg).text)
        except Exception as e:
            return {"page": page, "error": str(e)[:200]}
        for s in d.get("sections") or []:
            if not 0 <= s.get("k", -1) < len(part):
                continue
            si, a, b = part[s["k"]]
            for r in s.get("rows") or []:
                if a <= r.get("r", -1) < b:
                    for j, v in enumerate(r.get("values") or []):
                        if j < len(G["sections"][si]["columns"]) and v is not None:
                            vals[f"{si},{r['r']},{j}"] = v
    return {"page": page, "model": model, "values": vals}


def todo_tables():
    """published tables whose facts are all original (no Q-prefixed corrected facts)"""
    pub = A.published()
    con = sqlite3.connect(f"file:{g.DB}?mode=ro", uri=True)
    corrected = {t for (t,) in con.execute("select distinct table_id from facts where fact_id like 'Q%'")}
    return {t: v for t, v in pub.items() if t not in corrected}


def run(limit=None):
    from google import genai
    pub = todo_tables()
    con = sqlite3.connect(f"file:{g.DB}?mode=ro", uri=True)
    tids = sorted(pub)[:limit] if limit else sorted(pub)
    meta = {t: con.execute("select title, year, canonical_name from tables where table_id=?", (t,)).fetchone() for t in tids}
    grids = {t: grid(con, t)[0] for t in tids}
    client = genai.Client(api_key=g.env_key())
    print(f"{len(tids)} tables, {sum(len(x) for x in map(chunks, grids.values()))} calls per model", flush=True)
    jobs = []
    for model in MODELS:
        os.makedirs(os.path.join(OD, model), exist_ok=True)
        for t in tids:
            p = os.path.join(OD, model, pub[t][0] + ".json")
            if not os.path.exists(p):
                jobs.append((model, t, p))
    print(f"{len(jobs)} readings to do", flush=True)
    done = collections.Counter()

    def job(model, t, p):
        d = read_table(client, model, pub[t][0], pub[t][1], meta[t], grids[t])
        d["table_id"] = t
        if "error" not in d:
            json.dump(d, open(p, "w"))
        else:
            print(f"  failed {model} {pub[t][0][:60]}: {d['error'][:160]}", flush=True)
        return model, "error" in d
    with ThreadPoolExecutor(WORKERS) as ex:
        for k, fu in enumerate(as_completed([ex.submit(job, *j) for j in jobs]), 1):
            m, err = fu.result(); done[(m, err)] += 1
            if k % 50 == 0:
                print(f"  {k}/{len(jobs)} {dict(done)} calls={STATS['calls']} avg={STATS['secs'] / max(1, STATS['calls']):.0f}s "
                      f"rate_limited={RC._cool.get('n429', 0)}", flush=True)
    print(dict(done), "\nSECOND_RUN_DONE", flush=True)


def _pmf(G, cell, si, ri, ci, value):
    """Persons = Males + Females in the same row and column group, with `value` at (si, ri, ci)"""
    me = cell.get((si, ri, ci))
    if not me or me["sex"] not in ("All", "Male", "Female"):
        return None
    vals = {}
    for (a, b, c), f in cell.items():
        if a == si and b == ri and f["group"] == me["group"] and f["sex"] in ("All", "Male", "Female") and f["value"] is not None:
            if f["sex"] in vals:
                return None                    # several columns per sex in this group: ambiguous
            vals[f["sex"]] = value if c == ci else f["value"]
    if set(vals) != {"All", "Male", "Female"}:
        return None
    return abs(vals["All"] - vals["Male"] - vals["Female"]) <= 0.5


def fits(G, cell, si, ri, ci, value):
    c = {"si": si, "ri": ri, "ci": ci}
    a, b = V._fits(G, c, value), _pmf(G, cell, si, ri, ci, value)
    vs = [x for x in (a, b) if x is not None]
    return None if not vs else all(vs)


def readings(t, page):
    out = []
    for m in MODELS:
        p = os.path.join(OD, m, page + ".json")
        out.append(json.load(open(p))["values"] if os.path.exists(p) else None)
    return out


def num(x):
    return None if x is None else float(x)


def candidates():
    """per table: stats and the cells where both readers agree on a value other than the published one"""
    pub = todo_tables()
    con = sqlite3.connect(f"file:{g.DB}?mode=ro", uri=True)
    res, undo = {}, undo_map()
    for t, (page, imgs) in sorted(pub.items()):
        fr, pr = readings(t, page)
        if fr is None or pr is None:
            continue
        G, cell = grid(con, t, undo)
        st = collections.Counter(cells=len([c for c in cell.values() if c["value"] is not None]))
        cand = []
        for (si, ri, ci), f in cell.items():
            v = f["value"]
            if v is None:
                continue
            k = f"{si},{ri},{ci}"
            a, b = num(fr.get(k)), num(pr.get(k))
            if a is None or b is None:
                st["unread"] += 1; continue
            st["shared"] += 1
            if a == b == v:
                st["both_confirm"] += 1
            elif a == b:
                st["both_differ"] += 1
                cand.append({"si": si, "ri": ri, "ci": ci, "fact_id": f["fact_id"], "old": v, "new": a,
                             "fits_old": fits(G, cell, si, ri, ci, v), "fits_new": fits(G, cell, si, ri, ci, a),
                             "row": G["sections"][si]["rows"][ri]["label"], "section": G["sections"][si]["name"],
                             "group": G["sections"][si]["columns"][ci]["group"], "sub": G["sections"][si]["columns"][ci]["sub"]})
            elif a == v or b == v:
                st["one_confirms"] += 1
            else:
                st["neither"] += 1
        res[t] = {"page": page, "imgs": imgs, "stats": st, "cand": cand}
    return res


def third():
    from google import genai
    con = sqlite3.connect(f"file:{g.DB}?mode=ro", uri=True)
    res = candidates()
    out = os.path.join(OD, "third.jsonl")
    done = {json.loads(l)["fact_id"] for l in open(out)} if os.path.exists(out) else set()
    todo = []
    for t, r in res.items():
        st = r["stats"]
        if st["both_differ"] > MAX_CHANGE * max(1, st["shared"]):
            continue
        meta = con.execute("select title, year, canonical_name from tables where table_id=?", (t,)).fetchone()
        for c in r["cand"]:
            if not c["fits_old"] and not c["fits_new"] and c["fact_id"] not in done:
                todo.append((t, r["imgs"], meta, c))
    print(f"{len(todo)} cells for the third reader", flush=True)
    client = genai.Client(api_key=g.env_key())

    def job(t, imgs, meta, c):
        d = A.read_one(client, THIRD, {"section": c["section"], "row": c["row"], "group": c["group"], "sub": c["sub"]}, imgs, meta)
        return dict(d, fact_id=c["fact_id"], table_id=t)
    with open(out, "a") as f, ThreadPoolExecutor(WORKERS) as ex:
        for k, fu in enumerate(as_completed([ex.submit(job, *j) for j in todo]), 1):
            d = fu.result()
            if not d.get("error"):
                f.write(json.dumps(d) + "\n"); f.flush()
            if k % 500 == 0:
                print(f"  third {k}/{len(todo)}", flush=True)
    print("THIRD_DONE", flush=True)


def decide():
    res = candidates()
    tp = os.path.join(OD, "third.jsonl")
    T = {d["fact_id"]: A.norm(d.get("printed")) for d in map(json.loads, open(tp))} if os.path.exists(tp) else {}
    xp = os.path.join(OD, "excluded_fixes.txt")
    EXCL = {l.split("#")[0].strip() for l in open(xp)} - {""} if os.path.exists(xp) else set()
    ap = os.path.join(OD, "adjudicate.jsonl")      # Gemini Pro forced choice (adjudicate_cells.py)
    AJ = {d["fact_id"]: d["verdict"] for d in map(json.loads, open(ap))} if os.path.exists(ap) else {}
    fixes, rows, tstats = [], [], []
    tot = collections.Counter()
    for t, r in res.items():
        st = r["stats"]
        layout_differs = st["both_differ"] > MAX_CHANGE * max(1, st["shared"]) and st["both_differ"] >= 3
        n_fix = 0
        for c in r["cand"]:
            if c["fact_id"] in EXCL:
                why = "kept: excluded by hand (out/second_reading/excluded_fixes.txt)"
            elif layout_differs:
                why = "table not changed: readers differ on more than 20% of cells"
            elif c["fits_old"] and not c["fits_new"]:
                why = "kept: printed arithmetic holds with the published value"
            elif c["fits_new"] and not c["fits_old"]:
                why = "fixed: arithmetic"
            elif not c["fits_new"] and not c["fits_old"]:       # no arithmetic, or it fails either way
                th = T.get(c["fact_id"])
                aj = AJ.get(c["fact_id"])
                why = (("fixed: three readers and adjudication" if aj == "%g" % c["new"] else
                        "kept: three readers, adjudication pending" if aj is None else "kept: adjudication differs")
                       if th is not None and th == "%g" % c["new"] else
                       "kept: third reader not read yet" if c["fact_id"] not in T else "kept: third reader differs")
            else:
                why = "kept: arithmetic undecided"
            rows.append(dict(table_id=t, page=r["page"], fact_id=c["fact_id"], section=c["section"], row=c["row"],
                             group=c["group"], sub=c["sub"], old=c["old"], new=c["new"], decision=why))
            tot[why] += 1
            if why.startswith("fixed"):
                n_fix += 1
                fixes.append(dict(fact_id=c["fact_id"], table_id=t, old=c["old"], new=c["new"], evidence=why[7:]))
        tstats.append(dict(table_id=t, page=r["page"], **{k: st[k] for k in
                      ("cells", "shared", "both_confirm", "one_confirms", "both_differ", "neither", "unread")},
                      fixed=n_fix, layout_differs=int(layout_differs)))
    os.makedirs(OD, exist_ok=True)
    for name, data in (("decisions.csv", rows), ("table_stats.csv", tstats)):
        if data:
            with open(os.path.join(OD, name), "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(data[0])); w.writeheader(); w.writerows(data)
    S = collections.Counter()
    for s in tstats:
        for k in ("cells", "shared", "both_confirm", "one_confirms", "both_differ", "neither", "unread"):
            S[k] += s[k]
    print(f"{len(tstats)} tables read twice; {S['cells']:,} cells, {S['shared']:,} read by both")
    print(f"  both readers = published {S['both_confirm']:,} ({S['both_confirm'] / max(1, S['shared']):.1%}); "
          f"one = published {S['one_confirms']:,}; both agree on another value {S['both_differ']:,}; neither {S['neither']:,}")
    print(f"  tables with layout differences (not changed): {sum(s['layout_differs'] for s in tstats)}")
    for k, v in tot.most_common():
        print(f"  {k:62} {v:,}")
    return fixes, tstats


def apply_fixes(con, fixes=None):
    """patch the DB from the overlay (idempotent: a fact is changed only while it still holds the old value)"""
    p = os.path.join(V1, "cell_fixes.csv")
    if fixes is None:
        if not os.path.exists(p):
            return 0
        fixes = list(csv.DictReader(open(p)))
    n = 0
    per = collections.Counter()
    for f in fixes:
        cur = con.execute("update facts set value=? where fact_id=? and table_id=? and abs(value-?)<1e-9",
                          (float(f["new"]), f["fact_id"], f["table_id"], float(f["old"])))
        n += cur.rowcount; per[f["table_id"]] += 1
    for t, k in per.items():
        old = con.execute("select review_note from tables where table_id=?", (t,)).fetchone()[0] or ""
        old = re.sub(r"\s*\|?\s*Second reading \(" + re.escape(TAG) + r"\)[^|]*", "", old).strip(" |")
        note = (f"Second reading ({TAG}): {k} cell{'s' if k > 1 else ''} corrected where two independent readings of "
                f"the scan agreed on a different number, confirmed by the table's printed totals or a third reading.")
        con.execute("update tables set review_note=? where table_id=?", ((old + " | " if old else "") + note, t))
    con.commit()
    return n


def apply():
    fixes, _ = decide()
    # every published cell an independent reading questions but that was not changed, for users to see
    with open(os.path.join(OD, "decisions.csv")) as f, open(os.path.join(V1, "disputed_cells.csv"), "w", newline="") as o:
        w = csv.writer(o); w.writerow(["fact_id", "table_id", "section", "row", "column", "published", "second_reading", "why_kept"])
        for d in csv.DictReader(f):
            if not d["decision"].startswith("fixed"):
                w.writerow([d["fact_id"], d["table_id"], d["section"], d["row"], " | ".join(x for x in (d["group"], d["sub"]) if x),
                            "%g" % float(d["old"]), "%g" % float(d["new"]), d["decision"][6:]])
    p = os.path.join(V1, "cell_fixes.csv")
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["fact_id", "table_id", "old", "new", "evidence"]); w.writeheader(); w.writerows(fixes)
    con = sqlite3.connect(g.DB)
    print(f"cell_fixes.csv: {len(fixes):,} fixes; DB cells changed now: {apply_fixes(con, fixes):,}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "run":
        run(int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None)
    elif cmd == "third":
        third()
    elif cmd == "decide":
        decide()
    elif cmd == "apply":
        apply()
    else:
        print(__doc__)
