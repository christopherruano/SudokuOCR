#!/usr/bin/env python3
"""Column check (atlas/gemini_column_check.py PROMPT, Gemini 3.8 Flash, credits key only) of each table
against the pages find_pages.py located on the drive (out/found_pages.jsonl, coverage >= 0.6).

  python3 atlas/verify_found_pages.py     -> out/found_pages_check.jsonl
"""
import os, sys, json, collections, sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import gemini_column_check as g
from find_right_scan import render

MODEL = "gemini-3.8-flash"
OUT = os.path.join(HERE, "out", "found_pages_check.jsonl")


def item(con, r):
    tid = r["table_id"]
    ti, y, pl = con.execute("select title,year,canonical_name from tables where table_id=?", (tid,)).fetchone()
    cols, rows = collections.OrderedDict(), set()
    for sec, row, gr, s, v in con.execute("select section,row_label_std,column_group_std,sub_column_std,value from facts "
                                          "where table_id=? order by rowid", (tid,)):
        k = f"{gr or ''} | {s or ''}"; cols.setdefault(k, [])
        if len(cols[k]) < 3:
            cols[k].append(f"{row}={v:g}" if v is not None else f"{row}=blank")
        rows.add((sec, row))
    imgs = []
    for p in r["found_pages"]:                  # native scan of each found page, cached under <pdf>_p<page>
        name = f"{os.path.basename(r['pdf']).rsplit('.', 1)[0]}_p{p}"
        if not any(os.path.exists(os.path.join(g.NATIVE, name + e)) for e in (".jpg", ".png")):
            b, mime = render(r["pdf"], p)
            open(os.path.join(g.NATIVE, name + (".png" if mime == "image/png" else ".jpg")), "wb").write(b)
        imgs.append(name + ".png")
    return dict(page=tid, table_id=tid, title=ti, year=y, place=pl, imgs=imgs, n_rows=len(rows), n_columns=len(cols),
                columns=[f"{k}  [first: {'; '.join(v)}]" for k, v in list(cols.items())[:120]])


def current_checks():
    """table_id -> the latest column check made on the table's current found pages"""
    fp = {json.loads(l)["table_id"]: json.loads(l) for l in open(os.path.join(HERE, "out", "found_pages.jsonl"))}
    out = {}
    for l in (open(OUT) if os.path.exists(OUT) else []):
        d = json.loads(l); r = fp.get(d["table_id"])
        if r and [int(os.path.basename(i).rsplit("_p", 1)[1].split(".")[0]) for i in d["found_images"]] == r["found_pages"]:
            out[d["table_id"]] = d
    return out


def main():
    from google import genai
    con = sqlite3.connect(f"file:{g.DB}?mode=ro", uri=True)
    rs = [json.loads(l) for l in open(os.path.join(HERE, "out", "found_pages.jsonl"))]
    # local matches need 60% of the table's numbers on the pages; model matches need 10 of 12 sampled
    # cells read back exactly from the pages (confirm_values.py)
    conf = {}
    cp = os.path.join(HERE, "out", "confirm_values.jsonl")
    for l in (open(cp) if os.path.exists(cp) else []):
        d = json.loads(l)
        if d["kind"] == "found" and d["confirmed"] is not None:
            conf[(d["table_id"], tuple(d["pages"]))] = d["confirmed"]
    rs = [r for r in rs if r["found_pages"] and (r["coverage"] >= 0.6 if r.get("method") != "model"
                                                 else conf.get((r["table_id"], tuple(r["found_pages"])), 0) >= 10)]
    key = lambda d: (d["table_id"], tuple(os.path.basename(i) for i in d["found_images"]))
    done = {key(json.loads(l)) for l in open(OUT)} if os.path.exists(OUT) else set()
    items = [it for it in (item(con, r) for r in rs) if key(dict(it, found_images=it["imgs"])) not in done]
    print(f"{len(items)} tables to check on their found pages", flush=True)
    client = genai.Client(api_key=g.env_key())
    c = collections.Counter()
    with open(OUT, "a") as f, ThreadPoolExecutor(16) as ex:
        futs = {ex.submit(g.check, client, MODEL, it): it for it in items}
        for fu in as_completed(futs):
            it = futs[fu]; d = fu.result()
            d.update(table_id=it["table_id"], found_images=it["imgs"])
            f.write(json.dumps(d) + "\n"); f.flush(); c[d["verdict"]] += 1
    print(dict(c), "\nVERIFY_DONE", flush=True)


if __name__ == "__main__":
    main()
