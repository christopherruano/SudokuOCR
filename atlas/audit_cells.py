#!/usr/bin/env python3
"""Blind cell-level accuracy audit of the published corpus.

1. sample (fixed seed, saved before any reading): N cells drawn uniformly at random from every
   published cell (tables on the site, withdrawn tables excluded) -> out/audit/sample.jsonl
2. read: Gemini 3.8 Flash and Gemini 3.1 Pro (credits key only) each read, from the table's published
   scan(s), the number printed at the cell's row and column. They are NOT shown the digitized value.
   -> out/audit/read_<model>.jsonl
3. score: where both readers give the same number it is the reference; the digitized value is correct
   iff it equals the reference. Cells where the readers disagree (or either finds nothing) are listed
   for review against the scan (out/audit/disputed.jsonl); reviewer verdicts go in
   out/audit/review.jsonl as {"n": ..., "printed": "..."|null}.

  python3 atlas/audit_cells.py sample [N]
  python3 atlas/audit_cells.py read
  python3 atlas/audit_cells.py score
"""
import os, re, sys, json, glob, time, random, sqlite3, hashlib, collections, math
from concurrent.futures import ThreadPoolExecutor, as_completed
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import gemini_column_check as g

AD = os.path.join(HERE, "out", os.environ.get("AUDIT", "audit")); os.makedirs(AD, exist_ok=True)   # AUDIT=audit2 for a re-audit
SEED = int(os.environ.get("AUDIT_SEED", 20261017))
SITE = os.path.join(HERE, "..", "census_site")
ADDED = os.path.join(HERE, "..", "export_final/sendout_final/sendout_v1/added_pages")
MODELS = ["gemini-3.8-flash", "gemini-3.1-pro-preview"]
PROMPT = """These are the scanned page(s) of the census table "{title}" ({year}, {place}). Find the cell at

  section: {section}
  row: "{row}"
  column: "{col}"

and report the number printed in that cell exactly as printed (digits, with any decimal point; no commas). If the cell is printed as a dash or blank, report "-". If you cannot find this row and column on the pages, report null. Read the page; do not guess."""
SCHEMA = {"type": "object", "properties": {"printed": {"type": "string", "nullable": True},
          "note": {"type": "string"}}, "required": ["printed", "note"]}


def published():
    """table_id -> (page id, [image names]) for every table page on the site that carries data"""
    con = sqlite3.connect(f"file:{g.DB}?mode=ro", uri=True)
    tid_of = {hashlib.md5(t.encode()).hexdigest()[:7]: t for (t,) in con.execute("select table_id from tables")}
    out = {}
    for f in glob.glob(os.path.join(SITE, "tables", "*.html")):
        s = open(f, errors="ignore").read()
        if "<title>Table Withdrawn</title>" in s[:1500] or "<title>Moved</title>" in s[:1500]:
            continue
        p = os.path.basename(f)[:-5]; t = tid_of.get(p.rsplit("_", 1)[1])
        imgs = re.findall(r'src="\.\./img/([^"]+)"', s)
        if t and imgs:
            out[t] = (p, imgs)
    return out


def sample(n):
    pub = published()
    con = sqlite3.connect(f"file:{g.DB}?mode=ro", uri=True)
    rows = []
    for t in pub:
        for fid, sec, r, gr, s, v in con.execute("select fact_id, section, row_label_std, column_group_std, sub_column_std, value "
                                                 "from facts where table_id=? and value is not null", (t,)):
            rows.append((t, fid, sec, r, gr, s, v))
    rnd = random.Random(SEED)
    pick = rnd.sample(rows, n)
    with open(os.path.join(AD, "sample.jsonl"), "w") as f:
        for k, (t, fid, sec, r, gr, s, v) in enumerate(pick, 1):
            f.write(json.dumps({"n": k, "table_id": t, "page": pub[t][0], "fact_id": fid, "section": sec, "row": r,
                                "group": gr, "sub": s, "value": v, "corrected": str(fid).startswith("Q")}) + "\n")
    print(f"{n} cells sampled from {len(rows):,} published cells in {len(pub):,} tables "
          f"({sum(1 for x in pick if str(x[1]).startswith('Q'))} from corrected tables)")


def image(name):
    """bytes, mime of a published scan: added pages from disk, others via the native scan"""
    stem = name.rsplit(".", 1)[0]
    p = os.path.join(ADDED, stem + ".png")
    if os.path.exists(p):
        return open(p, "rb").read(), "image/png"
    b, mime, _ = g.native_scan(name)
    return b, mime


def read_one(client, model, c, imgs, meta):
    from google.genai import types
    parts = []
    for i in imgs[:8]:
        b, mime = image(i); parts.append(types.Part.from_bytes(data=b, mime_type=mime))
    col = " | ".join(x for x in (c["group"], c["sub"]) if x)
    parts.append(types.Part.from_text(text=PROMPT.format(title=meta[0], year=meta[1], place=meta[2],
                                                         section=c["section"] or "(none)", row=c["row"], col=col)))
    cfg = types.GenerateContentConfig(response_mime_type="application/json", response_schema=SCHEMA, temperature=0)
    for attempt in range(8):
        try:
            return json.loads(client.models.generate_content(model=model, contents=parts, config=cfg).text)
        except Exception as e:
            msg = str(e); time.sleep(min(60, 2 ** attempt + random.random() * 2))
    return {"printed": None, "note": "error " + msg[:150], "error": True}


def read():
    from google import genai
    pub = published()
    con = sqlite3.connect(f"file:{g.DB}?mode=ro", uri=True)
    cells = [json.loads(l) for l in open(os.path.join(AD, "sample.jsonl"))]
    meta = {c["table_id"]: con.execute("select title, year, canonical_name from tables where table_id=?",
                                       (c["table_id"],)).fetchone() for c in cells}
    client = genai.Client(api_key=g.env_key())
    for model in MODELS:
        out = os.path.join(AD, f"read_{model}.jsonl")
        done = {json.loads(l)["n"] for l in open(out) if not json.loads(l).get("error")} if os.path.exists(out) else set()
        todo = [c for c in cells if c["n"] not in done]

        def job(c):
            return dict(read_one(client, model, c, pub[c["table_id"]][1], meta[c["table_id"]]), n=c["n"])
        with open(out, "a") as f, ThreadPoolExecutor(16) as ex:
            for fu in as_completed([ex.submit(job, c) for c in todo]):
                f.write(json.dumps(fu.result()) + "\n"); f.flush()
        print(model, "done", flush=True)


def norm(x):
    if x is None:
        return None
    x = str(x).strip().replace(",", "")
    if x in ("-", "—", "–", "", "..", "...", "nil") or set(x) <= set(".…-—–· "):
        return "0"                  # a printed dash / dots means nil, stored as 0
    try:
        return "%g" % float(x)
    except ValueError:
        return None


def wilson(k, n, z=1.96):
    if not n:
        return (0, 0)
    p = k / n; d = 1 + z * z / n
    c = p + z * z / (2 * n); h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def score():
    cells = {c["n"]: c for c in map(json.loads, open(os.path.join(AD, "sample.jsonl")))}
    R = {m: {d["n"]: d for d in map(json.loads, open(os.path.join(AD, f"read_{m}.jsonl")))} for m in MODELS}
    rev_p = os.path.join(AD, "review.jsonl")
    rev = {d["n"]: d for d in map(json.loads, open(rev_p))} if os.path.exists(rev_p) else {}
    res, disputed = {}, []
    for n, c in cells.items():
        a, b = (norm(R[m].get(n, {}).get("printed")) for m in MODELS)
        mine = "%g" % c["value"]
        if n in rev:
            ref = norm(rev[n].get("printed")); how = "review"
        elif a is not None and a == b:
            ref = a; how = "two readers agree"
        else:
            disputed.append(dict(c, flash=a, pro=b)); continue
        res[n] = (ref == mine, how, ref)
    with open(os.path.join(AD, "disputed.jsonl"), "w") as f:
        for d in disputed:
            f.write(json.dumps(d) + "\n")
    tot = collections.Counter()
    for n, (ok, how, ref) in res.items():
        grp = "corrected" if cells[n]["corrected"] else "original"
        tot[(grp, ok)] += 1; tot[("all", ok)] += 1
    for grp in ("all", "original", "corrected"):
        k, m = tot[(grp, True)], tot[(grp, True)] + tot[(grp, False)]
        lo, hi = wilson(k, m)
        print(f"{grp:10} {k}/{m} correct = {k / max(1, m):.1%}  (95% CI {lo:.1%}-{hi:.1%})")
    print(f"unresolved (readers disagree, awaiting review): {len(disputed)}")
    wrong = [dict(cells[n], reference=ref) for n, (ok, how, ref) in res.items() if not ok]
    with open(os.path.join(AD, "errors.jsonl"), "w") as f:
        for d in wrong:
            f.write(json.dumps(d) + "\n")
    print(f"errors listed in out/audit/errors.jsonl: {len(wrong)}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "sample":
        sample(int(sys.argv[2]) if len(sys.argv) > 2 else 400)
    elif cmd == "read":
        read()
    elif cmd == "score":
        score()
    else:
        print(__doc__)
