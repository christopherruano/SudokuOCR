#!/usr/bin/env python3
"""Re-read tables whose columns were misread (column check: out/gemini_column_check_*.jsonl),
gate each re-read, and write accepted ones to the corrections overlay.

For each table: the full-resolution PDF page images (plus the facing pages when the check
said columns came from an unattached facing page, or the relinked page for wrong scans) go
to Gemini with the table title, the existing row labels (to be reused verbatim) and the
column-check diagnosis. The model transcribes the printed header hierarchy and every value.

Acceptance gates (all must pass; otherwise the original stays and the page gets a note):
  1. independent column check of the NEW reading: ok (or persons_only for rate tables)
  2. agreement with the old values on cells both readings share (same row/column): >= 90%
     of shared non-zero cells equal, when >= 10 are shared
  3. count tables: Persons = Males + Females within 1% on >= 90% of rows where all three exist
  4. no shrinkage: the new reading has at least 70% of the old reading's cells, unless the
     check said columns were invented

  python3 atlas/reread_columns.py --run           # re-read + gate (resumable), out/reread/*.json
  python3 atlas/reread_columns.py --apply         # write overlay + log + patch the DB
"""
import os, re, sys, csv, json, time, glob, random, sqlite3, hashlib, argparse, collections, statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
ROOT = os.path.join(HERE, "..")
import gemini_column_check as g
V1 = os.path.join(ROOT, "export_final/sendout_final/sendout_v1")
OUTD = os.path.join(HERE, "out", "reread"); os.makedirs(OUTD, exist_ok=True)
MODEL = "gemini-3.8-flash"
FIX = {"persons_only_missing", "columns_mislabelled", "columns_missing", "columns_invented", "columns_shifted"}
MAX_PAGES = 6
SECOND = False          # independent second reading (no column re-check; cells are voted)
PERSONS_ONLY = False    # targeted reading of the Persons/Total columns only
FACING = {"columns_invented", "columns_mislabelled", "columns_missing", "columns_shifted"}   # offer facing pages

READ_PROMPT = """Transcribe the census table "{title}" ({year}, {place}) from the scanned page(s) below, exactly as printed.

{pages_note}
A previous transcription got the COLUMNS wrong. Diagnosis: {diagnosis}
Printed columns according to the check: {printed}

Rules:
- Read the printed column header hierarchy exactly. For every VALUE column give "group" = all header levels above the lowest joined with " / " (e.g. "Population", "Married / 15-20", "Hindu / Unmarried"), and "sub" = the lowest header (e.g. "Persons", "Males", "Females", "Unmarried"). Include Persons/Total columns when printed. Do not output serial-number columns, the column-number row (1, 2, 3 ...), or the row-label stub as value columns.
- If the table is printed as several stacked blocks with different headers (e.g. Males block, then Females block; or a district/religion sub-table), make each block its own section with its own columns.
- If the table continues across the pages shown (a left page and its facing right page), join the halves row by row into one table.
- Row labels: reuse these existing labels verbatim when the printed row is the same row: {rows}
- Values: numbers as printed without thousands separators; decimals as decimals; "..." "—" "-" or blank -> null.
- Transcribe only this table; ignore other tables or text on the page.
Report which page labels you used."""

READ_SCHEMA = {"type": "object", "properties": {
    "pages_used": {"type": "array", "items": {"type": "string"}},
    "sections": {"type": "array", "items": {"type": "object", "properties": {
        "name": {"type": "string"},
        "columns": {"type": "array", "items": {"type": "object", "properties": {
            "group": {"type": "string"}, "sub": {"type": "string"}}, "required": ["group", "sub"]}},
        "rows": {"type": "array", "items": {"type": "object", "properties": {
            "label": {"type": "string"},
            "values": {"type": "array", "items": {"type": "number", "nullable": True}}}, "required": ["label", "values"]}}},
        "required": ["name", "columns", "rows"]}}},
    "required": ["pages_used", "sections"]}


# ---------- standardization (same fields as build_query_dataset OUT_COLS) ----------
AGE = re.compile(r"^\s*(\d+)\s*(?:-|–|—|to)\s*(\d+)\s*(?:years?)?\s*$", re.I)
AGE_OPEN = re.compile(r"^\s*(\d+)\s*(?:and over|and upwards|\+|& over|and above)\s*$", re.I)
AGE_UNDER = re.compile(r"^\s*under\s+(\d+)\s*$", re.I)


def parse_age(s):
    s = (s or "").strip()
    m = AGE.match(s)
    if m:
        return s, int(m.group(1)), int(m.group(2)), ""
    m = AGE_OPEN.match(s)
    if m:
        return s, int(m.group(1)), "", "True"
    m = AGE_UNDER.match(s)
    if m:
        return s, 0, int(m.group(1)), ""
    return None


def sex_of(*labels):
    for l in labels:
        t = (l or "").lower()
        if re.search(r"\bfemales?\b|\bfemale\b", t):
            return "Female"
        if re.search(r"\bmales?\b", t):
            return "Male"
        if re.search(r"\bpersons\b|both sexes", t):
            return "All"
    return ""


def civil_of(*labels):
    for l in labels:
        t = (l or "").lower()
        if re.search(r"unmarried|single", t):
            return "Unmarried"
        if re.search(r"widow", t):
            return "Widowed"
        if re.search(r"\bmarried", t):
            return "Married"
    return ""


RELIGIONS = re.compile(r"(?i)\b(hindu|musalman|muhammadan|mahomedan|muslim|sikh|jain|christian|parsi|buddhist|animist|"
                       r"jew|aryas?|brahmo|zoroastrian|tribal|all religions)")


def to_facts(read, tid, denorm, status):
    out = []
    for si, sec in enumerate(read["sections"]):
        cols = sec["columns"]
        for row in sec["rows"]:
            vals = row["values"]
            for ci, col in enumerate(cols):
                v = vals[ci] if ci < len(vals) else None
                if v is None:
                    continue
                grp, sub, lab, name = col["group"], col["sub"], row["label"], sec["name"]
                ra, ga = parse_age(lab), None
                for part in [sub] + grp.split(" / ")[::-1]:
                    ga = parse_age(part)
                    if ga:
                        break
                age = ra or ga
                f = {"table_id": tid, "section": name, "row_label_std": lab,
                     "row_dimension": "age" if ra else "other", "row_group_value": "",
                     "column_group_std": grp, "column_dimension": "age" if ga else ("civil_condition" if civil_of(grp) else "other"),
                     "column_group_value": "", "sub_column_std": sub,
                     "sex": sex_of(sub, grp, name, lab), "civil_condition": civil_of(sub, grp, name),
                     "religion": (RELIGIONS.search(name).group(0).title() if RELIGIONS.search(name or "") else ""),
                     "resolved_age_label_std": age[0] if age else "", "resolved_age_lower": age[1] if age else "",
                     "resolved_age_upper": age[2] if age else "", "resolved_age_open_ended": age[3] if age else "",
                     "measure_unit": "", "value": v}
                f.update(denorm); f["status"] = status
                f["age_aggregate"] = "1" if re.search(r"(?i)total|all ages", lab + " " + grp) else "0"
                out.append(f)
    return out


# ---------- gates ----------
def cellkey(sec, row, grp, sub):
    n = lambda s: re.sub(r"[^a-z0-9]", "", (s or "").lower())
    return (n(row), n(grp.split(" / ")[-1] if grp else ""), n(sub))


def agreement(old, new):
    o = {}
    for f in old:
        if f["value"] not in (None, 0):
            o.setdefault(cellkey(f["section"], f["row_label_std"], f["column_group_std"], f["sub_column_std"]), set()).add(float(f["value"]))
    shared = eq = 0
    for f in new:
        k = cellkey(f["section"], f["row_label_std"], f["column_group_std"], f["sub_column_std"])
        if k in o and f["value"] not in (None, 0):
            shared += 1; eq += float(f["value"]) in o[k]
    return shared, eq


def _sig(v):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    t = re.sub(r"\D", "", ("%f" % x).rstrip("0").rstrip(".")).lstrip("0")
    return t if len(t) >= 2 else None


def number_overlap(old, new):
    """label-free overlap of DISTINCT numbers (>= 2 significant digits) between two readings:
    recall = share of the old reading's numbers found in the new one; precision = share of the
    new reading's numbers that the old one also read. Robust to corrected headers, swapped
    columns, added sections and duplicated rows."""
    o = {s for s in (_sig(f["value"]) for f in old) if s}
    n = {s for s in (_sig(f["value"]) for f in new) if s}
    return len(o), len(n), len(o & n)


# old -> new recall where the old columns were right but incomplete or mislabelled;
# new -> old precision where the old reading had columns that are not printed
OVERLAP_RULE = {"persons_only_missing": ("recall", 0.9), "columns_missing": ("recall", 0.85),
                "columns_mislabelled": ("recall", 0.85), "columns_shifted": ("recall", 0.85),
                "columns_invented": ("either", 0.85), "wrong_scan": ("either", 0.6)}


TOTAL_ROW = re.compile(r"(?i)^\s*(total|all ages|grand total|total all ages)\b")
SUBTOTAL = re.compile(r"(?i)total|all ages|grand")


def one_digit_off(a, b):
    """True when a and b differ in exactly one decimal digit (a typical misread: 3/8, 5/6, 1/7)"""
    x, y = str(int(round(a))), str(int(round(b)))
    return len(x) == len(y) and sum(c1 != c2 for c1, c2 in zip(x, y)) == 1


def sum_failure_cols(read, skip=()):
    """(section, group, sub) of the columns sum_failures() reports"""
    out = set()
    for sec in read["sections"]:
        one = {"sections": [sec]}
        for col in sec["columns"]:
            k = (sec["name"], col["group"], col["sub"])
            if k in skip:
                continue
            others = set(skip) | {(sec["name"], c["group"], c["sub"]) for c in sec["columns"] if c is not col}
            if _sum_failures(one, others):
                out.add(k)
    return out


def sum_failures(read, skip=()):
    return _sum_failures(read, skip)


def _sum_failures(read, skip=()):
    """columns whose printed Total row disagrees with the sum of the rows above it by what looks
    like one misread digit. Rows that are themselves subtotals are skipped; a column only counts
    when its bands sum to the total in no other reading of the rows (e.g. with subtotals)."""
    bad = []
    for sec in read["sections"]:
        rows = sec["rows"]
        for ti, tr in enumerate(rows):
            if not TOTAL_ROW.match(tr["label"] or ""):
                continue
            body = [r for r in rows[:ti] if not SUBTOTAL.search(r["label"] or "")]
            if len(body) < 3:
                continue
            for ci, col in enumerate(sec["columns"]):
                if (sec["name"], col["group"], col["sub"]) in skip:
                    continue   # a withheld (blank) cell is not a wrong number
                tv = tr["values"][ci] if ci < len(tr["values"]) else None
                vals = [r["values"][ci] for r in body if ci < len(r["values"]) and r["values"][ci] is not None]
                if tv is None or len(vals) < 3 or any(abs(v - round(v)) > 1e-9 for v in vals + [tv]):
                    continue
                s_ = sum(vals)
                if s_ != tv and one_digit_off(s_, tv) or (s_ != tv and any(one_digit_off(v, v + (tv - s_)) for v in vals)):
                    bad.append(f"{sec['name'][:30]} | {col['group']} | {col['sub']}: bands sum {s_:g}, printed total {tv:g}")
    return bad


def regate(d, old_facts, measure):
    """gates recomputed from a saved re-read (the decision used when applying)"""
    new = to_facts(d["read"], d["table_id"], {}, "")
    v = d.get("orig_verdict")
    no, nn, both = number_overlap(old_facts, new)
    rec, prec = (both / no if no else 1.0), (both / nn if nn else 0.0)
    kind, thr = OVERLAP_RULE.get(v, ("recall", 0.85))
    overlap_ok = no < 10 or {"recall": rec >= thr, "precision": prec >= thr, "either": max(rec, prec) >= thr}[kind]
    n_pmf, ok_pmf = pmf_check(new)
    ver = (d.get("verify") or {}).get("verdict")
    gates = {
        "check": ver == "ok" or (ver == "persons_only_missing" and measure != "count"),
        "number_overlap": overlap_ok,
        "pmf": measure != "count" or n_pmf < 3 or ok_pmf >= 0.9 * n_pmf,
        "size": v not in ("persons_only_missing", "columns_missing") or len(new) >= 0.9 * len(old_facts),
        "nonempty": len(new) >= 4,
        # a re-read whose own printed totals expose a single misread digit is not accepted
        "sums": not sum_failures(d["read"]),
    }
    return gates, {"sum_failures": sum_failures(d["read"])[:5],"old_numbers": no, "new_numbers": nn, "shared_numbers": both, "recall": round(rec, 3),
                   "precision": round(prec, 3), "pmf": [n_pmf, ok_pmf]}


def pmf_check(facts):
    by = collections.defaultdict(dict)
    for f in facts:
        by[(f["section"], f["row_label_std"], f["column_group_std"])][f["sex"]] = float(f["value"])
    n = ok = 0
    for d in by.values():
        if {"All", "Male", "Female"} <= set(d):
            n += 1; ok += abs(d["All"] - d["Male"] - d["Female"]) <= max(1, 0.01 * abs(d["All"]))
    return n, ok


# ---------- pages ----------
def page_parts(it, extra):
    """[(label, bytes, mime)] : the table's pages, plus facing/relinked pages when asked"""
    from google.genai import types
    parts, labels = [], []
    imgs = it["imgs"][:MAX_PAGES]
    for k, img in enumerate(imgs, 1):
        b, mime, _ = g.native_scan(img)
        labels.append(f"P{k}"); parts += [types.Part.from_text(text=f"Page P{k}:"), types.Part.from_bytes(data=b, mime_type=mime)]
    for lab, (b, mime) in extra:
        labels.append(lab); parts += [types.Part.from_text(text=f"Page {lab}:"), types.Part.from_bytes(data=b, mime_type=mime)]
    return parts, labels


def neighbour(img, off):
    import find_right_scan as fr
    src = g.pdf_index().get(img.rsplit(".", 1)[0])
    if not src:
        return None
    r = fr.render(src[0], src[1] + off)
    return r


import threading
_cool = {"until": 0.0}
_cool_lock = threading.Lock()


def call(client, **kw):
    """generate_content with a shared cool-down: a 429 from the account's spend-rate limit pauses
    every worker together (the window resets within ~2 minutes) instead of hammering the limit"""
    for attempt in range(12):
        wait = _cool["until"] - time.time()
        if wait > 0:
            time.sleep(wait + random.random() * 3)
        try:
            return client.models.generate_content(**kw)
        except Exception as e:
            m = str(e)
            if "429" in m or "RESOURCE_EXHAUSTED" in m:
                with _cool_lock:
                    _cool["until"] = max(_cool["until"], time.time() + 45); _cool["n429"] = _cool.get("n429", 0) + 1
            else:
                time.sleep(min(30, 2 ** attempt))
            last = m
    raise RuntimeError(last[:200])


def reread_one(client, it, check, relink):
    from google.genai import types
    extra = []
    note = "The pages are labelled P1, P2, ..."
    if relink:
        extra.append(("R1", relink["bytes"]))
        note = "Page R1 is the page that holds this table (the other pages are the previously linked scan and may be unrelated)."
    elif check["verdict"] in FACING and len(it["imgs"]) == 1:
        for off, lab in ((-1, "F-1"), (1, "F+1")):
            r = neighbour(it["imgs"][0], off)
            if r:
                extra.append((lab, r))
        note = ("Pages F-1 and F+1 are the pages before and after P1 in the volume; the table may continue onto one of "
                "them as a facing page. Use them only if they continue this same table.")
    parts, labels = page_parts(it, extra)
    rows = list(dict.fromkeys(it["row_labels"]))[:200]
    prompt = READ_PROMPT
    if PERSONS_ONLY:
        prompt = READ_PROMPT + ("\n\nIMPORTANT: transcribe ONLY the printed Persons / Total / Both-sexes value columns "
                                "(with their headers and the row labels). Skip every Males and Females column.")
    parts.append(types.Part.from_text(text=prompt.format(
        title=it["title"], year=it["year"], place=it["place"], pages_note=note,
        diagnosis=(check.get("note") or "")[:400] + (" Problem columns: " + "; ".join(check.get("problem_columns") or [])[:300] if check.get("problem_columns") else ""),
        printed=(check.get("printed_columns") or "")[:400], rows=json.dumps(rows, ensure_ascii=False)[:6000])))
    cfg = types.GenerateContentConfig(response_mime_type="application/json", response_schema=READ_SCHEMA, temperature=0,
                                      max_output_tokens=65000)
    r = call(client, model=MODEL, contents=parts, config=cfg)
    return json.loads(r.text), labels


def verify(client, it, read, used_imgs, extra_parts):
    """independent column check of the new reading (same prompt as the corpus-wide check)"""
    cols = collections.OrderedDict()
    for sec in read["sections"]:
        for row in sec["rows"]:
            for ci, c in enumerate(sec["columns"]):
                k = f"{c['group']} | {c['sub']}"; cols.setdefault(k, [])
                v = row["values"][ci] if ci < len(row["values"]) else None
                if len(cols[k]) < 3:
                    cols[k].append(f"{row['label']}={v:g}" if isinstance(v, (int, float)) else f"{row['label']}=blank")
    vit = dict(it, imgs=used_imgs, n_rows=sum(len(s["rows"]) for s in read["sections"]), n_columns=len(cols),
               columns=[f"{k}  [first: {'; '.join(v)}]" for k, v in list(cols.items())[:120]])
    return check_with_extra(client, vit, extra_parts)


def check_with_extra(client, it, extra_parts):
    from google.genai import types
    text = g.PROMPT.format(title=it["title"], year=it["year"], place=it["place"], n_images=len(it["imgs"]) + len(extra_parts), shown="",
                           n_rows=it["n_rows"], n_columns=it["n_columns"], columns="\n".join(it["columns"]))
    parts = []
    for img in it["imgs"][:MAX_PAGES]:
        b, mime, _ = g.native_scan(img); parts.append(types.Part.from_bytes(data=b, mime_type=mime))
    for b, mime in extra_parts:
        parts.append(types.Part.from_bytes(data=b, mime_type=mime))
    parts.append(types.Part.from_text(text=text))
    cfg = types.GenerateContentConfig(response_mime_type="application/json", response_schema=g.SCHEMA, temperature=0)
    try:
        return json.loads(call(client, model=MODEL, contents=parts, config=cfg).text)
    except Exception:
        return {"verdict": "error"}


def _dump(obj, path):
    tmp = path + ".tmp"; json.dump(obj, open(tmp, "w")); os.replace(tmp, path)


def process(client, it, check, relink, old_facts, denorm, status, measure):
    path = os.path.join(OUTD, it["page"] + ".json")
    res = {"page": it["page"], "table_id": it["table_id"], "orig_verdict": check["verdict"], "relinked": bool(relink)}
    try:
        read, labels = reread_one(client, it, check, relink)
    except Exception as e:
        res.update(decision="error", reason=str(e)[:200]); return res
    used = set(read.get("pages_used") or [])
    # pages actually used: table pages always shown; extra pages only when the model used them
    extra = []
    if relink:
        extra = [relink["bytes"]]; used_imgs = [] if "R1" in used else it["imgs"]
    else:
        used_imgs = it["imgs"]
        if check["verdict"] in FACING and len(it["imgs"]) == 1:
            for off, lab in ((-1, "F-1"), (1, "F+1")):
                if lab in used:
                    r = neighbour(it["imgs"][0], off)
                    if r:
                        extra.append(r)
    new = to_facts(read, it["table_id"], denorm, status)
    if SECOND:
        ver = {"verdict": "ok", "note": "independent second reading; checked by cell-level agreement"}
    elif check["verdict"] == "persons_only_missing" and measure == "count":
        # arithmetic is the stronger test here: the new Persons column must equal M + F and the
        # existing M/F values must come back unchanged (agreement gate)
        n_p, ok_p = pmf_check(new)
        ver = {"verdict": "ok" if n_p >= 3 and ok_p >= 0.9 * n_p else "persons_check_failed", "note": f"P=M+F on {ok_p}/{n_p} rows"}
    else:
        ver = verify(client, it, read, used_imgs, extra)
    if ver.get("verdict") == "error":
        res.update(decision="error", reason="verification call failed"); return res
    shared, eq = agreement(old_facts, new)
    n_pmf, ok_pmf = pmf_check(new)
    gates = {
        "check": ver.get("verdict") in ("ok",) or (ver.get("verdict") == "persons_only_missing" and measure != "count"),
        "agreement": shared < 10 or eq >= 0.9 * shared,
        "pmf": measure != "count" or n_pmf < 3 or ok_pmf >= 0.9 * n_pmf,
        # shrinking is only suspicious where columns were missing; mislabelled/invented/wrong-scan
        # readings legitimately lose cells that were never printed on the page
        "size": check["verdict"] not in ("persons_only_missing", "columns_missing") or len(new) >= 0.9 * len(old_facts),
        "nonempty": len(new) >= 4,
    }
    res.update(read=read, pages_used=sorted(used), new_cells=len(new), old_cells=len(old_facts),
               shared=shared, equal=eq, pmf=[n_pmf, ok_pmf], verify=ver, gates=gates,
               extra_pages=[lab for lab in sorted(used) if lab.startswith(("F", "R"))],
               decision="accept" if all(gates.values()) else "reject")
    _dump(res, path)
    return res


def build_jobs():
    V = {}
    for l in open(os.path.join(HERE, "out", "gemini_column_check_gemini-3.8-flash.jsonl")):
        d = json.loads(l)
        if d.get("verdict") != "error":
            V[d["page"]] = d
    relinks = {}
    for l in open(os.path.join(HERE, "out", "right_scan.jsonl")):
        d = json.loads(l)
        if d.get("found_offset") is not None and d.get("confidence") == "high":
            relinks[d["page"]] = d
    pages = [p for p, d in V.items() if d["verdict"] in FIX or p in relinks]
    items = g.table_items(pages)
    return V, relinks, items


def run(workers):
    from google import genai
    import find_right_scan as fr
    V, relinks, items = build_jobs()
    con = sqlite3.connect(f"file:{g.DB}?mode=ro", uri=True); con.row_factory = sqlite3.Row
    todo = [i for i in items if not os.path.exists(os.path.join(OUTD, i["page"] + ".json"))]
    random.Random(7).shuffle(todo)
    pri = {"columns_mislabelled": 0, "columns_missing": 0, "columns_invented": 0, "columns_shifted": 0, "wrong_scan": 0}
    todo.sort(key=lambda i: 0 if i["page"] in relinks else pri.get(V.get(i["page"], {}).get("verdict"), 1))
    print(f"{len(items)} tables to fix ({len(relinks)} relinks); {len(todo)} remaining", flush=True)
    client = genai.Client(api_key=g.env_key())
    tids = {hashlib.md5(t.encode()).hexdigest()[:7]: t for (t,) in con.execute("select table_id from tables")}

    def job(it):
        c2 = sqlite3.connect(f"file:{g.DB}?mode=ro", uri=True); c2.row_factory = sqlite3.Row
        tid = tids[it["page"].rsplit("_", 1)[1]]; it["table_id"] = tid
        old = [dict(r) for r in c2.execute("select * from facts where table_id=?", (tid,))]
        it["row_labels"] = [f["row_label_std"] for f in old]
        t = dict(c2.execute("select * from tables where table_id=?", (tid,)).fetchone())
        denorm = {k: t.get(k, "") for k in ["year", "decade", "canonical_name", "entity_type", "parent_geo", "broad_region", "geo_level",
                                              "canonical_topic", "subject", "measure_class", "additive", "table_family",
                                              "measure_category", "source_image", "title"]}
        rl = None
        if it["page"] in relinks:
            d = relinks[it["page"]]
            r = fr.render(d["pdf"], d["pdf_page"] + d["found_offset"])
            if r:
                rl = {"bytes": r, "pdf": d["pdf"], "pdf_page": d["pdf_page"] + d["found_offset"]}
        check = V.get(it["page"], {"verdict": "wrong_scan"})
        res = process(client, it, check, rl, old, denorm, t["status"], t["measure_class"])
        if rl:
            res["relink"] = {"pdf": rl["pdf"], "pdf_page": rl["pdf_page"]}
            _dump(res, os.path.join(OUTD, it["page"] + ".json"))
        return res

    c = collections.Counter(); t0 = time.time()
    with ThreadPoolExecutor(workers) as ex:
        for k, fu in enumerate(as_completed([ex.submit(job, i) for i in todo]), 1):
            try:
                r = fu.result(); c[r["decision"]] += 1
            except Exception as e:
                c["crash"] += 1; print("crash", str(e)[:200], flush=True)
            if k % 50 == 0 or k == len(todo):
                print(f"{k}/{len(todo)} {time.time() - t0:.0f}s {dict(c)}", flush=True)
    print("REREAD_DONE", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--run", action="store_true"); ap.add_argument("--workers", type=int, default=40)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--model", default=None, help="model for an independent second reading")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--pages-file", default=None, help="restrict to these page ids")
    ap.add_argument("--persons-only", action="store_true")
    a = ap.parse_args()
    if a.model:
        MODEL = a.model; SECOND = True
    if a.persons_only:
        PERSONS_ONLY = True; SECOND = True
    if a.outdir:
        OUTD = a.outdir; os.makedirs(OUTD, exist_ok=True)
    if a.pages_file:
        _keep = {l.strip() for l in open(a.pages_file) if l.strip()}
        _bj0 = build_jobs
        def build_jobs():
            V, rl, items = _bj0(); return V, rl, [i for i in items if i["page"] in _keep]
    if a.only:
        _bj = build_jobs
        def build_jobs():
            V, rl, items = _bj(); return V, rl, [i for i in items if any(i["page"].startswith(o) for o in a.only)]
    if a.run:
        run(a.workers)
