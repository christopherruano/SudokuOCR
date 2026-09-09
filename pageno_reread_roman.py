#!/usr/bin/env python3
"""Roman-aware re-read of the scans the Arabic-only pass left null (~282).

Census supplementary/subsidiary tables are paginated in ROMAN numerals (xiii, xlii,
lvii ...). Their titles are ALSO Roman ("TABLE VIII", "SUPPLEMENTARY TABLE 6"), so a
model can confuse the table number with the page number. To guard against that we make
Gemini report, separately:
  - the standalone PAGE numeral it sees (roman or arabic),
  - the TITLE text and the table-number-in-title,
so we (and the human reviewer) can cross-check page_number != table_number.

Reads the FULL page (these are often landscape supplementary tables where the page
number sits at a rotated edge), with a top-strip retry. Output is used as SUGGESTIONS
in the review app, not auto-applied.

Input : pageno_parts/gemini_reads.jsonl  (entries with page_number == null)
Output: pageno_parts/roman_reads.jsonl
Usage : python3 pageno_reread_roman.py [--workers 30] [--limit N]
"""
import argparse, io, json, os, re, sys, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
from PIL import Image

ROOT = os.path.dirname(os.path.abspath(__file__))
V1 = os.path.join(ROOT, "export_final/sendout_final/sendout_v1")
PARTS = os.path.join(V1, "pageno_parts")
JSONL = os.path.join(PARTS, "gemini_reads.jsonl")
OUT = os.path.join(PARTS, "roman_reads.jsonl")

load_dotenv(os.path.join(ROOT, ".env"))
MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.1-pro-preview")
BASE_URL = os.environ.get("HARVARD_GEMINI_BASE_URL") or os.environ.get("GEMINI_BASE_URL") or ""

PROMPT = (
    "This is one scanned page from a historical printed Indian Census volume (1872-1941). "
    "Find the printed PAGE NUMBER. It may be a ROMAN numeral (e.g. xiii, xlii, lvii) — common "
    "on supplementary/subsidiary tables — OR Arabic (e.g. 42). It is a STANDALONE numeral, "
    "usually centered at the very top of the page or in a top outer corner. On wide/rotated "
    "(landscape) tables it sits at the rotated top edge.\n"
    "CRITICAL: do NOT confuse the page number with the TABLE number printed inside a title such "
    "as 'TABLE VIII', 'SUPPLEMENTARY TABLE 6', 'TABLE C. IV', or 'Imperial Table VII'. The table "
    "number is part of a title; the page number stands alone. Also ignore the census year (1901 etc).\n"
    "Return STRICT JSON only:\n"
    '{"page_number_raw": <exact numeral as printed, e.g. "xlii" or "42", or null>, '
    '"numeral_system": "roman"|"arabic"|null, '
    '"page_number_int": <integer value or null>, '
    '"title_text": <short header/title text you see, or null>, '
    '"table_number_in_title": <e.g. "VIII" or "C. IV", or null>, '
    '"confidence": "high"|"medium"|"low"}\n'
    "Report page_number only if you actually see a standalone page numeral. If you cannot, use null. "
    "Never copy the table number into the page number."
)

ROMAN = {"m": 1000, "d": 500, "c": 100, "l": 50, "x": 10, "v": 5, "i": 1}


def roman_to_int(s):
    s = (s or "").strip().lower()
    if not s or any(ch not in ROMAN for ch in s):
        return None
    tot, prev = 0, 0
    for ch in reversed(s):
        v = ROMAN[ch]
        tot += -v if v < prev else v
        prev = max(prev, v)
    return tot if 0 < tot < 4000 else None


class KeyState:
    def __init__(self, key, slot):
        self.key, self.slot = key, slot
        self.lock = threading.Lock()
        self.last_call = 0.0


def load_keys():
    multi = os.environ.get("HARVARD_GEMINI_API_KEYS", "")
    keys = [k.strip() for k in multi.split(",") if k.strip()]
    if not keys and os.environ.get("HARVARD_GEMINI_API_KEY"):
        keys = [os.environ["HARVARD_GEMINI_API_KEY"].strip()]
    if not keys and os.environ.get("GEMINI_API_KEY"):
        keys = [os.environ["GEMINI_API_KEY"].strip()]
    if not keys:
        sys.exit("no keys")
    return [KeyState(k, i) for i, k in enumerate(keys)]


def make_client(key):
    from google import genai
    from google.genai import types
    http = {"timeout": 300_000 if BASE_URL else 120_000}
    if BASE_URL:
        http["base_url"] = BASE_URL
    return genai.Client(api_key=key, http_options=types.HttpOptions(**http)), types


def img_png(path, mode, max_w=1500):
    im = Image.open(path).convert("RGB")
    if mode == "top":
        w, h = im.size
        im = im.crop((0, 0, w, max(1, int(h * 0.34))))
    if im.width > max_w:
        im = im.resize((max_w, int(im.height * max_w / im.width)))
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


def parse_json(t):
    if not t:
        return None
    t = re.sub(r"^```(?:json)?", "", t.strip()).strip()
    t = re.sub(r"```$", "", t).strip()
    m = re.search(r"\{.*\}", t, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def coerce(d):
    if not isinstance(d, dict):
        return None
    raw = d.get("page_number_raw")
    sysn = (d.get("numeral_system") or "").lower()
    pi = d.get("page_number_int")
    if isinstance(pi, str):
        pi = int(pi) if pi.strip().isdigit() else None
    if isinstance(pi, float):
        pi = int(pi)
    # recompute from raw to be safe
    if raw:
        rs = str(raw).strip().strip(".[]() ")
        if re.fullmatch(r"\d{1,4}", rs):
            pi, sysn = int(rs), "arabic"
        else:
            rv = roman_to_int(rs)
            if rv is not None:
                pi, sysn = rv, "roman"
    if not isinstance(pi, int) or not (0 < pi < 5000):
        pi = None
    return {
        "page_number_int": pi,
        "page_number_raw": raw,
        "numeral_system": sysn if pi is not None else None,
        "title_text": d.get("title_text"),
        "table_number_in_title": d.get("table_number_in_title"),
        "confidence": d.get("confidence") or "",
    }


def call(client, types, img_bytes, ks, min_interval, max_attempts=4):
    cfg = types.GenerateContentConfig(temperature=0.0, response_mime_type="application/json")
    part = types.Part.from_bytes(data=img_bytes, mime_type="image/png")
    last = ""
    for attempt in range(max_attempts):
        with ks.lock:
            if min_interval > 0 and ks.last_call:
                w = min_interval - (time.time() - ks.last_call)
                if w > 0:
                    time.sleep(w)
            ks.last_call = time.time()
        try:
            resp = client.models.generate_content(model=MODEL, contents=[part, PROMPT], config=cfg)
            return resp.text or ""
        except Exception as exc:
            last = f"{type(exc).__name__}: {exc}"
            if not any(t in last for t in ["429", "500", "502", "503", "504", "UNAVAILABLE", "RESOURCE_EXHAUSTED", "INTERNAL", "DEADLINE"]) or attempt == max_attempts - 1:
                raise
            time.sleep(min(60, 4 * (2 ** attempt)))
    raise RuntimeError(last)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=30)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--min-interval", type=float, default=0.2)
    args = ap.parse_args()

    reads = {}
    for line in open(JSONL):
        line = line.strip()
        if line:
            r = json.loads(line)
            reads[r["imghash"]] = r
    nulls = [r for r in reads.values() if r.get("page_number") is None]

    done = set()
    if os.path.exists(OUT):
        for line in open(OUT):
            try:
                done.add(json.loads(line)["imghash"])
            except Exception:
                pass
    todo = [r for r in nulls if r["imghash"] not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"model={MODEL} base_url={'set' if BASE_URL else 'NONE'}  nulls={len(nulls)} todo={len(todo)} done={len(done)}", flush=True)
    if not todo:
        print("nothing to do")
        return

    keys = load_keys()
    clients = {ks.slot: make_client(ks.key) for ks in keys}
    wl = threading.Lock()
    cnt = {"n": 0, "roman": 0, "arabic": 0, "null": 0, "err": 0, "fb": 0}
    t0 = time.time()
    fh = open(OUT, "a")

    def work(i, r):
        ks = keys[i % len(keys)]
        client, types = clients[ks.slot]
        scan = os.path.join(ROOT, r["scan_path"])
        try:
            raw = call(client, types, img_png(scan, "full"), ks, args.min_interval)
            p = coerce(parse_json(raw))
            fb = False
            if not p or p.get("page_number_int") is None:
                fb = True
                raw2 = call(client, types, img_png(scan, "top"), ks, args.min_interval)
                p2 = coerce(parse_json(raw2))
                if p2 and p2.get("page_number_int") is not None:
                    p = p2
            if not p:
                p = {"page_number_int": None, "page_number_raw": None, "numeral_system": None,
                     "title_text": None, "table_number_in_title": None, "confidence": ""}
            return {"imghash": r["imghash"], "scan_path": r["scan_path"],
                    "asm_ids": r.get("asm_ids", []), "table_ids": r.get("table_ids", []),
                    "used_fallback": fb, **p}, None
        except Exception as exc:
            return {"imghash": r["imghash"], "scan_path": r["scan_path"], "asm_ids": r.get("asm_ids", []),
                    "table_ids": r.get("table_ids", []), "page_number_int": None,
                    "error": f"{type(exc).__name__}: {exc}"[:200]}, exc

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(work, i, r): r for i, r in enumerate(todo)}
        for fut in as_completed(futs):
            rec, _ = fut.result()
            with wl:
                fh.write(json.dumps(rec) + "\n"); fh.flush()
                cnt["n"] += 1
                if rec.get("error"):
                    cnt["err"] += 1
                elif rec.get("page_number_int") is None:
                    cnt["null"] += 1
                else:
                    cnt[rec.get("numeral_system") or "arabic"] = cnt.get(rec.get("numeral_system") or "arabic", 0) + 1
                if rec.get("used_fallback"):
                    cnt["fb"] += 1
                n = cnt["n"]
                if n % 25 == 0 or n == len(todo):
                    el = time.time() - t0
                    print(f"  {n}/{len(todo)} roman={cnt.get('roman',0)} arabic={cnt.get('arabic',0)} "
                          f"null={cnt['null']} err={cnt['err']} fb={cnt['fb']} ETA {((len(todo)-n)/(n/el))/60:.1f}m", flush=True)
    fh.close()
    print(f"DONE in {(time.time()-t0)/60:.1f}m  roman={cnt.get('roman',0)} arabic={cnt.get('arabic',0)} null={cnt['null']} err={cnt['err']}")


if __name__ == "__main__":
    main()
