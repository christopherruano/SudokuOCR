#!/usr/bin/env python3
"""Read the printed top-of-page page number from each unique scan using Harvard Gemini,
in parallel across all configured keys. Resumable; checkpoints to JSONL.

Input : export_final/sendout_final/sendout_v1/pageno_parts/worklist.json
Output: export_final/sendout_final/sendout_v1/pageno_parts/gemini_reads.jsonl   (one line per scan)

Strategy for accuracy:
- Crop the TOP strip of the page (the running head where the printed page number lives),
  at full resolution, so the small numeral stays legible. Send that to Gemini first.
- If Gemini can't see a page number in the strip, retry once with the FULL image.
- temperature 0, strict JSON. Page number only (ignore "TABLE VII" roman numerals & years).

Usage:
  python3 pageno_read_gemini.py [--limit N] [--workers W] [--only-review] [--strip-frac 0.30]
"""
import argparse, base64, io, json, os, re, sys, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
from PIL import Image

ROOT = os.path.dirname(os.path.abspath(__file__))
V1 = os.path.join(ROOT, "export_final/sendout_final/sendout_v1")
PARTS = os.path.join(V1, "pageno_parts")
WORKLIST = os.path.join(PARTS, "worklist.json")
OUT_JSONL = os.path.join(PARTS, "gemini_reads.jsonl")

load_dotenv(os.path.join(ROOT, ".env"))
MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.1-pro-preview")
BASE_URL = os.environ.get("HARVARD_GEMINI_BASE_URL") or os.environ.get("GEMINI_BASE_URL") or ""

PROMPT = (
    "This is a scanned page from a historical printed Indian Census volume (1872-1941). "
    "I am showing you the TOP STRIP of the page (the running head / header area). "
    "Read ONLY the printed PAGE NUMBER that appears in the header — an Arabic numeral "
    "(e.g. 140, 141), usually in a top corner or centered at the top. "
    "Do NOT report the table number (e.g. the Roman numerals in 'TABLE VII'), the census "
    "year (e.g. 1901), province numbers, or any number from inside the data table. "
    "If a continuation word like 'concluded', 'contd', 'continued', or a marker like '(2)' "
    "appears near the header/title, report it in continuation_marker. "
    "Return STRICT JSON only, no prose:\n"
    '{"page_number": <integer or null>, "page_visible": <true/false>, '
    '"continuation_marker": <string or null>, "confidence": "high"|"medium"|"low"}\n'
    "page_visible=true only if you can actually see a printed page number. "
    "If no page number is legible, page_number=null and page_visible=false. "
    "Never guess a page number."
)

FULL_PROMPT = PROMPT.replace(
    "I am showing you the TOP STRIP of the page (the running head / header area). ",
    "I am showing you the FULL page. The printed page number is in the header at the very TOP. ",
)


class KeyState:
    def __init__(self, key, slot):
        self.key = key
        self.slot = slot
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
        sys.exit("No Gemini keys found in .env")
    return [KeyState(k, i) for i, k in enumerate(keys)]


def make_client(key):
    from google import genai
    from google.genai import types
    http = {"timeout": 300_000 if BASE_URL else 120_000}
    if BASE_URL:
        http["base_url"] = BASE_URL
    return genai.Client(api_key=key, http_options=types.HttpOptions(**http)), types


def top_strip_png(path, frac):
    im = Image.open(path).convert("RGB")
    w, h = im.size
    strip = im.crop((0, 0, w, max(1, int(h * frac))))
    buf = io.BytesIO()
    strip.save(buf, format="PNG")
    return buf.getvalue()


def full_png(path, max_w=1400):
    im = Image.open(path).convert("RGB")
    if im.width > max_w:
        im = im.resize((max_w, int(im.height * max_w / im.width)))
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


def parse_json(text):
    if not text:
        return None
    t = text.strip()
    # strip code fences if present
    t = re.sub(r"^```(?:json)?", "", t).strip()
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
    pn = d.get("page_number")
    if isinstance(pn, str):
        s = pn.strip()
        pn = int(s) if s.isdigit() else None
    if isinstance(pn, float):
        pn = int(pn)
    if not isinstance(pn, int):
        pn = None
    if pn is not None and not (0 < pn < 5000):  # sanity: census page numbers
        pn = None
    return {
        "page_number": pn,
        "page_visible": bool(d.get("page_visible")) and pn is not None,
        "continuation_marker": (d.get("continuation_marker") or None),
        "confidence": d.get("confidence") or "",
    }


def call(client, types, prompt, img_bytes, ks, min_interval, max_attempts=4):
    cfg = types.GenerateContentConfig(temperature=0.0, response_mime_type="application/json")
    part = types.Part.from_bytes(data=img_bytes, mime_type="image/png")
    last = ""
    for attempt in range(max_attempts):
        with ks.lock:
            if min_interval > 0 and ks.last_call:
                wait = min_interval - (time.time() - ks.last_call)
                if wait > 0:
                    time.sleep(wait)
            ks.last_call = time.time()
        try:
            resp = client.models.generate_content(model=MODEL, contents=[part, prompt], config=cfg)
            return resp.text or ""
        except Exception as exc:
            last = f"{type(exc).__name__}: {exc}"
            transient = any(tok in last for tok in
                            ["429", "500", "502", "503", "504", "UNAVAILABLE", "RESOURCE_EXHAUSTED", "INTERNAL", "DEADLINE"])
            if not transient or attempt == max_attempts - 1:
                raise
            time.sleep(min(60, 4 * (2 ** attempt)))
    raise RuntimeError(last or "gemini failed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=30)
    ap.add_argument("--only-review", action="store_true")
    ap.add_argument("--strip-frac", type=float, default=0.30)
    ap.add_argument("--min-interval", type=float, default=0.25)
    args = ap.parse_args()

    worklist = json.load(open(WORKLIST))

    if args.only_review:
        import csv
        verd = list(csv.DictReader(open(os.path.join(V1, "tableVII_verdicts.csv"))))
        members = json.load(open(os.path.join(V1, "assembled_members.json")))
        review_asm = {r["table_id"] for r in verd if r["status"] in {"unrecoverable", "retrying"} and r["table_id"] in members}
        worklist = [w for w in worklist if set(w["asm_ids"]) & review_asm]

    # resume: skip imghashes already in JSONL
    done = set()
    if os.path.exists(OUT_JSONL):
        for line in open(OUT_JSONL):
            try:
                done.add(json.loads(line)["imghash"])
            except Exception:
                pass
    todo = [w for w in worklist if w["imghash"] not in done]
    if args.limit:
        todo = todo[: args.limit]

    print(f"model={MODEL}  base_url={'set' if BASE_URL else 'NONE'}", flush=True)
    keys = load_keys()
    print(f"keys={len(keys)}  workers={args.workers}  todo={len(todo)}  (already done={len(done)})", flush=True)
    if not todo:
        print("nothing to do.")
        return

    # one client per key (thread-safe enough; calls serialized per-key by lock for timing only)
    clients = {}
    for ks in keys:
        clients[ks.slot] = make_client(ks.key)

    write_lock = threading.Lock()
    counter = {"n": 0, "vis": 0, "null": 0, "err": 0, "fallback": 0}
    t0 = time.time()
    fh = open(OUT_JSONL, "a")

    def work(idx, w):
        ks = keys[idx % len(keys)]
        client, types = clients[ks.slot]
        scan = os.path.join(ROOT, w["scan_path"])
        try:
            strip = top_strip_png(scan, args.strip_frac)
            raw = call(client, types, PROMPT, strip, ks, args.min_interval)
            parsed = coerce(parse_json(raw))
            used_fallback = False
            if not parsed or parsed.get("page_number") is None:
                # fallback: full image, once
                used_fallback = True
                fb = full_png(scan)
                raw2 = call(client, types, FULL_PROMPT, fb, ks, args.min_interval)
                p2 = coerce(parse_json(raw2))
                if p2 and p2.get("page_number") is not None:
                    parsed = p2
            if not parsed:
                parsed = {"page_number": None, "page_visible": False, "continuation_marker": None, "confidence": ""}
            rec = {"imghash": w["imghash"], "scan_path": w["scan_path"],
                   "table_ids": w["table_ids"], "asm_ids": w["asm_ids"],
                   "used_fallback": used_fallback, **parsed}
            return rec, None
        except Exception as exc:
            return {"imghash": w["imghash"], "scan_path": w["scan_path"], "table_ids": w["table_ids"],
                    "asm_ids": w["asm_ids"], "page_number": None, "page_visible": False,
                    "continuation_marker": None, "confidence": "", "error": f"{type(exc).__name__}: {exc}"[:200]}, exc

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(work, i, w): w for i, w in enumerate(todo)}
        for fut in as_completed(futs):
            rec, err = fut.result()
            with write_lock:
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                counter["n"] += 1
                if rec.get("error"):
                    counter["err"] += 1
                elif rec.get("page_number") is not None:
                    counter["vis"] += 1
                else:
                    counter["null"] += 1
                if rec.get("used_fallback"):
                    counter["fallback"] += 1
                n = counter["n"]
                if n % 50 == 0 or n == len(todo):
                    el = time.time() - t0
                    rate = n / el if el else 0
                    eta = (len(todo) - n) / rate if rate else 0
                    print(f"  {n}/{len(todo)}  vis={counter['vis']} null={counter['null']} "
                          f"err={counter['err']} fb={counter['fallback']}  "
                          f"{rate:.1f}/s  ETA {eta/60:.1f}m", flush=True)
    fh.close()
    el = time.time() - t0
    print(f"\nDONE {counter['n']} scans in {el/60:.1f}m  "
          f"({counter['n']/el:.2f}/s)  vis={counter['vis']} null={counter['null']} err={counter['err']}")


if __name__ == "__main__":
    main()
