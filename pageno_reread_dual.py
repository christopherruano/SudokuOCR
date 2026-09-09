#!/usr/bin/env python3
"""Dual-number re-read: for pages that may show a Roman number at the TOP (a
within-section page number) AND an Arabic number at the BOTTOM (the through-volume
page number), capture BOTH. We order by the Arabic/volume number (bottom preferred),
and keep the Roman as a separate 'section_page' field.

Sends two crops per scan in one request (top strip + bottom strip, full-res) so both
small numerals stay legible; falls back to the full image if both come back empty.

Targets the scans the Arabic-only pass left null (the supplementary/subsidiary tables,
where this dual numbering happens). Output: pageno_parts/dualnum_reads.jsonl
Usage: python3 pageno_reread_dual.py [--workers 24] [--limit N] [--all]
"""
import argparse, io, json, os, re, sys, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
from PIL import Image

ROOT = os.path.dirname(os.path.abspath(__file__))
V1 = os.path.join(ROOT, "export_final/sendout_final/sendout_v1")
PARTS = os.path.join(V1, "pageno_parts")
JSONL = os.path.join(PARTS, "gemini_reads.jsonl")
OUT = os.path.join(PARTS, "dualnum_reads.jsonl")

load_dotenv(os.path.join(ROOT, ".env"))
MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.1-pro-preview")
BASE_URL = os.environ.get("HARVARD_GEMINI_BASE_URL") or os.environ.get("GEMINI_BASE_URL") or ""

PROMPT = (
    "These are crops from ONE scanned page of a historical printed Indian Census volume. "
    "Image 1 is the TOP of the page; Image 2 is the BOTTOM of the page.\n"
    "Census supplementary tables often carry TWO page numbers: a ROMAN numeral (e.g. xxxviii) "
    "that paginates a section, usually at the TOP; and an ARABIC numeral (e.g. 136) that is the "
    "through-volume page number, often at the BOTTOM. Read BOTH if present.\n"
    "Do NOT report the table number from a title like 'TABLE VII'/'SUPPLEMENTARY TABLE 6' as a "
    "page number. Ignore the census year and the JSTOR footer text.\n"
    "Return STRICT JSON only:\n"
    '{"top": {"raw": <numeral as printed or null>, "system": "roman"|"arabic"|null}, '
    '"bottom": {"raw": <numeral as printed or null>, "system": "roman"|"arabic"|null}, '
    '"page_type": "data_table"|"errata_corrections"|"text"|"index"|"other", '
    '"confidence": "high"|"medium"|"low"}\n'
    "Use null for a position with no page number. Never invent a number."
)
PROMPT_FULL = PROMPT.replace("Image 1 is the TOP of the page; Image 2 is the BOTTOM of the page.",
                             "This is the FULL page; look at both its top and bottom for page numbers.")

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


def to_int(raw, system):
    if raw is None:
        return None
    s = str(raw).strip().strip(".[]() ")
    if re.fullmatch(r"\d{1,4}", s):
        return int(s)
    rv = roman_to_int(s)
    return rv


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


def strip_png(im, max_w=1500):
    if im.width > max_w:
        im = im.resize((max_w, int(im.height * max_w / im.width)))
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


def crops(path):
    im = Image.open(path).convert("RGB")
    w, h = im.size
    top = im.crop((0, 0, w, max(1, int(h * 0.22))))
    bot = im.crop((0, int(h * 0.82), w, h))
    return strip_png(top), strip_png(bot), strip_png(im)


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


def derive(d):
    """From {top,bottom,page_type,confidence} compute volume_page, section_page, page."""
    top = d.get("top") or {}
    bot = d.get("bottom") or {}
    t_sys, b_sys = (top.get("system") or "").lower(), (bot.get("system") or "").lower()
    t_int = to_int(top.get("raw"), t_sys)
    b_int = to_int(bot.get("raw"), b_sys)
    # recompute systems from the raw value (more reliable)
    def sysof(raw, fallback):
        s = str(raw or "").strip().strip(".[]() ")
        if re.fullmatch(r"\d{1,4}", s):
            return "arabic"
        if roman_to_int(s) is not None:
            return "roman"
        return fallback
    if top.get("raw"):
        t_sys = sysof(top.get("raw"), t_sys)
    if bot.get("raw"):
        b_sys = sysof(bot.get("raw"), b_sys)
    # volume page = the ARABIC number, bottom preferred
    if b_int is not None and b_sys == "arabic":
        volume = b_int
    elif t_int is not None and t_sys == "arabic":
        volume = t_int
    else:
        volume = None
    # section page = the ROMAN number
    section = None
    if t_int is not None and t_sys == "roman":
        section = t_int
    elif b_int is not None and b_sys == "roman":
        section = b_int
    page = volume if volume is not None else section
    return {
        "page": page, "volume_page": volume, "section_page": section,
        "top_raw": top.get("raw"), "top_system": t_sys or None,
        "bottom_raw": bot.get("raw"), "bottom_system": b_sys or None,
        "page_type": d.get("page_type"), "confidence": d.get("confidence") or "",
    }


def call(client, types, parts_bytes, prompt, ks, mi, max_attempts=4):
    cfg = types.GenerateContentConfig(temperature=0.0, response_mime_type="application/json")
    parts = [types.Part.from_bytes(data=b, mime_type="image/png") for b in parts_bytes] + [prompt]
    last = ""
    for attempt in range(max_attempts):
        with ks.lock:
            if mi > 0 and ks.last_call:
                w = mi - (time.time() - ks.last_call)
                if w > 0:
                    time.sleep(w)
            ks.last_call = time.time()
        try:
            return (client.models.generate_content(model=MODEL, contents=parts, config=cfg).text or "")
        except Exception as exc:
            last = f"{type(exc).__name__}: {exc}"
            if not any(t in last for t in ["429", "500", "502", "503", "504", "UNAVAILABLE", "RESOURCE_EXHAUSTED", "INTERNAL", "DEADLINE"]) or attempt == max_attempts - 1:
                raise
            time.sleep(min(60, 4 * (2 ** attempt)))
    raise RuntimeError(last)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--min-interval", type=float, default=0.2)
    ap.add_argument("--all", action="store_true", help="all 2492 scans (default: just the null/Roman set)")
    args = ap.parse_args()

    reads = {}
    for line in open(JSONL):
        line = line.strip()
        if line:
            r = json.loads(line)
            reads[r["imghash"]] = r
    if args.all:
        targets = list(reads.values())
    else:
        targets = [r for r in reads.values() if r.get("page_number") is None]

    done = set()
    if os.path.exists(OUT):
        for line in open(OUT):
            try:
                done.add(json.loads(line)["imghash"])
            except Exception:
                pass
    todo = [r for r in targets if r["imghash"] not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"model={MODEL} base_url={'set' if BASE_URL else 'NONE'} targets={len(targets)} todo={len(todo)} done={len(done)}", flush=True)
    if not todo:
        print("nothing to do")
        return

    keys = load_keys()
    clients = {ks.slot: make_client(ks.key) for ks in keys}
    wl = threading.Lock()
    cnt = {"n": 0, "dual": 0, "vol": 0, "sec_only": 0, "null": 0, "err": 0, "fb": 0}
    t0 = time.time()
    fh = open(OUT, "a")

    def work(i, r):
        ks = keys[i % len(keys)]
        client, types = clients[ks.slot]
        scan = os.path.join(ROOT, r["scan_path"])
        try:
            top, bot, full = crops(scan)
            raw = call(client, types, [top, bot], PROMPT, ks, args.min_interval)
            d = parse_json(raw)
            der = derive(d) if d else None
            fb = False
            if not der or der.get("page") is None:
                fb = True
                raw2 = call(client, types, [full], PROMPT_FULL, ks, args.min_interval)
                d2 = parse_json(raw2)
                if d2 and derive(d2).get("page") is not None:
                    der = derive(d2)
            if not der:
                der = {"page": None, "volume_page": None, "section_page": None,
                       "top_raw": None, "top_system": None, "bottom_raw": None,
                       "bottom_system": None, "page_type": None, "confidence": ""}
            return {"imghash": r["imghash"], "scan_path": r["scan_path"],
                    "asm_ids": r.get("asm_ids", []), "table_ids": r.get("table_ids", []),
                    "used_fallback": fb, **der}, None
        except Exception as exc:
            return {"imghash": r["imghash"], "scan_path": r["scan_path"], "page": None,
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
                elif rec.get("volume_page") is not None and rec.get("section_page") is not None:
                    cnt["dual"] += 1
                elif rec.get("volume_page") is not None:
                    cnt["vol"] += 1
                elif rec.get("section_page") is not None:
                    cnt["sec_only"] += 1
                else:
                    cnt["null"] += 1
                if rec.get("used_fallback"):
                    cnt["fb"] += 1
                n = cnt["n"]
                if n % 25 == 0 or n == len(todo):
                    el = time.time() - t0
                    print(f"  {n}/{len(todo)} dual(roman+arabic)={cnt['dual']} arabic_only={cnt['vol']} "
                          f"roman_only={cnt['sec_only']} null={cnt['null']} err={cnt['err']} "
                          f"ETA {((len(todo)-n)/(n/el))/60:.1f}m", flush=True)
    fh.close()
    print(f"DONE {(time.time()-t0)/60:.1f}m  dual={cnt['dual']} arabic_only={cnt['vol']} "
          f"roman_only={cnt['sec_only']} null={cnt['null']} err={cnt['err']}")


if __name__ == "__main__":
    main()
