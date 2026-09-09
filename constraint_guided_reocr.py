#!/usr/bin/env python3
"""Constraint-guided targeted re-OCR (Gemini, all Harvard keys in parallel).

For pages with a SMALL number of constraint failures (a couple cells / one section — NOT whole-table
garble), we tell Gemini exactly which cells fail which identity (M+F=Persons, U+M+W=Total) and ask it to
RE-READ ONLY those cells from the scan. We apply the corrected values and keep them ONLY if they reduce
the page's failures (and never touch human_corrected pages). Original backed up to *_oneshot.preconstraint.json.

  python3 constraint_guided_reocr.py [--max-fail-rows 4] [--apply]   (dry-run by default)
"""
import sys, os, re, json, glob, time, copy, shutil
from concurrent.futures import ThreadPoolExecutor

APPLY = "--apply" in sys.argv
MAXFAIL = int(sys.argv[sys.argv.index("--max-fail-rows") + 1]) if "--max-fail-rows" in sys.argv else 4
LIMIT = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else 0
ENGINE = sys.argv[sys.argv.index("--engine") + 1] if "--engine" in sys.argv else "gemini"
REASON = "--reason" in sys.argv      # ask the model to REASON the fix from the constraints (vs re-read)
MISTRAL_MODEL = sys.argv[sys.argv.index("--mistral-model") + 1] if "--mistral-model" in sys.argv else "mistral-medium-latest"
sys.argv = ["x"]
import census_review_app as A
from PIL import Image  # noqa

ROOT = os.path.dirname(os.path.abspath(__file__))
IMG = os.path.join(ROOT, "census_site/img")
CORPUS = os.path.join(ROOT, "export/ocr_results_referenced/digitized_tables")
MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.1-pro-preview")


def _env():
    env = {}
    for line in open(os.path.join(ROOT, ".env")):
        if "=" in line and not line.strip().startswith("#"):
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip().strip('"').strip("'")
    return env


_ENV = _env()
BASE_URL = _ENV.get("HARVARD_GEMINI_BASE_URL") or _ENV.get("GEMINI_BASE_URL") or ""


def load_keys():
    return [k.strip() for k in _ENV.get("HARVARD_GEMINI_API_KEYS", "").split(",") if k.strip()]


def make_client(key):
    from google import genai
    from google.genai import types
    http = {"timeout": 300_000 if BASE_URL else 120_000}
    if BASE_URL:
        http["base_url"] = BASE_URL          # Harvard HUIT proxy — keys aren't valid against Google direct
    return genai.Client(api_key=key, http_options=types.HttpOptions(**http)), types


def gemini(client, types, prompt, img_bytes, json_mode=True, attempts=3):
    cfg = types.GenerateContentConfig(temperature=0.0,
                                      **({"response_mime_type": "application/json"} if json_mode else {}))
    part = types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg")
    for a in range(attempts):
        try:
            r = client.models.generate_content(model=MODEL, contents=[part, prompt], config=cfg)
            return r.text or ""
        except Exception as exc:
            if a == attempts - 1:
                raise
            time.sleep(4 * (2 ** a))


def mistral_client(key):
    from mistralai import Mistral
    return Mistral(api_key=key)


def mistral_read(client, prompt, img_bytes, attempts=3):
    import base64
    b64 = base64.b64encode(img_bytes).decode()
    for a in range(attempts):
        try:
            r = client.chat.complete(
                model=MISTRAL_MODEL, temperature=0, response_format={"type": "json_object"},
                messages=[{"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": f"data:image/jpeg;base64,{b64}"}]}])
            return r.choices[0].message.content or ""
        except Exception:
            if a == attempts - 1:
                raise
            time.sleep(4 * (2 ** a))


def parse_json(t):
    if not t:
        return None
    t = re.sub(r"^```(?:json)?", "", t.strip()).strip()
    t = re.sub(r"```$", "", t).strip()
    v = None
    try:
        v = json.loads(t)
    except Exception:
        m = re.search(r"[\[{].*[\]}]", t, re.DOTALL)   # also handles arrays, not just objects
        if m:
            try:
                v = json.loads(m.group(0))
            except Exception:
                v = None
    if isinstance(v, list):                            # Gemini sometimes returns [{...}] — merge to one dict
        out = {}
        for item in v:
            if isinstance(item, dict):
                out.update(item)
        return out or None
    return v if isinstance(v, dict) else None


def row_label(row, rh):
    return (str(row.get(rh, "")) if rh else "") or ""


def page_fail_rows(d):
    """total distinct failing rows across the page's (non-transposed) sections"""
    n = 0
    rows_total = 0
    for s in d.get("data", []):
        rows = s.get("rows", [])
        if A.is_transposed(rows):
            continue
        n += len(set(k[0] for k in A.check_constraints(rows)))
        rows_total += len(rows)
    return n, rows_total


def build_prompt(d):
    lines = [
        "You are re-reading specific cells from a scanned historical Indian census table (Table VII / C-III:",
        "Age, Sex and Civil Condition). Our OCR has a few cells that fail arithmetic identities. RE-READ from",
        "the IMAGE only the cells listed below and return their correct values as printed.",
        "Identities that must hold per row: within a civil-condition group Males+Females=Persons; and across",
        "groups per column Unmarried+Married+Widowed=Total.\nCells to re-check:",
    ]
    for s in d.get("data", []):
        rows = s.get("rows", [])
        if A.is_transposed(rows):
            continue
        rh = next((k for k in rows[0] if not isinstance(rows[0].get(k), dict)), None) if rows else None
        viol = A.check_constraints(rows)
        byfail = {}   # (row_idx, reason) -> list of exact (group, subkey) cells in that failing identity
        for (ri, g, sk), reason in viol.items():
            byfail.setdefault((ri, reason), []).append((g, sk))
        for (ri, reason), cells in byfail.items():     # single-cell prompt — the one that found real fixes
            lbl = row_label(rows[ri], rh) or f"row{ri}"
            desc = []
            for g, sk in cells:
                v = rows[ri].get(g)
                desc.append(f"{g}.{sk}={v.get(sk) if isinstance(v, dict) else v}")
            lines.append(f'  Row "{lbl}": identity "{reason}" fails. Re-read EXACTLY these cells: ' + ", ".join(desc))
    lines.append('\nFor each failing identity, find the SINGLE component cell whose printed digits differ from what')
    lines.append("we listed, and correct THAT cell so the identity holds. Do NOT change a Total/Persons cell to mask")
    lines.append("a misread component, and do not change cells that are already read correctly.")
    lines.append('Return ONLY the cells you are correcting, as JSON keyed by the EXACT row label and group/subkey names:')
    lines.append('  {"<row label>": {"<group>": {"<subkey>": <integer>}}}   (integers as printed, no commas)')
    return "\n".join(lines)


def _norm(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def reason_prompt(d):
    lines = ["A row from a historical Indian census Table VII (Age, Sex & Civil Condition). Identities that",
             "MUST hold: within each civil-condition group Males+Females=Persons; and across groups per column",
             "Unmarried+Married+Widowed=Total.\nOCR'd values:"]
    for s in d.get("data", []):
        rows = s.get("rows", [])
        if A.is_transposed(rows):
            continue
        rh = next((k for k in rows[0] if not isinstance(rows[0].get(k), dict)), None) if rows else None
        for ri in sorted({k[0] for k in A.check_constraints(rows)}):
            lbl = (str(rows[ri].get(rh, "")) if rh else f"r{ri}") or f"r{ri}"
            cells = [f"{g}{{" + ", ".join(f"{k}={v.get(k)}" for k in v) + "}" for g, v in rows[ri].items() if isinstance(v, dict)]
            lines.append(f'  Row "{lbl}": ' + " ; ".join(cells))
    lines.append("\nREASON which cell(s) are misread and what value(s) make ALL identities hold (a frequent error")
    lines.append("is a males/females mis-split: Persons right, split wrong, so BOTH sex cells change). Confirm the")
    lines.append("digits against the scan image. Explain briefly, THEN end with the fix as JSON on its own line:")
    lines.append('  {"<row label>": {"<group>": {"<subkey>": <int>}}}')
    return "\n".join(lines)


def apply_fix(d, corrected):
    """Apply Gemini's corrections. FUZZY-match labels/group/subkey names (the data has trailing periods
    like 'MARRIED.' that Gemini drops), and only touch rows that are CURRENTLY failing a constraint (so a
    clean row that happens to share a label isn't clobbered)."""
    nd = copy.deepcopy(d)
    applied = 0
    cor_n = {_norm(k): v for k, v in corrected.items()}
    for s in nd.get("data", []):
        rows = s.get("rows", [])
        rh = next((k for k in rows[0] if not isinstance(rows[0].get(k), dict)), None) if rows else None
        for row in rows:
            if not A.check_constraints([row]):              # only correct rows that actually fail
                continue
            cor = cor_n.get(_norm(row_label(row, rh)))
            if not isinstance(cor, dict):
                continue
            gmap = {_norm(g): g for g in row if isinstance(row[g], dict)}
            for cg, subs in (cor.items() if isinstance(cor, dict) else []):
                rg = gmap.get(_norm(cg))
                if not (rg and isinstance(subs, dict)):
                    continue
                skmap = {_norm(k): k for k in row[rg]}
                for csk, val in subs.items():
                    rsk = skmap.get(_norm(csk))
                    if rsk and isinstance(val, (int, float)):
                        row[rg][rsk] = int(val); applied += 1
    return nd, applied


def process(task):
    (tid, ih, d, page), key = task
    try:
        img = open(os.path.join(IMG, ih + ".jpg"), "rb").read()
        prompt = reason_prompt(d) if REASON else build_prompt(d)
        if ENGINE == "mistral":
            cor = parse_json(mistral_read(mistral_client(key), prompt, img))
        else:
            client, types = make_client(key)
            cor = parse_json(gemini(client, types, prompt, img, json_mode=not REASON))
        if not cor:
            return {"page": page, "status": "no-parse", "old": None, "new": None, "ih": ih}
        nd, applied = apply_fix(d, cor)
        old, _ = page_fail_rows(d)
        new, _ = page_fail_rows(nd)
        if new < old and applied:
            return {"page": page, "status": "FIX", "old": old, "new": new, "ih": ih, "nd": nd, "tid": tid}
        return {"page": page, "status": "no-help", "old": old, "new": new, "ih": ih}
    except Exception as exc:
        return {"page": page, "status": f"err:{type(exc).__name__}", "old": None, "new": None, "ih": ih}


def main():
    if ENGINE == "mistral":
        keys = [_ENV.get("MISTRAL_API_KEY", "")]
        if not keys[0]:
            sys.exit("no MISTRAL_API_KEY in .env")
    else:
        keys = load_keys()
        if not keys:
            sys.exit("no HARVARD_GEMINI_API_KEYS in .env")
    # find small-failure pages (skip human_corrected, transposed, garbled, no image)
    targets = []
    for run in A.assembly_runs():
        for m in run.get("members", []):
            d, ih, _ = A.load_table(m.get("table_id", ""))
            if not d or (d.get("metadata", {}) or {}).get("human_corrected"):
                continue
            if not (ih and os.path.exists(os.path.join(IMG, ih + ".jpg"))):
                continue
            nf, nrows = page_fail_rows(d)
            if 1 <= nf <= MAXFAIL and (nrows == 0 or nf / nrows < 0.25):
                targets.append((m.get("table_id", ""), ih, d, m.get("page")))
    print(f"small-failure pages (<= {MAXFAIL} failing rows, not garbled/transposed/human): {len(targets)} "
          f"| engine: {ENGINE}{'/' + MISTRAL_MODEL if ENGINE == 'mistral' else ''} | {'APPLY' if APPLY else 'dry-run'}")
    if LIMIT:
        targets = targets[:LIMIT]
        print(f"  (limited to first {len(targets)} for this run)")
    if not targets:
        return
    workers = len(keys) if ENGINE == "gemini" else 6
    tasks = [(t, keys[i % len(keys)]) for i, t in enumerate(targets)]
    results = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for r in ex.map(process, tasks):
            results.append(r)
    fixes = [r for r in results if r["status"] == "FIX"]
    print(f"  {ENGINE} resolved/reduced: {len(fixes)} | no help: {sum(1 for r in results if r['status']=='no-help')} "
          f"| no-parse: {sum(1 for r in results if r['status']=='no-parse')} | errors: {sum(1 for r in results if r['status'].startswith('err'))}")
    for r in sorted(fixes, key=lambda x: x['new'])[:30]:
        print(f"    p{r['page']}: {r['old']} -> {r['new']} failing rows  {'(would apply)' if not APPLY else 'APPLIED'}")
    if APPLY:
        for r in fixes:
            jf = glob.glob(os.path.join(CORPUS, "*", f"{r['ih']}_oneshot.json"))
            if not jf:
                continue
            shutil.copy(jf[0], jf[0].replace("_oneshot.json", "_oneshot.preconstraint.json"))
            r["nd"].setdefault("metadata", {})["constraint_fixed"] = True
            json.dump(r["nd"], open(jf[0], "w"))
        print(f"  applied {len(fixes)} constraint-guided fixes (backups: *_oneshot.preconstraint.json)")


if __name__ == "__main__":
    main()
