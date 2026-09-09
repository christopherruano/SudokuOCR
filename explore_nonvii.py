#!/usr/bin/env python3
"""Render + Mistral-OCR a small set of non-VII frozen pages and dump the raw markdown
plus the current (broken) oneshot structure, so we can design a GENERIC md->oneshot
converter. Reuses mistral_frozen's render + OCR call. Read-only on the corpus."""
import sys, os, glob, json
sys.argv = ["x"]
import run_mistral_ocr as RM
import mistral_frozen as MZ
import detect_misfires as MF

ROOT = os.path.dirname(os.path.abspath(__file__))


def show(ih):
    print("\n" + "=" * 100)
    print("IMGHASH", ih)
    cp = MZ.corpus_path(ih)
    print("corpus :", cp)
    if cp:
        d = json.load(open(cp))
        ttl = (d.get("metadata", {}) or {}).get("title", "")
        print("title  :", ttl)
        print("col_groups(meta):", (d.get("metadata", {}) or {}).get("column_groups"))
        for s in d.get("data", []):
            rows = s.get("rows", [])
            if not rows:
                continue
            rh = next((k for k in rows[0] if not isinstance(rows[0].get(k), dict)), None)
            mf = MF.section_misfires(rows, rh)
            print(f"  SECTION '{s.get('name','')[:40]}'  rows={len(rows)} rowhdr={rh!r} frozen={len(mf)}")
            print("    first row keys:", list(rows[0].keys()))
            print("    sample row:", json.dumps(rows[0])[:300])
    rp = MZ.ensure_render(ih)
    print("render :", rp)
    return ih, rp, cp


def main():
    ihs = [x.strip() for x in open(sys.argv_orig[1])] if False else \
          [x.strip() for x in open("/private/tmp/claude-501/-Users-chrisruano-Research-SudokuOCR/5f3ddb5c-80ba-4984-9580-17da5d5af496/scratchpad/samples.txt") if x.strip()]
    from mistralai import Mistral
    client = Mistral(api_key=MZ.ENV.get("MISTRAL_API_KEY"))
    os.makedirs(RM.OUT, exist_ok=True)
    for ih in ihs:
        _, rp, cp = show(ih)
        if not rp:
            print("  NO RENDER — skip")
            continue
        ih2, st = RM.ocr_one(client, ih, rp)
        print("  mistral ocr:", st)
        md = os.path.join(RM.OUT, f"{ih}.mistral.md")
        if os.path.exists(md):
            txt = open(md).read()
            print(f"  --- MARKDOWN ({len(txt)} chars) first 90 lines ---")
            for ln in txt.splitlines()[:90]:
                print("  | " + ln[:200])


if __name__ == "__main__":
    main()
