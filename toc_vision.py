#!/usr/bin/env python3
"""Vision-read census volume front matter -> the TOC chapter/table/page spine.
Reads export/toc_renders/manifest.json, calls Gemini per front page with a TOC-extraction prompt,
keeps the entries from pages that ARE a contents / list-of-tables. Output: export/toc_spine/{saoa}.json
  run via run_toc_vision.sh (sets the Gemini key).  flags: --saoa S1,S2  --limit N
"""
import os, sys, json
from pipeline import call_gemini, encode_image, parse_json_response

OUT = "export/toc_spine"
PROMPT = """You are reading ONE page from the FRONT MATTER of a historical Indian census volume (1871-1951).
Decide if this page is part of a TABLE OF CONTENTS / "LIST OF TABLES" / "LIST OF SUBSIDIARY TABLES" /
"INDEX OF TABLES" / "CONTENTS" — i.e. a printed list that maps tables (or chapters) to page numbers.

If it IS such a list, extract EVERY entry exactly as printed. For each entry give:
  - chapter:   the chapter name/number it falls under, if the list shows chapters (else "")
  - table_num: the table identifier as printed (e.g. "IMPERIAL TABLE VII", "Subsidiary Table II",
               "Table XIV-A", "C-III") or "" if none
  - title:     the full descriptive title as printed
  - page:      the printed page number where it begins (integer if shown, else "")

Return STRICT JSON only, no prose:
{"is_toc": true|false,
 "page_kind": "contents|list_of_tables|list_of_subsidiary_tables|index|other",
 "entries": [{"chapter":"", "table_num":"", "title":"", "page":""}]}
If this is NOT a contents/list page, return {"is_toc": false, "page_kind":"other", "entries": []}.
"""


def read_volume(pages):
    entries, toc_pages = [], []
    for p in pages:
        try:
            raw = call_gemini(encode_image(p), PROMPT)
            d = parse_json_response(raw)
        except Exception as e:
            print(f"  {os.path.basename(p)}: ERROR {str(e)[:70]}", flush=True)
            continue
        # robustness: the model sometimes returns a bare list, or null. Coerce to our schema.
        if isinstance(d, list):
            if d and isinstance(d[0], dict) and any(k in d[0] for k in ("title", "table_num", "page")):
                d = {"is_toc": True, "page_kind": "list", "entries": d}
            else:
                d = {"is_toc": False, "entries": []}
        if not isinstance(d, dict):
            d = {"is_toc": False, "entries": []}
        if d and d.get("is_toc") and d.get("entries"):
            es = d.get("entries", []) or []
            for e in es:
                e["_src"] = os.path.basename(p)
            entries.extend(es)
            toc_pages.append((os.path.basename(p), d.get("page_kind"), len(es)))
            print(f"  {os.path.basename(p)}: TOC[{d.get('page_kind')}] {len(es)} entries", flush=True)
        else:
            print(f"  {os.path.basename(p)}: not-toc", flush=True)
    return entries, toc_pages


def main():
    os.makedirs(OUT, exist_ok=True)
    man = json.load(open("export/toc_renders/manifest.json"))
    saoa_filter = sys.argv[sys.argv.index("--saoa")+1].split(",") if "--saoa" in sys.argv else None
    limit = int(sys.argv[sys.argv.index("--limit")+1]) if "--limit" in sys.argv else None
    done = 0
    for v in man:
        if saoa_filter and v["saoa"] not in saoa_filter:
            continue
        if os.path.exists(f"{OUT}/{v['saoa']}.json"):  # resume
            continue
        print(f"=== volume {v['saoa']} ({v['year']}) — {len(v['front_pages'])} front pages ===", flush=True)
        entries, toc_pages = read_volume(v["front_pages"])
        json.dump({"saoa": v["saoa"], "year": v["year"], "toc_pages": toc_pages, "entries": entries},
                  open(f"{OUT}/{v['saoa']}.json", "w"), indent=1)
        print(f"  -> {len(entries)} entries from {len(toc_pages)} TOC pages -> {OUT}/{v['saoa']}.json", flush=True)
        done += 1
        if limit and done >= limit:
            break
    print("toc_vision done", flush=True)


if __name__ == "__main__":
    main()
