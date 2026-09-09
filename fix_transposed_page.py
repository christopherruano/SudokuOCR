#!/usr/bin/env python3
"""Constraint-guided re-OCR for TRANSPOSED Table VII pages (age bands in rows; categories/religions in
columns; subtotal TOTAL + Grand Total rows). The existing constraint_guided_reocr.py only handles the
HORIZONTAL identities; this handles the VERTICAL ones: in each column, the age-band rows must sum to the
Grand Total (and the rows in each block must sum to that block's TOTAL). We tell Gemini exactly which
columns are off and by how much, ask it to re-read those columns from the scan, apply the corrected cells,
and KEEP the re-read only if the vertical failures drop. Backup: *_oneshot.prevert.json.
  python3 fix_transposed_page.py <saoa> <page> [--apply]
  python3 fix_transposed_page.py --ih <imghash> [--apply]
"""
import sys, os, json, re, glob, copy
_apply = "--apply" in sys.argv
sys.argv2 = [a for a in sys.argv if a != "--apply"]
sys.argv = ["x"]
import census_review_app as A
import constraint_guided_reocr as C

CORPUS = "export/ocr_results_referenced/digitized_tables"


def vviol(d):
    """count + list distinct vertical violations across the page's sections"""
    out = []
    for s in d.get("data", []):
        rows = s.get("rows", [])
        for (ri, g, k), msg in A.check_constraints(rows).items():
            out.append((s.get("name", ""), ri, g, k, msg))
    return out


def norm(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def build_vert_prompt(d):
    lines = ["A historical Indian census TABLE VII, TRANSPOSED layout: each ROW is an age band (e.g. 0-1, 1-2,",
             "…) plus subtotal rows (TOTAL) and a GRAND TOTAL row; each COLUMN is a category split into",
             "sub-columns (Persons/Males/Females or similar). VERTICAL identity that MUST hold: in every column,",
             "the age-band rows sum to the GRAND TOTAL row (and each block's rows sum to its TOTAL).",
             "Our OCR currently FAILS these columns (left = our summed value, right = the printed total cell):"]
    for s in d.get("data", []):
        rows = s.get("rows", [])
        rh = next((k for k in rows[0] if not isinstance(rows[0].get(k), dict)), None) if rows else None
        for (ri, g, k), msg in A.check_constraints(rows).items():
            lines.append(f'  column "{g} / {k}": {msg}')
    lines += ["\nThese gaps are small (usually ONE misread digit in that column — a data cell OR the total cell).",
              "RE-READ each failing column top-to-bottom from the SCAN IMAGE and return ONLY the corrected cells",
              "as JSON on the final line, keyed by the row label then column group then sub-column:",
              '  {"<row label>": {"<column group>": {"<sub-column>": <int>}}}',
              "Return the actual printed digits — if the column genuinely doesn't reconcile on the page (a",
              "printer's error), return {} rather than inventing a number."]
    return "\n".join(lines)


def apply_vert(d, corrected):
    nd = copy.deepcopy(d)
    cor_n = {norm(k): v for k, v in corrected.items()}
    applied = 0
    for s in nd.get("data", []):
        rows = s.get("rows", [])
        rh = next((k for k in rows[0] if not isinstance(rows[0].get(k), dict)), None) if rows else None
        for row in rows:
            cor = cor_n.get(norm(row.get(rh, "") if rh else ""))
            if not isinstance(cor, dict):
                continue
            gmap = {norm(g): g for g in row if isinstance(row[g], dict)}
            for cg, subs in cor.items():
                rg = gmap.get(norm(cg))
                if not (rg and isinstance(subs, dict)):
                    continue
                skmap = {norm(x): x for x in row[rg]}
                for csk, val in subs.items():
                    rsk = skmap.get(norm(csk))
                    if rsk and isinstance(val, (int, float)) and int(val) != row[rg].get(rsk):
                        row[rg][rsk] = int(val); applied += 1
    return nd, applied


def main():
    args = [a for a in sys.argv2[1:]]
    if "--ih" in args:
        ih = args[args.index("--ih") + 1]
        jf = (glob.glob(os.path.join(CORPUS, "*", f"{ih}_oneshot.json")) or [None])[0]
        tid = jf
    else:
        saoa, page = args[0], int(args[1])
        m = next((m for r in A.assembly_runs() if str(r["saoa"]) == str(saoa)
                  for m in r.get("members", []) if m.get("page") == page), None)
        if not m:
            sys.exit(f"no page {page} in saoa {saoa}")
        d0, ih, _ = A.load_table(m["table_id"]); tid = m["table_id"]
        jf = (glob.glob(os.path.join(CORPUS, "*", f"{ih}_oneshot.json")) or [None])[0]
    d = json.load(open(jf))
    before = vviol(d)
    print(f"page imghash {ih} · {sum(len(s.get('rows',[])) for s in d.get('data',[]))} rows · "
          f"{len(before)} vertical failure(s)")
    prompt = build_vert_prompt(d)
    print("\n--- prompt sent to Gemini (what's flagged) ---")
    print(prompt[:1100])
    img = open(os.path.join(C.IMG, ih + ".jpg"), "rb").read()
    engine = sys.argv2[sys.argv2.index("--engine") + 1] if "--engine" in sys.argv2 else "gemini"
    print(f"\n[engine: {engine}]")
    if engine == "mistral":
        cor = C.parse_json(C.mistral_read(C.mistral_client(C._ENV.get("MISTRAL_API_KEY", "")), prompt, img))
    else:
        client, types = C.make_client(C.load_keys()[0])
        cor = C.parse_json(C.gemini(client, types, prompt, img, json_mode=True))
    print(f"\n--- Gemini returned {len(cor) if isinstance(cor, dict) else 0} corrected row(s) ---")
    nd, applied = apply_vert(d, cor or {})
    after = vviol(nd)
    print(f"applied {applied} cell change(s) · vertical failures {len(before)} -> {len(after)}")
    # show what changed
    shown = 0
    for s0, s1 in zip(d.get("data", []), nd.get("data", [])):
        for r0, r1 in zip(s0.get("rows", []), s1.get("rows", [])):
            for g in r0:
                if isinstance(r0[g], dict):
                    for k in r0[g]:
                        if str(r0[g][k]) != str(r1[g].get(k)) and shown < 15:
                            rh = next((x for x in r0 if not isinstance(r0[x], dict)), None)
                            print(f"    {r0.get(rh,'')}/{g}/{k}: {r0[g][k]} -> {r1[g].get(k)}"); shown += 1
    if _apply and len(after) < len(before) and applied:
        import shutil
        shutil.copy(jf, jf.replace("_oneshot.json", "_oneshot.prevert.json"))
        nd.setdefault("metadata", {})["constraint_reocr"] = True
        json.dump(nd, open(jf, "w"))
        print("  ✅ APPLIED (backup *_oneshot.prevert.json)")
    elif _apply:
        print("  — not applied (no improvement)")


if __name__ == "__main__":
    main()
