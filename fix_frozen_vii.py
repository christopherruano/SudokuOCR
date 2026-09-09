#!/usr/bin/env python3
"""De-freeze the remaining Table VII frozen pages by MAPPING (not re-OCR'ing). Mistral already read these
correctly; the converter just couldn't handle the layout. For each page we give Gemini the reliable
Mistral markdown + the corpus's exact section/row structure and ask it ONLY to copy the markdown numbers
into that structure (row-by-INDEX, so repeated age labels like CHRISTIAN 0-5 / JEW 0-5 don't collide).
Then we validate: apply only if it REMOVES the frozen columns and keeps the same rows. Backup to
*_oneshot.prefrozen2.json. Constraint violations that remain are reported for the human to finish.
  python3 fix_frozen_vii.py [--limit N] [--apply] [--only <imghash>]
"""
import sys, os, json, glob, csv, re, shutil
_apply = "--apply" in sys.argv
_limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else 0
_only = sys.argv[sys.argv.index("--only") + 1] if "--only" in sys.argv else None
sys.argv = ["x"]
import constraint_guided_reocr as C
import detect_misfires as MF
import census_review_app as A

ROOT = os.path.dirname(os.path.abspath(__file__))
V1 = os.path.join(ROOT, "export_final/sendout_final/sendout_v1")
MD = os.path.join(V1, "mistral_md")
CORPUS = os.path.join(ROOT, "export/ocr_results_referenced/digitized_tables")
STD_GROUPS = ["population", "unmarried", "married", "widowed"]


def norm(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def corpus_file(ih):
    g = glob.glob(os.path.join(CORPUS, "*", f"{ih}_oneshot.json"))
    return g[0] if g else None


def schema_of(d):
    """Return (rowhdr, group_map) if the page uses the standard POP/UNM/MAR/WID×persons/males/females
    schema, else None. group_map: canonical(population..) -> actual key in data."""
    for s in d.get("data", []):
        rows = s.get("rows", [])
        if not rows:
            continue
        gm = {}
        for g in rows[0]:
            if isinstance(rows[0][g], dict):
                for canon in STD_GROUPS:
                    if norm(g).startswith(canon):
                        gm[canon] = g
        if len(gm) == 4:
            rh = next((k for k in rows[0] if not isinstance(rows[0].get(k), dict)), None)
            return rh, gm
    return None


def build_prompt(d, rh):
    lines = ["You are mapping an already-OCR'd Imperial Table VII (Age, Sex & Civil Condition) from a",
             "RELIABLE OCR markdown into a fixed structure. ONLY copy the markdown's numbers — never invent",
             "or alter a value. Blank / '...' / '—' = 0. For UNMARRIED/MARRIED/WIDOWED, if the markdown shows",
             "only Males and Females (no Persons sub-column), set persons = males + females.\n",
             "The page has these sections, each with rows IN THIS EXACT ORDER (map values to THIS order):"]
    secs = []
    for s in d.get("data", []):
        rows = s.get("rows", [])
        if not rows:
            continue
        labels = [str(r.get(rh, "")) for r in rows]
        secs.append((s.get("name", ""), labels))
        lines.append(f'  Section "{s.get("name","")}":')
        for i, lb in enumerate(labels):
            lines.append(f"    row{i}: {lb}")
    lines.append("\nOCR markdown of this page:\n")
    md = open(os.path.join(MD, _cur_ih + ".mistral.md")).read()
    lines.append(md[:9000])
    lines.append('\nOutput ONLY JSON: {"sections":[{"rows":[[POPp,POPm,POPf,UNMp,UNMm,UNMf,MARp,MARm,MARf,'
                 'WIDp,WIDm,WIDf], ...one 12-int array per row, in the order above...]}, ...]}.'
                 ' Same number of sections and rows as given.')
    return "\n".join(lines), secs


def apply_map(d, rh, gm, parsed):
    nd = json.loads(json.dumps(d))
    psecs = parsed.get("sections", [])
    dsecs = [s for s in nd.get("data", []) if s.get("rows")]
    if len(psecs) != len(dsecs):
        return None, "section-count-mismatch"
    applied = 0
    for ps, ds in zip(psecs, dsecs):
        prows = ps.get("rows", [])
        drows = ds["rows"]
        if len(prows) != len(drows):
            return None, "row-count-mismatch"
        for arr, row in zip(prows, drows):
            if not isinstance(arr, list) or len(arr) < 12:
                continue
            v = [int(x) if isinstance(x, (int, float)) else 0 for x in arr[:12]]
            row[gm["population"]] = {"persons": v[0], "males": v[1], "females": v[2]}
            row[gm["unmarried"]] = {"persons": v[3], "males": v[4], "females": v[5]}
            row[gm["married"]] = {"persons": v[6], "males": v[7], "females": v[8]}
            row[gm["widowed"]] = {"persons": v[9], "males": v[10], "females": v[11]}
            applied += 1
    return nd, applied


def frozen_count(d):
    n = 0
    for s in d.get("data", []):
        rows = s.get("rows", [])
        rh = next((k for k in rows[0] if not isinstance(rows[0].get(k), dict)), None) if rows else None
        n += len(MF.section_misfires(rows, rh))
    return n


def viol_count(d):
    return sum(len(A.check_constraints(s.get("rows", []))) for s in d.get("data", []))


def vii_frozen():
    out = []
    for r in csv.DictReader(open(os.path.join(ROOT, "export/misfire_tables.csv"))):
        t = r["title"].upper()
        if ("TABLE VII" in t or "AGE, SEX" in t) and "VIII" not in t and "XVII" not in t:
            out.append(r["imghash"])
    return out


_cur_ih = None


def main():
    global _cur_ih
    keys = C.load_keys()
    ihs = [_only] if _only else vii_frozen()
    if _limit:
        ihs = ihs[:_limit]
    applied = flagged = noschema = nomd = 0
    for i, ih in enumerate(ihs):
        _cur_ih = ih
        cf = corpus_file(ih)
        if not cf or not os.path.exists(os.path.join(MD, ih + ".mistral.md")):
            nomd += 1; print(f"  {ih[:12]} — no corpus/markdown, skip"); continue
        d = json.load(open(cf))
        if (d.get("metadata", {}) or {}).get("human_corrected"):
            print(f"  {ih[:12]} — human_corrected, skip"); continue
        sch = schema_of(d)
        if not sch:
            noschema += 1; print(f"  {ih[:12]} — non-standard schema (subsidiary/transposed), FLAG for manual"); continue
        rh, gm = sch
        of = frozen_count(d)
        prompt, _secs = build_prompt(d, rh)
        client, types = C.make_client(keys[i % len(keys)])
        try:
            cfg = types.GenerateContentConfig(temperature=0.0, response_mime_type="application/json")
            txt = client.models.generate_content(model=C.MODEL, contents=[prompt], config=cfg).text or ""
        except Exception as e:
            print(f"  {ih[:12]} — gemini error {str(e)[:50]}"); flagged += 1; continue
        parsed = C.parse_json(txt)
        if not parsed:
            print(f"  {ih[:12]} — no JSON parsed, FLAG"); flagged += 1; continue
        nd, res = apply_map(d, rh, gm, parsed)
        if nd is None:
            print(f"  {ih[:12]} — {res}, FLAG for manual"); flagged += 1; continue
        nf, nv = frozen_count(nd), viol_count(nd)
        ok = nf < of and nf == 0
        tag = "✅ de-frozen" if ok else "✗ still frozen"
        print(f"  {ih[:12]} frozen {of}->{nf} · {nv} constraint-viol remain · {res} cells  {tag}{' APPLY' if ok and _apply else ''}")
        if ok:
            applied += 1
            if _apply:
                shutil.copy(cf, cf.replace("_oneshot.json", "_oneshot.prefrozen2.json"))
                json.dump(nd, open(cf, "w"))
        else:
            flagged += 1
    print(f"\n{'APPLIED' if _apply else 'DRY'}: de-frozen {applied} | flagged {flagged} | non-standard-schema {noschema} "
          f"| no-md {nomd}  (of {len(ihs)})")


if __name__ == "__main__":
    main()
