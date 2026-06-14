#!/usr/bin/env python3
"""Convert cached Mistral-OCR markdown into our native oneshot.json, IN PLACE in the
results tree, but ONLY when Mistral is a safe improvement over the existing Gemini
read for that scan:
  - Mistral parsed >=4 age rows with civil-condition columns, AND
  - either Mistral captures a civil-condition column Gemini DROPPED, or
    Mistral agrees with Gemini on >=85% of shared cells (and has >= as many rows).
Gemini's raw read is preserved as <imghash>_oneshot.gemini.json (keep old copies).
Then reingest.sh ingests these, the half-page/continuation assembly merges them, and
validate_tableVII's U+M+W=Population gate is the final arbiter — Mistral is never
trusted blind. Only touches scans of currently imperfect (unrecoverable/retrying)
tables, so perfect tables are left alone."""
import os, re, csv, json, glob, sqlite3, shutil
import census_explorer as ce
import parse_mistral_md as P

csv.field_size_limit(10 ** 7)
V1 = os.path.dirname(ce.DB)
MD = os.path.join(V1, "mistral_md")
HEX = re.compile(r"([0-9a-f]{40})")
GROUP_LABEL = {"pop": "Population.", "unm": "Unmarried.", "mar": "Married.", "wid": "Widowed."}


def find_oneshot(ih):
    for root, _, files in os.walk("export/ocr_results_referenced/digitized_tables"):
        for fn in files:
            if fn == f"{ih}_oneshot.json":
                return os.path.join(root, fn)
    return None


def gem_cells(path):
    try:
        d = json.load(open(path))
    except Exception:
        return {}, 0
    out = {}; nrows = 0
    for s in d.get("data", []):
        for r in s.get("rows", []):
            nrows += 1
            age = None
            for v in r.values():
                if isinstance(v, str) and re.search(r"\d+\s*[-—–]\s*\d+", v):
                    age = v; break
            age = age or (list(r.values())[0] if r else "")
            ak = re.sub(r"\s+", "", str(age).lower())[:8]
            for col, val in r.items():
                if isinstance(val, dict):
                    g = ("pop" if re.search(r"popul", col, re.I) else "unm" if re.search(r"unmarr", col, re.I)
                         else "mar" if re.search(r"marri", col, re.I) else "wid" if re.search(r"widow", col, re.I) else None)
                    if not g:
                        continue
                    for sub, sv in val.items():
                        sx = "m" if ("male" in sub.lower() and "fem" not in sub.lower()) else "f" if "female" in sub.lower() else None
                        if sx and isinstance(sv, (int, float)):
                            out[(ak, g, sx)] = int(sv)
    return out, nrows


def mistral_cells(sections):
    out = {}
    for s in sections:
        for r in s["rows"]:
            ak = re.sub(r"\s+", "", r["age"].lower())[:8]
            for g in ("pop", "unm", "mar", "wid"):
                for sx in ("m", "f"):
                    if f"{g}_{sx}" in r:
                        out[(ak, g, sx)] = r[f"{g}_{sx}"]
    return out


def to_oneshot(sections, title, src):
    groups = set()
    data = []
    for s in sections:
        rows = []
        for r in s["rows"]:
            row = {"Age.": r["age"]}
            for g in ("pop", "unm", "mar", "wid"):
                m, f = r.get(f"{g}_m"), r.get(f"{g}_f")
                if m is None and f is None:
                    continue
                groups.add(GROUP_LABEL[g])
                cell = {}
                if m is not None: cell["males"] = m
                if f is not None: cell["females"] = f
                if m is not None and f is not None: cell["persons"] = m + f
                row[GROUP_LABEL[g]] = cell
            if len(row) > 1:
                rows.append(row)
        if rows:
            data.append({"name": s["name"], "rows": rows})
    return {"source_image": src, "extraction_method": "mistral_ocr",
            "metadata": {"title": title, "region": "", "year": 0, "data_type": "absolute",
                         "column_groups": sorted(groups)},
            "data": data,
            "acceptance": {"version": "mistral_ocr", "decision": "accepted_high_confidence", "trust_score": 88}}


def main():
    # process ALL cached Mistral reads (incl. BOTH halves of half-page pairs, so an
    # assembled table is Mistral-consistent throughout rather than mixed-source).
    emitted = skipped_poor = skipped_disagree = no_gemini = already = 0
    for md in glob.glob(os.path.join(MD, "*.mistral.md")):
        ih = os.path.basename(md).split(".")[0]
        secs = P.mistral_parse(md)
        mrows = sum(len(s["rows"]) for s in secs)
        mcols = {g for s in secs for r in s["rows"] for g in ("pop", "unm", "mar", "wid") if f"{g}_m" in r or f"{g}_f" in r}
        if mrows < 4 or not mcols:
            skipped_poor += 1; continue
        gpath = find_oneshot(ih)
        if not gpath:
            no_gemini += 1; continue
        gbak = gpath.replace("_oneshot.json", "_oneshot.gemini.json")
        if os.path.exists(gbak):
            already += 1; continue          # idempotent: already Mistral-emitted
        gc, grows = gem_cells(gpath)
        gcols = {k[1] for k in gc}
        mc = mistral_cells(secs)
        shared = set(mc) & set(gc)
        agree = sum(1 for k in shared if mc[k] == gc[k])
        ratio = agree / len(shared) if shared else 1.0
        adds_col = bool(mcols - gcols)
        # SAFE-improvement gate
        if not (adds_col or (ratio >= 0.85 and mrows >= grows)):
            skipped_disagree += 1; continue
        title = json.load(open(gpath)).get("metadata", {}).get("title", "")
        src = json.load(open(gpath)).get("source_image", ih + ".png")
        shutil.copy(gpath, gpath.replace("_oneshot.json", "_oneshot.gemini.json"))
        json.dump(to_oneshot(secs, title, src), open(gpath, "w"), indent=1)
        emitted += 1
    print(f"Mistral oneshot emitted (replacing Gemini): {emitted}")
    print(f"  skipped — poor parse: {skipped_poor} | low agreement & no new col: {skipped_disagree} | already emitted: {already} | no gemini file: {no_gemini}")
    print("Next: bash reingest.sh  (half-page/continuation assembly + U+M+W gate re-validate), then build_verdicts.")


if __name__ == "__main__":
    main()
