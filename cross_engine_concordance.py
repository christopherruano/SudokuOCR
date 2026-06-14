#!/usr/bin/env python3
"""Distinguish 'our OCR is wrong' from 'the printed source doesn't sum' for the
failing Table VII, using two INDEPENDENT engines (Gemini in the DB vs cached Mistral).
For each failing table with a cached Mistral read, align cells and classify:
  - likely_faithful  : engines AGREE on the cells but U+M+W != Population -> the
    PRINTED table itself doesn't sum (a source typo); our read is faithful and the
    table is effectively correct (fast-Accept candidate in the review app).
  - ocr_disagreement : engines DISAGREE on the breaking cell(s) -> a real OCR error;
    the correct value is likely the one that makes the sum work (review priority).
  - no_mistral / unparseable : can't compare.
Read-only (writes concordance_review.csv); safe to run alongside the page-ordering window."""
import os, re, csv, glob, sqlite3
import census_explorer as ce
import parse_mistral_md as P

csv.field_size_limit(10 ** 7)
V1 = os.path.dirname(ce.DB)
MD = os.path.join(V1, "mistral_md")
HEX = re.compile(r"([0-9a-f]{40})")
CIV = {"U": re.compile(r"unmarr|single|never", re.I), "M": re.compile(r"marri", re.I),
       "W": re.compile(r"widow", re.I), "P": re.compile(r"popul|persons|total", re.I)}
AGE = re.compile(r"\d+\s*[-–]\s*\d+|\d+\s*and over|under|all ages|total", re.I)


def civ(s):
    for k in ("U", "W", "M", "P"):
        if CIV[k].search(s or ""):
            return k
    return ""


def nkey(a):
    return re.sub(r"\s+", "", (a or "").lower())[:8]


def gem_cells(con, tid):
    out = {}
    for f in con.execute("SELECT row_label_std,column_group_std,sub_column_std,sex,value FROM facts WHERE table_id=?", (tid,)):
        kc = civ(f["column_group_std"]) or civ(f["sub_column_std"]) or civ(f["row_label_std"])
        am = AGE.search(f["row_label_std"] or "") or AGE.search(f["column_group_std"] or "")
        age = nkey(am.group(0)) if am else ""
        sx = (f["sex"] or f["sub_column_std"] or "").lower()[:1]
        try:
            v = int(float(f["value"]))
        except (ValueError, TypeError):
            continue
        if kc and age and sx in ("m", "f"):
            out[(age, sx, kc)] = v
    return out


def mistral_cells_for(tid, con):
    # gather mistral reads for this table's member scans
    raw = con.execute("SELECT source_image,member_images FROM tables WHERE table_id=?", (tid,)).fetchone()
    if not raw:
        return {}
    hs = set(HEX.findall((raw["source_image"] or "") + " " + (raw["member_images"] or "")))
    out = {}
    for h in hs:
        p = os.path.join(MD, f"{h}.mistral.md")
        if not os.path.exists(p):
            continue
        for sec in P.mistral_parse(p):
            for r in sec["rows"]:
                age = nkey(r["age"])
                for g in ("pop", "unm", "mar", "wid"):
                    gk = {"pop": "P", "unm": "U", "mar": "M", "wid": "W"}[g]
                    for sx in ("m", "f"):
                        if f"{g}_{sx}" in r:
                            out[(age, sx, gk)] = r[f"{g}_{sx}"]
    return out


def main():
    led = [r for r in csv.DictReader(open(os.path.join(V1, "tableVII_verdicts.csv")))
           if r["status"] in ("unrecoverable", "retrying")]
    con = sqlite3.connect(f"file:{ce.DB}?mode=ro", uri=True); con.row_factory = sqlite3.Row
    out = []
    n_faithful = n_disagree = n_nomistral = 0
    for r in led:
        tid = r["table_id"]
        g = gem_cells(con, tid)
        m = mistral_cells_for(tid, con)
        shared = set(g) & set(m)
        if len(shared) < 6:
            n_nomistral += 1
            out.append({"table_id": tid, "verdict": r["status"], "klass": "no_mistral_overlap",
                        "shared_cells": len(shared), "agree_pct": "", "note": r["reason"][:50]})
            continue
        agree = sum(1 for k in shared if g[k] == m[k])
        pct = round(100 * agree / len(shared))
        # do BOTH engines fail the sum on the same (age,sex)?
        def sumfails(cells):
            bad = 0; chk = 0
            ages = {(a, s) for (a, s, k) in cells}
            for (a, s) in ages:
                d = {k: cells.get((a, s, k)) for k in ("U", "M", "W", "P")}
                if None in d.values():
                    continue
                chk += 1; bad += abs(d["P"] - (d["U"] + d["M"] + d["W"])) > max(2, 0.02 * max(1, d["P"]))
            return bad, chk
        gb, gc = sumfails(g)
        mb, mc = sumfails(m)
        both_fail = gc and mc and gb > 0 and mb > 0
        if pct >= 90 and both_fail:
            klass = "likely_faithful_source_typo"; n_faithful += 1
        elif pct < 80:
            klass = "ocr_disagreement_real_error"; n_disagree += 1
        else:
            klass = "mixed"; n_disagree += 1
        out.append({"table_id": tid, "verdict": r["status"], "klass": klass,
                    "shared_cells": len(shared), "agree_pct": pct, "note": r["reason"][:50]})
    out.sort(key=lambda r: (r["klass"], -(r["agree_pct"] or 0)))
    with open(os.path.join(V1, "concordance_review.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["table_id", "verdict", "klass", "shared_cells", "agree_pct", "note"])
        w.writeheader(); [w.writerow(x) for x in out]
    print(f"failing tables analyzed: {len(led)}")
    print(f"  likely FAITHFUL (engines agree, source itself doesn't sum -> fast-Accept): {n_faithful}")
    print(f"  OCR disagreement (engines differ -> real error, review priority): {n_disagree}")
    print(f"  no Mistral overlap (can't cross-check, needs eyes): {n_nomistral}")
    print("wrote concordance_review.csv")


if __name__ == "__main__":
    main()
