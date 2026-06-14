#!/usr/bin/env python3
"""Reconcile the failing Table VII where Gemini (DB) and Mistral (cached) DISAGREE on a
cell, using the U+M+W=Population constraint to pick the right value — a correction that
is DUAL-SUPPORTED (a value an independent engine actually read AND that makes the
arithmetic consistent), never a fabricated digit.

Per failing (age,sex) where U+M+W != Population: try substituting each of U/M/W/P with
Mistral's value; if EXACTLY ONE substitution makes the equation hold AND that value
differs from Gemini's, apply it. If 0 or >1 candidates, leave it (no guessing). A table
that fully passes after reconciliation is written to the corrected_facts overlay (and
logged to reconciled.csv for spot-checking). Originals kept (overlay only).
Writes only files (no DB rebuild) — safe alongside the page-ordering window."""
import os, re, csv, sqlite3
import census_explorer as ce
import build_query_dataset as B
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


def mistral_cells_for(con, tid):
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
                for g, gk in (("pop", "P"), ("unm", "U"), ("mar", "M"), ("wid", "W")):
                    for sx in ("m", "f"):
                        if f"{g}_{sx}" in r:
                            out[(age, sx, gk)] = r[f"{g}_{sx}"]
    return out


def ok(P_, U_, M_, W_):
    return abs(P_ - (U_ + M_ + W_)) <= max(2, 0.02 * max(1, P_))


def main():
    led = [r for r in csv.DictReader(open(os.path.join(V1, "tableVII_verdicts.csv")))
           if r["status"] in ("unrecoverable", "retrying")]
    con = sqlite3.connect(f"file:{ce.DB}?mode=ro", uri=True); con.row_factory = sqlite3.Row

    new_facts, reconciled = [], []
    fid = 1
    for r in led:
        tid = r["table_id"]
        facts = [dict(x) for x in con.execute("SELECT * FROM facts WHERE table_id=?", (tid,))]
        m = mistral_cells_for(con, tid)
        if not m:
            continue
        # index gemini fact rows by (age,sex,civ)
        idx = {}
        for f in facts:
            kc = civ(f["column_group_std"]) or civ(f["sub_column_std"]) or civ(f["row_label_std"])
            am = AGE.search(f["row_label_std"] or "") or AGE.search(f["column_group_std"] or "")
            age = nkey(am.group(0)) if am else ""
            sx = (f["sex"] or f["sub_column_std"] or "").lower()[:1]
            try:
                f["_v"] = int(float(f["value"]))
            except (ValueError, TypeError):
                continue
            if kc and age and sx in ("m", "f"):
                idx[(age, sx, kc)] = f
        # per (age,sex), reconcile
        ages = {(a, s) for (a, s, k) in idx}
        changes = 0; checked = 0; still_bad = 0
        for (a, s) in ages:
            cur = {k: idx.get((a, s, k)) for k in ("U", "M", "W", "P")}
            if any(v is None for v in cur.values()):
                continue
            checked += 1
            vals = {k: cur[k]["_v"] for k in cur}
            if ok(vals["P"], vals["U"], vals["M"], vals["W"]):
                continue
            # find single substitution from Mistral that fixes it
            cands = []
            for k in ("U", "M", "W", "P"):
                mv = m.get((a, s, k))
                if mv is None or mv == vals[k]:
                    continue
                trial = dict(vals); trial[k] = mv
                if ok(trial["P"], trial["U"], trial["M"], trial["W"]):
                    cands.append((k, mv))
            if len(cands) == 1:
                k, mv = cands[0]
                idx[(a, s, k)]["value"] = str(mv); idx[(a, s, k)]["_v"] = mv
                changes += 1
            else:
                still_bad += 1
        if changes and still_bad == 0 and checked:
            # table now fully passes -> emit corrected facts
            for f in facts:
                row = {c: f.get(c, "") for c in B.OUT_COLS if c in f}
                row["fact_id"] = f"X{fid:09d}"; fid += 1
                row["status"] = "corrected"
                new_facts.append(row)
            reconciled.append({"table_id": tid, "verdict": r["status"], "cells_fixed": changes})

    # merge with existing corrections (keep old; replace reconciled tables)
    import json
    cf = os.path.join(V1, "corrected_facts.csv"); ct = os.path.join(V1, "corrected_tables.json")
    rec_ids = {x["table_id"] for x in reconciled}
    prev = [r for r in csv.DictReader(open(cf))] if os.path.exists(cf) else []
    prev = [r for r in prev if r.get("table_id") not in rec_ids]
    prev_counts = (json.load(open(ct)).get("counts", {}) if os.path.exists(ct) else {})
    counts = {k: v for k, v in prev_counts.items() if k not in rec_ids}
    for x in reconciled:
        counts[x["table_id"]] = x["cells_fixed"]
    with open(cf, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=B.OUT_COLS, extrasaction="ignore")
        w.writeheader(); [w.writerow(r) for r in prev + new_facts]
    json.dump({"corrected": list(counts), "counts": counts}, open(ct, "w"))
    with open(os.path.join(V1, "reconciled.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["table_id", "verdict", "cells_fixed"]); w.writeheader()
        [w.writerow(x) for x in reconciled]
    print(f"failing tables with a Mistral read: examined")
    print(f"  RECONCILED to passing (dual-supported corrections): {len(reconciled)} tables")
    print(f"  corrected cells written: {len(new_facts)} fact rows into the overlay")
    print(f"  total corrected tables in overlay now: {len(counts)}")
    print("These fold in on the next build_explorer_db (e.g. the page-ordering window's reingest).")
    print("reconciled.csv lists them for spot-checking in the review app.")


if __name__ == "__main__":
    main()
