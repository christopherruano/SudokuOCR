#!/usr/bin/env python3
"""Parse Mistral-OCR markdown (standard Table VII orientation: Age in rows;
Population/Unmarried/Married/Widowed each x Persons/Males/Females in columns) into
our row schema {age, pop_m, pop_f, unm_m, unm_f, mar_m, mar_f, wid_m, wid_f}, split
into sections (A-Total, B-Hindu, ...). Output matches opus_reread_result.json so the
existing kind-aware validator (apply_opus_reread.py) accepts/rejects it — the
arithmetic gate (U+M+W=Population) is what catches Mistral's failures. We DO NOT
trust Mistral blindly; only constraint-valid reads survive.
Returns reads via mistral_parse(path) and, run directly, reports the constraint
pass-rate over the comparison sample (the watchfulness check)."""
import os, re, glob, json

GROUP = [("pop", re.compile(r"popul|persons? dealt|total population", re.I)),
         ("unm", re.compile(r"unmarr|single|never", re.I)),
         ("mar", re.compile(r"^marri|\bmarried", re.I)),
         ("wid", re.compile(r"widow", re.I))]
SUB = [("m", re.compile(r"male", re.I)), ("f", re.compile(r"female", re.I))]
SECT = re.compile(r"^[A-Z][—\-—.\s]*([A-Z][A-Za-z' ]+)\.?\s*$")
NUM = re.compile(r"^-?[\d,]+$")
AGE = re.compile(r"\d+\s*[-—–]\s*\d+|\d+\s*and over|under|total|all ages", re.I)


def _cells(line):
    return [c.strip() for c in line.strip().strip("|").split("|")]


def _group_of(name):
    for k, rx in GROUP:
        if rx.search(name):
            return k
    return None


def mistral_parse(path):
    """Return list of sections [{name, rows:[{age, pop_m,...}]}] for one .mistral.md."""
    lines = open(path).read().splitlines()
    sections, cur_name, i = [], "", 0
    while i < len(lines):
        ln = lines[i]
        s = ln.strip()
        m = SECT.match(s)
        if s and "|" not in s and m and len(s) < 40:
            cur_name = m.group(1).strip()
            i += 1; continue
        if s.startswith("|"):
            # gather the markdown table block
            block = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                block.append(lines[i]); i += 1
            rows = _parse_block(block)
            if rows:
                sections.append({"name": cur_name, "rows": rows})
            continue
        i += 1
    return sections


def _parse_block(block):
    grid = [_cells(b) for b in block if "---" not in b]
    if len(grid) < 3:
        return []
    # find the two header rows: one with group names, one with Persons/Males/Females
    hdr_grp = hdr_sub = None
    for r in range(min(3, len(grid))):
        joined = " ".join(grid[r]).lower()
        if hdr_grp is None and any(g[1].search(joined) for g in GROUP):
            hdr_grp = grid[r]
        if hdr_sub is None and ("male" in joined or "persons" in joined):
            hdr_sub = grid[r]
    if not hdr_grp:
        return []
    ncol = max(len(hdr_grp), len(hdr_sub or []))
    # column index -> civil-group: forward-fill group names across their span
    colgrp = {}
    last = None
    for c in range(ncol):
        nm = hdr_grp[c] if c < len(hdr_grp) else ""
        g = _group_of(nm)
        if g:
            last = g
        colgrp[c] = last
    # column index -> sub (m/f); from the Persons/Males/Females header
    colsub = {}
    if hdr_sub:
        for c in range(len(hdr_sub)):
            for k, rx in SUB:
                if rx.search(hdr_sub[c]):
                    colsub[c] = k
    rows = []
    for g in grid:
        if not g or not AGE.search(g[0]):
            continue
        age = g[0]
        rec = {"age": age}
        for c in range(1, min(len(g), ncol)):
            grp, sub = colgrp.get(c), colsub.get(c)
            if not grp or not sub:
                continue
            v = g[c].replace(",", "")
            if NUM.match(g[c]):
                rec[f"{grp}_{sub}"] = int(v)
        if len(rec) > 1:
            rows.append(rec)
    return rows


def _check(sections):
    """U+M+W == Population per (age,sex). Returns (checked, bad)."""
    checked = bad = 0
    for sec in sections:
        for r in sec["rows"]:
            for sx in ("m", "f"):
                P, U, M, W = (r.get(f"pop_{sx}"), r.get(f"unm_{sx}"), r.get(f"mar_{sx}"), r.get(f"wid_{sx}"))
                if None in (P, U, M, W):
                    continue
                checked += 1
                bad += abs(P - (U + M + W)) > max(2, 0.02 * max(1, P))
    return checked, bad


def main():
    V1 = "export_final/sendout_final/sendout_v1"
    md = sorted(glob.glob(f"{V1}/comparison/*.mistral.md"))
    print(f"parsing {len(md)} Mistral markdown reads + checking U+M+W=Population...\n")
    tot_ok = tot_part = tot_none = 0
    for p in md:
        secs = mistral_parse(p)
        nrows = sum(len(s["rows"]) for s in secs)
        checked, bad = _check(secs)
        verdict = ("NO_CONSTRAINT" if checked == 0 else
                   "PASS" if bad <= 0.05 * checked else
                   "PARTIAL" if bad <= 0.25 * checked else "FAIL")
        if verdict == "PASS": tot_ok += 1
        elif verdict in ("PARTIAL", "NO_CONSTRAINT"): tot_part += 1
        else: tot_none += 1
        print(f"  {os.path.basename(p)[:14]} | secs={len(secs)} rows={nrows:3d} | UMW=P checked={checked:3d} bad={bad:3d} -> {verdict}")
    print(f"\nMistral constraint pass: full={tot_ok}  partial/none={tot_part}  fail={tot_none}  (of {len(md)})")
    print("Only PASS reads would be accepted as corrections; PARTIAL/FAIL are caught by the gate.")


if __name__ == "__main__":
    main()
