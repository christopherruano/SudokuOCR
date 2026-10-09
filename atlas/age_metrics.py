#!/usr/bin/env python3
"""Age-structure indicators from heterogeneous census age tables.

For one (table, place) the facts are grouped into *slices*: every combination of
the non-age, non-sex dimensions (section, civil condition, religion, and the
column group when the column axis is not age). The slice describing the whole
population is the one with the largest all-ages total. Within it we choose a
set of non-overlapping age bands that tiles [0, top) and accept the slice only
if the tiled bands sum to the printed all-ages total (within TOL) — the table
checks itself. From the accepted tiling:

  child    % of persons of stated age aged 0-9 (needs a band boundary at 10)
  old      % of persons aged 60+   (needs a band boundary at 60)
  sexratio females per 1,000 males (all-ages totals)
"""
import collections, re

TOL = 0.02
CC = re.compile(r"\b(unmarried|single|married|mar-|widow\w*|widow-)")
SEXES = {"Male": "m", "Female": "f", "Persons": "p", "Total": "p", "": "p", None: "p"}


def _tiling(bands):
    """bands: {(lo, hi_or_None): value}. Greedy left-to-right cover from 0 using the
    finest band starting at each point. Handles single years ("5" = [5,6)) and
    inclusive printing ("0-4, 5-9" = [0,5),[5,10)). Returns [(lo, hi, v)] or None."""
    by_lo = collections.defaultdict(list)
    for (lo, hi), v in bands.items():
        if lo is None:
            continue
        hi = 999 if hi is None else (lo + 1 if hi == lo else hi)
        if hi <= lo:
            continue
        by_lo[lo].append((hi, v))
    cur, out = 0, []
    while cur in by_lo:
        hi, v = min(by_lo[cur])
        if hi < 999 and hi not in by_lo and hi + 1 in by_lo:
            hi += 1                                   # inclusive upper bound
        out.append((cur, hi, v))
        if hi >= 999:
            return out
        cur = hi
    return out or None


OPEN_UP = re.compile(r"(?i)^\s*(\d{1,3})\s*(?:years?\s*)?(?:and|&)\s*(?:upwards?|over|above|more)|"
                     r"^\s*(?:above|over)\s*(\d{1,3})|^\s*(\d{1,3})\s*\+\s*$")
UNDER = re.compile(r"(?i)^\s*(?:under|below|less than)\s*(\d{1,2})\b")


def parse_band(label):
    """Fallback for bands the resolver left without bounds: '60 and Upwards' -> (60, None),
    'above 55' -> (55, None), 'under 1' -> (0, 1). Returns (lo, hi) or None."""
    lab = str(label or "")
    m = OPEN_UP.search(lab)
    if m:
        return int(next(g for g in m.groups() if g)), None
    m = UNDER.search(lab)
    if m:
        return 0, int(m.group(1))
    return None


NOT_STATED = re.compile(r"(?i)not stated|unknown|unspecified|not returned")


def _shares(til, denom):
    m = {}
    if 10 in {hi for _, hi, _ in til}:
        m["child"] = 100 * sum(v for lo, hi, v in til if hi <= 10) / denom
    if 60 in {lo for lo, _, _ in til}:
        m["old"] = 100 * sum(v for lo, hi, v in til if lo >= 60) / denom
    return m


def slice_metrics(rows, unit="count"):
    """rows: dicts with sex, lo, hi, label, value for ONE slice. Returns metrics or None.
    count tables: tiled bands (+ 'age not stated') must equal the printed total;
    shares are of persons of stated age. rate tables (per 1,000 / 10,000): the tiled
    bands must sum to the base."""
    per = {"m": {}, "f": {}, "p": {}}
    tot = {"m": None, "f": None, "p": None}
    ns = {"m": 0.0, "f": 0.0, "p": 0.0}
    for r in rows:
        s = SEXES.get(r["sex"], None)
        if s is None or r["value"] is None:
            continue
        if r["lo"] is None:
            lab = str(r["label"] or "")
            if NOT_STATED.search(lab):
                ns[s] += r["value"]
            elif lab.lower().startswith("total") or lab in ("All ages",):
                tot[s] = max(tot[s] or 0, r["value"])
            continue
        per[s][(r["lo"], r["hi"])] = r["value"]
    if unit == "rate":
        res = []
        for s in ("p", "m", "f"):
            til = _tiling(per[s]) if per[s] else None
            if not til or til[-1][1] < 999:
                continue
            tsum = sum(v for _, _, v in til)
            base = next((b for b in (100, 1000, 10000, 100000) if abs(tsum - b) / b <= TOL), None)
            if base:
                res.append((s, _shares(til, tsum), base))
        if not res:
            return None
        if res[0][0] == "p":
            m, how = res[0][1], f"distribution per {res[0][2]:,} sums to base"
        else:
            keys = set.intersection(*[set(x[1]) for x in res])
            m = {k: sum(x[1][k] for x in res) / len(res) for k in keys}
            how = f"per-{res[0][2]:,} distributions sum to base; mean of {len(res)} sexes"
        if not m:
            return None
        m.update(total=0, check=how, unit="rate")
        return m
    if per["m"] and per["f"]:
        keys = set(per["m"]) & set(per["f"])
        bands = {k: per["m"][k] + per["f"][k] for k in keys}
        total = (tot["m"] + tot["f"]) if tot["m"] and tot["f"] else tot["p"]
        nst = ns["m"] + ns["f"] or ns["p"]
    else:
        bands, total, nst = per["p"], tot["p"], ns["p"]
    til = _tiling(bands)
    if not til:
        return None
    s = sum(v for _, _, v in til)
    if total is None:
        if til[-1][1] < 999:
            return None
        total, check = s + nst, "open-ended bands, no printed total"
    else:
        if total <= 0 or abs(s + nst - total) / total > TOL:
            return None
        check = f"bands{' + not-stated' if nst else ''} = printed total ({s + nst:,.0f}/{total:,.0f})"
    if s < 50:
        return None
    m = _shares(til, s)
    if not m:
        return None
    m.update(total=total, check=check, unit="count")
    if tot["m"] and tot["f"]:
        m["sexratio"] = 1000 * tot["f"] / tot["m"]
    elif per["m"] and per["f"]:
        sm, sf = sum(per["m"].values()), sum(per["f"].values())
        if sm:
            m["sexratio"] = 1000 * sf / sm
    return m


def best_metrics(facts):
    """facts: list of dicts for one (table, place) with keys section, civil_condition,
    religion, column_dimension, column_group_std, sex, resolved_age_lower/upper,
    resolved_age_label_std, value, measure_unit. Returns the metrics of the largest
    self-consistent slice, or None."""
    sl = collections.defaultdict(list)
    for f in facts:
        unit = "count" if f.get("measure_unit") in (None, "", "count") else "rate" if f.get("measure_unit") == "rate" else None
        if unit is None:
            continue
        cg = f["column_group_std"] if f.get("column_dimension") != "age" else ""
        cc = f.get("civil_condition") or ""
        if re.fullmatch(r"(?i)\s*(population|total|all|persons|all ages)?\.?\s*", cc):
            cc = ""                                   # the slice's total, not a civil condition
        key = (unit, f.get("section") or "", cc, f.get("religion") or "", cg or "")
        lo, hi, lab = f.get("resolved_age_lower"), f.get("resolved_age_upper"), f.get("resolved_age_label_std")
        if lo is None:
            raw = f.get("column_group_std") if f.get("column_dimension") == "age" else f.get("row_label_std")
            pb = parse_band(lab) or parse_band(raw)
            if pb:
                lo, hi = pb
            elif not lab:
                lab = raw
        sl[key].append({"sex": f.get("sex"), "lo": lo, "hi": hi, "label": lab, "value": f.get("value")})
    # Slices that are a single civil condition (unmarried / married / widowed) are
    # parts of the population, not the whole: never pick one alone, but offer
    # their sum (per remaining key) as a synthetic whole-population slice.
    cand = {}
    merged = collections.defaultdict(list)
    for key, rows in sl.items():
        text = " ".join(key[1:]).lower()
        cc = CC.search(text)
        if cc:
            rest = (key[0],) + tuple(CC.sub("", k).strip(" .-") for k in key[1:])
            w = cc.group(0)
            kind = "unm" if w in ("unmarried", "single") else "wid" if w.startswith("widow") else "mar"
            merged[rest].append((kind, rows))
        else:
            cand[key] = rows
    for rest, parts in merged.items():
        kinds = {k for k, _ in parts}
        if {"unm", "mar", "wid"} <= kinds:
            summed = collections.defaultdict(float)
            meta = {}
            for _, rows in parts:
                for r in rows:
                    if r["value"] is None:
                        continue
                    k = (r["sex"], r["lo"], r["hi"], r["label"])
                    summed[k] += r["value"]; meta[k] = r
            cand[rest + ("sum of civil conditions",)] = [
                dict(meta[k], value=v) for k, v in summed.items()]
    best = None
    for key, rows in cand.items():
        m = slice_metrics(rows, key[0])
        if m and (best is None or (m["unit"] == "count", m["total"]) > (best[1]["unit"] == "count", best[1]["total"])):
            best = (key, m)
    if not best:
        return None
    best[1]["slice"] = " / ".join(k for k in best[0][1:] if k) or "(whole table)"
    return best[1]
