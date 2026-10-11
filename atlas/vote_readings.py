#!/usr/bin/env python3
"""Cell-level vote between two independent readings of a table (Gemini 3.8 Flash, out/reread/;
Gemini 3.1 Pro, out/reread_pro/).

Cells are matched by (row label, the set of words in the column headers) so that the two
readings may order header levels differently. A cell is published when both readings give the
same value. A disagreement is resolved by the table's own printed arithmetic: the candidate that
makes its column add up to the printed Total row, or its row add up to the row's printed total
(or to a 100/1,000/10,000 base) wins. Unresolved cells are withheld.

vote(flash_read, pro_read) -> (cells, stats)
  cells: list of (section, row label, group, sub, value) taken from the Flash reading's layout
  stats: matched, agreed, resolved, withheld, flash_cells, pro_cells
"""
import re, collections

TOT = re.compile(r"(?i)\b(total|all ages|both sexes|persons)\b")
BASES = (100.0, 1000.0, 10000.0)


def _n(s):
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def _words(*parts):
    return frozenset(w for p in parts for w in re.findall(r"[a-z]+|\d+", (p or "").lower()))


def cells_of(read):
    out = []
    for si, sec in enumerate(read["sections"]):
        for ri, row in enumerate(sec["rows"]):
            for ci, col in enumerate(sec["columns"]):
                v = row["values"][ci] if ci < len(row["values"]) else None
                if v is not None:
                    out.append(dict(si=si, ri=ri, ci=ci, sec=sec["name"], row=row["label"], group=col["group"],
                                    sub=col["sub"], value=float(v), key=(_n(row["label"]), _words(col["group"], col["sub"]))))
    return out


def _index(cells):
    idx = collections.defaultdict(list)
    for c in cells:
        idx[c["key"]].append(c)
    return idx


def _fits(read, c, value):
    """does `value` at cell c make the table's printed arithmetic hold (column vs Total row, or
    row vs its Total column / base)? returns True, False, or None if no constraint applies"""
    sec = read["sections"][c["si"]]
    rows, cols = sec["rows"], sec["columns"]
    verdicts = []
    # column constraint: a Total row below c
    for ti, tr in enumerate(rows):
        if ti > c["ri"] and TOT.search(tr["label"] or "") and c["ci"] < len(tr["values"]) and tr["values"][c["ci"]] is not None:
            body = [r for r in rows[:ti] if not TOT.search(r["label"] or "")]
            vals = []
            for r in body:
                if c["ci"] < len(r["values"]) and r["values"][c["ci"]] is not None:
                    vals.append(value if r is rows[c["ri"]] else r["values"][c["ci"]])
            if len(vals) >= 2:
                verdicts.append(abs(sum(vals) - tr["values"][c["ci"]]) <= 0.5 + 1e-6 * abs(tr["values"][c["ci"]]))
            break
    # row constraint: a Total column in the same row, summing the columns that share its group
    row = rows[c["ri"]]
    for tj, tc in enumerate(cols):
        if tj != c["ci"] and TOT.search(tc["sub"] or "") and (tc["group"] or "") == (cols[c["ci"]]["group"] or "") \
                and tj < len(row["values"]) and row["values"][tj] is not None:
            same = [j for j, cc in enumerate(cols) if cc["group"] == tc["group"] and j != tj and j < len(row["values"])
                    and row["values"][j] is not None]
            if c["ci"] in same and len(same) >= 2:
                s = sum(value if j == c["ci"] else row["values"][j] for j in same)
                verdicts.append(abs(s - row["values"][tj]) <= 0.5 + 1e-6 * abs(row["values"][tj]))
            break
    if not verdicts:
        # base constraint: the row's cells within the group sum to a printed base
        grp = cols[c["ci"]]["group"]
        same = [j for j, cc in enumerate(cols) if cc["group"] == grp and j < len(row["values"]) and row["values"][j] is not None]
        if len(same) >= 2:
            s = sum(value if j == c["ci"] else row["values"][j] for j in same)
            if any(abs(s - b) <= 0.6 for b in BASES):
                verdicts.append(True)
    return None if not verdicts else all(verdicts)


def _same_shape(fr, pr, si):
    """section si has the same number of rows and columns in both readings (positions comparable)"""
    if si >= len(pr["sections"]):
        return False
    a, b = fr["sections"][si], pr["sections"][si]
    return len(a["rows"]) == len(b["rows"]) and len(a["columns"]) == len(b["columns"])


WITHHELD = []      # (section, group, sub) of columns with a withheld cell in the last vote()


def vote(fr, pr):
    WITHHELD.clear()
    fc, pc = cells_of(fr), cells_of(pr)
    pidx = _index(pc)
    ppos = {(c["si"], c["ri"], c["ci"]): c for c in pc}
    shape = {si: _same_shape(fr, pr, si) for si in range(len(fr["sections"]))}
    used = collections.Counter()
    out, st = [], collections.Counter(flash_cells=len(fc), pro_cells=len(pc))
    for c in fc:
        lst = pidx.get(c["key"], [])
        k = used[c["key"]]; used[c["key"]] += 1
        if k < len(lst):
            p = lst[k]
        elif shape.get(c["si"]) and (c["si"], c["ri"], c["ci"]) in ppos:
            p = ppos[(c["si"], c["ri"], c["ci"])]; st["matched_by_position"] += 1
        else:
            st["unmatched"] += 1; WITHHELD.append((c["sec"], c["group"], c["sub"]))
            continue
        st["matched"] += 1
        if c["value"] == p["value"]:
            st["agreed"] += 1; out.append(c); continue
        ff, pf = _fits(fr, c, c["value"]), _fits(fr, c, p["value"])
        if ff and not pf:
            st["resolved"] += 1; out.append(c)
        elif pf and not ff:
            st["resolved"] += 1; out.append(dict(c, value=p["value"]))
        else:
            st["withheld"] += 1; WITHHELD.append((c["sec"], c["group"], c["sub"]))
    keep = _pmf_guard(out, st)
    WITHHELD.extend((c["sec"], c["group"], c["sub"]) for c in out if c not in keep)
    out = keep
    # (section, row label, group, sub, value, (section index, row index)) -- the position keeps
    # rows apart that share a label (e.g. a "Total" row repeated for every district)
    return [(c["sec"], c["row"], c["group"], c["sub"], c["value"], (c["si"], c["ri"])) for c in out], dict(st)


def _sexof(sub):
    t = (sub or "").lower()
    return "F" if re.search(r"female", t) else "M" if re.search(r"\bmales?\b", t) else "P" if re.search(r"persons|total|both", t) else None


def _one_digit(a, b):
    x, y = str(int(round(a))), str(int(round(b)))
    return len(x) == len(y) and sum(i != j for i, j in zip(x, y)) == 1


def _pmf_guard(out, st):
    """even cells both readings agree on are withheld when the printed identity Persons = Males +
    Females fails and the cell is one digit away from satisfying it (a shared misread)"""
    trip = collections.defaultdict(dict)
    for i, c in enumerate(out):
        sx = _sexof(c["sub"])
        if sx:                                   # sexes as columns
            trip[("c", c["si"], c["ri"], c["group"])][sx] = i
        rx = _sexof(c["row"])
        if rx:                                   # sexes as rows (transposed tables)
            trip[("r", c["si"], c["ci"])][rx] = i
    drop = set()
    for d in trip.values():
        if set(d) >= {"P", "M", "F"}:
            P, M, F = (out[d[k]]["value"] for k in "PMF")
            if abs(P - (M + F)) > 0.5:
                for k, want in (("P", M + F), ("M", P - F), ("F", P - M)):
                    if _one_digit(out[d[k]]["value"], want):
                        drop.add(d[k])
    st["withheld_by_arithmetic"] = len(drop)
    return [c for i, c in enumerate(out) if i not in drop]


def _match(fr, other):
    """index of each cell of reading fr -> the matching cell of `other` (label, then position)"""
    fc, oc = cells_of(fr), cells_of(other)
    oidx = _index(oc)
    opos = {(c["si"], c["ri"], c["ci"]): c for c in oc}
    shape = {si: _same_shape(fr, other, si) for si in range(len(fr["sections"]))}
    used = collections.Counter(); m = {}
    for i, c in enumerate(fc):
        lst = oidx.get(c["key"], [])
        k = used[c["key"]]; used[c["key"]] += 1
        if k < len(lst):
            m[i] = lst[k]
        elif shape.get(c["si"]) and (c["si"], c["ri"], c["ci"]) in opos:
            m[i] = opos[(c["si"], c["ri"], c["ci"])]
    return fc, m


def vote3(fr, pr, tr):
    """majority of three independent readings per cell (Flash layout): a value is published when
    at least two readings give it; otherwise the printed totals decide, else it is withheld"""
    WITHHELD.clear()
    fc, mp = _match(fr, pr)
    _, mt = _match(fr, tr)
    out, st = [], collections.Counter(flash_cells=len(fc))
    for i, c in enumerate(fc):
        vals = [c["value"]] + [m[i]["value"] for m in (mp, mt) if i in m]
        cnt = collections.Counter(vals)
        v, n = cnt.most_common(1)[0]
        if n >= 2:
            st["agreed"] += 1; out.append(dict(c, value=v))
        else:
            fits = [x for x in set(vals) if _fits(fr, c, x)]
            if len(fits) == 1:
                st["resolved"] += 1; out.append(dict(c, value=fits[0]))
            else:
                st["withheld"] += 1; WITHHELD.append((c["sec"], c["group"], c["sub"]))
    keep = _pmf_guard(out, st)
    WITHHELD.extend((c["sec"], c["group"], c["sub"]) for c in out if c not in keep)
    return [(c["sec"], c["row"], c["group"], c["sub"], c["value"], (c["si"], c["ri"])) for c in keep], dict(st)
