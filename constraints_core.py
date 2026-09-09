#!/usr/bin/env python3
"""constraints_core.py — SINGLE SOURCE OF TRUTH for arithmetic-identity checking of digitized
census tables (oneshot JSON shape: {"metadata": {...}, "data": [{"name",..., "rows":[...]}]}).

Fixes the audited defects of the old census_review_app checkers:
  D1  transposed detection missed '-30.', 'OVER 60.', '15—19.' (em-dash), '0 to 9' headers
  D2  detected-transposed sections lost the M+F=Persons identity entirely
  D3  transposed sections without a Total/Grand/All-ages row got ZERO checks, silently
  D4  '0 violations' was indistinguishable from 'nothing was checked'
plus kind gating: count identities were fired at proportional/rate/ratio tables and vice versa.

Vocabulary
  layout (per SECTION): age_rows   — age lives on the row axis (classic Table VII)
                        transposed — age lives on the COLUMN-group axis
                        flat_area  — rows are areas/districts/religions, groups are not ages
                        unknown    — no dict-valued groups / no rows
  kind (per TABLE):     count | distribution | percentage | rate | ratio | index | text
                        rate/ratio/index/text are NOT_CHECKABLE (no arithmetic identity applies)
  status:               PASS / FAIL / NOT_CHECKABLE / NOT_ANCHORED
                        NOT_ANCHORED = identities apply in principle but ZERO fired (no anchor
                        row/keys) — "0 violations" NEVER means "0 checks" any more.

Public API
  detect_layout(rows)                       -> layout string
  classify_kind(meta, rows=None)            -> (kind, base)
  check_section(rows, kind, base, layout)   -> SectionReport
  check_table(d)                            -> TableReport   (honours metadata.human_verified
                                               and metadata.constraints_irrelevant short-circuits)
  count_violations(rows) / distribution_violations(rows, base) — legacy-shaped
                                               {(row_idx, group, subkey): reason} dicts
"""
import re
from dataclasses import dataclass, field

INF = 10 ** 9

# ---------------------------------------------------------------------------
# regexes (the audited, widened patterns)
# ---------------------------------------------------------------------------
# age-like COLUMN-GROUP name: 'AGE', '0-5', '15—19.' (en/em dash), '0 to 9', 'under 5',
# '60 & over', 'over 60', 'all ages', bare '-30.' (open-low band)
AGE_COL = re.compile(
    r"\bage\b|\d+\s*(?:[-–—]|to)\s*\d+|\bunder\b|\d+\s*&?\s*(?:and\s*)?over"
    r"|over\s*\d+|all\s*ages|^\W*[-–—]\s*\d+", re.I)
# age-like ROW label (anchored — row labels are terser than headers)
AGE_ROW = re.compile(
    r"^\W*\d+\s*(?:[-–—]|to)\s*\d+|^\W*under\b|^\W*\d+\s*(?:&|and)?\s*over|^\W*over\s*\d+", re.I)
TOT_RE = re.compile(r"\btotal\b|\bgrand\b|all ages", re.I)
# title says the numbers are proportional even when data_type claims 'absolute'
PROP_TITLE = re.compile(
    r"per\s*1[,.]?000|per\s*10[,.]?000|per\s*mille|percent|proportion\b|distribution of\s*(1|10)", re.I)
NOT_STATED = re.compile(r"unspecified|not\s*stated|unknown|un-?specified", re.I)

NOT_CHECKABLE_KINDS = ("rate", "ratio", "index", "text")

# identity ids
PMF = "pmf"                       # Males + Females = Persons, per group-cell
UMW_GROUP = "umw_group"           # Unmarried+Married+Widowed(+ns) = Total across GROUPS, per subkey
UMW_SUBKEY = "umw_subkey"         # same identity when civil condition lives in SUBKEYS
VERTICAL = "vertical"             # anchored column sums (data rows -> Total/Grand Total row)
AGE_CHAIN = "age_chain"           # age-hierarchy vertical for age_rows layouts
CROSS_GROUP = "cross_group"       # transposed: All-ages column-group = Σ age-band column-groups
SUM_TO_BASE = "sum_to_base"       # distribution/percentage: parts sum to the base
ALL_IDENTITIES = (PMF, UMW_GROUP, UMW_SUBKEY, VERTICAL, AGE_CHAIN, CROSS_GROUP, SUM_TO_BASE)
# STRUCTURAL identities assume an additivity model (which rows/groups sum where). The model must
# SELF-VALIDATE: in a section where it never passes once, the model — not the digits — is wrong
# (continuation pages, nested unlabeled subtotals, column-wise distributions), so its violations
# are withdrawn and the section counts as unanchored for that identity. LOCAL identities
# (M+F=P, U+M+W=T) are always valid for count tables and are never withdrawn.
STRUCTURAL_IDENTITIES = (VERTICAL, AGE_CHAIN, CROSS_GROUP, SUM_TO_BASE)


def _num(x):
    s = str(x).replace(",", "").strip()
    if re.fullmatch(r"-?\d+", s):
        return int(s)
    try:
        return float(s)
    except Exception:
        return None


def _row_header(rows):
    if not rows:
        return None
    return next((k for k in rows[0] if not isinstance(rows[0].get(k), dict)), None)


def _dict_groups(rows):
    groups = []
    for r in rows:
        for k, v in r.items():
            if isinstance(v, dict) and k not in groups:
                groups.append(k)
    return groups


# ---------------------------------------------------------------------------
# layout
# ---------------------------------------------------------------------------
def detect_layout(rows):
    """age_rows | transposed | flat_area | unknown, per SECTION.

    Transposed iff >=3 dict groups and >=50% of group names look age-like — BUT the mandatory
    age_rows guard wins first: if >=40% of the ROW labels are age-like, the rows are the age axis
    (prevents the 424 structural false positives from age-band subtotal double-counting)."""
    if not rows:
        return "unknown"
    groups = _dict_groups(rows)
    rh = _row_header(rows)
    labels = []
    if rh is not None:
        labels = [str(r.get(rh, "") or "") for r in rows]
        labels = [l for l in labels if l.strip()]
    if labels and sum(1 for l in labels if AGE_ROW.search(l)) >= 0.4 * len(labels):
        return "age_rows"                                   # MANDATORY guard: rows are the age axis
    if len(groups) >= 3 and sum(1 for g in groups if AGE_COL.search(g or "")) >= 0.5 * len(groups):
        return "transposed"
    if groups:
        return "flat_area"
    return "unknown"


# ---------------------------------------------------------------------------
# kind
# ---------------------------------------------------------------------------
_KIND_NORM = {
    "absolute": "count", "count": "count", "counts": "count", "raw": "count",
    "proportional": "_prop", "per_n": "_prop", "per_1000": "_prop", "per_10000": "_prop",
    "percentage": "percentage", "percent": "percentage",
    "ratio": "ratio", "rate": "rate", "average": "rate", "index": "index",
    "text": "text", "map": "text",
    "none": "", "mixed": "", "": "",
}
_RATIO_KWS = ("females per", "female per", "males per", "per 1,000 males", "per 1000 males",
              "per 100 males", "to 100 males", "sex ratio", "ratio of", " ratio")
_RATE_KWS = ("death rate", "birth rate", "death-rate", "birth-rate", "per mille",
             "mortality rate", "rate per", "rate of", "vital rate")
_PCT_KWS = ("per cent", "per-cent", "percent", "percentage", "%")


def classify_kind(meta, rows=None):
    """(kind, base) for a table. Ports oneshot.infer_table_kind, plus:
    - off-enum data_type values are normalized to the nearest kind
    - a proportional TITLE (per 1,000 / per 10,000 / per mille / percent / proportion /
      distribution of 1[0],000) OVERRIDES data_type='absolute' (610 mislabeled files audited)."""
    meta = meta or {}
    dt = str(meta.get("data_type") if meta.get("data_type") is not None else "").strip().lower()
    title = (meta.get("title") or "")
    tl = title.lower()
    denom = meta.get("denominator") or 0
    try:
        denom = int(denom)
    except Exception:
        denom = 0

    kind0 = _KIND_NORM.get(dt, "")            # off-enum garbage -> '' -> infer below
    if kind0 in ("count", "") and (PROP_TITLE.search(title) or denom in (100, 1000, 10000)):
        kind0 = "_prop"                       # title/denominator override
    if kind0 == "count":
        return "count", None
    if kind0 == "percentage":
        return "percentage", 100
    if kind0 in NOT_CHECKABLE_KINDS:
        return kind0, None
    if kind0 == "":
        return "count", None                  # dtype not proportional, no denominator -> count

    # proportional family — disambiguate by title/denominator (port of oneshot.infer_table_kind)
    if denom == 100 or any(k in tl for k in _PCT_KWS):
        return "percentage", 100
    if any(k in tl for k in _RATIO_KWS):
        return "ratio", None
    if any(k in tl for k in _RATE_KWS):
        return "rate", None
    if "index" in tl:
        return "index", None
    base = None
    if re.search(r"per\s*10[,.]?000|of\s*10[,.]?000", tl):
        base = 10000
    elif re.search(r"per\s*1[,.]?000|per\s*mille|of\s*1[,.]?000", tl):
        base = 1000
    elif denom in (1000, 10000):
        base = denom
    return "distribution", base


# ---------------------------------------------------------------------------
# age-band parsing + hierarchy chain
# ---------------------------------------------------------------------------
def parse_age_span(label):
    """'5-10'/'5 to 10'/'15—19.' -> (5,10|19); 'under 5' -> (0,5); '60 and over'/'over 60'/'60+'
    -> (60,INF); bare '-30.' (open-low band) -> (None,30); '7' -> (7,7);
    'Total'/'All ages' (rangeless) -> 'TOTAL'; anything else -> None."""
    s = str(label or "").strip().lower()
    s = s.replace("—", "-").replace("–", "-")
    s = re.sub(r"[.·]+\s*$", "", s).strip()
    if not s:
        return None
    m = re.search(r"(\d+)\s*(?:-|to)\s*(\d+)", s)
    if m:
        lo, hi = int(m.group(1)), int(m.group(2))
        return (lo, hi) if hi >= lo else None
    m = re.search(r"\b(?:under|below)\s*(\d+)", s)
    if m:
        return (0, int(m.group(1)))
    m = re.search(r"(\d+)\s*(?:&|and)?\s*(?:and\s*)?(?:over|upwards|above)\b|\bover\s*(\d+)|(\d+)\s*\+", s)
    if m:
        lo = int(next(g for g in m.groups() if g))
        return (lo, INF)
    if TOT_RE.search(s):
        return "TOTAL"
    m = re.fullmatch(r"-\s*(\d+)", s)
    if m:
        return (None, int(m.group(1)))        # open-low band ('-30.' = up to 30)
    m = re.fullmatch(r"(\d+)", s)
    if m:
        v = int(m.group(1))
        return (v, v)                          # single year of age
    return None


def build_age_chain(items, require_complete=False):
    """items: [(key, span)] with span=(lo,hi) tuples (lo may be None = open-low). Return the list of
    keys forming ONE disjoint leaf chain that tiles the age axis, or None when the structure is
    ambiguous (overlaps that aren't clean subtotals, gaps, duplicated bands). Subtotal rows —
    spans strictly containing chain members — are tolerated but NOT summed. This is the
    age-hierarchy model: it is what stops naive double-counting of age-band subtotals.

    require_complete: additionally demand the chain covers ALL ages — starts at 0/1 and ends in an
    open band ('N and over'). An 'all ages' Total may only be anchored by a complete chain;
    continuation pages (bands 30-35… with the previous page's bands missing) must NOT anchor it."""
    pool = [(k, s) for k, s in items if isinstance(s, tuple)]
    if len(pool) < 2:
        return None
    chain, c = [], None
    while pool:
        if c is None:                               # choose the starting band
            finite = [x for x in pool if x[1][0] is not None]
            if finite:
                mlo = min(x[1][0] for x in finite)
                cands = [x for x in finite if x[1][0] == mlo]
            else:
                cands = pool[:]                     # all open-low: sequential by upper bound
        else:
            cands = [x for x in pool
                     if (x[1][0] is not None and x[1][0] in (c, c + 1))
                     or (x[1][0] is None and x[1][1] > c)]
        if not cands:
            break
        pick = min(cands, key=lambda x: (x[1][1], x[1][0] if x[1][0] is not None else -1))
        chain.append(pick)
        c = pick[1][1]
        pool.remove(pick)
        if c >= INF:
            break
    if len(chain) < 2:
        return None
    lo0 = chain[0][1][0] if chain[0][1][0] is not None else 0
    for k, s in pool:                               # leftovers must be contained subtotals
        slo = s[0] if s[0] is not None else lo0
        if not (slo >= lo0 and s[1] <= c):
            return None                             # a band outside the chain -> ambiguous, refuse
    if require_complete and not (lo0 <= 1 and c >= INF):
        return None                                 # partial coverage (continuation page) -> no anchor
    return [k for k, s in chain]


# ---------------------------------------------------------------------------
# reports
# ---------------------------------------------------------------------------
@dataclass
class Violation:
    row: int
    group: str
    subkey: str
    identity: str
    lhs: object = None
    rhs: object = None
    magnitude: float = 0
    reason: str = ""


@dataclass
class SectionReport:
    name: str = ""
    layout: str = "unknown"
    kind: str = "count"
    status: str = "NOT_ANCHORED"
    n_rows: int = 0
    numeric_cells: int = 0
    checked_cells: int = 0
    coverage: float = 0.0
    stats: dict = field(default_factory=dict)      # identity -> {"n_checked", "n_pass"}
    violations: list = field(default_factory=list)
    suppressed: dict = field(default_factory=dict)  # identity -> n_checked withdrawn (model invalid)

    @property
    def fail_rows(self):
        return {v.row for v in self.violations}

    def viol_map(self):
        """legacy shape: {(row, group, subkey): reason}"""
        return {(v.row, v.group, v.subkey): v.reason for v in self.violations}


@dataclass
class TableReport:
    kind: str = "count"
    base: object = None
    status: str = "NOT_ANCHORED"
    human_verified: bool = False
    constraints_irrelevant: bool = False
    sections: list = field(default_factory=list)
    stats: dict = field(default_factory=dict)
    numeric_cells: int = 0
    checked_cells: int = 0
    coverage: float = 0.0
    n_fail_rows: int = 0                            # distinct (section, row) pairs failing

    @property
    def violations(self):
        return [v for s in self.sections for v in s.violations]


class _Ctx:
    """violation/stat accumulator for one section pass."""

    def __init__(self):
        self.stats = {}
        self.violations = []
        self.cells_by_ident = {}                    # ident -> {(row, group, subkey)} cells touched

    @property
    def checked(self):
        out = set()
        for s in self.cells_by_ident.values():
            out |= s
        return out

    def fired(self, ident, ok, cells, viols=()):
        st = self.stats.setdefault(ident, {"n_checked": 0, "n_pass": 0})
        st["n_checked"] += 1
        if ok:
            st["n_pass"] += 1
        self.cells_by_ident.setdefault(ident, set()).update(cells)
        self.violations.extend(viols)

    def suppress(self, ident):
        """Withdraw an identity whose structural model failed to validate in this section."""
        n = self.stats.pop(ident, {"n_checked": 0})["n_checked"]
        self.cells_by_ident.pop(ident, None)
        self.violations = [v for v in self.violations if v.identity != ident]
        return n


# ---------------------------------------------------------------------------
# the identities
# ---------------------------------------------------------------------------
def _pk_mk_fk(v):
    pk = next((k for k in v if re.search(r"person|total|both", k, re.I)), None)
    mk = next((k for k in v if re.search(r"male", k, re.I) and not re.search(r"female", k, re.I)), None)
    fk = next((k for k in v if re.search(r"female", k, re.I)), None)
    return pk, mk, fk


def _check_pmf(rows, ctx):
    """M+F=Persons per group-cell — ALL layouts (the transposed layouts lost this before)."""
    for ri, r in enumerate(rows):
        for g, v in r.items():
            if not isinstance(v, dict):
                continue
            pk, mk, fk = _pk_mk_fk(v)
            if not (pk and mk and fk):
                continue
            p, m, f = _num(v.get(pk)), _num(v.get(mk)), _num(v.get(fk))
            if None in (p, m, f):
                continue
            cells = [(ri, g, kk) for kk in (pk, mk, fk)]
            if m + f == p:
                ctx.fired(PMF, True, cells)
            else:
                reason = f"M+F={m + f} ≠ P={p}"
                ctx.fired(PMF, False, cells,
                          [Violation(ri, g, kk, PMF, m + f, p, abs(m + f - p), reason)
                           for kk in (pk, mk, fk)])


def _civil_groups(groups):
    unm = next((g for g in groups if re.search(r"unmarried|single|never", g, re.I)), None)
    mar = next((g for g in groups if re.search(r"married", g, re.I) and not re.search(r"unmarried", g, re.I)), None)
    wid = next((g for g in groups if re.search(r"widow|divorc", g, re.I)), None)
    tot = next((g for g in groups if re.search(r"total|population|all ages|grand", g, re.I)), None)
    ns = next((g for g in groups if NOT_STATED.search(g)), None)
    return unm, mar, wid, tot, ns


def _check_umw_group(rows, ctx):
    """Unmarried+Married+Widowed(+not-stated) = Total across GROUPS, per subkey."""
    for ri, r in enumerate(rows):
        groups = {g: v for g, v in r.items() if isinstance(v, dict)}
        unm, mar, wid, tot, ns = _civil_groups(groups)
        if not (unm and mar and wid and tot):
            continue
        subs = set(groups[tot]) & set(groups[unm]) & set(groups[mar]) & set(groups[wid])
        for sk in subs:
            t, u, m, w = (_num(groups[tot].get(sk)), _num(groups[unm].get(sk)),
                          _num(groups[mar].get(sk)), _num(groups[wid].get(sk)))
            n = _num(groups[ns].get(sk)) if ns and isinstance(groups.get(ns), dict) else 0
            n = 0 if n is None else n
            if None in (t, u, m, w):
                continue
            gg = [tot, unm, mar, wid] + ([ns] if ns else [])
            cells = [(ri, g, sk) for g in gg]
            if u + m + w + n == t:
                ctx.fired(UMW_GROUP, True, cells)
            else:
                reason = f"U+M+W={u + m + w + n} ≠ Total={t}"
                ctx.fired(UMW_GROUP, False, cells,
                          [Violation(ri, g, sk, UMW_GROUP, u + m + w + n, t,
                                     abs(u + m + w + n - t), reason) for g in gg])


def _check_umw_subkey(rows, ctx):
    """U+M+W(+not stated)=Total when civil condition lives in the SUBKEYS of each group
    (the 'Total./Single./Married./Widowed./Civil condition not stated.' shape — D4)."""
    for ri, r in enumerate(rows):
        for g, v in r.items():
            if not isinstance(v, dict):
                continue
            tk = next((k for k in v if re.search(r"^total|persons", k.strip(), re.I)), None)
            uk = next((k for k in v if re.search(r"unmarr|single|never", k, re.I)), None)
            mk = next((k for k in v if re.search(r"marri", k, re.I) and not re.search(r"unmarr", k, re.I)), None)
            wk = next((k for k in v if re.search(r"widow", k, re.I)), None)
            if not (tk and uk and mk and wk):
                continue
            nsk = next((k for k in v if NOT_STATED.search(k)), None)
            t, u, m, w = _num(v.get(tk)), _num(v.get(uk)), _num(v.get(mk)), _num(v.get(wk))
            n = _num(v.get(nsk)) if nsk else 0
            n = 0 if n is None else n
            if None in (t, u, m, w):
                continue
            kk = [tk, uk, mk, wk] + ([nsk] if nsk else [])
            cells = [(ri, g, k) for k in kk]
            if u + m + w + n == t:
                ctx.fired(UMW_SUBKEY, True, cells)
            else:
                reason = f"U+M+W={u + m + w + n} ≠ Total={t}"
                ctx.fired(UMW_SUBKEY, False, cells,
                          [Violation(ri, g, k, UMW_SUBKEY, u + m + w + n, t,
                                     abs(u + m + w + n - t), reason) for k in kk])


def _check_vertical(rows, rh, ctx):
    """Anchored vertical (column-sum) identity for transposed/flat_area layouts: data rows sum to
    the Total/Grand-Total row per (group, subkey) column; earlier Totals = the sum of their block.
    The grand row accepts EITHER additive reading (sum of leaf rows, or sum of the block totals,
    or its own block) so a trailing block-total is not misread as a failing grand total."""
    if not rows or rh is None:
        return
    def cell(i, g, k):
        v = rows[i].get(g)
        return _num(v.get(k)) if isinstance(v, dict) else None
    def lbl(i):
        return str(rows[i].get(rh, "") or "")
    total_idx = [i for i in range(len(rows)) if TOT_RE.search(lbl(i))]
    if not total_idx:
        return
    cols = [(g, k) for g, v in rows[0].items() if isinstance(v, dict) for k in v]
    data_idx = [i for i in range(len(rows)) if i not in total_idx]

    def colsum(idxs, g, k):
        vals = [cell(i, g, k) for i in idxs]
        return None if (not vals or any(v is None for v in vals)) else sum(vals)

    grand = total_idx[-1]
    prev_g = total_idx[-2] if len(total_idx) > 1 else -1
    own_block = [i for i in data_idx if prev_g < i < grand]
    for g, k in cols:
        gt = cell(grand, g, k)
        if gt is None:
            continue
        leaf = colsum(data_idx, g, k)
        subs = colsum(total_idx[:-1], g, k) if (len(total_idx) > 1 and data_idx) else None
        blk = colsum(own_block, g, k) if own_block else None
        readings = [x for x in (leaf, subs, blk) if x is not None]
        if leaf is None and blk is None:
            continue                                 # never anchor on total-rows alone (degenerate
                                                     # sections where every label reads 'Total')
        cells = [(i, g, k) for i in data_idx + [grand]]
        if any(x == gt for x in readings):
            ctx.fired(VERTICAL, True, cells)
        else:
            s = leaf if leaf is not None else readings[0]
            reason = f"column sum {s} ≠ {lbl(grand).strip()[:16]}={gt}"
            ctx.fired(VERTICAL, False, cells,
                      [Violation(grand, g, k, VERTICAL, s, gt, abs(s - gt), reason)])
    prev = -1
    for ti in total_idx[:-1]:                       # earlier total = the sum of its block
        block = [i for i in data_idx if prev < i < ti]
        for g, k in cols:
            tv = cell(ti, g, k)
            s = colsum(block, g, k) if block else None
            if tv is None or s is None:
                continue
            cells = [(i, g, k) for i in block + [ti]]
            if s == tv:
                ctx.fired(VERTICAL, True, cells)
            else:
                reason = f"rows above sum to {s} ≠ {lbl(ti).strip()[:16]}={tv}"
                ctx.fired(VERTICAL, False, cells,
                          [Violation(ti, g, k, VERTICAL, s, tv, abs(s - tv), reason)])
        prev = ti


def _check_age_chain(rows, rh, ctx):
    """Vertical identity for age_rows layouts, via the age-hierarchy model: build ONE disjoint
    leaf chain of age bands from the row labels and require chain-sum == the Total/All-ages row.
    'Not stated/Unspecified' rows are extra addends. NEVER sums naively (subtotals excluded);
    refuses (fires nothing) when the band structure is ambiguous."""
    if not rows or rh is None:
        return
    spans, total_rows, ns_rows = [], [], []
    for ri, r in enumerate(rows):
        lbl = str(r.get(rh, "") or "")
        sp = parse_age_span(lbl)
        if sp == "TOTAL":
            total_rows.append(ri)
        elif isinstance(sp, tuple):
            spans.append((ri, sp))
        elif NOT_STATED.search(lbl):
            ns_rows.append(ri)
    if len(total_rows) != 1:                        # no anchor, or ambiguous multi-total section
        return
    chain = build_age_chain(spans, require_complete=True)
    if not chain:
        return                                      # partial page / ambiguous bands: nothing to anchor
    tot = total_rows[0]
    cols = [(g, k) for g, v in rows[0].items() if isinstance(v, dict) for k in v]
    def cell(i, g, k):
        v = rows[i].get(g)
        return _num(v.get(k)) if isinstance(v, dict) else None
    for g, k in cols:
        tv = cell(tot, g, k)
        if tv is None:
            continue
        vals = [cell(i, g, k) for i in chain + ns_rows]
        if any(v is None for v in vals):
            continue
        s = sum(vals)
        cells = [(i, g, k) for i in chain + ns_rows + [tot]]
        if s == tv:
            ctx.fired(AGE_CHAIN, True, cells)
        else:
            reason = f"age-band sum {s} ≠ Total={tv}"
            ctx.fired(AGE_CHAIN, False, cells,
                      [Violation(tot, g, k, AGE_CHAIN, s, tv, abs(s - tv), reason)])


def _check_cross_group(rows, ctx):
    """TRANSPOSED horizontal identity: the Total/All-ages column GROUP = Σ age-band column groups
    per row and subkey (age-hierarchy aware). This anchors the 1,104 transposed files that have
    no Total ROW at all. 'Unspecified/not stated' groups count as addends."""
    groups = _dict_groups(rows)
    total_g = [g for g in groups
               if parse_age_span(g) == "TOTAL" or re.search(r"total|population", g, re.I)]
    total_g = [g for g in total_g if not isinstance(parse_age_span(g), tuple)]
    if len(total_g) != 1:
        return
    tg = total_g[0]
    spans, ns_groups, unparsed = [], [], []
    for g in groups:
        if g == tg:
            continue
        sp = parse_age_span(g)
        if isinstance(sp, tuple):
            spans.append((g, sp))
        elif NOT_STATED.search(g):
            ns_groups.append(g)
        else:
            unparsed.append(g)
    if unparsed:                                    # a non-age sibling group -> identity unsafe
        return
    chain = build_age_chain(spans, require_complete=True)
    if not chain:
        return                                      # partial column coverage -> Total not anchorable
    for ri, r in enumerate(rows):
        tv_d = r.get(tg)
        if not isinstance(tv_d, dict):
            continue
        for sk in tv_d:
            tv = _num(tv_d.get(sk))
            if tv is None:
                continue
            vals = []
            for g in chain + ns_groups:
                v = r.get(g)
                vals.append(_num(v.get(sk)) if isinstance(v, dict) else None)
            if any(v is None for v in vals):
                continue
            s = sum(vals)
            cells = [(ri, g, sk) for g in chain + ns_groups + [tg]]
            if s == tv:
                ctx.fired(CROSS_GROUP, True, cells)
            else:
                reason = f"Σ age groups={s} ≠ {str(tg).strip()[:16]}={tv}"
                ctx.fired(CROSS_GROUP, False, cells,
                          [Violation(ri, g, sk, CROSS_GROUP, s, tv, abs(s - tv), reason)
                           for g in chain + ns_groups + [tg]])


def _check_sum_to_base(rows, base, ctx, tol_frac=0.003, abs_tol=None):
    """distribution/percentage identity: the parts within each column-group sum to the base.
    Accepts EITHER reading (all parts, or excluding a total-ish part) — subkey labels in these
    tables are unreliable. Tolerance = ±0.3% of base for distributions, ±0.5 for percentages."""
    bases = [base] if base else [1000, 10000]
    for ri, r in enumerate(rows):
        for g, v in r.items():
            if not isinstance(v, dict):
                continue
            allnums = [(k, _num(val)) for k, val in v.items()]
            allnums = [(k, x) for k, x in allnums if x is not None]
            if len(allnums) < 2:
                continue
            excl = [(k, x) for k, x in allnums
                    if not re.search(r"total|persons|both|grand", k, re.I)]
            candidates = [sum(x for _, x in allnums)]
            if len(excl) >= 2:
                candidates.append(sum(x for _, x in excl))
            cells = [(ri, g, k) for k, _ in allnums]
            ok = False
            for b in bases:
                tol = abs_tol if abs_tol is not None else max(1, round(tol_frac * b))
                if any(abs(s - b) <= tol for s in candidates):
                    ok = True
                    break
            if ok:
                ctx.fired(SUM_TO_BASE, True, cells)
            else:
                s = candidates[0]
                b = base or (1000 if s < 5000 else 10000)
                reason = f"Σparts={s} ≠ base={b}"
                ctx.fired(SUM_TO_BASE, False, cells,
                          [Violation(ri, g, k, SUM_TO_BASE, s, b, abs(s - b), reason)
                           for k, _ in allnums])


# ---------------------------------------------------------------------------
# section / table drivers
# ---------------------------------------------------------------------------
def _numeric_cells(rows):
    cells = set()
    for ri, r in enumerate(rows):
        for g, v in r.items():
            if isinstance(v, dict):
                for k, val in v.items():
                    if _num(val) is not None:
                        cells.add((ri, g, k))
    return cells


def check_section(rows, kind="count", base=None, layout=None, name=""):
    """Run every identity that applies to (kind, layout) over one section's rows.
    Single-sex sections simply never fire M+F=P (no persons+males+females triple) and
    civil-condition-slice sections never fire UMW (no U/M/W siblings) — both keep whatever
    identities their keys DO support, plus the vertical/cross-group anchors."""
    layout = layout or detect_layout(rows)
    ctx = _Ctx()
    numeric = _numeric_cells(rows)
    rep = SectionReport(name=name, layout=layout, kind=kind, n_rows=len(rows),
                        numeric_cells=len(numeric))
    if kind in NOT_CHECKABLE_KINDS:
        rep.status = "NOT_CHECKABLE"
        return rep
    rh = _row_header(rows)
    if kind == "count":
        _check_pmf(rows, ctx)
        _check_umw_group(rows, ctx)
        _check_umw_subkey(rows, ctx)
        if layout in ("transposed", "flat_area"):
            _check_vertical(rows, rh, ctx)
        elif layout == "age_rows":
            _check_age_chain(rows, rh, ctx)          # hierarchy-aware; NEVER a naive column sum
        if layout == "transposed":
            _check_cross_group(rows, ctx)
    elif kind == "distribution":
        _check_sum_to_base(rows, base, ctx, tol_frac=0.003)
    elif kind == "percentage":
        _check_sum_to_base(rows, 100, ctx, abs_tol=0.5)
    for ident in STRUCTURAL_IDENTITIES:             # self-validation gate (see definition above)
        st = ctx.stats.get(ident)
        if st and st["n_pass"] == 0:
            rep.suppressed[ident] = ctx.suppress(ident)
    rep.stats = ctx.stats
    rep.violations = ctx.violations
    rep.checked_cells = len(ctx.checked & numeric)
    rep.coverage = rep.checked_cells / len(numeric) if numeric else 0.0
    n_checked = sum(s["n_checked"] for s in ctx.stats.values())
    if n_checked == 0:
        rep.status = "NOT_ANCHORED"                  # identities apply but nothing could fire
    elif ctx.violations:
        rep.status = "FAIL"
    else:
        rep.status = "PASS"
    return rep


def count_violations(rows, layout=None):
    """Legacy-shaped {(row, group, subkey): reason} for the COUNT identities on one section."""
    return check_section(rows, "count", None, layout).viol_map()


def distribution_violations(rows, base=None):
    """Legacy-shaped {(row, group, subkey): reason} for the sum-to-base identity."""
    return check_section(rows, "distribution", base).viol_map()


def check_table(d):
    """Whole-table report. Honours the metadata short-circuits exactly as the old
    table_fail_rows did: human_verified -> trusted (PASS, 0 fails); constraints_irrelevant ->
    a row only fails when BOTH the count identities AND the sum-to-base identity reject it."""
    d = d or {}
    meta = d.get("metadata", {}) or {}
    kind, base = classify_kind(meta)
    rep = TableReport(kind=kind, base=base,
                      human_verified=bool(meta.get("human_verified")),
                      constraints_irrelevant=bool(meta.get("constraints_irrelevant")))
    if rep.human_verified:
        rep.status = "PASS"                          # a human confirmed the digits against the ink
        return rep

    dual = rep.constraints_irrelevant                # reviewer says identities don't apply as-is:
    fail_rows = set()
    for si, sec in enumerate(d.get("data", []) or []):
        rows = sec.get("rows", []) or []
        if not rows:
            continue
        name = sec.get("name", "")
        if dual:
            sc = check_section(rows, "count", None, name=name)
            sp = check_section(rows, "distribution", base, name=name)
            both = sc.fail_rows & sp.fail_rows       # either identity passing = validated
            sc.violations = [v for v in sc.violations if v.row in both]
            sc.status = ("FAIL" if sc.violations else
                         ("PASS" if (sc.stats or sp.stats) else "NOT_ANCHORED"))
            for ident, st in sp.stats.items():
                sc.stats.setdefault(ident, {"n_checked": 0, "n_pass": 0})
                sc.stats[ident]["n_checked"] += st["n_checked"]
                sc.stats[ident]["n_pass"] += st["n_pass"]
            sc.checked_cells = max(sc.checked_cells, sp.checked_cells)
            sc.coverage = sc.checked_cells / sc.numeric_cells if sc.numeric_cells else 0.0
            rep.sections.append(sc)
            fail_rows.update((si, ri) for ri in both)
        else:
            sr = check_section(rows, kind, base, name=name)
            rep.sections.append(sr)
            fail_rows.update((si, ri) for ri in sr.fail_rows)

    for sr in rep.sections:
        rep.numeric_cells += sr.numeric_cells
        rep.checked_cells += sr.checked_cells
        for ident, st in sr.stats.items():
            t = rep.stats.setdefault(ident, {"n_checked": 0, "n_pass": 0})
            t["n_checked"] += st["n_checked"]
            t["n_pass"] += st["n_pass"]
    rep.coverage = rep.checked_cells / rep.numeric_cells if rep.numeric_cells else 0.0
    rep.n_fail_rows = len(fail_rows)
    if kind in NOT_CHECKABLE_KINDS:
        rep.status = "NOT_CHECKABLE"
    elif sum(s["n_checked"] for s in rep.stats.values()) == 0:
        rep.status = "NOT_ANCHORED"
    elif fail_rows:
        rep.status = "FAIL"
    else:
        rep.status = "PASS"
    return rep
