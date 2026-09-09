#!/usr/bin/env python3
import html
import subprocess
"""Census review app — review BOTH the individual Table VII reads AND the multi-page assembly.
Run:  python3 census_review_app.py    -> http://localhost:5058
  /            landing (counts + links)
  /tables      imperfect Table VII reads: scan + matrix, mark accept / needs-reocr / note
  /assembly    the detected multi-page runs: member scans + section list in page order,
               confirm the pages are really ONE table in the right order (correct / wrong / note)
Verdicts saved to export/review_verdicts/{tables,assembly}.json (resumable).
"""
import json, csv, os, re, glob, html, collections, hashlib
from flask import Flask, request, jsonify, Response, send_file

import constraints_core   # single source of truth for identity checking (kind- & layout-aware)

ROOT = os.path.dirname(os.path.abspath(__file__))
V1 = os.path.join(ROOT, "export_final/sendout_final/sendout_v1")
CORPUS = os.path.join(ROOT, "export/ocr_results_referenced")
OUT = os.path.join(ROOT, "export/review_verdicts")
os.makedirs(OUT, exist_ok=True)
app = Flask(__name__)


def _verd(kind):
    p = os.path.join(OUT, f"{kind}.json")
    return json.load(open(p)) if os.path.exists(p) else {}


def _save_verd(kind, d):
    json.dump(d, open(os.path.join(OUT, f"{kind}.json"), "w"), indent=1)


def load_table(table_id):
    """table_id is a corpus-relative path .../digitized_tables/<dir>/<imghash>. Return (data, imghash, dir)."""
    tid = table_id.replace("export/ocr_results_referenced/", "").strip("/")
    # table_id already ends in '..._oneshot'  -> the file is {tid}.json
    jf = os.path.join(CORPUS, tid + ".json")
    if not os.path.exists(jf):
        hit = glob.glob(os.path.join(CORPUS, "digitized_tables", "*", os.path.basename(tid) + ".json"))
        jf = hit[0] if hit else None
    if not jf or not os.path.exists(jf):
        # catalog-style id ending in '__{40-hex imghash}_oneshot' -> glob the corpus file by imghash
        m = re.search(r"([0-9a-f]{40})(?:_oneshot)?$", tid)
        if m:
            hit = glob.glob(os.path.join(CORPUS, "digitized_tables", "*", m.group(1) + "_oneshot.json"))
            jf = hit[0] if hit else None
    if not jf or not os.path.exists(jf):
        return None, None, None
    try:
        d = json.load(open(jf))
    except Exception:
        return None, None, None
    ih = re.search(r"([0-9a-f]{40})_oneshot", os.path.basename(jf))
    return d, (ih.group(1) if ih else ""), os.path.dirname(jf)


def section_table(name, rows, limit=60):
    """Render one section the SAME way the static site does (table.mx): a sticky row-label column on
    the left, then grouped headers (POPULATION/UNMARRIED/...) over persons/males/females sub-columns.
    Table VII cells are nested dicts, expanded into the grouped sub-columns."""
    if not rows:
        return ""
    esc = lambda x: html.escape(str(x))
    scalar, dictc = [], []
    for r in rows:
        for k, v in r.items():
            if isinstance(v, dict):
                if k not in dictc:
                    dictc.append(k)
            elif k not in scalar:
                scalar.append(k)
    rh_col = scalar[0] if scalar else None
    groups = [(sc, [None]) for sc in scalar[1:]]      # extra scalar cols = single-col groups
    for dc in dictc:
        subs = []
        for r in rows:
            v = r.get(dc)
            if isinstance(v, dict):
                for sk in v:
                    if sk not in subs:
                        subs.append(sk)
        groups.append((dc, subs or [None]))
    has_groups = bool(dictc)

    out = [f"<h4>{esc(name) or '(section)'}</h4><table class=mx>"]
    if has_groups:
        out.append("<thead><tr><th rowspan=2 class=rh></th>")
        for g, subs in groups:
            out.append(f"<th class=grp colspan={len(subs)}>{esc(g)}</th>")
        out.append("</tr><tr>")
        for g, subs in groups:
            for sk in subs:
                out.append(f"<th>{esc(sk) if sk else esc(g)}</th>")
        out.append("</tr></thead>")
    else:
        out.append("<thead><tr><th class=rh></th>")
        for g, _ in groups:
            out.append(f"<th>{esc(g)}</th>")
        out.append("</tr></thead>")
    out.append("<tbody>")
    for r in rows[:limit]:
        out.append(f"<tr><td class=rh>{esc(r.get(rh_col, '')) if rh_col else ''}</td>")
        for g, subs in groups:
            v = r.get(g)
            for sk in subs:
                cell = v.get(sk, "") if isinstance(v, dict) else (v if sk is None else "")
                out.append(f"<td>{esc(cell)}</td>")
        out.append("</tr>")
    out.append("</tbody></table>")
    if len(rows) > limit:
        out.append(f"<div style='color:#789;font-size:11px'>… {len(rows)-limit} more rows</div>")
    return "".join(out)


def matrix_html(d):
    if not d:
        return "<i>no data</i>"
    out = [section_table(s.get("name", ""), s.get("rows", []), limit=40) for s in d.get("data", [])]
    return "".join(x for x in out if x) or "<i>no rows</i>"


# ---------------- editable matrix with constraint highlighting ----------------
def _num(x):
    s = str(x).replace(",", "").strip()
    if re.fullmatch(r"-?\d+", s):
        return int(s)
    try:
        return float(s)
    except Exception:
        return None


_AGEHDR = constraints_core.AGE_COL      # widened: em-dash, 'to', 'over N', bare '-N' bands


def is_transposed(rows):
    """True when AGE is the COLUMN dimension (groups like 'Age, 5-14 Unmarried') instead of the rows — a
    transposed Table VII / C-III. Delegates to constraints_core.detect_layout, which also applies the
    age-row guard (if the ROW labels are age bands, the rows are the age axis and this is NOT transposed)."""
    return constraints_core.detect_layout(rows) == "transposed"


def check_vertical(rows, rh):
    """VERTICAL (total-row anchored column-sum) identity — delegates to constraints_core. Returns the
    legacy {(row_idx, group, subkey): reason} shape. Only fires when every contributing cell is present,
    so partial reads don't false-flag."""
    ctx = constraints_core._Ctx()
    constraints_core._check_vertical(rows, rh, ctx)
    return {(v.row, v.group, v.subkey): v.reason for v in ctx.violations}


def check_constraints(rows):
    """Return {(row_idx, group, subkey): reason} for cells that violate the COUNT identities.
    Delegates to constraints_core, which is layout-aware:
      - M+F=Persons per group-cell on EVERY layout (transposed included — old code dropped it)
      - U+M+W(+not stated)=Total at group level AND at subkey level
      - anchored vertical sums for transposed/flat_area layouts
      - age-hierarchy vertical for age-in-rows layouts (never a naive column sum)
      - cross-group horizontal (All-ages group = Σ age-band groups) for transposed layouts"""
    return constraints_core.count_violations(rows)


def check_proportional(rows):
    """Identity for PROPORTIONAL / distribution tables ('per 1,000' or 'per 10,000 of each sex'): the
    parts within each column-group sum to the base. Delegates to constraints_core (±0.3% of base,
    either-reading acceptance). Returns the same {(row_idx, group, subkey): reason} shape."""
    return constraints_core.distribution_violations(rows)


def check_missing_columns(d):
    """Catch the None-disruption defect: a group-column (e.g. UNMARRIED persons/males) that is None on
    EVERY row while a sibling column (females) has real data — the column is missing/frozen, but M+F=P and
    U+M+W=Total silently SKIP it because None isn't checkable. Returns a set of (sec_idx,row_idx) to flag."""
    def _mk(v, pat):
        return next((kk for kk in v if re.search(pat, kk, re.I)), None)
    bad = set()
    for si, sec in enumerate(d.get("data", [])):
        rows = sec.get("rows", [])
        if len(rows) < 3:
            continue
        # per group, gather the M and F columns' null-counts across the section
        mnull, fnull, mtot, ftot = {}, {}, {}, {}
        for r in rows:
            for g, v in r.items():
                if not isinstance(v, dict):
                    continue
                mk = _mk(v, r"^male|persons.*male|\bmales?\b")
                fk = _mk(v, r"female")
                mk = next((k for k in v if re.search(r"male", k, re.I) and not re.search(r"female", k, re.I)), None)
                if mk is not None:
                    mtot[g] = mtot.get(g, 0) + 1
                    if v.get(mk) in (None, "", "None"):
                        mnull[g] = mnull.get(g, 0) + 1
                if fk is not None:
                    ftot[g] = ftot.get(g, 0) + 1
                    if v.get(fk) in (None, "", "None"):
                        fnull[g] = fnull.get(g, 0) + 1
        # DEFECT: a group where one of the M/F pair is systematically null while the OTHER has data —
        # i.e. half the male/female pair is frozen/missing (the real None-disruption). Excludes: proportional
        # tables, absent-persons columns (legit in transposed age tables), and groups that aren't a genuine
        # male+female PAIR (a "female"-only ratio/rank/page column is not a broken M/F pair).
        if (d.get("metadata", {}) or {}).get("data_type") == "proportional":
            continue
        defect_groups = set()
        for g in set(mtot) | set(ftot):
            if mtot.get(g, 0) < 3 or ftot.get(g, 0) < 3:   # must genuinely have BOTH a male AND female column
                continue
            m_all_null = mnull.get(g, 0) == mtot.get(g, 0)
            f_all_null = fnull.get(g, 0) == ftot.get(g, 0)
            m_has = mtot.get(g, 0) - mnull.get(g, 0) > 0
            f_has = ftot.get(g, 0) - fnull.get(g, 0) > 0
            if (m_all_null and f_has) or (f_all_null and m_has):
                defect_groups.add(g)
        if not defect_groups:
            continue
        for ri, r in enumerate(rows):
            for g in defect_groups:
                if isinstance(r.get(g), dict):
                    bad.add((si, ri)); break
    return bad


def table_fail_rows(d):
    """Kind- and layout-aware failing-row count for a whole table dict (honours metadata). Delegates to
    constraints_core.check_table: count tables get the count identities, distribution/percentage tables
    get sum-to-base, rate/ratio/index tables get NO identity (nothing to fail); human_verified and
    constraints_irrelevant short-circuits preserved. Returns int (distinct failing rows)."""
    rep = constraints_core.check_table(d)
    if rep.human_verified:
        return 0                       # a human/agent confirmed the digits against the ink — trusted
    n = rep.n_fail_rows
    # add rows where a group column is systematically None while a sibling has data (None-disruption)
    mc = check_missing_columns(d)
    n += len({ri for (si, ri) in mc})
    return n


def constraint_report(rows):
    """Return (viol_cells, failures): viol_cells maps (row_idx,group,subkey)->reason for highlighting;
    failures is an explicit human list: [(row_label, plain-English description)] of every failing
    constraint, so the reviewer knows exactly which cells to check."""
    viol = check_constraints(rows)
    failures = []
    rh = None
    for r in rows:
        for k, v in r.items():
            if not isinstance(v, dict):
                rh = k; break
        if rh:
            break
    seen = set()
    for (ri, g, sk), reason in viol.items():
        key = (ri, reason)
        if key in seen:
            continue
        seen.add(key)
        label = (str(rows[ri].get(rh, "")) if rh else f"row {ri}") or f"row {ri}"
        failures.append((label, reason))
    return viol, failures


def suggest_row(row):
    """Where >=2 constraints (M+F=P and U+M+W=Total) independently agree on the SAME value that differs
    from the read, that cell is the misread one — return [(group, subkey, current, suggested)]. This is
    the 'the OTHER cell is wrong' hint, e.g. M+F=50409 AND U+M+W=50409 but Total reads 50400 -> 50409."""
    groups = {g: v for g, v in row.items() if isinstance(v, dict)}
    unm = next((g for g in groups if re.search(r"unmarried|single|never", g, re.I)), None)
    mar = next((g for g in groups if re.search(r"married", g, re.I) and not re.search(r"unmarried", g, re.I)), None)
    wid = next((g for g in groups if re.search(r"widow|divorc", g, re.I)), None)
    tot = next((g for g in groups if re.search(r"total|population|all ages|grand", g, re.I)), None)
    out = []
    for g, v in groups.items():
        pk = next((k for k in v if re.search(r"person|total|both", k, re.I)), None)
        mk = next((k for k in v if re.search(r"male", k, re.I) and not re.search(r"female", k, re.I)), None)
        fk = next((k for k in v if re.search(r"female", k, re.I)), None)
        for sk in v:
            cands = []
            if pk and mk and fk:
                p, m, f = _num(v.get(pk)), _num(v.get(mk)), _num(v.get(fk))
                if sk == pk and None not in (m, f):
                    cands.append(m + f)
                elif sk == mk and None not in (p, f):
                    cands.append(p - f)
                elif sk == fk and None not in (p, m):
                    cands.append(p - m)
            if unm and mar and wid and tot:
                if g == tot:
                    vals = [_num(groups[x].get(sk)) for x in (unm, mar, wid)]
                    if None not in vals:
                        cands.append(sum(vals))
                elif g in (unm, mar, wid):
                    tv = _num(groups[tot].get(sk)); ov = [_num(groups[x].get(sk)) for x in (unm, mar, wid) if x != g]
                    if tv is not None and None not in ov:
                        cands.append(tv - sum(ov))
            cur = _num(v.get(sk))
            if len(cands) >= 2 and len(set(cands)) == 1 and cands[0] != cur:
                out.append((g, sk, cur, cands[0]))
    return out


def blame_cell(row):
    """Reviewer's heuristic, encoded: a single misread cell sits at the intersection of TWO constraints —
    its group's M+F=Persons AND its column's Unmarried+Married+Widowed=Population. When BOTH of those fail
    for a cell, it's almost always the wrong one, and the two failing identities pin its correct value
    (they agree). Returns confident repairs [(group, subkey, current, suggested, n_failing)] ranked by how
    many failing constraints each cell is in (2 = both modes — the strongest signal)."""
    groups = {g: v for g, v in row.items() if isinstance(v, dict)}
    unm = next((g for g in groups if re.search(r"unmarried|single|never", g, re.I)), None)
    mar = next((g for g in groups if re.search(r"married", g, re.I) and not re.search(r"unmarried", g, re.I)), None)
    wid = next((g for g in groups if re.search(r"widow|divorc", g, re.I)), None)
    tot = next((g for g in groups if re.search(r"total|population|all ages|grand", g, re.I)), None)

    def roles(v):
        return (next((k for k in v if re.search(r"person|total|both", k, re.I)), None),
                next((k for k in v if re.search(r"male", k, re.I) and not re.search(r"female", k, re.I)), None),
                next((k for k in v if re.search(r"female", k, re.I)), None))

    h1_fail = {}                                   # group -> M+F != Persons ?
    for g, v in groups.items():
        pk, mk, fk = roles(v)
        if pk and mk and fk:
            p, m, f = _num(v.get(pk)), _num(v.get(mk)), _num(v.get(fk))
            if None not in (p, m, f):
                h1_fail[g] = (m + f != p)
    v_fail = {}                                    # subkey -> U+M+W != Population ?
    if unm and mar and wid and tot:
        for sk in groups[tot]:
            tv = _num(groups[tot].get(sk)); ov = [_num(groups[x].get(sk)) for x in (unm, mar, wid)]
            if tv is not None and None not in ov:
                v_fail[sk] = (sum(ov) != tv)

    suspects = []
    for g, v in groups.items():
        pk, mk, fk = roles(v)
        for sk in v:
            nfail = (1 if h1_fail.get(g) else 0) + (1 if v_fail.get(sk) else 0)
            if nfail == 0:
                continue
            cur = _num(v.get(sk)); cands = []
            if pk and mk and fk:
                p, m, f = _num(v.get(pk)), _num(v.get(mk)), _num(v.get(fk))
                if None not in (p, m, f):
                    if sk == pk: cands.append(m + f)
                    elif sk == mk: cands.append(p - f)
                    elif sk == fk: cands.append(p - m)
            if unm and mar and wid and tot:
                if g == tot:
                    vals = [_num(groups[x].get(sk)) for x in (unm, mar, wid)]
                    if None not in vals: cands.append(sum(vals))
                elif g in (unm, mar, wid):
                    tv = _num(groups[tot].get(sk)); ov = [_num(groups[x].get(sk)) for x in (unm, mar, wid) if x != g]
                    if tv is not None and None not in ov: cands.append(tv - sum(ov))
            agree = len(cands) >= 2 and len(set(cands)) == 1 and cands[0] != cur
            if agree:
                suspects.append((nfail, g, sk, cur, cands[0]))
    # the cell in the MOST failing constraints, with both identities agreeing on its value, is the fix
    suspects.sort(key=lambda s: -s[0])
    return [(g, sk, cur, sug, nf) for nf, g, sk, cur, sug in suspects]


def section_table_edit(name, rows, sec_idx, viol, limit=80, tid=""):
    """Like section_table, but every cell is an editable <input> tagged with its location (and its
    member table_id, for the assembled multi-page view), and constraint-violating cells are red."""
    if not rows:
        return ""
    esc = lambda x: html.escape(str(x))
    scalar, dictc = [], []
    for r in rows:
        for k, v in r.items():
            (dictc if isinstance(v, dict) else scalar).append(k) if k not in (dictc if isinstance(v, dict) else scalar) else None
    rh_col = scalar[0] if scalar else None
    groups = [(sc, [None]) for sc in scalar[1:]]
    for dc in dictc:
        subs = []
        for r in rows:
            v = r.get(dc)
            if isinstance(v, dict):
                for sk in v:
                    if sk not in subs:
                        subs.append(sk)
        groups.append((dc, subs or [None]))
    has = bool(dictc)

    def cell(ri, g, sk, val):
        key = (ri, g, sk if sk is not None else "")
        bad = viol.get(key)
        pe = bad and str(bad).startswith("\U0001f58b")   # 🖋 = confirmed print/source error (amber, not red)
        hl = "#ffe6b3" if pe else ("#ffd4d4" if bad else "")
        style = f"background:{hl};" if bad else ""
        ttl = f" title=\"{esc(bad)}\"" if bad else ""
        return (f"<td style='{style}padding:1px'{ttl}>"
                f"<input value=\"{esc(val)}\" data-tid=\"{esc(tid)}\" data-s={sec_idx} data-r={ri} "
                f"data-g=\"{esc(g)}\" data-k=\"{esc(sk if sk is not None else '')}\" "
                f"oninput=\"dirty=true\" style='width:64px;border:1px solid #e2e6ee;"
                f"{('background:'+hl+';') if bad else ''}text-align:right;font:12px Menlo,monospace;padding:1px 3px'></td>")

    out = [f"<h4>{esc(name) or '(section)'}</h4><table class=mx>"]
    if has:
        out.append("<thead><tr><th rowspan=2 class=rh></th>")
        for g, subs in groups:
            out.append(f"<th class=grp colspan={len(subs)}>{esc(g)}</th>")
        out.append("</tr><tr>")
        for g, subs in groups:
            for sk in subs:
                out.append(f"<th>{esc(sk) if sk else esc(g)}</th>")
        out.append("</tr></thead>")
    else:
        out.append("<thead><tr><th class=rh></th>" + "".join(f"<th>{esc(g)}</th>" for g, _ in groups) + "</tr></thead>")
    out.append("<tbody>")
    for ri, r in enumerate(rows[:limit]):
        rl = r.get(rh_col, "") if rh_col else ""
        out.append(f"<tr><td class=rh style='padding:1px'><input value=\"{esc(rl)}\" data-tid=\"{esc(tid)}\" data-s={sec_idx} data-r={ri} "
                   f"data-g=\"{esc(rh_col or '')}\" data-k=\"\" oninput=\"dirty=true\" "
                   f"style='width:150px;border:1px solid #e2e6ee;font:12px system-ui;padding:1px 3px'></td>")
        for g, subs in groups:
            v = r.get(g)
            for sk in subs:
                cv = v.get(sk, "") if isinstance(v, dict) else (v if sk is None else "")
                out.append(cell(ri, g, sk, cv))
        out.append("</tr>")
    out.append("</tbody></table>")
    return "".join(out)



def _mark_print_errors(d, viol):
    """If this table is a confirmed / ink-verified print (source) error, prefix its failing-cell reasons
    with 🖋 so they render AMBER ('confirmed print error — do not fix') instead of red ('misread to fix')."""
    md = d.get("metadata", {}) or {}
    notes = " ".join(str(v) for k, v in md.items() if "note" in k.lower()).lower()
    is_pe = (md.get("source_error") or md.get("native_verified") or md.get("human_verified")
             or any(w in notes for w in ("print error", "printed error", "compositor", "irreducible",
                                         "ink-verified", "source anomaly", "tabulation error")))
    if not is_pe:
        return viol
    return {k: ("\U0001f58b confirmed print/source error (ink-verified) — " + str(v)) for k, v in viol.items()}

def matrix_edit_html(d):
    if not d:
        return "<i>no data</i>"
    out, nviol = [], 0
    for si, s in enumerate(d.get("data", [])):
        rows = s.get("rows", [])
        viol = _mark_print_errors(d, check_constraints(rows))
        nviol += len(set((k[0],) for k in viol))
        out.append(section_table_edit(s.get("name", ""), rows, si, viol))
    body = "".join(x for x in out if x) or "<i>no rows</i>"
    banner = (f"<div style='margin:4px 0;color:#9c0006;font-weight:600'>⚠ {nviol} row(s) fail a constraint "
              f"(highlighted red) — click a cell to fix, then Save</div>" if nviol else
              "<div style='margin:4px 0;color:#0a5d1e;font-weight:600'>✓ all constraints satisfied</div>")
    return banner + body


def run_of(tid):
    """The assembly run whose members include this table page (by imghash), else None."""
    m = re.search(r"([0-9a-f]{40})", tid or "")
    ih = m.group(1) if m else None
    if not ih:
        return None
    for run in assembly_runs():
        if any(ih in mm.get("table_id", "") for mm in run.get("members", [])):
            return run
    return None


def neighborhood_of(tid, reach=4):
    """Ad-hoc 'run' for a standalone table the assembly didn't group: the physically-adjacent pages in
    the same volume, so the user still sees the full multi-page table to check constraints across it."""
    g = re.search(r"([0-9a-f]{16})_t", tid or "")
    gi = re.search(r"([0-9a-f]{40})", tid or "")
    dh = g.group(1) if g else None
    ih = gi.group(1) if gi else None
    if not dh or not ih:
        return None
    vp, _ = vol_index()
    pages = vp.get(dh, {})
    mypg = next((pg for pg, rec in pages.items() if rec and rec[0] == ih), None)
    if mypg is None:
        return None
    members = []
    for pg in range(mypg - reach, mypg + reach + 1):
        rec = pages.get(pg)
        if rec and rec[0]:
            members.append({"table_id": f"digitized_tables/{rec[1]}/{rec[0]}_oneshot", "page": pg})
    return {"members": members, "page_range": [members[0]["page"], members[-1]["page"]]} if len(members) > 1 else None


def matrix_assembled_edit_html(run):
    """Show the WHOLE multi-page table: every member page's sections combined and editable (each cell
    saves back to its own page), constraint-checked across the complete rows. A single page is only a
    fragment, so this is what makes M+F=P / U+M+W=Total meaningful."""
    members = sorted(run.get("members", []), key=lambda m: m.get("page", 0))
    blocks, total_viol = [], 0
    for m in members:
        mt = m.get("table_id", "")
        d, mih, _ = load_table(mt)
        if not d:
            continue
        for si, s in enumerate(d.get("data", [])):
            rows = s.get("rows", [])
            viol = check_constraints(rows)
            total_viol += len(set((k[0],) for k in viol))
            tbl = section_table_edit(f"p{m.get('page', '')} · {s.get('name', '')}", rows, si, viol, tid=mt)
            if tbl:
                blocks.append(tbl)
    body = "".join(blocks) or "<i>no rows</i>"
    banner = (f"<div style='margin:4px 0;color:#9c0006;font-weight:600'>⚠ {total_viol} row(s) fail a constraint "
              f"across this {len(members)}-page table — fix cells then Save</div>" if total_viol else
              f"<div style='margin:4px 0;color:#0a5d1e;font-weight:600'>✓ all constraints satisfied across the "
              f"{len(members)}-page assembled table</div>")
    return banner + body


def assembled_panels(run, cur_ih):
    """Per-page panels for the paginated assembled view. Each member page becomes one scan panel (left)
    and one matrix panel (right) shown one at a time. Constraints are checked on every page and also
    aggregated into a single cross-page failure list with page jumps, so the reviewer sees EXACTLY which
    constraint fails on which row/page."""
    members = sorted(run.get("members", []), key=lambda m: m.get("page", 0))
    scan_panels, mat_panels, gfails = [], [], []
    cur_idx = 0
    for idx, m in enumerate(members):
        mt = m.get("table_id", "")
        d, mih, _ = load_table(mt)
        page = m.get("page", "")
        if mih and mih == cur_ih:
            cur_idx = idx
        dh_run = run.get("dochash", "")
        fb = (f" onerror=\"this.onerror=null;this.src='/render/{dh_run}/{page}'\""
              if str(page).isdigit() and dh_run else "")
        scan_panels.append(
            f"<div class=pscan data-p={idx} style='display:none'>"
            f"<img class=zimg src='/scan/{mih}'{fb} onclick='zoomtog(this)' title='click = zoom in / out'></div>")
        mblocks, pfails = [], []
        if d:
            for si, s in enumerate(d.get("data", [])):
                rows = s.get("rows", [])
                viol, fails = constraint_report(rows)
                viol = _mark_print_errors(d, viol)   # amber for confirmed print/source errors
                tbl = section_table_edit(s.get("name", ""), rows, si, viol, tid=mt)
                if tbl:
                    mblocks.append(tbl)
                rh = next((k for k in rows[0] if not isinstance(rows[0].get(k), dict)), None) if rows else None
                sugg = {}
                for ri, row in enumerate(rows):
                    lbl = (str(row.get(rh, "")) if rh else f"row {ri}") or f"row {ri}"
                    bc = blame_cell(row)            # cells in the MOST failing constraints, value pinned by both
                    if bc:
                        sugg[lbl] = "; ".join(
                            f"{html.escape(g)}.{html.escape(sk)} = <b>{val}</b>"
                            + (" ⊕<i>in both M+F=P &amp; column checks — most likely the misread</i>" if nf >= 2 else "")
                            for g, sk, c, val, nf in bc[:3])
                    else:
                        s2 = suggest_row(row)
                        if s2:
                            sugg[lbl] = ", ".join(f"{html.escape(g)}.{html.escape(sk)} = <b>{val}</b>" for g, sk, c, val in s2)
                byrow = collections.OrderedDict()
                for label, reason in fails:
                    byrow.setdefault(label, []).append(reason)
                for label, reasons in byrow.items():
                    sg = sugg.get(label)
                    pfails.append((label, "; ".join(reasons), sg))
                    gfails.append((page, idx, label, "; ".join(reasons), sg))
        items = ""
        for label, reason, sg in pfails:
            hint = f" &rarr; <span class=sugg>likely set {sg}</span>" if sg else ""
            items += f"<li>row <b>{html.escape(str(label))}</b> — {html.escape(reason)}{hint}</li>"
        pf = (f"<div class=pfail><b>⚠ this page:</b><ul>{items}</ul></div>" if pfails
              else "<div class=pok>✓ this page satisfies M+F=Persons and U+M+W=Total</div>")
        flagged = bool(((d or {}).get("metadata", {}) or {}).get("needs_reocr"))
        btn = (f"<button class=pgreocr id=ro_{idx} onclick=\"reocrPage('{mih}',{idx})\">"
               + ("✓ flagged for re-OCR" if flagged else "✗ re-OCR just this page") + "</button>")
        is_title = bool(((d or {}).get("metadata", {}) or {}).get("title_page"))
        empty_note = ("<div style='color:#6b4ea0;font-weight:700;padding:10px;background:#f3eef9;border-radius:6px'>"
                      "📑 Title / caption page — kept in the table's page range for provenance, no data extracted "
                      "(scan shown on the left).</div>" if is_title else "<i>no rows</i>")
        pf_or = "" if is_title else pf
        mat_panels.append(
            f"<div class=pmat data-p={idx} style='display:none'>"
            f"<div class=ptitle>Page {page} &middot; member {idx + 1} of {len(members)} {btn}</div>{pf_or}"
            f"{''.join(mblocks) or empty_note}</div>")
    return scan_panels, mat_panels, gfails, len(members), cur_idx


# ---- data ----
def imperfect_tables():
    p = os.path.join(V1, "tableVII_imperfect.csv")
    rows = list(csv.DictReader(open(p))) if os.path.exists(p) else []
    # exclude assembled-table IDs (asm_*): they have NO single-page read, so this per-page view renders them
    # blank (no data + no image). Assembled multi-page tables are reviewed in /assembly, not here.
    rows = [r for r in rows if not str(r.get("table_id", "")).startswith("asm_")]
    # NOTE: we deliberately do NOT exclude the "unrecoverable" verdict — proven (2026-06-29) to be a largely
    # FALSE label (left/right half-page spreads, Race/Caste & distribution tables, Gemini column-drops are all
    # recoverable). Show them; the worst-first severity sort surfaces the ones most worth your time.
    return rows


def assembly_runs():
    p = os.path.join(ROOT, "export/tableVII_runs.json")
    return json.load(open(p)) if os.path.exists(p) else []


def _run_quality():
    """(saoa, 's-e') -> honest quality label from the first-cut catalog (verified/minor-issues/needs-review)."""
    q = {}
    p = os.path.join(ROOT, "export/first_cut/catalog.csv")
    if os.path.exists(p):
        for r in csv.DictReader(open(p)):
            q[(r["doc"], r["page_range"])] = r["quality"]
    return q


_DIFF_RE = re.compile(r"(\d+)\s*≠.*?=\s*(-?\d+)")    # "M+F=12 ≠ P=15" / "column sum 1362 ≠ ...=1367"


def _fail_mag(msg):
    m = _DIFF_RE.search(str(msg))
    try:
        return abs(int(m.group(1)) - int(m.group(2))) if m else 0
    except Exception:
        return 0


def table_severity(d, count_weight=200, cap=9999):
    """(n_fail_rows, total_magnitude, max_magnitude, severity). severity = capped_total_mag + W*n_fail_rows
    — a linear combination so BIG single failures AND MANY failures both float to the top. Per-cell
    magnitude is capped (a garbage cell read as 4e14 shouldn't infinitely drown out genuine messy tables).
    KIND-GATED via constraints_core.check_table: proportional/rate/ratio tables no longer accumulate
    fake count-identity failures, and human_verified tables score 0."""
    rep = constraints_core.check_table(d or {})
    mags, rows_failed = [], set()
    for si, sr in enumerate(rep.sections):
        for v in sr.violations:
            mags.append(min(int(v.magnitude or 0), cap))
            rows_failed.add((sr.name or si, v.row))
    n = len(rows_failed); total = sum(mags); mx = max(mags) if mags else 0
    return n, total, mx, total + count_weight * n


def _load_sev(which):
    """cached severity map written by compute_severity.py (export/severity_<which>.json)."""
    p = os.path.join(ROOT, f"export/severity_{which}.json")
    try:
        return json.load(open(p)) if os.path.exists(p) else {}
    except Exception:
        return {}


CSS = """body{font:14px/1.5 system-ui;margin:0;background:#f6f8fb;color:#1a2233}
header{background:#1f3864;color:#fff;padding:10px 18px}header a{color:#ffd966;margin-right:16px;text-decoration:none}
.wrap{display:flex;gap:14px;padding:14px}.scan{flex:0 0 46%}.scan img{width:100%;border:1px solid #ccd;border-radius:6px}
.data{flex:1;overflow:auto;max-height:88vh}table{border-collapse:collapse;font-size:12px;margin:4px 0}
th,td{border:1px solid #dde;padding:2px 6px}th{background:#eef3fb}.sec{font-weight:700;color:#1f3864;margin-top:8px}
h4{margin:8px 0 2px;color:#1f3864;font-size:13px}
table.mx{border-collapse:collapse;margin-bottom:10px;font:12px/1.3 Menlo,Consolas,monospace}
table.mx th,table.mx td{border:1px solid #d6dde2;padding:2px 7px;text-align:right;white-space:nowrap}
table.mx th{background:#eef3f7;text-align:center}
table.mx td.rh,table.mx th.rh{text-align:left;background:#fafbfc;font-weight:600;white-space:normal;max-width:230px;position:sticky;left:0}
table.mx thead .grp{background:#dde7ee}
.btns{position:sticky;bottom:0;z-index:20;background:#fff;border-top:2px solid #1f3864;margin:0;padding:9px 18px;box-shadow:0 -2px 8px rgba(0,0,0,.08)}button{padding:7px 14px;border:0;border-radius:6px;margin-right:8px;font-weight:600;cursor:pointer}
.ok{background:#c6efce;color:#0a5d1e}.bad{background:#ffc7ce;color:#9c0006}.skip{background:#dde}
.meta{margin:6px 18px;color:#456}.prog{float:right}.mem{display:inline-block;margin:2px;padding:1px 6px;background:#eef;border-radius:4px;font-size:11px}
.run img{width:200px;height:150px;object-fit:cover;object-position:top;border:1px solid #ccd;border-radius:5px}
.runpages{display:flex;gap:10px;overflow-x:auto;padding:6px 0}.pg{flex:0 0 auto;text-align:center;font-size:11px}"""


@app.route("/")
def home():
    return Response(f"""<!doctype html><meta charset=utf-8><style>{CSS}</style>
<header><b>Census review</b> <a href=/tables>Tables ({len(imperfect_tables())})</a>
<a href=/assembly>Assembly ({len(assembly_runs())})</a></header>
<div class=meta>Review both to get Table VII over the line (a run/table drops off the moment it's reviewed):<br>
&bull; <b>Tables</b> — the {len(imperfect_tables())} imperfect individual reads: is the matrix right? accept / needs-reocr.<br>
&bull; <b>Assembly</b> — the {len(assembly_runs())} multi-page runs: are these pages really ONE table, in order? confirm / wrong.</div>""",
                    mimetype="text/html")


def find_scan(imghash, source_image="", table_dir=""):
    """resolve a scan (png OR jpg) by imghash — TARGETED lookups (no slow recursive glob)."""
    cands = [os.path.join(ROOT, "census_site/img", imghash + ".jpg"),
             os.path.join(ROOT, "census_site/img", imghash + ".png")]
    if table_dir:
        cands += [os.path.join(CORPUS, "digitized_tables", table_dir, imghash + ".png"),
                  os.path.join(CORPUS, "digitized_tables", table_dir, imghash + ".jpg")]
    if source_image:
        cands.append(source_image)
    for c in cands:
        if c and os.path.exists(c):
            return c
    for ext in (".png", ".jpg"):
        hit = glob.glob(os.path.join(CORPUS, "digitized_tables", "*", imghash + ext))
        if hit:
            return hit[0]
    return None


THUMBDIR = os.path.join(ROOT, "export/review_thumbs")


def thumb_of(src, w=360):
    """downscaled, cached thumbnail of a scan (fast page-strip loading; full-res still on click)."""
    os.makedirs(THUMBDIR, exist_ok=True)
    key = hashlib.md5(f"{src}:{w}".encode()).hexdigest() + ".jpg"
    cache = os.path.join(THUMBDIR, key)
    if not os.path.exists(cache):
        try:
            from PIL import Image
            im = Image.open(src)
            im.thumbnail((w, w * 3))  # tall pages ok
            im.convert("RGB").save(cache, "JPEG", quality=78)
        except Exception:
            return src  # fall back to full-res if PIL/resize fails
    return cache


_IH_PAGE = None


def _ih_page_index():
    """imghash -> (dochash, pdf_page) from the page master, so ANY page can be rendered from the drive
    even when no local scan jpg is cached. Built once, cached."""
    global _IH_PAGE
    if _IH_PAGE is not None:
        return _IH_PAGE
    _IH_PAGE = {}
    try:
        for r in csv.DictReader(open(os.path.join(V1, "page_number_status.csv"))):
            ih0 = r.get("imghash", "")
            pg0 = r.get("pdf_page", "")
            dh0 = r.get("dochash", "")
            if ih0 and dh0 and str(pg0).isdigit():
                _IH_PAGE.setdefault(ih0, (dh0, int(pg0)))
    except Exception:
        pass
    return _IH_PAGE


@app.route("/scan/<ih>")
def scan(ih):
    si = request.args.get("si", "")
    f = find_scan(ih, si)
    if not f:
        # last resort: load the table by imghash -> its source_image, else live-render from the drive
        jf = glob.glob(os.path.join(CORPUS, "digitized_tables", "*", ih + "_oneshot.json"))
        if jf:
            try:
                dd = json.load(open(jf[0]))
            except Exception:
                dd = {}
            sij = (dd.get("source_image") or "")
            if sij and os.path.exists(sij):
                f = sij
            else:
                dh = re.search(r"([0-9a-f]{16})_t", os.path.basename(os.path.dirname(jf[0])))
                pg = re.search(r"_p(\d+)", sij) or re.search(r"_p(\d+)", os.path.basename(jf[0]))
                if dh and pg:
                    return render_page(dh.group(1), int(pg.group(1)))   # render the page from the drive
    if not f:
        # robust fallback: look the imghash up in the page master and render from the drive
        dp = _ih_page_index().get(ih)
        if dp:
            resp = render_page(dp[0], dp[1])
            if not (isinstance(resp, tuple) and resp[1] == 404):
                return resp
    if not f:
        return ("not found", 404)
    if request.args.get("thumb"):
        return send_file(thumb_of(f, int(request.args.get("thumb") or 360)))
    return send_file(f)


# ---- volume page index (for showing the page BEFORE / AFTER a run) ----
_VOLPAGE = None
_VOLCP = None


def vol_index():
    """dochash -> {pdf_page: (imghash, table_dir, title)} from the master page document."""
    global _VOLPAGE, _VOLCP
    if _VOLPAGE is not None:
        return _VOLPAGE, _VOLCP
    _VOLPAGE = collections.defaultdict(dict)
    for r in csv.DictReader(open(os.path.join(V1, "page_number_status.csv"))):
        if not str(r.get("pdf_page", "")).isdigit():
            continue
        dm = re.match(r"([0-9a-f]{16})_t", r["table_dir"])
        if not dm:
            continue
        pg = int(r["pdf_page"]); dh = dm.group(1)
        prev = _VOLPAGE[dh].get(pg)
        if prev is None or len(r.get("title", "")) > len(prev[2]):
            _VOLPAGE[dh][pg] = (r["imghash"], r["table_dir"], r.get("title", ""))
    # dochash -> (cluster_path, year) for on-demand drive render of gap pages.
    # WIDE map (cc_volumes + documents + provenance) so EVERY volume can render its before/after pages.
    _VOLCP = {}
    for fn, col in (("cc_volumes.csv", "cluster_path"), ("documents.csv", "source_pdf"),
                    ("provenance.csv", "source_pdf")):
        p = os.path.join(V1, fn)
        if not os.path.exists(p):
            continue
        for r in csv.DictReader(open(p)):
            cp = r.get(col)
            if cp and "census_pdfs" in cp:
                dh = hashlib.sha1(cp.encode()).hexdigest()[:16]
                if dh not in _VOLCP:
                    ym = re.search(r"/(\d{4})/", cp)
                    _VOLCP[dh] = (cp, ym.group(1) if ym else "")
    return _VOLPAGE, _VOLCP


def resolve_page(dh, pg, ih="", table_dir=""):
    """Single source of truth for showing ANY page: local scan if the PNG exists, else render LIVE
    from the original drive PDF (dh+pg). Returns (thumb_src, full_src) or (None, None) if unreachable."""
    if ih and find_scan(ih, table_dir=table_dir):
        return f"/scan/{ih}?thumb=480", f"/scan/{ih}"
    if dh and dh in vol_index()[1]:
        return f"/render/{dh}/{pg}", f"/render/{dh}/{pg}"
    return None, None


def page_img(dh, pg, ih, table_dir, lazy=True):
    """img tag (or clear warning) for a run member page — always traceable to local scan OR drive PDF."""
    thumb, full = resolve_page(dh, pg, ih, table_dir)
    ld = "lazy" if lazy else "eager fetchpriority=high"
    if thumb:
        return (f"<img class=run loading={ld} src='{thumb}' onclick=\"zoom('{full}')\" title='click to zoom' "
                f"onerror=\"this.nextElementSibling.style.display='flex';this.style.display='none'\">"
                f"<div class=cxerr style='display:none;width:340px;height:200px;background:#fde;color:#900;"
                f"align-items:center;justify-content:center;text-align:center;padding:8px'>⚠ p{pg} couldn't load<br>(drive disconnected? re-seat &amp; reload)</div>")
    return (f"<div style='width:340px;height:200px;background:#fde;color:#900;display:flex;"
            f"align-items:center;justify-content:center;text-align:center;padding:8px'>"
            f"⚠ p{pg} not traceable<br>(no local scan &amp; volume not on drive)</div>")


def context_card(dh, pg, label):
    """Greyed context card for the page before/after a run. ALWAYS shows the page if at all possible:
    local scan if present, else render it live from the drive PDF. Clear demarcation only if truly
    unreachable (page out of range, or drive not mounted)."""
    vp, vcp = vol_index()
    if pg < 1:
        return ""  # no page before page 1
    rec = vp.get(dh, {}).get(pg)
    title = (rec[2] if rec else "")
    src = full = note = None
    # 1) local scan, only if the PNG actually exists on disk
    if rec and find_scan(rec[0], table_dir=rec[1]):
        ih = rec[0]; src = f"/scan/{ih}?thumb=480"; full = f"/scan/{ih}"; note = html.escape(title[:46])
    # 2) else render live from the drive PDF (the page IS on the drive)
    elif dh in vcp:
        src = full = f"/render/{dh}/{pg}"
        note = (html.escape(title[:40]) + " · <i>from drive</i>") if title and title != "Blank Page" else "<i>rendered from drive</i>"
    if src:
        body = (f"<img class=run loading=eager fetchpriority=high src='{src}' style='opacity:.92' "
                f"onclick=\"zoom('{full}')\" onerror=\"this.parentNode.querySelector('.cxerr').style.display='block';this.style.display='none'\" title='click to zoom'>"
                f"<div class=cxerr style='display:none;width:340px;height:200px;background:#fde;color:#900;"
                f"display:none;align-items:center;justify-content:center;text-align:center;padding:8px'>"
                f"⚠ p{pg} couldn't load.<br>Drive disconnected? Re-seat &amp; reload.</div>"
                f"<br><b>p{pg}</b> · {note}")
    else:
        body = (f"<div style='width:340px;height:200px;background:#fde;color:#900;display:flex;"
                f"align-items:center;justify-content:center;text-align:center;padding:8px'>"
                f"⚠ p{pg} not accessible<br>(volume not found / drive not mounted)</div>")
    return (f"<div class=pg style='border:2px dashed #c8a;background:#faf5fb;padding:4px;border-radius:6px'>"
            f"<div style='color:#a06;font-weight:700;font-size:11px'>{label}</div>{body}</div>")


DRIVE = "/Volumes/Expansion/census_pdfs"


@app.route("/render/<dh>/<int:pg>")
def render_page(dh, pg):
    """render a single PDF page on-demand from the drive (for gap context pages)."""
    _, vcp = vol_index()
    if dh not in vcp:
        return ("no volume", 404)
    cp, yr = vcp[dh]
    base = os.path.basename(cp)
    path = os.path.join(DRIVE, yr, base)
    if not os.path.exists(path):                      # year guess wrong / not seated -> search the drive
        hits = glob.glob(os.path.join(DRIVE, "*", base))
        if not hits:
            return ("PDF not on drive (check it's mounted)", 404)
        path = hits[0]
    cache = os.path.join(ROOT, "export/context_renders", f"{dh}_p{pg}.png")
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    if not os.path.exists(cache):
        try:
            import fitz
            doc = fitz.open(path)
            if 1 <= pg <= doc.page_count:
                pix = doc.load_page(pg - 1).get_pixmap(matrix=fitz.Matrix(150 / 72, 150 / 72))
                pix.save(cache)
            doc.close()
        except Exception:
            return ("render failed", 404)
    return send_file(cache) if os.path.exists(cache) else ("not found", 404)



def _ocr_tables(ih):
    """read-only parsed matrices for a stitch-pair half, rendered like the review screens."""
    import glob as _g
    hits = _g.glob(os.path.join(CORPUS, "digitized_tables", "*", f"{ih}_oneshot.json"))
    if not hits:
        return "<i style='color:#789'>no OCR read</i>"
    try:
        d = json.load(open(hits[0]))
    except Exception:
        return "<i style='color:#900'>unreadable JSON</i>"
    out = []
    for s in d.get("data", [])[:6]:
        if s.get("rows"):
            out.append(section_table(s.get("name", ""), s["rows"], limit=40))
    return "".join(out) or "<i style='color:#789'>no rows</i>"


@app.route("/stitch_review")
def stitch_review():
    """Manual stitch queue: label-less right-half + proposed left partner, scans side by side.
    Accept = merge is right (applied if shapes align) · Reject = wrong pairing · Skip."""
    q = json.load(open(os.path.join(ROOT, "export/stitch_manual_queue.json")))
    v = {}
    vp = os.path.join(ROOT, "export/stitch_verdicts.json")
    if os.path.exists(vp):
        try: v = json.load(open(vp))
        except Exception: v = {}
    todo = [p for p in q if p["right_ih"] not in v]
    # provable-merge candidates first (Accept can auto-apply those)
    todo.sort(key=lambda p: 0 if "not provable" in p.get("why", "") else 1)
    if not todo:
        return Response(f"<style>{CSS}</style><header><a href=/>home</a></header><h2>Stitch queue done ✅ ({len(v)} verdicts)</h2>", mimetype="text/html")
    p = todo[0]
    return Response(f"""<!doctype html><meta charset=utf-8><title>stitch review</title><style>{CSS}
.pair{{display:flex;gap:6px;padding:8px}} .pair div{{flex:1;text-align:center}}
.pair img{{max-width:100%;border:1px solid #ccd;border-radius:6px}}</style>
<header><a href=/>home</a> <b>Stitch review</b> · {len(todo)} pending · {len(v)} done ·
 <span style='color:#ffd966'>{html.escape(p.get("why",""))}</span></header>
<div class=meta><b>Is the RIGHT page the continuation of the LEFT page's rows?</b> (same table, rows aligned)
 · doc {p["dochash"][:10]} · left p{p["left_pg"]} / right p{p["right_pg"]} · {html.escape(p.get("title",""))}</div>
<div class=pair>
 <div><b>LEFT p{p["left_pg"]} (labels)</b><br><img src="/scan/{p["left_ih"]}"></div>
 <div><b>RIGHT p{p["right_pg"]} (no labels)</b><br><img src="/scan/{p["right_ih"]}"></div>
</div>
<div style="margin:8px 14px;color:#1f3864;font-weight:700">⬇ OCR'd data (left page, then right page)</div>
<div style="padding:0 14px 90px">{_ocr_tables(p["left_ih"])}<hr style="border:1px dashed #cbd">{_ocr_tables(p["right_ih"])}</div>
<div class=btns>
 <button class=ok onclick="sv('accept')">✓ Same table — stitch [1]</button>
 <button class=bad onclick="sv('reject')">✗ Not a pair [2]</button>
 <button class=skip onclick="sv('skip')">skip [3]</button>
 <button onclick="sv('degraded')" style="background:#8e7cc3;color:#fff">🖼 Degraded scan — re-render &amp; re-OCR [4]</button>
 <button onclick="sv('toc')" style="background:#b8860b;color:#fff" title="not a data table — it's a table of contents (we harvest these for authoritative page numbers)">📑 TOC page [5]</button>
</div>
<script>function sv(x){{fetch('/stitch_verdict',{{method:'POST',headers:{{'Content-Type':'application/json'}},
 body:JSON.stringify({{right_ih:{json.dumps(p["right_ih"])},v:x}})}}).then(()=>location.reload());}}
document.addEventListener('keydown',e=>{{if(e.key=='1')sv('accept');if(e.key=='2')sv('reject');if(e.key=='3')sv('skip');if(e.key=='4')sv('degraded');if(e.key=='5')sv('toc');}});</script>""",
                    mimetype="text/html")


@app.route("/stitch_verdict", methods=["POST"])
def stitch_verdict():
    j = request.get_json()
    vp = os.path.join(ROOT, "export/stitch_verdicts.json")
    v = {}
    if os.path.exists(vp):
        try: v = json.load(open(vp))
        except Exception: v = {}
    v[j["right_ih"]] = j["v"]
    json.dump(v, open(vp, "w"), indent=0)
    return jsonify(ok=True, n=len(v))


@app.route("/tables_list")
def tables_list():
    """The QUEUE, visible before clicking in: every pending table with severity + fails, PRIME-badged when
    the table feeds the best-per-place-year dataset. Rows link straight into /tables review."""
    items = imperfect_tables()
    v = _verd("tables")
    sev = _load_sev("tables")
    todo = [r for r in items if r.get("table_id") not in v]
    todo.sort(key=lambda r: -sev.get(r.get("table_id", ""), {}).get("sev", 0))
    # PRIME = member of a needs-review/minor run that is the chosen best table for a place-year
    prime_tids = set()
    prime_meta = {}
    try:
        import zipfile as _zf, io as _io
        z = _zf.ZipFile(os.path.join(ROOT, "census_site/data/tableVII_best.zip"))
        best = list(csv.DictReader(_io.TextIOWrapper(z.open("catalog.csv"))))
        want = {(r["doc"], r["page_range"]): r for r in best if r["quality"] in ("needs-review", "minor-issues", "incomplete")}
        _na = set()
        _naf = os.path.join(ROOT, "export/constraints_irrelevant.json")
        if os.path.exists(_naf):
            try: _na = set(json.load(open(_naf)))
            except Exception: _na = set()
        import re as _re
        for run in assembly_runs():
            key = (str(run["saoa"]), f"{run['page_range'][0]}-{run['page_range'][1]}")
            if key in want:
                for m in run.get("members", []):
                    _m = _re.search(r"([0-9a-f]{40})", m.get("table_id", ""))
                    if _m and _m.group(1) in _na:
                        continue                       # constraints-N/A page: verified inapplicable, skip
                    prime_tids.add(m.get("table_id", ""))
                    prime_meta[m.get("table_id", "")] = f"{want[key]['province']} {want[key]['year']} ({want[key]['quality']})"
    except Exception:
        pass
    todo.sort(key=lambda r: (0 if r.get("table_id") in prime_tids else 1,
                             -sev.get(r.get("table_id", ""), {}).get("sev", 0)))
    import urllib.parse as _up
    rows = []
    for i, r in enumerate(todo):
        tid = r.get("table_id", "")
        s = sev.get(tid, {})
        pm = prime_meta.get(tid, "")
        pb = (f"<span style='background:#0a5d1e;color:#fff;padding:1px 7px;border-radius:4px;font-size:11px;"
              f"font-weight:700'>PRIME</span> <span style='color:#456;font-size:11px'>{html.escape(pm)}</span>") if tid in prime_tids else ""
        sv = s.get("sev", 0)
        fl = s.get("fails", s.get("n", 0))
        sevb = (f"<span style='background:#9c0006;color:#fff;padding:1px 7px;border-radius:4px;font-size:11px'>"
                f"{fl} fails · sev {sv:,}</span>") if sv else "<span style='color:#9aa;font-size:11px'>structural</span>"
        rows.append(f"<tr><td>{i+1}</td><td>{pb}</td><td>{sevb}</td>"
                    f"<td><a href='/tables?tid={_up.quote(tid, safe='')}'>"
                    f"<b>{html.escape((r.get('title') or tid)[:64])}</b></a></td>"
                    f"<td style='color:#789;font-size:12px'>{html.escape((r.get('issues') or r.get('category') or '')[:44])}</td></tr>")
    nprime = sum(1 for t in todo if t.get("table_id") in prime_tids)
    return Response(f"""<!doctype html><meta charset=utf-8><title>tables queue</title><style>{CSS}
table.q{{border-collapse:collapse;margin:12px 18px;font-size:13px}}table.q td,table.q th{{border:1px solid #dde;padding:4px 9px}}
table.q th{{background:#eef3f7}}</style>
<header><a href=/>home</a><a href=/tables>▶ start reviewing</a><b>Tables queue</b> ·
 {len(todo)} pending ({nprime} PRIME — these feed the best-per-place-year dataset) · {len(v)} done</header>
<div class=meta>PRIME first (worst first within each tier). Click a row to open it in the review screen.
Rules: obvious misread → fix &amp; save · printed total genuinely wrong → ⚖ Source error · proportional table → ∅ Constraints N/A.</div>
<table class=q><tr><th>#</th><th>prime</th><th>severity</th><th>table</th><th>issue</th></tr>{''.join(rows)}</table>""",
                    mimetype="text/html")


@app.route("/tables")
def tables():
    items = imperfect_tables()
    v = _verd("tables")
    i = int(request.args.get("i", 0))
    todo = [r for r in items if r.get("table_id") not in v]
    _sev = _load_sev("tables")
    todo.sort(key=lambda r: -_sev.get(r.get("table_id", ""), {}).get("sev", 0))   # biggest/most failures first
    # PRIORITY: the human review list (export/final_review_queue.json) comes FIRST, in that order — these
    # are the pages you flagged; they lead the queue even if cleared from the severity list.
    _pqf = os.path.join(ROOT, "export/final_review_queue.json")
    if os.path.exists(_pqf):
        try:
            _pq = json.load(open(_pqf))
        except Exception:
            _pq = []
        _byih = {}
        for r in todo:
            mm = re.search(r"([0-9a-f]{40})", r.get("table_id", ""))
            if mm:
                _byih[mm.group(1)] = r
        priority, rest = [], list(todo)
        for it in _pq:
            ih0 = it.get("imghash", "")
            if ih0 in _byih:
                priority.append(_byih[ih0]); rest.remove(_byih[ih0])
            else:
                hit = glob.glob(os.path.join(CORPUS, "digitized_tables", "*", f"{ih0}_oneshot.json"))
                if hit:
                    base = hit[0].replace("export/ocr_results_referenced/", "").replace(".json", "")
                    priority.append({"table_id": base, "page": ""})
        todo = priority + rest
    if not todo:
        return Response(f"<style>{CSS}</style><header><a href=/>home</a></header><h2>All {len(items)} tables reviewed ✅</h2>", mimetype="text/html")
    tid_q = request.args.get("tid")              # stable table id — survives queue re-sorts (preferred)
    if tid_q:
        idx = next((i for i, x in enumerate(todo) if x.get("table_id") == tid_q), None)
        if idx is not None:
            t = idx
            r = todo[t]
        else:
            # tid not in the imperfect queue (e.g. a human-verified / print-error page you want to re-open):
            # load it directly by imghash so ANY page is reviewable here.
            m = re.search(r"([0-9a-f]{40})", tid_q)
            hit = glob.glob(os.path.join(CORPUS, "digitized_tables", "*", f"{(m.group(1) if m else tid_q)}_oneshot.json"))
            if hit:
                base = hit[0].replace("export/ocr_results_referenced/", "").replace(".json", "")
                r = {"table_id": base, "page": ""}
                t = 0
            else:
                t = max(0, min(int(request.args.get("t", 0)), len(todo) - 1))
                r = todo[t]
    else:
        t = max(0, min(int(request.args.get("t", 0)), len(todo) - 1))
        r = todo[t]
    tid = r.get("table_id", "")
    d, ih, ddir = load_table(tid)
    _cs = _sev.get(tid, {})
    _fl = _cs.get("fails", _cs.get("n", 0))
    sevbadge = (f"<span style='background:#9c0006;color:#fff;padding:1px 8px;border-radius:4px;font-weight:700' "
                f"title='ranked worst-first · {_cs.get('n',0)} failing rows · total magnitude {_cs.get('mag',0)}'>"
                f"&#9888; {_fl} fails · sev {_cs.get('sev',0):,}</span>") if _cs.get("sev") else ""
    title = (((d or {}).get("metadata", {}) or {}).get("title", "")) if d else ""
    run = run_of(tid) or neighborhood_of(tid)
    if run:
        scan_panels, mat_panels, gfails, n, cur_idx = assembled_panels(run, ih)
        pglist = [m.get("page", "") for m in sorted(run.get("members", []), key=lambda m: m.get("page", 0))]
        if gfails:
            head = gfails[:40]
            more = (f"<li>… and {len(gfails) - 40} more — use ◀/▶ to step through each page's full list</li>"
                    if len(gfails) > 40 else "")
            gf = (f"<div class=gfail><b>⚠ {len(gfails)} constraint failure(s) across {n} pages</b> — click a page to jump:<ul>"
                  + "".join(f"<li><a href=# onclick='showp({idx});return false'>p{pg}</a> &middot; row "
                            f"<b>{html.escape(str(l))}</b> — {html.escape(rs)}"
                            f"{(' &rarr; <span class=sugg>likely set ' + sg + '</span>') if sg else ''}</li>"
                            for pg, idx, l, rs, sg in head)
                  + more + "</ul></div>")
        else:
            gf = f"<div class=gok>✓ all constraints satisfied across all {n} pages</div>"
        nav = (f"<div class=nav><button type=button onclick='showp(cur-1)'>◀ Prev</button>"
               f"<span>page <b id=pn>1</b> / {n} &middot; <span id=ppg></span></span>"
               f"<button type=button onclick='showp(cur+1)'>Next ▶</button>"
               f"<span class=zhint>scan is on the left — click it to zoom; the matrix stays here</span></div>")
        scanblock = "".join(scan_panels)
        datablock = nav + gf + "".join(mat_panels)
        ctx = (f"<b style='background:#1f3864;color:#fff;padding:1px 8px;border-radius:4px'>doc {html.escape(str(run.get('saoa','?')))}</b>"
               f" · <b>{n}-page assembled table</b> · pp{run['page_range'][0]}–{run['page_range'][1]}")
    else:
        scanblock = f"<div class=pscan data-p=0 style='display:none'><img class=zimg src='/scan/{ih}' onclick='zoomtog(this)' title='click = zoom'></div>"
        datablock = (f"<div class=nav><span>single page (not part of a multi-page run)</span>"
                     f"<span class=zhint>click the scan to zoom</span></div>"
                     f"<div class=pmat data-p=0 style='display:none'>{matrix_edit_html(d)}</div>")
        n, cur_idx, pglist = 1, 0, [r.get("page", "")]
        ctx = ("<b style='background:#777;color:#fff;padding:1px 8px;border-radius:4px'>single page</b>"
               " · not part of a multi-page run")
    # neighbor pages (2 before start, 2 after end) rendered live from the drive — same as /assembly
    if run and run.get("dochash"):
        dh2 = run["dochash"]; s2, e2 = run["page_range"][0], run["page_range"][1]
        try:
            nbefore = max(2, min(60, int(request.args.get("before", 2))))
            nafter = max(2, min(60, int(request.args.get("after", 2))))
        except Exception:
            nbefore = nafter = 2
        def neigh(p, label):
            return (f"<div style='display:flex;flex-direction:column;gap:4px;align-items:center'>"
                    f"{context_card(dh2, p, label)}"
                    f"<button class=incbtn onclick='includePage({p})'>➕ include p{p} (OCR it)</button>"
                    f"<button class=titlebtn onclick='includeTitle({p})'>📑 title page (no OCR)</button></div>")
        before_ps = [p for p in range(s2 - nbefore, s2) if p >= 1]
        after_ps = list(range(e2 + 1, e2 + nafter + 1))
        cc = ([neigh(p, f"◀ before start · p{p}" + (" (just before)" if p == s2 - 1 else "")) for p in before_ps]
              + [neigh(p, f"after end · p{p} ▶" + (" (just after)" if p == e2 + 1 else "")) for p in after_ps])
        import urllib.parse
        qtid = urllib.parse.quote(tid, safe="")   # key the render-10 links to THIS table, not its position
        moreb = f"/tables?tid={qtid}&before={nbefore + 10}&after={nafter}"
        morea = f"/tables?tid={qtid}&before={nbefore}&after={nafter + 10}"
        op = " open" if (nbefore > 2 or nafter > 2) else ""
        ctxstrip = (f"<details{op}><summary class=ctxsum>▸ neighbor pages — {len(before_ps)} before start (p{s2}) &amp; "
                    f"{len(after_ps)} after end (p{e2}) &nbsp;·&nbsp; <a href='{moreb}'>◀ render 10 earlier</a> &nbsp;·&nbsp; "
                    f"<a href='{morea}'>render 10 later ▶</a> &nbsp; <b>➕ pulls a page in — including a far page pulls in "
                    f"everything up to it</b></summary><div class=ctxrow>{''.join(cc)}</div></details>")
    else:
        ctxstrip = ""
    return Response(f"""<!doctype html><meta charset=utf-8><title>doc {html.escape(str(run.get('saoa','?'))) if run else '—'} · review</title><style>{CSS}
.wrap{{display:flex;gap:0}}
.scancol{{flex:0 0 50%;overflow:auto;max-height:93vh;background:#272c33;padding:5px}}
.zimg{{width:100%;display:block;cursor:zoom-in;border:1px solid #445;border-radius:4px}}
.zimg.zoomed{{width:240%;max-width:none;cursor:zoom-out}}
.matcol{{flex:1;overflow:auto;max-height:93vh;padding:6px 12px}}
.nav{{position:sticky;top:0;background:#fff;border-bottom:1px solid #cdd;padding:6px 0 7px;display:flex;gap:12px;align-items:center;z-index:10}}
.nav button{{padding:5px 12px;background:#1f3864;color:#fff;border-radius:5px;margin:0}}
.nav .zhint{{color:#789;font-size:11px;margin-left:auto}}
.gfail{{background:#fff3f3;border:1px solid #f3c0c0;border-radius:6px;padding:6px 10px;margin:7px 0;font-size:12px;color:#9c0006}}
.gfail ul,.pfail ul{{margin:4px 0;padding-left:20px}}.gfail a{{color:#9c0006;font-weight:700}}
.gok,.pok{{color:#0a5d1e;font-weight:600;margin:7px 0;font-size:12px}}
.pfail{{background:#fff3f3;border:1px solid #f3c0c0;border-radius:6px;padding:5px 9px;margin:5px 0;font-size:12px;color:#9c0006}}
.ptitle{{font-weight:700;color:#1f3864;margin:4px 0 2px}}
.sugg{{color:#0a5d1e;font-weight:700;background:#e8f6ec;padding:0 4px;border-radius:3px}}
header a{{color:#ffd966;text-decoration:none}}
.pgreocr{{background:#ffe1e1;color:#9c0006;border:1px solid #e0a0a0;border-radius:5px;padding:2px 8px;font-size:11px;font-weight:700;cursor:pointer;margin-left:8px}}
.pgreocr.flagged{{background:#c6efce;color:#0a5d1e;border-color:#9bd0a8}}
.ctxsum{{cursor:pointer;padding:6px 12px;background:#f3eef6;font-weight:600;color:#7a2a6a;border-bottom:1px solid #e0cfe0}}
.ctxrow{{display:flex;gap:10px;overflow-x:auto;padding:8px 12px;background:#f7f2f8}}
.run{{width:240px;height:340px;object-fit:contain;cursor:zoom-in;background:#fff;border:1px solid #ccd}}
.pg{{flex:0 0 auto;font-size:11px;text-align:center;color:#555}}
.incbtn{{background:#0a5d1e;color:#fff;font-size:11px;padding:4px 8px;border:0;border-radius:5px;cursor:pointer;margin:0}}
.incbtn:hover{{background:#0c7a27}}
.titlebtn{{background:#6b4ea0;color:#fff;font-size:11px;padding:4px 8px;border:0;border-radius:5px;cursor:pointer;margin:0}}
.titlebtn:hover{{background:#7d5fb5}}
#lb{{display:none;position:fixed;inset:0;background:rgba(10,12,20,.94);z-index:100;overflow:auto;cursor:zoom-out}}
#lb img{{display:block;margin:14px auto;box-shadow:0 0 30px #000}} #lb.fit img{{max-width:96vw;max-height:94vh}}
#lbhint{{position:fixed;top:8px;right:14px;color:#cde;font-size:12px;z-index:101}}
</style>
<header><a href=/>home</a> <a href=/tables_list style="color:#ffd966">☰ queue</a> <a href=/grow_queue style="color:#ffd966">OCR queue</a><a id=doneBadge href=/grow_queue style="display:none;background:#0a5d1e;color:#fff;border-radius:10px;padding:1px 9px;margin-left:6px;text-decoration:none"></a> <b>Table review</b> &middot;
 {('<a href="/tables?t='+str(t-1)+'" style="color:#ffd966">◀ prev group</a>') if t>0 else '<span style="color:#8aa">◀ prev</span>'}
 &nbsp;<b>group {t+1} / {len(todo)}</b>&nbsp;
 {('<a href="/tables?t='+str(t+1)+'" style="color:#ffd966">next group ▶</a>') if t<len(todo)-1 else '<span style="color:#8aa">next ▶</span>'}
 <span class=prog>{len(items)-len(todo)} done</span></header>
<div class=meta>{sevbadge} &middot; {html.escape(title or tid)[:80]} &middot; {ctx} &middot; issue: <b>{html.escape(r.get('issues') or r.get('category') or '?')}</b></div>
{ctxstrip}
<div class=wrap><div class=scancol>{scanblock}</div><div class=matcol>{datablock}</div></div>
<div id=lb class=fit><span id=lbhint>click = toggle fit/100% · R = rotate · Esc/click-out = close</span><button onclick="event.stopPropagation();rotLb()" style="position:fixed;top:8px;left:14px;z-index:102;padding:6px 13px;background:#1f3864;color:#fff;border:0;border-radius:6px;cursor:pointer;font-weight:600">⟳ rotate</button><img id=lbimg></div>
<div class=btns>
 <button onclick="save()" style="background:#1f3864;color:#fff">💾 Save edits [S]</button>
 <input id=note placeholder="comment (e.g. 'col 3 misread')"
        style="width:26%;padding:6px 9px;border:1px solid #cce;border-radius:6px;margin:0 8px">
 <button class=ok onclick="v('accept')">✓ Accept [1]</button>
 <button class=bad onclick="v('reocr')">✗ Needs re-OCR [2]</button>
 <button onclick="sourceErr()" style="background:#e8a33d;color:#3a2a00">⚖ Source error [4]</button>
 <button onclick="constraintsNA()" style="background:#8e7cc3;color:#fff" title="distribution / proportion / rate table — the arithmetic identities (M+F=P, column sums) don't apply here">&#8709; Constraints N/A</button>
 <button class=skip style="background:#e0d4f5" onclick="rotAll()" title="rotate scans 90° clockwise (key: R)">⟳ Rotate [R]</button>
 <button class=skip onclick="v('skip')">skip [3]</button>
 <button class=skip style="background:#cfd8ff" onclick="undoTable()">↩ Undo last submit</button>
 <span style='color:#789'>&nbsp;⌘S/Ctrl+S = save &amp; next · ←/→ page · [1]accept [2]re-OCR [4]source-error [3]skip</span></div>
<script>const TID={json.dumps(tid)}; let dirty=false; var cur=0; const NP={n}; const PG={json.dumps(pglist)}; const START={cur_idx}; const T={t};
const SAOA={json.dumps(str(run.get('saoa'))) if (run and run.get('saoa')) else 'null'}, BST={(run.get('page_range') or [0,0])[0] if run else 0}, BEND={(run.get('page_range') or [0,0])[1] if run else 0};
function growBanner(txt){{
 var b=document.getElementById('growban');
 if(!b){{b=document.createElement('div');b.id='growban';
  b.style.cssText='position:fixed;bottom:64px;right:16px;background:#1f3864;color:#fff;padding:9px 15px;border-radius:7px;z-index:300;font-size:13px;box-shadow:0 3px 12px rgba(0,0,0,.3)';
  document.body.appendChild(b);}}
 b.innerHTML=txt;
}}
function pollGrow(dir){{
 fetch('/grow_status',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{saoa:SAOA}})}})
  .then(r=>r.json()).then(g=>{{
   if(g.done){{growBanner('✅ pages OCR\\'d — loading…');location.href='/tables?tid='+encodeURIComponent(TID)+dir;}}
   else{{growBanner('⏳ OCRing '+g.missing.length+" page(s) from the drive… they'll appear here automatically");setTimeout(function(){{pollGrow(dir);}},6000);}}
  }}).catch(function(){{setTimeout(function(){{pollGrow(dir);}},8000);}});
}}
function includePage(p){{
 if(!SAOA){{alert('this table is not part of an assembled run');return;}}
 var dir = (p < BST) ? '&before=10' : (p > BEND ? '&after=10' : '');
 fetch('/include_page',{{method:'POST',headers:{{'Content-Type':'application/json'}},
  body:JSON.stringify({{saoa:SAOA,start:BST,end:BEND,page:p}})}}).then(r=>r.json()).then(j=>{{
   if(!j.ok){{alert('include failed: '+(j.err||''));return;}}
   if(j.ocr_started){{growBanner('⏳ OCRing '+(j.ocr_pages?j.ocr_pages.length:1)+' page(s) from the drive…');pollGrow(dir);return;}}
   location.href='/tables?tid='+encodeURIComponent(TID)+dir;   // already OCR'd -> reload shows it + the next 10
 }});
}}
function includeTitle(p){{
 if(!SAOA){{alert('this table is not part of an assembled run');return;}}
 var dir = (p < BST) ? '&before=10' : (p > BEND ? '&after=10' : '');
 fetch('/include_title',{{method:'POST',headers:{{'Content-Type':'application/json'}},
  body:JSON.stringify({{saoa:SAOA,start:BST,end:BEND,page:p}})}}).then(r=>r.json()).then(j=>{{
   if(j.already_data){{alert('p'+p+' already has '+j.rows+" rows of real data — that's a DATA page, not a title page. Use ➕ include (OCR it) instead. Left as-is.");return;}}
   if(!j.ok){{alert('failed: '+(j.err||''));return;}}
   if(j.image_cached===false){{alert('p'+p+" added as a title page, but its scan could NOT be cached (drive not connected). Reconnect the drive and click OCR queue → 'cache title images' so it shows offline.");}}
   if(j.ocr_started){{growBanner('⏳ OCRing '+(j.ocr_pages?j.ocr_pages.length:0)+' page(s) pulled in between the title page and the old start…');pollGrow(dir);return;}}
   location.href='/tables?tid='+encodeURIComponent(TID)+dir;   // in the range as a title page (no OCR), no constraint noise
 }});
}}
function showp(i){{ if(i<0||i>=NP)return; cur=i;
 document.querySelectorAll('.pscan').forEach(e=>e.style.display=(+e.dataset.p===i)?'block':'none');
 document.querySelectorAll('.pmat').forEach(e=>e.style.display=(+e.dataset.p===i)?'block':'none');
 var pn=document.getElementById('pn'); if(pn)pn.textContent=i+1;
 var pp=document.getElementById('ppg'); if(pp)pp.textContent='p'+(PG[i]!=null?PG[i]:'');
 var sc=document.querySelector('.scancol'); if(sc)sc.scrollTop=0; }}
function zoomtog(img){{ img.classList.toggle('zoomed'); }}
var fit=true; var lbrot=0; var throt=0;
function zoom(u){{var lb=document.getElementById('lb');var im=document.getElementById('lbimg');im.src=u;lbrot=0;im.style.transform='';lb.style.display='block';fit=true;lb.classList.add('fit');}}
function rotLb(){{lbrot=(lbrot+90)%360;document.getElementById('lbimg').style.transform='rotate('+lbrot+'deg)';}}
function rotAll(){{throt=(throt+90)%360;document.querySelectorAll('img.zimg').forEach(function(i){{i.style.transform='rotate('+throt+'deg)';}});}}
document.getElementById('lb').onclick=function(e){{if(e.target.id=='lbimg'){{fit=!fit;this.classList.toggle('fit',fit);}}else{{this.style.display='none';}}}};
function save(){{
 const edits=[...document.querySelectorAll('input[data-s]')].map(i=>(
   {{tid:i.dataset.tid||'',s:+i.dataset.s,r:+i.dataset.r,g:i.dataset.g,k:i.dataset.k,value:i.value}}));
 fetch('/save_table',{{method:'POST',headers:{{'Content-Type':'application/json'}},
  body:JSON.stringify({{table_id:TID,edits:edits,note:document.getElementById('note').value}})}})
  .then(r=>r.json()).then(j=>{{if(j.ok){{dirty=false;sessionStorage.setItem('tblcur',cur);location.href='/tables?tid='+encodeURIComponent(TID);}}else alert('save failed: '+(j.err||''));}});
}}
function sourceErr(){{
 var n=document.getElementById('note').value.trim();
 if(!n) n='source error — transcription faithful, the printed table does not reconcile';
 fetch('/source_error',{{method:'POST',headers:{{'Content-Type':'application/json'}},
  body:JSON.stringify({{table_id:TID,note:n}})}}).then(()=>{{location.href='/tables?t='+T;}});
}}
function v(verd){{fetch('/tv',{{method:'POST',headers:{{'Content-Type':'application/json'}},
 body:JSON.stringify({{table_id:TID,verdict:verd,note:document.getElementById('note').value}})}}).then(()=>{{location.href='/tables?t='+T;}});}}
function constraintsNA(){{fetch('/constraints_na',{{method:'POST',headers:{{'Content-Type':'application/json'}},
 body:JSON.stringify({{table_id:TID,note:document.getElementById('note').value}})}}).then(()=>{{location.href='/tables?t='+T;}});}}
function undoTable(){{fetch('/undo_table',{{method:'POST'}}).then(r=>r.json()).then(j=>{{location.href='/tables?t=0';}});}}
function reocrPage(ih,idx){{
 var note=prompt("Re-OCR this page — what's wrong with it? (a note for the agent to act on; leave blank for a plain re-OCR)","")||"";
 fetch('/reocr_page',{{method:'POST',headers:{{'Content-Type':'application/json'}},
 body:JSON.stringify({{imghash:ih,note:note}})}}).then(r=>r.json()).then(j=>{{if(j.ok){{var b=document.getElementById('ro_'+idx);if(b){{b.textContent='✓ flagged'+(note?' + note':'');b.classList.add('flagged');}}}}else alert('flag failed');}});}}
window.addEventListener('beforeunload',e=>{{if(dirty){{e.preventDefault();e.returnValue='';}}}});
document.addEventListener('keydown',e=>{{
 if((e.key=='s'||e.key=='S')&&(e.metaKey||e.ctrlKey)){{e.preventDefault();save();return;}}   // ⌘S / Ctrl+S saves + advances
 if(e.target.tagName=='INPUT')return;
 if(e.key=='Escape'){{document.getElementById('lb').style.display='none';return;}}
 if(e.key=='r'||e.key=='R'){{if(document.getElementById('lb').style.display=='block')rotLb();else rotAll();return;}}
 if(document.getElementById('lb').style.display=='block')return;
 if(e.key=='ArrowLeft'){{showp(cur-1);return;}} if(e.key=='ArrowRight'){{showp(cur+1);return;}}
 if(e.key=='s'||e.key=='S')save();if(e.key=='1')v('accept');if(e.key=='2')v('reocr');if(e.key=='3')v('skip');if(e.key=='4')sourceErr();}});
// grid keyboard nav: Enter=down, Shift+Enter=up (same column); Tab=right, Shift+Tab=left (DOM order).
// Always select the destination's content so you can retype immediately.
document.addEventListener('keydown',function(e){{
 var el=e.target; if(el.tagName!=='INPUT'||el.dataset.s===undefined)return;
 var scope=el.closest('.pmat')||document, t=null;
 function colMove(dr){{ // same column (group+subkey), row +dr
  var g=el.dataset.g,k=el.dataset.k,r=+el.dataset.r,s=el.dataset.s,n=null;
  scope.querySelectorAll('input[data-s="'+s+'"]').forEach(function(c){{if(c.dataset.g===g&&c.dataset.k===k&&(+c.dataset.r)===r+dr)n=c;}});
  return n;
 }}
 function domMove(d){{ // previous/next editable cell in visual order
  var all=[].slice.call(scope.querySelectorAll('input[data-s]')); var i=all.indexOf(el); return all[i+d]||null;
 }}
 if(e.key==='Enter'){{t=colMove(e.shiftKey?-1:1); e.preventDefault();}}
 else if(e.key==='Tab'){{t=domMove(e.shiftKey?-1:1); e.preventDefault();}}
 if(t){{t.focus(); t.select();}}
}});
// focusing any cell selects its content so you can just type the correct value
document.addEventListener('focusin',function(e){{if(e.target.tagName==='INPUT'&&e.target.dataset.s!==undefined)e.target.select();}});
var _rc=sessionStorage.getItem('tblcur'); sessionStorage.removeItem('tblcur'); showp(_rc!=null?+_rc:START);
// poll for finished re-OCR runs -> header badge + desktop alert
(function(){{var last=-1;
 try{{if(window.Notification&&Notification.permission==='default')Notification.requestPermission();}}catch(e){{}}
 function poll(){{fetch('/reocr_done').then(r=>r.json()).then(j=>{{
   var b=document.getElementById('doneBadge');
   if(b){{ if(j.n>0){{b.style.display='inline';b.textContent='🔔 '+j.n+" re-OCR'd ready";}} else b.style.display='none'; }}
   if(last>=0 && j.n>last){{try{{if(window.Notification&&Notification.permission==='granted')
     new Notification((j.n-last)+' run(s) finished re-OCR',{{body:'Ready for full-run review — see OCR queue'}});}}catch(e){{}}}}
   last=j.n;
  }}).catch(function(){{}});}}
 poll(); setInterval(poll,25000);}})();
</script>""",
                    mimetype="text/html")


@app.route("/before_after")
def before_after():
    """Final review with BEFORE/AFTER toggle: scan on the left; on the right, switch between the original
    read (earliest backup) and the current corrected version. Current is editable (red=failing). Covers
    the action items + the confirmed ink-error pages. Source: export/final_review_queue.json."""
    qf = os.path.join(ROOT, "export/final_review_queue.json")
    if not os.path.exists(qf):
        return "<h2 style='font:16px system-ui;margin:40px'>No final review queue.</h2>"
    items = json.load(open(qf))
    n = len(items)
    i = max(0, min(int(request.args.get("i", 0)), n - 1)) if n else 0
    if not n:
        return "<h2>Empty.</h2>"
    it = items[i]
    ih = it["imghash"]
    tid = ih + "_oneshot"

    def render(dd, editable):
        parts = []
        for si, sec in enumerate(dd.get("data", [])):
            rows = sec.get("rows", [])
            if not rows:
                continue
            viol = dict(check_constraints(rows))
            if editable:
                parts.append(section_table_edit(sec.get("name", ""), rows, si, viol, limit=120, tid=tid))
            else:
                parts.append(section_table(sec.get("name", ""), rows, limit=120))
        return "".join(parts) or "<i>no rows</i>"

    after_body = render(json.load(open(it["corpus"])), True) if os.path.exists(it["corpus"]) else "<i>missing</i>"
    before_body = render(json.load(open(it["before"])), False) if it.get("before") and os.path.exists(it["before"]) else "<i style='color:#999'>no earlier version saved</i>"
    kindbadge = "🖋 confirmed ink error" if it["kind"] == "ink_error" else "🔧 needs action"
    cmt = html.escape(it.get("comment", "") or "")
    return f"""<!doctype html><meta charset=utf-8><title>Before/After {i+1}/{n}</title>
<style>body{{font:14px system-ui;margin:0;background:#fbfdff;color:#1a2233}}
header{{background:#1f3864;color:#fff;padding:10px 18px;position:sticky;top:0;z-index:5}}header a{{color:#ffd966;font-weight:700}}
button{{border:0;border-radius:6px;cursor:pointer;color:#fff;padding:6px 12px}}
.split{{display:flex}}.left{{width:44%;position:sticky;top:52px;align-self:flex-start;height:calc(100vh - 56px);overflow:auto;border-right:2px solid #d8dfe8;background:#f5f7fa}}
.left img{{width:100%;cursor:zoom-in}}.right{{width:56%;height:calc(100vh - 56px);overflow:auto;padding:0 14px 90px}}
.tog{{margin:8px 0}}.tog button{{background:#ccd;color:#333}}.tog button.on{{background:#1f3864;color:#fff}}
.guide{{background:#fff8e6;border:1px solid #e8d48a;border-radius:8px;padding:8px 12px;margin:10px 0;font-size:13px}}
table{{border-collapse:collapse;margin:8px 0}}td,th{{border:1px solid #dde;padding:2px 6px;font-size:12px}}</style>
<header>Before / After · <b>{i+1} / {n}</b> ·
{'<a href="?i='+str(i-1)+'">◀</a>' if i>0 else '◀'} {'<a href="?i='+str(i+1)+'">▶</a>' if i<n-1 else '▶'}
&nbsp;·&nbsp; <b>{kindbadge}</b> · fails now: <span style="color:#ffd966">{it['fails']}</span>
 &nbsp; <button onclick="ok('{ih}')" style="background:#0a5d1e;font-weight:700">✓ Approve &amp; next</button>
 <button onclick="saveEdits()" style="background:#b8860b">💾 Save edits</button>
 <button onclick="rrn('{ih}')" style="background:#456">re-read</button>
 <span id=msg style="margin-left:8px;color:#9be89b"></span></header>
<div class=split>
 <div class=left><img id=scan src="/scan/{ih}" onclick="zt()"></div>
 <div class=right>
  <div class=guide><b>{html.escape(it['desc'])}</b>{('<br><b>Your comment:</b> '+cmt) if cmt else ''}</div>
  <div class=tog><b>View:</b>
   <button id=bA class=on onclick="show('A')">After (current, editable)</button>
   <button id=bB onclick="show('B')">Before (original read)</button>
   <span style="font-size:12px;color:#789;margin-left:8px">backups: {', '.join(it.get('backups',[])) or 'none'}</span></div>
  <div id=after>{after_body}</div>
  <div id=before style="display:none">{before_body}</div>
 </div>
</div>
<script>
const nxt='?i={i+1}'; const TID={json.dumps(ih+'_oneshot')};
function show(w){{document.getElementById('after').style.display=w=='A'?'block':'none';document.getElementById('before').style.display=w=='B'?'block':'none';
 document.getElementById('bA').className=w=='A'?'on':'';document.getElementById('bB').className=w=='B'?'on':'';}}
function zt(){{const im=document.getElementById('scan');im.style.width=im.style.width=='200%'?'100%':'200%';}}
function saveEdits(){{const edits=[...document.querySelectorAll('#after input[data-s]')].map(x=>({{tid:x.dataset.tid,s:+x.dataset.s,r:+x.dataset.r,g:x.dataset.g,k:x.dataset.k,v:x.value}}));
 fetch('/save_table',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{table_id:TID,edits:edits}})}}).then(r=>r.json()).then(j=>{{document.getElementById('msg').textContent=j.ok?'✓ saved — reloading':'✗ fail';if(j.ok)setTimeout(()=>location.reload(),700);}});}}
function ok(ih){{fetch('/approve_table',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{left_ih:ih}})}}).then(()=>{{document.getElementById('msg').textContent='✓ approved';setTimeout(()=>location.href=nxt,400);}});}}
function rrn(ih){{fetch('/reread_numbers',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{left_ih:ih}})}}).then(()=>document.getElementById('msg').textContent='✓ queued re-read');}}
</script>"""


@app.route("/assembly_review")
def assembly_review():
    """Side-by-side review of the assembly review queue (export/prime_assembly_review_queue.csv): the
    window's full guidance (issue + description + suggested next step) on the right beside the editable
    data, the scan on the left. For adding-a-section / label-recovery cases, edit inline + Save."""
    import csv as _csv, glob as _g
    qf = os.path.join(ROOT, "export/prime_assembly_review_queue.csv")
    if not os.path.exists(qf):
        return "<h2 style='font:16px system-ui;margin:40px'>No review queue file.</h2>"
    items = list(_csv.DictReader(open(qf)))
    approved = set(json.load(open(os.path.join(ROOT, "export/acceptance_approved.json")))) if os.path.exists(os.path.join(ROOT, "export/acceptance_approved.json")) else set()
    n = len(items)
    i = max(0, min(int(request.args.get("i", 0)), n - 1)) if n else 0
    if not n:
        return "<h2 style='font:16px system-ui;margin:40px'>Empty queue.</h2>"
    it = items[i]
    ih = it["imghash"]
    fp = _g.glob(os.path.join(CORPUS, "digitized_tables", "*", f"{ih}_oneshot.json"))
    tid = ih + "_oneshot"
    body = "<i>corpus file not found</i>"
    if fp:
        d = json.load(open(fp[0]))
        parts = []
        for si, sec in enumerate(d.get("data", [])):
            rows = sec.get("rows", [])
            if not rows:
                continue
            viol = dict(check_constraints(rows))
            parts.append(section_table_edit(sec.get("name", ""), rows, si, viol, limit=120, tid=tid))
        body = "".join(parts) or "<i>no rows</i>"
    done = "✓ resolved (record only)" if it["issue_type"] == "confirmed_genuine_print_error" or ih in approved else ""
    cf = os.path.join(ROOT, "export/table_comments.json")
    cm = json.load(open(cf)) if os.path.exists(cf) else {}
    prior = cm.get(ih, "")
    return f"""<!doctype html><meta charset=utf-8><title>Assembly review {i+1}/{n}</title>
<style>body{{font:14px system-ui;margin:0;background:#fbfdff;color:#1a2233}}
header{{background:#1f3864;color:#fff;padding:10px 18px;position:sticky;top:0;z-index:5}}header a{{color:#ffd966;font-weight:700}}
button{{border:0;border-radius:6px;cursor:pointer;color:#fff;padding:6px 12px}}
.split{{display:flex}}.left{{width:46%;position:sticky;top:52px;align-self:flex-start;height:calc(100vh - 56px);overflow:auto;border-right:2px solid #d8dfe8;background:#f5f7fa}}
.left img{{width:100%;cursor:zoom-in}}.right{{width:54%;height:calc(100vh - 56px);overflow:auto;padding:0 14px 90px}}
.guide{{background:#fff8e6;border:1px solid #e8d48a;border-radius:8px;padding:10px 14px;margin:12px 0}}
table{{border-collapse:collapse;margin:8px 0}}td,th{{border:1px solid #dde;padding:2px 6px;font-size:12px}}</style>
<header>Assembly review · <b>{i+1} / {n}</b> ·
{'<a href="?i='+str(i-1)+'">◀ Prev</a>' if i>0 else 'Prev'} &nbsp; {'<a href="?i='+str(i+1)+'">Next ▶</a>' if i<n-1 else 'Next'}
&nbsp;·&nbsp; <b>{html.escape(it['place'])} · p{it['page']}</b> · <span style="color:#ffd966">{html.escape(it['issue_type'])}</span> {done}
 &nbsp; <button onclick="ok('{ih}')" style="background:#0a5d1e;font-weight:700">✓ Approve &amp; next</button>
 <button onclick="saveEdits()" style="background:#b8860b">💾 Save edits</button>
 <button onclick="rrn('{ih}')" style="background:#456">re-read</button>
 <span id=msg style="margin-left:8px;color:#9be89b"></span></header>
<div class=split>
 <div class=left><img id=scan src="/scan/{ih}" onclick="zt()"></div>
 <div class=right>
  <div class=guide><b>What the review found</b><br>{html.escape(it['description'])}
   <br><br><b>Suggested next step</b><br>{html.escape(it['suggested_next_step'])}</div>
  <input id=cmt placeholder="your note…" value="{html.escape(prior)}" style="width:70%;padding:4px 6px;border:1px solid #cce;border-radius:5px" onblur="savec('{ih}')">
  <div style="margin:8px 0;color:#7a1f1f;font-weight:700">data (red = failing; edit + Save)</div>
  {body}
 </div>
</div>
<script>
const nxt='?i={i+1}'; const TID={json.dumps(ih+'_oneshot')}; let zoomed=false;
function zt(){{zoomed=!zoomed;const im=document.getElementById('scan');im.style.width=zoomed?'200%':'100%';im.style.cursor=zoomed?'zoom-out':'zoom-in';}}
function saveEdits(){{
 const edits=[...document.querySelectorAll('input[data-s]')].map(x=>({{tid:x.dataset.tid,s:+x.dataset.s,r:+x.dataset.r,g:x.dataset.g,k:x.dataset.k,v:x.value}}));
 fetch('/save_table',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{table_id:TID,edits:edits}})}})
  .then(r=>r.json()).then(j=>{{document.getElementById('msg').textContent=j.ok?('✓ saved '+(j.changed||0)+' — reloading'):'✗ '+(j.err||'fail');if(j.ok)setTimeout(()=>location.reload(),700);}});
}}
function ok(ih){{fetch('/approve_table',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{left_ih:ih}})}}).then(()=>{{document.getElementById('msg').textContent='✓ approved';setTimeout(()=>location.href=nxt,400);}});}}
function rrn(ih){{var c=document.getElementById('cmt').value;fetch('/table_comment',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{left_ih:ih,comment:c}})}}).then(()=>fetch('/reread_numbers',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{left_ih:ih}})}})).then(()=>document.getElementById('msg').textContent='✓ queued re-read');}}
function savec(ih){{fetch('/table_comment',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{left_ih:ih,comment:document.getElementById('cmt').value}})}});}}
</script>"""


@app.route("/prime_problems")
def prime_problems():
    """Definitive worst-first queue of EVERY prime member page with residual fail-rows (uses the current
    layout-aware + missing-column checker, so it catches everything the old queue missed). Shows scan +
    data + the same fix/approve/comment actions as /acceptance."""
    import csv as _csv, io as _io, zipfile as _zip
    z = _zip.ZipFile(os.path.join(ROOT, "census_site/data/tableVII_best.zip"))
    best = [r for r in _csv.DictReader(_io.TextIOWrapper(z.open("catalog.csv")))
            if r["quality"] in ("needs-review", "minor-issues", "incomplete")]
    runs = {(str(r["saoa"]), f"{r['page_range'][0]}-{r['page_range'][1]}"): r for r in assembly_runs()}
    approved = set(json.load(open(os.path.join(ROOT, "export/acceptance_approved.json")))) if os.path.exists(os.path.join(ROOT, "export/acceptance_approved.json")) else set()
    items = []
    for b in best:
        run = runs.get((b["doc"], b["page_range"]))
        if not run:
            continue
        for m in run.get("members", []):
            import glob as _g2
            d, ih, _ = load_table(m["table_id"])
            if not d or ih in approved:
                continue
            if (d.get("metadata", {}) or {}).get("human_verified"):
                continue
            fp = _g2.glob(os.path.join(CORPUS, "digitized_tables", "*", f"{ih}_oneshot.json"))
            if not fp:
                continue
            tf = table_fail_rows(d)
            if tf > 0:
                items.append({"ih": ih, "tf": tf, "place": f"{b['province']} {b['year']}",
                              "page": m.get("page"), "quality": b["quality"],
                              "title": (d.get("metadata", {}) or {}).get("title", "")[:50], "path": fp[0]})
    items.sort(key=lambda x: -x["tf"])
    n = len(items)
    if not n:
        return "<h2 style='font:16px system-ui;margin:40px'>🎉 No prime pages with residual fails — all clear.</h2>"
    i = max(0, min(int(request.args.get("i", 0)), n - 1))
    s = items[i]
    d = json.load(open(s["path"]))
    tid = s["ih"] + "_oneshot"
    # build per-section violation maps: arithmetic fails + missing-column (blank M/F) cells, both red
    body_parts = []
    for si, sec in enumerate(d.get("data", [])):
        rows = sec.get("rows", [])
        if not rows:
            continue
        viol = dict(check_constraints(rows))
        # mark blank cells of a broken M/F group (the None-disruption defect) as red too
        mc = check_missing_columns({"data": [sec]})
        if mc:
            for ri, r in enumerate(rows):
                for g, v in r.items():
                    if isinstance(v, dict):
                        for k, val in v.items():
                            if val in (None, "", "None") and re.search(r"male|female", k, re.I):
                                viol[(ri, g, k)] = "blank column (missing M/F data)"
        body_parts.append(section_table_edit(sec.get("name", ""), rows, si, viol, limit=80, tid=tid))
    body = "".join(body_parts) or "<i>no rows</i>"
    cf = os.path.join(ROOT, "export/table_comments.json")
    cm = json.load(open(cf)) if os.path.exists(cf) else {}
    prior = cm.get(s["ih"], "")
    return f"""<!doctype html><meta charset=utf-8><title>Prime problems {i+1}/{n}</title>
<style>body{{font:14px system-ui;margin:0;background:#fbfdff;color:#1a2233}}
header{{background:#7a1f1f;color:#fff;padding:10px 18px;position:sticky;top:0;z-index:5}}header a{{color:#ffd966;font-weight:700}}
button{{border:0;border-radius:6px;cursor:pointer;color:#fff;padding:6px 12px}}
.split{{display:flex;gap:0;align-items:flex-start}}
.left{{width:48%;position:sticky;top:96px;align-self:flex-start;height:calc(100vh - 100px);overflow:auto;border-right:2px solid #e3d0d0;background:#f7f2f2}}
.left img{{width:100%;cursor:zoom-in;transition:transform .1s}}
.right{{width:52%;overflow:auto;height:calc(100vh - 100px);padding:0 12px 90px}}
table{{border-collapse:collapse;margin:8px 0}}td,th{{border:1px solid #dde;padding:2px 6px;font-size:12px}}</style>
<header>Prime problems · <b>{i+1} / {n}</b> pages with fails ·
{'<a href="?i='+str(i-1)+'">◀ Prev</a>' if i>0 else 'Prev'} &nbsp;
{'<a href="?i='+str(i+1)+'">Next ▶</a>' if i<n-1 else 'Next'} &nbsp;·&nbsp; <a href="/tables_list">queue</a>
&nbsp;·&nbsp; <b>{html.escape(s['place'])} · p{s['page']}</b> · <span style="color:#ffb3b3">{s['tf']} fail-row(s)</span>
 &nbsp; <button onclick="ok('{s['ih']}')" style="background:#0a5d1e;font-weight:700">✓ Approve &amp; next</button>
 <button onclick="saveEdits()" style="background:#b8860b">💾 Save edits</button>
 <button onclick="rrn('{s['ih']}')" style="background:#456">re-read</button>
 <button onclick="na('{s['ih']}')" style="background:#6b4ea0">∅ N/A</button>
 <span id=msg style="margin-left:8px;color:#9be89b"></span></header>
<div style="margin:4px 18px;font-size:12px;color:#555">{html.escape(s['title'])} · <span style="background:#ffd4d4;padding:0 5px">red = failing cell</span> (hover for reason) · click scan to zoom ·
 <input id=cmt placeholder="comment…" value="{html.escape(prior)}" style="width:30%;padding:3px 6px;border:1px solid #cce;border-radius:5px" onblur="savec('{s['ih']}')"></div>
<div class=split>
 <div class=left><img id=scan src="/scan/{s['ih']}" onclick="zoomToggle()"></div>
 <div class=right>{body}</div>
</div>
<script>
const nxt='?i={i+1}'; const TID={json.dumps(s['ih']+'_oneshot')}; let dirty=false;
let zoomed=false;
function zoomToggle(){{zoomed=!zoomed;const im=document.getElementById('scan');im.style.width=zoomed?'200%':'100%';im.style.cursor=zoomed?'zoom-out':'zoom-in';}}
document.addEventListener('input',e=>{{if(e.target.matches('input[data-s]'))dirty=true;}});
function saveEdits(){{
 const edits=[...document.querySelectorAll('input[data-s]')].map(i=>({{tid:i.dataset.tid,s:+i.dataset.s,r:+i.dataset.r,g:i.dataset.g,k:i.dataset.k,v:i.value}}));
 fetch('/save_table',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{table_id:TID,edits:edits}})}})
  .then(r=>r.json()).then(j=>{{document.getElementById('msg').textContent=j.ok?('✓ saved '+(j.changed||0)+' edits — reloading'):'✗ '+(j.err||'save failed');if(j.ok){{dirty=false;setTimeout(()=>location.reload(),700);}}}});
}}
function post(u,ih){{return fetch(u,{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{left_ih:ih,table_id:ih}})}}).then(r=>r.json());}}
function ok(ih){{post('/approve_table',ih).then(()=>{{document.getElementById('msg').textContent='✓ approved — next…';setTimeout(()=>location.href=nxt,400);}});}}
function rrn(ih){{var c=document.getElementById('cmt').value;fetch('/table_comment',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{left_ih:ih,comment:c}})}}).then(()=>post('/reread_numbers',ih)).then(j=>{{document.getElementById('msg').textContent='✓ queued re-read';}});}}
function na(ih){{fetch('/constraints_na',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{table_id:ih}})}}).then(()=>{{document.getElementById('msg').textContent='✓ marked N/A — next…';setTimeout(()=>location.href=nxt,400);}});}}
function savec(ih){{fetch('/table_comment',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{left_ih:ih,comment:document.getElementById('cmt').value}})}}).then(()=>document.getElementById('msg').textContent='✓ saved');}}
</script>"""


@app.route("/acceptance")
def acceptance():
    """Acceptance spot-check for the newly-stitched left/right merges: each shows the LEFT scan (labels),
    the RIGHT scan (the merged-in half), and the resulting merged data table. Scroll/step to sanity-check
    the numbers against the ink before release. Read-only."""
    import glob as _g
    prime_only = request.args.get("prime", "1") != "0"   # default: PRIME tables only (time-efficient)
    # imghashes of every member of a best-per-place-year (prime) run
    prime_ihs = set()
    if prime_only:
        try:
            import csv as _csv, io as _io, zipfile as _zip
            z = _zip.ZipFile(os.path.join(ROOT, "census_site/data/tableVII_best.zip"))
            prime_keys = {(r["doc"], r["page_range"]) for r in _csv.DictReader(_io.TextIOWrapper(z.open("catalog.csv")))}
            for run in assembly_runs():
                if (str(run["saoa"]), f"{run['page_range'][0]}-{run['page_range'][1]}") in prime_keys:
                    for mem in run.get("members", []):
                        mm = re.search(r"([0-9a-f]{40})", mem.get("table_id", ""))
                        if mm:
                            prime_ihs.add(mm.group(1))
        except Exception:
            prime_only = False   # if the best set isn't available, fall back to all
    stitched = []
    for f in _g.glob(os.path.join(CORPUS, "digitized_tables", "*", "*_oneshot.json")):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        md = d.get("metadata", {}) or {}
        rih = md.get("stitched_halfpage")
        if not rih:
            continue
        m = re.search(r"([0-9a-f]{40})", f)
        lih = m.group(1) if m else ""
        if prime_only and lih not in prime_ihs:
            continue
        stitched.append({"left_ih": lih, "right_ih": rih,
                         "by": md.get("stitch_approved_by", ""), "title": (md.get("title") or "")[:60],
                         "fails": table_fail_rows(d), "path": f})
    stitched.sort(key=lambda x: (x["fails"] == 0, x["title"]))   # residual-carrying first
    n = len(stitched)
    i = max(0, min(int(request.args.get("i", 0)), n - 1)) if n else 0
    if not n:
        return "<h2>No stitched tables found.</h2>"
    s = stitched[i]
    d = json.load(open(s["path"]))
    body = "".join(section_table(sec.get("name", ""), sec["rows"], limit=60)
                   for sec in d.get("data", []) if sec.get("rows")) or "<i>no rows</i>"
    flag = (f"<b style='color:#9c0006'>⚠ {s['fails']} residual fail-row(s)</b>" if s["fails"]
            else "<b style='color:#0a5d1e'>✓ all identities close</b>")
    # existing human comment for this table
    _cf = os.path.join(ROOT, "export/table_comments.json")
    _cm = json.load(open(_cf)) if os.path.exists(_cf) else {}
    prior = _cm.get(s["left_ih"], "")
    # the agent's own note from the read that produced this stitch (composite or alignment)
    agent_note = ""
    for src in (f"export/composite_reads/{s['left_ih']}.json", f"export/alignment_reads/{s['right_ih']}.json"):
        p = os.path.join(ROOT, src)
        if os.path.exists(p):
            try:
                agent_note = (json.load(open(p)).get("notes") or "").strip()
            except Exception:
                pass
            if agent_note:
                break
    md_flags = d.get("metadata", {}) or {}
    prov = ""
    if md_flags.get("composite_restitched"):
        prov = " · <span style='color:#6b4ea0'>composite-restitched</span>"
    return f"""<!doctype html><meta charset=utf-8><title>Acceptance {i+1}/{n}</title>
<style>body{{font:14px system-ui;margin:0;background:#fbfdff;color:#1a2233}}
header{{background:#1f3864;color:#fff;padding:10px 18px;position:sticky;top:0}}
header a{{color:#ffd966;font-weight:700}}
.pair{{display:flex;gap:10px;margin:10px 18px}}.pair>div{{flex:1}}
img{{max-width:100%;border:1px solid #ccd;border-radius:6px}}
table{{border-collapse:collapse;margin:8px 18px 90px}}td,th{{border:1px solid #dde;padding:2px 6px;font-size:12px}}</style>
<header>Acceptance spot-check · <b>{i+1} / {n}</b> {'PRIME ' if prime_only else ''}stitched tables ·
{'<a href="?i='+str(i-1)+('&prime=0' if not prime_only else '')+'">◀ Prev</a>' if i>0 else 'Prev'} &nbsp;
{'<a href="?i='+str(i+1)+('&prime=0' if not prime_only else '')+'">Next ▶</a>' if i<n-1 else 'Next'} &nbsp;·&nbsp;
{'<a href="?i=0&prime=0">show all</a>' if prime_only else '<a href="?i=0">prime only</a>'} &nbsp;·&nbsp;
<a href="/tables_list">queue</a></header>
<div style="margin:8px 18px"><b>{s['title']}</b> — {flag} — merged by <code>{s['by']}</code>{prov}
 &nbsp; <button onclick="ok('{s['left_ih']}')" style="background:#0a5d1e;color:#fff;border:0;padding:6px 16px;border-radius:6px;cursor:pointer;font-weight:700">✓ Looks good (approve &amp; next)</button>
 &nbsp; <button onclick="rrn('{s['left_ih']}')" style="background:#456;color:#fff;border:0;padding:5px 12px;border-radius:6px;cursor:pointer">stitch OK but a read is wrong</button>
 &nbsp; <button onclick="rej('{s['left_ih']}')" style="background:#9c0006;color:#fff;border:0;padding:5px 12px;border-radius:6px;cursor:pointer">✗ stitch is wrong</button>
 &nbsp; <a href="/tables" style="font-size:12px">edit cells by hand ▸</a>
 <span id=msg style="margin-left:10px;color:#0a5d1e"></span></div>
{"<div style='margin:2px 18px;color:#6b4ea0;font-size:12px'>📝 agent note: "+html.escape(agent_note)+"</div>" if agent_note else ""}
<div style="margin:6px 18px">
 <textarea id=cmt placeholder="your comment on this table (e.g. 'row 3 widowed looks off')… saved for you + fed to any re-read" style="width:60%;height:44px;padding:6px;border:1px solid #cce;border-radius:6px">{html.escape(prior)}</textarea>
 <button onclick="savec('{s['left_ih']}')" style="background:#1f3864;color:#fff;border:0;padding:6px 12px;border-radius:6px;cursor:pointer;vertical-align:top">Save comment</button>
 <span id=cmsg style="margin-left:8px;color:#0a5d1e"></span></div>
<script>
function ok(lih){{
 fetch('/approve_table',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{left_ih:lih}})}})
  .then(r=>r.json()).then(j=>{{document.getElementById('msg').textContent='✓ approved ('+j.total+' total) — next…';
    setTimeout(()=>location.href='?i={i+1}'+({str(not prime_only).lower()}?'&prime=0':''),500);}});
}}
function savec(lih){{
 fetch('/table_comment',{{method:'POST',headers:{{'Content-Type':'application/json'}},
   body:JSON.stringify({{left_ih:lih,comment:document.getElementById('cmt').value}})}})
   .then(r=>r.json()).then(j=>{{document.getElementById('cmsg').textContent=j.saved?'✓ saved':'✓ cleared';}});
}}
function rrn(lih){{
 // save any comment first so it guides the re-read, then queue a keep-stitch re-read
 var c=document.getElementById('cmt').value;
 fetch('/table_comment',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{left_ih:lih,comment:c}})}})
  .then(()=>fetch('/reread_numbers',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{left_ih:lih}})}}))
  .then(r=>r.json()).then(j=>{{document.getElementById('msg').textContent=j.ok?'✓ stitch kept — numbers queued for re-read ('+j.queued+' total)':'✗ failed';}});
}}
function rej(lih){{
 fetch('/reject_stitch',{{method:'POST',headers:{{'Content-Type':'application/json'}},
   body:JSON.stringify({{left_ih:lih}})}}).then(r=>r.json()).then(j=>{{
   document.getElementById('msg').textContent = j.ok
     ? '✓ un-merged'+(j.restored?' (restored '+j.restored+')':'')+' — queued for composite re-read ('+j.queued+' total). Advancing…'
     : '✗ '+(j.err||'failed');
   if(j.ok) setTimeout(()=>location.href='?i={i}',900);
 }});
}}
</script>
<div class=pair>
 <div><b>LEFT (labels)</b><br><img src="/scan/{s['left_ih']}"></div>
 <div><b>RIGHT (merged-in half)</b><br><img src="/scan/{s['right_ih']}"></div>
</div>
<div style="margin:6px 18px;color:#1f3864;font-weight:700">⬇ merged data</div>
{body}"""


@app.route("/table_comment", methods=["POST"])
def table_comment():
    """Save a free-text comment on a table (keyed by left_ih). Comments persist and show on the acceptance
    page + feed future re-read tasks so a human note ('row 3 looks off') guides the machine."""
    j = request.get_json()
    lih = j.get("left_ih", "")
    cf = os.path.join(ROOT, "export/table_comments.json")
    c = json.load(open(cf)) if os.path.exists(cf) else {}
    txt = (j.get("comment") or "").strip()
    if txt:
        c[lih] = txt
    else:
        c.pop(lih, None)
    json.dump(c, open(cf, "w"), indent=1)
    return jsonify(ok=True, saved=bool(txt), n=len(c))


@app.route("/approve_table", methods=["POST"])
def approve_table():
    """The table looks GOOD. Mark it human-verified (durable) so it's trusted, drops off the review list,
    and is never silently re-flagged. Records the approval + advances."""
    import glob as _g
    j = request.get_json()
    lih = j.get("left_ih", "")
    hits = _g.glob(os.path.join(CORPUS, "digitized_tables", "*", f"{lih}_oneshot.json"))
    if hits:
        d = json.load(open(hits[0]))
        d.setdefault("metadata", {})["human_verified"] = True
        json.dump(d, open(hits[0], "w"))
    af = os.path.join(ROOT, "export/acceptance_approved.json")
    a = json.load(open(af)) if os.path.exists(af) else []
    if lih not in a:
        a.append(lih)
    json.dump(a, open(af, "w"), indent=1)
    return jsonify(ok=True, approved=lih, total=len(a))


@app.route("/reread_numbers", methods=["POST"])
def reread_numbers():
    """The STITCH is GOOD but a READ is wrong. Keep the merge intact; queue a composite re-read that treats
    the pairing as CONFIRMED and only re-reads cells to close identities. keep-best applies the result."""
    import glob as _g
    j = request.get_json()
    lih = j.get("left_ih", "")
    hits = _g.glob(os.path.join(CORPUS, "digitized_tables", "*", f"{lih}_oneshot.json"))
    if not hits:
        return jsonify(ok=False, err="left page not found")
    lp = hits[0]
    d = json.load(open(lp))
    rih = (d.get("metadata", {}) or {}).get("stitched_halfpage", "")
    composite_ok = False
    try:
        os.makedirs(os.path.join(ROOT, "export/composite_tasks"), exist_ok=True)
        cimg = os.path.join(ROOT, f"export/composite_tasks/{lih}.jpg")
        subprocess.run(["python3", os.path.join(ROOT, "make_composite.py"), lih, rih, cimg],
                       check=True, capture_output=True, timeout=60)
        secs = []
        for sec in d.get("data", []):
            rows = sec.get("rows", [])
            rh = next((k for k in rows[0] if not isinstance(rows[0].get(k), dict)), None) if rows else None
            secs.append({"name": sec.get("name", ""), "n_rows": len(rows),
                         "labels": [str(r.get(rh, ""))[:22] for r in rows]})
        # pairing_confirmed tells the agent: do NOT re-pair, only fix digits to close identities
        cf = os.path.join(ROOT, "export/table_comments.json")
        cm = json.load(open(cf)) if os.path.exists(cf) else {}
        json.dump({"left_ih": lih, "right_ih": rih, "title": (d.get("metadata", {}) or {}).get("title", "")[:40],
                   "composite": cimg, "corpus": lp, "sections": secs,
                   "pairing_confirmed": True, "human_comment": cm.get(lih, "")},
                  open(os.path.join(ROOT, f"export/composite_tasks/{lih}.json"), "w"))
        composite_ok = True
    except Exception:
        pass
    rq = os.path.join(ROOT, "export/reread_queue.json")
    q = json.load(open(rq)) if os.path.exists(rq) else []
    q.append({"left_ih": lih, "right_ih": rih, "composite_ready": composite_ok, "keep_stitch": True})
    json.dump(q, open(rq, "w"), indent=1)
    return jsonify(ok=True, left=lih, composite=composite_ok, queued=len(q))


@app.route("/reject_stitch", methods=["POST"])
def reject_stitch():
    """The stitch is WRONG. Un-merge it: restore the pre-stitch backup (reverting the left page to its
    unstitched state), clear the stitch metadata, and queue the pair for the composite-image re-read."""
    import glob as _g
    j = request.get_json()
    lih = j.get("left_ih", "")
    hits = _g.glob(os.path.join(CORPUS, "digitized_tables", "*", f"{lih}_oneshot.json"))
    if not hits:
        return jsonify(ok=False, err="left page not found")
    lp = hits[0]
    d = json.load(open(lp))
    rih = (d.get("metadata", {}) or {}).get("stitched_halfpage", "")
    # revert to the most recent pre-merge backup
    restored = None
    for bak in ("_oneshot.prestitch.json", "_oneshot.preverified.json", "_oneshot.prefloodfix.json"):
        bp = lp.replace("_oneshot.json", bak)
        if os.path.exists(bp):
            try:
                json.load(open(bp)); shutil.copy(bp, lp); restored = bak; break
            except Exception:
                continue
    if not restored:
        # no backup: just strip the stitch metadata so it's no longer treated as merged
        d.get("metadata", {}).pop("stitched_halfpage", None)
        d.get("metadata", {}).pop("stitch_approved_by", None)
        json.dump(d, open(lp, "w"))
    # build the side-by-side composite image + task NOW so the re-read is ready to run
    composite_ok = False
    try:
        os.makedirs(os.path.join(ROOT, "export/composite_tasks"), exist_ok=True)
        cimg = os.path.join(ROOT, f"export/composite_tasks/{lih}.jpg")
        subprocess.run(["python3", os.path.join(ROOT, "make_composite.py"), lih, rih, cimg],
                       check=True, capture_output=True, timeout=60)
        secs = []
        dd = json.load(open(lp))
        for sec in dd.get("data", []):
            rows = sec.get("rows", [])
            rh = next((k for k in rows[0] if not isinstance(rows[0].get(k), dict)), None) if rows else None
            secs.append({"name": sec.get("name", ""), "n_rows": len(rows),
                         "labels": [str(r.get(rh, ""))[:22] for r in rows]})
        json.dump({"left_ih": lih, "right_ih": rih, "title": (dd.get("metadata", {}) or {}).get("title", "")[:40],
                   "composite": cimg, "corpus": lp, "sections": secs, "from_reject": True},
                  open(os.path.join(ROOT, f"export/composite_tasks/{lih}.json"), "w"))
        composite_ok = True
    except Exception:
        pass
    # queue for composite re-read (the re-read agent picks up composite_tasks/*.json with from_reject)
    rq = os.path.join(ROOT, "export/stitch_rejects.json")
    q = json.load(open(rq)) if os.path.exists(rq) else []
    q.append({"left_ih": lih, "right_ih": rih, "note": (j.get("note") or "").strip(),
              "restored": restored, "composite_ready": composite_ok})
    json.dump(q, open(rq, "w"), indent=1)
    return jsonify(ok=True, rejected=lih, right=rih, restored=restored,
                   composite=composite_ok, queued=len(q))


@app.route("/tv", methods=["POST"])
def tv():
    j = request.get_json()
    v = _verd("tables")
    v[j["table_id"]] = {"v": j["verdict"], "note": (j.get("note") or "").strip()}
    _save_verd("tables", v)
    return jsonify(ok=True, n=len(v))


@app.route("/source_error", methods=["POST"])
def source_error():
    """Accept a table as a SOURCE error — the read is faithful but the printed table doesn't reconcile.
    Records the verdict AND protects the faithful read on the non-reconciling pages (sets human_corrected
    + source_error) so no future re-OCR/merge/frozen pass overwrites it."""
    j = request.get_json()
    tid = j.get("table_id", "")
    note = (j.get("note") or "").strip()
    v = _verd("tables")
    v[tid] = {"v": "source_error", "note": note}
    _save_verd("tables", v)
    run = run_of(tid) or neighborhood_of(tid)
    members = (run or {}).get("members", []) or [{"table_id": tid}]
    protected = 0
    for m in members:
        mt = m.get("table_id", "")
        d, _ih, _ = load_table(mt)
        if not d:
            continue
        fails = sum(len(check_constraints(s.get("rows", []))) for s in d.get("data", []))
        if fails == 0:
            continue                              # only protect pages that actually fail (the source-error ones)
        path = os.path.join(CORPUS, mt + ".json")
        try:
            dd = json.load(open(path))
            md = dd.setdefault("metadata", {})
            md["human_corrected"] = True; md["source_error"] = True
            json.dump(dd, open(path, "w"))
            protected += 1
        except Exception:
            pass
    return jsonify(ok=True, protected=protected)


@app.route("/constraints_na", methods=["POST"])
def constraints_na():
    """Mark a table whose arithmetic constraints DON'T APPLY (a distribution / proportion / rate table,
    Race-Caste, etc. — M+F=P, U+M+W=Total and column sums are meaningless here). Removes it from the queue,
    marks the corpus read durably (metadata.constraints_irrelevant) and records its imghash so severity and
    future validation ignore its 'failures'."""
    j = request.get_json()
    tid = j.get("table_id", "")
    note = (j.get("note") or "").strip()
    v = _verd("tables")
    v[tid] = {"v": "constraints_na", "note": note}
    _save_verd("tables", v)
    d, ih, _ = load_table(tid)
    naf = os.path.join(ROOT, "export/constraints_irrelevant.json")
    naset = set()
    if os.path.exists(naf):
        try:
            naset = set(json.load(open(naf)))
        except Exception:
            naset = set()
    marked = 0
    if ih:
        naset.add(ih)
        for p in glob.glob(os.path.join(CORPUS, "*", f"{ih}_oneshot.json")):
            try:
                dd = json.load(open(p))
                dd.setdefault("metadata", {})["constraints_irrelevant"] = True
                json.dump(dd, open(p, "w"))
                marked += 1
            except Exception:
                pass
    json.dump(sorted(naset), open(naf, "w"))
    return jsonify(ok=True, marked=marked, total=len(naset))


@app.route("/undo_table", methods=["POST"])
def undo_table():
    """Remove the most-recently-submitted table verdict so that table comes back into the queue
    (recover an accidental Accept/re-OCR/skip)."""
    v = _verd("tables")
    if v:
        last = list(v.keys())[-1]
        v.pop(last)
        _save_verd("tables", v)
        return jsonify(ok=True, restored=last)
    return jsonify(ok=True, restored=None)


@app.route("/reocr_page", methods=["POST"])
def reocr_page():
    """Flag ONE page (by imghash) for re-OCR — sets metadata.needs_reocr on its oneshot and appends to
    export/reocr_page_flags.txt, which the per-page re-OCR pass reads. Leaves the rest of the table alone."""
    ih = (request.get_json(force=True) or {}).get("imghash", "")
    if not re.fullmatch(r"[0-9a-f]{40}", ih or ""):
        return jsonify(ok=False, err="bad imghash")
    flagf = os.path.join(ROOT, "export/reocr_page_flags.txt")
    have = set(open(flagf).read().split()) if os.path.exists(flagf) else set()
    if ih not in have:
        open(flagf, "a").write(ih + "\n")
    note = ((request.get_json(force=True) or {}).get("note") or "").strip()
    title = ""
    for jf in glob.glob(os.path.join(ROOT, "export/ocr_results_referenced/digitized_tables", "*", f"{ih}_oneshot.json")):
        try:
            d = json.load(open(jf))
            md = d.setdefault("metadata", {})
            md["needs_reocr"] = True
            if note:
                md["reocr_note"] = note
            title = md.get("title", "") or title
            json.dump(d, open(jf, "w"))
        except Exception:
            pass
    if note:
        import datetime
        nf = os.path.join(ROOT, "export/reocr_notes.json")
        notes = {}
        if os.path.exists(nf):
            try:
                notes = json.load(open(nf))
            except Exception:
                notes = {}
        notes[ih] = {"note": note, "title": title[:70], "ts": datetime.datetime.now().strftime("%m-%d %H:%M"), "done": False}
        json.dump(notes, open(nf, "w"), indent=1)
    return jsonify(ok=True, noted=bool(note))


@app.route("/save_table", methods=["POST"])
def save_table():
    """Persist inline cell edits into the corpus read JSON(s). In the assembled multi-page view each
    cell carries its own member table_id, so edits route back to the correct page. Backs up originals
    to *_oneshot.preedit.json and marks human_corrected."""
    j = request.get_json()
    main_tid = j.get("table_id", "")

    def parse(v):
        s = str(v).replace(",", "").strip()
        if re.fullmatch(r"-?\d+", s):
            return int(s)
        try:
            return float(s)
        except Exception:
            return s

    def resolve(tid):
        t = tid.replace("export/ocr_results_referenced/", "").strip("/")
        jf = os.path.join(CORPUS, t + ".json")
        if not os.path.exists(jf):
            hit = glob.glob(os.path.join(CORPUS, "digitized_tables", "*", os.path.basename(t) + ".json"))
            jf = hit[0] if hit else None
        return jf

    by_tid = {}
    for e in j.get("edits", []):
        by_tid.setdefault(e.get("tid") or main_tid, []).append(e)

    changed = 0
    for tid, edits in by_tid.items():
        jf = resolve(tid)
        if not jf or not os.path.exists(jf):
            continue
        try:
            d = json.load(open(jf))
        except Exception:
            continue
        data = d.get("data", [])
        tid_changed = 0
        for e in edits:
            si, ri, g, k = e.get("s"), e.get("r"), e.get("g"), e.get("k")
            if si is None or ri is None or not g or si >= len(data):
                continue
            rows = data[si].get("rows", [])
            if ri >= len(rows):
                continue
            row = rows[ri]
            nv = parse(e.get("value", ""))
            if k:
                if not isinstance(row.get(g), dict):
                    row[g] = {}
                if str(row[g].get(k)) != str(nv):
                    tid_changed += 1
                row[g][k] = nv
            else:
                if str(row.get(g)) != str(nv):
                    tid_changed += 1
                row[g] = nv
        if tid_changed == 0:        # this page had NO actual edits -> leave it completely untouched
            continue
        bak = jf.replace("_oneshot.json", "_oneshot.preedit.json")
        if not os.path.exists(bak):
            try:
                open(bak, "w").write(open(jf).read())   # original (file not yet rewritten)
            except Exception:
                pass
        d.setdefault("metadata", {})["human_corrected"] = True
        json.dump(d, open(jf, "w"))
        changed += tid_changed
    note = (j.get("note") or "").strip()
    if note:                              # persist comments even on a plain Save (not just verdicts)
        cp = os.path.join(OUT, "comments.json")
        try:
            c = json.load(open(cp)) if os.path.exists(cp) else {}
        except Exception:
            c = {}
        c[main_tid] = note
        json.dump(c, open(cp, "w"), indent=1)
    return jsonify(ok=True, changed=changed, noted=bool(note))


def stitched_matrix(members):
    """preview: stack every member page's sections into one table (the would-be assembled result)."""
    blocks = []
    total_rows = 0
    for m in members:
        d, ih, ddir = load_table(m["table_id"])
        if not d:
            blocks.append(f"<div class=ssec style='color:#a06'>p{m['page']}: (no data)</div>")
            continue
        for s in d.get("data", []):
            rows = s.get("rows", [])
            if not rows:
                continue
            total_rows += len(rows)
            blocks.append(section_table(f"p{m['page']} · {s.get('name', '')}", rows, limit=80))
    return f"<div style='color:#456;margin:4px 0'>stitched preview — {total_rows} rows across {len(members)} pages</div>" + "".join(blocks)


def low_conf_index(runs):
    """current-run key -> [AI low-confidence reasons], matched from boundary_verdicts.json by saoa +
    a boundary page adjacent to the run's range. Lets the review surface the AI-uncertain runs first."""
    vf = os.path.join(ROOT, "export/boundary_verdicts.json")
    out = {}
    if os.path.exists(vf):
        try:
            verds = json.load(open(vf))
        except Exception:
            verds = []
        for x in verds:
            if x.get("confidence") != "low":
                continue
            sa = x["run"].split(":")[0]
            bp = int(x["boundary_page"])
            for run in runs:
                if run["saoa"] == sa and run["page_range"][0] - 2 <= bp <= run["page_range"][1] + 2:
                    out.setdefault(f"{run['saoa']}:{run['page_range']}", []).append(
                        f"{x['side']} p{bp}: {x.get('reason', '')[:70]}")
                    break
    return out


@app.route("/assembly")
def assembly():
    runs = assembly_runs()
    v = _verd("assembly")
    lc = low_conf_index(runs)
    q = _run_quality()
    def qof(r):
        return q.get((str(r["saoa"]), f"{r['page_range'][0]}-{r['page_range'][1]}"), "")
    # earlier queue: every run not yet reviewed, AI-uncertain ones first (a run leaves the moment it's reviewed)
    todo = [r for r in runs if f"{r['saoa']}:{r['page_range']}" not in v]
    _sevr = _load_sev("runs")
    def _rsev(r):
        return _sevr.get(f"{r['saoa']}:{r['page_range']}", {}).get("sev", 0)
    # bounds-suspects (printed-title evidence the start/end is off) jump the queue — fix bounds before data
    _bs = {}
    _bsp = os.path.join(ROOT, "export/bounds_suspects.json")
    if os.path.exists(_bsp):
        try:
            _bs = {f"{x['saoa']}:{x['range']}": x["evidence"] for x in json.load(open(_bsp))}
        except Exception:
            _bs = {}
    todo.sort(key=lambda r: (0 if f"{r['saoa']}:{r['page_range']}" in _bs else 1,
                             -_rsev(r), 0 if f"{r['saoa']}:{r['page_range']}" in lc else 1))
    if not todo:
        return Response(f"<style>{CSS}</style><header><a href=/>home</a></header><h2>All {len(runs)} runs reviewed ✅</h2>", mimetype="text/html")
    r = todo[0]
    key = f"{r['saoa']}:{r['page_range']}"
    lcnote = lc.get(key)
    lcbar = ('<div class=flag style="background:#e7f0ff;border-color:#9bd;color:#234">🤖 <b>AI was unsure here — '
             'your call most needed:</b> ' + ' · '.join(html.escape(x) for x in lcnote) + '</div>') if lcnote else ''
    if key in _bs:
        lcbar += ('<div class=flag style="background:#fff3e0;border-color:#e0b070;color:#5a3a00">📐 <b>Bounds suspect '
                  f'({len(_bs)} flagged, shown first):</b> '
                  + ' · '.join(html.escape(x) for x in _bs[key])
                  + ' — check the neighbor pages below, then use <b>Set exact bounds</b>.</div>')
    dh = r.get("dochash", "")
    s0, e0 = r["page_range"][0], r["page_range"][1]
    # 2 pages BEFORE the start — does the table begin earlier? (closest to start on the right)
    pages_html = [context_card(dh, p, ("◀ before start · p%d" % p) + (" (just before)" if p == s0 - 1 else ""))
                  for p in (s0 - 2, s0 - 1) if p >= 1]
    for m in r["members"]:
        d, ih, ddir = load_table(m["table_id"])
        ttl = (((d or {}).get("metadata", {}) or {}).get("title", "") or "")[:48]
        sec = "; ".join(s.get("name", "") for s in (d or {}).get("data", []))[:50] if d else ""
        # ih may be "" for bgap placeholders -> resolve_page falls back to the drive PDF via dh+page
        if not ih:
            mm = re.search(r"([0-9a-f]{40})_oneshot", m["table_id"])
            ih = mm.group(1) if mm else ""
        tdir = os.path.basename(os.path.dirname(m["table_id"]))
        pages_html.append(f"<div class=pg>{page_img(dh, m['page'], ih, tdir)}"
                          f"<br><b>p{m['page']}</b> · {html.escape(ttl)}<br><span style='color:#789'>{html.escape(sec)}</span></div>")
    # 2 pages AFTER the end — does the table continue past the boundary? (closest to end on the left)
    for p in (e0 + 1, e0 + 2):
        pages_html.append(context_card(dh, p, ("after end · p%d ▶" % p) + (" (just after)" if p == e0 + 1 else "")))
    stype = r.get("start_type", "?")
    flagbar = ('<div class=flag>⚠ <b>Possible boundary issue:</b> ' + ' · '.join(html.escape(f) for f in r['flags']) + '</div>') if r.get('flags') else ''
    return Response(f"""<!doctype html><meta charset=utf-8><style>{CSS}
.bar{{position:sticky;top:0;z-index:10;background:#fff;border-bottom:2px solid #1f3864;padding:8px 18px;box-shadow:0 2px 6px rgba(0,0,0,.08)}}
.bar .q{{font-size:15px;font-weight:700;color:#1f3864;margin-bottom:5px}}
.bar .meta2{{font-size:12px;color:#456;margin-bottom:6px}}
.bar input{{width:46%;padding:7px 10px;border:1px solid #cce;border-radius:6px;margin-right:8px}}
.bar button{{padding:8px 15px}} .flag{{margin:6px 18px;padding:6px 12px;background:#fff3cd;border:1px solid #ffe08a;border-radius:6px;color:#7a5b00}}
.run{{width:340px;height:460px;object-fit:contain;cursor:zoom-in;background:#fff;border:1px solid #ccd}} .run:hover{{box-shadow:0 0 0 3px #1f3864aa}}
.runpages{{display:flex;gap:12px;overflow-x:auto;padding:10px 18px;background:#eef1f6}}
#lb{{display:none;position:fixed;inset:0;background:rgba(10,12,20,.94);z-index:100;overflow:auto;cursor:zoom-out}}
#lb img{{display:block;margin:14px auto;box-shadow:0 0 30px #000}} #lb.fit img{{max-width:96vw;max-height:94vh}}
#lbhint{{position:fixed;top:8px;right:14px;color:#cde;font-size:12px;z-index:101}}
.ssec{{font-weight:700;color:#1f3864;font-size:12px;margin:8px 18px 2px}} table.stitch{{border-collapse:collapse;font-size:11px;margin:0 18px}}
table.stitch th,table.stitch td{{border:1px solid #dde;padding:1px 5px}} table.stitch th{{background:#eef3fb}}
.toggle{{margin:8px 18px;cursor:pointer;color:#1f3864;font-weight:600}} #stitch{{display:none}}</style>
<div class=bar>{lcbar}
 <div class=q>{len(todo)} run(s) left · <b style='color:#9c0006'>sorted worst-first</b> (severity {_rsev(r):,} · {_sevr.get(f"{r['saoa']}:{r['page_range']}",{}).get('n',0)} failing rows) · quality:
   <b style='color:{"#9c0006" if qof(r)=="needs-review" else "#b8860b"}'>{html.escape(qof(r) or "review")}</b> — Same table, continuing in this page order? &nbsp;
   <span style='font-weight:400;color:#789'>(scroll pages below ↓)</span></div>
 <div class=meta2>Table {html.escape(r['tnum'])} · {html.escape(str(r.get('region','')))} {html.escape(str(r.get('year','')))}
   · pages <b>{r['page_range'][0]}–{r['page_range'][1]}</b> ({r['n_pages']}p)
   · start <b style='color:{"#0a5d1e" if stype=="clean_start" else "#9c0006"}'>{html.escape(stype.replace("_"," "))}</b>
   · end <b>{html.escape(r.get("end_type","?").replace("_"," "))}</b></div>
 <input id=note placeholder="comment (e.g. 'missing start page', 'ends 1 page early', 'p51 not part of this')">
 <button class=ok onclick="v('correct')">✓ Correct [1]</button>
 <button class=bad onclick="v('wrong')">✗ Wrong [2]</button>
 <button class=bad style="background:#ffe0b2;color:#7a4a00" onclick="v('boundary')">⚠ Boundary off [4]</button>
 <button class=skip onclick="v('skip')">skip [3]</button>
 <button class=skip style="background:#cfd8ff" onclick="undo()">← Back [B]</button>
 <button class=skip style="background:#e0d4f5" onclick="rotAll()" title="rotate all scans 90° clockwise (key: R)">⟳ Rotate scans [R]</button>
 <a href=/ style="float:right;color:#789">home</a>
 <div style="margin-top:6px;padding-top:6px;border-top:1px dashed #cdd">
   <b style="color:#1f3864">Set exact bounds →</b>
   start <input id=bs type=number value={r['page_range'][0]} style="width:64px;margin-right:4px">
   end <input id=be type=number value={r['page_range'][1]} style="width:64px;margin-right:4px">
   <button onclick="setbounds()" style="background:#1f3864;color:#fff">⤢ Set &amp; rebuild [S]</button>
   <span id=bmsg style="color:#456;margin-left:10px;font-size:12px"></span>
   <span style="color:#9aa;font-size:11px;margin-left:6px">(rebuilds members from page order; any page not yet OCR'd is queued)</span>
 </div>
</div>
{flagbar}
<div class=runpages>{''.join(pages_html)}</div>
<div class=toggle onclick="var s=document.getElementById('stitch');s.style.display=s.style.display=='block'?'none':'block'">▸ toggle stitched-matrix preview (what the assembled table looks like)</div>
<div id=stitch>{stitched_matrix(r['members'])}</div>
<div id=lb class=fit><span id=lbhint>click = toggle fit/100% · R = rotate · Esc/click-out = close</span><button onclick="event.stopPropagation();rotLb()" style="position:fixed;top:8px;left:14px;z-index:102;padding:6px 13px;background:#1f3864;color:#fff;border:0;border-radius:6px;cursor:pointer;font-weight:600">⟳ rotate</button><img id=lbimg></div>
<script>
var fit=true; var lbrot=0; var throt=0;
function zoom(u){{var lb=document.getElementById('lb');var im=document.getElementById('lbimg');im.src=u;lbrot=0;im.style.transform='';lb.style.display='block';fit=true;lb.classList.add('fit');}}
function rotLb(){{lbrot=(lbrot+90)%360;document.getElementById('lbimg').style.transform='rotate('+lbrot+'deg)';}}
function rotAll(){{throt=(throt+90)%360;document.querySelectorAll('img.run').forEach(function(i){{i.style.transform='rotate('+throt+'deg)';}});}}
document.getElementById('lb').onclick=function(e){{
 if(e.target.id=='lbimg'){{fit=!fit;this.classList.toggle('fit',fit);}} else {{this.style.display='none';}}
}};
function v(verd){{fetch('/av',{{method:'POST',headers:{{'Content-Type':'application/json'}},
 body:JSON.stringify({{key:{json.dumps(key)},verdict:verd,note:document.getElementById('note').value}})}}).then(()=>location.reload());}}
function undo(){{fetch('/undo',{{method:'POST'}}).then(()=>location.reload());}}
function setbounds(){{
 var s=+document.getElementById('bs').value, e=+document.getElementById('be').value;
 var msg=document.getElementById('bmsg'); msg.textContent='setting…';
 fetch('/set_bounds',{{method:'POST',headers:{{'Content-Type':'application/json'}},
  body:JSON.stringify({{saoa:{json.dumps(str(r['saoa']))},orig:[{r['page_range'][0]},{r['page_range'][1]}],start:s,end:e}})}})
  .then(x=>x.json()).then(j=>{{
   if(!j.ok){{msg.textContent='✗ '+(j.err||'failed');return;}}
   var mp=(j.missing&&j.missing.length)?(' · '+j.missing.length+' page(s) queued for OCR: '+j.missing.join(',')):'';
   msg.textContent='✓ '+j.new[0]+'–'+j.new[1]+' ('+j.n_members+' pages'+mp+') — next run…';
   setTimeout(()=>location.reload(),1000);
  }});
}}
document.addEventListener('keydown',e=>{{if(e.target.tagName=='INPUT'){{if(e.key=='Enter'&&(e.target.id=='bs'||e.target.id=='be'))setbounds();return;}}
 if(e.key=='s'||e.key=='S'){{setbounds();return;}}
 if(e.key=='Escape'){{document.getElementById('lb').style.display='none';return;}}
 if(e.key=='r'||e.key=='R'){{if(document.getElementById('lb').style.display=='block')rotLb();else rotAll();return;}}
 if(document.getElementById('lb').style.display=='block')return;
 if(e.key=='1')v('correct');if(e.key=='2')v('wrong');if(e.key=='3')v('skip');if(e.key=='4')v('boundary');
 if(e.key=='b'||e.key=='B'||e.key=='Backspace')undo();}});</script>""",
                    mimetype="text/html")


@app.route("/av", methods=["POST"])
def av():
    j = request.get_json()
    v = _verd("assembly")
    v[j["key"]] = {"v": j["verdict"], "note": (j.get("note") or "").strip()}
    _save_verd("assembly", v)
    return jsonify(ok=True, n=len(v))


@app.route("/set_bounds", methods=["POST"])
def set_bounds_route():
    """Set a run's exact page-range, rebuild members, mark it boundary-fixed so it leaves the queue."""
    import boundary_apply
    j = request.get_json()
    res = boundary_apply.set_bounds(j["saoa"], j["orig"], j["start"], j["end"])
    if res.get("ok"):
        v = _verd("assembly")
        newkey = f"{res['saoa']}:{res['new']}"
        oldkey = f"{j['saoa']}:{[int(j['orig'][0]), int(j['orig'][1])]}"
        prev = (v.get(oldkey, {}) or {}).get("note", "")
        v[newkey] = {"v": "boundary", "human_bounds": True,
                     "note": (f"bounds {j['orig']}→{res['new']}" + (f" · {prev}" if prev else "")).strip()}
        _save_verd("assembly", v)
    return jsonify(**res)


@app.route("/include_page", methods=["POST"])
def include_page():
    """Pull ONE neighbor page into the current run's assembly: extend the bounds to cover it and rebuild
    members. If the page is already OCR'd it appears immediately; if not, kick off an on-demand Gemini
    OCR (renders from the drive, writes into the corpus) so it shows up once done."""
    import boundary_apply, subprocess
    j = request.get_json()
    saoa = str(j["saoa"]); s = int(j["start"]); e = int(j["end"]); pg = int(j["page"])
    res = boundary_apply.set_bounds(saoa, [s, e], min(s, pg), max(e, pg))
    if not res.get("ok"):
        return jsonify(**res)
    if res.get("missing"):
        log = open(os.path.join(ROOT, "fasrc/logs/grow_ocr.log"), "a")
        subprocess.Popen(["python3", os.path.join(ROOT, "grow_batch.py"), saoa],
                         stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
        res["ocr_started"] = True
        res["ocr_pages"] = res["missing"]
    else:
        res["included"] = True
    return jsonify(**res)


@app.route("/include_title", methods=["POST"])
def include_title():
    """Pull a page into the run's RANGE as a title/caption page — extends bounds for provenance but does
    NOT OCR it (empty placeholder; no rows, no constraints, never queued for OCR)."""
    import boundary_apply
    j = request.get_json()
    saoa = str(j["saoa"]); s = int(j["start"]); e = int(j["end"]); pg = int(j["page"])
    run = next((r for r in assembly_runs() if str(r["saoa"]) == saoa), None)
    if not run:
        return jsonify(ok=False, err="run not found")
    mk = boundary_apply.mark_title_page(run["dochash"], pg)
    if not mk.get("ok"):
        return jsonify(**mk)                       # e.g. already_data -> caller warns, leaves it alone
    res = boundary_apply.set_bounds(saoa, [s, e], min(s, pg), max(e, pg))
    # record the title page on the run (provenance)
    runs = json.load(open(os.path.join(ROOT, "export/tableVII_runs.json")))
    r2 = next((r for r in runs if str(r["saoa"]) == saoa), None)
    if r2 is not None:
        tp = set(r2.get("title_pages", [])); tp.add(pg); r2["title_pages"] = sorted(tp)
        json.dump(runs, open(os.path.join(ROOT, "export/tableVII_runs.json"), "w"), indent=0)
    res["title_page"] = pg
    res["image_cached"] = mk.get("image_cached", False)
    # OCR any real pages pulled in BETWEEN the title page and the old start (grow_ocr skips the title
    # placeholder since it counts as a member). This is what the title button was missing.
    if res.get("missing"):
        import subprocess
        subprocess.Popen(["python3", os.path.join(ROOT, "grow_batch.py"), saoa],
                         stdout=open(os.path.join(ROOT, "fasrc/logs/grow_ocr.log"), "a"),
                         stderr=subprocess.STDOUT, cwd=ROOT)
        res["ocr_started"] = True
        res["ocr_pages"] = res["missing"]
    return jsonify(**res)


@app.route("/cache_titles", methods=["POST", "GET"])
def cache_titles():
    """Render+cache the scan for every title page whose image isn't cached yet — run this when the drive
    is reconnected to recover title pages that were saved while it was disconnected."""
    import boundary_apply
    done = miss = 0
    for r in assembly_runs():
        for pg in r.get("title_pages", []):
            m, _ = boundary_apply.member_for(r["dochash"], pg)
            if not m:
                continue
            ih = m["table_id"].rsplit("/", 1)[1].replace("_oneshot", "")
            if os.path.exists(os.path.join(ROOT, "census_site/img", ih + ".jpg")):
                done += 1
            elif boundary_apply.cache_page_image(r["dochash"], pg, ih):
                done += 1
            else:
                miss += 1
    return jsonify(ok=True, cached=done, still_missing=miss)


@app.route("/grow_queue")
def grow_queue():
    """Live progress page for on-demand (grow) OCR: current job's per-page status + any runs still
    waiting on un-OCR'd pages. Auto-refreshes while a job is running."""
    running = bool(os.popen("pgrep -f 'grow_batch.py|grow_ocr.py' 2>/dev/null").read().strip())
    prog = None
    pf = os.path.join(ROOT, "export/grow_progress.json")
    if os.path.exists(pf):
        try:
            prog = json.load(open(pf))
        except Exception:
            prog = None
    pending = [(r["saoa"], r.get("region", ""), r["page_range"], r["missing_pages"])
               for r in assembly_runs() if r.get("missing_pages")]
    COL = {"done": "#0a5d1e", "ocring": "#1f3864", "failed": "#9c0006", "pending": "#9aa"}
    body = ""
    if prog:
        live = running and prog.get("state") == "running"
        state = ("🟢 <b>OCRing now</b>" if live else
                 ("✅ finished" if prog.get("state") == "done" else "idle"))

        def bar(done, total):
            total = total or 1; pct = int(100 * done / total)
            return (f"<div style='background:#e3e8f0;border-radius:8px;height:26px;width:520px;overflow:hidden;border:1px solid #ccd'>"
                    f"<div style='background:#0a5d1e;height:100%;width:{pct}%;transition:width .4s;color:#fff;"
                    f"text-align:center;line-height:26px;font-weight:700'>{done}/{total} ({pct}%)</div></div>")

        if "pages" in prog:                                   # old single-run per-page format
            pages = prog.get("pages", [])
            done = sum(1 for p in pages if p.get("status") == "done")
            chips = " ".join(
                f"<span style='display:inline-block;padding:3px 9px;margin:2px;border-radius:5px;color:#fff;"
                f"background:{COL.get(p.get('status'), '#9aa')}'>p{p.get('page')} {('⏳' if p.get('status')=='ocring' else '')}{p.get('status')}</span>"
                for p in pages)
            rng = prog.get("range", ["?", "?"])
            body += (f"<h3>grow-OCR — doc {html.escape(str(prog.get('saoa','?')))} pp{rng[0]}–{rng[1]} · {state}</h3>"
                     f"{bar(done, len(pages))}<div style='margin:10px 0'>{chips}</div>")
        else:                                                 # new parallel-batch format
            total = prog.get("total", 0); placed = prog.get("placed", 0); aff = prog.get("affected", 0)
            if prog.get("state") == "done":
                body += (f"<h3>Parallel grow-OCR batch (17 keys) · ✅ finished — placed {placed}/{total} reads "
                         f"across {aff} run(s)</h3>{bar(placed, total)}")
            else:
                body += (f"<h3>Parallel grow-OCR batch (17 keys) · 🟢 OCRing {total} page(s) across {aff} run(s)</h3>"
                         f"<div style='color:#456;margin:6px 0'>17 keys running in parallel — reads land in a burst at the "
                         f"end of each round. Finished runs appear below.</div>")
    if pending:
        body += "<h3>⏳ Runs still waiting on un-OCR'd pages</h3><ul>"
        for sa, reg, rng, mp in pending:
            body += f"<li>run <b>{sa}</b> {html.escape(str(reg))} pp{rng[0]}–{rng[1]} — pages needing OCR: <b>{mp}</b></li>"
        body += "</ul>"
    else:
        body += "<p style='color:#0a5d1e;font-weight:700;font-size:16px'>✅ No runs have pending un-OCR'd pages — everything you included is in.</p>"
    # ---- per-page re-OCR watcher (the '✗ re-OCR this page' flags) ----
    wrunning = bool(os.popen("pgrep -f 'reocr_watch.sh' 2>/dev/null").read().strip())
    wstate, wflag = "not running", 0
    sf = os.path.join(ROOT, "export/reocr_watch_status.json")
    if os.path.exists(sf):
        try:
            st = json.load(open(sf)); wstate = st.get("state", "?"); wflag = st.get("flagged", 0)
        except Exception:
            pass
    if not wrunning:
        wstate = "⚠ NOT RUNNING — flags won't auto-process"
    fc = 0
    ff = os.path.join(ROOT, "export/reocr_page_flags.txt")
    if os.path.exists(ff):
        fc = len([x for x in open(ff).read().split() if x.strip()])
    last = ""
    rf = os.path.join(ROOT, "export/reocr_batch_result.md")
    if os.path.exists(rf):
        last = "<pre style='background:#f0f3f8;padding:8px;border-radius:6px;font-size:12px;white-space:pre-wrap'>" + html.escape(open(rf).read()[:1500]) + "</pre>"
    badge = {"processing": "#0a5d1e", "settling": "#1f3864", "idle": "#789", "drive-down": "#9c0006"}.get(wstate, "#9c0006")
    # notes the reviewer attached when flagging (the agent acts on these)
    notes_html = ""
    nf2 = os.path.join(ROOT, "export/reocr_notes.json")
    if os.path.exists(nf2):
        try:
            nd = json.load(open(nf2)); pend = [(ih, n) for ih, n in nd.items() if not n.get("done")]
        except Exception:
            pend = []
        if pend:
            notes_html = ("<div style='margin:8px 0;padding:8px 12px;background:#fff7e6;border:1px solid #f0d8a0;border-radius:6px'>"
                          f"<b>📝 {len(pend)} flagged with a note for the agent to act on:</b><ul style='margin:4px 0'>"
                          + "".join(f"<li><code>{ih[:12]}</code> «{html.escape(n.get('note',''))}» <span style='color:#789'>{html.escape(n.get('title',''))}</span></li>"
                                    for ih, n in pend) + "</ul></div>")
    # runs that finished re-OCR, queued at the back for full-run review
    done_html = ""
    cf2 = os.path.join(ROOT, "export/reocr_completed_runs.json")
    if os.path.exists(cf2):
        try:
            dr = [v for v in json.load(open(cf2)).values() if not v.get("reviewed")]
        except Exception:
            dr = []
        if dr:
            done_html = ("<div style='margin:8px 0;padding:8px 12px;background:#e8f6ec;border:1px solid #9bd0a8;border-radius:6px'>"
                         f"<b>🔔 {len(dr)} run(s) finished re-OCR — at the BACK of the queue for full-run review:</b><ul style='margin:4px 0'>"
                         + "".join(f"<li>run <b>{html.escape(str(v.get('saoa')))}</b> pp{v.get('range',['?','?'])[0]}–{v.get('range',['?','?'])[1]} "
                                   f"· {v.get('fails')} fails · {html.escape(v.get('title',''))} <span style='color:#789'>({v.get('ts')})</span></li>"
                                   for v in dr[-30:]) + "</ul></div>")
    body += (f"<h3 style='margin-top:22px'>Per-page re-OCR (your “✗ re-OCR this page” flags)</h3>"
             f"<div>watcher: <b style='color:{badge}'>{html.escape(str(wstate))}</b> &middot; "
             f"<b>{fc}</b> page(s) flagged &amp; pending</div>{notes_html}{done_html}"
             f"<div style='color:#789;font-size:12px;margin:4px 0'>flagged pages re-OCR automatically within ~20–40s — "
             f"so they're done by the time you come back. Last batch:</div>{last or '<i>none yet</i>'}")
    refresh = "<meta http-equiv=refresh content=4>" if (running or wstate in ("processing", "settling") or fc) else ""
    return Response(f"""<!doctype html><meta charset=utf-8>{refresh}
<style>body{{font:14px/1.6 system-ui;margin:0;background:#f6f8fb;color:#1a2233}}
header{{background:#1f3864;color:#fff;padding:10px 18px}}header a{{color:#ffd966;text-decoration:none;margin-right:14px}}
.box{{padding:16px 22px}}h3{{color:#1f3864}}</style>
<header><a href=/tables>← tables</a><a href=/>home</a> <b>OCR queue / progress</b>
 {'<span style="color:#9fe6b0">● auto-refreshing</span>' if running else '<span style="color:#cdd">○ idle (no job running)</span>'}</header>
<div class=box>{body}
<div style='margin:16px 0'><button onclick="fetch('/cache_titles',{{method:'POST'}}).then(r=>r.json()).then(j=>alert('cached '+j.cached+' title-page image(s)'+(j.still_missing?(' · '+j.still_missing+' still missing — drive connected?'):' · all good')))" style='padding:7px 13px;background:#6b4ea0;color:#fff;border:0;border-radius:6px;cursor:pointer'>📑 cache title-page images (run after reconnecting the drive)</button></div>
<p style='color:#789;margin-top:18px;font-size:12px'>This page auto-refreshes every 4s while an OCR job is running. "missing pages" = a page is in a run's range but not yet OCR'd. Title pages render from a local cache so they show even when the drive is disconnected.</p></div>""",
                    mimetype="text/html")


@app.route("/reocr_done")
def reocr_done():
    """Runs that finished re-OCR and are queued (at the back) for full-run review — drives the /tables
    desktop alert + badge and the /grow_queue list."""
    cf = os.path.join(ROOT, "export/reocr_completed_runs.json")
    done = []
    if os.path.exists(cf):
        try:
            for v in json.load(open(cf)).values():
                if not v.get("reviewed"):
                    done.append({"saoa": v.get("saoa"), "range": v.get("range"), "ts": v.get("ts"),
                                 "fails": v.get("fails"), "title": v.get("title", "")})
        except Exception:
            pass
    return jsonify(n=len(done), runs=done[-30:])


@app.route("/reocr_done_seen", methods=["POST"])
def reocr_done_seen():
    """Mark all completed re-OCR runs as reviewed (clears the alert)."""
    cf = os.path.join(ROOT, "export/reocr_completed_runs.json")
    if os.path.exists(cf):
        try:
            d = json.load(open(cf))
            for v in d.values():
                v["reviewed"] = True
            json.dump(d, open(cf, "w"), indent=1)
        except Exception:
            pass
    return jsonify(ok=True)


@app.route("/grow_status", methods=["POST"])
def grow_status():
    """How many pages of a run are still un-OCR'd — the 'include page' poller reloads when it hits 0."""
    import boundary_apply
    saoa = str((request.get_json() or {}).get("saoa", ""))
    run = next((r for r in assembly_runs() if str(r["saoa"]) == saoa), None)
    if not run:
        return jsonify(done=True, missing=[])
    dh = run["dochash"]; s, e = run["page_range"]
    missing = [pg for pg in range(s, e + 1) if not boundary_apply.member_for(dh, pg)[1]]
    return jsonify(done=(len(missing) == 0), missing=missing)


@app.route("/undo", methods=["POST"])
def undo():
    """remove the most-recently-recorded assembly verdict so that run reappears for re-judging."""
    v = _verd("assembly")
    if v:
        last = list(v.keys())[-1]
        v.pop(last)
        _save_verd("assembly", v)
        return jsonify(ok=True, undone=last, n=len(v))
    return jsonify(ok=True, undone=None, n=0)


# ============================================================================
#  /documents  —  PER SOURCE-DOCUMENT view of the one Imperial Table VII and its
#  fan-out into districts / sub-areas (territory · religion · part · civil splits).
#  Top level = the source document (region + year); drill in = every sub-area we
#  can find, each with OCR/gap status, plus the surrounding pages (rendered from
#  the drive) so gaps & boundaries can be added.  Built from:
#     export/document_tableVII_index.csv   (one row per source document)
#     export/tableVII_subtables.csv        (one row per sub-area leaf, page-ordered)
#  Read-only on the assembly; boundary edits reuse the existing /assembly tooling.
# ============================================================================
_DOCIDX = _LEAFIDX = _DOCRUNS = None


def doc_index():
    global _DOCIDX, _LEAFIDX
    if _DOCIDX is None:
        p = os.path.join(ROOT, "export/document_tableVII_index.csv")
        l = os.path.join(ROOT, "export/tableVII_subtables.csv")
        _DOCIDX = list(csv.DictReader(open(p))) if os.path.exists(p) else []
        leaves = list(csv.DictReader(open(l))) if os.path.exists(l) else []
        _LEAFIDX = collections.defaultdict(list)
        for r in leaves:
            _LEAFIDX[r["dochash"]].append(r)
    return _DOCIDX, _LEAFIDX


def doc_runs():
    """dochash -> list of manual assembly runs (page_range, saoa) from tableVII_runs.json."""
    global _DOCRUNS
    if _DOCRUNS is None:
        _DOCRUNS = collections.defaultdict(list)
        rp = os.path.join(ROOT, "export/tableVII_runs.json")
        if os.path.exists(rp):
            for r in json.load(open(rp)):
                _DOCRUNS[r["dochash"]].append(r)
    return _DOCRUNS


def page_record(dh, pg):
    """(imghash, table_dir, title, data|None) for one page — data present iff an OCR read exists."""
    vp, _ = vol_index()
    rec = vp.get(dh, {}).get(pg)
    if not rec:
        return "", "", "", None
    ih, tdir, title = rec
    jf = os.path.join(CORPUS, "digitized_tables", tdir, ih + "_oneshot.json")
    data = None
    if os.path.exists(jf):
        try:
            data = json.load(open(jf))
        except Exception:
            data = None
    else:
        data, _, _ = load_table(ih)   # fallback: imghash glob
    return ih, tdir, title, data


UTYPE_BADGE = {
    "territory_leaf": ("#0a5d1e", "#c6efce", "DISTRICT/TALUK"),
    "city_leaf":      ("#7a4a00", "#ffe0b2", "CITY"),
    "district_rows":  ("#1f3864", "#dde7ee", "BY-DISTRICTS (rows)"),
    "religion_leaf":  ("#5b2a86", "#e7d6f5", "RELIGION"),
    "civil_leaf":     ("#8a6d00", "#fff3cd", "CIVIL-COND"),
    "part_leaf":      ("#234", "#e7f0ff", "PART"),
    "general":        ("#456", "#e9eef3", "GENERAL/SUMMARY"),
}


# ---- coverage over the WHOLE drive (all 427 volumes) ----
_COVERAGE = None


def doc_coverage():
    global _COVERAGE
    if _COVERAGE is None:
        p = os.path.join(ROOT, "export/document_coverage.csv")
        rows = list(csv.DictReader(open(p))) if os.path.exists(p) else []
        _COVERAGE = (rows, {r["saoa"]: r for r in rows})
    return _COVERAGE


def saoa_pdf_path(saoa):
    hits = glob.glob(os.path.join(DRIVE, "**", f"saoa.crl.{saoa}.pdf"), recursive=True)
    return hits[0] if hits else None


STAGE_BADGE = {
    "ASSEMBLED":   ("#0a5d1e", "#c6efce", "ASSEMBLED · VII OCR'd"),
    "PARTIAL":     ("#8a6d00", "#fff3cd", "PARTIAL · has gaps"),
    "LOCATED":     ("#1f3864", "#dde7ee", "LOCATED · not OCR'd"),
    "NEEDS_LOOK":  ("#7a4a00", "#ffe0b2", "NEEDS LOOK · scattered"),
    "NOT_LOCATED": ("#9c0006", "#ffd7d7", "NOT LOCATED · use TOC"),
    "NEEDS_OCR":   ("#5b2a86", "#e7d6f5", "NO TEXT · use TOC"),
}
STAGE_ORDER = ["PARTIAL", "LOCATED", "NEEDS_LOOK", "NOT_LOCATED", "NEEDS_OCR", "ASSEMBLED"]


@app.route("/render_saoa/<saoa>/<int:pg>")
def render_saoa(saoa, pg):
    """Render ANY page of ANY drive volume by saoa id (for browsing every document / its TOC)."""
    path = saoa_pdf_path(saoa)
    if not path:
        return ("PDF not on drive (mounted?)", 404)
    cache = os.path.join(ROOT, "export/context_renders", f"saoa_{saoa}_p{pg}.png")
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    if not os.path.exists(cache):
        try:
            import fitz
            doc = fitz.open(path)
            if 1 <= pg <= doc.page_count:
                doc.load_page(pg - 1).get_pixmap(matrix=fitz.Matrix(150 / 72, 150 / 72)).save(cache)
            doc.close()
        except Exception:
            return ("render failed", 404)
    if not os.path.exists(cache):
        return ("page out of range", 404)
    if request.args.get("thumb"):
        return send_file(thumb_of(cache, int(request.args.get("thumb") or 200)))
    return send_file(cache)


def browse_panel(saoa, cov):
    """TOC strip + candidate VII pages + a page jumper + confirm-range→queue-OCR — for ANY document,
    so the human can look at every volume and (if we missed the VII) find it via the printed TOC."""
    npages = int(cov.get("pages") or 0)
    cand = (cov.get("scan_candidate") or cov.get("vii_pages") or "").split(";")[0]
    cs = ce = ""
    if "-" in cand:
        a, b = cand.split("-")[:2]
        if a.isdigit() and b.isdigit():
            cs, ce = a, b
    toc_n = min(10, npages) if npages else 8
    strip = lambda pgs, lbl, hot: (
        f"<div style='font-size:11px;color:#456;margin:6px 18px 2px'><b>{lbl}</b></div>"
        f"<div class=runpages style='padding:0 18px'>" + "".join(
            f"<div class=pg><img class=run loading=lazy src='/render_saoa/{saoa}/{p}?thumb=240' "
            f"onclick=\"view({p})\" style='cursor:pointer;{'outline:3px solid #b8860b;' if hot else ''}'>"
            f"<br><b>p{p}</b></div>" for p in pgs) + "</div>")
    toc = strip(range(1, toc_n + 1), "① TOC / front matter (click a page to view; find “Table VII / Age” and its page no.)", False)
    candstrip = ""
    if cs:
        cps = list(range(int(cs), min(int(ce), int(cs) + 13) + 1))
        candstrip = strip(cps, f"② Candidate Table VII pages ({cand}) — verify these ARE Age/Sex/Civil Condition", True)
    # ---- sticky top control bar: viewer nav + set START/END from current page + range building + confirm ----
    bar = (
        "<div id=vbar style='position:sticky;top:0;z-index:30;background:#fff;border-bottom:2px solid #1f3864;"
        "padding:7px 18px;box-shadow:0 2px 6px rgba(0,0,0,.08)'>"
        "<div style='font-size:13px'>"
        f"<b style='color:#1f3864'>📄 PDF page <span id=vlabel style='font-size:16px'>1</span>{(' / ' + str(npages)) if npages else ''}</b> &nbsp;"
        "<button onclick='view(cur-1)'>◀ prev</button> <button onclick='view(cur+1)'>next ▶</button> "
        f"&nbsp;jump <input id=jp type=number min=1 max={npages or 9999} style='width:62px;padding:3px' "
        "onkeydown=\"if(event.key=='Enter')view(+this.value)\"> <button onclick='view(+document.getElementById(\"jp\").value)'>go</button>"
        "</div>"
        "<div style='margin-top:6px;font-size:13px'>"
        "<button class=ok onclick='setStart()' title='set range start = the page you are viewing'>⟖ START = current</button> "
        f"start <input id=cs type=number value='{cs}' style='width:58px;padding:4px'> &nbsp;"
        "<button class=ok onclick='setEnd()' title='set range end = the page you are viewing'>END = current ⟗</button> "
        f"end <input id=ce type=number value='{ce}' style='width:58px;padding:4px'> "
        "<button onclick='addBlock()' style='background:#dde7ee;color:#1f3864'>＋ add block</button> "
        f"&nbsp;ranges <input id=cranges value='{cand}' placeholder='26-92, 100-140' "
        "style='width:20%;padding:4px;border:1px solid #9bd;border-radius:6px;font-family:monospace'> "
        "<button onclick=\"document.getElementById('cranges').value=''\" style='background:#eee' title='clear ranges'>✕</button> "
        "<button class=ok onclick='confirmOcr()'>✓ Confirm &amp; queue OCR</button> "
        "<span id=cmsg style='color:#0a5d1e;margin-left:6px;font-size:12px'></span>"
        "</div></div>")
    viewer = ("<div style='text-align:center;margin:8px 0'>"
              "<div id=vcap style='font-weight:700;color:#1f3864;font-size:15px;margin-bottom:4px'>PDF page 1</div>"
              "<img id=viewer src='' style='max-width:82%;max-height:80vh;"
              "border:1px solid #ccd;border-radius:6px;cursor:zoom-in' onclick=\"zoom(this.src)\">"
              "<div style='color:#789;font-size:11px'>(click image to zoom)</div></div>")
    marks = (
        "<div class=btns style='position:static;box-shadow:none;border-top:1px dashed #cce;margin-top:8px'>"
        "<input id=cnote placeholder='note (e.g. “VII per TOC p.viii”, “no Table VII in this volume”)' "
        "style='width:36%;padding:5px;border:1px solid #cce;border-radius:6px;margin-right:8px'>"
        "<button class=skip onclick=\"markReviewed('no_vii')\">mark: no Table VII here</button> "
        "<button class=skip onclick=\"markReviewed('reviewed')\">mark reviewed</button></div>")
    panel = (
        f"<div style='border:2px solid #1f3864;border-radius:8px;margin:12px 18px;padding:0 0 10px;background:#fbfcff'>"
        f"{bar}"
        f"<div style='padding:8px 10px 0'>"
        f"<div style='font-weight:700;color:#1f3864;font-size:15px'>Find &amp; confirm this document's Table VII</div>"
        f"<div style='color:#456;font-size:12px;margin:2px 0 6px'>{npages} pages. Navigate to a page, click "
        f"<b>START = current</b> / <b>END = current</b> to autofill, <b>add block</b> (repeat for split Part I/II), then confirm. "
        f"Pages ±10 around the one you're viewing are pre-cached as you go.</div>"
        f"{viewer}{toc}{candstrip}{marks}</div></div>")
    js = ("<div id=lb><div id=lbcap style='position:fixed;top:10px;left:16px;color:#dfe8ff;font-size:16px;"
          "font-weight:700;z-index:101;background:rgba(20,30,60,.7);padding:3px 12px;border-radius:6px'></div>"
          "<img id=lbimg></div><script>"
          "var cur=1, SA='" + saoa + "', NP=" + str(npages or 0) + ", pf={};"
          "function pnum(p){return 'PDF page '+p+(NP?(' / '+NP):'');}"
          "function prefetch(p){for(var d=-10;d<=10;d++){var n=p+d;if(n<1||(NP&&n>NP)||pf[n])continue;"
          "pf[n]=1;var im=new Image();im.src='/render_saoa/'+SA+'/'+n;}}"
          "function view(p){if(!p||p<1)return;if(NP&&p>NP)return;cur=p;"
          "document.getElementById('viewer').src='/render_saoa/'+SA+'/'+p;"
          "document.getElementById('vlabel').textContent=p;"
          "document.getElementById('vcap').textContent=pnum(p);"
          "document.getElementById('jp').value=p;prefetch(p);}"
          "function setStart(){document.getElementById('cs').value=cur;"
          "document.getElementById('cmsg').textContent='start = p'+cur;}"
          "function setEnd(){document.getElementById('ce').value=cur;"
          "document.getElementById('cmsg').textContent='end = p'+cur;}"
          "function zoom(u){var l=document.getElementById('lb');document.getElementById('lbimg').src=u;"
          "var mm=(''+u).match(/\\/(\\d+)(?:\\?|$)/);"
          "document.getElementById('lbcap').textContent=mm?('PDF page '+mm[1]):'';l.style.display='block';}"
          "document.getElementById('lb').onclick=function(){this.style.display='none';};"
          "function addBlock(){var s=+document.getElementById('cs').value,e=+document.getElementById('ce').value;"
          "if(!s||!e||e<s){alert('set a valid start/end first (use START/END = current)');return;}"
          "var f=document.getElementById('cranges');f.value=(f.value?f.value.replace(/\\s*$/,'')+', ':'')+s+'-'+e;"
          "document.getElementById('cmsg').textContent='added block '+s+'-'+e;}"
          "function confirmOcr(){var spec=document.getElementById('cranges').value.trim();"
          "if(!spec){var s=+document.getElementById('cs').value,e=+document.getElementById('ce').value;"
          "if(!s||!e||e<s){alert('add at least one range');return;}spec=s+'-'+e;}"
          "document.getElementById('cmsg').textContent='queuing…';"
          "fetch('/doc_confirm_ocr',{method:'POST',headers:{'Content-Type':'application/json'},"
          "body:JSON.stringify({saoa:SA,ranges:spec,note:document.getElementById('cnote').value})})"
          ".then(x=>x.json()).then(j=>{var m=document.getElementById('cmsg');"
          "if(!j.ok){m.style.color='#9c0006';m.textContent='✗ '+(j.err||'failed');return;}"
          "if(j.drive_online===false){m.style.color='#9c0006';m.textContent='⚠ queued '+j.n_pages+' pp ['+j.ranges+'] — DRIVE NOT MOUNTED, will OCR once you remount it';return;}"
          "m.style.color='#0a5d1e';m.textContent='✓ queued '+j.n_pages+' pp ['+j.ranges+']'+(j.drainer_started?' · OCR started':' · added to queue');});}"
          "function markReviewed(v){fetch('/docv',{method:'POST',headers:{'Content-Type':'application/json'},"
          "body:JSON.stringify({dh:'saoa:'+SA,v:v,note:document.getElementById('cnote').value})}).then(()=>location.href='/documents');}"
          "view(1);</script>")
    return panel + js


@app.route("/documents")
def documents():
    docs, leaves = doc_index()
    cov_list, cov_by_saoa = doc_coverage()
    saoa = request.args.get("saoa")
    dh = request.args.get("dh")
    if dh and not saoa:  # legacy link → resolve to saoa
        saoa = next((r["saoa"] for r in cov_list if r.get("dochash") == dh), None)
    if not saoa:
        # ---- LIST: EVERY drive volume (all 427), grouped by stage, worst-first ----
        flt = request.args.get("f", "")
        vd = _verd("documents")
        counts = collections.Counter(r["stage"] for r in cov_list)
        rows = [r for r in cov_list if (not flt or r["stage"] == flt)]
        # WORTH-DOING first: unverified TABLES volumes (the imperial VII lives there) jump the queue;
        # already-reviewed docs sink. export/documents_priority.json is built from tables_volume_status.csv.
        _pr = {}
        _prp = os.path.join(ROOT, "export/documents_priority.json")
        if os.path.exists(_prp):
            try:
                _pr = json.load(open(_prp))
            except Exception:
                _pr = {}
        def _dprio(r):
            if vd.get(r.get("dochash", "")) or vd.get("saoa:" + r["saoa"]):
                return 8                                    # already reviewed -> bottom
            return _pr.get(r["saoa"], {}).get("rank", 5)
        # autobounds prefill hints (text-layer predictions; treat as 'look here', not gospel)
        _ab = {}
        _abp = os.path.join(ROOT, "export/vii_autobounds2_predictions.csv")
        if os.path.exists(_abp):
            try:
                _ab = {r["saoa"]: r["predicted"] for r in csv.DictReader(open(_abp)) if r.get("predicted")}
            except Exception:
                _ab = {}
        rows.sort(key=lambda r: (_dprio(r), STAGE_ORDER.index(r["stage"]) if r["stage"] in STAGE_ORDER else 9,
                                 -int(r["our_gaps"] or 0), r.get("year", "")))
        fbar = " · ".join(
            f"<a href='/documents?f={s}'>{s} ({counts.get(s,0)})</a>" for s in STAGE_ORDER)
        head = (f"<style>{CSS}</style><header><a href=/>home</a><a href=/tables>Tables</a>"
                f"<a href=/assembly>Assembly</a><b>Documents</b></header>"
                f"<div class=meta><b>Every source document on the drive ({len(cov_list)}).</b> "
                f"Table VII located for {counts['ASSEMBLED']+counts['PARTIAL']+counts['LOCATED']}. "
                f"Open any to browse its pages / TOC and confirm the Table VII. Filter: "
                f"<a href='/documents'>all</a> · {fbar}</div>")
        out = ["<table style='margin:10px 18px;font-size:13px'><tr>"
               "<th>stage</th><th>region / title</th><th>year</th><th>pages</th>"
               "<th>VII pages</th><th>sub-areas</th><th></th></tr>"]
        for r in rows:
            col, bg, lbl = STAGE_BADGE.get(r["stage"], ("#456", "#eee", r["stage"]))
            key = "saoa:" + r["saoa"]
            done = vd.get(r.get("dochash", ""), {}) or vd.get(key, {})
            chk = "✅" if done else ""
            label = r.get("region") or r.get("title") or ("saoa " + r["saoa"])
            _p = _pr.get(r["saoa"]) if not done else None
            worth = ""
            if _p:
                _rk = _p.get("rank", 5)
                _badge = ("🎯 DO THIS", "#0a5d1e") if _rk == 0 else \
                         (("⚑ ONLY SOURCE", "#8a6d00") if _rk == 2 else ("⛔ SKIP", "#8b8b8b"))
                worth = (f"<span style='background:{_badge[1]};color:#fff;padding:1px 7px;border-radius:4px;"
                         f"font-size:11px;font-weight:700' title='{html.escape(_p.get('why',''))}'>{_badge[0]}</span> ")
            _abhint = ("<br><span style='color:#6b4ea0;font-size:11px' title='autobounds suggestion (text-layer; verify)'>🤖 "
                       + html.escape(_ab[r["saoa"]][:26]) + "</span>") if (r["saoa"] in _ab and not done) else ""
            out.append(
                f"<tr><td>{worth}<span style='background:{bg};color:{col};padding:1px 7px;border-radius:4px;font-size:11px'>{lbl}</span></td>"
                f"<td><a href='/documents?saoa={r['saoa']}'><b>{html.escape(label[:44])}</b></a></td>"
                f"<td>{html.escape(r.get('year',''))}</td>"
                f"<td style='text-align:right'>{r.get('pages','')}</td>"
                f"<td style='text-align:right'>{html.escape(r.get('vii_pages','') or r.get('scan_candidate',''))}{_abhint}</td>"
                f"<td style='text-align:right'>{r.get('n_subareas','')}</td>"
                f"<td>{chk}</td></tr>")
        out.append("</table>")
        return Response(head + "".join(out), mimetype="text/html")

    # ---- DETAIL: one source document ----
    cov = cov_by_saoa.get(saoa)
    if not cov:
        return ("unknown document", 404)
    dh = cov.get("dochash") or dh
    d = next((x for x in docs if x["dochash"] == dh), None) if dh else None
    dleaves = leaves.get(dh, []) if dh else []
    runs = doc_runs().get(dh, []) if dh else []
    vd = _verd("documents")
    verd = vd.get(dh, {}) or vd.get("saoa:" + saoa, {})

    # completeness recomputed LIVE from the actual reads (matches what is shown)
    cards = []
    n_ocr = n_gap = 0
    for L in dleaves:
        pgs = [int(x) for x in L["pages"].split()] if L["pages"] else []
        col, bg, lbl = UTYPE_BADGE.get(L["unit_type"], UTYPE_BADGE["general"])
        pcells = []
        gapn = 0
        for pg in pgs:
            ih, tdir, title, data = page_record(dh, pg)
            ocrd = data is not None
            if ocrd:
                n_ocr += 1
            else:
                n_gap += 1
                gapn += 1
            mx = matrix_html(data) if ocrd else "<i style='color:#9c0006'>— GAP: page not yet OCR'd —</i>"
            tag = ("<span style='color:#0a5d1e'>✓ read</span>" if ocrd
                   else "<span style='color:#9c0006;font-weight:700'>✗ gap</span>")
            pcells.append(
                f"<div class=pg style='max-width:340px'>{page_img(dh, pg, ih, tdir)}"
                f"<br><b>p{pg}</b> {tag}<br><span style='color:#789;font-size:11px'>{html.escape((title or '')[:54])}</span>"
                f"<details><summary style='font-size:11px;color:#1f3864;cursor:pointer'>matrix</summary>{mx}</details></div>")
        cards.append(
            f"<div style='border:1px solid #cdd;border-radius:8px;margin:10px 18px;padding:8px;background:#fff'>"
            f"<div style='font-size:14px;font-weight:700;color:#1f3864'>"
            f"<span style='background:{bg};color:{col};padding:1px 8px;border-radius:4px;font-size:11px'>{lbl}</span> "
            f"{html.escape(L['subtable_label'])} "
            f"<span style='font-weight:400;color:#789;font-size:12px'>· pp {L['page_first']}–{L['page_last']} · "
            f"{L['n_ocrd']}/{L['n_pages']} read{' · ' + str(gapn) + ' gap' if gapn else ''}</span></div>"
            f"<div class=runpages>{''.join(pcells)}</div></div>")

    col, bg, lbl = STAGE_BADGE.get(cov["stage"], ("#456", "#eee", cov["stage"]))
    region = cov.get("region") or cov.get("title") or ("saoa " + saoa)
    _vr = verd.get("ranges") or ([("%d-%d" % tuple(verd["range"]))] if verd.get("range") else [])
    vbadge = (f"<span style='background:#c6efce;color:#0a5d1e;padding:2px 10px;border-radius:5px'>✓ {html.escape(verd.get('v',''))}"
              f"{(' pp ' + ', '.join(_vr)) if _vr else ''}</span>" if verd else "")
    # header + completeness (processed docs show OCR fan-out stats; all docs show stage + candidate)
    if d and dleaves:
        pct = int(round(100 * n_ocr / (n_ocr + n_gap))) if (n_ocr + n_gap) else 0
        runbar = " · ".join(f"<a href='/assembly'>{r['page_range'][0]}–{r['page_range'][1]}</a>" for r in runs) or "<i>none</i>"
        fanline = (f"Fanned out into <b>{len(dleaves)} sub-areas</b> · "
                   f"OCR'd <b style='color:{'#0a5d1e' if pct>=95 else '#9c0006'}'>{n_ocr}/{n_ocr+n_gap} ({pct}%)</b> · "
                   f"<b style='color:{'#9c0006' if n_gap else '#0a5d1e'}'>{n_gap} gaps</b> · assembly: {runbar} "
                   f"<span style='color:#789'>(edit in <a href=/assembly>Assembly</a>)</span>")
    else:
        fanline = (f"<b>Not yet organized here.</b> Text-layer candidate Table VII: "
                   f"<b>{html.escape(cov.get('scan_candidate') or cov.get('vii_pages') or '—')}</b> "
                   f"({html.escape(cov.get('scan_status',''))}). Browse below to confirm / find it.")
    head = (f"<style>{CSS}\n.run{{width:300px;height:200px;object-fit:contain;object-position:top;cursor:zoom-in}}"
            f"\n#lb{{display:none;position:fixed;inset:0;background:rgba(10,12,20,.94);z-index:100;overflow:auto;cursor:zoom-out}}"
            f"\n#lb img{{display:block;margin:14px auto;max-width:96vw}}</style>"
            f"<header><a href=/documents>◀ all documents</a>"
            f"<span style='background:{bg};color:{col};padding:1px 8px;border-radius:4px;font-size:12px'>{lbl}</span> "
            f"<b>{html.escape(region)} · {html.escape(cov.get('year',''))}</b> {vbadge}</header>"
            f"<div class=meta>{html.escape(cov.get('title',''))} · saoa <code>{saoa}</code> · {cov.get('pages','?')} pages<br>{fanline}</div>")
    # fan-out cards (processed docs) then the universal browse/find panel
    fanout_block = ("".join(cards)) if cards else ""
    if cards:
        fanout_block = "<div class=meta style='margin-top:8px'><b>Already-organized sub-areas</b> (OCR'd reads):</div>" + fanout_block
    panel = browse_panel(saoa, cov)
    return Response(head + panel + fanout_block, mimetype="text/html")


@app.route("/docv", methods=["POST"])
def docv():
    j = request.get_json()
    v = _verd("documents")
    v[j["dh"]] = {"v": j["v"], "note": (j.get("note") or "").strip()}
    _save_verd("documents", v)
    return jsonify(ok=True, n=len(v))


def parse_ranges(spec):
    """'26-92, 100-140, 178' -> sorted list of pages; supports single pages and a-b blocks."""
    pages = set(); norm = []
    for tok in re.split(r"[,\s;]+", str(spec or "").strip()):
        if not tok:
            continue
        m = re.match(r"^(\d+)\s*[-–]\s*(\d+)$", tok)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            if b >= a:
                pages.update(range(a, b + 1)); norm.append(f"{a}-{b}")
        elif tok.isdigit():
            pages.add(int(tok)); norm.append(tok)
    return sorted(pages), norm


@app.route("/doc_confirm_ocr", methods=["POST"])
def doc_confirm_ocr():
    """Human confirmed a document's Table VII page range(s) → record them and QUEUE those pages for OCR
    (rendered from the drive, Gemini). Supports MULTIPLE ranges (VII split into Part I / Part II / …).
    Append-only queue; a single memory-safe drainer sweeps it and skips already-OCR'd pages."""
    import subprocess
    j = request.get_json()
    saoa = str(j["saoa"])
    spec = j.get("ranges")
    if not spec and j.get("start") and j.get("end"):   # legacy single-range callers
        spec = f"{int(j['start'])}-{int(j['end'])}"
    pages, norm = parse_ranges(spec)
    if not pages:
        return jsonify(ok=False, err="no valid ranges")
    path = saoa_pdf_path(saoa) or ""
    # record the confirmed range(s) as this document's verdict
    v = _verd("documents")
    v["saoa:" + saoa] = {"v": "vii_confirmed", "ranges": norm, "note": (j.get("note") or "").strip()}
    _save_verd("documents", v)
    # append to the OCR queue (one entry per confirm; drainer dedups + skips already-OCR'd)
    q = os.path.join(ROOT, "export/documents_ocr_queue.jsonl")
    with open(q, "a") as f:
        f.write(json.dumps({"saoa": saoa, "path": path, "ranges": norm, "pages": pages}) + "\n")
    # drive must be mounted to render/OCR; if not, the range is safely queued and OCRs on remount
    drive_online = os.path.isdir(DRIVE)
    started = False
    drainer = os.path.join(ROOT, "build_documents_ocr.py")
    if drive_online and os.path.exists(drainer):
        running = subprocess.run(["pgrep", "-f", "build_documents_ocr.py"],
                                 capture_output=True).stdout.strip()
        if not running:
            os.makedirs(os.path.join(ROOT, "fasrc/logs"), exist_ok=True)
            subprocess.Popen(["python3", drainer],
                             stdout=open(os.path.join(ROOT, "fasrc/logs/documents_ocr.log"), "a"),
                             stderr=subprocess.STDOUT, cwd=ROOT)
            started = True
    return jsonify(ok=True, n_pages=len(pages), ranges=", ".join(norm),
                   drainer_started=started, drive_online=drive_online)


if __name__ == "__main__":
    print("Census review app -> http://localhost:5058  (Tables + Assembly + Documents)")
    app.run(port=5058, debug=False)
