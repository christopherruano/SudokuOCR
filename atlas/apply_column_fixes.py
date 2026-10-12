#!/usr/bin/env python3
"""Apply the column re-reads (atlas/out/reread/*.json, from reread_columns.py) to the corpus.

  accepted re-read  -> corrections overlay (sendout_v1/corrected_facts.csv + corrected_tables.json,
                       originals untouched on disk) and the same facts patched into the explorer DB;
                       pages the reading used (facing page / relinked page) are added to the
                       table's scans; review note records the correction
  rejected re-read  -> original kept; review note says the column check flagged it
  wrong scan, right page not found -> table withdrawn from the public site (static_exclude.txt;
                       listed in out/withdrawn_wrong_scan.csv)
Every decision is logged in out/column_fix_log.csv.

Idempotent: re-running replaces this script's own earlier corrections.
"""
import os, re, sys, csv, json, glob, sqlite3, hashlib, collections
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
ROOT = os.path.join(HERE, "..")
import reread_columns as R
import vote_readings as V_
import gemini_column_check as g
sys.path.insert(0, ROOT)
import build_query_dataset as B

V1 = os.path.join(ROOT, "export_final/sendout_final/sendout_v1")
DB = os.path.join(V1, "census_explorer.sqlite")
ADDED = os.path.join(V1, "added_pages"); os.makedirs(ADDED, exist_ok=True)
TAG = "column check 2026-10"
csv.field_size_limit(10 ** 8)


def add_page(pdf, page):
    """full-resolution page image saved as <pdf>_p<page>.png; returns its absolute path"""
    import fitz
    name = f"{os.path.basename(pdf).rsplit('.', 1)[0]}_p{page}.png"
    dest = os.path.join(ADDED, name)
    if not os.path.exists(dest):
        doc = fitz.open(pdf)
        doc.load_page(page - 1).get_pixmap(matrix=fitz.Matrix(150 / 72, 150 / 72), colorspace=fitz.csGRAY).save(dest)
    return os.path.abspath(dest)


SAFE_ONLY = False


def main(dry=False):
    con = sqlite3.connect(DB); con.row_factory = sqlite3.Row
    V = {}
    for l in open(os.path.join(HERE, "out", f"gemini_column_check_{R.MODEL}.jsonl")):
        d = json.loads(l)
        if d.get("verdict") != "error":
            V[d["page"]] = d
    tid_of = {hashlib.md5(t.encode()).hexdigest()[:7]: t for (t,) in con.execute("select table_id from tables")}
    res = {}
    for f in glob.glob(os.path.join(R.OUTD, "*.json")):
        try:
            d = json.load(open(f)); res[d["page"]] = d
        except Exception:
            continue
    ADJ = load_adjudications()
    for f in glob.glob(os.path.join(HERE, "out", "claude_adjudication", "batch_*.json")):
        for pk in json.load(open(f)):
            PACKETS[pk["page"]] = pk
    cleared = []
    log, new_facts, accepted, withdrawn = [], [], {}, []
    global ORIG
    ORIG = load_originals(con)
    notes, images = {}, {}
    for page, d in res.items():
        tid = d.get("table_id") or tid_of.get(page.rsplit("_", 1)[1])
        if not tid:
            continue
        t = dict(con.execute("select * from tables where table_id=?", (tid,)).fetchone())
        dn = {k: t.get(k, "") for k in B.DENORM if k != "status"}
        verdict = d.get("orig_verdict")
        if d.get("decision") in ("accept", "reject") and d.get("read"):
            old = [dict(r) for r in con.execute("select * from facts where table_id=?", (tid,))] if tid not in ORIG else ORIG[tid]
            ORIG[tid] = old
            gates, stats = R.regate(d, old, t["measure_class"])
            d["gates"], d["stats"] = gates, stats
            d["decision"] = "accept" if all(gates.values()) else "reject"
        persons_added = None
        if verdict == "persons_only_missing" and d.get("read"):
            # only individually verified Persons cells enter; the old M/F are kept unchanged, so the
            # whole-reading gates are irrelevant here
            facts, persons_added = persons_merge(ORIG[tid], d["read"], tid, t["measure_class"])
            ppath = os.path.join(HERE, "out", "reread_persons", page + ".json")
            pread = (json.load(open(ppath)).get("read") if os.path.exists(ppath) else None)
            if pread:                         # targeted Persons-only reading: add any further verified cells
                facts2, n2 = persons_merge(facts, pread, tid, t["measure_class"], all_persons=True)
                if n2:
                    facts, persons_added = facts2, persons_added + n2
            d["gates"] = {"persons_verified": persons_added > 0}
            d["decision"] = "accept" if persons_added > 0 else "reject"
        voted = None
        if not SAFE_ONLY and verdict != "persons_only_missing" and d.get("read"):
            # any structural re-read with an independent second reading is decided by the vote
            pp = os.path.join(HERE, "out", "reread_pro", page + ".json")
            prd = json.load(open(pp)) if os.path.exists(pp) else {}
            if not prd.get("read"):
                d = dict(d, decision="held")          # no second reading yet
            else:
                tp = os.path.join(HERE, "out", "reread_third", page + ".json")
                trd = json.load(open(tp)) if os.path.exists(tp) else {}
                if trd.get("read"):          # a third independent reading: majority of three per cell
                    cells, vst = V_.vote3(d["read"], prd["read"], trd["read"]); vst["readings"] = 3
                else:
                    cells, vst = V_.vote(d["read"], prd["read"])
                cov = (vst.get("agreed", 0) + vst.get("resolved", 0) - vst.get("withheld_by_arithmetic", 0)) / max(1, vst["flash_cells"])
                skip = set(V_.WITHHELD)
                vread = read_from_cells(cells)
                # overlap / size / column check judged on the full reading; the sum check on the
                # values actually published, skipping columns that have a withheld (blank) cell
                gates, stats = R.regate(d, ORIG[tid], t["measure_class"])
                gates["sums"] = not R.sum_failures(vread, skip)
                cols_withheld = 0
                if not gates["sums"]:
                    # withhold only the columns whose printed totals do not add up; publish the rest
                    badc = R.sum_failure_cols(vread, skip)
                    before = sum(1 for c in cells)
                    cells = [c for c in cells if (DASH.sub("-", c[0] or ""), DASH.sub("-", c[2] or ""), DASH.sub("-", c[3] or "")) not in
                             {(DASH.sub("-", a or ""), DASH.sub("-", b or ""), DASH.sub("-", c_ or "")) for a, b, c_ in badc}]
                    vread = read_from_cells(cells)
                    cols_withheld = len(badc)
                    cov = cov * (len(cells) / max(1, before))
                    gates["sums"] = not R.sum_failures(vread, skip) and cov >= 0.7
                    vst = dict(vst, columns_withheld_by_totals=cols_withheld)
                gates["two_readings_agree"] = cov >= (0.9 if not cols_withheld else 0.7)
                # with every published cell confirmed by two independent readings, agreement with
                # the OLD reading is no longer a digit-safety test (and the old reading is what was
                # broken); it is kept in the log only
                gates.pop("number_overlap", None); gates.pop("size", None)
                d = dict(d, read=vread, gates=gates, stats=dict(stats, **vst, coverage=round(cov, 3)),
                         decision="accept" if all(gates.values()) else "reject")
                voted = vst
        adj_note = None
        if not SAFE_ONLY and d.get("decision") != "accept" and page in ADJ and verdict != "wrong_scan":
            res_adj = adjudicate(d, ADJ[page], verdict, tid, t, page)
            if res_adj:
                kind_, payload, adj_note, adj_kind = res_adj
                if kind_ == "accept_facts":
                    facts, persons_added = payload, -1
                    d = dict(d, decision="accept", adjudication=adj_kind)
                elif kind_ == "accept_read":
                    d = dict(d, read=payload, decision="accept", adjudication=adj_kind)
                elif kind_ == "clear":
                    d = dict(d, decision="cleared", adjudication=adj_kind)
                else:
                    d = dict(d, decision="reject", adjudication=adj_kind)
        if SAFE_ONLY and verdict != "persons_only_missing":
            # structural fixes wait for the independent second reading and cell voting
            d = dict(d, decision="held")
        if d.get("decision") == "accept":
            if verdict != "persons_only_missing":
                facts = R.to_facts(d["read"], tid, dn, t["status"])
            for k, fr in enumerate(facts):
                fr["fact_id"] = f"Q{hashlib.md5(tid.encode()).hexdigest()[:8]}{k:06d}"
            new_facts += facts; accepted[tid] = len(facts)
            # scans: relinked page replaces the old scan; facing pages are appended
            extra = d.get("extra_pages") or []
            srcs = (t["member_images"] or t["source_image"] or "").split("|")
            srcs = [s for s in srcs if s.strip()]
            if d.get("relink"):
                srcs = [add_page(d["relink"]["pdf"], d["relink"]["pdf_page"])]
            else:
                first = re.search(r"([0-9a-f]{40})", srcs[0]) if srcs else None
                src = g.pdf_index().get(first.group(1)) if first else None
                for lab in extra:
                    if lab in ("F-1", "F+1") and src:
                        p = add_page(src[0], src[1] + (-1 if lab == "F-1" else 1))
                        srcs = [p] + srcs if lab == "F-1" else srcs + [p]
            images[tid] = "|".join(srcs)
            what = {"persons_only_missing": f"{persons_added} Persons/Total cells added (each verified against the existing Males and Females; no existing value changed)",
                    "columns_mislabelled": "column headers corrected",
                    "columns_missing": "missing columns added",
                    "columns_invented": "columns not printed on the page removed",
                    "columns_shifted": "column alignment corrected"}.get(verdict, "re-read from the correct page")
            if d.get("relink"):
                what = "linked to the correct scan page and re-read"
            elif extra:
                what += "; facing page attached"
            how = ""
            if voted:
                wh = voted.get("withheld", 0) + voted.get("withheld_by_arithmetic", 0) + voted.get("unmatched", 0)
                how = (("; every published cell was given by at least two of three independent readings (Gemini 3.8 Flash, "
                        "3.1 Pro and 3.7 Flash) or settled by the table's printed totals") if voted.get("readings") == 3 else
                       "; every published cell was read identically by two independent models (Gemini 3.8 Flash and "
                       "3.1 Pro) or settled by the table's printed totals" + (f"; {wh} cells on which they disagreed are left blank" if wh else "")
                       + (f"; {voted['columns_withheld_by_totals']} columns whose printed totals did not add up are left blank"
                          if voted.get("columns_withheld_by_totals") else ""))
            notes[tid] = f"Re-read after automated column check ({TAG}): {what}{how}."
            if d.get("adjudication"):
                notes[tid] = f"Corrected after automated column check ({TAG}) and review against the scan: {adj_note}."
        elif d.get("decision") == "cleared":
            cleared.append(tid)          # reviewer found the flag was a false alarm: no note
        elif d.get("decision") == "held":
            if verdict == "wrong_scan" or d.get("relinked"):
                withdrawn.append(tid)
            else:
                notes[tid] = (f"Automated column check ({TAG}) found a column problem ({verdict.replace('_', ' ')}); "
                              "a corrected reading is being verified. Use with care.")
        else:
            if verdict == "wrong_scan" or d.get("relinked"):
                withdrawn.append(tid)
            else:
                failed = [k for k, v in (d.get("gates") or {}).items() if not v]
                notes[tid] = (f"Automated column check ({TAG}) found a column problem ({verdict.replace('_', ' ')}); "
                              f"a corrected re-read did not pass verification ({', '.join(failed) or d.get('reason', 'error')}). Use with care.")
                if d.get("adjudication"):
                    notes[tid] = (f"Automated column check ({TAG}) found a column problem ({verdict.replace('_', ' ')}); "
                                  f"not corrected: {adj_note}. Use with care.")
        log.append({"page": page, "table_id": tid, "check_verdict": verdict, "decision": d.get("decision"),
                    "old_cells": d.get("old_cells"), "new_cells": d.get("new_cells"),
                    "number_recall": (d.get("stats") or {}).get("recall"), "number_precision": (d.get("stats") or {}).get("precision"),
                    "verify_verdict": (d.get("verify") or {}).get("verdict"),
                    "failed_gates": ";".join(k for k, v in (d.get("gates") or {}).items() if not v),
                    "relink": json.dumps(d.get("relink")) if d.get("relink") else "", "extra_pages": ";".join(d.get("extra_pages") or [])})
    # wrong scans with no re-read at all (right page not found)
    for page, d in V.items():
        if d["verdict"] == "wrong_scan" and page not in res:
            tid = tid_of.get(page.rsplit("_", 1)[1])
            if tid:
                withdrawn.append(tid)
                log.append({"page": page, "table_id": tid, "check_verdict": "wrong_scan", "decision": "withdrawn"})

    # ---- right pages located on the drive (find_pages.py) and confirmed by the column check ----
    fpc = os.path.join(HERE, "out", "found_pages_check.jsonl")
    fpp = {json.loads(l)["table_id"]: json.loads(l) for l in open(os.path.join(HERE, "out", "found_pages.jsonl"))} \
        if os.path.exists(fpc) else {}
    for l in (open(fpc) if fpc and os.path.exists(fpc) else []):
        d = json.loads(l); tid = d["table_id"]; r = fpp.get(tid)
        if not r or d["verdict"] not in ("ok", "persons_only_missing") or d.get("confidence") == "low" or tid in accepted:
            continue
        was = "withdrawn" if tid in withdrawn else "noted"
        withdrawn = [t for t in withdrawn if t != tid]
        images[tid] = ";".join(add_page(r["pdf"], p) for p in r["found_pages"])
        notes[tid] = (f"Scan relinked ({TAG}): the page(s) this table was read from were located in the source volume "
                      f"(pages {', '.join(map(str, r['found_pages']))}; {r['coverage']:.0%} of its numbers found there) "
                      "and its columns were confirmed against them"
                      + ("; the printed Persons column was not digitized" if d["verdict"] == "persons_only_missing" else "") + ".")
        log.append({"page": "", "table_id": tid, "check_verdict": d["verdict"], "decision": "relinked_" + was,
                    "relink": json.dumps({"pdf": os.path.basename(r["pdf"]), "pages": r["found_pages"]})})

    # ---- frozen rows / copied and serial-number columns (reviewed) ----
    FZ = frozen_fixes(con)
    fact_by_tid = collections.defaultdict(list)
    for f in new_facts:
        fact_by_tid[f["table_id"]].append(f)
    excl_now = set(withdrawn)
    for tid, fx in FZ.items():
        if tid in excl_now:
            continue
        base = fact_by_tid.get(tid) or [dict(r) for r in con.execute("select * from facts where table_id=?", (tid,))]
        if not base:
            continue
        kept, removed, restored = apply_frozen(base, fx)
        if not removed and not restored:
            continue
        t = dict(con.execute("select * from tables where table_id=?", (tid,)).fetchone())
        for k, fr in enumerate(kept):
            fr["fact_id"] = f"Q{hashlib.md5(tid.encode()).hexdigest()[:8]}{k:06d}"
        new_facts = [f for f in new_facts if f["table_id"] != tid] + kept
        accepted[tid] = len(kept)
        what = ("a serial-number column stored as data was removed" if fx.get("serial") and not fx["drop_rows"] else
                f"{removed} frozen or copied cells (one value repeated where the scan prints different values) were removed"
                + (f" and {restored} re-read cells verified by Persons = Males + Females restored" if restored else ""))
        notes[tid] = ((notes.get(tid, "") + " ") if tid in notes else "") + f"Reviewed against the scan ({TAG}): {what}."
        log.append({"page": "", "table_id": tid, "check_verdict": "frozen_or_copied", "decision": "accept",
                    "old_cells": len(base), "new_cells": len(kept)})
    if dry:
        with open(os.path.join(HERE, "out", "column_fix_log.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(log[0])); w.writeheader(); w.writerows(log)
        c = collections.Counter((r["check_verdict"], r["decision"]) for r in log)
        print(f"DRY RUN: would accept {len(accepted)} tables ({len(new_facts):,} cells); withdraw {len(set(withdrawn))}")
        for k, v in sorted(c.items()):
            print(f"  {k[0]:22} {k[1]:10} {v}")
        return accepted, set(withdrawn), notes
    # ---- overlay files (originals untouched) ----
    cf, ctj = os.path.join(V1, "corrected_facts.csv"), os.path.join(V1, "corrected_tables.json")
    prev = [r for r in csv.DictReader(open(cf)) if r["table_id"] not in accepted and not r["fact_id"].startswith("Q")] if os.path.exists(cf) else []
    prev_counts = (json.load(open(ctj)) if os.path.exists(ctj) else {}).get("counts", {})
    counts = {k: v for k, v in prev_counts.items() if k not in accepted}; counts.update(accepted)
    with open(cf, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=B.OUT_COLS, extrasaction="ignore"); w.writeheader()
        for r in prev + new_facts:
            w.writerow(r)
    json.dump({"corrected": list(counts), "counts": counts}, open(ctj, "w"))
    ovp = os.path.join(V1, "column_fix_overlay.json")
    prev_withdrawn = set(json.load(open(ovp)).get("withdrawn", [])) if os.path.exists(ovp) else set()
    json.dump({"images": images, "notes": notes, "withdrawn": sorted(set(withdrawn))}, open(ovp, "w"))
    # exclusions = the list as it was before this script + this run's withdrawals (a table this
    # script withdrew earlier and has now verified on its correct page comes back)
    ex = os.path.join(V1, "static_exclude.txt")
    basep = os.path.join(V1, "static_exclude_base.txt")     # the list before the column check
    base = {l.strip() for l in open(basep) if l.strip()} if os.path.exists(basep) else (
        {l.strip() for l in open(ex) if l.strip()} - prev_withdrawn)
    open(ex, "w").write("\n".join(sorted(base | set(withdrawn))) + "\n")

    # ---- patch the DB in place (same result as a full rebuild with the overlay) ----
    patch_db(con, accepted, new_facts, images, notes, cleared)
    with open(os.path.join(HERE, "out", "column_fix_log.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(log[0])); w.writeheader(); w.writerows(log)
    with open(os.path.join(HERE, "out", "withdrawn_wrong_scan.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["table_id"]); [w.writerow([t]) for t in sorted(set(withdrawn))]
    c = collections.Counter((r["check_verdict"], r["decision"]) for r in log)
    print(f"accepted {len(accepted)} tables ({len(new_facts):,} cells); withdrawn {len(set(withdrawn))}; notes {len(notes)}")
    for k, v in sorted(c.items()):
        print(f"  {k[0]:22} {k[1]:10} {v}")
    return accepted, set(withdrawn), notes


def load_originals(con):
    """original facts per table, before this script's corrections (from the backup copy of the
    pre-correction DB facts, saved on first run) so re-runs gate against the original reading"""
    p = os.path.join(HERE, "out", "column_fix_original_facts.jsonl")
    out = {}
    if os.path.exists(p):
        for l in open(p):
            d = json.loads(l); out[d["table_id"]] = d["facts"]
    return out


def save_originals(con, tids):
    p = os.path.join(HERE, "out", "column_fix_original_facts.jsonl")
    have = set(load_originals(con))
    with open(p, "a") as f:
        for t in tids:
            if t not in have:
                f.write(json.dumps({"table_id": t, "facts": [dict(r) for r in con.execute("select * from facts where table_id=?", (t,))]}) + "\n")


DASH = re.compile(r"[\u2012-\u2015\u2212]")


def load_adjudications():
    """reviewer (Claude) verdicts from reading the scans: out/claude_adjudication/verdicts_*.jsonl,
    minus verdicts set aside as unreliable (excluded_verdicts.json) or failed by the audit"""
    out = {}
    exf = os.path.join(HERE, "out", "claude_adjudication", "excluded_verdicts.json")
    excluded = set(json.load(open(exf))) if os.path.exists(exf) else set()
    for f in glob.glob(os.path.join(HERE, "out", "claude_adjudication", "verdicts_*.jsonl")):
        for l in open(f):
            try:
                d = json.loads(l)
            except Exception:
                continue
            if d.get("page") not in excluded:
                out[d["page"]] = d
    return out


def _num(x):
    try:
        return None if x is None else float(str(x).replace(",", ""))
    except ValueError:
        return None


def adjudicate(d, adj, verdict, tid, t, page):
    """apply a reviewer verdict to a table the machine readings could not settle.
    Returns (decision, read_or_facts, note_suffix, kind) or None."""
    if not adj:
        return None
    if adj.get("task") == "persons":
        if adj.get("persons_printed") is False:
            return None        # audit: such calls were contradicted 8/8; the flag stays
        rows = {}
        pk = PACKETS.get(page, {})
        for r in pk.get("persons_rows", []):
            rows[r["id"]] = r
        old = ORIG[tid]
        added = []
        n = lambda x: re.sub(r"[^a-z0-9]", "", (x or "").lower())
        for c in adj.get("persons") or []:
            r, P = rows.get(c.get("id")), _num(c.get("printed"))
            if r is None or P is None:
                continue
            same = [f for f in old if f["section"] == r["section"] and f["row_label_std"] == r["row"]
                    and f["column_group_std"] == r["column_group"] and f.get("value") is not None]
            M = [f for f in same if (f.get("sex") or "") == "Male"]; F = [f for f in same if (f.get("sex") or "") == "Female"]
            if len(M) == 1 and len(F) == 1 and abs(P - (float(M[0]["value"]) + float(F[0]["value"]))) <= 0.5:
                nf = dict(M[0]); nf.update(sub_column_std="persons", sex="All", value=P); added.append(nf)
        if not added:
            return None
        return ("accept_facts", [dict(f) for f in old] + added,
                f"{len(added)} Persons/Total cells added, read from the scan by a reviewer and each verified against the existing Males and Females; no existing value changed",
                "persons_reviewer")
    if not adj.get("columns_ok"):
        return ("keep", None, f"a reviewer found the column structure still wrong: {(adj.get('column_issue') or '')[:160]}", "columns_wrong")
    fr = d.get("read")
    if not fr:
        return None
    printed = {c["id"]: _num(c.get("printed")) for c in adj.get("cells") or []}
    if adj.get("task") == "spotcheck":
        fc = V_.cells_of(fr)
        byid = {f"{c['si']}.{c['ri']}.{c['ci']}": c for c in fc}
        checked = [(byid[i], v) for i, v in printed.items() if i in byid and v is not None]
        bad = [1 for c, v in checked if float(c["value"]) != v]
        if len(checked) < 20 or bad:
            return ("keep", None, f"a reviewer spot-check found {len(bad)} of {len(checked)} sampled cells differing from the scan", "spotcheck_failed")
        cells = [(c["sec"], c["row"], c["group"], c["sub"], c["value"], (c["si"], c["ri"])) for c in fc]
        return ("accept_read", read_from_cells(cells),
                f"single machine reading; its columns and a random sample of {len(checked)} cells were checked against the scan by a reviewer with no errors found",
                "spotcheck_ok")
    # disputes: machine-agreed cells + disputed cells whose printed value (reviewer) equals a machine reading
    pp = os.path.join(HERE, "out", "reread_pro", page + ".json")
    pr = json.load(open(pp)).get("read") if os.path.exists(pp) else None
    if not pr:
        return None
    tp = os.path.join(HERE, "out", "reread_third", page + ".json")
    tr = json.load(open(tp)).get("read") if os.path.exists(tp) else None
    fc, mp = V_._match(fr, pr)
    mt = V_._match(fr, tr)[1] if tr else {}
    cells, settled, unresolved = [], 0, 0
    for i, c in enumerate(fc):
        vals = [c["value"]] + [m[i]["value"] for m in (mp, mt) if i in m]
        v, k = collections.Counter(vals).most_common(1)[0]
        cid = f"{c['si']}.{c['ri']}.{c['ci']}"
        if k >= 2:
            cells.append((c["sec"], c["row"], c["group"], c["sub"], v, (c["si"], c["ri"])))
        elif cid in printed and printed[cid] is not None and printed[cid] in vals and V_._fits(fr, c, printed[cid]) is True:
            cells.append((c["sec"], c["row"], c["group"], c["sub"], printed[cid], (c["si"], c["ri"]))); settled += 1
        else:
            unresolved += 1
    if len(cells) < 0.9 * max(1, len(fc)):
        return ("keep", None, f"{unresolved} cells remain unresolved after review", "too_many_unresolved")
    vread = read_from_cells(cells)
    if R.sum_failures(vread):
        return ("keep", None, "printed totals still do not add up after review", "sums")
    return ("accept_read", vread,
            f"cells the machine readings disagreed on were read from the scan by a reviewer and kept only where they match a machine reading and the table's printed totals confirm them ({settled} settled, {unresolved} left blank)",
            "disputes_settled")


PACKETS = {}


def read_from_cells(cells):
    """rebuild a reading (sections > columns > rows) from voted cells in their original order.
    Rows are identified by position, never by label (labels repeat, e.g. one "Total" per
    district). Dash variants (0—10 / 0-10) are normalised. A section whose non-zero cells mostly
    (>= 90%, >= 10 cells) repeat a larger section's -- a page transcribed alone and again merged
    with its facing page -- is dropped; cells only it has are added to the larger section, and a
    cell the two copies disagree on is withheld."""
    D = lambda x: DASH.sub("-", x or "")
    cells = [(D(a), D(b), D(c), D(d), v, pos) for a, b, c, d, v, pos in cells]
    norm = lambda x: re.sub(r"\s+", "", (x or "").lower())
    sets = collections.defaultdict(set)
    for sec, row, grp, sub, v, pos in cells:
        if v not in (0, 0.0, None):          # zeros say nothing about two sections being copies
            sets[sec].add((norm(row), norm(grp), norm(sub), v))
    names = list(dict.fromkeys(c[0] for c in cells))
    into = {}
    for a in names:
        for b in names:
            if b != a and b not in into and len(sets[a]) >= 10 and len(sets[a]) <= len(sets[b]) \
                    and len(sets[a] & sets[b]) >= 0.9 * len(sets[a]) \
                    and (len(sets[a]) < len(sets[b]) or names.index(a) > names.index(b)):
                into[a] = b; break
    keep = [c for c in cells if c[0] not in into]
    if into:
        have = {}
        for c in keep:
            have.setdefault((c[0], norm(c[1]), norm(c[2]), norm(c[3])), []).append(c)
        for c in cells:
            if c[0] in into:
                k = (into[c[0]], norm(c[1]), norm(c[2]), norm(c[3]))
                if k not in have:                  # a cell only the duplicate has
                    keep.append((into[c[0]],) + c[1:]); have[k] = [keep[-1]]
                elif len(have[k]) == 1 and have[k][0][4] != c[4]:
                    keep.remove(have[k][0]); have[k] = []       # copies disagree: withheld
    secs = collections.OrderedDict()
    for sec, row, grp, sub, v, pos in keep:
        S = secs.setdefault(sec, {"cols": collections.OrderedDict(), "rows": collections.OrderedDict()})
        S["cols"].setdefault((grp, sub), None)
        r = S["rows"].setdefault(pos if sec not in into.values() or pos[0] == names.index(sec) else ("m",) + pos, {"label": row, "v": {}})
        r["v"][(grp, sub)] = v
    out = {"sections": []}
    for name, S in secs.items():
        cols = list(S["cols"])
        out["sections"].append({"name": name, "columns": [{"group": g, "sub": s} for g, s in cols],
                                "rows": [{"label": r["label"], "values": [r["v"].get(c) for c in cols]} for r in S["rows"].values()]})
    return out


def persons_merge(old, read, tid, measure, all_persons=False):
    """Persons-only fix that never changes a previously read number: keep every old fact and add
    a Persons cell only where it is verified -- count tables: new Persons == old Males + old
    Females (within 0.5); other tables: the re-read's Males and Females for that row/column
    equal the old ones exactly (so the row/column alignment of the new Persons is proven)."""
    n = lambda x: re.sub(r"[^a-z0-9]", "", (x or "").lower())
    key = lambda f: (n(f["row_label_std"]), n((f["column_group_std"] or "").split(" / ")[-1]))
    def sx(f):
        return f.get("sex") if f.get("sex") in ("Male", "Female", "All") else \
            R.sex_of(f.get("sub_column_std"), f.get("column_group_std"), f.get("section"), f.get("row_label_std"))
    def key2(f):           # the sex word removed from the key, wherever it sits
        strip = lambda x: re.sub(r"(?i)\b(fe)?males?\b|\bpersons\b|both sexes|\btotal\b", "", x or "")
        return (n(strip(f.get("section"))), n(strip(f["row_label_std"])), n(strip((f["column_group_std"] or "").split(" / ")[-1])))
    oldk = collections.defaultdict(dict)
    for f in old:
        if sx(f) in ("Male", "Female") and f["value"] is not None:
            oldk[key(f)].setdefault(sx(f), f); oldk[("2",) + key2(f)].setdefault(sx(f), f)
    newk = collections.defaultdict(dict)
    for f in R.to_facts(read, tid, {}, ""):
        if all_persons:      # a Persons-only reading: every column is a Persons/Total column ("P", "Total")
            f["sex"] = "All"
        newk[key(f)].setdefault(f["sex"], f); newk[("2",) + key2(f)].setdefault(f["sex"], f)
    added = []
    for k, d in newk.items():
        if "All" not in d or k not in oldk or not {"Male", "Female"} <= set(oldk[k]):
            continue
        if any(sx(f) == "All" and (key(f) == k or ("2",) + key2(f) == k) for f in old):
            continue
        if k[0] == "2" and any(a.get("sex") == "All" and (("2",) + key2(a)) == k for a in added):
            continue
        P, M, F = float(d["All"]["value"]), float(oldk[k]["Male"]["value"]), float(oldk[k]["Female"]["value"])
        if measure == "count":
            ok = abs(P - (M + F)) <= 0.5
        else:
            ok = ("Male" in d and "Female" in d and float(d["Male"]["value"]) == M and float(d["Female"]["value"]) == F)
        if ok:
            f = dict(oldk[k]["Male"])        # table metadata from the existing cell, labels from the Persons cell read
            if k[0] == "2":
                f.update(section=d["All"]["section"], row_label_std=d["All"]["row_label_std"],
                         column_group_std=d["All"]["column_group_std"])
            f.update(sub_column_std=d["All"]["sub_column_std"] or "persons", sex="All", value=P)
            added.append(f)
    return [dict(f) for f in old] + added, len(added)


def frozen_fixes(con):
    """Reviewed frozen rows / copied or serial-number columns (out/frozen_review/verdicts_*.jsonl and a
    corpus scan for serial-number columns). Returns {table_id: (drop predicate list, verified cells,
    note)} and writes out/frozen_exceptions.csv (rows/columns a reviewer confirmed are printed so)."""
    fixes = collections.defaultdict(lambda: {"drop_rows": set(), "drop_cols": set(), "add": [], "n": 0})
    exc = []
    n = lambda x: re.sub(r"[^a-z0-9]", "", (x or "").lower())
    det = {}
    for r in csv.DictReader(open(os.path.join(HERE, "out", "column_audit_detail.csv"))):
        det.setdefault(r["page"], []).append(r)
    pk = {}
    for f in glob.glob(os.path.join(HERE, "out", "frozen_review", "batch_*.json")):
        for p_ in json.load(open(f)):
            pk[p_["page"]] = p_
    for f in glob.glob(os.path.join(HERE, "out", "frozen_review", "verdicts_*.jsonl")):
        for l in open(f):
            try:
                v = json.loads(l)
            except Exception:
                continue
            P = pk.get(v["page"])
            if not P:
                continue
            iss = next((i for i in P["issues"] if i["id"] == v.get("id")), None)
            if not iss:
                continue
            label = iss.get("row") if iss["type"] == "frozen_row" else iss.get("column")
            if v.get("verdict") == "genuine":
                exc.append({"page": v["page"], "kind": iss["type"], "label": label, "note": (v.get("note") or "")[:200]})
                continue
            fx = fixes[P["table_id"]]
            if iss["type"] == "frozen_row":
                fx["drop_rows"].add(n(label))
                for c in v.get("cells") or []:
                    if c.get("printed") is not None:
                        fx["add"].append(("row", label, c.get("section"), c.get("column"), c["printed"]))
            else:
                fx["drop_cols"].add(n(label.replace(" | ", "")))
    # serial-number columns stored as data (row indices 1..n), corpus-wide
    for t, g_, s_, vals in con.execute("""select table_id, column_group_std, sub_column_std, group_concat(value) from facts
            where lower(column_group_std) in ('age','district','serial','serial no','serial number','sl. no.','no.','value')
               or lower(sub_column_std) in ('age','district') group by 1,2,3"""):
        vs = [float(x) for x in (vals or "").split(",") if x not in ("", None)]
        if len(vs) >= 3 and all(x.is_integer() for x in vs) and sorted(set(vs)) == [float(i) for i in range(int(min(vs)), int(min(vs)) + len(set(vs)))]:
            fixes[t]["drop_cols"].add(n((g_ or "") + (s_ or ""))); fixes[t]["serial"] = True
    with open(os.path.join(HERE, "out", "frozen_exceptions.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["page", "kind", "label", "note"]); w.writeheader(); w.writerows(exc)
        man = os.path.join(HERE, "out", "frozen_exceptions_manual.csv")   # confirmed by eye against the scan
        if os.path.exists(man):
            w.writerows(csv.DictReader(open(man)))
    return fixes


def apply_frozen(facts, fx):
    """remove frozen rows / copied columns; restore a reviewer-read cell only when its row then satisfies
    Persons = Males + Females exactly"""
    n = lambda x: re.sub(r"[^a-z0-9]", "", (x or "").lower())
    def colkey(f):
        return n((f.get("section") or "") + (f.get("column_group_std") or "") + (f.get("sub_column_std") or "")), \
               n((f.get("column_group_std") or "") + (f.get("sub_column_std") or ""))
    keep, dropped = [], []
    for f in facts:
        ck = colkey(f)
        if n(f.get("row_label_std")) in fx["drop_rows"] or ck[0] in fx["drop_cols"] or ck[1] in fx["drop_cols"]:
            dropped.append(f)
        else:
            keep.append(f)
    restored = 0
    by = collections.defaultdict(dict)
    for f in dropped:
        if n(f.get("row_label_std")) in fx["drop_rows"]:
            by[(f.get("section"), f.get("row_label_std"), f.get("column_group_std"))][f.get("sex") or ""] = f
    rv = {}
    for kind, row, sec, col, printed in fx["add"]:
        rv[(n(row), n(sec), n(col))] = float(str(printed).replace(",", ""))
    for (sec, row, grp), d in by.items():
        vals = {}
        for sx, f in d.items():
            k = (n(row), n(sec), n((f.get("column_group_std") or "") + " | " + (f.get("sub_column_std") or "")))
            k2 = (n(row), n((sec or "")[:50]), n((f.get("column_group_std") or "") + " | " + (f.get("sub_column_std") or "")))
            if k in rv or k2 in rv:
                vals[sx] = (f, rv.get(k, rv.get(k2)))
        if {"All", "Male", "Female"} <= set(vals) and abs(vals["All"][1] - vals["Male"][1] - vals["Female"][1]) <= 0.5:
            for sx in ("All", "Male", "Female"):
                f, v = vals[sx]; keep.append(dict(f, value=v)); restored += 1
    return keep, len(dropped) - restored, restored


def patch_db(con, accepted, new_facts, images, notes, cleared=()):
    save_originals(con, accepted)
    for tid in cleared:
        old = con.execute("select review_note from tables where table_id=?", (tid,)).fetchone()[0] or ""
        con.execute("update tables set review_note=? where table_id=?",
                    (re.sub(r"\s*\|?\s*(Re-read after|Corrected after|Automated column check)[^|]*\(" + re.escape(TAG) + r"\)[^|]*", "", old).strip(" |"), tid))
    cols = [r[1] for r in con.execute("pragma table_info(facts)")]
    con.executemany("delete from facts where table_id=?", [(t,) for t in accepted])
    num = {"value": float, "resolved_age_lower": int, "resolved_age_upper": int, "year": int}

    def cast(c, v):
        if c in num:
            try:
                return num[c](float(v)) if v not in ("", None) else None
            except (TypeError, ValueError):
                return None
        return v
    con.executemany(f"insert into facts ({','.join(cols)}) values ({','.join('?' * len(cols))})",
                    [[cast(c, f.get(c, "")) for c in cols] for f in new_facts])
    for tid, n in accepted.items():
        con.execute("update tables set n_facts=? where table_id=?", (n, tid))
    for tid, im in images.items():
        con.execute("update tables set member_images=? where table_id=?", (im, tid))
    for tid, note in notes.items():
        old = con.execute("select review_note from tables where table_id=?", (tid,)).fetchone()[0] or ""
        old = re.sub(r"\s*(Re-read after automated column check|Corrected after automated column check|Automated column check|Scan relinked) \(" + re.escape(TAG) + r"\)[^|]*", "", old).strip(" |")
        con.execute("update tables set review_note=? where table_id=?", ((old + " | " if old else "") + note, tid))
    con.commit()


if __name__ == "__main__":
    SAFE_ONLY = "--safe" in sys.argv
    main(dry="--dry" in sys.argv)
