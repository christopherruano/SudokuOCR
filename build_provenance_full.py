#!/usr/bin/env python3
"""FULL provenance for every Table VII read — connects the chain image -> OCR -> assembly -> stitching and
classifies WHERE each table sits on its path to perfection (with the concrete next action to get it there).
Joins (all on imghash):
  page_number_status.csv  -> source lineage (source_pdf, pdf_page, dochash) + assembly membership
  the corpus oneshot JSON -> engine, #rows, column groups, half-type, human_corrected/stitched/source_error flags
  tableVII_verdicts.csv   -> perfect / retrying / unrecoverable + reason + last_issues
  first_cut/catalog.csv   -> the parent run's honest quality label
  severity_tables.json    -> failure severity (magnitude + count)
  constraints_irrelevant.json / stitch_lefts|rights.json
Writes:
  export/provenance_full.csv      one row per Table VII read, the whole lineage + stage + next_action
  export/provenance_summary.csv   counts by stage (the path-to-perfection funnel)
  python3 build_provenance_full.py
"""
import sys, os, json, csv, re, glob
sys.argv = ["x"]
import census_review_app as A
import constraints_core

ROOT = os.path.dirname(os.path.abspath(__file__))
V1 = os.path.join(ROOT, "export_final/sendout_final/sendout_v1")
CORPUS = os.path.join(ROOT, "export/ocr_results_referenced/digitized_tables")
IH_RE = re.compile(r"([0-9a-f]{40})")


def imof(s):
    m = IH_RE.search(str(s) or "")
    return m.group(1) if m else None


def half_type(groups):
    g = {x.lower() for x in groups}
    has_unm = any("unmarr" in x for x in g)
    has_wid = any("widow" in x for x in g)
    has_civ = any(("marr" in x or "widow" in x) for x in g)   # any civil-condition wing
    has_tot = any(("total" in x or "popul" in x) for x in g)
    if not has_civ:
        # no civil-condition wings: judge by the COLUMN age span (transposed layouts).
        # Full span 0 -> old ages/Total = self-complete table, NOT a half —
        # a bare 'Total' group must not flag it.
        lo = hi = None
        n_age = 0
        for x in groups:
            if len(str(x)) > 30:              # long phrase (rate/description), not an age band
                continue
            sp = constraints_core.parse_age_span(x)
            if isinstance(sp, tuple) and isinstance(sp[0], int):
                if sp[0] > 130:               # calendar years (1881, 1891, ...), not ages
                    continue
                n_age += 1
                lo = sp[0] if lo is None else min(lo, sp[0])
                h = sp[0] if sp[1] is None else min(sp[1], 10**6)
                hi = h if hi is None else max(hi, h)
        if lo is None or n_age < 2:
            return "full"                     # no real age-column axis -> nothing to stitch
        if lo <= 1 and (has_tot or hi >= 55):
            return "full"                     # complete age run -> self-complete
        if lo >= 5:
            return "RIGHT_half"               # columns start mid-range: continuation half
        if hi < 55 and not has_tot:
            return "LEFT_half"                # truncated before the old ages/Total
        return "full"
    if (has_tot or has_unm) and not has_wid:
        return "LEFT_half"
    if has_wid and not has_tot and not has_unm:
        return "RIGHT_half"
    return "full"


def classify(rec):
    """Return (stage, next_action) — where this read is on the path to perfection.
    CONSTRAINTS_N/A now comes from the checker's own verdict — the table's KIND is not checkable
    (rate/ratio/index/text: no arithmetic identity exists) or the reviewer pressed Constraints-N/A —
    instead of the old RACE_RE title guess. Checkable kinds (count/distribution/percentage) are
    actually validated, and the arithmetic verdict is constraints_core's kind-gated fail count."""
    iss = (rec["last_issues"] + " " + rec["verdict_reason"]).lower()
    if rec["n_rows"] == 0:
        return "EMPTY_TITLE", "non-data page (title / blank / 'no table') — not a data table"
    if rec["constraints_irrelevant"] or rec["report_status"] == "NOT_CHECKABLE":
        return "CONSTRAINTS_N/A", f"none — {rec['kind']} table, no arithmetic identity applies (correct as-is)"
    if rec["stitched"]:
        return "RECOVERED_stitch", "done — recovered by left/right stitch"
    if rec["human_corrected"]:
        return "HUMAN_VERIFIED", "done — hand-corrected / source-error accepted"
    if rec["verdict_status"] == "perfect" and rec["core_fails"] == 0:
        return "PERFECT", "done"
    if rec["half_type"] in ("LEFT_half", "RIGHT_half"):
        return "NEEDS_STITCH", "OCR the facing half-page, then stitch (stitch_pilot.py)"
    if rec["core_fails"] == 0 and rec["verdict_status"] != "unrecoverable":
        return "PERFECT", "done"
    if "missing_cols" in iss or "dropped_subcols" in iss:
        return "MISTRAL_RECOVER", "Mistral re-OCR — columns present on page but dropped"
    if rec["verdict_status"] == "retrying" or (0 < rec["core_fails"] <= 4):
        return "RE-OCR_MINOR", "constraint-guided re-OCR / quick digit fix"
    if "age bands not present" in iss or "source scan incomplete" in iss or "few_age_bands" in iss:
        return "SOURCE_LIMITED?", "verify scan — likely truncated source OR a spread (check facing page)"
    if "identical defects" in iss or "plateaued" in iss:
        return "MISTRAL_RECOVER", "Mistral re-OCR — Gemini plateaued"
    return "NEEDS_REVIEW", "manual review (bigger discrepancies)"


def main():
    # page master -> lineage by imghash
    pm = {}
    for r in csv.DictReader(open(os.path.join(V1, "page_number_status.csv"))):
        pm[r["imghash"]] = r
    # verdicts by imghash
    verd = {}
    for r in csv.DictReader(open(os.path.join(V1, "tableVII_verdicts.csv"))):
        ih = imof(r["table_id"])
        if ih:
            verd[ih] = r
    # run quality by (saoa, range) from first-cut
    runq = {}
    for r in csv.DictReader(open(os.path.join(ROOT, "export/first_cut/catalog.csv"))):
        runq[(r["doc"], r["page_range"])] = r["quality"]
    sev = json.load(open(os.path.join(ROOT, "export/severity_tables.json"))) if os.path.exists(os.path.join(ROOT, "export/severity_tables.json")) else {}
    sev_by_ih = {imof(k): v for k, v in sev.items() if imof(k)}
    na = set(json.load(open(os.path.join(ROOT, "export/constraints_irrelevant.json")))) if os.path.exists(os.path.join(ROOT, "export/constraints_irrelevant.json")) else set()

    rows = []
    for f in glob.glob(os.path.join(CORPUS, "*", "*_oneshot.json")):
        td = os.path.basename(os.path.dirname(f)).upper()
        if not (("VII" in td or "AGE-SEX" in td) and "VIII" not in td and "XVII" not in td):
            continue
        ih = imof(os.path.basename(f))
        if not ih:
            continue
        try:
            d = json.load(open(f))
        except Exception:
            continue
        meta = d.get("metadata", {}) or {}
        groups = {g for s in d.get("data", []) for row in s.get("rows", []) for g, vv in row.items() if isinstance(vv, dict)}
        nrows = sum(len(s.get("rows", [])) for s in d.get("data", []))
        p = pm.get(ih, {})
        vd = verd.get(ih, {})
        saoa = p.get("assembly_run", "") or vd.get("table_id", "")
        rng = p.get("run_range", "")
        sv = sev_by_ih.get(ih, {})
        rep = constraints_core.check_table(d)        # kind-gated arithmetic verdict (authoritative)
        core_fails = A.table_fail_rows(d)            # rep.n_fail_rows + missing-column defect rows
        rec = {
            # IMAGE
            "imghash": ih, "dochash": p.get("dochash", ""), "source_pdf": p.get("source_pdf", ""),
            "pdf_page": p.get("pdf_page", ""),
            # OCR
            "table_dir": os.path.basename(os.path.dirname(f)), "engine": d.get("extraction_method", ""),
            "n_rows": nrows, "groups": "|".join(sorted(groups))[:80], "title": (meta.get("title", "") or "")[:70],
            "region": meta.get("region", ""), "year": meta.get("year", ""),
            "human_corrected": bool(meta.get("human_corrected")), "source_error": bool(meta.get("source_error")),
            "constraints_irrelevant": bool(meta.get("constraints_irrelevant")) or ih in na,
            "stitched": bool(meta.get("stitched_halfpage")),
            "half_type": half_type(groups),
            # ASSEMBLY
            "assembly_run": p.get("assembly_run", ""), "run_range": rng, "run_position": p.get("run_position", ""),
            "run_quality": runq.get((str(p.get("assembly_run", "")), rng), ""),
            "title_role": p.get("title_role", ""),
            # STATUS
            "verdict_status": vd.get("status", ""), "verdict_reason": (vd.get("reason", "") or "")[:60],
            "last_issues": (vd.get("last_issues", "") or "")[:50],
            "n_fails": sv.get("fails", sv.get("n", 0)), "sev": sv.get("sev", 0),
            # CONSTRAINT ENGINE (constraints_core — kind- and layout-aware)
            "kind": rep.kind, "report_status": rep.status,
            "coverage": round(rep.coverage, 3), "core_fails": core_fails,
        }
        rec["stage"], rec["next_action"] = classify(rec)
        rows.append(rec)

    cols = ["imghash", "dochash", "source_pdf", "pdf_page", "table_dir", "engine", "title", "region", "year",
            "n_rows", "groups", "half_type", "human_corrected", "source_error", "constraints_irrelevant",
            "stitched", "assembly_run", "run_range", "run_position", "run_quality", "title_role",
            "verdict_status", "verdict_reason", "last_issues", "n_fails", "sev",
            "kind", "report_status", "coverage", "core_fails", "stage", "next_action"]
    out = os.path.join(ROOT, "export/provenance_full.csv")
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols); w.writeheader()
        for r in rows:
            w.writerow(r)

    from collections import Counter
    stages = Counter(r["stage"] for r in rows)
    ORDER = ["PERFECT", "HUMAN_VERIFIED", "RECOVERED_stitch", "CONSTRAINTS_N/A",
             "RE-OCR_MINOR", "NEEDS_STITCH", "MISTRAL_RECOVER", "NEEDS_REVIEW", "SOURCE_LIMITED?", "EMPTY_TITLE"]
    DONE = ("PERFECT", "HUMAN_VERIFIED", "RECOVERED_stitch", "CONSTRAINTS_N/A")
    with open(os.path.join(ROOT, "export/provenance_summary.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["stage", "count", "meaning"])
        meaning = {"PERFECT": "passes all checks", "HUMAN_VERIFIED": "you fixed/accepted it",
                   "RECOVERED_stitch": "recovered by left/right stitch", "CONSTRAINTS_N/A": "identities don't apply (correct)",
                   "RE-OCR_MINOR": "1 re-OCR / quick digit fix away", "NEEDS_STITCH": "OCR facing half + stitch",
                   "MISTRAL_RECOVER": "Mistral re-OCR (dropped/plateaued)", "NEEDS_REVIEW": "manual review",
                   "SOURCE_LIMITED?": "verify scan — maybe irreducible"}
        for s in ORDER + [x for x in stages if x not in ORDER]:
            if stages.get(s):
                w.writerow([s, stages[s], meaning.get(s, "")])

    data = len(rows) - stages.get("EMPTY_TITLE", 0)
    done = sum(stages[s] for s in DONE)
    recoverable = sum(stages.get(s, 0) for s in ("RE-OCR_MINOR", "NEEDS_STITCH", "MISTRAL_RECOVER", "NEEDS_REVIEW"))
    print(f"provenance_full.csv: {len(rows)} Table VII reads ({data} data tables, {stages.get('EMPTY_TITLE',0)} title/blank)")
    print("\n=== path to perfection (data tables) ===")
    for s in ORDER:
        if stages.get(s) and s != "EMPTY_TITLE":
            tag = "(done/usable)" if s in DONE else "-> needs the next_action"
            print(f"  {s:18} {stages[s]:5}  {tag}")
    print(f"\n  DONE/USABLE: {done} ({100*done//max(1,data)}% of {data}) · "
          f"RECOVERABLE w/ known fix: {recoverable} · maybe source-limited: {stages.get('SOURCE_LIMITED?',0)}")


if __name__ == "__main__":
    main()
