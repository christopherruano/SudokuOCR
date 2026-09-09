#!/usr/bin/env python3
"""Full-corpus constraint report — the paper's methods table. Runs constraints_core over EVERY
oneshot JSON in the referenced corpus and writes:
  export/constraint_report.csv          one row per read: kind, layout, per-identity-class
                                        n_checked/n_pass/n_fail, coverage, status
                                        (PASS / FAIL / NOT_CHECKABLE / NOT_ANCHORED)
  export/constraint_report_summary.csv  kind × decade and kind × volume pivots:
                                        pass-rate among checked reads + mean coverage
"0 violations" never means "0 checks": NOT_ANCHORED reads are counted separately from PASS.
  python3 constraint_corpus_report.py
"""
import os, sys, csv, json, re, glob
from collections import defaultdict

import constraints_core as CC

ROOT = os.path.dirname(os.path.abspath(__file__))
V1 = os.path.join(ROOT, "export_final/sendout_final/sendout_v1")
CORPUS = os.path.join(ROOT, "export/ocr_results_referenced/digitized_tables")
IH_RE = re.compile(r"([0-9a-f]{40})")
IDENTS = list(CC.ALL_IDENTITIES)


def main():
    # lineage: imghash -> (dochash, source_pdf) from the page master
    pm = {}
    pmf = os.path.join(V1, "page_number_status.csv")
    if os.path.exists(pmf):
        for r in csv.DictReader(open(pmf)):
            pm[r.get("imghash", "")] = (r.get("dochash", ""), r.get("source_pdf", ""))

    files = sorted(glob.glob(os.path.join(CORPUS, "*", "*_oneshot.json")))
    rows_out = []
    n_err = 0
    for f in files:
        m = IH_RE.search(os.path.basename(f))
        ih = m.group(1) if m else ""
        tdir = os.path.basename(os.path.dirname(f))
        try:
            d = json.load(open(f))
        except Exception:
            n_err += 1
            continue
        meta = d.get("metadata", {}) or {}
        rep = CC.check_table(d)
        # dominant layout = layout of the largest checked section
        secs = [(sr.n_rows, sr.layout) for sr in rep.sections]
        layout = max(secs)[1] if secs else "unknown"
        dochash, vol = pm.get(ih, ("", ""))
        if not vol:
            vol = tdir.split("_t")[0].split("_grow")[0]      # doc prefix of the crop dir
        rec = {
            "imghash": ih, "dochash": dochash, "volume": os.path.basename(vol),
            "year": meta.get("year", ""), "province": meta.get("region", ""),
            "title": (meta.get("title", "") or "")[:60],
            "kind": rep.kind, "layout": layout, "status": rep.status,
            "human_verified": int(rep.human_verified),
            "constraints_irrelevant": int(rep.constraints_irrelevant),
            "n_fail_rows": rep.n_fail_rows,
            "numeric_cells": rep.numeric_cells, "checked_cells": rep.checked_cells,
            "coverage": round(rep.coverage, 4),
        }
        for ident in IDENTS:
            st = rep.stats.get(ident, {})
            nc, np_ = st.get("n_checked", 0), st.get("n_pass", 0)
            rec[f"{ident}_checked"] = nc
            rec[f"{ident}_pass"] = np_
            rec[f"{ident}_fail"] = nc - np_
        rows_out.append(rec)

    cols = (["imghash", "dochash", "volume", "year", "province", "title", "kind", "layout",
             "status", "human_verified", "constraints_irrelevant", "n_fail_rows",
             "numeric_cells", "checked_cells", "coverage"]
            + [f"{i}_{s}" for i in IDENTS for s in ("checked", "pass", "fail")])
    out = os.path.join(ROOT, "export/constraint_report.csv")
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows_out:
            w.writerow(r)

    # ---- pivots: kind × decade, kind × volume ------------------------------------------------
    def decade(y):
        m = re.search(r"(18|19)\d\d", str(y))
        return f"{int(m.group(0)) // 10 * 10}s" if m else "?"

    def pivot(keyfn):
        agg = defaultdict(lambda: {"reads": 0, "PASS": 0, "FAIL": 0,
                                   "NOT_ANCHORED": 0, "NOT_CHECKABLE": 0, "cov": 0.0})
        for r in rows_out:
            a = agg[(r["kind"], keyfn(r))]
            a["reads"] += 1
            a[r["status"]] = a.get(r["status"], 0) + 1
            a["cov"] += r["coverage"]
        return agg

    piv_out = os.path.join(ROOT, "export/constraint_report_summary.csv")
    with open(piv_out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["pivot", "kind", "key", "reads", "pass", "fail", "not_anchored",
                    "not_checkable", "pass_rate_among_checked", "mean_coverage"])
        for name, keyfn in (("kind_x_decade", lambda r: decade(r["year"])),
                            ("kind_x_volume", lambda r: r["volume"])):
            agg = pivot(keyfn)
            for (kind, key) in sorted(agg):
                a = agg[(kind, key)]
                checked = a["PASS"] + a["FAIL"]
                w.writerow([name, kind, key, a["reads"], a["PASS"], a["FAIL"],
                            a["NOT_ANCHORED"], a["NOT_CHECKABLE"],
                            round(a["PASS"] / checked, 4) if checked else "",
                            round(a["cov"] / a["reads"], 4) if a["reads"] else 0])

    # ---- console headline ---------------------------------------------------------------------
    from collections import Counter
    kinds = Counter(r["kind"] for r in rows_out)
    stats = Counter((r["kind"], r["status"]) for r in rows_out)
    lays = Counter((r["layout"], r["status"]) for r in rows_out if r["kind"] == "count")
    print(f"constraint_report.csv: {len(rows_out)} reads ({n_err} unparseable) -> {out}")
    print(f"summary pivots -> {piv_out}\n")
    print(f"{'kind':14} {'reads':>6} {'pass':>6} {'fail':>6} {'anchorless':>10} {'n/a':>6} {'pass-rate':>9}")
    for k, n in kinds.most_common():
        p, fl = stats.get((k, "PASS"), 0), stats.get((k, "FAIL"), 0)
        na, nc = stats.get((k, "NOT_ANCHORED"), 0), stats.get((k, "NOT_CHECKABLE"), 0)
        pr = f"{p / (p + fl):.1%}" if p + fl else "—"
        print(f"{k:14} {n:6} {p:6} {fl:6} {na:10} {nc:6} {pr:>9}")
    print("\ncount tables by layout:")
    for lay in ("age_rows", "transposed", "flat_area", "unknown"):
        p, fl = lays.get((lay, "PASS"), 0), lays.get((lay, "FAIL"), 0)
        na = lays.get((lay, "NOT_ANCHORED"), 0)
        pr = f"{p / (p + fl):.1%}" if p + fl else "—"
        print(f"  {lay:12} pass={p:5} fail={fl:5} not_anchored={na:5} pass-rate={pr}")
    cov = [r["coverage"] for r in rows_out if r["status"] in ("PASS", "FAIL")]
    if cov:
        print(f"\nmean cell coverage among checked reads: {sum(cov) / len(cov):.1%}")


if __name__ == "__main__":
    main()
